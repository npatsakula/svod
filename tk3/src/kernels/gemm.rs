//! `c = scale·act(a·bᵀ + bias) + residual` (or `scale·act(gate)·up` for a
//! gated weight) as a K-streaming block pipeline with the epilogue on the f32
//! accumulator.

use std::fmt;

use super::{Act, Batch, batch_offset, bound, konst, load_f32};
use crate::atoms::Target;
use crate::build::*;
use crate::ir::*;
use crate::layouts::WarpGrid;
use crate::lower::Lowering;
use crate::schedule::Schedule;

/// What runs on the accumulator before the single rounding at the store.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Epilogue {
    /// A `[n]` (`[2n]` gated) bias parameter.
    pub bias: bool,
    pub act: Act,
    /// The weight is `[2n, k]`, gate rows over up rows; the output is
    /// `act(gate)·up` (SwiGLU with `Silu`, GeGLU with `Gelu`).
    pub gated: bool,
    /// A `[m, n]` residual parameter added last.
    pub residual: bool,
    /// Multiplies the activated (gated) value, before the residual add.
    pub scale: Option<Scale>,
    /// The output (and residual) is f32: the accumulator is stored unrounded.
    pub out_f32: bool,
}

impl Epilogue {
    /// Nothing but the rounding, as a constant.
    pub const DEFAULT: Self =
        Self { bias: false, act: Act::None, gated: false, residual: false, scale: None, out_f32: false };
}

/// The tune store keys by this text: an epilogue without a scale prints as
/// it did before the field existed, so its stored choices still apply.
impl fmt::Debug for Epilogue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("Epilogue");
        s.field("bias", &self.bias).field("act", &self.act).field("gated", &self.gated);
        s.field("residual", &self.residual);
        if let Some(scale) = self.scale {
            s.field("scale", &scale);
        }
        if self.out_f32 {
            s.field("out_f32", &true);
        }
        s.finish()
    }
}

/// An f32 output scale, compared and hashed by its bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scale(u32);

impl Scale {
    pub const fn new(value: f32) -> Self {
        Self(value.to_bits())
    }

    pub fn get(self) -> f32 {
        f32::from_bits(self.0)
    }
}

impl fmt::Debug for Scale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(f)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemmCfg {
    /// `[bm, bn, bk]`.
    pub tile: [usize; 3],
    pub stages: usize,
    /// `[rows, cols]` of the warp grid over the output tile.
    pub warps: [u32; 2],
    /// Tile rows walked together so resident blocks share B in L2 (0 = row-major).
    pub group_m: usize,
    pub unroll: bool,
}

impl GemmCfg {
    pub fn lowering(&self, target: Target) -> Lowering {
        let prefetch = target.prefetch();
        Lowering {
            target,
            schedule: Schedule::Uniform { prefetch, unroll: self.unroll },
            grid: WarpGrid { rows: self.warps[0], cols: self.warps[1] },
            swizzle: true,
        }
    }

    /// Static shared memory of the operand ring, in bytes of a 16-bit type.
    pub fn smem_bytes(&self, gated: bool) -> usize {
        let [bm, bn, bk] = self.tile;
        self.stages * (bm + bn * if gated { 2 } else { 1 }) * bk * 2
    }
}

/// `m` rows per batch; the batch, when there is one, walks grid z.
#[derive(Clone, Debug, PartialEq)]
pub struct GemmSpec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: Batch,
    pub epilogue: Epilogue,
    pub cfg: GemmCfg,
}

/// Parameters in order: `a [batch·m, k]`, `b [n, k]` (`[2n, k]` gated),
/// `bias` if any, `residual [batch·m, n]` if any, `c [batch·m, n]`. Rows past
/// `m` and weight rows past `n` are clamped reads whose results are never
/// stored; `k` must be a multiple of `bk`.
pub fn gemm<T: Elem>(spec: &GemmSpec) -> Program {
    let GemmSpec { m, k: kk, cfg, .. } = *spec;
    let [bm, _, bk] = cfg.tile;
    mainloop_gemm::<T, _>("gemm", spec, m * kk, 1, |k, a, bb, row0| {
        let a_view = k.view(a, batch_offset(bb, m * kk), [kk, 1], Shape::new(bm, bk), [bound(m, bm), None]);
        let a_view = k.at(a_view, row0, 0);
        move |k: &mut Kernel, _step: Sc, koff: Sc| k.at(a_view, 0, koff)
    })
}

/// The GEMM around an A operand of `a_elems` elements per batch whose
/// K-step tiles `a` describes: given the kernel, the parameter, the batch
/// index and the block's first row, it returns the source of the `[bm, bk]`
/// tile at `(step, step·bk)`. The output dtype follows the epilogue.
///
/// With `split > 1` (a single static batch) grid z splits the reduction:
/// split `z` walks steps `z·trips/split..`, and the kernel writes its raw f32
/// accumulator to `c [split, m, n]` for [`split_merge`] to finish.
pub(crate) fn mainloop_gemm<T: Elem, S: Fn(&mut Kernel, Sc, Sc) -> Gmem<T>>(
    name: &str,
    spec: &GemmSpec,
    a_elems: usize,
    split: usize,
    a: impl FnOnce(&mut Kernel, ParamRef<T>, &Option<Sc>, Sc) -> S,
) -> Program {
    if split > 1 {
        build::<T, F32, S>(name, spec, a_elems, split, a)
    } else if spec.epilogue.out_f32 {
        build::<T, F32, S>(name, spec, a_elems, 1, a)
    } else {
        build::<T, T, S>(name, spec, a_elems, 1, a)
    }
}

fn build<T: Elem, U: Elem, S: Fn(&mut Kernel, Sc, Sc) -> Gmem<T>>(
    name: &str,
    spec: &GemmSpec,
    a_elems: usize,
    split: usize,
    a_source: impl FnOnce(&mut Kernel, ParamRef<T>, &Option<Sc>, Sc) -> S,
) -> Program {
    let GemmSpec { m, n, k: kk, ref batch, epilogue, cfg } = *spec;
    let [bm, bn, bk] = cfg.tile;
    assert!(kk.is_multiple_of(bk), "k is a multiple of bk");
    let trips = kk / bk;
    let partial = split > 1;
    assert!(!partial || (*batch == Batch::Static(1) && !epilogue.gated && trips.is_multiple_of(split)));
    // A partial kernel stores the bare accumulator.
    let epi = if partial { Epilogue::DEFAULT } else { epilogue };
    let halves = if epi.gated { 2 } else { 1 };
    let cap = batch.capacity();
    let mut k = Kernel::new(name);
    let a = k.param::<T>("a", ParamKind::In, cap * a_elems);
    let b = k.param::<T>("b", ParamKind::In, halves * n * kk);
    let bias = epi.bias.then(|| k.param::<T>("bias", ParamKind::In, halves * n));
    let residual = epi.residual.then(|| k.param::<U>("residual", ParamKind::In, cap * m * n));
    let c = k.param::<U>("c", ParamKind::Out, cap.max(split) * m * n);
    let (gm, gn) = (m.div_ceil(bm), n.div_ceil(bn));
    let (gz, bb) = if partial { (Sc::from(split), Some(k.block(2))) } else { batch.axis(&mut k) };
    k.grid([Sc::from(gm), Sc::from(gn), gz]);
    k.warps(cfg.warps[0] * cfg.warps[1]);

    let (bx, by) = tile_order(&mut k, gm, gn, cfg.group_m);
    let (row0, col0) = (bx * bm, by * bn);
    let n_bound = bound(n, bn);
    let a_tile = a_source(&mut k, a, if partial { &None } else { &bb }, row0.clone());
    let b_view = |k: &mut Kernel, half: usize| {
        let v = k.view(b, half * n * kk, [kk, 1], Shape::new(bn, bk), [n_bound.clone(), None]);
        k.at(v, col0.clone(), 0)
    };
    let stages = cfg.stages;
    let first_step = partial.then(|| bb.clone().expect("the split index") * (trips / split));
    let sa = Shape::new(bm, bk);
    let accs = if epi.gated {
        let halves = [b_view(&mut k, 0), b_view(&mut k, 1)];
        mainloop(&mut k, sa, &a_tile, halves, trips / split, stages, bk, first_step).to_vec()
    } else {
        let whole = [b_view(&mut k, 0)];
        mainloop(&mut k, sa, &a_tile, whole, trips / split, stages, bk, first_step).to_vec()
    };
    let tile = Tile2 { m, n, bm, bn, row0, col0 };
    finish::<T, U>(&mut k, accs, epi, bias, residual, c, &bb, &tile);
    k.finish()
}

/// Where a block's `[bm, bn]` output tile sits in an `[m, n]` output.
struct Tile2 {
    m: usize,
    n: usize,
    bm: usize,
    bn: usize,
    row0: Sc,
    col0: Sc,
}

/// The epilogue on the f32 accumulators (two when gated) and the store.
#[allow(clippy::too_many_arguments)]
fn finish<T: Elem, U: Elem>(
    k: &mut Kernel,
    accs: Vec<Regs<F32>>,
    epi: Epilogue,
    bias: Option<ParamRef<T>>,
    residual: Option<ParamRef<U>>,
    c: ParamRef<U>,
    bb: &Option<Sc>,
    t: &Tile2,
) {
    let Tile2 { m, n, bm, bn, ref row0, ref col0 } = *t;
    let (m_bound, n_bound) = (bound(m, bm), bound(n, bn));
    let mut accs = accs.into_iter().enumerate().map(|(half, acc)| match bias {
        Some(bias) => {
            let v = k.view(bias, half * n, [0, 1], Shape::new(1, bn), [None, n_bound.clone()]);
            let v = k.at(v, 0, col0.clone());
            let v = load_f32(k, v);
            k.binary(acc, v, BinaryOp::Add)
        }
        None => acc,
    });
    let first = accs.next().expect("an accumulator");
    let up = accs.next();
    let mut out = epi.act.apply(k, first);
    if let Some(up) = up {
        out = k.binary(out, up, BinaryOp::Mul);
    }
    if let Some(scale) = epi.scale {
        let s = konst(k, out, f64::from(scale.get()));
        out = k.binary(out, s, BinaryOp::Mul);
    }
    let tile = |k: &mut Kernel, p: ParamRef<U>| {
        let v = k.view(p, batch_offset(bb, m * n), [n, 1], Shape::new(bm, bn), [m_bound.clone(), n_bound.clone()]);
        k.at(v, row0.clone(), col0.clone())
    };
    if let Some(residual) = residual {
        let r = tile(k, residual);
        let r = load_f32(k, r);
        out = k.binary(out, r, BinaryOp::Add);
    }
    let out = k.cast::<F32, U>(out);
    let c_view = tile(k, c);
    k.store(c_view, out);
}

/// The second program of a split reduction: `c = epilogue(Σ_z partial[z])`
/// over `partial [split, m, n]` f32, `bm × bn` tiles per block.
#[derive(Clone, Debug, PartialEq)]
pub struct MergeSpec {
    pub m: usize,
    pub n: usize,
    pub split: usize,
    pub epilogue: Epilogue,
    /// `[bm, bn]`, powers of two.
    pub tile: [usize; 2],
}

impl MergeSpec {
    pub fn lowering(&self, target: Target) -> Lowering {
        GemmCfg { tile: [self.tile[0], self.tile[1], 16], stages: 1, warps: [4, 1], group_m: 0, unroll: false }
            .lowering(target)
    }
}

/// Parameters in order: `partial [split, m, n]` f32, `bias [n]` if any,
/// `residual [m, n]` if any, `c [m, n]` (f32 when `out_f32`).
pub fn split_merge<T: Elem>(spec: &MergeSpec) -> Program {
    if spec.epilogue.out_f32 { merge::<T, F32>(spec) } else { merge::<T, T>(spec) }
}

fn merge<T: Elem, U: Elem>(spec: &MergeSpec) -> Program {
    let MergeSpec { m, n, split, epilogue: epi, tile: [bm, bn] } = *spec;
    assert!(!epi.gated, "a gated weight is not split");
    let mut k = Kernel::new("split_merge");
    let partial = k.param::<F32>("partial", ParamKind::In, split * m * n);
    let bias = epi.bias.then(|| k.param::<T>("bias", ParamKind::In, n));
    let residual = epi.residual.then(|| k.param::<U>("residual", ParamKind::In, m * n));
    let c = k.param::<U>("c", ParamKind::Out, m * n);
    k.grid([Sc::from(m.div_ceil(bm)), Sc::from(n.div_ceil(bn)), Sc::from(1)]);
    k.warps(4);
    let (row0, col0) = (k.block(0) * bm, k.block(1) * bn);
    let zero = k.zeros::<F32>(Shape::new(bm, bn));
    let [sum] = k.loop_(split, [zero], |k, z, [acc]| {
        let part = k.view(partial, z * (m * n), [n, 1], Shape::new(bm, bn), [bound(m, bm), bound(n, bn)]);
        let part = k.at(part, row0.clone(), col0.clone());
        let part = k.load(part);
        [k.binary(acc, part, BinaryOp::Add)]
    });
    let tile = Tile2 { m, n, bm, bn, row0, col0 };
    finish::<T, U>(&mut k, vec![sum], epi, bias, residual, c, &None, &tile);
    k.finish()
}

/// The `(row, col)` tile of this block, walking groups of `group_m` tile rows.
fn tile_order(k: &mut Kernel, gm: usize, gn: usize, group_m: usize) -> (Sc, Sc) {
    if group_m == 0 {
        return (k.block(0), k.block(1));
    }
    let id = k.block(0) + k.block(1) * gm;
    let first_m = id.clone() / (group_m * gn) * group_m;
    let rows = (Sc::from(gm) - first_m.clone()).min(group_m);
    let within = id % (group_m * gn);
    (first_m + within.clone() % rows.clone(), within / rows)
}

/// The K-streaming pipeline: the `sa`-shaped A tile `a(step, step·bk)` and
/// `N` B tiles per step into a `stages`-deep shared ring, one f32
/// accumulator per B.
#[allow(clippy::too_many_arguments)]
fn mainloop<T: Elem, const N: usize>(
    k: &mut Kernel,
    sa: Shape,
    a: &impl Fn(&mut Kernel, Sc, Sc) -> Gmem<T>,
    bs: [Gmem<T>; N],
    trips: usize,
    stages: usize,
    bk: usize,
    first_step: Option<Sc>,
) -> [Regs<F32>; N] {
    let sb = k.shape(bs[0]);
    let a_s = k.smem::<T>("a_s", stages * sa.elems());
    let b_s = [(); N].map(|()| k.smem::<T>("b_s", stages * sb.elems()));
    let init = [(); N].map(|()| k.zeros::<F32>(Shape::new(sa.rows, sb.rows)));
    k.pipeline(
        trips,
        stages,
        init,
        |k, step, slot| {
            let step = first_step.map_or(step.clone(), |s| s + step);
            let koff = step.clone() * bk;
            let a_g = a(k, step, koff.clone());
            let a_t = k.smem_slot::<T>(a_s, slot.clone(), sa);
            k.stage(a_t, a_g, CopyMode::Async);
            for (src, alloc) in bs.into_iter().zip(b_s) {
                let g = k.at(src, 0, koff.clone());
                let t = k.smem_slot::<T>(alloc, slot.clone(), sb);
                k.stage(t, g, CopyMode::Async);
            }
        },
        |k, _step, slot, accs| {
            let a_t = k.smem_slot::<T>(a_s, slot.clone(), sa);
            let mut i = 0;
            accs.map(|acc| {
                let b_t = k.smem_slot::<T>(b_s[i], slot.clone(), sb);
                i += 1;
                k.mma(acc, a_t, false, b_t, true)
            })
        },
    )
}
