//! `c = act(a·bᵀ + bias) + residual` (or `act(gate)·up` for a gated weight)
//! as a K-streaming block pipeline with the epilogue on the f32 accumulator.

use super::{Act, Batch, batch_offset, bound, load_f32};
use crate::atoms::Target;
use crate::build::*;
use crate::ir::*;
use crate::layouts::WarpGrid;
use crate::lower::Lowering;
use crate::schedule::{Prefetch, Schedule};

/// What runs on the accumulator before the single rounding at the store.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Epilogue {
    /// A `[n]` (`[2n]` gated) bias parameter.
    pub bias: bool,
    pub act: Act,
    /// The weight is `[2n, k]`, gate rows over up rows; the output is
    /// `act(gate)·up` (SwiGLU with `Silu`, GeGLU with `Gelu`).
    pub gated: bool,
    /// A `[m, n]` residual parameter added last.
    pub residual: bool,
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
        Lowering {
            target,
            schedule: Schedule::Uniform { prefetch: Prefetch::CpAsync, unroll: self.unroll },
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
    let GemmSpec { m, n, k: kk, ref batch, epilogue: epi, cfg } = *spec;
    let [bm, bn, bk] = cfg.tile;
    assert!(kk.is_multiple_of(bk), "k is a multiple of bk");
    let halves = if epi.gated { 2 } else { 1 };
    let cap = batch.capacity();
    let mut k = Kernel::new("gemm");
    let a = k.param::<T>("a", ParamKind::In, cap * m * kk);
    let b = k.param::<T>("b", ParamKind::In, halves * n * kk);
    let bias = epi.bias.then(|| k.param::<T>("bias", ParamKind::In, halves * n));
    let residual = epi.residual.then(|| k.param::<T>("residual", ParamKind::In, cap * m * n));
    let c = k.param::<T>("c", ParamKind::Out, cap * m * n);
    let (gm, gn) = (m.div_ceil(bm), n.div_ceil(bn));
    let (gz, bb) = batch.axis(&mut k);
    k.grid([Sc::from(gm), Sc::from(gn), gz]);
    k.warps(cfg.warps[0] * cfg.warps[1]);

    let (bx, by) = tile_order(&mut k, gm, gn, cfg.group_m);
    let (row0, col0) = (bx * bm, by * bn);
    let (m_bound, n_bound) = (bound(m, bm), bound(n, bn));
    let a_view = k.view(a, batch_offset(&bb, m * kk), [kk, 1], Shape::new(bm, bk), [m_bound.clone(), None]);
    let a_view = k.at(a_view, row0.clone(), 0);
    let b_view = |k: &mut Kernel, half: usize| {
        let v = k.view(b, half * n * kk, [kk, 1], Shape::new(bn, bk), [n_bound.clone(), None]);
        k.at(v, col0.clone(), 0)
    };
    let (trips, stages) = (kk / bk, cfg.stages);
    let accs = if epi.gated {
        let halves = [b_view(&mut k, 0), b_view(&mut k, 1)];
        mainloop(&mut k, a_view, halves, trips, stages, bk).to_vec()
    } else {
        let whole = [b_view(&mut k, 0)];
        mainloop(&mut k, a_view, whole, trips, stages, bk).to_vec()
    };
    let mut accs = accs.into_iter().enumerate().map(|(half, acc)| match bias {
        Some(bias) => {
            let v = k.view(bias, half * n, [0, 1], Shape::new(1, bn), [None, n_bound.clone()]);
            let v = k.at(v, 0, col0.clone());
            let v = load_f32(&mut k, v);
            k.binary(acc, v, BinaryOp::Add)
        }
        None => acc,
    });
    let first = accs.next().expect("an accumulator");
    let up = accs.next();
    let mut out = epi.act.apply(&mut k, first);
    if let Some(up) = up {
        out = k.binary(out, up, BinaryOp::Mul);
    }
    let tile = |k: &mut Kernel, p: ParamRef<T>| {
        let v = k.view(p, batch_offset(&bb, m * n), [n, 1], Shape::new(bm, bn), [m_bound.clone(), n_bound.clone()]);
        k.at(v, row0.clone(), col0.clone())
    };
    if let Some(residual) = residual {
        let r = tile(&mut k, residual);
        let r = load_f32(&mut k, r);
        out = k.binary(out, r, BinaryOp::Add);
    }
    let out = k.cast::<F32, T>(out);
    let c_view = tile(&mut k, c);
    k.store(c_view, out);
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

/// The K-streaming pipeline: one A tile and `N` B tiles per step into a
/// `stages`-deep shared ring, one f32 accumulator per B.
fn mainloop<T: Elem, const N: usize>(
    k: &mut Kernel,
    a: Gmem<T>,
    bs: [Gmem<T>; N],
    trips: usize,
    stages: usize,
    bk: usize,
) -> [Regs<F32>; N] {
    let (sa, sb) = (k.shape(a), k.shape(bs[0]));
    let a_s = k.smem::<T>("a_s", stages * sa.elems());
    let b_s = [(); N].map(|()| k.smem::<T>("b_s", stages * sb.elems()));
    let init = [(); N].map(|()| k.zeros::<F32>(Shape::new(sa.rows, sb.rows)));
    k.pipeline(
        trips,
        stages,
        init,
        |k, step, slot| {
            let koff = step * bk;
            for (src, alloc, shape) in
                std::iter::once((a, a_s, sa)).chain(bs.into_iter().zip(b_s).map(|(b, s)| (b, s, sb)))
            {
                let g = k.at(src, 0, koff.clone());
                let t = k.smem_slot::<T>(alloc, slot.clone(), shape);
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
