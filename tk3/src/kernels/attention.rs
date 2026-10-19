//! Flash attention forward: Q resident in registers, K/V streamed through a
//! shared ring, the online-softmax state `(m, l, o)` carried.

use super::rows::NormCfg;
use super::{Batch, batch_offset, bound, load_f32};
use crate::atoms::Target;
use crate::build::*;
use crate::ir::*;
use crate::layouts::WarpGrid;
use crate::lower::Lowering;
use crate::schedule::Schedule;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaCfg {
    pub bq: usize,
    pub bkv: usize,
    pub stages: usize,
    /// Key splits: each is a block along grid x writing a partial result
    /// that [`combine`] merges; `1` writes the output directly.
    pub splits: usize,
}

impl FaCfg {
    pub const fn new(bq: usize, bkv: usize, stages: usize) -> Self {
        Self { bq, bkv, stages, splits: 1 }
    }

    /// One warp per 16 query rows: row reductions stay within a warp.
    pub fn warps(&self) -> u32 {
        (self.bq / 16) as u32
    }

    pub fn lowering(&self, target: Target) -> Lowering {
        let prefetch = target.prefetch();
        Lowering {
            target,
            schedule: Schedule::Uniform { prefetch, unroll: false },
            grid: WarpGrid { rows: self.warps(), cols: 1 },
            swizzle: true,
        }
    }

    /// Static shared memory of the K and V rings, in bytes of a 16-bit type.
    pub fn smem_bytes(&self, d: usize) -> usize {
        2 * self.stages * self.bkv * d * 2
    }
}

/// Which keys each query sees; the masks apply together. A query with no
/// visible key yields NaN, as a softmax over an empty row does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AttnMask {
    /// Key `j` is visible to query `i` only when `j ≤ i`.
    pub causal: bool,
    /// Keys within `[i − left, i + right]` of query `i`.
    pub window: Option<(usize, usize)>,
    /// A `[batch]` i32 parameter of valid key counts: keys at and past the
    /// count are hidden. A count of 0 sees key 0 unless a row is appended,
    /// so a lane always has a key and a finite output.
    pub key_lens: bool,
    /// A `[batch, tk rounded up to 8]` i32 parameter, nonzero where the key
    /// is visible.
    pub key_mask: bool,
    /// A `[batch, t]` i32 parameter, non-decreasing along `t`: keys before
    /// `seg_start[i]` are hidden from query `i` (packed rows' segment starts).
    pub seg_start: bool,
    /// An additive score bias in the stream type, `scores·scale + bias`
    /// before the masks hide anything; the [`Bias`] rows say whose.
    pub bias: Option<Bias>,
}

/// The rows of the `bias` parameter, `[rows, heads, t, key_mask_stride(keys)]`
/// with `keys` the scored keys: `tk`, plus one with a row appended (its
/// column last).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bias {
    /// One row every batch shares.
    Shared,
    /// A row per batch.
    PerBatch,
}

/// Keys and values read from a cache of `[rows, tk, heads_total, d]`, the
/// `kv_heads` heads from `head_start` on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cache {
    pub rows: usize,
    pub heads_total: usize,
    pub head_start: usize,
    /// A `[batch]` i32 parameter: the cache row each batch reads.
    pub row_map: bool,
    /// `[batch, kv_heads, d]` key and value parameters, one row per batch
    /// scored after the cached keys (the token a decoder step projected).
    pub appended: bool,
}

/// The row stride of the `key_mask` and `bias` parameters: 8-element runs
/// stay aligned.
pub fn key_mask_stride(tk: usize) -> usize {
    tk.next_multiple_of(8)
}

/// `[batch, t, heads, d]` queries against `[batch, tk, kv_heads, d]` keys and
/// values; query head `h` reads KV head `h / (heads / kv_heads)`.
#[derive(Clone, Debug, PartialEq)]
pub struct AttnSpec {
    pub batch: Batch,
    pub t: usize,
    pub tk: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub d: usize,
    pub mask: AttnMask,
    pub cache: Option<Cache>,
    pub scale: f32,
    pub cfg: FaCfg,
}

impl AttnSpec {
    /// Parameters after the outputs, in order.
    pub fn extra_params(&self) -> Vec<&'static str> {
        let AttnMask { key_lens, key_mask, seg_start, .. } = self.mask;
        let cache =
            self.cache.unwrap_or(Cache { rows: 0, heads_total: 0, head_start: 0, row_map: false, appended: false });
        [
            (key_lens, "key_lens"),
            (key_mask, "key_mask"),
            (seg_start, "seg_start"),
            (cache.row_map, "row_map"),
            (cache.appended, "k_app"),
            (cache.appended, "v_app"),
            (self.mask.bias.is_some(), "bias"),
        ]
        .into_iter()
        .filter(|(on, _)| *on)
        .map(|(_, name)| name)
        .collect()
    }
}

/// The scores of a key block starting at the scalar, masked.
type Mask<'a> = &'a dyn Fn(&mut Kernel, Regs<F32>, Sc) -> Regs<F32>;

/// The online-softmax step over one key block: `q_r` against `k_t`, the
/// scores masked by `mask(block_start)`, folded into `(m, l, o)` with `v_t`.
struct Step<'a, T> {
    q_r: Regs<T>,
    scale_log2e: f64,
    shape: Shape,
    mask: Mask<'a>,
}

fn step<T: Elem, PK: TierMark, PV: TierMark>(
    k: &mut Kernel,
    st: &Step<'_, T>,
    k_t: Tile<PK, T>,
    v_t: Tile<PV, T>,
    block_start: Sc,
    [m, l, o]: [Regs<F32>; 3],
) -> [Regs<F32>; 3] {
    let zero = k.zeros::<F32>(st.shape);
    let s = k.mma(zero, st.q_r, false, k_t, true);
    let scale = k.fill::<F32>(st.shape, Const::Float(st.scale_log2e));
    let s = k.binary(s, scale, BinaryOp::Mul);
    let s = (st.mask)(k, s, block_start);
    let block_max = k.reduce(s, Axis::Row, ReduceOp::Max);
    let m_new = k.binary(m, block_max, BinaryOp::Max);
    let corr = k.binary(m, m_new, BinaryOp::Sub);
    let corr = k.unary(corr, UnaryOp::Exp2);
    let p = k.binary(s, m_new, BinaryOp::Sub);
    let p = k.unary(p, UnaryOp::Exp2);
    let block_sum = k.reduce(p, Axis::Row, ReduceOp::Sum);
    let l = k.binary(l, corr, BinaryOp::Mul);
    let l = k.binary(l, block_sum, BinaryOp::Add);
    let o = k.binary(o, corr, BinaryOp::Mul);
    let p16 = k.cast::<F32, T>(p);
    let o = k.mma(o, p16, false, v_t, false);
    [m_new, l, o]
}

/// `o = softmax(q·kᵀ·scale + bias)·v`. Parameters in order: `q`, `k`, `v`, then `o`
/// (one split) or the f32 partials `o_part [splits, batch, t, heads, d]`,
/// `m_part` and `l_part [splits, batch, t, heads]`, then
/// [`AttnSpec::extra_params`]. Key blocks no query of the tile sees are
/// skipped through the first block and the trip count, the rest shared
/// evenly by the splits; only blocks a mask edge crosses pay for the mask,
/// except under a key mask or a bias, which every block reads.
pub fn attention<T: Elem>(spec: &AttnSpec) -> Program {
    let AttnSpec { ref batch, t, tk, heads, kv_heads, d, mask, cache, scale, cfg } = *spec;
    let AttnMask { causal, window, key_lens, key_mask, seg_start, bias } = mask;
    let FaCfg { bq, bkv, stages, splits } = cfg;
    let cap = batch.capacity();
    let (kv_rows, heads_total, head_start) =
        cache.map_or((cap, kv_heads, 0), |c| (c.rows, c.heads_total, c.head_start));
    let mut k = Kernel::new("flash_attention");
    let q = k.param::<T>("q", ParamKind::In, cap * t * heads * d);
    let kk = k.param::<T>("k", ParamKind::In, kv_rows * tk * heads_total * d);
    let v = k.param::<T>("v", ParamKind::In, kv_rows * tk * heads_total * d);
    let o = (splits == 1).then(|| k.param::<T>("o", ParamKind::Out, cap * t * heads * d));
    let parts = (splits > 1).then(|| {
        let o = k.param::<F32>("o_part", ParamKind::Out, splits * cap * t * heads * d);
        let m = k.param::<F32>("m_part", ParamKind::Out, splits * cap * t * heads);
        let l = k.param::<F32>("l_part", ParamKind::Out, splits * cap * t * heads);
        (o, m, l)
    });
    let lens = key_lens.then(|| k.param::<I32>("key_lens", ParamKind::In, cap));
    let mask_stride = key_mask_stride(tk);
    let masks = key_mask.then(|| k.param::<I32>("key_mask", ParamKind::In, cap * mask_stride));
    let segs = seg_start.then(|| k.param::<I32>("seg_start", ParamKind::In, cap * t));
    let row_map = cache.is_some_and(|c| c.row_map).then(|| k.param::<I32>("row_map", ParamKind::In, cap));
    let appended = cache.is_some_and(|c| c.appended).then(|| {
        let k_app = k.param::<T>("k_app", ParamKind::In, cap * kv_heads * d);
        let v_app = k.param::<T>("v_app", ParamKind::In, cap * kv_heads * d);
        (k_app, v_app)
    });
    let keys = tk + usize::from(appended.is_some());
    let bias_stride = key_mask_stride(keys);
    let bias = bias.map(|rows| {
        let n = if rows == Bias::Shared { 1 } else { cap };
        (k.param::<T>("bias", ParamKind::In, n * heads * t * bias_stride), rows)
    });
    let (gz, bb) = batch.axis(&mut k);
    k.grid([Sc::from(t.div_ceil(bq) * splits), Sc::from(heads), gz]);
    k.warps(cfg.warps());
    let k_s = k.smem::<T>("k_s", stages * bkv * d);
    let v_s = k.smem::<T>("v_s", stages * bkv * d);

    let (qb, split, hh) = (k.block(0) / splits, k.block(0) % splits, k.block(1));
    let q_off = qb * bq;
    let b = bb.clone().unwrap_or(Sc::from(0));
    // Keys past the length are masked; keys past `tk` too when blocks overhang it.
    let keys_masked = key_lens || !tk.is_multiple_of(bkv);
    let len = match lens {
        Some(lens) if appended.is_some() => k.load_scalar(lens, b.clone()).min(tk),
        Some(lens) => k.load_scalar(lens, b.clone()).max(1).min(tk),
        None => Sc::from(tk),
    };
    let (q_stride, kv_stride) = (heads * d, heads_total * d);
    let rows = |k: &mut Kernel, p: ParamRef<T>| {
        let v = k.view(
            p,
            batch_offset(&bb, t * q_stride) + hh.clone() * d,
            [q_stride, 1],
            Shape::new(bq, d),
            [bound(t, bq), None],
        );
        k.at(v, q_off.clone(), 0)
    };
    let q_view = rows(&mut k, q);
    let kv_row = match row_map {
        Some(map) => k.load_scalar(map, b.clone()),
        None => b.clone(),
    };
    let kv_head = hh.clone() / (heads / kv_heads);
    let kv_base = kv_row * (tk * kv_stride) + (kv_head.clone() + head_start) * d;
    let kv_bound = keys_masked.then(|| len.clone());
    let k_view = k.view(kk, kv_base.clone(), [kv_stride, 1], Shape::new(bkv, d), [kv_bound.clone(), None]);
    let v_view = k.view(v, kv_base, [kv_stride, 1], Shape::new(bkv, d), [kv_bound, None]);
    let mask_view = masks.map(|m| {
        let view = k.view(m, b.clone() * mask_stride, [mask_stride, 1], Shape::new(1, bkv), [None, bound(tk, bkv)]);
        k.at(view, 0, 0)
    });
    // Past `keys` (the appended column's tile always overhangs it) reads nothing.
    let bias_view = bias.map(|(p, rows)| {
        let batch = if rows == Bias::Shared { Sc::from(0) } else { batch_offset(&bb, heads * t * bias_stride) };
        let cols = if appended.is_some() { Some(Sc::from(keys)) } else { bound(keys, bkv) };
        let bounds = [bound(t, bq), cols];
        k.view(p, batch + hh.clone() * (t * bias_stride), [bias_stride, 1], Shape::new(bq, bkv), bounds)
    });
    // The tile's segment starts, loaded once; the mask is monotonic, so the
    // first and last rows bound the tile's.
    let seg = segs.map(|s| {
        let view = k.view(s, b.clone() * t, [1, 1], Shape::new(bq, 1), [bound(t, bq), None]);
        let tile = k.at(view, q_off.clone(), 0);
        let last = (q_off.clone() + bq - 1).min(t - 1);
        (k.load(tile), k.load_scalar(s, b.clone() * t + q_off.clone()), k.load_scalar(s, b.clone() * t + last))
    });

    // Key blocks `first..end` hold every key some query of the tile sees;
    // this split takes an even share of them.
    let mut first = Sc::from(0);
    if let Some((left, _)) = window {
        first = first.max((q_off.clone().max(left) - left) / bkv);
    }
    if let Some((_, seg_first, _)) = &seg {
        first = first.max(seg_first.clone() / bkv);
    }
    let mut end = Sc::from(tk.div_ceil(bkv));
    if causal {
        end = end.min((q_off.clone() + bq + bkv - 1) / bkv);
    }
    if let Some((_, right)) = window {
        end = end.min((q_off.clone() + bq + right + bkv - 1) / bkv);
    }
    if key_lens {
        end = end.min((len.clone() + bkv - 1) / bkv);
    }
    let n = end - first.clone();
    let s_first = first.clone() + n.clone() * split.clone() / splits;
    let s_end = first + n * (split.clone() + 1) / splits;
    let blocks = s_end - s_first.clone();

    let shape = Shape::new(bq, bkv);
    let q_r = k.load(q_view);
    let m0 = k.fill::<F32>(Shape::new(bq, 1), Const::Float(-1e30));
    let l0 = k.zeros::<F32>(Shape::new(bq, 1));
    let o0 = k.zeros::<F32>(Shape::new(bq, d));
    let scale_log2e = (scale * std::f32::consts::LOG2_E) as f64;
    // The key coordinates of a block's scores and the query rows'.
    let coords = |k: &mut Kernel, block_start: Sc| {
        let col = k.coord(shape, Axis::Col);
        let start = k.splat::<I32>(shape, block_start);
        let col = k.binary(col, start, BinaryOp::Add);
        let row = k.coord(shape, Axis::Row);
        let q_off = k.splat::<I32>(shape, q_off.clone());
        (col, k.binary(row, q_off, BinaryOp::Add))
    };
    // The bias tile of the key block at `col`, in the scores' log2 units.
    let biased = |k: &mut Kernel, s: Regs<F32>, col: Sc| -> Regs<F32> {
        let Some(view) = bias_view else { return s };
        let tile = k.at(view, q_off.clone(), col);
        let b = load_f32(k, tile);
        let log2e = k.fill::<F32>(shape, Const::Float(std::f64::consts::LOG2_E));
        let b = k.binary(b, log2e, BinaryOp::Mul);
        k.binary(s, b, BinaryOp::Add)
    };
    let hide = |k: &mut Kernel, keep: Regs<Bool>, plain: Regs<F32>| {
        let hidden = k.fill::<F32>(shape, Const::Float(f64::NEG_INFINITY));
        k.where_(keep, plain, hidden)
    };
    let masked = |k: &mut Kernel, plain: Regs<F32>, block_start: Sc| {
        let (col, row) = coords(k, block_start.clone());
        let mut keep: Option<Regs<Bool>> = None;
        let mut and = |k: &mut Kernel, cond| {
            keep = Some(match keep {
                Some(keep) => k.binary(keep, cond, BinaryOp::And),
                None => cond,
            });
        };
        if causal {
            let cond = k.compare(col, row, BinaryOp::Le);
            and(k, cond);
        }
        if keys_masked {
            let len = k.splat::<I32>(shape, len.clone());
            let cond = k.compare(col, len, BinaryOp::Lt);
            and(k, cond);
        }
        if let Some((left, right)) = window {
            let left = k.splat::<I32>(shape, left);
            let reach = k.binary(col, left, BinaryOp::Add);
            let cond = k.compare(row, reach, BinaryOp::Le);
            and(k, cond);
            let right = k.splat::<I32>(shape, right);
            let reach = k.binary(row, right, BinaryOp::Add);
            let cond = k.compare(col, reach, BinaryOp::Le);
            and(k, cond);
        }
        if let Some(view) = mask_view {
            let tile = k.at(view, 0, block_start.clone());
            let visible = k.load(tile);
            // Against a full-shape zero: the predicate takes the tile's shape.
            let zero = k.zeros::<I32>(shape);
            let cond = k.compare(visible, zero, BinaryOp::Ne);
            and(k, cond);
        }
        if let Some((seg, _, _)) = &seg {
            let cond = k.compare(*seg, col, BinaryOp::Le);
            and(k, cond);
        }
        hide(k, keep.expect("a mask"), plain)
    };
    let edges = causal || keys_masked || window.is_some() || seg.is_some();
    let mask = |k: &mut Kernel, s: Regs<F32>, block_start: Sc| -> Regs<F32> {
        let s = biased(k, s, block_start.clone());
        if key_mask {
            return masked(k, s, block_start);
        }
        if !edges {
            return s;
        }
        let block_end = block_start.clone() + bkv;
        // Whether a mask edge crosses this block.
        let mut partial = Sc::from(0);
        if causal {
            partial = partial.or(q_off.clone().lt(block_end.clone()));
        }
        if keys_masked {
            partial = partial.or(len.clone().lt(block_end.clone()));
        }
        if let Some((left, right)) = window {
            let left_edge = (block_start.clone() + left).lt(q_off.clone() + bq - 1);
            let right_edge = (q_off.clone() + right + 1).lt(block_end.clone());
            partial = partial.or(left_edge).or(right_edge);
        }
        if let Some((_, _, seg_last)) = &seg {
            partial = partial.or(block_start.clone().lt(seg_last.clone()));
        }
        let [out] = k.select_if(partial, |k| [masked(k, s, block_start.clone())], |_| [s]);
        out
    };
    let st = Step { q_r, scale_log2e, shape, mask: &mask };
    let [m, l, acc] = k.pipeline(
        blocks,
        stages,
        [m0, l0, o0],
        |k, step, slot| {
            let kv_off = (s_first.clone() + step) * bkv;
            let k_g = k.at(k_view, kv_off.clone(), 0);
            let v_g = k.at(v_view, kv_off, 0);
            let k_t = k.smem_slot::<T>(k_s, slot.clone(), Shape::new(bkv, d));
            let v_t = k.smem_slot::<T>(v_s, slot, Shape::new(bkv, d));
            k.stage(k_t, k_g, CopyMode::Async);
            k.stage(v_t, v_g, CopyMode::Async);
        },
        |k, step_, slot, carried| {
            let k_t = k.smem_slot::<T>(k_s, slot.clone(), Shape::new(bkv, d));
            let v_t = k.smem_slot::<T>(v_s, slot, Shape::new(bkv, d));
            step(k, &st, k_t, v_t, (s_first.clone() + step_) * bkv, carried)
        },
    );
    // The appended row, a block holding one key that every split scores
    // and all but the last hide: a hidden block leaves `(m, l, o)` as they
    // were, and a branch would cost more than the step.
    let [m, l, acc] = match appended {
        Some((k_app, v_app)) => {
            let app = |k: &mut Kernel, p: ParamRef<T>| {
                k.view(
                    p,
                    batch_offset(&bb, kv_heads * d) + kv_head.clone() * d,
                    [kv_heads * d, 1],
                    Shape::new(bkv, d),
                    [Some(Sc::from(1)), None],
                )
            };
            let (k_g, v_g) = (app(&mut k, k_app), app(&mut k, v_app));
            let last = split.clone().eq(splits - 1);
            let one = |k: &mut Kernel, s: Regs<F32>, _: Sc| {
                let s = biased(k, s, Sc::from(tk));
                let col = k.coord(shape, Axis::Col);
                let zero = k.zeros::<I32>(shape);
                let first_col = k.compare(col, zero, BinaryOp::Eq);
                let owner = k.splat::<I32>(shape, last.clone());
                let owned = k.compare(owner, zero, BinaryOp::Ne);
                let keep = k.binary(first_col, owned, BinaryOp::And);
                hide(k, keep, s)
            };
            let st = Step { q_r, scale_log2e, shape, mask: &one };
            step(&mut k, &st, k_g, v_g, Sc::from(0), [m, l, acc])
        }
        None => [m, l, acc],
    };
    match (o, parts) {
        (Some(o), _) => {
            let out = k.binary(acc, l, BinaryOp::Div);
            let out = k.cast::<F32, T>(out);
            let o_view = rows(&mut k, o);
            k.store(o_view, out);
        }
        (None, Some((o_part, m_part, l_part))) => {
            let per_split = cap * t * heads;
            let o_view = k.view(
                o_part,
                split.clone() * (per_split * d) + batch_offset(&bb, t * q_stride) + hh.clone() * d,
                [q_stride, 1],
                Shape::new(bq, d),
                [bound(t, bq), None],
            );
            let o_view = k.at(o_view, q_off.clone(), 0);
            k.store(o_view, acc);
            for (p, val) in [(m_part, m), (l_part, l)] {
                let view = k.view(
                    p,
                    split.clone() * per_split + batch_offset(&bb, t * heads) + hh.clone(),
                    [heads, 1],
                    Shape::new(bq, 1),
                    [bound(t, bq), None],
                );
                let view = k.at(view, q_off.clone(), 0);
                k.store(view, val);
            }
        }
        (None, None) => unreachable!("an output"),
    }
    k.finish()
}

/// `[splits, batch, t, heads]` rows of partial results merged into
/// `o [batch, t, heads, d]`.
#[derive(Clone, Debug, PartialEq)]
pub struct CombineSpec {
    pub batch: Batch,
    pub t: usize,
    pub heads: usize,
    pub d: usize,
    pub splits: usize,
    pub cfg: NormCfg,
}

/// Parameters in order: `o_part`, `m_part`, `l_part` (f32, as the
/// attention kernel writes them), `o`.
pub fn combine<T: Elem>(spec: &CombineSpec) -> Program {
    let CombineSpec { ref batch, t, heads, d, splits, cfg: NormCfg { br } } = *spec;
    let (cap, rows) = (batch.capacity(), t * heads);
    let mut k = Kernel::new("combine_splits");
    let o_part = k.param::<F32>("o_part", ParamKind::In, splits * cap * rows * d);
    let m_part = k.param::<F32>("m_part", ParamKind::In, splits * cap * rows);
    let l_part = k.param::<F32>("l_part", ParamKind::In, splits * cap * rows);
    let o = k.param::<T>("o", ParamKind::Out, cap * rows * d);
    let (gz, bb) = batch.axis(&mut k);
    k.grid([Sc::from(rows.div_ceil(br)), Sc::from(1), gz]);
    k.warps(br as u32);
    let row0 = k.block(0) * br;
    let part = |k: &mut Kernel, p: ParamRef<F32>, split: usize, width: usize| {
        let base = Sc::from(split * cap * rows * width) + batch_offset(&bb, rows * width);
        let view = k.view(p, base, [width, 1], Shape::new(br, width), [bound(rows, br), None]);
        let tile = k.at(view, row0.clone(), 0);
        k.load(tile)
    };
    let ms: Vec<Regs<F32>> = (0..splits).map(|s| part(&mut k, m_part, s, 1)).collect();
    let mut m = ms[0];
    for &ms in &ms[1..] {
        m = k.binary(m, ms, BinaryOp::Max);
    }
    let (mut l, mut acc) = (None, None);
    for (s, &ms) in ms.iter().enumerate() {
        let w = k.binary(ms, m, BinaryOp::Sub);
        let w = k.unary(w, UnaryOp::Exp2);
        let ls = part(&mut k, l_part, s, 1);
        let ls = k.binary(ls, w, BinaryOp::Mul);
        let os = part(&mut k, o_part, s, d);
        let os = k.binary(os, w, BinaryOp::Mul);
        l = Some(match l {
            Some(l) => k.binary(l, ls, BinaryOp::Add),
            None => ls,
        });
        acc = Some(match acc {
            Some(acc) => k.binary(acc, os, BinaryOp::Add),
            None => os,
        });
    }
    let (l, acc) = (l.expect("a split"), acc.expect("a split"));
    let out = k.binary(acc, l, BinaryOp::Div);
    let out = k.cast::<F32, T>(out);
    let view = k.view(o, batch_offset(&bb, rows * d), [d, 1], Shape::new(br, d), [bound(rows, br), None]);
    let view = k.at(view, row0, 0);
    k.store(view, out);
    k.finish()
}
