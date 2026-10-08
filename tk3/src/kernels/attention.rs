//! Flash attention forward: Q resident in registers, K/V streamed through a
//! shared ring, the online-softmax state `(m, l, o)` carried.

use super::{Batch, batch_offset, bound};
use crate::atoms::Target;
use crate::build::*;
use crate::ir::*;
use crate::layouts::WarpGrid;
use crate::lower::Lowering;
use crate::schedule::{Prefetch, Schedule};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaCfg {
    pub bq: usize,
    pub bkv: usize,
    pub stages: usize,
}

impl FaCfg {
    /// One warp per 16 query rows: row reductions stay within a warp.
    pub fn warps(&self) -> u32 {
        (self.bq / 16) as u32
    }

    pub fn lowering(&self, target: Target) -> Lowering {
        Lowering {
            target,
            schedule: Schedule::Uniform { prefetch: Prefetch::CpAsync, unroll: false },
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
    /// count are hidden.
    pub key_lens: bool,
    /// A `[batch, tk rounded up to 8]` i32 parameter, nonzero where the key
    /// is visible.
    pub key_mask: bool,
    /// A `[batch, t]` i32 parameter, non-decreasing along `t`: keys before
    /// `seg_start[i]` are hidden from query `i` (packed rows' segment starts).
    pub seg_start: bool,
}

impl AttnMask {
    /// Parameters the masks add after `o`, in order.
    pub fn params(&self) -> impl Iterator<Item = &'static str> {
        [(self.key_lens, "key_lens"), (self.key_mask, "key_mask"), (self.seg_start, "seg_start")]
            .into_iter()
            .filter_map(|(on, name)| on.then_some(name))
    }
}

/// The row stride of the `key_mask` parameter: 8-element runs stay aligned.
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
    pub scale: f32,
    pub cfg: FaCfg,
}

/// `o = softmax(q·kᵀ·scale)·v`. Parameters in order: `q`, `k`, `v`, `o`, then
/// the mask parameters ([`AttnMask::params`]). Key blocks no query of the
/// tile sees are skipped through the first block and the trip count; only
/// blocks a mask edge crosses pay for the mask, except under a key mask,
/// which every block reads.
pub fn attention<T: Elem>(spec: &AttnSpec) -> Program {
    let AttnSpec { ref batch, t, tk, heads, kv_heads, d, mask, scale, cfg } = *spec;
    let AttnMask { causal, window, key_lens, key_mask, seg_start } = mask;
    let FaCfg { bq, bkv, stages } = cfg;
    let cap = batch.capacity();
    let mut k = Kernel::new("flash_attention");
    let q = k.param::<T>("q", ParamKind::In, cap * t * heads * d);
    let kk = k.param::<T>("k", ParamKind::In, cap * tk * kv_heads * d);
    let v = k.param::<T>("v", ParamKind::In, cap * tk * kv_heads * d);
    let o = k.param::<T>("o", ParamKind::Out, cap * t * heads * d);
    let lens = key_lens.then(|| k.param::<I32>("key_lens", ParamKind::In, cap));
    let mask_stride = key_mask_stride(tk);
    let masks = key_mask.then(|| k.param::<I32>("key_mask", ParamKind::In, cap * mask_stride));
    let segs = seg_start.then(|| k.param::<I32>("seg_start", ParamKind::In, cap * t));
    let (gz, bb) = batch.axis(&mut k);
    k.grid([Sc::from(t.div_ceil(bq)), Sc::from(heads), gz]);
    k.warps(cfg.warps());
    let k_s = k.smem::<T>("k_s", stages * bkv * d);
    let v_s = k.smem::<T>("v_s", stages * bkv * d);

    let (qb, hh) = (k.block(0), k.block(1));
    let q_off = qb * bq;
    let b = bb.clone().unwrap_or(Sc::from(0));
    // Keys past the length are masked; keys past `tk` too when blocks overhang it.
    let keys_masked = key_lens || !tk.is_multiple_of(bkv);
    let len = match lens {
        Some(lens) => k.load_scalar(lens, b.clone()).max(1).min(tk),
        None => Sc::from(tk),
    };
    let (q_stride, kv_stride) = (heads * d, kv_heads * d);
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
    let o_view = rows(&mut k, o);
    let kv_base = batch_offset(&bb, tk * kv_stride) + hh / (heads / kv_heads) * d;
    let kv_bound = keys_masked.then(|| len.clone());
    let k_view = k.view(kk, kv_base.clone(), [kv_stride, 1], Shape::new(bkv, d), [kv_bound.clone(), None]);
    let v_view = k.view(v, kv_base, [kv_stride, 1], Shape::new(bkv, d), [kv_bound, None]);
    let mask_view = masks.map(|m| {
        let view = k.view(m, b.clone() * mask_stride, [mask_stride, 1], Shape::new(1, bkv), [None, bound(tk, bkv)]);
        k.at(view, 0, 0)
    });
    // The tile's segment starts, loaded once; the mask is monotonic, so the
    // first and last rows bound the tile's.
    let seg = segs.map(|s| {
        let view = k.view(s, b.clone() * t, [1, 1], Shape::new(bq, 1), [bound(t, bq), None]);
        let tile = k.at(view, q_off.clone(), 0);
        let last = (q_off.clone() + bq - 1).min(t - 1);
        (k.load(tile), k.load_scalar(s, b.clone() * t + q_off.clone()), k.load_scalar(s, b.clone() * t + last))
    });

    // Key blocks `first..end` hold every key some query of the tile sees.
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
    let blocks = (end - first.clone()).max(1);

    let q_r = k.load(q_view);
    let m0 = k.fill::<F32>(Shape::new(bq, 1), Const::Float(-1e30));
    let l0 = k.zeros::<F32>(Shape::new(bq, 1));
    let o0 = k.zeros::<F32>(Shape::new(bq, d));
    let scale_log2e = (scale * std::f32::consts::LOG2_E) as f64;
    let [_m, l, acc] = k.pipeline(
        blocks,
        stages,
        [m0, l0, o0],
        |k, step, slot| {
            let kv_off = (first.clone() + step) * bkv;
            let k_g = k.at(k_view, kv_off.clone(), 0);
            let v_g = k.at(v_view, kv_off, 0);
            let k_t = k.smem_slot::<T>(k_s, slot.clone(), Shape::new(bkv, d));
            let v_t = k.smem_slot::<T>(v_s, slot, Shape::new(bkv, d));
            k.stage(k_t, k_g, CopyMode::Async);
            k.stage(v_t, v_g, CopyMode::Async);
        },
        |k, step, slot, [m, l, o]| {
            let k_t = k.smem_slot::<T>(k_s, slot.clone(), Shape::new(bkv, d));
            let v_t = k.smem_slot::<T>(v_s, slot, Shape::new(bkv, d));
            let zero = k.zeros::<F32>(Shape::new(bq, bkv));
            let s = k.mma(zero, q_r, false, k_t, true);
            let scale = k.fill::<F32>(Shape::new(bq, bkv), Const::Float(scale_log2e));
            let mut s = k.binary(s, scale, BinaryOp::Mul);
            let block_start = (first.clone() + step) * bkv;
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
            let masked = |k: &mut Kernel, plain| {
                let shape = Shape::new(bq, bkv);
                let col = k.coord(shape, Axis::Col);
                let start = k.splat::<I32>(shape, block_start.clone());
                let col = k.binary(col, start, BinaryOp::Add);
                let row = k.coord(shape, Axis::Row);
                let q_off = k.splat::<I32>(shape, q_off.clone());
                let row = k.binary(row, q_off, BinaryOp::Add);
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
                let hidden = k.fill::<F32>(shape, Const::Float(f64::NEG_INFINITY));
                k.where_(keep.expect("a mask"), plain, hidden)
            };
            if key_mask {
                s = masked(k, s);
            } else if causal || keys_masked || window.is_some() || seg.is_some() {
                let plain = s;
                [s] = k.select_if(partial, |k| [masked(k, plain)], |_| [plain]);
            }
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
        },
    );
    let out = k.binary(acc, l, BinaryOp::Div);
    let out = k.cast::<F32, T>(out);
    k.store(o_view, out);
    k.finish()
}
