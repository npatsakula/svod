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
    /// Key `j` is visible to query `i` only when `j ≤ i`.
    pub causal: bool,
    /// A `[batch]` i32 parameter of valid key counts follows `o`.
    pub key_lens: bool,
    pub scale: f32,
    pub cfg: FaCfg,
}

/// `o = softmax(q·kᵀ·scale)·v`. Parameters in order: `q`, `k`, `v`, `o`, then
/// `key_lens` if any. Causal blocks past the diagonal and key blocks past the
/// length are skipped through the trip count; only blocks crossing the
/// diagonal or the last key pay for the mask.
pub fn attention<T: Elem>(spec: &AttnSpec) -> Program {
    let AttnSpec { ref batch, t, tk, heads, kv_heads, d, causal, key_lens, scale, cfg } = *spec;
    let FaCfg { bq, bkv, stages } = cfg;
    let cap = batch.capacity();
    let mut k = Kernel::new("flash_attention");
    let q = k.param::<T>("q", ParamKind::In, cap * t * heads * d);
    let kk = k.param::<T>("k", ParamKind::In, cap * tk * kv_heads * d);
    let v = k.param::<T>("v", ParamKind::In, cap * tk * kv_heads * d);
    let o = k.param::<T>("o", ParamKind::Out, cap * t * heads * d);
    let lens = key_lens.then(|| k.param::<I32>("key_lens", ParamKind::In, cap));
    let (gz, bb) = batch.axis(&mut k);
    k.grid([Sc::from(t.div_ceil(bq)), Sc::from(heads), gz]);
    k.warps(cfg.warps());
    let k_s = k.smem::<T>("k_s", stages * bkv * d);
    let v_s = k.smem::<T>("v_s", stages * bkv * d);

    let (qb, hh) = (k.block(0), k.block(1));
    let q_off = qb * bq;
    // Keys past the length are masked; keys past `tk` too when blocks overhang it.
    let keys_masked = key_lens || !tk.is_multiple_of(bkv);
    let len = match lens {
        Some(lens) => {
            let b = bb.clone().unwrap_or(Sc::from(0));
            k.load_scalar(lens, b).max(1).min(tk)
        }
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

    let mut blocks = Sc::from(tk.div_ceil(bkv));
    if causal {
        blocks = blocks.min((q_off.clone() + bq + bkv - 1) / bkv);
    }
    if key_lens {
        blocks = blocks.min((len.clone() + bkv - 1) / bkv);
    }

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
            let kv_off = step * bkv;
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
            if causal || keys_masked {
                let block_end = step.clone() * bkv + bkv;
                let mut partial = Sc::from(0);
                if causal {
                    partial = partial.or(q_off.clone().lt(block_end.clone()));
                }
                if keys_masked {
                    partial = partial.or(len.clone().lt(block_end));
                }
                let plain = s;
                [s] = k.select_if(
                    partial,
                    |k| {
                        let shape = Shape::new(bq, bkv);
                        let col = k.coord(shape, Axis::Col);
                        let kv_off = k.splat::<I32>(shape, step.clone() * bkv);
                        let col = k.binary(col, kv_off, BinaryOp::Add);
                        let mut keep = None;
                        if causal {
                            let row = k.coord(shape, Axis::Row);
                            let q_off = k.splat::<I32>(shape, q_off.clone());
                            let row = k.binary(row, q_off, BinaryOp::Add);
                            keep = Some(k.compare(col, row, BinaryOp::Le));
                        }
                        if keys_masked {
                            let len = k.splat::<I32>(shape, len.clone());
                            let valid = k.compare(col, len, BinaryOp::Lt);
                            keep = Some(match keep {
                                Some(keep) => k.binary(keep, valid, BinaryOp::And),
                                None => valid,
                            });
                        }
                        let masked = k.fill::<F32>(shape, Const::Float(f64::NEG_INFINITY));
                        [k.where_(keep.expect("a mask"), plain, masked)]
                    },
                    |_| [plain],
                );
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
