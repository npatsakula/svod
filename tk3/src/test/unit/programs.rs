use crate::build::*;
use crate::ir::*;

/// `c[m, n] = a[m, k] · b[n, k]ᵀ` as a block-level pipeline: a `[bm, bn]` tile per
/// block over a `stages`-deep ring of `[bm, bk]`/`[bn, bk]` shared slots, with a
/// bound batch variable `b` on grid axis 2 (unused by the addressing).
pub fn gemm_nt(m: usize, n: usize, kk: usize, bm: usize, bn: usize, bk: usize, stages: usize) -> Program {
    gemm_nt_ordered(m, n, kk, bm, bn, bk, stages, 0)
}

/// [`gemm_nt`] walking the tile grid in groups of `group_m` tile rows (0 =
/// row-major), so blocks resident together share B columns in L2.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nt_ordered(
    m: usize,
    n: usize,
    kk: usize,
    bm: usize,
    bn: usize,
    bk: usize,
    stages: usize,
    group_m: usize,
) -> Program {
    let mut k = Kernel::new("gemm");
    let trips = kk / bk;
    let a = k.param::<BF16>("a", ParamKind::In, m * kk);
    let b = k.param::<BF16>("b", ParamKind::In, n * kk);
    let c = k.param::<BF16>("c", ParamKind::Out, m * n);
    let batch = k.var("b", 1, 8);
    let [gm, gn] = [Sc::from(m / bm), Sc::from(n / bn)];
    k.grid([gm, gn, batch]);
    k.warps(8);
    let a_s = k.smem::<BF16>("a_s", stages * bm * bk);
    let b_s = k.smem::<BF16>("b_s", stages * bn * bk);

    let (bx, by) = if group_m == 0 {
        (k.block(0), k.block(1))
    } else {
        let (gm, gn) = (m / bm, n / bn);
        let id = k.block(0) + k.block(1) * gm;
        let group = id.clone() / (group_m * gn);
        let first_m = group * group_m;
        let rows = (Sc::from(gm) - first_m.clone()).min(group_m);
        let within = id % (group_m * gn);
        let bx = first_m + within.clone() % rows.clone();
        let by = within / rows;
        (bx, by)
    };
    let a_view = k.view(a, 0, [kk, 1], Shape::new(bm, bk), [None, None]);
    let b_view = k.view(b, 0, [kk, 1], Shape::new(bn, bk), [None, None]);
    let row0 = bx * bm;
    let col0 = by * bn;
    let a_view = k.at(a_view, row0.clone(), 0);
    let b_view = k.at(b_view, col0.clone(), 0);
    let acc0 = k.zeros::<F32>(Shape::new(bm, bn));
    let [acc] = k.pipeline(
        trips,
        stages,
        [acc0],
        |k, step, slot| {
            let koff = step * bk;
            let a_g = k.at(a_view, 0, koff.clone());
            let b_g = k.at(b_view, 0, koff);
            let a_t = k.smem_slot::<BF16>(a_s, slot.clone(), Shape::new(bm, bk));
            let b_t = k.smem_slot::<BF16>(b_s, slot, Shape::new(bn, bk));
            k.stage(a_t, a_g, CopyMode::Async);
            k.stage(b_t, b_g, CopyMode::Async);
        },
        |k, _step, slot, [acc]| {
            let a_t = k.smem_slot::<BF16>(a_s, slot.clone(), Shape::new(bm, bk));
            let b_t = k.smem_slot::<BF16>(b_s, slot, Shape::new(bn, bk));
            [k.mma(acc, a_t, false, b_t, true)]
        },
    );
    let out = k.cast::<F32, BF16>(acc);
    let c_view = k.view(c, 0, [n, 1], Shape::new(bm, bn), [None, None]);
    let c_view = k.at(c_view, row0, col0);
    k.store(c_view, out);
    k.finish()
}

/// Shapes and options of a flash-attention forward program.
#[derive(Clone, Copy, Debug)]
pub struct FaSpec {
    /// Maximum batch (the live batch is the bound variable `b`).
    pub batch: usize,
    pub t: usize,
    pub tk: usize,
    pub heads: usize,
    pub d: usize,
    pub bq: usize,
    pub bkv: usize,
    pub stages: usize,
    pub causal: bool,
    /// A `[batch]` i32 parameter of valid key counts follows `o`.
    pub key_lens: bool,
    pub scale: f32,
}

/// `o = softmax(q · kᵀ · scale) · v` over `[batch, t, heads, d]` tensors: Q
/// resident in registers, K/V streamed through the pipeline, the online
/// softmax state `(m, l, o)` carried; causal blocks past the diagonal are
/// skipped through the dynamic extent, boundary blocks are masked.
pub fn flash_attention(spec: FaSpec) -> Program {
    let FaSpec { batch, t, tk, heads, d, bq, bkv, stages, causal, key_lens, scale } = spec;
    let mut k = Kernel::new("flash_attention");
    let q = k.param::<BF16>("q", ParamKind::In, batch * t * heads * d);
    let kk = k.param::<BF16>("k", ParamKind::In, batch * tk * heads * d);
    let v = k.param::<BF16>("v", ParamKind::In, batch * tk * heads * d);
    let o = k.param::<BF16>("o", ParamKind::Out, batch * t * heads * d);
    let lens = key_lens.then(|| k.param::<I32>("key_lens", ParamKind::In, batch));
    let live = k.var("b", 1, batch as i64);
    k.grid([Sc::from(t / bq), Sc::from(heads), live]);
    k.warps((bq / 16) as u32);
    let k_s = k.smem::<BF16>("k_s", stages * bkv * d);
    let v_s = k.smem::<BF16>("v_s", stages * bkv * d);

    let (qb, hh, bb) = (k.block(0), k.block(1), k.block(2));
    let q_off = qb * bq;
    let row_stride = heads * d;
    let len = match lens {
        Some(lens) => k.load_scalar(lens, bb.clone()).max(1),
        None => Sc::from(tk),
    };
    let q_view = k.view(
        q,
        bb.clone() * (t * row_stride) + hh.clone() * d + q_off.clone() * row_stride,
        [row_stride, 1],
        Shape::new(bq, d),
        [None, None],
    );
    let kv_base = bb.clone() * (tk * row_stride) + hh.clone() * d;
    let k_view = k.view(kk, kv_base.clone(), [row_stride, 1], Shape::new(bkv, d), [Some(len.clone()), None]);
    let v_view = k.view(v, kv_base, [row_stride, 1], Shape::new(bkv, d), [Some(len.clone()), None]);
    let o_view = k.view(
        o,
        bb * (t * row_stride) + hh * d + q_off.clone() * row_stride,
        [row_stride, 1],
        Shape::new(bq, d),
        [None, None],
    );

    // Key blocks this query block attends to: up to the diagonal when causal,
    // up to the last valid key when lengths are given.
    let mut blocks = Sc::from(tk / bkv);
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
            let k_t = k.smem_slot::<BF16>(k_s, slot.clone(), Shape::new(bkv, d));
            let v_t = k.smem_slot::<BF16>(v_s, slot, Shape::new(bkv, d));
            k.stage(k_t, k_g, CopyMode::Async);
            k.stage(v_t, v_g, CopyMode::Async);
        },
        |k, step, slot, [m, l, o]| {
            let k_t = k.smem_slot::<BF16>(k_s, slot.clone(), Shape::new(bkv, d));
            let v_t = k.smem_slot::<BF16>(v_s, slot, Shape::new(bkv, d));
            let zero = k.zeros::<F32>(Shape::new(bq, bkv));
            let s = k.mma(zero, q_r, false, k_t, true);
            let scale = k.fill::<F32>(Shape::new(bq, bkv), Const::Float(scale_log2e));
            let mut s = k.binary(s, scale, BinaryOp::Mul);
            if causal || key_lens {
                // Only a block crossing the diagonal or the key boundary pays
                // for the mask; the others take the plain path.
                let block_end = step.clone() * bkv + bkv;
                let mut partial = Sc::from(0);
                if causal {
                    partial = partial.or(q_off.clone().lt(block_end.clone()));
                }
                if key_lens {
                    partial = partial.or(len.clone().lt(block_end));
                }
                let plain = s;
                [s] = k.select_if(
                    partial,
                    |k| {
                        let col = k.coord(Shape::new(bq, bkv), Axis::Col);
                        let kv_off = k.splat::<I32>(Shape::new(bq, bkv), step.clone() * bkv);
                        let col = k.binary(col, kv_off, BinaryOp::Add);
                        let mut keep = None;
                        if causal {
                            let row = k.coord(Shape::new(bq, bkv), Axis::Row);
                            let q_off = k.splat::<I32>(Shape::new(bq, bkv), q_off.clone());
                            let row = k.binary(row, q_off, BinaryOp::Add);
                            keep = Some(k.compare(col, row, BinaryOp::Le));
                        }
                        if key_lens {
                            let len = k.splat::<I32>(Shape::new(bq, bkv), len.clone());
                            let valid = k.compare(col, len, BinaryOp::Lt);
                            keep = Some(match keep {
                                Some(keep) => k.binary(keep, valid, BinaryOp::And),
                                None => valid,
                            });
                        }
                        let masked = k.fill::<F32>(Shape::new(bq, bkv), Const::Float(f64::NEG_INFINITY));
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
            let p16 = k.cast::<F32, BF16>(p);
            let o = k.mma(o, p16, false, v_t, false);
            [m_new, l, o]
        },
    );
    let out = k.binary(acc, l, BinaryOp::Div);
    let out = k.cast::<F32, BF16>(out);
    k.store(o_view, out);
    k.finish()
}
