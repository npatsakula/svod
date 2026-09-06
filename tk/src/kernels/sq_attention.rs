//! FP32 single-query attention for decoder inference.
//!
//! One wave owns one `(batch, head)`. Q stays resident in registers while K/V
//! stream over N; lane `l` owns dimensions `l + j*wave_size`. Dot products use
//! XOR-shuffle all-reduces and a one-pass stable online softmax. Long unmasked
//! attention can split K/V into contiguous chunks and associatively merge their
//! FP32 softmax states. There is no LDS or MFMA.

use std::sync::Arc;

use smallvec::smallvec;
use snafu::ensure;
use svod_dtype::{AmdArch, CudaArch, DType};
use svod_ir::{ConstValue, UOp};
use svod_tensor::Tensor;

use crate::index::{Idx, flat_index, flat_offset, index_off_gated, load_at, load_off_gated};
use crate::scaffold::GlSpec;
use crate::{ArchCaps, ArchSet, Kernel};

/// Architectures on which the scalar shuffle implementation is supported: the AMD
/// pair plus CUDA from Ampere up (the kernel needs only `shfl.sync` and `ex2`).
pub const SQ_ATTENTION_SUPPORTED_ARCHS: ArchSet =
    ArchSet::amd(&[AmdArch::Gfx942, AmdArch::Gfx1151]).with_cuda_from(CudaArch::from_compute_capability(8, 0));

/// Compile-time masking options for [`single_query_attention`].
#[derive(Clone, Copy)]
pub struct SqAttentionOpts<'a> {
    /// Optional `[B]` i32 valid-key counts. Keys `0..key_lens[b]` are valid.
    pub key_lens: Option<&'a Tensor>,
    /// Also include key `N-1`. Required when `key_lens` is present; this is the
    /// Whisper self-cache layout where the current token occupies the final slot.
    pub include_last: bool,
    /// Number of contiguous K/V chunks. Values above one are supported only for
    /// unmasked attention when `N` is divisible by `split`.
    pub split: usize,
}

impl Default for SqAttentionOpts<'_> {
    fn default() -> Self {
        Self { key_lens: None, include_last: false, split: 1 }
    }
}

fn cidx(v: i64) -> Arc<UOp> {
    UOp::index_const(v)
}

fn f32c(v: f64) -> Arc<UOp> {
    UOp::const_(DType::Float32, ConstValue::Float(v))
}

#[derive(Clone, Copy)]
pub(crate) struct HeadSelection {
    pub(crate) count: usize,
    pub(crate) total: usize,
    pub(crate) offset: usize,
}

/// Build the one-wave single-query attention kernel.
///
/// ABI is `out, q, k, v, [key_lens]`, with sequence-major `[B,S,H,D]` Q/output
/// and `[B,S,H_total,D]` K/V globals.
pub(crate) fn build_single_query_attention(
    ker: &Kernel,
    b: usize,
    n: usize,
    heads: HeadSelection,
    d: usize,
    masked: bool,
    include_last: bool,
) {
    let wave = ker.caps.wave_size;
    Kernel::assert_divisible(d, wave, "single-query attention D");
    assert!(n > 0, "single-query attention N must be > 0");
    assert!(!masked || include_last, "masked single-query attention must include the appended key");
    let ept = d / wave;
    let warp = ker.warp();
    let f32 = DType::Float32;

    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[b, 1, heads.count, d], f32.clone())],
        &[
            GlSpec::new(&[b, 1, heads.count, d], f32.clone()),
            GlSpec::new(&[b, n, heads.total, d], f32.clone()),
            GlSpec::new(&[b, n, heads.total, d], f32.clone()),
        ],
    );
    let (out, q, k, v) = (outs[0].clone(), ins[0].clone(), ins[1].clone(), ins[2].clone());
    let batch = ker.grid_y();
    let head = ker.grid_x();
    let packed_head = head.add(&cidx(heads.offset as i64));
    let lane = ker.laneid();
    let prefix = masked.then(|| {
        let lens = ker.gl(&[b], DType::Int32);
        load_at(lens.uop(), lens.shape(), &[Idx::from(&batch)])
    });

    let q_reg = ker.alloc_reg(ept, f32.clone());
    let o_reg = ker.alloc_reg(ept, f32.clone());
    let max_reg = ker.alloc_reg(1, f32.clone());
    let norm_reg = ker.alloc_reg(1, f32.clone());
    let scale = f32c(std::f64::consts::LOG2_E / (d as f64).sqrt());

    let mut init = Vec::with_capacity(2 * ept + 2);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let qv = load_at(q.uop(), q.shape(), &[Idx::from(&batch), Idx::Const(0), Idx::from(&head), Idx::from(dim)])
            .mul(&scale);
        init.push(flat_index(&q_reg, &[ept], &[Idx::Const(j as i64)]).store(qv));
        init.push(flat_index(&o_reg, &[ept], &[Idx::Const(j as i64)]).store(f32c(0.0)));
    }
    init.push(flat_index(&max_reg, &[1], &[Idx::Const(0)]).store(f32c(f64::NEG_INFINITY)));
    init.push(flat_index(&norm_reg, &[1], &[Idx::Const(0)]).store(f32c(0.0)));
    let initialized = UOp::group(init);
    let q_reg = q_reg.after(smallvec![initialized.clone()]);
    let o_reg = o_reg.after(smallvec![initialized.clone()]);
    let max_reg = max_reg.after(smallvec![initialized.clone()]);
    let norm_reg = norm_reg.after(smallvec![initialized]);

    let lp = match &prefix {
        Some(prefix) => ker.loop_dynamic(prefix.add(&cidx(1))),
        None => ker.loop_static(n as i64),
    };
    let loop_index = lp.index().clone();
    // Whisper keeps the current token after the fixed cache. A masked launch
    // streams only the valid prefix, then maps its final iteration to that slot.
    let key = match &prefix {
        Some(prefix) => {
            UOp::try_where(loop_index.lt(prefix), loop_index.clone(), cidx(n as i64 - 1)).expect("select appended key")
        }
        None => loop_index,
    };
    let q_loop = q_reg.after(smallvec![key.clone()]);
    let o_loop = o_reg.after(smallvec![key.clone()]);
    let max_loop = max_reg.after(smallvec![key.clone()]);
    let norm_loop = norm_reg.after(smallvec![key.clone()]);

    let mut dot = f32c(0.0);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let qv = load_at(&q_loop, &[ept], &[Idx::Const(j as i64)]);
        let kv =
            load_at(k.uop(), k.shape(), &[Idx::from(&batch), Idx::from(&key), Idx::from(&packed_head), Idx::from(dim)]);
        dot = dot.add(&qv.mul(&kv));
    }
    let score = warp.wave_reduce_scalar(dot, |a, p| a.add(p));
    let old_max = load_at(&max_loop, &[1], &[Idx::Const(0)]);
    let old_norm = load_at(&norm_loop, &[1], &[Idx::Const(0)]);
    let next_max = old_max.max(&score);
    let alpha = old_max.sub(&next_max).try_exp2().expect("exp2 alpha");
    let beta = score.sub(&next_max).try_exp2().expect("exp2 beta");
    let new_norm = old_norm.mul(&alpha).add(&beta);

    let max_store = flat_index(&max_reg, &[1], &[Idx::Const(0)]).store(next_max);
    let norm_store = flat_index(&norm_reg.after(smallvec![max_store.clone()]), &[1], &[Idx::Const(0)]).store(new_norm);
    let mut output_stores = Vec::with_capacity(ept);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let old_o = load_at(&o_loop, &[ept], &[Idx::Const(j as i64)]);
        let vv =
            load_at(v.uop(), v.shape(), &[Idx::from(&batch), Idx::from(&key), Idx::from(&packed_head), Idx::from(dim)]);
        let new_o = old_o.mul(&alpha).add(&vv.mul(&beta));
        output_stores.push(
            flat_index(&o_reg.after(smallvec![norm_store.clone()]), &[ept], &[Idx::Const(j as i64)]).store(new_o),
        );
    }
    let output_group = UOp::group(output_stores);
    ker.push_store(output_group, o_reg.clone());
    let ended = lp.close();

    let final_o = o_reg.after(smallvec![ended.clone()]);
    let final_norm = norm_reg.after(smallvec![ended]);
    let denom = load_at(&final_norm, &[1], &[Idx::Const(0)]);
    let mut stores = Vec::with_capacity(ept);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let value = load_at(&final_o, &[ept], &[Idx::Const(j as i64)]).try_div(&denom).expect("normalize");
        stores.push(
            flat_index(out.uop(), out.shape(), &[Idx::from(&batch), Idx::Const(0), Idx::from(&head), Idx::from(dim)])
                .store(value),
        );
    }
    ker.push_store(UOp::group(stores), out.uop().clone());
}

/// Build one unnormalized online-softmax state per contiguous K/V split.
///
/// ABI is `numerator, stats, q, k, v`; K/V use `[B,N,H_total,D]`, while
/// outputs are `[B,S,H,D]` and `[B,S,H,2]`, where the final axis of stats is
/// `(max, norm)`.
pub(crate) fn build_single_query_attention_partial(
    ker: &Kernel,
    b: usize,
    n: usize,
    heads: HeadSelection,
    d: usize,
    splits: usize,
) {
    const SUBGROUP: usize = 8;
    let wave = ker.caps.wave_size;
    Kernel::assert_divisible(d, wave, "single-query attention D");
    Kernel::assert_divisible(d, SUBGROUP, "split single-query attention D");
    assert!(splits > 1 && n.is_multiple_of(splits), "split attention requires equal non-empty chunks");
    let ept = d / wave;
    let dot_ept = d / SUBGROUP;
    let chunk = n / splits;
    let groups = wave / SUBGROUP;
    let tiles = chunk.div_ceil(groups);
    let warp = ker.warp();
    let f32 = DType::Float32;

    let (outs, ins) = ker.bind_abi(
        &[
            GlSpec::new(&[b, splits, heads.count, d], f32.clone()),
            GlSpec::new(&[b, splits, heads.count, 2], f32.clone()),
        ],
        &[
            GlSpec::new(&[b, 1, heads.count, d], f32.clone()),
            GlSpec::new(&[b, n, heads.total, d], f32.clone()),
            GlSpec::new(&[b, n, heads.total, d], f32.clone()),
        ],
    );
    let (numerator, stats) = (outs[0].clone(), outs[1].clone());
    let (q, k, v) = (ins[0].clone(), ins[1].clone(), ins[2].clone());
    let head = ker.grid_x();
    let packed_head = head.add(&cidx(heads.offset as i64));
    let batch = ker.grid_y();
    let split = ker.grid_z();
    let lane = ker.laneid();

    let q_reg = ker.alloc_reg(dot_ept, f32.clone());
    let o_reg = ker.alloc_reg(ept, f32.clone());
    let max_reg = ker.alloc_reg(1, f32.clone());
    let norm_reg = ker.alloc_reg(1, f32.clone());
    let scale = f32c(std::f64::consts::LOG2_E / (d as f64).sqrt());
    let subgroup_lane = warp.subgroup_laneid(SUBGROUP);
    let group = lane.floor_div(&cidx(SUBGROUP as i64));
    let mut init = Vec::with_capacity(dot_ept + ept + 2);
    for j in 0..dot_ept {
        let dim = subgroup_lane.add(&cidx((j * SUBGROUP) as i64));
        let qv = load_at(q.uop(), q.shape(), &[Idx::from(&batch), Idx::Const(0), Idx::from(&head), Idx::from(dim)])
            .mul(&scale);
        init.push(flat_index(&q_reg, &[dot_ept], &[Idx::Const(j as i64)]).store(qv));
    }
    for j in 0..ept {
        init.push(flat_index(&o_reg, &[ept], &[Idx::Const(j as i64)]).store(f32c(0.0)));
    }
    init.push(flat_index(&max_reg, &[1], &[Idx::Const(0)]).store(f32c(f64::NEG_INFINITY)));
    init.push(flat_index(&norm_reg, &[1], &[Idx::Const(0)]).store(f32c(0.0)));
    let initialized = UOp::group(init);
    let q_reg = q_reg.after(smallvec![initialized.clone()]);
    let o_reg = o_reg.after(smallvec![initialized.clone()]);
    let max_reg = max_reg.after(smallvec![initialized.clone()]);
    let norm_reg = norm_reg.after(smallvec![initialized]);

    let lp = ker.loop_static(tiles as i64);
    let tile_offset = lp.index().mul(&cidx(groups as i64));
    let group_offset = tile_offset.add(&group);
    let valid = group_offset.lt(&cidx(chunk as i64));
    let key = split.mul(&cidx(chunk as i64)).add(&group_offset);
    let q_loop = q_reg.after(smallvec![key.clone()]);
    let o_loop = o_reg.after(smallvec![key.clone()]);
    let max_loop = max_reg.after(smallvec![key.clone()]);
    let norm_loop = norm_reg.after(smallvec![key.clone()]);
    let mut dot = f32c(0.0);
    for j in 0..dot_ept {
        let dim = subgroup_lane.add(&cidx((j * SUBGROUP) as i64));
        let qv = load_at(&q_loop, &[dot_ept], &[Idx::Const(j as i64)]);
        let k_off =
            flat_offset(k.shape(), &[Idx::from(&batch), Idx::from(&key), Idx::from(&packed_head), Idx::from(dim)]);
        let kv = load_off_gated(k.uop(), k_off, valid.clone(), f32c(0.0));
        dot = dot.add(&qv.mul(&kv));
    }
    let score = warp.subgroup_reduce_scalar(dot, SUBGROUP, |a, p| a.add(p));
    let score = UOp::try_where(valid.clone(), score, f32c(f64::NEG_INFINITY)).expect("mask tail score");
    let old_max = load_at(&max_loop, &[1], &[Idx::Const(0)]);
    let old_norm = load_at(&norm_loop, &[1], &[Idx::Const(0)]);
    let tile_max = warp.wave_reduce_scalar(score.clone(), |a, p| a.max(p));
    let next_max = old_max.max(&tile_max);
    let alpha = old_max.sub(&next_max).try_exp2().expect("exp2 alpha");
    let beta = score.sub(&next_max).try_exp2().expect("exp2 beta");
    let representative = subgroup_lane.eq(&cidx(0));
    let norm_term = UOp::try_where(representative, beta.clone(), f32c(0.0)).expect("one beta per subgroup");
    let tile_norm = warp.wave_reduce_scalar(norm_term, |a, p| a.add(p));
    let max_store = flat_index(&max_reg, &[1], &[Idx::Const(0)]).store(next_max);
    let norm_store = flat_index(&norm_reg.after(smallvec![max_store.clone()]), &[1], &[Idx::Const(0)])
        .store(old_norm.mul(&alpha).add(&tile_norm));
    let group_betas: Vec<_> = (0..groups).map(|g| warp.broadcast_scalar(&beta, (g * SUBGROUP) as i64)).collect();
    let mut output_stores = Vec::with_capacity(ept);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let old_o = load_at(&o_loop, &[ept], &[Idx::Const(j as i64)]);
        let mut tile_o = f32c(0.0);
        for (g, group_beta) in group_betas.iter().enumerate() {
            let group_key_offset = tile_offset.add(&cidx(g as i64));
            let group_valid = group_key_offset.lt(&cidx(chunk as i64));
            let group_key = split.mul(&cidx(chunk as i64)).add(&group_key_offset);
            let v_off = flat_offset(
                v.shape(),
                &[Idx::from(&batch), Idx::from(&group_key), Idx::from(&packed_head), Idx::from(dim.clone())],
            );
            let vv = load_off_gated(v.uop(), v_off, group_valid, f32c(0.0));
            tile_o = tile_o.add(&vv.mul(group_beta));
        }
        output_stores.push(
            flat_index(&o_reg.after(smallvec![norm_store.clone()]), &[ept], &[Idx::Const(j as i64)])
                .store(old_o.mul(&alpha).add(&tile_o)),
        );
    }
    ker.push_store(UOp::group(output_stores), o_reg.clone());
    let ended = lp.close();

    let final_o = o_reg.after(smallvec![ended.clone()]);
    let final_max = max_reg.after(smallvec![ended.clone()]);
    let final_norm = norm_reg.after(smallvec![ended]);
    let mut numerator_stores = Vec::with_capacity(ept);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        numerator_stores.push(
            flat_index(
                numerator.uop(),
                numerator.shape(),
                &[Idx::from(&batch), Idx::from(&split), Idx::from(&head), Idx::from(dim)],
            )
            .store(load_at(&final_o, &[ept], &[Idx::Const(j as i64)])),
        );
    }
    ker.push_store(UOp::group(numerator_stores), numerator.uop().clone());

    let lane_zero = lane.eq(&cidx(0));
    let max_off = flat_offset(stats.shape(), &[Idx::from(&batch), Idx::from(&split), Idx::from(&head), Idx::Const(0)]);
    let norm_off = flat_offset(stats.shape(), &[Idx::from(&batch), Idx::from(&split), Idx::from(&head), Idx::Const(1)]);
    let stats_stores = UOp::group(vec![
        index_off_gated(stats.uop(), max_off, lane_zero.clone()).store(load_at(&final_max, &[1], &[Idx::Const(0)])),
        index_off_gated(stats.uop(), norm_off, lane_zero).store(load_at(&final_norm, &[1], &[Idx::Const(0)])),
    ]);
    ker.push_store(stats_stores, stats.uop().clone());
}

/// Merge split online-softmax states without rereading K/V.
pub(crate) fn build_single_query_attention_merge(ker: &Kernel, b: usize, h: usize, d: usize, splits: usize) {
    let wave = ker.caps.wave_size;
    Kernel::assert_divisible(d, wave, "single-query attention D");
    let ept = d / wave;
    let f32 = DType::Float32;
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[b, 1, h, d], f32.clone())],
        &[GlSpec::new(&[b, splits, h, d], f32.clone()), GlSpec::new(&[b, splits, h, 2], f32.clone())],
    );
    let (out, numerator, stats) = (outs[0].clone(), ins[0].clone(), ins[1].clone());
    let head = ker.grid_x();
    let batch = ker.grid_y();
    let lane = ker.laneid();
    let o_reg = ker.alloc_reg(ept, f32.clone());
    let max_reg = ker.alloc_reg(1, f32.clone());
    let norm_reg = ker.alloc_reg(1, f32.clone());
    let mut init = Vec::with_capacity(ept + 2);
    for j in 0..ept {
        init.push(flat_index(&o_reg, &[ept], &[Idx::Const(j as i64)]).store(f32c(0.0)));
    }
    init.push(flat_index(&max_reg, &[1], &[Idx::Const(0)]).store(f32c(f64::NEG_INFINITY)));
    init.push(flat_index(&norm_reg, &[1], &[Idx::Const(0)]).store(f32c(0.0)));
    let initialized = UOp::group(init);
    let o_reg = o_reg.after(smallvec![initialized.clone()]);
    let max_reg = max_reg.after(smallvec![initialized.clone()]);
    let norm_reg = norm_reg.after(smallvec![initialized]);

    let lp = ker.loop_static(splits as i64);
    let split = lp.index().clone();
    let o_loop = o_reg.after(smallvec![split.clone()]);
    let max_loop = max_reg.after(smallvec![split.clone()]);
    let norm_loop = norm_reg.after(smallvec![split.clone()]);
    let old_max = load_at(&max_loop, &[1], &[Idx::Const(0)]);
    let old_norm = load_at(&norm_loop, &[1], &[Idx::Const(0)]);
    let partial_max =
        load_at(stats.uop(), stats.shape(), &[Idx::from(&batch), Idx::from(&split), Idx::from(&head), Idx::Const(0)]);
    let partial_norm =
        load_at(stats.uop(), stats.shape(), &[Idx::from(&batch), Idx::from(&split), Idx::from(&head), Idx::Const(1)]);
    let next_max = old_max.max(&partial_max);
    let alpha = old_max.sub(&next_max).try_exp2().expect("exp2 merge alpha");
    let beta = partial_max.sub(&next_max).try_exp2().expect("exp2 merge beta");
    let max_store = flat_index(&max_reg, &[1], &[Idx::Const(0)]).store(next_max);
    let norm_store = flat_index(&norm_reg.after(smallvec![max_store.clone()]), &[1], &[Idx::Const(0)])
        .store(old_norm.mul(&alpha).add(&partial_norm.mul(&beta)));
    let mut output_stores = Vec::with_capacity(ept);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let old_o = load_at(&o_loop, &[ept], &[Idx::Const(j as i64)]);
        let partial_o = load_at(
            numerator.uop(),
            numerator.shape(),
            &[Idx::from(&batch), Idx::from(&split), Idx::from(&head), Idx::from(dim)],
        );
        output_stores.push(
            flat_index(&o_reg.after(smallvec![norm_store.clone()]), &[ept], &[Idx::Const(j as i64)])
                .store(old_o.mul(&alpha).add(&partial_o.mul(&beta))),
        );
    }
    ker.push_store(UOp::group(output_stores), o_reg.clone());
    let ended = lp.close();

    let final_o = o_reg.after(smallvec![ended.clone()]);
    let final_norm = norm_reg.after(smallvec![ended]);
    let denom = load_at(&final_norm, &[1], &[Idx::Const(0)]);
    let mut stores = Vec::with_capacity(ept);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let value = load_at(&final_o, &[ept], &[Idx::Const(j as i64)]).try_div(&denom).expect("normalize merge");
        stores.push(
            flat_index(out.uop(), out.shape(), &[Idx::from(&batch), Idx::Const(0), Idx::from(&head), Idx::from(dim)])
                .store(value),
        );
    }
    ker.push_store(UOp::group(stores), out.uop().clone());
}

/// Graph-native FP32 single-query attention.
///
/// Q is `[B,1,H,D]`, K/V are `[B,N,H_total,D]`, and output is `[B,1,H,D]`.
/// The first `H` K/V heads are selected.
/// Returns `Ok(None)` when the target is outside [`SQ_ATTENTION_SUPPORTED_ARCHS`].
/// No generic SDPA fallback is performed here.
pub fn single_query_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    opts: SqAttentionOpts<'_>,
) -> crate::LaunchResult<Option<Tensor>> {
    single_query_attention_packed(q, k, v, 0, opts)
}

/// Graph-native FP32 single-query attention over selected heads in packed K/V.
///
/// Q is `[B,1,H,D]`, K/V are `[B,N,H_total,D]`, and heads
/// `head_offset..head_offset+H` are selected without materializing a slice.
pub fn single_query_attention_packed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_offset: usize,
    opts: SqAttentionOpts<'_>,
) -> crate::LaunchResult<Option<Tensor>> {
    let qd = crate::launch::concrete_dims(q, "single-query attention", "q", 4)?;
    let kd = crate::launch::concrete_dims(k, "single-query attention", "k", 4)?;
    let vd = crate::launch::concrete_dims(v, "single-query attention", "v", 4)?;
    let (b, n, h, h_total, d) = (qd[0], kd[1], qd[2], kd[2], qd[3]);
    let dtype = q.uop().dtype();
    let masked = opts.key_lens.is_some();
    let splits = opts.split;
    let heads = HeadSelection { count: h, total: h_total, offset: head_offset };

    ensure!(
        qd[1] == 1,
        crate::launch::DimMultipleSnafu {
            kernel: "single-query attention",
            dim: "Q sequence",
            value: qd[1],
            multiple: 1usize
        }
    );
    ensure!(
        kd[0] == b,
        crate::launch::OperandDimMismatchSnafu { kernel: "single-query attention", dim: "K batch B", a: kd[0], b }
    );
    ensure!(
        head_offset <= h_total && h <= h_total - head_offset,
        crate::launch::OperandDimMismatchSnafu {
            kernel: "single-query attention",
            dim: "selected K heads (offset + H <= H_total)",
            a: head_offset.saturating_add(h),
            b: h_total
        }
    );
    ensure!(
        kd[3] == d,
        crate::launch::OperandDimMismatchSnafu {
            kernel: "single-query attention",
            dim: "K head dim D",
            a: kd[3],
            b: d
        }
    );
    for (dim, a, expected) in [
        ("V batch B", vd[0], b),
        ("V sequence N", vd[1], n),
        ("V total heads H_total", vd[2], h_total),
        ("V head dim D", vd[3], d),
    ] {
        ensure!(
            a == expected,
            crate::launch::OperandDimMismatchSnafu { kernel: "single-query attention", dim, a, b: expected }
        );
    }
    ensure!(
        n > 0,
        crate::launch::DimMultipleSnafu {
            kernel: "single-query attention",
            dim: "N (> 0)",
            value: n,
            multiple: 1usize
        }
    );
    ensure!(
        !masked || opts.include_last,
        crate::launch::DimMultipleSnafu {
            kernel: "single-query attention",
            dim: "include_last (required with key_lens)",
            value: opts.include_last as usize,
            multiple: 1usize
        }
    );
    ensure!(
        splits > 0 && (!masked || splits == 1) && (masked || n.is_multiple_of(splits)),
        crate::launch::DimMultipleSnafu {
            kernel: "single-query attention",
            dim: "split (unmasked divisor of N; masked requires 1)",
            value: splits,
            multiple: 1usize
        }
    );
    if let Some(lens) = opts.key_lens {
        let ld = crate::launch::concrete_dims(lens, "single-query attention", "key_lens", 1)?;
        ensure!(
            ld == [b],
            crate::launch::OperandDimMismatchSnafu { kernel: "single-query attention", dim: "key_lens B", a: ld[0], b }
        );
        ensure!(
            lens.uop().dtype() == DType::Int32,
            crate::launch::DtypeSnafu { kernel: "single-query attention", got: lens.uop().dtype(), expected: "i32" }
        );
    }

    crate::launch_custom(
        &q.device(),
        SQ_ATTENTION_SUPPORTED_ARCHS,
        move |_arch| {
            ensure!(
                dtype == DType::Float32,
                crate::launch::DtypeSnafu { kernel: "single-query attention", got: dtype.clone(), expected: "f32" }
            );
            ensure!(
                k.uop().dtype() == DType::Float32,
                crate::launch::DtypeSnafu { kernel: "single-query attention", got: k.uop().dtype(), expected: "f32" }
            );
            ensure!(
                v.uop().dtype() == DType::Float32,
                crate::launch::DtypeSnafu { kernel: "single-query attention", got: v.uop().dtype(), expected: "f32" }
            );
            Ok(())
        },
        // The kernel loads `d / wave` elements per lane, so a head dim the
        // arch's wave size does not divide is a fit failure of THIS runtime
        // instance (wave is 32 or 64 by arch), not a caller bug: decline to
        // `Ok(None)` and let the caller's generic attention path take over.
        move |arch| d.is_multiple_of(ArchCaps::for_arch(arch).wave_size),
        move |arch| {
            let caps = ArchCaps::for_arch(arch);
            if splits == 1 {
                let out = Tensor::empty(&[b, 1, h, d], DType::Float32);
                let mut inputs = vec![q, k, v];
                if let Some(lens) = opts.key_lens {
                    inputs.push(lens);
                }
                crate::graph_launch(
                    "sq_attention",
                    [h as i64, b as i64, 1],
                    caps.wave_size as i64,
                    out,
                    &inputs,
                    caps,
                    move |ker| {
                        build_single_query_attention(ker, b, n, heads, d, masked, opts.include_last);
                        ker.finish(1)
                    },
                )
            } else {
                let partials = crate::graph_launch_multi(
                    "sq_attention_partial",
                    [h as i64, b as i64, splits as i64],
                    caps.wave_size as i64,
                    vec![
                        Tensor::empty(&[b, splits, h, d], DType::Float32),
                        Tensor::empty(&[b, splits, h, 2], DType::Float32),
                    ],
                    &[q, k, v],
                    caps,
                    move |ker| {
                        build_single_query_attention_partial(ker, b, n, heads, d, splits);
                        ker.finish(2)
                    },
                )?;
                crate::graph_launch(
                    "sq_attention_merge",
                    [h as i64, b as i64, 1],
                    caps.wave_size as i64,
                    Tensor::empty(&[b, 1, h, d], DType::Float32),
                    &[&partials[0], &partials[1]],
                    caps,
                    move |ker| {
                        build_single_query_attention_merge(ker, b, h, d, splits);
                        ker.finish(1)
                    },
                )
            }
        },
    )
}
