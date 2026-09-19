//! Hand-coded optimization heuristics for kernel optimization.
//!
//! Hand-coded heuristics give reasonable performance without auto-tuning.
//! Applies optimizations in order: TC → Image → GroupReduce → Upcasts → Unroll → Local → Thread.

use std::sync::Arc;

use smallvec::SmallVec;
use svod_ir::uop::{reaching, reaching_each};
use svod_ir::{AxisId, AxisType, BinaryOp, Op, RendererDevice, TernaryOp, UOp};

use crate::optimizer::config::{HeuristicsConfig, TcOpt};
use crate::optimizer::error::OptError;
use crate::optimizer::renderer::{Renderer, TcTilePolicy, TensorCore};
use crate::optimizer::tc::{MatmulPattern, matmul_operands};
use crate::optimizer::{Opt, Scheduler, apply_opt};
use svod_ir::ops;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default vectorization factor for UPCAST when no other heuristic applies.
/// Value 4 provides good SIMD utilization on most architectures (SSE/NEON).
pub const DEFAULT_UPCAST_FACTOR: usize = 4;

/// Cumulative LOCAL size budget per kernel (tinygrad heuristic.py:184).
const LOCAL_BUDGET: usize = 128;

/// Block sizes a global axis may be padded to, best first.
const PAD_BLOCKS: [usize; 3] = [32, 16, 8];

/// `size` rounded up to a multiple of `align`, when the padded (masked) tail
/// adds at most 5% of extra work.
fn padded_extent(size: usize, align: usize) -> Option<usize> {
    let padded = size.div_ceil(align) * align;
    ((padded - size) * 20 <= size).then_some(padded)
}

/// The value of an integer CONST, if that is what this is.
fn const_int(uop: &Arc<UOp>) -> Option<i64> {
    match uop.op() {
        Op::Const(cv) => match cv.0 {
            svod_ir::ConstValue::Int(value) => Some(value),
            _ => None,
        },
        _ => None,
    }
}

/// The constant extent of a RANGE, if it has one.
fn const_extent(rng: &Arc<UOp>) -> Option<usize> {
    match rng.op() {
        Op::Range(ops::Range { end, .. }) => const_int(end).filter(|&size| size > 0).map(|size| size as usize),
        _ => None,
    }
}

/// Product of the constant extents of every axis of one of `axis_types`.
fn extent_product(scheduler: &Scheduler, axis_types: &[AxisType]) -> usize {
    scheduler.ranges_of(axis_types).iter().filter_map(const_extent).product::<usize>().max(1)
}

/// Trips the accumulator is reused over: every reduce axis the kernel still
/// carries, which is what the core left of its own K times the reduces it did
/// not take. A convolution's taps stay as loops around the WMMA and the
/// accumulator is set up and written back once for all of them, so counting the
/// core's K axis alone understates the depth ninefold on a 3x3 — and then caps
/// the warp tile far below what the register budget allows.
fn reduce_depth(scheduler: &Scheduler) -> usize {
    extent_product(scheduler, &[AxisType::Reduce, AxisType::GroupReduce])
}

/// LOCAL size for a global axis none of the standard sizes divides, with the
/// PADTO alignment it needs first.
///
/// Two options compete on the fraction of lanes that do useful work in a
/// block of `cumulative * threads` (the block occupies whole waves of
/// `wave_size` lanes, and padded elements are computed then masked): the
/// largest divisor of `size` within `budget`, and the largest of
/// [`PAD_BLOCKS`] whose padding stays cheap ([`padded_extent`]). Ties keep
/// the exact divisor.
fn local_fallback(size: usize, cumulative: usize, budget: usize, wave_size: usize) -> Option<(usize, Option<usize>)> {
    let lane_efficiency = |threads: usize, useful: usize, total: usize| {
        let block = cumulative * threads;
        block as f64 / (block.div_ceil(wave_size) * wave_size) as f64 * useful as f64 / total as f64
    };
    let divisor = (2..=budget.min(size)).rev().find(|d| size.is_multiple_of(*d));
    let padded = PAD_BLOCKS
        .into_iter()
        .filter(|&block| block <= budget)
        .find_map(|block| padded_extent(size, block).map(|padded| (block, padded)));
    match (divisor, padded) {
        (Some(d), Some((block, padded))) if lane_efficiency(block, size, padded) > lane_efficiency(d, size, size) => {
            Some((block, Some(block)))
        }
        (Some(d), _) => Some((d, None)),
        (None, Some((block, _))) => Some((block, Some(block))),
        (None, None) => None,
    }
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

/// Apply hand-coded optimization heuristics to a kernel.
///
/// Heuristics are applied in order:
/// 1. Tensor cores (if matmul pattern)
/// 2. Image upcasts (if image type)
/// 3. Grouped reduction (if large reduce)
/// 4. Masked upcasts (small masked dims)
/// 5. Heuristic upcasts (stride-based ranking)
/// 6. Unroll (small reduction loops)
/// 7. Default upcast (fallback)
/// 8. Local dims (GPU workgroup)
/// 9. Threading (CPU parallel)
pub fn hand_coded_optimizations(scheduler: &mut Scheduler, config: &HeuristicsConfig) {
    use tracing::debug;

    debug!("hand_coded_optimizations: starting");

    // 1. Tensor cores (skip other opts if applied). Try TC first; return on
    // success (with post-TC UPCAST/LOCAL extras).
    if try_tensor_cores(scheduler, config) {
        debug!("hand_coded_optimizations: tensor cores applied, skipping remaining opts");
        return;
    }

    // 2. Image upcasts
    apply_image_upcasts(scheduler);

    // 2.5. Matvec fast-path
    if apply_matvec_fast_path(scheduler, config) {
        debug!("hand_coded_optimizations: matvec fast-path applied, skipping remaining opts");
        return;
    }

    // 3. Grouped reduction: few outputs share a block, many outputs get a wave
    // per row.
    if !try_grouped_reduction(scheduler, config) {
        try_warp_row_reduction(scheduler, config);
    }

    // Guard: no more opts if we are grouping.
    if scheduler.group_for_reduces() > 0 {
        debug!("hand_coded_optimizations: group_for_reduces active, skipping remaining opts");
        return;
    }

    // 4. Masked upcasts
    apply_masked_upcasts(scheduler);

    // 5. Heuristic upcasts (stride-based ranking).
    apply_heuristic_upcasts(scheduler);

    // 6. Unroll (BEFORE threading).
    apply_unroll(scheduler);

    // 7. Default upcast
    apply_default_upcast(scheduler);

    // 8. Local dims
    apply_local_dims(scheduler, config);

    // 9. Threading
    debug!("hand_coded_optimizations: calling apply_threading with max_threads={}", config.thread_count);
    let threading_applied = apply_threading(scheduler, config.thread_count);
    debug!(threading_applied, "hand_coded_optimizations: apply_threading completed");
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Check if kernel has matmul pattern: REDUCE(ADD) of MUL of INDEX ops.
pub fn has_matmul_pattern(scheduler: &Scheduler) -> bool {
    let Some(reduceop) = scheduler.reduceop() else { return false };
    matmul_operands(&reduceop)
        .is_some_and(|(a, b)| [a, b].iter().all(|operand| matches!(operand.unwrap_cast().op(), Op::Index(..))))
}

/// Check if axis is masked (appears in WHERE conditionals).
pub fn is_masked(scheduler: &Scheduler, axis: usize) -> bool {
    let rngs = scheduler.rngs();
    if axis >= rngs.len() {
        return false;
    }
    let target_rng = &rngs[axis];

    let mut reaches_target = reaching(target_rng);
    for node in scheduler.ast().toposort() {
        if let Op::Ternary(TernaryOp::Where, cond, _, _) = node.op()
            && reaches_target.contains(cond)
        {
            return true;
        }
    }
    false
}

/// Check if axis has broadcast pattern (stride-0 in some buffer).
pub fn has_broadcast_pattern(scheduler: &Scheduler, axis: usize) -> bool {
    let rngs = scheduler.rngs();
    if axis >= rngs.len() {
        return false;
    }
    let target_rng = &rngs[axis];

    // One memo shared by every buffer and index: the target range is fixed, so
    // each node's answer is computed at most once for the whole scan.
    let mut reaches_target = reaching(target_rng);
    for buf in scheduler.bufs() {
        if !reaches_target.contains(buf) {
            continue;
        }
        if let Op::Index(ops::Index { indices, .. }) = buf.op() {
            let in_index = indices.iter().any(|idx| reaches_target.contains(idx));
            if !in_index {
                return true;
            }
        }
    }
    false
}

/// Count strides for axis in buffer accesses. Returns (num_buffers, sum_strides).
///
/// - num_strides: number of buffers whose index references this range
/// - sum_strides: sum of actual stride values from the index's ADD decomposition
///   (1 for unit stride, CONST value for `range * CONST`)
pub fn count_strides(scheduler: &Scheduler, axis: usize) -> (usize, usize) {
    let rngs = scheduler.rngs();
    if axis >= rngs.len() {
        return (0, 0);
    }
    let mut reaches_target = reaching(&rngs[axis]);
    strides_of(&linearized_indices(scheduler.bufs()), &rngs[axis], |idx| reaches_target.contains(idx))
}

/// The combined linearized index of each buffer access, WHERE unwrapped.
fn linearized_indices(bufs: &[Arc<UOp>]) -> Vec<Arc<UOp>> {
    bufs.iter()
        .filter_map(|buf| match buf.op() {
            Op::Index(ops::Index { indices, .. }) => {
                Some(indices.first().map(|i| i.get_idx()).unwrap_or_else(|| buf.clone()))
            }
            _ => None,
        })
        .collect()
}

/// `count_strides` over precomputed indices; `reaches` answers whether an index
/// depends on `target_rng`, so one reachability memo can serve every axis.
fn strides_of(
    indices: &[Arc<UOp>],
    target_rng: &Arc<UOp>,
    mut reaches: impl FnMut(&Arc<UOp>) -> bool,
) -> (usize, usize) {
    let mut num_strides = 0;
    let mut sum_strides: usize = 0;
    for idx in indices {
        num_strides += usize::from(reaches(idx));

        for term in idx.split_uop(BinaryOp::Add) {
            if Arc::ptr_eq(&term, target_rng) {
                // c is rng → stride 1
                sum_strides += 1;
            } else if let Op::Binary(BinaryOp::Mul, lhs, rhs) = term.op() {
                // c.op is Ops.MUL and one side is rng and other is CONST.
                // A reversed axis (a flipped view, as `conv_transpose2d` builds)
                // indexes with a negative constant; like `min_stride`, that is
                // no forward stride at all — and must not wrap into `usize`.
                let stride = if Arc::ptr_eq(lhs, target_rng) {
                    const_int(rhs)
                } else if Arc::ptr_eq(rhs, target_rng) {
                    const_int(lhs)
                } else {
                    None
                };
                sum_strides += stride.filter(|&v| v > 0).unwrap_or(0) as usize;
            }
        }
    }
    (num_strides, sum_strides)
}

/// The smallest stride any buffer addresses `target_rng` with, in elements.
///
/// This is the quantity both memory rules key off: it is how far apart in
/// memory two neighbouring values of the axis land, so it decides whether a
/// wave walking the axis covers one contiguous run and whether a vector load
/// along it is one transaction. It is read off the linearized index — a bare
/// `target_rng` term is stride 1, `target_rng * c` is stride `c` — never off
/// the shape. `None` when no buffer's index moves with the axis at all.
fn min_stride(indices: &[Arc<UOp>], target_rng: &Arc<UOp>) -> Option<usize> {
    indices
        .iter()
        .flat_map(|idx| idx.split_uop(BinaryOp::Add))
        .filter_map(|term| {
            if Arc::ptr_eq(&term, target_rng) {
                return Some(1);
            }
            let Op::Binary(BinaryOp::Mul, lhs, rhs) = term.op() else { return None };
            let stride = if Arc::ptr_eq(lhs, target_rng) {
                const_int(rhs)
            } else if Arc::ptr_eq(rhs, target_rng) {
                const_int(lhs)
            } else {
                None
            };
            stride.filter(|&stride| stride > 0).map(|stride| stride as usize)
        })
        .min()
}

/// The memory transaction a lane's access is rounded up to, in bytes.
const SECTOR_BYTES: usize = 32;

/// Each buffer access as `(linearized index, element size in bytes)`.
fn buffer_accesses(bufs: &[Arc<UOp>]) -> Vec<(Arc<UOp>, usize)> {
    bufs.iter()
        .filter_map(|buf| match buf.op() {
            Op::Index(ops::Index { indices, .. }) => {
                Some((indices.first().map(|i| i.get_idx()).unwrap_or_else(|| buf.clone()), buf.dtype().base().bytes()))
            }
            _ => None,
        })
        .collect()
}

/// The furthest apart, in bytes, two neighbouring lanes stepping `rng` land in
/// any one buffer. A buffer whose index ignores the axis contributes nothing.
///
/// This is what decides where `lidx0` belongs. Lanes at a stride below one
/// [`SECTOR_BYTES`] share their transactions; past that each lane pays a whole
/// sector, so the axis to hand the fastest thread index is one *every* buffer
/// keeps within a sector — the worst buffer is what a wave waits for, which is
/// why this is a maximum and not a sum. When no axis manages it (a transposing
/// kernel, contiguous on one side and strided on the other) there is nothing to
/// win and the older ranking stands.
fn lane_span_bytes(accesses: &[(Arc<UOp>, usize)], rng: &Arc<UOp>) -> usize {
    accesses
        .iter()
        .map(|(idx, bytes)| min_stride(std::slice::from_ref(idx), rng).unwrap_or(0).saturating_mul(*bytes))
        .max()
        .unwrap_or(0)
}

/// Whether every buffer's index moves with `rng`.
///
/// An output axis some buffer ignores is a reuse axis: each of its values
/// re-reads that buffer, and spending a whole wave on one of its elements
/// throws the reuse away.
fn addressed_everywhere(accesses: &[(Arc<UOp>, usize)], rng: &Arc<UOp>) -> bool {
    accesses.iter().all(|(idx, _)| min_stride(std::slice::from_ref(idx), rng).is_some())
}

/// Widths a single memory access moves, in bytes. Anything else lowers to a
/// run of narrow accesses whatever the axis looks like.
fn is_vector_width(bytes: usize) -> bool {
    matches!(bytes, 4 | 8 | 16)
}

/// The widest element some buffer walks contiguously along `rng`, in bytes.
/// `None` when no buffer addresses the axis with stride 1, in which case no
/// upcast width along it can become a single access.
fn contiguous_element_bytes(accesses: &[(Arc<UOp>, usize)], rng: &Arc<UOp>) -> Option<usize> {
    accesses
        .iter()
        .filter(|(idx, _)| min_stride(std::slice::from_ref(idx), rng) == Some(1))
        .map(|&(_, bytes)| bytes)
        .max()
}

// ============================================================================
// SIMPLE HEURISTICS
// ============================================================================

/// Image-specific upcasting/unrolling.
///
/// For image buffers, find a unit-stride axis whose extent is divisible by 4.
/// Prefer UPCAST on that axis when it's output-parallel; otherwise UNROLL the
/// same axis when it's a reduction axis.
pub fn apply_image_upcasts(scheduler: &mut Scheduler) -> bool {
    let mut applied = false;

    // Snapshot to avoid borrow conflicts while mutating scheduler.
    let bufs = scheduler.bufs().to_vec();
    for buf in bufs {
        let Op::Index(ops::Index { buffer, indices, .. }) = buf.op() else {
            continue;
        };
        // The rank-3-with-4-channels shape is only an image when the buffer says so;
        // an ordinary rank-3 tensor must not be upcast on this path.
        if !buffer.dtype().is_image() {
            continue;
        }
        if !buffer.shape().ok().flatten().is_some_and(|shape| shape.len() == 3 && shape[2].as_const() == Some(4)) {
            continue;
        }

        let Some(first_idx) = indices.first() else {
            continue;
        };
        let linear_idx = first_idx.get_idx();

        // Choose first range term in linearized index with size % 4 == 0.
        let axis = linear_idx
            .split_uop(BinaryOp::Add)
            .into_iter()
            .filter_map(|term| {
                if !matches!(term.op(), Op::Range(ops::Range { end, .. }) if end.divides(4).is_some()) {
                    return None;
                }
                scheduler.rngs().iter().position(|r| Arc::ptr_eq(r, &term))
            })
            .next();

        let Some(axis) = axis else {
            continue;
        };

        if scheduler.upcastable_dims().contains(&axis) {
            if apply_opt(scheduler, &Opt::upcast(axis, 4), true).is_ok() {
                applied = true;
            }
        } else {
            let unrollable = scheduler.unrollable_dims();
            if let Some(logical_axis) = unrollable.iter().position(|&i| i == axis)
                && apply_opt(scheduler, &Opt::unroll(logical_axis, 4), true).is_ok()
            {
                applied = true;
            }
        }
    }

    applied
}

/// Default upcast fallback: 4x vectorization on the innermost upcastable axis.
///
/// Tinygrad `hand_coded_optimizations` (codegen/opt/heuristic.py:155-158):
/// `if not k.upcasted and k.upcastable_dims and full_shape[upcastable_dims[-1]] % 4 == 0`.
/// `upcasted` counts UPCAST *and* UNROLL axes, so an unrolled reduce already
/// suppresses this fallback.
pub fn apply_default_upcast(scheduler: &mut Scheduler) -> bool {
    use tracing::debug;

    if scheduler.upcasted() {
        debug!("apply_default_upcast: skipping (already upcasted or unrolled)");
        return false;
    }
    let Some(axis_idx) = scheduler.upcastable_dims().last().copied() else {
        debug!("apply_default_upcast: no upcastable dims");
        return false;
    };

    let size = scheduler.full_shape()[axis_idx];
    if size % DEFAULT_UPCAST_FACTOR as i64 != 0 {
        debug!(axis_idx, size, factor = DEFAULT_UPCAST_FACTOR, "apply_default_upcast: skipping (size not divisible)");
        return false;
    }

    let result = apply_opt(scheduler, &Opt::upcast(axis_idx, DEFAULT_UPCAST_FACTOR), true);
    debug!(?result, axis = axis_idx, factor = DEFAULT_UPCAST_FACTOR, "apply_default_upcast: apply_opt result");
    result.is_ok()
}

/// Unroll reduction loops.
///
/// Conditions: `unrollable_dims.not_empty() AND (upcast_size() <= 4 OR no UNROLL axes) AND upcast_size() < 64`
/// - Small dims (size <= 32): full unroll (amount=0)
/// - Large dims: partial unroll by 4
pub fn apply_unroll(scheduler: &mut Scheduler) -> bool {
    use tracing::debug;

    let unrollable = scheduler.unrollable_dims();
    if unrollable.is_empty() {
        return false;
    }

    let upcast_size = scheduler.upcast_size();
    let has_unroll = !scheduler.axes_of(&[AxisType::Unroll]).is_empty();

    if upcast_size >= 64 || (upcast_size > 4 && has_unroll) {
        debug!(upcast_size, has_unroll, "apply_unroll: skipping (upcast_size guard)");
        return false;
    }

    // Get last unrollable dim's size.
    let last_unrollable = *unrollable.last().unwrap();
    let rngs = scheduler.rngs();
    let size = if last_unrollable < rngs.len()
        && let Op::Range(ops::Range { end, .. }) = rngs[last_unrollable].op()
        && let Op::Const(cv) = end.op()
        && let svod_ir::ConstValue::Int(sz) = cv.0
    {
        sz as usize
    } else {
        return false;
    };

    let logical_idx = unrollable.len() - 1;

    if size <= 32 {
        // Full unroll (amount=0 means full unroll).
        // UNROLL creates expanded scalar operations (not vectors like UPCAST),
        // so non-power-of-2 sizes are safe.
        debug!(last_unrollable, size, "apply_unroll: full unroll");
        if apply_opt(scheduler, &Opt::unroll(logical_idx, 0), true).is_ok() {
            // If small, try unrolling a second reduce dimension too.
            if size <= 3 {
                let unrollable2 = scheduler.unrollable_dims();
                if let Some(&last2) = unrollable2.last() {
                    let rngs2 = scheduler.rngs();
                    if last2 < rngs2.len()
                        && let Op::Range(ops::Range { end, .. }) = rngs2[last2].op()
                        && let Op::Const(cv) = end.op()
                        && let svod_ir::ConstValue::Int(sz2) = cv.0
                        && sz2 <= 3
                    {
                        let _ = apply_opt(scheduler, &Opt::unroll(unrollable2.len() - 1, 0), true);
                    }
                }
            }
            return true;
        }
    }

    // Partial unroll by 4
    for splits in [4] {
        if size % splits == 0 {
            debug!(last_unrollable, size, splits, "apply_unroll: partial unroll");
            if apply_opt(scheduler, &Opt::unroll(logical_idx, splits), true).is_ok() {
                return true;
            }
        }
    }

    false
}

// ============================================================================
// INTERMEDIATE HEURISTICS
// ============================================================================

/// Upcast small masked dimensions (size <= 7).
///
/// Collects all masked-upcastable axes first, then applies in REVERSE order.
/// Reverse iteration is critical — upcast of a higher-indexed axis doesn't shift
/// lower-indexed axes in the rngs list, preserving index validity.
pub fn apply_masked_upcasts(scheduler: &mut Scheduler) -> bool {
    let upcastable = scheduler.upcastable_dims();

    // Phase 1: Collect candidates.
    let mut product: i64 = 1;
    let mut to_upcast: Vec<(usize, usize)> = Vec::new();

    for axis_idx in upcastable {
        if !is_masked(scheduler, axis_idx) {
            continue;
        }
        let rngs = scheduler.rngs();
        if axis_idx >= rngs.len() {
            continue;
        }
        let rng = &rngs[axis_idx];
        if let Op::Range(ops::Range { end, .. }) = rng.op()
            && let Op::Const(cv) = end.op()
            && let svod_ir::ConstValue::Int(size) = cv.0
            && size > 1
            && size <= 7
            && product * size <= 49
        {
            to_upcast.push((axis_idx, size as usize));
            product *= size;
        }
    }

    // Phase 2: Apply in reverse order.
    let mut applied = false;
    for (axis_idx, size) in to_upcast.into_iter().rev() {
        if apply_opt(scheduler, &Opt::upcast(axis_idx, size), true).is_ok() {
            applied = true;
        }
    }
    applied
}

/// Reduce axes long enough that a wave-wide split still leaves a serial loop.
const MIN_WARP_REDUCE: usize = 64;

/// The product of the output extents a reduce kernel writes.
fn output_count(scheduler: &Scheduler) -> i64 {
    let full_shape = scheduler.full_shape();
    scheduler.upcastable_dims().iter().map(|&i| full_shape.get(i).copied().unwrap_or(1)).product()
}

/// Output count up to which [`try_grouped_reduction`] takes the kernel; above
/// it [`try_warp_row_reduction`] gets its chance, so the two never compete.
fn grouped_output_threshold(config: &HeuristicsConfig) -> i64 {
    if config.disable_locals { 240 } else { 2048 }
}

/// Grouped reduction for small output dimensions.
///
/// When the product of upcastable output dimensions is small (<= 2048,
/// or 240 when local selection is disabled), apply GROUPTOP on output axes to enable
/// local reduction.
pub fn try_grouped_reduction(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    if !scheduler.renderer().has_local || !scheduler.renderer().has_shared {
        return false;
    }

    if output_count(scheduler) > grouped_output_threshold(config) {
        return false;
    }

    // Try GROUPTOP on axes 0..3 with size 16; first one wins.
    for axis in 0..3 {
        if apply_opt(scheduler, &Opt::grouptop(axis, 16), true).is_ok() {
            return true;
        }
    }
    false
}

/// One wave per output row, for reductions with many rows.
///
/// [`try_grouped_reduction`] declines a kernel whose output count exceeds its
/// threshold, so a row reduce with thousands of rows falls through to the
/// generic path and ends up with one *thread* per row: the reduce stays a
/// serial loop and neighbouring lanes address memory a whole row apart, which
/// on a 128-byte sector is a near-total waste of every fetch.
///
/// Splitting a wave off the reduce axis instead makes the lanes of one wave
/// walk one row together. The gate is a memory-layout question, not a shape
/// one: the axis must be addressed with stride 1 by some buffer (so the lanes
/// of a wave are contiguous), and long enough after the split to keep a serial
/// loop over whole waves ([`MIN_WARP_REDUCE`]). The output axes stay in the
/// grid, so the block is exactly one wave and the two-stage reduction shares a
/// single wave's worth of scratch.
///
/// The trailing UNROLL is what turns the per-lane loads into a strided burst
/// the coalescer can merge across the wave; without it each iteration issues
/// one narrow load.
pub fn try_warp_row_reduction(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    use tracing::debug;

    if !scheduler.renderer().has_local || !scheduler.renderer().has_shared || config.disable_locals {
        return false;
    }
    // Few outputs: `try_grouped_reduction` owns that case, and its narrower
    // split leaves more of the reduce in the grid where the parallelism is.
    if output_count(scheduler) <= grouped_output_threshold(config) {
        return false;
    }

    let wave = scheduler.renderer().wave_size();
    let accesses = buffer_accesses(scheduler.bufs());
    let rngs = scheduler.rngs();
    // A wave per row spends the whole reduce on one output element, which only
    // pays when every output element reads its own data. An output axis some
    // buffer ignores is a reuse axis — a matmul's N, a projection's output
    // column — and amortizing the shared operand over it is worth more than the
    // coalescing.
    if !scheduler
        .axes_of(&[AxisType::Global, AxisType::Weak])
        .iter()
        .all(|&a| addressed_everywhere(&accesses, &rngs[a]))
    {
        return false;
    }

    // Innermost first: the last reduce axis is the one a row-major layout makes
    // contiguous, and grouping it leaves the outer reduces as plain loops.
    let axis = scheduler.axes_of(&[AxisType::Reduce]).into_iter().enumerate().rev().find(|&(_, axis)| {
        const_extent(&rngs[axis]).is_some_and(|extent| extent >= MIN_WARP_REDUCE && extent.is_multiple_of(wave))
            && (1..=SECTOR_BYTES).contains(&lane_span_bytes(&accesses, &rngs[axis]))
    });
    let Some((logical_axis, _)) = axis else { return false };

    let mut trial = scheduler.clone();
    if apply_opt(&mut trial, &Opt::group(logical_axis, wave), true).is_err() {
        return false;
    }

    // Each lane now walks its row in strides of `wave`; unrolling the remainder
    // issues the loads of several strides together, so the wave covers one
    // contiguous run per instruction instead of one element per lane.
    if let Some(remaining) = trial.unrollable_dims().last().copied()
        && const_extent(&trial.rngs()[remaining]).is_some_and(|extent| extent.is_multiple_of(DEFAULT_UPCAST_FACTOR))
    {
        let logical = trial.unrollable_dims().len() - 1;
        let _ = apply_opt(&mut trial, &Opt::unroll(logical, DEFAULT_UPCAST_FACTOR), true);
    }

    debug!(logical_axis, wave, "try_warp_row_reduction: applied");
    *scheduler = trial;
    true
}

/// Apply matmul-specific 2D output tiling (register blocking).
///
/// For matmul `C[M,N] = A[M,K] @ B[K,N]`, this creates a tile of output elements
/// that are computed together, amortizing memory loads across multiple outputs.
///
/// Achieves 8×8 register blocking with 64 scalar accumulators by applying UPCAST
/// to both M and N output axes:
/// - UPCAST M by up to 8 → 8 rows of output
/// - UPCAST N by up to 8 → 8 cols of output → up to 8×8 = 64 outputs
///
/// The devectorize pass (no_vectorized_alu) converts these to independent scalar
/// accumulators via MulAcc splitting.
///
/// Tile sizes are chosen flexibly based on divisibility: tries 8, 7, 6, 5, 4 in order.
pub fn apply_matmul_tiling(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    use tracing::debug;

    // Only apply to matmul patterns
    if !has_matmul_pattern(scheduler) {
        return false;
    }

    // Skip if output_upcast is disabled in config
    if !config.output_upcast {
        debug!("apply_matmul_tiling: skipped (output_upcast disabled)");
        return false;
    }

    // Output axes are GLOBAL/LOCAL/LOOP. After the OUTER→LOOP migration,
    // matmul output axes arrive as Loop, so no Outer arm is needed.
    let output_axes = scheduler.axes_of(&[AxisType::Global, AxisType::Local, AxisType::Weak]);
    debug!(output_axes = ?output_axes, "apply_matmul_tiling: output axes");

    // Need at least 2 output axes for 2D tiling
    if output_axes.len() < 2 {
        debug!("apply_matmul_tiling: not enough output axes (need 2)");
        return false;
    }

    // Upcast factors in decreasing order of preference
    // Larger tiles = more register blocking = better memory amortization
    const UPCAST_FACTORS: [usize; 5] = [8, 7, 6, 5, 4];

    // Collect axes with their sizes
    let rngs = scheduler.rngs();
    let mut axes_with_sizes: Vec<(usize, usize)> = Vec::new();

    for &axis_idx in output_axes.iter().take(2) {
        if axis_idx >= rngs.len() {
            continue;
        }
        if let Op::Range(ops::Range { end, .. }) = rngs[axis_idx].op()
            && let Op::Const(cv) = end.op()
            && let svod_ir::ConstValue::Int(size) = cv.0
            && size >= 4
        {
            axes_with_sizes.push((axis_idx, size as usize));
        }
    }

    if axes_with_sizes.len() < 2 {
        debug!(found = axes_with_sizes.len(), "apply_matmul_tiling: not enough output axes");
        return false;
    }

    // Apply UPCAST to each axis with the largest divisible factor
    let mut applied = false;
    for (axis_idx, size) in axes_with_sizes {
        // Find largest factor that divides size evenly
        if let Some(&factor) = UPCAST_FACTORS.iter().find(|&&f| size >= f && size % f == 0)
            && apply_opt(scheduler, &Opt::upcast(axis_idx, factor), true).is_ok()
        {
            debug!(axis = axis_idx, factor, size, "apply_matmul_tiling: applied UPCAST");
            applied = true;
        }
    }

    applied
}

/// Legacy function for compatibility - calls apply_matmul_tiling
pub fn apply_matmul_output_upcasting(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    apply_matmul_tiling(scheduler, config)
}

fn find_axis_by_axis_id(scheduler: &Scheduler, axis_id: AxisId) -> Option<usize> {
    scheduler.rngs().iter().enumerate().find_map(|(i, rng)| {
        if let Op::Range(ops::Range { axis_id: id, .. }) = rng.op()
            && id == &axis_id
        {
            return Some(i);
        }
        None
    })
}

/// The matvec fast path's tiling on one device: `block` rows per workgroup,
/// `lanes` threads splitting each row's reduce, `rows` accumulated per thread,
/// and the reduce `unroll` behind the lanes (`0`: none).
///
/// The AMD targets put one wave on each row group, so a row's reduce is one
/// coalesced sweep with the lanes unrolled to the vector access width; the
/// other targets keep tinygrad's 8x4x4 split. Each of the config's matvec
/// fields overrides its default.
struct MatvecTile {
    block: usize,
    lanes: usize,
    rows: usize,
    unroll: usize,
}

impl MatvecTile {
    fn resolve(config: &HeuristicsConfig, renderer: &Renderer, elem_bytes: usize) -> Self {
        let amd = matches!(
            renderer.device,
            RendererDevice::AmdRdna3 | RendererDevice::AmdRdna4 | RendererDevice::AmdCdna3 | RendererDevice::AmdCdna4
        );
        let (lanes, rows, unroll) =
            if amd { (renderer.wave_size(), 1, renderer.access_bytes() / elem_bytes.max(1)) } else { (8, 4, 0) };
        Self {
            block: config.matvec_blocksize.unwrap_or(4),
            lanes: config.threads_per_row.unwrap_or(lanes),
            rows: config.rows_per_thread.unwrap_or(rows),
            unroll,
        }
    }
}

/// The physical axis whose range carries `axis_id`, after opts have moved it.
fn axis_of(scheduler: &Scheduler, axis_id: &AxisId) -> Option<usize> {
    find_axis_by_axis_id(scheduler, axis_id.clone())
}

fn axis_id_of(scheduler: &Scheduler, axis: usize) -> Option<AxisId> {
    match scheduler.rngs().get(axis)?.op() {
        Op::Range(ops::Range { axis_id, .. }) => Some(axis_id.clone()),
        _ => None,
    }
}

/// Matvec fast path: `y[.., n] = sum_k x[.., k] * w[n, k]` with a contiguous
/// reduce in the first operand.
///
/// Output axes that only one operand indexes and that fit an upcast — a skinny
/// batch of rows through the same weights, a decoder step's tokens — are upcast
/// so each thread reads a weight row once for all of them. The reduce is then
/// split across `lanes` threads (GROUP), one row axis takes `block` rows per
/// workgroup (LOCAL, padded to the tile when that stays cheap) and `rows` per
/// thread (UPCAST), and the reduce left to each lane is unrolled to the vector
/// access width. GROUP precedes UNROLL: an unrolled reduce range no longer
/// belongs to its REDUCE. GROUP is best effort, the rest of the tile is not:
/// a declined split still beats the generic tail.
pub fn apply_matvec_fast_path(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    use tracing::debug;

    if !scheduler.renderer().has_local || !scheduler.renderer().has_shared || !config.matvec_enabled {
        return false;
    }
    let Some(reduceop) = scheduler.reduceop() else {
        return false;
    };
    let Some((left, right)) = matmul_operands(&reduceop) else {
        return false;
    };
    if scheduler.full_shape().len() < 2 {
        return false;
    }
    let (left, right) = (left.unwrap_cast(), right.unwrap_cast());
    let (idx0_src, idx1_src) = match (left.op(), right.op()) {
        (Op::Index(ops::Index { indices: i0, .. }), Op::Index(ops::Index { indices: i1, .. })) => {
            match (i0.first(), i1.first()) {
                (Some(i0), Some(i1)) => (i0.get_idx(), i1.get_idx()),
                _ => return false,
            }
        }
        _ => return false,
    };
    let tile = MatvecTile::resolve(config, scheduler.renderer(), left.dtype().bytes());
    if tile.block == 0 || tile.lanes == 0 || tile.rows == 0 || (tile.block <= 1 && tile.lanes <= 1 && tile.rows <= 1) {
        return false;
    }

    // The reduce the lanes split: a top-level ADD term of the first operand's
    // index, i.e. contiguous there.
    let reduce_ranges = scheduler.ranges_of(&[AxisType::Reduce]);
    let idx0_terms = idx0_src.split_uop(BinaryOp::Add);
    let Some(reduce_logical) =
        reduce_ranges.iter().position(|rng| idx0_terms.iter().any(|term| Arc::ptr_eq(term, rng)))
    else {
        return false;
    };
    if !matches!(reduce_ranges[reduce_logical].op(), Op::Range(ops::Range { end, .. }) if end.divides(tile.lanes as i64).is_some())
    {
        return false;
    }

    // Output axes only one operand indexes: small ones are upcast, one of the
    // rest is the row axis.
    let (ranges0, ranges1) = (idx0_src.ranges(), idx1_src.ranges());
    let exclusive =
        |rng: &Arc<UOp>| ranges0.iter().any(|r| Arc::ptr_eq(r, rng)) != ranges1.iter().any(|r| Arc::ptr_eq(r, rng));
    let full_shape = scheduler.full_shape();
    let upcast_max = scheduler.renderer().upcast_max;
    let (mut small, mut candidates) = (Vec::new(), Vec::new());
    for axis in scheduler.axes_of(&[AxisType::Global]) {
        let extent = usize::try_from(full_shape[axis]).unwrap_or(0);
        if extent > 1 && extent <= upcast_max && exclusive(&scheduler.rngs()[axis]) {
            small.push((axis, extent));
        } else {
            candidates.push(axis);
        }
    }
    // A second large exclusive axis makes this a matrix product, whose rows
    // would each re-read the weights: not this path.
    if candidates.iter().filter(|&&axis| exclusive(&scheduler.rngs()[axis])).count() > 1 {
        return false;
    }
    if candidates.is_empty() {
        // Nothing left for the rows: the smallest exclusive axis is them.
        candidates.extend(small.pop().map(|(axis, _)| axis));
    }
    let Some(small_ids) = small
        .iter()
        .map(|&(axis, extent)| axis_id_of(scheduler, axis).map(|id| (id, extent)))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };

    let row_tile = tile.block * tile.rows;
    for row_axis in candidates {
        let Some(&extent) = full_shape.get(row_axis) else { continue };
        let Ok(extent) = usize::try_from(extent) else { continue };
        if extent == 0 {
            continue;
        }
        // An axis the row tile does not divide is padded to it when that stays cheap.
        let padto = !extent.is_multiple_of(row_tile);
        if padto && padded_extent(extent, row_tile).is_none() {
            continue;
        }
        let Some(row_id) = axis_id_of(scheduler, row_axis) else { continue };

        let mut trial = scheduler.clone();
        let applied = (|| -> Result<(), OptError> {
            if padto {
                apply_opt(&mut trial, &Opt::padto(row_axis, row_tile), true)?;
            }
            for (id, extent) in &small_ids {
                let axis = axis_of(&trial, id).ok_or(OptError::MissingAxisParameter)?;
                apply_opt(&mut trial, &Opt::upcast(axis, *extent), true)?;
            }
            // Best effort: a GROUP the renderer declines (a nested reduce, a
            // grouped size past its shared memory) leaves the row tile worth
            // keeping, as a serial per-lane reduce.
            if tile.lanes > 1 {
                let _ = apply_opt(&mut trial, &Opt::group(reduce_logical, tile.lanes), true);
            }
            if tile.block > 1 {
                let axis = axis_of(&trial, &row_id).ok_or(OptError::MissingAxisParameter)?;
                apply_opt(&mut trial, &Opt::local(axis, tile.block), true)?;
            }
            if tile.rows > 1 {
                let axis = axis_of(&trial, &row_id).ok_or(OptError::MissingAxisParameter)?;
                apply_opt(&mut trial, &Opt::upcast(axis, tile.rows), true)?;
            }
            Ok(())
        })();
        if applied.is_err() || trial.applied_opts.is_empty() {
            continue;
        }

        // The reduce each lane still walks, unrolled to the access width or the
        // widest power of two below it that divides evenly. Best effort: the
        // tile stands without it.
        if tile.unroll > 1 {
            let unrollable = trial.unrollable_dims();
            let residual: Vec<usize> = unrollable
                .iter()
                .enumerate()
                .filter(|(_, axis)| {
                    matches!(trial.rngs()[**axis].op(), Op::Range(ops::Range { axis_type: AxisType::Reduce, .. }))
                })
                .map(|(logical, _)| logical)
                .collect();
            if let [logical] = residual[..]
                && let Op::Range(ops::Range { end, .. }) = trial.rngs()[unrollable[logical]].op()
            {
                let mut unroll = tile.unroll;
                while unroll > 1 && end.divides(unroll as i64).is_none() {
                    unroll /= 2;
                }
                if unroll > 1 {
                    let _ = apply_opt(&mut trial, &Opt::unroll(logical, unroll), true);
                }
            }
        }

        debug!(row_axis, ?small, tile.block, tile.lanes, tile.rows, tile.unroll, "apply_matvec_fast_path: applied");
        *scheduler = trial;
        return true;
    }

    false
}

/// CPU threading for parallelizable loop axes.
///
/// 1. Descending thread list: [32, 16, 12, 8, 6, 5, 4, 3, 2]
/// 2. Minimum work check: skip if `prod(full_shape) / 131072 < threads`
/// 3. Only LOOP axes (matmul output dims are Loop from rangeify)
pub fn apply_threading(scheduler: &mut Scheduler, max_threads: usize) -> bool {
    use tracing::debug;

    if !scheduler.renderer().has_threads || max_threads <= 1 {
        return false;
    }

    // Minimum work check: prod(full_shape) // (128 << 10) < threads → skip.
    // Use conservative upper-bound extents for symbolic range ends (vmax/const_factor)
    // so dynamic kernels don't underestimate work and collapse to tiny thread counts.
    let total_elements = estimate_total_elements(scheduler);

    const THREAD_LIST: [usize; 9] = [32, 16, 12, 8, 6, 5, 4, 3, 2];
    let counts =
        THREAD_LIST.into_iter().filter(|&threads| threads <= max_threads && total_elements / 131072 >= threads as i64);

    for threads in counts.clone() {
        // Only thread LOOP axes.
        let loop_axes = scheduler.axes_of(&[AxisType::Weak]);
        let mut thread_applied = false;
        for &axis_idx in &loop_axes {
            let rngs = scheduler.rngs();
            if axis_idx >= rngs.len() {
                continue;
            }
            if matches!(rngs[axis_idx].op(), Op::Range(ops::Range { end, .. }) if end.divides(threads as i64).is_some())
            {
                thread_applied = apply_opt(scheduler, &Opt::thread(axis_idx, threads), true).is_ok();
                if thread_applied {
                    debug!(axis = axis_idx, threads, "apply_threading: applied THREAD");
                }
                break;
            }
        }
        if thread_applied {
            return true;
        }
    }

    // No count divides any loop axis (a prime extent would run single-threaded):
    // pad the first axis whose padding stays cheap to the largest count.
    for threads in counts {
        let loop_axes = scheduler.axes_of(&[AxisType::Weak]);
        for &axis_idx in &loop_axes {
            let Some(size) = scheduler.rngs().get(axis_idx).and_then(const_extent) else { continue };
            let mut trial = scheduler.clone();
            if padded_extent(size, threads).is_some()
                && apply_opt(&mut trial, &Opt::padto(axis_idx, threads), true).is_ok()
                && apply_opt(&mut trial, &Opt::thread(axis_idx, threads), true).is_ok()
            {
                debug!(axis = axis_idx, threads, "apply_threading: applied PADTO + THREAD");
                *scheduler = trial;
                return true;
            }
        }
    }

    false
}

fn estimate_total_elements(scheduler: &Scheduler) -> i64 {
    let mut prod: i128 = 1;
    for rng in scheduler.rngs() {
        let extent = match rng.op() {
            Op::Range(ops::Range { end, .. }) => {
                if let Op::Const(cv) = end.op()
                    && let svod_ir::ConstValue::Int(sz) = cv.0
                    && sz > 0
                {
                    sz
                } else if let Some(vmax) = end.vmax().try_int() {
                    vmax.max(1)
                } else {
                    let cf = end.const_factor();
                    if cf > 0 { cf } else { 1 }
                }
            }
            _ => 1,
        };
        prod = (prod.saturating_mul(extent as i128)).min(i64::MAX as i128);
    }
    prod.max(1) as i64
}

// ============================================================================
// COMPLEX HEURISTICS
// ============================================================================

/// Heuristic upcasts based on stride analysis.
///
/// - Only enters the loop when `prod(output_shape[upcastable_dims]) >= 1024`
/// - Terminates when `upcast_size() >= 32`
/// - Uses factors `[3, 4]`
/// - Ranks by `(num_strides, sum_strides)` ascending (fewest strides = best)
/// - Excludes axes NOT stride-0 in any buffer (broadcast check)
pub fn apply_heuristic_upcasts(scheduler: &mut Scheduler) -> bool {
    use tracing::debug;

    let mut applied = false;
    let mut upcasted_axes: Vec<usize> = Vec::new();
    // Only a lane-parallel backend cares which width lands in one access: a CPU
    // kernel's UPCAST becomes a loop the vectorizer re-widths anyway, and
    // changing it there would move code generation for no measured reason.
    let prefer_vector_width = scheduler.renderer().has_local;

    loop {
        // While prod(output_shape[upcastable_dims]) >= 1024 and upcast_size() < 32:
        let upcastable = scheduler.upcastable_dims();
        if upcastable.is_empty() {
            break;
        }

        let output_shape_product: i64 = {
            let rngs = scheduler.rngs();
            upcastable
                .iter()
                .filter_map(|&idx| {
                    if idx < rngs.len()
                        && let Op::Range(ops::Range { end, .. }) = rngs[idx].op()
                        && let Op::Const(cv) = end.op()
                        && let svod_ir::ConstValue::Int(sz) = cv.0
                    {
                        Some(sz)
                    } else {
                        None
                    }
                })
                .product()
        };

        if output_shape_product < 1024 || scheduler.upcast_size() >= 32 {
            debug!(
                output_shape_product,
                upcast_size = scheduler.upcast_size(),
                "apply_heuristic_upcasts: terminating (threshold)"
            );
            break;
        }

        // Build choices: (num_strides, sum_strides, axis, vector_rank, amount)
        // for axis × upcast_amount in upcastable_dims × [3, 4].
        let mut choices: Vec<(usize, usize, usize, usize, usize)> = Vec::new();

        // One walk over the buffer indices records which of the existing
        // UPCAST/UNROLL ranges and candidate axes each node reaches, so every
        // per-axis question below is a set lookup.
        let rngs = scheduler.rngs();
        let candidates: Vec<usize> = upcastable.iter().copied().filter(|axis| !upcasted_axes.contains(axis)).collect();
        let upcast_and_unroll_ranges = scheduler.ranges_of(&[AxisType::Upcast, AxisType::Unroll]);
        let targets: Vec<Arc<UOp>> =
            upcast_and_unroll_ranges.iter().chain(candidates.iter().map(|&axis| &rngs[axis])).cloned().collect();
        let mut reach = reaching_each(&targets);
        let bufs = scheduler.bufs();
        let accesses = buffer_accesses(bufs);
        let indices: Vec<Arc<UOp>> = accesses.iter().map(|(idx, _)| idx.clone()).collect();

        // Stride-0 check: an axis must be NOT in some buffer's index backward
        // slice in which all existing UPCAST/UNROLL ranges ARE, so only those
        // buffers matter, as the ids of the targets their indices reach.
        let full_upcast_bufs: Vec<Vec<u64>> = bufs
            .iter()
            .filter_map(|buf| {
                let Op::Index(ops::Index { indices, .. }) = buf.op() else { return None };
                let mut reached = Vec::new();
                for idx in indices {
                    reached.extend(reach.get(idx).iter().map(|target| target.id));
                }
                upcast_and_unroll_ranges.iter().all(|range| reached.contains(&range.id)).then_some(reached)
            })
            .collect();

        for axis_idx in candidates {
            let rng = &rngs[axis_idx];
            if !full_upcast_bufs.iter().any(|reached| !reached.contains(&rng.id)) {
                continue;
            }

            let size = if let Op::Range(ops::Range { end, .. }) = rng.op()
                && let Op::Const(cv) = end.op()
                && let svod_ir::ConstValue::Int(sz) = cv.0
            {
                sz
            } else {
                continue;
            };
            let amounts: SmallVec<[usize; 2]> =
                [3, 4].into_iter().filter(|&amount| size % amount as i64 == 0).collect();
            if amounts.is_empty() {
                continue;
            }

            let (num_strides, sum_strides) =
                strides_of(&indices, rng, |idx| reach.get(idx).iter().any(|target| target.id == rng.id));
            let vectorizable = contiguous_element_bytes(&accesses, rng).filter(|_| prefer_vector_width);
            choices.extend(amounts.into_iter().map(|amount| {
                // A width that fills a machine vector on a contiguous axis wins
                // its axis: the upcast then lowers to one wide access instead
                // of `amount` narrow ones. Everything else keeps the ascending
                // order, so a non-contiguous axis is unaffected.
                let rank = usize::from(!vectorizable.is_some_and(|bytes| is_vector_width(amount * bytes)));
                (num_strides, sum_strides, axis_idx, rank, amount)
            }));
        }

        if choices.is_empty() {
            debug!("apply_heuristic_upcasts: no valid choices, breaking");
            break;
        }

        // Sort ascending by (num_strides, sum_strides) — fewest strides wins
        choices.sort();
        let (_, _, best_axis, _, best_amount) = choices[0];

        debug!(best_axis, best_amount, "apply_heuristic_upcasts: applying upcast");
        if apply_opt(scheduler, &Opt::upcast(best_axis, best_amount), true).is_ok() {
            upcasted_axes.push(best_axis);
            applied = true;
        } else {
            break;
        }
    }

    applied
}

/// Stride-ranked LOCAL workgroup configuration.
///
/// In a kernel that is nothing but its memory traffic, `lidx0` — the
/// fastest-moving thread index — goes to the axis every buffer keeps within one
/// [`SECTOR_BYTES`] ([`lane_span_bytes`]), so the lanes of a wave cover one
/// contiguous run of memory instead of landing one row apart. It only reorders:
/// the sizes are the ones this heuristic always picked, and a kernel with no
/// such axis — a transposing copy, strided on one side whichever axis leads —
/// is untouched, as is any kernel carrying a reduce. The rest keep the
/// expand-first ranking (stride-0
/// in some buffer = broadcast, then higher axis indices), with sizes from
/// [32, 16, 8, 4, 3, 2] for axis 0 and [16, 8, 4, 3, 2] for the others and a
/// cumulative LOCAL size ≤ 128. An axis none of the sizes divides (Whisper's
/// 51865 = 5·11·23·41 vocabulary) falls back to [`local_fallback`] instead of
/// running one thread per block.
pub fn apply_local_dims(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    if !scheduler.renderer().has_local || config.disable_locals {
        return false;
    }
    let budget = LOCAL_BUDGET.min(scheduler.renderer().local_max.unwrap_or(LOCAL_BUDGET));

    let eligible_axes = scheduler.axes_of(&[AxisType::Global, AxisType::Weak]);
    let full_shape = scheduler.full_shape();
    let accesses = buffer_accesses(scheduler.bufs());

    // Rank by (is_lane_axis, has_expand_pattern, axis_index) descending. An axis
    // no buffer moves with spans nothing and would make every lane read one
    // address, so a lane axis has to actually move (span >= 1).
    let mut candidates: Vec<(usize, bool, usize)> = Vec::new();
    for &axis in &eligible_axes {
        let rngs = scheduler.rngs();
        if axis >= rngs.len()
            || !matches!(rngs[axis].op(), Op::Range(ops::Range { end, .. }) if matches!(end.op(), Op::Const(..)))
        {
            continue;
        }
        candidates.push((lane_span_bytes(&accesses, &rngs[axis]), has_broadcast_pattern(scheduler, axis), axis));
    }
    // Only for a kernel that is nothing but its memory traffic. Where a reduce
    // loop sits inside the block the thread mapping is no longer the only thing
    // the block shape decides — a stencil's halo and the loop's own reuse both
    // ride on it — and the reordering measured worse.
    let lane_axis = scheduler
        .reduceop()
        .is_none()
        .then(|| {
            candidates
                .iter()
                .filter(|&&(span, ..)| (1..=SECTOR_BYTES).contains(&span))
                .min_by_key(|&&(span, _, axis)| (span, axis))
                .map(|&(_, _, axis)| axis)
        })
        .flatten();

    let mut local_axis_ranking: Vec<(bool, bool, usize)> =
        candidates.into_iter().map(|(_, is_expand, axis)| (Some(axis) == lane_axis, is_expand, axis)).collect();
    local_axis_ranking.sort_by(|a, b| b.cmp(a));

    // Collect LOCAL candidates with cumulative size constraint: (axis, size, padto).
    let mut to_local: Vec<(usize, usize, Option<usize>)> = Vec::new();
    for &(_, _, axis) in &local_axis_ranking {
        let cumulative_local: usize = to_local.iter().map(|(_, sz, _)| *sz).product::<usize>().max(1);
        let axis_size = full_shape[axis];
        if axis_size <= 0 {
            continue;
        }
        let axis_size = axis_size as usize;

        // Axis 0 gets [32, 16, 8, 4, 3, 2]; others get [16, 8, 4, 3, 2].
        let candidates: &[usize] = if axis == 0 { &[32, 16, 8, 4, 3, 2] } else { &[16, 8, 4, 3, 2] };

        let local_sz = candidates
            .iter()
            .copied()
            .find(|&x| axis_size.is_multiple_of(x) && cumulative_local * x <= LOCAL_BUDGET)
            .map(|sz| (sz, None))
            .or_else(|| {
                local_fallback(axis_size, cumulative_local, budget / cumulative_local, scheduler.renderer().wave_size())
            });

        if let Some((sz, padto)) = local_sz {
            to_local.push((axis, sz, padto));
        }
    }

    // Apply at most 3 LOCALs, the lane axis first: `lidx0` is handed to the
    // LOCAL range created first, so application order *is* the thread mapping.
    // Each target is re-found by its axis id, because a split renumbers the
    // list and the lane axis is applied out of index order.
    let mut to_apply: Vec<(usize, usize, Option<usize>)> = to_local.into_iter().take(3).collect();
    to_apply.sort_by_key(|&(axis, ..)| (Some(axis) != lane_axis, axis));
    let targets: Vec<(Option<AxisId>, usize, Option<usize>)> = to_apply
        .into_iter()
        .map(|(axis, local_sz, padto)| {
            let id = scheduler.rngs().get(axis).and_then(|rng| match rng.op() {
                Op::Range(ops::Range { axis_id, .. }) => Some(axis_id.clone()),
                _ => None,
            });
            (id, local_sz, padto)
        })
        .collect();

    let mut applied = false;
    for (axis_id, local_sz, padto) in targets {
        let Some(axis) = axis_id.and_then(|id| find_axis_by_axis_id(scheduler, id)) else { continue };
        let mut trial = scheduler.clone();
        if padto.is_some_and(|align| apply_opt(&mut trial, &Opt::padto(axis, align), true).is_err()) {
            continue;
        }
        if apply_opt(&mut trial, &Opt::local(axis, local_sz), true).is_ok() {
            *scheduler = trial;
            applied = true;
        }
    }
    applied
}

/// Factors a post-TC UPCAST may grow the warp tile by, best first.
///
/// Tinygrad's `[5, 4, 3, 2]` ladder extended to 8, the widest UPCAST a GPU
/// renderer accepts, so a lane can reach a square tile on an `m16n8` core.
const TC_GROWTH_FACTORS: [usize; 5] = [8, 5, 4, 3, 2];

/// Warp tiles a grid keeps before a wider per-warp tile stops paying for
/// itself. Doubling the tile halves the warps, and a grid that no longer covers
/// the device's multiprocessors loses more than the operand traffic the wider
/// tile saves. The heuristic cannot see the multiprocessor count, so this is a
/// floor and not a target: it only bites on outputs small enough that the full
/// register budget would leave a few dozen warps for the whole GPU.
const TC_MIN_WARP_TILES: usize = 192;

/// Post-TC growth `(m, n)` for the per-warp output tile, in tensor-core tiles.
///
/// [`tc::apply`](crate::optimizer::tc) leaves one warp computing the
/// instruction's own `dims.1 x dims.0` (M x N) tile with `tc.lane_tile()`
/// accumulators per lane. The post-TC UPCASTs multiply that tile; three limits
/// bound how far, and the tightest wins:
///
/// * `budget` — accumulators a lane may hold ([`TcTilePolicy::LaneBudget`]);
/// * `tiles / TC_MIN_WARP_TILES` — a wider tile means fewer warps, and a grid
///   that no longer covers the multiprocessors costs more than it saves;
/// * `k_tiles` — the accumulator is set up and written back once per K loop, so
///   a reduction with few steps cannot amortise a lane full of them.
///
/// `upcast_max` then caps each single UPCAST, which also keeps the recorded
/// [`Opt`] replayable.
///
/// Within those the growth is split so the warp tile comes out square: a
/// `Wm x Wn` tile reads `(Wm + Wn) * K` operand elements for `Wm * Wn * K`
/// MACs, and here the operands are read straight from global memory — there is
/// no shared-memory stage to amortise a lopsided tile — so the square tile
/// moves the least memory per flop. M is grown first, so the remainder left by
/// an M extent that does not divide is spent on N.
fn tc_warp_tile_growth(
    tc: &TensorCore,
    budget: usize,
    upcast_max: usize,
    tiles: usize,
    [m_tiles, n_tiles, k_tiles]: [usize; 3],
) -> (usize, usize) {
    let growth = (budget / tc.lane_tile()).min(tiles / TC_MIN_WARP_TILES).min(k_tiles).max(1);
    // Square tile: dims.1 * m == dims.0 * n with m * n == growth, so
    // m == sqrt(growth * dims.0 / dims.1), rounded up (M is the longer side of
    // an `m16n8` tile, so rounding down would spend the whole budget on N).
    let square = (growth * tc.dims.0).div_ceil(tc.dims.1);
    let m_cap = square.isqrt() + usize::from(square.isqrt().pow(2) < square);
    let grow = |extent: usize, cap: usize| {
        TC_GROWTH_FACTORS.into_iter().find(|&f| f <= cap && extent.is_multiple_of(f)).unwrap_or(1)
    };
    let m = grow(m_tiles, m_cap.min(upcast_max));
    (m, grow(n_tiles, (growth / m).min(upcast_max)))
}

/// Split `rngs[dim]` (`0` = N, `1` = M, then the extra growth axes) by `sz`,
/// recording the opt. `false` when the axis is no longer in the kernel.
fn tc_split(scheduler: &mut Scheduler, rngs: &mut [Arc<UOp>], dim: usize, sz: usize, new_type: AxisType) -> bool {
    let Some(idx) = scheduler.rngs().iter().position(|r| Arc::ptr_eq(r, &rngs[dim])) else { return false };
    let Ok((replaced, _)) = scheduler.shift_to(rngs[dim].clone(), sz, new_type, false, None) else { return false };
    scheduler.applied_opts.push(if new_type == AxisType::Upcast { Opt::upcast(idx, sz) } else { Opt::local(idx, sz) });
    rngs[dim] = replaced;
    true
}

/// Whether `rng`'s extent divides by `sz`.
fn divides(rng: &Arc<UOp>, sz: usize) -> bool {
    matches!(rng.op(), Op::Range(ops::Range { end, .. }) if end.divides(sz as i64).is_some())
}

/// Factors the fixed step grows the warp tile by, best first.
const TC_STEP_FACTORS: [usize; 4] = [5, 4, 3, 2];

/// The load instructions a lane issues for one tensor-core fragment of
/// `operand`: a fragment whose run along the reduce axis `k` is contiguous
/// arrives as vector accesses, a strided one costs an access per element.
fn fragment_cost(operand: &Arc<UOp>, k: &Arc<UOp>, elements: usize, access_bytes: usize) -> usize {
    let bufs: Vec<Arc<UOp>> =
        operand.backward_slice().into_iter().filter(|node| matches!(node.op(), Op::Index(..))).collect();
    match min_stride(&linearized_indices(&bufs), k) {
        Some(1) => elements.div_ceil((access_bytes / operand.dtype().base().bytes().max(1)).max(1)),
        _ => elements,
    }
}

/// Where the output of a matmul may grow once the tensor core has landed.
///
/// Growing along an output axis re-reads the fragments of the operands that
/// axis indexes and holds the fragment of the operand it does not: an M-like
/// axis re-reads A and reuses B, an N-like axis the other way round. So each
/// direction is worth what one fragment of the operand it reuses costs a lane
/// to load ([`fragment_cost`]), and the widest reuse is the growth that saves
/// the most memory traffic.
struct TcGrowth {
    /// What one A / B fragment costs a lane.
    costs: (usize, usize),
    /// Output axes outside the tensor core's own M and N — a convolution's
    /// second spatial axis, a batch — with the cost each one's growth reuses.
    extra: Vec<(Arc<UOp>, usize)>,
}

impl TcGrowth {
    /// The axis outside the tensor core's M and N whose growth holds a pricier
    /// fragment than N's does, if the kernel has one: where the step's second
    /// UPCAST would re-read that fragment once per lane tile, stacking warps
    /// along this axis leaves one fragment for the whole block.
    fn stacking_axis(&self, scheduler: &Scheduler) -> Option<Arc<UOp>> {
        self.extra
            .iter()
            .filter(|(rng, reuse)| *reuse > self.costs.0 && scheduler.rngs().iter().any(|r| Arc::ptr_eq(r, rng)))
            .max_by_key(|(_, reuse)| *reuse)
            .map(|(rng, _)| rng.clone())
    }

    /// The growth of `pattern` under `axis_choice` on a tensor core with
    /// `ept` elements per thread and `access_bytes`-wide vector accesses.
    fn of(pattern: &MatmulPattern, axis_choice: usize, ept: (usize, usize), access_bytes: usize) -> Self {
        let (n_axis, m_axis, k_axis) = &pattern.axis_choices[axis_choice];
        let costs = (
            fragment_cost(&pattern.in0, k_axis, ept.0, access_bytes),
            fragment_cost(&pattern.in1, k_axis, ept.1, access_bytes),
        );
        let others = |ranges: &[Arc<UOp>], taken: &Arc<UOp>, reuse: usize| {
            ranges.iter().filter(|r| !Arc::ptr_eq(r, taken)).map(|r| (r.clone(), reuse)).collect::<Vec<_>>()
        };
        let mut extra = others(&pattern.in0_ranges, m_axis, costs.1);
        extra.extend(others(&pattern.in1_ranges, n_axis, costs.0));
        Self { costs, extra }
    }
}

/// Tile the matmul left over by [`tc::apply`](crate::optimizer::tc) across
/// warps and blocks, following the renderer's [`TcTilePolicy`]. `axes` is the
/// `[N, M, K]` the tensor core returned, `growth` the directions the warp tile
/// may take.
fn apply_tc_tiling(scheduler: &mut Scheduler, growth: &TcGrowth, axes: &[Arc<UOp>; 3]) {
    let mut rngs = vec![axes[0].clone(), axes[1].clone()];
    let tc = scheduler.renderer().tensor_cores[scheduler.selected_tc_index.unwrap_or(0)].clone();

    match scheduler.renderer().tc_tile_policy() {
        TcTilePolicy::FixedStep => {
            // One step of the ladder on `rngs[dim]`: UPCAST by [5,4,3,2], LOCAL by [4,2].
            let step = |scheduler: &mut Scheduler, rngs: &mut Vec<Arc<UOp>>, dim: usize, new_type| {
                let factors: &[usize] = if new_type == AxisType::Upcast { &TC_STEP_FACTORS } else { &[4, 2] };
                if let Some(&sz) = factors.iter().find(|&&sz| divides(&rngs[dim], sz)) {
                    tc_split(scheduler, rngs, dim, sz, new_type);
                }
            };
            // UPCAST M (dim=1), as the step has always started.
            step(scheduler, &mut rngs, 1, AxisType::Upcast);

            // The rest of the growth goes to the axis that holds the pricier
            // operand fragment. An axis outside the tensor core's M and N takes
            // it into the block: the warps of a block then share that fragment
            // through one cache line each, where UPCASTing N would have every
            // lane re-read it once per tile it holds. This is what a
            // channels-last convolution's second spatial axis does against a
            // weight the reduce strides over — with N's own fragment cheap,
            // there is nothing for a wider lane tile to save.
            match growth.stacking_axis(scheduler).filter(|_| scheduler.renderer().has_local) {
                Some(stack) => {
                    rngs.push(stack);
                    step(scheduler, &mut rngs, 2, AxisType::Local);
                }
                // UPCAST N (dim=0) with factors [5,4,3,2], then LOCAL N with [4,2].
                None => {
                    step(scheduler, &mut rngs, 0, AxisType::Upcast);
                    if scheduler.renderer().has_local {
                        step(scheduler, &mut rngs, 0, AxisType::Local);
                    }
                }
            }
        }
        TcTilePolicy::LaneBudget { accum_max } => {
            let (m_grow, n_grow) = tc_warp_tile_growth(
                &tc,
                accum_max,
                scheduler.renderer().upcast_max,
                extent_product(scheduler, &[AxisType::Global]),
                [const_extent(&rngs[1]).unwrap_or(1), const_extent(&rngs[0]).unwrap_or(1), reduce_depth(scheduler)],
            );
            for (dim, sz) in [(1usize, m_grow), (0, n_grow)] {
                if sz > 1 {
                    tc_split(scheduler, &mut rngs, dim, sz, AxisType::Upcast);
                }
            }

            // Stack warps into a block only up to one wave: the fragments come
            // straight from global memory, so past the wave a block of several
            // warps shares nothing and the split only coarsens the grid. A
            // tensor core narrower than the wave (Intel Xe issues its DPAS
            // across 8 lanes) still needs its warps stacked to fill one.
            let per_block = scheduler.renderer().wave_size() / tc.threads.max(1);
            if scheduler.renderer().has_local && per_block > 1 && divides(&rngs[0], per_block) {
                tc_split(scheduler, &mut rngs, 0, per_block, AxisType::Local);
            }
        }
    }
}

/// Tensor core optimization for matmul patterns.
///
/// - Guard: skip when >1 reduce axis under [`TcOpt::Strict`]
/// - Apply TC opts via tc::apply, capturing returned axes `[N, M, K]`
/// - Post-TC: tile across warps and blocks ([`apply_tc_tiling`])
pub fn try_tensor_cores(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    use crate::optimizer::config::TcUsage;
    use crate::optimizer::tc;

    if config.tc_enabled == TcUsage::Disabled {
        return false;
    }
    if scheduler.renderer().tensor_cores.is_empty() {
        return false;
    }

    // Strict keeps tinygrad's TC_OPT=0 rule: one reduce axis only. The default
    // Relaxed level lets `tc::apply` pick a divisible reduce axis and leave the
    // rest as loops, which is what a conv's (channels, taps) reduce needs.
    let reduce_count = scheduler.axes_of(&[AxisType::GroupReduce, AxisType::Reduce]).len();
    if reduce_count != 1 && config.tc_opt == TcOpt::Strict {
        return false;
    }

    let pattern = match tc::detect_matmul(scheduler) {
        Ok(Some(pattern)) => pattern,
        _ => return false,
    };

    // The WMMA needs clean M/N *output* ranges: a tensor core tiles the matmul's
    // own M/N/K, splitting M/N into Warp/Local/Upcast. If the matmul output is
    // consumed by a downstream reduce (e.g. `min_over_K(x@cᵀ)`), that output axis
    // is itself a Reduce axis — tiling it makes the downstream reduce span the
    // tensor-core Warp/Local axes and share the matmul's reduce loops, so one
    // physical loop ends up closed by two ENDs (invalid LLVM phi). Decline TC in
    // that case and let the generic reduce path handle the fused kernel.
    let output_is_reduce = pattern
        .in0_ranges
        .iter()
        .chain(pattern.in1_ranges.iter())
        .any(|r| matches!(r.op(), Op::Range(ops::Range { axis_type: AxisType::Reduce, .. })));
    if output_is_reduce {
        tracing::debug!(
            "try_tensor_cores: matmul output axis is a reduce axis (fused reduce-after-matmul); skipping TC"
        );
        return false;
    }

    let axis_choice_count = pattern.axis_choices.len();

    // Take the choices in increasing padded work, and at equal work the deeper
    // reduce first ([`tc::axis_choice_rank`]). The detection order is the axes'
    // own, which for a conv offers the 3-wide tap as K before the channels;
    // applying whichever of those happens to come first pads the taps to the
    // core's K edge and leaves the channels as a scalar loop. Sorting is stable,
    // so choices the rank cannot compare keep that detection order.
    let order = {
        let renderer = scheduler.renderer();
        let rank: Vec<_> = (0..axis_choice_count)
            .map(|choice| tc::axis_choice_rank(&pattern, renderer, config.tc_select.as_i32(), choice))
            .collect();
        let mut order: Vec<usize> = (0..axis_choice_count).collect();
        order.sort_by(|&a, &b| match (rank[a], rank[b]) {
            (Some((pad_a, k_a)), Some((pad_b, k_b))) => pad_a.total_cmp(&pad_b).then(k_b.cmp(&k_a)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        });
        order
    };

    let mut rejections = Vec::new();

    for axis_choice in order {
        // Clone the scheduler for trial - if this axis choice fails, no partial mutations.
        let mut trial = scheduler.clone();
        let tc_result = tc::apply_with_axis_choice(
            &mut trial,
            config.tc_select.as_i32(),
            config.tc_opt.as_usize(),
            config.tc_enabled.as_usize(),
            Some(axis_choice),
        );

        let axes = match tc_result {
            Ok(axes) => axes,
            Err(err) => {
                let err_msg = err.to_string();
                tracing::debug!(axis_choice, reason = %err_msg, "try_tensor_cores: axis choice rejected");
                rejections.push((axis_choice, err_msg));
                continue;
            }
        };

        // Record the TC opt with explicit axis choice.
        let opt = Opt::tc(
            Some(axis_choice),
            config.tc_select.as_i32(),
            config.tc_opt.as_usize(),
            config.tc_enabled.as_usize(),
        );
        trial.applied_opts.push(opt);

        let (ept, access_bytes) = {
            let renderer = trial.renderer();
            (renderer.tensor_cores[trial.selected_tc_index.unwrap_or(0)].elements_per_thread, renderer.access_bytes())
        };
        apply_tc_tiling(&mut trial, &TcGrowth::of(&pattern, axis_choice, (ept.0, ept.1), access_bytes), &axes);

        *scheduler = trial;
        return true;
    }

    tracing::debug!(?rejections, "try_tensor_cores: all axis choices rejected");
    false
}
