//! Hand-coded optimization heuristics for kernel optimization.
//!
//! Hand-coded heuristics give reasonable performance without auto-tuning.
//! Applies optimizations in order: TC → Image → GroupReduce → Upcasts → Unroll → Local → Thread.

use std::sync::Arc;

use smallvec::SmallVec;
use svod_ir::uop::{reaching, reaching_each};
use svod_ir::{AxisId, AxisType, BinaryOp, Op, TernaryOp, UOp};

use crate::optimizer::config::HeuristicsConfig;
use crate::optimizer::tc::matmul_operands;
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

/// Threads of a block are issued in warps of this many lanes on every renderer
/// with locals (CUDA, Metal SIMD-groups, RDNA wave32); a block whose size is
/// not a multiple leaves its last warp partly idle.
const WARP_LANES: usize = 32;

/// Block sizes a global axis may be padded to, best first.
const PAD_BLOCKS: [usize; 3] = [32, 16, 8];

/// `size` rounded up to a multiple of `align`, when the padded (masked) tail
/// adds at most 5% of extra work.
fn padded_extent(size: usize, align: usize) -> Option<usize> {
    let padded = size.div_ceil(align) * align;
    ((padded - size) * 20 <= size).then_some(padded)
}

/// The constant extent of a RANGE, if it has one.
fn const_extent(rng: &Arc<UOp>) -> Option<usize> {
    match rng.op() {
        Op::Range(ops::Range { end, .. }) => match end.op() {
            Op::Const(cv) => match cv.0 {
                svod_ir::ConstValue::Int(size) if size > 0 => Some(size as usize),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

/// LOCAL size for a global axis none of the standard sizes divides, with the
/// PADTO alignment it needs first.
///
/// Two options compete on the fraction of lanes that do useful work in a
/// block of `cumulative * threads` (the block occupies whole warps, and
/// padded elements are computed then masked): the largest divisor of `size`
/// within `budget`, and the largest of [`PAD_BLOCKS`] whose padding stays
/// cheap ([`padded_extent`]). Ties keep the exact divisor.
fn local_fallback(size: usize, cumulative: usize, budget: usize) -> Option<(usize, Option<usize>)> {
    let lane_efficiency = |threads: usize, useful: usize, total: usize| {
        let block = cumulative * threads;
        block as f64 / (block.div_ceil(WARP_LANES) * WARP_LANES) as f64 * useful as f64 / total as f64
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

    // 3. Grouped reduction
    try_grouped_reduction(scheduler, config);

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
                // c.op is Ops.MUL and one side is rng and other is CONST
                if Arc::ptr_eq(lhs, target_rng)
                    && let Op::Const(cv) = rhs.op()
                    && let svod_ir::ConstValue::Int(v) = cv.0
                {
                    sum_strides += v as usize;
                } else if Arc::ptr_eq(rhs, target_rng)
                    && let Op::Const(cv) = lhs.op()
                    && let svod_ir::ConstValue::Int(v) = cv.0
                {
                    sum_strides += v as usize;
                }
            }
        }
    }
    (num_strides, sum_strides)
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

/// Grouped reduction for small output dimensions.
///
/// When the product of upcastable output dimensions is small (<= 2048,
/// or 240 when local selection is disabled), apply GROUPTOP on output axes to enable
/// local reduction.
pub fn try_grouped_reduction(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    if !scheduler.renderer().has_local || !scheduler.renderer().has_shared {
        return false;
    }

    // prod(output_shape[i] for i in upcastable_dims) <= threshold
    let upcastable = scheduler.upcastable_dims();
    let full_shape = scheduler.full_shape();
    let group_for_reduces: i64 = upcastable.iter().map(|&i| full_shape.get(i).copied().unwrap_or(1)).product();

    let threshold: i64 = if config.disable_locals { 240 } else { 2048 };
    if group_for_reduces > threshold {
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

/// Matvec fast-path.
///
/// Applies `GROUP` on the reduce axis and `LOCAL`/`UPCAST` on one global output
/// axis when the index structure matches matrix-vector style access.
pub fn apply_matvec_fast_path(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    use tracing::debug;

    let block_size = config.matvec_blocksize;
    let threads_per_row = config.threads_per_row;
    let rows_per_thread = config.rows_per_thread;

    if !scheduler.renderer().has_local
        || !scheduler.renderer().has_shared
        || !config.matvec_enabled
        || (block_size <= 1 && threads_per_row <= 1 && rows_per_thread <= 1)
    {
        return false;
    }

    if block_size == 0 || threads_per_row == 0 || rows_per_thread == 0 {
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
            let Some(i0) = i0.first() else {
                return false;
            };
            let Some(i1) = i1.first() else {
                return false;
            };
            (i0.get_idx(), i1.get_idx())
        }
        _ => return false,
    };

    let Some(first_reduce_rng) = scheduler.ranges_of(&[AxisType::Reduce]).first().cloned() else {
        return false;
    };

    // 1) idx0 must contain the first reduce range as a top-level ADD term.
    // 2) idx1 must include all ranges used by idx0.
    let idx0_has_first_reduce = idx0_src.split_uop(BinaryOp::Add).iter().any(|u| Arc::ptr_eq(u, &first_reduce_rng));
    if !idx0_has_first_reduce {
        return false;
    }

    let idx1_ranges = idx1_src.ranges();
    if !idx0_src.ranges().iter().all(|r| idx1_ranges.iter().any(|cand| Arc::ptr_eq(cand, r))) {
        return false;
    }

    if !matches!(first_reduce_rng.op(), Op::Range(ops::Range { end, .. }) if end.divides(threads_per_row as i64).is_some())
    {
        return false;
    }

    let Some(row_tile) = block_size.checked_mul(rows_per_thread) else {
        return false;
    };
    if row_tile == 0 {
        return false;
    }

    let full_shape = scheduler.full_shape();
    for global_idx in scheduler.axes_of(&[AxisType::Global]) {
        let Some(&global_dim) = full_shape.get(global_idx) else {
            continue;
        };
        if global_dim <= 0 {
            continue;
        }
        // An axis the row tile does not divide is padded to it when that stays cheap.
        let global_dim = global_dim as usize;
        let padto = !global_dim.is_multiple_of(row_tile);
        if padto && padded_extent(global_dim, row_tile).is_none() {
            continue;
        }

        let mut trial = scheduler.clone();
        if padto && apply_opt(&mut trial, &Opt::padto(global_idx, row_tile), true).is_err() {
            continue;
        }

        // GROUP is best-effort in this fast path.
        if threads_per_row > 1 {
            let _ = apply_opt(&mut trial, &Opt::group(0, threads_per_row), true);
        }

        let mut current_axis = global_idx;
        let axis_id = trial.rngs().get(current_axis).and_then(|rng| {
            if let Op::Range(ops::Range { axis_id, .. }) = rng.op() { Some(axis_id.clone()) } else { None }
        });

        if block_size > 1 {
            if apply_opt(&mut trial, &Opt::local(current_axis, block_size), true).is_err() {
                continue;
            }
            if let Some(axis_id) = axis_id {
                if let Some(updated_axis) = find_axis_by_axis_id(&trial, axis_id) {
                    current_axis = updated_axis;
                } else if rows_per_thread > 1 {
                    continue;
                }
            }
        }

        if rows_per_thread > 1 && apply_opt(&mut trial, &Opt::upcast(current_axis, rows_per_thread), true).is_err() {
            continue;
        }

        debug!(global_idx, block_size, threads_per_row, rows_per_thread, "apply_matvec_fast_path: applied");
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

        // Build choices: (num_strides, sum_strides, axis, upcast_amount)
        // for axis × upcast_amount in upcastable_dims × [3, 4].
        let mut choices: Vec<(usize, usize, usize, usize)> = Vec::new();

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
        let indices = linearized_indices(bufs);

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
            choices.extend(amounts.into_iter().map(|amount| (num_strides, sum_strides, axis_idx, amount)));
        }

        if choices.is_empty() {
            debug!("apply_heuristic_upcasts: no valid choices, breaking");
            break;
        }

        // Sort ascending by (num_strides, sum_strides) — fewest strides wins
        choices.sort();
        let (_, _, best_axis, best_amount) = choices[0];

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
/// Prioritizes expand axes (stride-0 in some buffer = broadcast) for LOCAL,
/// then higher axis indices. Tries sizes [32, 16, 8, 4, 3, 2] for axis 0
/// and [16, 8, 4, 3, 2] for others, with cumulative LOCAL size ≤ 128. An
/// axis none of them divides (Whisper's 51865 = 5·11·23·41 vocabulary) falls
/// back to [`local_fallback`] instead of running one thread per block.
pub fn apply_local_dims(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    if !scheduler.renderer().has_local || config.disable_locals {
        return false;
    }
    let budget = LOCAL_BUDGET.min(scheduler.renderer().local_max.unwrap_or(LOCAL_BUDGET));

    // Rank axes by (has_expand_pattern, axis_index) — expand axes (stride-0 in
    // some buffer = broadcast) first, then higher axis indices.
    let eligible_axes = scheduler.axes_of(&[AxisType::Global, AxisType::Weak]);
    let full_shape = scheduler.full_shape();

    let mut local_axis_ranking: Vec<(bool, usize)> = Vec::new();
    for &axis in &eligible_axes {
        let rngs = scheduler.rngs();
        if axis >= rngs.len() {
            continue;
        }
        // Only CONST-end ranges (no symbolic dims)
        if let Op::Range(ops::Range { end, .. }) = rngs[axis].op() {
            if !matches!(end.op(), Op::Const(..)) {
                continue;
            }
        } else {
            continue;
        }
        let is_expand = has_broadcast_pattern(scheduler, axis);
        local_axis_ranking.push((is_expand, axis));
    }

    // Sort descending by (is_expand, axis) — expand axes first, higher index first
    local_axis_ranking.sort_by(|a, b| b.cmp(a));

    // Collect LOCAL candidates with cumulative size constraint: (axis, size, padto).
    let mut to_local: Vec<(usize, usize, Option<usize>)> = Vec::new();
    for &(_, axis) in &local_axis_ranking {
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
            .or_else(|| local_fallback(axis_size, cumulative_local, budget / cumulative_local));

        if let Some((sz, padto)) = local_sz {
            to_local.push((axis, sz, padto));
        }
    }

    // Apply at most 3 LOCALs, sorted by axis (ascending)
    // Track deleted shapes: if local_sz == full_shape[axis], axis merges and shifts indices
    let mut to_apply: Vec<(usize, usize, Option<usize>)> = to_local.into_iter().take(3).collect();
    to_apply.sort();

    let mut applied = false;
    let mut deleted_shape = 0usize;
    for (axis, local_sz, padto) in to_apply {
        let adjusted_axis = axis - deleted_shape;
        let mut axis_size = full_shape[axis] as usize;
        let mut trial = scheduler.clone();
        if let Some(align) = padto {
            if apply_opt(&mut trial, &Opt::padto(adjusted_axis, align), true).is_err() {
                continue;
            }
            axis_size = axis_size.div_ceil(align) * align;
        }
        if apply_opt(&mut trial, &Opt::local(adjusted_axis, local_sz), true).is_ok() {
            *scheduler = trial;
            applied = true;
            if local_sz == axis_size {
                deleted_shape += 1;
            }
        }
    }
    applied
}

/// Tensor core optimization for matmul patterns.
///
/// - Guard: skip when >1 reduce axis unless tc_opt >= 1
/// - Apply TC opts via tc::apply, capturing returned axes `[N, M, K]`
/// - Post-TC: UPCAST M then N with `[5,4,3,2]`, LOCAL N with `[4,2]`
pub fn try_tensor_cores(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    use crate::optimizer::config::TcUsage;
    use crate::optimizer::tc;

    if config.tc_enabled == TcUsage::Disabled {
        return false;
    }
    if scheduler.renderer().tensor_cores.is_empty() {
        return false;
    }

    // Guard: require exactly one reduce axis unless TC_OPT >= 1.
    let reduce_count = scheduler.axes_of(&[AxisType::GroupReduce, AxisType::Reduce]).len();
    if reduce_count != 1 && config.tc_opt.as_usize() < 1 {
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

    let mut rejections = Vec::new();

    for axis_choice in 0..axis_choice_count {
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

        // Post-TC extras: UPCAST M/N then LOCAL N.
        {
            let mut tc_rngs = [axes[0].clone(), axes[1].clone()];

            // UPCAST M (dim=1) then N (dim=0) with factors [5,4,3,2]
            for tc_dim in [1usize, 0] {
                for &sz in &[5usize, 4, 3, 2] {
                    if matches!(tc_rngs[tc_dim].op(), Op::Range(ops::Range { end, .. }) if end.divides(sz as i64).is_some())
                    {
                        if let Some(rng_idx) = trial.rngs().iter().position(|r| Arc::ptr_eq(r, &tc_rngs[tc_dim]))
                            && let Ok((replaced, _)) =
                                trial.shift_to(tc_rngs[tc_dim].clone(), sz, AxisType::Upcast, false, None)
                        {
                            trial.applied_opts.push(Opt::upcast(rng_idx, sz));
                            tc_rngs[tc_dim] = replaced;
                        }
                        break;
                    }
                }
            }

            // LOCAL N (dim=0) with factors [4,2]
            if trial.renderer().has_local {
                for &sz in &[4usize, 2] {
                    if matches!(tc_rngs[0].op(), Op::Range(ops::Range { end, .. }) if end.divides(sz as i64).is_some())
                    {
                        if let Some(rng_idx) = trial.rngs().iter().position(|r| Arc::ptr_eq(r, &tc_rngs[0]))
                            && trial.shift_to(tc_rngs[0].clone(), sz, AxisType::Local, false, None).is_ok()
                        {
                            trial.applied_opts.push(Opt::local(rng_idx, sz));
                        }
                        break;
                    }
                }
            }
        }

        *scheduler = trial;
        return true;
    }

    tracing::debug!(?rejections, "try_tensor_cores: all axis choices rejected");
    false
}
