//! Tensor Core (TC) optimization - Hardware-accelerated matrix multiplication.
//!
//! Implements pattern matching, selection, swizzle, and application for tensor core ops.
//! Supports NVIDIA (WMMA), AMD (Matrix Cores), Intel Xe, and Apple Metal hardware.

use std::collections::HashMap;
use std::sync::Arc;

use smallvec::SmallVec;
use svod_ir::{AxisId, AxisType, BinaryOp, ConstValue, Op, ReduceOp, UOp, UOpKey, WmmaMetadata, WmmaUpcastAxes};

use crate::argsort;
use crate::optimizer::{
    Renderer, Scheduler,
    error::*,
    renderer::{SwizzleAxis, TcOpt, TensorCore},
};

// ============================================================================
// PATTERN MATCHING
// ============================================================================

/// Information about a detected matmul pattern. `in0`/`in1` are the tensor-core
/// operands: the MUL inputs with an exact integer widening peeled ([`tc_operand`]).
#[derive(Debug, Clone)]
pub struct MatmulPattern {
    pub reduce_op: Arc<UOp>,
    pub in0: Arc<UOp>,
    pub in1: Arc<UOp>,
    pub in0_ranges: Vec<Arc<UOp>>,
    pub in1_ranges: Vec<Arc<UOp>>,
    pub red_ranges: Vec<Arc<UOp>>,
    pub axis_choices: Vec<(Arc<UOp>, Arc<UOp>, Arc<UOp>)>,
}

/// Detect matmul pattern: REDUCE(ADD, MUL(in0, in1), ...reduce_ranges)
pub fn detect_matmul(scheduler: &Scheduler) -> Result<Option<MatmulPattern>, OptError> {
    let reduce_op = match scheduler.reduceop() {
        Some(op) => op,
        None => {
            tracing::debug!("no matmul: the kernel has no REDUCE");
            return Ok(None);
        }
    };

    let Some((in0, in1)) = matmul_operands(&reduce_op) else {
        tracing::debug!(
            src = reduce_op.op().sources().first().map(|s| AsRef::<str>::as_ref(s.op())),
            "no matmul: the REDUCE is not an ADD over a MUL, modulo casts"
        );
        return Ok(None);
    };
    let in0_all_ranges = get_ranges(&in0);
    let in1_all_ranges = get_ranges(&in1);

    let red_ranges: Vec<_> = if let Op::Reduce(svod_ir::ops::Reduce { ranges, .. }) = reduce_op.op() {
        ranges.iter().cloned().collect()
    } else {
        vec![]
    };

    // Find unique ranges (M and N dimensions)
    let in0_ranges: Vec<_> =
        in0_all_ranges.iter().filter(|r| !in1_all_ranges.iter().any(|r2| Arc::ptr_eq(r, r2))).cloned().collect();

    let in1_ranges: Vec<_> =
        in1_all_ranges.iter().filter(|r| !in0_all_ranges.iter().any(|r2| Arc::ptr_eq(r, r2))).cloned().collect();

    // Sort by axis_id descending
    let mut in0_ranges = in0_ranges;
    let mut in1_ranges = in1_ranges;
    let mut red_ranges = red_ranges;
    in0_ranges.sort_by_key(|r| std::cmp::Reverse(get_axis_id(r)));
    in1_ranges.sort_by_key(|r| std::cmp::Reverse(get_axis_id(r)));
    red_ranges.sort_by_key(|r| std::cmp::Reverse(get_axis_id(r)));

    // Generate all axis choices (N, M, K) using explicit loops to avoid closure ownership issues
    let mut axis_choices = Vec::with_capacity(in1_ranges.len() * in0_ranges.len() * red_ranges.len());
    for n in &in1_ranges {
        for m in &in0_ranges {
            for k in &red_ranges {
                axis_choices.push((n.clone(), m.clone(), k.clone()));
            }
        }
    }

    if axis_choices.is_empty() {
        // M and N are the ranges exclusive to one operand; K comes from the
        // REDUCE itself. A fused producer that makes both operands reach the
        // same ranges empties one of these and the matmul disappears.
        tracing::debug!(
            m = in0_ranges.len(),
            n = in1_ranges.len(),
            k = red_ranges.len(),
            "no matmul: no (N, M, K) triple, so an operand has no exclusive range"
        );
        return Ok(None);
    }

    Ok(Some(MatmulPattern { reduce_op, in0, in1, in0_ranges, in1_ranges, red_ranges, axis_choices }))
}

/// The tensor-core operands of a matmul-shaped `REDUCE(ADD, MUL(a, b))`, with
/// a cast above the MUL and exact integer widenings on `a`/`b` peeled.
pub(crate) fn matmul_operands(reduce: &Arc<UOp>) -> Option<(Arc<UOp>, Arc<UOp>)> {
    let Op::Reduce(svod_ir::ops::Reduce { reduce_op: ReduceOp::Add, src, .. }) = reduce.op() else { return None };
    let mul = src.unwrap_cast();
    let Op::Binary(BinaryOp::Mul, a, b) = mul.op() else { return None };
    Some((tc_operand(a), tc_operand(b)))
}

/// The tensor-core operand behind a MUL input. A value-preserving integer
/// widening (`int8 → int32`) is exact under a tensor core, which multiplies at
/// full width before accumulating, so the narrow source is matched instead.
pub(crate) fn tc_operand(operand: &Arc<UOp>) -> Arc<UOp> {
    match operand.op() {
        Op::Cast(svod_ir::ops::Cast { src, .. })
            if src.dtype().is_int()
                && operand.dtype().is_int()
                && svod_dtype::DType::can_safe_cast(src.dtype(), operand.dtype()) =>
        {
            src.clone()
        }
        _ => operand.clone(),
    }
}

fn get_ranges(uop: &Arc<UOp>) -> Vec<Arc<UOp>> {
    uop.backward_slice().into_iter().filter(|node| matches!(node.op(), Op::Range(..))).collect()
}

fn get_axis_id(range: &Arc<UOp>) -> AxisId {
    let Op::Range(svod_ir::ops::Range { axis_id, .. }) = range.op() else {
        unreachable!("range list contains non-RANGE")
    };
    axis_id.clone()
}

fn get_range_size(range: &Arc<UOp>) -> Option<i64> {
    if let Op::Range(svod_ir::ops::Range { end, .. }) = range.op()
        && let Op::Const(cv) = end.op()
        && let ConstValue::Int(size) = cv.0
    {
        return Some(size);
    }
    None
}

// ============================================================================
// SELECTION
// ============================================================================

/// Result of tensor core selection.
#[derive(Debug, Clone)]
pub struct TcSelection {
    pub tc_index: usize,
    pub axes: (Arc<UOp>, Arc<UOp>, Arc<UOp>),
}

/// Select appropriate tensor core for the given matmul pattern.
pub fn select_tensor_core(
    pattern: &MatmulPattern,
    renderer: &Renderer,
    tc_select: i32,
    axis_choice: usize,
) -> Result<Option<TcSelection>, OptError> {
    let tensor_cores = if tc_select == -1 {
        &renderer.tensor_cores[..]
    } else {
        let idx = tc_select as usize;
        if idx >= renderer.tensor_cores.len() {
            return ValidationFailedSnafu { op: "TC", reason: "tc_select index out of bounds" }.fail();
        }
        &renderer.tensor_cores[idx..idx + 1]
    };

    // Use `.scalar()` (returns Option) instead of `.base()` so Image dtypes
    // don't silently masquerade as Float32 (`base()` maps `Image` → Float32).
    // Reject Image dtypes outright — TCs operate on plain Scalar/Vector dtypes.
    let in0_dt = &pattern.in0.dtype();
    let in1_dt = &pattern.in1.dtype();
    let out_dt = &pattern.reduce_op.dtype();
    if in0_dt.is_image() || in1_dt.is_image() || out_dt.is_image() {
        return Ok(None);
    }
    let Some(mut in0_scalar) = in0_dt.scalar() else { return Ok(None) };
    let Some(mut in1_scalar) = in1_dt.scalar() else { return Ok(None) };
    let Some(out_scalar) = out_dt.scalar() else { return Ok(None) };

    // An FP8 input the renderer has no matrix core for reaches the WMMA as
    // f16, whether dtype emulation widens it after TC application or the
    // renderer converts it natively (RDNA4). Match it against the f16 core
    // rather than missing the TC opportunity or claiming a native FP8 one.
    let f16_core_for_fp8 = |scalar: svod_dtype::ScalarDType| {
        scalar.is_fp8()
            && !renderer.supports_matrix_dtype(scalar)
            && renderer.supports_dtype(svod_dtype::ScalarDType::Float16)
    };
    if f16_core_for_fp8(in0_scalar) {
        in0_scalar = svod_dtype::ScalarDType::Float16;
    }
    if f16_core_for_fp8(in1_scalar) {
        in1_scalar = svod_dtype::ScalarDType::Float16;
    }

    for (tc_idx, tc) in tensor_cores.iter().enumerate() {
        if tc.dtype_in.is_image() || tc.dtype_out.is_image() {
            continue;
        }
        let (Some(tc_in_scalar), Some(tc_out_scalar)) = (tc.dtype_in.scalar(), tc.dtype_out.scalar()) else {
            continue;
        };

        if in0_scalar != tc_in_scalar || in1_scalar != tc_in_scalar || out_scalar != tc_out_scalar {
            continue;
        }

        if axis_choice >= pattern.axis_choices.len() {
            continue;
        }

        let axes = pattern.axis_choices[axis_choice].clone();

        let actual_tc_idx = if tc_select == -1 {
            renderer.tensor_cores.iter().position(|t| std::ptr::eq(t, tc)).unwrap_or(tc_idx)
        } else {
            tc_select as usize
        };

        return Ok(Some(TcSelection { tc_index: actual_tc_idx, axes }));
    }

    Ok(None)
}

// ============================================================================
// SWIZZLE
// ============================================================================

/// Generate the base shape from tensor core opts.
pub fn base_shape(tc: &TensorCore) -> Vec<SwizzleAxis> {
    let reduce_count = (tc.dims.2 as f64).log2().floor() as usize;
    let mut ret = Vec::with_capacity(tc.opts.len() + reduce_count);
    let (mut u_cnt, mut l_cnt) = (0, 0);

    for opt in &tc.opts {
        match opt {
            TcOpt::Upcast(_) => {
                ret.push(SwizzleAxis::Upcast(u_cnt));
                u_cnt += 1;
            }
            TcOpt::Local(_) => {
                ret.push(SwizzleAxis::Local(l_cnt));
                l_cnt += 1;
            }
        }
    }
    for i in 0..reduce_count {
        ret.push(SwizzleAxis::Reduce(i));
    }
    ret
}

fn generate_remaps(tc: &TensorCore) -> Vec<HashMap<SwizzleAxis, SwizzleAxis>> {
    let local_count = tc.opts.iter().filter(|opt| opt.is_local()).count();
    let upcast_count = tc.opts.iter().filter(|opt| opt.is_upcast()).count();
    let reduce_count = (tc.dims.2 as f64).log2().floor() as usize;

    let mut fwd_shape = Vec::with_capacity(local_count + upcast_count + reduce_count);
    (0..local_count).for_each(|i| fwd_shape.push(SwizzleAxis::Local(i)));
    (0..upcast_count).for_each(|i| fwd_shape.push(SwizzleAxis::Upcast(i)));
    (0..reduce_count).for_each(|i| fwd_shape.push(SwizzleAxis::Reduce(i)));

    [&tc.swizzle.0, &tc.swizzle.1]
        .iter()
        .map(|part| {
            let mut flattened = Vec::new();
            flattened.extend_from_slice(&part.0);
            flattened.extend_from_slice(&part.1);
            flattened.extend_from_slice(&part.2);

            fwd_shape.iter().enumerate().filter_map(|(i, &key)| flattened.get(i).map(|&v| (key, v))).collect()
        })
        .collect()
}

/// Compute permutation indices for the given shape.
pub fn permutes_for_shape(tc: &TensorCore, shape: &[SwizzleAxis]) -> (Vec<usize>, Vec<usize>) {
    let remaps = generate_remaps(tc);
    let perms: Vec<Vec<usize>> = remaps
        .iter()
        .map(|remap| {
            shape
                .iter()
                .enumerate()
                .map(|(i, &axis)| remap.get(&axis).and_then(|&r| shape.iter().position(|&s| s == r)).unwrap_or(i))
                .collect()
        })
        .collect();

    (perms[0].clone(), perms[1].clone())
}

/// Get the number of reduce axes for the tensor core (log2 of K dimension).
pub fn get_reduce_axes_count(tc: &TensorCore) -> usize {
    (tc.dims.2 as f64).log2().floor() as usize
}

// ============================================================================
// APPLICATION
// ============================================================================

const NO_COMPATIBLE_TC: &str = "no compatible tensor core found";

/// Largest padded tail a tensor-core tile may add at `tc_opt = 2`, as a
/// percentage of the axis. Wide enough for a sequence that misses the tile by
/// a few rows (1500 -> 1504, 15 -> 16), too tight for a beam-width M to pay
/// for a full tile (5 -> 16), where a memory-bound kernel is the right answer.
/// `tc_opt = 3` keeps only PADTO's own limit.
const TC_PAD_BUDGET_PERCENT: usize = 25;

fn within_pad_budget(size: usize, tile: usize) -> bool {
    let padded = size.div_ceil(tile) * tile;
    (padded - size) * 100 <= size * TC_PAD_BUDGET_PERCENT
}

/// FLOP per operand byte, at the unpadded shape, past which the budget no
/// longer applies: such a kernel is compute-bound on every tensor-core GPU, so
/// even a tile padded well beyond the budget beats the scalar kernel it
/// displaces (a 20x20 conv output pads 20 -> 32 for 1.6x the MACs on a core
/// several times faster), where a beam-width GEMV at a few FLOP per byte only
/// pays for the padding.
const COMPUTE_BOUND_INTENSITY: f64 = 64.0;

/// `2·M·N·K / bytes(A + B + C)` over the pattern's whole M, N and K extents;
/// `None` when any of them is symbolic.
fn arithmetic_intensity(pattern: &MatmulPattern) -> Option<f64> {
    let extent =
        |ranges: &[Arc<UOp>]| ranges.iter().map(|r| get_range_size(r).map(|s| s as f64)).product::<Option<f64>>();
    let (m, n, k) = (extent(&pattern.in0_ranges)?, extent(&pattern.in1_ranges)?, extent(&pattern.red_ranges)?);
    let bytes = pattern.in0.dtype().bytes().max(pattern.in1.dtype().bytes()) as f64;
    Some(2.0 * m * n * k / (bytes * (m * k + n * k + m * n)))
}

/// How an axis choice ranks against its siblings: the MAC work it pays for
/// padding, as a multiple of what its unpadded shape would do, and then the
/// extent of the reduce it puts on K.
///
/// [`detect_matmul`] enumerates the `(N, M, K)` triples by descending axis id,
/// which for a convolution puts the innermost tap on K. Padding a 3-wide tap to
/// a 16-wide core edge is 5.3x the MACs, where reducing over the channels
/// instead divides exactly — so the order the axes happen to carry must not
/// decide which choice is applied. The reduce extent breaks a tie because the
/// accumulator is set up and written back once per K loop, and a short one
/// cannot amortise a lane full of them.
///
/// `None` when the choice has no compatible core or any of its three extents is
/// symbolic — such a choice cannot be applied at all.
pub fn axis_choice_rank(
    pattern: &MatmulPattern,
    renderer: &Renderer,
    tc_select: i32,
    axis_choice: usize,
) -> Option<(f64, i64)> {
    let selection = select_tensor_core(pattern, renderer, tc_select, axis_choice).ok()??;
    let tc = &renderer.tensor_cores[selection.tc_index];
    let (n_range, m_range, k_range) = &selection.axes;
    let padded: f64 = [(n_range, tc.dims.0), (m_range, tc.dims.1), (k_range, tc.dims.2)]
        .into_iter()
        .map(|(axis, edge)| {
            let size = get_range_size(axis)?;
            let size = usize::try_from(size).ok().filter(|&s| s > 0)?;
            Some((size.div_ceil(edge) * edge) as f64 / size as f64)
        })
        .product::<Option<f64>>()?;
    Some((padded, get_range_size(k_range)?))
}

fn apply_axis_choice_impl(
    scheduler: &mut Scheduler,
    pattern: &MatmulPattern,
    tc_select: i32,
    tc_opt: usize,
    use_tensor_cores: usize,
    axis_choice: usize,
) -> Result<[Arc<UOp>; 3], OptError> {
    let tc_selection = select_tensor_core(pattern, &scheduler.ren, tc_select, axis_choice)?
        .ok_or_else(|| ValidationFailedSnafu { op: "TC", reason: NO_COMPATIBLE_TC }.build())?;

    // Record which TC was actually picked; beam's `validate_limits` reads
    // this to compute the correct `tc_up` divisor when the renderer offers
    // multiple TC variants.
    scheduler.selected_tc_index = Some(tc_selection.tc_index);

    // Clone the TensorCore to avoid borrow conflicts when applying PADTO
    let tc = scheduler.ren.tensor_cores[tc_selection.tc_index].clone();
    let (n_range, m_range, k_range) = &tc_selection.axes;
    // Mutable axes array - may be updated after PADTO
    let mut axes = [n_range.clone(), m_range.clone(), k_range.clone()];

    // Padding check and application (tc_opt >= 2)
    // When tc_opt >= 2, we use PADTO to align non-divisible dimensions
    // instead of rejecting them outright.
    if tc_opt >= 2 {
        // Collect padding operations needed (can't mutate axes while iterating)
        let tc_dims = [tc.dims.0, tc.dims.1, tc.dims.2];
        let mut padding_ops: Vec<(usize, usize, usize)> = Vec::new(); // (axes_idx, scheduler_idx, tc_dim)
        let compute_bound =
            arithmetic_intensity(pattern).is_some_and(|flop_per_byte| flop_per_byte >= COMPUTE_BOUND_INTENSITY);

        for (i, (axis, &tc_dim)) in axes.iter().zip(&tc_dims).enumerate() {
            match get_range_size(axis) {
                Some(size) => {
                    if !(size as usize).is_multiple_of(tc_dim) {
                        // Padded rows are not free: they stream the same weights
                        // and multiply the MACs. A 5-row M on a 16-row core is
                        // 3.2x the work of a memory-bound GEMV, and BEAM times
                        // it as a win only because the tile it displaces is
                        // worse still. A compute-bound kernel is the other way
                        // round: the padded core still runs several times faster
                        // than the scalar loop, so the budget steps aside.
                        if tc_opt == 2 && !compute_bound && !within_pad_budget(size as usize, tc_dim) {
                            return ValidationFailedSnafu {
                                op: "TC",
                                reason: "padding to the tensor-core tile would add too much work",
                            }
                            .fail();
                        }
                        let axis_idx = scheduler.rngs().iter().position(|r| Arc::ptr_eq(r, axis)).ok_or_else(|| {
                            ValidationFailedSnafu { op: "TC", reason: "axis not found in scheduler ranges" }.build()
                        })?;
                        padding_ops.push((i, axis_idx, tc_dim));
                    }
                }
                // PADTO can't pad an unknown extent, and even a provably
                // divisible symbolic axis fires pathological tilings in the
                // divisibility-keyed heuristics — symbolic axes never TC.
                None => {
                    return ValidationFailedSnafu { op: "TC", reason: "symbolic dimension cannot use tensor cores" }
                        .fail();
                }
            }
        }

        // Apply padding operations sequentially. Propagate the inner OptError
        // directly rather than collapsing it — the PADTO failure detail (4x work
        // limit, unsafe ops, symbolic extent) is more actionable than a generic
        // "padding failed" message.
        for (axes_idx, scheduler_idx, tc_dim) in padding_ops {
            crate::optimizer::opts::apply_opt(scheduler, &crate::optimizer::Opt::padto(scheduler_idx, tc_dim), false)?;

            // Update axes to the new padded range (PADTO substitutes the old range)
            axes[axes_idx] = scheduler.rngs()[scheduler_idx].clone();
        }
    } else {
        // Without tc_opt >= 2, reject non-divisible dimensions. Symbolic dims
        // never TC: silently skipping the check lowers a tile loop over an
        // extent the tile may not cover.
        for (i, axis) in axes.iter().enumerate() {
            let tc_dim = match i {
                0 => tc.dims.0,
                1 => tc.dims.1,
                _ => tc.dims.2,
            };
            match get_range_size(axis) {
                Some(size) if (size as usize).is_multiple_of(tc_dim) => {}
                _ => {
                    return ValidationFailedSnafu { op: "TC", reason: "dimension not divisible by tensor core size" }
                        .fail();
                }
            }
        }
    }

    // Create WARP dimension
    let mut warp = UOp::range_axis(
        UOp::index_const(tc.threads as i64),
        AxisId::Renumbered(scheduler.maxarg() + 1),
        AxisType::Warp,
    );

    // Step 1: Apply TC opts via shift_to — splits each axis into (reduced, new_rng)
    let two = UOp::index_const(2);
    let mut ne: Vec<Arc<UOp>> = Vec::with_capacity(tc.opts.len());

    for opt in &tc.opts {
        match opt {
            TcOpt::Upcast(dim) => {
                let (replaced, new_rng) = scheduler.shift_to(axes[*dim].clone(), 2, AxisType::Upcast, false, None)?;
                axes[*dim] = replaced;
                ne.push(new_rng);
            }
            TcOpt::Local(dim) => {
                let warp_mod = warp
                    .try_mod(&two)
                    .map_err(|_| ValidationFailedSnafu { op: "TC", reason: "warp mod failed" }.build())?;
                let (replaced, new_rng) =
                    scheduler.shift_to(axes[*dim].clone(), 2, AxisType::Local, false, Some(warp_mod))?;
                axes[*dim] = replaced;
                warp = warp
                    .try_div(&two)
                    .map_err(|_| ValidationFailedSnafu { op: "TC", reason: "warp div failed" }.build())?;
                ne.push(new_rng);
            }
        }
    }

    // K-dimension UNROLL splits
    for (_idx, amt) in tc.get_reduce_axes() {
        let (replaced, new_rng) = scheduler.shift_to(axes[2].clone(), amt, AxisType::Unroll, false, None)?;
        axes[2] = replaced;
        ne.push(new_rng);
    }

    // Build WMMA UOp (if use_tensor_cores == 1)
    if use_tensor_cores == 1 {
        // Step 2: Re-extract sources from updated AST
        let updated_reduce = scheduler
            .reduceop()
            .ok_or_else(|| ValidationFailedSnafu { op: "TC", reason: "REDUCE missing after shift_to" }.build())?;

        // Validate that the REDUCE still contains MUL pattern after shift_to
        let (reduce_src, updated_reduce_ranges) = match updated_reduce.op() {
            Op::Reduce(svod_ir::ops::Reduce { src, ranges, .. }) => (src.clone(), ranges.clone()),
            _ => unreachable!(),
        };
        let mul = reduce_src.unwrap_cast();
        if !matches!(mul.op(), Op::Binary(BinaryOp::Mul, ..)) {
            return ValidationFailedSnafu { op: "TC", reason: "expected MUL inside REDUCE" }.fail();
        }

        // Step 3: Apply swizzle permutation via placeholders
        let bshape = base_shape(&tc);
        let (perm_a, perm_b) = permutes_for_shape(&tc, &bshape);
        let inv_a = argsort(&perm_a);
        let inv_b = argsort(&perm_b);

        // PLACEHOLDER is a temporary axis namespace, so ordinal identities are
        // deterministic and cannot collide with live kernel ranges.
        let placeholders: Vec<Arc<UOp>> = (0..ne.len())
            .map(|i| UOp::range_axis(UOp::index_const(2), AxisId::Renumbered(i), AxisType::Placeholder))
            .collect();

        // Substitute ne → placeholders in REDUCE subtree
        let subst_to_ph: HashMap<UOpKey, Arc<UOp>> =
            ne.iter().zip(&placeholders).map(|(n, ph)| (UOpKey(n.clone()), ph.clone())).collect();
        let ret = updated_reduce.substitute(&subst_to_ph);

        // Re-extract sources from substituted REDUCE
        let ret_src = match ret.op() {
            Op::Reduce(svod_ir::ops::Reduce { src, .. }) => src.clone(),
            _ => unreachable!(),
        };
        let ret_mul = ret_src.unwrap_cast();
        let (ret_a, ret_b) = match ret_mul.op() {
            Op::Binary(BinaryOp::Mul, a, b) => (tc_operand(a), tc_operand(b)),
            _ => unreachable!(),
        };

        // Substitute placeholders → permuted ne for each source
        let subst_a: HashMap<UOpKey, Arc<UOp>> =
            placeholders.iter().enumerate().map(|(i, ph)| (UOpKey(ph.clone()), ne[inv_a[i]].clone())).collect();
        let subst_b: HashMap<UOpKey, Arc<UOp>> =
            placeholders.iter().enumerate().map(|(i, ph)| (UOpKey(ph.clone()), ne[inv_b[i]].clone())).collect();

        // An fp8 operand on a part with no fp8 matrix core was matched to the
        // f16 core above; widen it here so the WMMA sees its own input dtype.
        // The renderer converts natively where it can and dtype emulation
        // decomposes the cast elsewhere.
        let widen = |src: Arc<UOp>| match (src.dtype().scalar(), tc.dtype_in.scalar()) {
            (Some(from), Some(to)) if from.is_fp8() && from != to => src.cast(svod_dtype::DType::Scalar(to)),
            _ => src,
        };
        let src_a = widen(ret_a.substitute(&subst_a));
        let src_b = widen(ret_b.substitute(&subst_b));

        // Step 4: Build tc_upcast_axes from ne ranges
        //
        // `ne` mirrors `tc.opts` order (upcast and local interleaved), with reduce
        // entries appended after `ne[tc.opts.len()..]`. We must filter by opt type
        // to extract only upcast entries, not assume positional layout.
        let upcast_ne: Vec<&Arc<UOp>> =
            tc.opts.iter().zip(ne.iter()).filter(|(opt, _)| opt.is_upcast()).map(|(_, rng)| rng).collect();
        let reduce_ne: Vec<&Arc<UOp>> = ne[tc.opts.len()..].iter().collect();

        // base_upcast_ne: reversed([reduce, upcast]) = [upcast_reversed, reduce_reversed]
        let mut base_upcast_ne: Vec<&Arc<UOp>> = Vec::new();
        base_upcast_ne.extend(&reduce_ne);
        base_upcast_ne.extend(&upcast_ne);
        base_upcast_ne.reverse();

        let base_upcast_axes: Vec<(AxisId, usize)> = base_upcast_ne
            .iter()
            .map(|rng| match rng.op() {
                Op::Range(svod_ir::ops::Range { axis_id, .. }) => (axis_id.clone(), 2),
                _ => unreachable!(),
            })
            .collect();

        // Slice by log2(elements_per_thread)
        let n_a = (tc.elements_per_thread.0 as f64).log2() as usize;
        let n_b = (tc.elements_per_thread.1 as f64).log2() as usize;
        let n_c = (tc.elements_per_thread.2 as f64).log2() as usize;
        let mut a_axes = base_upcast_axes[..n_a].to_vec();
        let mut b_axes = base_upcast_axes[..n_b].to_vec();
        let mut c_axes = base_upcast_axes[..n_c].to_vec();
        let all_input_axes: Vec<_> = a_axes.iter().chain(&b_axes).cloned().collect();
        for axes in [&mut a_axes, &mut b_axes, &mut c_axes] {
            for (axis, _) in &all_input_axes {
                if !axes.iter().any(|(existing, _)| existing == axis) {
                    axes.push((axis.clone(), 1));
                }
            }
        }

        // Step 5: Construct WMMA
        // Compute TC reduce axis IDs early (needed for metadata)
        let tc_reduce_aids: Vec<AxisId> = ne[tc.opts.len()..]
            .iter()
            .filter_map(|r| match r.op() {
                Op::Range(svod_ir::ops::Range { axis_id, .. }) => Some(axis_id.clone()),
                _ => None,
            })
            .collect();

        let metadata = WmmaMetadata {
            name: tc.wmma_name(),
            dims: tc.dims,
            dtype_in: tc.dtype_in.clone(),
            dtype_out: tc.dtype_out.clone(),
            device: scheduler.ren.device,
            threads: tc.threads,
            upcast_axes: Some(WmmaUpcastAxes { a: a_axes, b: b_axes, c: c_axes.clone() }),
            reduce_axes: tc_reduce_aids.clone(),
        };

        // The WMMA C/accumulator operand carries the full per-thread D-register
        // width (`elements_per_thread.2`, == prod(c_axes)), NOT a scalar — see
        // tinygrad postrange.py:300-303 which builds the zero accumulator as
        // `dtype_out.vec(elements_per_thread[2])`. The direct WMMA expander
        // preserves C unchanged before reconstructing the output coordinates,
        // so this width must already match the hardware accumulator fragment.
        let c_count = tc.elements_per_thread.2;
        let zero_scalar = if tc.dtype_out.is_float() {
            UOp::const_(tc.dtype_out.clone(), ConstValue::Float(0.0))
        } else {
            UOp::const_(tc.dtype_out.clone(), ConstValue::Int(0))
        };
        let zero_acc = zero_scalar.broadcast(c_count);
        let mut tc_uop = UOp::wmma(src_a, src_b, zero_acc, metadata);

        // Preserve only ranges reachable from the original REDUCE range
        // sources. Operand ranges in the WMMA backward slice are not residual
        // reductions (postrange.py `_apply_tc_opt`, lines 310-313).
        let mut extra: SmallVec<[Arc<UOp>; 4]> = UOp::sink(updated_reduce_ranges.into_vec())
            .toposort()
            .into_iter()
            .filter(|r| matches!(r.op(), Op::Range(svod_ir::ops::Range { axis_id, .. }) if !tc_reduce_aids.contains(axis_id)))
            .collect();
        // Deterministic nesting (outer = lowest axis_id); slice may list a range once.
        extra.sort_by_key(get_axis_id);
        extra.dedup_by_key(|r| get_axis_id(r));
        if !extra.is_empty() {
            tc_uop = tc_uop.reduce(extra, ReduceOp::Add);
        }

        // Substitute REDUCE → WMMA chain in the AST
        let mut subst_map = HashMap::new();
        subst_map.insert(UOpKey(updated_reduce), tc_uop);
        let new_ast = scheduler.ast().substitute(&subst_map);
        scheduler.set_ast(new_ast);
    }

    Ok(axes)
}

fn tc_reject_reason(err: &OptError) -> &'static str {
    match err {
        OptError::Spec { .. } => "tensor spec verification failed",
        OptError::ValidationFailed { reason, .. } => reason,
        OptError::InvalidArgType { .. } => "invalid argument type",
        OptError::AxisOutOfBounds { .. } => "axis out of bounds",
        OptError::DivisionError { .. } => "division constraint violated",
        OptError::SymbolicDivisionError { .. } => "symbolic divisibility constraint",
        OptError::ExpectedRangeOperation => "expected range operation",
        OptError::MissingAxisParameter => "missing axis parameter",
        OptError::UnsupportedFeature { .. } => "unsupported backend feature",
        OptError::MissingRendererCapabilities => "missing renderer capabilities",
        OptError::DeviceLimitExceeded { .. } => "device limit exceeded",
        OptError::BeamWorker { .. } => "BEAM worker failure",
    }
}

/// Apply tensor core optimization to the scheduler.
///
/// If `axis_choice` is provided, only that axis candidate is attempted.
/// Otherwise all axis candidates are tried in order until one succeeds.
pub fn apply_with_axis_choice(
    scheduler: &mut Scheduler,
    tc_select: i32,
    tc_opt: usize,
    use_tensor_cores: usize,
    axis_choice: Option<usize>,
) -> Result<[Arc<UOp>; 3], OptError> {
    if !scheduler.applied_opts.is_empty() {
        return ValidationFailedSnafu { op: "TC", reason: "tensor core opts must be first" }.fail();
    }
    if use_tensor_cores == 0 || use_tensor_cores > 2 {
        return ValidationFailedSnafu { op: "TC", reason: "use_tensor_cores must be 1 or 2" }.fail();
    }
    if tc_opt > 3 {
        return ValidationFailedSnafu { op: "TC", reason: "tc_opt must be 0, 1, 2, or 3" }.fail();
    }
    if tc_select < -1 {
        return ValidationFailedSnafu { op: "TC", reason: "tc_select must be >= -1" }.fail();
    }

    let pattern = detect_matmul(scheduler)?
        .ok_or_else(|| ValidationFailedSnafu { op: "TC", reason: "no matmul pattern detected" }.build())?;

    let choices: Vec<usize> = if let Some(choice) = axis_choice {
        if choice >= pattern.axis_choices.len() {
            return ValidationFailedSnafu { op: "TC", reason: "axis choice out of bounds" }.fail();
        }
        vec![choice]
    } else {
        (0..pattern.axis_choices.len()).collect()
    };

    let mut failures: Vec<(usize, &'static str)> = Vec::new();
    let tc_choices: Vec<i32> = if tc_select == -1 {
        (0..scheduler.ren.tensor_cores.len()).map(|idx| idx as i32).collect()
    } else {
        vec![tc_select]
    };
    let mut last_err: Option<OptError> = None;

    // Cap total trials to bound compile time when both axis_choices and
    // tensor_cores are large. 64 covers realistic combinations (>16 axes ×
    // >4 TC variants is exceedingly rare) without aborting useful searches.
    const TC_RETRY_BUDGET: usize = 64;
    let mut trials = 0usize;

    'outer: for choice in choices {
        for &tc_choice in &tc_choices {
            if trials >= TC_RETRY_BUDGET {
                tracing::debug!(
                    trials,
                    budget = TC_RETRY_BUDGET,
                    "tensor core retry budget exhausted; aborting search"
                );
                break 'outer;
            }
            trials += 1;

            let mut trial = scheduler.clone();
            match apply_axis_choice_impl(&mut trial, &pattern, tc_choice, tc_opt, use_tensor_cores, choice) {
                Ok(axes) => {
                    *scheduler = trial;
                    return Ok(axes);
                }
                Err(err) => {
                    let reason = tc_reject_reason(&err);
                    tracing::debug!(
                        axis_choice = choice,
                        tc_select = tc_choice,
                        reason,
                        error = %err,
                        "tensor core axis choice rejected"
                    );
                    failures.push((choice, reason));
                    // Every core but the matching one reports a dtype mismatch,
                    // which says nothing about why the matching core declined.
                    // Keep the first substantive rejection instead of the last.
                    if last_err.is_none() || reason != NO_COMPATIBLE_TC {
                        last_err = Some(err);
                    }
                }
            }
        }
    }

    tracing::debug!(requested_axis_choice = ?axis_choice, failures = ?failures, "tensor core optimization rejected");

    if let Some(err) = last_err {
        Err(err)
    } else {
        ValidationFailedSnafu { op: "TC", reason: NO_COMPATIBLE_TC }.fail()
    }
}

/// Apply tensor core optimization, auto-trying axis choices.
pub fn apply(
    scheduler: &mut Scheduler,
    tc_select: i32,
    tc_opt: usize,
    use_tensor_cores: usize,
) -> Result<[Arc<UOp>; 3], OptError> {
    apply_with_axis_choice(scheduler, tc_select, tc_opt, use_tensor_cores, None)
}

// ============================================================================
// MODULE SHIMS (backwards compatibility for tests)
// ============================================================================

/// Pattern matching functions (was opts::tc::matching).
pub mod matching {
    pub use super::{MatmulPattern, detect_matmul};
}

/// Selection functions (was opts::tc::selection).
pub mod selection {
    pub use super::{TcSelection, select_tensor_core};
}

/// Swizzle functions (was opts::tc::swizzle).
pub mod swizzle {
    pub use super::{base_shape, get_reduce_axes_count, permutes_for_shape};
}
