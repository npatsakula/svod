//! Consolidated pattern matchers for rangeify transformations.
//!
//! This module contains all pattern matchers used during scheduling/rangeify:
//! - Early cleanup rewrites (DETACH, CONTIGUOUS_BACKWARD removal)
//! - Movement op → STAGE conversion
//! - Buffer folding and removal
//! - Kernel splitting patterns (BUFFER → PARAM, AFTER handling)
//! - Codegen preparation (NOOP removal, INDEX linearization)
//! - Buffer limit enforcement
//!
//! Consolidated from: patterns.rs, codegen_patterns.rs, movement_patterns.rs,
//! split_patterns.rs, buffer_limits.rs

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use smallvec::SmallVec;
use svod_device::DeviceSpec;
use svod_dtype::{AddrSpace, DType};
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::properties::{RangesProperty, SoundVminVmaxProperty};
use svod_ir::{AxisId, AxisType, BinaryOp, BufferizeOpts, ConstValue, Op, ReduceOp, UOp, UOpKey, UnaryOp};

use crate::TypedPatternMatcher;
use crate::rangeify::transforms::{cast_to_dtype, get_range_size, partition_reduce_ranges};

pub use crate::devectorize::pm_add_loads;
use svod_ir::ops;

fn is_codegen_param(node: &Arc<UOp>) -> bool {
    matches!(node.op(), Op::Param(..))
        && node.tag().as_ref().is_some_and(|tags| tags.contains(&svod_ir::uop::canonical::TAG_CODEGEN_PARAM))
}

fn mark_codegen_param(node: Arc<UOp>) -> Arc<UOp> {
    let mut tags = node.tag().clone().unwrap_or_default();
    if !tags.contains(&svod_ir::uop::canonical::TAG_CODEGEN_PARAM) {
        tags.push(svod_ir::uop::canonical::TAG_CODEGEN_PARAM);
    }
    node.with_tag(tags)
}

// Forward declarations for types from other modules
use super::indexing::IndexingContext;
use super::indexing::no_range;
use super::indexing::ranges_equal;
use super::kernel::{LocalAddBufferContext, RangeifyBufferContext};
use super::kernel::{SplitReduceOpConfig, split_reduceop};
use super::transforms::transform_sources_with_bufferize;

// ============================================================================
// HELPER FUNCTIONS (private)
// ============================================================================

/// Check if a UOp has zero total size (any shape dimension is 0).
fn has_zero_size(uop: &Arc<UOp>) -> bool {
    match uop.shape() {
        Ok(Some(shape)) => shape.iter().any(|d| d.as_const() == Some(0)),
        _ => false,
    }
}

/// Ops whose buffers must materialize and therefore cannot be inlined.
pub fn is_always_run_op(op: &Op) -> bool {
    matches!(op, Op::Contiguous(..) | Op::Copy(..) | Op::Noop)
}

/// Element count of a shaped UOp, or `None` when a dimension is symbolic.
fn static_numel(uop: &Arc<UOp>) -> Option<usize> {
    uop.shape().ok().flatten()?.iter().try_fold(1usize, |count, dim| count.checked_mul(dim.as_const()?))
}

/// Whether a COPY over this movement op needs its source materialised.
///
/// Tinygrad `schedule/rangeify.py:150` asks for `r.numel() != r.base.numel()` — the
/// view resizes — or a source with no contiguous view offset, i.e. one a movement
/// op reorders. Morok has no view-offset model, so PERMUTE/FLIP anywhere in the
/// chain stands in for the second half; a symbolic extent resolves to `False`
/// exactly as `resolve(..., False)` does upstream.
fn copy_needs_contiguous(src: &Arc<UOp>) -> bool {
    if let (Some(moved), Some(base)) = (static_numel(src), static_numel(&src.base()))
        && moved != base
    {
        return true;
    }
    let mut node = src.clone();
    loop {
        match node.op() {
            Op::Permute(..) | Op::Flip(..) => return true,
            Op::Reshape(ops::Reshape { src, .. })
            | Op::Expand(ops::Expand { src, .. })
            | Op::Pad(ops::Pad { src, .. })
            | Op::Shrink(ops::Shrink { src, .. })
            | Op::Multi(ops::Multi { src, .. }) => node = src.clone(),
            _ => return false,
        }
    }
}

/// Whether an op is elementwise, i.e. tinygrad's `GroupOp.Elementwise`
/// (`uop/__init__.py:112`): ALU plus the two casts. A source that is elementwise
/// can be split off into its own kernel without changing what it computes, which
/// is what buffer-limit enforcement relies on.
pub fn is_elementwise(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) | Op::Cast(..) | Op::BitCast(..))
}

// ============================================================================
// EARLY CLEANUP PATTERNS
// ============================================================================

/// Both `Mul` operands are non-constant sized integers that `wide` (a sized
/// integer) holds exactly, so forming the product at `wide` cannot wrap. A
/// constant operand is excluded: `neg(x)` is `x * -1`, and a `-1` already
/// materialised in the narrow type (255 for uint8) must keep wrapping.
fn widens_int_product(a: &Arc<UOp>, b: &Arc<UOp>, wide: &DType) -> bool {
    let sized_int = |dtype: &DType| dtype.scalar().is_some_and(|s| s.is_signed() || s.is_unsigned());
    sized_int(wide)
        && [a, b].iter().all(|operand| {
            let dtype = operand.dtype();
            !matches!(operand.base().op(), Op::Const(..) | Op::VConst(..))
                && sized_int(&dtype)
                && dtype != *wide
                && DType::can_safe_cast(dtype, wide.clone())
        })
}

/// Pattern matcher for early cleanup rewrites during scheduling.
///
/// This handles schedule-specific cleanup:
/// - Integer products under a widening cast are formed at the wide type
/// - DETACH removal (gradient computation marker no longer needed)
/// - CONTIGUOUS_BACKWARD removal (gradient computation marker no longer needed)
/// - Zero-size tensor folding
pub fn early_rewrites() -> TypedPatternMatcher {
    crate::patterns! {
        // A widening integer cast of a product applies to the operands, so the
        // product is formed at the accumulator's width instead of wrapping in
        // the narrow type. That is what an integer accumulate dtype means
        // (ONNX MatMulInteger, tensor cores multiply at full width) and what C
        // integer promotion already gave the clang backend.
        Cast { src: Mul(a, b), dtype: wide } if widens_int_product(a, b, wide)
            => a.cast(wide.clone()).try_mul(&b.cast(wide.clone())).ok(),
        // mop_cleanup: merge adjacent untagged RESHAPEs.
        x @ Reshape { src: x2, new_shape } if x.tag().is_none() && x2.tag().is_none() && matches!(x2.op(), Op::Reshape(..)) => {
            let Op::Reshape(ops::Reshape { src, .. }) = x2.op() else { return None };
            Some(UOp::new(Op::Reshape(ops::Reshape { src: src.clone(), new_shape: new_shape.clone() }), x.dtype()))
        },
        Detach { src: x } => x.clone(),
        ContiguousBackward { src: x } => x.clone(),
        // A COPY transfers one contiguous range, so a source that is resized or
        // reordered by a movement op must be materialised first (tinygrad
        // `schedule/rangeify.py:149`). Without this the transfer is sized by the
        // base, not the moved view, and the destination is under-allocated.
        copy @ Copy { src, .. } if src.op().is_movement() && copy_needs_contiguous(src)
            => Some(copy.with_sources(vec![src.contiguous()])),
        // Same-device COPY is a no-op and returns its source verbatim, tag included
        // (tinygrad `schedule/rangeify.py:153`). The barrier role a tagged COPY used
        // to carry is covered by `is_always_run_op(Copy)`.
        Copy { src, device } if src.device_spec().as_ref() == Some(device)
            => src.clone(),
        // Reduce of zero-sized input → identity element.
        reduce @ Reduce { src: x, ranges: _, reduce_op: _, num_axes: _ }
            if has_zero_size(x) && !has_zero_size(reduce) => {
            let Op::Reduce(ops::Reduce { reduce_op, .. }) = reduce.op() else { return None };
            let identity = crate::symbolic::dce::reduce_identity(*reduce_op, reduce.dtype());
            let Op::Const(value) = identity.op() else { unreachable!("reduction identity must be constant") };
            Some(reduce.const_like(value.0))
        },

        // Any non-SINK op with zero size → const 0.
        x if !matches!(x.op(), Op::Sink(..)) && has_zero_size(x) => {
            let replacement = x.const_like(0).rtag(x.tag().clone()).rorigin(x.origin());
            (!Arc::ptr_eq(&replacement, x)).then_some(replacement)
        }
    }
}

// ============================================================================
// RANGEIFY TRANSFORMATION PATTERNS
// ============================================================================

/// Create patterns for applying rangeify transformation with IndexingContext.
///
/// Pattern order:
/// 1. Tensor REDUCE → ranged REDUCE conversion
/// 2. PAD → WHERE conversion (convert_pad_to_where_to_keep_behavior_local)
/// 3. ALL ops get source bufferization (including movement ops)
/// 4. Movement ops get removed (simple - just return source)
pub fn apply_rangeify_patterns() -> TypedPatternMatcher<IndexingContext> {
    crate::patterns! {
        @context IndexingContext;
        // Tensor REDUCE conversion MUST come first, before STAGE wraps it.
        x @ Reduce { src: _, ranges: _, reduce_op: _, num_axes: _ }
            => convert_reduce_with_context(x, ctx),
        // PAD → WHERE conversion BEFORE bufferization.
        x @ Pad { src: _, begin_pads: _, end_pads: _ } => convert_pad_to_where(x, ctx),
        // STACK → WHERE select on the leading range, BEFORE bufferization.
        x @ Stack { sources: _ } => convert_stack_to_where(x, ctx),
        // ALL ops (including movement) get source bufferization.
        x => apply_bufferize_transform(x, ctx),
        // Movement ops get removed AFTER bufferization - simple logic
        x if x.op().is_movement() => remove_movement_op(x, ctx),
    }
}

/// Apply STAGE transformation to op sources.
///
/// When sources change, the new node gets a different Arc identity. We must
/// transfer range_map + realize_map so downstream patterns (e.g. `remove_movement_op`)
/// can find the new node's context — same as `convert_reduceaxis_with_context`.
fn apply_bufferize_transform(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Arc<UOp>> {
    if let Some(new_sources) = transform_sources_with_bufferize(x, ctx) {
        let new_node = x.with_sources(new_sources);
        // Transfer context to new identity
        if let Some((in_rngs, out_rngs)) = ctx.get_ranges(x) {
            ctx.set_ranges(&new_node, in_rngs.clone(), out_rngs.clone());
        }
        if let Some(realize_axes) = ctx.get_realize_axes(x).cloned() {
            ctx.mark_realize(&new_node, realize_axes);
        }
        return Some(new_node);
    }
    None
}

/// Convert PAD → WHERE(combined_valid, source, 0).
///
/// Extracts validity conditions from PAD's input ranges (WHERE-Invalid patterns)
/// and wraps the PAD's data source in a WHERE that produces 0 for padded regions.
fn convert_pad_to_where(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Arc<UOp>> {
    let (input_ranges, _) = ctx.get_ranges(x)?.clone();
    let sources = transform_sources_with_bufferize(x, ctx).unwrap_or_else(|| x.op().sources().into_iter().collect());
    let bx = x.with_sources(sources);

    let mut valid = UOp::const_(DType::Bool, ConstValue::Bool(true));
    for r in &input_ranges {
        valid = valid.try_and_op(&r.get_valid()).ok()?;
    }

    let base = x.dtype().scalar()?;
    let zero = UOp::const_(x.dtype(), ConstValue::zero(base));
    UOp::try_where(valid, bx.op().sources().first()?.clone(), zero).ok()
}

/// Convert a shaped STACK into a WHERE chain selecting on its leading range.
///
/// Every source is indexed at the *same* trailing ranges, so producers shared
/// between the stacked slices collapse to one node (Tinygrad
/// `convert_stack_to_where`). Shape-payload STACKs carry no ranges and are left
/// alone.
fn convert_stack_to_where(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Arc<UOp>> {
    if x.dtype() == DType::Void {
        return None;
    }
    let selector = ctx.get_ranges(x)?.1.first()?.clone();
    let sources = transform_sources_with_bufferize(x, ctx).unwrap_or_else(|| x.op().sources().into_iter().collect());
    let (last, rest) = sources.split_last()?;
    rest.iter().enumerate().try_rfold(last.clone(), |acc, (k, source)| {
        let key = UOp::const_(selector.dtype(), ConstValue::Int(k as i64));
        UOp::try_where(selector.try_cmpeq(&key).ok()?, source.clone(), acc).ok()
    })
}

/// Remove movement ops after source bufferization.
///
/// Removes when the movement op has range context or when its source is INDEX.
fn remove_movement_op(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Arc<UOp>> {
    let src = x.op().sources().first()?.clone();

    if ctx.get_ranges(x).is_some() || matches!(src.op(), Op::Index(..)) {
        return Some(src);
    }

    None
}

/// Convert tensor-form REDUCE to ranged loop-form REDUCE using IndexingContext.
///
/// - Tensor form has no ranges and `num_axes > 0`.
/// - Loop form carries the leading input ranges and has `num_axes == 0`.
/// - Transfer range_map + realize_map to new identity
fn convert_reduce_with_context(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Arc<UOp>> {
    let Op::Reduce(ops::Reduce { src, ranges, reduce_op, num_axes }) = x.op() else {
        return None;
    };
    if *num_axes == 0 {
        return None;
    }
    debug_assert!(ranges.is_empty(), "tensor-form REDUCE must not already have loop ranges");

    let (input_ranges, output_ranges) = ctx.get_ranges(x)?.clone();
    let bx_sources = transform_sources_with_bufferize(x, ctx).unwrap_or_else(|| x.op().sources().into_iter().collect());
    let indexed_src = bx_sources.first().cloned().unwrap_or_else(|| src.clone());
    let reduce_ranges: SmallVec<[Arc<UOp>; 4]> = input_ranges.iter().take(*num_axes).cloned().collect();
    let target = indexed_src.reduce(reduce_ranges, *reduce_op).rorigin(x.origin());
    let target = if let Some(t) = x.tag() { target.with_tag(t.clone()) } else { target };

    // Transfer context to new identity (range_map + realize_map only)
    ctx.set_ranges(&target, input_ranges, output_ranges);
    if let Some(realize_axes) = ctx.get_realize_axes(x).cloned() {
        ctx.mark_realize(&target, realize_axes);
    }

    Some(target)
}

// ============================================================================
// BUFFER FOLDING PATTERNS
// ============================================================================

/// Const folding through STAGE / INDEX / COPY / MSTACK and noop-stage
/// removal.
#[tracing::instrument]
pub fn buffer_folding() -> TypedPatternMatcher {
    crate::patterns! {
        Stage { compute: c @ Const(_), .. } => c.clone(),
        Index { buffer: c @ Const(_), .. } => c.clone(),
        Copy { src: c @ Const(_), .. } => c.clone(),
        idx @ Index { buffer: MStack { buffers }, .. }
            if !buffers.is_empty() && matches!(buffers[0].base().op(), Op::Const(_))
            => {
                let base = buffers[0].base();
                if let Op::Const(cv) = base.op() {
                    Some(idx.const_like(cv.0))
                } else {
                    None
                }
            },
        Index { buffer: buf @ Stage { compute, ranges, .. }, indices }
            if ranges_equal(ranges, indices) && !matches!(compute.op(), Op::Slice(..))
            => {
                // Merge tags, shrink to stage shape.
                let mut merged = SmallVec::<[usize; 2]>::new();
                if let Some(t) = compute.tag() { merged.extend(t.iter().copied()); }
                if let Some(t) = buf.tag() { merged.extend(t.iter().copied()); }
                let tag = if merged.is_empty() { None } else { Some(merged) };
                // Tags merge because both identities must reach the output map; origin
                // keeps the primary one and the harvest union recovers the rest.
                let result = compute.rtag(tag).rorigin(compute.origin().or_else(|| buf.origin()));
                // .shrink((0, s) for s in b2.shape) if b2.shape.
                // try_shrink has noop detection (returns self when result shape == shrink shape).
                if !ranges.is_empty()
                    && let (Ok(Some(buf_shape)), Ok(Some(_))) = (buf.shape(), result.shape()) {
                        let shrink_ranges: Vec<_> = buf_shape.iter()
                            .map(|s| (svod_ir::SInt::Const(0), s.clone()))
                            .collect();
                        if let Ok(shrunk) = result.try_shrink(&shrink_ranges) {
                            return Some(shrunk);
                        }
                    }
                Some(result)
            },
    }
}

/// Strip dead axes (size-1 or unreferenced ranges) from STAGE,
/// preserving the original shape via RESHAPE + EXPAND.
pub fn dead_axis_removal() -> TypedPatternMatcher {
    crate::patterns! {
        // Filter dead axes from STAGE with shape preservation
        stage @ Stage { compute, ranges, opts } => {
            cleanup_dead_axes_bufferize(stage, compute, ranges, opts)
        },
    }
}

/// Clean up dead axes from STAGE with shape preservation.
///
/// When removing dead axes (ranges with size 1 or ranges not used by compute):
/// 1. Create new STAGE with only live ranges
/// 2. RESHAPE to insert size-1 dims for dead axes
/// 3. EXPAND to restore original shape
///
/// This preserves shape semantics for downstream operations.
fn cleanup_dead_axes_bufferize(
    stage: &Arc<UOp>,
    compute: &Arc<UOp>,
    ranges: &SmallVec<[Arc<UOp>; 4]>,
    opts: &BufferizeOpts,
) -> Option<Arc<UOp>> {
    use svod_ir::SInt;
    use svod_ir::shape::Shape;

    // Don't optimize ALWAYS_RUN_OPS or AFTER (tinygrad `schedule/rangeify.py:198`).
    // AFTER is a buffer-identity wrapper: ranges define consumer access, not the
    // computation's own shape, so dead-axis pruning would mangle assign-chain
    // semantics. COPY joins them via `is_always_run_op`, matching the guard
    // `remove_bufferize` already applies: shrinking a copy's destination would
    // under-allocate the transfer.
    if !opts.removable || is_always_run_op(compute.op()) || matches!(compute.op(), Op::After(..)) {
        return None;
    }

    // Get original STAGE shape (now available after Fix 1)
    let original_shape = stage.shape().ok().flatten()?;

    // Get compute's ranges to check if a range is used
    let compute_ranges = compute.ranges();

    let mut new_ranges = Vec::new();
    let mut reshape_dims: Shape = SmallVec::new();
    let mut had_dead = false;

    for (i, range) in ranges.iter().enumerate() {
        // Skip symbolic range ends — dead-axis cleanup cannot prove
        // shape/range equivalence before binding values.
        if let Op::Range(ops::Range { end, .. }) = range.op()
            && !matches!(end.op(), Op::Const(_))
        {
            return None;
        }

        // A range is dead if:
        // 1. It's a CONST (already dead)
        // 2. OR it's a RANGE with size 1
        // 3. OR it's a RANGE not in compute's ranges
        let is_const = matches!(range.op(), Op::Const(_));
        let is_unused = matches!(range.op(), Op::Range(..)) && !compute_ranges.iter().any(|r| Arc::ptr_eq(r, range));

        if is_const || is_unused {
            reshape_dims.push(SInt::Const(1)); // Dead axis → size 1
            had_dead = true;
        } else {
            // Live axis: keep range and original dimension
            new_ranges.push(Arc::clone(range));
            {
                let dim = original_shape.get(i)?;
                reshape_dims.push(dim.clone());
            }
        }
    }

    if !had_dead {
        return None;
    }

    // Create STAGE with fewer (or zero) ranges
    let reduced = UOp::stage(compute.clone(), new_ranges, opts.clone());

    // RESHAPE to insert size-1 dims for dead axes
    let reshaped = reduced.try_reshape(&reshape_dims).ok()?;

    // EXPAND to restore original shape
    reshaped.try_expand(original_shape).ok()
}

// ============================================================================
// BUFFER REMOVAL PATTERNS
// ============================================================================

/// Cost-bounded inlining of `INDEX(STAGE(...))` plus two cleanup rules
/// that fire after substitution:
/// - `STORE(x, x) → NOOP`
/// - `END(NOOP, ..) → NOOP`
pub fn pm_remove_bufferize() -> TypedPatternMatcher {
    crate::patterns! {
        Index { buffer: Stage { compute: src, ranges: buf_ranges, opts }, indices: idx_ranges, .. }
            => {
                remove_bufferize(src, buf_ranges, idx_ranges, opts)
            },
        Store { index: x, value: y, .. } if Arc::ptr_eq(x, y) => Some(UOp::noop()),
        End { computation: noop, .. } if matches!(noop.op(), Op::Noop) => Some(noop.clone()),
    }
}

/// Inline a STAGE into its INDEX consumer by substituting buffer ranges
/// with the consumer's index ranges. Bails when:
/// 1. `src` is an always-run op or the Stage is non-removable
///    (multi-consumer realize boundary).
/// 2. The compute touches more than 3 distinct GLOBAL Bufferizes / MStacks /
///    Params / AFTER buffers (would expand kernel input pressure).
/// 3. Any reduce body reads a buffer (would compound reads inside the loop).
///
/// CONST range keys are skipped during substitution — they're broadcast slots,
/// not real loop variables.
fn remove_bufferize(
    src: &Arc<UOp>,
    buf_ranges: &SmallVec<[Arc<UOp>; 4]>,
    idx_ranges: &SmallVec<[Arc<UOp>; 4]>,
    opts: &BufferizeOpts,
) -> Option<Arc<UOp>> {
    use std::collections::{HashMap, HashSet};

    debug_assert_eq!(buf_ranges.len(), idx_ranges.len(), "INDEX/STAGE range arity mismatch");
    debug_assert!(
        buf_ranges.iter().all(|r| matches!(r.op(), Op::Range(..) | Op::Const(_))),
        "STAGE ranges must be Range or Const"
    );

    if is_always_run_op(src.op()) || !opts.removable {
        return None;
    }

    let mut accessed_buffers: Vec<Arc<UOp>> = Vec::new();
    let mut reduces: Vec<Arc<UOp>> = Vec::new();
    let mut visited: HashSet<UOpKey> = HashSet::new();

    fn collect(
        uop: &Arc<UOp>,
        buffers: &mut Vec<Arc<UOp>>,
        reduces: &mut Vec<Arc<UOp>>,
        visited: &mut HashSet<UOpKey>,
    ) {
        if !visited.insert(UOpKey(Arc::clone(uop))) {
            return;
        }
        match uop.op() {
            // AFTER is a buffer identity: it costs its own buffer, once, and the
            // producers it orders against are not read by this compute.
            Op::After(..) => {
                buffers.push(uop.buf_uop());
                return;
            }
            // STORE doesn't count, and we don't look inside it.
            Op::Store(..) => return,
            // GLOBAL Stage and MStack count + stop traversal.
            Op::Stage(ops::Stage { opts, .. }) if opts.addrspace == AddrSpace::Global => {
                buffers.push(Arc::clone(uop));
                return;
            }
            Op::MStack(..) => {
                buffers.push(Arc::clone(uop));
                return;
            }
            // PARAM (and pre-normalize BUFFER) count but traversal continues.
            Op::Param(..) | Op::Buffer(..) => {
                buffers.push(Arc::clone(uop));
            }
            Op::Reduce(..) => reduces.push(Arc::clone(uop)),
            _ => {}
        }
        for child in uop.op().sources() {
            collect(&child, buffers, reduces, visited);
        }
    }
    collect(src, &mut accessed_buffers, &mut reduces, &mut visited);

    let mut seen: HashSet<UOpKey> = HashSet::new();
    accessed_buffers.retain(|b| seen.insert(UOpKey(Arc::clone(b))));

    if accessed_buffers.len() > 3 {
        tracing::debug!(
            src_id = src.id,
            src_op = src.op().as_ref(),
            buf_count = accessed_buffers.len(),
            "remove_bufferize: KEPT (>3 accessed buffers)"
        );
        return None;
    }

    if !reduces.is_empty() {
        let reduce_sources: Vec<Arc<UOp>> = reduces
            .iter()
            .filter_map(|r| if let Op::Reduce(ops::Reduce { src, .. }) = r.op() { Some(Arc::clone(src)) } else { None })
            .collect();
        if !reduce_sources.is_empty() {
            let sink = UOp::sink(reduce_sources);
            let buffer_in_reduce =
                sink.any_in_subtree(|n| matches!(n.op(), Op::Param(..) | Op::Buffer(..) | Op::Stage(..)));
            if buffer_in_reduce {
                tracing::debug!(
                    src_id = src.id,
                    src_op = src.op().as_ref(),
                    "remove_bufferize: KEPT (buffer_in_reduce)"
                );
                return None;
            }
        }
    }

    // Skip CONST keys (broadcast slots, not loop vars) and dead-load `Invalid`
    // values — substituting a bare Invalid index would poison the inlined expr.
    let subs_map: HashMap<UOpKey, Arc<UOp>> = buf_ranges
        .iter()
        .zip(idx_ranges.iter())
        .filter(|(k, v)| !matches!(k.op(), Op::Const(_)) && !UOp::is_invalid_marker(v))
        .map(|(k, v)| (UOpKey(Arc::clone(k)), Arc::clone(v)))
        .collect();
    Some(src.substitute_gated(&subs_map))
}

// ============================================================================
// REDUCTION SIMPLIFY PATTERNS
// ============================================================================

/// Pattern matcher for splitting large tensor-form REDUCE operations.
/// Must run before tensor REDUCE → ranged REDUCE conversion.
pub fn split_reduceop_patterns() -> TypedPatternMatcher<SplitReduceOpConfig> {
    crate::patterns! {
        @context SplitReduceOpConfig;
        reduce @ Reduce { src: _, ranges: _, reduce_op: _, num_axes: _ }
            => split_reduceop(reduce, ctx),
    }
}

/// Pattern matcher for reduce_unparented: remove ranges not referenced by body.
///
/// Factored out so it can be shared between `pm_reduce_simplify` and the inner
/// `reduce_collapse` pattern matchers without duplication.
pub(crate) fn pm_reduce_unparented() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        reduce @ Reduce { src, ranges, reduce_op: reduce_op @ (ReduceOp::Add | ReduceOp::Max | ReduceOp::Mul), num_axes } => {
            // `dead_loop_patterns` folds a trip-1 RANGE to its only value and
            // `IndexingContext::new_range` never builds one, so a CONST in the range
            // list is a single point: unparented by construction, scaling by one.
            // Anything else is a shape this rule has no trip count for.
            if !ranges.iter().all(|r| matches!(r.op(), Op::Range(..) | Op::Const(..))) {
                return None;
            }
            let src_ranges = src.in_scope_ranges();
            let (parented, unparented) = partition_reduce_ranges(ranges, src_ranges);

            if unparented.is_empty() {
                return None;
            }

            let mut result = if !parented.is_empty() || reduce.dtype() != src.dtype() {
                src.reduce_with_num_axes(parented, *reduce_op, *num_axes)
            } else {
                Arc::clone(src)
            };

            match reduce_op {
                ReduceOp::Add => {
                    for size in unparented.iter().filter_map(get_range_size) {
                        let size_casted = cast_to_dtype(&size, &result.dtype())?;
                        result = result.try_mul(&size_casted).ok()?;
                    }
                }
                ReduceOp::Mul => {
                    for size in unparented.iter().filter_map(get_range_size) {
                        let size_casted = cast_to_dtype(&size, &result.dtype())?;
                        result = result.try_pow(&size_casted).ok()?;
                    }
                }
                ReduceOp::Max => {}
                _ => unreachable!("pattern only accepts ADD, MAX, and MUL"),
            }

            Some(result)
        },
    }
}

/// Check if a UOp's DAG references any of the given reduce ranges.
///
/// Uses `in_scope_ranges` (cached property) to check if any of the given ranges
/// appear in the UOp's dependency graph.
fn references_any_reduce_range(uop: &Arc<UOp>, ranges: &SmallVec<[Arc<UOp>; 4]>) -> bool {
    let in_scope = uop.in_scope_ranges();
    ranges.iter().any(|r| in_scope.contains(&r.id))
}

/// Split a UOp by multiplication into leaf factors.
///
/// E.g. `(a * b) * c` → `[a, b, c]`
fn split_mul_factors(uop: &Arc<UOp>) -> SmallVec<[Arc<UOp>; 4]> {
    match uop.op() {
        Op::Binary(BinaryOp::Mul, a, b) => {
            let mut factors = split_mul_factors(a);
            factors.extend(split_mul_factors(b));
            factors
        }
        _ => smallvec::smallvec![uop.clone()],
    }
}

/// Factor multiplicative terms that don't depend on reduce ranges outside the REDUCE.
///
/// For ADD reduce: `REDUCE(x * c, ADD, ranges)` → `REDUCE(x, ADD, ranges) * c`
/// For MAX reduce: same, but only if the outside factor's vmin >= 0.
fn reduce_mul_chain(
    src: &Arc<UOp>,
    ranges: &SmallVec<[Arc<UOp>; 4]>,
    reduce_op: ReduceOp,
    num_axes: usize,
) -> Option<Arc<UOp>> {
    if src.dtype().is_float() {
        return None;
    }
    let factors = split_mul_factors(src);
    if factors.len() < 2 {
        return None;
    }

    let mut inside: SmallVec<[Arc<UOp>; 4]> = SmallVec::new();
    let mut outside: SmallVec<[Arc<UOp>; 4]> = SmallVec::new();

    for factor in &factors {
        if references_any_reduce_range(factor, ranges) {
            inside.push(factor.clone());
        } else {
            // For MAX reduce, only factor out non-negative values
            if reduce_op == ReduceOp::Max {
                let is_non_negative = match SoundVminVmaxProperty::get(factor).as_ref().map(|bounds| &bounds.0) {
                    Some(ConstValue::Int(v)) => *v >= 0,
                    Some(ConstValue::UInt(_)) => true,
                    Some(ConstValue::Float(v)) => *v >= 0.0,
                    Some(ConstValue::Bool(_)) => true,
                    _ => false,
                };
                if !is_non_negative {
                    inside.push(factor.clone());
                    continue;
                }
            }
            outside.push(factor.clone());
        }
    }

    if outside.is_empty() {
        return None;
    }

    // Reconstruct inside product (if all factors are outside, reduce over const 1)
    let inner = inside.into_iter().reduce(|a, b| a.mul(&b)).unwrap_or_else(|| src.const_like(1i64));
    let reduced = inner.reduce_with_num_axes(ranges.clone(), reduce_op, num_axes);

    // Multiply by outside product
    let mut result = reduced;
    for factor in &outside {
        result = result.mul(factor);
    }

    Some(result)
}

/// Pattern matcher for reduction simplifications (mega-pass path).
///
/// - `pm_reduce_unparented`: remove ranges not referenced by body
/// - `REDUCE(ADD) → reduce_collapse(src, ranges)`: delegate to procedural wrapper
/// - `reduce_mul_chain`: factor range-independent multipliers outside REDUCE
///
/// Distributive and bound patterns live inside `reduce_collapse_inner_patterns()`
/// and run via `reduce_collapse`.
pub fn pm_reduce_simplify() -> &'static TypedPatternMatcher {
    static CACHED: std::sync::LazyLock<TypedPatternMatcher> = std::sync::LazyLock::new(|| {
        pm_reduce_unparented()
            + crate::patterns! {
                Reduce { src, ranges, reduce_op, num_axes } if *reduce_op == ReduceOp::Add && *num_axes == 0
                    => super::transforms::reduce_collapse(src, ranges),

                Reduce { src, ranges, reduce_op, num_axes }
                    if matches!(reduce_op, ReduceOp::Add | ReduceOp::Max)
                    && matches!(src.op(), Op::Binary(BinaryOp::Mul, _, _))
                    => reduce_mul_chain(src, ranges, *reduce_op, *num_axes),
            }
    });
    &CACHED
}

// ============================================================================
// MOVEMENT OP PATTERNS
// ============================================================================

/// Push movement ops (RESHAPE / PERMUTE / EXPAND / PAD / SHRINK / FLIP)
/// through INDEX, AFTER, and END so they can be folded into surrounding
/// loop-range arithmetic.
pub fn movement_op_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        idx @ Index { buffer: mop, indices } if mop.op().is_movement() => {
            transform_movement_through_index(mop, indices, idx)
        },
        after @ After { passthrough: r, deps }
            if r.op().is_movement() || matches!(r.op(), Op::Index(..))
            => {
                super::transforms::push_op_through_after(after, r, deps)
            },
        end @ End { computation: mop, ranges } if mop.op().is_movement()
            => {
                let src = &mop.op().sources()[0];
                Some(end.with_sources(std::iter::once(src.clone()).chain(ranges.iter().cloned()).collect()))
            },
    }
}

/// Transform a movement op through INDEX by applying the movement to indices.
pub(crate) fn transform_movement_through_index(
    mop: &Arc<UOp>,
    indices: &SmallVec<[Arc<UOp>; 4]>,
    index: &Arc<UOp>,
) -> Option<Arc<UOp>> {
    use super::indexing::{apply_movement_op, apply_reshape_ranges};

    let src = &mop.op().sources()[0];
    let src_shape = src.shape().ok()??;
    let mop_shape = mop.shape().ok()??;

    if indices.len() == mop_shape.len() {
        let transformed = apply_movement_op(mop.op(), src_shape, indices.as_slice());
        return UOp::index().buffer(src.clone()).indices(transformed).dtype(index.dtype()).call().ok();
    }

    if !matches!(mop.op(), Op::Reshape(..)) {
        return None;
    }

    let suffix_len = mop_shape.len().checked_sub(indices.len())?;
    let src_prefix = src_shape.len().checked_sub(suffix_len)?;
    if src_shape[src_prefix..] != mop_shape[indices.len()..] {
        return None;
    }
    if src_prefix == 0 {
        return (src.dtype() == index.dtype()).then(|| src.clone());
    }

    let transformed = apply_reshape_ranges(&src_shape[..src_prefix], &mop_shape[..indices.len()], indices);
    let ret = UOp::index().buffer(src.clone()).indices(transformed).dtype(index.dtype()).call().ok()?;
    (ret.shape().ok()? == index.shape().ok()?).then_some(ret)
}

// ============================================================================
// CODEGEN PREPARATION PATTERNS
// ============================================================================

/// Create zero UOp for a given dtype (scalar or vector).
fn dtype_zero(dtype: DType) -> Arc<UOp> {
    let base = dtype.base();
    let zero = ConstValue::zero(base);
    if dtype.is_vector() {
        UOp::stack((0..dtype.count()).map(|_| UOp::const_(DType::Scalar(base), zero)).collect())
    } else {
        UOp::const_(dtype, zero)
    }
}

/// Create patterns for codegen preparation.
///
/// Note: Multi-index INDEX ops are preserved through the pipeline and
/// linearized at codegen time (not here). This prevents Binary(Range*stride)
/// expressions in the IR.
///
/// # CONTIGUOUS Hint Extraction
///
/// Extracts optimization hints from CONTIGUOUS.opts into ctx.opts for later use.
pub fn rangeify_codegen_patterns() -> TypedPatternMatcher<LocalAddBufferContext> {
    crate::patterns! {
        @context LocalAddBufferContext;
        // NOOP → zero constant (scalar or vector)
        noop @ Noop if noop.dtype().base() != svod_dtype::ScalarDType::Void => {
            Some(dtype_zero(noop.dtype()))
        },
        // CONTIGUOUS: extract hints and return source.
        Contiguous { src, opts } => {
            if !opts.is_empty() {
                ctx.opts.extend(opts.iter().cloned());
            }
            Some(src.clone())
        },
    }
}

// ============================================================================
// CALL-WRAPPER SPLITTING PATTERNS
// ============================================================================

/// Extract base dtype from a Ptr type, or return the type as-is.
fn extract_base_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Ptr { base, .. } => (*base).clone(),
        other => other,
    }
}

/// Extract buffer from AFTER passthrough (handles MStack/MSelect).
fn extract_buffer_from_after(passthrough: &Arc<UOp>) -> Arc<UOp> {
    match passthrough.op() {
        Op::MStack(ops::MStack { buffers }) if !buffers.is_empty() => buffers[0].clone(),
        Op::MSelect(ops::MSelect { buffer, .. }) => buffer.clone(),
        _ => passthrough.clone(),
    }
}

fn storage_addrspace(node: &Arc<UOp>) -> Option<AddrSpace> {
    match node.op() {
        Op::Buffer(ops::Buffer { arg, .. }) | Op::Param(ops::Param { arg, .. }) => arg.addrspace,
        Op::Slice(ops::Slice { buffer, .. }) | Op::MSelect(ops::MSelect { buffer, .. }) => storage_addrspace(buffer),
        Op::After(ops::After { passthrough, .. }) => storage_addrspace(passthrough),
        Op::MStack(ops::MStack { buffers }) => buffers.first().and_then(storage_addrspace),
        _ => None,
    }
}

/// Find output PARAM from a CALL body AST.
fn find_kernel_output(ast: &Arc<UOp>) -> Option<Arc<UOp>> {
    for node in ast.toposort() {
        // Use store_buffer() helper to get buffer from STORE via its INDEX child
        if let Some(buffer) = node.store_buffer() {
            let output_buf = match buffer.op() {
                Op::Index(ops::Index { buffer: inner_buf, .. }) => inner_buf.clone(),
                _ => buffer.clone(),
            };
            if is_codegen_param(&output_buf) {
                return Some(output_buf);
            }
        }
    }
    None
}

fn map_after_like_node(node: &Arc<UOp>, ctx: &mut LocalAddBufferContext) -> Option<Arc<UOp>> {
    // buf = after.buf_uop(); if buf is MSTACK/MSELECT, descend into its first source.
    let mut buf = node.buf_uop();
    buf = match buf.op() {
        Op::MStack(ops::MStack { buffers }) if !buffers.is_empty() => buffers[0].clone(),
        Op::MSelect(ops::MSelect { buffer, .. }) => buffer.clone(),
        _ => buf,
    };

    // Only global storage participates in the CALL tuple. Internal storage is
    // unwrapped for kernel use without consuming a PARAM slot or CALL argument.
    match storage_addrspace(&buf) {
        Some(AddrSpace::Global) => {}
        Some(AddrSpace::Local | AddrSpace::Reg) => return Some(buf),
        None => return None,
    }

    // A reused buffer (the level-interval planner aliases non-overlapping
    // lifetimes) can be the target of more than one AFTER within a kernel's
    // input cone. That is legal: keep the last writer as this kernel's source
    // (matching tinygrad's `{u.buf_uop: u for u in afters}` last-wins), and let
    // `fix_assign` re-derive the cross-kernel ordering globally from `buf_uop`.
    // Either way the node is replaced by its buffer so consumers read the buffer.
    ctx.map_buffer(buf.clone(), node.clone());
    Some(buf)
}

/// Create patterns for to_param transformation (normalizes buffers to codegen PARAMs).
pub fn to_param_patterns() -> TypedPatternMatcher<RangeifyBufferContext> {
    crate::patterns! {
        @context RangeifyBufferContext;
        // Only global buffers participate in the CALL ABI.
        buf @ Buffer { arg } if arg.addrspace == Some(AddrSpace::Global) => {
            let size = buf.buffer_size()?;
            let replacement = UOp::param(ctx.next_global(), size, extract_base_dtype(buf.dtype()), arg.device.clone());
            ctx.map_buffer(buf.clone(), replacement.clone());
            Some(replacement)
        },
        // Remove BIND: extract var and track it with its bound value
        Bind { var, value } => {
            let bound_val = match value.op() {
                Op::Const(cv) => cv.0.try_int(),
                _ => None,
            };
            ctx.add_var(var.clone(), bound_val);
            Some(var.clone())
        },
        // Handle AFTER: extract buffer and track dependency
        after @ After { passthrough } => {
            let buf = extract_buffer_from_after(passthrough);
            if matches!(storage_addrspace(&buf), Some(AddrSpace::Local | AddrSpace::Reg)) {
                return Some(buf);
            }
            ctx.map_buffer(buf.clone(), after.clone());
            Some(buf)
        },
        // Replace RANGE(end=0) with CONST(0)
        Range { end } if matches!(end.op(), Op::Const(v) if v.0.is_zero()) => {
            ctx.next_range();
            Some(UOp::index_const(0))
        },
        // Renumber RANGE axis_id (Unrenumbered → Renumbered)
        Range { end, axis_id, axis_type } if matches!(axis_id, AxisId::Unrenumbered(_)) => {
            Some(UOp::range_axis(end.clone(), AxisId::Renumbered(ctx.next_range()), *axis_type))
        },
        // Replace CALL references with their output buffer
        Call { body, args: _, info: _ } => find_kernel_output(body),
    }
}

/// Create patterns for to_param transformation using LocalAddBufferContext.
///
/// Creates per-kernel codegen PARAMs with sequential slots.
pub fn local_to_param_patterns() -> TypedPatternMatcher<LocalAddBufferContext> {
    crate::patterns! {
        @context LocalAddBufferContext;
        // Only global storage is part of the CALL ABI. REG/LOCAL BUFFERs remain
        // structured compiler-managed allocations and must not consume slots.
        buf @ Buffer { arg } if arg.addrspace == Some(AddrSpace::Global) => {
            let size = buf.buffer_size()?;
            let replacement = mark_codegen_param(UOp::param(
                ctx.next_param_slot(),
                size,
                extract_base_dtype(buf.dtype()),
                arg.device.clone(),
            ));
            if !ctx.has_buffer(buf) {
                ctx.map_buffer(buf.clone(), buf.clone());
            }
            Some(replacement)
        },
        // Pre-kernel PARAM → codegen PARAM. The missing device metadata prevents
        // this rule from repeatedly matching its own replacement.
        buf @ Param { arg }
            if !is_codegen_param(buf) && arg.device.is_some() && arg.addrspace == Some(AddrSpace::Global)
            => {
            let Op::Param(ops::Param { shape, .. }) = buf.op() else { unreachable!() };
            let mut arg = arg.clone();
            arg.slot = ctx.next_param_slot();
            let replacement = mark_codegen_param(UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg }), buf.dtype()).rtag(buf.tag().clone()));
            if !ctx.has_buffer(buf) {
                ctx.map_buffer(buf.clone(), buf.clone());
            }
            Some(replacement)
        },
        // Remove BIND in AST while preserving the binding as a CALL source.
        b @ Bind { var, value } => {
            let _ = b;
            let bound_val = match value.op() {
                Op::Const(cv) => cv.0.try_int(),
                _ => None,
            };
            let mut tags = var.tag().clone().unwrap_or_default();
            if !tags.contains(&svod_ir::uop::canonical::TAG_CALL_BIND_PARAM) {
                tags.push(svod_ir::uop::canonical::TAG_CALL_BIND_PARAM);
            }
            let call_var = var.with_tag(tags);
            ctx.add_var(call_var.bind(value.clone()), call_var, bound_val);
            Some(var.clone())
        },
        // Handle AFTER, MSTACK, MSELECT uniformly.
        after @ After { passthrough: _ } => map_after_like_node(after, ctx),
        m @ MStack { buffers: _ } => map_after_like_node(m, ctx),
        m @ MSelect { buffer: _, device_index: _ } => map_after_like_node(m, ctx),
        // Replace RANGE(end=0) with CONST(0)
        Range { end } if matches!(end.op(), Op::Const(v) if v.0.is_zero()) => {
            ctx.next_range();
            Some(UOp::index_const(0))
        },
        // Renumber Unrenumbered Range axis_id.
        Range { end, axis_id, axis_type } if matches!(axis_id, AxisId::Unrenumbered(_)) => {
            Some(UOp::range_axis(end.clone(), AxisId::Renumbered(ctx.next_range()), *axis_type))
        },
    }
}

/// Create pattern matcher for split_kernels.
///
/// Matches STORE and END operations and calls split_store on them.
pub fn split_kernels_pattern() -> TypedPatternMatcher<Vec<Arc<UOp>>> {
    use super::kernel::split_store;
    crate::patterns! {
        @context Vec<Arc<UOp>>;
        x @ Store { index: _, value: _, .. } => split_store(ctx, x),
        x @ End { computation: _ } => split_store(ctx, x),
    }
}

// ============================================================================
// BUFFER LIMIT PATTERNS
// ============================================================================

/// Extract device specification from a UOp graph (first device found).
pub fn extract_device_from_graph(root: &Arc<UOp>) -> Option<DeviceSpec> {
    let mut visited = HashSet::new();

    fn visit(uop: &Arc<UOp>, visited: &mut HashSet<UOpKey>) -> Option<DeviceSpec> {
        let key = UOpKey(Arc::clone(uop));
        if !visited.insert(key) {
            return None;
        }

        match uop.op() {
            Op::Buffer(ops::Buffer { arg, .. }) => return arg.device.clone(),
            Op::Copy(ops::Copy { device, .. }) | Op::AllReduce(ops::AllReduce { device, .. }) => {
                return Some(device.clone());
            }
            Op::Stage(ops::Stage { opts, .. }) => {
                if let Some(device_spec) = &opts.device {
                    return Some(device_spec.clone());
                }
            }
            _ => {}
        }

        for child in uop.op().sources() {
            if let Some(device) = visit(&child, visited) {
                return Some(device);
            }
        }

        None
    }

    visit(root, &mut visited)
}

/// Create pattern matcher for buffer limit enforcement.
///
/// Uses `binary [*]` and `ternary [*]` to match all binary/ternary ops,
/// checks if they access too many buffers, and forces bufferization of
/// elementwise sources to GLOBAL memory if so.
#[allow(unused_variables)] // `op` is used by macro expansion
pub fn buffer_limit_patterns(max_buffers: usize) -> TypedPatternMatcher<IndexingContext> {
    crate::patterns! {
        @context IndexingContext;
        for op in binary [*] {
            tree@op(a, b) => {
                check_buffer_limit(tree, &[a.clone(), b.clone()], max_buffers, ctx)
            },
        }

        for op in ternary [*] {
            tree@op(a, b, c) => {
                check_buffer_limit(tree, &[a.clone(), b.clone(), c.clone()], max_buffers, ctx)
            },
        }
    }
}

/// Check buffer limit and force bufferization if exceeded.
fn check_buffer_limit(
    tree: &Arc<UOp>,
    sources: &[Arc<UOp>],
    max_buffers: usize,
    ctx: &mut IndexingContext,
) -> Option<Arc<UOp>> {
    let all_buffers = collect_accessed_buffers(sources);

    if all_buffers.len() > max_buffers.saturating_sub(1) {
        let mut any_changed = false;
        let new_sources: Vec<_> = sources
            .iter()
            .map(|src| {
                if is_elementwise(src) {
                    let new = force_bufferize(src, ctx);
                    if !Arc::ptr_eq(&new, src) {
                        any_changed = true;
                    }
                    new
                } else {
                    src.clone()
                }
            })
            .collect();

        if any_changed {
            return Some(tree.with_sources(new_sources));
        }
    }
    None
}

/// Distinct kernel arguments the sources would read.
///
/// Mirrors tinygrad's `_limit_bufs` visitor (`schedule/rangeify.py:176`): STAGE,
/// AFTER, PARAM, MSELECT and MSTACK each cost exactly one argument and stop the
/// walk. Morok additionally sees pre-normalization BUFFER nodes, and skips
/// LOCAL/REG storage, which is compiler-managed and never consumes an argument
/// slot. Only GLOBAL STAGEs are realize boundaries here; a LOCAL/REG one is
/// materialised inside the kernel, so its own reads are what cost arguments.
fn collect_accessed_buffers(sources: &[Arc<UOp>]) -> Vec<Arc<UOp>> {
    let mut all_buffers = Vec::new();
    let mut visited = HashSet::new();

    fn collect_recursive(uop: &Arc<UOp>, buffers: &mut Vec<Arc<UOp>>, visited: &mut HashSet<UOpKey>) {
        let key = UOpKey(Arc::clone(uop));
        if !visited.insert(key) {
            return;
        }
        match uop.op() {
            // AFTER is a buffer identity: it costs its own buffer, once, and the
            // producers it orders against are not read by this kernel.
            Op::After(..) => {
                buffers.push(uop.buf_uop());
                return;
            }
            Op::Stage(ops::Stage { opts, .. }) if opts.addrspace == AddrSpace::Global => {
                buffers.push(Arc::clone(uop));
                return; // Stop at GLOBAL stage
            }
            Op::MStack(..) | Op::MSelect(..) => {
                buffers.push(Arc::clone(uop));
                return;
            }
            Op::Buffer(..) | Op::Param(..) => {
                if !matches!(storage_addrspace(uop), Some(AddrSpace::Local | AddrSpace::Reg)) {
                    buffers.push(Arc::clone(uop));
                }
                return;
            }
            _ => {}
        }
        for child in uop.op().sources() {
            collect_recursive(&child, buffers, visited);
        }
    }

    for src in sources {
        collect_recursive(src, &mut all_buffers, &mut visited);
    }

    // Deduplicate
    let mut seen = HashSet::new();
    all_buffers.retain(|b| seen.insert(UOpKey(Arc::clone(b))));
    all_buffers
}

/// Force bufferization of a computation to GLOBAL memory.
///
/// The new STAGE's axes are the ranges still *open* at `src` — tinygrad's
/// `s.ranges` (`uop/ops.py:483`), which drops whatever a nested STAGE, REDUCE or
/// END already closed. `UOp::ranges()` is the wider "every RANGE in the cone",
/// so it must be filtered through `in_scope_ranges()`: an already-closed range
/// is not an axis of this value, and rewriting it would rebuild every producer
/// that binds it — the copies compound through a chain of bufferized kernels.
fn force_bufferize(src: &Arc<UOp>, ctx: &mut IndexingContext) -> Arc<UOp> {
    let scope = src.in_scope_ranges();
    let original_ranges: Vec<_> = src.ranges().into_iter().filter(|range| scope.contains(&range.id)).collect();
    if original_ranges.is_empty() {
        return Arc::clone(src);
    }
    let end_ranges: Vec<_> = original_ranges
        .iter()
        .map(|range| match range.op() {
            Op::Range(ops::Range { axis_type: AxisType::Device, .. }) => range.clone(),
            Op::Range(ops::Range { end, .. }) => ctx.new_range_from_uop(end, AxisType::Weak),
            _ => range.clone(),
        })
        .collect();
    let substitutions: HashMap<_, _> = original_ranges
        .iter()
        .zip(&end_ranges)
        .filter(|(old, new)| !Arc::ptr_eq(old, new))
        .map(|(old, new)| (UOpKey(old.clone()), new.clone()))
        .collect();
    let compute = src.substitute(&substitutions);
    let opts = BufferizeOpts { device: None, local_axis: None, addrspace: AddrSpace::Global, removable: true };
    let bufferized = UOp::stage(compute, end_ranges, opts);
    UOp::index().buffer(bufferized).indices(original_ranges).call().unwrap_or_else(|_| Arc::clone(src))
}

// ============================================================================
// FMA DECOMPOSITION (a*b+c → MulAcc)
// ============================================================================

/// FMA pattern detection: a*b+c → MulAcc(a,b,c).
///
/// Applied late (post-optimization) so earlier passes can still work with
/// Add(Mul) structure. Only matches float types where FMA provides benefit
/// (maps to llvm.fma intrinsic).
pub fn pm_fma_decomposition() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        // (a*b)+c or c+(a*b) → MulAcc(a,b,c) using commutative matching
        // Dtype equality guard is an early-out; try_mulacc also validates matching dtypes.
        Add[Mul(a, b), c] if a.dtype().is_float() && a.dtype() == b.dtype() && a.dtype() == c.dtype() => {
            UOp::try_mulacc(a.clone(), b.clone(), c.clone()).ok()
        },
    }
}

// ============================================================================
// PM_LOAD_COLLAPSE - Collapse REDUCE with conditional loads
// ============================================================================

/// Check if UOp has no INDEX (load) in backward slice.
///
/// Used for index overflow protection pattern - we want to ensure
/// we don't do math on a loaded index since that can cause overflow.
/// Backed by the cached `has_index_in_sources` flag rather than a per-call DFS.
pub(crate) fn no_load(u: &Arc<UOp>) -> bool {
    !u.has_index_in_sources()
}

/// Check if a UOp represents a zero constant.
fn is_const_zero(u: &Arc<UOp>) -> bool {
    if let Op::Const(cv) = u.op() { cv.0.is_zero() } else { false }
}

/// Compute minimum of two UOps: `min(a, b) = -max(-a, -b)`.
///
/// Encoded this way (not as WHERE) so that the MAX bounds elimination rule
/// `max(x, y) → x when x.vmin >= y.vmax` can simplify boundary cases like
/// `min(x, N)` when `x.vmax == N`.
fn uop_min(a: &Arc<UOp>, b: &Arc<UOp>) -> Option<Arc<UOp>> {
    let neg_a = a.neg();
    let neg_b = b.neg();
    let max_neg = neg_a.try_max(&neg_b).ok()?;
    Some(max_neg.neg())
}

/// Try to collapse a REDUCE with conditional/gated patterns.
///
/// Core gated collapse logic shared by NE (Pattern 3) and EQ (Pattern 3b).
///
/// Substitutes range with `idx.cast(r.dtype).valid(in_bounds)` in the expression,
/// producing `where(in_bounds, expr[r:=valid_idx], 0)`.
fn gated_collapse_core(idx: &Arc<UOp>, range: &Arc<UOp>, end: &Arc<UOp>, expr: &Arc<UOp>) -> Option<Arc<UOp>> {
    let idx_casted = idx.cast(range.dtype());
    let zero = UOp::index_const(0);
    let in_bounds = idx_casted.try_cmpge(&zero).ok()?.try_and_op(&idx_casted.try_cmplt(end).ok()?).ok()?;
    let valid_idx = idx_casted.valid(in_bounds.clone());
    let subs: std::collections::HashMap<UOpKey, Arc<UOp>> = [(UOpKey(range.clone()), valid_idx)].into_iter().collect();
    let substituted = expr.substitute(&subs);
    let zero_like = UOp::const_(expr.dtype(), ConstValue::zero(expr.dtype().base()));
    UOp::try_where(in_bounds, substituted, zero_like).ok()
}

/// Whether `node` depends on `range` — the question the reduce-collapse family
/// actually asks, where `no_range` asks the stricter "depends on no range at all".
///
/// `RangesProperty` is the cached backward slice of RANGEs and excludes the node
/// itself, so a bare range is tested separately. Reading the cache beats a
/// memoized walk here: this runs per candidate inside the rewrite fixed point,
/// and a slice holds a handful of ranges.
fn reaches(node: &Arc<UOp>, range: &Arc<UOp>) -> bool {
    Arc::ptr_eq(node, range) || RangesProperty::get(node).iter().any(|r| Arc::ptr_eq(r, range))
}

/// Whether casting `src` to `dtype` keeps its values distinct — either the
/// destination covers the source format outright, or it covers the values this
/// node can actually take. A cast that merged two values would let several steps
/// of the reduction satisfy the equality, and keeping one would drop the rest.
fn cast_is_injective(src: &Arc<UOp>, dtype: &DType) -> bool {
    if !dtype.is_int() || !src.dtype().is_int() {
        return false;
    }
    let (lo, hi) = (dtype.min_value(), dtype.max_value());
    if src.dtype().min_value() >= lo && src.dtype().max_value() <= hi {
        return true;
    }
    let bound = |value: &ConstValue| match value {
        ConstValue::Int(v) => Some(*v as f64),
        ConstValue::UInt(v) => Some(*v as f64),
        _ => None,
    };
    bound(src.vmin()).is_some_and(|v| v >= lo) && bound(src.vmax()).is_some_and(|v| v <= hi)
}

/// Rewrite `idx == cmp` into the equivalent `idx' == range`, peeling casts and
/// range-invariant arithmetic off `cmp` and applying the inverse to `idx`.
///
/// The compared side is hardly ever the bare range: `gather` builds its arange
/// from a reduce that collapses to `(r + 1) + (-1)` under the index-dtype casts,
/// which is `r` and does not match as `r`.
fn solve_for_range(idx: &Arc<UOp>, cmp: &Arc<UOp>, range: &Arc<UOp>) -> Option<Arc<UOp>> {
    let (mut idx, mut cmp) = (idx.clone(), cmp.clone());
    // Every step replaces `cmp` with one of its sources, so its depth bounds the walk.
    loop {
        if Arc::ptr_eq(&cmp, range) {
            return Some(idx);
        }
        let step = match cmp.op() {
            Op::Cast(ops::Cast { src, .. }) if cast_is_injective(src, &cmp.dtype()) => {
                (idx.cast(src.dtype()), src.clone())
            }
            Op::Binary(BinaryOp::Add, x, y) if !reaches(y, range) => (idx.try_sub(y).ok()?, x.clone()),
            Op::Binary(BinaryOp::Add, x, y) if !reaches(x, range) => (idx.try_sub(x).ok()?, y.clone()),
            Op::Binary(BinaryOp::Sub, x, y) if !reaches(y, range) => (idx.try_add(y).ok()?, x.clone()),
            _ => return None,
        };
        (idx, cmp) = step;
    }
}

/// Reduction collapse patterns:
/// 1. Sum of `where(r < cut, 0, val)` → `clamp(end-cut, 0, end) * val`
/// 2. Sum of `where(r < cut, val, 0)` → `clamp(cut, 0, end) * val`
/// 3. Sum of `where(idx != r, 0, expr)` → `where(in_bounds, expr[r:=idx], 0)`
/// 4. Sum of `where((r >= lower) & (r < upper), val, 0)` → two-sided bounds
fn try_reduce_collapse(
    _reduce: &Arc<UOp>,
    src: &Arc<UOp>,
    ranges: &SmallVec<[Arc<UOp>; 4]>,
    reduce_op: ReduceOp,
) -> Option<Arc<UOp>> {
    // Only handle Add for now.
    if reduce_op != ReduceOp::Add {
        return None;
    }

    // Must have exactly one range
    if ranges.len() != 1 {
        return None;
    }

    let range = &ranges[0];
    let Op::Range(ops::Range { end, .. }) = range.op() else {
        return None;
    };

    // Pattern: WHERE(cond, true_val, false_val)
    let Op::Ternary(svod_ir::TernaryOp::Where, cond, true_val, false_val) = src.op() else {
        return None;
    };

    // Pattern 1: where(r < cut, 0, val) → (end - cut).max(0).min(end) * val
    if let Op::Binary(BinaryOp::Lt, lt_lhs, cut) = cond.op()
        && Arc::ptr_eq(lt_lhs, range)
        && is_const_zero(true_val)
        && no_range(false_val)
    {
        // count = (end - cut).max(0).min(end)  -- symbolic UOp arithmetic
        let zero = UOp::index_const(0);
        let diff = end.try_sub(cut).ok()?;
        let non_negative = diff.try_max(&zero).ok()?;
        let count = uop_min(&non_negative, end)?;
        let count_casted = count.cast(false_val.dtype());
        return count_casted.try_mul(false_val).ok();
    }

    // Pattern 2: where(r < cut, val, 0) → cut.max(0).min(end) * val
    if let Op::Binary(BinaryOp::Lt, lt_lhs, cut) = cond.op()
        && Arc::ptr_eq(lt_lhs, range)
        && is_const_zero(false_val)
        && no_range(true_val)
    {
        // count = cut.max(0).min(end)  -- symbolic UOp arithmetic
        let zero = UOp::index_const(0);
        let clamped = cut.try_max(&zero).ok()?;
        let count = uop_min(&clamped, end)?;
        let count_casted = count.cast(true_val.dtype());
        return count_casted.try_mul(true_val).ok();
    }

    // Pattern 2b: where(r >= lower, val, 0) → (end - lower).max(0).min(end) * val
    // Handles Ge directly (bound from below), equivalent to Pattern 1 with inverted condition.
    if let Some(lower) = extract_ge_lower_bound(cond, range)
        && is_const_zero(false_val)
        && no_range(true_val)
        && no_range(&lower)
    {
        let zero = UOp::index_const(0);
        let diff = end.try_sub(&lower).ok()?;
        let non_negative = diff.try_max(&zero).ok()?;
        let count = uop_min(&non_negative, end)?;
        let count_casted = count.cast(true_val.dtype());
        return count_casted.try_mul(true_val).ok();
    }

    // Pattern 2c: where(r >= lower, 0, val) → lower.max(0).min(end) * val
    // Inverted Ge: value when condition is FALSE (r < lower).
    if let Some(lower) = extract_ge_lower_bound(cond, range)
        && is_const_zero(true_val)
        && no_range(false_val)
        && no_range(&lower)
    {
        let zero = UOp::index_const(0);
        let clamped = lower.try_max(&zero).ok()?;
        let count = uop_min(&clamped, end)?;
        let count_casted = count.cast(false_val.dtype());
        return count_casted.try_mul(false_val).ok();
    }

    // Pattern 3: where(idx != r, 0, expr) — NE gated collapse
    // Pattern 3b: where(idx == r, expr, 0) — EQ gated collapse (Svod-specific)
    //
    // Both collapse to: where(in_bounds, expr[r:=idx.valid(v)], 0)
    // NE: idx != r with zero in true_val, expression in false_val
    // EQ: idx == r with expression in true_val, zero in false_val
    //
    // Which side carries the range decides the orientation. `no_range` cannot
    // decide it: it asks whether a node reaches *any* range, and a gather's
    // index reaches the output ranges by construction — it is a different index
    // per output element — so neither side is range-free, both tests fail and
    // the arm gives up on the very shape it was written for. `solve_for_range`
    // then moves whatever wraps the range over onto the index.
    //
    // A shape that is not NE/EQ over this range must FALL THROUGH to Pattern 4:
    // an `And` condition is exactly what the two-sided rule below is written for.
    if let Some((lhs, rhs, expr)) = match cond.op() {
        Op::Binary(BinaryOp::Ne, lhs, rhs) if is_const_zero(true_val) => Some((lhs, rhs, false_val)),
        Op::Binary(BinaryOp::Eq, lhs, rhs) if is_const_zero(false_val) => Some((lhs, rhs, true_val)),
        _ => None,
    } {
        // Exactly one side may carry the range; with it on both there is no
        // single step to pick out, and with it on neither there is nothing to
        // pick it out of.
        let oriented = match (reaches(lhs, range), reaches(rhs, range)) {
            (false, true) => Some((lhs, rhs)),
            (true, false) => Some((rhs, lhs)),
            _ => None,
        };
        if let Some((idx, cmp)) = oriented
            && let Some(idx) = solve_for_range(idx, cmp, range)
        {
            return gated_collapse_core(&idx, range, end, expr);
        }
    }

    // Pattern 4: Two-sided bounds
    // where((r >= lower) & (r < upper), val, 0) → (upper.min(end) - lower.max(0)).max(0).min(end) * val
    // Handles two AST representations:
    //   A: ((r < lower).logical_not() & (r < upper)) - NOT(LT) form
    //   B: ((r >= lower) & (r < upper)) - direct GE form
    if let Op::Binary(BinaryOp::And, lhs_cond, rhs_cond) = cond.op()
        && is_const_zero(false_val)
        && no_range(true_val)
    {
        // Try to extract (r >= lower) from lhs_cond - either NOT(LT) or GE form
        let lower_bound = extract_ge_lower_bound(lhs_cond, range).or_else(|| extract_ge_lower_bound(rhs_cond, range));

        // Try to extract (r < upper) from rhs_cond or lhs_cond
        let upper_bound = extract_lt_upper_bound(rhs_cond, range).or_else(|| extract_lt_upper_bound(lhs_cond, range));

        if let (Some(lower), Some(upper)) = (lower_bound, upper_bound)
            && no_range(&lower)
            && no_range(&upper)
        {
            // (upper.min(end) - lower.max(0)).max(0).min(end) * val
            let zero = UOp::index_const(0);
            let clamped_upper = uop_min(&upper, end)?;
            let clamped_lower = lower.try_max(&zero).ok()?;
            let diff = clamped_upper.try_sub(&clamped_lower).ok()?;
            let non_negative = diff.try_max(&zero).ok()?;
            let count = uop_min(&non_negative, end)?;
            let count_casted = count.cast(true_val.dtype());
            return count_casted.try_mul(true_val).ok();
        }
    }

    None
}

/// Extract lower bound from (r >= lower) condition.
/// Handles both NOT(r < lower) and (r >= lower) forms.
fn extract_ge_lower_bound(cond: &Arc<UOp>, range: &Arc<UOp>) -> Option<Arc<UOp>> {
    // Form A: NOT(r < lower)
    if let Op::Unary(UnaryOp::Not, lt_cond) = cond.op()
        && let Op::Binary(BinaryOp::Lt, lt_lhs, lower) = lt_cond.op()
        && Arc::ptr_eq(lt_lhs, range)
    {
        return Some(lower.clone());
    }
    // Form B: r >= lower (represented as NOT(r < lower) or Ge(r, lower))
    if let Op::Binary(BinaryOp::Ge, ge_lhs, lower) = cond.op()
        && Arc::ptr_eq(ge_lhs, range)
    {
        return Some(lower.clone());
    }
    None
}

/// Extract upper bound from (r < upper) condition.
fn extract_lt_upper_bound(cond: &Arc<UOp>, range: &Arc<UOp>) -> Option<Arc<UOp>> {
    if let Op::Binary(BinaryOp::Lt, lt_lhs, upper) = cond.op()
        && Arc::ptr_eq(lt_lhs, range)
    {
        return Some(upper.clone());
    }
    None
}

/// Try to collapse a REDUCE when a scalar PARAM can be factored out.
///
/// Pattern: (PARAM & y).where(c, 0).reduce(ADD) → y.where(c, 0).reduce(ADD) * PARAM.cast(c.dtype)
fn try_param_factor(src: &Arc<UOp>, ranges: &SmallVec<[Arc<UOp>; 4]>) -> Option<Arc<UOp>> {
    let Op::Ternary(svod_ir::TernaryOp::Where, cond, true_val, false_val) = src.op() else {
        return None;
    };
    if !is_const_zero(false_val) {
        return None;
    }

    // Match AND(PARAM, y) or AND(y, PARAM).
    let Op::Binary(BinaryOp::And, and_lhs, and_rhs) = cond.op() else {
        return None;
    };

    let is_alu_param = |u: &Arc<UOp>| matches!(u.op(), Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none());
    let (define_var, other) = if is_alu_param(and_lhs) {
        (and_lhs.clone(), and_rhs.clone())
    } else if is_alu_param(and_rhs) {
        (and_rhs.clone(), and_lhs.clone())
    } else {
        return None;
    };

    // Build: other.where(c, 0).reduce(ADD) * DEFINE_VAR.cast(c.dtype)
    let inner_where = UOp::try_where(other, true_val.clone(), false_val.clone()).ok()?;
    let inner_reduce = inner_where.reduce(ranges.clone(), ReduceOp::Add);
    let casted_var = define_var.cast(true_val.dtype());
    inner_reduce.try_mul(&casted_var).ok()
}

/// Arithmetic lifting for comparisons.
///
/// Lifts operations out of Lt comparisons when they don't depend on ranges:
/// - (x + y) < c → x < (c - y) when y, c are range-free
/// - (x * y) < c → x < ceil(c/y) when y > 0, y, c range-free
///
/// Also handles `.or_casted()` variants where lhs is wrapped in a CAST:
/// - Cast(x + y) < c → x < (c.cast(inner_dtype) - y)
/// - Cast(x * y) < c → x < ceil(c.cast(inner_dtype)/y)
fn try_lift_arithmetic_from_lt(cond: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Binary(BinaryOp::Lt, lhs, rhs) = cond.op() else {
        return None;
    };

    // Both rhs must be range-free for lifting
    if !no_range(rhs) {
        return None;
    }

    // Unwrap optional CAST to get the inner expression (or_casted pattern).
    // When CAST is present, we need to cast the rhs constant to the inner dtype.
    let (inner_lhs, effective_rhs) = if let Op::Cast(ops::Cast { src, .. }) = lhs.op() {
        let inner_dtype = src.dtype();
        let casted_rhs = rhs.cast(inner_dtype);
        (src.as_ref(), casted_rhs)
    } else {
        (lhs.as_ref(), rhs.clone())
    };

    // Pattern: (x + y) < c → x < (c - y). ADD is commutative, so try both
    // operand orders: tinygrad's UPat matches commutative sources in either
    // position (`(UPat.var("x")+UPat.var("y")) < UPat.var("c")`,
    // `codegen/simplify.py:101`), and morok's canonical ordering puts the
    // range-free operand first whenever it sorts below the RANGE.
    if let Op::Binary(BinaryOp::Add, x, y) = inner_lhs.op() {
        if no_range(y) {
            let new_rhs = effective_rhs.try_sub(y).ok()?;
            return x.try_cmplt(&new_rhs).ok();
        }
        if no_range(x) {
            let new_rhs = effective_rhs.try_sub(x).ok()?;
            return y.try_cmplt(&new_rhs).ok();
        }
    }

    // Pattern: (x * y) < c → x < ceil(c/y) when y > 0, either operand order.
    if let Op::Binary(BinaryOp::Mul, x, y) = inner_lhs.op() {
        let ceil_div = |num: &Arc<UOp>, den: &Arc<UOp>| -> Option<Arc<UOp>> {
            // Check den > 0 via vmin.
            let ConstValue::Int(den_min) = den.vmin() else { return None };
            if *den_min <= 0 {
                return None;
            }
            // ceil(c/den) = (c + den - 1) / den
            let one = UOp::index_const(1);
            num.try_add(den).ok()?.try_sub(&one).ok()?.try_div(den).ok()
        };
        if no_range(y)
            && let Some(new_rhs) = ceil_div(&effective_rhs, y)
        {
            return x.try_cmplt(&new_rhs).ok();
        }
        if no_range(x)
            && let Some(new_rhs) = ceil_div(&effective_rhs, x)
        {
            return y.try_cmplt(&new_rhs).ok();
        }
    }

    None
}

/// Arithmetic lifting for EQ comparisons (Svod-specific).
///
/// Isolates range-containing operands from arithmetic in EQ conditions:
/// - (x + y) == c → x == (c - y) or y == (c - x)
/// - (x - y) == c → x == (c + y) or y == (x - c)
/// - Cast(x ± y) == c → same with c cast to inner dtype
fn try_lift_arithmetic_from_eq(cond: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Binary(BinaryOp::Eq, raw_lhs, raw_rhs) = cond.op() else { return None };

    // Normalize: range-containing side on lhs, range-free on rhs.
    // The pattern `Eq[_, c] if no_range(c)` matches commutatively, but
    // cond.op() returns operands in storage order which may differ.
    let (lhs, rhs) = if no_range(raw_rhs) {
        (raw_lhs, raw_rhs)
    } else if no_range(raw_lhs) {
        (raw_rhs, raw_lhs)
    } else {
        return None;
    };

    // Unwrap optional CAST, adjusting rhs to inner dtype
    let (inner_lhs, effective_rhs) = if let Op::Cast(ops::Cast { src, .. }) = lhs.op() {
        (src.as_ref(), rhs.cast(src.dtype()))
    } else {
        (lhs.as_ref(), rhs.clone())
    };

    match inner_lhs.op() {
        Op::Binary(BinaryOp::Add, x, y) if no_range(y) => x.try_cmpeq(&effective_rhs.try_sub(y).ok()?).ok(),
        Op::Binary(BinaryOp::Add, x, y) if no_range(x) => y.try_cmpeq(&effective_rhs.try_sub(x).ok()?).ok(),
        Op::Binary(BinaryOp::Sub, x, y) if no_range(y) => x.try_cmpeq(&effective_rhs.try_add(y).ok()?).ok(),
        Op::Binary(BinaryOp::Sub, x, y) if no_range(x) => y.try_cmpeq(&x.try_sub(&effective_rhs).ok()?).ok(),
        _ => None,
    }
}

/// Arithmetic lifting for Ge comparisons.
///
/// Lifts operations out of Ge comparisons when they don't depend on ranges:
/// - (x + y) >= c → x >= (c - y) when y, c are range-free
fn try_lift_arithmetic_from_ge(cond: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Binary(BinaryOp::Ge, lhs, rhs) = cond.op() else {
        return None;
    };

    if !no_range(rhs) {
        return None;
    }

    // (x + y) >= c → x >= (c - y) when y is range-free
    if let Op::Binary(BinaryOp::Add, x, y) = lhs.op() {
        if no_range(y) {
            return x.try_cmpge(&rhs.try_sub(y).ok()?).ok();
        }
        if no_range(x) {
            return y.try_cmpge(&rhs.try_sub(x).ok()?).ok();
        }
    }

    None
}

/// Pattern matcher for load collapse optimizations.
///
/// Collapses REDUCE operations with gated/conditional loads.
///
/// Key optimizations:
/// 1. Bounded sum reduction: `sum(1 for i in range(n) if i >= k)` → `n - k`
/// 2. Two-sided bounds: `sum(1 for i in range(n) if lower <= i < upper)` → clamped count
/// 3. Gated load collapse: `sum(where(idx == r, val, 0))` → direct indexed load
/// 4. Arithmetic lifting: push comparisons through arithmetic operations
/// 5. DEFINE_VAR factoring: `(dv & y).where(c, 0).reduce(ADD)` → `y.where(c,0).reduce(ADD) * dv`
/// 6. MUL casted bool: `x * gate:bool.cast()` → `gate.where(x, 0)`
/// 7. NE lifting: `(x + y) != c` → `x != (c - y)`
/// 8. Index overflow protection: `(x:index + y) < c` → `x < (c - y)` when x has loads
pub fn pm_load_collapse() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        // Match REDUCE(ADD) with a single range → full reduce_load_collapse algorithm.
        //
        // Goes straight to reduce_load_collapse (gated toposort + DEFINE_VAR
        // substitution). All arithmetic lifting, NE lifting, and .or_casted()
        // patterns live inside the inner matcher
        // (build_reduce_load_collapse_matcher), not at this level.
        _reduce @ Reduce { src, ranges, reduce_op, num_axes }
            if ranges.len() == 1 && *reduce_op == ReduceOp::Add && *num_axes == 0
            => {
                super::transforms::reduce_load_collapse(src, ranges)
            },

        // Index overflow undo rule: (x:index + y) < c → x < (c - y)
        // Only when x has loads but y, c don't — prevents overflow on loaded indices.
        // This undoes the arithmetic lifting that pm_reduce_load_collapse may have
        // applied when the lifted form risks integer overflow on loaded values.
        Lt(Add(x, y), c)
            if x.dtype() == DType::WeakInt && !no_load(x) && no_load(y) && no_load(c)
            => {
                let new_c = c.try_sub(y).ok()?;
                x.try_cmplt(&new_c).ok()
            },
    }
}

// ============================================================================
// PM_REDUCE_COLLAPSE - Inner patterns for reduce_collapse loop
// ============================================================================
// Used inside reduce_collapse's per-range iteration to algebraically
// eliminate the synthetic REDUCE node.

/// Inner pattern matcher used inside `reduce_collapse` per-range iteration.
///
/// Combines reduce-specific algebraic patterns with full symbolic simplification.
/// Does NOT include a recursive `reduce_collapse` call (would infinite-loop).
pub fn build_reduce_collapse_matcher() -> &'static TypedPatternMatcher<()> {
    static CACHED: std::sync::LazyLock<TypedPatternMatcher<()>> =
        // Pair the reduce-specific patterns with full `symbolic`.
        std::sync::LazyLock::new(|| {
            reduce_collapse_inner_patterns() + crate::symbolic::symbolic() + crate::symbolic::pm_fold_cast_const()
        });
    &CACHED
}

/// Extended pattern matcher for `reduce_load_collapse`.
///
/// Combines the basic `reduce_collapse` patterns with NE lifting (only needed
/// in the per-kernel `reduce_load_collapse` path).
///
/// Does NOT include the REDUCE→reduce_load_collapse call to avoid infinite recursion.
pub fn build_reduce_load_collapse_matcher() -> &'static TypedPatternMatcher<()> {
    static CACHED: std::sync::LazyLock<TypedPatternMatcher<()>> =
        std::sync::LazyLock::new(|| build_reduce_collapse_matcher() + ne_lifting_patterns());
    &CACHED
}

/// NE lifting patterns for the extended `reduce_load_collapse` path.
///
/// NE lifting is only needed here (not in the basic mega-pass matcher).
///
/// Note: index overflow undo lives in `pm_load_collapse()` (the outer matcher),
/// not here — inside reduce_collapse, external inputs are DEFINE_VARs with no loads,
/// so `!no_load(x)` never matches.
fn ne_lifting_patterns() -> TypedPatternMatcher<()> {
    crate::patterns! {
        // NE lifting: (x + y) != c → x != (c - y) when no_range(y, c)
        Ne(Add(x, y), c) if no_range(y) && no_range(c) => {
            let new_c = c.try_sub(y).ok()?;
            x.try_cmpne(&new_c).ok()
        },

        // .or_casted() NE: Cast(x + y) != c → x != (c.cast(inner_dtype) - y)
        Ne(Cast { src: inner, .. }, c) if no_range(c) => {
            let Op::Binary(BinaryOp::Add, x, y) = inner.op() else { return None };
            if !no_range(y) { return None; }
            let casted_c = c.cast(inner.dtype());
            let new_c = casted_c.try_sub(y).ok()?;
            x.try_cmpne(&new_c).ok()
        },
    }
}

/// Reduce-specific algebraic patterns for use inside `reduce_collapse`.
///
/// 1. reduce_unparented: remove ranges not used by src
/// 2. Lt lifting: (x+y) < c → x < (c-y), (x*y) < c → x < ceil(c/y), including .or_casted()
/// 3. Ge lifting: (x+y) >= c → x >= (c-y)
/// 4. Distributive: (x+y).reduce(ADD) → x.reduce(ADD) + y.reduce(ADD)
/// 5. Bound-from-below/above/two-sided/gated collapse on REDUCE(ADD)
/// 6. DEFINE_VAR factoring: (dv & y).where(c,0).reduce(ADD)
/// 7. MUL casted bool: x * gate:bool.cast() → gate.where(x, 0)
/// 8. EQ lifting: (x+y)==c → x==(c-y), Svod-specific for gather's EQ pattern
fn reduce_collapse_inner_patterns() -> TypedPatternMatcher<()> {
    // Start with reduce_unparented (shared with pm_reduce_simplify)
    pm_reduce_unparented().with_context()
    // Lt/Ge arithmetic lifting: push range-free operands to rhs of comparison.
    // Handles direct and .or_casted() (CAST-wrapped) Add/Mul forms.
    + crate::patterns! {
        cond @ Lt(_, rhs) if no_range(rhs) => try_lift_arithmetic_from_lt(cond),
        cond @ Ge(_, rhs) if no_range(rhs) => try_lift_arithmetic_from_ge(cond),

        // Distributive: (x+y).reduce(ADD) → x.reduce(ADD) + y.reduce(ADD)
        Reduce { src, ranges, reduce_op, num_axes } if *reduce_op == ReduceOp::Add => {
            let Op::Binary(BinaryOp::Add, x, y) = src.op() else { return None };
            let x_reduced = x.reduce_with_num_axes(ranges.clone(), ReduceOp::Add, *num_axes);
            let y_reduced = y.reduce_with_num_axes(ranges.clone(), ReduceOp::Add, *num_axes);
            x_reduced.try_add(&y_reduced).ok()
        },

        // Bound patterns + DEFINE_VAR factoring on REDUCE(ADD) with single range
        // These match the synthetic REDUCE created by reduce_collapse's per-range iteration.
        // Patterns: bound-from-below, bound-from-above, two-sided, gated NE/EQ collapse.
        reduce @ Reduce { src, ranges, reduce_op, num_axes }
            if !ranges.is_empty() && *reduce_op == ReduceOp::Add && *num_axes == 0
            => {
                try_reduce_collapse(reduce, src, ranges, ReduceOp::Add)
                    .or_else(|| try_param_factor(src, ranges))
            },

        // MUL casted bool: x * gate:bool.cast() → gate.where(x, 0)
        Mul[x, Cast { src: gate, .. }] if gate.dtype() == DType::Bool => {
            let zero = UOp::const_(x.dtype(), ConstValue::zero(x.dtype().base()));
            UOp::try_where(gate.clone(), x.clone(), zero).ok()
        },

        // EQ lifting: isolate range-containing operands from arithmetic in EQ conditions.
        // Needed because gather emits EQ directly. Belongs in this matcher
        // because the mega-pass buffer_folding inlines arange constants
        // beforehand; per-kernel pm_load_collapse runs before buffer_folding
        // and can't see through LOADs.
        cond @ Eq[_, c] if no_range(c) => try_lift_arithmetic_from_eq(cond),
    }
}

// ============================================================================
// LATE DECOMPOSITION PATTERNS (get_late_rewrite_patterns)
// ============================================================================

/// MOD → AND optimization for power-of-two modulus.
///
/// x % 2^n → x & (2^n - 1)
///
/// This is a common optimization that converts expensive modulo operations
/// into cheap bitwise AND when the divisor is a power of two.
/// Only applies to integer types.
pub fn pm_mod_to_and() -> &'static TypedPatternMatcher<()> {
    use svod_ir::types::ConstValue;
    crate::cached_patterns! {
        // x % c where c is power of two → x & (c - 1)
        FloorMod(x, _c @const(c_val)) => {
            // Only apply to integer types
            if !x.dtype().is_int() { return None; }

            let n = match c_val {
                ConstValue::Int(v) if v > 0 && (v as u64).is_power_of_two() => v,
                ConstValue::UInt(v) if v > 0 && v.is_power_of_two() => v as i64,
                _ => return None,
            };
            // x % n → x & (n - 1)
            let mask = UOp::const_(x.dtype(), ConstValue::Int(n - 1));
            x.try_and_op(&mask).ok()
        },
    }
}

/// Multiply → Shift optimization for power-of-two multiplier.
///
/// x * 2^n → x << n
///
/// Converts multiplication by power-of-two into left shift.
/// Only applies to integer types.
pub fn pm_mul_to_shl() -> &'static TypedPatternMatcher<()> {
    use svod_ir::types::ConstValue;
    crate::cached_patterns! {
        // x * c where c is power of two → x << log2(c)
        // Note: Only applies to integer types, but we check inside the closure
        Mul[x, _c @const(c_val)] => {
            // Only apply to integer types
            if !x.dtype().is_int() { return None; }

            let (n, shift) = match c_val {
                ConstValue::Int(v) if v > 0 && (v as u64).is_power_of_two() => (v as u64, (v as u64).trailing_zeros()),
                ConstValue::UInt(v) if v > 0 && v.is_power_of_two() => (v, v.trailing_zeros()),
                _ => return None,
            };
            if n == 1 { return Some(x.clone()); } // x * 1 → x (handled elsewhere but be safe)
            let shift_amount = UOp::const_(x.dtype(), ConstValue::Int(shift as i64));
            x.try_shl_op(&shift_amount).ok()
        },
    }
}

/// Negation decompositions:
/// - x * -1 → NEG(x)
/// - x + NEG(y) → SUB(x, y)
pub fn pm_neg_from_mul() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        // x * -1 → NEG(x)
        // Uses raw UOp::new to avoid infinite loop since .neg() now produces MUL(x, -1).
        Mul[x, _c @const(c_val)] if c_val.is_neg_one() => {
            let dtype = x.dtype();
            Some(UOp::new(Op::Unary(UnaryOp::Neg, x.clone()), dtype))
        },
        // x + NEG(y) → SUB(x, y).
        Add[x, Neg(y)] => UOp::alu(BinaryOp::Sub, x.clone(), y.clone()),
    }
}

/// Threefry2x32 PRNG decomposition.
///
/// No real hardware supports THREEFRY natively. Decomposes to uint32 arithmetic:
/// split 64-bit x/key into halves, apply 5 rounds of add-rotate-xor mixing,
/// recombine to uint64.
pub fn pm_threefry_decomp() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        Threefry(x, key) if x.dtype() == DType::UInt64 => {
            Some(threefry2x32(x, key))
        },
    }
}

/// Threefry2x32 mixing algorithm (Random123 library).
fn threefry2x32(x: &Arc<UOp>, key: &Arc<UOp>) -> Arc<UOp> {
    let u32_dt = DType::UInt32;
    let u64_dt = DType::UInt64;
    let shift32 = UOp::const_(u64_dt.clone(), ConstValue::Int(32));

    // Split x and key from uint64 to two uint32. Narrowing casts truncate, so
    // `.cast(u32)` / `>> 32` need no mask; this is also the form the uint64
    // pack-cancellation rules in `symbolic_simple` invert.
    let x0 = x.cast(u32_dt.clone());
    let x1 = x.shr(&shift32).cast(u32_dt.clone());
    let key0 = key.cast(u32_dt.clone());
    let key1 = key.shr(&shift32).cast(u32_dt.clone());

    // Key schedule: ks = [key1, key0 ^ key1 ^ 0x1BD11BDA, key0]
    let skein_const = UOp::const_(u32_dt.clone(), ConstValue::UInt(0x1BD11BDA));
    let ks = [key1.clone(), key0.xor(&key1).xor(&skein_const), key0.clone()];

    let rotations: [[u32; 4]; 2] = [[13, 15, 26, 6], [17, 29, 16, 24]];

    // Initialize: xr = [x0 + ks[2], x1 + ks[0]]
    let mut xr0 = x0.add(&ks[2]);
    let mut xr1 = x1.add(&ks[0]);

    // 5 rounds of mixing
    for i in 0..5u32 {
        for &r in &rotations[i as usize % 2] {
            let new_x0 = xr0.add(&xr1);
            // Barrel rotate: (xr1 << r) + (xr1 >> (32-r)).
            let rot_left = xr1.shl(&UOp::const_(u32_dt.clone(), ConstValue::Int(r as i64)));
            let rot_right = xr1.shr(&UOp::const_(u32_dt.clone(), ConstValue::Int(32 - r as i64)));
            let rotated = rot_left.add(&rot_right);
            xr1 = new_x0.xor(&rotated);
            xr0 = new_x0;
        }
        // Key injection
        xr0 = xr0.add(&ks[i as usize % 3]);
        let round_const = UOp::const_(u32_dt.clone(), ConstValue::UInt((i + 1) as u64));
        xr1 = xr1.add(&ks[(i as usize + 1) % 3]).add(&round_const);
    }

    // Recombine: (xr1.cast(u64) << 32) | xr0.cast(u64)
    xr1.cast(u64_dt.clone()).shl(&shift32).or_(&xr0.cast(u64_dt))
}

/// DeMorgan's law: NOT(x) & NOT(y) → NOT(x | y).
///
/// Reduces one NOT instruction by factoring out negation.
pub fn pm_demorgan() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        And[Not(x), Not(y)] if x.dtype().is_bool() => x.or_(y).not(),
    }
}

/// SHL+ADD → MULACC fusion.
///
/// After `pm_mul_to_shl` converts `x * 2^n → x << n`, expressions like
/// `(x << n) + c` can fuse to MULACC(x, 2^n, c) for backends with FMA.
pub fn pm_shl_add_to_mulacc() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        Add[Shl(x, _n @const(nv)), c] => {
            let ConstValue::Int(v) = nv else { return None };
            if !(0..64).contains(&v) { return None; }
            let multiplier = UOp::const_(x.dtype(), ConstValue::Int(1i64 << v));
            UOp::try_mulacc(x.clone(), multiplier, c.clone()).ok()
        },
    }
}

/// Divide → Shift optimization for power-of-two divisor.
///
/// For unsigned integers: x // 2^n → x >> n
/// For signed integers: (x + (x<0).where(n-1, 0)) >> n
///   (handles rounding towards zero for negative dividends)
///
/// Shifts are typically 2-5x faster than divisions on modern CPUs and GPUs.
pub fn pm_div_to_shr() -> &'static TypedPatternMatcher<()> {
    use svod_ir::types::ConstValue;
    use svod_ir::uop::cached_property::CachedProperty;
    use svod_ir::uop::properties::VminVmaxProperty;

    crate::cached_patterns! {
        // C-style x / c where c is power of two.
        CDiv(x, _c @const(c_val)) => {
            // Only apply to integer types
            if !x.dtype().is_int() { return None; }

            let n = match c_val {
                ConstValue::Int(v) if v > 0 && (v as u64).is_power_of_two() => v,
                ConstValue::UInt(v) if v > 0 && v.is_power_of_two() => v as i64,
                _ => return None,
            };

            // Skip trivial case: x // 1 → x (handled elsewhere)
            if n == 1 { return None; }

            let shift = (n as u64).trailing_zeros() as i64;
            let shift_const = UOp::const_(x.dtype(), ConstValue::Int(shift));

            // Check if x is always non-negative via vmin/vmax analysis
            let (vmin, _) = VminVmaxProperty::get(x);
            let is_non_negative = match vmin {
                ConstValue::Int(v) => *v >= 0,
                ConstValue::UInt(_) => true, // unsigned always non-negative
                _ => false,
            };

            if is_non_negative || x.dtype().is_unsigned() {
                // Unsigned case: x // 2^n → x >> n
                x.try_shr_op(&shift_const).ok()
            } else {
                // Signed case with potentially negative dividend:
                // (x + (x < 0).where(n - 1, 0)) >> n
                // This bias corrects for rounding towards zero
                let zero = UOp::const_(x.dtype(), ConstValue::Int(0));
                let bias = UOp::const_(x.dtype(), ConstValue::Int(n - 1));
                let x_neg = x.try_cmplt(&zero).ok()?;
                let adjustment = UOp::try_where(x_neg, bias, zero).ok()?;
                let adjusted = x.try_add(&adjustment).ok()?;
                adjusted.try_shr_op(&shift_const).ok()
            }
        },
    }
}

/// MAX decomposition: MAX(a, b) → (a < b).where(b, a)
///
/// For backends that don't have native MAX support, decompose into
/// comparison and conditional select.
pub fn pm_max_decomposition() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        // MAX(a, b) → (a < b).where(b, a)
        Max(a, b) => {
            let cond = a.try_cmplt(b).ok()?;
            UOp::try_where(cond, b.clone(), a.clone()).ok()
        },
    }
}

/// SQRT decomposition: SQRT(x) → POW(x, 0.5)
///
/// For backends that don't have native SQRT support, decompose into
/// power operation with exponent 0.5.
pub fn pm_sqrt_decomposition() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        // SQRT(x) → POW(x, 0.5)
        Sqrt(x) if x.dtype().is_float() => {
            let half = UOp::const_(x.dtype(), svod_ir::types::ConstValue::Float(0.5));
            x.try_pow(&half).ok()
        },
    }
}

/// ERF decomposition using Abramowitz & Stegun 7.1.26 polynomial approximation.
///
/// Decomposed here because svod keeps Erf as a UOp. `@llvm.erf` is a libcall
/// intrinsic (not a native hardware op like sqrt/fabs), so it requires libm
/// linkage which the LLVM JIT doesn't provide.
///
/// erf(x) = sign(x) * (1 - t * P(t) * exp(-x²))
/// where t = 1 / (1 + 0.3275911 * |x|)
///       P(t) = polyN(t, [1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592])
pub fn pm_erf_decomposition() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        Erf(x) if x.dtype().is_float() => {
            let dt = x.dtype();
            let f = |v: f64| UOp::const_(dt.clone(), ConstValue::Float(v));

            let abs_x = x.abs();
            let t = f(1.0).try_div(&f(1.0).try_add(&f(0.3275911).try_mul(&abs_x).ok()?).ok()?).ok()?;

            // Horner's method: ((((a4*t + a3)*t + a2)*t + a1)*t + a0)
            let poly = f(1.061405429);
            let poly = poly.try_mul(&t).ok()?.try_add(&f(-1.453152027)).ok()?;
            let poly = poly.try_mul(&t).ok()?.try_add(&f(1.421413741)).ok()?;
            let poly = poly.try_mul(&t).ok()?.try_add(&f(-0.284496736)).ok()?;
            let poly = poly.try_mul(&t).ok()?.try_add(&f(0.254829592)).ok()?;

            // exp(-x²)
            let exp_val = x.square().neg().try_exp().ok()?;

            // sign(x) * (1 - t * poly * exp(-x²))
            let inner = f(1.0).try_sub(&t.try_mul(&poly).ok()?.try_mul(&exp_val).ok()?).ok()?;
            x.sign().try_mul(&inner).ok()
        },
    }
}

/// FDIV → MUL reciprocal optimization for floating-point division by constant.
///
/// x / c → x * (1/c) for float constants
///
/// Multiplication is typically 2-3x faster than division on modern CPUs and GPUs.
/// Guards against divide by zero (leaves as FDIV to preserve IEEE 754 semantics).
pub fn pm_fdiv_to_mul() -> &'static TypedPatternMatcher<()> {
    use svod_ir::types::ConstValue;
    crate::cached_patterns! {
        // x / c → x * (1/c) for float constants
        Fdiv(x, _c @const(c_val)) => {
            // Only apply to float types
            if !x.dtype().is_float() { return None; }

            let f = match c_val {
                ConstValue::Float(v) => v,
                _ => return None,
            };

            // Guard against divide by zero - leave as FDIV to preserve IEEE 754 semantics
            if f == 0.0 { return None; }

            // Also guard against denormalized reciprocals that could cause precision loss
            let recip = 1.0 / f;
            if !recip.is_finite() { return None; }

            let recip_const = UOp::const_(x.dtype(), ConstValue::Float(recip));
            x.try_mul(&recip_const).ok()
        },
    }
}

/// Comparison negation patterns for integers.
///
/// Simplify negated comparisons into equivalent direct comparisons:
/// - !(x < c) → (c-1) < x  (for integers)
/// - !(c < x) → x < (c+1)  (for integers)
/// - (c1 < x) & (x < c2) → x == (c1+1)  (when c2 == c1+2, range compression)
pub fn pm_comparison_negations() -> &'static TypedPatternMatcher<()> {
    use svod_ir::types::ConstValue;

    crate::cached_patterns! {
        // !(x < c) → (c-1) < x for integers
        // When x >= c, that's equivalent to (c-1) < x
        Not(Lt(x, _c @const(c_val))) if x.dtype().is_int() => {
            let v = match c_val {
                ConstValue::Int(v) => v,
                ConstValue::UInt(v) => i64::try_from(v).ok()?,
                _ => return None,
            };
            // Guard against underflow
            let c_minus_1 = v.checked_sub(1)?;
            let c_minus_1_const = UOp::const_(x.dtype(), ConstValue::Int(c_minus_1));
            c_minus_1_const.try_cmplt(x).ok()
        },

        // !(c < x) → x < (c+1) for integers
        // When x <= c, that's equivalent to x < (c+1)
        Not(Lt(_c @const(c_val), x)) if x.dtype().is_int() => {
            let v = match c_val {
                ConstValue::Int(v) => v,
                ConstValue::UInt(v) => i64::try_from(v).ok()?,
                _ => return None,
            };
            // Guard against overflow
            let c_plus_1 = v.checked_add(1)?;
            let c_plus_1_const = UOp::const_(x.dtype(), ConstValue::Int(c_plus_1));
            x.try_cmplt(&c_plus_1_const).ok()
        },

        // Range compression: (c1 < x) & (x < c2) → x == (c1+1) when c2 == c1+2
        // When x is in the open interval (c1, c2) and c2 - c1 == 2, x must be c1+1
        And[Lt(_c1 @const(c1_val), x), Lt(x2, _c2 @const(c2_val))]
            if x.dtype().is_int() && Arc::ptr_eq(x, x2)
            => {
                let v1 = match c1_val {
                    ConstValue::Int(v) => v,
                    ConstValue::UInt(v) => i64::try_from(v).ok()?,
                    _ => return None,
                };
                let v2 = match c2_val {
                    ConstValue::Int(v) => v,
                    ConstValue::UInt(v) => i64::try_from(v).ok()?,
                    _ => return None,
                };
                // Only apply if c2 == c1 + 2 (single value in range)
                if v2 != v1.checked_add(2)? { return None; }
                let target = UOp::const_(x.dtype(), ConstValue::Int(v1 + 1));
                x.try_cmpeq(&target).ok()
            },

        // x*-1 < c → -c < x for integers.
        // When comparing a negated value with a constant, flip the comparison.
        Lt(Mul(x, _neg1 @const(neg_val)), _c @const(c_val)) if x.dtype().is_int() => {
            // Check that we're multiplying by -1
            if !matches!(neg_val, ConstValue::Int(-1)) { return None; }

            let c = match c_val {
                ConstValue::Int(v) => v,
                ConstValue::UInt(v) => i64::try_from(v).ok()?,
                _ => return None,
            };

            // x*-1 < c → -c < x
            let neg_c = c.checked_neg()?;
            let neg_c_const = UOp::const_(x.dtype(), ConstValue::Int(neg_c));
            neg_c_const.try_cmplt(x).ok()
        },

        // x*-1 < y*c → y*(-c) < x for integers
        // When comparing negated x with scaled y, flip and negate scale
        Lt(Mul(x, _neg1 @const(neg_val)), Mul(y, _c @const(c_val))) if x.dtype().is_int() => {
            // Check that we're multiplying x by -1
            if !matches!(neg_val, ConstValue::Int(-1)) { return None; }

            let c = match c_val {
                ConstValue::Int(v) => v,
                ConstValue::UInt(v) => i64::try_from(v).ok()?,
                _ => return None,
            };

            // x*-1 < y*c → y*(-c) < x
            let neg_c = c.checked_neg()?;
            let neg_c_const = UOp::const_(y.dtype(), ConstValue::Int(neg_c));
            let y_neg_c = y.try_mul(&neg_c_const).ok()?;
            y_neg_c.try_cmplt(x).ok()
        },
    }
}

// ============================================================================

// ============================================================================
// PM_HALF_BF16_CAST - same-width float casts route via f32
// ============================================================================

/// f16 ↔ bf16 have equal width, so no single LLVM `fptrunc`/`fpext` exists —
/// and a plain `cast(f32).cast(dst)` chain gets folded back by the cast
/// simplifier (rewrite ping-pong). Route the f32→bf16 step through bits
/// (round-to-nearest-even on the high 16, tinygrad `pm_manual_bf16_cast`).
pub fn pm_half_bf16_cast() -> &'static TypedPatternMatcher<()> {
    use svod_dtype::{DType, ScalarDType};
    crate::cached_patterns! {
        x @ Cast { src, .. } if x.dtype().base().is_float()
            && src.dtype().base().is_float()
            && x.dtype().base().bytes() == src.dtype().base().bytes()
            && x.dtype().base() != src.dtype().base()
        => {
            let vc = x.dtype().vcount();
            let (f32t, u32t) = (
                DType::Scalar(ScalarDType::Float32).vec(vc).expect("scalar dtype is vectorizable"),
                DType::Scalar(ScalarDType::UInt32).vec(vc).expect("scalar dtype is vectorizable"),
            );
            let u16t = DType::Scalar(ScalarDType::UInt16).vec(vc).expect("scalar dtype is vectorizable");
            if x.dtype().base() == ScalarDType::BFloat16 {
                // f16 → f32 (fpext) → RNE-round the low 16 bits → bf16 payload.
                let bits = src.cast(f32t).bitcast(u32t.clone());
                let half_bit = bits.shr(&bits.const_like(16)).and_(&bits.const_like(1));
                let rounded = bits.try_add(&half_bit.try_add(&bits.const_like(0x7fff)).ok()?).ok()?;
                Some(rounded.shr(&rounded.const_like(16)).cast(u16t).bitcast(x.dtype()))
            } else {
                // bf16 → f16: widen through bits (u16 << 16 = exact f32), then fptrunc.
                let wide = src.bitcast(u16t).cast(u32t.clone());
                let f = wide.shl(&wide.const_like(16)).bitcast(f32t);
                Some(f.cast(x.dtype()))
            }
        },
    }
}
