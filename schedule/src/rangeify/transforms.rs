//! Consolidated transformation functions for rangeify.
//!
//! This module contains:
//! - Main `rangeify()` entry point
//! - Movement op → STAGE+INDEX transformation helpers
//! - STAGE → STORE conversion
//! - Reduction simplifications (reduce_unparented, reduce_collapse)
//! - Range flattening (flatten_range_impl)
//! - Cycle detection (find_bufs)
//!
//! Consolidated from: transform.rs, bufferize_to_store.rs, reduce_simplify.rs,
//! flatten_range.rs, cycle_detection.rs

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use super::context::RangeifyContext;
use super::indexing::IndexingContext;
use super::kernel::RangeifyBufferContext;
use rustc_hash::FxHashMap;
use smallvec::{SmallVec, smallvec};
use svod_ir::ops;
use svod_ir::shape::Shape;
use svod_ir::uop::{Nodes, SliceMemo};
use svod_ir::{AddrSpace, AxisType, BinaryOp, BufferizeOpts, ConstValue, DType, Op, UOp, UOpKey};

// ============================================================================
// ADD_TAGS
// ============================================================================

/// Context for the add_tags pass.
pub struct AddTagsCtx {
    /// Sequential list of tagged UOps (index = tag value).
    pub uop_list: Vec<Arc<UOp>>,
    /// UOps excluded from tagging (e.g., nodes inside CALL bodies).
    excluded: HashSet<UOpKey>,
}

impl Default for AddTagsCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl AddTagsCtx {
    pub fn new() -> Self {
        Self { uop_list: Vec::new(), excluded: HashSet::new() }
    }
}

/// Ops that should NOT be tagged. MStack/MSelect are handled separately with conditional logic.
fn should_skip_tag(op: &Op) -> bool {
    matches!(
        op,
        Op::Param(..)
            | Op::Const(_)
            | Op::Unique(_)
            | Op::LUnique(_)
            | Op::Bind(..)
            | Op::Call(..)
            | Op::End(..)
            | Op::Range(..)
    ) || op.is_movement()
}

/// Create the add_tags pattern matcher.
///
/// Assigns sequential integer tags `[i]` to each taggable UOp. Tags track which
/// original tensor UOps map to which final kernel outputs through the pipeline.
pub fn add_tags_patterns() -> crate::TypedPatternMatcher<AddTagsCtx> {
    crate::patterns! {
        @context AddTagsCtx;
        x => {
            if x.tag().is_some() || ctx.excluded.contains(&UOpKey(x.clone())) { return None; }
            // Call: exclude call body subgraph from tagging.
            if let Op::Call(ops::Call { body, .. }) = x.op() {
                for u in body.toposort() {
                    ctx.excluded.insert(UOpKey(u));
                }
            }
            if should_skip_tag(x.op()) { return None; }
            // MStack/MSelect: only tag if NOT all sources are PARAM.
            if matches!(x.op(), Op::MStack(..) | Op::MSelect(..))
                && x.op().sources().iter().all(|s| matches!(s.op(), Op::Param(..)))
            {
                return None;
            }
            ctx.uop_list.push(x.clone());
            Some(x.with_tag(smallvec![ctx.uop_list.len() - 1]))
        },
    }
}

fn resolve_single_function(function: &Arc<UOp>) -> svod_ir::Result<Option<Arc<UOp>>> {
    let Op::Function(ops::Function { body, args, info }) = function.op() else {
        return Ok(None);
    };

    if info.precompile {
        return Ok(None);
    }

    // Shared with shape inference: sparse positional slots are valid and
    // substitutions are single-pass so actual argument graphs are not treated
    // as part of the callable implementation.
    let subs = svod_ir::shape::function_param_substitutions(body, args)?;
    let resolved = body.substitute_walk_preserve_calls(&subs).rtag(function.tag().clone()).rorigin(function.origin());
    Ok(Some(resolved))
}

fn resolve_traversal_sources(node: &Arc<UOp>) -> Vec<Arc<UOp>> {
    match node.op() {
        // CALL is an opaque boundary: only rewrite/resolve call arguments.
        Op::Call(ops::Call { args, .. }) => args.iter().cloned().collect(),
        // A precompiled FUNCTION is equally opaque. In particular, do not
        // resolve nested FUNCTIONs from its implementation body.
        Op::Function(ops::Function { args, info, .. }) if info.precompile => args.iter().cloned().collect(),
        // PROGRAM internals are also boundaries in this pass.
        Op::Program(..) => Vec::new(),
        _ => node.op().sources().into_iter().collect(),
    }
}

fn resolve_toposort(root: &Arc<UOp>) -> Vec<Arc<UOp>> {
    let mut visited = HashSet::new();
    let mut result = Vec::new();
    let mut stack = vec![(root.clone(), false)];

    while let Some((node, processed)) = stack.pop() {
        let ptr = Arc::as_ptr(&node);
        if visited.contains(&ptr) {
            continue;
        }

        if processed {
            visited.insert(ptr);
            result.push(node);
            continue;
        }

        stack.push((node.clone(), true));
        let mut children = Vec::new();
        for child in resolve_traversal_sources(&node) {
            if !visited.contains(&Arc::as_ptr(&child)) {
                children.push(child);
            }
        }
        for child in children.into_iter().rev() {
            stack.push((child, false));
        }
    }

    result
}

/// Resolve FUNCTION nodes by substituting PARAM(slot=i) with function arguments.
///
/// Preserves callable/boundary FUNCTION bodies as opaque CALLs and keeps
/// precompile functions intact.
pub(crate) fn resolve_calls(root: Arc<UOp>) -> svod_ir::Result<Arc<UOp>> {
    let topo = resolve_toposort(&root);
    let mut rewritten: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    for node in topo {
        let old_sources = node.op().sources();
        let new_sources: Vec<Arc<UOp>> = old_sources
            .iter()
            .map(|src| rewritten.get(&UOpKey(src.clone())).cloned().unwrap_or_else(|| src.clone()))
            .collect();
        let sources_changed = old_sources.len() == new_sources.len()
            && old_sources.iter().zip(new_sources.iter()).any(|(a, b)| !Arc::ptr_eq(a, b));

        let mut current = if sources_changed { node.with_sources(new_sources) } else { node.clone() };
        if let Some(resolved) = resolve_single_function(&current)? {
            current = resolved;
        }
        // Pinned rangeify extracts only a direct GETTUPLE(TUPLE(...), i).
        current = resolve_gettuple(&current);

        rewritten.insert(UOpKey(node), current);
    }

    Ok(rewritten.remove(&UOpKey(root.clone())).unwrap_or(root))
}

/// Resolve `GETTUPLE(TUPLE(srcs), i)` to `srcs[i]`.
fn resolve_gettuple(node: &Arc<UOp>) -> Arc<UOp> {
    let Op::GetTuple(ops::GetTuple { src, index }) = node.op() else {
        return node.clone();
    };
    let Op::Tuple(ops::Tuple { src: inner }) = src.op() else { return node.clone() };
    inner.get(*index).cloned().unwrap_or_else(|| node.clone())
}

// ============================================================================
// PUBLIC API
// ============================================================================

/// Main rangeify transformation entry point.
///
/// Converts movement operations (RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP)
/// into STAGE + INDEX operations with explicit loop ranges.
pub fn rangeify(sink: Arc<UOp>) -> svod_ir::Result<(Arc<UOp>, RangeifyContext)> {
    let result = rangeify_with_map(sink)?;
    Ok((result.sink, result.context))
}

/// Result of rangeify transformation.
pub struct RangeifyResult {
    /// The transformed sink node
    pub sink: Arc<UOp>,
    /// Context with range information
    pub context: RangeifyContext,
    /// Tagged UOps from add_tags pass (index = tag value).
    /// Used for tag-based becomes_map construction.
    pub uop_list: Vec<Arc<UOp>>,
}

/// Main rangeify transformation entry point with becomes_map tracking.
///
/// Like `rangeify`, but also returns a `becomes_map` that tracks which
/// original nodes were transformed. This is essential for global graph
/// coordination when multiple tensors share subgraphs.
///
/// # Pipeline
///
/// **Pre-stage**: multi_pm + supported-subset validation, then add_tags
/// **Stage 0**: Range assignment (run_rangeify)
/// **Stage 1**: movement_op_patterns (BOTTOM_UP) - Early movement ops
/// **Stage 2**: pm_load_collapse - Collapse load tensor indexing
/// **Stage 3**: pm_split_ranges + pm_flatten_range - Range splitting
/// **Stage 4**: sym + pm_flatten_range - Initial symbolic (TOP_DOWN)
/// **Stage 5**: pm_simplify_ranges - Simplify/merge ranges
/// **Stage 6**: apply_opts - Post-range optimization (happens in optimizer)
#[tracing::instrument(skip_all)]
pub fn rangeify_with_map(sink: Arc<UOp>) -> svod_ir::Result<RangeifyResult> {
    // Scheduling is not part of any caller's scope: a graph realized inside one must
    // not stamp that scope onto the compiler artifacts this pipeline mints. Node
    // attribution already sits on the graph and is carried forward explicitly.
    let _detached = svod_ir::origin::OriginScope::suspend();

    // Resolve the exact hardware-independent multi subset before tags capture
    // tensor identity and before range assignment can materialize its sources.
    let t_stage = std::time::Instant::now();
    svod_ir::dump_canonical_stage("pre_multi", &sink);
    let mut sink = crate::rewrite::graph_rewrite_preserve_calls(&crate::multi::multi_pm(), sink, &mut ());
    svod_ir::dump_canonical_stage("post_multi", &sink);
    crate::multi::validate_supported_subset(&sink)?;
    sink = crate::rewrite::graph_rewrite_preserve_calls(&crate::multi::lower_allreduce_pm(), sink, &mut ());
    crate::multi::validate_no_unresolved_allreduce(&sink)?;
    tracing::debug!(
        uop.tree = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "pre-rangeify multi-device resolution complete"
    );
    // add_tags: assign sequential integer tags to UOps.
    // Tags track tensor identity after shard-local structure is normalized.
    let t_stage = std::time::Instant::now();
    let mut tag_ctx = AddTagsCtx::new();
    sink = crate::rewrite::graph_rewrite_bottom_up_preserve_calls(&add_tags_patterns(), sink, &mut tag_ctx);
    let output_tag_order: HashMap<usize, usize> = match sink.op() {
        Op::Sink(ops::Sink { sources, .. }) => sources
            .iter()
            .enumerate()
            .flat_map(|(position, source)| {
                let tags = source.tag().clone().or_else(|| source.base().tag().clone()).unwrap_or_default();
                tags.into_iter().map(move |tag| (tag, position))
            })
            .collect(),
        _ => sink.tag().as_ref().into_iter().flatten().copied().map(|tag| (tag, 0)).collect(),
    };
    let uop_list = tag_ctx.uop_list;
    tracing::debug!(
        tagged_count = uop_list.len(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "add_tags complete"
    );

    // resolve_function before heavy range/movement stages.
    let t_stage = std::time::Instant::now();
    sink = resolve_calls(sink)?;
    crate::multi::validate_supported_subset(&sink)?;
    tracing::debug!(
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "resolve_function complete"
    );

    // Combined earliest bottom-up rewrite, including reduction splitting.
    let t_stage = std::time::Instant::now();
    static EARLY_COMBINED: LazyLock<crate::TypedPatternMatcher<super::kernel::SplitReduceOpConfig>> =
        LazyLock::new(|| {
            super::patterns::movement_op_patterns().with_context::<super::kernel::SplitReduceOpConfig>()
                + super::patterns::early_rewrites().with_context::<super::kernel::SplitReduceOpConfig>()
                + super::patterns::split_reduceop_patterns()
        });
    let mut split_config = super::kernel::SplitReduceOpConfig::default();
    sink = crate::rewrite::graph_rewrite_bottom_up_preserve_calls(&*EARLY_COMBINED, sink, &mut split_config);
    tracing::debug!(
        uop.tree = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "earliest rewrites complete"
    );

    // =========================================================================
    // Stage 0: Range assignment + apply rangeify patterns.
    // Includes pm_generate_realize_map, assign loop, and pm_apply_rangeify
    // (REDUCE_AXIS→REDUCE, PAD→WHERE, STAGE+INDEX).
    // =========================================================================
    let t_stage = std::time::Instant::now();
    let (rangeified, mut indexing_ctx) = super::indexing::run_rangeify(sink)?;
    sink = rangeified;
    tracing::debug!(
        uop.tree = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "Stage 0: range assignment + apply rangeify complete"
    );

    // =========================================================================
    // Fused fixpoint: symbolic + reduce-simplify + movement-op rewriting +
    // const folding + dead-axis pruning + stage removal. Pattern groups
    // re-fire each other's outputs until the graph stabilizes — splitting
    // them into separate passes loses simplifications that only appear after
    // a substitution.
    // =========================================================================
    {
        let t_stage = std::time::Instant::now();
        use std::sync::LazyLock;
        static MEGA_PASS: LazyLock<crate::TypedPatternMatcher> = LazyLock::new(|| {
            crate::symbolic::symbolic()
                + super::patterns::pm_reduce_simplify()
                + super::patterns::movement_op_patterns()
                + super::patterns::buffer_folding()
                + super::patterns::dead_axis_removal()
                + super::patterns::pm_remove_bufferize()
        });
        let mega_pass = &*MEGA_PASS;
        tracing::debug!(
            total_patterns = mega_pass.len(),
            wildcard_count = mega_pass.wildcard_count(),
            indexed_buckets = mega_pass.indexed_count(),
            "mega-pass pattern stats"
        );
        sink = crate::rewrite::graph_rewrite_preserve_calls(mega_pass, sink, &mut ());
        tracing::debug!(
            node_count = sink.node_count(),
            elapsed_ms = t_stage.elapsed().as_millis() as u64,
            "mega-pass complete"
        );
    }

    // Stages 2a-6 (load_collapse, split_ranges, symbolic+flatten, simplify_ranges,
    // split_store) now run per-kernel in optimizer::apply_pre_optimization().

    // Rebuild SINK from transformed versions of the original outputs only.
    // Intermediate tags also survive rangeify, but they are not public outputs.
    {
        let mut filtered: Vec<Arc<UOp>> = sink
            .backward_slice()
            .into_iter()
            .filter(|s| {
                let valid_op = matches!(
                    s.base().op(),
                    Op::Stage(..) | Op::MStack(..) | Op::Const(_) | Op::Param(..) | Op::After(..)
                );
                let output =
                    s.tag().as_ref().is_some_and(|tags| tags.iter().any(|tag| output_tag_order.contains_key(tag)));
                valid_op && output
            })
            .collect();
        if !filtered.is_empty() {
            filtered.sort_by_key(|node| {
                node.tag()
                    .as_ref()
                    .and_then(|tags| tags.iter().filter_map(|tag| output_tag_order.get(tag)).min().copied())
                    .unwrap_or(usize::MAX)
            });
            tracing::debug!(filtered = filtered.len(), "SINK rebuilt from tagged backward slice");
            sink = UOp::sink(filtered);
        }
    }

    // Buffer limit enforcement
    if let Some(device) = super::patterns::extract_device_from_graph(&sink)
        && let Some(limit) = device.max_buffers()
    {
        let t_stage = std::time::Instant::now();
        let limit_matcher = super::patterns::buffer_limit_patterns(limit);
        sink = crate::rewrite::graph_rewrite_preserve_calls(&limit_matcher, sink, &mut indexing_ctx);
        tracing::debug!(
            uop.tree = sink.tree(),
            elapsed_ms = t_stage.elapsed().as_millis() as u64,
            "Stage 7b: buffer limit enforcement complete"
        );
    }

    // =========================================================================
    // Stage 8: Post-range optimization happens in optimizer module (apply_opts)
    // =========================================================================

    // Build RangeifyContext for return
    let rangeify_ctx = RangeifyContext { range_counter: indexing_ctx.range_counter(), range_map: HashMap::new() };

    svod_ir::dump_canonical_stage("rangeified", &sink);
    Ok(RangeifyResult { sink, context: rangeify_ctx, uop_list })
}

/// Pattern matcher for range flattening.
///
/// Extracts all RANGE operations from nested END/REDUCE/STORE structures.
pub fn pm_flatten_range() -> &'static crate::TypedPatternMatcher {
    crate::cached_patterns! {
        r @ End { computation: _, ranges } if !ranges.is_empty() => flatten_range_impl(r),
        r @ Reduce { src: _, ranges, reduce_op: _ } if !ranges.is_empty() => flatten_range_impl(r),
    }
}

// ============================================================================
// RANGE SPLITTING (pm_split_ranges equivalent)
// ============================================================================

/// Context for tracking ranges that should be split via modulo decomposition.
///
/// When we see `RANGE % const`, we mark the range for splitting at the SINK.
pub struct SplitRangesContext {
    /// Maps RANGE ids to their modulo constant, or `None` for ranges that must
    /// not be split at all (image stores address two coordinates directly).
    pub marked_ranges: HashMap<u64, Option<i64>>,
}

impl SplitRangesContext {
    pub fn new() -> Self {
        Self { marked_ranges: HashMap::new() }
    }
}

impl Default for SplitRangesContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern matcher for range splitting via modulo arithmetic.
///
/// Context-collecting pass that:
/// 1. Marks RANGE ops used in `RANGE % const` expressions
/// 2. At SINK, substitutes marked ranges with divmod decomposition
///
/// Example transformation for `RANGE(12) % 4`:
/// - Original: `r = RANGE(12)`
/// - After: `r_div = RANGE(3) * 4`, `r_mod = RANGE(4)`, substitute `r → r_div + r_mod`
///
pub fn pm_split_ranges() -> crate::TypedPatternMatcher<SplitRangesContext> {
    crate::patterns! {
        @context SplitRangesContext;

        // Mark RANGE % const: record the modulo constant for this range
        _modop @ FloorMod(r @ Range { end: _, axis_id: _, axis_type: _ }, c @ Const(_)) => {
            mark_range_mod(ctx, r, c);
            None // Don't transform yet, just mark
        },

        // An image STORE pins its index ranges against splitting.
        Store { index, value: _, gate: _ } => {
            dont_split_ranges_for_image(ctx, index);
            None
        },

        // At SINK: perform the substitution
        sink @ Sink { sources: _ } if !ctx.marked_ranges.is_empty() => {
            do_split_ranges_substitute(ctx, sink)
        },
    }
}

/// Extract i64 from a Const UOp.
fn const_uop_to_i64(c: &Arc<UOp>) -> Option<i64> {
    match c.op() {
        Op::Const(cv) => match cv.0 {
            ConstValue::Int(v) => Some(v),
            ConstValue::UInt(v) => Some(v as i64),
            _ => None,
        },
        _ => None,
    }
}

/// Check if a RANGE end is divisible by the modulo constant.
fn is_divisible_range_end(end: &Arc<UOp>, c: &Arc<UOp>) -> bool {
    let Some(mod_val) = const_uop_to_i64(c) else {
        return false;
    };
    matches!(end.op(), Op::Const(_)) && end.divides(mod_val).is_some()
}

/// Mark a range for modulo decomposition.
fn mark_range_mod(ctx: &mut SplitRangesContext, r: &Arc<UOp>, c: &Arc<UOp>) {
    if ctx.marked_ranges.contains_key(&r.id) {
        return;
    }

    // Warp and device ranges are not looped over, so they cannot be split.
    let Op::Range(ops::Range { end, axis_type, .. }) = r.op() else {
        return;
    };
    if matches!(axis_type, AxisType::Warp | AxisType::Device) {
        return;
    }
    if !is_divisible_range_end(end, c) {
        return;
    }
    if let Some(mod_val) = const_uop_to_i64(c) {
        ctx.marked_ranges.insert(r.id, Some(mod_val));
    }
}

/// Pin every range an image STORE indexes with, so the SINK substitution leaves it
/// alone: an image address is a pair of coordinates, not a flat offset that a
/// divmod split can reconstruct.
fn dont_split_ranges_for_image(ctx: &mut SplitRangesContext, index: &Arc<UOp>) {
    let Op::Index(ops::Index { buffer, indices }) = index.op() else { return };
    if !buffer.dtype().is_image() {
        return;
    }
    for range in indices.iter().flat_map(|index| index.ranges()) {
        ctx.marked_ranges.insert(range.id, None);
    }
}

/// Perform the substitution at SINK level.
///
/// For each marked RANGE with `end` divisible by `mod_val`:
/// - Create `r_outer = RANGE(end / mod_val)` with same axis type, shifted axis_id
/// - Create `r_inner = RANGE(mod_val)` with same axis type, shifted axis_id
/// - Substitute `r → r_outer * mod_val + r_inner`
fn do_split_ranges_substitute(ctx: &mut SplitRangesContext, sink: &Arc<UOp>) -> Option<Arc<UOp>> {
    use svod_ir::rewrite::graph_rewrite_bottom_up;

    if ctx.marked_ranges.is_empty() {
        return None;
    }

    // Build substitution map
    let mut subs: HashMap<u64, Arc<UOp>> = HashMap::new();

    // Traverse in graph order so multiple splits serialize deterministically.
    let topo = sink.toposort();

    for uop in &topo {
        if let Some(&Some(mod_val)) = ctx.marked_ranges.get(&uop.id)
            && let Op::Range(ops::Range { end, axis_id, axis_type, .. }) = uop.op()
        {
            let Some(outer_end) = end.divides(mod_val) else {
                continue;
            };

            let outer_range = UOp::range_axis(outer_end, axis_id.child(0), *axis_type);

            let inner_range = UOp::range_axis(UOp::index_const(mod_val), axis_id.child(1), *axis_type);

            // Substitution: r → outer * mod_val + inner
            let mod_const = UOp::index_const(mod_val);
            let outer_scaled = outer_range.mul(&mod_const);
            let combined = outer_scaled.add(&inner_range);

            subs.insert(uop.id, combined);
        }
    }

    if subs.is_empty() {
        return None;
    }

    // Apply substitutions using the substitute pattern
    let substitute_pm = crate::patterns! {
        r @ Range { end: _, axis_id: _, axis_type: _ } if subs.contains_key(&r.id) => {
            subs.get(&r.id).cloned()
        },
    };

    let result = graph_rewrite_bottom_up(&substitute_pm, sink.clone(), &mut ());
    // Tinygrad's do_substitute returns ret.simplify() here. At the pinned
    // revision UOp.simplify is exactly symbolic + pm_fold_cast_const.
    static SPLIT_SIMPLIFY: std::sync::LazyLock<crate::TypedPatternMatcher> =
        std::sync::LazyLock::new(|| crate::symbolic::symbolic() + crate::symbolic::pm_fold_cast_const());
    let result = crate::rewrite::graph_rewrite(&*SPLIT_SIMPLIFY, result, &mut ());

    // Clear the context after substitution
    ctx.marked_ranges.clear();

    Some(result)
}

// ============================================================================
// TRANSFORM HELPERS (movement ops → STAGE + INDEX)
// ============================================================================

/// Transform a UOp's sources by adding STAGE + INDEX where needed.
pub fn transform_sources_with_bufferize(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Vec<Arc<UOp>>> {
    if matches!(x.op(), Op::Stage(..) | Op::Index(..)) {
        return None;
    }

    let sources = x.op().sources();
    if sources.is_empty() {
        return None;
    }

    let data_sources = super::indexing::data_sources(x);

    // INDEX is only added when `x in ctx.range_map`. For SINK (not in
    // range_map), realized sources still get STAGE but no INDEX.
    let input_ranges = if let Some((input, _)) = ctx.get_ranges(x) { input.clone() } else { Vec::new() };

    let mut new_sources = Vec::with_capacity(sources.len());
    let mut any_changed = false;

    for (source_index, src) in sources.iter().enumerate() {
        // Op sources are ordered as data sources followed by metadata/ranges.
        let is_data_source = source_index < data_sources.len();
        let source_ranges =
            if is_data_source { super::indexing::broadcast_ranges(x, src, &input_ranges) } else { Vec::new() };
        let new_src = if is_data_source || ctx.get_realize_axes(src).is_some() {
            transform_single_source(x, src, &source_ranges, ctx)
        } else {
            src.clone()
        };
        if !Arc::ptr_eq(&new_src, src) {
            any_changed = true;
        }
        new_sources.push(new_src);
    }

    if any_changed { Some(new_sources) } else { None }
}

/// Flatten multi-range STAGE to single-range via RESHAPE to 1D.
///
/// 1. Reshapes multi-dim ranges to a single flat index via apply_reshape_ranges
/// 2. Creates new STAGE with single computed range
/// 3. Wraps with RESHAPE back to original shape for downstream movement ops
/// 4. For symbolic range ends, adds SHRINK to symbolic shape
///
/// After this, `bufferize_to_store` only sees single-range STAGE.
fn flatten_bufferize(stage: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Stage(ops::Stage { compute, ranges, opts }) = stage.op() else { return None };
    if ranges.len() <= 1 {
        return None;
    }
    // Extract shape from ranges: RANGE(end) → SInt::from(end), CONST(0) → 1
    let shape: Vec<svod_ir::SInt> = ranges
        .iter()
        .map(|r| match r.op() {
            Op::Range(ops::Range { end, .. }) => svod_ir::SInt::from(end.clone()),
            _ => svod_ir::SInt::from(1usize),
        })
        .collect();

    // Flatten: apply_reshape_ranges(in_shape=(prod,), out_shape=shape, rngs=ranges)
    let flat_shape = vec![svod_ir::sint_prod(&shape)];
    let ranges_vec: Vec<Arc<UOp>> = ranges.iter().cloned().collect();
    let flat_indices = super::indexing::apply_reshape_ranges(&flat_shape, &shape, &ranges_vec);
    assert_eq!(flat_indices.len(), 1, "flatten_bufferize: expected 1 flat index, got {}", flat_indices.len());
    // New STAGE with single range
    let flat_buf = UOp::stage(compute.clone(), vec![flat_indices[0].clone()], opts.clone());

    // RESHAPE back to original shape.
    let shape_smallvec: Shape = shape.iter().cloned().collect();
    let reshaped = flat_buf.try_reshape(&shape_smallvec).expect("flatten_bufferize: try_reshape failed");

    // For symbolic range ends, add SHRINK to symbolic shape.
    let has_symbolic =
        ranges.iter().any(|r| matches!(r.op(), Op::Range(ops::Range { end, .. }) if !matches!(end.op(), Op::Const(_))));

    if has_symbolic {
        let sym_ranges: Vec<(svod_ir::SInt, svod_ir::SInt)> = ranges
            .iter()
            .map(|r| match r.op() {
                Op::Range(ops::Range { end, .. }) => (svod_ir::SInt::from(0usize), svod_ir::SInt::from(end.clone())),
                _ => (svod_ir::SInt::from(0usize), svod_ir::SInt::from(1usize)),
            })
            .collect();
        Some(reshaped.try_shrink(&sym_ranges).expect("flatten_bufferize: try_shrink failed for symbolic ranges"))
    } else {
        Some(reshaped)
    }
}

/// Push movement op or INDEX through AFTER: `AFTER(r(x, ...), deps) → r(AFTER(x, deps), ...)`.
///
/// Reuses the original op's remaining sources directly (no roundtrip/validation).
///
/// Tag placement follows tinygrad `schedule/rangeify.py:54-55`: the rebuilt AFTER
/// keeps the original AFTER's tag (`a.replace`), and the outer movement node is
/// constructed fresh, so it carries no tag.
pub(crate) fn push_op_through_after(
    after: &Arc<UOp>,
    r: &Arc<UOp>,
    deps: &SmallVec<[Arc<UOp>; 4]>,
) -> Option<Arc<UOp>> {
    let inner_src = &r.op().sources()[0];
    let new_after_sources = std::iter::once(inner_src.clone()).chain(deps.iter().cloned()).collect();
    let new_after = after.with_sources(new_after_sources);
    let new_sources = std::iter::once(new_after).chain(r.op().sources().into_iter().skip(1)).collect();
    Some(r.with_sources(new_sources).rtag(None))
}

/// Transform a single source by adding STAGE + INDEX if needed.
///
/// Non-recursive: only handles immediate buffer-like and realized sources.
/// Movement ops and compute ops are left for the BPM rewrite engine to process
/// individually.
///
/// INDEX nodes are created with a single linear index, computed from the
/// buffer's dimensional ranges and the consumer's index expressions. This
/// eliminates the need for a later linearization pass.
pub(crate) fn transform_single_source(
    consumer: &Arc<UOp>,
    src: &Arc<UOp>,
    input_ranges: &[Arc<UOp>],
    ctx: &mut IndexingContext,
) -> Arc<UOp> {
    // Case 1: Buffer-like op → add multi-index INDEX
    // Unlike Case 2 (STAGE), we can't linearize here because the buffer's
    // dimensional structure isn't directly available from ctx — the output_ranges
    // may contain PAD validity expressions, not clean RANGE ops.
    // Multi-index INDEX is preserved through the pipeline; codegen linearizes at render time.
    if matches!(
        src.op(),
        Op::Buffer(..) | Op::Param(..) | Op::Slice(..) | Op::MStack(..) | Op::MSelect(..) | Op::After(..)
    ) {
        if !input_ranges.is_empty() {
            let indices = linearize_static_indices(src, input_ranges).unwrap_or_else(|| input_ranges.to_vec());
            return UOp::index()
                .buffer(Arc::clone(src))
                .indices(indices)
                .call()
                .expect("Failed to create INDEX for buffer source");
        }
        return Arc::clone(src);
    }

    // Case 2: source needs realization → wrap in STAGE + INDEX
    let realize_axes_opt = ctx.get_realize_axes(src).cloned();
    if let Some(ref realize_axes) = realize_axes_opt {
        let (_, output_ranges) = ctx.get_ranges(src).expect("Realized op must have ranges");

        let closed_ranges: Vec<_> = output_ranges
            .iter()
            .enumerate()
            .filter(|(i, _)| realize_axes.contains(i))
            .map(|(_, r)| Arc::clone(r))
            .collect();

        if matches!(src.op(), Op::Store(..)) {
            ctx.clear_realize(src);
            let end_ranges: SmallVec<[Arc<UOp>; 4]> =
                closed_ranges.into_iter().filter(|range| matches!(range.op(), Op::Range(..))).collect();
            return if end_ranges.is_empty() { Arc::clone(src) } else { src.end(end_ranges) };
        }

        // removable = consumer is not COPY and src is not ALWAYS_CONTIGUOUS.
        let is_copy_consumer = matches!(consumer.op(), Op::Copy(..));
        let is_always_contiguous_src = super::indexing::is_always_contiguous(src);
        let removable = !ctx.is_non_removable_realize(src) && !is_copy_consumer && !is_always_contiguous_src;
        let addrspace = if output_ranges.len() == realize_axes.len() { AddrSpace::Global } else { AddrSpace::Local };
        tracing::debug!(
            src_id = src.id,
            src_op = src.op().as_ref(),
            consumer_id = consumer.id,
            consumer_op = consumer.op().as_ref(),
            realize_axes = ?realize_axes,
            output_ranges_len = output_ranges.len(),
            addrspace = ?addrspace,
            removable = removable,
            "STAGE decision"
        );
        // Propagate source device to STAGE opts.
        let device = src.device_spec();
        let opts = BufferizeOpts { device, local_axis: None, addrspace, removable };

        // tag=s.tag if GLOBAL, else None.
        let buf_tag = if addrspace == AddrSpace::Global { src.tag().clone() } else { None };
        // The tag is the tensor-output identity and only GLOBAL stages keep it; the
        // origin is attribution and always belongs to the computation being staged.
        let bufferized = UOp::stage(Arc::clone(src), closed_ranges.clone(), opts).rorigin(src.origin());
        let bufferized = if let Some(t) = buf_tag { bufferized.with_tag(t) } else { bufferized };

        let index_ranges: Vec<_> = input_ranges
            .iter()
            .enumerate()
            .filter(|(i, _)| realize_axes.contains(i))
            .map(|(_, r)| Arc::clone(r))
            .collect();

        if !index_ranges.is_empty() {
            // Create multi-index INDEX; linearization happens in pm_add_buffers_patterns
            // via linearize_index_on_bufferize (BPM pattern).
            return UOp::index()
                .buffer(bufferized)
                .indices(index_ranges)
                .call()
                .expect("Failed to create INDEX after STAGE");
        } else {
            return bufferized;
        }
    }

    // Default: no transformation — BPM engine handles movement/compute ops individually
    Arc::clone(src)
}

/// Generic buffers are one-dimensional at the renderer boundary. Preserve
/// image/dynamic coordinates, but flatten statically-shaped tensor accesses at
/// their rangeify producer.
fn linearize_static_indices(buffer: &Arc<UOp>, indices: &[Arc<UOp>]) -> Option<Vec<Arc<UOp>>> {
    if indices.len() <= 1 || buffer.dtype().is_image() {
        return None;
    }
    // Image coordinates are a (y, x) pair the renderer addresses directly; flattening
    // them into one linear offset makes the access unrenderable.
    let shape = buffer.shape().ok().flatten()?;
    if shape.len() != indices.len() {
        return None;
    }
    let extents = shape.iter().map(|dim| dim.as_const()).collect::<Option<Vec<_>>>()?;
    let mut stride = 1i64;
    let mut terms = Vec::with_capacity(indices.len());
    for (index, extent) in indices.iter().zip(extents.iter()).rev() {
        terms.push(if stride == 1 { index.clone() } else { index.mul(&UOp::index_const(stride)) });
        stride = stride.checked_mul(*extent as i64)?;
    }
    Some(vec![terms.into_iter().reduce(|lhs, rhs| lhs.add(&rhs))?])
}

// ============================================================================
// STAGE TO STORE CONVERSION
// ============================================================================

/// Calculate buffer size from STAGE ranges.
///
/// `size = prod(int(r.vmax+1) for r in ranges)`. RANGE UOps have `vmax = end-1`,
/// so `vmax+1 = end`. For flattened STAGE (single computed expression),
/// `vmax+1` gives the total flat size.
fn calculate_size_from_ranges(ranges: &SmallVec<[Arc<UOp>; 4]>) -> usize {
    if ranges.is_empty() {
        return 1;
    }

    ranges
        .iter()
        .map(|r| {
            // int(r.vmax+1) — works for both RANGE and computed expressions.
            let vmax = r.vmax();
            match vmax {
                ConstValue::Int(v) if *v >= 0 => (*v + 1) as usize,
                ConstValue::UInt(v) => (*v + 1) as usize,
                other => panic!(
                    "Cannot allocate buffer: range vmax resolved to {:?}. \
                     Buffers require concrete sizes (no symbolic-sized buffers).",
                    other
                ),
            }
        })
        .product()
}

/// Sort ranges by (axis_id, axis_type) for correct row-major linearization.
///
/// Ensures multi-dimensional ranges are linearized in the correct order
/// regardless of their insertion order in the graph.
fn sort_ranges_by_axis_id(ranges: &SmallVec<[Arc<UOp>; 4]>) -> SmallVec<[Arc<UOp>; 4]> {
    let mut sorted: Vec<_> = ranges.iter().cloned().collect();
    sorted.sort_by(|a, b| match (a.op(), b.op()) {
        (
            Op::Range(ops::Range { axis_id: a_id, axis_type: a_type, .. }),
            Op::Range(ops::Range { axis_id: b_id, axis_type: b_type, .. }),
        ) => (a_id, a_type.priority()).cmp(&(b_id, b_type.priority())),
        (Op::Range(..), _) => std::cmp::Ordering::Less,
        (_, Op::Range(..)) => std::cmp::Ordering::Greater,
        _ => std::cmp::Ordering::Equal,
    });
    sorted.into()
}

/// Collect RANGE UOps from STAGE ranges, traversing flattened expressions.
///
/// After `flatten_bufferize`, `ranges[0]` may be a computed expression (Add/Mul of RANGEs)
/// rather than a direct RANGE UOp. This helper traverses all range entries:
/// - Direct RANGE UOps are collected immediately
/// - Non-CONST expressions are traversed via `.ranges()` to find embedded RANGE UOps
/// - CONST entries (collapsed singleton dims) are skipped
/// - Deduplicates by UOp id
fn collect_range_uops(ranges: &SmallVec<[Arc<UOp>; 4]>) -> SmallVec<[Arc<UOp>; 4]> {
    let mut collected = SmallVec::new();
    for r in ranges.iter() {
        if matches!(r.op(), Op::Range(..)) {
            collected.push(r.clone());
        } else if !matches!(r.op(), Op::Const(_)) {
            for rng in r.ranges().iter() {
                if !collected.iter().any(|c: &Arc<UOp>| c.id == rng.id) {
                    collected.push(rng.clone());
                }
            }
        }
    }
    collected
}

fn new_lunique_buffer(
    ctx: &mut RangeifyBufferContext,
    device: svod_ir::DeviceSpec,
    size: usize,
    dtype: DType,
) -> Arc<UOp> {
    let shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![svod_ir::SInt::Const(size)]);
    // Keep deterministic schedule-local slots disjoint from runtime BUFFER slots.
    // Cache restoration replaces these with fresh runtime slots before execution.
    let slot = (1usize << (usize::BITS - 1)) | ctx.next_lunique();
    let arg = svod_ir::ParamArg::buffer(slot, dtype.clone(), AddrSpace::Global, Some(device));
    UOp::new(Op::Buffer(ops::Buffer { shape, arg: arg.into() }), dtype)
        .with_tag(smallvec::smallvec![svod_ir::uop::canonical::TAG_SCHEDULE_LOCAL_BUFFER])
}

/// Convert STAGE operation to STORE with buffer allocation and END wrapping.
///
/// # Arguments
///
/// * `bufferize_op` - The STAGE UOp to convert
/// * `ctx` - Kernel context for tracking buffers and generating IDs
pub fn bufferize_to_store(bufferize_op: &Arc<UOp>, ctx: &mut RangeifyBufferContext) -> Option<Arc<UOp>> {
    let (compute, ranges, opts) = match bufferize_op.op() {
        Op::Stage(ops::Stage { compute, ranges, opts }) => {
            tracing::debug!(
                bufferize_id = bufferize_op.id,
                compute_id = compute.id,
                ranges_len = ranges.len(),
                "bufferize_to_store: CONVERTING STAGE to STORE→AFTER"
            );
            (compute, ranges, opts)
        }
        _ => return None,
    };

    // The STAGE is the last node that knows which computation this buffer serves;
    // the BUFFER itself is origin-opaque, so the STORE/END/AFTER chain carries the
    // attribution forward to the kernel cut (copy- and alias-only kernels included).
    let origin = bufferize_op.origin().or_else(|| compute.origin());

    // Calculate size and base dtype upfront.
    let size = calculate_size_from_ranges(ranges);
    let base_dtype = match bufferize_op.dtype() {
        DType::Ptr { base, .. } => (*base).clone(),
        other => other,
    };

    // Get end_ranges for wrapping stores via `.end(*rngs)`.
    let end_ranges: SmallVec<[Arc<UOp>; 4]> = sort_ranges_by_axis_id(&collect_range_uops(ranges));

    // =========================================================================
    // Case 0: STAGE(AFTER(...)) reuses the underlying buffer.
    // =========================================================================
    if let Op::After(ops::After { passthrough, deps }) = compute.op() {
        let buf = passthrough.buf_uop().base();
        let mut ended_stores = SmallVec::<[Arc<UOp>; 4]>::new();

        for dep in deps {
            let Op::Store(ops::Store { index: store_index, value, .. }) = dep.op() else {
                continue;
            };
            if !matches!(store_index.op(), Op::Index(..)) {
                continue;
            }

            let mut store_target = store_index.clone();
            if let Op::Index(ops::Index { buffer: target_buffer, .. }) = store_target.op()
                && let Op::Stage(ops::Stage { compute: target_compute, .. }) = target_buffer.op()
                && matches!(target_compute.op(), Op::Index(..))
            {
                store_target = target_compute.clone();
            }

            if Arc::ptr_eq(value, &store_target) {
                continue;
            }

            let mut combined = SmallVec::<[Arc<UOp>; 4]>::new();
            let mut seen = HashSet::new();
            for range in store_target.ranges().iter().chain(end_ranges.iter()) {
                if seen.insert(range.id) {
                    combined.push(range.clone());
                }
            }
            let combined = sort_ranges_by_axis_id(&combined);
            ended_stores.push(store_target.store_value(value.clone()).end(combined).rorigin(origin));
        }

        let result = if ended_stores.is_empty() { buf } else { buf.after(ended_stores).rorigin(origin) };
        ctx.map_buffer(bufferize_op.clone(), result.clone());
        return Some(result);
    }

    // LOCAL stages survive kernel splitting and are lowered by pm_add_local_buffers.
    if opts.addrspace == AddrSpace::Local {
        return None;
    }

    let buffer = if let Some(existing_buffer) = ctx.get_buffer(bufferize_op) {
        existing_buffer.clone()
    } else {
        // Create BUFFER node — BUFFER → PARAM conversion happens later in split_store.
        let device = opts.device.clone().unwrap_or_else(svod_dtype::default_device::default_device);
        new_lunique_buffer(ctx, device, size, base_dtype.clone())
    };

    // Collect active RANGE UOps from the ranges. Sort by axis_id for correct
    // row-major linearization.
    let active_ranges: SmallVec<[Arc<UOp>; 4]> = collect_range_uops(ranges);
    let sorted_ranges = sort_ranges_by_axis_id(&active_ranges);

    // Broadcast buffer for STORE-side INDEX only.
    // The AFTER return uses the unbroadcast buffer so consumers can broadcast it properly.
    let vcount = compute.dtype().vcount();
    let store_buffer = if vcount > 1 { buffer.broadcast(vcount) } else { buffer.clone() };

    let store_target = if !sorted_ranges.is_empty() {
        // After flatten_bufferize, ranges[0] may be the already-linearized flat index.
        // Use it directly. For non-flattened single-range, the RANGE is used directly.
        assert!(
            ranges.len() <= 1 || ranges.iter().all(|r| matches!(r.op(), Op::Const(_))),
            "bufferize_to_store: unexpected multi-range in general path after flatten_bufferize"
        );
        let [idx] = ranges.as_slice() else { unreachable!("STAGE must be flattened before store conversion") };
        let idx = idx.clone();
        UOp::index()
            .buffer(store_buffer)
            .indices(vec![idx])
            .call()
            .expect("Failed to create INDEX for STAGE-to-STORE conversion")
    } else {
        // Scalar store: create INDEX with buffer + index 0.
        UOp::index()
            .buffer(store_buffer)
            .indices(vec![UOp::index_const(0)])
            .call()
            .expect("Failed to create INDEX for scalar STORE")
    };

    // Create STORE and wrap with END if there are output ranges: .store().end(*rngs).
    //
    // The END wrapper is critical because:
    // 1. split_store looks for END { computation: STORE, ranges } pattern
    // 2. END.ranges define the iteration space for the OUTPUT (not internal computations)
    // 3. For scalar stores (e.g., REDUCE results), no END wrapping (ranges is empty)
    // 4. REDUCE's loop is handled by pm_reduce which creates its own END internally
    // NOTE: STORE takes (index, value) - buffer is accessed via index.buffer
    let store = store_target.store_value(compute.clone()).rorigin(origin);

    // Determine END ranges: use only actual RANGE ops from STAGE.
    // CONST(0) entries are excluded because `.ranges` only collects RANGE UOps.
    // END should only wrap with actual iteration ranges, not collapsed singletons.
    let end_ranges: SmallVec<[Arc<UOp>; 4]> = sorted_ranges.clone();

    let do_store = if !end_ranges.is_empty() { store.end(end_ranges).rorigin(origin) } else { store };

    let result = buffer.after(SmallVec::from_elem(do_store, 1)).rorigin(origin);
    ctx.map_buffer(bufferize_op.clone(), result.clone());

    Some(result)
}

// ============================================================================
// REDUCTION SIMPLIFICATIONS
// ============================================================================

/// Partition ranges into parented and unparented.
pub(crate) fn partition_reduce_ranges(
    ranges: &SmallVec<[Arc<UOp>; 4]>,
    src_ranges: &[u64],
) -> (SmallVec<[Arc<UOp>; 4]>, Vec<Arc<UOp>>) {
    let mut parented = SmallVec::new();
    let mut unparented = Vec::new();

    for range in ranges {
        if src_ranges.contains(&range.id) {
            parented.push(Arc::clone(range));
        } else {
            unparented.push(Arc::clone(range));
        }
    }

    (parented, unparented)
}

pub(crate) fn get_range_size(range: &Arc<UOp>) -> Option<Arc<UOp>> {
    if let Op::Range(ops::Range { end, .. }) = range.op() { Some(Arc::clone(end)) } else { None }
}

/// Collapse REDUCE(ADD) by algebraic simplification, parameterized by pattern matcher.
///
/// For each reduce range:
/// 1. Gated toposort to find nodes "in scope" of the range
/// 2. Replace external inputs (nodes NOT in scope) with synthetic DEFINE_VAR
/// 3. Wrap substituted body in a synthetic REDUCE
/// 4. Run algebraic patterns (bound-from-below/above, distributive, etc.)
/// 5. If REDUCE is eliminated (no_range), reverse-substitute back
fn reduce_collapse_with(src: &Arc<UOp>, ranges: &[Arc<UOp>], pm: &crate::TypedPatternMatcher<()>) -> Option<Arc<UOp>> {
    use svod_ir::ReduceOp;

    if ranges.is_empty() {
        return None;
    }

    let mut u = Arc::clone(src);

    for range in ranges {
        // 1. Gated toposort: find nodes "in scope" of this range
        // Visited in toposort order, as tinygrad's dict is: iterating the set would
        // name the synthetic vars in hash order and make the pass nondeterministic.
        let in_scope_nodes = u.toposort_filtered(|node| node.in_scope_ranges().contains(&range.id));
        let in_scope: HashSet<UOpKey> = in_scope_nodes.iter().cloned().map(UOpKey).collect();

        // Bail if nested REDUCE or STORE in scope (can't collapse through these)
        if in_scope.iter().any(|k| matches!(k.0.op(), Op::Reduce(..) | Op::Store(..))) {
            return None;
        }

        // 2. Identify external inputs and substitute with scalar PARAMs.
        let mut replaces: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
        for node in &in_scope_nodes {
            node.op().map_child(|child| {
                let key = UOpKey(child.clone());
                if in_scope.contains(&key) || replaces.contains_key(&key) {
                    return;
                }
                if matches!(child.op(), Op::Const(_) | Op::VConst(..)) || matches!(child.op(), Op::Param(..)) {
                    return;
                }
                let vmin = match child.vmin() {
                    ConstValue::Invalid => return,
                    ConstValue::Int(i) => *i,
                    ConstValue::UInt(u) => *u as i64,
                    ConstValue::Float(f) => *f as i64,
                    ConstValue::Bool(b) => *b as i64,
                };
                let vmax = match child.vmax() {
                    ConstValue::Invalid => return,
                    ConstValue::Int(i) => *i,
                    ConstValue::UInt(u) => *u as i64,
                    ConstValue::Float(f) => *f as i64,
                    ConstValue::Bool(b) => *b as i64,
                };
                let var = UOp::variable(format!("in{}", replaces.len()), vmin, vmax, child.dtype());
                replaces.insert(key, var);
            });
        }

        // 3. Build synthetic REDUCE: substituted_body.reduce([range], ADD)
        let substituted = u.substitute(&replaces);
        let synthetic_reduce = substituted.reduce(smallvec![range.clone()], ReduceOp::Add);

        // 4. Apply algebraic patterns to try eliminating the range
        let result = crate::rewrite::graph_rewrite(pm, synthetic_reduce, &mut ());

        // 5. Check range eliminated (a plain walk, NOT in_scope_ranges,
        //    since REDUCE "ends" ranges and would give a false positive)
        if result.any_in_subtree(|x| matches!(x.op(), Op::Range(..))) {
            return None;
        }

        // 6. Reverse substitute the temporary PARAMs.
        let reverse: HashMap<UOpKey, Arc<UOp>> = replaces.into_iter().map(|(k, v)| (UOpKey(v), k.0)).collect();
        u = result.substitute(&reverse);
    }

    Some(u)
}

/// Collapse REDUCE using `pm_reduce_collapse` patterns.
pub fn reduce_collapse(src: &Arc<UOp>, ranges: &[Arc<UOp>]) -> Option<Arc<UOp>> {
    reduce_collapse_with(src, ranges, super::patterns::build_reduce_collapse_matcher())
}

/// Collapse REDUCE using extended `pm_reduce_load_collapse` patterns.
///
/// Includes `.or_casted()` variants, NE lifting, and the full `pm_load_collapse`
/// non-REDUCE patterns on top of `pm_reduce_collapse`.
pub fn reduce_load_collapse(src: &Arc<UOp>, ranges: &[Arc<UOp>]) -> Option<Arc<UOp>> {
    reduce_collapse_with(src, ranges, super::patterns::build_reduce_load_collapse_matcher())
}

pub(crate) fn cast_to_dtype(value: &Arc<UOp>, target_dtype: &svod_dtype::DType) -> Option<Arc<UOp>> {
    use svod_dtype::DType;

    let scalar_type = match target_dtype {
        DType::Scalar(s) => DType::Scalar(*s),
        DType::Vector { scalar, .. } => DType::Scalar(*scalar),
        _ => return None,
    };

    let casted = value.cast(scalar_type);

    if target_dtype.is_vector() {
        let count = target_dtype.count();
        let elements: SmallVec<[Arc<UOp>; 4]> = (0..count).map(|_| casted.clone()).collect();
        Some(UOp::stack(elements))
    } else {
        Some(casted)
    }
}

// ============================================================================
// RANGE SIMPLIFICATION
// ============================================================================

/// Per-pass state of `pm_simplify_ranges`: the bounds collected from INDEX
/// validity gates, plus the slice analyses `simplify_merge_adjacent` needs for
/// every END/REDUCE it visits, memoized so no match toposorts its own slice.
pub struct SimplifyRangesContext {
    bounds: HashMap<UOpKey, Arc<UOp>>,
    reduces: SliceMemo<Nodes>,
    divmod_counts: FxHashMap<u64, usize>,
}

impl Default for SimplifyRangesContext {
    fn default() -> Self {
        Self {
            bounds: HashMap::new(),
            reduces: SliceMemo::new(|uop| matches!(uop.op(), Op::Reduce(..))),
            divmod_counts: FxHashMap::default(),
        }
    }
}

/// Simplify ranges by merging adjacent ranges to reduce divmod operations.
///
/// Merges adjacent ranges when the merge reduces the number of IDIV and MOD
/// operations in the computation graph. The merged range is then decomposed
/// back using divmod to preserve correctness.
///
/// Validation:
/// - Both ranges must appear in the same REDUCE operations (consistent scoping)
/// - Both ranges must have the same axis type
///
/// Example:
/// - Original: Two ranges R1(16) and R2(8)
/// - Merge: Create R_merged(128), decompose as R1 = merged // 8 and R2 = merged % 8
/// - Accept: Only if this reduces or maintains the divmod count
pub fn simplify_merge_adjacent(ctx: &mut SimplifyRangesContext, u: &Arc<UOp>) -> Option<Arc<UOp>> {
    use crate::passes::linearize_index::count_divmod;

    // Get ended ranges for this operation
    let ended_ranges = match u.op() {
        Op::End(ops::End { computation: _, ranges }) => ranges.clone(),
        Op::Reduce(ops::Reduce { ranges, .. }) => ranges.clone(),
        _ => return None,
    };

    if ended_ranges.len() < 2 {
        return None;
    }

    let reduces = ctx.reduces.get(u);
    let mut divmod_count = |uop: &Arc<UOp>| *ctx.divmod_counts.entry(uop.id).or_insert_with(|| count_divmod(uop));

    // Cumulative merging: try all pairs and accumulate successful merges into `current`.
    let mut current = Arc::clone(u);
    let mut changed = false;

    // Re-extract ranges from current for each iteration
    let pairs: Vec<(usize, usize)> = if matches!(u.op(), Op::End(..)) {
        (0..ended_ranges.len() - 1).map(|i| (i, i + 1)).collect()
    } else {
        let mut perms = Vec::new();
        for i in 0..ended_ranges.len() {
            for j in 0..ended_ranges.len() {
                if i != j {
                    perms.push((i, j));
                }
            }
        }
        perms
    };

    for (i0, i1) in pairs {
        let r0 = &ended_ranges[i0];
        let r1 = &ended_ranges[i1];

        let (r0_axis_type, r0_end) = match r0.op() {
            Op::Range(ops::Range { end, axis_type, .. }) => (axis_type, end),
            _ => continue,
        };
        let (r1_axis_type, r1_end) = match r1.op() {
            Op::Range(ops::Range { end, axis_type, .. }) => (axis_type, end),
            _ => continue,
        };

        if r0_axis_type != r1_axis_type {
            continue;
        }

        // Check same REDUCE scope.
        let valid_reduce_scope = reduces.iter().all(|reduce| {
            let Op::Reduce(ops::Reduce { ranges: rngs, .. }) = reduce.op() else { return true };
            let r0_in = rngs.iter().any(|rng| Arc::ptr_eq(rng, r0));
            let r1_in = rngs.iter().any(|rng| Arc::ptr_eq(rng, r1));
            r0_in == r1_in
        });
        if !valid_reduce_scope {
            continue;
        }

        // Both ends must be constant. A symbolic end (a runtime-bound batch,
        // say) buys nothing here — the divmod count is unchanged, since the
        // decomposition trades two ranges for one plus a `// s1` and a `% s1`
        // — while the `s0 * s1` end it produces is symbolic, so every
        // downstream const-only opt filter (upcast, unroll, local dims, tensor
        // cores) drops the merged axis and with it the const axis that was
        // folded into it.
        let (Some(s0), Some(s1)) = (const_uop_to_i64(r0_end), const_uop_to_i64(r1_end)) else {
            continue;
        };
        if s0 <= 0 || s1 <= 0 || s0.checked_mul(s1).is_none() {
            continue;
        }

        let merged_size_uop = r0_end.mul(r1_end);
        let merged_range = r0.with_sources(vec![merged_size_uop]);

        let new_r0 = merged_range.floor_div(r1_end);
        let new_r1 = merged_range.mod_(r1_end);

        let mut subs: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
        subs.insert(UOpKey(r0.clone()), new_r0);
        subs.insert(UOpKey(r1.clone()), new_r1);

        // Apply substitution and simplify.
        let rewritten = current.substitute(&subs);
        static MERGE_SYM: std::sync::LazyLock<crate::TypedPatternMatcher> = std::sync::LazyLock::new(|| {
            crate::symbolic::symbolic() + crate::symbolic::pm_fold_cast_const() + pm_flatten_range()
        });
        let simplified = crate::rewrite::graph_rewrite(&*MERGE_SYM, rewritten, &mut ());

        // Accept if divmod count is reduced or equal.
        if divmod_count(&simplified) <= divmod_count(&current) {
            current = simplified;
            changed = true;
        }
    }

    if changed { Some(current) } else { None }
}

/// Collect the per-range upper bounds an INDEX proves.
///
/// Port of tinygrad `codegen/simplify.py:43-53`, widened to every index. Upstream
/// reads `idx.src[1]` alone because its INDEX carries one flattened index by this
/// stage; morok still has multi-index INDEX here whenever `linearize_static_indices`
/// bails (symbolic or dynamic dims), so a range used only in `indices[1..]` would
/// otherwise be neither bounded nor protected — and could be shrunk by a bound
/// harvested from a different access.
///
/// Guards are merged across indices keeping the largest bound; every range an
/// index uses without a guard of its own pins that range to its original end.
fn mark_gated(ctx: &mut SimplifyRangesContext, idx: &Arc<UOp>) {
    let Op::Index(ops::Index { indices, .. }) = idx.op() else { return };

    fn pin(ctx: &mut SimplifyRangesContext, range: &Arc<UOp>) {
        if let Op::Range(ops::Range { end, .. }) = range.op() {
            ctx.bounds.insert(UOpKey(range.clone()), end.clone());
        }
    }

    let is_gate = |index: &Arc<UOp>| matches!(index.op(), Op::Ternary(svod_ir::TernaryOp::Where, _, _, _));
    if !indices.iter().any(is_gate) {
        // No gate anywhere: every range the access reaches is an ungated use.
        for range in idx.ranges() {
            pin(ctx, &range);
        }
        return;
    }

    let mut guards: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    let mut ungated: Vec<Arc<UOp>> = Vec::new();
    for index in indices {
        let index_guards: HashMap<UOpKey, Arc<UOp>> = if is_gate(index) {
            index
                .get_valid()
                .split_uop(BinaryOp::And)
                .into_iter()
                .filter_map(|valid| match valid.op() {
                    Op::Binary(BinaryOp::Lt, range, bound)
                        if matches!(range.op(), Op::Range(..)) && matches!(bound.op(), Op::Const(_)) =>
                    {
                        Some((UOpKey(range.clone()), bound.clone()))
                    }
                    _ => None,
                })
                .collect()
        } else {
            HashMap::new()
        };

        let expression = index.get_idx();
        for range in expression.ranges() {
            if !index_guards.contains_key(&UOpKey(range.clone())) {
                ungated.push(range.clone());
            }
        }
        for (range, bound) in index_guards {
            let larger = guards.get(&range).is_none_or(|old| {
                const_uop_to_i64(old).zip(const_uop_to_i64(&bound)).is_some_and(|(old, new)| old < new)
            });
            if larger {
                guards.insert(range, bound);
            }
        }
    }

    // A range may feed several gated accesses. The largest bound is required
    // to preserve every access covered by the original iteration space.
    for (range, bound) in guards {
        let larger = ctx
            .bounds
            .get(&range)
            .is_none_or(|old| const_uop_to_i64(old).zip(const_uop_to_i64(&bound)).is_some_and(|(old, new)| old < new));
        if larger {
            ctx.bounds.insert(range, bound);
        }
    }

    // Any ungated use protects the range from narrowing.
    for range in &ungated {
        pin(ctx, range);
    }
}

fn protect_reduce_ranges(ctx: &mut SimplifyRangesContext, ranges: &[Arc<UOp>]) {
    for range in ranges {
        if let Op::Range(ops::Range { end, .. }) = range.op() {
            ctx.bounds.insert(UOpKey(range.clone()), end.clone());
        }
    }
}

fn substitute_simplified_ranges(ctx: &mut SimplifyRangesContext, sink: &Arc<UOp>) -> Option<Arc<UOp>> {
    let substitutions = ctx
        .bounds
        .iter()
        .filter_map(|(range, bound)| match range.0.op() {
            Op::Range(..) => Some((range.clone(), range.0.with_sources(vec![bound.clone()]))),
            _ => None,
        })
        .collect();
    ctx.bounds.clear();

    let substituted = sink.substitute(&substitutions);
    if Arc::ptr_eq(&substituted, sink) {
        return None;
    }

    // At 8c8b43de UOp.simplify is exactly this tier.
    static SIMPLIFY: std::sync::LazyLock<crate::TypedPatternMatcher> =
        std::sync::LazyLock::new(|| crate::symbolic::symbolic() + crate::symbolic::pm_fold_cast_const());
    Some(crate::rewrite::graph_rewrite(&*SIMPLIFY, substituted, &mut ()))
}

/// Merge ranges and narrow ranges proven bounded by every INDEX access.
pub fn pm_simplify_ranges() -> crate::TypedPatternMatcher<SimplifyRangesContext> {
    crate::patterns! {
        @context SimplifyRangesContext;

        // Rule order matches Tinygrad: merge, collect gates, protect REDUCE, substitute at SINK.
        u @ End { computation: _, ranges: _ } => simplify_merge_adjacent(ctx, u),
        u @ Reduce { src: _, ranges: _, reduce_op: _ } => simplify_merge_adjacent(ctx, u),
        idx @ Index { buffer: _, indices: _ } => {
            mark_gated(ctx, idx);
            None
        },
        _red @ Reduce { src: _, ranges, reduce_op: _ } => {
            protect_reduce_ranges(ctx, ranges);
            None
        },
        sink @ Sink { sources: _ } => substitute_simplified_ranges(ctx, sink),
    }
}

// ============================================================================
// RANGE FLATTENING
// ============================================================================

/// Flatten nested RANGE operations into canonical form.
pub fn flatten_range_impl(r: &Arc<UOp>) -> Option<Arc<UOp>> {
    let off = match r.op() {
        Op::Reduce(..) => 1,
        Op::End(..) => 1,
        _ => return None,
    };

    let original_sources = r.op().sources();
    let original_ranges = &original_sources[off..];
    if original_ranges.is_empty() {
        return None;
    }

    // BOOL/VOID sources are reduction backedges, not ranges to flatten.
    let is_backedge = |source: &&Arc<UOp>| source.dtype() == DType::Bool || source.dtype() == DType::Void;
    let backedges: Vec<Arc<UOp>> = original_ranges.iter().filter(is_backedge).cloned().collect();
    let sink = UOp::sink(original_ranges.iter().filter(|source| !is_backedge(source)).cloned().collect());
    let new_ranges: Vec<Arc<UOp>> =
        sink.toposort().into_iter().filter(|uop| matches!(uop.op(), Op::Range(..))).collect();

    let mut new_sources = original_sources[..off].to_vec();
    new_sources.extend(new_ranges);
    new_sources.extend(backedges);

    if new_sources.len() == original_sources.len()
        && new_sources.iter().zip(original_sources.iter()).all(|(a, b)| Arc::ptr_eq(a, b))
    {
        return None;
    }

    Some(r.with_sources(new_sources))
}

/// Apply range flattening to a computation graph.
pub fn flatten_ranges(root: &Arc<UOp>) -> Arc<UOp> {
    let mut replacements: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    for node in root.toposort() {
        if let Some(flattened) = flatten_range_impl(&node) {
            replacements.insert(UOpKey(node.clone()), flattened);
        }
    }

    root.substitute(&replacements)
}

// ============================================================================
// CYCLE DETECTION
// ============================================================================

/// Detect conflicting buffer identities reached through different INDEX source ops.
pub fn find_bufs(store: &Arc<UOp>) {
    let indices = store
        .toposort_filtered(|uop| !matches!(uop.op(), Op::After(..)))
        .into_iter()
        .filter(|uop| matches!(uop.op(), Op::Index(..)));
    let mut read_from = HashMap::new();

    for index in indices {
        let Op::Index(ops::Index { buffer, .. }) = index.op() else { unreachable!() };
        let buf = buffer.buf_uop();
        if !matches!(buf.op(), Op::Buffer(..) | Op::Param(..)) {
            continue;
        }
        let source_op = std::mem::discriminant(buffer.op());
        if let Some(previous) = read_from.insert(UOpKey(buf.clone()), source_op)
            && previous != source_op
        {
            panic!("cycle detected while indexing {buf:?}");
        }
    }
}

// ============================================================================
// PM_ADD_BUFFERS PATTERNS
// ============================================================================

/// Convert a contiguous DISK movement into Tinygrad's SLICE metadata.
fn late_buffer_slice(compute: &Arc<UOp>, stage: &Arc<UOp>) -> Option<Arc<UOp>> {
    use svod_ir::uop::cached_property::CachedProperty;
    use svod_ir::uop::properties::VminVmaxProperty;

    let Op::Stage(ops::Stage { opts, ranges, .. }) = stage.op() else { return None };

    // Only for DISK device
    if !matches!(&opts.device, Some(d) if d.is_disk()) {
        return None;
    }

    // Compute size from ranges (product of range ends)
    let size: usize = ranges
        .iter()
        .map(|r| {
            if let Op::Range(ops::Range { end, .. }) = r.op()
                && let (_, svod_ir::ConstValue::Int(v)) = VminVmaxProperty::get(end)
            {
                return *v as usize;
            }
            if let Op::Const(_) = r.op() {
                return 1; // const 0 index contributes dim of 1
            }
            1
        })
        .product();

    // Walk up from compute to find the INDEX node. The BITCAST/CONTIGUOUS
    // input is the starting node; look INTO its children for an INDEX. After
    // rangeify, the BITCAST's source should be an INDEX or contain one.
    let mut x = compute.clone();
    loop {
        // Check if any SOURCE of x is an INDEX
        if x.op().sources().iter().any(|s| matches!(s.op(), Op::Index(..))) {
            break;
        }
        // For BITCAST/CONTIGUOUS (the starting node), look into their source
        if matches!(x.op(), Op::BitCast(..) | Op::Contiguous(..)) {
            x = x.op().sources().first()?.clone();
            continue;
        }
        // Don't cross other elementwise ops
        if matches!(x.op(), Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) | Op::Cast(..)) {
            return None;
        }
        x = x.op().sources().first()?.clone();
    }
    let index = x.op().sources().iter().find(|s| matches!(s.op(), Op::Index(..)))?.clone();

    // Compute byte offset.
    let offset: usize = if let Op::Index(ops::Index { indices, .. }) = index.op() {
        if indices.is_empty() {
            // Scalar: offset from first index's constant arg.
            0
        } else {
            // Shaped: sum of index vmin values
            let mut total: i64 = 0;
            for idx in indices.iter() {
                let (vmin, _) = VminVmaxProperty::get(idx);
                if let svod_ir::ConstValue::Int(v) = vmin {
                    total += v;
                }
            }
            total.max(0) as usize
        }
    } else {
        0
    };

    // Get base buffer (the DISK BUFFER UOp)
    let base = index.base();

    let slice = base.contiguous_slice(size, offset, compute.dtype());

    let new_sources: Vec<Arc<UOp>> = std::iter::once(slice).chain(ranges.iter().cloned()).collect();
    Some(UOp::stage(new_sources[0].clone(), new_sources[1..].to_vec(), opts.clone()))
}

/// A CALL argument is a whole buffer, so the views rangeify can leave on one come
/// off: a RESHAPE wrapper (tinygrad's "remove any RESHAPEs on KERNEL"), and an
/// INDEX (its `pm_no_indexing_calls`). The INDEX appears when the argument is
/// shared with a ranged consumer: `push_op_through_after` lifts a custom kernel
/// output's reshape into one node that is both the next CALL's argument and,
/// say, an elementwise op's source, and indexing it for the op indexes the
/// argument too.
fn strip_views_on_callable_sources(callable: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Call(ops::Call { body, args, info }) = callable.op() else {
        return None;
    };
    let whole = |src: &Arc<UOp>| match src.op() {
        Op::Reshape(ops::Reshape { src: inner, .. }) => Some(inner.clone()),
        Op::Index(ops::Index { buffer, .. }) => Some(buffer.clone()),
        _ => None,
    };
    if !args.iter().any(|src| whole(src).is_some()) {
        return None;
    }
    let rewritten: SmallVec<[Arc<UOp>; 4]> = args.iter().map(|src| whole(src).unwrap_or_else(|| src.clone())).collect();
    Some(body.call(rewritten, info.clone()).rtag(callable.tag().clone()).rorigin(callable.origin()))
}

/// Create pattern matcher for adding buffers (STAGE → STORE conversion).
///
/// Uses `allow_locals=false`. Shared RangeifyBufferContext ensures unique buffer
/// IDs across all pattern matches.
pub fn pm_add_buffers_patterns() -> &'static crate::TypedPatternMatcher<super::kernel::RangeifyBufferContext> {
    static PM: LazyLock<crate::TypedPatternMatcher<super::kernel::RangeifyBufferContext>> =
        LazyLock::new(build_add_buffers_patterns);
    &PM
}

fn build_add_buffers_patterns() -> crate::TypedPatternMatcher<super::kernel::RangeifyBufferContext> {
    super::patterns::movement_op_patterns().with_context::<super::kernel::RangeifyBufferContext>()
        + crate::patterns! {
            @context super::kernel::RangeifyBufferContext;
            // Flatten multi-range STAGE to 1D.
            buf @ Stage { compute: _ } if matches!(buf.op(), Op::Stage(ops::Stage { ranges, .. }) if ranges.len() > 1)
                => { flatten_bufferize(buf) },
            // DISK STAGE(BITCAST|CONTIGUOUS) → SLICE.
            buf @ Stage { compute }
                if matches!(compute.op(), Op::BitCast(..) | Op::Contiguous(..))
                => late_buffer_slice(compute, buf),
            // STAGE → STORE conversion (allow_locals=false: treat local as global).
            buf @ Stage { compute: _ } => {
                bufferize_to_store(buf, ctx)
            },
            // Strip RESHAPE / INDEX views on CALL sources.
            c @ Call { body: _, args: _, info: _ } => { strip_views_on_callable_sources(c) },
        }
}
