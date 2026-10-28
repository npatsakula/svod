//! Range assignment and indexing context for rangeify transformation.
//!
//! This module provides the core range assignment algorithm that converts
//! movement operations into explicit loop ranges.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use indexmap::IndexSet;
use svod_ir::{AxisId, AxisType, BinaryOp, ConstValue, Op, SInt, UOp, UOpKey};
use tracing::{debug, info_span, instrument, trace};

use crate::argsort;
use svod_ir::ops;

// ============================================================================
// Context
// ============================================================================

/// (input_ranges, output_ranges) for a UOp.
type UOpRanges = (Vec<Arc<UOp>>, Vec<Arc<UOp>>);

/// Context for range assignment during rangeify.
#[derive(Default)]
pub struct IndexingContext {
    /// Maps UOps to realize status: Some(axes) = needs realization on axes.
    pub realize_map: HashMap<UOpKey, Option<Vec<usize>>>,
    /// Realization boundaries that must survive buffer-removal.
    non_removable_realizes: HashSet<UOpKey>,
    /// Maps each UOp to its (input_ranges, output_ranges).
    pub range_map: HashMap<UOpKey, UOpRanges>,
    /// Counter for generating unique range IDs.
    range_idx: usize,
}

impl IndexingContext {
    /// Create a new indexing context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create new RANGE with unique ID. Returns const 0 if size is 1.
    ///
    /// Ranges are created with `AxisId::Unrenumbered` to mark them as needing
    /// renumbering. The `renumber_range` pattern will later convert them to
    /// `AxisId::Renumbered` with sequential IDs starting from 0 for each kernel.
    pub fn new_range(&mut self, size: &SInt, axistype: AxisType) -> Arc<UOp> {
        // If size is already a RANGE UOp, return it unchanged.
        if let SInt::Symbolic(u) = size
            && matches!(u.op(), Op::Range(..))
        {
            return Arc::clone(u);
        }
        // Check if size is constant 1
        if let SInt::Const(1) = size {
            return UOp::index_const(0);
        }

        // Create range with Unrenumbered axis_id
        let axis_id = AxisId::Unrenumbered(self.range_idx);
        self.range_idx += 1;

        let size_uop = match size {
            SInt::Const(value) => UOp::index_const(*value as i64),
            SInt::Symbolic(value) => value.clone(),
            SInt::Infer => panic!("cannot create a range from an inferred dimension"),
        };

        UOp::range_axis(size_uop, axis_id, axistype)
    }

    /// Create a new RANGE from an existing UOp end value.
    /// Used when converting REDUCE ranges to LOOP ranges during bufferization.
    ///( when bufferizing, REDUCE ranges become LOOP)
    pub fn new_range_from_uop(&mut self, end: &Arc<UOp>, axis_type: AxisType) -> Arc<UOp> {
        let axis_id = AxisId::Unrenumbered(self.range_idx);
        self.range_idx += 1;
        UOp::range_axis(Arc::clone(end), axis_id, axis_type)
    }

    /// Mark a UOp for realization on all axes.
    pub fn mark_realize_all(&mut self, uop: &Arc<UOp>) -> svod_ir::Result<()> {
        if let Some(shape) = uop.shape()? {
            let axes = (0..shape.len()).collect();
            self.realize_map.insert(UOpKey(Arc::clone(uop)), Some(axes));
        }
        Ok(())
    }

    /// Mark a UOp for realization on specific axes.
    pub fn mark_realize(&mut self, uop: &Arc<UOp>, axes: Vec<usize>) {
        self.realize_map.insert(UOpKey(Arc::clone(uop)), Some(axes));
    }

    /// Mark a UOp for realization before its shape/range axes have been assigned.
    pub fn mark_realize_pending(&mut self, uop: &Arc<UOp>) {
        self.realize_map.insert(UOpKey(Arc::clone(uop)), None);
    }

    /// Mark a UOp for realization as a required memory dependency boundary.
    pub fn mark_realize_non_removable(&mut self, uop: &Arc<UOp>) {
        let key = UOpKey(Arc::clone(uop));
        self.realize_map.insert(key.clone(), None);
        self.non_removable_realizes.insert(key);
    }

    /// Check if a UOp is in the realize map.
    pub fn should_realize(&self, uop: &Arc<UOp>) -> bool {
        self.realize_map.contains_key(&UOpKey(Arc::clone(uop)))
    }

    /// Get the realize axes for a UOp.
    pub fn get_realize_axes(&self, uop: &Arc<UOp>) -> Option<&Vec<usize>> {
        self.realize_map.get(&UOpKey(Arc::clone(uop))).and_then(|opt| opt.as_ref())
    }

    /// Remove a UOp from the realize map once its realization boundary has been emitted.
    pub fn clear_realize(&mut self, uop: &Arc<UOp>) {
        let key = UOpKey(Arc::clone(uop));
        self.realize_map.remove(&key);
        self.non_removable_realizes.remove(&key);
    }

    /// Check if a realization boundary must not be inlined.
    pub fn is_non_removable_realize(&self, uop: &Arc<UOp>) -> bool {
        self.non_removable_realizes.contains(&UOpKey(Arc::clone(uop)))
    }

    /// Set the range map for a UOp.
    pub fn set_ranges(&mut self, uop: &Arc<UOp>, input_ranges: Vec<Arc<UOp>>, output_ranges: Vec<Arc<UOp>>) {
        self.range_map.insert(UOpKey(Arc::clone(uop)), (input_ranges, output_ranges));
    }

    /// Get the ranges for a UOp.
    pub fn get_ranges(&self, uop: &Arc<UOp>) -> Option<&UOpRanges> {
        self.range_map.get(&UOpKey(Arc::clone(uop)))
    }

    /// Get the current range counter value.
    pub fn range_counter(&self) -> usize {
        self.range_idx
    }
}

// ============================================================================
// Core Algorithm
// ============================================================================

/// Key for the movement-op cache: everything upstream's `@functools.cache` sees on
/// `apply_movement_op(op, in_shape, arg, rngs)` and `_apply_reshape(in_shape, out_shape,
/// urngs)` (tinygrad/schedule/indexing.py:158,171) — the op, its own arg, the input
/// shape and the range tuple, never the source being moved. Shapes and range tuples
/// collapse to one hash-consed `id` each.
#[derive(PartialEq, Eq, Hash)]
struct MovementKey {
    op: std::mem::Discriminant<Op>,
    arg: UOpKey,
    in_shape: UOpKey,
    rngs: UOpKey,
}

/// `_apply_reshape(in_shape, out_shape, urngs)`.
#[derive(PartialEq, Eq, Hash)]
struct ReshapeKey {
    in_shape: UOpKey,
    out_shape: UOpKey,
    rngs: UOpKey,
}

/// Both caches are process-global, exactly as upstream's `@functools.cache`: a hit
/// skips building the index chain and its `graph_rewrite` entirely, across every
/// `run_rangeify`.
type Memo<K> = std::sync::LazyLock<std::sync::Mutex<rustc_hash::FxHashMap<K, Vec<Arc<UOp>>>>>;
static MOVEMENT_CACHE: Memo<MovementKey> = std::sync::LazyLock::new(Default::default);
static RESHAPE_CACHE: Memo<ReshapeKey> = std::sync::LazyLock::new(Default::default);

/// One hash-consed identity per input tuple. The key owns the node, which is what
/// keeps its `id` stable: interning holds only weak references, so a tuple that no
/// live key mentions is re-interned with a fresh id.
fn tuple_key(sources: Vec<Arc<UOp>>) -> UOpKey {
    UOpKey(UOp::sink(sources))
}

fn shape_key(shape: &[SInt]) -> UOpKey {
    tuple_key(shape.iter().map(SInt::arithmetic_uop).collect())
}

/// Whether these inputs are already memoised, for the reuse test.
///
/// Hash-consing makes two equal index chains the same `Arc`, so pointer equality
/// alone cannot tell a cache hit from a rebuild; only the cache itself can.
#[cfg(test)]
pub(crate) fn movement_cache_holds(op: &Op, in_shape: &[SInt], rngs: &[Arc<UOp>]) -> bool {
    MOVEMENT_CACHE.lock().expect("movement cache poisoned").contains_key(&movement_key(op, in_shape, rngs))
}

fn cached<K: Eq + std::hash::Hash>(memo: &Memo<K>, key: K, f: impl FnOnce() -> Vec<Arc<UOp>>) -> Vec<Arc<UOp>> {
    if let Some(hit) = memo.lock().expect("movement cache poisoned").get(&key) {
        return hit.clone();
    }
    let result = f();
    memo.lock().expect("movement cache poisoned").insert(key, result.clone());
    result
}

/// Run range assignment on a UOp graph. Returns (transformed_sink, context).
#[instrument(skip(sink), fields(sink_id = sink.id))]
pub fn run_rangeify(sink: Arc<UOp>) -> svod_ir::Result<(Arc<UOp>, IndexingContext)> {
    let mut ctx = IndexingContext::new();

    // Step 1: Generate realize map via pattern matcher (pm_generate_realize_map)
    // bottom_up=True — patterns see ORIGINAL children
    crate::rewrite::graph_rewrite_bottom_up_preserve_calls(pm_generate_realize_map(), sink.clone(), &mut ctx);

    // Step 2: Get toposort (root-to-leaves) and consumer map. Shape/index args
    // are not dataflow consumers and must not influence range assignment.
    let consumer_map = consumer_map_for_data_sources(&sink);

    // Use forward toposort (root first) for range propagation
    let forward_topo: Vec<_> = sink.toposort_call_aware(false).into_iter().rev().collect();

    // Step 3: Assign ranges via forward traversal
    assign_ranges(&forward_topo, &consumer_map, &mut ctx)?;

    // Step 4: Apply rangeify patterns (pm_apply_rangeify)
    // Converts tensor REDUCE to ranged REDUCE, PAD→WHERE, creates STAGE+INDEX, removes movement ops.
    // Must run bottom_up so patterns see ORIGINAL children (bottom_up=True).
    let rangeify_matcher = super::patterns::apply_rangeify_patterns();
    let transformed_sink = crate::rewrite::graph_rewrite_bottom_up_preserve_calls(&rangeify_matcher, sink, &mut ctx);

    // Tinygrad has no recovery loop: neither PAD nor the tensor-form REDUCE survives
    // pm_apply_rangeify. Fail loudly rather than leaking one into the kernel AST.
    if transformed_sink
        .toposort_call_aware(false)
        .iter()
        .any(|node| matches!(node.op(), Op::Pad(..) | Op::ReduceAxis(..)))
    {
        return Err(svod_ir::Error::SymbolicShapeUnsupported {
            operation: "rangeify left a high-level PAD/ReduceAxis in the kernel graph",
        });
    }

    Ok((transformed_sink, ctx))
}

/// Consumers of each node, in discovery order and without repeats — a consumer
/// that reads the same source twice (`y * y`) counts once, matching upstream's
/// `consumer_map[x][c] = None` dict insert (indexing.py:202-205).
type ConsumerMap = HashMap<UOpKey, IndexSet<UOpKey>>;

fn consumer_map_for_data_sources(sink: &Arc<UOp>) -> ConsumerMap {
    let topo = sink.toposort_call_aware(false);
    let mut consumer_map: ConsumerMap = topo.iter().map(|u| (UOpKey(u.clone()), IndexSet::new())).collect();
    for consumer in topo {
        for source in data_sources(&consumer) {
            if let Some(consumers) = consumer_map.get_mut(&UOpKey(source)) {
                consumers.insert(UOpKey(consumer.clone()));
            }
        }
    }
    consumer_map
}

/// Data-bearing sources used by range assignment and source indexing.
/// Mirrors Tinygrad's `data_srcs`: movement/control metadata and AFTER deps do
/// not consume tensor iteration ranges.
pub(crate) fn data_sources(uop: &Arc<UOp>) -> Vec<Arc<UOp>> {
    match uop.op() {
        Op::Param(..) | Op::Buffer(..) | Op::Range(..) | Op::Special(..) | Op::Bind(..) => Vec::new(),
        op if op.is_movement() => op.sources().first().cloned().into_iter().collect(),
        Op::Index(ops::Index { buffer, .. }) => vec![buffer.clone()],
        Op::Slice(ops::Slice { buffer, .. }) => vec![buffer.clone()],
        Op::Stage(ops::Stage { compute, .. }) => vec![compute.clone()],
        Op::Reduce(ops::Reduce { src, .. }) | Op::ReduceAxis(ops::ReduceAxis { src, .. }) => vec![src.clone()],
        Op::After(ops::After { passthrough, .. }) => vec![passthrough.clone()],
        Op::End(ops::End { computation, .. }) => vec![computation.clone()],
        _ => uop.op().sources().into_iter().collect(),
    }
}

/// Consumer axes the source is broadcast over: the ones added on its left plus
/// the ones expanding a singleton source dim (`broadcast_axes`, ops.py:80-83).
/// `None` when the two shapes do not broadcast.
fn broadcast_axes(source: &Arc<UOp>, consumer: &Arc<UOp>) -> Option<(usize, Vec<usize>)> {
    let (Ok(Some(consumer_shape)), Ok(Some(source_shape))) = (consumer.shape(), source.shape()) else {
        return None;
    };
    let left_pad = consumer_shape.len().checked_sub(source_shape.len())?;
    let mut axes: Vec<usize> = (0..left_pad).collect();
    axes.extend(source_shape.iter().enumerate().filter_map(|(axis, source_dim)| {
        let consumer_axis = left_pad + axis;
        (source_dim.as_const() == Some(1) && consumer_shape[consumer_axis].as_const() != Some(1))
            .then_some(consumer_axis)
    }));
    Some((left_pad, axes))
}

/// Map a consumer's ranges onto one broadcastable source.
pub(crate) fn broadcast_ranges(consumer: &Arc<UOp>, source: &Arc<UOp>, ranges: &[Arc<UOp>]) -> Vec<Arc<UOp>> {
    if !is_broadcastable_op(consumer) {
        return ranges.to_vec();
    }
    let Some((left_pad, target_axes)) = broadcast_axes(source, consumer) else {
        return ranges.to_vec();
    };

    ranges
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis >= left_pad)
        .map(|(axis, range)| if target_axes.contains(&axis) { UOp::index_const(0) } else { range.clone() })
        .collect()
}

/// The ranges `x`'s consumers iterate that `x` itself broadcasts over —
/// upstream's `broadcast_ending_ranges` (indexing.py:221-223):
///
///   ended = [rctx.range_map[c][0][i] for c in consumer_map[x]
///            if c in rctx.range_map and c.op in GroupOp.Broadcastable
///            for i in broadcast_axes(x.shape, c.shape)]
///   broadcast_ending_ranges = list(UOp.sink(*ended).ranges)
fn broadcast_ending_ranges(x: &Arc<UOp>, consumers: &[Arc<UOp>], ctx: &IndexingContext) -> Vec<Arc<UOp>> {
    let mut ended = Vec::new();
    for consumer in consumers.iter().filter(|c| is_broadcastable_op(c)) {
        let (Some((_, axes)), Some((consumer_in, _))) = (broadcast_axes(x, consumer), ctx.get_ranges(consumer)) else {
            continue;
        };
        ended.extend(axes.into_iter().filter_map(|axis| consumer_in.get(axis).cloned()));
    }
    ended.iter().flat_map(collect_ranges_from_uop).collect()
}

fn is_broadcastable_op(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Binary(..) | Op::Ternary(..))
}

/// Pattern matcher for generating the realize map (`pm_generate_realize_map`).
///
/// Marks which UOps need to be materialized to buffers:
/// - SINK sources (if not always-contiguous)
/// - COPY, CONTIGUOUS, STORE (always realized)
/// - Sources of COPY, MSTACK, MSELECT (realized if not always-contiguous)
/// - Inputs of custom-kernel CALLs (realized and pinned non-removable)
///
/// Patterns return `None` (no rewrite) — context side-effects mark nodes in the realize map.
pub(crate) fn pm_generate_realize_map() -> &'static crate::TypedPatternMatcher<IndexingContext> {
    crate::cached_patterns! {
        @context IndexingContext;

        // `realize_custom_kernel_srcs` (indexing.py:44-49): a hand-written kernel
        // reads its inputs through PARAM slots, so each one must already be a
        // buffer — and must stay one, hence non-removable.
        _c @ Call { body, args, info: _ }
            if matches!(body.op(), Op::Sink(..) | Op::Program(..)) => {
            for arg in args {
                let src = through_reshapes(arg);
                if !is_always_contiguous(src) {
                    ctx.mark_realize_non_removable(src);
                }
            }
            None
        },

        // Always realize STORE, and realize its value first when it reads the
        // same base buffer (WAR hazard: without the temp, overlapping
        // self-assigns can read a value that an earlier loop iteration
        // already overwrote).
        x @ Store { index, value } => {
            ctx.mark_realize_pending(x);
            // `realize_store_after_src` (indexing.py:37-40): a SLICE that is the
            // direct source of the STORE needs no buffer of its own — the store
            // target already is the output. A movement op on the destination
            // means the two do not line up, so the SLICE keeps its buffer.
            if matches!(value.op(), Op::Slice(..))
                && ctx.should_realize(value)
                && !index.any_in_subtree(|n| {
                    matches!(n.op(), Op::Shrink(..) | Op::Permute(..) | Op::Flip(..) | Op::Pad(..))
                })
            {
                ctx.clear_realize(value);
            }
            if !self_read_misses_store(index, value, index.base().id) {
                ctx.mark_realize_non_removable(value);
            }
            None
        },
        x @ Contiguous { src: _ } => { ctx.mark_realize_all(x).ok(); None },
        x @ Copy { src, .. } => {
            ctx.mark_realize_all(x).ok();
            if !is_always_contiguous(&src.base()) {
                ctx.mark_realize_all(src).ok();
            }
            None
        },
        // MStack/MSelect → realize sources
        x @ MStack { buffers: _ } => {
            for src in x.op().sources() {
                // realize_srcs: guard on src.base.op, realize src.
                if !is_always_contiguous(&src.base()) {
                    ctx.mark_realize_all(&src).ok();
                }
            }
            None
        },
        x @ MSelect { device_index: _ } => {
            for src in x.op().sources() {
                // realize_srcs: guard on src.base.op, realize src.
                if !is_always_contiguous(&src.base()) {
                    ctx.mark_realize_all(&src).ok();
                }
            }
            None
        },
    }
}

/// `node` seen through any number of RESHAPEs; other movement ops stay.
fn through_reshapes(node: &Arc<UOp>) -> &Arc<UOp> {
    let mut cur = node;
    while let Op::Reshape(ops::Reshape { src, .. }) = cur.op() {
        cur = src;
    }
    cur
}

/// Whether two shrink windows over one buffer provably cover disjoint memory.
///
/// True when on some axis the offset gap, once simplified, clears the other
/// window in either direction: `a` starts at or past the end of `b`, or ends
/// at or before the start of `b`. That is the scan case, where step `t` reads
/// slot `t` and writes slot `t + 1`, and its strided variants.
fn windows_disjoint(a: (&[Arc<UOp>], &[Arc<UOp>]), b: (&[Arc<UOp>], &[Arc<UOp>])) -> bool {
    use svod_ir::uop::cached_property::CachedProperty;
    use svod_ir::uop::properties::SoundVminVmaxProperty;

    let ((a_off, a_size), (b_off, b_size)) = (a, b);
    a_off.len() == b_off.len()
        && a_off.iter().zip(b_off).zip(a_size.iter().zip(b_size)).any(|((ao, bo), (asz, bsz))| {
            let (Some(asz), Some(bsz)) = (asz.vmax().try_int(), bsz.vmax().try_int()) else { return false };
            let Ok(gap) = ao.try_sub(bo) else { return false };
            let gap = crate::rewrite::graph_rewrite(crate::symbolic::patterns::symbolic(), gap, &mut ());
            let Some((min, max)) = *SoundVminVmaxProperty::get(&gap) else { return false };
            min.try_int().is_some_and(|min| min >= bsz) || max.try_int().is_some_and(|max| max <= -asz)
        })
}

/// The `(offsets, sizes, viewed shape)` of a SHRINK that windows `base`
/// directly, seen through any number of reshapes.
type ShrinkView = (Vec<Arc<UOp>>, Vec<Arc<UOp>>, Option<svod_ir::shape::Shape>);

/// `None` for offsets without sound bounds: reading through their casts
/// would miss a wrap-around, so such a SHRINK is not a window we can place.
fn shrink_over_base(node: &Arc<UOp>, base: u64) -> Option<ShrinkView> {
    use svod_ir::uop::cached_property::CachedProperty;
    use svod_ir::uop::properties::SoundVminVmaxProperty;

    let Op::Shrink(ops::Shrink { src, offsets, sizes }) = node.op() else { return None };
    (through_reshapes(src).id == base && SoundVminVmaxProperty::get(offsets).is_some())
        .then(|| (extract_shape_uops(offsets), extract_shape_uops(sizes), src.shape().ok().flatten().cloned()))
}

/// Whether every read of `base` inside a STORE's value misses the slot the
/// STORE writes (vacuously, when the value never reads `base`).
///
/// The STORE rule's WAR temp exists for self-assigns that overlap; a scan
/// reading slot `t` and writing slot `t + 1` does not, and paying for the temp
/// there costs one extra kernel and one extra buffer round trip per time step.
///
/// Every unproven case keeps the temp: a STORE target that is not a SHRINK, a
/// size whose bound exceeds the gap, a non-SHRINK read of the base, a movement
/// op other than RESHAPE between the SHRINK and the base, or two windows viewed
/// through different shapes.
fn self_read_misses_store(index: &Arc<UOp>, value: &Arc<UOp>, base: u64) -> bool {
    let store = shrink_over_base(index, base);
    let mut aliasing: HashSet<u64> = HashSet::new();
    for node in value.toposort() {
        if node.id != base && !node.op().sources().iter().any(|src| aliasing.contains(&src.id)) {
            continue;
        }
        if let Some((store_off, store_size, store_shape)) = &store
            && let Some((off, size, shape)) = shrink_over_base(&node, base)
            && shape == *store_shape
            && windows_disjoint((store_off, store_size), (&off, &size))
        {
            continue;
        }
        aliasing.insert(node.id);
    }
    !aliasing.contains(&value.id)
}

/// Check if a UOp is always contiguous (doesn't need realization).
///
/// Aligned with ALWAYS_CONTIGUOUS.
/// When the source of a STAGE is in this set, the STAGE gets `removable: false`,
/// preventing it from being inlined by buffer removal.
pub(crate) fn is_always_contiguous(uop: &Arc<UOp>) -> bool {
    matches!(
        uop.op(),
        Op::Contiguous(..)
            | Op::After(..)
            | Op::Buffer(..)
            | Op::Slice(..)
            | Op::Const(_)
            | Op::Bind(..)
            | Op::MSelect(..)
            | Op::MStack(..)
            | Op::Param(..)
            | Op::Load(..)
            | Op::Call(..)
            | Op::Function(..)
    )
}

/// Check if a UOp represents constant true (handles unsimplified OR expressions).
fn is_const_true(uop: &Arc<UOp>) -> bool {
    match uop.op() {
        Op::Const(cv) => matches!(cv.0, ConstValue::Bool(true)),
        Op::Binary(BinaryOp::Or, a, b) => is_const_true(a) && is_const_true(b),
        _ => false,
    }
}

/// Merge ranges from multiple consumers. Creates new ranges and marks realization when needed.
#[instrument(skip(uop, consumer_rngs, ctx), fields(uop_id = uop.id))]
pub(crate) fn merge_consumer_ranges(
    uop: &Arc<UOp>,
    consumer_rngs: &[Vec<Arc<UOp>>],
    ctx: &mut IndexingContext,
) -> svod_ir::Result<Vec<Arc<UOp>>> {
    let Some(shape) = uop.shape()? else {
        return Ok(Vec::new());
    };

    let num_dims = shape.len();

    // Transpose: consumer_rngs[consumer_idx][dim_idx] → all_rngs[dim_idx][consumer_idx]
    let mut all_rngs: Vec<Vec<Arc<UOp>>> = vec![Vec::new(); num_dims];
    for consumer_rng in consumer_rngs {
        for (dim_idx, range) in consumer_rng.iter().enumerate() {
            if dim_idx < num_dims {
                all_rngs[dim_idx].push(Arc::clone(range));
            }
        }
    }

    let mut out_rngs = Vec::new();
    let mut realize_axes = Vec::new();

    // Compute all_all_same FIRST — if ANY dimension
    // has incompatible ranges across consumers, ALL dimensions get realized.
    // With PCONTIG=0 (default): condition per-dim = `all_all_same || (PCONTIG && all_same(dim))`.
    // When all_all_same=False and PCONTIG=0, this is always False → all dims realized.
    let all_all_same = all_rngs.iter().all(|dim_ranges| {
        // `all_same([])` is True (helpers.py:31); upstream's `zip(*consumer_rngs)`
        // (indexing.py:249) truncates such a dim away rather than realizing it.
        if dim_ranges.is_empty() {
            return true;
        }
        if dim_ranges.iter().skip(1).all(|r| Arc::ptr_eq(&dim_ranges[0], r)) {
            return true;
        }
        let indices: Vec<_> = dim_ranges.iter().map(|r| r.get_idx()).collect();
        all_ranges_same(&indices)
    });

    for (dim_idx, dim_ranges) in all_rngs.iter().enumerate() {
        if dim_ranges.is_empty() {
            out_rngs.push(ctx.new_range(&shape[dim_idx], AxisType::Weak));
            realize_axes.push(dim_idx);
            continue;
        }

        // FAST PATH: If all ranges are pointer-equal, return original unchanged
        if dim_ranges.iter().skip(1).all(|r| Arc::ptr_eq(&dim_ranges[0], r)) && all_all_same {
            out_rngs.push(Arc::clone(&dim_ranges[0]));
            continue;
            // all_all_same=False but this dim is same → still realize (PCONTIG=0 behavior)
        }

        let indices: Vec<_> = dim_ranges.iter().map(|r| r.get_idx()).collect();
        let valids: Vec<_> = dim_ranges.iter().map(|r| r.get_valid()).collect();
        let ranges_same = all_ranges_same(&indices);

        // if all_all_same or (PCONTIG and all_same): merge
        // With PCONTIG=0 (default): only merge when all_all_same is True.
        if all_all_same {
            debug!(dim_idx, ranges_same, all_all_same, "merge_consumer_ranges: merging dimension");
            let merged_idx = Arc::clone(&indices[0]);
            let merged_valid = if valids.len() == 1 {
                Arc::clone(&valids[0])
            } else {
                valids.iter().skip(1).try_fold(Arc::clone(&valids[0]), |acc, v| acc.try_or_op(v))?
            };

            // Build WHERE(valid, idx, Invalid) and simplify immediately.
            // Without this simplification, unsimplified WHERE/Not chains accumulate
            // and cause oscillation in downstream symbolic passes.
            let merged_range = if is_const_true(&merged_valid) {
                merged_idx
            } else {
                let raw = UOp::try_where(merged_valid, merged_idx, UOp::invalid_marker())?;
                // Uses full `symbolic` here (not symbolic_simple)
                crate::rewrite::graph_rewrite(crate::symbolic::patterns::symbolic(), raw, &mut ())
            };
            out_rngs.push(merged_range);
        } else {
            debug!(dim_idx, "merge_consumer_ranges: creating NEW Weak range (ranges not compatible)");
            out_rngs.push(ctx.new_range(&shape[dim_idx], AxisType::Weak));
            realize_axes.push(dim_idx);
        }
    }

    if !realize_axes.is_empty() {
        debug!(realize_axes = ?realize_axes, "range conflict detected - marking axes for realization");
        ctx.mark_realize(uop, realize_axes.clone());
    }

    Ok(out_rngs)
}

/// Assign input/output ranges for each UOp via reverse toposort traversal.
#[instrument(skip_all)]
fn assign_ranges(
    reverse_topo: &[Arc<UOp>],
    consumer_map: &ConsumerMap,
    ctx: &mut IndexingContext,
) -> svod_ir::Result<()> {
    // Local variable for ending_ranges - only used within this function
    let mut ending_ranges: HashMap<UOpKey, Vec<Arc<UOp>>> = HashMap::new();

    for x in reverse_topo {
        if matches!(x.op(), Op::Unique(_)) {
            continue;
        }

        // Skip callable boundaries, AFTER, and MSTACK/MSELECT during range assignment.
        if matches!(
            x.op(),
            Op::Call(..) | Op::Function(..) | Op::Linear(..) | Op::After(..) | Op::MStack(..) | Op::MSelect(..)
        ) {
            continue;
        }

        let _span = info_span!("assign_range", uop_id = x.id, op = x.op().as_ref()).entered();

        let consumers: Vec<Arc<UOp>> =
            consumer_map.get(&UOpKey(x.clone())).into_iter().flatten().map(|c| Arc::clone(&c.0)).collect();
        let consumer_rngs: Vec<Vec<Arc<UOp>>> =
            consumers.iter().filter_map(|c| ctx.get_ranges(c).map(|(inp, _)| broadcast_ranges(c, x, inp))).collect();

        debug!(
            num_consumers = consumers.len(),
            consumer_rngs_len = consumer_rngs.len(),
            consumer_ids = ?consumers.iter().map(|c| c.id).collect::<Vec<_>>(),
            "Consumer info"
        );

        // Inherit ending_ranges from consumers
        // ending_ranges propagate from consumers → producers (backward in data flow)
        let mut inherited_ending: Vec<Arc<UOp>> = Vec::new();
        for consumer in &consumers {
            inherited_ending.extend(ending_ranges.get(&UOpKey(consumer.clone())).cloned().unwrap_or_default());
        }
        if !inherited_ending.is_empty() {
            debug!(
                node_id = x.id,
                inherited_count = inherited_ending.len(),
                consumer_ids = ?consumers.iter().map(|c| c.id).collect::<Vec<_>>(),
                "ending_ranges: node inherits from consumers"
            );
        }
        // `ended` / `broadcast_ending_ranges` (indexing.py:221-223), plus the
        // "fusion decision: REDUCE before the broadcast" row at :225.
        let broadcast_ending = broadcast_ending_ranges(x, &consumers, ctx);
        if matches!(x.op(), Op::Reduce(..)) {
            inherited_ending.extend(broadcast_ending.iter().cloned());
        }
        ending_ranges.insert(UOpKey(x.clone()), inherited_ending);

        let mut out_rngs = if ctx.should_realize(x) {
            // Realized op: create fresh ranges for all dimensions.
            // CONTIGUOUS, COPY, STORE, and ops marked by ending_ranges all land here.
            let shape = match x.op() {
                Op::Store(ops::Store { index, .. }) => {
                    let s = index.shape()?.cloned();
                    // STORE without an inferable index shape is ill-formed —
                    // fall-through would leak unrealized stores into later
                    // passes.
                    if s.is_none() {
                        return Err(svod_ir::Error::StoreMissingShape { uop_id: x.id });
                    }
                    s
                }
                _ => x.shape()?.cloned(),
            };
            if let Some(shape) = shape {
                debug!(
                    node_id = x.id,
                    op = x.op().as_ref(),
                    dims = shape.len(),
                    "REALIZE via realize_map (fresh ranges)"
                );
                let rngs: Vec<_> = shape.iter().map(|s| ctx.new_range(s, AxisType::Weak)).collect();
                let axes: Vec<usize> = (0..shape.len()).collect();
                ctx.realize_map.insert(UOpKey(x.clone()), Some(axes));
                // Clear ending_ranges when realized
                ending_ranges.insert(UOpKey(x.clone()), Vec::new());
                rngs
            } else {
                continue;
            }
        } else if consumer_rngs.is_empty() {
            continue;
        } else if consumer_rngs.len() == 1 {
            consumer_rngs[0].clone()
        } else {
            merge_consumer_ranges(x, &consumer_rngs, ctx)?
        };

        debug!(should_realize = ctx.should_realize(x), out_rngs_len = out_rngs.len(), "output ranges computed");

        // Check ending_ranges FIRST (before in_rngs computation)
        // ending_ranges realization happens BEFORE input ranges
        // This is critical: in_rngs must be computed from the FINAL out_rngs after realization
        let ending = ending_ranges.get(&UOpKey(x.clone())).cloned().unwrap_or_default();
        if !ending.is_empty() {
            debug!(
                ending_count = ending.len(),
                triggers_realization = matches!(x.op(), Op::Reduce(..)) || is_elementwise_op(x),
                "Ending ranges detected (pre-in_rngs check)"
            );
        }
        // Use ending ranges directly without filtering (matches upstream behavior).
        let filtered_ending = ending.clone();

        if !filtered_ending.is_empty() && (matches!(x.op(), Op::Reduce(..)) || is_elementwise_op(x)) {
            if let Some(shape) = x.shape().ok().flatten() {
                // Start with existing realize_axes (from merge_consumer_ranges)
                let mut realize_axes: Vec<usize> = ctx.get_realize_axes(x).cloned().unwrap_or_default();

                // `if not (PCONTIG > 1) or any(any(rr.arg > e.arg ...) ...)`
                // With PCONTIG=0 (default), `not (0 > 1)` = True, so ALL axes are unconditionally
                // realized when ending_ranges are present. This is critical for layernorm-style
                // patterns where `centered = x - mean` is shared between output and variance paths.
                // The ending_ranges from EXPAND (broadcasting mean/inv_std) must trigger full
                // realization of elementwise ops in the backward slice.
                for (i, _r) in out_rngs.iter().enumerate() {
                    if realize_axes.contains(&i) {
                        continue;
                    }
                    realize_axes.push(i);
                }

                debug!(
                    node_id = x.id,
                    op = x.op().as_ref(),
                    ending_count = ending.len(),
                    realize_axes = ?realize_axes,
                    "SELECTIVE REALIZATION via ending_ranges"
                );

                // Clear ending_ranges after handling
                ending_ranges.insert(UOpKey(x.clone()), Vec::new());

                if !realize_axes.is_empty() {
                    // Mark for realization
                    ctx.mark_realize(x, realize_axes.clone());

                    // Selectively replace only realized axes (preserve others)
                    out_rngs = out_rngs
                        .iter()
                        .enumerate()
                        .map(|(i, r)| {
                            if realize_axes.contains(&i) {
                                if let Some(dim) = shape.get(i) {
                                    ctx.new_range(dim, AxisType::Weak)
                                } else {
                                    Arc::clone(r)
                                }
                            } else {
                                Arc::clone(r)
                            }
                        })
                        .collect();
                }
            } else {
                ending_ranges.insert(UOpKey(x.clone()), Vec::new());
            }
        }

        // `ending_ranges[x] += broadcast_ending_ranges` (indexing.py:284): the
        // ranges a consumer broadcasts this node over keep propagating to its
        // producers even when the clears above emptied the inherited set.
        ending_ranges.entry(UOpKey(x.clone())).or_default().extend(broadcast_ending);

        // NOW compute in_rngs from the FINAL out_rngs (after any realization updates)
        let in_rngs = match x.op() {
            Op::Reshape(ops::Reshape { src, .. })
            | Op::Permute(ops::Permute { src, .. })
            | Op::Expand(ops::Expand { src, .. })
            | Op::Pad(ops::Pad { src, .. })
            | Op::Shrink(ops::Shrink { src, .. })
            | Op::Flip(ops::Flip { src, .. }) => {
                if let Some(in_shape) = src.shape()? {
                    apply_movement_op(x.op(), in_shape, &out_rngs)
                } else {
                    out_rngs.clone()
                }
            }
            // STACK prepends a selection axis; its sources live one axis below.
            Op::Stack(..) => out_rngs.iter().skip(1).cloned().collect(),
            Op::Reduce(ops::Reduce { src, num_axes, .. }) if *num_axes > 0 => {
                if let Some(in_shape) = src.shape()? {
                    if tracing::enabled!(tracing::Level::TRACE) {
                        let out_shape = x.shape()?;
                        trace!(
                            uop.id = x.id,
                            reduce.num_axes = *num_axes,
                            in_shape.len = in_shape.len(),
                            out_shape.len = ?out_shape.as_ref().map(|s| s.len()),
                            out_rngs.len = out_rngs.len(),
                            "tensor REDUCE range assignment"
                        );
                    }

                    let mut rngs = Vec::with_capacity(in_shape.len());
                    rngs.extend(in_shape.iter().take(*num_axes).map(|s| ctx.new_range(s, AxisType::Reduce)));
                    rngs.extend(out_rngs.iter().cloned());
                    rngs
                } else {
                    out_rngs.clone()
                }
            }
            _ => out_rngs.clone(),
        };

        debug!(in_rngs_len = in_rngs.len(), "input ranges computed");

        // EXPAND marks ranges as ending when broadcasting to static dimensions
        // "if the EXPAND is used to inject a range, we don't mark it as ending_ranges. otherwise we do."
        if let Op::Expand(ops::Expand { new_shape, .. }) = x.op() {
            // Check if new_shape is all static (no RANGE ops being injected in the shape)
            let shape_is_static = extract_shape_from_uop(new_shape).iter().all(|s| match s {
                SInt::Const(_) | SInt::Infer => true,
                SInt::Symbolic(uop) => !matches!(uop.op(), Op::Range(..)),
            });

            debug!(
                expand_id = x.id,
                shape_is_static = shape_is_static,
                in_rngs_len = in_rngs.len(),
                out_rngs_len = out_rngs.len(),
                in_rngs_ids = ?in_rngs.iter().map(|r| (r.id, format!("{:?}", std::mem::discriminant(r.op())))).collect::<Vec<_>>(),
                out_rngs_ids = ?out_rngs.iter().map(|r| (r.id, format!("{:?}", std::mem::discriminant(r.op())))).collect::<Vec<_>>(),
                "ending_ranges: EXPAND being processed"
            );

            if shape_is_static {
                // Ranges that changed (in_rngs != out_rngs) are "ending"
                // These are the output ranges that were collapsed to const 0 in in_rngs
                // upstream `.ranges.keys()` returns ALL range types without filtering.
                let mut changed_ranges: Vec<Arc<UOp>> = Vec::new();
                for (inp, out) in in_rngs.iter().zip(out_rngs.iter()) {
                    if !Arc::ptr_eq(inp, out) {
                        changed_ranges.extend(collect_ranges_from_uop(out));
                    }
                }

                if !changed_ranges.is_empty() {
                    debug!(
                        expand_id = x.id,
                        changed_ranges_count = changed_ranges.len(),
                        changed_range_ids = ?changed_ranges.iter().map(|r| r.id).collect::<Vec<_>>(),
                        "ending_ranges: EXPAND marking ranges as ending"
                    );
                    let mut ending = ending_ranges.get(&UOpKey(x.clone())).cloned().unwrap_or_default();
                    ending.extend(changed_ranges);
                    ending_ranges.insert(UOpKey(x.clone()), ending);
                }
            }
        }

        ctx.set_ranges(x, in_rngs, out_rngs);
    }
    Ok(())
}

// ============================================================================
// Movement Op Helpers (from helpers.rs)
// ============================================================================

/// Transform ranges through a movement op (SHRINK, PERMUTE, FLIP, EXPAND, PAD, RESHAPE).
fn movement_key(op: &Op, in_shape: &[SInt], rngs: &[Arc<UOp>]) -> MovementKey {
    let arg = match op {
        Op::Permute(ops::Permute { axes, .. }) => tuple_key(axes.iter().map(|&a| UOp::index_const(a as i64)).collect()),
        Op::Flip(ops::Flip { axes, .. }) => tuple_key(axes.iter().map(|&f| UOp::index_const(i64::from(f))).collect()),
        // Every other movement arg is itself a UOp source behind the moved one.
        _ => tuple_key(op.sources().iter().skip(1).cloned().collect()),
    };
    MovementKey { op: std::mem::discriminant(op), arg, in_shape: shape_key(in_shape), rngs: tuple_key(rngs.to_vec()) }
}

pub fn apply_movement_op(op: &Op, in_shape: &[SInt], rngs: &[Arc<UOp>]) -> Vec<Arc<UOp>> {
    cached(&MOVEMENT_CACHE, movement_key(op, in_shape, rngs), || apply_movement_op_uncached(op, in_shape, rngs))
}

fn apply_movement_op_uncached(op: &Op, in_shape: &[SInt], rngs: &[Arc<UOp>]) -> Vec<Arc<UOp>> {
    match op {
        Op::Shrink(ops::Shrink { offsets, .. }) => {
            // Matches upstream:
            // case Ops.SHRINK: rngs = tuple(a if ss == 0 else a+ss for a,(ss,_) in zip(rngs, arg))
            let begin_uops = extract_shape_uops(offsets);
            rngs.iter()
                .zip(begin_uops.iter())
                .map(|(rng, begin)| {
                    // Skip add when begin is zero (concrete optimization)
                    if is_const_zero(begin) {
                        Arc::clone(rng)
                    } else {
                        rng.try_add(begin).expect("SHRINK: try_add failed")
                    }
                })
                .collect()
        }

        Op::Permute(ops::Permute { axes, .. }) => {
            let inv_perm = argsort(axes);
            inv_perm.iter().map(|&i| Arc::clone(&rngs[i])).collect()
        }

        Op::Flip(ops::Flip { axes: flips, .. }) => rngs
            .iter()
            .zip(in_shape.iter())
            .zip(flips.iter())
            .map(|((rng, shape), &flip)| {
                if !flip {
                    Arc::clone(rng)
                } else {
                    let shape_uop = shape.arithmetic_uop();
                    let shape_minus_1 = shape_uop.try_sub(&UOp::index_const(1)).unwrap();
                    shape_minus_1.try_sub(rng).unwrap()
                }
            })
            .collect(),

        Op::Expand(ops::Expand { new_shape, .. }) => {
            let new_shape_vals = extract_shape_from_uop(new_shape);

            // When rngs.len() < new_shape_vals.len(), pad from the left with CONST(0)
            // to align indices with trailing dimensions (same logic as RESHAPE padding).
            let padded_rngs: Vec<Arc<UOp>> = if rngs.len() < new_shape_vals.len() {
                let padding = new_shape_vals.len() - rngs.len();
                let mut v = Vec::with_capacity(new_shape_vals.len());
                for _ in 0..padding {
                    v.push(UOp::index_const(0));
                }
                v.extend(rngs.iter().cloned());
                v
            } else {
                rngs.to_vec()
            };

            // Also pad in_shape from the left with CONST(1) if needed
            let padded_in_shape: Vec<SInt> = if in_shape.len() < new_shape_vals.len() {
                let padding = new_shape_vals.len() - in_shape.len();
                let mut v = Vec::with_capacity(new_shape_vals.len());
                for _ in 0..padding {
                    v.push(SInt::Const(1));
                }
                v.extend(in_shape.iter().cloned());
                v
            } else {
                in_shape.to_vec()
            };

            padded_rngs
                .iter()
                .zip(padded_in_shape.iter())
                .zip(new_shape_vals.iter())
                .map(|((rng, in_sh), out_sh)| {
                    let expanding = match (in_sh, out_sh) {
                        (SInt::Const(1), SInt::Const(n)) if *n > 1 => true,
                        (SInt::Const(1), SInt::Symbolic(_)) => true,
                        _ => false,
                    };
                    if expanding { UOp::index_const(0) } else { Arc::clone(rng) }
                })
                // The leading axes an EXPAND adds to a lower-rank source index
                // nothing in it (Tinygrad `indexing.py`: `rngs[len(arg):]`).
                .skip(new_shape_vals.len().saturating_sub(in_shape.len()))
                .collect()
        }

        Op::Pad(ops::Pad { begin_pads, end_pads, .. }) => {
            let begin_uops = extract_shape_uops(begin_pads);
            let end_uops = extract_shape_uops(end_pads);
            rngs.iter()
                .zip(in_shape.iter())
                .zip(begin_uops.iter().zip(end_uops.iter()))
                .map(|((rng, shape), (begin, end))| {
                    if is_const_zero(begin) && is_const_zero(end) {
                        return Arc::clone(rng);
                    }
                    let shape_plus_begin = shape.arithmetic_uop().try_add(begin).unwrap();
                    let valid_low = rng.try_cmplt(begin).unwrap().not();
                    let valid_high = rng.try_cmplt(&shape_plus_begin).unwrap();
                    let valid = valid_low.try_and_op(&valid_high).unwrap();
                    // graph_rewrite(validity, symbolic+pm_simplify_valid)
                    static PAD_SIMPLIFY: std::sync::LazyLock<crate::TypedPatternMatcher> =
                        std::sync::LazyLock::new(|| {
                            crate::symbolic::patterns::symbolic()
                                + crate::symbolic::valid_simplification::pm_simplify_valid()
                        });
                    let valid = crate::rewrite::graph_rewrite(&*PAD_SIMPLIFY, valid, &mut ());
                    let adjusted_rng = rng.try_sub(begin).unwrap();
                    UOp::try_where(valid, adjusted_rng, UOp::invalid_marker()).unwrap()
                })
                .collect()
        }

        Op::Reshape(ops::Reshape { new_shape, .. }) => {
            let new_shape_vals = extract_shape_from_uop(new_shape);

            // Optimization: If in_shape == new_shape, this is a no-op reshape
            if in_shape.len() == new_shape_vals.len() {
                let mut is_same_shape = true;
                for (in_dim, out_dim) in in_shape.iter().zip(new_shape_vals.iter()) {
                    match (in_dim, out_dim) {
                        (SInt::Const(a), SInt::Const(b)) if a == b => continue,
                        (SInt::Symbolic(a), SInt::Symbolic(b)) if a.id == b.id => continue,
                        _ => {
                            is_same_shape = false;
                            break;
                        }
                    }
                }
                if is_same_shape {
                    return rngs.to_vec();
                }
            }

            // PLACEHOLDER canonicalization + reshape
            with_placeholder_canonicalization(rngs, |canonical| {
                apply_reshape_core(in_shape, &new_shape_vals, canonical)
            })
        }

        _ => panic!("apply_movement_op called with non-movement op: {:?}", op),
    }
}

/// Core RESHAPE: flatten `rngs` by `out_shape` strides, decompose into `in_shape` via FloorMod/FloorDiv,
/// then run full symbolic simplification.
///
/// Matches upstream `_apply_reshape`.
/// Callers should PLACEHOLDER-canonicalize `rngs` before calling this.
fn apply_reshape_core(in_shape: &[SInt], out_shape: &[SInt], rngs: &[Arc<UOp>]) -> Vec<Arc<UOp>> {
    use svod_ir::rewrite::graph_rewrite;

    let key =
        ReshapeKey { in_shape: shape_key(in_shape), out_shape: shape_key(out_shape), rngs: tuple_key(rngs.to_vec()) };
    if let Some(hit) = RESHAPE_CACHE.lock().expect("movement cache poisoned").get(&key) {
        return hit.clone();
    }

    // Pad with CONST(0) on the left when rngs.len() < out_shape.len()
    // (trailing-dimension alignment for partial INDEX)
    let padded_rngs: Vec<Arc<UOp>> = if rngs.len() < out_shape.len() {
        let padding = out_shape.len() - rngs.len();
        let mut v = Vec::with_capacity(out_shape.len());
        for _ in 0..padding {
            v.push(UOp::index_const(0));
        }
        v.extend(rngs.iter().cloned());
        v
    } else {
        rngs.to_vec()
    };

    // Flatten: combined = sum(acc_i * rng_i) with acc computed from out_shape
    let mut acc = UOp::index_const(1);
    let mut axes_in = Vec::new();
    for (shape_dim, rng) in out_shape.iter().zip(padded_rngs.iter()).rev() {
        axes_in.push(acc.try_mul(rng).unwrap());
        let dim_uop = shape_dim.arithmetic_uop();
        acc = acc.try_mul(&dim_uop).unwrap();
    }
    let combined = axes_in.into_iter().reduce(|a, b| a.try_add(&b).unwrap()).unwrap_or_else(|| UOp::index_const(0));

    // Unflatten `combined` into in_shape via FloorMod/FloorDiv. Redundant mods are
    // dropped by the downstream RESHAPE_SIMPLIFY chain.
    let mut axes_out = Vec::new();
    let mut remaining = combined;
    for shape_dim in in_shape.iter().rev() {
        let dim_uop = shape_dim.arithmetic_uop();
        axes_out.push(remaining.try_mod(&dim_uop).unwrap());
        remaining = remaining.try_div(&dim_uop).unwrap();
    }
    axes_out.reverse();

    // Simplify ("This simplify is doing a lot of heavy lifting")
    static RESHAPE_SIMPLIFY: std::sync::LazyLock<crate::TypedPatternMatcher> = std::sync::LazyLock::new(|| {
        crate::symbolic::patterns::symbolic()
            + crate::symbolic::valid_simplification::pm_simplify_valid()
            + crate::symbolic::valid_simplification::pm_drop_and_clauses()
    });
    let simplified = graph_rewrite(&*RESHAPE_SIMPLIFY, UOp::sink(axes_out), &mut ());
    let result = match simplified.op() {
        Op::Sink(ops::Sink { sources, .. }) => sources.iter().cloned().collect(),
        _ => vec![simplified],
    };
    RESHAPE_CACHE.lock().expect("movement cache poisoned").insert(key, result.clone());
    result
}

/// Reshape ranges from `out_shape` to `in_shape` via flatten + unflatten.
///
/// Public wrapper around `apply_reshape_core` with PLACEHOLDER canonicalization.
/// Used by `flatten_bufferize` to convert multi-dim ranges to 1D.
pub fn apply_reshape_ranges(in_shape: &[SInt], out_shape: &[SInt], rngs: &[Arc<UOp>]) -> Vec<Arc<UOp>> {
    with_placeholder_canonicalization(rngs, |canonical| apply_reshape_core(in_shape, out_shape, canonical))
}

/// Canonicalize RANGE UOps to PLACEHOLDER before calling `f`, then restore.
fn with_placeholder_canonicalization(rngs: &[Arc<UOp>], f: impl FnOnce(&[Arc<UOp>]) -> Vec<Arc<UOp>>) -> Vec<Arc<UOp>> {
    let sink = UOp::sink(rngs.to_vec());
    // Canonicalize only live/in-scope ranges.
    let in_scope = sink.in_scope_ranges();
    let ranges_in_expr: Vec<Arc<UOp>> = sink.ranges().iter().filter(|r| in_scope.contains(&r.id)).cloned().collect();

    let mut sub_map: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    let mut reverse_map: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    let mut reverse_axis_map: HashMap<usize, Arc<UOp>> = HashMap::new();
    for (i, r) in ranges_in_expr.iter().enumerate() {
        let Op::Range(ops::Range { end, .. }) = r.op() else { continue };
        let placeholder = UOp::range_axis(end.clone(), AxisId::Renumbered(i), AxisType::Placeholder);
        sub_map.insert(UOpKey(r.clone()), placeholder.clone());
        reverse_map.insert(UOpKey(placeholder), r.clone());
        reverse_axis_map.insert(i, r.clone());
    }

    if sub_map.is_empty() {
        return f(rngs);
    }

    let canonical_sink = sink.substitute(&sub_map);
    let canonical_rngs: Vec<Arc<UOp>> = match canonical_sink.op() {
        Op::Sink(ops::Sink { sources, .. }) => sources.iter().cloned().collect(),
        _ => vec![canonical_sink],
    };

    let result = f(&canonical_rngs);

    let result_sink = UOp::sink(result);
    let restored = result_sink.substitute(&reverse_map);
    let mut output: Vec<Arc<UOp>> = match restored.op() {
        Op::Sink(ops::Sink { sources, .. }) => sources.iter().cloned().collect(),
        _ => vec![restored],
    };

    // If rewrite changed placeholder internals (e.g., `end` expr), structural
    // reverse_map can miss restoration. Recover by axis id.
    let mut axis_restore_map: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    for r in UOp::sink(output.clone()).ranges().iter() {
        if let Op::Range(ops::Range { axis_id: AxisId::Renumbered(i), axis_type: AxisType::Placeholder, .. }) = r.op()
            && let Some(orig) = reverse_axis_map.get(i)
        {
            axis_restore_map.insert(UOpKey(r.clone()), orig.clone());
        }
    }
    if !axis_restore_map.is_empty() {
        let axis_restored = UOp::sink(output).substitute(&axis_restore_map);
        output = match axis_restored.op() {
            Op::Sink(ops::Sink { sources, .. }) => sources.iter().cloned().collect(),
            _ => vec![axis_restored],
        };
    }

    debug_assert!(
        !output.iter().any(|r| {
            let scope = r.in_scope_ranges();
            r.ranges()
                .into_iter()
                .filter(|rng| scope.contains(&rng.id))
                .any(|rng| matches!(rng.op(), Op::Range(ops::Range { axis_type: AxisType::Placeholder, .. })))
        }),
        "Placeholder-typed ranges leaked into output"
    );

    output
}

/// Check if a UOp is a constant zero.
fn is_const_zero(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Const(cv) if cv.0 == ConstValue::Int(0))
}

/// Extract individual dimensions from a scalar/STACK shape argument.
/// Returns individual element UOps — may be CONST or symbolic expressions.
/// Matches upstream `marg` which extracts via `sgep`.
fn extract_shape_uops(uop: &Arc<UOp>) -> Vec<Arc<UOp>> {
    match uop.op() {
        Op::Cast(ops::Cast { src, .. }) | Op::BitCast(ops::BitCast { src, .. }) => extract_shape_uops(src),
        Op::Stack(ops::Stack { sources }) => sources.to_vec(),
        Op::Const(_) => vec![uop.clone()],
        Op::VConst(ops::VConst { values }) => values
            .iter()
            .map(|cv| match cv {
                ConstValue::Int(n) => UOp::index_const(*n),
                ConstValue::UInt(n) => UOp::index_const(*n as i64),
                _ => panic!("Expected int/uint constant in VConst shape uops"),
            })
            .collect(),
        _ if uop.dtype().is_int() => vec![uop.clone()],
        _ => panic!("expected STACK or scalar integer for shape uops, got {:?}", uop.op()),
    }
}

/// Extract shape from a UOp (for RESHAPE new_shape, EXPAND new_shape).
fn extract_shape_from_uop(uop: &Arc<UOp>) -> Vec<SInt> {
    match uop.op() {
        Op::Cast(ops::Cast { src, .. }) | Op::BitCast(ops::BitCast { src, .. }) => extract_shape_from_uop(src),
        Op::Stack(ops::Stack { sources }) => sources
            .iter()
            .map(|source| match source.op() {
                Op::Const(cv) => match cv.0 {
                    ConstValue::Int(n) => SInt::Const(n as usize),
                    ConstValue::UInt(n) => SInt::Const(n as usize),
                    _ => SInt::Symbolic(source.clone()),
                },
                _ => SInt::Symbolic(source.clone()),
            })
            .collect(),
        Op::Const(cv) => match cv.0 {
            ConstValue::Int(n) => vec![SInt::Const(n as usize)],
            _ => panic!("Expected int constant for shape"),
        },
        // VConst with empty values = scalar (0-d tensor)
        Op::VConst(ops::VConst { values }) if values.is_empty() => vec![],
        Op::VConst(ops::VConst { values }) => values
            .iter()
            .map(|cv| match cv {
                ConstValue::Int(n) => SInt::Const(*n as usize),
                ConstValue::UInt(n) => SInt::Const(*n as usize),
                _ => panic!("Expected int/uint constant in VConst shape"),
            })
            .collect(),
        _ if uop.dtype().is_int() => vec![SInt::Symbolic(uop.clone())],
        _ => panic!("expected STACK or scalar integer for shape, got {:?}", uop.op()),
    }
}

// ============================================================================
// Range Utilities (from helpers.rs)
// ============================================================================

/// Check if two range lists are pointer-equal (same UOps).
pub fn ranges_equal(ranges1: &[Arc<UOp>], ranges2: &[Arc<UOp>]) -> bool {
    ranges1.len() == ranges2.len() && ranges1.iter().zip(ranges2).all(|(r1, r2)| Arc::ptr_eq(r1, r2))
}

/// Check if all ranges have identical index expressions (ignoring validity masks).
pub fn all_ranges_same(ranges: &[Arc<UOp>]) -> bool {
    if ranges.is_empty() {
        return true;
    }
    let first_idx = ranges[0].get_idx();
    ranges.iter().skip(1).all(|r| Arc::ptr_eq(&first_idx, &r.get_idx()))
}

/// Check if range is dead (size ≤ 1). Uses vmax analysis.
pub fn is_dead_axis(range: &Arc<UOp>) -> bool {
    if !matches!(range.op(), Op::Range(..)) {
        return false;
    }
    match range.vmax() {
        ConstValue::Int(v) => *v <= 0,
        ConstValue::UInt(v) => *v == 0,
        _ => false,
    }
}

/// Check if UOp has no RANGE anywhere in its backward slice (loop-invariant).
///
/// Backed by the cached `RangesProperty`, so this is O(1) after the graph's
/// first traversal instead of a fresh DFS per call.
pub fn no_range(uop: &Arc<UOp>) -> bool {
    uop.ranges().is_empty()
}

/// Extract RANGE size as i64. Returns None for symbolic ranges.
pub fn range_size_as_i64(range: &Arc<UOp>) -> Option<i64> {
    if let Op::Range(ops::Range { end, .. }) = range.op() {
        match end.op() {
            Op::Const(cv) => match cv.0 {
                ConstValue::Int(n) => Some(n),
                ConstValue::UInt(n) => Some(n as i64),
                _ => None,
            },
            _ => None,
        }
    } else {
        None
    }
}

// ============================================================================
// Helpers for patterns (from helpers.rs)
// ============================================================================

/// Check if value is identity for op (Add: 0, Mul: 1, And: -1, Or/Xor: 0).
pub fn is_identity_value(value: &ConstValue, op: &BinaryOp, is_right: bool) -> bool {
    match (op, value) {
        (BinaryOp::Add, ConstValue::Int(0)) => true,
        (BinaryOp::Add, ConstValue::Float(f)) if *f == 0.0 => true,
        (BinaryOp::Sub, ConstValue::Int(0)) if is_right => true,
        (BinaryOp::Sub, ConstValue::Float(f)) if is_right && *f == 0.0 => true,
        (BinaryOp::Mul, ConstValue::Int(1)) => true,
        (BinaryOp::Mul, ConstValue::Float(f)) if *f == 1.0 => true,
        (BinaryOp::FloorDiv, ConstValue::Int(1)) if is_right => true,
        (BinaryOp::Fdiv, ConstValue::Float(f)) if is_right && *f == 1.0 => true,
        (BinaryOp::Or, ConstValue::Int(0)) => true,
        (BinaryOp::Xor, ConstValue::Int(0)) => true,
        (BinaryOp::And, ConstValue::Int(-1)) => true,
        _ => false,
    }
}

/// Check if value is zero/annihilator for op (Mul: 0, And: 0).
pub fn is_zero_value(value: &ConstValue, op: &BinaryOp) -> bool {
    match (op, value) {
        (BinaryOp::Mul, ConstValue::Int(0)) => true,
        (BinaryOp::Mul, ConstValue::Float(f)) if *f == 0.0 => true,
        (BinaryOp::And, ConstValue::Int(0)) => true,
        _ => false,
    }
}

/// Extract the constant value from a UOp if it's a CONST operation.
pub fn get_const_value(uop: &Arc<UOp>) -> Option<ConstValue> {
    match uop.op() {
        Op::Const(cv) => Some(cv.0),
        _ => None,
    }
}

/// Check if a UOp is a constant with a specific value.
pub fn is_const(uop: &Arc<UOp>, value: &ConstValue) -> bool {
    get_const_value(uop).as_ref() == Some(value)
}

// ============================================================================
// Ending Ranges Helpers (for nested reduction detection)
// ============================================================================

/// Collect all RANGE UOps from an expression tree.
fn collect_ranges_from_uop(uop: &Arc<UOp>) -> Vec<Arc<UOp>> {
    use std::collections::HashSet;
    let mut ranges = Vec::new();
    let mut seen = HashSet::new();

    for node in uop.toposort() {
        if matches!(node.op(), Op::Range(..)) {
            let key = UOpKey(Arc::clone(&node));
            if seen.insert(key) {
                ranges.push(node);
            }
        }
    }
    ranges
}

/// Check if UOp is an elementwise operation (matches upstream GroupOp.Elementwise).
fn is_elementwise_op(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Binary(..) | Op::Unary(..) | Op::Ternary(..) | Op::Cast(..) | Op::BitCast(..))
}
