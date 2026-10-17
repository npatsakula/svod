//! Consolidated kernel-graph construction and pipeline orchestration.
//!
//! This module contains:
//! - RangeifyBufferContext for tracking state during kernel splitting
//! - split_store for splitting computation at STORE boundaries
//! - try_get_kernel_graph for full pipeline orchestration
//! - Two-stage reduction splitting (split_reduceop)
//!
//! Consolidated from: kernel_context.rs, split_kernel.rs, pipeline.rs,
//! buffer_cost.rs, split_reduceop.rs

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use indexmap::IndexMap;
use smallvec::SmallVec;
use snafu::{ResultExt, Snafu};
use svod_ir::origin::{OriginId, OriginSet};
use svod_ir::{CallInfo, Op, SInt, UOp, UOpKey};
use tracing::{debug, trace};

pub use svod_ir::KernelInfo;
use svod_ir::ops;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum KernelGraphError {
    #[snafu(display("kernel graph construction failed: {source}"))]
    Ir { source: svod_ir::Error },
    #[snafu(display("kernel graph specification failed: {source}"))]
    Spec { source: crate::spec::SpecError },
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for split_reduceop optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitReduceOpConfig {
    /// Minimum input/output ratio to trigger splitting (default: 32768)
    pub split_threshold: usize,
    /// Max output buffer size as 2^N elements (default: 22 = 4M elements)
    pub output_size_bits: u32,
    /// Max split divisor (default: 256)
    pub max_divisor: usize,
    /// Min split divisor (default: 8)
    pub min_divisor: usize,
    /// Enable/disable the optimization (default: true)
    pub enabled: bool,
}

impl Default for SplitReduceOpConfig {
    fn default() -> Self {
        Self { split_threshold: 32768, output_size_bits: 22, max_divisor: 256, min_divisor: 8, enabled: true }
    }
}

impl SplitReduceOpConfig {
    pub fn max_output_size(&self) -> usize {
        2_usize.pow(self.output_size_bits)
    }
}

// ============================================================================
// RANGEIFY BUFFER CONTEXT
// ============================================================================

/// Context for tracking state during kernel splitting.
///
/// Simplified from original 8 fields to 6 fields, removing Svod-specific
/// `kernel_deps` and `buffer_id_mapping` that are no longer needed after
/// aligning with fix_assign approach.
#[derive(Clone)]
pub struct RangeifyBufferContext {
    pub global_counter: usize,
    pub local_counter: usize,
    pub lunique_counter: usize,
    pub buffer_map: HashMap<UOpKey, Arc<UOp>>,
    /// Bound variables: maps variable name → (variable UOp, optional bound value).
    /// Populated when BIND(variable, CONST) is stripped during kernel splitting.
    /// The UOp is kept for kernel sources; the i64 is the concrete bound value
    /// (None for schedule-loop wrappers — Range-bound variables).
    pub vars: HashMap<String, (Arc<UOp>, Option<i64>)>,
    pub range_counter: usize,
}

impl RangeifyBufferContext {
    pub fn new() -> Self {
        Self::with_lunique_start(0)
    }

    pub fn with_lunique_start(lunique_start: usize) -> Self {
        Self {
            global_counter: 0,
            local_counter: 0,
            lunique_counter: lunique_start,
            buffer_map: HashMap::new(),
            vars: HashMap::new(),
            range_counter: 0,
        }
    }

    pub fn next_global(&mut self) -> usize {
        let id = self.global_counter;
        self.global_counter += 1;
        id
    }

    pub fn next_local(&mut self) -> usize {
        let id = self.local_counter;
        self.local_counter += 1;
        id
    }

    pub fn next_lunique(&mut self) -> usize {
        let id = self.lunique_counter;
        self.lunique_counter += 1;
        id
    }

    pub fn next_range(&mut self) -> usize {
        let id = self.range_counter;
        self.range_counter += 1;
        id
    }

    pub fn has_buffer(&self, buf: &Arc<UOp>) -> bool {
        self.buffer_map.contains_key(&UOpKey(buf.clone()))
    }

    pub fn get_buffer(&self, buf: &Arc<UOp>) -> Option<&Arc<UOp>> {
        self.buffer_map.get(&UOpKey(buf.clone()))
    }

    pub fn map_buffer(&mut self, original: Arc<UOp>, replacement: Arc<UOp>) {
        self.buffer_map.insert(UOpKey(original), replacement);
    }

    /// Track a bound variable UOp and its concrete value.
    pub fn add_var(&mut self, var: Arc<UOp>, value: Option<i64>) {
        let name = match var.op() {
            Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none() => arg.name.clone(),
            _ => None,
        };
        if let Some(name) = name {
            self.vars.insert(name, (var, value));
        }
    }
}

impl Default for RangeifyBufferContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// LOCAL ADD BUFFER CONTEXT (Per-Kernel Context)
// ============================================================================

/// Per-kernel context for tracking state during kernel splitting.
///
/// Based on `LocalAddBufferContext`.
/// This is used within `split_store` for each individual kernel being created.
///
/// IMPORTANT: Uses IndexMap for `map` to maintain insertion order.
/// This is critical because PARAM slot indices are assigned in the order
/// patterns match, and kernel sources must be in the same order for correct
/// buffer indexing during execution.
#[derive(Default)]
pub struct LocalAddBufferContext {
    /// PARAM slot counter (`dg`)
    pub param_slot: usize,
    /// Buffer → AFTER mapping (IndexMap maintains insertion order)
    pub map: IndexMap<UOpKey, Arc<UOp>>,
    /// Bound variables: binding UOp (typically BIND) -> (variable UOp, optional bound value).
    ///
    /// Uses IndexMap to preserve insertion order for CALL source argument parity.
    pub vars: IndexMap<UOpKey, (Arc<UOp>, Option<i64>)>,
    /// Range renumber counter
    pub range: usize,
    /// Optimization hints extracted from CONTIGUOUS.opts (ctx.opts)
    pub opts: Vec<svod_ir::ContiguousHint>,
}

impl LocalAddBufferContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get next PARAM slot index (`ctx.dg`).
    pub fn next_param_slot(&mut self) -> usize {
        let id = self.param_slot;
        self.param_slot += 1;
        id
    }

    /// Get next range renumber index.
    pub fn next_range(&mut self) -> usize {
        let id = self.range;
        self.range += 1;
        id
    }

    /// Track a bound variable and its binding source.
    pub fn add_var(&mut self, binding: Arc<UOp>, var: Arc<UOp>, value: Option<i64>) {
        let var_name = |uop: &Arc<UOp>| match uop.op() {
            Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none() => arg.name.clone(),
            _ => None,
        };
        if let Some(name) = var_name(&var) {
            // Keep latest binding for a variable name while preserving insertion order.
            if let Some(existing_key) = self.vars.iter().find_map(|(k, (existing_var, _))| {
                if var_name(existing_var).as_deref() == Some(name.as_str()) { Some(k.clone()) } else { None }
            }) {
                self.vars.swap_remove(&existing_key);
            }
            self.vars.insert(UOpKey(binding), (var, value));
        }
    }

    /// Map a buffer to its AFTER wrapper.
    pub fn map_buffer(&mut self, buf: Arc<UOp>, after: Arc<UOp>) {
        self.map.insert(UOpKey(buf), after);
    }

    /// Check if buffer is already mapped.
    pub fn has_buffer(&self, buf: &Arc<UOp>) -> bool {
        self.map.contains_key(&UOpKey(buf.clone()))
    }
}

// ============================================================================
// SPLIT STORE INTO CALL WRAPPERS
// ============================================================================

/// Extract the stored value from a STORE/END(STORE) structure.
///
/// Used to check if the stored value is COPY/SLICE without traversing
/// the entire subgraph.
fn extract_stored_value(ret: &Arc<UOp>) -> &Arc<UOp> {
    match ret.op() {
        Op::Store(ops::Store { value, .. }) => value,
        Op::End(ops::End { computation, .. }) => match computation.op() {
            Op::Store(ops::Store { value, .. }) => value,
            _ => ret,
        },
        _ => ret,
    }
}

fn extract_after_callable(deps: &SmallVec<[Arc<UOp>; 4]>) -> Option<Arc<UOp>> {
    deps.iter().find_map(|d| match d.op() {
        Op::Call(..) => Some(d.clone()),
        Op::End(ops::End { computation, .. }) if matches!(computation.op(), Op::Call(..)) => Some(computation.clone()),
        _ => None,
    })
}

/// Split STORE and END operations into individual kernels.
///
/// Based on split_store.
/// Simplified from 280 lines to ~80 lines using LocalAddBufferContext.
pub fn split_store(_ctx: &mut Vec<Arc<UOp>>, x: &Arc<UOp>) -> Option<Arc<UOp>> {
    use super::patterns::{local_to_param_patterns, rangeify_codegen_patterns};
    use crate::rewrite::graph_rewrite_bottom_up;

    trace!(uop_id = x.id, op = ?std::mem::discriminant(x.op()), "split_store: entering");

    // DEVICE ranges are launch lanes and remain open across a callable boundary.
    // Any computational range still makes this an interior STORE. `ranges()`
    // clones every RANGE reachable through the AFTER chain, so consult it only
    // when something is actually in scope.
    let scope = x.in_scope_ranges();
    if !scope.is_empty()
        && x.ranges()
            .into_iter()
            .filter(|range| scope.contains(&range.id))
            .any(|range| !matches!(range.op(), Op::Range(ops::Range { axis_type: svod_ir::AxisType::Device, .. })))
    {
        return None;
    }

    // Verify operation type (only STORE and END(STORE) are valid)
    let is_valid = match x.op() {
        Op::Store(..) => true,
        Op::End(ops::End { computation, .. }) => matches!(computation.op(), Op::Store(..)),
        _ => false,
    };
    if !is_valid {
        return None;
    }

    // Per-kernel context (LocalAddBufferContext)
    let mut lctx = LocalAddBufferContext::new();

    // Context-dependent rewrite per kernel.
    //
    // Context-free movement and flatten-range patterns were already
    // applied in try_get_kernel_graph's pre-pass. Here we only run patterns that
    // need LocalAddBufferContext (Buffer/Param→codegen PARAM, Bind, After, Range renumber,
    // NOOP→zero, Contiguous→extract opts).
    let ret = {
        use std::sync::LazyLock;
        static PM_CTX_DEP: LazyLock<crate::TypedPatternMatcher<LocalAddBufferContext>> =
            LazyLock::new(|| local_to_param_patterns() + rangeify_codegen_patterns());
        graph_rewrite_bottom_up(&*PM_CTX_DEP, x.clone(), &mut lctx)
    };
    // Harvest attribution before the body loses it: `origin` is what this kernel is
    // charged to (the stored value's root), `origins` is everything that fused into it.
    // The fallback walks root-first, so an unattributed store target charges the
    // kernel to the nearest attributed node above it rather than to whichever leaf
    // happens to be deepest.
    let (fallback, origins): (Option<OriginId>, OriginSet) =
        if svod_ir::origin::any_interned() { ret.kernel_attribution() } else { (None, OriginSet::new()) };
    let origin: Option<OriginId> = extract_stored_value(&ret).origin().or(fallback);
    let ret = if origins.is_empty() { ret } else { ret.without_origins() };

    let closed_ranges = match ret.op() {
        Op::End(ops::End { ranges, .. }) if !ranges.is_empty() => Some(ranges.clone()),
        _ => None,
    };

    // Check for COPY/SLICE directly on the stored value.
    // No graph traversal needed — just walk the STORE/END structure.
    let stored = extract_stored_value(&ret);
    let ast = if matches!(stored.op(), Op::Copy(..) | Op::Slice(..)) {
        // Keep host-side effects as direct call bodies.
        if let Some(ranges) = &closed_ranges { stored.end(ranges.clone()) } else { stored.clone() }
    } else {
        // Mark AST SINK structurally so it hash-cons-distinguishes from
        // unmarked SINKs and the gate predicate can short-circuit on it.
        UOp::sink_with_info(vec![ret], KernelInfo::default())
    };

    // Build CALL from context
    // Args: lctx.map.values() (buffer → AFTER mappings) + bound variable sources.
    let sources: SmallVec<[Arc<UOp>; 4]> =
        lctx.map.values().cloned().chain(lctx.vars.keys().map(|k| k.0.clone())).collect();

    let call = ast.call(sources.clone(), CallInfo { origin, origins, ..CallInfo::default() });
    debug!(
        call_id = call.id,
        num_sources = sources.len(),
        map_size = lctx.map.len(),
        vars_size = lctx.vars.len(),
        "split_store: created call"
    );

    Some(call)
}

/// A non-copy kernel reads and writes one device.
///
/// Tinygrad enforces this in `assert_all_same_devices` (`schedule/__init__.py:141`),
/// after the copy-kernel rewrites in `pm_copy_from_store` have had their chance to
/// turn a genuine cross-device store into a COPY. Without it a mixed-device CALL
/// compiles for one device and is handed the other device's pointer.
fn validate_normal_kernel_devices(root: &Arc<UOp>) -> svod_ir::Result<()> {
    for node in root.toposort() {
        let Op::Call(ops::Call { body, args, .. }) = node.op() else { continue };
        if !matches!(body.op(), Op::Sink(..)) {
            continue;
        }

        let mut devices: Vec<svod_dtype::DeviceSpec> = Vec::new();
        for arg in args {
            if matches!(arg.op(), Op::Bind(..)) {
                continue;
            }
            let Some(device) = arg.device_spec() else { continue };
            if !devices.contains(&device) {
                devices.push(device);
            }
        }
        if devices.len() > 1 {
            return Err(svod_ir::Error::KernelSplitMixedDevices { devices });
        }
    }

    Ok(())
}

/// Fix inter-kernel dependencies (like fix_assign).
///
/// Based on upstream.
/// When kernel B reads from a buffer that kernel A writes to, this function
/// ensures kernel B's AFTER node depends on kernel A's AFTER node.
///
/// Uses buf_uop() to walk through AFTER chains and get underlying buffer IDs.
fn fix_assign(root: &Arc<UOp>) -> svod_ir::Result<Arc<UOp>> {
    // Map buf_uop().id -> AFTER node that produces it
    let mut kernel_assign: HashMap<u64, Arc<UOp>> = HashMap::new();
    let mut assign_rep: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    let afters: Vec<Arc<UOp>> = root.toposort().into_iter().filter(|u| matches!(u.op(), Op::After(..))).collect();

    for u in &afters {
        let Op::After(ops::After { passthrough, .. }) = u.op() else {
            continue;
        };
        kernel_assign.insert(passthrough.buf_uop().id, u.clone());
    }

    for u in afters {
        let Op::After(ops::After { passthrough, deps }) = u.op() else {
            continue;
        };

        // Use buf_uop() to get underlying buffer ID (handles AFTER chains)
        let buf_id = passthrough.buf_uop().id;

        // Get callable wrapper from deps.
        let Some(callable) = extract_after_callable(deps) else {
            continue;
        };

        let Op::Call(ops::Call { args, .. }) = callable.op() else {
            continue;
        };
        let sources: SmallVec<[Arc<UOp>; 4]> = args.clone();

        for s in &sources {
            // Check kernel sources for buffer dependencies
            if !matches!(s.op(), Op::Buffer(..) | Op::Param(..)) {
                continue;
            }
            let s_buf_id = s.buf_uop().id;
            if s_buf_id == buf_id {
                continue;
            }
            let Some(a) = kernel_assign.get(&s_buf_id) else {
                continue;
            };

            // Same-kernel check by callable identity. Skip if both AFTERs
            // belong to the same callable — avoids spurious WAR deps between
            // outputs of the same multi-output kernel.
            if let Op::After(ops::After { deps: a_deps, .. }) = a.op()
                && let Some(a_callable) = extract_after_callable(a_deps)
                && Arc::ptr_eq(&a_callable, &callable)
            {
                continue;
            }

            // Cycle detection
            if u.any_in_subtree(|x| matches!(x.op(), Op::After(..)) && x.buf_uop().id == s_buf_id) {
                return Err(svod_ir::Error::KernelSplitDependencyCycle {
                    writer_buffer: buf_id,
                    read_buffer: s_buf_id,
                });
            }

            // Add dependency: a.replace(src=a.src+(u,))
            if let Op::After(ops::After { passthrough: a_passthrough, deps: a_deps }) = a.op() {
                let mut new_deps = a_deps.clone();
                new_deps.push(u.clone());
                let new_a = a_passthrough.after(new_deps);
                assign_rep.insert(UOpKey(a.clone()), new_a.clone());
                kernel_assign.insert(s_buf_id, new_a);
            }
        }
    }

    Ok(if assign_rep.is_empty() { root.clone() } else { root.substitute(&assign_rep) })
}

// ============================================================================
// PIPELINE
// ============================================================================

/// Run the kernel splitting pipeline.
///
/// # Returns
/// Returns `(result, RangeifyBufferContext)`.
pub fn try_get_kernel_graph(root: Arc<UOp>) -> Result<(Arc<UOp>, RangeifyBufferContext), KernelGraphError> {
    // The cut promises origin-free kernel bodies. Nodes minted here — the AST SINK,
    // the END wrapper, the CALL — would otherwise adopt the scope a caller happens
    // to be realizing inside and fork every cache keyed on the body.
    let _detached = svod_ir::origin::OriginScope::suspend();

    let (pre_cut, ctx) = kernel_graph_pre_cut(root);

    let t_stage = std::time::Instant::now();
    let after_split = split_all_stores(&pre_cut);
    tracing::debug!(elapsed_ms = t_stage.elapsed().as_millis() as u64, "kernel split: split_all_stores complete");

    validate_normal_kernel_devices(&after_split).context(IrSnafu)?;

    let t_stage = std::time::Instant::now();
    let result = fix_assign(&after_split).context(IrSnafu)?;
    tracing::debug!(elapsed_ms = t_stage.elapsed().as_millis() as u64, "kernel split: fix_assign complete");

    if crate::spec::spec_enabled() {
        crate::spec::verify_kernel_graph(&result).context(SpecSnafu)?;
    }

    svod_ir::dump_canonical_stage("kernel_ast", &result);
    Ok((result, ctx))
}

/// Everything before the kernel cut: STAGE → STORE buffering and the one-shot
/// flatten pass. Split out because this is the last point where origins are still
/// on the nodes — `split_store` harvests them onto the CALL and strips the body.
pub fn kernel_graph_pre_cut(root: Arc<UOp>) -> (Arc<UOp>, RangeifyBufferContext) {
    use super::transforms::pm_add_buffers_patterns;
    use crate::rewrite::graph_rewrite_bottom_up;

    let lunique_start = root
        .toposort()
        .into_iter()
        .filter_map(|u| match u.op() {
            Op::LUnique(id) => Some(*id + 1),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let mut ctx = RangeifyBufferContext::with_lunique_start(lunique_start);

    // PASS 1: stage → store (pm_gate_kernel_sink + pm_add_buffers + pm_add_range_tags, bottom_up=True)
    let t_stage = std::time::Instant::now();
    static PM: std::sync::LazyLock<crate::TypedPatternMatcher<RangeifyBufferContext>> =
        std::sync::LazyLock::new(|| {
            use svod_ir::op::pattern_derived::OpKey;
            use svod_ir::pattern::RewriteResult;
            let mut matcher = pm_add_buffers_patterns().clone();
            // Skip the SINK subtree of an already-formed kernel AST.
            matcher.add(&[OpKey::Sink], |node, _ctx| {
                if matches!(node.op(), Op::Sink(ops::Sink { info: Some(_), .. })) {
                    RewriteResult::Gate(node.clone())
                } else {
                    RewriteResult::NoMatch
                }
            });
            matcher
        });
    let after_buffers = graph_rewrite_bottom_up(&*PM, root, &mut ctx);
    tracing::debug!(elapsed_ms = t_stage.elapsed().as_millis() as u64, "kernel split: pm_add_buffers complete");

    trace!(tree = %after_buffers.tree_full(), "after pm_add_buffers");

    // Pre-run pm_flatten_range on the FULL graph ONCE before kernel splitting.
    //
    // split_store includes pm_flatten_range but NOT pm_mops
    // (those were already applied in earlier pipeline stages). Running flatten_range once
    // on the full graph avoids redundant per-kernel traversals on overlapping subgraphs.
    let t_stage = std::time::Instant::now();
    let after_ctx_free = graph_rewrite_bottom_up(super::transforms::pm_flatten_range(), after_buffers, &mut ());
    tracing::debug!(
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "kernel split: pm_flatten_range pre-pass complete"
    );

    (after_ctx_free, ctx)
}

/// Split all STORE/END operations into CALL wrappers.
///
/// Matches upstream:
///   `graph_rewrite(tsink, pm_gate_kernel_sink + split_kernels, bottom_up=True)`
///
/// All patterns run in bpm (Stage 0, see ORIGINAL children):
/// - Gate on marked SINK nodes to prevent descending into already-split subtrees
/// - STORE/END → CALL via split_store
fn split_all_stores(root: &Arc<UOp>) -> Arc<UOp> {
    use svod_ir::op::pattern_derived::OpKey;
    use svod_ir::pattern::RewriteResult;
    use svod_ir::rewrite::graph_rewrite_bottom_up;

    // Combined gate + split in bpm (pm_gate_kernel_sink + split_kernels, bottom_up=True)
    let mut matcher = crate::patterns! {
        @context Vec<Arc<UOp>>;
        node @ Store { index: _, value: _ } => split_store(ctx, node),
        node @ End { computation, .. }
            if matches!(computation.op(), Op::Store(..) | Op::End(..))
            => split_store(ctx, node),
    };
    // Skip the subtree of an already-formed kernel AST or PROGRAM.
    matcher.add(&[OpKey::Sink, OpKey::Program], |node, _ctx| {
        if matches!(node.op(), Op::Sink(ops::Sink { info: Some(_), .. }) | Op::Program(..)) {
            RewriteResult::Gate(node.clone())
        } else {
            RewriteResult::NoMatch
        }
    });

    let mut ctx = Vec::new();
    graph_rewrite_bottom_up(&matcher, root.clone(), &mut ctx)
}

// ============================================================================
// SPLIT REDUCEOP
// ============================================================================

/// Extract all RANGE axis IDs from a UOp tree.
pub fn collect_range_ids(indexed: &Arc<UOp>) -> Vec<usize> {
    let mut range_ids: Vec<usize> = indexed
        .toposort()
        .into_iter()
        .filter_map(
            |node| if let Op::Range(ops::Range { axis_id, .. }) = node.op() { Some(axis_id.value()) } else { None },
        )
        .collect();

    range_ids.sort_unstable();
    range_ids.dedup();
    range_ids
}

#[derive(Debug, Clone)]
struct SplitCandidate {
    dimension: usize,
    divisor: usize,
    #[allow(dead_code)]
    output_size: usize,
}

fn detect_expanded_dimensions(source: &Arc<UOp>, input_shape: &[SInt]) -> Vec<bool> {
    let ranges: Vec<Arc<UOp>> = input_shape
        .iter()
        .enumerate()
        .map(|(axis_id, dim)| match dim {
            SInt::Const(n) if *n > 1 => {
                let end = UOp::index_const(*n as i64);
                UOp::range_axis(end, svod_ir::AxisId::Unrenumbered(axis_id), svod_ir::AxisType::Weak)
            }
            _ => UOp::index_const(0),
        })
        .collect();

    let indexed = match UOp::index().buffer(Arc::clone(source)).indices(ranges).call() {
        Ok(idx) => idx,
        Err(_) => return vec![false; input_shape.len()],
    };

    let base = source.base();
    let noop = UOp::noop();
    let mut substitutions = HashMap::new();
    substitutions.insert(UOpKey(base), noop);

    let substituted = indexed.substitute(&substitutions);

    use super::patterns::movement_op_patterns;
    use crate::rewrite::graph_rewrite_bottom_up;

    // pm_mops (movement ops, bottom_up=True)
    use std::sync::LazyLock;
    static PM_MOPS: LazyLock<crate::TypedPatternMatcher> = LazyLock::new(movement_op_patterns);
    let transformed = graph_rewrite_bottom_up(&*PM_MOPS, substituted, &mut ());

    let surviving_range_ids = collect_range_ids(&transformed);
    let surviving_set: HashSet<usize> = surviving_range_ids.into_iter().collect();

    input_shape.iter().enumerate().map(|(axis_id, _)| !surviving_set.contains(&axis_id)).collect()
}

fn find_split_candidates(
    reduce: &Arc<UOp>,
    input_shape: &[SInt],
    is_expanded: &[bool],
    config: &SplitReduceOpConfig,
) -> Vec<SplitCandidate> {
    let Op::Reduce(ops::Reduce { num_axes, .. }) = reduce.op() else {
        return vec![];
    };

    let output_shape = match reduce.shape() {
        Ok(Some(shape)) => shape,
        _ => return vec![],
    };

    let output_size: usize = output_shape.iter().filter_map(|s| s.as_const()).product();

    let mut candidates = Vec::new();

    for axis in 0..*num_axes {
        if axis >= is_expanded.len() || is_expanded[axis] {
            continue;
        }

        let dim_size = match &input_shape[axis] {
            SInt::Const(n) => *n,
            SInt::Symbolic(_) | SInt::Infer => continue,
        };

        for divisor in (config.min_divisor..=config.max_divisor).rev() {
            if dim_size % divisor != 0 {
                continue;
            }

            let new_output_size = output_size * divisor;

            if new_output_size > config.max_output_size() {
                continue;
            }

            candidates.push(SplitCandidate { dimension: axis, divisor, output_size: new_output_size });
        }
    }

    candidates
}

fn apply_split_transformation(
    source: &Arc<UOp>,
    reduce: &Arc<UOp>,
    candidate: &SplitCandidate,
    input_shape: &[SInt],
) -> Option<Arc<UOp>> {
    let Op::Reduce(ops::Reduce { reduce_op, num_axes, .. }) = reduce.op() else {
        return None;
    };

    let dim_to_split = candidate.dimension;
    let divisor = candidate.divisor;
    let dim_size = input_shape[dim_to_split].as_const()?;
    let remainder = dim_size / divisor;

    let mut splitted_shape: SmallVec<[SInt; 4]> = SmallVec::new();
    for (i, dim) in input_shape.iter().enumerate() {
        if i == dim_to_split {
            splitted_shape.push(SInt::Const(divisor));
            splitted_shape.push(SInt::Const(remainder));
        } else {
            splitted_shape.push(dim.clone());
        }
    }

    let reshaped = source.try_reshape(&splitted_shape).ok()?;

    let mut permutation: Vec<usize> = (0..splitted_shape.len()).filter(|&i| i != dim_to_split).collect();
    permutation.push(dim_to_split);

    let permuted = reshaped.try_permute(permutation.clone()).ok()?;

    let first_reduce = permuted.try_reduce_axis(*reduce_op, (0..*num_axes).collect()).ok()?;

    let contiguous = first_reduce.contiguous();

    let output_shape = contiguous.shape().ok()??;
    let split_axis = output_shape.len() - 1;

    let second_reduce = contiguous.try_reduce_axis(*reduce_op, vec![split_axis]).ok()?;

    let final_shape = reduce.shape().ok()??;

    second_reduce.try_reshape(final_shape).ok()
}

/// Split a large tensor-form REDUCE into two stages.
pub fn split_reduceop(reduce: &Arc<UOp>, config: &SplitReduceOpConfig) -> Option<Arc<UOp>> {
    if !config.enabled {
        return None;
    }

    let Op::Reduce(ops::Reduce { src: source, ranges, num_axes, .. }) = reduce.op() else {
        return None;
    };
    if *num_axes == 0 || !ranges.is_empty() {
        return None;
    }

    let input_shape = source.shape().ok()??;
    let output_shape = reduce.shape().ok()??;

    if !input_shape.iter().all(|s| s.is_const()) {
        return None;
    }

    let input_size: usize = input_shape.iter().map(|s| s.as_const().unwrap()).product();
    let output_size: usize = output_shape.iter().map(|s| s.as_const().unwrap()).product();

    if output_size == 0 {
        return None;
    }

    let ratio = input_size / output_size;
    if ratio < config.split_threshold {
        return None;
    }

    let is_expanded = detect_expanded_dimensions(source, input_shape);
    let candidates = find_split_candidates(reduce, input_shape, &is_expanded, config);

    if candidates.is_empty() {
        return None;
    }

    apply_split_transformation(source, reduce, &candidates[0], input_shape)
}

#[cfg(test)]
#[path = "../test/unit/rangeify/kernel_internal.rs"]
mod tests;
