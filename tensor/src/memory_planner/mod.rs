//! Memory planner for buffer reuse optimization.
//!
//! Liveness is measured in EXECUTION LEVELS (the runtime's parallel-execution
//! waves), not schedule order. Because the executor runs each level fully —
//! with a hard barrier and synchronous CPU writes — before the next, a buffer
//! last used in level `L` may share storage with one first used in a level
//! `> L` with NO injected dependency. Reuse is strictly cross-level: same-level
//! buffers may run concurrently, and the runtime's conflict check (by buffer
//! handle id) is blind to arena-view aliasing, so same-level sharing is unsafe.
//!
//! The planner injects ZERO ordering edges — reuse safety comes entirely from
//! the level barrier. The previous design injected edges and was the source of
//! a buffer-clobber drift.
//!
//! # Algorithm
//!
//! 1. **Levels**: [`compute_item_levels`] assigns each schedule item its
//!    execution level (matching the runtime's per-op leveling).
//! 2. **Liveness**: track `[first_level, last_level]` of each buffer.
//! 3. **Packing**: [`PlannerMode::Arena`] packs buffers into a per-device TLSF
//!    arena keyed by level; [`PlannerMode::Remap`] pools whole buffers by
//!    `(device, dtype, size)`. Both only ever overlap level-disjoint buffers.
//! 4. **Apply**: map logical buffers to physical (arena views / pooled buffers).
//!
//! `SVOD_MEMORY_PLANNER=off` disables planning entirely — the escape hatch if
//! a workload with very wide levels regresses on peak memory (level-interval
//! reuse is less aggressive than dependency-forced reuse).

mod tlsf;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use snafu::ResultExt;
use svod_device::Buffer;
use svod_dtype::{DType, DeviceSpec};
use svod_ir::{Op, UOp};
use tracing::{debug, trace};

use crate::schedule::Schedule;
use svod_ir::ops;

/// Minimum block size for buffer pooling (256-byte alignment, matching tinygrad).
const MIN_BLOCK_SIZE: usize = 256;

/// Selects the buffer-allocation strategy used by the planner entrypoint.
///
/// - `Disabled` short-circuits the planner and emits no replacements.
/// - `Remap` runs liveness-based pool reuse: groups buffers by
///   `(device, dtype, rounded_size)` and lets disjoint-lifetime buffers
///   share an underlying allocation.
/// - `Arena` packs all plannable buffers into one per-device arena using a TLSF
///   allocator and rewrites each logical buffer as a `Buffer::view` into it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PlannerMode {
    /// Skip the planner entirely. Each `Buffer` keeps its original allocation
    /// and is freed by lazy `Drop`. Useful for memory-debugging baselines.
    Disabled,
    /// Liveness-based pool reuse: groups buffers by
    /// `(device, dtype, rounded_size)` and lets level-disjoint buffers
    /// share an underlying allocation via `Arc<Buffer>` swap.
    Remap,
    /// Arena packing: pack every plannable buffer into a per-device arena using
    /// a TLSF allocator and rewrite each logical buffer as a fresh
    /// `Buffer::view` into it.
    #[default]
    Arena,
}

/// Pure parser for the `SVOD_MEMORY_PLANNER` env var, exposed for testing.
///
/// Default (env unset) is [`PlannerMode::Arena`], matching tinygrad's
/// `NO_MEMORY_PLANNER=0` default — the arena planner runs unless the user
/// explicitly opts out. `remap` / `pool` keep the older liveness-based pool
/// reuse for parity with the previous default if a workload regresses.
pub fn parse_mode(raw: Option<&str>) -> PlannerMode {
    let Some(raw) = raw else {
        return PlannerMode::Arena;
    };
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "0" | "off" | "none" | "disabled" => PlannerMode::Disabled,
        "remap" | "pool" => PlannerMode::Remap,
        // "1" | "on" | "arena" | "" or any unrecognized → Arena (tinygrad default)
        _ => PlannerMode::Arena,
    }
}

/// Read `SVOD_MEMORY_PLANNER` from the environment and resolve to a [`PlannerMode`].
pub fn mode_from_env() -> PlannerMode {
    parse_mode(std::env::var("SVOD_MEMORY_PLANNER").ok().as_deref())
}

type LogicalBufferAlias = (usize, usize, DType, Vec<usize>);

/// Round up to the nearest multiple of block_size.
#[inline]
fn round_up(size: usize, block_size: usize) -> usize {
    size.div_ceil(block_size) * block_size
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Key for buffer pooling - groups buffers that can be reused interchangeably.
///
/// Buffer reuse is shape-agnostic: codegen reads logical shape from the UOp
/// graph, runtime dispatch passes raw `*mut u8` pointers, and the planner
/// skips output buffers (the only consumers of `Buffer::shape()` via
/// `as_array`/`as_array_mut`). Two non-output buffers with the same
/// `(device, dtype, rounded_size)` are interchangeable storage regardless
/// of their logical shapes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferPoolKey {
    /// Device where buffer is allocated.
    pub device: DeviceSpec,
    /// Data type of buffer elements.
    pub dtype: DType,
    /// Buffer size in bytes (rounded up to MIN_BLOCK_SIZE).
    pub size: usize,
}

/// Liveness information for a buffer, measured in EXECUTION LEVELS.
///
/// The runtime executes `op_levels` with a hard barrier between levels (level
/// L fully completes — CPU writes visible — before L+1 starts). So a buffer
/// last used in level `last_level` may safely share storage with one first
/// used in a level `> last_level`, with NO injected dependency. Equality of
/// levels is unsafe (same-level ops run concurrently and the runtime's
/// handle-id conflict check can't see arena-view aliasing).
#[derive(Debug, Clone)]
pub struct BufferLiveness {
    /// Lowest execution level that uses this buffer.
    pub first_level: usize,
    /// Highest execution level that uses this buffer.
    pub last_level: usize,
    /// Pool key for buffer grouping.
    pub pool_key: BufferPoolKey,
    /// Representative logical buffer for this allocation ID.
    pub prototype: Buffer,
}

/// Buffer allocation/deallocation event for timeline scheduling.
#[derive(Debug, Clone)]
struct BufferEvent {
    /// Execution level when this event occurs.
    timestep: usize,
    /// True for allocation, false for deallocation.
    is_alloc: bool,
    /// Physical buffer allocation identifier.
    buffer_id: u64,
}

/// Collected planner inputs derived from schedule traversal.
struct PlannerInput {
    /// Liveness keyed by physical buffer allocation ID.
    liveness: HashMap<u64, BufferLiveness>,
    /// Logical schedule slots that are eligible for replacement.
    occurrences: Vec<(BufferKey, u64)>,
    metrics: MemoryPlannerMetrics,
}

/// Key to identify a buffer within a schedule.
///
/// We use (kernel_index, buffer_index) because the same UOp ID might appear
/// in multiple kernels due to buffer sharing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferKey {
    /// Index of kernel in the schedule.
    pub kernel_idx: usize,
    /// Index of buffer within that kernel's buffer list.
    pub buffer_idx: usize,
}

/// Result of memory planning.
#[derive(Debug)]
pub struct MemoryPlannerResult {
    /// Mapping from (kernel_idx, buffer_idx) to replacement buffer.
    /// Only contains entries for buffers that were replaced.
    pub buffer_replace: HashMap<BufferKey, Buffer>,
    /// Total memory saved through buffer reuse (in bytes).
    pub memory_saved: usize,
    /// Number of buffers that were reused.
    pub buffers_reused: usize,
    /// Detailed demand, exclusion, and physical-allocation measurements.
    pub metrics: MemoryPlannerMetrics,
}

/// Why a logical allocation was conservatively excluded from reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlannerExclusionReason {
    Disk,
    ViewOrOffset,
    AlreadyAllocated,
    Output,
    AliasedStorage,
    NonSinkOperation,
    GatedStore,
}

/// Unique allocation count and logical bytes excluded for one reason.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlannerExclusionStats {
    pub allocations: usize,
    pub bytes: usize,
}

/// Measurements for one planner invocation. Byte counts are exact logical
/// buffer sizes unless explicitly named `rounded` or `committed`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryPlannerMetrics {
    /// Policy used for this measurement.
    pub mode: PlannerMode,
    /// Unique allocations eligible for planning after exclusions.
    pub logical_allocations: usize,
    /// Exact bytes requested by eligible logical allocations over the plan.
    pub logical_bytes: usize,
    /// Eligible bytes after applying the planner's 256-byte request rounding.
    pub rounded_bytes: usize,
    /// Maximum exact eligible bytes live in any execution level.
    pub logical_peak_bytes: usize,
    /// Bytes committed to per-device arenas; zero outside arena mode.
    pub arena_committed_bytes: usize,
    /// Physical bytes representing eligible allocations after planning.
    pub physical_bytes: usize,
    /// Cumulative request rounding overhead across logical allocations.
    pub padding_bytes: usize,
    /// Arena commitment above peak exact live demand, including alignment
    /// holes and independently resident per-device arenas.
    pub fragmentation_bytes: usize,
    /// Logical allocations consolidated into an already-counted physical
    /// allocation (a pooled buffer or per-device arena).
    pub reused_allocations: usize,
    /// Exact logical bytes avoided relative to separate eligible allocations.
    pub reused_bytes: usize,
    /// Unique excluded allocation counts and exact bytes by first reason.
    pub exclusions: BTreeMap<PlannerExclusionReason, PlannerExclusionStats>,
}

// ============================================================================
// LIVENESS ANALYSIS
// ============================================================================

/// Analyze buffer liveness across the schedule.
///
/// Tracks first and last appearance of each buffer, skipping:
/// - Already allocated buffers (inputs)
/// - Output buffers
/// - Transfer operations
///
/// Buffer ids excluded from reuse, grouped by the reason that first applies.
#[derive(Default)]
struct ExcludedBufferIds {
    aliased: HashSet<u64>,
    non_sink: HashSet<u64>,
    /// Written by a gated STORE: only part of the buffer is written, so arena
    /// mode must not pack a later tenant over the untouched bytes. Tinygrad's
    /// TLSF planner (`engine/memory.py:17,70`) has no equivalent because it
    /// never packs a new tenant over a live one.
    gated_store: HashSet<u64>,
}

fn collect_excluded_buffer_ids(schedule: &Schedule) -> ExcludedBufferIds {
    // Alias detection groups views/buffers that share the same underlying
    // storage. Keying by `Buffer::id()` would miss views (since each view
    // mints a fresh handle id); keying by `storage_id()` correctly groups
    // every view of one allocation under one bucket.
    let mut by_storage: HashMap<u64, HashSet<LogicalBufferAlias>> = HashMap::new();
    let mut gated_store = HashSet::new();
    for item in schedule {
        for buffer in &item.buffers {
            by_storage.entry(buffer.storage_id().0).or_default().insert((
                buffer.offset(),
                buffer.size(),
                buffer.dtype(),
                buffer.shape().to_vec(),
            ));
        }

        let by_uop_id: HashMap<u64, u64> =
            item.buffer_uop_ids.iter().copied().zip(item.buffers.iter().map(|b| b.id().0)).collect();
        for node in item.ast.toposort() {
            let Op::Store(ops::Store { index, gate: Some(_), .. }) = node.op() else { continue };
            gated_store.extend(indexed_buffer(index).and_then(|uop| by_uop_id.get(&uop.buf_uop().id)).copied());
        }
    }

    // Map aliased storage ids back to handle ids — every Buffer in the
    // schedule whose storage has multiple distinct views is non-plannable.
    let aliased_storages: HashSet<u64> =
        by_storage.into_iter().filter_map(|(sid, views)| (views.len() > 1).then_some(sid)).collect();
    let aliased_ids = schedule.iter().flat_map(|item| {
        item.buffers
            .iter()
            .filter(|b| aliased_storages.contains(&b.storage_id().0))
            .map(|b| b.id().0)
            .collect::<Vec<_>>()
    });

    let non_sink = schedule
        .iter()
        .filter(|item| !matches!(item.ast.op(), Op::Sink(..)))
        .flat_map(|item| item.buffers.iter().map(|b| b.id().0))
        .collect();
    ExcludedBufferIds { aliased: aliased_ids.collect(), non_sink, gated_store }
}

/// The buffer an INDEX addresses, looking through the casts and arithmetic a
/// lowered store index can carry.
fn indexed_buffer(index: &Arc<UOp>) -> Option<&Arc<UOp>> {
    match index.op() {
        Op::Index(ops::Index { buffer, .. }) => Some(buffer),
        other => other.children().into_iter().find_map(indexed_buffer),
    }
}

fn exclusion_reason(
    buffer: &Buffer,
    output_buffer_ids: &HashSet<u64>,
    excluded: &ExcludedBufferIds,
) -> Option<PlannerExclusionReason> {
    let id = buffer.id().0;
    if buffer.allocator().device_spec().is_disk() {
        Some(PlannerExclusionReason::Disk)
    } else if buffer.offset() != 0 {
        Some(PlannerExclusionReason::ViewOrOffset)
    } else if buffer.is_allocated() {
        Some(PlannerExclusionReason::AlreadyAllocated)
    } else if output_buffer_ids.contains(&id) {
        Some(PlannerExclusionReason::Output)
    } else if excluded.aliased.contains(&id) {
        Some(PlannerExclusionReason::AliasedStorage)
    } else if excluded.non_sink.contains(&id) {
        Some(PlannerExclusionReason::NonSinkOperation)
    } else if excluded.gated_store.contains(&id) {
        Some(PlannerExclusionReason::GatedStore)
    } else {
        None
    }
}

/// Compute a per-schedule-item execution level via the SHARED leveling routine
/// ([`svod_runtime::compute_topological_levels`]) — the same function the
/// runtime executor uses for `op_levels`. Keyed on `kernel.id` +
/// `ScheduleItem.dependencies`; `levels[i]` is item `i`'s longest
/// dependency-path length. Returns `Err` (loud) on a cyclic or unresolved-dep
/// schedule, matching the runtime — the planner no longer silently skips.
///
/// Sharing one implementation guarantees these levels equal the runtime's
/// per-op levels (so level-interval reuse decisions match real execution
/// order), provided op emission stays 1:1 with schedule items — enforced by the
/// the `op_count` assert in `prepare_execution_plan`. `ScheduleItem::instance_dependencies`
/// is fed in for the same reason it is fed to the executor.
pub fn compute_item_levels(schedule: &Schedule) -> crate::Result<Vec<usize>> {
    let node_ids: Vec<u64> = schedule.iter().map(|it| it.kernel.id).collect();
    let callable_deps: Vec<Vec<u64>> = schedule.iter().map(|it| it.dependencies.clone()).collect();

    // Positional edges must be honoured exactly as the runtime honours them:
    // a schedule loop replays one kernel id many times, so `callable_deps`
    // alone collapses every iteration into one wave and the planner would
    // retire a buffer the last iteration still reads.
    let index_deps: Vec<Vec<usize>> = schedule.iter().map(|it| it.instance_dependencies.clone()).collect();
    let waves = svod_runtime::compute_topological_levels(
        &node_ids,
        &callable_deps,
        (!index_deps.is_empty()).then_some(index_deps.as_slice()),
    )
    .context(crate::error::ExecutionSnafu)?;
    let mut level_of = vec![0usize; schedule.len()];
    for (level_idx, wave) in waves.iter().enumerate() {
        for &node_idx in wave {
            level_of[node_idx] = level_idx;
        }
    }
    Ok(level_of)
}

/// Derive each buffer's live level-interval from `item.buffers` alone.
///
/// # Completeness invariant
///
/// `item.buffers` is closed over every buffer the kernel reads or writes — so
/// iterating it here cannot under-count a buffer's lifetime (which, with zero
/// injected ordering edges, would be the only way reuse could clobber live
/// data). It holds because a kernel's buffer set is built from its CALL args,
/// and those args come from a FULL bottom-up traversal of the kernel AST:
/// `split_store` (`schedule/src/rangeify/kernel.rs`) runs `graph_rewrite_bottom_up`
/// with `local_to_param_patterns`/`map_after_like_node`, recording every
/// Buffer/Param/After/MStack/MSelect node into the CALL, which
/// `collect_callable_buffers` (`tensor/src/schedule.rs`) materializes into
/// `item.buffers`.
///
/// Documentation only — no verifying assert: at planning time the AST is
/// already PARAM-rewritten (Buffer→codegen `PARAM`, no handle id), so a check
/// re-deriving "buffers the AST touches" would duplicate `collect_callable_buffers`
/// and risk false positives. The invariant is enforced upstream at CALL build.
fn analyze_liveness(
    schedule: &Schedule,
    item_levels: &[usize],
    output_buffer_ids: &HashSet<u64>,
    mode: PlannerMode,
) -> PlannerInput {
    let excluded = collect_excluded_buffer_ids(schedule);
    let mut liveness: HashMap<u64, BufferLiveness> = HashMap::new();
    let mut occurrences: Vec<(BufferKey, u64)> = Vec::new();
    let mut metrics = MemoryPlannerMetrics { mode, ..Default::default() };
    let mut classified = HashSet::new();

    for (step_idx, item) in schedule.iter().enumerate() {
        let level = item_levels[step_idx];
        for (buf_idx, buffer) in item.buffers.iter().enumerate() {
            let key = BufferKey { kernel_idx: step_idx, buffer_idx: buf_idx };
            let buf_id = buffer.id().0;

            if let Some(reason) = exclusion_reason(buffer, output_buffer_ids, &excluded) {
                if classified.insert(buf_id) {
                    let excluded = metrics.exclusions.entry(reason).or_default();
                    excluded.allocations += 1;
                    excluded.bytes += buffer.size();
                }
                trace!(step_idx, buf_idx, buffer_id = buf_id, "skipping buffer in memory planner");
                continue;
            }

            occurrences.push((key, buf_id));

            let pool_key = BufferPoolKey {
                device: buffer.allocator().device_spec(),
                dtype: buffer.dtype(),
                size: round_up(buffer.size(), MIN_BLOCK_SIZE),
            };

            liveness
                .entry(buf_id)
                .and_modify(|info| {
                    info.first_level = info.first_level.min(level);
                    info.last_level = info.last_level.max(level);
                })
                .or_insert_with(|| BufferLiveness {
                    first_level: level,
                    last_level: level,
                    pool_key,
                    prototype: buffer.clone(),
                });
        }
    }

    metrics.logical_allocations = liveness.len();
    metrics.logical_bytes = liveness.values().map(|info| info.prototype.size()).sum();
    metrics.rounded_bytes = liveness.values().map(|info| info.pool_key.size).sum();
    metrics.padding_bytes = metrics.rounded_bytes.saturating_sub(metrics.logical_bytes);
    let max_level = liveness.values().map(|info| info.last_level).max();
    metrics.logical_peak_bytes = max_level.map_or(0, |max_level| {
        (0..=max_level)
            .map(|level| {
                liveness
                    .values()
                    .filter(|info| info.first_level <= level && level <= info.last_level)
                    .map(|info| info.prototype.size())
                    .sum()
            })
            .max()
            .unwrap_or(0)
    });

    debug!(num_optimizable = liveness.len(), "liveness analysis complete");

    PlannerInput { liveness, occurrences, metrics }
}

// ============================================================================
// EVENT TIMELINE
// ============================================================================

/// Build sorted event timeline from liveness information.
///
/// Events are sorted by (timestep, is_alloc) so that:
/// - Earlier timesteps come first
/// - At the same timestep, frees (is_alloc=false) come before allocs (is_alloc=true)
///
/// This ordering allows immediate reuse of freed buffers.
fn build_event_timeline(liveness: &HashMap<u64, BufferLiveness>) -> Vec<BufferEvent> {
    let mut events = Vec::with_capacity(liveness.len() * 2);

    for (&buf_id, info) in liveness {
        // Allocate at the buffer's first level; free one level past its last.
        events.push(BufferEvent { timestep: info.first_level, is_alloc: true, buffer_id: buf_id });
        events.push(BufferEvent { timestep: info.last_level + 1, is_alloc: false, buffer_id: buf_id });
    }

    // Sort by (timestep, is_alloc) — false < true ensures frees precede allocs
    // at the same level. A free at `last_level + 1` can only be matched by an
    // alloc at `first_level >= last_level + 1`, i.e. `last_level < first_level`
    // (strictly level-disjoint), so pool reuse never aliases within a level.
    events.sort_by_key(|e| (e.timestep, e.is_alloc, e.buffer_id));

    events
}

// ============================================================================
// POOL-BASED ALLOCATION
// ============================================================================

/// Process events and compute buffer replacements using pool-based allocation.
///
/// For each allocation event:
/// - Try to reuse a buffer from the pool with matching key
/// - If no match, the buffer keeps its original allocation
///
/// For each deallocation event:
/// - Return the buffer to the pool for future reuse
fn process_events(
    events: &[BufferEvent],
    liveness: &HashMap<u64, BufferLiveness>,
    occurrences: &[(BufferKey, u64)],
) -> (HashMap<BufferKey, Buffer>, usize, usize, usize) {
    let mut free_pools: HashMap<BufferPoolKey, Vec<Buffer>> = HashMap::new();
    let mut memory_saved: usize = 0;
    let mut buffers_reused: usize = 0;
    let mut chosen_by_id: HashMap<u64, Buffer> = HashMap::new();

    // Track live assignment during timeline simulation.
    let mut active_buffers: HashMap<u64, Buffer> = HashMap::new();

    for event in events {
        let info = match liveness.get(&event.buffer_id) {
            Some(info) => info,
            None => continue,
        };
        let pool_key = &info.pool_key;

        if event.is_alloc {
            // Reuse a pooled buffer if available. The event timeline guarantees
            // any pooled buffer was freed at a strictly-earlier level, so reuse
            // is safe with no injected dependency (per-level barrier).
            if let Some(pool) = free_pools.get_mut(pool_key)
                && let Some(reused) = pool.pop()
            {
                trace!(level = event.timestep, reused_buffer_id = reused.id().0, "reusing buffer from pool");
                chosen_by_id.insert(event.buffer_id, reused.clone());
                active_buffers.insert(event.buffer_id, reused);
                memory_saved += pool_key.size;
                buffers_reused += 1;
                continue;
            }

            // No reuse - use original buffer
            chosen_by_id.insert(event.buffer_id, info.prototype.clone());
            active_buffers.insert(event.buffer_id, info.prototype.clone());
        } else {
            // Deallocation - return buffer to pool
            if let Some(buffer) = active_buffers.remove(&event.buffer_id) {
                free_pools.entry(pool_key.clone()).or_default().push(buffer);
            }
        }
    }

    let mut buffer_replace: HashMap<BufferKey, Buffer> = HashMap::new();
    for (key, buf_id) in occurrences {
        if let Some(chosen) = chosen_by_id.get(buf_id)
            && chosen.id().0 != *buf_id
        {
            buffer_replace.insert(*key, chosen.clone());
        }
    }

    let mut physical_ids = HashSet::new();
    let physical_bytes =
        chosen_by_id.values().filter(|buffer| physical_ids.insert(buffer.id().0)).map(Buffer::size).sum();
    (buffer_replace, memory_saved, buffers_reused, physical_bytes)
}

// ============================================================================
// ARENA-BASED ALLOCATION
// ============================================================================

/// Arena identifier: buffers are packed into one arena per device and class
/// of memory, so a device-local intermediate stays device-local; reuse safety
/// comes entirely from strict level-disjointness (see [`BufferLiveness`]), so
/// no copy/compute lane split or injected dependency is needed.
type LaneKey = (DeviceSpec, svod_device::BufferSpec);

/// Tinygrad-style arena planner: replaces every plannable buffer with a
/// `Buffer::view` into a per-device arena allocated by [`tlsf::TlsfAllocator`].
///
/// Liveness is measured in execution LEVELS: a buffer occupies its arena offset
/// from its `first_level` until `last_level + 1`. The TLSF timeline (frees
/// before allocs at equal level) therefore only ever overlaps level-disjoint
/// buffers, and the per-level barrier makes that reuse safe with no injected
/// dependency. Each logical buffer is swapped for its arena view via
/// [`apply_buffer_replacements`].
fn memory_plan_arena(
    schedule: &Schedule,
    item_levels: &[usize],
    output_buffer_ids: &HashSet<u64>,
) -> MemoryPlannerResult {
    let planner_input = analyze_liveness(schedule, item_levels, output_buffer_ids, PlannerMode::Arena);
    let empty_result = |mut metrics: MemoryPlannerMetrics| {
        metrics.physical_bytes = metrics.logical_bytes;
        MemoryPlannerResult { buffer_replace: HashMap::new(), memory_saved: 0, buffers_reused: 0, metrics }
    };
    let liveness = planner_input.liveness;
    if liveness.is_empty() {
        return empty_result(planner_input.metrics);
    }

    let lane_key = |id: u64| -> LaneKey {
        let prototype = &liveness[&id].prototype;
        (prototype.allocator().device_spec(), prototype.spec())
    };

    // Per-buffer rounded size: round to `block_size` so the TLSF bucket math stays correct.
    let nbytes_rounded: HashMap<u64, usize> =
        liveness.iter().map(|(&id, info)| (id, round_up(info.prototype.size(), MIN_BLOCK_SIZE))).collect();

    let events = build_event_timeline(&liveness);

    // Per-device TLSF allocators. Generous size budget = 2 × Σ(rounded sizes)
    // so even worst-case fragmentation can fit.
    let total_bytes: usize = nbytes_rounded.values().sum();
    let arena_budget = total_bytes.saturating_mul(2).max(MIN_BLOCK_SIZE);
    let mut tlsfs: HashMap<LaneKey, tlsf::TlsfAllocator> = HashMap::new();
    let mut offsets: HashMap<u64, usize> = HashMap::new();
    let mut peaks: HashMap<LaneKey, usize> = HashMap::new();

    for event in &events {
        let lane = lane_key(event.buffer_id);
        let info = &liveness[&event.buffer_id];
        let alloc =
            tlsfs.entry(lane.clone()).or_insert_with(|| tlsf::TlsfAllocator::new(arena_budget, 0, MIN_BLOCK_SIZE, 32));
        if event.is_alloc {
            let req = nbytes_rounded[&event.buffer_id];
            let off = match alloc.alloc(req, 1) {
                Ok(o) => o,
                Err(e) => {
                    tracing::warn!(?e, "arena planner: TLSF alloc failed; skipping arena rewrite");
                    return empty_result(planner_input.metrics);
                }
            };
            offsets.insert(event.buffer_id, off);
            // Peak reflects actual byte usage (`buf.arg * itemsize`), not bucket-rounded size.
            let used_end = off + info.prototype.size();
            let peak = peaks.entry(lane.clone()).or_insert(0);
            if used_end > *peak {
                *peak = used_end;
            }
        } else if let Some(off) = offsets.get(&event.buffer_id).copied()
            && let Err(e) = alloc.free(off)
        {
            tracing::warn!(?e, "arena planner: TLSF free failed; skipping arena rewrite");
            return empty_result(planner_input.metrics);
        }
    }

    // Allocate one arena buffer per device, sized to its peak. Precompute a
    // lane→prototype map so we don't re-scan `liveness` once per lane.
    let mut lane_proto: HashMap<LaneKey, Buffer> = HashMap::with_capacity(peaks.len());
    for (&id, info) in &liveness {
        lane_proto.entry(lane_key(id)).or_insert_with(|| info.prototype.clone());
    }
    let mut arenas: HashMap<LaneKey, Buffer> = HashMap::new();
    for (lane, &peak) in &peaks {
        if peak == 0 {
            continue;
        }
        let arena_size = round_up(peak, MIN_BLOCK_SIZE);
        let prototype = lane_proto.get(lane).expect("every populated lane must have a prototype");
        let arena = Buffer::new(prototype.allocator_arc(), svod_dtype::DType::UInt8, vec![arena_size], lane.1);
        arenas.insert(lane.clone(), arena);
    }

    // Build buffer_replace by viewing each plannable buffer's slice of its
    // arena. `Buffer::view` mints a fresh handle id per view, so disjoint views
    // appear as independent buffers to the runtime hazard model. A view failure
    // aborts the WHOLE rewrite (all-or-nothing) — a partial rewrite would leave
    // one buffer un-aliased while others assume the full plan.
    let mut buffer_replace: HashMap<BufferKey, Buffer> = HashMap::new();
    let mut buffers_reused = 0usize;
    for (key, buf_id) in &planner_input.occurrences {
        let Some(&offset) = offsets.get(buf_id) else {
            continue;
        };
        let Some(arena) = arenas.get(&lane_key(*buf_id)) else {
            continue;
        };
        let byte_size = liveness[buf_id].prototype.size();
        let view = match arena.view(offset, byte_size) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(?e, "arena planner: view failed; aborting arena rewrite");
                return empty_result(planner_input.metrics);
            }
        };
        buffer_replace.insert(*key, view);
        buffers_reused += 1;
    }

    let arena_total: usize = peaks.values().map(|&p| round_up(p, MIN_BLOCK_SIZE)).sum();
    let memory_saved = total_bytes.saturating_sub(arena_total);
    let mut metrics = planner_input.metrics;
    metrics.arena_committed_bytes = arena_total;
    metrics.physical_bytes = arena_total;
    metrics.fragmentation_bytes = arena_total.saturating_sub(metrics.logical_peak_bytes);
    metrics.reused_allocations = metrics.logical_allocations.saturating_sub(arenas.len());
    metrics.reused_bytes = metrics.logical_bytes.saturating_sub(arena_total);

    debug!(
        buffers_planned = liveness.len(),
        buffers_replaced = buffers_reused,
        memory_saved_bytes = memory_saved,
        arena_count = arenas.len(),
        "arena memory planner complete"
    );

    MemoryPlannerResult { buffer_replace, memory_saved, buffers_reused, metrics }
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

/// Run memory planner on a schedule.
///
/// Analyzes buffer lifetimes and identifies opportunities for buffer reuse.
/// Returns a mapping from logical buffers to physical buffers.
///
/// # Arguments
///
/// * `schedule` - The execution schedule to optimize
/// * `output_buffer_ids` - IDs of output buffers that must not be reused
/// * `mode` - Selects the planner strategy. [`PlannerMode::Disabled`] returns
///   an empty result without analyzing the schedule. [`PlannerMode::Remap`]
///   runs liveness-based pool reuse. [`PlannerMode::Arena`] runs the
///   tinygrad-style arena packing pass via [`memory_plan_arena`].
///
/// # Returns
///
/// `MemoryPlannerResult` containing buffer replacements and statistics.
/// * `item_levels` - Per-schedule-item execution level (see
///   [`compute_item_levels`]); liveness and reuse are measured in these levels.
#[allow(rustdoc::private_intra_doc_links)]
pub fn memory_planner(
    schedule: &Schedule,
    item_levels: &[usize],
    output_buffer_ids: &HashSet<u64>,
    mode: PlannerMode,
) -> MemoryPlannerResult {
    if schedule.is_empty() {
        return MemoryPlannerResult {
            buffer_replace: HashMap::new(),
            memory_saved: 0,
            buffers_reused: 0,
            metrics: MemoryPlannerMetrics { mode, ..Default::default() },
        };
    }

    if matches!(mode, PlannerMode::Arena) {
        return memory_plan_arena(schedule, item_levels, output_buffer_ids);
    }

    // Phase 1: Liveness analysis
    let planner_input = analyze_liveness(schedule, item_levels, output_buffer_ids, mode);
    if matches!(mode, PlannerMode::Disabled) {
        let mut metrics = planner_input.metrics;
        metrics.physical_bytes = metrics.logical_bytes;
        return MemoryPlannerResult { buffer_replace: HashMap::new(), memory_saved: 0, buffers_reused: 0, metrics };
    }
    let liveness = planner_input.liveness;

    if liveness.is_empty() {
        debug!("no optimizable buffers found");
        return MemoryPlannerResult {
            buffer_replace: HashMap::new(),
            memory_saved: 0,
            buffers_reused: 0,
            metrics: planner_input.metrics,
        };
    }

    // Phase 2: Build event timeline
    let events = build_event_timeline(&liveness);

    // Phase 3: Process events and compute replacements
    let (buffer_replace, memory_saved, buffers_reused, physical_bytes) =
        process_events(&events, &liveness, &planner_input.occurrences);
    let mut metrics = planner_input.metrics;
    metrics.physical_bytes = physical_bytes;
    metrics.reused_allocations = buffers_reused;
    metrics.reused_bytes = metrics.logical_bytes.saturating_sub(metrics.physical_bytes);

    debug!(
        buffers_analyzed = liveness.len(),
        buffers_reused,
        memory_saved_bytes = memory_saved,
        "memory planner complete"
    );

    MemoryPlannerResult { buffer_replace, memory_saved, buffers_reused, metrics }
}

/// Apply buffer replacements to the schedule.
///
/// Modifies the schedule in place, replacing logical buffers with their
/// physical replacements.
pub fn apply_buffer_replacements(schedule: &mut Schedule, replacements: &HashMap<BufferKey, Buffer>) {
    for (&key, replacement) in replacements {
        if let Some(item) = schedule.get_mut(key.kernel_idx)
            && let Some(buffer) = item.buffers.get_mut(key.buffer_idx)
        {
            *buffer = replacement.clone();
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
#[path = "../test/unit/memory_planner.rs"]
mod tests;
