//! Beam search auto-tuning for kernel optimization.
//!
//! Implements a beam search algorithm that explores the optimization space
//! to find high-performance kernel configurations. This is slower than
//! heuristic-based optimization but can achieve ML-quality performance.
//!
//! # Algorithm
//!
//! 1. Start with base scheduler
//! 2. Generate all valid actions (OptOps applications)
//! 3. Compile and time each candidate
//! 4. Keep top K (beam width) by timing
//! 5. Repeat until the best time stops improving by `min_progress`, or no candidate remains
//!
//! # Caching
//!
//! Results are cached to disk using sled. The cache key includes the AST,
//! optimizer behavior, exact renderer capabilities, and compiler identity.
//! Caching can be disabled via the IGNORE_BEAM_CACHE environment variable.

use std::sync::Arc;
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use svod_ir::{AxisType, ConstValue, Op, UOp};

use super::Scheduler;
use super::config::{BeamConfig, HeuristicsConfig};
use super::error::*;
use super::opts::apply_opt;
use super::types::{Opt, OptArg, OptOps};
use svod_ir::ops;

// ============================================================================
// ACTION SPACE
// ============================================================================

/// Pre-computed action space for beam search.
pub static BEAM_ACTIONS: Lazy<Vec<Opt>> = Lazy::new(|| {
    let mut actions = Vec::with_capacity(600);

    // UPCAST: axes 0-7, amounts [0, 2, 3, 4, 5, 7]
    // amount=0 means "full size" - handled specially in apply
    for &amt in &[0, 2, 3, 4, 5, 7] {
        for axis in 0..8 {
            actions.push(Opt::upcast(axis, amt));
        }
    }

    // UNROLL: axes 0-4, amounts [0, 4, 7]
    for &amt in &[0, 4, 7] {
        for axis in 0..5 {
            actions.push(Opt::unroll(axis, amt));
        }
    }

    // LOCAL: axes 0-5, amounts [2, 3, 4, 8, 13, 16, 29]
    for &amt in &[2, 3, 4, 8, 13, 16, 29] {
        for axis in 0..6 {
            actions.push(Opt::local(axis, amt));
        }
    }
    // GROUPTOP: axes 0-2, amounts [13, 16, 28, 29, 32, 49, 64, 256]
    for &amt in &[13, 16, 28, 29, 32, 49, 64, 256] {
        for axis in 0..3 {
            actions.push(Opt::grouptop(axis, amt));
        }
    }

    // GROUP: axes 0-2, amounts [0, 4, 8, 16]
    for &amt in &[0, 4, 8, 16] {
        for axis in 0..3 {
            actions.push(Opt::group(axis, amt));
        }
    }

    if std::env::var("BEAM_PADTO").ok().and_then(|value| value.parse::<usize>().ok()).unwrap_or(0) != 0 {
        for axis in 0..7 {
            actions.push(Opt::padto(axis, 32));
        }
    }

    // Hand-tuned LOCAL extras outside the grid.
    actions.push(Opt::local(0, 32));
    actions.push(Opt::local(6, 2));

    // TC: tensor cores. 1 default-axis action + 18 axis variants = 19 actions.
    // `detect_matmul` lists every (N, M, K) with the operands one way round and
    // then the other; a conv with two spatial axes and a three-axis reduce has
    // twelve. Survivors after post-compile dedup are unchanged compared to a
    // wider brute-force enumeration because `seen_libs` collapses duplicate
    // kernels, and an out-of-range choice fails in `apply_opt` at no cost.
    const TC_AXIS_CHOICES: usize = 18;
    const TC_OPT_DEFAULT: usize = 0;
    const TC_OPT_AXIS: usize = 2;
    let use_tc = std::env::var("TC").ok().and_then(|value| value.parse().ok()).unwrap_or(1);
    let tc_opt = std::env::var("TC_OPT").ok().and_then(|value| value.parse().ok()).unwrap_or(TC_OPT_AXIS);
    actions.push(Opt::tc(Some(0), -1, TC_OPT_DEFAULT, use_tc));
    for axis_choice in 0..TC_AXIS_CHOICES {
        actions.push(Opt::tc(Some(axis_choice), -1, tc_opt, use_tc));
    }

    // SWAP: axis pairs
    for a0 in 0..5 {
        for a1 in (a0 + 1)..5 {
            actions.push(Opt::swap(a0, a1));
        }
    }

    // THREAD: Tinygrad's fixed amount-major grid. Applicability is decided by
    // apply_opt against the candidate's post-action shape.
    for &amt in &[2, 3, 4, 5, 8, 12, 16, 24, 32, 64] {
        for axis in 0..3 {
            actions.push(Opt::thread(axis, amt));
        }
    }

    actions
});

// ============================================================================
// ACTION GENERATION & FILTERING
// ============================================================================

/// `(op, axis)` pairs that have an `arg=0` (full-axis) variant in
/// [`BEAM_ACTIONS`]. Used by [`passes_prefilter`] to dedup the explicit
/// `arg=axis_size` variants whenever the `arg=0` variant covers the same case.
static FULL_AXIS_VARIANTS: Lazy<std::collections::HashSet<(OptOps, usize)>> = Lazy::new(|| {
    BEAM_ACTIONS
        .iter()
        .filter_map(|opt| {
            let axis = opt.axis?;
            match opt.arg {
                OptArg::Int(0) => Some((opt.op, axis)),
                _ => None,
            }
        })
        .collect()
});

/// Pre-apply filter with two early-rejects:
///
/// 1. The action's logical axis can't be resolved (would always fail in
///    `apply_opt`). Skips the candidate clone+apply roundtrip.
/// 2. The action's `arg` already equals the axis's full size AND an `arg=0`
///    variant exists in `BEAM_ACTIONS` for the same `(op, axis)`. The two
///    actions produce the same kernel post-codegen, so we drop the explicit
///    one to halve dedup work.
fn passes_prefilter(scheduler: &Scheduler, action: &Opt) -> bool {
    // TC and NOLOCALS skip the filter — they have no logical axis.
    if action.op == OptOps::TC || action.axis.is_none() {
        return true;
    }
    // Resolve the logical axis to a real axis. Failure → action would fail
    // at apply time; skip now.
    let real_axis = match scheduler.real_axis(action.op, action.axis) {
        Ok(a) if a >= 0 => a as usize,
        _ => return false,
    };
    if real_axis >= scheduler.shape_len() {
        return false;
    }
    // Dedup: skip if `arg == full_shape[real_axis]` and an `arg=0` variant
    // covers the same case. Only `OptArg::Int` carries a comparable arg.
    if let OptArg::Int(arg) = action.arg
        && arg > 0
        && let Some(&size) = scheduler.full_shape().get(real_axis)
        && size as usize == arg
        && let Some(axis) = action.axis
        && FULL_AXIS_VARIANTS.contains(&(action.op, axis))
    {
        return false;
    }
    true
}

/// `BEAM_DEBUG=1` toggles eprintln! tracing of action survival across
/// the prefilter/apply/limit/time stages. Cheap when disabled (one env-cached
/// bool check per call); useful for diagnosing why an action class never wins.
fn beam_debug_enabled() -> bool {
    static CACHED: Lazy<bool> = Lazy::new(|| {
        std::env::var("BEAM_DEBUG").ok().map(|value| value.parse::<u8>().unwrap_or(1) > 0).unwrap_or(false)
    });
    *CACHED
}

/// Per-stage candidate counts, broken out by [`OptOps`] kind. Aggregated by
/// [`generate_actions`] when [`beam_debug_enabled`] is on.
#[derive(Default, Debug)]
struct ActionStageCounts {
    attempted: std::collections::HashMap<OptOps, usize>,
    prefilter_dropped: std::collections::HashMap<OptOps, usize>,
    apply_dropped: std::collections::HashMap<OptOps, usize>,
    limit_dropped: std::collections::HashMap<OptOps, usize>,
    survived: std::collections::HashMap<OptOps, usize>,
}

/// Generate all valid next-states from the current scheduler.
///
/// Applies each action from `BEAM_ACTIONS` and filters to those that:
/// 1. Pass the cheap [`passes_prefilter`] gate (axis resolves, no arg-eq-size dup)
/// 2. Apply successfully (divisibility, bounds, etc.)
/// 3. Pass limit checks (upcast size, local size, UOp count)
fn generate_actions(scheduler: &Scheduler, config: &BeamConfig) -> Vec<Scheduler> {
    let debug = beam_debug_enabled();
    let mut counts = ActionStageCounts::default();
    let mut out = Vec::with_capacity(BEAM_ACTIONS.len());

    for action in BEAM_ACTIONS.iter() {
        if debug {
            *counts.attempted.entry(action.op).or_insert(0) += 1;
        }
        if !passes_prefilter(scheduler, action) {
            if debug {
                *counts.prefilter_dropped.entry(action.op).or_insert(0) += 1;
            }
            continue;
        }
        let mut candidate = scheduler.clone();
        match apply_opt(&mut candidate, action, true) {
            Ok(()) => {
                if !validate_limits(&candidate, config) {
                    if debug {
                        *counts.limit_dropped.entry(action.op).or_insert(0) += 1;
                    }
                    continue;
                }
                if debug {
                    *counts.survived.entry(action.op).or_insert(0) += 1;
                }
                out.push(candidate);
            }
            Err(_) => {
                if debug {
                    *counts.apply_dropped.entry(action.op).or_insert(0) += 1;
                }
            }
        }
    }

    if config.enable_nolocals {
        let action = Opt::nolocals();
        let mut candidate = scheduler.clone();
        if apply_opt(&mut candidate, &action, true).is_ok() && validate_limits(&candidate, config) {
            out.push(candidate);
        }
    }

    if debug {
        let ops_in_order = [
            OptOps::TC,
            OptOps::UPCAST,
            OptOps::UNROLL,
            OptOps::LOCAL,
            OptOps::GROUP,
            OptOps::GROUPTOP,
            OptOps::THREAD,
            OptOps::SWAP,
            OptOps::PADTO,
            OptOps::NOLOCALS,
        ];
        eprintln!("[beam] generate_actions: {} survivors", out.len());
        // Print every action class, not only the ones with non-zero
        // `attempted`. A class with `attempted=0` means the BEAM_ACTIONS
        // static doesn't even contain that variant — useful for catching
        // missing actions vs. catastrophically high apply/limit drops.
        for op in ops_in_order {
            let a = counts.attempted.get(&op).copied().unwrap_or(0);
            let pf = counts.prefilter_dropped.get(&op).copied().unwrap_or(0);
            let ap = counts.apply_dropped.get(&op).copied().unwrap_or(0);
            let lim = counts.limit_dropped.get(&op).copied().unwrap_or(0);
            let s = counts.survived.get(&op).copied().unwrap_or(0);
            eprintln!("  {op:?}: attempted={a:3} prefilter={pf:3} apply_err={ap:3} limit={lim:3} survived={s:3}");
        }
    }

    out
}

/// The hand-coded kernel, timed with the first BEAM wave and held against the
/// search's answer.
///
/// Greedy width-K expansion prunes a lineage whose single actions each lose,
/// so a multi-opt heuristic stack (e.g. the matvec GROUP+LOCAL+UPCAST+UNROLL
/// one) is unreachable from the bare scheduler. Timing it makes the search's
/// answer never worse than the heuristic one. It does not join the beam: a
/// complete plan outruns every single action in the first wave, its children
/// fill the second, and the search then ends as soon as nothing beats it, so
/// at width 4 a 40x40 stride-2 conv stayed on the heuristics' padded 2.0 ms
/// tile where the same search without the seed reached 1.2 ms. Held aside,
/// the seed can still win but cannot steer.
///
/// `None` when the heuristics add nothing or land outside the search's limits,
/// or under `BEAM_SEED=0`, which measures what the seed is worth.
fn heuristic_seed(scheduler: &Scheduler, config: &BeamConfig) -> Option<Scheduler> {
    static ENABLED: Lazy<bool> =
        Lazy::new(|| std::env::var("BEAM_SEED").ok().map(|value| value.parse::<u8>().unwrap_or(1) > 0).unwrap_or(true));
    if !*ENABLED {
        return None;
    }
    let mut seed = scheduler.clone();
    super::heuristics::hand_coded_optimizations(&mut seed, &HeuristicsConfig::from_env());
    (seed.applied_opts != scheduler.applied_opts && validate_limits(&seed, config)).then_some(seed)
}

/// Under `BEAM_DEBUG`, report once whether the [`heuristic_seed`] survived
/// compilation and how its timing compares to the wave's winner.
fn debug_seed_fate(seed: Option<&[Opt]>, first_wave: bool, timed: &[(Scheduler, Duration)]) {
    let Some(opts) = seed.filter(|_| first_wave && beam_debug_enabled()) else { return };
    let timing = timed.iter().find(|(state, _)| state.applied_opts == opts).map(|(_, timing)| *timing);
    eprintln!("[beam] seed {opts:?}: timing={timing:?} wave best={:?}", timed.first().map(|(_, timing)| *timing));
}

/// The faster of the search's answer and the [`heuristic_seed`], if it was timed.
fn better(
    searched: Option<(Scheduler, Duration)>,
    seed: Option<(Scheduler, Duration)>,
) -> Option<(Scheduler, Duration)> {
    match (searched, seed) {
        (Some(searched), Some(seed)) => Some(if seed.1 < searched.1 { seed } else { searched }),
        (searched, seed) => searched.or(seed),
    }
}

/// Pull the seed's timing out of the first wave, so it competes at the end
/// rather than in the beam.
fn hold_seed(
    timed: &mut Vec<(Scheduler, Duration)>,
    seed: Option<&[Opt]>,
    first_wave: bool,
) -> Option<(Scheduler, Duration)> {
    let seed = seed.filter(|_| first_wave)?;
    let at = timed.iter().position(|(state, _)| state.applied_opts == seed)?;
    Some(timed.remove(at))
}

/// Validate that a scheduler state is within configured limits.
///
/// Per-candidate filter: reject if `(up_axes_prod / tc_up) > max_upcast`
/// or `local_axes_prod > max_local`, where `tc_up = prod(tc.dims) /
/// tc.threads` if a TC is active else 1.
///
/// The `tc_up` divisor accounts for the TC tile's contribution to the
/// total UPCAST/UNROLL product — without it, applying TC immediately
/// saturates `max_upcast` (e.g. `METAL_888` `prod((8,8,8))/32 = 16`),
/// blocking any post-TC UPCAST composition.
fn validate_limits(scheduler: &Scheduler, config: &BeamConfig) -> bool {
    let upcast_sz = product_of_axes(scheduler, &[AxisType::Upcast, AxisType::Unroll]);
    let local_sz = product_of_axes(scheduler, &[AxisType::Local, AxisType::Warp, AxisType::GroupReduce]);
    let tc_up = active_tc_upcast(scheduler);

    upcast_sz / tc_up <= config.max_upcast && local_sz <= config.max_local
}

/// Reconstruct one remote BEAM candidate without creating candidate UOps in
/// the parent process.
///
/// The suffix past `base_opt_count` is replayed whole: one action for an
/// expanded candidate, the full stack for a [`heuristic_seed`]. [`passes_prefilter`]
/// is not re-run here — it is a parent-side generation filter, and a seed's opts
/// never went through it.
pub fn apply_remote_candidate(
    mut scheduler: Scheduler,
    base_opt_count: usize,
    opts: &[Opt],
    config: &BeamConfig,
) -> Option<Scheduler> {
    if opts.len() < base_opt_count || opts[..base_opt_count] != scheduler.applied_opts {
        return None;
    }
    if opts.len() == base_opt_count {
        return validate_limits(&scheduler, config).then_some(scheduler);
    }
    for opt in &opts[base_opt_count..] {
        apply_opt(&mut scheduler, opt, true).ok()?;
    }
    validate_limits(&scheduler, config).then_some(scheduler)
}

/// Return `prod(tc.dims) / tc.threads` for the active TC, or 1 if none.
///
/// Uses `scheduler.selected_tc_index` (recorded by `apply_axis_choice_impl`)
/// rather than guessing from the renderer's TC list. For multi-TC renderers
/// (e.g. SM89 with f16+bf16+tf32 variants) this is the only correct
/// accounting.
fn active_tc_upcast(scheduler: &Scheduler) -> usize {
    let Some(idx) = scheduler.selected_tc_index else {
        return 1;
    };
    scheduler
        .ren
        .tensor_cores
        .get(idx)
        .map(|tc| {
            let prod = tc.dims.0 * tc.dims.1 * tc.dims.2;
            prod / tc.threads.max(1)
        })
        .unwrap_or(1)
}

/// Calculate product of dimension sizes for given axis types.
fn product_of_axes(scheduler: &Scheduler, types: &[AxisType]) -> usize {
    scheduler
        .rngs()
        .iter()
        .filter_map(|rng| {
            if let Op::Range(ops::Range { axis_type, end, .. }) = rng.op()
                && types.contains(axis_type)
                && let Op::Const(cv) = end.op()
                && let ConstValue::Int(sz) = cv.0
            {
                Some(sz as usize)
            } else {
                None
            }
        })
        .product::<usize>()
        .max(1)
}

// ============================================================================
// BEAM SEARCH ALGORITHM
// ============================================================================

/// Beam search result containing optimized scheduler and timing.
pub struct BeamResult {
    /// Optimized scheduler state.
    pub scheduler: Scheduler,
    /// Best timing achieved.
    pub timing: Duration,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Total candidates evaluated.
    pub candidates_evaluated: usize,
    /// Total candidates generated by action expansion.
    pub generated: usize,
    /// Legacy structural-IR metric. Production BEAM follows Tinygrad's compiled
    /// binary identity and leaves this at zero.
    pub unique_ir: usize,
    /// Candidates successfully compiled by the backend.
    pub compiled: usize,
    /// Exact unique compiled binaries or sources.
    pub unique_binary: usize,
    /// Candidates benchmarked on the target backend.
    pub benchmarked: usize,
    /// Aggregate wall time spent in each search stage.
    pub stage_timings: BeamStageTimings,
}

/// Aggregate BEAM pipeline timings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BeamStageTimings {
    pub generation: Duration,
    pub filtering: Duration,
    pub compilation: Duration,
    pub binary_dedup: Duration,
    pub benchmarking: Duration,
}

/// A compiled candidate with an exact binary/source identity.
pub struct CompiledCandidate<T> {
    pub artifact: T,
    pub binary_key: Vec<u8>,
    /// `None` when the AST has no reliable count; such a candidate is never
    /// filtered as bloated and never lowers the bar for the others.
    pub compute_ops: Option<u64>,
    pub preparation: Duration,
    pub compilation: Duration,
}

/// Metrics returned by the `compile_and_time` closure for each candidate.
///
/// Timing drives ranking; the IR hash drives `seen_libs` dedup; the compute-op
/// count drives the `least_compute_ops*1000` filter.
#[derive(Debug, Clone, Copy)]
pub struct CandidateMetrics {
    /// Best execution timing across the run loop (`min(tms)`).
    pub timing: Duration,
    /// Hash of the post-codegen IR — kernels that lower to the same IR are
    /// guaranteed to compile to the same object, so we skip duplicates.
    pub ir_hash: u64,
    /// Cheap upper bound on the kernel's compute work; used by the
    /// `least_compute_ops*1000` filter to discard degenerate candidates.
    /// `None` when no reliable count exists, which exempts the candidate.
    pub compute_ops: Option<u64>,
}

/// The `least_compute_ops*1000` bloat filter: fold `compute_ops` into the
/// running minimum and report whether the candidate is a thousand times the
/// leanest one seen so far.
fn bloated(least_compute_ops: &mut u64, compute_ops: Option<u64>) -> bool {
    let Some(compute_ops) = compute_ops else { return false };
    *least_compute_ops = (*least_compute_ops).min(compute_ops);
    least_compute_ops.saturating_mul(1000) < compute_ops
}

/// Hash a UOp tree to a `u64` for `seen_libs` dedup.
///
/// Uses the pre-computed `content_hash` field on `UOp` (see
/// `ir/src/uop/hash_consing.rs`), which is the same structural hash the
/// hash-consing cache and `schedule_cache` rely on. O(1) — read the cached
/// field instead of re-walking the graph.
pub fn hash_post_codegen_ir(uop: &Arc<UOp>) -> u64 {
    uop.content_hash
}

// The symbolic compute-ops estimate is an AST-only walk, so it lives in `ir`
// (shared with the runtime profiler's roofline). Re-exported here to keep the
// `schedule` BEAM call sites and public surface unchanged.
pub use svod_ir::compute_ops_estimate;

/// Run beam search optimization.
///
/// # Arguments
///
/// * `scheduler` - Initial scheduler state
/// * `config` - Beam search configuration
/// * `compile_and_time` - Function to compile and time a scheduler state
///
/// # Returns
///
/// `BeamResult` containing the best scheduler found and performance metrics.
///
/// # Example
///
/// ```ignore
/// let config = BeamConfig::default();
/// let compile_and_time = |s: &Scheduler, early_stop: Option<Duration>| {
///     let ast = s.get_optimized_ast(None);
///     let kernel = compile_kernel(&ast)?;
///     let bench = benchmark_kernel(&kernel, ..., early_stop)?;
///     Some(CandidateMetrics { timing: bench.min, ir_hash: ..., compute_ops: ... })
/// };
///
/// let result = beam_search(scheduler, &config, compile_and_time)?;
/// println!("Best time: {:?}", result.timing);
/// ```
pub fn beam_search<F>(scheduler: Scheduler, config: &BeamConfig, compile_and_time: F) -> Result<BeamResult, OptError>
where
    F: Fn(&Scheduler, Option<Duration>) -> Option<CandidateMetrics> + Sync,
{
    let mut iterations = 0;
    let mut candidates_evaluated = 0;

    // Initialize beam with `Duration::MAX` so the first iteration has no
    // incumbent to beat. Avoids one wasted compile+time per `beam_search`
    // invocation (also charged on cache replay through `OPT_CACHE`).
    let mut beam: Vec<(Scheduler, Duration)> = vec![(scheduler.clone(), Duration::MAX)];
    let mut seed = heuristic_seed(&scheduler, config);
    let seed_opts = seed.as_ref().map(|state| state.applied_opts.clone());
    let mut held_seed = None;

    // `seen_libs` and `least_compute_ops` persist across the entire beam
    // search. Identity-keyed dedup carries across iterations, so a kernel
    // produced at iter N and re-produced (via a different opt order) at
    // iter N+1 only gets compiled+timed once.
    let mut seen_libs: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut least_compute_ops: u64 = u64::MAX;

    // No total search budget; terminates on empty candidate set, empty timed
    // list, `min_progress` floor, or sub-noise gain. Per-candidate compile
    // budgets live separately in the backend's compile worker.
    loop {
        iterations += 1;

        // 1. EXPAND: Generate all valid next states from current beam (sequential)
        // Note: Scheduler is not Sync due to OnceCell caches, so expansion is sequential
        let mut candidates: Vec<Scheduler> = beam.iter().flat_map(|(s, _)| generate_actions(s, config)).collect();
        candidates.extend(seed.take());

        if candidates.is_empty() {
            break;
        }

        // Reject any candidate whose first run already exceeds 3× the current beam best.
        let beam_best = beam.first().map(|(_, t)| *t);
        let early_stop = beam_best.and_then(|t| t.checked_mul(3));

        // 2. COMPILE & TIME: Evaluate performance
        let mut timed: Vec<(Scheduler, Duration)> = Vec::new();
        for s in candidates {
            let Some(metrics) = compile_and_time(&s, early_stop) else { continue };

            if !seen_libs.insert(metrics.ir_hash) {
                continue;
            }
            if bloated(&mut least_compute_ops, metrics.compute_ops) {
                continue;
            }

            timed.push((s, metrics.timing));
        }

        candidates_evaluated += timed.len();

        if timed.is_empty() {
            break;
        }

        if beam_debug_enabled() {
            // Bucket survivors by the *last* applied opt (the one this iteration
            // just stacked on). Useful for spotting "TC compiled but lost on
            // timing vs UPCAST" vs "TC never survived to timing at all".
            let mut by_op: std::collections::HashMap<OptOps, (usize, Duration)> = std::collections::HashMap::new();
            for (s, t) in &timed {
                if let Some(opt) = s.applied_opts.last() {
                    let entry = by_op.entry(opt.op).or_insert((0, Duration::MAX));
                    entry.0 += 1;
                    if *t < entry.1 {
                        entry.1 = *t;
                    }
                }
            }
            eprintln!("[beam iter {iterations}] timed survivors by last-op (count, best):");
            let ops_in_order = [
                OptOps::TC,
                OptOps::UPCAST,
                OptOps::UNROLL,
                OptOps::LOCAL,
                OptOps::GROUP,
                OptOps::GROUPTOP,
                OptOps::THREAD,
            ];
            for op in ops_in_order {
                if let Some((cnt, best)) = by_op.get(&op) {
                    eprintln!("  {op:?}: count={cnt:3} best={best:?}");
                }
            }
        }

        // 3. SORT: Sort by timing (best first)
        let mut sorted = timed;
        sorted.sort_by_key(|(_, t)| *t);
        debug_seed_fate(seed_opts.as_deref(), iterations == 1, &sorted);
        if let Some(seed) = hold_seed(&mut sorted, seed_opts.as_deref(), iterations == 1) {
            held_seed = Some(seed);
        }
        if sorted.is_empty() {
            break;
        }

        // 4. CHECK TERMINATION — exit when the new best is already below
        //    the progress floor (fast-enough kernel) OR when the gain over
        //    the incumbent is sub-noise. Sub-noise gains don't justify a
        //    next compile round.
        let best_new = sorted[0].1;
        let best_old = beam.first().map(|(_, t)| *t).unwrap_or(Duration::MAX);
        let min_progress = Duration::from_nanos(config.min_progress_ns);
        let absolute_floor = best_new < min_progress;
        let no_real_gain = best_old.saturating_sub(best_new) < min_progress;

        if absolute_floor || no_real_gain {
            // When exiting AND we did improve, pin the beam to the single
            // new winner so callers see it.
            if best_new < best_old {
                beam = sorted.into_iter().take(1).collect();
            }
            break;
        }

        // 5. PRUNE: Keep top K by timing
        beam = sorted.into_iter().take(config.beam_width).collect();
    }

    let (best_scheduler, best_timing) =
        better(beam.into_iter().next(), held_seed).unwrap_or((scheduler, Duration::MAX));

    Ok(BeamResult {
        scheduler: best_scheduler,
        timing: best_timing,
        iterations,
        candidates_evaluated,
        generated: candidates_evaluated,
        unique_ir: candidates_evaluated,
        compiled: candidates_evaluated,
        unique_binary: candidates_evaluated,
        benchmarked: candidates_evaluated,
        stage_timings: BeamStageTimings::default(),
    })
}

/// Run BEAM with unordered compile completions and serialized immediate timing.
///
/// `compile_wave` mirrors Tinygrad's `imap_unordered`: it emits one completed
/// compile at a time with the original candidate index. The callback benchmarks
/// that artifact before the next completion is accepted, so no compiled wave is
/// retained in the parent.
pub fn beam_search_staged<C, FC, FT>(
    scheduler: Scheduler,
    config: &BeamConfig,
    mut compile_wave: FC,
    benchmark: FT,
) -> Result<BeamResult, OptError>
where
    FC: FnMut(&[Scheduler], &mut dyn FnMut(usize, CompiledCandidate<C>)),
    FT: Fn(&C, Option<Duration>) -> Option<Duration>,
{
    let mut result = BeamResult {
        scheduler: scheduler.clone(),
        timing: Duration::MAX,
        iterations: 0,
        candidates_evaluated: 0,
        generated: 0,
        unique_ir: 0,
        compiled: 0,
        unique_binary: 0,
        benchmarked: 0,
        stage_timings: BeamStageTimings::default(),
    };
    let mut beam = vec![(scheduler.clone(), Duration::MAX)];
    let mut seed = heuristic_seed(&scheduler, config);
    let seed_opts = seed.as_ref().map(|state| state.applied_opts.clone());
    let mut held_seed = None;
    let mut seen_binary = std::collections::HashSet::new();

    loop {
        result.iterations += 1;

        let started = Instant::now();
        let mut candidates: Vec<Scheduler> =
            beam.iter().flat_map(|(state, _)| generate_actions(state, config)).collect();
        candidates.extend(seed.take());
        result.stage_timings.generation += started.elapsed();
        result.generated += candidates.len();
        if candidates.is_empty() {
            break;
        }

        let beam_best = beam.first().map(|(_, timing)| *timing);
        let early_stop = beam_best.and_then(|timing| timing.checked_mul(3));
        let mut timed = Vec::new();
        // Tinygrad resets this for each candidate wave and updates it in
        // completion order, before adding the binary to `seen_libs`.
        let mut least_compute_ops = u64::MAX;
        compile_wave(&candidates, &mut |index, compiled| {
            if index >= candidates.len() {
                return;
            }
            result.compiled += 1;
            result.stage_timings.filtering += compiled.preparation;
            result.stage_timings.compilation += compiled.compilation;
            if bloated(&mut least_compute_ops, compiled.compute_ops) {
                return;
            }
            let started = Instant::now();
            if !seen_binary.insert(compiled.binary_key) {
                result.stage_timings.binary_dedup += started.elapsed();
                return;
            }
            result.stage_timings.binary_dedup += started.elapsed();
            result.unique_binary += 1;
            let started = Instant::now();
            if let Some(timing) = benchmark(&compiled.artifact, early_stop) {
                result.benchmarked += 1;
                timed.push((candidates[index].clone(), timing));
            }
            result.stage_timings.benchmarking += started.elapsed();
        });
        result.candidates_evaluated = result.benchmarked;
        if timed.is_empty() {
            break;
        }

        timed.sort_by_key(|(_, timing)| *timing);
        debug_seed_fate(seed_opts.as_deref(), result.iterations == 1, &timed);
        if let Some(seed) = hold_seed(&mut timed, seed_opts.as_deref(), result.iterations == 1) {
            held_seed = Some(seed);
        }
        if timed.is_empty() {
            break;
        }
        let best_new = timed[0].1;
        let best_old = beam.first().map(|(_, timing)| *timing).unwrap_or(Duration::MAX);
        let min_progress = Duration::from_nanos(config.min_progress_ns);
        if best_new < min_progress || best_old.saturating_sub(best_new) < min_progress {
            if best_new < best_old {
                beam = timed.into_iter().take(1).collect();
            }
            break;
        }
        beam = timed.into_iter().take(config.beam_width).collect();
    }

    let (best_scheduler, best_timing) =
        better(beam.into_iter().next(), held_seed).unwrap_or((scheduler, Duration::MAX));
    result.scheduler = best_scheduler;
    result.timing = best_timing;
    Ok(result)
}

/// BEAM search whose candidate scheduler construction and compilation happen
/// in external workers. The parent retains only optimization sequences.
pub fn beam_search_remote_staged<C, FC, FT>(
    scheduler: Scheduler,
    config: &BeamConfig,
    mut compile_wave: FC,
    benchmark: FT,
) -> Result<BeamResult, OptError>
where
    FC: FnMut(&[Vec<Opt>], &mut dyn FnMut(usize, CompiledCandidate<C>)) -> Result<(), OptError>,
    FT: Fn(&C, Option<Duration>) -> Option<Duration>,
{
    let mut result = BeamResult {
        scheduler: scheduler.clone(),
        timing: Duration::MAX,
        iterations: 0,
        candidates_evaluated: 0,
        generated: 0,
        unique_ir: 0,
        compiled: 0,
        unique_binary: 0,
        benchmarked: 0,
        stage_timings: BeamStageTimings::default(),
    };
    let mut beam = vec![(scheduler.clone(), Duration::MAX)];
    let mut seed = heuristic_seed(&scheduler, config);
    let seed_opts = seed.as_ref().map(|state| state.applied_opts.clone());
    let mut held_seed = None;
    let mut seen_binary = std::collections::HashSet::new();

    loop {
        result.iterations += 1;
        let started = Instant::now();
        let mut candidates = beam.iter().flat_map(|(state, _)| generate_actions(state, config)).collect::<Vec<_>>();
        candidates.extend(seed.take());
        let candidate_opts = candidates.iter().map(|candidate| candidate.applied_opts.clone()).collect::<Vec<_>>();
        result.stage_timings.generation += started.elapsed();
        result.generated += candidates.len();
        if candidates.is_empty() {
            break;
        }

        let early_stop = beam.first().and_then(|(_, timing)| timing.checked_mul(3));
        let mut timed = Vec::new();
        let mut least_compute_ops = u64::MAX;
        compile_wave(&candidate_opts, &mut |index, compiled| {
            if index >= candidates.len() {
                return;
            }
            result.compiled += 1;
            result.stage_timings.filtering += compiled.preparation;
            result.stage_timings.compilation += compiled.compilation;
            if bloated(&mut least_compute_ops, compiled.compute_ops) {
                return;
            }
            let started = Instant::now();
            if !seen_binary.insert(compiled.binary_key) {
                result.stage_timings.binary_dedup += started.elapsed();
                return;
            }
            result.stage_timings.binary_dedup += started.elapsed();
            result.unique_binary += 1;
            let started = Instant::now();
            if let Some(timing) = benchmark(&compiled.artifact, early_stop) {
                result.benchmarked += 1;
                timed.push((candidates[index].clone(), timing));
            }
            result.stage_timings.benchmarking += started.elapsed();
        })?;
        result.candidates_evaluated = result.benchmarked;
        if timed.is_empty() {
            break;
        }
        timed.sort_by_key(|(_, timing)| *timing);
        debug_seed_fate(seed_opts.as_deref(), result.iterations == 1, &timed);
        if let Some(seed) = hold_seed(&mut timed, seed_opts.as_deref(), result.iterations == 1) {
            held_seed = Some(seed);
        }
        if timed.is_empty() {
            break;
        }
        let best_new = timed[0].1;
        let best_old = beam.first().map(|(_, timing)| *timing).unwrap_or(Duration::MAX);
        let min_progress = Duration::from_nanos(config.min_progress_ns);
        if best_new < min_progress || best_old.saturating_sub(best_new) < min_progress {
            if best_new < best_old {
                beam = timed.into_iter().take(1).collect();
            }
            break;
        }
        beam = timed.into_iter().take(config.beam_width).collect();
    }

    let (winner, timing) = better(beam.into_iter().next(), held_seed).unwrap_or((scheduler, Duration::MAX));
    result.scheduler = winner;
    result.timing = timing;
    Ok(result)
}

// ============================================================================
// REPLAY
// ============================================================================

/// Replay a sequence of optimizations on a scheduler.
///
/// Used to restore cached beam search results.
pub fn replay_opts(mut scheduler: Scheduler, opts: &[Opt]) -> Result<Scheduler, OptError> {
    for opt in opts {
        apply_opt(&mut scheduler, opt, true)?;
    }
    Ok(scheduler)
}

/// Get the applied optimizations from a scheduler.
pub fn get_applied_opts(scheduler: &Scheduler) -> &[Opt] {
    &scheduler.applied_opts
}

// ============================================================================
// CACHING
// ============================================================================

/// Global sled database for beam search cache.
///
/// Lazy-initialized on first access. Returns None if cache directory
/// cannot be created or database cannot be opened.
///
/// `SVOD_BEAM_CACHE_DIR` overrides the location, mirroring
/// `SVOD_OBJECT_CACHE_DIR`. sled takes an **exclusive lock** on the database, so
/// two processes sharing one location cannot both open it — under a
/// process-per-test runner that means every process after the first sees `None`.
/// The override is the only portable way to give them separate roots:
/// `dirs::cache_dir()` reads `XDG_CACHE_HOME` on Linux but returns
/// `~/Library/Caches` unconditionally on macOS, so redirecting that variable
/// isolates the tests on one platform and silently does nothing on the other.
static CACHE_DB: Lazy<Option<sled::Db>> = Lazy::new(|| {
    let cache_dir = match std::env::var_os("SVOD_BEAM_CACHE_DIR") {
        Some(path) => std::path::PathBuf::from(path),
        None => dirs::cache_dir()?.join("svod"),
    };
    std::fs::create_dir_all(&cache_dir).ok()?;
    match sled::open(cache_dir.join("beam_cache")) {
        Ok(db) => Some(db),
        Err(error) => {
            tracing::warn!(
                %error,
                dir = %cache_dir.display(),
                "BEAM cache unavailable: every kernel is searched afresh and nothing is saved \
                 (another svod process holding the database lock is the usual cause)"
            );
            None
        }
    }
});

/// Cache key for beam search results.
///
/// Includes the limit configuration (max_upcast, max_local, max_uops) so that
/// changing caps invalidates cached entries: replaying opts produced under a
/// looser cap could reintroduce a kernel that no longer satisfies the new cap.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CacheKey {
    /// On-disk key schema. Bump whenever replay semantics change.
    schema: u32,
    /// Hash of the AST structure.
    ast_hash: u64,
    /// Beam width used for search.
    beam_width: usize,
    /// Renderer/TC backend.
    device: svod_ir::RendererDevice,
    /// Full target/capability/rewrite identity.
    renderer_fingerprint: u64,
    /// Exact compiler backend, target, toolchain, flags, and ABI identity.
    compiler_identity: String,
    /// Upcast/unroll product cap at search time.
    max_upcast: usize,
    /// Local/warp/group_reduce product cap at search time.
    max_local: usize,
    /// UOp count cap at search time.
    max_uops: usize,
    /// Benchmark samples used for ranking.
    num_runs: usize,
    /// Search termination threshold in nanoseconds.
    min_progress_ns: u64,
    /// NOLOCALS action-space gate.
    enable_nolocals: bool,
    /// Candidate acceptance compile timeout.
    compile_timeout_secs: u64,
    /// Post-optimization behavior not represented by BeamConfig.
    behavior_fingerprint: u64,
    /// Identity of the action space the plan was searched in.
    action_space: u64,
}

/// Structural hash of a beam action space.
///
/// [`BEAM_ACTIONS`] is built from `BEAM_PADTO`, `TC` and `TC_OPT`, so a cached
/// plan is only replayable under the action space that produced it. Tinygrad's
/// `search.py:116` key is `{ast, amt, allow_test_size, device, suffix}` and has
/// the same hazard (its `actions` list reads the same env vars at import); this
/// is a deliberate go-beyond.
pub(crate) fn action_space_hash(actions: &[Opt]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    actions.hash(&mut hasher);
    hasher.finish()
}

impl CacheKey {
    /// Create a cache key from a scheduler and config.
    fn from_scheduler(
        scheduler: &Scheduler,
        config: &BeamConfig,
        compiler_identity: &str,
        behavior_fingerprint: u64,
    ) -> Self {
        // Use structural hash for cross-run stability. The recursive Hash for UOp
        // traverses (dtype, op) of the entire DAG — same AST structure produces
        // the same hash regardless of process-local ids.
        use std::hash::{Hash, Hasher};
        debug_assert!(
            scheduler.ast().toposort().iter().all(|node| node.origin().is_none()),
            "kernel ASTs are stripped at the cut; an origin here would fork the on-disk beam cache"
        );
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        scheduler.ast().hash(&mut hasher);
        let ast_hash = hasher.finish();

        Self {
            schema: 12,
            ast_hash,
            beam_width: config.beam_width,
            device: scheduler.ren.device,
            renderer_fingerprint: scheduler.ren.cache_fingerprint(),
            compiler_identity: compiler_identity.to_string(),
            max_upcast: config.max_upcast,
            max_local: config.max_local,
            max_uops: config.max_uops,
            num_runs: config.num_runs,
            min_progress_ns: config.min_progress_ns,
            enable_nolocals: config.enable_nolocals,
            compile_timeout_secs: config.compile_timeout_secs,
            behavior_fingerprint,
            action_space: action_space_hash(&BEAM_ACTIONS),
        }
    }

    /// Convert to bytes for database key.
    fn to_bytes(&self) -> Vec<u8> {
        let device_str = self.device.canonical();
        let mut bytes = Vec::with_capacity(84 + self.compiler_identity.len() + device_str.len());
        bytes.extend_from_slice(&self.schema.to_le_bytes());
        bytes.extend_from_slice(&self.ast_hash.to_le_bytes());
        bytes.extend_from_slice(&self.renderer_fingerprint.to_le_bytes());
        bytes.extend_from_slice(&self.beam_width.to_le_bytes());
        bytes.extend_from_slice(&self.max_upcast.to_le_bytes());
        bytes.extend_from_slice(&self.max_local.to_le_bytes());
        bytes.extend_from_slice(&self.max_uops.to_le_bytes());
        bytes.extend_from_slice(&self.num_runs.to_le_bytes());
        bytes.extend_from_slice(&self.min_progress_ns.to_le_bytes());
        bytes.push(u8::from(self.enable_nolocals));
        bytes.extend_from_slice(&self.compile_timeout_secs.to_le_bytes());
        bytes.extend_from_slice(&self.behavior_fingerprint.to_le_bytes());
        bytes.extend_from_slice(&self.action_space.to_le_bytes());
        bytes.extend_from_slice(&self.compiler_identity.len().to_le_bytes());
        bytes.extend_from_slice(self.compiler_identity.as_bytes());
        bytes.extend_from_slice(device_str.as_bytes());
        bytes
    }
}

/// Serialize applied opts to bytes for caching using bincode.
fn serialize_opts(opts: &[Opt]) -> Vec<u8> {
    bincode::serde::encode_to_vec(opts, bincode::config::standard()).expect("Opt serialization should not fail")
}

/// Deserialize opts from cached bytes using bincode.
fn deserialize_opts(bytes: &[u8]) -> Option<Vec<Opt>> {
    bincode::serde::decode_from_slice(bytes, bincode::config::standard()).ok().map(|(opts, _)| opts)
}

fn cached_opt_suffix<'a>(scheduler: &Scheduler, cached: &'a [Opt]) -> Option<&'a [Opt]> {
    cached.starts_with(&scheduler.applied_opts).then(|| &cached[scheduler.applied_opts.len()..])
}

/// Replay a cached opt sequence on top of the opts `scheduler` already carries.
fn replay_cached(scheduler: &Scheduler, cached_opts: &[Opt]) -> Result<Scheduler, OptError> {
    tracing::info!(opts = ?cached_opts, "Beam cache HIT - replaying opts");
    let suffix = cached_opt_suffix(scheduler, cached_opts)
        .ok_or(OptError::ValidationFailed { op: "BEAM cache", reason: "cached opts do not extend base opts" })?;
    replay_opts(scheduler.clone(), suffix)
}

/// Get cached beam search result.
fn cache_get(key: &CacheKey) -> Option<Vec<Opt>> {
    let db = CACHE_DB.as_ref()?;
    let bytes = db.get(key.to_bytes()).ok()??;
    let opts = deserialize_opts(&bytes);
    tracing::debug!(ast_hash = format_args!("{:016x}", key.ast_hash), hit = opts.is_some(), "Beam cache lookup");
    opts
}

/// Store beam search result in cache.
fn cache_put(key: &CacheKey, opts: &[Opt]) {
    if let Some(db) = CACHE_DB.as_ref()
        && db.insert(key.to_bytes(), serialize_opts(opts)).is_ok()
    {
        // Flush to disk to ensure persistence across runs
        let _ = db.flush();
    }
}

fn cacheable(result: &BeamResult) -> bool {
    result.benchmarked > 0 && result.timing != Duration::MAX
}

/// Remove a stale cache entry.
fn cache_invalidate(key: &CacheKey) {
    if let Some(db) = CACHE_DB.as_ref() {
        let _ = db.remove(key.to_bytes());
        let _ = db.flush();
    }
}

/// Run beam search with disk caching, replaying a cached plan when one exists.
///
/// `behavior_fingerprint` identifies post-optimization behavior that
/// `BeamConfig` does not capture (see `OptimizerConfig::transcendental` and
/// `disable_fast_idiv`). It is a required argument: a wrapper that pinned it to
/// 0 would silently share cache entries across differing post-opt behavior.
pub fn beam_search_cached_with_behavior<F>(
    scheduler: Scheduler,
    config: &BeamConfig,
    behavior_fingerprint: u64,
    compile_and_time: F,
) -> Result<BeamResult, OptError>
where
    F: Fn(&Scheduler, Option<Duration>) -> Option<CandidateMetrics> + Sync,
{
    let key = CacheKey::from_scheduler(&scheduler, config, "", behavior_fingerprint);

    // Check cache (unless disabled)
    if !config.disable_cache
        && let Some(cached_opts) = cache_get(&key)
    {
        // Replay cached optimizations. If replay fails (stale entry from code changes),
        // or the replayed scheduler exceeds the current limits (looser cap at search
        // time, tighter cap now), invalidate and fall through to fresh search.
        let replayed = replay_cached(&scheduler, &cached_opts);
        match replayed {
            Ok(replayed) if validate_limits(&replayed, config) => {
                if let Some(metrics) = compile_and_time(&replayed, None)
                    && metrics.timing != Duration::MAX
                {
                    return Ok(BeamResult {
                        scheduler: replayed,
                        timing: metrics.timing,
                        iterations: 0,
                        candidates_evaluated: 1,
                        generated: 0,
                        unique_ir: 1,
                        compiled: 1,
                        unique_binary: 1,
                        benchmarked: 1,
                        stage_timings: BeamStageTimings::default(),
                    });
                }
                cache_invalidate(&key);
            }
            Ok(_) => {
                tracing::warn!("Beam cache replayed scheduler violates limits - invalidating");
                cache_invalidate(&key);
            }
            Err(e) => {
                tracing::warn!(?e, "Beam cache replay failed (stale entry?) - invalidating");
                cache_invalidate(&key);
            }
        }
    }

    tracing::info!("Beam cache MISS - running search");
    // Run beam search
    let result = beam_search(scheduler, config, compile_and_time)?;

    // Cache result (unless disabled)
    if !config.disable_cache && cacheable(&result) {
        cache_put(&key, &result.scheduler.applied_opts);
    }

    Ok(result)
}

/// Run staged BEAM with persistent caching and exact compiler identity.
pub fn beam_search_cached_staged<C, FC, FT>(
    scheduler: Scheduler,
    config: &BeamConfig,
    compiler_identity: &str,
    behavior_fingerprint: u64,
    mut compile_wave: FC,
    benchmark: FT,
) -> Result<BeamResult, OptError>
where
    FC: FnMut(&[Scheduler], &mut dyn FnMut(usize, CompiledCandidate<C>)),
    FT: Fn(&C, Option<Duration>) -> Option<Duration>,
{
    let key = CacheKey::from_scheduler(&scheduler, config, compiler_identity, behavior_fingerprint);
    if !config.disable_cache
        && let Some(cached_opts) = cache_get(&key)
    {
        let replayed = replay_cached(&scheduler, &cached_opts);
        match replayed {
            Ok(replayed) if validate_limits(&replayed, config) => {
                let mut compiled_count = 0;
                let mut filtering = Duration::ZERO;
                let mut compilation = Duration::ZERO;
                let mut benchmarking = Duration::ZERO;
                let mut timing = Duration::MAX;
                compile_wave(std::slice::from_ref(&replayed), &mut |index, candidate| {
                    if index != 0 {
                        return;
                    }
                    compiled_count += 1;
                    filtering += candidate.preparation;
                    compilation += candidate.compilation;
                    let started = Instant::now();
                    timing = benchmark(&candidate.artifact, None).unwrap_or(Duration::MAX);
                    benchmarking += started.elapsed();
                });
                let benchmarked = usize::from(timing != Duration::MAX);
                if benchmarked > 0 {
                    return Ok(BeamResult {
                        scheduler: replayed,
                        timing,
                        iterations: 0,
                        candidates_evaluated: benchmarked,
                        generated: 0,
                        unique_ir: 0,
                        compiled: compiled_count,
                        unique_binary: compiled_count,
                        benchmarked,
                        stage_timings: BeamStageTimings {
                            filtering,
                            compilation,
                            benchmarking,
                            ..BeamStageTimings::default()
                        },
                    });
                }
                cache_invalidate(&key);
            }
            Ok(_) => cache_invalidate(&key),
            Err(_) => cache_invalidate(&key),
        }
    }

    let result = beam_search_staged(scheduler, config, &mut compile_wave, &benchmark)?;
    if !config.disable_cache && cacheable(&result) {
        cache_put(&key, &result.scheduler.applied_opts);
    }
    Ok(result)
}

/// Cached variant of [`beam_search_remote_staged`]. Cache replay constructs
/// only the single winning scheduler in the parent.
pub fn beam_search_cached_remote<C, FC, FT>(
    scheduler: Scheduler,
    config: &BeamConfig,
    compiler_identity: &str,
    behavior_fingerprint: u64,
    mut compile_wave: FC,
    benchmark: FT,
) -> Result<BeamResult, OptError>
where
    FC: FnMut(&[Vec<Opt>], &mut dyn FnMut(usize, CompiledCandidate<C>)) -> Result<(), OptError>,
    FT: Fn(&C, Option<Duration>) -> Option<Duration>,
{
    let key = CacheKey::from_scheduler(&scheduler, config, compiler_identity, behavior_fingerprint);
    if !config.disable_cache
        && let Some(cached_opts) = cache_get(&key)
    {
        let replayed = replay_cached(&scheduler, &cached_opts);
        match replayed {
            Ok(replayed) if validate_limits(&replayed, config) => {
                let mut timing = Duration::MAX;
                let mut compiled_count = 0;
                let mut filtering = Duration::ZERO;
                let mut compilation = Duration::ZERO;
                let mut benchmarking = Duration::ZERO;
                let replay_result = compile_wave(std::slice::from_ref(&cached_opts), &mut |index, candidate| {
                    if index != 0 {
                        return;
                    }
                    compiled_count += 1;
                    filtering += candidate.preparation;
                    compilation += candidate.compilation;
                    let started = Instant::now();
                    timing = benchmark(&candidate.artifact, None).unwrap_or(Duration::MAX);
                    benchmarking += started.elapsed();
                });
                if let Err(error) = replay_result {
                    cache_invalidate(&key);
                    return Err(error);
                }
                let benchmarked = usize::from(timing != Duration::MAX);
                if benchmarked > 0 {
                    return Ok(BeamResult {
                        scheduler: replayed,
                        timing,
                        iterations: 0,
                        candidates_evaluated: benchmarked,
                        generated: 0,
                        unique_ir: 0,
                        compiled: compiled_count,
                        unique_binary: compiled_count,
                        benchmarked,
                        stage_timings: BeamStageTimings { filtering, compilation, benchmarking, ..Default::default() },
                    });
                }
                cache_invalidate(&key);
            }
            Ok(_) | Err(_) => cache_invalidate(&key),
        }
    }

    let result = beam_search_remote_staged(scheduler, config, &mut compile_wave, &benchmark)?;
    if !config.disable_cache && cacheable(&result) {
        cache_put(&key, &result.scheduler.applied_opts);
    }
    Ok(result)
}

/// Clear the beam search cache.
///
/// Useful for testing or when invalidating cached results.
pub fn clear_cache() {
    if let Some(db) = CACHE_DB.as_ref() {
        let _ = db.clear();
    }
}

#[cfg(test)]
#[path = "../test/unit/optimizer/beam_internal.rs"]
mod tests;
