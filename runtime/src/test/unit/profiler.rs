//! Unit tests for the profiler data model, table rendering (GPU-free), lane
//! metrics, and [`RunProfile::merge`] accumulation semantics.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use svod_device::device::Program;
use svod_device::hcq::{
    DeviceQueue, QueueKind, QueueMergeLimits, SemanticLinkedPlan, TopologyOperation, TopologyOperationKind,
    TopologyResource, schedule_device_lanes,
};
use svod_device::profile::{SqttSimds, StablePstate};
use svod_device::{AmdCounter, CounterSet, CudaCounter, PmcCounter};
use svod_dtype::DeviceSpec;
use svod_ir::UOp;
use svod_ir::origin::{self, Origin, OriginFrame, OriginId, SourceLocation};
use test_case::test_case;

use crate::kernel_cache::CachedKernel;
use crate::profiler::{
    KernelProfile, OperationTiming, OriginView, PmcSelection, ProfileOptions, RunProfile, StageProfile, UNATTRIBUTED,
    aggregate_origins, analyze_execution_lanes, has_origins, parse_pmc,
};

/// Every counter of every backend round-trips through its token, and tokens are
/// unique across backends so `from_token` needs no device context.
#[test]
fn pmc_counter_token_roundtrip() {
    let all: Vec<PmcCounter> = AmdCounter::all()
        .into_iter()
        .map(PmcCounter::Amd)
        .chain(CudaCounter::all().into_iter().map(PmcCounter::Cuda))
        .collect();
    for &c in &all {
        assert_eq!(PmcCounter::from_token(c.token()), Some(c), "token roundtrip for {c:?}");
    }
    let mut tokens: Vec<&str> = all.iter().map(|c| c.token()).collect();
    tokens.sort_unstable();
    let unique = tokens.len();
    tokens.dedup();
    assert_eq!(tokens.len(), unique, "counter tokens collide across backends");

    assert_eq!(PmcCounter::from_token("nope"), None);
    assert_eq!(
        PmcCounter::from_token("BUSY"),
        Some(PmcCounter::Amd(AmdCounter::SqBusyCycles)),
        "case-insensitive alias"
    );
}

/// The CUPTI metric names carry the `.sum` rollup: `ConfigAddMetrics` rejects a
/// bare base name.
#[test]
fn cuda_counter_metrics_are_rollups() {
    for c in CudaCounter::all() {
        let metric = c.metric();
        assert!(metric.ends_with(".sum"), "{c:?} metric {metric} lacks a rollup suffix");
        assert!(metric.contains("__"), "{c:?} metric {metric} is not a PerfWorks name");
    }
}

/// `SVOD_PMC` parsing, and what each resulting selection enables. Counters are
/// off unless asked for, and all-unknown tokens fall back to the default set
/// rather than silently profiling nothing.
#[test]
fn pmc_selection_is_parsed_and_resolved() {
    let backend: Vec<PmcCounter> = AmdCounter::all().into_iter().map(PmcCounter::Amd).collect();
    assert_eq!(ProfileOptions::default().counters, PmcSelection::None);
    assert_eq!(ProfileOptions::default().iters, 1);
    assert!(ProfileOptions::default().static_analysis);
    assert_eq!(ProfileOptions::default().origin_depth, None, "rollups default to the leaf scope");

    assert_eq!(parse_pmc(""), PmcSelection::None);
    assert_eq!(parse_pmc("0"), PmcSelection::None);
    assert!(!PmcSelection::None.is_enabled());
    assert!(PmcSelection::None.resolve(&backend).is_empty());

    assert_eq!(parse_pmc("1"), PmcSelection::Default);
    assert_eq!(parse_pmc("bogus"), PmcSelection::Default, "all-unknown tokens fall back to the default set");
    assert!(PmcSelection::Default.is_enabled());
    assert_eq!(PmcSelection::Default.resolve(&backend), backend, "Default takes the backend's set");

    let valu = PmcCounter::Amd(AmdCounter::SqInstsValu);
    let waves = PmcCounter::Amd(AmdCounter::SqWaves);
    assert_eq!(parse_pmc("valu,waves"), PmcSelection::Custom(vec![valu, waves]));
    assert_eq!(PmcSelection::Custom(vec![valu]).resolve(&backend), vec![valu], "an explicit list ignores the default");

    // Tokens resolve across backends; the arming context drops what it cannot collect.
    assert_eq!(parse_pmc("dram"), PmcSelection::Custom(vec![PmcCounter::Cuda(CudaCounter::DramBytes)]));
}

/// `SVOD_ORIGIN_DEPTH` parsing. Only a positive count is a depth: zero would key
/// every rollup row on nothing and render the whole run `<unattributed>`, so it
/// is rejected like any other non-depth value.
#[test_case(None, None; "unset keeps the leaf")]
#[test_case(Some(""), None; "empty")]
#[test_case(Some("0"), None; "zero is no cut")]
#[test_case(Some("3"), Some(3); "a positive depth")]
#[test_case(Some(" 2 "), Some(2); "surrounding space")]
#[test_case(Some("x"), None; "garbage")]
fn from_env_parses_the_origin_depth(value: Option<&str>, expected: Option<usize>) {
    // `set_var` mutates the whole process; these cases are the only readers.
    static ENV: Mutex<()> = Mutex::new(());
    let _guard = ENV.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    unsafe {
        match value {
            Some(value) => std::env::set_var("SVOD_ORIGIN_DEPTH", value),
            None => std::env::remove_var("SVOD_ORIGIN_DEPTH"),
        }
    }
    let parsed = ProfileOptions::from_env().origin_depth;
    unsafe { std::env::remove_var("SVOD_ORIGIN_DEPTH") };
    assert_eq!(parsed, expected, "SVOD_ORIGIN_DEPTH={value:?}");
}

/// `SVOD_SQTT_PSTATE` picks the traced pass's pin; unset or unknown leaves the clock to the device.
#[test_case(None, None; "unset")]
#[test_case(Some("standard"), Some(StablePstate::Standard); "standard")]
#[test_case(Some("peak"), Some(StablePstate::Peak); "peak")]
#[test_case(Some("min_sclk"), None; "unknown")]
fn from_env_parses_the_sqtt_pstate(value: Option<&str>, expected: Option<StablePstate>) {
    static ENV: Mutex<()> = Mutex::new(());
    let _guard = ENV.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    unsafe {
        match value {
            Some(value) => std::env::set_var("SVOD_SQTT_PSTATE", value),
            None => std::env::remove_var("SVOD_SQTT_PSTATE"),
        }
    }
    let parsed = ProfileOptions::from_env().sqtt_pstate;
    unsafe { std::env::remove_var("SVOD_SQTT_PSTATE") };
    assert_eq!(parsed, expected, "SVOD_SQTT_PSTATE={value:?}");
}

/// `SVOD_SQTT_SIMD` picks each SE's traced SIMD; unset or unparsable keeps CU 0's two SIMDs alternating.
#[test_case(None, SqttSimds::default(); "unset")]
#[test_case(Some("3"), SqttSimds([3; 8]); "one")]
#[test_case(Some("1,0"), SqttSimds::cycle(&[1, 0]); "list")]
#[test_case(Some("x"), SqttSimds::default(); "unknown")]
fn from_env_parses_the_sqtt_simds(value: Option<&str>, expected: SqttSimds) {
    static ENV: Mutex<()> = Mutex::new(());
    let _guard = ENV.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    unsafe {
        match value {
            Some(value) => std::env::set_var("SVOD_SQTT_SIMD", value),
            None => std::env::remove_var("SVOD_SQTT_SIMD"),
        }
    }
    let parsed = ProfileOptions::from_env().sqtt_simds;
    unsafe { std::env::remove_var("SVOD_SQTT_SIMD") };
    assert_eq!(parsed, expected, "SVOD_SQTT_SIMD={value:?}");
}

#[test]
fn render_table_empty_and_host_only() {
    assert_eq!(RunProfile::default().render_table(), "", "an empty report renders nothing");

    // A host-only stage (no kernels) renders a single wall line, no metric table.
    let mut rp = RunProfile::default();
    rp.push(StageProfile::host("mel", Duration::from_millis(3)));
    let out = rp.render_table();
    assert!(out.contains("mel"), "host stage name present: {out:?}");
    assert!(out.contains("host"), "host stage tagged host: {out:?}");
    assert!(!out.contains("GFLOP/s"), "no metric columns for host-only: {out:?}");
}

#[test]
fn merge_accumulates_same_named_stages_and_appends_new() {
    let mut a = RunProfile::default();
    a.push(StageProfile::host("mel", Duration::from_millis(2)));
    let mut enc = StageProfile::host("encoder", Duration::from_millis(10));
    enc.meta.insert("rtf".into(), "0.02".into());
    a.push(enc);

    let mut b = RunProfile::default();
    let mut enc2 = StageProfile::host("encoder", Duration::from_millis(5)); // same name → sum wall + meta
    enc2.meta.insert("chunks".into(), "4".into());
    b.push(enc2);
    b.push(StageProfile::host("decode", Duration::from_millis(3))); // new name → appended

    a.merge(b);

    let names: Vec<&str> = a.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["mel", "encoder", "decode"], "matched stays in place, new appends");
    assert_eq!(a.stage("mel").unwrap().wall, Duration::from_millis(2), "untouched");

    let enc = a.stage("encoder").unwrap();
    assert_eq!(enc.wall, Duration::from_millis(15), "10 + 5 summed");
    assert_eq!(enc.meta.get("rtf").map(String::as_str), Some("0.02"), "kept");
    assert_eq!(enc.meta.get("chunks").map(String::as_str), Some("4"), "folded in");
}

/// Metadata both stages carry is summed in the wider of the two forms; a key
/// only one side holds, or a value that is not a number, is kept as it was.
#[test]
fn merge_sums_numeric_metadata_and_keeps_the_rest() {
    let stage = |pairs: &[(&str, &str)]| {
        let mut stage = StageProfile::host("decode", Duration::from_millis(1));
        stage.meta.extend(pairs.iter().map(|(key, value)| ((*key).to_string(), (*value).to_string())));
        let mut profile = RunProfile::default();
        profile.push(stage);
        profile
    };

    let mut a = stage(&[
        ("dispatches", "2"),
        ("accumulated_wall_ms", "1.500"),
        ("widened", "2"),
        ("timing_semantics", "host wall"),
        ("only_a", "7"),
    ]);
    a.merge(stage(&[
        ("dispatches", "3"),
        ("accumulated_wall_ms", "0.25"),
        ("widened", "0.5"),
        ("timing_semantics", "device wall"),
        ("only_b", "9"),
    ]));

    let meta = &a.stage("decode").unwrap().meta;
    assert_eq!(meta["dispatches"], "5", "integers stay integers");
    assert_eq!(meta["accumulated_wall_ms"], "1.750", "floats print with three decimals");
    assert_eq!(meta["widened"], "2.500", "a fractional side widens the whole sum");
    assert_eq!(meta["timing_semantics"], "device wall", "a non-numeric value is still overwritten");
    assert_eq!(meta["only_a"], "7", "a key only the left side holds is kept");
    assert_eq!(meta["only_b"], "9", "a key only the right side holds is folded in");
}

fn resource(id: u64) -> TopologyResource {
    TopologyResource { id, owner: DeviceSpec::Cpu, start: 0, end: 64 }
}

fn topology_op(operation: usize, queue: QueueKind, reads: &[u64], writes: &[u64]) -> TopologyOperation {
    TopologyOperation {
        operation,
        lane: DeviceQueue { device: DeviceSpec::Cpu, queue },
        reads: reads.iter().copied().map(resource).collect(),
        writes: writes.iter().copied().map(resource).collect(),
        kind: TopologyOperationKind::Execute,
    }
}

fn semantic_plan(operations: &[TopologyOperation]) -> SemanticLinkedPlan {
    let lanes = schedule_device_lanes(operations, QueueMergeLimits::NO_MERGE, |executor, owner| executor == owner);
    let mut signal = 0x1000;
    SemanticLinkedPlan::from_lane_submissions(lanes, |_| {
        signal += 16;
        [signal - 8, signal]
    })
    .unwrap()
}

fn timing(operation: usize, millis: u64) -> OperationTiming {
    OperationTiming { operation, copy_leg: None, duration: Duration::from_millis(millis) }
}

/// Independent compute/copy forks at t=0, joined by a third op. The compute
/// lane waits three ms after its first command for the longer copy lane, and
/// the two lanes overlap for the five ms the compute op runs.
#[test]
fn host_fork_join_lane_metrics_measure_overlap_and_join_wait() {
    let plan = semantic_plan(&[
        topology_op(0, QueueKind::Compute(0), &[], &[1]),
        topology_op(1, QueueKind::Copy(0), &[], &[2]),
        topology_op(2, QueueKind::Compute(0), &[1, 2], &[3]),
    ]);
    let metrics = analyze_execution_lanes(&plan, &[timing(0, 5), timing(1, 8), timing(2, 2)]);

    assert_eq!(metrics.makespan, Duration::from_millis(10));
    assert_eq!(metrics.busy, Duration::from_millis(15));
    assert_eq!(metrics.wait, Duration::from_millis(3));
    assert_eq!(metrics.overlap, Duration::from_millis(5));
    let compute = metrics.lanes.iter().find(|lane| lane.lane.queue == QueueKind::Compute(0)).unwrap();
    assert_eq!(compute.makespan, Duration::from_millis(10));
    assert_eq!(compute.busy, Duration::from_millis(7));
    assert_eq!(compute.wait, Duration::from_millis(3));
    assert_eq!(compute.overlap, Duration::from_millis(5));
}

/// The same three ops chained by RAW hazards instead: nothing overlaps, and
/// the makespan is the serial sum.
#[test]
fn alternating_copy_compute_metrics_preserve_serial_hazards() {
    let plan = semantic_plan(&[
        topology_op(0, QueueKind::Compute(0), &[], &[1]),
        topology_op(1, QueueKind::Copy(0), &[1], &[2]),
        topology_op(2, QueueKind::Compute(0), &[2], &[3]),
    ]);
    let metrics = analyze_execution_lanes(&plan, &[timing(0, 2), timing(1, 3), timing(2, 4)]);

    assert_eq!(metrics.makespan, Duration::from_millis(9));
    assert_eq!(metrics.busy, Duration::from_millis(9));
    assert_eq!(metrics.wait, Duration::from_millis(5));
    assert_eq!(metrics.overlap, Duration::ZERO);
    assert!(metrics.lanes.iter().all(|lane| lane.overlap == Duration::ZERO));
}

// ── origin rollups ─────────────────────────────────────────────────────────

/// Nothing dispatches here: a `KernelProfile` only needs a named program to key
/// the entry-point breakdown by.
#[derive(Debug)]
struct NamedProgram(&'static str);

impl Program for NamedProgram {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        self.0
    }
}

fn cached_kernel(name: &'static str) -> Arc<CachedKernel> {
    let unit = || [UOp::index_const(1), UOp::index_const(1), UOp::index_const(1)];
    Arc::new(CachedKernel {
        program: Box::new(NamedProgram(name)),
        device: "CPU".into(),
        code: String::new(),
        entry_point: name.into(),
        var_names: Vec::new(),
        globals: vec![0],
        outs: vec![0],
        ins: Vec::new(),
        global_size: unit(),
        local_size: Some(unit()),
    })
}

/// A dispatch of `name` taking `micros` of wall (no GPU stamps, so
/// `gpu_or_wall` is exactly that), charged to `origin` and fused from `origins`.
fn dispatch(name: &'static str, micros: u64, origin: Option<OriginId>, origins: &[OriginId]) -> KernelProfile {
    KernelProfile {
        kernel: cached_kernel(name),
        device: DeviceSpec::Cpu,
        origin,
        origins: origins.iter().copied().collect(),
        num_buffers: 1,
        wall: Duration::from_micros(micros),
        gpu_start_ns: None,
        gpu_end_ns: None,
        static_info: None,
        counters: None,
        sqtt: None,
    }
}

/// `merge_min` keeps the fastest pass's timing, but counters only ever come
/// from a counted pass — which is always the slow one, because collecting them
/// serializes the context and replays each kernel. Dropping them with the slow
/// sample would leave a counted run reporting no counters at all.
#[test]
fn merge_min_keeps_the_best_time_and_the_captured_counters() {
    let counted = |micros, value| {
        let mut k = dispatch("k", micros, None, &[]);
        k.counters = Some(CounterSet {
            values: [(PmcCounter::Cuda(CudaCounter::SmWarpsLaunched), value)].into_iter().collect(),
        });
        k
    };
    let stage =
        |k| RunProfile { stages: vec![StageProfile::gpu("profile", Duration::ZERO, vec![k])], origin_depth: None };

    // Counted pass first, then a faster disarmed one.
    let mut report = stage(counted(900, 32));
    report.merge_min(stage(dispatch("k", 10, None, &[])));
    let kept = &report.stages[0].kernels[0];
    assert_eq!(kept.wall, Duration::from_micros(10), "the disarmed pass times the kernel");
    let values = &kept.counters.as_ref().expect("counters survive the faster sample").values;
    assert_eq!(values[&PmcCounter::Cuda(CudaCounter::SmWarpsLaunched)], 32);

    // ...and in the other order, where the faster sample arrives first.
    let mut report = stage(dispatch("k", 10, None, &[]));
    report.merge_min(stage(counted(900, 32)));
    let kept = &report.stages[0].kernels[0];
    assert_eq!(kept.wall, Duration::from_micros(10));
    assert_eq!(
        kept.counters.as_ref().expect("counters are adopted from the slower sample").values
            [&PmcCounter::Cuda(CudaCounter::SmWarpsLaunched)],
        32
    );
}

/// Intern is independent of capture, so scopes can be built directly here
/// without reshaping any graph.
fn module(parent: Option<OriginId>, name: &str) -> OriginId {
    origin::intern(Origin { parent, frame: OriginFrame::Module { name: name.into() } })
}

fn call(parent: Option<OriginId>, op: &'static str) -> OriginId {
    origin::intern(Origin {
        parent,
        frame: OriginFrame::Call { op, at: SourceLocation::new("tensor/src/arithmetic.rs", 31, 9) },
    })
}

/// `encoder.layers.0.{ffn1,attn}` and `ctc_head`, the leaves being call frames
/// under the deepest module — the shape the tensor entry points mint.
struct Scopes {
    encoder: OriginId,
    ffn1: OriginId,
    ffn1_call: OriginId,
    attn_call: OriginId,
    ctc_head: OriginId,
}

fn scopes() -> Scopes {
    let encoder = module(None, "encoder");
    let layer = module(Some(encoder), "layers.0");
    let ffn1 = module(Some(layer), "ffn1");
    let attn = module(Some(layer), "attn");
    Scopes {
        encoder,
        ffn1,
        ffn1_call: call(Some(ffn1), "mul"),
        attn_call: call(Some(attn), "matmul"),
        ctc_head: module(None, "ctc_head"),
    }
}

/// Two encoder leaves plus one head kernel: 100 + 200 + 300 µs.
fn stage_kernels() -> Vec<KernelProfile> {
    let s = scopes();
    vec![
        dispatch("ffn1_gemm", 100, Some(s.ffn1_call), &[s.ffn1_call, s.ffn1]),
        dispatch("attn_gemm", 200, Some(s.attn_call), &[s.attn_call]),
        dispatch("head_gemm", 300, Some(s.ctc_head), &[s.ctc_head]),
    ]
}

/// Exclusive rows partition the dispatches at every depth, so they always sum
/// to the profiled total and count every dispatch exactly once.
#[test_case(Some(1); "root frames only")]
#[test_case(Some(2); "two frames")]
#[test_case(Some(3); "three frames")]
#[test_case(None; "leaf")]
fn exclusive_rollup_sums_to_the_total(depth: Option<usize>) {
    let kernels = stage_kernels();
    let rows = aggregate_origins(&kernels, OriginView::Exclusive, depth);
    assert_eq!(rows.iter().map(|r| r.total).sum::<Duration>(), Duration::from_micros(600), "{depth:?}");
    assert_eq!(rows.iter().map(|r| r.count).sum::<usize>(), kernels.len(), "{depth:?}");
    assert!(rows.windows(2).all(|w| (w[0].total, &w[1].path) >= (w[1].total, &w[0].path)), "sorted desc by total");
    // The secondary entry-point breakdown of a row accounts for the whole row.
    for row in &rows {
        assert_eq!(row.kernels.iter().map(|k| k.total).sum::<Duration>(), row.total, "{}", row.path);
    }
}

/// A depth cut merges the leaves under their common ancestor; the leaf view
/// keeps them apart.
#[test]
fn exclusive_rollup_keys_are_cut_to_the_depth() {
    let kernels = stage_kernels();
    let paths = |depth| {
        aggregate_origins(&kernels, OriginView::Exclusive, depth)
            .into_iter()
            .map(|r| (r.path, r.total))
            .collect::<std::collections::BTreeMap<_, _>>()
    };

    assert_eq!(
        paths(Some(1)),
        [("ctc_head".to_owned(), Duration::from_micros(300)), ("encoder".to_owned(), Duration::from_micros(300))]
            .into_iter()
            .collect(),
        "both encoder leaves merge into the root row"
    );
    assert_eq!(
        paths(None),
        [
            ("ctc_head".to_owned(), Duration::from_micros(300)),
            ("encoder.layers.0.attn".to_owned(), Duration::from_micros(200)),
            ("encoder.layers.0.ffn1".to_owned(), Duration::from_micros(100)),
        ]
        .into_iter()
        .collect(),
    );
}

/// Inclusive rows charge a dispatch to every ancestor, so a parent's total
/// contains its children's and the rows deliberately over-sum.
#[test]
fn inclusive_rollup_rolls_children_into_parents() {
    let kernels = stage_kernels();
    let rows: std::collections::BTreeMap<String, (usize, Duration)> =
        aggregate_origins(&kernels, OriginView::Inclusive, None)
            .into_iter()
            .map(|r| (r.path, (r.count, r.total)))
            .collect();

    assert_eq!(rows["encoder"], (2, Duration::from_micros(300)), "parent holds both leaves");
    assert_eq!(rows["encoder.layers.0"], (2, Duration::from_micros(300)));
    assert_eq!(rows["encoder.layers.0.ffn1"], (1, Duration::from_micros(100)));
    assert_eq!(rows["encoder.layers.0.attn"], (1, Duration::from_micros(200)));
    assert_eq!(rows["ctc_head"], (1, Duration::from_micros(300)));
    assert!(
        rows.values().map(|(_, total)| *total).sum::<Duration>() > Duration::from_micros(600),
        "inclusive rows overlap and do not sum to the total"
    );
}

/// The union, not the primary, drives the inclusive view: a kernel that fused a
/// scope it is not charged to still shows up under it.
#[test]
fn inclusive_rollup_follows_the_fused_set() {
    let s = scopes();
    let kernels = vec![dispatch("fused", 100, Some(s.ctc_head), &[s.ctc_head, s.encoder])];

    let exclusive = aggregate_origins(&kernels, OriginView::Exclusive, None);
    assert_eq!(exclusive.len(), 1);
    assert_eq!(exclusive[0].path, "ctc_head", "charged once, to the primary");

    let inclusive: Vec<String> =
        aggregate_origins(&kernels, OriginView::Inclusive, None).into_iter().map(|r| r.path).collect();
    assert_eq!(inclusive, ["ctc_head", "encoder"], "equal totals, ordered by path");
}

/// Capture off (or a scope installer that misses some code) leaves dispatches
/// unattributed rather than dropping their time from the rollup.
#[test_case(OriginView::Exclusive; "exclusive")]
#[test_case(OriginView::Inclusive; "inclusive")]
fn dispatches_without_a_scope_land_in_one_unattributed_row(view: OriginView) {
    let s = scopes();
    let kernels = vec![dispatch("bare", 100, None, &[]), dispatch("scoped", 50, Some(s.ctc_head), &[s.ctc_head])];
    let rows = aggregate_origins(&kernels, view, None);

    let bare = rows.iter().find(|r| r.path == UNATTRIBUTED).expect("unattributed row");
    assert_eq!((bare.count, bare.total), (1, Duration::from_micros(100)));
    assert_eq!(bare.kernels.len(), 1);
    assert_eq!(bare.kernels[0].name, "bare");
    assert_eq!(rows.iter().map(|r| r.total).sum::<Duration>(), Duration::from_micros(150));
}

/// Call frames are the flat `file:line` layer under a module path, never a
/// rollup level: they neither appear in a key nor consume a depth frame.
#[test]
fn call_frames_never_appear_in_rollup_keys() {
    let s = scopes();
    // A call frame in the middle of the chain, not only at the leaf.
    let under_call = module(Some(s.ffn1_call), "linear2");
    let kernels = vec![dispatch("k", 100, Some(under_call), &[under_call])];

    for depth in [Some(1), Some(2), Some(3), Some(4), Some(9), None] {
        for view in [OriginView::Exclusive, OriginView::Inclusive] {
            for row in aggregate_origins(&kernels, view, depth) {
                assert!(!row.path.contains('@'), "{view:?} {depth:?}: {}", row.path);
                assert!(!row.path.contains("arithmetic.rs"), "{view:?} {depth:?}: {}", row.path);
            }
        }
    }
    let leaf = aggregate_origins(&kernels, OriginView::Exclusive, None);
    assert_eq!(leaf[0].path, "encoder.layers.0.ffn1.linear2", "four module frames, the call dropped");
}

/// A depth past the leaf is the leaf: rollups can ask for a fixed depth without
/// special-casing shallow paths.
#[test]
fn truncation_past_the_leaf_is_stable() {
    let kernels = stage_kernels();
    let leaf = aggregate_origins(&kernels, OriginView::Exclusive, None);
    for depth in [4, 5, 64] {
        let rows = aggregate_origins(&kernels, OriginView::Exclusive, Some(depth));
        let (a, b): (Vec<_>, Vec<_>) =
            (rows.iter().map(|r| (&r.path, r.total)).collect(), leaf.iter().map(|r| (&r.path, r.total)).collect());
        assert_eq!(a, b, "depth {depth} past the deepest path");
    }
    // Depth zero keeps no frame at all, mirroring `origin::truncate(id, 0)`.
    let none = aggregate_origins(&kernels, OriginView::Exclusive, Some(0));
    assert_eq!(none.len(), 1);
    assert_eq!(none[0].path, UNATTRIBUTED);
    assert_eq!(none[0].total, Duration::from_micros(600), "and still accounts for every dispatch");
}

// ── rendering ──────────────────────────────────────────────────────────────

fn stage_without_origins() -> StageProfile {
    StageProfile::gpu(
        "encoder",
        Duration::from_millis(7),
        vec![dispatch("gemm", 300, None, &[]), dispatch("gemm", 100, None, &[]), dispatch("cast", 50, None, &[])],
    )
}

/// The name-keyed table is exactly what it was before origins existed: no
/// origin section, and every number unchanged. The fixture is the whole
/// rendering, so a stray column or row would fail here.
#[test]
fn render_without_origins_is_byte_identical() {
    let mut profile = RunProfile::default();
    profile.push(stage_without_origins());

    assert_eq!(
        profile.render_table(),
        // `mean µs` is padded to its byte length, not its display width — the
        // column has looked like this since before origins, and must not move.
        "encoder: 3 dispatches, GPU 0.450 ms\n\
         name  cnt  total ms   mean µs     %  \n\
         gemm    2     0.400     200.0  88.9  \n\
         cast    1     0.050      50.0  11.1  \n"
    );
    assert_eq!(profile.render_table(), profile.render_table_at(Some(2)), "depth is moot without origins");
    assert_eq!(profile.to_string(), profile.render_report_at(None), "Display is the leaf-depth report");
    assert!(!profile.to_string().contains("origin"), "no origin section: {}", profile);
    assert_eq!(
        profile.to_string(),
        "encoder: wall 7.0 ms, profiled exec GPU 0.450 ms\n\
         3 dispatches (0 GPU-stamped), total 0.450 ms\n\
         \x20 total ms  count    mean \u{b5}s      %  name\n\
         \x20    0.400      2      200.0   88.9  gemm\n\
         \x20    0.050      1       50.0   11.1  cast\n\n",
        "the histogram report, down to the trailing blank line the old `writeln!` left"
    );
    assert!(!has_origins(&profile.stages[0].kernels));
}

/// With origins present both renderers grow one section, in both views, and the
/// small-table detail lists the entry points under each exclusive row.
#[test]
fn render_with_origins_adds_both_views_and_kernel_detail() {
    let mut profile = RunProfile::default();
    profile.push(StageProfile::gpu("encoder", Duration::from_millis(7), stage_kernels()));
    let table = profile.render_table_at(Some(2));

    assert!(table.starts_with("encoder: 3 dispatches, GPU 0.600 ms\n"), "{table}");
    assert!(table.contains("origin rollup (depth 2, exclusive; rows sum to the total)"), "{table}");
    assert!(table.contains("origin rollup (depth 2, inclusive; parents contain children, rows overlap)"), "{table}");
    assert!(table.contains("origin path"), "{table}");
    assert!(table.contains("encoder.layers.0"), "{table}");
    assert!(table.contains("· ffn1_gemm"), "exclusive rows list their kernels: {table}");

    // The report path (GigaAM's `Display`) carries the same section.
    let report = profile.render_report_at(Some(1));
    assert!(report.contains("origin rollup (depth 1"), "{report}");
    assert!(report.contains("profiled exec GPU"), "the histogram is still there: {report}");
}

/// A profile carries the depth it was produced at, so every no-argument
/// renderer cuts the rollups without the caller repeating it — this is what
/// makes `SVOD_ORIGIN_DEPTH` reach `cargo bench`'s `render_table()`.
#[test]
fn stored_origin_depth_drives_the_default_renderers() {
    let mut profile = RunProfile { origin_depth: Some(2), ..Default::default() };
    profile.push(StageProfile::gpu("encoder", Duration::from_millis(7), stage_kernels()));

    let table = profile.render_table();
    assert_eq!(table, profile.render_table_at(Some(2)), "render_table() is the stored depth");
    assert!(table.contains("origin rollup (depth 2"), "{table}");
    assert!(table.contains("encoder.layers.0"), "{table}");
    assert!(!table.contains("encoder.layers.0.ffn1"), "the leaf rows are cut away: {table}");
    assert_eq!(profile.to_string(), profile.render_report_at(Some(2)), "Display is the stored depth");
    assert_eq!(profile.to_json(), profile.to_json_at(Some(2)), "the export too");

    // The explicit variants still override it, in both directions.
    assert!(profile.render_table_at(None).contains("encoder.layers.0.ffn1"), "leaf override");
    assert!(profile.render_table_at(Some(1)).contains("origin rollup (depth 1"));
    assert_ne!(profile.render_table_at(None), table);
}

/// Both merges carry the depth, so an accumulator seeded with
/// `RunProfile::default()` still renders at the depth its passes were made with.
#[test]
fn merges_carry_the_origin_depth() {
    let deep = || RunProfile { origin_depth: Some(3), ..Default::default() };

    let mut merged = RunProfile::default();
    merged.merge(deep());
    assert_eq!(merged.origin_depth, Some(3), "merge adopts the incoming depth");

    let mut min_merged = RunProfile::default();
    min_merged.merge_min(deep());
    assert_eq!(min_merged.origin_depth, Some(3), "merge_min too");

    let mut own = RunProfile { origin_depth: Some(1), ..Default::default() };
    own.merge(deep());
    assert_eq!(own.origin_depth, Some(1), "an explicit depth is never overwritten");
}

// ── JSON export ────────────────────────────────────────────────────────────

/// The export round-trips through `serde_json::Value`, resolves every id it
/// mentions through the embedded origin nodes, and carries both rollups.
#[test]
fn json_export_round_trips_and_embeds_the_referenced_origins() {
    let mut profile = RunProfile::default();
    profile.push(StageProfile::host("mel", Duration::from_millis(2)));
    let mut stage = StageProfile::gpu("encoder", Duration::from_millis(7), stage_kernels());
    stage.meta.insert("rtf".into(), "0.02".into());
    profile.push(stage);
    // Interned but never dispatched: it must not reach the export.
    let unreferenced = module(None, "not_dispatched");
    let s = scopes();

    let json: serde_json::Value = serde_json::from_str(&profile.to_json_at(Some(2))).expect("valid JSON");
    assert_eq!(json["origin_depth"], serde_json::json!(2));
    assert_eq!(json["stages"][0]["name"], "mel");
    assert_eq!(json["stages"][0]["dispatches"], 0);
    assert_eq!(json["stages"][1]["meta"]["rtf"], "0.02");
    assert_eq!(json["stages"][1]["dispatches"], 3);

    let kernels = json["stages"][1]["kernels"].as_array().expect("kernel rows");
    assert_eq!(kernels.len(), 3, "one row per (entry point, primary origin)");
    assert_eq!(kernels[0]["name"], "head_gemm", "sorted by total time");
    assert_eq!(kernels[0]["count"], 1);
    assert!((kernels[0]["total_ms"].as_f64().unwrap() - 0.3).abs() < 1e-9);
    assert_eq!(kernels[0]["origin"], "ctc_head");
    assert_eq!(kernels[0]["origin_id"], s.ctc_head.get(), "the raw id sits beside the rendered path");
    // The full path keeps the call frame the rollup key drops.
    let ffn1 = kernels.iter().find(|k| k["name"] == "ffn1_gemm").expect("ffn1 row");
    assert!(
        ffn1["origin"].as_str().unwrap().starts_with("encoder.layers.0.ffn1 @ mul tensor/src/arithmetic.rs:31"),
        "{ffn1}"
    );
    assert_eq!(ffn1["origins"].as_array().unwrap().len(), 2);
    assert_eq!(ffn1["origin_ids"], serde_json::json!([s.ffn1.get(), s.ffn1_call.get()]), "ids match, id-ordered");

    let exclusive = json["stages"][1]["origins_exclusive"].as_array().expect("exclusive rows");
    assert!(exclusive.iter().all(|r| r["path"] != serde_json::Value::Null));
    assert!((exclusive.iter().map(|r| r["percent"].as_f64().unwrap()).sum::<f64>() - 100.0).abs() < 1e-6);
    let inclusive = json["stages"][1]["origins_inclusive"].as_array().expect("inclusive rows");
    assert!(inclusive.len() >= exclusive.len(), "ancestors add rows");

    // The exported origins are the ancestor closure of the referenced ids — not
    // the process-global arena — and every parent link resolves inside it, so a
    // consumer rebuilds any path offline.
    let nodes = json["origins"].as_array().expect("origin nodes");
    let by_id: std::collections::HashMap<u64, &serde_json::Value> =
        nodes.iter().map(|node| (node["id"].as_u64().expect("id"), node)).collect();
    let ids: Vec<u64> = nodes.iter().map(|node| node["id"].as_u64().unwrap()).collect();
    assert!(ids.windows(2).all(|w| w[0] < w[1]), "id-ordered and deduplicated: {ids:?}");
    for node in nodes {
        if let Some(parent) = node["parent"].as_u64() {
            assert!(by_id.contains_key(&parent), "closed under parents: {node}");
        }
    }

    // Ancestors of a referenced leaf are pulled in; unreferenced scopes are not.
    assert!(ids.contains(&(s.encoder.get() as u64)), "the root of a referenced leaf: {ids:?}");
    assert!(!ids.contains(&(unreferenced.get() as u64)), "an unreferenced scope: {ids:?}");
    assert!(nodes.len() < origin::snapshot().len(), "narrower than the whole arena");

    // A row's id resolves to the frame the path was rendered from.
    let head = by_id[&(s.ctc_head.get() as u64)];
    assert_eq!(head["frame"], serde_json::json!({ "Module": { "name": "ctc_head" } }));
    assert_eq!(head["parent"], serde_json::Value::Null, "a root scope");
    let call = by_id[&(s.ffn1_call.get() as u64)];
    assert_eq!(call["parent"], s.ffn1.get(), "call frames keep their module parent");
}

/// A traced dispatch's handle as the AMD one behaves: its trace reads only once retired, and it counts
/// the trace buffers live.
struct TracedHandle {
    live: Arc<std::sync::atomic::AtomicUsize>,
    retired: Arc<std::sync::atomic::AtomicBool>,
}

impl TracedHandle {
    fn new(live: &Arc<std::sync::atomic::AtomicUsize>, retired: &Arc<std::sync::atomic::AtomicBool>) -> Arc<Self> {
        live.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Arc::new(Self { live: Arc::clone(live), retired: Arc::clone(retired) })
    }
}

impl Drop for TracedHandle {
    fn drop(&mut self) {
        self.live.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl svod_device::DispatchTimestamps for TracedHandle {
    fn timestamps_ns(&self) -> Option<(u64, u64)> {
        Some((10, 25))
    }

    fn sqtt(&self) -> Option<svod_device::profile::SqttTrace> {
        self.retired.load(std::sync::atomic::Ordering::SeqCst).then(|| svod_device::profile::SqttTrace {
            arch: svod_dtype::AmdArch::Gfx1201,
            per_se: vec![],
            code: None,
            mclk_mhz: None,
        })
    }
}

/// Tier 5's drained pass synchronizes each traced dispatch, reads its trace and drops its handle
/// before the next: one kernel's trace buffers live at a time, every trace kept. The plain pass holds
/// every handle until its one synchronize.
#[test]
fn drained_dispatches_hold_one_trace_at_a_time() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let (live, retired) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicBool::new(false)));
    let mut finalizer = crate::profiler::SubmissionProfileFinalizer::with_capacity(3);
    for name in ["a", "b", "c"] {
        retired.store(false, Ordering::SeqCst);
        let handle = TracedHandle::new(&live, &retired);
        assert_eq!(live.load(Ordering::SeqCst), 1, "{name}: an earlier trace is still live");
        let sync = || {
            retired.store(true, Ordering::SeqCst);
            Ok(())
        };
        finalizer.push_drained(dispatch(name, 1, None, &[]), Some(handle), sync).unwrap();
        assert_eq!(live.load(Ordering::SeqCst), 0, "{name}: its handle outlived the drain");
    }
    let profiles = finalizer.finish(|| panic!("nothing is left to synchronize")).unwrap();
    assert!(profiles.iter().all(|p| p.sqtt.is_some() && (p.gpu_start_ns, p.gpu_end_ns) == (Some(10), Some(25))));

    let mut held = crate::profiler::SubmissionProfileFinalizer::with_capacity(3);
    retired.store(false, Ordering::SeqCst);
    for name in ["a", "b", "c"] {
        held.push(dispatch(name, 1, None, &[]), Some(TracedHandle::new(&live, &retired)));
    }
    assert_eq!(live.load(Ordering::SeqCst), 3);
    let profiles = held
        .finish(|| {
            retired.store(true, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
    assert!(profiles.iter().all(|p| p.sqtt.is_some()));
    assert_eq!(live.load(Ordering::SeqCst), 0);
}
