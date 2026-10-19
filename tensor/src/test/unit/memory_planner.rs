use super::*;
use crate::schedule::ScheduleItem;
use std::collections::HashSet;
use std::sync::Arc;
use svod_device::Buffer;
use svod_ir::UOp;
use test_case::test_case;

fn make_buffer(numel: usize) -> Buffer {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    Buffer::new(alloc, DType::Float32, vec![numel], Default::default())
}

/// A SINK schedule item. `id` seeds both the buffer-uop id AND the kernel ast
/// (`sink([const(id)])`) so each item gets a DISTINCT hash-consed `kernel.id`
/// — required for [`chain_deps`] to produce distinct execution levels.
fn make_sink_item(id: u64, buffer: Buffer) -> ScheduleItem {
    let ast = UOp::sink(vec![UOp::native_const(id as f32)]);
    ScheduleItem {
        kernel: ast.clone(),
        ast,
        buffers: vec![buffer],
        buffer_uop_ids: vec![id],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        instance_dependencies: Vec::new(),
        loop_var_names: HashSet::new(),
    }
}

fn make_nonsink_item(id: u64, buffer: Buffer) -> ScheduleItem {
    let ast = UOp::native_const(id as f32);
    ScheduleItem {
        kernel: ast.clone(),
        ast,
        buffers: vec![buffer],
        buffer_uop_ids: vec![id],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        instance_dependencies: Vec::new(),
        loop_var_names: HashSet::new(),
    }
}

fn make_store_item(buffer_uop: &Arc<UOp>, buffer: Buffer, index: Arc<UOp>, gate: Option<Arc<UOp>>) -> ScheduleItem {
    let value = UOp::native_const(1.0f32);
    let store = match gate {
        Some(gate) => index.store_gated(value, gate),
        None => index.store(value),
    };
    let ast = UOp::sink(vec![store]);
    ScheduleItem {
        kernel: ast.clone(),
        ast,
        buffers: vec![buffer],
        buffer_uop_ids: vec![buffer_uop.id],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        instance_dependencies: Vec::new(),
        loop_var_names: HashSet::new(),
    }
}

/// Make item `i` depend on item `i-1`, so `compute_item_levels` assigns each a
/// distinct level `0, 1, 2, …`. Buffers at distinct levels become reuse-eligible.
fn chain_deps(items: &mut [ScheduleItem]) {
    for i in 1..items.len() {
        let prev = items[i - 1].kernel.id;
        items[i].dependencies = vec![prev];
    }
}

/// Run the planner, computing item levels the same way `prepare_execution_plan` does.
fn plan(schedule: &Schedule, output_ids: &HashSet<u64>, mode: PlannerMode) -> MemoryPlannerResult {
    let levels = compute_item_levels(schedule).expect("compute_item_levels");
    memory_planner(schedule, &levels, output_ids, mode)
}

#[test]
fn test_round_up() {
    assert_eq!(round_up(100, 0x1000), 0x1000);
    assert_eq!(round_up(0x1000, 0x1000), 0x1000);
    assert_eq!(round_up(0x1001, 0x1000), 0x2000);
    assert_eq!(round_up(0, 0x1000), 0);
}

#[test]
fn test_round_up_256_block() {
    // 256-byte alignment.
    assert_eq!(round_up(1, 256), 256);
    assert_eq!(round_up(256, 256), 256);
    assert_eq!(round_up(257, 256), 512);
    assert_eq!(round_up(0, 256), 0);
}

#[test]
fn test_parse_mode_default_is_arena() {
    // Env unset (`NO_MEMORY_PLANNER=0`) → arena planner runs.
    assert_eq!(parse_mode(None), PlannerMode::Arena);
    assert_eq!(parse_mode(Some("")), PlannerMode::Arena);
}

#[test]
fn test_prepare_config_planner_mode_follows_env() {
    // `default()` and `From<OptimizerConfig>` hardcoded Arena, so the
    // `SVOD_MEMORY_PLANNER` escape hatch only worked on `from_env()`.
    let expected = parse_mode(std::env::var("SVOD_MEMORY_PLANNER").ok().as_deref());
    for (name, config) in [
        ("default", crate::PrepareConfig::default()),
        ("from_env", crate::PrepareConfig::from_env()),
        ("for_cpu_backend", crate::PrepareConfig::for_cpu_backend(crate::CpuBackend::Clang)),
        ("from_optimizer", crate::PrepareConfig::from(svod_schedule::OptimizerConfig::default())),
    ] {
        assert_eq!(config.planner_mode, expected, "{name}");
    }
}

#[test]
fn test_parse_mode_disabled_aliases() {
    for raw in ["0", "off", "none", "disabled", "OFF", " disabled ", "Disabled"] {
        assert_eq!(parse_mode(Some(raw)), PlannerMode::Disabled, "raw={raw:?}");
    }
}

#[test]
fn test_parse_mode_remap_aliases() {
    // `remap` / `pool` opt into the older liveness-based pool reuse.
    for raw in ["remap", "pool", "POOL", "Remap", " remap "] {
        assert_eq!(parse_mode(Some(raw)), PlannerMode::Remap, "raw={raw:?}");
    }
}

#[test]
fn test_parse_mode_arena_aliases() {
    for raw in ["arena", "ARENA", " arena ", "1", "on"] {
        assert_eq!(parse_mode(Some(raw)), PlannerMode::Arena, "raw={raw:?}");
    }
}

#[test]
fn test_parse_mode_unknown_falls_back_to_arena() {
    // Unknown values must not crash — default to the arena mode rather than
    // silently regressing to a different strategy.
    assert_eq!(parse_mode(Some("garbage")), PlannerMode::Arena);
}

// ============================================================================
// compute_item_levels
// ============================================================================

#[test]
fn test_compute_item_levels_longest_path() {
    // Diamond DAG: A→B, A→C, B→D, C→D ⇒ levels A=0, B=1, C=1, D=2.
    let mut items: Vec<ScheduleItem> = (0..4).map(|i| make_sink_item(i, make_buffer(256))).collect();
    let (a, b, c) = (items[0].kernel.id, items[1].kernel.id, items[2].kernel.id);
    items[1].dependencies = vec![a];
    items[2].dependencies = vec![a];
    items[3].dependencies = vec![b, c];

    assert_eq!(compute_item_levels(&items).expect("levels"), vec![0, 1, 1, 2]);
}

#[test]
fn test_compute_item_levels_chain() {
    let mut items: Vec<ScheduleItem> = (0..4).map(|i| make_sink_item(i, make_buffer(256))).collect();
    chain_deps(&mut items);
    assert_eq!(compute_item_levels(&items).expect("levels"), vec![0, 1, 2, 3]);
}

/// A schedule loop chains its trips by item index, not callable id: the
/// positional edge alone must push the second trip into a later level.
#[test]
fn test_compute_item_levels_honours_instance_dependencies() {
    let mut items: Vec<ScheduleItem> = (0..2).map(|i| make_sink_item(i, make_buffer(256))).collect();
    assert_eq!(compute_item_levels(&items).expect("levels"), vec![0, 0]);

    items[1].instance_dependencies = vec![0];
    assert_eq!(compute_item_levels(&items).expect("levels"), vec![0, 1]);
}

#[test]
fn test_compute_item_levels_errors_on_cycle() {
    // Two items depending on each other → no in-degree-0 node → cycle. The
    // shared leveling fn returns Err (was a release-stripped debug_assert).
    let mut items: Vec<ScheduleItem> = (0..2).map(|i| make_sink_item(i, make_buffer(256))).collect();
    let (a, b) = (items[0].kernel.id, items[1].kernel.id);
    items[0].dependencies = vec![b];
    items[1].dependencies = vec![a];

    let err = compute_item_levels(&items).expect_err("cyclic schedule must error");
    assert!(matches!(err.kind(), crate::error::ErrorKind::Execution { .. }), "unexpected error: {err:?}");
}

#[test]
fn test_compute_item_levels_errors_on_unresolved_dep() {
    // A dep id absent from the schedule. The old planner silently skipped it;
    // the unified fn errors loudly, matching the runtime executor.
    let mut items: Vec<ScheduleItem> = vec![make_sink_item(1, make_buffer(256))];
    items[0].dependencies = vec![999];

    let err = compute_item_levels(&items).expect_err("unresolved dep must error");
    assert!(matches!(err.kind(), crate::error::ErrorKind::Execution { .. }), "unexpected error: {err:?}");
}

#[test]
fn test_buffer_pool_key_equality() {
    let key1 = BufferPoolKey { device: DeviceSpec::Cpu, dtype: DType::Float32, size: 0x1000 };
    let key2 = BufferPoolKey { device: DeviceSpec::Cpu, dtype: DType::Float32, size: 0x1000 };
    let key3 = BufferPoolKey { device: DeviceSpec::Cpu, dtype: DType::Float32, size: 0x2000 };

    assert_eq!(key1, key2);
    assert_ne!(key1, key3);
}

#[test]
fn test_buffer_pool_key_is_shape_agnostic() {
    // Two non-output buffers with the same (device, dtype, rounded_size) share
    // the same pool regardless of logical shape — codegen reads shape from the
    // UOp graph, not from the Buffer.
    let b_flat = make_buffer(256);
    let b_2d = svod_device::Buffer::new(
        svod_device::registry::cpu().expect("cpu"),
        DType::Float32,
        vec![16, 16],
        Default::default(),
    );

    let key_flat = BufferPoolKey {
        device: b_flat.allocator().device_spec(),
        dtype: b_flat.dtype(),
        size: round_up(b_flat.size(), 0x1000),
    };
    let key_2d = BufferPoolKey {
        device: b_2d.allocator().device_spec(),
        dtype: b_2d.dtype(),
        size: round_up(b_2d.size(), 0x1000),
    };

    assert_eq!(key_flat, key_2d, "shape-only differences must not split pools");
}

#[test]
fn test_event_timeline_ordering() {
    let mut liveness: HashMap<u64, BufferLiveness> = HashMap::new();
    liveness.insert(
        1,
        BufferLiveness {
            first_level: 0,
            last_level: 1,
            pool_key: BufferPoolKey { device: DeviceSpec::Cpu, dtype: DType::Float32, size: 0x1000 },
            prototype: make_buffer(256),
        },
    );
    liveness.insert(
        2,
        BufferLiveness {
            first_level: 2,
            last_level: 3,
            pool_key: BufferPoolKey { device: DeviceSpec::Cpu, dtype: DType::Float32, size: 0x1000 },
            prototype: make_buffer(256),
        },
    );

    let events = build_event_timeline(&liveness);

    // Sorted by (timestep, is_alloc): free (false) before alloc (true) at the same level.
    assert_eq!(events.len(), 4);
    let mut prev_key = (0usize, false);
    for event in &events {
        let key = (event.timestep, event.is_alloc);
        assert!(key >= prev_key, "events not sorted: {key:?} should come after {prev_key:?}");
        prev_key = key;
    }
}

#[test]
fn test_empty_schedule() {
    let schedule = vec![];
    let result = plan(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert!(result.buffer_replace.is_empty());
    assert_eq!(result.memory_saved, 0);
    assert_eq!(result.buffers_reused, 0);
}

#[test]
fn test_memory_planner_disabled_short_circuits() {
    // Would reuse under Remap (b1 at a later level than b0). Disabled emits nothing.
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);

    let mut schedule = vec![make_sink_item(10, b0), make_sink_item(11, b1)];
    chain_deps(&mut schedule);
    let result = plan(&schedule, &HashSet::new(), PlannerMode::Disabled);

    assert!(result.buffer_replace.is_empty());
    assert_eq!(result.buffers_reused, 0);
    assert_eq!(result.memory_saved, 0);
}

#[test]
fn test_memory_planner_reuses_across_levels() {
    // b0 at level 0, b1 at level 1 → b1 may reuse b0's storage (cross-level).
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);

    let mut schedule = vec![make_sink_item(10, b0.clone()), make_sink_item(11, b1.clone())];
    chain_deps(&mut schedule);
    let result = plan(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 1);
    let key = BufferKey { kernel_idx: 1, buffer_idx: 0 };
    let replacement = result.buffer_replace.get(&key).expect("second buffer should be remapped");
    assert_eq!(replacement.id(), b0.id());
}

#[test]
fn test_memory_planner_no_reuse_within_a_level() {
    // Both buffers at level 0 (no deps) are simultaneously live → no reuse.
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);

    let schedule = vec![make_sink_item(10, b0), make_sink_item(11, b1)];
    let result = plan(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 0, "same-level buffers must not share storage");
}

#[test]
fn test_memory_planner_reuses_unmasked_store_outputs() {
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);
    let target = UOp::new_buffer(DeviceSpec::Cpu, 256, DType::Float32);
    let index = UOp::index().buffer(target.clone()).indices(vec![UOp::index_const(0)]).call().unwrap();

    let mut schedule = vec![make_store_item(&target, b0.clone(), index, None), make_sink_item(61, b1)];
    chain_deps(&mut schedule);
    let result = plan(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 1);
    let key = BufferKey { kernel_idx: 1, buffer_idx: 0 };
    let replacement = result.buffer_replace.get(&key).expect("second buffer should be remapped");
    assert_eq!(replacement.id(), b0.id());
}

#[test_case(false; "bare_index")]
#[test_case(true; "index_behind_a_cast")]
fn test_memory_planner_skips_gated_store_outputs(wrap_index: bool) {
    // b1 is at a later level and would reuse b0 — but a gated store writes only
    // part of b0, so arena mode must not pack a later tenant over it.
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);
    let target = UOp::new_buffer(DeviceSpec::Cpu, 256, DType::Float32);
    let index = UOp::index().buffer(target.clone()).indices(vec![UOp::index_const(0)]).call().unwrap();
    let index = if wrap_index { index.cast(DType::Index) } else { index };

    let mut schedule = vec![make_store_item(&target, b0, index, Some(UOp::native_const(true))), make_sink_item(62, b1)];
    chain_deps(&mut schedule);
    let result = plan(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 0);
    assert!(result.buffer_replace.is_empty());
    assert_eq!(result.metrics.exclusions[&PlannerExclusionReason::GatedStore].allocations, 1);
}

#[test]
fn test_memory_planner_skips_non_sink_noopt_buffers() {
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);

    let mut schedule = vec![make_nonsink_item(20, b0), make_sink_item(21, b1)];
    chain_deps(&mut schedule);
    let result = plan(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 0);
    assert!(result.buffer_replace.is_empty());
}

#[test]
fn test_memory_planner_skips_buffers_with_views() {
    let b0 = make_buffer(256);
    let b0_view = b0.view(4, b0.size() - 4).unwrap();
    let b1 = make_buffer(256);

    let mut schedule = vec![make_sink_item(30, b0), make_sink_item(31, b0_view), make_sink_item(32, b1)];
    chain_deps(&mut schedule);
    let result = plan(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 0);
    assert!(result.buffer_replace.is_empty());
}

#[test]
fn test_memory_planner_skips_lone_nonzero_offset_view() {
    let b0 = make_buffer(256);
    let b0_view = b0.view(4, b0.size() - 4).unwrap();
    let b1 = make_buffer(255);

    let mut schedule = vec![make_sink_item(35, b0_view), make_sink_item(36, b1)];
    chain_deps(&mut schedule);
    let result = plan(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 0);
    assert!(result.buffer_replace.is_empty());
}

#[test]
fn test_memory_planner_skips_buffers_with_shape_aliases() {
    let b0 = make_buffer(256);
    let b0_alias = b0.view(0, b0.size() - 4).unwrap();
    let b1 = make_buffer(256);

    let mut schedule = vec![make_sink_item(40, b0), make_sink_item(41, b0_alias), make_sink_item(42, b1)];
    chain_deps(&mut schedule);
    let result = plan(&schedule, &HashSet::new(), PlannerMode::Remap);

    assert_eq!(result.buffers_reused, 0);
    assert!(result.buffer_replace.is_empty());
}

// ============================================================================
// Arena planner (PlannerMode::Arena) tests
// ============================================================================

#[test]
fn test_arena_packs_disjoint_levels_into_one_arena() {
    // Three buffers at distinct levels collapse into views over one arena.
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);
    let b2 = make_buffer(256);

    let mut schedule =
        vec![make_sink_item(50, b0.clone()), make_sink_item(51, b1.clone()), make_sink_item(52, b2.clone())];
    chain_deps(&mut schedule);
    let result = plan(&schedule, &HashSet::new(), PlannerMode::Arena);

    assert_eq!(result.buffer_replace.len(), 3, "every plannable buffer must get an arena view");

    let storage_ids: std::collections::HashSet<_> = result.buffer_replace.values().map(|b| b.storage_id().0).collect();
    assert_eq!(storage_ids.len(), 1, "all three views must share one underlying arena allocation");

    let handle_ids: std::collections::HashSet<_> = result.buffer_replace.values().map(|b| b.id().0).collect();
    assert_eq!(handle_ids.len(), 3, "each view must carry a distinct handle id");
}

/// An arena holds one class of memory: device-local intermediates pack into
/// a device-local arena, host-visible ones into another, never one shared.
#[test]
fn test_arena_keeps_each_buffer_class() {
    let alloc = || svod_device::registry::cpu().expect("cpu allocator");
    let local = svod_device::BufferSpec { cpu_access: false, ..Default::default() };
    let l0 = Buffer::new(alloc(), DType::Float32, vec![256], local);
    let l1 = Buffer::new(alloc(), DType::Float32, vec![256], local);
    let h0 = make_buffer(256);

    let mut schedule =
        vec![make_sink_item(55, l0.clone()), make_sink_item(56, h0.clone()), make_sink_item(57, l1.clone())];
    chain_deps(&mut schedule);
    let result = plan(&schedule, &HashSet::new(), PlannerMode::Arena);

    let view = |item: usize| &result.buffer_replace[&BufferKey { kernel_idx: item, buffer_idx: 0 }];
    assert_eq!(view(0).spec(), local);
    assert_eq!(view(2).spec(), local);
    assert_eq!(view(1).spec(), svod_device::BufferSpec::default());
    assert_eq!(view(0).storage_id(), view(2).storage_id(), "device-local buffers share their arena");
    assert_ne!(view(0).storage_id(), view(1).storage_id(), "the classes never share one");
}

#[test]
fn test_arena_excludes_output_buffers() {
    // Output buffers must never get rewritten — the runtime returns them
    // directly to the caller.
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);

    let mut schedule = vec![make_sink_item(60, b0.clone()), make_sink_item(61, b1.clone())];
    chain_deps(&mut schedule);
    let mut output_ids = HashSet::new();
    output_ids.insert(b0.id().0);

    let result = plan(&schedule, &output_ids, PlannerMode::Arena);

    let key0 = BufferKey { kernel_idx: 0, buffer_idx: 0 };
    assert!(!result.buffer_replace.contains_key(&key0), "output buffer must not be rewritten into arena view");
}

#[test]
fn test_arena_mode_dispatches_to_arena_planner_not_remap() {
    // Construct a workload Remap would *not* reuse (different sizes → different
    // pool keys) but Arena *can* pack.
    let b0 = make_buffer(256);
    let b1 = make_buffer(512);

    let mut schedule = vec![make_sink_item(70, b0), make_sink_item(71, b1)];
    chain_deps(&mut schedule);
    let remap = plan(&schedule, &HashSet::new(), PlannerMode::Remap);
    let arena = plan(&schedule, &HashSet::new(), PlannerMode::Arena);

    assert!(remap.buffer_replace.is_empty(), "Remap can't pack different-size buffers");
    assert_eq!(arena.buffer_replace.len(), 2, "Arena packs even when sizes differ");
}

#[test]
fn test_arena_disabled_mode_short_circuits_unchanged() {
    let b0 = make_buffer(256);
    let b1 = make_buffer(256);

    let mut schedule = vec![make_sink_item(80, b0), make_sink_item(81, b1)];
    chain_deps(&mut schedule);
    let result = plan(&schedule, &HashSet::new(), PlannerMode::Disabled);

    assert!(result.buffer_replace.is_empty());
    assert_eq!(result.buffers_reused, 0);
}

#[test]
fn test_arena_storage_sharers_have_disjoint_levels() {
    // Core safety invariant: any two buffers that end up sharing arena storage
    // (same storage_id + overlapping byte range) must have STRICTLY DISJOINT
    // execution-level intervals — otherwise they could be live at the same
    // parallel level and race (the runtime's conflict check is blind to
    // arena-view aliasing). Replaces the old reuse-dependency regression test.
    let sizes: &[usize] = &[1024, 320, 256, 512, 64, 800, 128, 256];
    let buffers: Vec<Buffer> = sizes.iter().map(|&n| make_buffer(n)).collect();
    let mut schedule: Vec<ScheduleItem> =
        buffers.iter().enumerate().map(|(i, b)| make_sink_item(i as u64, b.clone())).collect();
    // Re-reference buffer[0] at a later item so its level interval SPANS the
    // chain — it must then never share storage with anything.
    schedule.push(make_sink_item(999, buffers[0].clone()));
    chain_deps(&mut schedule);

    let levels = compute_item_levels(&schedule).expect("levels");
    let result = memory_planner(&schedule, &levels, &HashSet::new(), PlannerMode::Arena);

    // Per-buffer level interval = [min, max] level of items referencing it.
    let mut interval: HashMap<u64, (usize, usize)> = HashMap::new();
    for (step, item) in schedule.iter().enumerate() {
        for buf in &item.buffers {
            let e = interval.entry(buf.id().0).or_insert((levels[step], levels[step]));
            e.0 = e.0.min(levels[step]);
            e.1 = e.1.max(levels[step]);
        }
    }

    let views: Vec<(BufferKey, Buffer)> = result.buffer_replace.iter().map(|(k, v)| (*k, v.clone())).collect();
    for i in 0..views.len() {
        for j in (i + 1)..views.len() {
            let (ki, vi) = &views[i];
            let (kj, vj) = &views[j];
            if vi.storage_id() != vj.storage_id() {
                continue;
            }
            let bi = schedule[ki.kernel_idx].buffers[ki.buffer_idx].id().0;
            let bj = schedule[kj.kernel_idx].buffers[kj.buffer_idx].id().0;
            if bi == bj {
                continue; // same logical buffer, same storage — expected
            }
            let (oi, ei) = (vi.offset(), vi.offset() + vi.size());
            let (oj, ej) = (vj.offset(), vj.offset() + vj.size());
            if oi >= ej || oj >= ei {
                continue; // disjoint byte ranges — fine
            }
            let (ai0, ai1) = interval[&bi];
            let (aj0, aj1) = interval[&bj];
            assert!(
                ai1 < aj0 || aj1 < ai0,
                "buffers sharing arena storage [{oi},{ei}) & [{oj},{ej}) have overlapping level intervals \
                 [{ai0},{ai1}] vs [{aj0},{aj1}]"
            );
        }
    }
}

#[test]
fn test_planner_metrics_report_padding_commitment_and_reuse() {
    // Each logical buffer is 400 bytes and rounds to 512; the two live at
    // distinct levels, so both planners reuse one allocation for both.
    let mut schedule = vec![make_sink_item(100, make_buffer(100)), make_sink_item(101, make_buffer(100))];
    chain_deps(&mut schedule);

    // Arena commits one rounded 512-byte block for 800 logical / 1024 rounded bytes.
    let arena = plan(&schedule, &HashSet::new(), PlannerMode::Arena);
    assert!(arena.memory_saved > 0, "packing two cross-level buffers must report savings");
    let arena = arena.metrics;
    assert_eq!((arena.logical_allocations, arena.logical_bytes, arena.rounded_bytes), (2, 800, 1024));
    assert_eq!((arena.padding_bytes, arena.logical_peak_bytes), (224, 400));
    assert_eq!((arena.arena_committed_bytes, arena.physical_bytes, arena.fragmentation_bytes), (512, 512, 112));
    assert_eq!((arena.reused_allocations, arena.reused_bytes), (1, 288));

    // Remap hands back the un-rounded buffer, so physical == one logical allocation.
    let remap = plan(&schedule, &HashSet::new(), PlannerMode::Remap).metrics;
    assert_eq!((remap.logical_bytes, remap.rounded_bytes, remap.physical_bytes), (800, 1024, 400));
    assert_eq!((remap.reused_allocations, remap.reused_bytes), (1, 400));
}

#[test]
fn test_disabled_mode_measures_baseline_and_exclusion_reasons() {
    let eligible = make_buffer(64);
    let output = make_buffer(32);
    let transfer_owned = make_buffer(16);
    let mut schedule = vec![
        make_sink_item(110, eligible),
        make_sink_item(111, output.clone()),
        make_nonsink_item(112, transfer_owned),
    ];
    chain_deps(&mut schedule);
    let result = plan(&schedule, &HashSet::from([output.id().0]), PlannerMode::Disabled);

    assert_eq!(result.metrics.mode, PlannerMode::Disabled);
    assert_eq!(result.metrics.logical_allocations, 1);
    assert_eq!(result.metrics.logical_bytes, 256);
    assert_eq!(result.metrics.physical_bytes, 256);
    assert_eq!(result.metrics.reused_allocations, 0);
    assert_eq!(result.metrics.exclusions[&PlannerExclusionReason::Output].allocations, 1);
    assert_eq!(result.metrics.exclusions[&PlannerExclusionReason::Output].bytes, 128);
    assert_eq!(result.metrics.exclusions[&PlannerExclusionReason::NonSinkOperation].allocations, 1);
    assert_eq!(result.metrics.exclusions[&PlannerExclusionReason::NonSinkOperation].bytes, 64);
}

#[test]
fn test_fork_join_same_level_buffers_never_alias_overlapping_arena_bytes() {
    let mut schedule: Vec<_> = (0..4).map(|i| make_sink_item(120 + i, make_buffer(256))).collect();
    let root = schedule[0].kernel.id;
    let (left, right) = (schedule[1].kernel.id, schedule[2].kernel.id);
    schedule[1].dependencies = vec![root];
    schedule[2].dependencies = vec![root];
    schedule[3].dependencies = vec![left, right];
    assert_eq!(compute_item_levels(&schedule).unwrap(), [0, 1, 1, 2]);

    let result = plan(&schedule, &HashSet::new(), PlannerMode::Arena);
    let left = &result.buffer_replace[&BufferKey { kernel_idx: 1, buffer_idx: 0 }];
    let right = &result.buffer_replace[&BufferKey { kernel_idx: 2, buffer_idx: 0 }];
    assert_eq!(left.storage_id(), right.storage_id());
    assert!(
        left.offset() + left.size() <= right.offset() || right.offset() + right.size() <= left.offset(),
        "same-level fork allocations overlap: left=[{},{}), right=[{},{})",
        left.offset(),
        left.offset() + left.size(),
        right.offset(),
        right.offset() + right.size(),
    );
    assert_eq!(result.metrics.logical_peak_bytes, left.size() + right.size());
}

#[test]
fn test_real_numeric_result_matches_with_planner_disabled_and_arena() {
    fn run(mode: PlannerMode) -> Vec<f32> {
        let a = crate::Tensor::from_slice([1.0f32, -2.0, 3.5, 4.0]);
        let b = crate::Tensor::from_slice([0.5f32, 3.0, -1.5, 2.0]);
        let first = (&a + &b).unwrap();
        let second = (&first * &a).unwrap();
        let output = (&second + &b).unwrap();
        let config = crate::PrepareConfig { planner_mode: mode, disable_schedule_cache: true, ..Default::default() };
        output.realize_with(&config).expect("realize numeric planner differential");
        output.as_vec::<f32>().expect("numeric output")
    }

    let disabled = run(PlannerMode::Disabled);
    let arena = run(PlannerMode::Arena);
    assert_eq!(arena, disabled);
    assert_eq!(arena, [2.0, 1.0, 5.5, 26.0]);
}
