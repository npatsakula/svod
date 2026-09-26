use crate::device::{AbiParamDescriptor, AbiParamKind};
use crate::hcq::{
    ClikeKernargLayout, Command, CommandBufferCache, CommandField, CopyLeg, CpuQueueExecutor, DeviceQueue,
    LaneSubmission, LaneWait, LinkPatchValues, LoweredCommandBuffer, NullHcq, PatchEncoding, PatchSite, PatchSource,
    PatchTable, PlaceholderKind, PlaceholderPacking, PlaceholderRequest, QueueKind, QueueMergeLimits,
    RuntimePatchValues, SemanticLinkedPlan, SemanticLinkedSubmission, StoreScope, Submission, SubmissionExecutionError,
    SystemField, SystemPatchValues, TopologyOperation, TopologyOperationKind, TopologyResource, schedule_device_lanes,
};
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::ops;

fn storage(slot: usize) -> AbiParamDescriptor {
    AbiParamDescriptor { slot, kind: AbiParamKind::Storage(AddrSpace::Global), dtype: DType::Float32, name: None }
}

fn scalar(slot: usize, name: &str) -> AbiParamDescriptor {
    AbiParamDescriptor { slot, kind: AbiParamKind::Scalar, dtype: DType::Int32, name: Some(name.into()) }
}

/// Buffers and scalars share one dense, slot-ordered kernarg record: pointers
/// are 8-byte aligned and scalars keep their natural width, whether the ABI's
/// slots are contiguous or sparse.
#[test]
fn clike_kernargs_pack_in_slot_order_at_natural_alignment() {
    let mut dst = [0xcc; 32];
    let dense = ClikeKernargLayout::from_abi(&[storage(0), storage(1), scalar(2, "low"), scalar(3, "high")]);
    let written = dense.pack(&mut dst, &[0x1122_3344_5566_7788, 0x99aa_bbcc_ddee_ff00], &[-2, 0x1234_5678]).unwrap();
    assert_eq!(written, 24);
    assert_eq!(&dst[0..8], &0x1122_3344_5566_7788u64.to_le_bytes());
    assert_eq!(&dst[8..16], &0x99aa_bbcc_ddee_ff00u64.to_le_bytes());
    assert_eq!(&dst[16..20], &(-2i32).to_le_bytes());
    assert_eq!(&dst[20..24], &0x1234_5678i32.to_le_bytes());
    assert_eq!(&dst[24..], &[0xcc; 8], "packing never writes past the record");

    // Sparse slots keep their relative order and the pointer that follows a
    // scalar is realigned rather than packed tight.
    let mut dst = [0u8; 24];
    ClikeKernargLayout::from_abi(&[storage(0), scalar(1, "n"), storage(5)])
        .pack(&mut dst, &[0x1000, 0x5000], &[7])
        .unwrap();
    assert_eq!(&dst[..8], &0x1000u64.to_le_bytes());
    assert_eq!(&dst[8..12], &7i32.to_le_bytes());
    assert_eq!(&dst[16..24], &0x5000u64.to_le_bytes());
}

#[test]
fn program_kernargs_interleave_storage_and_scalars_by_slot() {
    let slotted = |name: &str, slot: usize| {
        let var = svod_ir::UOp::variable(name.into(), 0, 16, DType::Int32);
        let svod_ir::Op::Param(ops::Param { shape, arg }) = var.op() else { panic!("variable PARAM") };
        let mut arg = arg.clone();
        arg.slot = slot;
        svod_ir::UOp::new(svod_ir::Op::Param(ops::Param { shape: shape.clone(), arg }), DType::Int32)
    };
    let mut info = svod_ir::ProgramInfo { globals: vec![0, 2], ..Default::default() };
    info.vars = vec![slotted("low", 1), slotted("high", 3)];
    let abi = vec![
        storage(0),
        AbiParamDescriptor::from_param(&info.vars[0]).unwrap(),
        storage(2),
        AbiParamDescriptor::from_param(&info.vars[1]).unwrap(),
    ];

    let mut dst = [0xcc; 32];
    let written = ClikeKernargLayout::pack_program(&info, &abi, &mut dst, &[0x1000, 0x3000], &[7, -3]).unwrap();
    assert_eq!(written, 28);
    assert_eq!(&dst[0..8], &0x1000u64.to_le_bytes());
    assert_eq!(&dst[8..12], &7i32.to_le_bytes());
    assert_eq!(&dst[12..16], &[0; 4], "the next pointer is realigned, not packed tight");
    assert_eq!(&dst[16..24], &0x3000u64.to_le_bytes());
    assert_eq!(&dst[24..28], &(-3i32).to_le_bytes());

    // Sparse global slots address buffers by compact ordinal, so the caller must
    // supply exactly one address per global.
    let sparse = svod_ir::ProgramInfo { globals: vec![0, 5], ..Default::default() };
    let abi = sparse.globals.iter().map(|&slot| storage(slot)).collect::<Vec<_>>();
    let mut dst = [0u8; 16];
    ClikeKernargLayout::pack_program(&sparse, &abi, &mut dst, &[0x1111, 0x5555], &[]).unwrap();
    assert_eq!(&dst[..8], &0x1111u64.to_le_bytes());
    assert_eq!(&dst[8..], &0x5555u64.to_le_bytes());
    let err = ClikeKernargLayout::pack_program(&sparse, &abi, &mut dst, &[0x1111], &[])
        .expect_err("compact buffer arity must be exact");
    assert!(matches!(err, crate::Error::ProgramAbiMismatch { .. }), "{err:?}");
}

/// One packing rule for every kernarg site. Tinygrad bump-allocates its kernarg
/// blocks at alignment 8 (`runtime/support/hcq.py:352`); 16 covers the largest
/// AMDHSA member alignment without the 128-byte inflation morok used to apply.
#[test_case::test_case(&[8, 12, 4] => (vec![0, 16, 32], 36); "records are aligned, the total is not padded")]
#[test_case::test_case(&[16, 16] => (vec![0, 16], 32); "already-aligned records pack tight")]
#[test_case::test_case(&[] => (vec![], 0); "no records")]
fn kernarg_offsets_pack_records_at_one_alignment(sizes: &[usize]) -> (Vec<usize>, usize) {
    crate::hcq::kernarg_offsets(sizes.iter().copied(), 16)
}

#[test]
fn placeholder_packing_aliases_scratch_and_aligns_kernargs() {
    let packing = PlaceholderPacking::pack(&[
        PlaceholderRequest { kind: PlaceholderKind::Scratch, bytes: 64 },
        PlaceholderRequest { kind: PlaceholderKind::Kernargs, bytes: 20 },
        PlaceholderRequest { kind: PlaceholderKind::Scratch, bytes: 256 },
        PlaceholderRequest { kind: PlaceholderKind::Kernargs, bytes: 12 },
    ]);
    assert_eq!(packing.offsets, [0, 0, 0, 32], "scratch requests alias, kernarg requests bump");
    assert_eq!((packing.scratch_bytes, packing.kernarg_bytes), (256, 44));
}

// ------------------------------------------------------------ neutral executors

fn dispatch(kernel_object: u64, completion_signal: u64) -> crate::hcq::ComputeDispatch {
    crate::hcq::ComputeDispatch {
        workgroup_size: [64, 1, 1],
        grid_size: [1024, 1, 1],
        private_segment_size: 0,
        group_segment_size: 0,
        kernel_object,
        kernarg_address: 0x3000,
        completion_signal,
        barrier: true,
        amd_pm4: None,
    }
}

#[test]
fn null_hcq_enforces_timeline_dependencies_and_order() {
    let signal = 0x1000;
    let mut null = NullHcq::default();
    let mut blocked = Submission::new(QueueKind::Compute(0));
    blocked.push(Command::Wait { signal_address: signal, value: 2 });
    assert!(null.submit(&blocked).is_err(), "an unsatisfied wait must not run");

    null.set_signal(signal, 1);
    let compute = dispatch(0x2000, 0x4000);
    let mut submit = Submission::new(QueueKind::Compute(0));
    submit
        .push(Command::Wait { signal_address: signal, value: 1 })
        .push(Command::MemoryBarrier)
        .push(Command::Compute(compute.clone()))
        .push(Command::Store { dst: signal, value: 2, scope: StoreScope::System });
    null.submit(&submit).unwrap();

    assert_eq!(
        null.trace().iter().map(|(_, command)| command.clone()).collect::<Vec<_>>(),
        submit.commands,
        "every command runs, in order"
    );
    null.submit(&blocked).unwrap();
}

#[test]
fn null_hcq_timestamps_use_deterministic_queue_clock() {
    let mut null = NullHcq::with_clock(1_000, 25);
    let mut compute = Submission::new(QueueKind::Compute(0));
    compute
        .push(Command::Timestamp { dst: 0x40 })
        .push(Command::Execute { operation: 0 })
        .push(Command::Timestamp { dst: 0x48 })
        .push(Command::Store { dst: 0x20, value: 1, scope: StoreScope::System });
    let mut copy = Submission::new(QueueKind::Copy(0));
    copy.push(Command::Wait { signal_address: 0x20, value: 1 })
        .push(Command::Timestamp { dst: 0x50 })
        .push(Command::Copy { dst: 0x2000, src: 0x1000, bytes: 16 })
        .push(Command::Timestamp { dst: 0x58 });

    null.submit(&compute).unwrap();
    null.submit(&copy).unwrap();
    // One tick per command, shared across queues.
    assert_eq!(null.signal_value(0x40), Some(1_000));
    assert_eq!(null.signal_value(0x48), Some(1_025));
    assert_eq!(null.signal_value(0x50), Some(1_050));
    assert_eq!(null.signal_value(0x58), Some(1_075));
}

#[test]
fn cpu_hcq_mixed_compute_copy_waits_and_finalizers_are_ordered() {
    let source = [3u8, 1, 4, 1];
    let mut intermediate = [0u8; 4];
    let mut destination = [0u8; 4];
    let mut executor = CpuQueueExecutor::with_clock(100, 10);
    let mut compute = Submission::new(QueueKind::Compute(0));
    compute
        .push(Command::MemoryBarrier)
        .push(Command::Execute { operation: 7 })
        .push(Command::Copy { dst: intermediate.as_mut_ptr() as u64, src: source.as_ptr() as u64, bytes: source.len() })
        .push(Command::Timestamp { dst: 0x30 })
        .push(Command::Store { dst: 0x20, value: 1, scope: StoreScope::System });
    let mut copy = Submission::new(QueueKind::Copy(0));
    copy.push(Command::Wait { signal_address: 0x20, value: 1 })
        .push(Command::Copy {
            dst: destination.as_mut_ptr() as u64,
            src: intermediate.as_ptr() as u64,
            bytes: intermediate.len(),
        })
        .push(Command::Timestamp { dst: 0x38 })
        .push(Command::Store { dst: 0x28, value: 2, scope: StoreScope::System });

    let mut operations = Vec::new();
    unsafe {
        executor.submit(&compute, |operation| {
            operations.push(operation);
            Ok::<_, ()>(())
        })
    }
    .unwrap();
    unsafe { executor.submit(&copy, |_| Ok::<_, ()>(())) }.unwrap();

    assert_eq!(operations, [7]);
    assert_eq!(destination, source, "the copy queue observes the compute queue's staged bytes");
    assert_eq!(executor.signal_value(0x30), Some(100));
    assert_eq!(executor.signal_value(0x38), Some(110));
    assert_eq!(executor.signal_value(0x28), Some(2));
}

#[test]
fn cpu_and_null_compute_errors_do_not_publish_finalizers() {
    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.push(Command::Execute { operation: 9 }).push(Command::Store {
        dst: 0x20,
        value: 1,
        scope: StoreScope::System,
    });

    let mut cpu = CpuQueueExecutor::default();
    let error = unsafe { cpu.submit(&submission, |_| Err("CPU failure")) }.unwrap_err();
    assert!(matches!(error, SubmissionExecutionError::Execute("CPU failure")));
    assert_eq!(cpu.signal_value(0x20), None);

    let mut null = NullHcq::default();
    let error = null.submit_with(&submission, |_| Err("null failure")).unwrap_err();
    assert!(matches!(error, SubmissionExecutionError::Execute("null failure")));
    assert_eq!(null.signal_value(0x20), None);
}

// -------------------------------------------------------------- patched replays

#[test]
fn inserting_profile_commands_preserves_patch_ownership() {
    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.push(Command::Wait { signal_address: 0, value: 1 });
    submission.bind(0, CommandField::WaitAddress, PatchSource::System(SystemField::TimelineSignal(0))).unwrap();
    submission.insert(0, Command::Timestamp { dst: 0x80 });
    assert_eq!(submission.patches()[0].command, 1, "the bound command's index shifts with it");
    assert_eq!(submission.commands[1], Command::Wait { signal_address: 0, value: 1 });
}

#[test]
fn semantic_link_retains_structure_and_repatches_runtime_and_system_fields() {
    let mut submission = Submission::new(QueueKind::Copy(0));
    submission.push(Command::Wait { signal_address: 0, value: 0 });
    submission.bind(0, CommandField::WaitAddress, PatchSource::System(SystemField::TimelineSignal(0))).unwrap();
    submission.bind(0, CommandField::WaitValue, PatchSource::System(SystemField::TimelineValue(0))).unwrap();
    submission.push(Command::Copy { dst: 0, src: 0, bytes: 4 });
    submission.bind(1, CommandField::CopyDst, PatchSource::RuntimeBuffer(0)).unwrap();
    submission.bind(1, CommandField::CopySrc, PatchSource::RuntimeBuffer(1)).unwrap();

    let linked = SemanticLinkedSubmission::new(submission);
    let static_ptr = linked.static_submission().commands.as_ptr();
    let mut replay = linked.replay_buffer();
    let mut system = SystemPatchValues::default();
    let mut apply = |value: u64, dst: u64, src: u64| {
        system.0.insert(SystemField::TimelineSignal(0), 0x1000);
        system.0.insert(SystemField::TimelineValue(0), value);
        linked.patch(&mut replay, &RuntimePatchValues { buffers: vec![dst, src], vars: vec![] }, &system).unwrap();
        replay.submission().commands.clone()
    };
    assert_eq!(
        apply(7, 0x2000, 0x3000),
        [Command::Wait { signal_address: 0x1000, value: 7 }, Command::Copy { dst: 0x2000, src: 0x3000, bytes: 4 },]
    );
    assert_eq!(
        apply(8, 0x4000, 0x5000),
        [Command::Wait { signal_address: 0x1000, value: 8 }, Command::Copy { dst: 0x4000, src: 0x5000, bytes: 4 },]
    );
    assert_eq!(linked.static_submission().commands.as_ptr(), static_ptr, "the template is never reallocated");
    assert_eq!(linked.static_submission().commands[0], Command::Wait { signal_address: 0, value: 0 });
}

#[test]
fn neutral_patch_tables_cache_link_bytes_and_scatter_replays() {
    let lowered = LoweredCommandBuffer {
        bytes: vec![0xaa; 24],
        patches: PatchTable::from_sites(vec![
            PatchSite { byte_offset: 0, encoding: PatchEncoding::U64, source: PatchSource::LinkAddress(0), addend: 0 },
            PatchSite {
                byte_offset: 8,
                encoding: PatchEncoding::U64,
                source: PatchSource::RuntimeBuffer(0),
                addend: 16,
            },
            PatchSite { byte_offset: 16, encoding: PatchEncoding::U32, source: PatchSource::RuntimeVar(0), addend: 0 },
            PatchSite {
                byte_offset: 20,
                encoding: PatchEncoding::U32,
                source: PatchSource::System(SystemField::TimelineValue(0)),
                addend: 0,
            },
        ]),
    };
    assert_eq!((lowered.patches.link.len(), lowered.patches.runtime.len(), lowered.patches.system.len()), (1, 2, 1));

    let mut cache = CommandBufferCache::default();
    let linked = cache.link(&lowered, &LinkPatchValues(vec![0x1122_3344_5566_7788])).unwrap();
    let linked_again = cache.link(&lowered, &LinkPatchValues(vec![0x1122_3344_5566_7788])).unwrap();
    assert!(std::sync::Arc::ptr_eq(&linked, &linked_again), "identical link values reuse the linked image");
    let immutable = linked.static_bytes().to_vec();

    let mut replay = linked.replay_buffer();
    let mut system = SystemPatchValues::default();
    system.0.insert(SystemField::TimelineValue(0), 3);
    linked.patch(&mut replay, &RuntimePatchValues { buffers: vec![0x2000], vars: vec![7] }, &system).unwrap();
    assert_eq!(&replay.bytes()[8..16], &0x2010u64.to_le_bytes(), "the site's addend is applied");

    system.0.insert(SystemField::TimelineValue(0), 4);
    linked.patch(&mut replay, &RuntimePatchValues { buffers: vec![0x9000], vars: vec![-2] }, &system).unwrap();
    assert_eq!(&replay.bytes()[8..16], &0x9010u64.to_le_bytes());
    assert_eq!(&replay.bytes()[16..20], &(-2i32).to_le_bytes());
    assert_eq!(linked.static_bytes(), immutable);
}

#[test]
fn linked_buffer_cache_is_scoped_by_context_and_device() {
    let lowered = LoweredCommandBuffer { bytes: vec![0; 8], patches: PatchTable::default() };
    let values = LinkPatchValues::default();
    let mut cache = CommandBufferCache::default();
    let first = cache.link_for_context(1, &gpu(0), &lowered, &values).unwrap();
    assert!(std::sync::Arc::ptr_eq(&first, &cache.link_for_context(1, &gpu(0), &lowered, &values).unwrap()));
    assert!(!std::sync::Arc::ptr_eq(&first, &cache.link_for_context(2, &gpu(0), &lowered, &values).unwrap()));
    assert!(!std::sync::Arc::ptr_eq(&first, &cache.link_for_context(1, &gpu(1), &lowered, &values).unwrap()));
}

#[test]
fn concurrent_device_replays_patch_private_buffers() {
    let mut submission = Submission::new(QueueKind::Copy(0));
    submission.push(Command::Copy { dst: 0, src: 0, bytes: 8 });
    submission.bind(0, CommandField::CopyDst, PatchSource::RuntimeBuffer(0)).unwrap();
    submission.bind(0, CommandField::CopySrc, PatchSource::RuntimeBuffer(1)).unwrap();
    let linked = std::sync::Arc::new(SemanticLinkedSubmission::new(submission));
    std::thread::scope(|scope| {
        for lane in 0..2u64 {
            let linked = std::sync::Arc::clone(&linked);
            scope.spawn(move || {
                let mut replay = linked.replay_buffer();
                linked
                    .patch(
                        &mut replay,
                        &RuntimePatchValues { buffers: vec![0x1000 + lane, 0x2000 + lane], vars: vec![] },
                        &SystemPatchValues::default(),
                    )
                    .unwrap();
                assert_eq!(
                    replay.submission().commands[0],
                    Command::Copy { dst: 0x1000 + lane, src: 0x2000 + lane, bytes: 8 }
                );
            });
        }
    });
    assert_eq!(linked.static_submission().commands[0], Command::Copy { dst: 0, src: 0, bytes: 8 });
}

#[test]
fn timeline_rollover_switches_signal_and_requests_one_reset() {
    let mut timeline = crate::hcq::EpochTimeline::with_next([0x1000, 0x2000], crate::hcq::TIMELINE_ROLLOVER + 1);
    assert_eq!(timeline.reserve(), crate::hcq::TimelinePoint { signal_address: 0x2000, value: 1 });
    assert_eq!(timeline.take_reset(), Some(0x2000));
    assert_eq!(timeline.take_reset(), None, "rollover reset ownership is consumed once");
}

// ------------------------------------------------------------- lane scheduling

fn gpu(id: usize) -> DeviceSpec {
    DeviceSpec::Amd { device_id: id }
}

fn lane(device: usize, queue: QueueKind) -> DeviceQueue {
    DeviceQueue { device: gpu(device), queue }
}

fn resource(id: u64, owner: usize) -> TopologyResource {
    resource_range(id, owner, 0, 16)
}

fn resource_range(id: u64, owner: usize, start: usize, end: usize) -> TopologyResource {
    TopologyResource { id, owner: gpu(owner), start, end }
}

fn execute(
    operation: usize,
    lane: DeviceQueue,
    reads: Vec<TopologyResource>,
    writes: Vec<TopologyResource>,
) -> TopologyOperation {
    TopologyOperation { operation, lane, reads, writes, kind: TopologyOperationKind::Execute }
}

fn copy_op(operation: usize, src: TopologyResource, dst: TopologyResource) -> TopologyOperation {
    TopologyOperation {
        operation,
        lane: DeviceQueue { device: dst.owner.clone(), queue: QueueKind::Copy(0) },
        reads: vec![src.clone()],
        writes: vec![dst.clone()],
        kind: TopologyOperationKind::Copy { src, dst, bytes: 16 },
    }
}

/// Every lane gets its own pair of timeline signals, keyed by device and queue.
fn lane_signals(lane: &DeviceQueue) -> [u64; 2] {
    let device = match &lane.device {
        DeviceSpec::Amd { device_id } => *device_id as u64 + 1,
        DeviceSpec::Cpu => 0,
        _ => 0x100,
    };
    let queue = match lane.queue {
        QueueKind::Compute(number) => number as u64 * 4,
        QueueKind::Copy(number) => number as u64 * 4 + 2,
    };
    let first = 0x1000 + device * 0x100 + queue * 0x10;
    [first, first + 8]
}

/// Schedule with each device only able to reach its own resources.
fn schedule_local(operations: &[TopologyOperation], limits: QueueMergeLimits) -> Vec<LaneSubmission> {
    schedule_device_lanes(operations, limits, |executor, owner| executor == owner)
}

/// Operation ids in the order the neutral executor runs them.
fn null_order(lanes: Vec<LaneSubmission>) -> Vec<usize> {
    let plan = SemanticLinkedPlan::from_lane_submissions(lanes, lane_signals).unwrap();
    let mut order = Vec::new();
    plan.execute_null(&mut NullHcq::default(), |_, command| {
        order.push(command.operation);
        Ok::<_, ()>(())
    })
    .unwrap();
    order
}

#[test]
fn direct_copy_runs_on_the_executor_that_declares_access() {
    let mut op = copy_op(0, resource(1, 0), resource(2, 1));
    op.lane.device = gpu(2);
    let scheduled = schedule_device_lanes(&[op], QueueMergeLimits::UNLIMITED, |executor, owner| {
        executor == &gpu(2) && matches!(owner, DeviceSpec::Amd { .. })
    });
    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduled[0].lane.device, gpu(2));
    assert_eq!(scheduled[0].commands[0].copy_leg, Some(CopyLeg::Direct));
    assert_eq!(null_order(scheduled), [0]);
}

/// Without declared peer access a cross-device copy splits into two staged legs,
/// each waiting only on its own producer rather than on a global barrier.
#[test]
fn inaccessible_cross_device_copy_stages_through_the_host() {
    let produced = resource(20, 0);
    let output = resource(21, 1);
    let operations =
        [execute(0, lane(0, QueueKind::Compute(0)), vec![], vec![produced.clone()]), copy_op(1, produced, output)];
    let scheduled = schedule_local(&operations, QueueMergeLimits::UNLIMITED);

    assert_eq!(scheduled.len(), 3, "compute, source->host, target<-host");
    assert_eq!(scheduled[1].lane.device, gpu(0));
    assert_eq!(scheduled[1].commands[0].copy_leg, Some(CopyLeg::ToHost));
    assert_eq!(scheduled[1].waits, [LaneWait { lane: scheduled[0].lane.clone(), value: 1 }]);
    assert_eq!(scheduled[2].lane.device, gpu(1));
    assert_eq!(scheduled[2].commands[0].copy_leg, Some(CopyLeg::FromHost));
    assert_eq!(scheduled[2].waits, [LaneWait { lane: scheduled[1].lane.clone(), value: 1 }]);
    assert_eq!(null_order(scheduled), [0, 1, 1], "both legs carry the original operation id");
}

#[test]
fn queue_merge_limits_split_exactly_after_boundary() {
    let operations =
        (0..5).map(|operation| execute(operation, lane(0, QueueKind::Compute(0)), vec![], vec![])).collect::<Vec<_>>();
    let merged = schedule_local(&operations, QueueMergeLimits { max_submissions: 2, max_commands: 2 });
    assert_eq!(merged.iter().map(|lane| lane.commands.len()).collect::<Vec<_>>(), [2, 2, 1]);
    let unmerged = schedule_local(&operations, QueueMergeLimits::NO_MERGE);
    assert_eq!(unmerged.iter().map(|lane| lane.commands.len()).collect::<Vec<_>>(), [1, 1, 1, 1, 1]);
}

#[test]
fn equal_queue_numbers_on_different_devices_keep_distinct_timelines() {
    let operations = [
        execute(0, lane(0, QueueKind::Compute(0)), vec![], vec![]),
        execute(1, lane(1, QueueKind::Compute(0)), vec![], vec![]),
    ];
    let lanes = schedule_local(&operations, QueueMergeLimits::UNLIMITED);
    let plan = SemanticLinkedPlan::from_lane_submissions(lanes.clone(), lane_signals).unwrap();
    assert_ne!(plan.bindings()[0].point.signal_address, plan.bindings()[1].point.signal_address);
    assert_eq!(null_order(lanes), [0, 1]);
}

/// A hazard on overlapping bytes makes the consumer lane wait on the producer
/// lane's timeline, in either direction between compute and copy queues.
#[test_case::test_case(&[], &[(0, 16)], &[(0, 16)], &[], true; "read after write")]
#[test_case::test_case(&[(0, 16)], &[], &[], &[(0, 16)], true; "write after read")]
#[test_case::test_case(&[], &[(0, 16)], &[], &[(0, 16)], true; "write after write")]
#[test_case::test_case(&[], &[(0, 9)], &[(8, 16)], &[], true; "overlapping byte ranges")]
#[test_case::test_case(&[], &[(0, 8)], &[(8, 16)], &[], false; "disjoint byte ranges")]
fn topology_hazards_wait_for_the_producer_lane(
    producer_reads: &[(usize, usize)],
    producer_writes: &[(usize, usize)],
    consumer_reads: &[(usize, usize)],
    consumer_writes: &[(usize, usize)],
    hazard: bool,
) {
    let spans = |ranges: &[(usize, usize)]| {
        ranges.iter().map(|&(start, end)| resource_range(50, 0, start, end)).collect::<Vec<_>>()
    };
    let producer = lane(0, QueueKind::Compute(0));
    let operations = [
        execute(0, producer.clone(), spans(producer_reads), spans(producer_writes)),
        execute(1, lane(0, QueueKind::Copy(0)), spans(consumer_reads), spans(consumer_writes)),
    ];
    let lanes = schedule_local(&operations, QueueMergeLimits::NO_MERGE);
    let expected: &[LaneWait] = if hazard { &[LaneWait { lane: producer, value: 1 }] } else { &[] };
    assert_eq!(lanes[1].waits, expected);
    assert_eq!(null_order(lanes), [0, 1]);
}

#[test]
fn compute_copy_dependencies_execute_in_both_directions() {
    let first = resource(30, 0);
    let second = resource(31, 0);
    let operations = [
        execute(0, lane(0, QueueKind::Compute(0)), vec![], vec![first.clone()]),
        copy_op(1, first, second.clone()),
        execute(2, lane(0, QueueKind::Compute(0)), vec![second], vec![]),
    ];
    let lanes = schedule_local(&operations, QueueMergeLimits::NO_MERGE);
    assert_eq!(lanes[1].waits[0].lane.queue, QueueKind::Compute(0));
    assert_eq!(lanes[2].waits[0].lane.queue, QueueKind::Copy(0));
    assert_eq!(null_order(lanes), [0, 1, 2]);
}

/// Merging moves a lane's published boundary to the end of the merged run, so a
/// dependent lane must wait for that later value, not for the producer's own.
#[test]
fn merged_waits_target_and_execute_published_boundaries() {
    let produced = resource(60, 0);
    let operations = [
        execute(0, lane(0, QueueKind::Compute(0)), vec![], vec![produced.clone()]),
        execute(1, lane(0, QueueKind::Compute(0)), vec![], vec![]),
        execute(2, lane(0, QueueKind::Copy(0)), vec![produced], vec![]),
    ];
    let lanes = schedule_local(&operations, QueueMergeLimits::UNLIMITED);
    assert_eq!(lanes.len(), 2);
    assert_eq!(lanes[0].signal_value, 2);
    assert_eq!(lanes[1].waits[0].value, 2);
    assert_eq!(null_order(lanes.clone()), [0, 1, 2]);

    let plan = SemanticLinkedPlan::from_lane_submissions(lanes, lane_signals).unwrap();
    let mut order = Vec::new();
    unsafe {
        plan.execute_cpu(&mut CpuQueueExecutor::default(), |_, command| {
            order.push(command.operation);
            Ok::<_, ()>(())
        })
    }
    .unwrap();
    assert_eq!(order, [0, 1, 2]);
}

#[test]
fn native_adapter_preserves_single_device_submission_shape() {
    let value = resource(70, 0);
    let operations = [
        execute(0, lane(0, QueueKind::Compute(0)), vec![], vec![value.clone()]),
        execute(1, lane(0, QueueKind::Copy(0)), vec![value], vec![]),
    ];
    let lanes = schedule_local(&operations, QueueMergeLimits::NO_MERGE);
    let plan = SemanticLinkedPlan::from_lane_submissions(lanes, lane_signals).unwrap();
    let native = plan.native_submissions().unwrap();

    assert_eq!(native.len(), 3);
    assert_eq!(native[0].lane().queue, QueueKind::Compute(0));
    assert!(matches!(
        native[0].static_submission().commands.as_slice(),
        [Command::MemoryBarrier, Command::Wait { .. }, Command::Execute { operation: 0 }, Command::Store { .. }]
    ));
    assert!(matches!(
        native[1].static_submission().commands.as_slice(),
        [
            Command::MemoryBarrier,
            Command::Wait { .. },
            Command::Wait { .. },
            Command::MemoryBarrier,
            Command::Execute { operation: 1 },
            Command::Store { .. }
        ]
    ));
    assert!(matches!(native[2].static_submission().commands.as_slice(), [Command::Wait { .. }, Command::Store { .. }]));
}

/// A store publishes its lane's writes only where another lane waits on it,
/// the finalizer included; any other store marks progress for work its own
/// queue runs behind it anyway. A lane that waits on another acquires after.
#[test]
fn native_stores_publish_only_where_another_lane_waits() {
    use Command::{Execute, MemoryBarrier, Store, Wait};
    use StoreScope::{Queue, System};

    let compute = lane(0, QueueKind::Compute(0));
    let (copied, landed) = (resource(80, 0), resource(81, 0));
    let operations = [
        execute(0, compute.clone(), vec![], vec![copied.clone()]),
        execute(1, compute.clone(), vec![], vec![]),
        copy_op(2, copied, landed.clone()),
        execute(3, compute.clone(), vec![landed], vec![]),
        execute(4, compute, vec![], vec![]),
    ];
    let lanes = schedule_local(&operations, QueueMergeLimits::NO_MERGE);
    assert_eq!(null_order(lanes.clone()), [0, 1, 2, 3, 4]);
    let plan = SemanticLinkedPlan::from_lane_submissions(lanes, lane_signals).unwrap();
    let native = plan.native_submissions().unwrap();
    let commands = |index: usize| native[index].static_submission().commands.as_slice();

    assert_eq!(native.len(), 6);
    // The copy reads operation 0's output.
    assert!(matches!(commands(0), [MemoryBarrier, Wait { .. }, Execute { operation: 0 }, Store { scope: System, .. }]));
    assert!(matches!(commands(1), [Execute { operation: 1 }, Store { scope: Queue, .. }]));
    // Operation 3 reads the copy's, and the finalizer waits on the copy lane.
    assert!(matches!(
        commands(2),
        [MemoryBarrier, Wait { .. }, Wait { .. }, MemoryBarrier, Execute { operation: 2 }, Store { scope: System, .. }]
    ));
    assert!(matches!(commands(3), [Wait { .. }, MemoryBarrier, Execute { operation: 3 }, Store { scope: Queue, .. }]));
    // The finalizer runs on the same queue: it publishes for the host.
    assert!(matches!(commands(4), [Execute { operation: 4 }, Store { scope: Queue, .. }]));
    assert!(matches!(commands(5), [Wait { .. }, Store { scope: System, .. }]));
}
