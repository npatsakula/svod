use super::test_support::{
    MockAmdIface, amd_alloc_or_skip, ensure_hw_signal_pool, mock_device, mock_device_with_signals, replay_dwords,
    scripted_error,
};
use crate::amd::connector::PoolQueue;
use crate::amd::iface::PublicationStage;
use crate::amd::queue::*;
use crate::amd::sys::hsa::{
    hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM, hsa_kernel_dispatch_packet_t, kernel_dispatch_header,
};
use crate::error::Error;
use crate::hcq::{AmdPm4Dispatch, Command, ComputeDispatch, QueueKind, StoreScope, Submission};
use std::sync::Arc;

// ---------------------------------------------------------------- packet forms

#[test]
fn aql_dispatch_packet_matches_the_hsa_layout() {
    // TYPE_KERNEL_DISPATCH | barrier | sys-acquire | sys-release.
    let sys = hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM as u16;
    assert_eq!(kernel_dispatch_header(), 2 | (1 << 8) | (sys << 9) | (sys << 11));
    assert_eq!(size_of::<hsa_kernel_dispatch_packet_t>(), AQL_PACKET_BYTES);
}

#[test_case::test_case([64, 1, 1], [1024, 1, 1] => 1; "one dimension")]
#[test_case::test_case([8, 8, 1], [256, 256, 1] => 2; "two dimensions")]
#[test_case::test_case([4, 4, 4], [64, 64, 64] => 3; "three dimensions")]
fn build_dispatch_picks_the_used_dimension_count(workgroup: [u16; 3], grid: [u32; 3]) -> u32 {
    let packet = build_dispatch_packet(workgroup, grid, 0, 0, 0, 0, 0);
    // `setup` (dims) is the high 16 bits of the header/setup union's full_header.
    (unsafe { packet.__bindgen_anon_1.full_header } >> 16) & 0b11
}

#[test]
fn hcq_command_stream_limits_are_checked_in_release_builds() {
    use crate::amd::sys::pm4::{INDIRECT_BUFFER_SIZE_MASK, INDIRECT_BUFFER_VALID, PACKET3_INDIRECT_BUFFER, packet3};

    assert!(validate_pm4_dword_count(1024).is_ok());
    assert!(matches!(
        validate_pm4_dword_count(1025),
        Err(Error::CommandStreamTooLarge { kind: "PM4 ring submission", actual: 1025, limit: 1024 })
    ));

    let aql_slots = COMPUTE_RING_BYTES / AQL_PACKET_BYTES;
    assert!(validate_aql_packet_count(aql_slots - 1).is_ok());
    assert!(matches!(
        validate_aql_packet_count(aql_slots),
        Err(Error::CommandStreamTooLarge { kind: "AQL ring submission", .. })
    ));

    assert!(build_aql_vendor_ib_packet(0x1000, INDIRECT_BUFFER_SIZE_MASK).is_ok());
    assert!(matches!(
        build_aql_vendor_ib_packet(0x1000, INDIRECT_BUFFER_SIZE_MASK + 1),
        Err(Error::CommandStreamTooLarge { kind: "PM4 indirect buffer", .. })
    ));
    assert_eq!(
        build_pm4_indirect_buffer(0x1122_3344_5566_7788, 1025).unwrap(),
        [packet3(PACKET3_INDIRECT_BUFFER, 2), 0x5566_7788, 0x1122_3344, 1025 | INDIRECT_BUFFER_VALID]
    );
}

#[test]
fn linked_transaction_limits_include_aggregate_and_sdma_wrap_padding() {
    assert_eq!(validate_linked_compute_lengths(true, 64, &[16, 20]).unwrap(), 9);
    assert!(matches!(
        validate_linked_compute_lengths(true, 64, &[32, 32]),
        Err(Error::CommandStreamTooLarge { kind: "PM4 linked transaction", actual: 16, limit: 15 })
    ));
    assert_eq!(validate_linked_compute_lengths(false, 256, &[64, 128]).unwrap(), 3);
    assert!(validate_linked_compute_lengths(false, 256, &[65]).is_err());

    // Starting 8 bytes before ring end, a 16-byte packet needs 8 bytes of NOP
    // padding before it, then the next packet follows without another wrap.
    assert_eq!(linked_sdma_published_bytes(56, 64, &[16, 8]).unwrap(), 32);
    assert!(matches!(
        linked_sdma_published_bytes(56, 64, &[32, 32]),
        Err(Error::CommandStreamTooLarge { kind: "SDMA linked transaction", .. })
    ));
}

const PM4_RING_SLOTS: usize = 4_194_304;
const PM4_RING_SLOTS_U64: u64 = PM4_RING_SLOTS as u64;

#[test_case::test_case(20_000, 7_485 => 7_485; "reader in the writer's epoch")]
#[test_case::test_case(PM4_RING_SLOTS_U64 + 24_588, 7_485 => PM4_RING_SLOTS_U64 + 7_485; "reader caught up into the new epoch")]
#[test_case::test_case(PM4_RING_SLOTS_U64 + 7_485, PM4_RING_SLOTS_U64 - 10 => PM4_RING_SLOTS_U64 - 10; "reader still in the previous epoch")]
fn pm4_read_pointer_reconstructs_the_producer_epoch(write_idx: u64, read_idx: u64) -> u64 {
    absolute_pm4_read_idx(write_idx, read_idx, PM4_RING_SLOTS)
}

#[test]
fn sdma_linear_copy_dwords_layout() {
    let dw = crate::amd::sys::sdma::copy_linear(0x1_0000_2000, 0x2_0000_3000, 4096);
    assert_eq!([dw[0], dw[1], dw[3], dw[4], dw[5], dw[6]], [0x01, 4095, 0x0000_2000, 1, 0x0000_3000, 2]);
}

#[test_case::test_case(9 => [5, 0x2000, 1, 7]; "cdna mtype")]
#[test_case::test_case(11 => [0x0003_0005, 0x2000, 1, 7]; "rdna mtype")]
fn sdma_fence_mtype_matches_tinygrad_by_arch(target_major: u32) -> [u32; 4] {
    crate::amd::sys::sdma::fence(0x1_0000_2000, 7, target_major)
}

/// A minimal one-workgroup dispatch; tests override only the fields they vary
/// via struct-update syntax.
fn dispatch(kernel_object: u64, kernarg_address: u64) -> ComputeDispatch {
    ComputeDispatch {
        workgroup_size: [8, 1, 1],
        grid_size: [16, 1, 1],
        private_segment_size: 0,
        group_segment_size: 0,
        kernel_object,
        kernarg_address,
        completion_signal: 0,
        barrier: true,
        amd_pm4: None,
    }
}

/// The AMD-specific half of a dispatch, with the scratch SGPR disabled.
fn amd_pm4(rsrc: [u32; 3], program_address: u64, workgroup_count: [u32; 3]) -> AmdPm4Dispatch {
    AmdPm4Dispatch {
        rsrc,
        program_address,
        enable_private_segment_sgpr: false,
        workgroup_count,
        wave32: true,
        target_major: 11,
    }
}

// ------------------------------------------------------- queue lifecycle (mock)

#[test]
fn compute_queue_construction_unwinds_every_allocation_stage() {
    for (xccs, allocation_stages) in [(1, 4), (2, 5)] {
        for fail_at in 0..allocation_stages {
            let (iface, allocator) = mock_device(xccs);
            for _ in 0..fail_at {
                iface.script_alloc(Ok(()));
            }
            iface.script_alloc(Err(scripted_error("compute queue allocation")));

            assert!(AmdComputeQueue::create(&allocator).is_err(), "xccs={xccs}, fail_at={fail_at}");
            let counts = (iface.allocation_count(), iface.free_count(), iface.live_handle_count());
            assert_eq!(counts, (fail_at, fail_at, 0), "xccs={xccs}, fail_at={fail_at}");
            assert!(iface.free_issues().is_empty());
        }
    }
}

#[test]
fn compute_queue_setup_success_failure_and_active_rollback_are_owned() {
    let (iface, allocator) = mock_device(1);
    iface.script_setup(Err(scripted_error("setup")));
    assert!(AmdComputeQueue::create(&allocator).is_err());
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (4, 4, 0));

    let queue = AmdComputeQueue::create(&allocator).expect("queue");
    assert_eq!(iface.live_queue_count(), 1);
    drop(queue);
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (8, 8, 0));
    assert_eq!((iface.queue_setup_count(), iface.queue_teardown_count(), iface.live_queue_count()), (1, 1, 0));

    // A leaked doorbell still retires the queue and all of its backing.
    let queue = AmdComputeQueue::create(&allocator).expect("queue with leaked doorbell");
    iface.script_teardown(Ok(crate::amd::iface::QueueTeardown::DoorbellLeaked { errno: 12 }));
    drop(queue);
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (12, 12, 0));
    assert_eq!(iface.live_queue_count(), 0);
    assert!(iface.free_issues().is_empty());

    // A rollback that cannot destroy the just-created queue poisons the device
    // and quarantines everything the CP may still be reading.
    let (iface, allocator) = mock_device(1);
    iface.script_setup(Err(Error::AmdQueueStillActive { queue_id: 77, cause: "scripted rollback failure".into() }));
    assert!(matches!(AmdComputeQueue::create(&allocator), Err(Error::AmdQueueStillActive { .. })));
    assert!(allocator.dev.is_poisoned());
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (4, 0, 4));
}

#[test]
fn copy_queue_construction_unwinds_ring_signal_and_staging_stages() {
    for fail_at in 0..3 {
        let (iface, allocator) = mock_device_with_signals(1);
        let baseline = iface.allocation_count();
        for _ in 0..fail_at {
            iface.script_alloc(Ok(()));
        }
        iface.script_alloc(Err(scripted_error("copy queue allocation")));

        assert!(AmdCopyQueue::create(&allocator).is_err(), "fail_at={fail_at}");
        assert_eq!(iface.allocation_count() - baseline, fail_at, "fail_at={fail_at}");
        assert_eq!((iface.free_count(), iface.live_handle_count()), (fail_at, baseline), "fail_at={fail_at}");
        // Only the staging stage runs after the ring is live, so only it tears down.
        let ring_live = usize::from(fail_at == 2);
        assert_eq!((iface.queue_setup_count(), iface.queue_teardown_count()), (ring_live, ring_live));
        assert!(iface.free_issues().is_empty());
    }

    // An exhausted signal pool grows another chunk instead of failing the queue.
    let (iface, allocator) = mock_device(1);
    let pool = crate::amd::signal::SignalPool::new(&allocator, 64).expect("signal pool");
    let held = (0..pool.capacity()).map(|_| pool.acquire().unwrap()).collect::<Vec<_>>();
    allocator.dev.core().install_signal_pool(Arc::clone(&pool));
    let queue = AmdCopyQueue::create(&allocator).expect("copy queue over a grown signal pool");
    assert_eq!(pool.capacity(), 128);
    assert_eq!((iface.allocation_count(), iface.free_count()), (5, 0));
    drop(queue);
    assert_eq!((iface.free_count(), iface.queue_teardown_count()), (3, 1));
    drop(held);

    // A destroy failure while unwinding leaves the ring mapped and poisons the device.
    let (iface, allocator) = mock_device_with_signals(1);
    let baseline = iface.allocation_count();
    iface.script_alloc(Ok(()));
    iface.script_alloc(Ok(()));
    iface.script_alloc(Err(scripted_error("staging")));
    iface.script_teardown(Err(scripted_error("partial queue destroy")));
    assert!(AmdCopyQueue::create(&allocator).is_err());
    assert!(allocator.dev.is_poisoned());
    assert_eq!((iface.allocation_count(), iface.free_count()), (baseline + 2, 0));
    assert_eq!((iface.live_handle_count(), iface.live_queue_count()), (baseline + 2, 1));
}

#[test]
fn copy_queue_destroy_failure_at_drop_poisons_and_quarantines() {
    let (iface, allocator) = mock_device_with_signals(1);
    let baseline = iface.allocation_count();
    let queue = AmdCopyQueue::create(&allocator).expect("copy queue");
    iface.script_teardown(Err(scripted_error("destroy")));
    drop(queue);
    assert!(allocator.dev.is_poisoned());
    assert_eq!(iface.allocation_count() - baseline, 3);
    assert_eq!((iface.free_count(), iface.live_handle_count()), (0, baseline + 3));
    assert_eq!(iface.live_queue_count(), 1);
}

#[test]
fn copy_queue_drains_registered_linked_finalizers_before_teardown() {
    let (iface, allocator) = mock_device_with_signals(1);
    let queue = AmdCopyQueue::create(&allocator).expect("copy queue");
    let signals = allocator.dev.core().signal_pool().expect("signal pool").clone();
    let signal = Arc::new(signals.acquire().expect("slot"));
    signal.reset(5);

    // Linked plans run on their own timeline, so the copy queue only knows
    // about their SDMA work through this registration.
    let finalizer = crate::amd::connector::SubmissionFinalizer::timeline(Arc::clone(&signal), 5, None);
    queue.register_inflight(Arc::clone(&finalizer));
    assert_eq!(queue.inflight_len(), 1);
    queue.register_inflight(finalizer);
    assert_eq!(queue.inflight_len(), 1, "retired entries are pruned, not accumulated");

    drop(queue);
    assert!(!allocator.dev.is_poisoned());
    assert_eq!((iface.free_count(), iface.queue_teardown_count()), (3, 1));
}

#[test]
fn pool_queue_construction_unwinds_queue_arena_and_scratch_stages() {
    let (iface, allocator) = mock_device_with_signals(1);
    let baseline = iface.allocation_count();
    for fail_at in 0..6 {
        let (allocations, frees) = (iface.allocation_count(), iface.free_count());
        for _ in 0..fail_at {
            iface.script_alloc(Ok(()));
        }
        iface.script_alloc(Err(scripted_error("pool allocation")));
        assert!(PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).is_err());
        assert_eq!(iface.allocation_count() - allocations, fail_at, "fail_at={fail_at}");
        assert_eq!(iface.free_count() - frees, fail_at, "fail_at={fail_at}");
        assert_eq!(iface.live_handle_count(), baseline, "fail_at={fail_at}");
        assert!(iface.free_issues().is_empty());
    }

    let (allocations, frees) = (iface.allocation_count(), iface.free_count());
    drop(PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue"));
    assert_eq!(iface.allocation_count() - allocations, 6);
    assert_eq!(iface.free_count() - frees, 6);
    assert_eq!(iface.live_handle_count(), baseline);
}

#[test]
fn quarantined_queue_leaks_its_backing_without_a_poisoned_device() {
    let (iface, allocator) = mock_device(1);
    let baseline = iface.allocation_count();
    let mut queue = AmdComputeQueue::create(&allocator).expect("queue");
    // Quarantine alone must keep the ring/GART/EOP mapped: the KFD queue is
    // never destroyed, so the CP may still be reading them. The decision is
    // queue-local and does not depend on the device poison latch.
    queue.quarantine();
    drop(queue);
    assert!(!allocator.dev.is_poisoned());
    assert_eq!((iface.free_count(), iface.live_handle_count()), (0, baseline + 4));
    assert_eq!((iface.queue_teardown_count(), iface.live_queue_count()), (0, 1));
}

#[test]
fn pool_failed_drain_and_panic_abandonment_quarantine_every_backing() {
    let (iface, allocator) = mock_device_with_signals(1);
    let baseline = iface.allocation_count();
    let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");
    pool.next_pm4();
    iface.script_wait(Err(scripted_error("drain")));
    drop(pool);
    assert!(allocator.dev.is_poisoned());
    assert_eq!((iface.free_count(), iface.live_handle_count()), (0, baseline + 6));

    // A panic unwind quarantines the lane it abandons but must NOT poison the
    // process-global core: tinygrad latches per-device error state on drain
    // timeouts and faults only.
    let (iface, allocator) = mock_device_with_signals(1);
    let baseline = iface.allocation_count();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).unwrap();
        panic!("scripted pool abandonment");
    }));
    assert!(result.is_err());
    assert!(!allocator.dev.is_poisoned());
    assert_eq!((iface.free_count(), iface.live_handle_count()), (0, baseline + 6));
    assert_eq!(iface.queue_teardown_count(), 0, "an abandoned queue is never destroyed");
}

#[test]
fn scratch_growth_preserves_old_state_on_drain_or_allocation_failure() {
    let (iface, allocator) = mock_device_with_signals(1);
    let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");
    let allocations = iface.allocation_count();
    pool.next_pm4();
    iface.script_wait(Err(scripted_error("scratch drain")));
    assert!(pool.ensure_has_local_memory(4096).is_err());
    assert_eq!(iface.allocation_count(), allocations, "failed drain must not allocate replacement scratch");
    assert_eq!(iface.free_count(), 0, "failed drain must not free old scratch");
    assert!(allocator.dev.is_poisoned());
    drop(pool);
    assert_eq!(iface.free_count(), 0);

    let (iface, allocator) = mock_device_with_signals(1);
    let baseline = iface.allocation_count();
    let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");
    iface.script_alloc(Err(scripted_error("replacement scratch allocation")));
    assert!(pool.ensure_has_local_memory(4096).is_err());
    assert!(!allocator.dev.is_poisoned());
    assert_eq!(iface.allocation_count(), baseline + 6);
    assert_eq!(iface.free_count(), 0, "old scratch must survive replacement allocation failure");
    drop(pool);
    assert_eq!(iface.free_count(), 6);

    let (iface, allocator) = mock_device_with_signals(1);
    let baseline = iface.allocation_count();
    let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");
    pool.ensure_has_local_memory(4096).expect("scratch growth");
    assert_eq!(iface.allocation_count(), baseline + 7);
    assert_eq!(iface.free_count(), 1, "successful publication frees exactly the drained old scratch");
    // Idempotent: an already-satisfied request neither drains nor reallocates.
    pool.ensure_has_local_memory(4096).expect("satisfied scratch request");
    pool.ensure_has_local_memory(1).expect("smaller scratch request");
    assert_eq!(iface.alloc_count_for_tag(crate::amd::va_registry::AllocTag::Scratch), 2);
    assert_eq!(iface.allocation_count(), baseline + 7);
    drop(pool);
    assert_eq!(iface.free_count(), 7);
    assert!(iface.free_issues().is_empty());
}

#[test]
fn lane_acquisition_times_out_instead_of_parking_forever() {
    let (_iface, allocator) = mock_device_with_signals(1);
    let core = Arc::clone(allocator.dev.core());
    let held = core.lease_queue(&allocator).expect("first lane");

    // The synthetic device has a single lane, so this cannot be satisfied: a
    // lease leaked by a wedged publisher must surface as a typed timeout.
    let error = core.lease_queue(&allocator).expect_err("a full pool must not park forever");
    assert!(matches!(error, Error::TimelineTimeout { what: "AMD lane acquisition", .. }), "{error:?}");

    drop(held);
    core.lease_queue(&allocator).expect("a released lane is acquired again");
}

#[test]
fn linked_publication_headroom_waits_before_taking_either_guard() {
    let (_iface, allocator) = mock_device_with_signals(1);
    let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");
    let copy = AmdCopyQueue::create(&allocator).expect("copy queue");

    // An idle ring clears both waits without holding a guard, so the guards can
    // then be taken back-to-back.
    pool.queue().wait_publication_headroom(&[64]).expect("compute headroom");
    copy.wait_publication_headroom(&[64]).expect("copy headroom");
    drop((
        pool.queue().prepare_linked_publication(&[64]).expect("compute guard"),
        copy.prepare_linked_publication(&[64]).expect("copy guard"),
    ));

    // Malformed lengths still fail before any guard is taken.
    assert!(pool.queue().wait_publication_headroom(&[7]).is_err());
    assert!(copy.wait_publication_headroom(&[7]).is_err());
}

#[test]
fn each_lane_reuses_its_own_linked_command_buffers() {
    let (_iface, allocator) = mock_device_with_signals(1);
    let first = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");
    let second = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");

    let lowered = crate::hcq::LoweredCommandBuffer { bytes: vec![0; 8], patches: crate::hcq::PatchTable::default() };
    let values = crate::hcq::LinkPatchValues::default();
    let linked = first.link(&lowered, &values).expect("link");
    let again = first.link(&lowered, &values).expect("relink");
    let other_lane = second.link(&lowered, &values).expect("other lane link");

    assert!(Arc::ptr_eq(&linked, &again), "a lane must reuse its own linked storage");
    assert!(!Arc::ptr_eq(&linked, &other_lane), "linked storage must not cross lanes");
}

// ----------------------------------------------------------- publication stages

/// Scripts publication checkpoint `stage` (0 = after reservation, 1 = before
/// doorbell, 2 = after doorbell) to fail — or to panic — drives one `submit`
/// through it, and asserts the queue reached exactly the checkpoints up to that
/// stage and no further.
fn publication_fails_at(iface: &MockAmdIface, stage: usize, panicking: bool, submit: impl FnOnce()) {
    const STAGES: [PublicationStage; 3] =
        [PublicationStage::AfterReservation, PublicationStage::BeforeDoorbell, PublicationStage::AfterDoorbell];
    let reached = iface.publication_stages().len();
    for _ in 0..stage {
        iface.script_publication(Ok(()));
    }
    if panicking {
        iface.script_publication_panic();
    } else {
        iface.script_publication(Err(scripted_error("publication")));
    }
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(submit)).is_err();
    assert_eq!(unwound, panicking, "a scripted publication panic must unwind out of the submission");
    assert_eq!(&iface.publication_stages()[reached..], &STAGES[..=stage]);
}

fn compute_submission() -> Submission {
    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.push(Command::MemoryBarrier).push(Command::Compute(ComputeDispatch {
        workgroup_size: [1, 1, 1],
        grid_size: [1, 1, 1],
        amd_pm4: Some(amd_pm4([0, 0, 0], 0x1000, [1, 1, 1])),
        ..dispatch(0x1000, 0x2000)
    }));
    submission
}

/// Everything up to the doorbell is transactional: the ring index and the
/// reserved timeline value are restored and the device stays usable. Once the
/// doorbell rings the CP owns the packets, so a failure there poisons the device
/// and quarantines its backing instead.
#[test_case::test_case(1; "pm4 (single-XCC)")]
#[test_case::test_case(2; "aql (multi-XCC)")]
fn compute_publication_rolls_back_before_the_doorbell_and_poisons_after(xccs: u32) {
    let submission = compute_submission();

    let (iface, allocator) = mock_device_with_signals(xccs);
    let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");
    for (stage, panicking) in [(0, false), (1, false), (1, true)] {
        publication_fails_at(&iface, stage, panicking, || {
            let _ = pool.queue().submit_hcq_dispatch(&pool, &submission, &[], &[]);
        });
        // The rolled-back reservation is handed straight back out, so the ring
        // never advances across the three abandoned attempts.
        assert_eq!((pool.pm4_value(), pool.queue().ring_write_idx()), (1, 0), "xccs={xccs}, stage={stage}");
        assert!(!allocator.dev.is_poisoned(), "xccs={xccs}, stage={stage}");
    }

    for panicking in [false, true] {
        let (iface, allocator) = mock_device_with_signals(xccs);
        let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");
        publication_fails_at(&iface, 2, panicking, || {
            let _ = pool.queue().submit_hcq_dispatch(&pool, &submission, &[], &[]);
        });
        assert_eq!(pool.pm4_value(), 2, "xccs={xccs}, panicking={panicking}");
        assert!(allocator.dev.is_poisoned(), "xccs={xccs}, panicking={panicking}");
        drop(pool);
        assert_eq!(iface.free_count(), 0, "poisoned resources must be quarantined");
    }
}

#[test]
fn copy_publication_rolls_back_before_the_doorbell_and_poisons_after() {
    let (iface, allocator) = mock_device_with_signals(1);
    let queue = AmdCopyQueue::create(&allocator).expect("copy queue");
    for (stage, panicking) in [(0, false), (1, false), (1, true)] {
        publication_fails_at(&iface, stage, panicking, || {
            let _ = queue.copy_fenced(0x1000, 0x2000, 4);
        });
        assert_eq!(queue.ring_write_idx(), 0, "abandoned copy packets must be rolled back (stage={stage})");
        assert!(!allocator.dev.is_poisoned(), "stage={stage}");
    }

    for panicking in [false, true] {
        let (iface, allocator) = mock_device_with_signals(1);
        let baseline = iface.allocation_count();
        let queue = AmdCopyQueue::create(&allocator).expect("copy queue");
        publication_fails_at(&iface, 2, panicking, || {
            let _ = queue.copy_fenced(0x1000, 0x2000, 4);
        });
        assert!(allocator.dev.is_poisoned(), "panicking={panicking}");
        drop(queue);
        assert_eq!((iface.free_count(), iface.live_handle_count()), (0, baseline + 3));
        assert_eq!(iface.live_queue_count(), 1);
    }
}

// ----------------------------------------------------------- HCQ → AMD packets

fn pm4_state() -> Pm4LoweringState {
    Pm4LoweringState {
        scratch_address: 0x1111_2222_3333_4400,
        tmpring_size: 0x55,
        target_major: 11,
        completion_xcc_mask: None,
        queue_event_mailbox: None,
    }
}

fn aql_control_state(multi_xcc: bool) -> Pm4LoweringState {
    Pm4LoweringState { target_major: 9, completion_xcc_mask: multi_xcc.then_some(1), ..pm4_state() }
}

fn compute_of(commands: impl IntoIterator<Item = Command>) -> Submission {
    let mut submission = Submission::new(QueueKind::Compute(0));
    for command in commands {
        submission.push(command);
    }
    submission
}

#[test]
fn hcq_compute_lowers_to_exact_aql_fields() {
    let command = ComputeDispatch {
        workgroup_size: [8, 4, 2],
        grid_size: [128, 64, 2],
        private_segment_size: 96,
        group_segment_size: 512,
        completion_signal: 0x1234_5678_9abc_def0,
        barrier: false,
        ..dispatch(0x1122_3344_5566_7788, 0x8877_6655_4433_2210)
    };
    let packet = lower_hcq_compute(&command).unwrap();
    assert_eq!([packet.workgroup_size_x, packet.workgroup_size_y, packet.workgroup_size_z], [8, 4, 2]);
    assert_eq!([packet.grid_size_x, packet.grid_size_y, packet.grid_size_z], command.grid_size);
    assert_eq!((packet.private_segment_size, packet.group_segment_size), (96, 512));
    assert_eq!(packet.kernel_object, command.kernel_object);
    assert_eq!(packet.kernarg_address as u64, command.kernarg_address);
    assert_eq!(packet.completion_signal.handle, command.completion_signal);
    let header = unsafe { packet.__bindgen_anon_1.full_header } as u16;
    assert_eq!(header & (1 << crate::amd::sys::hsa::hsa_packet_header_t_HSA_PACKET_HEADER_BARRIER), 0);

    // The AQL ring path emits exactly those dwords, unrepacked.
    let aql = lower_hcq_aql(&compute_of([Command::Compute(command)])).unwrap();
    let mut expected = [0u32; 16];
    // SAFETY: the AQL dispatch packet is exactly 16 POD dwords.
    unsafe { std::ptr::copy_nonoverlapping(&packet as *const _ as *const u32, expected.as_mut_ptr(), 16) };
    assert_eq!(aql, [expected]);
}

#[test]
fn hcq_amd_rejects_unsupported_packet_forms_wide_waits_and_oversize_workgroups() {
    let copy_on_compute = compute_of([Command::Copy { dst: 1, src: 2, bytes: 4 }]);
    assert!(lower_hcq_pm4(&copy_on_compute, pm4_state()).unwrap_err().to_string().contains("does not support"));

    let mut execute_on_copy = Submission::new(QueueKind::Copy(0));
    execute_on_copy.push(Command::Execute { operation: 0 });
    assert!(lower_hcq_sdma(&execute_on_copy, 11, None).unwrap_err().to_string().contains("does not support"));

    let wide = compute_of([Command::Wait { signal_address: 0x1000, value: u32::MAX as u64 + 1 }]);
    assert!(lower_hcq_pm4(&wide, pm4_state()).unwrap_err().to_string().contains("32-bit"));

    assert!(
        lower_hcq_compute(&ComputeDispatch { workgroup_size: [u16::MAX as u32 + 1, 1, 1], ..dispatch(0, 0) }).is_err()
    );
}

/// PM4 lowering is per-command concatenation: each command contributes its own
/// dword run, and a mixed submission is exactly those runs in submission order
/// with nothing inserted between them.
#[test]
fn hcq_pm4_command_goldens_concatenate_in_submission_order() {
    const ADDRESS: u64 = 0x1122_3344_5566_7788;
    let goldens: [(Command, &[u32]); 4] = [
        (
            Command::Wait { signal_address: ADDRESS, value: 7 },
            &[0xc005_3c00, 0x15, 0x5566_7788, 0x1122_3344, 7, 0xffff_ffff, 4],
        ),
        (
            Command::MemoryBarrier,
            &[
                0xc005_3c00,
                0x43,
                0xe26,
                0xe27,
                0x1000,
                0x1000,
                0x20,
                0xc006_5800,
                0,
                0xffff_ffff,
                0xffff_ffff,
                0,
                0,
                0,
                0xc3f1,
            ],
        ),
        (
            Command::Timestamp { dst: ADDRESS },
            &[
                0xc006_4900,
                0x514,
                0x0200_0000,
                0,
                0,
                0,
                0,
                0,
                0xc006_4900,
                0x514,
                0x6000_0000,
                0x5566_7788,
                0x1122_3344,
                0,
                0,
                0,
                0xc006_5800,
                0,
                0xffff_ffff,
                0xffff_ffff,
                0,
                0,
                0,
                0xc3f1,
            ],
        ),
        (
            Command::Store { dst: ADDRESS, value: 0xaabb_ccdd_eeff_0011, scope: StoreScope::System },
            &[0xc006_4900, 0x70f514, 0x4000_0000, 0x5566_7788, 0x1122_3344, 0xeeff_0011, 0xaabb_ccdd, 0],
        ),
    ];

    for (command, expected) in &goldens {
        assert_eq!(lower_hcq_pm4(&compute_of([command.clone()]), pm4_state()).unwrap(), *expected, "{command:?}");
    }
    let mixed = compute_of(goldens.iter().map(|(command, _)| command.clone()));
    let concatenated = goldens.iter().flat_map(|(_, dwords)| dwords.iter().copied()).collect::<Vec<_>>();
    assert_eq!(lower_hcq_pm4(&mixed, pm4_state()).unwrap(), concatenated);
}

/// A queue-scoped store is the same end-of-pipe write without the GCR
/// writeback and invalidate, on gfx10+ and gfx9 alike.
#[test]
fn hcq_pm4_queue_scoped_store_leaves_the_caches() {
    const ADDRESS: u64 = 0x1122_3344_5566_7788;
    let store = |scope| compute_of([Command::Store { dst: ADDRESS, value: 0xaabb_ccdd_eeff_0011, scope }]);
    assert_eq!(
        lower_hcq_pm4(&store(StoreScope::Queue), pm4_state()).unwrap(),
        [0xc006_4900, 0x514, 0x4000_0000, 0x5566_7788, 0x1122_3344, 0xeeff_0011, 0xaabb_ccdd, 0]
    );
    for state in [pm4_state(), aql_control_state(false), aql_control_state(true)] {
        let queue = lower_hcq_pm4(&store(StoreScope::Queue), state).unwrap();
        let system = lower_hcq_pm4(&store(StoreScope::System), state).unwrap();
        let differing: Vec<usize> = (0..system.len()).filter(|&i| queue[i] != system[i]).collect();
        let event = usize::from(state.completion_xcc_mask.is_some()) * 2 + 1;
        assert_eq!(differing, [event], "target {}", state.target_major);
        assert_eq!(queue[event], system[event] & queue[event], "only cache bits drop");
    }
}

#[test]
fn hcq_pm4_compute_golden() {
    let submission = compute_of([Command::Compute(ComputeDispatch {
        workgroup_size: [8, 4, 2],
        grid_size: [128, 32, 8],
        amd_pm4: Some(amd_pm4([1, 2, 3], 0x12_3456_7800, [16, 8, 4])),
        ..dispatch(0, 0x0000_00ab_cdef_0010)
    })]);
    assert_eq!(
        lower_hcq_pm4(&submission, pm4_state()).unwrap(),
        [
            0xc006_5800,
            0,
            0xffff_ffff,
            0xffff_ffff,
            0,
            0,
            0,
            0x3f0,
            0xc002_7600,
            0x20c,
            0x1234_5678,
            0,
            0xc002_7600,
            0x212,
            1,
            2,
            0xc001_7600,
            0x228,
            3,
            0xc001_7600,
            0x218,
            0x55,
            0xc002_7600,
            0x210,
            0x2233_3344,
            0x0011_1122,
            0xc003_7600,
            0x21b,
            0,
            0,
            0,
            0xc002_7600,
            0x240,
            0xcdef_0010,
            0xab,
            0xc001_7600,
            0x215,
            0,
            0xc008_7600,
            0x204,
            0,
            0,
            0,
            8,
            4,
            2,
            0,
            0,
            0xc003_1500,
            16,
            8,
            4,
            0x8005,
            0xc000_4600,
            0x407,
        ]
    );
}

#[test]
fn hcq_sdma_mixed_submission_golden() {
    let mut submission = Submission::new(QueueKind::Copy(0));
    submission
        .push(Command::MemoryBarrier)
        .push(Command::Wait { signal_address: 0x1_0000_1000, value: 9 })
        .push(Command::Copy { dst: 0x2_0000_2000, src: 0x3_0000_3000, bytes: 16 })
        .push(Command::Timestamp { dst: 0x4_0000_4000 })
        .push(Command::Store { dst: 0x5_0000_5000, value: 0x5566_7788, scope: StoreScope::System });
    assert_eq!(
        lower_hcq_sdma(&submission, 11, None).unwrap(),
        [
            0xd000_0008,
            0x1000,
            1,
            9,
            u32::MAX,
            0x0fff_0004,
            1,
            15,
            0,
            0x3000,
            3,
            0x2000,
            2,
            0x20d,
            0x4000,
            4,
            0x0003_0005,
            0x5000,
            5,
            0x5566_7788,
        ]
    );
}

#[test]
fn hcq_queue_event_mailbox_stores_raise_the_kfd_interrupt() {
    use crate::hcq::{CommandField, PatchSource};
    const MAILBOX: u64 = 0x7_0000_1234;
    let int_sel = |dword: u32| (dword >> 24) & 0b11;

    // PM4: the polled timeline store stays interrupt-free; only the mailbox
    // store interrupts, carrying the event id in both value and ctxid and no
    // cache flush (tinygrad ops_amd.py:388-393).
    let submission = compute_of([
        Command::Store { dst: 0x1_0000_1000, value: 7, scope: StoreScope::System },
        Command::Store { dst: MAILBOX, value: 9, scope: StoreScope::System },
    ]);
    let q = lower_hcq_pm4(&submission, Pm4LoweringState { queue_event_mailbox: Some(MAILBOX), ..pm4_state() }).unwrap();
    assert_eq!(q.len(), 16);
    assert_eq!(int_sel(q[2]), 0);
    assert_eq!(int_sel(q[10]), crate::amd::sys::pm4::INT_SEL_INTERRUPT_AFTER_WRITE);
    assert_eq!(q[9] & crate::amd::sys::pm4::RELEASE_MEM_CACHE_FLUSH_ALL, 0);
    assert_eq!(&q[11..16], &[MAILBOX as u32, (MAILBOX >> 32) as u32, 9, 0, 9]);
    // Without the mailbox address every store stays a plain memory write.
    assert_eq!(int_sel(lower_hcq_pm4(&submission, pm4_state()).unwrap()[10]), 0);

    // SDMA: mailbox fence followed by SDMA_OP_TRAP (ops_amd.py:490-492).
    let mut copy = Submission::new(QueueKind::Copy(0));
    copy.push(Command::Store { dst: MAILBOX, value: 9, scope: StoreScope::System });
    assert_eq!(
        lower_hcq_sdma(&copy, 11, Some(MAILBOX)).unwrap(),
        [0x0003_0005, MAILBOX as u32, (MAILBOX >> 32) as u32, 9, 6, 9]
    );
    assert_eq!(lower_hcq_sdma(&copy, 11, None).unwrap().len(), 4);
    // The extra TRAP dwords keep the SDMA patch cursor aligned.
    copy.bind(0, CommandField::StoreDst, PatchSource::LinkAddress(0)).unwrap();
    copy.push(Command::Store { dst: 0x2000, value: 1, scope: StoreScope::System });
    copy.bind(1, CommandField::StoreDst, PatchSource::LinkAddress(1)).unwrap();
    let lowered = lower_hcq_sdma_command_buffer(&copy, 11, Some(MAILBOX)).unwrap();
    assert_eq!(lowered.patches.link.iter().map(|site| site.byte_offset).collect::<Vec<_>>(), [4, 8, 28, 32]);
}

#[test]
fn hcq_sdma_zero_byte_copy_consumes_its_bindings_without_packets() {
    use crate::hcq::{CommandField, PatchSource};
    let mut submission = Submission::new(QueueKind::Copy(0));
    submission.push(Command::Copy { dst: 0, src: 0, bytes: 0 });
    submission.bind(0, CommandField::CopySrc, PatchSource::RuntimeBuffer(0)).unwrap();
    submission.bind(0, CommandField::CopyDst, PatchSource::RuntimeBuffer(1)).unwrap();
    let lowered = lower_hcq_sdma_command_buffer(&submission, 11, None).expect("zero-byte copy must lower");
    assert!(lowered.bytes.is_empty());
    assert!(lowered.patches.runtime.is_empty());
}

// -------------------------------------------------------------- dynamic replay

#[test]
fn hcq_pm4_replay_patches_vars_and_addresses_without_relowering() {
    use crate::hcq::{
        CommandBufferCache, CommandField, LinkPatchValues, PatchEncoding, PatchSource, RuntimePatchValues, SystemField,
        SystemPatchValues,
    };

    let mut submission = compute_of([
        Command::Wait { signal_address: 0, value: 0 },
        Command::Compute(ComputeDispatch {
            amd_pm4: Some(AmdPm4Dispatch { enable_private_segment_sgpr: true, ..amd_pm4([1, 2, 3], 0, [16, 1, 1]) }),
            ..dispatch(0, 0)
        }),
    ]);
    submission.bind(0, CommandField::WaitAddress, PatchSource::System(SystemField::TimelineSignal(0))).unwrap();
    submission.bind(0, CommandField::WaitValue, PatchSource::System(SystemField::TimelineValue(0))).unwrap();
    submission.bind(1, CommandField::ComputeProgramAddress, PatchSource::LinkAddress(0)).unwrap();
    submission.bind(1, CommandField::ComputeKernargAddress, PatchSource::RuntimeBuffer(0)).unwrap();
    submission.bind(1, CommandField::ComputeGrid(0), PatchSource::RuntimeVar(0)).unwrap();
    submission.bind(1, CommandField::ComputeScratchAddress, PatchSource::System(SystemField::ScratchAddress)).unwrap();
    submission.bind(1, CommandField::ComputeScratchTmpring, PatchSource::System(SystemField::ScratchTmpring)).unwrap();

    let lowered = lower_hcq_pm4_command_buffer(&submission, pm4_state()).unwrap();
    assert_eq!(
        (lowered.patches.link.len(), lowered.patches.runtime.len(), lowered.patches.system.len()),
        (2, 3, 8),
        "one patch site per patched dword"
    );
    let linked = CommandBufferCache::default().link(&lowered, &LinkPatchValues(vec![0x12_3456_7800])).unwrap();
    let static_bytes = linked.static_bytes().to_vec();
    let mut replay = linked.replay_buffer();
    let mut system = SystemPatchValues::default();
    let mut apply = |signal: u64, value: u64, scratch: u64, tmpring: u64, buffer: u64, var: i64| {
        system.0.insert(SystemField::TimelineSignal(0), signal);
        system.0.insert(SystemField::TimelineValue(0), value);
        system.0.insert(SystemField::ScratchAddress, scratch);
        system.0.insert(SystemField::ScratchTmpring, tmpring);
        linked
            .patch(&mut replay, &RuntimePatchValues { buffers: vec![buffer], vars: vec![var] }, &system)
            .expect("replay patch");
        replay.bytes().to_vec()
    };
    let first = apply(0x1000, 3, 0x1234_5678_9000, 0x55, 0xaaaa_bbbb_cccc_0000, 32);
    let second = apply(0x9000, 4, 0x2234_5678_a000, 0x66, 0x1111_2222_3333_0000, 64);
    assert_ne!(first, second);

    let dwords = replay_dwords(&second);
    assert_eq!(&dwords[2..5], &[0x9000, 0, 4]);
    let descriptor_high =
        lowered.patches.system.iter().find(|site| site.encoding == PatchEncoding::High32Or(1 << 31)).unwrap();
    assert_eq!(dwords[descriptor_high.byte_offset / 4], 0x8000_2234);
    assert_eq!(linked.static_bytes(), static_bytes, "the linked image stays immutable across replays");
    for site in &lowered.patches.link {
        assert_eq!(
            &second[site.byte_offset..site.byte_offset + 4],
            &static_bytes[site.byte_offset..site.byte_offset + 4],
            "link-time addresses are baked into the image, never re-patched"
        );
    }
}

#[test]
fn hcq_sdma_replay_patches_every_chunk_of_a_split_copy() {
    use crate::hcq::{
        CommandBufferCache, CommandField, LinkPatchValues, PatchSource, RuntimePatchValues, SystemField,
        SystemPatchValues,
    };
    const MAX_COPY: usize = crate::amd::sys::sdma::SDMA_MAX_COPY_BYTES;

    let mut submission = Submission::new(QueueKind::Copy(0));
    submission
        .push(Command::Copy { dst: 0, src: 0, bytes: MAX_COPY + 8 })
        .push(Command::Timestamp { dst: 0 })
        .push(Command::Store { dst: 0, value: 0, scope: StoreScope::System });
    submission.bind(0, CommandField::CopySrc, PatchSource::RuntimeBuffer(0)).unwrap();
    submission.bind(0, CommandField::CopyDst, PatchSource::RuntimeBuffer(1)).unwrap();
    submission.bind(1, CommandField::TimestampDst, PatchSource::System(SystemField::Timestamp(0))).unwrap();
    submission.bind(2, CommandField::StoreDst, PatchSource::LinkAddress(0)).unwrap();
    submission.bind(2, CommandField::StoreValue, PatchSource::System(SystemField::TimelineValue(0))).unwrap();

    let lowered = lower_hcq_sdma_command_buffer(&submission, 11, None).unwrap();
    assert_eq!(lowered.patches.runtime.len(), 8, "src/dst lo+hi for both chunks");
    let linked = CommandBufferCache::default().link(&lowered, &LinkPatchValues(vec![0x7000])).unwrap();
    let static_bytes = linked.static_bytes().to_vec();
    let mut replay = linked.replay_buffer();
    let mut system = SystemPatchValues::default();
    let mut apply = |timestamp: u64, value: u64, src: u64, dst: u64| {
        system.0.insert(SystemField::Timestamp(0), timestamp);
        system.0.insert(SystemField::TimelineValue(0), value);
        linked
            .patch(&mut replay, &RuntimePatchValues { buffers: vec![src, dst], vars: vec![] }, &system)
            .expect("replay patch");
        replay_dwords(replay.bytes())
    };

    // The second chunk starts SDMA_MAX_COPY_BYTES into both buffers.
    let first = apply(0x8000, 5, 0x1_0000_0000, 0x2_0000_0000);
    assert_eq!([first[3], first[4]], [0, 1]);
    assert_eq!([first[10], first[11]], [MAX_COPY as u32, 1]);
    let second = apply(0xa000, 6, 0x3_0000_1000, 0x4_0000_2000);
    assert_eq!([second[3], second[4]], [0x1000, 3]);
    assert_eq!([second[10], second[11]], [0x0040_1000, 3]);
    assert_eq!(linked.static_bytes(), static_bytes);
}

#[test]
fn hcq_aql_replay_patches_vars_and_addresses_without_kernel_completion() {
    use crate::hcq::{
        CommandBufferCache, CommandField, LinkPatchValues, PatchSource, RuntimePatchValues, SystemPatchValues,
    };

    let mut submission = compute_of([Command::MemoryBarrier, Command::Compute(dispatch(0, 0))]);
    submission.bind(1, CommandField::ComputeKernelObject, PatchSource::LinkAddress(0)).unwrap();
    submission.bind(1, CommandField::ComputeKernargAddress, PatchSource::RuntimeBuffer(0)).unwrap();
    submission.bind(1, CommandField::ComputeGrid(0), PatchSource::RuntimeVar(0)).unwrap();
    let lowered = lower_hcq_aql_command_buffer(&submission).unwrap();
    let linked = CommandBufferCache::default().link(&lowered, &LinkPatchValues(vec![0x1234_5600])).unwrap();
    let immutable = linked.static_bytes().to_vec();
    let mut replay = linked.replay_buffer();
    let system = SystemPatchValues::default();

    for (buffer, var) in [(0x1_0000_1000u64, 32u32), (0x2_0000_2000, 64)] {
        linked
            .patch(&mut replay, &RuntimePatchValues { buffers: vec![buffer], vars: vec![var.into()] }, &system)
            .expect("replay patch");
        assert_eq!(&replay.bytes()[12..16], &var.to_le_bytes(), "grid_size_x");
        assert_eq!(&replay.bytes()[40..48], &buffer.to_le_bytes(), "kernarg_address");
        assert_eq!(&replay.bytes()[56..64], &0u64.to_le_bytes(), "completion is owned by the control stream");
    }
    assert_eq!(linked.static_bytes(), immutable);
}

/// The AQL submission program splits in two: an on-ring AQL half (vendor IB
/// packets around the kernel dispatch) and a PM4 control half that owns the
/// waits and the timeline stores.
#[test]
fn hcq_aql_submission_program_keeps_wait_store_and_dispatch_on_device() {
    use crate::hcq::{
        CommandBufferCache, CommandField, LinkPatchValues, PatchSource, RuntimePatchValues, SystemField,
        SystemPatchValues,
    };

    let mut submission = compute_of([
        Command::Wait { signal_address: 0, value: 0 },
        Command::MemoryBarrier,
        Command::Compute(dispatch(0, 0)),
        Command::Store { dst: 0, value: 0, scope: StoreScope::System },
    ]);
    submission.bind(0, CommandField::WaitAddress, PatchSource::System(SystemField::TimelineSignal(0))).unwrap();
    submission.bind(0, CommandField::WaitValue, PatchSource::System(SystemField::TimelineValue(0))).unwrap();
    submission.bind(2, CommandField::ComputeKernelObject, PatchSource::LinkAddress(0)).unwrap();
    submission.bind(2, CommandField::ComputeKernargAddress, PatchSource::RuntimeBuffer(0)).unwrap();
    submission.bind(3, CommandField::StoreDst, PatchSource::System(SystemField::TimelineSignal(1))).unwrap();
    submission.bind(3, CommandField::StoreValue, PatchSource::System(SystemField::TimelineValue(1))).unwrap();

    let lowered =
        lower_hcq_aql_submission_program(&submission, aql_control_state(true), PatchSource::LinkAddress(1)).unwrap();
    assert_eq!(lowered.aql.bytes.len(), 3 * AQL_PACKET_BYTES, "IB, dispatch, IB");

    let links = LinkPatchValues(vec![0x1234_5600, 0x8000_0000]);
    let aql = CommandBufferCache::default().link(&lowered.aql, &links).unwrap();
    let control = CommandBufferCache::default().link(&lowered.control, &links).unwrap();
    let mut aql_replay = aql.replay_buffer();
    let mut control_replay = control.replay_buffer();
    let mut system = SystemPatchValues::default();
    system.0.insert(SystemField::TimelineSignal(0), 0x1000);
    system.0.insert(SystemField::TimelineValue(0), 7);
    system.0.insert(SystemField::TimelineSignal(1), 0x2000);
    system.0.insert(SystemField::TimelineValue(1), 8);
    let runtime = RuntimePatchValues { buffers: vec![0x3000], vars: vec![] };
    aql.patch(&mut aql_replay, &runtime, &system).unwrap();
    control.patch(&mut control_replay, &runtime, &system).unwrap();

    assert_eq!(&aql_replay.bytes()[8..16], &0x8000_0000u64.to_le_bytes());
    assert_eq!(&aql_replay.bytes()[AQL_PACKET_BYTES + 40..AQL_PACKET_BYTES + 48], &0x3000u64.to_le_bytes());
    let trailing_ib =
        u64::from_le_bytes(aql_replay.bytes()[2 * AQL_PACKET_BYTES + 8..2 * AQL_PACKET_BYTES + 16].try_into().unwrap());
    assert!(trailing_ib > 0x8000_0000, "trailing IB points at the linked store run");
    let control_words = replay_dwords(control_replay.bytes());
    assert_eq!(&control_words[2..5], &[0x1000, 0, 7]);
    assert_eq!(&control_words[control_words.len() - 5..control_words.len() - 2], &[0x2000, 0, 8]);
}

/// Multi-XCC control streams predicate their finalizer on one XCC; single-XCC
/// parts run the same stream with no PRED_EXEC at all.
#[test_case::test_case(true, 1; "multi-XCC predicates the finalizer")]
#[test_case::test_case(false, 0; "single-XCC omits PRED_EXEC")]
fn hcq_aql_control_only_finalizer_predicates_by_xcc_count(multi_xcc: bool, pred_execs: usize) {
    use crate::hcq::{CommandField, PatchSource, SystemField};

    let mut submission = compute_of([
        Command::Wait { signal_address: 0, value: 0 },
        Command::Store { dst: 0, value: 0, scope: StoreScope::System },
    ]);
    submission.bind(0, CommandField::WaitAddress, PatchSource::System(SystemField::TimelineSignal(0))).unwrap();
    submission.bind(0, CommandField::WaitValue, PatchSource::System(SystemField::TimelineValue(0))).unwrap();
    submission.bind(1, CommandField::StoreDst, PatchSource::System(SystemField::TimelineSignal(1))).unwrap();
    submission.bind(1, CommandField::StoreValue, PatchSource::System(SystemField::TimelineValue(1))).unwrap();

    let lowered =
        lower_hcq_aql_submission_program(&submission, aql_control_state(multi_xcc), PatchSource::LinkAddress(0))
            .unwrap();
    assert_eq!(lowered.aql.bytes.len(), AQL_PACKET_BYTES, "a control-only finalizer needs one vendor IB packet");
    let words = replay_dwords(&lowered.control.bytes);
    let pred = crate::amd::sys::pm4::packet3(crate::amd::sys::pm4::PACKET3_PRED_EXEC, 0);
    assert_eq!(words.iter().filter(|&&word| word == pred).count(), pred_execs);
    if multi_xcc {
        assert_eq!(&words[words.len() - 10..words.len() - 8], &crate::amd::sys::pm4::pred_exec(1, 8));
    }
}

/// Timeline completion — and, when profiling, the start/end timestamps — is
/// owned by predicated PM4 in the control stream: no AQL packet may carry a
/// native completion signal.
#[test_case::test_case(None, 1; "timeline only")]
#[test_case::test_case(Some((0x3000, 0x3008)), 3; "profiled adds start and end stamps")]
fn hcq_aql_timeline_completion_never_uses_the_aql_completion_signal(profile: Option<(u64, u64)>, pred_execs: usize) {
    use crate::hcq::PatchSource;

    let submission = compute_of([
        Command::MemoryBarrier,
        Command::Compute(ComputeDispatch { grid_size: [64, 1, 1], ..dispatch(0x1234_5600, 0x9000) }),
    ]);
    let finalized = finalize_hcq_aql_timeline_submission(&submission, 0x2000, 7, 8, profile).unwrap();
    let lowered =
        lower_hcq_aql_submission_program(&finalized, aql_control_state(true), PatchSource::LinkAddress(0)).unwrap();

    assert_eq!(lowered.aql.bytes.len(), 3 * AQL_PACKET_BYTES, "prefix IB, kernel, terminal IB");
    for packet in lowered.aql.bytes.as_chunks::<AQL_PACKET_BYTES>().0 {
        assert_eq!(&packet[56..64], &0u64.to_le_bytes(), "no AQL packet owns native completion");
    }
    assert_eq!(&lowered.aql.bytes[AQL_PACKET_BYTES + 32..AQL_PACKET_BYTES + 40], &0x1234_5600u64.to_le_bytes());

    let control = replay_dwords(&lowered.control.bytes);
    let pred = crate::amd::sys::pm4::packet3(crate::amd::sys::pm4::PACKET3_PRED_EXEC, 0);
    assert_eq!(control.iter().filter(|&&word| word == pred).count(), pred_execs);
    assert_eq!(&control[2..5], &[0x2000, 0, 7], "the control stream waits on the previous timeline point");
    assert_eq!(&control[control.len() - 10..control.len() - 8], &crate::amd::sys::pm4::pred_exec(1, 8));
    assert_eq!(&control[control.len() - 5..control.len() - 2], &[0x2000, 0, 8]);
    for stamp in [0x3000u32, 0x3008] {
        assert_eq!(control.iter().filter(|&&word| word == stamp).count(), usize::from(profile.is_some()));
    }

    // A dispatch that keeps its own completion signal is rejected outright.
    let mut invalid = submission;
    let Command::Compute(dispatch) = &mut invalid.commands[1] else { unreachable!() };
    dispatch.completion_signal = 0x4000;
    let err = lower_hcq_aql_command_buffer(&invalid).unwrap_err();
    assert!(err.to_string().contains("completion must remain unset"), "{err}");
}

// ------------------------------------------------------------- hardware probes

/// Live SDMA staging roundtrip: a device-local (`host_ptr: None`) buffer is
/// filled via `_copyin` and read back via `_copyout`, exercising the real SDMA
/// copy + fence + signal-wait path. A wrong fence fails via the copy timeout
/// rather than hanging.
#[test]
fn sdma_device_local_roundtrip() {
    use crate::allocator::{Allocator, BufferSpec, RawBuffer};
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let core = alloc.dev.core();
    ensure_hw_signal_pool(&alloc);
    if core.copy_queue().is_none() {
        core.install_copy_queue(AmdCopyQueue::create(&alloc).expect("copy queue"));
        core.set_has_sdma_queue(true);
    }

    let spec = BufferSpec { cpu_access: false, ..Default::default() };
    // Span > staging size (4 MiB) to exercise multi-chunk staging.
    let n = 5 * 1024 * 1024usize;
    let buf = alloc._alloc(n, &spec, false).expect("device-local alloc");
    assert!(matches!(buf, RawBuffer::AmdDevice { host_ptr: None, .. }), "buffer must be device-only");

    let src: Vec<u8> = (0..n).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
    alloc._copyin(&buf, 0, &src).expect("copyin");
    let mut out = vec![0u8; n];
    alloc._copyout(&mut out, &buf, 0).expect("copyout");
    assert_eq!(src, out, "SDMA host↔device roundtrip must preserve bytes");

    let buf2 = alloc._alloc(n, &spec, false).expect("device-local alloc 2");
    alloc._transfer(&buf2, 0, &buf, 0, n).expect("transfer");
    let mut out2 = vec![0u8; n];
    alloc._copyout(&mut out2, &buf2, 0).expect("copyout 2");
    assert_eq!(src, out2, "SDMA device→device transfer must preserve bytes");

    alloc._free(buf, &spec);
    alloc._free(buf2, &spec);
}

/// On real AQL hardware (multi-XCC CDNA, or gfx11+ under `SVOD_AMD_AQL=1`),
/// `set_aql_scratch` must land the scratch descriptor at the right
/// `amd_queue_t` offsets in the GART page the firmware reads. PM4 queues
/// program scratch via registers and own no descriptor, so there the write is
/// a no-op.
#[test]
fn set_aql_scratch_round_trips_through_gart() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let q = AmdComputeQueue::create(&alloc).expect("create compute queue");
    if q.is_pm4() {
        return;
    }
    let (va, size, tmpring, _rounded, handle, desc) =
        crate::amd::device::alloc_scratch(alloc.dev.core().iface(), &alloc.dev.node, &alloc.dev.arch, 256)
            .expect("alloc scratch");
    assert_ne!(desc, crate::amd::device::AqlScratchDesc::default(), "every AQL arch must synthesize a descriptor");
    q.set_aql_scratch(&desc);
    assert_eq!(q.read_aql_scratch(), desc, "the GART descriptor must match what we wrote");
    assert_eq!((desc.backing_va, desc.tmpring_size), (va, tmpring));
    alloc.dev.core().iface().free_raw(va, size, handle);
}
