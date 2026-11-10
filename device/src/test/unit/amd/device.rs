use super::test_support::MockAmdIface;
use crate::amd::device::*;
use crate::amd::iface::{AmdIface, QueueTeardown, RingDesc};
use crate::error::Error;
use std::sync::Arc;
use svod_dtype::AmdArch;

/// gfx9 SQ_BUF_RSRC scratch descriptor layout:
///   WORD0 = lo32(va)
///   WORD1 = hi32(va)[15:0] | SWIZZLE_ENABLE(bit31)
///   WORD2 = lo32(size_per_xcc)   (NUM_RECORDS)
///   WORD3 = DST_SEL=XYZW, NUM_FORMAT=UINT, DATA_FORMAT=32, ELEMENT_SIZE=1,
///           INDEX_STRIDE=3, ADD_TID_ENABLE=1
#[test]
fn aql_scratch_descriptor_gfx9_encoding() {
    let va: u64 = 0x1234_5678_9abc_d000;
    let d = AqlScratchDesc::gfx9(va, 0x0004_0000, 0xDEAD, 256);
    assert_eq!(d.resource_descriptor, [0x9abc_d000, 0x8000_5678, 0x0004_0000, 0x00EA_4FAC]);
    assert_eq!((d.backing_va, d.tmpring_size), (va, 0xDEAD));
    assert_eq!(d.wave64_lane_byte_size, 256, "wave64: priv_seg * 64 / 64");
}

/// Both generations record the per-lane stride the backing was sized with,
/// the request rounded to the wave's scratch unit over 64 lanes, as ROCr does.
#[test_case::test_case(AmdArch::Gfx942, 112; "gfx9, 16-byte lanes")]
#[test_case::test_case(AmdArch::Gfx1100, 100; "gfx11, 4-byte lanes")]
fn aql_scratch_lane_size_is_the_rounded_stride(arch: AmdArch, lane: u32) {
    let (iface, allocator) = super::test_support::mock_device(1);
    let iface = Arc::clone(&iface) as Arc<dyn AmdIface>;
    let (.., size_per_thread, _, desc) = alloc_scratch(&iface, &allocator.dev.node, &arch, 100).expect("scratch");
    assert_eq!((size_per_thread, desc.wave64_lane_byte_size), (lane, lane));
}

/// `wave_scratch` is truncated to the arch's WAVESIZE field width; the wave
/// count keeps the low 12 bits regardless.
#[test_case::test_case(AmdArch::Gfx942 => 0x1FFF; "cdna, 13 bits")]
#[test_case::test_case(AmdArch::Gfx1100 => 0x7FFF; "rdna3, 15 bits")]
#[test_case::test_case(AmdArch::Gfx1200 => 0x3FFFF; "rdna4, 18 bits")]
fn pack_tmpring_wavesize_width_by_arch(arch: AmdArch) -> u32 {
    assert_eq!(pack_tmpring(0xABC, 0, &arch) & 0xFFF, 0xABC);
    pack_tmpring(1, 0x3FFFF, &arch) >> 12
}

/// Exclusive lanes fill the bounded hardware pool, then become available for
/// atomic reuse only after their non-clone lease drops.
#[test]
#[ignore = "manual hardware probe; needs a real AMD GPU"]
fn queue_leases_are_exclusive_and_reused() {
    use std::collections::HashSet;
    let Some(alloc) = super::test_support::amd_alloc_or_skip() else { return };
    super::test_support::ensure_hw_signal_pool(&alloc);
    let core = alloc.dev.core();
    let n = core.hw_queues();
    assert!(n >= 1, "hw_queues must be >= 1");
    let mut leases: Vec<_> = (0..n).map(|_| core.lease_queue(&alloc).expect("lease")).collect();
    let distinct: HashSet<_> = leases.iter().map(|lease| lease.queue_ptr()).collect();
    assert_eq!(distinct.len(), n, "simultaneous leases must own distinct queues");
    let released = leases.pop().unwrap().queue_ptr();
    let reused = core.lease_queue(&alloc).expect("reuse released lane");
    assert_eq!(reused.queue_ptr(), released, "a released lane is reused without creating another queue");
}

/// The mock backend's queue bookkeeping, which the queue-lifecycle tests read.
#[test]
fn mock_queue_setup_and_teardown_are_accounted_and_scripted() {
    let iface = Arc::new(MockAmdIface::default());
    let desc = RingDesc {
        ring_gpu: 0x1000,
        gart_gpu: 0x2000,
        wptr_offset: 0,
        rptr_offset: 8,
        eop_gpu: 0,
        eop_size: 0,
        ctx_gpu: 0,
        ctx_save_restore_size: 0,
        ctl_stack_size: 0,
        ring_size: 0x4000,
        gpu_id: 1,
        queue_type: 2,
    };

    iface.script_setup(Err(Error::AmdIoctl { ioctl: "mock setup", errno: 5 }));
    assert!(matches!(iface.setup_ring(&desc), Err(Error::AmdIoctl { ioctl: "mock setup", errno: 5 })));
    let queue = iface.setup_ring(&desc).expect("default setup");
    assert_eq!(queue.doorbell_base.as_ptr() as usize % 0x1000, 0);
    assert_eq!((iface.queue_setup_count(), iface.live_queue_count()), (1, 1));

    iface.script_teardown(Err(Error::AmdIoctl { ioctl: "mock teardown", errno: 16 }));
    assert!(matches!(
        iface.teardown_ring(queue.queue_id, queue.doorbell_base),
        Err(Error::AmdIoctl { ioctl: "mock teardown", errno: 16 })
    ));
    assert_eq!(iface.live_queue_count(), 1, "a failed teardown leaves the queue live");
    iface.script_teardown(Ok(QueueTeardown::Complete));
    assert_eq!(iface.teardown_ring(queue.queue_id, queue.doorbell_base).unwrap(), QueueTeardown::Complete);
    assert_eq!((iface.queue_teardown_count(), iface.live_queue_count()), (1, 0));
}
