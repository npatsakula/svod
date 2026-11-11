use super::test_support::{
    MockAmdCall, MockAmdIface, MockFreeIssue, amd_alloc_or_skip, install_signal_pool, mock_device,
};
use crate::allocator::{Allocator, AmdBufferGuard, BufferSpec, RawBuffer};
use crate::amd::iface::{AllocKind, AmdIface, HUGE_PAGE, reserve_va};
use crate::amd::va_registry::AllocTag;
use crate::error::Error;
use std::sync::Arc;
use test_case::test_case;

/// Live VRAM alloc → free round-trip. Skipped on hosts that can't open an
/// AmdDevice (no GPU, unsupported arch, missing perms).
#[test]
fn alloc_free_roundtrip_if_hw_supports() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let opts = BufferSpec { cpu_access: true, ..Default::default() };
    let buf = alloc.alloc(4096, &opts, /*zero=*/ true).expect("alloc 4 KiB");
    assert_eq!(buf.size(), 4096);
    assert!(buf.cpu_accessible());
    alloc.free(buf, 4096, &opts);
}

#[test_case(4096 => 4096; "a small buffer keeps its pages")]
#[test_case(HUGE_PAGE - 4096 => HUGE_PAGE - 4096; "just below a huge page")]
#[test_case(HUGE_PAGE => HUGE_PAGE; "one huge page")]
#[test_case(HUGE_PAGE + 1 => 2 * HUGE_PAGE; "a partial huge page rounds up")]
#[test_case(35 * HUGE_PAGE / 2 => 18 * HUGE_PAGE; "a large buffer with a tail")]
fn a_data_buffer_takes_whole_huge_pages(size: usize) -> usize {
    let (_iface, alloc) = mock_device(1);
    let opts = BufferSpec { cpu_access: true, ..Default::default() };
    let buffer = alloc.alloc(size, &opts, false).expect("mock allocation");
    let bytes = buffer.size();
    alloc.free(buffer, size, &opts);
    bytes
}

/// The reservation is the VA the GPU maps: large ones start on a huge page
/// whatever their length, and only the reserved range stays mapped.
#[test_case(HUGE_PAGE; "one huge page")]
#[test_case(HUGE_PAGE + 4096; "a page past it")]
#[test_case(35 * HUGE_PAGE / 2; "a large buffer with a tail")]
fn a_large_reservation_starts_on_a_huge_page(size: usize) {
    let va = reserve_va(size).expect("reservation");
    assert_eq!(va as usize % HUGE_PAGE, 0);
    // SAFETY: the range was reserved above and nothing else uses it.
    assert_eq!(unsafe { libc::munmap(va, size) }, 0);
}

#[test]
fn raw_allocation_is_page_aligned_stable_and_freed_once() {
    let (iface, alloc) = mock_device(1);
    let opts = BufferSpec { cpu_access: true, ..Default::default() };
    let buffer = alloc.alloc(17, &opts, true).expect("mock allocation");
    let RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(host_ptr), size, .. } = &buffer else {
        panic!("unexpected buffer: {buffer:?}")
    };
    let (gpu_addr, host_ptr) = (*gpu_addr, *host_ptr);
    assert_eq!(gpu_addr as usize % 0x1000, 0);
    assert_eq!(host_ptr.as_ptr() as u64, gpu_addr);
    assert_eq!(*size, 0x1000, "a sub-page request is rounded up to a whole page");
    unsafe { host_ptr.as_ptr().write(0x5a) };
    assert_eq!(unsafe { host_ptr.as_ptr().read() }, 0x5a);
    assert_eq!((iface.allocation_count(), iface.live_handle_count()), (1, 1));

    alloc.free(buffer, 17, &opts);
    assert_eq!((iface.free_count(), iface.live_handle_count()), (1, 0));
    assert_eq!(
        iface.transcript().iter().map(std::mem::discriminant).collect::<Vec<_>>(),
        [
            std::mem::discriminant(&MockAmdCall::Alloc { size: 0, cpu_access: false, zero: false }),
            std::mem::discriminant(&MockAmdCall::Free { gpu_va: 0, size: 0, handle: 0 }),
        ]
    );
}

/// A construction guard reclaims its buffer when a later step fails, but leaves
/// it mapped whenever the device may still be touching it — a poison latch, or
/// an unwind that abandoned the owning object mid-construction.
#[test_case::test_case(false, false, 1; "error return reclaims")]
#[test_case::test_case(true, false, 0; "poisoned device quarantines")]
#[test_case::test_case(false, true, 0; "panic unwind quarantines")]
fn construction_guard_reclaims_unless_the_device_may_still_read(poison: bool, panicking: bool, frees: usize) {
    let (iface, alloc) = mock_device(1);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> crate::error::Result<()> {
        let _guard = AmdBufferGuard::new(alloc.alloc(64, &BufferSpec::default(), true)?);
        if poison {
            alloc.dev.core().poison("synthetic fault");
        }
        assert!(!panicking, "synthetic construction panic");
        Err(Error::Runtime { message: "later construction step failed".into() })
    }));
    assert_eq!(result.is_err(), panicking);
    assert_eq!(iface.allocation_count(), 1);
    assert_eq!((iface.free_count(), iface.live_handle_count()), (frees, 1 - frees));
}

/// The mock's own free accounting, which every lifecycle test asserts empty.
#[test]
fn mock_detects_double_and_unknown_frees() {
    let iface = MockAmdIface::default();
    let allocation = iface.alloc_raw(1, AllocKind::UncachedGtt, AllocTag::Gtt, true, false).expect("allocation");
    iface.free_raw(allocation.gpu_va, allocation.size, allocation.handle);
    iface.free_raw(allocation.gpu_va, allocation.size, allocation.handle);
    iface.free_raw(0xdead_0000, 0x1000, 999);
    assert_eq!(
        iface.free_issues(),
        [
            MockFreeIssue::DoubleFree { handle: allocation.handle },
            MockFreeIssue::UnknownFree { gpu_va: 0xdead_0000, size: 0x1000, handle: 999 },
        ]
    );
}

#[test]
fn user_buffer_free_failed_drain_poisons_and_quarantines_allocation() {
    let (iface, alloc) = mock_device(1);
    install_signal_pool(&alloc);
    let pool = crate::amd::connector::PoolQueue::new_with_resources(Arc::clone(alloc.dev.core()), &alloc).unwrap();
    let buffer = alloc.alloc(64, &BufferSpec::default(), false).unwrap();
    let allocations = iface.allocation_count();
    pool.next_pm4();
    iface.script_wait(Err(Error::AmdIoctl { ioctl: "mock user-buffer drain", errno: 5 }));

    alloc.free(buffer, 64, &BufferSpec::default());
    assert!(alloc.dev.is_poisoned());
    assert_eq!(iface.allocation_count(), allocations);
    assert_eq!((iface.free_count(), iface.live_handle_count()), (0, allocations));
    drop(pool);
    assert_eq!(iface.free_count(), 0);
}

/// A host store into VRAM through the BAR.
#[derive(Clone, Copy, Debug)]
enum HostVramWrite {
    ZeroFill,
    CopyIn,
    Transfer,
}

/// Each host store into a `cpu_access` VRAM buffer ends on an HDP flush, once
/// its stores are done, and the next submission needs no flush of its own.
#[test_case(HostVramWrite::ZeroFill; "zero-filled allocation")]
#[test_case(HostVramWrite::CopyIn; "copy-in")]
#[test_case(HostVramWrite::Transfer; "host transfer")]
fn a_host_vram_write_is_flushed_before_the_next_submission(write: HostVramWrite) {
    let (iface, alloc) = mock_device(1);
    install_signal_pool(&alloc);
    iface.set_hdp_flush_register();
    let pool =
        crate::amd::connector::PoolQueue::new_with_resources(Arc::clone(alloc.dev.core()), &alloc).expect("pool queue");
    let opts = BufferSpec { cpu_access: true, ..Default::default() };
    let src = alloc.alloc(64, &opts, false).expect("source");
    let dst = alloc.alloc(64, &opts, false).expect("destination");

    let mark = iface.transcript().len();
    let zeroed = match write {
        HostVramWrite::ZeroFill => Some(alloc.alloc(64, &opts, true).expect("zeroed")),
        HostVramWrite::CopyIn => alloc._copyin(&dst, 8, &[7; 16]).map(|()| None).expect("copy-in"),
        HostVramWrite::Transfer => alloc._transfer(&dst, 0, &src, 0, 64).map(|()| None).expect("transfer"),
    };
    let calls = &iface.transcript()[mark..];
    assert_eq!(calls.last(), Some(&MockAmdCall::HdpFlush), "{write:?}: {calls:?}");
    assert_eq!(calls.iter().filter(|call| **call == MockAmdCall::HdpFlush).count(), 1, "{write:?}: {calls:?}");

    let mark = iface.transcript().len();
    let mut submission = crate::hcq::Submission::new(crate::hcq::QueueKind::Compute(0));
    submission.push(crate::hcq::Command::MemoryBarrier);
    pool.queue().submit_hcq_dispatch(&pool, &submission, &[], &[], &[]).expect("submit");
    assert!(!iface.transcript()[mark..].contains(&MockAmdCall::HdpFlush), "{write:?}");
    // Retire the mock timeline, so the frees below drain at once.
    crate::sync::TimelineSignal::set(&**pool.pm4_signal(), pool.pm4_value());

    for buffer in [Some(src), Some(dst), zeroed].into_iter().flatten() {
        alloc.free(buffer, 64, &opts);
    }
}

/// Without SDMA every buffer is host-mapped VRAM, but one the host never
/// stores into needs no flush; a backend without the flush register skips it.
#[test_case(true; "with the register")]
#[test_case(false; "without the register")]
fn only_host_stores_into_vram_flush(register: bool) {
    let (iface, alloc) = mock_device(1);
    if register {
        iface.set_hdp_flush_register();
    }
    let opts = BufferSpec { cpu_access: true, ..Default::default() };
    let unwritten = alloc.alloc(64, &opts, false).expect("allocation");
    let control = alloc.alloc_uncached(64).expect("system memory");
    assert_eq!(iface.call_count(|call| *call == MockAmdCall::HdpFlush), 0);
    let zeroed = alloc.alloc(64, &opts, true).expect("zeroed");
    assert_eq!(iface.call_count(|call| *call == MockAmdCall::HdpFlush), usize::from(register));
    for buffer in [unwritten, control, zeroed] {
        alloc.free(buffer, 64, &opts);
    }
}
