//! Unit tests for PMC PM4 stream construction (GPU-free).
//!
//! `build_streams` resolves every perf-counter register and chooses SET_SH vs
//! SET_UCONFIG by absolute address — so this exercises the register table and
//! address windows without a GPU (it would have caught a missing `_HI` register).

use super::test_support::{MockAmdIface, mock_device};
use crate::allocator::RawBuffer;
use crate::amd::connector::SubmissionFinalizer;
use crate::amd::pmc::{PmcGrid, PmcHandle, build_streams, readback_bytes};
use crate::error::Error;
use crate::profile::AmdCounter;
use std::sync::Arc;

#[test]
fn readback_sizing() {
    let grid = PmcGrid { se: 2, sa: 2, wgp: 5 };
    assert_eq!(grid.instances(), 20);
    assert_eq!(readback_bytes(3, &grid), 3 * 20 * 4);
}

/// Every counter resolves to a register and a window, and a gfx1151-scale grid
/// with all of them stays well under the 1024-dword single-dispatch ring budget
/// (the readback is the dominant contributor).
#[test]
fn build_streams_resolves_all_registers_within_the_dispatch_budget() {
    let grid = PmcGrid { se: 2, sa: 2, wgp: 5 };
    let (start, read) = build_streams(&AmdCounter::all(), &grid, 0x1_0000);
    assert!(!start.is_empty(), "start stream programs SELECTs + CTRL");
    assert!(!read.is_empty(), "read stream copies counters out");
    assert!(start.len() + read.len() < 900, "pmc streams = {} dwords", start.len() + read.len());
}

fn mock_pmc_handle()
-> (Arc<MockAmdIface>, Arc<crate::amd::signal::SignalPool>, PmcHandle, Arc<crate::amd::signal::AmdSignal>) {
    let (iface, allocator) = mock_device(1);
    let pool = crate::amd::signal::SignalPool::new(&allocator, 64).unwrap();
    let signal = Arc::new(pool.acquire().unwrap());
    let finalizer = SubmissionFinalizer::timeline(Arc::clone(&signal), 1, None);
    let buffer = allocator.alloc_uncached(64).unwrap();
    let host = match &buffer {
        RawBuffer::AmdDevice { host_ptr: Some(host), .. } => *host,
        other => panic!("unexpected readback buffer: {other:?}"),
    };
    let handle = PmcHandle::new(Arc::clone(&signal), finalizer, buffer, host, Vec::new(), 0);
    (iface, pool, handle, signal)
}

#[test]
fn mock_pmc_readback_frees_after_retirement() {
    let (iface, pool, handle, signal) = mock_pmc_handle();
    signal.reset(1);
    drop(handle);
    assert_eq!(iface.counts(), (2, 1, 1));
    drop(signal);
    drop(pool);
    assert_eq!(iface.counts(), (2, 2, 0));
    assert!(iface.free_issues().is_empty());
}

#[test]
fn mock_pmc_failed_drain_poisons_and_quarantines_readback() {
    let (iface, pool, handle, signal) = mock_pmc_handle();
    iface.script_wait(Err(Error::AmdIoctl { ioctl: "mock PMC drain", errno: 5 }));
    drop(handle);
    assert!(signal.wait_signal_value(1, 1).is_err());
    assert_eq!(iface.counts(), (2, 0, 2));
    drop(signal);
    drop(pool);
    assert_eq!(iface.counts(), (2, 0, 2));
}
