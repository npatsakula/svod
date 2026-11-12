//! Unit tests for PMC PM4 stream construction (GPU-free).
//!
//! `build_streams` resolves every perf-counter register and chooses SET_SH vs
//! SET_UCONFIG by absolute address — so this exercises the register table and
//! address windows without a GPU (it would have caught a missing `_HI` register).

use super::sqtt::{Packet, decode, writes};
use super::test_support::{MockAmdIface, mock_device, mock_device_queues};
use crate::allocator::RawBuffer;
use crate::amd::connector::{OwnerCtx, SubmissionFinalizer};
use crate::amd::pmc::{PmcGrid, PmcHandle, build_streams, check_target, readback_bytes};
use crate::amd::regwrite::RegWriter;
use crate::device::PlanContext;
use crate::error::Error;
use crate::profile::{AmdCounter, CudaCounter, PmcCounter};
use std::sync::Arc;
use svod_dtype::AmdArch;
use test_case::test_case;

#[test]
fn readback_sizing() {
    let grid = PmcGrid { se: 2, sa: 2, wgp: 5 };
    assert_eq!(grid.instances(), 20);
    assert_eq!(readback_bytes(3, &grid), 3 * 20 * 4);
}

/// Every counter resolves to a register and a window, and a gfx1151-scale grid
/// with all of them stays well under the 1024-dword single-dispatch ring budget
/// (the readback is the dominant contributor).
#[test_case(AmdArch::Gfx1100)]
#[test_case(AmdArch::Gfx1151)]
#[test_case(AmdArch::Gfx1201)]
fn build_streams_resolves_all_registers_within_the_dispatch_budget(arch: AmdArch) {
    let grid = PmcGrid { se: 2, sa: 2, wgp: 5 };
    let (start, read) = build_streams(arch, &AmdCounter::all(), &grid, 0x1_0000);
    assert!(!start.is_empty(), "start stream programs SELECTs + CTRL");
    assert!(!read.is_empty(), "read stream copies counters out");
    assert!(start.len() + read.len() < 900, "pmc streams = {} dwords", start.len() + read.len());
}

/// The counters land on the even SELECTs with the target's selectors (gfx12 renumbered
/// `SQ_INSTS_VALU`), the perfmon toggles through `CP_PERFMON_CNTL` (`_1` on gfx12, at the same
/// address), and the read copies counter `i`'s `_LO` at every WGP.
#[test_case(AmdArch::Gfx1151, "regCP_PERFMON_CNTL", [3, 4, 62])]
#[test_case(AmdArch::Gfx1201, "regCP_PERFMON_CNTL_1", [3, 4, 50])]
fn streams_select_the_targets_counters(arch: AmdArch, cntl: &str, selects: [u32; 3]) {
    let gc = RegWriter::for_arch(arch).unwrap();
    assert_eq!(gc.addr(cntl) * 4, 0x36020, "Mesa's R_036020_CP_PERFMON_CNTL");
    let grid = PmcGrid { se: 2, sa: 2, wgp: 4 };
    let (start, read) = build_streams(arch, &AmdCounter::all(), &grid, 0x1_0000);
    let (start, read) = (decode(&start), decode(&read));
    for (i, sel) in selects.into_iter().enumerate() {
        assert_eq!(writes(&gc, &start, &format!("regSQ_PERFCOUNTER{}_SELECT", i * 2)), [sel], "counter {i}");
    }
    assert_eq!(writes(&gc, &start, cntl), [0, 0, 1], "stop, then reset and start");
    assert_eq!(writes(&gc, &read, cntl), [1 | 1 << 10, 0], "sample, then stop");
    let copies = |i: usize| {
        let lo = gc.addr(&format!("regSQ_PERFCOUNTER{i}_LO"));
        read.iter().filter(|p| matches!(p, Packet::Copy { src, .. } if *src == lo)).count()
    };
    assert_eq!([copies(0), copies(1), copies(2)], [16; 3]);
}

#[test_case(AmdArch::Gfx942, true; "cdna")]
#[test_case(AmdArch::Gfx950, true; "cdna4")]
#[test_case(AmdArch::Gfx1201, false; "aql")]
fn unsupported_targets_are_refused(arch: AmdArch, pm4: bool) {
    assert!(matches!(check_target(arch, pm4), Err(Error::PmcUnsupported { .. })));
}

/// Arming keeps the AMD counters on a PM4 device; AQL queues refuse them and arm nothing, while
/// another backend's counters and a disarm stay accepted.
#[test]
fn set_pmc_refuses_aql_queues() {
    let amd: Vec<PmcCounter> = AmdCounter::all().into_iter().map(PmcCounter::Amd).collect();
    let (_iface, allocator) = mock_device(1);
    let owner = OwnerCtx::new(Arc::clone(allocator.dev.core()), allocator);
    assert_eq!(owner.set_pmc(&amd).unwrap(), 3);
    assert_eq!(owner.pmc_counters(), AmdCounter::all());

    let (_iface, allocator) = mock_device_queues(1, false);
    let owner = OwnerCtx::new(Arc::clone(allocator.dev.core()), allocator);
    assert!(matches!(owner.set_pmc(&amd), Err(Error::PmcUnsupported { .. })));
    assert!(owner.pmc_counters().is_empty());
    assert_eq!(owner.set_pmc(&[PmcCounter::Cuda(CudaCounter::DramBytes)]).unwrap(), 0);
    assert_eq!(owner.set_pmc(&[]).unwrap(), 0);
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
