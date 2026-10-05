//! AMD SQ hardware performance-counter programming (PMC) for gfx11 (RDNA3.5).
//!
//! The PM4 sequence: program `SQ_PERFCOUNTER*_SELECT`, start/stop the perfmon via
//! `CP_PERFMON_CNTL`, and `COPY_DATA` the per-WGP counters into a GTT buffer —
//! summed across the SE/SA/WGP grid when read back. gfx11-only; callers gate on
//! `target_major == 11` and stable power state (`profile_standard`).

use std::ptr::NonNull;
use std::sync::Arc;

use crate::allocator::RawBuffer;
use crate::amd::am::regs::{self, RegDef};
use crate::amd::connector::SubmissionFinalizer;
use crate::amd::signal::AmdSignal;
use crate::amd::sys::pm4;
use crate::amd::topology::AmdNode;
use crate::profile::{AmdCounter, CounterSet, PmcCounter};

// gfx11_5_0 GC IP segment bases. Verified against Mesa register constants:
// regGRBM_GFX_INDEX → abs 0xC200 dword = 0x30800 byte (R_030800);
// regSQ_PERFCOUNTER0_SELECT → abs 0xD9C0 dword = 0x36700 byte (R_036700).
const GC_BASES: [u64; 2] = [0x1260, 0xA000];

const SET_SH_START: u64 = pm4::PACKET3_SET_SH_REG_START as u64;
const SET_SH_END: u64 = 0x3000;
const SET_UCONFIG_START: u64 = pm4::PACKET3_SET_UCONFIG_REG_START as u64;

/// Whether any AMD GPU is pinned to a stable power state (`profile_standard`),
/// required for meaningful perf-counter values. Scans the DRM sysfs nodes; on a
/// single-GPU host this is exact. Set it with `amd-smi set -l stable_std`.
pub fn stable_pstate() -> bool {
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return false;
    };
    entries.flatten().any(|e| {
        std::fs::read_to_string(e.path().join("device/power_dpm_force_performance_level"))
            .is_ok_and(|s| s.trim() == "profile_standard")
    })
}

/// gfx11 SQ perf-counter selector (perf_sel) for each [`AmdCounter`].
fn sq_perf_sel(c: AmdCounter) -> u32 {
    match c {
        AmdCounter::SqBusyCycles => 3,
        AmdCounter::SqWaves => 4,
        AmdCounter::SqInstsValu => 62,
    }
}

fn reg(name: &str) -> &'static RegDef {
    regs::find(regs::GC_11_5_0, name).unwrap_or_else(|| panic!("gfx11_5_0 register table missing {name}"))
}

/// Absolute dword address (`base[segment] + offset`).
fn abs_addr(r: &RegDef) -> u64 {
    GC_BASES[r.segment as usize] + r.offset as u64
}

/// Emit a register write as a SET_SH_REG or SET_UCONFIG_REG packet, chosen by the
/// register's absolute address window.
fn wreg(out: &mut Vec<u32>, name: &str, fields: &[(&str, u32)]) {
    let r = reg(name);
    let val = r.encode(fields) as u32;
    let a = abs_addr(r);
    if (SET_SH_START..SET_SH_END).contains(&a) {
        out.extend(pm4::set_sh_reg((a - SET_SH_START) as u32, &[val]));
    } else {
        out.extend(pm4::set_uconfig_reg((a - SET_UCONFIG_START) as u32, &[val]));
    }
}

/// `GRBM_GFX_INDEX` set to broadcast all SE/SA/instances (for setup/control writes).
fn grbm_broadcast(out: &mut Vec<u32>) {
    wreg(
        out,
        "regGRBM_GFX_INDEX",
        &[("se_broadcast_writes", 1), ("sa_broadcast_writes", 1), ("instance_broadcast_writes", 1)],
    );
}

/// `GRBM_GFX_INDEX` indexed to one (se, sa, wgp) so the following COPY_DATA reads
/// that WGP's counter. `instance_index` packs the WGP into bits [2+].
fn grbm_index(out: &mut Vec<u32>, se: u32, sa: u32, wgp: u32) {
    wreg(out, "regGRBM_GFX_INDEX", &[("se_index", se), ("sa_index", sa), ("instance_index", wgp << 2)]);
}

/// The SE/SA/WGP grid to iterate for SQ counters, derived from KFD topology.
#[derive(Clone, Copy, Debug)]
pub struct PmcGrid {
    pub se: u32,
    pub sa: u32,
    pub wgp: u32,
}

impl PmcGrid {
    pub fn from_node(node: &AmdNode) -> Self {
        let xcc = node.num_xcc.max(1);
        let sa = node.simd_arrays_per_engine.max(1);
        let se = (node.array_count.max(1) / sa / xcc).max(1);
        let cu_total = (node.simd_count.max(1) / node.simd_per_cu.max(1)) / xcc;
        let cu_per_sa = (cu_total / node.array_count.max(1)).max(1);
        let wgp = (cu_per_sa / 2).max(1); // WGP = 2 CU on RDNA
        // Bound the per-dispatch packet count against pathological topology.
        Self { se, sa, wgp: wgp.min(16) }
    }

    pub fn instances(&self) -> u32 {
        self.se * self.sa * self.wgp
    }
}

/// Bytes of readback buffer needed for `n_counters` over this grid: one 32-bit
/// `_LO` word per (counter, instance).
pub fn readback_bytes(n_counters: usize, grid: &PmcGrid) -> usize {
    n_counters * grid.instances() as usize * 4
}

/// Build the PMC start (program + arm) and read (sample + copy-out) PM4 streams.
///
/// Layout of the readback buffer at `buf_va`: counter `i`'s instance `j` (in
/// se/sa/wgp order) holds one little-endian u32 (`_LO`) at byte
/// `(i * instances + j) * 4`. Only the low 32 bits are read — ample for a
/// profiled kernel (sub-second), and it keeps the per-dispatch PM4 stream within
/// the ring's single-dispatch budget. The read emits `GRBM_GFX_INDEX` once per
/// instance, then copies every counter at that instance.
pub fn build_streams(counters: &[AmdCounter], grid: &PmcGrid, buf_va: u64) -> (Vec<u32>, Vec<u32>) {
    let instances = grid.instances();

    // ── start: stop, program SELECTs + CTRL, enable, start ──
    let mut start = Vec::new();
    grbm_broadcast(&mut start);
    wreg(&mut start, "regCP_PERFMON_CNTL", &[("perfmon_state", 0)]);
    wreg(&mut start, "regSQ_PERFCOUNTER_CTRL", &[("cs_en", 1), ("ps_en", 1), ("gs_en", 1), ("hs_en", 1)]);
    wreg(&mut start, "regSQ_PERFCOUNTER_CTRL2", &[("force_en", 1), ("vmid_en", 0xffff)]);
    for (i, &c) in counters.iter().enumerate() {
        // gfx11 SQ uses even-numbered SELECT registers; readback uses index i.
        wreg(&mut start, &format!("regSQ_PERFCOUNTER{}_SELECT", i * 2), &[("perf_sel", sq_perf_sel(c))]);
    }
    wreg(&mut start, "regCOMPUTE_PERFCOUNT_ENABLE", &[("perfcount_enable", 1)]);
    grbm_broadcast(&mut start);
    wreg(&mut start, "regCP_PERFMON_CNTL", &[("perfmon_state", 0)]);
    wreg(&mut start, "regCP_PERFMON_CNTL", &[("perfmon_state", 1)]);

    // ── read: sample, then copy each counter's _LO at every instance ──
    let los: Vec<u32> =
        counters.iter().enumerate().map(|(i, _)| abs_addr(reg(&format!("regSQ_PERFCOUNTER{i}_LO"))) as u32).collect();
    let mut read = Vec::new();
    grbm_broadcast(&mut read);
    wreg(&mut read, "regCP_PERFMON_CNTL", &[("perfmon_state", 1), ("perfmon_sample_enable", 1)]);
    let mut j = 0u32;
    for se in 0..grid.se {
        for sa in 0..grid.sa {
            for wgp in 0..grid.wgp {
                grbm_index(&mut read, se, sa, wgp);
                for (i, &lo) in los.iter().enumerate() {
                    let off = buf_va + ((i as u32 * instances + j) as u64) * 4;
                    read.extend(pm4::copy_data_reg_to_mem(lo, off));
                }
                j += 1;
            }
        }
    }
    // Stop the perfmon after the final sample so it doesn't keep counting on the
    // GPU after profiling (global CP state — leaving it on perturbs other work).
    grbm_broadcast(&mut read);
    wreg(&mut read, "regCP_PERFMON_CNTL", &[("perfmon_state", 0)]);
    (start, read)
}

/// A profiled-dispatch handle that also carries PMC counters. Holds the GTT
/// readback buffer alive until `counters` is read after sync,
/// and delegates timestamps to the dispatch's timestamp signal.
pub struct PmcHandle {
    ts: Arc<AmdSignal>,
    finalizer: Arc<SubmissionFinalizer>,
    _buf: RawBuffer,
    host: NonNull<u8>,
    counters: Vec<AmdCounter>,
    instances: u32,
}

// SAFETY: `host` points into a GTT allocation owned by `_buf` (kept alive for the
// handle's lifetime); reads happen only after the owning plan synchronizes.
unsafe impl Send for PmcHandle {}
unsafe impl Sync for PmcHandle {}

impl PmcHandle {
    pub(crate) fn new(
        ts: Arc<AmdSignal>,
        finalizer: Arc<SubmissionFinalizer>,
        buf: RawBuffer,
        host: NonNull<u8>,
        counters: Vec<AmdCounter>,
        instances: u32,
    ) -> Self {
        Self { ts, finalizer, _buf: buf, host, counters, instances }
    }
}

impl Drop for PmcHandle {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        if let Err(error) = self.finalizer.wait(30_000) {
            if let RawBuffer::AmdDevice { device, .. } = &self._buf {
                device.core().poison(&error.to_string());
            }
            tracing::warn!(?error, "PmcHandle drop: readback allocation quarantined");
            return;
        }
        self._buf.free_amd_device_in_place();
    }
}

impl crate::sync::DispatchTimestamps for PmcHandle {
    fn timestamps_ns(&self) -> Option<(u64, u64)> {
        self.ts.timestamps_ns()
    }

    fn counters(&self) -> Option<CounterSet> {
        let mut values = std::collections::BTreeMap::new();
        for (i, &c) in self.counters.iter().enumerate() {
            let mut sum = 0u64;
            for j in 0..self.instances {
                // SAFETY: `host` is a live GTT buffer of instances*counters u32s,
                // written by COPY_DATA and flushed before the host reads post-sync.
                let v = unsafe {
                    let p = self.host.as_ptr().add(((i as u32 * self.instances + j) as usize) * 4) as *const u32;
                    p.read_unaligned()
                };
                sum = sum.saturating_add(v as u64);
            }
            values.insert(PmcCounter::Amd(c), sum);
        }
        Some(CounterSet { values })
    }
}
