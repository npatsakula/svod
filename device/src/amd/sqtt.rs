//! AMD SQTT (thread trace) capture on gfx11/gfx12 PM4 queues.
//!
//! A port of Tinygrad's `ops_amd.py` `sqtt_start` / `sqtt_setup_exec` /
//! `sqtt_stop` / `sqtt_trace` (whose register values follow Mesa's
//! `ac_sqtt.c`), for one traced dispatch at a time. Each SE writes its own
//! window of the trace buffer; after the dispatch every SE's write pointer is
//! copied out, and the handle reads `(wptr & 0x1FFF_FFFF) * 32` bytes of it.

use std::os::fd::{AsRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::Arc;

use snafu::ensure;
use svod_dtype::AmdArch;

use crate::allocator::{Allocator, AmdBufferGuard, BufferSpec, RawBuffer};
use crate::amd::AmdAllocator;
use crate::amd::connector::SubmissionFinalizer;
use crate::amd::regwrite::RegWriter;
use crate::amd::sys::{drm, ioctl, pm4};
use crate::amd::topology::AmdNode;
use crate::error::{Error, Result, SqttUnsupportedSnafu};
use crate::profile::{CodeImage, SeTrace, SqttConfig, SqttTrace};
use crate::sync::DispatchTimestamps;

/// `SQ_TT_RT_FREQ_4096_CLK` (gfx11 `soc` enum): a timestamp token every 4096 clocks.
const RT_FREQ_4096_CLK: u32 = 2;
/// `SQ_TT_WTYPE_INCLUDE_CS_BIT`: trace compute waves only.
const WTYPE_INCLUDE_CS: u32 = 1 << 6;
/// `SQ_TT_TOKEN_MASK_{SQDEC,SHDEC,GFXUDEC,COMP,CONTEXT}_BIT`: the register writes
/// a trace records.
const REG_INCLUDE: u32 = 0x1F;
/// `1 << SQ_TT_TOKEN_EXCLUDE_PERF_SHIFT`, excluded on gfx11.
const TOKEN_EXCLUDE_PERF: u32 = 1 << 11;
/// `1 << SQ_TT_TOKEN_EXCLUDE_{VMEMEXEC,ALUEXEC,VALUINST,IMMEDIATE,INST}_SHIFT`:
/// the instruction tokens an SE outside the `se_mask` leaves out.
const TOKEN_EXCLUDE_ITRACE: u32 = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 5) | (1 << 8);
/// What gfx12 excludes on an SE outside the `se_mask` (Tinygrad hardcodes it:
/// the gfx12 enum lacks most shifts, and the value matches gfx11's).
const TOKEN_EXCLUDE_NO_ITRACE_GFX12: u32 = 0x927;
/// Bits of `SQ_THREAD_TRACE_WPTR.offset`, in 32-byte units.
const WPTR_OFFSET_MASK: u32 = 0x1FFF_FFFF;
/// `COMPUTE_STATIC_THREAD_MGMT_SE0..7`, the per-SE CU masks Tinygrad narrows.
const THREAD_MGMT_SES: u32 = 8;
const BUFFER_ALIGN: usize = 1 << 12;

/// The shader-engine geometry the streams cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SqttTopology {
    pub se: u32,
    pub cu_per_se: u32,
    /// KFD's `cu_per_simd_array`.
    pub cu_per_sa: u32,
}

impl SqttTopology {
    pub fn from_node(node: &AmdNode) -> Self {
        let xcc = node.num_xcc.max(1);
        let se = (node.array_count / node.simd_arrays_per_engine.max(1) / xcc).max(1);
        let cu = node.simd_count / node.simd_per_cu.max(1) / xcc;
        Self { se, cu_per_se: cu / se, cu_per_sa: node.cu_per_simd_array }
    }
}

/// Refuse what the capture does not cover: CDNA (gfx9's stream and tokens
/// differ) and AQL queues, whose dispatch the CP launches without our PM4.
pub fn check_target(arch: AmdArch, pm4_queues: bool) -> Result<()> {
    ensure!(
        matches!(arch.gfx_major(), 11 | 12),
        SqttUnsupportedSnafu { reason: format!("{arch} (only gfx11 and gfx12 are traced)") }
    );
    ensure!(pm4_queues, SqttUnsupportedSnafu { reason: "AQL queues (thread trace needs PM4 queues)" });
    Ok(())
}

pub fn check_config(config: &SqttConfig) -> Result<()> {
    ensure!(config.simd < 4, SqttUnsupportedSnafu { reason: format!("simd {} (a WGP SIMD is 0-3)", config.simd) });
    // BUF0_SIZE holds the window in 4 KiB units in a 22-bit field.
    ensure!(
        config.bytes_per_se > 0
            && config.bytes_per_se.is_multiple_of(BUFFER_ALIGN)
            && config.bytes_per_se >> 12 < 1 << 22,
        SqttUnsupportedSnafu {
            reason: format!("bytes_per_se {} (a nonzero multiple of 4 KiB below 16 GiB)", config.bytes_per_se)
        }
    );
    Ok(())
}

/// The traced dispatch's start and stop PM4 streams.
///
/// SE `i` writes the `bytes_per_se` window at `buf_va + i * bytes_per_se`
/// (4 KiB aligned) and its final write pointer to the `u32` at `wptr_va + 4 * i`.
/// `workgroups` sizes the CU masks of `limit_cu`.
pub fn build_streams(
    config: &SqttConfig,
    arch: AmdArch,
    topology: &SqttTopology,
    workgroups: u64,
    buf_va: u64,
    wptr_va: u64,
) -> Result<(Vec<u32>, Vec<u32>)> {
    check_target(arch, true)?;
    check_config(config)?;
    let gc = RegWriter::for_arch(arch).expect("gfx11 and gfx12 GC registers are vendored");
    let gfx12 = arch.gfx_major() >= 12;
    let window = config.bytes_per_se as u64;
    let cu_masks = if config.limit_cu { Some(cu_masks(topology, workgroups)?) } else { None };

    let mut start = Vec::new();
    start.extend_from_slice(&pm4::acquire_mem());
    spi_config(&gc, &mut start, true);
    for se in 0..topology.se {
        gc.grbm_se(&mut start, se);
        let base = buf_va + se as u64 * window;
        let (lo, hi) = ((base >> 12) as u32, (base >> 44) as u32);
        if gfx12 {
            gc.wreg(&mut start, "regSQ_THREAD_TRACE_BUF0_SIZE", &[("size", (window >> 12) as u32)]);
            gc.wreg_value(&mut start, "regSQ_THREAD_TRACE_BUF0_BASE_LO", lo);
            gc.wreg_value(&mut start, "regSQ_THREAD_TRACE_BUF0_BASE_HI", hi);
        } else {
            gc.wreg(&mut start, "regSQ_THREAD_TRACE_BUF0_SIZE", &[("size", (window >> 12) as u32), ("base_hi", hi)]);
            gc.wreg_value(&mut start, "regSQ_THREAD_TRACE_BUF0_BASE", lo);
        }
        gc.wreg(
            &mut start,
            "regSQ_THREAD_TRACE_MASK",
            &[("wtype_include", WTYPE_INCLUDE_CS), ("simd_sel", config.simd as u32), ("wgp_sel", 0), ("sa_sel", 0)],
        );
        let itrace = (config.se_mask >> se) & 1 == 1;
        let token_exclude = match (gfx12, itrace) {
            (false, true) => TOKEN_EXCLUDE_PERF,
            (false, false) => TOKEN_EXCLUDE_PERF | TOKEN_EXCLUDE_ITRACE,
            (true, true) => 0,
            (true, false) => TOKEN_EXCLUDE_NO_ITRACE_GFX12,
        };
        let mut token_mask =
            vec![("reg_include", REG_INCLUDE), ("token_exclude", token_exclude), ("bop_events_token_include", 1)];
        if gfx12 {
            token_mask.push(("exclude_barrier_wait", 1));
        }
        gc.wreg(&mut start, "regSQ_THREAD_TRACE_TOKEN_MASK", &token_mask);
        trace_ctrl(&gc, &mut start, gfx12, true);
    }
    gc.grbm_broadcast(&mut start);
    gc.wreg(&mut start, "regCOMPUTE_THREAD_TRACE_ENABLE", &[("thread_trace_enable", 1)]);
    start.extend_from_slice(&pm4::acquire_mem());
    if let Some(masks) = &cu_masks {
        thread_mgmt(&gc, &mut start, masks);
    }

    let status = gc.addr("regSQ_THREAD_TRACE_STATUS");
    let status_reg = gc.reg("regSQ_THREAD_TRACE_STATUS");
    let wptr = gc.addr("regSQ_THREAD_TRACE_WPTR");
    let mut stop = Vec::new();
    stop.extend_from_slice(&pm4::event_write(pm4::THREAD_TRACE_MARKER, 0));
    stop.extend_from_slice(&pm4::acquire_mem());
    gc.grbm_broadcast(&mut stop);
    gc.wreg(&mut stop, "regCOMPUTE_THREAD_TRACE_ENABLE", &[("thread_trace_enable", 0)]);
    stop.extend_from_slice(&pm4::event_write(pm4::THREAD_TRACE_FINISH, 0));
    for se in 0..topology.se {
        gc.grbm_se(&mut stop, se);
        stop.extend_from_slice(&pm4::wait_reg_mem_reg(status, 0, field_mask(status_reg, "finish_pending")));
        trace_ctrl(&gc, &mut stop, gfx12, false);
        stop.extend_from_slice(&pm4::wait_reg_mem_reg(status, 0, field_mask(status_reg, "busy")));
        stop.extend_from_slice(&pm4::event_write(pm4::CS_PARTIAL_FLUSH, pm4::EVENT_INDEX_PARTIAL_FLUSH));
        stop.extend_from_slice(&pm4::copy_data_reg_to_mem_confirmed(wptr, wptr_va + se as u64 * 4));
    }
    gc.grbm_broadcast(&mut stop);
    spi_config(&gc, &mut stop, false);
    stop.extend_from_slice(&pm4::acquire_mem());
    if cu_masks.is_some() {
        // The masks are queue state: give later dispatches every CU back.
        thread_mgmt(&gc, &mut stop, &[u32::MAX; THREAD_MGMT_SES as usize]);
    }
    Ok((start, stop))
}

fn spi_config(gc: &RegWriter, out: &mut Vec<u32>, tracing: bool) {
    gc.wreg(
        out,
        "regSPI_CONFIG_CNTL",
        &[
            ("ps_pkr_priority_cntl", 3),
            ("exp_priority_order", 3),
            ("gpr_write_priority", 0x2c688),
            ("enable_sqg_bop_events", tracing as u32),
            ("enable_sqg_top_events", tracing as u32),
        ],
    );
}

fn trace_ctrl(gc: &RegWriter, out: &mut Vec<u32>, gfx12: bool, tracing: bool) {
    let mut fields = vec![
        ("draw_event_en", 1),
        ("spi_stall_en", 1),
        ("sq_stall_en", 1),
        ("reg_at_hwm", 2),
        ("hiwater", 1),
        ("util_timer", 1),
        ("mode", tracing as u32),
    ];
    if !gfx12 {
        fields.push(("rt_freq", RT_FREQ_4096_CLK));
    }
    gc.wreg(out, "regSQ_THREAD_TRACE_CTRL", &fields);
}

fn thread_mgmt(gc: &RegWriter, out: &mut Vec<u32>, masks: &[u32]) {
    for (se, &mask) in masks.iter().enumerate() {
        gc.wreg_value(out, &format!("regCOMPUTE_STATIC_THREAD_MGMT_SE{se}"), mask);
    }
}

fn field_mask(r: &crate::amd::am::regs::RegDef, name: &str) -> u32 {
    let f = r.field(name).unwrap_or_else(|| panic!("register {} has no field {name}", r.name));
    (((1u64 << (f.hi - f.lo + 1)) - 1) << f.lo) as u32
}

/// Tinygrad's `SQTT_LIMIT_SE=1` masks: enough CUs per SE for `workgroups` at
/// four waves a CU, one more on SE 0, kept to the first SA's CUs.
pub fn cu_masks(topology: &SqttTopology, workgroups: u64) -> Result<[u32; THREAD_MGMT_SES as usize]> {
    let sa_bits = topology.cu_per_sa / 2;
    ensure!(
        sa_bits > 0 && topology.cu_per_se > 0,
        SqttUnsupportedSnafu { reason: format!("limit_cu on an unknown CU topology {topology:?}") }
    );
    let ones = |bits: u64| if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 };
    let per_se = workgroups / (topology.cu_per_se as u64 * 4);
    let sa_mask = ones(sa_bits as u64);
    Ok(std::array::from_fn(|se| {
        let cu_mask = ones(per_se + (se == 0) as u64);
        ((cu_mask & sa_mask) | ((cu_mask & (sa_mask << 16)) << 16)) as u32
    }))
}

/// Bytes SE's trace holds given its copied-out write pointer, and whether the
/// window filled. gfx11.0 reports the pointer from address 0 rather than from
/// the window's base.
pub fn trace_bytes(arch: AmdArch, wptr: u32, se_buf_va: u64, window: usize) -> (usize, bool) {
    let mut bytes = (wptr & WPTR_OFFSET_MASK) as u64 * 32;
    if arch.is_rdna3() {
        bytes = bytes.wrapping_sub(((se_buf_va / 32) & WPTR_OFFSET_MASK as u64) * 32);
    }
    if bytes > window as u64 {
        tracing::warn!(bytes, window, "SQTT write pointer past its window");
        return (window, true);
    }
    // The pointer stops at the last 32 bytes when the window overflows.
    (bytes as usize, bytes as usize >= window.saturating_sub(32))
}

/// The buffers and streams of one traced dispatch, before submission.
pub(crate) struct Capture {
    trace: AmdBufferGuard,
    trace_host: NonNull<u8>,
    wptrs: AmdBufferGuard,
    wptrs_host: NonNull<u8>,
    pub(crate) start: Vec<u32>,
    pub(crate) stop: Vec<u32>,
    config: SqttConfig,
    arch: AmdArch,
    ses: u32,
    code: Option<CodeImage>,
}

impl Capture {
    /// The trace buffer is host-visible VRAM, or uncached GTT where none is
    /// left; the write pointers live in uncached GTT.
    pub(crate) fn new(
        allocator: &AmdAllocator,
        config: SqttConfig,
        workgroups: u64,
        code: Option<CodeImage>,
    ) -> Result<Self> {
        let core = allocator.dev.core();
        let arch = core.arch;
        let topology = SqttTopology::from_node(&core.node);
        check_config(&config)?;
        let bytes = config.bytes_per_se * topology.se as usize;
        let spec = BufferSpec { cpu_access: true, nolru: true, ..Default::default() };
        let trace = match allocator.alloc(bytes, &spec, false) {
            Ok(buffer) => buffer,
            Err(error) => {
                tracing::debug!(?error, bytes, "SQTT trace buffer falls back to uncached GTT");
                allocator.alloc_uncached(bytes)?
            }
        };
        let trace = AmdBufferGuard::new(trace);
        let wptrs = AmdBufferGuard::new(allocator.alloc_uncached((topology.se as usize * 4).max(64))?);
        let (trace_va, trace_host) = host_visible(trace.buffer())?;
        let (wptr_va, wptrs_host) = host_visible(wptrs.buffer())?;
        let (start, stop) = build_streams(&config, arch, &topology, workgroups, trace_va, wptr_va)?;
        Ok(Self { trace, trace_host, wptrs, wptrs_host, start, stop, config, arch, ses: topology.se, code })
    }

    /// The handle that reads the trace once `finalizer` retired, delegating
    /// timestamps and counters to `inner`.
    pub(crate) fn into_handle(
        self,
        inner: Arc<dyn DispatchTimestamps>,
        finalizer: Arc<SubmissionFinalizer>,
    ) -> SqttHandle {
        SqttHandle {
            inner,
            finalizer,
            trace: self.trace.into_inner(),
            trace_host: self.trace_host,
            wptrs: self.wptrs.into_inner(),
            wptrs_host: self.wptrs_host,
            config: self.config,
            arch: self.arch,
            ses: self.ses,
            code: self.code,
        }
    }
}

fn host_visible(buffer: &RawBuffer) -> Result<(u64, NonNull<u8>)> {
    match buffer {
        RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(host), .. } => Ok((*gpu_addr, *host)),
        _ => Err(Error::NotHostVisible { what: "SQTT buffer" }),
    }
}

/// A traced dispatch's handle: keeps the trace and write-pointer buffers alive
/// until the trace is read after sync.
pub struct SqttHandle {
    inner: Arc<dyn DispatchTimestamps>,
    finalizer: Arc<SubmissionFinalizer>,
    trace: RawBuffer,
    trace_host: NonNull<u8>,
    wptrs: RawBuffer,
    wptrs_host: NonNull<u8>,
    config: SqttConfig,
    arch: AmdArch,
    ses: u32,
    code: Option<CodeImage>,
}

// SAFETY: the host pointers map allocations the handle owns; they are read
// only once the dispatch that writes them retired.
unsafe impl Send for SqttHandle {}
unsafe impl Sync for SqttHandle {}

impl Drop for SqttHandle {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        if let Err(error) = self.finalizer.wait(30_000) {
            if let RawBuffer::AmdDevice { device, .. } = &self.trace {
                device.core().poison(&error.to_string());
            }
            tracing::warn!(?error, "SqttHandle drop: trace allocations quarantined");
            return;
        }
        self.trace.free_amd_device_in_place();
        self.wptrs.free_amd_device_in_place();
    }
}

impl DispatchTimestamps for SqttHandle {
    fn timestamps_ns(&self) -> Option<(u64, u64)> {
        self.inner.timestamps_ns()
    }

    fn counters(&self) -> Option<crate::profile::CounterSet> {
        self.inner.counters()
    }

    fn sqtt(&self) -> Option<SqttTrace> {
        if !self.finalizer.retired() {
            return None;
        }
        let RawBuffer::AmdDevice { gpu_addr: trace_va, .. } = &self.trace else {
            unreachable!("an SQTT trace buffer is an AMD allocation")
        };
        let window = self.config.bytes_per_se;
        let per_se = (0..self.ses)
            .map(|se| {
                // SAFETY: `wptrs_host` maps `ses` u32s written by COPY_DATA before
                // the retired dispatch's timeline store.
                let wptr = unsafe { (self.wptrs_host.as_ptr() as *const u32).add(se as usize).read_volatile() };
                let offset = se as usize * window;
                let (len, truncated) = trace_bytes(self.arch, wptr, trace_va + offset as u64, window);
                // SAFETY: the SE's window lies inside the `ses * window` buffer.
                let bytes = unsafe { std::slice::from_raw_parts(self.trace_host.as_ptr().add(offset), len) }.to_vec();
                SeTrace { se, itrace: (self.config.se_mask >> se) & 1 == 1, bytes, truncated }
            })
            .collect();
        Some(SqttTrace { arch: self.arch, per_se, code: self.code.clone() })
    }
}

/// An amdgpu DRM context holding the device at `AMDGPU_CTX_STABLE_PSTATE_STANDARD`
/// (Mesa RADV's route for SQTT). Freeing the context restores the level the
/// device had. The kernel's `amdgpu_ctx_ioctl` checks no capability for it;
/// it refuses with `EBUSY` while another context holds the level.
pub(crate) struct DrmStablePstate {
    fd: OwnedFd,
    ctx_id: u32,
}

impl DrmStablePstate {
    pub(crate) fn acquire(render: &OwnedFd) -> Result<Self> {
        let fd = render
            .try_clone()
            .map_err(|e| Error::AmdIoctl { ioctl: "dup(render node)", errno: e.raw_os_error().unwrap_or(0) })?;
        let mut args = drm::drm_amdgpu_ctx {
            in_: drm::drm_amdgpu_ctx_in { op: drm::AMDGPU_CTX_OP_ALLOC_CTX, ..Default::default() },
        };
        ctx_ioctl(&fd, &mut args, "AMDGPU_CTX_OP_ALLOC_CTX")?;
        // SAFETY: ALLOC_CTX filled the `out.alloc` member.
        let ctx_id = unsafe { args.out.alloc.ctx_id };
        let guard = Self { fd, ctx_id };
        let mut args = drm::drm_amdgpu_ctx {
            in_: drm::drm_amdgpu_ctx_in {
                op: drm::AMDGPU_CTX_OP_SET_STABLE_PSTATE,
                flags: drm::AMDGPU_CTX_STABLE_PSTATE_STANDARD,
                ctx_id,
                ..Default::default()
            },
        };
        ctx_ioctl(&guard.fd, &mut args, "AMDGPU_CTX_OP_SET_STABLE_PSTATE")?;
        Ok(guard)
    }
}

impl Drop for DrmStablePstate {
    fn drop(&mut self) {
        let mut args = drm::drm_amdgpu_ctx {
            in_: drm::drm_amdgpu_ctx_in { op: drm::AMDGPU_CTX_OP_FREE_CTX, ctx_id: self.ctx_id, ..Default::default() },
        };
        if let Err(error) = ctx_ioctl(&self.fd, &mut args, "AMDGPU_CTX_OP_FREE_CTX") {
            tracing::warn!(?error, "SQTT stable-pstate context leaked until the process exits");
        }
    }
}

fn ctx_ioctl(fd: &OwnedFd, args: &mut drm::drm_amdgpu_ctx, op: &'static str) -> Result<()> {
    // SAFETY: `fd` is an open amdgpu render node and `args` the ioctl's union.
    unsafe { ioctl::amdgpu_ctx(fd.as_raw_fd(), args) }
        .map(drop)
        .map_err(|e| Error::AmdIoctl { ioctl: op, errno: e as i32 })
}
