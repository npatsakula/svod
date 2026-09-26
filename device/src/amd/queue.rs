//! AMD KFD-direct command queues.
//!
//! - [`AmdComputeQueue`]: 16 MiB AQL ring, doorbell-driven kernel dispatch.
//! - [`AmdCopyQueue`]: SDMA queue for device↔device / device↔host copies.
//!
//! Both share the same KFD `AMDKFD_IOC_CREATE_QUEUE` mechanism but use
//! different `queue_type` codes. AQL packets are 64 bytes (`HsaKernelDispatchPacket`
//! + `HsaBarrierAndPacket`); SDMA submissions are raw dword sequences.
//!
//! Dispatch goes through HCQ command-buffer lowering (single-XCC PM4 ring, fenced on the
//! `PoolQueue`'s monotonic counter) or the native AQL path (multi-XCC CDNA,
//! completion via queue-owned PM4 timeline stores). `AmdCopyQueue::copy_fenced` stages
//! host↔device / device↔device copies via SDMA, fenced on its own timeline.

#![cfg(unix)]

use std::ptr::NonNull;
use std::sync::Arc;

use parking_lot::Mutex;
use tracing::debug;

use crate::allocator::{Allocator, AmdBufferGuard, BufferSpec};

use crate::amd::AmdAllocator;
use crate::amd::connector::{PoolQueue, SubmissionFinalizer};
use crate::amd::device::AmdDeviceCore;
use crate::amd::signal::Timeline;
use crate::amd::sys::hsa::{
    hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM, hsa_kernel_dispatch_packet_t,
    hsa_packet_header_t_HSA_PACKET_HEADER_BARRIER, hsa_packet_header_t_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
    hsa_packet_header_t_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE, hsa_packet_header_t_HSA_PACKET_HEADER_TYPE,
    hsa_packet_type_t_HSA_PACKET_TYPE_INVALID, hsa_packet_type_t_HSA_PACKET_TYPE_VENDOR_SPECIFIC, hsa_signal_t,
    kernel_dispatch_header,
};
use crate::amd::sys::kfd;
use crate::amd::sys::pm4;
use crate::amd::sys::sdma;
use crate::error::{Error, Result};

/// AQL packets are exactly 64 bytes.
pub const AQL_PACKET_BYTES: usize = 64;
/// Packet-header dword (dword 0) with type = INVALID. Written into a ring slot
/// before the body, and used to pre-fill the ring at creation, so the AQL packet
/// processor treats an unpublished/half-written slot as "not yet produced"
/// rather than latching a torn packet (ROCr fills the ring with this).
const INVALID_AQL_HEADER: u32 = hsa_packet_type_t_HSA_PACKET_TYPE_INVALID << hsa_packet_header_t_HSA_PACKET_HEADER_TYPE;
/// 16 MiB ring — the compute-ring default size.
pub const COMPUTE_RING_BYTES: usize = 16 * 1024 * 1024;
/// Match Tinygrad's KFD SDMA ring size. Linked HCQ plans can enqueue long copy
/// bursts asynchronously, unlike the synchronous staging path.
pub const COPY_RING_BYTES: usize = 16 * 1024 * 1024;

/// Conservative upper bound on the dwords a single PM4 dispatch writes to the
/// ring (wait, HDP flush, acquire_mem, the SET_SH_REG stream, DISPATCH_DIRECT,
/// RELEASE_MEM — a typical dispatch is ~150). Bounds in-flight dispatches so
/// the host can never lap the ring.
const MAX_DISPATCH_DWORDS: usize = 1024;
/// Max un-retired dispatches allowed before back-pressure blocks the host.
/// Chosen so the combined ring footprint stays at half the ring even in the
/// worst case (`* MAX_DISPATCH_DWORDS`), leaving generous margin while still
/// letting the host run thousands of dispatches ahead of the GPU.
const RING_MAX_INFLIGHT: u64 = (COMPUTE_RING_BYTES / 4 / MAX_DISPATCH_DWORDS / 2) as u64;

pub(crate) fn validate_pm4_dword_count(dwords: usize) -> Result<()> {
    if dwords == 0 || dwords > MAX_DISPATCH_DWORDS {
        return Err(Error::CommandStreamTooLarge {
            kind: "PM4 ring submission",
            actual: dwords,
            limit: MAX_DISPATCH_DWORDS,
        });
    }
    Ok(())
}

pub(crate) fn build_pm4_indirect_buffer(pm4_addr: u64, pm4_count: usize) -> Result<[u32; 4]> {
    let count = u32::try_from(pm4_count).map_err(|_| Error::CommandStreamTooLarge {
        kind: "PM4 indirect buffer",
        actual: pm4_count,
        limit: pm4::INDIRECT_BUFFER_SIZE_MASK as usize,
    })?;
    if count == 0 || count > pm4::INDIRECT_BUFFER_SIZE_MASK {
        return Err(Error::CommandStreamTooLarge {
            kind: "PM4 indirect buffer",
            actual: pm4_count,
            limit: pm4::INDIRECT_BUFFER_SIZE_MASK as usize,
        });
    }
    Ok([
        pm4::packet3(pm4::PACKET3_INDIRECT_BUFFER, 2),
        pm4_addr as u32,
        (pm4_addr >> 32) as u32,
        count | pm4::INDIRECT_BUFFER_VALID,
    ])
}

/// Build an AQL vendor-specific packet (64 bytes) pointing the AQL packet
/// processor at a PM4 indirect buffer (`pm4_addr`, `pm4_count` dwords). Used to
/// run a PM4 `ACQUIRE_MEM` (full instruction-/scalar-cache + L2 invalidate) on
/// the AQL queue: the AQL header acquire fence covers DATA caches only — NOT the
/// instruction cache — so a code object placed on a recycled VA must be preceded
/// by this explicit invalidate, mirroring ROCr `GpuAgent::InvalidateCodeCaches`
/// at code-object load. The BARRIER bit + system-scope fences serialize each
/// control run around native dispatch packets. Multi-XCC timestamp and timeline
/// writes inside these IBs are predicated to XCC0.
pub fn build_aql_vendor_ib_packet(pm4_addr: u64, pm4_count: u32) -> Result<[u32; 16]> {
    if pm4_count == 0 || pm4_count > pm4::INDIRECT_BUFFER_SIZE_MASK {
        return Err(Error::CommandStreamTooLarge {
            kind: "PM4 indirect buffer",
            actual: pm4_count as usize,
            limit: pm4::INDIRECT_BUFFER_SIZE_MASK as usize,
        });
    }
    let header: u32 = hsa_packet_type_t_HSA_PACKET_TYPE_VENDOR_SPECIFIC
        | (1 << hsa_packet_header_t_HSA_PACKET_HEADER_BARRIER)
        | (hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM << hsa_packet_header_t_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
        | (hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM << hsa_packet_header_t_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    let mut p = [0u32; 16];
    // dw0: AQL header (low 16) | AMD vendor-IB format count = 1 (high 16).
    p[0] = header | (1 << 16);
    p[1] = pm4::packet3(pm4::PACKET3_INDIRECT_BUFFER, 2);
    p[2] = pm4_addr as u32;
    p[3] = (pm4_addr >> 32) as u32;
    p[4] = pm4_count | pm4::INDIRECT_BUFFER_VALID;
    p[5] = 10; // poll interval
    Ok(p)
}

/// Pack a kernel-dispatch packet describing a single launch.
///
/// `kernel_object` = GPU VA of the kernel descriptor (from the loaded code
/// object, via `AmdProgram`).
#[allow(clippy::too_many_arguments)]
pub fn build_dispatch_packet(
    workgroup_size: [u16; 3],
    grid_size: [u32; 3],
    private_segment_size: u32,
    group_segment_size: u32,
    kernel_object: u64,
    kernarg_address: u64,
    completion_signal: u64,
) -> hsa_kernel_dispatch_packet_t {
    build_dispatch_packet_barrier(
        workgroup_size,
        grid_size,
        private_segment_size,
        group_segment_size,
        kernel_object,
        kernarg_address,
        completion_signal,
        /*barrier=*/ true,
    )
}

/// Like [`build_dispatch_packet`] but with explicit control over the header
/// BARRIER bit. When `barrier` is `false` the bit is cleared, so the AQL packet
/// processor does NOT wait for all prior packets to COMPLETE before launching
/// this kernel. All other fields are identical to [`build_dispatch_packet`].
#[allow(clippy::too_many_arguments)]
pub fn build_dispatch_packet_barrier(
    workgroup_size: [u16; 3],
    grid_size: [u32; 3],
    private_segment_size: u32,
    group_segment_size: u32,
    kernel_object: u64,
    kernarg_address: u64,
    completion_signal: u64,
    barrier: bool,
) -> hsa_kernel_dispatch_packet_t {
    let dims: u16 = if grid_size[2] > 1 {
        3
    } else if grid_size[1] > 1 {
        2
    } else {
        1
    };
    let mut p = hsa_kernel_dispatch_packet_t::default();
    // `header` (u16) and `setup` (dims, in bits 0-1) share a union with the
    // `full_header` u32; setting the latter writes both in one little-endian
    // store (header = low half, setup = high half). Clear ONLY the BARRIER bit
    // in the low 16 when `barrier == false`; the `dims << 16` high half and the
    // fence-scope bits are preserved.
    let mut full_header = u32::from(kernel_dispatch_header()) | (u32::from(dims) << 16);
    if !barrier {
        full_header &= !(1u32 << hsa_packet_header_t_HSA_PACKET_HEADER_BARRIER);
    }
    p.__bindgen_anon_1.full_header = full_header;
    p.workgroup_size_x = workgroup_size[0];
    p.workgroup_size_y = workgroup_size[1];
    p.workgroup_size_z = workgroup_size[2];
    p.grid_size_x = grid_size[0];
    p.grid_size_y = grid_size[1];
    p.grid_size_z = grid_size[2];
    p.private_segment_size = private_segment_size;
    p.group_segment_size = group_segment_size;
    p.kernel_object = kernel_object;
    p.kernarg_address = kernarg_address as *mut std::os::raw::c_void;
    p.completion_signal = hsa_signal_t { handle: completion_signal };
    p
}

/// Lower one backend-neutral HCQ compute command to an AMD AQL packet.
/// Geometry conversion is deliberately outside this function: callers provide
/// the exact AQL work-item grid in `ComputeDispatch::grid_size`.
pub fn lower_hcq_compute(command: &crate::hcq::ComputeDispatch) -> Result<hsa_kernel_dispatch_packet_t> {
    let workgroup_size = command.workgroup_size.map(|v| {
        u16::try_from(v).map_err(|_| Error::Runtime { message: format!("AMD HCQ workgroup dimension {v} exceeds u16") })
    });
    let [x, y, z] = workgroup_size;
    Ok(build_dispatch_packet_barrier(
        [x?, y?, z?],
        command.grid_size,
        command.private_segment_size,
        command.group_segment_size,
        command.kernel_object,
        command.kernarg_address,
        command.completion_signal,
        command.barrier,
    ))
}

/// Lower a neutral compute submission to native AQL packets. AQL has no
/// standalone packet for the HCQ memory barrier; its intent is carried by the
/// following dispatch's BARRIER bit and system-scope acquire/release fences.
pub fn lower_hcq_aql(submission: &crate::hcq::Submission) -> Result<Vec<[u32; 16]>> {
    if !matches!(submission.queue, crate::hcq::QueueKind::Compute(_)) {
        return Err(Error::Runtime {
            message: format!("AQL lowering requires a compute queue, got {:?}", submission.queue),
        });
    }
    let mut packets = Vec::new();
    let mut pending_barrier = false;
    for command in &submission.commands {
        match command {
            crate::hcq::Command::MemoryBarrier => pending_barrier = true,
            crate::hcq::Command::Compute(dispatch) => {
                if pending_barrier && !dispatch.barrier {
                    return Err(Error::Runtime {
                        message: "AMD AQL MemoryBarrier requires the following compute dispatch barrier bit".into(),
                    });
                }
                let packet = lower_hcq_compute(dispatch)?;
                let mut dwords = [0u32; 16];
                // SAFETY: the AQL dispatch packet is exactly 16 dwords and POD.
                unsafe { std::ptr::copy_nonoverlapping(&packet as *const _ as *const u32, dwords.as_mut_ptr(), 16) };
                packets.push(dwords);
                pending_barrier = false;
            }
            _ => return Err(unsupported_hcq(submission.queue, command)),
        }
    }
    if pending_barrier {
        return Err(Error::Runtime { message: "AMD AQL submission ends with an unconsumed MemoryBarrier".into() });
    }
    Ok(packets)
}

/// Wrap one ordinary AQL dispatch in the queue-owned timeline protocol.
/// Kernel completion handles must remain unset; optional profiling uses PM4
/// timestamp stores in the surrounding control runs.
pub fn finalize_hcq_aql_timeline_submission(
    submission: &crate::hcq::Submission,
    counter_address: u64,
    previous: u64,
    next: u64,
    timestamps: Option<(u64, u64)>,
) -> Result<crate::hcq::Submission> {
    let computes = submission
        .commands
        .iter()
        .filter_map(|command| match command {
            crate::hcq::Command::Compute(dispatch) => Some(dispatch),
            _ => None,
        })
        .collect::<Vec<_>>();
    if computes.len() != 1 {
        return Err(Error::Runtime {
            message: format!("AMD AQL ordinary dispatch requires exactly one compute command, got {}", computes.len()),
        });
    }
    if computes[0].completion_signal != 0 {
        return Err(Error::Runtime {
            message: "AMD AQL kernel completion must remain unset; the queue timeline owns completion".into(),
        });
    }

    let mut finalized = crate::hcq::Submission::new(submission.queue);
    finalized.push(crate::hcq::Command::Wait { signal_address: counter_address, value: previous });
    for command in &submission.commands {
        if matches!(command, crate::hcq::Command::Compute(_))
            && let Some((start, _)) = timestamps
        {
            finalized.push(crate::hcq::Command::Timestamp { dst: start });
        }
        finalized.push(command.clone());
        if matches!(command, crate::hcq::Command::Compute(_))
            && let Some((_, end)) = timestamps
        {
            finalized.push(crate::hcq::Command::Timestamp { dst: end });
        }
    }
    finalized.push(crate::hcq::Command::Store {
        dst: counter_address,
        value: next,
        scope: crate::hcq::StoreScope::System,
    });
    Ok(finalized)
}

/// Lower AQL while recording native patch sites from the packet layout emitted
/// above. Completion is deliberately not patchable: queue-owned PM4 timeline
/// stores finalize every supported AQL submission.
pub fn lower_hcq_aql_command_buffer(submission: &crate::hcq::Submission) -> Result<crate::hcq::LoweredCommandBuffer> {
    use crate::hcq::{Command, CommandField as F, PatchEncoding, PatchSite};

    let packets = lower_hcq_aql(submission)?;
    let mut sites = Vec::new();
    let mut consumed = std::collections::BTreeSet::new();
    let mut packet = 0usize;
    for (command_index, command) in submission.commands.iter().enumerate() {
        if let Command::Compute(dispatch) = command {
            if dispatch.completion_signal != 0 {
                return Err(Error::Runtime {
                    message: "AMD linked AQL kernel completion must remain unset; the queue timeline owns completion"
                        .into(),
                });
            }
            let base = packet * AQL_PACKET_BYTES;
            let mut site = |field, byte_offset, encoding| {
                if let Some(source) = command_binding(submission, command_index, field) {
                    sites.push(PatchSite { byte_offset: base + byte_offset, encoding, source, addend: 0 });
                    consumed.insert((command_index, field));
                }
            };
            for axis in 0..3u8 {
                site(F::ComputeWorkgroup(axis), 4 + axis as usize * 2, PatchEncoding::U16);
                site(F::ComputeGrid(axis), 12 + axis as usize * 4, PatchEncoding::U32);
            }
            site(F::ComputeKernelObject, 32, PatchEncoding::U64);
            site(F::ComputeKernargAddress, 40, PatchEncoding::U64);
            packet += 1;
        }
    }
    if consumed.len() != submission.patches().len() {
        let missing = submission.patches().iter().find(|p| !consumed.contains(&(p.command, p.field))).unwrap();
        return Err(Error::Runtime {
            message: format!("AMD AQL lowering cannot patch {:?} on command {}", missing.field, missing.command),
        });
    }

    let mut bytes = Vec::with_capacity(packets.len() * AQL_PACKET_BYTES);
    for packet in packets {
        bytes.extend_from_slice(dwords_as_bytes(&packet));
    }
    Ok(crate::hcq::LoweredCommandBuffer { bytes, patches: crate::hcq::PatchTable::from_sites(sites) })
}

/// AQL submission program matching Tinygrad's `AMDComputeAQLQueue._prep_aql`:
/// native dispatch packets stay in the AQL stream while every contiguous run of
/// PM4-only HCQ commands is stored in resident memory and invoked by a barriered
/// vendor-IB packet. This is what gives AQL queues arbitrary memory wait/store
/// semantics without inventing unsupported architected AQL packet forms.
pub struct LoweredAqlSubmissionProgram {
    pub aql: crate::hcq::LoweredCommandBuffer,
    pub control: crate::hcq::LoweredCommandBuffer,
}

pub fn lower_hcq_aql_submission_program(
    submission: &crate::hcq::Submission,
    state: Pm4LoweringState,
    control_address: crate::hcq::PatchSource,
) -> Result<LoweredAqlSubmissionProgram> {
    use crate::hcq::{Command, LoweredCommandBuffer, PatchEncoding, PatchSite, PatchTable, Submission};

    if !matches!(submission.queue, crate::hcq::QueueKind::Compute(_)) {
        return Err(Error::Runtime { message: "AQL submission program requires a compute queue".into() });
    }

    let mut aql_bytes = Vec::new();
    let mut aql_sites = Vec::new();
    let mut control_bytes = Vec::new();
    let mut control_sites = Vec::new();
    let mut run = Submission::new(submission.queue);

    let flush_run = |run: &mut Submission,
                     aql_bytes: &mut Vec<u8>,
                     aql_sites: &mut Vec<PatchSite>,
                     control_bytes: &mut Vec<u8>,
                     control_sites: &mut Vec<PatchSite>|
     -> Result<()> {
        if run.commands.is_empty() {
            return Ok(());
        }
        let lowered = lower_hcq_pm4_command_buffer(run, state)?;
        let control_offset = control_bytes.len();
        control_bytes.extend_from_slice(&lowered.bytes);
        for mut site in lowered.patches.link.into_iter().chain(lowered.patches.runtime).chain(lowered.patches.system) {
            site.byte_offset += control_offset;
            control_sites.push(site);
        }

        let packet_offset = aql_bytes.len();
        let pm4_count = u32::try_from(lowered.bytes.len() / 4).map_err(|_| Error::CommandStreamTooLarge {
            kind: "PM4 indirect buffer",
            actual: lowered.bytes.len() / 4,
            limit: pm4::INDIRECT_BUFFER_SIZE_MASK as usize,
        })?;
        let packet = build_aql_vendor_ib_packet(0, pm4_count)?;
        aql_bytes.extend_from_slice(dwords_as_bytes(&packet));
        aql_sites.push(PatchSite {
            byte_offset: packet_offset + 8,
            encoding: PatchEncoding::U64,
            source: control_address,
            addend: control_offset as u64,
        });
        run.clear();
        Ok(())
    };

    for (command_index, command) in submission.commands.iter().enumerate() {
        if matches!(command, Command::Compute(_)) {
            flush_run(&mut run, &mut aql_bytes, &mut aql_sites, &mut control_bytes, &mut control_sites)?;
            let mut one = Submission::new(submission.queue);
            one.push(command.clone());
            for patch in submission.patches().iter().filter(|patch| patch.command == command_index) {
                one.bind(0, patch.field, patch.source)?;
            }
            let lowered = lower_hcq_aql_command_buffer(&one)?;
            let offset = aql_bytes.len();
            aql_bytes.extend_from_slice(&lowered.bytes);
            for mut site in
                lowered.patches.link.into_iter().chain(lowered.patches.runtime).chain(lowered.patches.system)
            {
                site.byte_offset += offset;
                aql_sites.push(site);
            }
        } else {
            let out = run.commands.len();
            run.push(command.clone());
            for patch in submission.patches().iter().filter(|patch| patch.command == command_index) {
                run.bind(out, patch.field, patch.source)?;
            }
        }
    }
    flush_run(&mut run, &mut aql_bytes, &mut aql_sites, &mut control_bytes, &mut control_sites)?;

    Ok(LoweredAqlSubmissionProgram {
        aql: LoweredCommandBuffer { bytes: aql_bytes, patches: PatchTable::from_sites(aql_sites) },
        control: LoweredCommandBuffer { bytes: control_bytes, patches: PatchTable::from_sites(control_sites) },
    })
}

/// Host-known state needed while lowering a neutral command buffer to raw PM4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pm4LoweringState {
    pub scratch_address: u64,
    pub tmpring_size: u32,
    pub target_major: u32,
    /// Restrict PM4 timestamp and timeline writes to one XCC. This is required
    /// for PM4 control runs launched through a multi-XCC AQL vendor IB.
    pub completion_xcc_mask: Option<u32>,
    /// Address of the device's KFD queue-event mailbox, when it has one. A
    /// `Store` to exactly this address is the interrupt companion of a queue
    /// timeline store and lowers to an interrupting `RELEASE_MEM` carrying the
    /// event id in `ctxid`; every other `Store` stays a plain memory write.
    pub queue_event_mailbox: Option<u64>,
}

/// Append Tinygrad's KFD interrupt companion to a just-pushed queue-timeline
/// store (`ops_amd.py:391-393`): write the event id into the device's event
/// mailbox so a host blocked in `WAIT_EVENTS` wakes on completion instead of at
/// its next poll tier. No-op on backends without a KFD event.
fn push_queue_event_mailbox(submission: &mut crate::hcq::Submission, core: &AmdDeviceCore) {
    if let Some(mailbox) = core.queue_event_mailbox() {
        submission.push(crate::hcq::Command::Store {
            dst: mailbox.address,
            value: mailbox.event_id.into(),
            scope: crate::hcq::StoreScope::System,
        });
    }
}

/// Queue-finalized dispatch result. The queue owns timestamp allocation and
/// packet insertion; callers only retain the resulting handle until collection.
pub(crate) struct HcqDispatchResult {
    pub(crate) finalizer: Arc<SubmissionFinalizer>,
    pub(crate) timestamps: Option<Arc<crate::amd::signal::AmdSignal>>,
}

fn unsupported_hcq(queue: crate::hcq::QueueKind, command: &crate::hcq::Command) -> Error {
    Error::Runtime { message: format!("AMD {queue:?} does not support HCQ command {command:?}") }
}

fn value_u32(kind: &str, value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Runtime {
        message: format!("AMD {kind} value {value} exceeds the 32-bit hardware timeline comparison field"),
    })
}

/// Lower a complete neutral compute submission to one ordered PM4 stream.
/// Queue-invalid packet forms are errors rather than silently changing engines.
pub fn lower_hcq_pm4(submission: &crate::hcq::Submission, state: Pm4LoweringState) -> Result<Vec<u32>> {
    if !matches!(submission.queue, crate::hcq::QueueKind::Compute(_)) {
        return Err(Error::Runtime {
            message: format!("PM4 lowering requires a compute queue, got {:?}", submission.queue),
        });
    }
    let is_gfx9 = state.target_major == 9;
    let mut q = Vec::new();
    for command in &submission.commands {
        match command {
            crate::hcq::Command::Wait { signal_address, value } => {
                q.extend_from_slice(&pm4::wait_reg_mem(*signal_address, value_u32("wait", *value)?, u32::MAX));
            }
            crate::hcq::Command::MemoryBarrier => {
                q.extend_from_slice(&pm4::hdp_flush());
                if is_gfx9 {
                    q.extend_from_slice(&pm4::acquire_mem_gfx9());
                } else {
                    q.extend_from_slice(&pm4::acquire_mem());
                }
            }
            crate::hcq::Command::Store { dst, value, scope } => {
                if let Some(mask) = state.completion_xcc_mask {
                    q.extend_from_slice(&pm4::pred_exec(mask, 8));
                }
                if state.queue_event_mailbox == Some(*dst) {
                    q.extend_from_slice(&pm4::release_mem_event(*dst, value_u32("event id", *value)?, is_gfx9));
                } else {
                    // Work later on this queue reads through the same L2: a PM4
                    // queue has one XCC and every dispatch first invalidates the
                    // CU caches, and an AQL dispatch carries its own system-scope
                    // fences. Only a store someone else waits on writes L2 back.
                    let flush = *scope == crate::hcq::StoreScope::System;
                    q.extend_from_slice(&pm4::release_mem_write(*dst, *value, true, flush, false, is_gfx9));
                }
            }
            crate::hcq::Command::Timestamp { dst } => {
                // Tinygrad: EOP drain, clock write, then acquire the timestamp.
                let mut timestamp = Vec::new();
                timestamp.extend_from_slice(&pm4::release_mem_order(is_gfx9));
                timestamp.extend_from_slice(&pm4::release_mem_timestamp(*dst, is_gfx9));
                if is_gfx9 {
                    timestamp.extend_from_slice(&pm4::acquire_mem_gfx9());
                } else {
                    timestamp.extend_from_slice(&pm4::acquire_mem());
                }
                if let Some(mask) = state.completion_xcc_mask {
                    q.extend_from_slice(&pm4::pred_exec(mask, timestamp.len() as u32));
                }
                q.extend(timestamp);
            }
            crate::hcq::Command::Compute(dispatch) => {
                let native = dispatch.amd_pm4.as_ref().ok_or_else(|| Error::Runtime {
                    message: "AMD PM4 compute requires ComputeDispatch::amd_pm4 metadata".into(),
                })?;
                if native.target_major != state.target_major {
                    return Err(Error::Runtime {
                        message: format!(
                            "AMD PM4 command targets gfx{}, queue lowering targets gfx{}",
                            native.target_major, state.target_major
                        ),
                    });
                }
                let mut user_data = Vec::with_capacity(if native.enable_private_segment_sgpr { 6 } else { 2 });
                if native.enable_private_segment_sgpr {
                    user_data.extend_from_slice(&[
                        state.scratch_address as u32,
                        (state.scratch_address >> 32) as u32 | (1 << 31),
                        u32::MAX,
                        0x20c1_4000,
                    ]);
                }
                user_data
                    .extend_from_slice(&[dispatch.kernarg_address as u32, (dispatch.kernarg_address >> 32) as u32]);
                build_exec_pm4(
                    &mut q,
                    native.rsrc[0],
                    native.rsrc[1],
                    native.rsrc[2],
                    native.program_address,
                    &user_data,
                    state.scratch_address,
                    state.tmpring_size,
                    dispatch.workgroup_size,
                    native.workgroup_count,
                    native.wave32,
                    native.target_major,
                );
            }
            crate::hcq::Command::Copy { .. } | crate::hcq::Command::Execute { .. } => {
                return Err(unsupported_hcq(submission.queue, command));
            }
        }
    }
    Ok(q)
}

/// Lower a complete neutral copy submission to one ordered SDMA stream.
pub fn lower_hcq_sdma(
    submission: &crate::hcq::Submission,
    target_major: u32,
    queue_event_mailbox: Option<u64>,
) -> Result<Vec<u32>> {
    if !matches!(submission.queue, crate::hcq::QueueKind::Copy(_)) {
        return Err(Error::Runtime {
            message: format!("SDMA lowering requires a copy queue, got {:?}", submission.queue),
        });
    }
    let mut q = Vec::new();
    for command in &submission.commands {
        match command {
            crate::hcq::Command::Wait { signal_address, value } => {
                q.extend_from_slice(&sdma::poll_regmem_geq(*signal_address, value_u32("wait", *value)?));
            }
            // Tinygrad defines memory barriers only for compute queues. The
            // scheduler emits this common prologue on first queue use; SDMA FIFO
            // ordering makes it an empty packet sequence.
            crate::hcq::Command::MemoryBarrier => {}
            crate::hcq::Command::Copy { dst, src, bytes } => {
                let mut off = 0usize;
                while off < *bytes {
                    let n = (*bytes - off).min(sdma::SDMA_MAX_COPY_BYTES);
                    q.extend_from_slice(&sdma::copy_linear(*src + off as u64, *dst + off as u64, n));
                    off += n;
                }
            }
            crate::hcq::Command::Timestamp { dst } => q.extend_from_slice(&sdma::timestamp_global(*dst)),
            crate::hcq::Command::Store { dst, value, .. } => {
                let value = value_u32("store", *value)?;
                q.extend_from_slice(&sdma::fence(*dst, value, target_major));
                if queue_event_mailbox == Some(*dst) {
                    q.extend_from_slice(&sdma::trap(value));
                }
            }
            crate::hcq::Command::Compute(_) | crate::hcq::Command::Execute { .. } => {
                return Err(unsupported_hcq(submission.queue, command));
            }
        }
    }
    Ok(q)
}

fn dwords_to_le_bytes(dwords: &[u32]) -> Vec<u8> {
    dwords.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn record_u64_sites(
    sites: &mut Vec<crate::hcq::PatchSite>,
    dword: usize,
    source: crate::hcq::PatchSource,
    addend: u64,
) {
    sites.push(crate::hcq::PatchSite {
        byte_offset: dword * 4,
        encoding: crate::hcq::PatchEncoding::Low32,
        source,
        addend,
    });
    sites.push(crate::hcq::PatchSite {
        byte_offset: (dword + 1) * 4,
        encoding: crate::hcq::PatchEncoding::High32,
        source,
        addend,
    });
}

fn command_binding(
    submission: &crate::hcq::Submission,
    command: usize,
    field: crate::hcq::CommandField,
) -> Option<crate::hcq::PatchSource> {
    submission.patches().iter().find(|patch| patch.command == command && patch.field == field).map(|patch| patch.source)
}

/// Lower PM4 and return backend-originated scatter metadata. Packet offsets are
/// recorded from the exact packet append positions below, never inferred by
/// searching the resulting byte stream.
pub fn lower_hcq_pm4_command_buffer(
    submission: &crate::hcq::Submission,
    state: Pm4LoweringState,
) -> Result<crate::hcq::LoweredCommandBuffer> {
    use crate::hcq::{Command, CommandField as F, PatchEncoding, PatchSite};

    let dwords = lower_hcq_pm4(submission, state)?;
    let mut sites = Vec::new();
    let mut consumed = std::collections::BTreeSet::new();
    let mut cursor = 0usize;
    for (command_index, command) in submission.commands.iter().enumerate() {
        let mut one = crate::hcq::Submission::new(submission.queue);
        one.push(command.clone());
        let command_len = lower_hcq_pm4(&one, state)?.len();
        macro_rules! pair {
            ($field:expr, $dword:expr) => {
                if let Some(source) = command_binding(submission, command_index, $field) {
                    record_u64_sites(&mut sites, cursor + $dword, source, 0);
                    consumed.insert((command_index, $field));
                }
            };
        }
        macro_rules! word {
            ($field:expr, $dword:expr) => {
                if let Some(source) = command_binding(submission, command_index, $field) {
                    sites.push(PatchSite {
                        byte_offset: (cursor + $dword) * 4,
                        encoding: PatchEncoding::U32,
                        source,
                        addend: 0,
                    });
                    consumed.insert((command_index, $field));
                }
            };
        }
        match command {
            Command::Wait { .. } => {
                pair!(F::WaitAddress, 2);
                word!(F::WaitValue, 4);
            }
            Command::Store { .. } => {
                let pred = usize::from(state.completion_xcc_mask.is_some()) * 2;
                pair!(F::StoreDst, pred + 3);
                pair!(F::StoreValue, pred + 5);
            }
            Command::Timestamp { .. } => {
                let pred = usize::from(state.completion_xcc_mask.is_some()) * 2;
                pair!(F::TimestampDst, pred + 11);
            }
            Command::Compute(dispatch) => {
                let native = dispatch.amd_pm4.as_ref().expect("validated by lower_hcq_pm4");
                let acquire = if state.target_major == 9 { 7 } else { 8 };
                if let Some(source) = command_binding(submission, command_index, F::ComputeProgramAddress) {
                    sites.push(PatchSite {
                        byte_offset: (cursor + acquire + 2) * 4,
                        encoding: PatchEncoding::Low32ShiftRight(8),
                        source,
                        addend: 0,
                    });
                    sites.push(PatchSite {
                        byte_offset: (cursor + acquire + 3) * 4,
                        encoding: PatchEncoding::High32ShiftRight(8),
                        source,
                        addend: 0,
                    });
                    consumed.insert((command_index, F::ComputeProgramAddress));
                }
                let scratch_packet = acquire + 4 + 4 + 3 + 3;
                if let Some(source) = command_binding(submission, command_index, F::ComputeScratchTmpring) {
                    sites.push(PatchSite {
                        byte_offset: (cursor + scratch_packet - 1) * 4,
                        encoding: PatchEncoding::U32,
                        source,
                        addend: 0,
                    });
                    consumed.insert((command_index, F::ComputeScratchTmpring));
                }
                if let Some(source) = command_binding(submission, command_index, F::ComputeScratchAddress) {
                    sites.push(PatchSite {
                        byte_offset: (cursor + scratch_packet + 2) * 4,
                        encoding: PatchEncoding::Low32ShiftRight(8),
                        source,
                        addend: 0,
                    });
                    sites.push(PatchSite {
                        byte_offset: (cursor + scratch_packet + 3) * 4,
                        encoding: PatchEncoding::High32ShiftRight(8),
                        source,
                        addend: 0,
                    });
                    consumed.insert((command_index, F::ComputeScratchAddress));
                }
                let user_packet = scratch_packet + 4 + 5;
                if native.enable_private_segment_sgpr
                    && let Some(source) = command_binding(submission, command_index, F::ComputeScratchAddress)
                {
                    sites.push(PatchSite {
                        byte_offset: (cursor + user_packet + 2) * 4,
                        encoding: PatchEncoding::Low32,
                        source,
                        addend: 0,
                    });
                    sites.push(PatchSite {
                        byte_offset: (cursor + user_packet + 3) * 4,
                        encoding: PatchEncoding::High32Or(1 << 31),
                        source,
                        addend: 0,
                    });
                }
                let kernarg_word = user_packet + 2 + usize::from(native.enable_private_segment_sgpr) * 4;
                pair!(F::ComputeKernargAddress, kernarg_word);
                let resource_packet = user_packet + 2 + if native.enable_private_segment_sgpr { 6 } else { 2 };
                let local_packet = resource_packet + 3;
                for axis in 0..3u8 {
                    word!(F::ComputeWorkgroup(axis), local_packet + 5 + axis as usize);
                }
                let dispatch_packet = local_packet + 10;
                for axis in 0..3u8 {
                    word!(F::ComputeGrid(axis), dispatch_packet + 1 + axis as usize);
                }
            }
            Command::MemoryBarrier => {}
            Command::Copy { .. } | Command::Execute { .. } => unreachable!("validated by lower_hcq_pm4"),
        }
        cursor += command_len;
    }
    if consumed.len() != submission.patches().len() {
        let missing =
            submission.patches().iter().find(|patch| !consumed.contains(&(patch.command, patch.field))).unwrap();
        return Err(Error::Runtime {
            message: format!("AMD PM4 lowering cannot patch {:?} on command {}", missing.field, missing.command),
        });
    }
    Ok(crate::hcq::LoweredCommandBuffer {
        bytes: dwords_to_le_bytes(&dwords),
        patches: crate::hcq::PatchTable::from_sites(sites),
    })
}

/// SDMA counterpart to [`lower_hcq_pm4_command_buffer`]. Chunked copies emit a
/// pair of address sites per chunk with the chunk offset as lowering metadata.
pub fn lower_hcq_sdma_command_buffer(
    submission: &crate::hcq::Submission,
    target_major: u32,
    queue_event_mailbox: Option<u64>,
) -> Result<crate::hcq::LoweredCommandBuffer> {
    use crate::hcq::{Command, CommandField as F, PatchEncoding, PatchSite};

    let dwords = lower_hcq_sdma(submission, target_major, queue_event_mailbox)?;
    let mut sites = Vec::new();
    let mut consumed = std::collections::BTreeSet::new();
    let mut cursor = 0usize;
    for (command_index, command) in submission.commands.iter().enumerate() {
        match command {
            Command::Wait { .. } => {
                if let Some(source) = command_binding(submission, command_index, F::WaitAddress) {
                    record_u64_sites(&mut sites, cursor + 1, source, 0);
                    consumed.insert((command_index, F::WaitAddress));
                }
                if let Some(source) = command_binding(submission, command_index, F::WaitValue) {
                    sites.push(PatchSite {
                        byte_offset: (cursor + 3) * 4,
                        encoding: PatchEncoding::U32,
                        source,
                        addend: 0,
                    });
                    consumed.insert((command_index, F::WaitValue));
                }
                cursor += 6;
            }
            Command::MemoryBarrier => {}
            Command::Copy { bytes, .. } => {
                // Bindings are consumed by the command, not by its chunks: a
                // zero-byte copy emits no packets (tinygrad `AMDCopyQueue.copy`
                // loops over `ceil(size / max_copy_size)` chunks the same way,
                // `ops_amd.py:474-484`) yet still has to satisfy the arity check
                // below, or a linked capture containing one hard-fails.
                let src = command_binding(submission, command_index, F::CopySrc);
                let dst = command_binding(submission, command_index, F::CopyDst);
                consumed.extend(src.map(|_| (command_index, F::CopySrc)));
                consumed.extend(dst.map(|_| (command_index, F::CopyDst)));
                let mut offset = 0usize;
                while offset < *bytes {
                    if let Some(source) = src {
                        record_u64_sites(&mut sites, cursor + 3, source, offset as u64);
                    }
                    if let Some(source) = dst {
                        record_u64_sites(&mut sites, cursor + 5, source, offset as u64);
                    }
                    let n = (*bytes - offset).min(sdma::SDMA_MAX_COPY_BYTES);
                    offset += n;
                    cursor += 7;
                }
            }
            Command::Timestamp { .. } => {
                if let Some(source) = command_binding(submission, command_index, F::TimestampDst) {
                    record_u64_sites(&mut sites, cursor + 1, source, 0);
                    consumed.insert((command_index, F::TimestampDst));
                }
                cursor += 3;
            }
            Command::Store { dst, .. } => {
                if let Some(source) = command_binding(submission, command_index, F::StoreDst) {
                    record_u64_sites(&mut sites, cursor + 1, source, 0);
                    consumed.insert((command_index, F::StoreDst));
                }
                if let Some(source) = command_binding(submission, command_index, F::StoreValue) {
                    sites.push(PatchSite {
                        byte_offset: (cursor + 3) * 4,
                        encoding: PatchEncoding::U32,
                        source,
                        addend: 0,
                    });
                    consumed.insert((command_index, F::StoreValue));
                }
                cursor += 4 + 2 * usize::from(queue_event_mailbox == Some(*dst));
            }
            Command::Compute(_) | Command::Execute { .. } => unreachable!("validated by lower_hcq_sdma"),
        }
    }
    if consumed.len() != submission.patches().len() {
        let missing =
            submission.patches().iter().find(|patch| !consumed.contains(&(patch.command, patch.field))).unwrap();
        return Err(Error::Runtime {
            message: format!("AMD SDMA lowering cannot patch {:?} on command {}", missing.field, missing.command),
        });
    }
    Ok(crate::hcq::LoweredCommandBuffer {
        bytes: dwords_to_le_bytes(&dwords),
        patches: crate::hcq::PatchTable::from_sites(sites),
    })
}

/// Compute queue. Wraps either a `KFD_IOC_QUEUE_TYPE_COMPUTE` (PM4) ring on
/// single-XCC GPUs (gfx11/12 default) or a `KFD_IOC_QUEUE_TYPE_COMPUTE_AQL`
/// ring on multi-XCC CDNA. The two paths share the same KFD setup, doorbell
/// mapping, and submit primitive — the only differences are the packet
/// format we write into the ring and whether the GART contains an
/// `amd_queue_t` AQL descriptor.
///
/// The lane lease prevents co-tenant publication. The mutex additionally makes
/// the safe Rust API sound until publication methods take a tokenized mutable
/// lease directly.
pub struct AmdComputeQueue {
    inner: Mutex<QueueInner>,
    state: QueueState,
    /// Immutable device identity (kfd_fd, drm_fd, node, arch, poison latch).
    core: Arc<AmdDeviceCore>,
    /// `true` when this queue submits raw PM4 dwords directly; `false` when
    /// it submits AQL packets (with PM4 wrapped in AQL vendor IB packets).
    /// Decided at queue creation from `num_xcc`, fixed for the queue's lifetime.
    is_pm4: bool,
}

/// Copy queue (SDMA). Stages host↔device and device↔device copies for
/// device-local VRAM buffers. Each [`copy_fenced`](AmdCopyQueue::copy_fenced)
/// is serialised under `inner` and fenced on its own [`Timeline`]: the SDMA
/// `fence` packet writes the timeline value into the GTT signal slot the host
/// busy-polls, so completion needs no interrupt/TRAP.
pub struct AmdCopyQueue {
    inner: Mutex<QueueInner>,
    state: QueueState,
    core: Arc<AmdDeviceCore>,
    timeline: Arc<Timeline>,
    /// Host-visible GTT bounce buffer for host↔device staging. A device-local
    /// VRAM buffer has no host mapping, so `_copyin`/`_copyout` memcpy through
    /// this and DMA the other leg. Locked for the whole chunked transfer so
    /// concurrent copies don't clobber it.
    staging: Mutex<StagingBuf>,
    /// Finalizers for linked-plan SDMA work this queue published. Those plans
    /// advance their own plan-local timeline rather than `timeline`, so without
    /// this the queue could be torn down under live SDMA traffic. Pruned on
    /// every registration, so a long-lived queue never accumulates retired
    /// entries. (Tinygrad needs no equivalent: it has one timeline signal per
    /// device — `support/hcq.py` `HCQCompiled.timeline_signal`.)
    inflight: Mutex<std::collections::VecDeque<Arc<SubmissionFinalizer>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QueueState {
    Constructing,
    Active,
    Destroyed,
    Quarantined,
}

struct StagingBuf {
    _buf: crate::allocator::RawBuffer,
    host: NonNull<u8>,
    gpu: u64,
    size: usize,
}

impl Drop for StagingBuf {
    fn drop(&mut self) {
        self._buf.free_amd_device_in_place();
    }
}

// SAFETY: `host`/`gpu` address a stable GTT mapping owned by `_buf`; all access
// is serialised under `AmdCopyQueue::staging`'s Mutex.
unsafe impl Send for StagingBuf {}

struct QueueInner {
    /// 16 MiB ring buffer; host-visible so we can write packets directly.
    ring_host: NonNull<u8>,
    /// GPU VA of the ring — `AMDKFD_IOC_UPDATE_QUEUE` re-validates the ring
    /// address on every call, so the descriptor-reload remap needs it.
    ring_gpu: u64,
    ring_size: usize,
    /// Per-queue doorbell (`*mut u64` MMIO).
    doorbell: NonNull<u64>,
    /// mmap base of the doorbell page, kept so the queue can `munmap` it on
    /// teardown (each queue maps its own page).
    doorbell_base: NonNull<u8>,
    /// Host pointer to the GART-resident `write_dispatch_id` slot — KFD
    /// reads this in addition to the doorbell. It must be updated before
    /// every doorbell ring. Skipping it makes the GPU's
    /// command processor see the doorbell change but stall on a stale wptr.
    write_ptr_host: NonNull<u64>,
    /// KFD-updated queue read pointer. SDMA publication waits for unread ring
    /// space before writing, matching Tinygrad's `put_value - read_ptr` guard.
    read_ptr_host: NonNull<u64>,
    /// Host base of the GART page (the `AmdQueueT` descriptor). On AQL queues
    /// the scratch fields at fixed offsets are patched here when the
    /// connector's scratch buffer is (re)allocated; see `set_aql_scratch`.
    gart_host: NonNull<u8>,
    /// Host base of the `queue_inactive_signal` amd_signal_t (AQL only). The CP
    /// trap handler writes its exception code into `value` (`+8`) and halts the
    /// queue; reading it tells us WHY a wedge happened (e.g. `0x401`).
    qinactive_host: Option<NonNull<u8>>,
    /// Index of the next packet (in AQL_PACKET_BYTES-sized slots). For SDMA
    /// queues this is the next byte offset; type checks ensure callers don't
    /// confuse them.
    write_idx: u64,
    /// Owned KFD queue id (held for the future destroy ioctl; reading it
    /// inside the queue isn't useful since the ioctl takes it directly).
    #[allow(dead_code)]
    queue_id: u32,
    /// Set by every quarantine transition of the owning queue. `Drop` then
    /// leaks the ring/GART/EOP backing instead of unmapping it: the CP may
    /// still be reading it because the KFD queue was never destroyed. This is
    /// queue-local on purpose — the decision must not depend on the ambient
    /// process-panicking flag or on a device-wide poison latch.
    quarantined: bool,
    /// Owned bookkeeping buffers we need to keep alive. The EOP and ctx-save
    /// buffers stay alive for the lifetime of the queue — KFD reads them
    /// asynchronously as part of the compute dispatch hardware state.
    _ring_buf: crate::allocator::RawBuffer,
    _gart_buf: crate::allocator::RawBuffer,
    _eop_buf: Option<crate::allocator::RawBuffer>,
    _ctx_buf: Option<crate::allocator::RawBuffer>,
    _qinactive_buf: Option<crate::allocator::RawBuffer>,
}

// SAFETY: ring/doorbell access goes through Mutex; underlying buffers are
// allocator-owned and stable.
unsafe impl Send for QueueInner {}
unsafe impl Sync for QueueInner {}

/// Owns an activated queue only while later construction remains fallible.
/// Teardown runs before the backing `QueueInner` is dropped; teardown failure
/// quarantines the inner so `QueueInner::drop` leaks its mappings.
struct ActivatedQueueGuard {
    inner: Option<QueueInner>,
    core: Arc<AmdDeviceCore>,
    state: QueueState,
}

impl ActivatedQueueGuard {
    fn new(inner: QueueInner, core: Arc<AmdDeviceCore>) -> Self {
        Self { inner: Some(inner), core, state: QueueState::Constructing }
    }

    fn into_inner(mut self) -> QueueInner {
        self.state = QueueState::Active;
        self.inner.take().expect("activated queue guard already disarmed")
    }
}

impl Drop for ActivatedQueueGuard {
    fn drop(&mut self) {
        let Some(inner) = self.inner.as_mut() else { return };
        debug_assert_eq!(self.state, QueueState::Constructing);
        // Unwinding only abandons this lane. Tinygrad latches `error_state`
        // per device on a drain timeout or a fault (`hcq.py` `HWQueue`
        // synchronize), never on an abandoned construction, so quarantining the
        // backing is the whole remedy here.
        if std::thread::panicking() {
            self.state = QueueState::Quarantined;
            inner.quarantined = true;
            tracing::warn!("partially constructed queue abandoned during panic unwind; backing quarantined");
            return;
        }
        let (queue_id, doorbell_base) = (inner.queue_id, inner.doorbell_base);
        match self.core.iface().teardown_ring(queue_id, doorbell_base) {
            Ok(_) => self.state = QueueState::Destroyed,
            Err(error) => {
                self.state = QueueState::Quarantined;
                inner.quarantined = true;
                self.core.poison(&error.to_string());
                tracing::warn!(?error, "partial queue teardown failed; backing allocations quarantined");
            }
        }
    }
}

impl Drop for QueueInner {
    /// Free the queue's KFD-allocated VRAM/GTT backings. `RawBuffer` itself
    /// has no `Drop` (the existing `AmdAllocator::_free` consumes RawBuffer
    /// by destructure), so a queue dropped directly — as happens for
    /// pool queues — would otherwise leak ~50 MiB of ring + GART +
    /// EOP + ctx-save every time. We call the in-place free path
    /// (`RawBuffer::free_amd_device_in_place`) for each. `AmdComputeQueue::
    /// Drop` has already invoked `kfd_destroy_queue` AND `PoolQueue::Drop`
    /// has drained the queue, so the GPU is idle on these buffers.
    ///
    /// Skipped once the owning queue quarantined this backing, and during panic
    /// unwind: `PoolQueue::Drop` and `AmdComputeQueue::Drop` both skip their
    /// drain/destroy on panic, so the GPU's CP may still be reading the
    /// ring/GART. Unmapping them here would fault the VM and could crash before
    /// the panic's diagnostics flush. Accept the buffer leak — the OS reclaims
    /// at process exit.
    fn drop(&mut self) {
        if self.quarantined || std::thread::panicking() {
            tracing::warn!(queue_id = self.queue_id, "quarantined AMD queue: leaking ring/GART/EOP backing");
            return;
        }
        self._ring_buf.free_amd_device_in_place();
        self._gart_buf.free_amd_device_in_place();
        if let Some(eop) = self._eop_buf.as_ref() {
            eop.free_amd_device_in_place();
        }
        if let Some(ctx) = self._ctx_buf.as_ref() {
            ctx.free_amd_device_in_place();
        }
        if let Some(qinactive) = self._qinactive_buf.as_ref() {
            qinactive.free_amd_device_in_place();
        }
    }
}

impl Drop for AmdComputeQueue {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

impl Drop for AmdCopyQueue {
    fn drop(&mut self) {
        if self.state != QueueState::Active {
            return;
        }
        // Unwinding abandons this queue's in-flight copies but says nothing
        // about the device, so quarantine without poisoning (tinygrad latches
        // per-device error state on drain timeouts and faults only).
        if std::thread::panicking() {
            self.quarantine("copy queue abandoned during panic unwind");
            return;
        }
        // Both this queue's own staging timeline and every linked plan whose
        // SDMA work it published must retire before the ring is unmapped.
        if let Err(error) = self.drain_inflight().and_then(|()| self.timeline.drain(COPY_TIMEOUT_MS)) {
            self.core.poison(&error.to_string());
            self.quarantine("copy queue drain failed; queue and backing quarantined");
            return;
        }
        let (queue_id, doorbell_base) = {
            let inner = self.inner.lock();
            (inner.queue_id, inner.doorbell_base)
        };
        if let Err(error) = self.core.iface().teardown_ring(queue_id, doorbell_base) {
            self.core.poison(&error.to_string());
            self.quarantine("copy queue teardown failed; backing allocations quarantined");
        } else {
            self.state = QueueState::Destroyed;
        }
    }
}

impl AmdComputeQueue {
    /// Destroy the KFD queue exactly once. Backing remains owned by `inner` and
    /// is released by normal field drop only after this reaches `Destroyed`.
    pub(crate) fn close(&mut self) -> Result<()> {
        if self.state == QueueState::Destroyed {
            return Ok(());
        }
        // A panic unwind abandons this queue, not the device: quarantine the
        // lane and leave the process-global poison latch to real hardware
        // faults and drain timeouts (tinygrad's per-device `error_state`).
        if self.state == QueueState::Quarantined || std::thread::panicking() {
            self.quarantine();
            return Err(self
                .core
                .poison_error()
                .unwrap_or_else(|| Error::Runtime { message: "AMD compute queue is quarantined".into() }));
        }
        debug_assert_eq!(self.state, QueueState::Active);
        let (queue_id, doorbell_base) = {
            let inner = self.inner.get_mut();
            (inner.queue_id, inner.doorbell_base)
        };
        match self.core.iface().teardown_ring(queue_id, doorbell_base) {
            Ok(_) => {
                self.state = QueueState::Destroyed;
                Ok(())
            }
            Err(error) => {
                self.core.poison(&error.to_string());
                self.quarantine();
                tracing::warn!(?error, "compute queue teardown failed; backing allocations quarantined");
                Err(error)
            }
        }
    }

    /// Abandon this queue's hardware state: the KFD queue is never destroyed
    /// and its ring/GART/EOP backing is leaked rather than unmapped under a
    /// possibly live command processor.
    pub(crate) fn quarantine(&mut self) {
        if self.state != QueueState::Destroyed {
            self.state = QueueState::Quarantined;
        }
        self.inner.get_mut().quarantined = true;
    }
}

impl QueueInner {
    /// Append raw PM4 dwords to the ring, wrapping at dword granularity.
    /// `write_idx` is counted in dwords for PM4 queues.
    /// Caller holds the queue lock — this is part of one atomic dispatch.
    ///
    /// Ring overflow is prevented up-stream by `wait_dispatch_headroom` (which
    /// bounds in-flight dispatches via the timeline signal), so a single push
    /// must never exceed the per-dispatch budget.
    fn push_pm4(&mut self, dwords: &[u32]) {
        let ring_dwords = self.ring_size / 4;
        debug_assert!(
            dwords.len() <= MAX_DISPATCH_DWORDS,
            "single dispatch ({} dwords) exceeds MAX_DISPATCH_DWORDS ({MAX_DISPATCH_DWORDS}); \
             raise the bound or lower RING_MAX_INFLIGHT",
            dwords.len(),
        );
        let mut idx = (self.write_idx as usize) % ring_dwords;
        for &dw in dwords {
            // SAFETY: ring_host points to ring_size bytes; idx < ring_dwords.
            unsafe { std::ptr::write_volatile((self.ring_host.as_ptr() as *mut u32).add(idx), dw) };
            idx = (idx + 1) % ring_dwords;
        }
        self.write_idx += dwords.len() as u64;
    }

    fn push_pm4_bytes(&mut self, bytes: &[u8]) {
        let ring_dwords = self.ring_size / 4;
        let mut idx = (self.write_idx as usize) % ring_dwords;
        for word in bytes.as_chunks::<4>().0 {
            let value = u32::from_le_bytes(*word);
            // SAFETY: ring_host spans ring_size bytes; idx < ring_dwords.
            unsafe { std::ptr::write_volatile((self.ring_host.as_ptr() as *mut u32).add(idx), value) };
            idx = (idx + 1) % ring_dwords;
        }
        self.write_idx += (bytes.len() / 4) as u64;
    }

    /// Write one 64-byte AQL packet at the current slot. `write_idx` counts
    /// 64-byte slots for AQL queues.
    fn push_aql(&mut self, bytes: &[u8]) {
        debug_assert_eq!(bytes.len(), AQL_PACKET_BYTES);
        let off = (self.write_idx as usize * AQL_PACKET_BYTES) % self.ring_size;
        // SAFETY: ring_host is mmapped + size-validated; off bounded by ring_size.
        let dst = unsafe { self.ring_host.as_ptr().add(off) };
        let real_header = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        // ROCr `BlitKernel::PopulateQueue` publish protocol: write the packet body
        // with an INVALID-type header FIRST, fence, then store the real header
        // LAST. On a host-visible device/large-BAR ring the packet processor can
        // otherwise latch a valid-typed packet over a torn body (stale grid_size /
        // kernarg_address / completion_signal) and stall the CP with no fault —
        // rptr frozen on the packet, queue_inactive=0 (the exact wedge we see).
        unsafe {
            // dwords 1..16 (body) then dword 0 = INVALID header.
            std::ptr::copy_nonoverlapping(bytes.as_ptr().add(4), dst.add(4), AQL_PACKET_BYTES - 4);
            std::ptr::write_volatile(dst as *mut u32, INVALID_AQL_HEADER);
        }
        // Body must be globally visible before the real header is published.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        // SAFETY: same slot; single 32-bit store flips the type to valid last.
        unsafe { std::ptr::write_volatile(dst as *mut u32, real_header) };
        self.write_idx += 1;
    }

    /// Publish the current `write_idx` to GART + ring the doorbell. AQL uses
    /// the **last completed** slot (`write_idx - 1`); PM4 uses the **next**
    /// dword (`write_idx`).
    fn ring_doorbell(&self, is_pm4: bool) {
        // GART wptr first: without it KFD sees the doorbell
        // change but reads a stale wptr.
        unsafe { std::ptr::write_volatile(self.write_ptr_host.as_ptr(), self.write_idx) };
        // SeqCst (not Release): on x86 a Release fence is a no-op, so it does NOT
        // drain the CPU write-combining buffer or order the wptr/packet stores
        // ahead of the doorbell MMIO write. SeqCst lowers to an `mfence`, which
        // does. Without it the CP can observe the doorbell while the wptr /
        // ring packet / kernarg stores are still buffered (stale read → wedge).
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let doorbell_value = if is_pm4 { self.write_idx } else { self.write_idx - 1 };
        // SAFETY: doorbell is mmapped MMIO; aligned 64-bit store.
        unsafe { std::ptr::write_volatile(self.doorbell.as_ptr(), doorbell_value) };
    }
}

pub(crate) struct LinkedComputePublication<'a> {
    inner: parking_lot::MutexGuard<'a, QueueInner>,
    is_pm4: bool,
}

impl LinkedComputePublication<'_> {
    pub(crate) fn publish(&mut self, replay: &crate::hcq::ReplayCommandBuffer) {
        if self.is_pm4 {
            self.inner.push_pm4_bytes(replay.bytes());
        } else {
            for packet in replay.bytes().as_chunks::<AQL_PACKET_BYTES>().0 {
                self.inner.push_aql(packet);
            }
        }
        self.inner.ring_doorbell(self.is_pm4);
    }
}

pub(crate) struct LinkedCopyPublication<'a> {
    inner: parking_lot::MutexGuard<'a, QueueInner>,
}

/// Exclusive ring writer that restores the producer index when a publication is
/// abandoned — ordinary failure or panic unwind — before its doorbell rings.
///
/// Owning the `&mut QueueInner` is the point: nothing reaches the ring except
/// through this guard, so "snapshot the index before the first push" is enforced
/// by borrowck instead of call-site discipline, and rollback always lands on the
/// pre-submission index rather than on whatever the index happened to be when a
/// timeline reservation was constructed.
struct RingRollback<'a> {
    inner: &'a mut QueueInner,
    saved: u64,
    committed: bool,
}

impl<'a> RingRollback<'a> {
    fn new(inner: &'a mut QueueInner) -> Self {
        let saved = inner.write_idx;
        Self { inner, saved, committed: false }
    }

    /// Keep everything written so far: the doorbell rang and the packets now
    /// belong to the GPU.
    fn commit(mut self) {
        self.committed = true;
    }
}

impl std::ops::Deref for RingRollback<'_> {
    type Target = QueueInner;

    fn deref(&self) -> &QueueInner {
        self.inner
    }
}

impl std::ops::DerefMut for RingRollback<'_> {
    fn deref_mut(&mut self) -> &mut QueueInner {
        self.inner
    }
}

impl Drop for RingRollback<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.inner.write_idx = self.saved;
        }
    }
}

/// Rolls an unpublished queue-timeline reservation back on ordinary failure or
/// panic. Once a doorbell has been rung, abandonment poisons the device instead.
/// The ring producer index is owned by the paired [`RingRollback`].
struct TimelineReservation<'a> {
    pool: &'a PoolQueue,
    core: &'a AmdDeviceCore,
    value: u64,
    published: bool,
    committed: bool,
}

impl<'a> TimelineReservation<'a> {
    fn new(pool: &'a PoolQueue, core: &'a AmdDeviceCore, value: u64) -> Self {
        Self { pool, core, value, published: false, committed: false }
    }

    fn mark_published(&mut self) {
        self.published = true;
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for TimelineReservation<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if self.published {
            self.core.poison("AMD queue publication failed after ringing its doorbell");
        } else if !self.pool.rollback_pm4(self.value) {
            self.core.poison("AMD timeline reservation rollback lost publication authority");
        }
    }
}

struct CopyTimelineReservation<'a> {
    timeline: &'a Timeline,
    core: &'a AmdDeviceCore,
    value: u64,
    published: bool,
    committed: bool,
}

impl<'a> CopyTimelineReservation<'a> {
    fn new(timeline: &'a Timeline, core: &'a AmdDeviceCore, value: u64) -> Self {
        Self { timeline, core, value, published: false, committed: false }
    }

    fn mark_published(&mut self) {
        self.published = true;
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for CopyTimelineReservation<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if self.published {
            self.core.poison("AMD copy publication failed after ringing its doorbell");
        } else if !self.timeline.rollback(self.value) {
            self.core.poison("AMD copy timeline reservation rollback lost publication authority");
        }
    }
}

impl LinkedCopyPublication<'_> {
    pub(crate) fn publish(&mut self, replay: &crate::hcq::ReplayCommandBuffer) {
        push_sdma_bytes(&mut self.inner, replay.bytes());
        unsafe { std::ptr::write_volatile(self.inner.write_ptr_host.as_ptr(), self.inner.write_idx) };
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        unsafe { std::ptr::write_volatile(self.inner.doorbell.as_ptr(), self.inner.write_idx) };
    }
}

impl AmdComputeQueue {
    /// Predict whether `create` would build a PM4 queue for this device,
    /// WITHOUT allocating anything. Single-XCC GPUs (the gfx11/12 default)
    /// use the PM4 path (`KFD_IOC_QUEUE_TYPE_COMPUTE`), submitting raw PM4
    /// dwords directly into the ring; multi-XCC CDNA uses AQL, where each
    /// dispatch is a 64-byte AQL packet and PM4 helpers are wrapped via the
    /// vendor IB packet. `SVOD_AMD_AQL` set to anything but `"0"` forces AQL.
    /// Used by `AmdGraph::capture` to skip the (multi-MiB) per-graph queue
    /// build on AQL hardware where the graph path is unsupported anyway.
    pub fn will_use_pm4(core: &AmdDeviceCore) -> bool {
        let force_aql = std::env::var("SVOD_AMD_AQL").ok().map(|s| s != "0").unwrap_or(false);
        !force_aql && core.node.num_xcc.max(1) == 1
    }

    pub fn create(allocator: &AmdAllocator) -> Result<Box<Self>> {
        let core = allocator.dev.core();
        // `SVOD_AMD_AQL=1` forces AQL even on single-XCC, useful for
        // bisecting PM4 vs AQL bring-up issues.
        let is_pm4 = Self::will_use_pm4(core);
        let queue_type = if is_pm4 { kfd::KFD_IOC_QUEUE_TYPE_COMPUTE } else { kfd::KFD_IOC_QUEUE_TYPE_COMPUTE_AQL };
        let inner = create_queue(allocator, queue_type, COMPUTE_RING_BYTES, !is_pm4, /*needs_cwsr=*/ true)?;
        debug!(gpu_id = core.node.gpu_id, num_xcc = core.node.num_xcc, is_pm4 = is_pm4, "AmdComputeQueue created");
        Ok(Box::new(Self { inner: Mutex::new(inner), state: QueueState::Active, core: Arc::clone(core), is_pm4 }))
    }

    /// `true` when this queue submits raw PM4 dwords (single-XCC); `false`
    /// for the AQL path. Read by callers in `program.rs` to pick the right
    /// dispatch builder.
    pub fn is_pm4(&self) -> bool {
        self.is_pm4
    }

    /// Block until at most `RING_MAX_INFLIGHT` dispatches are un-retired, so a
    /// host running `wait=false` faster than the GPU can't lap the ring and
    /// overwrite unconsumed packets. Bounds the combined ring footprint to
    /// `RING_MAX_INFLIGHT * MAX_DISPATCH_DWORDS` (half the ring).
    ///
    /// Gates on the queue's PM4 counter SIGNAL — the proven completion
    /// primitive `drain_all` already uses — not the PM4 read pointer (whose
    /// COMPUTE-queue semantics are unreliable, which would deadlock a spin).
    /// The dispatches we wait on were submitted (doorbell rung) in prior calls,
    /// so the GPU will signal them; the wait always makes progress.
    fn wait_dispatch_headroom(&self, pool: &PoolQueue) -> Result<()> {
        let last_reserved = pool.pm4_value().saturating_sub(1);
        if last_reserved > RING_MAX_INFLIGHT {
            let target = last_reserved - RING_MAX_INFLIGHT;
            pool.pm4_signal().wait_signal_value(target, 30_000).inspect_err(|e| self.core.poison(&e.to_string()))?;
        }
        Ok(())
    }

    /// Lower and publish one per-call neutral compute submission. PM4 wraps the
    /// common intent in its monotonic wait/store timeline. AQL carries the same
    /// wait/store timeline in barriered vendor IB packets around a native kernel
    /// packet whose completion field remains unset.
    ///
    /// The caller's exclusive lane lease keeps PM4 counter reservation and ring
    /// publication ordered. The backend-local `inner` guard is uncontended.
    ///
    /// Sequence (timestamps are present only when the submission requests them):
    /// `wait(counter, prev) → memory_barrier → [timestamp] → exec → [timestamp]
    /// → signal(counter, next)`.
    /// Returns the counter value this dispatch signals.
    pub(crate) fn submit_hcq_dispatch(
        &self,
        pool: &PoolQueue,
        submission: &crate::hcq::Submission,
        pmc_start: &[u32],
        pmc_read: &[u32],
    ) -> Result<HcqDispatchResult> {
        debug_assert!(
            Arc::ptr_eq(&self.core, pool.core()),
            "submit_hcq_dispatch: pool core ≠ queue core (queue gpu_id={}, pool gpu_id={}); \
             cross-device dispatch silently corrupts scratch/counter VAs",
            self.core.node.gpu_id,
            pool.core().node.gpu_id,
        );
        if let Some(err) = self.core.poison_error() {
            return Err(err);
        }
        if !self.is_pm4 {
            if !pmc_start.is_empty() || !pmc_read.is_empty() {
                return Err(Error::Runtime { message: "AMD AQL PMC register profiling is unsupported".into() });
            }
            pool.ensure_pm4_headroom()?;
            self.wait_dispatch_headroom(pool)?;
            let counter_addr = pool.pm4_signal().value_addr();
            let prev = pool.pm4_value().saturating_sub(1);
            let next = pool.pm4_value();
            let timestamps = if submission.profile_requested() {
                let signal = pool.acquire_timestamp_signal()?;
                signal.reset(0);
                Some(signal)
            } else {
                None
            };

            let mut finalized = finalize_hcq_aql_timeline_submission(
                submission,
                counter_addr,
                prev,
                next,
                timestamps.as_ref().map(|signal| (signal.start_ts_addr(), signal.end_ts_addr())),
            )?;
            push_queue_event_mailbox(&mut finalized, &self.core);

            let program = lower_hcq_aql_submission_program(
                &finalized,
                Pm4LoweringState {
                    scratch_address: pool.scratch_gpu_va(),
                    tmpring_size: pool.tmpring_size(),
                    target_major: self.core.arch.gfx_major(),
                    completion_xcc_mask: (self.core.node.num_xcc > 1).then_some(1),
                    queue_event_mailbox: self.core.queue_event_mailbox().map(|mailbox| mailbox.address),
                },
                crate::hcq::PatchSource::LinkAddress(0),
            )?;
            let control_offset = pool.arena().bump(program.control.bytes.len(), 16)?;
            let control_gpu = pool.arena().gpu_at(control_offset);
            // SAFETY: the arena reservation covers the full resident PM4 control stream.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    program.control.bytes.as_ptr(),
                    pool.arena().host_at(control_offset),
                    program.control.bytes.len(),
                );
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            let linked = pool.link(&program.aql, &crate::hcq::LinkPatchValues(vec![control_gpu]))?;
            self.publish_linked_aql_timeline(pool, &linked.replay_buffer(), next)?;
            let finalizer = SubmissionFinalizer::timeline(Arc::clone(pool.pm4_signal()), next, timestamps.clone());
            return Ok(HcqDispatchResult { timestamps, finalizer });
        }

        // The caller owns the lane across kernarg bump, write, and publication,
        // so the PM4 counter and ring remain in the same order.
        // Keep the PM4 counter < 2^32 (drain+reset at the watermark) before
        // reserving this dispatch's value.
        pool.ensure_pm4_headroom()?;
        // Ring back-pressure: block if too many dispatches are in flight, so an
        // async (`wait=false`) burst can't lap the ring. Outside the inner lock.
        self.wait_dispatch_headroom(pool)?;
        let counter_addr = pool.pm4_signal().value_addr();
        let scratch_addr = pool.scratch_gpu_va();
        let tmpring_size = pool.tmpring_size();
        let target_major = self.core.arch.gfx_major();
        let mut g = self.inner.lock();
        let prev = pool.pm4_value().saturating_sub(1);
        let next = pool.pm4_value();

        let timestamps = if submission.profile_requested() {
            let signal = pool.acquire_timestamp_signal()?;
            signal.reset(0);
            Some(signal)
        } else {
            None
        };
        let ts_addrs = timestamps.as_ref().map(|signal| (signal.start_ts_addr(), signal.end_ts_addr()));

        let mut finalized = crate::hcq::Submission::new(submission.queue);
        finalized.push(crate::hcq::Command::Wait { signal_address: counter_addr, value: prev });
        for command in &submission.commands {
            if matches!(command, crate::hcq::Command::Compute(_))
                && let Some((start_addr, _)) = ts_addrs
            {
                finalized.push(crate::hcq::Command::Timestamp { dst: start_addr });
            }
            finalized.push(command.clone());
            if matches!(command, crate::hcq::Command::Compute(_))
                && let Some((_, end_addr)) = ts_addrs
            {
                finalized.push(crate::hcq::Command::Timestamp { dst: end_addr });
            }
        }
        finalized.push(crate::hcq::Command::Store {
            dst: counter_addr,
            value: next,
            scope: crate::hcq::StoreScope::System,
        });
        push_queue_event_mailbox(&mut finalized, &self.core);

        let state = Pm4LoweringState {
            scratch_address: scratch_addr,
            tmpring_size,
            target_major,
            completion_xcc_mask: None,
            queue_event_mailbox: self.core.queue_event_mailbox().map(|mailbox| mailbox.address),
        };
        let q = if pmc_start.is_empty() && pmc_read.is_empty() {
            lower_hcq_pm4(&finalized, state)?
        } else {
            // PMC register programs are backend diagnostics rather than neutral
            // HCQ commands. Keep their established bracketing while all semantic
            // queue commands use the common lowerer.
            let compute = finalized
                .commands
                .iter()
                .position(|command| matches!(command, crate::hcq::Command::Compute(_)))
                .ok_or_else(|| Error::Runtime { message: "AMD PM4 HCQ submission has no compute command".into() })?;
            let mut prefix = crate::hcq::Submission::new(finalized.queue);
            prefix.commands.extend_from_slice(&finalized.commands[..compute]);
            let mut middle = crate::hcq::Submission::new(finalized.queue);
            middle.commands.extend_from_slice(&finalized.commands[compute..finalized.commands.len() - 1]);
            let mut suffix = crate::hcq::Submission::new(finalized.queue);
            suffix.commands.push(finalized.commands.last().unwrap().clone());
            let mut q = lower_hcq_pm4(&prefix, state)?;
            q.extend_from_slice(pmc_start);
            q.extend(lower_hcq_pm4(&middle, state)?);
            q.extend_from_slice(pmc_read);
            q.extend(lower_hcq_pm4(&suffix, state)?);
            q
        };

        validate_pm4_dword_count(q.len())?;
        wait_pm4_headroom(&g, q.len()).inspect_err(|error| self.core.poison(&error.to_string()))?;
        let mut ring = RingRollback::new(&mut g);
        let reserved = pool.next_pm4();
        debug_assert_eq!(reserved, next, "queue lease must serialize timeline reservation");
        let mut reservation = TimelineReservation::new(pool, &self.core, reserved);
        self.core.publication_checkpoint(crate::amd::iface::PublicationStage::AfterReservation)?;
        ring.push_pm4(&q);
        self.core.publication_checkpoint(crate::amd::iface::PublicationStage::BeforeDoorbell)?;
        ring.ring_doorbell(/*is_pm4=*/ true);
        reservation.mark_published();
        ring.commit();
        self.core.publication_checkpoint(crate::amd::iface::PublicationStage::AfterDoorbell)?;
        reservation.commit();
        let finalizer = SubmissionFinalizer::timeline(Arc::clone(pool.pm4_signal()), next, timestamps.clone());
        Ok(HcqDispatchResult { finalizer, timestamps })
    }

    /// Push a pre-built PM4 dword stream into the ring with ONE doorbell — the
    /// primitive behind `AmdHwQueue::submit` (the graph's atomic submit).
    /// Blits `cmds` into the ring, advances the write index, rings the doorbell.
    ///
    /// `dwords` is normally the 4-dword `PACKET3_INDIRECT_BUFFER` reference to
    /// the graph's bound `hw_page`; the CP then runs the whole captured chain
    /// inline.
    ///
    /// With per-connector queues, each graph owns its connector and submits
    /// through ITS OWN ring. The graph's own `comp_queue` `Mutex<AmdHwQueue>`
    /// serialises capture vs replay within one graph; this primitive only
    /// takes the queue's inner lock.
    pub fn submit_dwords(&self, dwords: &[u32]) -> Result<()> {
        debug_assert!(self.is_pm4, "submit_dwords on AQL queue");
        if let Some(err) = self.core.poison_error() {
            return Err(err);
        }
        validate_pm4_dword_count(dwords.len())?;
        // The lane lease gives one writer a contiguous run + doorbell. No
        // `Release` fence here —
        // `ring_doorbell` already issues its own publication barrier.
        let mut g = self.inner.lock();
        wait_pm4_headroom(&g, dwords.len()).inspect_err(|error| self.core.poison(&error.to_string()))?;
        g.push_pm4(dwords);
        g.ring_doorbell(/*is_pm4=*/ true);
        Ok(())
    }

    /// Patch and publish a linked PM4 graph through the same counter discipline
    /// as ordinary PM4 dispatch. The native command stream was lowered once at
    /// capture; replay only updates invocation and queue-owned fields.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn replay_linked_pm4(
        &self,
        pool: &PoolQueue,
        linked: &crate::hcq::LinkedCommandBuffer,
        replay: &mut crate::hcq::ReplayCommandBuffer,
        resident_host: *mut u8,
        resident_gpu: u64,
        runtime: &crate::hcq::RuntimePatchValues,
        system: &mut crate::hcq::SystemPatchValues,
    ) -> Result<Arc<SubmissionFinalizer>> {
        debug_assert!(self.is_pm4, "replay_linked_pm4 on AQL queue");
        pool.ensure_pm4_headroom()?;
        self.wait_dispatch_headroom(pool)?;
        let counter_addr = pool.pm4_signal().value_addr();
        let prev = pool.pm4_value().saturating_sub(1);
        let next = pool.pm4_value();
        system.0.insert(crate::hcq::SystemField::TimelineSignal(0), counter_addr);
        system.0.insert(crate::hcq::SystemField::TimelineValue(0), prev);
        system.0.insert(crate::hcq::SystemField::TimelineValue(1), next);
        system.0.insert(crate::hcq::SystemField::ScratchAddress, pool.scratch_gpu_va());
        system.0.insert(crate::hcq::SystemField::ScratchTmpring, pool.tmpring_size() as u64);
        linked.patch(replay, runtime, system)?;
        let indirect = build_pm4_indirect_buffer(resident_gpu, replay.bytes().len() / 4)?;
        // SAFETY: graph capture owns a host-visible resident allocation at least
        // as large as this replay and serializes mutation through its state lock.
        unsafe { std::ptr::copy_nonoverlapping(replay.bytes().as_ptr(), resident_host, replay.bytes().len()) };
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let mut g = self.inner.lock();
        wait_pm4_headroom(&g, indirect.len()).inspect_err(|error| self.core.poison(&error.to_string()))?;
        let mut ring = RingRollback::new(&mut g);
        let reserved = pool.next_pm4();
        debug_assert_eq!(reserved, next, "queue lease must serialize timeline reservation");
        let mut reservation = TimelineReservation::new(pool, &self.core, reserved);
        self.core.publication_checkpoint(crate::amd::iface::PublicationStage::AfterReservation)?;
        ring.push_pm4(&indirect);
        self.core.publication_checkpoint(crate::amd::iface::PublicationStage::BeforeDoorbell)?;
        ring.ring_doorbell(/*is_pm4=*/ true);
        reservation.mark_published();
        ring.commit();
        self.core.publication_checkpoint(crate::amd::iface::PublicationStage::AfterDoorbell)?;
        reservation.commit();
        Ok(SubmissionFinalizer::timeline(Arc::clone(pool.pm4_signal()), next, None))
    }

    /// Patch and publish a linked AQL stream plus its resident PM4 control IBs
    /// through the queue-owned timeline. Kernel packets carry no completion;
    /// the trailing control IB publishes `next` after all prior AQL packets.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn replay_linked_aql_timeline(
        &self,
        pool: &PoolQueue,
        linked: &crate::hcq::LinkedCommandBuffer,
        replay: &mut crate::hcq::ReplayCommandBuffer,
        control_linked: &crate::hcq::LinkedCommandBuffer,
        control_replay: &mut crate::hcq::ReplayCommandBuffer,
        control_host: *mut u8,
        runtime: &crate::hcq::RuntimePatchValues,
        system: &mut crate::hcq::SystemPatchValues,
    ) -> Result<Arc<SubmissionFinalizer>> {
        debug_assert!(!self.is_pm4, "replay_linked_aql_timeline on PM4 queue");
        pool.ensure_pm4_headroom()?;
        self.wait_dispatch_headroom(pool)?;
        let counter_addr = pool.pm4_signal().value_addr();
        let prev = pool.pm4_value().saturating_sub(1);
        let next = pool.pm4_value();
        system.0.insert(crate::hcq::SystemField::TimelineSignal(0), counter_addr);
        system.0.insert(crate::hcq::SystemField::TimelineValue(0), prev);
        system.0.insert(crate::hcq::SystemField::TimelineValue(1), next);
        control_linked.patch(control_replay, runtime, system)?;
        // SAFETY: graph capture owns a host-visible allocation at least as large
        // as the immutable control replay and serializes replay through its mutex.
        unsafe {
            std::ptr::copy_nonoverlapping(control_replay.bytes().as_ptr(), control_host, control_replay.bytes().len());
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        linked.patch(replay, runtime, system)?;
        validate_aql_packet_count(replay.bytes().len() / AQL_PACKET_BYTES)?;
        self.publish_linked_aql_timeline(pool, replay, next)?;
        Ok(SubmissionFinalizer::timeline(Arc::clone(pool.pm4_signal()), next, None))
    }

    /// Lower and publish one complete HCQ command buffer with one doorbell.
    pub fn submit_hcq_pm4(&self, submission: &crate::hcq::Submission, state: Pm4LoweringState) -> Result<()> {
        debug_assert!(self.is_pm4, "submit_hcq_pm4 on AQL queue");
        let dwords = lower_hcq_pm4(submission, state)?;
        self.submit_dwords(&dwords)
    }

    /// Re-blit a captured sequence of 64-byte AQL packets into the ring with ONE
    /// doorbell — the AQL (multi-XCC) analogue of [`submit_dwords`], used by the
    /// graph replay. Each packet is either a vendor IB (pointing at a captured
    /// PM4 run) or a native kernel-dispatch packet; the AQL packet processor runs
    /// them in order. Back-pressure is the graph's per-replay timeline wait (one
    /// replay in flight), so no `wait_dispatch_headroom` is needed here.
    pub fn submit_aql(&self, packets: &[[u32; 16]]) -> Result<()> {
        debug_assert!(!self.is_pm4, "submit_aql on PM4 queue");
        if let Some(err) = self.core.poison_error() {
            return Err(err);
        }
        validate_aql_packet_count(packets.len())?;
        // The lane lease gives one writer a contiguous packet run + doorbell.
        let mut g = self.inner.lock();
        wait_aql_headroom(&g, packets.len()).inspect_err(|error| self.core.poison(&error.to_string()))?;
        for p in packets {
            g.push_aql(dwords_as_bytes(p));
        }
        g.ring_doorbell(/*is_pm4=*/ false);
        Ok(())
    }

    pub fn submit_linked_aql(&self, replay: &crate::hcq::ReplayCommandBuffer) -> Result<()> {
        if !replay.bytes().len().is_multiple_of(AQL_PACKET_BYTES) {
            return Err(Error::Runtime { message: "linked AQL stream is not packet aligned".into() });
        }
        let packets = replay
            .bytes()
            .as_chunks::<AQL_PACKET_BYTES>()
            .0
            .iter()
            .map(|bytes| {
                let mut packet = [0u32; 16];
                for (word, bytes) in packet.iter_mut().zip(bytes.as_chunks::<4>().0) {
                    *word = u32::from_le_bytes(*bytes);
                }
                packet
            })
            .collect::<Vec<_>>();
        self.submit_aql(&packets)
    }

    fn publish_linked_aql_timeline(
        &self,
        pool: &PoolQueue,
        replay: &crate::hcq::ReplayCommandBuffer,
        next: u64,
    ) -> Result<()> {
        if !replay.bytes().len().is_multiple_of(AQL_PACKET_BYTES) {
            return Err(Error::Runtime { message: "linked AQL stream is not packet aligned".into() });
        }
        let packets = replay.bytes().len() / AQL_PACKET_BYTES;
        validate_aql_packet_count(packets)?;
        let mut g = self.inner.lock();
        wait_aql_headroom(&g, packets).inspect_err(|error| self.core.poison(&error.to_string()))?;
        let mut ring = RingRollback::new(&mut g);
        let reserved = pool.next_pm4();
        debug_assert_eq!(reserved, next, "queue lease must serialize timeline reservation");
        let mut reservation = TimelineReservation::new(pool, &self.core, reserved);
        self.core.publication_checkpoint(crate::amd::iface::PublicationStage::AfterReservation)?;
        for packet in replay.bytes().as_chunks::<AQL_PACKET_BYTES>().0 {
            ring.push_aql(packet);
        }
        self.core.publication_checkpoint(crate::amd::iface::PublicationStage::BeforeDoorbell)?;
        ring.ring_doorbell(/*is_pm4=*/ false);
        reservation.mark_published();
        ring.commit();
        self.core.publication_checkpoint(crate::amd::iface::PublicationStage::AfterDoorbell)?;
        reservation.commit();
        Ok(())
    }

    /// Wait for linked-publication headroom WITHOUT pinning the queue lock
    /// across the poll. A linked plan spanning both engines waits each ring
    /// here first and only then takes the two guards back-to-back; holding the
    /// compute guard while polling the shared SDMA ring stalled every host
    /// staging copy for up to the full timeout. Tinygrad completes the compute
    /// `_submit` before the copy one and never nests engine locks.
    pub(crate) fn wait_publication_headroom(&self, byte_lengths: &[usize]) -> Result<()> {
        if let Some(error) = self.core.poison_error() {
            return Err(error);
        }
        // `ring_size` and `is_pm4` are fixed at queue creation.
        let units = validate_linked_compute_lengths(self.is_pm4, self.inner.lock().ring_size, byte_lengths)?;
        let what = if self.is_pm4 { "PM4 linked publication headroom" } else { "AQL linked publication headroom" };
        spin_until_headroom(what, HEADROOM_TIMEOUT_MS, || {
            let inner = self.inner.lock();
            if self.is_pm4 { pm4_shortfall(&inner, units) } else { aql_shortfall(&inner, units) }
        })
        .inspect_err(|error| self.core.poison(&error.to_string()))
    }

    /// Take the publication guard, re-verifying headroom. Cheap after
    /// [`wait_publication_headroom`](Self::wait_publication_headroom); it keeps
    /// its own bounded wait so a ring consumed between the two calls still
    /// blocks rather than lapping.
    pub(crate) fn prepare_linked_publication(&self, byte_lengths: &[usize]) -> Result<LinkedComputePublication<'_>> {
        if let Some(error) = self.core.poison_error() {
            return Err(error);
        }
        let inner = self.inner.lock();
        let units = validate_linked_compute_lengths(self.is_pm4, inner.ring_size, byte_lengths)?;
        if self.is_pm4 {
            wait_pm4_headroom(&inner, units).inspect_err(|error| self.core.poison(&error.to_string()))?;
        } else {
            wait_aql_headroom(&inner, units).inspect_err(|error| self.core.poison(&error.to_string()))?;
        }
        Ok(LinkedComputePublication { inner, is_pm4: self.is_pm4 })
    }

    /// Patch the AQL `amd_queue_t` scratch descriptor in the GART page. The AQL
    /// packet processor reads private-segment (scratch) config from here, so it
    /// must be refreshed whenever the connector's scratch buffer is allocated or
    /// grown. No-op on PM4 queues, where scratch goes through registers per
    /// dispatch. The caller holds the queue idle (the connector drains its
    /// timeline before a scratch realloc), so no in-flight dispatch can observe
    /// a half-written descriptor.
    /// Publish an AQL scratch descriptor so the CP firmware actually consumes
    /// it. ROCr documents that CP FW caches its copy of `amd_queue_t` at
    /// queue-connect and re-reads it only on a queue re-map
    /// (`AqlQueue::Suspend/Resume`, `amd_aql_queue.cpp:795-800`), so an
    /// in-place GART write alone is not guaranteed visible. Sequence mirrors
    /// ROCr: unmap (`queue_percentage = 0`) → write the descriptor → remap
    /// (`= 100`, FW re-reads). No-op on PM4 queues, whose scratch rides in
    /// per-dispatch `SET_SH_REG` packets.
    ///
    /// The caller holds the exclusive lane lease with the queue drained. A
    /// failed remap leaves the queue unmapped — unusable — so it poisons the
    /// device rather than letting later timeline waits burn their deadline.
    pub(crate) fn publish_aql_descriptor(&self, desc: &crate::amd::device::AqlScratchDesc) -> Result<()> {
        if self.is_pm4 {
            return Ok(());
        }
        let (queue_id, ring_gpu, ring_size) = {
            let g = self.inner.lock();
            (g.queue_id, g.ring_gpu, g.ring_size as u32)
        };
        self.core.iface().update_queue_percentage(queue_id, ring_gpu, ring_size, 0)?;
        self.set_aql_scratch(desc);
        self.core
            .iface()
            .update_queue_percentage(queue_id, ring_gpu, ring_size, kfd::KFD_MAX_QUEUE_PERCENTAGE)
            .inspect_err(|e| self.core.poison(&format!("AQL descriptor remap failed; queue left unmapped: {e}")))
    }

    pub(crate) fn set_aql_scratch(&self, desc: &crate::amd::device::AqlScratchDesc) {
        if self.is_pm4 {
            return;
        }
        use crate::amd::sys::hsa;
        // Called during construction or under the lane's exclusive lease.
        let base = self.inner.lock().gart_host.as_ptr();
        // SAFETY: `base` is the GART page we mmapped; every offset lands inside
        // the 256-byte AmdQueueT descriptor that occupies the page.
        unsafe {
            std::ptr::write_volatile(base.add(hsa::OFFSET_COMPUTE_TMPRING_SIZE) as *mut u32, desc.tmpring_size);
            let rd = base.add(hsa::OFFSET_SCRATCH_RESOURCE_DESCRIPTOR) as *mut u32;
            for (i, w) in desc.resource_descriptor.iter().enumerate() {
                std::ptr::write_volatile(rd.add(i), *w);
            }
            std::ptr::write_volatile(
                base.add(hsa::OFFSET_SCRATCH_BACKING_MEMORY_LOCATION) as *mut u64,
                desc.backing_va,
            );
            std::ptr::write_volatile(
                base.add(hsa::OFFSET_SCRATCH_WAVE64_LANE_BYTE_SIZE) as *mut u32,
                desc.wave64_lane_byte_size,
            );
            // Force the host writes through to the GART page before returning:
            // the page is mapped write-combining, so a CPU `Release` fence alone
            // does not guarantee the command processor sees the new descriptor on
            // the next dispatch. A read-back drains the WC buffers (and serialises
            // with the fence below). Without this, a mid-stream scratch grow
            // intermittently leaves the CP reading the stale descriptor — which,
            // once the old backing is freed, points at unmapped VRAM and wedges
            // the queue with no fault event.
            std::ptr::read_volatile(base.add(hsa::OFFSET_COMPUTE_TMPRING_SIZE) as *const u32);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    }

    /// Read the AQL scratch descriptor back out of the GART page — the bytes the
    /// firmware actually sees. Test-only: validates that `set_aql_scratch` wrote
    /// the right values at the right offsets on real hardware.
    #[cfg(test)]
    pub(crate) fn read_aql_scratch(&self) -> crate::amd::device::AqlScratchDesc {
        use crate::amd::sys::hsa;
        let base = self.inner.lock().gart_host.as_ptr();
        unsafe {
            let rd = base.add(hsa::OFFSET_SCRATCH_RESOURCE_DESCRIPTOR) as *const u32;
            crate::amd::device::AqlScratchDesc {
                tmpring_size: std::ptr::read_volatile(base.add(hsa::OFFSET_COMPUTE_TMPRING_SIZE) as *const u32),
                resource_descriptor: [
                    std::ptr::read_volatile(rd),
                    std::ptr::read_volatile(rd.add(1)),
                    std::ptr::read_volatile(rd.add(2)),
                    std::ptr::read_volatile(rd.add(3)),
                ],
                backing_va: std::ptr::read_volatile(base.add(hsa::OFFSET_SCRATCH_BACKING_MEMORY_LOCATION) as *const u64),
                wave64_lane_byte_size: std::ptr::read_volatile(
                    base.add(hsa::OFFSET_SCRATCH_WAVE64_LANE_BYTE_SIZE) as *const u32
                ),
            }
        }
    }

    /// The CP exception code the trap handler wrote into `queue_inactive_signal`
    /// when it halted the queue (e.g. `0x401` insufficient-scratch), or `None`
    /// if no exception is recorded (value `0`) / on PM4 queues. Used to turn a
    /// blind dispatch timeout into a diagnosable error.
    /// Ring producer index. Test-only: proves an abandoned publication left no
    /// packets behind.
    #[cfg(test)]
    pub(crate) fn ring_write_idx(&self) -> u64 {
        self.inner.lock().write_idx
    }

    pub(crate) fn inactive_exception(&self) -> Option<i64> {
        let h = self.inner.lock().qinactive_host?;
        // SAFETY: host-visible amd_signal_t; `value` is the i64 at +8.
        let code = unsafe { std::ptr::read_volatile(h.as_ptr().add(8) as *const i64) };
        (code != 0).then_some(code)
    }
}

pub(crate) fn validate_aql_packet_count(packets: usize) -> Result<()> {
    let slots = COMPUTE_RING_BYTES / AQL_PACKET_BYTES;
    if packets == 0 || packets >= slots {
        return Err(Error::CommandStreamTooLarge { kind: "AQL ring submission", actual: packets, limit: slots - 1 });
    }
    Ok(())
}

pub(crate) fn validate_linked_compute_lengths(is_pm4: bool, ring_size: usize, byte_lengths: &[usize]) -> Result<usize> {
    let unit = if is_pm4 { 4 } else { AQL_PACKET_BYTES };
    let kind = if is_pm4 { "PM4 linked transaction" } else { "AQL linked transaction" };
    let capacity = ring_size / unit;
    let units = byte_lengths.iter().try_fold(0usize, |total, &bytes| {
        if bytes == 0 || bytes % unit != 0 {
            return Err(Error::Runtime { message: format!("{kind} stream is not {unit}-byte aligned") });
        }
        total.checked_add(bytes / unit).ok_or(Error::CommandStreamTooLarge {
            kind,
            actual: usize::MAX,
            limit: capacity - 1,
        })
    })?;
    if units == 0 || units >= capacity {
        return Err(Error::CommandStreamTooLarge { kind, actual: units, limit: capacity - 1 });
    }
    Ok(units)
}

pub(crate) fn linked_sdma_published_bytes(write_idx: u64, ring_size: usize, byte_lengths: &[usize]) -> Result<u64> {
    let mut next = write_idx;
    for &bytes in byte_lengths {
        if bytes == 0 || bytes % 4 != 0 || bytes >= ring_size {
            return Err(Error::CommandStreamTooLarge {
                kind: "SDMA linked transaction",
                actual: bytes,
                limit: ring_size - 4,
            });
        }
        let pos = (next as usize) % ring_size;
        if pos + bytes > ring_size {
            next += (ring_size - pos) as u64;
        }
        next = next.checked_add(bytes as u64).ok_or(Error::CommandStreamTooLarge {
            kind: "SDMA linked transaction",
            actual: usize::MAX,
            limit: ring_size - 4,
        })?;
    }
    let published = next.saturating_sub(write_idx);
    if published >= ring_size as u64 {
        return Err(Error::CommandStreamTooLarge {
            kind: "SDMA linked transaction",
            actual: published.min(usize::MAX as u64) as usize,
            limit: ring_size - 4,
        });
    }
    Ok(published)
}

pub(crate) fn absolute_pm4_read_idx(write_idx: u64, reported_read: u64, capacity: usize) -> u64 {
    let capacity = capacity as u64;
    let mut read = write_idx / capacity * capacity + reported_read % capacity;
    if read > write_idx {
        read = read.saturating_sub(capacity);
    }
    read
}

/// Bound on any single ring-headroom wait.
const HEADROOM_TIMEOUT_MS: u64 = 30_000;

/// Spin (then yield) until `probe` reports headroom. `probe` returns `None`
/// once the ring has room, or the `(target, observed read index)` pair to
/// report if the deadline expires. Taking a closure lets a caller that must not
/// pin a ring re-take the queue lock on every attempt.
fn spin_until_headroom(
    what: &'static str,
    timeout_ms: u64,
    mut probe: impl FnMut() -> Option<(u64, u64)>,
) -> Result<()> {
    let start = std::time::Instant::now();
    loop {
        let Some((target, current)) = probe() else { return Ok(()) };
        if start.elapsed().as_millis() as u64 >= timeout_ms {
            return Err(Error::TimelineTimeout { what, target, current, waited_ms: timeout_ms });
        }
        std::hint::spin_loop();
        if start.elapsed().as_micros() >= 100 {
            std::thread::yield_now();
        }
    }
}

/// `None` when `dwords` fit, else the producer index that must be retired to
/// plus the consumer index observed now.
fn pm4_shortfall(g: &QueueInner, dwords: usize) -> Option<(u64, u64)> {
    let capacity = g.ring_size / 4;
    // PM4 reports only the queue-relative RPTR bits. Reconstruct its epoch from
    // the monotonic producer index, as KFD does when restoring a HQD.
    let reported_read = unsafe { std::ptr::read_volatile(g.read_ptr_host.as_ptr()) };
    let read = absolute_pm4_read_idx(g.write_idx, reported_read, capacity);
    let needed = g.write_idx + dwords as u64;
    (needed - read > capacity as u64).then(|| (needed - capacity as u64, read))
}

fn aql_shortfall(g: &QueueInner, packets: usize) -> Option<(u64, u64)> {
    let slots = (g.ring_size / AQL_PACKET_BYTES) as u64;
    let read = unsafe { std::ptr::read_volatile(g.read_ptr_host.as_ptr()) };
    let needed = g.write_idx + packets as u64;
    (needed - read > slots).then(|| (needed - slots, read))
}

fn sdma_shortfall(g: &QueueInner, published: u64) -> Option<(u64, u64)> {
    let read = unsafe { std::ptr::read_volatile(g.read_ptr_host.as_ptr()) };
    let needed = g.write_idx + published;
    (needed - read > g.ring_size as u64).then(|| (needed - g.ring_size as u64, read))
}

fn wait_pm4_headroom(g: &QueueInner, dwords: usize) -> Result<()> {
    let capacity = g.ring_size / 4;
    if dwords == 0 || dwords >= capacity {
        return Err(Error::CommandStreamTooLarge { kind: "PM4 ring submission", actual: dwords, limit: capacity - 1 });
    }
    spin_until_headroom("PM4 ring headroom", HEADROOM_TIMEOUT_MS, || pm4_shortfall(g, dwords))
}

fn wait_aql_headroom(g: &QueueInner, packets: usize) -> Result<()> {
    spin_until_headroom("AQL ring headroom", HEADROOM_TIMEOUT_MS, || aql_shortfall(g, packets))
}

/// View a `[u32; 16]` AQL packet as its 64 little-endian bytes.
fn dwords_as_bytes(p: &[u32; 16]) -> &[u8] {
    // SAFETY: 16 u32 == 64 bytes, contiguous, any bit pattern valid.
    unsafe { std::slice::from_raw_parts(p.as_ptr() as *const u8, AQL_PACKET_BYTES) }
}

/// Build the PM4 SET_SH_REG + DISPATCH_DIRECT stream for a single-XCC dispatch,
/// appending into `q` (minus SQTT/PMC/dispatch_ptr).
/// The shader entry point is pre-shifted right by 8 (COMPUTE_PGM_LO/HI hold the
/// upper bits of a 256-byte-aligned address). `wave32` comes from
/// `kd.kernel_code_properties & 0x400`; `cs_w32_en` is gfx11/12-only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_exec_pm4(
    q: &mut Vec<u32>,
    rsrc1: u32,
    rsrc2: u32,
    rsrc3: u32,
    prog_addr: u64,
    user_data: &[u32],
    scratch_addr: u64,
    tmpring_size: u32,
    local: [u32; 3],
    grid: [u32; 3],
    wave32: bool,
    target_major: u32,
) {
    // 1. Pre-dispatch cache-invalidate, narrowed to skip L2 on both arches: the
    //    prologue + the prior dispatch's EOP release already handle L2, so this
    //    only needs the per-CU caches (gfx10+ GCR NO_GLI_GL2; gfx9 CP_COHER
    //    minus the TC/L2 actions).
    if target_major == 9 {
        q.extend_from_slice(&pm4::acquire_mem_gfx9_narrow());
    } else {
        q.extend_from_slice(&pm4::acquire_mem_with(pm4::GCR_FLAGS_NO_GLI_GL2));
    }

    // 2. Shader address: COMPUTE_PGM_LO/HI hold (prog_addr >> 8).
    let prog_shr = prog_addr >> 8;
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_PGM_LO, &[prog_shr as u32, (prog_shr >> 32) as u32]));

    // 3. RSRC1/2 together; RSRC3 separately (gfx9 uses a different SH offset).
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_PGM_RSRC1, &[rsrc1, rsrc2]));
    let rsrc3_reg = if target_major == 9 { pm4::COMPUTE_PGM_RSRC3_GFX9 } else { pm4::COMPUTE_PGM_RSRC3 };
    q.extend(pm4::set_sh_reg(rsrc3_reg, &[rsrc3]));

    // 4. Scratch / tmpring (valid base required for wave init on RDNA3+ even
    //    when SCRATCH_EN=0).
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_TMPRING_SIZE, &[tmpring_size]));
    let scratch_shr = scratch_addr >> 8;
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_DISPATCH_SCRATCH_BASE_LO, &[scratch_shr as u32, (scratch_shr >> 32) as u32]));

    // 5. Restart points always zero (no preempt-resume).
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_RESTART_X, &[0, 0, 0]));

    // 6. COMPUTE_USER_DATA_0..N — user SGPR pre-load (scratch desc + kernarg ptr),
    //    assembled by the caller.
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_USER_DATA_0, user_data));

    // 7. RESOURCE_LIMITS: 0 = no per-SH wave caps.
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_RESOURCE_LIMITS, &[0]));

    // 8. START_X..NUM_THREAD_Z + 2 reserved (local size in NUM_THREAD_*).
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_START_X, &[0, 0, 0, local[0], local[1], local[2], 0, 0]));

    // 9. Launch.
    let mut di = pm4::DISPATCH_INITIATOR_FORCE_START_AT_000 | pm4::DISPATCH_INITIATOR_COMPUTE_SHADER_EN;
    if target_major != 9 && wave32 {
        di |= pm4::DISPATCH_INITIATOR_CS_W32_EN;
    }
    q.extend_from_slice(&pm4::dispatch_direct(grid, di));

    // 10. CS_PARTIAL_FLUSH so the next dispatch sees clean state.
    q.extend_from_slice(&pm4::event_write(pm4::CS_PARTIAL_FLUSH, pm4::EVENT_INDEX_PARTIAL_FLUSH));
}

/// Per-copy completion-wait timeout. A staging copy that never signals means
/// a wedged SDMA engine — fail loud rather than spin forever.
const COPY_TIMEOUT_MS: u64 = 30_000;

/// Host-visible bounce-buffer size. One SDMA packet covers `SDMA_MAX_COPY_BYTES`
/// (4 MiB), so a staging buffer that size makes each chunk a single copy.
const STAGING_BYTES: usize = sdma::SDMA_MAX_COPY_BYTES;

impl AmdCopyQueue {
    /// Retain a linked-plan finalizer whose SDMA work this queue published, so
    /// teardown waits for it. Retired entries are dropped first.
    pub(crate) fn register_inflight(&self, finalizer: Arc<SubmissionFinalizer>) {
        let mut inflight = self.inflight.lock();
        inflight.retain(|entry| !entry.retired());
        inflight.push_back(finalizer);
    }

    /// Wait every registered linked-plan finalizer, then forget the retired
    /// ones. Snapshot under the lock, wait outside it.
    fn drain_inflight(&self) -> Result<()> {
        let snapshot = self.inflight.lock().iter().cloned().collect::<Vec<_>>();
        for finalizer in &snapshot {
            finalizer.wait(COPY_TIMEOUT_MS)?;
        }
        self.inflight.lock().retain(|entry| !entry.retired());
        Ok(())
    }

    /// Registered linked-plan finalizers. Test-only.
    #[cfg(test)]
    pub(crate) fn inflight_len(&self) -> usize {
        self.inflight.lock().len()
    }

    /// See [`AmdComputeQueue::quarantine`].
    fn quarantine(&mut self, reason: &str) {
        self.state = QueueState::Quarantined;
        self.inner.get_mut().quarantined = true;
        tracing::warn!(reason, "AMD copy queue quarantined");
    }

    pub fn create(allocator: &AmdAllocator) -> Result<Arc<Self>> {
        let core = Arc::clone(allocator.dev.core());
        let inner = ActivatedQueueGuard::new(
            create_queue(
                allocator,
                kfd::KFD_IOC_QUEUE_TYPE_SDMA,
                COPY_RING_BYTES,
                /*aql=*/ false,
                /*needs_cwsr=*/ false,
            )?,
            Arc::clone(&core),
        );
        let signal = core
            .signal_pool()
            .ok_or_else(|| Error::AmdAllocFailed { reason: "copy queue needs the signal pool installed first".into() })?
            .acquire()?;
        let timeline = Timeline::new(Arc::new(signal));
        let staging_buf = AmdBufferGuard::new(
            allocator.alloc_uncached_tagged(STAGING_BYTES, crate::amd::va_registry::AllocTag::Staging)?,
        );
        let (gpu, host) = match staging_buf.buffer() {
            crate::allocator::RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
            _ => return Err(Error::NotHostVisible { what: "staging buffer" }),
        };
        let staging = Mutex::new(StagingBuf { _buf: staging_buf.into_inner(), host, gpu, size: STAGING_BYTES });
        Ok(Arc::new(Self {
            inner: Mutex::new(inner.into_inner()),
            state: QueueState::Active,
            core,
            timeline,
            staging,
            inflight: Mutex::new(std::collections::VecDeque::new()),
        }))
    }

    /// Lower and publish one complete HCQ copy command buffer with one
    /// doorbell. Completion is represented by explicit `Store` finalizers in
    /// the submission; this method therefore does not add a private fence.
    pub fn submit_hcq(&self, submission: &crate::hcq::Submission) -> Result<()> {
        if let Some(err) = self.core.poison_error() {
            return Err(err);
        }
        let dwords =
            lower_hcq_sdma(submission, self.core.arch.gfx_major(), self.core.queue_event_mailbox().map(|m| m.address))?;
        if dwords.len() * 4 >= COPY_RING_BYTES {
            return Err(Error::CommandStreamTooLarge {
                kind: "SDMA ring submission",
                actual: dwords.len() * 4,
                limit: COPY_RING_BYTES - 4,
            });
        }
        let mut g = self.inner.lock();
        wait_sdma_headroom(&g, dwords.len() * 4).inspect_err(|error| self.core.poison(&error.to_string()))?;
        push_sdma(&mut g, &dwords);
        unsafe { std::ptr::write_volatile(g.write_ptr_host.as_ptr(), g.write_idx) };
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        unsafe { std::ptr::write_volatile(g.doorbell.as_ptr(), g.write_idx) };
        Ok(())
    }

    /// SDMA counterpart of
    /// [`AmdComputeQueue::wait_publication_headroom`](AmdComputeQueue::wait_publication_headroom).
    pub(crate) fn wait_publication_headroom(&self, byte_lengths: &[usize]) -> Result<()> {
        if let Some(error) = self.core.poison_error() {
            return Err(error);
        }
        let published = {
            let inner = self.inner.lock();
            linked_sdma_published_bytes(inner.write_idx, inner.ring_size, byte_lengths)?
        };
        spin_until_headroom("SDMA linked publication headroom", COPY_TIMEOUT_MS, || {
            sdma_shortfall(&self.inner.lock(), published)
        })
        .inspect_err(|error| self.core.poison(&error.to_string()))
    }

    pub(crate) fn prepare_linked_publication(&self, byte_lengths: &[usize]) -> Result<LinkedCopyPublication<'_>> {
        if let Some(error) = self.core.poison_error() {
            return Err(error);
        }
        let inner = self.inner.lock();
        let published = linked_sdma_published_bytes(inner.write_idx, inner.ring_size, byte_lengths)?;
        wait_sdma_sequence_headroom(&inner, published).inspect_err(|error| self.core.poison(&error.to_string()))?;
        Ok(LinkedCopyPublication { inner })
    }

    /// Stage host bytes into device-local VRAM at `dst_gpu`, chunked through the
    /// bounce buffer. Each chunk: memcpy host→staging, DMA staging→device.
    pub fn host_to_device(&self, dst_gpu: u64, src: &[u8]) -> Result<()> {
        let st = self.staging.lock();
        let mut off = 0usize;
        while off < src.len() {
            let n = (src.len() - off).min(st.size);
            // SAFETY: staging host mapping spans `st.size`; `n <= st.size`.
            unsafe { std::ptr::copy_nonoverlapping(src[off..].as_ptr(), st.host.as_ptr(), n) };
            self.copy_fenced(st.gpu, dst_gpu + off as u64, n)?;
            off += n;
        }
        Ok(())
    }

    /// Read device-local VRAM at `src_gpu` into host bytes via the bounce
    /// buffer. Each chunk: DMA device→staging, memcpy staging→host.
    pub fn device_to_host(&self, dst: &mut [u8], src_gpu: u64) -> Result<()> {
        let st = self.staging.lock();
        let mut off = 0usize;
        while off < dst.len() {
            let n = (dst.len() - off).min(st.size);
            self.copy_fenced(src_gpu + off as u64, st.gpu, n)?;
            // SAFETY: staging host mapping spans `st.size`; `n <= st.size`.
            unsafe { std::ptr::copy_nonoverlapping(st.host.as_ptr(), dst[off..].as_mut_ptr(), n) };
            off += n;
        }
        Ok(())
    }

    /// Ring producer index (bytes). Test-only; see
    /// [`AmdComputeQueue::ring_write_idx`].
    #[cfg(test)]
    pub(crate) fn ring_write_idx(&self) -> u64 {
        self.inner.lock().write_idx
    }

    /// Direct device→device VRAM copy (no staging needed).
    pub fn device_to_device(&self, dst_gpu: u64, src_gpu: u64, size: usize) -> Result<()> {
        self.copy_fenced(src_gpu, dst_gpu, size)
    }

    /// Zero `size` bytes of device-local VRAM at `dst_gpu`. The hardware
    /// zero-init path only covers host-mapped buffers (`iface` skips device-only
    /// VRAM), so a device-local `zero=true` allocation is filled here: zero the
    /// staging buffer once, then DMA it over the destination in chunks.
    pub fn device_zero(&self, dst_gpu: u64, size: usize) -> Result<()> {
        let st = self.staging.lock();
        // Each chunk DMAs at most `min(remaining, st.size)` bytes from the front
        // of staging, so only that many need to be zero — no point memsetting the
        // whole 4 MiB buffer when zeroing a small allocation.
        // SAFETY: staging host mapping spans `st.size`; `size.min(st.size) ≤ st.size`.
        unsafe { std::ptr::write_bytes(st.host.as_ptr(), 0, size.min(st.size)) };
        let mut off = 0usize;
        while off < size {
            let n = (size - off).min(st.size);
            self.copy_fenced(st.gpu, dst_gpu + off as u64, n)?;
            off += n;
        }
        Ok(())
    }

    /// Copy `size` bytes `src` → `dst` (both GPU VAs) and block until the SDMA
    /// engine has finished. Chunks at `SDMA_MAX_COPY_BYTES`, fences the batch on
    /// the queue's timeline, rings the doorbell, then busy-polls the signal slot
    /// the fence writes. The reserve→push→doorbell sequence is serialised under
    /// `inner`; the busy-poll wait runs after the lock is released (so the lock
    /// is never held across a multi-second GPU wait).
    pub fn copy_fenced(&self, src: u64, dst: u64, size: usize) -> Result<()> {
        if let Some(error) = self.core.poison_error() {
            return Err(error);
        }
        if size == 0 {
            return Ok(());
        }
        let finalizer = {
            let mut g = self.inner.lock();
            let copy_packets = size.div_ceil(sdma::SDMA_MAX_COPY_BYTES);
            // Copy chunks + the timeline fence + the event mailbox fence and trap.
            wait_sdma_headroom(&g, copy_packets * 7 * 4 + 10 * 4)
                .inspect_err(|error| self.core.poison(&error.to_string()))?;
            let mut ring = RingRollback::new(&mut g);
            let mut off = 0usize;
            while off < size {
                let n = (size - off).min(sdma::SDMA_MAX_COPY_BYTES);
                push_sdma(&mut ring, &sdma::copy_linear(src + off as u64, dst + off as u64, n));
                off += n;
            }
            // Reserve + fence the timeline value the host waits on.
            let target = self.timeline.next();
            let mut reservation = CopyTimelineReservation::new(&self.timeline, &self.core, target);
            self.core.publication_checkpoint(crate::amd::iface::PublicationStage::AfterReservation)?;
            push_sdma(&mut ring, &sdma::fence(self.timeline.value_addr(), target as u32, self.core.arch.gfx_major()));
            if let Some(mailbox) = self.core.queue_event_mailbox() {
                // Tinygrad `AMDCopyQueue.signal` (ops_amd.py:490-492).
                push_sdma(&mut ring, &sdma::fence(mailbox.address, mailbox.event_id, self.core.arch.gfx_major()));
                push_sdma(&mut ring, &sdma::trap(mailbox.event_id));
            }
            self.core.publication_checkpoint(crate::amd::iface::PublicationStage::BeforeDoorbell)?;
            // GART wptr first, then doorbell — same ordering as the compute
            // queue. SDMA doorbell + wptr are byte counters (= write_idx). Both
            // are written after the last abortable step, so an abandoned
            // submission never leaves an advertised wptr behind.
            unsafe { std::ptr::write_volatile(ring.write_ptr_host.as_ptr(), ring.write_idx) };
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            unsafe { std::ptr::write_volatile(ring.doorbell.as_ptr(), ring.write_idx) };
            reservation.mark_published();
            ring.commit();
            self.core.publication_checkpoint(crate::amd::iface::PublicationStage::AfterDoorbell)?;
            reservation.commit();
            SubmissionFinalizer::timeline(Arc::clone(self.timeline.signal()), target, None)
        };
        // Wait for this submission's exact fence outside the queue lock. At the
        // rollover watermark, drain any later reservations before resetting the
        // shared 32-bit memory timeline. Coherence for copied data is handled by
        // the consuming compute dispatch's full acquire_mem prologue.
        finalizer.wait(COPY_TIMEOUT_MS).inspect_err(|error| self.core.poison(&error.to_string()))?;
        if self.timeline.current() > crate::amd::signal::TIMELINE_WRAP_WATERMARK {
            // Serialize the snapshot, wait, and reset against all SDMA timeline
            // reservations. Otherwise another copy can reserve after the drain
            // snapshot and have its in-flight generation reset underneath it.
            let _queue = self.inner.lock();
            if self.timeline.current() > crate::amd::signal::TIMELINE_WRAP_WATERMARK {
                self.timeline.drain(COPY_TIMEOUT_MS).inspect_err(|error| self.core.poison(&error.to_string()))?;
                self.timeline.reset_after_drain();
            }
        }
        Ok(())
    }
}

/// Append SDMA dwords into the byte-indexed copy ring, padding with NOPs
/// (op 0 = zero) to the ring end when a packet would straddle the wrap point —
/// the SDMA engine misparses a torn packet. `write_idx` is a monotonic byte
/// counter; the doorbell publishes it verbatim.
fn push_sdma(g: &mut QueueInner, dwords: &[u32]) {
    let bytes = std::mem::size_of_val(dwords);
    let pos = (g.write_idx as usize) % g.ring_size;
    if pos + bytes > g.ring_size {
        let pad = g.ring_size - pos;
        // SAFETY: ring_host spans ring_size bytes; pos + pad == ring_size.
        unsafe { std::ptr::write_bytes(g.ring_host.as_ptr().add(pos), 0, pad) };
        g.write_idx += pad as u64;
    }
    let pos = (g.write_idx as usize) % g.ring_size;
    // SAFETY: pos + bytes ≤ ring_size after the wrap guard above.
    unsafe { std::ptr::copy_nonoverlapping(dwords.as_ptr() as *const u8, g.ring_host.as_ptr().add(pos), bytes) };
    g.write_idx += bytes as u64;
}

fn push_sdma_bytes(g: &mut QueueInner, bytes: &[u8]) {
    let pos = (g.write_idx as usize) % g.ring_size;
    if pos + bytes.len() > g.ring_size {
        let pad = g.ring_size - pos;
        // SAFETY: ring_host spans ring_size bytes; pos + pad == ring_size.
        unsafe { std::ptr::write_bytes(g.ring_host.as_ptr().add(pos), 0, pad) };
        g.write_idx += pad as u64;
    }
    let pos = (g.write_idx as usize) % g.ring_size;
    // SAFETY: the prepared publication validated alignment, length, and wrap.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), g.ring_host.as_ptr().add(pos), bytes.len()) };
    g.write_idx += bytes.len() as u64;
}

fn wait_sdma_headroom(g: &QueueInner, bytes: usize) -> Result<()> {
    if bytes >= g.ring_size {
        return Err(Error::CommandStreamTooLarge {
            kind: "SDMA ring submission",
            actual: bytes,
            limit: g.ring_size - 4,
        });
    }
    let pos = (g.write_idx as usize) % g.ring_size;
    let published = (bytes + if pos + bytes > g.ring_size { g.ring_size - pos } else { 0 }) as u64;
    spin_until_headroom("SDMA ring headroom", COPY_TIMEOUT_MS, || sdma_shortfall(g, published))
}

fn wait_sdma_sequence_headroom(g: &QueueInner, published: u64) -> Result<()> {
    spin_until_headroom("SDMA linked transaction headroom", COPY_TIMEOUT_MS, || sdma_shortfall(g, published))
}

impl std::fmt::Debug for AmdComputeQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdComputeQueue").field("gpu_id", &self.core.node.gpu_id).finish_non_exhaustive()
    }
}

impl std::fmt::Debug for AmdCopyQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdCopyQueue").field("gpu_id", &self.core.node.gpu_id).finish_non_exhaustive()
    }
}

fn create_queue(
    allocator: &AmdAllocator,
    queue_type: u32,
    ring_size: usize,
    aql: bool,
    needs_cwsr: bool,
) -> Result<QueueInner> {
    let dev = allocator.dev.core();
    // Ring + GART are both VRAM with COHERENT | UNCACHED | PUBLIC flags
    // (uncached + cpu-accessible). Using plain VRAM (no UNCACHED) makes
    // KFD reject the create_queue ioctl with EINVAL.
    let ring_buf =
        AmdBufferGuard::new(allocator.alloc_uncached_tagged(ring_size, crate::amd::va_registry::AllocTag::QueueRing)?);
    let (ring_gpu, ring_host) = match ring_buf.buffer() {
        crate::allocator::RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
        _ => return Err(Error::NotHostVisible { what: "queue ring" }),
    };
    // Pre-fill every AQL ring slot's header with INVALID (ROCr does this) so the
    // packet processor never treats an unpublished slot (zeroed = VENDOR_SPECIFIC)
    // as a real packet if it reads ahead. PM4 rings don't use AQL headers.
    if aql {
        for slot in 0..(ring_size / AQL_PACKET_BYTES) {
            // SAFETY: slot*64 < ring_size; dword 0 is the header.
            unsafe {
                std::ptr::write_volatile(
                    ring_host.as_ptr().add(slot * AQL_PACKET_BYTES) as *mut u32,
                    INVALID_AQL_HEADER,
                )
            };
        }
    }
    // GART page holds the AQL queue descriptor (`amd_queue_t`, 256 bytes).
    // rptr/wptr live at fixed offsets inside it; KFD reads the descriptor
    // when wiring up the queue. The GART page is a 0x100-byte uncached,
    // cpu-accessible allocation.
    let gart_buf =
        AmdBufferGuard::new(allocator.alloc_uncached_tagged(0x100, crate::amd::va_registry::AllocTag::QueueGart)?);
    let (gart_gpu, gart_host) = match gart_buf.buffer() {
        crate::allocator::RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
        _ => return Err(Error::NotHostVisible { what: "GART page" }),
    };

    let mut qinactive_buf: Option<AmdBufferGuard> = None;
    let mut qinactive_host: Option<NonNull<u8>> = None;
    if aql {
        // A host-visible amd_signal_t the CP trap handler writes its exception
        // code into (e.g. 0x401 insufficient-scratch) when it halts the queue.
        // Without a real handle the CP can't report WHY it halted (silent wedge).
        let qi_buf =
            AmdBufferGuard::new(allocator.alloc_uncached_tagged(64, crate::amd::va_registry::AllocTag::QueueInactive)?);
        let (qi_gpu, qi_host) = match qi_buf.buffer() {
            crate::allocator::RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
            _ => return Err(Error::NotHostVisible { what: "queue_inactive_signal" }),
        };
        // SAFETY: fresh 64-byte buffer. amd_signal_t: kind=USER@0, value=0@8.
        unsafe {
            std::ptr::write_bytes(qi_host.as_ptr(), 0, 64);
            std::ptr::write_volatile(
                qi_host.as_ptr() as *mut i64,
                crate::amd::sys::hsa::amd_signal_kind_t_AMD_SIGNAL_KIND_USER as i64,
            );
        }
        // Initialize the GART descriptor.
        // max_cu_id is total CUs across all XCCs - 1 (cu_cnt*xccs-1).
        let cu_cnt = dev.node.simd_count.max(1) / dev.node.simd_per_cu.max(1);
        let waves_per_cu = dev.node.max_waves_per_simd * dev.node.simd_per_cu;
        // `queue_properties` is the u32 storage field; the generated property
        // constants are `amd_queue_properties_t` (c_int), narrowed here.
        let desc = crate::amd::sys::hsa::amd_queue_t {
            queue_properties: (crate::amd::sys::hsa::amd_queue_properties_t_AMD_QUEUE_PROPERTIES_IS_PTR64
                | crate::amd::sys::hsa::amd_queue_properties_t_AMD_QUEUE_PROPERTIES_ENABLE_PROFILING)
                as u32,
            read_dispatch_id_field_base_byte_offset: crate::amd::sys::hsa::OFFSET_READ_DISPATCH_ID as u32,
            max_cu_id: cu_cnt.saturating_sub(1),
            max_wave_id: waves_per_cu.saturating_sub(1),
            queue_inactive_signal: crate::amd::sys::hsa::hsa_signal_t { handle: qi_gpu },
            ..Default::default()
        };
        // SAFETY: gart_host points to a 4 KiB region we just allocated.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &desc as *const _ as *const u8,
                gart_host.as_ptr(),
                std::mem::size_of::<crate::amd::sys::hsa::amd_queue_t>(),
            );
        }
        qinactive_buf = Some(qi_buf);
        qinactive_host = Some(qi_host);
    }

    // Both AQL and plain COMPUTE queues use the same rptr/wptr offsets — the
    // `amd_queue_t::{read,write}_dispatch_id` byte offsets are passed
    // unconditionally. KFD validates these
    // against the queue type's expected layout; using (0, 8) for plain
    // COMPUTE produces EINVAL.
    let wptr_offset: u64 = crate::amd::sys::hsa::OFFSET_WRITE_DISPATCH_ID as u64;
    let rptr_offset: u64 = crate::amd::sys::hsa::OFFSET_READ_DISPATCH_ID as u64;
    // Compute queues need EOP + ctx-save buffers. Sizing:
    //   ctx_save_restore_size (ioctl arg) = wg_data_size + ctl_stack_size
    //   cwsr_buffer_size (alloc size)     = round_up((ctx_save_restore_size
    //                                          + debug_memory_size) * xccs,
    //                                          PAGESIZE)
    // The buffer is larger than what we
    // tell KFD by `debug_memory_size * xccs` — that overflow region is where
    // KFD writes debug-trap state. Undersizing causes corruption when CWSR
    // fires; oversizing is harmless.
    //
    // EOP and ctx-save are *plain VRAM* (no PUBLIC/COHERENT/UNCACHED flags):
    // they're written by the GPU during preemption and never read from the
    // CPU, so the default allocation flags suffice.
    // SDMA queues take no EOP/ctx-save buffers (CWSR is a compute-shader
    // preemption mechanism) — both are zero for SDMA. Compute queues
    // size them per the CWSR contract below.
    let (eop_buf, ctx_buf, eop_gpu, ctx_gpu, eop_size, ctx_save_restore_size, ctl_stack_size) = if needs_cwsr {
        let (wg_data_size, ctl_stack_size, debug_memory_size) = compute_ctx_sizes(dev);
        let xccs = dev.node.num_xcc.max(1) as usize;
        let ctx_save_restore_size = wg_data_size + ctl_stack_size;
        let cwsr_buffer_size = ((ctx_save_restore_size + debug_memory_size) * xccs).next_multiple_of(0x1000);
        let plain = BufferSpec { cpu_access: false, nolru: true, ..Default::default() };
        let eop_buf = AmdBufferGuard::new(allocator.alloc(0x1000, &plain, /*zero=*/ false)?);
        // ctx-save MUST be host-visible and zeroed: we write the per-XCC CWSR
        // header (`HsaUserContextSaveAreaHeader`) the CP reads on every context
        // save/restore (MES preempts a busy queue as routine runlist scheduling).
        // Without the header, a restore reads garbage `DebugOffset`/`DebugSize`
        // and the queue silently strands (rptr frozen, no fault) — the exact
        // multi-XCC wedge. Mirrors libhsakmt `fill_cwsr_header`.
        let ctx_spec = BufferSpec { cpu_access: true, nolru: true, ..Default::default() };
        let ctx_buf = AmdBufferGuard::new(allocator.alloc(cwsr_buffer_size, &ctx_spec, /*zero=*/ true)?);
        let eop_gpu = match eop_buf.buffer() {
            crate::allocator::RawBuffer::AmdDevice { gpu_addr, .. } => *gpu_addr,
            _ => 0,
        };
        let (ctx_gpu, ctx_host) = match ctx_buf.buffer() {
            crate::allocator::RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
            _ => return Err(Error::NotHostVisible { what: "ctx-save buffer" }),
        };
        // Per-XCC `HsaUserContextSaveAreaHeader` (40 bytes): DebugOffset@16,
        // DebugSize@20 (ErrorReason@24 / ErrorEventId@32 stay 0 — no event).
        // SAFETY: ctx_host is the zeroed cwsr_buffer_size region; each header sits
        // at `i * ctx_save_restore_size`, well within the buffer.
        unsafe {
            for i in 0..xccs {
                let hdr = ctx_host.as_ptr().add(i * ctx_save_restore_size);
                std::ptr::write_volatile(hdr.add(16) as *mut u32, ((xccs - i) * ctx_save_restore_size) as u32);
                std::ptr::write_volatile(hdr.add(20) as *mut u32, (debug_memory_size * xccs) as u32);
            }
        }
        (Some(eop_buf), Some(ctx_buf), eop_gpu, ctx_gpu, 0x1000u64, ctx_save_restore_size as u32, ctl_stack_size as u32)
    } else {
        (None, None, 0u64, 0u64, 0u64, 0u32, 0u32)
    };
    let _ = aql; // queue_type already encodes AQL vs plain COMPUTE

    // CREATE_QUEUE + doorbell mmap through the backend seam. The ring/GART/EOP/
    // ctx buffers above are allocated by us (above the seam); the iface only
    // activates the HQD (register the queue + map its doorbell).
    let desc = crate::amd::iface::RingDesc {
        ring_gpu,
        gart_gpu,
        wptr_offset,
        rptr_offset,
        eop_gpu,
        eop_size,
        ctx_gpu,
        ctx_save_restore_size,
        ctl_stack_size,
        ring_size,
        gpu_id: dev.node.gpu_id,
        queue_type,
    };
    let qh = match dev.iface().setup_ring(&desc) {
        Ok(handle) => handle,
        Err(error @ Error::AmdQueueStillActive { .. }) => {
            dev.poison(&error.to_string());
            return Err(error);
        }
        Err(error) => return Err(error),
    };
    let queue_id = qh.queue_id;
    let doorbell = qh.doorbell;
    let doorbell_base = qh.doorbell_base;

    // SAFETY: gart_host points to the GART page we just mmapped; the
    // write_dispatch_id field lives at a fixed offset inside the AmdQueueT
    // descriptor we wrote into the page.
    let write_ptr_host = unsafe { NonNull::new_unchecked(gart_host.as_ptr().add(wptr_offset as usize) as *mut u64) };
    let read_ptr_host = unsafe { NonNull::new_unchecked(gart_host.as_ptr().add(rptr_offset as usize) as *mut u64) };

    Ok(QueueInner {
        ring_host,
        ring_gpu,
        ring_size,
        doorbell,
        doorbell_base,
        write_ptr_host,
        read_ptr_host,
        gart_host,
        write_idx: 0,
        queue_id,
        quarantined: false,
        qinactive_host,
        _ring_buf: ring_buf.into_inner(),
        _gart_buf: gart_buf.into_inner(),
        _eop_buf: eop_buf.map(AmdBufferGuard::into_inner),
        _ctx_buf: ctx_buf.map(AmdBufferGuard::into_inner),
        _qinactive_buf: qinactive_buf.map(AmdBufferGuard::into_inner),
    })
}

/// Compute (wg_data_size, ctl_stack_size, debug_memory_size) for the ctx-save /
/// restore region.
fn compute_ctx_sizes(dev: &AmdDeviceCore) -> (usize, usize, usize) {
    const PAGE: usize = 0x1000;
    let sgrp_per_cu: usize = 0x4000;
    let hwreg_per_cu: usize = 0x1000;
    let is_cdna4 = dev.arch == svod_dtype::AmdArch::Gfx950;
    let lds_per_cu: usize = if is_cdna4 { (dev.node.lds_size_in_kb as usize) << 10 } else { 0x10000 };

    // VGPR-per-CU branches on a small whitelist of gfx-target
    // tuples: CDNA (gfx9.x) uses 0x80000, the listed
    // RDNA3/RDNA4 tuples use 0x60000, Gfx1102 alone uses 0x40000.
    let vgpr_per_cu: usize = match dev.arch {
        svod_dtype::AmdArch::Gfx942 | svod_dtype::AmdArch::Gfx950 => 0x80000,
        svod_dtype::AmdArch::Gfx1100
        | svod_dtype::AmdArch::Gfx1101
        | svod_dtype::AmdArch::Gfx1151
        | svod_dtype::AmdArch::Gfx1200
        | svod_dtype::AmdArch::Gfx1201 => 0x60000,
        svod_dtype::AmdArch::Gfx1102 => 0x40000,
    };

    let xccs = dev.node.num_xcc.max(1) as usize;
    let cu_cnt = ((dev.node.simd_count.max(1) / dev.node.simd_per_cu.max(1)) as usize / xccs).max(1);
    let waves_per_cu = (dev.node.max_waves_per_simd as usize) * (dev.node.simd_per_cu as usize);
    let wave_cnt = if dev.arch.is_cdna() {
        // gfx9 caps waves at min(cu_cnt*40, se_cnt*xccs*512). se_cnt is the
        // shader-engine count per XCC (array_count / simd_arrays_per_engine /
        // xccs). KFD >= 6.11 sizes the CWSR debug-memory area from this exact
        // formula and rejects CREATE_QUEUE with EINVAL if our buffer is short,
        // so it must match the kernel's `kfd_queue_acquire_buffers` value.
        let se_cnt = (dev.node.array_count as usize / (dev.node.simd_arrays_per_engine.max(1) as usize) / xccs).max(1);
        (cu_cnt * 40).min(se_cnt * xccs * 512)
    } else {
        cu_cnt * waves_per_cu
    };

    let wg_data_size = (vgpr_per_cu + sgrp_per_cu + lds_per_cu + hwreg_per_cu) * cu_cnt;
    let wg_data_size = wg_data_size.next_multiple_of(PAGE);

    let waves_factor = if dev.arch.is_cdna() { 8 } else { 12 };
    let ctl_stack_size = (waves_factor * wave_cnt + 8 + 40).next_multiple_of(PAGE);
    // `debug_memory_size = round_up(wave_cnt * 32, 64)`.
    let debug_memory_size = (wave_cnt * 32).next_multiple_of(64);

    (wg_data_size, ctl_stack_size, debug_memory_size)
}
