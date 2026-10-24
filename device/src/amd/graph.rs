//! AMD graph capture through backend-neutral HCQ submissions.
//!
//! Capture lowers and links one immutable native command stream. Replay updates
//! canonical kernargs plus the linked stream's runtime/system patch sites and
//! publishes it with one doorbell. No AQL packet or PM4 stream is rebuilt.

#![cfg(unix)]

use std::sync::Arc;

use crate::allocator::{AmdBufferGuard, RawBuffer};
use crate::amd::AmdAllocator;
use crate::amd::connector::OwnerCtx;
use crate::amd::program::AmdProgram;
use crate::amd::queue::{
    AQL_PACKET_BYTES, Pm4LoweringState, build_pm4_indirect_buffer, lower_hcq_aql_submission_program,
    lower_hcq_pm4_command_buffer, validate_aql_packet_count,
};
use crate::device::{Graph, GraphKernel};
use crate::error::{Error, Result};
use crate::hcq::{
    AmdPm4Dispatch, Command, CommandField, ComputeDispatch, KERNARG_ALIGN, LinkPatchValues, LinkedCommandBuffer,
    PatchSource, QueueKind, ReplayCommandBuffer, RuntimePatchValues, StoreScope, Submission, SystemField,
    SystemPatchValues, kernarg_offsets,
};

struct KernargSlot {
    host: *mut u8,
    buffers: Vec<u64>,
    vals: Vec<i64>,
    buffer_count: usize,
    var_count: usize,
    record_size: usize,
    /// Built once at capture: the ABI is fixed for the graph's lifetime, so
    /// rebuilding it on every replay was pure overhead.
    layout: crate::hcq::ClikeKernargLayout,
}

enum NativeGraph {
    Aql,
    Pm4,
}

struct ReplayState {
    command: ReplayCommandBuffer,
    profile_command: Option<ReplayCommandBuffer>,
    control: Option<ReplayCommandBuffer>,
    profile_control: Option<ReplayCommandBuffer>,
    /// Arguments the graph-owned kernarg storage currently holds, so an
    /// identical replay skips repacking every slot (tinygrad's
    /// `_prev_resolved_syms` skip in `graph/hcq.py`).
    packed: Option<(Vec<u64>, Vec<i64>)>,
    #[cfg(test)]
    packs: usize,
}

struct ProfileGraph {
    linked: Arc<LinkedCommandBuffer>,
    control: Option<GraphControl>,
    signal_count: usize,
}

struct GraphControl {
    linked: Arc<LinkedCommandBuffer>,
    host: *mut u8,
    gpu: u64,
    buffer: RawBuffer,
}

impl Drop for GraphControl {
    fn drop(&mut self) {
        self.buffer.free_amd_device_in_place();
    }
}

/// A linked AMD command buffer and graph-owned canonical kernarg storage.
pub struct AmdGraph {
    owner: OwnerCtx,
    max_private: u32,
    _programs: Vec<Arc<crate::amd::program::CodeObject>>,
    native: NativeGraph,
    linked: Arc<LinkedCommandBuffer>,
    control: Option<GraphControl>,
    profile: Option<ProfileGraph>,
    state: parking_lot::Mutex<ReplayState>,
    slots: Vec<KernargSlot>,
    kernargs_buf: RawBuffer,
}

// SAFETY: raw pointers refer to the graph-owned host mapping. Replay writes are
// serialized by `state`, and drop drains the queue before freeing that mapping.
unsafe impl Send for AmdGraph {}
unsafe impl Sync for AmdGraph {}

fn lower_graph_submission(
    allocator: &AmdAllocator,
    lane: &crate::amd::connector::PoolQueue,
    submission: &Submission,
    links: &[u64],
    state: Pm4LoweringState,
    pm4: bool,
) -> Result<(Arc<LinkedCommandBuffer>, Option<GraphControl>)> {
    if pm4 {
        let lowered = lower_hcq_pm4_command_buffer(submission, state)?;
        build_pm4_indirect_buffer(0, lowered.bytes.len() / 4)?;
        let linked = lane.link(&lowered, &LinkPatchValues(links.to_vec()))?;
        let buffer = AmdBufferGuard::new(
            allocator.alloc_host_visible_tagged(lowered.bytes.len().max(16), crate::amd::va_registry::AllocTag::Gtt)?,
        );
        let (gpu, host) = match buffer.buffer() {
            RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(host), .. } => (*gpu_addr, host.as_ptr()),
            _ => return Err(Error::NotHostVisible { what: "PM4 graph command buffer" }),
        };
        return Ok((Arc::clone(&linked), Some(GraphControl { linked, host, gpu, buffer: buffer.into_inner() })));
    }

    let control_link = links.len();
    let program = lower_hcq_aql_submission_program(submission, state, PatchSource::LinkAddress(control_link))?;
    validate_aql_packet_count(program.aql.bytes.len() / AQL_PACKET_BYTES)?;
    let buffer = AmdBufferGuard::new(
        allocator
            .alloc_host_visible_tagged(program.control.bytes.len().max(16), crate::amd::va_registry::AllocTag::Gtt)?,
    );
    let (gpu, host) = match buffer.buffer() {
        RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(host), .. } => (*gpu_addr, host.as_ptr()),
        _ => return Err(Error::NotHostVisible { what: "AQL graph control program" }),
    };
    let mut native_links = links.to_vec();
    native_links.push(gpu);
    let values = LinkPatchValues(native_links);
    let control = lane.link(&program.control, &values)?;
    let linked = lane.link(&program.aql, &values)?;
    Ok((linked, Some(GraphControl { linked: control, host, gpu, buffer: buffer.into_inner() })))
}

impl Drop for AmdGraph {
    fn drop(&mut self) {
        if std::thread::panicking() {
            tracing::warn!("AmdGraph drop during panic unwind: in-flight replay abandoned");
            return;
        }
        if let Err(error) = self.owner.synchronize() {
            tracing::warn!(?error, "AmdGraph drop: synchronize failed; storage quarantined");
            return;
        }
        self.kernargs_buf.free_amd_device_in_place();
    }
}

impl AmdGraph {
    pub fn capture(allocator: &AmdAllocator, kernels: &[GraphKernel]) -> Result<Option<Box<dyn Graph>>> {
        Ok(Self::capture_amd(allocator, kernels)?.map(|graph| graph as Box<dyn Graph>))
    }

    /// Immutable linked command stream as captured. Test-only.
    #[cfg(test)]
    pub(crate) fn linked_bytes(&self) -> &[u8] {
        self.linked.static_bytes()
    }

    pub(crate) fn capture_amd(allocator: &AmdAllocator, kernels: &[GraphKernel]) -> Result<Option<Box<Self>>> {
        if kernels.is_empty() {
            return Ok(None);
        }
        let mut programs = Vec::with_capacity(kernels.len());
        for kernel in kernels {
            let Some(program) = kernel.program.as_any().downcast_ref::<AmdProgram>() else { return Ok(None) };
            programs.push(program);
        }
        let device = Arc::clone(programs[0].device());
        if programs.iter().skip(1).any(|program| !Arc::ptr_eq(program.device(), &device)) {
            return Ok(None);
        }
        if let Some(error) = device.core().poison_error() {
            return Err(error);
        }
        let pm4 = crate::amd::queue::AmdComputeQueue::will_use_pm4(device.core());
        if pm4 && !device.core().pm4_graph() {
            return Ok(None);
        }

        let owner = OwnerCtx::new(Arc::clone(device.core()), allocator.clone());
        let lane = owner.lease()?;
        let max_private = programs.iter().map(|p| p.private_segment_size()).max().unwrap_or(128).max(128);
        lane.ensure_has_local_memory(max_private)?;

        for (kernel, program) in kernels.iter().zip(&programs) {
            let (buffer_count, var_count) = program.arg_counts();
            if kernel.buffers.len() != buffer_count || kernel.vals.len() != var_count {
                return Err(Error::ProgramAbiMismatch {
                    reason: format!(
                        "AMD graph kernel '{}' expected {buffer_count} buffers/{var_count} vars, got {}/{}",
                        kernel.program.name(),
                        kernel.buffers.len(),
                        kernel.vals.len()
                    ),
                });
            }
        }
        let (offsets, bytes) =
            kernarg_offsets(programs.iter().map(|program| program.kernarg_record_size()), KERNARG_ALIGN);
        let kernargs_buf = AmdBufferGuard::new(
            allocator.alloc_host_visible_tagged(bytes.max(16), crate::amd::va_registry::AllocTag::Kernarg)?,
        );
        let (kernargs_gpu, kernargs_host) = match kernargs_buf.buffer() {
            RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(host), .. } => (*gpu_addr, host.as_ptr()),
            _ => return Err(Error::NotHostVisible { what: "graph kernargs" }),
        };

        let mut submission = Submission::new(QueueKind::Compute(0));
        let wait = submission.commands.len();
        submission.push(Command::Wait { signal_address: 0, value: 0 });
        submission.bind(wait, CommandField::WaitAddress, PatchSource::System(SystemField::TimelineSignal(0)))?;
        submission.bind(wait, CommandField::WaitValue, PatchSource::System(SystemField::TimelineValue(0)))?;
        // ONE memory barrier per graph, matching tinygrad's
        // `comp_queues[dev].memory_barrier()` at the head of a captured device
        // queue (`graph/hcq.py:157`). Per-dispatch coherence is the narrow
        // `acquire_mem` + `CS_PARTIAL_FLUSH` that `build_exec_pm4` already
        // emits, and the AQL packets carry the header BARRIER bit; a full HDP
        // flush plus full-invalidate acquire per kernel bought nothing. Morok
        // keeps the barrier AFTER the timeline wait, matching its own
        // `Wait -> MemoryBarrier -> Compute` per-call finalization.
        submission.push(Command::MemoryBarrier);
        let mut links = Vec::with_capacity(kernels.len() * 2);
        let mut slots = Vec::with_capacity(kernels.len());
        for (((kernel, program), &offset), index) in kernels.iter().zip(&programs).zip(&offsets).zip(0usize..) {
            let slot_gpu = kernargs_gpu + offset as u64;
            let (buffer_count, var_count) = program.arg_counts();
            slots.push(KernargSlot {
                // SAFETY: offsets were packed within the graph allocation.
                host: unsafe { kernargs_host.add(offset) },
                buffers: kernel.buffers.iter().map(|p| *p as u64).collect(),
                vals: kernel.vals.clone(),
                buffer_count,
                var_count,
                record_size: program.kernarg_record_size(),
                layout: crate::hcq::ClikeKernargLayout::from_abi(program.abi()),
            });

            let g = kernel.global_size.unwrap_or([1, 1, 1]);
            let l = kernel.local_size.unwrap_or([1, 1, 1]);
            let (rsrc1, rsrc2, rsrc3) = program.rsrc();
            let (wave32, target_major) = program.wave32_target();
            let command = submission.commands.len();
            submission.push(Command::Compute(ComputeDispatch {
                workgroup_size: l.map(|v| v as u32),
                grid_size: if pm4 {
                    g.map(|v| v as u32)
                } else {
                    [g[0] * l[0], g[1] * l[1], g[2] * l[2]].map(|v| v as u32)
                },
                private_segment_size: program.private_segment_size(),
                group_segment_size: program.group_segment_size(),
                kernel_object: 0,
                kernarg_address: 0,
                completion_signal: 0,
                barrier: true,
                amd_pm4: Some(AmdPm4Dispatch {
                    rsrc: [rsrc1, rsrc2, rsrc3],
                    program_address: 0,
                    enable_private_segment_sgpr: program.enable_private_segment_sgpr(),
                    workgroup_count: g.map(|v| v as u32),
                    wave32,
                    target_major,
                }),
            }));
            let program_link = links.len();
            links.push(if pm4 { program.pm4_prog_addr() } else { program.aql_prog_addr() });
            let kernarg_link = links.len();
            links.push(slot_gpu);
            submission.bind(
                command,
                if pm4 { CommandField::ComputeProgramAddress } else { CommandField::ComputeKernelObject },
                PatchSource::LinkAddress(program_link),
            )?;
            submission.bind(command, CommandField::ComputeKernargAddress, PatchSource::LinkAddress(kernarg_link))?;
            if pm4 {
                submission.bind(
                    command,
                    CommandField::ComputeScratchAddress,
                    PatchSource::System(SystemField::ScratchAddress),
                )?;
                submission.bind(
                    command,
                    CommandField::ComputeScratchTmpring,
                    PatchSource::System(SystemField::ScratchTmpring),
                )?;
            }
            let _ = (index, &kernel.deps); // FIFO barriers conservatively satisfy every dependency edge.
        }
        let store = submission.commands.len();
        submission.push(Command::Store { dst: 0, value: 0, scope: StoreScope::System });
        submission.bind(store, CommandField::StoreDst, PatchSource::System(SystemField::TimelineSignal(0)))?;
        submission.bind(store, CommandField::StoreValue, PatchSource::System(SystemField::TimelineValue(1)))?;

        let profile_linked = {
            let mut profiled = submission.clone();
            profiled.request_profile();
            let computes: Vec<usize> = profiled
                .commands
                .iter()
                .enumerate()
                .filter_map(|(index, command)| matches!(command, Command::Compute(_)).then_some(index))
                .collect();
            for (slot, &index) in computes.iter().enumerate().rev() {
                profiled.insert(index + 1, Command::Timestamp { dst: 0 });
                profiled.bind(
                    index + 1,
                    CommandField::TimestampDst,
                    PatchSource::System(SystemField::Timestamp((slot * 2 + 1) as u32)),
                )?;
                profiled.insert(index, Command::Timestamp { dst: 0 });
                profiled.bind(
                    index,
                    CommandField::TimestampDst,
                    PatchSource::System(SystemField::Timestamp((slot * 2) as u32)),
                )?;
            }
            let state = Pm4LoweringState {
                scratch_address: lane.scratch_gpu_va(),
                tmpring_size: lane.tmpring_size(),
                target_major: device.core().arch.gfx_major(),
                completion_xcc_mask: (!pm4 && device.core().node.num_xcc > 1).then_some(1),
                // Captured timeline stores are placeholders patched per replay, so
                // they never carry the KFD interrupt companion.
                queue_event_mailbox: None,
            };
            Some(lower_graph_submission(allocator, &lane, &profiled, &links, state, pm4)?)
        };

        let state = Pm4LoweringState {
            scratch_address: lane.scratch_gpu_va(),
            tmpring_size: lane.tmpring_size(),
            target_major: device.core().arch.gfx_major(),
            completion_xcc_mask: (!pm4 && device.core().node.num_xcc > 1).then_some(1),
            // Captured timeline stores are placeholders patched per replay, so
            // they never carry the KFD interrupt companion.
            queue_event_mailbox: None,
        };
        let (linked, control) = lower_graph_submission(allocator, &lane, &submission, &links, state, pm4)?;
        let command = linked.replay_buffer();
        let profile_command = profile_linked.as_ref().map(|(linked, _)| linked.replay_buffer());
        let control_command = control.as_ref().map(|control| control.linked.replay_buffer());
        let profile_control_command = profile_linked
            .as_ref()
            .and_then(|(_, control)| control.as_ref().map(|control| control.linked.replay_buffer()));
        Ok(Some(Box::new(Self {
            owner,
            max_private,
            _programs: programs.iter().map(|program| program.code_object()).collect(),
            native: if pm4 { NativeGraph::Pm4 } else { NativeGraph::Aql },
            linked,
            control,
            profile: profile_linked.map(|(linked, control)| ProfileGraph {
                linked,
                control,
                signal_count: kernels.len(),
            }),
            state: parking_lot::Mutex::new(ReplayState {
                command,
                profile_command,
                control: control_command,
                profile_control: profile_control_command,
                packed: None,
                #[cfg(test)]
                packs: 0,
            }),
            slots,
            kernargs_buf: kernargs_buf.into_inner(),
        })))
    }

    /// Pack graph kernargs the way a replay does and report the number of packs
    /// performed so far. Test-only: proving the skip through a real replay
    /// needs a GPU to retire the first submission.
    #[cfg(test)]
    pub(crate) fn kernarg_pack_probe(&self, buffers: &[u64], vals: &[i64]) -> Result<usize> {
        let mut state = self.state.lock();
        self.patch_kernargs(&mut state, buffers, vals)?;
        Ok(state.packs)
    }

    fn patch_kernargs(&self, state: &mut ReplayState, buffers: &[u64], vals: &[i64]) -> Result<()> {
        let expected_buffers: usize = self.slots.iter().map(|slot| slot.buffer_count).sum();
        let expected_vals: usize = self.slots.iter().map(|slot| slot.var_count).sum();
        if (!buffers.is_empty() && buffers.len() != expected_buffers)
            || (!vals.is_empty() && vals.len() != expected_vals)
        {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "AMD graph replay expected {expected_buffers} buffers/{expected_vals} vars, got {}/{}",
                    buffers.len(),
                    vals.len()
                ),
            });
        }
        // Graph-owned kernarg storage is written by nothing else, so identical
        // arguments mean the bytes are already correct.
        if state.packed.as_ref().is_some_and(|(packed, packed_vals)| packed == buffers && packed_vals == vals) {
            return Ok(());
        }
        let mut buffer_offset = 0;
        let mut var_offset = 0;
        for slot in &self.slots {
            let slot_buffers = if buffers.is_empty() {
                &slot.buffers
            } else {
                &buffers[buffer_offset..buffer_offset + slot.buffer_count]
            };
            let slot_vals = if vals.is_empty() { &slot.vals } else { &vals[var_offset..var_offset + slot.var_count] };
            // SAFETY: each slot is disjoint, graph-owned, and replay is serialized.
            let dst = unsafe { std::slice::from_raw_parts_mut(slot.host, slot.record_size) };
            slot.layout.pack(dst, slot_buffers, slot_vals)?;
            buffer_offset += slot.buffer_count;
            var_offset += slot.var_count;
        }
        state.packed = Some((buffers.to_vec(), vals.to_vec()));
        #[cfg(test)]
        {
            state.packs += 1;
        }
        Ok(())
    }
}

impl Graph for AmdGraph {
    fn completion_token(&self) -> Option<Arc<dyn crate::sync::CompletionToken>> {
        self.owner.completion_token()
    }

    fn replay(&self, buffers: &[u64], vals: &[i64]) -> Result<()> {
        if let Some(error) = self.owner.core().poison_error() {
            return Err(error);
        }
        let mut state = self.state.lock();
        // Not covered by the linked stream's timeline wait: that wait runs on
        // the GPU, whereas `patch_kernargs` and the replay-buffer patch below
        // rewrite graph-owned host storage a still-in-flight previous replay
        // may be reading. Retiring the owner's last submission first is what
        // makes those in-place rewrites safe.
        self.owner.synchronize()?;
        let lane = self.owner.lease()?;
        lane.ensure_has_local_memory(self.max_private)?;
        self.patch_kernargs(&mut state, buffers, vals)?;
        match self.native {
            NativeGraph::Aql => {
                let control = self.control.as_ref().expect("AQL graph control");
                let ReplayState { command, control: control_command, .. } = &mut *state;
                let mut system = SystemPatchValues::default();
                let finalizer = lane.queue().replay_linked_aql_timeline(
                    lane.pool(),
                    &self.linked,
                    command,
                    &control.linked,
                    control_command.as_mut().expect("AQL graph control replay"),
                    control.host,
                    &RuntimePatchValues::default(),
                    &mut system,
                )?;
                self.owner.set_newest(finalizer);
            }
            NativeGraph::Pm4 => {
                let resident = self.control.as_ref().expect("PM4 graph resident command buffer");
                let mut system = SystemPatchValues::default();
                let finalizer = lane.queue().replay_linked_pm4(
                    lane.pool(),
                    &self.linked,
                    &mut state.command,
                    resident.host,
                    resident.gpu,
                    &RuntimePatchValues::default(),
                    &mut system,
                )?;
                self.owner.set_newest(finalizer);
            }
        }
        Ok(())
    }

    fn replay_profiled(
        &self,
        buffers: &[u64],
        vals: &[i64],
    ) -> Result<Option<Vec<Arc<dyn crate::DispatchTimestamps>>>> {
        let Some(profile) = &self.profile else { return Ok(None) };
        if let Some(error) = self.owner.core().poison_error() {
            return Err(error);
        }
        let mut state = self.state.lock();
        // Not covered by the linked stream's timeline wait: that wait runs on
        // the GPU, whereas `patch_kernargs` and the replay-buffer patch below
        // rewrite graph-owned host storage a still-in-flight previous replay
        // may be reading. Retiring the owner's last submission first is what
        // makes those in-place rewrites safe.
        self.owner.synchronize()?;
        let lane = self.owner.lease()?;
        lane.ensure_has_local_memory(self.max_private)?;
        self.patch_kernargs(&mut state, buffers, vals)?;
        let signals = (0..profile.signal_count).map(|_| lane.acquire_timestamp_signal()).collect::<Result<Vec<_>>>()?;
        for signal in &signals {
            signal.reset(0);
        }
        let mut system = SystemPatchValues::default();
        for (slot, signal) in signals.iter().enumerate() {
            system.0.insert(SystemField::Timestamp((slot * 2) as u32), signal.start_ts_addr());
            system.0.insert(SystemField::Timestamp((slot * 2 + 1) as u32), signal.end_ts_addr());
        }
        match self.native {
            NativeGraph::Aql => {
                let control = profile.control.as_ref().expect("profile AQL graph control");
                let ReplayState { profile_command, profile_control, .. } = &mut *state;
                let finalizer = lane.queue().replay_linked_aql_timeline(
                    lane.pool(),
                    &profile.linked,
                    profile_command.as_mut().expect("profile graph command"),
                    &control.linked,
                    profile_control.as_mut().expect("profile AQL graph control replay"),
                    control.host,
                    &RuntimePatchValues::default(),
                    &mut system,
                )?;
                self.owner.set_newest(finalizer);
            }
            NativeGraph::Pm4 => {
                let resident = profile.control.as_ref().expect("profile PM4 graph resident command buffer");
                let command = state.profile_command.as_mut().expect("profile graph command");
                let finalizer = lane.queue().replay_linked_pm4(
                    lane.pool(),
                    &profile.linked,
                    command,
                    resident.host,
                    resident.gpu,
                    &RuntimePatchValues::default(),
                    &mut system,
                )?;
                self.owner.set_newest(finalizer);
            }
        }
        self.owner.synchronize()?;
        Ok(Some(signals.into_iter().map(|signal| signal as Arc<dyn crate::DispatchTimestamps>).collect()))
    }
}
