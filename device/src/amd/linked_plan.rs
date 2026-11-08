//! Native replay for ordinary dynamic `ExecutionPlan` PROGRAM/COPY calls.

use std::sync::Arc;

use crate::allocator::{AmdBufferGuard, RawBuffer};
use crate::amd::connector::{OwnerCtx, PoolQueue, SubmissionFinalizer};
use crate::amd::program::AmdProgram;
use crate::amd::queue::{
    AQL_PACKET_BYTES, COPY_RING_BYTES, Pm4LoweringState, lower_hcq_aql_submission_program,
    lower_hcq_pm4_command_buffer, lower_hcq_sdma_command_buffer, push_queue_event_mailbox, validate_aql_packet_count,
    validate_pm4_dword_count,
};
use crate::device::PlanCall;
use crate::error::{Error, Result};
use crate::hcq::{
    AmdPm4Dispatch, Command, CommandField, ComputeDispatch, KERNARG_ALIGN, LinkPatchValues, LinkedCommandBuffer,
    PatchSource, QueueKind, ReplayCommandBuffer, RuntimePatchValues, SemanticLinkedPlan, Submission,
    SubmissionTimelines, SystemField, SystemPatchValues, kernarg_offsets,
};

pub(crate) fn native_topology_decline(
    plan: &SemanticLinkedPlan,
    has_copy_queue: bool,
) -> Option<crate::device::NativeReplayDecline> {
    if let Some(operation) = plan.staged_copy() {
        return Some(crate::device::NativeReplayDecline::StagedCopy { operation });
    }
    if !has_copy_queue && plan.lanes().iter().any(|submission| matches!(submission.lane.queue, QueueKind::Copy(_))) {
        return Some(crate::device::NativeReplayDecline::BackendUnsupported);
    }
    let mut devices = plan.lanes().iter().map(|submission| &submission.lane.device);
    let expected = devices.next()?.clone();
    devices
        .find(|device| **device != expected)
        .map(|actual| crate::device::NativeReplayDecline::MixedComputeDevices { expected, actual: actual.clone() })
}

struct KernargSlot {
    operation: usize,
    host: *mut u8,
    record_size: usize,
    buffers: usize,
    vars: usize,
    abi: Vec<crate::device::AbiParamDescriptor>,
}

struct NativeSubmission {
    queue: QueueKind,
    operation_slot: Option<u32>,
    linked: Arc<LinkedCommandBuffer>,
    replay: ReplayCommandBuffer,
    control: Option<NativeControlProgram>,
}

struct NativeControlProgram {
    linked: Arc<LinkedCommandBuffer>,
    replay: ReplayCommandBuffer,
    host: *mut u8,
    buffer: RawBuffer,
}

struct PreparedPublication {
    finalizer: Arc<SubmissionFinalizer>,
    core: Arc<crate::amd::device::AmdDeviceCore>,
    published: bool,
}

pub(crate) struct ReplayFailure {
    pub(crate) error: Error,
    pub(crate) published: bool,
}

impl PreparedPublication {
    fn new(finalizer: Arc<SubmissionFinalizer>, core: Arc<crate::amd::device::AmdDeviceCore>) -> Self {
        Self { finalizer, core, published: false }
    }

    fn publish(mut self) {
        self.finalizer.mark_published();
        self.published = true;
    }
}

impl Drop for PreparedPublication {
    fn drop(&mut self) {
        if !self.published {
            self.finalizer.mark_failed();
            self.core.poison("AMD linked publication abandoned after finalizer registration");
        }
    }
}

impl Drop for NativeControlProgram {
    fn drop(&mut self) {
        self.buffer.free_amd_device_in_place();
    }
}

pub(crate) struct AmdLinkedPlan {
    pm4: bool,
    submissions: Vec<NativeSubmission>,
    slots: Vec<KernargSlot>,
    _signals: Vec<Arc<crate::amd::signal::AmdSignal>>,
    timelines: SubmissionTimelines,
    kernargs: RawBuffer,
    operation_count: usize,
    max_private: u32,
    programs: Vec<(usize, u64, u64)>,
    _code: Vec<Arc<crate::amd::program::CodeObject>>,
}

// SAFETY: pointers target `kernargs`; OwnerCtx's linked-plan mutex serializes
// writes and queue publication, and replay waits before reusing the storage.
unsafe impl Send for AmdLinkedPlan {}

impl Drop for AmdLinkedPlan {
    fn drop(&mut self) {
        self.kernargs.free_amd_device_in_place();
    }
}

impl AmdLinkedPlan {
    pub(crate) fn capture(
        owner: &OwnerCtx,
        lane: &PoolQueue,
        semantic: &SemanticLinkedPlan,
        calls: &[PlanCall<'_>],
    ) -> Result<Option<Self>> {
        if calls.iter().any(|call| matches!(call, PlanCall::Unsupported)) {
            return Ok(None);
        }
        let programs = calls
            .iter()
            .filter_map(|call| match call {
                PlanCall::Program { program, .. } => program.as_any().downcast_ref::<AmdProgram>(),
                _ => None,
            })
            .collect::<Vec<_>>();
        if programs.is_empty()
            || programs.len() != calls.iter().filter(|call| matches!(call, PlanCall::Program { .. })).count()
        {
            return Ok(None);
        }
        if programs.iter().any(|program| !Arc::ptr_eq(program.device().core(), owner.core())) {
            return Ok(None);
        }
        let max_private = programs.iter().map(|p| p.private_segment_size()).max().unwrap_or(0);
        lane.ensure_has_local_memory(max_private)?;
        let pm4 = lane.queue().is_pm4();
        let allocator = owner.allocator();
        let records: Vec<(usize, usize)> = calls
            .iter()
            .enumerate()
            .filter_map(|(operation, call)| match call {
                PlanCall::Program { program, .. } => {
                    let program = program.as_any().downcast_ref::<AmdProgram>().unwrap();
                    Some((operation, program.kernarg_record_size()))
                }
                _ => None,
            })
            .collect();
        let (packed, bytes) = kernarg_offsets(records.iter().map(|(_, size)| *size), KERNARG_ALIGN);
        let mut offsets = vec![None; calls.len()];
        for (&(operation, _), offset) in records.iter().zip(packed) {
            offsets[operation] = Some(offset);
        }
        let kernargs = AmdBufferGuard::new(
            allocator.alloc_host_visible_tagged(bytes.max(16), crate::amd::va_registry::AllocTag::Kernarg)?,
        );
        let (kernargs_gpu, kernargs_host) = match kernargs.buffer() {
            RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(host), .. } => (*gpu_addr, host.as_ptr()),
            _ => return Err(Error::NotHostVisible { what: "linked plan kernargs" }),
        };

        let semantic_submissions = semantic.native_submissions()?;
        let mut slots = Vec::new();
        let mut links = Vec::new();
        let mut native = Vec::new();
        let mut operation_count = 0usize;
        for linked_semantic in &semantic_submissions {
            let original = linked_semantic.static_submission();
            let operation_slot = original
                .commands
                .iter()
                .any(|command| matches!(command, Command::Execute { .. }))
                .then(|| operation_count as u32 + 1);
            let mut submission = Submission::new(original.queue);
            for (command_index, command) in original.commands.iter().enumerate() {
                match command {
                    Command::Execute { operation } => match &calls[*operation] {
                        PlanCall::Program { program, global_size, local_size, .. } => {
                            let program = program.as_any().downcast_ref::<AmdProgram>().unwrap();
                            let g = global_size.unwrap_or([1, 1, 1]);
                            let l = local_size.unwrap_or([1, 1, 1]);
                            let (r1, r2, r3) = program.rsrc();
                            let (wave32, target_major) = program.wave32_target();
                            let out = submission.commands.len();
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
                                    rsrc: [r1, r2, r3],
                                    program_address: 0,
                                    enable_private_segment_sgpr: program.enable_private_segment_sgpr(),
                                    workgroup_count: g.map(|v| v as u32),
                                    wave32,
                                    target_major,
                                }),
                            }));
                            let program_link = links.len();
                            links.push(if pm4 { program.pm4_prog_addr() } else { program.aql_prog_addr() });
                            submission.bind(
                                out,
                                if pm4 {
                                    CommandField::ComputeProgramAddress
                                } else {
                                    CommandField::ComputeKernelObject
                                },
                                PatchSource::LinkAddress(program_link),
                            )?;
                            let kernarg_link = links.len();
                            let offset = offsets[*operation].unwrap();
                            links.push(kernargs_gpu + offset as u64);
                            submission.bind(
                                out,
                                CommandField::ComputeKernargAddress,
                                PatchSource::LinkAddress(kernarg_link),
                            )?;
                            for axis in 0..3u8 {
                                submission.bind(
                                    out,
                                    CommandField::ComputeWorkgroup(axis),
                                    PatchSource::RuntimeVar(operation * 6 + axis as usize),
                                )?;
                                submission.bind(
                                    out,
                                    CommandField::ComputeGrid(axis),
                                    PatchSource::RuntimeVar(operation * 6 + 3 + axis as usize),
                                )?;
                            }
                            if pm4 {
                                submission.bind(
                                    out,
                                    CommandField::ComputeScratchAddress,
                                    PatchSource::System(SystemField::ScratchAddress),
                                )?;
                                submission.bind(
                                    out,
                                    CommandField::ComputeScratchTmpring,
                                    PatchSource::System(SystemField::ScratchTmpring),
                                )?;
                            }
                            let (buffers, vars) = program.arg_counts();
                            slots.push(KernargSlot {
                                operation: *operation,
                                host: unsafe { kernargs_host.add(offset) },
                                record_size: program.kernarg_record_size(),
                                buffers,
                                vars,
                                abi: program.abi().to_vec(),
                            });
                            operation_count += 1;
                        }
                        PlanCall::Copy { bytes, .. } => {
                            let out = submission.commands.len();
                            submission.push(Command::Copy { dst: 0, src: 0, bytes: *bytes });
                            submission.bind(out, CommandField::CopyDst, PatchSource::RuntimeBuffer(operation * 2))?;
                            submission.bind(
                                out,
                                CommandField::CopySrc,
                                PatchSource::RuntimeBuffer(operation * 2 + 1),
                            )?;
                            operation_count += 1;
                        }
                        PlanCall::Unsupported => unreachable!(),
                    },
                    Command::MemoryBarrier => {
                        submission.push(Command::MemoryBarrier);
                    }
                    other => {
                        let out = submission.commands.len();
                        submission.push(other.clone());
                        for patch in original.patches().iter().filter(|patch| patch.command == command_index) {
                            submission.bind(out, patch.field, patch.source)?;
                        }
                    }
                }
            }
            if submission.commands.is_empty() {
                continue;
            }
            // Each submission ends on a timeline store, and the last one is the
            // plan's: its interrupt wakes a host blocked in `WAIT_EVENTS`.
            push_queue_event_mailbox(&mut submission, lane.core());
            let queue_event_mailbox = lane.core().queue_event_mailbox().map(|mailbox| mailbox.address);
            let mut control = None;
            let lowered = match submission.queue {
                QueueKind::Compute(_) if pm4 => lower_hcq_pm4_command_buffer(
                    &submission,
                    Pm4LoweringState {
                        scratch_address: lane.scratch_gpu_va(),
                        tmpring_size: lane.tmpring_size(),
                        target_major: lane.core().arch.gfx_major(),
                        completion_xcc_mask: None,
                        queue_event_mailbox,
                    },
                )?,
                QueueKind::Compute(_) => {
                    let control_link = links.len();
                    let program = lower_hcq_aql_submission_program(
                        &submission,
                        Pm4LoweringState {
                            scratch_address: lane.scratch_gpu_va(),
                            tmpring_size: lane.tmpring_size(),
                            target_major: lane.core().arch.gfx_major(),
                            completion_xcc_mask: (lane.core().node.num_xcc > 1).then_some(1),
                            queue_event_mailbox,
                        },
                        PatchSource::LinkAddress(control_link),
                    )?;
                    let control_buffer = AmdBufferGuard::new(allocator.alloc_host_visible_tagged(
                        program.control.bytes.len().max(16),
                        crate::amd::va_registry::AllocTag::Gtt,
                    )?);
                    let (gpu, host) = match control_buffer.buffer() {
                        RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(host), .. } => (*gpu_addr, host.as_ptr()),
                        _ => return Err(Error::NotHostVisible { what: "AQL HCQ submission program" }),
                    };
                    links.push(gpu);
                    let linked_control = lane.link(&program.control, &LinkPatchValues(links.clone()))?;
                    control = Some(NativeControlProgram {
                        replay: linked_control.replay_buffer(),
                        linked: linked_control,
                        host,
                        buffer: control_buffer.into_inner(),
                    });
                    program.aql
                }
                QueueKind::Copy(_) => {
                    lower_hcq_sdma_command_buffer(&submission, lane.core().arch.gfx_major(), queue_event_mailbox)?
                }
            };
            match submission.queue {
                QueueKind::Compute(_) if pm4 => validate_pm4_dword_count(lowered.bytes.len() / 4)?,
                QueueKind::Compute(_) => validate_aql_packet_count(lowered.bytes.len() / AQL_PACKET_BYTES)?,
                QueueKind::Copy(_) if lowered.bytes.len() >= COPY_RING_BYTES => {
                    return Err(Error::CommandStreamTooLarge {
                        kind: "SDMA ring submission",
                        actual: lowered.bytes.len(),
                        limit: COPY_RING_BYTES - 4,
                    });
                }
                QueueKind::Copy(_) => {}
            }
            let linked = lane.link(&lowered, &LinkPatchValues(links.clone()))?;
            native.push(NativeSubmission {
                queue: submission.queue,
                operation_slot,
                replay: linked.replay_buffer(),
                linked,
                control,
            });
        }

        let signal_pool = owner
            .core()
            .signal_pool()
            .cloned()
            .ok_or_else(|| Error::Runtime { message: "linked plan needs AMD signal pool".into() })?;
        let mut signals = Vec::with_capacity(6);
        for _ in 0..6 {
            let signal = Arc::new(signal_pool.acquire()?);
            signal.reset(0);
            signals.push(signal);
        }
        let address = |index: usize| signals[index].value_addr();
        let timelines =
            SubmissionTimelines::new([address(0), address(1)], [address(2), address(3)], [address(4), address(5)]);
        Ok(Some(Self {
            pm4,
            submissions: native,
            slots,
            _signals: signals,
            timelines,
            kernargs: kernargs.into_inner(),
            operation_count,
            max_private,
            programs: calls
                .iter()
                .enumerate()
                .filter_map(|(operation, call)| match call {
                    PlanCall::Program { program, .. } => {
                        let program = program.as_any().downcast_ref::<AmdProgram>().unwrap();
                        Some((operation, program.pm4_prog_addr(), program.aql_prog_addr()))
                    }
                    _ => None,
                })
                .collect(),
            _code: programs.iter().map(|program| program.code_object()).collect(),
        }))
    }

    /// Each native submission's linked stream as captured. Test-only.
    #[cfg(test)]
    pub(crate) fn submission_bytes(&self) -> Vec<&[u8]> {
        self.submissions.iter().map(|submission| submission.linked.static_bytes()).collect()
    }

    pub(crate) fn replay(
        &mut self,
        owner: &OwnerCtx,
        lane: &PoolQueue,
        calls: &[PlanCall<'_>],
    ) -> std::result::Result<(), ReplayFailure> {
        let timelines_before = self.timelines.clone();
        let mut published = false;
        let result = (|| -> Result<()> {
            for &(operation, pm4_address, aql_address) in &self.programs {
                let Some(PlanCall::Program { program, .. }) = calls.get(operation) else {
                    return Err(Error::ProgramStageMismatch {
                        stage: "AMD linked replay",
                        reason: format!("operation {operation} is no longer a PROGRAM"),
                    });
                };
                let Some(program) = program.as_any().downcast_ref::<AmdProgram>() else {
                    return Err(Error::ProgramStageMismatch {
                        stage: "AMD linked replay",
                        reason: format!("operation {operation} is no longer an AMD PROGRAM"),
                    });
                };
                if !Arc::ptr_eq(program.device().core(), owner.core())
                    || program.pm4_prog_addr() != pm4_address
                    || program.aql_prog_addr() != aql_address
                {
                    return Err(Error::ProgramStageMismatch {
                        stage: "AMD linked replay",
                        reason: format!("operation {operation} changed program or physical device"),
                    });
                }
            }
            lane.ensure_has_local_memory(self.max_private)?;
            for slot in &self.slots {
                let PlanCall::Program { buffers, vals, .. } = &calls[slot.operation] else { unreachable!() };
                if buffers.len() != slot.buffers || vals.len() != slot.vars {
                    return Err(Error::ProgramAbiMismatch {
                        reason: "AMD linked plan invocation arity changed".into(),
                    });
                }
                let dst = unsafe { std::slice::from_raw_parts_mut(slot.host, slot.record_size) };
                crate::hcq::ClikeKernargLayout::from_abi(&slot.abi).pack(dst, buffers, vals)?;
            }

            let mut runtime = RuntimePatchValues { buffers: vec![0; calls.len() * 2], vars: vec![0; calls.len() * 6] };
            for (operation, call) in calls.iter().enumerate() {
                match call {
                    PlanCall::Program { global_size, local_size, .. } => {
                        let g = global_size.unwrap_or([1, 1, 1]);
                        let l = local_size.unwrap_or([1, 1, 1]);
                        for axis in 0..3 {
                            runtime.vars[operation * 6 + axis] = l[axis] as i64;
                            runtime.vars[operation * 6 + 3 + axis] =
                                if self.pm4 { g[axis] } else { g[axis] * l[axis] } as i64;
                        }
                    }
                    PlanCall::Copy { dst, src, .. } => {
                        runtime.buffers[operation * 2] = *dst;
                        runtime.buffers[operation * 2 + 1] = *src;
                    }
                    PlanCall::Unsupported => unreachable!(),
                }
            }

            let mut system = SystemPatchValues::default();
            let prior = self.timelines.device();
            system.0.insert(SystemField::TimelineSignal(0), prior.signal_address);
            system.0.insert(SystemField::TimelineValue(0), prior.value);
            for submission in &self.submissions {
                if let Some(slot) = submission.operation_slot {
                    let p = self.timelines.reserve_queue(submission.queue);
                    system.0.insert(SystemField::TimelineSignal(slot), p.signal_address);
                    system.0.insert(SystemField::TimelineValue(slot), p.value);
                }
            }
            let final_point = self.timelines.finalize_device();
            system.0.insert(SystemField::TimelineSignal(self.operation_count as u32 + 1), final_point.signal_address);
            system.0.insert(SystemField::TimelineValue(self.operation_count as u32 + 1), final_point.value);
            system.0.insert(SystemField::ScratchAddress, lane.scratch_gpu_va());
            system.0.insert(SystemField::ScratchTmpring, lane.tmpring_size() as u64);
            for address in self.timelines.take_resets() {
                self._signals.iter().find(|signal| signal.value_addr() == address).unwrap().reset(0);
            }

            // Complete every fallible host patch before registering a finalizer
            // or publishing the first doorbell.
            for submission in self.submissions.iter_mut() {
                if let Some(control) = &mut submission.control {
                    control.linked.patch(&mut control.replay, &runtime, &system)?;
                }
                submission.linked.patch(&mut submission.replay, &runtime, &system)?;
                if matches!(submission.queue, QueueKind::Copy(_)) && owner.core().copy_queue().is_none() {
                    return Err(Error::Runtime { message: "AMD linked plan COPY requires SDMA queue".into() });
                }
            }

            let compute_lengths = self
                .submissions
                .iter()
                .filter(|submission| matches!(submission.queue, QueueKind::Compute(_)))
                .map(|submission| submission.replay.bytes().len())
                .collect::<Vec<_>>();
            let copy_lengths = self
                .submissions
                .iter()
                .filter(|submission| matches!(submission.queue, QueueKind::Copy(_)))
                .map(|submission| submission.replay.bytes().len())
                .collect::<Vec<_>>();
            // Wait each ring's headroom with no guard held, then take both
            // guards back-to-back: holding the compute guard while polling the
            // process-shared SDMA ring stalled every host staging copy for up
            // to the full timeout, and doubled this plan's worst case.
            let copy_queue = owner.core().copy_queue();
            if !compute_lengths.is_empty() {
                lane.queue().wait_publication_headroom(&compute_lengths)?;
            }
            if !copy_lengths.is_empty() {
                copy_queue.unwrap().wait_publication_headroom(&copy_lengths)?;
            }
            let mut compute_publication = (!compute_lengths.is_empty())
                .then(|| lane.queue().prepare_linked_publication(&compute_lengths))
                .transpose()?;
            let mut copy_publication = (!copy_lengths.is_empty())
                .then(|| copy_queue.unwrap().prepare_linked_publication(&copy_lengths))
                .transpose()?;
            owner.core().publication_checkpoint(crate::amd::iface::PublicationStage::AfterReservation)?;
            owner.core().publication_checkpoint(crate::amd::iface::PublicationStage::BeforeDoorbell)?;

            // Device-wide drains must see this final point before the first
            // doorbell. A partial publication poisons the core; a host-only
            // failure rolls timeline reservations back below.
            let newest = if self.pm4 || !self.submissions.is_empty() {
                let signal = self
                    ._signals
                    .iter()
                    .find(|signal| signal.value_addr() == final_point.signal_address)
                    .unwrap()
                    .clone();
                let progress =
                    self._signals.iter().filter(|candidate| !Arc::ptr_eq(candidate, &signal)).cloned().collect();
                let finalizer = SubmissionFinalizer::prepared_timeline(signal, final_point.value, progress);
                lane.register_inflight(Arc::clone(&finalizer));
                // The plan's SDMA work rides a plan-local timeline, not the
                // copy queue's own, so the queue must retain this finalizer or
                // it could tear its ring down under live traffic.
                if !copy_lengths.is_empty()
                    && let Some(copy) = copy_queue
                {
                    copy.register_inflight(Arc::clone(&finalizer));
                }
                Some(finalizer)
            } else {
                None
            };
            let publication = newest
                .as_ref()
                .map(|finalizer| PreparedPublication::new(Arc::clone(finalizer), Arc::clone(owner.core())));
            for submission in self.submissions.iter_mut() {
                if !self.pm4
                    && matches!(submission.queue, QueueKind::Compute(_))
                    && let Some(control) = &submission.control
                {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            control.replay.bytes().as_ptr(),
                            control.host,
                            control.replay.bytes().len(),
                        );
                    }
                    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                }
                match submission.queue {
                    QueueKind::Compute(_) => compute_publication.as_mut().unwrap().publish(&submission.replay),
                    QueueKind::Copy(_) => copy_publication.as_mut().unwrap().publish(&submission.replay),
                }
                published = true;
                owner.core().publication_checkpoint(crate::amd::iface::PublicationStage::AfterDoorbell)?;
            }
            if let Some(publication) = publication {
                publication.publish();
            }
            if let Some(finalizer) = newest {
                owner.set_newest(finalizer);
            }
            Ok(())
        })();
        if result.is_err() && !published {
            self.timelines = timelines_before;
        }
        result.map_err(|error| ReplayFailure { error, published })
    }
}
