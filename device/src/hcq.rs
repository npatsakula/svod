//! Host command queue primitives shared by hardware backends.
//!
//! PROGRAM remains the compiler/runtime boundary. At execution time its globals
//! have already been selected in ABI order; this module resolves those buffers
//! to device addresses, packs C-like kernargs, and describes queue commands in
//! their exact submission order. Backends lower the commands to native packets.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::error::{Error, Result};
use svod_dtype::DeviceSpec;

/// Queue selected for a submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueKind {
    Compute(u32),
    Copy(u32),
}

/// A queue belongs to one concrete device. Keeping device identity above
/// `Submission` preserves the native packet ABI while allowing the neutral
/// scheduler to distinguish equal queue numbers on different devices.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceQueue {
    pub device: DeviceSpec,
    pub queue: QueueKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TopologyResource {
    pub id: u64,
    pub owner: DeviceSpec,
    pub start: usize,
    pub end: usize,
}

impl TopologyResource {
    fn overlaps(&self, other: &Self) -> bool {
        self.id == other.id && self.owner == other.owner && self.start < other.end && other.start < self.end
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyOperationKind {
    Execute,
    Copy { src: TopologyResource, dst: TopologyResource, bytes: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyOperation {
    pub operation: usize,
    pub lane: DeviceQueue,
    pub reads: Vec<TopologyResource>,
    pub writes: Vec<TopologyResource>,
    pub kind: TopologyOperationKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyLeg {
    Direct,
    ToHost,
    FromHost,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyCommand {
    pub operation: usize,
    pub copy_leg: Option<CopyLeg>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneWait {
    pub lane: DeviceQueue,
    pub value: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneSubmission {
    pub lane: DeviceQueue,
    pub waits: Vec<LaneWait>,
    pub commands: Vec<TopologyCommand>,
    pub signal_value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueMergeLimits {
    pub max_submissions: usize,
    pub max_commands: usize,
}

impl QueueMergeLimits {
    pub const UNLIMITED: Self = Self { max_submissions: 0, max_commands: 0 };
    pub const NO_MERGE: Self = Self { max_submissions: 1, max_commands: 0 };
}

/// Partition neutral operations by concrete device/queue lane, insert host
/// staging when either endpoint is inaccessible, derive cross-lane hazards,
/// and merge only while both limits are satisfied. `can_access(executor,
/// owner)` must report real target accessibility; callers must not infer peer
/// access merely from matching backend families.
pub fn schedule_device_lanes(
    operations: &[TopologyOperation],
    limits: QueueMergeLimits,
    mut can_access: impl FnMut(&DeviceSpec, &DeviceSpec) -> bool,
) -> Vec<LaneSubmission> {
    #[derive(Clone)]
    struct Expanded {
        operation: usize,
        lane: DeviceQueue,
        reads: Vec<TopologyResource>,
        writes: Vec<TopologyResource>,
        leg: Option<CopyLeg>,
    }

    let host = DeviceSpec::Cpu;
    let mut expanded = Vec::new();
    for op in operations {
        match &op.kind {
            TopologyOperationKind::Copy { src, dst, bytes }
                if !(can_access(&op.lane.device, &src.owner) && can_access(&op.lane.device, &dst.owner)) =>
            {
                // A unique host resource prevents unrelated staged copies from
                // aliasing while still creating the required dependency edge.
                let stage = TopologyResource {
                    id: u64::MAX - op.operation as u64,
                    owner: host.clone(),
                    start: 0,
                    end: (*bytes).max(1),
                };
                expanded.push(Expanded {
                    operation: op.operation,
                    lane: DeviceQueue { device: src.owner.clone(), queue: QueueKind::Copy(0) },
                    reads: vec![src.clone()],
                    writes: vec![stage.clone()],
                    leg: Some(CopyLeg::ToHost),
                });
                expanded.push(Expanded {
                    operation: op.operation,
                    lane: DeviceQueue { device: dst.owner.clone(), queue: QueueKind::Copy(0) },
                    reads: vec![stage],
                    writes: vec![dst.clone()],
                    leg: Some(CopyLeg::FromHost),
                });
            }
            TopologyOperationKind::Copy { src, dst, .. } => expanded.push(Expanded {
                operation: op.operation,
                lane: op.lane.clone(),
                reads: vec![src.clone()],
                writes: vec![dst.clone()],
                leg: Some(CopyLeg::Direct),
            }),
            TopologyOperationKind::Execute => expanded.push(Expanded {
                operation: op.operation,
                lane: op.lane.clone(),
                reads: op.reads.clone(),
                writes: op.writes.clone(),
                leg: None,
            }),
        }
    }

    let hazard = |a: &Expanded, b: &Expanded| {
        a.writes.iter().any(|x| b.reads.iter().any(|y| x.overlaps(y)))
            || a.reads.iter().any(|x| b.writes.iter().any(|y| x.overlaps(y)))
            || a.writes.iter().any(|x| b.writes.iter().any(|y| x.overlaps(y)))
    };
    let mut sequence: HashMap<DeviceQueue, u64> = HashMap::new();
    let mut prior: Vec<(Expanded, u64)> = Vec::new();
    let mut raw = Vec::with_capacity(expanded.len());
    for op in expanded {
        let value = {
            let next = sequence.entry(op.lane.clone()).or_default();
            *next += 1;
            *next
        };
        let mut latest: HashMap<DeviceQueue, u64> = HashMap::new();
        for (producer, producer_value) in &prior {
            if producer.lane != op.lane && hazard(producer, &op) {
                latest
                    .entry(producer.lane.clone())
                    .and_modify(|v| *v = (*v).max(*producer_value))
                    .or_insert(*producer_value);
            }
        }
        let mut waits = latest.into_iter().map(|(lane, value)| LaneWait { lane, value }).collect::<Vec<_>>();
        waits.sort_by_key(|w| {
            (
                w.lane.device.canonicalize(),
                match w.lane.queue {
                    QueueKind::Compute(n) => (0, n),
                    QueueKind::Copy(n) => (1, n),
                },
            )
        });
        raw.push(LaneSubmission {
            lane: op.lane.clone(),
            waits,
            commands: vec![TopologyCommand { operation: op.operation, copy_leg: op.leg }],
            signal_value: value,
        });
        prior.push((op, value));
    }

    let mut merged: Vec<LaneSubmission> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    for submission in raw {
        let append = merged.last().is_some_and(|last| {
            last.lane == submission.lane
                && (limits.max_submissions == 0 || counts.last().copied().unwrap_or(0) < limits.max_submissions)
                && (limits.max_commands == 0 || last.commands.len() + submission.commands.len() <= limits.max_commands)
        });
        if append {
            let last = merged.last_mut().unwrap();
            for wait in submission.waits {
                if !last.waits.iter().any(|w| w.lane == wait.lane && w.value >= wait.value) {
                    last.waits.retain(|w| w.lane != wait.lane);
                    last.waits.push(wait);
                }
            }
            last.commands.extend(submission.commands);
            last.signal_value = submission.signal_value;
            *counts.last_mut().unwrap() += 1;
        } else {
            merged.push(submission);
            counts.push(1);
        }
    }
    // A merged submission publishes only its final signal. Rewrite every wait
    // to the first published boundary containing the logical producer value.
    // This must happen after merging: retaining an interior value describes a
    // timeline point which no executable submission actually stores.
    for index in 0..merged.len() {
        let mut waits: HashMap<DeviceQueue, u64> = HashMap::new();
        for wait in std::mem::take(&mut merged[index].waits) {
            let value = merged
                .iter()
                .find(|producer| producer.lane == wait.lane && producer.signal_value >= wait.value)
                .map(|producer| producer.signal_value)
                .unwrap_or(wait.value);
            waits.entry(wait.lane).and_modify(|current| *current = (*current).max(value)).or_insert(value);
        }
        merged[index].waits = waits.into_iter().map(|(lane, value)| LaneWait { lane, value }).collect();
        merged[index].waits.sort_by_key(|wait| {
            (
                wait.lane.device.canonicalize(),
                match wait.lane.queue {
                    QueueKind::Compute(n) => (0, n),
                    QueueKind::Copy(n) => (1, n),
                },
            )
        });
    }
    merged
}

/// Hardware wait fields are 32-bit on the current HCQ backends. Switch to the
/// shadow signal at half range, matching Tinygrad's timeline rollover headroom.
pub const TIMELINE_ROLLOVER: u64 = 1 << 31;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelinePoint {
    pub signal_address: u64,
    pub value: u64,
}

/// One HCQ timeline with a shadow signal for rollover. The signal address and
/// monotonic value are reserved together, preventing callers from advancing a
/// counter without also carrying the signal generation it belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochTimeline {
    signals: [u64; 2],
    active: usize,
    next: u64,
    pending_reset: Option<u64>,
}

impl EpochTimeline {
    pub fn new(signals: [u64; 2]) -> Self {
        Self { signals, active: 0, next: 1, pending_reset: None }
    }

    pub fn with_next(signals: [u64; 2], next: u64) -> Self {
        Self { signals, active: 0, next, pending_reset: None }
    }

    pub fn current(&self) -> TimelinePoint {
        TimelinePoint { signal_address: self.signals[self.active], value: self.next.saturating_sub(1) }
    }

    pub fn reserve(&mut self) -> TimelinePoint {
        if self.next > TIMELINE_ROLLOVER {
            self.active ^= 1;
            self.next = 1;
            self.pending_reset = Some(self.signals[self.active]);
        }
        let point = TimelinePoint { signal_address: self.signals[self.active], value: self.next };
        self.next += 1;
        point
    }

    pub fn take_reset(&mut self) -> Option<u64> {
        self.pending_reset.take()
    }
}

/// Submission-owned device/compute/copy epochs. Finalizers reserve the device
/// point; operation submissions reserve queue points and never manipulate raw
/// counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmissionTimelines {
    device: EpochTimeline,
    compute: EpochTimeline,
    copy: EpochTimeline,
}

impl SubmissionTimelines {
    pub fn new(device: [u64; 2], compute: [u64; 2], copy: [u64; 2]) -> Self {
        Self {
            device: EpochTimeline::new(device),
            compute: EpochTimeline::new(compute),
            copy: EpochTimeline::new(copy),
        }
    }

    pub fn device(&self) -> TimelinePoint {
        self.device.current()
    }

    pub fn reserve_queue(&mut self, queue: QueueKind) -> TimelinePoint {
        match queue {
            QueueKind::Compute(_) => self.compute.reserve(),
            QueueKind::Copy(_) => self.copy.reserve(),
        }
    }

    pub fn finalize_device(&mut self) -> TimelinePoint {
        self.device.reserve()
    }

    /// Signal slots which changed generation while constructing the current
    /// epoch. The submission executor resets these host-visible locations only
    /// after the preceding epoch has retired and before publishing this epoch.
    pub fn take_resets(&mut self) -> Vec<u64> {
        [&mut self.device, &mut self.compute, &mut self.copy]
            .into_iter()
            .filter_map(EpochTimeline::take_reset)
            .collect()
    }
}

/// Backend-neutral compute dispatch after host-side GETADDR resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeDispatch {
    pub workgroup_size: [u32; 3],
    pub grid_size: [u32; 3],
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    pub kernel_object: u64,
    pub kernarg_address: u64,
    pub completion_signal: u64,
    pub barrier: bool,
    /// Raw-PM4 launch state. Backends which dispatch through an architected
    /// packet format (for example AMD AQL) ignore this metadata.
    pub amd_pm4: Option<AmdPm4Dispatch>,
}

/// AMD state which cannot be recovered from an AQL kernel descriptor GPU VA.
/// Keeping it attached to the neutral compute command lets the AMD backend
/// choose PM4 or AQL at queue-lowering time without re-entering `Program`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdPm4Dispatch {
    pub rsrc: [u32; 3],
    pub program_address: u64,
    pub enable_private_segment_sgpr: bool,
    pub workgroup_count: [u32; 3],
    pub wave32: bool,
    pub target_major: u32,
}

/// Who reads, through a [`Command::Store`]'s signal, the memory the queue
/// wrote before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreScope {
    /// Only later commands on the same queue, which already run behind it: a
    /// progress mark that no other lane or host waits on for data.
    Queue,
    /// Whoever waits on the signal: the host, another queue or engine.
    System,
}

/// One command in a hardware queue submission. Vector order is semantic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Wait {
        signal_address: u64,
        value: u64,
    },
    MemoryBarrier,
    /// Execute a prepared runtime operation. This remains backend-neutral: the
    /// execution plan owns the program and buffers while the submission owns
    /// ordering and dependency semantics.
    Execute {
        operation: usize,
    },
    Compute(ComputeDispatch),
    Copy {
        dst: u64,
        src: u64,
        bytes: usize,
    },
    Timestamp {
        dst: u64,
    },
    Store {
        dst: u64,
        value: u64,
        scope: StoreScope,
    },
}

/// A command buffer submitted atomically to one hardware queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submission {
    pub queue: QueueKind,
    pub commands: Vec<Command>,
    patches: Vec<CommandPatch>,
    profile: bool,
}

/// Error produced while a host submission executor is consuming an ordered
/// submission. Queue errors and prepared-operation errors remain distinct so a
/// runtime can preserve its own typed errors instead of flattening them.
#[derive(Debug)]
pub enum SubmissionExecutionError<E> {
    Queue(Error),
    Execute(E),
}

impl Submission {
    pub fn new(queue: QueueKind) -> Self {
        Self { queue, commands: Vec::new(), patches: Vec::new(), profile: false }
    }

    pub fn push(&mut self, command: Command) -> &mut Self {
        self.commands.push(command);
        self
    }

    pub fn clear(&mut self) {
        self.commands.clear();
        self.patches.clear();
        self.profile = false;
    }

    pub fn wait_for(&mut self, point: TimelinePoint) -> &mut Self {
        self.push(Command::Wait { signal_address: point.signal_address, value: point.value })
    }

    pub fn signal(&mut self, point: TimelinePoint) -> &mut Self {
        self.push(Command::Store { dst: point.signal_address, value: point.value, scope: StoreScope::System })
    }

    /// Insert a command while preserving semantic patch bindings.
    pub fn insert(&mut self, index: usize, command: Command) {
        self.commands.insert(index, command);
        for patch in &mut self.patches {
            if patch.command >= index {
                patch.command += 1;
            }
        }
    }

    /// Bind a semantic command field to a link- or replay-time value. The
    /// backend lowerer, which knows the native packet layout, turns this into
    /// byte patch sites. Callers never provide native byte offsets.
    pub fn bind(&mut self, command: usize, field: CommandField, source: PatchSource) -> Result<&mut Self> {
        if command >= self.commands.len() {
            return Err(Error::Runtime {
                message: format!(
                    "HCQ patch references command {command}, submission has {} commands",
                    self.commands.len()
                ),
            });
        }
        if self.patches.iter().any(|p| p.command == command && p.field == field) {
            return Err(Error::Runtime { message: format!("HCQ command {command} field {field:?} is bound twice") });
        }
        self.patches.push(CommandPatch { command, field, source });
        self.patches.sort_unstable();
        Ok(self)
    }

    pub fn patches(&self) -> &[CommandPatch] {
        &self.patches
    }

    /// Request queue-timeline timestamps for executable commands in this
    /// submission. Allocation, command insertion, and release are deliberately
    /// deferred to the backend queue finalizer.
    pub fn request_profile(&mut self) -> &mut Self {
        self.profile = true;
        self
    }

    pub fn profile_requested(&self) -> bool {
        self.profile
    }
}

/// Host executor for CPU and backend-neutral HCQ submissions. Signal addresses
/// are opaque keys, matching hardware queue timelines; copy addresses are host
/// pointers supplied by the runtime. A failed command stops the submission, so
/// trailing completion stores are never published after failed work.
#[derive(Debug)]
pub struct CpuQueueExecutor {
    signals: std::collections::HashMap<u64, u64>,
    clock_ns: u64,
    clock_step_ns: u64,
}

impl Default for CpuQueueExecutor {
    fn default() -> Self {
        Self { signals: std::collections::HashMap::new(), clock_ns: 0, clock_step_ns: 1 }
    }
}

impl CpuQueueExecutor {
    pub fn with_clock(start_ns: u64, step_ns: u64) -> Self {
        Self { clock_ns: start_ns, clock_step_ns: step_ns, ..Self::default() }
    }

    pub fn set_signal(&mut self, address: u64, value: u64) {
        self.signals.insert(address, value);
    }

    pub fn signal_value(&self, address: u64) -> Option<u64> {
        self.signals.get(&address).copied()
    }

    /// # Safety
    ///
    /// Every non-empty [`Command::Copy`] must contain valid host addresses for
    /// `bytes` readable/writable bytes. The referenced storage must remain live
    /// and obey the submission's hazard ordering until this method returns.
    pub unsafe fn submit<E>(
        &mut self,
        submission: &Submission,
        mut execute: impl FnMut(usize) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), SubmissionExecutionError<E>> {
        for command in &submission.commands {
            match command {
                Command::Wait { signal_address, value } => {
                    let current = self.signals.get(signal_address).copied().unwrap_or(0);
                    if current < *value {
                        return Err(SubmissionExecutionError::Queue(Error::Runtime {
                            message: format!(
                                "CPU HCQ unsatisfied wait at {signal_address:#x}: current {current}, target {value}"
                            ),
                        }));
                    }
                }
                Command::MemoryBarrier => std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst),
                Command::Execute { operation } => {
                    execute(*operation).map_err(SubmissionExecutionError::Execute)?;
                }
                Command::Compute(_) => {
                    return Err(SubmissionExecutionError::Queue(Error::Runtime {
                        message: "CPU HCQ cannot execute a hardware Compute dispatch; use Execute".into(),
                    }));
                }
                Command::Copy { dst, src, bytes } => {
                    if *bytes != 0 && (*dst == 0 || *src == 0) {
                        return Err(SubmissionExecutionError::Queue(Error::Runtime {
                            message: "CPU HCQ copy has a null host address".into(),
                        }));
                    }
                    // SAFETY: HCQ call construction owns address resolution and
                    // buffer hazard ordering. `ptr::copy` also permits overlap.
                    if *bytes != 0 {
                        unsafe { std::ptr::copy(*src as *const u8, *dst as *mut u8, *bytes) };
                    }
                }
                Command::Timestamp { dst } => {
                    self.signals.insert(*dst, self.clock_ns);
                    self.clock_ns = self.clock_ns.wrapping_add(self.clock_step_ns);
                }
                Command::Store { dst, value, .. } => {
                    self.signals.insert(*dst, *value);
                }
            }
        }
        Ok(())
    }
}

/// Semantic fields which a packet lowerer may expose as patchable. A field is
/// deliberately command-level: native offsets are produced only while lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CommandField {
    WaitAddress,
    WaitValue,
    ComputeKernelObject,
    ComputeKernargAddress,
    ComputeCompletionSignal,
    ComputeProgramAddress,
    ComputeScratchAddress,
    ComputeScratchTmpring,
    ComputeWorkgroup(u8),
    ComputeGrid(u8),
    CopyDst,
    CopySrc,
    TimestampDst,
    StoreDst,
    StoreValue,
}

/// Backend-neutral system values owned by queue/runtime state rather than a
/// graph invocation's input buffers and scalar variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SystemField {
    TimelineSignal(u32),
    TimelineValue(u32),
    Timestamp(u32),
    ScratchAddress,
    ScratchTmpring,
    KernargBase,
}

/// Value classes mirrored from HCQ2's link/runtime/system partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PatchSource {
    /// Address of immutable linked storage such as a program or command buffer.
    LinkAddress(usize),
    /// GETADDR of a positional invocation buffer.
    RuntimeBuffer(usize),
    /// Positional ProgramInfo variable.
    RuntimeVar(usize),
    /// Queue timeline, timestamp, scratch, or other runtime-owned field.
    System(SystemField),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CommandPatch {
    pub command: usize,
    pub field: CommandField,
    pub source: PatchSource,
}

/// Native scalar encoding recorded by a backend lowerer at packet emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PatchEncoding {
    U16,
    U32,
    U64,
    Low32,
    High32,
    High32Or(u32),
    Low32ShiftRight(u8),
    High32ShiftRight(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PatchSite {
    pub byte_offset: usize,
    pub encoding: PatchEncoding,
    pub source: PatchSource,
    /// Packet-derived adjustment, used for each chunk of a lowered copy.
    pub addend: u64,
}

/// Deterministic patch partitions. Each vector is sorted by native byte offset.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct PatchTable {
    pub link: Vec<PatchSite>,
    pub runtime: Vec<PatchSite>,
    pub system: Vec<PatchSite>,
}

impl PatchTable {
    pub fn from_sites(mut sites: Vec<PatchSite>) -> Self {
        sites.sort_unstable();
        let mut out = Self::default();
        for site in sites {
            match site.source {
                PatchSource::LinkAddress(_) => out.link.push(site),
                PatchSource::RuntimeBuffer(_) | PatchSource::RuntimeVar(_) => out.runtime.push(site),
                PatchSource::System(_) => out.system.push(site),
            }
        }
        out
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct LinkPatchValues(pub Vec<u64>);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimePatchValues {
    pub buffers: Vec<u64>,
    pub vars: Vec<i64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SystemPatchValues(pub BTreeMap<SystemField, u64>);

/// Packet bytes plus lowering-originated metadata, before immutable addresses
/// have been linked.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoweredCommandBuffer {
    pub bytes: Vec<u8>,
    pub patches: PatchTable,
}

impl LoweredCommandBuffer {
    pub fn link(&self, values: &LinkPatchValues) -> Result<LinkedCommandBuffer> {
        let mut bytes = self.bytes.clone();
        apply_sites(&mut bytes, &self.patches.link, |source| match source {
            PatchSource::LinkAddress(slot) => values.0.get(slot).copied(),
            _ => None,
        })?;
        Ok(LinkedCommandBuffer {
            static_bytes: bytes.into(),
            runtime_patches: self.patches.runtime.clone(),
            system_patches: self.patches.system.clone(),
        })
    }
}

/// Immutable linked packet stream shared by all replays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkedCommandBuffer {
    static_bytes: Arc<[u8]>,
    runtime_patches: Vec<PatchSite>,
    system_patches: Vec<PatchSite>,
}

impl LinkedCommandBuffer {
    pub fn static_bytes(&self) -> &[u8] {
        &self.static_bytes
    }

    pub fn replay_buffer(&self) -> ReplayCommandBuffer {
        ReplayCommandBuffer { bytes: self.static_bytes.to_vec() }
    }

    pub fn patch(
        &self,
        replay: &mut ReplayCommandBuffer,
        runtime: &RuntimePatchValues,
        system: &SystemPatchValues,
    ) -> Result<()> {
        if replay.bytes.len() != self.static_bytes.len() {
            return Err(Error::Runtime {
                message: "HCQ replay buffer does not belong to linked command buffer".into(),
            });
        }
        apply_sites(&mut replay.bytes, &self.runtime_patches, |source| match source {
            PatchSource::RuntimeBuffer(slot) => runtime.buffers.get(slot).copied(),
            // ProgramInfo vars use the canonical C-like i32 ABI, including
            // two's-complement encoding for negative values.
            PatchSource::RuntimeVar(slot) => runtime.vars.get(slot).map(|v| *v as i32 as u32 as u64),
            _ => None,
        })?;
        apply_sites(&mut replay.bytes, &self.system_patches, |source| match source {
            PatchSource::System(field) => system.0.get(&field).copied(),
            _ => None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCommandBuffer {
    bytes: Vec<u8>,
}

/// Linked semantic commands for host backends. This is the CPU/null analogue
/// of [`LinkedCommandBuffer`]: command structure is retained once, while only
/// bound scalar fields are changed for each replay.
#[derive(Debug, Clone)]
pub struct SemanticLinkedSubmission {
    lane: DeviceQueue,
    submission: Arc<Submission>,
}

#[derive(Debug, Clone)]
pub struct SemanticReplaySubmission {
    lane: DeviceQueue,
    submission: Submission,
}

impl SemanticLinkedSubmission {
    pub fn new(submission: Submission) -> Self {
        let lane = DeviceQueue { device: DeviceSpec::Cpu, queue: submission.queue };
        Self { lane, submission: Arc::new(submission) }
    }

    pub fn new_for_lane(lane: DeviceQueue, submission: Submission) -> Result<Self> {
        if lane.queue != submission.queue {
            return Err(Error::Runtime {
                message: format!(
                    "HCQ semantic lane {:?} disagrees with submission queue {:?}",
                    lane.queue, submission.queue
                ),
            });
        }
        Ok(Self { lane, submission: Arc::new(submission) })
    }

    pub fn lane(&self) -> &DeviceQueue {
        &self.lane
    }

    pub fn static_submission(&self) -> &Submission {
        &self.submission
    }

    pub fn replay_buffer(&self) -> SemanticReplaySubmission {
        SemanticReplaySubmission { lane: self.lane.clone(), submission: (*self.submission).clone() }
    }

    pub fn patch(
        &self,
        replay: &mut SemanticReplaySubmission,
        runtime: &RuntimePatchValues,
        system: &SystemPatchValues,
    ) -> Result<()> {
        if replay.submission.commands.len() != self.submission.commands.len()
            || replay.submission.queue != self.submission.queue
            || replay.lane != self.lane
        {
            return Err(Error::Runtime { message: "HCQ semantic replay does not belong to linked submission".into() });
        }
        for patch in self.submission.patches() {
            let value = match patch.source {
                PatchSource::RuntimeBuffer(slot) => runtime.buffers.get(slot).copied(),
                PatchSource::RuntimeVar(slot) => runtime.vars.get(slot).map(|&value| value as u64),
                PatchSource::System(field) => system.0.get(&field).copied(),
                PatchSource::LinkAddress(_) => continue,
            }
            .ok_or_else(|| Error::Runtime { message: format!("HCQ patch value missing for {:?}", patch.source) })?;
            patch_command_field(&mut replay.submission.commands[patch.command], patch.field, value)?;
        }
        Ok(())
    }
}

impl SemanticReplaySubmission {
    pub fn lane(&self) -> &DeviceQueue {
        &self.lane
    }

    pub fn submission(&self) -> &Submission {
        &self.submission
    }
}

/// Concrete mapping from a scheduler-local lane value to the signal point an
/// executor will publish. Logical values may skip after queue merging, while
/// concrete points remain dense on each device/queue timeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineBinding {
    pub lane: DeviceQueue,
    pub logical_value: u64,
    pub point: TimelinePoint,
}

/// Executable host/null form of the device-aware lane topology. The original
/// [`LaneSubmission`] commands remain the source of truth, including copy-leg
/// identity; queue commands are materialized only when submitted to a host
/// executor.
#[derive(Debug, Clone)]
pub struct SemanticLinkedPlan {
    lanes: Arc<[LaneSubmission]>,
    bindings: Arc<[TimelineBinding]>,
}

impl SemanticLinkedPlan {
    /// Reserve one concrete timeline per `DeviceQueue` and bind each published
    /// merged boundary. `timeline_signals` is called once for every concrete
    /// lane, including equal queue numbers belonging to different devices.
    pub fn from_lane_submissions(
        lanes: Vec<LaneSubmission>,
        mut timeline_signals: impl FnMut(&DeviceQueue) -> [u64; 2],
    ) -> Result<Self> {
        let mut timelines: HashMap<DeviceQueue, EpochTimeline> = HashMap::new();
        let mut bindings = Vec::with_capacity(lanes.len());
        for submission in &lanes {
            let timeline = timelines
                .entry(submission.lane.clone())
                .or_insert_with(|| EpochTimeline::new(timeline_signals(&submission.lane)));
            bindings.push(TimelineBinding {
                lane: submission.lane.clone(),
                logical_value: submission.signal_value,
                point: timeline.reserve(),
            });
        }

        for submission in &lanes {
            for wait in &submission.waits {
                if !bindings.iter().any(|binding| binding.lane == wait.lane && binding.logical_value == wait.value) {
                    return Err(Error::Runtime {
                        message: format!(
                            "HCQ lane {:?} waits for unpublished {:?} value {}",
                            submission.lane, wait.lane, wait.value
                        ),
                    });
                }
            }
        }
        Ok(Self { lanes: lanes.into(), bindings: bindings.into() })
    }

    pub fn lanes(&self) -> &[LaneSubmission] {
        &self.lanes
    }

    pub fn bindings(&self) -> &[TimelineBinding] {
        &self.bindings
    }

    pub fn staged_copy(&self) -> Option<usize> {
        self.lanes.iter().flat_map(|lane| &lane.commands).find_map(|command| {
            matches!(command.copy_leg, Some(CopyLeg::ToHost | CopyLeg::FromHost)).then_some(command.operation)
        })
    }

    /// Materialize the established single-device submission ABI from the
    /// authoritative lane topology. Native backends use this adapter to retain
    /// their packet shape; scheduling and dependency discovery stay above it.
    pub fn native_submissions(&self) -> Result<Vec<SemanticLinkedSubmission>> {
        use CommandField::{StoreDst, StoreValue, WaitAddress, WaitValue};

        if self.lanes.is_empty() {
            return Ok(Vec::new());
        }
        if self.lanes.iter().any(|lane| lane.commands.len() != 1) {
            return Err(Error::Runtime { message: "native HCQ lowering requires unmerged lane submissions".into() });
        }
        let device = self.lanes[0].lane.device.clone();
        if self.lanes.iter().any(|lane| lane.lane.device != device) {
            return Err(Error::Runtime { message: "native HCQ lowering requires one concrete device".into() });
        }

        let bind_point = |submission: &mut Submission, command: usize, slot: u32, store: bool| -> Result<()> {
            let (address, value) = if store { (StoreDst, StoreValue) } else { (WaitAddress, WaitValue) };
            submission.bind(command, address, PatchSource::System(SystemField::TimelineSignal(slot)))?.bind(
                command,
                value,
                PatchSource::System(SystemField::TimelineValue(slot)),
            )?;
            Ok(())
        };
        let producers = self
            .lanes
            .iter()
            .enumerate()
            .map(|(index, lane)| ((lane.lane.clone(), lane.signal_value), index))
            .collect::<HashMap<_, _>>();
        let producer_of = |wait: &LaneWait| {
            producers.get(&(wait.lane.clone(), wait.value)).copied().ok_or_else(|| Error::Runtime {
                message: format!("native HCQ wait has no producer: {:?} value {}", wait.lane, wait.value),
            })
        };
        let final_lane = DeviceQueue { device, queue: QueueKind::Compute(0) };
        let latest =
            self.lanes.iter().enumerate().map(|(index, lane)| (lane.lane.clone(), index)).collect::<HashMap<_, _>>();
        // A store publishes its lane's writes only to the lanes that wait on it,
        // the finalizer's among them; any other store just marks progress for
        // work its own queue already runs in order behind it.
        let mut published = HashSet::new();
        for wait in self.lanes.iter().flat_map(|lane| &lane.waits) {
            published.insert(producer_of(wait)?);
        }
        published.extend(latest.iter().filter(|(lane, _)| **lane != final_lane).map(|(_, &index)| index));
        let mut first_use = HashSet::new();
        let mut submissions = Vec::with_capacity(self.lanes.len() + 1);

        for (index, lane) in self.lanes.iter().enumerate() {
            let mut submission = Submission::new(lane.lane.queue);
            if first_use.insert(lane.lane.clone()) {
                submission.push(Command::MemoryBarrier);
                let command = submission.commands.len();
                submission.push(Command::Wait { signal_address: 0, value: 0 });
                bind_point(&mut submission, command, 0, false)?;
            }
            for wait in &lane.waits {
                let command = submission.commands.len();
                submission.push(Command::Wait { signal_address: 0, value: 0 });
                bind_point(&mut submission, command, producer_of(wait)? as u32 + 1, false)?;
            }
            // The producers' writes reached memory, but this queue's caches may
            // still hold what was there before: its queue-scoped stores leave
            // them in place.
            if !lane.waits.is_empty() {
                submission.push(Command::MemoryBarrier);
            }
            submission.push(Command::Execute { operation: lane.commands[0].operation });
            let command = submission.commands.len();
            let scope = if published.contains(&index) { StoreScope::System } else { StoreScope::Queue };
            submission.push(Command::Store { dst: 0, value: 0, scope });
            bind_point(&mut submission, command, index as u32 + 1, true)?;
            submissions.push(SemanticLinkedSubmission::new_for_lane(lane.lane.clone(), submission)?);
        }

        let mut finalizer = Submission::new(final_lane.queue);
        if !first_use.contains(&final_lane) {
            finalizer.push(Command::MemoryBarrier);
            let command = finalizer.commands.len();
            finalizer.push(Command::Wait { signal_address: 0, value: 0 });
            bind_point(&mut finalizer, command, 0, false)?;
        }
        let mut latest = latest.into_iter().collect::<Vec<_>>();
        latest.sort_by_key(|(lane, _)| match lane.queue {
            QueueKind::Compute(number) => (0, number),
            QueueKind::Copy(number) => (1, number),
        });
        for (lane, producer) in latest {
            if lane != final_lane {
                let command = finalizer.commands.len();
                finalizer.push(Command::Wait { signal_address: 0, value: 0 });
                bind_point(&mut finalizer, command, producer as u32 + 1, false)?;
            }
        }
        let command = finalizer.commands.len();
        finalizer.push(Command::Store { dst: 0, value: 0, scope: StoreScope::System });
        bind_point(&mut finalizer, command, self.lanes.len() as u32 + 1, true)?;
        submissions.push(SemanticLinkedSubmission::new_for_lane(final_lane, finalizer)?);
        Ok(submissions)
    }

    pub fn submission(&self, index: usize) -> Option<Submission> {
        let lane = self.lanes.get(index)?;
        let completion = self.binding(&lane.lane, lane.signal_value)?;
        let mut submission = Submission::new(lane.lane.queue);
        for wait in &lane.waits {
            submission.wait_for(self.binding(&wait.lane, wait.value)?);
        }
        for command in &lane.commands {
            submission.push(Command::Execute { operation: command.operation });
        }
        submission.signal(completion);
        Some(submission)
    }

    pub fn execute_null<E>(
        &self,
        executor: &mut NullHcq,
        mut execute: impl FnMut(&DeviceQueue, &TopologyCommand) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), SubmissionExecutionError<E>> {
        for (index, lane) in self.lanes.iter().enumerate() {
            let submission = self.submission(index).expect("validated semantic linked plan");
            let mut command_index = 0;
            executor.submit_with(&submission, |operation| {
                let command = &lane.commands[command_index];
                command_index += 1;
                debug_assert_eq!(operation, command.operation);
                execute(&lane.lane, command)
            })?;
        }
        Ok(())
    }

    /// # Safety
    ///
    /// The callback must uphold the memory and hazard requirements of each
    /// unresolved operation. Materialized plan commands themselves contain no
    /// raw host copy addresses.
    pub unsafe fn execute_cpu<E>(
        &self,
        executor: &mut CpuQueueExecutor,
        mut execute: impl FnMut(&DeviceQueue, &TopologyCommand) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), SubmissionExecutionError<E>> {
        for (index, lane) in self.lanes.iter().enumerate() {
            let submission = self.submission(index).expect("validated semantic linked plan");
            let mut command_index = 0;
            // SAFETY: generated commands contain only waits, Execute callbacks,
            // and stores. The caller owns safety of the unresolved operation.
            unsafe {
                executor.submit(&submission, |operation| {
                    let command = &lane.commands[command_index];
                    command_index += 1;
                    debug_assert_eq!(operation, command.operation);
                    execute(&lane.lane, command)
                })?;
            }
        }
        Ok(())
    }

    fn binding(&self, lane: &DeviceQueue, logical_value: u64) -> Option<TimelinePoint> {
        self.bindings
            .iter()
            .find(|binding| &binding.lane == lane && binding.logical_value == logical_value)
            .map(|binding| binding.point)
    }
}

fn patch_command_field(command: &mut Command, field: CommandField, value: u64) -> Result<()> {
    let valid = match (command, field) {
        (Command::Wait { signal_address, .. }, CommandField::WaitAddress) => {
            *signal_address = value;
            true
        }
        (Command::Wait { value: dst, .. }, CommandField::WaitValue) => {
            *dst = value;
            true
        }
        (Command::Copy { dst, .. }, CommandField::CopyDst) => {
            *dst = value;
            true
        }
        (Command::Copy { src, .. }, CommandField::CopySrc) => {
            *src = value;
            true
        }
        (Command::Timestamp { dst }, CommandField::TimestampDst) => {
            *dst = value;
            true
        }
        (Command::Store { dst, .. }, CommandField::StoreDst) => {
            *dst = value;
            true
        }
        (Command::Store { value: dst, .. }, CommandField::StoreValue) => {
            *dst = value;
            true
        }
        (Command::Compute(dispatch), CommandField::ComputeKernelObject) => {
            dispatch.kernel_object = value;
            true
        }
        (Command::Compute(dispatch), CommandField::ComputeKernargAddress) => {
            dispatch.kernarg_address = value;
            true
        }
        (Command::Compute(dispatch), CommandField::ComputeCompletionSignal) => {
            dispatch.completion_signal = value;
            true
        }
        (Command::Compute(dispatch), CommandField::ComputeProgramAddress) => {
            if let Some(pm4) = &mut dispatch.amd_pm4 {
                pm4.program_address = value;
                true
            } else {
                false
            }
        }
        (Command::Compute(dispatch), CommandField::ComputeWorkgroup(axis)) if axis < 3 => {
            dispatch.workgroup_size[axis as usize] = value as u32;
            true
        }
        (Command::Compute(dispatch), CommandField::ComputeGrid(axis)) if axis < 3 => {
            dispatch.grid_size[axis as usize] = value as u32;
            true
        }
        _ => false,
    };
    if !valid {
        return Err(Error::Runtime { message: format!("HCQ field {field:?} is invalid for semantic command") });
    }
    Ok(())
}

impl ReplayCommandBuffer {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Link cache keyed by packet bytes, exact lowering metadata, and immutable
/// address values. Dynamic invocation values are intentionally not part of it.
#[derive(Debug, Default)]
pub struct CommandBufferCache {
    entries:
        std::collections::HashMap<(u64, DeviceSpec, LoweredCommandBuffer, LinkPatchValues), Arc<LinkedCommandBuffer>>,
}

impl CommandBufferCache {
    /// Single-shot link with a placeholder `(0, DeviceSpec::Cpu)` context.
    ///
    /// Only safe for a cache that is not shared across devices or lanes —
    /// production AMD paths link through `PoolQueue::link`, which supplies the
    /// real lane and device.
    pub fn link(
        &mut self,
        lowered: &LoweredCommandBuffer,
        values: &LinkPatchValues,
    ) -> Result<Arc<LinkedCommandBuffer>> {
        self.link_for_context(0, &DeviceSpec::Cpu, lowered, values)
    }

    /// Cache linked storage only within the context/device whose virtual
    /// addresses it contains. Equal bytes on another device are not reusable.
    pub fn link_for_context(
        &mut self,
        context: u64,
        device: &DeviceSpec,
        lowered: &LoweredCommandBuffer,
        values: &LinkPatchValues,
    ) -> Result<Arc<LinkedCommandBuffer>> {
        let key = (context, device.clone(), lowered.clone(), values.clone());
        if let Some(linked) = self.entries.get(&key) {
            return Ok(Arc::clone(linked));
        }
        let linked = Arc::new(lowered.link(values)?);
        self.entries.insert(key, Arc::clone(&linked));
        Ok(linked)
    }
}

fn apply_sites(
    bytes: &mut [u8],
    sites: &[PatchSite],
    mut resolve: impl FnMut(PatchSource) -> Option<u64>,
) -> Result<()> {
    for site in sites {
        let raw = resolve(site.source)
            .ok_or_else(|| Error::Runtime { message: format!("HCQ patch value missing for {:?}", site.source) })?;
        let value = raw.wrapping_add(site.addend);
        let (encoded, width) = match site.encoding {
            PatchEncoding::U16 => {
                let value = u16::try_from(value).map_err(|_| Error::Runtime {
                    message: format!("HCQ patch value {value} does not fit u16 at byte {}", site.byte_offset),
                })?;
                (value as u64, 2)
            }
            PatchEncoding::U32 => {
                let value = u32::try_from(value).map_err(|_| Error::Runtime {
                    message: format!("HCQ patch value {value} does not fit u32 at byte {}", site.byte_offset),
                })?;
                (value as u64, 4)
            }
            PatchEncoding::U64 => (value, 8),
            PatchEncoding::Low32 => (value as u32 as u64, 4),
            PatchEncoding::High32 => ((value >> 32) as u32 as u64, 4),
            PatchEncoding::High32Or(bits) => (((value >> 32) as u32 | bits) as u64, 4),
            PatchEncoding::Low32ShiftRight(shift) => ((value >> shift) as u32 as u64, 4),
            PatchEncoding::High32ShiftRight(shift) => (((value >> shift) >> 32) as u32 as u64, 4),
        };
        let end = site
            .byte_offset
            .checked_add(width)
            .ok_or_else(|| Error::Runtime { message: "HCQ patch offset overflow".into() })?;
        let buffer_len = bytes.len();
        let dst = bytes.get_mut(site.byte_offset..end).ok_or_else(|| Error::Runtime {
            message: format!("HCQ patch at byte {} exceeds {buffer_len}-byte command buffer", site.byte_offset),
        })?;
        dst.copy_from_slice(&encoded.to_le_bytes()[..width]);
    }
    Ok(())
}

/// Canonical Tinygrad C-like kernarg ABI in ascending PARAM slot order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClikeKernargLayout {
    pub globals: usize,
    pub vars: usize,
    storage_order: Vec<bool>,
}

impl ClikeKernargLayout {
    pub fn from_abi(abi: &[crate::device::AbiParamDescriptor]) -> Self {
        let storage_order = abi.iter().map(crate::device::AbiParamDescriptor::is_storage).collect::<Vec<_>>();
        let globals = storage_order.iter().filter(|&&storage| storage).count();
        Self { globals, vars: storage_order.len() - globals, storage_order }
    }

    pub fn packed_size(&self) -> usize {
        let mut cursor = 0usize;
        for &storage in &self.storage_order {
            let width = if storage { 8 } else { 4 };
            cursor = cursor.next_multiple_of(width) + width;
        }
        cursor
    }

    /// Resolve PROGRAM globals from positional call-buffer GETADDR values and
    /// preserve ProgramInfo's canonical globals/vars ABI order.
    pub fn pack_program(
        info: &svod_ir::ProgramInfo,
        abi: &[crate::device::AbiParamDescriptor],
        dst: &mut [u8],
        call_buffer_addresses: &[u64],
        vars: &[i64],
    ) -> Result<usize> {
        let abi_globals = abi.iter().filter(|arg| arg.is_storage()).map(|arg| arg.slot).collect::<Vec<_>>();
        let abi_vars = abi.iter().filter(|arg| !arg.is_storage()).cloned().collect::<Vec<_>>();
        let info_vars =
            info.vars.iter().map(crate::device::AbiParamDescriptor::from_param).collect::<Result<Vec<_>>>()?;
        let var_names = info_vars.iter().map(|arg| arg.name.clone().unwrap_or_default()).collect::<Vec<_>>();
        crate::device::validate_abi_descriptors(abi, info.globals.len(), &var_names)?;
        if abi_globals != info.globals || abi_vars != info_vars {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "kernarg ABI {abi:?} disagrees with ProgramInfo globals={:?}, vars={info_vars:?}",
                    info.globals
                ),
            });
        }
        if call_buffer_addresses.len() != info.globals.len() {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "HCQ PROGRAM expected {} compact call buffers for slots {:?}, got {}",
                    info.globals.len(),
                    info.globals,
                    call_buffer_addresses.len()
                ),
            });
        }
        Self::from_abi(abi).pack(dst, call_buffer_addresses, vars)
    }

    /// Pack resolved GETADDR and scalar values in the descriptor's unified
    /// ascending-slot order.
    pub fn pack(&self, dst: &mut [u8], globals: &[u64], vars: &[i64]) -> Result<usize> {
        if globals.len() != self.globals || vars.len() != self.vars {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "HCQ kernargs expected {} globals/{} vars, got {}/{}",
                    self.globals,
                    self.vars,
                    globals.len(),
                    vars.len()
                ),
            });
        }
        let needed = self.packed_size();
        if dst.len() < needed {
            return Err(Error::Runtime {
                message: format!("HCQ kernarg destination has {} bytes, needs {needed}", dst.len()),
            });
        }
        dst[..needed].fill(0);
        let (mut global, mut var) = (0usize, 0usize);
        let mut cursor = 0usize;
        for &storage in &self.storage_order {
            if storage {
                cursor = cursor.next_multiple_of(8);
                dst[cursor..cursor + 8].copy_from_slice(&globals[global].to_le_bytes());
                global += 1;
                cursor += 8;
            } else {
                cursor = cursor.next_multiple_of(4);
                dst[cursor..cursor + 4].copy_from_slice(&(vars[var] as i32).to_le_bytes());
                var += 1;
                cursor += 4;
            }
        }
        Ok(cursor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceholderKind {
    Scratch,
    Kernargs,
}

/// Byte alignment of one kernarg record.
///
/// Tinygrad bump-allocates kernarg blocks at alignment 8
/// (`runtime/support/hcq.py:352`). 16 is the largest alignment any AMDHSA
/// kernarg member requires, so it is the smallest value that is correct for
/// every record we emit; the previous 128 (a cache-line guess) inflated linked
/// plans and graphs by up to 8x.
pub const KERNARG_ALIGN: usize = 16;

/// Pack `sizes` as consecutive `align`-aligned kernarg records, returning each
/// record's byte offset plus the total allocation size.
pub fn kernarg_offsets(sizes: impl IntoIterator<Item = usize>, align: usize) -> (Vec<usize>, usize) {
    let mut offsets = Vec::new();
    let mut bytes = 0usize;
    for size in sizes {
        bytes = bytes.next_multiple_of(align);
        offsets.push(bytes);
        bytes += size;
    }
    (offsets, bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaceholderRequest {
    pub kind: PlaceholderKind,
    pub bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaceholderPacking {
    pub offsets: Vec<usize>,
    pub scratch_bytes: usize,
    pub kernarg_bytes: usize,
}

impl PlaceholderPacking {
    /// HCQ2 packing: scratch placeholders alias one max-sized allocation;
    /// kernarg blocks are distinct and [`KERNARG_ALIGN`]-aligned. The two kinds
    /// interleave, so this cannot delegate to [`kernarg_offsets`] directly.
    pub fn pack(requests: &[PlaceholderRequest]) -> Self {
        let mut offsets = Vec::with_capacity(requests.len());
        let mut scratch_bytes = 0usize;
        let mut kernarg_bytes = 0usize;
        for request in requests {
            match request.kind {
                PlaceholderKind::Scratch => {
                    offsets.push(0);
                    scratch_bytes = scratch_bytes.max(request.bytes);
                }
                PlaceholderKind::Kernargs => {
                    kernarg_bytes = kernarg_bytes.next_multiple_of(KERNARG_ALIGN);
                    offsets.push(kernarg_bytes);
                    kernarg_bytes += request.bytes;
                }
            }
        }
        Self { offsets, scratch_bytes, kernarg_bytes }
    }
}

/// Deterministic HCQ target used to test scheduling without a device. It models
/// timeline waits/stores and records every accepted command in submission order.
#[derive(Debug)]
pub struct NullHcq {
    signals: std::collections::HashMap<u64, u64>,
    trace: Vec<(QueueKind, Command)>,
    clock_ns: u64,
    clock_step_ns: u64,
}

impl Default for NullHcq {
    fn default() -> Self {
        Self { signals: std::collections::HashMap::new(), trace: Vec::new(), clock_ns: 0, clock_step_ns: 1 }
    }
}

impl NullHcq {
    /// Construct a deterministic device clock. Timestamp commands write the
    /// current nanosecond value and advance by `step_ns`.
    pub fn with_clock(start_ns: u64, step_ns: u64) -> Self {
        Self { clock_ns: start_ns, clock_step_ns: step_ns, ..Self::default() }
    }

    pub fn set_signal(&mut self, address: u64, value: u64) {
        self.signals.insert(address, value);
    }

    pub fn submit(&mut self, submission: &Submission) -> Result<()> {
        match self.submit_with(submission, |_| Ok::<_, std::convert::Infallible>(())) {
            Ok(()) => Ok(()),
            Err(SubmissionExecutionError::Queue(error)) => Err(error),
            Err(SubmissionExecutionError::Execute(error)) => match error {},
        }
    }

    /// Submit while interpreting neutral `Execute` commands through a host
    /// callback. Copy/compute packet commands remain trace-only on the null
    /// target, while waits, timestamps, and finalizer stores are modeled.
    pub fn submit_with<E>(
        &mut self,
        submission: &Submission,
        mut execute: impl FnMut(usize) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), SubmissionExecutionError<E>> {
        for command in &submission.commands {
            if let Command::Wait { signal_address, value } = command {
                let current = self.signals.get(signal_address).copied().unwrap_or(0);
                if current < *value {
                    return Err(SubmissionExecutionError::Queue(Error::Runtime {
                        message: format!(
                            "null HCQ unsatisfied wait at {signal_address:#x}: current {current}, target {value}"
                        ),
                    }));
                }
            }
            if let Command::Execute { operation } = command {
                execute(*operation).map_err(SubmissionExecutionError::Execute)?;
            }
            if let Command::Store { dst, value, .. } = command {
                self.signals.insert(*dst, *value);
            }
            if let Command::Timestamp { dst } = command {
                self.signals.insert(*dst, self.clock_ns);
                self.clock_ns = self.clock_ns.wrapping_add(self.clock_step_ns);
            }
            self.trace.push((submission.queue, command.clone()));
        }
        Ok(())
    }

    pub fn trace(&self) -> &[(QueueKind, Command)] {
        &self.trace
    }

    pub fn signal_value(&self, address: u64) -> Option<u64> {
        self.signals.get(&address).copied()
    }
}
