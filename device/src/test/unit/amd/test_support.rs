//! Shared scaffolding for the AMD tests: the host-only [`MockAmdIface`] backend
//! and the gating helpers that skip hardware probes on unsupported hosts.
//!
//! Sibling modules under `crate::test::unit::amd` reach these via
//! `super::test_support`.

use crate::amd::AmdAllocator;
use crate::amd::connector::PoolQueue;

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::{HashMap, HashSet, VecDeque};
use std::ptr::NonNull;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::amd::device::AmdDevice;
use crate::amd::iface::{
    AllocKind, AllocResult, AmdIface, PublicationStage, QueueEventMailbox, QueueHandle, QueueTeardown, RingDesc,
};
use crate::amd::va_registry::AllocTag;
use crate::error::{Error, Result};

const MOCK_ALIGNMENT: usize = 0x1000;
const MOCK_DOORBELL_BYTES: usize = 0x2000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MockAmdCall {
    Alloc { size: usize, cpu_access: bool, zero: bool },
    Free { gpu_va: u64, size: usize, handle: u64 },
    SetupRing { ring_size: usize, queue_type: u32 },
    TeardownRing { queue_id: u32 },
    UpdateQueue { queue_id: u32, percentage: u32 },
    WaitEvents { timeout_ms: u32 },
    PublicationCheckpoint { stage: PublicationStage },
    HdpFlush,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MockFreeIssue {
    DoubleFree { handle: u64 },
    UnknownFree { gpu_va: u64, size: usize, handle: u64 },
}

struct MockAllocation {
    ptr: NonNull<u8>,
    layout: Layout,
}

// SAFETY: the allocation has no thread affinity and access to its lifetime is
// serialized through MockAmdIface's state mutex.
unsafe impl Send for MockAllocation {}

impl MockAllocation {
    fn new(size: usize) -> Result<Self> {
        let layout = Layout::from_size_align(size, MOCK_ALIGNMENT)
            .map_err(|_| Error::AmdAllocFailed { reason: format!("invalid mock allocation layout: {size}") })?;
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| Error::AmdAllocFailed { reason: format!("mock host allocation failed: {size}") })?;
        Ok(Self { ptr, layout })
    }
}

impl Drop for MockAllocation {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

struct MockLiveAllocation {
    _memory: MockAllocation,
    gpu_va: u64,
    size: usize,
}

struct MockQueue {
    doorbell: MockAllocation,
}

enum MockPublicationOutcome {
    Return(Result<()>),
    Panic,
}

#[derive(Default)]
struct MockAmdState {
    next_handle: u64,
    next_queue_id: u32,
    allocations: usize,
    frees: usize,
    queue_setups: usize,
    queue_teardowns: usize,
    live: HashMap<u64, MockLiveAllocation>,
    freed: HashSet<u64>,
    queues: HashMap<u32, MockQueue>,
    alloc_script: VecDeque<Result<()>>,
    setup_script: VecDeque<Result<()>>,
    teardown_script: VecDeque<Result<QueueTeardown>>,
    wait_script: VecDeque<Result<Option<Error>>>,
    publication_script: VecDeque<MockPublicationOutcome>,
    alloc_tags: Vec<(AllocTag, AllocKind)>,
    transcript: Vec<MockAmdCall>,
    free_issues: Vec<MockFreeIssue>,
    queue_event_mailbox: Option<QueueEventMailbox>,
    ctx_save_areas: Vec<(u64, u32)>,
    hdp_flush_register: bool,
    during_hdp_flush: Option<Box<dyn FnOnce() + Send>>,
}

/// Host-only AMD backend with page-aligned, stable allocations and explicit
/// lifecycle accounting. Scripted outcomes are consumed FIFO; an empty script
/// uses the successful default for that operation.
#[derive(Default)]
pub(crate) struct MockAmdIface {
    state: Mutex<MockAmdState>,
}

impl std::fmt::Debug for MockAmdIface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock();
        f.debug_struct("MockAmdIface")
            .field("allocations", &state.allocations)
            .field("frees", &state.frees)
            .field("live_handles", &state.live.len())
            .field("live_queues", &state.queues.len())
            .finish()
    }
}

impl MockAmdIface {
    pub(crate) fn script_alloc(&self, outcome: Result<()>) {
        self.state.lock().alloc_script.push_back(outcome);
    }

    /// Let the next `ok` allocations succeed and fail the one after, as `stage`.
    pub(crate) fn script_alloc_fail_at(&self, ok: usize, stage: &str) {
        let mut state = self.state.lock();
        state.alloc_script.extend((0..ok).map(|_| Ok(())));
        state.alloc_script.push_back(Err(scripted_error(stage)));
    }

    pub(crate) fn script_setup(&self, outcome: Result<()>) {
        self.state.lock().setup_script.push_back(outcome);
    }

    pub(crate) fn script_teardown(&self, outcome: Result<QueueTeardown>) {
        self.state.lock().teardown_script.push_back(outcome);
    }

    /// Give the mock KFD's queue-completion mailbox, as a real event page does.
    pub(crate) fn set_queue_event_mailbox(&self, mailbox: QueueEventMailbox) {
        self.state.lock().queue_event_mailbox = Some(mailbox);
    }

    /// Give the mock KFD's MMIO remap page, whose HDP flush register the host
    /// writes after its VRAM stores; without it the flush is skipped.
    pub(crate) fn set_hdp_flush_register(&self) {
        self.state.lock().hdp_flush_register = true;
    }

    /// Run `hook` inside the next HDP flush, after its write and before it lands.
    pub(crate) fn during_next_hdp_flush(&self, hook: impl FnOnce() + Send + 'static) {
        self.state.lock().during_hdp_flush = Some(Box::new(hook));
    }

    pub(crate) fn script_wait(&self, outcome: Result<Option<Error>>) {
        self.state.lock().wait_script.push_back(outcome);
    }

    pub(crate) fn script_publication(&self, outcome: Result<()>) {
        self.state.lock().publication_script.push_back(MockPublicationOutcome::Return(outcome));
    }

    pub(crate) fn script_publication_panic(&self) {
        self.state.lock().publication_script.push_back(MockPublicationOutcome::Panic);
    }

    /// `(allocations, frees, live handles)` so far.
    pub(crate) fn counts(&self) -> (usize, usize, usize) {
        let state = self.state.lock();
        (state.allocations, state.frees, state.live.len())
    }

    pub(crate) fn allocation_count(&self) -> usize {
        self.state.lock().allocations
    }

    pub(crate) fn free_count(&self) -> usize {
        self.state.lock().frees
    }

    /// Successful allocations recorded for one [`AllocTag`], live or freed.
    pub(crate) fn alloc_count_for_tag(&self, tag: AllocTag) -> usize {
        self.alloc_kinds_for_tag(tag).len()
    }

    /// The [`AllocKind`] of each successful allocation recorded for `tag`.
    pub(crate) fn alloc_kinds_for_tag(&self, tag: AllocTag) -> Vec<AllocKind> {
        self.state.lock().alloc_tags.iter().filter(|(recorded, _)| *recorded == tag).map(|(_, kind)| *kind).collect()
    }

    pub(crate) fn queue_setup_count(&self) -> usize {
        self.state.lock().queue_setups
    }

    pub(crate) fn queue_teardown_count(&self) -> usize {
        self.state.lock().queue_teardowns
    }

    pub(crate) fn live_queue_count(&self) -> usize {
        self.state.lock().queues.len()
    }

    /// Each compute queue's ctx-save area: its address (the mock GPU VA is the
    /// host address) and per-XCC stride, in setup order.
    pub(crate) fn ctx_save_areas(&self) -> Vec<(u64, u32)> {
        self.state.lock().ctx_save_areas.clone()
    }

    pub(crate) fn transcript(&self) -> Vec<MockAmdCall> {
        self.state.lock().transcript.clone()
    }

    pub(crate) fn free_issues(&self) -> Vec<MockFreeIssue> {
        self.state.lock().free_issues.clone()
    }

    /// Publication checkpoints reached so far, in call order.
    pub(crate) fn publication_stages(&self) -> Vec<PublicationStage> {
        self.state
            .lock()
            .transcript
            .iter()
            .filter_map(|call| match call {
                MockAmdCall::PublicationCheckpoint { stage } => Some(*stage),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn call_count(&self, matches: impl Fn(&MockAmdCall) -> bool) -> usize {
        self.state.lock().transcript.iter().filter(|call| matches(call)).count()
    }
}

/// A synthetic `xccs`-XCC device (1 selects the PM4 path, >1 the AQL path) over
/// a fresh mock backend.
pub(crate) fn mock_device(xccs: u32) -> (Arc<MockAmdIface>, AmdAllocator) {
    let iface = Arc::new(MockAmdIface::default());
    let dev = AmdDevice::synthetic_with_xcc(Arc::clone(&iface) as Arc<dyn AmdIface>, xccs);
    (iface, AmdAllocator { dev, device_id: 0 })
}

/// [`mock_device`] with a signal pool installed — the precondition for every
/// queue that owns a completion timeline.
pub(crate) fn mock_device_with_signals(xccs: u32) -> (Arc<MockAmdIface>, AmdAllocator) {
    let (iface, allocator) = mock_device(xccs);
    install_signal_pool(&allocator);
    (iface, allocator)
}

/// A fresh lane of `allocator`'s device.
pub(crate) fn pool_queue(allocator: &AmdAllocator) -> Arc<PoolQueue> {
    PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), allocator).expect("pool queue")
}

pub(crate) fn install_signal_pool(allocator: &AmdAllocator) {
    allocator.dev.core().install_signal_pool(crate::amd::signal::SignalPool::new(allocator, 64).expect("signal pool"));
}

/// Play the GPU retiring everything `lane` published: XCC0's timeline stores
/// and, on AQL, the completion of each submission's last packet.
pub(crate) fn retire_lane(lane: &PoolQueue) {
    crate::sync::TimelineSignal::set(&**lane.pm4_signal(), lane.pm4_value() - 1);
    if let Some(retired) = lane.retired() {
        while crate::sync::TimelineSignal::value(&**retired.signal()) + 1 < retired.current() {
            retired.signal().complete_packet();
        }
    }
}

pub(crate) fn scripted_error(stage: &str) -> Error {
    Error::Runtime { message: format!("scripted {stage} failure") }
}

pub(crate) fn replay_dwords(bytes: &[u8]) -> Vec<u32> {
    bytes.as_chunks::<4>().0.iter().copied().map(u32::from_le_bytes).collect()
}

impl AmdIface for MockAmdIface {
    fn alloc_raw(
        &self,
        size: usize,
        kind: AllocKind,
        tag: AllocTag,
        cpu_access: bool,
        zero: bool,
    ) -> Result<AllocResult> {
        let size = size.max(1).next_multiple_of(MOCK_ALIGNMENT);
        let mut state = self.state.lock();
        state.transcript.push(MockAmdCall::Alloc { size, cpu_access, zero });
        if let Some(outcome) = state.alloc_script.pop_front() {
            outcome?;
        }
        if zero && !cpu_access {
            return Err(Error::AmdAllocFailed { reason: "mock cannot zero a device-only allocation".into() });
        }
        let memory = MockAllocation::new(size)?;
        let gpu_va = memory.ptr.as_ptr() as u64;
        let host_ptr = cpu_access.then_some(memory.ptr);
        state.next_handle = state.next_handle.max(1);
        let handle = state.next_handle;
        state.next_handle += 1;
        let previous = state.live.insert(handle, MockLiveAllocation { _memory: memory, gpu_va, size });
        debug_assert!(previous.is_none());
        state.allocations += 1;
        state.alloc_tags.push((tag, kind));
        Ok(AllocResult { gpu_va, host_ptr, handle, size })
    }

    fn free_raw(&self, gpu_va: u64, size: usize, handle: u64) {
        let mut state = self.state.lock();
        state.transcript.push(MockAmdCall::Free { gpu_va, size, handle });
        let Some(allocation) = state.live.remove(&handle) else {
            let issue = if state.freed.contains(&handle) {
                MockFreeIssue::DoubleFree { handle }
            } else {
                MockFreeIssue::UnknownFree { gpu_va, size, handle }
            };
            state.free_issues.push(issue);
            return;
        };
        if allocation.gpu_va != gpu_va || allocation.size != size {
            state.live.insert(handle, allocation);
            state.free_issues.push(MockFreeIssue::UnknownFree { gpu_va, size, handle });
            return;
        }
        // Removing and dropping the backing allocation invalidates the address
        // only after the matching free has been recorded.
        drop(allocation);
        state.freed.insert(handle);
        state.frees += 1;
    }

    fn setup_ring(&self, desc: &RingDesc) -> Result<QueueHandle> {
        let mut state = self.state.lock();
        state.transcript.push(MockAmdCall::SetupRing { ring_size: desc.ring_size, queue_type: desc.queue_type });
        if let Some(outcome) = state.setup_script.pop_front() {
            outcome?;
        }
        let doorbell = MockAllocation::new(MOCK_DOORBELL_BYTES)?;
        let doorbell_base = doorbell.ptr;
        let doorbell_ptr = doorbell.ptr.cast::<u64>();
        state.next_queue_id = state.next_queue_id.max(1);
        let queue_id = state.next_queue_id;
        state.next_queue_id += 1;
        state.queues.insert(queue_id, MockQueue { doorbell });
        if desc.ctx_gpu != 0 {
            state.ctx_save_areas.push((desc.ctx_gpu, desc.ctx_save_restore_size));
        }
        state.queue_setups += 1;
        Ok(QueueHandle { queue_id, doorbell_base, doorbell: doorbell_ptr })
    }

    fn teardown_ring(&self, queue_id: u32, doorbell_base: NonNull<u8>) -> Result<QueueTeardown> {
        let mut state = self.state.lock();
        state.transcript.push(MockAmdCall::TeardownRing { queue_id });
        if let Some(outcome) = state.teardown_script.pop_front() {
            let outcome = outcome?;
            let queue = state.queues.remove(&queue_id).ok_or_else(|| Error::AmdQueueStillActive {
                queue_id,
                cause: "mock teardown of unknown queue".into(),
            })?;
            if queue.doorbell.ptr != doorbell_base {
                state.queues.insert(queue_id, queue);
                return Err(Error::AmdQueueStillActive { queue_id, cause: "mock doorbell mismatch".into() });
            }
            state.queue_teardowns += 1;
            return Ok(outcome);
        }
        let queue = state
            .queues
            .remove(&queue_id)
            .ok_or_else(|| Error::AmdQueueStillActive { queue_id, cause: "mock teardown of unknown queue".into() })?;
        if queue.doorbell.ptr != doorbell_base {
            state.queues.insert(queue_id, queue);
            return Err(Error::AmdQueueStillActive { queue_id, cause: "mock doorbell mismatch".into() });
        }
        state.queue_teardowns += 1;
        Ok(QueueTeardown::Complete)
    }

    fn update_queue_percentage(&self, queue_id: u32, _ring_gpu: u64, _ring_size: u32, percentage: u32) -> Result<()> {
        self.state.lock().transcript.push(MockAmdCall::UpdateQueue { queue_id, percentage });
        Ok(())
    }

    fn wait_events(&self, timeout_ms: u32, _age: &mut u64) -> Result<Option<Error>> {
        let scripted = {
            let mut state = self.state.lock();
            state.transcript.push(MockAmdCall::WaitEvents { timeout_ms });
            state.wait_script.pop_front()
        };
        if let Some(outcome) = scripted {
            return outcome;
        }
        if timeout_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(timeout_ms.into()));
        }
        Ok(None)
    }

    fn queue_event_mailbox(&self) -> Option<QueueEventMailbox> {
        self.state.lock().queue_event_mailbox
    }

    fn flush_hdp(&self) {
        let hook = {
            let mut state = self.state.lock();
            if !state.hdp_flush_register {
                return;
            }
            state.transcript.push(MockAmdCall::HdpFlush);
            state.during_hdp_flush.take()
        };
        if let Some(hook) = hook {
            hook();
        }
    }

    fn publication_checkpoint(&self, stage: PublicationStage) -> Result<()> {
        let mut state = self.state.lock();
        state.transcript.push(MockAmdCall::PublicationCheckpoint { stage });
        match state.publication_script.pop_front() {
            Some(MockPublicationOutcome::Return(outcome)) => outcome,
            Some(MockPublicationOutcome::Panic) => panic!("scripted publication panic at {stage:?}"),
            None => Ok(()),
        }
    }
}

/// Open the device-0 AMD allocator, or `None` on any host that lacks a supported
/// AMD GPU — no `/dev/kfd`, unsupported arch, or missing permissions. Hardware
/// tests early-return on `None`.
pub(crate) fn amd_alloc_or_skip() -> Option<AmdAllocator> {
    AmdAllocator::new(0).ok()
}

/// Install the signal pool a hardware probe needs, unless a device factory
/// already did (both installers are idempotent).
pub(crate) fn ensure_hw_signal_pool(alloc: &AmdAllocator) {
    if alloc.dev.core().signal_pool().is_none() {
        install_signal_pool(alloc);
    }
}

/// `true` if `alloc` drives a multi-XCC (CDNA SPX) device. The AQL probes are
/// meaningless on a single-XCC part, so they gate on this.
pub(crate) fn require_multi_xcc(alloc: &AmdAllocator) -> bool {
    alloc.dev.node.num_xcc.max(1) > 1
}

/// `true` if `alloc` drives a single-XCC (RDNA / gfx11/12) PM4 device. The PM4
/// graph-capture probes only exercise the PM4 indirect-buffer path.
pub(crate) fn require_single_xcc(alloc: &AmdAllocator) -> bool {
    alloc.dev.node.num_xcc.max(1) == 1
}
