//! Exclusive AMD compute lanes and logical execution contexts.
//!
//! A `PoolQueue` bundles the dispatch state that backs a single KFD compute
//! queue: the queue itself (ring + doorbell), a kernarg bump arena, the scratch
//! backing, the compute timeline counter, and linked-plan finalizers. A queue is
//! published through only while a non-clone `QueueLease` owns its lane bit.
//! Uncontended acquisition is one atomic compare-exchange; bounded contention
//! parks instead of co-tenanting a mutable hardware ring.
//!
//! `OwnerCtx` is logical per-plan state. It owns completion bookkeeping,
//! profiling configuration, and replay templates, but not a queue. The whole-
//! pool drain
//! (`PoolQueue::drain_all`, reached via `AmdDeviceCore::synchronize_all`) is the
//! host-visibility/free fence used by `AmdAllocator::_copyin`/`_copyout`/`_free`.

#![cfg(unix)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::{Condvar, Mutex};

use crate::amd::AmdAllocator;
use crate::amd::device::{AmdDeviceCore, ScratchState, alloc_scratch};
use crate::amd::kernarg::KernargArena;
use crate::amd::queue::AmdComputeQueue;
use crate::amd::signal::{AmdSignal, TIMELINE_WRAP_WATERMARK, Timeline};
use crate::error::{Error, Result};
use crate::sync::TimelineSignal;

/// One hardware compute lane, retained by [`QueuePool`] and published through
/// only by its current [`QueueLease`].
///
/// Owns: KFD compute queue, kernarg arena, scratch backing, PM4 completion
/// counter, and lane-level linked-plan finalizers.
#[derive(Debug)]
pub(crate) struct PoolQueue {
    /// Shared immutable identity. Cloned across all queues backed by the same
    /// physical AMD:N (and across `AmdDevice` for back-compat).
    core: Arc<AmdDeviceCore>,
    /// The KFD compute queue: ring, doorbell, and GART. Its backend-local mutex
    /// is uncontended because publication requires the lane's unique lease.
    queue: Box<AmdComputeQueue>,
    /// Kernel-argument bump arena (16 MiB GTT). One per DEVICE, shared by every
    /// lane: the wrap path drains all of them through
    /// `AmdDeviceCore::synchronize_all`, so a wrapped slot is provably free
    /// whichever lane wrote it. Freed by `Drop for KernargArena` once the last
    /// sharing queue drops, after `Drop for PoolQueue` has drained.
    arena: Arc<KernargArena>,
    /// Scratch backing. Grown on demand by [`ensure_has_local_memory`](Self::ensure_has_local_memory)
    /// while the lane's exclusive lease is held.
    scratch_state: Mutex<ScratchState>,
    /// PM4 monotonic completion counter + its signal. Used by the PM4 single-XCC
    /// dispatch path and the SDMA-style monotonic drain. (The SDMA copy queue
    /// has its own separate `Timeline`; this one is untouched by SDMA.) AQL
    /// carries this timeline's waits and stores through vendor IB packets.
    pm4_counter: Arc<Timeline>,
    /// Pool-level in-flight linked-plan finalizers (FIFO) for all owners.
    /// `synchronize_all` drains them via the core's `Weak<PoolQueue>` registry
    /// without touching the queue.
    inflight: Mutex<VecDeque<Arc<SubmissionFinalizer>>>,
    /// Linked command buffers minted against this lane. Linked storage embeds
    /// this lane's GPU virtual addresses (scratch, kernarg arena, control
    /// program), so it is only reusable within the lane that produced it —
    /// hence one cache per lane rather than a fresh
    /// `CommandBufferCache::default()` at every link site, which never hit.
    link_cache: Mutex<crate::hcq::CommandBufferCache>,
}

/// Atomic lane claims, split from queue construction so exclusivity can be
/// tested without AMD hardware.
#[derive(Debug)]
pub(crate) struct LaneClaims {
    claimed: AtomicU64,
    capacity: usize,
}

impl LaneClaims {
    pub(crate) fn new(capacity: usize) -> Self {
        Self { claimed: AtomicU64::new(0), capacity: capacity.clamp(1, u64::BITS as usize) }
    }

    pub(crate) fn try_claim(&self, initialized: usize) -> Option<usize> {
        let count = initialized.min(self.capacity);
        let valid = if count == u64::BITS as usize { u64::MAX } else { (1u64 << count).wrapping_sub(1) };
        let mut observed = self.claimed.load(Ordering::Acquire);
        loop {
            let available = valid & !observed;
            if available == 0 {
                return None;
            }
            let slot = available.trailing_zeros() as usize;
            match self.claimed.compare_exchange_weak(
                observed,
                observed | (1u64 << slot),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(slot),
                Err(actual) => observed = actual,
            }
        }
    }

    pub(crate) fn claim_new(&self, slot: usize) {
        let previous = self.claimed.fetch_or(1u64 << slot, Ordering::AcqRel);
        debug_assert_eq!(previous & (1u64 << slot), 0);
    }

    pub(crate) fn release(&self, slot: usize) {
        let previous = self.claimed.fetch_and(!(1u64 << slot), Ordering::Release);
        debug_assert_ne!(previous & (1u64 << slot), 0);
    }
}

/// How long one lane-acquisition park may last before it counts as expired.
/// Matches the device drain bound (tinygrad's `HCQDEV_WAIT_TIMEOUT_MS`).
#[cfg(not(test))]
const LANE_ACQUIRE_TIMEOUT_MS: u64 = 30_000;
#[cfg(test)]
const LANE_ACQUIRE_TIMEOUT_MS: u64 = 25;
/// Consecutive expiries tolerated before acquisition fails. One expiry can be a
/// legitimately long dispatch holding its lane; two in a row is a wedge.
const LANE_ACQUIRE_MAX_EXPIRIES: u32 = 2;

/// Bounded lazy pool of compute lanes. Queue creation is serialized and cold;
/// acquisition of an initialized idle lane is lock-free.
#[derive(Debug)]
pub(crate) struct QueuePool {
    queues: Box<[OnceLock<Arc<PoolQueue>>]>,
    initialized: AtomicUsize,
    claims: LaneClaims,
    create_lock: Mutex<()>,
    wait_lock: Mutex<()>,
    available: Condvar,
}

impl QueuePool {
    pub(crate) fn new(capacity: usize) -> Self {
        let capacity = capacity.clamp(1, u64::BITS as usize);
        Self {
            queues: (0..capacity).map(|_| OnceLock::new()).collect(),
            initialized: AtomicUsize::new(0),
            claims: LaneClaims::new(capacity),
            create_lock: Mutex::new(()),
            wait_lock: Mutex::new(()),
            available: Condvar::new(),
        }
    }

    /// Claim an idle lane, creating one while the pool is below capacity.
    /// Parking is bounded: a lease leaked by a wedged publisher must surface as
    /// a typed timeout rather than hanging every subsequent dispatch.
    pub(crate) fn acquire(&self, core: &Arc<AmdDeviceCore>, allocator: &AmdAllocator) -> Result<QueueLease> {
        let mut expiries = 0u32;
        loop {
            if let Some(error) = core.poison_error() {
                return Err(error);
            }
            let initialized = self.initialized.load(Ordering::Acquire);
            if let Some(slot) = self.claims.try_claim(initialized) {
                let queue = Arc::clone(self.queues[slot].get().expect("initialized lane missing queue"));
                return Ok(QueueLease { core: Arc::clone(core), slot, queue: Some(queue) });
            }

            {
                let _create = self.create_lock.lock();
                let initialized = self.initialized.load(Ordering::Acquire);
                if let Some(slot) = self.claims.try_claim(initialized) {
                    let queue = Arc::clone(self.queues[slot].get().expect("initialized lane missing queue"));
                    return Ok(QueueLease { core: Arc::clone(core), slot, queue: Some(queue) });
                }
                if initialized < self.claims.capacity {
                    let queue = PoolQueue::new_with_resources(Arc::clone(core), allocator)?;
                    self.claims.claim_new(initialized);
                    self.queues[initialized].set(Arc::clone(&queue)).expect("queue lane initialized twice");
                    self.initialized.store(initialized + 1, Ordering::Release);
                    return Ok(QueueLease { core: Arc::clone(core), slot: initialized, queue: Some(queue) });
                }
            }

            // Pair the retry with release's wait mutex to avoid a lost wakeup.
            let mut wait = self.wait_lock.lock();
            if let Some(error) = core.poison_error() {
                return Err(error);
            }
            let initialized = self.initialized.load(Ordering::Acquire);
            if let Some(slot) = self.claims.try_claim(initialized) {
                let queue = Arc::clone(self.queues[slot].get().expect("initialized lane missing queue"));
                return Ok(QueueLease { core: Arc::clone(core), slot, queue: Some(queue) });
            }
            if self.available.wait_for(&mut wait, std::time::Duration::from_millis(LANE_ACQUIRE_TIMEOUT_MS)).timed_out()
            {
                expiries += 1;
                if expiries >= LANE_ACQUIRE_MAX_EXPIRIES {
                    return Err(Error::TimelineTimeout {
                        what: "AMD lane acquisition",
                        target: self.claims.capacity as u64,
                        current: u64::from(self.claims.claimed.load(Ordering::Acquire).count_ones()),
                        waited_ms: LANE_ACQUIRE_TIMEOUT_MS * u64::from(LANE_ACQUIRE_MAX_EXPIRIES),
                    });
                }
            } else {
                expiries = 0;
            }
        }
    }

    fn release(&self, slot: usize) {
        let _wait = self.wait_lock.lock();
        self.claims.release(slot);
        self.available.notify_one();
    }

    pub(crate) fn notify_poisoned(&self) {
        let _wait = self.wait_lock.lock();
        self.available.notify_all();
    }
}

/// Exclusive publication authority for one hardware compute lane.
pub(crate) struct QueueLease {
    core: Arc<AmdDeviceCore>,
    slot: usize,
    queue: Option<Arc<PoolQueue>>,
}

impl QueueLease {
    #[inline]
    pub fn pool(&self) -> &PoolQueue {
        self.queue.as_deref().expect("queue lease already released")
    }

    #[cfg(test)]
    pub fn queue_ptr(&self) -> *const PoolQueue {
        Arc::as_ptr(self.queue.as_ref().expect("queue lease already released"))
    }
}

impl std::ops::Deref for QueueLease {
    type Target = PoolQueue;

    fn deref(&self) -> &Self::Target {
        self.pool()
    }
}

impl std::fmt::Debug for QueueLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueueLease").field("slot", &self.slot).finish_non_exhaustive()
    }
}

impl Drop for QueueLease {
    fn drop(&mut self) {
        if self.queue.take().is_some() {
            self.core.queue_pool().release(self.slot);
        }
    }
}

/// Completion resources attached by the AMD queue finalizer to one HCQ
/// submission. This is the only object retained by owner and queue lifecycle
/// code; native AQL decrement semantics and PM4 memory-timeline semantics stay
/// private to the backend.
#[derive(Debug)]
pub(crate) struct SubmissionFinalizer {
    signal: Arc<AmdSignal>,
    value: u64,
    progress: Vec<Arc<AmdSignal>>,
    _timestamps: Option<Arc<AmdSignal>>,
    publication: Mutex<PublicationState>,
    publication_changed: Condvar,
    code: Mutex<Vec<Arc<crate::amd::program::CodeObject>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PublicationState {
    Prepared,
    Published,
    Failed,
}

impl SubmissionFinalizer {
    pub(crate) fn timeline(signal: Arc<AmdSignal>, value: u64, timestamps: Option<Arc<AmdSignal>>) -> Arc<Self> {
        Arc::new(Self {
            signal,
            value,
            progress: Vec::new(),
            _timestamps: timestamps,
            publication: Mutex::new(PublicationState::Published),
            publication_changed: Condvar::new(),
            code: Mutex::new(Vec::new()),
        })
    }

    pub(crate) fn prepared_timeline(signal: Arc<AmdSignal>, value: u64, progress: Vec<Arc<AmdSignal>>) -> Arc<Self> {
        Arc::new(Self {
            signal,
            value,
            progress,
            _timestamps: None,
            publication: Mutex::new(PublicationState::Prepared),
            publication_changed: Condvar::new(),
            code: Mutex::new(Vec::new()),
        })
    }

    pub(crate) fn mark_published(&self) {
        *self.publication.lock() = PublicationState::Published;
        self.publication_changed.notify_all();
    }

    pub(crate) fn mark_failed(&self) {
        *self.publication.lock() = PublicationState::Failed;
        self.publication_changed.notify_all();
    }

    pub(crate) fn retain_code(&self, code: Arc<crate::amd::program::CodeObject>) {
        self.code.lock().push(code);
    }

    /// Wait for this submission's terminal timeline point, bounded by
    /// `timeout_ms` across BOTH phases. A publisher that faults or panics may
    /// never transition a `Prepared` finalizer, so parking untimed here wedged
    /// the caller for good; tinygrad bounds every device wait with
    /// `HCQDEV_WAIT_TIMEOUT_MS`.
    pub fn wait(&self, timeout_ms: u64) -> Result<()> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        let mut publication = self.publication.lock();
        while *publication == PublicationState::Prepared {
            if let Some(error) = self.signal.device_poison() {
                return Err(error);
            }
            if self.publication_changed.wait_until(&mut publication, deadline).timed_out()
                && *publication == PublicationState::Prepared
            {
                return Err(Error::TimelineTimeout {
                    what: "AMD submission publication",
                    target: self.value,
                    current: self.signal.value(),
                    waited_ms: timeout_ms,
                });
            }
        }
        match *publication {
            PublicationState::Published => {
                drop(publication);
                // Whatever the publication wait consumed comes off the signal
                // wait's budget; never hand it 0, which means "no timeout".
                let remaining = deadline.saturating_duration_since(std::time::Instant::now()).as_millis() as u64;
                self.signal.wait_signal_value_with_progress(self.value, remaining.max(1), &self.progress)
            }
            PublicationState::Failed => Err(Error::Runtime {
                message: "AMD submission failed before its terminal timeline point was published".into(),
            }),
            PublicationState::Prepared => unreachable!(),
        }
    }

    pub(crate) fn retired(&self) -> bool {
        match *self.publication.lock() {
            PublicationState::Published => self.signal.value() >= self.value,
            PublicationState::Failed => true,
            PublicationState::Prepared => false,
        }
    }
}

impl crate::sync::CompletionToken for SubmissionFinalizer {
    fn wait(&self, timeout_ms: u64) -> Result<()> {
        SubmissionFinalizer::wait(self, timeout_ms)
    }

    fn retired(&self) -> bool {
        SubmissionFinalizer::retired(self)
    }
}

impl PoolQueue {
    /// Link `lowered` against this lane, reusing the linked storage when the
    /// same packets and immutable addresses were linked here before.
    ///
    /// Keyed on `(lane, device)` rather than `CommandBufferCache::link`'s
    /// `(0, DeviceSpec::Cpu)` placeholder: the addresses baked into a linked
    /// buffer are only valid inside the lane's VM.
    pub(crate) fn link(
        &self,
        lowered: &crate::hcq::LoweredCommandBuffer,
        values: &crate::hcq::LinkPatchValues,
    ) -> Result<Arc<crate::hcq::LinkedCommandBuffer>> {
        let device = svod_dtype::DeviceSpec::Amd { device_id: self.core.node.node_id as usize };
        self.link_cache.lock().link_for_context(self as *const Self as u64, &device, lowered, values)
    }

    /// Build a pool queue with its own KFD compute queue + kernarg arena +
    /// pre-reserved scratch. The PM4 counter signal is acquired from the core's
    /// shared `SignalPool` (which the factory must have installed first —
    /// `AmdDeviceCore::install_signal_pool`).
    ///
    /// Registers `Weak::self` in the core's queue list so device-wide
    /// `synchronize_all` (called by `AmdAllocator::_copyin`/`_copyout`/`_free`)
    /// drains every queue before any host-visible buffer free.
    pub fn new_with_resources(core: Arc<AmdDeviceCore>, allocator: &AmdAllocator) -> Result<Arc<Self>> {
        // Order matters: every step that allocates must come BEFORE
        // `alloc_scratch`. Earlier-built resources (`AmdComputeQueue`,
        // `KernargArena`, signal slot, counter Arc) all have RAII cleanup, so a
        // failure before the scratch alloc unwinds via `?`. The scratch backing
        // is the lone raw KFD allocation — keeping it last means a failure
        // before then leaks nothing.
        let queue = AmdComputeQueue::create(allocator)?;
        let arena = {
            let mut shared = core.kernarg_arena.lock();
            match shared.upgrade() {
                Some(arena) => arena,
                None => {
                    let arena = KernargArena::new(allocator, &core)?;
                    *shared = Arc::downgrade(&arena);
                    arena
                }
            }
        };
        let pool = core.signal_pool().cloned().ok_or_else(|| Error::Runtime {
            message: "PoolQueue::new_with_resources: signal pool not installed on core — \
                      install via AmdDeviceCore::install_signal_pool before building any queue"
                .into(),
        })?;
        let pm4_counter = Timeline::new(Arc::new(pool.acquire()?));
        // Pre-reserve a generous scratch buffer once (ROCr-style stable pool): the
        // mid-stream `set_aql_scratch` REALLOC (new VA + KFD-unmap of the old) on a
        // live multi-XCC queue silently wedges the CP. Pre-sizing here so no typical
        // kernel triggers a grow.
        let (scratch_va, scratch_size, tmpring_size, size_per_thread, scratch_handle, aql_desc) =
            alloc_scratch(core.iface(), &core.node, &core.arch, 2048)?;
        let q = Arc::new(Self {
            core,
            queue,
            arena,
            scratch_state: Mutex::new(ScratchState {
                gpu_va: scratch_va,
                size_per_thread,
                tmpring_size,
                handle: scratch_handle,
                size: scratch_size,
            }),
            pm4_counter,
            inflight: Mutex::new(VecDeque::new()),
            link_cache: Mutex::default(),
        });
        // Register in the core so `synchronize_all` can drain this queue's
        // in-flight work via `Weak<PoolQueue>`, reading only signal slots and
        // never touching the queue. Opportunistic GC of dropped entries.
        {
            let mut list = q.core.connectors.lock();
            list.retain(|w| w.strong_count() > 0);
            list.push(Arc::downgrade(&q));
        }
        // Publish the initial scratch descriptor into the AQL queue's GART page
        // (no-op on PM4 queues). Must happen before the first dispatch, and
        // must go through the unmap/remap cycle: the CP cached the descriptor
        // at CREATE_QUEUE, before scratch existed.
        q.queue().publish_aql_descriptor(&aql_desc)?;
        Ok(q)
    }

    /// Borrow this queue's KFD compute queue.
    #[inline]
    pub fn queue(&self) -> &AmdComputeQueue {
        &self.queue
    }

    /// Borrow this queue's kernarg arena.
    #[inline]
    pub fn arena(&self) -> &KernargArena {
        &self.arena
    }

    /// The immutable core this queue dispatches against.
    #[inline]
    pub fn core(&self) -> &Arc<AmdDeviceCore> {
        &self.core
    }

    /// PM4 completion-counter signal (forwards to the shared `Timeline`).
    pub fn pm4_signal(&self) -> &Arc<AmdSignal> {
        self.pm4_counter.signal()
    }

    /// Reserve the next PM4 counter value (`fetch_add(1)`). The caller emits a
    /// `RELEASE_MEM` packet that writes this value to the counter's signal slot.
    pub fn next_pm4(&self) -> u64 {
        self.pm4_counter.next()
    }

    pub(crate) fn rollback_pm4(&self, reserved: u64) -> bool {
        self.pm4_counter.rollback(reserved)
    }

    /// Highest submitted PM4 counter value (the value the next `signal` packet
    /// would write). A drain waits until the GPU has written `value - 1`.
    pub fn pm4_value(&self) -> u64 {
        self.pm4_counter.current()
    }

    /// Current scratch buffer GPU VA. Read under the scratch mutex; tiny lock
    /// window on the dispatch hot path.
    pub fn scratch_gpu_va(&self) -> u64 {
        self.scratch_state.lock().gpu_va
    }

    /// Packed `COMPUTE_TMPRING_SIZE` for the current scratch buffer.
    pub fn tmpring_size(&self) -> u32 {
        self.scratch_state.lock().tmpring_size
    }

    /// Drain ALL submitted GPU work on this queue (every owner's). Blocks until
    /// the PM4 counter observes `pm4_value() - 1`, then waits every in-flight
    /// linked-plan timeline. Reads only signal slots and does not interfere with
    /// lane acquisition.
    pub fn drain_all(&self) -> Result<()> {
        if let Some(err) = self.core.poison_error() {
            return Err(err);
        }
        // Drain the monotonic PM4 counter (PM4 / SDMA-style work)...
        self.pm4_counter.drain(30_000).inspect_err(|e| self.core.poison(&e.to_string()))?;
        // ...then every in-flight linked-plan timeline.
        // Snapshot under the lock, wait outside it, then drop the retired ones —
        // `retain` keeps any signal a concurrent dispatch armed after the
        // snapshot, so we never lose track of still-pending work.
        let snapshot: Vec<Arc<SubmissionFinalizer>> = self.inflight.lock().iter().cloned().collect();
        for finalizer in &snapshot {
            finalizer.wait(30_000).inspect_err(|e| self.core.poison(&e.to_string()))?;
        }
        self.inflight.lock().retain(|finalizer| !finalizer.retired());
        Ok(())
    }

    /// Allocate a profiling/timestamp slot. Its returned handle owns the slot;
    /// replay and runtime finalizers retain it until timestamp collection.
    pub fn acquire_timestamp_signal(&self) -> Result<Arc<AmdSignal>> {
        let pool =
            self.core.signal_pool().cloned().ok_or_else(|| Error::Runtime {
                message: "acquire_timestamp_signal: signal pool not installed".into(),
            })?;
        let sig = pool.acquire()?;
        Ok(Arc::new(sig))
    }

    /// Register an AQL submission finalizer as in-flight (FIFO,
    /// pool-level). Kept alive until it retires and a later acquire/drain
    /// reclaims it.
    pub fn register_inflight(&self, finalizer: Arc<SubmissionFinalizer>) {
        let mut inflight = self.inflight.lock();
        inflight.retain(|entry| !entry.retired());
        inflight.push_back(finalizer);
    }

    /// Keep the PM4 counter below 2^32 on the dispatch hot path.
    ///
    /// A drain resets the counter on wraparound, but a queue dispatched in a
    /// long `wait=false` PM4 loop never hits one, so the full-u64 counter would
    /// climb past 2^32 while the GPU's `RELEASE_MEM` writes only the low 32
    /// bits. A later drain would then wait for a full-u64 `target` the slot can
    /// never reach → false 30 s timeout. Calling this before reserving each
    /// counter value forces the drain+reset at the 2^31 watermark, so the
    /// reserved value stays `< 2^32` and the `as u32` truncations stay lossless.
    /// The exclusive lane lease makes the check + drain sequential.
    pub fn ensure_pm4_headroom(&self) -> Result<()> {
        if self.pm4_counter.current() > TIMELINE_WRAP_WATERMARK {
            self.drain_all()?;
            self.pm4_counter.reset_after_drain();
        }
        Ok(())
    }

    /// Ensure the queue's scratch backing has at least `private_segment_size`
    /// bytes per thread, growing it on demand. The old scratch buffer is freed
    /// (drain → unmap → munmap → free).
    pub fn ensure_has_local_memory(&self, private_segment_size: u32) -> Result<()> {
        // One check is enough: `QueueLease` is the exclusive publication
        // authority for this lane, so no concurrent grow can land between the
        // read and the replacement below, and no publisher can enqueue the
        // stale scratch VA during the transaction.
        if private_segment_size <= self.scratch_state.lock().size_per_thread {
            return Ok(());
        }
        self.drain_all()?;
        let (va, size, tmpring, rounded, handle, aql_desc) =
            alloc_scratch(self.core.iface(), &self.core.node, &self.core.arch, private_segment_size)?;
        let old = {
            let mut state = self.scratch_state.lock();
            let old = (state.gpu_va, state.size, state.handle);
            *state = ScratchState { gpu_va: va, size_per_thread: rounded, tmpring_size: tmpring, handle, size };
            old
        };
        // The exclusive lane lease keeps the new host state and live AQL descriptor
        // atomic with respect to publication. The successful drain proves the
        // old backing is no longer referenced by this lane — scratch is
        // lane-private, so no sibling lane can reference it either.
        self.queue.publish_aql_descriptor(&aql_desc)?;
        self.core.iface().free_raw(old.0, old.1, old.2);
        Ok(())
    }
}

impl Drop for PoolQueue {
    /// Drain in-flight GPU work before the queue dies (device close — the LAST
    /// `Arc<PoolQueue>` dropping), so a downstream host read of a buffer this
    /// queue wrote doesn't race a still-running kernel, and free the scratch
    /// backing (`ScratchState` is `Copy` with no `Drop`).
    ///
    /// Skipped during panic unwind: `drain_all` can block up to ~30 s per queue
    /// and an unwinding test would pay N × 30 s before teardown. The in-flight
    /// work is then abandoned — the caller saw a panic anyway. The lane is
    /// quarantined but the device is NOT poisoned: an unwind says nothing about
    /// the hardware, and tinygrad latches its per-device `error_state` only on
    /// a drain timeout or a reported fault.
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.queue.quarantine();
            tracing::warn!(
                "PoolQueue drop during panic unwind: skipping drain; \
                 in-flight GPU work + scratch backing abandoned"
            );
            return;
        }
        if let Err(e) = self.drain_all() {
            self.queue.quarantine();
            tracing::warn!(?e, "PoolQueue drop: drain failed; hardware allocations quarantined");
            return;
        }
        if self.queue.close().is_err() {
            return;
        }
        let state = *self.scratch_state.lock();
        self.core.iface().free_raw(state.gpu_va, state.size, state.handle);
    }
}

/// Logical per-plan context. Queue ownership is acquired separately through a
/// non-clone [`QueueLease`]. Direct fallback retains one lease as its replay
/// session so per-kernel trait calls preserve FIFO ordering.
pub(crate) struct OwnerCtx {
    core: Arc<AmdDeviceCore>,
    allocator: AmdAllocator,
    session: Mutex<Option<QueueLease>>,
    /// This owner's newest HCQ submission, independent of native queue path.
    newest: Mutex<Option<Arc<SubmissionFinalizer>>>,
    /// PMC: hardware counters to collect on profiling dispatches (empty = off).
    pmc: Mutex<Vec<crate::profile::AmdCounter>>,
    linked_plan: Mutex<Option<crate::amd::linked_plan::AmdLinkedPlan>>,
}

impl OwnerCtx {
    pub fn new(core: Arc<AmdDeviceCore>, allocator: AmdAllocator) -> Self {
        Self {
            core,
            allocator,
            session: Mutex::new(None),
            newest: Mutex::new(None),
            pmc: Mutex::new(Vec::new()),
            linked_plan: Mutex::new(None),
        }
    }

    /// Hardware counters to collect on this owner's profiling dispatches.
    pub fn pmc_counters(&self) -> Vec<crate::profile::AmdCounter> {
        self.pmc.lock().clone()
    }

    #[inline]
    pub fn core(&self) -> &Arc<AmdDeviceCore> {
        &self.core
    }

    pub(crate) fn allocator(&self) -> &AmdAllocator {
        &self.allocator
    }

    pub fn lease(&self) -> Result<QueueLease> {
        self.core.lease_queue(&self.allocator)
    }

    fn finish_session(&self) {
        drop(self.session.lock().take());
    }

    /// Record this owner's most recent in-flight finalizer (the one to wait on
    /// in the owner-local `synchronize`).
    pub fn set_newest(&self, finalizer: Arc<SubmissionFinalizer>) {
        *self.newest.lock() = Some(finalizer);
    }

    /// This owner's newest finalizer as a scoped-sync completion token. A new
    /// dispatch epoch retires the previous one first, so this token subsumes
    /// every earlier submission of this owner.
    pub fn completion_token(&self) -> Option<Arc<dyn crate::sync::CompletionToken>> {
        self.newest.lock().clone().map(|finalizer| finalizer as Arc<dyn crate::sync::CompletionToken>)
    }

    /// Owner-local drain: wait on ONLY this owner's last submitted work. AQL:
    /// wait its newest signal. PM4: wait the shared counter to reach this
    /// owner's high value. Polls the device poison latch and bails on fault.
    pub fn synchronize(&self) -> Result<()> {
        if let Some(err) = self.core.poison_error() {
            return Err(err);
        }
        // Dispatch holds `session` through `set_newest`, so taking the snapshot
        // under the same lock cannot miss an already-doorbelled submission.
        let (finalizer, lease) = {
            let mut session = self.session.lock();
            let finalizer = self.newest.lock().clone();
            (finalizer, session.take())
        };
        drop(lease);
        if let Some(finalizer) = finalizer {
            finalizer.wait(30_000).inspect_err(|e| self.core.poison(&e.to_string()))?;
        }
        Ok(())
    }
}

impl Drop for OwnerCtx {
    fn drop(&mut self) {
        if self.linked_plan.get_mut().is_none() {
            return;
        }
        // The cached linked plan owns host-visible kernarg storage referenced by
        // asynchronous dispatches. Fence the owner's final submission before
        // Rust drops the plan and unmaps that storage.
        if let Err(e) = self.synchronize() {
            tracing::warn!(?e, "OwnerCtx drop: linked-plan work could not be drained; storage quarantined");
            if let Some(plan) = self.linked_plan.get_mut().take() {
                std::mem::forget(plan);
            }
        }
    }
}

impl std::fmt::Debug for OwnerCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnerCtx").finish_non_exhaustive()
    }
}

/// Direct per-operation fallback retains one exclusive lane for the context's
/// replay session, preserving queue FIFO ordering until context drop.
impl crate::device::PlanContext for OwnerCtx {
    unsafe fn dispatch(
        &self,
        program: &dyn crate::device::Program,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
        profile: bool,
    ) -> Result<Option<Arc<dyn crate::DispatchTimestamps>>> {
        // A plan is single-backend, and this context was minted by one of its
        // AmdPrograms, so every kernel it dispatches is an AmdProgram. The
        // downcast recovers our own concrete type — a construction invariant,
        // not a runtime check.
        let amd = program
            .as_any()
            .downcast_ref::<crate::amd::AmdProgram>()
            .expect("AMD PlanContext dispatched a non-AMD program");
        if !Arc::ptr_eq(amd.device().core(), &self.core) {
            return Err(Error::Runtime {
                message: "AMD PlanContext received a program from another physical device".into(),
            });
        }
        let mut session = self.session.lock();
        let result = (|| {
            if session.is_none() {
                // A new epoch may use another lane; retire the previous epoch
                // before relying on this lane's FIFO ordering.
                if let Some(finalizer) = self.newest.lock().clone() {
                    finalizer.wait(30_000).inspect_err(|e| self.core.poison(&e.to_string()))?;
                }
                *session = Some(self.lease()?);
            }
            let lane = session.as_ref().unwrap();
            // Kept under the session mutex despite the drain it may run: that
            // mutex is what lets `synchronize` snapshot `newest` without racing
            // an already-doorbelled submission, and a grow only drains when the
            // pre-sized scratch is genuinely too small.
            lane.ensure_has_local_memory(amd.private_segment_size())?;
            // Only profiling callers retain timestamp handles long enough to
            // synchronize them; fire-and-forget dispatch must not arm probes.
            unsafe {
                amd.execute_on(self, lane.pool(), buffers, vals, global_size, local_size, /*wait=*/ false, profile)
            }
        })();
        if result.is_err() {
            drop(session.take());
        }
        result
    }

    fn completion_token(&self) -> Option<Arc<dyn crate::sync::CompletionToken>> {
        OwnerCtx::completion_token(self)
    }

    fn replay_linked_plan(
        &self,
        semantic: &crate::hcq::SemanticLinkedPlan,
        calls: &[crate::device::PlanCall<'_>],
    ) -> Result<crate::device::NativeReplayOutcome> {
        if let Some(reason) =
            crate::amd::linked_plan::native_topology_decline(semantic, self.core.copy_queue().is_some())
        {
            return Ok(crate::device::NativeReplayOutcome::Declined(reason));
        }
        let mut plan = self.linked_plan.lock();
        // Release a direct-session lease and retire this owner's prior mutable
        // replay storage before trying to claim a native replay lane.
        self.synchronize()?;
        let lane = self.lease()?;
        if plan.is_none() {
            let Some(captured) = crate::amd::linked_plan::AmdLinkedPlan::capture(self, lane.pool(), semantic, calls)?
            else {
                return Ok(crate::device::NativeReplayOutcome::Declined(
                    crate::device::NativeReplayDecline::BackendUnsupported,
                ));
            };
            *plan = Some(captured);
        }
        if let Err(failure) = plan.as_mut().unwrap().replay(self, lane.pool(), calls) {
            if failure.published {
                self.core.poison(&failure.error.to_string());
            }
            return Err(failure.error);
        }
        Ok(crate::device::NativeReplayOutcome::Executed)
    }

    fn set_pmc(&self, counters: &[crate::profile::PmcCounter]) -> usize {
        let mut armed = self.pmc.lock();
        *armed = counters
            .iter()
            .filter_map(|c| match c {
                crate::profile::PmcCounter::Amd(c) => Some(*c),
                _ => None,
            })
            .collect();
        armed.len()
    }

    fn pmc_available(&self) -> bool {
        crate::amd::pmc::stable_pstate()
    }

    fn pmc_default(&self) -> Vec<crate::profile::PmcCounter> {
        crate::profile::AmdCounter::all().into_iter().map(crate::profile::PmcCounter::Amd).collect()
    }

    fn synchronize(&self) -> Result<()> {
        OwnerCtx::synchronize(self)
    }

    fn finish_replay(&self) -> Result<()> {
        self.finish_session();
        Ok(())
    }
}
