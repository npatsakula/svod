//! `AmdSignal`: GTT-coherent timeline counter polled by the CPU.
//!
//! A `SignalPool` carves a single host-visible GTT page into 64-byte slots;
//! each [`AmdSignal`] owns a slot and exposes the `value_addr` GPU virtual
//! address so kernels / AQL barrier packets can write completion values.
//! CPU polling reads the same memory through the slot's `host_ptr`.

#![cfg(unix)]

use std::any::Any;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::allocator::{AmdBufferGuard, RawBuffer};
use crate::amd::AmdAllocator;
use crate::amd::device::AmdDeviceCore;
use crate::error::{Error, Result};
use crate::sync::TimelineSignal;

/// Spin / yield budget before escalating to a KFD WAIT_EVENTS sleep: cheap
/// polling for short waits, kernel blocking for long ones so a stalled wait
/// doesn't pin a CPU.
#[cfg(not(test))]
const WAIT_EVENTS_ESCALATE_MS: u64 = 200;
#[cfg(test)]
const WAIT_EVENTS_ESCALATE_MS: u64 = 1;

/// 64-byte slot laid out as an `amd_signal_t` (kind@0, value@8). The AQL packet
/// processor reads a dispatch packet's `completion_signal.handle` as the struct
/// base and atomically decrements the `value` field at offset 8; PM4 RELEASE_MEM
/// and SDMA fence writes target that same `value`. So every signal — compute
/// completion or copy timeline — shares this one layout.
const SLOT_BYTES: usize = 64;
/// 4 KiB page / 64 B per slot = 64 signals per page — the page-alignment quantum
/// the pool rounds its slot count up to.
const SLOTS_PER_PAGE: usize = 64;
/// Byte offset of the `value` counter inside an `amd_signal_t` slot.
const SIGNAL_VALUE_OFFSET: usize = 8;
/// Byte offsets of dispatch timestamps (`amd_signal_t.start_ts` / `.end_ts`).
/// AMD HCQ submission finalizers write these fields with explicit PM4 timestamp
/// commands on both PM4 and AQL queues.
const SIGNAL_START_TS_OFFSET: usize = 32;
const SIGNAL_END_TS_OFFSET: usize = 40;
/// The GPU clock counter feeding the timestamps ticks at the architected
/// 100 MHz on AMD GPUs (10 ns per tick — tinygrad's `timestamp_divider=100`).
const NS_PER_TICK: u64 = 10;

/// A pool-allocated AMD signal.
///
/// The atomic counter the GPU writes to lives at `value_addr` (GPU VA) and
/// is also reachable via `host_ptr` for CPU polling. `Drop` returns the
/// slot to its pool. The pool keeps the underlying VRAM allocation alive.
pub struct AmdSignal {
    slot: u32,
    /// `amd_signal_t` struct base (GPU VA), used to derive timestamp fields.
    base_gpu: u64,
    /// GPU VA of the `value` counter (`base_gpu + SIGNAL_VALUE_OFFSET`) — what
    /// PM4/SDMA packets write and what the host polls.
    value_addr: u64,
    host_ptr: NonNull<AtomicU64>,
    pool: Weak<SignalPool>,
    /// Owning device core — used to escalate long waits to
    /// `AMDKFD_IOC_WAIT_EVENTS` on the device's `queue_event`. `Weak` so
    /// signals don't extend device lifetime.
    device: Weak<AmdDeviceCore>,
}

// SAFETY: AtomicU64 covers all reads/writes; the host pointer comes from a
// shared mmap and is stable for the pool's lifetime.
unsafe impl Send for AmdSignal {}
unsafe impl Sync for AmdSignal {}

impl AmdSignal {
    /// GPU VA of the `value` counter — what PM4/SDMA wait/signal packets write
    /// and the host polls (`amd_signal_t.value`, at +8 from the struct base).
    pub fn value_addr(&self) -> u64 {
        self.value_addr
    }

    /// GPU VA of the dispatch `start_ts` field (`base_gpu + 32`). On the
    /// single-XCC PM4 path the CP does not auto-stamp dispatches (the AQL path's
    /// `ENABLE_PROFILING` does), so a profiling dispatch targets this address
    /// with a `release_mem_timestamp` GPU-clock probe before the kernel launches.
    #[inline]
    pub fn start_ts_addr(&self) -> u64 {
        self.base_gpu + SIGNAL_START_TS_OFFSET as u64
    }

    /// GPU VA of the dispatch `end_ts` field (`base_gpu + 40`). See
    /// [`start_ts_addr`](Self::start_ts_addr).
    #[inline]
    pub fn end_ts_addr(&self) -> u64 {
        self.base_gpu + SIGNAL_END_TS_OFFSET as u64
    }

    /// The owning device's latched fault, if any. Lets waiters that are not
    /// polling this slot (a submission still awaiting publication) bail on a
    /// dead device instead of burning their whole deadline.
    pub(crate) fn device_poison(&self) -> Option<Error> {
        self.device.upgrade().and_then(|device| device.poison_error())
    }

    /// Slot index inside the pool. Useful for debugging.
    pub fn slot(&self) -> u32 {
        self.slot
    }

    /// Current value (host read of the coherent slot).
    #[inline]
    fn load(&self) -> u64 {
        // SAFETY: NonNull valid for the pool's lifetime; AtomicU64 is race-free.
        unsafe { self.host_ptr.as_ref().load(Ordering::Acquire) }
    }

    /// Reset a slot before assigning it to a timeline or timestamp probe.
    /// Monotonic PM4/SDMA timelines start at zero and receive literal stores;
    /// stale profiling stamps are always cleared on reuse.
    #[inline]
    pub(crate) fn reset(&self, value: u64) {
        // SAFETY: the full 64-byte slot is mapped; ts fields at +32/+40.
        unsafe {
            let base = (self.host_ptr.as_ptr() as *mut u8).sub(SIGNAL_VALUE_OFFSET);
            std::ptr::write_volatile(base.add(SIGNAL_START_TS_OFFSET) as *mut u64, 0);
            std::ptr::write_volatile(base.add(SIGNAL_END_TS_OFFSET) as *mut u64, 0);
            self.host_ptr.as_ref().store(value, Ordering::Release);
        }
    }

    /// Tiered busy-wait until `ready(value)` holds, or `timeout_ms` of *no
    /// progress* elapses, or KFD reports a GPU fault. Shared by the
    /// monotonic timeline waits.
    ///
    /// Early-exit on fault is load-bearing for BEAM search: a bad kernel config
    /// may fault the GPU, and paying the full timeout per rejected candidate is
    /// unaffordable.
    fn poll_until(&self, target: u64, timeout_ms: u64, what: &'static str, progress: &[Arc<AmdSignal>]) -> Result<()> {
        let mut start = std::time::Instant::now();
        let mut prev = u64::MAX;
        let mut progress_values = vec![u64::MAX; progress.len()];
        loop {
            if let Some(error) = self.device.upgrade().and_then(|device| device.poison_error()) {
                return Err(error);
            }
            let v = self.load();
            if v >= target {
                return Ok(());
            }
            let mut advanced = v != prev;
            prev = v;
            for (signal, previous) in progress.iter().zip(&mut progress_values) {
                let value = signal.load();
                advanced |= value != *previous;
                *previous = value;
            }
            if advanced {
                start = std::time::Instant::now();
            }
            if timeout_ms > 0 && start.elapsed().as_millis() as u64 >= timeout_ms {
                // A hung kernel almost always raised a fault; surface it
                // alongside the deadline.
                let fault = self.device.upgrade().and_then(|d| d.poll_faults_nonblocking());
                return Err(fault.unwrap_or(Error::TimelineTimeout {
                    what,
                    target,
                    current: v,
                    waited_ms: timeout_ms,
                }));
            }
            if let Some(fault) = self.spin_or_escalate(start)? {
                return Err(fault);
            }
        }
    }

    /// Spin-wait until the value is ≥ `target` (increment convention — SDMA
    /// fence / monotonic timeline writes a literal increasing value).
    pub(crate) fn wait_signal_value(&self, target: u64, timeout_ms: u64) -> Result<()> {
        self.poll_until(target, timeout_ms, "wait_signal_value", &[])
    }

    pub(crate) fn wait_signal_value_with_progress(
        &self,
        target: u64,
        timeout_ms: u64,
        progress: &[Arc<AmdSignal>],
    ) -> Result<()> {
        self.poll_until(target, timeout_ms, "wait_signal_value", progress)
    }

    /// CP-written dispatch timestamps in nanoseconds, valid only after the
    /// signal retired. `None` until then, or when
    /// the slot was never targeted by timestamp commands (both stamps
    /// zeroed by `reset`).
    pub fn timestamps_ns(&self) -> Option<(u64, u64)> {
        // SAFETY: the full 64-byte slot is mapped; value lives at +8, so the
        // slot base is host_ptr − SIGNAL_VALUE_OFFSET.
        let (start, end) = unsafe {
            let base = (self.host_ptr.as_ptr() as *const u8).sub(SIGNAL_VALUE_OFFSET);
            (
                std::ptr::read_volatile(base.add(SIGNAL_START_TS_OFFSET) as *const u64),
                std::ptr::read_volatile(base.add(SIGNAL_END_TS_OFFSET) as *const u64),
            )
        };
        (start != 0 && end >= start).then(|| (start * NS_PER_TICK, end * NS_PER_TICK))
    }

    /// Tiered polling backoff: tight spin → `yield_now` → KFD `WAIT_EVENTS`
    /// once we've burned `WAIT_EVENTS_ESCALATE_MS` of wall time. The kernel
    /// wakes us when the device's `queue_event` fires, eliminating CPU burn
    /// for stalled or long-running dispatches.
    ///
    /// Returns the typed `WAIT_EVENTS` failure directly. Otherwise, `Some`
    /// carries a reported GPU fault and `None` means a normal wake-up, timeout,
    /// dropped device, or the pure spin/yield path.
    #[inline]
    fn spin_or_escalate(&self, start: std::time::Instant) -> Result<Option<Error>> {
        let elapsed_ms = start.elapsed().as_millis() as u64;
        if elapsed_ms >= WAIT_EVENTS_ESCALATE_MS
            && let Some(dev) = self.device.upgrade()
        {
            // Sleep in the kernel for at most another tier worth of time;
            // on return we re-check the host value and either complete
            // or escalate again. wait_events watches the three KFD
            // events (queue, mem fault, hw fault); a fault is returned
            // here so we bail with the actual error instead of grinding
            // through the rest of the timeout.
            match dev.wait_events(WAIT_EVENTS_ESCALATE_MS as u32) {
                Ok(Some(fault)) => return Ok(Some(fault)),
                Ok(None) => return Ok(None),
                Err(error) => return Err(error),
            }
        }
        std::hint::spin_loop();
        if start.elapsed().as_micros() >= 100 {
            std::thread::yield_now();
        }
        Ok(None)
    }
}

impl std::fmt::Debug for AmdSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdSignal")
            .field("slot", &self.slot)
            .field("value_addr", &format_args!("{:#x}", self.value_addr))
            .field("value", &self.value())
            .finish()
    }
}

impl Drop for AmdSignal {
    /// Tinygrad's `HCQSignal.__del__` (`support/hcq.py:250-251`) returns the
    /// slot unconditionally, and so does this — leaking a slot on every caught
    /// panic drains the pool for no benefit, and the slot is host memory.
    ///
    /// Documented divergence: a poisoned device may still have a wedged command
    /// processor writing this slot, so a poisoned device never recycles slots.
    fn drop(&mut self) {
        if self.device.upgrade().is_some_and(|device| device.is_poisoned()) {
            return;
        }
        if let Some(pool) = self.pool.upgrade() {
            pool.release_slot(self.slot);
        }
    }
}

/// Watermark for the 2^32 timeline wraparound. PM4 WAIT_REG_MEM/RELEASE_MEM
/// compare the low 32 bits of the signal slot, so the counter must stay below
/// 2^32; we drain + reset at 2^31 to keep headroom.
pub const TIMELINE_WRAP_WATERMARK: u64 = 1 << 31;

/// A connector's timeline: an owned monotonic counter plus the shared signal
/// the GPU writes on dispatch completion. This is the ONE primitive that
/// crosses owners — a `PoolQueue` dispatches against it (advancing `value`), and
/// any thread can *drain* it (read `value`, poll the signal slot) without taking
/// lane publication authority. The registry on `AmdDeviceCore` holds
/// `Weak<PoolQueue>`, and `drain_all` fences in-flight work purely through these
/// atomics plus retained linked-plan timelines, keeping concurrent dispatch unblocked.
#[derive(Debug)]
pub struct Timeline {
    signal: Arc<AmdSignal>,
    /// Highest reserved value + ... i.e. the next value `next()` hands out.
    /// Starts at 1; the value a dispatch SIGNALS is `next()`'s return.
    value: AtomicU64,
}

impl Timeline {
    pub fn new(signal: Arc<AmdSignal>) -> Arc<Self> {
        signal.reset(0);
        Arc::new(Self { signal, value: AtomicU64::new(1) })
    }

    /// The shared completion timeline (for emitting wait/signal packets).
    #[inline]
    pub fn signal(&self) -> &Arc<AmdSignal> {
        &self.signal
    }

    /// GPU VA of the signal counter — what PM4/AQL wait/signal packets target.
    #[inline]
    pub fn value_addr(&self) -> u64 {
        self.signal.value_addr()
    }

    /// Reserve the next timeline value (`fetch_add(1)`); the caller emits a
    /// signal packet writing this value on completion.
    #[inline]
    pub fn next(&self) -> u64 {
        self.value.fetch_add(1, Ordering::AcqRel)
    }

    /// Undo the most recent reservation before its doorbell was rung. Queue
    /// publication authority guarantees there can be no later reservation.
    pub(crate) fn rollback(&self, reserved: u64) -> bool {
        self.value.compare_exchange(reserved + 1, reserved, Ordering::AcqRel, Ordering::Acquire).is_ok()
    }

    /// Highest value reserved so far (the value the next `signal` packet writes
    /// is `current()`; the last reserved is `current() - 1`).
    #[inline]
    pub fn current(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }

    /// Block until the GPU has written the current `value - 1` snapshot. This
    /// never resets the generation because callers that do not hold the queue's
    /// publication lock can race a later reservation.
    pub fn drain(&self, timeout_ms: u64) -> Result<()> {
        let target = self.value.load(Ordering::Acquire).saturating_sub(1);
        if target == 0 {
            return Ok(());
        }
        self.signal.wait_signal_value(target, timeout_ms)?;
        Ok(())
    }

    /// Reset a drained generation. The caller must hold the same lock that
    /// serializes `next()` with queue publication and must have just drained.
    pub fn reset_after_drain(&self) {
        if self.value.load(Ordering::Acquire) > TIMELINE_WRAP_WATERMARK {
            debug_assert!(self.signal.value() >= self.value.load(Ordering::Acquire).saturating_sub(1));
            self.signal.reset(0);
            self.value.store(1, Ordering::Release);
        }
    }
}

impl crate::sync::DispatchTimestamps for AmdSignal {
    fn timestamps_ns(&self) -> Option<(u64, u64)> {
        AmdSignal::timestamps_ns(self)
    }
}

impl TimelineSignal for AmdSignal {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn value(&self) -> u64 {
        // SAFETY: NonNull is valid for the pool's lifetime; AtomicU64 reads
        // are race-free.
        unsafe { self.host_ptr.as_ref().load(Ordering::Acquire) }
    }

    fn set(&self, value: u64) {
        // SAFETY: same as `value`.
        unsafe { self.host_ptr.as_ref().store(value, Ordering::Release) };
    }

    fn wait(&self, target: u64, timeout_ms: u64) -> Result<()> {
        // Tiered strategy (spin → yield → KFD WAIT_EVENTS sleep) + fault
        // surfacing live in the shared `poll_until` helper.
        self.poll_until(target, timeout_ms, "wait", &[])
    }
}

/// One GTT-backed run of `chunk_slots` consecutive signal slots.
struct SignalChunk {
    buffer: RawBuffer,
    base_gpu: u64,
    base_host: NonNull<u8>,
}

#[derive(Default)]
struct PoolState {
    chunks: Vec<SignalChunk>,
    free_slots: Vec<u32>,
}

/// Pool over host-visible GTT chunks; hands out [`AmdSignal`]s. A flat
/// per-owner model needs only a handful of slots, but DAG dispatch reserves one
/// per kernel of the largest captured graph (low hundreds for GigaAM), so the
/// initial chunk spans several pages and exhaustion grows another chunk rather
/// than failing (tinygrad `HCQCompiled.new_signal`, `support/hcq.py:452-458`).
///
/// Slot ids are a flat numbering across chunks: chunk `i` owns
/// `i * chunk_slots .. (i + 1) * chunk_slots`, so a released slot needs no
/// chunk bookkeeping.
pub struct SignalPool {
    /// Used to carve additional chunks on exhaustion. Its `Arc<AmdDevice>` is
    /// the same one every chunk's `RawBuffer` already holds.
    allocator: AmdAllocator,
    chunk_slots: usize,
    state: Mutex<PoolState>,
    /// Captured at pool creation; signals downgrade-clone this into a `Weak`
    /// so `wait` can call `AmdDeviceCore::wait_events` for KFD escalation.
    device: Arc<AmdDeviceCore>,
}

// SAFETY: AtomicU64 covers concurrent reads/writes through a chunk's
// `base_host`; chunks and free slots are mutex-protected and not aliased.
unsafe impl Send for SignalPool {}
unsafe impl Sync for SignalPool {}

impl Drop for SignalPool {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        if let Err(error) = self.device.synchronize_all() {
            tracing::warn!(?error, "SignalPool drop: backing allocations quarantined");
            return;
        }
        for chunk in &self.state.get_mut().chunks {
            chunk.buffer.free_amd_device_in_place();
        }
    }
}

impl SignalPool {
    /// Allocate the first GTT chunk from `allocator` and partition it.
    ///
    /// Critical: the signal page must be **GTT-coherent + uncached** so that
    /// the GPU's decrement of the completion_signal field is immediately
    /// visible to the host (otherwise it sits in GPU L2 and we spin
    /// forever).
    pub fn new(allocator: &AmdAllocator, slots: usize) -> Result<Arc<Self>> {
        // Round up to a whole page so every GTT allocation is page-aligned and
        // every byte is usable as a slot.
        let pool = Arc::new(Self {
            allocator: AmdAllocator { dev: Arc::clone(&allocator.dev), device_id: allocator.device_id },
            chunk_slots: slots.max(1).next_multiple_of(SLOTS_PER_PAGE),
            state: Mutex::new(PoolState::default()),
            device: Arc::clone(allocator.dev.core()),
        });
        pool.grow(&mut pool.state.lock())?;
        Ok(pool)
    }

    /// Append one more chunk's worth of slots. The caller holds `state`.
    fn grow(&self, state: &mut PoolState) -> Result<()> {
        let buffer = AmdBufferGuard::new(
            self.allocator
                .alloc_uncached_tagged(SLOT_BYTES * self.chunk_slots, crate::amd::va_registry::AllocTag::SignalPool)?,
        );
        let (base_gpu, base_host) = match buffer.buffer() {
            RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
            _ => return Err(Error::NotHostVisible { what: "SignalPool" }),
        };
        let first = (state.chunks.len() * self.chunk_slots) as u32;
        state.chunks.push(SignalChunk { buffer: buffer.into_inner(), base_gpu, base_host });
        // Pop low slots first.
        state.free_slots.extend((first..first + self.chunk_slots as u32).rev());
        Ok(())
    }

    /// Carve off a new signal, growing the pool when every slot is in use.
    pub fn acquire(self: &Arc<Self>) -> Result<AmdSignal> {
        let mut state = self.state.lock();
        if state.free_slots.is_empty() {
            self.grow(&mut state)?;
        }
        let slot = state.free_slots.pop().expect("a grown pool has free slots");
        let chunk = &state.chunks[slot as usize / self.chunk_slots];
        let offset = (slot as usize % self.chunk_slots) * SLOT_BYTES;
        let base_gpu = chunk.base_gpu + offset as u64;
        // Lay out the amd_signal_t: zero the 64-byte slot, then set kind=USER so
        // the AQL packet processor treats it as a value signal, value stays 0.
        // SAFETY: offset + SLOT_BYTES <= the chunk size by construction.
        let slot_host = unsafe { chunk.base_host.as_ptr().add(offset) };
        unsafe {
            std::ptr::write_bytes(slot_host, 0, SLOT_BYTES);
            std::ptr::write_volatile(
                slot_host as *mut i64,
                crate::amd::sys::hsa::amd_signal_kind_t_AMD_SIGNAL_KIND_USER as i64,
            );
        }
        let value_addr = base_gpu + SIGNAL_VALUE_OFFSET as u64;
        // SAFETY: the 8-byte value field sits at +SIGNAL_VALUE_OFFSET in the slot.
        let host_ptr = unsafe { NonNull::new_unchecked(slot_host.add(SIGNAL_VALUE_OFFSET) as *mut AtomicU64) };
        unsafe { host_ptr.as_ref().store(0, Ordering::Release) };
        Ok(AmdSignal {
            slot,
            base_gpu,
            value_addr,
            host_ptr,
            pool: Arc::downgrade(self),
            device: Arc::downgrade(&self.device),
        })
    }

    fn release_slot(&self, slot: u32) {
        self.state.lock().free_slots.push(slot);
    }

    /// Currently-free slot count. Used by graph capture to decide whether a DAG
    /// reservation (one slot per kernel, held for the graph's life) would leave
    /// enough headroom for per-op AQL back-pressure + PM4 counters; if not,
    /// capture falls back to blanket-BARRIER instead of starving dispatch.
    pub fn free(&self) -> usize {
        self.state.lock().free_slots.len()
    }

    /// Slots carved so far, across every chunk.
    pub fn capacity(&self) -> usize {
        self.state.lock().chunks.len() * self.chunk_slots
    }
}

impl std::fmt::Debug for SignalPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock();
        f.debug_struct("SignalPool")
            .field("chunks", &state.chunks.len())
            .field("slots_total", &(state.chunks.len() * self.chunk_slots))
            .field("slots_free", &state.free_slots.len())
            .finish()
    }
}
