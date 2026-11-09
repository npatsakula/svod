//! `KernargArena`: ring allocator for AMDGPU kernel-argument buffers.
//!
//! Sized at 16 MiB GTT-coherent. **One per device**, shared by every lane
//! (tinygrad allocates one `kernargs_buf` per `HCQCompiled`,
//! `support/hcq.py` `HCQCompiled.__init__`; four private 16 MiB arenas is
//! four times the resizable-BAR pressure for no gain). On AQL queues it also
//! holds each dispatch's PM4 control program.
//!
//! A submission reserves everything it places here as one [`KernargSlot`],
//! writes it, and publishes the slot with the lane timeline point that retires
//! the submission. Slots are handed out in ring order and reclaimed oldest
//! first, so a reservation that meets the oldest live slot waits for exactly
//! that slot: for its publication while its lane is still writing it, then for
//! its timeline point. Neither a slot that is reserved but not yet submitted
//! nor one the GPU may still read is handed out again, and no other lane's
//! work is waited on.

#![cfg(unix)]

use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::allocator::{AmdBufferGuard, RawBuffer};
use crate::amd::AmdAllocator;
use crate::amd::device::AmdDeviceCore;
use crate::amd::signal::AmdSignal;
use crate::error::{Error, Result};
use crate::sync::TimelineSignal;

const ARENA_BYTES: usize = 16 * 1024 * 1024;
/// Bound on one reservation's wait for the slots it reuses (tinygrad's
/// `HCQDEV_WAIT_TIMEOUT_MS`).
#[cfg(not(test))]
const RETIRE_TIMEOUT_MS: u64 = 30_000;
#[cfg(test)]
const RETIRE_TIMEOUT_MS: u64 = 2_000;

pub struct KernargArena {
    pub base_gpu: u64,
    pub base_host: NonNull<u8>,
    pub size: usize,
    ring: Mutex<Ring>,
    /// Notified whenever a pending slot is published or abandoned.
    settled: Condvar,
    /// Poisoned when a wait on the GPU fails. `Weak` because the program that
    /// owns this arena also indirectly owns the core via `Arc<AmdDevice>`.
    core: Weak<AmdDeviceCore>,
    _buffer: RawBuffer,
}

// SAFETY: the ring is Mutex-protected and a reserved range is written only
// through its unique `KernargSlot`; pointer + buffer are stable for the
// arena's lifetime.
unsafe impl Send for KernargArena {}
unsafe impl Sync for KernargArena {}

/// When a live slot may be reused.
#[derive(Debug)]
enum Retire {
    /// Reserved, not yet submitted: its lane is still writing it.
    Pending,
    /// Submitted: free once the lane's timeline reaches the value.
    Timeline(Arc<AmdSignal>, u64),
    /// Abandoned before submission, or its lane's timeline drained.
    Done,
}

#[derive(Debug)]
struct Live {
    start: usize,
    retire: Retire,
}

/// Live slots in reservation order, which is ring order. `head` is the end of
/// the newest one, so with any slot live `head != oldest.start`: either the
/// live slots run linearly from the oldest to `head`, or they wrapped and
/// `head` sits below the oldest.
#[derive(Debug, Default)]
struct Ring {
    head: usize,
    live: VecDeque<Live>,
}

impl Ring {
    fn reclaim(&mut self) {
        while let Some(oldest) = self.live.front() {
            match &oldest.retire {
                Retire::Done => {}
                Retire::Timeline(signal, value) if signal.value() >= *value => {}
                _ => return,
            }
            self.live.pop_front();
        }
    }

    /// Where `size` bytes fit ahead of `head` without reaching the oldest live
    /// slot, wrapping to the arena start when the tail is too short.
    fn fit(&self, size: usize, align: usize, capacity: usize) -> Option<usize> {
        let start = self.head.next_multiple_of(align);
        let fits_tail = start + size <= capacity;
        match self.live.front().map(|oldest| oldest.start) {
            None => Some(if fits_tail { start } else { 0 }),
            Some(oldest) if self.head > oldest => {
                if fits_tail {
                    Some(start)
                } else {
                    (size < oldest).then_some(0)
                }
            }
            Some(oldest) => (start + size < oldest).then_some(start),
        }
    }
}

impl Drop for KernargArena {
    /// Free the 16 MiB CPU-visible VRAM backing. `RawBuffer` lacks a `Drop` (the
    /// allocator path consumes it by destructure), so the arena would otherwise
    /// leak its allocation. This runs when the LAST `PoolQueue` sharing the
    /// arena drops, after that queue's `Drop` has drained it; the core keeps
    /// only a `Weak`, so the arena does not outlive the lanes that use it.
    fn drop(&mut self) {
        self._buffer.free_amd_device_in_place();
    }
}

impl KernargArena {
    pub fn new(allocator: &AmdAllocator, core: &Arc<AmdDeviceCore>) -> Result<Arc<Self>> {
        // Tinygrad keeps both the ordinary kernarg arena and graph-owned
        // kernargs in CPU-visible VRAM. Queue publication performs the required
        // store fence before ringing the doorbell.
        let buffer = AmdBufferGuard::new(
            allocator.alloc_host_visible_tagged(ARENA_BYTES, crate::amd::va_registry::AllocTag::Kernarg)?,
        );
        let (base_gpu, base_host) = match buffer.buffer() {
            RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
            _ => return Err(Error::NotHostVisible { what: "kernarg arena" }),
        };
        Ok(Arc::new(Self {
            base_gpu,
            base_host,
            size: ARENA_BYTES,
            ring: Mutex::default(),
            settled: Condvar::new(),
            core: Arc::downgrade(core),
            _buffer: buffer.into_inner(),
        }))
    }

    /// Reserve `size` bytes aligned to `align`, waiting for the oldest live
    /// slots in the way to retire. The wait on a submitted slot keeps the ring
    /// locked, since no other reservation can pass that slot either; the wait
    /// on a pending one releases it so the slot's lane can publish.
    ///
    /// A lane publishes or drops its slot before it reserves again: its own
    /// pending slot would otherwise be what its next reservation waits for.
    pub fn reserve(&self, size: usize, align: usize) -> Result<KernargSlot<'_>> {
        if size > self.size {
            return Err(Error::AmdAllocFailed {
                reason: format!("kernarg request {size} exceeds arena {}", self.size),
            });
        }
        // An empty request still takes a byte, so every live slot has an extent.
        let extent = size.max(1);
        let deadline = Instant::now() + Duration::from_millis(RETIRE_TIMEOUT_MS);
        let mut ring = self.ring.lock();
        loop {
            ring.reclaim();
            if let Some(start) = ring.fit(extent, align, self.size) {
                ring.head = start + extent;
                ring.live.push_back(Live { start, retire: Retire::Pending });
                return Ok(KernargSlot { arena: self, offset: start, len: size, published: false });
            }
            let core = self.core.upgrade();
            if let Some(error) = core.as_ref().and_then(|core| core.poison_error()) {
                return Err(error);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            let oldest = ring.live.front().expect("an arena that cannot fit a request holds a live slot");
            match &oldest.retire {
                Retire::Timeline(signal, value) => {
                    let remaining_ms = (remaining.as_millis() as u64).max(1);
                    signal.wait_signal_value(*value, remaining_ms).inspect_err(|error| {
                        if let Some(core) = &core {
                            core.poison(&error.to_string());
                        }
                    })?;
                }
                Retire::Pending if remaining.is_zero() => {
                    return Err(Error::TimelineTimeout {
                        what: "AMD kernarg slot publication",
                        target: oldest.start as u64,
                        current: ring.head as u64,
                        waited_ms: RETIRE_TIMEOUT_MS,
                    });
                }
                Retire::Pending => {
                    self.settled.wait_for(&mut ring, remaining);
                }
                Retire::Done => unreachable!("reclaim pops retired slots off the front"),
            }
        }
    }

    /// Mark every slot submitted on `signal` retired. A lane calls this once
    /// drained, before it resets its timeline: the reset lowers the signal
    /// below the values its slots already reached.
    pub(crate) fn retire_timeline(&self, signal: &Arc<AmdSignal>) {
        for slot in &mut self.ring.lock().live {
            if matches!(&slot.retire, Retire::Timeline(owner, _) if Arc::ptr_eq(owner, signal)) {
                slot.retire = Retire::Done;
            }
        }
    }

    fn settle(&self, offset: usize, retire: Retire) {
        let mut ring = self.ring.lock();
        let slot = ring.live.iter_mut().rev().find(|slot| slot.start == offset);
        slot.expect("a reserved slot stays live until it settles").retire = retire;
        self.settled.notify_all();
    }

    /// Live slots as `(offset, state)`, oldest first.
    #[cfg(test)]
    pub(crate) fn live_slots(&self) -> Vec<(usize, &'static str)> {
        let state = |retire: &Retire| match retire {
            Retire::Pending => "pending",
            Retire::Timeline(..) => "submitted",
            Retire::Done => "done",
        };
        self.ring.lock().live.iter().map(|slot| (slot.start, state(&slot.retire))).collect()
    }
}

impl std::fmt::Debug for KernargArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ring = self.ring.lock();
        f.debug_struct("KernargArena")
            .field("base_gpu", &format_args!("{:#x}", self.base_gpu))
            .field("size", &self.size)
            .field("head", &ring.head)
            .field("live", &ring.live.len())
            .finish_non_exhaustive()
    }
}

/// An exclusively owned arena range. Dropped unpublished, it is abandoned and
/// reusable at once: the GPU never saw it.
#[must_use = "a slot is published with the submission that reads it"]
pub struct KernargSlot<'a> {
    arena: &'a KernargArena,
    offset: usize,
    len: usize,
    published: bool,
}

impl KernargSlot<'_> {
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn gpu(&self) -> u64 {
        self.arena.base_gpu + self.offset as u64
    }

    pub fn host(&mut self) -> &mut [u8] {
        // SAFETY: the ring hands this range to no one else while the slot is
        // live, and the GPU reads it only once it is published.
        unsafe { std::slice::from_raw_parts_mut(self.arena.base_host.as_ptr().add(self.offset), self.len) }
    }

    /// Hand the slot to the submission that signals `value` on `signal`, as
    /// soon as its doorbell has rung.
    pub fn publish(mut self, signal: &Arc<AmdSignal>, value: u64) {
        self.published = true;
        self.arena.settle(self.offset, Retire::Timeline(Arc::clone(signal), value));
    }
}

impl Drop for KernargSlot<'_> {
    fn drop(&mut self) {
        if !self.published {
            self.arena.settle(self.offset, Retire::Done);
        }
    }
}
