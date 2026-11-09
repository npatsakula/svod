//! `KernargArena`: bump allocator for AMDGPU kernel-argument buffers.
//!
//! Sized at 16 MiB GTT-coherent. **One per device**, shared by every lane
//! (tinygrad allocates one `kernargs_buf` per `HCQCompiled`,
//! `support/hcq.py` `HCQCompiled.__init__`; four private 16 MiB arenas is
//! four times the resizable-BAR pressure for no gain). Each `Program::execute`
//! claims `kernarg_size` bytes (16-byte aligned per ABI) under the arena's
//! cursor lock, so concurrent lanes never share a slot. The arena wraps when it
//! fills; on wrap we drain every live `PoolQueue` via the arena's owning
//! `AmdDeviceCore` — without that drain a wrap can clobber kernargs the GPU is
//! still consuming (the host can sprint ahead on a `wait=false` burst). That
//! device-wide drain is exactly what makes one shared arena safe.

#![cfg(unix)]

use std::ptr::NonNull;
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::allocator::{AmdBufferGuard, RawBuffer};
use crate::amd::AmdAllocator;
use crate::amd::device::AmdDeviceCore;
use crate::error::{Error, Result};

const ARENA_BYTES: usize = 16 * 1024 * 1024;

pub struct KernargArena {
    pub base_gpu: u64,
    pub base_host: NonNull<u8>,
    pub size: usize,
    cursor: Mutex<usize>,
    /// Back-reference to the device core, used to drain every live pool queue
    /// on wrap (`AmdDeviceCore::synchronize_all`). `Weak` because the program
    /// that owns this arena also indirectly owns the core via `Arc<AmdDevice>`
    /// — a strong handle here would form a cycle.
    core: Weak<AmdDeviceCore>,
    _buffer: RawBuffer,
}

// SAFETY: cursor is Mutex-protected; pointer + buffer are stable for the
// arena's lifetime.
unsafe impl Send for KernargArena {}
unsafe impl Sync for KernargArena {}

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
            cursor: Mutex::new(0),
            core: Arc::downgrade(core),
            _buffer: buffer.into_inner(),
        }))
    }

    /// Reserve `size` bytes (aligned to `align`) and return the byte offset
    /// into the arena. Wraps to the beginning if `size` doesn't fit — and on
    /// wrap drains every live connector first so we don't overwrite kernargs
    /// the GPU is still reading.
    pub fn bump(&self, size: usize, align: usize) -> Result<usize> {
        if size > self.size {
            return Err(Error::AmdAllocFailed {
                reason: format!("kernarg request {size} exceeds arena {}", self.size),
            });
        }
        let mut cur = self.cursor.lock();
        let aligned = (*cur).next_multiple_of(align);
        if aligned + size <= self.size {
            *cur = aligned + size;
            return Ok(aligned);
        }
        // Wrap with the cursor held: a lane bumping during the drain finds
        // the arena already restarted and takes the next slot. Released, every
        // lane that met the full arena wrapped on its own and got offset 0.
        if let Some(core) = self.core.upgrade() {
            core.synchronize_all()?;
        }
        *cur = size;
        Ok(0)
    }

    pub fn gpu_at(&self, offset: usize) -> u64 {
        self.base_gpu + offset as u64
    }

    /// # Safety
    /// Caller must ensure `offset + size <= self.size` and that no concurrent
    /// writer holds the same slot. The exclusive lane lease held across bump +
    /// write + dispatch guarantees the only producer of a given slot
    /// is the caller of `bump`, and that the GPU reads it before it is reused.
    pub unsafe fn host_at(&self, offset: usize) -> *mut u8 {
        unsafe { self.base_host.as_ptr().add(offset) }
    }
}

impl std::fmt::Debug for KernargArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernargArena")
            .field("base_gpu", &format_args!("{:#x}", self.base_gpu))
            .field("size", &self.size)
            .field("cursor", &*self.cursor.lock())
            .finish_non_exhaustive()
    }
}
