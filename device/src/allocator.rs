use std::alloc::Layout;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Mutex;

use crate::error::*;

/// 64-byte aligned buffer for SIMD operations (covers SSE/AVX/AVX-512).
///
/// The C codegen emits vector types with alignment attributes (e.g. `aligned(32)` for
/// `double4`). Clang then generates aligned load/store instructions (`vmovaps`) that
/// segfault on unaligned pointers. This buffer guarantees all allocations are
/// 64-byte aligned to satisfy any current SIMD width.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
}

const BUFFER_ALIGN: usize = 64;

impl AlignedBuffer {
    pub fn new_zeroed(size: usize) -> Self {
        if size == 0 {
            return Self { ptr: NonNull::dangling(), len: 0 };
        }
        let layout = Layout::from_size_align(size, BUFFER_ALIGN).expect("invalid buffer layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, len: size }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Deref for AlignedBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        if self.len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) } }
    }
}

impl DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        if self.len == 0 { &mut [] } else { unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) } }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.len > 0 {
            let layout = Layout::from_size_align(self.len, BUFFER_ALIGN).unwrap();
            unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

/// Opaque handle to device memory.
///
/// # Safety
///
/// `RawBuffer` uses `UnsafeCell` for interior mutability without locking overhead.
/// Thread safety is guaranteed at a higher level by the scheduler:
///
/// 1. **Allocation**: `OnceLock` in `BufferData` ensures single initialization
/// 2. **Buffer Access**: The scheduler guarantees exclusive access to each buffer
///    during kernel execution - no two kernels access the same buffer concurrently
/// 3. **Kernel Execution**: Raw pointers passed to JIT code; Rust doesn't access
///    buffer data during execution
///
/// Buffer synchronization is the scheduler's responsibility, not the buffer's.
pub enum RawBuffer {
    Cpu {
        data: UnsafeCell<AlignedBuffer>,
        cpu_accessible: bool,
    },
    /// Memory-mapped file region (read-only). Used by DISK device.
    Mmap {
        data: memmap2::Mmap,
        size: usize,
    },
    /// AMD GPU VRAM/GTT buffer allocated via KFD ioctls.
    ///
    /// `gpu_addr` is the GPU virtual address that kernels see in their
    /// kernarg slot. `host_ptr` is `Some(_)` only when `cpu_accessible`; the
    /// pointer is a host-side mmap of the same buffer, suitable for memcpy.
    /// `handle` is KFD's opaque allocation handle, used for the matching
    /// free/unmap ioctls. `device` keeps the underlying KFD/DRM fds alive
    /// for the lifetime of the buffer.
    AmdDevice {
        gpu_addr: u64,
        host_ptr: Option<std::ptr::NonNull<u8>>,
        size: usize,
        handle: u64,
        device: std::sync::Arc<crate::amd::AmdDevice>,
    },
    /// Metal buffer with `MTLResourceStorageModeShared` storage.
    ///
    /// `contents` is the host mapping of the same unified-memory allocation:
    /// what the CPU memcpys through and, plus the view offset, what
    /// `Buffer::as_raw_ptr` hands to `Program::execute`, which resolves it back
    /// to `(MTLBuffer, offset)` through the device's pointer registry.
    Metal {
        buffer: crate::metal::objc::ObjcId,
        contents: std::ptr::NonNull<u8>,
        size: usize,
        device: std::sync::Arc<crate::metal::MetalDevice>,
    },
    /// CUDA allocation. `device_ptr` is what kernels receive in their kernarg
    /// slot; `host_ptr` is `Some` for managed and pinned memory (`memory`
    /// says which), the CPU side of the same allocation.
    Cuda {
        device_ptr: u64,
        host_ptr: Option<std::ptr::NonNull<u8>>,
        size: usize,
        memory: crate::cuda::CudaMemory,
        device: std::sync::Arc<crate::cuda::CudaDevice>,
    },
}

// SAFETY: RawBuffer access is synchronized by the scheduler at a higher level.
// See RawBuffer documentation for detailed safety invariants.
unsafe impl Send for RawBuffer {}
unsafe impl Sync for RawBuffer {}

impl RawBuffer {
    /// Free this buffer's GPU-side backing if it's an AMD device buffer.
    ///
    /// `AmdAllocator::_free` consumes the buffer via destructure; for
    /// containers that hold `RawBuffer` directly without going through the
    /// allocator (queue rings / GART / EOP / ctx-save / pm4_ibs and kernarg
    /// arenas), this method is the cleanup hook. Internal to the crate so
    /// only owners that know their resource is AMD-device-backed call it.
    pub(crate) fn free_amd_device_in_place(&self) {
        if let RawBuffer::AmdDevice { gpu_addr, size, handle, device, .. } = self {
            if std::thread::panicking() {
                tracing::warn!(gpu_addr, size, "quarantining AMD allocation during panic unwind");
                return;
            }
            if device.core().poison_error().is_some() {
                tracing::warn!(gpu_addr, size, "quarantining AMD allocation referenced by a poisoned device");
                return;
            }
            device.core().iface().free_raw(*gpu_addr, *size, *handle);
        }
    }
}

/// Construction guard for an AMD allocation before ownership is transferred to
/// a long-lived object. Ordinary failures reclaim it; poisoned devices retain
/// the mapping through `free_amd_device_in_place`'s quarantine policy.
pub(crate) struct AmdBufferGuard(Option<RawBuffer>);

impl AmdBufferGuard {
    pub(crate) fn new(buffer: RawBuffer) -> Self {
        Self(Some(buffer))
    }

    pub(crate) fn buffer(&self) -> &RawBuffer {
        self.0.as_ref().expect("AMD buffer guard already disarmed")
    }

    pub(crate) fn into_inner(mut self) -> RawBuffer {
        self.0.take().expect("AMD buffer guard already disarmed")
    }
}

impl Drop for AmdBufferGuard {
    fn drop(&mut self) {
        if let Some(buffer) = self.0.as_ref() {
            buffer.free_amd_device_in_place();
        }
    }
}

// UnsafeCell doesn't implement Debug, so we implement it manually
impl std::fmt::Debug for RawBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RawBuffer::Cpu { cpu_accessible, .. } => {
                f.debug_struct("Cpu").field("cpu_accessible", cpu_accessible).finish_non_exhaustive()
            }
            RawBuffer::Mmap { size, .. } => f.debug_struct("Mmap").field("size", size).finish_non_exhaustive(),
            RawBuffer::AmdDevice { gpu_addr, size, host_ptr, .. } => f
                .debug_struct("AmdDevice")
                .field("gpu_addr", gpu_addr)
                .field("size", size)
                .field("cpu_accessible", &host_ptr.is_some())
                .finish_non_exhaustive(),
            RawBuffer::Metal { contents, size, .. } => {
                f.debug_struct("Metal").field("contents", contents).field("size", size).finish_non_exhaustive()
            }
            RawBuffer::Cuda { device_ptr, size, memory, .. } => f
                .debug_struct("Cuda")
                .field("device_ptr", &format_args!("{device_ptr:#x}"))
                .field("size", size)
                .field("memory", memory)
                .finish_non_exhaustive(),
        }
    }
}

impl RawBuffer {
    /// Get the size of the buffer in bytes.
    pub fn size(&self) -> usize {
        // SAFETY: Reading .len() doesn't alias with content access and is immutable after allocation
        match self {
            RawBuffer::Cpu { data, .. } => unsafe { (&*data.get()).len() },
            RawBuffer::Mmap { size, .. } => *size,
            RawBuffer::AmdDevice { size, .. } => *size,
            RawBuffer::Metal { size, .. } => *size,
            RawBuffer::Cuda { size, .. } => *size,
        }
    }

    /// Get whether this buffer is CPU-accessible.
    pub fn cpu_accessible(&self) -> bool {
        match self {
            RawBuffer::Cpu { cpu_accessible, .. } => *cpu_accessible,
            RawBuffer::Mmap { .. } => true,
            RawBuffer::AmdDevice { host_ptr, .. } => host_ptr.is_some(),
            RawBuffer::Metal { .. } => true,
            RawBuffer::Cuda { host_ptr, .. } => host_ptr.is_some(),
        }
    }
}

/// Buffer allocation spec. It is the *whole* LRU cache key `(size, spec)`,
/// hence `Hash + Eq + Copy`.
///
/// `zero_init` is intentionally NOT a field — the backend allocator never
/// zeroes (`_alloc` returns raw memory); Svod threads it as a separate `alloc`
/// argument so it does not split the cache. A zeroed and a non-zeroed buffer of
/// the same spec are interchangeable, because a cache hit re-zeroes on demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "proptest", derive(proptest_derive::Arbitrary))]
pub struct BufferSpec {
    /// GTT-coherent uncached memory (signal/ring/kernarg). Distinct cache type
    /// from VRAM — can't be reused as cached.
    pub uncached: bool,
    /// CPU-accessible mapping.
    ///
    /// CPU allocator: always honored (host memory is always accessible).
    /// AMD allocator: adds a host BAR mmap (`host_ptr: Some`).
    /// Metal allocator: always honored (shared storage mode).
    pub cpu_access: bool,
    /// Host (GTT/userptr) memory rather than device VRAM.
    pub host: bool,
    /// Never cache this buffer in the LRU pool: free goes straight to teardown.
    /// For lifetime-bound buffers (code object, scratch, queue/signal infra).
    pub nolru: bool,
}

impl Default for BufferSpec {
    fn default() -> Self {
        Self { uncached: false, cpu_access: true, host: false, nolru: false }
    }
}

/// Device memory allocator: the public `alloc`/`free` are thin wrappers over
/// the runtime-implemented `_alloc`/`_free`; copy/transfer/offset/map are
/// overridable hooks that default to "unsupported".
///
/// Object-safe (used as `Arc<dyn Allocator>`): the opaque is the single
/// [`RawBuffer`] enum, so copy hooks take `&RawBuffer` + an explicit byte
/// offset (the view offset lives on [`crate::Buffer`], not on `RawBuffer`).
pub trait Allocator: Send + Sync + std::fmt::Debug {
    /// Allocate `size` bytes. `zero` requests zero-initialized memory (a Svod
    /// extension applied on top of `_alloc`, not part of the cache key).
    fn alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> Result<RawBuffer> {
        self._alloc(size, options, zero)
    }

    /// Free a buffer. `size` is the originally-requested allocation size (the
    /// LRU cache key); the base allocator ignores it and just releases the
    /// handle. The `RawBuffer` is consumed (and dropped) here.
    fn free(&self, buffer: RawBuffer, size: usize, options: &BufferSpec) {
        let _ = size;
        self._free(buffer, options);
    }

    /// Backend allocation.
    fn _alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> Result<RawBuffer>;

    /// Backend free. Default drops the `RawBuffer` (CPU/host memory frees via
    /// `Drop`); device backends override to release driver handles.
    fn _free(&self, _buffer: RawBuffer, _options: &BufferSpec) {}

    /// Copy host bytes into `dest[dest_off..dest_off+src.len()]`.
    fn _copyin(&self, _dest: &RawBuffer, _dest_off: usize, _src: &[u8]) -> Result<()> {
        UnsupportedSnafu { op: "copyin" }.fail()
    }

    /// Copy `src[src_off..src_off+dest.len()]` out into host bytes.
    fn _copyout(&self, _dest: &mut [u8], _src: &RawBuffer, _src_off: usize) -> Result<()> {
        UnsupportedSnafu { op: "copyout" }.fail()
    }

    /// Same-device copy of `sz` bytes.
    fn _transfer(
        &self,
        _dest: &RawBuffer,
        _dest_off: usize,
        _src: &RawBuffer,
        _src_off: usize,
        _sz: usize,
    ) -> Result<()> {
        UnsupportedSnafu { op: "transfer" }.fail()
    }

    /// Mint a sub-buffer view (for cross-device base views).
    fn _offset(&self, _buf: &RawBuffer, _size: usize, _offset: usize) -> Result<RawBuffer> {
        UnsupportedSnafu { op: "offset" }.fail()
    }

    /// Map a foreign buffer into this device's address space.
    fn _map(&self, _buf: &RawBuffer) -> Result<RawBuffer> {
        UnsupportedSnafu { op: "map" }.fail()
    }

    /// Unmap a previously mapped buffer.
    fn _unmap(&self, _mb: &RawBuffer) {}

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
    fn name(&self) -> &str;

    /// Get the device specification for this allocator.
    fn device_spec(&self) -> svod_dtype::DeviceSpec;

    /// Whether this allocator can keep intermediate buffers device-local (no
    /// host mapping), so the scheduler should allocate non-output intermediates
    /// with `cpu_access: false`. Defaults to `false` (host-visible everywhere);
    /// backends with a device→device + host↔device copy path (e.g. AMD via the
    /// SDMA copy queue) override it. Decorators forward to their inner allocator.
    fn supports_device_local(&self) -> bool {
        false
    }
}

/// CPU allocator using system memory.
#[derive(Debug, Clone)]
pub struct CpuAllocator;

impl Allocator for CpuAllocator {
    fn _alloc(&self, size: usize, options: &BufferSpec, _zero: bool) -> Result<RawBuffer> {
        // `AlignedBuffer::new_zeroed` always zeroes, so `_zero` is implicitly
        // satisfied on CPU regardless of the flag.
        let data = AlignedBuffer::new_zeroed(size);
        Ok(RawBuffer::Cpu { data: UnsafeCell::new(data), cpu_accessible: options.cpu_access })
    }

    fn _copyin(&self, dest: &RawBuffer, dest_off: usize, src: &[u8]) -> Result<()> {
        match dest {
            RawBuffer::Cpu { data, .. } => {
                // SAFETY: scheduler guarantees exclusive access during buffer ops.
                let buf = unsafe { &mut *data.get() };
                buf[dest_off..dest_off + src.len()].copy_from_slice(src);
                Ok(())
            }
            other => unreachable!("CpuAllocator::_copyin on non-CPU buffer: {other:?}"),
        }
    }

    fn _copyout(&self, dest: &mut [u8], src: &RawBuffer, src_off: usize) -> Result<()> {
        match src {
            RawBuffer::Cpu { data, .. } => {
                // SAFETY: scheduler guarantees no concurrent writes during buffer ops.
                let buf = unsafe { &*data.get() };
                dest.copy_from_slice(&buf[src_off..src_off + dest.len()]);
                Ok(())
            }
            other => unreachable!("CpuAllocator::_copyout on non-CPU buffer: {other:?}"),
        }
    }

    fn _transfer(&self, dest: &RawBuffer, dest_off: usize, src: &RawBuffer, src_off: usize, sz: usize) -> Result<()> {
        match (dest, src) {
            (RawBuffer::Cpu { data: dst, .. }, RawBuffer::Cpu { data: src, .. }) => {
                if std::ptr::eq(dst, src) {
                    // Avoid creating aliased references when two buffer
                    // handles share the same allocation.
                    let buf = unsafe { &mut *dst.get() };
                    buf.copy_within(src_off..src_off + sz, dest_off);
                    return Ok(());
                }
                // SAFETY: distinct allocations; scheduler guarantees exclusivity.
                let dst_buf = unsafe { &mut *dst.get() };
                let src_buf = unsafe { &*src.get() };
                dst_buf[dest_off..dest_off + sz].copy_from_slice(&src_buf[src_off..src_off + sz]);
                Ok(())
            }
            _ => UnsupportedSnafu { op: "transfer" }.fail(),
        }
    }

    fn name(&self) -> &str {
        "CPU"
    }

    fn device_spec(&self) -> svod_dtype::DeviceSpec {
        svod_dtype::DeviceSpec::Cpu
    }
}

/// DISK allocator using memory-mapped files.
/// Read-only — cannot execute kernels. Data is transferred via COPY.
#[derive(Debug, Clone)]
pub struct DiskAllocator {
    path: std::path::PathBuf,
}

impl DiskAllocator {
    pub fn new(path: std::path::PathBuf) -> Self {
        Self { path }
    }
}

impl Allocator for DiskAllocator {
    fn _alloc(&self, size: usize, _options: &BufferSpec, _zero: bool) -> Result<RawBuffer> {
        let file = std::fs::File::open(&self.path).map_err(|e| crate::Error::CopyFailed {
            reason: format!("DISK: failed to open {}: {e}", self.path.display()),
        })?;
        let file_size = file
            .metadata()
            .map_err(|e| crate::Error::CopyFailed {
                reason: format!("DISK: failed to read metadata for {}: {e}", self.path.display()),
            })?
            .len() as usize;
        if size > file_size {
            return Err(crate::Error::CopyFailed {
                reason: format!("DISK: requested {size} bytes but {} is only {file_size} bytes", self.path.display()),
            });
        }
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| crate::Error::CopyFailed {
            reason: format!("DISK: mmap failed for {}: {e}", self.path.display()),
        })?;
        Ok(RawBuffer::Mmap { data: mmap, size })
    }

    fn _copyout(&self, dest: &mut [u8], src: &RawBuffer, src_off: usize) -> Result<()> {
        match src {
            RawBuffer::Mmap { data, .. } => {
                dest.copy_from_slice(&data[src_off..src_off + dest.len()]);
                Ok(())
            }
            other => unreachable!("DiskAllocator::_copyout on non-Mmap buffer: {other:?}"),
        }
    }

    fn _copyin(&self, _dest: &RawBuffer, _dest_off: usize, _src: &[u8]) -> Result<()> {
        // DISK is read-only: never write through the mmap.
        Err(crate::Error::CopyFailed { reason: "DISK device is read-only: copyin not supported".into() })
    }

    fn name(&self) -> &str {
        "DISK"
    }

    fn device_spec(&self) -> svod_dtype::DeviceSpec {
        svod_dtype::DeviceSpec::Disk { path: self.path.clone() }
    }
}

/// LRU allocator that caches freed buffers for reuse:
///
/// - the cache is keyed on the whole `(size, BufferSpec)`;
/// - `free` recycles into the pool *without synchronizing* — the
///   timeline-drain-before-teardown lives in the backend `_free` (e.g.
///   `AmdAllocator::_free`), reached only on real release (overflow, `nolru`,
///   or `free_cache`);
/// - on allocation failure `free_cache` releases every pooled buffer through
///   the backend `_free` and the alloc is retried.
///
/// The cache key uses the *requested* `size` for both `alloc` and `free` (the
/// `size` arg to `free`), so a backend that rounds up its actual allocation
/// (e.g. AMD page-rounding) still reuses buffers — unlike keying on the
/// buffer's rounded size, which would never match the request.
#[derive(Debug)]
pub(crate) struct LruAllocator {
    inner: Box<dyn Allocator>,
    cache: Mutex<HashMap<(usize, BufferSpec), Vec<RawBuffer>>>,
    max_buffers_per_size: usize,
    name: String,
}

impl LruAllocator {
    pub fn new(inner: Box<dyn Allocator>) -> Self {
        Self::with_capacity(inner, 32)
    }

    pub fn with_capacity(inner: Box<dyn Allocator>, max_buffers_per_size: usize) -> Self {
        let name = inner.name().to_string();
        Self { inner, cache: Mutex::new(HashMap::new()), max_buffers_per_size, name }
    }

    /// Release every pooled buffer through the backend `_free`.
    /// Routing through `inner.free` is essential: `RawBuffer` has no `Drop`, so
    /// merely clearing the map would leak GPU mappings.
    fn free_cache(&self) {
        let drained: Vec<((usize, BufferSpec), Vec<RawBuffer>)> = {
            let mut cache = self.cache.lock().unwrap();
            cache.drain().collect()
        };
        for ((size, options), buffers) in drained {
            for buf in buffers {
                self.inner.free(buf, size, &options);
            }
        }
    }

    /// Get the number of cached buffers for a specific size and cpu_access flag.
    /// Only available in tests for cache introspection.
    #[cfg(test)]
    pub(crate) fn cache_count(&self, size: usize, cpu_access: bool) -> usize {
        let key = (size, BufferSpec { cpu_access, ..Default::default() });
        let cache = self.cache.lock().unwrap();
        cache.get(&key).map(|v| v.len()).unwrap_or(0)
    }

    /// Get the total number of cached buffers across all keys.
    /// Only available in tests for cache introspection.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn total_cached(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        cache.values().map(|v| v.len()).sum()
    }

    /// Re-zero a buffer popped from the cache. Returns `Err` to signal "drop
    /// this buffer and allocate fresh instead" (device-only AMD VRAM, where a
    /// host memset is impossible until SDMA lands).
    fn zero_cached(&self, buffer: &RawBuffer) -> Result<bool> {
        // SAFETY: buffer just popped from cache — no other references exist.
        match buffer {
            RawBuffer::Cpu { data, .. } => {
                unsafe { (*data.get()).fill(0) };
                Ok(true)
            }
            // DISK is read-only and never LRU-cached (the registry hands out a
            // bare DiskAllocator), so this arm is unreachable in practice; if a
            // recycle ever routed here, a host memset through the read-only mmap
            // is impossible — surface it instead of panicking.
            RawBuffer::Mmap { .. } => UnsupportedSnafu { op: "zero-init read-only DISK mmap" }.fail(),
            RawBuffer::AmdDevice { host_ptr: Some(ptr), size, gpu_addr, device, .. } => {
                // This VA was just recycled from the pool: wait its recorded
                // producers (usually already retired via the pop fence) before
                // the host memset, which isn't ordered on the GPU timeline.
                device.core().wait_storage(*gpu_addr)?;
                unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, *size) };
                Ok(true)
            }
            RawBuffer::AmdDevice { host_ptr: None, .. } => Ok(false),
            RawBuffer::Metal { contents, size, device, .. } => {
                // A recycled buffer may still be read by in-flight kernels; the
                // host memset is not ordered on the GPU timeline.
                device.synchronize()?;
                unsafe { std::ptr::write_bytes(contents.as_ptr(), 0, *size) };
                Ok(true)
            }
            // A device-side memset on the copy lane, ordered after the
            // storage's in-flight producers; works for every CUDA memory kind.
            RawBuffer::Cuda { device_ptr, size, device, .. } => {
                device.zero(*device_ptr, (*size).max(1))?;
                Ok(true)
            }
        }
    }
}

impl Drop for LruAllocator {
    fn drop(&mut self) {
        self.free_cache();
    }
}

impl Allocator for LruAllocator {
    fn alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> Result<RawBuffer> {
        // nolru never pools: deterministic free.
        if options.nolru {
            return self.inner.alloc(size, options, zero);
        }
        let key = (size, *options);

        // Pop from the per-key pool if present.
        let buffer = {
            let mut cache = self.cache.lock().unwrap();
            if let Some(buffers) = cache.get_mut(&key)
                && let Some(buffer) = buffers.pop()
            {
                if buffers.is_empty() {
                    cache.remove(&key);
                }
                Some(buffer)
            } else {
                None
            }
        }; // Drop lock before any (re)allocation.

        if let Some(buffer) = buffer {
            // A recycled VA may still be referenced by the previous owner's
            // in-flight kernels (`free` never drains). Fence on the storage's
            // recorded producers before handing it out — nearly free once
            // everything has retired.
            let fenced = match &buffer {
                RawBuffer::AmdDevice { gpu_addr, device, .. } => device.core().wait_storage(*gpu_addr),
                RawBuffer::Cuda { device_ptr, device, .. } => device.wait_storage(*device_ptr),
                _ => Ok(()),
            };
            if let Err(error) = fenced {
                self.inner.free(buffer, size, options);
                return Err(error);
            }
            if zero {
                match self.zero_cached(&buffer) {
                    Ok(true) => {}
                    Ok(false) => {
                        // Device-only buffer we can't memset on the host: free
                        // it and allocate fresh rather than returning stale data.
                        self.inner.free(buffer, size, options);
                        return self.inner.alloc(size, options, zero);
                    }
                    Err(error) => {
                        self.inner.free(buffer, size, options);
                        return Err(error);
                    }
                }
            }
            return Ok(buffer);
        }

        // Cache miss → backend alloc; on failure drain the pool and retry once.
        match self.inner.alloc(size, options, zero) {
            Ok(buffer) => Ok(buffer),
            Err(e) => {
                self.free_cache();
                self.inner.alloc(size, options, zero).map_err(|_| e)
            }
        }
    }

    fn free(&self, buffer: RawBuffer, size: usize, options: &BufferSpec) {
        // nolru bypasses the pool — real free now.
        if options.nolru {
            self.inner.free(buffer, size, options);
            return;
        }

        // Recycle into the pool. NOTE: no synchronize here — the LRU recycle is
        // intentionally undrained; the timeline drain happens in the backend
        // `_free` on real teardown (`AmdAllocator::_free`). On overflow route
        // through `inner.free` so the handle is actually released (RawBuffer
        // has no Drop).
        let overflow = {
            let mut cache = self.cache.lock().unwrap();
            let buffers = cache.entry((size, *options)).or_default();
            if buffers.len() < self.max_buffers_per_size {
                buffers.push(buffer);
                None
            } else {
                Some(buffer)
            }
        };
        if let Some(buf) = overflow {
            self.inner.free(buf, size, options);
        }
    }

    // The decorator forwards the backend hooks to the wrapped allocator.
    fn _alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> Result<RawBuffer> {
        self.inner._alloc(size, options, zero)
    }
    fn _free(&self, buffer: RawBuffer, options: &BufferSpec) {
        self.inner._free(buffer, options);
    }
    fn _copyin(&self, dest: &RawBuffer, dest_off: usize, src: &[u8]) -> Result<()> {
        self.inner._copyin(dest, dest_off, src)
    }
    fn _copyout(&self, dest: &mut [u8], src: &RawBuffer, src_off: usize) -> Result<()> {
        self.inner._copyout(dest, src, src_off)
    }
    fn _transfer(&self, dest: &RawBuffer, dest_off: usize, src: &RawBuffer, src_off: usize, sz: usize) -> Result<()> {
        self.inner._transfer(dest, dest_off, src, src_off, sz)
    }
    fn _offset(&self, buf: &RawBuffer, size: usize, offset: usize) -> Result<RawBuffer> {
        self.inner._offset(buf, size, offset)
    }
    fn _map(&self, buf: &RawBuffer) -> Result<RawBuffer> {
        self.inner._map(buf)
    }
    fn _unmap(&self, mb: &RawBuffer) {
        self.inner._unmap(mb);
    }

    fn synchronize(&self) -> Result<()> {
        self.inner.synchronize()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn device_spec(&self) -> svod_dtype::DeviceSpec {
        self.inner.device_spec()
    }

    fn supports_device_local(&self) -> bool {
        self.inner.supports_device_local()
    }
}
