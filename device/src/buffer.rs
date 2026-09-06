use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use smallvec::{SmallVec, smallvec};
use svod_dtype::DType;

use snafu::ResultExt;
use svod_dtype::ext::HasDType;

use crate::allocator::{Allocator, BufferSpec, RawBuffer};
use crate::error::{
    ImmutableBufferSnafu, InvalidViewSnafu, NdarrayShapeSnafu, NotCpuAccessibleSnafu, Result, SizeMismatchSnafu,
    TypeMismatchSnafu, UnsupportedSnafu,
};

/// Global counter for unique buffer IDs.
///
/// Uses `AtomicU64` to generate unique IDs across threads.
/// IDs are monotonically increasing and never reused.
static BUFFER_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_buffer_id() -> u64 {
    BUFFER_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Unique identifier for a buffer handle.
///
/// Distinct identity per view: each `Buffer` value carries its own `BufferId`,
/// including views — so two disjoint slices of a shared arena have different
/// ids and the parallel hazard model can treat them as independent. Use
/// [`Buffer::storage_id`] when storage-identity (rather than handle-identity)
/// matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u64);

/// Shared buffer data that can be referenced by multiple views.
#[derive(Debug)]
struct BufferData {
    /// Stable per-storage identifier minted when the underlying allocation
    /// is created. Distinct from the per-handle [`Buffer::id`]: every
    /// `Buffer` value (including views) gets a fresh handle id, but every
    /// view of one allocation shares the same `storage_id`. Used by code
    /// that needs storage identity (e.g. alias detection in the memory
    /// planner) without falling into the `Arc::as_ptr` aliasing trap.
    storage_id: BufferId,
    /// Lazily-initialized raw buffer (lock-free after first allocation).
    raw: OnceLock<RawBuffer>,
    allocator: Arc<dyn Allocator>,
    /// Total size of the underlying allocation in bytes.
    total_size: usize,
    /// Allocation spec (the LRU cache key alongside `total_size`).
    options: BufferSpec,
    /// Whether to zero-initialize on allocation. Threaded into `alloc` as a
    /// side argument rather than a `BufferSpec` field so it does not split the
    /// cache (see [`BufferSpec`]).
    zero_init: bool,
    /// One-way immutability seal for shared weight storages: host writes
    /// through any handle or view are refused once set (see
    /// [`Buffer::mark_immutable`]).
    immutable: std::sync::atomic::AtomicBool,
}

impl BufferData {
    fn new(allocator: Arc<dyn Allocator>, size: usize, options: BufferSpec, zero_init: bool) -> Self {
        Self {
            storage_id: BufferId(next_buffer_id()),
            raw: OnceLock::new(),
            allocator,
            total_size: size,
            options,
            zero_init,
            immutable: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Ensure the buffer is allocated, allocating if necessary.
    /// Uses lock-free OnceLock for efficient repeated checks.
    fn ensure_allocated(&self) -> Result<()> {
        if self.raw.get().is_some() {
            return Ok(());
        }

        // Allocate - if another thread beat us, that's fine
        let raw = self.allocator.alloc(self.total_size, &self.options, self.zero_init)?;

        // Try to set - if another thread beat us, free this allocation
        if let Err(raw) = self.raw.set(raw) {
            // Another thread won the race - free our allocation
            self.allocator.free(raw, self.total_size, &self.options);
        }

        Ok(())
    }

    /// Check if the buffer is currently allocated.
    fn is_allocated(&self) -> bool {
        self.raw.get().is_some()
    }

    /// Get raw buffer reference (buffer must be allocated).
    fn raw(&self) -> &RawBuffer {
        self.raw.get().expect("buffer not allocated")
    }
}

impl Drop for BufferData {
    fn drop(&mut self) {
        // Free the buffer if it was allocated
        if let Some(raw) = self.raw.take() {
            self.allocator.free(raw, self.total_size, &self.options);
        }
    }
}

/// A device buffer that may be a view into another buffer.
///
/// Handle-identity (`id`) is per-`Buffer` value, including views — each view
/// produces a distinct identity. Storage-identity (the underlying
/// `Arc<BufferData>`) is shared between a buffer and its views; use
/// [`Buffer::storage_id`] to compare it.
#[derive(Debug, Clone)]
pub struct Buffer {
    /// Per-handle unique identifier. Views get fresh ids; storage is shared
    /// via `data` independently.
    id: BufferId,
    /// Shared data for the base allocation.
    data: Arc<BufferData>,
    /// Offset into the base buffer (in bytes).
    offset: usize,
    /// Size of this view (in bytes).
    size: usize,
    /// Data type of the buffer elements.
    dtype: DType,
    /// Shape of the tensor (stack-allocated for 0-4D tensors).
    shape: SmallVec<[usize; 4]>,
}

/// A [`DeviceSpec`](svod_dtype::DeviceSpec) resolved to the backend identity
/// that [`Buffer::matches_native`] compares against.
///
/// `AmdDevice::open` takes a process-global cache mutex, so a caller checking
/// many buffers against one device resolves this once. The AMD core is opened
/// lazily, on the first buffer that is actually AMD-backed: a host-backed
/// buffer merely *tagged* AMD is a mismatch, not a reason to demand that the
/// GPU be openable.
pub enum NativeDevice {
    Host(svod_dtype::DeviceSpec),
    Amd { device_id: usize, core: OnceLock<Arc<crate::amd::AmdDeviceCore>> },
}

impl NativeDevice {
    pub fn resolve(spec: &svod_dtype::DeviceSpec) -> Self {
        match spec {
            svod_dtype::DeviceSpec::Amd { device_id } => Self::Amd { device_id: *device_id, core: OnceLock::new() },
            host => Self::Host(host.clone()),
        }
    }
}

impl Buffer {
    /// Device which owns the underlying allocation.
    pub fn device_spec(&self) -> svod_dtype::DeviceSpec {
        self.data.allocator.device_spec()
    }

    /// Verify that an AMD-tagged buffer is backed by the exact physical KFD
    /// device, not merely by an allocator reporting the same display spec.
    ///
    /// Resolving the spec costs a lock on the process-global AMD device cache,
    /// so a caller validating many buffers against one device should resolve a
    /// [`NativeDevice`] once and use [`Buffer::matches_native`].
    pub fn matches_native_device(&self, expected: &svod_dtype::DeviceSpec) -> Result<bool> {
        self.matches_native(&NativeDevice::resolve(expected))
    }

    /// [`matches_native_device`](Self::matches_native_device) against an
    /// already-resolved device.
    pub fn matches_native(&self, expected: &NativeDevice) -> Result<bool> {
        match expected {
            NativeDevice::Host(spec) => Ok(self.device_spec() == *spec),
            NativeDevice::Amd { device_id, core } => {
                self.data.ensure_allocated()?;
                let RawBuffer::AmdDevice { device, .. } = self.data.raw() else {
                    return Ok(false);
                };
                let expected = match core.get() {
                    Some(core) => core,
                    None => {
                        let opened = Arc::clone(crate::amd::AmdDevice::open(*device_id)?.core());
                        core.get_or_init(|| opened)
                    }
                };
                Ok(Arc::ptr_eq(device.core(), expected))
            }
        }
    }

    /// Create a new buffer with lazy allocation (not zero-initialized).
    pub fn new(allocator: Arc<dyn Allocator>, dtype: DType, shape: Vec<usize>, options: BufferSpec) -> Self {
        Self::new_with_zero_init(allocator, dtype, shape, options, false)
    }

    /// Create a new buffer with lazy allocation, controlling zero-initialization.
    pub fn new_with_zero_init(
        allocator: Arc<dyn Allocator>,
        dtype: DType,
        shape: Vec<usize>,
        options: BufferSpec,
        zero_init: bool,
    ) -> Self {
        let size = dtype.bytes() * shape.iter().product::<usize>();
        Self {
            id: BufferId(next_buffer_id()),
            data: Arc::new(BufferData::new(allocator, size, options, zero_init)),
            offset: 0,
            size,
            dtype,
            shape: SmallVec::from_vec(shape),
        }
    }

    /// Create a new buffer with immediate allocation (not zero-initialized).
    pub fn allocate(
        allocator: Arc<dyn Allocator>,
        dtype: DType,
        shape: Vec<usize>,
        options: BufferSpec,
    ) -> Result<Self> {
        let buffer = Self::new(allocator, dtype, shape, options);
        buffer.ensure_allocated()?;
        Ok(buffer)
    }

    /// Create a new buffer with immediate allocation, controlling zero-initialization.
    pub fn allocate_with_zero_init(
        allocator: Arc<dyn Allocator>,
        dtype: DType,
        shape: Vec<usize>,
        options: BufferSpec,
        zero_init: bool,
    ) -> Result<Self> {
        let buffer = Self::new_with_zero_init(allocator, dtype, shape, options, zero_init);
        buffer.ensure_allocated()?;
        Ok(buffer)
    }

    /// Create a view into this buffer.
    ///
    /// The view shares storage with `self` (same `Arc<BufferData>`) but gets
    /// a **fresh `BufferId`** so the runtime parallel-hazard model treats
    /// disjoint views of one arena as independent (each view is a distinct
    /// identity). Use [`Buffer::storage_id`] to compare storage identity
    /// instead.
    pub fn view(&self, offset: usize, size: usize) -> Result<Self> {
        // Validate view parameters
        if offset + size > self.size {
            return InvalidViewSnafu { offset, size, buffer_size: self.size }.fail();
        }

        Ok(Self {
            id: BufferId(next_buffer_id()),
            data: Arc::clone(&self.data),
            offset: self.offset + offset,
            size,
            dtype: self.dtype.clone(),
            // For views, shape is not well-defined without reshaping logic
            shape: smallvec![size / self.dtype.bytes()],
        })
    }

    /// Fork the single storage shared by `views` into a fresh, unshared
    /// allocation (same allocator, total size, [`BufferSpec`] and zero-init
    /// flag) and re-mint every view onto it at its original
    /// offset/size/dtype/shape, with fresh handle ids. When `copy_contents`
    /// is set and the source storage is allocated, the entire allocation is
    /// copied over first (on-device when possible).
    pub fn fork_views(views: &[&Buffer], copy_contents: bool) -> Result<Vec<Buffer>> {
        let Some(first) = views.first() else { return Ok(Vec::new()) };
        let storage = first.storage_id();
        for view in views {
            snafu::ensure!(view.storage_id() == storage, UnsupportedSnafu { op: "fork_views across storages" });
        }
        let total_size = first.data.total_size;
        let whole = |data: &Arc<BufferData>| Self {
            id: BufferId(next_buffer_id()),
            data: Arc::clone(data),
            offset: 0,
            size: total_size,
            dtype: DType::UInt8,
            shape: smallvec![total_size],
        };
        let fresh = Arc::new(BufferData::new(
            Arc::clone(&first.data.allocator),
            total_size,
            first.data.options,
            first.data.zero_init,
        ));
        if copy_contents && first.data.is_allocated() {
            whole(&fresh).copy_from(&whole(&first.data))?;
        }
        Ok(views
            .iter()
            .map(|view| Self {
                id: BufferId(next_buffer_id()),
                data: Arc::clone(&fresh),
                offset: view.offset,
                size: view.size,
                dtype: view.dtype.clone(),
                shape: view.shape.clone(),
            })
            .collect())
    }

    /// Whether the underlying storage was sealed immutable.
    pub fn is_immutable(&self) -> bool {
        self.data.immutable.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Seal the underlying storage against host writes (one-way). Used for
    /// shared weight storages: a write through ANY handle or view would
    /// corrupt every model reading it, so `copyin*`, `as_*_mut` and
    /// copy-destination paths fail with `Error::ImmutableBuffer` afterwards.
    /// Device-side kernel writes are not intercepted here — the planner and
    /// replicate write-set analyses keep sealed storages out of kernel write
    /// positions. Forking (`fork_views`) stays legal and yields fresh,
    /// mutable storage.
    pub fn mark_immutable(&self) {
        self.data.immutable.store(true, std::sync::atomic::Ordering::Release);
    }

    fn ensure_mutable(&self, op: &'static str) -> Result<()> {
        if self.is_immutable() {
            return ImmutableBufferSnafu { op, storage: self.data.storage_id.0 }.fail();
        }
        Ok(())
    }

    /// Ensure the underlying buffer is allocated.
    pub fn ensure_allocated(&self) -> Result<()> {
        self.data.ensure_allocated()
    }

    /// Check if the buffer is allocated.
    pub fn is_allocated(&self) -> bool {
        self.data.is_allocated()
    }

    /// Get the size of this buffer view in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the offset of this view in bytes.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    /// Get the shape of this buffer.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get a byte slice of the buffer data (CPU-accessible buffers only).
    ///
    /// Zero-copy. For realized tensors after `realize()`, this is safe because
    /// the scheduler guarantees no concurrent kernel writes.
    ///
    /// # Errors
    /// - `NotAllocated` if buffer hasn't been allocated
    /// - `NotCpuAccessible` for device-only buffers (use `copyout` instead)
    pub fn as_host_bytes(&self) -> Result<&[u8]> {
        self.ensure_allocated()?;
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, .. } => {
                // SAFETY: After realize(), no kernels are executing.
                // The scheduler guarantees exclusive access during kernel execution;
                // user code only accesses buffers between kernel runs.
                let bytes = unsafe { &(&(*data.get()))[self.offset..self.offset + self.size] };
                Ok(bytes)
            }
            RawBuffer::Mmap { data, .. } => Ok(&data[self.offset..self.offset + self.size]),
            RawBuffer::Metal { contents, device, .. } => {
                let base = metal_host_ptr(device, *contents, self.offset)?;
                Ok(unsafe { std::slice::from_raw_parts(base, self.size) })
            }
            RawBuffer::Cuda { device_ptr, host_ptr, device, .. } => {
                let base = cuda_host_ptr(device, *device_ptr, *host_ptr, self.offset)?;
                Ok(unsafe { std::slice::from_raw_parts(base, self.size) })
            }
            RawBuffer::AmdDevice { host_ptr: Some(ptr), gpu_addr, device, .. } => {
                // Async dispatch: drain before raw host access — host-pointer
                // reads/writes aren't ordered on the GPU timeline.
                device.core().wait_storage(*gpu_addr)?;
                // SAFETY: same invariants as the CPU arm — scheduler ensures
                // exclusivity, and the BAR-backed VRAM mapping is valid for
                // the lifetime of the RawBuffer.
                let base = unsafe { ptr.as_ptr().add(self.offset) };
                Ok(unsafe { std::slice::from_raw_parts(base, self.size) })
            }
            RawBuffer::AmdDevice { host_ptr: None, gpu_addr, size, .. } => {
                // Diagnostic: this is the path that fires when a buffer was
                // alloc'd with `cpu_access: false` (no host mmap). The
                // public Tensor / runtime path always uses
                // `BufferSpec::default()` (cpu_access: true), so any
                // hit here is a regression in some downstream allocation path.
                tracing::warn!(
                    buffer_id = self.id.0,
                    storage_id = self.data.storage_id.0,
                    gpu_addr = *gpu_addr,
                    full_size = *size,
                    view_offset = self.offset,
                    view_size = self.size,
                    allocator = self.data.allocator.name(),
                    "AMD buffer alloc'd without cpu_accessible=true; CPU read will fail"
                );
                NotCpuAccessibleSnafu.fail()
            }
        }
    }

    /// Get a mutable byte slice of the buffer data (CPU-accessible buffers only).
    ///
    /// # Safety contract (same as `as_host_bytes`)
    /// Caller must ensure no kernels are executing concurrently.
    ///
    /// # Errors
    /// - `NotAllocated` if buffer hasn't been allocated
    /// - `NotCpuAccessible` for device-only buffers
    #[allow(clippy::mut_from_ref)] // interior mutability via UnsafeCell
    pub fn as_host_bytes_mut(&self) -> Result<&mut [u8]> {
        self.ensure_mutable("as_host_bytes_mut")?;
        self.ensure_allocated()?;
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, .. } => {
                // SAFETY: Same invariant as as_host_bytes — user code only
                // accesses buffers between kernel runs. UnsafeCell provides
                // interior mutability through the shared Arc<BufferData>.
                let bytes = unsafe { &mut (&mut *data.get())[self.offset..self.offset + self.size] };
                Ok(bytes)
            }
            // Mmap is read-only — no mutable access
            RawBuffer::Mmap { .. } => NotCpuAccessibleSnafu.fail(),
            RawBuffer::Metal { contents, device, .. } => {
                let base = metal_host_ptr(device, *contents, self.offset)?;
                Ok(unsafe { std::slice::from_raw_parts_mut(base, self.size) })
            }
            RawBuffer::Cuda { device_ptr, host_ptr, device, .. } => {
                let base = cuda_host_ptr(device, *device_ptr, *host_ptr, self.offset)?;
                Ok(unsafe { std::slice::from_raw_parts_mut(base, self.size) })
            }
            RawBuffer::AmdDevice { host_ptr: Some(ptr), gpu_addr, device, .. } => {
                // Async dispatch: drain before raw host access — host-pointer
                // reads/writes aren't ordered on the GPU timeline.
                device.core().wait_storage(*gpu_addr)?;
                let base = unsafe { ptr.as_ptr().add(self.offset) };
                Ok(unsafe { std::slice::from_raw_parts_mut(base, self.size) })
            }
            RawBuffer::AmdDevice { host_ptr: None, .. } => NotCpuAccessibleSnafu.fail(),
        }
    }

    /// Typed immutable view over CPU-accessible buffer memory.
    ///
    /// Returns an ndarray view shaped according to the buffer's concrete dimensions.
    /// Only works for CPU-accessible buffers — fails for device-only memory.
    ///
    /// # Errors
    /// - `TypeMismatch` if `T::DTYPE` doesn't match buffer dtype
    /// - `NotCpuAccessible` for non-CPU-accessible buffers
    /// - `NotAllocated` if buffer hasn't been allocated
    pub fn as_array<T: HasDType>(&self) -> Result<ndarray::ArrayViewD<'_, T>> {
        self.ensure_allocated()?;
        if self.dtype != T::DTYPE {
            return TypeMismatchSnafu { expected: T::DTYPE, actual: self.dtype.clone() }.fail();
        }
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, .. } => {
                let bytes = unsafe { &(&(*data.get()))[self.offset..self.offset + self.size] };
                let count = bytes.len() / T::DTYPE.bytes();
                let typed = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) };
                ndarray::ArrayViewD::from_shape(ndarray::IxDyn(&self.shape), typed).context(NdarrayShapeSnafu)
            }
            RawBuffer::Mmap { data, .. } => {
                let bytes = &data[self.offset..self.offset + self.size];
                let count = bytes.len() / T::DTYPE.bytes();
                let typed = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) };
                ndarray::ArrayViewD::from_shape(ndarray::IxDyn(&self.shape), typed).context(NdarrayShapeSnafu)
            }
            RawBuffer::Metal { contents, device, .. } => {
                let bytes_ptr = metal_host_ptr(device, *contents, self.offset)? as *const T;
                let count = self.size / T::DTYPE.bytes();
                let typed = unsafe { std::slice::from_raw_parts(bytes_ptr, count) };
                ndarray::ArrayViewD::from_shape(ndarray::IxDyn(&self.shape), typed).context(NdarrayShapeSnafu)
            }
            RawBuffer::Cuda { device_ptr, host_ptr, device, .. } => {
                let bytes_ptr = cuda_host_ptr(device, *device_ptr, *host_ptr, self.offset)? as *const T;
                let count = self.size / T::DTYPE.bytes();
                let typed = unsafe { std::slice::from_raw_parts(bytes_ptr, count) };
                ndarray::ArrayViewD::from_shape(ndarray::IxDyn(&self.shape), typed).context(NdarrayShapeSnafu)
            }
            RawBuffer::AmdDevice { host_ptr: Some(ptr), gpu_addr, device, .. } => {
                // Async dispatch: drain before raw host access — host-pointer
                // reads/writes aren't ordered on the GPU timeline.
                device.core().wait_storage(*gpu_addr)?;
                let bytes_ptr = unsafe { ptr.as_ptr().add(self.offset) } as *const T;
                let count = self.size / T::DTYPE.bytes();
                let typed = unsafe { std::slice::from_raw_parts(bytes_ptr, count) };
                ndarray::ArrayViewD::from_shape(ndarray::IxDyn(&self.shape), typed).context(NdarrayShapeSnafu)
            }
            RawBuffer::AmdDevice { host_ptr: None, gpu_addr, size, .. } => {
                tracing::warn!(
                    buffer_id = self.id.0,
                    storage_id = self.data.storage_id.0,
                    gpu_addr = *gpu_addr,
                    full_size = *size,
                    view_offset = self.offset,
                    view_size = self.size,
                    requested_dtype = ?T::DTYPE,
                    allocator = self.data.allocator.name(),
                    "AMD buffer alloc'd without cpu_accessible=true; as_array() will fail"
                );
                NotCpuAccessibleSnafu.fail()
            }
        }
    }

    /// Typed mutable view over CPU-accessible buffer memory.
    ///
    /// Same as [`Self::as_array`] but allows writes. Caller must ensure no
    /// kernels are executing concurrently (safety is the caller's
    /// responsibility).
    ///
    /// # Errors
    /// Same as [`Self::as_array`].
    #[allow(clippy::mut_from_ref)]
    pub fn as_array_mut<T: HasDType>(&self) -> Result<ndarray::ArrayViewMutD<'_, T>> {
        self.ensure_mutable("as_array_mut")?;
        self.ensure_allocated()?;
        if self.dtype != T::DTYPE {
            return TypeMismatchSnafu { expected: T::DTYPE, actual: self.dtype.clone() }.fail();
        }
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, cpu_accessible } if *cpu_accessible => {
                let bytes = unsafe { &mut (&mut *data.get())[self.offset..self.offset + self.size] };
                let count = bytes.len() / T::DTYPE.bytes();
                let typed = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, count) };
                ndarray::ArrayViewMutD::from_shape(ndarray::IxDyn(&self.shape), typed).context(NdarrayShapeSnafu)
            }
            RawBuffer::Metal { contents, device, .. } => {
                let bytes_ptr = metal_host_ptr(device, *contents, self.offset)? as *mut T;
                let count = self.size / T::DTYPE.bytes();
                let typed = unsafe { std::slice::from_raw_parts_mut(bytes_ptr, count) };
                ndarray::ArrayViewMutD::from_shape(ndarray::IxDyn(&self.shape), typed).context(NdarrayShapeSnafu)
            }
            RawBuffer::Cuda { device_ptr, host_ptr: host_ptr @ Some(_), device, .. } => {
                let bytes_ptr = cuda_host_ptr(device, *device_ptr, *host_ptr, self.offset)? as *mut T;
                let count = self.size / T::DTYPE.bytes();
                let typed = unsafe { std::slice::from_raw_parts_mut(bytes_ptr, count) };
                ndarray::ArrayViewMutD::from_shape(ndarray::IxDyn(&self.shape), typed).context(NdarrayShapeSnafu)
            }
            RawBuffer::AmdDevice { host_ptr: Some(ptr), gpu_addr, device, .. } => {
                // Async dispatch: drain before raw host access — host-pointer
                // reads/writes aren't ordered on the GPU timeline.
                device.core().wait_storage(*gpu_addr)?;
                // SAFETY: BAR-backed VRAM mapping is valid for the buffer's
                // lifetime; scheduler ensures no concurrent kernel writes.
                let bytes_ptr = unsafe { ptr.as_ptr().add(self.offset) } as *mut T;
                let count = self.size / T::DTYPE.bytes();
                let typed = unsafe { std::slice::from_raw_parts_mut(bytes_ptr, count) };
                ndarray::ArrayViewMutD::from_shape(ndarray::IxDyn(&self.shape), typed).context(NdarrayShapeSnafu)
            }
            _ => NotCpuAccessibleSnafu.fail(),
        }
    }

    /// Zero-copy typed slice view (CPU-accessible only).
    pub fn as_slice<T: HasDType>(&self) -> Result<&[T]> {
        self.ensure_allocated()?;
        if self.dtype != T::DTYPE {
            return TypeMismatchSnafu { expected: T::DTYPE, actual: self.dtype.clone() }.fail();
        }
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, cpu_accessible } if *cpu_accessible => {
                let bytes = unsafe { &(&(*data.get()))[self.offset..self.offset + self.size] };
                let count = bytes.len() / T::DTYPE.bytes();
                Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) })
            }
            RawBuffer::Metal { contents, device, .. } => {
                let bytes_ptr = metal_host_ptr(device, *contents, self.offset)? as *const T;
                let count = self.size / T::DTYPE.bytes();
                Ok(unsafe { std::slice::from_raw_parts(bytes_ptr, count) })
            }
            RawBuffer::Cuda { device_ptr, host_ptr: host_ptr @ Some(_), device, .. } => {
                let bytes_ptr = cuda_host_ptr(device, *device_ptr, *host_ptr, self.offset)? as *const T;
                let count = self.size / T::DTYPE.bytes();
                Ok(unsafe { std::slice::from_raw_parts(bytes_ptr, count) })
            }
            RawBuffer::AmdDevice { host_ptr: Some(ptr), gpu_addr, device, .. } => {
                // Async dispatch: drain before raw host access — host-pointer
                // reads/writes aren't ordered on the GPU timeline.
                device.core().wait_storage(*gpu_addr)?;
                let bytes_ptr = unsafe { ptr.as_ptr().add(self.offset) } as *const T;
                let count = self.size / T::DTYPE.bytes();
                Ok(unsafe { std::slice::from_raw_parts(bytes_ptr, count) })
            }
            _ => NotCpuAccessibleSnafu.fail(),
        }
    }

    /// Read a single scalar value from the buffer (CPU-accessible only).
    ///
    /// Panics if the buffer contains more than one element.
    pub fn item<T: HasDType + Copy>(&self) -> Result<T> {
        let slice = self.as_slice::<T>()?;
        assert_eq!(slice.len(), 1, "item() requires exactly 1 element, got {}", slice.len());
        Ok(slice[0])
    }

    /// Get the allocator used by this buffer.
    pub fn allocator(&self) -> &dyn Allocator {
        &*self.data.allocator
    }

    /// Get an `Arc`-cloned handle to the allocator, suitable for constructing
    /// new buffers on the same device (used by the arena memory planner to
    /// allocate per-lane arenas matching prototype buffers' device).
    pub fn allocator_arc(&self) -> Arc<dyn Allocator> {
        Arc::clone(&self.data.allocator)
    }

    /// Get the unique identifier for this buffer **handle**.
    ///
    /// Each `Buffer` value (including each view) carries its own `BufferId`;
    /// disjoint views of one arena therefore have different ids. Used by the
    /// runtime parallel-hazard model. To compare storage identity (i.e. "do
    /// these two buffers share the same underlying allocation"), use
    /// [`Buffer::storage_id`] instead.
    pub fn id(&self) -> BufferId {
        self.id
    }

    /// Size of the underlying allocation in bytes (shared by every view of
    /// this buffer's storage). Distinct from [`Buffer::size`], which returns
    /// the view's size — for a non-view buffer the two are equal; for a view
    /// into an arena, `total_size` reports the arena's allocation size while
    /// `size` reports just the view's window.
    pub fn total_size(&self) -> usize {
        self.data.total_size
    }

    /// Stable identifier for this buffer's underlying allocation.
    ///
    /// Equal across a base buffer and all of its views, distinct between
    /// independent allocations. Unlike a heap-pointer probe, this id is
    /// minted once at allocation time and never reused — safe to use as a
    /// hash key or alias-detection key without worrying about
    /// allocator-reuse aliasing.
    pub fn storage_id(&self) -> BufferId {
        self.data.storage_id
    }

    /// Copy data from host memory into this buffer.
    ///
    /// Delegates to the allocator's `_copyin`. The per-backend logic lives on
    /// the allocator, not here.
    pub fn copyin(&mut self, src: &[u8]) -> Result<()> {
        self.ensure_mutable("copyin")?;
        self.ensure_allocated()?;

        let expected = self.size;
        let actual = src.len();
        snafu::ensure!(expected == actual, SizeMismatchSnafu { expected, actual });

        self.data.allocator._copyin(self.data.raw(), self.offset, src)
    }

    /// Copy `src` into this buffer starting at byte `dst_off`. Partial-write
    /// counterpart to [`copyout_prefix`] — used to seed a region of a
    /// device-local buffer (e.g. one lane's KV-cache row) from host memory
    /// via the copy engine, without a host-visible mapping.
    pub fn copyin_at(&mut self, dst_off: usize, src: &[u8]) -> Result<()> {
        self.ensure_mutable("copyin_at")?;
        self.ensure_allocated()?;
        let end = dst_off
            .checked_add(src.len())
            .ok_or(crate::error::Error::SizeMismatch { expected: self.size, actual: usize::MAX })?;
        snafu::ensure!(end <= self.size, SizeMismatchSnafu { expected: self.size, actual: end });
        self.data.allocator._copyin(self.data.raw(), self.offset + dst_off, src)
    }

    /// Copy data from this buffer to host memory.
    ///
    /// Delegates to the allocator's `_copyout`. Device backends synchronize
    /// their timeline inside `_copyout` before reading.
    pub fn copyout(&self, dst: &mut [u8]) -> Result<()> {
        self.ensure_allocated()?;

        let expected = self.size;
        let actual = dst.len();
        snafu::ensure!(expected == actual, SizeMismatchSnafu { expected, actual });

        self.data.allocator._copyout(dst, self.data.raw(), self.offset)
    }

    /// Copy the buffer's first `dst.len()` bytes to host memory — a prefix
    /// read for const-shaped outputs whose active region is shorter than the
    /// allocation (e.g. a partial last batch in `[max_batch, …]` buffers).
    pub fn copyout_prefix(&self, dst: &mut [u8]) -> Result<()> {
        self.ensure_allocated()?;

        snafu::ensure!(dst.len() <= self.size, SizeMismatchSnafu { expected: self.size, actual: dst.len() });

        self.data.allocator._copyout(dst, self.data.raw(), self.offset)
    }

    /// Copy data from another buffer to this buffer.
    ///
    /// Same allocator instance (same device) → on-device `_transfer`.
    /// Cross-backend → bounce through host via `_copyout` then `_copyin`
    /// (there is no CPU↔GPU `_transfer`; cross-backend COPY goes through
    /// host). The source device is synchronized before the host read so async
    /// dispatch never races a still-running writer.
    pub fn copy_from(&mut self, src: &Buffer) -> Result<()> {
        self.ensure_mutable("copy_from")?;
        self.ensure_allocated()?;
        src.ensure_allocated()?;

        let expected = self.size;
        let actual = src.size;
        snafu::ensure!(expected == actual, SizeMismatchSnafu { expected, actual });

        if Arc::ptr_eq(&self.data.allocator, &src.data.allocator) {
            self.data.allocator._transfer(self.data.raw(), self.offset, src.data.raw(), src.offset, self.size)
        } else {
            src.synchronize()?;
            let mut staging = vec![0u8; self.size];
            src.data.allocator._copyout(&mut staging, src.data.raw(), src.offset)?;
            self.data.allocator._copyin(self.data.raw(), self.offset, &staging)
        }
    }

    /// Copy `len` bytes from `src[src_off..]` into `self[dst_off..]`. Both
    /// buffers must live on the same allocator — this is the on-device
    /// `_transfer` path (SDMA when either side is device-local), so recurrent
    /// state rows can be recycled output→input without touching the host.
    pub fn copy_region_from(&mut self, dst_off: usize, src: &Buffer, src_off: usize, len: usize) -> Result<()> {
        self.ensure_mutable("copy_region_from")?;
        self.ensure_allocated()?;
        src.ensure_allocated()?;
        let dst_end = dst_off
            .checked_add(len)
            .ok_or(crate::error::Error::SizeMismatch { expected: self.size, actual: usize::MAX })?;
        let src_end = src_off
            .checked_add(len)
            .ok_or(crate::error::Error::SizeMismatch { expected: src.size, actual: usize::MAX })?;
        snafu::ensure!(dst_end <= self.size, SizeMismatchSnafu { expected: self.size, actual: dst_end });
        snafu::ensure!(src_end <= src.size, SizeMismatchSnafu { expected: src.size, actual: src_end });
        snafu::ensure!(
            Arc::ptr_eq(&self.data.allocator, &src.data.allocator),
            UnsupportedSnafu { op: "copy_region_from across allocators" }
        );
        self.data.allocator._transfer(self.data.raw(), self.offset + dst_off, src.data.raw(), src.offset + src_off, len)
    }

    /// Copy a region within this buffer to another region in the same buffer
    /// (on-device SDMA, no host round-trip). Used to relocate a cache row when
    /// lane compaction shifts a surviving lane to a new row. The regions must
    /// not overlap.
    pub fn copy_within(&mut self, dst_off: usize, src_off: usize, len: usize) -> Result<()> {
        self.ensure_mutable("copy_within")?;
        self.ensure_allocated()?;
        let dst_end = dst_off
            .checked_add(len)
            .ok_or(crate::error::Error::SizeMismatch { expected: self.size, actual: usize::MAX })?;
        let src_end = src_off
            .checked_add(len)
            .ok_or(crate::error::Error::SizeMismatch { expected: self.size, actual: usize::MAX })?;
        snafu::ensure!(dst_end <= self.size, SizeMismatchSnafu { expected: self.size, actual: dst_end });
        snafu::ensure!(src_end <= self.size, SizeMismatchSnafu { expected: self.size, actual: src_end });
        snafu::ensure!(
            len == 0 || dst_end <= src_off || src_end <= dst_off,
            crate::error::RuntimeSnafu { message: "copy_within regions must not overlap" }
        );
        // No borrow conflict: .raw() returns an owned RawBuffer handle, so we
        // can capture it twice without aliasing &mut self.
        let src_raw = self.data.raw();
        self.data.allocator._transfer(self.data.raw(), self.offset + dst_off, src_raw, self.offset + src_off, len)
    }

    /// Synchronize the device (wait for all operations to complete).
    pub fn synchronize(&self) -> Result<()> {
        self.data.allocator.synchronize()
    }

    /// Record `token` as an in-flight producer/reader of this buffer's
    /// storage for scoped host synchronization (the AMD and CUDA
    /// `wait_storage`). No-op on other backends and on storage that was
    /// never allocated (nothing can be in flight against it).
    pub fn record_completion(&self, token: &Arc<dyn crate::sync::CompletionToken>) {
        match self.data.raw.get() {
            Some(RawBuffer::AmdDevice { gpu_addr, device, .. }) => device.core().record_producer(*gpu_addr, token),
            Some(RawBuffer::Cuda { device_ptr, device, .. }) => device.record_producer(*device_ptr, token),
            _ => {}
        }
    }

    /// Get a raw pointer to the buffer data for kernel execution.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid while the buffer is allocated.
    /// The caller must ensure:
    /// - Buffer remains allocated during pointer lifetime
    /// - No conflicting accesses occur during kernel execution
    /// - Pointer is not used after buffer is freed
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not yet allocated.
    pub unsafe fn as_raw_ptr(&self) -> *mut u8 {
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, .. } => {
                // SAFETY: Caller is responsible for ensuring no conflicting access.
                // This is already an unsafe function - caller guarantees exclusive access.
                unsafe { (&mut *data.get()).as_mut_ptr().add(self.offset) }
            }
            RawBuffer::Mmap { data, .. } => {
                // Read-only mmap: writing through this pointer is UB.
                unsafe { data.as_ptr().add(self.offset) as *mut u8 }
            }
            RawBuffer::Metal { contents, .. } => {
                // Host pointer into the shared MTLBuffer; `MetalProgram::execute`
                // resolves it back to (MTLBuffer, offset) through the device's
                // pointer registry. Not drained here: this is the dispatch path
                // and GPU-side ordering is the queue's job.
                unsafe { contents.as_ptr().add(self.offset) }
            }
            RawBuffer::AmdDevice { gpu_addr, .. } => {
                // GPU virtual address — what AMD kernels see in their kernarg
                // buffer for buffer parameters. The CPU never dereferences
                // this pointer; it's just stuffed into the kernarg slot.
                (*gpu_addr as usize + self.offset) as *mut u8
            }
            // Device pointer for the kernarg slot, as on AMD.
            RawBuffer::Cuda { device_ptr, .. } => (*device_ptr as usize + self.offset) as *mut u8,
        }
    }

    /// Resolve this buffer view to the address consumed by a target program.
    /// This is Tinygrad HCQ's host-side GETADDR stage: AMD returns a GPU VA,
    /// while host backends return their process address.
    pub fn device_address(&self) -> Result<u64> {
        self.ensure_allocated()?;
        // SAFETY: the returned integer is used only while the owning Buffer is
        // retained by the execution plan; it is never dereferenced here.
        Ok(unsafe { self.as_raw_ptr() } as usize as u64)
    }

    /// Get the raw data pointer for testing buffer identity.
    ///
    /// This is used in tests to verify cache reuse by comparing pointer addresses.
    /// Returns the pointer to the underlying buffer data.
    #[cfg(test)]
    pub(crate) fn raw_data_ptr(&self) -> usize {
        let raw = self.data.raw();
        match raw {
            RawBuffer::Cpu { data, .. } => {
                // SAFETY: Only reading the pointer address for test comparison
                unsafe { (*data.get()).as_ptr() as usize }
            }
            RawBuffer::Mmap { data, .. } => data.as_ptr() as usize,
            RawBuffer::AmdDevice { gpu_addr, .. } => *gpu_addr as usize,
            RawBuffer::Metal { contents, .. } => contents.as_ptr() as usize,
            RawBuffer::Cuda { device_ptr, .. } => *device_ptr as usize,
        }
    }
}

/// Host pointer into a managed or pinned CUDA allocation after waiting the
/// storage's in-flight producers and readers (host access is not ordered
/// against the plan streams); device-only allocations have no host side.
fn cuda_host_ptr(
    device: &Arc<crate::cuda::CudaDevice>,
    device_ptr: u64,
    host_ptr: Option<std::ptr::NonNull<u8>>,
    offset: usize,
) -> Result<*mut u8> {
    let host_ptr = host_ptr.ok_or(crate::error::Error::NotCpuAccessible)?;
    device.wait_storage(device_ptr)?;
    // SAFETY: the view offset is bounded by the allocation (`Buffer::view`).
    Ok(unsafe { host_ptr.as_ptr().add(offset) })
}

/// Drain the Metal device before raw host access: shared-storage reads and
/// writes are not ordered against committed command buffers. Metal has no
/// per-storage producer table (the AMD `wait_storage` equivalent), so the drain
/// is device-wide; the in-flight list is pruned of completed buffers on every
/// dispatch, so it stays cheap.
fn metal_host_ptr(
    device: &Arc<crate::metal::MetalDevice>,
    contents: std::ptr::NonNull<u8>,
    offset: usize,
) -> Result<*mut u8> {
    device.synchronize()?;
    // SAFETY: the view offset is bounded by the allocation (`Buffer::view`).
    Ok(unsafe { contents.as_ptr().add(offset) })
}
