//! Device-local allocations by default; host-visible buffers are managed
//! memory (device-resident, migrated on host touch) where host access to it
//! is coherent, else pinned host memory mapped into the device, which is
//! also what `host` buffers get. Host ↔ device copies wait the storage's
//! in-flight producers and readers on the host; up to the bounce size a
//! copy-in is one `cuMemcpyHtoDAsync` on the copy lane published as the
//! storage's producer (the driver returns once a pageable source is staged,
//! the DMA retires in stream order), a copy-out one synchronous
//! `cuMemcpyDtoH`; above it both stage through the device's pinned bounce
//! buffer on the copy stream. Device-to-device copies and memsets are
//! asynchronous on the copy lane, ordered after the producers by event and
//! published as the new producer.

use std::ptr::NonNull;
use std::sync::Arc;

use svod_dtype::DeviceSpec;

use super::device::{CudaDevice, STAGING_BYTES};
use super::sys::{CU_MEM_ATTACH_GLOBAL, CU_MEMHOSTALLOC_DEVICEMAP, CU_MEMHOSTALLOC_PORTABLE, CUdeviceptr};
use crate::allocator::{Allocator, BufferSpec, RawBuffer};
use crate::error::UnsupportedSnafu;
use crate::{Error, Result};

/// What backs a [`RawBuffer::Cuda`] allocation, which decides its free call
/// and its host access path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaMemory {
    /// `cuMemAlloc`: device memory, no host mapping.
    Device,
    /// `cuMemAllocManaged`: one address valid on both sides, resident where
    /// last touched.
    Managed,
    /// `cuMemHostAlloc(DEVICEMAP)`: pinned host memory read by kernels over
    /// the bus.
    Pinned,
}

#[derive(Clone)]
pub struct CudaAllocator {
    pub dev: Arc<CudaDevice>,
    pub device_id: usize,
}

impl CudaAllocator {
    pub fn new(device_id: usize) -> Result<Self> {
        Ok(Self { dev: CudaDevice::open(device_id)?, device_id })
    }

    fn cuda_buffer(&self, buffer: &RawBuffer) -> (CUdeviceptr, Option<NonNull<u8>>, usize, CudaMemory) {
        match buffer {
            RawBuffer::Cuda { device_ptr, host_ptr, size, memory, .. } => (*device_ptr, *host_ptr, *size, *memory),
            other => unreachable!("CudaAllocator used with a non-CUDA buffer: {other:?}"),
        }
    }

    /// `host` → pinned; `cpu_access` → managed when host access to it is
    /// coherent with running kernels, else pinned (WDDM, pre-Pascal);
    /// otherwise device memory.
    pub(crate) fn memory_kind(options: &BufferSpec, managed: bool) -> CudaMemory {
        match (options.host, options.cpu_access, managed) {
            (true, ..) | (false, true, false) => CudaMemory::Pinned,
            (false, true, true) => CudaMemory::Managed,
            (false, false, _) => CudaMemory::Device,
        }
    }

    fn alloc_failed(&self, size: usize, error: Error) -> Error {
        let usage =
            self.dev.memory_info().map(|(free, total)| format!(" (free {free} / total {total})")).unwrap_or_default();
        Error::CudaAllocFailed { size, reason: format!("{error}{usage}") }
    }

    /// Copy `len` bytes in bounce-buffer-sized chunks through the device's
    /// pinned staging memory; `step(staging, done, chunk)` moves one chunk and
    /// must leave the staging memory reusable (stream synchronized).
    fn staged(&self, len: usize, mut step: impl FnMut(NonNull<u8>, usize, usize) -> Result<()>) -> Result<()> {
        self.dev.with_staging(|staging, capacity| {
            let mut done = 0;
            while done < len {
                let chunk = capacity.min(len - done);
                step(staging, done, chunk)?;
                done += chunk;
            }
            Ok(())
        })
    }
}

impl std::fmt::Debug for CudaAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaAllocator").field("device_id", &self.device_id).field("device", &self.dev).finish()
    }
}

impl Allocator for CudaAllocator {
    /// See [`Self::memory_kind`].
    fn _alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> Result<RawBuffer> {
        let api = self.dev.enter()?;
        let alloc_len = size.max(1);
        let memory = Self::memory_kind(options, self.dev.limits().managed_memory);
        let mut device_ptr: CUdeviceptr = 0;
        let mut host_ptr = None;
        // SAFETY: out-pointers to live slots; flags per `cuda.h`.
        let result = unsafe {
            match memory {
                CudaMemory::Device => (api.mem_alloc)(&mut device_ptr, alloc_len).check("cuMemAlloc"),
                CudaMemory::Managed => {
                    (api.mem_alloc_managed)(&mut device_ptr, alloc_len, CU_MEM_ATTACH_GLOBAL).check("cuMemAllocManaged")
                }
                CudaMemory::Pinned => {
                    let mut raw = std::ptr::null_mut();
                    (api.mem_host_alloc)(&mut raw, alloc_len, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP)
                        .check("cuMemHostAlloc")
                        .and_then(|()| {
                            (api.mem_host_get_device_pointer)(&mut device_ptr, raw, 0)
                                .check("cuMemHostGetDevicePointer")
                                .inspect_err(|_| {
                                    (api.mem_free_host)(raw);
                                })
                        })
                        .map(|()| host_ptr = NonNull::new(raw.cast::<u8>()))
                }
            }
        };
        result.map_err(|error| self.alloc_failed(size, error))?;
        if memory == CudaMemory::Managed {
            host_ptr = NonNull::new(device_ptr as usize as *mut u8);
        }
        self.dev.register_storage(device_ptr);
        let buffer = RawBuffer::Cuda { device_ptr, host_ptr, size, memory, device: Arc::clone(&self.dev) };
        if zero && let Err(error) = self.dev.zero(device_ptr, alloc_len) {
            self._free(buffer, options);
            return Err(error);
        }
        Ok(buffer)
    }

    fn _free(&self, buffer: RawBuffer, _options: &BufferSpec) {
        let RawBuffer::Cuda { device_ptr, host_ptr, size, memory, device } = &buffer else {
            tracing::debug!(?buffer, "CudaAllocator::free called with non-CUDA buffer; dropping");
            return;
        };
        // In-flight submissions may still reference the allocation; a failed
        // wait (or a poisoned context) cannot propagate from a free, so the
        // allocation is quarantined instead.
        if let Err(error) = device.wait_storage(*device_ptr) {
            tracing::warn!(?error, size, "CudaAllocator::free: wait failed; allocation quarantined");
            std::mem::forget(buffer);
            return;
        }
        device.unregister_storage(*device_ptr);
        let api = device.api();
        // SAFETY: the allocation this buffer owns, freed by the call that made it.
        let result = unsafe {
            match memory {
                CudaMemory::Device | CudaMemory::Managed => (api.mem_free)(*device_ptr),
                CudaMemory::Pinned => (api.mem_free_host)(host_ptr.map_or(std::ptr::null_mut(), |p| p.as_ptr().cast())),
            }
        };
        if let Err(error) = result.check("cuMemFree") {
            tracing::warn!(?error, size, "CudaAllocator::free failed");
        }
    }

    /// Host access is not ordered against the lanes, so the storage's
    /// in-flight producers and readers are waited first (a host overwrite is
    /// a WAR hazard against readers). Pinned memory is then a plain `memcpy`;
    /// device and managed memory take one copy-lane `cuMemcpyHtoDAsync` up
    /// to the bounce size, published as the storage's producer so later
    /// launches on any lane wait its DMA, and chunked pinned staging above it.
    fn _copyin(&self, dest: &RawBuffer, dest_off: usize, src: &[u8]) -> Result<()> {
        let (device_ptr, host_ptr, _, memory) = self.cuda_buffer(dest);
        self.dev.wait_storage(device_ptr)?;
        if src.is_empty() {
            return Ok(());
        }
        if let (CudaMemory::Pinned, Some(host)) = (memory, host_ptr) {
            // SAFETY: the caller bounds `dest_off + src.len()` by the allocation.
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), host.as_ptr().add(dest_off), src.len()) };
            return Ok(());
        }
        let api = self.dev.api();
        let dst = device_ptr + dest_off as u64;
        let stream = self.dev.copy_stream();
        if src.len() <= STAGING_BYTES {
            return self.dev.with_copy_lane(|dev| {
                // SAFETY: the destination range is bounded by the caller; the
                // driver returns once the pageable `src` is staged, so it is
                // free afterwards while the DMA retires in stream order.
                dev.check(
                    unsafe { (api.memcpy_htod_async)(dst, src.as_ptr().cast(), src.len(), stream) },
                    "cuMemcpyHtoDAsync",
                )?;
                dev.record_copy(&[device_ptr])
            });
        }
        self.staged(src.len(), |staging, done, chunk| {
            // SAFETY: `chunk` fits the bounce buffer; the destination range is
            // bounded by the caller; the copy is waited for before reuse.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr().add(done), staging.as_ptr(), chunk);
                self.dev.check(
                    (api.memcpy_htod_async)(dst + done as u64, staging.as_ptr().cast(), chunk, stream),
                    "cuMemcpyHtoDAsync",
                )?;
            }
            self.dev.stream_synchronize(stream)
        })
    }

    /// Mirror of [`Self::_copyin`].
    fn _copyout(&self, dest: &mut [u8], src: &RawBuffer, src_off: usize) -> Result<()> {
        let (device_ptr, host_ptr, _, memory) = self.cuda_buffer(src);
        self.dev.wait_storage(device_ptr)?;
        if dest.is_empty() {
            return Ok(());
        }
        if let (CudaMemory::Pinned, Some(host)) = (memory, host_ptr) {
            // SAFETY: as `_copyin`.
            unsafe { std::ptr::copy_nonoverlapping(host.as_ptr().add(src_off), dest.as_mut_ptr(), dest.len()) };
            return Ok(());
        }
        let api = self.dev.api();
        let source = device_ptr + src_off as u64;
        let len = dest.len();
        let out = dest.as_mut_ptr();
        if len <= STAGING_BYTES {
            // SAFETY: the source range is bounded by the caller; `dest` is
            // written before the call returns.
            return self.dev.check(unsafe { (api.memcpy_dtoh)(out.cast(), source, len) }, "cuMemcpyDtoH");
        }
        let stream = self.dev.copy_stream();
        self.staged(len, |staging, done, chunk| {
            // SAFETY: as `_copyin`; the host read happens after the copy retired.
            unsafe {
                self.dev.check(
                    (api.memcpy_dtoh_async)(staging.as_ptr().cast(), source + done as u64, chunk, stream),
                    "cuMemcpyDtoHAsync",
                )?;
                self.dev.stream_synchronize(stream)?;
                std::ptr::copy_nonoverlapping(staging.as_ptr(), out.add(done), chunk);
            }
            Ok(())
        })
    }

    /// Device-to-device: asynchronous on the copy lane, ordered on the GPU
    /// after everything still touching `src` or `dest`, and published as the
    /// new producer of both (the copy reads `src`, so a later host write to
    /// it must wait). Every later launch waits the copy. An overlapping
    /// range within one allocation (memory planning) bounces through a
    /// temporary so it keeps memmove semantics.
    fn _transfer(&self, dest: &RawBuffer, dest_off: usize, src: &RawBuffer, src_off: usize, sz: usize) -> Result<()> {
        if !matches!((dest, src), (RawBuffer::Cuda { .. }, RawBuffer::Cuda { .. })) {
            return UnsupportedSnafu { op: "transfer" }.fail();
        }
        let (dst_base, ..) = self.cuda_buffer(dest);
        let (src_base, ..) = self.cuda_buffer(src);
        let dst = dst_base + dest_off as u64;
        let source = src_base + src_off as u64;
        if sz == 0 || dst == source {
            return Ok(());
        }
        let api = self.dev.api();
        let stream = self.dev.copy_stream();
        let overlaps = dst < source + sz as u64 && source < dst + sz as u64;
        let bounce = overlaps
            .then(|| self._alloc(sz, &BufferSpec { cpu_access: false, ..BufferSpec::default() }, false))
            .transpose()?;
        let mut bases: smallvec::SmallVec<[u64; 3]> = smallvec::smallvec![dst_base];
        if src_base != dst_base {
            bases.push(src_base);
        }
        let result = self.dev.with_copy_lane(|dev| {
            dev.order_copies_after(&bases)?;
            match &bounce {
                Some(bounce) => {
                    let (tmp, ..) = self.cuda_buffer(bounce);
                    bases.push(tmp);
                    // SAFETY: three device ranges of `sz` bytes each, bounded by the caller.
                    unsafe {
                        dev.check((api.memcpy_dtod_async)(tmp, source, sz, stream), "cuMemcpyDtoDAsync")?;
                        dev.check((api.memcpy_dtod_async)(dst, tmp, sz, stream), "cuMemcpyDtoDAsync")?;
                    }
                }
                // SAFETY: two device ranges bounded by the caller.
                None => unsafe {
                    dev.check((api.memcpy_dtod_async)(dst, source, sz, stream), "cuMemcpyDtoDAsync")?;
                },
            }
            dev.record_copy(&bases)
        });
        if let Some(bounce) = bounce {
            // Waits the copies out of the bounce before releasing it.
            self._free(bounce, &BufferSpec::default());
        }
        result
    }

    fn synchronize(&self) -> Result<()> {
        self.dev.synchronize()
    }

    fn name(&self) -> &str {
        "CUDA"
    }

    fn device_spec(&self) -> DeviceSpec {
        DeviceSpec::Cuda { device_id: self.device_id }
    }

    fn supports_device_local(&self) -> bool {
        true
    }
}
