//! [`HostWrite`]: exclusive host stores into a buffer between launches.

use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;

use super::Buffer;
use crate::allocator::RawBuffer;
use crate::error::{HostWriteLiveSnafu, HostWriteRowSnafu, MisalignedHostWriteSnafu, Result, SizeMismatchSnafu};

/// Exclusive host write access to a buffer as `[T]`, from
/// [`Buffer::host_write`]. While it lives, no other guard, host read or copy
/// of the same storage succeeds through any handle or view, and a plan that
/// binds the storage refuses to launch. Dropping it publishes the stores: on
/// AMD VRAM, the device's next doorbell flushes them out of HDP.
///
/// Host-visible VRAM is write-combined over the PCIe BAR: stores stream, but
/// every read is an uncached round trip. Prefer the bulk writers
/// ([`copy_from`](Self::copy_from), [`copy_at`](Self::copy_at),
/// [`fill`](Self::fill), [`rows`](Self::rows)) to read-modify-write through
/// [`as_mut_slice`](Self::as_mut_slice).
pub struct HostWrite<'b, T> {
    buffer: &'b Buffer,
    ptr: NonNull<T>,
    len: usize,
    _elements: PhantomData<&'b mut [T]>,
}

impl<'b, T: Copy> HostWrite<'b, T> {
    pub(super) fn acquire(buffer: &'b mut Buffer) -> Result<Self> {
        buffer.ensure_mutable("host_write")?;
        buffer.ensure_allocated()?;
        let storage = buffer.data.storage_id.0;
        let claimed = buffer.data.host_writer.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed);
        snafu::ensure!(claimed.is_ok(), HostWriteLiveSnafu { op: "host_write", storage });
        let buffer = &*buffer;
        let ptr = buffer.host_write_ptr().and_then(|ptr| {
            let align = align_of::<T>();
            snafu::ensure!(
                (ptr as usize).is_multiple_of(align),
                MisalignedHostWriteSnafu { offset: buffer.offset, align }
            );
            Ok(NonNull::new(ptr.cast::<T>()).expect("host mappings are non-null"))
        });
        match ptr {
            Ok(ptr) => Ok(Self { buffer, ptr, len: buffer.size / size_of::<T>(), _elements: PhantomData }),
            Err(error) => {
                buffer.data.host_writer.store(false, Ordering::Release);
                Err(error)
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Overwrite the whole buffer with `src`, which must be as long.
    pub fn copy_from(&mut self, src: &[T]) -> Result<()> {
        snafu::ensure!(src.len() == self.len, SizeMismatchSnafu { expected: self.len, actual: src.len() });
        self.as_mut_slice().copy_from_slice(src);
        Ok(())
    }

    /// Overwrite the elements from `offset` on with `src`.
    pub fn copy_at(&mut self, offset: usize, src: &[T]) -> Result<()> {
        let end = offset.saturating_add(src.len());
        snafu::ensure!(end <= self.len, SizeMismatchSnafu { expected: self.len, actual: end });
        self.as_mut_slice()[offset..end].copy_from_slice(src);
        Ok(())
    }

    pub fn fill(&mut self, value: T) {
        self.as_mut_slice().fill(value);
    }

    /// Store each `(index, value)`. Every index is checked before any store.
    pub fn rows(&mut self, rows: &[(usize, T)]) -> Result<()> {
        if let Some(&(row, _)) = rows.iter().find(|(row, _)| *row >= self.len) {
            return HostWriteRowSnafu { row, len: self.len }.fail();
        }
        let slice = self.as_mut_slice();
        for &(row, value) in rows {
            slice[row] = value;
        }
        Ok(())
    }

    /// The buffer as a slice. Reading it back is slow on host-visible VRAM.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: the storage's writer flag makes this the only host access to
        // `len` elements the allocation keeps mapped while `buffer` lives.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for HostWrite<'_, T> {
    fn drop(&mut self) {
        // Mark before releasing the flag: a launch that sees the flag clear
        // also sees the deferred flush.
        if let Some(RawBuffer::AmdDevice { vram: true, device, .. }) = self.buffer.data.raw.get() {
            device.core().hdp().defer();
        }
        self.buffer.data.host_writer.store(false, Ordering::Release);
    }
}

impl<T> std::fmt::Debug for HostWrite<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostWrite").field("storage", &self.buffer.storage_id()).field("len", &self.len).finish()
    }
}
