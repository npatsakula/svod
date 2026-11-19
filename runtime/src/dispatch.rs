//! Kernel dispatch via libffi.
//!
//! `KernelCif` wraps libffi's `Cif` with Send+Sync so it can be stored on
//! kernel structs and shared across rayon threads. Thread-local buffers avoid
//! per-call allocation for the packed arguments.

use std::cell::RefCell;

use libffi::low::CodePtr;
use libffi::middle::{self, Cif, Type};
use smallvec::SmallVec;

/// Send+Sync wrapper for libffi Cif.
///
/// # Safety
///
/// `Cif` is `!Send + !Sync` due to raw pointer fields (conservative auto-trait).
/// Once prepared, a CIF is immutable — `Cif::call(&self)` only reads the
/// descriptor and `ffi_call` does not mutate it for non-closure calls.
/// All our CIFs describe stateless kernel signatures (pointers + i32 → void).
pub(crate) struct KernelCif {
    cif: Cif,
    arg_count: usize,
    /// Per argument: a pointer (storage) or an `i32` scalar.
    abi: Vec<bool>,
}

unsafe impl Send for KernelCif {}
unsafe impl Sync for KernelCif {}

impl KernelCif {
    pub fn from_abi(abi: &[svod_device::device::AbiParamDescriptor]) -> svod_device::Result<Self> {
        svod_device::device::reject_tensor_maps(abi, "CPU")?;
        let types =
            abi.iter().map(|arg| if arg.is_storage() { Type::pointer() } else { Type::i32() }).collect::<Vec<_>>();
        Ok(Self {
            cif: Cif::new(types, Type::void()),
            arg_count: abi.len(),
            abi: abi.iter().map(|arg| arg.is_storage()).collect(),
        })
    }

    /// Call the kernel, packing buffers and scalars into slots that match the
    /// CIF's declared argument types.
    ///
    /// libffi reads each argument through a pointer using the declared type's
    /// width, so a `Type::i32()` argument must point at an `i32`. Packing every
    /// slot into a `u64` happened to work only because a little-endian read of
    /// the low four bytes is the truncation we want; on a big-endian host it
    /// reads the high half and every scalar arrives as 0.
    ///
    /// Uses thread-local buffers for the packed args — zero allocation after
    /// warmup. The `SmallVec<[Arg; 32]>` avoids heap allocation for kernels
    /// with ≤32 arguments (the common case); kernels above that cap fall back
    /// to a heap allocation per dispatch.
    ///
    /// `var_patch`: if `Some((var_idx, value))`, patches
    /// `vals[var_idx]` to `value` before calling.
    #[inline]
    pub unsafe fn dispatch(
        &self,
        fn_ptr: *const (),
        buffers: &[*mut u8],
        vals: &[i64],
        var_patch: Option<(usize, usize)>,
    ) -> svod_device::Result<()> {
        let expected_buffers = self.abi.iter().filter(|&&storage| storage).count();
        let expected_vals = self.arg_count - expected_buffers;
        if buffers.len() != expected_buffers || vals.len() != expected_vals {
            return Err(svod_device::Error::ProgramAbiMismatch {
                reason: format!(
                    "kernel dispatch expected {expected_buffers} buffers/{expected_vals} scalars, got {}/{}",
                    buffers.len(),
                    vals.len()
                ),
            });
        }

        thread_local! {
            static PACKED: RefCell<PackedArgs> = RefCell::new(PackedArgs::default());
        }

        PACKED.with_borrow_mut(|packed| {
            packed.ptrs.clear();
            packed.scalars.clear();
            let (mut buffer_idx, mut var_idx) = (0usize, 0usize);
            for &storage in &self.abi {
                if storage {
                    packed.ptrs.push(buffers[buffer_idx]);
                    buffer_idx += 1;
                } else {
                    packed.scalars.push(vals[var_idx] as i32);
                    var_idx += 1;
                }
            }

            if let Some((var_idx, value)) = var_patch {
                let Some(slot) = packed.scalars.get_mut(var_idx) else { return };
                *slot = value as i32;
            }

            let mut ffi_args: SmallVec<[middle::Arg; 32]> = SmallVec::with_capacity(self.arg_count);
            let (mut buffer_idx, mut var_idx) = (0usize, 0usize);
            for &storage in &self.abi {
                if storage {
                    ffi_args.push(middle::arg(&packed.ptrs[buffer_idx]));
                    buffer_idx += 1;
                } else {
                    ffi_args.push(middle::arg(&packed.scalars[var_idx]));
                    var_idx += 1;
                }
            }

            unsafe {
                self.cif.call::<()>(CodePtr(fn_ptr as *mut _), &ffi_args);
            }
        });
        if let Some((var_idx, _)) = var_patch
            && var_idx >= expected_vals
        {
            return Err(svod_device::Error::ProgramAbiMismatch {
                reason: format!(
                    "kernel dispatch scalar patch index {var_idx} out of range for {expected_vals} scalars"
                ),
            });
        }
        Ok(())
    }
}

/// Typed argument slots. libffi dereferences each slot with the width the CIF
/// declares, so pointers and `i32` scalars need their own storage.
#[derive(Default)]
struct PackedArgs {
    ptrs: SmallVec<[*mut u8; 16]>,
    scalars: SmallVec<[i32; 16]>,
}
