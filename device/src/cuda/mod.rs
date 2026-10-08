//! NVIDIA GPU support through the CUDA driver API (`libcuda.so.1`).
//!
//! Compiled on every host: the driver is bound at runtime with `libloading`,
//! so where it is absent [`has_devices`] is `false` and the runtime never
//! registers the `CUDA` factory — the same "always compiled, hardware
//! detected at runtime" contract as [`crate::amd`] and [`crate::metal`].
//!
//! Model: one primary context per device; device-local allocations with
//! managed memory where a host mapping is requested; kernels loaded from a
//! `ptxas` cubin or from PTX text JIT-compiled by the driver; one
//! non-blocking stream per execution plan; event pairs for
//! GPU-clock timestamps; captured plans replay as CUDA graphs whose edges are
//! the host hazard analysis; host access waits only the storage's own
//! in-flight producers and readers (scoped synchronization, see `device`).

/// Declares a runtime-bound C API: a doc line, the library name for
/// diagnostics, the result type every entry point returns, the error type
/// `bind` fails with and how a loader message becomes one; then the Rust
/// field name, the exact export resolved with `dlsym`, and the C prototype
/// of each entry point.
macro_rules! dl_api {
    ($doc:literal, $library:expr, $result:ty, $error:ty, $wrap:expr;
     $($field:ident = $symbol:literal: fn($($arg:ty),* $(,)?);)*) => {
        #[doc = $doc]
        pub struct Api {
            $(pub $field: unsafe extern "C" fn($($arg),*) -> $result,)*
            // Declared last so the function pointers never outlive the library.
            _library: Library,
        }

        impl Api {
            fn bind(library: Library) -> std::result::Result<Self, $error> {
                Ok(Self {
                    $($field: crate::error::dlsym(&library, $library, $symbol.as_bytes()).map_err($wrap)?,)*
                    _library: library,
                })
            }
        }

        /// `(Rust name, dlsym symbol)` of every bound entry point.
        pub const SYMBOLS: &[(&str, &str)] = &[$((stringify!($field), $symbol)),*];
    };
}

/// The `.const` `u32` a kernel module exports when its shared memory is
/// dynamic: the bytes to opt the function into and pass at every launch.
pub const DYNAMIC_SHARED_BYTES_SYMBOL: &str = "svod_dynamic_shared_bytes";

pub mod allocator;
pub mod cupti;
pub mod device;
pub mod graph;
pub mod program;
pub mod sync;
#[doc(hidden)]
pub mod sys;

pub use allocator::{CudaAllocator, CudaMemory};
pub use device::{CudaDevice, CudaEvent, CudaLimits, CudaStream, has_devices};
pub use graph::CudaGraph;
pub use program::{CudaProgram, Launch, check_ptx_entry_abi, is_cubin, validate_cubin};
pub use sync::{CudaCompletionToken, CudaDispatchTimestamps, CudaPlanCtx};
