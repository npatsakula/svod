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

pub mod allocator;
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
