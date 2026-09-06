//! NVIDIA GPU runtime support.
//!
//! [`compile`] invokes the host `clang` with `--target=nvptx64-nvidia-cuda`
//! to lower the NVPTX LLVM renderer output to PTX text. With the CUDA
//! toolkit's `ptxas` installed the PTX is assembled to a cubin ahead of load;
//! without it the CUDA driver JITs the text at module load
//! (`svod_device::cuda::CudaProgram`), so no toolkit is required: clang is
//! already needed for the CPU and AMD paths.
//!
//! Everything here compiles on every host; the device factory is only
//! registered when `svod_device::cuda::has_devices()` is true.

pub mod compile;

pub use compile::{compile_ir_to_ptx, has_nvptx_target};
