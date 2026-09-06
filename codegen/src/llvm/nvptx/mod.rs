//! NVIDIA GPU LLVM IR text generation (NVPTX).
//!
//! Composed against [`cpu::render_uop`] as the base, exactly like `amd/`:
//! NVPTX-specific ops (`Special`, `Barrier`, LOCAL buffers, `Log2`, `Wmma`)
//! are intercepted here, everything else (ALU, INDEX, LOAD, STORE, CAST,
//! RANGE) falls through to the CPU emitter unchanged. clang lowers the module
//! to PTX text (`--target=nvptx64-nvidia-cuda -march=sm_XY`), which the CUDA
//! driver JITs at load.
//!
//! `ops` and `smem` also export typed `Op::Custom` builders for the warp and
//! shared-memory primitives a tile kernel composes by hand (`shfl.sync`,
//! `cp.async`, `ldmatrix`); the text renderer refuses them on other targets.
//!
//! [`cpu::render_uop`]: crate::llvm::cpu::render_uop

pub mod ops;
pub mod smem;
pub mod wmma;

pub use ops::{ShflMode, globaltimer, render_uop, shfl, shfl_bfly, shfl_down, shfl_idx, shfl_up};
pub use smem::{CpAsyncCache, cp_async, cp_async_16, cp_async_commit, cp_async_wait, cp_async_wait_all, ldmatrix};
