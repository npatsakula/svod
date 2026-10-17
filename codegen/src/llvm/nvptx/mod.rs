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
pub mod wgmma;
pub mod wmma;

pub use ops::{ShflMode, globaltimer, render_uop, shfl, shfl_bfly, shfl_down, shfl_idx, shfl_up};

/// The PTX ISA the emitted module declares, hence the oldest driver that
/// JITs it: 7.8 (CUDA 11.8 / R520) carries every `mma.sync` shape the
/// profiles select up to sm_90; the fp8 shapes of sm_89 and newer exist from
/// 8.4 (CUDA 12.4 / R550); each Blackwell target then needs the ISA that
/// introduced it (8.6 for sm_100/101, 8.7 for sm_120, 8.8 for sm_103/121),
/// and anything newer the latest one clang knows. Pinned rather than left to
/// clang, whose default follows whatever CUDA toolkit it finds (none: too old
/// for any tensor core).
pub fn ptx_isa(arch: svod_dtype::CudaArch) -> u32 {
    match (arch.major, arch.minor) {
        (10, 0..=2) => 86,
        (12, 0) => 87,
        (10.., _) => 88,
        _ if arch.has_fp8() => 84,
        _ => 78,
    }
}

/// The `-march` of `arch`: Hopper takes its architecture-specific feature
/// set (`sm_90a`), which the warpgroup products of [`wgmma`] need and which
/// runs on sm_90 alone, where every kernel is JIT-compiled for the device
/// at hand anyway.
pub fn march(arch: svod_dtype::CudaArch) -> String {
    match (arch.major, arch.minor) {
        (9, 0) => "sm_90a".to_string(),
        _ => arch.to_string(),
    }
}

/// Clang flags compiling rendered IR on stdin to PTX text on stdout for
/// `arch`. `-Wno-override-module` silences the note about the module's own
/// `target triple` (the renderer sets it to match).
pub fn clang_flags(arch: svod_dtype::CudaArch) -> Vec<String> {
    let mut flags: Vec<String> = ["-x", "ir", "-S", "-O3", "--target=nvptx64-nvidia-cuda"].map(str::to_string).into();
    flags.push(format!("-march={}", march(arch)));
    flags.push(format!("--cuda-feature=+ptx{}", ptx_isa(arch)));
    flags.extend(["-Wno-override-module", "-", "-o", "-"].map(str::to_string));
    flags
}
pub use smem::{CpAsyncCache, cp_async, cp_async_16, cp_async_commit, cp_async_wait, cp_async_wait_all, ldmatrix};
