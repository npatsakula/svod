//! AMD GPU runtime support.
//!
//! Splits cleanly into:
//! - [`compile`]: invokes the host `clang` with `--target=amdgcn-amd-amdhsa`
//!   to lower the AMD LLVM renderer output (Phase 1) to an AMDGPU code object.
//! - (later phases) device, allocator, queue, signal, program — KFD-direct
//!   AMD runtime.
//!
//! All entry points are *infallible to compile* on non-Linux hosts; runtime
//! calls return clean `Err(NoAmdGpu)` when there's no AMD GPU.

pub mod compile;

pub use crate::devices::amd::LocatedAmd;
pub use compile::{
    assemble_gcn, compile_ir_to_amd_asm, compile_ir_to_amd_object, compile_ir_to_amd_object_as, has_amdgpu_target,
};

use svod_codegen::llvm::LlvmTextRenderer;
use svod_dtype::AmdArch;

/// The renderer an AMD device renders every kernel with: a host build of a
/// device kernel (an SQTT capture's line-table twin) renders through it, so
/// the two cannot drift apart.
pub fn renderer(arch: AmdArch) -> LlvmTextRenderer {
    LlvmTextRenderer::amd(arch)
}
