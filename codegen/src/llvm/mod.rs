//! LLVM IR code generation.
//!
//! This module generates LLVM IR code from optimized UOp graphs.
//!
//! # Module Structure
//!
//! - `common/`: Shared utilities (types, ctx, target enum, GPU helpers)
//! - `cpu/`: CPU-specific rendering (host x86/AArch64 via clang)
//! - `amd/`: AMD GPU rendering (amdgcn LLVM IR via clang)
//! - `nvptx/`: NVIDIA GPU rendering (nvptx64 LLVM IR → PTX via clang)
//! - `text/`: Main entry point that orchestrates target-aware rendering

pub mod amd;
pub mod common;
pub mod cpu;
pub mod nvptx;
pub mod sched;
pub mod text;

pub use common::LlvmTarget;
pub use cpu::render_uop as cpu_render_uop;
pub use text::LlvmTextRenderer;

pub fn cpu_extra_matcher() -> svod_ir::pattern::TypedPatternMatcher<()> {
    svod_schedule::devectorize::bool_storage_patterns().clone()
}

pub fn amd_extra_matcher() -> svod_ir::pattern::TypedPatternMatcher<()> {
    svod_schedule::devectorize::amd_non_native_fp8_patterns().clone()
}

/// PTX bools are predicate registers but bytes in memory, so NVPTX shares the
/// CPU renderer's byte-storage rewrite (tinygrad `ptx.py` `ptx_matcher`).
pub fn nvptx_extra_matcher() -> svod_ir::pattern::TypedPatternMatcher<()> {
    svod_schedule::devectorize::bool_storage_patterns().clone()
}
