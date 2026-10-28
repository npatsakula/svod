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

pub use compile::{assemble_gcn, compile_ir_to_amd_asm, compile_ir_to_amd_object, has_amdgpu_target};
