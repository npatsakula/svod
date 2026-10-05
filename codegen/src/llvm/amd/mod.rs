//! AMD GPU LLVM IR text generation.
//!
//! Composed against [`cpu::render_uop`](crate::llvm::cpu::render_uop) as the base: AMD-specific ops are
//! intercepted here, everything else (ALU, INDEX, LOAD, STORE, CAST, RANGE)
//! falls through to the CPU emitter unchanged.

pub mod ops;
pub mod wmma;

pub use ops::render_uop;
