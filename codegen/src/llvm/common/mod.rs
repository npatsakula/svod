//! Common utilities for LLVM IR text generation.
//!
//! Shared between the CPU and GPU backends.

mod ctx;
pub mod gpu;
pub mod target;
pub mod types;

pub use ctx::RenderContext;
pub use target::{LlvmTarget, addr_space_num as target_addr_space_num};
pub use types::{addr_space_num, lcast, lconst, ldt};
