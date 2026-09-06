//! Device abstraction layer for tensor operations.
//!
//! This module provides a clean abstraction over different compute devices (CPU, AMD, CUDA, Metal, DISK)
//! with support for:
//! - Lazy buffer allocation
//! - Buffer views with zero-copy slicing
//! - LRU caching of allocations for performance
//! - Device-agnostic copy operations
//!
//! # Examples
//!
//! ```no_run
//! use svod_device::allocator::BufferSpec;
//! use svod_device::{Buffer, registry};
//! use svod_dtype::DType;
//!
//! // Get a CPU device
//! let cpu = registry::cpu().unwrap();
//!
//! // Create a buffer with lazy allocation
//! let buffer = Buffer::new(cpu, DType::Float32, vec![10, 10], BufferSpec::default());
//!
//! // Allocation happens on first use
//! buffer.ensure_allocated().unwrap();
//! ```
//!
//! The pre-HCQ queue API is intentionally absent:
//! ```compile_fail
//! use svod_device::{DynQueue, HardwareQueue};
//! ```

pub mod allocator;
pub mod amd;
pub mod buffer;
pub mod cuda;
pub mod device;
pub mod error;
pub mod hcq;
pub mod inprocess;
pub mod isa;
pub mod metal;
pub mod profile;
pub mod registry;
pub mod sync;

pub use buffer::{Buffer, BufferId};
pub use device::{
    CopyEndpoint, Graph, GraphFactory, GraphKernel, NativeReplayDecline, NativeReplayOutcome, PlanCall, PlanContext,
    Program,
};
pub use error::{Error, Result};
pub use inprocess::claim_inprocess_llvm;
pub use profile::{CounterSet, KernelResources, PmcCounter};
pub use sync::{CompletionToken, CpuTimelineSignal, DispatchTimestamps, TimelineSignal};

#[cfg(test)]
mod test;

// Re-export commonly used types
pub use allocator::{Allocator, BufferSpec, CpuAllocator};
pub use registry::{DeviceSpec, cpu, get_device};
