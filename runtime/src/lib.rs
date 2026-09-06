//! Runtime execution for svod kernels.
//!
//! Provides generic kernel execution interface with backend-specific implementations
//! (LLVM JIT, native shared libraries, CUDA, etc.).
//!
//! # Execution Model
//!
//! `execution_plan` is the canonical runtime path and executes prepared
//! operations in dependency order with hazard-aware host parallelism.
//! The `executor` module remains available for explicit parallel scheduling
//! scenarios.
//!
//! CPU work is submitted through `svod_device::hcq`; the compatibility queue
//! type is intentionally absent:
//! ```compile_fail
//! use svod_runtime::CpuQueue;
//! ```
//!
//! # Benchmarking
//!
//! The `benchmark` module provides timing utilities for measuring kernel
//! execution performance, used by beam search auto-tuning.

pub mod amd;
pub mod benchmark;
pub mod clang;
pub mod cuda;
pub mod custom_function;
pub mod device_registry;
pub mod devices;
pub(crate) mod dispatch;
pub mod error;
pub mod execution_plan;
pub mod executor;
pub mod jit_loader;
pub mod kernel_cache;
pub mod leveling;
pub mod llvm;
mod llvm_inprocess;
pub mod object_cache;
pub mod profiler;

#[cfg(test)]
pub mod test;

pub use benchmark::{BenchmarkConfig, BenchmarkResult, benchmark_kernel, warmup_thread_pool};
pub use custom_function::run_custom_function;
pub use device_registry::DEVICE_FACTORIES;
pub use devices::{
    cpu::{
        CpuBackend, cpu_device_with_backend, create_cpu_codegen, create_cpu_device, create_cpu_device_with_backend,
        ensure_thread_pool,
    },
    create_amd_codegen, create_cuda_codegen, create_cuda_device, create_cuda_program, create_metal_codegen,
    create_metal_device,
};
pub use error::*;
pub use execution_plan::{
    ExecutionPlan, ExecutionPlanBuilder, PreparedCopy, PreparedCustomFunction, PreparedKernel, PreparedOp,
};
pub use executor::{
    DeviceContext, ExecutionGraph, ExecutionNode, KernelBufferAccess, SyncStrategy, UnifiedExecutor, global_executor,
};
pub use kernel_cache::*;
pub use leveling::{compute_topological_levels, compute_topological_order};
pub use llvm::*;
pub use profiler::{
    KernelAggregate, KernelExport, KernelProfile, KernelShareExport, KernelStaticInfo, OriginAggregate, OriginExport,
    OriginNodeExport, OriginView, PmcSelection, ProfileExport, ProfileOptions, RunProfile, StageExport, StageProfile,
    UNATTRIBUTED, aggregate_origins, aggregate_profiles, has_origins, render_histogram, render_origins,
};
pub use svod_device::{CounterSet, KernelResources, PmcCounter};
