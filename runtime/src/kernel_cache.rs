//! Global kernel deduplication cache.
//!
//! This module provides a global concurrent cache that maps (program content hash,
//! compiler key) pairs to compiled kernels.
//! Uses papaya's lock-free HashMap for thread-safe access across parallel tensor operations.
//!
//! # Thread Safety
//!
//! All operations are thread-safe. Multiple threads can look up and compile kernels
//! concurrently without explicit synchronization.
//!
//! # Deduplication
//!
//! Callers key by the structural content hash of the rendered program
//! (`UOp::content_hash`) plus the compiler's `cache_key()`, so
//! identical programs share one compiled kernel while different compilers
//! (backend, target arch, options) never collide.

use std::sync::{Arc, OnceLock};

use papaya::HashMap;
use svod_device::device::Program;
use svod_ir::UOp;

/// Cached kernel that can be reused across tensors.
///
/// Note: This struct does not implement Clone because `Box<dyn Program>` is not Clone.
/// Use `Arc<CachedKernel>` for sharing.
pub struct CachedKernel {
    /// The compiled, executable program.
    pub program: Box<dyn Program>,
    /// Device string (e.g., "CPU", "CUDA:0").
    pub device: String,
    /// Generated source code (for debugging/profiling).
    pub code: String,
    /// Entry point name.
    pub entry_point: String,
    /// Variable names in order for converting HashMap to positional vals.
    /// Matches the order expected by the compiled program.
    pub var_names: Vec<String>,
    /// Global buffer slots in kernel argument order.
    /// Matches Tinygrad's ProgramSpec.globals semantics.
    pub globals: Vec<usize>,
    /// Output buffer slots written by STORE operations.
    /// Matches Tinygrad's ProgramSpec.outs semantics.
    pub outs: Vec<usize>,
    /// Input buffer slots read by LOAD operations.
    /// Matches Tinygrad's ProgramSpec.ins semantics.
    pub ins: Vec<usize>,
    /// Symbolic global work size evaluated with runtime vars before dispatch.
    pub global_size: [Arc<UOp>; 3],
    /// Symbolic local work size. None means direct global-id execution.
    pub local_size: Option<[Arc<UOp>; 3]>,
}

/// Cache key: (program content hash, compiler cache key).
type KernelKey = (u64, String);

// Global kernel dedup cache using lock-free concurrent HashMap.
//
// Maps (program hash, compiler key) -> Arc<CachedKernel>.
// Kernels live for the process lifetime — the cache is never torn down.
static KERNELS: OnceLock<HashMap<KernelKey, Arc<CachedKernel>>> = OnceLock::new();

fn kernels() -> &'static HashMap<KernelKey, Arc<CachedKernel>> {
    KERNELS.get_or_init(HashMap::new)
}

/// Get or compile a kernel by program hash and compiler key.
///
/// Thread-safe: if multiple threads call this with the same key concurrently,
/// exactly one will compile the kernel, and all others will receive a clone
/// of the Arc to that kernel.
///
/// # Arguments
///
/// * `program_hash` - Content hash of the rendered program
/// * `compiler_key` - The compiler's `cache_key()`
/// * `compile_fn` - Function to compile the kernel if not cached
///
/// # Returns
///
/// Arc to the cached kernel (either from cache or freshly compiled).
///
/// # Errors
///
/// Returns error if compilation fails
pub fn get_or_compile_kernel<F, E>(program_hash: u64, compiler_key: &str, compile_fn: F) -> Result<Arc<CachedKernel>, E>
where
    F: FnOnce() -> Result<CachedKernel, E>,
{
    let key = (program_hash, compiler_key.to_string());
    let map = kernels();
    let guard = map.guard();

    // Fast path: kernel already cached
    if let Some(cached) = map.get(&key, &guard) {
        return Ok(Arc::clone(cached));
    }

    // Slow path: compile kernel (expensive)
    let compiled = compile_fn()?;
    let cached = Arc::new(compiled);

    // Atomic insert - if another thread beat us, use their kernel
    use papaya::{Compute, Operation};
    match map.compute(
        key,
        |entry| match entry {
            Some((_, existing)) => Operation::Abort(Arc::clone(existing)),
            None => Operation::Insert(Arc::clone(&cached)),
        },
        &guard,
    ) {
        Compute::Inserted(_, kernel) => Ok(Arc::clone(kernel)),
        Compute::Aborted(kernel) => Ok(kernel),
        _ => Ok(cached),
    }
}

// No `clear_all` / `gc_unused_kernels`: the cache is intentionally
// process-static and deduped by `(program_hash, compiler_key)`. Identical programs share an
// `Arc<CachedKernel>` so cross-test interference is moot; a public bulk
// drop would burst `AmdProgram::Drop` (and equivalents) through the cache
// while in-flight dispatches still resolve through it — exactly the
// unmap-while-busy hazard the per-connector cleanup paths already create.
// Programs amortise across the process and the OS reclaims at exit.
