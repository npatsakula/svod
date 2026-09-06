//! CPU device implementation with selectable JIT backends.
//!
//! This module provides a Device instance for CPU execution using either:
//! - LLVM IR codegen (default): compiled in-process through libLLVM, falling
//!   back to `clang -x ir` when the library is not found
//! - Clang C codegen: human-readable source, useful for debugging kernels
//!
//! The backend can be selected via:
//! - `SVOD_CPU_BACKEND` environment variable ("clang" or "llvm")
//! - Explicit `create_cpu_device_with_backend()` call

use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use svod_device::Result;
use svod_device::device::{Compiler, Device, Program, ProgramSpec, Renderer, RuntimeFactory};
use svod_device::registry::DeviceRegistry;
use svod_dtype::DeviceSpec;
use svod_ir::UOp;

use crate::LlvmKernel;
use crate::clang::{ClangKernel, ClangToolchain, c_object_flags, compile_c_object, validate_c_object};
use crate::dispatch::KernelCif;
use crate::object_cache::{CompilerIdentity, OBJECT_CACHE_SCHEMA, ObjectCache, ObjectCacheKey};

/// CPU backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CpuBackend {
    /// Clang C codegen backend: C source compiled by `clang -c`.
    Clang,
    /// LLVM IR backend (default): in-process libLLVM, or `clang -x ir` as fallback.
    #[default]
    Llvm,
}

impl CpuBackend {
    /// Accepted `SVOD_CPU_BACKEND` spellings.
    const SPELLINGS: &str = "clang, CLANG, llvm, LLVM";

    /// Parse a `SVOD_CPU_BACKEND` value; `None` for anything but [`Self::SPELLINGS`].
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "clang" | "CLANG" => Some(CpuBackend::Clang),
            "llvm" | "LLVM" => Some(CpuBackend::Llvm),
            _ => None,
        }
    }

    /// Select the backend from `SVOD_CPU_BACKEND`. Unset or empty selects the
    /// default (LLVM); an unrecognised value warns and selects it too.
    pub fn from_env() -> Self {
        let value = std::env::var_os("SVOD_CPU_BACKEND").unwrap_or_default();
        let value = value.to_string_lossy();
        if value.is_empty() {
            return Self::default();
        }
        Self::parse(&value).unwrap_or_else(|| {
            tracing::warn!(
                %value,
                accepted = Self::SPELLINGS,
                "unrecognised SVOD_CPU_BACKEND, using the default {:?} backend",
                Self::default()
            );
            Self::default()
        })
    }
}

// =============================================================================
// Shared parallel execution
// =============================================================================

/// Size rayon's global pool — the one that runs `core_id`-split CPU kernels
/// and parallel kernel preparation — to `threads`. Rayon builds its global
/// pool once, so the first caller wins; a later call asking for a different
/// size keeps the existing pool and warns once.
pub fn ensure_thread_pool(threads: usize) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    if rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().is_err() {
        let current = rayon::current_num_threads();
        if current != threads {
            WARNED.call_once(|| {
                tracing::warn!(requested = threads, current, "rayon's global pool is already sized; keeping it")
            });
        }
    }
}

/// Execute a kernel function pointer in parallel across multiple threads.
///
/// # Safety
///
/// Buffer safety is guaranteed by the shift_to() transformation:
/// - Each core_id maps to disjoint output indices
/// - Index formula: `output[core_id * chunk_size + local_idx]`
///
/// Same buffer pointers can be safely passed to all threads because:
/// 1. Input buffers: Read-only access (no data race)
/// 2. Output buffers: Disjoint write regions per thread
unsafe fn execute_parallel(
    cif: &KernelCif,
    fn_ptr: *const (),
    buffers: &[*mut u8],
    vals: &[i64],
    var_names: &[String],
    core_count: usize,
) -> Result<()> {
    use rayon::prelude::*;

    let core_id_idx = var_names.iter().position(|n| n == "core_id").ok_or_else(|| svod_device::Error::Runtime {
        message: "parallel CPU launch requires core_id runtime variable".to_string(),
    })?;
    let fn_ptr_usize = fn_ptr as usize;

    // Convert raw pointers to usize for Send-safe cross-thread sharing.
    // Safety: buffer pointers are read-only and point to disjoint write
    // regions per thread (guaranteed by shift_to transformation).
    let buf_ptr = buffers.as_ptr() as usize;
    let buf_len = buffers.len();

    // Nested parallelism policy: if we're already inside rayon work, avoid
    // spawning another parallel loop for core_id kernels.
    if rayon::current_thread_index().is_some() {
        for core_id in 0..core_count {
            let bufs = unsafe { std::slice::from_raw_parts(buf_ptr as *const *mut u8, buf_len) };
            unsafe {
                cif.dispatch(fn_ptr_usize as *const (), bufs, vals, Some((core_id_idx, core_id)))?;
            }
        }
        return Ok(());
    }

    (0..core_count).into_par_iter().try_for_each(|core_id| -> Result<()> {
        let bufs = unsafe { std::slice::from_raw_parts(buf_ptr as *const *mut u8, buf_len) };
        unsafe {
            cif.dispatch(fn_ptr_usize as *const (), bufs, vals, Some((core_id_idx, core_id)))?;
        }
        Ok(())
    })?;

    Ok(())
}

// =============================================================================
// Shared kernel execution
// =============================================================================

/// Execute a kernel: parallel if global_size > 1, otherwise single-threaded.
unsafe fn execute_kernel(
    cif: &KernelCif,
    fn_ptr: *const (),
    buffers: &[*mut u8],
    vals: &[i64],
    var_names: &[String],
    global_size: Option<[usize; 3]>,
) -> Result<()> {
    let core_count = global_size.map(|[tc, _, _]| tc).filter(|&tc| tc > 1);
    if let Some(count) = core_count {
        unsafe { execute_parallel(cif, fn_ptr, buffers, vals, var_names, count) }
    } else {
        unsafe { cif.dispatch(fn_ptr, buffers, vals, None)? };
        Ok(())
    }
}

// =============================================================================
// Clang Backend
// =============================================================================

/// Clang program wrapper implementing the Program trait.
struct ClangProgram {
    kernel: ClangKernel,
}

impl Program for ClangProgram {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> Result<()> {
        unsafe {
            execute_kernel(self.kernel.cif(), self.kernel.fn_ptr(), buffers, vals, self.kernel.var_names(), global_size)
        }
    }

    fn name(&self) -> &str {
        self.kernel.name()
    }
}

/// Clang renderer wrapper implementing the Renderer trait.
struct ClangRendererWrapper {
    device: DeviceSpec,
}

fn renderer_supported_ops() -> svod_ir::RendererOps {
    let mut ops = svod_ir::RendererOps::all();
    ops.binary.remove(&svod_ir::BinaryOp::Threefry);
    ops.binary.remove(&svod_ir::BinaryOp::Max);
    ops
}

fn llvm_renderer_supported_ops() -> svod_ir::RendererOps {
    let mut ops = renderer_supported_ops();
    ops.unary.remove(&svod_ir::UnaryOp::Erf);
    ops
}

impl Renderer for ClangRendererWrapper {
    fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> Result<ProgramSpec> {
        let rendered = svod_codegen::c::render(ast, name.or(Some("kernel")))
            .map_err(|e| svod_device::Error::Runtime { message: format!("C rendering failed: {}", e) })?;

        Ok(super::program_spec(&rendered, &self.device, ast))
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }

    fn supported_ops(&self) -> svod_ir::RendererOps {
        renderer_supported_ops()
    }
}

/// Clang compiler. Unlike LLVM JIT, this boundary emits reusable object bytes;
/// the runtime factory only validates and loads those bytes.
struct ClangCompiler {
    cache: Option<Arc<ObjectCache>>,
    toolchain: ClangToolchain,
    flags: Vec<String>,
    identity: CompilerIdentity,
    cache_key: String,
}

impl Compiler for ClangCompiler {
    fn compile(&self, spec: &ProgramSpec) -> Result<svod_device::device::CompiledSpec> {
        let key = ObjectCacheKey::new(spec.src.as_bytes(), self.identity.clone());
        let bytes = if let Some(cache) = &self.cache {
            cache.get_or_compile(
                &key,
                |bytes| validate_c_object(bytes, &spec.name),
                || compile_c_object(&self.toolchain, &spec.src, &self.flags),
            )
        } else {
            compile_c_object(&self.toolchain, &spec.src, &self.flags)
                .and_then(|bytes| validate_c_object(&bytes, &spec.name).map(|()| bytes))
        }
        .map_err(runtime_as_device)?;
        let mut compiled = svod_device::device::CompiledSpec::from_bytes(
            spec.name.clone(),
            bytes,
            spec.ast.clone(),
            spec.abi.clone(),
        )?;
        compiled.global_size = spec.global_size.clone();
        compiled.local_size = spec.local_size.clone();
        Ok(compiled)
    }

    fn cache_key(&self) -> &str {
        &self.cache_key
    }
}

/// Runtime factory for creating Clang programs.
fn create_clang_program(spec: &svod_device::device::CompiledSpec) -> Result<Box<dyn Program>> {
    svod_device::device::validate_abi_descriptors(&spec.abi, spec.buf_count, &spec.var_names)?;
    if spec.bytes.is_empty() {
        return Err(svod_device::Error::Runtime { message: "Clang backend requires compiled object bytes".into() });
    }
    let kernel = ClangKernel::load_object_with_abi(&spec.bytes, &spec.name, spec.var_names.clone(), &spec.abi)
        .map_err(|e| svod_device::Error::Runtime { message: format!("Clang object load failed: {e}") })?;

    Ok(Box::new(ClangProgram { kernel }))
}

// =============================================================================
// LLVM Backend
// =============================================================================

/// LLVM program wrapper implementing the Program trait.
struct LlvmProgram {
    kernel: LlvmKernel,
}

impl Program for LlvmProgram {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> Result<()> {
        unsafe {
            execute_kernel(self.kernel.cif(), self.kernel.fn_ptr(), buffers, vals, self.kernel.var_names(), global_size)
        }
    }

    fn name(&self) -> &str {
        self.kernel.name()
    }
}

/// LLVM-text compiler; objects come from libLLVM in process or from clang.
struct LlvmCompiler {
    cache: Option<Arc<ObjectCache>>,
    producer: crate::llvm::LlvmObjectProducer,
    identity: CompilerIdentity,
    cache_key: String,
}

impl Compiler for LlvmCompiler {
    fn compile(&self, spec: &svod_device::device::ProgramSpec) -> Result<svod_device::device::CompiledSpec> {
        let key = ObjectCacheKey::new(spec.src.as_bytes(), self.identity.clone());
        let bytes = if let Some(cache) = &self.cache {
            cache.get_or_compile(
                &key,
                |bytes| crate::clang::validate_relocatable_object(bytes, &spec.name),
                || self.producer.compile(&spec.src),
            )
        } else {
            self.producer
                .compile(&spec.src)
                .and_then(|bytes| crate::clang::validate_relocatable_object(&bytes, &spec.name).map(|()| bytes))
        }
        .map_err(runtime_as_device)?;
        let mut compiled = svod_device::device::CompiledSpec::from_bytes(
            spec.name.clone(),
            bytes,
            spec.ast.clone(),
            spec.abi.clone(),
        )?;
        compiled.global_size = spec.global_size.clone();
        compiled.local_size = spec.local_size.clone();
        Ok(compiled)
    }

    fn cache_key(&self) -> &str {
        &self.cache_key
    }
}

/// LLVM renderer wrapper implementing the Renderer trait.
struct LlvmRendererWrapper {
    device: DeviceSpec,
}

impl Renderer for LlvmRendererWrapper {
    fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> Result<ProgramSpec> {
        let rendered = svod_codegen::llvm::text::render(ast, name.or(Some("kernel")))
            .map_err(|e| svod_device::Error::Runtime { message: format!("LLVM rendering failed: {}", e) })?;

        Ok(super::program_spec(&rendered, &self.device, ast))
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }

    fn supported_ops(&self) -> svod_ir::RendererOps {
        llvm_renderer_supported_ops()
    }

    fn extra_matcher(&self) -> Option<svod_ir::pattern::TypedPatternMatcher<()>> {
        Some(svod_codegen::llvm::cpu_extra_matcher())
    }
}

/// Runtime factory for creating LLVM programs.
fn create_llvm_program(spec: &svod_device::device::CompiledSpec) -> Result<Box<dyn Program>> {
    svod_device::device::validate_abi_descriptors(&spec.abi, spec.buf_count, &spec.var_names)?;
    if spec.bytes.is_empty() {
        return Err(svod_device::Error::Runtime { message: "LLVM backend requires compiled object bytes".into() });
    }
    let kernel =
        crate::LlvmKernel::load_object_with_abi(&spec.bytes, &spec.name, &spec.name, spec.var_names.clone(), &spec.abi)
            .map_err(|e| svod_device::Error::Runtime { message: format!("LLVM JIT compilation failed: {}", e) })?;

    Ok(Box::new(LlvmProgram { kernel }))
}

// =============================================================================
// Public API
// =============================================================================

/// Create a CPU device with the default backend.
///
/// The backend is selected by [`CpuBackend::from_env`]: `SVOD_CPU_BACKEND`
/// when set to an accepted spelling, otherwise LLVM.
pub fn create_cpu_device(registry: &DeviceRegistry) -> Result<Device> {
    create_cpu_device_with_backend(registry, CpuBackend::from_env())
}

/// Create a CPU device with a specific backend.
pub fn create_cpu_device_with_backend(registry: &DeviceRegistry, backend: CpuBackend) -> Result<Device> {
    let device_spec = DeviceSpec::Cpu;
    let allocator = registry.get(&device_spec)?;
    let (renderer, compiler) = create_cpu_codegen(backend)?;
    let runtime: RuntimeFactory = match backend {
        CpuBackend::Clang => Arc::new(create_clang_program),
        CpuBackend::Llvm => Arc::new(create_llvm_program),
    };
    Ok(Device::new(device_spec, allocator, renderer, compiler, runtime))
}

/// CPU devices memoized per backend, for the process-global allocator registry.
///
/// Building a CPU device probes the clang toolchain (`ClangToolchain::discover`
/// plus `target_identity`), which costs ~20 ms. Callers that resolve a device
/// per schedule item must not pay that per item.
static CPU_DEVICES: Lazy<RwLock<HashMap<CpuBackend, Arc<Device>>>> = Lazy::new(Default::default);

/// Get or create the shared CPU device for `backend`.
///
/// Repeated calls with the process-global allocator registry return the same
/// `Arc`; distinct backends get distinct devices. Any other registry bypasses
/// the cache, since a cached device holds the allocators it was built with.
pub fn cpu_device_with_backend(registry: &DeviceRegistry, backend: CpuBackend) -> Result<Arc<Device>> {
    if !std::ptr::eq(registry, svod_device::registry::registry()) {
        return Ok(Arc::new(create_cpu_device_with_backend(registry, backend)?));
    }
    if let Some(device) = CPU_DEVICES.read().get(&backend) {
        return Ok(Arc::clone(device));
    }
    let mut devices = CPU_DEVICES.write();
    if let Some(device) = devices.get(&backend) {
        return Ok(Arc::clone(device));
    }
    let device = Arc::new(create_cpu_device_with_backend(registry, backend)?);
    devices.insert(backend, Arc::clone(&device));
    Ok(device)
}

/// Construct CPU renderer/compiler components without creating an allocator or
/// executable runtime. Clean BEAM workers use this device-disabled path.
pub fn create_cpu_codegen(backend: CpuBackend) -> Result<(Arc<dyn Renderer>, Arc<dyn Compiler>)> {
    let device_spec = DeviceSpec::Cpu;
    match backend {
        CpuBackend::Clang => {
            let cache = ObjectCache::from_env().map_err(runtime_as_device)?.map(Arc::new);
            let toolchain = ClangToolchain::discover(cache.as_deref()).map_err(runtime_as_device)?;
            let flags = c_object_flags();
            let target_architecture = toolchain.target_identity(cache.as_deref(), &flags).map_err(runtime_as_device)?;
            let identity = CompilerIdentity {
                schema: OBJECT_CACHE_SCHEMA,
                backend: "cpu-clang".into(),
                target_architecture,
                toolchain: toolchain.identity().into(),
                flags: flags.clone(),
                abi: format!(
                    "svod-c-kernel-abi-v1;pointer-width={};endian={}",
                    usize::BITS,
                    if cfg!(target_endian = "little") { "little" } else { "big" }
                ),
                object_format: if cfg!(feature = "dlopen-fallback") {
                    "elf-shared-dlopen-v1".into()
                } else {
                    "elf-relocatable-svod-jit-loader-v1".into()
                },
            };
            let cache_key = identity.cache_key();
            Ok((
                Arc::new(ClangRendererWrapper { device: device_spec }),
                Arc::new(ClangCompiler { cache, toolchain, flags, identity, cache_key }),
            ))
        }
        CpuBackend::Llvm => {
            let cache = ObjectCache::from_env().map_err(runtime_as_device)?.map(Arc::new);
            let (producer, identity) =
                crate::llvm::llvm_object_producer(cache.as_deref()).map_err(runtime_as_device)?;
            let cache_key = identity.cache_key();
            Ok((
                Arc::new(LlvmRendererWrapper { device: device_spec }),
                Arc::new(LlvmCompiler { cache, producer, identity, cache_key }),
            ))
        }
    }
}

fn runtime_as_device(error: crate::Error) -> svod_device::Error {
    svod_device::Error::Runtime { message: error.to_string() }
}
