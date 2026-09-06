//! AMD GPU device factory.
//!
//! Wires together:
//! - `svod_codegen::llvm::LlvmTextRenderer::amd(arch)` for IR emission.
//! - `svod_runtime::amd::compile_ir_to_amd_object` for clang amdgcn compile.
//! - `svod_device::amd::AmdProgram` for ELF load + AQL dispatch.
//!
//! Construction returns `Err(NoAmdGpu)` cleanly on hosts that don't have a
//! supported AMD GPU; never panics.

use std::sync::Arc;

use svod_codegen::llvm::LlvmTextRenderer;
use svod_device::Result;
use svod_device::amd::{AmdAllocator, AmdCopyQueue, AmdGraph, AmdProgram, SignalPool};
use svod_device::device::{
    CompiledSpec, Compiler, Device, Graph, GraphFactory, GraphKernel, Program, ProgramSpec, Renderer, RuntimeFactory,
};
use svod_device::registry::DeviceRegistry;
use svod_dtype::{AmdArch, DeviceSpec};
use svod_ir::UOp;

use crate::clang::ClangToolchain;
use crate::object_cache::{CompilerIdentity, OBJECT_CACHE_SCHEMA, ObjectCache, ObjectCacheKey};

/// Create an `AMD:N` device end-to-end (allocator + renderer + compiler +
/// runtime). The arch is queried from KFD topology at device-open time and
/// stored on the opened `AmdDevice` (NOT in the `DeviceSpec`). The
/// `arch` parameter is the cache-key hint — kept so the compiler can emit
/// the right `-mcpu`.
pub fn create_amd_device(registry: &DeviceRegistry, device_id: usize, arch: AmdArch) -> Result<Device> {
    let spec = DeviceSpec::Amd { device_id };
    let allocator = registry.get(&spec)?;
    let (renderer, compiler) = create_amd_codegen(device_id, arch)?;
    // Build the per-device process-shared state: the signal pool (singleton
    // per physical AMD:N, lives on AmdDeviceCore). Each `ExecutionPlan` /
    // `AmdGraph` / per-call `Program::execute` leases or builds its OWN
    // connector (own KFD ring + kernarg arena + scratch + timeline), so no
    // compute-queue or arena is pre-built here.
    let amd_alloc = AmdAllocator::new(device_id)?;
    let device_handle = Arc::clone(&amd_alloc.dev);
    // Signal-pool sizing: per-op AQL dispatch needs only a few slots, but a
    // captured DAG graph reserves one slot per kernel (low hundreds) for its
    // lifetime, across several concurrent owners. 1024 slots (64 KiB GTT) covers
    // that with headroom; the pool rounds up to whole 64-slot pages.
    // Sized for the worst combination: a captured graph reserves a slot per
    // kernel for its lifetime while a profiled execution holds every
    // dispatch's signal until harvest. Slots are 64 B each — 4096 is 256 KiB
    // of GTT, cheap insurance against `SignalPool exhausted`.
    let signal_pool = SignalPool::new(&amd_alloc, 4096)?;
    // Seed the pool onto the device core so `PoolQueue::new_with_resources`
    // can acquire its PM4 counter signal.
    device_handle.core().install_signal_pool(signal_pool);
    // Bring up SDMA on CDNA so buffers can be device-local. Svod's direct KFD
    // SDMA queue is not safe alongside its PM4 queue on RDNA yet, so RDNA uses
    // the existing host-visible memmove path. Must decide before any _alloc,
    // which reads has_sdma_queue to select buffer visibility.
    let copy_queue = if !arch.is_cdna() || std::env::var_os("AMD_DISABLE_SDMA").is_some() {
        None
    } else {
        Some(AmdCopyQueue::create(&amd_alloc))
    };
    match copy_queue {
        Some(Ok(copy_queue)) => {
            device_handle.core().install_copy_queue(copy_queue);
            device_handle.core().set_has_sdma_queue(true);
        }
        Some(Err(e)) => {
            tracing::warn!(error = %e, "SDMA copy queue unavailable; AMD buffers stay host-visible");
        }
        None => {}
    }
    // PM4 graph capture is opt-in via `SVOD_PM4_GRAPH=1` (default OFF — it
    // regresses on gfx1151). Parse the env ONCE here into the per-device flag so
    // the capture path reads a plain bool: only "1" enables; "0"/empty/unset stay
    // OFF (presence alone no longer enables it), and there is no per-capture env
    // lookup to race with test toggles.
    if std::env::var("SVOD_PM4_GRAPH").as_deref() == Ok("1") {
        device_handle.core().set_pm4_graph(true);
    }
    // No default connector: every dispatcher leases/owns its own connector
    // (`Program::execute` leases per call; plans/graphs hold one for their
    // lifetime). The pool starts empty and warms on first lease.
    let runtime: RuntimeFactory = Arc::new(move |compiled: &CompiledSpec| -> Result<Box<dyn Program>> {
        svod_device::device::validate_abi_descriptors(&compiled.abi, compiled.buf_count, &compiled.var_names)?;
        // `CompiledSpec.bytes` is the clang-produced amdgcn ELF.
        if compiled.bytes.is_empty() {
            return Err(svod_device::Error::Runtime {
                message: "AMD RuntimeFactory: CompiledSpec has empty ELF bytes".into(),
            });
        }
        // We need an AmdAllocator inside the closure for AmdProgram::load
        // (it allocates the code-object VRAM buffer). Constructing a fresh
        // one is cheap — the shared DEVICE_CACHE returns the same
        // Arc<AmdDevice>, so no kernel ioctls re-execute.
        let alloc = AmdAllocator::new(device_id)?;
        let prg = AmdProgram::load(Arc::clone(&device_handle), &alloc, &compiled.bytes, &compiled.name, &compiled.abi)?;
        Ok(Box::new(prg) as Box<dyn Program>)
    });

    // Graph factory: pre-build a PM4 indirect buffer for a captured kernel
    // chain and replay it with one doorbell (`svod_device::amd::AmdGraph`).
    // Returns `Ok(None)` when the chain isn't graphable (AQL queue, non-AMD
    // program), so the caller falls back to per-call dispatch. A fresh
    // AmdAllocator shares the cached `Arc<AmdDevice>`, so capture allocates the
    // IB page through the same KFD VM with no extra device open.
    let graph: GraphFactory = Arc::new(move |kernels: &[GraphKernel]| -> Result<Option<Box<dyn Graph>>> {
        let alloc = AmdAllocator::new(device_id)?;
        AmdGraph::capture(&alloc, kernels)
    });

    Ok(Device::new(spec, allocator, renderer, compiler, runtime).with_graph(graph))
}

/// Construct AMD renderer/compiler components without opening KFD or creating
/// queues. Clean BEAM workers use this path with device usage disabled.
pub fn create_amd_codegen(device_id: usize, arch: AmdArch) -> Result<(Arc<dyn Renderer>, Arc<dyn Compiler>)> {
    let spec = DeviceSpec::Amd { device_id };
    let renderer = Arc::new(AmdRendererWrapper { device: spec, arch });
    let cache = ObjectCache::from_env().map_err(runtime_as_device)?.map(Arc::new);
    let toolchain = ClangToolchain::discover(cache.as_deref()).map_err(runtime_as_device)?;
    // The per-kernel `-nogpulib` decision comes from the IR, which is already
    // part of every object-cache key; the persisted identity records the
    // arch-stable, ocml-free flag set.
    let flags = crate::amd::compile::amd_object_flags("", arch);
    let identity = CompilerIdentity {
        schema: OBJECT_CACHE_SCHEMA,
        backend: "amd-clang".into(),
        target_architecture: format!("amdgcn-amd-amdhsa/{}", arch.mcpu()),
        toolchain: toolchain.identity().into(),
        flags,
        abi: format!("amdhsa-kernel-abi-v1;wave-size={}", arch.wave_size()),
        object_format: "elf64-amdgpu-code-object-relocatable-v1".into(),
    };
    let cache_key = identity.cache_key();
    let compiler = Arc::new(AmdCompiler { arch, cache, toolchain, identity, cache_key });
    Ok((renderer, compiler))
}

struct AmdRendererWrapper {
    device: DeviceSpec,
    arch: AmdArch,
}

impl Renderer for AmdRendererWrapper {
    fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> Result<ProgramSpec> {
        let renderer = LlvmTextRenderer::amd(self.arch);
        let rendered = svod_codegen::Renderer::render(&renderer, ast, name.or(Some("kernel")))
            .map_err(|e| svod_device::Error::Runtime { message: format!("AMD IR rendering failed: {e}") })?;
        Ok(super::program_spec(&rendered, &self.device, ast))
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }

    fn gpu_arch(&self) -> Option<svod_dtype::GpuArch> {
        Some(svod_dtype::GpuArch::Amd(self.arch))
    }

    fn supported_ops(&self) -> svod_ir::RendererOps {
        let mut ops = svod_ir::RendererOps::all();
        ops.binary.remove(&svod_ir::BinaryOp::Threefry);
        ops.binary.remove(&svod_ir::BinaryOp::Pow);
        ops.binary.remove(&svod_ir::BinaryOp::Max);
        for op in [
            svod_ir::UnaryOp::Exp,
            svod_ir::UnaryOp::Log,
            // Sin must decompose (tinygrad `llvmir.py` `llvm_intrinsics` is
            // exactly {sqrt, log2, exp2}): `@llvm.sin.f32` lowers to the
            // hardware `v_sin_f32` behind an f32 `1/(2π)` pre-scale, which is
            // only accurate for small arguments — sin(±1e6) comes back as
            // ±sin(π/8) because the reduction happens in f32 revolutions.
            // `v_exp_f32`/`v_log_f32`/`v_sqrt_f32` are ~1-ulp and stay native.
            svod_ir::UnaryOp::Sin,
            svod_ir::UnaryOp::Cos,
            svod_ir::UnaryOp::Tan,
            svod_ir::UnaryOp::Erf,
        ] {
            ops.unary.remove(&op);
        }
        ops
    }

    fn decompositor(&self) -> Option<svod_ir::pattern::TypedPatternMatcher<()>> {
        // Target Exp2/Log2/Sin/Sqrt selection is centralized in the scheduler.
        // This matcher only handles Morok's additional transcendental ops.
        Some(svod_ir::decompositions::amd_decomposition_patterns())
    }

    fn extra_matcher(&self) -> Option<svod_ir::pattern::TypedPatternMatcher<()>> {
        Some(svod_codegen::llvm::amd_extra_matcher())
    }
}

struct AmdCompiler {
    arch: AmdArch,
    cache: Option<Arc<ObjectCache>>,
    toolchain: ClangToolchain,
    identity: CompilerIdentity,
    cache_key: String,
}

impl Compiler for AmdCompiler {
    fn compile(&self, spec: &ProgramSpec) -> Result<CompiledSpec> {
        let key = ObjectCacheKey::new(spec.src.as_bytes(), self.identity.clone());
        let bytes = if let Some(cache) = &self.cache {
            cache.get_or_compile(
                &key,
                |bytes| crate::amd::compile::validate_amd_object(bytes, self.arch, &spec.name),
                || crate::amd::compile::compile_ir_to_amd_object_with(&self.toolchain, &spec.src, self.arch),
            )
        } else {
            crate::amd::compile::compile_ir_to_amd_object_with(&self.toolchain, &spec.src, self.arch).and_then(
                |bytes| crate::amd::compile::validate_amd_object(&bytes, self.arch, &spec.name).map(|()| bytes),
            )
        }
        .map_err(runtime_as_device)?;
        let mut compiled = CompiledSpec::from_bytes(spec.name.clone(), bytes, spec.ast.clone(), spec.abi.clone())?;
        compiled.global_size = spec.global_size.clone();
        compiled.local_size = spec.local_size.clone();
        Ok(compiled)
    }

    fn cache_key(&self) -> &str {
        &self.cache_key
    }
}

fn runtime_as_device(error: crate::Error) -> svod_device::Error {
    svod_device::Error::Runtime { message: error.to_string() }
}
