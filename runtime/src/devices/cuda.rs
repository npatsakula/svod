//! NVIDIA GPU device factory.
//!
//! Wires together:
//! - `svod_codegen::llvm::LlvmTextRenderer::nvptx(arch)` for IR emission.
//! - `svod_runtime::cuda::compile_ir_to_ptx` for the clang NVPTX compile,
//!   assembled to a cubin by `ptxas` when the toolkit is installed.
//! - `svod_device::cuda::CudaProgram` for module load + `cuLaunchKernel`.
//!
//! Construction fails cleanly (`NoCudaGpu` / `DeviceUnavailable`) on hosts
//! without a CUDA driver or device; never panics.

use std::sync::Arc;

use svod_codegen::llvm::LlvmTextRenderer;
use svod_device::Result;
use svod_device::cuda::{CudaDevice, CudaGraph, CudaProgram};
use svod_device::device::{
    CompiledSpec, Compiler, Device, GraphFactory, Program, ProgramSpec, Renderer, RuntimeFactory,
};
use svod_device::registry::DeviceRegistry;
use svod_dtype::{CudaArch, DeviceSpec, GpuArch};
use svod_ir::UOp;

use crate::clang::ClangToolchain;
use crate::cuda::compile::{Ptxas, compile_ir_to_ptx_with, ptx_flags, ptxas_flags, validate_ptx};
use crate::object_cache::{CompilerIdentity, OBJECT_CACHE_SCHEMA, ObjectCache, ObjectCacheKey};

/// Create a `CUDA:N` device end-to-end (allocator + renderer + compiler +
/// runtime + graph replay). `arch` is the device's compute capability as
/// reported by the driver (`svod_device::registry::resolve_cuda_arch`); it
/// selects `-march` and keys the object cache.
pub fn create_cuda_device(registry: &DeviceRegistry, device_id: usize, arch: CudaArch) -> Result<Device> {
    let spec = DeviceSpec::Cuda { device_id };
    let allocator = registry.get(&spec)?;
    let (renderer, compiler) = create_cuda_codegen(device_id, arch)?;
    let dev = CudaDevice::open(device_id)?;
    // Kernels are sized against the optimizer profile's static `.shared`
    // budget; one over the device's per-block limit only fails at JIT, so
    // refuse a device whose limit is below the profile up front.
    let shared_max = svod_schedule::OptimizerRenderer::for_cuda_arch(arch).shared_max;
    let shared_per_block = dev.limits().shared_per_block as usize;
    if shared_max > shared_per_block {
        return Err(svod_device::Error::DeviceUnavailable {
            reason: format!(
                "CUDA device {device_id} ({}) allows {shared_per_block} bytes of shared memory per block, \
                 below the {shared_max} bytes the {arch} optimizer profile assumes",
                dev.name()
            ),
        });
    }
    let runtime_dev = Arc::clone(&dev);
    let runtime: RuntimeFactory = Arc::new(move |compiled: &CompiledSpec| create_cuda_program(&runtime_dev, compiled));
    let graph: GraphFactory = Arc::new(move |kernels| CudaGraph::capture(Arc::clone(&dev), kernels));
    Ok(Device::new(spec, allocator, renderer, compiler, runtime).with_graph(graph))
}

/// Load a compiled kernel (a cubin or PTX text in `compiled.bytes`) into a
/// dispatchable program. The projection (`buf_count`, `var_names`) must agree
/// with the ABI before the loader derives the kernarg layout from it.
pub fn create_cuda_program(dev: &Arc<CudaDevice>, compiled: &CompiledSpec) -> Result<Box<dyn Program>> {
    svod_device::device::validate_abi_descriptors(&compiled.abi, compiled.buf_count, &compiled.var_names)?;
    if compiled.bytes.is_empty() {
        return Err(svod_device::Error::Runtime {
            message: "CUDA RuntimeFactory: CompiledSpec has an empty kernel image".into(),
        });
    }
    Ok(Box::new(CudaProgram::load(Arc::clone(dev), compiled)?))
}

/// Construct CUDA renderer/compiler components without opening the device.
/// Clean BEAM workers use this path with device usage disabled.
pub fn create_cuda_codegen(device_id: usize, arch: CudaArch) -> Result<(Arc<dyn Renderer>, Arc<dyn Compiler>)> {
    let spec = DeviceSpec::Cuda { device_id };
    let renderer = Arc::new(CudaRendererWrapper { device: spec, arch });
    let cache = ObjectCache::from_env().map_err(runtime_as_device)?.map(Arc::new);
    let toolchain = ClangToolchain::discover(cache.as_deref()).map_err(runtime_as_device)?;
    let ptxas = Ptxas::discover(cache.as_deref());
    let identity = cuda_compiler_identity(arch, &toolchain, ptxas.as_ref());
    let cache_key = identity.cache_key();
    let compiler = Arc::new(CudaCompiler { arch, cache, toolchain, ptxas, identity, cache_key });
    Ok((renderer, compiler))
}

/// The object-cache identity of the CUDA compile chain. Without `ptxas` the
/// object is PTX text that the driver JIT (and its own `~/.nv/ComputeCache`)
/// turns into SASS at load; with it the object is that PTX assembled into a
/// cubin. The two formats never share an entry.
pub(crate) fn cuda_compiler_identity(
    arch: CudaArch,
    clang: &ClangToolchain,
    ptxas: Option<&Ptxas>,
) -> CompilerIdentity {
    let mut flags = ptx_flags(arch);
    let (toolchain, object_format) = match ptxas {
        Some(ptxas) => {
            flags.extend(ptxas_flags(arch));
            (format!("{};{}", clang.identity(), ptxas.identity()), "cubin-v1")
        }
        None => (clang.identity().to_string(), "ptx-text-v1"),
    };
    CompilerIdentity {
        schema: OBJECT_CACHE_SCHEMA,
        backend: "nvptx-clang".into(),
        target_architecture: format!("nvptx64-nvidia-cuda/{arch}"),
        toolchain,
        flags,
        abi: format!("ptx-kernel-abi-v1;warp-size={}", arch.wave_size()),
        object_format: object_format.into(),
    }
}

struct CudaRendererWrapper {
    device: DeviceSpec,
    arch: CudaArch,
}

impl Renderer for CudaRendererWrapper {
    fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> Result<ProgramSpec> {
        let renderer = LlvmTextRenderer::nvptx(self.arch);
        let rendered = svod_codegen::Renderer::render(&renderer, ast, name.or(Some("kernel")))
            .map_err(|e| svod_device::Error::Runtime { message: format!("NVPTX IR rendering failed: {e}") })?;
        Ok(super::program_spec(&rendered, &self.device, ast))
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }

    fn gpu_arch(&self) -> Option<GpuArch> {
        Some(GpuArch::Cuda(self.arch))
    }

    fn supported_ops(&self) -> svod_ir::RendererOps {
        // AMD's table plus `Log2`: `sqrt.rn` and `ex2.approx` stay native.
        let mut ops = svod_ir::RendererOps::all();
        ops.binary.remove(&svod_ir::BinaryOp::Threefry);
        ops.binary.remove(&svod_ir::BinaryOp::Pow);
        ops.binary.remove(&svod_ir::BinaryOp::Max);
        for op in [
            svod_ir::UnaryOp::Exp,
            svod_ir::UnaryOp::Log,
            // `lg2.approx.f32` is a 2^-22.6 approximation where AMD's
            // `v_log_f32` is exact; the polynomial `xlog2` keeps the shared
            // 2e-6 test tolerances.
            svod_ir::UnaryOp::Log2,
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
        Some(svod_ir::decompositions::nvptx_decomposition_patterns())
    }

    fn extra_matcher(&self) -> Option<svod_ir::pattern::TypedPatternMatcher<()>> {
        Some(svod_codegen::llvm::nvptx_extra_matcher())
    }
}

struct CudaCompiler {
    arch: CudaArch,
    cache: Option<Arc<ObjectCache>>,
    toolchain: ClangToolchain,
    ptxas: Option<Ptxas>,
    identity: CompilerIdentity,
    cache_key: String,
}

impl Compiler for CudaCompiler {
    fn compile(&self, spec: &ProgramSpec) -> Result<CompiledSpec> {
        // The ABI is part of the key: a cubin's entry check runs at compile
        // time (below), so a cache hit must have been checked against the
        // ABI it is loaded with.
        let mut source = spec.src.clone().into_bytes();
        source.extend_from_slice(format!("\0abi={:?}", spec.abi).as_bytes());
        let key = ObjectCacheKey::new(&source, self.identity.clone());
        let validate = |bytes: &[u8]| match &self.ptxas {
            Some(_) => svod_device::cuda::validate_cubin(bytes, &spec.name)
                .map_err(|error| crate::Error::JitCompilation { reason: error.to_string() }),
            None => validate_ptx(bytes, self.arch, &spec.name),
        };
        let produce = || {
            let ptx = compile_ir_to_ptx_with(&self.toolchain, &spec.src, self.arch)?;
            let Some(ptxas) = &self.ptxas else { return Ok(ptx) };
            // The loader checks a PTX entry's parameters against the ABI but
            // cannot read a cubin's, so that check runs here, on the text.
            validate_ptx(&ptx, self.arch, &spec.name)?;
            let text = std::str::from_utf8(&ptx).expect("validated PTX is UTF-8");
            svod_device::cuda::check_ptx_entry_abi(text, &spec.name, &spec.abi)
                .map_err(|error| crate::Error::JitCompilation { reason: error.to_string() })?;
            ptxas.assemble(&ptx, self.arch)
        };
        let bytes = if let Some(cache) = &self.cache {
            cache.get_or_compile(&key, validate, produce)
        } else {
            produce().and_then(|bytes| validate(&bytes).map(|()| bytes))
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
