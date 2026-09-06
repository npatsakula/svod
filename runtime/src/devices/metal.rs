//! Metal (Apple GPU) device factory.
//!
//! Wires together:
//! - `svod_codegen::c::CRenderer::metal()` for MSL emission.
//! - `svod_device::metal::compile` for MSL → metallib (private
//!   `MTLCodeGenService`, or the public `newLibraryWithSource:` fallback).
//! - `svod_device::metal::MetalProgram` for pipeline load + dispatch.
//!
//! Construction fails cleanly (`DeviceUnavailable`) on hosts without Metal.

use std::path::PathBuf;
use std::sync::Arc;

use svod_codegen::c::CRenderer;
use svod_device::Result;
use svod_device::device::{
    CompiledSpec, Compiler, Device, GraphFactory, Program, ProgramSpec, Renderer, RuntimeFactory,
};
use svod_device::metal::compile::{
    codegen_service_available, compile_msl, compile_msl_public, macos_product_version, metal_std_flag,
    validate_metallib,
};
use svod_device::metal::{MetalDevice, MetalGraph, MetalProgram};
use svod_device::registry::DeviceRegistry;
use svod_dtype::{DeviceSpec, GpuArch, MetalFamily};
use svod_ir::UOp;

use crate::object_cache::{CompilerIdentity, OBJECT_CACHE_SCHEMA, ObjectCache, ObjectCacheKey};

/// Create a `METAL:N` device end-to-end (allocator + renderer + compiler +
/// runtime + indirect-command-buffer graph replay).
pub fn create_metal_device(registry: &DeviceRegistry, device_id: usize) -> Result<Device> {
    let spec = DeviceSpec::Metal { device_id };
    let allocator = registry.get(&spec)?;
    let (renderer, compiler) = create_metal_codegen(device_id)?;
    let dev = MetalDevice::open(device_id)?;
    let runtime_dev = Arc::clone(&dev);
    let runtime: RuntimeFactory = Arc::new(move |compiled: &CompiledSpec| -> Result<Box<dyn Program>> {
        create_metal_program(&runtime_dev, compiled)
    });
    let graph: GraphFactory = Arc::new(move |kernels| MetalGraph::capture(Arc::clone(&dev), kernels));
    Ok(Device::new(spec, allocator, renderer, compiler, runtime).with_graph(graph))
}

/// Load a compiled kernel (metallib bytes, or MSL source in fallback mode).
pub fn create_metal_program(dev: &Arc<MetalDevice>, compiled: &CompiledSpec) -> Result<Box<dyn Program>> {
    svod_device::device::validate_abi_descriptors(&compiled.abi, compiled.buf_count, &compiled.var_names)?;
    if compiled.bytes.is_empty() {
        return Err(svod_device::Error::Runtime {
            message: "Metal RuntimeFactory: CompiledSpec has empty metallib bytes".into(),
        });
    }
    let program = MetalProgram::load(Arc::clone(dev), &compiled.bytes, &compiled.name, &compiled.abi)?;
    Ok(Box::new(program) as Box<dyn Program>)
}

/// Construct Metal renderer/compiler components. Opening the (cached) device is
/// cheap and non-exclusive, so BEAM worker processes use this path too.
pub fn create_metal_codegen(device_id: usize) -> Result<(Arc<dyn Renderer>, Arc<dyn Compiler>)> {
    let spec = DeviceSpec::Metal { device_id };
    let dev = MetalDevice::open(device_id)?;
    let cache = ObjectCache::from_env().map_err(runtime_as_device)?.map(Arc::new);
    // `-fno-fast-math` keeps IEEE semantics (the numerics the shared test
    // tolerances assume). These flags determine the compiled output and so key
    // the object cache; the private-service vs `newLibraryWithSource:` transport
    // does NOT — both yield a payload the loader accepts (it dispatches on the
    // `MTLB` magic), so the identity must stay stable across them or a BEAM
    // worker (which wins the in-process-LLVM slot and uses the private service)
    // would disagree with a parent that lost it and used the public path.
    let flags: Vec<String> =
        format!("-fno-fast-math -std={} --driver-mode=metal -x metal -fno-caret-diagnostics", metal_std_flag())
            .split_whitespace()
            .map(str::to_string)
            .collect();
    // The module cache (metal_stdlib parse: ~250 ms → ~8 ms) is a private-path
    // compile speedup only; it does not change the output, so it stays out of
    // the identity flags (its path can vary per process) and is appended for the
    // actual compile.
    let params =
        format!("{} -fmodules-cache-path=\"{}\"", flags.join(" "), metal_modules_cache_dir(cache.as_deref()).display());
    let identity = CompilerIdentity {
        schema: OBJECT_CACHE_SCHEMA,
        backend: "metal".into(),
        target_architecture: format!("{}/air64", dev.family()),
        toolchain: format!("macos={}", macos_product_version().unwrap_or_else(|| "unknown".into())),
        flags,
        abi: "msl-kernel-abi-v1".into(),
        // A metallib (private service) and an MSL source payload (public
        // fallback) are interchangeable at load, so they share one format tag
        // and one cache slot; `validate_metallib` accepts either on read.
        object_format: "metallib-or-msl-v1".into(),
    };
    let cache_key = identity.cache_key();
    let renderer = Arc::new(MetalRendererWrapper { device: spec, family: dev.family() });
    let compiler = Arc::new(MetalCompiler { dev, cache, params, identity, cache_key });
    Ok((renderer, compiler))
}

/// LLVM's metal_stdlib module cache: a sibling of the object cache when one
/// exists (eviction there only touches `*.obj`/`*.lock`), else the XDG/HOME
/// location, else the OS temp dir.
fn metal_modules_cache_dir(cache: Option<&ObjectCache>) -> PathBuf {
    let dir = cache
        .map(|cache| cache.root().join("metal-modules"))
        .or_else(|| std::env::var_os("XDG_CACHE_HOME").map(|path| PathBuf::from(path).join("svod/metal-modules")))
        .or_else(|| std::env::var_os("HOME").map(|path| PathBuf::from(path).join(".cache/svod/metal-modules")))
        .unwrap_or_else(|| std::env::temp_dir().join("svod-metal-modules"));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

struct MetalRendererWrapper {
    device: DeviceSpec,
    family: MetalFamily,
}

impl Renderer for MetalRendererWrapper {
    fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> Result<ProgramSpec> {
        let rendered = svod_codegen::Renderer::render(&CRenderer::metal(), ast, name.or(Some("kernel")))
            .map_err(|e| svod_device::Error::Runtime { message: format!("MSL rendering failed: {e}") })?;
        Ok(super::program_spec(&rendered, &self.device, ast))
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }

    fn gpu_arch(&self) -> Option<GpuArch> {
        Some(GpuArch::Metal(self.family))
    }

    fn supported_ops(&self) -> svod_ir::RendererOps {
        let mut ops = svod_ir::RendererOps::all();
        // Same removals as clang/AMD: Threefry renders as bare XOR and Max
        // decomposes to a select; Pow and the non-native transcendentals go
        // through the shared decompositions. `sqrt/exp2/log2` are native and
        // `sin` renders as `precise::sin` (tinygrad's Metal choice).
        ops.binary.remove(&svod_ir::BinaryOp::Threefry);
        ops.binary.remove(&svod_ir::BinaryOp::Pow);
        ops.binary.remove(&svod_ir::BinaryOp::Max);
        for op in [
            svod_ir::UnaryOp::Exp,
            svod_ir::UnaryOp::Log,
            svod_ir::UnaryOp::Cos,
            svod_ir::UnaryOp::Tan,
            // MSL has no erf().
            svod_ir::UnaryOp::Erf,
        ] {
            ops.unary.remove(&op);
        }
        ops
    }

    fn decompositor(&self) -> Option<svod_ir::pattern::TypedPatternMatcher<()>> {
        // Transcendental + bf16 lowering over native exp2/log2; not amdgcn-specific.
        Some(svod_ir::decompositions::amd_decomposition_patterns())
    }

    fn extra_matcher(&self) -> Option<svod_ir::pattern::TypedPatternMatcher<()>> {
        Some(svod_codegen::llvm::cpu_extra_matcher())
    }
}

struct MetalCompiler {
    dev: Arc<MetalDevice>,
    cache: Option<Arc<ObjectCache>>,
    params: String,
    identity: CompilerIdentity,
    cache_key: String,
}

impl MetalCompiler {
    fn produce(&self, source: &str) -> crate::Result<Vec<u8>> {
        if codegen_service_available() {
            compile_msl(source, &self.params)
        } else {
            compile_msl_public(&self.dev, source)
        }
        .map_err(device_as_runtime)
    }
}

impl Compiler for MetalCompiler {
    fn compile(&self, spec: &ProgramSpec) -> Result<CompiledSpec> {
        let key = ObjectCacheKey::new(spec.src.as_bytes(), self.identity.clone());
        let validate = |bytes: &[u8]| validate_metallib(bytes, &spec.name).map_err(device_as_runtime);
        let bytes = if let Some(cache) = &self.cache {
            cache.get_or_compile(&key, validate, || self.produce(&spec.src))
        } else {
            self.produce(&spec.src).and_then(|bytes| validate(&bytes).map(|()| bytes))
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

fn device_as_runtime(error: svod_device::Error) -> crate::Error {
    crate::Error::JitCompilation { reason: error.to_string() }
}
