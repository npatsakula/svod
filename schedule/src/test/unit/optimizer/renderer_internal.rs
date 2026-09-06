use super::*;
use svod_ir::RendererDevice;

/// `Renderer::cpu()` is the runtime CPU target (threads, no shared memory);
/// `tinygrad_base_cpu` is the reference `Renderer()` the parity tests compare against.
#[test]
fn test_cpu_renderers_are_distinct_targets() {
    let runtime = Renderer::cpu();
    assert_eq!(runtime.device, RendererDevice::Cpu);
    assert!(!runtime.has_local && !runtime.has_shared && runtime.has_threads);
    assert!(runtime.tensor_cores.is_empty());

    let reference = Renderer::tinygrad_base_cpu();
    assert!(reference.has_local && reference.has_shared && !reference.has_threads);
    assert_eq!(reference.shared_max, 32768);
    assert_eq!(reference.global_max, Some(vec![0x8fff_ffff; 3]));
    assert_eq!(reference.local_max, Some(0x8fff_ffff));
}

#[test]
fn test_renderer_cuda() {
    let r = Renderer::cuda();
    assert_eq!(r.device, RendererDevice::CudaSm80); // Default is SM80/Ampere
    assert!(r.has_local && r.has_shared && !r.has_threads);
    assert!(r.shared_max > 0);
    assert!(!r.tensor_cores.is_empty());
}

#[test]
fn test_for_amd_arch_maps_each_family() {
    use svod_dtype::AmdArch;
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx942).device, RendererDevice::AmdCdna3);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx950).device, RendererDevice::AmdCdna4);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx1100).device, RendererDevice::AmdRdna3);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx1151).device, RendererDevice::AmdRdna3);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx1201).device, RendererDevice::AmdRdna4);
}

/// The fingerprint keys the compilation caches, so it must move with anything that
/// changes generated code — including two archs that share a `RendererDevice`.
#[test]
fn test_renderer_fingerprint_tracks_exact_target_and_capabilities() {
    use svod_dtype::AmdArch;

    let gfx1151 = Renderer::for_amd_arch(AmdArch::Gfx1151);
    assert_ne!(Renderer::for_amd_arch(AmdArch::Gfx1100).cache_fingerprint(), gfx1151.cache_fingerprint());

    let mut constrained = gfx1151.clone();
    constrained.upcast_max -= 1;
    assert_ne!(gfx1151.cache_fingerprint(), constrained.cache_fingerprint());

    constrained = gfx1151.clone();
    constrained.tensor_cores.clear();
    assert_ne!(gfx1151.cache_fingerprint(), constrained.cache_fingerprint());

    let all_ops = gfx1151.clone().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None);
    let mut fewer_ops = svod_ir::RendererOps::all();
    fewer_ops.binary.remove(&svod_ir::BinaryOp::Threefry);
    assert_ne!(
        all_ops.cache_fingerprint(),
        gfx1151.with_rewrite_capabilities(fewer_ops, None, None).cache_fingerprint()
    );
}

/// CDNA renders OCP FP8 natively but not the FNUZ encodings; RDNA renders neither.
#[test]
fn test_amd_fp8_dtype_capabilities_are_arch_specific() {
    use svod_dtype::{AmdArch, ScalarDType};

    for arch in [AmdArch::Gfx942, AmdArch::Gfx950] {
        let renderer = Renderer::for_amd_arch(arch);
        for dtype in [ScalarDType::FP8E4M3, ScalarDType::FP8E5M2] {
            assert!(renderer.supports_storage_dtype(dtype), "{arch} must keep OCP {dtype:?} storage");
            assert!(renderer.supports_conversion_dtype(dtype), "{arch} must keep OCP {dtype:?} conversion");
            assert!(renderer.supports_matrix_dtype(dtype), "{arch} must keep {dtype:?} matrix operands");
            assert!(!renderer.supports_alu_dtype(dtype), "{arch} must widen ordinary {dtype:?} ALU");
        }
        for dtype in [ScalarDType::FP8E4M3FNUZ, ScalarDType::FP8E5M2FNUZ] {
            assert!(!renderer.supports_dtype(dtype), "{arch} must decompose {dtype:?}");
        }
    }

    for arch in [AmdArch::Gfx1151, AmdArch::Gfx1201] {
        let renderer = Renderer::for_amd_arch(arch);
        for dtype in [ScalarDType::FP8E4M3, ScalarDType::FP8E5M2, ScalarDType::FP8E4M3FNUZ, ScalarDType::FP8E5M2FNUZ] {
            assert!(!renderer.supports_dtype(dtype), "{arch} must decompose {dtype:?} to f16");
        }
    }
}

#[test]
fn test_amd_tensor_core_tables_match_architecture() {
    use svod_dtype::AmdArch;

    // tinygrad `tc.py:132`: amd_cdna3 = amd_cdna_161632[:2] + amd_cdna_161616 -- four
    // cores, no fp32 input (the rate-neutral `v_mfma_f32_16x16x4_f32` is not offered).
    let gfx942 = Renderer::for_amd_arch(AmdArch::Gfx942);
    assert_eq!(gfx942.tensor_cores.len(), 4);
    assert!(gfx942.tensor_cores.iter().any(|tc| tc.dims == (16, 16, 32) && tc.dtype_in == DType::FP8E4M3));
    assert!(!gfx942.tensor_cores.iter().any(|tc| tc.dtype_in == DType::Float32));

    let gfx950 = Renderer::for_amd_arch(AmdArch::Gfx950);
    assert_eq!(gfx950.tensor_cores.len(), 8);
    assert!(gfx950.tensor_cores.iter().any(|tc| tc.dims == (16, 16, 128) && tc.dtype_in == DType::FP8E4M3));

    let gfx1151 = Renderer::for_amd_arch(AmdArch::Gfx1151);
    assert_eq!(gfx1151.tensor_cores.len(), 4);
    assert!(!gfx1151.tensor_cores.iter().any(|tc| tc.dtype_in.scalar_dtype().is_fp8()));
    assert!(gfx1151.tensor_cores.iter().any(|tc| tc.dtype_in == DType::Int8 && tc.dtype_out == DType::Int32));

    let gfx1201 = Renderer::for_amd_arch(AmdArch::Gfx1201);
    assert_eq!(gfx1201.tensor_cores.len(), 4);
    assert!(!gfx1201.tensor_cores.iter().any(|tc| tc.dtype_in.scalar_dtype().is_fp8()));
    assert!(gfx1201.tensor_cores.iter().any(|tc| tc.dtype_in == DType::Float16 && tc.dtype_out == DType::Float32));
    assert!(gfx1201.tensor_cores.iter().any(|tc| tc.dtype_in == DType::BFloat16 && tc.dtype_out == DType::BFloat16));

    let cuda = CUDA_81616.build(DType::Float16, DType::Float32);
    assert_eq!((cuda.dims, cuda.threads), ((8, 16, 16), 32));
    assert!(!cuda.opts.is_empty());
}

/// Only targets with a per-axis local limit report one.
#[test]
fn test_local_max_axes_match_renderer_capabilities() {
    assert_eq!(Renderer::cuda().local_max_axes(), Some([1024, 1024, 64]));
    assert_eq!(Renderer::webgpu().local_max_axes(), Some([256, 256, 64]));
    assert_eq!(Renderer::amd_cdna3().local_max_axes(), None);
    assert_eq!(Renderer::cpu().local_max_axes(), None);
    assert_eq!(Renderer::metal().local_max_axes(), None);
}

/// Tensor cores follow the Apple GPU family: none below Apple7 or on Intel-Mac
/// GPUs, and the family is part of the profile's identity.
#[test]
fn test_metal_profile_follows_gpu_family() {
    use svod_dtype::MetalFamily;
    let m4 = Renderer::for_metal_family(MetalFamily::Apple(9));
    assert_eq!(m4.tensor_cores.len(), Renderer::metal().tensor_cores.len());
    let m1 = Renderer::for_metal_family(MetalFamily::Apple(7));
    assert_eq!(m1.tensor_cores.len(), 5);
    for family in [MetalFamily::Apple(6), MetalFamily::Mac2, MetalFamily::Unknown] {
        assert!(Renderer::for_metal_family(family).tensor_cores.is_empty(), "{family}");
    }
    assert_ne!(m4.cache_fingerprint(), m1.cache_fingerprint());
    assert_ne!(m4.cache_fingerprint(), Renderer::metal().cache_fingerprint());
    assert_eq!(Renderer::for_metal_family(MetalFamily::Apple(9)).cache_fingerprint(), m4.cache_fingerprint());
}

/// Tinygrad `tc.get_cuda` minus fp8 plus int8: bf16/tf32, `m16n8k16` and the
/// s8 `m16n8k32` need sm_80, sm_75 keeps the f16 `m16n8k8` core only, and
/// older parts run without tensor cores. fp8 stays off even on sm_89+ (no
/// NVPTX cast lowering yet). The int8 core is the `CUDA_81632` fragment shape
/// with the `int8 -> int32` dtype pair the RDNA3 profile declares, so a
/// quantized linear selects it on both vendors.
#[test_case::test_case(7, 0, RendererDevice::CudaSm75, 0, false, false, false; "volta has no m16n8 mma")]
#[test_case::test_case(7, 5, RendererDevice::CudaSm75, 2, false, false, false; "turing")]
#[test_case::test_case(8, 0, RendererDevice::CudaSm80, 6, true, false, true; "ampere a100")]
#[test_case::test_case(8, 6, RendererDevice::CudaSm80, 6, true, false, true; "ampere ga10x")]
#[test_case::test_case(8, 9, RendererDevice::CudaSm80, 6, true, false, true; "ada withholds fp8")]
#[test_case::test_case(9, 0, RendererDevice::CudaSm80, 6, true, false, true; "hopper")]
#[test_case::test_case(12, 0, RendererDevice::CudaSm80, 6, true, false, true; "blackwell consumer")]
fn test_for_cuda_arch_follows_capability(
    major: u8,
    minor: u8,
    device: RendererDevice,
    tensor_cores: usize,
    bf16: bool,
    fp8: bool,
    int8: bool,
) {
    use svod_dtype::{CudaArch, ScalarDType};
    let arch = CudaArch::from_compute_capability(major, minor);
    let renderer = Renderer::for_cuda_arch(arch);
    assert_eq!(renderer.device, device);
    assert_eq!(renderer.tensor_cores.len(), tensor_cores);
    assert_eq!(renderer.target.as_deref(), Some(arch.to_string().as_str()));
    assert_eq!(renderer.supports_storage_dtype(ScalarDType::BFloat16), bf16);
    assert_eq!(renderer.supports_matrix_dtype(ScalarDType::BFloat16), bf16 && tensor_cores > 0);
    assert!(renderer.supports_storage_dtype(ScalarDType::Int8));
    assert_eq!(renderer.supports_matrix_dtype(ScalarDType::Int8), int8);
    let int8_cores: Vec<_> = renderer.tensor_cores.iter().filter(|tc| tc.dtype_in == DType::Int8).collect();
    assert_eq!(int8_cores.len(), usize::from(int8));
    if let Some(tc) = int8_cores.first() {
        let fp8_shape = CUDA_81632.build(DType::Int8, DType::Int32);
        assert_eq!(**tc, fp8_shape, "int8 must reuse the byte-wide m16n8k32 fragment layout");
        assert_eq!((tc.dims, tc.elements_per_thread, tc.dtype_out.clone()), ((8, 16, 32), (16, 8, 4), DType::Int32));
        let rdna3 = Renderer::amd_rdna3();
        assert!(
            rdna3.tensor_cores.iter().any(|amd| (&amd.dtype_in, &amd.dtype_out) == (&tc.dtype_in, &tc.dtype_out)),
            "CUDA int8 core must share the RDNA3 dtype pair"
        );
    }
    assert_eq!(renderer.supports_storage_dtype(ScalarDType::FP8E4M3), fp8);
    assert_eq!(renderer.supports_matrix_dtype(ScalarDType::FP8E4M3), fp8);
    assert!(!renderer.tensor_cores.iter().any(|tc| tc.dtype_in.scalar_dtype().is_fp8()));
    assert!(!renderer.supports_storage_dtype(ScalarDType::FP8E4M3FNUZ));
    assert!(renderer.tensor_cores.iter().all(|tc| tc.threads == 32 && (tc.dims.0, tc.dims.1) == (8, 16)));
    // No tf32 core unless explicitly allowed (tinygrad `ALLOW_TF32`).
    assert!(!renderer.tensor_cores.iter().any(|tc| tc.dtype_in == DType::Float32));
    assert_eq!(renderer.local_max_axes(), Some([1024, 1024, 64]));
}

/// Two capabilities sharing a profile still fingerprint apart (the target
/// string reaches the kernel cache), and the same capability is stable.
#[test]
fn test_for_cuda_arch_fingerprint_tracks_the_exact_capability() {
    use svod_dtype::CudaArch;
    let sm80 = Renderer::for_cuda_arch(CudaArch::from_compute_capability(8, 0));
    let sm86 = Renderer::for_cuda_arch(CudaArch::from_compute_capability(8, 6));
    assert_eq!(sm80.device, sm86.device);
    assert_ne!(sm80.cache_fingerprint(), sm86.cache_fingerprint());
    assert_eq!(
        Renderer::for_cuda_arch(CudaArch::from_compute_capability(8, 6)).cache_fingerprint(),
        sm86.cache_fingerprint()
    );
    assert_ne!(
        Renderer::cuda().cache_fingerprint(),
        sm80.cache_fingerprint(),
        "the arch-agnostic profile has no target"
    );
}

struct FakeCudaRenderer(svod_dtype::CudaArch);

impl svod_device::device::Renderer for FakeCudaRenderer {
    fn render(
        &self,
        ast: &std::sync::Arc<svod_ir::UOp>,
        name: Option<&str>,
    ) -> svod_device::Result<svod_device::device::ProgramSpec> {
        Ok(svod_device::device::ProgramSpec::new(
            name.unwrap_or("kernel").to_string(),
            String::new(),
            svod_dtype::DeviceSpec::Cuda { device_id: 0 },
            ast.clone(),
        ))
    }

    fn device(&self) -> &svod_dtype::DeviceSpec {
        static DEVICE: svod_dtype::DeviceSpec = svod_dtype::DeviceSpec::Cuda { device_id: 0 };
        &DEVICE
    }

    fn gpu_arch(&self) -> Option<svod_dtype::GpuArch> {
        Some(svod_dtype::GpuArch::Cuda(self.0))
    }

    fn supported_ops(&self) -> svod_ir::RendererOps {
        svod_ir::RendererOps::all()
    }

    fn decompositor(&self) -> Option<TypedPatternMatcher> {
        Some(svod_ir::decompositions::nvptx_decomposition_patterns())
    }

    fn extra_matcher(&self) -> Option<TypedPatternMatcher> {
        Some(svod_schedule_bool_storage())
    }
}

fn svod_schedule_bool_storage() -> TypedPatternMatcher {
    crate::devectorize::bool_storage_patterns().clone()
}

/// The matcher identities are part of the cache key; NVPTX must not share
/// AMD's or the generic backend's strings, or kernels compiled for one
/// target could be served to another.
#[test]
fn test_with_codegen_renderer_keys_nvptx_matchers_distinctly() {
    let arch = svod_dtype::CudaArch::from_compute_capability(8, 6);
    let cuda = Renderer::for_cuda_arch(arch).with_codegen_renderer(&FakeCudaRenderer(arch));
    assert_eq!(cuda.decomposition_profile, "nvptx-decomposition-v1");
    assert_eq!(cuda.extra_profile, "llvm-nvptx-extra-v1");
    assert_eq!(cuda.target.as_deref(), Some("sm_86"));

    let amd = Renderer::for_amd_arch(svod_dtype::AmdArch::Gfx1151).with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        None,
        None,
    );
    assert_ne!(cuda.decomposition_profile, amd.decomposition_profile);
    assert_ne!(cuda.cache_fingerprint(), Renderer::for_cuda_arch(arch).cache_fingerprint());
}
