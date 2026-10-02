//! White-box tests over `crate::optimizer::renderer`: the per-target capability
//! profiles, the cache fingerprint that keys every one of them, and the
//! codegen-renderer binding.

use smallvec::smallvec;
use svod_ir::{Op, UOp};
use test_case::test_case;

use super::*;

/// A GFX1151 renderer with one capability field edited, for fingerprint tests.
fn mutate(edit: fn(&mut Renderer)) -> Renderer {
    let mut renderer = Renderer::for_amd_arch(AmdArch::Gfx1151);
    edit(&mut renderer);
    renderer
}

/// Every target's device, memory flags, wave width, shared-memory budget and
/// per-axis local limit. The flags are checked in exact triples, so they subsume
/// the `has_local && has_shared` / `!has_*` spot checks of a per-backend profile,
/// and `shared_max` is pinned to its exact byte budget: it bounds every GROUP and
/// LOCAL decision, so a drifted value silently retunes the optimizer.
#[test_case(Renderer::cpu(), RendererDevice::Cpu, (false, false, true), 32, 0, None; "cpu")]
#[test_case(Renderer::tinygrad_base_cpu(), RendererDevice::Cpu, (true, true, false), 32, 32768, None; "tinygrad base cpu")]
#[test_case(Renderer::cuda_sm75(), RendererDevice::CudaSm75, (true, true, false), 32, 49152, Some([1024, 1024, 64]); "cuda sm75")]
#[test_case(Renderer::cuda(), RendererDevice::CudaSm80, (true, true, false), 32, 49152, Some([1024, 1024, 64]); "cuda")]
#[test_case(Renderer::cuda_sm89(false), RendererDevice::CudaSm89, (true, true, false), 32, 49152, Some([1024, 1024, 64]); "cuda sm89")]
#[test_case(Renderer::metal(), RendererDevice::Metal, (true, true, false), 32, 32768, None; "metal")]
#[test_case(Renderer::amd_rdna3(), RendererDevice::AmdRdna3, (true, true, false), 32, 65536, None; "rdna3")]
#[test_case(Renderer::amd_cdna3(), RendererDevice::AmdCdna3, (true, true, false), 64, 65536, None; "cdna3")]
#[test_case(Renderer::amd_cdna4(), RendererDevice::AmdCdna4, (true, true, false), 64, 65536, None; "cdna4")]
#[test_case(Renderer::intel_xe(), RendererDevice::IntelXe, (true, true, false), 32, 65536, None; "intel xe")]
#[test_case(Renderer::webgpu(), RendererDevice::WebGpu, (true, true, false), 32, 16384, Some([256, 256, 64]); "webgpu")]
fn renderer_profiles_match_their_capabilities(
    renderer: Renderer,
    device: RendererDevice,
    flags: (bool, bool, bool),
    wave: usize,
    shared_max: usize,
    local_max_axes: Option<[usize; 3]>,
) {
    assert_eq!(renderer.device, device);
    assert_eq!((renderer.has_local, renderer.has_shared, renderer.has_threads), flags);
    assert_eq!(renderer.wave_size(), wave);
    assert_eq!(renderer.local_max_axes(), local_max_axes);
    assert_eq!(renderer.shared_max, shared_max);
    assert_eq!(renderer.shared_max == 0, !renderer.has_shared, "the shared budget matches the flag");
}

/// The runtime CPU target is distinct from the tinygrad reference: no tensor
/// cores, no local cap, and a different fingerprint. The reference keeps
/// tinygrad's unbounded base-`Renderer` limits verbatim, because the parity
/// tests compare optimizer decisions taken under them.
#[test]
fn cpu_renderers_are_distinct_targets() {
    let (runtime, reference) = (Renderer::cpu(), Renderer::tinygrad_base_cpu());
    assert!(runtime.tensor_cores.is_empty());
    assert_eq!(runtime.local_max, None);
    assert_eq!(reference.shared_max, 32768);
    assert_eq!(reference.global_max, Some(vec![0x8fff_ffff; 3]));
    assert_eq!(reference.local_max, Some(0x8fff_ffff));
    assert_ne!(runtime.cache_fingerprint(), reference.cache_fingerprint());
}

/// Every gfx family maps to its optimizer profile, with the exact `mcpu` target.
#[test_case(AmdArch::Gfx942, RendererDevice::AmdCdna3; "cdna3")]
#[test_case(AmdArch::Gfx950, RendererDevice::AmdCdna4; "cdna4")]
#[test_case(AmdArch::Gfx1100, RendererDevice::AmdRdna3; "rdna3")]
#[test_case(AmdArch::Gfx1151, RendererDevice::AmdRdna3; "rdna3 apu")]
#[test_case(AmdArch::Gfx1201, RendererDevice::AmdRdna4; "rdna4")]
fn for_amd_arch_maps_each_family(arch: AmdArch, device: RendererDevice) {
    let renderer = Renderer::for_amd_arch(arch);
    assert_eq!(renderer.device, device);
    assert_eq!(renderer.target.as_deref(), Some(arch.mcpu()));
}

/// The fingerprint keys the compilation caches, so it must move with anything
/// that can change codegen: the target and every capability field.
#[test]
fn fingerprint_tracks_the_exact_target_and_every_capability() {
    let gfx1151 = Renderer::for_amd_arch(AmdArch::Gfx1151);
    assert_ne!(Renderer::for_amd_arch(AmdArch::Gfx1100).cache_fingerprint(), gfx1151.cache_fingerprint());

    let fields: Vec<(&str, Renderer)> = vec![
        ("target", mutate(|r| r.target = None)),
        ("has_local", mutate(|r| r.has_local = false)),
        ("has_shared", mutate(|r| r.has_shared = false)),
        ("has_threads", mutate(|r| r.has_threads = true)),
        ("shared_max", mutate(|r| r.shared_max += 1)),
        ("global_max", mutate(|r| r.global_max = None)),
        ("global_prod_max", mutate(|r| r.global_prod_max = None)),
        ("local_max", mutate(|r| r.local_max = None)),
        ("upcast_max", mutate(|r| r.upcast_max -= 1)),
        ("buffer_max", mutate(|r| r.buffer_max = Some(1))),
        ("supports_float4", mutate(|r| r.supports_float4 = !r.supports_float4)),
        ("tensor_cores", mutate(|r| r.tensor_cores.clear())),
        (
            "supported_dtypes",
            mutate(|r| {
                r.supported_dtypes.remove(&ScalarDType::Int32);
            }),
        ),
        ("decomposition profile", gfx1151.clone().with_rewrite_capabilities(RendererOps::all(), None, None)),
    ];
    for (name, renderer) in fields {
        assert_ne!(gfx1151.cache_fingerprint(), renderer.cache_fingerprint(), "{name} must change the key");
    }

    // The renderer's operation table is part of the identity too.
    let with_ops = gfx1151.clone().with_rewrite_capabilities(RendererOps::all(), None, None);
    let mut fewer = RendererOps::all();
    fewer.binary.remove(&svod_ir::BinaryOp::Threefry);
    assert_ne!(with_ops.cache_fingerprint(), gfx1151.with_rewrite_capabilities(fewer, None, None).cache_fingerprint());
}

/// CDNA and RDNA4 store and convert OCP FP8 natively but not the FNUZ
/// encodings; RDNA3 renders none of them. Only CDNA has an FP8 matrix core,
/// and every AMD part widens FP8 before its ALU.
#[test_case(AmdArch::Gfx942, ScalarDType::FP8E4M3, true, true; "cdna3 keeps OCP fp8")]
#[test_case(AmdArch::Gfx942, ScalarDType::FP8E4M3FNUZ, false, false; "cdna3 decomposes FNUZ fp8")]
#[test_case(AmdArch::Gfx1151, ScalarDType::FP8E4M3, false, false; "rdna3 decomposes OCP fp8")]
#[test_case(AmdArch::Gfx1201, ScalarDType::FP8E4M3, true, false; "rdna4 converts OCP fp8 without a matrix core")]
#[test_case(AmdArch::Gfx1201, ScalarDType::FP8E5M2, true, false; "rdna4 converts bf8 without a matrix core")]
#[test_case(AmdArch::Gfx1201, ScalarDType::FP8E5M2FNUZ, false, false; "rdna4 decomposes FNUZ fp8")]
fn amd_fp8_dtype_capabilities_are_arch_specific(arch: AmdArch, dtype: ScalarDType, supported: bool, matrix: bool) {
    let renderer = Renderer::for_amd_arch(arch);
    assert_eq!(renderer.supports_storage_dtype(dtype), supported, "{arch} storage");
    assert_eq!(renderer.supports_conversion_dtype(dtype), supported, "{arch} conversion");
    assert_eq!(renderer.supports_dtype(dtype), supported, "{arch} support");
    assert_eq!(renderer.supports_matrix_dtype(dtype), matrix, "{arch} matrix operand");
    assert_eq!(renderer.supports_alu_dtype(dtype), supported && !dtype.is_fp8(), "{arch} ALU widens fp8");
}

/// The AMD tensor-core tables: tinygrad `tc.py:132` for CDNA3 (four cores, no
/// int8) and the RDNA3/RDNA4 shapes. The accumulator is pinned alongside the
/// operand, or a core that silently widens (RDNA4's `bf16 -> bf16`, the only
/// non-widening pair on the vendor) would match on its input alone.
#[test_case(AmdArch::Gfx942, 4, (16, 16, 32), DType::FP8E4M3, DType::Float32; "cdna3")]
#[test_case(AmdArch::Gfx950, 8, (16, 16, 128), DType::FP8E4M3, DType::Float32; "cdna4")]
#[test_case(AmdArch::Gfx1151, 4, (16, 16, 16), DType::Int8, DType::Int32; "rdna3")]
#[test_case(AmdArch::Gfx1201, 5, (16, 16, 16), DType::BFloat16, DType::BFloat16; "rdna4")]
fn amd_tensor_core_tables_match_architecture(
    arch: AmdArch,
    len: usize,
    dims: (usize, usize, usize),
    dtype_in: DType,
    dtype_out: DType,
) {
    let renderer = Renderer::for_amd_arch(arch);
    assert_eq!(renderer.tensor_cores.len(), len, "{arch}");
    assert!(
        renderer.tensor_cores.iter().any(|tc| tc.dims == dims && tc.dtype_in == dtype_in && tc.dtype_out == dtype_out),
        "{arch}"
    );
    assert!(!renderer.tensor_cores.iter().any(|tc| tc.dtype_in == DType::Float32), "no fp32 input core");

    // RDNA3/RDNA4 declare no fp8 core; both declare the `iu8` one, which CDNA lacks.
    let fp8 = renderer.tensor_cores.iter().any(|tc| tc.dtype_in.scalar_dtype().is_fp8());
    assert_eq!(fp8, matches!(arch, AmdArch::Gfx942 | AmdArch::Gfx950), "{arch} fp8 core");
    assert_eq!(
        renderer.tensor_cores.iter().any(|tc| tc.dtype_in == DType::Int8 && tc.dtype_out == DType::Int32),
        !arch.is_cdna(),
        "{arch} int8 core"
    );
}

/// Tensor cores follow the Apple GPU family: none below Apple7 or on Intel-Mac
/// GPU families, five from Apple7 on, and the M4 family matches plain metal.
#[test_case(MetalFamily::Apple(9), 5; "an M1 family GPU has five cores")]
#[test_case(MetalFamily::Apple(7), 5; "Apple7 is the first family")]
#[test_case(MetalFamily::Apple(6), 0; "Apple6 has none")]
#[test_case(MetalFamily::Mac2, 0; "Intel Macs have none")]
#[test_case(MetalFamily::Unknown, 0; "an unknown family has none")]
fn metal_profile_follows_gpu_family(family: MetalFamily, cores: usize) {
    let renderer = Renderer::for_metal_family(family);
    assert_eq!(renderer.tensor_cores.len(), cores, "{family}");
    assert_eq!(renderer.target.as_deref(), Some(family.to_string().as_str()));
    assert_eq!(Renderer::for_metal_family(family).cache_fingerprint(), renderer.cache_fingerprint(), "stable");

    if family == MetalFamily::Apple(9) {
        assert_eq!(renderer.tensor_cores.len(), Renderer::metal().tensor_cores.len(), "the M4 family");
        assert_ne!(renderer.cache_fingerprint(), Renderer::for_metal_family(MetalFamily::Apple(7)).cache_fingerprint());
        assert_ne!(renderer.cache_fingerprint(), Renderer::metal().cache_fingerprint(), "the target separates them");
    }
}

/// Tinygrad `tc.get_cuda` minus fp8 plus int8: bf16/tf32, `m16n8k16` and the s8
/// `m16n8k32` fragment, plus every capability the compute version gates. The
/// byte-wide core is pinned to the RDNA3 dtype pair and to the `CUDA_81632`
/// fragment layout, `m16n8k16` to its `CUDA_81616` sibling.
#[test_case(7, 0, RendererDevice::CudaSm75, 0, None, false, false, false; "volta has no m16n8 mma")]
#[test_case(7, 5, RendererDevice::CudaSm75, 2, Some((4, 2, 4)), false, false, false; "turing keeps the m16n8k8 fragment")]
#[test_case(8, 0, RendererDevice::CudaSm80, 6, Some((8, 4, 4)), true, false, true; "ampere a100")]
#[test_case(8, 6, RendererDevice::CudaSm80, 6, Some((8, 4, 4)), true, false, true; "ampere ga10x")]
#[test_case(8, 9, RendererDevice::CudaSm80, 6, Some((8, 4, 4)), true, false, true; "ada withholds fp8")]
#[test_case(9, 0, RendererDevice::CudaSm80, 6, Some((8, 4, 4)), true, false, true; "hopper")]
#[test_case(12, 0, RendererDevice::CudaSm80, 6, Some((8, 4, 4)), true, false, true; "blackwell consumer")]
#[allow(clippy::too_many_arguments)]
fn for_cuda_arch_follows_capability(
    major: u8,
    minor: u8,
    device: RendererDevice,
    tensor_cores: usize,
    fp16_ept: Option<(usize, usize, usize)>,
    bf16: bool,
    fp8: bool,
    int8: bool,
) {
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
        // Spelled out rather than compared against `CUDA_81632.build(..)`, which
        // would only restate the constant the profile is built from.
        assert_eq!(
            (tc.dims, tc.elements_per_thread, tc.dtype_out.clone()),
            ((8, 16, 32), (16, 8, 4), DType::Int32),
            "the byte-wide m16n8k32 fragment layout"
        );
        assert_eq!(**tc, CUDA_81632.build(DType::Int8, DType::Int32), "built from the shared fragment constant");
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
    assert!(!renderer.tensor_cores.iter().any(|tc| tc.dtype_in == DType::Float32), "no tf32 unless allowed");

    // `CUDA_81616` and its `m16n8k8` Turing sibling, pinned by their operand
    // layout; the byte-wide `CUDA_81632` fragment is compared whole above.
    let fp16 = renderer.tensor_cores.iter().find(|tc| tc.dtype_in == DType::Float16);
    assert_eq!(fp16.map(|tc| tc.elements_per_thread), fp16_ept, "the fp16 operand fragment");
    if let Some(core) = fp16 {
        assert!(!core.opts.is_empty(), "every fragment declares its opt sequence");
    }
    assert!(renderer.tensor_cores.iter().all(|tc| tc.threads == 32 && (tc.dims.0, tc.dims.1) == (8, 16)));
}

/// Two capabilities sharing a profile still fingerprint apart (the target), and
/// a repeated construction is stable.
#[test]
fn for_cuda_arch_fingerprint_tracks_the_exact_capability() {
    let sm80 = Renderer::for_cuda_arch(CudaArch::from_compute_capability(8, 0));
    let sm86 = Renderer::for_cuda_arch(CudaArch::from_compute_capability(8, 6));
    assert_eq!(sm80.device, sm86.device);
    assert_ne!(sm80.cache_fingerprint(), sm86.cache_fingerprint());
    assert_eq!(
        Renderer::for_cuda_arch(CudaArch::from_compute_capability(8, 6)).cache_fingerprint(),
        sm86.cache_fingerprint()
    );
    assert_ne!(Renderer::cuda().cache_fingerprint(), sm80.cache_fingerprint(), "no target");
}

/// A TF32 opt-in is a distinct capability (it adds a core, and it changes the
/// fingerprint), so it must fingerprint apart.
#[test_case(false; "tf32 withheld")]
#[test_case(true; "tf32 allowed")]
fn tf32_opt_in_changes_the_capability(allow_tf32: bool) {
    assert_ne!(Renderer::cuda_sm80(false).cache_fingerprint(), Renderer::cuda_sm80(true).cache_fingerprint());
    assert_eq!(Renderer::cuda_sm80(allow_tf32).tensor_cores.iter().any(|tc| tc.dtype_in == DType::Float32), allow_tf32);
}

/// The `SVOD_TF32` opt-in reaches the profiles devices actually build. The env
/// is process-global, so each case runs in a child process with a clean
/// environment (the `config` probe pattern); the in-process env never mutates.
const TF32_PROBE: &str = "SVOD_TF32_PROBE";
const TF32_PROBE_NAME: &str = "optimizer::renderer::tests::tf32_env_opt_in_reaches_the_arch_profiles";

#[test]
fn tf32_env_opt_in_reaches_the_arch_profiles() {
    let Ok(case) = std::env::var(TF32_PROBE) else {
        let exe = std::env::current_exe().expect("current test binary");
        for (case, tf32) in [("off", None), ("on", Some("1")), ("zero", Some("0")), ("bare_true", Some("true"))] {
            let mut command = std::process::Command::new(&exe);
            command.args(["--exact", "--nocapture", TF32_PROBE_NAME]);
            command.env_clear().env(TF32_PROBE, case);
            if let Some(tf32) = tf32 {
                command.env("SVOD_TF32", tf32);
            }
            let output = command.output().expect("probe child");
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(output.status.success() && stdout.contains("probe ok"), "probe {case}:\n{stdout}");
        }
        return;
    };
    let expected = matches!(case.as_str(), "on" | "bare_true");
    for renderer in [Renderer::cuda(), Renderer::for_cuda_arch(CudaArch::from_compute_capability(8, 6))] {
        assert_eq!(
            renderer.tensor_cores.iter().any(|tc| tc.dtype_in == DType::Float32),
            expected,
            "{case}: tf32 shapes in the profile"
        );
    }
    // `cuda()` shares every field with `cuda_sm80(flag)` but the core set, so
    // its fingerprint pins exactly which baseline the env selected.
    assert_eq!(Renderer::cuda().cache_fingerprint(), Renderer::cuda_sm80(expected).cache_fingerprint(), "{case}");
    println!("probe ok");
}

/// The profile names the codegen binding per backend family, and the matcher
/// identities are part of the cache key: NVPTX must not share the profile or the
/// fingerprint with an explicitly bound AMD renderer. Every row also pins the
/// `target` the binding takes from the code renderer, since it is the rest of
/// the cache key.
#[test_case(Renderer::cpu(), "backend-decomposition-v1", "llvm-cpu-extra-v1", "cpu")]
#[test_case(Renderer::amd_rdna3(), "amd-decomposition-v1", "llvm-amd-fp8-extra-v1", "amd")]
#[test_case(Renderer::amd_cdna4(), "amd-decomposition-v1", "llvm-amd-fp8-extra-v1", "cdna")]
#[test_case(Renderer::metal(), "backend-decomposition-v1", "backend-extra-v1", "generic backend")]
#[test_case(Renderer::cuda_sm75(), "nvptx-decomposition-v1", "llvm-nvptx-extra-v1", "cuda sm75")]
#[test_case(Renderer::cuda(), "nvptx-decomposition-v1", "llvm-nvptx-extra-v1", "cuda sm80")]
#[test_case(Renderer::cuda_sm89(false), "nvptx-decomposition-v1", "llvm-nvptx-extra-v1", "cuda sm89")]
fn with_codegen_renderer_names_the_profiles(renderer: Renderer, decomposition: &str, extra: &str, name: &str) {
    let arch = CudaArch::from_compute_capability(8, 6);
    let bound = renderer.with_codegen_renderer(&FakeCudaRenderer(arch));
    assert_eq!(bound.decomposition_profile, decomposition, "{name}");
    assert_eq!(bound.extra_profile, extra, "{name}");
    assert_eq!(bound.target.as_deref(), Some("sm_86"), "{name}");
    assert!(bound.renderer_ops.is_some() && bound.extra_matcher.is_some(), "{name}");

    let nvptx = Renderer::for_cuda_arch(arch);
    assert_ne!(bound.cache_fingerprint(), nvptx.cache_fingerprint(), "{name} matchers key the cache");
}

/// A renderer `extra_matcher` that already wraps a LOCAL in AFTER runs before
/// barrier inference, so the LOCAL dependency must win that ordering.
#[test]
fn extra_matcher_local_dependency_precedes_barrier_inference() {
    let extra = crate::patterns! {
        Noop => {
            let local = UOp::buffer(0, 8, DType::Float32, svod_dtype::AddrSpace::Local, None);
            let store = UOp::index().buffer(local.clone()).indices(vec![UOp::index_const(0)]).call()
                .expect("index")
                .store_value(UOp::native_const(1.0f32));
            Some(local.after(smallvec![store]))
        },
    };
    let renderer = Renderer::cpu().with_rewrite_capabilities(RendererOps::all(), None, Some(extra));
    let rewritten =
        crate::rewrite::graph_rewrite(renderer.extra_matcher().unwrap(), UOp::new(Op::Noop, DType::Void), &mut ());
    let result = crate::optimizer::finish_final_rewrite(rewritten);

    assert!(
        matches!(result.op(), Op::After(svod_ir::ops::After { deps, .. })
            if matches!(deps.as_slice(), [barrier] if matches!(barrier.op(), Op::Barrier(..)))),
        "{}",
        result.tree()
    );
}

/// The renderer's supported-op table drives early decomposition: an op the table
/// withholds is decomposed away, and the two families are independent gates.
#[test_case(true, true; "threefry and erf kept")]
#[test_case(false, true; "threefry decomposed")]
#[test_case(true, false; "erf decomposed")]
fn supported_ops_control_decomposition(threefry_supported: bool, erf_supported: bool) {
    let x = UOp::const_(DType::UInt64, svod_ir::ConstValue::UInt(1));
    let key = UOp::const_(DType::UInt64, svod_ir::ConstValue::UInt(2));
    let threefry = UOp::new(Op::Binary(svod_ir::BinaryOp::Threefry, x, key), DType::UInt64);
    let erf = UOp::native_const(0.5f32).erf().unwrap();

    let mut ops = RendererOps::all();
    if !threefry_supported {
        ops.binary.remove(&svod_ir::BinaryOp::Threefry);
    }
    if !erf_supported {
        ops.unary.remove(&svod_ir::UnaryOp::Erf);
    }
    let patterns = crate::optimizer::early_decomposition_patterns(&ops);
    assert_eq!(
        crate::rewrite::graph_rewrite(&patterns, threefry, &mut ())
            .toposort()
            .iter()
            .any(|node| matches!(node.op(), Op::Binary(svod_ir::BinaryOp::Threefry, ..))),
        threefry_supported,
        "the binary gate"
    );
    assert_eq!(
        crate::rewrite::graph_rewrite(&patterns, erf, &mut ())
            .toposort()
            .iter()
            .any(|node| matches!(node.op(), Op::Unary(svod_ir::UnaryOp::Erf, _))),
        erf_supported,
        "the unary gate"
    );
}

/// A stand-in code renderer: enough of the `svod_device` renderer trait for
pub(crate) struct FakeCudaRenderer(pub(crate) CudaArch);

impl svod_device::device::Renderer for FakeCudaRenderer {
    fn render(
        &self,
        ast: &std::sync::Arc<UOp>,
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

    fn supported_ops(&self) -> RendererOps {
        RendererOps::all()
    }

    fn decompositor(&self) -> Option<TypedPatternMatcher> {
        Some(svod_ir::decompositions::nvptx_decomposition_patterns())
    }

    fn extra_matcher(&self) -> Option<TypedPatternMatcher> {
        Some(crate::devectorize::bool_storage_patterns().clone())
    }
}
