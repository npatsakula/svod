use std::sync::Arc;

use svod_dtype::{AmdArch, CudaArch, DType, DeviceSpec, GpuArch};
use svod_ir::{Op, UOp};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::kernels::sq_attention::{
    HeadSelection, SQ_ATTENTION_SUPPORTED_ARCHS, SqAttentionOpts, build_single_query_attention,
    build_single_query_attention_merge, build_single_query_attention_partial,
};
use crate::{ArchCaps, Kernel};
use svod_ir::ops;

const SM_86: GpuArch = GpuArch::Cuda(CudaArch::from_compute_capability(8, 6));

/// Every arch the kernel is built for: the AMD pair plus CUDA warp32.
fn all_caps() -> [ArchCaps; 3] {
    [ArchCaps::GFX942, ArchCaps::for_amd(AmdArch::Gfx1151), ArchCaps::for_arch(SM_86)]
}

fn buffers(b: usize, n: usize, h: usize, d: usize, masked: bool) -> Vec<Arc<UOp>> {
    let mut bufs = vec![
        UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::Float32),
    ];
    if masked {
        bufs.push(UOp::new_buffer(DeviceSpec::Cpu, b, DType::Int32));
    }
    bufs
}

fn sink(caps: ArchCaps, masked: bool) -> Arc<UOp> {
    let (b, n, h, h_total, d, head_offset) = (2, 5, 3, 7, 64, 2);
    let ker = Kernel::new(
        "sq_attention",
        [h as i64, b as i64, 1],
        caps.wave_size as i64,
        buffers(b, n, h_total, d, masked),
        caps,
    );
    let heads = HeadSelection { count: h, total: h_total, offset: head_offset };
    build_single_query_attention(&ker, b, n, heads, d, masked, masked);
    ker.finish(1)
}

fn split_sinks(caps: ArchCaps, splits: usize, d: usize) -> (Arc<UOp>, Arc<UOp>) {
    let (b, n, h, h_total, head_offset) = (2, 20, 3, 7, 2);
    let partial_buffers = vec![
        UOp::new_buffer(DeviceSpec::Cpu, b * splits * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * splits * h * 2, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h_total * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h_total * d, DType::Float32),
    ];
    let partial = Kernel::new(
        "sq_attention_partial",
        [h as i64, b as i64, splits as i64],
        caps.wave_size as i64,
        partial_buffers,
        caps,
    );
    let heads = HeadSelection { count: h, total: h_total, offset: head_offset };
    build_single_query_attention_partial(&partial, b, n, heads, d, splits);
    let partial = partial.finish(2);

    let merge_buffers = vec![
        UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * splits * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * splits * h * 2, DType::Float32),
    ];
    let merge = Kernel::new("sq_attention_merge", [h as i64, b as i64, 1], caps.wave_size as i64, merge_buffers, caps);
    build_single_query_attention_merge(&merge, b, h, d, splits);
    (partial, merge.finish(1))
}

#[test]
fn sq_attention_graph_shape_all_arches() {
    for caps in all_caps() {
        for masked in [false, true] {
            let topo = sink(caps, masked).toposort();
            let shuffles = topo.iter().filter(|u| matches!(u.op(), Op::Custom(..))).count();
            assert_eq!(shuffles, caps.wave_size.ilog2() as usize, "{:?}: one XOR reduction", caps.arch);
            assert!(
                topo.iter().any(|u| matches!(u.op(), Op::Unary(svod_ir::UnaryOp::Exp2, ..))),
                "{:?}: exp2",
                caps.arch
            );
            assert!(topo.iter().any(|u| matches!(u.op(), Op::Range(..))), "{:?}: streamed N loop", caps.arch);
            assert!(
                !topo.iter().any(
                    |u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local))
                ),
                "{:?}: no LDS",
                caps.arch
            );
            assert!(!topo.iter().any(|u| matches!(u.op(), Op::Wmma(..))), "{:?}: no MFMA/WMMA", caps.arch);
            assert!(!topo.iter().any(|u| matches!(u.op(), Op::Barrier(..))), "{:?}: no barrier", caps.arch);
        }
        for d in [64, 128] {
            let (partial, merge) = split_sinks(caps, 4, d);
            let partial_topo = partial.toposort();
            let partial_shuffles = partial_topo.iter().filter(|u| matches!(u.op(), Op::Custom(..))).count();
            let groups = caps.wave_size / 8;
            let expected = 3 + 2 * caps.wave_size.ilog2() as usize + groups;
            assert_eq!(
                partial_shuffles, expected,
                "{:?} D={d}: width-8 dot, tile reductions, and beta broadcasts",
                caps.arch
            );
            for (name, graph) in [("partial", partial), ("merge", merge)] {
                let topo = graph.toposort();
                assert!(topo.iter().any(|u| matches!(u.op(), Op::Range(..))), "{:?}: split {name} loop", caps.arch);
                assert!(
                    !topo.iter().any(
                        |u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local))
                    ),
                    "{:?}: split {name} no LDS",
                    caps.arch
                );
                assert!(
                    !topo.iter().any(|u| matches!(u.op(), Op::Wmma(..))),
                    "{:?}: split {name} no MFMA/WMMA",
                    caps.arch
                );
                assert!(
                    !topo.iter().any(|u| matches!(u.op(), Op::Barrier(..))),
                    "{:?}: split {name} no barrier",
                    caps.arch
                );
            }
        }
    }
}

/// The three kernels render on every supported arch with the arch's own
/// cross-lane primitive — `ds_bpermute` on AMD, `shfl.sync.bfly` (the XOR
/// reductions) plus `shfl.sync.idx` (the per-subgroup beta broadcast, partial
/// kernel only) on NVPTX — and never allocate LDS.
#[test_case(GpuArch::Amd(AmdArch::Gfx942), &["llvm.amdgcn.ds.bpermute"], &[]; "gfx942")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1151), &["llvm.amdgcn.ds.bpermute"], &[]; "gfx1151")]
#[test_case(SM_86, &["llvm.nvvm.shfl.sync.bfly.i32"], &["llvm.nvvm.shfl.sync.idx.i32"]; "sm_86")]
fn sq_attention_renders_per_arch(arch: GpuArch, reduce_intrinsics: &[&str], broadcast_intrinsics: &[&str]) {
    let caps = ArchCaps::for_arch(arch);
    let (renderer, opt_renderer) = match arch {
        GpuArch::Amd(amd) => {
            (svod_codegen::llvm::LlvmTextRenderer::amd(amd), svod_schedule::OptimizerRenderer::for_amd_arch(amd))
        }
        GpuArch::Cuda(cuda) => {
            (svod_codegen::llvm::LlvmTextRenderer::nvptx(cuda), svod_schedule::OptimizerRenderer::for_cuda_arch(cuda))
        }
        GpuArch::Metal(_) => unreachable!("no Metal case"),
    };
    let opt_renderer = opt_renderer.with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        svod_codegen::traits::Renderer::decompositor(&renderer),
        None,
    );
    let (partial, merge) = split_sinks(caps, 4, 64);
    for (name, graph, reduces, broadcasts) in [
        ("sq_attention", sink(caps, true), true, false),
        ("sq_attention_partial", partial, true, true),
        ("sq_attention_merge", merge, false, false),
    ] {
        let optimized =
            svod_schedule::apply_post_optimization_with_renderer(graph, &opt_renderer).expect("post optimization");
        let program =
            svod_codegen::program_pipeline::program_from_sink(optimized, DeviceSpec::Cpu).expect("final target graph");
        let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("linearize");
        let linear = linearized.toposort().into_iter().find(|u| matches!(u.op(), Op::Linear(..))).expect("LINEAR");
        let code = svod_codegen::traits::Renderer::render(&renderer, &linear, Some(name)).expect("render").code;
        for intrinsic in reduce_intrinsics {
            assert_eq!(code.contains(intrinsic), reduces, "{arch:?}: {name} reduce shuffle {intrinsic}");
        }
        for intrinsic in broadcast_intrinsics {
            assert_eq!(code.contains(intrinsic), broadcasts, "{arch:?}: {name} broadcast shuffle {intrinsic}");
        }
        assert!(!code.contains("@local"), "{arch:?}: {name} no LDS allocation");
    }
}

fn supported_device() -> bool {
    crate::target::check_target(&Tensor::empty(&[1], DType::Float32).device(), SQ_ATTENTION_SUPPORTED_ARCHS).is_ok()
}

#[test]
fn sq_attention_packed_validates_head_geometry() {
    let q = Tensor::empty(&[1, 1, 3, 64], DType::Float32);
    let k = Tensor::empty(&[1, 5, 4, 64], DType::Float32);
    let v = Tensor::empty(&[1, 5, 4, 64], DType::Float32);
    let err = crate::single_query_attention_packed(&q, &k, &v, 2, SqAttentionOpts::default())
        .err()
        .expect("out-of-range heads");
    assert!(matches!(err, crate::LaunchError::OperandDimMismatch { .. }));

    let bad_v = Tensor::empty(&[1, 5, 5, 64], DType::Float32);
    let err = crate::single_query_attention_packed(&q, &k, &bad_v, 1, SqAttentionOpts::default())
        .err()
        .expect("mismatched total heads");
    assert!(matches!(err, crate::LaunchError::OperandDimMismatch { .. }));
}

fn cpu_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dims: (usize, usize, usize, usize, usize),
    head_offset: usize,
    lens: Option<&[i32]>,
) -> Vec<f32> {
    let (b, n, h, h_total, d) = dims;
    let mut out = vec![0.0; b * h * d];
    for bi in 0..b {
        for hi in 0..h {
            let valid = |ni: usize| lens.is_none_or(|ls| ni < ls[bi] as usize || ni + 1 == n);
            let mut scores = vec![f32::NEG_INFINITY; n];
            for (ni, score) in scores.iter_mut().enumerate().filter(|(ni, _)| valid(*ni)) {
                let mut dot = 0.0;
                for di in 0..d {
                    dot += q[(bi * h + hi) * d + di] * k[((bi * n + ni) * h_total + hi + head_offset) * d + di];
                }
                *score = dot / (d as f32).sqrt();
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|x| (*x - max).exp()).collect();
            let norm: f32 = weights.iter().sum();
            for di in 0..d {
                for ni in 0..n {
                    out[(bi * h + hi) * d + di] +=
                        weights[ni] * v[((bi * n + ni) * h_total + hi + head_offset) * d + di] / norm;
                }
            }
        }
    }
    out
}

/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib sq_attention_numerical_gpu -- --ignored`.
#[test]
#[ignore]
fn sq_attention_numerical_gpu() {
    if !supported_device() {
        eprintln!("skip sq_attention_numerical_gpu: unsupported device/toolchain");
        return;
    }
    let (b, n, h, h_total, d, head_offset) = (2, 20, 3, 7, 64, 2);
    let mut q = Tensor::randn(&[b, 1, h, d]).expect("q");
    let mut k = Tensor::randn(&[b, n, h_total, d]).expect("k");
    let mut v = Tensor::randn(&[b, n, h_total, d]).expect("v");
    q.realize().expect("realize q");
    k.realize().expect("realize k");
    v.realize().expect("realize v");
    let qv = q.as_vec::<f32>().expect("q vec");
    let kv = k.as_vec::<f32>().expect("k vec");
    let vv = v.as_vec::<f32>().expect("v vec");

    for (lens, splits) in [(None, vec![1, 2, 4]), (Some(vec![7i32, 13]), vec![1])] {
        for split in splits {
            let mut lens_t = lens.as_ref().map(|x| Tensor::from_slice(x.as_slice()));
            if let Some(t) = &mut lens_t {
                t.realize().expect("realize lens");
            }
            let opts = SqAttentionOpts { key_lens: lens_t.as_ref(), include_last: lens.is_some(), split };
            let mut got = crate::single_query_attention_packed(&q, &k, &v, head_offset, opts)
                .expect("sq attention")
                .expect("supported");
            got.realize().expect("realize output");
            let got = got.as_vec::<f32>().expect("output vec");
            let expected = cpu_reference(&qv, &kv, &vv, (b, n, h, h_total, d), head_offset, lens.as_deref());
            let max_abs = got.iter().zip(&expected).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
            assert!(max_abs < 2e-4, "split {split} max abs error {max_abs}");
        }
    }

    // Production cross-cache geometry. A 150-key chunk leaves a ragged width-8
    // tile on both wave64 (8 groups) and wave32 (4 groups).
    let (b, n, h, h_total, d, head_offset) = (5, 1500, 20, 24, 64, 2);
    let mut q = Tensor::randn(&[b, 1, h, d]).expect("production q");
    let mut k = Tensor::randn(&[b, n, h_total, d]).expect("production k");
    let mut v = Tensor::randn(&[b, n, h_total, d]).expect("production v");
    q.realize().expect("realize production q");
    k.realize().expect("realize production k");
    v.realize().expect("realize production v");
    let expected = cpu_reference(
        &q.as_vec::<f32>().expect("production q vec"),
        &k.as_vec::<f32>().expect("production k vec"),
        &v.as_vec::<f32>().expect("production v vec"),
        (b, n, h, h_total, d),
        head_offset,
        None,
    );
    let mut got = crate::single_query_attention_packed(
        &q,
        &k,
        &v,
        head_offset,
        SqAttentionOpts { split: 10, ..Default::default() },
    )
    .expect("production sq attention")
    .expect("production supported");
    got.realize().expect("realize production output");
    let got = got.as_vec::<f32>().expect("production output vec");
    let max_abs = got.iter().zip(&expected).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 2e-4, "production split 10 max abs error {max_abs}");
}

/// A head dim the arch's wave size does not divide is a fit failure, not a
/// caller bug: the launch declines with `Ok(None)` so a dispatch layer (the
/// whisper decoder) falls back to its generic attention path. Host-portable:
/// off-GPU the arch resolution declines the same way, so the assertion holds
/// on every runner, and on real hardware it exercises the `applies` arm (no
/// supported arch has a wave size dividing 4).
#[test]
fn undivisible_head_dim_declines_instead_of_erroring() {
    let (b, n, h, d) = (1usize, 3usize, 2usize, 4usize);
    let q = Tensor::zeros(&[b, 1, h, d], DType::Float32).expect("q");
    let k = Tensor::zeros(&[b, n, h, d], DType::Float32).expect("k");
    let v = Tensor::zeros(&[b, n, h, d], DType::Float32).expect("v");
    let out = crate::single_query_attention(&q, &k, &v, SqAttentionOpts::default()).expect("declines, not errors");
    assert!(out.is_none(), "head dim 4 divides no supported wave size (32/64); the launch must fall back");
}
