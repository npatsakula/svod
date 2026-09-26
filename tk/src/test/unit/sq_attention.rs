use std::sync::Arc;

use svod_dtype::{AmdArch, CudaArch, DType, DeviceSpec, GpuArch};
use svod_ir::{Op, UOp};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::kernels::sq_attention::{
    HeadSelection, SQ_ATTENTION_SUPPORTED_ARCHS, SqAttentionOpts, SqGeom, SqMask, build_single_query_attention,
    build_single_query_attention_merge, build_single_query_attention_partial,
};
use crate::{ArchCaps, Kernel};
use svod_ir::ops;

const SM_86: GpuArch = GpuArch::Cuda(CudaArch::from_compute_capability(8, 6));

/// Every arch the kernel is built for: the AMD pair plus CUDA warp32.
fn all_caps() -> [ArchCaps; 3] {
    [ArchCaps::GFX942, ArchCaps::for_amd(AmdArch::Gfx1151), ArchCaps::for_arch(SM_86)]
}

/// Every mask the single-kernel builder is built for.
const MASKS: [SqMask; 3] = [SqMask::Whole, SqMask::PrefixAndLast, SqMask::PrefixAndAppended];

/// The ABI in declaration order: `out, q, k, v, [key_lens], [appended k, v]`.
fn buffers(b: usize, n: usize, h: usize, d: usize, mask: SqMask) -> Vec<Arc<UOp>> {
    let mut bufs = vec![
        UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::Float32),
    ];
    if mask != SqMask::Whole {
        bufs.push(UOp::new_buffer(DeviceSpec::Cpu, b, DType::Int32));
    }
    if mask == SqMask::PrefixAndAppended {
        bufs.extend([
            UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
            UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        ]);
    }
    bufs
}

fn sink(caps: ArchCaps, mask: SqMask) -> Arc<UOp> {
    let (b, n, h, h_total, d, head_offset) = (2, 5, 3, 7, 64, 2);
    let ker = Kernel::new(
        "sq_attention",
        [h as i64, b as i64, 1],
        caps.wave_size as i64,
        buffers(b, n, h_total, d, mask),
        caps,
    );
    let heads = HeadSelection { count: h, total: h_total, offset: head_offset };
    build_single_query_attention(&ker, SqGeom { b, kv_batch: b, n, heads, d, kv: DType::Float32 }, mask, false);
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
    build_single_query_attention_partial(
        &partial,
        SqGeom { b, kv_batch: b, n, heads, d, kv: DType::Float32 },
        splits,
        false,
    );
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
        for mask in MASKS {
            let topo = sink(caps, mask).toposort();
            let shuffles = topo.iter().filter(|u| matches!(u.op(), Op::Custom(..))).count();
            // The appended key is scored by a second, peeled dot product outside
            // the loop, so it carries its own XOR reduction.
            let reductions = 1 + usize::from(mask == SqMask::PrefixAndAppended);
            assert_eq!(
                shuffles,
                reductions * caps.wave_size.ilog2() as usize,
                "{:?} {mask:?}: {reductions} XOR reduction(s)",
                caps.arch
            );
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
        ("sq_attention", sink(caps, SqMask::PrefixAndLast), true, false),
        ("sq_attention_appended", sink(caps, SqMask::PrefixAndAppended), true, false),
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
    super::device_supported(SQ_ATTENTION_SUPPORTED_ARCHS)
}

#[test]
fn sq_attention_packed_validates_head_geometry() {
    let q = Tensor::empty(&[1, 1, 3, 64], DType::Float32);
    let k = Tensor::empty(&[1, 5, 4, 64], DType::Float32);
    let v = Tensor::empty(&[1, 5, 4, 64], DType::Float32);
    let err = crate::single_query_attention_packed(&q, &k, &v, 2, SqAttentionOpts::default())
        .expect_err("out-of-range heads");
    assert!(matches!(err, crate::LaunchError::OperandDimMismatch { .. }));

    let bad_v = Tensor::empty(&[1, 5, 5, 64], DType::Float32);
    let err = crate::single_query_attention_packed(&q, &k, &bad_v, 1, SqAttentionOpts::default())
        .expect_err("mismatched total heads");
    assert!(matches!(err, crate::LaunchError::OperandDimMismatch { .. }));
}

/// Attention over the cached keys a row may read, plus — where `appended` is
/// given as one `[B,1,H,D]` K/V pair — one more key outside the cache.
///
/// Without `appended`, `lens` follows the in-cache convention the kernel's
/// `include_last` mode implements: keys `0..lens[b]` and the final slot.
/// The appended key replaces the cache's final slot, so it needs a prefix to
/// follow and must match the selected heads and the cache's own width.
#[test]
fn sq_attention_validates_the_appended_key() {
    let (b, n, h, h_total, d) = (2usize, 5usize, 3usize, 7usize, 64usize);
    let q = Tensor::empty(&[b, 1, h, d], DType::Float32);
    let k = Tensor::empty(&[b, n, h_total, d], DType::Float16);
    let v = Tensor::empty(&[b, n, h_total, d], DType::Float16);
    let lens = Tensor::empty(&[b], DType::Int32);
    let (ak, av) = (Tensor::empty(&[b, 1, h, d], DType::Float16), Tensor::empty(&[b, 1, h, d], DType::Float16));
    let launch = |opts| crate::single_query_attention_packed(&q, &k, &v, 2, opts);

    let opts = SqAttentionOpts { appended: Some((&ak, &av)), ..Default::default() };
    assert!(
        matches!(launch(opts).expect_err("appended without key_lens"), crate::LaunchError::DimMultiple { .. }),
        "an appended key with no prefix to follow"
    );
    let opts =
        SqAttentionOpts { key_lens: Some(&lens), include_last: true, appended: Some((&ak, &av)), ..Default::default() };
    assert!(
        matches!(launch(opts).expect_err("both appended forms"), crate::LaunchError::DimMultiple { .. }),
        "the current token cannot be in the cache and beside it at once"
    );
    // The appended row is indexed by the selected head, so it carries `H` heads,
    // not the cache's `H_total`.
    let packed = Tensor::empty(&[b, 1, h_total, d], DType::Float16);
    let opts = SqAttentionOpts { key_lens: Some(&lens), appended: Some((&packed, &av)), ..Default::default() };
    assert!(
        matches!(launch(opts).expect_err("packed appended heads"), crate::LaunchError::OperandShape { .. }),
        "an appended key shaped like the packed cache"
    );
    let wide = Tensor::empty(&[b, 1, h, d], DType::Float32);
    let opts = SqAttentionOpts { key_lens: Some(&lens), appended: Some((&ak, &wide)), ..Default::default() };
    assert!(
        matches!(launch(opts).expect_err("appended dtype"), crate::LaunchError::Dtype { .. }),
        "an appended value wider than the cache it extends"
    );
}

fn cpu_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dims: (usize, usize, usize, usize, usize),
    head_offset: usize,
    lens: Option<&[i32]>,
    appended: Option<(&[f32], &[f32])>,
) -> Vec<f32> {
    let (b, n, h, h_total, d) = dims;
    let mut out = vec![0.0; b * h * d];
    for bi in 0..b {
        for hi in 0..h {
            let valid = |ni: usize| match lens {
                None => true,
                Some(ls) => ni < ls[bi] as usize || (appended.is_none() && ni + 1 == n),
            };
            let row = (bi * h + hi) * d;
            let mut keys: Vec<(&[f32], &[f32])> = (0..n)
                .filter(|&ni| valid(ni))
                .map(|ni| {
                    let o = ((bi * n + ni) * h_total + hi + head_offset) * d;
                    (&k[o..o + d], &v[o..o + d])
                })
                .collect();
            if let Some((ak, av)) = appended {
                keys.push((&ak[row..row + d], &av[row..row + d]));
            }
            let scores: Vec<f32> = keys
                .iter()
                .map(|(kr, _)| kr.iter().zip(&q[row..row + d]).map(|(a, b)| a * b).sum::<f32>() / (d as f32).sqrt())
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|x| (*x - max).exp()).collect();
            let norm: f32 = weights.iter().sum();
            for ((_, vr), w) in keys.iter().zip(&weights) {
                for di in 0..d {
                    out[row + di] += w * vr[di] / norm;
                }
            }
        }
    }
    out
}

/// One K/V cache serving every batch row must decode exactly as the same cache
/// replicated per row. Beam search reads an identical cross-attention cache from
/// every hypothesis, so binding it once is the difference between reading the
/// largest tensor in the step `B` times and reading it once.
///
/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib sq_attention_broadcast -- --ignored`.
#[test]
#[ignore]
fn sq_attention_broadcast_cache_matches_a_replicated_one() {
    if !supported_device() {
        eprintln!("skip sq_attention_broadcast_cache_matches_a_replicated_one: unsupported device/toolchain");
        return;
    }
    // Whisper's cross-attention geometry: five hypotheses over one 1500-frame window.
    let (b, n, h, h_total, d, head_offset) = (5usize, 1500usize, 20usize, 24usize, 64usize, 2usize);
    let q = Tensor::randn(&[b, 1, h, d]).expect("q");
    let shared_k = Tensor::randn(&[1, n, h_total, d]).expect("shared k");
    let shared_v = Tensor::randn(&[1, n, h_total, d]).expect("shared v");
    for t in [&q, &shared_k, &shared_v] {
        t.realize().expect("realize");
    }
    // The same bytes, laid out once per row, is what the kernel used to require.
    let tile = |t: &Tensor| {
        let wide = t.try_expand([b, n, h_total, d]).expect("expand").contiguous();
        wide.realize().expect("realize tiled");
        wide
    };
    let (wide_k, wide_v) = (tile(&shared_k), tile(&shared_v));

    for split in [1usize, 4] {
        let run = |k: &Tensor, v: &Tensor| {
            let opts = SqAttentionOpts { split: Some(split), ..Default::default() };
            let out = crate::single_query_attention_packed(&q, k, v, head_offset, opts)
                .expect("sq attention")
                .expect("supported");
            out.realize().expect("realize");
            out.as_vec::<f32>().expect("vec")
        };
        let shared = run(&shared_k, &shared_v);
        let replicated = run(&wide_k, &wide_v);
        assert_eq!(shared.len(), replicated.len());
        let max_abs = shared.iter().zip(&replicated).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
        assert_eq!(max_abs, 0.0, "split {split}: broadcast diverged from the replicated cache by {max_abs}");
    }
}

/// A narrowed cache must agree with the f32 one to within f16 quantization.
///
/// Whisper projects cross-K/V in f16 and only widened them to f32 because this
/// kernel demanded it, which doubled both the cache (2.3 GiB at large-v3) and the
/// bytes the kernel streams. Reading f16 directly is only sound if the softmax
/// still runs in f32, so this pins the output, not just the dtype plumbing.
///
/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib sq_attention_f16 -- --ignored`.
#[test]
#[ignore]
fn sq_attention_f16_cache_matches_f32_within_quantization() {
    if !supported_device() {
        eprintln!("skip sq_attention_f16_cache_matches_f32_within_quantization: unsupported device/toolchain");
        return;
    }
    let (b, n, h, h_total, d, head_offset) = (5usize, 1500usize, 20usize, 24usize, 64usize, 2usize);
    let q = Tensor::randn(&[b, 1, h, d]).expect("q");
    let k32 = Tensor::randn(&[b, n, h_total, d]).expect("k");
    let v32 = Tensor::randn(&[b, n, h_total, d]).expect("v");
    // Round-trip through f16 so the f32 run sees exactly the values the f16 run
    // will: this isolates the kernel's arithmetic from the cast's rounding.
    let narrow = |t: &Tensor| t.cast(DType::Float16);
    let widen = |t: &Tensor| t.cast(DType::Float16).cast(DType::Float32);
    let (k16, v16) = (narrow(&k32), narrow(&v32));
    let (k_ref, v_ref) = (widen(&k32), widen(&v32));
    for t in [&q, &k16, &v16, &k_ref, &v_ref] {
        t.realize().expect("realize");
    }

    for split in [1usize, 4] {
        let run = |k: &Tensor, v: &Tensor| {
            let opts = SqAttentionOpts { split: Some(split), ..Default::default() };
            let out = crate::single_query_attention_packed(&q, k, v, head_offset, opts)
                .expect("sq attention")
                .expect("supported");
            out.realize().expect("realize");
            out.as_vec::<f32>().expect("vec")
        };
        let reference = run(&k_ref, &v_ref);
        let narrowed = run(&k16, &v16);
        assert_eq!(reference.len(), narrowed.len());
        let max_abs = narrowed.iter().zip(&reference).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
        assert!(max_abs < 5e-3, "split {split}: f16 cache diverged from f32 by {max_abs:e}");
    }
}

/// With rows pointing at different caches, each must read the one it names.
///
/// This is the shape a batched decoder actually has: several hypothesis sets in
/// flight, each against its own audio window, so neither "one cache for all rows"
/// nor "a cache per row" describes it. The map lets one copy serve a whole
/// hypothesis set without serializing the windows.
///
/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib sq_attention_cache_map -- --ignored`.
#[test]
#[ignore]
fn sq_attention_cache_map_reads_the_row_it_names() {
    if !supported_device() {
        eprintln!("skip sq_attention_cache_map_reads_the_row_it_names: unsupported device/toolchain");
        return;
    }
    // Four query rows over two windows, interleaved so a row never reads its own index.
    let (b, caches, n, h, h_total, d, head_offset) = (4usize, 2usize, 256usize, 3usize, 7usize, 64usize, 2usize);
    let owners = [0i32, 1, 0, 1];
    let q = Tensor::randn(&[b, 1, h, d]).expect("q");
    let k = Tensor::randn(&[caches, n, h_total, d]).expect("k");
    let v = Tensor::randn(&[caches, n, h_total, d]).expect("v");
    for t in [&q, &k, &v] {
        t.realize().expect("realize");
    }
    let map = Tensor::from_slice(owners.as_slice());
    map.realize().expect("realize map");

    // The same caches laid out one per row — what the kernel required before.
    let widen = |t: &Tensor| {
        let rows: Vec<Tensor> = owners.iter().map(|&o| t.narrow(0, o as usize, 1usize).expect("row")).collect();
        let wide = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0).expect("cat").contiguous();
        wide.realize().expect("realize wide");
        wide
    };
    let (wide_k, wide_v) = (widen(&k), widen(&v));

    for split in [1usize, 4] {
        let mapped = crate::single_query_attention_packed(
            &q,
            &k,
            &v,
            head_offset,
            SqAttentionOpts { cache_map: Some(&map), split: Some(split), ..Default::default() },
        )
        .expect("mapped")
        .expect("supported");
        mapped.realize().expect("realize mapped");
        let replicated = crate::single_query_attention_packed(
            &q,
            &wide_k,
            &wide_v,
            head_offset,
            SqAttentionOpts { split: Some(split), ..Default::default() },
        )
        .expect("replicated")
        .expect("supported");
        replicated.realize().expect("realize replicated");

        let a = mapped.as_vec::<f32>().expect("mapped vec");
        let e = replicated.as_vec::<f32>().expect("replicated vec");
        let max_abs = a.iter().zip(&e).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert_eq!(max_abs, 0.0, "split {split}: mapped cache diverged from the replicated one by {max_abs}");
    }
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
    let q = Tensor::randn(&[b, 1, h, d]).expect("q");
    let k = Tensor::randn(&[b, n, h_total, d]).expect("k");
    let v = Tensor::randn(&[b, n, h_total, d]).expect("v");
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
            let opts = SqAttentionOpts {
                key_lens: lens_t.as_ref(),
                include_last: lens.is_some(),
                split: Some(split),
                ..Default::default()
            };
            let got = crate::single_query_attention_packed(&q, &k, &v, head_offset, opts)
                .expect("sq attention")
                .expect("supported");
            got.realize().expect("realize output");
            let got = got.as_vec::<f32>().expect("output vec");
            let expected = cpu_reference(&qv, &kv, &vv, (b, n, h, h_total, d), head_offset, lens.as_deref(), None);
            let max_abs = got.iter().zip(&expected).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
            assert!(max_abs < 2e-4, "split {split} max abs error {max_abs}");
        }
    }

    // Production cross-cache geometry. A 150-key chunk leaves a ragged width-8
    // tile on both wave64 (8 groups) and wave32 (4 groups).
    let (b, n, h, h_total, d, head_offset) = (5, 1500, 20, 24, 64, 2);
    let q = Tensor::randn(&[b, 1, h, d]).expect("production q");
    let k = Tensor::randn(&[b, n, h_total, d]).expect("production k");
    let v = Tensor::randn(&[b, n, h_total, d]).expect("production v");
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
        None,
    );
    let got = crate::single_query_attention_packed(
        &q,
        &k,
        &v,
        head_offset,
        SqAttentionOpts { split: Some(10), ..Default::default() },
    )
    .expect("production sq attention")
    .expect("production supported");
    got.realize().expect("realize production output");
    let got = got.as_vec::<f32>().expect("production output vec");
    let max_abs = got.iter().zip(&expected).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 2e-4, "production split 10 max abs error {max_abs}");
}

/// The decoder step never splices the K/V it just projected into the cache: it
/// hands them over as their own `[B,1,H,D]` pair and the kernel scores them after
/// the prefix. Every prefix a row can present must come out right — nothing
/// decoded yet (an empty loop whose state the peeled update alone normalizes), a
/// single key, and a prefix that fills the cache — against a packed cache read at
/// its stored width from a head offset of its own.
///
/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib sq_attention_appended -- --ignored`.
#[test]
#[ignore]
fn sq_attention_appended_key_matches_the_cpu_reference() {
    if !supported_device() {
        eprintln!("skip sq_attention_appended_key_matches_the_cpu_reference: unsupported device/toolchain");
        return;
    }
    let (b, n, h, h_total, d, head_offset) = (5usize, 16usize, 3usize, 7usize, 64usize, 2usize);
    let lens = [0i32, 1, 9, 15, 16];
    let lens_t = Tensor::from_slice(lens.as_slice());
    lens_t.realize().expect("realize lens");

    for (dtype, tolerance) in [(DType::Float32, 2e-4f32), (DType::Float16, 5e-3)] {
        let cached = |shape: &[usize]| {
            let t = Tensor::randn(shape).expect("operand").cast(dtype.clone());
            t.realize().expect("realize operand");
            t
        };
        let q = Tensor::randn(&[b, 1, h, d]).expect("q");
        q.realize().expect("realize q");
        let (k, v) = (cached(&[b, n, h_total, d]), cached(&[b, n, h_total, d]));
        let (append_k, append_v) = (cached(&[b, 1, h, d]), cached(&[b, 1, h, d]));

        let got = crate::single_query_attention_packed(
            &q,
            &k,
            &v,
            head_offset,
            SqAttentionOpts { key_lens: Some(&lens_t), appended: Some((&append_k, &append_v)), ..Default::default() },
        )
        .expect("appended sq attention")
        .expect("supported");
        got.realize().expect("realize output");

        // The reference sees exactly the values the kernel loads, so a narrowed
        // cache is compared against its own rounding, not the f32 draw.
        let widen = |t: &Tensor| {
            let w = t.cast(DType::Float32);
            w.realize().expect("realize widened");
            w.as_vec::<f32>().expect("widened vec")
        };
        let (wide_ak, wide_av) = (widen(&append_k), widen(&append_v));
        let expected = cpu_reference(
            &q.as_vec::<f32>().expect("q vec"),
            &widen(&k),
            &widen(&v),
            (b, n, h, h_total, d),
            head_offset,
            Some(&lens),
            Some((&wide_ak, &wide_av)),
        );
        let got = got.as_vec::<f32>().expect("output vec");
        let max_abs = got.iter().zip(&expected).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
        assert!(max_abs < tolerance, "{dtype:?}: appended key diverged from the reference by {max_abs:e}");
    }
}

/// Appending must decode as splicing did. The whisper step used to concatenate
/// the layer's cache slice with the projected row and ask for the final slot,
/// which copies the slice once per layer; the two forms score the same keys in
/// the same order, so only the copy is gone.
///
/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib sq_attention_appended -- --ignored`.
#[test]
#[ignore]
fn sq_attention_appended_key_matches_a_spliced_cache() {
    if !supported_device() {
        eprintln!("skip sq_attention_appended_key_matches_a_spliced_cache: unsupported device/toolchain");
        return;
    }
    // A short cache the lengths can span end to end, then whisper large-v3's own
    // self-attention geometry: five hypotheses over a 448-position cache packed
    // 32 layers deep, read from one layer's head offset.
    for (b, n, h, h_total, d, head_offset, lens) in [
        (5usize, 16usize, 3usize, 7usize, 64usize, 2usize, vec![0i32, 1, 9, 15, 16]),
        (5, 448, 20, 640, 64, 380, vec![0i32, 1, 200, 447, 448]),
    ] {
        let lens = {
            let t = Tensor::from_slice(lens.as_slice());
            t.realize().expect("realize lens");
            t
        };
        let f16 = |shape: &[usize]| {
            let t = Tensor::randn(shape).expect("operand").cast(DType::Float16);
            t.realize().expect("realize operand");
            t
        };
        let q = Tensor::randn(&[b, 1, h, d]).expect("q");
        q.realize().expect("realize q");
        let (k, v) = (f16(&[b, n, h_total, d]), f16(&[b, n, h_total, d]));
        let (append_k, append_v) = (f16(&[b, 1, h, d]), f16(&[b, 1, h, d]));

        let appended = crate::single_query_attention_packed(
            &q,
            &k,
            &v,
            head_offset,
            SqAttentionOpts { key_lens: Some(&lens), appended: Some((&append_k, &append_v)), ..Default::default() },
        )
        .expect("appended")
        .expect("supported");

        let splice = |cache: &Tensor, new: &Tensor| {
            let spliced = Tensor::cat(&[&cache.narrow(2, head_offset, h).expect("layer"), new], 1).expect("splice");
            spliced.realize().expect("realize spliced");
            spliced
        };
        let (spliced_k, spliced_v) = (splice(&k, &append_k), splice(&v, &append_v));
        let spliced = crate::single_query_attention(
            &q,
            &spliced_k,
            &spliced_v,
            SqAttentionOpts { key_lens: Some(&lens), include_last: true, ..Default::default() },
        )
        .expect("spliced")
        .expect("supported");
        Tensor::realize_batch([&appended, &spliced]).expect("realize both");

        let (a, e) = (appended.as_vec::<f32>().expect("appended vec"), spliced.as_vec::<f32>().expect("spliced vec"));
        let max_abs = a.iter().zip(&e).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(max_abs < 1e-6, "n {n}: the appended key diverged from the spliced cache by {max_abs:e}");
    }
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
    let q = Tensor::zeros(&[b, 1, h, d], DType::Float32);
    let k = Tensor::zeros(&[b, n, h, d], DType::Float32);
    let v = Tensor::zeros(&[b, n, h, d], DType::Float32);
    let out = crate::single_query_attention(&q, &k, &v, SqAttentionOpts::default()).expect("declines, not errors");
    assert!(out.is_none(), "head dim 4 divides no supported wave size (32/64); the launch must fall back");
}

// ─── Split policy ────────────────────────────────────────────────────────────

/// The policy aims the split at the wave budget over divisors that keep a
/// chunk, preferring chunks with no partial tail; an explicit zero budget
/// keeps one split, and so does a key range too short to chunk.
#[test_case(GpuArch::Amd(AmdArch::Gfx1151), (40, 32), 5, 20, 1500, 15, &[5, 10, 12, 15]; "whisper large cross attention on a 40-cu rdna part")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1151), (40, 32), 1, 6, 1500, 25, &[3, 5, 15, 25]; "tiny heads want more splits")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1151), (40, 32), 5, 20, 64, 1, &[]; "a short key range stays whole")]
#[test_case(GpuArch::Amd(AmdArch::Gfx942), (304, 32), 5, 20, 1500, 12, &[5, 6, 10, 12]; "a 304-cu wave64 part is capped by its 120-key floor")]
#[test_case(SM_86, (28, 16), 5, 20, 1500, 5, &[3, 4, 5, 6]; "whisper large cross attention on a 28-sm ampere part")]
#[test_case(SM_86, (28, 0), 5, 20, 1500, 1, &[]; "an explicit zero budget keeps one split")]
fn sq_policy_splits_toward_the_wave_budget(
    arch: GpuArch,
    (compute_units, waves_per_cu): (usize, usize),
    b: usize,
    h: usize,
    n: usize,
    split: usize,
    candidates: &[usize],
) {
    let policy = crate::SqPolicy::with_budget(arch, compute_units, waves_per_cu);
    assert_eq!(policy.split(b, h, n), split);
    let mut sorted = policy.candidates(b, h, n);
    sorted.sort_unstable();
    let mut expected = candidates.to_vec();
    expected.sort_unstable();
    assert_eq!(sorted, expected);
}

/// Every candidate the policy hands the tuner is a legal split of `n` that
/// leaves each wave its floor of loop trips, on either wave size.
#[test_case(GpuArch::Amd(AmdArch::Gfx1151); "wave32")]
#[test_case(GpuArch::Amd(AmdArch::Gfx942); "wave64")]
fn sq_policy_candidates_divide_the_keys(arch: GpuArch) {
    let policy = crate::SqPolicy::with_budget(arch, 40, 32);
    for n in [448, 1500, 1536, 3000] {
        for split in policy.candidates(5, 20, n) {
            assert!(split > 1, "n {n} split {split}");
            assert_eq!(n % split, 0, "n {n} split {split}");
            assert!(n / split >= policy.min_chunk, "n {n} split {split}");
        }
    }
}

/// The policy's split runs the two-kernel path and matches the single-kernel
/// answer: the whisper geometry, one shared f16 cache addressed by a map.
#[test]
fn sq_policy_split_matches_a_single_split() {
    if !supported_device() {
        return;
    }
    let (b, n, h, d) = (5, 1500, 20, 64);
    let device = Tensor::empty(&[1], DType::Float32).device();
    let q = Tensor::randn(&[b, 1, h, d]).expect("q").to(device.clone());
    let k = Tensor::randn(&[1, n, h, d]).expect("k").cast(DType::Float16).to(device.clone());
    let v = Tensor::randn(&[1, n, h, d]).expect("v").cast(DType::Float16).to(device.clone());
    let map = Tensor::zeros(&[b], DType::Int32).to(device);
    let run = |split: Option<usize>| {
        let out = crate::single_query_attention(
            &q,
            &k,
            &v,
            SqAttentionOpts { split, cache_map: Some(&map), ..Default::default() },
        )
        .expect("launch")
        .expect("supported");
        out.realize().expect("realize");
        out.as_vec::<f32>().expect("vec")
    };
    crate::tune::set_enabled(false);
    let (whole, policy) = (run(Some(1)), run(None));
    let max_abs = whole.iter().zip(&policy).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 1e-4, "policy split max abs error {max_abs}");
}

/// The device policy carries the device's own counts, not the family's
/// reference device. `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib sq_policy_reads -- --ignored --nocapture`.
#[test]
#[ignore]
fn sq_policy_reads_the_device_budget_gpu() {
    if !supported_device() {
        return;
    }
    let spec = svod_tensor::Tensor::empty(&[1], DType::Float32).device();
    let arch = crate::target::resolve_arch(&spec).expect("a GPU arch");
    let policy = crate::SqPolicy::for_device(&spec, arch);
    eprintln!("{policy:?}");
    match crate::target::compute_units(&spec).zip(crate::target::resident_waves_per_cu(&spec)) {
        Some(budget) => assert_eq!((policy.compute_units, policy.waves_per_cu), budget),
        // A part the launch supports but the probe does not describe.
        None => assert!(policy.split(5, 20, 1500) > 1, "the assumed budget still splits"),
    }
}

/// A device whose probe reports no wave budget — the KFD node without
/// `simd_per_cu`, the CUDA limits that failed to open — still splits a long
/// cross attention: the assumed budget stands in, so large-v3's 1500 keys do
/// not run as one latency-bound wave per `(row, head)`.
#[test_case(GpuArch::Amd(AmdArch::Gfx1151); "rdna wave32")]
#[test_case(SM_86; "ampere")]
fn sq_policy_falls_back_when_the_device_reports_no_budget(arch: GpuArch) {
    // No backend probe answers for a host spec: the unreported path.
    let policy = crate::SqPolicy::for_device(&DeviceSpec::Cpu, arch);
    assert_ne!(policy.waves_per_cu, 0, "an unreported budget must not disable the split");
    assert_eq!(policy.split(5, 20, 1500), 5, "whisper large-v3 cross attention");
    assert!(!policy.candidates(5, 20, 1500).is_empty(), "the tuner still gets candidates");
    // A short key range is still whole, and an explicit zero budget still opts out.
    assert_eq!(policy.split(5, 20, 64), 1);
    assert_eq!(crate::SqPolicy::with_budget(arch, 40, 0).split(5, 20, 1500), 1);
}
