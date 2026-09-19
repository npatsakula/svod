//! The channels-last convolution: tile applicability, a GPU-free build, and the
//! hardware-gated numerics against the graph's `conv2d`
//! (`SVOD_DEVICE={CUDA,AMD}:0 cargo test -p svod-tk --lib conv -- --ignored --nocapture`).

use svod_dtype::{AmdArch, DType, GpuArch};
use svod_ir::UOp;
use svod_tensor::Tensor;
use test_case::test_case;

use super::device_supported;
use crate::kernels::conv::{CONV_SUPPORTED_ARCHS, ConvGeom, build_conv, conv2d_nhwc, select_conv_cfg};
use crate::kernels::gemm::{Epilogue, GemmPolicy};

const RDNA4: GpuArch = GpuArch::Amd(AmdArch::Gfx1201);

fn geom(h: usize, cin: usize, cout: usize, k: usize, stride: usize) -> ConvGeom {
    ConvGeom { batch: 1, h, w: h, cin, cout, kh: k, kw: k, stride, pad: k / 2 }
}

/// Output extents and the GEMM shape follow PyTorch's `(h + 2p − k) / s + 1`.
#[test_case(geom(40, 192, 192, 3, 1), (40, 1600, 1728, 192); "3x3 stride 1")]
#[test_case(geom(80, 768, 768, 3, 2), (40, 1600, 6912, 768); "3x3 stride 2")]
#[test_case(geom(40, 384, 192, 1, 1), (40, 1600, 384, 192); "1x1")]
#[test_case(geom(20, 768, 768, 3, 1), (20, 400, 6912, 768); "20x20, a ragged M")]
fn geometry(g: ConvGeom, want: (usize, usize, usize, usize)) {
    assert_eq!((g.ho(), g.mkn().0, g.mkn().1, g.mkn().2), want);
}

/// A tile fits when a strip stays inside one tap (`cin` a multiple of `k_step`),
/// `cout` tiles by its N edge and K holds two strips; `M` may be ragged.
#[test_case(geom(40, 192, 192, 3, 1), true; "192 channels")]
#[test_case(geom(20, 768, 768, 3, 1), true; "ragged M of 400")]
#[test_case(geom(160, 48, 48, 3, 1), false; "48 channels do not fill a strip")]
#[test_case(geom(80, 384, 96, 3, 1), false; "96 output channels miss the N edge")]
#[test_case(geom(40, 384, 192, 1, 1), true; "1x1")]
#[test_case(geom(320, 3, 96, 3, 2), false; "an RGB stem")]
fn rdna4_tiles(g: ConvGeom, served: bool) {
    let cfg = select_conv_cfg(&GemmPolicy::for_arch(RDNA4), &g);
    assert_eq!(cfg.is_some(), served, "{cfg:?}");
    if let Some(cfg) = cfg {
        assert!(g.tiles(&cfg) && !cfg.l2_swizzle);
        let grid = g.grid_dims(&cfg);
        assert_eq!((grid[0] * grid[1]) as usize, g.blocks(&cfg));
        assert!(grid[1] as usize * cfg.block_m >= g.mkn().0, "the grid covers every output row");
    }
}

/// The kernel builds off the GPU, on every arch of the table, with and without a
/// residual — the row gather, the ragged store and the epilogue all lower.
#[test_case(RDNA4, geom(40, 192, 192, 3, 1), true; "rdna4 3x3 with residual")]
#[test_case(RDNA4, geom(20, 768, 768, 3, 2), false; "rdna4 ragged M")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1151), geom(40, 384, 192, 1, 1), false; "rdna3 1x1")]
#[test_case(GpuArch::Cuda(svod_dtype::CudaArch::from_compute_capability(8, 6)), geom(40, 192, 192, 3, 1), true; "sm86")]
fn the_kernel_builds(arch: GpuArch, g: ConvGeom, residual: bool) {
    let caps = crate::ArchCaps::for_arch(arch);
    let cfg = select_conv_cfg(&GemmPolicy::for_arch(arch), &g).expect("a tile");
    let (m, k, n) = g.mkn();
    let dt = DType::Float16;
    let mut sizes = vec![m * n, g.batch * g.h * g.w * g.cin, n * k, n];
    if residual {
        sizes.push(m * n);
    }
    let bufs = sizes.into_iter().map(|s| UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, s, dt.clone())).collect();
    let ker = crate::Kernel::new("conv2d_nhwc", g.grid_dims(&cfg), cfg.threads(caps.wave_size), bufs, caps);
    build_conv(&ker, g, cfg, dt, Epilogue::BiasAct { bias: (), residual: residual.then_some(()), act: true });
    let sink = ker.finish(cfg.acc_m);
    assert!(crate::kernel_fingerprint(&sink).digest != 0);
}

// ── Hardware-gated numerics ─────────────────────────────────────────────────

fn to_f32_vec(t: &Tensor) -> Vec<f32> {
    let f = t.cast(DType::Float32).contiguous();
    f.realize().expect("realize f32");
    f.as_vec::<f32>().expect("read f32")
}

fn operand(shape: &[usize], dtype: DType, seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5).collect();
    let t = Tensor::from_slice(v)
        .try_reshape(shape.iter().map(|&d| d as isize).collect::<Vec<_>>())
        .expect("reshape")
        .cast(dtype)
        .contiguous();
    t.realize().expect("realize operand");
    t
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |a, b| a.max(b.abs())).max(f32::MIN_POSITIVE);
    got.iter().zip(want).fold(0f32, |a, (g, w)| a.max((g - w).abs())) / scale
}

/// The kernel against the graph's `conv2d` over the same channels-last operands
/// (the graph reads them through permuted views), bias, SiLU and residual
/// included: both accumulate in f32 and round to the operand dtype in the same
/// places, so they differ by a summation order, two ulps at most.
#[test_case(geom(32, 384, 192, 1, 1), false, DType::Float16, 4e-3; "1x1, aligned M")]
#[test_case(ConvGeom { batch: 1, h: 34, w: 34, cin: 192, cout: 192, kh: 3, kw: 3, stride: 1, pad: 0 }, false, DType::Float16, 4e-3; "3x3 unpadded, aligned M")]
#[test_case(ConvGeom { batch: 1, h: 34, w: 34, cin: 64, cout: 192, kh: 3, kw: 3, stride: 1, pad: 0 }, false, DType::Float16, 4e-3; "3x3 unpadded, 64 channels, aligned M")]
#[test_case(geom(32, 192, 192, 3, 1), false, DType::Float16, 4e-3; "3x3, aligned M")]
#[test_case(geom(40, 192, 192, 3, 1), true, DType::Float16, 4e-3; "3x3 stride 1 with residual, f16")]
#[test_case(geom(40, 192, 192, 3, 1), false, DType::BFloat16, 8e-3; "3x3 stride 1, bf16")]
#[test_case(geom(80, 768, 768, 3, 2), false, DType::Float16, 4e-3; "3x3 stride 2")]
#[test_case(geom(40, 384, 192, 1, 1), false, DType::Float16, 4e-3; "1x1")]
#[test_case(geom(20, 768, 768, 3, 1), true, DType::Float16, 4e-3; "ragged M with residual")]
#[test_case(ConvGeom { batch: 2, h: 24, w: 40, cin: 64, cout: 128, kh: 3, kw: 3, stride: 1, pad: 1 }, false, DType::Float16, 4e-3; "batch 2, non-square")]
#[ignore]
fn conv_matches_the_graph_gpu(g: ConvGeom, residual: bool, dtype: DType, tol: f32) {
    if !device_supported(CONV_SUPPORTED_ARCHS) {
        eprintln!("skip conv_matches_the_graph_gpu: no supported device / toolchain");
        return;
    }
    let x = operand(&[g.batch, g.h, g.w, g.cin], dtype.clone(), 0.31);
    let w = operand(&[g.cout, g.kh, g.kw, g.cin], dtype.clone(), 0.17);
    let bias = operand(&[g.cout], dtype.clone(), 0.53);
    let res = residual.then(|| operand(&[g.batch, g.ho(), g.wo(), g.cout], dtype.clone(), 0.71));
    let y = conv2d_nhwc(&x, &w, &bias, res.as_ref(), g.stride, g.pad, true)
        .expect("conv2d build")
        .expect("the kernel applies");

    let want = x
        .try_permute(&[0, 3, 1, 2])
        .unwrap()
        .conv2d()
        .weight(&w.try_permute(&[0, 3, 1, 2]).unwrap())
        .stride(&[g.stride, g.stride])
        .padding(&[(g.pad as isize, g.pad as isize), (g.pad as isize, g.pad as isize)])
        .call()
        .expect("reference conv");
    let want = want.try_add(bias.try_reshape([1, g.cout as isize, 1, 1]).unwrap()).unwrap().silu().unwrap();
    let want = want.try_permute(&[0, 2, 3, 1]).unwrap();
    let want = match &res {
        Some(r) => want.try_add(r).unwrap(),
        None => want,
    };
    let (got, want) = (to_f32_vec(&y), to_f32_vec(&want));
    let err = rel_err(&got, &want);
    // Where the error sits: an interior pixel never touches the padding.
    let (ho, wo, n) = (g.ho(), g.wo(), g.cout);
    let interior = |i: usize| {
        let (p, _) = (i / n % (ho * wo), i % n);
        let (oy, ox) = (p / wo, p % wo);
        oy > 0 && ox > 0 && oy + 1 < ho && ox + 1 < wo
    };
    let pick = |inside: bool| -> (Vec<f32>, Vec<f32>) {
        (0..got.len()).filter(|&i| interior(i) == inside).map(|i| (got[i], want[i])).unzip()
    };
    let ((gi, wi), (gb, wb)) = (pick(true), pick(false));
    println!(
        "conv2d_nhwc {g:?}: relative error {err:e} (interior {:e}, border {:e})",
        rel_err(&gi, &wi),
        rel_err(&gb, &wb)
    );
    assert!(err < tol, "relative error {err} exceeds {tol}");
}
