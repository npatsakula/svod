//! The channels-last convolution: tile applicability, a GPU-free build, and the
//! hardware-gated numerics against the graph's `conv2d`
//! (`SVOD_DEVICE={CUDA,AMD}:0 cargo test -p svod-tk --lib conv -- --ignored --nocapture`).

use svod_dtype::{AmdArch, DType, GpuArch};
use svod_ir::UOp;
use svod_tensor::Tensor;
use test_case::test_case;

use super::device_supported;
use crate::kernels::conv::{
    CONV_SUPPORTED_ARCHS, ConvGeom, ConvPlan, build_conv, conv_candidates, conv_tile_seeds, conv2d_nhwc,
    conv2d_nhwc_worth_asking, declines, select_conv_cfg,
};
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmPolicy};
use crate::kernels::tiling::{Launched, TileBudget, Trial, agreement};

const RDNA4: GpuArch = GpuArch::Amd(AmdArch::Gfx1201);
const SM86: GpuArch = GpuArch::Cuda(svod_dtype::CudaArch::from_compute_capability(8, 6));

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

/// The rule a caller asks before it has a device: the lattice's narrowest edges
/// and the K floor. A 1x1 passes on K alone; whether its layout makes it worth
/// running is the caller's knowledge, not the kernel's.
#[test_case(96, 96, 9, true; "96 channels, K = 864")]
#[test_case(384, 96, 9, true; "the x head's reduction")]
#[test_case(64, 64, 9, true; "K = 576, the floor itself")]
#[test_case(32, 32, 9, false; "K = 288 is under the floor")]
#[test_case(48, 48, 9, false; "48 output channels miss the N edge")]
#[test_case(16, 64, 9, false; "16 input channels: K = 144")]
#[test_case(24, 64, 9, false; "24 input channels never fill a strip")]
#[test_case(1536, 768, 1, true; "a 1x1 passes on K")]
fn worth_asking_follows_the_lattice_and_the_floor(cin: usize, cout: usize, taps: usize, want: bool) {
    assert_eq!(conv2d_nhwc_worth_asking(cin, cout, taps), want);
}

/// On CUDA a shape only a fine tile serves is declined unless its grid starves
/// the device (fewer blocks than the SMs keep resident): the fine tile wins there
/// and loses to the graph's own kernel on a wide grid — whether it is the table's
/// 32x32 or, for a shape the table cannot tile, the lattice's 32-wide edge. A
/// shape a wide tile also serves is never declined, and RDNA never applies the
/// rule: its 128x64 tile is "fine" beside the 128x128 one, and backbone.1 lives
/// on it.
#[test_case(SM86, geom(80, 384, 96, 3, 1), true; "sm86 384-96 at 80: 600 fine blocks")]
#[test_case(SM86, geom(40, 768, 96, 3, 1), true; "sm86 768-96 at 40: 150")]
#[test_case(SM86, geom(20, 768, 96, 3, 1), false; "sm86 768-96 at 20: 39, starved")]
#[test_case(SM86, geom(80, 128, 32, 3, 1), true; "sm86 the s head at 80")]
#[test_case(SM86, geom(20, 512, 32, 3, 1), false; "sm86 the s head at 20")]
#[test_case(SM86, geom(20, 192, 192, 3, 1), false; "sm86 a wide tile also fits")]
#[test_case(SM86, geom(80, 96, 96, 3, 1), true; "sm86 96-96 at 80: the lattice's edge on 150 blocks")]
#[test_case(SM86, geom(20, 96, 96, 3, 1), false; "sm86 96-96 at 20: the lattice's edge on 12, starved")]
#[test_case(SM86, geom(160, 48, 48, 3, 1), false; "sm86 48 channels: nothing tiles it, nothing to decline")]
#[test_case(RDNA4, geom(80, 96, 96, 3, 1), false; "rdna4 never declines: the lattice measures")]
#[test_case(RDNA4, geom(320, 96, 192, 3, 2), false; "rdna4 keeps backbone.1 on its 128x64 tile")]
fn the_fine_tile_is_declined_on_a_wide_grid(arch: GpuArch, g: ConvGeom, declined: bool) {
    let (policy, caps) = (GemmPolicy::for_arch(arch), crate::ArchCaps::for_arch(arch));
    assert_eq!(declines(&policy, &caps, &g), declined);
    let plans = conv_candidates(&policy, &g, &caps);
    assert!(!declined || plans.is_empty(), "a declined shape offers no plan: {plans:?}");
    assert!(plans.iter().all(|p| g.tiles(&p.cfg())));
}

/// The CUDA table's stepped loop and staged store are the linear layers': a
/// convolution runs the whole-strip loop — the only one the tap-wise and patch
/// forms have, so the stepped tile does not tile it while the same tile on that
/// loop does — and the direct store its forms were measured with.
#[test]
fn convolutions_keep_the_whole_strip_loop_and_the_direct_store() {
    let (policy, caps) = (GemmPolicy::for_arch(SM86), crate::ArchCaps::for_arch(SM86));
    let stepped = *policy.tiles.iter().find(|cfg| cfg.stepped).expect("the CUDA table has a stepped tile");
    assert!(policy.tiles.iter().all(|cfg| cfg.stage_out), "the CUDA table stages its stores");
    let g = geom(40, 192, 192, 3, 1);
    assert!(!g.tiles(&stepped));
    assert!(g.tiles(&GemmCfg { stepped: false, stages: 2, ..stepped }));
    let plans = conv_candidates(&policy, &g, &caps);
    assert!(!plans.is_empty() && plans.iter().all(|p| !p.cfg().stepped && !p.cfg().stage_out), "{plans:?}");
    assert!(select_conv_cfg(&policy, &g).is_some_and(|cfg| !cfg.stage_out));
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

/// The graph's `conv2d` over the same channels-last operands (read through
/// permuted views), with the bias, SiLU and residual the kernel's epilogue folds
/// in — the `[batch, ho, wo, cout]` answer both forms are checked against.
fn reference(g: &ConvGeom, x: &Tensor, w: &Tensor, bias: &Tensor, res: Option<&Tensor>) -> Tensor {
    let y = x
        .try_permute(&[0, 3, 1, 2])
        .unwrap()
        .conv2d()
        .weight(&w.try_permute(&[0, 3, 1, 2]).unwrap())
        .stride(&[g.stride, g.stride])
        .padding(&[(g.pad as isize, g.pad as isize), (g.pad as isize, g.pad as isize)])
        .call()
        .expect("reference conv");
    let y = y.try_add(bias.try_reshape([1, g.cout as isize, 1, 1]).unwrap()).unwrap().silu().unwrap();
    let y = y.try_permute(&[0, 2, 3, 1]).unwrap();
    match res {
        Some(r) => y
            .try_add(r.try_reshape([g.batch as isize, g.ho() as isize, g.wo() as isize, g.cout as isize]).unwrap())
            .unwrap(),
        None => y,
    }
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

    let (got, want) = (to_f32_vec(&y), to_f32_vec(&reference(&g, &x, &w, &bias, res.as_ref())));
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

// ── The image-staged plan ───────────────────────────────────────────────────

/// The candidate list: every tile that serves the shape in its gathered form,
/// plus the two rewrites that take the tap out of the K index. Both are for a
/// `k > 1` convolution only — a 1x1 has no tap to take out — and the patch is
/// additionally `ldmatrix`-only (CUDA) and bounded by the strip it replaces,
/// which is what keeps it off the stride-2 shapes.
#[test_case(SM86, geom(40, 192, 192, 3, 1), (true, true); "sm86 3x3")]
#[test_case(SM86, geom(80, 768, 768, 3, 2), (false, true); "sm86 stride 2 stages five times the strip")]
#[test_case(SM86, geom(40, 384, 192, 1, 1), (false, false); "sm86 1x1 has no tap to unroll")]
#[test_case(RDNA4, geom(40, 192, 192, 3, 1), (false, false); "rdna4 has neither ldmatrix nor cp.async")]
fn the_rewrites_are_offered(arch: GpuArch, g: ConvGeom, expected: (bool, bool)) {
    let caps = crate::ArchCaps::for_arch(arch);
    let plans = conv_candidates(&GemmPolicy::for_arch(arch), &g, &caps);
    let any = |f: fn(&ConvPlan) -> bool| plans.iter().any(f);
    let got = (any(|p| matches!(p, ConvPlan::Patch(_))), any(|p| matches!(p, ConvPlan::Tapwise(_))));
    assert_eq!(got, expected, "{plans:?}");
    assert!(any(|p| matches!(p, ConvPlan::Gathered(_))), "the gathered form always stands");
    for plan in &plans {
        let grid = plan.grid_dims(&g);
        let cfg = plan.cfg();
        assert_eq!(grid[0] as usize, g.cout / cfg.block_n);
        // The M grid covers every output pixel, whichever way it is cut.
        assert!(grid[1] as usize * cfg.block_m >= g.mkn().0, "{plan:?} leaves output rows uncovered");
    }
}

/// Every plan the tuner may pick, built off the GPU: the patch fill, the
/// gathered `ldmatrix` views of it and the scattered store all lower. Run per
/// arch, because `tuned_conv_plan` builds every candidate it is offered just to
/// fingerprint it — a plan offered on an arch whose body it cannot lower takes
/// the process down on the *first* convolution, before anything is measured.
#[test_case(SM86, geom(40, 192, 192, 3, 1), true; "sm86 3x3 with residual")]
#[test_case(SM86, geom(20, 192, 192, 3, 1), false; "sm86 windows overhang a 20x20 image")]
#[test_case(SM86, geom(80, 768, 768, 3, 2), false; "sm86 3x3 stride 2")]
#[test_case(RDNA4, geom(40, 192, 192, 3, 1), true; "rdna4 3x3 with residual")]
#[test_case(RDNA4, geom(20, 192, 192, 3, 1), false; "rdna4 windows overhang a 20x20 image")]
#[test_case(RDNA4, geom(80, 768, 768, 3, 2), false; "rdna4 3x3 stride 2")]
fn every_plan_builds(arch: GpuArch, g: ConvGeom, residual: bool) {
    let caps = crate::ArchCaps::for_arch(arch);
    let plans = conv_candidates(&GemmPolicy::for_arch(arch), &g, &caps);
    let (m, k, n) = g.mkn();
    let dt = DType::Float16;
    for plan in plans {
        let mut sizes = vec![m * n, g.batch * g.h * g.w * g.cin, n * k, n];
        if residual {
            sizes.push(m * n);
        }
        let bufs = sizes.into_iter().map(|s| UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, s, dt.clone())).collect();
        let cfg = plan.cfg();
        let ker = crate::Kernel::new("conv2d_nhwc", plan.grid_dims(&g), cfg.threads(caps.wave_size), bufs, caps);
        plan.build(&ker, g, dt.clone(), Epilogue::BiasAct { bias: (), residual: residual.then_some(()), act: true });
        assert!(crate::kernel_fingerprint(&ker.finish(cfg.acc_m)).digest != 0, "{plan:?}");
    }
}

/// Every plan run on the device against the graph's `conv2d`. The numerics test
/// above only ever sees the static choice (tuning is off under test), so this is
/// what covers the tiles the chooser passes over — and, on CUDA, the
/// image-staged kernel.
#[test_case(geom(40, 192, 192, 3, 1), false; "3x3 stride 1")]
#[test_case(geom(40, 192, 192, 3, 1), true; "3x3 stride 1 with residual")]
#[test_case(geom(20, 192, 192, 3, 1), false; "windows overhang a 20x20 image")]
#[test_case(ConvGeom { batch: 2, h: 24, w: 40, cin: 64, cout: 128, kh: 3, kw: 3, stride: 1, pad: 1 }, false; "batch 2, non-square")]
#[ignore]
fn every_plan_matches_the_graph_gpu(g: ConvGeom, residual: bool) {
    if !device_supported(CONV_SUPPORTED_ARCHS) {
        eprintln!("skip every_plan_matches_the_graph_gpu: no supported device / toolchain");
        return;
    }
    let spec = Tensor::empty(&[1], DType::Float32).device();
    let arch = crate::target::resolve_supported_arch(&spec, CONV_SUPPORTED_ARCHS).expect("a supported arch");
    let caps = crate::ArchCaps::for_arch(arch);
    let dt = DType::Float16;
    let x = operand(&[g.batch, g.h, g.w, g.cin], dt.clone(), 0.31);
    let w = operand(&[g.cout, g.kh, g.kw, g.cin], dt.clone(), 0.17);
    let bias = operand(&[g.cout], dt.clone(), 0.53);
    let res = residual.then(|| operand(&[g.batch, g.ho(), g.wo(), g.cout], dt.clone(), 0.71));
    let want = to_f32_vec(&reference(&g, &x, &w, &bias, res.as_ref()));

    let (m, n) = (g.mkn().0, g.mkn().2);
    let plans = conv_candidates(&GemmPolicy::for_device(&spec, arch), &g, &caps);
    // The image-staged plan is the reason this test exists, but it is offered
    // only where `ldmatrix` and `cp.async` are. On an arch without them the
    // cover is what remains: the tiles the static chooser passes over.
    assert!(
        caps.cuda().is_none() || plans.iter().any(|p| matches!(p, ConvPlan::Patch(_))),
        "no image-staged plan to cover: {plans:?}"
    );
    for plan in plans {
        let epi = Epilogue::BiasAct { bias: (), residual: residual.then_some(()), act: true };
        let cfg = plan.cfg();
        let mut y = Tensor::empty(&[m, n], dt.clone()).to(spec.clone());
        let mut ins: Vec<&Tensor> = vec![&x, &w, &bias];
        ins.extend(res.as_ref());
        let (grid, block) = (plan.grid_dims(&g), cfg.threads(caps.wave_size));
        let (geom, dtc) = (g, dt.clone());
        crate::launch::run_kernel("conv2d_nhwc_test", grid, block, &mut [&mut y], &ins, move |ker| {
            plan.build(ker, geom, dtc, epi);
            ker.finish(cfg.acc_m)
        })
        .expect("run the plan");
        let err = rel_err(&to_f32_vec(&y), &want);
        println!("conv2d_nhwc {plan:?}: relative error {err:e}");
        assert!(err < 4e-3, "{plan:?}: relative error {err} exceeds 4e-3");
    }
}

/// Every tile the lattice walk can reach for `g` computes what the graph does.
/// The walk ranks by time alone ([`TileBudget::search`]), so a tile that
/// compiles, runs fast and computes the wrong thing wins the search unopposed —
/// which is how YOLO26-m's f16 parity broke on gfx1201 (16 px of box drift on
/// both arms of an A/B) while x's held. This is the check the search does not
/// make: the set reachable from the model's own seeds by one-step doublings and
/// halvings, each tile run on the device against the graph's answer. A tile
/// the device refuses to run is reported and not counted — the search skips it
/// the same way.
///
/// ```text
/// SVOD_DEVICE=AMD:0 cargo test --release -p svod-tk --lib every_lattice_tile -- --ignored --nocapture
/// ```
#[test_case(geom(80, 64, 64, 3, 1); "m bodies 64-64 at 80")]
#[test_case(geom(40, 128, 128, 3, 1); "m bodies 128-128 at 40")]
#[test_case(geom(20, 128, 128, 3, 1); "m backbone.8 bodies at 20")]
#[test_case(geom(80, 256, 64, 3, 1); "m head at 80")]
#[test_case(geom(40, 512, 64, 3, 1); "m head at 40")]
#[test_case(geom(20, 512, 64, 3, 1); "m head at 20")]
#[test_case(geom(20, 512, 256, 3, 1); "m neck.22 attn cv1")]
#[test_case(geom(20, 256, 512, 3, 1); "m neck.22 attn cv2")]
#[test_case(geom(320, 64, 128, 3, 2); "m backbone.1")]
#[test_case(geom(160, 256, 256, 3, 2); "m backbone.3")]
#[test_case(geom(80, 256, 256, 3, 2); "m neck.17")]
#[test_case(geom(80, 512, 512, 3, 2); "m backbone.5")]
#[test_case(geom(40, 512, 512, 3, 2); "m backbone.7 and neck.20")]
#[test_case(geom(160, 64, 64, 3, 2); "n backbone.3")]
#[test_case(geom(80, 128, 128, 3, 2); "n backbone.5")]
#[test_case(geom(40, 128, 256, 3, 2); "n backbone.7")]
#[test_case(geom(80, 64, 64, 3, 2); "n neck.17")]
#[test_case(geom(40, 128, 128, 3, 2); "n neck.20")]
#[test_case(geom(20, 128, 64, 3, 1); "n neck.22 attn cv1")]
#[test_case(geom(20, 64, 128, 3, 1); "n neck.22 attn cv2")]
#[ignore]
fn every_lattice_tile_matches_the_graph_gpu(g: ConvGeom) {
    if !device_supported(CONV_SUPPORTED_ARCHS) {
        eprintln!("skip every_lattice_tile_matches_the_graph_gpu: no supported device / toolchain");
        return;
    }
    let spec = Tensor::empty(&[1], DType::Float32).device();
    let arch = crate::target::resolve_supported_arch(&spec, CONV_SUPPORTED_ARCHS).expect("a supported arch");
    let caps = crate::ArchCaps::for_arch(arch);
    let Some(budget) = TileBudget::for_device(&spec, arch) else {
        eprintln!("skip every_lattice_tile_matches_the_graph_gpu: the device reports no limits to build a lattice on");
        return;
    };
    let policy = GemmPolicy::for_device(&spec, arch);
    let dt = DType::Float16;
    let x = operand(&[g.batch, g.h, g.w, g.cin], dt.clone(), 0.31);
    let w = operand(&[g.cout, g.kh, g.kw, g.cin], dt.clone(), 0.17);
    let bias = operand(&[g.cout], dt.clone(), 0.53);
    let want = to_f32_vec(&reference(&g, &x, &w, &bias, None));
    let (m, n) = (g.mkn().0, g.mkn().2);

    let mut tiles: Vec<GemmCfg> = conv_tile_seeds(&budget, &policy, &dt, &g);
    let mut next = 0;
    while next < tiles.len() {
        for cfg in budget.neighbours(&tiles[next], dt.bytes(), |cfg| g.tiles(cfg)) {
            if !tiles.contains(&cfg) {
                tiles.push(cfg);
            }
        }
        next += 1;
    }

    let mut wrong = Vec::new();
    for &cfg in &tiles {
        let plan = ConvPlan::Gathered(cfg);
        let epi = Epilogue::BiasAct { bias: (), residual: None, act: true };
        let mut y = Tensor::empty(&[m, n], dt.clone()).to(spec.clone());
        let ins: Vec<&Tensor> = vec![&x, &w, &bias];
        let (grid, block) = (plan.grid_dims(&g), cfg.threads(caps.wave_size));
        let (geom, dtc) = (g, dt.clone());
        let ran = crate::launch::run_kernel("conv2d_nhwc_lattice", grid, block, &mut [&mut y], &ins, move |ker| {
            plan.build(ker, geom, dtc, epi);
            ker.finish(cfg.acc_m)
        });
        match ran {
            Ok(_) => {
                let err = rel_err(&to_f32_vec(&y), &want);
                println!("{} {cfg:?}: relative error {err:e}", if err < 4e-3 { "ok " } else { "BAD" });
                if err >= 4e-3 {
                    wrong.push((cfg, err));
                }
            }
            Err(err) => println!("--- {cfg:?}: did not run: {err}"),
        }
    }
    println!("{} tiles reachable for {g:?}, {} wrong", tiles.len(), wrong.len());
    assert!(wrong.is_empty(), "tiles that compute the wrong thing: {wrong:?}");
}

/// The lattice search on the device, with one seed made fast and wrong: it runs
/// its tile as a 1x1 over the same pixels — a ninth of the work, the same output
/// shape, another answer. The real tiles read back one answer, the odd one is
/// dropped however fast it ran, and the winner computes the graph's `conv2d`.
#[test_case(geom(80, 64, 64, 3, 1); "m bodies")]
#[test_case(geom(40, 128, 128, 3, 2); "n neck.20, stride 2")]
#[ignore]
fn the_lattice_search_drops_a_fast_wrong_tile_gpu(g: ConvGeom) {
    if !device_supported(CONV_SUPPORTED_ARCHS) {
        eprintln!("skip the_lattice_search_drops_a_fast_wrong_tile_gpu: no supported device / toolchain");
        return;
    }
    let spec = Tensor::empty(&[1], DType::Float32).device();
    let arch = crate::target::resolve_supported_arch(&spec, CONV_SUPPORTED_ARCHS).expect("a supported arch");
    let caps = crate::ArchCaps::for_arch(arch);
    let Some(budget) = TileBudget::for_device(&spec, arch) else {
        eprintln!("skip the_lattice_search_drops_a_fast_wrong_tile_gpu: the device reports no limits");
        return;
    };
    let dt = DType::Float16;
    let x = operand(&[g.batch, g.h, g.w, g.cin], dt.clone(), 0.31);
    let w = operand(&[g.cout, g.kh, g.kw, g.cin], dt.clone(), 0.17);
    let bias = operand(&[g.cout], dt.clone(), 0.53);
    let pointwise = ConvGeom { h: g.ho(), w: g.wo(), kh: 1, kw: 1, stride: 1, pad: 0, ..g };
    let w1 = operand(&[g.cout, 1, 1, g.cin], dt.clone(), 0.29);
    let seeds = conv_tile_seeds(&budget, &GemmPolicy::for_device(&spec, arch), &dt, &g);
    let odd = *seeds.iter().find(|cfg| pointwise.tiles(cfg)).expect("a seed that also tiles the 1x1");
    let x1 = operand(&[g.batch, g.ho(), g.wo(), g.cin], dt.clone(), 0.37);

    let (m, n) = (g.mkn().0, g.mkn().2);
    let epi = Epilogue::BiasAct { bias: (), residual: None, act: true };
    let mut compile = |cfg: GemmCfg| {
        let (geom, x, w) = if cfg == odd { (pointwise, &x1, &w1) } else { (g, &x, &w) };
        let mut y = Tensor::empty(&[m, n], dt.clone()).to(spec.clone());
        let plan = ConvPlan::Gathered(cfg);
        let (grid, block) = (plan.grid_dims(&geom), cfg.threads(caps.wave_size));
        let dtc = dt.clone();
        let launch = crate::launch::compile_kernel("conv2d_nhwc_search", grid, block, &mut [&mut y], &[x, w, &bias], {
            move |ker| {
                plan.build(ker, geom, dtc, epi);
                ker.finish(cfg.acc_m)
            }
        })
        .ok()?;
        Some(Launched { launch, output: y })
    };
    let (won, _) = budget.search(&seeds, dt.bytes(), agreement(&dt), |cfg| g.tiles(cfg), &mut compile).expect("a tile");
    assert_ne!(won, odd, "the fast wrong tile won");
    // Its answer is all that kept it out: timed against the winner, it is faster.
    let best = |trial: &Launched| (0..20).filter_map(|_| trial.time()).min().expect("a device time");
    let (odd_trial, won_trial) = (compile(odd).expect("the odd tile"), compile(won).expect("the winner"));
    let (odd_time, won_time) = (best(&odd_trial), best(&won_trial));
    assert!(odd_time < won_time, "the odd tile ({odd_time:?}) was not faster than the winner ({won_time:?})");

    let got = won_trial.output().expect("the winner's output");
    let err = rel_err(&got, &to_f32_vec(&reference(&g, &x, &w, &bias, None)));
    assert!(err < 4e-3, "the winner {won:?} is off the graph by {err:e}");
}
