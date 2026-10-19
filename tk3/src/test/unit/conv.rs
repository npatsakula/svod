//! The implicit-GEMM convolution: the interpreter against an f64 direct
//! convolution, the emitted program's row decode and fills on sm_86, and
//! the config table's invariants.

use std::sync::Arc;

use proptest::prelude::*;
use svod_dtype::{DType, DeviceSpec, ScalarDType};
use svod_ir::{BinaryOp as UBinary, Op, UOp, ops};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::atoms::sm86;
use crate::build::F16;
use crate::interp::{round_to, run};
use crate::ir::Program;
use crate::kernels::conv::{ConvCfg, ConvGeom, ConvSpec, conv};
use crate::kernels::gemm::{Epilogue, GemmCfg};
use crate::kernels::{Act, Batch};
use crate::launch::graph_launch;
use crate::lower::lower;
use crate::ops::config::{cfg_fits, conv_candidates};

pub(super) fn geom(hw: [usize; 2], cin: usize, cout: usize, k: usize, s: usize, p: usize, d: usize) -> ConvGeom {
    ConvGeom { h: hw[0], w: hw[1], cin, cout, kernel: [k, k], stride: [s, s], pad: [p, p], dilation: [d, d] }
}

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
}

pub(super) fn halves(n: usize, seed: &mut u64, scale: f64) -> Vec<f64> {
    (0..n).map(|_| round_to(ScalarDType::Float16, scale * lcg(seed))).collect()
}

/// The parameters of `conv` in order, f16-valued: `x`, `w`, `bias`, `residual`, `y`.
pub(super) fn inputs(g: &ConvGeom, images: usize, epi: Epilogue, seed: u64) -> Vec<Vec<f64>> {
    let [ho, wo] = g.out_hw();
    let mut seed = seed;
    let mut p = vec![halves(images * g.h * g.w * g.cin, &mut seed, 1.0)];
    p.push(halves(g.cout * g.k(), &mut seed, 1.0 / (g.k() as f64).sqrt()));
    if epi.bias {
        p.push(halves(g.cout, &mut seed, 1.0));
    }
    if epi.residual {
        p.push(halves(images * ho * wo * g.cout, &mut seed, 2.0));
    }
    p.push(vec![0.0; images * ho * wo * g.cout]);
    p
}

/// Direct convolution in f64 with the epilogue, rounded once to the output type.
pub(super) fn reference(g: &ConvGeom, images: usize, epi: Epilogue, p: &[Vec<f64>]) -> Vec<f64> {
    let [ho, wo] = g.out_hw();
    let (x, w) = (&p[0], &p[1]);
    let bias = epi.bias.then(|| &p[2]);
    let residual = epi.residual.then(|| &p[2 + usize::from(epi.bias)]);
    let out = if epi.out_f32 { ScalarDType::Float32 } else { ScalarDType::Float16 };
    let mut y = vec![0.0; images * ho * wo * g.cout];
    for b in 0..images {
        for oy in 0..ho {
            for ox in 0..wo {
                for co in 0..g.cout {
                    let mut acc = bias.map_or(0.0, |v| v[co]);
                    for ky in 0..g.kernel[0] {
                        for kx in 0..g.kernel[1] {
                            let iy = (oy * g.stride[0] + ky * g.dilation[0]) as i64 - g.pad[0] as i64;
                            let ix = (ox * g.stride[1] + kx * g.dilation[1]) as i64 - g.pad[1] as i64;
                            if iy < 0 || ix < 0 || iy >= g.h as i64 || ix >= g.w as i64 {
                                continue;
                            }
                            let xi = ((b * g.h + iy as usize) * g.w + ix as usize) * g.cin;
                            let wi = ((co * g.kernel[0] + ky) * g.kernel[1] + kx) * g.cin;
                            acc += (0..g.cin).map(|c| x[xi + c] * w[wi + c]).sum::<f64>();
                        }
                    }
                    let v = match epi.act {
                        Act::None => acc,
                        Act::Silu => acc / (1.0 + (-acc).exp()),
                        Act::Gelu => 0.5 * acc * (1.0 + libm::erf(acc * std::f64::consts::FRAC_1_SQRT_2)),
                    };
                    let v = v * epi.scale.map_or(1.0, |s| f64::from(s.get()));
                    let o = ((b * ho + oy) * wo + ox) * g.cout + co;
                    y[o] = round_to(out, v + residual.map_or(0.0, |r| r[o]));
                }
            }
        }
    }
    y
}

/// `|got - want| ≤ tol·max(|want|, 1)`; returns the max abs diff.
pub(super) fn assert_close(what: &str, got: &[f64], want: &[f64], tol: f64) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let diff = (g - w).abs();
        worst = worst.max(diff);
        assert!(diff <= tol * w.abs().max(1.0), "{what}[{i}] = {g}, want {w}");
    }
    worst
}

/// One rounding to f16 (half an ulp, 2⁻¹¹) plus f32 accumulation; f32 out
/// keeps only the latter.
pub(super) fn tolerance(epi: Epilogue) -> f64 {
    if epi.out_f32 { 1e-4 } else { 2e-3 }
}

fn cfg(tile: [usize; 3], warps: [u32; 2], stages: usize) -> GemmCfg {
    GemmCfg { tile, stages, warps, group_m: 8, unroll: true }
}

fn check_interp(g: ConvGeom, batch: Batch, epi: Epilogue, c: GemmCfg) {
    let images = batch.capacity();
    let prog = conv::<F16>(&ConvSpec { batch: batch.clone(), geom: g, epilogue: epi, cfg: c, split: 1 });
    let params = inputs(&g, images, epi, 3);
    let want = reference(&g, images, epi, &params);
    let vars: Vec<(&str, i64)> = match &batch {
        Batch::Var { name, max, .. } => vec![(name.as_str(), *max)],
        Batch::Static(_) => vec![],
    };
    let got = run(&prog, params, &vars).unwrap();
    assert_close("y", got.last().unwrap(), &want, tolerance(epi));
}

const SILU_BIAS: Epilogue = Epilogue { bias: true, act: Act::Silu, ..Epilogue::DEFAULT };

#[test_case(geom([5, 7], 16, 24, 3, 1, 1, 1), 2, SILU_BIAS, [32, 32, 16]; "3x3 same, ragged m and n")]
#[test_case(geom([9, 8], 32, 48, 3, 2, 1, 1), 1, SILU_BIAS, [64, 48, 32]; "3x3 stride 2, 48-wide")]
#[test_case(geom([2, 2], 16, 96, 5, 1, 2, 1), 3, SILU_BIAS, [32, 96, 16]; "smaller than the kernel")]
#[test_case(geom([6, 6], 48, 16, 3, 1, 2, 2), 1, SILU_BIAS, [32, 32, 16]; "dilated")]
#[test_case(geom([3, 9], 16, 8, 3, 3, 0, 1), 2, SILU_BIAS, [32, 32, 16]; "one output row")]
#[test_case(geom([4, 4], 32, 32, 1, 1, 0, 1), 2, Epilogue { residual: true, ..SILU_BIAS }, [32, 32, 32]; "1x1 residual")]
#[test_case(geom([5, 5], 16, 40, 3, 1, 1, 1), 1, Epilogue { out_f32: true, residual: true, ..Epilogue::DEFAULT }, [32, 32, 16]; "f32 out and residual")]
fn conv_interpreter_matches_the_direct_convolution(g: ConvGeom, images: usize, epi: Epilogue, tile: [usize; 3]) {
    check_interp(g, Batch::Static(images), epi, cfg(tile, [2, 2], 2));
}

/// A bound batch walks grid z instead of folding into the rows.
#[test]
fn a_bound_batch_walks_grid_z() {
    let batch = Batch::Var { name: "b".into(), min: 1, max: 3 };
    check_interp(geom([6, 5], 16, 32, 3, 2, 1, 1), batch, SILU_BIAS, cfg([32, 32, 16], [2, 2], 3));
}

/// A reduction split over grid z: the partial program writes one f32
/// accumulator per split, the merge adds them under the whole epilogue.
#[test_case(3, Epilogue { residual: true, ..SILU_BIAS }; "three splits, residual")]
#[test_case(9, Epilogue { out_f32: true, ..SILU_BIAS }; "a split per tap, f32 out")]
fn split_conv_matches_the_direct_convolution(split: usize, epi: Epilogue) {
    let (g, images) = (geom([7, 6], 32, 40, 3, 1, 1, 1), 2);
    let spec =
        ConvSpec { batch: Batch::Static(images), geom: g, epilogue: epi, cfg: cfg([32, 32, 32], [2, 2], 2), split };
    let params = inputs(&g, images, epi, 6);
    let want = reference(&g, images, epi, &params);
    let [ho, wo] = g.out_hw();
    let progs = spec.programs::<F16>(&sm86());
    let partial = vec![0.0; split * images * ho * wo * g.cout];
    let parts = run(&progs[0].0, vec![params[0].clone(), params[1].clone(), partial], &[]).unwrap();
    let mut merge_in = vec![parts[2].clone()];
    merge_in.extend(params[2..].iter().cloned());
    let got = run(&progs[1].0, merge_in, &[]).unwrap();
    assert_close("y", got.last().unwrap(), &want, tolerance(epi));
}

/// The padding is data: an Inf in the corner pixel reaches exactly the
/// outputs whose window covers it, and no padded tap turns it into NaN.
#[test]
fn an_inf_reaches_only_the_windows_over_it() {
    let g = geom([6, 6], 16, 8, 3, 1, 1, 1);
    let epi = Epilogue::DEFAULT;
    let prog = conv::<F16>(&ConvSpec {
        batch: Batch::Static(1),
        geom: g,
        epilogue: epi,
        cfg: cfg([32, 32, 16], [2, 2], 2),
        split: 1,
    });
    let mut params = inputs(&g, 1, epi, 5);
    params[0][0] = f64::INFINITY;
    let y = run(&prog, params, &[]).unwrap().pop().unwrap();
    for (o, v) in y.iter().enumerate() {
        let (oy, ox) = (o / g.cout / 6, o / g.cout % 6);
        assert_eq!(!v.is_finite(), oy <= 1 && ox <= 1, "y[{oy}, {ox}] = {v}");
        assert!(oy <= 1 && ox <= 1 || !v.is_nan(), "padding turned the Inf into NaN at [{oy}, {ox}]");
    }
}

fn epilogues() -> impl Strategy<Value = Epilogue> {
    (any::<bool>(), any::<bool>(), prop_oneof![Just(Act::None), Just(Act::Silu)], any::<bool>())
        .prop_map(|(bias, residual, act, out_f32)| Epilogue { bias, residual, act, out_f32, ..Epilogue::DEFAULT })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Any small geometry, epilogue and tile: the interpreter computes the
    /// direct convolution.
    #[test]
    fn conv_matches_the_direct_convolution_for_any_geometry(
        images in 1usize..=3,
        hw in [1usize..=12, 1usize..=12],
        cin in prop_oneof![Just(16usize), Just(32), Just(48)],
        cout in (1usize..=12).prop_map(|c| 8 * c),
        k in prop_oneof![Just(1usize), Just(3), Just(5)],
        s in 1usize..=2,
        padded in any::<bool>(),
        epi in epilogues(),
        tile in prop_oneof![Just([32usize, 32, 16]), Just([64, 96, 16]), Just([32, 48, 16])],
    ) {
        let g = geom(hw, cin, cout, k, s, if padded { k / 2 } else { 0 }, 1);
        prop_assume!(g.out_hw().iter().all(|&d| d > 0));
        check_interp(g, Batch::Static(images), epi, cfg(tile, [2, 2], 2));
    }
}

// ---- the emitted program ---------------------------------------------------------

fn emitted(prog: Program, c: GemmCfg) -> Vec<Arc<UOp>> {
    let params =
        prog.params.iter().enumerate().map(|(i, p)| UOp::param(i, p.elems, DType::Scalar(p.dtype), None)).collect();
    let lowered = lower(prog, &c.lowering(sm86()), params, DeviceSpec::Cuda { device_id: 0 }).unwrap();
    let Op::Program(ops::Program { linear: Some(linear), .. }) = lowered.program.op() else { panic!("a program") };
    let Op::Linear(ops::Linear { ops }) = linear.op() else { panic!("a linear list") };
    ops.to_vec()
}

fn per_thread(u: &Arc<UOp>) -> bool {
    u.toposort().iter().any(|s| matches!(s.op(), Op::Special(ops::Special { name, .. }) if name == "lidx0"))
}

fn divisions(ops: &[Arc<UOp>]) -> usize {
    ops.iter()
        .filter(|u| {
            matches!(u.op(), Op::Binary(UBinary::CDiv | UBinary::CMod | UBinary::FloorDiv | UBinary::FloorMod, ..))
        })
        .filter(|u| per_thread(u))
        .count()
}

/// 192→192 3×3 at 40×40 on 64×64×32: the `(b, oy, ox)` decode of every
/// A chunk sits before the loop; inside it, each A chunk is one zero-filling
/// copy and no per-thread division is left.
#[test]
fn the_conv_row_decode_sits_in_the_prologue() {
    let c = cfg([64, 64, 32], [2, 2], 3);
    let g = geom([40, 40], 192, 192, 3, 1, 1, 1);
    let list =
        emitted(conv::<F16>(&ConvSpec { batch: Batch::Static(1), geom: g, epilogue: SILU_BIAS, cfg: c, split: 1 }), c);
    let start = list.iter().position(|u| matches!(u.op(), Op::Range(..))).unwrap();
    let end = start + list[start..].iter().position(|u| matches!(u.op(), Op::End(..))).unwrap();
    let body = &list[start..end];
    assert!(divisions(&list[..start]) > 0, "the decode is in the prologue");
    assert_eq!(divisions(body), 0, "no per-thread division in the loop");
    let zfill = body
        .iter()
        .filter(|u| matches!(u.op(), Op::Custom(ops::Custom { code, .. }) if code.contains("global.16.s(ptr")))
        .count();
    assert_eq!(zfill, 64 * 32 / 8 / 128 * 3, "one zero-filling copy per A chunk per unrolled step");
}

// ---- configs ---------------------------------------------------------------------

/// `(cin, cout, k, [h, w], stride)` of the YOLO26x dense 3×3 classes at
/// 640² (the memo's census), padded `k / 2`.
pub(super) const YOLO: [(usize, usize, usize, [usize; 2], usize); 9] = [
    (192, 192, 3, [40, 40], 1),
    (96, 96, 3, [80, 80], 1),
    (384, 384, 3, [160, 160], 2),
    (768, 768, 3, [80, 80], 2),
    (96, 192, 3, [320, 320], 2),
    (48, 48, 3, [160, 160], 1),
    (768, 768, 3, [40, 40], 2),
    (384, 96, 3, [80, 80], 1),
    (768, 96, 3, [20, 20], 1),
];

pub(super) fn yolo_geom((cin, cout, k, hw, s): (usize, usize, usize, [usize; 2], usize)) -> ConvGeom {
    geom(hw, cin, cout, k, s, k / 2, 1)
}

/// Every candidate of every YOLO class lowers on sm_86: its tiles split
/// into whole chunks per thread, `bk` divides `cin`, the ring fits.
#[test]
fn conv_candidates_lower() {
    let target = sm86();
    for class in YOLO {
        let g = yolo_geom(class);
        let [ho, wo] = g.out_hw();
        let cands = conv_candidates(&target, 1, ho * wo, &g);
        assert!(!cands.is_empty() || g.cin == 48, "{g:?}");
        for ConvCfg { gemm: c, split } in cands {
            assert!(
                cfg_fits(&target, &c) && g.cin.is_multiple_of(c.tile[2]) && c.smem_bytes(false) <= target.smem_bytes
            );
            assert!((g.k() / c.tile[2]).is_multiple_of(split));
        }
    }
}

#[test_case(96, 96, 80, 1, Some([64, 96, 32]); "96 channels get a 96-wide tile")]
#[test_case(192, 192, 40, 1, Some([64, 64, 32]); "192 at 40x40 fills the SMs with 64x64")]
#[test_case(384, 384, 160, 2, Some([128, 128, 32]); "a large grid takes the largest tile")]
#[test_case(768, 768, 40, 2, Some([32, 96, 32]); "400 rows pad 64-row tiles by a ninth")]
#[test_case(96, 48, 80, 1, Some([64, 48, 32]); "48 channels take a 48-wide tile")]
#[test_case(48, 48, 160, 1, None; "48 channels in no 48-wide fill")]
fn conv_lead_tiles(cin: usize, cout: usize, hw: usize, s: usize, tile: Option<[usize; 3]>) {
    let g = geom([hw, hw], cin, cout, 3, s, 1, 1);
    let [ho, wo] = g.out_hw();
    let lead = conv_candidates(&sm86(), 1, ho * wo, &g).first().map(|c| c.gemm.tile);
    assert_eq!(lead, tile);
}

/// Every YOLO26x 3×3 class but the 48-channel bodies has candidates; a
/// starved grid has split ones.
#[test]
fn only_the_48_channel_bodies_have_no_candidates() {
    let none: Vec<(usize, usize)> = YOLO
        .into_iter()
        .map(yolo_geom)
        .filter(|g| {
            let [ho, wo] = g.out_hw();
            conv_candidates(&sm86(), 1, ho * wo, g).is_empty()
        })
        .map(|g| (g.cin, g.cout))
        .collect();
    assert_eq!(none, [(48, 48)]);
    let starved = geom([20, 20], 192, 192, 3, 1, 1, 1);
    assert!(conv_candidates(&sm86(), 1, 400, &starved).iter().any(|c| c.split > 1));
}

#[test_case(3; "rgb stem")]
#[test_case(8; "eight channels")]
#[test_case(40; "a multiple of 8 only")]
fn too_few_channels_have_no_candidates(cin: usize) {
    assert!(conv_candidates(&sm86(), 1, 6400, &geom([80, 80], cin, 64, 3, 1, 1, 1)).is_empty());
}

// ---- on the device -----------------------------------------------------------------

fn cuda_target() -> Option<crate::atoms::Target> {
    let spec = svod_dtype::default_device::default_device();
    let target = matches!(spec, DeviceSpec::Cuda { .. }).then(|| crate::atoms::Target::for_device(&spec)).flatten();
    if target.is_none() {
        eprintln!("skipped: no CUDA device");
    }
    target
}

fn upload(v: &[f64], dtype: DType) -> Tensor {
    let t = Tensor::from_slice(v.iter().map(|&x| x as f32).collect::<Vec<_>>()).cast(dtype).contiguous();
    t.realize().unwrap();
    t
}

/// The convolution as the graph computes it in f32 from the same f16
/// values: NCHW views, one cast at the end.
fn graph_reference(g: &ConvGeom, images: usize, epi: Epilogue, p: &[Vec<f64>]) -> Vec<f32> {
    let [ho, wo] = g.out_hw();
    let f32_of = |v: &[f64], shape: Vec<isize>| upload(v, DType::Float32).try_reshape(shape).unwrap();
    let x = f32_of(&p[0], vec![images as isize, g.h as isize, g.w as isize, g.cin as isize]);
    let w = f32_of(&p[1], vec![g.cout as isize, g.kernel[0] as isize, g.kernel[1] as isize, g.cin as isize]);
    let bias = epi.bias.then(|| f32_of(&p[2], vec![g.cout as isize]));
    let padding = [(g.pad[0] as isize, g.pad[0] as isize), (g.pad[1] as isize, g.pad[1] as isize)];
    let y = x
        .try_permute(&[0, 3, 1, 2])
        .unwrap()
        .conv2d()
        .weight(&w.try_permute(&[0, 3, 1, 2]).unwrap())
        .maybe_bias(bias.as_ref())
        .stride(&g.stride)
        .dilation(&g.dilation)
        .padding(&padding)
        .call()
        .unwrap();
    let y = match epi.act {
        Act::Silu => y.silu().unwrap(),
        _ => y,
    };
    let mut y = y.try_permute(&[0, 2, 3, 1]).unwrap();
    if epi.residual {
        let shape = vec![images as isize, ho as isize, wo as isize, g.cout as isize];
        y = y.try_add(f32_of(&p[2 + usize::from(epi.bias)], shape)).unwrap();
    }
    let out = if epi.out_f32 { DType::Float32 } else { DType::Float16 };
    y.cast(out).cast(DType::Float32).contiguous().to_vec::<f32>().unwrap()
}

/// The kernel's output for `params` on the device, as f32.
fn on_device(spec: &ConvSpec, target: &crate::atoms::Target, params: &[Vec<f64>]) -> Vec<f32> {
    let out = if spec.epilogue.out_f32 { DType::Float32 } else { DType::Float16 };
    let n = params.len();
    let tensors: Vec<Tensor> = params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let output_typed = i == n - 1 || (spec.epilogue.residual && i == n - 2);
            upload(p, if output_typed { out.clone() } else { DType::Float16 })
        })
        .collect();
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let mut programs = spec.programs::<F16>(target).into_iter();
    let (prog, lowering) = programs.next().unwrap();
    let y = match programs.next() {
        None => graph_launch(prog, &lowering, &refs).unwrap(),
        Some((merge, merge_lowering)) => {
            let [ho, wo] = spec.geom.out_hw();
            let partial =
                Tensor::empty(&[spec.split * spec.batch.capacity() * ho * wo * spec.geom.cout], DType::Float32);
            let partial = graph_launch(prog, &lowering, &[refs[0], refs[1], &partial]).unwrap();
            let rest: Vec<&Tensor> = std::iter::once(&partial).chain(refs[2..].iter().copied()).collect();
            graph_launch(merge, &merge_lowering, &rest).unwrap()
        }
    };
    y.cast(DType::Float32).to_vec::<f32>().unwrap()
}

fn assert_close_f32(what: &str, got: &[f32], want: &[f32], tol: f32) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}: element count");
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let diff = (g - w).abs();
        worst = worst.max(diff);
        assert!(diff <= tol * w.abs().max(1.0), "{what}[{i}] = {g}, graph {w}");
    }
    worst
}

const RESIDUAL_SILU_BIAS: Epilogue = Epilogue { residual: true, ..SILU_BIAS };

/// Every candidate of every YOLO26x dense 3×3 class matches the graph's
/// convolution on the device (the tuner times on scratch buffers; this
/// sweep is what verifies what it may pick).
#[test]
fn every_conv_candidate_matches_the_graph_on_yolo_shapes() {
    let Some(target) = cuda_target() else { return };
    for class in YOLO {
        let g = yolo_geom(class);
        let [ho, wo] = g.out_hw();
        let params = inputs(&g, 1, RESIDUAL_SILU_BIAS, 13);
        let want = graph_reference(&g, 1, RESIDUAL_SILU_BIAS, &params);
        for c in conv_candidates(&target, 1, ho * wo, &g) {
            let (epilogue, split) = (RESIDUAL_SILU_BIAS, c.split);
            let spec = ConvSpec { batch: Batch::Static(1), geom: g, epilogue, cfg: c.gemm, split };
            let got = on_device(&spec, &target, &params);
            let worst = assert_close_f32(&format!("{g:?} {c:?}"), &got, &want, 2e-3);
            eprintln!("{}->{} {}x{} s{} {c:?}: max abs diff {worst:.2e}", g.cin, g.cout, g.h, g.w, g.stride[0]);
        }
    }
}

/// Folded batches, odd sizes, dilation and every epilogue on the device
/// against the graph and the f64 reference.
#[test_case(geom([7, 9], 32, 40, 3, 1, 1, 1), 3, RESIDUAL_SILU_BIAS, [64, 64, 32]; "ragged m and n, three images")]
#[test_case(geom([11, 6], 64, 96, 5, 2, 2, 1), 2, SILU_BIAS, [64, 96, 32]; "5x5 stride 2 on 96 wide")]
#[test_case(geom([10, 10], 16, 48, 3, 1, 2, 2), 1, SILU_BIAS, [64, 64, 16]; "dilated, 16 channels")]
#[test_case(geom([9, 9], 32, 64, 3, 1, 1, 1), 2, Epilogue { out_f32: true, residual: true, ..Epilogue::DEFAULT }, [64, 64, 32]; "f32 out and residual")]
fn conv_matches_on_the_device(g: ConvGeom, images: usize, epi: Epilogue, tile: [usize; 3]) {
    let Some(target) = cuda_target() else { return };
    let warps = if tile[1] == 48 { [2, 1] } else { [2, 2] };
    let spec = ConvSpec { batch: Batch::Static(images), geom: g, epilogue: epi, cfg: cfg(tile, warps, 3), split: 1 };
    let params = inputs(&g, images, epi, 21);
    let got = on_device(&spec, &target, &params);
    let want = reference(&g, images, epi, &params);
    let want: Vec<f32> = want.iter().map(|&w| w as f32).collect();
    assert_close_f32("y vs f64", &got, &want, tolerance(epi) as f32);
    assert_close_f32("y vs graph", &got, &graph_reference(&g, images, epi, &params), tolerance(epi) as f32);
}

/// An Inf in the first input element gives exactly the graph's set of
/// non-finite outputs: the padded taps of its windows read zero, never a
/// value multiplied by zero.
#[test]
fn an_inf_gives_the_graphs_non_finite_set_on_the_device() {
    let Some(target) = cuda_target() else { return };
    let g = geom([20, 20], 64, 64, 3, 1, 1, 1);
    let epi = SILU_BIAS;
    let mut params = inputs(&g, 1, epi, 8);
    params[0][0] = f64::INFINITY;
    let spec =
        ConvSpec { batch: Batch::Static(1), geom: g, epilogue: epi, cfg: cfg([64, 64, 32], [2, 2], 3), split: 1 };
    let got = on_device(&spec, &target, &params);
    let want = graph_reference(&g, 1, epi, &params);
    let bad = |v: &[f32]| v.iter().enumerate().filter(|(_, x)| !x.is_finite()).map(|(i, _)| i).collect::<Vec<_>>();
    assert_eq!(bad(&got), bad(&want));
    assert_eq!(bad(&got).len(), 4 * g.cout, "the four windows over the corner pixel");
}

/// A bound batch walks grid z: buffers at capacity, the plan executed at a
/// smaller live batch computes exactly those images.
#[test]
fn a_bound_batch_runs_the_live_images_on_the_device() {
    let Some(target) = cuda_target() else { return };
    let g = geom([12, 10], 32, 64, 3, 2, 1, 1);
    let [ho, wo] = g.out_hw();
    let (cap, live) = (3usize, 2usize);
    let spec = ConvSpec {
        batch: Batch::Var { name: "b".into(), min: 1, max: cap as i64 },
        geom: g,
        epilogue: RESIDUAL_SILU_BIAS,
        cfg: cfg([64, 64, 32], [2, 2], 3),
        split: 1,
    };
    let params = inputs(&g, cap, RESIDUAL_SILU_BIAS, 4);
    let tensors: Vec<Tensor> = params.iter().map(|p| upload(p, DType::Float16)).collect();
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let y = graph_launch(conv::<F16>(&spec), &spec.cfg.lowering(target), &refs).unwrap();
    let mut plan = y.prepare().unwrap();
    plan.execute_with_vars(&[("b", live as i64)]).unwrap();
    let mut bytes = vec![0u8; live * ho * wo * g.cout * 2];
    plan.output_buffer().unwrap().copyout_prefix(&mut bytes).unwrap();
    let got: Vec<f32> = bytes.chunks(2).map(|b| half_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect();
    let want = reference(&g, cap, RESIDUAL_SILU_BIAS, &params);
    let want: Vec<f32> = want[..got.len()].iter().map(|&w| w as f32).collect();
    assert_close_f32("live images", &got, &want, 2e-3);
}

fn half_to_f32(h: u16) -> f32 {
    let (sign, exp, frac) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
    let v = match exp {
        0 => frac * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -v } else { v }
}

/// `(cin, cout, input side, stride, channels-last in the model)` of the
/// memo's §3.4 classes at YOLO26x 640², batch 1, 3×3 padded 1.
const PROBE: [(usize, usize, usize, usize, bool); 12] = [
    (384, 384, 160, 2, false),
    (768, 768, 80, 2, false),
    (384, 384, 80, 2, false),
    (768, 768, 40, 2, false),
    (96, 192, 320, 2, false),
    (192, 192, 40, 1, true),
    (192, 192, 20, 1, true),
    (96, 96, 80, 1, true),
    (384, 96, 80, 1, false),
    (768, 96, 40, 1, false),
    (768, 96, 20, 1, false),
    (48, 48, 160, 1, true),
];

/// Device time of one run of `plan`: every kernel's, summed.
fn plan_ns(plan: &svod_runtime::ExecutionPlan) -> u64 {
    let profile = plan.execute_profiled().unwrap();
    profile.iter().filter_map(|k| k.gpu_end_ns?.checked_sub(k.gpu_start_ns?)).sum()
}

/// Throughput of every conv candidate per class, timed as the tune store
/// times them, against the graph's f16 conv (bias and SiLU at f32, one
/// rounding) over the layout the model holds; run under `BEAM=4` for the
/// graph arm the model gets. Prints µs and TFLOP/s and never asserts.
#[test]
#[ignore = "perf probe: needs a CUDA device"]
fn conv_throughput_probe() {
    let Some(target) = cuda_target() else { return };
    let dtype = DType::Float16;
    for (cin, cout, side, stride, chain) in PROBE {
        let g = geom([side, side], cin, cout, 3, stride, 1, 1);
        let [ho, wo] = g.out_hw();
        let flops = 2.0 * (ho * wo * cout * g.k()) as f64;
        let tflops = |ns: f64| flops / ns / 1e3;
        let label = format!("{cin}->{cout} s{stride} @{side}");
        let cands = conv_candidates(&target, 1, ho * wo, &g);
        let build = |c: ConvCfg| {
            let spec = ConvSpec { batch: Batch::Static(1), geom: g, epilogue: SILU_BIAS, cfg: c.gemm, split: c.split };
            spec.programs::<F16>(&target)
        };
        let times = crate::tune::measure(cands.iter().map(|&c| build(c)));
        for (ConvCfg { gemm: c, split }, t) in cands.iter().zip(&times) {
            let t =
                t.map_or("failed".into(), |t| format!("{:8.1} us {:5.1} TFLOP/s", t as f64 / 1e3, tflops(t as f64)));
            eprintln!("  {label} tk3 {:?} s{} {:?} unroll={} split {split}: {t}", c.tile, c.stages, c.warps, c.unroll);
        }

        let mut seed = 1;
        let mut rand = |shape: &[usize]| upload(&halves(shape.iter().product(), &mut seed, 1.0), dtype.clone());
        let shape = |dims: &[usize]| dims.iter().map(|&d| d as isize).collect::<Vec<_>>();
        let w = rand(&[cout, 3, 3, cin]).try_reshape(shape(&[cout, 3, 3, cin])).unwrap();
        let x = if chain {
            let x = rand(&[side, side, cin]).try_reshape(shape(&[1, side, side, cin])).unwrap();
            x.try_permute(&[0, 3, 1, 2]).unwrap()
        } else {
            rand(&[cin, side, side]).try_reshape(shape(&[1, cin, side, side])).unwrap()
        };
        let bias = rand(&[cout]);
        let y = x
            .conv2d()
            .weight(&w.try_permute(&[0, 3, 1, 2]).unwrap())
            .bias(&bias)
            .stride(&[stride, stride])
            .padding(&[(1, 1), (1, 1)])
            .acc_dtype(DType::Float32)
            .call()
            .unwrap()
            .silu()
            .unwrap()
            .cast(dtype.clone())
            .contiguous();
        let plan = y.prepare().unwrap();
        let warm = std::time::Instant::now();
        while warm.elapsed().as_millis() < 500 {
            plan.execute().unwrap();
        }
        let graph = (0..20).map(|_| plan_ns(&plan)).min().unwrap() as f64;
        let best = times.iter().flatten().min().map(|&t| t as f64);
        let tk3 = best.map_or("none".into(), |t| format!("{:.1} us {:.1} TFLOP/s", t / 1e3, tflops(t)));
        eprintln!("{label}: tk3 {tk3}; graph {:.1} us {:.1} TFLOP/s", graph / 1e3, tflops(graph));
    }
}
