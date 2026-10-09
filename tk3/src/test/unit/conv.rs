//! The implicit-GEMM convolution: the interpreter against an f64 direct
//! convolution, the emitted program's row decode and fills on sm_86, and
//! the config table's invariants.

use std::sync::Arc;

use proptest::prelude::*;
use svod_dtype::{DType, DeviceSpec, ScalarDType};
use svod_ir::{BinaryOp as UBinary, Op, UOp, ops};
use test_case::test_case;

use crate::atoms::sm86;
use crate::build::F16;
use crate::interp::{round_to, run};
use crate::ir::Program;
use crate::kernels::conv::{ConvGeom, ConvSpec, conv};
use crate::kernels::gemm::{Epilogue, GemmCfg};
use crate::kernels::{Act, Batch};
use crate::lower::lower;
use crate::ops::config::{conv_candidates, conv_cfg_fits};

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
    let prog = conv::<F16>(&ConvSpec { batch: batch.clone(), geom: g, epilogue: epi, cfg: c });
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

/// The padding is data: an Inf in the corner pixel reaches exactly the
/// outputs whose window covers it, and no padded tap turns it into NaN.
#[test]
fn an_inf_reaches_only_the_windows_over_it() {
    let g = geom([6, 6], 16, 8, 3, 1, 1, 1);
    let epi = Epilogue::DEFAULT;
    let prog =
        conv::<F16>(&ConvSpec { batch: Batch::Static(1), geom: g, epilogue: epi, cfg: cfg([32, 32, 16], [2, 2], 2) });
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
    let list = emitted(conv::<F16>(&ConvSpec { batch: Batch::Static(1), geom: g, epilogue: SILU_BIAS, cfg: c }), c);
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
        let cands = conv_candidates(&target, 1, ho * wo, g.cout, g.k(), g.cin);
        assert!(!cands.is_empty() || g.cin == 48, "{g:?}");
        for c in cands {
            assert!(conv_cfg_fits(&c) && g.cin.is_multiple_of(c.tile[2]) && c.smem_bytes(false) <= target.smem_bytes);
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
    let lead = conv_candidates(&sm86(), 1, ho * wo, cout, g.k(), cin).first().map(|c| c.tile);
    assert_eq!(lead, tile);
}

#[test_case(3; "rgb stem")]
#[test_case(8; "eight channels")]
#[test_case(40; "a multiple of 8 only")]
fn too_few_channels_have_no_candidates(cin: usize) {
    assert!(conv_candidates(&sm86(), 1, 6400, 64, 9 * cin, cin).is_empty());
}
