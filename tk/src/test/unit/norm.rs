//! Tests for the row-norm kernels ([`crate::kernels::norm`]): the GPU-free
//! applicability predicate, and the hardware-gated comparison of `rms_norm` /
//! `add_rms_norm` / `layer_norm` against the graph they replace.
//!
//! `SVOD_DEVICE=CUDA cargo test -p svod-tk --lib norm -- --ignored --nocapture`.

use proptest::prelude::*;
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, LayerNorm, RmsNorm};
use svod_tensor::{Variable, jit::shrink_batch};
use test_case::test_case;

use crate::kernels::norm::{NORM_SUPPORTED_ARCHS, add_rms_norm, layer_norm, rms_norm, select_norm_cfg};

use super::device_supported;

/// The wave width the tables below are written against (CUDA / RDNA).
const W32: usize = 32;

// ── Applicability (GPU-free) ─────────────────────────────────────────────────

/// A row is servable exactly when it divides into the wave and fits in
/// registers; the block shape then takes the widest wave count the row *count*
/// divides by.
#[test_case(4096, 1024, Some(8); "the layer norm at 8x512")]
#[test_case(4096, 128, Some(8); "a head norm at 8x512")]
#[test_case(128, 1024, Some(8); "a batch-1 prefill")]
#[test_case(4, 1024, Some(4); "four rows take a four-wave block")]
#[test_case(3, 1024, Some(1); "a prime row count falls to one wave")]
#[test_case(4096, 2048, Some(8); "the widest row that still fits in registers")]
#[test_case(4096, 2080, None; "a row past the register budget declines")]
#[test_case(4096, 1000, None; "a row that does not divide the wave declines")]
#[test_case(4096, 16, None; "a row narrower than the wave declines")]
fn select_norm_cfg_applicability(rows: usize, d: usize, rows_per_block: Option<usize>) {
    assert_eq!(select_norm_cfg(rows, d, W32).map(|c| c.rows_per_block), rows_per_block, "select_norm_cfg({rows}, {d})");
}

/// A rank-1 `x` is a structured `Err`, not a panic — the shape preconditions
/// resolve before any device dispatch, so this runs GPU-free.
#[test]
fn rms_norm_low_rank_operand_is_operand_rank_err() {
    let v = Tensor::randn(&[128]).expect("randn");
    let m = Tensor::randn(&[8, 128]).expect("randn");
    let e = rms_norm(&v, &v, 1e-6).expect_err("rank-1 x must error, not panic");
    assert!(matches!(e, crate::launch::Error::OperandRank { operand: "x", .. }), "got {e:?}");
    let e = rms_norm(&m, &m, 1e-6).expect_err("rank-2 weight must error, not panic");
    assert!(matches!(e, crate::launch::Error::OperandRank { operand: "weight", .. }), "got {e:?}");
}

/// `layer_norm` resolves the same preconditions GPU-free, its bias included.
#[test]
fn layer_norm_low_rank_operand_is_operand_rank_err() {
    let v = Tensor::randn(&[128]).expect("randn");
    let m = Tensor::randn(&[8, 128]).expect("randn");
    let e = layer_norm(&v, &v, None, 1e-5).expect_err("rank-1 x must error, not panic");
    assert!(matches!(e, crate::launch::Error::OperandRank { kernel: "layer-norm", operand: "x", .. }), "got {e:?}");
    let e = layer_norm(&m, &v, Some(&m), 1e-5).expect_err("rank-2 bias must error, not panic");
    assert!(matches!(e, crate::launch::Error::OperandRank { operand: "bias", .. }), "got {e:?}");
}

proptest! {
    /// Whatever the shape, an accepted config is launchable: a legal CUDA block
    /// whose waves each own one row, and a grid that covers every row once.
    #[test]
    fn prop_selected_norm_cfg_is_launchable(rows in 1usize..300, d in 1usize..80) {
        let d = d * 32;
        let Some(cfg) = select_norm_cfg(rows, d, W32) else { return Ok(()) };
        prop_assert!(rows.is_multiple_of(cfg.rows_per_block));
        prop_assert!(cfg.rows_per_block * W32 <= 1024, "block {} threads", cfg.rows_per_block * W32);
        prop_assert!(d.is_multiple_of(W32));
    }
}

// ── Hardware-gated correctness (CUDA sm_80+) ─────────────────────────────────

/// Realize, cast to f32, and read as a host `Vec<f32>`.
fn to_f32_vec(t: &Tensor) -> Vec<f32> {
    let f = t.cast(DType::Float32).contiguous();
    f.realize().expect("realize f32");
    f.as_vec::<f32>().expect("read f32")
}

/// A realized pseudo-random operand of `dtype`, deterministic in `seed` so the
/// kernel and the reference see identical roundings.
fn operand(shape: &[usize], dtype: DType, seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5).collect();
    let dims: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
    let t = Tensor::from_slice(v).try_reshape(dims).expect("reshape").cast(dtype).contiguous();
    t.realize().expect("realize operand");
    t
}

/// Largest elementwise difference relative to the reference's own magnitude.
fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let scale = want.iter().fold(0f32, |a, b| a.max(b.abs())).max(f32::MIN_POSITIVE);
    got.iter().zip(want).fold(0f32, |a, (g, w)| a.max((g - w).abs())) / scale
}

/// The kernel and the graph run the same ops in the same dtypes; only the row
/// reduce's summation order differs (a wave butterfly against the scheduler's
/// tree), so the result can move by the bf16 rounding of a different order: two
/// ulps, `2 · 2⁻⁸ ≈ 7.8e-3` of the output's magnitude.
const BF16_REL_TOL: f32 = 8e-3;

const EPS: f64 = 1e-6;

/// `rms_norm` against `Tensor::rms_norm_with` over the same bf16 operands, at
/// the two rows that matter (the 1024-wide hidden state and the 128-wide head)
/// and several row counts, including a rank-3 activation.
#[test_case(&[4096, 1024]; "hidden rows at 8x512")]
#[test_case(&[128, 1024]; "a batch-1 prefill")]
#[test_case(&[4, 1024]; "four rows")]
#[test_case(&[65536, 128]; "head rows at 8x512")]
#[test_case(&[8, 512, 1024]; "a rank-3 activation")]
#[test_case(&[8, 512, 16, 128]; "a rank-4 head view")]
#[ignore]
fn rms_norm_matches_the_graph_gpu(shape: &[usize]) {
    if !device_supported(NORM_SUPPORTED_ARCHS) {
        eprintln!("skip rms_norm_matches_the_graph_gpu: no CUDA sm_80+ device / toolchain");
        return;
    }
    let d = *shape.last().expect("a last axis");
    let x = operand(shape, DType::BFloat16, 0.31);
    let w = operand(&[d], DType::BFloat16, 0.17);
    let y = rms_norm(&x, &w, EPS).expect("rms_norm build").expect("the kernel applies to this shape");
    assert_eq!(y.dims().expect("dims"), shape, "the output keeps x's shape");
    let want = to_f32_vec(&RmsNorm::new(w.clone(), EPS).forward(&x).expect("reference rms norm"));
    let err = rel_err(&to_f32_vec(&y), &want);
    println!("rms_norm {shape:?}: relative error {err:e}");
    assert!(err < BF16_REL_TOL, "{shape:?}: relative error {err} exceeds {BF16_REL_TOL}");
}

/// `add_rms_norm` against the graph's `x + residual` then norm: `h` is the bf16
/// sum exactly, and `y` its norm.
#[test_case(&[4096, 1024]; "hidden rows at 8x512")]
#[test_case(&[128, 1024]; "a batch-1 prefill")]
#[test_case(&[65536, 128]; "head rows")]
#[test_case(&[8, 512, 1024]; "a rank-3 activation")]
#[ignore]
fn add_rms_norm_matches_the_graph_gpu(shape: &[usize]) {
    if !device_supported(NORM_SUPPORTED_ARCHS) {
        eprintln!("skip add_rms_norm_matches_the_graph_gpu: no CUDA sm_80+ device / toolchain");
        return;
    }
    let d = *shape.last().expect("a last axis");
    let x = operand(shape, DType::BFloat16, 0.31);
    let res = operand(shape, DType::BFloat16, 0.53);
    let w = operand(&[d], DType::BFloat16, 0.17);
    let (h, y) = add_rms_norm(&x, &res, &w, EPS).expect("add_rms_norm build").expect("the kernel applies");

    let want_h = x.try_add(&res).expect("reference residual add");
    let want_y = RmsNorm::new(w.clone(), EPS).forward(&want_h).expect("reference rms norm");
    // The residual add is a plain bf16 add in both paths, so `h` is exact.
    assert_eq!(to_f32_vec(&h), to_f32_vec(&want_h), "{shape:?}: the residual stream must match bit for bit");
    let err = rel_err(&to_f32_vec(&y), &to_f32_vec(&want_y));
    println!("add_rms_norm {shape:?}: relative error {err:e}");
    assert!(err < BF16_REL_TOL, "{shape:?}: relative error {err} exceeds {BF16_REL_TOL}");
}

/// The f16 operand dtype takes the same path and comes back as f16.
#[test]
#[ignore]
fn rms_norm_f16_matches_the_graph_gpu() {
    if !device_supported(NORM_SUPPORTED_ARCHS) {
        eprintln!("skip rms_norm_f16_matches_the_graph_gpu: no CUDA sm_80+ device / toolchain");
        return;
    }
    let (rows, d) = (256usize, 1024usize);
    let x = operand(&[rows, d], DType::Float16, 0.31);
    let w = operand(&[d], DType::Float16, 0.17);
    let y = rms_norm(&x, &w, EPS).expect("rms_norm build").expect("the kernel applies");
    assert_eq!(y.uop().dtype(), DType::Float16, "the output keeps the operand dtype");
    let want = to_f32_vec(&RmsNorm::new(w.clone(), EPS).forward(&x).expect("reference"));
    let err = rel_err(&to_f32_vec(&y), &want);
    println!("rms_norm f16 {rows}x{d}: relative error {err:e}");
    // f16 carries 11 mantissa bits, so its two-ulp band is 8x tighter than bf16's.
    assert!(err < BF16_REL_TOL / 8.0, "f16: relative error {err}");
}

/// `layer_norm` of `x` against `LayerNorm::forward` of `reference` — the same
/// values, `x` possibly behind a JIT-pinned view — with `y` in `x`'s dtype. The
/// kernel's parameters are `x`'s dtype and the graph widens them the same way,
/// so the two differ only in the order of the two row reduces.
fn check_layer_norm(x: &Tensor, reference: &Tensor, dtype: DType, bias: bool, tol: f32) {
    let dims = reference.dims().expect("dims");
    let d = *dims.last().expect("a last axis");
    let w = operand(&[d], dtype.clone(), 0.17);
    let b = bias.then(|| operand(&[d], dtype.clone(), 0.71));
    let y = layer_norm(x, &w, b.as_ref(), EPS).expect("layer_norm build").expect("the kernel applies to this shape");
    assert_eq!(y.dims().expect("dims"), dims, "the output has x's shape, static");
    assert_eq!(y.uop().dtype(), dtype, "the output keeps the operand dtype");
    let want = to_f32_vec(&LayerNorm::new(w, b, EPS).forward(reference).expect("reference layer norm"));
    let err = rel_err(&to_f32_vec(&y), &want);
    println!("layer_norm {dims:?} bias {bias}: relative error {err:e}");
    assert!(err < tol, "{dims:?} bias {bias}: relative error {err} exceeds {tol}");
}

/// `layer_norm` against the graph at ModernBERT's 768-wide rows and the other
/// widths the ladder serves, with and without a bias.
#[test_case(&[512, 768], false; "modernbert rows at 1x512")]
#[test_case(&[4096, 768], false; "modernbert rows at 8x512")]
#[test_case(&[8, 512, 768], true; "a rank-3 activation with a bias")]
#[test_case(&[128, 1024], true; "a 1024-wide row with a bias")]
#[test_case(&[4, 32], false; "a row one wave wide")]
#[test_case(&[3, 2048], true; "the widest row at a prime row count")]
#[ignore]
fn layer_norm_matches_the_graph_gpu(shape: &[usize], bias: bool) {
    if !device_supported(NORM_SUPPORTED_ARCHS) {
        eprintln!("skip layer_norm_matches_the_graph_gpu: no CUDA sm_80+ device / toolchain");
        return;
    }
    let x = operand(shape, DType::BFloat16, 0.31);
    check_layer_norm(&x, &x, DType::BFloat16, bias, BF16_REL_TOL);
}

/// Rows whose mean is thousands of times their spread: the kernel centers before it
/// squares, as the graph does. A one-pass `E[x²] − E[x]²` in f32 would lose the
/// variance to cancellation here (f32's ulp at `E[x²] ≈ 10⁶` is half of it).
/// f16, since bf16 cannot hold the spread at that offset.
#[test]
#[ignore]
fn layer_norm_f16_far_from_zero_mean_matches_the_graph_gpu() {
    if !device_supported(NORM_SUPPORTED_ARCHS) {
        eprintln!("skip layer_norm_f16_far_from_zero_mean_matches_the_graph_gpu: no CUDA sm_80+ device / toolchain");
        return;
    }
    let x = operand(&[256, 768], DType::Float16, 0.31).try_add(Tensor::const_(1000.0, DType::Float16)).expect("shift");
    let x = x.contiguous();
    x.realize().expect("realize shifted operand");
    check_layer_norm(&x, &x, DType::Float16, true, BF16_REL_TOL / 8.0);
}

/// A batch the JIT pins to one value is that value to the kernel, as it is to
/// the GEMM: the operand still carries it symbolically, and the launcher
/// reshapes it static.
#[test]
#[ignore]
fn layer_norm_takes_a_jit_pinned_batch_gpu() {
    if !device_supported(NORM_SUPPORTED_ARCHS) {
        eprintln!("skip layer_norm_takes_a_jit_pinned_batch_gpu: no CUDA sm_80+ device / toolchain");
        return;
    }
    let x = operand(&[2, 128, 768], DType::BFloat16, 0.31);
    let pinned = shrink_batch(&x, &Variable::new("b", 2, 2).bind(2).expect("bind")).expect("shrink");
    assert_eq!(pinned.shape().expect("shape")[0].as_const(), None, "the batch stays symbolic");
    check_layer_norm(&pinned, &x, DType::BFloat16, false, BF16_REL_TOL);
}

/// A row no block shape covers declines (`Ok(None)`) so the caller keeps its
/// graph; a malformed request is a structured `Err`. Both need a supported
/// device: on anything else `launch_custom` declines before it looks at the
/// request at all.
#[test]
#[ignore]
fn norm_outcomes_gpu() {
    if !device_supported(NORM_SUPPORTED_ARCHS) {
        eprintln!("skip norm_outcomes_gpu: no CUDA sm_80+ device / toolchain");
        return;
    }
    let ragged = operand(&[8, 1000], DType::BFloat16, 0.31);
    let w1000 = operand(&[1000], DType::BFloat16, 0.17);
    assert!(rms_norm(&ragged, &w1000, EPS).expect("ragged D builds").is_none(), "D % 32 != 0 must decline");

    let x = operand(&[8, 1024], DType::BFloat16, 0.31);
    let w = operand(&[1024], DType::BFloat16, 0.17);
    let f32x = operand(&[8, 1024], DType::Float32, 0.31);
    let e = rms_norm(&f32x, &w, EPS).expect_err("an f32 operand is a caller bug");
    assert!(matches!(e, crate::launch::Error::Dtype { kernel: "rms-norm", .. }), "got {e:?}");

    let e = rms_norm(&x, &w1000, EPS).expect_err("a weight that is not [D] is a caller bug");
    assert!(matches!(e, crate::launch::Error::OperandShape { operand: "weight", .. }), "got {e:?}");

    let e = add_rms_norm(&x, &operand(&[8, 512], DType::BFloat16, 0.5), &w, EPS)
        .expect_err("a residual of another shape is a caller bug");
    assert!(matches!(e, crate::launch::Error::OperandShape { operand: "residual", .. }), "got {e:?}");

    let b = operand(&[1024], DType::BFloat16, 0.71);
    assert!(layer_norm(&ragged, &w1000, None, EPS).expect("ragged D builds").is_none(), "D % 32 != 0 must decline");
    let e = layer_norm(&x, &w, Some(&w1000), EPS).expect_err("a bias that is not [D] is a caller bug");
    assert!(matches!(e, crate::launch::Error::OperandShape { kernel: "layer-norm", operand: "bias", .. }), "got {e:?}");
    let e = layer_norm(&x, &w, Some(&operand(&[1024], DType::Float16, 0.71)), EPS)
        .expect_err("a bias in another dtype is a caller bug");
    assert!(matches!(e, crate::launch::Error::Dtype { kernel: "layer-norm", .. }), "got {e:?}");
    let e = layer_norm(&f32x, &w, Some(&b), EPS).expect_err("an f32 operand is a caller bug");
    assert!(matches!(e, crate::launch::Error::Dtype { kernel: "layer-norm", .. }), "got {e:?}");
}
