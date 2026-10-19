//! Unit + property tests for the shared numeric comparator
//! [`crate::testing::allclose_f32`]. CPU-only (no device), so they run in normal
//! `cargo test` — the guard rails the custom-kernel checks depend on.

use proptest::prelude::*;

use crate::testing::allclose_f32;

#[test]
fn equal_slices_pass_at_zero_tol() {
    let v = vec![1.0f32, -2.0, 3.5, 0.0, 1e6];
    assert!(allclose_f32(&v, &v, 0.0, 0.0).ok);
}

#[test]
fn absolute_tolerance() {
    let got = [1.01f32, 2.0, 3.0];
    let exp = [1.0f32, 2.0, 3.0];
    assert!(allclose_f32(&got, &exp, 2e-2, 0.0).ok); // |Δ|=0.01 ≤ 2e-2
    assert!(!allclose_f32(&got, &exp, 1e-3, 0.0).ok); // 0.01 > 1e-3
}

#[test]
fn relative_tolerance_scales_with_reference() {
    // |Δ|=1.0; rtol·|e| = 0.02·100 = 2.0 ≥ 1.0 → pass.
    assert!(allclose_f32(&[101.0], &[100.0], 0.0, 2e-2).ok);
    // |Δ|=1.0; rtol·|e| = 0.02·10 = 0.2 < 1.0, atol 0 → fail.
    assert!(!allclose_f32(&[11.0], &[10.0], 0.0, 2e-2).ok);
}

#[test]
fn nan_in_got_vs_finite_always_fails() {
    // The silent-NaN hole the old `f32::max` loop swallowed: must be caught even
    // under an enormous tolerance.
    let got = [1.0f32, f32::NAN, 3.0];
    let exp = [1.0f32, 2.0, 3.0];
    let r = allclose_f32(&got, &exp, 1e9, 1e9);
    assert!(!r.ok);
    assert_eq!(r.first_failure, Some(1));
}

#[test]
fn all_non_finite_failure_reports_nonzero_max_err() {
    // When every mismatch is non-finite, max_abs_err must not stay at its 0.0
    // init (which read as a misleading "max abs err 0e0") — surface it as +inf.
    let got = [f32::NAN, f32::NAN];
    let exp = [1.0f32, 2.0];
    let r = allclose_f32(&got, &exp, 1e9, 1e9);
    assert!(!r.ok);
    assert_eq!(r.first_failure, Some(0));
    assert!(r.max_abs_err.is_infinite(), "all-NaN failure should report inf, got {}", r.max_abs_err);
}

#[test]
fn inf_empty_and_length_mismatch_fail() {
    assert!(!allclose_f32(&[f32::INFINITY], &[1.0], 1e9, 1e9).ok); // inf vs finite
    assert!(!allclose_f32(&[], &[], 1.0, 1.0).ok); // empty → no vacuous pass
    assert!(!allclose_f32(&[1.0, 2.0], &[1.0], 1.0, 1.0).ok); // length mismatch
}

#[test]
fn matching_nonfinite_pass() {
    assert!(allclose_f32(&[f32::NAN], &[f32::NAN], 0.0, 0.0).ok);
    assert!(allclose_f32(&[f32::INFINITY], &[f32::INFINITY], 0.0, 0.0).ok);
    assert!(!allclose_f32(&[f32::INFINITY], &[f32::NEG_INFINITY], 0.0, 0.0).ok); // sign mismatch
}

proptest! {
    /// Identical finite data is allclose at zero tolerance — no false negatives.
    #[test]
    fn prop_equal_is_allclose(v in prop::collection::vec(-1e3f32..1e3, 1..64)) {
        let r = allclose_f32(&v, &v, 0.0, 0.0);
        prop_assert!(r.ok, "{}", r.message);
        prop_assert_eq!(r.max_abs_err, 0.0);
    }

    /// A single injected NaN is always caught regardless of tolerance — the hole
    /// the comparator exists to close.
    #[test]
    fn prop_injected_nan_always_fails(
        mut v in prop::collection::vec(-1e3f32..1e3, 1..64),
        idx in any::<prop::sample::Index>(),
        tol in 0.0f32..1e6,
    ) {
        let exp = v.clone();
        let i = idx.index(v.len());
        v[i] = f32::NAN;
        prop_assert!(!allclose_f32(&v, &exp, tol, tol).ok);
    }

    /// A perturbation safely under `atol` (rtol 0) always passes. Uses half-atol
    /// margin so f32 rounding at large `|x|` can't push `|Δ|` over the bound.
    #[test]
    fn prop_within_atol_passes(
        base in prop::collection::vec(-1e2f32..1e2, 1..32),
        atol in 1e-3f32..1.0,
        sign in any::<bool>(),
    ) {
        let delta = if sign { 0.5 * atol } else { -0.5 * atol };
        let got: Vec<f32> = base.iter().map(|&x| x + delta).collect();
        let r = allclose_f32(&got, &base, atol, 0.0);
        prop_assert!(r.ok, "{}", r.message);
    }
}
