//! Numeric comparison helpers shared by custom-kernel correctness checks — the
//! one-shot [`custom_kernel_check!`](crate::custom_kernel_check) macro and any
//! property-based harness. Kept in the library (not behind `cfg(test)`) so the
//! `#[macro_export]` macro and downstream test crates can both reach it.

/// Outcome of an [`allclose_f32`] comparison.
pub struct AllcloseReport {
    /// `true` iff every element passed `|got − exp| ≤ atol + rtol·|exp|` and no
    /// finite/non-finite mismatch occurred.
    pub ok: bool,
    /// Largest `|got − exp|` over the finite-vs-finite pairs (`0.0` if none).
    pub max_abs_err: f32,
    /// Flat index of the first failing element, if any.
    pub first_failure: Option<usize>,
    /// Human-readable summary — the assertion message on failure.
    pub message: String,
}

/// Elementwise `allclose` for two f32 slices with combined absolute + relative
/// tolerance: each pair must satisfy `|got − exp| ≤ atol + rtol·|exp|`.
///
/// Stricter than a bare tolerance loop in two ways that matter for kernel checks:
/// an **empty or length-mismatched** pair fails (no vacuous pass), and a
/// **finite/non-finite mismatch** fails explicitly — a kernel emitting `NaN`/`inf`
/// where the reference is finite is caught, instead of being swallowed by
/// `f32::max`'s NaN-ignoring behavior. (Two non-finite values pass only when they
/// match in kind: `NaN`↔`NaN`, `+inf`↔`+inf`, `−inf`↔`−inf`.)
pub fn allclose_f32(got: &[f32], expected: &[f32], atol: f32, rtol: f32) -> AllcloseReport {
    if got.len() != expected.len() {
        return AllcloseReport {
            ok: false,
            max_abs_err: f32::INFINITY,
            first_failure: Some(0),
            message: format!("length mismatch: got {} vs expected {}", got.len(), expected.len()),
        };
    }
    if got.is_empty() {
        return AllcloseReport {
            ok: false,
            max_abs_err: 0.0,
            first_failure: Some(0),
            message: "empty comparison (no elements to check)".to_string(),
        };
    }

    let mut max_abs_err = 0.0f32;
    let mut first_failure = None;
    let mut fail_msg = String::new();
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        let fail = if !g.is_finite() || !e.is_finite() {
            // Non-finite: require identical classification (NaN↔NaN, ±inf matching).
            let mismatch = g.is_nan() != e.is_nan() || (!g.is_nan() && g != e);
            if mismatch {
                // A non-finite mismatch has no finite magnitude; surface it as +inf
                // so an all-non-finite failure reports `inf`, not a misleading `0e0`.
                max_abs_err = f32::INFINITY;
            }
            mismatch
        } else {
            let abs = (g - e).abs();
            if abs > max_abs_err {
                max_abs_err = abs;
            }
            abs > atol + rtol * e.abs()
        };
        if fail && first_failure.is_none() {
            first_failure = Some(i);
            fail_msg = format!(
                "element {i}: got {g:e}, expected {e:e} (|Δ|={:e}, tol={:e})",
                (g - e).abs(),
                atol + rtol * e.abs()
            );
        }
    }

    let ok = first_failure.is_none();
    let message = if ok {
        format!("allclose: max abs err {max_abs_err:e} within atol {atol:e} + rtol {rtol:e}·|e|")
    } else {
        format!("allclose FAILED at {fail_msg}; max abs err {max_abs_err:e}")
    };
    AllcloseReport { ok, max_abs_err, first_failure, message }
}
