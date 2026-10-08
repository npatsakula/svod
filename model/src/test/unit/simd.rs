use fearless_simd::{Level, Simd, dispatch, prelude::*};
use test_case::test_case;

use crate::simd::{sigmoid, tanh};

/// Applies a lane-wise function to every value; the tail that does not fill a
/// vector is padded with zeros.
fn apply<S: Simd>(simd: S, values: &[f32], op: impl Fn(S, S::f32s) -> S::f32s) -> Vec<f32> {
    let n = S::f32s::LEN;
    values
        .chunks(n)
        .flat_map(|chunk| {
            let mut lanes = vec![0.0; n];
            lanes[..chunk.len()].copy_from_slice(chunk);
            op(simd, S::f32s::from_slice(simd, &lanes)).as_slice()[..chunk.len()].to_vec()
        })
        .collect()
}

fn sigmoid_all(values: &[f32]) -> Vec<f32> {
    dispatch!(Level::new(), simd => apply(simd, values, sigmoid))
}

fn tanh_all(values: &[f32]) -> Vec<f32> {
    dispatch!(Level::new(), simd => apply(simd, values, tanh))
}

/// A dense sweep over [-20, 20] plus the points the formulas switch at: zero, the sign
/// change, the exponent floor and far past it on both sides.
fn inputs() -> Vec<f32> {
    let sweep = (-400..=400).map(|i| i as f32 * 0.05);
    let edges =
        [0.0, -0.0, 1e-30, -1e-30, 43.0, -43.0, 44.0, -44.0, 87.0, -87.0, 88.0, -88.0, 1e4, -1e4, f32::MAX, f32::MIN];
    sweep.chain(edges).collect()
}

fn reference_sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + f64::from(-x).exp())) as f32
}

#[test_case(sigmoid_all, reference_sigmoid, 2e-7; "sigmoid")]
#[test_case(tanh_all, |x| f64::from(x).tanh() as f32, 4e-7; "tanh")]
fn lanewise_matches_the_f64_reference(simd: fn(&[f32]) -> Vec<f32>, reference: fn(f32) -> f32, tolerance: f32) {
    let xs = inputs();
    for (&x, got) in xs.iter().zip(simd(&xs)) {
        let want = reference(x);
        assert!(got.is_finite(), "x = {x}: {got}");
        assert!((got - want).abs() <= tolerance, "x = {x}: got {got}, want {want}");
    }
}
