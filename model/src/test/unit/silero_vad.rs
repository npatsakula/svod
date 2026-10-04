//! Silero V5 VAD tests.
//!
//! Cheap default tier walks the forward graph on zero inputs and asserts the
//! symbolic output shape — milliseconds, no compile, mirrors mexus's WeSpeaker
//! pattern in `test/unit/wespeaker/tstp.rs`. The `#[ignore]`-gated tier below
//! exercises the full `VadInference` JIT path (prepare → step → output reads)
//! — useful when actively touching VAD wiring but too slow for default CI.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::silero_vad::{CONTEXT_SIZE, HIDDEN, NUM_SAMPLES, SileroVad, VadHead, VadInference};

// ---------------------------------------------------------------------------
// Cheap default tests (no realize): build forward graph, check symbolic shape.
// ---------------------------------------------------------------------------

/// `forward_chunk` returns `[1, 1 + 2*HIDDEN]` — concatenated `[prob, new_h,
/// new_c]`. Catches axis bugs in STFT-conv → 4 convs → LSTM step → final
/// conv without any kernel compile.
#[test]
fn forward_chunk_zero_weights_shape() {
    let vad = SileroVad::with_random_weights();

    let chunk = Tensor::zeros(&[1, CONTEXT_SIZE + NUM_SAMPLES], DType::Float32);
    let state_h = Tensor::zeros(&[1, HIDDEN], DType::Float32);
    let state_c = Tensor::zeros(&[1, HIDDEN], DType::Float32);

    let out = vad.forward_chunk(&chunk, &state_h, &state_c).unwrap();
    let shape = crate::test::max_dims(&out);
    // [B=1, prob(1) + h(HIDDEN) + c(HIDDEN)]
    assert_eq!(shape, vec![1, 1 + 2 * HIDDEN]);
}

// ---------------------------------------------------------------------------
// SIMD scan parity: lane-wise exp/tanh vs the scalar LSTM reference.
// ---------------------------------------------------------------------------

/// Deterministic ~U(-1,1) stream (LCG); the test needs reproducibility, not
/// statistical quality.
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

/// `VadHead::scan` is lane-wise SIMD with polynomial `exp`/`tanh`. The recurrence
/// compounds drift over steps, so a 200-step scan over O(1) gate inputs must
/// agree with a scalar `f32::exp`/`tanh` reference well below the VAD
/// threshold's meaningful resolution (probs only feed a 0.5 cut).
#[test]
fn simd_scan_matches_scalar_reference() {
    let h = HIDDEN;
    let n = 200;
    let mut seed = 0x5eed;
    let mut take = |len: usize| -> Vec<f32> { (0..len).map(|_| lcg(&mut seed) * 0.5).collect() };

    let head = VadHead {
        w_hh: ndarray::Array2::from_shape_vec((4 * h, h), take(4 * h * h)).unwrap(),
        final_w: ndarray::Array1::from_vec(take(h)),
        final_b: 0.1,
    };
    // Pre-activation gates [n, 4H] — what the feature JIT delivers.
    let gates_x = take(n * 4 * h);

    // Scalar reference recurrence over the same gates.
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let mut hs = ndarray::Array1::<f32>::zeros(h);
    let mut cs = vec![0.0f32; h];
    let mut expected = Vec::with_capacity(n);
    for t in 0..n {
        let gx = &gates_x[t * 4 * h..(t + 1) * 4 * h];
        let gh = head.w_hh.dot(&hs);
        let mut p = head.final_b;
        for j in 0..h {
            let gate = |k: usize| gx[k] + gh[k];
            let i = sigmoid(gate(j));
            let f = sigmoid(gate(h + j));
            let g = gate(2 * h + j).tanh();
            let o = sigmoid(gate(3 * h + j));
            cs[j] = f * cs[j] + i * g;
            hs[j] = o * cs[j].tanh();
            p += head.final_w[j] * hs[j].max(0.0);
        }
        expected.push(sigmoid(p));
    }

    let actual = head.scan(&gates_x, n);
    assert_eq!(actual.len(), n);
    let max_abs = actual.iter().zip(&expected).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 1e-5, "SIMD scan drifted from scalar reference: max |Δprob| = {max_abs}");
}

// ---------------------------------------------------------------------------
// Heavy tests (compile + execute): unique signal, gated behind --ignored.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "heavy: full SileroVad JIT compile + execute on random weights"]
fn vad_inference_runs_on_random_weights() {
    let vad = SileroVad::with_random_weights();
    let mut inf = VadInference::new(vad).expect("prepare");
    let waveform = vec![0.0_f32; 4 * NUM_SAMPLES];
    let probs = inf.probs(&waveform).expect("probs");
    assert_eq!(probs.len(), 4, "one prob per {NUM_SAMPLES}-sample chunk");
    for p in &probs {
        assert!(p.is_finite(), "non-finite VAD prob: {p}");
        assert!((0.0..=1.0).contains(p), "VAD prob outside [0, 1]: {p}");
    }
}

#[test]
#[ignore = "heavy: full VAD pipeline + chunker on random weights"]
fn vad_segment_random_weights_ranges_are_well_formed() {
    let vad = SileroVad::with_random_weights();
    let mut inf = VadInference::new(vad).expect("prepare");
    let waveform = vec![0.0_f32; 16_000];
    let ranges = inf.segment(&waveform, 0.5);
    for (start, end) in &ranges {
        assert!(start <= end, "inverted range: ({start}, {end})");
        assert!(*end <= waveform.len(), "range past waveform end: ({start}, {end})");
    }
}
