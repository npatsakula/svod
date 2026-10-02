//! Per-image batch correctness: image `i` of a batched run must equal a solo run
//! of image `i`.
//!
//! Nothing else in the tree asserts this. `yolo::model`'s symbolic-batch test
//! checks that the batch axis stays a variable, and `yolo::jit`'s checks that the
//! predictions *shape* follows the bound batch — neither looks at a value, so a
//! kernel that mixed two images would pass both.
//!
//! The property is load-bearing for the batched tk convolution: `ConvGeom::mkn`
//! folds the batch into the implicit GEMM's `M = batch·OH·OW`, and a strip row
//! decodes back to an image with `b = m / (ho·wo)`. Get that decode wrong and
//! images blend, which no shape assertion can see.
//!
//! ```text
//! cargo test --release -p svod-model --lib yolo::batch -- --ignored
//! SVOD_DEVICE=CUDA cargo test --release -p svod-model --lib yolo::batch -- --ignored
//! ```

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Module, StateDict};

use crate::jit::InputSpec;
use crate::yolo::{Yolo26Detect, Yolo26DetectJit, YoloConfig, YoloScale};

/// Weight scale as a multiple of the variance-preserving `sqrt(3/fan_in)`.
///
/// This is not cosmetic. YOLO26 is ~100 convolutions deep and inference-mode
/// batch norm is a fixed affine, so it rescales nothing: the per-layer variance
/// ratio is raised to the hundredth power and the end-to-end signal is a cliff.
/// Measured on Nano at 64², the inter-image spread is 8.3e-2 at gain 1.30, 1.16
/// at 1.50, and 4.4e2 at 1.60 (with `max |out|` already 1.6e3, past any plausible
/// box coordinate). At PyTorch's own default init — gain 1.0 — it is 1.5e-3: the
/// output is then just the anchor decode, identical for every input, and every
/// assertion below would pass vacuously.
///
/// 1.50 is the plateau: spread 1.16 against outputs bounded by ~60, i.e. real box
/// coordinates for a 64² image. The same value serves f16, which tracks f32 to
/// three digits here.
///
/// The assertions never trust this constant. They scale against the *measured*
/// spread and [`assert_healthy`] fails loudly if it collapses, so a change that
/// moves the cliff is reported rather than silently making the suite vacuous.
const GAIN: f64 = 1.50;

/// Square input side. 64 gives 8²+4²+2² = 84 anchors across the three strides.
const SIDE: usize = 64;

/// Classes. 2 keeps the score channels plural without widening the head.
const NC: usize = 2;

/// How far a batched run may drift from a solo run, as a fraction of the measured
/// inter-image spread.
///
/// Not bit-exactness: a different batch extent lets the scheduler pick a different
/// tiling, which reassociates the conv reductions. Measured on CPU at batch 2/3/4,
/// the worst drift is 3.8e-6 absolute in f32 (3.7e-6 of the spread) and exactly
/// zero in f16, so this leaves three orders of headroom for a GPU tiling that
/// reassociates harder — while still sitting two orders *below* a leak, which
/// moves a value by order the spread itself.
const REASSOCIATION_FRACTION: f32 = 1e-2;

/// Below this the model has collapsed to an input-independent constant and every
/// comparison would be meaningless. A healthy run gives ~1.16.
const MIN_SPREAD: f32 = 1e-2;

/// Deterministic per-image content. Distinct images are the whole point: with the
/// same image in every slot, a kernel that averaged across the batch would still
/// return the right answer.
fn image(side: usize, k: usize) -> Vec<f32> {
    (0..3 * side * side)
        .map(|i| {
            let (c, p) = (i / (side * side), i % (side * side));
            let (h, w) = (p / side, p % side);
            (((h * 7 + w * 13 + c * 29 + k * 101) % 251) as f32) / 251.0
        })
        .collect()
}

/// Per-tensor deterministic stream, seeded from the tensor's own key (FNV-1a).
///
/// Deliberately not `Tensor::uniform`: that draws from a process-global generator
/// which `manual_seed` does not fully reset (it bumps an epoch, and rand graphs
/// are cached across calls), and a `StateDict` is a `HashMap` whose iteration
/// order varies per run. Between them, two "identically seeded" models get
/// different weights — which surfaces as a batched run disagreeing with a solo run
/// by the spread of the whole output, i.e. indistinguishable from the leak this
/// file exists to detect. Keying the stream off the name makes every weight a pure
/// function of `(key, gain)`.
fn key_stream(key: &str) -> u64 {
    key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3))
}

/// splitmix64, mapped to `[-1, 1)`.
fn next_signed_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    ((z >> 11) as f64 / (1u64 << 53) as f64).mul_add(2.0, -1.0)
}

/// Checkpoint-format weights over a freshly built model's own keys, so the model
/// goes through the production `from_state_dict` path (batch-norm fold, dtype
/// cast, channels-innermost realize) rather than a test-only shortcut.
fn random_state_dict(template: &StateDict, gain: f64) -> StateDict {
    template
        .iter()
        .map(|(key, t)| {
            let dims = t.dims().expect("a freshly built model has concrete placeholder dims");
            let value = if dims.len() == 4 {
                let fan_in: usize = dims[1..].iter().product();
                let bound = gain * (3.0 / fan_in.max(1) as f64).sqrt();
                let mut state = key_stream(key);
                let n: usize = dims.iter().product();
                Tensor::from_slice((0..n).map(|_| (bound * next_signed_unit(&mut state)) as f32).collect::<Vec<f32>>())
            } else if key.ends_with("bn.weight") || key.ends_with("bn.running_var") {
                // Identity norm: the fold then contributes scale 1 and bias 0, so
                // `gain` alone sets the per-layer gain.
                Tensor::ones(&dims, DType::Float32)
            } else {
                Tensor::zeros(&dims, DType::Float32)
            };
            let shape = dims.iter().map(|&d| d as isize).collect::<Vec<_>>();
            (key.clone(), value.try_reshape(shape).expect("weight reshape").contiguous())
        })
        .collect()
}

fn random_model(dtype: &DType) -> Yolo26Detect {
    let template = Yolo26Detect::with_zero_weights(YoloConfig::new(YoloScale::Nano, NC)).state_dict("");
    let config = YoloConfig::new(YoloScale::Nano, NC).with_compute_dtype(dtype.clone());
    Yolo26Detect::from_state_dict(&random_state_dict(&template, GAIN), config).expect("random checkpoint loads")
}

fn input(batch: &[usize]) -> Tensor {
    let data: Vec<f32> = batch.iter().flat_map(|&k| image(SIDE, k)).collect();
    Tensor::from_slice(data)
        .try_reshape(vec![batch.len() as isize, 3, SIDE as isize, SIDE as isize])
        .expect("NCHW reshape of a batch·3·side² buffer")
}

/// Forward `batch` and split the `[B, 4+nc, A]` predictions into per-image slices.
fn predictions(model: &Yolo26Detect, batch: &[usize]) -> Vec<Vec<f32>> {
    let out = model.forward(&input(batch)).expect("forward");
    let dims = out.dims().expect("concrete prediction dims");
    assert_eq!(dims[0], batch.len(), "predictions lost the batch axis: {dims:?}");
    let flat = out.to_vec::<f32>().expect("realize predictions");
    let per = dims[1] * dims[2];
    assert_eq!(flat.len(), batch.len() * per);
    flat.chunks_exact(per).map(<[f32]>::to_vec).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |acc, (&x, &y)| acc.max((x - y).abs()))
}

/// The *smallest* difference between any two of these images. A leak has to move a
/// value by about this much to be real, so it sets the scale every tolerance below
/// is expressed in — and taking the minimum makes that scale the pessimistic one.
fn inter_image_spread(outputs: &[Vec<f32>]) -> f32 {
    let mut spread = f32::MAX;
    for (i, a) in outputs.iter().enumerate() {
        for b in &outputs[i + 1..] {
            spread = spread.min(max_abs_diff(a, b));
        }
    }
    spread
}

/// Finite, and actually input-dependent. Returns the spread to scale tolerances by.
fn assert_healthy(outputs: &[Vec<f32>]) -> f32 {
    for (i, o) in outputs.iter().enumerate() {
        assert!(o.iter().all(|v| v.is_finite()), "image {i} produced a non-finite prediction");
    }
    let spread = inter_image_spread(outputs);
    assert!(
        spread > MIN_SPREAD,
        "the model collapsed to an input-independent constant (inter-image spread {spread:.3e} \
         <= {MIN_SPREAD:.0e}); every comparison here would pass vacuously — retune GAIN"
    );
    spread
}

/// The core property, over several batch extents. An odd batch is included on
/// purpose: it is the one that leaves the GEMM's `M` ragged against the tile.
#[test_case::test_case(2, DType::Float32; "batch 2 f32")]
#[test_case::test_case(3, DType::Float32; "batch 3 f32 (ragged M)")]
#[test_case::test_case(4, DType::Float32; "batch 4 f32")]
#[test_case::test_case(2, DType::Float16; "batch 2 f16 (tk conv path on CUDA)")]
#[test_case::test_case(3, DType::Float16; "batch 3 f16 (tk conv path on CUDA)")]
#[ignore = "heavy: one full Yolo26n detect forward per image, plus one batched"]
fn batch_matches_solo_runs(batch: usize, dtype: DType) {
    let model = random_model(&dtype);
    let slots: Vec<usize> = (0..batch).collect();

    let solo: Vec<Vec<f32>> = slots.iter().map(|&k| predictions(&model, &[k]).remove(0)).collect();
    let spread = assert_healthy(&solo);
    let tol = spread * REASSOCIATION_FRACTION;

    for (i, (b, s)) in predictions(&model, &slots).iter().zip(&solo).enumerate() {
        let diff = max_abs_diff(b, s);
        assert!(
            diff <= tol,
            "image {i} of a batch of {batch} ({dtype:?}) differs from its solo run by {diff:.3e}, \
             tolerance {tol:.3e} (inter-image spread {spread:.3e}) — the batch axis is leaking"
        );
    }
}

/// A slot's output must not depend on what shares the batch with it. This catches
/// the leak the solo comparison can miss: a kernel that mixed images in a fixed,
/// position-dependent way could still agree with a solo run and only diverge once
/// the neighbours change.
#[test_case::test_case(DType::Float32; "f32")]
#[test_case::test_case(DType::Float16; "f16 (tk conv path on CUDA)")]
#[ignore = "heavy: three batched Yolo26n detect forwards"]
fn batch_slot_output_ignores_its_neighbours(dtype: DType) {
    let model = random_model(&dtype);

    // Same leading image, different companions; then the same image moved to a
    // different slot, which a position-dependent leak would also disturb.
    let with_b = predictions(&model, &[0, 1, 2]);
    let with_c = predictions(&model, &[0, 3, 4]);
    let moved = predictions(&model, &[5, 6, 0]);

    let spread = assert_healthy(&[with_b[0].clone(), with_b[1].clone(), with_c[1].clone()]);
    let tol = spread * REASSOCIATION_FRACTION;

    let diff = max_abs_diff(&with_b[0], &with_c[0]);
    assert!(
        diff <= tol,
        "slot 0 ({dtype:?}) moved by {diff:.3e} when only slots 1-2 changed, tolerance {tol:.3e} \
         (inter-image spread {spread:.3e}) — a neighbour is bleeding into it"
    );

    let diff = max_abs_diff(&with_b[0], &moved[2]);
    assert!(
        diff <= tol,
        "the same image ({dtype:?}) differs by {diff:.3e} between slot 0 and slot 2, tolerance \
         {tol:.3e} — the result depends on batch position"
    );
}

/// The same property through the JIT with the batch variable pinned, which is how
/// a batched model is actually served: `svod_tk::conv2d_nhwc` needs a statically
/// known dim 0, so a free `batch_var` fails `prepare` outright at a tensor-core
/// dtype rather than falling back to the graph conv. `with_b_fixed` is what makes
/// the tk path reachable at batch > 1 — and what lets the plan be graph-captured,
/// since capture is gated on no kernel carrying an unbound var.
#[test_case::test_case(2, DType::Float32; "batch 2 f32")]
#[test_case::test_case(3, DType::Float32; "batch 3 f32 (ragged M)")]
#[test_case::test_case(2, DType::Float16; "batch 2 f16 (tk conv path on CUDA)")]
#[ignore = "heavy: a full detect graph compile per batch extent"]
fn pinned_batch_jit_matches_solo_runs(batch: usize, dtype: DType) {
    let slots: Vec<usize> = (0..batch).collect();
    // One model for both sides: two builds would differ only if the weights were
    // not a pure function of the key, and that difference would read as a leak.
    let model = random_model(&dtype);
    let solo: Vec<Vec<f32>> = slots.iter().map(|&k| predictions(&model, &[k]).remove(0)).collect();
    let spread = assert_healthy(&solo);
    let tol = spread * REASSOCIATION_FRACTION;

    let mut jit = Yolo26DetectJit::new(model).with_b_fixed(batch);
    jit.prepare(InputSpec::f32(&[batch, 3, SIDE, SIDE])).expect("pinned-batch prepare");
    let batched: Vec<f32> = slots.iter().flat_map(|&k| image(SIDE, k)).collect();
    jit.images_mut().expect("images slot").copyin(bytemuck::cast_slice(&batched)).expect("copy the batch in");
    jit.execute_bound(batch as i64).expect("execute the pinned batch");

    let shape = jit.predictions_shape().expect("predictions shape");
    assert_eq!(shape[0], batch, "predictions lost the batch axis: {shape:?}");
    let flat = jit.predictions_to_vec::<f32>().expect("read predictions back");
    let per = shape[1] * shape[2];
    assert_eq!(flat.len(), batch * per);

    for (i, (b, s)) in flat.chunks_exact(per).zip(&solo).enumerate() {
        let diff = max_abs_diff(b, s);
        assert!(
            diff <= tol,
            "JIT image {i} of a pinned batch of {batch} ({dtype:?}) differs from its solo run by \
             {diff:.3e}, tolerance {tol:.3e} (inter-image spread {spread:.3e})"
        );
    }
}

/// A guard on the guards above: the tolerance has to actually reject a wrong
/// image. `tol` is a hundredth of the *smallest* difference between two distinct
/// images, so this holds by construction — but it is the construction that would
/// break if someone loosened [`REASSOCIATION_FRACTION`] or flattened the model,
/// and then every comparison above would start passing for the wrong reason.
#[test]
#[ignore = "heavy: two Yolo26n detect forwards"]
fn the_comparison_rejects_a_swapped_image() {
    let model = random_model(&DType::Float32);
    let solo: Vec<Vec<f32>> = (0..2).map(|k| predictions(&model, &[k]).remove(0)).collect();
    let spread = assert_healthy(&solo);
    let tol = spread * REASSOCIATION_FRACTION;
    let swapped = max_abs_diff(&solo[0], &solo[1]);
    assert!(
        swapped > tol * 10.0,
        "comparing image 0 against image 1 differs by only {swapped:.3e} against a tolerance of \
         {tol:.3e} — the assertions in this file could not tell two images apart"
    );
}
