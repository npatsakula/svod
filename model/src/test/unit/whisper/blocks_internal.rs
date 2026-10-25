//! Whisper block internals: FP16 LayerNorm and Linear must keep their affine
//! and bias epilogues in FP32 and round only once, at the final cast.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, LayerNorm, Linear};

use crate::whisper::blocks::linear_forward;

fn realized_f32(tensor: Tensor) -> Vec<f32> {
    tensor.cast(DType::Float32).to_vec::<f32>().unwrap()
}

fn fp16(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(values).try_reshape(shape.to_vec()).unwrap().cast(DType::Float16)
}

/// `legacy` is the all-FP16 formulation the epilogue replaced: it must differ,
/// or the fixture would pass without proving anything.
fn assert_rounds_once(actual: Tensor, reference: Tensor, legacy: Tensor, epilogue: &str) {
    let reference = realized_f32(reference);
    assert_eq!(realized_f32(actual), reference, "{epilogue} must run on the FP32 accumulator");
    assert_ne!(realized_f32(legacy), reference, "fixture must detect rounding before the {epilogue}");
}

#[test]
fn fp16_layernorm_keeps_affine_in_fp32_until_final_cast() {
    // Values whose affine result sits across an FP16 boundary, so rounding
    // the normalized value and each product (the legacy form) differs from
    // rounding once at the end: since a Float16 roundtrip rounds instead of
    // folding away, only such values keep this fixture discriminative.
    let x = fp16(&[0.15673828125, -0.1868896484375, -2.517578125, -0.53857421875], &[1, 4]);
    let weight = fp16(&[-0.387939453125, 0.90625, -12.2421875, -3.822265625], &[4]);
    let bias = fp16(&[-3.9140625, -3.234375, 4.2421875, -3.23046875], &[4]);
    let layer = LayerNorm::new(weight.clone(), Some(bias.clone()), 1e-5);
    let f32_of = |t: &Tensor| t.cast(DType::Float32);

    let reference = f32_of(&x)
        .layernorm(-1, layer.eps)
        .unwrap()
        .try_mul(f32_of(&weight))
        .unwrap()
        .try_add(f32_of(&bias))
        .unwrap()
        .cast(DType::Float16);
    let legacy = x.layernorm(-1, layer.eps).unwrap().try_mul(&weight).unwrap().try_add(&bias).unwrap();

    assert_rounds_once(layer.forward(&x).unwrap(), reference, legacy, "affine epilogue");
}

#[test]
fn fp16_linear_keeps_bias_epilogue_in_fp32_until_final_cast() {
    let x = fp16(&[0.3333, -0.1428, 0.0909, 0.0769], &[1, 4]);
    let weight = fp16(&[3.0, -5.0, 7.0, -11.0, -2.0, 4.0, -6.0, 8.0], &[2, 4]);
    let bias = fp16(&[-2.5, 2.0], &[2]);

    let reference = x
        .linear()
        .weight(&weight)
        .dtype(DType::Float32)
        .call()
        .unwrap()
        .try_add(bias.cast(DType::Float32))
        .unwrap()
        .cast(DType::Float16);
    let legacy = x.linear().weight(&weight).bias(&bias).call().unwrap();

    let layer = Linear::new(weight, Some(bias));
    assert_rounds_once(linear_forward(&layer, &x).unwrap(), reference, legacy, "bias addition");
}

/// The bias-less projection (Whisper's key) must stay on the plain path.
#[test]
fn biasless_linear_matches_the_plain_projection() {
    let x = fp16(&[0.3333, -0.1428, 0.0909, 0.0769], &[1, 4]);
    let weight = fp16(&[3.0, -5.0, 7.0, -11.0, -2.0, 4.0, -6.0, 8.0], &[2, 4]);
    let layer = Linear::new(weight.clone(), None);

    let expected = realized_f32(x.linear().weight(&weight).call().unwrap());
    assert_eq!(realized_f32(linear_forward(&layer, &x).unwrap()), expected);
}
