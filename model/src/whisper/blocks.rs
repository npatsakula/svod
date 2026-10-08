//! Whisper building blocks: the mixed-precision linear epilogue and the
//! sinusoidal positional embedding. The layer constructors live in
//! `init`.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Linear;
use svod_tk3::ops::{self, Act};

use super::error::Result;

/// `act(x·wᵀ + b)` through the tile op layer, whose kernel keeps the
/// accumulator, bias and activation in f32 until one final rounding. A scaled
/// or fp8 weight takes [`linear_forward`].
pub(crate) fn project(layer: &Linear, x: &Tensor, act: Act) -> Result<Tensor> {
    if layer.weight_scale.is_none() && layer.weight.dtype() == x.dtype() {
        let opts = ops::Linear { bias: layer.bias.as_ref(), act, ..ops::Linear::default() };
        return Ok(ops::linear(x, &layer.weight, opts)?);
    }
    let y = linear_forward(layer, x)?;
    Ok(match act {
        Act::None => y,
        Act::Gelu => y.gelu_exact()?,
        Act::Silu => y.silu()?,
    })
}

/// Whisper's linear forward. OpenAI keeps the matmul accumulator *and* the bias
/// addition in FP32 when activation and weight are both half precision, so the
/// result rounds exactly once, at the final cast. [`svod_tensor::nn::Layer`]'s
/// `forward` has no accumulator-dtype knob, so this stays a free function. An
/// fp8 weight takes the same path: its per-channel scale multiplies the f32
/// accumulator, so the reduce reads the fp8 weight directly and the decode
/// matvec keeps its fast path.
pub(crate) fn linear_forward(layer: &Linear, x: &Tensor) -> Result<Tensor> {
    let half = |dtype: &DType| *dtype == DType::Float16 || *dtype == DType::BFloat16;
    let output_dtype = x.dtype();
    let weight_dtype = layer.weight.dtype();
    if !(half(&output_dtype) && (half(&weight_dtype) || weight_dtype.is_fp8())) {
        return Ok(svod_tensor::nn::Layer::forward(layer, x)?);
    }
    let product = layer.apply_weight_scale(x.linear().weight(&layer.weight).dtype(DType::Float32).call()?)?;
    let sum = match &layer.bias {
        Some(bias) => product.try_add(bias.cast(DType::Float32))?,
        None => product,
    };
    Ok(sum.cast(output_dtype))
}

/// Sinusoidal positional embeddings matching `whisper.model.sinusoids()`:
/// a `[length, channels]` f32 constant, built in-graph.
pub fn sinusoids(length: usize, channels: usize, max_timescale: f64) -> Result<Tensor> {
    assert!(channels.is_multiple_of(2), "sinusoids require even channel count");
    let half = channels / 2;
    let log_increment = max_timescale.ln() / (half - 1) as f64;
    let inv = Tensor::arange(0, Some(half as i64), None)?.cast(DType::Float32).try_mul(-log_increment)?.try_exp()?;
    let scaled_time =
        Tensor::arange(0, Some(length as i64), None)?.cast(DType::Float32).try_unsqueeze(-1)?.try_mul(&inv)?;
    Ok(Tensor::cat(&[&scaled_time.sin()?, &scaled_time.cos()?], -1)?)
}
