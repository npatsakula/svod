//! Qwen3 pre-norm decoder layer:
//! ```text
//! h = x + attn(input_layernorm(x))
//! y = h + mlp(post_attention_layernorm(h))
//! ```
//!
//! Each residual add feeds exactly one norm, so the addend travels beside the
//! stream until [`svod_tk3::ops::add_rms_norm`] writes the sum and its norm
//! in one pass. The GEMM epilogue cannot take it: with the norm on the graph
//! path (a width without a norm kernel, a weight off the stream's dtype) a
//! launch whose residual operand the norm also reads fails kernel-graph
//! verification.

use svod_tensor::Tensor;
use svod_tensor::nn::{Module, RmsNorm};
use svod_tk3::ops;

use super::attention::Qwen3Attention;
use super::error::Result;
use super::feed_forward::Qwen3MLP;

/// `(x + delta, rms_norm(x + delta))`.
pub(crate) fn add_norm(norm: &RmsNorm, delta: Option<&Tensor>, x: &Tensor) -> Result<(Tensor, Tensor)> {
    Ok(match delta {
        Some(delta) => ops::add_rms_norm(delta, x, &norm.weight, norm.eps)?,
        None => (x.clone(), ops::rms_norm(x, &norm.weight, norm.eps)?),
    })
}

#[derive(Clone, Module)]
pub struct Qwen3DecoderLayer {
    pub input_layernorm: RmsNorm,
    #[module(key = "self_attn")]
    pub attention: Qwen3Attention,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: Qwen3MLP,
}

impl Qwen3DecoderLayer {
    pub fn empty(config: &super::Qwen3Config) -> Self {
        let dtype = config.dtype.clone();
        Self {
            input_layernorm: RmsNorm::with_dims(config.hidden_size, config.rms_norm_eps, dtype.clone()),
            attention: Qwen3Attention::empty(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps,
                dtype.clone(),
            ),
            post_attention_layernorm: RmsNorm::with_dims(config.hidden_size, config.rms_norm_eps, dtype.clone()),
            mlp: Qwen3MLP::empty(config.hidden_size, config.intermediate_size, dtype),
        }
    }

    pub fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor)) -> Result<Tensor> {
        let (h, mlp) = self.forward_unsummed(x, None, rope, None)?;
        Ok(h.try_add(&mlp)?)
    }

    /// The layer over the stream `x + delta`, returning the stream and the mlp
    /// output still to be added to it. `seg_start` is the packed rows' segment
    /// table (see [`super::Packing`]).
    pub(crate) fn forward_unsummed(
        &self,
        x: &Tensor,
        delta: Option<&Tensor>,
        rope: &(Tensor, Tensor),
        seg_start: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (x, x_norm) = add_norm(&self.input_layernorm, delta, x)?;
        let attn = self.attention.forward_packed(&x_norm, rope, seg_start)?;
        let (h, h_norm) = add_norm(&self.post_attention_layernorm, Some(&attn), &x)?;
        Ok((h, self.mlp.forward(&h_norm)?))
    }
}
