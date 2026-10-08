//! Qwen3 pre-norm decoder layer:
//! ```text
//! h = x + attn(input_layernorm(x))
//! y = h + mlp(post_attention_layernorm(h))
//! ```
//!
//! Each residual add rides the epilogue of the GEMM that produces its addend
//! (`o_proj`, `down_proj`).

use svod_tensor::Tensor;
use svod_tensor::nn::{Module, RmsNorm};
use svod_tk3::ops;

use super::attention::Qwen3Attention;
use super::error::Result;
use super::feed_forward::Qwen3MLP;

pub(crate) fn rms_norm(norm: &RmsNorm, x: &Tensor) -> Result<Tensor> {
    Ok(ops::rms_norm(x, &norm.weight, norm.eps)?)
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
        self.forward_packed(x, rope, None)
    }

    /// [`Self::forward`] over packed rows' segment table `seg_start` (see
    /// [`super::Packing`]).
    pub(crate) fn forward_packed(
        &self,
        x: &Tensor,
        rope: &(Tensor, Tensor),
        seg_start: Option<&Tensor>,
    ) -> Result<Tensor> {
        let x_norm = rms_norm(&self.input_layernorm, x)?;
        let h = self.attention.forward_packed(&x_norm, Some(x), rope, seg_start)?;
        let h_norm = rms_norm(&self.post_attention_layernorm, &h)?;
        self.mlp.forward_into(&h_norm, Some(&h))
    }
}
