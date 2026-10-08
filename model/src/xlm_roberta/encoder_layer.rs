//! XLM-RoBERTa post-norm encoder layer.
//!
//! Standard BERT/RoBERTa post-norm:
//! ```text
//! attn_out = LayerNorm_attn(x + Attention(x))
//! layer_out = LayerNorm_ffn(attn_out + FFN(attn_out))
//! ```

use svod_tensor::Tensor;
use svod_tensor::nn::{LayerNorm, Module};
use svod_tk3::ops;

use super::attention::XlmRobertaAttention;
use super::config::XlmRobertaConfig;
use super::error::Result;

use super::feed_forward::FeedForwardWeights;

#[derive(Clone, Module)]
pub struct EncoderLayer {
    pub attention: XlmRobertaAttention,
    #[module(key = "attention.output.LayerNorm")]
    pub attention_norm: LayerNorm,
    #[module(key = "")]
    pub feed_forward: FeedForwardWeights,
    #[module(key = "output.LayerNorm")]
    pub ffn_norm: LayerNorm,
}

impl EncoderLayer {
    pub fn empty(config: &XlmRobertaConfig) -> Self {
        let hidden = config.hidden_size;
        let eps = config.layer_norm_eps;
        let dtype = config.dtype.clone();
        Self {
            attention: XlmRobertaAttention::empty(hidden, config.num_attention_heads, config.head_dim(), dtype.clone()),
            attention_norm: LayerNorm::with_dims(hidden, false, eps, dtype.clone()),
            feed_forward: FeedForwardWeights::empty(hidden, config.intermediate_size, dtype.clone()),
            ffn_norm: LayerNorm::with_dims(hidden, false, eps, dtype),
        }
    }

    /// Forward. `x`: `(B, L, D)` → `(B, L, D)`. Post-norm.
    pub fn forward(&self, x: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let add_norm = |delta: &Tensor, x: &Tensor, ln: &LayerNorm| -> Result<Tensor> {
            Ok(ops::add_layer_norm(delta, x, &ln.weight, ln.bias.as_ref(), ln.eps)?.1)
        };
        let x = add_norm(&self.attention.forward(x, padding_mask)?, x, &self.attention_norm)?;
        add_norm(&self.feed_forward.forward(&x)?, &x, &self.ffn_norm)
    }
}
