//! ModernBERT pre-norm encoder layer.
//!
//! Direct port of `FlexBertUnpadPreNormLayer` (padded path). Standard pre-norm:
//! ```text
//! h = x + attn(attn_norm(x))      // attn_norm is Identity for layer 0
//! y = h + mlp(mlp_norm(h))
//! ```
//!
//! `skip_first_prenorm = true` makes layer 0's `attn_norm` an identity (its
//! state-dict key is absent), so `attn_norm` is `Option` and `None` for layer 0.

use svod_tensor::Tensor;
use svod_tensor::nn::{LayerNorm, Module};

use super::attention::ModernBertAttention;
use super::error::Result;

use super::mlp::ModernBertGlu;
use super::norm::layer_norm;

#[derive(Clone, Module)]
pub struct EncoderLayer {
    /// `None` for layer 0 (`skip_first_prenorm`).
    pub attn_norm: Option<LayerNorm>,
    #[module(key = "attn")]
    pub attention: ModernBertAttention,
    pub mlp_norm: LayerNorm,
    pub mlp: ModernBertGlu,
}

impl EncoderLayer {
    pub fn empty(config: &super::ModernBertConfig, layer_id: usize) -> Self {
        let hidden = config.hidden_size;
        let eps = config.layer_norm_eps;
        let dtype = config.dtype.clone();
        let window = if config.is_global_layer(layer_id) { None } else { Some(config.local_window()) };
        Self {
            attn_norm: (layer_id != 0).then(|| LayerNorm::with_dims(hidden, false, eps, dtype.clone())),
            attention: ModernBertAttention::empty(
                hidden,
                config.num_attention_heads,
                config.head_dim(),
                window,
                dtype.clone(),
            ),
            mlp_norm: LayerNorm::with_dims(hidden, false, eps, dtype.clone()),
            mlp: ModernBertGlu::empty(hidden, config.intermediate_size, dtype),
        }
    }

    /// Forward. `x`: `(B, L, D)` → `(B, L, D)`.
    pub fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor), padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let normed = match &self.attn_norm {
            Some(ln) => layer_norm(x, ln)?,
            None => x.clone(),
        };
        let h = self.attention.forward(&normed, rope, padding_mask, Some(x))?;
        self.mlp.forward(&layer_norm(&h, &self.mlp_norm)?, Some(&h))
    }
}
