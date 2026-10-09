//! ModernBERT transformer encoder: a stack of pre-norm [`EncoderLayer`]s with
//! per-layer RoPE. No internal embeddings or final norm — those live on the
//! root [`super::model::ModernBert`].
//!
//! Per-layer rotary tables are rebuilt on each forward (the sequence length is
//! concrete per call); global vs local theta is selected via
//! [`ModernBertConfig::rope_theta`].

use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use super::config::ModernBertConfig;
use super::encoder_layer::EncoderLayer;
use super::error::Result;

#[derive(Clone, Module)]
pub struct Encoder {
    #[module(skip)]
    pub config: ModernBertConfig,
    #[module(key = "")]
    pub layers: Vec<EncoderLayer>,
}

impl Encoder {
    pub fn empty(config: &ModernBertConfig) -> Self {
        let layers = (0..config.num_hidden_layers).map(|i| EncoderLayer::empty(config, i)).collect();
        Self { config: config.clone(), layers }
    }

    /// Run the encoder stack. `x`: `(B, L, D)` → `(B, L, D)`.
    /// `padding_mask`: optional bool `(B, L)` where `true` = real token,
    /// `false` = padding — the polarity SDPA's `key_padding_mask` wants.
    pub fn forward(&self, x: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let seq_len = x.dim_const(1)?;
        let head_dim = self.config.head_dim();
        let dtype = self.config.dtype.clone();

        // Two rotary bases (global / local) → two tables. Build them once
        // before the loop and select per layer; every global layer shares one,
        // every local layer the other.
        let table = |theta: f64| -> Result<(Tensor, Tensor)> {
            let (cos, sin) = Tensor::rope_table(theta, seq_len, head_dim, dtype.clone())?;
            let seq_major = |t: Tensor| t.try_reshape([1, seq_len, 1, head_dim / 2]);
            Ok((seq_major(cos)?, seq_major(sin)?))
        };
        let (global, local) = (table(self.config.global_rope_theta)?, table(self.config.local_rope_theta)?);

        let mut h = x.clone();
        for (layer_id, layer) in self.layers.iter().enumerate() {
            let rope = if self.config.is_global_layer(layer_id) { &global } else { &local };
            h = layer.forward(&h, rope, padding_mask)?;
        }
        Ok(h)
    }
}
