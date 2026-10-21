//! ModernBERT token embeddings (`FlexBertSansPositionEmbeddings`): a plain
//! `nn.Embedding` lookup followed by a LayerNorm. There are **no position
//! embeddings** — position information enters via RoPE in attention.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Embedding, Layer, LayerNorm, Module};
use svod_tk::ops;

use crate::init::embedding;

use super::error::Result;

#[derive(Clone, Module)]
pub struct Embeddings {
    pub tok_embeddings: Embedding,
    pub norm: LayerNorm,
}

impl Embeddings {
    pub fn empty(vocab_size: usize, hidden_size: usize, eps: f64, dtype: DType) -> Self {
        let tok_embeddings = embedding(vocab_size, hidden_size, dtype.clone());
        Self { tok_embeddings, norm: LayerNorm::with_dims(hidden_size, false, eps, dtype) }
    }

    /// Forward. `input_ids`: `(B, L)` int64 → `(B, L, D)`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let x = self.tok_embeddings.forward(input_ids)?;
        Ok(ops::layer_norm(&x, &self.norm.weight, self.norm.bias.as_ref(), self.norm.eps)?)
    }
}
