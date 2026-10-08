//! ModernBERT gated feed-forward (SwiGLU-style): `Linear(D, 2I)` → split
//! `[input | gate]` → `GELU(input) * gate` → `Linear(I, D)`. No biases.
//!
//! Mirrors `FlexBertGLU`. `Wi` output dim is `2*intermediate` (input and gate
//! branches concatenated along the last axis); `Wo` takes the gated `I`-dim
//! result back to `D`.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;
use svod_tk3::ops::{self, Act};

use crate::init::fan_in_uniform;

use super::error::Result;

#[derive(Clone, Module)]
pub struct ModernBertGlu {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[module(key = "Wi.weight")]
    pub wi_weight: Tensor,
    #[module(key = "Wo.weight")]
    pub wo_weight: Tensor,
}

impl ModernBertGlu {
    pub fn empty(hidden_size: usize, intermediate_size: usize, dtype: DType) -> Self {
        let wi_weight = fan_in_uniform(&[2 * intermediate_size, hidden_size], hidden_size, dtype.clone());
        let wo_weight = fan_in_uniform(&[hidden_size, intermediate_size], intermediate_size, dtype);
        Self { hidden_size, intermediate_size, wi_weight, wo_weight }
    }

    /// Forward. `x`: `(B, L, D)` → `(B, L, D)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // `Wi`'s `[input | gate]` rows are the op layer's gated order: the
        // first half takes the exact (erf) GELU, PyTorch's nn.GELU default.
        let gated = ops::linear(x, &self.wi_weight, ops::Linear { act: Act::Gelu, gated: true, ..Default::default() })?;
        Ok(ops::linear(&gated, &self.wo_weight, ops::Linear::default())?)
    }
}
