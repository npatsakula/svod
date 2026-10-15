//! ModernBERT gated feed-forward (SwiGLU-style): `Linear(D, 2I)` → split
//! `[input | gate]` → `GELU(input) * gate` → `Linear(I, D)`. No biases.
//!
//! Mirrors `FlexBertGLU`. `Wi` output dim is `2*intermediate` (input and gate
//! branches concatenated along the last axis); `Wo` takes the gated `I`-dim
//! result back to `D`.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::init::fan_in_uniform;

use super::error::Result;
use super::linear::linear;

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

    /// Forward. `x`: `(B, L, D)` → `(B, L, D)`, plus `residual` when given (the
    /// add rides the output projection).
    pub fn forward(&self, x: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        // (., 2I) → [input (., I) | gate (., I)].
        let h = linear(x, &self.wi_weight, None)?;
        let i = self.intermediate_size;
        // GELU(input) * gate — exact (erf) GELU matches PyTorch's nn.GELU default.
        let gated = h.narrow(-1, 0usize, i)?.gelu_exact()?.try_mul(&h.narrow(-1, i, i)?)?;
        linear(&gated, &self.wo_weight, residual)
    }
}
