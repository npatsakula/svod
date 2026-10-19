//! XLM-RoBERTa feed-forward block: `Linear(D, I, bias) → GELU(exact) → Linear(I, D, bias)`.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;
use svod_tk3::ops::{self, Act};

use crate::init::{fan_in_uniform, zeros};

use super::error::Result;

#[derive(Clone, Module)]
pub struct FeedForwardWeights {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[module(key = "intermediate.dense.weight")]
    pub intermediate_weight: Tensor,
    #[module(key = "intermediate.dense.bias")]
    pub intermediate_bias: Tensor,
    #[module(key = "output.dense.weight")]
    pub output_weight: Tensor,
    #[module(key = "output.dense.bias")]
    pub output_bias: Tensor,
}

impl FeedForwardWeights {
    pub fn empty(hidden_size: usize, intermediate_size: usize, dtype: DType) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            intermediate_weight: fan_in_uniform(&[intermediate_size, hidden_size], hidden_size, dtype.clone()),
            intermediate_bias: zeros(&[intermediate_size], dtype.clone()),
            output_weight: fan_in_uniform(&[hidden_size, intermediate_size], intermediate_size, dtype.clone()),
            output_bias: zeros(&[hidden_size], dtype),
        }
    }

    /// Forward. `x`: `(B, L, D)` → `(B, L, D)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let opts = ops::Linear { bias: Some(&self.intermediate_bias), act: Act::Gelu, ..ops::Linear::default() };
        let y = ops::linear(x, &self.intermediate_weight, opts)?;
        let opts = ops::Linear { bias: Some(&self.output_bias), ..ops::Linear::default() };
        Ok(ops::linear(&y, &self.output_weight, opts)?)
    }
}
