//! XLM-RoBERTa self-attention with separate biased Q/K/V/O projections.
//!
//! Standard BERT-style multi-head attention: separate `Linear(D, D, bias)` for
//! Q, K, V; reshape to `(B, H, L, hd)`; scaled dot-product attention (global,
//! no window); output `Linear(D, D, bias)`. No RoPE, no relative positions.
//!
//! State-dict keys match the published `BAAI/bge-m3` `pytorch_model.bin`:
//! `attention.self.{query,key,value}.{weight,bias}`,
//! `attention.output.dense.{weight,bias}`.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;
use svod_tk3::ops;

use crate::init::{fan_in_uniform, zeros};

use super::error::Result;

#[derive(Clone, Module)]
pub struct XlmRobertaAttention {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    #[module(key = "self.query.weight")]
    pub query_weight: Tensor,
    #[module(key = "self.query.bias")]
    pub query_bias: Tensor,
    #[module(key = "self.key.weight")]
    pub key_weight: Tensor,
    #[module(key = "self.key.bias")]
    pub key_bias: Tensor,
    #[module(key = "self.value.weight")]
    pub value_weight: Tensor,
    #[module(key = "self.value.bias")]
    pub value_bias: Tensor,
    #[module(key = "output.dense.weight")]
    pub out_weight: Tensor,
    #[module(key = "output.dense.bias")]
    pub out_bias: Tensor,
}

impl XlmRobertaAttention {
    pub fn empty(hidden_size: usize, num_heads: usize, head_dim: usize, dtype: DType) -> Self {
        Self {
            hidden_size,
            num_heads,
            head_dim,
            query_weight: fan_in_uniform(&[hidden_size, hidden_size], hidden_size, dtype.clone()),
            query_bias: zeros(&[hidden_size], dtype.clone()),
            key_weight: fan_in_uniform(&[hidden_size, hidden_size], hidden_size, dtype.clone()),
            key_bias: zeros(&[hidden_size], dtype.clone()),
            value_weight: fan_in_uniform(&[hidden_size, hidden_size], hidden_size, dtype.clone()),
            value_bias: zeros(&[hidden_size], dtype.clone()),
            out_weight: fan_in_uniform(&[hidden_size, hidden_size], hidden_size, dtype.clone()),
            out_bias: zeros(&[hidden_size], dtype),
        }
    }

    /// Forward. `x`: `(B, L, D)`. Returns `(B, L, D)`.
    /// `padding_mask`: optional bool `(B, L)` where `true` = real token,
    /// `false` = padding.
    pub fn forward(&self, x: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let project = |w: &Tensor, b: &Tensor| -> Result<Tensor> {
            Ok(ops::linear(x, w, ops::Linear { bias: Some(b), ..ops::Linear::default() })?
                .split_heads(self.num_heads)?)
        };
        let q = project(&self.query_weight, &self.query_bias)?;
        let k = project(&self.key_weight, &self.key_bias)?;
        let v = project(&self.value_weight, &self.value_bias)?;

        // SDPA, not the op layer: it takes key lengths, not a bool padding mask.
        let attn = q.scaled_dot_product_attention().key(&k).value(&v).maybe_key_padding_mask(padding_mask).call()?;

        let opts = ops::Linear { bias: Some(&self.out_bias), ..ops::Linear::default() };
        Ok(ops::linear(&attn.merge_heads()?, &self.out_weight, opts)?)
    }
}
