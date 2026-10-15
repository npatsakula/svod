//! XLM-RoBERTa self-attention with separate biased Q/K/V/O projections.
//!
//! Standard BERT-style multi-head attention: separate `Linear(D, D, bias)` for
//! Q, K, V; reshape to seq-major `(B, L, H, hd)`; global attention through
//! the shared `attend` — tk's flash attention on 16-bit activations whose
//! length tiles, SDPA otherwise; output `Linear(D, D, bias)`. No RoPE, no
//! relative positions.
//!
//! State-dict keys match the published `BAAI/bge-m3` `pytorch_model.bin`:
//! `attention.self.{query,key,value}.{weight,bias}`,
//! `attention.output.dense.{weight,bias}`.

use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::attention::{Keys, attend};
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
        let (b, l) = (x.dim(0)?, x.dim(1)?);
        let heads = [b.clone(), l.clone(), SInt::Const(self.num_heads), SInt::Const(self.head_dim)];
        let project = |w: &Tensor, bias: &Tensor| -> Result<Tensor> {
            Ok(x.linear().weight(w).bias(bias).call()?.try_reshape(heads.clone())?)
        };
        let q = project(&self.query_weight, &self.query_bias)?;
        let k = project(&self.key_weight, &self.key_bias)?;
        let v = project(&self.value_weight, &self.value_bias)?;

        let keys = padding_mask.map_or(Keys::All, Keys::Mask);
        let attn = attend(&q, &k, &v, false, keys)?.try_reshape([b, l, SInt::Const(self.hidden_size)])?;
        Ok(attn.linear().weight(&self.out_weight).bias(&self.out_bias).call()?)
    }
}
