//! Multi-head attention: self-attention (encoder/decoder) and cross-attention (decoder).
//!
//! Matches `whisper.model.MultiHeadAttention`. Key projection has no bias;
//! query, value, and output projections have bias.
//!
//! Attention scaling: Whisper pre-scales Q and K by `d_head^{-0.25}`, which
//! equals `d_head^{-0.5}` on the scores — identical to SDPA's default
//! `1/sqrt(d_head)`.  So we use the SDPA default scale.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Linear, Module};
use svod_tk3::ops::{self, Act, Attn};

use crate::init::{Bias, linear};
use crate::state::scoped;

use super::blocks::{linear_forward, project};
use super::error::Result;

#[derive(Clone, Module)]
pub struct MultiHeadAttention {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub out: Linear,
    pub n_head: usize,
}

impl MultiHeadAttention {
    pub fn empty(n_state: usize, n_head: usize) -> Self {
        Self::empty_dtype(n_state, n_head, DType::Float32)
    }

    pub fn empty_dtype(n_state: usize, n_head: usize, dtype: DType) -> Self {
        Self {
            query: linear(n_state, n_state, Bias::FanIn, dtype.clone()),
            key: linear(n_state, n_state, Bias::None, dtype.clone()),
            value: linear(n_state, n_state, Bias::FanIn, dtype.clone()),
            out: linear(n_state, n_state, Bias::FanIn, dtype),
            n_head,
        }
    }

    /// Forward pass. `xa = None` for self-attention, `Some(enc)` for cross-attention.
    /// `mask` is the causal mask for decoder self-attention (additive float mask).
    pub fn forward(&self, x: &Tensor, xa: Option<&Tensor>, mask: Option<&Tensor>) -> Result<Tensor> {
        Ok(self.forward_return_kv(x, xa, mask)?.0)
    }

    /// Bidirectional self-attention.
    pub(crate) fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let encoder = |layer: &Linear, x: &Tensor| project(layer, x, Act::None);
        let (q, k, v) = self.qkv(x, x, encoder)?;
        let out = self.attend(&q, &k, &v, Attn::default())?;
        scoped("out", || encoder(&self.out, &out))
    }

    pub fn forward_return_kv(
        &self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (q, k, v) = self.qkv(x, xa.unwrap_or(x), linear_forward)?;
        let out = self.attend(&q, &k, &v, Attn { causal: mask.is_some(), ..Attn::default() })?;
        let out = scoped("out", || linear_forward(&self.out, &out))?;
        Ok((out, k, v))
    }

    /// The decoder's few-row projections keep [`linear_forward`]: the tile
    /// GEMM is slower than the graph's at that height.
    fn qkv(
        &self,
        x: &Tensor,
        kv_input: &Tensor,
        project: impl Fn(&Linear, &Tensor) -> Result<Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let q = scoped("query", || project(&self.query, x))?;
        let k = scoped("key", || project(&self.key, kv_input))?;
        let v = scoped("value", || project(&self.value, kv_input))?;
        Ok((q, k, v))
    }

    /// `[B, S, D]` projections split into sequence-major `[B, S, H, Dh]` heads.
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, opts: Attn) -> Result<Tensor> {
        let split = |t: &Tensor| -> Result<Tensor> {
            Ok(t.try_reshape([t.dim(0)?, t.dim(1)?, self.n_head.into(), (t.dim_const(2)? / self.n_head).into()])?)
        };
        let out = ops::attention(&split(q)?, &split(k)?, &split(v)?, opts)?;
        Ok(out.try_reshape([q.dim(0)?, q.dim(1)?, q.dim_const(2)?.into()])?)
    }
}
