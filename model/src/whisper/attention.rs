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

use crate::init::{Bias, linear};
use crate::state::scoped;

use super::blocks::linear_forward;
use super::error::Result;

const MIN_PADDED_FA_SEQUENCE: usize = 1024;
const MAX_PADDED_FA_OVERHEAD_DIVISOR: usize = 16;

/// Returns the padded sequence length only when padding is both necessary and
/// sufficiently small for encoder-style self-attention.
pub(crate) fn padded_fa_sequence_len(causal: bool, q_len: usize, k_len: usize, v_len: usize) -> Option<usize> {
    let multiple = svod_tk::FLASH_ATTENTION_SEQUENCE_MULTIPLE;
    if causal || q_len != k_len || q_len != v_len || q_len < MIN_PADDED_FA_SEQUENCE || q_len.is_multiple_of(multiple) {
        return None;
    }
    let padded = q_len.next_multiple_of(multiple);
    (padded - q_len <= q_len / MAX_PADDED_FA_OVERHEAD_DIVISOR).then_some(padded)
}

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
        self.forward_with_key_lens(x, xa, mask, None)
    }

    pub(crate) fn forward_with_key_lens(
        &self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        key_lens: Option<&Tensor>,
    ) -> Result<Tensor> {
        let q = scoped("query", || linear_forward(&self.query, x))?;
        let kv_input = xa.unwrap_or(x);
        let k = scoped("key", || linear_forward(&self.key, kv_input))?;
        let v = scoped("value", || linear_forward(&self.value, kv_input))?;

        let out = self.fa_attention(&q, &k, &v, mask.is_some(), key_lens)?;
        scoped("out", || linear_forward(&self.out, &out))
    }

    pub fn forward_return_kv(
        &self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let q = scoped("query", || linear_forward(&self.query, x))?;
        let kv_input = xa.unwrap_or(x);
        let k = scoped("key", || linear_forward(&self.key, kv_input))?;
        let v = scoped("value", || linear_forward(&self.value, kv_input))?;

        let out = self.fa_attention(&q, &k, &v, mask.is_some(), None)?;
        let out = scoped("out", || linear_forward(&self.out, &out))?;
        Ok((out, k, v))
    }

    /// Q/K/V in [B, S, D] → split to [B, S, H, Dh], the layout of
    /// [`crate::attention::attend`]. `causal` controls the mask.
    fn fa_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
        key_lens: Option<&Tensor>,
    ) -> Result<Tensor> {
        let b = q.dim(0)?;
        let s = q.dim(1)?;
        let d = q.dim_const(2)?;

        // Split each to [B, S, H, Dh] — FA's layout (seq-major, no permute).
        // `split_heads` is the same split followed by the permute SDPA wants.
        let split = |t: &Tensor| -> Result<Tensor> {
            Ok(t.try_reshape([t.dim(0)?, t.dim(1)?, self.n_head.into(), (t.dim_const(2)? / self.n_head).into()])?)
        };
        let (q_fa, k_fa, v_fa) = (split(q)?, split(k)?, split(v)?);

        let out = crate::attention::attend(&q_fa, &k_fa, &v_fa, causal, key_lens)?;
        Ok(out.try_reshape([b, s, d.into()])?)
    }
}
