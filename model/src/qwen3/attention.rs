//! Qwen3 attention: GQA with per-head Q/K RMSNorm before RoPE, causal, no
//! projection biases.
//!
//! The checkpoint's `q_proj`/`k_proj`/`v_proj` read the same input, so they are
//! stacked into one `[(H + 2·Hkv)·Dh, D]` weight at load and split after a
//! single GEMM; the state dict keeps the published three-key layout. Heads are
//! kept sequence-major, `[B, L, H, Dh]`: the layout the op layer's heads and
//! attention ops produce and consume, so the head merge is a plain reshape.

use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, Module, RmsNorm, StateDict, get_tensor, prefixed};
use svod_tk3::ops::{self, Attn, Qkv};

use crate::init::fan_in_uniform;

use super::error::Result;

#[derive(Clone)]
pub struct Qwen3Attention {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// `q_proj.weight` over `k_proj.weight` over `v_proj.weight`.
    pub qkv_weight: Tensor,
    pub o_proj_weight: Tensor,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
}

impl Qwen3Attention {
    pub fn empty(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        eps: f64,
        dtype: svod_dtype::DType,
    ) -> Self {
        let qkv_rows = (num_heads + 2 * num_kv_heads) * head_dim;
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            qkv_weight: fan_in_uniform(&[qkv_rows, hidden_size], hidden_size, dtype.clone()),
            o_proj_weight: fan_in_uniform(&[hidden_size, num_heads * head_dim], num_heads * head_dim, dtype.clone()),
            q_norm: RmsNorm::with_dims(head_dim, eps, dtype.clone()),
            k_norm: RmsNorm::with_dims(head_dim, eps, dtype),
        }
    }

    /// Row extents of `qkv_weight`: `[q, k, v]`.
    fn qkv_rows(&self) -> [usize; 3] {
        let kv = self.num_kv_heads * self.head_dim;
        [self.num_heads * self.head_dim, kv, kv]
    }

    /// `x`: `(B, L, D)` → `(B, L, D)`. `rope`: sequence-major `(cos, sin)`,
    /// `[1, L, 1, Dh/2]` by position or `[B, L, 1, Dh/2]` by token.
    pub fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor)) -> Result<Tensor> {
        self.forward_packed(x, None, rope, None)
    }

    /// [`Self::forward`] over packed rows: `seg_start` `[B, L]` is the row
    /// index each token's sequence starts at (see [`super::Packing`]), hiding
    /// the sequences packed before it. Padding is on the right, behind the
    /// causal edge, so it needs no mask. `residual` is added in the `o_proj`
    /// GEMM's epilogue.
    pub(crate) fn forward_packed(
        &self,
        x: &Tensor,
        residual: Option<&Tensor>,
        rope: &(Tensor, Tensor),
        seg_start: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, l) = (x.dim(0)?, x.dim(1)?);
        let qkv = ops::linear(x, &self.qkv_weight, ops::Linear::default())?;
        let (q, k, v) = self.prologue(&qkv, rope)?;
        let attn = ops::attention(&q, &k, &v, Attn { causal: true, seg_start, ..Attn::default() })?;
        let attn = attn.try_reshape([b, l, SInt::Const(self.num_heads * self.head_dim)])?;
        Ok(ops::linear(&attn, &self.o_proj_weight, ops::Linear { residual, ..Default::default() })?)
    }

    /// The fused GEMM output `[B, L, (H + 2·Hkv)·Dh]` → the three sequence-major
    /// head tensors attention consumes: `q`/`k` normed over `Dh`, then rotated.
    ///
    /// `ops::heads` shares one `eps` and rejects norm weights or rope tables
    /// off `qkv`'s dtype (which the graph accepts), so those take the split /
    /// norm / rope graph here.
    fn prologue(&self, qkv: &Tensor, (cos, sin): &(Tensor, Tensor)) -> Result<(Tensor, Tensor, Tensor)> {
        let (q_norm, k_norm) = (&self.q_norm, &self.k_norm);
        let one_dtype = [&q_norm.weight, &k_norm.weight, cos, sin].iter().all(|t| t.dtype() == qkv.dtype());
        if one_dtype && q_norm.eps == k_norm.eps {
            let split = Qkv {
                heads: self.num_heads,
                kv_heads: self.num_kv_heads,
                head_dim: self.head_dim,
                q_norm: Some(&q_norm.weight),
                k_norm: Some(&k_norm.weight),
                eps: q_norm.eps,
                rope: Some((cos, sin)),
            };
            return Ok(ops::heads(qkv, split)?);
        }
        let (b, l) = (qkv.dim(0)?, qkv.dim(1)?);
        let parts = qkv.split(&self.qkv_rows(), -1)?;
        let heads = |p: &Tensor, h: usize| -> Result<Tensor> {
            Ok(p.try_reshape([b.clone(), l.clone(), SInt::Const(h), SInt::Const(self.head_dim)])?)
        };
        let q = q_norm.forward(&heads(&parts[0], self.num_heads)?)?.apply_rotary_emb(cos, sin, false)?;
        let k = k_norm.forward(&heads(&parts[1], self.num_kv_heads)?)?.apply_rotary_emb(cos, sin, false)?;
        Ok((q, k, heads(&parts[2], self.num_kv_heads)?))
    }
}

impl Module for Qwen3Attention {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        let mut start = 0;
        for (name, rows) in ["q_proj.weight", "k_proj.weight", "v_proj.weight"].iter().zip(self.qkv_rows()) {
            out.insert(prefixed(prefix, name), self.qkv_weight.narrow(0, start, rows).expect("stacked qkv weight"));
            start += rows;
        }
        out.insert(prefixed(prefix, "o_proj.weight"), self.o_proj_weight.clone());
        self.q_norm.write_state(&prefixed(prefix, "q_norm"), out);
        self.k_norm.write_state(&prefixed(prefix, "k_norm"), out);
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> svod_tensor::error::Result<()> {
        let q = get_tensor(sd, &prefixed(prefix, "q_proj.weight"))?;
        let k = get_tensor(sd, &prefixed(prefix, "k_proj.weight"))?;
        let v = get_tensor(sd, &prefixed(prefix, "v_proj.weight"))?;
        // One buffer, as in `Qwen3MLP`: a lazy `cat` is re-read per part in
        // the GEMM's K loop.
        self.qkv_weight = Tensor::cat(&[&q, &k, &v], 0)?.contiguous();
        self.qkv_weight.realize()?;
        self.o_proj_weight = get_tensor(sd, &prefixed(prefix, "o_proj.weight"))?;
        self.q_norm.load_state_dict(sd, &prefixed(prefix, "q_norm"))?;
        self.k_norm.load_state_dict(sd, &prefixed(prefix, "k_norm"))
    }
}
