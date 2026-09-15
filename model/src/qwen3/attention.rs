//! Qwen3 attention: GQA with per-head Q/K RMSNorm before RoPE, causal, no
//! projection biases.
//!
//! The checkpoint's `q_proj`/`k_proj`/`v_proj` read the same input, so they are
//! stacked into one `[(H + 2·Hkv)·Dh, D]` weight at load and split after a
//! single GEMM; the state dict keeps the published three-key layout. Heads are
//! kept sequence-major, `[B, L, H, Dh]`: the layout the hand flash-attention
//! kernel consumes and returns, so the head split and merge are plain
//! reshapes. The SDPA fallback permutes to head-major and back.

use snafu::ResultExt;
use svod_dtype::ScalarDType;
use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, Module, RmsNorm, StateDict, get_tensor, prefixed};

use crate::init::fan_in_uniform;

use super::error::{Result, TkSnafu};
use super::linear::{Projected, linear, linear_add};
use super::tk;

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

/// Causal grouped-query attention over sequence-major `[B, L, H, Dh]`: the
/// hand kernel when it applies (16-bit operands on a supported device at a
/// tiling shape), else SDPA. Unmasked: padding is on the right, behind the
/// causal edge.
pub(crate) fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    if matches!(q.dtype().base(), ScalarDType::Float16 | ScalarDType::BFloat16)
        && let Some(out) =
            svod_tk::flash_attention_with(q, k, v, svod_tk::FaOpts { causal: true, key_lens: None }).context(TkSnafu)?
    {
        return Ok(out);
    }
    let head_major = |t: &Tensor| t.try_permute(&[0, 2, 1, 3]);
    let out = head_major(q)?
        .scaled_dot_product_attention()
        .key(&head_major(k)?)
        .value(&head_major(v)?)
        .is_causal(true)
        .enable_gqa(true)
        .call()?;
    Ok(head_major(&out)?)
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

    /// `x`: `(B, L, D)` → `(B, L, D)`. `rope`: sequence-major `(cos, sin)`
    /// `[1, L, 1, Dh/2]`.
    pub fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor)) -> Result<Tensor> {
        Ok(self.forward_into(x, rope, None)?.into_tensor())
    }

    /// [`Self::forward`] with `residual` folded into the `o_proj` GEMM's
    /// epilogue when that kernel takes it (see [`linear_add`]).
    pub(crate) fn forward_into(
        &self,
        x: &Tensor,
        rope: &(Tensor, Tensor),
        residual: Option<&Tensor>,
    ) -> Result<Projected> {
        let (b, l) = (x.dim(0)?, x.dim(1)?);
        let qkv = linear(x, &self.qkv_weight)?;
        let (q, k, v) = self.prologue(&qkv, rope, (&b, &l))?;
        let attn = causal_attention(&q, &k, &v)?.try_reshape([b, l, SInt::Const(self.num_heads * self.head_dim)])?;
        linear_add(&attn, &self.o_proj_weight, residual)
    }

    /// The fused GEMM output `[B, L, (H + 2·Hkv)·Dh]` → the three sequence-major
    /// head tensors attention consumes. The hand kernel reads the row once and
    /// writes `q`/`k` (normed over `Dh`, then rotated) and `v` (copied); off its
    /// geometry or arch it declines and the split / head view / norm / rope graph
    /// runs instead, recomputing the strided views five times over.
    fn prologue(
        &self,
        qkv: &Tensor,
        rope: &(Tensor, Tensor),
        (b, l): (&SInt, &SInt),
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (cos, sin) = rope;
        let geom = tk::Heads { h: self.num_heads, h_kv: self.num_kv_heads, dh: self.head_dim };
        let concrete = qkv.shape()?.iter().all(|d| d.as_const().is_some());
        if matches!(qkv.dtype().base(), ScalarDType::Float16 | ScalarDType::BFloat16)
            && concrete
            && self.q_norm.eps == self.k_norm.eps
            && let Some(out) =
                tk::qkv_norm_rope(qkv, &self.q_norm.weight, &self.k_norm.weight, cos, sin, self.q_norm.eps, geom)
                    .context(TkSnafu)?
        {
            return Ok(out);
        }
        let parts = qkv.split(&self.qkv_rows(), -1)?;
        let heads = |p: &Tensor, h: usize| -> Result<Tensor> {
            Ok(p.try_reshape([b.clone(), l.clone(), SInt::Const(h), SInt::Const(self.head_dim)])?)
        };
        let q = self.q_norm.forward(&heads(&parts[0], self.num_heads)?)?.apply_rotary_emb(cos, sin, false)?;
        let k = self.k_norm.forward(&heads(&parts[1], self.num_kv_heads)?)?.apply_rotary_emb(cos, sin, false)?;
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
