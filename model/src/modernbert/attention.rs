//! ModernBERT multi-head attention with fused QKV and RoPE.
//!
//! Direct port of `FlexBertUnpadRopeAttention` (padded path): one fused
//! `Wqkv: Linear(D, 3D)` (no bias), RoPE applied to Q and K, scaled
//! dot-product attention, then `Wo: Linear(D, D)` (no bias).
//!
//! The fused QKV output along dim -1 is `[Q(H*hd) | K(H*hd) | V(H*hd)]`; the
//! heads op splits it into rotated sequence-major `(B, L, H, hd)` heads.
//! Sliding-window local layers pass a `window`; global layers pass `None`.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;
use svod_tk3::ops::{self, Attn, KeyMask, Qkv};

use crate::init::fan_in_uniform;

use super::error::Result;

#[derive(Clone, Module)]
pub struct ModernBertAttention {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// `None` for global layers; `Some((left, right))` for local layers.
    pub window: Option<(usize, usize)>,
    #[module(key = "Wqkv.weight")]
    pub qkv_weight: Tensor,
    #[module(key = "Wo.weight")]
    pub out_weight: Tensor,
}

impl ModernBertAttention {
    pub fn empty(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        window: Option<(usize, usize)>,
        dtype: DType,
    ) -> Self {
        let qkv_weight = fan_in_uniform(&[3 * hidden_size, hidden_size], hidden_size, dtype.clone());
        let out_weight = fan_in_uniform(&[hidden_size, hidden_size], hidden_size, dtype);
        Self { hidden_size, num_heads, head_dim, window, qkv_weight, out_weight }
    }

    /// Forward. `x`: `(B, L, D)`. Returns `residual + attn(x)`, `(B, L, D)`,
    /// the add in `Wo`'s epilogue. `rope`: the per-layer `(cos, sin)` table,
    /// `(1, L, 1, hd / 2)`.
    /// `padding_mask`: optional bool `(B, L)` where `true` = real token,
    /// `false` = padding.
    pub fn forward(
        &self,
        x: &Tensor,
        residual: &Tensor,
        rope: &(Tensor, Tensor),
        padding_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, l, d) = (x.dim(0)?, x.dim(1)?, self.hidden_size);
        let (cos, sin) = rope;
        let qkv = ops::linear(x, &self.qkv_weight, ops::Linear::default())?;
        let split = Qkv {
            heads: self.num_heads,
            kv_heads: self.num_heads,
            head_dim: self.head_dim,
            q_norm: None,
            k_norm: None,
            eps: 0.0,
            rope: Some((cos, sin)),
        };
        let (q, k, v) = ops::heads(&qkv, split)?;
        let keys = padding_mask.map_or(KeyMask::None, KeyMask::Bool);
        let attn = ops::attention(&q, &k, &v, Attn { keys, window: self.window, ..Attn::default() })?;

        let opts = ops::Linear { residual: Some(residual), ..Default::default() };
        Ok(ops::linear(&attn.try_reshape([b, l, d.into()])?, &self.out_weight, opts)?)
    }
}
