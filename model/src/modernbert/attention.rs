//! ModernBERT multi-head attention with fused QKV and RoPE.
//!
//! Direct port of `FlexBertUnpadRopeAttention` (padded path): one fused
//! `Wqkv: Linear(D, 3D)` (no bias), RoPE applied to Q and K, scaled
//! dot-product attention, then `Wo: Linear(D, D)` (no bias).
//!
//! The fused QKV output along dim -1 is `[Q(H*hd) | K(H*hd) | V(H*hd)]`; each
//! third becomes `(B, H, L, hd)` for RoPE + SDPA. Sliding-window local layers
//! pass a `window`; global layers pass `None`.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;
use svod_tk3::ops;

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

    /// Forward. `x`: `(B, L, D)`. Returns `(B, L, D)`.
    /// `rope`: the per-layer `(cos, sin)` table. `padding_mask`: optional bool
    /// `(B, L)` where `true` = real token, `false` = padding.
    pub fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor), padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let d = self.hidden_size;
        let (cos, sin) = rope;

        // Fused QKV: (B, L, 3D) → three (B, L, D) slices → (B, H, L, hd).
        let qkv = ops::linear(x, &self.qkv_weight, ops::Linear::default())?;
        let heads = |offset: usize| -> Result<Tensor> { Ok(qkv.narrow(-1, offset, d)?.split_heads(self.num_heads)?) };
        let q = heads(0)?.apply_rotary_emb(cos, sin, false)?;
        let k = heads(d)?.apply_rotary_emb(cos, sin, false)?;
        let v = heads(2 * d)?;

        // SDPA, not the op layer: it takes key lengths only, no bool padding
        // mask and no sliding window. The window restricts keys for local
        // layers (`None` for global ones).
        let attn = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .maybe_key_padding_mask(padding_mask)
            .maybe_window(self.window)
            .call()?;

        Ok(ops::linear(&attn.merge_heads()?, &self.out_weight, ops::Linear::default())?)
    }
}
