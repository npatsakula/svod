//! ModernBERT multi-head attention with fused QKV and RoPE.
//!
//! Direct port of `FlexBertUnpadRopeAttention` (padded path): one fused
//! `Wqkv: Linear(D, 3D)` (no bias), RoPE applied to Q and K, scaled
//! dot-product attention, then `Wo: Linear(D, D)` (no bias).
//!
//! The fused QKV output along dim -1 is `[Q(H*hd) | K(H*hd) | V(H*hd)]`. The
//! tk flash-attention kernel reads it whole as `(B, L, 3H, hd)` once the
//! projection has rotated Q and K in its store, or takes each third as
//! `(B, L, H, hd)`; the SDPA fallback as `(B, H, L, hd)`. Sliding-window local
//! layers pass a `window`; global layers pass `None`.

use snafu::ResultExt;
use svod_dtype::DType;
use svod_ir::origin::OriginScope;
use svod_ir::{ConstValue, SInt};
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::init::fan_in_uniform;

use super::error::{Result, TkSnafu};
use super::linear::{linear, tk_linear};

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

    /// Forward. `x`: `(B, L, D)`. Returns `(B, L, D)`, plus `residual` when
    /// given (the add rides the output projection).
    /// `rope`: the per-layer `(cos, sin)` table. `padding_mask`: optional
    /// `(B, L)` where non-zero = real token, zero = padding.
    pub fn forward(
        &self,
        x: &Tensor,
        rope: &(Tensor, Tensor),
        padding_mask: Option<&Tensor>,
        residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        let attn = match self.packed_flash(x, rope, padding_mask)? {
            Some(attn) => attn,
            None => {
                let qkv = linear(x, &self.qkv_weight, None)?;
                match self.flash(&qkv, rope, padding_mask)? {
                    Some(attn) => attn,
                    None => self.sdpa(&qkv, rope, padding_mask)?,
                }
            }
        };
        linear(&attn, &self.out_weight, residual)
    }

    /// The QKV projection with Q and K rotated in its store
    /// ([`svod_tk::Epilogue::Rope`]) and the tk flash-attention kernel reading
    /// all three heads straight out of its output
    /// ([`svod_tk::flash_attention_packed`]), so nothing passes over Q, K or V
    /// between the two: rotated apart and split into head views instead, the
    /// three are copied into buffers of their own, three kernels a layer.
    /// `None` where either kernel declines — a length off the attention tile
    /// (the padded path copies its operands anyway), or a device whose GEMM
    /// tiles do not stage their store.
    fn packed_flash(
        &self,
        x: &Tensor,
        rope: &(Tensor, Tensor),
        padding_mask: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        let (b, l) = (x.dim(0)?, x.dim_const(1)?);
        if !l.is_multiple_of(svod_tk::FLASH_ATTENTION_SEQUENCE_MULTIPLE) {
            return Ok(None);
        }
        let (h, hd) = (self.num_heads, self.head_dim);
        let table = |t: &Tensor| t.try_reshape([l as isize, hd as isize / 2]);
        let (cos, sin) = (table(&rope.0)?, table(&rope.1)?);
        let epilogue = svod_tk::Epilogue::Rope { cos: &cos, sin: &sin, seq: l, head_dim: hd, heads: 2 * h };
        let Some(qkv) = tk_linear(x, &self.qkv_weight, epilogue)? else { return Ok(None) };
        let qkv = qkv.try_reshape([b.clone(), l.into(), (3 * h).into(), hd.into()])?;
        let opts = svod_tk::FaOpts { causal: false, key_mask: padding_mask, window: self.window, ..Default::default() };
        let Some(out) = svod_tk::flash_attention_packed(&qkv, (h, h), opts).context(TkSnafu)? else {
            return Ok(None);
        };
        Ok(Some(out.try_reshape([b, SInt::Const(l), SInt::Const(self.hidden_size)])?))
    }

    /// The tk flash-attention kernel, window and key mask included, or `None`
    /// where it does not run: 32-bit activations (its operands are 16-bit), a
    /// batch the JIT leaves free (it needs a static grid), a device it does not
    /// support.
    ///
    /// A length off the kernel's tile runs padded to it: the kernel copies its
    /// operands anyway, so only those copies grow, the padded keys are masked
    /// and the padded rows dropped — the GEMMs around it never see them.
    fn flash(&self, qkv: &Tensor, rope: &(Tensor, Tensor), padding_mask: Option<&Tensor>) -> Result<Option<Tensor>> {
        let (b, l) = (qkv.dim(0)?, qkv.dim_const(1)?);
        let sixteen_bit = [DType::BFloat16, DType::Float16].contains(&qkv.dtype());
        if !sixteen_bit || svod_tk::launch::pinned_dim(&b).is_none() {
            return Ok(None);
        }
        let (d, hd) = (self.hidden_size, self.head_dim);
        let padded = l.next_multiple_of(svod_tk::FLASH_ATTENTION_SEQUENCE_MULTIPLE);
        let extra = (padded - l) as isize;
        // Sequence-major, so the head split is a view; the `[1, 1, L, hd/2]`
        // tables rotate it as `[1, L, 1, hd/2]`.
        let heads = |offset: usize| -> Result<Tensor> {
            Ok(qkv.narrow(-1, offset, d)?.try_reshape([b.clone(), l.into(), self.num_heads.into(), hd.into()])?)
        };
        let seq_major = |t: &Tensor| t.try_reshape([1, l as isize, 1, hd as isize / 2]);
        let (cos, sin) = (seq_major(&rope.0)?, seq_major(&rope.1)?);
        let tile = |t: Tensor| -> Result<Tensor> { Ok(t.try_pad(&[(0, 0), (0, extra), (0, 0), (0, 0)])?) };
        let q = tile(heads(0)?.apply_rotary_emb(&cos, &sin, false)?)?;
        let k = tile(heads(d)?.apply_rotary_emb(&cos, &sin, false)?)?;
        let v = tile(heads(2 * d)?)?;
        let key_mask = if extra == 0 {
            padding_mask.cloned()
        } else {
            // Every layer pads the same mask: built outside the layer's origin
            // scope, it is one kernel for all of them.
            let _shared = OriginScope::suspend();
            Some(match padding_mask {
                Some(mask) => mask.try_pad(&[(0, 0), (0, extra)])?,
                None => Tensor::arange(0, Some(padded as i64), None)?
                    .try_lt(Tensor::const_(ConstValue::Int(l as i64), DType::Int32))?
                    .try_unsqueeze(0)?
                    .try_expand([b.clone(), SInt::Const(padded)])?,
            })
        };
        let opts =
            svod_tk::FaOpts { causal: false, key_mask: key_mask.as_ref(), window: self.window, ..Default::default() };
        let Some(out) = svod_tk::flash_attention_with(&q, &k, &v, opts).context(TkSnafu)? else {
            return Ok(None);
        };
        Ok(Some(out.narrow(1, 0usize, l)?.try_reshape([b, SInt::Const(l), SInt::Const(d)])?))
    }

    /// Scaled dot-product attention over `(B, H, L, hd)` heads.
    fn sdpa(&self, qkv: &Tensor, (cos, sin): &(Tensor, Tensor), padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let d = self.hidden_size;
        let heads = |offset: usize| -> Result<Tensor> { Ok(qkv.narrow(-1, offset, d)?.split_heads(self.num_heads)?) };
        // Realized: left lazy, the rotation fuses into the QKᵀ reduce and is
        // recomputed once per key (H·L²·hd rotations instead of H·L·hd).
        let q = heads(0)?.apply_rotary_emb(cos, sin, false)?.contiguous();
        let k = heads(d)?.apply_rotary_emb(cos, sin, false)?.contiguous();
        let v = heads(2 * d)?;

        // Window restricts keys for local layers; the bon builder is
        // type-stated, so chain unconditionally (None for global layers).
        let attn = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .maybe_key_padding_mask(padding_mask)
            .maybe_window(self.window)
            .call()?;
        Ok(attn.merge_heads()?)
    }
}
