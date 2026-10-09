//! WavLM self-attention with gated bucketed relative position bias.
//!
//! Direct port of `WavLMSelfAttention` from
//! `submodules/DiariZen/diarizen/models/module/wav2vec2/components.py:549-725`.
//! Notable details:
//!
//! - **Per-layer head pruning**: Q/K/V/O linears are *physically resized* to
//!   `num_kept * head_dim` (where `num_kept = remaining_heads.len()`); the
//!   gating tensors and the shared `rel_attn_embed` table stay full-width and
//!   are indexed at runtime.
//! - **Bucketed relative positions** (`num_buckets=320`, `max_distance=800`):
//!   sign-aware with logarithmic spacing past `num_buckets/2`. The bucket ids
//!   depend only on `(L_q, L_k)`, so the table is built on the host as a small
//!   constant `i64` tensor.
//! - **Gating**: the gate is computed from the *raw input* `x` (NOT the
//!   projected query) reshaped to per-head form. `gru_rel_pos_linear` maps
//!   `head_dim → 8`; reshape to `(..., 2, 4)`, sum over the trailing axis,
//!   sigmoid, chunk in two; combine with `gru_rel_pos_const` to produce a
//!   `(B, total_num_heads, T, 1)` scalar that scales the position bias.
//! - **Position bias** flows between layers: layer 0 owns the bucketed table
//!   `rel_attn_embed` and produces a `(1, total_num_heads, L, L)` bias that
//!   the encoder passes down to every layer's attention forward. Each layer
//!   gates and head-selects independently.

use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::nn::{Layer, Linear, Module};
use svod_tensor::{Tensor, s};
use svod_tk3::ops::{self, Attn};

use crate::init::{Bias, linear, ones};

use super::config::WavLmConfig;
use super::error::Result;

// ---------------------------------------------------------------------------
// Bucket math (matches `_relative_positions_bucket(bidirectional=True)`)
// ---------------------------------------------------------------------------

/// Compute the (L_q, L_k) bucket-id table used to index `rel_attn_embed`.
/// `bidirectional=True` matches WavLM's convention.
pub fn compute_bucket_indices(
    query_length: usize,
    key_length: usize,
    num_buckets: usize,
    max_distance: usize,
) -> Vec<i64> {
    let half_buckets = num_buckets / 2;
    let max_exact = half_buckets / 2;
    let log_ratio = (max_distance as f64 / max_exact as f64).ln();

    let mut buckets = vec![0i64; query_length * key_length];
    for q in 0..query_length {
        for k in 0..key_length {
            let rel = k as i64 - q as i64;
            let sign_offset = if rel > 0 { half_buckets as i64 } else { 0 };
            let abs_rel = rel.unsigned_abs() as usize;

            let bucket = if abs_rel < max_exact {
                abs_rel
            } else {
                let log_pos = (abs_rel as f64 / max_exact as f64).ln() / log_ratio;
                let scaled = (log_pos * (half_buckets - max_exact) as f64) as usize;
                (max_exact + scaled).min(half_buckets - 1)
            };
            buckets[q * key_length + k] = sign_offset + bucket as i64;
        }
    }
    buckets
}

/// Materialize the bucket-index table as a `(L_q, L_k)` int64 `Tensor`.
pub fn bucket_index_tensor(
    query_length: usize,
    key_length: usize,
    num_buckets: usize,
    max_distance: usize,
) -> Result<Tensor> {
    let buckets = compute_bucket_indices(query_length, key_length, num_buckets, max_distance);
    Ok(Tensor::from_slice(&buckets).try_reshape([query_length as isize, key_length as isize])?)
}

/// Compute the un-gated position bias `(1, total_num_heads, L, L)` from the
/// shared `rel_attn_embed` table. The leading `1` broadcasts against the
/// batch dim downstream.
pub fn compute_position_bias(
    rel_attn_embed: &Tensor,
    query_len: usize,
    key_len: usize,
    num_buckets: usize,
    max_distance: usize,
) -> Result<Tensor> {
    let idx = bucket_index_tensor(query_len, key_len, num_buckets, max_distance)?;
    // Embedding lookup: (L, L) indices into (num_buckets, total_num_heads) →
    // (L, L, total_num_heads) → (1, total_num_heads, L, L) for the batch dim.
    let bias = rel_attn_embed.embedding(&idx)?.try_permute(&[2, 0, 1])?.try_unsqueeze(0)?;
    Ok(bias.contiguous())
}

// ---------------------------------------------------------------------------
// GatedRelPosAttention
// ---------------------------------------------------------------------------

/// Multi-head attention with gated bucketed relative-position bias. Forward
/// expects an `x` of shape `(B, L, embed_dim)` and a `position_bias` of
/// shape `(_, total_num_heads, L, L)` (broadcastable over batch).
///
/// Output: `(B, L, embed_dim)`.
#[derive(Clone, Module)]
pub struct GatedRelPosAttention {
    pub total_num_heads: usize,
    /// Surviving head indices (length = `num_kept`). Must be non-empty —
    /// callers should encode the empty-list case as `Option<Self>::None`.
    pub remaining_heads: Vec<usize>,

    // Q / K / V / O projections — physically sized for `num_kept * head_dim`.
    #[module(key = "q_proj")]
    pub q: Linear,
    #[module(key = "k_proj")]
    pub k: Linear,
    #[module(key = "v_proj")]
    pub v: Linear,
    #[module(key = "out_proj")]
    pub out: Linear,

    // Gating params — per-layer, full-width (NOT physically pruned).
    /// `head_dim → 8`.
    pub gru_rel_pos_linear: Linear,
    /// `(1, total_num_heads, 1, 1)`.
    pub gru_rel_pos_const: Tensor,
}

impl GatedRelPosAttention {
    pub fn empty(config: &WavLmConfig, layer_index: usize) -> Self {
        let embed_dim = config.encoder_embed_dim;
        let total_num_heads = config.encoder_total_num_heads[layer_index];
        let head_dim = config.encoder_head_dim;
        let remaining_heads = config.encoder_remaining_heads[layer_index].clone();
        let num_kept = remaining_heads.len();
        assert!(num_kept > 0, "GatedRelPosAttention::empty requires non-empty remaining_heads");

        let proj_out = num_kept * head_dim;
        let linear = |inp: usize, out: usize| linear(inp, out, Bias::Zero, DType::Float32);

        Self {
            total_num_heads,
            remaining_heads,
            q: linear(embed_dim, proj_out),
            k: linear(embed_dim, proj_out),
            v: linear(embed_dim, proj_out),
            out: linear(proj_out, embed_dim),
            gru_rel_pos_linear: linear(head_dim, 8),
            gru_rel_pos_const: ones(&[1, total_num_heads, 1, 1], DType::Float32),
        }
    }

    pub fn num_kept(&self) -> usize {
        self.remaining_heads.len()
    }

    /// Forward. `x`: `(B, L, embed_dim)`. `position_bias`: shape broadcastable
    /// to `(B, total_num_heads, L, L)` (e.g. `(1, total_num_heads, L, L)`).
    ///
    /// Python ref: `WavLMSelfAttention.forward` (components.py:668-725)
    /// + `SelfAttention.forward` (components.py:429-486)
    pub fn forward(&self, x: &Tensor, position_bias: &Tensor) -> Result<Tensor> {
        // Gate (lines 702-710). Py:703-704 reshape the *input* into per-head
        // form: `x.view(bsz, seq_len, total_num_heads, -1).permute(0, 2, 1, 3)`.
        let query_layer = x.split_heads(self.total_num_heads)?; // (B, h, L, hd)

        // Py:706-708  gate_a, gate_b = sigmoid(
        //   gru_rel_pos_linear(query_layer).view(B, h, L, 2, 4).sum(-1)
        // ).chunk(2, dim=-1)
        let gate = self.gru_rel_pos_linear.forward(&query_layer)?; // (B, h, L, 8)
        let shape = gate.shape()?;
        let split = [shape[0].clone(), shape[1].clone(), shape[2].clone(), 2usize.into(), 4usize.into()];
        let gate = gate.try_reshape(split)?.sum_with().axes(-1isize).call()?.sigmoid()?; // (B, h, L, 2)
        let mut chunks = gate.chunk(2, -1)?;
        let gate_b = chunks.pop().expect("chunk(2) yields 2 parts");
        let gate_a = chunks.pop().expect("chunk(2) yields 2 parts");

        // Py:709-713  attn_mask = (gate_a * (gate_b * const - 1) + 2) * position_bias,
        // then keep only the surviving heads.
        let gate = gate_a.try_mul(&gate_b.try_mul(&self.gru_rel_pos_const)?.try_sub(1.0)?)?.try_add(2.0)?;
        let bias = gate.try_mul(position_bias)?.getitem(s![.., self.remaining_heads.clone(), .., ..])?;

        // Q / K / V (lines 455-458) and the attention itself (461-472), the
        // gated bias added to the scaled scores, over sequence-major heads.
        let (nk, head_dim) = (self.num_kept(), x.dim_const(2)? / self.total_num_heads);
        let (b, l) = (shape[0].clone(), shape[2].clone()); // `x`'s batch and length
        let heads = |lin: &Linear| -> Result<Tensor> {
            let dims = [b.clone(), l.clone(), SInt::Const(nk), SInt::Const(head_dim)];
            Ok(lin.forward(x)?.try_reshape(dims)?)
        };
        let (q, k, v) = (heads(&self.q)?, heads(&self.k)?, heads(&self.v)?);
        let attended = ops::attention(&q, &k, &v, Attn { bias: Some(&bias), ..Attn::default() })?;

        // Py:478-480  merge the heads back and project out.
        Ok(self.out.forward(&attended.try_reshape([b, l, SInt::Const(nk * head_dim)])?)?)
    }
}
