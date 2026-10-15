//! Root Qwen3 decoder backbone: token embeddings → RoPE decoder stack →
//! final RMSNorm. Exposes `forward` returning the `(B, L, D)` last-hidden-state.
//!
//! Loads from a HuggingFace `model.safetensors` checkpoint with bare keys
//! (`embed_tokens.weight`, `layers.N.*`, `norm.weight` — no `model.` prefix).
//! Compute dtype is `config.dtype` (bf16 by default, f32 for CPU parity).
//!
//! **Padding convention: right.** Attention is causal, so a real token never
//! sees the padding after it and no mask is needed anywhere in the stack; the
//! hand flash-attention kernel runs unmasked. Rows are then read back at
//! [`last_token`]. RoPE only enters through position differences, so this is
//! the reference's left-padded result to rounding.
//!
//! [`Qwen3Model::forward_packed`] takes rows that hold several sequences end
//! to end: each sequence restarts its positions at 0 and a segment mask keeps
//! it from seeing the ones packed before it, so the row's padding shrinks to
//! whatever the packer could not fill.

use std::path::Path;

use snafu::ensure;
use svod_dtype::{DType, ScalarDType};
use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{Embedding, Layer, Module, RmsNorm};

use crate::state::{self, StateDict};

use super::config::Qwen3Config;
use super::decoder_layer::{Qwen3DecoderLayer, Residual};
use super::error::{ContextLengthSnafu, Result};

#[derive(Clone, Module)]
pub struct Qwen3Model {
    #[module(skip)]
    pub config: Qwen3Config,
    #[module(key = "embed_tokens")]
    pub embeddings: Embedding,
    pub layers: Vec<Qwen3DecoderLayer>,
    pub norm: RmsNorm,
    /// `(cos, sin)` over every position the model admits, `[1, P, 1, Dh/2]` —
    /// the sequence-major broadcast the attention layout wants. Realized once
    /// at construction; a forward slices its prefix.
    #[module(skip)]
    rope: (Tensor, Tensor),
    /// The table's angular frequencies, `[Dh/2]` f32, for rotating packed
    /// rows' tokens at their own positions.
    #[module(skip)]
    inv_freq: Tensor,
}

/// Positions the rope cache holds: the context rounded up to the attention
/// tile, because [`Qwen3Model::forward`] pads a sequence to that tile and
/// narrows the cache to the padded length. The rows past the context are only
/// ever read by that padding, which is causal-invisible.
fn rope_positions(config: &Qwen3Config) -> usize {
    config.max_position_embeddings.max(1).next_multiple_of(svod_tk::FLASH_ATTENTION_SEQUENCE_MULTIPLE)
}

/// Sequence-major `(cos, sin)` for `positions` positions, realized.
fn rope_cache(config: &Qwen3Config, positions: usize) -> svod_tensor::error::Result<(Tensor, Tensor)> {
    let (cos, sin) = Tensor::rope_table(config.rope_theta, positions, config.head_dim, config.dtype.clone())?;
    let seq_major = |t: Tensor| t.try_transpose(1, 2).expect("4-D rope table").contiguous();
    let (cos, sin) = (seq_major(cos), seq_major(sin));
    Tensor::realize_batch([&cos, &sin])?;
    Ok((cos, sin))
}

/// The hidden states at `index` `[B, S]` positions of each row: `[B, L, D]`
/// gathered along the sequence → `[B, S, D]`.
pub(crate) fn gather_tokens(hidden: &Tensor, index: &Tensor) -> Result<Tensor> {
    let (b, d) = (hidden.dim(0)?, hidden.dim(2)?);
    let s = index.dim(1)?;
    let index = index.try_reshape([b.clone(), s.clone(), SInt::Const(1)])?.try_expand([b, s, d])?;
    Ok(hidden.gather(1, &index)?)
}

/// The hidden state of each row's last real token: `[B, L, D]` gathered at
/// `lengths - 1` → `[B, D]`.
///
/// **`lengths` `[B]` is the caller's contract: each entry in `1..=L`.** The
/// values live on the device — a JIT plan binds them long after this graph is
/// built — so the range cannot be checked here without a sync, and the gather
/// would answer an entry outside it with a row of zeros (an all-zero embedding,
/// a flat `sigmoid(0)` score). The index saturates into the row instead, so a
/// degenerate length pools a real token: the same rule [`super::Qwen3Embedder`]
/// applies when it embeds an empty row as one pad token.
pub(crate) fn last_token(hidden: &Tensor, lengths: &Tensor) -> Result<Tensor> {
    let (b, l) = (hidden.dim(0)?, hidden.dim_const(1)?);
    let last = l.saturating_sub(1) as isize;
    let index = lengths.try_sub(1)?.clamp().min(0isize).max(last).call()?.try_reshape([b, SInt::Const(1)])?;
    Ok(gather_tokens(hidden, &index)?.try_squeeze(Some(1))?)
}

/// The layout of packed rows — several sequences end to end in one row of
/// `[B, L]` tokens, both tables `[B, L]` `i32`.
#[derive(Clone, Copy)]
pub struct Packing<'a> {
    /// Each token's position within its own sequence.
    pub positions: &'a Tensor,
    /// The row index of the first token of each token's sequence; a token
    /// attends to nothing before it.
    pub seg_start: &'a Tensor,
}

impl Qwen3Model {
    pub fn empty(config: Qwen3Config) -> Self {
        let dtype = config.dtype.clone();
        let embeddings = crate::init::embedding(config.vocab_size, config.hidden_size, dtype.clone());
        let layers = (0..config.num_hidden_layers).map(|_| Qwen3DecoderLayer::empty(&config)).collect();
        let norm = RmsNorm::with_dims(config.hidden_size, config.rms_norm_eps, dtype);
        let rope = rope_cache(&config, rope_positions(&config)).expect("even head_dim, positive context");
        let inv_freq = Tensor::rope_inv_freq(config.rope_theta, config.head_dim).expect("even head_dim");
        inv_freq.realize().expect("a [Dh/2] constant");
        Self { config, embeddings, layers, norm, rope, inv_freq }
    }

    /// The `(cos, sin)` prefix of the realized cache covering `positions`
    /// positions — sequence-major `[1, positions, 1, Dh/2]`.
    pub(crate) fn rope_prefix(&self, positions: usize) -> Result<(Tensor, Tensor)> {
        Ok((self.rope.0.narrow(1, 0, positions)?, self.rope.1.narrow(1, 0, positions)?))
    }

    /// `(cos, sin)` at each token's own position, `positions` `[B, L]` →
    /// `[B, L, 1, Dh/2]`, by the cache's own arithmetic (f32 angles, then the
    /// model dtype): the rows the cache holds at those positions, without a
    /// gather over its `P` rows.
    fn rope_at(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, l) = (positions.dim(0)?, positions.dim(1)?);
        let angles = positions
            .cast(DType::Float32)
            .try_reshape([b, l, SInt::Const(1), SInt::Const(1)])?
            .try_mul(&self.inv_freq)?;
        let dtype = self.config.dtype.clone();
        Ok((angles.cos()?.cast(dtype.clone()), angles.sin()?.cast(dtype)))
    }

    /// Sequence length the stack runs at: on a device with the hand kernel,
    /// 16-bit activations pad up to its tile so every layer takes the fast
    /// path; padded rows are causal-invisible and sliced off again.
    fn padded_len(&self, seq_len: usize) -> usize {
        let sixteen_bit = matches!(self.config.dtype.base(), ScalarDType::BFloat16 | ScalarDType::Float16);
        if sixteen_bit && svod_tk::flash_attention_supported(&self.embeddings.weight.device()) {
            seq_len.next_multiple_of(svod_tk::FLASH_ATTENTION_SEQUENCE_MULTIPLE)
        } else {
            seq_len
        }
    }

    /// Right-padded `input_ids` `(B, L)` → last-hidden-state `(B, L, D)`, for
    /// any `L` within the config's context.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim_const(1)?;
        let max_position_embeddings = self.config.max_position_embeddings;
        ensure!(seq_len <= max_position_embeddings, ContextLengthSnafu { seq_len, max_position_embeddings });
        let padded = self.padded_len(seq_len);
        // Any id is a valid pad under the causal mask; zero is the cheapest.
        let ids = if padded > seq_len {
            input_ids.try_pad(&[(0, 0), (0, (padded - seq_len) as isize)])?
        } else {
            input_ids.clone()
        };
        let rope = self.rope_prefix(padded)?;
        let h = self.stack(&ids, &rope, None)?;
        Ok(if padded > seq_len { h.narrow(1, 0, seq_len)? } else { h })
    }

    /// Packed `input_ids` `(B, L)` → last-hidden-state `(B, L, D)`: every
    /// token is rotated by its own position and sees only its own sequence.
    /// `L` is the caller's — the packer sizes rows to the attention tile.
    pub fn forward_packed(&self, input_ids: &Tensor, packing: &Packing) -> Result<Tensor> {
        let rope = self.rope_at(packing.positions)?;
        self.stack(input_ids, &rope, Some(packing.seg_start))
    }

    /// The decoder stack and final norm over embedded `ids`.
    fn stack(&self, ids: &Tensor, rope: &(Tensor, Tensor), seg_start: Option<&Tensor>) -> Result<Tensor> {
        // The stream travels unsummed between layers so each residual add is
        // absorbed by the norm that reads it (see `decoder_layer::Residual`).
        let mut h = Residual::from(self.embeddings.forward(ids)?);
        for layer in &self.layers {
            h = layer.forward_residual(h, rope, seg_start)?;
        }
        Ok(h.norm(&self.norm)?.1)
    }

    pub fn from_hub(model_id: &str, mut config: Qwen3Config) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", &mut config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: &mut Qwen3Config) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision)?;
        let cfg_path = repo.get("config.json")?;
        let parsed = Qwen3Config::from_json(&cfg_path)?;
        config.merge_structural_from(&parsed);

        let dir = crate::qwen3::download_safetensors(&repo)?;
        Self::from_safetensors_dir(&dir, config.clone())
    }

    pub fn from_safetensors(path: &Path, config: Qwen3Config) -> Result<Self> {
        let sd = state::load_safetensors(path)?;
        Self::from_state_dict(&sd, config)
    }

    /// Load from a directory containing `model.safetensors` (single-file) or
    /// `model.safetensors.index.json` + shards (multi-shard).
    pub fn from_safetensors_dir(dir: &Path, config: Qwen3Config) -> Result<Self> {
        let sd = state::load_safetensors_dir(dir)?;
        Self::from_state_dict(&sd, config)
    }

    pub fn from_state_dict(sd: &StateDict, config: Qwen3Config) -> Result<Self> {
        let dtype = config.dtype.clone();
        let mut model = Self::empty(config);
        model.load_state_dict(&state::cast_all(sd, dtype)?, "")?;
        Ok(model)
    }
}
