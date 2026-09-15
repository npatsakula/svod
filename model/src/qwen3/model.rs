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

use std::path::Path;

use svod_dtype::ScalarDType;
use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{Embedding, Layer, Module, RmsNorm};

use crate::state::{self, StateDict};

use super::config::Qwen3Config;
use super::decoder_layer::{Qwen3DecoderLayer, Residual};
use super::error::Result;

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
}

/// Sequence-major `(cos, sin)` for `positions` positions, realized.
fn rope_cache(config: &Qwen3Config, positions: usize) -> svod_tensor::error::Result<(Tensor, Tensor)> {
    let (cos, sin) = Tensor::rope_table(config.rope_theta, positions, config.head_dim, config.dtype.clone())?;
    let seq_major = |t: Tensor| t.try_transpose(1, 2).expect("4-D rope table").contiguous();
    let (cos, sin) = (seq_major(cos), seq_major(sin));
    Tensor::realize_batch([&cos, &sin])?;
    Ok((cos, sin))
}

/// The hidden state of each row's last real token: `[B, L, D]` gathered at
/// `lengths - 1` → `[B, D]`.
pub(crate) fn last_token(hidden: &Tensor, lengths: &Tensor) -> Result<Tensor> {
    let (b, d) = (hidden.dim(0)?, hidden.dim(2)?);
    let index = lengths.try_sub(1)?.try_reshape([b.clone(), SInt::Const(1), SInt::Const(1)])?.try_expand([
        b,
        SInt::Const(1),
        d,
    ])?;
    Ok(hidden.gather(1, &index)?.try_squeeze(Some(1))?)
}

impl Qwen3Model {
    pub fn empty(config: Qwen3Config) -> Self {
        let dtype = config.dtype.clone();
        let embeddings = crate::init::embedding(config.vocab_size, config.hidden_size, dtype.clone());
        let layers = (0..config.num_hidden_layers).map(|_| Qwen3DecoderLayer::empty(&config)).collect();
        let norm = RmsNorm::with_dims(config.hidden_size, config.rms_norm_eps, dtype);
        let rope = rope_cache(&config, config.max_position_embeddings).expect("even head_dim, positive context");
        Self { config, embeddings, layers, norm, rope }
    }

    /// The `(cos, sin)` prefix of the realized cache covering `positions`
    /// positions — sequence-major `[1, positions, 1, Dh/2]`.
    pub(crate) fn rope_prefix(&self, positions: usize) -> Result<(Tensor, Tensor)> {
        Ok((self.rope.0.narrow(1, 0, positions)?, self.rope.1.narrow(1, 0, positions)?))
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

    /// Right-padded `input_ids` `(B, L)` → last-hidden-state `(B, L, D)`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim_const(1)?;
        let padded = self.padded_len(seq_len);
        // Any id is a valid pad under the causal mask; zero is the cheapest.
        let ids = if padded > seq_len {
            input_ids.try_pad(&[(0, 0), (0, (padded - seq_len) as isize)])?
        } else {
            input_ids.clone()
        };
        let rope = self.rope_prefix(padded)?;

        // The stream travels unsummed between layers so each residual add is
        // absorbed by the norm that reads it (see `decoder_layer::Residual`).
        let mut h = Residual::from(self.embeddings.forward(&ids)?);
        for layer in &self.layers {
            h = layer.forward_residual(h, &rope)?;
        }
        let (_, h) = h.norm(&self.norm)?;
        Ok(if padded > seq_len { h.narrow(1, 0, seq_len)? } else { h })
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
        model.load_state_dict(&state::cast_all(sd, dtype), "")?;
        Ok(model)
    }
}
