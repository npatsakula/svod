//! Root ModernBERT backbone: token embeddings → RoPE encoder stack → final
//! LayerNorm. Exposes `forward` returning the `(B, L, D)` last-hidden-state.
//!
//! Loads from a HuggingFace `model.safetensors` checkpoint (the published
//! `answerdotai/ModernBERT-{base,large}` layout). Compute dtype is
//! `config.dtype` (bf16 by default, f32 for CPU parity tests); weights load at
//! their stored dtype and are cast to `config.dtype` on read.

use std::path::Path;

use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, LayerNorm, Module};

use crate::state::{self, StateDict};

use super::config::ModernBertConfig;
use super::embeddings::Embeddings;
use super::encoder::Encoder;
use super::error::Result;

#[derive(Clone, Module)]
pub struct ModernBert {
    #[module(skip)]
    pub config: ModernBertConfig,
    #[module(key = "model.embeddings")]
    pub embeddings: Embeddings,
    #[module(key = "model.layers")]
    pub encoder: Encoder,
    #[module(key = "model.final_norm")]
    pub final_norm: LayerNorm,
}

impl ModernBert {
    pub fn empty(config: ModernBertConfig) -> Self {
        let eps = config.layer_norm_eps;
        let dtype = config.dtype.clone();
        let embeddings = Embeddings::empty(config.vocab_size, config.hidden_size, eps, dtype.clone());
        let encoder = Encoder::empty(&config);
        let final_norm = LayerNorm::with_dims(config.hidden_size, false, eps, dtype);
        Self { config, embeddings, encoder, final_norm }
    }

    /// Eager forward: `input_ids` `(B, L)` int64 + optional `padding_mask`
    /// `(B, L)` bool → last-hidden-state `(B, L, D)`.
    pub fn forward(&self, input_ids: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let x = self.embeddings.forward(input_ids)?;
        let x = self.encoder.forward(&x, padding_mask)?;
        Ok(self.final_norm.forward(&x)?)
    }

    /// Download `config.json` + `model.safetensors` from a HuggingFace Hub
    /// repository and load the backbone. Config is parsed from `config.json`,
    /// then overridden with the caller's `dtype`/`max_batch_size` from `config`.
    pub fn from_hub(model_id: &str, mut config: ModernBertConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", &mut config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: &mut ModernBertConfig) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision)?;
        // Parse the published config.json, then splice back the caller-chosen
        // dtype / max_batch_size (those aren't in the on-disk config).
        let cfg_path = repo.get("config.json")?;
        let parsed = ModernBertConfig::from_json(&cfg_path)?;
        config.merge_structural_from(&parsed);

        let weights_path = repo.get("model.safetensors")?;
        Self::from_safetensors(&weights_path, config.clone())
    }

    /// Load from a `model.safetensors` checkpoint. Weights are cast to
    /// `config.dtype` as they are read.
    pub fn from_safetensors(path: &Path, config: ModernBertConfig) -> Result<Self> {
        let sd = state::load_safetensors(path)?;
        Self::from_state_dict(&sd, config)
    }

    /// Build from a preloaded state dict. Each weight is cast to
    /// `config.dtype`; keys absent from the checkpoint (e.g. layer-0
    /// `attn_norm`, the optional norm biases) are tolerated by the per-block
    /// loaders.
    pub fn from_state_dict(sd: &StateDict, config: ModernBertConfig) -> Result<Self> {
        let dtype = config.dtype.clone();
        let mut model = Self::empty(config);
        model.load_state_dict(&state::cast_all(sd, dtype)?, "")?;
        Ok(model)
    }
}
