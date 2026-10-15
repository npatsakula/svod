//! Root XLM-RoBERTa backbone: embeddings → encoder stack.
//!
//! Loads from a HuggingFace `pytorch_model.bin` checkpoint (the published
//! `BAAI/bge-m3` layout — there is no safetensors variant). Compute dtype is
//! `config.dtype` (bf16 by default, f32 for CPU parity tests); weights load at
//! their stored dtype and are cast to `config.dtype` on read.

use std::path::Path;

use snafu::ResultExt;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::state::{self, StateDict};

use super::config::XlmRobertaConfig;
use super::embeddings::XlmRobertaEmbeddings;
use super::encoder::XlmRobertaEncoder;
use super::error::{PickleSnafu, Result};

#[derive(Clone, Module)]
pub struct XlmRobertaModel {
    #[module(skip)]
    pub config: XlmRobertaConfig,
    pub embeddings: XlmRobertaEmbeddings,
    pub encoder: XlmRobertaEncoder,
}

impl XlmRobertaModel {
    pub fn empty(config: XlmRobertaConfig) -> Self {
        let eps = config.layer_norm_eps;
        let dtype = config.dtype.clone();
        let embeddings = XlmRobertaEmbeddings::empty(
            config.vocab_size,
            config.hidden_size,
            config.max_position_embeddings,
            config.type_vocab_size,
            eps,
            config.pad_token_id,
            dtype.clone(),
        );
        let encoder = XlmRobertaEncoder::empty(&config);
        Self { config, embeddings, encoder }
    }

    /// Eager forward: `input_ids` `(B, L)` int + optional `padding_mask`
    /// `(B, L)` bool → last-hidden-state `(B, L, D)`.
    pub fn forward(&self, input_ids: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let x = self.embeddings.forward(input_ids)?;
        self.encoder.forward(&x, padding_mask)
    }

    /// Download `config.json` + `pytorch_model.bin` from a HuggingFace Hub
    /// repository and load the backbone.
    pub fn from_hub(model_id: &str, mut config: XlmRobertaConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", &mut config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: &mut XlmRobertaConfig) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision)?;
        let cfg_path = repo.get("config.json")?;
        let parsed = XlmRobertaConfig::from_json(&cfg_path)?;
        config.merge_structural_from(&parsed);

        let weights_path = repo.get("pytorch_model.bin")?;
        Self::from_pytorch_bin(&weights_path, config.clone())
    }

    /// Load from a `pytorch_model.bin` checkpoint. Weights are cast to
    /// `config.dtype` as they are read.
    pub fn from_pytorch_bin(path: &Path, config: XlmRobertaConfig) -> Result<Self> {
        let sd = crate::wespeaker::pickle::load_flat_pytorch_bin(path, "").context(PickleSnafu)?;
        Self::from_state_dict(&sd, config)
    }

    /// Build from a preloaded state dict. Each weight is cast to
    /// `config.dtype`.
    pub fn from_state_dict(sd: &StateDict, config: XlmRobertaConfig) -> Result<Self> {
        let dtype = config.dtype.clone();
        let mut model = Self::empty(config);
        model.load_state_dict(&state::cast_all(sd, dtype)?, "")?;
        Ok(model)
    }
}
