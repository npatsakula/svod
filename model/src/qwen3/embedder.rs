//! Qwen3 embedding model: decoder backbone + last-token pooling + L2 normalize
//! (the `sentence-transformers` `modules.json` pipeline).
//!
//! Inputs are right-padded; `lengths` names each row's real token count and
//! the pooled token is the one at `lengths - 1`. Loads from the same
//! `model.safetensors` as [`Qwen3Model`] (bare keys, no `model.` prefix).

use std::path::Path;

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::state::{self, StateDict};

use super::config::Qwen3Config;
use super::error::Result;
use super::model::{Qwen3Model, last_token};

#[derive(Clone, Module)]
pub struct Qwen3Embedding {
    #[module(key = "")]
    pub model: Qwen3Model,
    pub normalize: bool,
}

impl Qwen3Embedding {
    pub fn empty(config: Qwen3Config) -> Self {
        Self { model: Qwen3Model::empty(config), normalize: true }
    }

    /// Right-padded `input_ids` `(B, L)` + `lengths` `(B)` → f32 embeddings
    /// `(B, D)`.
    pub fn encode(&self, input_ids: &Tensor, lengths: &Tensor) -> Result<Tensor> {
        let pooled = last_token(&self.model.forward(input_ids)?, lengths)?.cast(DType::Float32);
        Ok(if self.normalize { pooled.lp_normalize(-1, 2)? } else { pooled })
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

    /// Load from a directory containing `model.safetensors` or multi-shard files.
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
