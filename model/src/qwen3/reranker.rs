//! Qwen3 reranker: decoder backbone + tied LM head + last-token "Yes" logit score.
//!
//! `Qwen/Qwen3-Reranker-{0.6B,4B,8B}` are causal LMs fine-tuned for
//! cross-encoder reranking. The relevance score is the logit at the **"Yes"
//! token** (`yes_loc = 9454`) from the last non-pad position, optionally
//! passed through sigmoid.
//!
//! Pipeline (matching FlagEmbedding `FlagLLMReranker`):
//! 1. Qwen3 decoder forward → `(B, L, D)` hidden states
//! 2. Tied LM head: `hidden @ embed_tokens.weight.T` → `(B, L, V)` logits
//! 3. Last-token logit: `logits[b, lengths[b]-1, :]` → `(B, V)` (right-padded rows)
//! 4. Score: `logits[:, yes_loc]` → `(B,)`
//! 5. Optional sigmoid → `(B,)` normalized scores
//!
//! The checkpoint (`model.safetensors`) uses the `model.` prefix on backbone
//! keys (standard `Qwen3ForCausalLM` convention), which is exactly the prefix
//! this module nests the backbone under — so the published dict loads as is.

use std::path::Path;

use svod_tensor::Tensor;
use svod_tensor::nn::{Module, StateDict, get_tensor, prefixed};

use crate::state;

use super::config::Qwen3Config;
use super::error::Result;

use super::model::{Qwen3Model, last_token};

/// Token ID for "Yes" in the Qwen tokenizer — the reranker's positive class.
const YES_LOC: usize = 9454;

#[derive(Clone)]
pub struct Qwen3Reranker {
    pub model: Qwen3Model,
    /// Tied LM head weight — a clone of `embed_tokens.weight`.
    pub lm_head_weight: Tensor,
    pub yes_loc: usize,
    pub normalize: bool,
}

impl Qwen3Reranker {
    pub fn empty(config: Qwen3Config) -> Self {
        let model = Qwen3Model::empty(config);
        let hidden = model.config.hidden_size;
        let vocab = model.config.vocab_size;
        let dtype = model.config.dtype.clone();
        let lm_head_weight = crate::init::fan_in_uniform(&[vocab, hidden], hidden, dtype);
        Self { model, lm_head_weight, yes_loc: YES_LOC, normalize: true }
    }

    /// Right-padded `input_ids` `(B, L)` + `lengths` `(B)` → `(B,)` relevance
    /// scores.
    pub fn forward(&self, input_ids: &Tensor, lengths: &Tensor) -> Result<Tensor> {
        // Only the last token's logits are scored, so the LM head runs on
        // `(B, D)`, not `(B, L, D)`.
        let last_hidden = last_token(&self.model.forward(input_ids)?, lengths)?;
        let logits = last_hidden.linear().weight(&self.lm_head_weight).call()?;
        let scores = logits.take_index(-1, self.yes_loc as isize)?;
        Ok(if self.normalize { scores.sigmoid()? } else { scores })
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

/// The backbone nests under `model.` (the `Qwen3ForCausalLM` layout) and the
/// LM head is tied to `embed_tokens.weight`, so it is resolved on load rather
/// than stored — hand-written for the tie; every child derives.
impl Module for Qwen3Reranker {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        self.model.write_state(&prefixed(prefix, "model"), out);
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> svod_tensor::error::Result<()> {
        let backbone = prefixed(prefix, "model");
        self.model.load_state_dict(sd, &backbone)?;
        self.lm_head_weight = get_tensor(sd, &prefixed(&backbone, "embed_tokens.weight"))?;
        Ok(())
    }
}
