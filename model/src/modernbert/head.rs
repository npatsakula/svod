//! ModernBERT MLM (fill-mask) head — `FlexBertPredictionHead` + decoder,
//! mirroring `FlexBertForMaskedLM`.
//!
//! The published `answerdotai/ModernBERT-{base,large}` head is **LayerNorm +
//! GELU**: `head_pred_act: gelu` and `normalization: layernorm` in the pretrain
//! YAML (HF's `modeling_modernbert.py` also hardcodes GELU). RMSNorm + SiLU is
//! only the FlexBERT *code default*, overridden for the released checkpoints,
//! so this reuses [`svod_tensor::nn::LayerNorm`] and [`Tensor::gelu_exact`] with no new
//! normalization/activation ops.
//!
//! **Weight tying.** When `config.tie_word_embeddings` is `true` (the published
//! checkpoints), the decoder weight is the token embedding table. The head
//! resolves this at load time — cloning `model.embeddings.tok_embeddings.weight`
//! out of the same state dict — so [`MlmHead::forward`] is fully self-contained
//! (no call-time tensor argument), matching the `gigaam::CTCHead` idiom. When
//! tying is disabled, the head loads a standalone `decoder.weight` instead.

use std::path::Path;

use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, LayerNorm, Module, StateDict, get_tensor, prefixed};

use crate::init::{fan_in_uniform, zeros};
use crate::state;

use super::config::ModernBertConfig;
use super::error::Result;

use super::model::ModernBert;

/// The token-embedding key the tied decoder aliases. Lives on the backbone but
/// is present in the same `model.safetensors` checkpoint the head loads from.
const TIED_EMBEDDING_KEY: &str = "model.embeddings.tok_embeddings.weight";

/// `FlexBertPredictionHead` + the decoder. Owns all its weights: `head.dense`
/// (square Linear, no bias), `head.norm` (gamma-only LayerNorm), the decoder
/// weight `(V, D)` (tied-to-embeddings or standalone), and `decoder.bias`
/// (present iff `config.decoder_bias`).
#[derive(Clone)]
pub struct MlmHead {
    pub dense_weight: Tensor,
    pub norm: LayerNorm,
    /// `(V, D)`. When `tie_word_embeddings`, this is a clone of the embedding
    /// table (aliased under `model.embeddings.tok_embeddings.weight` in the
    /// checkpoint); otherwise a standalone `decoder.weight`.
    pub decoder_weight: Tensor,
    /// `decoder.bias (V,)` — `None` when `config.decoder_bias` is false.
    pub decoder_bias: Option<Tensor>,
}

impl MlmHead {
    pub fn empty(config: &ModernBertConfig) -> Self {
        let dtype = config.dtype.clone();
        let dense_weight = fan_in_uniform(&[config.hidden_size, config.hidden_size], config.hidden_size, dtype.clone());
        let norm = LayerNorm::with_dims(config.hidden_size, false, config.layer_norm_eps, dtype.clone());
        let decoder_weight =
            fan_in_uniform(&[config.vocab_size, config.hidden_size], config.hidden_size, dtype.clone());
        let decoder_bias = if config.decoder_bias { Some(zeros(&[config.vocab_size], dtype)) } else { None };
        Self { dense_weight, norm, decoder_weight, decoder_bias }
    }

    /// `hidden (B,L,D)` → logits `(B,L,V)`.
    ///
    /// Order matches `FlexBertPredictionHead.forward` (`norm(act(dense(x))`)
    /// followed by the decoder Linear `logits = h @ decoder_weight.T + bias`.
    /// The decoder weight is owned by the head (tied or standalone), so this is
    /// self-contained — no external tensor argument, mirroring `CTCHead`.
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let h = hidden.linear().weight(&self.dense_weight).call()?;
        let h = self.norm.forward(&h.gelu_exact()?)?;
        let logits = h.linear().weight(&self.decoder_weight).call()?;
        match &self.decoder_bias {
            Some(b) => Ok(logits.try_add(b)?),
            None => Ok(logits),
        }
    }
}

impl Module for MlmHead {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        out.insert(prefixed(prefix, "head.dense.weight"), self.dense_weight.clone());
        self.norm.write_state(&prefixed(prefix, "head.norm"), out);
        // Emit the decoder weight under its own key. For a tied checkpoint this
        // duplicates `model.embeddings.tok_embeddings.weight`; loaders dedup on
        // load by re-aliasing, and `ModernBertForMaskedLm::write_state` drops
        // the copy (see below).
        out.insert(prefixed(prefix, "decoder.weight"), self.decoder_weight.clone());
        if let Some(b) = &self.decoder_bias {
            out.insert(prefixed(prefix, "decoder.bias"), b.clone());
        }
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> svod_tensor::error::Result<()> {
        self.dense_weight = get_tensor(sd, &prefixed(prefix, "head.dense.weight"))?;
        self.norm.load_state_dict(sd, &prefixed(prefix, "head.norm"))?;
        // Resolve the decoder weight. Published (tied) checkpoints store it
        // only as the embedding table, so alias from there when the standalone
        // `decoder.weight` key is absent. `model.safetensors` carries the
        // embedding weight at its unprefixed backbone key; fall back to the
        // prefixed one for a head loaded under a non-empty prefix.
        self.decoder_weight = match sd.get(&prefixed(prefix, "decoder.weight")) {
            Some(w) => w.clone(),
            None => {
                get_tensor(sd, TIED_EMBEDDING_KEY).or_else(|_| get_tensor(sd, &prefixed(prefix, TIED_EMBEDDING_KEY)))?
            }
        };
        self.decoder_bias = sd.get(&prefixed(prefix, "decoder.bias")).cloned();
        Ok(())
    }
}

/// `FlexBertForMaskedLM`: the backbone + the MLM head. Loads from the same
/// `model.safetensors` as [`ModernBert`] (the backbone keys `model.*` and the
/// head keys `head.*` / `decoder.*` are disjoint, so one checkpoint populates
/// both). Returns `(B, L, V)` logits.
#[derive(Clone)]
pub struct ModernBertForMaskedLm {
    pub bert: ModernBert,
    pub head: MlmHead,
}

impl ModernBertForMaskedLm {
    pub fn empty(config: ModernBertConfig) -> Self {
        let head = MlmHead::empty(&config);
        let bert = ModernBert::empty(config);
        Self { bert, head }
    }

    /// Eager forward: `input_ids` `(B, L)` + optional `padding_mask` `(B, L)`
    /// bool → logits `(B, L, V)`.
    pub fn forward(&self, input_ids: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let hidden = self.bert.forward(input_ids, padding_mask)?;
        self.head.forward(&hidden)
    }

    /// Download `config.json` + `model.safetensors` from a HuggingFace Hub
    /// repository and load the full MLM model. Mirrors [`ModernBert::from_hub`].
    pub fn from_hub(model_id: &str, mut config: ModernBertConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", &mut config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: &mut ModernBertConfig) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision)?;
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

    /// Build from a preloaded state dict. Each weight is cast to `config.dtype`.
    pub fn from_state_dict(sd: &StateDict, config: ModernBertConfig) -> Result<Self> {
        let dtype = config.dtype.clone();
        let mut model = Self::empty(config);
        model.load_state_dict(&state::cast_all(sd, dtype)?, "")?;
        Ok(model)
    }
}

impl Module for ModernBertForMaskedLm {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        self.bert.write_state(prefix, out);
        // The backbone already emits the tied embedding weight under
        // `model.embeddings.tok_embeddings.weight`; drop the head's duplicate
        // `decoder.weight` copy so the composite round-trips to the published
        // (tied) checkpoint shape — exactly one `(V, D)` weight, under the
        // embedding key.
        let decoder_key = prefixed(prefix, "decoder.weight");
        for (k, v) in self.head.state_dict(prefix) {
            if k != decoder_key {
                out.insert(k, v);
            }
        }
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> svod_tensor::error::Result<()> {
        self.bert.load_state_dict(sd, prefix)?;
        self.head.load_state_dict(sd, prefix)
    }
}
