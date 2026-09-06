//! ModernBERT configuration, parsed from HuggingFace `config.json`.
//!
//! Mirrors the published schema of `answerdotai/ModernBERT-{base,large}`. The
//! [`RawModernBertConfig`] serde mirror captures the on-disk shape; the clean
//! [`ModernBertConfig`] keeps only the fields the Rust backbone consumes and
//! adds a caller-chosen compute [`DType`] (defaults to [`crate::default_compute_dtype`];
//! f32 for CPU parity tests).
//!
//! Per-layer global vs local attention: every `global_attn_every_n_layers`-th
//! layer (0-indexed) attends to the full sequence; the rest use a
//! `local_attention`-wide sliding window split evenly. Global layers use
//! `global_rope_theta`; local layers use `local_rope_theta`.

use std::path::Path;

use serde::Deserialize;
use svod_dtype::DType;

use super::error::{Error, Result};

/// Clean, resolved ModernBERT backbone config.
#[derive(Clone, Debug)]
pub struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    /// Rotary base for global-attention layers.
    pub global_rope_theta: f64,
    /// Rotary base for local (sliding-window) layers.
    pub local_rope_theta: f64,
    /// Sliding-window width for local layers (ModernBERT splits it evenly:
    /// each query attends to `local_attention/2` keys on each side).
    pub local_attention: usize,
    /// Global attention every N layers (0-indexed: layers 0, N, 2N, … are
    /// global; the rest are local).
    pub global_attn_every_n_layers: usize,
    pub pad_token_id: usize,
    pub tie_word_embeddings: bool,
    /// Whether the MLM decoder has a bias term (`config.json: decoder_bias`).
    /// `true` for the published `ModernBERT-{base,large}`; the weight is tied
    /// to the token embeddings, so only the bias is stored.
    pub decoder_bias: bool,
    /// Caller-chosen compute dtype (bf16 by default; f32 for CPU parity).
    pub dtype: DType,
    /// Upper bound on the symbolic batch variable in the JIT wrapper.
    pub max_batch_size: usize,
}

impl ModernBertConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// `(left, right)` window for a local layer. ModernBERT splits the
    /// `local_attention` width evenly; the published configs use 128 → (64, 64).
    pub fn local_window(&self) -> (usize, usize) {
        let half = self.local_attention / 2;
        (half, half)
    }

    /// `true` iff `layer_id` is a global-attention layer.
    pub fn is_global_layer(&self, layer_id: usize) -> bool {
        layer_id.is_multiple_of(self.global_attn_every_n_layers)
    }

    /// Rotary base for the given layer.
    pub fn rope_theta(&self, layer_id: usize) -> f64 {
        if self.is_global_layer(layer_id) { self.global_rope_theta } else { self.local_rope_theta }
    }

    /// Splice in the structural fields parsed from the published `config.json`,
    /// preserving the caller-chosen `dtype` / `max_batch_size`. Shared by the
    /// backbone (`ModernBert`) and MLM (`ModernBertForMaskedLm`) Hub loaders so
    /// both stay in sync without copy-pasting the field list.
    pub fn merge_structural_from(&mut self, parsed: &Self) {
        self.vocab_size = parsed.vocab_size;
        self.hidden_size = parsed.hidden_size;
        self.num_hidden_layers = parsed.num_hidden_layers;
        self.num_attention_heads = parsed.num_attention_heads;
        self.intermediate_size = parsed.intermediate_size;
        self.max_position_embeddings = parsed.max_position_embeddings;
        self.layer_norm_eps = parsed.layer_norm_eps;
        self.global_rope_theta = parsed.global_rope_theta;
        self.local_rope_theta = parsed.local_rope_theta;
        self.local_attention = parsed.local_attention;
        self.global_attn_every_n_layers = parsed.global_attn_every_n_layers;
        self.pad_token_id = parsed.pad_token_id;
        self.tie_word_embeddings = parsed.tie_word_embeddings;
        self.decoder_bias = parsed.decoder_bias;
    }
}

impl Default for ModernBertConfig {
    /// Placeholder config — structural fields are zero since [`from_hub`]
    /// overwrites them all from `config.json`. Only `dtype` and
    /// `max_batch_size` are caller-chosen.
    fn default() -> Self {
        Self {
            vocab_size: 0,
            hidden_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            intermediate_size: 0,
            max_position_embeddings: 0,
            layer_norm_eps: 0.0,
            global_rope_theta: 0.0,
            local_rope_theta: 0.0,
            local_attention: 0,
            global_attn_every_n_layers: 0,
            pad_token_id: 0,
            tie_word_embeddings: false,
            decoder_bias: false,
            dtype: crate::default_compute_dtype(),
            max_batch_size: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// config.json parsing
// ---------------------------------------------------------------------------

impl ModernBertConfig {
    /// Parse a HuggingFace `config.json`. Unrecognized fields are ignored; any
    /// of the structural fields below that is absent falls back to the
    /// ModernBERT-base defaults.
    pub fn from_json(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| Error::Config { message: format!("reading config.json: {e}") })?;
        Self::from_json_str(&data)
    }

    pub fn from_json_str(data: &str) -> Result<Self> {
        let raw: RawModernBertConfig =
            serde_json::from_str(data).map_err(|e| Error::Config { message: format!("JSON parse error: {e}") })?;
        Ok(Self::from_raw(raw))
    }

    fn from_raw(raw: RawModernBertConfig) -> Self {
        let d = ModernBertConfig::default();
        ModernBertConfig {
            vocab_size: raw.vocab_size.unwrap_or(d.vocab_size),
            hidden_size: raw.hidden_size.unwrap_or(d.hidden_size),
            num_hidden_layers: raw.num_hidden_layers.unwrap_or(d.num_hidden_layers),
            num_attention_heads: raw.num_attention_heads.unwrap_or(d.num_attention_heads),
            intermediate_size: raw.intermediate_size.unwrap_or(d.intermediate_size),
            max_position_embeddings: raw.max_position_embeddings.unwrap_or(d.max_position_embeddings),
            layer_norm_eps: raw.layer_norm_eps.or(raw.norm_eps).unwrap_or(d.layer_norm_eps),
            global_rope_theta: raw.global_rope_theta.unwrap_or(d.global_rope_theta),
            local_rope_theta: raw.local_rope_theta.unwrap_or(d.local_rope_theta),
            local_attention: raw.local_attention.unwrap_or(d.local_attention),
            global_attn_every_n_layers: raw.global_attn_every_n_layers.unwrap_or(d.global_attn_every_n_layers),
            pad_token_id: raw.pad_token_id.unwrap_or(d.pad_token_id),
            tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(d.tie_word_embeddings),
            decoder_bias: raw.decoder_bias.unwrap_or(d.decoder_bias),
            dtype: d.dtype,
            max_batch_size: d.max_batch_size,
        }
    }
}

/// Serde mirror of the published `config.json`. Every field is optional so a
/// missing field falls back to the base defaults rather than failing.
#[derive(Deserialize)]
struct RawModernBertConfig {
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    intermediate_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    /// ModernBERT publishes both `layer_norm_eps` and `norm_eps` (equal).
    layer_norm_eps: Option<f64>,
    norm_eps: Option<f64>,
    global_rope_theta: Option<f64>,
    local_rope_theta: Option<f64>,
    local_attention: Option<usize>,
    global_attn_every_n_layers: Option<usize>,
    pad_token_id: Option<usize>,
    tie_word_embeddings: Option<bool>,
    decoder_bias: Option<bool>,
}
