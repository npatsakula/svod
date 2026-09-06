//! XLM-RoBERTa configuration, parsed from HuggingFace `config.json`.
//!
//! Mirrors the published schema of `BAAI/bge-m3` (XLM-RoBERTa-large). The
//! [`RawXlmRobertaConfig`] serde mirror captures the on-disk shape; the clean
//! [`XlmRobertaConfig`] keeps only the fields the Rust backbone consumes and
//! adds a caller-chosen compute [`DType`] (defaults to [`crate::default_compute_dtype`];
//! f32 for CPU parity tests).

use std::path::Path;

use serde::Deserialize;
use svod_dtype::DType;

use super::error::{Error, Result};

/// Clean, resolved XLM-RoBERTa config.
#[derive(Clone, Debug)]
pub struct XlmRobertaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f64,
    /// XLM-RoBERTa padding index (= `pad_token_id`). Position IDs are offset
    /// by this value: real tokens start at `padding_idx + 1`.
    pub pad_token_id: usize,
    /// Caller-chosen compute dtype (bf16 by default; f32 for CPU parity).
    pub dtype: DType,
    /// Upper bound on the symbolic batch variable in the JIT wrapper.
    pub max_batch_size: usize,
}

impl XlmRobertaConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Splice in the structural fields parsed from the published `config.json`,
    /// preserving the caller-chosen `dtype` / `max_batch_size`. Shared by the
    /// backbone and head Hub loaders so both stay in sync.
    pub fn merge_structural_from(&mut self, parsed: &Self) {
        self.vocab_size = parsed.vocab_size;
        self.hidden_size = parsed.hidden_size;
        self.num_hidden_layers = parsed.num_hidden_layers;
        self.num_attention_heads = parsed.num_attention_heads;
        self.intermediate_size = parsed.intermediate_size;
        self.max_position_embeddings = parsed.max_position_embeddings;
        self.type_vocab_size = parsed.type_vocab_size;
        self.layer_norm_eps = parsed.layer_norm_eps;
        self.pad_token_id = parsed.pad_token_id;
    }

    /// Parse a HuggingFace `config.json`. Unrecognized fields are ignored; any
    /// of the structural fields below that is absent falls back to the
    /// XLM-RoBERTa-large defaults.
    pub fn from_json(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| Error::Config { message: format!("reading config.json: {e}") })?;
        Self::from_json_str(&data)
    }

    pub fn from_json_str(data: &str) -> Result<Self> {
        let raw: RawXlmRobertaConfig =
            serde_json::from_str(data).map_err(|e| Error::Config { message: format!("JSON parse error: {e}") })?;
        Ok(Self::from_raw(raw))
    }

    fn from_raw(raw: RawXlmRobertaConfig) -> Self {
        let base = xlm_roberta_large();
        XlmRobertaConfig {
            vocab_size: raw.vocab_size.unwrap_or(base.vocab_size),
            hidden_size: raw.hidden_size.unwrap_or(base.hidden_size),
            num_hidden_layers: raw.num_hidden_layers.unwrap_or(base.num_hidden_layers),
            num_attention_heads: raw.num_attention_heads.unwrap_or(base.num_attention_heads),
            intermediate_size: raw.intermediate_size.unwrap_or(base.intermediate_size),
            max_position_embeddings: raw.max_position_embeddings.unwrap_or(base.max_position_embeddings),
            type_vocab_size: raw.type_vocab_size.unwrap_or(base.type_vocab_size),
            layer_norm_eps: raw.layer_norm_eps.or(raw.norm_eps).unwrap_or(base.layer_norm_eps),
            pad_token_id: raw.pad_token_id.unwrap_or(base.pad_token_id),
            dtype: base.dtype,
            max_batch_size: base.max_batch_size,
        }
    }
}

/// `BAAI/bge-m3` / XLM-RoBERTa-large: 24 layers, hidden 1024, intermediate
/// 4096, 16 heads (head_dim 64), vocab 250002, position embeddings 8194.
pub fn xlm_roberta_large() -> XlmRobertaConfig {
    XlmRobertaConfig {
        vocab_size: 250002,
        hidden_size: 1024,
        num_hidden_layers: 24,
        num_attention_heads: 16,
        intermediate_size: 4096,
        max_position_embeddings: 8194,
        type_vocab_size: 1,
        layer_norm_eps: 1e-5,
        pad_token_id: 1,
        dtype: crate::default_compute_dtype(),
        max_batch_size: 1,
    }
}

/// Serde mirror of the published `config.json`. Every field is optional so a
/// missing field falls back to the large defaults rather than failing.
#[derive(Deserialize)]
struct RawXlmRobertaConfig {
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    intermediate_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    type_vocab_size: Option<usize>,
    /// XLM-RoBERTa publishes both `layer_norm_eps` and `norm_eps` (equal).
    layer_norm_eps: Option<f64>,
    norm_eps: Option<f64>,
    pad_token_id: Option<usize>,
}
