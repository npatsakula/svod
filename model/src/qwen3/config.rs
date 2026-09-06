//! Qwen3 configuration, parsed from HuggingFace `config.json`.
//!
//! Mirrors the published schema of `Qwen/Qwen3-Embedding-0.6B`. The clean
//! [`Qwen3Config`] keeps only the fields the Rust backbone consumes and adds
//! a caller-chosen compute [`DType`] (defaults to [`crate::default_compute_dtype`];
//! f32 for CPU parity tests).
//!
//! Key detail: `head_dim` is **explicit** and may differ from
//! `hidden_size / num_attention_heads`. For the 0.6B checkpoint,
//! `head_dim = 128` but `hidden_size / num_heads = 1024 / 16 = 64`.
//! The Q projection outputs `num_heads * head_dim = 2048`, not `hidden_size`.

use std::path::Path;

use serde::Deserialize;
use svod_dtype::DType;

use super::error::{Error, Result};

#[derive(Clone, Debug)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Explicit head dimension — NOT derived from `hidden_size / num_heads`.
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
    pub pad_token_id: usize,
    pub dtype: DType,
    pub max_batch_size: usize,
}

impl Qwen3Config {
    /// Number of Q heads per KV head (GQA group size).
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn merge_structural_from(&mut self, parsed: &Self) {
        self.vocab_size = parsed.vocab_size;
        self.hidden_size = parsed.hidden_size;
        self.num_hidden_layers = parsed.num_hidden_layers;
        self.num_attention_heads = parsed.num_attention_heads;
        self.num_key_value_heads = parsed.num_key_value_heads;
        self.head_dim = parsed.head_dim;
        self.intermediate_size = parsed.intermediate_size;
        self.max_position_embeddings = parsed.max_position_embeddings;
        self.rms_norm_eps = parsed.rms_norm_eps;
        self.rope_theta = parsed.rope_theta;
        self.attention_bias = parsed.attention_bias;
        self.tie_word_embeddings = parsed.tie_word_embeddings;
        self.pad_token_id = parsed.pad_token_id;
    }
}

/// `Qwen/Qwen3-Embedding-0.6B`: 28 layers, hidden 1024, 16 Q heads / 8 KV heads,
/// head_dim 128, intermediate 3072, vocab 151669.
pub fn qwen3_embedding_0_6b() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 151_669,
        hidden_size: 1024,
        num_hidden_layers: 28,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        head_dim: 128,
        intermediate_size: 3072,
        max_position_embeddings: 32_768,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        attention_bias: false,
        tie_word_embeddings: true,
        pad_token_id: 151_643,
        dtype: crate::default_compute_dtype(),
        max_batch_size: 1,
    }
}

impl Qwen3Config {
    pub fn from_json(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| Error::Config { message: format!("reading config.json: {e}") })?;
        Self::from_json_str(&data)
    }

    pub fn from_json_str(data: &str) -> Result<Self> {
        let raw: RawQwen3Config =
            serde_json::from_str(data).map_err(|e| Error::Config { message: format!("JSON parse error: {e}") })?;
        Ok(Self::from_raw(raw))
    }

    fn from_raw(raw: RawQwen3Config) -> Self {
        let base = qwen3_embedding_0_6b();
        let head_dim = raw.head_dim.unwrap_or_else(|| {
            raw.hidden_size.unwrap_or(base.hidden_size) / raw.num_attention_heads.unwrap_or(base.num_attention_heads)
        });
        Qwen3Config {
            vocab_size: raw.vocab_size.unwrap_or(base.vocab_size),
            hidden_size: raw.hidden_size.unwrap_or(base.hidden_size),
            num_hidden_layers: raw.num_hidden_layers.unwrap_or(base.num_hidden_layers),
            num_attention_heads: raw.num_attention_heads.unwrap_or(base.num_attention_heads),
            num_key_value_heads: raw.num_key_value_heads.unwrap_or(base.num_key_value_heads),
            head_dim,
            intermediate_size: raw.intermediate_size.unwrap_or(base.intermediate_size),
            max_position_embeddings: raw.max_position_embeddings.unwrap_or(base.max_position_embeddings),
            rms_norm_eps: raw.rms_norm_eps.unwrap_or(base.rms_norm_eps),
            rope_theta: raw.rope_theta.unwrap_or(base.rope_theta),
            attention_bias: raw.attention_bias.unwrap_or(base.attention_bias),
            tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(base.tie_word_embeddings),
            pad_token_id: raw.pad_token_id.unwrap_or(base.pad_token_id),
            dtype: base.dtype,
            max_batch_size: base.max_batch_size,
        }
    }
}

#[derive(Deserialize)]
struct RawQwen3Config {
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    intermediate_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    attention_bias: Option<bool>,
    tie_word_embeddings: Option<bool>,
    pad_token_id: Option<usize>,
}
