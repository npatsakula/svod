//! ModernBERT — a RoPE pre-norm BERT encoder (`answerdotai/ModernBERT-{base,large}`).
//!
//! Pure-Rust port of the published FlexBERT backbone: token embeddings →
//! alternating global / sliding-window RoPE attention layers → final LayerNorm.
//! Computes in bf16 on the AMD target; f32 for CPU parity. Tokenization is
//! out-of-band — the API takes pre-computed `input_ids`.

mod attention;
mod config;
mod embeddings;
mod encoder;
mod encoder_layer;
mod error;
mod head;
mod jit;
mod linear;
mod mlp;
mod model;
mod norm;

pub use attention::ModernBertAttention;
pub use config::ModernBertConfig;
pub use encoder::Encoder;
pub use encoder_layer::EncoderLayer;
pub use error::{Error, Result};
pub use head::{MlmHead, ModernBertForMaskedLm};
pub use jit::{ModernBertJit, ModernBertMlmJit};
pub use mlp::ModernBertGlu;
pub use model::ModernBert;
