//! ModernBERT — a RoPE pre-norm BERT encoder (`answerdotai/ModernBERT-{base,large}`).
//!
//! Pure-Rust port of the published FlexBERT backbone: token embeddings →
//! alternating global / sliding-window RoPE attention layers → final LayerNorm.
//! Computes in bf16 on the AMD target; f32 for CPU parity. Tokenization is
//! out-of-band — the API takes pre-computed `input_ids`.

mod attention;
mod classifier;
mod classifier_jit;
mod config;
mod embedder;
mod embedder_jit;
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
mod pipeline;
mod token_classifier;
mod token_classifier_jit;

pub use attention::ModernBertAttention;
#[cfg(test)]
pub(crate) use classifier::ClassifierHead;
#[cfg(test)]
pub(crate) use classifier::ModernBertClassificationModel;
pub use classifier::ModernBertClassifier;
pub use classifier_jit::ModernBertClassifierJit;
pub use config::{ClassifierPooling, ModernBertConfig};
pub use embedder::ModernBertEmbedder;
pub use embedder_jit::ModernBertEmbedderJit;
pub use encoder::Encoder;
pub use encoder_layer::EncoderLayer;
pub use error::{Error, Result};
pub use head::{MlmHead, ModernBertForMaskedLm};
pub use jit::{ModernBertJit, ModernBertMlmJit};
pub use mlp::ModernBertGlu;
pub use model::ModernBert;
pub use pipeline::{from_hub, from_hub_classifier, from_hub_token_classification};
#[cfg(test)]
pub(crate) use token_classifier::ModernBertTokenClassificationModel;
pub use token_classifier::ModernBertTokenClassifier;
pub use token_classifier_jit::ModernBertTokenClassifierJit;
