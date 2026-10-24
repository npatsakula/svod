//! XLM-RoBERTa — multilingual transformer encoder backbone.
//!
//! Pure-Rust port of the XLM-RoBERTa-large architecture: absolute learned
//! position embeddings (with the fairseq `make_positions` offset), post-norm
//! encoder layers, separate biased Q/K/V/O projections.
//!
//! Consumed by [`crate::bgem3`] (BGE-M3's dense, sparse and ColBERT heads),
//! and carries the encoder-pipeline heads: [`XlmRobertaEmbedder`] (BGE-M3's
//! dense vector, by [`from_hub`]), [`XlmRobertaClassifier`] (sequence
//! classification and cross-encoder rerankers such as
//! `BAAI/bge-reranker-v2-m3`, by [`from_hub_classifier`]) and
//! [`XlmRobertaTokenClassifier`] (e.g. SaT's sentence segmenters, by
//! [`from_hub_token_classification`]).
//! Computes in bf16 on the AMD target; f32 for CPU parity. Tokenization is
//! out-of-band — the API takes pre-computed `input_ids`.

pub mod attention;
mod classifier;
pub mod config;
mod embedder;
pub mod embeddings;
pub mod encoder;
pub mod encoder_layer;
pub mod error;
pub mod feed_forward;
pub mod jit;
pub mod model;
mod pipeline;
pub mod position_ids;
mod token_classifier;

pub use attention::XlmRobertaAttention;
#[cfg(test)]
pub(crate) use classifier::XlmRobertaClassificationModel;
pub use classifier::XlmRobertaClassifier;
pub use config::{XlmRobertaConfig, xlm_roberta_large};
pub use embedder::XlmRobertaEmbedder;
#[cfg(test)]
pub(crate) use embedder::load_backbone;
pub use embeddings::XlmRobertaEmbeddings;
pub use encoder::XlmRobertaEncoder;
pub use encoder_layer::EncoderLayer;
pub use error::{Error, Result};
pub use feed_forward::FeedForwardWeights;
pub use jit::{XlmRobertaClassifierJit, XlmRobertaEmbedderJit, XlmRobertaJit, XlmRobertaTokenClassifierJit};
pub use model::XlmRobertaModel;
pub use pipeline::{from_hub, from_hub_classifier, from_hub_token_classification};
pub use position_ids::position_ids_from_input_ids;
#[cfg(test)]
pub(crate) use token_classifier::XlmRobertaTokenClassificationModel;
pub use token_classifier::XlmRobertaTokenClassifier;
