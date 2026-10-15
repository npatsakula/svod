//! XLM-RoBERTa — multilingual transformer encoder backbone.
//!
//! Pure-Rust port of the XLM-RoBERTa-large architecture: absolute learned
//! position embeddings (with the fairseq `make_positions` offset), post-norm
//! encoder layers, separate biased Q/K/V/O projections.
//!
//! Consumed by [`crate::bgem3`] (BGE-M3 embedder + reranker), and carries a
//! token-classification head ([`XlmRobertaTokenClassifier`], e.g. SaT's
//! sentence segmenters, loaded by [`from_hub_token_classification`]).
//! Computes in bf16 on the AMD target; f32 for CPU parity. Tokenization is
//! out-of-band — the API takes pre-computed `input_ids`.

pub mod attention;
pub mod config;
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
pub use config::{XlmRobertaConfig, xlm_roberta_large};
pub use embeddings::XlmRobertaEmbeddings;
pub use encoder::XlmRobertaEncoder;
pub use encoder_layer::EncoderLayer;
pub use error::{Error, Result};
pub use feed_forward::FeedForwardWeights;
pub use jit::{XlmRobertaJit, XlmRobertaTokenClassifierJit};
pub use model::XlmRobertaModel;
pub use pipeline::{XlmRobertaTokenClassifierLoad, from_hub_token_classification};
pub use position_ids::position_ids_from_input_ids;
#[cfg(test)]
pub(crate) use token_classifier::XlmRobertaTokenClassificationModel;
pub use token_classifier::XlmRobertaTokenClassifier;
