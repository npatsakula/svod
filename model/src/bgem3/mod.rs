//! BGE-M3 — multilingual embedding model (`BAAI/bge-m3`).
//!
//! Built on the [`crate::xlm_roberta`] backbone. Three retrieval heads:
//!
//! - **Dense**: CLS pooling + L2 normalize → `(B, D)`
//! - **Sparse**: Linear → ReLU → scatter-to-vocab → `(B, vocab_size)`
//! - **ColBERT**: per-token Linear (skip CLS) + L2 normalize → `(B, L-1, Dc)`
//!
//! The dense vector alone runs in an encoder pipeline as
//! [`XlmRobertaEmbedder`](crate::xlm_roberta::XlmRobertaEmbedder) (loaded by
//! [`xlm_roberta::from_hub`](crate::xlm_roberta::from_hub)), and the
//! `BAAI/bge-reranker-v2-m3` cross-encoder as
//! [`XlmRobertaClassifier`](crate::xlm_roberta::XlmRobertaClassifier).

mod colbert_head;
mod embedder;
mod jit;
mod scoring;
mod sparse_head;

pub use colbert_head::ColbertHead;
pub use embedder::{BgeM3, BgeM3Output, EncodeOpts};
pub use jit::BgeM3ColbertJit;
pub use scoring::{colbert_score, dense_score, hybrid_score, sparse_score};
pub use sparse_head::SparseHead;
