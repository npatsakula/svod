//! JIT wrappers for the Qwen3 heads. Prepared at a concrete `[B, L]`: the
//! hand attention kernel takes no symbolic dimension, so batch and length are
//! plan shape and [`super::Qwen3Embedder`] keeps one plan per length bucket.

use svod_macros::jit_wrapper;

use super::embedder::Qwen3Embedding;
use super::reranker::Qwen3Reranker;

jit_wrapper! {
    Qwen3EmbeddingJit(Qwen3Embedding) {
        input_ids: Tensor,
        lengths: Tensor,

        outputs { embeddings }

        build(input_ids, lengths) {
            model.encode(input_ids, lengths)
        }
    }
}

jit_wrapper! {
    Qwen3RerankerJit(Qwen3Reranker) {
        input_ids: Tensor,
        lengths: Tensor,

        outputs { scores }

        build(input_ids, lengths) {
            model.forward(input_ids, lengths)
        }
    }
}
