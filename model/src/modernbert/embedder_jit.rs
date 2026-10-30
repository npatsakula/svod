//! JIT wrapper for the ModernBERT **embedder**: backbone forward (with the
//! padding mask) + fused pooling (and L2-normalize, when the checkpoint asks),
//! compiled as ONE static JIT plan. `input_ids` `(B, L)` int64 + `attention_mask` `(B, L)` int64 → finished
//! `(B, D)` f32 embeddings.
//!
//! Distinct from [`super::jit::ModernBertJit`] (the bare backbone JIT, which
//! returns the `(B, L, D)` last-hidden-state). The mask is numerically
//! load-bearing for embeddings — masked attention is what keeps pad tokens out
//! of real-token representations (see the backbone parity test). Fusing
//! pool+norm in the same plan keeps the `(B, L, D)` activations on-device; only
//! the small `(B, D)` embeddings are read back (same rationale as gigaam's fused
//! encoder+head JIT).

use svod_macros::jit_wrapper;

use crate::encoder_head::head_jit;

use super::embedder::ModernBertEmbeddingModel;

jit_wrapper! {
    ModernBertEmbedderJit(ModernBertEmbeddingModel) {
        input_ids: Tensor,
        attention_mask: Tensor,

        outputs { embeddings }

        build(input_ids, attention_mask) {
            model.forward(input_ids, attention_mask)
        }
    }
}

head_jit!(ModernBertEmbedderJit, ModernBertEmbeddingModel, embeddings_to_vec);
