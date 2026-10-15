//! JIT wrapper for the ModernBERT **embedder**: backbone forward (with the
//! padding mask) + fused masked mean-pool + L2-normalize, compiled as ONE static
//! JIT plan. `input_ids` `(B, L)` int64 + `attention_mask` `(B, L)` int64 → finished
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

use super::embedder::pool_embed;
use super::model::ModernBert;

jit_wrapper! {
    ModernBertEmbedderJit(ModernBert) {
        input_ids: Tensor,
        attention_mask: Tensor,

        outputs { embeddings }

        build(input_ids, attention_mask) {
            pool_embed(&model.forward(input_ids, Some(attention_mask))?, attention_mask)
        }
    }
}

head_jit!(ModernBertEmbedderJit, ModernBert, embeddings_to_vec);
