//! JIT wrapper for the ModernBERT **classifier**: backbone forward (with the
//! padding mask) + fused classification head (pool → dense → GELU → norm →
//! classifier linear), compiled as ONE static JIT plan. `input_ids` `(B, L)` int64 +
//! `attention_mask` `(B, L)` int64 → raw logits `(B, num_labels)` (f32).
//!
//! Mirrors [`super::embedder_jit::ModernBertEmbedderJit`] (which fuses
//! backbone + pool + L2-norm). The mask is numerically load-bearing for the
//! same reason — masked attention keeps pad tokens out of real-token
//! representations. Fusing the head keeps the `(B, L, D)` activations
//! on-device; only the small `(B, num_labels)` logits are read back.

use svod_macros::jit_wrapper;

use crate::encoder_head::head_jit;

use super::classifier::ModernBertClassificationModel;

jit_wrapper! {
    ModernBertClassifierJit(ModernBertClassificationModel) {
        input_ids: Tensor,
        attention_mask: Tensor,

        outputs { logits }

        build(input_ids, attention_mask) {
            model.forward(input_ids, attention_mask)
        }
    }
}

head_jit!(ModernBertClassifierJit, ModernBertClassificationModel, logits_to_vec);
