//! JIT wrapper for the ModernBERT **token classifier**: backbone forward (with
//! the padding mask) + fused token head (the prediction head over the full
//! `(B, L, D)` state — no pooling), compiled as ONE static JIT plan. `input_ids`
//! `(B, L)` int64 + `attention_mask` `(B, L)` int64 → per-token logits
//! `(B, L, num_labels)` (f32).
//!
//! Mirrors [`super::classifier_jit::ModernBertClassifierJit`] (which fuses
//! backbone + pool + head). The mask is numerically load-bearing for backbone
//! attention; the head itself never pools. Fusing keeps the `(B, L, D)`
//! activations on-device; only the `(B, L, num_labels)` logits are read back.

use svod_macros::jit_wrapper;

use super::token_classifier::ModernBertTokenClassificationModel;

jit_wrapper! {
    ModernBertTokenClassifierJit(ModernBertTokenClassificationModel) {
        input_ids: Tensor,
        attention_mask: Tensor,

        outputs { logits }

        build(input_ids, attention_mask) {
            model.forward(input_ids, attention_mask)
        }
    }
}
