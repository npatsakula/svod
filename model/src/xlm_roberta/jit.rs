//! JIT wrappers for [`super::model::XlmRobertaModel`] and the token classifier.
//! `XlmRobertaJit` returns the `(B, L, D)` last-hidden-state under a rebindable
//! batch; `XlmRobertaTokenClassifierJit` is a static plan fusing the backbone
//! and the token head, reading back only the `(B, L, num_labels)` logits.

use svod_macros::jit_wrapper;

use super::model::XlmRobertaModel;
use super::token_classifier::XlmRobertaTokenClassificationModel;
use crate::encoder_head::head_jit;

jit_wrapper! {
    XlmRobertaJit(XlmRobertaModel) {
        input_ids: Tensor,
        attention_mask: Tensor,

        batch_var b: (1, model.config.max_batch_size),
        outputs { hidden }

        build(input_ids, attention_mask) {
            model.forward(input_ids, Some(attention_mask))
        }
    }
}

jit_wrapper! {
    XlmRobertaTokenClassifierJit(XlmRobertaTokenClassificationModel) {
        input_ids: Tensor,
        attention_mask: Tensor,

        outputs { logits }

        build(input_ids, attention_mask) {
            model.forward(input_ids, attention_mask)
        }
    }
}

head_jit!(XlmRobertaTokenClassifierJit, XlmRobertaTokenClassificationModel, logits_to_vec);
