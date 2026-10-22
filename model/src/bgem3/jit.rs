//! JIT wrapper for BGE-M3's ColBERT vectors; its dense vector is
//! [`XlmRobertaEmbedder`](crate::xlm_roberta::XlmRobertaEmbedder)'s.

use svod_macros::jit_wrapper;

use super::embedder::BgeM3;

jit_wrapper! {
    BgeM3ColbertJit(BgeM3) {
        input_ids: Tensor,
        attention_mask: Tensor,

        batch_var b: (1, model.model.config.max_batch_size),
        outputs { colbert }

        build(input_ids, attention_mask) {
            model.encode_colbert(input_ids, attention_mask)
        }
    }
}
