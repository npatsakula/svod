//! The JIT plan of one inference step ([`NemotronDiar::step`]): audio and the
//! previous step's input in, speaker probabilities and this step's input out.
//! Shapes are concrete — batch included: a symbolic batch keeps the
//! flash-attention kernel out and lets the optimizer merge it into the
//! sequence axis, off tensor cores.

use svod_macros::jit_wrapper;

use super::model::NemotronDiar;

jit_wrapper! {
    NemotronDiarStepJit(NemotronDiar) {
        framed: Tensor,
        mel_valid: Tensor,
        previous: Tensor,
        sources: Tensor,
        seq_lens: Tensor,
        key_lens: Tensor,

        outputs { probs, input }

        build(framed, mel_valid, previous, sources, seq_lens, key_lens) {
            model.step(framed, mel_valid, previous, sources, seq_lens, key_lens)
        }
    }
}
