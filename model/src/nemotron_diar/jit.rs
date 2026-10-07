//! The JIT plan of one inference step ([`NemotronDiar::step`]): audio and the
//! previous step's input in, speaker probabilities and this step's input out.
//! The batch is a bound variable — the sessions a step holds — and every other
//! shape is concrete.

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

        batch_var b: (1, model.config.max_batch),
        outputs { probs, input }

        build(framed, mel_valid, previous, sources, seq_lens, key_lens) {
            model.step(framed, mel_valid, previous, sources, seq_lens, key_lens)
        }
    }
}
