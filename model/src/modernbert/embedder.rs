//! [`ModernBertEmbedder`] — finished embeddings from the ModernBERT backbone:
//! `input_ids` + `attention_mask` → mean-pooled, L2-normalized `(B, D)` vectors.
//!
//! Implements [`EncoderHead`] (with [`Embed`] fixing the output kinds) so it drops
//! straight into an
//! [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline). The model
//! owns the forward + fused pooling/normalization (via
//! [`ModernBertEmbedderJit`]); the pipeline owns chunking and profile assembly.
//!
//! For the one-call hub loader that builds this embedder alongside a matching
//! [`HfTokenizer`](svod_arch::pipelines::text::HfTokenizer), see
//! [`from_hub`](crate::modernbert::from_hub).
//!
//! Each batch runs on a static plan sized to it (see
//! [`HeadPlans`](crate::encoder_head::HeadPlans)), and the live rows are
//! sliced out of the plan's `[rows, D]` output.

use svod_arch::pipelines::text::{Embed, Embedding, EncoderHead, Encoding, RunProfile};
use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::encoder_head::{HeadError, HeadPlans};
use crate::modernbert::ModernBert;
use crate::modernbert::embedder_jit::ModernBertEmbedderJit;
use crate::modernbert::error::Result as ModelResult;

/// Masked mean-pool + L2-normalize → finished `(B, D)` f32 embeddings (the
/// sentence-transformers recipe for ModernBERT). `hidden`: `(B, L, D)`;
/// `attention_mask`: `(B, L)`, non-zero = real token. An all-pad row pools to
/// the zero vector and stays zero. The mean runs in f32, as the reference
/// pools: a bf16 mean rounds the pooled vector itself, an error no averaging
/// over tokens washes out.
pub(crate) fn pool_embed(hidden: &Tensor, attention_mask: &Tensor) -> ModelResult<Tensor> {
    Ok(hidden.cast(DType::Float32).masked_mean(attention_mask, 1)?.lp_normalize(-1, 2)?)
}

/// Finished-embeddings model over a [`ModernBert`] backbone. Construction is
/// cheap; each batch shape compiles on first use.
pub struct ModernBertEmbedder {
    plans: HeadPlans<ModernBertEmbedderJit>,
    hidden_size: usize,
}

impl ModernBertEmbedder {
    /// Embed batches of up to `max_batch` chunks of up to `max_seq` tokens
    /// (specials included). `max_batch`/`max_seq` are caller-chosen and
    /// typically flow in from the pipeline (the chunker's `max_seq` at
    /// assembly); `max_seq` may not exceed the checkpoint's
    /// `max_position_embeddings`. The model's config must already reflect the
    /// checkpoint (e.g. via [`ModernBert::from_hub`]).
    pub fn new(model: ModernBert, max_batch: usize, max_seq: usize) -> Result<Self, HeadError> {
        let config = &model.config;
        let (hidden_size, max_positions, pad_id) =
            (config.hidden_size, config.max_position_embeddings, config.pad_token_id);
        Ok(Self { plans: HeadPlans::new(model, max_batch, max_seq, max_positions, pad_id, "embed")?, hidden_size })
    }
}

impl EncoderHead for ModernBertEmbedder {
    type Output = Embedding;
    type Error = HeadError;

    fn capacity(&self) -> (usize, usize) {
        self.plans.capacity()
    }

    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> Result<(Vec<Embedding>, Option<RunProfile>), HeadError> {
        let run = self.plans.run(batch, profile)?;
        let embeddings =
            run.output.chunks_exact(self.hidden_size).map(|row| Embedding { values: row.to_vec() }).collect();
        Ok((embeddings, run.profile))
    }
}

impl Embed for ModernBertEmbedder {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}
