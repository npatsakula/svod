//! [`ModernBertEmbedder`] — finished embeddings from the ModernBERT backbone:
//! `input_ids` + `attention_mask` → pooled `(B, D)` vectors, pooled as the
//! checkpoint's sentence-transformers config says ([`Pooling`]; mean, then L2,
//! without one).
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

use std::path::Path;

use snafu::ResultExt;
use svod_arch::pipelines::text::{Embed, Embedding, EncoderHead, Encoding, RunProfile};
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::encoder_head::{HeadError, HeadOptions, HeadPlans, Pooling, pool};
use crate::modernbert::ModernBert;
use crate::modernbert::embedder_jit::ModernBertEmbedderJit;
use crate::modernbert::error::{HeadSnafu, Result as ModelResult};
use crate::modernbert::pipeline::load_checkpoint;

/// Backbone + pooling — the model type the embedder's JIT wraps.
#[derive(Clone, Module)]
pub(crate) struct ModernBertEmbeddingModel {
    #[module(key = "")]
    backbone: ModernBert,
    #[module(skip)]
    pooling: Pooling,
}

impl ModernBertEmbeddingModel {
    /// Fused forward: backbone → pool → finished `(B, D)` f32 embeddings.
    /// `mask`: `(B, L)`, non-zero = real token.
    pub(crate) fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> ModelResult<Tensor> {
        Ok(pool(&self.backbone.forward(input_ids, Some(mask))?, mask, self.pooling)?)
    }
}

/// Finished-embeddings model over a [`ModernBert`] backbone. Construction is
/// cheap; each batch shape compiles on first use, for the device the embedder
/// was built on.
pub struct ModernBertEmbedder {
    plans: HeadPlans<ModernBertEmbedderJit>,
    hidden_size: usize,
    pooling: Pooling,
}

impl ModernBertEmbedder {
    /// Build from a checkpoint's `config.json` and `model.safetensors`, pooling
    /// as `pooling` says ([`from_hub`](crate::modernbert::from_hub) reads it
    /// from the checkpoint). `opts.max_seq` defaults to the checkpoint's
    /// `max_position_embeddings`.
    pub fn from_files(config_json: &Path, weights: &Path, pooling: Pooling, opts: &HeadOptions) -> ModelResult<Self> {
        opts.build(|dtype| {
            let (config, sd, max_seq) = load_checkpoint(config_json, weights, dtype, opts.max_seq)?;
            let model = ModernBert::from_state_dict(&sd, config)?;
            Self::new(model, pooling, opts.max_batch, max_seq).context(HeadSnafu)
        })
    }

    /// Embed batches of up to `max_batch` chunks of up to `max_seq` tokens
    /// (specials included). `max_batch`/`max_seq` are caller-chosen and
    /// typically flow in from the pipeline (the chunker's `max_seq` at
    /// assembly); `max_seq` may not exceed the checkpoint's
    /// `max_position_embeddings`. The model's config must already reflect the
    /// checkpoint (e.g. via [`ModernBert::from_hub`]). Plans compile for the
    /// calling thread's default device, which should be the one the model's
    /// weights loaded onto.
    pub fn new(model: ModernBert, pooling: Pooling, max_batch: usize, max_seq: usize) -> Result<Self, HeadError> {
        let config = &model.config;
        let (hidden_size, max_positions, pad_id) =
            (config.hidden_size, config.max_position_embeddings, config.pad_token_id);
        let model = ModernBertEmbeddingModel { backbone: model, pooling };
        let plans = HeadPlans::new(model, max_batch, max_seq, max_positions, pad_id, "embed")?;
        Ok(Self { plans, hidden_size, pooling })
    }

    /// How each embedding is pooled.
    pub fn pooling(&self) -> Pooling {
        self.pooling
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

    fn warm_up(&mut self, batch: usize, len: usize) -> Result<(), HeadError> {
        self.plans.warm_up(batch, len)
    }
}

impl Embed for ModernBertEmbedder {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn normalized(&self) -> bool {
        self.pooling.normalize
    }
}
