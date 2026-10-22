//! [`XlmRobertaEmbedder`] — finished embeddings from the XLM-RoBERTa backbone:
//! `input_ids` + `attention_mask` → pooled `(B, D)` vectors, pooled as the
//! checkpoint's sentence-transformers config says ([`Pooling`]). BGE-M3's
//! dense vector is one: the `<s>` state, L2-normalized.
//!
//! Implements `svod_arch::pipelines::text::Embed` so it drops straight into an
//! [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline); the
//! backbone and the pooling fuse into one plan per batch shape (see
//! [`HeadPlans`](crate::encoder_head::HeadPlans)).

use std::path::Path;

use snafu::ResultExt;
use svod_arch::pipelines::text::{Embed, Embedding, EncoderHead, Encoding, RunProfile};
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use super::config::XlmRobertaConfig;
use super::error::{HeadSnafu, Result};
use super::jit::XlmRobertaEmbedderJit;
use super::model::XlmRobertaModel;
use super::pipeline::load_weights;
use crate::encoder_head::{HeadError, HeadOptions, HeadPlans, Pooling, pool};
use crate::state::{StateDict, cast_all};

/// The backbone from a checkpoint's weights, at the root or under `roberta.`
/// (as a fine-tune saves it), each weight cast to `config.dtype`.
pub(crate) fn load_backbone(sd: &StateDict, config: XlmRobertaConfig) -> Result<XlmRobertaModel> {
    let sd = cast_all(sd, config.dtype.clone())?;
    let prefix = if sd.contains_key("roberta.embeddings.word_embeddings.weight") { "roberta" } else { "" };
    let mut model = XlmRobertaModel::empty(config);
    model.load_state_dict(&sd, prefix)?;
    Ok(model)
}

/// Backbone + pooling — the model type the embedder's JIT wraps. The backbone
/// loads from the root, as BGE-M3 and other `XLMRobertaModel` checkpoints
/// publish it.
#[derive(Clone, Module)]
pub(crate) struct XlmRobertaEmbeddingModel {
    #[module(key = "")]
    backbone: XlmRobertaModel,
    #[module(skip)]
    pooling: Pooling,
}

impl XlmRobertaEmbeddingModel {
    /// Fused forward: backbone → pool → finished `(B, D)` f32 embeddings.
    /// `mask`: `(B, L)`, non-zero = real token.
    pub(crate) fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        Ok(pool(&self.backbone.forward(input_ids, Some(mask))?, mask, self.pooling)?)
    }
}

/// Finished-embeddings model over an [`XlmRobertaModel`] backbone.
/// Construction is cheap; each batch shape compiles on first use, for the
/// device the embedder was built on.
pub struct XlmRobertaEmbedder {
    plans: HeadPlans<XlmRobertaEmbedderJit>,
    hidden_size: usize,
    pooling: Pooling,
}

impl XlmRobertaEmbedder {
    /// Build from a checkpoint's `config.json` and weights (`model.safetensors`
    /// or `pytorch_model.bin`; the backbone at the root, or under `roberta.` as
    /// a fine-tune saves it), pooling as `pooling` says
    /// ([`from_hub`](crate::xlm_roberta::from_hub) reads it from the
    /// checkpoint). `opts.max_seq` defaults to
    /// [`XlmRobertaConfig::max_sequence_length`].
    pub fn from_files(config_json: &Path, weights: &Path, pooling: Pooling, opts: &HeadOptions) -> Result<Self> {
        opts.build(|dtype| {
            let config = XlmRobertaConfig { dtype, ..XlmRobertaConfig::from_json(config_json)? };
            let max_seq = opts.max_seq.unwrap_or(config.max_sequence_length());
            let model = load_backbone(&load_weights(weights)?, config)?;
            Self::new(model, pooling, opts.max_batch, max_seq).context(HeadSnafu)
        })
    }

    /// Embed batches of up to `max_batch` chunks of up to `max_seq` tokens
    /// (specials included). `max_seq` may not exceed
    /// [`XlmRobertaConfig::max_sequence_length`]; padding takes the
    /// checkpoint's `pad_token_id`. Plans compile for the calling thread's
    /// default device, which should be the one the weights loaded onto.
    pub fn new(
        model: XlmRobertaModel,
        pooling: Pooling,
        max_batch: usize,
        max_seq: usize,
    ) -> std::result::Result<Self, HeadError> {
        let config = &model.config;
        let (hidden_size, max_positions, pad_id) =
            (config.hidden_size, config.max_sequence_length(), config.pad_token_id);
        let model = XlmRobertaEmbeddingModel { backbone: model, pooling };
        let plans = HeadPlans::new(model, max_batch, max_seq, max_positions, pad_id, "embed")?;
        Ok(Self { plans, hidden_size, pooling })
    }

    /// How each embedding is pooled.
    pub fn pooling(&self) -> Pooling {
        self.pooling
    }
}

impl EncoderHead for XlmRobertaEmbedder {
    type Output = Embedding;
    type Error = HeadError;

    fn capacity(&self) -> (usize, usize) {
        self.plans.capacity()
    }

    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> std::result::Result<(Vec<Embedding>, Option<RunProfile>), HeadError> {
        let run = self.plans.run(batch, profile)?;
        let embeddings =
            run.output.chunks_exact(self.hidden_size).map(|row| Embedding { values: row.to_vec() }).collect();
        Ok((embeddings, run.profile))
    }

    fn warm_up(&mut self, batch: usize, len: usize) -> std::result::Result<(), HeadError> {
        self.plans.warm_up(batch, len)
    }
}

impl Embed for XlmRobertaEmbedder {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn normalized(&self) -> bool {
        self.pooling.normalize
    }
}
