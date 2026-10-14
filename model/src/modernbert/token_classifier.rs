//! [`ModernBertTokenClassifier`] — token classification over the ModernBERT
//! backbone (NER, POS tagging, chunking, …): `input_ids` + `attention_mask` →
//! per-token raw logits `(seq_len, num_labels)` per chunk.
//!
//! Implements `svod_arch::pipelines::text::ClassifyTokens` so it drops straight into
//! an [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline). The
//! model owns the forward + fused token head (via
//! [`ModernBertTokenClassifierJit`]); the pipeline owns chunking, profile
//! assembly, and span decoding (`labels_for_tokens` / `group_spans`).
//!
//! The token head is HF's `ModernBertPredictionHead` + `classifier` applied to
//! the full `(B, L, D)` last-hidden-state — no pooling, so every position gets a
//! logits row. It reuses the sequence-classification head
//! ([`ClassifierHead`]); the two heads differ only in pooling.

use svod_arch::pipelines::text::{ClassifyTokens, EncoderHead, Encoding, RunProfile, TokenClassification};
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::modernbert::classifier::ClassifierHead;
use crate::modernbert::config::ModernBertConfig;
use crate::modernbert::error::Result;
use crate::modernbert::head_jit::{HeadError, HeadPlans};
use crate::modernbert::model::ModernBert;
use crate::modernbert::token_classifier_jit::ModernBertTokenClassifierJit;
use crate::state::{self, StateDict};

// ─── composite model (backbone + head) ─────────────────────────────────────

/// Backbone + token-classification head — the model type wrapped by the JIT.
/// [`Self::forward`] fuses backbone → [`ClassifierHead::forward`] over the full
/// `(B, L, D)` hidden state into one graph. Unlike
/// [`ModernBertClassificationModel`](super::classifier::ModernBertClassificationModel)
/// there is no pooling — every token position gets a logits row.
#[derive(Clone, Module)]
pub(crate) struct ModernBertTokenClassificationModel {
    #[module(key = "")]
    pub(crate) backbone: ModernBert,
    #[module(key = "")]
    head: ClassifierHead,
}

impl ModernBertTokenClassificationModel {
    pub(crate) fn empty(config: &ModernBertConfig) -> Self {
        Self { backbone: ModernBert::empty(config.clone()), head: ClassifierHead::empty(config) }
    }

    /// Build from a preloaded state dict. Each weight is cast to `config.dtype`.
    pub(crate) fn from_state_dict(sd: &StateDict, config: &ModernBertConfig) -> Result<Self> {
        let mut model = Self::empty(config);
        model.load_state_dict(&state::cast_all(sd, config.dtype.clone())?, "")?;
        Ok(model)
    }

    /// Fused forward: backbone → prediction head → logits `(B, L, num_labels)`
    /// f32. `mask`: `(B, L)`, non-zero = real token — load-bearing for backbone
    /// attention; the head itself doesn't pool, so it consumes no mask.
    pub(crate) fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        self.head.forward(&self.backbone.forward(input_ids, Some(mask))?)
    }
}

// ─── runtime (owns JIT, impl EncoderHead + ClassifyTokens) ──────────────────────────

/// Finished token-classifier model. Construction is cheap; each batch shape
/// compiles on first use. Implements [`EncoderHead`] (with [`ClassifyTokens`]
/// fixing the output kinds) for drop-in use with
/// [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline).
pub struct ModernBertTokenClassifier {
    plans: HeadPlans<ModernBertTokenClassifierJit>,
    num_labels: usize,
}

impl ModernBertTokenClassifier {
    /// Classify the tokens of batches of up to `max_batch` chunks of up to
    /// `max_seq` tokens (see
    /// [`ModernBertEmbedder::new`](super::ModernBertEmbedder::new)).
    pub(crate) fn new(
        model: ModernBertTokenClassificationModel,
        max_batch: usize,
        max_seq: usize,
    ) -> std::result::Result<Self, HeadError> {
        let num_labels = model.head.num_labels();
        let max_positions = model.backbone.config.max_position_embeddings;
        Ok(Self { plans: HeadPlans::new(model, max_batch, max_seq, max_positions, "classify_tokens")?, num_labels })
    }
}

impl EncoderHead for ModernBertTokenClassifier {
    type Output = TokenClassification;
    type Error = HeadError;

    fn capacity(&self) -> (usize, usize) {
        self.plans.capacity()
    }

    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> std::result::Result<(Vec<TokenClassification>, Option<RunProfile>), HeadError> {
        let run = self.plans.run(batch, profile)?;
        // Output is row-major `(b, len, num_labels)`; each chunk occupies a
        // `len`-wide row slab. Slice to each chunk's live token count so the
        // returned grid is `(seq_len, num_labels)` — padding positions dropped.
        let (nl, stride) = (self.num_labels, run.len * self.num_labels);
        let results = batch
            .iter()
            .enumerate()
            .map(|(i, enc)| TokenClassification {
                logits: run.output[i * stride..i * stride + enc.input_ids.len() * nl].to_vec(),
                num_labels: nl,
            })
            .collect();
        Ok((results, run.profile))
    }
}

impl ClassifyTokens for ModernBertTokenClassifier {
    fn num_labels(&self) -> usize {
        self.num_labels
    }
}
