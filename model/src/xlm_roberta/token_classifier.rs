//! [`XlmRobertaTokenClassifier`] — token classification over the XLM-RoBERTa
//! backbone (HF `XLMRobertaForTokenClassification`; SaT's sentence-boundary
//! models): `input_ids` + `attention_mask` → per-token raw logits
//! `(seq_len, num_labels)` per chunk.
//!
//! Implements `svod_arch::pipelines::text::ClassifyTokens` so it drops straight
//! into an [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline).
//! The head is one `classifier` Linear over the full `(B, L, D)` state, fused
//! with the backbone into one plan per batch shape (see
//! [`HeadPlans`](crate::encoder_head::HeadPlans)).

use svod_arch::pipelines::text::{ClassifyTokens, EncoderHead, Encoding, RunProfile, TokenClassification};
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, Linear, Module};

use super::config::XlmRobertaConfig;
use super::error::Result;
use super::jit::XlmRobertaTokenClassifierJit;
use super::model::XlmRobertaModel;
use crate::encoder_head::{HeadError, HeadPlans};
use crate::init::{Bias, linear};
use crate::state::{StateDict, cast_all};

/// Backbone + token-classification head — the model type wrapped by the JIT.
/// Loads the published layout: the backbone under `roberta.`, the head under
/// `classifier.`.
#[derive(Clone, Module)]
pub(crate) struct XlmRobertaTokenClassificationModel {
    #[module(key = "roberta")]
    pub(crate) backbone: XlmRobertaModel,
    classifier: Linear,
}

impl XlmRobertaTokenClassificationModel {
    pub(crate) fn empty(config: XlmRobertaConfig) -> Self {
        let (d, n, dtype) = (config.hidden_size, config.num_labels, config.dtype.clone());
        Self { backbone: XlmRobertaModel::empty(config), classifier: linear(d, n, Bias::FanIn, dtype) }
    }

    /// Build from a preloaded state dict. Each weight is cast to `config.dtype`.
    pub(crate) fn from_state_dict(sd: &StateDict, config: XlmRobertaConfig) -> Result<Self> {
        let dtype = config.dtype.clone();
        let mut model = Self::empty(config);
        model.load_state_dict(&cast_all(sd, dtype)?, "")?;
        Ok(model)
    }

    /// Number of output labels (rows of the `classifier` weight).
    fn num_labels(&self) -> usize {
        self.classifier.weight.dim_const(0).expect("classifier weight is a concrete [num_labels, hidden]")
    }

    /// Fused forward: backbone → classifier → logits `(B, L, num_labels)` f32.
    /// `mask`: `(B, L)`, non-zero = real token.
    pub(crate) fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let hidden = self.backbone.forward(input_ids, Some(mask))?;
        Ok(self.classifier.forward(&hidden)?.cast(DType::Float32))
    }
}

/// Finished token-classifier model. Construction is cheap; each batch shape
/// compiles on first use. Implements [`EncoderHead`] (with [`ClassifyTokens`]
/// fixing the output kinds) for drop-in use with
/// [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline).
pub struct XlmRobertaTokenClassifier {
    plans: HeadPlans<XlmRobertaTokenClassifierJit>,
    num_labels: usize,
}

impl XlmRobertaTokenClassifier {
    /// Classify the tokens of batches of up to `max_batch` chunks of up to
    /// `max_seq` tokens (specials included). `max_seq` may not exceed
    /// [`XlmRobertaConfig::max_sequence_length`]; padding takes the
    /// checkpoint's `pad_token_id`.
    pub(crate) fn new(
        model: XlmRobertaTokenClassificationModel,
        max_batch: usize,
        max_seq: usize,
    ) -> std::result::Result<Self, HeadError> {
        let num_labels = model.num_labels();
        let config = &model.backbone.config;
        let (max_positions, pad_id) = (config.max_sequence_length(), config.pad_token_id);
        let plans = HeadPlans::new(model, max_batch, max_seq, max_positions, pad_id, "classify_tokens")?;
        Ok(Self { plans, num_labels })
    }
}

impl EncoderHead for XlmRobertaTokenClassifier {
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
        // Output is row-major `(b, len, num_labels)`: slice each chunk's live
        // tokens out of its `len`-wide row slab.
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

impl ClassifyTokens for XlmRobertaTokenClassifier {
    fn num_labels(&self) -> usize {
        self.num_labels
    }
}
