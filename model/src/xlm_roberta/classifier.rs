//! [`XlmRobertaClassifier`] — sequence classification over the XLM-RoBERTa
//! backbone (HF `XLMRobertaForSequenceClassification`): `input_ids` +
//! `attention_mask` → raw class logits `(B, num_labels)`.
//!
//! A cross-encoder reranker is one: `BAAI/bge-reranker-v2-m3` gives one logit
//! for a (query, passage) pair, which
//! [`classify_pair`](svod_arch::pipelines::text::EncoderPipeline::classify_pair)
//! encodes as `<s> query </s></s> passage </s>`.
//!
//! The head is HF's `RobertaClassificationHead` on the `<s>` state, fused with
//! the backbone into one plan per batch shape (see
//! [`HeadPlans`](crate::encoder_head::HeadPlans)).

use std::path::Path;

use snafu::ResultExt;
use svod_arch::pipelines::text::{Classification, Classify, EncoderHead, Encoding, RunProfile};
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use super::config::XlmRobertaConfig;
use super::error::{HeadSnafu, Result};
use super::jit::XlmRobertaClassifierJit;
use super::model::XlmRobertaModel;
use super::pipeline::load_weights;
use crate::encoder_head::{HeadError, HeadOptions, HeadPlans};
use crate::init::{fan_in_uniform, zeros};
use crate::state::{StateDict, cast_all};

/// HF's `RobertaClassificationHead`: `<s> → dense(D, D) → tanh →
/// out_proj(D, num_labels)`.
#[derive(Clone, Module)]
pub(crate) struct ClassificationHead {
    #[module(key = "dense.weight")]
    dense_weight: Tensor,
    #[module(key = "dense.bias")]
    dense_bias: Tensor,
    #[module(key = "out_proj.weight")]
    out_proj_weight: Tensor,
    #[module(key = "out_proj.bias")]
    out_proj_bias: Tensor,
}

impl ClassificationHead {
    fn empty(hidden_size: usize, num_labels: usize, dtype: DType) -> Self {
        Self {
            dense_weight: fan_in_uniform(&[hidden_size, hidden_size], hidden_size, dtype.clone()),
            dense_bias: zeros(&[hidden_size], dtype.clone()),
            out_proj_weight: fan_in_uniform(&[num_labels, hidden_size], hidden_size, dtype.clone()),
            out_proj_bias: zeros(&[num_labels], dtype),
        }
    }

    /// Number of output labels (rows of the `out_proj` weight).
    fn num_labels(&self) -> usize {
        self.out_proj_weight.dim_const(0).expect("out_proj weight is a concrete [num_labels, hidden]")
    }

    /// `hidden` `(B, L, D)` → logits `(B, num_labels)` from the `<s>` state.
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let cls = hidden.take_index(1, 0)?;
        let h = cls.linear().weight(&self.dense_weight).bias(&self.dense_bias).call()?.tanh()?;
        Ok(h.linear().weight(&self.out_proj_weight).bias(&self.out_proj_bias).call()?)
    }
}

/// Backbone + classification head — the model type wrapped by the JIT. Loads
/// the published layout: the backbone under `roberta.`, the head under
/// `classifier.`.
#[derive(Clone, Module)]
pub(crate) struct XlmRobertaClassificationModel {
    #[module(key = "roberta")]
    pub(crate) backbone: XlmRobertaModel,
    classifier: ClassificationHead,
}

impl XlmRobertaClassificationModel {
    pub(crate) fn empty(config: XlmRobertaConfig) -> Self {
        let (d, n, dtype) = (config.hidden_size, config.num_labels, config.dtype.clone());
        Self { backbone: XlmRobertaModel::empty(config), classifier: ClassificationHead::empty(d, n, dtype) }
    }

    /// Build from a preloaded state dict. Each weight is cast to `config.dtype`.
    pub(crate) fn from_state_dict(sd: &StateDict, config: XlmRobertaConfig) -> Result<Self> {
        let dtype = config.dtype.clone();
        let mut model = Self::empty(config);
        model.load_state_dict(&cast_all(sd, dtype)?, "")?;
        Ok(model)
    }

    /// Fused forward: backbone → head → logits `(B, num_labels)` f32. `mask`:
    /// `(B, L)`, non-zero = real token.
    pub(crate) fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let hidden = self.backbone.forward(input_ids, Some(mask))?;
        Ok(self.classifier.forward(&hidden)?.cast(DType::Float32))
    }
}

/// Finished sequence-classifier model. Construction is cheap; each batch shape
/// compiles on first use, for the device the classifier was built on.
/// Implements [`EncoderHead`] (with [`Classify`] fixing the output kinds) for
/// drop-in use with
/// [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline).
pub struct XlmRobertaClassifier {
    plans: HeadPlans<XlmRobertaClassifierJit>,
    num_labels: usize,
    id2label: Vec<String>,
}

impl XlmRobertaClassifier {
    /// Build from a checkpoint's `config.json` and weights (`model.safetensors`
    /// or `pytorch_model.bin`). `opts.max_seq` defaults to
    /// [`XlmRobertaConfig::max_sequence_length`].
    pub fn from_files(config_json: &Path, weights: &Path, opts: &HeadOptions) -> Result<Self> {
        opts.build(|dtype| {
            let config = XlmRobertaConfig { dtype, ..XlmRobertaConfig::from_json(config_json)? };
            let max_seq = opts.max_seq.unwrap_or(config.max_sequence_length());
            let model = XlmRobertaClassificationModel::from_state_dict(&load_weights(weights)?, config)?;
            Self::new(model, opts.max_batch, max_seq).context(HeadSnafu)
        })
    }

    /// Classify batches of up to `max_batch` chunks of up to `max_seq` tokens
    /// (see [`XlmRobertaEmbedder::new`](super::XlmRobertaEmbedder::new)).
    pub(crate) fn new(
        model: XlmRobertaClassificationModel,
        max_batch: usize,
        max_seq: usize,
    ) -> std::result::Result<Self, HeadError> {
        let num_labels = model.classifier.num_labels();
        let config = &model.backbone.config;
        let (max_positions, pad_id, id2label) =
            (config.max_sequence_length(), config.pad_token_id, config.id2label.clone());
        let plans = HeadPlans::new(model, max_batch, max_seq, max_positions, pad_id, "classify")?;
        Ok(Self { plans, num_labels, id2label })
    }

    /// Label names by id, from `config.json`'s `id2label` (empty when it has
    /// none).
    pub fn id2label(&self) -> &[String] {
        &self.id2label
    }
}

impl EncoderHead for XlmRobertaClassifier {
    type Output = Classification;
    type Error = HeadError;

    fn capacity(&self) -> (usize, usize) {
        self.plans.capacity()
    }

    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> std::result::Result<(Vec<Classification>, Option<RunProfile>), HeadError> {
        let run = self.plans.run(batch, profile)?;
        let classifications =
            run.output.chunks_exact(self.num_labels).map(|row| Classification { logits: row.to_vec() }).collect();
        Ok((classifications, run.profile))
    }

    fn warm_up(&mut self, batch: usize, len: usize) -> std::result::Result<(), HeadError> {
        self.plans.warm_up(batch, len)
    }
}

impl Classify for XlmRobertaClassifier {
    fn num_labels(&self) -> usize {
        self.num_labels
    }
}
