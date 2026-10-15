//! [`ModernBertClassifier`] — sequence classification over the ModernBERT
//! backbone: `input_ids` + `attention_mask` → raw class logits `(B, num_labels)`.
//!
//! Implements `svod_arch::pipelines::text::Classify` so it drops straight into an
//! [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline). The model
//! owns the forward + fused classification head (via
//! [`ModernBertClassifierJit`]); the pipeline owns chunking and profile
//! assembly.
//!
//! The classification head mirrors HF's `ModernBertForSequenceClassification`:
//! pool (cls or mean) → `head.dense` → GELU → `head.norm` → `classifier` linear.
//! Fused into one JIT plan so the `(B, L, D)` activations stay on-device.

use svod_arch::pipelines::text::{Classification, Classify, EncoderHead, Encoding, RunProfile};
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{LayerNorm, Linear, Module};
use svod_tk::ops::{self, Act};

use crate::encoder_head::{HeadError, HeadPlans};
use crate::init::{Bias, linear};
use crate::modernbert::classifier_jit::ModernBertClassifierJit;
use crate::modernbert::config::{ClassifierPooling, ModernBertConfig};
use crate::modernbert::error::Result;
use crate::modernbert::model::ModernBert;
use crate::state::{self, StateDict};

// ─── head weights ──────────────────────────────────────────────────────────

/// Classification head weights: HF `head.dense` + `head.norm` + `classifier`.
/// `head.dense` carries a bias only when `classifier_bias = true` (ModernBERT
/// default: none); the final `classifier` Linear's bias is HF's `nn.Linear`
/// default (`bias = True`), independent of `classifier_bias`. Both biases load
/// iff the checkpoint has them. Shared by the sequence-classification and
/// token-classification heads — both are HF's `ModernBertPredictionHead` +
/// `classifier`; the sequence head pools first, the token head does not.
#[derive(Clone, Module)]
pub(crate) struct ClassifierHead {
    #[module(key = "head.dense")]
    dense: Linear,
    #[module(key = "head.norm")]
    norm: LayerNorm,
    classifier: Linear,
}

impl ClassifierHead {
    pub(crate) fn empty(config: &ModernBertConfig) -> Self {
        let (d, n, dt) = (config.hidden_size, config.num_labels, config.dtype.clone());
        let dense_bias = if config.classifier_bias { Bias::FanIn } else { Bias::None };
        Self {
            dense: linear(d, d, dense_bias, dt.clone()),
            norm: LayerNorm::with_dims(d, config.norm_bias, config.layer_norm_eps, dt.clone()),
            classifier: linear(d, n, Bias::FanIn, dt),
        }
    }

    /// Number of output labels (rows of the `classifier` weight).
    pub(crate) fn num_labels(&self) -> usize {
        self.classifier.weight.dim_const(0).expect("classifier weight is a concrete [num_labels, hidden]")
    }

    /// HF `ModernBertPredictionHead` + the `classifier` Linear: `dense → GELU →
    /// LayerNorm → classifier → f32`. Applied to the pooled `(B, D)` state by the
    /// sequence head and to the full `(B, L, D)` state by the token head (the
    /// `Linear`s broadcast over the leading axes). The projections, with their
    /// biases and the GELU in the GEMM's store, and the norm go through tk's op
    /// layer, as the backbone's do.
    pub(crate) fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let project = |x: &Tensor, layer: &Linear, act| {
            ops::linear(x, &layer.weight, ops::Linear { bias: layer.bias.as_ref(), act, ..ops::Linear::default() })
        };
        let h = project(hidden, &self.dense, Act::Gelu)?;
        let h = ops::layer_norm(&h, &self.norm.weight, self.norm.bias.as_ref(), self.norm.eps)?;
        Ok(project(&h, &self.classifier, Act::None)?.cast(DType::Float32))
    }
}

// ─── composite model (backbone + head) ─────────────────────────────────────

/// Backbone + classification head — the model type wrapped by the JIT.
/// [`Self::forward`] fuses backbone → pool → dense → GELU → norm → classifier
/// into a single graph. The backbone (`model.*`) and head (`head.*`,
/// `classifier.*`) keys are disjoint, so both load from the root of one
/// checkpoint.
#[derive(Clone, Module)]
pub(crate) struct ModernBertClassificationModel {
    #[module(key = "")]
    pub(crate) backbone: ModernBert,
    #[module(key = "")]
    head: ClassifierHead,
    #[module(skip)]
    pooling: ClassifierPooling,
}

impl ModernBertClassificationModel {
    pub(crate) fn empty(config: &ModernBertConfig) -> Self {
        Self {
            backbone: ModernBert::empty(config.clone()),
            head: ClassifierHead::empty(config),
            pooling: config.classifier_pooling,
        }
    }

    /// Build from a preloaded state dict. Each weight is cast to `config.dtype`.
    pub(crate) fn from_state_dict(sd: &StateDict, config: &ModernBertConfig) -> Result<Self> {
        let mut model = Self::empty(config);
        model.load_state_dict(&state::cast_all(sd, config.dtype.clone())?, "")?;
        Ok(model)
    }

    /// Fused forward: backbone → pool → head → logits `(B, num_labels)` f32.
    /// `mask`: `(B, L)`, non-zero = real token.
    pub(crate) fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let hidden = self.backbone.forward(input_ids, Some(mask))?;
        let pooled = match self.pooling {
            ClassifierPooling::Cls => hidden.take_index(1, 0)?,
            // The mean accumulates in f32; the head runs at the model's dtype.
            ClassifierPooling::Mean => hidden.masked_mean(mask, 1)?.cast(hidden.dtype()),
        };
        self.head.forward(&pooled)
    }
}

// ─── runtime (owns JIT, impl EncoderHead + Classify) ───────────────────────────

/// Finished-classifier model. Construction is cheap; each batch shape compiles
/// on first use. Implements [`EncoderHead`] (with [`Classify`] fixing the
/// output kinds) for drop-in use with
/// [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline).
pub struct ModernBertClassifier {
    plans: HeadPlans<ModernBertClassifierJit>,
    num_classes: usize,
}

impl ModernBertClassifier {
    /// Classify batches of up to `max_batch` chunks of up to `max_seq` tokens
    /// (see [`ModernBertEmbedder::new`](super::ModernBertEmbedder::new)).
    pub(crate) fn new(
        model: ModernBertClassificationModel,
        max_batch: usize,
        max_seq: usize,
    ) -> std::result::Result<Self, HeadError> {
        let num_classes = model.head.num_labels();
        let config = &model.backbone.config;
        let (max_positions, pad_id) = (config.max_position_embeddings, config.pad_token_id);
        Ok(Self { plans: HeadPlans::new(model, max_batch, max_seq, max_positions, pad_id, "classify")?, num_classes })
    }
}

impl EncoderHead for ModernBertClassifier {
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
            run.output.chunks_exact(self.num_classes).map(|row| Classification { logits: row.to_vec() }).collect();
        Ok((classifications, run.profile))
    }
}

impl Classify for ModernBertClassifier {
    fn num_labels(&self) -> usize {
        self.num_classes
    }
}
