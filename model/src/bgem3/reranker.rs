//! BGE-reranker-v2-m3: XLM-RoBERTa backbone + sequence classification head.
//!
//! Cross-encoder reranker fine-tuned from `BAAI/bge-m3`. The tokenizer
//! concatenates `(query, passage)` into a single input sequence with
//! `<s> query </s></s> passage </s>`; the model runs the backbone, takes the
//! CLS token, applies the two-layer classification head (`dense → tanh →
//! out_proj`), and optionally sigmoid-normalizes.
//!
//! Same backbone as [`crate::xlm_roberta::XlmRobertaModel`] — only the head
//! differs. Loads from `model.safetensors`, whose backbone keys carry the
//! `roberta.` prefix this module nests the backbone under.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::init::{fan_in_uniform, zeros};
use crate::state::{StateDict, cast_all};
use crate::xlm_roberta::config::XlmRobertaConfig;
use crate::xlm_roberta::error::Result;

use crate::xlm_roberta::model::XlmRobertaModel;

/// `RobertaClassificationHead`: `dense(D, D) → tanh → out_proj(D, num_labels)`.
#[derive(Clone, Module)]
pub struct ClassificationHead {
    #[module(key = "dense.weight")]
    pub dense_weight: Tensor,
    #[module(key = "dense.bias")]
    pub dense_bias: Tensor,
    #[module(key = "out_proj.weight")]
    pub out_proj_weight: Tensor,
    #[module(key = "out_proj.bias")]
    pub out_proj_bias: Tensor,
}

impl ClassificationHead {
    pub fn empty(hidden_size: usize, num_labels: usize, dtype: DType) -> Self {
        Self {
            dense_weight: fan_in_uniform(&[hidden_size, hidden_size], hidden_size, dtype.clone()),
            dense_bias: zeros(&[hidden_size], dtype.clone()),
            out_proj_weight: fan_in_uniform(&[num_labels, hidden_size], hidden_size, dtype.clone()),
            out_proj_bias: zeros(&[num_labels], dtype),
        }
    }

    /// Forward: `CLS → dense → tanh → out_proj → (B, num_labels)`.
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let cls = hidden.take_index(1, 0)?;
        let h = cls.linear().weight(&self.dense_weight).bias(&self.dense_bias).call()?;
        let h = h.tanh()?;
        Ok(h.linear().weight(&self.out_proj_weight).bias(&self.out_proj_bias).call()?)
    }
}

#[derive(Clone, Module)]
pub struct BgeRerankerV2M3 {
    #[module(key = "roberta")]
    pub model: XlmRobertaModel,
    pub classifier: ClassificationHead,
}

impl BgeRerankerV2M3 {
    pub fn empty(config: XlmRobertaConfig) -> Self {
        let dtype = config.dtype.clone();
        let hidden = config.hidden_size;
        Self { model: XlmRobertaModel::empty(config), classifier: ClassificationHead::empty(hidden, 1, dtype) }
    }

    /// Forward: backbone → classification head → logits `(B, 1)`.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let hidden = self.model.forward(input_ids, attention_mask)?;
        self.classifier.forward(&hidden)
    }

    /// Score with optional sigmoid normalization. Returns `(B, 1)`.
    pub fn compute_score(&self, input_ids: &Tensor, attention_mask: &Tensor, normalize: bool) -> Result<Tensor> {
        let logits = self.forward(input_ids, Some(attention_mask))?;
        if normalize { Ok(logits.sigmoid()?) } else { Ok(logits) }
    }

    /// Download from HuggingFace Hub and load.
    pub fn from_hub(model_id: &str, mut config: XlmRobertaConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", &mut config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: &mut XlmRobertaConfig) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision)?;

        let cfg_path = repo.get("config.json")?;
        let parsed = XlmRobertaConfig::from_json(&cfg_path)?;
        config.merge_structural_from(&parsed);

        let weights_path = repo.get("model.safetensors")?;
        Self::from_safetensors(&weights_path, config.clone())
    }

    /// Load from a `model.safetensors` checkpoint (backbone keys under
    /// `roberta.`, head keys under `classifier.`).
    pub fn from_safetensors(path: &std::path::Path, config: XlmRobertaConfig) -> Result<Self> {
        let sd = crate::state::load_safetensors(path)?;
        Self::from_state_dict(&sd, config)
    }

    /// Build from a preloaded state dict in the published key layout.
    pub fn from_state_dict(sd: &StateDict, config: XlmRobertaConfig) -> Result<Self> {
        let dtype = config.dtype.clone();
        let mut model = Self::empty(config);
        model.load_state_dict(&cast_all(sd, dtype)?, "")?;
        Ok(model)
    }
}
