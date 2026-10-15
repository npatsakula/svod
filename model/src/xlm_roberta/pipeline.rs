//! One-call hub loader for an XLM-RoBERTa token classifier: fetch
//! `config.json` and `model.safetensors` from a pinned revision and
//! `tokenizer.json` (from the model's repository or
//! [`HubOptions::tokenizer_repo`] — SaT ships none), then assemble the
//! tokenizer + head ready for an
//! [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline).

use snafu::ResultExt;
use svod_arch::pipelines::text::HfTokenizer;

use super::config::XlmRobertaConfig;
use super::error::{HeadSnafu, Result, TokenizerSnafu};
use super::token_classifier::{XlmRobertaTokenClassificationModel, XlmRobertaTokenClassifier};
use crate::encoder_head::HubOptions;
use crate::hub::HubRepo;

/// Loaded token-classifier pair: the tokenizer, the head, and the dense
/// label-name table from `config.json` (`id2label`).
pub struct XlmRobertaTokenClassifierLoad {
    pub tokenizer: HfTokenizer,
    pub classifier: XlmRobertaTokenClassifier,
    pub id2label: Vec<String>,
}

/// Download a token-classification checkpoint and assemble an
/// [`XlmRobertaTokenClassifierLoad`]. `opts.max_seq` defaults to
/// [`XlmRobertaConfig::max_sequence_length`] (512 for a 514-row position
/// table).
pub fn from_hub_token_classification(model_id: &str, opts: &HubOptions) -> Result<XlmRobertaTokenClassifierLoad> {
    let repo = HubRepo::open(model_id, &opts.revision)?;
    let mut config = XlmRobertaConfig { dtype: opts.dtype.clone(), ..super::xlm_roberta_large() };
    config.merge_structural_from(&XlmRobertaConfig::from_json(&repo.get("config.json")?)?);
    let sd = crate::state::load_safetensors(&repo.get("model.safetensors")?)?;
    let tokenizer = HfTokenizer::from_path(&opts.tokenizer_json(&repo)?).context(TokenizerSnafu)?;
    let max_seq = opts.max_seq.unwrap_or(config.max_sequence_length());
    let id2label = config.id2label.clone();
    let model = XlmRobertaTokenClassificationModel::from_state_dict(&sd, config)?;
    let classifier = XlmRobertaTokenClassifier::new(model, opts.max_batch, max_seq).context(HeadSnafu)?;
    Ok(XlmRobertaTokenClassifierLoad { tokenizer, classifier, id2label })
}
