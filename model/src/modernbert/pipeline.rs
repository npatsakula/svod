//! One-call hub loaders for ModernBERT: fetch `config.json`,
//! `model.safetensors`, and `tokenizer.json` from a pinned revision, then
//! assemble the tokenizer + head ready for an
//! [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline). The three
//! loaders share one [`fetch_hub`] prelude; each supplies only its head-build
//! tail.

use snafu::ResultExt;
use svod_arch::pipelines::text::HfTokenizer;

use super::classifier::{ModernBertClassificationModel, ModernBertClassifier};
use super::config::ModernBertConfig;
use super::embedder::ModernBertEmbedder;
use super::error::{HeadSnafu, Result, TokenizerSnafu};
use super::model::ModernBert;
use super::token_classifier::{ModernBertTokenClassificationModel, ModernBertTokenClassifier};
use crate::encoder_head::HubOptions;
use crate::state::StateDict;

/// Fetch `config.json` + `model.safetensors` from `opts.revision`, and
/// `tokenizer.json` (see [`HubOptions::tokenizer_repo`]). The config takes the
/// caller's `dtype` and every structural field (including `id2label`) from the
/// downloaded `config.json` via [`ModernBertConfig::apply_checkpoint`]. Returns the resolved config, the
/// loaded state dict, the tokenizer, and the head's `max_seq`.
fn fetch_hub(model_id: &str, opts: &HubOptions) -> Result<(ModernBertConfig, StateDict, HfTokenizer, usize)> {
    let mut config = ModernBertConfig { dtype: opts.dtype.clone(), ..Default::default() };
    let repo = crate::hub::HubRepo::open(model_id, &opts.revision)?;
    config.apply_checkpoint(&ModernBertConfig::from_json(&repo.get("config.json")?)?);
    let sd = crate::state::load_safetensors(&repo.get("model.safetensors")?)?;
    let tokenizer = HfTokenizer::from_path(&opts.tokenizer_json(&repo)?).context(TokenizerSnafu)?;
    let max_seq = opts.max_seq.unwrap_or(config.max_position_embeddings);
    Ok((config, sd, tokenizer, max_seq))
}

// ── embedder ───────────────────────────────────────────────────────────────

/// Download `config.json` + `model.safetensors` + `tokenizer.json` from a
/// HuggingFace Hub repository and assemble the `(HfTokenizer,
/// ModernBertEmbedder)` pair ready for an
/// [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline).
pub fn from_hub(model_id: &str, opts: &HubOptions) -> Result<(HfTokenizer, ModernBertEmbedder)> {
    let (config, sd, tokenizer, max_seq) = fetch_hub(model_id, opts)?;
    let model = ModernBert::from_state_dict(&sd, config)?;
    let embedder = ModernBertEmbedder::new(model, opts.max_batch, max_seq).context(HeadSnafu)?;
    Ok((tokenizer, embedder))
}

// ── classifier ─────────────────────────────────────────────────────────────

/// Loaded sequence-classifier pair: the tokenizer, the head, and the dense
/// label-name table from `config.json` (`id2label`), so a caller can decode
/// predicted class ids to names without re-fetching the config.
pub struct ModernBertClassifierLoad {
    pub tokenizer: HfTokenizer,
    pub classifier: ModernBertClassifier,
    pub id2label: Vec<String>,
}

/// [`from_hub`] for a sequence-classification fine-tune: assembles a
/// [`ModernBertClassifierLoad`]. `id2label` is the dense label-name vec parsed
/// from `config.json`.
pub fn from_hub_classifier(model_id: &str, opts: &HubOptions) -> Result<ModernBertClassifierLoad> {
    let (config, sd, tokenizer, max_seq) = fetch_hub(model_id, opts)?;
    let model = ModernBertClassificationModel::from_state_dict(&sd, &config)?;
    let classifier = ModernBertClassifier::new(model, opts.max_batch, max_seq).context(HeadSnafu)?;
    Ok(ModernBertClassifierLoad { tokenizer, classifier, id2label: config.id2label })
}

// ── token classification ───────────────────────────────────────────────────

/// Loaded token-classifier pair: the tokenizer, the head, and the dense
/// label-name table from `config.json` (`id2label`), so a caller can decode
/// predicted token-label ids to names without re-fetching the config.
pub struct ModernBertTokenClassifierLoad {
    pub tokenizer: HfTokenizer,
    pub classifier: ModernBertTokenClassifier,
    pub id2label: Vec<String>,
}

/// [`from_hub`] for a token-classification fine-tune: assembles a
/// [`ModernBertTokenClassifierLoad`]. `id2label` is the dense label-name vec
/// parsed from `config.json`.
pub fn from_hub_token_classification(model_id: &str, opts: &HubOptions) -> Result<ModernBertTokenClassifierLoad> {
    let (config, sd, tokenizer, max_seq) = fetch_hub(model_id, opts)?;
    let model = ModernBertTokenClassificationModel::from_state_dict(&sd, &config)?;
    let classifier = ModernBertTokenClassifier::new(model, opts.max_batch, max_seq).context(HeadSnafu)?;
    Ok(ModernBertTokenClassifierLoad { tokenizer, classifier, id2label: config.id2label })
}
