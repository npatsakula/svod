//! Hub loaders for ModernBERT: fetch `config.json`, `model.safetensors`, and
//! `tokenizer.json` from a pinned revision, build the head from the two files
//! (each head's `from_files`), and hand it back beside its tokenizer, ready for
//! an [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline).

use std::path::{Path, PathBuf};

use snafu::ResultExt;
use svod_arch::pipelines::text::HfTokenizer;
use svod_dtype::DType;

use super::classifier::ModernBertClassifier;
use super::config::ModernBertConfig;
use super::embedder::ModernBertEmbedder;
use super::error::{Result, TokenizerSnafu};
use super::token_classifier::ModernBertTokenClassifier;
use crate::encoder_head::HubOptions;
use crate::state::StateDict;

/// The checkpoint's config computing in `dtype`, its weights, and the head's
/// `max_seq` (every position the checkpoint has unless the caller bounds it):
/// the shared prelude of the heads' `from_files`.
pub(super) fn load_checkpoint(
    config_json: &Path,
    weights: &Path,
    dtype: DType,
    max_seq: Option<usize>,
) -> Result<(ModernBertConfig, StateDict, usize)> {
    let config = ModernBertConfig { dtype, ..ModernBertConfig::from_json(config_json)? };
    let max_seq = max_seq.unwrap_or(config.max_position_embeddings);
    Ok((config, crate::state::load_safetensors(weights)?, max_seq))
}

/// `config.json` and `model.safetensors` from `opts.revision`, and the
/// tokenizer (see [`HubOptions::tokenizer_repo`]).
fn fetch_hub(model_id: &str, opts: &HubOptions) -> Result<(PathBuf, PathBuf, HfTokenizer)> {
    let repo = crate::hub::HubRepo::open(model_id, &opts.revision)?;
    let tokenizer = HfTokenizer::from_path(&opts.tokenizer_json(&repo)?).context(TokenizerSnafu)?;
    Ok((repo.get("config.json")?, repo.get("model.safetensors")?, tokenizer))
}

/// Download a ModernBERT checkpoint and build its embedder with
/// [`ModernBertEmbedder::from_files`] under `opts.head`, beside its tokenizer.
pub fn from_hub(model_id: &str, opts: &HubOptions) -> Result<(HfTokenizer, ModernBertEmbedder)> {
    let (config, weights, tokenizer) = fetch_hub(model_id, opts)?;
    Ok((tokenizer, ModernBertEmbedder::from_files(&config, &weights, &opts.head)?))
}

/// [`from_hub`] for a sequence-classification fine-tune
/// ([`ModernBertClassifier::from_files`]).
pub fn from_hub_classifier(model_id: &str, opts: &HubOptions) -> Result<(HfTokenizer, ModernBertClassifier)> {
    let (config, weights, tokenizer) = fetch_hub(model_id, opts)?;
    Ok((tokenizer, ModernBertClassifier::from_files(&config, &weights, &opts.head)?))
}

/// [`from_hub`] for a token-classification fine-tune
/// ([`ModernBertTokenClassifier::from_files`]).
pub fn from_hub_token_classification(
    model_id: &str,
    opts: &HubOptions,
) -> Result<(HfTokenizer, ModernBertTokenClassifier)> {
    let (config, weights, tokenizer) = fetch_hub(model_id, opts)?;
    Ok((tokenizer, ModernBertTokenClassifier::from_files(&config, &weights, &opts.head)?))
}
