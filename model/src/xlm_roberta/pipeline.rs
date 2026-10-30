//! Hub loaders for the XLM-RoBERTa heads: fetch `config.json` and the weights
//! from a pinned revision and `tokenizer.json` (from the model's repository or
//! [`HubOptions::tokenizer_repo`] — SaT ships none), then assemble the
//! tokenizer + head ready for an
//! [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline).

use std::path::{Path, PathBuf};

use hf_hub::HFError;
use snafu::ResultExt;
use svod_arch::pipelines::text::HfTokenizer;

use super::classifier::XlmRobertaClassifier;
use super::embedder::XlmRobertaEmbedder;
use super::error::{PickleSnafu, Result, TokenizerSnafu};
use super::token_classifier::XlmRobertaTokenClassifier;
use crate::encoder_head::{HubOptions, pooling_from_hub};
use crate::hub::HubRepo;
use crate::state::StateDict;

/// A checkpoint's weights: a `.bin` is a PyTorch pickle (BGE-M3 publishes no
/// safetensors on `main`), anything else safetensors.
pub(super) fn load_weights(path: &Path) -> Result<StateDict> {
    if path.extension().is_some_and(|e| e == "bin") {
        crate::wespeaker::pickle::load_flat_pytorch_bin(path, "").context(PickleSnafu)
    } else {
        Ok(crate::state::load_safetensors(path)?)
    }
}

/// `config.json` and the weights (`model.safetensors`, else
/// `pytorch_model.bin`) from `opts.revision`, and the tokenizer.
fn fetch_hub(repo: &HubRepo, opts: &HubOptions) -> Result<(PathBuf, PathBuf, HfTokenizer)> {
    let weights = match repo.get("model.safetensors") {
        Err(HFError::EntryNotFound { .. }) => repo.get("pytorch_model.bin")?,
        weights => weights?,
    };
    let tokenizer = HfTokenizer::from_path(&opts.tokenizer_json(repo)?).context(TokenizerSnafu)?;
    Ok((repo.get("config.json")?, weights, tokenizer))
}

/// Download an XLM-RoBERTa embedding checkpoint (BGE-M3's dense vector, say)
/// and build it with [`XlmRobertaEmbedder::from_files`] under `opts.head`,
/// pooling as its sentence-transformers config says (mean, then L2, without
/// one), beside its tokenizer.
pub fn from_hub(model_id: &str, opts: &HubOptions) -> Result<(HfTokenizer, XlmRobertaEmbedder)> {
    let repo = HubRepo::open(model_id, &opts.revision)?;
    let pooling = pooling_from_hub(&repo)?;
    let (config, weights, tokenizer) = fetch_hub(&repo, opts)?;
    Ok((tokenizer, XlmRobertaEmbedder::from_files(&config, &weights, pooling, &opts.head)?))
}

/// [`from_hub`] for a sequence-classification fine-tune or a cross-encoder
/// reranker ([`XlmRobertaClassifier::from_files`]).
pub fn from_hub_classifier(model_id: &str, opts: &HubOptions) -> Result<(HfTokenizer, XlmRobertaClassifier)> {
    let (config, weights, tokenizer) = fetch_hub(&HubRepo::open(model_id, &opts.revision)?, opts)?;
    Ok((tokenizer, XlmRobertaClassifier::from_files(&config, &weights, &opts.head)?))
}

/// [`from_hub`] for a token-classification checkpoint
/// ([`XlmRobertaTokenClassifier::from_files`]).
pub fn from_hub_token_classification(
    model_id: &str,
    opts: &HubOptions,
) -> Result<(HfTokenizer, XlmRobertaTokenClassifier)> {
    let (config, weights, tokenizer) = fetch_hub(&HubRepo::open(model_id, &opts.revision)?, opts)?;
    Ok((tokenizer, XlmRobertaTokenClassifier::from_files(&config, &weights, &opts.head)?))
}
