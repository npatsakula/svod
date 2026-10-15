//! One-call hub loader for an XLM-RoBERTa token classifier: fetch
//! `config.json` and `model.safetensors` from a pinned revision and
//! `tokenizer.json` (from the model's repository or
//! [`HubOptions::tokenizer_repo`] — SaT ships none), then assemble the
//! tokenizer + head ready for an
//! [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline).

use snafu::ResultExt;
use svod_arch::pipelines::text::HfTokenizer;

use super::error::{Result, TokenizerSnafu};
use super::token_classifier::XlmRobertaTokenClassifier;
use crate::encoder_head::HubOptions;
use crate::hub::HubRepo;

/// Download a token-classification checkpoint and build it with
/// [`XlmRobertaTokenClassifier::from_files`] under `opts.head`, beside its
/// tokenizer.
pub fn from_hub_token_classification(
    model_id: &str,
    opts: &HubOptions,
) -> Result<(HfTokenizer, XlmRobertaTokenClassifier)> {
    let repo = HubRepo::open(model_id, &opts.revision)?;
    let classifier =
        XlmRobertaTokenClassifier::from_files(&repo.get("config.json")?, &repo.get("model.safetensors")?, &opts.head)?;
    let tokenizer = HfTokenizer::from_path(&opts.tokenizer_json(&repo)?).context(TokenizerSnafu)?;
    Ok((tokenizer, classifier))
}
