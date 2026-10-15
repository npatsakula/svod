//! [`XlmRobertaTokenClassifier`]: the published key layout and the capacity its
//! position table allows (host-only), and parity against PyTorch on
//! `segment-any-text/sat-3l-sm` (heavy).
//!
//! # Goldens are not committed
//!
//! Generate them before running the `--ignored` tier (output lands in
//! `data/xlm_roberta/`, overridable via `SVOD_XLM_ROBERTA`):
//!
//! ```text
//! uv run model/scripts/generate_sat_token_golden.py
//! ```

use std::path::PathBuf;

use svod_arch::pipelines::text::{ClassifyTokens, EncoderHead, Encoding};
use svod_dtype::DType;
use svod_tensor::nn::Module;

use crate::encoder_head::HeadError;
use crate::state::StateDict;
use crate::test::unit::encoder_head::{assert_tracks_pytorch_bf16, golden_encodings, local_or_hub, real_rows};
use crate::test::unit::xlm_roberta::model::{expected_keys, sorted_keys, tiny_cfg};
use crate::test::unit::{load_golden_vec, max_abs_delta};
use crate::xlm_roberta::{XlmRobertaConfig, XlmRobertaTokenClassificationModel, XlmRobertaTokenClassifier};

/// The backbone's keys under `roberta.`, and the head's `classifier.`.
#[test]
fn state_dict_keys_match_published_layout() {
    let model = XlmRobertaTokenClassificationModel::empty(tiny_cfg());
    let mut want: Vec<String> = expected_keys().into_iter().map(|k| format!("roberta.{k}")).collect();
    want.extend(["classifier.bias".to_string(), "classifier.weight".to_string()]);
    want.sort();
    assert_eq!(sorted_keys(&model.state_dict("")), want);
}

/// A 514-row position table numbers 512 tokens (`pad_token_id = 1`): a longer
/// `max_seq` would let a chunk index past the table.
#[test_case::test_case(512 => matches Ok((1, 512)); "every numbered position")]
#[test_case::test_case(513 => matches Err(HeadError::BeyondPositions { max_seq: 513, max_positions: 512, .. }); "past the table")]
fn capacity_is_bounded_by_the_position_table(max_seq: usize) -> Result<(usize, usize), HeadError> {
    let cfg = XlmRobertaConfig { max_position_embeddings: 514, ..tiny_cfg() };
    XlmRobertaTokenClassifier::new(XlmRobertaTokenClassificationModel::empty(cfg), 1, max_seq).map(|c| c.capacity())
}

// ── parity (heavy) ──────────────────────────────────────────────────────────

/// `segment-any-text/sat-3l-sm`, at the revision the goldens were generated
/// from (the generate script's `--revision` default).
const SAT_REPO: (&str, &str) = ("segment-any-text/sat-3l-sm", "137da054051ad9f1eac42025f758db4ac9f22535");

fn sat_file(name: &str) -> PathBuf {
    local_or_hub("SVOD_XLM_ROBERTA", "xlm_roberta", SAT_REPO, name)
}

/// The SaT classifier at `dtype` over the rows of `golden`, sized to them, and
/// the golden.
fn load_sat_fixture(golden: &str, dtype: DType) -> (XlmRobertaTokenClassifier, Vec<Encoding>, StateDict) {
    let golden = crate::state::load_safetensors(&sat_file(golden)).expect("golden");
    let encodings = golden_encodings(&golden);
    let mut cfg = XlmRobertaConfig::from_json(&sat_file("config.json")).expect("parse config.json");
    cfg.dtype = dtype;
    let sd = crate::state::load_safetensors(&sat_file("model.safetensors")).expect("load weights");
    let model = XlmRobertaTokenClassificationModel::from_state_dict(&sd, cfg).expect("build model");
    let classifier = XlmRobertaTokenClassifier::new(model, encodings.len(), encodings[0].len()).expect("classifier");
    (classifier, encodings, golden)
}

/// Every chunk's logits, flat in the golden's `(B, L, labels)` layout.
fn classify(classifier: &mut XlmRobertaTokenClassifier, encodings: &[Encoding]) -> Vec<f32> {
    let refs: Vec<&Encoding> = encodings.iter().collect();
    let (out, _) = classifier.run_batch(&refs, false).expect("classify tokens");
    out.into_iter().flat_map(|c| c.logits).collect()
}

/// f32 logits at the real tokens against `XLMRobertaForTokenClassification`:
/// three short texts padded to 64 (a 128-token plan), and a full 512-token row
/// beside a padded one (the last position row, and the cap).
#[test_case::test_case("golden_sat.safetensors"; "short")]
#[test_case::test_case("golden_sat_512.safetensors"; "full position table")]
#[ignore = "heavy: real sat-3l-sm weights (HF Hub) + PyTorch golden"]
fn sat_logits_match_pytorch(golden: &str) {
    let (mut classifier, encodings, golden) = load_sat_fixture(golden, DType::Float32);
    let (mask, labels) = (load_golden_vec::<i64>(&golden, "attention_mask"), classifier.num_labels());
    let got = real_rows(&classify(&mut classifier, &encodings), &mask, labels);
    let want = real_rows(&load_golden_vec(&golden, "expected_logits"), &mask, labels);
    let delta = max_abs_delta(&got, &want);
    eprintln!("real-token logits max |delta| = {delta:.3e}");
    assert!(delta < 1e-3, "token logits drifted from PyTorch golden: max |delta| = {delta}");
}

/// The classifier in bf16 against the f32 golden, bounded by PyTorch's own
/// bf16 drift.
#[test_case::test_case("golden_sat.safetensors"; "short")]
#[test_case::test_case("golden_sat_512.safetensors"; "full position table")]
#[ignore = "heavy: real sat-3l-sm weights (HF Hub) + PyTorch golden, on a bf16 device"]
fn sat_bf16_tracks_pytorch_bf16(golden: &str) {
    let dtype = crate::default_compute_dtype();
    if dtype != DType::BFloat16 {
        eprintln!("skip sat_bf16_tracks_pytorch_bf16: the default device computes in {dtype:?}");
        return;
    }
    let (mut classifier, encodings, golden) = load_sat_fixture(golden, dtype);
    let mask = load_golden_vec::<i64>(&golden, "attention_mask");
    let got = classify(&mut classifier, &encodings);
    assert_tracks_pytorch_bf16("sat", got, &golden, "expected_logits", Some((&mask, classifier.num_labels())));
}
