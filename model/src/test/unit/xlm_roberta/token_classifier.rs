//! [`XlmRobertaTokenClassifier`]: the published key layout and the capacity its
//! position table allows (host-only), and parity on
//! `segment-any-text/sat-3l-sm` (heavy) against PyTorch and, for a long text's
//! stitched windows, against wtpsplit.
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

use svod_arch::pipelines::text::{
    ClassifyTokens, EncoderHead, EncoderPipeline, Encoding, HfTokenizer, SlidingWindowChunker, Tokenizer,
    WindowWeighting, sigmoid, token_scores_document,
};
use svod_dtype::default_device::set_default_device;
use svod_dtype::{DType, DeviceSpec};
use svod_tensor::nn::Module;

use crate::encoder_head::{HeadError, HeadOptions};
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

/// The tokenizer SaT uses (it ships none), at the revision the goldens were
/// generated with.
const TOKENIZER_REPO: (&str, &str) = ("FacebookAI/xlm-roberta-base", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089");

fn sat_file(name: &str) -> PathBuf {
    local_or_hub("SVOD_XLM_ROBERTA", "xlm_roberta", SAT_REPO, name)
}

/// The SaT classifier at `dtype` over the rows of `golden`, sized to them, and
/// the golden.
fn load_sat_fixture(golden: &str, dtype: DType) -> (XlmRobertaTokenClassifier, Vec<Encoding>, StateDict) {
    let golden = crate::state::load_safetensors(&sat_file(golden)).expect("golden");
    let encodings = golden_encodings(&golden);
    let opts = HeadOptions {
        max_batch: encodings.len(),
        max_seq: Some(encodings[0].len()),
        dtype: Some(dtype),
        ..HeadOptions::default()
    };
    let classifier =
        XlmRobertaTokenClassifier::from_files(&sat_file("config.json"), &sat_file("model.safetensors"), &opts)
            .expect("classifier");
    (classifier, encodings, golden)
}

/// Every chunk's logits, flat in the golden's `(B, L, labels)` layout, and
/// whether the plan launched tk's flash attention.
fn classify(classifier: &mut XlmRobertaTokenClassifier, encodings: &[Encoding]) -> (Vec<f32>, bool) {
    let refs: Vec<&Encoding> = encodings.iter().collect();
    let (out, profile) = classifier.run_batch(&refs, true).expect("classify tokens");
    let kernels = profile.expect("profiled run").stages.into_iter().flat_map(|stage| stage.kernels);
    let flash = kernels.into_iter().any(|k| k.kernel.entry_point.starts_with("flash_attention"));
    (out.into_iter().flat_map(|c| c.logits).collect(), flash)
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
    let got = real_rows(&classify(&mut classifier, &encodings).0, &mask, labels);
    let want = real_rows(&load_golden_vec(&golden, "expected_logits"), &mask, labels);
    let delta = max_abs_delta(&got, &want);
    eprintln!("real-token logits max |delta| = {delta:.3e}");
    assert!(delta < 1e-3, "token logits drifted from PyTorch golden: max |delta| = {delta}");
}

/// The classifier in bf16 — where its attention runs on tk's flash attention
/// wherever tk has it, which the f32 tests never reach — bounded by PyTorch's
/// own bf16 drift. It runs on a thread that defaults to the CPU: the plans
/// still compile for the device the classifier was built on.
#[test_case::test_case("golden_sat.safetensors"; "short")]
#[test_case::test_case("golden_sat_512.safetensors"; "full position table")]
#[ignore = "heavy: real sat-3l-sm weights (HF Hub) + PyTorch golden, on a bf16 device"]
fn sat_bf16_tracks_pytorch_bf16(golden: &str) {
    let dtype = crate::default_compute_dtype();
    if dtype != DType::BFloat16 {
        eprintln!("skip sat_bf16_tracks_pytorch_bf16: the default device computes in {dtype:?}");
        return;
    }
    let (mut classifier, encodings, golden) = load_sat_fixture(golden, dtype.clone());
    let mask = load_golden_vec::<i64>(&golden, "attention_mask");
    let run = std::thread::scope(|s| {
        s.spawn(|| {
            set_default_device(DeviceSpec::Cpu);
            classify(&mut classifier, &encodings)
        })
        .join()
    });
    let (got, flash) = run.expect("classify on a CPU-default thread");
    let device = svod_dtype::default_device::default_device();
    let tile = crate::attention::flash_attention_tile(&device, &dtype);
    eprintln!("sat on {device:?}: flash attention {flash}");
    assert_eq!(flash, tile.is_some(), "flash attention ran: {flash}, tk has it on {device:?}: {}", tile.is_some());
    assert_tracks_pytorch_bf16("sat", got, &golden, "expected_logits", Some((&mask, classifier.num_labels())));
}

/// One long text through the pipeline — sliding windows with the last pulled
/// back, the classifier, [`token_scores_document`] — against wtpsplit's
/// `extract`, which stitches SaT's windows (`golden_sat_windows`): the windows'
/// geometry, the logit averaging and its weighting. A token sits in up to eight
/// windows, and some tokens share a byte span. f32 on both sides.
#[test_case::test_case(WindowWeighting::Uniform, "expected_logits_uniform"; "uniform")]
#[test_case::test_case(WindowWeighting::Hat, "expected_logits_hat"; "hat")]
#[ignore = "heavy: real sat-3l-sm weights (HF Hub) + wtpsplit golden"]
fn sat_window_scores_match_wtpsplit(weighting: WindowWeighting, key: &str) {
    let golden = crate::state::load_safetensors(&sat_file("golden_sat_windows.safetensors")).expect("golden");
    let text = String::from_utf8(load_golden_vec::<u8>(&golden, "text")).expect("UTF-8 text");
    let stride = load_golden_vec::<i64>(&golden, "stride")[0] as usize;

    let tokenizer_json = local_or_hub("SVOD_XLM_ROBERTA", "xlm_roberta", TOKENIZER_REPO, "tokenizer.json");
    let mut tokenizer = HfTokenizer::from_path(tokenizer_json).expect("tokenizer");
    let enc = tokenizer.encode(&text).expect("encode");
    let content = enc.input_ids.iter().zip(&enc.special_tokens_mask).filter(|&(_, &special)| special == 0);
    let ids: Vec<i64> = content.map(|(&id, _)| id.into()).collect();
    assert_eq!(ids, load_golden_vec::<i64>(&golden, "input_ids"), "the text tokenizes as wtpsplit's does");

    let opts = HeadOptions { max_batch: 8, dtype: Some(DType::Float32), ..HeadOptions::default() };
    let classifier =
        XlmRobertaTokenClassifier::from_files(&sat_file("config.json"), &sat_file("model.safetensors"), &opts)
            .expect("classifier");
    let chunker = SlidingWindowChunker::new(classifier.capacity().1, stride).with_full_last_window();
    let mut pipeline = EncoderPipeline::new(tokenizer, chunker, classifier);
    let chunks = pipeline.classify_tokens(&text, ()).expect("classify tokens").chunks;
    let scores = token_scores_document(&chunks, weighting);

    let spans: Vec<(usize, usize)> = scores.iter().map(|t| (t.start, t.end)).collect();
    assert!(spans.windows(2).any(|pair| pair[0] == pair[1]), "no two tokens share a byte span");
    let got: Vec<f32> = scores.iter().map(|t| t.probs[0]).collect();
    let want: Vec<f32> = load_golden_vec::<f32>(&golden, key).into_iter().map(sigmoid).collect();
    assert_eq!(got.len(), want.len(), "scored tokens");
    let delta = max_abs_delta(&got, &want);
    eprintln!("{} windows of {} tokens: max |Δ| {delta:.3e}", chunks.len(), ids.len());
    assert!(delta < 1e-4, "stitched scores drifted from wtpsplit's: max |Δ| = {delta}");
}
