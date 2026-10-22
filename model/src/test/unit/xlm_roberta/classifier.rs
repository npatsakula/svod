//! [`XlmRobertaClassifier`]: the published `bge-reranker-v2-m3` key layout, a
//! save/load round trip and a batch on the CPU (host-only), and parity on
//! `BAAI/bge-reranker-v2-m3` (heavy): pair encoding and truncation against
//! HF's tokenizer, logits through the pipeline against PyTorch's, in f32 and
//! bf16.
//!
//! # Goldens are not committed
//!
//! Generate them before running the `--ignored` tier (output lands in
//! `data/bgem3/`, overridable via `SVOD_BGEM3`):
//!
//! ```text
//! uv run --project model/scripts model/scripts/generate_bge_golden.py
//! ```

use std::path::PathBuf;

use svod_arch::pipelines::text::{
    Chunker, EncoderHead, EncoderPipeline, HfTokenizer, PairTokenizer, TruncatingChunker,
};
use svod_dtype::default_device::with_default_device;
use svod_dtype::{DType, DeviceSpec};
use svod_tensor::nn::Module;

use crate::encoder_head::HeadOptions;
use crate::state::StateDict;
use crate::test::unit::encoder_head::{assert_tracks_pytorch_bf16, encoding, local_or_hub};
use crate::test::unit::xlm_roberta::model::{expected_keys, sorted_keys, tiny_cfg};
use crate::test::unit::{load_golden_vec, max_abs_delta};
use crate::xlm_roberta::{XlmRobertaClassificationModel, XlmRobertaClassifier};

/// The backbone under `roberta.`, the head under `classifier.`.
#[test]
fn state_dict_keys_match_published_layout() {
    let model = XlmRobertaClassificationModel::empty(tiny_cfg());
    let mut want: Vec<String> = expected_keys().iter().map(|k| format!("roberta.{k}")).collect();
    want.extend(
        ["dense.weight", "dense.bias", "out_proj.weight", "out_proj.bias"].iter().map(|k| format!("classifier.{k}")),
    );
    want.sort();
    assert_eq!(sorted_keys(&model.state_dict("")), want);
}

#[test]
fn state_dict_round_trip() {
    let sd = XlmRobertaClassificationModel::empty(tiny_cfg()).state_dict("");
    let mut model = XlmRobertaClassificationModel::empty(tiny_cfg());
    model.load_state_dict(&sd, "").expect("load");
    assert_eq!(sorted_keys(&sd), sorted_keys(&model.state_dict("")));
}

/// One row of `num_labels` logits per chunk, each the row the chunk gets
/// alone: a shorter chunk's padding stays out of its `<s>` state.
#[test]
fn classifies_a_batch_on_the_cpu() {
    with_default_device(DeviceSpec::Cpu, || {
        let model = XlmRobertaClassificationModel::empty(tiny_cfg());
        let mut classifier = XlmRobertaClassifier::new(model, 2, 8).expect("classifier");
        let (long, short) = (encoding(&[0, 10, 20, 30, 2], 0), encoding(&[0, 40, 2], 0));
        let batch = classifier.run_batch(&[&long, &short], false).expect("batch").0;
        let alone = classifier.run_batch(&[&short], false).expect("alone").0;
        assert_eq!(batch.iter().map(|c| c.logits.len()).collect::<Vec<_>>(), [2, 2]);
        let delta = max_abs_delta(&batch[1].logits, &alone[0].logits);
        assert!(delta < 1e-5, "the batched row drifted from the lone one: {delta}");
    })
}

// ── parity (heavy) ──────────────────────────────────────────────────────────

/// `BAAI/bge-reranker-v2-m3`, at the revision the golden was generated from.
const RERANKER_REPO: (&str, &str) = ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e");

fn reranker_file(name: &str) -> PathBuf {
    local_or_hub("SVOD_BGEM3", "bgem3", RERANKER_REPO, name)
}

/// The golden, its (query, passage) pairs and the length they were padded to.
fn reranker_golden() -> (StateDict, Vec<(String, String)>, usize) {
    let golden = crate::state::load_safetensors(&reranker_file("golden_reranker.safetensors")).expect("golden");
    let pairs = serde_json::from_slice(&load_golden_vec::<u8>(&golden, "pairs_json")).expect("pairs");
    let max_seq = golden["input_ids"].dim_const(-1).expect("concrete seq_len");
    (golden, pairs, max_seq)
}

fn tokenizer() -> HfTokenizer {
    HfTokenizer::from_path(reranker_file("tokenizer.json")).expect("tokenizer")
}

/// Each pair encodes, and the longest truncates (LongestFirst), as HF's
/// tokenizer did for the golden.
#[test]
#[ignore = "heavy: bge-reranker-v2-m3 tokenizer (HF Hub) + golden"]
fn reranker_pairs_encode_as_hf() {
    let (golden, pairs, max_seq) = reranker_golden();
    let (ids, mask) = (load_golden_vec::<i64>(&golden, "input_ids"), load_golden_vec::<i64>(&golden, "attention_mask"));
    let mut tokenizer = tokenizer();
    let mut chunker = TruncatingChunker::new(max_seq);
    for (i, (query, passage)) in pairs.iter().enumerate() {
        let chunk = chunker.chunk(&tokenizer.encode_pair(query, passage).expect("encode")).expect("chunk").remove(0);
        let real = mask[i * max_seq..][..max_seq].iter().filter(|&&m| m == 1).count();
        let got: Vec<i64> = chunk.encoding.input_ids.iter().map(|&id| id.into()).collect();
        assert_eq!(got, ids[i * max_seq..][..real], "pair {i}");
    }
}

/// Every pair's logits from the pipeline at `dtype`, in the golden's order.
fn rerank(pairs: &[(String, String)], max_seq: usize, dtype: DType) -> Vec<f32> {
    let opts =
        HeadOptions { max_batch: pairs.len(), max_seq: Some(max_seq), dtype: Some(dtype), ..HeadOptions::default() };
    let classifier =
        XlmRobertaClassifier::from_files(&reranker_file("config.json"), &reranker_file("model.safetensors"), &opts)
            .expect("classifier");
    let mut pipeline = EncoderPipeline::new(tokenizer(), TruncatingChunker::new(max_seq), classifier);
    let pairs: Vec<(&str, &str)> = pairs.iter().map(|(q, p)| (q.as_str(), p.as_str())).collect();
    let out = pipeline.classify_pair_batch(&pairs, ()).expect("rerank");
    out.results.into_iter().flat_map(|r| r.chunks.into_iter().flat_map(|c| c.logits)).collect()
}

/// f32 scores through `classify_pair_batch` against
/// `XLMRobertaForSequenceClassification` on HF's encoding.
#[test]
#[ignore = "heavy: real bge-reranker-v2-m3 weights (HF Hub) + PyTorch golden"]
fn reranker_logits_match_pytorch() {
    let (golden, pairs, max_seq) = reranker_golden();
    let delta = max_abs_delta(&rerank(&pairs, max_seq, DType::Float32), &load_golden_vec(&golden, "expected_logits"));
    eprintln!("reranker logits max |delta| = {delta:.3e}");
    assert!(delta < 1e-3, "reranker logits drifted from PyTorch golden: max |delta| = {delta}");
}

/// The reranker in bf16, bounded by PyTorch's own bf16 drift.
#[test]
#[ignore = "heavy: real bge-reranker-v2-m3 weights (HF Hub) + PyTorch golden, on a bf16 device"]
fn reranker_bf16_tracks_pytorch_bf16() {
    let dtype = crate::default_compute_dtype();
    if dtype != DType::BFloat16 {
        eprintln!("skip reranker_bf16_tracks_pytorch_bf16: the default device computes in {dtype:?}");
        return;
    }
    let (golden, pairs, max_seq) = reranker_golden();
    assert_tracks_pytorch_bf16("reranker", rerank(&pairs, max_seq, dtype), &golden, "expected_logits", None);
}
