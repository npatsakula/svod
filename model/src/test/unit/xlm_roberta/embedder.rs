//! [`XlmRobertaEmbedder`]: the backbone from either checkpoint layout and a
//! pooled batch on the CPU (host-only), and parity on `BAAI/bge-m3` (heavy):
//! its sentence-transformers pooling read from the hub, the texts' encoding
//! and the dense vectors through the pipeline against PyTorch's, in f32 and
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

use svod_arch::pipelines::text::{Embed, EncoderHead, EncoderPipeline, Tokenizer, TruncatingChunker};
use svod_dtype::default_device::with_default_device;
use svod_dtype::{DType, DeviceSpec};
use svod_tensor::nn::Module;

use crate::encoder_head::{HeadOptions, HubOptions, Pooling, PoolingMode};
use crate::state::StateDict;
use crate::test::unit::encoder_head::{assert_tracks_pytorch_bf16, encoding, local_or_hub};
use crate::test::unit::xlm_roberta::model::tiny_cfg;
use crate::test::unit::{load_golden_vec, max_abs_delta};
use crate::xlm_roberta::{XlmRobertaEmbedder, XlmRobertaModel, load_backbone};

/// A backbone saved at the root (BGE-M3) or under `roberta.` (a fine-tune)
/// loads the same weights.
#[test_case::test_case(""; "at the root")]
#[test_case::test_case("roberta"; "under roberta")]
fn the_backbone_loads_from_either_layout(prefix: &str) {
    let source = XlmRobertaModel::empty(tiny_cfg());
    let loaded = load_backbone(&source.state_dict(prefix), tiny_cfg()).expect("load");
    let word_embeddings = |model: &XlmRobertaModel| {
        let t = model.state_dict("")["embeddings.word_embeddings.weight"].clone();
        t.realize().expect("realize");
        t.as_vec::<f32>().expect("readout")
    };
    assert_eq!(word_embeddings(&loaded), word_embeddings(&source));
}

/// A pooled batch: one `hidden_size` vector per chunk, unit length when the
/// pooling normalizes, and each the vector the chunk gets alone.
#[test_case::test_case(PoolingMode::Cls; "cls")]
#[test_case::test_case(PoolingMode::Mean; "mean")]
fn embeds_a_batch_on_the_cpu(mode: PoolingMode) {
    with_default_device(DeviceSpec::Cpu, || {
        let pooling = Pooling { mode, normalize: true };
        let mut embedder =
            XlmRobertaEmbedder::new(XlmRobertaModel::empty(tiny_cfg()), pooling, 2, 8).expect("embedder");
        assert!(embedder.normalized());
        let (long, short) = (encoding(&[0, 10, 20, 30, 2], 0), encoding(&[0, 40, 2], 0));
        let batch = embedder.run_batch(&[&long, &short], false).expect("batch").0;
        let alone = embedder.run_batch(&[&short], false).expect("alone").0;
        for e in &batch {
            assert_eq!(e.values.len(), tiny_cfg().hidden_size);
            let norm = e.values.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
        }
        let delta = max_abs_delta(&batch[1].values, &alone[0].values);
        assert!(delta < 1e-5, "the batched row drifted from the lone one: {delta}");
    })
}

// ── parity (heavy) ──────────────────────────────────────────────────────────

/// `BAAI/bge-m3`, at the revision the golden was generated from.
const BGE_M3_REPO: (&str, &str) = ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181");

/// The golden, its texts and the length they were padded to.
fn dense_golden() -> (StateDict, Vec<String>, usize) {
    let path = local_or_hub("SVOD_BGEM3", "bgem3", BGE_M3_REPO, "golden_dense.safetensors");
    let golden = crate::state::load_safetensors(&path).expect("golden");
    let texts = serde_json::from_slice(&load_golden_vec::<u8>(&golden, "texts_json")).expect("texts");
    let max_seq = golden["input_ids"].dim_const(-1).expect("concrete seq_len");
    (golden, texts, max_seq)
}

/// The texts' dense vectors from `xlm_roberta::from_hub`'s pipeline at
/// `dtype`, after checking it pools as BGE-M3's config says and encodes the
/// texts as the golden's tokenizer did.
fn embed(golden: &StateDict, texts: &[String], max_seq: usize, dtype: DType) -> Vec<f32> {
    let head =
        HeadOptions { max_batch: texts.len(), max_seq: Some(max_seq), dtype: Some(dtype), ..HeadOptions::default() };
    let opts = HubOptions { revision: BGE_M3_REPO.1.into(), head, ..HubOptions::default() };
    let (tokenizer, embedder) = crate::xlm_roberta::from_hub(BGE_M3_REPO.0, &opts).expect("bge-m3");
    assert_eq!(embedder.pooling(), Pooling { mode: PoolingMode::Cls, normalize: true });
    let mut pipeline = EncoderPipeline::new(tokenizer, TruncatingChunker::new(max_seq), embedder).expect("pipeline");

    let (ids, mask) = (load_golden_vec::<i64>(golden, "input_ids"), load_golden_vec::<i64>(golden, "attention_mask"));
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    for (i, enc) in pipeline.tokenizer_mut().encode_batch(&refs).expect("encode").iter().enumerate() {
        let real = mask[i * max_seq..][..max_seq].iter().filter(|&&m| m == 1).count();
        let got: Vec<i64> = enc.input_ids.iter().map(|&id| id.into()).collect();
        assert_eq!(got, ids[i * max_seq..][..real], "text {i} encodes as the golden's");
    }
    let out = pipeline.embed_batch(&refs, ()).expect("embed");
    out.results.into_iter().flat_map(|r| r.chunks.into_iter().flat_map(|c| c.values)).collect()
}

/// f32 dense vectors against PyTorch's `<s>` state, L2-normalized.
#[test]
#[ignore = "heavy: real BGE-M3 weights (HF Hub) + PyTorch golden"]
fn dense_vectors_match_pytorch() {
    let (golden, texts, max_seq) = dense_golden();
    let got = embed(&golden, &texts, max_seq, DType::Float32);
    let delta = max_abs_delta(&got, &load_golden_vec(&golden, "expected_embedding"));
    eprintln!("dense max |delta| = {delta:.3e}");
    assert!(delta < 1e-3, "dense vectors drifted from PyTorch golden: max |delta| = {delta}");
}

/// The embedder in bf16, bounded by PyTorch's own bf16 drift.
#[test]
#[ignore = "heavy: real BGE-M3 weights (HF Hub) + PyTorch golden, on a bf16 device"]
fn dense_bf16_tracks_pytorch_bf16() {
    let dtype = crate::default_compute_dtype();
    if dtype != DType::BFloat16 {
        eprintln!("skip dense_bf16_tracks_pytorch_bf16: the default device computes in {dtype:?}");
        return;
    }
    let (golden, texts, max_seq) = dense_golden();
    assert_tracks_pytorch_bf16("dense", embed(&golden, &texts, max_seq, dtype), &golden, "expected_embedding", None);
}
