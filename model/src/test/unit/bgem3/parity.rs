//! Parity tests comparing Rust model outputs against PyTorch/HuggingFace golden
//! reference outputs.
//!
//! Run with:
//!   SVOD_BGEM3=$PWD/data/bgem3 cargo test -p svod-model --lib bgem3::parity -- --ignored

use crate::bgem3::{BgeM3, BgeRerankerV2M3, EncodeOpts};
use crate::state::{self, StateDict};
use crate::test::unit::{load_golden_vec, max_abs_delta};
use crate::xlm_roberta::XlmRobertaConfig;
use std::path::PathBuf;
use svod_dtype::DType;
use svod_tensor::Tensor;

const HUB_REPO: &str = "BAAI/bge-m3";
const RERANKER_HUB: &str = "BAAI/bge-reranker-v2-m3";

fn real_file(name: &str) -> PathBuf {
    resolve_file(name, HUB_REPO)
}

fn real_file_reranker(name: &str) -> PathBuf {
    resolve_file(name, RERANKER_HUB)
}

fn resolve_file(name: &str, hub_repo: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("SVOD_BGEM3") {
        let p = PathBuf::from(dir).join(name);
        if p.exists() {
            return p;
        }
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/bgem3").join(name);
    if p.exists() {
        return p;
    }
    let repo = crate::hub::HubRepo::open(hub_repo, "main").expect("HF Hub API");
    repo.get(name).unwrap_or_else(|_| panic!("download {name} from {hub_repo}"))
}

fn load_fixture() -> (BgeM3, StateDict) {
    let golden = state::load_safetensors(&real_file("golden.safetensors")).expect("golden");
    let mut cfg = XlmRobertaConfig::from_json(&real_file("config.json")).expect("config");
    cfg.dtype = DType::Float32;
    let dtype = cfg.dtype.clone();
    let backbone = crate::xlm_roberta::XlmRobertaModel::from_pytorch_bin(&real_file("pytorch_model.bin"), cfg).unwrap();
    let sparse = real_file("sparse_linear.pt");
    let sparse = if sparse.exists() {
        Some(crate::bgem3::SparseHead::from_pytorch_bin(&sparse, backbone.config.vocab_size, dtype.clone()).unwrap())
    } else {
        None
    };
    let colbert = real_file("colbert_linear.pt");
    let colbert = if colbert.exists() {
        Some(crate::bgem3::ColbertHead::from_pytorch_bin(&colbert, backbone.config.hidden_size, dtype).unwrap())
    } else {
        None
    };
    (BgeM3 { model: backbone, sparse_head: sparse, colbert_head: colbert, normalize_dense: true }, golden)
}

fn prep_inputs(golden: &StateDict) -> (Tensor, Tensor) {
    let ids = load_golden_vec::<i64>(golden, "input_ids");
    let shape = load_golden_vec::<i64>(golden, "input_ids_shape");
    let (b, t) = (shape[0] as usize, shape[1] as usize);
    let mask = load_golden_vec::<i64>(golden, "attention_mask");
    let input_ids = Tensor::from_slice(&ids).try_reshape([b as isize, t as isize]).unwrap();
    let mask = Tensor::from_slice(&mask).try_reshape([b as isize, t as isize]).unwrap();
    (input_ids, mask)
}

#[test]
#[ignore = "heavy: real BGE-M3 weights + PyTorch golden"]
fn last_hidden_state_matches_pytorch() {
    let (model, golden) = load_fixture();
    let (ids, mask) = prep_inputs(&golden);
    let out = model.model.forward(&ids, Some(&mask)).unwrap();
    out.realize().unwrap();
    let delta = max_abs_delta(&out.as_vec::<f32>().unwrap(), &load_golden_vec(&golden, "last_hidden_state"));
    assert!(delta < 1e-3, "last_hidden_state max |delta| = {delta:.6}");
}

#[test]
#[ignore = "heavy: real BGE-M3 weights + PyTorch golden"]
fn dense_vecs_match_pytorch() {
    let (model, golden) = load_fixture();
    let (ids, mask) = prep_inputs(&golden);
    let dense = model.encode_dense(&ids, &mask).unwrap();
    dense.realize().unwrap();
    let delta = max_abs_delta(&dense.as_vec::<f32>().unwrap(), &load_golden_vec(&golden, "dense_vecs"));
    assert!(delta < 1e-3, "dense_vecs max |delta| = {delta:.6}");
}

#[test]
#[ignore = "heavy: real BGE-M3 weights + PyTorch golden"]
fn sparse_vecs_match_pytorch() {
    let (model, golden) = load_fixture();
    if model.sparse_head.is_none() {
        eprintln!("skipping: no sparse_linear.pt");
        return;
    }
    let (ids, mask) = prep_inputs(&golden);
    let out = model
        .encode(&ids, &mask, EncodeOpts { return_dense: false, return_sparse: true, return_colbert: false })
        .unwrap();
    let sparse = out.sparse_vecs.unwrap();
    sparse.realize().unwrap();
    let delta = max_abs_delta(&sparse.as_vec::<f32>().unwrap(), &load_golden_vec(&golden, "sparse_vecs"));
    assert!(delta < 1e-3, "sparse_vecs max |delta| = {delta:.6}");
}

#[test]
#[ignore = "heavy: real BGE-M3 weights + PyTorch golden"]
fn colbert_vecs_match_pytorch() {
    let (model, golden) = load_fixture();
    if model.colbert_head.is_none() {
        eprintln!("skipping: no colbert_linear.pt");
        return;
    }
    let (ids, mask) = prep_inputs(&golden);
    let out = model
        .encode(&ids, &mask, EncodeOpts { return_dense: false, return_sparse: false, return_colbert: true })
        .unwrap();
    let colbert = out.colbert_vecs.unwrap();
    colbert.realize().unwrap();
    let delta = max_abs_delta(&colbert.as_vec::<f32>().unwrap(), &load_golden_vec(&golden, "colbert_vecs"));
    assert!(delta < 1e-2, "colbert_vecs max |delta| = {delta:.6}");
}

#[test]
#[ignore = "heavy: real BGE-M3 weights + PyTorch golden"]
fn ignoring_padding_diverges_from_golden() {
    let (model, golden) = load_fixture();
    let (ids, _) = prep_inputs(&golden);
    let t = load_golden_vec::<i64>(&golden, "input_ids_shape")[1] as usize;
    let mask = Tensor::from_slice(vec![1i64; t]).try_reshape([1isize, t as isize]).unwrap();
    let out = model.model.forward(&ids, Some(&mask)).unwrap();
    out.realize().unwrap();
    let delta = max_abs_delta(&out.as_vec::<f32>().unwrap(), &load_golden_vec(&golden, "last_hidden_state"));
    assert!(delta > 1e-2, "all-ones mask did NOT diverge (delta={delta:.6})");
}

// --- Reranker ---

fn load_reranker_fixture() -> (BgeRerankerV2M3, StateDict) {
    let golden = state::load_safetensors(&real_file("golden_reranker.safetensors")).expect("golden");
    let mut cfg = XlmRobertaConfig::from_json(&real_file_reranker("config.json")).expect("config");
    cfg.dtype = DType::Float32;
    let model = BgeRerankerV2M3::from_safetensors(&real_file_reranker("model.safetensors"), cfg).unwrap();
    (model, golden)
}

#[test]
#[ignore = "heavy: real BGE-reranker-v2-m3 weights + PyTorch golden"]
fn reranker_logits_match_pytorch() {
    let (model, golden) = load_reranker_fixture();
    let (ids, mask) = prep_inputs(&golden);
    let out = model.forward(&ids, Some(&mask)).unwrap();
    out.realize().unwrap();
    let delta = max_abs_delta(&out.as_vec::<f32>().unwrap(), &load_golden_vec(&golden, "logits"));
    assert!(delta < 1e-3, "reranker logits max |delta| = {delta:.6}");
}
