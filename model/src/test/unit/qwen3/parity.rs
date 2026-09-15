//! Golden parity tests against real `Qwen/Qwen3-Embedding-0.6B` weights.
//!
//! The goldens were produced by the HF reference with left padding; rows are
//! re-packed to this port's right-padding convention before comparison.
//!
//! Run with:
//! ```text
//! SVOD_QWEN3=$PWD/data/qwen3 cargo test -p svod-model --lib qwen3::parity -- --ignored
//! ```

use std::path::PathBuf;

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::qwen3::{Qwen3Config, Qwen3Embedding, Qwen3Model, Qwen3Reranker};
use crate::state::{self, StateDict};

const HUB_REPO: &str = "Qwen/Qwen3-Embedding-0.6B";
const RERANKER_HUB_REPO: &str = "Qwen/Qwen3-Reranker-0.6B";

fn resolve_in(repo: &str, name: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("SVOD_QWEN3") {
        let p = PathBuf::from(dir).join(name);
        if p.exists() {
            return p;
        }
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/qwen3").join(name);
    if p.exists() {
        return p;
    }
    let hub = crate::hub::HubRepo::open(repo, "main").expect("HF Hub API");
    hub.get(name).unwrap_or_else(|_| panic!("download {name} from {repo}"))
}

fn resolve_file(name: &str) -> PathBuf {
    resolve_in(HUB_REPO, name)
}

fn load_cfg(config: &str) -> Qwen3Config {
    let mut cfg = Qwen3Config::from_json(&resolve_file(config)).expect("config");
    cfg.dtype = DType::Float32;
    cfg
}

fn load_model() -> Qwen3Model {
    Qwen3Model::from_safetensors(&resolve_file("model.safetensors"), load_cfg("config.json")).expect("load model")
}

fn load_reranker() -> Qwen3Reranker {
    let cfg = load_cfg("reranker_config.json");
    Qwen3Reranker::from_safetensors(&resolve_in(RERANKER_HUB_REPO, "reranker_model.safetensors"), cfg)
        .expect("load reranker")
}

fn golden(name: &str) -> StateDict {
    state::load_safetensors(&resolve_file(name)).expect("load golden")
}

fn realized<T: svod_dtype::ext::HasDType + Clone + Default>(sd: &StateDict, key: &str) -> Vec<T> {
    let t = sd.get(key).unwrap_or_else(|| panic!("missing golden key: {key}")).clone();
    t.realize().unwrap();
    t.as_vec::<T>().unwrap()
}

/// The golden's left-padded `input_ids` + `attention_mask`, re-packed to
/// right padding: `(ids [B, L] i32, lengths [B] i32, per-row pad counts)`.
struct Rows {
    batch: usize,
    seq_len: usize,
    ids: Tensor,
    lengths: Tensor,
    left_pads: Vec<usize>,
}

fn rows(sd: &StateDict) -> Rows {
    let shape: Vec<i64> = realized(sd, "input_ids_shape");
    let (batch, seq_len) = (shape[0] as usize, shape[1] as usize);
    let ids: Vec<i64> = realized(sd, "input_ids");
    let mask: Vec<i64> = realized(sd, "attention_mask");
    let mut packed = vec![ids[0] as i32; batch * seq_len];
    let mut lengths = Vec::with_capacity(batch);
    let mut left_pads = Vec::with_capacity(batch);
    for b in 0..batch {
        let row = &ids[b * seq_len..(b + 1) * seq_len];
        let real: Vec<i32> = (0..seq_len).filter(|&s| mask[b * seq_len + s] != 0).map(|s| row[s] as i32).collect();
        assert!(mask[(b + 1) * seq_len - 1] != 0, "golden rows are left-padded");
        packed[b * seq_len..b * seq_len + real.len()].copy_from_slice(&real);
        lengths.push(real.len() as i32);
        left_pads.push(seq_len - real.len());
    }
    Rows {
        batch,
        seq_len,
        ids: Tensor::from_slice(packed).try_reshape([batch as isize, seq_len as isize]).unwrap(),
        lengths: Tensor::from_slice(lengths),
        left_pads,
    }
}

fn max_abs_delta(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
}

#[test]
#[ignore = "heavy: real Qwen3-Embedding-0.6B weights + PyTorch golden"]
fn last_hidden_state_matches_pytorch() {
    let model = load_model();
    let sd = golden("golden.safetensors");
    let rows = rows(&sd);
    let hidden = model.config.hidden_size;

    let out = model.forward(&rows.ids).unwrap();
    out.realize().unwrap();
    let got = out.as_vec::<f32>().unwrap();
    let want: Vec<f32> = realized(&sd, "last_hidden_state");
    assert_eq!(got.len(), want.len());

    // Real token `s` of row `b` sits at `s` here and at `left_pads[b] + s` in the golden.
    let mut worst = 0.0f32;
    for (b, &pad) in rows.left_pads.iter().enumerate() {
        for s in 0..rows.seq_len - pad {
            let ours = (b * rows.seq_len + s) * hidden;
            let theirs = (b * rows.seq_len + pad + s) * hidden;
            worst = worst.max(max_abs_delta(&got[ours..ours + hidden], &want[theirs..theirs + hidden]));
        }
    }
    assert!(worst < 1e-3, "real-token max_abs_delta = {worst:.6} (threshold 1e-3)");
}

#[test]
#[ignore = "heavy: real Qwen3-Embedding-0.6B weights + PyTorch golden"]
fn embeddings_match_pytorch() {
    let emb = Qwen3Embedding { model: load_model(), normalize: true };
    let sd = golden("golden.safetensors");
    let rows = rows(&sd);

    let out = emb.encode(&rows.ids, &rows.lengths).unwrap();
    out.realize().unwrap();
    let got = out.as_vec::<f32>().unwrap();
    let want: Vec<f32> = realized(&sd, "embeddings");
    assert_eq!(got.len(), want.len());
    let delta = max_abs_delta(&got, &want);
    assert!(delta < 1e-3, "max_abs_delta = {delta:.6} (threshold 1e-3)");
}

#[test]
#[ignore = "heavy: negative control — pooling the padded end must diverge"]
fn pooling_past_the_real_length_diverges_from_golden() {
    let emb = Qwen3Embedding { model: load_model(), normalize: true };
    let sd = golden("golden.safetensors");
    let rows = rows(&sd);
    assert!(rows.left_pads.iter().any(|&p| p > 0), "the golden batch has no padded row");

    let full = Tensor::from_slice(vec![rows.seq_len as i32; rows.batch]);
    let out = emb.encode(&rows.ids, &full).unwrap();
    out.realize().unwrap();
    let got = out.as_vec::<f32>().unwrap();
    let want: Vec<f32> = realized(&sd, "embeddings");
    let delta = max_abs_delta(&got, &want);
    assert!(delta > 1e-2, "pooling pad tokens did NOT diverge (delta={delta:.6})");
}

#[test]
#[ignore = "heavy: real Qwen3-Reranker-0.6B weights + PyTorch golden"]
fn reranker_scores_match_pytorch() {
    let reranker = load_reranker();
    let sd = golden("golden_reranker.safetensors");
    let rows = rows(&sd);

    let out = reranker.forward(&rows.ids, &rows.lengths).unwrap();
    out.realize().unwrap();
    let got = out.as_vec::<f32>().unwrap();
    let want: Vec<f32> = realized(&sd, "scores");
    assert_eq!(got.len(), want.len());
    let delta = max_abs_delta(&got, &want);
    assert!(delta < 1e-3, "max_abs_delta = {delta:.6} (threshold 1e-3)");
}
