use svod_dtype::DType;
use svod_tk::FLASH_ATTENTION_SEQUENCE_MULTIPLE as TILE;

use crate::qwen3::{Qwen3Config, Qwen3Embedder, Qwen3Embedding};

const EOS: u32 = 261;

fn tiny_cfg() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 300,
        hidden_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 16,
        intermediate_size: 64,
        max_position_embeddings: 2 * TILE,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_bias: false,
        tie_word_embeddings: true,
        pad_token_id: EOS as usize,
        dtype: DType::Float32,
    }
}

fn embedder_of(model: &Qwen3Embedding, max_batch: usize, max_len: usize) -> Qwen3Embedder {
    Qwen3Embedder::new(model.clone(), max_batch, max_len)
}

/// A row of `len` ids ending in the end-of-text token, distinct per `seed`.
fn row(seed: u32, len: usize) -> Vec<u32> {
    (0..len - 1).map(|i| (seed * 7 + i as u32 * 13) % 250).chain([EOS]).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn max_len_rounds_up_to_the_attention_tile() {
    let model = Qwen3Embedding::empty(tiny_cfg());
    assert_eq!(embedder_of(&model, 1, 1).max_len(), TILE);
    assert_eq!(embedder_of(&model, 1, TILE + 1).max_len(), 2 * TILE);
}

/// Rows come back in input order, unit-normalized, and each one is the same
/// vector whether it is embedded alone or packed next to longer rows.
#[test]
fn batching_preserves_order_and_rows() {
    let rows = [row(1, 7), row(2, 2), row(3, 12)];
    let model = Qwen3Embedding::empty(tiny_cfg());
    let batched = embedder_of(&model, 2, TILE).embed(&rows).unwrap();
    assert_eq!(batched.len(), 3);
    for (i, out) in batched.iter().enumerate() {
        assert_eq!(out.len(), 32);
        assert!((cosine(out, out) - 1.0).abs() < 1e-4, "unit norm for row {i}");
        let alone = &embedder_of(&model, 1, TILE).embed(&rows[i..=i]).unwrap()[0];
        let c = cosine(alone, out);
        assert!(c > 1.0 - 1e-4, "row {i} depends on its batch: cosine {c}");
    }
    assert!(cosine(&batched[0], &batched[1]) < 1.0 - 1e-3, "different rows, different vectors");
}

#[test]
fn rows_longer_than_max_len_are_truncated() {
    let model = Qwen3Embedding::empty(tiny_cfg());
    let mut e = embedder_of(&model, 1, TILE);
    let long = row(5, 2 * TILE);
    let truncated = &e.embed(&[&long]).unwrap()[0];
    let prefix = &e.embed(&[&long[..TILE]]).unwrap()[0];
    assert!(cosine(truncated, prefix) > 1.0 - 1e-4);
}

#[test]
fn profiled_run_records_one_stage_per_batch() {
    let model = Qwen3Embedding::empty(tiny_cfg());
    let (out, profile) = embedder_of(&model, 2, TILE).embed_profiled(&[row(1, 2), row(2, 3), row(3, 4)]).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(profile.stages.len(), 2);
    assert_eq!(profile.stages[0].name, format!("embed[2x{TILE}]"));
}
