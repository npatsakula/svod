use crate::qwen3::SEQUENCE_TILE as TILE;
use svod_dtype::DType;

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

/// Rows come back in input order, unit-normalized, and each one is the same
/// vector whether it is embedded alone or packed into a plan row next to
/// other rows — which every row of this set is: three rows in two plan rows.
#[test]
fn packing_preserves_order_and_rows() {
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

/// A row longer than the space left in every plan row waits for the next
/// batch; the batch's bucket is its longest row's, and a row's vector does
/// not depend on which batch it landed in.
#[test]
fn rows_that_do_not_fit_spill_to_the_next_batch() {
    let long = TILE - 10;
    let rows = [row(1, 20), row(2, long), row(3, long), row(4, long), row(5, 3)];
    let model = Qwen3Embedding::empty(tiny_cfg());
    let (out, profile) = embedder_of(&model, 2, TILE).embed_profiled(&rows).unwrap();
    // Three long rows and a 20-token one: two plan rows take two long rows and
    // the 3-token row, the third long row and the 20-token one wait.
    assert_eq!(profile.stages.len(), 2);
    assert!(profile.stages.iter().all(|s| s.name == format!("embed[2x{TILE}]")));
    for (i, row) in rows.iter().enumerate() {
        let alone = &embedder_of(&model, 1, TILE).embed(&[row]).unwrap()[0];
        let c = cosine(alone, &out[i]);
        assert!(c > 1.0 - 1e-4, "row {i} depends on its batch: cosine {c}");
    }
}

/// An empty row embeds as a single pad token rather than failing or reading
/// another row's slot.
#[test]
fn empty_rows_embed_as_one_pad_token() {
    let model = Qwen3Embedding::empty(tiny_cfg());
    let mut e = embedder_of(&model, 2, TILE);
    let out = e.embed(&[Vec::new(), row(1, 5)]).unwrap();
    let pad = &e.embed(&[vec![EOS]]).unwrap()[0];
    assert!(cosine(&out[0], pad) > 1.0 - 1e-4);
    assert!(out[0].iter().all(|x| x.is_finite()));
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

/// Short rows pack into one plan row up to the pooled-slot limit
/// (`TILE / 16` per row), so `2 · TILE / 16 + 1` rows take two batches.
#[test]
fn profiled_run_records_one_stage_per_batch() {
    let model = Qwen3Embedding::empty(tiny_cfg());
    let rows: Vec<Vec<u32>> = (0..2 * TILE / 16 + 1).map(|i| row(i as u32, 2)).collect();
    let (out, profile) = embedder_of(&model, 2, TILE).embed_profiled(&rows).unwrap();
    assert_eq!(out.len(), rows.len());
    assert_eq!(profile.stages.len(), 2);
    assert_eq!(profile.stages[0].name, format!("embed[2x{TILE}]"));
}
