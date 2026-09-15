//! Criterion wall-time bench of `Qwen3Embedder` on full-length synthetic
//! batches, the grid the HF / vLLM comparison uses. Weights come from
//! `SVOD_QWEN3_REPO` (a local directory) or `submodules/Qwen3-Embedding-0.6B`;
//! the bench skips itself when neither exists.
//!
//! Run: `SVOD_DEVICE=CUDA:0 cargo bench -p svod-model --bench qwen3_embed`

use std::path::PathBuf;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use svod_model::qwen3::{Qwen3Config, Qwen3Embedder, Qwen3Embedding, qwen3_embedding_0_6b};

/// `(batch, tokens per row)`.
const SHAPES: &[(usize, usize)] = &[(1, 128), (1, 512), (1, 2048), (8, 128), (8, 512), (32, 128), (16, 512)];

fn weights_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("SVOD_QWEN3_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../submodules/Qwen3-Embedding-0.6B"));
    dir.join("model.safetensors").exists().then_some(dir)
}

/// `batch` rows of `len` deterministic ids over the text vocabulary, each ending
/// in the end-of-text token.
fn rows(batch: usize, len: usize, eos: u32) -> Vec<Vec<u32>> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..batch)
        .map(|_| {
            let mut row: Vec<u32> = (1..len)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((state >> 33) % 150_000) as u32
                })
                .collect();
            row.push(eos);
            row
        })
        .collect()
}

fn bench_qwen3_embed(c: &mut Criterion) {
    let Some(dir) = weights_dir() else {
        eprintln!("svod-model qwen3_embed bench: skipped (no Qwen3-Embedding-0.6B weights)");
        return;
    };
    let mut config = qwen3_embedding_0_6b();
    config.merge_structural_from(&Qwen3Config::from_json(&dir.join("config.json")).expect("config.json"));
    let eos = config.pad_token_id as u32;
    let model = Qwen3Embedding::from_safetensors_dir(&dir, config).expect("load weights");

    let mut group = c.benchmark_group("qwen3_embed");
    group.sample_size(10);
    for &(batch, len) in SHAPES {
        let mut embedder = Qwen3Embedder::new(model.clone(), batch, len);
        let rows = rows(batch, len, eos);
        embedder.embed(&rows).expect("compile the plan");
        group.throughput(Throughput::Elements((batch * len) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{batch}x{len}")), &rows, |b, rows| {
            b.iter(|| embedder.embed(rows).expect("embed"))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_qwen3_embed);
criterion_main!(benches);
