//! ModernBERT batched embeddings demo.
//!
//! Demonstrates [`EncoderPipeline::embed_batch`] — the throughput path: several
//! texts tokenized, chunked, and encoded in one call, with all chunks across
//! all texts flattened into a single sub-batched stream. This is the analog of
//! [`modernbert_embed_infer`](super::modernbert_embed_infer) for the multi-text
//! case, and the only example exercising a `_batch` verb.
//!
//! Usage:
//!   cargo run -p svod-model --release --example modernbert_embed_batch_infer -- "first text" "second text"
//!   cargo run -p svod-model --release --example modernbert_embed_batch_infer -- --repo answerdotai/ModernBERT-base < texts.txt
//!   printf 'one\nanother\n' | cargo run -p svod-model --release --example modernbert_embed_batch_infer
//!   cargo run -p svod-model --release --example modernbert_embed_batch_infer -- --max-batch 8 --runs 5 --profile < texts.txt
//!
//! Reads newline-delimited texts from stdin when no positional args are given.

use std::io::{self, BufRead};
use std::time::Instant;

use clap::{Parser, ValueEnum};

use svod_arch::pipelines::text::{Embed, EncoderHead, EncoderPipeline, RunOptions, TruncatingChunker};
use svod_dtype::DType;
use svod_model::modernbert;

#[derive(Parser, Debug)]
#[command(about = "ModernBERT batched embeddings demo", long_about = None)]
struct Args {
    /// Texts to embed (reads newline-delimited stdin if none are given).
    text: Vec<String>,

    /// HF Hub repo with the ModernBERT weights + tokenizer.
    #[arg(long, default_value = "answerdotai/ModernBERT-base")]
    repo: String,

    /// HF Hub revision.
    #[arg(long, default_value = "main")]
    revision: String,

    /// Compute dtype (default: the device's, bf16 on a tensor-core GPU).
    #[arg(long, value_enum)]
    dtype: Option<Dtype>,

    /// Most chunks one execute runs: the pipeline sub-batches the flattened
    /// chunks to this many at a time, longest first.
    #[arg(long, default_value_t = 4)]
    max_batch: usize,

    /// Longest chunk in tokens, specials included (default: the checkpoint's
    /// `max_position_embeddings`). A batch runs at its own longest chunk.
    #[arg(long)]
    max_seq: Option<usize>,

    /// Collect and print the batch-level profile (of the last pass).
    #[arg(long)]
    profile: bool,

    /// Timed passes over the texts after the first, which compiles each batch
    /// shape it meets.
    #[arg(long, default_value_t = 0)]
    runs: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Dtype {
    F32,
    F16,
    Bf16,
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let texts: Vec<String> = if args.text.is_empty() {
        io::stdin().lock().lines().map_while(Result::ok).filter(|l| !l.is_empty()).collect()
    } else {
        args.text
    };
    if texts.is_empty() {
        return Err("no input texts (pass positional args or pipe newline-delimited stdin)".into());
    }
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    let dtype = match args.dtype {
        None => svod_model::default_compute_dtype(),
        Some(Dtype::F32) => DType::Float32,
        Some(Dtype::F16) => DType::Float16,
        Some(Dtype::Bf16) => DType::BFloat16,
    };
    let opts = modernbert::HubOptions {
        revision: args.revision.clone(),
        max_batch: args.max_batch,
        max_seq: args.max_seq,
        dtype,
    };

    let t_total = Instant::now();
    println!("Loading ModernBERT from {} ({})...", args.repo, args.revision);
    let (tokenizer, embedder) = modernbert::from_hub(&args.repo, &opts)?;
    let (_, max_seq) = embedder.capacity();
    println!(
        "Loaded: hidden_size={}, max_seq={}, max_batch={} — embedding {} text(s)",
        embedder.hidden_size(),
        max_seq,
        args.max_batch,
        texts.len()
    );

    let mut pipeline = EncoderPipeline::new(tokenizer, TruncatingChunker::new(max_seq), embedder);

    println!("\nBatched embed...");
    let t = Instant::now();
    let mut result = pipeline.embed_batch(&refs, RunOptions { profile: args.profile && args.runs == 0 })?;
    let dt = t.elapsed();
    for run in 1..=args.runs {
        let t = Instant::now();
        result = pipeline.embed_batch(&refs, RunOptions { profile: args.profile && run == args.runs })?;
        println!("  pass {run}: {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
    }

    assert_eq!(result.results.len(), texts.len());
    let total_chunks: usize = result.results.iter().map(|r| r.chunks.len()).sum();
    println!("  {total_chunks} chunk(s) across {} text(s), first pass {:.3}s", texts.len(), dt.as_secs_f32());
    for (i, text) in texts.iter().enumerate() {
        let emb = &result.results[i];
        let preview: String = text.chars().take(40).collect();
        for (c, chunk) in emb.chunks.iter().enumerate() {
            println!(
                "  [{i}] chunk {c} @ byte {}: dim={} L2={:.4}  {:?}{}",
                chunk.byte_offset,
                chunk.values.len(),
                l2(&chunk.values),
                preview,
                if text.len() > 40 { "…" } else { "" }
            );
        }
    }

    if let Some(prof) = &result.profile {
        println!("\n--- Batch profile ---\n{prof}");
    }
    println!("\nTotal: {:.2}s", t_total.elapsed().as_secs_f32());
    Ok(())
}
