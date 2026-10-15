//! ModernBERT embeddings inference demo.
//!
//! Loads a HuggingFace ModernBERT checkpoint (weights + tokenizer) in one call
//! and runs a text input through an [`EncoderPipeline`]: `HfTokenizer` →
//! chunker → `ModernBertEmbedder`. By default a `TruncatingChunker` is used;
//! pass `--window` + `--stride` to switch to `SlidingWindowChunker` for
//! long-document windowed embedding. This doubles as the runnable end-to-end
//! smoke test for the text pipeline — the analog of `gigaam_infer.rs` for audio.
//!
//! Usage:
//!   cargo run -p svod-model --release --example modernbert_embed_infer -- "hello world"
//!   cargo run -p svod-model --release --example modernbert_embed_infer -- --profile "hello world"
//!   cargo run -p svod-model --release --example modernbert_embed_infer -- --repo answerdotai/ModernBERT-base --max-batch 4 "text"
//!   cargo run -p svod-model --release --example modernbert_embed_infer -- --window 512 --stride 256 < long_doc.txt
//!
//! Reads stdin when no positional text is given.

use std::io::{self, Read};
use std::time::Instant;

use clap::{Parser, ValueEnum};

use svod_arch::pipelines::text::{
    Chunker, Embed, EncoderHead, EncoderPipeline, RunOptions, SlidingWindowChunker, Tokenizer, TruncatingChunker,
};
use svod_dtype::DType;
use svod_model::encoder_head::HubOptions;
use svod_model::modernbert;

#[derive(Parser, Debug)]
#[command(about = "ModernBERT embeddings demo", long_about = None)]
struct Args {
    /// Text to embed (reads stdin if omitted).
    text: Option<String>,

    /// HF Hub repo with the ModernBERT weights + tokenizer.
    #[arg(long, default_value = "answerdotai/ModernBERT-base")]
    repo: String,

    /// HF Hub revision.
    #[arg(long, default_value = "main")]
    revision: String,

    /// Compute dtype (default: the device's, bf16 on a tensor-core GPU).
    #[arg(long, value_enum)]
    dtype: Option<Dtype>,

    /// Most chunks one execute runs.
    #[arg(long, default_value_t = 1)]
    max_batch: usize,

    /// Longest chunk in tokens, specials included (default: the checkpoint's
    /// `max_position_embeddings`). A batch runs at its own longest chunk.
    #[arg(long)]
    max_seq: Option<usize>,

    /// Collect and print the per-stage profile.
    #[arg(long)]
    profile: bool,

    /// Total window size (incl. special tokens) for SlidingWindowChunker. When
    /// set, `--stride` is required and the chunker switches from truncating to
    /// sliding-window.
    #[arg(long)]
    window: Option<usize>,

    /// Step between window starts (content-token advance). Required with `--window`.
    #[arg(long)]
    stride: Option<usize>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Dtype {
    F32,
    F16,
    Bf16,
}

fn embed_and_report<T, C, E>(
    pipeline: &mut EncoderPipeline<T, C, E>,
    text: &str,
    profile: bool,
) -> Result<(), Box<dyn std::error::Error>>
where
    T: Tokenizer,
    C: Chunker,
    E: Embed,
{
    println!("\nEmbedding {} chars...", text.len());
    let t = Instant::now();
    let result = pipeline.embed(text, RunOptions { profile })?;
    let dt = t.elapsed();

    println!("  {} chunk(s)", result.chunks.len());
    for (i, chunk) in result.chunks.iter().enumerate() {
        let v = &chunk.values;
        println!(
            "  chunk {i} @ byte {}: dim={} | L2={:.4} | first 5: {:?}",
            chunk.byte_offset,
            v.len(),
            v.iter().map(|x| x * x).sum::<f32>().sqrt(),
            &v[..v.len().min(5)],
        );
    }
    if let Some(prof) = &result.profile {
        println!("\n--- Profile ---\n{prof}");
    }
    println!("Embed: {:.3}s", dt.as_secs_f32());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let text = match args.text {
        Some(t) => t,
        None => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };

    let dtype = match args.dtype {
        None => svod_model::default_compute_dtype(),
        Some(Dtype::F32) => DType::Float32,
        Some(Dtype::F16) => DType::Float16,
        Some(Dtype::Bf16) => DType::BFloat16,
    };
    let opts = HubOptions {
        revision: args.revision.clone(),
        max_batch: args.max_batch,
        max_seq: args.max_seq,
        dtype,
        tokenizer_repo: None,
    };

    let t_total = Instant::now();
    println!("Loading ModernBERT from {} ({})...", args.repo, args.revision);
    let (tokenizer, embedder) = modernbert::from_hub(&args.repo, &opts)?;
    let (_, max_seq) = embedder.capacity();
    println!("Loaded: hidden_size={}, max_seq={}, max_batch={}", embedder.hidden_size(), max_seq, args.max_batch);

    match (args.window, args.stride) {
        (Some(window), Some(stride)) => {
            if window > max_seq {
                return Err(format!("--window {window} exceeds the model's max_seq {max_seq}").into());
            }
            let chunker =
                SlidingWindowChunker::try_new(window, stride).map_err(|e| format!("invalid --window/--stride: {e}"))?;
            println!("Chunker: sliding window={window}, stride={stride}");
            let mut pipeline = EncoderPipeline::new(tokenizer, chunker, embedder);
            embed_and_report(&mut pipeline, &text, args.profile)?;
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err("--window and --stride must be used together".into());
        }
        (None, None) => {
            println!("Chunker: truncating max_seq={max_seq}");
            let mut pipeline = EncoderPipeline::new(tokenizer, TruncatingChunker::new(max_seq), embedder);
            embed_and_report(&mut pipeline, &text, args.profile)?;
        }
    }

    println!("\nTotal: {:.2}s", t_total.elapsed().as_secs_f32());
    Ok(())
}
