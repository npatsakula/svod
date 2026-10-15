//! ModernBERT long-document token classification via the chunk-level seam.
//!
//! Demonstrates the [`EncoderPipeline`] chunk seam ([`classify_tokens_chunks`])
//! and document-level span merging ([`group_spans_document`]) — neither of which
//! the other examples exercise. A long input is tokenized + windowed **once**
//! with a [`SlidingWindowChunker`] (overlapping windows), then the pre-built
//! chunks are fed straight through the encoder via [`classify_tokens_chunks`],
//! skipping the pipeline's tokenize/chunk stages. Finally per-window token
//! labels are decoded with [`labels_for_tokens`] and merged across the
//! overlapping windows with [`group_spans_document`], which dedups entities that
//! appear in more than one window's overlap region.
//!
//! This is the long-document sibling of `modernbert_token_infer.rs` (which uses
//! a `TruncatingChunker` + per-chunk [`group_spans`]).
//!
//! Usage:
//!   cargo run -p svod-model --release --example modernbert_token_chunks_infer -- "Barack Obama was born in Hawaii and later served as U.S. president"
//!   cargo run -p svod-model --release --example modernbert_token_chunks_infer -- --window 128 --stride 64 < long_doc.txt
//!   cargo run -p svod-model --release --example modernbert_token_chunks_infer -- --repo sanketrai/modernbert-base-conll2003-english-ner --profile < long_doc.txt
//!
//! Reads stdin when no positional text is given. `--window`/`--stride` default to
//! `max_seq` / `max_seq/2` (50% overlap) so overlapping windows are exercised.

use std::io::{self, Read};
use std::time::Instant;

use clap::{Parser, ValueEnum};

use svod_arch::pipelines::text::{
    Chunker, ClassifyTokens, EncoderHead, EncoderPipeline, RunOptions, Scheme, SlidingWindowChunker, Tokenizer,
    group_spans_document, labels_for_tokens,
};
use svod_dtype::DType;
use svod_model::encoder_head::HubOptions;
use svod_model::modernbert;

#[derive(Parser, Debug)]
#[command(about = "ModernBERT long-doc token classification via the chunk seam", long_about = None)]
struct Args {
    /// Text to tag (reads stdin if omitted).
    text: Option<String>,

    /// HF Hub repo with the ModernBERT token-classification weights + tokenizer.
    #[arg(long, default_value = "sanketrai/modernbert-base-conll2003-english-ner")]
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

    /// Total window size (incl. special tokens). Defaults to the model's max_seq.
    #[arg(long)]
    window: Option<usize>,

    /// Step between window starts. Defaults to window/2 (50% overlap).
    #[arg(long)]
    stride: Option<usize>,

    /// Label scheme for span grouping: `bio`, `bilou`, `iobes`, `flat`.
    #[arg(long, default_value = "bio")]
    scheme: String,

    /// Collect and print the per-stage profile.
    #[arg(long)]
    profile: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Dtype {
    F32,
    F16,
    Bf16,
}

fn parse_scheme(s: &str) -> Result<Scheme, Box<dyn std::error::Error>> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "bio" | "iob2" => Scheme::Bio,
        "bilou" => Scheme::Bilou,
        "iobes" => Scheme::Iobes,
        "flat" | "none" => Scheme::Flat,
        other => return Err(format!("unknown scheme {other:?}; expected bio|bilou|iobes|flat").into()),
    })
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
    let scheme = parse_scheme(&args.scheme)?;
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
    println!("Loading ModernBERT token classifier from {} ({})...", args.repo, args.revision);
    let load = modernbert::from_hub_token_classification(&args.repo, &opts)?;
    let (_, max_seq) = load.classifier.capacity();
    let num_labels = load.classifier.num_labels();
    println!("Loaded: max_seq={max_seq}, max_batch={}, num_labels={num_labels}", args.max_batch);

    let id2label = load.id2label;
    let label_of = |id: u32| id2label.get(id as usize).map_or("", |s| s.as_str());

    // Resolve window/stride: default to max_seq with 50% overlap so the sliding
    // window is exercised even without explicit args.
    let window = args.window.unwrap_or(max_seq);
    let stride = args.stride.unwrap_or((window / 2).max(1));
    if window > max_seq {
        return Err(format!("--window {window} exceeds the model's max_seq {max_seq}").into());
    }
    let chunker =
        SlidingWindowChunker::try_new(window, stride).map_err(|e| format!("invalid --window/--stride: {e}"))?;
    println!("Chunker: sliding window={window}, stride={stride}");

    // The pipeline owns the tokenizer + chunker + classifier; the chunk seam
    // (classify_tokens_chunks) bypasses the first two, so we drive them
    // explicitly here to tokenize + window once, then feed the chunks back in.
    let mut pipeline = EncoderPipeline::new(load.tokenizer, chunker, load.classifier);

    println!("\nTagging {} chars via the chunk seam (scheme = {})...", text.len(), args.scheme);
    let t = Instant::now();

    // Tokenize + chunk once (the reusable part — these chunks could be fed to
    // any other pipeline too).
    let encoding = pipeline.tokenizer_mut().encode(&text)?;
    let chunks = pipeline.chunker_mut().chunk(&encoding)?;
    println!("  {} window(s)", chunks.len());

    // Feed the pre-built chunks straight through the encoder (skips
    // tokenize/chunk; its profile carries encoder stages only).
    let result = pipeline.classify_tokens_chunks(&chunks, RunOptions { profile: args.profile })?;
    let dt = t.elapsed();

    // Decode per-window token labels, then merge entity spans across the whole
    // document. group_spans_document dedups entities that land in two windows'
    // overlap region — the whole point of sliding-window NER.
    let per_window: Vec<_> = result.chunks.iter().map(|c| labels_for_tokens(c, label_of)).collect();
    for (i, tokens) in per_window.iter().enumerate() {
        println!("  window {i} @ byte {}: {} content token(s)", result.chunks[i].byte_offset, tokens.len());
    }

    let entities = group_spans_document(&per_window, scheme);
    println!("\n  → {} document entit(y/ies) (merged across windows):", entities.len());
    for e in entities {
        let surface = text.get(e.start..e.end).unwrap_or("<oob>");
        println!("     {:<8} [{},{}) {:?}  score={:.3}", e.label, e.start, e.end, surface, e.score);
    }

    if let Some(prof) = &result.profile {
        println!("\n--- Profile ---\n{prof}");
    }
    println!("ClassifyTokens (chunk seam): {:.3}s", dt.as_secs_f32());
    println!("\nTotal: {:.2}s", t_total.elapsed().as_secs_f32());
    Ok(())
}
