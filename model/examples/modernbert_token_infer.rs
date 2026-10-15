//! ModernBERT token classification demo (NER / POS / chunking).
//!
//! Loads a HuggingFace ModernBERT token-classification checkpoint (weights +
//! tokenizer) in one call and runs a text input through an [`EncoderPipeline`]:
//! `HfTokenizer` → `TruncatingChunker` → `ModernBertTokenClassifier`. Decodes
//! per-token labels with [`labels_for_tokens`] and groups entity spans with
//! [`group_spans`] under a chosen [`Scheme`]. The token-classification analog of
//! `modernbert_classify_infer.rs`.
//!
//! Usage:
//!   cargo run -p svod-model --release --example modernbert_token_infer -- "Barack Obama was born in Hawaii"
//!   cargo run -p svod-model --release --example modernbert_token_infer -- --scheme bio --profile "Apple is based in Cupertino"
//!   cargo run -p svod-model --release --example modernbert_token_infer -- --repo sanketrai/modernbert-base-conll2003-english-ner "text"
//!
//! Reads stdin when no positional text is given.

use std::io::{self, Read};
use std::time::Instant;

use clap::{Parser, ValueEnum};

use svod_arch::pipelines::text::{
    ClassifyTokens, EncoderHead, EncoderPipeline, RunOptions, Scheme, TruncatingChunker, group_spans, labels_for_tokens,
};
use svod_dtype::DType;
use svod_model::encoder_head::HubOptions;
use svod_model::modernbert;

#[derive(Parser, Debug)]
#[command(about = "ModernBERT token classification demo", long_about = None)]
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

    println!("Chunker: truncating max_seq={max_seq}");
    let mut pipeline = EncoderPipeline::new(load.tokenizer, TruncatingChunker::new(max_seq), load.classifier);

    println!("\nTagging {} chars (scheme = {})...", text.len(), args.scheme);
    let t = Instant::now();
    let result = pipeline.classify_tokens(&text, RunOptions { profile: args.profile })?;
    let dt = t.elapsed();

    println!("  {} chunk(s)", result.chunks.len());
    for (i, chunk) in result.chunks.iter().enumerate() {
        let tokens = labels_for_tokens(chunk, label_of);
        println!("  chunk {i} @ byte {}: {} content token(s)", chunk.byte_offset, tokens.len());
        for tok in &tokens {
            let surface = text.get(tok.start..tok.end).unwrap_or("<oob>");
            println!(
                "    token {}: {:<8} [{},{}) {:?}  score={:.3}",
                tok.token_index, tok.label, tok.start, tok.end, surface, tok.score
            );
        }
        let spans = group_spans(&tokens, scheme);
        if !spans.is_empty() {
            println!("    → {} entit(y/ies):", spans.len());
            for e in spans {
                let surface = text.get(e.start..e.end).unwrap_or("<oob>");
                println!("       {:<8} [{},{}) {:?}  score={:.3}", e.label, e.start, e.end, surface, e.score);
            }
        }
    }

    if let Some(prof) = &result.profile {
        println!("\n--- Profile ---\n{prof}");
    }
    println!("ClassifyTokens: {:.3}s", dt.as_secs_f32());
    println!("\nTotal: {:.2}s", t_total.elapsed().as_secs_f32());
    Ok(())
}
