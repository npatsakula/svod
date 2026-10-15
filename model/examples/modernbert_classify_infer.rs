//! ModernBERT sentence classification demo.
//!
//! Loads a HuggingFace ModernBERT classification checkpoint (weights + tokenizer)
//! in one call and runs a text input through an [`EncoderPipeline`]:
//! `HfTokenizer` → `TruncatingChunker` → `ModernBertClassifier`. Prints raw
//! logits, argmax class, and softmax probabilities per chunk. The analog of
//! `modernbert_embed_infer.rs` for classification.
//!
//! Usage:
//!   cargo run -p svod-model --release --example modernbert_classify_infer -- "This movie was great!"
//!   cargo run -p svod-model --release --example modernbert_classify_infer -- --profile "Terrible waste of time"
//!   cargo run -p svod-model --release --example modernbert_classify_infer -- --repo AnkitAI/Sensible-ModernBERT-Sentiment-Analysis "text"
//!
//! Reads stdin when no positional text is given.

use std::io::{self, Read};
use std::time::Instant;

use clap::{Parser, ValueEnum};

use svod_arch::pipelines::text::{
    Classify, EncoderHead, EncoderPipeline, RunOptions, TruncatingChunker, argmax, softmax,
};
use svod_dtype::DType;
use svod_model::encoder_head::{HeadOptions, HubOptions};
use svod_model::modernbert;

#[derive(Parser, Debug)]
#[command(about = "ModernBERT sentence classification demo", long_about = None)]
struct Args {
    /// Text to classify (reads stdin if omitted).
    text: Option<String>,

    /// HF Hub repo with the ModernBERT classification weights + tokenizer.
    #[arg(long, default_value = "AnkitAI/Sensible-ModernBERT-Sentiment-Analysis")]
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
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Dtype {
    F32,
    F16,
    Bf16,
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

    let dtype = args.dtype.map(|dtype| match dtype {
        Dtype::F32 => DType::Float32,
        Dtype::F16 => DType::Float16,
        Dtype::Bf16 => DType::BFloat16,
    });
    let opts = HubOptions {
        revision: args.revision.clone(),
        head: HeadOptions { max_batch: args.max_batch, max_seq: args.max_seq, dtype, device: None },
        ..Default::default()
    };

    let t_total = Instant::now();
    println!("Loading ModernBERT classifier from {} ({})...", args.repo, args.revision);
    let (tokenizer, classifier) = modernbert::from_hub_classifier(&args.repo, &opts)?;
    let (_, max_seq) = classifier.capacity();
    let num_classes = classifier.num_labels();
    let id2label = classifier.id2label().to_vec();
    let label_of = |id: usize| id2label.get(id).map_or("", |s| s.as_str());
    println!("Loaded: max_seq={max_seq}, max_batch={}, num_classes={num_classes}", args.max_batch);

    println!("Chunker: truncating max_seq={max_seq}");
    let mut pipeline = EncoderPipeline::new(tokenizer, TruncatingChunker::new(max_seq), classifier);

    println!("\nClassifying {} chars...", text.len());
    let t = Instant::now();
    let result = pipeline.classify(&text, RunOptions { profile: args.profile })?;
    let dt = t.elapsed();

    println!("  {} chunk(s)", result.chunks.len());
    for (i, chunk) in result.chunks.iter().enumerate() {
        let probs = softmax(&chunk.logits);
        let best = argmax(&chunk.logits);
        let p = probs[best];
        let probs_str: Vec<String> = probs.iter().map(|p| format!("{p:.4}")).collect();
        println!(
            "  chunk {i} @ byte {}: logits={:?} → {best}={} (p={p:.4})  probs=[{}]",
            chunk.byte_offset,
            chunk.logits,
            label_of(best),
            probs_str.join(", "),
        );
    }

    if let Some(prof) = &result.profile {
        println!("\n--- Profile ---\n{prof}");
    }
    println!("Classify: {:.3}s", dt.as_secs_f32());
    println!("\nTotal: {:.2}s", t_total.elapsed().as_secs_f32());
    Ok(())
}
