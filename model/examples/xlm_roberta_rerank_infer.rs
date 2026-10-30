//! Cross-encoder reranking with BGE-reranker-v2-m3.
//!
//! Loads an XLM-RoBERTa sequence-classification checkpoint (weights +
//! tokenizer) and scores each passage against one query through an
//! [`EncoderPipeline`]: `HfTokenizer` encodes each (query, passage) pair as one
//! sequence, a [`SlidingWindowChunker`] windows a passage longer than one
//! window (repeating the query in each), and [`classification_document`] keeps
//! a passage's best window (MaxP). The plans the run will take are warmed up
//! first, as a server warms the shapes it expects, so the timed run compiles
//! nothing. Prints the passages best first, with their sigmoid scores.
//!
//! Usage:
//!   cargo run -p svod-model --release --example xlm_roberta_rerank_infer -- "what is panda?" "hi" "The giant panda is a bear species endemic to China."
//!   cargo run -p svod-model --release --example xlm_roberta_rerank_infer -- --window 128 --profile "query" "passage one" "passage two"

use std::time::Instant;

use clap::{Parser, ValueEnum};

use svod_arch::pipelines::text::{
    Chunker, DocumentPooling, EncoderPipeline, PairTokenizer, RunOptions, SlidingWindowChunker,
    classification_document, sigmoid,
};
use svod_dtype::DType;
use svod_model::encoder_head::{HeadOptions, HubOptions};
use svod_model::xlm_roberta;

#[derive(Parser, Debug)]
#[command(about = "Cross-encoder reranking with an XLM-RoBERTa classifier", long_about = None)]
struct Args {
    /// The query.
    query: String,

    /// The passages to rank against it.
    #[arg(required = true)]
    passages: Vec<String>,

    /// HF Hub repo with the reranker weights + tokenizer.
    #[arg(long, default_value = "BAAI/bge-reranker-v2-m3")]
    repo: String,

    /// HF Hub revision.
    #[arg(long, default_value = "main")]
    revision: String,

    /// Compute dtype (default: the device's, bf16 on a tensor-core GPU).
    #[arg(long, value_enum)]
    dtype: Option<Dtype>,

    /// Window in tokens, query and specials included.
    #[arg(long, default_value_t = 512)]
    window: usize,

    /// Step between a long passage's windows, in passage tokens.
    #[arg(long, default_value_t = 384)]
    stride: usize,

    /// Most windows one execute runs.
    #[arg(long, default_value_t = 8)]
    max_batch: usize,

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

    let dtype = args.dtype.map(|dtype| match dtype {
        Dtype::F32 => DType::Float32,
        Dtype::F16 => DType::Float16,
        Dtype::Bf16 => DType::BFloat16,
    });
    let opts = HubOptions {
        revision: args.revision.clone(),
        head: HeadOptions { max_batch: args.max_batch, max_seq: Some(args.window), dtype, device: None },
        ..Default::default()
    };

    let t_total = Instant::now();
    println!("Loading reranker from {} ({})...", args.repo, args.revision);
    let (tokenizer, reranker) = xlm_roberta::from_hub_classifier(&args.repo, &opts)?;
    let chunker = SlidingWindowChunker::try_new(args.window, args.stride)?;
    let mut pipeline = EncoderPipeline::new(tokenizer, chunker, reranker)?;

    let pairs: Vec<(&str, &str)> = args.passages.iter().map(|p| (args.query.as_str(), p.as_str())).collect();

    // The run batches its windows longest first, `max_batch` at a time: window
    // the pairs once to see those batches' shapes, and warm their plans.
    let t = Instant::now();
    let mut lens = Vec::new();
    for enc in pipeline.tokenizer_mut().encode_pair_batch(&pairs)? {
        lens.extend(pipeline.chunker_mut().chunk(&enc)?.iter().map(|c| c.encoding.len()));
    }
    lens.sort_unstable_by(|a, b| b.cmp(a));
    let shapes: Vec<(usize, usize)> = lens.chunks(args.max_batch).map(|batch| (batch.len(), batch[0])).collect();
    pipeline.warm_up(&shapes)?;
    println!("Warm-up {shapes:?}: {:.3}s", t.elapsed().as_secs_f32());

    let t = Instant::now();
    let out = pipeline.classify_pair_batch(&pairs, RunOptions { profile: args.profile })?;
    let dt = t.elapsed();

    let mut ranked: Vec<(usize, usize, f32)> = out
        .results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let logits = classification_document(&r.chunks, DocumentPooling::Max).ok().flatten().unwrap_or_default();
            (i, r.chunks.len(), logits.first().copied().unwrap_or(f32::NEG_INFINITY))
        })
        .collect();
    ranked.sort_by(|a, b| b.2.total_cmp(&a.2));

    println!("\nQuery: {:?}", args.query);
    for (rank, (i, windows, logit)) in ranked.into_iter().enumerate() {
        let passage = &args.passages[i];
        let shown: String = passage.chars().take(70).collect();
        println!("  {rank}. score={:.4} logit={logit:+.3} windows={windows}  {shown:?}", sigmoid(logit));
    }

    if let Some(prof) = &out.profile {
        println!("\n--- Profile ---\n{prof}");
    }
    println!("Rerank: {:.3}s", dt.as_secs_f32());
    println!("\nTotal: {:.2}s", t_total.elapsed().as_secs_f32());
    Ok(())
}
