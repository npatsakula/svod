//! Qwen3 embedding demo: tokenize a text set with the reference tokenizer,
//! embed it, and optionally score the vectors against reference embeddings
//! or print the per-batch kernel profile.
//!
//!   cargo run -p svod-model --release --example qwen3_embed -- --texts texts.txt
//!   cargo run -p svod-model --release --example qwen3_embed -- --texts data/qwen3/golden_tokens.json \
//!     --reference data/qwen3/hf_embeddings.safetensors --batch 8 --max-len 512 --profile
//!
//! `--texts` is one text per line, or a JSON `{"texts": [...]}`; a JSON with an
//! `"ids"` array (the HF tokenizer's ids per text) also checks the tokenizer
//! against it. `--reference` is a safetensors file of `[N, D]` f32 rows in the
//! same order (keys `embeddings_*`). Throughput on synthetic batches is
//! `cargo bench -p svod-model --bench qwen3_embed`.
//!
//! The tokenizer is the caller's concern, so it lives here: Qwen2's byte-level
//! BPE on `tiktoken-rs` from the published `tokenizer.json`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, ValueEnum};
use serde::Deserialize;
use serde_json::Value;
use tiktoken_rs::CoreBPE;
use unicode_normalization::UnicodeNormalization;

use svod_dtype::DType;
use svod_model::qwen3::{Qwen3Config, Qwen3Embedder, Qwen3Embedding, qwen3_embedding_0_6b};

type Error = Box<dyn std::error::Error>;

#[derive(Parser, Debug)]
#[command(about = "Qwen3 embedding demo", long_about = None)]
struct Args {
    /// HF Hub repo id, or a local directory with config.json, tokenizer.json
    /// and the safetensors.
    #[arg(long, default_value = "Qwen/Qwen3-Embedding-0.6B")]
    repo: String,

    /// Texts to embed: one per line, or JSON `{"texts": [...]}`.
    #[arg(long)]
    texts: PathBuf,

    /// Rows per prepared plan.
    #[arg(long, default_value_t = 8)]
    batch: usize,

    /// Truncation length in tokens (rounded up to the attention tile).
    #[arg(long, default_value_t = 512)]
    max_len: usize,

    /// Timed passes over the text set after the compiling first pass.
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// Print the per-batch kernel profile of one pass.
    #[arg(long)]
    profile: bool,

    /// Reference embeddings to score against (cosine per row).
    #[arg(long)]
    reference: Option<PathBuf>,

    /// Compute dtype (default: the device's, bf16 on a tensor-core GPU).
    #[arg(long, value_enum)]
    dtype: Option<Dtype>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Dtype {
    F32,
    F16,
    Bf16,
}

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let t_total = Instant::now();

    let mut config = qwen3_embedding_0_6b();
    if let Some(dtype) = args.dtype {
        config.dtype = match dtype {
            Dtype::F32 => DType::Float32,
            Dtype::F16 => DType::Float16,
            Dtype::Bf16 => DType::BFloat16,
        };
    }
    println!("Loading {} ({:?})...", args.repo, config.dtype);
    let dir = Path::new(&args.repo);
    let (model, tokenizer) = if dir.is_dir() {
        config.merge_structural_from(&Qwen3Config::from_json(&dir.join("config.json"))?);
        (Qwen3Embedding::from_safetensors_dir(dir, config)?, Tokenizer::from_json(&dir.join("tokenizer.json"))?)
    } else {
        let model = Qwen3Embedding::from_hub_with_revision(&args.repo, "main", &mut config)?;
        let (owner, name) = hf_hub::split_id(&args.repo);
        let path = hf_hub::HFClientSync::new()?.model(owner, name).download_file().filename("tokenizer.json").send()?;
        (model, Tokenizer::from_json(&path)?)
    };
    println!("Loaded in {:.2}s", t_total.elapsed().as_secs_f32());

    let mut embedder = Qwen3Embedder::new(model, args.batch, args.max_len);
    let TextSet { texts, ids } = load_texts(&args.texts)?;
    let rows: Vec<Vec<u32>> = texts.iter().map(|t| tokenizer.encode(t, embedder.max_len())).collect();
    if let Some(golden) = ids {
        let mismatches = rows.iter().zip(&golden).filter(|(got, want)| got != want).count();
        println!("Tokenizer vs reference ids: {mismatches} of {} texts differ", texts.len());
    }
    let tokens: usize = rows.iter().map(Vec::len).sum();
    println!("\n{} texts, {tokens} tokens, batch {} x max_len {}", texts.len(), args.batch, embedder.max_len());

    let t_first = Instant::now();
    let embeddings = embedder.embed(&rows)?;
    println!("First pass (compiles the plans): {:.2}s", t_first.elapsed().as_secs_f32());
    if let Some(reference) = &args.reference {
        score(&embeddings, reference)?;
    }

    let mut best = None;
    for run in 1..=args.runs {
        let started = Instant::now();
        embedder.embed(&rows)?;
        let wall = started.elapsed();
        println!("Run {run}/{}: {:.3}s", args.runs, wall.as_secs_f32());
        best = Some(best.map_or(wall, |b: std::time::Duration| b.min(wall)));
    }
    if let Some(best) = best {
        let secs = best.as_secs_f32();
        println!(
            "Best: {secs:.3}s  {:.1} texts/s  {:.1} ktok/s  ({:.2} ms per batch of {})",
            texts.len() as f32 / secs,
            tokens as f32 / secs / 1e3,
            secs * 1e3 / texts.len().div_ceil(args.batch) as f32,
            args.batch
        );
    }
    if args.profile {
        let (_, profile) = embedder.embed_profiled(&rows)?;
        println!("\n--- Profile ---\n{}", profile.render_table());
    }
    println!("\nTotal: {:.2}s", t_total.elapsed().as_secs_f32());
    Ok(())
}

/// The text set, with the reference tokenizer's ids when the file carries them.
struct TextSet {
    texts: Vec<String>,
    ids: Option<Vec<Vec<u32>>>,
}

fn load_texts(path: &Path) -> Result<TextSet, Error> {
    let data = std::fs::read_to_string(path)?;
    if path.extension().is_some_and(|e| e == "json") {
        let json: Value = serde_json::from_str(&data)?;
        let texts = json["texts"].as_array().ok_or("JSON needs a \"texts\" array")?;
        let texts = texts.iter().filter_map(|t| t.as_str().map(str::to_owned)).collect();
        let ids = json["ids"].as_array().map(|rows| {
            rows.iter()
                .map(|row| row.as_array().into_iter().flatten().filter_map(|v| v.as_u64().map(|v| v as u32)).collect())
                .collect()
        });
        return Ok(TextSet { texts, ids });
    }
    Ok(TextSet { texts: data.lines().filter(|l| !l.trim().is_empty()).map(str::to_owned).collect(), ids: None })
}

/// Cosine of each row against every `embeddings_*` key of the reference.
fn score(embeddings: &[Vec<f32>], reference: &Path) -> Result<(), Error> {
    let sd = svod_model::state::load_safetensors(reference)?;
    let mut keys: Vec<&String> = sd.keys().filter(|k| k.starts_with("embeddings")).collect();
    keys.sort();
    for key in keys {
        let want = sd[key].clone();
        want.realize()?;
        let want = want.as_vec::<f32>()?;
        let dim = embeddings.first().map_or(0, Vec::len);
        if want.len() != embeddings.len() * dim {
            return Err(format!("{key}: {} values for {} rows of {dim}", want.len(), embeddings.len()).into());
        }
        let cosines: Vec<f32> = embeddings
            .iter()
            .enumerate()
            .map(|(i, row)| row.iter().zip(&want[i * dim..]).map(|(a, b)| a * b).sum())
            .collect();
        let min = cosines.iter().copied().fold(f32::INFINITY, f32::min);
        let mean = cosines.iter().sum::<f32>() / cosines.len() as f32;
        println!("vs {key}: cosine min {min:.5} mean {mean:.5}");
    }
    Ok(())
}

// ─── The reference tokenizer ────────────────────────────────────────────────
//
// `Qwen2Tokenizer` via HF `tokenizers`: NFC, the pre-tokenizer regex,
// byte-level BPE, and one special token appended by the post-processor.
// Merge priority is the token id — the published merge list is in id order,
// which is what `CoreBPE`'s rank-driven merge needs.

struct Tokenizer {
    bpe: CoreBPE,
    /// The special token the post-processor appends to every sequence.
    eos: u32,
}

#[derive(Deserialize)]
struct TokenizerJson {
    model: BpeModel,
    added_tokens: Vec<AddedToken>,
    pre_tokenizer: Value,
    post_processor: Value,
}

#[derive(Deserialize)]
struct BpeModel {
    vocab: HashMap<String, u32>,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
}

/// Depth-first search for `key` in a JSON tree.
fn find<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    match value {
        Value::Object(map) => map.get(key).or_else(|| map.values().find_map(|child| find(child, key))),
        Value::Array(items) => items.iter().find_map(|child| find(child, key)),
        _ => None,
    }
}

/// GPT-2's byte → printable-unicode table, the alphabet `vocab` is written in.
fn byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    for (byte, slot) in table.iter_mut().enumerate() {
        let printable = matches!(byte as u8, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
        *slot = if printable {
            char::from_u32(byte as u32).expect("Latin-1 code point")
        } else {
            next += 1;
            char::from_u32(next - 1).expect("BMP code point")
        };
    }
    table
}

impl Tokenizer {
    fn from_json(path: &Path) -> Result<Self, Error> {
        let json: TokenizerJson = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let unicode_to_byte: HashMap<char, u8> =
            byte_to_unicode().iter().enumerate().map(|(byte, &c)| (c, byte as u8)).collect();
        let encoder: rustc_hash::FxHashMap<Vec<u8>, u32> = json
            .model
            .vocab
            .iter()
            .map(|(token, &id)| {
                let bytes = token
                    .chars()
                    .map(|c| {
                        unicode_to_byte.get(&c).copied().ok_or_else(|| format!("non byte-level vocab entry {token:?}"))
                    })
                    .collect::<Result<Vec<u8>, _>>()?;
                Ok((bytes, id))
            })
            .collect::<Result<_, Error>>()?;
        let specials: rustc_hash::FxHashMap<String, u32> =
            json.added_tokens.into_iter().map(|t| (t.content, t.id)).collect();

        let pattern =
            find(&json.pre_tokenizer, "Regex").and_then(Value::as_str).ok_or("pre_tokenizer has no Split regex")?;
        // The template must end in exactly one special token: `[A, <eos>]`.
        let appended = find(&json.post_processor, "single")
            .and_then(Value::as_array)
            .and_then(|single| single.last())
            .and_then(|last| find(last, "SpecialToken"))
            .and_then(|token| token.get("id"))
            .and_then(Value::as_str)
            .ok_or("post_processor template does not end in a special token")?;
        let eos = *specials.get(appended).ok_or_else(|| format!("unknown special token {appended:?}"))?;
        let bpe = CoreBPE::new(encoder, specials, pattern)?;
        Ok(Self { bpe, eos })
    }

    /// Token ids as the reference tokenizer produces them: NFC, special-token
    /// strings recognized in the text, byte-level BPE, truncation to
    /// `max_len - 1`, then the end-of-text token.
    fn encode(&self, text: &str, max_len: usize) -> Vec<u32> {
        let normalized: String = text.nfc().collect();
        let mut ids = self.bpe.encode_with_special_tokens(&normalized);
        ids.truncate(max_len.saturating_sub(1));
        ids.push(self.eos);
        ids
    }
}
