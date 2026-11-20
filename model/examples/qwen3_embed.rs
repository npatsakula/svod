//! Qwen3 embedding throughput benchmark: tokenize a text set with the
//! reference tokenizer, then for each batch size compile the plans the set
//! needs and time whole passes over it. Ends with a per-batch-size table in
//! kilotokens per second (real tokens only, padding excluded).
//!
//!   cargo run -p svod-model --release --example qwen3_embed -- --texts texts.txt
//!   qwen3_embed --repo <model dir> --texts texts.txt --device amd --dtype f16 --batches 1,8,32
//!
//! `--texts` is one text per line, or a JSON `{"texts": [...]}`; a JSON with an
//! `"ids"` array (the HF tokenizer's ids per text) also checks the tokenizer
//! against it. `--reference` is a safetensors file of `[N, D]` f32 rows in the
//! same order (keys `embeddings_*`), scored at every batch size.
//!
//! Outside a svod checkout BEAM needs its helper: copy `svod-beam-worker`
//! from the same build and set `SVOD_BEAM_WORKER` to its absolute path.
//!
//! The tokenizer is the caller's concern, so it lives here: Qwen2's byte-level
//! BPE on `tiktoken-rs` from the published `tokenizer.json`.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use serde::Deserialize;
use serde_json::Value;
use tiktoken_rs::CoreBPE;
use unicode_normalization::UnicodeNormalization;

use svod_device::registry::{self, DeviceSpecExt};
use svod_dtype::default_device::{default_device, set_default_device};
use svod_dtype::{DType, DeviceSpec, ScalarDType};
use svod_model::qwen3::{Qwen3Config, Qwen3Embedder, Qwen3Embedding, qwen3_embedding_0_6b};

type Error = Box<dyn std::error::Error>;

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// A progress line, stamped with the seconds since start.
macro_rules! say {
    ($($arg:tt)*) => {
        println!("[{:>7.1}s] {}", START.elapsed().as_secs_f64(), format_args!($($arg)*))
    };
}

/// How often a long step reports that it is still running.
const HEARTBEAT: Duration = Duration::from_secs(30);

#[derive(Parser, Debug)]
#[command(about = "Qwen3 embedding throughput benchmark", long_about = None)]
struct Args {
    /// HF Hub repo id, or a local directory with config.json, tokenizer.json
    /// and the safetensors.
    #[arg(long, default_value = "Qwen/Qwen3-Embedding-0.6B")]
    repo: String,

    /// Texts to embed: one per line, or JSON `{"texts": [...]}`.
    #[arg(long)]
    texts: PathBuf,

    /// Device: cpu, cuda, cuda:N, amd, amd:N (default: SVOD_DEVICE, else the
    /// platform's).
    #[arg(long)]
    device: Option<String>,

    /// Batch sizes to sweep: plan rows per execution.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16,32")]
    batches: Vec<usize>,

    /// Truncation length in tokens (rounded up to the length bucket).
    #[arg(long, default_value_t = 512)]
    max_len: usize,

    /// Timed passes over the text set per batch size.
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// Print the per-plan-run kernel profile of one pass per batch size.
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

/// The tokenized text set.
struct Corpus {
    rows: Vec<Vec<u32>>,
    tokens: usize,
}

/// One batch size's measurements.
struct Outcome {
    batch: usize,
    plans: usize,
    runs: usize,
    fill: f64,
    compile: Duration,
    passes: Vec<Duration>,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();
    if args.runs == 0 || args.batches.contains(&0) {
        return Err("--runs and every --batches entry must be positive".into());
    }
    let device = select_device(args.device.as_deref())?;
    tracing_subscriber::fmt::init();
    LazyLock::force(&START);

    let mut config = qwen3_embedding_0_6b();
    if let Some(dtype) = args.dtype {
        config.dtype = match dtype {
            Dtype::F32 => DType::Float32,
            Dtype::F16 => DType::Float16,
            Dtype::Bf16 => DType::BFloat16,
        };
    }
    say!("svod {} Qwen3 embedding benchmark", env!("CARGO_PKG_VERSION"));
    say!("device {}", describe(&device)?);
    say!(
        "dtype {}, BEAM {}, batches {:?}, {} timed passes each",
        dtype_name(&config.dtype),
        std::env::var("BEAM").unwrap_or_else(|_| "off".into()),
        args.batches,
        args.runs
    );

    say!("loading {} ...", args.repo);
    let started = Instant::now();
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
    say!("loaded in {:.1} s", started.elapsed().as_secs_f64());

    let max_len = Qwen3Embedder::new(model.clone(), 1, args.max_len).max_len();
    say!("tokenizing {} (max_len {max_len}) ...", args.texts.display());
    let corpus = tokenize(&tokenizer, &args.texts, max_len)?;

    let mut outcomes = Vec::new();
    for &batch in &args.batches {
        match bench_batch(&model, &corpus, batch, &args) {
            Ok(outcome) => outcomes.push(outcome),
            Err(e) => {
                say!("batch {batch} failed: {e}");
                summary(&outcomes, &corpus);
                return Err(e);
            }
        }
    }
    summary(&outcomes, &corpus);
    Ok(())
}

/// Pin `--device` before anything opens a runtime: exported for the worker
/// threads and the BEAM helper, and set on this thread.
fn select_device(spec: Option<&str>) -> Result<DeviceSpec, Error> {
    let Some(spec) = spec else { return Ok(default_device()) };
    let device = DeviceSpec::parse(spec)?;
    // Safety: single-threaded prologue, before any runtime threads spawn.
    unsafe { std::env::set_var("SVOD_DEVICE", device.canonicalize()) };
    set_default_device(device.clone());
    Ok(device)
}

fn dtype_name(dtype: &DType) -> String {
    match dtype.base() {
        ScalarDType::Float32 => "f32".into(),
        ScalarDType::Float16 => "f16".into(),
        ScalarDType::BFloat16 => "bf16".into(),
        other => format!("{other:?}"),
    }
}

/// The device spec with the GPU's name and architecture; opening it here
/// fails early on a device that is not there.
fn describe(device: &DeviceSpec) -> Result<String, Error> {
    let spec = device.canonicalize();
    Ok(match *device {
        DeviceSpec::Cuda { device_id } => {
            format!("{spec} ({}, {})", registry::resolve_cuda_name(device_id)?, registry::resolve_cuda_arch(device_id)?)
        }
        DeviceSpec::Amd { device_id } => format!("{spec} ({:?})", registry::resolve_amd_arch_from_topology(device_id)?),
        _ => spec,
    })
}

fn tokenize(tokenizer: &Tokenizer, path: &Path, max_len: usize) -> Result<Corpus, Error> {
    let started = Instant::now();
    let TextSet { texts, ids } = load_texts(path)?;
    let (rows, truncated): (Vec<Vec<u32>>, Vec<bool>) = texts.iter().map(|t| tokenizer.encode(t, max_len)).unzip();
    if let Some(golden) = ids {
        let mismatches = rows.iter().zip(&golden).filter(|(got, want)| got != want).count();
        say!("tokenizer vs reference ids: {mismatches} of {} texts differ", texts.len());
    }
    let mut lens: Vec<usize> = rows.iter().map(Vec::len).collect();
    lens.sort_unstable();
    let tokens = lens.iter().sum();
    let (Some(&min), Some(&max)) = (lens.first(), lens.last()) else { return Err("no texts to embed".into()) };
    say!(
        "{} texts, {tokens} tokens in {:.1} s: per text min {min}, median {}, max {max}; {} truncated",
        texts.len(),
        started.elapsed().as_secs_f64(),
        lens[lens.len() / 2],
        truncated.iter().filter(|&&t| t).count()
    );
    Ok(Corpus { rows, tokens })
}

/// Compile every plan the text set needs at `batch` rows, run one untimed
/// pass, then the timed ones.
fn bench_batch(model: &Qwen3Embedding, corpus: &Corpus, batch: usize, args: &Args) -> Result<Outcome, Error> {
    let mut embedder = Qwen3Embedder::new(model.clone(), batch, args.max_len);
    let buckets = embedder.buckets(&corpus.rows);
    let mut runs_per_bucket = BTreeMap::new();
    for &bucket in &buckets {
        *runs_per_bucket.entry(bucket).or_insert(0usize) += 1;
    }
    let fill = corpus.tokens as f64 / buckets.iter().map(|b| b * batch).sum::<usize>() as f64;
    let split: Vec<String> = runs_per_bucket.iter().rev().map(|(b, n)| format!("{n}×{b}")).collect();
    say!(
        "batch {batch}: {} plan runs a pass ({} tokens each), {:.1}% of plan tokens are text",
        buckets.len(),
        split.join(" + "),
        fill * 100.0
    );

    let started = Instant::now();
    for (i, &bucket) in runs_per_bucket.keys().rev().enumerate() {
        let what = format!("compiling plan {batch}×{bucket} ({}/{})", i + 1, runs_per_bucket.len());
        say!("{what} ...");
        let t = Instant::now();
        with_heartbeat(&what, || embedder.warm_up(bucket))?;
        say!("plan {batch}×{bucket} ready in {:.1} s", t.elapsed().as_secs_f64());
    }
    let compile = started.elapsed();

    say!("batch {batch}: untimed pass ...");
    let t = Instant::now();
    let embeddings = embedder.embed(&corpus.rows)?;
    say!("batch {batch} untimed pass: {:.3} s", t.elapsed().as_secs_f64());
    if let Some(reference) = &args.reference {
        score(&embeddings, reference)?;
    }
    let mut passes = Vec::with_capacity(args.runs);
    for run in 1..=args.runs {
        let t = Instant::now();
        embedder.embed(&corpus.rows)?;
        let wall = t.elapsed();
        say!(
            "batch {batch} pass {run}/{}: {:.3} s, {:.2} ktok/s",
            args.runs,
            wall.as_secs_f64(),
            ktok_per_s(corpus.tokens, wall)
        );
        passes.push(wall);
    }
    if args.profile {
        let (_, profile) = embedder.embed_profiled(&corpus.rows)?;
        println!("\n--- Profile, batch {batch} ---\n{}", profile.render_table());
    }
    Ok(Outcome { batch, plans: runs_per_bucket.len(), runs: buckets.len(), fill, compile, passes })
}

/// Run `work`, saying every [`HEARTBEAT`] that `what` is still going.
fn with_heartbeat<T>(what: &str, work: impl FnOnce() -> T) -> T {
    let (done, wait) = mpsc::channel::<()>();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let started = Instant::now();
            while let Err(RecvTimeoutError::Timeout) = wait.recv_timeout(HEARTBEAT) {
                say!("  … still {what}, {:.0} s so far", started.elapsed().as_secs_f64());
            }
        });
        let out = work();
        drop(done);
        out
    })
}

fn ktok_per_s(tokens: usize, wall: Duration) -> f64 {
    tokens as f64 / wall.as_secs_f64() / 1e3
}

fn summary(outcomes: &[Outcome], corpus: &Corpus) {
    if outcomes.is_empty() {
        return;
    }
    println!("\n{} texts, {} tokens a pass; ktok/s counts text tokens only", corpus.rows.len(), corpus.tokens);
    println!(
        "{:>5} {:>5} {:>9} {:>6} {:>10} {:>9} {:>9} {:>10} {:>11} {:>9}",
        "batch", "plans", "runs/pass", "fill", "compile s", "best s", "median s", "ktok/s", "ktok/s med", "texts/s"
    );
    for o in outcomes {
        let mut sorted = o.passes.clone();
        sorted.sort_unstable();
        let (best, median) = (sorted[0], sorted[sorted.len() / 2]);
        println!(
            "{:>5} {:>5} {:>9} {:>5.1}% {:>10.1} {:>9.3} {:>9.3} {:>10.2} {:>11.2} {:>9.1}",
            o.batch,
            o.plans,
            o.runs,
            o.fill * 100.0,
            o.compile.as_secs_f64(),
            best.as_secs_f64(),
            median.as_secs_f64(),
            ktok_per_s(corpus.tokens, best),
            ktok_per_s(corpus.tokens, median),
            corpus.rows.len() as f64 / best.as_secs_f64()
        );
    }
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
        // A NaN row must surface as the minimum; `f32::min` would skip it.
        let min = cosines.iter().copied().fold(f32::INFINITY, |m, c| if c.is_nan() || c < m { c } else { m });
        let mean = cosines.iter().sum::<f32>() / cosines.len() as f32;
        say!("vs {key}: cosine min {min:.5} mean {mean:.5}");
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
    /// `max_len - 1`, then the end-of-text token. The flag says whether the
    /// truncation cut anything.
    fn encode(&self, text: &str, max_len: usize) -> (Vec<u32>, bool) {
        let normalized: String = text.nfc().collect();
        let mut ids = self.bpe.encode_with_special_tokens(&normalized);
        let truncated = ids.len() >= max_len;
        ids.truncate(max_len.saturating_sub(1));
        ids.push(self.eos);
        (ids, truncated)
    }
}
