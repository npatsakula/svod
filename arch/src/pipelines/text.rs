//! EncoderHead-only text inference (embeddings + classification + token
//! classification): tokenize → chunk → model → aggregate. Host-side and
//! model-agnostic: a model implements only its irreducible part behind the
//! [`EncoderHead`] supertrait — [`Embed`] / [`Classify`] / [`ClassifyTokens`] are thin
//! specializations that fix the per-chunk output kind — and the heavy machinery
//! (sub-batching, truncation geometry, profile assembly, span decoding) lives in
//! trait defaults and free functions here.
//!
//! ```text
//! Tokenizer::encode ─▶ Encoding ─▶ [TextChunk]            (Chunker)
//!                                     │
//!                                     ▼
//!   EncoderHead::run_batch(tokenized chunks) ─▶ [EncoderHead::Output]
//!                                     │  EncoderPipeline assembles the profile
//!                                     ▼
//!                          {Embeddings, Classifications, …}
//! ```
//!
//! Text crosses the boundary as `&str`, token ids as `&[u32]`: this crate stays
//! free of the Tensor/device stack. The model owns ids → device internally.
//!
//! ## Design rationale
//!
//! **One supertrait, three specializations.** [`EncoderHead`] is the whole
//! model contract the pipeline needs (one batched forward + capacity), so the
//! heavy machinery — sub-batching, chunk geometry, profile assembly — is written
//! once over it. [`Embed`] / [`Classify`] / [`ClassifyTokens`] each fix
//! [`EncoderHead::Output`] and add one informational accessor; the verb facades
//! are bounded inherent impls, so a caller sees only the verbs their model
//! unlocks, with no coherence conflict between tasks.
//!
//! **Host-side and model-agnostic.** This crate never touches the Tensor/device
//! stack (text in as `&str`, ids as `&[u32]`, outputs as plain `Vec<f32>`), so
//! one assembled pipeline is reusable across any encoder backend; the model owns
//! the ids → device step.
//!
//! **Chunk seam.** The verb `_chunks` variants feed pre-built [`TextChunk`]s,
//! skipping tokenize/chunk entirely — tokenize once, then reuse the same chunks
//! across several pipelines. Their profile carries encoder stages only.

#[cfg(feature = "hf-tokenizers")]
use std::path::Path;
use std::time::Instant;

use snafu::{ResultExt, Snafu, ensure};

pub use svod_runtime::RunProfile;
use svod_runtime::StageProfile;

// ─── Results ────────────────────────────────────────────────────────────────

/// One chunk's finished embedding — already pooled and normalized by the model.
/// `TruncatingChunker` yields exactly one per input; [`SlidingWindowChunker`]
/// yields one per window. Position-agnostic: the [`Embed`] trait returns this,
/// and [`EncoderPipeline`] attaches each one's source position (see
/// [`ChunkEmbedding`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Embedding {
    pub values: Vec<f32>,
}

/// One finished embedding paired with the byte position where its source
/// [`TextChunk`] began in the original text — the per-chunk pipeline result.
/// `byte_offset` lets [`SlidingWindowChunker`] (or a token
/// classification pipeline) tell windows apart and re-base per-token byte spans
/// back to the source — the same field the chunker already records on
/// [`TextChunk`], now threaded through to the output. `values` is the pooled +
/// normalized vector (the model returns it as [`Embedding`]; the chunk result
/// flattens it for direct indexing).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChunkEmbedding {
    pub byte_offset: usize,
    pub values: Vec<f32>,
}

/// Aggregated pipeline output: one [`ChunkEmbedding`] per chunk, plus the
/// optional per-stage [`RunProfile`] the encoder collected. Profile stages are
/// free-form and extensible.
#[derive(Debug, Default)]
pub struct Embeddings {
    pub chunks: Vec<ChunkEmbedding>,
    pub profile: Option<RunProfile>,
}

/// Batch embedding result from [`embed_batch`](EncoderPipeline::embed_batch):
/// one [`Embeddings`] per input text, plus a shared [`RunProfile`] covering the
/// entire batch run. The profile is batch-level (tokenize, chunk, and embed are
/// timed as a whole), not per-text — each [`Embeddings`] inside `results`
/// carries `profile: None`.
#[derive(Debug, Default)]
pub struct BatchEmbeddings {
    pub results: Vec<Embeddings>,
    pub profile: Option<RunProfile>,
}

/// Per-call run switch, orthogonal to a pipeline's construction config
/// (sizing). Defaults to `false`, so one built pipeline serves profiled and
/// unprofiled runs without rebuilding.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunOptions {
    /// Collect a per-stage [`RunProfile`] on [`Embeddings::profile`].
    pub profile: bool,
}

impl From<()> for RunOptions {
    fn from(_: ()) -> Self {
        Self::default()
    }
}

// ─── Encoding (tokenizer output) ────────────────────────────────────────────

/// The tokenized form of one input: ids and the masks/offsets HF `tokenizers`
/// produces natively. All integer fields are `Vec<u32>` to match HF's types (no
/// casts in the adapter); the model casts to its buffer dtype at the boundary.
#[derive(Clone, Debug)]
pub struct Encoding {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    /// Zeros for RoBERTa / ModernBERT; carried for BERT-style pair inputs.
    pub token_type_ids: Vec<u32>,
    /// Byte offsets into the source text (HF `tokenizers` default). Each entry
    /// is the `(start, end)` byte span of one token; byte offsets (not char) so
    /// they index `&str` directly. Token classification re-bases per-token
    /// entity spans through these.
    pub offsets: Vec<(usize, usize)>,
    pub special_tokens_mask: Vec<u32>,
}

impl Encoding {
    /// Copy the five getter slices out of an HF `Encoding`. The borrow ends
    /// here — the adapter owns its own `Vec`s, so the source can be dropped.
    #[cfg(feature = "hf-tokenizers")]
    pub fn from_hf(enc: &tokenizers::Encoding) -> Self {
        Self {
            input_ids: enc.get_ids().to_vec(),
            attention_mask: enc.get_attention_mask().to_vec(),
            token_type_ids: enc.get_type_ids().to_vec(),
            offsets: enc.get_offsets().to_vec(),
            special_tokens_mask: enc.get_special_tokens_mask().to_vec(),
        }
    }

    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }
}

// ─── Tokenizer ──────────────────────────────────────────────────────────────

/// Text → [`Encoding`]. Implement [`encode`](Tokenizer::encode) (the primary
/// op); [`encode_batch`](Tokenizer::encode_batch) defaults to looping it.
pub trait Tokenizer {
    type Error: std::error::Error + 'static;

    fn encode(&mut self, text: &str) -> Result<Encoding, Self::Error>;

    /// Encode several inputs. Defaults to looping [`encode`](Tokenizer::encode).
    fn encode_batch(&mut self, texts: &[&str]) -> Result<Vec<Encoding>, Self::Error> {
        texts.iter().map(|t| self.encode(t)).collect()
    }
}

#[cfg(feature = "hf-tokenizers")]
/// Owns a `tokenizers::Error` (which is `Box<dyn std::error::Error + Send +
/// Sync>`) behind a sized, `Error`-implementing type. The boxed trait object is
/// `!Sized`, and std's `impl<E: Error> Error for Box<E>` requires `E: Sized`, so
/// `tokenizers::Error` itself does **not** satisfy `std::error::Error + Sized`
/// and cannot serve as a trait's `type Error` or a snafu `source` field.
/// Wrapping it here — manually, since snafu can only build on types that already
/// implement `Error` — restores a concrete, nameable error type that does.
/// Mirrors how every model crate (`wespeaker::Error`, `resnet::Error`, …) wraps
/// its inner errors; here the inner is already boxed, so the wrapper is a plain
/// newtype.
#[derive(Debug)]
pub struct HfTokenizerError(tokenizers::Error);

#[cfg(feature = "hf-tokenizers")]
impl std::fmt::Display for HfTokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

#[cfg(feature = "hf-tokenizers")]
impl std::error::Error for HfTokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.0)
    }
}

#[cfg(feature = "hf-tokenizers")]
impl From<tokenizers::Error> for HfTokenizerError {
    fn from(err: tokenizers::Error) -> Self {
        Self(err)
    }
}

#[cfg(feature = "hf-tokenizers")]
/// The provided [`Tokenizer`] impl: wraps a HuggingFace `tokenizers::Tokenizer`.
///
/// `encode` calls `inner.encode(text, add_special_tokens = true)`. It does
/// **not** configure truncation/padding — the [`Chunker`] owns the `max_seq`
/// policy, so [`SlidingWindowChunker`] can still see the full token stream.
///
/// HF fetching (`from_hub(repo)`) is intentionally absent here — `hf-hub` lives
/// in `svod-model`, which fetches `tokenizer.json` and hands the cached path to
/// [`from_path`](HfTokenizer::from_path).
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

#[cfg(feature = "hf-tokenizers")]
impl HfTokenizer {
    pub fn new(inner: tokenizers::Tokenizer) -> Self {
        Self { inner }
    }

    /// Load from a `tokenizer.json` path.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, HfTokenizerError> {
        let inner = tokenizers::Tokenizer::from_file(path)?;
        Ok(Self::new(inner))
    }

    /// Load from `tokenizer.json` bytes (e.g. fetched via `hf-hub` in
    /// `svod-model`).
    pub fn from_bytes<P: AsRef<[u8]>>(bytes: P) -> Result<Self, HfTokenizerError> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes)?;
        Ok(Self::new(inner))
    }
}

#[cfg(feature = "hf-tokenizers")]
impl Tokenizer for HfTokenizer {
    type Error = HfTokenizerError;

    fn encode(&mut self, text: &str) -> Result<Encoding, HfTokenizerError> {
        let enc = self.inner.encode(text, true)?;
        Ok(Encoding::from_hf(&enc))
    }

    fn encode_batch(&mut self, texts: &[&str]) -> Result<Vec<Encoding>, HfTokenizerError> {
        let inputs: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let encs = self.inner.encode_batch(inputs, true)?;
        Ok(encs.iter().map(Encoding::from_hf).collect())
    }
}

// ─── Chunker (chunk source) ─────────────────────────────────────────────────

/// One chunked slice of an [`Encoding`], plus the byte offset where it begins
/// in the source text. `byte_offset` lets a future token-classification
/// pipeline re-base per-token byte spans back to the original text.
#[derive(Clone, Debug)]
pub struct TextChunk {
    pub encoding: Encoding,
    pub byte_offset: usize,
}

/// Turns an [`Encoding`] into ordered [`TextChunk`]s; [`EncoderPipeline`] is
/// generic over it.
pub trait Chunker {
    type Error: std::error::Error + 'static;

    /// The sequence length this chunker targets — the value threaded into the
    /// encoder's JIT at assembly (see [`EncoderPipeline::assemble`]).
    fn max_seq(&self) -> usize;

    fn chunk(&mut self, enc: &Encoding) -> Result<Vec<TextChunk>, Self::Error>;

    /// Stage name for this chunker's wall in a profiled run (e.g. `"chunk"`).
    /// The pipeline's `embed`/`classify`/`classify_tokens` methods time `chunk` and
    /// record it under this label *when that call requests a profile*. Defaults
    /// to `"chunk"`.
    fn profile_label(&self) -> &'static str {
        "chunk"
    }
}

/// Drops ids beyond `max_seq` and emits a single chunk at `byte_offset = 0`. For
/// long documents that exceed `max_seq`, use [`SlidingWindowChunker`].
#[derive(Debug)]
pub struct TruncatingChunker {
    max_seq: usize,
}

/// Construction error for [`TruncatingChunker`]: the only invariant is
/// `max_seq >= 1` (zero would slice to an empty chunk). [`chunk`](Chunker::chunk)
/// itself is infallible, so this type carries no chunk-time variants.
#[derive(Debug, Snafu)]
pub enum TruncatingChunkerError {
    #[snafu(display("max_seq must be >= 1"))]
    MaxSeqTooSmall,
}

impl TruncatingChunker {
    /// Fallible form of [`new`](Self::new): returns `Err` unless `max_seq >= 1`.
    pub fn try_new(max_seq: usize) -> Result<Self, TruncatingChunkerError> {
        ensure!(max_seq >= 1, MaxSeqTooSmallSnafu);
        Ok(Self { max_seq })
    }

    /// Panics unless `max_seq >= 1`. A convenience over
    /// [`try_new`](Self::try_new) for callers with known-valid args.
    pub fn new(max_seq: usize) -> Self {
        Self::try_new(max_seq).unwrap_or_else(|e| match e {
            TruncatingChunkerError::MaxSeqTooSmall => panic!("max_seq must be >= 1"),
        })
    }
}

impl Chunker for TruncatingChunker {
    type Error = std::convert::Infallible;

    fn max_seq(&self) -> usize {
        self.max_seq
    }

    fn chunk(&mut self, enc: &Encoding) -> Result<Vec<TextChunk>, std::convert::Infallible> {
        // Keep every field's length consistent: slice to the common cap so ids,
        // masks, and offsets stay aligned.
        let take = self.max_seq.min(enc.input_ids.len());
        let encoding = Encoding {
            input_ids: enc.input_ids[..take].to_vec(),
            attention_mask: enc.attention_mask[..take].to_vec(),
            token_type_ids: enc.token_type_ids[..take].to_vec(),
            offsets: enc.offsets[..take].to_vec(),
            special_tokens_mask: enc.special_tokens_mask[..take].to_vec(),
        };
        Ok(vec![TextChunk { encoding, byte_offset: 0 }])
    }
}

// ─── SlidingWindowChunker ────────────────────────────────────────────────────

/// Detect boundary special-token counts: leading and trailing runs of
/// `special_tokens_mask == 1`. Interior specials (e.g. `[UNK]`) remain content.
fn boundary_specials(mask: &[u32]) -> (usize, usize) {
    let lead = mask.iter().take_while(|&&m| m == 1).count();
    let trail = mask.iter().rev().take_while(|&&m| m == 1).count().min(mask.len().saturating_sub(lead));
    (lead, trail)
}

/// Reassemble one windowed [`Encoding`]: `lead` boundary specials from the head,
/// the content body `[body_start..body_end]`, then `trail` specials from the
/// tail. All five fields are built in lockstep.
fn build_window(enc: &Encoding, lead: usize, trail: usize, content_start: usize, content_end: usize) -> Encoding {
    let body_start = lead + content_start;
    let body_end = lead + content_end;
    let tail_start = enc.input_ids.len() - trail;

    fn reassemble<T: Clone>(v: &[T], lead: usize, body_start: usize, body_end: usize, tail_start: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(lead + (body_end - body_start) + (v.len() - tail_start));
        out.extend_from_slice(&v[..lead]);
        out.extend_from_slice(&v[body_start..body_end]);
        out.extend_from_slice(&v[tail_start..]);
        out
    }

    Encoding {
        input_ids: reassemble(&enc.input_ids, lead, body_start, body_end, tail_start),
        attention_mask: reassemble(&enc.attention_mask, lead, body_start, body_end, tail_start),
        token_type_ids: reassemble(&enc.token_type_ids, lead, body_start, body_end, tail_start),
        offsets: reassemble(&enc.offsets, lead, body_start, body_end, tail_start),
        special_tokens_mask: reassemble(&enc.special_tokens_mask, lead, body_start, body_end, tail_start),
    }
}

/// Windows a long token stream into overlapping chunks of `window` total tokens
/// (specials + content), advancing `stride` tokens per step. Each window is a
/// well-formed sequence: boundary specials (`[CLS]`/`[SEP]` for BERT/ModernBERT)
/// are re-attached to every window, detected generically from
/// `special_tokens_mask` rather than hardcoded.
///
/// `window` is the total per-chunk length — `max_seq()` returns it so the
/// encoder JIT is sized for full windows. `stride` is the **step** between
/// consecutive window starts in content-token space: `stride == window` gives
/// adjacent (non-overlapping) chunks; `stride < window` gives an overlap of
/// `window - stride` tokens. Clamped to the content length so `stride == window`
/// never skips content despite specials consuming part of the budget.
///
/// The last window may be shorter than `window` when content doesn't divide
/// evenly — the embedder pads it. Each [`TextChunk`] carries the byte offset of
/// its first content token as `byte_offset`.
#[derive(Debug)]
pub struct SlidingWindowChunker {
    window: usize,
    stride: usize,
}

/// Construction / chunk error for [`SlidingWindowChunker`]. The two
/// construction variants are rejected by [`try_new`](SlidingWindowChunker::try_new);
/// [`WindowTooSmallForSpecials`](Self::WindowTooSmallForSpecials) is detected at
/// [`chunk`](Chunker::chunk) time, since the boundary-special budget comes from
/// the tokenizer's runtime output, not the constructor args.
#[derive(Debug, Snafu)]
pub enum SlidingWindowChunkerError {
    #[snafu(display("window must be >= 1"))]
    WindowTooSmall,
    #[snafu(display("stride must be in 1..=window"))]
    StrideOutOfRange,
    #[snafu(display("window ({window}) too small for {specials} boundary special tokens"))]
    WindowTooSmallForSpecials { window: usize, specials: usize },
}

impl SlidingWindowChunker {
    /// `window` = total sequence length per chunk (incl. specials);
    /// `stride` = step between window starts. Fallible form of [`new`](Self::new):
    /// returns `Err` unless `1 <= stride <= window`.
    pub fn try_new(window: usize, stride: usize) -> Result<Self, SlidingWindowChunkerError> {
        ensure!(window >= 1, WindowTooSmallSnafu);
        ensure!((1..=window).contains(&stride), StrideOutOfRangeSnafu);
        Ok(Self { window, stride })
    }

    /// `window` = total sequence length per chunk (incl. specials);
    /// `stride` = step between window starts. Panics unless `1 <= stride <= window`.
    /// A convenience over [`try_new`](Self::try_new) for callers with known-valid args.
    pub fn new(window: usize, stride: usize) -> Self {
        Self::try_new(window, stride).unwrap_or_else(|e| match e {
            SlidingWindowChunkerError::WindowTooSmall => panic!("window must be >= 1"),
            SlidingWindowChunkerError::StrideOutOfRange => panic!("stride must be in 1..=window"),
            SlidingWindowChunkerError::WindowTooSmallForSpecials { .. } => {
                unreachable!(
                    "WindowTooSmallForSpecials is only returned by `Chunker::chunk` — when a \
                     window can't fit the tokenizer's boundary specials — never by `try_new`; \
                     reaching it here means that construction-time invariant has changed"
                )
            }
        })
    }
}

impl Chunker for SlidingWindowChunker {
    type Error = SlidingWindowChunkerError;

    fn max_seq(&self) -> usize {
        self.window
    }

    fn chunk(&mut self, enc: &Encoding) -> Result<Vec<TextChunk>, SlidingWindowChunkerError> {
        let (lead, trail) = boundary_specials(&enc.special_tokens_mask);
        let content_len = enc.input_ids.len().saturating_sub(lead + trail);
        if content_len == 0 {
            return Ok(Vec::new());
        }

        let specials = lead + trail;
        let content_window = self.window.saturating_sub(specials);
        ensure!(content_window >= 1, WindowTooSmallForSpecialsSnafu { window: self.window, specials });

        let step = self.stride.min(content_window);

        let mut chunks = Vec::new();
        let mut start = 0;
        loop {
            let end = (start + content_window).min(content_len);
            let byte_offset = enc.offsets.get(lead + start).map_or(0, |o| o.0);
            chunks.push(TextChunk { encoding: build_window(enc, lead, trail, start, end), byte_offset });
            if end >= content_len {
                break;
            }
            start += step;
        }

        Ok(chunks)
    }
}

// ─── Classify results ────────────────────────────────────────────────────────

/// One chunk's raw class logits — the model applies the classification head
/// (and any fused pooling / normalization) before returning. Position-agnostic:
/// the [`Classify`] trait returns this, and [`EncoderPipeline`] attaches each
/// one's source position (see [`ChunkClassification`]). The caller derives
/// predictions (argmax, softmax, thresholding) from these logits.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Classification {
    pub logits: Vec<f32>,
}

/// One chunk's logits paired with the byte position where its source
/// [`TextChunk`] began — the per-chunk pipeline result, mirroring
/// [`ChunkEmbedding`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChunkClassification {
    pub byte_offset: usize,
    pub logits: Vec<f32>,
}

/// Aggregated classification output: one [`ChunkClassification`] per chunk,
/// plus the optional per-stage [`RunProfile`]. Mirrors [`Embeddings`].
#[derive(Debug, Default)]
pub struct Classifications {
    pub chunks: Vec<ChunkClassification>,
    pub profile: Option<RunProfile>,
}

/// Batch classification result from
/// [`classify_batch`](EncoderPipeline::classify_batch): one
/// [`Classifications`] per input text, plus a shared [`RunProfile`] covering
/// the entire batch run. Mirrors [`BatchEmbeddings`].
#[derive(Debug, Default)]
pub struct BatchClassifications {
    pub results: Vec<Classifications>,
    pub profile: Option<RunProfile>,
}

// ─── Token-classification results ────────────────────────────────────────────

/// One chunk's raw per-token logits — the model applies the token head (HF
/// `ModernBertPredictionHead` + `classifier`) to every position before returning.
/// Position-agnostic: the [`ClassifyTokens`] trait returns this, and
/// [`EncoderPipeline`] attaches each one's source position and per-token byte
/// spans (see [`ChunkTokenClassification`]). `logits` is a flat row-major
/// `(seq_len, num_labels)` grid where `seq_len` is the chunk's live token count
/// (padding already stripped); `seq_len = logits.len() / num_labels`. The caller
/// derives predictions (argmax, softmax, thresholding) and decodes spans from
/// these.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TokenClassification {
    pub logits: Vec<f32>,
    pub num_labels: usize,
}

/// One chunk's per-token logits paired with the byte position where its source
/// [`TextChunk`] began, plus the per-token geometry the caller needs to decode
/// spans — the per-chunk pipeline result, mirroring [`ChunkClassification`] and
/// [`ChunkEmbedding`]. `token_offsets` and `special_tokens_mask` run in lockstep
/// over the chunk's `seq_len` tokens (one row per token in `logits`).
///
/// `token_offsets` are **source-absolute** byte spans (the HF tokenizer returns
/// source-referential offsets, preserved through chunking), so `&text[start..end]`
/// slices a token directly — no rebase by `byte_offset` is needed. `byte_offset`
/// is kept for chunk ordering and cross-chunk grouping (the byte offset of the
/// chunk's first content token), consistent with the sibling result types.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChunkTokenClassification {
    pub byte_offset: usize,
    pub logits: Vec<f32>,
    pub num_labels: usize,
    pub token_offsets: Vec<(usize, usize)>,
    pub special_tokens_mask: Vec<u32>,
}

/// Aggregated token-classification output: one [`ChunkTokenClassification`] per
/// chunk, plus the optional per-stage [`RunProfile`]. Mirrors [`Classifications`].
#[derive(Debug, Default)]
pub struct TokenClassifications {
    pub chunks: Vec<ChunkTokenClassification>,
    pub profile: Option<RunProfile>,
}

/// Batch token-classification result from
/// [`classify_tokens_batch`](EncoderPipeline::classify_tokens_batch): one
/// [`TokenClassifications`] per input text, plus a shared [`RunProfile`]
/// covering the entire batch run. Mirrors [`BatchClassifications`].
#[derive(Debug, Default)]
pub struct BatchTokenClassifications {
    pub results: Vec<TokenClassifications>,
    pub profile: Option<RunProfile>,
}

// ─── EncoderHead (per-chunk model) ──────────────────────────────────────────────

/// Any encoder-only head: turns tokenized chunks into a per-chunk
/// position-agnostic output. The shared shape behind the three task traits
/// ([`Embed`] / [`Classify`] / [`ClassifyTokens`]) — they fix [`Output`](EncoderHead::Output)
/// and add one informational accessor each, so the pipeline machinery
/// (sub-batching, profile assembly) is written once over `EncoderHead` and the verbs
/// stay discoverable per task.
///
/// Implement [`run_batch`](EncoderHead::run_batch) (the throughput path — one padded
/// JIT execute); [`run`](EncoderHead::run) defaults to a batch-of-one. Pairing a
/// model output with its chunk's source position is the pipeline's job, not the
/// model's — the verb facades ([`EncoderPipeline::embed`] / [`classify`] /
/// [`classify_tokens`]) do it from the chunk + output types, so a model implements only
/// its irreducible part.
///
/// [`classify`]: EncoderPipeline::classify
/// [`classify_tokens`]: EncoderPipeline::classify_tokens
pub trait EncoderHead {
    /// Per-chunk model output, position-agnostic: [`Embedding`] /
    /// [`Classification`] / [`TokenClassification`].
    type Output: Default;
    type Error: std::error::Error + 'static;

    /// `(max_batch, max_seq)`: the most chunks one [`run_batch`](EncoderHead::run_batch)
    /// takes, and the longest chunk. The pipeline sub-batches to `max_batch`;
    /// [`assemble`](EncoderPipeline::assemble) sizes the encoder from the
    /// chunker's `max_seq`, so the two agree by construction.
    fn capacity(&self) -> (usize, usize);

    /// Run every chunk through the model (the model owns its batching/padding),
    /// returning one [`Output`](EncoderHead::Output) per input plus the per-stage
    /// [`RunProfile`] — populated only when `profile` is set (a per-call choice,
    /// so the same encoder serves profiled and unprofiled runs).
    fn run_batch(&mut self, batch: &[&Encoding], profile: bool) -> Result<ModelOutputs<Self>, Self::Error>;

    /// Run one chunk + its optional profile (batch-of-one fallback).
    fn run(&mut self, enc: &Encoding, profile: bool) -> Result<(Self::Output, Option<RunProfile>), Self::Error> {
        let (mut values, prof) = self.run_batch(&[enc], profile)?;
        Ok((values.pop().unwrap_or_default(), prof))
    }
}

/// Finished-embeddings head. A specialization of [`EncoderHead`] fixing
/// [`Output`](EncoderHead::Output) = [`Embedding`]; implement
/// [`EncoderHead::run_batch`] plus [`hidden_size`](Embed::hidden_size).
pub trait Embed: EncoderHead<Output = Embedding> {
    /// The model's hidden size — the length of each finished [`Embedding`] (the
    /// model pools the sequence dimension before returning it).
    fn hidden_size(&self) -> usize;
}

/// Sentence/text-classification head. A specialization of [`EncoderHead`] fixing
/// [`Output`](EncoderHead::Output) = [`Classification`]; implement
/// [`EncoderHead::run_batch`] plus [`num_labels`](Classify::num_labels).
pub trait Classify: EncoderHead<Output = Classification> {
    /// The model's label count — the length of each [`Classification`]'s
    /// `logits` vector.
    fn num_labels(&self) -> usize;
}

/// Token-classification head (NER, POS tagging, chunking, …). A specialization
/// of [`EncoderHead`] fixing [`Output`](EncoderHead::Output) = [`TokenClassification`];
/// implement [`EncoderHead::run_batch`] plus [`num_labels`](ClassifyTokens::num_labels).
pub trait ClassifyTokens: EncoderHead<Output = TokenClassification> {
    /// The label count — the trailing dim of each [`TokenClassification`]'s
    /// `logits` grid.
    fn num_labels(&self) -> usize;
}

// ─── EncoderPipeline (composer) ─────────────────────────────────────────────

/// The full pipeline: a [`Tokenizer`] + a [`Chunker`] + an [`EncoderHead`]. One
/// generic host-side composer for embeddings, classification, and token
/// classification — `embed` / `classify` / `classify_tokens` (and their `_batch`
/// siblings) are bounded facades over the same machinery, so a caller grepping
/// for "how do I embed" still finds a named verb. Build with
/// [`assemble`](EncoderPipeline::assemble) to size the encoder from the
/// chunker's `max_seq`, or [`new`](EncoderPipeline::new) to
/// compose three already-built parts.
pub struct EncoderPipeline<T: Tokenizer, C: Chunker, M: EncoderHead> {
    tokenizer: T,
    chunker: C,
    model: M,
}

/// Error from an [`EncoderPipeline`] run — one of the tokenize / chunk / encode
/// stage failures. The single error folds the three sub-trait errors; with one
/// pipeline there are no selector collisions, so no `#[snafu(module)]`.
#[derive(Debug, Snafu)]
pub enum EncoderPipelineError<
    T: std::error::Error + 'static,
    C: std::error::Error + 'static,
    M: std::error::Error + 'static,
> {
    #[snafu(display("tokenizing: {source}"))]
    Tokenize { source: T },
    #[snafu(display("chunking: {source}"))]
    Chunk { source: C },
    #[snafu(display("encoding: {source}"))]
    Encode { source: M },
}

/// The three sub-trait errors folded into [`EncoderPipelineError`]. Spelled once
/// here so the verb facades' signatures stay readable (the three-param form
/// trips `clippy::type_complexity`).
type PipelineError<T, C, M> =
    EncoderPipelineError<<T as Tokenizer>::Error, <C as Chunker>::Error, <M as EncoderHead>::Error>;

/// `(per-input model outputs, optional profile)` — [`EncoderHead::run_batch`]'s
/// result shape. Aliased because the inline form trips `clippy::type_complexity`.
type ModelOutputs<M> = (Vec<<M as EncoderHead>::Output>, Option<RunProfile>);

/// `(position-agnostic outputs, the source chunks they map to, optional profile)`
/// — the single-text run shape returned by [`EncoderPipeline::run_single_inner`].
/// The verb facades pair the parallel `outputs` / `chunks` into position-attached
/// results. Aliased (`clippy::type_complexity`).
type SingleRun<M> = (Vec<<M as EncoderHead>::Output>, Vec<TextChunk>, Option<RunProfile>);

/// `(per-text outputs, per-text source chunks, optional profile)` — the batch
/// run shape returned by [`EncoderPipeline::run_batch_inner`]. Aliased.
type BatchRun<M> = (Vec<Vec<<M as EncoderHead>::Output>>, Vec<Vec<TextChunk>>, Option<RunProfile>);

/// `(per-text model outputs, optional profile)` — the batch chunk-seam shape
/// returned by [`EncoderPipeline::run_chunks_batch`]. No chunks are carried: on
/// the seam path the caller owns them. Aliased (`clippy::type_complexity`).
type BatchModelOutputs<M> = (Vec<Vec<<M as EncoderHead>::Output>>, Option<RunProfile>);

impl<T: Tokenizer, C: Chunker, M: EncoderHead> EncoderPipeline<T, C, M> {
    pub fn new(tokenizer: T, chunker: C, model: M) -> Self {
        Self { tokenizer, chunker, model }
    }

    /// Build the encoder sized to the chunker's `max_seq`
    /// ([`Chunker::max_seq`]), then compose — so the caller never hand-threads
    /// the sequence length between chunker and encoder.
    pub fn assemble<EE>(tokenizer: T, chunker: C, build: impl FnOnce(usize) -> Result<M, EE>) -> Result<Self, EE> {
        let model = build(chunker.max_seq())?;
        Ok(Self::new(tokenizer, chunker, model))
    }

    /// Run a flat list of chunks through the encoder, sub-batching to respect
    /// its `max_batch` capacity. Sub-batches are formed longest chunk first, so
    /// a head that pads each batch to its longest chunk pads short chunks
    /// against short ones; the outputs come back in input order. Returns
    /// position-agnostic per-chunk outputs plus a merged profile (one per
    /// sub-batch, accumulated via [`RunProfile::merge`]). The verb facades pair
    /// these with the source chunks' positions afterwards. Shared by the
    /// single-text and batch paths.
    fn run_chunks_flat(&mut self, chunks: &[&TextChunk], profile: bool) -> Result<ModelOutputs<M>, M::Error> {
        if chunks.is_empty() {
            return Ok((Vec::new(), None));
        }
        let max_batch = self.model.capacity().0.max(1);
        let mut order: Vec<usize> = (0..chunks.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(chunks[i].encoding.len()));
        let mut all: Vec<M::Output> = std::iter::repeat_with(M::Output::default).take(chunks.len()).collect();
        let mut merged_prof: Option<RunProfile> = None;
        for batch in order.chunks(max_batch) {
            let encodings: Vec<&Encoding> = batch.iter().map(|&i| &chunks[i].encoding).collect();
            let (values, prof) = self.model.run_batch(&encodings, profile)?;
            for (&i, value) in batch.iter().zip(values) {
                all[i] = value;
            }
            if let Some(p) = prof {
                merged_prof.get_or_insert_with(RunProfile::default).merge(p);
            }
        }
        Ok((all, merged_prof))
    }

    /// Tokenize → chunk → encode → assemble profile for a single text. Returns
    /// the position-agnostic outputs, the source chunks they map to, and the
    /// assembled profile; the verb facades pair outputs↔chunks into the named
    /// aggregate ([`Embeddings`] / [`Classifications`] / …). [`RunOptions`] is a
    /// per-call switch: the same pipeline serves profiled and unprofiled runs
    /// without rebuilding. When `opts.profile` is set, the tokenize and chunk
    /// stages are timed and recorded ahead of the encoder's stages — so the
    /// profile leads with `[tokenize, <chunker label>, <encoder …>]`.
    fn run_single_inner(&mut self, text: &str, opts: RunOptions) -> Result<SingleRun<M>, PipelineError<T, C, M>> {
        let t = Instant::now();
        let encoding = self.tokenizer.encode(text).context(TokenizeSnafu)?;
        let tok_wall = t.elapsed();

        let t = Instant::now();
        let chunks = self.chunker.chunk(&encoding).context(ChunkSnafu)?;
        let chunk_wall = t.elapsed();

        let chunk_refs: Vec<&TextChunk> = chunks.iter().collect();
        let (outputs, enc_prof) = self.run_chunks_flat(&chunk_refs, opts.profile).context(EncodeSnafu)?;

        let profile = opts.profile.then(|| {
            let mut p = RunProfile::default();
            p.push(StageProfile::host("tokenize", tok_wall));
            p.push(StageProfile::host(self.chunker.profile_label(), chunk_wall));
            if let Some(rest) = enc_prof {
                p.merge(rest);
            }
            p
        });
        Ok((outputs, chunks, profile))
    }

    /// Tokenize → chunk → encode multiple texts in one call. All chunks from all
    /// texts are flattened into a single stream and batched through the encoder
    /// (sub-batched to `model.capacity().0` when the total exceeds it),
    /// maximizing throughput. Returns one (outputs, chunks) pair per input text —
    /// moved out of the flat stream by each text's chunk count — plus a shared
    /// batch-level profile.
    fn run_batch_inner(&mut self, texts: &[&str], opts: RunOptions) -> Result<BatchRun<M>, PipelineError<T, C, M>> {
        if texts.is_empty() {
            return Ok((Vec::new(), Vec::new(), None));
        }

        let t = Instant::now();
        let encodings = self.tokenizer.encode_batch(texts).context(TokenizeSnafu)?;
        let tok_wall = t.elapsed();

        let t = Instant::now();
        let mut all_chunks: Vec<TextChunk> = Vec::new();
        let mut counts: Vec<usize> = Vec::with_capacity(texts.len());
        for enc in &encodings {
            let chunks = self.chunker.chunk(enc).context(ChunkSnafu)?;
            counts.push(chunks.len());
            all_chunks.extend(chunks);
        }
        let chunk_wall = t.elapsed();

        let chunk_refs: Vec<&TextChunk> = all_chunks.iter().collect();
        let (all_outs, enc_prof) = self.run_chunks_flat(&chunk_refs, opts.profile).context(EncodeSnafu)?;

        // Reassemble per-text: drain the flat outputs AND the flat chunks by
        // each text's chunk count (moving, not cloning).
        let mut per_text_outs = Vec::with_capacity(texts.len());
        let mut per_text_chunks = Vec::with_capacity(texts.len());
        let mut out_iter = all_outs.into_iter();
        let mut chunk_iter = all_chunks.into_iter();
        for &count in &counts {
            per_text_outs.push(out_iter.by_ref().take(count).collect::<Vec<M::Output>>());
            per_text_chunks.push(chunk_iter.by_ref().take(count).collect::<Vec<TextChunk>>());
        }

        let profile = opts.profile.then(|| {
            let mut p = RunProfile::default();
            p.push(StageProfile::host("tokenize", tok_wall));
            p.push(StageProfile::host(self.chunker.profile_label(), chunk_wall));
            if let Some(rest) = enc_prof {
                p.merge(rest);
            }
            p
        });
        Ok((per_text_outs, per_text_chunks, profile))
    }

    /// Run pre-built [`TextChunk`]s through the encoder, skipping tokenize and
    /// chunk entirely. Sub-batches to respect the encoder's `max_batch`. Returns
    /// position-agnostic per-chunk outputs (in order) plus the encoder-only
    /// profile (no tokenize/chunk stages, since those ran before this call). The
    /// task verbs ([`embed_chunks`](Self::embed_chunks) /
    /// [`classify_chunks`](Self::classify_chunks) /
    /// [`classify_tokens_chunks`](Self::classify_tokens_chunks)) pair these outputs with the
    /// caller's chunks into the named, position-attached aggregate — that is the
    /// cross-pipeline reuse path (tokenize + chunk once, feed the same
    /// `Vec<TextChunk>` to several pipelines). Only the
    /// [`Encode`](EncoderPipelineError::Encode) arm is reachable here.
    fn run_chunks(
        &mut self,
        chunks: &[TextChunk],
        opts: RunOptions,
    ) -> Result<ModelOutputs<M>, PipelineError<T, C, M>> {
        let chunk_refs: Vec<&TextChunk> = chunks.iter().collect();
        self.run_chunks_flat(&chunk_refs, opts.profile).context(EncodeSnafu)
    }

    /// Run pre-built per-text chunk lists through the encoder — the batch form
    /// of [`run_chunks`](Self::run_chunks). All chunks across all texts are
    /// flattened into one stream and sub-batched across text boundaries (same
    /// throughput shape as [`run_batch_inner`](Self::run_batch_inner)), then
    /// re-split per text by each input's chunk count. Returns per-text
    /// position-agnostic outputs plus the encoder-only profile (no
    /// tokenize/chunk stages, since those ran before this call). Only the
    /// [`Encode`](EncoderPipelineError::Encode) arm is reachable here.
    fn run_chunks_batch(
        &mut self,
        per_text: &[Vec<TextChunk>],
        opts: RunOptions,
    ) -> Result<BatchModelOutputs<M>, PipelineError<T, C, M>> {
        let counts: Vec<usize> = per_text.iter().map(Vec::len).collect();
        let flat: Vec<&TextChunk> = per_text.iter().flatten().collect();
        let (all_outs, profile) = self.run_chunks_flat(&flat, opts.profile).context(EncodeSnafu)?;
        let mut out_iter = all_outs.into_iter();
        let results = counts.into_iter().map(|c| out_iter.by_ref().take(c).collect::<Vec<M::Output>>()).collect();
        Ok((results, profile))
    }

    pub fn tokenizer_mut(&mut self) -> &mut T {
        &mut self.tokenizer
    }

    pub fn chunker_mut(&mut self) -> &mut C {
        &mut self.chunker
    }

    /// Generic typed accessor for the encoder model. The task facades also
    /// expose `embedder_mut` / `classifier_mut` / `token_classifier_mut` aliases.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

// ─── Verb facades ───────────────────────────────────────────────────────────
//
// `embed` / `classify` / `classify_tokens` (and their `_batch` siblings) are bounded
// inherent impls over the same `EncoderPipeline`. Distinct bounds (Embed vs
// Classify vs ClassifyTokens) ⇒ no coherence conflict; a caller gets only the verbs
// their model's trait unlocks. Each wraps the generic inner into the task's
// named aggregate, pairing the model's position-agnostic outputs with their
// source chunks' positions (the old per-model `EncoderHead::attach`).

/// Pair finished embeddings with their source chunks' byte offsets — the
/// position-attachment step the embed facade runs after [`EncoderPipeline::run_single_inner`].
fn attach_embed(outputs: Vec<Embedding>, chunks: &[TextChunk]) -> Vec<ChunkEmbedding> {
    chunks.iter().zip(outputs).map(|(c, o)| ChunkEmbedding { byte_offset: c.byte_offset, values: o.values }).collect()
}

/// Pair class logits with their source chunks' byte offsets — the classify facade's
/// attachment step.
fn attach_classify(outputs: Vec<Classification>, chunks: &[TextChunk]) -> Vec<ChunkClassification> {
    chunks
        .iter()
        .zip(outputs)
        .map(|(c, o)| ChunkClassification { byte_offset: c.byte_offset, logits: o.logits })
        .collect()
}

/// Pair per-token logits with their source chunks' positions and per-token byte
/// geometry — the classify_tokens facade's attachment step. The only task that pulls
/// extra fields (`offsets` / `special_tokens_mask`) from the chunk.
fn attach_token(outputs: Vec<TokenClassification>, chunks: &[TextChunk]) -> Vec<ChunkTokenClassification> {
    chunks
        .iter()
        .zip(outputs)
        .map(|(c, o)| ChunkTokenClassification {
            byte_offset: c.byte_offset,
            token_offsets: c.encoding.offsets.clone(),
            special_tokens_mask: c.encoding.special_tokens_mask.clone(),
            logits: o.logits,
            num_labels: o.num_labels,
        })
        .collect()
}

impl<T: Tokenizer, C: Chunker, M: Embed> EncoderPipeline<T, C, M> {
    /// Tokenize → chunk → embed → assemble profile. With
    /// [`RunOptions::profile`] set, the profile leads with the tokenize and chunk
    /// stages, then the encoder's.
    pub fn embed(&mut self, text: &str, opts: impl Into<RunOptions>) -> Result<Embeddings, PipelineError<T, C, M>> {
        let opts = opts.into();
        let (outputs, chunks, profile) = self.run_single_inner(text, opts)?;
        Ok(Embeddings { chunks: attach_embed(outputs, &chunks), profile })
    }

    /// Tokenize → chunk → embed multiple texts in one call. Returns one
    /// [`Embeddings`] per input text — each carrying its own [`ChunkEmbedding`]s
    /// with correct `byte_offset`s — plus a shared batch-level [`RunProfile`] on
    /// [`BatchEmbeddings::profile`].
    pub fn embed_batch(
        &mut self,
        texts: &[&str],
        opts: impl Into<RunOptions>,
    ) -> Result<BatchEmbeddings, PipelineError<T, C, M>> {
        let opts = opts.into();
        let (per_text_outs, per_text_chunks, profile) = self.run_batch_inner(texts, opts)?;
        let results = per_text_outs
            .into_iter()
            .zip(per_text_chunks.iter())
            .map(|(outs, chunks)| Embeddings { chunks: attach_embed(outs, chunks), profile: None })
            .collect();
        Ok(BatchEmbeddings { results, profile })
    }

    /// Embed pre-built chunks — the chunk-level seam, wrapped into
    /// [`Embeddings`]: tokenize and chunk are skipped, so the profile holds the
    /// encoder's stages only.
    pub fn embed_chunks(
        &mut self,
        chunks: &[TextChunk],
        opts: impl Into<RunOptions>,
    ) -> Result<Embeddings, PipelineError<T, C, M>> {
        let opts = opts.into();
        let (outputs, profile) = self.run_chunks(chunks, opts)?;
        Ok(Embeddings { chunks: attach_embed(outputs, chunks), profile })
    }

    /// Embed pre-built per-text chunk lists — the batch chunk-level seam,
    /// wrapped into [`BatchEmbeddings`]: chunks are sub-batched across texts
    /// and re-split per text, and the profile holds the encoder's stages only.
    pub fn embed_chunks_batch(
        &mut self,
        per_text: &[Vec<TextChunk>],
        opts: impl Into<RunOptions>,
    ) -> Result<BatchEmbeddings, PipelineError<T, C, M>> {
        let opts = opts.into();
        let (per_text_outs, profile) = self.run_chunks_batch(per_text, opts)?;
        let results = per_text_outs
            .into_iter()
            .zip(per_text.iter())
            .map(|(outs, chunks)| Embeddings { chunks: attach_embed(outs, chunks), profile: None })
            .collect();
        Ok(BatchEmbeddings { results, profile })
    }

    /// Typed accessor for the embedder (the [`Embed`] model).
    pub fn embedder_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

impl<T: Tokenizer, C: Chunker, M: Classify> EncoderPipeline<T, C, M> {
    /// Tokenize → chunk → classify → assemble profile. With
    /// [`RunOptions::profile`] set, the profile leads with the tokenize and chunk
    /// stages, then the encoder's.
    pub fn classify(
        &mut self,
        text: &str,
        opts: impl Into<RunOptions>,
    ) -> Result<Classifications, PipelineError<T, C, M>> {
        let opts = opts.into();
        let (outputs, chunks, profile) = self.run_single_inner(text, opts)?;
        Ok(Classifications { chunks: attach_classify(outputs, &chunks), profile })
    }

    /// Tokenize → chunk → classify multiple texts in one call. Returns one
    /// [`Classifications`] per input text — each carrying its own
    /// [`ChunkClassification`]s with correct `byte_offset`s — plus a shared
    /// batch-level [`RunProfile`] on [`BatchClassifications::profile`].
    pub fn classify_batch(
        &mut self,
        texts: &[&str],
        opts: impl Into<RunOptions>,
    ) -> Result<BatchClassifications, PipelineError<T, C, M>> {
        let opts = opts.into();
        let (per_text_outs, per_text_chunks, profile) = self.run_batch_inner(texts, opts)?;
        let results = per_text_outs
            .into_iter()
            .zip(per_text_chunks.iter())
            .map(|(outs, chunks)| Classifications { chunks: attach_classify(outs, chunks), profile: None })
            .collect();
        Ok(BatchClassifications { results, profile })
    }

    /// Classify pre-built chunks — the chunk-level seam, wrapped into
    /// [`Classifications`]: tokenize and chunk are skipped, so the profile
    /// holds the encoder's stages only.
    pub fn classify_chunks(
        &mut self,
        chunks: &[TextChunk],
        opts: impl Into<RunOptions>,
    ) -> Result<Classifications, PipelineError<T, C, M>> {
        let opts = opts.into();
        let (outputs, profile) = self.run_chunks(chunks, opts)?;
        Ok(Classifications { chunks: attach_classify(outputs, chunks), profile })
    }

    /// Classify pre-built per-text chunk lists — the batch chunk-level seam,
    /// wrapped into [`BatchClassifications`]: chunks are sub-batched across
    /// texts and re-split per text, and the profile holds the encoder's stages
    /// only.
    pub fn classify_chunks_batch(
        &mut self,
        per_text: &[Vec<TextChunk>],
        opts: impl Into<RunOptions>,
    ) -> Result<BatchClassifications, PipelineError<T, C, M>> {
        let opts = opts.into();
        let (per_text_outs, profile) = self.run_chunks_batch(per_text, opts)?;
        let results = per_text_outs
            .into_iter()
            .zip(per_text.iter())
            .map(|(outs, chunks)| Classifications { chunks: attach_classify(outs, chunks), profile: None })
            .collect();
        Ok(BatchClassifications { results, profile })
    }

    /// Typed accessor for the classifier (the [`Classify`] model).
    pub fn classifier_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

impl<T: Tokenizer, C: Chunker, M: ClassifyTokens> EncoderPipeline<T, C, M> {
    /// Tokenize → chunk → classify_tokens → assemble profile. With
    /// [`RunOptions::profile`] set, the profile leads with the tokenize and chunk
    /// stages, then the encoder's. Decode spans with [`labels_for_tokens`] /
    /// [`group_spans`].
    pub fn classify_tokens(
        &mut self,
        text: &str,
        opts: impl Into<RunOptions>,
    ) -> Result<TokenClassifications, PipelineError<T, C, M>> {
        let opts = opts.into();
        let (outputs, chunks, profile) = self.run_single_inner(text, opts)?;
        Ok(TokenClassifications { chunks: attach_token(outputs, &chunks), profile })
    }

    /// Tokenize → chunk → classify_tokens multiple texts in one call. Returns one
    /// [`TokenClassifications`] per input text — each carrying its own
    /// [`ChunkTokenClassification`]s with correct positions — plus a shared
    /// batch-level [`RunProfile`] on [`BatchTokenClassifications::profile`].
    pub fn classify_tokens_batch(
        &mut self,
        texts: &[&str],
        opts: impl Into<RunOptions>,
    ) -> Result<BatchTokenClassifications, PipelineError<T, C, M>> {
        let opts = opts.into();
        let (per_text_outs, per_text_chunks, profile) = self.run_batch_inner(texts, opts)?;
        let results = per_text_outs
            .into_iter()
            .zip(per_text_chunks.iter())
            .map(|(outs, chunks)| TokenClassifications { chunks: attach_token(outs, chunks), profile: None })
            .collect();
        Ok(BatchTokenClassifications { results, profile })
    }

    /// ClassifyTokens pre-built chunks — the chunk-level seam, wrapped into
    /// [`TokenClassifications`]: tokenize and chunk are skipped, so the profile
    /// holds the encoder's stages only.
    pub fn classify_tokens_chunks(
        &mut self,
        chunks: &[TextChunk],
        opts: impl Into<RunOptions>,
    ) -> Result<TokenClassifications, PipelineError<T, C, M>> {
        let opts = opts.into();
        let (outputs, profile) = self.run_chunks(chunks, opts)?;
        Ok(TokenClassifications { chunks: attach_token(outputs, chunks), profile })
    }

    /// ClassifyTokens pre-built per-text chunk lists — the batch chunk-level
    /// seam, wrapped into [`BatchTokenClassifications`]: chunks are sub-batched
    /// across texts and re-split per text, and the profile holds the encoder's
    /// stages only.
    pub fn classify_tokens_chunks_batch(
        &mut self,
        per_text: &[Vec<TextChunk>],
        opts: impl Into<RunOptions>,
    ) -> Result<BatchTokenClassifications, PipelineError<T, C, M>> {
        let opts = opts.into();
        let (per_text_outs, profile) = self.run_chunks_batch(per_text, opts)?;
        let results = per_text_outs
            .into_iter()
            .zip(per_text.iter())
            .map(|(outs, chunks)| TokenClassifications { chunks: attach_token(outs, chunks), profile: None })
            .collect();
        Ok(BatchTokenClassifications { results, profile })
    }

    /// Typed accessor for the token classifier (the [`ClassifyTokens`] model).
    pub fn token_classifier_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

// ─── Span / label decoding (host free-fns) ──────────────────────────────────

/// Label scheme for [`group_spans`]. `Bio`/`Bilou`/`Iobes` reconstruct multi-
/// token entity spans from prefixed labels (`B-PER`, `I-PER`, `L-PER`, `E-PER`,
/// `S-PER`, `O`, …); `Flat` is a no-op — one span per token (POS tagging,
/// chunking without a prefix scheme).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    Bio,
    Bilou,
    Iobes,
    Flat,
}

/// One content token's decoded prediction: the argmax label id, its display
/// name (borrowing from the caller's label table), its source-absolute byte
/// span, and its confidence `score` (the softmax probability of the argmax
/// label). Produced by [`labels_for_tokens`]; consumed by [`group_spans`].
/// Special tokens are already filtered out, so `token_index` runs contiguously
/// over the chunk's content tokens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenLabel<'a> {
    pub label_id: u32,
    pub label: &'a str,
    pub start: usize,
    pub end: usize,
    /// Index within the content-token vec (contiguous from 0).
    pub token_index: usize,
    /// Softmax probability of the argmax label — the per-token confidence.
    pub score: f32,
}

/// One decoded span: a contiguous run of same-type tokens (NER under a prefix
/// scheme) or a single token (POS / `Scheme::Flat`). `start`/`end` are
/// source-absolute byte offsets; `token_range` indexes the content-token vec;
/// `score` is the **minimum** member-token confidence (the conservative
/// aggregate for thresholding — a span survives a `score >= t` filter only when
/// every constituent token does). Owned (the `label` is detached from the
/// caller's label table via [`TokenLabel`]'s `&str` → `String` promotion inside
/// [`group_spans`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Entity {
    pub label: String,
    pub label_id: u32,
    pub start: usize,
    pub end: usize,
    pub token_range: std::ops::Range<usize>,
    /// Conservative confidence: the min of the member tokens' [`TokenLabel`]
    /// scores (single-token spans carry that token's score unchanged).
    pub score: f32,
}

/// Argmax each token's logits, drop special tokens, and pair every surviving
/// content token with its source byte span, display label, and confidence
/// score. `id2label` resolves a label id to its name (e.g. `"B-PER"`) as a
/// borrowed `&str` — zero per-token allocation when the label table is a
/// `Vec<String>` / `&[&str]`. Tokens whose `special_tokens_mask` is set are
/// skipped. The `score` is the softmax probability of the argmax label (max of
/// [`softmax`] over the row), computed in f32. This is the per-token view (POS /
/// chunking / [`Scheme::Flat`]); follow with [`group_spans`] to reconstruct
/// entity spans under a prefix scheme.
pub fn labels_for_tokens<'a, F>(chunk: &ChunkTokenClassification, id2label: F) -> Vec<TokenLabel<'a>>
where
    F: Fn(u32) -> &'a str,
{
    let nl = chunk.num_labels.max(1);
    let mut out = Vec::new();
    let mut content_idx = 0;
    for (t, (row, special)) in chunk.logits.chunks_exact(nl).zip(&chunk.special_tokens_mask).enumerate() {
        if *special != 0 {
            continue;
        }
        let id = argmax(row);
        // Argmax probability = softmax(row)[id], computed in place (max-subtracted
        // for stability) rather than allocating the full probability vector. An
        // empty row scores 0.0; an all-zero row yields the uniform 1/nl.
        let score = if row.is_empty() {
            0.0
        } else {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
            (row[id] - max).exp() / sum
        };
        let label_id = id as u32;
        let (start, end) = chunk.token_offsets.get(t).copied().unwrap_or((0, 0));
        out.push(TokenLabel { label_id, label: id2label(label_id), start, end, token_index: content_idx, score });
        content_idx += 1;
    }
    out
}

/// Group a content-token label sequence into [`Entity`] spans under the given
/// [`Scheme`]. Scans left-to-right, opening a span on a `B`/`U`/`S` (or a stray
/// `I`/`L`/`E` whose type mismatches the open span — treated leniently as a new
/// opener), extending on matching `I`, and closing on `O` / a type change / an
/// explicit `L`/`E` closer. `Scheme::Flat` emits one entity per token. Broken
/// transitions never panic: a stray `I-` opens a fresh span (HF's lenient
/// `group_entities` convention).
pub fn group_spans(tokens: &[TokenLabel<'_>], scheme: Scheme) -> Vec<Entity> {
    if matches!(scheme, Scheme::Flat) {
        return tokens
            .iter()
            .map(|t| Entity {
                label: t.label.to_string(),
                label_id: t.label_id,
                start: t.start,
                end: t.end,
                token_range: t.token_index..t.token_index + 1,
                score: t.score,
            })
            .collect();
    }

    let mut entities: Vec<Entity> = Vec::new();
    // Open span: (type, label_id, start_byte, end_byte, start_token_index, min_score)
    let mut open: Option<(String, u32, usize, usize, usize, f32)> = None;

    for t in tokens {
        let (prefix, typ) = parse_tag(t.label, scheme);
        let typ = typ.unwrap_or_else(|| t.label.to_string());
        match prefix {
            Prefix::O => {
                close_span(&mut open, &mut entities, t.token_index);
            }
            Prefix::B | Prefix::U | Prefix::S => {
                close_span(&mut open, &mut entities, t.token_index);
                if matches!(prefix, Prefix::U | Prefix::S) {
                    entities.push(Entity {
                        label: typ,
                        label_id: t.label_id,
                        start: t.start,
                        end: t.end,
                        token_range: t.token_index..t.token_index + 1,
                        score: t.score,
                    });
                } else {
                    open = Some((typ, t.label_id, t.start, t.end, t.token_index, t.score));
                }
            }
            Prefix::I => {
                if let Some((ot, _, _, oe, _, ms)) = open.as_mut()
                    && *ot == typ
                {
                    *oe = t.end;
                    *ms = (*ms).min(t.score);
                    continue;
                }
                // Stray I- (no matching open span): lenient — open a fresh span.
                close_span(&mut open, &mut entities, t.token_index);
                open = Some((typ, t.label_id, t.start, t.end, t.token_index, t.score));
            }
            Prefix::L | Prefix::E => {
                let matches_open = open.as_ref().is_some_and(|(ot, _, _, _, _, _)| *ot == typ);
                if matches_open {
                    let (_, lid, start, _, start_token, min_so_far) = open.take().unwrap();
                    entities.push(Entity {
                        label: typ,
                        label_id: lid,
                        start,
                        end: t.end,
                        token_range: start_token..t.token_index + 1,
                        score: min_so_far.min(t.score),
                    });
                    continue;
                }
                // Stray L-/E-: emit a single-token span.
                close_span(&mut open, &mut entities, t.token_index);
                entities.push(Entity {
                    label: typ,
                    label_id: t.label_id,
                    start: t.start,
                    end: t.end,
                    token_range: t.token_index..t.token_index + 1,
                    score: t.score,
                });
            }
        }
    }
    // Flush any span open at end-of-sequence. Its end-token index is the count
    // of content tokens (range exclusive end).
    close_span(&mut open, &mut entities, tokens.len());

    entities
}

/// Document-level span grouping for multi-chunk (e.g. sliding-window) token
/// classification: the cross-chunk companion to [`group_spans`].
///
/// Overlapping windows predict the same source tokens more than once, and an
/// entity straddling a window boundary is split across chunks. This collapses
/// the overlap: every per-chunk [`TokenLabel`] is flattened into one stream,
/// deduplicated by byte span (the highest-confidence prediction wins a
/// disagreement), re-indexed contiguously, and re-grouped under `scheme` — so
/// the split halves of a boundary entity merge into one [`Entity`].
///
/// `per_chunk` is one `Vec<TokenLabel>` per chunk (each produced by
/// [`labels_for_tokens`]); the returned [`Entity`] spans carry source-absolute
/// byte ranges and conservative min-score aggregation, exactly like
/// [`group_spans`]. For non-overlapping chunkers (`TruncatingChunker`, or
/// `SlidingWindowChunker` with `stride == window`) this is equivalent to
/// concatenating the chunks; it matters only when windows overlap.
pub fn group_spans_document<'a>(per_chunk: &[Vec<TokenLabel<'a>>], scheme: Scheme) -> Vec<Entity> {
    // Flatten, then sort by (start, end, score desc) so dedup_by keeps the
    // highest-confidence prediction for each byte span.
    let mut all: Vec<TokenLabel<'a>> = per_chunk.iter().flatten().copied().collect();
    all.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(a.end.cmp(&b.end))
            .then(b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal))
    });
    all.dedup_by(|a, b| a.start == b.start && a.end == b.end);

    // Re-index contiguously: group_spans reads token_index for token_range.
    for (i, t) in all.iter_mut().enumerate() {
        t.token_index = i;
    }

    group_spans(&all, scheme)
}

/// Tag prefix decoded by [`parse_tag`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Prefix {
    O,
    B,
    I,
    L,
    U,
    E,
    S,
}

/// Split a label string into `(prefix, type)`. Recognizes the prefixes valid for
/// `scheme`; an unrecognized prefix (or a scheme-mismatched one) falls back to
/// `Begin` with the whole label as the type, so flat labels (`"NOUN"`, `"O"`-less)
/// degrade to one span per token rather than panicking.
fn parse_tag(label: &str, scheme: Scheme) -> (Prefix, Option<String>) {
    let (p, t) = match label.split_once('-') {
        Some((p, t)) => (p, Some(t.to_string())),
        None => (label, None),
    };
    let prefix = match p {
        "O" => Prefix::O,
        "B" => Prefix::B,
        "I" => Prefix::I,
        "L" if matches!(scheme, Scheme::Bilou) => Prefix::L,
        "U" if matches!(scheme, Scheme::Bilou) => Prefix::U,
        "E" if matches!(scheme, Scheme::Iobes) => Prefix::E,
        "S" if matches!(scheme, Scheme::Iobes) => Prefix::S,
        _ => return (Prefix::B, Some(label.to_string())),
    };
    (prefix, t)
}

/// Flush an open span (if any) into `out` spanning `[start_token .. end_token)`.
fn close_span(open: &mut Option<(String, u32, usize, usize, usize, f32)>, out: &mut Vec<Entity>, end_token: usize) {
    if let Some((typ, lid, start, end, start_token, score)) = open.take() {
        out.push(Entity { label: typ, label_id: lid, start, end, token_range: start_token..end_token, score });
    }
}

/// Index of the maximum element, or `0` for an empty slice. NaN-safe: NaN is
/// treated as equal to everything (never wins a strict greater-than), so a NaN
/// row yields `0` rather than panicking — unlike `partial_cmp(b).unwrap()`.
pub fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Softmax probabilities (max-subtracted for numerical stability), summing to
/// ~1. An empty slice yields an empty vec. Used to turn raw class logits into
/// probabilities alongside [`argmax`].
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}
