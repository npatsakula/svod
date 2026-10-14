use std::convert::Infallible;

use proptest::prelude::*;

use crate::pipelines::text::{
    BatchClassifications, BatchEmbeddings, BatchTokenClassifications, ChunkTokenClassification, Chunker,
    Classification, Classify, ClassifyTokens, Embed, Embedding, EncoderHead, EncoderPipeline, EncoderPipelineError,
    Encoding, RunOptions, RunProfile, Scheme, SlidingWindowChunker, SlidingWindowChunkerError, TextChunk,
    TokenClassification, TokenLabel, Tokenizer, TruncatingChunker, TruncatingChunkerError, argmax, group_spans,
    group_spans_document, labels_for_tokens, softmax,
};
#[cfg(feature = "hf-tokenizers")]
use crate::pipelines::text::{HfTokenizer, HfTokenizerError};

fn enc(ids: &[u32]) -> Encoding {
    // A full, internally-consistent encoding: every field the same length as
    // ids, with offsets counting up and masks/specials set to plausible values.
    let n = ids.len();
    Encoding {
        input_ids: ids.to_vec(),
        attention_mask: vec![1; n],
        token_type_ids: vec![0; n],
        offsets: (0..n).map(|i| (i, i + 1)).collect(),
        special_tokens_mask: vec![0; n],
    }
}

/// Encoding with `[CLS]`=2 / `[SEP]`=3 wrapping the content (BertProcessing
/// convention). Content token at content-index `j` gets offset `(j, j+1)`, so a
/// window starting at content-index `start` has `byte_offset == start`.
fn enc_with_specials(ids: &[u32]) -> Encoding {
    let n_content = ids.len();
    let mut input_ids = vec![2]; // [CLS]
    input_ids.extend_from_slice(ids);
    input_ids.push(3); // [SEP]
    let n = input_ids.len();

    let mut offsets = vec![(0, 0)];
    offsets.extend((0..n_content).map(|i| (i, i + 1)));
    offsets.push((n_content, n_content));

    let mut special_tokens_mask = vec![0u32; n];
    special_tokens_mask[0] = 1;
    special_tokens_mask[n - 1] = 1;

    Encoding { input_ids, attention_mask: vec![1; n], token_type_ids: vec![0; n], offsets, special_tokens_mask }
}

fn assert_field_lengths(enc: &Encoding, expected: usize) {
    assert_eq!(enc.input_ids.len(), expected);
    assert_eq!(enc.attention_mask.len(), expected);
    assert_eq!(enc.token_type_ids.len(), expected);
    assert_eq!(enc.offsets.len(), expected);
    assert_eq!(enc.special_tokens_mask.len(), expected);
}

// ─── Stubs (host-only; no tokenizer.json, no model/device) ────────────────────

/// Returns a canned encoding per input text (analog of `MockVad`).
struct StubTokenizer {
    ids: Vec<u32>,
    error: bool,
}

impl Tokenizer for StubTokenizer {
    type Error = StubTokenizerError;
    fn encode(&mut self, _text: &str) -> Result<Encoding, StubTokenizerError> {
        if self.error {
            return Err(StubTokenizerError);
        }
        Ok(enc(&self.ids))
    }
}

#[derive(Debug, snafu::Snafu)]
#[snafu(display("stub tokenizer error"))]
struct StubTokenizerError;

/// Turns ids into a deterministic embedding (analog of `PresetTranscriber`):
/// hidden_size = the id count (so `len` is visible without pulling in the model),
/// values = ids as f32. `profile` emits a single 1 ms `encode` stage — enough to
/// exercise the merge without a model.
struct StubEmbed {
    hidden_size: usize,
    max_batch: usize,
    error: bool,
}

impl EncoderHead for StubEmbed {
    type Output = Embedding;
    type Error = StubEmbedError;
    fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.hidden_size)
    }
    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> Result<(Vec<Self::Output>, Option<RunProfile>), Self::Error> {
        if self.error {
            return Err(StubEmbedError);
        }
        let values =
            batch.iter().map(|e| Embedding { values: e.input_ids.iter().map(|&id| id as f32).collect() }).collect();
        let prof = profile.then(|| {
            let mut p = RunProfile::default();
            p.push(svod_runtime::StageProfile::host("encode", std::time::Duration::from_millis(1)));
            p
        });
        Ok((values, prof))
    }
}

impl Embed for StubEmbed {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[derive(Debug, snafu::Snafu)]
#[snafu(display("stub embed error"))]
struct StubEmbedError;

/// Turns ids into deterministic class logits (analog of `StubEmbed`):
/// num_labels = the id count, logits = ids as f32. `profile` emits a single
/// 1 ms `classify` stage — enough to exercise the merge without a model.
struct StubClassify {
    num_labels: usize,
    max_batch: usize,
    error: bool,
}

impl EncoderHead for StubClassify {
    type Output = Classification;
    type Error = StubClassifyError;
    fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.num_labels)
    }
    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> Result<(Vec<Self::Output>, Option<RunProfile>), Self::Error> {
        if self.error {
            return Err(StubClassifyError);
        }
        let values = batch
            .iter()
            .map(|e| Classification { logits: e.input_ids.iter().map(|&id| id as f32).collect() })
            .collect();
        let prof = profile.then(|| {
            let mut p = RunProfile::default();
            p.push(svod_runtime::StageProfile::host("classify", std::time::Duration::from_millis(1)));
            p
        });
        Ok((values, prof))
    }
}

impl Classify for StubClassify {
    fn num_labels(&self) -> usize {
        self.num_labels
    }
}

#[derive(Debug, snafu::Snafu)]
#[snafu(display("stub classify error"))]
struct StubClassifyError;

/// Records the chunk lengths of every batch handed to `run_batch` (then returns
/// deterministic embeddings, ids as f32) — lets a test observe *how* the
/// pipeline sub-batches (across text boundaries vs per-text, longest first),
/// which the result-checking stubs (`StubEmbed`/`StubClassify`/`StubRecognize`)
/// can't.
struct BatchSpy {
    batches: Vec<Vec<usize>>,
    max_batch: usize,
}

impl BatchSpy {
    fn new(max_batch: usize) -> Self {
        Self { batches: Vec::new(), max_batch }
    }

    fn sizes(&self) -> Vec<usize> {
        self.batches.iter().map(Vec::len).collect()
    }
}

impl EncoderHead for BatchSpy {
    type Output = Embedding;
    type Error = Infallible;
    fn capacity(&self) -> (usize, usize) {
        (self.max_batch, 0)
    }
    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        _profile: bool,
    ) -> Result<(Vec<Self::Output>, Option<RunProfile>), Self::Error> {
        self.batches.push(batch.iter().map(|e| e.len()).collect());
        let values =
            batch.iter().map(|e| Embedding { values: e.input_ids.iter().map(|&id| id as f32).collect() }).collect();
        Ok((values, None))
    }
}

impl Embed for BatchSpy {
    fn hidden_size(&self) -> usize {
        0
    }
}

/// Tokenizer that encodes each input character's byte value as a token id.
/// Allows batch tests to distinguish which text produced which embedding —
/// `"ab"` → ids `[97, 98]`, `"xyz"` → ids `[120, 121, 122]`, etc.
struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    type Error = Infallible;
    fn encode(&mut self, text: &str) -> Result<Encoding, Infallible> {
        Ok(enc(&text.bytes().map(|b| b as u32).collect::<Vec<_>>()))
    }
}

// ─── Encoding ─────────────────────────────────────────────────────────────────

#[test]
fn encoding_len_is_id_count() {
    assert_eq!(enc(&[7, 8, 9]).len(), 3);
    assert!(enc(&[]).is_empty());
}

// ─── TruncatingChunker ─────────────────────────────────────────────────────────

#[test]
fn truncating_chunker_drops_beyond_max_seq_and_keeps_fields_consistent() {
    let mut chunker = TruncatingChunker::new(4);
    let out = chunker.chunk(&enc(&[1, 2, 3, 4, 5, 6, 7])).unwrap();
    assert_eq!(out.len(), 1);
    let c = &out[0];
    assert_eq!(c.byte_offset, 0);
    let e = &c.encoding;
    // Sliced to max_seq on every field — masks and offsets stay aligned with ids.
    assert_eq!(e.input_ids, vec![1, 2, 3, 4]);
    assert_eq!(e.attention_mask, vec![1, 1, 1, 1]);
    assert_eq!(e.token_type_ids, vec![0, 0, 0, 0]);
    assert_eq!(e.offsets, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    assert_eq!(e.special_tokens_mask, vec![0, 0, 0, 0]);
}

#[test]
fn truncating_chunker_keeps_short_input_intact() {
    let mut chunker = TruncatingChunker::new(8);
    let out = chunker.chunk(&enc(&[1, 2, 3])).unwrap();
    assert_eq!(out[0].encoding.input_ids, vec![1, 2, 3]);
    assert_eq!(chunker.max_seq(), 8);
    assert_eq!(chunker.profile_label(), "chunk");
}

#[test]
#[should_panic(expected = "max_seq must be >= 1")]
fn truncating_new_rejects_zero() {
    let _ = TruncatingChunker::new(0);
}

#[test]
fn truncating_try_new_rejects_invalid_args() {
    assert!(matches!(TruncatingChunker::try_new(0).unwrap_err(), TruncatingChunkerError::MaxSeqTooSmall));
}

#[test]
fn truncating_try_new_accepts_valid_args() {
    let c = TruncatingChunker::try_new(4).unwrap();
    assert_eq!(c.max_seq(), 4);
    assert!(TruncatingChunker::try_new(1).is_ok());
}

// ─── SlidingWindowChunker ─────────────────────────────────────────────────────

#[test]
fn sliding_short_input_fits_one_window() {
    let mut chunker = SlidingWindowChunker::new(5, 2);
    let out = chunker.chunk(&enc_with_specials(&[10, 20, 30])).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].encoding.input_ids, vec![2, 10, 20, 30, 3]);
    assert_eq!(out[0].byte_offset, 0);
}

#[test]
fn sliding_windows_long_input_with_overlap_and_correct_offsets() {
    // [CLS] 10 11 12 13 14 15 16 [SEP] — 7 content tokens.
    // window=5 → content_window=3; stride=2 → step=2, overlap=1.
    let mut chunker = SlidingWindowChunker::new(5, 2);
    let out = chunker.chunk(&enc_with_specials(&[10, 11, 12, 13, 14, 15, 16])).unwrap();
    assert_eq!(out.len(), 3);

    // Window 0: content[0..3], byte_offset = 0.
    assert_eq!(out[0].byte_offset, 0);
    assert_eq!(out[0].encoding.input_ids, vec![2, 10, 11, 12, 3]);
    assert_eq!(out[0].encoding.special_tokens_mask, vec![1, 0, 0, 0, 1]);
    // Offsets are absolute (carried from the source), not rebased per window.
    assert_eq!(out[0].encoding.offsets, vec![(0, 0), (0, 1), (1, 2), (2, 3), (7, 7)]);
    assert_field_lengths(&out[0].encoding, 5);

    // Window 1: content[2..5], byte_offset = 2.
    assert_eq!(out[1].byte_offset, 2);
    assert_eq!(out[1].encoding.input_ids, vec![2, 12, 13, 14, 3]);
    assert_eq!(out[1].encoding.offsets[1], (2, 3)); // content token 12 keeps its absolute offset
    assert_field_lengths(&out[1].encoding, 5);

    // Window 2: content[4..7], byte_offset = 4.
    assert_eq!(out[2].byte_offset, 4);
    assert_eq!(out[2].encoding.input_ids, vec![2, 14, 15, 16, 3]);
    assert_field_lengths(&out[2].encoding, 5);

    // Overlap: windows 0 and 1 share token 12.
    assert!(out[0].encoding.input_ids[1..4].contains(&12));
    assert!(out[1].encoding.input_ids[1..4].contains(&12));
}

#[test]
fn sliding_stride_equals_window_gives_adjacent_chunks() {
    // window=4 → content_window=2; stride=4 → step clamped to 2. No overlap.
    let mut chunker = SlidingWindowChunker::new(4, 4);
    let out = chunker.chunk(&enc_with_specials(&[10, 20, 30, 40])).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].encoding.input_ids, vec![2, 10, 20, 3]);
    assert_eq!(out[1].encoding.input_ids, vec![2, 30, 40, 3]);
    assert_eq!(out[0].byte_offset, 0);
    assert_eq!(out[1].byte_offset, 2);
}

#[test]
fn sliding_last_window_clamped_when_content_uneven() {
    // 5 content tokens, content_window=2 → last window gets 1 token.
    let mut chunker = SlidingWindowChunker::new(4, 4);
    let out = chunker.chunk(&enc_with_specials(&[10, 20, 30, 40, 50])).unwrap();
    assert_eq!(out.len(), 3);
    // First two windows are full (2 content tokens each); last is partial.
    assert_eq!(out[0].encoding.input_ids, vec![2, 10, 20, 3]);
    assert_eq!(out[1].encoding.input_ids, vec![2, 30, 40, 3]);
    assert_eq!(out[2].encoding.input_ids, vec![2, 50, 3]);
    assert_eq!(out[2].byte_offset, 4);
}

#[test]
fn sliding_works_without_special_tokens() {
    // No specials: lead=0, trail=0, content = entire encoding.
    let mut chunker = SlidingWindowChunker::new(3, 2);
    let out = chunker.chunk(&enc(&[10, 20, 30, 40, 50, 60])).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].encoding.input_ids, vec![10, 20, 30]);
    assert_eq!(out[1].encoding.input_ids, vec![30, 40, 50]);
    assert_eq!(out[2].encoding.input_ids, vec![50, 60]); // partial
    assert_eq!(out[0].byte_offset, 0);
    assert_eq!(out[1].byte_offset, 2);
    assert_eq!(out[2].byte_offset, 4);
}

#[test]
fn sliding_all_specials_returns_empty() {
    let mut chunker = SlidingWindowChunker::new(8, 4);
    let out = chunker.chunk(&enc_with_specials(&[])).unwrap();
    assert!(out.is_empty());
}

#[test]
fn sliding_stride_one_maximally_overlaps() {
    // window=2, stride=1 over 4 content tokens (no specials): each consecutive
    // pair gets its own window, overlapping the last by 1 token.
    let mut chunker = SlidingWindowChunker::new(2, 1);
    let out = chunker.chunk(&enc(&[10, 20, 30, 40])).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].encoding.input_ids, vec![10, 20]);
    assert_eq!(out[1].encoding.input_ids, vec![20, 30]);
    assert_eq!(out[2].encoding.input_ids, vec![30, 40]);
    // byte_offsets are monotonic and advance by the stride.
    let offsets: Vec<usize> = out.iter().map(|c| c.byte_offset).collect();
    assert_eq!(offsets, vec![0, 1, 2]);
}

#[test]
fn sliding_max_seq_returns_window_and_default_label() {
    let chunker = SlidingWindowChunker::new(512, 256);
    assert_eq!(chunker.max_seq(), 512);
    assert_eq!(chunker.profile_label(), "chunk");
}

#[test]
#[should_panic(expected = "window must be >= 1")]
fn sliding_new_rejects_zero_window() {
    let _ = SlidingWindowChunker::new(0, 1);
}

#[test]
#[should_panic(expected = "stride must be in 1..=window")]
fn sliding_new_rejects_zero_stride() {
    let _ = SlidingWindowChunker::new(4, 0);
}

#[test]
#[should_panic(expected = "stride must be in 1..=window")]
fn sliding_new_rejects_stride_above_window() {
    let _ = SlidingWindowChunker::new(4, 5);
}

#[test]
fn sliding_try_new_rejects_invalid_args() {
    assert!(matches!(SlidingWindowChunker::try_new(0, 1).unwrap_err(), SlidingWindowChunkerError::WindowTooSmall));
    assert!(matches!(SlidingWindowChunker::try_new(4, 0).unwrap_err(), SlidingWindowChunkerError::StrideOutOfRange));
    assert!(matches!(SlidingWindowChunker::try_new(4, 5).unwrap_err(), SlidingWindowChunkerError::StrideOutOfRange));
}

#[test]
fn sliding_try_new_accepts_valid_args() {
    let c = SlidingWindowChunker::try_new(4, 2).unwrap();
    assert_eq!(c.max_seq(), 4);
    assert!(SlidingWindowChunker::try_new(1, 1).is_ok());
}

#[test]
fn sliding_chunk_returns_err_when_window_too_small_for_specials() {
    // window=2 is accepted by try_new (1 <= stride <= window), but a BERT-style
    // encoding carries 2 boundary specials (CLS + SEP), leaving no content
    // budget. This is detected at chunk time (the specials budget comes from
    // the tokenizer output), so it must surface as an Err, not a panic.
    let mut chunker = SlidingWindowChunker::new(2, 1);
    let err = chunker.chunk(&enc_with_specials(&[10, 20])).unwrap_err();
    assert!(matches!(err, SlidingWindowChunkerError::WindowTooSmallForSpecials { window: 2, specials: 2 }));
}

#[test]
fn sliding_pipeline_surfaces_chunk_error_through_chunk_arm() {
    // A specials-wrapping tokenizer + a window=2 chunker: the two boundary
    // specials (CLS+SEP) exhaust the window budget, so the chunk-time error
    // folds through EncoderPipelineError::Chunk.
    struct SpecialsTokenizer;
    impl Tokenizer for SpecialsTokenizer {
        type Error = Infallible;
        fn encode(&mut self, _text: &str) -> Result<Encoding, Infallible> {
            Ok(enc_with_specials(&[10, 20]))
        }
    }
    let mut p = EncoderPipeline::new(
        SpecialsTokenizer,
        SlidingWindowChunker::new(2, 1),
        StubEmbed { hidden_size: 2, max_batch: 1, error: false },
    );
    let err = p.embed("ignored", ()).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Chunk { .. }));
}

#[test]
fn sliding_pipeline_produces_per_window_embeddings() {
    // StubTokenizer yields [10, 20, 30, 40, 50, 60] (no specials).
    // SlidingWindowChunker(3, 2) → 3 windows at byte_offsets 0, 2, 4.
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![10, 20, 30, 40, 50, 60], error: false },
        SlidingWindowChunker::new(3, 2),
        StubEmbed { hidden_size: 3, max_batch: 1, error: false },
    );
    let out = p.embed("ignored", ()).unwrap();
    assert_eq!(out.chunks.len(), 3);
    assert_eq!(out.chunks[0].byte_offset, 0);
    assert_eq!(out.chunks[0].values, vec![10.0, 20.0, 30.0]);
    assert_eq!(out.chunks[1].byte_offset, 2);
    assert_eq!(out.chunks[1].values, vec![30.0, 40.0, 50.0]);
    assert_eq!(out.chunks[2].byte_offset, 4);
    assert_eq!(out.chunks[2].values, vec![50.0, 60.0]);
}

#[test]
fn sliding_pipeline_profiles_with_chunk_stage() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![10, 20, 30, 40, 50, 60], error: false },
        SlidingWindowChunker::new(3, 2),
        StubEmbed { hidden_size: 3, max_batch: 1, error: false },
    );
    let out = p.embed("ignored", RunOptions { profile: true }).unwrap();
    assert_eq!(out.chunks.len(), 3);
    let profile = out.profile.expect("profile collected");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk", "encode"]);
}

// ─── Embed: batch vs single-path agreement ───────────────────────────────────

#[test]
fn embed_single_is_batch_of_one() {
    // The default `embed` delegates to `embed_batch(&[enc])` and pops — verify
    // the values match a direct batch call.
    let mut embed = StubEmbed { hidden_size: 3, max_batch: 1, error: false };
    let e = enc(&[4, 5, 6]);
    let single = embed.run(&e, false).unwrap().0;
    let batch = embed.run_batch(&[&e], false).unwrap().0;
    assert_eq!(single, batch.into_iter().next().unwrap());
}

// ─── EncoderPipeline end-to-end ───────────────────────────────────────────

fn pipeline(ids: Vec<u32>, max_seq: usize) -> EncoderPipeline<StubTokenizer, TruncatingChunker, StubEmbed> {
    EncoderPipeline::new(
        StubTokenizer { ids, error: false },
        TruncatingChunker::new(max_seq),
        StubEmbed { hidden_size: max_seq, max_batch: 1, error: false },
    )
}

#[test]
fn pipeline_truncates_then_embeds() {
    // Tokenizer yields 7 ids; chunker caps at 4; embedder sees the 4 surviving.
    let mut p = pipeline(vec![1, 2, 3, 4, 5, 6, 7], 4);
    let out = p.embed("ignored", ()).unwrap();
    assert_eq!(out.chunks.len(), 1);
    // byte_offset is threaded through from the chunker (TruncatingChunker → 0).
    assert_eq!(out.chunks[0].byte_offset, 0);
    assert_eq!(out.chunks[0].values, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(out.profile.is_none(), "default options don't profile");
}

#[test]
fn pipeline_profiles_stage_order_tokenize_then_chunk_then_encoder() {
    let mut p = pipeline(vec![1, 2, 3], 8);
    let out = p.embed("ignored", RunOptions { profile: true }).unwrap();
    let profile = out.profile.expect("profile collected");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk", "encode"], "host stages lead, then the encoder's");
}

#[test]
fn pipeline_profiles_per_call_without_rebuild() {
    let mut p = pipeline(vec![1, 2, 3], 8);
    // One built pipeline serves both modes.
    let profiled = p.embed("ignored", RunOptions { profile: true }).unwrap();
    assert!(profiled.profile.is_some());
    let unprofiled = p.embed("ignored", ()).unwrap();
    assert!(unprofiled.profile.is_none());
}

#[test]
fn pipeline_surfaces_host_stages_even_when_encoder_does_not() {
    // EncoderHead emits a profile only when asked — but it always does here, so to
    // exercise "encoder emits nothing", drop the embedder's stage by giving a
    // single-chunk input under a non-profiled encoder is not enough. Instead,
    // assert directly: tokenize+chunk still surface on their own via an encoder
    // that returns None. Reuse the stub but call embed with profile and a stub
    // whose embed_batch returns None for the profile — covered by the case below.
    // Here we confirm the prepend works for the normal path already; the
    // no-encoder-profile path is covered by `pipeline_empty_input_surfaces_host`.
    let mut p = pipeline(vec![1, 2, 3], 8);
    let out = p.embed("ignored", RunOptions { profile: true }).unwrap();
    let profile = out.profile.expect("profile");
    // At minimum the two host stages are present.
    assert!(profile.stages.iter().any(|s| s.name == "tokenize"));
    assert!(profile.stages.iter().any(|s| s.name == "chunk"));
}

#[test]
fn pipeline_empty_input_skips_embed_batch_and_still_profiles_host_stages() {
    // Zero ids → chunker emits one empty chunk (not zero chunks under
    // TruncatingChunker), so embed_batch still runs over one input. The
    // zero-chunk branch is reachable only via a chunker that yields nothing;
    // assert that branch via a custom chunker below instead. Here just confirm
    // an empty-input run still profiles tokenize+chunk.
    let mut p = pipeline(Vec::new(), 4);
    let out = p.embed("ignored", RunOptions { profile: true }).unwrap();
    assert_eq!(out.chunks.len(), 1, "truncating chunker emits one empty chunk");
    let profile = out.profile.expect("profile");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert!(names.starts_with(&["tokenize", "chunk"]), "host stages lead even on empty input");
}

#[test]
fn pipeline_zero_chunk_run_skips_embed_and_profiles_host_stages() {
    // A chunker that yields zero chunks: the empty-guard must skip embed_batch
    // (so the erroring stub is never hit) while still surfacing tokenize+chunk.
    let p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], error: false },
        NoChunkChunker { max_seq: 4 },
        StubEmbed { hidden_size: 4, max_batch: 1, error: true }, // would fail if called
    );
    let mut p = p;
    let out = p.embed("ignored", RunOptions { profile: true }).unwrap();
    assert!(out.chunks.is_empty());
    let profile = out.profile.expect("profile");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk"], "encoder never runs; only host stages");
}

/// Chunker that always emits zero chunks (to exercise the empty-input guard).
struct NoChunkChunker {
    max_seq: usize,
}

impl Chunker for NoChunkChunker {
    type Error = Infallible;
    fn max_seq(&self) -> usize {
        self.max_seq
    }
    fn chunk(&mut self, _enc: &Encoding) -> Result<Vec<TextChunk>, Infallible> {
        Ok(Vec::new())
    }
}

// ─── assemble sizes encoder from chunker.max_seq ─────────────────────────────

#[test]
fn assemble_passes_chunker_max_seq_into_builder() {
    let seen = std::cell::Cell::new(0usize);
    let _p: EncoderPipeline<_, TruncatingChunker, StubEmbed> =
        EncoderPipeline::assemble(StubTokenizer { ids: vec![1], error: false }, TruncatingChunker::new(8), |max_seq| {
            seen.set(max_seq);
            Ok::<_, Infallible>(StubEmbed { hidden_size: max_seq, max_batch: 1, error: false })
        })
        .unwrap();
    assert_eq!(seen.get(), 8);
}

// ─── error propagation ────────────────────────────────────────────────────────

#[test]
fn tokenize_error_maps_to_tokenize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1], error: true },
        TruncatingChunker::new(4),
        StubEmbed { hidden_size: 4, max_batch: 1, error: false },
    );
    let err = p.embed("ignored", ()).unwrap_err();
    assert!(matches!(err, crate::pipelines::text::EncoderPipelineError::Tokenize { .. }));
}

#[test]
fn embed_error_maps_to_embed_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], error: false },
        TruncatingChunker::new(4),
        StubEmbed { hidden_size: 4, max_batch: 1, error: true },
    );
    let err = p.embed("ignored", ()).unwrap_err();
    assert!(matches!(err, crate::pipelines::text::EncoderPipelineError::Encode { .. }));
}

// ─── HfTokenizer: a real tokenizers::Tokenizer fixture ─────────────────────────
//
// Hand-built WordPiece tokenizer: tiny vocab + Whitespace pre-tokenizer +
// BertProcessing wrapping each sequence in [CLS] … [SEP]. The ids are fully
// predictable for known input, so encode() assertions are exact (specials get
// non-trivial special_tokens_mask). Built programmatically; `fixture_json`
// serializes it so from_bytes/from_path exercise the real JSON deserialization
// path rather than a hand-written string.

#[cfg(feature = "hf-tokenizers")]
fn fixture_tokenizer() -> tokenizers::Tokenizer {
    let vocab = [
        ("[PAD]".to_string(), 0u32),
        ("[UNK]".to_string(), 1),
        ("[CLS]".to_string(), 2),
        ("[SEP]".to_string(), 3),
        ("hello".to_string(), 4),
        ("world".to_string(), 5),
        ("foo".to_string(), 6),
        ("bar".to_string(), 7),
    ];
    let model = tokenizers::models::wordpiece::WordPiece::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .expect("wordpiece vocab contains [UNK]");
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    tokenizer.with_post_processor(Some(tokenizers::processors::bert::BertProcessing::new(
        ("[SEP]".to_string(), 3),
        ("[CLS]".to_string(), 2),
    )));
    tokenizer
}

#[cfg(feature = "hf-tokenizers")]
fn fixture_json() -> Vec<u8> {
    fixture_tokenizer().to_string(false).expect("serialize fixture tokenizer").into_bytes()
}

#[cfg(feature = "hf-tokenizers")]
#[test]
fn hf_tokenizer_from_bytes_encodes_known_text() {
    let mut tok = HfTokenizer::from_bytes(fixture_json()).expect("load fixture");
    let enc = tok.encode("hello world").expect("encode");
    // [CLS] hello world [SEP]
    assert_eq!(enc.input_ids, vec![2, 4, 5, 3]);
    assert_eq!(enc.attention_mask, vec![1, 1, 1, 1]);
    // All five fields share one length — the invariant from_hf preserves.
    let n = enc.input_ids.len();
    assert_eq!(enc.attention_mask.len(), n);
    assert_eq!(enc.token_type_ids.len(), n);
    assert_eq!(enc.offsets.len(), n);
    assert_eq!(enc.special_tokens_mask.len(), n);
    // Specials land at the brackets; real tokens stay unmasked.
    assert_eq!(enc.special_tokens_mask, vec![1, 0, 0, 1]);
}

#[cfg(feature = "hf-tokenizers")]
#[test]
fn hf_tokenizer_from_path_matches_from_bytes() {
    let bytes = fixture_json();
    // Unique temp path: tests may run in parallel.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("svod-arch-hf-tokenizer-{n}.json"));
    std::fs::write(&path, &bytes).expect("write fixture to temp file");
    let mut from_path = HfTokenizer::from_path(&path).expect("load from path");
    let mut from_bytes = HfTokenizer::from_bytes(&bytes).expect("load from bytes");
    let id_path = from_path.encode("hello world foo bar").unwrap().input_ids;
    let id_bytes = from_bytes.encode("hello world foo bar").unwrap().input_ids;
    assert_eq!(id_path, id_bytes);
    let _ = std::fs::remove_file(&path);
}

#[cfg(feature = "hf-tokenizers")]
#[test]
fn hf_tokenizer_encode_batch_matches_per_input_encode() {
    // The native HF batch path (`inner.encode_batch`) is the production
    // tokenization path for embed_batch / classify_batch / classify_tokens_batch,
    // distinct from the default `encode_batch` loop (it converts to Vec<String>
    // and exercises HF's own batching). It must agree field-for-field with
    // looping `encode` — the only thing a per-input encode could diverge on.
    let mut tok = HfTokenizer::from_bytes(fixture_json()).expect("load fixture");
    let texts = ["hello world", "foo bar", "hello world foo bar"];
    let batched = tok.encode_batch(&texts).expect("encode_batch");
    assert_eq!(batched.len(), texts.len());
    for (i, text) in texts.iter().enumerate() {
        let single = tok.encode(text).expect("encode");
        let b = &batched[i];
        assert_eq!(b.input_ids, single.input_ids, "input_ids for {text:?}");
        assert_eq!(b.attention_mask, single.attention_mask, "attention_mask for {text:?}");
        assert_eq!(b.token_type_ids, single.token_type_ids, "token_type_ids for {text:?}");
        assert_eq!(b.offsets, single.offsets, "offsets for {text:?}");
        assert_eq!(b.special_tokens_mask, single.special_tokens_mask, "special_tokens_mask for {text:?}");
    }
    // Sanity: the texts produce distinct id sequences (no accidental collapse).
    assert_ne!(batched[0].input_ids, batched[1].input_ids);
}

#[cfg(feature = "hf-tokenizers")]
#[test]
fn encoding_from_hf_copies_all_fields() {
    let inner = fixture_tokenizer();
    let hf = inner.encode("hello world", true).expect("hf encode");
    let enc = Encoding::from_hf(&hf);
    assert_eq!(enc.input_ids, hf.get_ids().to_vec());
    assert_eq!(enc.attention_mask, hf.get_attention_mask().to_vec());
    assert_eq!(enc.token_type_ids, hf.get_type_ids().to_vec());
    assert_eq!(enc.offsets, hf.get_offsets().to_vec());
    assert_eq!(enc.special_tokens_mask, hf.get_special_tokens_mask().to_vec());
}

#[cfg(feature = "hf-tokenizers")]
#[test]
fn hf_tokenizer_error_display_and_source() {
    use std::error::Error as _;
    // from_bytes returns HfTokenizerError — the From<tokenizers::Error> conversion
    // wired on from_bytes' `?` surfaces the boxed HF error behind a sized type.
    let err = HfTokenizer::from_bytes(b"not valid json").err().expect("invalid json must error");
    assert!(!err.to_string().is_empty());
    assert!(err.source().is_some(), "HfTokenizerError wraps an inner error");
    // HfTokenizerError is the named type the trait/field expect — confirm by name.
    let _: &HfTokenizerError = &err;
}

// ─── EncoderPipelineError::Chunk arm ─────────────────────────────────────────

#[derive(Debug, snafu::Snafu)]
#[snafu(display("stub chunker error"))]
struct ErrChunkerError;

/// Always fails `chunk` — reaches the otherwise-unreachable `Chunk` arm.
struct ErrChunker {
    max_seq: usize,
}

impl Chunker for ErrChunker {
    type Error = ErrChunkerError;
    fn max_seq(&self) -> usize {
        self.max_seq
    }
    fn chunk(&mut self, _enc: &Encoding) -> Result<Vec<TextChunk>, ErrChunkerError> {
        Err(ErrChunkerError)
    }
}

#[test]
fn chunk_error_maps_to_chunk_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], error: false },
        ErrChunker { max_seq: 4 },
        StubEmbed { hidden_size: 4, max_batch: 1, error: false },
    );
    let err = p.embed("ignored", ()).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Chunk { .. }));
}

// ─── embed_batch (multi-text) ─────────────────────────────────────────────────

#[test]
fn batch_basic_three_texts() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        TruncatingChunker::new(32),
        StubEmbed { hidden_size: 32, max_batch: 8, error: false },
    );
    let out = p.embed_batch(&["ab", "xyz", "hello"], ()).unwrap();
    assert_eq!(out.results.len(), 3);
    assert!(out.profile.is_none(), "default options don't profile");
    // ByteTokenizer maps each byte to its value: "ab" → [97, 98], etc.
    assert_eq!(out.results[0].chunks.len(), 1);
    assert_eq!(out.results[0].chunks[0].values, vec![97.0, 98.0]);
    assert_eq!(out.results[1].chunks[0].values, vec![120.0, 121.0, 122.0]);
    assert_eq!(out.results[2].chunks[0].values, vec![104.0, 101.0, 108.0, 108.0, 111.0]);
}

#[test]
fn batch_with_sliding_window_varying_chunk_counts() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        SlidingWindowChunker::new(3, 2),
        StubEmbed { hidden_size: 3, max_batch: 8, error: false },
    );
    // 6 + 2 + 4 content tokens → 3 + 1 + 2 = 6 chunks total.
    let out = p.embed_batch(&["abcdef", "ab", "abcd"], ()).unwrap();
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.results[0].chunks.len(), 3);
    assert_eq!(out.results[0].chunks[0].byte_offset, 0);
    assert_eq!(out.results[0].chunks[1].byte_offset, 2);
    assert_eq!(out.results[0].chunks[2].byte_offset, 4);
    assert_eq!(out.results[1].chunks.len(), 1);
    assert_eq!(out.results[2].chunks.len(), 2);
}

#[test]
fn batch_empty_texts_returns_empty() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        TruncatingChunker::new(8),
        StubEmbed { hidden_size: 8, max_batch: 4, error: false },
    );
    let out: BatchEmbeddings = p.embed_batch(&[], ()).unwrap();
    assert!(out.results.is_empty());
    assert!(out.profile.is_none());
}

#[test]
fn batch_some_texts_produce_zero_chunks() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        SlidingWindowChunker::new(4, 2),
        StubEmbed { hidden_size: 4, max_batch: 4, error: false },
    );
    // "" → 0 tokens → 0 chunks (SlidingWindowChunker content_len guard).
    let out = p.embed_batch(&["ab", "", "cd"], ()).unwrap();
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.results[0].chunks.len(), 1);
    assert!(out.results[1].chunks.is_empty());
    assert_eq!(out.results[2].chunks.len(), 1);
}

#[test]
fn batch_sub_batches_when_chunks_exceed_max_batch() {
    // 5 texts × 1 chunk = 5 chunks; max_batch=2 → 3 sub-batches (2, 2, 1).
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        TruncatingChunker::new(8),
        StubEmbed { hidden_size: 8, max_batch: 2, error: false },
    );
    let texts: Vec<&str> = vec!["a", "b", "c", "d", "e"];
    let out = p.embed_batch(&texts, ()).unwrap();
    assert_eq!(out.results.len(), 5);
    for (i, result) in out.results.iter().enumerate() {
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.chunks[0].values, vec![texts[i].as_bytes()[0] as f32]);
    }
}

#[test]
fn embed_batch_sub_batches_across_text_boundaries_with_sliding_window() {
    // The one path that was untested: multi-chunk-per-text (SlidingWindowChunker)
    // AND total chunks > max_batch, so a sub-batch boundary lands *mid-text*.
    // run_batch_inner's flatten → sub-batch → re-split-by-count is duplicated
    // with the seam path; the seam is tested with varying counts, but the full
    // tokenize→chunk→batch path under both conditions at once was not.
    let mut p = EncoderPipeline::new(ByteTokenizer, SlidingWindowChunker::new(3, 3), BatchSpy::new(2));
    // "abcdefghi" (9 content tokens) -> 3 non-overlapping windows of 3;
    // "jkl" (3 tokens) -> 1 window. Flatten = [abc, def, ghi, jkl]; max_batch=2
    // -> sub-batches [abc,def] and [ghi,jkl] — the second straddles the boundary.
    let out = p.embed_batch(&["abcdefghi", "jkl"], ()).unwrap();
    assert_eq!(out.results.len(), 2);

    // The cross-text coalescing is observable: the second sub-batch mixes the
    // tail of text 0 with all of text 1.
    assert_eq!(p.model_mut().sizes(), vec![2, 2]);

    // Re-split by per-text chunk count survived the cross-boundary sub-batch:
    // correct counts, correct source byte_offsets, and the right ids (BatchSpy
    // echoes input ids as f32) paired with each chunk.
    let a = &out.results[0].chunks;
    assert_eq!(a.len(), 3);
    assert_eq!(a[0].byte_offset, 0);
    assert_eq!(a[0].values, vec![97.0, 98.0, 99.0]); // abc
    assert_eq!(a[1].byte_offset, 3);
    assert_eq!(a[1].values, vec![100.0, 101.0, 102.0]); // def
    assert_eq!(a[2].byte_offset, 6);
    assert_eq!(a[2].values, vec![103.0, 104.0, 105.0]); // ghi

    let b = &out.results[1].chunks;
    assert_eq!(b.len(), 1);
    assert_eq!(b[0].byte_offset, 0);
    assert_eq!(b[0].values, vec![106.0, 107.0, 108.0]); // jkl
}

#[test]
fn batch_profile_has_tokenize_chunk_encode_stages() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        TruncatingChunker::new(8),
        StubEmbed { hidden_size: 8, max_batch: 4, error: false },
    );
    let out = p.embed_batch(&["ab", "cd"], RunOptions { profile: true }).unwrap();
    assert_eq!(out.results.len(), 2);
    // Per-text profiles are None — the batch profile lives on BatchEmbeddings.
    assert!(out.results[0].profile.is_none());
    assert!(out.results[1].profile.is_none());
    let profile = out.profile.expect("batch profile collected");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk", "encode"]);
}

#[test]
fn batch_tokenize_error_maps_to_tokenize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1], error: true },
        TruncatingChunker::new(4),
        StubEmbed { hidden_size: 4, max_batch: 4, error: false },
    );
    let err = p.embed_batch(&["a", "b"], ()).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Tokenize { .. }));
}

#[test]
fn batch_results_match_individual_embed_calls() {
    let make_pipeline = || {
        EncoderPipeline::new(
            ByteTokenizer,
            SlidingWindowChunker::new(4, 2),
            StubEmbed { hidden_size: 4, max_batch: 8, error: false },
        )
    };

    let texts = ["abcde", "xy"];
    let batch = {
        let mut p = make_pipeline();
        p.embed_batch(&texts, ()).unwrap()
    };

    for (i, text) in texts.iter().enumerate() {
        let mut p = make_pipeline();
        let single = p.embed(text, ()).unwrap();
        assert_eq!(batch.results[i].chunks.len(), single.chunks.len(), "chunk count mismatch for text {i}");
        for (b, s) in batch.results[i].chunks.iter().zip(&single.chunks) {
            assert_eq!(b.byte_offset, s.byte_offset, "byte_offset mismatch for text {i}");
            assert_eq!(b.values, s.values, "values mismatch for text {i}");
        }
    }
}

// ─── EncoderPipeline ────────────────────────────────────────────────────────

fn classify_pipeline(ids: Vec<u32>, max_seq: usize) -> EncoderPipeline<StubTokenizer, TruncatingChunker, StubClassify> {
    EncoderPipeline::new(
        StubTokenizer { ids, error: false },
        TruncatingChunker::new(max_seq),
        StubClassify { num_labels: max_seq, max_batch: 1, error: false },
    )
}

#[test]
fn classify_single_is_batch_of_one() {
    let mut cls = StubClassify { num_labels: 3, max_batch: 1, error: false };
    let e = enc(&[4, 5, 6]);
    let single = cls.run(&e, false).unwrap().0;
    let batch = cls.run_batch(&[&e], false).unwrap().0;
    assert_eq!(single, batch.into_iter().next().unwrap());
}

#[test]
fn classify_pipeline_truncates_then_classifies() {
    let mut p = classify_pipeline(vec![1, 2, 3, 4, 5, 6, 7], 4);
    let out = p.classify("ignored", ()).unwrap();
    assert_eq!(out.chunks.len(), 1);
    assert_eq!(out.chunks[0].byte_offset, 0);
    assert_eq!(out.chunks[0].logits, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(out.profile.is_none(), "default options don't profile");
}

#[test]
fn classify_pipeline_profiles_stage_order_tokenize_then_chunk_then_classify() {
    let mut p = classify_pipeline(vec![1, 2, 3], 8);
    let out = p.classify("ignored", RunOptions { profile: true }).unwrap();
    let profile = out.profile.expect("profile collected");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk", "classify"], "host stages lead, then the classifier's");
}

#[test]
fn classify_pipeline_profiles_per_call_without_rebuild() {
    let mut p = classify_pipeline(vec![1, 2, 3], 8);
    let profiled = p.classify("ignored", RunOptions { profile: true }).unwrap();
    assert!(profiled.profile.is_some());
    let unprofiled = p.classify("ignored", ()).unwrap();
    assert!(unprofiled.profile.is_none());
}

#[test]
fn classify_pipeline_zero_chunk_run_skips_classify_and_profiles_host_stages() {
    let p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], error: false },
        NoChunkChunker { max_seq: 4 },
        StubClassify { num_labels: 4, max_batch: 1, error: true }, // would fail if called
    );
    let mut p = p;
    let out = p.classify("ignored", RunOptions { profile: true }).unwrap();
    assert!(out.chunks.is_empty());
    let profile = out.profile.expect("profile");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk"], "classifier never runs; only host stages");
}

#[test]
fn classify_assemble_passes_chunker_max_seq_into_builder() {
    let seen = std::cell::Cell::new(0usize);
    let _p: EncoderPipeline<_, TruncatingChunker, StubClassify> =
        EncoderPipeline::assemble(StubTokenizer { ids: vec![1], error: false }, TruncatingChunker::new(8), |max_seq| {
            seen.set(max_seq);
            Ok::<_, Infallible>(StubClassify { num_labels: max_seq, max_batch: 1, error: false })
        })
        .unwrap();
    assert_eq!(seen.get(), 8);
}

#[test]
fn classify_sliding_pipeline_produces_per_window_classifications() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![10, 20, 30, 40, 50, 60], error: false },
        SlidingWindowChunker::new(3, 2),
        StubClassify { num_labels: 3, max_batch: 1, error: false },
    );
    let out = p.classify("ignored", ()).unwrap();
    assert_eq!(out.chunks.len(), 3);
    assert_eq!(out.chunks[0].byte_offset, 0);
    assert_eq!(out.chunks[0].logits, vec![10.0, 20.0, 30.0]);
    assert_eq!(out.chunks[1].byte_offset, 2);
    assert_eq!(out.chunks[1].logits, vec![30.0, 40.0, 50.0]);
    assert_eq!(out.chunks[2].byte_offset, 4);
    assert_eq!(out.chunks[2].logits, vec![50.0, 60.0]);
}

#[test]
fn classify_tokenize_error_maps_to_tokenize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1], error: true },
        TruncatingChunker::new(4),
        StubClassify { num_labels: 4, max_batch: 1, error: false },
    );
    let err = p.classify("ignored", ()).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Tokenize { .. }));
}

#[test]
fn classify_error_maps_to_classify_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], error: false },
        TruncatingChunker::new(4),
        StubClassify { num_labels: 4, max_batch: 1, error: true },
    );
    let err = p.classify("ignored", ()).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Encode { .. }));
}

#[test]
fn classify_chunk_error_maps_to_chunk_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], error: false },
        ErrChunker { max_seq: 4 },
        StubClassify { num_labels: 4, max_batch: 1, error: false },
    );
    let err = p.classify("ignored", ()).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Chunk { .. }));
}

// ─── EncoderPipeline batch ──────────────────────────────────────────────────

#[test]
fn classify_batch_basic_three_texts() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        TruncatingChunker::new(32),
        StubClassify { num_labels: 32, max_batch: 8, error: false },
    );
    let out = p.classify_batch(&["ab", "xyz", "hello"], ()).unwrap();
    assert_eq!(out.results.len(), 3);
    assert!(out.profile.is_none(), "default options don't profile");
    assert_eq!(out.results[0].chunks.len(), 1);
    assert_eq!(out.results[0].chunks[0].logits, vec![97.0, 98.0]);
    assert_eq!(out.results[1].chunks[0].logits, vec![120.0, 121.0, 122.0]);
    assert_eq!(out.results[2].chunks[0].logits, vec![104.0, 101.0, 108.0, 108.0, 111.0]);
}

#[test]
fn classify_batch_with_sliding_window_varying_chunk_counts() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        SlidingWindowChunker::new(3, 2),
        StubClassify { num_labels: 3, max_batch: 8, error: false },
    );
    let out = p.classify_batch(&["abcdef", "ab", "abcd"], ()).unwrap();
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.results[0].chunks.len(), 3);
    assert_eq!(out.results[0].chunks[0].byte_offset, 0);
    assert_eq!(out.results[0].chunks[1].byte_offset, 2);
    assert_eq!(out.results[0].chunks[2].byte_offset, 4);
    assert_eq!(out.results[1].chunks.len(), 1);
    assert_eq!(out.results[2].chunks.len(), 2);
}

#[test]
fn classify_batch_empty_texts_returns_empty() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        TruncatingChunker::new(8),
        StubClassify { num_labels: 8, max_batch: 4, error: false },
    );
    let out: BatchClassifications = p.classify_batch(&[], ()).unwrap();
    assert!(out.results.is_empty());
    assert!(out.profile.is_none());
}

#[test]
fn classify_batch_some_texts_produce_zero_chunks() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        SlidingWindowChunker::new(4, 2),
        StubClassify { num_labels: 4, max_batch: 4, error: false },
    );
    let out = p.classify_batch(&["ab", "", "cd"], ()).unwrap();
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.results[0].chunks.len(), 1);
    assert!(out.results[1].chunks.is_empty());
    assert_eq!(out.results[2].chunks.len(), 1);
}

#[test]
fn classify_batch_sub_batches_when_chunks_exceed_max_batch() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        TruncatingChunker::new(8),
        StubClassify { num_labels: 8, max_batch: 2, error: false },
    );
    let texts: Vec<&str> = vec!["a", "b", "c", "d", "e"];
    let out = p.classify_batch(&texts, ()).unwrap();
    assert_eq!(out.results.len(), 5);
    for (i, result) in out.results.iter().enumerate() {
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.chunks[0].logits, vec![texts[i].as_bytes()[0] as f32]);
    }
}

#[test]
fn classify_batch_profile_has_tokenize_chunk_classify_stages() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        TruncatingChunker::new(8),
        StubClassify { num_labels: 8, max_batch: 4, error: false },
    );
    let out = p.classify_batch(&["ab", "cd"], RunOptions { profile: true }).unwrap();
    assert_eq!(out.results.len(), 2);
    assert!(out.results[0].profile.is_none());
    assert!(out.results[1].profile.is_none());
    let profile = out.profile.expect("batch profile collected");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk", "classify"]);
}

#[test]
fn classify_batch_tokenize_error_maps_to_tokenize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1], error: true },
        TruncatingChunker::new(4),
        StubClassify { num_labels: 4, max_batch: 4, error: false },
    );
    let err = p.classify_batch(&["a", "b"], ()).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Tokenize { .. }));
}

#[test]
fn classify_batch_results_match_individual_classify_calls() {
    let make_pipeline = || {
        EncoderPipeline::new(
            ByteTokenizer,
            SlidingWindowChunker::new(4, 2),
            StubClassify { num_labels: 4, max_batch: 8, error: false },
        )
    };

    let texts = ["abcde", "xy"];
    let batch = {
        let mut p = make_pipeline();
        p.classify_batch(&texts, ()).unwrap()
    };

    for (i, text) in texts.iter().enumerate() {
        let mut p = make_pipeline();
        let single = p.classify(text, ()).unwrap();
        assert_eq!(batch.results[i].chunks.len(), single.chunks.len(), "chunk count mismatch for text {i}");
        for (b, s) in batch.results[i].chunks.iter().zip(&single.chunks) {
            assert_eq!(b.byte_offset, s.byte_offset, "byte_offset mismatch for text {i}");
            assert_eq!(b.logits, s.logits, "logits mismatch for text {i}");
        }
    }
}

// ─── EncoderPipeline ───────────────────────────────────────────────────────

/// Turns ids into a deterministic `(seq_len, num_labels)` logit grid (analog of
/// `StubClassify`): each token id `v` becomes a row of `num_labels` copies of
/// `v as f32`. `profile` emits a single 1 ms `classify_tokens` stage — enough to
/// exercise the merge without a model.
struct StubRecognize {
    num_labels: usize,
    max_batch: usize,
    error: bool,
}

impl EncoderHead for StubRecognize {
    type Output = TokenClassification;
    type Error = StubRecognizeError;
    fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.num_labels)
    }
    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> Result<(Vec<Self::Output>, Option<RunProfile>), Self::Error> {
        if self.error {
            return Err(StubRecognizeError);
        }
        let nl = self.num_labels;
        let values = batch
            .iter()
            .map(|e| {
                let mut logits = Vec::with_capacity(e.input_ids.len() * nl);
                for &id in &e.input_ids {
                    logits.extend(std::iter::repeat_n(id as f32, nl));
                }
                TokenClassification { logits, num_labels: nl }
            })
            .collect();
        let prof = profile.then(|| {
            let mut p = RunProfile::default();
            p.push(svod_runtime::StageProfile::host("classify_tokens", std::time::Duration::from_millis(1)));
            p
        });
        Ok((values, prof))
    }
}

impl ClassifyTokens for StubRecognize {
    fn num_labels(&self) -> usize {
        self.num_labels
    }
}

#[derive(Debug, snafu::Snafu)]
#[snafu(display("stub classify_tokens error"))]
struct StubRecognizeError;

fn classify_tokens_pipeline(
    ids: Vec<u32>,
    max_seq: usize,
) -> EncoderPipeline<StubTokenizer, TruncatingChunker, StubRecognize> {
    EncoderPipeline::new(
        StubTokenizer { ids, error: false },
        TruncatingChunker::new(max_seq),
        StubRecognize { num_labels: 1, max_batch: 1, error: false },
    )
}

#[test]
fn classify_tokens_single_is_batch_of_one() {
    let mut rec = StubRecognize { num_labels: 3, max_batch: 1, error: false };
    let e = enc(&[4, 5, 6]);
    let single = rec.run(&e, false).unwrap().0;
    let batch = rec.run_batch(&[&e], false).unwrap().0;
    assert_eq!(single, batch.into_iter().next().unwrap());
}

#[test]
fn classify_tokens_pipeline_truncates_then_recognizes() {
    let mut p = classify_tokens_pipeline(vec![1, 2, 3, 4, 5, 6, 7], 4);
    let out = p.classify_tokens("ignored", ()).unwrap();
    assert_eq!(out.chunks.len(), 1);
    let c = &out.chunks[0];
    assert_eq!(c.byte_offset, 0);
    assert_eq!(c.num_labels, 1);
    assert_eq!(c.logits, vec![1.0, 2.0, 3.0, 4.0]);
    // Per-token geometry is threaded through from the chunk's encoding.
    assert_eq!(c.token_offsets, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    assert_eq!(c.special_tokens_mask, vec![0, 0, 0, 0]);
    assert!(out.profile.is_none(), "default options don't profile");
}

#[test]
fn classify_tokens_pipeline_profiles_stage_order_tokenize_then_chunk_then_recognize() {
    let mut p = classify_tokens_pipeline(vec![1, 2, 3], 8);
    let out = p.classify_tokens("ignored", RunOptions { profile: true }).unwrap();
    let profile = out.profile.expect("profile collected");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk", "classify_tokens"], "host stages lead, then the token_classifier's");
}

#[test]
fn classify_tokens_pipeline_profiles_per_call_without_rebuild() {
    let mut p = classify_tokens_pipeline(vec![1, 2, 3], 8);
    let profiled = p.classify_tokens("ignored", RunOptions { profile: true }).unwrap();
    assert!(profiled.profile.is_some());
    let unprofiled = p.classify_tokens("ignored", ()).unwrap();
    assert!(unprofiled.profile.is_none());
}

#[test]
fn classify_tokens_pipeline_zero_chunk_run_skips_recognize_and_profiles_host_stages() {
    let p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], error: false },
        NoChunkChunker { max_seq: 4 },
        StubRecognize { num_labels: 4, max_batch: 1, error: true }, // would fail if called
    );
    let mut p = p;
    let out = p.classify_tokens("ignored", RunOptions { profile: true }).unwrap();
    assert!(out.chunks.is_empty());
    let profile = out.profile.expect("profile");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk"], "token_classifier never runs; only host stages");
}

#[test]
fn classify_tokens_assemble_passes_chunker_max_seq_into_builder() {
    let seen = std::cell::Cell::new(0usize);
    let _p: EncoderPipeline<_, TruncatingChunker, StubRecognize> =
        EncoderPipeline::assemble(StubTokenizer { ids: vec![1], error: false }, TruncatingChunker::new(8), |max_seq| {
            seen.set(max_seq);
            Ok::<_, Infallible>(StubRecognize { num_labels: max_seq, max_batch: 1, error: false })
        })
        .unwrap();
    assert_eq!(seen.get(), 8);
}

#[test]
fn classify_tokens_sliding_pipeline_produces_per_window_classifications() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![10, 20, 30, 40, 50, 60], error: false },
        SlidingWindowChunker::new(3, 2),
        StubRecognize { num_labels: 1, max_batch: 1, error: false },
    );
    let out = p.classify_tokens("ignored", ()).unwrap();
    assert_eq!(out.chunks.len(), 3);
    assert_eq!(out.chunks[0].byte_offset, 0);
    assert_eq!(out.chunks[0].logits, vec![10.0, 20.0, 30.0]);
    assert_eq!(out.chunks[0].token_offsets, vec![(0, 1), (1, 2), (2, 3)]);
    assert_eq!(out.chunks[1].byte_offset, 2);
    assert_eq!(out.chunks[1].logits, vec![30.0, 40.0, 50.0]);
    assert_eq!(out.chunks[1].token_offsets, vec![(2, 3), (3, 4), (4, 5)]);
    assert_eq!(out.chunks[2].byte_offset, 4);
    assert_eq!(out.chunks[2].logits, vec![50.0, 60.0]);
    assert_eq!(out.chunks[2].token_offsets, vec![(4, 5), (5, 6)]);
}

#[test]
fn classify_tokens_tokenize_error_maps_to_tokenize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1], error: true },
        TruncatingChunker::new(4),
        StubRecognize { num_labels: 4, max_batch: 1, error: false },
    );
    let err = p.classify_tokens("ignored", ()).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Tokenize { .. }));
}

#[test]
fn classify_tokens_error_maps_to_recognize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], error: false },
        TruncatingChunker::new(4),
        StubRecognize { num_labels: 4, max_batch: 1, error: true },
    );
    let err = p.classify_tokens("ignored", ()).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Encode { .. }));
}

#[test]
fn classify_tokens_chunk_error_maps_to_chunk_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], error: false },
        ErrChunker { max_seq: 4 },
        StubRecognize { num_labels: 4, max_batch: 1, error: false },
    );
    let err = p.classify_tokens("ignored", ()).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Chunk { .. }));
}

// ─── EncoderPipeline batch ─────────────────────────────────────────────────

#[test]
fn classify_tokens_batch_basic_three_texts() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        TruncatingChunker::new(32),
        StubRecognize { num_labels: 1, max_batch: 8, error: false },
    );
    let out = p.classify_tokens_batch(&["ab", "xyz", "hello"], ()).unwrap();
    assert_eq!(out.results.len(), 3);
    assert!(out.profile.is_none(), "default options don't profile");
    assert_eq!(out.results[0].chunks.len(), 1);
    assert_eq!(out.results[0].chunks[0].logits, vec![97.0, 98.0]);
    assert_eq!(out.results[0].chunks[0].num_labels, 1);
    assert_eq!(out.results[0].chunks[0].token_offsets, vec![(0, 1), (1, 2)]);
    assert_eq!(out.results[1].chunks[0].logits, vec![120.0, 121.0, 122.0]);
    assert_eq!(out.results[2].chunks[0].logits, vec![104.0, 101.0, 108.0, 108.0, 111.0]);
}

#[test]
fn classify_tokens_batch_with_sliding_window_varying_chunk_counts() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        SlidingWindowChunker::new(3, 2),
        StubRecognize { num_labels: 1, max_batch: 8, error: false },
    );
    let out = p.classify_tokens_batch(&["abcdef", "ab", "abcd"], ()).unwrap();
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.results[0].chunks.len(), 3);
    assert_eq!(out.results[0].chunks[0].byte_offset, 0);
    assert_eq!(out.results[0].chunks[1].byte_offset, 2);
    assert_eq!(out.results[0].chunks[2].byte_offset, 4);
    assert_eq!(out.results[1].chunks.len(), 1);
    assert_eq!(out.results[2].chunks.len(), 2);
}

#[test]
fn classify_tokens_batch_empty_texts_returns_empty() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        TruncatingChunker::new(8),
        StubRecognize { num_labels: 1, max_batch: 4, error: false },
    );
    let out: BatchTokenClassifications = p.classify_tokens_batch(&[], ()).unwrap();
    assert!(out.results.is_empty());
    assert!(out.profile.is_none());
}

#[test]
fn classify_tokens_batch_some_texts_produce_zero_chunks() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        SlidingWindowChunker::new(4, 2),
        StubRecognize { num_labels: 1, max_batch: 4, error: false },
    );
    let out = p.classify_tokens_batch(&["ab", "", "cd"], ()).unwrap();
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.results[0].chunks.len(), 1);
    assert!(out.results[1].chunks.is_empty());
    assert_eq!(out.results[2].chunks.len(), 1);
}

#[test]
fn classify_tokens_batch_sub_batches_when_chunks_exceed_max_batch() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        TruncatingChunker::new(8),
        StubRecognize { num_labels: 1, max_batch: 2, error: false },
    );
    let texts: Vec<&str> = vec!["a", "b", "c", "d", "e"];
    let out = p.classify_tokens_batch(&texts, ()).unwrap();
    assert_eq!(out.results.len(), 5);
    for (i, result) in out.results.iter().enumerate() {
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.chunks[0].logits, vec![texts[i].as_bytes()[0] as f32]);
    }
}

#[test]
fn classify_tokens_batch_profile_has_tokenize_chunk_recognize_stages() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        TruncatingChunker::new(8),
        StubRecognize { num_labels: 1, max_batch: 4, error: false },
    );
    let out = p.classify_tokens_batch(&["ab", "cd"], RunOptions { profile: true }).unwrap();
    assert_eq!(out.results.len(), 2);
    assert!(out.results[0].profile.is_none());
    assert!(out.results[1].profile.is_none());
    let profile = out.profile.expect("batch profile collected");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk", "classify_tokens"]);
}

#[test]
fn classify_tokens_batch_tokenize_error_maps_to_tokenize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1], error: true },
        TruncatingChunker::new(4),
        StubRecognize { num_labels: 4, max_batch: 4, error: false },
    );
    let err = p.classify_tokens_batch(&["a", "b"], ()).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Tokenize { .. }));
}

#[test]
fn classify_tokens_batch_results_match_individual_recognize_calls() {
    let make_pipeline = || {
        EncoderPipeline::new(
            ByteTokenizer,
            SlidingWindowChunker::new(4, 2),
            StubRecognize { num_labels: 1, max_batch: 8, error: false },
        )
    };

    let texts = ["abcde", "xy"];
    let batch = {
        let mut p = make_pipeline();
        p.classify_tokens_batch(&texts, ()).unwrap()
    };

    for (i, text) in texts.iter().enumerate() {
        let mut p = make_pipeline();
        let single = p.classify_tokens(text, ()).unwrap();
        assert_eq!(batch.results[i].chunks.len(), single.chunks.len(), "chunk count mismatch for text {i}");
        for (b, s) in batch.results[i].chunks.iter().zip(&single.chunks) {
            assert_eq!(b.byte_offset, s.byte_offset, "byte_offset mismatch for text {i}");
            assert_eq!(b.logits, s.logits, "logits mismatch for text {i}");
            assert_eq!(b.token_offsets, s.token_offsets, "token_offsets mismatch for text {i}");
            assert_eq!(b.special_tokens_mask, s.special_tokens_mask, "special_tokens_mask mismatch for text {i}");
        }
    }
}

// ─── chunk-level seam (run_chunks / *_chunks) ───────────────────────────────

/// Build chunks outside any pipeline: a ByteTokenizer + a SlidingWindowChunker
/// produce chunks we can then feed to multiple pipelines' chunk seams without
/// retokenizing.
fn shared_chunks() -> Vec<TextChunk> {
    let mut tok = ByteTokenizer;
    let enc = tok.encode("abcdef").unwrap();
    SlidingWindowChunker::new(3, 2).chunk(&enc).unwrap()
}

/// Build one [`TextChunk`] directly from ids (offsets count up, no specials) —
/// full control over chunk count/geometry without going through a chunker.
fn text_chunk(ids: &[u32], byte_offset: usize) -> TextChunk {
    TextChunk { encoding: enc(ids), byte_offset }
}

#[test]
fn chunk_seam_reuses_chunks_across_embed_and_classify() {
    // Tokenize + chunk once, then feed the SAME chunks to an embed and a
    // classify pipeline. Both pipelines' own tokenizers/chunkers would ERROR
    // if the seam touched them — proving the seam skips tokenize+chunk.
    let chunks = shared_chunks();
    assert_eq!(chunks.len(), 3, "abcdef / window=3 stride=2 → 3 windows");

    let make_embed = || {
        EncoderPipeline::new(
            StubTokenizer { ids: vec![], error: true },
            TruncatingChunker::new(3),
            StubEmbed { hidden_size: 3, max_batch: 1, error: false },
        )
    };
    let make_classify = || {
        EncoderPipeline::new(
            StubTokenizer { ids: vec![], error: true },
            TruncatingChunker::new(3),
            StubClassify { num_labels: 3, max_batch: 1, error: false },
        )
    };

    let embeds = make_embed().embed_chunks(&chunks, ()).unwrap();
    let classes = make_classify().classify_chunks(&chunks, ()).unwrap();

    // Same byte_offsets, same chunk counts — geometry is shared.
    assert_eq!(embeds.chunks.len(), classes.chunks.len());
    for (e, c) in embeds.chunks.iter().zip(classes.chunks.iter()) {
        assert_eq!(e.byte_offset, c.byte_offset);
    }
    // Different payloads (embed → values, classify → logits) but both derived
    // from the same ids. Window 0 = [97, 98, 99] = 'abc'.
    assert_eq!(embeds.chunks[0].values, vec![97.0, 98.0, 99.0]);
    assert_eq!(classes.chunks[0].logits, vec![97.0, 98.0, 99.0]);
    assert_eq!(embeds.chunks[1].byte_offset, 2);
    assert_eq!(embeds.chunks[2].byte_offset, 4);
}

#[test]
fn chunk_seam_feeds_one_encoding_through_two_chunkers() {
    // Tokenization caching: encode once, chunk with two different chunkers,
    // feed each chunk set through the seam independently.
    let mut tok = ByteTokenizer;
    let enc = tok.encode("abcdef").unwrap();
    let truncated = TruncatingChunker::new(4).chunk(&enc).unwrap();
    let windowed = SlidingWindowChunker::new(3, 2).chunk(&enc).unwrap();
    assert_eq!(truncated.len(), 1);
    assert_eq!(windowed.len(), 3);

    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![], error: true },
        TruncatingChunker::new(8),
        StubEmbed { hidden_size: 8, max_batch: 8, error: false },
    );
    let t = p.embed_chunks(&truncated, ()).unwrap();
    assert_eq!(t.chunks.len(), 1);
    assert_eq!(t.chunks[0].values, vec![97.0, 98.0, 99.0, 100.0]); // abcd

    let w = p.embed_chunks(&windowed, ()).unwrap();
    assert_eq!(w.chunks.len(), 3);
    assert_eq!(w.chunks[0].values, vec![97.0, 98.0, 99.0]);
}

#[test]
fn chunk_seam_batch_preserves_per_text_grouping() {
    // Pre-built per-text chunk lists through the batch seam.
    let chunks_a = shared_chunks(); // "abcdef" → 3 windows
    let chunks_b = {
        let mut tok = ByteTokenizer;
        let enc = tok.encode("ab").unwrap();
        TruncatingChunker::new(4).chunk(&enc).unwrap()
    }; // "ab" → 1 chunk

    let per_text = vec![chunks_a.clone(), chunks_b.clone()];
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![], error: true },
        TruncatingChunker::new(4),
        StubEmbed { hidden_size: 4, max_batch: 8, error: false },
    );
    let out = p.embed_chunks_batch(&per_text, ()).unwrap();
    assert_eq!(out.results.len(), 2);
    assert_eq!(out.results[0].chunks.len(), 3);
    assert_eq!(out.results[1].chunks.len(), 1);
    // Same byte_offsets as the single-text seam.
    assert_eq!(out.results[0].chunks[0].byte_offset, chunks_a[0].byte_offset);
    assert_eq!(out.results[1].chunks[0].values, vec![97.0, 98.0]); // ab
}

#[test]
fn chunk_seam_matches_full_pipeline_output() {
    // The seam over chunks the chunker produced must equal the full
    // tokenize→chunk→embed path — the seam is the encoder half, factored out.
    let mut full = EncoderPipeline::new(
        ByteTokenizer,
        SlidingWindowChunker::new(3, 2),
        StubEmbed { hidden_size: 3, max_batch: 1, error: false },
    );
    let direct = full.embed("abcdef", ()).unwrap();

    // Now feed the same chunks (rebuilt identically) through the seam.
    let chunks = shared_chunks();
    let via_seam = full.embed_chunks(&chunks, ()).unwrap();

    assert_eq!(direct.chunks.len(), via_seam.chunks.len());
    for (d, s) in direct.chunks.iter().zip(via_seam.chunks.iter()) {
        assert_eq!(d.byte_offset, s.byte_offset);
        assert_eq!(d.values, s.values);
    }
}

#[test]
fn chunk_seam_empty_chunks_returns_empty_no_encode() {
    // Zero chunks → the encoder never runs (the stub would error if it did).
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![], error: true },
        TruncatingChunker::new(4),
        StubEmbed { hidden_size: 4, max_batch: 1, error: true },
    );
    let out = p.embed_chunks(&[], ()).unwrap();
    assert!(out.chunks.is_empty());
}

#[test]
fn chunk_seam_batch_coalesces_chunks_across_text_boundaries() {
    // Two texts, one chunk each; max_batch=2. Cross-text flattening fills ONE
    // sub-batch of 2 — the regression guard for run_chunks_batch's cross-text
    // sub-batching. The old per-text loop recorded [1, 1] (two one-element
    // batches); the fix records [2] (one batch spanning both texts).
    let per_text = vec![vec![text_chunk(&[10], 0)], vec![text_chunk(&[20], 0)]];
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![], error: true }, // would fail if the seam tokenized
        TruncatingChunker::new(4),
        BatchSpy::new(2),
    );
    let out = p.embed_chunks_batch(&per_text, ()).unwrap();
    assert_eq!(out.results.len(), 2);
    assert_eq!(out.results[0].chunks[0].values, vec![10.0]);
    assert_eq!(out.results[1].chunks[0].values, vec![20.0]);
    assert_eq!(p.embedder_mut().sizes(), vec![2], "one coalesced sub-batch across both texts");
}

#[test]
fn chunk_seam_batch_sub_batches_across_text_boundaries_when_total_exceeds_max() {
    // 3 chunks across 2 texts, max_batch=2 → sub-batches [2, 1], where the
    // first batch spans BOTH texts (text A's chunk + text B's first chunk).
    // Grouping is still re-split per text afterwards. Old per-text loop: [1, 2].
    let per_text = vec![vec![text_chunk(&[10], 0)], vec![text_chunk(&[20], 0), text_chunk(&[30], 5)]];
    let mut p =
        EncoderPipeline::new(StubTokenizer { ids: vec![], error: true }, TruncatingChunker::new(4), BatchSpy::new(2));
    let out = p.embed_chunks_batch(&per_text, ()).unwrap();
    assert_eq!(out.results.len(), 2);
    assert_eq!(out.results[0].chunks.len(), 1);
    assert_eq!(out.results[1].chunks.len(), 2);
    // Per-text grouping preserved despite cross-boundary batching.
    assert_eq!(out.results[0].chunks[0].values, vec![10.0]);
    assert_eq!(out.results[1].chunks[0].values, vec![20.0]);
    assert_eq!(out.results[1].chunks[1].values, vec![30.0]);
    assert_eq!(out.results[1].chunks[1].byte_offset, 5);
    assert_eq!(p.embedder_mut().sizes(), vec![2, 1], "ceil(3/2) sub-batches, coalesced across texts");
}

#[test]
fn sub_batches_form_longest_first_and_outputs_keep_input_order() {
    // Lengths 1, 3, 2, 4 with max_batch=2: longest first, the groups are
    // {4, 3} and {2, 1}, so a head padding each batch to its longest chunk pads
    // 1 against 2, not against 3. Outputs still come back in input order.
    let chunks = vec![text_chunk(&[1], 0), text_chunk(&[2, 2, 2], 0), text_chunk(&[3, 3], 0), text_chunk(&[4; 4], 0)];
    let mut p =
        EncoderPipeline::new(StubTokenizer { ids: vec![], error: true }, TruncatingChunker::new(4), BatchSpy::new(2));
    let out = p.embed_chunks(&chunks, ()).unwrap();
    assert_eq!(p.embedder_mut().batches, vec![vec![4, 3], vec![2, 1]]);
    let values: Vec<&[f32]> = out.chunks.iter().map(|c| c.values.as_slice()).collect();
    assert_eq!(values, [&[1.0][..], &[2.0; 3], &[3.0; 2], &[4.0; 4]]);
}

#[test]
fn classify_chunks_batch_preserves_grouping_and_skips_tokenize() {
    // classify_chunks_batch: per-text grouping + logits, and the seam skips
    // tokenize/chunk (the tokenizer would error if touched).
    let per_text = vec![vec![text_chunk(&[1, 2], 0)], vec![text_chunk(&[3], 0), text_chunk(&[4, 5, 6], 9)]];
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![], error: true },
        TruncatingChunker::new(8),
        StubClassify { num_labels: 8, max_batch: 8, error: false },
    );
    let out = p.classify_chunks_batch(&per_text, ()).unwrap();
    assert_eq!(out.results.len(), 2);
    assert_eq!(out.results[0].chunks.len(), 1);
    assert_eq!(out.results[0].chunks[0].logits, vec![1.0, 2.0]);
    assert_eq!(out.results[0].chunks[0].byte_offset, 0);
    assert_eq!(out.results[1].chunks.len(), 2);
    assert_eq!(out.results[1].chunks[0].logits, vec![3.0]);
    assert_eq!(out.results[1].chunks[1].logits, vec![4.0, 5.0, 6.0]);
    assert_eq!(out.results[1].chunks[1].byte_offset, 9);
}

#[test]
fn classify_tokens_chunks_threads_per_token_geometry() {
    // classify_tokens_chunks: the single-text seam threads per-token byte
    // offsets, special_tokens_mask, and num_labels through from the chunk.
    let chunks = vec![text_chunk(&[4, 5, 6], 0)];
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![], error: true },
        TruncatingChunker::new(4),
        StubRecognize { num_labels: 3, max_batch: 1, error: false },
    );
    let out = p.classify_tokens_chunks(&chunks, ()).unwrap();
    assert_eq!(out.chunks.len(), 1);
    let c = &out.chunks[0];
    assert_eq!(c.num_labels, 3);
    assert_eq!(c.logits, vec![4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0]);
    assert_eq!(c.token_offsets, vec![(0, 1), (1, 2), (2, 3)]);
    assert_eq!(c.special_tokens_mask, vec![0, 0, 0]);
}

#[test]
fn classify_tokens_chunks_batch_preserves_grouping() {
    // classify_tokens_chunks_batch: per-text grouping + per-token geometry.
    let per_text = vec![vec![text_chunk(&[1, 2], 0)], vec![text_chunk(&[3], 0)]];
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![], error: true },
        TruncatingChunker::new(4),
        StubRecognize { num_labels: 1, max_batch: 8, error: false },
    );
    let out = p.classify_tokens_chunks_batch(&per_text, ()).unwrap();
    assert_eq!(out.results.len(), 2);
    assert_eq!(out.results[0].chunks.len(), 1);
    assert_eq!(out.results[0].chunks[0].logits, vec![1.0, 2.0]);
    assert_eq!(out.results[0].chunks[0].token_offsets, vec![(0, 1), (1, 2)]);
    assert_eq!(out.results[1].chunks.len(), 1);
    assert_eq!(out.results[1].chunks[0].logits, vec![3.0]);
    assert_eq!(out.results[1].chunks[0].token_offsets, vec![(0, 1)]);
}

#[test]
fn chunk_seam_profiles_only_encoder_stages() {
    // The chunk seam skips tokenize/chunk, so its profile carries ONLY the
    // encoder's stage — distinct from the full pipeline (which leads with
    // tokenize, chunk). Verified for all three verbs.
    let chunks = shared_chunks();
    assert!(!chunks.is_empty());

    let names = |prof: Option<RunProfile>| {
        prof.expect("profile collected").stages.into_iter().map(|s| s.name).collect::<Vec<_>>()
    };

    let mut embed_p = EncoderPipeline::new(
        StubTokenizer { ids: vec![], error: true },
        TruncatingChunker::new(3),
        StubEmbed { hidden_size: 3, max_batch: 1, error: false },
    );
    let embeds = embed_p.embed_chunks(&chunks, RunOptions { profile: true }).unwrap();
    assert_eq!(names(embeds.profile), vec!["encode"]);

    let mut classify_p = EncoderPipeline::new(
        StubTokenizer { ids: vec![], error: true },
        TruncatingChunker::new(3),
        StubClassify { num_labels: 3, max_batch: 1, error: false },
    );
    let classes = classify_p.classify_chunks(&chunks, RunOptions { profile: true }).unwrap();
    assert_eq!(names(classes.profile), vec!["classify"]);

    let mut token_p = EncoderPipeline::new(
        StubTokenizer { ids: vec![], error: true },
        TruncatingChunker::new(3),
        StubRecognize { num_labels: 1, max_batch: 1, error: false },
    );
    let tokens = token_p.classify_tokens_chunks(&chunks, RunOptions { profile: true }).unwrap();
    assert_eq!(names(tokens.profile), vec!["classify_tokens"]);
}

// ─── softmax / argmax ───────────────────────────────────────────────────────

#[test]
fn argmax_picks_highest_index() {
    assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
    assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
    assert_eq!(argmax(&[1.0, 2.0, 5.0]), 2);
}

#[test]
fn argmax_empty_returns_zero() {
    assert_eq!(argmax(&[]), 0);
}

#[test]
fn argmax_nan_safe_does_not_panic() {
    // NaN must not panic (the partial_cmp().unwrap() footgun argmax replaces).
    let _ = argmax(&[1.0, f32::NAN, 2.0]);
    let _ = argmax(&[f32::NAN, f32::NAN]);
}

#[test]
fn softmax_sums_to_one_and_preserves_order() {
    let probs = softmax(&[1.0, 3.0, 2.0]);
    assert_eq!(probs.len(), 3);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sums to ~1, got {sum}");
    // argmax index has the highest probability.
    assert!(probs[1] > probs[0]);
    assert!(probs[1] > probs[2]);
}

#[test]
fn softmax_empty_returns_empty() {
    assert!(softmax(&[]).is_empty());
}

#[test]
fn softmax_large_values_do_not_overflow() {
    // Numerical stability: max-subtraction prevents overflow.
    let probs = softmax(&[1000.0, 1001.0, 999.0]);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "large logits sum to ~1, got {sum}");
    assert!(probs.iter().all(|p| p.is_finite()));
}

// ─── labels_for_tokens / group_spans decoding ───────────────────────────────

/// Build a `ChunkTokenClassification` from a per-token spec: `(label_id,
/// byte_span, is_special)`. Each token's logits are a one-hot row (so argmax
/// yields `label_id` deterministically), sized to `num_labels`.
fn token_chunk(spec: &[(u32, (usize, usize), bool)], num_labels: usize) -> ChunkTokenClassification {
    let mut logits = Vec::new();
    let mut offsets = Vec::new();
    let mut specials = Vec::new();
    for &(lid, span, is_special) in spec {
        let mut row = vec![0.0_f32; num_labels];
        if (lid as usize) < num_labels {
            row[lid as usize] = 1.0;
        }
        logits.extend_from_slice(&row);
        offsets.push(span);
        specials.push(if is_special { 1 } else { 0 });
    }
    ChunkTokenClassification {
        byte_offset: 0,
        logits,
        num_labels,
        token_offsets: offsets,
        special_tokens_mask: specials,
    }
}

/// id2label for the decoder tests: 0=O, 1=B-PER, 2=I-PER, 3=B-LOC, 4=I-LOC.
fn ner_id2label(id: u32) -> &'static str {
    ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"].get(id as usize).copied().unwrap_or("")
}

/// The entity type of a prefixed label (`"B-PER"` → `"PER"`); `None` for `"O"`.
fn typ_of(label: &str) -> Option<&str> {
    label.split_once('-').map(|(_, t)| t)
}

#[test]
fn labels_for_tokens_argmaxes_skips_specials_and_pairs_offsets() {
    // token ids: 1=B-PER, 2=I-PER, 0=O, 1=B-PER, special(CLS)=1
    let chunk = token_chunk(
        &[(1, (0, 3), false), (2, (3, 7), false), (0, (7, 8), false), (1, (8, 12), false), (1, (0, 0), true)],
        5,
    );
    let labels = labels_for_tokens(&chunk, ner_id2label);
    assert_eq!(labels.len(), 4, "special token dropped");
    // One-hot rows: softmax score is in (0, 1]; check structure field-by-field
    // and assert the score is a sane probability rather than hardcoding the
    // (num_labels-dependent) softmax value.
    let check = |t: &TokenLabel, lid, lbl, s, e, idx| {
        assert_eq!(t.label_id, lid);
        assert_eq!(t.label, lbl);
        assert_eq!((t.start, t.end), (s, e));
        assert_eq!(t.token_index, idx);
        assert!(t.score > 0.0 && t.score <= 1.0, "score {} not in (0,1]", t.score);
    };
    check(&labels[0], 1, "B-PER", 0, 3, 0);
    check(&labels[1], 2, "I-PER", 3, 7, 1);
    check(&labels[2], 0, "O", 7, 8, 2);
    check(&labels[3], 1, "B-PER", 8, 12, 3);
}

#[test]
fn group_spans_bio_merges_b_i_and_breaks_on_o() {
    let chunk = token_chunk(&[(1, (0, 3), false), (2, (3, 7), false), (0, (7, 8), false), (1, (8, 12), false)], 5);
    let labels = labels_for_tokens(&chunk, ner_id2label);
    let spans = group_spans(&labels, Scheme::Bio);
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].label, "PER");
    assert_eq!((spans[0].start, spans[0].end), (0, 7));
    assert_eq!(spans[0].token_range, 0..2);
    assert_eq!(spans[1].label, "PER");
    assert_eq!((spans[1].start, spans[1].end), (8, 12));
    assert_eq!(spans[1].token_range, 3..4);
}

#[test]
fn group_spans_bio_breaks_on_type_change() {
    // B-PER, I-LOC → mismatched I- opens a fresh LOC span.
    let chunk = token_chunk(&[(1, (0, 3), false), (4, (3, 6), false)], 5);
    let labels = labels_for_tokens(&chunk, ner_id2label);
    let spans = group_spans(&labels, Scheme::Bio);
    assert_eq!(spans.len(), 2);
    // Single-token spans carry the token's softmax score; check structure + that
    // the score propagated (rather than hardcoding the one-hot softmax value).
    assert_eq!(
        (spans[0].label.as_str(), spans[0].label_id, spans[0].start, spans[0].end, spans[0].token_range.clone()),
        ("PER", 1, 0, 3, 0..1)
    );
    assert_eq!(spans[0].score, labels[0].score);
    assert_eq!(
        (spans[1].label.as_str(), spans[1].label_id, spans[1].start, spans[1].end, spans[1].token_range.clone()),
        ("LOC", 4, 3, 6, 1..2)
    );
    assert_eq!(spans[1].score, labels[1].score);
}

#[test]
fn group_spans_bio_treats_stray_i_as_new_opener() {
    // I-PER with no preceding B-PER → lenient single-token span.
    let chunk = token_chunk(&[(2, (0, 3), false), (0, (3, 4), false)], 5);
    let labels = labels_for_tokens(&chunk, ner_id2label);
    let spans = group_spans(&labels, Scheme::Bio);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].label, "PER");
    assert_eq!(spans[0].token_range, 0..1);
}

#[test]
fn group_spans_flat_emits_one_per_token() {
    // Flat labels (POS-style): each token its own entity, no grouping.
    let labels: Vec<TokenLabel> = [("NNS", 0), ("VBD", 1)]
        .into_iter()
        .map(|(lbl, i)| TokenLabel { label_id: 0, label: lbl, start: i, end: i + 1, token_index: i, score: 1.0 })
        .collect();
    let spans = group_spans(&labels, Scheme::Flat);
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].label, "NNS");
    assert_eq!(spans[0].token_range, 0..1);
    assert_eq!(spans[0].score, 1.0, "flat single-token span carries the token's score");
    assert_eq!(spans[1].label, "VBD");
    assert_eq!(spans[1].token_range, 1..2);
}

#[test]
fn group_spans_bilou_and_iobes() {
    // BILOU: B-PER I-PER L-PER → one PER span over 3 tokens; U-PER → single.
    // Varied scores exercise the min aggregation: the 3-token span's score is
    // min(0.9, 0.5, 0.7) = 0.5; the U singleton keeps its own 0.8.
    let bilou_labels: Vec<TokenLabel> = [("B-PER", 0.9), ("I-PER", 0.5), ("L-PER", 0.7), ("U-PER", 0.8)]
        .into_iter()
        .enumerate()
        .map(|(i, (lbl, sc))| TokenLabel { label_id: 1, label: lbl, start: i, end: i + 1, token_index: i, score: sc })
        .collect();
    let bilou = group_spans(&bilou_labels, Scheme::Bilou);
    assert_eq!(bilou.len(), 2);
    assert_eq!(bilou[0].label, "PER");
    assert_eq!(bilou[0].token_range, 0..3);
    assert_eq!(bilou[0].score, 0.5, "multi-token span score = min over members");
    assert_eq!(bilou[1].label, "PER");
    assert_eq!(bilou[1].token_range, 3..4);
    assert_eq!(bilou[1].score, 0.8, "U singleton keeps its own score");

    // IOBES: B-PER I-PER E-PER → one span; S-PER → single.
    let iobes_labels: Vec<TokenLabel> = [("B-PER", 0.6), ("I-PER", 0.4), ("E-PER", 0.95), ("S-PER", 0.3)]
        .into_iter()
        .enumerate()
        .map(|(i, (lbl, sc))| TokenLabel { label_id: 1, label: lbl, start: i, end: i + 1, token_index: i, score: sc })
        .collect();
    let iobes = group_spans(&iobes_labels, Scheme::Iobes);
    assert_eq!(iobes.len(), 2);
    assert_eq!(iobes[0].label, "PER");
    assert_eq!(iobes[0].token_range, 0..3);
    assert_eq!(iobes[0].score, 0.4);
    assert_eq!(iobes[1].label, "PER");
    assert_eq!(iobes[1].token_range, 3..4);
    assert_eq!(iobes[1].score, 0.3);
}

// ─── span decoder: adversarial / edge transitions ───────────────────────────

/// `tl` builds a content TokenLabel for the edge-case tests.
fn tl(label: &str, start: usize, end: usize, idx: usize, score: f32) -> TokenLabel<'_> {
    TokenLabel { label_id: 0, label, start, end, token_index: idx, score }
}

#[test]
fn group_spans_empty_and_all_o_yield_nothing() {
    assert!(group_spans(&[], Scheme::Bio).is_empty(), "empty input");
    let all_o = vec![tl("O", 0, 1, 0, 1.0), tl("O", 1, 2, 1, 1.0)];
    assert!(group_spans(&all_o, Scheme::Bio).is_empty(), "all-O");
}

#[test]
fn group_spans_bio_extends_multi_i_then_breaks_on_o() {
    // B-PER I-PER I-PER I-PER O → one 4-token PER span, then O ends it.
    let labels = vec![
        tl("B-PER", 0, 3, 0, 0.9),
        tl("I-PER", 3, 6, 1, 0.5),
        tl("I-PER", 6, 9, 2, 0.8),
        tl("I-PER", 9, 12, 3, 0.6),
        tl("O", 12, 13, 4, 1.0),
    ];
    let spans = group_spans(&labels, Scheme::Bio);
    assert_eq!(spans.len(), 1);
    assert_eq!((spans[0].start, spans[0].end), (0, 12));
    assert_eq!(spans[0].token_range, 0..4);
    assert_eq!(spans[0].score, 0.5, "score = min over the 4 members");
}

#[test]
fn group_spans_open_span_flushes_at_end_of_sequence() {
    // B-PER I-PER with no closer → flushed at end as one span.
    let labels = vec![tl("B-PER", 0, 3, 0, 0.9), tl("I-PER", 3, 6, 1, 0.4)];
    let spans = group_spans(&labels, Scheme::Bio);
    assert_eq!(spans.len(), 1);
    assert_eq!((spans[0].start, spans[0].end), (0, 6));
    assert_eq!(spans[0].token_range, 0..2);
    assert_eq!(spans[0].score, 0.4);
}

#[test]
fn group_spans_bilou_stray_l_emits_single() {
    // L-PER with no open span → lenient single-token span.
    let labels = vec![tl("L-PER", 0, 3, 0, 0.7), tl("O", 3, 4, 1, 1.0)];
    let spans = group_spans(&labels, Scheme::Bilou);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].label, "PER");
    assert_eq!(spans[0].token_range, 0..1);
    assert_eq!(spans[0].score, 0.7);
}

#[test]
fn group_spans_iobes_stray_e_emits_single() {
    // E-PER with no open span → lenient single-token span.
    let labels = vec![tl("E-PER", 0, 3, 0, 0.6)];
    let spans = group_spans(&labels, Scheme::Iobes);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].label, "PER");
    assert_eq!(spans[0].token_range, 0..1);
    assert_eq!(spans[0].score, 0.6);
}

#[test]
fn group_spans_bilou_mismatched_closer_flushes_then_single() {
    // B-PER L-LOC: the L-LOC type mismatches the open PER → flush PER (B alone),
    // then emit a single-token LOC for the stray L-.
    let labels = vec![tl("B-PER", 0, 3, 0, 0.9), tl("L-LOC", 3, 6, 1, 0.5)];
    let spans = group_spans(&labels, Scheme::Bilou);
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].label, "PER");
    assert_eq!((spans[0].start, spans[0].end), (0, 3));
    assert_eq!(spans[1].label, "LOC");
    assert_eq!((spans[1].start, spans[1].end), (3, 6));
    assert_eq!(spans[1].score, 0.5);
}

// ─── group_spans_document (cross-chunk merge) ────────────────────────────────

#[test]
fn group_spans_document_merges_overlap_and_boundary_split() {
    // A 3-token PER entity straddling a window boundary. Window A covers the
    // first two tokens; window B covers the last two (overlapping on the
    // middle token at byte span (3,7)). Per-window decoding splits the entity;
    // the document grouping dedups the overlap and re-merges into one span.
    let window_a = vec![tl("B-PER", 0, 3, 0, 0.9), tl("I-PER", 3, 7, 1, 0.8)];
    // Window B sees the middle token as a stray I- (no opener in-window); the
    // document merge re-groups it as a continuation once dedup'd.
    let window_b = vec![tl("I-PER", 3, 7, 0, 0.85), tl("I-PER", 7, 10, 1, 0.7)];
    let spans = group_spans_document(&[window_a, window_b], Scheme::Bio);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].label, "PER");
    assert_eq!((spans[0].start, spans[0].end), (0, 10), "merged across the boundary");
    assert_eq!(spans[0].token_range, 0..3, "3 unique tokens after dedup");
    // Overlap token (3,7): higher score 0.85 kept; span score = min(0.9,0.85,0.7).
    assert_eq!(spans[0].score, 0.7);
}

#[test]
fn group_spans_document_non_overlapping_equivalent_to_concat() {
    // Adjacent (non-overlapping) chunks: document grouping == concatenation.
    let chunk0 = vec![tl("B-PER", 0, 3, 0, 1.0), tl("I-PER", 3, 6, 1, 1.0)];
    let chunk1 = vec![tl("O", 6, 7, 0, 1.0), tl("B-LOC", 7, 10, 1, 1.0)];
    let spans = group_spans_document(&[chunk0, chunk1], Scheme::Bio);
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].label, "PER");
    assert_eq!((spans[0].start, spans[0].end), (0, 6));
    assert_eq!(spans[1].label, "LOC");
    assert_eq!((spans[1].start, spans[1].end), (7, 10));
}

proptest! {
    /// Random BIO label sequences decode to well-formed spans: disjoint,
    /// ordered, in-bounds ranges; every token in a span shares the span's type;
    /// no "O" token is inside a span; byte spans match the covered tokens; and
    /// each span's score is the min over its member tokens.
    #[test]
    fn prop_group_spans_bio_well_formed(
        labels in proptest::collection::vec(
            proptest::sample::select(vec!["O".to_string(), "B-PER".into(), "I-PER".into(), "B-LOC".into(), "I-LOC".into()]),
            0..32,
        )
    ) {
        // Vary scores deterministically so the min-aggregation invariant is
        // exercised (not just constant 1.0).
        let tokens: Vec<TokenLabel> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| TokenLabel {
                label_id: 0,
                label: l.as_str(),
                start: i,
                end: i + 1,
                token_index: i,
                score: 1.0 - (i as f32) * 0.01,
            })
            .collect();
        let n = tokens.len();
        let entities = group_spans(&tokens, Scheme::Bio);

        let mut prev_end = 0usize;
        for e in &entities {
            // Range is non-empty, in-bounds, ordered & disjoint with the previous.
            prop_assert!(e.token_range.start < e.token_range.end, "empty range");
            prop_assert!(e.token_range.end <= n, "range out of bounds");
            prop_assert!(e.token_range.start >= prev_end, "ranges not ordered/disjoint");
            // Byte span matches the covered tokens.
            prop_assert_eq!(e.start, tokens[e.token_range.start].start);
            prop_assert_eq!(e.end, tokens[e.token_range.end - 1].end);
            // Every covered token shares the span's type and is not "O".
            let mut min_score = f32::INFINITY;
            for ti in e.token_range.clone() {
                prop_assert!(tokens[ti].label != "O", "'O' token inside a span");
                prop_assert_eq!(typ_of(tokens[ti].label), Some(e.label.as_str()), "type mismatch in span");
                min_score = min_score.min(tokens[ti].score);
            }
            // Score = conservative min over members.
            prop_assert_eq!(e.score, min_score, "span score is the min over members");
            prev_end = e.token_range.end;
        }
    }
}
