use std::convert::Infallible;
use std::time::Duration;

use proptest::prelude::*;
use test_case::test_case;

#[cfg(feature = "hf-tokenizers")]
use crate::pipelines::text::HfTokenizer;
use crate::pipelines::text::{
    ChunkTokenClassification, Chunker, Classification, Classify, ClassifyTokens, Embed, Embedding, EncoderHead,
    EncoderPipeline, EncoderPipelineError, Encoding, RunOptions, RunProfile, Scheme, SlidingWindowChunker,
    SlidingWindowChunkerError, TextChunk, TokenClassification, TokenLabel, TokenScores, Tokenizer, TruncatingChunker,
    TruncatingChunkerError, WindowWeighting, argmax, group_spans, group_spans_document, labels_for_tokens, sigmoid,
    softmax, token_scores_document,
};

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

fn stage_names(profile: Option<RunProfile>) -> Vec<String> {
    profile.expect("profile collected").stages.into_iter().map(|s| s.name).collect()
}

// ─── Stubs (host-only; no tokenizer.json, no model/device) ────────────────────

/// Returns the same canned encoding for every input text.
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

/// Encodes each input byte as its own token id: `"ab"` → `[97, 98]`, so batch
/// tests can tell which text produced which output.
struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    type Error = Infallible;
    fn encode(&mut self, text: &str) -> Result<Encoding, Infallible> {
        Ok(enc(&text.bytes().map(|b| b as u32).collect::<Vec<_>>()))
    }
}

/// A stub head's profile: one fixed 1 ms `stage`, so stages merged over
/// sub-batches add up exactly.
fn stub_profile(profile: bool, stage: &str) -> Option<RunProfile> {
    profile.then(|| {
        let mut p = RunProfile::default();
        p.push(svod_runtime::StageProfile::host(stage, Duration::from_millis(1)));
        p
    })
}

/// Echoes ids as the embedding (values = ids as f32); profiles as `encode`.
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
        Ok((values, stub_profile(profile, "encode")))
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

/// Echoes ids as the class logits; profiles as `classify`.
struct StubClassify {
    num_labels: usize,
    max_batch: usize,
}

impl EncoderHead for StubClassify {
    type Output = Classification;
    type Error = Infallible;
    fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.num_labels)
    }
    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> Result<(Vec<Self::Output>, Option<RunProfile>), Self::Error> {
        let values = batch
            .iter()
            .map(|e| Classification { logits: e.input_ids.iter().map(|&id| id as f32).collect() })
            .collect();
        Ok((values, stub_profile(profile, "classify")))
    }
}

impl Classify for StubClassify {
    fn num_labels(&self) -> usize {
        self.num_labels
    }
}

/// Turns each token id `v` into a row of `num_labels` copies of `v as f32`;
/// profiles as `classify_tokens`.
struct StubRecognize {
    num_labels: usize,
    max_batch: usize,
}

impl EncoderHead for StubRecognize {
    type Output = TokenClassification;
    type Error = Infallible;
    fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.num_labels)
    }
    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> Result<(Vec<Self::Output>, Option<RunProfile>), Self::Error> {
        let nl = self.num_labels;
        let values = batch
            .iter()
            .map(|e| {
                let logits = e.input_ids.iter().flat_map(|&id| std::iter::repeat_n(id as f32, nl)).collect();
                TokenClassification { logits, num_labels: nl }
            })
            .collect();
        Ok((values, stub_profile(profile, "classify_tokens")))
    }
}

impl ClassifyTokens for StubRecognize {
    fn num_labels(&self) -> usize {
        self.num_labels
    }
}

/// Records the chunk lengths of every batch handed to `run_batch` (then echoes
/// ids as embeddings) — lets a test observe *how* the pipeline sub-batches.
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
}

#[test]
#[should_panic(expected = "max_seq must be >= 1")]
fn truncating_new_rejects_zero() {
    let _ = TruncatingChunker::new(0);
}

#[test]
fn truncating_try_new_bounds_max_seq_at_one() {
    assert!(matches!(TruncatingChunker::try_new(0).unwrap_err(), TruncatingChunkerError::MaxSeqTooSmall));
    assert!(TruncatingChunker::try_new(1).is_ok());
}

// ─── SlidingWindowChunker ─────────────────────────────────────────────────────

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

    // Each window's first content token, counted over the text's content.
    assert_eq!(out.iter().map(|c| c.first_token).collect::<Vec<_>>(), [0, 2, 4]);
}

#[test]
fn sliding_last_window_clamped_when_content_uneven() {
    // window=4 → content_window=2; stride=4 → step clamped to 2, so no content
    // is skipped. 5 content tokens → the last window gets 1.
    let mut chunker = SlidingWindowChunker::new(4, 4);
    let out = chunker.chunk(&enc_with_specials(&[10, 20, 30, 40, 50])).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].encoding.input_ids, vec![2, 10, 20, 3]);
    assert_eq!(out[1].encoding.input_ids, vec![2, 30, 40, 3]);
    assert_eq!(out[2].encoding.input_ids, vec![2, 50, 3]);
    assert_eq!(out[2].byte_offset, 4);
}

/// Pulled back, the last window ends at the text's end holding a full window
/// of content, and its first token and byte offset move with it. A text
/// shorter than one window stays one window. Each row is `(first_token,
/// byte_offset, content tokens)` per window.
#[test_case(6, 5, 2 => vec![(0, 0, 3), (2, 2, 3), (3, 3, 3)]; "the last window steps back")]
#[test_case(5, 4, 4 => vec![(0, 0, 2), (2, 2, 2), (3, 3, 2)]; "adjacent windows overlap at the end")]
#[test_case(2, 5, 2 => vec![(0, 0, 2)]; "shorter than a window")]
fn sliding_full_last_window_ends_full(content: u32, window: usize, stride: usize) -> Vec<(usize, usize, usize)> {
    let ids: Vec<u32> = (10..10 + content).collect();
    let mut chunker = SlidingWindowChunker::new(window, stride).with_full_last_window();
    let chunks = chunker.chunk(&enc_with_specials(&ids)).unwrap();
    chunks.iter().map(|c| (c.first_token, c.byte_offset, c.encoding.len() - 2)).collect()
}

#[test]
#[should_panic(expected = "window must be >= 1")]
fn sliding_new_rejects_zero_window() {
    let _ = SlidingWindowChunker::new(0, 1);
}

#[test]
#[should_panic(expected = "stride must be in 1..=window")]
fn sliding_new_rejects_stride_above_window() {
    let _ = SlidingWindowChunker::new(4, 5);
}

#[test]
fn sliding_try_new_bounds_window_and_stride() {
    assert!(matches!(SlidingWindowChunker::try_new(0, 1).unwrap_err(), SlidingWindowChunkerError::WindowTooSmall));
    assert!(matches!(SlidingWindowChunker::try_new(4, 0).unwrap_err(), SlidingWindowChunkerError::StrideOutOfRange));
    assert!(matches!(SlidingWindowChunker::try_new(4, 5).unwrap_err(), SlidingWindowChunkerError::StrideOutOfRange));
    assert!(SlidingWindowChunker::try_new(1, 1).is_ok());
    // The encoder is sized for full windows, not for the stride.
    assert_eq!(SlidingWindowChunker::try_new(4, 2).unwrap().max_seq(), 4);
}

#[test]
fn sliding_chunk_needs_one_content_slot_beyond_specials() {
    // CLS + SEP take two slots: window=3 leaves one content token per window,
    // window=2 leaves none. The specials come from the tokenizer output, so the
    // shortfall is a chunk-time Err, not a constructor panic.
    let e = enc_with_specials(&[10, 20]);
    let ids: Vec<Vec<u32>> =
        SlidingWindowChunker::new(3, 1).chunk(&e).unwrap().into_iter().map(|c| c.encoding.input_ids).collect();
    assert_eq!(ids, [[2, 10, 3], [2, 20, 3]]);
    let err = SlidingWindowChunker::new(2, 1).chunk(&e).unwrap_err();
    assert!(matches!(err, SlidingWindowChunkerError::WindowTooSmallForSpecials { window: 2, specials: 2 }));
}

// ─── EncoderPipeline: single text ─────────────────────────────────────────────

#[test]
fn encoder_head_run_is_batch_of_one() {
    let mut embed = StubEmbed { hidden_size: 3, max_batch: 1, error: false };
    let e = enc(&[4, 5, 6]);
    let single = embed.run(&e, false).unwrap().0;
    let batch = embed.run_batch(&[&e], false).unwrap().0;
    assert_eq!(single, batch.into_iter().next().unwrap());
}

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
    assert_eq!(out.chunks[0].byte_offset, 0);
    assert_eq!(out.chunks[0].values, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(out.profile.is_none(), "default options don't profile");
}

#[test]
fn pipeline_profiles_stage_order_tokenize_then_chunk_then_encoder() {
    let mut p = pipeline(vec![1, 2, 3], 8);
    let out = p.embed("ignored", RunOptions { profile: true }).unwrap();
    assert_eq!(stage_names(out.profile), ["tokenize", "chunk", "encode"], "host stages lead, then the encoder's");
}

#[test]
fn pipeline_zero_chunk_run_profiles_only_host_stages() {
    // "" has no content tokens, so the sliding chunker yields no chunks and the
    // encoder contributes no stage.
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        SlidingWindowChunker::new(4, 2),
        StubEmbed { hidden_size: 4, max_batch: 1, error: false },
    );
    let out = p.embed("", RunOptions { profile: true }).unwrap();
    assert!(out.chunks.is_empty());
    assert_eq!(stage_names(out.profile), ["tokenize", "chunk"]);
}

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

#[test]
fn tokenize_error_maps_to_tokenize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1], error: true },
        TruncatingChunker::new(4),
        StubEmbed { hidden_size: 4, max_batch: 1, error: false },
    );
    let err = p.embed("ignored", ()).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Tokenize { .. }));
}

#[derive(Debug, snafu::Snafu)]
#[snafu(display("stub chunker error"))]
struct ErrChunkerError;

/// Always fails `chunk`.
struct ErrChunker;

impl Chunker for ErrChunker {
    type Error = ErrChunkerError;
    fn max_seq(&self) -> usize {
        4
    }
    fn chunk(&mut self, _enc: &Encoding) -> Result<Vec<TextChunk>, ErrChunkerError> {
        Err(ErrChunkerError)
    }
}

#[test]
fn chunk_error_maps_to_chunk_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], error: false },
        ErrChunker,
        StubEmbed { hidden_size: 4, max_batch: 1, error: false },
    );
    let err = p.embed("ignored", ()).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Chunk { .. }));
}

#[test]
fn encode_error_maps_to_encode_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], error: false },
        TruncatingChunker::new(4),
        StubEmbed { hidden_size: 4, max_batch: 1, error: true },
    );
    let err = p.embed("ignored", ()).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Encode { .. }));
}

// ─── HfTokenizer: a real tokenizers::Tokenizer fixture ─────────────────────────
//
// Hand-built WordPiece tokenizer: tiny vocab + Whitespace pre-tokenizer +
// BertProcessing wrapping each sequence in [CLS] … [SEP]. The ids are fully
// predictable for known input, so encode() assertions are exact (specials get
// non-trivial special_tokens_mask). Built programmatically; `fixture_json`
// serializes it so from_bytes exercises the real JSON deserialization path
// rather than a hand-written string.

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
    // [CLS] hello world [SEP]; every field is copied out of the HF encoding.
    assert_eq!(enc.input_ids, vec![2, 4, 5, 3]);
    assert_eq!(enc.attention_mask, vec![1, 1, 1, 1]);
    assert_eq!(enc.token_type_ids, vec![0, 0, 0, 0]);
    assert_eq!(enc.offsets, vec![(0, 0), (0, 5), (6, 11), (0, 0)]);
    assert_eq!(enc.special_tokens_mask, vec![1, 0, 0, 1]);
}

#[cfg(feature = "hf-tokenizers")]
#[test]
fn hf_tokenizer_encode_batch_matches_per_input_encode() {
    // The native HF batch path (`inner.encode_batch`) is the production
    // tokenization path for embed_batch / classify_batch / classify_tokens_batch,
    // distinct from the default `encode_batch` loop. It must agree field-for-field
    // with looping `encode`.
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
    assert_eq!(out.results[0].chunks.len(), 1);
    assert_eq!(out.results[0].chunks[0].values, vec![97.0, 98.0]);
    assert_eq!(out.results[1].chunks[0].values, vec![120.0, 121.0, 122.0]);
    assert_eq!(out.results[2].chunks[0].values, vec![104.0, 101.0, 108.0, 108.0, 111.0]);
}

#[test]
fn batch_empty_texts_profiles_host_stages() {
    // Like a single run that yields no chunks: host stages only.
    let mut p = EncoderPipeline::new(
        ByteTokenizer,
        TruncatingChunker::new(8),
        StubEmbed { hidden_size: 8, max_batch: 4, error: false },
    );
    let out = p.embed_batch(&[], RunOptions { profile: true }).unwrap();
    assert!(out.results.is_empty());
    assert_eq!(stage_names(out.profile), ["tokenize", "chunk"]);
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
fn embed_batch_sub_batches_across_text_boundaries_with_sliding_window() {
    // "abcdefghi" → 3 windows, "jkl" → 1. Flattened [abc, def, ghi, jkl] with
    // max_batch=2 → sub-batches [abc, def] and [ghi, jkl]; the second straddles
    // the text boundary, and outputs are re-split per text by chunk count.
    let mut p = EncoderPipeline::new(ByteTokenizer, SlidingWindowChunker::new(3, 3), BatchSpy::new(2));
    let out = p.embed_batch(&["abcdefghi", "jkl"], ()).unwrap();
    assert_eq!(out.results.len(), 2);
    assert_eq!(p.model_mut().sizes(), vec![2, 2]);

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
    assert!(out.results.iter().all(|r| r.profile.is_none()));
    assert_eq!(stage_names(out.profile), ["tokenize", "chunk", "encode"]);
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

// ─── classify / classify_tokens facades ──────────────────────────────────────

#[test]
fn classify_sliding_pipeline_produces_per_window_classifications() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![10, 20, 30, 40, 50, 60], error: false },
        SlidingWindowChunker::new(3, 2),
        StubClassify { num_labels: 3, max_batch: 1 },
    );
    let out = p.classify("ignored", RunOptions { profile: true }).unwrap();
    assert_eq!(out.chunks.len(), 3);
    assert_eq!(out.chunks[0].byte_offset, 0);
    assert_eq!(out.chunks[0].logits, vec![10.0, 20.0, 30.0]);
    assert_eq!(out.chunks[1].byte_offset, 2);
    assert_eq!(out.chunks[1].logits, vec![30.0, 40.0, 50.0]);
    assert_eq!(out.chunks[2].byte_offset, 4);
    assert_eq!(out.chunks[2].logits, vec![50.0, 60.0]);
    assert_eq!(stage_names(out.profile), ["tokenize", "chunk", "classify"]);
}

#[test]
fn classify_batch_basic_three_texts() {
    let mut p =
        EncoderPipeline::new(ByteTokenizer, TruncatingChunker::new(32), StubClassify { num_labels: 32, max_batch: 8 });
    let out = p.classify_batch(&["ab", "xyz", "hello"], RunOptions { profile: true }).unwrap();
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.results[0].chunks.len(), 1);
    assert_eq!(out.results[0].chunks[0].logits, vec![97.0, 98.0]);
    assert_eq!(out.results[1].chunks[0].logits, vec![120.0, 121.0, 122.0]);
    assert_eq!(out.results[2].chunks[0].logits, vec![104.0, 101.0, 108.0, 108.0, 111.0]);
    assert!(out.results.iter().all(|r| r.profile.is_none()));
    assert_eq!(stage_names(out.profile), ["tokenize", "chunk", "classify"]);
}

#[test]
fn classify_tokens_sliding_pipeline_produces_per_window_classifications() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![10, 20, 30, 40, 50, 60], error: false },
        SlidingWindowChunker::new(3, 2),
        StubRecognize { num_labels: 1, max_batch: 1 },
    );
    let out = p.classify_tokens("ignored", RunOptions { profile: true }).unwrap();
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
    assert_eq!(out.chunks.iter().map(|c| c.first_token).collect::<Vec<_>>(), [0, 2, 4]);
    assert_eq!(stage_names(out.profile), ["tokenize", "chunk", "classify_tokens"]);
}

#[test]
fn classify_tokens_batch_basic_three_texts() {
    let mut p =
        EncoderPipeline::new(ByteTokenizer, TruncatingChunker::new(32), StubRecognize { num_labels: 1, max_batch: 8 });
    let out = p.classify_tokens_batch(&["ab", "xyz", "hello"], RunOptions { profile: true }).unwrap();
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.results[0].chunks.len(), 1);
    assert_eq!(out.results[0].chunks[0].logits, vec![97.0, 98.0]);
    assert_eq!(out.results[0].chunks[0].num_labels, 1);
    assert_eq!(out.results[0].chunks[0].token_offsets, vec![(0, 1), (1, 2)]);
    assert_eq!(out.results[1].chunks[0].logits, vec![120.0, 121.0, 122.0]);
    assert_eq!(out.results[2].chunks[0].logits, vec![104.0, 101.0, 108.0, 108.0, 111.0]);
    assert!(out.results.iter().all(|r| r.profile.is_none()));
    assert_eq!(stage_names(out.profile), ["tokenize", "chunk", "classify_tokens"]);
}

// ─── chunk-level seam (*_chunks / *_chunks_batch) ───────────────────────────

/// "abcdef" through a SlidingWindowChunker(3, 2): windows `abc`, `cde`, `ef` at
/// byte offsets 0, 2, 4.
fn shared_chunks() -> Vec<TextChunk> {
    let enc = ByteTokenizer.encode("abcdef").unwrap();
    SlidingWindowChunker::new(3, 2).chunk(&enc).unwrap()
}

/// Build one [`TextChunk`] directly from ids (offsets count up, no specials).
fn text_chunk(ids: &[u32], byte_offset: usize) -> TextChunk {
    TextChunk { encoding: enc(ids), byte_offset, first_token: 0 }
}

/// A pipeline driven only through its chunk seam; its tokenizer and chunker
/// never run.
fn seam<M: EncoderHead>(model: M) -> EncoderPipeline<ByteTokenizer, TruncatingChunker, M> {
    EncoderPipeline::new(ByteTokenizer, TruncatingChunker::new(1), model)
}

#[test]
fn classify_chunks_pairs_logits_with_chunk_offsets() {
    let mut p = seam(StubClassify { num_labels: 3, max_batch: 1 });
    let out = p.classify_chunks(&shared_chunks(), ()).unwrap();
    let offsets: Vec<usize> = out.chunks.iter().map(|c| c.byte_offset).collect();
    assert_eq!(offsets, [0, 2, 4]);
    assert_eq!(out.chunks[0].logits, [97.0, 98.0, 99.0]);
    assert_eq!(out.chunks[2].logits, [101.0, 102.0]);
}

#[test]
fn chunk_seam_batch_sub_batches_across_text_boundaries_when_total_exceeds_max() {
    // 3 chunks across 2 texts, max_batch=2 → sub-batches [2, 1], where the
    // first batch spans BOTH texts. Grouping is still re-split per text.
    let per_text = vec![vec![text_chunk(&[10], 0)], vec![text_chunk(&[20], 0), text_chunk(&[30], 5)]];
    let mut p = seam(BatchSpy::new(2));
    let out = p.embed_chunks_batch(&per_text, ()).unwrap();
    assert_eq!(out.results.len(), 2);
    assert_eq!(out.results[0].chunks.len(), 1);
    assert_eq!(out.results[1].chunks.len(), 2);
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
    let mut p = seam(BatchSpy::new(2));
    let out = p.embed_chunks(&chunks, ()).unwrap();
    assert_eq!(p.embedder_mut().batches, vec![vec![4, 3], vec![2, 1]]);
    let values: Vec<&[f32]> = out.chunks.iter().map(|c| c.values.as_slice()).collect();
    assert_eq!(values, [&[1.0][..], &[2.0; 3], &[3.0; 2], &[4.0; 4]]);
}

#[test]
fn zero_capacity_sub_batches_one_chunk_at_a_time() {
    let mut p = seam(BatchSpy::new(0));
    p.embed_chunks(&[text_chunk(&[1], 0), text_chunk(&[2], 0)], ()).unwrap();
    assert_eq!(p.embedder_mut().sizes(), [1, 1]);
}

#[test]
fn classify_chunks_batch_preserves_grouping() {
    let per_text = vec![vec![text_chunk(&[1, 2], 0)], vec![text_chunk(&[3], 0), text_chunk(&[4, 5, 6], 9)]];
    let mut p = seam(StubClassify { num_labels: 8, max_batch: 8 });
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
    // Per-token byte offsets and the special-token mask come from the chunk,
    // the logit grid and num_labels from the head.
    let chunks = vec![TextChunk { encoding: enc_with_specials(&[4, 5]), byte_offset: 0, first_token: 0 }];
    let mut p = seam(StubRecognize { num_labels: 2, max_batch: 1 });
    let out = p.classify_tokens_chunks(&chunks, ()).unwrap();
    assert_eq!(out.chunks.len(), 1);
    let c = &out.chunks[0];
    assert_eq!(c.num_labels, 2);
    assert_eq!(c.logits, vec![2.0, 2.0, 4.0, 4.0, 5.0, 5.0, 3.0, 3.0]);
    assert_eq!(c.token_offsets, vec![(0, 0), (0, 1), (1, 2), (2, 2)]);
    assert_eq!(c.special_tokens_mask, vec![1, 0, 0, 1]);
}

#[test]
fn classify_tokens_chunks_batch_preserves_grouping() {
    let per_text = vec![vec![text_chunk(&[1, 2], 0)], vec![text_chunk(&[3], 0)]];
    let mut p = seam(StubRecognize { num_labels: 1, max_batch: 8 });
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
    // The seam skips tokenize/chunk, so the profile holds only the head's stage.
    // max_batch=1 runs the 3 chunks as 3 sub-batches, whose 1 ms stages merge
    // into one 3 ms stage.
    let chunks = shared_chunks();
    let per_text = vec![chunks[..1].to_vec(), chunks[1..].to_vec()];
    let profiled = RunOptions { profile: true };
    let stages = |prof: Option<RunProfile>| {
        prof.expect("profile collected").stages.into_iter().map(|s| (s.name, s.wall)).collect::<Vec<_>>()
    };
    let three_ms = |name: &str| vec![(name.to_string(), Duration::from_millis(3))];

    let mut p = seam(StubEmbed { hidden_size: 3, max_batch: 1, error: false });
    assert_eq!(stages(p.embed_chunks(&chunks, profiled).unwrap().profile), three_ms("encode"));
    assert_eq!(stages(p.embed_chunks_batch(&per_text, profiled).unwrap().profile), three_ms("encode"));

    let mut p = seam(StubClassify { num_labels: 3, max_batch: 1 });
    assert_eq!(stages(p.classify_chunks(&chunks, profiled).unwrap().profile), three_ms("classify"));
    assert_eq!(stages(p.classify_chunks_batch(&per_text, profiled).unwrap().profile), three_ms("classify"));

    let mut p = seam(StubRecognize { num_labels: 1, max_batch: 1 });
    assert_eq!(stages(p.classify_tokens_chunks(&chunks, profiled).unwrap().profile), three_ms("classify_tokens"));
    let batch = p.classify_tokens_chunks_batch(&per_text, profiled).unwrap();
    assert_eq!(stages(batch.profile), three_ms("classify_tokens"));
}

// ─── softmax / argmax ───────────────────────────────────────────────────────

#[test_case(&[1.0, 3.0, 2.0], 1; "max")]
#[test_case(&[], 0; "empty")]
#[test_case(&[5.0, f32::NAN, 1.0], 0; "NaN after the max")]
#[test_case(&[f32::NAN, 5.0, 1.0], 1; "NaN first")]
#[test_case(&[f32::NAN, f32::NAN], 0; "all NaN")]
fn argmax_returns_max_index(row: &[f32], expected: usize) {
    assert_eq!(argmax(row), expected);
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
        first_token: 0,
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

/// `tl` builds a content TokenLabel for the edge-case tests.
fn tl(label: &str, start: usize, end: usize, idx: usize, score: f32) -> TokenLabel<'_> {
    TokenLabel { label_id: 0, label, start, end, token_index: idx, score }
}

#[test]
fn labels_for_tokens_argmaxes_skips_specials_and_pairs_offsets() {
    // A leading special ([CLS]) puts raw token indices one ahead of content
    // indices: spans follow the raw index, token_index the content one.
    let chunk = token_chunk(
        &[(1, (0, 0), true), (1, (0, 3), false), (2, (3, 7), false), (0, (7, 8), false), (1, (8, 12), false)],
        5,
    );
    let labels = labels_for_tokens(&chunk, ner_id2label);
    let expect = |label_id: u32, start, end, token_index| {
        let mut row = [0.0; 5];
        row[label_id as usize] = 1.0;
        let score = softmax(&row)[label_id as usize];
        TokenLabel { label_id, label: ner_id2label(label_id), start, end, token_index, score }
    };
    assert_eq!(labels, [expect(1, 0, 3, 0), expect(2, 3, 7, 1), expect(0, 7, 8, 2), expect(1, 8, 12, 3)]);
}

/// Token indices count the text's content tokens from the chunk's first; an
/// interior special keeps its place, though it decodes to nothing.
#[test]
fn labels_for_tokens_index_from_the_chunk_start() {
    let spec = [(0, (0, 0), true), (1, (5, 6), false), (0, (6, 6), true), (2, (7, 8), false), (0, (8, 8), true)];
    let chunk = ChunkTokenClassification { first_token: 5, ..token_chunk(&spec, 5) };
    let indices: Vec<usize> = labels_for_tokens(&chunk, ner_id2label).iter().map(|t| t.token_index).collect();
    assert_eq!(indices, [5, 7]);
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

#[test]
fn group_spans_bio_extends_multi_i_then_breaks_on_o() {
    // B-PER I-PER I-PER I-PER O → one 4-token PER span, then O ends it. The
    // scores are not monotone, so the span keeps the min, not the latest.
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

/// A span's range follows its members' token indices — a chunk from the middle
/// of a text keeps the text's numbering.
#[test]
fn group_spans_ranges_follow_token_indices() {
    let spans = group_spans(&[tl("B-PER", 10, 13, 7, 1.0), tl("I-PER", 13, 16, 8, 1.0)], Scheme::Bio);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].token_range, 7..9);
}

#[test_case("NOUN", Scheme::Bio; "bare label")]
#[test_case("L-PER", Scheme::Bio; "L outside BILOU")]
#[test_case("U-PER", Scheme::Iobes; "U outside BILOU")]
#[test_case("E-PER", Scheme::Bilou; "E outside IOBES")]
#[test_case("S-PER", Scheme::Bio; "S outside IOBES")]
fn group_spans_unrecognized_tag_opens_a_span_per_token(label: &str, scheme: Scheme) {
    // An unknown or scheme-mismatched tag is a B- with the whole label as type.
    let spans = group_spans(&[tl(label, 0, 1, 0, 1.0), tl(label, 1, 2, 1, 1.0)], scheme);
    let got: Vec<_> = spans.iter().map(|e| (e.label.as_str(), e.token_range.clone())).collect();
    assert_eq!(got, [(label, 0..1), (label, 1..2)]);
}

// ─── group_spans_document (cross-chunk merge) ────────────────────────────────

#[test]
fn group_spans_document_merges_overlap_and_boundary_split() {
    // Windows A and B overlap on two tokens (1 and 2). Flattened, B's copy of
    // the overlap follows A's last token, so only sorting pairs the duplicates
    // up. On token 2 they disagree: B's confident I-PER must beat A's O, or the
    // entity splits. B's trailing O closes the span.
    let window_a = vec![tl("B-PER", 0, 3, 0, 0.9), tl("I-PER", 3, 7, 1, 0.8), tl("O", 7, 10, 2, 0.4)];
    let window_b = vec![
        tl("I-PER", 3, 7, 1, 0.6),
        tl("I-PER", 7, 10, 2, 0.85),
        tl("I-PER", 10, 14, 3, 0.75),
        tl("O", 14, 15, 4, 1.0),
    ];
    let spans = group_spans_document(&[window_a, window_b], Scheme::Bio);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].label, "PER");
    assert_eq!((spans[0].start, spans[0].end), (0, 14), "merged across the boundary");
    assert_eq!(spans[0].token_range, 0..4, "4 unique tokens after dedup");
    assert_eq!(spans[0].score, 0.75);
}

#[test]
fn group_spans_document_non_overlapping_equivalent_to_concat() {
    // Adjacent (non-overlapping) chunks: document grouping == concatenation,
    // with token ranges counted across the whole document.
    let chunk0 = vec![tl("B-PER", 0, 3, 0, 1.0), tl("I-PER", 3, 6, 1, 1.0)];
    let chunk1 = vec![tl("O", 6, 7, 2, 1.0), tl("B-LOC", 7, 10, 3, 1.0)];
    let spans = group_spans_document(&[chunk0, chunk1], Scheme::Bio);
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].label, "PER");
    assert_eq!((spans[0].start, spans[0].end), (0, 6));
    assert_eq!(spans[0].token_range, 0..2);
    assert_eq!(spans[1].label, "LOC");
    assert_eq!((spans[1].start, spans[1].end), (7, 10));
    assert_eq!(spans[1].token_range, 3..4);
}

/// Distinct tokens sharing a byte span (a lone `▁` and the emoji after it)
/// both survive the merge: windows meet on token indices.
#[test]
fn group_spans_document_keeps_tokens_that_share_a_span() {
    let window_a = vec![tl("B-PER", 0, 4, 0, 0.9), tl("I-PER", 0, 4, 1, 0.8)];
    let window_b = vec![tl("I-PER", 0, 4, 1, 0.7), tl("O", 5, 6, 2, 1.0)];
    let spans = group_spans_document(&[window_a, window_b], Scheme::Bio);
    let got: Vec<_> = spans.iter().map(|e| (e.token_range.clone(), e.score)).collect();
    assert_eq!(got, [(0..2, 0.8)]);
}

// ─── token_scores_document (sigmoid scores) ─────────────────────────────────

/// A window of `rows` content-token logits from token `first_token` on,
/// between a leading and a trailing special that score high (and must be
/// skipped). Token `i` spans bytes `(i, i + 1)`.
fn scored_chunk(first_token: usize, rows: &[&[f32]]) -> ChunkTokenClassification {
    let num_labels = rows[0].len();
    let special_row = vec![9.0; num_labels];
    let logits = [&special_row[..]].into_iter().chain(rows.iter().copied()).chain([&special_row[..]]).flatten();
    let content = (first_token..first_token + rows.len()).map(|i| (i, i + 1));
    let mut special_tokens_mask = vec![0; rows.len() + 2];
    (special_tokens_mask[0], special_tokens_mask[rows.len() + 1]) = (1, 1);
    ChunkTokenClassification {
        byte_offset: first_token,
        first_token,
        logits: logits.copied().collect(),
        num_labels,
        token_offsets: [(0, 0)].into_iter().chain(content).chain([(0, 0)]).collect(),
        special_tokens_mask,
    }
}

/// `(token_index, span, probs)` per scored token.
fn score_rows(scores: &TokenScores) -> Vec<(usize, (usize, usize), Vec<f32>)> {
    scores.iter().map(|t| (t.token_index, (t.start, t.end), t.probs.to_vec())).collect()
}

#[test]
fn token_scores_take_a_sigmoid_per_content_token() {
    let scores = token_scores_document(&[scored_chunk(0, &[&[2.0], &[-1.0]])], WindowWeighting::Uniform);
    assert_eq!(scores.num_labels(), 1);
    assert_eq!(score_rows(&scores), [(0, (0, 1), vec![sigmoid(2.0)]), (1, (1, 2), vec![sigmoid(-1.0)])]);
}

/// Labels score independently: two labels may both be likely.
#[test]
fn token_scores_score_each_label_on_its_own() {
    let scores = token_scores_document(&[scored_chunk(0, &[&[1.0, 1.0]])], WindowWeighting::Uniform);
    assert_eq!(score_rows(&scores), [(0, (0, 1), vec![sigmoid(1.0); 2])]);
}

/// Overlapping windows average a token's logits, then take the sigmoid: token
/// 1 gets sigmoid((2 - 4) / 2), not the mean of sigmoid(2) and sigmoid(-4).
/// A token one window holds keeps its own logit.
#[test]
fn token_scores_average_logits_across_windows() {
    let windows = [scored_chunk(0, &[&[0.5], &[2.0]]), scored_chunk(1, &[&[-4.0], &[1.5]])];
    let scores = token_scores_document(&windows, WindowWeighting::Uniform);
    let want = [(0, (0, 1), vec![sigmoid(0.5)]), (1, (1, 2), vec![sigmoid(-1.0)]), (2, (2, 3), vec![sigmoid(1.5)])];
    assert_eq!(score_rows(&scores), want);
}

/// Under `Hat`, token 1 sits mid-window in the first window (weight 1) and at
/// the edge of the second (weight 1/3 of three): the mean leans to the first
/// window's logit, (2·1 - 2·⅓) / (1 + ⅓) = 1.
#[test_case(WindowWeighting::Uniform => 0.0; "uniform")]
#[test_case(WindowWeighting::Hat => 1.0; "hat")]
fn token_scores_weigh_window_edges(weighting: WindowWeighting) -> f32 {
    let windows = [scored_chunk(0, &[&[0.0], &[2.0], &[0.0]]), scored_chunk(1, &[&[-2.0], &[0.0], &[0.0]])];
    let p = token_scores_document(&windows, weighting).iter().nth(1).expect("token 1").probs[0];
    // Back to the averaged logit, rounded past the f32 noise of the weights.
    ((p / (1.0 - p)).ln() * 1e4).round() / 1e4
}

/// Distinct tokens sharing a byte span stay distinct rows.
#[test]
fn token_scores_keep_tokens_that_share_a_span() {
    let (mut a, mut b) = (scored_chunk(0, &[&[1.0], &[2.0]]), scored_chunk(1, &[&[2.0], &[3.0]]));
    (a.token_offsets[1], a.token_offsets[2], b.token_offsets[1]) = ((0, 4), (0, 4), (0, 4));
    let rows = score_rows(&token_scores_document(&[a, b], WindowWeighting::Uniform));
    let want = [(0, (0, 4), vec![sigmoid(1.0)]), (1, (0, 4), vec![sigmoid(2.0)]), (2, (2, 3), vec![sigmoid(3.0)])];
    assert_eq!(rows, want);
}

#[test]
fn token_scores_of_no_windows_are_empty() {
    assert!(token_scores_document(&[], WindowWeighting::Hat).is_empty());
}

#[test]
#[should_panic(expected = "every window of a text scores the same labels")]
fn token_scores_refuse_windows_with_different_labels() {
    token_scores_document(&[scored_chunk(0, &[&[1.0]]), scored_chunk(1, &[&[1.0, 2.0]])], WindowWeighting::Uniform);
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
        // Non-monotone scores, so a span's min is not just its first or last member.
        let tokens: Vec<TokenLabel> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| TokenLabel {
                label_id: 0,
                label: l.as_str(),
                start: i,
                end: i + 1,
                token_index: i,
                score: ((i * 7) % 10) as f32 / 10.0 + 0.05,
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
