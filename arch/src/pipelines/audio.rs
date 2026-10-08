//! Composable long-form ASR pipeline: split → transcribe → crop → stitch.
//!
//! The orchestration is host-side and model-agnostic. A model implements only
//! its irreducible part — a [`Vad`] produces per-frame probabilities, a
//! [`Transcriber`] turns one decode-window of audio into text + word times —
//! and the heavy machinery (chunking, decode-window geometry, core-crop, and
//! stitching) lives in trait defaults here.
//!
//! ```text
//! Vad::probs ─▶ chunks_from_probs ─▶ [AudioChunk]          (VadSplitter)
//!                                        │
//!                                        ▼
//!   Transcriber::transcribe_windows(decode windows) ─▶ [Transcript]
//!                                        │  crop to core + stitch (default)
//!                                        ▼
//!                                  Transcription
//! ```
//!
//! All audio crosses the boundary as host `&[f32]` (see the module rationale):
//! decode windows are zero-copy sub-slices of the waveform, and the crate stays
//! free of the Tensor/device stack. The model owns audio → mel → device tensor
//! internally.

use std::ops::Range;
use std::time::Instant;

use snafu::{ResultExt, Snafu};
use svod_ir::origin::OriginScope;

pub use svod_runtime::RunProfile;
use svod_runtime::StageProfile;

pub use crate::rnnt::{Segment, Word};
use crate::vad::{AudioChunk, ChunkerOpts, chunks_from_probs, strict_chunk_sample_bound};

// ─── Results ────────────────────────────────────────────────────────────────

/// One decode-window's transcription, with word times **relative to the window
/// start** (`0.0` == `decode_start`). Returned by a [`Transcriber`]; the
/// pipeline crops these to the chunk's core before emitting.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transcript {
    /// Uncropped text decoded for this window.
    pub text: String,
    /// Word timings relative to the decode-window start.
    pub words: Vec<Word>,
    /// Phrase-level segments from timestamp-token splitting (Whisper). Empty for
    /// models that don't produce segments. Like `words`, times are
    /// window-relative.
    pub segments: Vec<Segment>,
    /// Detected/source language code (e.g. `"ru"`) when the transcriber resolves
    /// one, `None` otherwise. Populated by models that run language detection
    /// (Whisper); left empty by models that don't.
    pub language: Option<String>,
    /// Seconds of the window the decode consumed, for a model that opts into
    /// [`WindowAdvance`]: the read head advances by this. `None` falls back to
    /// the end of the last segment.
    pub consumed_sec: Option<f32>,
}

/// One speech region's final transcript. `start_sec`/`end_sec` reference the
/// original audio; `words` (when present) are core-relative — add `start_sec`
/// for an absolute timeline.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChunkResult {
    pub start_sec: f32,
    pub end_sec: f32,
    pub text: String,
    pub words: Option<Vec<Word>>,
    /// Phrase-level segments (when `RunOptions::segments` is set), cropped to
    /// the chunk's core. Times are core-relative (add `start_sec` for absolute).
    pub segments: Option<Vec<Segment>>,
}

/// Aggregated pipeline output: chunk texts joined by single spaces (empties
/// dropped), the per-chunk results, and the optional per-stage [`RunProfile`]
/// the transcriber collected. Profile stages are free-form and extensible, so
/// any model or caller can add custom ones.
#[derive(Debug, Default)]
pub struct Transcription {
    pub text: String,
    pub chunks: Vec<ChunkResult>,
    pub profile: Option<RunProfile>,
}

impl Transcription {
    /// Stitch per-chunk results: texts joined by single spaces, empties dropped.
    pub fn from_chunks(chunks: Vec<ChunkResult>, profile: Option<RunProfile>) -> Self {
        let text = chunks.iter().map(|c| c.text.as_str()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");
        Self { text, chunks, profile }
    }
}

/// Per-call run switches, orthogonal to a [`Transcriber`]'s construction config
/// (sizing, decoder choice). All default to `false`, so one built [`Asr`]
/// serves profiled/unprofiled, words-on/off, and segments-on/off runs without
/// rebuilding.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunOptions {
    /// Surface per-word timestamps on [`ChunkResult::words`]. The core-crop runs
    /// regardless (it owns each chunk's text); this only decides whether the
    /// cropped words are returned.
    pub words: bool,
    /// Surface phrase-level [`Segment`]s on [`ChunkResult::segments`]. Like
    /// `words`, the split runs regardless; this only decides whether the cropped
    /// segments are returned.
    pub segments: bool,
    /// Collect a per-stage [`RunProfile`] on [`Transcription::profile`].
    pub profile: bool,
}

/// Advance a model's decode window by what it actually decoded, not by a fixed
/// stride.
///
/// Whisper ends a window at the last timestamp it emitted, which routinely lands
/// well inside it. The reference implementation therefore advances its read head
/// to that timestamp (`seek += seek_delta`) and decodes the remainder next, so
/// every sample is decoded exactly once. Striding a fixed window instead leaves
/// the tail untranscribed; re-decoding a whole window to recover that tail
/// transcribes the overlap twice, which costs about half as many windows again.
///
/// Opting in via [`Transcriber::window_advance`] walks each region seek-style.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowAdvance {
    /// A shortfall shorter than this (seconds) counts as covering the window —
    /// trailing silence rather than dropped speech.
    pub min_gap_sec: f32,
    /// Floor on one advance, as a fraction of the window. A decode that reaches
    /// almost nothing is a failed decode, not a reason to inch forward; without
    /// a floor it would take a window per second of audio.
    pub min_advance_fraction: f32,
}

impl Default for WindowAdvance {
    fn default() -> Self {
        Self { min_gap_sec: 1.0, min_advance_fraction: 0.5 }
    }
}

impl WindowAdvance {
    /// Seconds to advance after a window of `window_sec` reached `covered_sec`.
    fn advance_sec(&self, covered_sec: f32, window_sec: f32) -> f32 {
        if covered_sec <= 0.0 || covered_sec >= window_sec - self.min_gap_sec {
            return window_sec;
        }
        covered_sec.max(window_sec * self.min_advance_fraction).min(window_sec)
    }
}

// ─── Word crop / stitch (pure host machinery) ────────────────────────────────

/// Crop decoded words back to a chunk's core and drop the rest.
///
/// Word times are relative to the decode-window start; the core begins
/// `core_offset_sec` into the window and spans `core_duration` seconds. A word
/// is kept iff its midpoint falls inside the core — so a word produced in the
/// pad/pre-roll context (or duplicated in two adjacent decode windows) survives
/// in at most one chunk. Survivors are re-based to core-relative time.
pub fn crop_words_to_core(words: Vec<Word>, core_offset_sec: f32, core_duration: f32) -> Vec<Word> {
    words
        .into_iter()
        .filter_map(|mut w| {
            let rel_start = w.start - core_offset_sec;
            let rel_end = w.end - core_offset_sec;
            let mid = 0.5 * (rel_start + rel_end);
            if !(0.0..core_duration).contains(&mid) {
                return None;
            }
            w.start = rel_start.clamp(0.0, core_duration);
            w.end = rel_end.clamp(w.start, core_duration);
            Some(w)
        })
        .collect()
}

/// Crop decoded segments back to a chunk's core and drop the rest. Same
/// midpoint-keep logic as [`crop_words_to_core`], applied to [`Segment`]s.
pub fn crop_segments_to_core(segments: Vec<Segment>, core_offset_sec: f32, core_duration: f32) -> Vec<Segment> {
    segments
        .into_iter()
        .filter_map(|mut s| {
            let rel_start = s.start - core_offset_sec;
            let rel_end = s.end - core_offset_sec;
            let mid = 0.5 * (rel_start + rel_end);
            if !(0.0..core_duration).contains(&mid) {
                return None;
            }
            s.start = rel_start.clamp(0.0, core_duration);
            s.end = rel_end.clamp(s.start, core_duration);
            Some(s)
        })
        .collect()
}

/// Concatenate exact word fragments and trim only the complete text boundary.
/// Whitespace-only fragments are ignored; meaningful internal whitespace stays
/// exactly where the producing tokenizer placed it.
pub fn words_to_text(words: &[Word]) -> String {
    let mut text = String::new();
    for word in words {
        if !word.text.trim().is_empty() {
            text.push_str(&word.text);
        }
    }
    text.trim().to_string()
}

// ─── VAD ─────────────────────────────────────────────────────────────────────

/// A frame-level voice-activity detector: audio → per-frame speech
/// probabilities. Implement [`probs`](Vad::probs) (the primary op — long-form
/// runs over a single waveform); [`probs_batch`](Vad::probs_batch) defaults to
/// looping it.
pub trait Vad {
    type Error: std::error::Error + 'static;

    /// Input samples covered by one probability (the VAD frame stride).
    fn samples_per_prob(&self) -> usize;

    /// Per-frame speech probabilities for one waveform.
    fn probs(&mut self, waveform: &[f32]) -> Result<Vec<f32>, Self::Error>;

    /// Probabilities for several waveforms. Defaults to looping [`Self::probs`];
    /// override when a VAD can batch whole clips for throughput (e.g. tuning
    /// sweeps).
    fn probs_batch(&mut self, waveforms: &[&[f32]]) -> Result<Vec<Vec<f32>>, Self::Error> {
        waveforms.iter().map(|w| self.probs(w)).collect()
    }
}

// ─── Splitter (chunk source) ──────────────────────────────────────────────────

/// Turns a waveform into ordered, bounded [`AudioChunk`]s (core + decode
/// window). The VAD-driven [`VadSplitter`] and the no-VAD [`FixedLengthSplitter`]
/// both implement it; [`Asr`] is generic over it.
pub trait Splitter {
    type Error: std::error::Error + 'static;

    fn split(&mut self, waveform: &[f32]) -> Result<Vec<AudioChunk>, Self::Error>;

    /// Upper bound (in samples) on the longest chunk this splitter can emit.
    /// [`Asr::assemble`] passes it to the transcriber builder so the model can
    /// size its buffers to the splitter rather than the caller threading a magic
    /// number. Defaults to `usize::MAX` (unbounded — the transcriber clamps to
    /// its own hard capacity); override when the chunk length is bounded by
    /// config.
    fn max_chunk_samples(&self) -> usize {
        usize::MAX
    }

    /// Stage name for this splitter's wall in a profiled run (e.g. `"vad"`).
    /// [`Asr::transcribe`] times `split` and records it under this label *when
    /// that call requests a profile* — profiling is a per-call choice, not baked
    /// into the splitter. Defaults to `"split"`.
    fn profile_label(&self) -> &'static str {
        "split"
    }
}

/// VAD-driven splitter: `vad.probs(wav)` → [`chunks_from_probs`] with `opts`.
/// The chunker config (sample rate, `align_to`, pad, pre-roll, durations) is
/// baked in at assembly — typically from a [`Transcriber`]'s primitive bounds.
pub struct VadSplitter<V: Vad> {
    vad: V,
    opts: ChunkerOpts,
}

impl<V: Vad> VadSplitter<V> {
    pub fn new(vad: V, opts: ChunkerOpts) -> Self {
        Self { vad, opts }
    }
}

#[derive(Debug, Snafu)]
pub enum VadSplitError<E: std::error::Error + 'static> {
    #[snafu(display("running VAD: {source}"))]
    Probs { source: E },
    #[snafu(display("chunking: {source}"))]
    Chunk { source: crate::vad::Error },
}

impl<V: Vad> Splitter for VadSplitter<V> {
    type Error = VadSplitError<V::Error>;

    /// Upper bound (in samples) on the longest chunk under the baked
    /// [`ChunkerOpts`] — the same math the chunker uses internally (see
    /// [`strict_chunk_sample_bound`]). When a soft `target_duration` engages,
    /// chunks are re-split at `1.5·target` (≤ max), so the ceiling is that, not
    /// `strict_limit` — letting the transcriber size to the real chunk length
    /// instead of padding every small target chunk up to `strict_limit` (which
    /// otherwise inflates RTF on long audio).
    fn max_chunk_samples(&self) -> usize {
        let o = &self.opts;
        let probs_per_sec = o.sample_rate as f32 / o.samples_per_prob.max(1) as f32;
        let strict_limit_probs = (o.strict_limit_duration * probs_per_sec).ceil() as usize;
        let max_probs = (o.max_duration * probs_per_sec).ceil() as usize;
        // Mirror chunks_from_probs's split_cap exactly: a target engages when
        // 0 < target_probs < max_probs, and re-splits at (1.5·target).min(max),
        // so the bound matches the real ceiling (never under-sizes the JIT).
        let cap_probs = o
            .target_duration
            .map(|d| (d * probs_per_sec).ceil() as usize)
            .filter(|&t| t > 0 && t < max_probs)
            .map(|t| (t + t / 2).min(max_probs))
            .unwrap_or(strict_limit_probs);
        let radius = o.trough_search_probs.unwrap_or(o.min_silence_probs);
        strict_chunk_sample_bound(cap_probs, radius, o.samples_per_prob, o.pad_samples, o.align_to)
    }

    fn profile_label(&self) -> &'static str {
        "vad"
    }

    fn split(&mut self, waveform: &[f32]) -> Result<Vec<AudioChunk>, Self::Error> {
        // The stage name the profiler groups by is also the root of every
        // origin path the VAD graph mints, so its kernels roll up under the
        // same label the pipeline records the split wall under.
        let probs = {
            let _origin = OriginScope::label(self.profile_label());
            self.vad.probs(waveform).context(ProbsSnafu)?
        };
        // The chunker clamps chunk ends to the real audio (the final VAD window
        // is zero-padded, so the prob grid overshoots the waveform). The length
        // is only known here, so set it per call over the baked sentinel.
        let mut opts = self.opts.clone();
        opts.max_total_samples = Some(waveform.len());
        chunks_from_probs(&probs, &opts).context(ChunkSnafu)
    }
}

/// No-VAD splitter: fixed-length non-overlapping windows (decode == core).
/// Non-final chunks are aligned to `align_to`; the last keeps its tail.
pub struct FixedLengthSplitter {
    window_samples: usize,
    align_to: usize,
}

impl FixedLengthSplitter {
    pub fn new(window_samples: usize, align_to: usize) -> Self {
        Self { window_samples: window_samples.max(1), align_to: align_to.max(1) }
    }
}

impl Splitter for FixedLengthSplitter {
    type Error = std::convert::Infallible;

    /// At most `window_samples` per chunk, but a non-final chunk is widened to
    /// `align_to` when `align_to > window_samples` (the `span.max(align_to)`
    /// floor), so the ceiling is the larger of the two.
    fn max_chunk_samples(&self) -> usize {
        self.window_samples.max(self.align_to)
    }

    fn split(&mut self, waveform: &[f32]) -> Result<Vec<AudioChunk>, Self::Error> {
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < waveform.len() {
            let nominal_end = start.saturating_add(self.window_samples).min(waveform.len());
            let end = if nominal_end == waveform.len() {
                nominal_end
            } else {
                let span = ((nominal_end - start) / self.align_to) * self.align_to;
                start + span.max(self.align_to)
            };
            chunks.push(AudioChunk::new(start, end));
            start = end;
        }
        Ok(chunks)
    }
}

// ─── Transcriber (per-window model) ───────────────────────────────────────────

/// Transcribes a decode-window of audio → text + word times relative to the
/// window. Implement [`transcribe_windows`](Transcriber::transcribe_windows)
/// (batched, the model owns its batch geometry); the single-window method is a
/// sequential fallback. The pipeline machinery — decode-window slicing,
/// core-crop, and stitching — is the [`transcribe_chunks`](Transcriber::transcribe_chunks)
/// default and needs no model code.
pub trait Transcriber {
    type Error: std::error::Error + 'static;

    /// The model's audio sample rate (Hz). Used only to convert sample indices
    /// to seconds — the pipeline does **not** validate the input against it.
    /// Waveforms cross the boundary as rate-less `&[f32]`, so the caller is
    /// responsible for feeding audio at this rate (resample first otherwise).
    fn sample_rate(&self) -> u32;

    /// Transcribe every decode window (the model owns internal batching),
    /// returning uncropped per-window transcripts plus the per-stage
    /// [`RunProfile`] — populated only when `opts.profile` is set. `opts` carries the
    /// per-call switches, so one transcriber serves profiled/unprofiled and
    /// words-on/off runs. Defaults to
    /// looping [`Self::transcribe_window`] and **merging** its per-window profiles (via
    /// [`RunProfile::merge`]); override for a model that batches the encoder and
    /// profiles the batch as a whole.
    fn transcribe_windows(
        &mut self,
        windows: &[&[f32]],
        opts: RunOptions,
    ) -> Result<(Vec<Transcript>, Option<RunProfile>), Self::Error> {
        let mut transcripts = Vec::with_capacity(windows.len());
        let mut prof: Option<RunProfile> = None;
        for w in windows {
            let (transcript, stage) = self.transcribe_window(w, opts)?;
            transcripts.push(transcript);
            if let Some(stage) = stage {
                prof.get_or_insert_with(RunProfile::default).merge(stage);
            }
        }
        Ok((transcripts, prof))
    }

    /// Transcribe one window + its optional profile (sequential fallback).
    /// Implement this OR [`Self::transcribe_windows`].
    fn transcribe_window(
        &mut self,
        window: &[f32],
        opts: RunOptions,
    ) -> Result<(Transcript, Option<RunProfile>), Self::Error> {
        let (mut transcripts, prof) = self.transcribe_windows(&[window], opts)?;
        Ok((transcripts.pop().unwrap_or_default(), prof))
    }

    /// Opt in to advancing by what each window decoded rather than by the
    /// splitter's stride. `None` (the default) takes every window's transcript as
    /// covering its whole core. Models whose decode can end early — Whisper,
    /// which stops at its last emitted timestamp — return `Some`.
    fn window_advance(&self) -> Option<WindowAdvance> {
        None
    }

    /// Decode each chunk's window, crop its words back to the core, stitch, and
    /// carry the profile (per [`RunOptions`]). Pure host machinery over
    /// [`Self::transcribe_windows`]; models don't override.
    fn transcribe_chunks(
        &mut self,
        waveform: &[f32],
        chunks: &[AudioChunk],
        opts: RunOptions,
    ) -> Result<Transcription, Self::Error> {
        // No speech regions (silence/empty audio): skip the empty
        // `transcribe_windows` call entirely (some models index off the batch
        // count and would underflow on zero windows).
        if chunks.is_empty() {
            return Ok(Transcription::default());
        }
        let sample_rate = self.sample_rate() as f32;
        let Some(advance) = self.window_advance() else {
            return self.transcribe_windows_striding(waveform, chunks, opts);
        };

        // Adjacent chunks tile one stretch of audio, so the read head carries
        // across them: a splitter's boundary is where the *region* ends, not
        // where a window must. A gap between chunks (a VAD's silence) ends one.
        let mut results: Vec<ChunkResult> = Vec::with_capacity(chunks.len());
        let mut prof: Option<RunProfile> = None;
        for region in coalesce_regions(chunks, waveform.len()) {
            let mut seek = region.start;
            while seek < region.end {
                let decode_end = seek.saturating_add(region.window_len).min(waveform.len());
                let (mut transcripts, window_profile) =
                    self.transcribe_windows(&[&waveform[seek..decode_end]], opts)?;
                if let Some(stage) = window_profile {
                    prof.get_or_insert_with(RunProfile::default).merge(stage);
                }
                let transcript = transcripts.pop().unwrap_or_default();

                let window_sec = (decode_end - seek) as f32 / sample_rate;
                let covered =
                    transcript.consumed_sec.unwrap_or_else(|| transcript.segments.last().map_or(0.0, |s| s.end));
                let step = advance.advance_sec(covered, window_sec);
                let step = (((step * sample_rate) as usize).max(1)).min(decode_end - seek);

                // The core owns text up to the advance; anything the window
                // decoded past it belongs to the window that starts there.
                let core_end = seek.saturating_add(step).min(region.end);
                let geom = ChunkWindow::new(
                    &AudioChunk::with_decode(seek, core_end, seek, decode_end),
                    waveform.len(),
                    sample_rate,
                );
                results.push(geom.finish(transcript, opts));
                seek = seek.saturating_add(step);
            }
        }

        Ok(Transcription::from_chunks(results, prof))
    }

    /// Decode every chunk once, at the splitter's own boundaries. The path for a
    /// model whose windows always cover their core.
    fn transcribe_windows_striding(
        &mut self,
        waveform: &[f32],
        chunks: &[AudioChunk],
        opts: RunOptions,
    ) -> Result<Transcription, Self::Error> {
        let sample_rate = self.sample_rate() as f32;
        let metas: Vec<ChunkWindow> =
            chunks.iter().map(|chunk| ChunkWindow::new(chunk, waveform.len(), sample_rate)).collect();
        let windows: Vec<&[f32]> = metas.iter().map(|m| &waveform[m.decode_range()]).collect();
        let (transcripts, prof) = self.transcribe_windows(&windows, opts)?;
        let results: Vec<ChunkResult> = transcripts.into_iter().zip(&metas).map(|(t, m)| m.finish(t, opts)).collect();
        Ok(Transcription::from_chunks(results, prof))
    }

    /// [`transcribe_chunks`](Self::transcribe_chunks) with default
    /// [`RunOptions`] (no words, segments, or profile).
    fn transcribe_chunks_default(
        &mut self,
        waveform: &[f32],
        chunks: &[AudioChunk],
    ) -> Result<Transcription, Self::Error> {
        self.transcribe_chunks(waveform, chunks, RunOptions::default())
    }
}

/// One stretch of audio a read head walks, and the window length to walk it with.
struct Region {
    start: usize,
    end: usize,
    window_len: usize,
}

/// Merge chunks that touch into one region. A splitter striding fixed windows
/// produces one region for the whole waveform; a VAD produces one per utterance.
fn coalesce_regions(chunks: &[AudioChunk], waveform_len: usize) -> Vec<Region> {
    let mut regions: Vec<Region> = Vec::new();
    for chunk in chunks {
        let start = chunk.start_sample.min(waveform_len);
        let end = chunk.end_sample.min(waveform_len);
        if end <= start {
            continue;
        }
        let window_len = chunk.decode_len().max(end - start);
        match regions.last_mut() {
            Some(last) if last.end == start => {
                last.end = end;
                last.window_len = last.window_len.max(window_len);
            }
            _ => regions.push(Region { start, end, window_len }),
        }
    }
    regions
}

/// Decode geometry for one chunk: which slice of the waveform to decode, and
/// how to crop that decode back to the chunk's core. Public so a caller that
/// pools windows from several waveforms into one
/// [`Transcriber::transcribe_windows`] batch can finish each chunk itself.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkWindow {
    decode_start: usize,
    decode_end: usize,
    start_sec: f32,
    end_sec: f32,
    core_offset_sec: f32,
}

impl ChunkWindow {
    pub fn new(chunk: &AudioChunk, waveform_len: usize, sample_rate: f32) -> Self {
        let decode_end = chunk.decode_end_sample.min(waveform_len);
        Self {
            decode_start: chunk.decode_start_sample.min(decode_end),
            decode_end,
            start_sec: chunk.start_sample as f32 / sample_rate,
            end_sec: chunk.end_sample.min(waveform_len) as f32 / sample_rate,
            core_offset_sec: chunk.start_sample.saturating_sub(chunk.decode_start_sample) as f32 / sample_rate,
        }
    }

    /// Sample range of the waveform to feed the transcriber.
    pub fn decode_range(&self) -> Range<usize> {
        self.decode_start..self.decode_end
    }

    fn core_duration(&self) -> f32 {
        self.end_sec - self.start_sec
    }

    /// Crop this window's transcript to the core and emit the chunk's result.
    pub fn finish(&self, transcript: Transcript, opts: RunOptions) -> ChunkResult {
        let core_duration = self.core_duration();
        let words = crop_words_to_core(transcript.words, self.core_offset_sec, core_duration);
        let segments = crop_segments_to_core(transcript.segments, self.core_offset_sec, core_duration);
        // Cropped fragments own the core text when available; otherwise retain
        // models that only provide complete-window text.
        let text = if words.is_empty() { transcript.text.trim().to_string() } else { words_to_text(&words) };
        ChunkResult {
            start_sec: self.start_sec,
            end_sec: self.end_sec,
            text,
            words: opts.words.then_some(words),
            segments: opts.segments.then_some(segments),
        }
    }
}

// ─── Asr (composer) ───────────────────────────────────────────────────────────

/// The full pipeline: a chunk source ([`Splitter`]) plus a per-window model
/// ([`Transcriber`]). `transcribe` runs `splitter.split` →
/// `transcriber.transcribe_chunks`. Build with [`assemble`](Asr::assemble) to
/// size the (eagerly-JIT-prepared) transcriber from the splitter's chunk
/// ceiling, or [`new`](Asr::new) to compose two already-built parts. The input
/// is rate-less `&[f32]`; feed audio at the transcriber's
/// [`sample_rate`](Transcriber::sample_rate) (no validation is performed).
pub struct Asr<S: Splitter, T: Transcriber> {
    splitter: S,
    transcriber: T,
}

#[derive(Debug, Snafu)]
pub enum AsrError<SE: std::error::Error + 'static, TE: std::error::Error + 'static> {
    #[snafu(display("splitting: {source}"))]
    Split { source: SE },
    #[snafu(display("transcribing: {source}"))]
    Transcribe { source: TE },
}

impl<S: Splitter, T: Transcriber> Asr<S, T> {
    pub fn new(splitter: S, transcriber: T) -> Self {
        Self { splitter, transcriber }
    }

    /// Build the transcriber eagerly, sized to the splitter's chunk ceiling
    /// ([`Splitter::max_chunk_samples`]), then compose. `build` runs the model's
    /// JIT prepare up front — there is no lazy/first-call cost — so the caller
    /// never hand-threads the buffer size between splitter and transcriber.
    pub fn assemble<E>(splitter: S, build: impl FnOnce(usize) -> Result<T, E>) -> Result<Self, E> {
        let transcriber = build(splitter.max_chunk_samples())?;
        Ok(Self::new(splitter, transcriber))
    }

    /// Split → transcribe → crop → stitch. [`RunOptions`] are per-call switches:
    /// the same `Asr` serves profiled/unprofiled, words-on/off, and
    /// segments-on/off runs without rebuilding. When `opts.profile` is set,
    /// the splitter's wall is timed and recorded under its
    /// [`profile_label`](Splitter::profile_label) ahead of the transcriber's
    /// stages.
    pub fn transcribe(
        &mut self,
        waveform: &[f32],
        opts: RunOptions,
    ) -> Result<Transcription, AsrError<S::Error, T::Error>> {
        let t = Instant::now();
        let chunks = self.splitter.split(waveform).context(SplitSnafu)?;
        let split_wall = t.elapsed();
        let mut transcription = self.transcriber.transcribe_chunks(waveform, &chunks, opts).context(TranscribeSnafu)?;
        if opts.profile {
            // Lead with the split (e.g. `vad`) stage, then the transcriber's.
            let mut p = RunProfile::default();
            p.push(StageProfile::host(self.splitter.profile_label(), split_wall));
            if let Some(rest) = transcription.profile.take() {
                p.merge(rest);
            }
            transcription.profile = Some(p);
        }
        Ok(transcription)
    }

    /// [`transcribe`](Self::transcribe) with default [`RunOptions`] (no words,
    /// segments, or profile) — the common case, without spelling out the struct.
    pub fn transcribe_default(&mut self, waveform: &[f32]) -> Result<Transcription, AsrError<S::Error, T::Error>> {
        self.transcribe(waveform, RunOptions::default())
    }

    pub fn splitter_mut(&mut self) -> &mut S {
        &mut self.splitter
    }

    pub fn transcriber_mut(&mut self) -> &mut T {
        &mut self.transcriber
    }
}
