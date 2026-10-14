use std::convert::Infallible;

use crate::pipelines::audio::{
    Asr, ChunkResult, FixedLengthSplitter, RunOptions, RunProfile, Splitter, Transcriber, Transcript, Vad, VadSplitter,
    WindowAdvance, crop_words_to_core, words_to_text,
};
use crate::rnnt::{Segment, Word};
use crate::vad::{AudioChunk, ChunkerOpts};

fn word(text: &str, start: f32, end: f32) -> Word {
    Word { text: text.to_string(), start, end }
}

// ─── Mocks ───────────────────────────────────────────────────────────────────

/// VAD that ignores the audio and returns a fixed probability vector.
struct MockVad {
    probs: Vec<f32>,
    samples_per_prob: usize,
}

impl Vad for MockVad {
    type Error = Infallible;
    fn samples_per_prob(&self) -> usize {
        self.samples_per_prob
    }
    fn probs(&mut self, _waveform: &[f32]) -> Result<Vec<f32>, Infallible> {
        Ok(self.probs.clone())
    }
}

/// Transcriber that returns a preset transcript per window (1 sample = 1 s).
struct PresetTranscriber {
    out: Vec<Transcript>,
}

impl Transcriber for PresetTranscriber {
    type Error = Infallible;
    fn sample_rate(&self) -> u32 {
        1
    }
    fn transcribe_windows(
        &mut self,
        windows: &[&[f32]],
        _opts: RunOptions,
    ) -> Result<(Vec<Transcript>, Option<RunProfile>), Infallible> {
        assert_eq!(windows.len(), self.out.len(), "preset length must match window count");
        Ok((self.out.clone(), None))
    }
}

fn fast_chunker_opts() -> ChunkerOpts {
    ChunkerOpts {
        sample_rate: 1,
        samples_per_prob: 1,
        threshold: 0.5,
        min_duration: 1.0,
        max_duration: 10.0,
        target_duration: None,
        strict_limit_duration: 15.0,
        min_speech_probs: 1,
        min_silence_probs: 1,
        merge_gap_probs: 0,
        trough_search_probs: None,
        trough_threshold: None,
        pad_samples: 0,
        preroll_samples: 0,
        align_to: 1,
        max_total_samples: None,
    }
}

// ─── crop / stitch ────────────────────────────────────────────────────────────

#[test]
fn crop_keeps_midpoints_inside_core_and_rebases() {
    // core starts 2s into the window, spans 5s (window time [2, 7)).
    let words = vec![word("a", 0.5, 1.5), word("b", 3.5, 4.5), word("c", 7.5, 8.5)];
    let cropped = crop_words_to_core(words, 2.0, 5.0);
    assert_eq!(cropped, vec![word("b", 1.5, 2.5)]);
}

#[test]
fn words_to_text_concatenates_fragments_and_drops_blanks() {
    let words =
        vec![word("  hello", 0.0, 0.1), word("", 0.1, 0.1), word(" \t", 0.1, 0.1), word("  world!  ", 0.2, 0.3)];
    assert_eq!(words_to_text(&words), "hello  world!");
    assert_eq!(words_to_text(&[]), "");
}

#[test]
fn cropped_fragments_reconstruct_without_spacing_heuristics() {
    let words = vec![word(" prefix", 0.0, 1.0), word(" 23", 2.0, 3.0), word("-м", 3.0, 4.0), word(" доме", 4.0, 5.0)];
    let cropped = crop_words_to_core(words, 1.5, 4.0);
    assert_eq!(words_to_text(&cropped), "23-м доме");
}

// ─── Transcriber::transcribe_chunks default (geometry + crop + stitch) ─────────

#[test]
fn transcribe_chunks_slices_decode_windows_crops_and_stitches() {
    let waveform = vec![0.0_f32; 50];
    let chunks = vec![AudioChunk::with_decode(10, 20, 5, 25), AudioChunk::with_decode(30, 40, 28, 42)];
    // Window A: only "a1" (mid 7, in core [5,15)) survives; "pre"/"post" are pad.
    // Window B: "b1" (mid 4, in core [2,12)) survives.
    let mut asr = PresetTranscriber {
        out: vec![
            Transcript {
                text: String::new(),
                words: vec![word("pre", 1.0, 3.0), word("a1", 6.0, 8.0), word("post", 16.0, 18.0)],
                ..Default::default()
            },
            Transcript { text: String::new(), words: vec![word("b1", 3.0, 5.0)], ..Default::default() },
        ],
    };
    let out = asr.transcribe_chunks(&waveform, &chunks, RunOptions { words: true, ..Default::default() }).unwrap();
    assert_eq!(out.text, "a1 b1");
    assert_eq!(
        out.chunks,
        vec![
            ChunkResult {
                start_sec: 10.0,
                end_sec: 20.0,
                text: "a1".to_string(),
                words: Some(vec![word("a1", 1.0, 3.0)]),
                ..Default::default()
            },
            ChunkResult {
                start_sec: 30.0,
                end_sec: 40.0,
                text: "b1".to_string(),
                words: Some(vec![word("b1", 1.0, 3.0)]),
                ..Default::default()
            },
        ]
    );
    assert!(out.profile.is_none());
}

#[test]
fn transcribe_chunks_empty_returns_default() {
    // Silence-only audio yields no chunks → empty Transcription, and
    // transcribe_windows is never called (no zero-window batch-math underflow).
    let mut t = PresetTranscriber { out: Vec::new() };
    let out = t.transcribe_chunks(&[0.0_f32; 10], &[], RunOptions { words: true, ..Default::default() }).unwrap();
    assert_eq!(out.text, "");
    assert!(out.chunks.is_empty());
    assert!(out.profile.is_none());
}

#[test]
fn transcribe_chunks_omits_words_when_not_wanted() {
    let waveform = vec![0.0_f32; 20];
    let chunks = vec![AudioChunk::new(0, 10)];
    let mut asr = PresetTranscriber {
        out: vec![Transcript { text: String::new(), words: vec![word("hi", 1.0, 2.0)], ..Default::default() }],
    };
    // default options → words: false, so cropped words are not surfaced.
    let out = asr.transcribe_chunks(&waveform, &chunks, ().into()).unwrap();
    assert_eq!(out.text, "hi");
    assert_eq!(out.chunks[0].words, None);
}

// ─── Splitters ────────────────────────────────────────────────────────────────

#[test]
fn vad_splitter_runs_probs_then_chunks() {
    let mut splitter = VadSplitter::new(MockVad { probs: vec![1.0; 4], samples_per_prob: 1 }, fast_chunker_opts());
    let chunks = splitter.split(&[0.0_f32; 4]).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 4)]);
}

#[test]
fn fixed_length_splitter_strides_aligned_with_unaligned_tail() {
    let mut splitter = FixedLengthSplitter::new(10, 4);
    let chunks = splitter.split(&[0.0_f32; 26]).unwrap();
    // window 10 floored to align 4 → 8; final tail keeps its remainder.
    assert_eq!(chunks, vec![AudioChunk::new(0, 8), AudioChunk::new(8, 16), AudioChunk::new(16, 26)]);
    assert_eq!(splitter.max_chunk_samples(), 10, "window ≥ align → window");
}

#[test]
fn fixed_length_splitter_empty_waveform() {
    let mut splitter = FixedLengthSplitter::new(10, 1);
    assert!(splitter.split(&[]).unwrap().is_empty());
}

#[test]
fn fixed_length_splitter_align_exceeds_window_overshoots_terminally() {
    // align > window: span floors to 0, widened to align. The overshoot chunk is
    // always terminal and ends past the waveform — transcribe_chunks clamps the
    // slice. max_chunk_samples is then the larger of the two (align).
    let mut splitter = FixedLengthSplitter::new(2, 16);
    assert_eq!(splitter.max_chunk_samples(), 16);
    let chunks = splitter.split(&[0.0_f32; 20]).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 16), AudioChunk::new(16, 32)]);
}

#[test]
fn vad_splitter_max_chunk_bound_and_label() {
    let splitter = VadSplitter::new(MockVad { probs: vec![1.0; 4], samples_per_prob: 1 }, fast_chunker_opts());
    // strict_chunk_sample_bound((15+2)*1 + 0 + 2*1) = 19 from fast_chunker_opts.
    assert_eq!(splitter.max_chunk_samples(), 19);
    assert_eq!(splitter.profile_label(), "vad");
}

// ─── Asr composer end-to-end ──────────────────────────────────────────────────

#[test]
fn asr_composes_split_and_transcribe() {
    // 3 fixed windows → 3 preset transcripts → stitched.
    let splitter = FixedLengthSplitter::new(10, 1);
    let transcriber = PresetTranscriber {
        out: vec![
            Transcript { text: String::new(), words: vec![word("one", 1.0, 2.0)], ..Default::default() },
            Transcript { text: String::new(), words: vec![word("two", 1.0, 2.0)], ..Default::default() },
            Transcript { text: String::new(), words: vec![word("three", 1.0, 2.0)], ..Default::default() },
        ],
    };
    let mut asr = Asr::new(splitter, transcriber);
    let out = asr.transcribe(&[0.0_f32; 25], ()).unwrap();
    assert_eq!(out.text, "one two three");
}

// ─── Profile merge (default transcribe_windows over a single-window model) ─────

/// Single-window transcriber emitting a 1 ms `decode` stage per window — the
/// default `transcribe_windows` must merge these, not drop them.
struct ProfilingTranscriber;

impl Transcriber for ProfilingTranscriber {
    type Error = Infallible;
    fn sample_rate(&self) -> u32 {
        1
    }
    fn transcribe_window(
        &mut self,
        _window: &[f32],
        opts: RunOptions,
    ) -> Result<(Transcript, Option<RunProfile>), Infallible> {
        // Emit the stage only when this call asked for a profile.
        let prof = opts.profile.then(|| {
            let mut p = RunProfile::default();
            p.push(svod_runtime::StageProfile::host("decode", std::time::Duration::from_millis(1)));
            p
        });
        Ok((Transcript { text: "x".to_string(), words: Vec::new(), ..Default::default() }, prof))
    }
}

#[test]
fn transcribe_windows_default_merges_per_window_profiles() {
    // Two windows, no batch override → the default loops transcribe_window and
    // merges both `decode` stages into one with summed wall (1 ms × 2).
    let waveform = vec![0.0_f32; 20];
    let chunks = vec![AudioChunk::new(0, 10), AudioChunk::new(10, 20)];
    let out = ProfilingTranscriber
        .transcribe_chunks(&waveform, &chunks, RunOptions { profile: true, ..Default::default() })
        .unwrap();
    let profile = out.profile.expect("profile collected");
    assert_eq!(profile.stages.len(), 1, "stages merged by name");
    assert_eq!(profile.stage("decode").unwrap().wall, std::time::Duration::from_millis(2));
}

#[test]
fn asr_profiles_per_call_without_rebuild() {
    // One built Asr serves both modes: a profiled call surfaces the splitter's
    // `vad` stage ahead of the transcriber's `decode`; an unprofiled call on the
    // same instance yields no profile (no rebuild to toggle).
    let splitter = VadSplitter::new(MockVad { probs: vec![1.0; 4], samples_per_prob: 1 }, fast_chunker_opts());
    let mut asr = Asr::new(splitter, ProfilingTranscriber);

    let profiled = asr.transcribe(&[0.0_f32; 4], RunOptions { profile: true, ..Default::default() }).unwrap();
    let profile = profiled.profile.expect("profile");
    let stages: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(stages, vec!["vad", "decode"], "VAD stage leads the merged profile");

    assert!(asr.transcribe(&[0.0_f32; 4], ()).unwrap().profile.is_none());
}

#[test]
fn asr_surfaces_split_stage_even_when_transcriber_does_not() {
    // Profiled run, but the transcriber emits no profile: the `vad` stage still
    // surfaces on its own.
    let splitter = VadSplitter::new(MockVad { probs: vec![1.0; 4], samples_per_prob: 1 }, fast_chunker_opts());
    let transcriber = PresetTranscriber {
        out: vec![Transcript { text: String::new(), words: vec![word("a", 1.0, 2.0)], ..Default::default() }],
    };
    let mut asr = Asr::new(splitter, transcriber);
    let out = asr.transcribe(&[0.0_f32; 4], RunOptions { profile: true, ..Default::default() }).unwrap();
    let profile = out.profile.expect("vad-only profile surfaces");
    let stages: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(stages, vec!["vad"]);
}

#[test]
fn assemble_sizes_transcriber_from_splitter_bound() {
    // Asr::assemble passes the splitter's max_chunk_samples into the eager
    // transcriber builder — no caller-threaded buffer size.
    let seen = std::cell::Cell::new(0usize);
    let splitter = FixedLengthSplitter::new(10, 4); // max_chunk = max(10, 4) = 10
    let _asr: Asr<_, PresetTranscriber> = Asr::assemble(splitter, |max_chunk| {
        seen.set(max_chunk);
        Ok::<_, Infallible>(PresetTranscriber { out: Vec::new() })
    })
    .unwrap();
    assert_eq!(seen.get(), 10);
}

// ─── Window advance ──────────────────────────────────────────────────────────

fn segment(text: &str, start: f32, end: f32) -> Segment {
    Segment { text: text.to_string(), start, end }
}

/// Answers one preset transcript per `transcribe_windows` call and records the
/// window lengths it was handed, so a test can drive and observe the seek walk.
/// 1 sample = 1 s.
struct WalkTranscriber {
    answers: std::collections::VecDeque<Transcript>,
    advance: Option<WindowAdvance>,
    window_lens: Vec<usize>,
}

impl WalkTranscriber {
    fn new(advance: Option<WindowAdvance>, answers: Vec<Transcript>) -> Self {
        Self { answers: answers.into(), advance, window_lens: Vec::new() }
    }
}

/// One window's decoded text, reaching `covered` seconds into it.
fn reached(text: &str, covered: f32) -> Transcript {
    Transcript { text: text.to_string(), segments: vec![segment(text, 0.0, covered)], ..Default::default() }
}

impl Transcriber for WalkTranscriber {
    type Error = Infallible;
    fn sample_rate(&self) -> u32 {
        1
    }
    fn window_advance(&self) -> Option<WindowAdvance> {
        self.advance
    }
    fn transcribe_windows(
        &mut self,
        windows: &[&[f32]],
        _opts: RunOptions,
    ) -> Result<(Vec<Transcript>, Option<RunProfile>), Infallible> {
        self.window_lens.extend(windows.iter().map(|w| w.len()));
        Ok((windows.iter().map(|_| self.answers.pop_front().unwrap_or_default()).collect(), None))
    }
}

#[test]
fn the_next_window_starts_where_the_last_decode_stopped() {
    let waveform = vec![0.0_f32; 60];
    // One fixed 30 s stride over 60 s: two chunks, one contiguous region.
    let chunks = vec![AudioChunk::new(0, 30), AudioChunk::new(30, 60)];
    let mut t = WalkTranscriber::new(
        Some(WindowAdvance::default()),
        vec![reached("a", 20.0), reached("b", 25.0), reached("c", 30.0)],
    );
    let out = t.transcribe_chunks(&waveform, &chunks, RunOptions::default()).unwrap();

    // Advances 20 then 25, so windows open at 0, 20, 45 - never re-decoding
    // ground the previous window already owned.
    assert_eq!(out.chunks.iter().map(|c| c.start_sec).collect::<Vec<_>>(), vec![0.0, 20.0, 45.0]);
    assert_eq!(out.text, "a b c");
    // The last window is clipped by the waveform, not by the stride.
    assert_eq!(t.window_lens, vec![30, 30, 15]);
}

/// A decode that consumed `consumed` seconds of its window while its segments
/// stop at `segment_end` — the two disagree on purpose.
fn consumed(text: &str, consumed: f32, segment_end: f32) -> Transcript {
    Transcript { consumed_sec: Some(consumed), ..reached(text, segment_end) }
}

#[test]
fn a_reported_consumption_outranks_the_last_segments_end() {
    let waveform = vec![0.0_f32; 60];
    let chunks = vec![AudioChunk::new(0, 30), AudioChunk::new(30, 60)];
    // Segments that stop at 5 s would advance by the floor (half a window);
    // the reported consumption walks 20 then 25 instead.
    let mut t = WalkTranscriber::new(
        Some(WindowAdvance::default()),
        vec![consumed("a", 20.0, 5.0), consumed("b", 25.0, 5.0), consumed("c", 30.0, 5.0)],
    );
    let out = t.transcribe_chunks(&waveform, &chunks, RunOptions::default()).unwrap();

    assert_eq!(out.chunks.iter().map(|c| c.start_sec).collect::<Vec<_>>(), vec![0.0, 20.0, 45.0]);
    assert_eq!(t.window_lens, vec![30, 30, 15]);
    assert_eq!(out.text, "a b c");
}

#[test]
fn a_window_that_covers_itself_advances_a_whole_stride() {
    let waveform = vec![0.0_f32; 60];
    let chunks = vec![AudioChunk::new(0, 30), AudioChunk::new(30, 60)];
    let mut t = WalkTranscriber::new(Some(WindowAdvance::default()), vec![reached("a", 29.5), reached("b", 30.0)]);
    let out = t.transcribe_chunks(&waveform, &chunks, RunOptions::default()).unwrap();
    assert_eq!(t.window_lens, vec![30, 30], "half a second short is trailing silence, not dropped speech");
    assert_eq!(out.chunks.iter().map(|c| c.start_sec).collect::<Vec<_>>(), vec![0.0, 30.0]);
}

#[test]
fn a_decode_that_reaches_almost_nothing_still_advances_half_a_window() {
    let waveform = vec![0.0_f32; 60];
    let chunks = vec![AudioChunk::new(0, 60)];
    // Reaching 1 s of 60 would otherwise mean sixty windows for sixty seconds.
    let mut t = WalkTranscriber::new(Some(WindowAdvance::default()), vec![reached("a", 1.0), reached("b", 30.0)]);
    t.transcribe_chunks(&waveform, &chunks, RunOptions::default()).unwrap();
    assert_eq!(t.window_lens.len(), 2, "the advance floor bounds windows at twice the minimum");
}

#[test]
fn a_gap_between_chunks_ends_a_region() {
    let waveform = vec![0.0_f32; 100];
    // A VAD's two utterances with silence between them: the read head must not
    // walk from the first into the silence.
    let chunks = vec![AudioChunk::new(0, 20), AudioChunk::new(60, 80)];
    let mut t = WalkTranscriber::new(Some(WindowAdvance::default()), vec![reached("a", 20.0), reached("b", 20.0)]);
    let out = t.transcribe_chunks(&waveform, &chunks, RunOptions::default()).unwrap();
    assert_eq!(out.chunks.iter().map(|c| c.start_sec).collect::<Vec<_>>(), vec![0.0, 60.0]);
}

#[test]
fn without_the_opt_in_chunks_are_decoded_at_the_splitters_boundaries() {
    let waveform = vec![0.0_f32; 60];
    let chunks = vec![AudioChunk::new(0, 30), AudioChunk::new(30, 60)];
    let mut t = WalkTranscriber::new(None, vec![reached("a", 10.0), reached("b", 10.0)]);
    let out = t.transcribe_chunks(&waveform, &chunks, RunOptions::default()).unwrap();
    assert_eq!(t.window_lens, vec![30, 30], "one batched pass, no seek walk");
    assert_eq!(out.chunks.iter().map(|c| c.start_sec).collect::<Vec<_>>(), vec![0.0, 30.0]);
}

// ─── Per-call switches reach the model ────────────────────────────────────────

/// Records the [`RunOptions`] each window call received.
#[derive(Default)]
struct OptsRecorder {
    seen: Vec<RunOptions>,
}

impl Transcriber for OptsRecorder {
    type Error = Infallible;
    fn sample_rate(&self) -> u32 {
        1
    }
    fn transcribe_windows(
        &mut self,
        windows: &[&[f32]],
        opts: RunOptions,
    ) -> Result<(Vec<Transcript>, Option<RunProfile>), Infallible> {
        self.seen.push(opts);
        Ok((windows.iter().map(|_| Transcript::default()).collect(), None))
    }
}

/// A model decides per call whether to do words-only work (Whisper's aligner),
/// so the pipeline hands it the caller's switches unchanged.
#[test]
fn window_calls_carry_the_run_options() {
    let waveform = vec![0.0_f32; 10];
    let mut asr = Asr::new(FixedLengthSplitter::new(5, 1), OptsRecorder::default());
    let opts = RunOptions { words: true, segments: false, profile: true };
    asr.transcribe(&waveform, opts).unwrap();
    let seen = &asr.transcriber_mut().seen;
    assert!(!seen.is_empty());
    assert!(seen.iter().all(|o| o.words && o.profile && !o.segments), "{seen:?}");
}
