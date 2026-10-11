use std::cmp::Ordering;
use std::convert::Infallible;

use proptest::prelude::*;

use crate::rnnt::{
    BatchBlockStep, BatchJointStep, BatchLabelStep, BlockTapes, JointStep, RnntDecoder, RnntOpts, TokenEmission, Word,
};

/// NaN-safe argmax — the device backend computes the joint argmax on-GPU and
/// returns the index; the mock mirrors that over its scripted logits. NaN is
/// treated as smaller than any number (first non-NaN wins; all-NaN → 0).
fn argmax_nan_safe(frame: &[f32]) -> usize {
    let cmp = |a: f32, b: f32| {
        a.partial_cmp(&b).unwrap_or_else(|| match (a.is_nan(), b.is_nan()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => Ordering::Equal,
        })
    };
    let mut best = 0usize;
    for i in 1..frame.len() {
        if cmp(frame[i], frame[best]).is_gt() {
            best = i;
        }
    }
    best
}

// ─── Mock backend ─────────────────────────────────────────────────────────

/// Returns a scripted sequence of logits frames, one per `step` call. The
/// search loop advances through the script in order.
///
/// `committed_token` records the `prev_token` parameter at each step. `commit`
/// increments `commits`; `reset` increments `resets`. The mock exposes enough
/// observability to assert every contract on `JointStep` from the search loop.
struct MockJointStep {
    /// Logits for each `step` call, in order.
    script: Vec<Vec<f32>>,
    /// Cursor into `script`.
    cursor: usize,
    /// `(encoder_frame_first_value, prev_token)` recorded at each `step`.
    /// Stores only the first encoder element so equality checks stay cheap.
    pub seen: Vec<(f32, Option<usize>)>,
    /// Mirror of the search loop's notion of committed prev_token. After
    /// `commit`, this latches whatever `prev_token` would be next: the
    /// search-loop observable `prev_token` for the *next* step.
    pub commits: usize,
    pub resets: usize,
}

impl MockJointStep {
    fn new(script: Vec<Vec<f32>>) -> Self {
        Self { script, cursor: 0, seen: Vec::new(), commits: 0, resets: 0 }
    }
}

impl JointStep for MockJointStep {
    type Error = Infallible;

    fn step(&mut self, encoder_frame: &[f32], prev_token: Option<usize>) -> Result<usize, Self::Error> {
        let frame_first = *encoder_frame.first().unwrap_or(&0.0);
        self.seen.push((frame_first, prev_token));
        let idx = self.cursor.min(self.script.len() - 1);
        self.cursor += 1;
        Ok(argmax_nan_safe(&self.script[idx]))
    }

    fn commit(&mut self) {
        self.commits += 1;
    }

    fn reset(&mut self) {
        self.resets += 1;
    }
}

/// Batched mock: each lane consumes its own token-id script in step order;
/// inactive lanes don't advance their cursor (mirrors a real backend whose
/// inactive-lane outputs are ignored). Tracks per-lane commits and asserts the
/// `commit` masks only ever cover lanes the loop just stepped.
struct MockBatchStep {
    scripts: Vec<Vec<usize>>,
    cursors: Vec<usize>,
    pub commits: Vec<usize>,
    pub resets: usize,
}

impl MockBatchStep {
    fn new(scripts: Vec<Vec<usize>>) -> Self {
        let lanes = scripts.len();
        Self { scripts, cursors: vec![0; lanes], commits: vec![0; lanes], resets: 0 }
    }
}

impl BatchJointStep for MockBatchStep {
    type Error = Infallible;

    fn batch(&self) -> usize {
        self.scripts.len()
    }

    fn step(&mut self, _t: usize, _prev: &[usize], active: &[bool], out: &mut [usize]) -> Result<(), Self::Error> {
        for i in 0..self.scripts.len() {
            if active[i] {
                let idx = self.cursors[i].min(self.scripts[i].len() - 1);
                self.cursors[i] += 1;
                out[i] = self.scripts[i][idx];
            }
        }
        Ok(())
    }

    fn commit(&mut self, lanes: &[bool]) {
        for (i, &c) in lanes.iter().enumerate() {
            if c {
                self.commits[i] += 1;
            }
        }
    }

    fn reset(&mut self) {
        self.resets += 1;
        self.cursors.fill(0);
    }
}

/// Label-looping mock: same per-lane scripts as [`MockBatchStep`] (each lane
/// consumes its script in its own joint-call order), plus counters proving the
/// predictor only runs on emission rounds.
struct MockBatchLabelStep {
    scripts: Vec<Vec<usize>>,
    cursors: Vec<usize>,
    pub commits: Vec<usize>,
    pub predicts: usize,
    pub joints: usize,
    pub resets: usize,
}

impl MockBatchLabelStep {
    fn new(scripts: Vec<Vec<usize>>) -> Self {
        let lanes = scripts.len();
        Self { scripts, cursors: vec![0; lanes], commits: vec![0; lanes], predicts: 0, joints: 0, resets: 0 }
    }
}

impl BatchLabelStep for MockBatchLabelStep {
    type Error = Infallible;

    fn batch(&self) -> usize {
        self.scripts.len()
    }

    fn predict(&mut self, _prev: &[usize]) -> Result<(), Self::Error> {
        self.predicts += 1;
        Ok(())
    }

    fn commit(&mut self, lanes: &[bool]) {
        for (i, &c) in lanes.iter().enumerate() {
            if c {
                self.commits[i] += 1;
            }
        }
    }

    fn joint(&mut self, _t: &[usize], active: &[bool], out: &mut [usize]) -> Result<(), Self::Error> {
        self.joints += 1;
        for i in 0..self.scripts.len() {
            if active[i] {
                let idx = self.cursors[i].min(self.scripts[i].len() - 1);
                self.cursors[i] += 1;
                out[i] = self.scripts[i][idx];
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.resets += 1;
        self.cursors.fill(0);
    }
}

/// Device-block mock: a plain-Rust transcription of the WIND device
/// `forward_block` per-step greedy logic (`model/src/gigaam/rnnt/block.rs`),
/// driven by the same per-lane token-id scripts as [`MockBatchLabelStep`]. Each
/// `run_block` advances every lane `block_steps` window steps and fills the
/// lane-major `[lanes * block_steps]` tapes the host consumer reads.
///
/// Per step, an in-bounds lane evaluates the joint over a window of `window`
/// frames against the FIXED predictor state and jumps to the first non-blank:
/// it scans the leading blanks (advancing `time`), and on the first non-blank
/// emits one token at that frame (a jump to a fresh frame resets the same-frame
/// run length; the `max_symbols` cap then forces a single-frame advance). An
/// all-blank window advances `time` by the in-bounds run length. Each step
/// produces exactly one tape entry per lane (the device's single windowed op),
/// so the blank frames skipped within a step never cost a tape slot — the WIND
/// win. `window == 1` reduces to the per-frame baseline.
///
/// The script is consumed along the greedy path: `first_nb + 1` tokens on an
/// emit step (leading blanks + the emit) and the in-bounds blank count on an
/// all-blank step — identical total consumption to [`MockBatchLabelStep`], so
/// transcripts match regardless of `window` or `block_steps`.
struct MockBatchBlockStep {
    scripts: Vec<Vec<usize>>,
    cursors: Vec<usize>,
    valid: Vec<usize>,
    blank: usize,
    max_symbols: usize,
    block_steps: usize,
    window: usize,
    // Carried per-lane state (device-resident on a real backend).
    time: Vec<usize>,
    prev: Vec<usize>,
    symbols: Vec<usize>,
    // Tape buffers, reused each block; lane-major `[lanes * block_steps]`.
    tokens: Vec<i32>,
    emit: Vec<i32>,
    frames: Vec<i32>,
    pub resets: usize,
}

impl MockBatchBlockStep {
    fn new(
        scripts: Vec<Vec<usize>>,
        valid: Vec<usize>,
        blank: usize,
        max_symbols: usize,
        block_steps: usize,
        window: usize,
    ) -> Self {
        let lanes = scripts.len();
        assert_eq!(valid.len(), lanes, "MockBatchBlockStep: one valid length per lane");
        let cap = lanes * block_steps;
        Self {
            scripts,
            cursors: vec![0; lanes],
            valid,
            blank,
            max_symbols: max_symbols.max(1),
            block_steps,
            window: window.max(1),
            time: vec![0; lanes],
            prev: vec![blank; lanes],
            symbols: vec![0; lanes],
            tokens: vec![0; cap],
            emit: vec![0; cap],
            frames: vec![0; cap],
            resets: 0,
        }
    }

    /// Peek the scripted argmax `ahead` frames into lane `i`'s window without
    /// advancing the cursor (clamps at the script end like the other mocks).
    fn peek(&self, lane: usize, ahead: usize) -> usize {
        let sc = &self.scripts[lane];
        sc[(self.cursors[lane] + ahead).min(sc.len() - 1)]
    }
}

impl BatchBlockStep for MockBatchBlockStep {
    type Error = Infallible;

    fn batch(&self) -> usize {
        self.scripts.len()
    }

    fn block_steps(&self) -> usize {
        self.block_steps
    }

    fn run_block(&mut self) -> Result<BlockTapes<'_>, Self::Error> {
        let lanes = self.scripts.len();
        let k = self.block_steps;
        let w = self.window;
        for s in 0..k {
            for i in 0..lanes {
                let j = i * k + s;
                let in_bounds = self.time[i] < self.valid[i];
                // safe_t = time when in bounds, else last (valid - 1); recorded
                // on the frame tape (filtered by emit == 0 when out of bounds).
                let safe_t = if in_bounds { self.time[i] as i64 } else { self.valid[i] as i64 - 1 };
                if !in_bounds {
                    self.tokens[j] = 0;
                    self.emit[j] = 0;
                    self.frames[j] = safe_t as i32;
                    continue;
                }
                // Scan the window against the fixed state: leading blanks, then
                // the first non-blank. `consumed` counts script tokens on the
                // greedy path (the off-path window tail is never consumed).
                let mut consumed = 0usize;
                let mut emit_off: Option<usize> = None;
                while consumed < w && self.time[i] + consumed < self.valid[i] {
                    let tok = self.peek(i, consumed);
                    consumed += 1;
                    if tok != self.blank {
                        emit_off = Some(consumed - 1);
                        break;
                    }
                }
                self.cursors[i] += consumed;
                if let Some(off) = emit_off {
                    let tok = self.scripts[i][(self.cursors[i] - 1).min(self.scripts[i].len() - 1)];
                    let frame = self.time[i] + off;
                    self.tokens[j] = tok as i32;
                    self.emit[j] = 1;
                    self.frames[j] = frame as i32;
                    self.prev[i] = tok;
                    // A jump to a fresh frame (off >= 1) resets the same-frame
                    // counter before counting this emission.
                    let sym_base = if off >= 1 { 0 } else { self.symbols[i] };
                    let symbols1 = sym_base + 1;
                    self.time[i] += off;
                    if symbols1 >= self.max_symbols {
                        self.time[i] += 1; // cap forces a single-frame advance
                        self.symbols[i] = 0;
                    } else {
                        self.symbols[i] = symbols1;
                    }
                } else {
                    // All in-bounds window frames blank: advance by the run length
                    // (= min(window, valid - time)); one inert tape entry.
                    self.tokens[j] = 0;
                    self.emit[j] = 0;
                    self.frames[j] = safe_t as i32;
                    self.time[i] += consumed;
                    self.symbols[i] = 0;
                }
            }
        }
        let active_any = (0..lanes).any(|i| self.time[i] < self.valid[i]);
        Ok(BlockTapes { tokens: &self.tokens, emit: &self.emit, frames: &self.frames, active_any })
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.resets += 1;
        self.cursors.fill(0);
        self.time.fill(0);
        self.symbols.fill(0);
        self.prev.fill(self.blank);
        Ok(())
    }
}

/// B=1 mock over a token-id script (the argmax already applied), for per-lane
/// equivalence checks against the batched loop.
struct MockTokenStep {
    script: Vec<usize>,
    cursor: usize,
    pub commits: usize,
}

impl JointStep for MockTokenStep {
    type Error = Infallible;

    fn step(&mut self, _enc: &[f32], _prev: Option<usize>) -> Result<usize, Self::Error> {
        let idx = self.cursor.min(self.script.len() - 1);
        self.cursor += 1;
        Ok(self.script[idx])
    }

    fn commit(&mut self) {
        self.commits += 1;
    }

    fn reset(&mut self) {}
}

// ─── Helpers ──────────────────────────────────────────────────────────────

fn abc_vocab() -> Vec<String> {
    vec!["a".into(), "b".into(), "c".into()]
}

fn one_hot_logits(winner: usize, total_vocab: usize) -> Vec<f32> {
    let mut v = vec![-10.0f32; total_vocab];
    v[winner] = 0.0;
    v
}

/// Build encoder frames where frame `t` is filled with value `t as f32`.
/// `enc_hidden = 1` keeps tests focused on the search loop.
fn linspace_encoder(n_frames: usize) -> Vec<f32> {
    (0..n_frames).map(|t| t as f32).collect()
}

fn decoder_with(vocab: Vec<String>, max_symbols: usize) -> RnntDecoder {
    RnntDecoder::new(vocab, RnntOpts { max_symbols_per_step: max_symbols })
}

// ─── Unit tests ───────────────────────────────────────────────────────────

#[test]
fn test_rnnt_blank_only_emits_empty() {
    let decoder = decoder_with(abc_vocab(), 10);
    let blank = decoder.blank_id();
    let total = decoder.total_vocab();
    // Every frame: blank wins on the first inner call → outer loop advances.
    let mut mock = MockJointStep::new(vec![one_hot_logits(blank, total); 5]);

    let enc = linspace_encoder(5);
    let text = decoder.decode(&enc, 5, 5, 1, &mut mock).unwrap();
    assert_eq!(text, "");
    assert_eq!(mock.commits, 0, "blank-only decode must not commit");
    assert_eq!(mock.resets, 1, "exactly one reset per decode");
    assert_eq!(mock.seen.len(), 5, "one step per frame");
    // prev_token stays None throughout (no commits).
    for (_, prev) in &mock.seen {
        assert_eq!(*prev, None);
    }
}

#[test]
fn test_rnnt_one_emit_per_frame() {
    let decoder = decoder_with(abc_vocab(), 10);
    let blank = decoder.blank_id();
    let total = decoder.total_vocab();
    // Per frame: one non-blank, then blank → emit once and advance.
    let script = vec![
        one_hot_logits(0, total), // emit 'a'
        one_hot_logits(blank, total),
        one_hot_logits(1, total), // emit 'b'
        one_hot_logits(blank, total),
        one_hot_logits(2, total), // emit 'c'
        one_hot_logits(blank, total),
    ];
    let mut mock = MockJointStep::new(script);

    let enc = linspace_encoder(3);
    let text = decoder.decode(&enc, 3, 3, 1, &mut mock).unwrap();
    assert_eq!(text, "abc");
    assert_eq!(mock.commits, 3);
    assert_eq!(mock.resets, 1);
    // First step is empty-prefix (None); subsequent steps see the previously
    // emitted token.
    assert_eq!(mock.seen[0].1, None);
    assert_eq!(mock.seen[1].1, Some(0)); // prev_token after emitting 'a'
    assert_eq!(mock.seen[2].1, Some(0)); // first inner call of frame 1
    assert_eq!(mock.seen[3].1, Some(1));
    assert_eq!(mock.seen[4].1, Some(1));
    assert_eq!(mock.seen[5].1, Some(2));
}

#[test]
fn test_rnnt_multi_emit_per_frame() {
    let decoder = decoder_with(abc_vocab(), 10);
    let blank = decoder.blank_id();
    let total = decoder.total_vocab();
    // Frame 0: emit 'a', 'b', then blank → moves on. Frame 1: blank only.
    let script = vec![
        one_hot_logits(0, total),
        one_hot_logits(1, total),
        one_hot_logits(blank, total),
        one_hot_logits(blank, total),
    ];
    let mut mock = MockJointStep::new(script);

    let enc = linspace_encoder(2);
    let text = decoder.decode(&enc, 2, 2, 1, &mut mock).unwrap();
    assert_eq!(text, "ab");
    assert_eq!(mock.commits, 2);
}

#[test]
fn test_rnnt_max_symbols_caps_inner_loop() {
    let decoder = decoder_with(abc_vocab(), 3);
    let total = decoder.total_vocab();
    // Always emit 'a' — without max_symbols cap this would loop forever.
    // Single frame, so max_symbols_per_step caps to 3 emissions.
    let mut mock = MockJointStep::new(vec![one_hot_logits(0, total); 32]);

    let enc = linspace_encoder(1);
    let text = decoder.decode(&enc, 1, 1, 1, &mut mock).unwrap();
    assert_eq!(text, "aaa", "max_symbols_per_step must cap emissions");
    assert_eq!(mock.commits, 3);
    assert_eq!(mock.seen.len(), 3, "no extra step calls past the cap");
}

#[test]
fn test_rnnt_nan_logits_fall_through_argmax() {
    let decoder = decoder_with(abc_vocab(), 10);
    let total = decoder.total_vocab();
    // Frame 0: NaN-poisoned frame. argmax_nan_safe treats NaN as
    // smaller-than-anything, so the first non-NaN wins. With logits
    // [NaN, NaN, NaN, NaN] argmax is index 0 — emits 'a'. We then schedule a
    // blank to escape the inner loop.
    let blank = decoder.blank_id();
    let script = vec![vec![f32::NAN; total], one_hot_logits(blank, total)];
    let mut mock = MockJointStep::new(script);

    let enc = linspace_encoder(1);
    let text = decoder.decode(&enc, 1, 1, 1, &mut mock).unwrap();
    assert_eq!(text, "a");
}

#[test]
fn test_rnnt_empty_vocab() {
    let decoder = decoder_with(Vec::new(), 10);
    let mut mock = MockJointStep::new(vec![vec![1.0; 1]]);
    let text = decoder.decode(&[], 0, 0, 1, &mut mock).unwrap();
    assert_eq!(text, "");
    assert_eq!(mock.resets, 0, "no reset for empty vocab — early-exit before loop");
    assert_eq!(mock.seen.len(), 0);
}

#[test]
fn test_rnnt_zero_valid_frames() {
    let decoder = decoder_with(abc_vocab(), 10);
    let mut mock = MockJointStep::new(vec![one_hot_logits(0, 4)]);
    let enc = linspace_encoder(3);
    let text = decoder.decode(&enc, 3, 0, 1, &mut mock).unwrap();
    assert_eq!(text, "");
    assert_eq!(mock.seen.len(), 0);
}

#[test]
fn test_rnnt_stride_clamps_valid_frames() {
    let decoder = decoder_with(abc_vocab(), 10);
    let blank = decoder.blank_id();
    let total = decoder.total_vocab();
    // valid_frames > stride: only `stride` frames are consumed.
    let mut mock = MockJointStep::new(vec![one_hot_logits(blank, total); 2]);
    let enc = linspace_encoder(2);
    let _ = decoder.decode(&enc, 2, 5, 1, &mut mock).unwrap();
    assert_eq!(mock.seen.len(), 2);
}

#[test]
fn test_rnnt_with_timestamps() {
    let decoder = decoder_with(abc_vocab(), 10);
    let blank = decoder.blank_id();
    let total = decoder.total_vocab();
    // Frame 0: emit 'a', blank. Frame 1: blank. Frame 2: emit 'b', 'c', blank.
    let script = vec![
        one_hot_logits(0, total),
        one_hot_logits(blank, total),
        one_hot_logits(blank, total),
        one_hot_logits(1, total),
        one_hot_logits(2, total),
        one_hot_logits(blank, total),
    ];
    let mut mock = MockJointStep::new(script);

    let enc = linspace_encoder(3);
    let (text, emissions) = decoder.decode_with_timestamps(&enc, 3, 3, 1, &mut mock).unwrap();
    assert_eq!(text, "abc");
    let pairs: Vec<(usize, usize)> = emissions.iter().map(|e| (e.token_id, e.frame)).collect();
    assert_eq!(pairs, vec![(0, 0), (1, 2), (2, 2)]);
}

#[test]
fn test_rnnt_step_receives_correct_encoder_frame() {
    let decoder = decoder_with(abc_vocab(), 10);
    let blank = decoder.blank_id();
    let total = decoder.total_vocab();
    let mut mock = MockJointStep::new(vec![one_hot_logits(blank, total); 4]);

    // Per-frame encoder values: 10, 20, 30, 40.
    let enc = vec![10.0f32, 20.0, 30.0, 40.0];
    let _ = decoder.decode(&enc, 4, 4, 1, &mut mock).unwrap();
    let frame_values: Vec<f32> = mock.seen.iter().map(|(v, _)| *v).collect();
    assert_eq!(frame_values, vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_rnnt_backend_error_surfaces_with_frame_index() {
    use snafu::Snafu;

    #[derive(Debug, Snafu)]
    #[snafu(display("boom"))]
    struct Boom;

    struct FailingStep;
    impl JointStep for FailingStep {
        type Error = Boom;
        fn step(&mut self, _: &[f32], _: Option<usize>) -> Result<usize, Self::Error> {
            Err(Boom)
        }
        fn commit(&mut self) {}
        fn reset(&mut self) {}
    }

    let decoder = decoder_with(abc_vocab(), 10);
    let enc = linspace_encoder(3);
    let mut step = FailingStep;
    let err = decoder.decode(&enc, 3, 3, 1, &mut step).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("frame 0"), "error message preserves frame index, got: {msg}");
}

#[cfg(feature = "serde")]
#[test]
fn test_rnnt_opts_serde_default_roundtrip() {
    let opts: RnntOpts = serde_json::from_str("{}").unwrap();
    assert_eq!(opts.max_symbols_per_step, 10);

    let opts: RnntOpts = serde_json::from_str(r#"{"max_symbols_per_step": 5}"#).unwrap();
    assert_eq!(opts.max_symbols_per_step, 5);
}

// ─── decode_batch ─────────────────────────────────────────────────────────

/// Batched lockstep decode must reproduce the B=1 loop per lane, lane scripts
/// consumed in each lane's own step order, regardless of frame raggedness.
fn assert_batch_matches_b1(scripts: Vec<Vec<usize>>, valid_frames: Vec<usize>, max_symbols: usize) {
    let decoder = decoder_with(abc_vocab(), max_symbols);
    let mut batch = MockBatchStep::new(scripts.clone());
    let batched = decoder.decode_batch(&valid_frames, &mut batch).unwrap();
    assert_eq!(batch.resets, 1);

    for (i, (script, &valid)) in scripts.iter().zip(&valid_frames).enumerate() {
        let mut single = MockTokenStep { script: script.clone(), cursor: 0, commits: 0 };
        let (text, emissions) =
            decoder.decode_with_timestamps(&linspace_encoder(valid.max(1)), valid, valid, 1, &mut single).unwrap();
        assert_eq!(batched[i].0, text, "lane {i} text");
        assert_eq!(batched[i].1, emissions, "lane {i} emissions");
        assert_eq!(batch.commits[i], single.commits, "lane {i} commits");
    }
}

#[test]
fn test_rnnt_batch_matches_b1_ragged() {
    let blank = abc_vocab().len();
    // Lane 0: emit a, blank, emit b, blank (2 frames). Lane 1: blanks only
    // (1 frame). Lane 2: multi-emit then cap (3 frames).
    assert_batch_matches_b1(
        vec![vec![0, blank, 1, blank], vec![blank; 4], vec![0, 1, 2, 0, 1, 2, blank, blank, blank]],
        vec![2, 1, 3],
        2,
    );
}

#[test]
fn test_rnnt_batch_zero_valid_lane_emits_nothing() {
    let blank = abc_vocab().len();
    let decoder = decoder_with(abc_vocab(), 10);
    let mut batch = MockBatchStep::new(vec![vec![0, blank], vec![0, blank]]);
    let out = decoder.decode_batch(&[1, 0], &mut batch).unwrap();
    assert_eq!(out[0].0, "a");
    assert_eq!(out[1].0, "", "zero-valid lane must stay silent");
    assert_eq!(batch.commits[1], 0);
}

#[test]
fn test_rnnt_batch_fewer_items_than_lanes() {
    let blank = abc_vocab().len();
    let decoder = decoder_with(abc_vocab(), 10);
    let mut batch = MockBatchStep::new(vec![vec![1, blank], vec![0, blank], vec![2, blank], vec![2, blank]]);
    let out = decoder.decode_batch(&[1, 1], &mut batch).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!((out[0].0.as_str(), out[1].0.as_str()), ("b", "a"));
    assert_eq!(&batch.commits[2..], &[0, 0], "padding lanes never commit");
}

// ─── decode_batch_labels ──────────────────────────────────────────────────

/// Label-looping must reproduce lockstep `decode_batch` exactly: same texts,
/// emissions, and commit counts per lane (greedy is lane-independent).
fn assert_labels_match_lockstep(scripts: Vec<Vec<usize>>, valid_frames: Vec<usize>, max_symbols: usize) {
    let decoder = decoder_with(abc_vocab(), max_symbols);
    let mut lockstep = MockBatchStep::new(scripts.clone());
    let expected = decoder.decode_batch(&valid_frames, &mut lockstep).unwrap();

    let mut labels = MockBatchLabelStep::new(scripts);
    let got = decoder.decode_batch_labels(&valid_frames, &mut labels).unwrap();

    assert_eq!(got, expected);
    assert_eq!(labels.commits, lockstep.commits);
    assert_eq!(labels.resets, 1);
}

#[test_case::test_case(vec![vec![0, 3, 1, 3], vec![3; 4], vec![0, 1, 2, 0, 1, 2, 3, 3, 3]], vec![2, 1, 3], 2; "ragged with cap")]
#[test_case::test_case(vec![vec![3; 8]], vec![8], 10; "blank only")]
#[test_case::test_case(vec![vec![0, 1, 2, 3], vec![2, 3, 1, 3]], vec![2, 2], 10; "multi emit")]
#[test_case::test_case(vec![vec![0, 3], vec![0, 3]], vec![1, 0], 10; "zero valid lane")]
fn test_rnnt_labels_match_lockstep(scripts: Vec<Vec<usize>>, valid: Vec<usize>, max_symbols: usize) {
    assert_labels_match_lockstep(scripts, valid, max_symbols);
}

#[test]
fn test_rnnt_labels_predictor_runs_only_on_emission_rounds() {
    let blank = abc_vocab().len();
    let decoder = decoder_with(abc_vocab(), 10);
    // 4 blank-only frames: predictor must run exactly once (the empty prefix).
    let mut labels = MockBatchLabelStep::new(vec![vec![blank; 4]]);
    decoder.decode_batch_labels(&[4], &mut labels).unwrap();
    assert_eq!(labels.predicts, 1, "blank advances must not invoke the predictor");
    assert_eq!(labels.joints, 4);

    // 1 frame, 2 emissions + blank: 1 prefix + 2 emission rounds.
    let mut labels = MockBatchLabelStep::new(vec![vec![0, 1, blank]]);
    decoder.decode_batch_labels(&[1], &mut labels).unwrap();
    assert_eq!(labels.predicts, 3);
}

// ─── decode_batch_blocks ──────────────────────────────────────────────────

/// The WIND device-block loop must reproduce label-looping (and hence lockstep)
/// exactly: identical texts and `(token_id, frame)` emissions per lane,
/// independent of the block stride OR the decode window. Pins the device
/// `forward_block` behavior, which had no isolated test.
fn assert_blocks_match_labels(
    scripts: Vec<Vec<usize>>,
    valid_frames: Vec<usize>,
    max_symbols: usize,
    block_steps: usize,
    window: usize,
) {
    let decoder = decoder_with(abc_vocab(), max_symbols);

    let mut labels = MockBatchLabelStep::new(scripts.clone());
    let expected = decoder.decode_batch_labels(&valid_frames, &mut labels).unwrap();

    let mut blocks =
        MockBatchBlockStep::new(scripts, valid_frames.clone(), decoder.blank_id(), max_symbols, block_steps, window);
    let got = decoder.decode_batch_blocks(&valid_frames, &mut blocks).unwrap();
    assert_eq!(blocks.resets, 1, "exactly one reset per decode");

    assert_eq!(got, expected, "block stride {block_steps}, window {window}");
}

#[test_case::test_case(vec![vec![0, 3, 1, 3], vec![3; 4], vec![0, 1, 2, 0, 1, 2, 3, 3, 3]], vec![2, 1, 3], 2; "ragged with cap")]
#[test_case::test_case(vec![vec![3; 8]], vec![8], 10; "blank only")]
#[test_case::test_case(vec![vec![0, 1, 2, 3], vec![2, 3, 1, 3]], vec![2, 2], 10; "multi emit")]
#[test_case::test_case(vec![vec![0, 3], vec![0, 3]], vec![1, 0], 10; "zero valid lane")]
// WIND-specific subtleties the window scan must get right at W in {2,4,8}:
#[test_case::test_case(vec![vec![3; 10]], vec![10], 10; "blank run longer than window")]
#[test_case::test_case(vec![vec![3, 3, 3, 0, 3]], vec![5], 10; "first nonblank at last window offset")]
#[test_case::test_case(vec![vec![3, 3, 0]], vec![3], 10; "emit at valid-1 window spills past valid")]
#[test_case::test_case(vec![vec![3, 3, 0]], vec![4], 1; "jumped then cap")]
#[test_case::test_case(vec![vec![0, 0, 0, 3]], vec![1], 2; "same-frame multi emit cap inside window")]
fn test_rnnt_blocks_match_labels(scripts: Vec<Vec<usize>>, valid: Vec<usize>, max_symbols: usize) {
    // The stride is a host-readback knob and the window is the WIND blank-skip
    // width; equivalence must hold for every combination, including strides that
    // split a frame's multi-emit run across block boundaries and windows that
    // span more than a full blank run.
    for block_steps in [1, 3, 16] {
        for window in [1, 2, 4, 8] {
            assert_blocks_match_labels(scripts.clone(), valid.clone(), max_symbols, block_steps, window);
        }
    }
}

// ─── frames_to_words ──────────────────────────────────────────────────────

fn em(token_id: usize, frame: usize) -> TokenEmission {
    TokenEmission { token_id, frame }
}

fn assert_words(got: &[Word], expected: &[(&str, f32, f32)]) {
    assert_eq!(got.len(), expected.len(), "word count mismatch: got {got:?}, expected {expected:?}");
    for (g, (text, start, end)) in got.iter().zip(expected.iter()) {
        assert_eq!(&g.text, text);
        assert!((g.start - start).abs() < 1e-5, "start: got {}, expected {}", g.start, start);
        assert!((g.end - end).abs() < 1e-5, "end: got {}, expected {}", g.end, end);
    }
}

#[test]
fn test_frames_to_words_sentencepiece_boundaries() {
    let decoder = decoder_with(vec!["\u{2581}hello".into(), "\u{2581}world".into(), "!".into()], 10);
    let emissions = vec![em(0, 0), em(1, 10), em(2, 12)];
    let words = decoder.frames_to_words(&emissions, 0.04);
    assert_words(&words, &[("hello", 0.0, 0.04), (" world!", 0.40, 0.52)]);
}

#[test]
fn test_frames_to_words_multi_piece_word() {
    let decoder = decoder_with(vec!["\u{2581}при".into(), "ве".into(), "т".into()], 10);
    let emissions = vec![em(0, 5), em(1, 6), em(2, 7)];
    let words = decoder.frames_to_words(&emissions, 0.04);
    assert_words(&words, &[("привет", 0.20, 0.32)]);
}

#[test]
fn test_frames_to_words_empty_emissions() {
    let decoder = decoder_with(vec!["\u{2581}x".into()], 10);
    assert!(decoder.frames_to_words(&[], 0.04).is_empty());
}

#[test]
fn test_frames_to_words_literal_space_separator() {
    let decoder = decoder_with(vec!["hello".into(), " ".into(), "world".into()], 10);
    let emissions = vec![em(0, 0), em(1, 2), em(2, 3)];
    let words = decoder.frames_to_words(&emissions, 0.1);
    assert_words(&words, &[("hello", 0.0, 0.1), (" world", 0.3, 0.4)]);
}

#[test]
fn test_frames_to_words_bare_marker_collapses_to_single_space() {
    // A bare marker is whitespace-only and is dropped, while the following
    // fragment retains its own SentencePiece boundary.
    let decoder = decoder_with(vec!["\u{2581}hi".into(), "\u{2581}".into(), "\u{2581}mom".into()], 10);
    let emissions = vec![em(0, 0), em(1, 1), em(2, 2)];
    let words = decoder.frames_to_words(&emissions, 0.04);
    assert_words(&words, &[("hi", 0.0, 0.04), (" mom", 0.08, 0.12)]);
    assert_eq!(crate::pipelines::audio::words_to_text(&words), "hi mom");
}

// ─── Property tests ───────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Output token count never exceeds `valid_frames * max_symbols_per_step`.
    /// The cap is the only termination guarantee for adversarial backends.
    #[test]
    fn prop_rnnt_output_length_bounded(
        valid_frames in 0usize..40,
        max_symbols in 1usize..6,
    ) {
        let decoder = decoder_with(abc_vocab(), max_symbols);
        let total = decoder.total_vocab();
        // Pathological backend: always emits 'a' (never blank) — forces the
        // cap to fire on every frame.
        let script_len = valid_frames * max_symbols + 1;
        let mut mock = MockJointStep::new(vec![one_hot_logits(0, total); script_len.max(1)]);
        let enc = linspace_encoder(valid_frames.max(1));
        let text = decoder.decode(&enc, valid_frames, valid_frames, 1, &mut mock).unwrap();
        prop_assert!(text.chars().count() <= valid_frames * max_symbols);
    }

    /// Number of `commit` calls equals output token count. The search loop
    /// must commit exactly once per non-blank emission, never on blank.
    #[test]
    fn prop_rnnt_commit_count_matches_emissions(
        n_frames in 1usize..15,
        seed in any::<u64>(),
    ) {
        let decoder = decoder_with(abc_vocab(), 10);
        let blank = decoder.blank_id();
        let total = decoder.total_vocab();
        // Pseudo-random script of (token | blank) entries, large enough that
        // the search loop always finds a blank within max_symbols.
        let mut state = seed;
        let mut script = Vec::with_capacity(n_frames * 12);
        for _ in 0..(n_frames * 12) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // 60% blank (so inner loop terminates promptly), 40% non-blank.
            let pick = if (state >> 32) % 5 < 3 { blank } else { ((state >> 32) % 3) as usize };
            script.push(one_hot_logits(pick, total));
        }
        let mut mock = MockJointStep::new(script);
        let enc = linspace_encoder(n_frames);
        let text = decoder.decode(&enc, n_frames, n_frames, 1, &mut mock).unwrap();
        prop_assert_eq!(text.chars().count(), mock.commits);
    }

    /// Batched lockstep decode ≡ per-lane B=1 decode for random scripts and
    /// ragged frame counts.
    #[test]
    fn prop_rnnt_batch_matches_b1(
        n_lanes in 1usize..6,
        max_symbols in 1usize..4,
        seed in any::<u64>(),
    ) {
        let blank = abc_vocab().len();
        let mut state = seed;
        let mut scripts = Vec::with_capacity(n_lanes);
        let mut valid = Vec::with_capacity(n_lanes);
        for _ in 0..n_lanes {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let frames = ((state >> 32) % 8) as usize; // 0..8 frames per lane
            valid.push(frames);
            let mut script = Vec::with_capacity(frames * max_symbols + 1);
            for _ in 0..(frames * max_symbols + 1) {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let pick = if (state >> 32) % 5 < 3 { blank } else { ((state >> 32) % 3) as usize };
                script.push(pick);
            }
            scripts.push(script);
        }
        assert_batch_matches_b1(scripts, valid, max_symbols);
    }

    /// WIND device-block decode ≡ label-looping decode for random scripts,
    /// ragged frame counts, block strides, and decode windows. Pins
    /// `forward_block` against the reference greedy loop without a GPU.
    #[test]
    fn prop_rnnt_blocks_match_labels(
        n_lanes in 1usize..6,
        max_symbols in 1usize..4,
        block_steps in 1usize..6,
        window in 1usize..9,
        seed in any::<u64>(),
    ) {
        let blank = abc_vocab().len();
        let mut state = seed;
        let mut scripts = Vec::with_capacity(n_lanes);
        let mut valid = Vec::with_capacity(n_lanes);
        for _ in 0..n_lanes {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let frames = ((state >> 32) % 8) as usize; // 0..8 frames per lane
            valid.push(frames);
            // Enough tokens to cover the worst case (every frame caps): the
            // mock clamps the cursor at the end regardless.
            let mut script = Vec::with_capacity(frames * max_symbols + 1);
            for _ in 0..(frames * max_symbols + 1) {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let pick = if (state >> 32) % 5 < 3 { blank } else { ((state >> 32) % 3) as usize };
                script.push(pick);
            }
            scripts.push(script);
        }
        // All-zero-valid is handled by an early return in the driver; skip it so
        // the reset-count assertion stays meaningful.
        if valid.iter().any(|&v| v > 0) {
            assert_blocks_match_labels(scripts, valid, max_symbols, block_steps, window);
        }
    }

    /// Timestamps are non-decreasing and each frame index is in `[0, valid_frames)`.
    #[test]
    fn prop_rnnt_timestamps_monotonic_and_in_bounds(
        n_frames in 1usize..15,
        seed in any::<u64>(),
    ) {
        let decoder = decoder_with(abc_vocab(), 10);
        let blank = decoder.blank_id();
        let total = decoder.total_vocab();
        let mut state = seed;
        let mut script = Vec::with_capacity(n_frames * 12);
        for _ in 0..(n_frames * 12) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let pick = if (state >> 32) % 5 < 3 { blank } else { ((state >> 32) % 3) as usize };
            script.push(one_hot_logits(pick, total));
        }
        let mut mock = MockJointStep::new(script);
        let enc = linspace_encoder(n_frames);
        let (_, emissions) = decoder.decode_with_timestamps(&enc, n_frames, n_frames, 1, &mut mock).unwrap();
        for w in emissions.windows(2) {
            prop_assert!(w[0].frame <= w[1].frame);
        }
        for e in &emissions {
            prop_assert!(e.frame < n_frames);
        }
    }
}

/// Resuming decodes from the lanes' present state: no reset, the same tapes.
#[test]
fn resume_batch_blocks_skips_the_reset() {
    let decoder = decoder_with(abc_vocab(), 10);
    let blank = decoder.blank_id();
    let scripts = vec![vec![0, blank, 1, 1, blank, 2], vec![blank, 2, blank, blank, 0, blank]];
    let valid = vec![6, 5];
    let mut fresh = MockBatchBlockStep::new(scripts.clone(), valid.clone(), blank, 10, 4, 2);
    let want = decoder.decode_batch_blocks(&valid, &mut fresh).unwrap();
    assert_eq!(fresh.resets, 1);
    let mut resumed = MockBatchBlockStep::new(scripts, valid.clone(), blank, 10, 4, 2);
    let got = decoder.resume_batch_blocks(&valid, &mut resumed).unwrap();
    assert_eq!(resumed.resets, 0);
    assert_eq!(got, want);
    assert!(!got[0].1.is_empty());
}
