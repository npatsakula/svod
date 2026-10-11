//! RNN-T (transducer) greedy decoder.
//!
//! The decoder owns the search loop and the vocabulary. The predictor + joint
//! networks live behind the [`JointStep`] trait, which the model crate
//! implements with JIT calls. This crate stays Tensor-free and operates only
//! on plain `&[f32]` slices.
//!
//! Algorithm (B=1) mirrors `gigaam/decoding.py:139–207`:
//! ```text
//! step.reset()
//! prev_token = None
//! for t in 0..valid_frames:
//!   enc_t = encoder_frames[t * enc_hidden ..][..enc_hidden]
//!   for _ in 0..max_symbols_per_step:
//!     k = step.step(enc_t, prev_token)?   // argmax token, computed by the backend
//!     if k == blank_id { break }
//!     emit token k; prev_token = Some(k); step.commit()
//! ```
//!
//! The backend computes the joint argmax itself and returns the chosen token,
//! so this crate never sees a logit vector — on a GPU backend that keeps the
//! per-step host readback to a single integer.
//!
//! The "tentative / committed" predictor-state semantics on [`JointStep`] keep
//! the trait stateless from the search loop's perspective: the search loop
//! only knows about `prev_token` and never sees the LSTM hidden state. The
//! implementation owns and rolls its own state.
//!
//! # Layout convention
//!
//! `encoder_frames` is row-major `[stride_frames, enc_hidden]` for a single
//! batch item. `valid_frames` clamps padding from JIT static shapes — only
//! frames `0..valid_frames` are consumed.

use snafu::{IntoError, Snafu};

#[cfg(feature = "serde")]
use serde::Deserialize;

// ─── Errors ───────────────────────────────────────────────────────────────

/// Failure modes for [`RnntDecoder`]. The only failure surface is the user's
/// [`JointStep`] backend; anything else is a usage bug.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum RnntDecodeError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    #[snafu(display("predictor/joint backend failed at frame {frame}"))]
    Backend { source: E, frame: usize },
}

// ─── Options ──────────────────────────────────────────────────────────────

/// Tunables for [`RnntDecoder`]. Default `max_symbols_per_step = 10` matches
/// `gigaam/decoding.py:108`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct RnntOpts {
    /// Maximum non-blank tokens emitted per encoder frame. Caps a runaway
    /// inner loop if the joint never emits blank.
    pub max_symbols_per_step: usize,
}

impl Default for RnntOpts {
    fn default() -> Self {
        Self { max_symbols_per_step: 10 }
    }
}

// ─── Token emissions / words ──────────────────────────────────────────────

/// One non-blank token emitted by the greedy decoder. `frame` is the
/// encoder-frame index (`t` in the outer search loop) at which the joint
/// network selected this token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenEmission {
    pub token_id: usize,
    pub frame: usize,
}

/// An exact transcript fragment and its `[start, end)` time span in seconds.
///
/// `text` retains tokenizer-provided boundary whitespace and punctuation.
/// Concatenate fragments directly, then trim only the complete transcript or
/// chunk boundary. Whitespace-only fragments are not emitted.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

/// A phrase-level segment of a transcript with `[start, end)` time in seconds.
/// Produced by timestamp-token splitting (e.g. Whisper's `<|t0|> text <|t1|>`
/// pairs). The containing result defines the time origin: decode-window-relative
/// in `Transcript`, core-relative in `ChunkResult`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Segment {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

struct PendingWord {
    text: String,
    first_frame: usize,
    last_frame: usize,
}

fn flush_pending(pending: &mut Option<PendingWord>, frame_shift: f32, words: &mut Vec<Word>) {
    let Some(p) = pending.take() else { return };
    if !p.text.trim().is_empty() {
        words.push(Word {
            text: p.text,
            start: p.first_frame as f32 * frame_shift,
            end: (p.last_frame + 1) as f32 * frame_shift,
        });
    }
}

// ─── JointStep trait ──────────────────────────────────────────────────────

/// Bridge between the search loop and the predictor/joint backend.
///
/// The implementation owns the per-utterance LSTM state. The search loop
/// never sees it — it only drives `step` (compute logits), `commit` (promote
/// the last step's predictor state to the new committed state, on non-blank
/// emission) and `reset` (clear committed state at the start of an
/// utterance).
///
/// On a blank emission, the search loop simply does not call `commit`, and
/// the next `step` call's predictor input is the same `prev_token` as before
/// the blank. The implementation must therefore make `step` idempotent w.r.t.
/// committed state: each call reads from committed state and writes only to
/// tentative state.
pub trait JointStep {
    /// Backend-specific error (typically a JIT execution error).
    type Error: std::error::Error + Send + Sync + 'static;

    /// Run the predictor + joint for `(encoder_frame, prev_token)` against the
    /// LATEST COMMITTED predictor state and return the argmax token id over the
    /// full vocab (`blank_id` is `vocabulary.len()`). `prev_token = None`
    /// selects the empty-prefix initial state. The post-step predictor state is
    /// stashed internally as "tentative" and is only used if
    /// [`commit`](Self::commit) is called.
    fn step(&mut self, encoder_frame: &[f32], prev_token: Option<usize>) -> Result<usize, Self::Error>;

    /// Promote the last `step`'s tentative state to committed. The arch
    /// decoder calls this exactly once per non-blank emission.
    fn commit(&mut self);

    /// Reset committed state to the empty-prefix initial. Called once per
    /// utterance, before the first frame.
    fn reset(&mut self);
}

/// Per-lane result of [`RnntDecoder::decode_batch`]: decoded text + token
/// emissions in decode order.
pub type LaneDecode = (String, Vec<TokenEmission>);

// ─── BatchJointStep trait ─────────────────────────────────────────────────

/// B-lane batched variant of [`JointStep`]: one fused GPU dispatch advances all
/// lanes at the shared frame index, amortizing the per-step dispatch cost across
/// the independent items. Per-lane semantics are identical to [`JointStep`]:
/// each lane keeps its own committed/tentative predictor state, lanes use the
/// committed prefix on every step, and only [`commit`](Self::commit)-masked
/// lanes promote tentative → committed.
///
/// `prev[i]` is the lane's previous non-blank token (`blank_id` selects the
/// empty-prefix initial state, matching `prev_token = None` in [`JointStep`]).
/// `active[i]` masks both the inputs the lane sees and which `out[i]` slots are
/// trustworthy — inactive lanes carry garbage.
pub trait BatchJointStep {
    /// Backend-specific error (typically a JIT execution error).
    type Error: std::error::Error + Send + Sync + 'static;

    /// Lane capacity (must cover every `decode_batch` call's item count).
    fn batch(&self) -> usize;

    /// One batched step at frame `t`: write `prev` + committed state for every
    /// lane, execute once, return the joint argmax token of each ACTIVE lane in
    /// `out[i]`. Slices all have length [`batch`](Self::batch).
    fn step(&mut self, t: usize, prev: &[usize], active: &[bool], out: &mut [usize]) -> Result<(), Self::Error>;

    /// Promote tentative → committed for masked lanes. Called exactly once per
    /// inner step when at least one lane emitted non-blank.
    fn commit(&mut self, lanes: &[bool]);

    /// Reset all lanes to the empty-prefix initial state.
    fn reset(&mut self);
}

// ─── BatchLabelStep trait ─────────────────────────────────────────────────

/// Label-looping batched backend: the predictor (the expensive recurrent
/// network) and the joint (cheap projection + argmax) execute separately, so
/// blank-advance steps cost only the joint. Lanes carry per-lane frame
/// indices — there is no shared `t`. Per-lane semantics match
/// [`BatchJointStep`] exactly (greedy is lane-independent), so transcripts
/// are identical to lockstep decode.
pub trait BatchLabelStep {
    /// Backend-specific error (typically a JIT execution error).
    type Error: std::error::Error + Send + Sync + 'static;

    /// Lane capacity (must cover every `decode_batch_labels` call's item count).
    fn batch(&self) -> usize;

    /// Run the predictor for every lane's `prev` token over the committed
    /// state, producing a tentative state + the joint's predictor input.
    /// Called once before the loop (all-blank prefix) and once per
    /// emitted-label round — never per blank advance.
    fn predict(&mut self, prev: &[usize]) -> Result<(), Self::Error>;

    /// Promote the tentative predictor state for masked lanes.
    fn commit(&mut self, lanes: &[bool]);

    /// Joint + argmax at per-lane frames `t[i]` against the last `predict`
    /// output; `out[i]` is valid for ACTIVE lanes. Slices have length
    /// [`batch`](Self::batch).
    fn joint(&mut self, t: &[usize], active: &[bool], out: &mut [usize]) -> Result<(), Self::Error>;

    /// Reset all lanes to the empty-prefix initial state.
    fn reset(&mut self);
}

// ─── BatchBlockStep trait ─────────────────────────────────────────────────

/// One block's tapes, lane-major `[batch * block_steps]`, step-major within a
/// lane. `emit[i*K+k] != 0` flags a real emission of `tokens[i*K+k]` at
/// encoder frame `frames[i*K+k]`.
pub struct BlockTapes<'a> {
    pub tokens: &'a [i32],
    pub emit: &'a [i32],
    pub frames: &'a [i32],
    /// Any lane still has frames left after this block.
    pub active_any: bool,
}

/// Device-resident block decode: every state (per-lane time/prev/symbols +
/// predictor state) lives on the device; one `run_block` advances all lanes
/// [`block_steps`](Self::block_steps) decode steps and the host only reads the
/// token tape. Greedy semantics per lane are identical to [`BatchLabelStep`].
///
/// The device implementation (`svod_model::gigaam::rnnt::block::forward_block`)
/// uses WIND windowed non-blank detection: each step evaluates the joint over a
/// window of frames against the fixed predictor state and jumps to the first
/// non-blank. On this static-shape, masked-execution runtime WIND subsumes
/// label-looping — collapsing a blank run into one step cuts the predictor/joint
/// evaluations and host syncs a per-frame loop spends on blanks — at
/// byte-identical output. The trait contract is unchanged: one tape entry per
/// lane per step, `emit`-flagged emissions at `frames`.
pub trait BatchBlockStep {
    /// Backend-specific error (typically a JIT execution error).
    type Error: std::error::Error + Send + Sync + 'static;

    /// Lane capacity.
    fn batch(&self) -> usize;

    /// Steps per block (the tape stride).
    fn block_steps(&self) -> usize;

    /// Advance every lane `block_steps` steps; recycle carried state on
    /// device; return the tapes.
    fn run_block(&mut self) -> Result<BlockTapes<'_>, Self::Error>;

    /// Reset all lanes to the empty-prefix initial state (zero time/symbols,
    /// blank prev, zero LSTM state).
    fn reset(&mut self) -> Result<(), Self::Error>;
}

// ─── Decoder ──────────────────────────────────────────────────────────────

/// Greedy RNN-T decoder. Owns the vocabulary; the blank token id is implicit
/// at `vocabulary.len()`.
///
/// Decode is `&self` — all per-utterance mutable state lives in the
/// [`JointStep`] backend the caller supplies.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct RnntDecoder {
    vocabulary: Vec<String>,
    #[cfg_attr(feature = "serde", serde(default))]
    opts: RnntOpts,
}

impl RnntDecoder {
    pub fn new(vocabulary: Vec<String>, opts: RnntOpts) -> Self {
        Self { vocabulary, opts }
    }

    pub fn vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    pub fn opts(&self) -> &RnntOpts {
        &self.opts
    }

    pub fn blank_id(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn total_vocab(&self) -> usize {
        self.vocabulary.len() + 1
    }

    /// Group token emissions from [`decode_with_timestamps`](Self::decode_with_timestamps)
    /// into words and assign each a `[start, end)` time span in seconds.
    /// A new word begins on:
    ///
    /// - a piece prefixed with `▁` (U+2581, SentencePiece word marker), or
    /// - a piece equal to a single ASCII space (char-based vocabularies).
    ///
    /// Other pieces concatenate onto the current word. `start = first_frame *
    /// frame_shift`; `end = (last_frame + 1) * frame_shift`. Whitespace-only
    /// candidates are dropped.
    ///
    /// Port of `gigaam/timestamps_utils.py:frames_to_words`.
    pub fn frames_to_words(&self, emissions: &[TokenEmission], frame_shift: f32) -> Vec<Word> {
        const SP_MARK: char = '\u{2581}';

        let mut words: Vec<Word> = Vec::new();
        let mut pending: Option<PendingWord> = None;
        let mut separator = String::new();

        for e in emissions {
            let piece = match self.vocabulary.get(e.token_id) {
                Some(p) => p.as_str(),
                None => continue,
            };
            if let Some(stripped) = piece.strip_prefix(SP_MARK) {
                flush_pending(&mut pending, frame_shift, &mut words);
                separator.clear();
                let text = if words.is_empty() { stripped.to_string() } else { format!(" {stripped}") };
                pending = Some(PendingWord { text, first_frame: e.frame, last_frame: e.frame });
            } else if piece == " " {
                flush_pending(&mut pending, frame_shift, &mut words);
                separator.push(' ');
            } else {
                match &mut pending {
                    Some(p) => {
                        p.text.push_str(piece);
                        p.last_frame = e.frame;
                    }
                    None => {
                        let mut text = std::mem::take(&mut separator);
                        text.push_str(piece);
                        pending = Some(PendingWord { text, first_frame: e.frame, last_frame: e.frame });
                    }
                }
            }
        }
        flush_pending(&mut pending, frame_shift, &mut words);
        words
    }

    /// Greedy decode. `encoder_frames` is row-major `[stride_frames, enc_hidden]`.
    /// Empty vocabulary or `valid_frames == 0` yield an empty string.
    pub fn decode<S: JointStep>(
        &self,
        encoder_frames: &[f32],
        stride_frames: usize,
        valid_frames: usize,
        enc_hidden: usize,
        step: &mut S,
    ) -> Result<String, RnntDecodeError<S::Error>> {
        let (text, _emissions) =
            self.decode_inner(encoder_frames, stride_frames, valid_frames, enc_hidden, step, false)?;
        Ok(text)
    }

    /// Greedy decode + per-emission `(token_id, frame)` pairs. `emissions[i]`
    /// records which token was emitted and at which encoder frame, in
    /// decoder output order. Pair with [`Self::frames_to_words`] to recover
    /// word-level timestamps from a SentencePiece vocabulary.
    pub fn decode_with_timestamps<S: JointStep>(
        &self,
        encoder_frames: &[f32],
        stride_frames: usize,
        valid_frames: usize,
        enc_hidden: usize,
        step: &mut S,
    ) -> Result<(String, Vec<TokenEmission>), RnntDecodeError<S::Error>> {
        self.decode_inner(encoder_frames, stride_frames, valid_frames, enc_hidden, step, true)
    }

    fn decode_inner<S: JointStep>(
        &self,
        encoder_frames: &[f32],
        stride_frames: usize,
        valid_frames: usize,
        enc_hidden: usize,
        step: &mut S,
        keep_emissions: bool,
    ) -> Result<(String, Vec<TokenEmission>), RnntDecodeError<S::Error>> {
        if self.vocabulary.is_empty() || valid_frames == 0 {
            return Ok((String::new(), Vec::new()));
        }
        let blank_id = self.blank_id();
        let n_frames = stride_frames.min(valid_frames);

        step.reset();

        let mut text = String::new();
        let mut emissions = if keep_emissions { Vec::with_capacity(n_frames) } else { Vec::new() };
        let mut prev_token: Option<usize> = None;

        for t in 0..n_frames {
            let base = t * enc_hidden;
            let enc_t = &encoder_frames[base..base + enc_hidden];

            for _ in 0..self.opts.max_symbols_per_step {
                let k = step.step(enc_t, prev_token).map_err(|e| BackendSnafu { frame: t }.into_error(e))?;
                if k == blank_id {
                    break;
                }
                // Non-blank: emit and commit. The commit advances the
                // backend's committed predictor state to the post-step
                // state we just produced.
                text.push_str(&self.vocabulary[k]);
                if keep_emissions {
                    emissions.push(TokenEmission { token_id: k, frame: t });
                }
                prev_token = Some(k);
                step.commit();
            }
        }

        Ok((text, emissions))
    }

    /// Batched greedy decode: all lanes advance frame-in-lockstep through one
    /// [`BatchJointStep`] (one fused dispatch per inner step). Lane `i` consumes
    /// frames `0..valid_frames[i]` and evolves exactly like the B=1
    /// [`decode`](Self::decode) loop — blank ends the lane's inner loop, a
    /// non-blank emits, sets `prev[i]`, and commits the lane's state.
    /// Returns `(text, emissions)` per lane.
    pub fn decode_batch<S: BatchJointStep>(
        &self,
        valid_frames: &[usize],
        step: &mut S,
    ) -> Result<Vec<LaneDecode>, RnntDecodeError<S::Error>> {
        let b = valid_frames.len();
        let lanes = step.batch();
        assert!(b <= lanes, "decode_batch: {b} items exceed the backend's {lanes} lanes");
        let blank_id = self.blank_id();
        let mut texts = vec![String::new(); b];
        let mut emissions = vec![Vec::new(); b];
        if self.vocabulary.is_empty() {
            return Ok(texts.into_iter().zip(emissions).collect());
        }
        let max_t = valid_frames.iter().copied().max().unwrap_or(0);

        step.reset();
        let mut prev = vec![blank_id; lanes];
        let mut active = vec![false; lanes];
        let mut out = vec![blank_id; lanes];
        let mut commit_lanes = vec![false; lanes];

        for t in 0..max_t {
            for i in 0..lanes {
                active[i] = i < b && t < valid_frames[i];
            }
            for _ in 0..self.opts.max_symbols_per_step {
                if !active.iter().any(|&a| a) {
                    break;
                }
                step.step(t, &prev, &active, &mut out).map_err(|e| BackendSnafu { frame: t }.into_error(e))?;
                let mut any_commit = false;
                for i in 0..b {
                    if !active[i] {
                        commit_lanes[i] = false;
                        continue;
                    }
                    let k = out[i];
                    if k == blank_id {
                        active[i] = false;
                        commit_lanes[i] = false;
                    } else {
                        texts[i].push_str(&self.vocabulary[k]);
                        emissions[i].push(TokenEmission { token_id: k, frame: t });
                        prev[i] = k;
                        commit_lanes[i] = true;
                        any_commit = true;
                    }
                }
                if any_commit {
                    step.commit(&commit_lanes);
                }
            }
        }

        Ok(texts.into_iter().zip(emissions).collect())
    }

    /// Label-looping batched greedy decode (NeMo arXiv 2406.06220 adapted to
    /// lane waves): the joint runs every step at per-lane frame indices; the
    /// predictor runs only after a round with at least one non-blank emission.
    /// Per-lane greedy decisions are byte-identical to [`Self::decode_batch`]
    /// (greedy is lane-independent and `max_symbols_per_step` is enforced as a
    /// per-frame run length) — only the call schedule changes: ~equal joint
    /// count, predictor count drops to the emission rounds.
    pub fn decode_batch_labels<S: BatchLabelStep>(
        &self,
        valid_frames: &[usize],
        step: &mut S,
    ) -> Result<Vec<LaneDecode>, RnntDecodeError<S::Error>> {
        let b = valid_frames.len();
        let lanes = step.batch();
        assert!(b <= lanes, "decode_batch_labels: {b} items exceed the backend's {lanes} lanes");
        let blank_id = self.blank_id();
        let max_symbols = self.opts.max_symbols_per_step.max(1);
        let mut texts = vec![String::new(); b];
        let mut emissions = vec![Vec::new(); b];
        if self.vocabulary.is_empty() {
            return Ok(texts.into_iter().zip(emissions).collect());
        }

        step.reset();
        let mut prev = vec![blank_id; lanes];
        let mut time = vec![0usize; lanes];
        let mut symbols = vec![0usize; lanes]; // run length at the current frame
        let mut active = vec![false; lanes];
        let mut out = vec![blank_id; lanes];
        let mut commit_lanes = vec![false; lanes];

        // Empty-prefix predictor output for the first joint round.
        step.predict(&prev).map_err(|e| BackendSnafu { frame: 0usize }.into_error(e))?;

        loop {
            let mut any_active = false;
            for i in 0..lanes {
                active[i] = i < b && time[i] < valid_frames[i];
                any_active |= active[i];
            }
            if !any_active {
                break;
            }
            step.joint(&time, &active, &mut out).map_err(|e| BackendSnafu { frame: time[0] }.into_error(e))?;
            let mut any_commit = false;
            for i in 0..b {
                commit_lanes[i] = false;
                if !active[i] {
                    continue;
                }
                let k = out[i];
                if k == blank_id {
                    time[i] += 1;
                    symbols[i] = 0;
                } else {
                    texts[i].push_str(&self.vocabulary[k]);
                    emissions[i].push(TokenEmission { token_id: k, frame: time[i] });
                    prev[i] = k;
                    commit_lanes[i] = true;
                    any_commit = true;
                    symbols[i] += 1;
                    if symbols[i] >= max_symbols {
                        time[i] += 1;
                        symbols[i] = 0;
                    }
                }
            }
            if any_commit {
                // Promote the emitting lanes' tentative state FIRST — the next
                // predictor round must consume the post-emission committed
                // state (mirrors lockstep's step-then-commit order; predicting
                // first re-reads the pre-emission state and drops tokens).
                step.commit(&commit_lanes);
                step.predict(&prev).map_err(|e| BackendSnafu { frame: time[0] }.into_error(e))?;
            }
        }

        Ok(texts.into_iter().zip(emissions).collect())
    }

    /// Device-block greedy decode: the entire label-looping loop runs on the
    /// device in [`BatchBlockStep::block_steps`]-sized blocks; the host only
    /// appends tape entries flagged by `emit`. Per-lane output is identical to
    /// [`Self::decode_batch_labels`] / [`Self::decode_batch`].
    pub fn decode_batch_blocks<S: BatchBlockStep>(
        &self,
        valid_frames: &[usize],
        step: &mut S,
    ) -> Result<Vec<LaneDecode>, RnntDecodeError<S::Error>> {
        let b = valid_frames.len();
        let lanes = step.batch();
        assert!(b <= lanes, "decode_batch_blocks: {b} items exceed the backend's {lanes} lanes");
        if self.vocabulary.is_empty() || valid_frames.iter().all(|&v| v == 0) {
            return Ok(vec![(String::new(), Vec::new()); b]);
        }
        step.reset().map_err(|e| BackendSnafu { frame: 0usize }.into_error(e))?;
        self.run_blocks(valid_frames, step)
    }

    /// [`Self::decode_batch_blocks`] from the lanes' present state instead of
    /// a reset: a chunk of a stream continues where its last chunk stopped,
    /// the backend having set every lane's frame cursor to the chunk's start.
    /// Emission frames are relative to the chunk.
    pub fn resume_batch_blocks<S: BatchBlockStep>(
        &self,
        valid_frames: &[usize],
        step: &mut S,
    ) -> Result<Vec<LaneDecode>, RnntDecodeError<S::Error>> {
        let b = valid_frames.len();
        assert!(b <= step.batch(), "resume_batch_blocks: {b} items exceed the backend's {} lanes", step.batch());
        if self.vocabulary.is_empty() || valid_frames.iter().all(|&v| v == 0) {
            return Ok(vec![(String::new(), Vec::new()); b]);
        }
        self.run_blocks(valid_frames, step)
    }

    fn run_blocks<S: BatchBlockStep>(
        &self,
        valid_frames: &[usize],
        step: &mut S,
    ) -> Result<Vec<LaneDecode>, RnntDecodeError<S::Error>> {
        let b = valid_frames.len();
        let k = step.block_steps();
        let mut texts = vec![String::new(); b];
        let mut emissions = vec![Vec::new(); b];
        loop {
            let tapes = step.run_block().map_err(|e| BackendSnafu { frame: 0usize }.into_error(e))?;
            for i in 0..b {
                for s in 0..k {
                    let j = i * k + s;
                    if tapes.emit[j] != 0 && (tapes.frames[j] as usize) < valid_frames[i] {
                        let tok = tapes.tokens[j] as usize;
                        texts[i].push_str(&self.vocabulary[tok]);
                        emissions[i].push(TokenEmission { token_id: tok, frame: tapes.frames[j] as usize });
                    }
                }
            }
            if !tapes.active_any {
                break;
            }
        }

        Ok(texts.into_iter().zip(emissions).collect())
    }
}
