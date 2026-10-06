//! VAD-aware chunker for long-form ASR.
//!
//! Operates on `&[f32]` per-frame speech probabilities — the output of any
//! frame-level VAD — and packs them into bounded-length [`AudioChunk`]s
//! suitable for feeding to an encoder one chunk at a time. Speech-bearing
//! regions of the waveform are preserved; pure-silence regions between
//! chunks are dropped.
//!
//! The chunker is purely algorithmic: no Tensor or model dependency, no
//! coupling to a specific VAD. The output is sample-index ranges that any
//! downstream decoder can consume.
//!
//! # Algorithm
//!
//! ```text
//! 1. threshold + smoothing  → speech runs (prob-grid indices)
//! 2. split runs ≥ strict_limit at internal prob troughs (balanced toward
//!    geometric targets, preferring real silence via trough_threshold)
//! 3. greedy-pack runs into chunks of ~[min_duration, max_duration]
//!    (closing at inter-segment silence rather than mid-speech)
//! 4. prob indices → samples: the speech run is the text-owning core, with an
//!    optional gap pre-roll into the preceding silence; the decode window is
//!    the core padded for acoustic context and aligned to align_to
//! ```
//!
//! Each [`AudioChunk`] carries a non-overlapping **core** (`start..end`) and a
//! possibly-wider **decode window** (`decode_start..decode_end`). The core owns
//! output text; words decoded in the pad/pre-roll region are cropped back to
//! the core downstream, so extra context never duplicates text at a seam.
//!
//! All knobs live in [`ChunkerOpts`]; nothing inside the algorithm hardcodes
//! sample rates, prob granularity, or alignment.

pub(crate) mod segment;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use snafu::{Snafu, ensure};

use segment::threshold_segments;

// ─── Config ───────────────────────────────────────────────────────────────

/// Configuration for [`chunks_from_probs`].
///
/// All `*_duration` fields are wall-clock seconds; the chunker converts to
/// prob-grid indices via `(sample_rate, samples_per_prob)`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct ChunkerOpts {
    /// Sample rate of the source waveform in Hz.
    pub sample_rate: u32,
    /// Number of input samples covered by one entry of the `probs` array.
    /// Match the stride of the upstream frame-level VAD. Required so the
    /// chunker stays VAD-agnostic.
    pub samples_per_prob: usize,
    /// Speech threshold: prob entries `>= threshold` count as speech.
    pub threshold: f32,
    /// Soft minimum chunk duration. The chunker won't voluntarily close a
    /// chunk shorter than this.
    pub min_duration: f32,
    /// Soft maximum chunk duration. Past `min_duration`, the chunk closes
    /// at the next inter-segment silence (or, for a single long run, at a
    /// local prob trough) instead of extending past max.
    pub max_duration: f32,
    /// Soft target chunk duration (seconds) for deletion-sensitive autoregressive
    /// decoders (RN-T). When `Some(t)`, pack chunks toward `t` and re-split any
    /// still exceeding `1.5·t` at the nearest silence (via `split_long_runs`), so
    /// chunks cluster near `t` — short, locally-coherent contexts — instead of
    /// greedily filling to `max_duration`. `None` (default) keeps the
    /// fill-to-max behaviour. A good start is ~0.3·`max_duration`.
    ///
    /// Precedence with the other duration knobs: `t` supersedes `min_duration`
    /// for packing — the effective pack floor is `min(min_duration, t)` — so a
    /// target below `min_duration` legitimately yields shorter chunks. A target
    /// `>= max_duration` is a no-op (it clamps to the greedy fill-to-max path).
    /// Must be `> 0` when set (enforced by `validate`).
    pub target_duration: Option<f32>,
    /// Hard ceiling. A single VAD segment longer than this is split
    /// internally at prob-trough argmins so no output chunk exceeds it.
    /// Also caps chunk length when an under-min chunk would otherwise
    /// be extended past this.
    pub strict_limit_duration: f32,
    /// Pre-segmentation smoothing: a speech run must contain at least this
    /// many above-threshold probs to be retained.
    pub min_speech_probs: usize,
    /// Pre-segmentation smoothing: a silence gap must span at least this
    /// many below-threshold probs to terminate a speech run.
    pub min_silence_probs: usize,
    /// Two speech runs separated by ≤ this many silence probs are merged
    /// before chunking.
    pub merge_gap_probs: usize,
    /// Window radius (in prob-grid units) for the trough-argmin search when
    /// splitting overlong runs. `None` (default) reuses `min_silence_probs`,
    /// which is fine when smoothing tightness and trough-search width happen
    /// to want the same scale; set explicitly to decouple them.
    pub trough_search_probs: Option<usize>,
    /// Secondary threshold (typically lower than `threshold`) for
    /// `split_long_runs`. When `Some(t)`, search the full legal split
    /// range for the frame closest to the geometric target with prob
    /// `< t`; fall back to the narrow argmin around the target when no
    /// frame qualifies. `None` always uses narrow argmin.
    pub trough_threshold: Option<f32>,
    /// Symmetric pad in samples added to each chunk's **decode window** start/
    /// end (clamped at 0 and the implicit waveform end). Gives the encoder
    /// acoustic context at chunk boundaries; words decoded in the pad are
    /// cropped back to the core downstream so they never duplicate at a seam.
    pub pad_samples: usize,
    /// Max pre-roll (in samples) pulled into a chunk's **core** from the
    /// preceding silence gap. Capped at half the gap (cores stay disjoint) and
    /// at the remaining `strict_limit` headroom (a pre-rolled core never
    /// exceeds the hard cap or the decode-buffer bound). Moves the core-
    /// ownership boundary left so a boundary word ahead of the VAD-detected
    /// onset stays owned by this chunk instead of cropped away. `0` disables it.
    pub preroll_samples: usize,
    /// Snap chunk boundaries to integer multiples of this many samples.
    /// `1` = sample-precise. Set to the encoder's effective frame stride
    /// (e.g. `mel_hop * subsample_factor`) so chunks land on encoder-frame
    /// boundaries. Pathological values (e.g. > min_duration) are the
    /// caller's responsibility — boundaries can shift by up to
    /// `align_to - 1` samples.
    pub align_to: usize,
    /// True total sample count of the source waveform, when known. The prob
    /// grid rounds up past the real audio (the final VAD window is zero-padded),
    /// so `probs_len * samples_per_prob` overshoots the waveform end. Set this to
    /// `waveform.len()` to clamp every chunk's end to the real audio at the
    /// source; `None` (default) falls back to the prob-grid bound, so callers
    /// that don't set it must clamp downstream.
    pub max_total_samples: Option<usize>,
}

impl Default for ChunkerOpts {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            samples_per_prob: 512,
            threshold: 0.5,
            min_duration: 15.0,
            max_duration: 22.0,
            target_duration: None,
            strict_limit_duration: 30.0,
            min_speech_probs: 8,
            min_silence_probs: 4,
            merge_gap_probs: 8,
            trough_search_probs: None,
            trough_threshold: None,
            pad_samples: 0,
            preroll_samples: 0,
            align_to: 1,
            max_total_samples: None,
        }
    }
}

// ─── Output ───────────────────────────────────────────────────────────────

/// A speech-bearing region plus the waveform window used to decode it.
///
/// `start_sample..end_sample` is the non-overlapping **core** that owns output
/// text — derive `start_sec`/`end_sec` from it to offset per-chunk transcripts.
/// `decode_start_sample..decode_end_sample` is the possibly-wider **decode
/// window** fed to the encoder for acoustic context; decoded words whose
/// midpoint falls outside the core are cropped downstream, so the extra
/// context never duplicates text at a seam. Sample indices reference the
/// *original* waveform; an end may exceed the waveform length (the final VAD
/// window is zero-padded) — callers clamp at slice time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioChunk {
    pub start_sample: usize,
    pub end_sample: usize,
    pub decode_start_sample: usize,
    pub decode_end_sample: usize,
}

impl AudioChunk {
    /// Core-only chunk: the decode window equals the core (no extra context).
    pub fn new(start_sample: usize, end_sample: usize) -> Self {
        Self { start_sample, end_sample, decode_start_sample: start_sample, decode_end_sample: end_sample }
    }

    /// Chunk with an explicit decode window around the core.
    pub fn with_decode(
        start_sample: usize,
        end_sample: usize,
        decode_start_sample: usize,
        decode_end_sample: usize,
    ) -> Self {
        Self { start_sample, end_sample, decode_start_sample, decode_end_sample }
    }

    /// Length of the text-owning core, in samples.
    pub fn core_len(&self) -> usize {
        self.end_sample.saturating_sub(self.start_sample)
    }

    /// Length of the decode window (model input), in samples.
    pub fn decode_len(&self) -> usize {
        self.decode_end_sample.saturating_sub(self.decode_start_sample)
    }
}

// ─── Errors ───────────────────────────────────────────────────────────────

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("samples_per_prob must be > 0"))]
    ZeroSamplesPerProb,
    #[snafu(display("align_to must be > 0"))]
    ZeroAlignTo,
    #[snafu(display("min_duration ({min}) must be ≤ max_duration ({max})"))]
    MinExceedsMax { min: f32, max: f32 },
    #[snafu(display("max_duration ({max}) must be ≤ strict_limit_duration ({strict})"))]
    MaxExceedsStrict { max: f32, strict: f32 },
    #[snafu(display("target_duration ({target}) must be > 0 when set"))]
    TargetNonPositive { target: f32 },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Upper bound (in samples) on any chunk [`chunks_from_probs`] can emit:
/// `strict_limit + 2·trough_radius` (split_long_runs slack) `+ 2·pad +
/// 2·align_to` (post-process slack at waveform edges + alignment snap).
/// The `2·align_to` term covers both snaps a decode window takes in
/// `post_process`: `decode_start` floors and `decode_end` ceils, each
/// moving the boundary by up to `align_to − 1`. Single source of truth for
/// downstream callers that need to size buffers or assert the contract.
pub fn strict_chunk_sample_bound(
    strict_limit_probs: usize,
    trough_radius: usize,
    samples_per_prob: usize,
    pad_samples: usize,
    align_to: usize,
) -> usize {
    (strict_limit_probs + 2 * trough_radius) * samples_per_prob + 2 * pad_samples + 2 * align_to
}

// ─── Public entry point ───────────────────────────────────────────────────

/// Pack VAD speech probabilities into bounded-length chunks.
///
/// Output chunks cover only speech-bearing portions of the waveform; silence
/// between chunks is dropped. Boundaries are padded by `opts.pad_samples` and
/// snapped to `opts.align_to` multiples (start floored, end ceil'd, so
/// coverage is preserved). Adjacent chunks that overlap after padding are
/// merged.
pub fn chunks_from_probs(probs: &[f32], opts: &ChunkerOpts) -> Result<Vec<AudioChunk>> {
    validate(opts)?;
    if probs.is_empty() {
        return Ok(Vec::new());
    }

    let probs_per_sec = opts.sample_rate as f32 / opts.samples_per_prob as f32;
    let strict_limit_probs = (opts.strict_limit_duration * probs_per_sec).ceil() as usize;
    let min_probs = (opts.min_duration * probs_per_sec).ceil() as usize;
    let max_probs = (opts.max_duration * probs_per_sec).ceil() as usize;

    let trough_radius = opts.trough_search_probs.unwrap_or(opts.min_silence_probs);
    let trough_threshold = opts.trough_threshold;

    // Halve the silence-sensitivity knobs and retry if `threshold_segments`
    // produced any segment exceeding `strict_limit_probs` — gives `split_long_runs`
    // less work / more silence to cut at. Floor at 2 because a single
    // sub-threshold prob is reliably a VAD micro-dip mid-word, not silence.
    let mut adapted = opts.clone();
    let segments = loop {
        let segs = threshold_segments(probs, &adapted);
        let any_over = segs.iter().any(|&(s, e)| e - s > strict_limit_probs);
        if !any_over || adapted.min_silence_probs <= 2 {
            break segs;
        }
        adapted.min_silence_probs = (adapted.min_silence_probs / 2).max(2);
        adapted.merge_gap_probs = (adapted.merge_gap_probs / 2).max(1);
    };
    let segments = split_long_runs(segments, probs, trough_radius, trough_threshold, strict_limit_probs);

    // With a soft `target_duration`, pack chunks toward the target then re-split
    // any still exceeding 1.5·target at the silence nearest their midpoint
    // (reusing `split_long_runs` with narrow argmin — bound-safe since the cap
    // stays below `max ≤ strict`). Short, silence-aligned contexts stop an
    // autoregressive RN-T decoder from "skipping" (deleting) earlier tokens on
    // long chunks. `None` keeps the greedy fill-to-max behaviour.
    // Engage only for a real soft target below max; `>= max` clamps to nothing,
    // so it stays the greedy default.
    let target_probs =
        opts.target_duration.map(|d| (d * probs_per_sec).ceil() as usize).filter(|&t| t > 0 && t < max_probs);
    let chunks = match target_probs {
        Some(t) => {
            // Pack toward `t`, cap the re-split at 1.5·t (never above max). The
            // pack floor is min(min_duration, t): a target below min_duration
            // lowers the floor so chunks can actually reach the smaller target.
            let pack_cap = t.min(max_probs);
            let split_cap = (t + t / 2).min(max_probs);
            let packed = pack_segments(&segments, min_probs.min(pack_cap), pack_cap, strict_limit_probs);
            split_long_runs(packed, probs, trough_radius, None, split_cap)
        }
        None => pack_segments(&segments, min_probs, max_probs, strict_limit_probs),
    };

    Ok(post_process(&chunks, probs.len(), opts))
}

// ─── Internals ────────────────────────────────────────────────────────────

fn validate(opts: &ChunkerOpts) -> Result<()> {
    ensure!(opts.samples_per_prob != 0, ZeroSamplesPerProbSnafu);
    ensure!(opts.align_to != 0, ZeroAlignToSnafu);
    ensure!(
        opts.min_duration <= opts.max_duration,
        MinExceedsMaxSnafu { min: opts.min_duration, max: opts.max_duration }
    );
    ensure!(
        opts.max_duration <= opts.strict_limit_duration,
        MaxExceedsStrictSnafu { max: opts.max_duration, strict: opts.strict_limit_duration }
    );
    // A target >= max is a no-op (clamped in chunks_from_probs), but a
    // non-positive target is a config error rather than a silent no-op.
    if let Some(t) = opts.target_duration {
        ensure!(t > 0.0, TargetNonPositiveSnafu { target: t });
    }
    Ok(())
}

/// Break any speech segment whose length exceeds `strict_limit_probs` into
/// `ceil(len / strict_limit)` near-equal pieces, choosing each split point
/// as the prob argmin within ±`search_radius` of the geometric target. Lands
/// on natural pauses inside long unbroken runs instead of hard-cutting at
/// fixed time intervals.
///
/// Each emitted piece is at least `len / (2 * n)` long. Without that floor
/// a wide `search_radius` can let the argmin land arbitrarily close to a
/// split's neighbours and produce 1-prob shards that downstream code has to
/// special-case. With the floor the worst-case shrinkage is half the
/// average piece length.
fn split_long_runs(
    segments: Vec<(usize, usize)>,
    probs: &[f32],
    search_radius: usize,
    trough_threshold: Option<f32>,
    strict_limit_probs: usize,
) -> Vec<(usize, usize)> {
    if strict_limit_probs == 0 {
        return segments;
    }
    let mut out = Vec::with_capacity(segments.len());
    for (start, end) in segments {
        let len = end - start;
        if len <= strict_limit_probs {
            out.push((start, end));
            continue;
        }
        let n = len.div_ceil(strict_limit_probs);
        let min_piece = (len / (2 * n)).max(1);
        let mut cur = start;
        for k in 1..n {
            let target = start + (len * k) / n;
            let pieces_left = n - k;
            // Constrain the argmin window so this split is at least
            // min_piece away from cur and from `end - pieces_left * min_piece`
            // (i.e. each remaining piece can still hit min_piece).
            let lo_narrow = target.saturating_sub(search_radius).max(cur + min_piece);
            let hi_floor = end.saturating_sub(pieces_left * min_piece);
            let hi_narrow = (target + search_radius).min(hi_floor.saturating_sub(1));

            // With `trough_threshold`: prefer a real silence frame anywhere
            // in the legal range (closest to target for balance) over the
            // narrow-radius argmin which may land inside speech.
            let trough_split = trough_threshold.and_then(|t| {
                let lo_wide = cur + min_piece;
                let hi_wide = hi_floor.saturating_sub(1);
                if hi_wide < lo_wide {
                    return None;
                }
                let slice = &probs[lo_wide..=hi_wide];
                slice
                    .iter()
                    .enumerate()
                    .filter(|&(_, &p)| p < t)
                    .min_by_key(|(i, _)| (lo_wide + i).abs_diff(target))
                    .map(|(i, _)| lo_wide + i)
            });
            let split = if let Some(s) = trough_split {
                s
            } else if hi_narrow >= lo_narrow {
                lo_narrow + argmin(&probs[lo_narrow..=hi_narrow])
            } else {
                // Constraints incompatible (radius wider than the available
                // slack). Fall back to the geometric target, clamped so the
                // remaining pieces are still non-empty.
                target.clamp(cur + min_piece, hi_floor.saturating_sub(1).max(cur + min_piece))
            };
            if split > cur && split < end {
                out.push((cur, split));
                cur = split;
            }
        }
        if cur < end {
            out.push((cur, end));
        }
    }
    out
}

fn argmin(slice: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = slice[0];
    for (i, &v) in slice.iter().enumerate().skip(1) {
        if v < best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Greedy-concat speech segments into bounded-length chunks. Closes a chunk
/// when the next segment would push it past `max_probs` AND either the
/// current chunk has reached `min_probs` *or* extending would exceed
/// `strict_limit_probs` (the hard ceiling).
fn pack_segments(
    segments: &[(usize, usize)],
    min_probs: usize,
    max_probs: usize,
    strict_limit_probs: usize,
) -> Vec<(usize, usize)> {
    let mut chunks = Vec::new();
    let mut cur: Option<(usize, usize)> = None;
    for &(s, e) in segments {
        match cur {
            None => cur = Some((s, e)),
            Some((cs, ce)) => {
                let prospective = e - cs;
                let cur_len = ce - cs;
                if prospective > max_probs && (cur_len >= min_probs || prospective > strict_limit_probs) {
                    chunks.push((cs, ce));
                    cur = Some((s, e));
                } else {
                    cur = Some((cs, e));
                }
            }
        }
    }
    if let Some(c) = cur {
        chunks.push(c);
    }
    chunks
}

/// Convert prob-index core ranges to sample ranges and derive decode windows.
/// The core `[start, end]` owns output text; the decode window is the core
/// padded for acoustic context. Pre-roll pulls each post-silence core start
/// back into the preceding gap (bounded by half the gap and the strict-limit
/// headroom) so the first word after a pause isn't clipped. Pre-roll and pad
/// are both capped at half the gap to the neighbour, so cores stay disjoint
/// and adjacent decode windows overlap by at most `align_to - 1` samples
/// (filtered downstream by the core-crop, never duplicating text).
fn post_process(chunks: &[(usize, usize)], probs_len: usize, opts: &ChunkerOpts) -> Vec<AudioChunk> {
    // The prob grid overshoots the real audio (final window zero-padded). Clamp
    // to the true waveform length when the caller provided it.
    let grid = probs_len * opts.samples_per_prob;
    let max_sample = opts.max_total_samples.unwrap_or(grid).min(grid);
    let pad = opts.pad_samples;
    let align = opts.align_to;
    let spp = opts.samples_per_prob;

    // Hard cap on core length (incl. pre-roll): the strict-limit budget the
    // chunker packed to, so pre-roll never pushes a core past the
    // decode-buffer bound.
    let probs_per_sec = opts.sample_rate as f32 / spp as f32;
    let strict_limit_samples = (opts.strict_limit_duration * probs_per_sec).ceil() as usize * spp;

    // Pre-rolled core start for chunk `i`. Core ends are never moved, so each
    // adjusted start depends only on its own raw core and the previous core's
    // (clamped) end — independent of other chunks' pre-roll.
    let core_start_adj = |i: usize| -> usize {
        let (s, e) = chunks[i];
        let raw_start = s * spp;
        let core_end = (e * spp).min(max_sample);
        let gap_room = if i == 0 {
            raw_start
        } else {
            let prev_core_end = (chunks[i - 1].1 * spp).min(max_sample);
            raw_start.saturating_sub(prev_core_end) / 2
        };
        let headroom = strict_limit_samples.saturating_sub(core_end.saturating_sub(raw_start));
        raw_start - opts.preroll_samples.min(gap_room).min(headroom)
    };

    let mut out: Vec<AudioChunk> = Vec::with_capacity(chunks.len());
    for (i, &(_s, e)) in chunks.iter().enumerate() {
        let core_start = core_start_adj(i);
        let core_end = (e * spp).min(max_sample);
        if core_end <= core_start {
            continue;
        }

        // Decode-window pad capped at half the gap to the neighbouring
        // (pre-rolled) core — full margin at the waveform edges. Floor
        // division keeps adjacent raw decode windows from crossing a core.
        let pad_left = if i == 0 {
            pad.min(core_start)
        } else {
            let prev_core_end = (chunks[i - 1].1 * spp).min(max_sample);
            pad.min(core_start.saturating_sub(prev_core_end) / 2)
        };
        let pad_right = if i + 1 == chunks.len() {
            pad.min(max_sample.saturating_sub(core_end))
        } else {
            pad.min(core_start_adj(i + 1).saturating_sub(core_end) / 2)
        };

        let decode_start = ((core_start - pad_left) / align) * align;
        let mut decode_end = (core_end + pad_right).min(max_sample).div_ceil(align) * align;
        if decode_end > max_sample {
            decode_end = max_sample;
        }
        if decode_end <= decode_start {
            continue;
        }
        out.push(AudioChunk::with_decode(core_start, core_end, decode_start, decode_end));
    }
    out
}
