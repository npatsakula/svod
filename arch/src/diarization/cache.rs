//! Arrival-Order Speaker Cache (AOSC) with its FIFO queue, the streaming state
//! of Sortformer diarizers (NeMo `SortformerModules.streaming_update` /
//! `_compress_spkcache`, transformers `Nemotron3DiarizationSpeakerCache`).
//!
//! Every chunk is encoded together with the cached frames `[cache | fifo]`.
//! The chunk's own frames then join the FIFO; once it overflows, its oldest
//! frames move to the speaker cache, and an overflowing cache is compressed:
//! each speaker keeps the frames that best characterize it, grouped by speaker
//! in arrival order, with a few learned silence frames per speaker.
//!
//! The cache holds no embeddings, only the bookkeeping: after every step it
//! says which rows of that step's input make up the next context
//! ([`SpeakerCache::layout`]), so the embeddings can stay wherever the model
//! keeps them (on the device) and be gathered there.

use std::cmp::Ordering;

use snafu::ensure;

use super::{InvalidCacheConfigSnafu, InvalidUpdateSnafu, Result};

/// Sizes and the compression score policy of a [`SpeakerCache`]. Frame counts
/// are in encoder frames.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeakerCacheConfig {
    pub num_speakers: usize,
    /// Speaker cache capacity.
    pub cache_len: usize,
    /// FIFO queue capacity.
    pub fifo_len: usize,
    /// Minimum number of frames moved from an overflowing FIFO to the cache.
    pub update_period: usize,
    /// Cache slots per speaker reserved for the silence embedding.
    pub silence_frames_per_speaker: usize,
    /// Probability floor of the frame scores' logs.
    pub score_threshold: f32,
    /// Score bonus of the frames just moved from the FIFO.
    pub latest_frames_boost: f32,
    /// Share of a speaker's budget whose best frames get the strong bonus.
    pub strong_boost_rate: f32,
    /// Share of a speaker's budget whose best frames get the weak bonus.
    pub weak_boost_rate: f32,
    /// Share of a speaker's budget of positive-score frames past which its
    /// non-positive frames are dropped.
    pub min_positive_rate: f32,
}

impl SpeakerCacheConfig {
    /// Cache slots of one speaker, excluding its silence slots.
    fn budget(&self) -> usize {
        self.cache_len / self.num_speakers - self.silence_frames_per_speaker
    }

    fn validate(&self) -> Result<()> {
        let invalid = |reason: &str| InvalidCacheConfigSnafu { reason: reason.to_string() };
        ensure!(self.num_speakers > 0, invalid("num_speakers must be positive"));
        ensure!(
            self.cache_len / self.num_speakers > self.silence_frames_per_speaker,
            invalid("every speaker needs a cache slot besides its silence slots")
        );
        for rate in [self.strong_boost_rate, self.weak_boost_rate, self.min_positive_rate] {
            ensure!(rate.is_finite() && rate >= 0.0, invalid("score rates must be finite and non-negative"));
        }
        ensure!(
            self.score_threshold > 0.0 && self.score_threshold < 1.0,
            invalid("score_threshold must lie in (0, 1)")
        );
        Ok(())
    }
}

/// Where a row of the next context comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextRow {
    /// That row of the input of the step just run.
    Step(usize),
    /// The learned silence embedding.
    Silence,
}

/// The streaming state of one audio stream; see the [module docs](self).
#[derive(Clone, Debug)]
pub struct SpeakerCache {
    config: SpeakerCacheConfig,
    min_positive: usize,
    strong_boosted: usize,
    weak_boosted: usize,
    /// The next context: cache rows, then FIFO rows.
    layout: Vec<ContextRow>,
    cache_frames: usize,
    /// Speaker probabilities of the cache rows, `num_speakers` per row.
    probs: Vec<f32>,
    /// Whether the cache was compressed at least once: its frames are then out
    /// of order, so only the probabilities stored with them describe them.
    compressed: bool,
}

impl SpeakerCache {
    pub fn new(config: SpeakerCacheConfig) -> Result<Self> {
        config.validate()?;
        let budget = config.budget() as f32;
        Ok(Self {
            min_positive: (budget * config.min_positive_rate).floor() as usize,
            strong_boosted: (budget * config.strong_boost_rate).floor() as usize,
            weak_boosted: (budget * config.weak_boost_rate).floor() as usize,
            layout: Vec::with_capacity(config.cache_len + config.fifo_len),
            cache_frames: 0,
            probs: Vec::with_capacity((config.cache_len + 1) * config.num_speakers),
            compressed: false,
            config,
        })
    }

    pub fn config(&self) -> &SpeakerCacheConfig {
        &self.config
    }

    /// Back to the empty state of a new stream.
    pub fn reset(&mut self) {
        self.layout.clear();
        self.cache_frames = 0;
        self.probs.clear();
        self.compressed = false;
    }

    pub fn cache_frames(&self) -> usize {
        self.cache_frames
    }

    pub fn fifo_frames(&self) -> usize {
        self.layout.len() - self.cache_frames
    }

    /// Frames the next chunk is preceded by: the cache, then the FIFO.
    pub fn context_frames(&self) -> usize {
        self.layout.len()
    }

    /// The next context's rows (cache, then FIFO) in terms of the input of
    /// the last [`update`](Self::update)d step, whose first
    /// [`context_frames`](Self::context_frames) rows were the context before
    /// it, followed by the chunk.
    pub fn layout(&self) -> &[ContextRow] {
        &self.layout
    }

    /// Cached speaker probabilities, one row per cache frame.
    pub fn cache_probs(&self) -> &[f32] {
        &self.probs
    }

    /// Push one processed step whose input was the current context followed by
    /// `num_chunk` chunk frames (and possibly look-ahead frames, which open the
    /// next chunk and stay out of the FIFO).
    ///
    /// `probs` holds the speaker probabilities of the step input, at the
    /// encoder frame rate and zeroed on its padding frames, covering at least
    /// the context and the chunk.
    pub fn update(&mut self, num_chunk: usize, probs: &[f32]) -> Result<()> {
        let speakers = self.config.num_speakers;
        let (num_cache, num_fifo) = (self.cache_frames(), self.fifo_frames());
        let queued = num_fifo + num_chunk;
        ensure!(
            probs.len().is_multiple_of(speakers) && probs.len() / speakers >= num_cache + queued,
            InvalidUpdateSnafu {
                reason: format!(
                    "{} probabilities do not cover the {} cached and chunk frames",
                    probs.len(),
                    num_cache + queued
                )
            }
        );

        // Before this update the context sat at the head of the step input.
        let popped = self.popped_frames(queued);
        let mut cache: Vec<ContextRow> = (0..num_cache + popped).map(ContextRow::Step).collect();
        if popped > 0 {
            if !self.compressed {
                // An uncompressed cache still holds plain past frames, whose
                // probabilities this step re-estimates.
                self.probs.clear();
                self.probs.extend_from_slice(&probs[..num_cache * speakers]);
            }
            self.probs.extend_from_slice(&probs[num_cache * speakers..(num_cache + popped) * speakers]);
            if cache.len() > self.config.cache_len {
                cache = self.compress(&cache);
                self.compressed = true;
            }
        }
        self.cache_frames = cache.len();
        self.layout = cache;
        self.layout.extend((num_cache + popped..num_cache + queued).map(ContextRow::Step));
        Ok(())
    }

    /// No frame moves until the FIFO overflows; then at least `update_period`
    /// of the oldest do, and as many more as restore its capacity.
    fn popped_frames(&self, queued: usize) -> usize {
        if queued <= self.config.fifo_len {
            return 0;
        }
        self.config.update_period.max(queued - self.config.fifo_len).min(queued)
    }

    /// Per-frame, per-speaker importance `[frames][speakers]`: high for a frame
    /// confidently attributed to that speaker alone, `-inf` for frames the
    /// speaker is silent in (and, for a speaker with enough positive frames,
    /// for its non-positive ones).
    fn frame_scores(&self) -> Vec<f32> {
        let speakers = self.config.num_speakers;
        let threshold = self.config.score_threshold;
        let half_log = 0.5f32.ln();
        let mut scores = vec![0.0f32; self.probs.len()];
        for (row, out) in self.probs.chunks(speakers).zip(scores.chunks_mut(speakers)) {
            let complement = |p: f32| (1.0 - p).max(threshold).ln();
            let complements: f32 = row.iter().map(|&p| complement(p)).sum();
            for (&p, score) in row.iter().zip(out.iter_mut()) {
                *score = if p > 0.5 {
                    p.max(threshold).ln() - complement(p) + complements - half_log
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
        for speaker in 0..speakers {
            let column = || scores.iter().skip(speaker).step_by(speakers);
            if column().filter(|&&s| s > 0.0).count() >= self.min_positive {
                for score in scores.iter_mut().skip(speaker).step_by(speakers) {
                    if *score <= 0.0 {
                        *score = f32::NEG_INFINITY;
                    }
                }
            }
        }
        scores
    }

    /// Keep `cache_len` of the `rows` (whose probabilities are `self.probs`):
    /// the best-scored ones per speaker, grouped by speaker in arrival order,
    /// padded with silence.
    fn compress(&mut self, rows: &[ContextRow]) -> Vec<ContextRow> {
        let (speakers, cache_len) = (self.config.num_speakers, self.config.cache_len);
        let frames = rows.len();
        let mut scores = self.frame_scores();
        // the frames past the capacity are the ones just moved from the FIFO
        for score in &mut scores[cache_len * speakers..] {
            *score += self.config.latest_frames_boost;
        }
        let half_log = 0.5f32.ln();
        for speaker in 0..speakers {
            for (count, boost) in [(self.strong_boosted, -2.0 * half_log), (self.weak_boosted, -half_log)] {
                let mut order: Vec<usize> = (0..frames).collect();
                let at = |f: usize| scores[f * speakers + speaker];
                let count = count.min(frames);
                if count > 0 {
                    order.select_nth_unstable_by(count - 1, |&a, &b| descending(at(a), at(b), a, b));
                }
                for &frame in &order[..count] {
                    scores[frame * speakers + speaker] += boost;
                }
            }
        }

        // Every speaker scores its frames plus its silence slots (`+inf`), in
        // one flat speaker-major ranking.
        let scored = frames + self.config.silence_frames_per_speaker;
        let flat = |i: usize| {
            let (speaker, frame) = (i / scored, i % scored);
            if frame < frames { scores[frame * speakers + speaker] } else { f32::INFINITY }
        };
        let mut order: Vec<usize> = (0..scored * speakers).collect();
        let kept = cache_len.min(order.len());
        order.select_nth_unstable_by(kept - 1, |&a, &b| descending(flat(a), flat(b), a, b));
        let mut kept: Vec<usize> =
            order[..kept].iter().map(|&i| if flat(i) == f32::NEG_INFINITY { usize::MAX } else { i }).collect();
        kept.sort_unstable();

        let old_probs = std::mem::take(&mut self.probs);
        kept.into_iter()
            .map(|index| {
                let frame = if index == usize::MAX { frames } else { (index % scored).min(frames) };
                if frame < frames {
                    self.probs.extend_from_slice(&old_probs[frame * speakers..(frame + 1) * speakers]);
                    rows[frame]
                } else {
                    self.probs.extend(std::iter::repeat_n(0.0, speakers));
                    ContextRow::Silence
                }
            })
            .collect()
    }
}

/// Highest score first; the earlier index wins a tie, so the ranking is
/// deterministic.
fn descending(a: f32, b: f32, ia: usize, ib: usize) -> Ordering {
    b.total_cmp(&a).then(ia.cmp(&ib))
}
