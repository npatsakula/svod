use proptest::prelude::*;
use test_case::test_case;

use crate::diarization::{ContextRow, SpeakerCache, SpeakerCacheConfig};

/// The id a silence row carries in [`Stream`].
const SILENCE: i64 = -1;

fn config(speakers: usize, cache_len: usize, fifo_len: usize, update_period: usize) -> SpeakerCacheConfig {
    SpeakerCacheConfig {
        num_speakers: speakers,
        cache_len,
        fifo_len,
        update_period,
        silence_frames_per_speaker: 1,
        score_threshold: 0.25,
        latest_frames_boost: 0.05,
        strong_boost_rate: 0.75,
        weak_boost_rate: 1.5,
        min_positive_rate: 0.5,
    }
}

/// A stream of frames numbered in arrival order, the cache's layouts applied
/// to them the way the model gathers embeddings: `context` holds the frame id
/// of every context row.
struct Stream {
    cache: SpeakerCache,
    context: Vec<i64>,
    pushed: i64,
}

impl Stream {
    fn new(config: SpeakerCacheConfig) -> Self {
        Self { cache: SpeakerCache::new(config).unwrap(), context: Vec::new(), pushed: 0 }
    }

    /// One step over `chunk` new frames, every step-input row given `row`.
    fn push(&mut self, chunk: usize, row: &[f32]) {
        let probs = row.repeat(self.context.len() + chunk);
        self.push_with(chunk, &probs);
    }

    fn push_with(&mut self, chunk: usize, probs: &[f32]) {
        let step: Vec<i64> = self.context.iter().copied().chain(self.pushed..self.pushed + chunk as i64).collect();
        self.cache.update(chunk, probs).unwrap();
        self.pushed += chunk as i64;
        self.context = self
            .cache
            .layout()
            .iter()
            .map(|row| match *row {
                ContextRow::Step(i) => step[i],
                ContextRow::Silence => SILENCE,
            })
            .collect();
        assert_eq!(self.context.len(), self.cache.context_frames());
    }
}

#[test]
fn frames_stay_in_fifo_until_it_overflows() {
    let mut stream = Stream::new(config(2, 16, 4, 3));
    for (cached, queued) in [(0, 2), (0, 4), (3, 3)] {
        stream.push(2, &[0.9, 0.1]);
        assert_eq!((stream.cache.cache_frames(), stream.cache.fifo_frames()), (cached, queued));
    }
    assert_eq!(stream.context, vec![0, 1, 2, 3, 4, 5]);
}

#[test_case(3, 2, 3; "update period dominates")]
#[test_case(1, 6, 6; "overflow dominates")]
#[test_case(10, 6, 10; "never more than queued")]
fn popped_frames_restore_fifo_capacity(update_period: usize, chunk: usize, expected_cache: usize) {
    let mut stream = Stream::new(config(2, 32, 4, update_period));
    stream.push(4, &[0.9, 0.1]);
    stream.push(chunk, &[0.9, 0.1]);
    assert_eq!(stream.cache.cache_frames(), expected_cache);
    assert_eq!(stream.cache.context_frames(), 4 + chunk);
}

#[test]
fn uncompressed_cache_probs_are_reestimated_every_pop() {
    let mut stream = Stream::new(config(2, 32, 0, 1));
    stream.push_with(2, &[0.9, 0.1, 0.8, 0.2]);
    assert_eq!(stream.cache.cache_probs(), &[0.9, 0.1, 0.8, 0.2]);
    // the next step re-scores the two cached frames before the new one
    stream.push_with(1, &[0.3, 0.7, 0.4, 0.6, 0.5, 0.5]);
    assert_eq!(stream.cache.cache_probs(), &[0.3, 0.7, 0.4, 0.6, 0.5, 0.5]);
}

/// Two speakers, a budget of 3 frames each (8 / 2 - 1 silence slot): speaker 0
/// talks in frames 0..5, speaker 1 in 5..9. Strong boosts go to each speaker's
/// best 2 frames, weak ones to its best 4; the latest frame (8) gets a small
/// bonus. The cache keeps both silence slots, then the 6 best frames, ties to
/// the earlier frame — grouped by speaker, in arrival order.
#[test]
fn compression_keeps_the_best_frames_per_speaker() {
    let mut stream = Stream::new(config(2, 8, 0, 9));
    let probs: Vec<f32> = (0..9).flat_map(|f| if f < 5 { [0.9, 0.1] } else { [0.1, 0.9] }).collect();
    stream.push_with(9, &probs);
    assert_eq!(stream.cache.cache_frames(), 8);
    assert_eq!(stream.context, vec![0, 1, 2, 3, SILENCE, 5, 8, SILENCE]);
    assert_eq!(&stream.cache.cache_probs()[4 * 2..5 * 2], &[0.0, 0.0]);

    // Compressed, the cache keeps its own probabilities: the next pop no
    // longer re-estimates them.
    let kept = stream.cache.cache_probs().to_vec();
    stream.push(1, &[0.6, 0.6]);
    assert_eq!(&stream.cache.cache_probs()[..2], &kept[..2]);
    assert_eq!(stream.context[0], 0);
}

#[test]
fn silent_speakers_leave_silence_slots() {
    // Nobody above 0.5: every slot becomes silence.
    let mut stream = Stream::new(config(2, 4, 0, 6));
    stream.push(6, &[0.2, 0.2]);
    assert_eq!(stream.context, vec![SILENCE; 4]);
}

#[test]
fn reset_empties_the_state() {
    let mut stream = Stream::new(config(2, 4, 0, 6));
    stream.push(6, &[0.9, 0.9]);
    stream.cache.reset();
    assert_eq!(stream.cache.context_frames(), 0);
    assert!(stream.cache.cache_probs().is_empty());
}

#[test_case(config(0, 8, 0, 1); "no speakers")]
#[test_case(config(4, 4, 0, 1); "no room besides silence")]
#[test_case(SpeakerCacheConfig { score_threshold: 1.0, ..config(2, 8, 0, 1) }; "threshold out of range")]
#[test_case(SpeakerCacheConfig { weak_boost_rate: f32::NAN, ..config(2, 8, 0, 1) }; "nan rate")]
fn invalid_configs_are_rejected(config: SpeakerCacheConfig) {
    assert!(SpeakerCache::new(config).is_err());
}

#[test]
fn probabilities_must_cover_the_step() {
    let mut cache = SpeakerCache::new(config(2, 8, 4, 1)).unwrap();
    assert!(cache.update(2, &[0.0; 3]).is_err(), "partial row");
    assert!(cache.update(2, &[0.0; 2]).is_err(), "short of the chunk");
}

proptest! {
    /// Over any sequence of chunks: the FIFO and the cache stay within their
    /// capacities, nothing is lost before the first compression, and every
    /// context row is a pushed frame or silence.
    #[test]
    fn capacities_and_provenance_hold(
        chunks in prop::collection::vec((1usize..12, prop::collection::vec(0.0f32..1.0, 2)), 1..40),
        fifo_len in 0usize..10,
        update_period in 1usize..12,
    ) {
        let mut stream = Stream::new(config(2, 12, fifo_len, update_period));
        let mut compressed = false;
        for (len, row) in chunks {
            stream.push(len, &row);
            prop_assert!(stream.cache.fifo_frames() <= fifo_len);
            prop_assert!(stream.cache.cache_frames() <= 12);
            prop_assert_eq!(stream.cache.cache_probs().len(), stream.cache.cache_frames() * 2);
            compressed |= (stream.context.len() as i64) < stream.pushed;
            if !compressed {
                prop_assert_eq!(&stream.context, &(0..stream.pushed).collect::<Vec<_>>());
            }
            for &id in &stream.context {
                prop_assert!(id == SILENCE || (0..stream.pushed).contains(&id));
            }
        }
    }
}
