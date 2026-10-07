//! Per-frame speaker probabilities to speech segments: NeMo's
//! `binarization_vectorized` + `filtering` (`vad_utils.py`), the
//! post-processing of its end-to-end diarizers.

use std::io::{self, Write};

/// Thresholds and duration filters turning one speaker's probability track
/// into segments. Seconds throughout. The default is plain `> 0.5`
/// thresholding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Binarization {
    /// A frame above `onset` starts (or continues) speech.
    pub onset: f32,
    /// A frame below `offset` ends speech. With `onset >= offset`, frames in
    /// between keep the previous state; with `onset < offset`, frames strictly
    /// between them toggle it.
    pub offset: f32,
    /// Widen every segment by this much before its start (clamped at 0).
    pub pad_onset: f32,
    /// Widen every segment by this much after its end.
    pub pad_offset: f32,
    /// Drop speech segments shorter than this.
    pub min_duration_on: f32,
    /// Fill non-speech gaps shorter than this.
    pub min_duration_off: f32,
    /// Apply `min_duration_on` before `min_duration_off` (NeMo's default).
    pub filter_speech_first: bool,
}

impl Default for Binarization {
    fn default() -> Self {
        Self {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
            filter_speech_first: true,
        }
    }
}

/// One speaker's speech, `[start, end)` in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeakerSegment {
    pub speaker: usize,
    pub start: f32,
    pub end: f32,
}

impl Binarization {
    /// One speaker's track to `[start, end)` intervals in seconds, sorted.
    pub fn segments(&self, track: impl IntoIterator<Item = f32>, frame_sec: f32) -> Vec<(f32, f32)> {
        let mut segments = Vec::new();
        let (mut active, mut start) = (false, 0usize);
        let mut frames = 0;
        for (frame, p) in track.into_iter().enumerate() {
            let next = if self.onset >= self.offset {
                if p > self.onset {
                    true
                } else if p < self.offset {
                    false
                } else {
                    active
                }
            } else if p >= self.offset {
                true
            } else if p <= self.onset {
                false
            } else {
                !active
            };
            match (active, next) {
                (false, true) => start = frame,
                (true, false) => segments.push((start, frame)),
                _ => {}
            }
            active = next;
            frames = frame + 1;
        }
        if active {
            segments.push((start, frames));
        }

        let mut segments: Vec<(f32, f32)> = segments
            .into_iter()
            .map(|(s, e)| ((s as f32 * frame_sec - self.pad_onset).max(0.0), e as f32 * frame_sec + self.pad_offset))
            .filter(|(s, e)| e > s)
            .collect();
        if self.pad_onset > 0.0 || self.pad_offset > 0.0 {
            segments = merge_overlaps(segments);
        }
        self.filter(segments)
    }

    /// NeMo `filtering`: drop short speech and fill short gaps, in the
    /// configured order.
    fn filter(&self, mut segments: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
        let drop_short = |segments: Vec<(f32, f32)>| -> Vec<(f32, f32)> {
            if self.min_duration_on > 0.0 {
                segments.into_iter().filter(|(s, e)| e - s >= self.min_duration_on).collect()
            } else {
                segments
            }
        };
        let fill_gaps = |segments: Vec<(f32, f32)>| -> Vec<(f32, f32)> {
            if self.min_duration_off <= 0.0 || segments.is_empty() {
                return segments;
            }
            let gaps: Vec<(f32, f32)> =
                segments.windows(2).map(|w| (w[0].1, w[1].0)).filter(|(s, e)| e - s < self.min_duration_off).collect();
            merge_overlaps(segments.into_iter().chain(gaps).collect())
        };
        if self.filter_speech_first {
            segments = fill_gaps(drop_short(segments));
        } else {
            segments = drop_short(fill_gaps(segments));
        }
        segments
    }
}

/// Sort by start and merge intervals that overlap or touch.
fn merge_overlaps(mut segments: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
    segments.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f32, f32)> = Vec::with_capacity(segments.len());
    for (start, end) in segments {
        match merged.last_mut() {
            Some(last) if last.1 >= start => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// Segments of every speaker of a `[frames, num_speakers]` row-major
/// probability matrix whose rows are `frame_sec` apart, sorted by start then
/// speaker.
pub fn speaker_segments(
    probs: &[f32],
    num_speakers: usize,
    frame_sec: f32,
    binarization: &Binarization,
) -> Vec<SpeakerSegment> {
    assert!(num_speakers > 0 && probs.len().is_multiple_of(num_speakers), "probs must hold whole speaker rows");
    let mut segments: Vec<SpeakerSegment> = (0..num_speakers)
        .flat_map(|speaker| {
            let track = probs.iter().skip(speaker).step_by(num_speakers).copied();
            binarization.segments(track, frame_sec).into_iter().map(move |(start, end)| SpeakerSegment {
                speaker,
                start,
                end,
            })
        })
        .collect();
    segments.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.speaker.cmp(&b.speaker)));
    segments
}

/// RTTM `SPEAKER` lines for one recording, speakers named `speaker_<k>`.
pub fn write_rttm(mut out: impl Write, recording: &str, segments: &[SpeakerSegment]) -> io::Result<()> {
    for segment in segments {
        writeln!(
            out,
            "SPEAKER {recording} 1 {:.3} {:.3} <NA> <NA> speaker_{} <NA> <NA>",
            segment.start,
            segment.end - segment.start,
            segment.speaker
        )?;
    }
    Ok(())
}
