use proptest::prelude::*;
use test_case::test_case;

use crate::diarization::{Binarization, SpeakerSegment, speaker_segments, write_rttm};

const FRAME: f32 = 0.5;

fn segments(binarization: Binarization, track: &[f32]) -> Vec<(f32, f32)> {
    binarization.segments(track.iter().copied(), FRAME)
}

#[test]
fn threshold_is_strict_and_runs_close_at_the_end() {
    let track = [0.2, 0.6, 0.5, 0.9, 0.1, 0.7, 0.8];
    // 0.5 is neither above onset nor below offset: it keeps the run going
    assert_eq!(segments(Binarization::default(), &track), vec![(0.5, 2.0), (2.5, 3.5)]);
}

#[test]
fn hysteresis_holds_between_thresholds() {
    let binarization = Binarization { onset: 0.7, offset: 0.3, ..Default::default() };
    let track = [0.5, 0.8, 0.5, 0.4, 0.2, 0.6, 0.9];
    assert_eq!(segments(binarization, &track), vec![(0.5, 2.0), (3.0, 3.5)]);
}

#[test]
fn inverted_thresholds_toggle_between_them() {
    // onset < offset: values strictly between flip the state (NeMo semantics)
    let binarization = Binarization { onset: 0.3, offset: 0.7, ..Default::default() };
    let track = [0.5, 0.5, 0.8, 0.5, 0.2];
    assert_eq!(segments(binarization, &track), vec![(0.0, 0.5), (1.0, 1.5)]);
}

#[test]
fn padding_widens_clamps_and_merges() {
    let binarization = Binarization { pad_onset: 0.75, pad_offset: 0.25, ..Default::default() };
    let track = [0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9];
    assert_eq!(segments(binarization, &track), vec![(0.0, 2.25), (2.75, 4.25)]);
}

/// Two 0.5 s blips 0.5 s apart: dropping short speech first loses both,
/// filling short gaps first joins them into one long enough to keep.
#[test_case(true, vec![]; "speech first")]
#[test_case(false, vec![(0.0, 1.5)]; "gaps first")]
fn duration_filters_follow_their_order(speech_first: bool, expected: Vec<(f32, f32)>) {
    let binarization = Binarization {
        min_duration_on: 1.0,
        min_duration_off: 1.0,
        filter_speech_first: speech_first,
        ..Default::default()
    };
    assert_eq!(segments(binarization, &[0.9, 0.1, 0.9]), expected);
}

#[test]
fn speakers_interleave_by_start() {
    let probs = [0.9, 0.1, 0.9, 0.9, 0.1, 0.9];
    let got = speaker_segments(&probs, 2, FRAME, &Binarization::default());
    assert_eq!(
        got,
        vec![SpeakerSegment { speaker: 0, start: 0.0, end: 1.0 }, SpeakerSegment { speaker: 1, start: 0.5, end: 1.5 },]
    );
}

#[test]
fn rttm_lines_carry_onset_and_duration() {
    let mut out = Vec::new();
    write_rttm(&mut out, "rec", &[SpeakerSegment { speaker: 3, start: 1.25, end: 2.0 }]).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "SPEAKER rec 1 1.250 0.750 <NA> <NA> speaker_3 <NA> <NA>\n");
}

proptest! {
    /// Default binarization is exactly the runs of frames above 0.5.
    #[test]
    fn default_binarization_is_runs_above_half(track in prop::collection::vec(0.0f32..1.0, 0..200)) {
        let mut expected = Vec::new();
        let mut start = None;
        for (i, &p) in track.iter().chain([0.0].iter()).enumerate() {
            match (start, p > 0.5) {
                (None, true) => start = Some(i),
                (Some(s), false) => {
                    expected.push((s as f32 * FRAME, i as f32 * FRAME));
                    start = None;
                }
                _ => {}
            }
        }
        prop_assert_eq!(segments(Binarization::default(), &track), expected);
    }

    /// Any binarization yields sorted, disjoint, in-bounds segments.
    #[test]
    fn segments_are_sorted_and_disjoint(
        track in prop::collection::vec(0.0f32..1.0, 0..200),
        onset in 0.0f32..1.0,
        offset in 0.0f32..1.0,
        pads in (0.0f32..2.0, 0.0f32..2.0),
        durations in (0.0f32..3.0, 0.0f32..3.0),
        speech_first: bool,
    ) {
        let binarization = Binarization {
            onset,
            offset,
            pad_onset: pads.0,
            pad_offset: pads.1,
            min_duration_on: durations.0,
            min_duration_off: durations.1,
            filter_speech_first: speech_first,
        };
        let got = segments(binarization, &track);
        let end = track.len() as f32 * FRAME + pads.1;
        for (s, e) in &got {
            prop_assert!(0.0 <= *s && s < e && *e <= end + 1e-4);
        }
        for pair in got.windows(2) {
            prop_assert!(pair[0].1 < pair[1].0);
        }
    }
}
