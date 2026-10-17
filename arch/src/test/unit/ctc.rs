use proptest::prelude::*;

use crate::ctc::{BeamDecoder, BeamOpts, CtcDecoder, DecodeError, GreedyDecoder, NanBehavior};

/// Vocab + blank index used across most tests. Indices: a=0, b=1, c=2, blank=3.
fn abc_vocab() -> Vec<String> {
    vec!["a".into(), "b".into(), "c".into()]
}

/// Build a single frame strongly favoring `winner` (logit 0.0; everything else -10.0).
fn frame_for(winner: usize, total_vocab: usize) -> Vec<f32> {
    let mut frame = vec![-10.0f32; total_vocab];
    frame[winner] = 0.0;
    frame
}

fn concat(frames: &[Vec<f32>]) -> Vec<f32> {
    frames.iter().flatten().copied().collect()
}

#[test]
fn test_greedy_simple() {
    let decoder = GreedyDecoder::new(abc_vocab());
    let total_vocab = decoder.total_vocab();
    let logits = concat(&[frame_for(0, total_vocab), frame_for(1, total_vocab), frame_for(2, total_vocab)]);
    assert_eq!(decoder.decode(&logits, 3, 3), "abc");
}

#[test]
fn test_greedy_collapse_runs() {
    let decoder = GreedyDecoder::new(abc_vocab());
    let total_vocab = decoder.total_vocab();
    let blank = decoder.blank_id();
    // a, a, blank, b, c, c, blank, c, a → "abcca".
    let frames = [
        frame_for(0, total_vocab),
        frame_for(0, total_vocab),
        frame_for(blank, total_vocab),
        frame_for(1, total_vocab),
        frame_for(2, total_vocab),
        frame_for(2, total_vocab),
        frame_for(blank, total_vocab),
        frame_for(2, total_vocab),
        frame_for(0, total_vocab),
    ];
    assert_eq!(decoder.decode(&concat(&frames), 9, 9), "abcca");
}

#[test]
fn test_greedy_drops_blank() {
    let decoder = GreedyDecoder::new(abc_vocab());
    let total_vocab = decoder.total_vocab();
    let blank = decoder.blank_id();
    let frames: Vec<Vec<f32>> = (0..5).map(|_| frame_for(blank, total_vocab)).collect();
    assert_eq!(decoder.decode(&concat(&frames), 5, 5), "");
}

#[test]
fn test_greedy_handles_nan_frame() {
    let decoder = GreedyDecoder::new(abc_vocab());
    let total_vocab = decoder.total_vocab();
    let frames = vec![frame_for(0, total_vocab), vec![f32::NAN; total_vocab], frame_for(1, total_vocab)];
    // NaN frame: argmax_nan_safe returns 0. 'a' is same as previous, collapsed.
    // t=2 produces 'b'. Final: "ab".
    assert_eq!(decoder.decode(&concat(&frames), 3, 3), "ab");
}

#[test]
fn test_greedy_empty_vocab() {
    let decoder = GreedyDecoder::new(Vec::new());
    let frames: Vec<Vec<f32>> = (0..3).map(|_| vec![1.0, 2.0, 3.0]).collect();
    assert_eq!(decoder.decode(&concat(&frames), 3, 3), "");
}

#[test]
fn test_greedy_zero_valid_frames() {
    let decoder = GreedyDecoder::new(abc_vocab());
    let total_vocab = decoder.total_vocab();
    let frames: Vec<Vec<f32>> = (0..3).map(|_| frame_for(0, total_vocab)).collect();
    assert_eq!(decoder.decode(&concat(&frames), 3, 0), "");
}

#[test]
fn test_greedy_batch_matches_per_item() {
    let decoder = GreedyDecoder::new(abc_vocab());
    let total_vocab = decoder.total_vocab();
    let blank = decoder.blank_id();
    let stride = 4;
    let item0 = concat(&[
        frame_for(0, total_vocab),
        frame_for(0, total_vocab),
        frame_for(blank, total_vocab),
        frame_for(1, total_vocab),
    ]);
    let item1 = concat(&[
        frame_for(2, total_vocab),
        frame_for(blank, total_vocab),
        frame_for(2, total_vocab),
        frame_for(blank, total_vocab),
    ]);
    let mut batched = item0.clone();
    batched.extend(item1.clone());
    let valid = [4usize, 3];
    let batch_out = decoder.decode_batch(&batched, stride, &valid);
    assert_eq!(batch_out.len(), 2);
    assert_eq!(batch_out[0], decoder.decode(&item0, stride, 4));
    assert_eq!(batch_out[1], decoder.decode(&item1, stride, 3));
}

#[test]
fn test_argmax_per_frame_simple() {
    let decoder = GreedyDecoder::new(abc_vocab());
    let total_vocab = decoder.total_vocab();
    let blank = decoder.blank_id();
    let frames = [frame_for(0, total_vocab), frame_for(2, total_vocab), frame_for(blank, total_vocab)];
    assert_eq!(decoder.argmax_per_frame(&concat(&frames), 3, 3), vec![0, 2, blank]);
}

#[test]
fn test_argmax_per_frame_clamps_valid_frames() {
    let decoder = GreedyDecoder::new(abc_vocab());
    let total_vocab = decoder.total_vocab();
    let frames = [frame_for(0, total_vocab), frame_for(1, total_vocab), frame_for(2, total_vocab)];
    assert_eq!(decoder.argmax_per_frame(&concat(&frames), 3, 2), vec![0, 1]);
}

#[test]
fn test_decode_with_timestamps_greedy() {
    let decoder = GreedyDecoder::new(abc_vocab());
    let total_vocab = decoder.total_vocab();
    let blank = decoder.blank_id();
    // a, a, blank, b, c, c, blank, c, a — emits at frames [0, 3, 4, 7, 8].
    // Run-collapse: a-a kept once at t=0; b at t=3; c at t=4; c at t=7 (re-emit
    // after blank); a at t=8.
    let frames = [
        frame_for(0, total_vocab),
        frame_for(0, total_vocab),
        frame_for(blank, total_vocab),
        frame_for(1, total_vocab),
        frame_for(2, total_vocab),
        frame_for(2, total_vocab),
        frame_for(blank, total_vocab),
        frame_for(2, total_vocab),
        frame_for(0, total_vocab),
    ];
    let (text, timestamps) = decoder.decode_with_timestamps(&concat(&frames), 9, 9);
    assert_eq!(text, "abcca");
    assert_eq!(timestamps, vec![0, 3, 4, 7, 8]);
}

#[test]
fn test_decode_with_timestamps_beam() {
    // Beam timestamps record the first frame at which a prefix was *speculatively*
    // added to the suffix tree, which is generally earlier than the greedy
    // commit-frame because the beam considers all tokens at every step. Greedy's
    // timestamps record the commit-frame.
    let mut decoder = BeamDecoder::new(abc_vocab(), BeamOpts::default());
    let greedy = GreedyDecoder::new(abc_vocab());
    let total_vocab = greedy.total_vocab();
    let blank = greedy.blank_id();
    let frames = [
        frame_for(0, total_vocab),
        frame_for(blank, total_vocab),
        frame_for(1, total_vocab),
        frame_for(blank, total_vocab),
        frame_for(2, total_vocab),
    ];
    let logits = concat(&frames);

    let (greedy_text, greedy_ts) = greedy.decode_with_timestamps(&logits, 5, 5);
    let (beam_text, beam_ts) = decoder.decode_with_timestamps(&logits, 5, 5).unwrap();

    assert_eq!(greedy_text, "abc");
    assert_eq!(greedy_ts, vec![0, 2, 4], "greedy: commit-frame per emitted token");
    assert_eq!(beam_text, "abc");
    // Beam: 'a' first explored at t=0 (root + 'a' extension), 'b' as a child of
    // 'a' first explored at t=1 (every beam entry tries every token), 'c' as a
    // child of 'ab' first explored at t=2.
    assert_eq!(beam_ts, vec![0, 1, 2], "beam: first-speculative-add frame per prefix node");
}

#[test]
fn test_beam_finds_better_path() {
    // Vocab: a=0, b=1, blank=2.
    let mut decoder = BeamDecoder::new(vec!["a".into(), "b".into()], BeamOpts::default());
    let frames = [
        vec![-0.1f32, -10.0, -10.0], // t=0: a
        vec![-1.0f32, -10.0, -0.5],  // t=1: blank just edges out a
        vec![-0.1f32, -10.0, -10.0], // t=2: a
    ];
    let logits = concat(&frames);

    let greedy = GreedyDecoder::new(decoder.vocabulary().to_vec());
    assert_eq!(greedy.decode(&logits, 3, 3), "aa");

    let beam_out = decoder.decode(&logits, 3, 3).unwrap();
    assert_eq!(beam_out, "aa");
}

#[test]
fn test_beam_size_token_pruning_safe() {
    let blank = 3usize;
    let total_vocab = blank + 1;
    let mut frame_a = vec![-50.0f32; total_vocab];
    frame_a[0] = 0.0;
    frame_a[blank] = -1.0;
    let mut frame_b = vec![-50.0f32; total_vocab];
    frame_b[1] = 0.0;
    frame_b[blank] = -1.0;
    let frames = [frame_a.clone(), frame_b.clone(), frame_a, frame_b];
    let logits = concat(&frames);

    let mut no_prune = BeamDecoder::new(abc_vocab(), BeamOpts::default());
    let mut with_prune = BeamDecoder::new(
        abc_vocab(),
        BeamOpts { beam_size: 100, beam_size_token: Some(2), beam_threshold: Some(20.0), on_nan: NanBehavior::Skip },
    );
    assert_eq!(no_prune.decode(&logits, 4, 4).unwrap(), with_prune.decode(&logits, 4, 4).unwrap());
}

#[test]
fn test_beam_zero_size_returns_empty() {
    let mut decoder = BeamDecoder::new(
        abc_vocab(),
        BeamOpts { beam_size: 0, beam_size_token: None, beam_threshold: None, on_nan: NanBehavior::Skip },
    );
    let total_vocab = decoder.total_vocab();
    let frames: Vec<Vec<f32>> = (0..3).map(|_| frame_for(0, total_vocab)).collect();
    assert_eq!(decoder.decode(&concat(&frames), 3, 3).unwrap(), "");
}

#[test]
fn test_beam_scratch_reuse_across_calls() {
    // Hammer the same decoder twice with different inputs and confirm both
    // outputs are correct — exercises tree.reset() between runs.
    let mut decoder = BeamDecoder::new(abc_vocab(), BeamOpts::default());
    let total_vocab = decoder.total_vocab();
    let blank = decoder.blank_id();

    let logits1 = concat(&[frame_for(0, total_vocab), frame_for(1, total_vocab)]);
    let logits2 = concat(&[frame_for(2, total_vocab), frame_for(blank, total_vocab), frame_for(2, total_vocab)]);

    assert_eq!(decoder.decode(&logits1, 2, 2).unwrap(), "ab");
    assert_eq!(decoder.decode(&logits2, 3, 3).unwrap(), "cc");
}

#[test]
fn test_ctc_decoder_enum_dispatch() {
    let total_vocab = 4;
    let logits = concat(&[frame_for(0, total_vocab), frame_for(1, total_vocab)]);

    let mut greedy = CtcDecoder::Greedy(GreedyDecoder::new(abc_vocab()));
    assert_eq!(greedy.decode(&logits, 2, 2).unwrap(), "ab");

    let mut beam = CtcDecoder::Beam(Box::new(BeamDecoder::new(abc_vocab(), BeamOpts::default())));
    assert_eq!(beam.decode(&logits, 2, 2).unwrap(), "ab");
}

#[test]
fn test_beam_nan_skip_default() {
    let mut decoder = BeamDecoder::new(abc_vocab(), BeamOpts::default());
    let total_vocab = decoder.total_vocab();
    let frames = vec![frame_for(0, total_vocab), vec![f32::NAN; total_vocab], frame_for(1, total_vocab)];
    let out = decoder.decode(&concat(&frames), 3, 3).unwrap();
    // NaN frame skipped; result should be "ab" (a then b, blank not needed
    // since the NaN frame doesn't contribute to either prefix's blank/no-blank
    // mass).
    assert_eq!(out, "ab");
}

#[test]
fn test_beam_nan_error_mode() {
    let mut decoder = BeamDecoder::new(
        abc_vocab(),
        BeamOpts { beam_size: 100, beam_size_token: None, beam_threshold: Some(20.0), on_nan: NanBehavior::Error },
    );
    let total_vocab = decoder.total_vocab();
    let frames = vec![frame_for(0, total_vocab), vec![f32::NAN; total_vocab], frame_for(1, total_vocab)];
    let err = decoder.decode(&concat(&frames), 3, 3).unwrap_err();
    match err {
        DecodeError::NanInLogits { frame } => assert_eq!(frame, 1),
    }
}

#[test]
fn test_beam_nan_propagate_mode() {
    let mut decoder = BeamDecoder::new(
        abc_vocab(),
        BeamOpts { beam_size: 100, beam_size_token: None, beam_threshold: Some(20.0), on_nan: NanBehavior::Propagate },
    );
    let total_vocab = decoder.total_vocab();
    let frames = vec![frame_for(0, total_vocab), vec![f32::NAN; total_vocab], frame_for(1, total_vocab)];
    // Output is unspecified — just confirm it doesn't panic.
    let _ = decoder.decode(&concat(&frames), 3, 3);
}

#[test]
fn test_suffix_tree_round_trip_via_beam() {
    // Indirectly tests SuffixTree::add_node / get_child / iter_from by way of
    // building a known prefix through beam decode and asserting the walk-back
    // matches input order.
    let mut decoder = BeamDecoder::new(abc_vocab(), BeamOpts::default());
    let total_vocab = decoder.total_vocab();
    let blank = decoder.blank_id();
    let frames = [
        frame_for(0, total_vocab),
        frame_for(blank, total_vocab),
        frame_for(1, total_vocab),
        frame_for(blank, total_vocab),
        frame_for(2, total_vocab),
        frame_for(blank, total_vocab),
        frame_for(0, total_vocab),
    ];
    let (text, frames_ts) = decoder.decode_with_timestamps(&concat(&frames), 7, 7).unwrap();
    assert_eq!(text, "abca");
    // Timestamps are first-speculative-add frames, which depend on beam
    // scheduling — we only assert the structural invariants:
    //  - one timestamp per emitted character,
    //  - each timestamp lies within the input range,
    //  - timestamps are monotonically non-decreasing (a child node was added
    //    no earlier than its parent).
    assert_eq!(frames_ts.len(), text.chars().count());
    for &t in &frames_ts {
        assert!(t < 7, "timestamp {t} out of range for 7 frames");
    }
    for w in frames_ts.windows(2) {
        assert!(w[0] <= w[1], "timestamps not monotonic: {frames_ts:?}");
    }
}

#[cfg(feature = "serde")]
#[test]
fn test_deserialize_greedy_canonical() {
    let json = r#"{"type": "greedy", "vocabulary": ["a", "b", "c"]}"#;
    let dec: CtcDecoder = serde_json::from_str(json).unwrap();
    match dec {
        CtcDecoder::Greedy(d) => {
            assert_eq!(d.vocabulary(), &["a".to_string(), "b".to_string(), "c".to_string()]);
            assert_eq!(d.blank_id(), 3);
        }
        _ => panic!("expected Greedy"),
    }
}

#[cfg(feature = "serde")]
#[test]
fn test_deserialize_beam_canonical() {
    let json = r#"{
        "type": "beam",
        "vocabulary": ["a", "b"],
        "beam_size": 50,
        "beam_threshold": 15.0,
        "on_nan": "error"
    }"#;
    let dec: CtcDecoder = serde_json::from_str(json).unwrap();
    match dec {
        CtcDecoder::Beam(d) => {
            assert_eq!(d.vocabulary(), &["a".to_string(), "b".to_string()]);
            assert_eq!(d.blank_id(), 2);
            assert_eq!(d.opts().beam_size, 50);
            assert_eq!(d.opts().beam_threshold, Some(15.0));
            // beam_size_token omitted in JSON → falls back to default Some(8).
            assert_eq!(d.opts().beam_size_token, Some(8));
            assert_eq!(d.opts().on_nan, NanBehavior::Error);
        }
        _ => panic!("expected Beam"),
    }
}

#[cfg(feature = "serde")]
#[test]
fn test_deserialize_beam_uses_defaults_when_omitted() {
    let json = r#"{"type": "beam", "vocabulary": ["a"]}"#;
    let dec: CtcDecoder = serde_json::from_str(json).unwrap();
    match dec {
        CtcDecoder::Beam(d) => {
            assert_eq!(d.opts().beam_size, 100);
            assert_eq!(d.opts().beam_threshold, Some(20.0));
            assert_eq!(d.opts().beam_size_token, Some(8));
            assert_eq!(d.opts().on_nan, NanBehavior::Skip);
        }
        _ => panic!("expected Beam"),
    }
}

proptest! {
    /// When per-frame argmax has a clear margin, beam search of any beam size
    /// should agree with greedy on the decoded string.
    #[test]
    fn prop_beam_matches_greedy_on_clear_margin(
        winners in proptest::collection::vec(0usize..4, 1..16),
    ) {
        let total_vocab = 4; // a, b, c, blank
        let frames: Vec<Vec<f32>> = winners.iter().map(|&w| frame_for(w, total_vocab)).collect();
        let logits = concat(&frames);
        let n = winners.len();

        let greedy = GreedyDecoder::new(abc_vocab());
        let greedy_out = greedy.decode(&logits, n, n);

        for &beam_size in &[1usize, 2, 8] {
            let mut beam = BeamDecoder::new(
                abc_vocab(),
                BeamOpts {
                    beam_size,
                    beam_size_token: None,
                    beam_threshold: Some(20.0),
                    on_nan: NanBehavior::Skip,
                },
            );
            let beam_out = beam.decode(&logits, n, n).unwrap();
            prop_assert_eq!(
                beam_out.clone(),
                greedy_out.clone(),
                "beam_size={} disagreed with greedy on winners={:?}",
                beam_size,
                winners
            );
        }
    }
}

proptest! {
    /// Collapsing a label sequence run by run, the last label carried, emits
    /// what the whole sequence's timestamped decode does.
    #[test]
    fn collapse_carries_the_label_across_runs(
        labels in prop::collection::vec(0usize..4, 1..40),
        cuts in prop::collection::vec(0usize..40, 0..4),
    ) {
        let decoder = GreedyDecoder::new(abc_vocab());
        let total = decoder.total_vocab();
        let logits = concat(&labels.iter().map(|&l| frame_for(l, total)).collect::<Vec<_>>());
        let (text, frames) = decoder.decode_with_timestamps(&logits, labels.len(), labels.len());
        let mut cuts: Vec<usize> = cuts.into_iter().map(|c| c.min(labels.len())).collect();
        cuts.push(labels.len());
        cuts.sort_unstable();
        let (mut prev, mut start, mut got) = (decoder.blank_id(), 0, Vec::new());
        for cut in cuts {
            let run = &labels[start..cut];
            got.extend(decoder.collapse(run, &mut prev).into_iter().map(|(token, at)| (token, start + at)));
            start = cut;
        }
        let want: Vec<(usize, usize)> = text
            .chars()
            .zip(&frames)
            .map(|(c, &f)| (abc_vocab().iter().position(|v| v == &c.to_string()).unwrap(), f))
            .collect();
        prop_assert_eq!(got, want);
    }
}
