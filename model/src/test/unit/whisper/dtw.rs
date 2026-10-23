//! DTW alignment unit tests.

use svod_arch::pipelines::audio::Segment;

use crate::whisper::dtw;

#[test]
fn dtw_diagonal_path() {
    // Identity cost matrix: diagonal path [0,0], [1,1], ..., [n-1,n-1]
    let n = 5;
    let mut cost = vec![0.0f32; n * n];
    for i in 0..n {
        cost[i * n + i] = -1.0; // diagonal is cheapest
    }

    let (text, time) = dtw::dtw(&cost, n, n);
    assert_eq!(text.len(), time.len());
    assert_eq!(text.len(), 2 * n - 1); // DTW path length for diagonal

    // Path should end at (n-1, n-1) and start at (0, 0)
    assert_eq!(*text.first().unwrap(), 0);
    assert_eq!(*time.first().unwrap(), 0);
    assert_eq!(*text.last().unwrap(), n - 1);
    assert_eq!(*time.last().unwrap(), n - 1);
}

#[test]
fn dtw_asymmetric() {
    // 2 text tokens, 5 audio frames. Cost favors repeating text tokens.
    let n_rows = 2;
    let n_cols = 5;
    let cost = vec![
        -1.0, -1.0, -1.0, 0.0, 0.0, // token 0 aligns to frames 0-2
        0.0, 0.0, 0.0, -1.0, -1.0, // token 1 aligns to frames 3-4
    ];

    let (text, time) = dtw::dtw(&cost, n_rows, n_cols);
    assert_eq!(text.len(), time.len());
    // Path must visit all rows and columns
    assert!(text.contains(&0));
    assert!(text.contains(&1));
    assert!(time.contains(&0));
    assert!(time.contains(&4));
}

#[test]
fn median_filter_simple() {
    // Input: 1×5, filter_width=3, reflect padding
    let data = vec![1.0f32, 3.0, 2.0, 5.0, 4.0];
    let out = dtw::median_filter(&data, 1, 5, 3);
    // j=0: reflect(-1)=idx 1 → [3,1,3] → median 3
    // j=1: [1,3,2] → median 2
    // j=2: [3,2,5] → median 3
    // j=3: [2,5,4] → median 4
    // j=4: reflect(5)=2*4-5=3 → [5,4,5] → median 5
    assert_eq!(out, vec![3.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn median_filter_preserves_smooth() {
    let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let out = dtw::median_filter(&data, 1, 10, 3);
    // Already smooth — median should barely change it
    for i in 1..9 {
        assert!((out[i] - data[i]).abs() < 1.0, "median filter changed smooth data at {i}");
    }
}

#[test]
fn selected_alignment_ignores_compiled_padding() {
    let (heads, text_stride, audio_stride) = (2, 8, 6);
    let (valid_text, valid_audio) = (5, 4);
    let mut a = vec![0.0f32; heads * text_stride * audio_stride];
    let mut b = vec![1000.0f32; a.len()];
    for head in 0..heads {
        for text in 0..valid_text {
            for frame in 0..valid_audio {
                let index = (head * text_stride + text) * audio_stride + frame;
                let value = -((text as isize - frame as isize).abs() as f32) + head as f32 * 0.1;
                a[index] = value;
                b[index] = value;
            }
        }
    }

    let path_a = dtw::find_alignment_path_selected(&a, heads, text_stride, audio_stride, valid_text, valid_audio, 3, 1);
    let path_b = dtw::find_alignment_path_selected(&b, heads, text_stride, audio_stride, valid_text, valid_audio, 3, 1);
    assert_eq!(path_a, path_b);
}

#[test]
fn word_timings_accept_empty_alignment_path() {
    let timings = dtw::path_to_word_timings(&[], &[], &[0, 1], &["hello".to_string()], &[vec![1]], &[0.5], 50.0);
    assert!(timings.is_empty());
}

#[test]
fn word_probability_averages_decoder_token_probabilities() {
    let timings = dtw::path_to_word_timings(
        &[0, 1, 2],
        &[0, 1, 2],
        &[0, 2],
        &["hello".to_string()],
        &[vec![10, 11]],
        &[0.2, 0.8],
        50.0,
    );
    assert_eq!(timings.len(), 1);
    assert!((timings[0].probability - 0.5).abs() < 1e-6);
}

#[test]
fn missing_token_probabilities_do_not_produce_nan() {
    let timings = dtw::path_to_word_timings(&[0, 1], &[0, 1], &[0, 1], &["hello".to_string()], &[vec![10]], &[], 50.0);
    assert_eq!(timings[0].probability, 0.0);
}

#[test]
fn punctuation_attaches_without_changing_timing() {
    let timings = dtw::path_to_word_timings(
        &[0, 1, 2],
        &[0, 1, 2],
        &[0, 1, 2],
        &["hello".to_string(), "!".to_string()],
        &[vec![10], vec![11]],
        &[0.8, 0.9],
        50.0,
    );
    assert_eq!(timings[0].word, "hello!");
    assert_eq!(timings[0].tokens, [10, 11]);
    assert_eq!(timings[0].start, 0.02);
    assert_eq!(timings[0].end, 0.04);
    assert!(timings[1].word.is_empty());
}

#[test]
fn sentence_boundary_long_duration_is_truncated() {
    let mut words = vec![
        dtw::WordTiming { word: "hello".into(), tokens: vec![1], start: 0.0, end: 0.2, probability: 1.0 },
        dtw::WordTiming { word: ".".into(), tokens: vec![2], start: 0.2, end: 2.2, probability: 1.0 },
        dtw::WordTiming { word: " next".into(), tokens: vec![3], start: 2.2, end: 2.4, probability: 1.0 },
    ];
    dtw::clean_up_word_timings(&mut words);
    assert!((words[1].end - 0.6).abs() < 1e-6);
    assert_eq!(words[0].word, "hello.");
}

/// DTW row `0` is `<|notimestamps|>`; text token `b` begins at row `b + 1`.
#[test]
fn word_timings_start_at_the_first_text_row() {
    let text = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3];
    let time: Vec<usize> = (0..14).collect();
    let timings = dtw::path_to_word_timings(
        &text,
        &time,
        &[0, 2, 3],
        &["ab".to_string(), "c".to_string()],
        &[vec![1, 2], vec![3]],
        &[1.0; 3],
        50.0,
    );
    let spans: Vec<(f32, f32)> = timings.iter().map(|word| (word.start, word.end)).collect();
    assert_eq!(spans.len(), 2);
    for (actual, expected) in spans.iter().zip([(0.04, 0.20), (0.20, 0.28)]) {
        assert!((actual.0 - expected.0).abs() < 1e-6, "{spans:?}");
        assert!((actual.1 - expected.1).abs() < 1e-6, "{spans:?}");
    }
    // No word may begin where the `<|notimestamps|>` row begins.
    assert!(timings[0].start > 0.0);
}

#[test]
fn dtw_tie_breaks_match_the_reference() {
    // All-zero costs tie every move; the reference prefers diagonal, then up,
    // then left (strict comparisons), so the path hugs the first column.
    let (text, time) = dtw::dtw(&[0.0; 6], 2, 3);
    assert_eq!(text, [0, 1, 1, 1]);
    assert_eq!(time, [0, 0, 1, 2]);

    let (text, time) = dtw::dtw(&[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0], 3, 4);
    assert_eq!(text, [0, 0, 1, 2]);
    assert_eq!(time, [0, 1, 2, 3]);
}

#[test]
fn shifted_diagonal_alignment_is_golden() {
    // Text token `t` scores best at frame `t - 1`, so the DTW path walks the
    // shifted diagonal and the tail frames stay on the last text row.
    let (n_heads, text_stride, audio_stride) = (1, 6, 6);
    let (valid_text, valid_audio, sot_len, medfilt) = (6, 6, 1, 3);
    let mut qk = vec![0.0f32; n_heads * text_stride * audio_stride];
    for text in 0..text_stride {
        for frame in 0..audio_stride {
            qk[text * audio_stride + frame] = -((text as isize - frame as isize - 1).abs() as f32);
        }
    }
    let (text, time) = dtw::find_alignment_path_selected(
        &qk,
        n_heads,
        text_stride,
        audio_stride,
        valid_text,
        valid_audio,
        medfilt,
        sot_len,
    );
    assert_eq!(text, [0, 1, 1, 1, 2, 2, 3, 3, 3]);
    assert_eq!(time, [0, 0, 1, 2, 2, 3, 3, 4, 5]);
}

fn segment(start: f32, end: f32) -> Segment {
    Segment { text: String::new(), start, end }
}

fn timing(word: &str, start: f32, end: f32) -> dtw::WordTiming {
    dtw::WordTiming { word: word.into(), tokens: vec![1], start, end, probability: 1.0 }
}

#[test]
fn segment_start_clamps_a_long_first_word() {
    // 3.3 - 0.3 == 3.0: the segment start and the median-long word agree.
    let mut words = vec![timing("hello", 0.0, 3.3), timing(" there", 3.3, 3.5)];
    let carried = dtw::repair_segment_edges(&mut words, &[segment(3.0, 3.5)], 0.3, 3.2);
    assert_eq!(words[0].start, 3.0);
    assert_eq!(carried, 3.5);
}

#[test]
fn segment_start_clamp_keeps_a_median_long_word() {
    // `max(0, min(end - median, segment.start))`: with the word ending 0.2 s
    // after the segment start the reference shortens it to the median instead
    // of starting it at the segment start.
    let mut words = vec![timing("hello", 0.0, 3.2), timing(" there", 3.2, 3.5)];
    dtw::repair_segment_edges(&mut words, &[segment(3.0, 3.5)], 0.3, 3.2);
    assert!((words[0].start - 2.9).abs() < 1e-6);
}

#[test]
fn segment_end_clamps_a_long_last_word() {
    let mut words = vec![timing("a", 0.0, 0.2), timing(" b", 0.2, 0.5), timing(" c", 4.6, 9.98)];
    let carried = dtw::repair_segment_edges(&mut words, &[segment(0.0, 5.0)], 0.3, 0.0);
    assert_eq!(words[2].end, 5.0);
    assert_eq!(carried, 5.0);
}

#[test]
fn a_pause_before_the_first_word_carries_across_windows() {
    let words_before = vec![timing("hello", 0.0, 4.0), timing(" there", 4.0, 4.2)];
    let mut first = words_before.clone();
    // No speech before this window: the 4 s lead-in is clamped to twice the
    // median word duration.
    dtw::repair_segment_edges(&mut first, &[segment(0.0, 4.2)], 0.3, 0.0);
    assert!((first[0].start - 3.4).abs() < 1e-6);

    let mut next = words_before;
    // The previous window ended at 4.0 s, so there is no pause to repair.
    dtw::repair_segment_edges(&mut next, &[segment(0.0, 4.2)], 0.3, 4.0);
    assert_eq!(next[0].start, 0.0);
}

#[test]
fn word_boundaries_without_words_do_not_panic() {
    let words = ["hello".to_string()];
    assert!(dtw::path_to_word_timings(&[0, 1], &[0, 1], &[], &words, &[vec![1]], &[1.0], 50.0).is_empty());
    assert!(dtw::path_to_word_timings(&[0, 1], &[0, 1], &[0], &words, &[vec![1]], &[1.0], 50.0).is_empty());
}

fn diagonal_qk(text_stride: usize, audio_stride: usize) -> Vec<f32> {
    let mut qk = vec![0.0f32; text_stride * audio_stride];
    for text in 0..text_stride {
        for frame in 0..audio_stride {
            qk[text * audio_stride + frame] = -((text as isize - frame as isize - 1).abs() as f32);
        }
    }
    qk
}

#[test]
fn one_audio_frame_places_every_token_at_zero() {
    let (text_stride, audio_stride) = (6, 1);
    let qk = diagonal_qk(text_stride, audio_stride);
    let (text, time) = dtw::find_alignment_path_selected(&qk, 1, text_stride, audio_stride, 6, 1, 7, 1);
    assert_eq!(text, [0, 1, 2, 3]);
    assert!(time.iter().all(|&frame| frame == 0));

    let timings = dtw::path_to_word_timings(
        &text,
        &time,
        &[0, 2, 3],
        &["ab".to_string(), "c".to_string()],
        &[vec![1, 2], vec![3]],
        &[1.0; 3],
        50.0,
    );
    assert!(timings.iter().all(|word| word.start == 0.0));
    assert!(timings.iter().all(|word| word.end <= 1.0 / 50.0), "{timings:?}");
}

#[test]
fn two_audio_frames_are_not_clamped_to_the_edges() {
    // The seven-wide reflect padding would fold both frames onto one edge; the
    // narrowed filter leaves each cell alone instead.
    assert_eq!(dtw::median_filter(&[5.0, 1.0], 1, 2, 7), [5.0, 1.0]);
    assert_eq!(dtw::median_filter(&[5.0], 1, 1, 7), [5.0]);
    assert_eq!(dtw::median_filter(&[0.0, 100.0, 1.0], 1, 3, 7), [100.0, 1.0, 100.0]);

    let (text_stride, audio_stride) = (6, 2);
    let qk = diagonal_qk(text_stride, audio_stride);
    let (text, time) = dtw::find_alignment_path_selected(&qk, 1, text_stride, audio_stride, 6, 2, 7, 1);
    assert_eq!(text, [0, 1, 2, 3]);
    assert_eq!(time, [0, 1, 1, 1]);
}

#[test]
fn alignment_without_text_tokens_is_empty() {
    // `valid_text == sot_len + 2` is prompt, `<|notimestamps|>`, and EOT: no
    // text row to align.
    let qk = vec![0.0f32; 4];
    let (text, time) = dtw::find_alignment_path_selected(&qk, 1, 4, 1, 4, 3, 7, 2);
    assert!(text.is_empty() && time.is_empty());
    let (text, time) = dtw::find_alignment_path_selected(&qk, 1, 4, 1, 3, 3, 7, 1);
    assert!(text.is_empty() && time.is_empty());
}
