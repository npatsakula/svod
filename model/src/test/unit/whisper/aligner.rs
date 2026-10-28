//! Alignment lane construction and DTW word-timing tests.

use crate::whisper::aligner::{WhisperAligner, lane_text_tokens};
use crate::whisper::dtw;
use crate::whisper::{ModelDimensions, Whisper, WhisperSize};

#[test]
fn probabilities_follow_their_tokens_through_timestamp_filtering() {
    // 50364 is a timestamp token: dropping it must drop its probability too.
    let (tokens, probabilities) = lane_text_tokens(&[10, 11, 50364, 12], &[0.9, 0.8, 0.7, 0.6], 50256, 8).unwrap();
    assert_eq!(tokens, [10, 11, 12]);
    assert_eq!(probabilities, [0.9, 0.8, 0.6]);

    let timings = dtw::path_to_word_timings(
        &[0, 1, 2],
        &[0, 1, 2],
        &[0, 2, 3],
        &["ab".to_string(), "c".to_string()],
        &[vec![10, 11], vec![12]],
        &probabilities,
        50.0,
    );
    assert_eq!(timings.len(), 2);
    assert!((timings[0].probability - 0.85).abs() < 1e-6);
    assert!((timings[1].probability - 0.6).abs() < 1e-6);
}

#[test]
fn a_lane_longer_than_the_prepared_context_is_an_error() {
    assert!(lane_text_tokens(&[10, 11, 12], &[0.9, 0.8, 0.7], 50256, 2).is_err());
    // Timestamp tokens do not consume the text context.
    assert!(lane_text_tokens(&[10, 50364, 11], &[0.9, 0.8, 0.7], 50256, 2).is_ok());
}

#[test]
fn missing_decoder_probabilities_become_zero() {
    let (tokens, probabilities) = lane_text_tokens(&[10, 11], &[0.9], 50256, 8).unwrap();
    assert_eq!(tokens, [10, 11]);
    assert_eq!(probabilities, [0.9, 0.0]);
}

#[test]
fn a_size_preset_that_does_not_match_the_model_is_rejected() {
    let model = Whisper::empty(ModelDimensions::for_size(WhisperSize::Tiny));
    assert!(WhisperAligner::new(model, WhisperSize::Base, 1, 16).is_err());
}
