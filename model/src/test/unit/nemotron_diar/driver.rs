//! The chunked driver on a tiny random model: streaming, batching and the
//! bucketed step plans must not change what a session computes.

use proptest::prelude::*;

use crate::nemotron_diar::{Diarizer, NemotronDiar, NemotronDiarConfig, Profile};

const SAMPLE_RATE: usize = 16000;
const SPEAKERS: usize = 2;

/// 2 layers, 16 wide, 16 mels stacked by 4, a 2-speaker head; a 16-frame
/// cache so long inputs compress it.
fn tiny_config(max_batch: usize) -> NemotronDiarConfig {
    let config = r#"{
        "audio_config": {"hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2,
            "num_hidden_layers": 2, "num_mel_bins": 16, "subsampling_factor": 4, "max_position_embeddings": 128,
            "rope_parameters": {"rope_theta": 10000.0}},
        "head_config": {"hidden_size": 8, "num_speakers": 2},
        "streaming_config": {"fifo_length": 6, "speaker_cache_update_period": 4, "speaker_cache_length": 16,
            "speaker_cache_silence_frames_per_speaker": 1, "prediction_score_threshold": 0.25,
            "latest_frames_score_boost": 0.05, "strong_boost_rate": 0.75, "weak_boost_rate": 1.5,
            "min_positive_scores_rate": 0.5},
        "chunk_length": 5, "chunk_right_context": 2, "fifo_length": 3, "speaker_cache_update_period": 4
    }"#;
    let processor = r#"{"feature_extractor": {"sampling_rate": 16000, "n_fft": 64, "hop_length": 16,
        "win_length": 48, "preemphasis": 0.97}}"#;
    let mut config = NemotronDiarConfig::from_json_strs(config, processor).unwrap();
    config.max_batch = max_batch;
    config
}

fn model(max_batch: usize) -> NemotronDiar {
    NemotronDiar::empty(tiny_config(max_batch))
}

fn audio(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(7);
    (0..len)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            0.3 * (i as f32 * 0.07 * (1.0 + (i / 400) as f32 % 3.0)).sin() + 0.2 * noise
        })
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn output_covers_every_mel_frame() {
    let mut diarizer = Diarizer::offline(model(1)).unwrap();
    for len in [16, 200, 1000, 3333] {
        let out = diarizer.diarize(&audio(len, 1), SAMPLE_RATE).unwrap();
        assert_eq!(out.frames(), len / 16 + 1, "length {len}");
        assert!(out.probs.iter().all(|p| (0.0..=1.0).contains(p)));
    }
}

#[test]
fn sample_rate_mismatch_is_rejected() {
    let mut diarizer = Diarizer::offline(model(1)).unwrap();
    assert!(diarizer.diarize(&audio(100, 1), 8000).is_err());
}

#[test]
fn finished_session_rejects_audio() {
    let diarizer = Diarizer::offline(model(1)).unwrap();
    let mut session = diarizer.session();
    session.finish();
    assert!(session.push(&[0.0; 4]).is_err());
}

/// Several recordings batched into one step plan match each run alone. Rows
/// of a batch can land in a larger bucket than alone, so the attention
/// reductions span different padding: equal up to rounding.
#[test]
fn batched_recordings_match_single_runs() {
    let audios: Vec<Vec<f32>> = [700, 2500, 4100].iter().enumerate().map(|(i, &n)| audio(n, i as u32)).collect();
    let weights = model(1);
    let mut wide = weights.clone();
    wide.config.max_batch = 3;
    let mut single = Diarizer::offline(weights).unwrap();
    let alone: Vec<Vec<f32>> = audios.iter().map(|a| single.diarize(a, SAMPLE_RATE).unwrap().probs).collect();
    let mut batched = Diarizer::offline(wide).unwrap();
    let refs: Vec<&[f32]> = audios.iter().map(Vec::as_slice).collect();
    for (got, want) in batched.diarize_batch(&refs, SAMPLE_RATE).unwrap().iter().zip(&alone) {
        assert!(max_diff(&got.probs, want) < 1e-5);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    /// Audio streamed in arbitrary pieces gives the probabilities of the whole
    /// recording run under the same profile.
    #[test]
    fn streaming_matches_offline(
        len in 300usize..4000,
        pieces in prop::collection::vec(1usize..700, 1..10),
        chunk_len in 1usize..6,
        right_context in 0usize..3,
    ) {
        let profile = Profile { chunk_len, right_context, fifo_len: 6, update_period: 4 };
        let mut diarizer = Diarizer::new(model(1), profile).unwrap();
        let signal = audio(len, len as u32);
        let offline = diarizer.diarize(&signal, SAMPLE_RATE).unwrap().probs;

        let mut session = diarizer.session();
        let mut streamed = Vec::new();
        let mut fed = 0;
        for piece in pieces.iter().cycle() {
            if fed == len {
                break;
            }
            let to = (fed + piece).min(len);
            session.push(&signal[fed..to]).unwrap();
            fed = to;
            diarizer.run(&mut [&mut session]).unwrap();
            streamed.extend(session.take_probs());
        }
        session.finish();
        diarizer.run(&mut [&mut session]).unwrap();
        streamed.extend(session.take_probs());
        prop_assert_eq!(streamed.len(), (len / 16 + 1) * SPEAKERS);
        prop_assert!(max_diff(&streamed, &offline) < 1e-5);
    }
}
