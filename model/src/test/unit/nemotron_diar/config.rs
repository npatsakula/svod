use test_case::test_case;

use super::{CONFIG_JSON, PROCESSOR_JSON};
use crate::audio::{MelLog, MelScale, PadMode};
use crate::nemotron_diar::{NemotronDiarConfig, Profile, StreamingMode};

fn config() -> NemotronDiarConfig {
    NemotronDiarConfig::from_json_strs(CONFIG_JSON, PROCESSOR_JSON).unwrap()
}

#[test]
fn published_config_parses() {
    let c = config();
    assert_eq!((c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_hidden_layers), (512, 2048, 8, 31));
    assert_eq!((c.num_mel_bins, c.subsampling_factor, c.head_hidden_size, c.num_speakers), (128, 8, 192, 8));
    assert_eq!(c.rope_theta, 10_000.0);
    assert_eq!(c.offline, Profile { chunk_len: 340, right_context: 40, fifo_len: 40, update_period: 300 });
    assert_eq!((c.cache.cache_len, c.cache.silence_frames_per_speaker), (264, 1));
    assert_eq!(c.frame_sec(), 0.01);
}

#[test]
fn mel_front_end_is_nemo_preprocessor() {
    let mel = config().mel_config();
    assert_eq!((mel.sample_rate, mel.n_fft, mel.win_length, mel.hop_length, mel.n_mels), (16000, 512, 400, 160, 128));
    assert!(mel.center && !mel.periodic);
    assert_eq!((mel.mel_scale, mel.pad_mode, mel.preemphasis), (MelScale::Slaney, PadMode::Zero, Some(0.97)));
    assert_eq!(mel.log, MelLog::LnAdd { guard: 2f64.powi(-24) });
}

#[test_case(StreamingMode::LowLatency, 9, 4, 541; "1.04 s")]
#[test_case(StreamingMode::VeryLowLatency, 6, 2, 536; "0.64 s")]
#[test_case(StreamingMode::UltraLowLatency, 3, 1, 532; "0.32 s")]
fn streaming_profiles_follow_the_model_card(mode: StreamingMode, chunk: usize, look_ahead: usize, capacity: usize) {
    let c = config();
    let profile = c.streaming_profile(mode);
    assert_eq!(profile, Profile { chunk_len: chunk, right_context: look_ahead, fifo_len: 264, update_period: 222 });
    assert_eq!(c.step_capacity(&profile), capacity);
    assert!(c.validate_profile(&profile).is_ok());
}

#[test]
fn offline_step_capacity() {
    let c = config();
    assert_eq!(c.step_capacity(&c.offline), 264 + 40 + 340 + 40);
}

#[test_case(Profile { chunk_len: 0, right_context: 4, fifo_len: 264, update_period: 222 }; "empty chunk")]
fn invalid_profiles_are_rejected(profile: Profile) {
    assert!(config().validate_profile(&profile).is_err());
}

/// The encoder pads a step to `SEQ_ALIGN`, so the RoPE table must cover the
/// padded step, not just the capacity: the published offline step is 684
/// frames, padded to 704.
#[test_case(704, true; "covers the padded step")]
#[test_case(703, false; "one position short")]
fn profiles_are_checked_at_the_padded_length(max_positions: usize, accepted: bool) {
    let json = CONFIG_JSON
        .replace("\"max_position_embeddings\": 5000", &format!("\"max_position_embeddings\": {max_positions}"));
    let parsed = NemotronDiarConfig::from_json_strs(&json, PROCESSOR_JSON);
    assert_eq!(parsed.is_ok(), accepted);
    if let Ok(c) = parsed {
        assert_eq!(c.step_capacity(&c.offline), 684);
    }
}

#[test]
fn zero_max_batch_is_rejected() {
    let mut c = config();
    c.max_batch = 0;
    assert!(c.validate_profile(&c.offline).is_err());
}

#[test]
fn grouped_query_checkpoints_are_rejected() {
    let gqa = CONFIG_JSON.replace("\"num_key_value_heads\": 8", "\"num_key_value_heads\": 4");
    assert!(NemotronDiarConfig::from_json_strs(&gqa, PROCESSOR_JSON).is_err());
    assert_eq!(config().num_key_value_heads, 8);
}

#[test]
fn malformed_configs_are_rejected() {
    assert!(NemotronDiarConfig::from_json_strs("{}", PROCESSOR_JSON).is_err());
    let odd_heads = CONFIG_JSON.replace("\"num_attention_heads\": 8", "\"num_attention_heads\": 7");
    assert!(NemotronDiarConfig::from_json_strs(&odd_heads, PROCESSOR_JSON).is_err());
}
