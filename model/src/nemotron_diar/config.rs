//! Nemotron-3-Diarization configuration: the HF `config.json` (encoder, head,
//! speaker-cache policy) and `processor_config.json` (mel front-end), plus the
//! chunking [`Profile`]s inference runs under.

use std::path::Path;

use serde::Deserialize;
use svod_arch::diarization::SpeakerCacheConfig;
use svod_dtype::DType;

use crate::audio::{MelConfig, MelLog, MelScale, PadMode};

use super::error::{ConfigSnafu, Result};
use super::model::SEQ_ALIGN;

/// NeMo's `log_zero_guard_value` for `log_zero_guard_type="add"`.
const LOG_GUARD: f64 = 1.0 / (1 << 24) as f64;

#[derive(Clone, Debug)]
pub struct NemotronDiarConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub preemphasis: f32,
    pub num_mel_bins: usize,
    /// Mel frames stacked into one encoder frame, and encoder frames upsampled
    /// back to mel frames by the head.
    pub subsampling_factor: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    /// Equal to `num_attention_heads`: the stacked QKV projection assumes it.
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub rope_theta: f64,
    /// Longest step the RoPE table covers.
    pub max_positions: usize,
    pub head_hidden_size: usize,
    pub num_speakers: usize,
    /// Chunking of whole recordings.
    pub offline: Profile,
    /// FIFO sizes of streaming sessions; their chunk sizes come from a
    /// [`StreamingMode`].
    pub streaming_fifo_len: usize,
    pub streaming_update_period: usize,
    /// Speaker-cache score policy, shared by every profile.
    pub cache: CachePolicy,
    /// Compute dtype of the encoder.
    pub dtype: DType,
    /// Streams one JIT step processes together.
    pub max_batch: usize,
}

/// How one session is cut into chunks, in encoder frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    /// Frames each step emits.
    pub chunk_len: usize,
    /// Look-ahead frames each step attends to past its chunk (they open the
    /// next chunk).
    pub right_context: usize,
    pub fifo_len: usize,
    pub update_period: usize,
}

/// The model card's streaming latencies: `(chunk, look-ahead)` of 80 ms frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StreamingMode {
    /// 1.04 s: chunk 9, look-ahead 4.
    #[default]
    LowLatency,
    /// 0.64 s: chunk 6, look-ahead 2.
    VeryLowLatency,
    /// 0.32 s: chunk 3, look-ahead 1.
    UltraLowLatency,
}

impl StreamingMode {
    pub fn chunk_sizes(self) -> (usize, usize) {
        match self {
            Self::LowLatency => (9, 4),
            Self::VeryLowLatency => (6, 2),
            Self::UltraLowLatency => (3, 1),
        }
    }
}

/// The speaker-cache fields of `streaming_config`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct CachePolicy {
    #[serde(rename = "speaker_cache_length")]
    pub cache_len: usize,
    #[serde(rename = "speaker_cache_silence_frames_per_speaker")]
    pub silence_frames_per_speaker: usize,
    #[serde(rename = "prediction_score_threshold")]
    pub score_threshold: f32,
    #[serde(rename = "latest_frames_score_boost")]
    pub latest_frames_boost: f32,
    pub strong_boost_rate: f32,
    pub weak_boost_rate: f32,
    #[serde(rename = "min_positive_scores_rate")]
    pub min_positive_rate: f32,
}

impl NemotronDiarConfig {
    /// Parse `config.json` and `processor_config.json`. The compute dtype
    /// defaults to f32 and the batch to 1.
    pub fn from_json_files(config: &Path, processor: &Path) -> Result<Self> {
        let read = |path: &Path| {
            std::fs::read_to_string(path)
                .map_err(|e| ConfigSnafu { message: format!("{}: {e}", path.display()) }.build())
        };
        Self::from_json_strs(&read(config)?, &read(processor)?)
    }

    pub fn from_json_strs(config: &str, processor: &str) -> Result<Self> {
        let parse_error = |e: serde_json::Error| ConfigSnafu { message: e.to_string() }.build();
        let raw: RawConfig = serde_json::from_str(config).map_err(parse_error)?;
        let features: RawProcessor = serde_json::from_str(processor).map_err(parse_error)?;
        let features = features.feature_extractor;
        let (audio, head, streaming) = (raw.audio_config, raw.head_config, raw.streaming_config);
        let config = Self {
            sample_rate: features.sampling_rate,
            n_fft: features.n_fft,
            hop_length: features.hop_length,
            win_length: features.win_length,
            preemphasis: features.preemphasis,
            num_mel_bins: audio.num_mel_bins,
            subsampling_factor: audio.subsampling_factor,
            hidden_size: audio.hidden_size,
            intermediate_size: audio.intermediate_size,
            num_attention_heads: audio.num_attention_heads,
            num_key_value_heads: audio.num_key_value_heads.unwrap_or(audio.num_attention_heads),
            num_hidden_layers: audio.num_hidden_layers,
            rope_theta: audio.rope_parameters.rope_theta,
            max_positions: audio.max_position_embeddings,
            head_hidden_size: head.hidden_size,
            num_speakers: head.num_speakers,
            offline: Profile {
                chunk_len: raw.chunk_length,
                right_context: raw.chunk_right_context,
                fifo_len: raw.fifo_length,
                update_period: raw.speaker_cache_update_period,
            },
            streaming_fifo_len: streaming.fifo_length,
            streaming_update_period: streaming.speaker_cache_update_period,
            cache: streaming.policy,
            dtype: DType::Float32,
            max_batch: 1,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        let fail = |message: String| ConfigSnafu { message }.fail();
        if self.num_mel_bins == 0 || self.win_length == 0 || self.win_length > self.n_fft {
            return fail(format!("unsupported front-end: n_fft {} / win {}", self.n_fft, self.win_length));
        }
        if self.audio_head_dim() * self.num_attention_heads != self.hidden_size
            || !self.audio_head_dim().is_multiple_of(2)
        {
            return fail(format!(
                "hidden {} does not split into {} even heads",
                self.hidden_size, self.num_attention_heads
            ));
        }
        if self.num_key_value_heads != self.num_attention_heads {
            return fail(format!(
                "{} key/value heads for {} attention heads: grouped-query attention is not supported",
                self.num_key_value_heads, self.num_attention_heads
            ));
        }
        if self.subsampling_factor == 0 || self.num_speakers == 0 {
            return fail("subsampling factor and speaker count must be positive".into());
        }
        self.validate_profile(&self.offline)
    }

    pub fn validate_profile(&self, profile: &Profile) -> Result<()> {
        if profile.chunk_len == 0 {
            return ConfigSnafu { message: "chunk_len must be positive" }.fail();
        }
        if self.max_batch == 0 {
            return ConfigSnafu { message: "max_batch must be positive" }.fail();
        }
        // The encoder pads a step to `SEQ_ALIGN` (see `NemotronDiar::classify`).
        let padded = self.step_capacity(profile).next_multiple_of(SEQ_ALIGN);
        if padded > self.max_positions {
            return ConfigSnafu {
                message: format!(
                    "a step of {} frames (padded to {padded}) exceeds the {} RoPE positions",
                    self.step_capacity(profile),
                    self.max_positions
                ),
            }
            .fail();
        }
        svod_arch::diarization::SpeakerCache::new(self.cache_config(profile))
            .map(drop)
            .map_err(|e| ConfigSnafu { message: e.to_string() }.build())
    }

    pub fn audio_head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// The chunking of a streaming session in `mode`.
    pub fn streaming_profile(&self, mode: StreamingMode) -> Profile {
        let (chunk_len, right_context) = mode.chunk_sizes();
        Profile {
            chunk_len,
            right_context,
            fifo_len: self.streaming_fifo_len,
            update_period: self.streaming_update_period,
        }
    }

    pub fn cache_config(&self, profile: &Profile) -> SpeakerCacheConfig {
        let policy = &self.cache;
        SpeakerCacheConfig {
            num_speakers: self.num_speakers,
            cache_len: policy.cache_len,
            fifo_len: profile.fifo_len,
            update_period: profile.update_period,
            silence_frames_per_speaker: policy.silence_frames_per_speaker,
            score_threshold: policy.score_threshold,
            latest_frames_boost: policy.latest_frames_boost,
            strong_boost_rate: policy.strong_boost_rate,
            weak_boost_rate: policy.weak_boost_rate,
            min_positive_rate: policy.min_positive_rate,
        }
    }

    /// Encoder frames one step attends to at most: the full cache and FIFO,
    /// the chunk and its look-ahead.
    pub fn step_capacity(&self, profile: &Profile) -> usize {
        self.cache.cache_len + profile.fifo_len + profile.chunk_len + profile.right_context
    }

    /// NeMo's `AudioToMelSpectrogramPreprocessor` as this checkpoint sets it:
    /// symmetric Hann zero-padded to `n_fft`, zero `center` padding,
    /// whole-signal pre-emphasis, Slaney mels, `ln(x + 2^-24)`, no
    /// normalization.
    pub fn mel_config(&self) -> MelConfig {
        MelConfig {
            sample_rate: self.sample_rate,
            n_fft: self.n_fft,
            hop_length: self.hop_length,
            win_length: self.win_length,
            n_mels: self.num_mel_bins,
            center: true,
            mel_scale: MelScale::Slaney,
            periodic: false,
            pad_mode: PadMode::Zero,
            preemphasis: Some(self.preemphasis),
            log: MelLog::LnAdd { guard: LOG_GUARD },
        }
    }

    /// Seconds per output (mel) frame.
    pub fn frame_sec(&self) -> f32 {
        self.hop_length as f32 / self.sample_rate as f32
    }
}

#[derive(Deserialize)]
struct RawConfig {
    audio_config: RawAudio,
    head_config: RawHead,
    streaming_config: RawStreaming,
    chunk_length: usize,
    chunk_right_context: usize,
    fifo_length: usize,
    speaker_cache_update_period: usize,
}

#[derive(Deserialize)]
struct RawAudio {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    num_hidden_layers: usize,
    num_mel_bins: usize,
    subsampling_factor: usize,
    max_position_embeddings: usize,
    rope_parameters: RawRope,
}

#[derive(Deserialize)]
struct RawRope {
    rope_theta: f64,
}

#[derive(Deserialize)]
struct RawHead {
    hidden_size: usize,
    num_speakers: usize,
}

#[derive(Deserialize)]
struct RawStreaming {
    fifo_length: usize,
    speaker_cache_update_period: usize,
    #[serde(flatten)]
    policy: CachePolicy,
}

#[derive(Deserialize)]
struct RawProcessor {
    feature_extractor: RawFeatures,
}

#[derive(Deserialize)]
struct RawFeatures {
    sampling_rate: usize,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    preemphasis: f32,
}
