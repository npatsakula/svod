//! GigaAM JSON config parsing.
//!
//! `GigaAmConfig::from_json` reads `config.json` and produces a fully-resolved
//! [`GigaAmConfig`]. Parsing is via `serde_json` with mirror `Raw*` structs
//! that match the on-disk shape (`cfg.model.cfg.{preprocessor,encoder,head,
//! decoding}`); a thin `From`-style projection then validates cross-field
//! invariants and dispatches the `decoding._target_` union substring-style.
//!
//! Substring dispatch (rather than `#[serde(tag = "_target_")]`) is intentional:
//! the `_target_` paths drift across upstream NeMo versions, so exact-rename
//! enum variants would break each release. The leaf decoder types
//! (`GreedyDecoder`, `BeamDecoder`) are themselves `Deserialize`, so the
//! within-variant fields still ride serde.

use std::path::Path;

use serde::Deserialize;
use snafu::ResultExt;
use svod_arch::ctc::{CtcDecoder, GreedyDecoder};

use super::error::{ConfigIoSnafu, ConfigSnafu, Error, Result};

#[derive(Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubsamplingMode {
    Conv1d,
    Conv2d,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConvNormType {
    LayerNorm,
    BatchNorm,
}

#[derive(Clone)]
pub struct GigaAmConfig {
    pub max_batch_size: usize,
    pub n_mels: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub conv_kernel: usize,
    pub subsampling_factor: usize,
    pub subsampling_mode: SubsamplingMode,
    pub subs_kernel_size: usize,
    pub conv_norm_type: ConvNormType,
    pub vocab_size: usize,
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub mel_center: bool,
    pub max_mel_frames: usize,
    pub max_encoder_frames: usize,
    /// CTC decoder built from the `decoding` section of the config, or an
    /// empty-vocabulary greedy decoder for synthetic configs that don't
    /// declare one.
    pub decoder: CtcDecoder,
    /// Transducer-specific config, populated when `decoding._target_` ends
    /// in `RNNTGreedyDecoding` (or the head config has predictor/joint
    /// blocks). `None` for CTC checkpoints.
    pub transducer: Option<TransducerConfig>,
    /// The cache-aware streaming variant of the encoder, from the top-level
    /// `streaming` block a streaming fine-tune's `config.json` carries.
    /// `None` is the offline encoder.
    pub streaming: Option<StreamingConfig>,
}

/// The streaming encoder: the same parameters, with causal subsampling and
/// convolutions and block attention over chunks of `chunk_size` encoder
/// frames (40 ms each). A query sees its own chunk and the
/// `left_context_chunks` before it; `right_context_chunks == 1` lets it see
/// the next chunk too, at a chunk of extra latency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamingConfig {
    pub chunk_size: usize,
    /// `None` is every earlier chunk: the cache then spans `max_encoder_frames`.
    pub left_context_chunks: Option<usize>,
    /// 0 or 1.
    pub right_context_chunks: usize,
}

impl StreamingConfig {
    /// The keys and values a lane keeps per layer.
    pub fn cache_frames(&self, max_encoder_frames: usize) -> usize {
        self.left_context_chunks.map_or(max_encoder_frames, |left| left * self.chunk_size)
    }

    /// Encoder frames a step runs its layers over: the chunk, plus the
    /// look-ahead chunk when there is one.
    pub fn step_frames(&self) -> usize {
        self.chunk_size * (1 + self.right_context_chunks)
    }

    fn validate(self) -> Result<Self> {
        if self.chunk_size == 0 || self.right_context_chunks > 1 {
            return Err(Error::Streaming {
                message: format!(
                    "chunk_size {} must be positive and right_context_chunks {} 0 or 1",
                    self.chunk_size, self.right_context_chunks
                ),
            });
        }
        Ok(self)
    }
}

/// RNN-T-specific config extracted from the JSON `head.decoder` /
/// `head.joint` / `decoding` blocks. See `submodules/GigaAM/gigaam/decoder.py`
/// for the reference shape.
#[derive(Clone, Debug)]
pub struct TransducerConfig {
    pub pred_hidden: usize,
    pub pred_rnn_layers: usize,
    pub joint_hidden: usize,
    /// `vocabulary.len() + 1` — includes the blank token at the end.
    pub num_classes: usize,
    pub max_symbols_per_step: usize,
    pub vocabulary: Vec<String>,
    /// True when the vocabulary entries are SentencePiece pieces (apply
    /// `▁ → space` post-processing on the decoded string).
    pub sentencepiece: bool,
}

impl GigaAmConfig {
    pub fn from_json(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).context(ConfigIoSnafu)?;
        let root: serde_json::Value = serde_json::from_str(&data).context(ConfigSnafu)?;
        let leaf = root.pointer("/cfg/model/cfg").ok_or_else(|| Error::DecoderConfig {
            message: "config.json missing required path /cfg/model/cfg".into(),
        })?;
        let raw: RawModelCfg = serde_json::from_value(leaf.clone()).context(ConfigSnafu)?;
        let streaming = match root.get("streaming") {
            Some(block) => serde_json::from_value::<RawStreaming>(block.clone()).context(ConfigSnafu)?.into_config()?,
            None => None,
        };
        Self::from_raw(raw, streaming)
    }

    fn from_raw(raw: RawModelCfg, streaming: Option<StreamingConfig>) -> Result<Self> {
        validate_preprocessor(&raw.preprocessor)?;
        validate_encoder(&raw.encoder)?;
        if streaming.is_some() && !matches!(raw.encoder.subsampling, SubsamplingMode::Conv1d) {
            return Err(Error::Streaming { message: "streaming needs conv1d subsampling".into() });
        }

        // `max_mel_frames` is the pre-subsampling sequence-length bound. Configs that
        // only specify `pos_emb_max_len` (the post-subsampling encoder bound) need it
        // multiplied by `subsampling_factor` so audio approaching the encoder cap
        // isn't rejected at the JIT input stage.
        let max_encoder_frames = raw.encoder.pos_emb_max_len;
        let max_mel_frames = raw
            .encoder
            .max_mel_frames
            .or(raw.encoder.max_seq_len)
            .unwrap_or(max_encoder_frames * raw.encoder.subsampling_factor);
        let subs_kernel = match &raw.encoder.subsampling {
            SubsamplingMode::Conv1d => raw.encoder.subs_kernel_size,
            SubsamplingMode::Conv2d => 3,
        };
        let max_sub_frames = subsampled_len(subs_kernel, max_mel_frames);
        if max_sub_frames > max_encoder_frames {
            return Err(Error::DecoderConfig {
                message: format!(
                    "max_mel_frames ({max_mel_frames}) subsamples to {max_sub_frames} encoder frames, exceeding pos_emb_max_len ({max_encoder_frames})"
                ),
            });
        }
        // CTC configs put `num_classes` directly on `head`; RNN-T configs nest
        // it under `head.decoder.num_classes` / `head.joint.num_classes`.
        let vocab_size = raw
            .head
            .num_classes
            .or_else(|| raw.head.decoder.as_ref().and_then(|d| d.num_classes))
            .or_else(|| raw.head.joint.as_ref().and_then(|j| j.num_classes))
            .ok_or_else(|| Error::DecoderConfig {
                message: "missing num_classes (head.num_classes or head.{decoder,joint}.num_classes)".into(),
            })?;
        let decoder = raw_to_decoder(raw.decoding.as_ref(), vocab_size)?;
        let transducer = raw_to_transducer(&raw.head, raw.decoding.as_ref(), vocab_size)?;
        Ok(Self {
            max_batch_size: raw.encoder.max_batch_size,
            n_mels: raw.preprocessor.features,
            d_model: raw.encoder.d_model,
            n_heads: raw.encoder.n_heads,
            n_layers: raw.encoder.n_layers,
            d_ff: raw.encoder.d_model * raw.encoder.ff_expansion_factor,
            conv_kernel: raw.encoder.conv_kernel_size,
            subsampling_factor: raw.encoder.subsampling_factor,
            subsampling_mode: raw.encoder.subsampling,
            subs_kernel_size: raw.encoder.subs_kernel_size,
            conv_norm_type: raw.encoder.conv_norm_type,
            vocab_size,
            sample_rate: raw.preprocessor.sample_rate,
            n_fft: raw.preprocessor.n_fft,
            hop_length: raw.preprocessor.hop_length,
            win_length: raw.preprocessor.win_length,
            mel_center: raw.preprocessor.center,
            max_mel_frames,
            max_encoder_frames,
            decoder,
            transducer,
            streaming,
        })
    }
}

// ─── Serde mirror structs (private) ───────────────────────────────────────
//
// On-disk shape is `cfg.model.cfg.{preprocessor,encoder,head,decoding}`; the
// outer wrappers are navigated via `serde_json::Value::pointer` in `from_json`
// rather than mirrored here so this file stays focused on the leaf shape.

#[derive(Deserialize)]
struct RawModelCfg {
    preprocessor: RawPreprocessor,
    encoder: RawEncoder,
    head: RawHead,
    #[serde(default)]
    decoding: Option<serde_json::Value>,
}

/// The top-level `streaming` block: the fine-tune's settings, `enabled`
/// false or a missing block being the offline model. `left_context_chunks`
/// of -1 is unbounded.
#[derive(Deserialize)]
struct RawStreaming {
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default = "default_chunk_size")]
    chunk_size: usize,
    #[serde(default = "default_left_context")]
    left_context_chunks: i64,
    #[serde(default)]
    right_context_chunks: usize,
}

impl RawStreaming {
    fn into_config(self) -> Result<Option<StreamingConfig>> {
        if !self.enabled {
            return Ok(None);
        }
        let left_context_chunks = usize::try_from(self.left_context_chunks).ok();
        let config = StreamingConfig {
            chunk_size: self.chunk_size,
            left_context_chunks,
            right_context_chunks: self.right_context_chunks,
        };
        config.validate().map(Some)
    }
}

#[derive(Deserialize)]
struct RawPreprocessor {
    features: usize,
    sample_rate: usize,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    #[serde(default = "default_true")]
    center: bool,
    #[serde(default)]
    mel_scale: Option<String>,
    #[serde(default)]
    mel_norm: Option<String>,
}

#[derive(Deserialize)]
struct RawEncoder {
    d_model: usize,
    ff_expansion_factor: usize,
    n_heads: usize,
    n_layers: usize,
    conv_kernel_size: usize,
    subsampling_factor: usize,
    #[serde(default = "default_self_attention_model")]
    self_attention_model: String,
    #[serde(default = "default_subs_kernel_size")]
    subs_kernel_size: usize,
    #[serde(default = "default_subsampling_mode")]
    subsampling: SubsamplingMode,
    #[serde(default = "default_conv_norm_type")]
    conv_norm_type: ConvNormType,
    #[serde(default = "default_pos_emb_max_len")]
    pos_emb_max_len: usize,
    #[serde(default)]
    max_mel_frames: Option<usize>,
    #[serde(default)]
    max_seq_len: Option<usize>,
    #[serde(default = "default_max_batch_size")]
    max_batch_size: usize,
}

#[derive(Deserialize)]
struct RawHead {
    #[serde(default)]
    num_classes: Option<usize>,
    #[serde(default)]
    decoder: Option<RawHeadDecoder>,
    #[serde(default)]
    joint: Option<RawHeadJoint>,
}

#[derive(Deserialize)]
struct RawHeadDecoder {
    pred_hidden: usize,
    pred_rnn_layers: usize,
    #[serde(default)]
    num_classes: Option<usize>,
}

#[derive(Deserialize)]
struct RawHeadJoint {
    joint_hidden: usize,
    #[serde(default)]
    num_classes: Option<usize>,
}

fn default_true() -> bool {
    true
}
fn default_chunk_size() -> usize {
    8
}
fn default_left_context() -> i64 {
    -1
}
fn default_subs_kernel_size() -> usize {
    3
}
fn default_subsampling_mode() -> SubsamplingMode {
    SubsamplingMode::Conv2d
}
fn default_conv_norm_type() -> ConvNormType {
    ConvNormType::BatchNorm
}
fn default_pos_emb_max_len() -> usize {
    5000
}
fn default_self_attention_model() -> String {
    "rotary".into()
}
fn default_max_batch_size() -> usize {
    32
}

fn validate_preprocessor(pre: &RawPreprocessor) -> Result<()> {
    if let Some(scale) = pre.mel_scale.as_deref()
        && scale != "htk"
    {
        return Err(Error::DecoderConfig {
            message: format!(
                "unsupported mel_scale {scale:?}; Svod GigaAM currently matches torchaudio's HTK mel frontend"
            ),
        });
    }
    if let Some(norm) = pre.mel_norm.as_deref() {
        return Err(Error::DecoderConfig {
            message: format!(
                "unsupported mel_norm {norm:?}; Svod GigaAM currently supports only null/no mel normalization"
            ),
        });
    }
    if pre.n_fft != pre.win_length {
        return Err(Error::DecoderConfig {
            message: format!(
                "unsupported mel frontend n_fft ({}) != win_length ({}); current GigaAM parity path requires equal FFT/window lengths",
                pre.n_fft, pre.win_length
            ),
        });
    }
    Ok(())
}

fn validate_encoder(encoder: &RawEncoder) -> Result<()> {
    if encoder.self_attention_model != "rotary" {
        return Err(Error::DecoderConfig {
            message: format!(
                "unsupported self_attention_model {:?}; Svod GigaAM currently implements rotary attention only",
                encoder.self_attention_model
            ),
        });
    }
    if encoder.subsampling_factor != 4 {
        return Err(Error::DecoderConfig {
            message: format!(
                "unsupported subsampling_factor {}; Svod GigaAM currently implements exactly two stride-2 subsampling layers",
                encoder.subsampling_factor
            ),
        });
    }
    Ok(())
}

/// Encoder frames produced by the two stride-2 subsampling convolutions
/// (kernel `kernel_size`, `(kernel_size - 1) / 2` padding, applied twice).
/// The single definition behind `StridingSubsampling::output_length`,
/// `GigaAmConfig` validation and the transcriber's chunk bookkeeping.
///
/// Saturating: a degenerate `mel_frames < kernel_size - 2 * pad` (only
/// reachable for an empty / sub-frame window) must clamp to 0, not wrap.
pub(crate) fn subsampled_len(kernel_size: usize, mel_frames: usize) -> usize {
    let pad = (kernel_size - 1) / 2;
    let mut len = mel_frames;
    for _ in 0..2 {
        len = len.saturating_add(2 * pad).saturating_sub(kernel_size) / 2 + 1;
    }
    len
}

// ─── Decoder + transducer dispatch ────────────────────────────────────────

fn raw_to_decoder(decoding: Option<&serde_json::Value>, vocab_size: usize) -> Result<CtcDecoder> {
    let Some(decoding) = decoding else {
        return Ok(CtcDecoder::Greedy(GreedyDecoder::new(Vec::new())));
    };
    if decoding.is_null() {
        return Ok(CtcDecoder::Greedy(GreedyDecoder::new(Vec::new())));
    }
    let target = decoding["_target_"].as_str().unwrap_or("");
    let decoder: CtcDecoder = if target.contains("CTCGreedyDecoding") {
        let g: GreedyDecoder = serde_json::from_value(decoding.clone()).context(ConfigSnafu)?;
        CtcDecoder::Greedy(g)
    } else if target.contains("CTCBeamDecoding") {
        let b: svod_arch::ctc::BeamDecoder = serde_json::from_value(decoding.clone()).context(ConfigSnafu)?;
        CtcDecoder::Beam(Box::new(b))
    } else {
        // Unknown / missing target. If there's a vocabulary array, default to
        // greedy; otherwise empty.
        let vocab: Vec<String> = decoding["vocabulary"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        CtcDecoder::Greedy(GreedyDecoder::new(vocab))
    };
    if !decoder.vocabulary().is_empty() && decoder.total_vocab() != vocab_size {
        return Err(Error::DecoderConfig {
            message: format!(
                "decoder vocabulary length + 1 ({}) != head.num_classes ({}); \
                 CTC convention is one blank token appended after the vocabulary",
                decoder.total_vocab(),
                vocab_size
            ),
        });
    }
    Ok(decoder)
}

fn raw_to_transducer(
    head: &RawHead,
    decoding: Option<&serde_json::Value>,
    vocab_size: usize,
) -> Result<Option<TransducerConfig>> {
    let target = decoding.and_then(|d| d["_target_"].as_str()).unwrap_or("");
    let has_decoder = head.decoder.is_some();
    let has_joint = head.joint.is_some();
    if !(target.contains("RNNT") || (has_decoder && has_joint)) {
        return Ok(None);
    }
    let dec = head
        .decoder
        .as_ref()
        .ok_or_else(|| Error::DecoderConfig { message: "RNN-T config: missing head.decoder block".into() })?;
    let joint = head
        .joint
        .as_ref()
        .ok_or_else(|| Error::DecoderConfig { message: "RNN-T config: missing head.joint block".into() })?;
    let max_symbols_per_step = decoding.and_then(|d| d["max_symbols_per_step"].as_u64()).unwrap_or(10) as usize;
    // Vocabulary preference: `decoding.vocabulary` (CTC convention reused for
    // RNN-T configs). For SentencePiece RNN-T checkpoints (e.g. v3_e2e_rnnt)
    // this is `null` in the JSON config; the actual pieces ship as
    // `tokenizer.model` and are loaded via `from_safetensors_with_tokenizer`.
    // Empty here is fine — `from_state_dict` will splice in the override.
    let vocabulary: Vec<String> = decoding
        .and_then(|d| d["vocabulary"].as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    // SentencePiece iff `decoding.model_path` is a non-empty string, else
    // char-wise.
    let sentencepiece =
        decoding.and_then(|d| d.get("model_path")).and_then(|v| v.as_str()).map(|s| !s.is_empty()).unwrap_or(false);
    Ok(Some(TransducerConfig {
        pred_hidden: dec.pred_hidden,
        pred_rnn_layers: dec.pred_rnn_layers,
        joint_hidden: joint.joint_hidden,
        num_classes: vocab_size,
        max_symbols_per_step,
        vocabulary,
        sentencepiece,
    }))
}
