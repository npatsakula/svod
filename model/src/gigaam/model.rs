//! Unified GigaAM model: shared Conformer encoder + variant head (CTC | RN-T).
//!
//! `GigaAm` collapses the previously parallel `GigaAm` (CTC) and
//! `GigaAmRnnt` (RN-T) types into one config-driven struct. The
//! `config.transducer.is_some()` discriminator picks the head at load time:
//! `None` ⇒ `Head::Ctc(CTCHead)`, `Some(_)` ⇒ `Head::Rnnt { head, runtime }`
//! where `RnntRuntime` carries the RN-T-only inference metadata (vocabulary,
//! max symbols per step, SentencePiece flag).
//!
//! The decoder layer in `svod_arch` is not unified by this struct — CTC and
//! RN-T still use their respective `CtcDecoder` / `JointStep` shapes — but the
//! model construction, weight loading, and encoder JIT all flow through one
//! type.

use std::path::Path;

use svod_dtype::DType;

use crate::audio::{MelConfig, MelScale, MelSpectrogram, PadMode};
use crate::state::{self, StateDict};
use svod_tensor::nn::Module;

use crate::gigaam::ctc::CTCHead;
use crate::gigaam::encoder::Encoder;
use crate::gigaam::error::Error;

use crate::gigaam::rnnt::RnntHead;
use crate::gigaam::{GigaAmConfig, Result, StreamingConfig, remap};
use crate::sentencepiece;

fn prepare_scaled_weights(sd: &mut StateDict, dtype: &DType) -> Result<()> {
    let scales: Vec<_> = sd
        .keys()
        .filter_map(|key| {
            if let Some(prefix) = key.strip_suffix(".weight_scale") {
                Some((key.clone(), format!("{prefix}.weight"), format!("{prefix}.act_scale")))
            } else {
                key.strip_suffix("_weight_scale")
                    .map(|prefix| (key.clone(), format!("{prefix}_proj"), format!("{prefix}_act_scale")))
            }
        })
        .collect();
    let mut promoted = Vec::new();
    for (scale_key, weight_key, act_scale_key) in scales {
        let scale = sd.get(&scale_key).expect("scale key came from the state dict").clone();
        let weight = sd
            .get(&weight_key)
            .ok_or_else(|| Error::CheckpointConfig {
                message: format!("quantization scale {scale_key} has no matching weight {weight_key}"),
            })?
            .clone();
        let weight_dtype = weight.dtype();
        let encoder_weight = weight_key.starts_with("encoder.")
            || weight_key.starts_with("model.encoder.")
            || weight_key.starts_with("layers.")
            || weight_key.starts_with("subsampling.");
        sd.remove(&act_scale_key);
        let quantized = if weight_dtype == DType::FP8E4M3 || (weight_dtype.is_signed() && !encoder_weight) {
            weight.clone()
        } else if weight_dtype == DType::UInt8 {
            // Compatibility with the repository's original raw-bit FP8 export.
            weight.bitcast(DType::FP8E4M3)?
        } else if weight_dtype.is_signed() {
            continue;
        } else {
            return Err(Error::CheckpointConfig {
                message: format!("quantized weight {weight_key} has unsupported dtype {weight_dtype:?}"),
            });
        };
        let target_dtype = if encoder_weight { dtype.clone() } else { DType::Float32 };
        let dims = weight.dims().map_err(|_| Error::CheckpointConfig {
            message: format!("quantized weight {weight_key} has symbolic output dimension"),
        })?;
        let mut scale_shape = vec![1isize; dims.len()];
        scale_shape[0] = dims[0] as isize;
        let dequantized =
            quantized.cast(target_dtype.clone()).try_mul(&scale.cast(target_dtype).try_reshape(scale_shape)?)?;
        sd.remove(&scale_key);
        promoted.push((weight_key, dequantized.contiguous()));
    }
    svod_tensor::Tensor::realize_batch(promoted.iter().map(|(_, tensor)| tensor))?;
    for (weight_key, tensor) in promoted {
        sd.insert(weight_key, tensor);
    }
    Ok(())
}

/// WER+RTF-tuned soft chunk target (seconds) for the GigaAM RN-T pipeline.
/// Short chunks cut the RN-T decoder's autoregressive skip-deletions (the
/// largest long-form WER win on the Russian benchmark) and keep the encoder's
/// `max_t_mel` in the 1024-frame power-of-two bucket. Tuned for both FireRed and
/// Silero front-ends — it is encoder/decoder-driven, so it is VAD-independent.
pub const TUNED_TARGET_SECS: f32 = 5.7;

/// Unified GigaAM model. The `head` enum carries either a CTC projection or
/// an RN-T predictor+joint pair; pattern-match (or use [`Head::as_ctc`] /
/// [`Head::as_rnnt`]) to drive the head-specific inference path.
#[derive(Clone)]
pub struct GigaAm {
    pub config: GigaAmConfig,
    pub encoder: Encoder,
    pub head: Head,
}

/// Head variant. `Ctc` holds the small Conv1d projection consumed by
/// `svod_arch::ctc` decoders. `Rnnt` holds the predictor+joint pair plus
/// the runtime metadata (vocab, max-symbols-per-step, SP flag) used by the
/// arch's `JointStep`-driven decoder.
#[derive(Clone)]
pub enum Head {
    Ctc(CTCHead),
    Rnnt { head: RnntHead, runtime: RnntRuntime },
}

/// RN-T-only runtime metadata. Lives inside [`Head::Rnnt`] so the CTC path
/// stays free of fields it would never use.
#[derive(Clone)]
pub struct RnntRuntime {
    /// Token strings indexed by predictor class. Length is `num_classes - 1`
    /// (the last class is the blank, not a vocabulary entry).
    pub vocabulary: Vec<String>,
    /// Max non-blank emissions per encoder frame in the greedy search.
    pub max_symbols_per_step: usize,
    /// `true` if `vocabulary` is SentencePiece pieces (post-process `▁` → space
    /// on the output transcript).
    pub sentencepiece: bool,
}

impl Head {
    pub fn as_ctc(&self) -> Option<&CTCHead> {
        if let Head::Ctc(h) = self { Some(h) } else { None }
    }

    pub fn as_rnnt(&self) -> Option<(&RnntHead, &RnntRuntime)> {
        if let Head::Rnnt { head, runtime } = self { Some((head, runtime)) } else { None }
    }

    /// Try-accessor for the CTC variant, returning a typed `DecoderConfig`
    /// error when the head is RN-T. Used by the head-side JIT wrappers so
    /// "wrong head type" surfaces as a normal `Error` instead of a panic.
    pub(crate) fn expect_ctc(&self, ctx: &str) -> Result<&CTCHead> {
        self.as_ctc().ok_or_else(|| Error::DecoderConfig {
            message: format!("{ctx} requires a CTC head; this model has an RN-T head"),
        })
    }

    /// Try-accessor for the RN-T variant. Mirrors [`Head::expect_ctc`].
    pub(crate) fn expect_rnnt(&self, ctx: &str) -> Result<(&RnntHead, &RnntRuntime)> {
        self.as_rnnt().ok_or_else(|| Error::DecoderConfig {
            message: format!("{ctx} requires an RN-T head; this model has a CTC head"),
        })
    }
}

impl GigaAm {
    /// Run the encoder as the streaming variant `streaming` describes (no
    /// parameters change). A checkpoint fine-tuned for it loads that way by
    /// itself; the offline weights stream too, at reduced accuracy.
    pub fn enable_streaming(&mut self, streaming: StreamingConfig) -> Result<()> {
        let streaming = streaming.validate(&self.config.subsampling_mode)?;
        self.config.streaming = Some(streaming);
        self.encoder.streaming = Some(streaming);
        Ok(())
    }

    /// The log-mel front end this model's checkpoint was trained on.
    pub fn mel_spectrogram(&self) -> MelSpectrogram {
        let cfg = &self.config;
        MelSpectrogram::new(&MelConfig {
            sample_rate: cfg.sample_rate,
            n_fft: cfg.n_fft,
            hop_length: cfg.hop_length,
            win_length: cfg.win_length,
            n_mels: cfg.n_mels,
            center: cfg.mel_center,
            mel_scale: MelScale::Htk,
            periodic: true,
            pad_mode: PadMode::Reflect,
            preemphasis: None,
            log: MelSpectrogram::LOG,
        })
    }

    /// Recommended soft chunk-target duration (seconds) for this model, or `None`
    /// for greedy fill-to-max. RN-T benefits from the target-split (autoregressive
    /// skip-deletion); CTC (non-autoregressive) does not, so it returns `None`.
    pub fn recommended_chunk_secs(&self) -> Option<f32> {
        self.head.as_rnnt().is_some().then_some(TUNED_TARGET_SECS)
    }

    /// Load from a HuggingFace Hub repository (`main` revision).
    pub fn from_hub(model_id: &str) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main")
    }

    /// Load from a HuggingFace Hub repository at a specific branch/revision.
    /// Auto-detects head type from `config.transducer.is_some()`; fetches
    /// `tokenizer.model` only when the config asks for RN-T.
    pub fn from_hub_with_revision(model_id: &str, revision: &str) -> Result<Self> {
        Self::from_hub_with_revision_and_weights_and_encoder_dtype(
            model_id,
            revision,
            "model.safetensors",
            DType::Float16,
        )
    }

    /// Load a named Hub checkpoint with an explicit encoder compute dtype.
    pub fn from_hub_with_revision_and_weights_and_encoder_dtype(
        model_id: &str,
        revision: &str,
        weights: &str,
        encoder_dtype: DType,
    ) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision)?;
        let config_path = repo.get("config.json")?;
        let weights_path = repo.get(weights)?;
        let config = GigaAmConfig::from_json(&config_path)?;
        // SentencePiece-RN-T variants (e.g. `v3_e2e_rnnt`) ship the tokenizer
        // as `tokenizer.model`. CTC variants don't have one; skip the fetch.
        let tokenizer_path = if config.transducer.is_some() { repo.get("tokenizer.model").ok() } else { None };
        Self::from_safetensors_with_encoder_dtype(&weights_path, tokenizer_path.as_deref(), config, encoder_dtype)
    }

    /// Load from a directory containing `config.json` + `model.safetensors`
    /// (and optionally `tokenizer.model` for RN-T configs).
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_dir_with_weights_and_encoder_dtype(dir, "model.safetensors", DType::Float16)
    }

    /// Load a named local checkpoint with an explicit encoder compute dtype.
    pub fn from_dir_with_weights_and_encoder_dtype(dir: &Path, weights: &str, encoder_dtype: DType) -> Result<Self> {
        let config_path = dir.join("config.json");
        let weights_path = dir.join(weights);
        let config = GigaAmConfig::from_json(&config_path)?;
        let tokenizer_path = dir.join("tokenizer.model");
        let tokenizer_path =
            if config.transducer.is_some() && tokenizer_path.exists() { Some(tokenizer_path) } else { None };
        Self::from_safetensors_with_encoder_dtype(&weights_path, tokenizer_path.as_deref(), config, encoder_dtype)
    }

    /// Load weights + (optional) SentencePiece tokenizer and assemble the
    /// model. `tokenizer` is ignored for CTC configs.
    pub fn from_safetensors(weights: &Path, tokenizer: Option<&Path>, config: GigaAmConfig) -> Result<Self> {
        Self::from_safetensors_with_encoder_dtype(weights, tokenizer, config, DType::Float16)
    }

    /// Load a safetensors checkpoint with an explicit encoder compute dtype.
    pub fn from_safetensors_with_encoder_dtype(
        weights: &Path,
        tokenizer: Option<&Path>,
        config: GigaAmConfig,
        encoder_dtype: DType,
    ) -> Result<Self> {
        let sd = state::load_safetensors(weights)?;
        let vocab_override = tokenizer
            .map(sentencepiece::load_vocab)
            .transpose()
            .map_err(|e| Error::DecoderConfig { message: e.to_string() })?;
        Self::from_state_dict_with_encoder_dtype(&sd, config, vocab_override, encoder_dtype)
    }

    /// Build from a pre-loaded state dict. `vocab_override` (RN-T only) wins
    /// over `config.transducer.vocabulary` if `Some`.
    ///
    /// Auto-detects PyTorch key format (`encoder.` / `model.encoder.` /
    /// `head.decoder.` / `head.joint.` prefixes) and remaps to svod layout
    /// before loading.
    pub fn from_state_dict(sd: &StateDict, config: GigaAmConfig, vocab_override: Option<Vec<String>>) -> Result<Self> {
        Self::from_state_dict_with_encoder_dtype(sd, config, vocab_override, DType::Float16)
    }

    /// Build from a state dict while converting all floating encoder parameters
    /// to one compute dtype. A coherent conversion prevents FP32 affine biases
    /// from promoting reduced-precision activations before attention.
    pub fn from_state_dict_with_encoder_dtype(
        sd: &StateDict,
        config: GigaAmConfig,
        vocab_override: Option<Vec<String>>,
        encoder_dtype: DType,
    ) -> Result<Self> {
        if encoder_dtype != DType::Float16 && encoder_dtype != DType::BFloat16 && encoder_dtype != DType::Float32 {
            return Err(Error::EncoderDtype { dtype: encoder_dtype });
        }
        let is_pytorch = sd.keys().any(|k| {
            k.starts_with("encoder.")
                || k.starts_with("model.encoder.")
                || k.starts_with("head.decoder.")
                || k.starts_with("head.joint.")
        });
        let mut sd_owned = sd.clone();
        prepare_scaled_weights(&mut sd_owned, &encoder_dtype)?;
        if is_pytorch {
            sd_owned = remap::remap_pytorch(sd_owned, &config)?;
        }
        let mut casts = Vec::new();
        for (key, tensor) in &mut sd_owned {
            if (key.starts_with("subsampling.") || key.starts_with("layers."))
                && tensor.dtype().is_float()
                && tensor.dtype() != encoder_dtype
                // Both scale spellings stay at checkpoint precision: `<x>.weight_scale`
                // (FFN) and `<x>_weight_scale` (MHSA/conv), matching `prepare_scaled_weights`.
                && !key.ends_with("weight_scale")
            {
                *tensor = tensor.cast(encoder_dtype.clone()).contiguous();
                casts.push(tensor.clone());
            }
        }
        // Realized once: a lazy cast is re-run by every encoder call that reads the weight.
        svod_tensor::Tensor::realize_batch(&casts)?;
        let sd = &sd_owned;

        let encoder = Encoder::from_state_dict(sd, &config)?;

        let head = match &config.transducer {
            None => {
                let mut h = CTCHead::empty(&config);
                h.load_state_dict(sd, "head")?;
                let weight_shape = h.weight.dims().ok();
                let bias_shape = h.bias.dims().ok();
                let expected_weight = vec![config.vocab_size, config.d_model, 1];
                let expected_bias = vec![config.vocab_size];
                if weight_shape.as_ref() != Some(&expected_weight) || bias_shape.as_ref() != Some(&expected_bias) {
                    return Err(Error::CheckpointConfig {
                        message: format!(
                            "CTC head shapes weight={weight_shape:?}, bias={bias_shape:?} do not match config num_classes={} and d_model={} (expected weight={expected_weight:?}, bias={expected_bias:?})",
                            config.vocab_size, config.d_model
                        ),
                    });
                }
                Head::Ctc(h)
            }
            Some(tr) => {
                let vocabulary = vocab_override.unwrap_or_else(|| tr.vocabulary.clone());
                if vocabulary.len() + 1 != tr.num_classes {
                    return Err(Error::DecoderConfig {
                        message: format!(
                            "RN-T vocabulary length + 1 ({}) != num_classes ({}); \
                             convention is one blank token at the end",
                            vocabulary.len() + 1,
                            tr.num_classes
                        ),
                    });
                }
                let mut h = RnntHead::empty(
                    config.d_model,
                    tr.pred_hidden,
                    tr.pred_rnn_layers,
                    tr.joint_hidden,
                    tr.num_classes,
                );
                h.load_state_dict(sd, "head")?;
                h.predictor.prepare_for_inference()?;
                Head::Rnnt {
                    head: h,
                    runtime: RnntRuntime {
                        vocabulary,
                        max_symbols_per_step: tr.max_symbols_per_step,
                        sentencepiece: tr.sentencepiece,
                    },
                }
            }
        };

        Ok(Self { config, encoder, head })
    }

    /// Build a model with zero-initialized weights from `config` alone.
    /// Head variant follows `config.transducer.is_some()`.
    pub fn with_random_weights(config: GigaAmConfig) -> Self {
        let encoder = Encoder::with_random_weights(&config);
        let head = match &config.transducer {
            None => Head::Ctc(CTCHead::empty(&config)),
            Some(tr) => Head::Rnnt {
                head: RnntHead::empty(
                    config.d_model,
                    tr.pred_hidden,
                    tr.pred_rnn_layers,
                    tr.joint_hidden,
                    tr.num_classes,
                ),
                runtime: RnntRuntime {
                    vocabulary: tr.vocabulary.clone(),
                    max_symbols_per_step: tr.max_symbols_per_step,
                    sentencepiece: tr.sentencepiece,
                },
            },
        };
        Self { config, encoder, head }
    }
}
