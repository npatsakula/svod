//! The Nemotron-3-Diarization network (transformers
//! `Nemotron3DiarizationForAudioFrameClassification`), one chunk per
//! [`NemotronDiar::step`]:
//!
//! - [`NemotronDiar::embed`]: host-framed audio → log-mel → stacks of
//!   `subsampling_factor` frames → linear → encoder input embeddings. These are
//!   what the speaker cache keeps.
//! - the step input `[cache | fifo | chunk | look-ahead]`, gathered on the
//!   device from the previous step's input, the chunk and the silence
//!   embedding, as the host's [`SpeakerCache`] layout says.
//! - [`NemotronDiar::classify`]: pre-LN RoPE Transformer → projection →
//!   sub-pixel upsampling back to the mel frame rate → per-speaker sigmoid.
//!
//! [`SpeakerCache`]: svod_arch::diarization::SpeakerCache
//!
//! The network runs in `config.dtype` (the front-end's spectrogram aside), as
//! the reference runs in its model dtype; the embeddings the step input is
//! gathered from and the probabilities stay f32.

use std::path::Path;

use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{Conv1d, Layer, LayerNorm, Linear, Module, get_tensor, prefixed};
use svod_tk3::ops::{self, Act, Attn, KeyMask};

use crate::audio::MelSpectrogram;
use crate::init::{Bias, conv1d, layer_norm, linear};
use crate::state::{self, StateDict, scoped, scoped_index};

use super::config::NemotronDiarConfig;
use super::error::Result;

/// Sequence multiple the encoder pads a step to where flash attention does
/// not set one. Measured on CPU: the 541-frame streaming step runs 2x faster
/// padded to 576 than as is, and the 684-frame offline step fastest at 704.
const SEQ_ALIGN: usize = 64;

/// The checkpoint's repository.
pub const HUB_REPO: &str = "nvidia/Nemotron-3-Diarization";

/// `act(x·wᵀ + b) + residual`, the bias, activation and residual add in the
/// GEMM's epilogue where the tile kernel runs.
fn project(layer: &Linear, x: &Tensor, act: Act, residual: Option<&Tensor>) -> Result<Tensor> {
    Ok(ops::linear(x, &layer.weight, ops::Linear { bias: layer.bias.as_ref(), act, residual, gated: false })?)
}

fn norm(layer: &LayerNorm, x: &Tensor) -> Result<Tensor> {
    Ok(ops::layer_norm(x, &layer.weight, layer.bias.as_ref(), layer.eps)?)
}

/// Self-attention with RoPE. The checkpoint's bias-free `q_proj`, `k_proj` and
/// `v_proj` are held stacked in one realized buffer, so the three projections
/// are one GEMM (a lazy `cat` would be re-read per part in its K loop).
#[derive(Clone)]
pub struct Attention {
    /// `q_proj.weight` over `k_proj.weight` over `v_proj.weight`.
    pub qkv_weight: Tensor,
    pub o_proj: Linear,
    pub num_heads: usize,
}

impl Attention {
    fn empty(config: &NemotronDiarConfig) -> Self {
        let (d, dtype) = (config.hidden_size, config.dtype.clone());
        Self {
            qkv_weight: crate::init::fan_in_uniform(&[3 * d, d], d, dtype.clone()),
            o_proj: linear(d, d, Bias::FanIn, dtype),
            num_heads: config.num_attention_heads,
        }
    }

    /// `residual + o_proj(attention(x))`. `x`: `[B, S, D]`; `rope`: `(cos,
    /// sin)` as `[1, S, 1, Dh / 2]`; `key_lens`: `[B]` attended keys per row.
    fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor), key_lens: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let (b, s, d) = (x.dim(0)?, x.dim(1)?, x.dim_const(2)?);
        let qkv = ops::linear(x, &self.qkv_weight, ops::Linear::default())?;
        let heads = |part: usize| {
            qkv.narrow(-1, part * d, d)?.try_reshape([
                b.clone(),
                s.clone(),
                SInt::Const(self.num_heads),
                SInt::Const(d / self.num_heads),
            ])
        };
        let (cos, sin) = rope;
        // Materialized: left lazy, the rotation fuses into the score GEMM and is
        // recomputed inside its reduction.
        let q = heads(0)?.apply_rotary_emb(cos, sin, false)?.contiguous();
        let k = heads(1)?.apply_rotary_emb(cos, sin, false)?.contiguous();
        let v = heads(2)?;
        let opts = Attn { keys: KeyMask::Lens(key_lens), ..Attn::default() };
        let out = ops::attention(&q, &k, &v, opts)?;
        project(&self.o_proj, &out.try_reshape([b, s, SInt::Const(d)])?, Act::None, Some(residual))
    }
}

impl Module for Attention {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        let d = self.qkv_weight.dim_const(1).expect("a 2-D stacked qkv weight");
        for (part, name) in ["q_proj.weight", "k_proj.weight", "v_proj.weight"].iter().enumerate() {
            out.insert(prefixed(prefix, name), self.qkv_weight.narrow(0, part * d, d).expect("stacked qkv weight"));
        }
        self.o_proj.write_state(&prefixed(prefix, "o_proj"), out);
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> svod_tensor::error::Result<()> {
        let part = |name: &str| get_tensor(sd, &prefixed(prefix, name));
        let (q, k, v) = (part("q_proj.weight")?, part("k_proj.weight")?, part("v_proj.weight")?);
        self.qkv_weight = Tensor::cat(&[&q, &k, &v], 0)?.contiguous();
        self.qkv_weight.realize()?;
        self.o_proj.load_state_dict(sd, &prefixed(prefix, "o_proj"))
    }
}

#[derive(Clone, Module)]
pub struct Mlp {
    pub fc1: Linear,
    pub fc2: Linear,
}

#[derive(Clone, Module)]
pub struct EncoderLayer {
    pub layer_norm1: LayerNorm,
    pub self_attn: Attention,
    pub layer_norm2: LayerNorm,
    pub mlp: Mlp,
}

impl EncoderLayer {
    fn empty(config: &NemotronDiarConfig) -> Self {
        let (d, ff, dtype) = (config.hidden_size, config.intermediate_size, config.dtype.clone());
        Self {
            layer_norm1: layer_norm(d, dtype.clone()),
            self_attn: Attention::empty(config),
            layer_norm2: layer_norm(d, dtype.clone()),
            mlp: Mlp { fc1: linear(d, ff, Bias::FanIn, dtype.clone()), fc2: linear(ff, d, Bias::FanIn, dtype) },
        }
    }

    /// Pre-LN: `x + attn(ln1(x))`, then `+ mlp(ln2(·))` with exact GELU.
    fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor), key_lens: &Tensor) -> Result<Tensor> {
        let normed = norm(&self.layer_norm1, x)?;
        let x = scoped("self_attn", || self.self_attn.forward(&normed, rope, key_lens, x))?;
        scoped("mlp", || {
            let h = project(&self.mlp.fc1, &norm(&self.layer_norm2, &x)?, Act::Gelu, None)?;
            project(&self.mlp.fc2, &h, Act::None, Some(&x))
        })
    }
}

#[derive(Clone, Module)]
pub struct NemotronDiar {
    #[module(skip)]
    pub config: NemotronDiarConfig,
    #[module(skip)]
    mel: MelSpectrogram,
    /// RoPE `(cos, sin)` for every position, `[max_positions, 1, Dh / 2]` in
    /// the compute dtype, realized once: left lazy, every layer's rotation
    /// recomputes the pow/sin/cos.
    #[module(skip)]
    rope: (Tensor, Tensor),
    /// Stacked mel frames → encoder input, no bias.
    #[module(key = "model.audio_tower.embedder.projection")]
    pub projection: Linear,
    #[module(key = "model.audio_tower.input_layer_norm")]
    pub input_norm: LayerNorm,
    #[module(key = "model.audio_tower.layers")]
    pub layers: Vec<EncoderLayer>,
    #[module(key = "model.audio_tower.layer_norm")]
    pub final_norm: LayerNorm,
    #[module(key = "model.proj")]
    pub proj: Linear,
    /// Sub-pixel convolution: `subsampling_factor` output frames per encoder
    /// frame, on the channel axis.
    #[module(key = "model.upsampler.conv")]
    pub upsampler: Conv1d,
    #[module(key = "classifier.dense")]
    pub dense: Linear,
    #[module(key = "classifier.out_proj")]
    pub out_proj: Linear,
    /// Learned embedding of the speaker cache's silence slots.
    #[module(key = "silence_embeds")]
    pub silence: Tensor,
}

impl NemotronDiar {
    /// Randomly initialized weights in the checkpoint's layout.
    pub fn empty(config: NemotronDiarConfig) -> Self {
        let (f32, dtype) = (DType::Float32, config.dtype.clone());
        let (d, head, stack) =
            (config.hidden_size, config.head_hidden_size, config.subsampling_factor * config.num_mel_bins);
        let (cos, sin) =
            Tensor::rope_table(config.rope_theta, config.max_positions, config.audio_head_dim(), config.dtype.clone())
                .expect("an even head dim and positions");
        let position_major = |t: Tensor| {
            t.try_squeeze(Some(0)).expect("a 4-D rope table").try_permute(&[1, 0, 2]).expect("3-D").contiguous()
        };
        let rope = (position_major(cos), position_major(sin));
        Tensor::realize_batch([&rope.0, &rope.1]).expect("realizing the RoPE table");
        Self {
            mel: MelSpectrogram::new(&config.mel_config()),
            rope,
            projection: linear(stack, d, Bias::None, dtype.clone()),
            input_norm: layer_norm(d, config.dtype.clone()),
            layers: (0..config.num_hidden_layers).map(|_| EncoderLayer::empty(&config)).collect(),
            final_norm: layer_norm(d, config.dtype.clone()),
            proj: linear(d, head, Bias::FanIn, dtype.clone()),
            upsampler: conv1d(head, head * config.subsampling_factor, 3, Bias::FanIn, dtype.clone())
                .with_padding((1, 1)),
            dense: linear(head, head, Bias::FanIn, dtype.clone()),
            out_proj: linear(head, config.num_speakers, Bias::FanIn, dtype),
            silence: crate::init::zeros(&[d], f32),
            config,
        }
    }

    /// `config.json`, `processor_config.json` and `model.safetensors` of
    /// [`HUB_REPO`], with the caller's compute dtype and batch.
    pub fn from_hub(dtype: DType, max_batch: usize) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(HUB_REPO, "main")?;
        let mut config =
            NemotronDiarConfig::from_json_files(&repo.get("config.json")?, &repo.get("processor_config.json")?)?;
        config.dtype = dtype;
        config.max_batch = max_batch;
        Self::from_safetensors(&repo.get("model.safetensors")?, config)
    }

    pub fn from_safetensors(path: &Path, config: NemotronDiarConfig) -> Result<Self> {
        Self::from_state_dict(state::load_safetensors(path)?, config)
    }

    /// The weights are cast to `config.dtype` and realized once (a lazy cast
    /// would rerun on every step); the silence embedding stays f32.
    pub fn from_state_dict(mut sd: StateDict, config: NemotronDiarConfig) -> Result<Self> {
        let mut casts = Vec::new();
        for (key, tensor) in &mut sd {
            // The silence embedding is a row of the f32 table the step input is gathered from.
            let target = if key == "silence_embeds" { DType::Float32 } else { config.dtype.clone() };
            if tensor.dtype() != target {
                *tensor = tensor.cast(target).contiguous();
                casts.push(tensor.clone());
            }
        }
        Tensor::realize_batch(&casts)?;
        let mut model = Self::empty(config);
        model.load_state_dict(&sd, "")?;
        Ok(model)
    }

    pub fn mel(&self) -> &MelSpectrogram {
        &self.mel
    }

    /// Host-framed audio rows `[B, L]` ([`crate::audio::FrameCursor::stage`])
    /// and their valid mel frame counts `[B]` → encoder input embeddings
    /// `[B, M / subsampling_factor, hidden]` (f32), `M = (L - n_fft) / hop + 1`
    /// mel frames. Frames past the valid count are zero, as are the stacks they
    /// fill.
    pub fn embed(&self, framed: &Tensor, frames: &Tensor) -> Result<Tensor> {
        let mel = scoped("mel", || self.mel.forward_tensor(framed, frames))?;
        let (b, mel_frames) = (mel.dim(0)?, mel.dim_const(2)?);
        let factor = self.config.subsampling_factor;
        assert!(mel_frames.is_multiple_of(factor), "framed rows must cover whole stacks of mel frames");
        // Materialized frame-major in the compute dtype: read through the
        // permute, the stacks are strided GEMM operands.
        let stacked = mel
            .try_permute(&[0, 2, 1])?
            .try_reshape([b, SInt::Const(mel_frames / factor), SInt::Const(factor * self.config.num_mel_bins)])?
            .cast(self.config.dtype.clone())
            .contiguous();
        Ok(scoped("embedder", || project(&self.projection, &stacked, Act::None, None))?.cast(DType::Float32))
    }

    /// One step: encoder input embeddings `[B, S, hidden]` (f32), each row
    /// holding `seq_lens[b]` frames of which the first `key_lens[b]` are
    /// attended to → speaker probabilities `[B, S · subsampling_factor,
    /// num_speakers]` (f32).
    ///
    /// Rows past `seq_lens` are padding: zeroed before the upsampling
    /// convolution, whose window would otherwise read them across the end of
    /// the step. Frames between `key_lens` and `seq_lens` are encoded but never
    /// attended to (the mel padding of a recording's last stack).
    pub fn classify(&self, embeds: &Tensor, seq_lens: &Tensor, key_lens: &Tensor) -> Result<Tensor> {
        let config = &self.config;
        let seq = embeds.dim_const(1)?;
        // The encoder sees the step padded to a length the kernels like:
        // `SEQ_ALIGN` for the scheduler (an odd length such as the 541-frame
        // streaming step halves its throughput), and the tile kernels'
        // preference on top. `key_lens` hides the padding, which is cut again
        // before the head.
        let padded = ops::preferred_len(&embeds.device(), &config.dtype, seq.next_multiple_of(SEQ_ALIGN));
        let embeds_padded = match padded - seq {
            0 => embeds.clone(),
            pad => embeds.try_pad(&[(0, 0), (0, pad as isize), (0, 0)])?,
        };
        // [S, 1, Dh/2] → [1, S, 1, Dh/2], the seq-major head layout.
        let table = |t: &Tensor| t.narrow(0, 0_usize, padded)?.try_unsqueeze(0);
        let rope = (table(&self.rope.0)?, table(&self.rope.1)?);

        let mut x = scoped("input_norm", || norm(&self.input_norm, &embeds_padded.cast(config.dtype.clone())))?;
        for (index, layer) in self.layers.iter().enumerate() {
            x = scoped_index("layers", index, || layer.forward(&x, &rope, key_lens))?;
        }
        let x = scoped("final_norm", || norm(&self.final_norm, &x.narrow(1, 0_usize, seq)?))?;

        let valid = Tensor::sequence_mask(seq_lens, seq)?.cast(config.dtype.clone()).try_unsqueeze(-1)?;
        let hidden = scoped("proj", || project(&self.proj, &x, Act::None, None))?.try_mul(&valid)?;
        let head = config.head_hidden_size;
        let upsampled = scoped("upsampler", || self.upsampler.forward(&hidden.try_permute(&[0, 2, 1])?))?
            .try_permute(&[0, 2, 1])?
            .try_reshape([embeds.dim(0)?, SInt::Const(seq * config.subsampling_factor), SInt::Const(head)])?;
        let logits = scoped("classifier", || -> Result<Tensor> {
            let h = project(&self.dense, &upsampled.relu()?.contiguous(), Act::None, None)?;
            project(&self.out_proj, &h.relu()?.contiguous(), Act::None, None)
        })?;
        Ok(logits.cast(DType::Float32).sigmoid()?)
    }

    /// One step, the front-end and the classifier in one graph:
    ///
    /// - `framed` `[B, L]`, `mel_valid` `[B]`: the chunk and its look-ahead as
    ///   for [`embed`](Self::embed), `E` encoder frames.
    /// - `previous` `[B, P, hidden]`: the input of the stream's previous step.
    /// - `sources` `[B, S]`: every row of this step's input as a row of
    ///   `[previous (P) | chunk (E) | silence | zero]`.
    ///
    /// Returns the speaker probabilities ([`classify`](Self::classify)) and the
    /// gathered `[B, S, hidden]` step input, the next step's `previous`.
    pub fn step(
        &self,
        framed: &Tensor,
        mel_valid: &Tensor,
        previous: &Tensor,
        sources: &Tensor,
        seq_lens: &Tensor,
        key_lens: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let chunk = self.embed(framed, mel_valid)?;
        let (b, hidden) = (previous.dim(0)?, self.config.hidden_size);
        let row = |t: Tensor| {
            t.try_reshape([SInt::Const(1), SInt::Const(1), SInt::Const(hidden)])?.try_expand([
                b.clone(),
                SInt::Const(1),
                SInt::Const(hidden),
            ])
        };
        let silence = row(self.silence.clone())?;
        let zero = row(Tensor::zeros(&[hidden], DType::Float32))?;
        // One buffer to gather from; the gather lowers to indexed row loads.
        let table = scoped("gather", || Tensor::cat(&[previous, &chunk, &silence, &zero], 1).map(|t| t.contiguous()))?;
        let index = sources.try_unsqueeze(-1)?.try_expand([b, sources.dim(1)?, SInt::Const(hidden)])?;
        let input = scoped("gather", || table.gather(1, &index))?.contiguous();
        Ok((self.classify(&input, seq_lens, key_lens)?, input))
    }
}
