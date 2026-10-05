//! GTCRN speech enhancement — a pure-Rust port of the upstream
//! [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) (ShuffleNetV2-style ultra-
//! tiny model, 23.67K params). A noisy waveform in → an enhanced waveform out.
//!
//! ## Recipe
//!
//! ```no_run
//! use svod_model::gtcrn::{Gtcrn, GtcrnJit};
//! use svod_model::jit::InputSpec;
//!
//! let model = Gtcrn::from_hub()?;
//! let mut jit = GtcrnJit::new(model);
//! // 4 seconds of 16 kHz audio, waveform in and waveform out.
//! jit.prepare(InputSpec::f32(&[1, 4 * 16000]))?;
//!
//! // copy the [1, L] noisy samples into `jit.waveform_mut()?`, then:
//! jit.execute()?;
//! let _enhanced = jit.output()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The analysis STFT, the mask network and the synthesis ISTFT are one graph
//! ([`Gtcrn::enhance`]); [`Gtcrn::forward`] is the spectrogram→spectrogram
//! middle on its own. `examples/gtcrn_enhance.rs` runs the pipeline over a WAV.

mod blocks;
mod error;
mod jit;
pub mod stream;

pub use error::{Error, Result};
pub use jit::GtcrnJit;

use std::f64::consts::TAU;
use std::path::Path;

use snafu::ResultExt;
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{BatchNorm2d, Conv2d, ConvTranspose2d, LayerNorm, Linear, Module, Window};

use crate::init::{fan_in_uniform, ones, zeros};
use crate::state::{self, StateDict};

use error::HubSnafu;

use blocks::{Conv, ConvBlock, Dpgrnn, Erb, GTConvBlock, Grnn, GruWeights, Tra};

/// Extract a single channel (axis 1) from a `(B, 2, T, F)` tensor → `(B, T, F)`.
pub fn channel(t: &Tensor, idx: usize) -> Result<Tensor> {
    Ok(t.narrow(1, idx, 1usize)?.try_squeeze(Some(1))?)
}

/// HuggingFace repo publishing the converted checkpoint + golden.
pub const HUB_REPO: &str = "vpermilp/gtcrn";

/// The converted checkpoint's filename inside [`HUB_REPO`].
const CHECKPOINT: &str = "gtcrn.safetensors";

/// STFT geometry — the upstream GTCRN defaults (`gtcrn.py` / `infer.py`).
pub const N_FFT: usize = 512;
pub const HOP: usize = 256;

/// Input/output spectral bins (`N_FFT / 2 + 1`).
pub const N_FREQ: usize = N_FFT / 2 + 1;

/// GTCRN's analysis *and* synthesis window (`infer.py:18`): a periodic Hann
/// raised to the 0.5 power. Tabulated on the host in f64, so the transforms
/// upload their DFT kernels as constants instead of rebuilding them per run.
pub fn window() -> Window {
    let w: Vec<f32> = (0..N_FFT).map(|n| (0.5 - 0.5 * (TAU * n as f64 / N_FFT as f64).cos()).sqrt() as f32).collect();
    Window::Custom(Tensor::from_slice(w))
}

/// STFT frames a `len`-sample waveform yields under `center` padding: the
/// padded length is always `len + N_FFT`, so the count reduces to this.
pub const fn num_frames(len: usize) -> usize {
    len / HOP + 1
}

// Encoder/decoder channel/layout constants (GTCRN.__init__).
const C_IN: usize = 3; // mag + real + imag
const C_NET: usize = 16; // encoder/decoder working channels
const C_SFE: usize = 9; // 3 * SFE kernel
/// F entering the DPGRNNs: the 129 ERB bands survive two stride-2, kernel-5,
/// padding-2 encoder convs — 129 → 65 → 33 — matching `DPGRNN(16, 33, 16)`.
const DPGRNN_WIDTH: usize = 33;
const DPGRNN_HIDDEN: usize = 16;

/// PyTorch's `nn.BatchNorm2d` default epsilon. The checkpoint carries raw
/// `running_mean`/`running_var`, which [`BatchNorm2d`] folds at forward time.
const BN_EPS: f64 = 1e-5;

/// The full GTCRN enhancement network: ERB → SFE → encoder → 2× DPGRNN →
/// decoder → ERB⁻¹ → complex-ratio-mask multiply. Construct via
/// [`Gtcrn::from_hub`] / [`Gtcrn::from_safetensors`].
#[derive(Clone, Module)]
pub struct Gtcrn {
    pub erb: Erb,
    #[module(key = "encoder.en_convs")]
    encoder: [EncoderLayer; 5],
    dpgrnn1: Dpgrnn,
    dpgrnn2: Dpgrnn,
    #[module(key = "decoder.de_convs")]
    decoder: [DecoderLayer; 5],
}

/// An encoder layer is either a plain [`ConvBlock`] or a [`GTConvBlock`].
#[derive(Clone, Module)]
enum EncoderLayer {
    Conv(ConvBlock),
    Gt(Box<GTConvBlock>),
}

/// A decoder layer is either a [`GTConvBlock`] (transpose) or a [`ConvBlock`]
/// (transpose). Both decoder variants use `use_deconv=true`.
#[derive(Clone, Module)]
enum DecoderLayer {
    Gt(Box<GTConvBlock>),
    Conv(ConvBlock),
}

impl Gtcrn {
    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    /// Noisy `(B, L)` waveform → enhanced waveform, the analysis STFT,
    /// [`forward`](Self::forward) and the synthesis ISTFT in one graph.
    ///
    /// Both transforms use [`window`] with `center = True`, as `infer.py`
    /// does, so the output holds `(num_frames(L) - 1) · HOP` samples — exactly
    /// `L` when `L` is a multiple of [`HOP`].
    pub fn enhance(&self, waveform: &Tensor) -> Result<Tensor> {
        let spec = waveform.stft().n_fft(N_FFT).hop(HOP).window(window()).call()?;
        Ok(self.forward(&spec)?.istft().n_fft(N_FFT).hop(HOP).window(window()).call()?)
    }

    /// Run the network on a `(B, F=257, T, 2)` complex spectrogram, returning
    /// the enhanced `(B, 257, T, 2)` spectrogram. Mirrors `GTCRN.forward`.
    pub fn forward(&self, spec: &Tensor) -> Result<Tensor> {
        // spec: (B, F, T, 2). Split real/imag and take the magnitude, each
        // permuted from (B, F, T) to (B, T, F).
        let spec_real = spec.complex_real()?.try_permute(&[0, 2, 1])?;
        let spec_imag = spec.complex_imag()?.try_permute(&[0, 2, 1])?;
        let spec_mag = spec.magnitude(1e-12)?.try_permute(&[0, 2, 1])?;

        // feat = stack([mag, real, imag], dim=1) -> (B, 3, T, 257).
        let feat = Tensor::stack(&[&spec_mag, &spec_real, &spec_imag], 1)?;

        let feat = self.erb.bm(&feat)?; // (B,3,T,129)
        let feat = blocks::sfe(&feat, 3, C_IN)?; // (B,9,T,129)

        // Encoder: collect skip outputs.
        let mut en_outs: Vec<Tensor> = Vec::with_capacity(self.encoder.len());
        let mut x = feat;
        for layer in &self.encoder {
            x = layer.forward(&x)?;
            en_outs.push(x.clone());
        }

        x = self.dpgrnn1.forward(&x)?;
        x = self.dpgrnn2.forward(&x)?;

        // Decoder: add skip output then apply layer.
        for (skip, layer) in en_outs.iter().rev().zip(&self.decoder) {
            x = layer.forward(&x.try_add(skip)?)?;
        }

        // m = erb.bs(m_feat) -> (B, 2, T, 129).
        let m = self.erb.bs(&x)?;

        // Complex ratio mask: spec is (B,F,T,2) already; bring the mask into the
        // same layout and multiply as complex numbers.
        Ok(spec.complex_mul(&m.try_permute(&[0, 3, 2, 1])?)?) // (B,F,T,2)
    }

    // -----------------------------------------------------------------------
    // Loaders
    // -----------------------------------------------------------------------

    /// Download `CHECKPOINT` from [`HUB_REPO`] and load it.
    pub fn from_hub() -> Result<Self> {
        Self::from_hub_with_revision("main")
    }

    pub fn from_hub_with_revision(revision: &str) -> Result<Self> {
        let path = crate::hub::HubRepo::open(HUB_REPO, revision)
            .and_then(|repo| repo.get(CHECKPOINT))
            .context(HubSnafu { context: format!("{HUB_REPO}@{revision}/{CHECKPOINT}") })?;
        Self::from_safetensors(&path)
    }

    /// Load from a local converted safetensors file (see
    /// `scripts/convert_gtcrn.py`).
    pub fn from_safetensors(path: &Path) -> Result<Self> {
        Self::from_state_dict(&state::load_safetensors(path)?)
    }

    pub fn from_state_dict(sd: &StateDict) -> Result<Self> {
        let mut model = Self::with_random_weights();
        model.load_state_dict(sd, "")?;
        Ok(model)
    }

    /// Build with random weights matching the GTCRN layout (for tests/JIT
    /// exercising without a checkpoint).
    pub fn with_random_weights() -> Self {
        Self {
            erb: Erb::empty(),
            encoder: default_encoder(),
            dpgrnn1: empty_dpgrnn(),
            dpgrnn2: empty_dpgrnn(),
            decoder: default_decoder(),
        }
    }
}

impl EncoderLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Conv(c) => c.forward(x),
            Self::Gt(g) => g.forward(x),
        }
    }
}

impl DecoderLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gt(g) => g.forward(x),
            Self::Conv(c) => c.forward(x),
        }
    }
}

// --------------------------------------------------------------------------- //
// Default architecture construction (mirrors GTCRN/Encoder/Decoder __init__)
// --------------------------------------------------------------------------- //

/// `nn.Conv2d(in_ch, out_ch, kernel, groups=groups)` at PyTorch's fan-in
/// uniform init; the weight is `[out, in/groups, kH, kW]`.
fn conv2d(in_ch: usize, out_ch: usize, kernel: (usize, usize), groups: usize) -> Conv2d {
    let fan_in = in_ch / groups * kernel.0 * kernel.1;
    Conv2d::new(weight(&[out_ch, in_ch / groups, kernel.0, kernel.1], fan_in), Some(bias(out_ch, fan_in)))
        .with_groups(groups)
}

/// `nn.ConvTranspose2d(in_ch, out_ch, kernel, groups=groups)`; the weight is
/// `[in, out/groups, kH, kW]` — PyTorch's layout, which the checkpoint holds,
/// so `in` and `out` swap places relative to [`conv2d`].
fn deconv2d(in_ch: usize, out_ch: usize, kernel: (usize, usize), groups: usize) -> ConvTranspose2d {
    let fan_in = out_ch / groups * kernel.0 * kernel.1;
    ConvTranspose2d::new(weight(&[in_ch, out_ch / groups, kernel.0, kernel.1], fan_in), Some(bias(out_ch, fan_in)))
        .with_groups(groups)
}

fn weight(shape: &[usize], fan_in: usize) -> Tensor {
    fan_in_uniform(shape, fan_in, DType::Float32)
}

fn bias(out_ch: usize, fan_in: usize) -> Tensor {
    fan_in_uniform(&[out_ch], fan_in, DType::Float32)
}

fn batch_norm(channels: usize) -> BatchNorm2d {
    BatchNorm2d::with_dims(channels, BN_EPS, DType::Float32)
}

/// `nn.PReLU()`: one slope, initialized to 0.25.
fn prelu_slope() -> Tensor {
    Tensor::full(&[1], 0.25f32, DType::Float32).contiguous()
}

/// `nn.GRU(input_size, hidden_size, batch_first=True)`, one layer.
fn gru(input_size: usize, hidden_size: usize) -> GruWeights {
    let gates = hidden_size * 3;
    GruWeights {
        hidden_size,
        weight_ih: weight(&[gates, input_size], hidden_size),
        weight_hh: weight(&[gates, hidden_size], hidden_size),
        bias_ih: weight(&[gates], hidden_size),
        bias_hh: weight(&[gates], hidden_size),
    }
}

/// `ConvBlock(…, is_last)`: the conv, its BN, and a PReLU slope unless the
/// block ends in `Tanh`.
fn conv_block(conv: Conv, out_ch: usize, is_last: bool) -> ConvBlock {
    ConvBlock { conv, bn: batch_norm(out_ch), act: (!is_last).then(prelu_slope), is_last }
}

/// The `(1,5)` stride-`(1,2)` padding-`(0,2)` conv both ends of the network use.
fn wide_conv(conv: Conv) -> Conv {
    match conv {
        Conv::Normal(c) => Conv::Normal(c.with_stride((1, 2)).with_padding(((0, 0), (2, 2)))),
        Conv::Transposed(c) => Conv::Transposed(c.with_stride((1, 2)).with_padding(((0, 0), (2, 2)))),
    }
}

fn default_encoder() -> [EncoderLayer; 5] {
    [
        EncoderLayer::Conv(conv_block(wide_conv(Conv::Normal(conv2d(C_SFE, C_NET, (1, 5), 1))), C_NET, false)),
        EncoderLayer::Conv(conv_block(wide_conv(Conv::Normal(conv2d(C_NET, C_NET, (1, 5), 2))), C_NET, false)),
        EncoderLayer::Gt(Box::new(gt_conv(1, false))),
        EncoderLayer::Gt(Box::new(gt_conv(2, false))),
        EncoderLayer::Gt(Box::new(gt_conv(5, false))),
    ]
}

fn default_decoder() -> [DecoderLayer; 5] {
    [
        DecoderLayer::Gt(Box::new(gt_conv(5, true))),
        DecoderLayer::Gt(Box::new(gt_conv(2, true))),
        DecoderLayer::Gt(Box::new(gt_conv(1, true))),
        DecoderLayer::Conv(conv_block(wide_conv(Conv::Transposed(deconv2d(C_NET, C_NET, (1, 5), 2))), C_NET, false)),
        DecoderLayer::Conv(conv_block(wide_conv(Conv::Transposed(deconv2d(C_NET, 2, (1, 5), 1))), 2, true)),
    ]
}

/// `GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(p,1), dilation=(d,1))`.
/// The encoder pads T by `(0,0)` and the decoder crops it by `(2d, 2d)`; the
/// block's causal `(2d, 0)` pre-pad is folded into the depth conv either way.
fn gt_conv(dilation_t: usize, use_deconv: bool) -> GTConvBlock {
    let half = C_NET / 2;
    let pad_size = 2 * dilation_t; // (kernel[0] - 1) * dilation[0]
    let crop = if use_deconv { pad_size as isize } else { 0 };
    let mk = |in_ch: usize, out_ch: usize, kernel, groups| {
        if use_deconv {
            Conv::Transposed(deconv2d(in_ch, out_ch, kernel, groups))
        } else {
            Conv::Normal(conv2d(in_ch, out_ch, kernel, groups))
        }
    };
    let depth = match mk(C_NET, C_NET, (3, 3), C_NET) {
        Conv::Normal(c) => Conv::Normal(c.with_padding(((crop, crop), (1, 1))).with_dilation((dilation_t, 1))),
        Conv::Transposed(c) => Conv::Transposed(c.with_padding(((crop, crop), (1, 1))).with_dilation((dilation_t, 1))),
    };
    GTConvBlock {
        in_channels: C_NET,
        point_conv1: mk(half * 3, C_NET, (1, 1), 1),
        point_bn1: batch_norm(C_NET),
        point_act: prelu_slope(),
        depth_conv: depth.with_causal_pad(pad_size),
        depth_bn: batch_norm(C_NET),
        depth_act: prelu_slope(),
        point_conv2: mk(C_NET, half, (1, 1), 1),
        point_bn2: batch_norm(half),
        tra: Tra {
            gru: gru(half, half * 2),
            fc: Linear::new(weight(&[half, half * 2], half * 2), Some(bias(half, half * 2))),
        },
    }
}

pub(crate) fn empty_dpgrnn() -> Dpgrnn {
    // DPGRNN(input_size=C=16, width=F=33, hidden_size=16) from GTCRN.__init__.
    // The RNNs operate on the C=16 channel dim: intra_rnn (bidirectional,
    // GRNN input=16 split 8/8, hidden=4 → output 8·2·2=16) runs along F=33;
    // inter_rnn (unidirectional, GRNN input=16 split 8/8, hidden=8 → output 16)
    // runs along T. Width (F=33) is only the LayerNorm normalized-shape.
    let half = DPGRNN_HIDDEN / 2;
    let grnn = |hidden: usize, bidirectional: bool| Grnn {
        rnn1_f: gru(half, hidden),
        rnn1_b: bidirectional.then(|| gru(half, hidden)),
        rnn2_f: gru(half, hidden),
        rnn2_b: bidirectional.then(|| gru(half, hidden)),
    };
    let fc = || {
        Linear::new(weight(&[DPGRNN_HIDDEN, DPGRNN_HIDDEN], DPGRNN_HIDDEN), Some(bias(DPGRNN_HIDDEN, DPGRNN_HIDDEN)))
    };
    let ln = || {
        LayerNorm::new(
            ones(&[DPGRNN_WIDTH, DPGRNN_HIDDEN], DType::Float32),
            Some(zeros(&[DPGRNN_WIDTH, DPGRNN_HIDDEN], DType::Float32)),
            Dpgrnn::LN_EPS,
        )
        .with_axis(-2)
    };
    Dpgrnn {
        intra_rnn: grnn(half / 2, true),
        intra_fc: fc(),
        intra_ln: ln(),
        inter_rnn: grnn(half, false),
        inter_fc: fc(),
        inter_ln: ln(),
    }
}
