//! Short-time Fourier transform and complex helpers.
//!
//! [`Tensor::stft`] and [`Tensor::istft`] are graph ops: the DFT basis, the
//! analysis window and the overlap-add normalization are all built from
//! `arange`/`cos`/`sin`, so a transform is a convolution against a constant
//! kernel rather than a host-side FFT. That keeps an audio front-end (mel
//! filterbank, VAD, speech enhancement) inside one JIT graph. The `[2F, 1,
//! n_fft]` kernel is materialized before the convolution: left lazy it would
//! fuse in and cost a `cos`/`sin` per multiply-add, 20-30× the convolution —
//! and when the window is known on the host (a named one, or a `Custom`
//! buffer) the kernel is tabulated there in f64 and uploaded once, like the
//! mel filterbank, so no launch rebuilds it per run. Both operands are also
//! zero-padded to extents the optimizer can tile (`FRAME_ALIGN`,
//! `BIN_ALIGN`) and the surplus trimmed afterwards; the padded form never
//! leaves this crate.
//!
//! ## Conventions
//!
//! Everything follows `torch.stft` / `torch.istft`:
//!
//! - Input `[B, L]` (or `[L]`), output `[B, F, T, 2]` (or `[F, T, 2]`) with the
//!   trailing axis holding `(real, imag)` — the layout GTCRN's mask branch and
//!   Silero's magnitude branch both consume.
//! - `F = n_fft / 2 + 1` when `onesided` (the default), else `n_fft`.
//! - `T = (L' - n_fft) / hop + 1`, where `L'` is `L + n_fft` under
//!   `center` (reflect padding of `n_fft / 2` on both sides) and `L` otherwise.
//! - `window` is periodic (`torch.hann_window(periodic=true)`) unless
//!   `periodic(false)` asks for the symmetric form Kaldi uses; a `win_length`
//!   below `n_fft` is zero-padded symmetrically, as torch does.
//! - `normalized` scales the forward transform by `1 / sqrt(n_fft)`.
//!
//! The complex helpers ([`magnitude`](Tensor::magnitude),
//! [`power`](Tensor::power), [`complex_mul`](Tensor::complex_mul), …) operate
//! on that same trailing-2 layout, so a spectrogram never has to be split into
//! two tensors by hand.
//!
//! [`Tensor::mel_spectrogram`] chains `stft → power → mel_filterbank → log`
//! into `[B, n_mels, T]`, following `torchaudio.transforms.MelSpectrogram`
//! (HTK scale, no filter normalization) or `librosa.filters.mel` (Slaney scale
//! and area normalization — Whisper's `mel_filters.npz`). The filterbank is a
//! constant `[n_mels, F]` table built on the host in f64 and uploaded once: it
//! is at most a few tens of kilobytes, and building it in-graph would only add
//! kernels for the piecewise Slaney conversions with no precision to gain.

use std::f64::consts::{LOG10_2, TAU};

use bon::bon;
use snafu::{OptionExt, ensure};
use svod_dtype::DType;
use svod_ir::SInt;

use crate::Tensor;
use crate::error::{
    ExclusiveParamsSnafu, FloatDTypeRequiredSnafu, NdimExactSnafu, NdimMinimumSnafu, ParamRangeSnafu,
    ShapeMismatchSnafu,
};

type Result<T> = crate::Result<T>;

/// Floor on the window-square overlap-add divisor in [`Tensor::istft`].
///
/// `istft` divides by `Σ_t w²(n - t·hop)`. Torch requires that sum to be
/// non-zero everywhere (the NOLA condition); where it is not — `hop` larger
/// than the window support, or a window with interior zeros — the frame is
/// unrecoverable. Rather than fail at graph-build time on a value that is only
/// known after realization, the divisor is clamped to this floor, which drives
/// those samples to zero instead of to infinity.
const NOLA_EPS: f64 = 1e-11;

/// Extents the [`Tensor::stft`] / [`Tensor::istft`] convolutions are padded
/// to: the frame axis `T` up to a multiple of `FRAME_ALIGN`, the bin axis `F`
/// (and `istft`'s `hop` output channels) up to a multiple of `BIN_ALIGN`, so
/// the kernel's `2F'` channel axis is a multiple of 16. The optimizer tiles a
/// reduce only along axes it can split evenly, and the natural extents rarely
/// are: Whisper's 30 s is `T = 3001` (prime) frames of `2F = 402 = 2·3·67`
/// channels, which gets a 2×3 register tile (25 ms on CPU); `3008 × 416` gets
/// 16×4×4 (2.3 ms); and `F = 201` itself is prime, which a downstream
/// contraction over the bins (the mel filterbank) suffers as well. Both
/// padded operands are materialized before the convolution — a lazy pad
/// fuses its bounds check into the multiply-add loop (+30% on CPU), and so is
/// the convolution's result: a trim applied to it lazily would be pushed into
/// the reduce and hand the natural extents back. The extra frames are masked
/// to zero in the convolution's epilogue (they straddle the signal's tail);
/// the extra bins are zero rows of the kernel. The padding is a tiling detail
/// of this module: every public op trims it before returning.
const FRAME_ALIGN: usize = 8;
const BIN_ALIGN: usize = 8;

/// Analysis window for [`Tensor::stft`] / [`Tensor::istft`].
///
/// The named windows are cosine sums whose denominator `d` the `periodic`
/// flag picks — `n` (`torch.*_window(periodic=true)`) or `n - 1` (the
/// symmetric form, Kaldi's) — on [`Tensor::window`] and on the transforms'
/// builders alike; `Custom` carries an explicit `[n]` tensor, which is used
/// as is (a GTCRN-style `hann_window(n).sqrt()`, say).
#[derive(Clone, Default)]
pub enum Window {
    /// All ones — no tapering.
    Rectangular,
    /// `0.5 - 0.5·cos(2πk/d)`.
    #[default]
    Hann,
    /// `0.54 - 0.46·cos(2πk/d)`.
    Hamming,
    /// `(0.5 - 0.5·cos(2πk/d))^0.85` — Kaldi's default (`window_type =
    /// "povey"`), which Kaldi builds in the symmetric form (`periodic = false`).
    Povey,
    /// Explicit `[win_length]` window.
    Custom(Tensor),
}

impl std::fmt::Debug for Window {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rectangular => f.write_str("Rectangular"),
            Self::Hann => f.write_str("Hann"),
            Self::Hamming => f.write_str("Hamming"),
            Self::Povey => f.write_str("Povey"),
            Self::Custom(_) => f.write_str("Custom(..)"),
        }
    }
}

impl Window {
    /// `(a0, a1, p)` of the cosine-sum form `(a0 - a1·cos(2πk/d))^p`.
    const fn cosine_sum(&self) -> Option<(f64, f64, f64)> {
        match self {
            Self::Hann => Some((0.5, 0.5, 1.0)),
            Self::Hamming => Some((0.54, 0.46, 1.0)),
            Self::Povey => Some((0.5, 0.5, 0.85)),
            _ => None,
        }
    }

    /// `d` of the cosine-sum form: `n` (DFT-even) or `n - 1` (symmetric).
    const fn denominator(n: usize, periodic: bool) -> usize {
        if periodic || n == 1 { n } else { n - 1 }
    }
}

/// A `Custom` window must be exactly `[n]`; the graph and host paths both
/// check it here, so a `[1, n]` tensor is rejected whether or not it is
/// realized.
fn ensure_custom_shape(w: &Tensor, n: usize) -> Result<()> {
    let ndim = w.ndim()?;
    ensure!(ndim == 1, NdimExactSnafu { op: "window", expected: 1_usize, actual: ndim });
    let len = w.dim_const(0)?;
    ensure!(len == n, ShapeMismatchSnafu { context: "window", expected: format!("[{n}]"), actual: format!("[{len}]") });
    Ok(())
}

impl Tensor {
    /// Build a `[n]` window in the graph.
    ///
    /// `periodic` picks the DFT-even form (denominator `n`, what an STFT wants)
    /// over the symmetric one (denominator `n - 1`, what a filter design
    /// wants), matching `torch.hann_window`'s `periodic` flag.
    /// [`Window::Custom`] ignores `periodic` and only checks the length.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use svod_tensor::nn::Window;
    /// # use svod_dtype::DType;
    /// let w = Tensor::window(&Window::Hann, 4, true, DType::Float32).unwrap();
    /// assert_eq!(w.to_vec::<f32>().unwrap(), vec![0.0, 0.5, 1.0, 0.5]);
    /// ```
    #[track_caller]
    pub fn window(kind: &Window, n: usize, periodic: bool, dtype: DType) -> Result<Tensor> {
        origin_call!("window");
        ensure!(n > 0, ParamRangeSnafu { op: "window", param: "n", value: n.to_string(), constraint: "> 0" });
        match kind {
            Window::Custom(w) => {
                ensure_custom_shape(w, n)?;
                Ok(w.cast(dtype))
            }
            Window::Rectangular => Ok(Tensor::ones(&[n], dtype)),
            cosine => {
                let (a0, a1, p) = cosine.cosine_sum().expect("Rectangular and Custom handled above");
                let k = Tensor::arange_f64(0.0, n as f64, 1.0, DType::Float32)?;
                let phase = k.try_mul(TAU / Window::denominator(n, periodic) as f64)?;
                let w = phase.cos()?.try_mul(-a1)?.try_add(a0)?;
                Ok(if p == 1.0 { w } else { w.try_pow(p)? }.cast(dtype))
            }
        }
    }
}

/// Frequency warping of [`Tensor::mel_filterbank`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MelScale {
    /// `2595·log10(1 + f/700)` — torchaudio's default (`mel_scale="htk"`).
    #[default]
    Htk,
    /// librosa's default: linear (200/3 Hz per mel) below 1 kHz, logarithmic
    /// above (`mel_scale="slaney"`, `librosa.hz_to_mel(htk=False)`).
    Slaney,
}

/// Filter normalization of [`Tensor::mel_filterbank`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MelNorm {
    /// Each triangle scaled by `2 / (f_right - f_left)` so it has unit area in
    /// Hz — `librosa.filters.mel(norm="slaney")`, `melscale_fbanks(norm="slaney")`.
    Slaney,
}

/// Log compression applied by [`Tensor::mel_spectrogram`] / [`Tensor::mel_log`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MelLog {
    /// `ln(clamp(x, min, max))` — GigaAM's `torch.log(x.clamp(1e-9, 1e9))`.
    Ln { min: f64, max: f64 },
    /// Whisper's `log_mel_spectrogram` tail: `log10(max(x, 1e-10))`, floored
    /// at 8 below the maximum of each `[n_mels, T]` signal, then `(x + 4) / 4`.
    Whisper,
}

/// Hz per mel on the linear part of the Slaney scale, and its log step above
/// 1 kHz (`ln(6.4) / 27`): `librosa.hz_to_mel(htk=False)`.
const SLANEY_F_SP: f64 = 200.0 / 3.0;
const SLANEY_MIN_LOG_HZ: f64 = 1000.0;
const SLANEY_MIN_LOG_MEL: f64 = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;

impl MelScale {
    fn hz_to_mel(self, hz: f64) -> f64 {
        match self {
            Self::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
            Self::Slaney if hz >= SLANEY_MIN_LOG_HZ => {
                SLANEY_MIN_LOG_MEL + (hz / SLANEY_MIN_LOG_HZ).ln() / (6.4f64.ln() / 27.0)
            }
            Self::Slaney => hz / SLANEY_F_SP,
        }
    }

    fn mel_to_hz(self, mel: f64) -> f64 {
        match self {
            Self::Htk => 700.0 * (10f64.powf(mel / 2595.0) - 1.0),
            Self::Slaney if mel >= SLANEY_MIN_LOG_MEL => {
                SLANEY_MIN_LOG_HZ * ((mel - SLANEY_MIN_LOG_MEL) * (6.4f64.ln() / 27.0)).exp()
            }
            Self::Slaney => mel * SLANEY_F_SP,
        }
    }
}

/// `[n_mels, cols]` triangular filters over the first `n_fft / 2 + 1`
/// columns (the rest zero), row-major, in f64 host math.
///
/// `n_mels + 2` points are spaced evenly on the mel axis between `f_min` and
/// `f_max`; filter `m` ramps up from point `m` to `m + 1` and down to `m + 2`
/// over the FFT bin frequencies `k · sample_rate / n_fft` — the
/// `max(0, min(up, down))` form of `torchaudio.functional.melscale_fbanks`.
#[allow(clippy::too_many_arguments)]
fn mel_table(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    f_min: f64,
    f_max: f64,
    scale: MelScale,
    norm: Option<MelNorm>,
    cols: usize,
) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let (m_lo, m_hi) = (scale.hz_to_mel(f_min), scale.hz_to_mel(f_max));
    let points: Vec<f64> =
        (0..n_mels + 2).map(|i| scale.mel_to_hz(m_lo + (m_hi - m_lo) * i as f64 / (n_mels + 1) as f64)).collect();
    let mut table = vec![0f32; n_mels * cols];
    for (row, edges) in table.chunks_mut(cols).zip(points.windows(3)) {
        let (lo, mid, hi) = (edges[0], edges[1], edges[2]);
        let gain = match norm {
            Some(MelNorm::Slaney) => 2.0 / (hi - lo),
            None => 1.0,
        };
        for (k, w) in row.iter_mut().take(n_bins).enumerate() {
            let f = k as f64 * sample_rate as f64 / n_fft as f64;
            let up = (f - lo) / (mid - lo);
            let down = (hi - f) / (hi - mid);
            *w = (up.min(down).max(0.0) * gain) as f32;
        }
    }
    table
}

impl Tensor {
    /// `[n_mels, n_fft / 2 + 1]` mel filterbank — `melscale_fbanks` /
    /// `librosa.filters.mel` — as a constant tensor; `power @ fb.T` (or
    /// `fb @ power` on a `[F, T]` spectrogram) yields mel energies.
    ///
    /// The table is built on the host in f64 and uploaded once; see the
    /// module docs for why it is not assembled in-graph.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use svod_tensor::nn::{MelNorm, MelScale};
    /// # use svod_dtype::DType;
    /// let fb = Tensor::mel_filterbank(16000, 400, 80, 0.0, 8000.0, MelScale::Slaney, Some(MelNorm::Slaney), DType::Float32).unwrap();
    /// assert_eq!(fb.dims().unwrap(), vec![80, 201]);
    /// ```
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn mel_filterbank(
        sample_rate: usize,
        n_fft: usize,
        n_mels: usize,
        f_min: f64,
        f_max: f64,
        scale: MelScale,
        norm: Option<MelNorm>,
        dtype: DType,
    ) -> Result<Tensor> {
        origin_call!("mel_filterbank");
        Self::mel_filterbank_cols(sample_rate, n_fft, n_mels, f_min, f_max, scale, norm, n_fft / 2 + 1, dtype)
    }

    /// [`mel_filterbank`](Tensor::mel_filterbank) widened to `cols` zero-padded
    /// columns, for a contraction against a bin axis padded to [`BIN_ALIGN`].
    #[allow(clippy::too_many_arguments)]
    fn mel_filterbank_cols(
        sample_rate: usize,
        n_fft: usize,
        n_mels: usize,
        f_min: f64,
        f_max: f64,
        scale: MelScale,
        norm: Option<MelNorm>,
        cols: usize,
        dtype: DType,
    ) -> Result<Tensor> {
        let op = "mel_filterbank";
        ensure!(
            sample_rate > 0,
            ParamRangeSnafu { op, param: "sample_rate", value: sample_rate.to_string(), constraint: "> 0" }
        );
        ensure!(n_fft > 0, ParamRangeSnafu { op, param: "n_fft", value: n_fft.to_string(), constraint: "> 0" });
        ensure!(n_mels > 0, ParamRangeSnafu { op, param: "n_mels", value: n_mels.to_string(), constraint: "> 0" });
        ensure!(f_min >= 0.0, ParamRangeSnafu { op, param: "f_min", value: f_min.to_string(), constraint: ">= 0" });
        ensure!(f_max > f_min, ParamRangeSnafu { op, param: "f_max", value: f_max.to_string(), constraint: "> f_min" });
        ensure!(dtype.is_float(), FloatDTypeRequiredSnafu { op, arg: "dtype", dtype: dtype.clone() });
        upload(mel_table(sample_rate, n_fft, n_mels, f_min, f_max, scale, norm, cols), &[n_mels, cols], dtype)
    }

    /// The log compression of [`mel_spectrogram`](Tensor::mel_spectrogram) on
    /// its own, for a caller that trims frames first (Whisper keeps
    /// `T - 1` of the `center` frames before taking the per-signal maximum).
    #[track_caller]
    pub fn mel_log(&self, log: MelLog) -> Result<Tensor> {
        origin_call!("mel_log");
        match log {
            MelLog::Ln { min, max } => self.clamp().min(min).max(max).call()?.try_log(),
            MelLog::Whisper => {
                let ndim = self.ndim()?;
                ensure!(ndim >= 2, NdimMinimumSnafu { op: "mel_log", min: 2_usize, actual: ndim });
                let x = self.maximum(1e-10)?.try_log2()?.try_mul(LOG10_2)?;
                let floor = x.max_with().axes(&[-2isize, -1][..]).keepdim(true).call()?.try_sub(8.0)?;
                x.maximum(&floor)?.try_add(4.0)?.try_div(4.0)
            }
        }
    }
}

// =========================================================================
// Kernel construction
// =========================================================================

/// `(cos, sin)` of `2πkn/n_fft` as `[n_bins, n_fft]` Float32 matrices.
///
/// `k·n` is reduced modulo `n_fft` in integers first, so the angle stays in
/// `[0, 2π)` and f32 `cos`/`sin` keep full precision even for a large `n_fft`
/// (the naive angle reaches `2π·n_fft/2` and loses ~4 decimal digits at 512).
fn dft_basis(n_fft: usize, n_bins: usize) -> Result<(Tensor, Tensor)> {
    let k = Tensor::arange(0, Some(n_bins as i64), None)?.try_reshape([n_bins as isize, 1])?;
    let n = Tensor::arange(0, Some(n_fft as i64), None)?.try_reshape([1, n_fft as isize])?;
    let phase = k.try_mul(&n)?.try_mod(n_fft as i64)?.cast(DType::Float32).try_mul(TAU / n_fft as f64)?;
    Ok((phase.cos()?, phase.sin()?))
}

/// `[2F', 1, n_fft]` analysis kernel: rows `0..F` are `w[n]·cos(2πkn/N)` and
/// rows `F'..F' + F` are `-w[n]·sin(2πkn/N)` (the rest zero), so one `conv1d`
/// with `stride = hop` emits the framed real and imaginary parts stacked on
/// the channel axis, each half padded to `F'` bins.
fn analysis_kernel(n_fft: usize, n_bins: usize, padded_bins: usize, win: &Tensor, dtype: DType) -> Result<Tensor> {
    let (cos, sin) = dft_basis(n_fft, n_bins)?;
    let pad = [(0, (padded_bins - n_bins) as isize), (0, 0)];
    let re = cos.try_mul(win)?.try_pad(&pad)?;
    let im = sin.neg().try_mul(win)?.try_pad(&pad)?;
    Ok(Tensor::cat(&[&re, &im], 0)?.try_reshape([2 * padded_bins as isize, 1, n_fft as isize])?.cast(dtype))
}

/// `[2F', n_fft]` synthesis table: row `k` (`F' + k`) times the real
/// (imaginary) part of bin `k` gives that bin's share of the windowed inverse
/// DFT of a frame, the rows past `F` zero.
///
/// Under `onesided`, bins `1..F-1` stand in for their conjugate partners and
/// carry a factor of two; DC and (for even `n_fft`) Nyquist carry one, and
/// their imaginary parts are dropped exactly as `irfft` — and therefore
/// `torch.istft` — does.
fn synthesis_kernel(
    n_fft: usize,
    n_bins: usize,
    padded_bins: usize,
    onesided: bool,
    win: &Tensor,
    dtype: DType,
) -> Result<Tensor> {
    let (cos, sin) = dft_basis(n_fft, n_bins)?;
    let inv = 1.0 / n_fft as f64;
    let (re, im) = if onesided {
        // `edge` is 1.0 on the bins without a conjugate partner, 0.0 elsewhere;
        // `2 - edge` and `2·(1 - edge)` are then the real/imaginary weights.
        let k = Tensor::arange(0, Some(n_bins as i64), None)?;
        let nyquist = if n_fft.is_multiple_of(2) { (n_fft / 2) as i64 } else { -1 };
        let edge = k.try_eq(0_i64)?.try_bitor(&k.try_eq(nyquist)?)?.cast(DType::Float32);
        let edge = edge.try_reshape([n_bins as isize, 1])?;
        (edge.try_mul(-1.0)?.try_add(2.0)?, edge.try_add(-1.0)?.try_mul(-2.0)?)
    } else {
        (Tensor::ones(&[1, 1], DType::Float32), Tensor::ones(&[1, 1], DType::Float32))
    };
    let pad = [(0, (padded_bins - n_bins) as isize), (0, 0)];
    let re = cos.try_mul(&re)?.try_mul(inv)?.try_mul(win)?.try_pad(&pad)?;
    let im = sin.try_mul(&im)?.try_mul(-inv)?.try_mul(win)?.try_pad(&pad)?;
    Ok(Tensor::cat(&[&re, &im], 0)?.cast(dtype))
}

/// Overlap-add as one convolution over the frame axis. A frame's `n_fft`
/// synthesized samples span `taps = ⌈n_fft / hop⌉` output blocks of `hop`
/// samples, and output block `m` sums block `k` of frame `m - k` over
/// `k < taps`: with the frames zero-padded by `taps - 1` on the left, that is
/// `conv1d` over `[B, rows, frames]` with a `[hop', rows, taps]` weight,
/// `W[j, c, kk] = S[c, (taps - 1 - kk)·hop + j]` for a `[rows, n_fft]`
/// synthesis table `S` (zero past `n_fft`), the rows past `hop` zero. The
/// frame IDFT and the overlap-add are then a single reduce of `rows × taps`
/// per output sample — `2F'·⌈n_fft/hop⌉`, against the `n_fft × 2F` of a
/// stride-dilated `conv_transpose`.
fn ola_table(table: &[f32], n_fft: usize, rows: usize, hop: usize, padded_hop: usize, taps: usize) -> Vec<f32> {
    let mut w = vec![0f32; padded_hop * rows * taps];
    for j in 0..hop {
        for c in 0..rows {
            for kk in 0..taps {
                let n = (taps - 1 - kk) * hop + j;
                if n < n_fft {
                    w[(j * rows + c) * taps + kk] = table[c * n_fft + n];
                }
            }
        }
    }
    w
}

/// Graph twin of [`ola_table`] over a `[rows, n_fft]` table.
fn ola_weight(table: &Tensor, n_fft: usize, hop: usize, padded_hop: usize, taps: usize) -> Result<Tensor> {
    let rows = table.dim_const(0)? as isize;
    table
        .try_pad(&[(0, 0), (0, (taps * hop - n_fft) as isize)])?
        .try_reshape([rows, taps as isize, hop as isize])?
        .flip(&[1])?
        .try_pad(&[(0, 0), (0, 0), (0, (padded_hop - hop) as isize)])?
        .try_permute(&[2, 0, 1])
}

/// `1 / max(Σ_t w²[n - t·hop], NOLA_EPS)` over samples `start..start + take`
/// of the overlap-add buffer — the per-sample divisor of `torch.istft`.
fn inverse_envelope(win: &[f64], hop: usize, frames: usize, start: usize, take: usize) -> Vec<f32> {
    let mut envelope = vec![0f64; (frames - 1) * hop + win.len()];
    for t in 0..frames {
        for (n, w) in win.iter().enumerate() {
            envelope[t * hop + n] += w * w;
        }
    }
    envelope[start..start + take].iter().map(|&v| (1.0 / v.max(NOLA_EPS)) as f32).collect()
}

/// The window zero-padded to `n_fft`, centered as `torch.stft` centers it.
fn framed_window(kind: &Window, n_fft: usize, win_length: usize, periodic: bool, dtype: DType) -> Result<Tensor> {
    let w = Tensor::window(kind, win_length, periodic, dtype)?;
    if win_length == n_fft {
        return Ok(w);
    }
    let left = ((n_fft - win_length) / 2) as isize;
    w.try_pad(&[(left, (n_fft - win_length) as isize - left)])
}

/// [`framed_window`] as host f64 values when no graph has to run for it: a
/// named window, or a `Custom` one already backed by a Float32 buffer (a
/// `from_slice` window). A lazy `Custom` window yields `None`, and its
/// caller builds the DFT kernel in the graph instead; its shape is checked
/// here all the same.
fn host_framed_window(kind: &Window, n_fft: usize, win_length: usize, periodic: bool) -> Result<Option<Vec<f64>>> {
    let w: Vec<f64> = match kind {
        Window::Custom(w) => {
            ensure_custom_shape(w, win_length)?;
            match w.buffer().is_some().then(|| w.as_vec::<f32>().ok()).flatten() {
                Some(host) => host.into_iter().map(f64::from).collect(),
                None => return Ok(None),
            }
        }
        Window::Rectangular => vec![1.0; win_length],
        cosine => {
            let (a0, a1, p) = cosine.cosine_sum().expect("Rectangular and Custom handled above");
            let denom = Window::denominator(win_length, periodic) as f64;
            (0..win_length).map(|k| (a0 - a1 * (TAU * k as f64 / denom).cos()).powf(p)).collect()
        }
    };
    let mut framed = vec![0.0; n_fft];
    let left = (n_fft - win_length) / 2;
    framed[left..left + win_length].copy_from_slice(&w);
    Ok(Some(framed))
}

/// Host twin of [`analysis_kernel`] / [`synthesis_kernel`], `[2·half, n_fft]`
/// row-major with `half >= F` (the surplus rows of each half zero): row `k` is
/// `w[n]·re(k)·cos(2πkn/N)`, row `half + k` is `w[n]·im(k)·sin(2πkn/N)`.
/// Built in f64 and uploaded once, the way [`mel_table`] is, so the kernel
/// enters a plan as an input buffer rather than a launch.
fn dft_table(n_fft: usize, n_bins: usize, half: usize, win: &[f64], weights: impl Fn(usize) -> (f64, f64)) -> Vec<f32> {
    let mut table = vec![0f32; 2 * half * n_fft];
    for k in 0..n_bins {
        let (re, im) = weights(k);
        for n in 0..n_fft {
            let angle = TAU * ((k * n) % n_fft) as f64 / n_fft as f64;
            table[k * n_fft + n] = (win[n] * re * angle.cos()) as f32;
            table[(half + k) * n_fft + n] = (win[n] * im * angle.sin()) as f32;
        }
    }
    table
}

/// Per-bin `(re, im)` weights of the synthesis kernel — see [`synthesis_kernel`].
fn synthesis_weights(n_fft: usize, onesided: bool) -> impl Fn(usize) -> (f64, f64) {
    let inv = 1.0 / n_fft as f64;
    let nyquist = if n_fft.is_multiple_of(2) { Some(n_fft / 2) } else { None };
    move |k| {
        let edge = onesided && (k == 0 || Some(k) == nyquist);
        let (re, im) = if onesided { if edge { (1.0, 0.0) } else { (2.0, 2.0) } } else { (1.0, 1.0) };
        (re * inv, -im * inv)
    }
}

/// Upload a host table as a `shape`d constant of `dtype`.
fn upload(table: Vec<f32>, shape: &[usize], dtype: DType) -> Result<Tensor> {
    let t = Tensor::from_slice(table).try_reshape(shape.iter().map(|&d| d as isize).collect::<Vec<_>>())?;
    Ok(if dtype == DType::Float32 { t } else { t.cast(dtype) })
}

/// `(hop, win_length)` with the torch defaults applied and validated.
fn resolve(op: &'static str, n_fft: usize, hop: Option<usize>, win_length: Option<usize>) -> Result<(usize, usize)> {
    ensure!(n_fft > 0, ParamRangeSnafu { op, param: "n_fft", value: n_fft.to_string(), constraint: "> 0" });
    let hop = hop.unwrap_or((n_fft / 4).max(1));
    ensure!(hop > 0, ParamRangeSnafu { op, param: "hop", value: hop.to_string(), constraint: "> 0" });
    let win_length = win_length.unwrap_or(n_fft);
    ensure!(
        win_length > 0 && win_length <= n_fft,
        ParamRangeSnafu { op, param: "win_length", value: win_length.to_string(), constraint: "in 1..=n_fft" }
    );
    Ok((hop, win_length))
}

/// Reflect-pad the trailing axis only, leaving every leading axis — a symbolic
/// batch included — untouched. [`Tensor::pad_with`] needs every dimension to be
/// concrete, which an unbound `Variable` batch is not.
fn reflect_pad_last(x: &Tensor, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let ndim = x.ndim()?;
    let len = x.dim_const(-1)?;
    ensure!(
        pad < len,
        ParamRangeSnafu { op: "stft", param: "n_fft / 2", value: pad.to_string(), constraint: "< signal length" }
    );
    let axis = ndim as isize - 1;
    let wing = |begin: usize, end: usize| -> Result<Tensor> {
        let mut ranges: Vec<Option<(SInt, SInt)>> = vec![None; ndim];
        ranges[ndim - 1] = Some((SInt::from(begin), SInt::from(end)));
        x.try_shrink(ranges)?.flip(&[axis])
    };
    let (left, right) = (wing(1, 1 + pad)?, wing(len - 1 - pad, len - 1)?);
    Tensor::cat(&[&left, x, &right], axis)
}

// =========================================================================
// stft / istft
// =========================================================================

#[bon]
impl Tensor {
    /// Short-time Fourier transform of a `[B, L]` (or `[L]`) real signal,
    /// returning `[B, F, T, 2]` (or `[F, T, 2]`) with `(real, imag)` trailing.
    ///
    /// Mirrors `torch.stft(..., return_complex=false)`. Implemented as one
    /// `conv1d` against a materialized `[2F, 1, n_fft]` windowed DFT kernel,
    /// so the whole transform stays in the graph and the batch axis may be
    /// symbolic.
    ///
    /// Defaults: `hop = n_fft / 4`, `win_length = n_fft`, periodic Hann window,
    /// `center`, `onesided`, no normalization. `periodic(false)` builds a
    /// named window in its symmetric form (denominator `win_length - 1`, as
    /// Kaldi does); `Window::Custom` ignores the flag.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use svod_tensor::nn::Window;
    /// let x = Tensor::from_slice(vec![0.25f32; 64]);
    /// let spec = x.stft().n_fft(16).hop(4).window(Window::Hann).call().unwrap();
    /// assert_eq!(spec.dims().unwrap(), vec![9, 17, 2]);
    /// ```
    #[builder]
    #[track_caller]
    pub fn stft(
        &self,
        n_fft: usize,
        hop: Option<usize>,
        win_length: Option<usize>,
        #[builder(default)] window: Window,
        #[builder(default = true)] periodic: bool,
        #[builder(default = true)] center: bool,
        #[builder(default = true)] onesided: bool,
        #[builder(default = false)] normalized: bool,
    ) -> Result<Tensor> {
        origin_call!("stft");
        let (spec, n_bins, frames) =
            self.stft_aligned(n_fft, hop, win_length, &window, periodic, center, onesided, normalized, None, false)?;
        let out = spec.narrow(1, 0_usize, n_bins)?.narrow(2, 0_usize, frames)?;
        if self.ndim()? == 1 { out.try_squeeze(Some(0)) } else { Ok(out) }
    }

    /// [`stft`](Tensor::stft) at the extents its convolution runs at, before
    /// the trim: `[B, F', T', 2]` — the batch axis kept even for a `[L]` input
    /// — with the bin and frame counts rounded up to tileable multiples
    /// ([`BIN_ALIGN`], [`FRAME_ALIGN`]), plus the true `(F, T)`. The surplus
    /// bins and frames are zero, so a consumer that contracts over the bins or
    /// maps frames elementwise can keep the tileable extents and trim at the
    /// end, as [`mel_spectrogram`](Tensor::mel_spectrogram) does.
    ///
    /// `preemphasis` filters the signal, `x[n] - a·x[n - 1]` (`x[-1] = 0`),
    /// and `remove_dc` subtracts each frame's mean over the `win_length`
    /// samples under the window before it — Kaldi's `preemph_coeff` and
    /// `remove_dc_offset`. Kaldi filters each frame on its own, with the
    /// frame's first sample as its own predecessor, so the signal-wide
    /// filter equals it only where the window's first weight is zero: with
    /// `preemphasis`, the window must be host-known and start at exactly
    /// zero (Hann, Povey; not Hamming, not Rectangular, not a lazy `Custom`
    /// one), and anything else is rejected. The mean removal is a spectral
    /// correction: `DFT(w·(x - c)) = DFT(w·x) - c·DFT(w)`, the means coming
    /// from a box convolution at the frame stride and `(1 - a)·DFT(w)` being
    /// a constant.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stft_aligned(
        &self,
        n_fft: usize,
        hop: Option<usize>,
        win_length: Option<usize>,
        window: &Window,
        periodic: bool,
        center: bool,
        onesided: bool,
        normalized: bool,
        preemphasis: Option<f64>,
        remove_dc: bool,
    ) -> Result<(Tensor, usize, usize)> {
        let (hop, win_length) = resolve("stft", n_fft, hop, win_length)?;
        let ndim = self.ndim()?;
        ensure!(ndim == 1 || ndim == 2, NdimExactSnafu { op: "stft", expected: 2_usize, actual: ndim });
        let dtype = self.dtype();
        ensure!(dtype.is_float(), FloatDTypeRequiredSnafu { op: "stft", arg: "input", dtype: dtype.clone() });
        let left = (n_fft - win_length) / 2;
        let host_win = host_framed_window(window, n_fft, win_length, periodic)?;
        ensure!(
            preemphasis.is_none() || host_win.as_ref().is_some_and(|w| w[left] == 0.0),
            ParamRangeSnafu {
                op: "stft",
                param: "window",
                value: format!("{window:?}"),
                constraint: "a host-known window whose first weight is zero (Hann, Povey) under `preemphasis`",
            }
        );

        let x = if ndim == 1 { self.try_unsqueeze(0)? } else { self.clone() };
        let x = if center { reflect_pad_last(&x, n_fft / 2)? } else { x };
        let len = x.dim_const(-1)?;
        ensure!(
            len >= n_fft,
            ParamRangeSnafu { op: "stft", param: "padded length", value: len.to_string(), constraint: ">= n_fft" }
        );
        let frames = (len - n_fft) / hop + 1;
        let padded_frames = frames.next_multiple_of(FRAME_ALIGN);
        // Samples the padded frame count reads; the signal may already hold
        // more (a tail shorter than `hop`), which the convolution ignores.
        let padded_len = (padded_frames - 1) * hop + n_fft;
        let x = if padded_len > len { x.try_pad(&[(0, 0), (0, (padded_len - len) as isize)])? } else { x };
        // The frame means read the raw signal, the transform the emphasized
        // one; each convolution reads a plain buffer, never a pad gate.
        let raw = if remove_dc { x.contiguous() } else { x };
        let x = match preemphasis {
            Some(a) => {
                let previous = raw.try_pad(&[(0, 0), (1, 0)])?.narrow(-1, 0_usize, raw.dim_const(-1)?)?;
                raw.try_sub(&previous.try_mul(a)?)?.contiguous()
            }
            None if padded_len > len => raw.contiguous(),
            None => raw.clone(),
        };

        let n_bins = if onesided { n_fft / 2 + 1 } else { n_fft };
        let padded_bins = n_bins.next_multiple_of(BIN_ALIGN);
        let rows = 2 * padded_bins;
        // `dc` is `(1 - a)·DFT(w)` per row — the kernel's row sums — with the
        // analysis sign convention: what subtracting a constant from a frame
        // subtracts from its windowed spectrum.
        let dc_scale = 1.0 - preemphasis.unwrap_or(0.0);
        let (kernel, dc) = match host_win {
            Some(win) => {
                let table = dft_table(n_fft, n_bins, padded_bins, &win, |_| (1.0, -1.0));
                let dc = table
                    .chunks(n_fft)
                    .map(|row| (row.iter().map(|&v| f64::from(v)).sum::<f64>() * dc_scale) as f32)
                    .collect();
                (upload(table, &[rows, 1, n_fft], dtype.clone())?, upload(dc, &[rows, 1], dtype)?)
            }
            None => {
                let win = framed_window(window, n_fft, win_length, periodic, dtype.clone())?;
                let kernel = analysis_kernel(n_fft, n_bins, padded_bins, &win, dtype)?.contiguous();
                (kernel.clone(), kernel.sum(-1isize)?.try_mul(dc_scale)?)
            }
        };

        // [B, 1, L'] -> [B, 2F', T'] -> [B, 2, F', T'] -> [B, F', T', 2]. The
        // padding frames straddle the signal's tail (`hop < n_fft`), so a
        // host `[1, 1, T']` frame mask zeroes them in the convolution's
        // epilogue; it is an input buffer like the kernel, not a launch.
        let y = x.try_unsqueeze(1)?.conv1d().weight(&kernel).stride(hop).call()?;
        let y = if remove_dc {
            let taps =
                (0..n_fft).map(|n| if (left..left + win_length).contains(&n) { 1.0 / win_length as f32 } else { 0.0 });
            let mean = raw
                .try_unsqueeze(1)?
                .conv1d()
                .weight(&upload(taps.collect(), &[1, 1, n_fft], y.dtype())?)
                .stride(hop)
                .call()?;
            y.try_sub(&mean.try_mul(&dc)?)?
        } else {
            y
        };
        let y = if padded_frames > frames {
            let valid = (0..padded_frames).map(|t| if t < frames { 1.0 } else { 0.0 }).collect();
            y.try_mul(&upload(valid, &[1, 1, padded_frames], y.dtype())?)?
        } else {
            y
        };
        let y = y.contiguous();
        let y = if normalized { y.try_mul(1.0 / (n_fft as f64).sqrt())? } else { y };
        Ok((y.unflatten(1, &[2, padded_bins as isize])?.try_permute(&[0, 2, 3, 1])?, n_bins, frames))
    }

    /// Inverse short-time Fourier transform of a `[B, F, T, 2]` (or `[F, T, 2]`)
    /// spectrogram, returning `[B, L]` (or `[L]`).
    ///
    /// Mirrors `torch.istft`: each frame is inverse-transformed, multiplied by
    /// the synthesis window, overlap-added, and divided by the overlap-add of
    /// the squared window. The first three steps are one `conv1d` over the
    /// frame axis against a host-built `[hop, 2F, ⌈n_fft/hop⌉]` kernel (see
    /// `ola_table`) and the divisor is a host-built `[L]` table, so the
    /// transform is one reduce of `2F·⌈n_fft/hop⌉` per output sample plus an
    /// elementwise epilogue.
    ///
    /// The parameters (`periodic` included: the synthesis window is the
    /// analysis one) must match the [`stft`](Tensor::stft) that produced the
    /// input. Without `length`, the result is `(T - 1) · hop` samples under
    /// `center` (the analysis padding is trimmed) and `(T - 1) · hop + n_fft`
    /// otherwise; with `length` it is trimmed or zero-padded to exactly that.
    ///
    /// # NOLA
    ///
    /// Reconstruction is exact only where the window satisfies NOLA — the
    /// overlap-add of `window²` is non-zero at every sample. `hann` at 75%
    /// overlap does; a `hop` wider than `win_length` does not. Samples where
    /// the divisor falls below `1e-11` are returned as zero rather than as
    /// infinity, so a NOLA violation shows up as silence, not as NaN.
    #[builder]
    #[track_caller]
    pub fn istft(
        &self,
        n_fft: usize,
        hop: Option<usize>,
        win_length: Option<usize>,
        #[builder(default)] window: Window,
        #[builder(default = true)] periodic: bool,
        #[builder(default = true)] center: bool,
        #[builder(default = true)] onesided: bool,
        #[builder(default = false)] normalized: bool,
        length: Option<usize>,
    ) -> Result<Tensor> {
        origin_call!("istft");
        let (hop, win_length) = resolve("istft", n_fft, hop, win_length)?;
        let ndim = self.ndim()?;
        ensure!(ndim == 3 || ndim == 4, NdimExactSnafu { op: "istft", expected: 4_usize, actual: ndim });
        let dtype = self.dtype();
        ensure!(dtype.is_float(), FloatDTypeRequiredSnafu { op: "istft", arg: "input", dtype: dtype.clone() });

        let x = if ndim == 3 { self.try_unsqueeze(0)? } else { self.clone() };
        let shape = x.shape()?;
        let n_bins = x.dim_const(1)?;
        let expected = if onesided { n_fft / 2 + 1 } else { n_fft };
        let frames = x.dim_const(2)?;
        let pair = x.dim_const(3)?;
        ensure!(
            n_bins == expected && pair == 2,
            ShapeMismatchSnafu {
                context: "istft",
                expected: format!("[.., {expected}, T, 2]"),
                actual: format!("[.., {n_bins}, {frames}, {pair}]"),
            }
        );
        ensure!(
            frames > 0,
            ParamRangeSnafu { op: "istft", param: "frames", value: frames.to_string(), constraint: "> 0" }
        );

        let taps = n_fft.div_ceil(hop);
        let padded_bins = n_bins.next_multiple_of(BIN_ALIGN);
        let padded_hop = hop.next_multiple_of(BIN_ALIGN);
        let rows = 2 * padded_bins;
        // Output blocks of `hop` samples: `frames + taps - 1` cover the
        // overlap-add buffer, rounded up to a tileable count. The convolution
        // reads `taps - 1` zero frames before the first and the surplus after
        // the last, so the input needs no boundary gate.
        let out_frames = (frames + taps - 1).next_multiple_of(FRAME_ALIGN);
        let frame_pad = ((taps - 1) as isize, (out_frames - frames) as isize);

        // [B, F, T, 2] -> [B, 2, F', T''] -> [B, 2F', T'']; the batch stays symbolic.
        let z = x
            .try_permute(&[0, 3, 1, 2])?
            .try_pad(&[(0, 0), (0, 0), (0, (padded_bins - n_bins) as isize), frame_pad])?
            .try_reshape([shape[0].clone(), SInt::from(rows), SInt::from(out_frames + taps - 1)])?;
        let z = if normalized { z.try_mul((n_fft as f64).sqrt())? } else { z };
        let z = z.contiguous();

        let out_len = (frames - 1) * hop + n_fft;
        let start = if center { n_fft / 2 } else { 0 };
        let keep = length.unwrap_or(out_len - 2 * start);
        let take = keep.min(out_len - start);
        // `[B, hop', M]` blocks -> `[B, M·hop]` samples -> the kept range.
        let samples = |blocks: Tensor| -> Result<Tensor> {
            let batch = blocks.shape()?[0].clone();
            blocks
                .try_permute(&[0, 2, 1])?
                .narrow(2, 0_usize, hop)?
                .try_reshape([batch, SInt::from(out_frames * hop)])?
                .narrow(-1, start, take)
        };

        // The overlap-add weight and the inverse window envelope: host tables
        // for a host-known window, otherwise built in-graph (the envelope by
        // the same overlap-add of window² over a frame of ones).
        let (weight, inverse) = match host_framed_window(&window, n_fft, win_length, periodic)? {
            Some(win) => {
                let table = dft_table(n_fft, n_bins, padded_bins, &win, synthesis_weights(n_fft, onesided));
                (
                    upload(
                        ola_table(&table, n_fft, rows, hop, padded_hop, taps),
                        &[padded_hop, rows, taps],
                        dtype.clone(),
                    )?,
                    upload(inverse_envelope(&win, hop, frames, start, take), &[1, take], dtype)?,
                )
            }
            None => {
                let win = framed_window(&window, n_fft, win_length, periodic, dtype.clone())?;
                let table = synthesis_kernel(n_fft, n_bins, padded_bins, onesided, &win, dtype.clone())?;
                let wsq = win.square().try_reshape([1, n_fft as isize])?;
                let ones = Tensor::ones(&[1, 1, frames], dtype).try_pad(&[(0, 0), (0, 0), frame_pad])?;
                let envelope = ones.conv1d().weight(&ola_weight(&wsq, n_fft, hop, padded_hop, taps)?).call()?;
                (
                    ola_weight(&table, n_fft, hop, padded_hop, taps)?.contiguous(),
                    samples(envelope)?.maximum(NOLA_EPS)?.reciprocal()?,
                )
            }
        };

        // Materialized at `[B, hop', M]`: read lazily, the sample trim would be
        // pushed into the reduce and hand it one flat, untileable sample axis.
        let y = samples(z.conv1d().weight(&weight).call()?.contiguous())?.try_mul(&inverse)?;
        let y = if take < keep { y.try_pad(&[(0, 0), (0, (keep - take) as isize)])? } else { y };
        if ndim == 3 { y.try_squeeze(Some(0)) } else { Ok(y) }
    }

    /// Mel spectrogram of a `[B, L]` (or `[L]`) signal: `[B, n_mels, T]` (or
    /// `[n_mels, T]`), with `T` as [`stft`](Tensor::stft) counts frames.
    ///
    /// `stft` (its `hop`, `win_length`, `window`, `center` defaults) → `|X|^power`
    /// → [`mel_filterbank`](Tensor::mel_filterbank) → optional
    /// [`mel_log`](Tensor::mel_log). With the defaults this is
    /// `torchaudio.transforms.MelSpectrogram` (HTK scale, no normalization,
    /// power 2, `f_max = sample_rate / 2`); `MelScale::Slaney` with
    /// `MelNorm::Slaney` is `librosa.feature.melspectrogram`.
    ///
    /// Kaldi's `fbank` is the same chain with `preemphasis` (`x[n] - a·x[n - 1]`
    /// on the signal, accepted only with a window whose first weight is
    /// zero — Hann, Povey — so it equals Kaldi's per-frame filter exactly),
    /// `remove_dc` (each frame's mean over the `win_length` samples under the
    /// window subtracted before the pre-emphasis), a `Window::Povey` in its
    /// symmetric form (`periodic(false)`) and a `filterbank` of its own: a
    /// `[n_mels, F]` table given in place of `n_mels` (`sample_rate`, `f_min`,
    /// `f_max`, `mel_scale` and `norm` are then unused). A `filterbank` still
    /// in the graph costs one launch per run to widen; a host-backed one is
    /// widened on the host.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// let x = Tensor::from_slice(vec![0.25f32; 1600]);
    /// let mel = x.mel_spectrogram().sample_rate(16000).n_fft(400).hop(160).n_mels(64).call().unwrap();
    /// assert_eq!(mel.dims().unwrap(), vec![64, 11]);
    /// ```
    #[builder]
    #[track_caller]
    pub fn mel_spectrogram(
        &self,
        sample_rate: Option<usize>,
        n_fft: usize,
        hop: Option<usize>,
        win_length: Option<usize>,
        #[builder(default)] window: Window,
        #[builder(default = true)] periodic: bool,
        #[builder(default = true)] center: bool,
        preemphasis: Option<f64>,
        #[builder(default = false)] remove_dc: bool,
        n_mels: Option<usize>,
        filterbank: Option<&Tensor>,
        #[builder(default = 0.0)] f_min: f64,
        f_max: Option<f64>,
        #[builder(default)] mel_scale: MelScale,
        norm: Option<MelNorm>,
        #[builder(default = 2.0)] power: f64,
        log: Option<MelLog>,
    ) -> Result<Tensor> {
        origin_call!("mel_spectrogram");
        let op = "mel_spectrogram";
        ensure!(power > 0.0, ParamRangeSnafu { op, param: "power", value: power.to_string(), constraint: "> 0" });
        ensure!(
            n_mels.is_some() != filterbank.is_some(),
            ExclusiveParamsSnafu { op, options: "`n_mels` and `filterbank`" }
        );
        // The STFT's padding bins and frames ride through the filterbank
        // contraction so it, too, gets tileable extents: the filterbank is
        // widened with zero columns over the padding bins (`F = n_fft / 2 + 1`
        // is prime for the usual `n_fft`), and the padding frames hold zero
        // power, which no log here lets past the true frames: `Ln` is
        // elementwise, and the Whisper per-signal maximum floors every value
        // at `log10(1e-10)`, the exact value of a zero frame, so those frames
        // never exceed the real maximum.
        let (spec, n_bins, frames) =
            self.stft_aligned(n_fft, hop, win_length, &window, periodic, center, true, false, preemphasis, remove_dc)?;
        let energy = spec.power()?;
        let energy = if power == 2.0 {
            energy
        } else if power == 1.0 {
            energy.try_sqrt()?
        } else {
            energy.try_pow(power / 2.0)?
        };
        let cols = energy.dim_const(1)?;
        let fb = match (filterbank, n_mels) {
            (Some(fb), _) => {
                let ndim = fb.ndim()?;
                ensure!(ndim == 2, NdimExactSnafu { op, expected: 2_usize, actual: ndim });
                let (rows, bins) = (fb.dim_const(0)?, fb.dim_const(1)?);
                ensure!(
                    bins == n_bins,
                    ShapeMismatchSnafu {
                        context: op,
                        expected: format!("[n_mels, {n_bins}]"),
                        actual: format!("[{rows}, {bins}]")
                    }
                );
                match fb.buffer().is_some().then(|| fb.as_vec::<f32>().ok()).flatten() {
                    Some(host) => {
                        let mut table = vec![0f32; rows * cols];
                        for (wide, row) in table.chunks_mut(cols).zip(host.chunks(n_bins)) {
                            wide[..n_bins].copy_from_slice(row);
                        }
                        upload(table, &[rows, cols], self.dtype())?
                    }
                    None => fb.try_pad(&[(0, 0), (0, (cols - n_bins) as isize)])?.cast(self.dtype()).contiguous(),
                }
            }
            (None, n_mels) => {
                let n_mels = n_mels.expect("exactly one of n_mels and filterbank is set");
                let sample_rate = sample_rate.context(ParamRangeSnafu {
                    op,
                    param: "sample_rate",
                    value: "none",
                    constraint: "set when `n_mels` builds the filterbank",
                })?;
                let f_max = f_max.unwrap_or(sample_rate as f64 / 2.0);
                Tensor::mel_filterbank_cols(
                    sample_rate,
                    n_fft,
                    n_mels,
                    f_min,
                    f_max,
                    mel_scale,
                    norm,
                    cols,
                    self.dtype(),
                )?
            }
        };
        let mel = fb.matmul(&energy)?;
        let mel = match log {
            Some(log) => mel.mel_log(log)?,
            None => mel,
        };
        let mel = mel.narrow(-1, 0_usize, frames)?;
        if self.ndim()? == 1 { mel.try_squeeze(Some(0)) } else { Ok(mel) }
    }
}

// =========================================================================
// Complex helpers on the trailing-2 layout
// =========================================================================

impl Tensor {
    /// `(real, imag)` of a `[..., 2]` complex tensor, each `[...]`.
    fn complex_parts(&self, op: &'static str) -> Result<(Tensor, Tensor)> {
        let ndim = self.ndim()?;
        ensure!(ndim >= 1, NdimMinimumSnafu { op, min: 1_usize, actual: ndim });
        let last = self.dim_const(-1)?;
        ensure!(
            last == 2,
            ShapeMismatchSnafu { context: op, expected: "[.., 2]".to_string(), actual: format!("[.., {last}]") }
        );
        Ok((
            self.narrow(-1, 0_usize, 1_usize)?.try_squeeze(Some(-1))?,
            self.narrow(-1, 1_usize, 1_usize)?.try_squeeze(Some(-1))?,
        ))
    }

    /// Real part of a `[..., 2]` complex tensor.
    #[track_caller]
    pub fn complex_real(&self) -> Result<Tensor> {
        origin_call!("complex_real");
        Ok(self.complex_parts("complex_real")?.0)
    }

    /// Imaginary part of a `[..., 2]` complex tensor.
    #[track_caller]
    pub fn complex_imag(&self) -> Result<Tensor> {
        origin_call!("complex_imag");
        Ok(self.complex_parts("complex_imag")?.1)
    }

    /// `re² + im²` of a `[..., 2]` complex tensor — the power spectrum, without
    /// the square root a magnitude would pay for.
    #[track_caller]
    pub fn power(&self) -> Result<Tensor> {
        origin_call!("power");
        let (re, im) = self.complex_parts("power")?;
        re.square().try_add(im.square())
    }

    /// `sqrt(re² + im² + eps)` of a `[..., 2]` complex tensor.
    ///
    /// `eps` guards the gradient at the origin; pass `0.0` for the plain
    /// modulus (or call [`complex_abs`](Tensor::complex_abs)).
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// let z = Tensor::from_slice([3.0f32, 4.0]).try_reshape([1, 2]).unwrap();
    /// assert_eq!(z.magnitude(0.0).unwrap().to_vec::<f32>().unwrap(), vec![5.0]);
    /// ```
    #[track_caller]
    pub fn magnitude(&self, eps: f64) -> Result<Tensor> {
        origin_call!("magnitude");
        let p = self.power()?;
        if eps == 0.0 { p.try_sqrt() } else { p.try_add(eps)?.try_sqrt() }
    }

    /// Modulus of a `[..., 2]` complex tensor — [`magnitude`](Tensor::magnitude)
    /// with no epsilon.
    #[track_caller]
    pub fn complex_abs(&self) -> Result<Tensor> {
        origin_call!("complex_abs");
        self.magnitude(0.0)
    }

    /// Elementwise complex product of two `[..., 2]` tensors:
    /// `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`.
    ///
    /// This is the complex ratio mask GTCRN applies to its input spectrogram.
    #[track_caller]
    pub fn complex_mul(&self, other: &Tensor) -> Result<Tensor> {
        origin_call!("complex_mul");
        let (a, b) = self.complex_parts("complex_mul")?;
        let (c, d) = other.complex_parts("complex_mul")?;
        let re = a.try_mul(&c)?.try_sub(&b.try_mul(&d)?)?;
        let im = a.try_mul(&d)?.try_add(&b.try_mul(&c)?)?;
        Tensor::stack(&[&re, &im], -1)
    }

    /// `[..., 2]` complex tensor from magnitude and phase:
    /// `mag·(cos φ + i·sin φ)`.
    #[track_caller]
    pub fn complex_from_polar(mag: &Tensor, phase: &Tensor) -> Result<Tensor> {
        origin_call!("complex_from_polar");
        let re = mag.try_mul(&phase.cos()?)?;
        let im = mag.try_mul(&phase.sin()?)?;
        Tensor::stack(&[&re, &im], -1)
    }
}
