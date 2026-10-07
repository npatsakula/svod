//! Audio preprocessing: log-mel spectrograms in the graph
//! ([`Tensor::mel_spectrogram`]), the host only staging samples.
//!
//! A JIT has a fixed input capacity, and `center` reflect-pads a signal with
//! its own tail — which a row zero-extended to that capacity would hide, so
//! its last frames would read zeros where the reference reads the reflection.
//! Rows of one known length (Whisper's 30 s `pad_or_trim` windows) reflect in
//! the graph; rows of differing lengths (VAD windows sharing one GigaAM JIT)
//! are reflect-padded on the host by [`MelSpectrogram::frame_into`] and
//! transformed with `center = false`, so every row's frames match a
//! transform of that row alone.

use svod_macros::jit_wrapper;
use svod_tensor::Tensor;
use svod_tensor::nn::{self, MelLog, MelNorm, Window};

type Result<T> = svod_tensor::error::Result<T>;

/// Mel filterbank scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[derive(Default)]
pub enum MelScale {
    /// HTK mel scale: `2595·log10(1+f/700)`, peak height 1 (unnormalized).
    /// Matches torchaudio's `melscale_fbanks(slk_norm=None)`.
    #[default]
    Htk,
    /// librosa Slaney scale: linear below 1 kHz, log above; area-normalized
    /// triangles. Matches `librosa.filters.mel(norm='slaney')` and
    /// Whisper's pre-computed `mel_filters.npz`.
    Slaney,
}

impl MelScale {
    fn filterbank(self) -> (nn::MelScale, Option<MelNorm>) {
        match self {
            Self::Htk => (nn::MelScale::Htk, None),
            Self::Slaney => (nn::MelScale::Slaney, Some(MelNorm::Slaney)),
        }
    }
}

/// How `center` extends a signal by `n_fft / 2` on each side.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PadMode {
    /// `torch.stft(pad_mode="reflect")`, torchaudio's default.
    #[default]
    Reflect,
    /// `torch.stft(pad_mode="constant")`, NeMo's front-end.
    Zero,
}

/// Configuration for mel spectrogram extraction.
#[derive(Clone, Debug)]
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub center: bool,
    pub mel_scale: MelScale,
    /// `torch.hann_window(periodic=..)`: torchaudio's `true`, NeMo's `false`.
    pub periodic: bool,
    pub pad_mode: PadMode,
    /// `x[n] - a·x[n - 1]` over the whole signal, its first sample kept, before
    /// the `center` padding — NeMo's order.
    pub preemphasis: Option<f32>,
    /// Compression of [`MelSpectrogram::forward_tensor`].
    pub log: MelLog,
}

/// Log-mel spectrogram extractor over one config: Hann window, power 2,
/// `f_max = sample_rate / 2`.
#[derive(Clone)]
pub struct MelSpectrogram {
    config: MelConfig,
}

impl MelSpectrogram {
    /// `torch.log(x.clamp(1e-9, 1e9))`, GigaAM's compression.
    pub const LOG: MelLog = MelLog::Ln { min: 1e-9, max: 1e9 };

    /// The config is validated when a graph is built from it.
    pub fn new(config: &MelConfig) -> Self {
        Self { config: config.clone() }
    }

    pub fn config(&self) -> &MelConfig {
        &self.config
    }

    pub fn n_mels(&self) -> usize {
        self.config.n_mels
    }

    pub fn num_frames(&self, waveform_len: usize) -> usize {
        let signal_len = self.framed_len(waveform_len);
        let (n_fft, hop) = (self.config.n_fft, self.config.hop_length);
        if signal_len >= n_fft { (signal_len - n_fft) / hop + 1 } else { 0 }
    }

    /// Samples a host-framed row holds per window: the window plus its
    /// `n_fft / 2` padding on each side under `center`.
    pub fn framed_len(&self, waveform_len: usize) -> usize {
        waveform_len + 2 * self.center_pad()
    }

    /// Samples a host-framed row needs for `frames` consecutive frames.
    pub fn frames_len(&self, frames: usize) -> usize {
        if frames == 0 { 0 } else { (frames - 1) * self.config.hop_length + self.config.n_fft }
    }

    fn center_pad(&self) -> usize {
        if self.config.center { self.config.n_fft / 2 } else { 0 }
    }

    /// Host framing for rows of differing lengths: `waveform` pre-emphasized
    /// and padded (under `center`) into the head of `out`, the rest zeroed. A
    /// row of `(num_frames - 1) · hop + n_fft` samples already covers every
    /// frame, so padding past the end of a shorter `out` is dropped.
    pub fn frame_into(&self, waveform: &[f32], out: &mut [f32]) {
        let n = self.framed_len(waveform.len()).min(out.len());
        assert!(
            self.config.pad_mode == PadMode::Zero || self.center_pad() < waveform.len().max(1),
            "reflect padding requires pad ({}) < signal length ({}); multi-bounce reflection is not supported",
            self.center_pad(),
            waveform.len(),
        );
        let emphasized = self.config.preemphasis.map(|_| {
            let mut cursor = self.cursor();
            cursor.push(waveform);
            cursor.samples
        });
        let samples = emphasized.as_deref().unwrap_or(waveform);
        let signal = Signal { samples, start: 0, len: Some(samples.len()), mode: self.config.pad_mode };
        signal.stage(-(self.center_pad() as isize), &mut out[..n]);
        out[n..].fill(0.0);
    }

    /// An incremental stager of one signal whose frames match the frames
    /// [`frame_into`](Self::frame_into) stages for the whole signal.
    pub fn cursor(&self) -> FrameCursor {
        FrameCursor {
            hop: self.config.hop_length,
            n_fft: self.config.n_fft,
            pad: self.center_pad(),
            mode: self.config.pad_mode,
            preemphasis: self.config.preemphasis,
            samples: Vec::new(),
            start: 0,
            previous: None,
            len: None,
            first_frame: 0,
        }
    }

    fn mel_power(&self, x: &Tensor, center: bool) -> Result<Tensor> {
        let (scale, norm) = self.config.mel_scale.filterbank();
        x.mel_spectrogram()
            .sample_rate(self.config.sample_rate)
            .n_fft(self.config.n_fft)
            .hop(self.config.hop_length)
            .win_length(self.config.win_length)
            .window(Window::Hann)
            .periodic(self.config.periodic)
            .center(center)
            .n_mels(self.config.n_mels)
            .mel_scale(scale)
            .maybe_norm(norm)
            .call()
    }

    /// Mel power of raw `[B, L]` (or `[L]`) samples, pre-emphasis and
    /// `center` padding in the graph: `[B, n_mels, num_frames(L)]`.
    pub fn forward_power_tensor(&self, samples: &Tensor) -> Result<Tensor> {
        let samples = match self.config.preemphasis {
            Some(a) => {
                let len = samples.dim_const(-1)?;
                let mut shift = vec![(0, 0); samples.ndim()?];
                *shift.last_mut().expect("a signal has an axis") = (1, 0);
                let previous = samples.try_pad(&shift)?.narrow(-1, 0_usize, len)?;
                samples.try_sub(&previous.try_mul(f64::from(a))?)?
            }
            None => samples.clone(),
        };
        let pad = self.center_pad();
        match self.config.pad_mode {
            PadMode::Zero if pad > 0 => {
                let mut padding = vec![(0, 0); samples.ndim()?];
                *padding.last_mut().expect("a signal has an axis") = (pad as isize, pad as isize);
                self.mel_power(&samples.try_pad(&padding)?, false)
            }
            _ => self.mel_power(&samples, self.config.center),
        }
    }

    /// Log-mel of a `[B, L']` batch of host-framed rows
    /// ([`frame_into`](Self::frame_into), [`FrameCursor::stage`]):
    /// `[B, n_mels, (L' - n_fft) / hop + 1]` with the columns past each row's
    /// `frames` (`[B]` valid frame counts) zeroed, as an encoder's mel input
    /// expects them.
    pub fn forward_tensor(&self, framed: &Tensor, frames: &Tensor) -> Result<Tensor> {
        let mel = self.mel_power(framed, false)?.mel_log(self.config.log)?;
        let valid = Tensor::sequence_mask(frames, mel.dim_const(-1)?)?.cast(mel.dtype()).try_unsqueeze(1)?;
        mel.try_mul(&valid)
    }
}

/// Incremental host framing of one signal fed in pieces (a live stream): it
/// stages any run of frames whose samples have arrived exactly as
/// [`MelSpectrogram::frame_into`] stages them from the whole signal, so a
/// stream transformed chunk by chunk yields the whole-signal frames.
///
/// Holds the pre-emphasized samples from the first frame not yet
/// [`discard`](Self::discard)ed on.
#[derive(Clone, Debug)]
pub struct FrameCursor {
    hop: usize,
    n_fft: usize,
    pad: usize,
    mode: PadMode,
    preemphasis: Option<f32>,
    /// Pre-emphasized samples from absolute index `start` on.
    samples: Vec<f32>,
    start: usize,
    /// The last raw sample pushed, for the filter across pushes.
    previous: Option<f32>,
    /// The signal length once [`finish`](Self::finish)ed.
    len: Option<usize>,
    /// The first frame [`stage`](Self::stage) may still ask for.
    first_frame: usize,
}

impl FrameCursor {
    /// Samples pushed so far.
    pub fn received(&self) -> usize {
        self.start + self.samples.len()
    }

    pub fn is_finished(&self) -> bool {
        self.len.is_some()
    }

    /// Append samples. Panics after [`finish`](Self::finish).
    pub fn push(&mut self, samples: &[f32]) {
        assert!(self.len.is_none(), "push after finish");
        self.samples.reserve(samples.len());
        for &x in samples {
            let y = match (self.preemphasis, self.previous) {
                (Some(a), Some(prev)) => x - a * prev,
                _ => x,
            };
            self.previous = Some(x);
            self.samples.push(y);
        }
    }

    /// End the signal: its tail padding becomes known, so its last frames
    /// become available.
    pub fn finish(&mut self) {
        self.len = Some(self.received());
    }

    /// Frames whose samples have all arrived; every frame of the signal once
    /// it is finished.
    pub fn available(&self) -> usize {
        let reach = match self.len {
            Some(len) => len + 2 * self.pad,
            // a reflected head mirrors samples `1..=pad`
            None if self.mode == PadMode::Reflect && self.received() <= self.pad => return 0,
            None => self.received() + self.pad,
        };
        if reach >= self.n_fft { (reach - self.n_fft) / self.hop + 1 } else { 0 }
    }

    /// Stage frames `first..` into `out` (`(frames - 1) · hop + n_fft`
    /// samples for `frames` frames); samples the signal has not reached read
    /// as padding. Panics if they precede the [`discard`](Self::discard)ed
    /// prefix.
    pub fn stage(&self, first: usize, out: &mut [f32]) {
        assert!(first >= self.first_frame, "frame {first} precedes the discarded prefix ({})", self.first_frame);
        let origin = (first * self.hop) as isize - self.pad as isize;
        Signal { samples: &self.samples, start: self.start, len: self.len, mode: self.mode }.stage(origin, out);
    }

    /// Drop the samples no frame from `frame` on reads.
    pub fn discard(&mut self, frame: usize) {
        self.first_frame = self.first_frame.max(frame);
        // A reflected tail reads back up to `pad + 1` samples before the end.
        let keep_tail = self.received().saturating_sub(self.pad + 1);
        let first_read = (frame * self.hop).saturating_sub(self.pad).min(keep_tail);
        // A reflected head reads samples `1..=pad` while frames still reach before 0.
        if first_read > self.start && frame * self.hop >= self.pad {
            self.samples.drain(..first_read - self.start);
            self.start = first_read;
        }
    }
}

/// A (pre-emphasized) signal held from absolute sample `start` on, read past
/// its edges through `mode`.
struct Signal<'a> {
    samples: &'a [f32],
    start: usize,
    /// Total length once known: past it the pad mode applies.
    len: Option<usize>,
    mode: PadMode,
}

impl Signal<'_> {
    fn at(&self, i: isize) -> f32 {
        let end = self.start + self.samples.len();
        let i = match self.mode {
            PadMode::Zero => i,
            PadMode::Reflect if i < 0 => -i,
            PadMode::Reflect => match self.len {
                Some(len) if i as usize >= len => 2 * len as isize - 2 - i,
                _ => i,
            },
        };
        if i < self.start as isize || i as usize >= end { 0.0 } else { self.samples[i as usize - self.start] }
    }

    /// Absolute samples `origin..origin + out.len()` into `out`.
    fn stage(&self, origin: isize, out: &mut [f32]) {
        let lo = (self.start as isize - origin).clamp(0, out.len() as isize) as usize;
        let hi = ((self.start + self.samples.len()) as isize - origin).clamp(lo as isize, out.len() as isize) as usize;
        if self.mode == PadMode::Zero {
            out[..lo].fill(0.0);
            let from = (origin + lo as isize) as usize - self.start;
            out[lo..hi].copy_from_slice(&self.samples[from..from + hi - lo]);
            out[hi..].fill(0.0);
        } else {
            for (k, slot) in out.iter_mut().enumerate() {
                *slot = self.at(origin + k as isize);
            }
        }
    }
}

// Front-end JIT: host-framed rows + valid frame counts -> log-mel
// `[B, n_mels, T]`, whose device output feeds an encoder JIT's mel input.
jit_wrapper! {
    MelJit(MelSpectrogram) {
        framed: Tensor,
        frames: Tensor,

        build(framed, frames) {
            model.forward_tensor(framed, frames)
        }
    }
}

/// Reflect-pad a signal by `pad` samples on each side, mirroring PyTorch's
/// `Reflect1d`: the boundary element is not duplicated, and `pad` must be
/// strictly less than the signal length (single-bounce reflection only).
#[cfg(test)]
pub(crate) fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let len = signal.len();
    assert!(
        pad < len,
        "reflect_pad requires pad ({pad}) < signal length ({len}); multi-bounce reflection is not supported",
    );

    let mut padded = Vec::with_capacity(len + 2 * pad);
    for i in (1..=pad).rev() {
        padded.push(signal[i]);
    }
    padded.extend_from_slice(signal);
    for i in 1..=pad {
        padded.push(signal[len - 1 - i]);
    }
    padded
}
