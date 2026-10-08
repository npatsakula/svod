//! Whisper mel spectrogram: the Slaney filterbank of Whisper's
//! `assets/mel_filters.npz` over `pad_or_trim`med 30 s windows, the trailing
//! STFT frame dropped, then `log10` + clamp + normalize — in the graph.

use svod_macros::jit_wrapper;
use svod_tensor::Tensor;
use svod_tensor::nn::MelLog;

use crate::audio::{MelConfig, MelScale, MelSpectrogram};

use super::config::{HOP_LENGTH, N_FFT, N_FRAMES, N_SAMPLES, SAMPLE_RATE};

/// Whisper-specific mel spectrogram extractor.
#[derive(Clone)]
pub struct WhisperMel {
    inner: MelSpectrogram,
}

impl WhisperMel {
    pub fn new(n_mels: usize) -> Self {
        let inner = MelSpectrogram::new(&MelConfig {
            sample_rate: SAMPLE_RATE,
            n_fft: N_FFT,
            hop_length: HOP_LENGTH,
            win_length: N_FFT,
            n_mels,
            center: true,
            mel_scale: MelScale::Slaney,
            periodic: true,
            pad_mode: crate::audio::PadMode::Reflect,
            preemphasis: None,
            log: crate::audio::MelSpectrogram::LOG,
        });
        Self { inner }
    }

    pub fn n_mels(&self) -> usize {
        self.inner.n_mels()
    }

    /// `whisper.audio.pad_or_trim` into `out` (`[N_SAMPLES]`): the window
    /// zero-padded or cut to 30 seconds — the host's whole share of the
    /// front-end.
    pub fn pad_or_trim_into(waveform: &[f32], out: &mut [f32]) {
        let copy_len = waveform.len().min(N_SAMPLES);
        out[..copy_len].copy_from_slice(&waveform[..copy_len]);
        out[copy_len..].fill(0.0);
    }

    /// `whisper.audio.log_mel_spectrogram` over `[B, N_SAMPLES]` rows of
    /// [`pad_or_trim_into`](Self::pad_or_trim_into): `[B, n_mels, N_FRAMES]`.
    /// Every row is exactly 30 s, so `center` reflects in the graph; the
    /// trailing frame (`torch.stft(...)[..., :-1]`) goes before the log so
    /// the per-signal maximum sees exactly the frames Whisper keeps.
    pub fn forward_tensor(&self, samples: &Tensor) -> svod_tensor::error::Result<Tensor> {
        self.inner.forward_power_tensor(samples)?.narrow(-1, 0_usize, N_FRAMES)?.mel_log(MelLog::Whisper)
    }
}

// Front-end JIT: `[B, N_SAMPLES]` pad-or-trimmed windows -> log-mel
// `[B, n_mels, N_FRAMES]`, copied on-device into the encoder JIT's mel input.
jit_wrapper! {
    WhisperMelJit(WhisperMel) {
        samples: Tensor,

        build(samples) {
            model.forward_tensor(samples)
        }
    }
}
