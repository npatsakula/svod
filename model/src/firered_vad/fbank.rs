//! Kaldi-compatible log-mel fbank for FireRedVAD, matching
//! `kaldi-native-fbank` with the upstream options (16 kHz, 25 ms / 10 ms,
//! `snip_edges`, `dither = 0`, 80 bins): per-frame DC removal, per-frame
//! pre-emphasis 0.97, Povey window, power spectrum over a 512-point FFT,
//! triangular mel bins on `1127·ln(1 + f/700)` between 20 Hz and Nyquist,
//! natural log floored at `f32::EPSILON` — [`Tensor::mel_spectrogram`] with
//! its Kaldi options (`preemphasis`, `remove_dc`, the symmetric `Povey`
//! window, Kaldi's own `filterbank`).
//!
//! Kaldi left-aligns its 400-sample window in the 512-point frame, where the
//! transform centers a short window [`CENTERING`] samples in; prepending that
//! many zeros to the signal makes the centered placement read Kaldi's frames.
//! `snip_edges` is `center = false` with Kaldi's frame count, which is the
//! convolution's: `1 + (L - 400) / 160`.

use snafu::ensure;
use svod_macros::jit_wrapper;
use svod_tensor::error::ParamRangeSnafu;
use svod_tensor::nn::{MelLog, Window};
use svod_tensor::{PrepareConfig, Tensor};

use super::{FRAME_LENGTH, FRAME_SHIFT, N_MELS};
use crate::jit::InputSpec;

const N_FFT: usize = 512; // next power of two >= FRAME_LENGTH
const N_BINS: usize = N_FFT / 2 + 1;
/// Where the transform places a `FRAME_LENGTH` window inside an `N_FFT` frame.
const CENTERING: usize = (N_FFT - FRAME_LENGTH) / 2;
const LOW_FREQ: f64 = 20.0;
const SAMPLE_RATE: f64 = 16_000.0;
/// Kaldi reads 16-bit PCM without normalization; Svod waveforms are `[-1, 1]`.
const INT16_SCALE: f64 = 32_768.0;
const PREEMPH: f64 = 0.97;

/// The fbank graph: `[B, L]` samples in `[-1, 1]` to `[B, T, N_MELS]`
/// pre-CMVN log-mel rows, `T = 1 + (L - FRAME_LENGTH) / FRAME_SHIFT`.
#[derive(Clone)]
pub struct KaldiFbank {
    /// Kaldi's `[N_MELS, N_BINS]` mel banks.
    banks: Vec<f32>,
}

impl Default for KaldiFbank {
    fn default() -> Self {
        Self::new()
    }
}

impl KaldiFbank {
    pub fn new() -> Self {
        Self { banks: kaldi_mel_banks() }
    }

    /// `snip_edges` frame count: only complete 25 ms windows produce a frame.
    pub fn num_frames(n_samples: usize) -> usize {
        if n_samples < FRAME_LENGTH { 0 } else { 1 + (n_samples - FRAME_LENGTH) / FRAME_SHIFT }
    }

    /// Samples `frames` frames read: the window is zero past `FRAME_LENGTH`,
    /// but the transform frames `N_FFT` samples.
    pub fn samples(frames: usize) -> usize {
        frames.checked_sub(1).map_or(0, |last| last * FRAME_SHIFT + N_FFT)
    }

    /// `[B, L]` -> `[B, T, N_MELS]`, `L` covering [`samples`](Self::samples)`(T)`.
    pub fn forward_tensor(&self, samples: &Tensor) -> svod_tensor::error::Result<Tensor> {
        let banks = Tensor::from_slice(self.banks.clone()).try_reshape([N_MELS as isize, N_BINS as isize])?;
        samples
            .try_mul(INT16_SCALE)?
            .try_pad(&[(0, 0), (CENTERING as isize, 0)])?
            .mel_spectrogram()
            .n_fft(N_FFT)
            .hop(FRAME_SHIFT)
            .win_length(FRAME_LENGTH)
            .window(Window::Povey)
            .periodic(false)
            .center(false)
            .preemphasis(PREEMPH)
            .remove_dc(true)
            .filterbank(&banks)
            .log(MelLog::Ln { min: f32::EPSILON as f64, max: f64::INFINITY })
            .call()?
            .try_transpose(-1, -2)
    }
}

/// Kaldi `MelBanks`: triangles on the `1127·ln(1 + f/700)` mel axis over the
/// first `N_FFT/2` FFT bins (Nyquist bin excluded, as in Kaldi), 20 Hz to
/// Nyquist, as a `[N_MELS, N_BINS]` table.
fn kaldi_mel_banks() -> Vec<f32> {
    let mel = |f: f64| 1127.0 * (1.0 + f / 700.0).ln();
    let fft_bin_width = SAMPLE_RATE / N_FFT as f64;
    let (mel_low, mel_high) = (mel(LOW_FREQ), mel(SAMPLE_RATE / 2.0));
    let delta = (mel_high - mel_low) / (N_MELS + 1) as f64;

    let mut table = vec![0f32; N_MELS * N_BINS];
    for (m, row) in table.chunks_mut(N_BINS).enumerate() {
        let (left, center, right) =
            (mel_low + m as f64 * delta, mel_low + (m + 1) as f64 * delta, mel_low + (m + 2) as f64 * delta);
        for (i, w) in row.iter_mut().enumerate().take(N_FFT / 2) {
            let mel_f = mel(i as f64 * fft_bin_width);
            if mel_f > left && mel_f < right {
                *w = (if mel_f <= center { (mel_f - left) / delta } else { (right - mel_f) / delta }) as f32;
            }
        }
    }
    table
}

/// `constraint` on a caller-supplied size the prepared JIT shape cannot bend
/// to, as a tensor parameter error (`?` lifts it into the JIT's).
pub(super) fn check(
    ok: bool,
    op: &'static str,
    param: &'static str,
    value: impl ToString,
    constraint: &'static str,
) -> svod_tensor::error::Result<()> {
    ensure!(ok, ParamRangeSnafu { op, param, value: value.to_string(), constraint });
    Ok(())
}

jit_wrapper! {
    KaldiFbankJit(KaldiFbank) {
        samples: Tensor,

        build(samples) {
            model.forward_tensor(samples)
        }
    }
}

/// Host driver over a fixed-shape `KaldiFbankJit`: `batch` windows of
/// `capacity` frames per execute, frames being independent under
/// `snip_edges`. The output buffer is shaped for a device copy into a model
/// JIT's `[batch, capacity, N_MELS]` input, so features never round-trip
/// through the host on that path.
pub struct FireRedFbank {
    jit: KaldiFbankJit,
    /// `[batch, row]` host staging for the device-local input, `row` the
    /// samples of `capacity` frames: one `copyin` of the filled rows per
    /// execute.
    staging: Vec<f32>,
    row: usize,
    capacity: usize,
}

impl FireRedFbank {
    /// Prepare the JIT for `batch` windows of `capacity` frames per execute.
    pub fn new(batch: usize, capacity: usize) -> crate::jit::Result<Self> {
        check(
            batch >= 1 && capacity >= 1,
            "FireRedFbank::new",
            "(batch, capacity)",
            format!("({batch}, {capacity})"),
            "both >= 1",
        )?;
        let mut jit = KaldiFbankJit::new(KaldiFbank::new());
        let row = KaldiFbank::samples(capacity);
        jit.prepare_with_config(InputSpec::f32(&[batch, row]).device_local(), &PrepareConfig::device_local())?;
        Ok(Self { jit, staging: vec![0.0; batch * row], row, capacity })
    }

    pub fn batch(&self) -> usize {
        self.staging.len() / self.row
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn num_frames(&self, n_samples: usize) -> usize {
        KaldiFbank::num_frames(n_samples)
    }

    /// One execute: row `i` holds the `capacity` frames of `waveform` from
    /// frame `first_frames[i]` on (frames outside the waveform read zeros).
    /// Returns the `[batch, capacity, N_MELS]` output buffer; rows past
    /// `first_frames.len()` are stale.
    pub fn forward_windows(
        &mut self,
        waveform: &[f32],
        first_frames: &[isize],
    ) -> crate::jit::Result<&svod_device::Buffer> {
        let (windows, batch) = (first_frames.len(), self.batch());
        check(
            windows <= batch,
            "FireRedFbank::forward_windows",
            "windows",
            format!("{windows} (batch {batch})"),
            "<= batch",
        )?;
        let len = waveform.len() as isize;
        for (row, &first) in self.staging.chunks_mut(self.row).zip(first_frames) {
            let start = first * FRAME_SHIFT as isize;
            let (lo, hi) = (start.clamp(0, len), (start + self.row as isize).clamp(0, len));
            let head = (lo - start).clamp(0, self.row as isize) as usize;
            let tail = head + (hi - lo) as usize;
            row[..head].fill(0.0);
            row[head..tail].copy_from_slice(&waveform[lo as usize..hi as usize]);
            row[tail..].fill(0.0);
        }
        let filled = &self.staging[..windows * self.row];
        self.jit.samples_mut()?.copyin_at(0, bytemuck::cast_slice(filled))?;
        self.jit.execute()?;
        self.jit.output()
    }

    /// Extract `[num_frames * N_MELS]` row-major log-mel features from a
    /// `[-1, 1]`-scale waveform (pre-CMVN — normalization happens inside the
    /// model graph) — the host-side form, `batch · capacity` frames per
    /// execute.
    pub fn forward(&mut self, waveform: &[f32]) -> crate::jit::Result<Vec<f32>> {
        let mut feat = vec![0.0f32; KaldiFbank::num_frames(waveform.len()) * N_MELS];
        let block = self.batch() * self.capacity;
        for (k, rows) in feat.chunks_mut(block * N_MELS).enumerate() {
            let windows = (rows.len() / N_MELS).div_ceil(self.capacity);
            let firsts: Vec<isize> = (0..windows).map(|i| ((k * self.batch() + i) * self.capacity) as isize).collect();
            self.forward_windows(waveform, &firsts)?.copyout_prefix(bytemuck::cast_slice_mut(rows))?;
        }
        Ok(feat)
    }
}
