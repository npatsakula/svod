//! Silero V5 voice-activity detection.
//!
//! The forward pass mirrors the upstream Silero architecture: STFT via a
//! convolutional filterbank, four 1D conv blocks, an LSTM cell carrying
//! `(h, c)` between chunks, and a sigmoid head that produces a per-chunk
//! speech probability.
//!
//! [`VadInference::probs`] exposes the raw per-chunk probability array (one
//! entry per [`NUM_SAMPLES`] samples). [`VadInference::segment`] feeds those
//! into [`svod_arch::vad::chunks_from_probs`] to produce sample ranges
//! suitable for long-form ASR — see the `svod-arch::vad` module for
//! tunable knobs (min/max chunk duration, alignment, padding, etc.).

mod splitter;

pub use splitter::{SileroVadSplitter, SileroVadSplitterError};

use std::path::Path;

use fearless_simd::{Level, Simd, dispatch, prelude::*};
use snafu::Snafu;
use svod_dtype::DType;
use svod_macros::jit_wrapper;
use svod_tensor::Tensor;
use svod_tensor::nn::{Conv1d, LSTMCell, Layer, PadMode};

use crate::init::{Bias, conv1d, fan_in_uniform};
use crate::state;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("{source}"), context(false))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("{source}"), context(false))]
    State {
        #[snafu(source(from(crate::state::Error, Box::new)))]
        source: Box<crate::state::Error>,
    },
    #[snafu(display("hub error: {source}"), context(false))]
    Hub { source: hf_hub::HFError },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Number of input samples covered by one VAD probability entry. Exposed so
/// callers can build [`svod_arch::vad::ChunkerOpts`] with the right
/// `samples_per_prob`.
pub const NUM_SAMPLES: usize = 512;
pub(crate) const CONTEXT_SIZE: usize = 64;
const STFT_PAD: usize = 64;
const CUTOFF: usize = 128 + 1;
pub(crate) const HIDDEN: usize = 128;
const CHUNK_LEN: usize = CONTEXT_SIZE + NUM_SAMPLES;

pub struct SileroVad {
    stft_conv: Conv1d,
    conv1: Conv1d,
    conv2: Conv1d,
    conv3: Conv1d,
    conv4: Conv1d,
    lstm: LSTMCell,
    final_conv: Conv1d,
}

impl SileroVad {
    pub fn from_hub() -> Result<Self> {
        let repo = crate::hub::HubRepo::open("vpermilp/silero-vad", "main")?;
        let path = repo.get("silero_vad_16k.safetensors")?;
        Self::from_safetensors(&path)
    }

    pub fn from_safetensors(path: &Path) -> Result<Self> {
        let sd = state::load_safetensors(path)?;
        Ok(Self {
            stft_conv: Conv1d::new(get(&sd, "stft_conv.weight")?, None).with_stride(128),
            conv1: Conv1d::new(get(&sd, "conv1.weight")?, Some(get(&sd, "conv1.bias")?)).with_padding((1, 1)),
            conv2: Conv1d::new(get(&sd, "conv2.weight")?, Some(get(&sd, "conv2.bias")?))
                .with_stride(2)
                .with_padding((1, 1)),
            conv3: Conv1d::new(get(&sd, "conv3.weight")?, Some(get(&sd, "conv3.bias")?))
                .with_stride(2)
                .with_padding((1, 1)),
            conv4: Conv1d::new(get(&sd, "conv4.weight")?, Some(get(&sd, "conv4.bias")?)).with_padding((1, 1)),
            lstm: LSTMCell::new(
                get(&sd, "lstm_cell.weight_ih")?,
                get(&sd, "lstm_cell.weight_hh")?,
                get(&sd, "lstm_cell.bias_ih")?,
                get(&sd, "lstm_cell.bias_hh")?,
            ),
            final_conv: Conv1d::new(get(&sd, "final_conv.weight")?, Some(get(&sd, "final_conv.bias")?)),
        })
    }

    /// Build with random weights matching the Silero V5 16 kHz layout. Strides
    /// and paddings mirror [`Self::from_safetensors`]; the lazy
    /// `fan_in_uniform` graphs keep the forward path from collapsing under
    /// const-folding so the JIT pipeline can be exercised without a checkpoint.
    pub fn with_random_weights() -> Self {
        let dt = DType::Float32;
        let mk_conv = |[out, inp, k]: [usize; 3], bias: Bias, configure: fn(Conv1d) -> Conv1d| -> Conv1d {
            configure(conv1d(inp, out, k, bias, dt.clone()))
        };

        Self {
            stft_conv: mk_conv([258, 1, 256], Bias::None, |c| c.with_stride(128)),
            conv1: mk_conv([128, 129, 3], Bias::FanIn, |c| c.with_padding((1, 1))),
            conv2: mk_conv([64, 128, 3], Bias::FanIn, |c| c.with_stride(2).with_padding((1, 1))),
            conv3: mk_conv([64, 64, 3], Bias::FanIn, |c| c.with_stride(2).with_padding((1, 1))),
            conv4: mk_conv([128, 64, 3], Bias::FanIn, |c| c.with_padding((1, 1))),
            lstm: LSTMCell::new(
                fan_in_uniform(&[4 * HIDDEN, HIDDEN], HIDDEN, dt.clone()),
                fan_in_uniform(&[4 * HIDDEN, HIDDEN], HIDDEN, dt.clone()),
                fan_in_uniform(&[4 * HIDDEN], HIDDEN, dt.clone()),
                fan_in_uniform(&[4 * HIDDEN], HIDDEN, dt.clone()),
            ),
            final_conv: mk_conv([1, 128, 1], Bias::FanIn, |c| c),
        }
    }

    /// Per-window convolutional front-end (STFT filterbank + four conv blocks),
    /// **batched** over the leading axis. Input `chunks: [B, CHUNK_LEN]`, output
    /// `[B, HIDDEN]` — the LSTM input feature for each window. This part of the
    /// forward pass is **not** recurrent, so all `B` windows run in one batched
    /// dispatch; the recurrent LSTM + head runs separately (on the host) over
    /// these features. See [`VadInference::probs`].
    pub fn forward_features(&self, chunks: &Tensor) -> Result<Tensor> {
        let x = chunks
            .pad_with()
            .padding(&[(0, 0), (0, STFT_PAD as isize)])
            .mode(PadMode::Reflect)
            .call()?
            .try_unsqueeze(1)?;

        // `stft_conv` IS `Tensor::stft`'s `[2F, 1, n_fft]` analysis kernel with
        // Silero's window baked in, and the reflect pad above is the (right-only)
        // framing `center = false` wants — so the transform is that conv, and only
        // the `[B, 2F, T] -> [B, F, T, 2]` regroup and the modulus come from the
        // STFT helpers. Rebuilding the basis in-graph from `stft()` would discard
        // the checkpoint's own kernel.
        let x = self.stft_conv.forward(&x)?;
        let x = x.unflatten(1, &[2, CUTOFF as isize])?.try_permute(&[0, 2, 3, 1])?.complex_abs()?;

        let x = self.conv1.forward(&x)?.relu()?;
        let x = self.conv2.forward(&x)?.relu()?;
        let x = self.conv3.forward(&x)?.relu()?;
        Ok(self.conv4.forward(&x)?.relu()?.try_squeeze(Some(-1))?)
    }

    /// Conv front-end + the LSTM's non-recurrent input projection, batched:
    /// `[B, CHUNK_LEN] -> [B, 4*HIDDEN]` pre-activation gates
    /// (`W_ih·feat + bias_ih + bias_hh`, PyTorch `[i,f,g,o]` order). Hoisting
    /// the projection into the JIT leaves the host scan only the recurrent
    /// `W_hh·h` and activations. Bias order: `(feat·W_ihᵀ + b_ih) + b_hh`.
    pub fn forward_gates(&self, chunks: &Tensor) -> Result<Tensor> {
        let feat = self.forward_features(chunks)?;
        Ok(feat.linear().weight(&self.lstm.weight_ih).bias(&self.lstm.bias_ih).call()?.try_add(&self.lstm.bias_hh)?)
    }

    pub fn forward_chunk(&self, chunk: &Tensor, state_h: &Tensor, state_c: &Tensor) -> Result<Tensor> {
        let x = self.forward_features(chunk)?;

        let (new_h, new_c) = self.lstm.step(&x, state_h, state_c)?;

        let prob = new_h.try_unsqueeze(-1)?.relu()?;
        let prob = self
            .final_conv
            .forward(&prob)?
            .sigmoid()?
            .try_squeeze(Some(-1))?
            .mean_with()
            .axes(-1isize)
            .keepdim(true)
            .call()?;

        Ok(Tensor::cat(&[&prob, &new_h, &new_c], 1)?)
    }
}

fn get(sd: &state::StateDict, key: &str) -> Result<Tensor> {
    Ok(state::get_tensor(sd, key)?)
}

/// Max windows per batched conv-front-end dispatch. Larger = fewer dispatches
/// (less per-dispatch round-trip latency) but more VRAM + a longer compile.
const FEATURE_BATCH: usize = 4096;

jit_wrapper! {
    SileroVadFeatureJit(SileroVad) {
        chunks: Tensor,

        build(chunks) {
            // [FEATURE_BATCH, CHUNK_LEN] -> [FEATURE_BATCH, 4*HIDDEN] LSTM gate
            // pre-activations (conv features + input projection, biases folded).
            // Fixed batch (not a runtime var): the front-end is row-independent,
            // so partial batches just fill fewer rows and ignore the rest — and
            // a symbolic leading dim trips the reflect-pad lowering.
            model.forward_gates(chunks)
        }
    }
}

/// Host-resident recurrent weights for the scan. The input projection
/// (`W_ih`, biases) lives in the feature JIT ([`SileroVad::forward_gates`]);
/// only the recurrence stays host-side — per-step work is too small for a
/// GPU launch and measured 3x slower as a CPU-device JIT than 8-lane SIMD.
pub(crate) struct VadHead {
    pub(crate) w_hh: ndarray::Array2<f32>,    // [4H, H]
    pub(crate) final_w: ndarray::Array1<f32>, // [H]
    pub(crate) final_b: f32,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl VadHead {
    /// Recurrent LSTM + sigmoid head over `[n, 4H]` pre-activation gates from
    /// the feature JIT (`W_ih·feat + biases`, PyTorch `[i,f,g,o]` order),
    /// SIMD activations at the widest ISA the CPU has.
    pub(crate) fn scan(&self, gates_x: &[f32], n: usize) -> Vec<f32> {
        debug_assert_eq!(gates_x.len(), n * 4 * self.final_w.len(), "gates shape");
        dispatch!(Level::new(), simd => self.scan_inner(simd, gates_x, n))
    }

    #[inline(always)]
    fn scan_inner<S: Simd>(&self, simd: S, gates_x: &[f32], n: usize) -> Vec<f32> {
        let lanes_n = S::f32s::LEN;
        let h = self.final_w.len();
        debug_assert_eq!(h % lanes_n, 0, "HIDDEN must be a multiple of the SIMD width");

        let lanes = |v: &[f32], j: usize| S::f32s::from_slice(simd, &v[j..j + lanes_n]);
        let w = self.final_w.as_slice().expect("final_w");
        let zero = S::f32s::splat(simd, 0.0);

        let mut hs = ndarray::Array1::<f32>::zeros(h);
        let mut cs = vec![0.0f32; h];
        let mut gh = ndarray::Array1::<f32>::zeros(4 * h); // recurrent projection scratch, reused per step
        let mut probs = Vec::with_capacity(n);
        for t in 0..n {
            let gx = &gates_x[t * 4 * h..(t + 1) * 4 * h];
            // gh = W_hh · h, written in place to avoid a per-step heap allocation.
            ndarray::linalg::general_mat_vec_mul(1.0, &self.w_hh, &hs, 0.0, &mut gh);
            let ghs = gh.as_slice().expect("contiguous gh");
            let hss = hs.as_slice_mut().expect("contiguous hs");
            let mut p = zero;
            for j in (0..h).step_by(lanes_n) {
                let gate = |k: usize| lanes(gx, k) + lanes(ghs, k);
                let i = crate::simd::sigmoid(simd, gate(j));
                let f = crate::simd::sigmoid(simd, gate(h + j));
                let g = crate::simd::tanh(simd, gate(2 * h + j));
                let o = crate::simd::sigmoid(simd, gate(3 * h + j));
                let c = f * lanes(&cs, j) + i * g;
                let hv = o * crate::simd::tanh(simd, c);
                c.store_slice(&mut cs[j..j + lanes_n]);
                hv.store_slice(&mut hss[j..j + lanes_n]);
                p += lanes(w, j) * hv.max(zero);
            }
            probs.push(sigmoid(self.final_b + p.as_slice().iter().sum::<f32>()));
        }
        probs
    }
}

pub struct VadInference {
    jit: SileroVadFeatureJit,
    head: VadHead,
}

impl svod_arch::pipelines::audio::Vad for VadInference {
    type Error = crate::jit::JitError;

    fn samples_per_prob(&self) -> usize {
        NUM_SAMPLES
    }

    fn probs(&mut self, waveform: &[f32]) -> std::result::Result<Vec<f32>, Self::Error> {
        // Inherent `probs` — the conv front-end + head over the whole waveform.
        VadInference::probs(self, waveform)
    }
}

impl VadInference {
    pub fn new(vad: SileroVad) -> crate::jit::Result<Self> {
        use crate::jit::InputSpec;

        let h = HIDDEN;
        // Pull the recurrent weights to host before `vad` moves into the JIT.
        let to_vec = |t: &Tensor| -> crate::jit::Result<Vec<f32>> { Ok(t.to_vec::<f32>()?) };
        let w_hh = ndarray::Array2::from_shape_vec((4 * h, h), to_vec(&vad.lstm.weight_hh)?).expect("w_hh shape");
        let final_w = ndarray::Array1::from_vec(to_vec(&vad.final_conv.weight)?); // [1,H,1] flat = H
        let final_b = match &vad.final_conv.bias {
            Some(b) => to_vec(b)?[0],
            None => 0.0,
        };
        let head = VadHead { w_hh, final_w, final_b };

        let mut jit = SileroVadFeatureJit::new(vad);
        // Device-local output: the [FEATURE_BATCH, 4*HIDDEN] gates readback
        // (8 MiB per dispatch) goes over the SDMA copy queue instead of the
        // ~21 MB/s host-mapped BAR — same pattern as the encoder output.
        jit.prepare_with_config(
            InputSpec::f32(&[FEATURE_BATCH, CHUNK_LEN]),
            &svod_tensor::PrepareConfig::device_local(),
        )?;
        Ok(Self { jit, head })
    }

    /// Run Silero V5 across the waveform and collect one speech probability per
    /// [`NUM_SAMPLES`]-sample window. Output length is
    /// `ceil(waveform.len() / NUM_SAMPLES)`. The conv front-end runs **batched**
    /// on the GPU (a handful of dispatches); the recurrent LSTM + sigmoid head
    /// scan runs on the host — eliminating the old one-tiny-dispatch-per-window
    /// path whose per-dispatch round-trip latency dominated.
    pub fn probs(&mut self, waveform: &[f32]) -> crate::jit::Result<Vec<f32>> {
        let total = waveform.len();
        if total == 0 {
            return Ok(Vec::new());
        }
        let pad_len = (NUM_SAMPLES - total % NUM_SAMPLES) % NUM_SAMPLES;
        let padded_len = CONTEXT_SIZE + total + pad_len;
        let mut padded = vec![0.0f32; padded_len];
        padded[CONTEXT_SIZE..CONTEXT_SIZE + total].copy_from_slice(waveform);

        let n_chunks = (total + pad_len) / NUM_SAMPLES;
        let h = HIDDEN;

        // Phase 1: batched conv front-end + LSTM input projection on the GPU
        // -> pre-activation gates [n_chunks, 4H].
        let t_feat = std::time::Instant::now();
        let mut gates = vec![0.0f32; n_chunks * 4 * h];
        let mut done = 0usize;
        while done < n_chunks {
            let b = (n_chunks - done).min(FEATURE_BATCH);
            {
                let mut view = self.jit.chunks_view_mut::<f32>()?;
                let slice = view.as_slice_mut().expect("contiguous chunks");
                for i in 0..b {
                    let start = (done + i) * NUM_SAMPLES;
                    slice[i * CHUNK_LEN..(i + 1) * CHUNK_LEN].copy_from_slice(&padded[start..start + CHUNK_LEN]);
                }
            }
            self.jit.execute()?;
            // Device-local output: SDMA-stage only the valid rows out.
            let dst = &mut gates[done * 4 * h..(done + b) * 4 * h];
            self.jit.output()?.copyout_prefix(bytemuck::cast_slice_mut(dst))?;
            done += b;
        }
        let feature_ms = t_feat.elapsed().as_secs_f64() * 1e3;

        // Phase 2: recurrent LSTM + sigmoid head on the host (SIMD).
        let t_scan = std::time::Instant::now();
        let probs = self.head.scan(&gates, n_chunks);
        let scan_ms = t_scan.elapsed().as_secs_f64() * 1e3;

        tracing::info!(
            target: "svod_model::silero_vad",
            n_chunks,
            feature_ms,
            scan_ms,
            "silero vad probs breakdown (batched conv + host LSTM scan)",
        );
        Ok(probs)
    }

    /// Convenience wrapper around [`Self::probs`] +
    /// [`svod_arch::vad::chunks_from_probs`] with default chunker knobs and
    /// the given `threshold`. Errors from the JIT or chunker are swallowed —
    /// callers that need fault-visibility should drive `probs()` and
    /// `chunks_from_probs` directly.
    ///
    /// Chunk ends are clamped to `waveform.len()` via `ChunkerOpts::
    /// max_total_samples`: the prob→sample mapping rounds the final window up
    /// past the audio (the trailing zero-pad), and a speech region can't extend
    /// beyond the waveform.
    pub fn segment(&mut self, waveform: &[f32], threshold: f32) -> Vec<(usize, usize)> {
        let Ok(probs) = self.probs(waveform) else { return Vec::new() };
        let opts = svod_arch::vad::ChunkerOpts {
            threshold,
            samples_per_prob: NUM_SAMPLES,
            max_total_samples: Some(waveform.len()),
            ..svod_arch::vad::ChunkerOpts::default()
        };
        svod_arch::vad::chunks_from_probs(&probs, &opts)
            .unwrap_or_default()
            .into_iter()
            .map(|c| (c.start_sample, c.end_sample))
            .collect()
    }
}
