//! FireRedVAD voice-activity detection (DFSMN, arXiv:2603.10420).
//!
//! The DFSMN is purely feed-forward — the "memory" layers are depthwise
//! convolutions over time, not recurrent state — so the **entire** model runs
//! as one batched device JIT behind a graph fbank; the host only stages
//! samples and stitches window probabilities. The FSMN lookback/lookahead
//! use asymmetric conv padding instead of the reference's
//! symmetric-pad-then-slice (verified bit-identical by
//! `scripts/convert_firered_vad.py --selfcheck`).
//!
//! Two variants ship: the non-streaming model below (lookback + lookahead,
//! whole-utterance windowed inference behind [`FireRedVadSplitter`]) and the
//! causal streaming `Stream-VAD` checkpoint (per-chunk JIT with on-device
//! conv caches, incremental [`FireRedVadStreamer`] API).
//!
//! [`FireRedVadInference::probs`] yields one speech probability per
//! [`FRAME_SHIFT`]-sample fbank frame. Long inputs run as overlapping
//! `CHUNK_T`-frame windows with a `HALO`-frame margin on each side; only
//! each window's core is kept, so every kept frame sees its full receptive
//! field (±`HALO`) of real context and the stitched output equals a
//! single full-length forward up to float reassociation.

mod fbank;
mod splitter;
mod stream;

pub use fbank::{FireRedFbank, KaldiFbank};
pub use splitter::{FireRedVadSplitter, FireRedVadSplitterError};
pub use stream::{
    FireRedVadStream, FireRedVadStreamError, FireRedVadStreamer, StreamFlush, StreamVadConfig, StreamVadPostprocessor,
    VadEvent,
};

use std::path::Path;

use snafu::Snafu;
use svod_dtype::DType;
use svod_macros::jit_wrapper;
use svod_tensor::Tensor;

use crate::init::fan_in_uniform;
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

/// Fbank hop: samples covered by one probability entry (10 ms at 16 kHz).
pub const FRAME_SHIFT: usize = 160;
/// Fbank window length in samples (25 ms at 16 kHz).
pub const FRAME_LENGTH: usize = 400;
pub const N_MELS: usize = 80;

const HIDDEN: usize = 256; // H
const PROJ: usize = 128; // P
/// FSMN filter order (`N1 == N2`); per-layer temporal reach in frames.
const ORDER: usize = 20;
/// DFSMN blocks after `fsmn1` (`R - 1`).
const BLOCKS: usize = 7;
/// Total FSMN layers (`R`).
const LAYERS: usize = BLOCKS + 1;

/// Frames per JIT window (20 s of audio).
pub(crate) const CHUNK_T: usize = 2000;
/// Stacked receptive-field radius: each of the [`LAYERS`] FSMN layers reaches
/// ±[`ORDER`] frames, so a kept frame inset by `HALO` from the window edge
/// sees only real context.
pub(crate) const HALO: usize = LAYERS * ORDER;
/// Kept (non-overlapping) frames per window.
pub(crate) const CORE: usize = CHUNK_T - 2 * HALO;
/// Windows per JIT dispatch. Fixed batch (the JIT graph has a static shape):
/// rows are independent, so partial batches fill fewer rows and the stitcher
/// ignores the rest.
pub(crate) const BATCH: usize = 8;

/// The checkpoint stores each FSMN time filter as a `[P, 1, 1, ORDER]`
/// conv2d kernel over a fake height axis; `conv1d` takes `[P, 1, ORDER]`.
pub(crate) fn time_filter(weight: &Tensor) -> Result<Tensor> {
    Ok(weight.try_reshape([PROJ as isize, 1, ORDER as isize])?)
}

/// One FSMN memory layer: depthwise lookback + lookahead filters over time
/// (weights `[P, 1, ORDER]`), residual added by the caller-side formula
/// `memory = x + lookback + lookahead`.
#[derive(Clone)]
struct Fsmn {
    lookback: Tensor,
    lookahead: Tensor,
}

impl Fsmn {
    /// `[N, T, P] -> [N, T, P]`. Lookback pads `(ORDER-1, 0)` (causal);
    /// lookahead pads `(0, ORDER)` and drops the first output column — index
    /// `t` then sums frames `[t+1, t+ORDER]`, with `t = T-1` falling entirely
    /// in the zero pad (the reference's appended zero).
    ///
    /// `valid` (`[N, 1, T]`, `{0, 1}`) zeroes pad rows before the convs —
    /// the reference's `masked_fill` batch-padding path. Pointwise layers turn
    /// zero-filled pad rows into bias activations, so without the mask the
    /// convs would leak them into real frames; masked, a pad row contributes
    /// exactly what conv zero-padding would.
    fn forward(&self, x: &Tensor, valid: Option<&Tensor>) -> Result<Tensor> {
        let len = x.dim_const(1)?;
        let k = ORDER as isize;
        // [N,T,P] -> [N,P,T]: conv1d wants (N, C, L).
        let mut xt = x.try_transpose(-1, -2)?;
        if let Some(valid) = valid {
            xt = xt.try_mul(valid)?;
        }
        let filter = |weight: &Tensor, padding: (isize, isize)| -> Result<Tensor> {
            Ok(xt.conv1d().weight(weight).groups(PROJ).padding(padding).call()?)
        };
        let lookback = filter(&self.lookback, (k - 1, 0))?;
        let lookahead = filter(&self.lookahead, (0, k))?.narrow(-1, 1usize, len)?;
        let memory = xt.try_add(&lookback)?.try_add(&lookahead)?;
        Ok(memory.try_transpose(-1, -2)?)
    }
}

/// `fc1 (P->H) + ReLU -> fc2 (H->P, bias-free) -> FSMN -> + skip`.
#[derive(Clone)]
struct DfsmnBlock {
    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fsmn: Fsmn,
}

impl DfsmnBlock {
    fn forward(&self, x: &Tensor, valid: Option<&Tensor>) -> Result<Tensor> {
        let h = x.linear().weight(&self.fc1_weight).bias(&self.fc1_bias).call()?.relu()?;
        let p = h.linear().weight(&self.fc2_weight).call()?;
        Ok(self.fsmn.forward(&p, valid)?.try_add(x)?)
    }
}

#[derive(Clone)]
pub struct FireRedVad {
    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fc2_bias: Tensor,
    fsmn1: Fsmn,
    blocks: Vec<DfsmnBlock>,
    dnn_weight: Tensor,
    dnn_bias: Tensor,
    out_weight: Tensor,
    out_bias: Tensor,
    cmvn_means: Tensor,
    cmvn_istd: Tensor,
}

/// HF Hub repo holding the converted weights (+ the parity-test golden).
pub(crate) const HUB_REPO: &str = "vpermilp/firered_vad";

/// Download a file from [`HUB_REPO`] into the local HF cache.
pub(crate) fn hub_file(name: &str) -> Result<std::path::PathBuf> {
    let repo = crate::hub::HubRepo::open(HUB_REPO, "main")?;
    Ok(repo.get(name)?)
}

impl FireRedVad {
    pub fn from_hub() -> Result<Self> {
        Self::from_safetensors(&hub_file("firered_vad.safetensors")?)
    }

    /// Load from a safetensors produced by `scripts/convert_firered_vad.py`
    /// (model weights + `cmvn_means`/`cmvn_istd` stats).
    pub fn from_safetensors(path: &Path) -> Result<Self> {
        let sd = state::load_safetensors(path)?;
        let get = |key: &str| -> Result<Tensor> { Ok(state::get_tensor(&sd, key)?) };
        let fsmn = |prefix: &str| -> Result<Fsmn> {
            Ok(Fsmn {
                lookback: time_filter(&get(&format!("{prefix}.lookback.weight"))?)?,
                lookahead: time_filter(&get(&format!("{prefix}.lookahead.weight"))?)?,
            })
        };
        let blocks = (0..BLOCKS)
            .map(|i| {
                Ok(DfsmnBlock {
                    fc1_weight: get(&format!("blocks.{i}.fc1.weight"))?,
                    fc1_bias: get(&format!("blocks.{i}.fc1.bias"))?,
                    fc2_weight: get(&format!("blocks.{i}.fc2.weight"))?,
                    fsmn: fsmn(&format!("blocks.{i}"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            fc1_weight: get("fc1.weight")?,
            fc1_bias: get("fc1.bias")?,
            fc2_weight: get("fc2.weight")?,
            fc2_bias: get("fc2.bias")?,
            fsmn1: fsmn("fsmn1")?,
            blocks,
            dnn_weight: get("dnn.weight")?,
            dnn_bias: get("dnn.bias")?,
            out_weight: get("out.weight")?,
            out_bias: get("out.bias")?,
            cmvn_means: get("cmvn_means")?,
            cmvn_istd: get("cmvn_istd")?,
        })
    }

    /// Random weights with the released checkpoint's layout. Lazy
    /// `fan_in_uniform` graphs keep the forward path from collapsing under
    /// const-folding so the JIT pipeline can be exercised without weights.
    pub fn with_random_weights() -> Self {
        let dt = DType::Float32;
        let lin = |out: usize, inp: usize| fan_in_uniform(&[out, inp], inp, dt.clone());
        let bias = |out: usize, fan_in: usize| fan_in_uniform(&[out], fan_in, dt.clone());
        let fsmn = || Fsmn {
            lookback: fan_in_uniform(&[PROJ, 1, ORDER], ORDER, dt.clone()),
            lookahead: fan_in_uniform(&[PROJ, 1, ORDER], ORDER, dt.clone()),
        };
        Self {
            fc1_weight: lin(HIDDEN, N_MELS),
            fc1_bias: bias(HIDDEN, N_MELS),
            fc2_weight: lin(PROJ, HIDDEN),
            fc2_bias: bias(PROJ, HIDDEN),
            fsmn1: fsmn(),
            blocks: (0..BLOCKS)
                .map(|_| DfsmnBlock {
                    fc1_weight: lin(HIDDEN, PROJ),
                    fc1_bias: bias(HIDDEN, PROJ),
                    fc2_weight: lin(PROJ, HIDDEN),
                    fsmn: fsmn(),
                })
                .collect(),
            dnn_weight: lin(HIDDEN, PROJ),
            dnn_bias: bias(HIDDEN, PROJ),
            out_weight: lin(1, HIDDEN),
            out_bias: bias(1, HIDDEN),
            cmvn_means: fan_in_uniform(&[N_MELS], N_MELS, dt.clone()),
            cmvn_istd: fan_in_uniform(&[N_MELS], N_MELS, dt),
        }
    }

    /// Full DFSMN forward: pre-CMVN fbank `[B, T, N_MELS]` -> speech
    /// probabilities `[B, T]`. CMVN runs in-graph as the first op.
    ///
    /// `valid` (`[B, T, 1]`, `{0, 1}`) marks rows holding real frames; pad
    /// rows are masked out of every FSMN conv, so their probabilities are
    /// garbage but the valid rows match a pad-free forward exactly. `None`
    /// means all rows are real.
    pub fn forward(&self, feat: &Tensor, valid: Option<&Tensor>) -> Result<Tensor> {
        // [B,T,1] -> [B,1,T], broadcast over P in the convs' (N,C,L) layout.
        let valid = valid.map(|v| v.try_transpose(-1, -2)).transpose()?;
        let x = feat.try_sub(&self.cmvn_means)?.try_mul(&self.cmvn_istd)?;
        let h = x.linear().weight(&self.fc1_weight).bias(&self.fc1_bias).call()?.relu()?;
        let mut m = h.linear().weight(&self.fc2_weight).bias(&self.fc2_bias).call()?.relu()?;
        m = self.fsmn1.forward(&m, valid.as_ref())?;
        for block in &self.blocks {
            m = block.forward(&m, valid.as_ref())?;
        }
        let d = m.linear().weight(&self.dnn_weight).bias(&self.dnn_bias).call()?.relu()?;
        let logits = d.linear().weight(&self.out_weight).bias(&self.out_bias).call()?;
        Ok(logits.sigmoid()?.try_squeeze(Some(-1))?)
    }
}

jit_wrapper! {
    FireRedVadJit(FireRedVad) {
        feat: Tensor,
        valid: Tensor,

        build(feat, valid) {
            // [BATCH, CHUNK_T, N_MELS] + [BATCH, CHUNK_T, 1] valid mask ->
            // [BATCH, CHUNK_T] speech probs. Fixed batch + window length:
            // pad rows are conv-masked via `valid`, so only the real-frame
            // rows the stitcher reads are meaningful.
            model.forward(feat, Some(valid))
        }
    }
}

pub struct FireRedVadInference {
    jit: FireRedVadJit,
    /// `[BATCH, CHUNK_T, N_MELS]` host staging for [`probs`](Self::probs):
    /// one `copyin` of the filled rows per batch.
    staging: Vec<f32>,
}

impl FireRedVadInference {
    pub fn new(model: FireRedVad) -> crate::jit::Result<Self> {
        use crate::jit::InputSpec;
        let mut jit = FireRedVadJit::new(model);
        jit.prepare_with_config(
            InputSpec::f32(&[BATCH, CHUNK_T, N_MELS]).device_local(),
            InputSpec::f32(&[BATCH, CHUNK_T, 1]),
            &svod_tensor::PrepareConfig::device_local(),
        )?;
        Ok(Self { jit, staging: vec![0.0; BATCH * CHUNK_T * N_MELS] })
    }

    /// First frame of window `w` (negative at the true start: the leading
    /// halo reads zeros) and the row span `src_lo..src_hi` of real frames it
    /// holds at row offset `dst_lo`.
    fn span(w: usize, n_frames: usize) -> (isize, usize, usize, usize) {
        let start = (w * CORE) as isize - HALO as isize;
        let src_lo = start.max(0) as usize;
        let src_hi = (start + CHUNK_T as isize).min(n_frames as isize) as usize;
        (start, src_lo, src_hi, (src_lo as isize - start) as usize)
    }

    /// Speech probabilities for a pre-CMVN fbank `feat` (row-major
    /// `[n_frames, N_MELS]`), one per frame — the host-feature entry point;
    /// [`FireRedVadProbs`] feeds the fbank JIT's output on-device instead.
    pub fn probs(&mut self, feat: &[f32], n_frames: usize) -> crate::jit::Result<Vec<f32>> {
        debug_assert_eq!(feat.len(), n_frames * N_MELS, "feat shape");
        let staging = &mut self.staging;
        Self::run(&mut self.jit, n_frames, |jit, done, b| {
            let rows = &mut staging[..b * CHUNK_T * N_MELS];
            rows.fill(0.0);
            for (i, row) in rows.chunks_mut(CHUNK_T * N_MELS).enumerate() {
                let (_, src_lo, src_hi, dst_lo) = Self::span(done + i, n_frames);
                row[dst_lo * N_MELS..(dst_lo + src_hi - src_lo) * N_MELS]
                    .copy_from_slice(&feat[src_lo * N_MELS..src_hi * N_MELS]);
            }
            Ok(jit.feat_mut()?.copyin_at(0, bytemuck::cast_slice(rows))?)
        })
    }

    /// Speech probabilities of a `[-1, 1]` waveform: each window's frames come
    /// straight from `fbank` (a `[BATCH, CHUNK_T]` JIT) and are copied into
    /// the model's input on-device.
    pub fn probs_from_samples(&mut self, fbank: &mut FireRedFbank, waveform: &[f32]) -> crate::jit::Result<Vec<f32>> {
        let shape = (fbank.batch(), fbank.capacity());
        fbank::check(
            shape == (BATCH, CHUNK_T),
            "FireRedVadInference::probs_from_samples",
            "fbank (batch, capacity)",
            format!("{shape:?}"),
            "FireRedFbank::new(BATCH, CHUNK_T)",
        )?;
        let n_frames = fbank.num_frames(waveform.len());
        Self::run(&mut self.jit, n_frames, |jit, done, b| {
            let firsts: Vec<isize> = (0..b).map(|i| Self::span(done + i, n_frames).0).collect();
            let windows = fbank.forward_windows(waveform, &firsts)?;
            Ok(jit.feat_mut()?.copy_region_from(0, windows, 0, b * CHUNK_T * N_MELS * size_of::<f32>())?)
        })
    }

    /// Windowed inference: frames are packed into `CHUNK_T`-frame windows
    /// advancing by `CORE`, each padded with `HALO` frames of neighbour
    /// context (zeros past the true edges, matching the conv zero-padding a
    /// full-length forward would see), and only the core region is kept — so
    /// the stitched result equals a single full-length forward up to float
    /// reassociation. `load(jit, first_window, b)` fills the batch's first
    /// `b` feat rows; the valid mask and the stitching are shared. Rows past
    /// `b` are stale, and neither read back nor mixed into the real rows —
    /// every op in the graph is per row.
    fn run(
        jit: &mut FireRedVadJit,
        n_frames: usize,
        mut load: impl FnMut(&mut FireRedVadJit, usize, usize) -> crate::jit::Result<()>,
    ) -> crate::jit::Result<Vec<f32>> {
        let n_windows = n_frames.div_ceil(CORE);
        let mut probs = vec![0.0f32; n_frames];
        let mut out = vec![0.0f32; BATCH * CHUNK_T];

        let mut done = 0usize;
        while done < n_windows {
            let b = (n_windows - done).min(BATCH);
            load(jit, done, b)?;
            {
                let mut view = jit.valid_view_mut::<f32>()?;
                let slice = view.as_slice_mut().expect("contiguous valid");
                slice.fill(0.0);
                for i in 0..b {
                    let (_, src_lo, src_hi, dst_lo) = Self::span(done + i, n_frames);
                    slice[i * CHUNK_T + dst_lo..i * CHUNK_T + dst_lo + (src_hi - src_lo)].fill(1.0);
                }
            }
            jit.execute()?;
            jit.output()?.copyout_prefix(bytemuck::cast_slice_mut(&mut out[..b * CHUNK_T]))?;
            for i in 0..b {
                let core_lo = (done + i) * CORE;
                let core_len = CORE.min(n_frames - core_lo);
                probs[core_lo..core_lo + core_len]
                    .copy_from_slice(&out[i * CHUNK_T + HALO..i * CHUNK_T + HALO + core_len]);
            }
            done += b;
        }
        Ok(probs)
    }
}

/// Trailing moving-average window for [`smooth_trailing`], matching upstream
/// FireRedVAD's `VadPostprocessor` default.
pub(crate) const DEFAULT_SMOOTH_WINDOW: usize = 5;

/// Waveform-level FireRedVAD: fbank JIT → DFSMN JIT → trailing smoothing,
/// yielding one speech probability per [`FRAME_SHIFT`] samples. Implements
/// [`Vad`](svod_arch::pipelines::audio::Vad), so the arch
/// [`VadSplitter`](svod_arch::pipelines::audio::VadSplitter) (assembled by
/// [`FireRedVadSplitter`]) drives the chunking.
pub struct FireRedVadProbs {
    fbank: FireRedFbank,
    vad: FireRedVadInference,
    smooth_window: usize,
}

impl FireRedVadProbs {
    /// Wrap a loaded model into the waveform→probs front-end. `smooth_window`
    /// is the trailing moving-average span (`DEFAULT_SMOOTH_WINDOW` upstream).
    pub fn new(model: FireRedVad, smooth_window: usize) -> crate::jit::Result<Self> {
        Ok(Self { fbank: FireRedFbank::new(BATCH, CHUNK_T)?, vad: FireRedVadInference::new(model)?, smooth_window })
    }
}

impl svod_arch::pipelines::audio::Vad for FireRedVadProbs {
    type Error = crate::jit::JitError;

    fn samples_per_prob(&self) -> usize {
        FRAME_SHIFT
    }

    fn probs(&mut self, waveform: &[f32]) -> std::result::Result<Vec<f32>, Self::Error> {
        let probs = self.vad.probs_from_samples(&mut self.fbank, waveform)?;
        Ok(smooth_trailing(&probs, self.smooth_window))
    }
}

/// Upstream FireRedVAD probability smoothing
/// (`VadPostprocessor._smooth_prob`): a trailing moving average of the last
/// `w` probs, with the first `w - 1` entries replaced by the cumulative mean
/// of the prefix (compensating the average's ramp-up).
pub(crate) fn smooth_trailing(probs: &[f32], w: usize) -> Vec<f32> {
    if w <= 1 {
        return probs.to_vec();
    }
    (0..probs.len())
        .map(|i| {
            let lo = (i + 1).saturating_sub(w);
            probs[lo..=i].iter().sum::<f32>() / (i + 1 - lo) as f32
        })
        .collect()
}
