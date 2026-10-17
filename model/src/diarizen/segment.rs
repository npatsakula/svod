//! Chunked capture-graph-reuse driver for the DiariZen segmentation model.
//!
//! Slides a fixed 16 s window (pyannote geometry) over the input waveform,
//! batches the chunks, and reuses one captured `ExecutionPlan` across every
//! batch — the batch dim `b` is a runtime variable rebound per batch, the
//! window length is a baked constant. Stops at stacked per-chunk powerset
//! log-probs `(num_chunks, frames, K)`, the pyannote `skip_aggregation=True`
//! boundary (no overlap-add, no clustering, no RTTM).

use svod_tensor::Tensor;

use crate::jit::InputSpec;

use super::config::chunk_plan;
use super::error::{Result, SampleRateMismatchSnafu};

use super::jit::DiariZenSegmentationJit;
use super::model::DiariZenSegmentationModel;

/// Sliding-window geometry in seconds, mirroring pyannote's `SlidingWindow`
/// (`start=0`, `duration=chunk_size`, `step=chunk_size * segmentation_step`).
#[derive(Clone, Copy, Debug)]
pub struct SlidingWindow {
    pub start: f32,
    pub duration: f32,
    pub step: f32,
}

/// Per-chunk powerset log-probs, stacked WITHOUT overlap-add — the output of
/// pyannote's `Inference(..., skip_aggregation=True)`.
#[derive(Clone)]
pub struct SegmentOutput {
    /// `(num_chunks, frames, K)` log-probs over the powerset speaker subsets.
    pub logits: Tensor,
    /// Frames per chunk (derived from the prepared plan, e.g. 799 for 16 s).
    pub frames_per_chunk: usize,
    pub num_chunks: usize,
    /// Chunk-axis sliding window in seconds (`start=0`, `duration=chunk_size`,
    /// `step=chunk_size*segmentation_step`).
    pub window: SlidingWindow,
    /// Per-frame sliding window in seconds — the model's receptive-field grid
    /// (`step = extractor_stride/sr`, `duration = receptive_field/sr`). A
    /// downstream consumer maps frame `f` of chunk `c` to absolute time via
    /// this plus `window`.
    pub frame_window: SlidingWindow,
    /// Analysis-window length in input samples (for re-windowing embeddings).
    pub window_samples: usize,
    /// Hop between consecutive windows in input samples.
    pub hop_samples: usize,
}

impl SegmentOutput {
    /// Sample range `[start, end)` of chunk `c` in the original waveform. The
    /// tail of the last chunk may extend past the audio (it was zero-padded).
    pub fn chunk_sample_range(&self, chunk: usize) -> (usize, usize) {
        let start = chunk * self.hop_samples;
        (start, start + self.window_samples)
    }
}

/// Captures the segmentation forward graph once and reuses it across all
/// chunk-batches of a long waveform.
pub struct DiariZenSegmenter {
    jit: DiariZenSegmentationJit,
    max_batch: usize,
    window_samples: usize,
    hop_samples: usize,
    frames: usize,
    k: usize,
    sample_rate: u32,
    window: SlidingWindow,
    frame_window: SlidingWindow,
}

impl DiariZenSegmenter {
    /// Builds the JIT and captures the forward graph once, sized to
    /// `[inference_batch_size, 1, window_samples]`. All geometry is read from
    /// `model.config`.
    pub fn new(model: DiariZenSegmentationModel) -> Result<Self> {
        let cfg = &model.config;
        let max_batch = cfg.inference_batch_size.max(1);
        let window_samples = cfg.window_samples();
        let hop_samples = cfg.hop_samples();
        let k = cfg.powerset_class_count();
        let sample_rate = cfg.sample_rate;
        let window = SlidingWindow {
            start: 0.0,
            duration: cfg.chunk_size_seconds,
            step: cfg.chunk_size_seconds * cfg.segmentation_step,
        };
        // Per-frame receptive-field grid (pyannote `Model._receptive_field`).
        // No-padding convs ⇒ start = center(0) − (RF−1)/2 = 0.
        let sr = sample_rate as f32;
        let frame_window = SlidingWindow {
            start: 0.0,
            duration: cfg.wavlm.receptive_field_samples() as f32 / sr,
            step: cfg.wavlm.extractor_stride() as f32 / sr,
        };

        let mut jit = DiariZenSegmentationJit::new(model).with_b_bound(max_batch);
        jit.prepare(InputSpec::f32(&[max_batch, 1, window_samples]))?;

        // Frames-per-chunk is fixed (the window length is concrete); read it
        // off the plan's live output shape `(b, frames, k)` instead of
        // hardcoding.
        let frames = jit.logits_shape()?[1];

        Ok(Self { jit, max_batch, window_samples, hop_samples, frames, k, sample_rate, window, frame_window })
    }

    /// Slides the window over `waveform` (mono fp32 PCM) and returns stacked
    /// per-chunk log-probs `(num_chunks, frames, K)`. Errors if `sample_rate`
    /// differs from the model's configured rate (the window geometry is
    /// derived from it — no resampling is done here).
    pub fn segment(&mut self, waveform: &[f32], sample_rate: u32) -> Result<SegmentOutput> {
        snafu::ensure!(
            sample_rate == self.sample_rate,
            SampleRateMismatchSnafu { wav_sr: sample_rate, model_sr: self.sample_rate }
        );
        let n = waveform.len();
        let num_chunks = chunk_plan(n, self.window_samples, self.hop_samples);
        let row = self.frames * self.k;
        let mut acc: Vec<f32> = Vec::with_capacity(num_chunks * row);

        for start in (0..num_chunks).step_by(self.max_batch) {
            let real = (num_chunks - start).min(self.max_batch);

            // Pack `real` window slices into input rows 0..real (each row's tail
            // past the audio end stays zero — the trailing-window pad).
            {
                let mut view = self.jit.waveforms_view_mut::<f32>()?;
                let slice = view.as_slice_mut().expect("contiguous waveforms buffer");
                slice.fill(0.0);
                for bi in 0..real {
                    let s = (start + bi) * self.hop_samples;
                    let e = (s + self.window_samples).min(n);
                    let dst = bi * self.window_samples;
                    slice[dst..dst + (e - s)].copy_from_slice(&waveform[s..e]);
                }
            }

            // Rebind the batch var to the live count; the shrink makes the plan
            // compute exactly `real` rows, so the live output is
            // `(real, frames, k)` — a prefix of the max-batch-sized buffer.
            self.jit.execute_bound(real as i64)?;

            acc.extend(self.jit.logits_to_vec::<f32>()?);
        }

        let logits = Tensor::from_slice(&acc).try_reshape([num_chunks, self.frames, self.k])?;
        Ok(SegmentOutput {
            logits,
            frames_per_chunk: self.frames,
            num_chunks,
            window: self.window,
            frame_window: self.frame_window,
            window_samples: self.window_samples,
            hop_samples: self.hop_samples,
        })
    }
}
