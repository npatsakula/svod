//! Assembles a FireRedVAD-driven [`VadSplitter`] for long-form chunking.
//!
//! [`FireRedVadSplitter`] is a thin builder: it bakes a
//! [`ChunkerOpts`](svod_arch::vad::ChunkerOpts) from encoder
//! [`EncoderBounds`](crate::audio::EncoderBounds) plus FireRedVAD-tuned knobs and
//! wraps a [`FireRedVadProbs`] front-end in the arch
//! [`VadSplitter`](svod_arch::pipelines::audio::VadSplitter). Boundaries use the
//! encoder-aware arch chunker (not the upstream postprocessor's state machine);
//! only the probabilities match upstream.

use std::path::Path;

use bon::bon;
use snafu::{ResultExt, Snafu};
use svod_arch::pipelines::audio::{Vad, VadSplitter};

use crate::audio::{ChunkerKnobs, EncoderBounds};
use crate::firered_vad::{DEFAULT_SMOOTH_WINDOW, FireRedVad, FireRedVadProbs};

/// Builder namespace for a configured `VadSplitter<FireRedVadProbs>`.
pub struct FireRedVadSplitter;

#[bon]
impl FireRedVadSplitter {
    /// Bake a [`VadSplitter`] from a [`FireRedVadProbs`] front-end and `bounds`.
    /// Duration knobs are encoder-packing oriented (target 15-22 s chunks, hard
    /// 30 s limit, clamped to encoder capacity); the prob-count knobs default to
    /// upstream FireRedVAD's frame counts (`min_speech=20`, `min_silence=20`,
    /// `merge=0`, at 10 ms per prob). `threshold` consults `SVOD_VAD_THRESHOLD`.
    #[builder]
    pub fn builder(
        vad: FireRedVadProbs,
        bounds: EncoderBounds,
        #[builder(default = std::env::var("SVOD_VAD_THRESHOLD").ok().and_then(|s| s.parse().ok()).unwrap_or(0.4))]
        threshold: f32,
        #[builder(default = 15.0)] min_duration: f32,
        #[builder(default = 22.0)] max_duration: f32,
        /// Explicit soft target chunk duration (seconds), an override of the model
        /// recommendation supplied via `bounds.recommended_target_secs`. `None`
        /// (default) defers to `SVOD_VAD_TARGET_CHUNK_SECS` then that
        /// recommendation. Chunks pack toward the target and re-split at 1.5× it
        /// (the long-form deletion fix); a value `>= max_duration` is a no-op.
        target_duration: Option<f32>,
        #[builder(default = 30.0)] strict_limit_duration: f32,
        #[builder(default = 20)] min_speech_probs: usize,
        #[builder(default = 20)] min_silence_probs: usize,
        #[builder(default = 0)] merge_gap_probs: usize,
        trough_search_probs: Option<usize>,
        /// Pad budget (samples) per chunk side. Default `1600` (= 100 ms at
        /// 16 kHz). The actual pad is capped at half the silence gap to the
        /// neighbouring chunk — chunks never overlap into each other's speech,
        /// but at seams with enough surrounding silence the encoder sees up to
        /// this many extra samples of context on each side.
        #[builder(default = 1600)]
        pad_samples: usize,
        /// Max pre-roll (seconds) pulled into a chunk's core from the preceding
        /// silence, capped at half the gap. Default `0` (off) — FireRedVAD
        /// onsets are tight; set `SVOD_VAD_PREROLL_SECS` to enable. Mirrors the
        /// Silero splitter's knob for symmetry.
        #[builder(default = std::env::var("SVOD_VAD_PREROLL_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0))]
        preroll_secs: f32,
    ) -> VadSplitter<FireRedVadProbs> {
        // Resolve the soft target: explicit override > SVOD_VAD_TARGET_CHUNK_SECS.
        // `None` defers to the model recommendation, which `chunker_opts` ORs in
        // from `bounds.recommended_target_secs`.
        let target_duration =
            target_duration.or_else(|| std::env::var("SVOD_VAD_TARGET_CHUNK_SECS").ok().and_then(|s| s.parse().ok()));
        let opts = bounds.chunker_opts(
            vad.samples_per_prob(),
            ChunkerKnobs {
                threshold,
                min_duration,
                max_duration,
                target_duration,
                strict_limit_duration,
                min_speech_probs,
                min_silence_probs,
                merge_gap_probs,
                trough_search_probs,
                pad_samples,
                preroll_samples: (preroll_secs.max(0.0) * bounds.sample_rate as f32).round() as usize,
            },
        );
        VadSplitter::new(vad, opts)
    }

    /// Convenience: download the converted FireRedVAD weights from HF Hub,
    /// prepare the JIT, and bake the splitter. The soft chunk target defaults to
    /// `bounds.recommended_target_secs` (the model recommendation); env vars
    /// still apply.
    pub fn from_hub(bounds: &EncoderBounds) -> Result<VadSplitter<FireRedVadProbs>, FireRedVadSplitterError> {
        Self::from_model(FireRedVad::from_hub().context(LoadSnafu)?, bounds)
    }

    pub fn from_safetensors(
        path: &Path,
        bounds: &EncoderBounds,
    ) -> Result<VadSplitter<FireRedVadProbs>, FireRedVadSplitterError> {
        Self::from_model(FireRedVad::from_safetensors(path).context(LoadSnafu)?, bounds)
    }

    /// Bake a splitter around an already-loaded model (cheap to clone: weights
    /// are shared), e.g. one VAD load serving several encoders' bounds.
    pub fn from_model(
        model: FireRedVad,
        bounds: &EncoderBounds,
    ) -> Result<VadSplitter<FireRedVadProbs>, FireRedVadSplitterError> {
        let vad = FireRedVadProbs::new(model, DEFAULT_SMOOTH_WINDOW).context(InferenceSnafu)?;
        Ok(Self::builder().vad(vad).bounds(*bounds).build())
    }
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum FireRedVadSplitterError {
    #[snafu(display("loading FireRedVAD model: {source}"))]
    Load {
        #[snafu(source(from(crate::firered_vad::Error, Box::new)))]
        source: Box<crate::firered_vad::Error>,
    },
    #[snafu(display("building FireRedVAD JIT: {source}"))]
    Inference {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
}
