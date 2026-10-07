//! GigaAM's [`Transcriber`](svod_arch::pipelines::audio::Transcriber): the
//! per-window half of the long-form ASR pipeline.
//!
//! [`GigaAmTranscriber`] turns each decode-window of audio into a
//! [`Transcript`](svod_arch::pipelines::audio::Transcript) (text + window-
//! relative word times); the arch pipeline owns the splitting, decode-window
//! geometry, core-crop, and stitching. Every JIT is prepared eagerly at
//! construction, sized to the splitter's `max_chunk_samples`. Pair it with a
//! [`Splitter`](svod_arch::pipelines::audio::Splitter) in an
//! [`Asr`](svod_arch::pipelines::audio::Asr).
//!
//! It hides the CTC vs RN-T asymmetry: the fused encoder+head JIT (CTC) vs the
//! standalone encoder + deferred lane-wave block decode (RN-T), SentencePiece
//! `▁ → space` post-processing, the encoder-output transpose RN-T needs, and
//! the CTC `frames_to_words` grouping all live inside [`HeadDecoder`].

use std::time::{Duration, Instant};

use bon::bon;
use snafu::{ResultExt, Snafu};
use svod_arch::ctc::CtcDecoder;
use svod_arch::pipelines::audio::RunOptions;
use svod_arch::rnnt::{RnntDecoder, RnntOpts};
use svod_runtime::{RunProfile, StageProfile};
use svod_tensor::PrepareConfig;

pub use svod_arch::rnnt::Word;

use crate::audio::{EncoderBounds, MelConfig, MelJit, MelSpectrogram};
use crate::gigaam::ctc::GigaAmCtcJit;
use crate::gigaam::jit::GigaAmEncoderJit;
use crate::gigaam::model::{GigaAm, Head};
use crate::gigaam::rnnt::RnntBlockBackend;
use crate::gigaam::{SubsamplingMode, subsampled_len};
use crate::jit::InputSpec;
use crate::state::scoped;

/// User-facing knobs for [`GigaAmTranscriber`].
///
/// Construct with [`TranscribeOpts::builder`] (per-field overrides) or
/// [`TranscribeOpts::from_env`] (read `SVOD_*` env vars with sensible
/// fallbacks). The two agree — `from_env()` is just `builder().build()` —
/// so `builder().beam_decode(true).build()` still consults env for the rest
/// of the fields.
///
/// Field defaults consult these env vars:
///
/// | Field             | Env var                | Fallback |
/// |-------------------|------------------------|----------|
/// | `beam_decode`     | `SVOD_BEAM_DECODE=1`  | `false`  |
/// | `max_scores_mib`  | `SVOD_MAX_SCORES_MIB` | `256`    |
///
/// These are **structural** (they shape the eagerly-built JIT/decoder). Per-run
/// behaviour — word timestamps and profiling — is not here: it's passed per call
/// as [`RunOptions`](svod_arch::pipelines::audio::RunOptions) to
/// [`Asr::transcribe`](svod_arch::pipelines::audio::Asr::transcribe), so one
/// transcriber serves words-on/off and profiled/unprofiled runs.
///
/// VAD-specific knobs (`threshold`, `min_duration`, …) live on the splitter
/// (e.g. `FireRedVadSplitter`), not here.
#[derive(Clone, Debug)]
pub struct TranscribeOpts {
    /// Promote the model's config-default CTC decoder to a beam decoder
    /// (no-op for RN-T).
    pub beam_decode: bool,
    /// Per-allocation budget for the SDPA scores buffer. Caps `max_batch`
    /// so two simultaneously live `[B, H, T_sub², dtype]` scores tensors
    /// stay under `2 × max_scores_mib` MiB.
    pub max_scores_mib: usize,
}

impl Default for TranscribeOpts {
    fn default() -> Self {
        Self::builder().build()
    }
}

#[bon]
impl TranscribeOpts {
    /// Build via the [`bon`](mod@bon) builder. Each field default consults its
    /// `SVOD_*` env var (see the struct docs for the full table) before
    /// falling back to a literal — so `builder().build()` produces the same
    /// values as [`from_env`](Self::from_env), and partial overrides
    /// (`.beam_decode(true).build()`) still env-read the rest.
    #[builder]
    pub fn builder(
        #[builder(default = std::env::var("SVOD_BEAM_DECODE").as_deref() == Ok("1"))] beam_decode: bool,
        #[builder(default = std::env::var("SVOD_MAX_SCORES_MIB").ok().and_then(|s| s.parse().ok()).unwrap_or(256))]
        max_scores_mib: usize,
    ) -> Self {
        Self { beam_decode, max_scores_mib }
    }

    /// Build from `SVOD_*` env vars with the same fallbacks as the
    /// builder. Equivalent to `Self::builder().build()`.
    pub fn from_env() -> Self {
        Self::builder().build()
    }
}

/// Per-head decoder + JIT state. CTC needs a bounds-tied head JIT (Conv1d
/// projection); RN-T's block JIT rides with [`RnntBlockBackend`].
/// One instance per `Transcriber`, so the variant-size disparity is
/// irrelevant — boxing would just add an allocation.
#[allow(clippy::large_enum_variant)]
pub(crate) enum HeadDecoder {
    Ctc { jit: GigaAmCtcJit, decoder: CtcDecoder },
    Rnnt { backend: RnntBlockBackend, decoder: RnntDecoder },
}

/// CTC equivalent of [`RnntDecoder::frames_to_words`].
///
/// Walks the decoded `text` in lockstep with `frames` (CTC's
/// `decode_with_timestamps` returns one frame index per emitted *token*; for
/// GigaAM's char-level vocab one token == one Unicode scalar, so `text.chars()`
/// is the right zip target). Splits on ASCII space — no SentencePiece on the
/// CTC side. Returns chunk-relative `[start, end)` in seconds.
pub(crate) fn ctc_frames_to_words(text: &str, frames: &[usize], frame_shift: f32) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut current = String::new();
    let mut separator = String::new();
    let mut first_frame = 0usize;
    let mut last_frame = 0usize;

    let commit = |words: &mut Vec<Word>, current: &mut String, first: usize, last: usize| {
        if !current.is_empty() {
            words.push(Word {
                text: std::mem::take(current),
                start: first as f32 * frame_shift,
                end: (last + 1) as f32 * frame_shift,
            });
        }
    };

    for (ch, &frame) in text.chars().zip(frames.iter()) {
        if ch == ' ' {
            commit(&mut words, &mut current, first_frame, last_frame);
            separator.push(ch);
            continue;
        }
        if current.is_empty() {
            first_frame = frame;
            current.push_str(&std::mem::take(&mut separator));
        }
        current.push(ch);
        last_frame = frame;
    }
    commit(&mut words, &mut current, first_frame, last_frame);
    words
}

fn rnnt_decode_err(e: svod_arch::rnnt::RnntDecodeError<crate::jit::JitError>) -> TranscribeError {
    TranscribeError::RnntDecode { source: Box::new(e) }
}

/// [`Transcriber::transcribe_windows`] output: uncropped per-window transcripts
/// (words relative to each window) plus the optional per-stage GPU profile.
type WindowDecode = Result<(Vec<svod_arch::pipelines::audio::Transcript>, Option<RunProfile>), TranscribeError>;

// ─── Errors ───────────────────────────────────────────────────────────────

/// Per-window decode failures. Variants stay pattern-matchable rather than
/// type-erased into `Box<dyn Error>`, mirroring `RnntDecodeError<JitError>`.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum TranscribeError {
    #[snafu(display("{source}"), context(false))]
    Jit {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("{source}"))]
    CtcDecode { source: svod_arch::ctc::DecodeError },
    #[snafu(display("{source}"))]
    RnntDecode { source: Box<svod_arch::rnnt::RnntDecodeError<crate::jit::JitError>> },
    #[snafu(display("{source}"))]
    Model {
        #[snafu(source(from(crate::gigaam::error::Error, Box::new)))]
        source: Box<crate::gigaam::error::Error>,
    },
    #[snafu(display("{source}"), context(false))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("{source}"), context(false))]
    Device {
        #[snafu(source(from(svod_device::error::Error, Box::new)))]
        source: Box<svod_device::error::Error>,
    },
    #[snafu(display(
        "chunk mel length {mel_frames} exceeds transcriber capacity {max_t_mel}; \
         size the transcriber from Splitter::max_chunk_samples (e.g. via Asr::assemble)"
    ))]
    CapacityExceeded { mel_frames: usize, max_t_mel: usize },
}

// ─── Transcriber ──────────────────────────────────────────────────────────

/// Per-window GigaAM transcriber: the prepared encoder/head JITs + mel
/// front-end. Implements [`svod_arch::pipelines::audio::Transcriber`]; pair it
/// with a [`Splitter`](svod_arch::pipelines::audio::Splitter) in an
/// [`Asr`](svod_arch::pipelines::audio::Asr). Every JIT is prepared eagerly at
/// construction, sized to the `max_chunk_samples` the splitter will emit
/// (clamped to the encoder's hard ceiling).
pub struct GigaAmTranscriber {
    model: GigaAm,
    mel: MelSpectrogram,
    /// Graph front-end over host-framed windows; its device output feeds the
    /// encoder's mel input.
    mel_jit: MelJit,
    /// Host staging for the mel JIT's device-local `[max_batch, framed_len]`
    /// input: one `copyin` of the batch's rows instead of kernels reading
    /// pinned host memory over the bus. Rows past the batch are stale; their
    /// frame count is zero, so the mel graph masks them to zero.
    framed: Vec<f32>,
    head_decoder: HeadDecoder,
    encoder_jit: Option<GigaAmEncoderJit>,
    max_batch: usize,
    max_t_mel: usize,
}

impl GigaAmTranscriber {
    /// Build the transcriber and prepare every JIT eagerly, sized to
    /// `max_chunk_samples` (the splitter's emission ceiling) clamped to the
    /// encoder's capacity. `model` is cloned into each JIT (cheap: weights are
    /// shared via `Tensor` handle Arcs).
    pub fn new(model: GigaAm, opts: TranscribeOpts, max_chunk_samples: usize) -> Result<Self, TranscribeError> {
        let mel = MelSpectrogram::new(&MelConfig {
            sample_rate: model.config.sample_rate,
            n_fft: model.config.n_fft,
            hop_length: model.config.hop_length,
            win_length: model.config.win_length,
            n_mels: model.config.n_mels,
            center: model.config.mel_center,
            mel_scale: crate::audio::MelScale::Htk,
            periodic: true,
            pad_mode: crate::audio::PadMode::Reflect,
            preemphasis: None,
            log: crate::audio::MelSpectrogram::LOG,
        });

        let subsampling_factor = model.config.subsampling_factor;
        let hop_length = model.config.hop_length;
        let model_bounds = EncoderBounds {
            sample_rate: model.config.sample_rate as u32,
            hop_length,
            subsampling_factor,
            max_mel_frames: model.config.max_mel_frames,
            // Internal JIT-sizing bounds only; not used for chunking.
            recommended_target_secs: None,
        };
        // Clamp the splitter's emission ceiling to encoder capacity, then round
        // up to the next power of two so the JIT codegen sees a clean
        // factorisation.
        let chunk_samples_cap = max_chunk_samples.min(model_bounds.max_samples());
        let chunk_mel = (chunk_samples_cap / hop_length).saturating_add(2 * subsampling_factor);
        let max_t_mel = chunk_mel.max(1).next_power_of_two().min(model.config.max_mel_frames).max(subsampling_factor);

        // SDPA scores `[B, H, T_sub², dtype]` are live twice during attention;
        // budget `max_batch` so they stay under `2 * max_scores_mib`.
        let t_sub_max = (max_t_mel / subsampling_factor).max(1);
        let scores_dtype_bytes = model.encoder.input_dtype().bytes();
        let bytes_per_batch = model.config.n_heads * t_sub_max * t_sub_max * scores_dtype_bytes;
        let target_scores_bytes = opts.max_scores_mib * 1024 * 1024;
        let max_batch_by_memory = (target_scores_bytes / bytes_per_batch.max(1)).max(1);
        let max_batch = max_batch_by_memory.min(model.config.max_batch_size);

        // Device-local output: the [B, T_sub, vocab] log-probs come back in
        // one copy over the copy engine. Decoding straight from the host-mapped
        // BAR costs a PCIe round trip per float — 6 s of a 13 s run on a
        // discrete AMD card.
        let prepare_config = PrepareConfig::device_local();
        // The encoder's mel input is device-local: it is only ever written by
        // an on-device copy from the mel JIT's output.
        let mel_spec = InputSpec::f32(&[max_batch, model.config.n_mels, max_t_mel]).device_local();
        let lengths_spec = InputSpec::i32(&[max_batch]);
        // Framed rows long enough for exactly `max_t_mel` frames.
        let framed_len = (max_t_mel - 1) * hop_length + model.config.n_fft;
        let mut mel_jit = MelJit::new(mel.clone());
        scoped("mel", || {
            mel_jit.prepare_with_config(
                InputSpec::f32(&[max_batch, framed_len]).device_local(),
                InputSpec::i32(&[max_batch]),
                &PrepareConfig::device_local(),
            )
        })?;
        let framed = vec![0.0f32; max_batch * framed_len];

        // The standalone encoder JIT exists only for the RN-T path (it shares
        // the encoder with the predictor/joint step JITs). CTC fuses the
        // encoder into `GigaAmCtcJit`, so `encoder_jit` stays `None` there —
        // that fusion is what keeps the encoder output on-device.
        let mut encoder_jit: Option<GigaAmEncoderJit> = None;

        let head_decoder = match &model.head {
            Head::Ctc(_) => {
                let decoder = if opts.beam_decode {
                    match &model.config.decoder {
                        CtcDecoder::Greedy(g) => CtcDecoder::Beam(Box::new(svod_arch::ctc::BeamDecoder::new(
                            g.vocabulary().to_vec(),
                            svod_arch::ctc::BeamOpts::default(),
                        ))),
                        other => other.clone(),
                    }
                } else {
                    model.config.decoder.clone()
                };
                let mut jit = GigaAmCtcJit::new(model.clone());
                // The stage name the profiler groups by is the root of every
                // origin path the capture below mints.
                scoped("ctc_head", || jit.prepare_with_config(mel_spec, lengths_spec, &prepare_config))?;
                HeadDecoder::Ctc { jit, decoder }
            }
            Head::Rnnt { runtime, .. } => {
                let mut enc = GigaAmEncoderJit::new(model.clone());
                // Device-local output: the [B, T_sub, d_model] readback goes
                // over the SDMA copy queue instead of the ~21 MB/s host-mapped
                // BAR (the old first-execute hang was tied to per-execute
                // schedule re-instantiation under runtime vars; the plan is
                // all-static now).
                let enc_config = PrepareConfig::device_local();
                scoped("encoder", || enc.prepare_with_config(mel_spec, lengths_spec, &enc_config))?;
                encoder_jit = Some(enc);
                // Decode lanes are independent of the encoder batch: wider
                // waves amortize the per-step launch floor over more chunks
                // (steps per wave = max frames in the wave, not the sum).
                // State per lane is tiny; 64 lanes cost 1.3x a 32-lane block
                // for 2x the chunks (RTX 3060), halving the wave count.
                const DECODE_LANES: usize = 64;
                let subs_kernel = match model.config.subsampling_mode {
                    SubsamplingMode::Conv1d => model.config.subs_kernel_size,
                    SubsamplingMode::Conv2d => 3,
                };
                let max_t_sub = subsampled_len(subs_kernel, max_t_mel);
                let backend =
                    scoped("decode", || RnntBlockBackend::from_model(model.clone(), DECODE_LANES, max_t_sub))?;
                let decoder = RnntDecoder::new(
                    runtime.vocabulary.clone(),
                    RnntOpts { max_symbols_per_step: runtime.max_symbols_per_step },
                );
                HeadDecoder::Rnnt { backend, decoder }
            }
        };

        Ok(Self { model, mel, mel_jit, framed, head_decoder, encoder_jit, max_batch, max_t_mel })
    }

    /// Windows one compiled dispatch decodes. Every dispatch runs at this batch
    /// shape, so a caller pooling windows fills batches of exactly this size.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }
}

impl svod_arch::pipelines::audio::Transcriber for GigaAmTranscriber {
    type Error = TranscribeError;

    fn sample_rate(&self) -> u32 {
        self.model.config.sample_rate as u32
    }

    /// Encode + decode each decode-window's audio into uncropped per-window
    /// transcripts (word times relative to the window start). Owns the full
    /// batched encoder + per-head decode (CTC fused; RN-T deferred lane-wave)
    /// and the per-stage profile; the trait's `transcribe_chunks` default does
    /// the core-crop and stitch.
    fn transcribe_windows(&mut self, windows: &[&[f32]], opts: RunOptions) -> WindowDecode {
        let profile = opts.profile;
        use svod_arch::pipelines::audio::{Transcript, words_to_text};

        // No windows (silence-only audio): nothing to encode. Guards the
        // `num_chunks.div_ceil(max_batch) - 1` batch-index math below.
        if windows.is_empty() {
            return Ok((Vec::new(), profile.then(RunProfile::default)));
        }

        let sample_rate_hz = self.model.config.sample_rate;
        let d_model = self.model.config.d_model;
        let subs_kernel_size = match self.model.config.subsampling_mode {
            SubsamplingMode::Conv1d => self.model.config.subs_kernel_size,
            SubsamplingMode::Conv2d => 3,
        };
        let max_t_mel = self.max_t_mel;
        let max_batch = self.max_batch;
        // The JIT runs at constant shape `[max_batch, *, max_t_mel]`, so the
        // encoder output buffer is always `[max_batch, max_t_sub, *]`.
        let max_t_sub = subsampled_len(subs_kernel_size, max_t_mel);

        let mel_lens: Vec<usize> = windows.iter().map(|w| self.mel.num_frames(w.len())).collect();
        // A window longer than the JIT was sized for would overrun the mel
        // JIT's framed rows; fail cleanly instead. `Asr::assemble`
        // sizes the transcriber from the splitter, so this never fires there —
        // it guards a hand-mismatched splitter/transcriber pair.
        if let Some(&mel_frames) = mel_lens.iter().find(|&&m| m > max_t_mel) {
            return CapacityExceededSnafu { mel_frames, max_t_mel }.fail();
        }
        let num_chunks = windows.len();
        let mut transcripts: Vec<Transcript> = Vec::with_capacity(num_chunks);
        // RN-T: encoder frames accumulated across all encode batches, decoded
        // afterwards in backend-wide lane waves (fills `transcripts` in order).
        let mut all_frames: Vec<Vec<f32>> = Vec::new();
        let mut all_valid: Vec<usize> = Vec::new();
        // Per-stage wall-clock, accumulated across batches. The JITs submit
        // async; the GPU drains on the first host read, so each stage timer is
        // bounded by its drain point.
        let (mut t_mel, mut t_encoder, mut t_decode) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
        let profile_batch = profile.then(|| 3.min(num_chunks.div_ceil(max_batch).saturating_sub(1)) * max_batch);
        let mut prof = profile.then(RunProfile::default);
        for chunk_batch_start in (0..num_chunks).step_by(max_batch) {
            let b = (num_chunks - chunk_batch_start).min(max_batch);
            let chunk_lengths: Vec<usize> = (0..b).map(|bi| mel_lens[chunk_batch_start + bi]).collect();

            // Frame the windows on the host, upload them to the mel JIT's
            // device-local input and run it; the output is copied to the
            // encoder on-device.
            let t_stage = Instant::now();
            let framed_len = self.framed.len() / max_batch;
            for (bi, row) in self.framed.chunks_mut(framed_len).take(b).enumerate() {
                self.mel.frame_into(windows[chunk_batch_start + bi], row);
            }
            self.mel_jit.framed_mut()?.copyin_at(0, bytemuck::cast_slice(&self.framed[..b * framed_len]))?;
            pack_lengths_buffer(self.mel_jit.frames_view_mut::<i32>()?, &chunk_lengths);
            self.mel_jit.execute()?;
            let batch_mels = self.mel_jit.output()?;
            t_mel += t_stage.elapsed();

            // CTC's `GigaAmCtcJit` is the fused encoder+head (log-probs, no host
            // round-trip); RN-T runs the standalone encoder JIT, decoded later.
            match &mut self.head_decoder {
                HeadDecoder::Ctc { jit, decoder } => {
                    let t_pack = Instant::now();
                    jit.mel_mut()?.copy_from(batch_mels)?;
                    pack_lengths_buffer(jit.lengths_view_mut::<i32>()?, &chunk_lengths);
                    t_mel += t_pack.elapsed();

                    let t_enc = Instant::now();
                    if profile_batch == Some(chunk_batch_start) {
                        let kernels = jit.execute_profiled()?;
                        if let Some(p) = &mut prof {
                            p.push(StageProfile::gpu("ctc_head", Duration::ZERO, kernels));
                        }
                    } else {
                        jit.execute()?;
                    }
                    let total_vocab = decoder.total_vocab();
                    let item_stride = max_t_sub * total_vocab;
                    // One prefix copyout drains the fused encoder+head dispatch
                    // and skips the inactive lanes of a partial last batch.
                    let mut raw = vec![0f32; b * item_stride];
                    jit.log_probs()?.copyout_prefix(bytemuck::cast_slice_mut(&mut raw))?;
                    t_encoder += t_enc.elapsed();
                    let flat: &[f32] = &raw;
                    for (bi, mel_len) in chunk_lengths.iter().enumerate() {
                        let actual_sub = subsampled_len(subs_kernel_size, *mel_len);
                        // Frames span the decode window; frame_shift maps a frame
                        // index to window-relative seconds.
                        let window_len = windows[chunk_batch_start + bi].len();
                        let frame_shift = (window_len as f32 / sample_rate_hz as f32) / (actual_sub.max(1) as f32);
                        let item_slice = &flat[bi * item_stride..bi * item_stride + item_stride];

                        let t_dec = Instant::now();
                        let (text, frames) = decoder
                            .decode_with_timestamps(item_slice, max_t_sub, actual_sub)
                            .context(CtcDecodeSnafu)?;
                        t_decode += t_dec.elapsed();

                        let words = ctc_frames_to_words(&text, &frames, frame_shift);
                        transcripts.push(Transcript { text, words, ..Default::default() });
                    }
                }
                HeadDecoder::Rnnt { .. } => {
                    let enc_jit = self.encoder_jit.as_mut().expect("RN-T path has a standalone encoder JIT");
                    let t_pack = Instant::now();
                    enc_jit.mel_mut()?.copy_from(batch_mels)?;
                    pack_lengths_buffer(enc_jit.lengths_view_mut::<i32>()?, &chunk_lengths);
                    t_mel += t_pack.elapsed();

                    let t_enc = Instant::now();
                    if profile_batch == Some(chunk_batch_start) {
                        let kernels = enc_jit.execute_profiled()?;
                        if let Some(p) = &mut prof {
                            p.push(StageProfile::gpu("encoder", Duration::ZERO, kernels));
                        }
                    } else {
                        enc_jit.execute()?;
                    }
                    let item_stride = max_t_sub * d_model;
                    // Frame-major [B, max_t_sub, d_model]: one contiguous prefix
                    // copyout drains the dispatch and skips inactive lanes.
                    let mut raw = vec![0f32; b * item_stride];
                    enc_jit.frames()?.copyout_prefix(bytemuck::cast_slice_mut(&mut raw))?;
                    t_encoder += t_enc.elapsed();
                    let flat: &[f32] = &raw;
                    // Lanes decouple from the encoder batch: collect every
                    // chunk's frames, decode them in one wide wave after the loop.
                    for (bi, mel_len) in chunk_lengths.iter().enumerate() {
                        let actual_sub = subsampled_len(subs_kernel_size, *mel_len);
                        let base = bi * item_stride;
                        all_frames.push(flat[base..base + actual_sub * d_model].to_vec());
                        all_valid.push(actual_sub);
                    }
                }
            }
        }

        // RN-T: decode every chunk in lane waves as wide as the backend
        // (steps per wave = the wave's max frames, not the sum over batches).
        if let HeadDecoder::Rnnt { backend, decoder } = &mut self.head_decoder {
            let lanes = svod_arch::rnnt::BatchBlockStep::batch(backend);
            for wave_start in (0..all_frames.len()).step_by(lanes) {
                let wave_end = (wave_start + lanes).min(all_frames.len());
                let valid = &all_valid[wave_start..wave_end];

                let t_dec = Instant::now();
                backend.bind_batch(&all_frames[wave_start..wave_end], valid)?;
                let lane_results = decoder.decode_batch_blocks(valid, backend).map_err(rnnt_decode_err)?;
                t_decode += t_dec.elapsed();

                for (li, (_raw, emissions)) in lane_results.into_iter().enumerate() {
                    let window_len = windows[wave_start + li].len();
                    let frame_shift = (window_len as f32 / sample_rate_hz as f32) / (valid[li].max(1) as f32);
                    // `frames_to_words` converts SentencePiece boundaries into
                    // exact leading-space fragments for direct reconstruction.
                    let words = decoder.frames_to_words(&emissions, frame_shift);
                    transcripts.push(Transcript { text: words_to_text(&words), words, ..Default::default() });
                }
            }
        }

        if let HeadDecoder::Rnnt { backend, .. } = &self.head_decoder {
            let s = &backend.stats;
            let block_steps = svod_arch::rnnt::BatchBlockStep::block_steps(backend) as u64;
            let lanes = svod_arch::rnnt::BatchBlockStep::batch(backend) as u64;
            let frames_total: usize = all_valid.iter().sum();
            tracing::info!(
                target: "svod_model::gigaam::transcribe",
                n_blocks = s.n_blocks,
                steps_per_lane = s.n_blocks * block_steps,
                tokens_emitted = s.steps_emitted,
                tape_slots = s.n_blocks * lanes * block_steps,
                frames_total,
                exec_ms = s.t_exec.as_secs_f64() * 1e3,
                read_ms = s.t_read.as_secs_f64() * 1e3,
                "rnnt block stats",
            );
        }

        tracing::info!(
            target: "svod_model::gigaam::transcribe",
            num_chunks,
            mel_ms = t_mel.as_secs_f64() * 1e3,
            encoder_ms = t_encoder.as_secs_f64() * 1e3,
            decode_ms = t_decode.as_secs_f64() * 1e3,
            "gigaam stage breakdown",
        );

        if let Some(p) = &mut prof {
            // GPU stages share the accumulated encoder wall; prepend the
            // host-only mel stage so display order is mel → encoder.
            for s in &mut p.stages {
                s.wall = t_encoder;
            }
            p.stages.insert(0, StageProfile::host("mel", t_mel));
        }

        Ok((transcripts, prof))
    }
}

/// Pack per-chunk mel-frame counts into a JIT lengths input view `[max_batch]`,
/// zero-padding unused entries.
fn pack_lengths_buffer(mut view: ndarray::ArrayViewMutD<'_, i32>, chunk_lengths: &[usize]) {
    let slice = view.as_slice_mut().expect("contiguous lengths buffer");
    slice.fill(0);
    for (dst, &len) in slice.iter_mut().zip(chunk_lengths) {
        *dst = len as i32;
    }
}
