//! Prepared Whisper recognition and aligned-transcription stages.
//!
//! Each decode window runs through a concrete-capacity encoder, a prefill that
//! also projects the cross-attention caches, and fixed-slot cached decoding.
//! The independent aligner replays finalized tokens through a teacher-forced
//! graph and computes word timings on the host.

use std::time::Instant;

use snafu::Snafu;
use svod_arch::pipelines::audio::{RunOptions, Segment, Transcriber, Transcript, WindowAdvance};
use svod_dtype::DType;
use svod_runtime::{RunProfile, StageProfile};
use svod_tensor::PrepareConfig;

use crate::jit::InputSpec;

use super::aligner::{WhisperAligner, WhisperAlignmentInput};
use super::config::{N_AUDIO_CTX, N_FRAMES, N_SAMPLES, SAMPLE_RATE, WhisperSize};
use super::decode::{
    DecodeOptions, DecodeResult, attempt_strategies, decode_err, detect_language_profile, prefill_decode_seed,
    run_fixed_slot_decode, split_into_segments, strategy_width, window_seek,
};
use super::jit::{WhisperDecoderStepJit, WhisperEncoderJit, WhisperPrefillJit};
use super::mel::{WhisperMel, WhisperMelJit};
use super::model::Whisper;
use super::plan::WhisperPlan;
use super::profile::{CopyProfile, GraphProfile};
use super::tokenizer::WhisperTokenizer;

pub use svod_arch::rnnt::Word;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum TranscribeError {
    #[snafu(display("{source}"), context(false))]
    Jit {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("{source}"), context(false))]
    Model {
        #[snafu(source(from(super::error::Error, Box::new)))]
        source: Box<super::error::Error>,
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
}

type Result<T> = std::result::Result<T, TranscribeError>;

/// Prepared timestamp-enabled recognizer. It owns only recognition graphs;
/// word alignment is a separate [`WhisperAligner`] stage.
pub struct WhisperRecognizer {
    /// Graph front-end; its device output feeds the encoder's mel input.
    mel_jit: WhisperMelJit,
    /// Host staging for the mel JIT's device-local `[max_batch, N_SAMPLES]`
    /// input: one `copyin` of the batch's rows instead of kernels reading
    /// pinned host memory over the bus. Rows past the batch are stale; the
    /// mel and encoder graphs are per row and only the batch's rows are read.
    samples: Vec<f32>,
    encoder_jit: WhisperEncoderJit,
    /// Prompt plus one window of encoder features; also answers language
    /// detection, since its first logits row depends on SOT alone.
    prefill_jit: WhisperPrefillJit,
    /// Concrete fixed-capacity step graph. Requests keep stable row ownership;
    /// inactive rows execute with ignored outputs.
    step_jit: WhisperDecoderStepJit,
    tokenizer: WhisperTokenizer,
    options: DecodeOptions,
    feature_dtype: DType,
    n_audio_state: usize,
    n_vocab: usize,
    n_text_ctx: usize,
    max_batch: usize,
    /// Max concurrent decode lanes in the batched step JIT.
    max_lanes: usize,
    plan: WhisperPlan,
}

struct RecognizedWindow {
    result: DecodeResult,
    cross_k: svod_device::Buffer,
    cross_v: svod_device::Buffer,
    audio_samples: usize,
    segments: Vec<Segment>,
    consumed_sec: f32,
}

impl RecognizedWindow {
    fn new(
        result: DecodeResult,
        (cross_k, cross_v): (svod_device::Buffer, svod_device::Buffer),
        audio_samples: usize,
        tokenizer: &WhisperTokenizer,
    ) -> Self {
        let window_sec = audio_samples as f32 / SAMPLE_RATE as f32;
        let segments = split_into_segments(&result.tokens, tokenizer, window_sec);
        let consumed_sec = window_seek(&result.tokens, tokenizer, window_sec);
        Self { result, cross_k, cross_v, audio_samples, segments, consumed_sec }
    }

    fn transcript(&self, words: Vec<Word>) -> Transcript {
        Transcript {
            text: self.result.text.clone(),
            words,
            segments: self.segments.clone(),
            language: self.result.language.clone(),
            consumed_sec: Some(self.consumed_sec),
        }
    }
}

/// Fold one batch's stages and its copy stages into the run profile.
fn fold_profile(run: &mut Option<RunProfile>, batch: Option<RunProfile>, copies: &CopyProfile) {
    let (Some(run), Some(mut batch)) = (run.as_mut(), batch) else { return };
    for stage in copies.stages() {
        batch.push(stage);
    }
    run.merge(batch);
}

impl WhisperRecognizer {
    pub fn new(
        model: Whisper,
        tokenizer: WhisperTokenizer,
        options: DecodeOptions,
        max_chunk_samples: usize,
    ) -> Result<Self> {
        let plan = WhisperPlan::for_recognizer(&model.dims);
        Self::new_with_plan(model, tokenizer, options, max_chunk_samples, plan)
    }

    pub fn new_with_plan(
        model: Whisper,
        tokenizer: WhisperTokenizer,
        options: DecodeOptions,
        max_chunk_samples: usize,
        plan: WhisperPlan,
    ) -> Result<Self> {
        plan.validate().map_err(decode_err)?;
        options.validate()?;
        if attempt_strategies(&options).into_iter().any(|strategy| strategy_width(strategy) > plan.decoder_slots) {
            return Err(decode_err("configured decode beam width exceeds decoder_slots").into());
        }
        if max_chunk_samples > N_SAMPLES {
            let msg = format!("Whisper decode windows are limited to {N_SAMPLES} samples, got {max_chunk_samples}");
            return Err(decode_err(&msg).into());
        }
        let dims = &model.dims;
        let (n_mels, n_audio_state, n_vocab, n_text_ctx) =
            (dims.n_mels, dims.n_audio_state, dims.n_vocab, dims.n_text_ctx);
        let (layer_heads, d_head) = (dims.n_text_layer * dims.n_text_head, dims.n_text_state / dims.n_text_head);
        let (feature_dtype, cache_dtype) = (dims.dtype.clone(), dims.cache_dtype());
        let (max_batch, max_lanes) = (plan.encoder_batch, plan.decoder_slots);
        // Device-local outputs everywhere: the prefill and step logits are read
        // with one copyout per row and the caches move on-device. Reading the
        // step logits through the host mapping instead costs a BAR read per
        // row per token — 4 ms each on a discrete AMD card, 97 s of a 10-minute
        // clip.
        let prepare_config = PrepareConfig::device_local();

        let mut mel_jit = WhisperMelJit::new(WhisperMel::new(n_mels));
        mel_jit.prepare_with_config(
            InputSpec::f32(&[max_batch, N_SAMPLES]).device_local(),
            &PrepareConfig::device_local(),
        )?;
        let samples = vec![0.0f32; max_batch * N_SAMPLES];

        // Encoder JIT: [max_batch, n_mels, N_FRAMES], device-local on both
        // sides — the mel input is only ever written by an on-device copy
        // from the mel JIT's output.
        let mut encoder_jit = WhisperEncoderJit::new(model.clone());
        encoder_jit.prepare_with_config(
            InputSpec::f32(&[max_batch, n_mels, N_FRAMES]).device_local(),
            &PrepareConfig::device_local(),
        )?;

        // The prompt is structural: multilingual [SOT, language, task],
        // English-only [SOT]. Compiled once at that length, reused every window.
        let prompt_len = if model.is_multilingual() { 3 } else { 1 };
        let mut prefill_jit = WhisperPrefillJit::new(model.clone());
        prefill_jit.prepare_with_config(
            InputSpec::i32(&[1, prompt_len]),
            InputSpec::new(&[1, N_AUDIO_CTX, n_audio_state], feature_dtype.clone()).device_local(),
            &prepare_config,
        )?;

        // Fixed concrete batch keeps tensor-core dimensions static and avoids
        // cache movement when lanes finish. There is one cross cache per
        // concurrently-decoded window, not per decoder row: every row of an
        // attempt reads its owner's cache through `cross_cache_map`.
        let cache = |rows: usize, positions: usize| {
            InputSpec::new(&[rows, positions, layer_heads, d_head], cache_dtype.clone()).device_local()
        };
        let mut step_jit = WhisperDecoderStepJit::new(model);
        step_jit.prepare_with_config(
            InputSpec::i32(&[max_lanes, 1]),
            cache(max_lanes, n_text_ctx),
            cache(max_lanes, n_text_ctx),
            cache(max_batch, N_AUDIO_CTX),
            cache(max_batch, N_AUDIO_CTX),
            InputSpec::i32(&[max_lanes]),
            InputSpec::i32(&[max_lanes]),
            &prepare_config,
        )?;
        // The step graph runs every lane each dispatch, reserved or not, and
        // the kernels index the cross cache and the positional embedding by
        // these values without bounds checks, so every lane must hold a valid
        // one from the start. Zero is always in range; seeding and stepping
        // then overwrite the lanes in use.
        step_jit.cross_cache_map_mut()?.as_host_bytes_mut()?.fill(0);
        step_jit.self_key_lens_mut()?.as_host_bytes_mut()?.fill(0);

        Ok(Self {
            mel_jit,
            samples,
            encoder_jit,
            prefill_jit,
            step_jit,
            tokenizer,
            options,
            feature_dtype,
            n_audio_state,
            n_vocab,
            n_text_ctx,
            max_batch,
            max_lanes,
            plan,
        })
    }

    /// Override the decode language (`None` ⇒ auto-detect) for subsequent
    /// [`Transcriber::transcribe_windows`] calls. Lets a reusable transcriber
    /// serve requests with differing languages without rebuilding the JITs.
    pub fn set_language(&mut self, language: Option<String>) {
        self.options.language = language;
    }

    /// Max concurrent decode lanes the batched step JIT was compiled for.
    /// The scheduler treats this as the GPU concurrency bound.
    pub fn max_lanes(&self) -> usize {
        self.max_lanes
    }

    pub fn plan(&self) -> &WhisperPlan {
        &self.plan
    }

    /// Recognize up to one encoder batch of windows.
    fn recognize_windows(
        &mut self,
        windows: &[&[f32]],
        profile: bool,
    ) -> Result<(Vec<RecognizedWindow>, Option<RunProfile>, CopyProfile)> {
        let b = windows.len();
        let mut copies = CopyProfile::new(profile);
        if b == 0 {
            return Ok((Vec::new(), profile.then(RunProfile::default), copies));
        }
        if b > self.max_batch {
            return Err(decode_err("more windows than the encoder batch").into());
        }
        let (mut encoder, mut language, mut prefill, mut steps) = (
            GraphProfile::new(profile),
            GraphProfile::new(profile),
            GraphProfile::new(profile),
            GraphProfile::new(profile),
        );

        // ── Mel: pad-or-trim the windows, upload them to the mel JIT's
        // device-local input, run it, and copy its output across on-device.
        let started = Instant::now();
        for (row, window) in self.samples.chunks_mut(N_SAMPLES).zip(windows) {
            WhisperMel::pad_or_trim_into(window, row);
        }
        let src: &[u8] = bytemuck::cast_slice(&self.samples[..b * N_SAMPLES]);
        let fence = self.mel_jit.samples_mut()?.clone();
        let mel_jit = &mut self.mel_jit;
        copies.h2d("mel_input", 1, src.len(), &fence, || -> Result<()> {
            Ok(mel_jit.samples_mut()?.copyin_at(0, src)?)
        })?;
        self.mel_jit.execute()?;
        self.encoder_jit.mel_mut()?.copy_from(self.mel_jit.output()?)?;
        let t_mel = started.elapsed();

        // ── Encode: one dispatch for b windows ───────────────────────────
        encoder.execute(
            &mut self.encoder_jit,
            |jit| -> Result<()> { Ok(jit.execute()?) },
            |jit| {
                let kernels = jit.execute_profiled_static()?;
                jit.output()?.synchronize()?;
                Ok(kernels)
            },
        )?;

        // ── Prefill per window: its cross caches are projected from its slice
        // of the encoder output, and its seed outlives every fallback attempt.
        let feature_bytes = N_AUDIO_CTX * self.n_audio_state * self.feature_dtype.bytes();
        let mut detected_language: Option<String> = None;
        let mut seeds = Vec::with_capacity(b);
        let mut decode_options = Vec::with_capacity(b);
        for bi in 0..b {
            let features = self.encoder_jit.output()?.clone();
            let prefill_jit = &mut self.prefill_jit;
            copies.d2d("prefill_features", 1, feature_bytes, &features, || -> Result<()> {
                Ok(prefill_jit.audio_features_mut()?.copy_region_from(
                    0,
                    &features,
                    bi * feature_bytes,
                    feature_bytes,
                )?)
            })?;

            let mut options = self.options.clone();
            if !self.tokenizer.multilingual {
                options.language = Some("en".into());
            } else if options.language.is_none() {
                // One recording is one language: the first window's answer
                // serves the rest of this call.
                if detected_language.is_none() {
                    let detection = detect_language_profile(
                        &mut self.prefill_jit,
                        self.n_vocab,
                        &self.tokenizer,
                        &mut copies,
                        &mut language,
                    )?;
                    detected_language = Some(detection.language);
                }
                options.language.clone_from(&detected_language);
            }
            seeds.push(prefill_decode_seed(
                &mut self.prefill_jit,
                &self.tokenizer,
                &options,
                self.n_text_ctx,
                self.n_vocab,
                &mut copies,
                &mut prefill,
            )?);
            decode_options.push(options);
        }

        let started = Instant::now();
        let (results, stats) = run_fixed_slot_decode(
            &seeds,
            &decode_options,
            &mut self.step_jit,
            self.max_lanes,
            &self.tokenizer,
            self.n_text_ctx,
            self.n_vocab,
            &mut copies,
            &mut steps,
        )?;
        let t_scheduler = started.elapsed();

        let recognized = results
            .into_iter()
            .zip(&decode_options)
            .zip(seeds)
            .zip(windows)
            .map(|(((mut result, options), seed), window)| {
                if result.should_skip(options) {
                    result.clear_speech();
                }
                RecognizedWindow::new(result, seed.into_cross_kv(), window.len(), &self.tokenizer)
            })
            .collect();

        let prof = if profile {
            let mut p = RunProfile::default();
            p.push(StageProfile::host("mel", t_mel));
            p.push(encoder.stage("encoder"));
            if language.executions != 0 {
                p.push(language.stage("language_detection"));
            }
            p.push(prefill.stage("prefill"));
            let mut step = steps.stage("decoder_step");
            stats.annotate(&mut step);
            p.push(step);
            let mut scheduler = StageProfile::host("decoder_scheduler_total", t_scheduler);
            scheduler.meta.insert(
                "timing_semantics".into(),
                "non-additive end-to-end control-loop wall including host work, decoder_step waits, and synchronized cache/control copies".into(),
            );
            p.push(scheduler);
            Some(p)
        } else {
            None
        };
        Ok((recognized, prof, copies))
    }
}

impl Transcriber for WhisperRecognizer {
    type Error = TranscribeError;

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE as u32
    }

    /// Whisper ends a window at its last emitted timestamp, which routinely
    /// lands short of the window edge, so the next window starts there.
    fn window_advance(&self) -> Option<WindowAdvance> {
        Some(WindowAdvance::default())
    }

    fn transcribe_windows(
        &mut self,
        windows: &[&[f32]],
        opts: RunOptions,
    ) -> Result<(Vec<Transcript>, Option<RunProfile>)> {
        let mut transcripts = Vec::with_capacity(windows.len());
        let mut run = opts.profile.then(RunProfile::default);
        // Recognized windows own device-resident cross K/V snapshots. Consume
        // them one encoder batch at a time instead of retaining one pair for
        // every window in a long recording.
        for batch in windows.chunks(self.max_batch) {
            let (recognized, batch_profile, copies) = self.recognize_windows(batch, opts.profile)?;
            transcripts.extend(recognized.iter().map(|window| window.transcript(Vec::new())));
            fold_profile(&mut run, batch_profile, &copies);
        }
        Ok((transcripts, run))
    }
}

/// Recognizer composed with the independent, fixed-shape word aligner. A call
/// asking for words (`RunOptions::words`) gets them DTW-aligned; one that does
/// not skips the aligner. The prepared recognition graph is the same either way.
pub struct WhisperAlignedTranscriber {
    recognizer: WhisperRecognizer,
    aligner: WhisperAligner,
}

impl WhisperAlignedTranscriber {
    pub fn new(
        model: Whisper,
        tokenizer: WhisperTokenizer,
        options: DecodeOptions,
        size: WhisperSize,
        max_chunk_samples: usize,
    ) -> Result<Self> {
        let plan = WhisperPlan::for_model(&model.dims, size);
        Self::new_with_plan(model, tokenizer, options, size, max_chunk_samples, plan)
    }

    pub fn new_with_plan(
        model: Whisper,
        tokenizer: WhisperTokenizer,
        options: DecodeOptions,
        size: WhisperSize,
        max_chunk_samples: usize,
        plan: WhisperPlan,
    ) -> Result<Self> {
        plan.validate().map_err(decode_err)?;
        // The aligner replays what recognition emitted, so it is sized to the
        // decode budget rather than to the whole text context.
        let max_tokens = options.sample_len.unwrap_or(model.dims.n_text_ctx / 2);
        let aligner = WhisperAligner::new(model.clone(), size, plan.alignment_batch, max_tokens)?;
        let recognizer = WhisperRecognizer::new_with_plan(model, tokenizer, options, max_chunk_samples, plan)?;
        Ok(Self { recognizer, aligner })
    }

    pub fn set_language(&mut self, language: Option<String>) {
        self.recognizer.set_language(language);
    }

    fn align_recognized(
        &mut self,
        recognized: &[RecognizedWindow],
        copies: &mut CopyProfile,
        profile: &mut Option<RunProfile>,
        last_speech_timestamp: &mut f32,
    ) -> Result<Vec<Transcript>> {
        let task = self.recognizer.options.task;
        let tokenizer = &self.recognizer.tokenizer;
        let mut transcripts = Vec::with_capacity(recognized.len());
        for chunk in recognized.chunks(self.recognizer.plan.alignment_batch) {
            let inputs: Vec<_> = chunk
                .iter()
                .map(|window| WhisperAlignmentInput {
                    cross_k: &window.cross_k,
                    cross_v: &window.cross_v,
                    decoded_tokens: &window.result.tokens,
                    token_probs: &window.result.token_probs,
                    language: window.result.language.as_deref(),
                    task,
                    audio_samples: window.audio_samples,
                    segments: &window.segments,
                })
                .collect();
            let (words, alignment) =
                self.aligner.align_batch_profiled(&inputs, tokenizer, copies, last_speech_timestamp)?;
            if let Some(profile) = profile {
                profile.push(alignment.graph.stage("alignment_graph"));
                profile.push(StageProfile::host("alignment_cpu_dtw", alignment.cpu_dtw_wall));
            }
            transcripts.extend(chunk.iter().zip(words).map(|(window, words)| window.transcript(words)));
        }
        Ok(transcripts)
    }
}

impl Transcriber for WhisperAlignedTranscriber {
    type Error = TranscribeError;

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE as u32
    }

    /// Whisper ends a window at its last emitted timestamp, which routinely
    /// lands short of the window edge, so the next window starts there.
    fn window_advance(&self) -> Option<WindowAdvance> {
        Some(WindowAdvance::default())
    }

    fn transcribe_windows(
        &mut self,
        windows: &[&[f32]],
        opts: RunOptions,
    ) -> Result<(Vec<Transcript>, Option<RunProfile>)> {
        let mut transcripts = Vec::with_capacity(windows.len());
        let mut run = opts.profile.then(RunProfile::default);
        // `add_word_timestamps` carries the last spoken timestamp from window
        // to window for its pause clamp; a new recording starts from zero.
        let mut last_speech_timestamp = 0.0;
        for batch in windows.chunks(self.recognizer.max_batch) {
            let (recognized, mut batch_profile, mut copies) = self.recognizer.recognize_windows(batch, opts.profile)?;
            if opts.words {
                transcripts.extend(self.align_recognized(
                    &recognized,
                    &mut copies,
                    &mut batch_profile,
                    &mut last_speech_timestamp,
                )?);
            } else {
                transcripts.extend(recognized.iter().map(|window| window.transcript(Vec::new())));
            }
            fold_profile(&mut run, batch_profile, &copies);
        }
        Ok((transcripts, run))
    }
}
