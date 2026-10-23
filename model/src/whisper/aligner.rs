//! Fixed-shape teacher-forced decoder alignment and host-side DTW.

use std::time::{Duration, Instant};

use svod_arch::pipelines::audio::Segment;
use svod_device::Buffer;
use svod_dtype::DType;

use crate::jit::InputSpec;

use super::config::{HOP_LENGTH, ModelDimensions, N_AUDIO_CTX, N_FRAMES, TOKENS_PER_SECOND, WhisperSize};
use super::decode::{WhisperTask, decode_err, read_f32};
use super::dtw::{find_alignment_path_selected, path_to_word_timings_in_segments};
use super::error::Result;
use super::jit::{WhisperAlignmentJit, WhisperAlignmentModel};
use super::model::Whisper;
use super::profile::{CopyProfile, GraphProfile};
use super::tokenizer::WhisperTokenizer;
use super::transcribe::Word;

/// Prepared alignment stage. Its graph is fully static and is replayed once
/// for each finalized recognition result.
pub struct WhisperAligner {
    jit: WhisperAlignmentJit,
    n_heads: usize,
    batch_size: usize,
    /// Token positions the graph is prepared for: prompt, `<|notimestamps|>`,
    /// text, EOT.
    text_ctx: usize,
    cache_dtype: DType,
    cache_bytes: usize,
}

/// Inputs for one lane of a prepared alignment batch.
pub struct WhisperAlignmentInput<'a> {
    /// Device-resident packed cross-attention K cache for one recognition window.
    pub cross_k: &'a Buffer,
    /// Device-resident packed cross-attention V cache for one recognition window.
    pub cross_v: &'a Buffer,
    /// Recognition tokens, including timestamp tokens when emitted.
    pub decoded_tokens: &'a [u32],
    /// Decoder probability corresponding to each decoded token.
    pub token_probs: &'a [f32],
    /// Resolved language code used to reconstruct the decoder prompt.
    pub language: Option<&'a str>,
    /// Decoder task used to reconstruct the prompt.
    pub task: WhisperTask,
    /// Unpadded source-audio length in samples.
    pub audio_samples: usize,
    /// Recogniser timestamp segments for this window, window-relative. Their
    /// first start and last end repair the aligned edge words.
    pub segments: &'a [Segment],
}

#[derive(Debug, Default)]
pub(crate) struct AlignmentProfile {
    pub(crate) graph: GraphProfile,
    pub(crate) cpu_dtw_wall: Duration,
}

/// One lane's replayed prompt and the text it aligns.
struct Lane {
    text: Vec<u32>,
    token_probs: Vec<f32>,
    valid_text: usize,
    prompt_len: usize,
}

/// Rows a tensor-core tile covers. The relaxed tensor-core level does not pad,
/// so a replay whose row count is not a multiple of it (229 was) lowers every
/// projection to the scalar path, an order of magnitude behind the encoder's
/// tiles on the same weights.
const TC_ROWS: usize = 16;

/// Token positions the replay is prepared for: `max_tokens` of text after a
/// `prompt_len` prompt, with `<|notimestamps|>` and EOT, in whole tensor-core
/// tiles and never past the model's `n_text_ctx`.
pub(crate) fn replay_rows(max_tokens: usize, prompt_len: usize, n_text_ctx: usize) -> usize {
    (max_tokens + prompt_len + 2).next_multiple_of(TC_ROWS).min(n_text_ctx)
}

impl WhisperAligner {
    /// `max_tokens` bounds the text tokens one window hands over; the graph is
    /// sized to that plus the prompt and terminators, rounded up to whole
    /// tensor-core tiles, never past the model's context.
    pub fn new(model: Whisper, size: WhisperSize, batch_size: usize, max_tokens: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(decode_err("alignment batch must be non-zero"));
        }
        let heads = size.alignment_heads().to_vec();
        let dims = &model.dims;
        let expected = ModelDimensions::for_size(size);
        if dims.n_mels != expected.n_mels
            || dims.n_vocab != expected.n_vocab
            || (dims.n_audio_state, dims.n_audio_head, dims.n_audio_layer)
                != (expected.n_audio_state, expected.n_audio_head, expected.n_audio_layer)
            || (dims.n_text_state, dims.n_text_head, dims.n_text_layer)
                != (expected.n_text_state, expected.n_text_head, expected.n_text_layer)
        {
            let msg = format!("whisper size {} does not match the model dimensions", size.name());
            return Err(decode_err(&msg));
        }
        let (layer_heads, d_head) = (dims.n_text_layer * dims.n_text_head, dims.n_text_state / dims.n_text_head);
        let prompt_len = if dims.is_multilingual() { 3 } else { 1 };
        let text_ctx = replay_rows(max_tokens, prompt_len, dims.n_text_ctx);
        let cache_dtype = dims.cache_dtype();
        let cache_bytes = N_AUDIO_CTX * layer_heads * d_head * cache_dtype.bytes();
        let cache_spec =
            InputSpec::new(&[batch_size, N_AUDIO_CTX, layer_heads, d_head], cache_dtype.clone()).device_local();
        let mut jit = WhisperAlignmentJit::new(WhisperAlignmentModel::new(model, heads.clone()));
        // The attention output is read back with one copyout; see the
        // recogniser's `prepare_config` for why the host mapping is avoided.
        jit.prepare_with_config(
            cache_spec.clone(),
            cache_spec,
            InputSpec::i32(&[batch_size, text_ctx]),
            &svod_tensor::PrepareConfig::device_local(),
        )?;
        Ok(Self { jit, n_heads: heads.len(), batch_size, text_ctx, cache_dtype, cache_bytes })
    }

    /// Align up to the concrete batch capacity prepared at construction. The
    /// window order is the audio order, so the pause clamp carries from one
    /// input to the next.
    pub fn align_batch(
        &mut self,
        inputs: &[WhisperAlignmentInput<'_>],
        tokenizer: &WhisperTokenizer,
    ) -> Result<Vec<Vec<Word>>> {
        let mut last_speech_timestamp = 0.0;
        self.align_batch_profiled(inputs, tokenizer, &mut CopyProfile::default(), &mut last_speech_timestamp)
            .map(|(words, _)| words)
    }

    pub(crate) fn align_batch_profiled(
        &mut self,
        inputs: &[WhisperAlignmentInput<'_>],
        tokenizer: &WhisperTokenizer,
        copies: &mut CopyProfile,
        last_speech_timestamp: &mut f32,
    ) -> Result<(Vec<Vec<Word>>, AlignmentProfile)> {
        if inputs.len() > self.batch_size {
            let msg = format!("alignment input {} exceeds prepared batch {}", inputs.len(), self.batch_size);
            return Err(decode_err(&msg));
        }
        if inputs.is_empty() {
            return Ok((Vec::new(), AlignmentProfile::default()));
        }

        let (cache_bytes, text_ctx) = (self.cache_bytes, self.text_ctx);
        let cache_dtype = &self.cache_dtype;
        let jit = &mut self.jit;
        let fits = |cache: &Buffer, packed: &Buffer| {
            cache.dtype() == *cache_dtype
                && cache.size() == cache_bytes
                && std::ptr::eq(packed.allocator(), cache.allocator())
        };
        copies.d2d(
            "alignment_packing",
            inputs.len() * 2,
            inputs.len() * cache_bytes * 2,
            inputs[0].cross_k,
            || -> Result<()> {
                let pack = |packed: &mut Buffer, cache: &Buffer, lane: usize| {
                    if !fits(cache, packed) {
                        return Err(decode_err("alignment cross cache has invalid dtype, size, or allocator"));
                    }
                    Ok(packed.copy_region_from(lane * cache_bytes, cache, 0, cache_bytes)?)
                };
                for (lane, input) in inputs.iter().enumerate() {
                    pack(jit.cross_k_mut()?, input.cross_k, lane)?;
                    pack(jit.cross_v_mut()?, input.cross_v, lane)?;
                }
                Ok(())
            },
        )?;

        let mut packed_tokens = vec![tokenizer.eot() as i32; self.batch_size * text_ctx];
        let mut lanes = Vec::with_capacity(inputs.len());
        for (lane, input) in inputs.iter().enumerate() {
            let mut tokens = vec![tokenizer.sot()];
            if tokenizer.multilingual {
                let language = input.language.unwrap_or("en");
                tokens.push(tokenizer.language_token_for(language).unwrap_or_else(|| tokenizer.sot()));
                tokens.push(match input.task {
                    WhisperTask::Transcribe => tokenizer.transcribe(),
                    WhisperTask::Translate => tokenizer.translate(),
                });
            }
            let prompt_len = tokens.len();
            tokens.push(tokenizer.no_timestamps());
            // Text the replay holds after the prompt, `<|notimestamps|>`, and
            // EOT. A longer lane is refused rather than silently truncated.
            let capacity = text_ctx.saturating_sub(prompt_len + 2);
            let (text, token_probs) =
                lane_text_tokens(input.decoded_tokens, input.token_probs, tokenizer.eot(), capacity)?;
            tokens.extend_from_slice(&text);
            tokens.push(tokenizer.eot());
            for (index, token) in tokens.iter().enumerate() {
                packed_tokens[lane * text_ctx + index] = *token as i32;
            }
            lanes.push(Lane { text, token_probs, valid_text: tokens.len(), prompt_len });
        }

        let bytes: &[u8] = bytemuck::cast_slice(&packed_tokens);
        let fence = jit.tokens_mut()?.clone();
        copies.h2d("alignment_tokens", 1, bytes.len(), &fence, || -> Result<()> {
            jit.tokens_mut()?.as_host_bytes_mut()?.copy_from_slice(bytes);
            Ok(())
        })?;
        let mut graph = GraphProfile::new(copies.enabled());
        graph.execute(
            jit,
            |jit| -> Result<()> { Ok(jit.execute()?) },
            |jit| {
                let kernels = jit.execute_profiled_static()?;
                jit.output()?.synchronize()?;
                Ok(kernels)
            },
        )?;
        let qk_stride = self.n_heads * text_ctx * N_AUDIO_CTX;
        let output = jit.output()?;
        let count = inputs.len() * qk_stride;
        let qk = copies.d2h("alignment_qk", 1, count * size_of::<f32>(), output, || read_f32(output, 0, count))?;

        let cpu_started = Instant::now();
        let mut words = Vec::with_capacity(inputs.len());
        for (lane, (input, meta)) in inputs.iter().zip(lanes).enumerate() {
            let lane_qk = &qk[lane * qk_stride..(lane + 1) * qk_stride];
            let valid_audio = (input.audio_samples / HOP_LENGTH).min(N_FRAMES) / 2;
            let (text_indices, time_indices) = find_alignment_path_selected(
                lane_qk,
                self.n_heads,
                text_ctx,
                N_AUDIO_CTX,
                meta.valid_text,
                valid_audio,
                7,
                meta.prompt_len,
            );
            let (lane_words, carried) = words_from_path(
                &text_indices,
                &time_indices,
                &meta.text,
                &meta.token_probs,
                input.language,
                tokenizer,
                input.segments,
                *last_speech_timestamp,
            );
            *last_speech_timestamp = carried;
            words.push(lane_words);
        }
        Ok((words, AlignmentProfile { graph, cpu_dtw_wall: cpu_started.elapsed() }))
    }
}

/// One lane's aligned text: decoder tokens below EOT, each paired with the
/// probability the decoder gave that same token. Timestamp and EOT tokens are
/// dropped, so their probabilities must be dropped with them. `capacity` is
/// the text the prepared replay holds; a longer lane is an error rather than a
/// silent truncation.
pub(crate) fn lane_text_tokens(
    decoded: &[u32],
    token_probs: &[f32],
    eot: u32,
    capacity: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let mut tokens = Vec::new();
    let mut probabilities = Vec::new();
    for (index, &token) in decoded.iter().enumerate() {
        if token >= eot {
            continue;
        }
        if tokens.len() == capacity {
            let msg = format!("alignment text exceeds the prepared {capacity}-token context");
            return Err(decode_err(&msg));
        }
        tokens.push(token);
        probabilities.push(token_probs.get(index).copied().unwrap_or(0.0));
    }
    Ok((tokens, probabilities))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn words_from_path(
    text_indices: &[usize],
    time_indices: &[usize],
    text_tokens: &[u32],
    token_probs: &[f32],
    language: Option<&str>,
    tokenizer: &WhisperTokenizer,
    segments: &[Segment],
    last_speech_timestamp: f32,
) -> (Vec<Word>, f32) {
    let (word_strings, word_token_lists) = tokenizer.split_to_word_tokens_for_language(text_tokens, language);
    let mut word_boundaries = vec![0usize];
    for tokens in &word_token_lists {
        word_boundaries.push(word_boundaries.last().copied().unwrap() + tokens.len());
    }
    let (timings, last_speech_timestamp) = path_to_word_timings_in_segments(
        text_indices,
        time_indices,
        &word_boundaries,
        &word_strings,
        &word_token_lists,
        token_probs,
        TOKENS_PER_SECOND,
        segments,
        last_speech_timestamp,
    );
    let words = timings
        .into_iter()
        .filter(|word| !word.word.trim().is_empty())
        .map(|word| Word { text: word.word, start: word.start, end: word.end })
        .collect();
    (words, last_speech_timestamp)
}
