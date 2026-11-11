//! Scheduled Whisper decoding, temperature fallback, and language detection.

use std::cmp::Ordering;
use std::collections::VecDeque;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use svod_arch::pipelines::audio::Segment;
use svod_device::{Buffer, BufferSpec};
use svod_runtime::StageProfile;

use super::config::TOKENS_PER_SECOND;
use super::error::{Error, Result};
use super::jit::{WhisperDecoderStepJit, WhisperPrefillJit};
use super::profile::{CopyProfile, GraphProfile};
use super::tokenizer::WhisperTokenizer;
use super::vocab::{logsumexp, scaled_exp, top_k_logprobs};

// ─── Language detection ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct LanguageDetection {
    pub language: String,
    pub language_token: u32,
    pub probabilities: Vec<(String, f32)>,
}

/// Detect the spoken language from the prefill graph's SOT-conditioned logits.
pub fn detect_language(
    prefill_jit: &mut WhisperPrefillJit,
    n_vocab: usize,
    tokenizer: &WhisperTokenizer,
) -> Result<LanguageDetection> {
    detect_language_profile(prefill_jit, n_vocab, tokenizer, &mut CopyProfile::default(), &mut GraphProfile::default())
}

pub(crate) fn detect_language_profile(
    prefill_jit: &mut WhisperPrefillJit,
    n_vocab: usize,
    tokenizer: &WhisperTokenizer,
    copies: &mut CopyProfile,
    graph: &mut GraphProfile,
) -> Result<LanguageDetection> {
    // Row 0 of the prefill logits is conditioned on SOT alone, so the tokens
    // after it may be anything; SOT again keeps the prompt well-formed.
    let prompt_len = prefill_jit.tokens_mut()?.size() / size_of::<i32>();
    write_prefill_tokens(prefill_jit, &vec![tokenizer.sot() as i32; prompt_len], copies, "language_tokens")?;
    run_prefill_graph(prefill_jit, graph)?;
    let logits = read_prefill_logits(prefill_jit, n_vocab, copies, "language_logits")?;

    let tokens = tokenizer.all_language_tokens();
    let scores: Vec<f32> = tokens.iter().map(|&token| logits[token as usize]).collect();
    let Some(&language_token) = tokens.get(argmax(&scores)) else {
        return Err(decode_err("model has no language tokens"));
    };
    let logsum = logsumexp(&scores);
    let mut probabilities: Vec<(String, f32)> = tokenizer
        .all_language_codes()
        .into_iter()
        .zip(&scores)
        .map(|(code, &score)| (code, (score - logsum).exp()))
        .collect();
    probabilities.sort_by(|a, b| b.1.total_cmp(&a.1));
    let language = tokenizer.code_for_token(language_token).unwrap_or_else(|| "en".into());
    Ok(LanguageDetection { language, language_token, probabilities })
}

// ─── Decode options & result ────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WhisperTask {
    Transcribe,
    Translate,
}

impl std::str::FromStr for WhisperTask {
    type Err = &'static str;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "transcribe" => Ok(Self::Transcribe),
            "translate" => Ok(Self::Translate),
            _ => Err("expected `transcribe` or `translate`"),
        }
    }
}

/// Search algorithm for the first decode attempt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecodeStrategy {
    /// Deterministic token-by-token argmax.
    Greedy,
    /// Beam search with a concrete number of decoder rows.
    Beam { size: usize },
    /// Multinomial sampling at a positive temperature.
    Sample { temperature: f32 },
}

impl DecodeStrategy {
    fn temperature(self) -> f32 {
        match self {
            Self::Greedy | Self::Beam { .. } => 0.0,
            Self::Sample { temperature } => temperature,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecodeOptions {
    /// Whether to transcribe source speech or translate it to English.
    pub task: WhisperTask,
    /// Source language code, or `None` for automatic detection.
    pub language: Option<String>,
    /// Search algorithm for the first decode attempt.
    pub strategy: DecodeStrategy,
    /// Sampling temperatures retried in order when an attempt fails a quality
    /// gate; empty disables fallback.
    pub fallback_temperatures: Vec<f32>,
    /// Retry when the text's zlib compression ratio exceeds this: repetition.
    pub compression_ratio_threshold: Option<f32>,
    /// Retry below this average log-probability. A window above it is never
    /// skipped as silence, whatever its no-speech probability.
    pub logprob_threshold: Option<f32>,
    /// Base seed for reproducible per-request sampling streams.
    pub sampling_seed: Option<u64>,
    /// Maximum generated token count; defaults to half the text context.
    pub sample_len: Option<usize>,
    /// Suppress blank/space as the first generated token.
    pub suppress_blank: bool,
    /// Token IDs to suppress; `-1` expands to Whisper's non-speech set.
    pub suppress_tokens: Option<Vec<i32>>,
    /// Latest timestamp permitted at the beginning of a window, in seconds.
    pub max_initial_timestamp: Option<f32>,
    /// Skip likely silence when no-speech probability exceeds this threshold.
    pub no_speech_threshold: Option<f32>,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            task: WhisperTask::Transcribe,
            language: None,
            strategy: DecodeStrategy::Beam { size: 5 },
            fallback_temperatures: vec![0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold: Some(2.4),
            logprob_threshold: Some(-1.0),
            sampling_seed: None,
            sample_len: None,
            suppress_blank: true,
            suppress_tokens: Some(vec![-1]),
            max_initial_timestamp: Some(1.0),
            no_speech_threshold: Some(0.6),
        }
    }
}

impl DecodeOptions {
    /// Validate strategy geometry and sampling parameters before graph preparation.
    pub fn validate(&self) -> Result<()> {
        let valid_temperature = |temperature: f32| temperature.is_finite() && temperature > 0.0;
        let check = |ok: bool, message: &str| if ok { Ok(()) } else { Err(decode_err(message)) };
        check(!matches!(self.strategy, DecodeStrategy::Beam { size: 0 }), "beam size must be non-zero")?;
        check(
            !matches!(self.strategy, DecodeStrategy::Sample { temperature } if !valid_temperature(temperature)),
            "sampling temperature must be finite and positive",
        )?;
        check(
            self.fallback_temperatures.iter().all(|&temperature| valid_temperature(temperature)),
            "fallback sampling temperatures must be finite and positive",
        )?;
        check(
            self.compression_ratio_threshold.is_none_or(|threshold| threshold.is_finite() && threshold > 0.0),
            "compression ratio threshold must be finite and positive",
        )?;
        check(self.logprob_threshold.is_none_or(f32::is_finite), "log-probability threshold must be finite")?;
        check(
            self.no_speech_threshold.is_none_or(|threshold| (0.0..=1.0).contains(&threshold)),
            "no-speech threshold must be between zero and one",
        )
    }
}

#[derive(Clone, Debug)]
pub struct DecodeResult {
    pub tokens: Vec<u32>,
    pub token_probs: Vec<f32>,
    pub text: String,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    /// Sampling temperature of the accepted attempt; zero for greedy or beam.
    pub temperature: f32,
    pub compression_ratio: f32,
    pub language: Option<String>,
}

impl DecodeResult {
    /// Silence: no-speech probability over the threshold, unless the decode
    /// was confident anyway.
    pub fn should_skip(&self, options: &DecodeOptions) -> bool {
        options.no_speech_threshold.is_some_and(|threshold| self.no_speech_prob > threshold)
            && options.logprob_threshold.is_none_or(|threshold| self.avg_logprob <= threshold)
    }

    pub fn clear_speech(&mut self) {
        self.tokens.clear();
        self.token_probs.clear();
        self.text.clear();
    }
}

/// Whether a finished attempt fails a quality gate and the next temperature
/// should be tried. Low confidence over silence is not worth retrying.
pub(crate) fn check_fallback(result: &DecodeResult, options: &DecodeOptions) -> bool {
    let repetitive = options.compression_ratio_threshold.is_some_and(|threshold| result.compression_ratio > threshold);
    let low_confidence = options.logprob_threshold.is_some_and(|threshold| result.avg_logprob < threshold);
    let silence =
        options.no_speech_threshold.is_some_and(|threshold| result.no_speech_prob > threshold) && low_confidence;
    (repetitive || low_confidence) && !silence
}

// ─── Prefill ─────────────────────────────────────────────────────────────────

pub(crate) struct PrefillMetadata {
    pub(crate) initial_tokens: Vec<u32>,
    /// Prompt length: where sampled tokens start, and how many cache positions
    /// prefill filled.
    pub(crate) sample_begin: usize,
    pub(crate) suppress_tokens: Vec<i32>,
    /// The prompt's last logits row: the distribution over the first sampled token.
    pub(crate) logits: Vec<f32>,
    pub(crate) no_speech_prob: f32,
}

/// Immutable output of one window's prefill. Fallback attempts reuse it rather
/// than rerunning prefill; the snapshots stay device-local and are copied into
/// decoder rows when an attempt takes them over.
pub(crate) struct DecodeSeed {
    pub(crate) metadata: PrefillMetadata,
    self_k: Buffer,
    self_v: Buffer,
    pub(crate) cross_k: Buffer,
    pub(crate) cross_v: Buffer,
    /// Bytes of one cached position in the self cache.
    per_pos_bytes: usize,
}

impl DecodeSeed {
    /// Discard the attempt-local self cache and retain the immutable cross cache.
    pub(crate) fn into_cross_kv(self) -> (Buffer, Buffer) {
        (self.cross_k, self.cross_v)
    }
}

pub(crate) fn prefill_decode_seed(
    prefill_jit: &mut WhisperPrefillJit,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    n_text_ctx: usize,
    n_vocab: usize,
    copies: &mut CopyProfile,
    graph: &mut GraphProfile,
) -> Result<DecodeSeed> {
    let metadata = execute_prefill(prefill_jit, tokenizer, options, n_vocab, copies, graph)?;
    if metadata.sample_begin > n_text_ctx {
        return Err(decode_err("prefill prompt exceeds text context"));
    }
    // The graph's outputs are overwritten by the next window, so the seed
    // snapshots them.
    let (self_k, self_v) = (prefill_jit.self_k()?, prefill_jit.self_v()?);
    let (cross_k, cross_v) = (prefill_jit.cross_k()?, prefill_jit.cross_v()?);
    let bytes = self_k.size() + self_v.size() + cross_k.size() + cross_v.size();
    let (self_k, self_v, cross_k, cross_v) = copies.d2d("seed_snapshots", 4, bytes, self_k, || -> Result<_> {
        Ok((
            clone_device_cache(self_k)?,
            clone_device_cache(self_v)?,
            clone_device_cache(cross_k)?,
            clone_device_cache(cross_v)?,
        ))
    })?;
    build_decode_seed(metadata, self_k, self_v, cross_k, cross_v)
}

pub(crate) fn clone_device_cache(src: &Buffer) -> Result<Buffer> {
    let element = src.dtype().bytes();
    if element == 0 || !src.size().is_multiple_of(element) {
        return Err(decode_err("prefill cache is not element-aligned"));
    }
    let mut clone = Buffer::allocate(
        src.allocator_arc(),
        src.dtype(),
        vec![src.size() / element],
        BufferSpec { cpu_access: false, ..BufferSpec::default() },
    )?;
    clone.copy_from(src)?;
    Ok(clone)
}

pub(crate) fn build_decode_seed(
    metadata: PrefillMetadata,
    self_k: Buffer,
    self_v: Buffer,
    cross_k: Buffer,
    cross_v: Buffer,
) -> Result<DecodeSeed> {
    let positions = metadata.sample_begin;
    if positions == 0
        || self_k.size() == 0
        || self_k.size() != self_v.size()
        || !self_k.size().is_multiple_of(positions)
    {
        return Err(decode_err("invalid prefill self-cache geometry"));
    }
    if cross_k.size() == 0 || cross_k.size() != cross_v.size() {
        return Err(decode_err("invalid prefill cross-cache geometry"));
    }
    if [&self_v, &cross_k, &cross_v].iter().any(|cache| !std::ptr::eq(self_k.allocator(), cache.allocator())) {
        return Err(decode_err("prefill caches use different allocators"));
    }
    let per_pos_bytes = self_k.size() / positions;
    Ok(DecodeSeed { metadata, self_k, self_v, cross_k, cross_v, per_pos_bytes })
}

fn execute_prefill(
    prefill_jit: &mut WhisperPrefillJit,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    n_vocab: usize,
    copies: &mut CopyProfile,
    graph: &mut GraphProfile,
) -> Result<PrefillMetadata> {
    let mut initial_tokens = vec![tokenizer.sot()];
    if tokenizer.multilingual {
        let language = options.language.as_ref().ok_or_else(|| decode_err("language required"))?;
        let task = match options.task {
            WhisperTask::Transcribe => tokenizer.transcribe(),
            WhisperTask::Translate => tokenizer.translate(),
        };
        initial_tokens.extend([tokenizer.language_token_for(language).unwrap_or_else(|| tokenizer.sot()), task]);
    }
    let sample_begin = initial_tokens.len();
    let prompt: Vec<i32> = initial_tokens.iter().map(|&token| token as i32).collect();
    write_prefill_tokens(prefill_jit, &prompt, copies, "prefill_tokens")?;
    run_prefill_graph(prefill_jit, graph)?;
    let mut logits = read_prefill_logits(prefill_jit, sample_begin * n_vocab, copies, "prefill_logits")?;
    // No-speech is read at SOT, the first row; the last row seeds sampling.
    let no_speech_prob =
        tokenizer.no_speech().map(|token| softmax_prob(&logits[..n_vocab], token as usize)).unwrap_or(f32::NAN);
    logits.drain(..(sample_begin - 1) * n_vocab);
    Ok(PrefillMetadata {
        initial_tokens,
        sample_begin,
        suppress_tokens: get_suppress_tokens(tokenizer, options),
        logits,
        no_speech_prob,
    })
}

fn write_prefill_tokens(
    jit: &mut WhisperPrefillJit,
    tokens: &[i32],
    copies: &mut CopyProfile,
    name: &'static str,
) -> Result<()> {
    let fence = jit.tokens_mut()?.clone();
    copies.h2d(name, 1, size_of_val(tokens), &fence, || -> Result<()> {
        let mut dst = jit.tokens_host_write::<i32>()?;
        if dst.len() != tokens.len() {
            return Err(decode_err("prompt length differs from the prepared prefill graph"));
        }
        Ok(dst.copy_from(tokens)?)
    })
}

fn run_prefill_graph(jit: &mut WhisperPrefillJit, graph: &mut GraphProfile) -> Result<()> {
    graph.execute(
        jit,
        |jit| -> Result<()> { Ok(jit.execute()?) },
        |jit| {
            let kernels = jit.execute_profiled_static()?;
            jit.logits()?.synchronize()?;
            Ok(kernels)
        },
    )
}

fn read_prefill_logits(
    jit: &WhisperPrefillJit,
    count: usize,
    copies: &mut CopyProfile,
    name: &'static str,
) -> Result<Vec<f32>> {
    let logits = jit.logits()?;
    copies.d2h(name, 1, count * size_of::<f32>(), logits, || read_f32(logits, 0, count))
}

/// Copy `count` floats from element `offset` of a device buffer over the copy
/// engine. Reading a device-resident buffer through its host mapping faults
/// the pages across one at a time; one bulk copy of the same range moves the
/// same data as a single transfer.
pub(crate) fn read_f32(buffer: &Buffer, offset: usize, count: usize) -> Result<Vec<f32>> {
    let mut out = vec![0f32; count];
    let element = size_of::<f32>();
    buffer.view(offset * element, count * element)?.copyout_prefix(bytemuck::cast_slice_mut(&mut out))?;
    Ok(out)
}

// ─── Fixed-slot mixed-strategy scheduler ────────────────────────────────────

pub(crate) fn strategy_width(strategy: DecodeStrategy) -> usize {
    match strategy {
        DecodeStrategy::Beam { size } => size,
        DecodeStrategy::Greedy | DecodeStrategy::Sample { .. } => 1,
    }
}

pub(crate) fn attempt_strategies(options: &DecodeOptions) -> Vec<DecodeStrategy> {
    std::iter::once(options.strategy)
        .chain(options.fallback_temperatures.iter().map(|&temperature| DecodeStrategy::Sample { temperature }))
        .collect()
}

pub(crate) fn collect_ordered<T>(results: Vec<Option<T>>) -> std::result::Result<Vec<T>, &'static str> {
    results.into_iter().map(|result| result.ok_or("missing scheduled result")).collect()
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DecodeScheduleStats {
    pub(crate) dispatches: usize,
    pub(crate) active_row_steps: usize,
    pub(crate) reserved_row_steps: usize,
    pub(crate) capacity_row_steps: usize,
    pub(crate) cache_clone_ops: usize,
    pub(crate) cache_clone_bytes: usize,
    pub(crate) attempts: usize,
    pub(crate) fallback_attempts: usize,
}

impl DecodeScheduleStats {
    /// Attach the counters to a stage. They are plain sums, so per-window
    /// profiles add up when the pipeline merges them; utilization is
    /// `active_row_steps / capacity_row_steps`.
    pub(crate) fn annotate(&self, stage: &mut StageProfile) {
        for (key, value) in [
            ("dispatches", self.dispatches),
            ("active_row_steps", self.active_row_steps),
            ("reserved_row_steps", self.reserved_row_steps),
            ("capacity_row_steps", self.capacity_row_steps),
            ("cache_clone_ops", self.cache_clone_ops),
            ("cache_clone_bytes", self.cache_clone_bytes),
            ("attempts", self.attempts),
            ("fallback_attempts", self.fallback_attempts),
        ] {
            stage.meta.insert(key.into(), value.to_string());
        }
    }
}

/// Small independently-testable allocator enforcing whole-attempt admission.
#[derive(Debug)]
pub(crate) struct SlotAllocator {
    owners: Vec<Option<usize>>,
}

impl SlotAllocator {
    pub(crate) fn new(capacity: usize) -> Self {
        Self { owners: vec![None; capacity] }
    }

    pub(crate) fn reserve(
        &mut self,
        owner: usize,
        width: usize,
    ) -> std::result::Result<Option<Vec<usize>>, &'static str> {
        if width == 0 {
            return Err("attempt width must be non-zero");
        }
        if width > self.owners.len() {
            return Err("decode attempt width exceeds decoder slots");
        }
        let free: Vec<usize> =
            self.owners.iter().enumerate().filter_map(|(row, current)| current.is_none().then_some(row)).collect();
        if free.len() < width {
            return Ok(None);
        }
        let rows = free[..width].to_vec();
        for &row in &rows {
            self.owners[row] = Some(owner);
        }
        Ok(Some(rows))
    }

    pub(crate) fn release(&mut self, owner: usize) {
        for slot in &mut self.owners {
            if *slot == Some(owner) {
                *slot = None;
            }
        }
    }

    fn reserved(&self) -> usize {
        self.owners.iter().filter(|owner| owner.is_some()).count()
    }

    #[cfg(test)]
    pub(crate) fn owners(&self) -> &[Option<usize>] {
        &self.owners
    }
}

/// One request's live search over its reserved rows. Greedy and sampling are
/// beams of width one whose single candidate per step is picked rather than
/// ranked; everything after candidate generation is shared.
struct Attempt {
    strategy_index: usize,
    strategy: DecodeStrategy,
    reserved_rows: Vec<usize>,
    /// Positions cached so far, which is also the position decoded next.
    pos: usize,
    generated: usize,
    active: Vec<BeamHypothesis>,
    /// The row each active hypothesis occupies.
    rows: Vec<usize>,
    finished: Vec<BeamHypothesis>,
    next_logical_id: usize,
}

impl Attempt {
    fn width(&self) -> usize {
        self.reserved_rows.len()
    }

    fn is_done(&self, sample_len: usize) -> bool {
        self.generated >= sample_len || self.active.is_empty() || self.finished.len() >= self.width()
    }

    /// Score one hypothesis's logits row into the candidates this strategy
    /// considers: the top `size + 1` for a beam, one pick otherwise.
    #[allow(clippy::too_many_arguments)]
    fn candidates(
        &self,
        parent_index: usize,
        row: usize,
        logits: &mut [f32],
        seed: &PrefillMetadata,
        tokenizer: &WhisperTokenizer,
        options: &DecodeOptions,
        rng: &mut StdRng,
        out: &mut Vec<BeamCandidate>,
    ) {
        let hypothesis = &self.active[parent_index];
        apply_logit_filters(
            logits,
            tokenizer,
            options,
            &hypothesis.tokens,
            seed.sample_begin,
            self.generated,
            &seed.suppress_tokens,
        );
        let picks = match self.strategy {
            DecodeStrategy::Beam { size } => top_k_logprobs(logits, size + 1),
            strategy => {
                let token = pick_token_with_rng(logits, strategy.temperature(), rng) as usize;
                vec![(token, logits[token] - logsumexp(logits))]
            }
        };
        out.extend(picks.into_iter().map(|(token, logprob)| BeamCandidate {
            parent_index,
            parent_logical_id: hypothesis.logical_id,
            parent_row: row,
            token_id: token as u32,
            token_logprob: logprob,
            sum_logprob: hypothesis.sum_logprob + logprob,
        }));
    }

    /// Rank the candidates, retire the finished, and assign survivors to rows.
    fn advance(&mut self, candidates: Vec<BeamCandidate>, eot: u32) -> Result<RowAssignment> {
        let width = self.width();
        let (active, finished, survivors) = select_beam_candidates(
            &self.active,
            candidates,
            width,
            eot,
            width - self.finished.len(),
            &mut self.next_logical_id,
        );
        self.finished.extend(finished);
        let assignment = plan_beam_rows(&self.reserved_rows, &survivors).map_err(decode_err)?;
        self.active = active;
        self.rows.clone_from(&assignment.rows);
        self.generated += 1;
        Ok(assignment)
    }
}

#[allow(clippy::too_many_arguments)]
fn start_attempt(
    strategy_index: usize,
    strategy: DecodeStrategy,
    rows: Vec<usize>,
    seed: &DecodeSeed,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    rng: &mut StdRng,
    sample_len: usize,
) -> Result<Attempt> {
    let metadata = &seed.metadata;
    let root = BeamHypothesis {
        logical_id: 0,
        tokens: metadata.initial_tokens.clone(),
        token_probs: Vec::new(),
        sum_logprob: 0.0,
    };
    let mut attempt = Attempt {
        strategy_index,
        strategy,
        rows: vec![rows[0]],
        reserved_rows: rows,
        pos: metadata.sample_begin,
        generated: 0,
        active: vec![root],
        finished: Vec::new(),
        next_logical_id: 1,
    };
    if sample_len == 0 {
        return Ok(attempt);
    }
    // The prompt's logits are the first step. Every reserved row holds the
    // prefill cache, so the children it fans out to need no cache copies.
    let mut logits = metadata.logits.clone();
    let mut candidates = Vec::new();
    attempt.candidates(0, attempt.rows[0], &mut logits, metadata, tokenizer, options, rng, &mut candidates);
    attempt.advance(candidates, tokenizer.eot())?;
    Ok(attempt)
}

fn seed_attempt_rows(
    jit: &mut WhisperDecoderStepJit,
    rows: &[usize],
    cross_slot: usize,
    seed: &DecodeSeed,
    n_text_ctx: usize,
) -> Result<()> {
    // Every row of this attempt decodes the same window, so its cross-attention
    // cache is the same bytes: one copy in the attempt's own cross slot, with
    // every row pointed at it. The slot is the request index, not a decoder
    // row -- there is one live attempt per request, so no two live attempts
    // can name the same slot.
    let cross_stride = seed.cross_k.size();
    copy_device_cache_row(jit.cross_k_mut()?, cross_slot, cross_stride, &seed.cross_k)?;
    copy_device_cache_row(jit.cross_v_mut()?, cross_slot, cross_stride, &seed.cross_v)?;
    let self_stride = n_text_ctx * seed.per_pos_bytes;
    for &row in rows {
        copy_device_cache_row(jit.self_k_cache_mut()?, row, self_stride, &seed.self_k)?;
        copy_device_cache_row(jit.self_v_cache_mut()?, row, self_stride, &seed.self_v)?;
    }
    let slot = i32::try_from(cross_slot).map_err(|_| decode_err("cross cache row exceeds i32"))?;
    let rows: Vec<_> = rows.iter().map(|&row| (row, slot)).collect();
    Ok(jit.cross_cache_map_host_write::<i32>()?.rows(&rows)?)
}

/// Append the step's K/V output for `row` at cache position `pos`.
fn append_row_cache(
    jit: &mut WhisperDecoderStepJit,
    row: usize,
    pos: usize,
    per_pos_bytes: usize,
    row_stride_bytes: usize,
) -> Result<()> {
    let dst = row * row_stride_bytes + pos * per_pos_bytes;
    let src = row * per_pos_bytes;
    jit.copy_output_to_self_k_cache(1, dst, src, per_pos_bytes)?;
    Ok(jit.copy_output_to_self_v_cache(2, dst, src, per_pos_bytes)?)
}

fn clone_cache_prefix(
    jit: &mut WhisperDecoderStepJit,
    copies: &[CacheCopy],
    positions: usize,
    per_pos_bytes: usize,
    row_stride_bytes: usize,
) -> Result<()> {
    let len = positions * per_pos_bytes;
    for copy in copies {
        let (src, dst) = (copy.src_row * row_stride_bytes, copy.dst_row * row_stride_bytes);
        jit.self_k_cache_mut()?.copy_within(dst, src, len)?;
        jit.self_v_cache_mut()?.copy_within(dst, src, len)?;
    }
    Ok(())
}

fn finish_attempt(
    attempt: Attempt,
    seed: &DecodeSeed,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
) -> Result<DecodeResult> {
    let (eot, sample_begin) = (tokenizer.eot(), seed.metadata.sample_begin);
    let width = attempt.width();
    let best = finalize_beam_hypotheses(attempt.active, attempt.finished, width, eot, sample_begin)
        .ok_or_else(|| decode_err("attempt produced no hypothesis"))?;
    let tokens: Vec<u32> = best.tokens[sample_begin..].iter().copied().take_while(|&token| token != eot).collect();
    let token_probs: Vec<f32> = best.token_probs.into_iter().take(tokens.len()).collect();
    finish_decode(&tokens, &token_probs, tokenizer, best.sum_logprob, seed.metadata.no_speech_prob, options)
}

/// Decode all requests through one concrete `[decoder_slots, ...]` step graph.
/// Attempts reserve their full width atomically and retain every reserved row,
/// including inactive beam rows, until quality acceptance or fallback requeue.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_fixed_slot_decode(
    seeds: &[DecodeSeed],
    request_options: &[DecodeOptions],
    step_jit: &mut WhisperDecoderStepJit,
    capacity: usize,
    tokenizer: &WhisperTokenizer,
    n_text_ctx: usize,
    n_vocab: usize,
    copies: &mut CopyProfile,
    graph: &mut GraphProfile,
) -> Result<(Vec<DecodeResult>, DecodeScheduleStats)> {
    if seeds.len() != request_options.len() {
        return Err(decode_err("decode seed/options count mismatch"));
    }
    for options in request_options {
        options.validate()?;
        if attempt_strategies(options).into_iter().any(|strategy| strategy_width(strategy) > capacity) {
            return Err(decode_err("decode attempt width exceeds decoder slots"));
        }
    }

    let eot = tokenizer.eot();
    let strategies: Vec<_> = request_options.iter().map(attempt_strategies).collect();
    let sample_lens: Vec<usize> =
        request_options.iter().map(|options| options.sample_len.unwrap_or(n_text_ctx / 2)).collect();
    let mut queue: VecDeque<_> = (0..seeds.len()).map(|request| (request, 0usize)).collect();
    let mut allocator = SlotAllocator::new(capacity);
    let mut attempts: Vec<Option<Attempt>> = (0..seeds.len()).map(|_| None).collect();
    let mut results: Vec<Option<DecodeResult>> = (0..seeds.len()).map(|_| None).collect();
    let mut stats = DecodeScheduleStats::default();
    let mut rngs: Vec<_> =
        request_options.iter().enumerate().map(|(request, options)| sampling_rng(options, request)).collect();

    while results.iter().any(Option::is_none) {
        while let Some(&(request, strategy_index)) = queue.front() {
            let strategy = strategies[request][strategy_index];
            let Some(rows) = allocator.reserve(request, strategy_width(strategy)).map_err(decode_err)? else {
                break;
            };
            queue.pop_front();
            let mut options = request_options[request].clone();
            options.strategy = strategy;
            let seed = &seeds[request];
            let attempt = start_attempt(
                strategy_index,
                strategy,
                rows,
                seed,
                tokenizer,
                &options,
                &mut rngs[request],
                sample_lens[request],
            )?;
            let fence = step_jit.self_k_cache_mut()?.clone();
            let bytes = attempt.width() * seed.self_k.size() * 2 + seed.cross_k.size() * 2;
            copies.d2d("scheduler_seeding", attempt.width() * 2 + 2, bytes, &fence, || -> Result<()> {
                seed_attempt_rows(step_jit, &attempt.reserved_rows, request, seed, n_text_ctx)
            })?;
            attempts[request] = Some(attempt);
            stats.attempts += 1;
            stats.fallback_attempts += usize::from(strategy_index > 0);
        }

        let active_requests: Vec<usize> =
            attempts.iter().enumerate().filter_map(|(request, attempt)| attempt.as_ref().map(|_| request)).collect();
        if active_requests.is_empty() {
            return Err(decode_err("fixed-slot scheduler made no progress"));
        }

        // Controls: the token each live row decodes and its position, one
        // host write per buffer. Attempts done from the prompt alone need none.
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        for &request in &active_requests {
            let attempt = attempts[request].as_ref().expect("active attempt");
            if attempt.is_done(sample_lens[request]) {
                continue;
            }
            let pos = i32::try_from(attempt.pos).map_err(|_| decode_err("decoder position exceeds i32"))?;
            for (hypothesis, &row) in attempt.active.iter().zip(&attempt.rows) {
                tokens.push((row, *hypothesis.tokens.last().expect("hypotheses start from the prompt") as i32));
                positions.push((row, pos));
            }
        }

        if !tokens.is_empty() {
            let fence = step_jit.token_mut()?.clone();
            copies.h2d("decoder_controls", 2, tokens.len() * 2 * size_of::<i32>(), &fence, || -> Result<()> {
                step_jit.token_host_write::<i32>()?.rows(&tokens)?;
                Ok(step_jit.self_key_lens_host_write::<i32>()?.rows(&positions)?)
            })?;
            stats.dispatches += 1;
            stats.capacity_row_steps += capacity;
            stats.reserved_row_steps += allocator.reserved();
            stats.active_row_steps += tokens.len();
            graph.execute(
                step_jit,
                |jit| -> Result<()> { Ok(jit.execute()?) },
                |jit| {
                    let kernels = jit.execute_profiled_static()?;
                    jit.logits()?.synchronize()?;
                    Ok(kernels)
                },
            )?;

            for &request in &active_requests {
                let attempt = attempts[request].as_mut().expect("active attempt");
                if attempt.is_done(sample_lens[request]) {
                    continue;
                }
                let seed = &seeds[request];
                let (pos, per_pos_bytes) = (attempt.pos, seed.per_pos_bytes);
                let row_stride = n_text_ctx * per_pos_bytes;
                let mut options = request_options[request].clone();
                options.strategy = attempt.strategy;

                let rows = attempt.rows.clone();
                let fence = step_jit.new_self_k()?.clone();
                copies.d2d(
                    "cache_append",
                    rows.len() * 2,
                    rows.len() * per_pos_bytes * 2,
                    &fence,
                    || -> Result<()> {
                        rows.iter().try_for_each(|&row| append_row_cache(step_jit, row, pos, per_pos_bytes, row_stride))
                    },
                )?;
                let fence = step_jit.logits()?.clone();
                let logits = copies.d2h(
                    "decoder_logits",
                    rows.len(),
                    rows.len() * n_vocab * size_of::<f32>(),
                    &fence,
                    || read_logits_rows(step_jit, &rows, n_vocab),
                )?;
                let mut candidates = Vec::new();
                for (parent_index, (&row, mut logits)) in rows.iter().zip(logits).enumerate() {
                    attempt.candidates(
                        parent_index,
                        row,
                        &mut logits,
                        &seed.metadata,
                        tokenizer,
                        &options,
                        &mut rngs[request],
                        &mut candidates,
                    );
                }
                let assignment = attempt.advance(candidates, eot)?;
                attempt.pos += 1;

                let cloned = assignment.copies.len();
                let clone_bytes = cloned * attempt.pos * per_pos_bytes * 2;
                let fence = step_jit.self_k_cache_mut()?.clone();
                copies.d2d("beam_clone", cloned * 2, clone_bytes, &fence, || -> Result<()> {
                    clone_cache_prefix(step_jit, &assignment.copies, attempt.pos, per_pos_bytes, row_stride)
                })?;
                stats.cache_clone_ops += cloned;
                stats.cache_clone_bytes += clone_bytes;
            }
        }

        for request in active_requests {
            let done = attempts[request]
                .as_ref()
                .is_some_and(|attempt| attempt.is_done(sample_lens[request]) || attempt.pos >= n_text_ctx);
            if !done {
                continue;
            }
            let attempt = attempts[request].take().expect("completed attempt");
            let strategy_index = attempt.strategy_index;
            let mut options = request_options[request].clone();
            options.strategy = attempt.strategy;
            let result = finish_attempt(attempt, &seeds[request], tokenizer, &options)?;
            allocator.release(request);
            let retry = strategies[request].get(strategy_index + 1).is_some()
                && check_fallback(&result, &request_options[request]);
            if retry {
                queue.push_back((request, strategy_index + 1));
            } else {
                results[request] = Some(result);
            }
        }
    }

    Ok((collect_ordered(results).map_err(decode_err)?, stats))
}

// ─── Batched JIT buffer row helpers ─────────────────────────────────────────

/// Seed one physical cache row from an immutable device-local snapshot.
pub(crate) fn copy_device_cache_row(
    buf: &mut Buffer,
    row: usize,
    row_stride_bytes: usize,
    data: &Buffer,
) -> Result<()> {
    // A raw region copy, so the two must agree on the element type.
    if buf.dtype() != data.dtype() {
        return Err(decode_err("cache seed and destination row have different dtypes"));
    }
    if !std::ptr::eq(buf.allocator(), data.allocator()) {
        return Err(decode_err("cache seed and decoder row use different allocators"));
    }
    let off = row * row_stride_bytes;
    if off + data.size() > buf.size() || data.size() > row_stride_bytes {
        return Err(decode_err("cache seed row is out of bounds"));
    }
    Ok(buf.copy_region_from(off, data, 0, data.size())?)
}

/// The logits rows of `rows`, fetched as one copy spanning the lowest to the
/// highest: an attempt's rows are adjacent, so that is the same bytes as one
/// transfer per row without the per-transfer latency.
fn read_logits_rows(jit: &WhisperDecoderStepJit, rows: &[usize], n_vocab: usize) -> Result<Vec<Vec<f32>>> {
    let (Some(&first), Some(&last)) = (rows.iter().min(), rows.iter().max()) else {
        return Ok(Vec::new());
    };
    let span = read_f32(jit.logits()?, first * n_vocab, (last + 1 - first) * n_vocab)?;
    Ok(rows.iter().map(|&row| span[(row - first) * n_vocab..(row + 1 - first) * n_vocab].to_vec()).collect())
}

// ─── Cached beam search ─────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BeamHypothesis {
    /// Stable search identity. Decoder rows are deliberately not part of it.
    pub(crate) logical_id: usize,
    pub(crate) tokens: Vec<u32>,
    pub(crate) token_probs: Vec<f32>,
    pub(crate) sum_logprob: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BeamCandidate {
    pub(crate) parent_index: usize,
    pub(crate) parent_logical_id: usize,
    pub(crate) parent_row: usize,
    pub(crate) token_id: u32,
    pub(crate) token_logprob: f32,
    pub(crate) sum_logprob: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BeamSurvivor {
    pub(crate) logical_id: usize,
    pub(crate) parent_row: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CacheCopy {
    pub(crate) src_row: usize,
    pub(crate) dst_row: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RowAssignment {
    /// Destination row for each survivor, in survivor (logical rank) order.
    pub(crate) rows: Vec<usize>,
    pub(crate) copies: Vec<CacheCopy>,
}

/// Assign fixed decoder rows without scratch storage or copy cycles.
///
/// The first selected child of each parent retains that parent's row. Further
/// children use only reserved rows whose old hypotheses are no longer live.
pub(crate) fn plan_beam_rows(
    reserved_rows: &[usize],
    survivors: &[BeamSurvivor],
) -> std::result::Result<RowAssignment, &'static str> {
    let mut unique_reserved = reserved_rows.to_vec();
    unique_reserved.sort_unstable();
    unique_reserved.dedup();
    if unique_reserved.len() != reserved_rows.len() {
        return Err("reserved beam rows must be unique");
    }
    if survivors.len() > reserved_rows.len() {
        return Err("more survivors than reserved beam rows");
    }
    if survivors.iter().any(|survivor| !unique_reserved.contains(&survivor.parent_row)) {
        return Err("survivor parent row is not reserved");
    }

    let mut live_parent_rows = Vec::new();
    for survivor in survivors {
        if !live_parent_rows.contains(&survivor.parent_row) {
            live_parent_rows.push(survivor.parent_row);
        }
    }
    let mut dead_rows = reserved_rows.iter().copied().filter(|row| !live_parent_rows.contains(row));
    let mut retained = Vec::new();
    let mut rows = Vec::with_capacity(survivors.len());
    let mut copies = Vec::new();
    for survivor in survivors {
        if !retained.contains(&survivor.parent_row) {
            retained.push(survivor.parent_row);
            rows.push(survivor.parent_row);
        } else {
            let dst_row = dead_rows.next().ok_or("insufficient inactive rows for duplicate beam children")?;
            rows.push(dst_row);
            copies.push(CacheCopy { src_row: survivor.parent_row, dst_row });
        }
    }
    Ok(RowAssignment { rows, copies })
}

fn candidate_order(a: &BeamCandidate, b: &BeamCandidate) -> Ordering {
    b.sum_logprob
        .total_cmp(&a.sum_logprob)
        .then_with(|| a.parent_logical_id.cmp(&b.parent_logical_id))
        .then_with(|| a.token_id.cmp(&b.token_id))
        .then_with(|| a.parent_index.cmp(&b.parent_index))
}

/// Deterministically rank candidates and split completed from active children.
pub(crate) fn select_beam_candidates(
    parents: &[BeamHypothesis],
    mut candidates: Vec<BeamCandidate>,
    beam_size: usize,
    eot: u32,
    finished_capacity: usize,
    next_logical_id: &mut usize,
) -> (Vec<BeamHypothesis>, Vec<BeamHypothesis>, Vec<BeamSurvivor>) {
    candidates.sort_by(candidate_order);
    let mut active = Vec::with_capacity(beam_size);
    let mut finished = Vec::new();
    let mut survivors = Vec::with_capacity(beam_size);
    for candidate in candidates {
        if active.len() >= beam_size {
            break;
        }
        let Some(parent) = parents.get(candidate.parent_index) else {
            continue;
        };
        let logical_id = *next_logical_id;
        *next_logical_id += 1;
        let mut child = parent.clone();
        child.logical_id = logical_id;
        child.tokens.push(candidate.token_id);
        child.token_probs.push(candidate.token_logprob.exp());
        child.sum_logprob = candidate.sum_logprob;
        if candidate.token_id == eot {
            if finished.len() < finished_capacity {
                finished.push(child);
            }
        } else {
            active.push(child);
            survivors.push(BeamSurvivor { logical_id, parent_row: candidate.parent_row });
        }
    }
    (active, finished, survivors)
}

/// Backfill unfinished hypotheses with EOT and choose the normalized best.
pub(crate) fn finalize_beam_hypotheses(
    active: Vec<BeamHypothesis>,
    mut finished: Vec<BeamHypothesis>,
    beam_size: usize,
    eot: u32,
    sample_begin: usize,
) -> Option<BeamHypothesis> {
    for mut hypothesis in active {
        if finished.len() >= beam_size {
            break;
        }
        if hypothesis.tokens.last().is_none_or(|&token| token != eot) {
            hypothesis.tokens.push(eot);
        }
        finished.push(hypothesis);
    }
    let score = |hypothesis: &BeamHypothesis| {
        hypothesis.sum_logprob / hypothesis.tokens.len().saturating_sub(sample_begin + 1).max(1) as f32
    };
    finished.sort_by(|a, b| score(b).total_cmp(&score(a)).then_with(|| a.logical_id.cmp(&b.logical_id)));
    finished.into_iter().next()
}

// ─── Result helpers ─────────────────────────────────────────────────────────

/// Split a decoded token stream into timestamp-bounded segments.
///
/// The decoder emits paired timestamp tokens (`<|t0|> text <|t1|> text <|t2|>...`)
/// during timestamp-enabled recognition. This function finds
/// consecutive timestamp-token pairs — the boundary between segments — and
/// returns one [`Segment`] per slice, with window-relative start/end times
/// decoded from the timestamp token values.
///
/// Ported from the OpenAI reference (`transcribe.py:339-367`). When no
/// consecutive timestamp pairs are found, returns a single segment spanning
/// the whole token stream.
pub fn split_into_segments(tokens: &[u32], tokenizer: &WhisperTokenizer, window_duration: f32) -> Vec<Segment> {
    let ts_begin = tokenizer.timestamp_begin();
    let is_ts = |t: u32| t >= ts_begin;

    // Find indices where two adjacent tokens are both timestamps — these are
    // segment boundaries (the closing ts of one segment + the opening ts of the
    // next, shared).
    let boundaries: Vec<usize> = (1..tokens.len()).filter(|&i| is_ts(tokens[i - 1]) && is_ts(tokens[i])).collect();

    let mut segments = Vec::new();
    let terminal_timestamp = tokens.last().is_some_and(|&token| is_ts(token))
        && tokens.get(tokens.len().saturating_sub(2)).is_none_or(|&token| !is_ts(token));

    if boundaries.is_empty() {
        // Whisper treats this as one window-relative segment. If any timestamp
        // was emitted, its last value limits the segment duration.
        let start = 0.0;
        let end = tokens
            .iter()
            .rev()
            .find(|&&token| is_ts(token))
            .filter(|&&token| token != ts_begin)
            .map(|&token| token_to_seconds(token, ts_begin))
            .unwrap_or(window_duration)
            .clamp(0.0, window_duration.max(0.0));
        let text = tokenizer.decode(tokens);
        let text = text.trim();
        if !text.is_empty() && end > start {
            segments.push(Segment { text: text.to_string(), start, end });
        }
        return segments;
    }

    let mut last_slice = 0;
    for &boundary in &boundaries {
        if boundary > last_slice {
            segments.push(segment_from_tokens(&tokens[last_slice..boundary], tokenizer, ts_begin, window_duration));
        }
        last_slice = boundary;
    }

    // An unfinished tail is excluded; it will be decoded again from the last
    // completed timestamp boundary by long-form host orchestration.
    if terminal_timestamp && tokens.len() > last_slice {
        segments.push(segment_from_tokens(&tokens[last_slice..], tokenizer, ts_begin, window_duration));
    }

    // Filter empty segments (can happen when consecutive timestamps have no text between them).
    segments.retain(|s| !s.text.is_empty() && s.end > s.start);
    segments
}

/// How far into the window the reference decoder moves its read head after
/// this token stream: to the last completed timestamp pair when the stream
/// ended mid-segment, otherwise past the whole window. A stream ending in a
/// lone timestamp means nothing was spoken after it, and a stream without
/// timestamp pairs is one segment covering the window.
pub fn window_seek(tokens: &[u32], tokenizer: &WhisperTokenizer, window_duration: f32) -> f32 {
    let ts_begin = tokenizer.timestamp_begin();
    let is_ts = |t: u32| t >= ts_begin;
    let lone_ending = tokens.len() >= 2 && !is_ts(tokens[tokens.len() - 2]) && is_ts(tokens[tokens.len() - 1]);
    let last_pair = tokens.windows(2).rposition(|pair| is_ts(pair[0]) && is_ts(pair[1]));
    match last_pair {
        Some(index) if !lone_ending => token_to_seconds(tokens[index], ts_begin).clamp(0.0, window_duration),
        _ => window_duration,
    }
}

/// Decode one timestamp-bounded slice into a [`Segment`].
fn segment_from_tokens(slice: &[u32], tokenizer: &WhisperTokenizer, ts_begin: u32, window_duration: f32) -> Segment {
    let extent = window_duration.max(0.0);
    let start = slice
        .first()
        .filter(|&&t| t >= ts_begin)
        .map(|&t| token_to_seconds(t, ts_begin))
        .unwrap_or(0.0)
        .clamp(0.0, extent);
    let end = slice
        .last()
        .filter(|&&t| t >= ts_begin)
        .map(|&t| token_to_seconds(t, ts_begin))
        .unwrap_or(start)
        .clamp(start, extent);
    let text = tokenizer.decode(slice).trim().to_string();
    Segment { text, start, end }
}

/// Convert a timestamp token id to seconds: `(id - timestamp_begin) / TOKENS_PER_SECOND`.
fn token_to_seconds(token: u32, ts_begin: u32) -> f32 {
    (token - ts_begin) as f32 / TOKENS_PER_SECOND
}

fn finish_decode(
    tokens: &[u32],
    token_probs: &[f32],
    tokenizer: &WhisperTokenizer,
    sum_logprob: f32,
    no_speech_prob: f32,
    options: &DecodeOptions,
) -> Result<DecodeResult> {
    let text = tokenizer.decode(tokens);
    let avg_logprob = sum_logprob / (tokens.len() + 1) as f32;
    let compression_ratio = compression_ratio_text(&text);
    Ok(DecodeResult {
        tokens: tokens.to_vec(),
        token_probs: token_probs.to_vec(),
        text,
        avg_logprob,
        no_speech_prob,
        temperature: options.strategy.temperature(),
        compression_ratio,
        language: options.language.clone(),
    })
}

fn pick_token_with_rng(logits: &[f32], temperature: f32, rng: &mut impl RngExt) -> u32 {
    if temperature > 0.0 { sample_from_logits(logits, temperature, rng) } else { argmax(logits) as u32 }
}

pub(crate) fn derived_sampling_seed(base: u64, request: usize) -> u64 {
    if request == 0 {
        return base;
    }
    let mut value = base ^ (request as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn sampling_rng(options: &DecodeOptions, request: usize) -> StdRng {
    let seed = options.sampling_seed.map(|base| derived_sampling_seed(base, request)).unwrap_or_else(rand::random);
    StdRng::seed_from_u64(seed)
}

pub(crate) fn decode_err(msg: &str) -> Error {
    Error::Decode { msg: msg.into() }
}

// ─── Logit filter helpers ───────────────────────────────────────────────────

/// The tokens suppressed at every step: the caller's list, with `-1` standing
/// for Whisper's non-speech set, plus the prompt specials.
fn get_suppress_tokens(tokenizer: &WhisperTokenizer, options: &DecodeOptions) -> Vec<i32> {
    let mut tokens: Vec<i32> = options.suppress_tokens.clone().unwrap_or_default();
    if tokens.contains(&-1) {
        tokens.retain(|&t| t >= 0);
        tokens.extend(tokenizer.non_speech_tokens().iter().map(|&t| t as i32));
    }
    tokens.extend(
        [tokenizer.transcribe(), tokenizer.translate(), tokenizer.sot(), tokenizer.sot_prev(), tokenizer.sot_lm()]
            .into_iter()
            .chain(tokenizer.no_speech())
            .map(|t| t as i32),
    );
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

fn apply_logit_filters(
    logits: &mut [f32],
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    tokens: &[u32],
    sample_begin: usize,
    step: usize,
    suppress_tokens: &[i32],
) {
    let suppress = |logits: &mut [f32], token: usize| {
        if let Some(logit) = logits.get_mut(token) {
            *logit = f32::NEG_INFINITY;
        }
    };
    if options.suppress_blank && step == 0 {
        for &t in tokenizer.blank_tokens() {
            suppress(logits, t as usize);
        }
        suppress(logits, tokenizer.eot() as usize);
    }
    for &t in suppress_tokens {
        if let Ok(token) = usize::try_from(t) {
            suppress(logits, token);
        }
    }
    apply_timestamp_rules(logits, tokenizer, tokens, sample_begin, options);
}

fn apply_timestamp_rules(
    logits: &mut [f32],
    tokenizer: &WhisperTokenizer,
    tokens: &[u32],
    sample_begin: usize,
    options: &DecodeOptions,
) {
    let ts_begin = tokenizer.timestamp_begin() as usize;
    let eot = tokenizer.eot() as usize;
    let no_ts = tokenizer.no_timestamps() as usize;
    if no_ts < logits.len() {
        logits[no_ts] = f32::NEG_INFINITY;
    }

    let sampled = &tokens[sample_begin.min(tokens.len())..];
    let is_ts = |t: &u32| (*t as usize) >= ts_begin;
    let last_was_ts = sampled.last().is_some_and(is_ts);
    let penultimate_was_ts = sampled.len() < 2 || is_ts(&sampled[sampled.len() - 2]);

    if last_was_ts {
        if penultimate_was_ts {
            logits[ts_begin..].fill(f32::NEG_INFINITY);
        } else {
            logits[..eot].fill(f32::NEG_INFINITY);
        }
    }

    if let Some(&last_ts) = sampled.iter().rev().find(|t| is_ts(t)) {
        // Timestamps never go backwards; after a closing timestamp the next
        // opening one may repeat it.
        let first_allowed = (last_ts as usize + usize::from(!(last_was_ts && !penultimate_was_ts))).min(logits.len());
        logits[ts_begin..first_allowed].fill(f32::NEG_INFINITY);
    }

    if tokens.len() == sample_begin {
        logits[..ts_begin].fill(f32::NEG_INFINITY);
        if let Some(max_init) = options.max_initial_timestamp {
            let last_allowed = ts_begin + (max_init * TOKENS_PER_SECOND).round() as usize;
            if last_allowed + 1 < logits.len() {
                logits[last_allowed + 1..].fill(f32::NEG_INFINITY);
            }
        }
    }

    let ts_logprob = logsumexp(&logits[ts_begin..]);
    let text_max = logits[..ts_begin].iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if ts_logprob > text_max {
        logits[..eot].fill(f32::NEG_INFINITY);
    }
}

// ─── Math helpers ───────────────────────────────────────────────────────────

fn argmax(arr: &[f32]) -> usize {
    arr.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(i, _)| i).unwrap_or(0)
}

fn softmax_prob(logits: &[f32], idx: usize) -> f32 {
    logits.get(idx).map_or(0.0, |&logit| (logit - logsumexp(logits)).exp())
}

/// `len(text) / len(zlib(text))`, the reference's repetition measure; its
/// framing overhead is part of the ratio, so the codec matters.
fn compression_ratio_text(text: &str) -> f32 {
    use std::io::Write;
    let raw = text.as_bytes();
    if raw.is_empty() {
        return 1.0;
    }
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    let _ = encoder.write_all(raw);
    let compressed = encoder.finish().unwrap_or_default().len().max(1);
    raw.len() as f32 / compressed as f32
}

/// Multinomial sampling from logits at temperature T. Matches the OpenAI
/// reference's `Categorical(logits=logits/T).sample()` (`decoding.py:283`):
/// a max-subtracted softmax followed by inverse-CDF sampling.
fn sample_from_logits(logits: &[f32], temperature: f32, rng: &mut impl RngExt) -> u32 {
    let (weights, sum) = scaled_exp(logits, temperature);
    let mut remaining = rng.random::<f32>() * sum;
    weights
        .iter()
        .position(|&weight| {
            remaining -= weight;
            remaining <= 0.0
        })
        .unwrap_or(weights.len().saturating_sub(1)) as u32
}
