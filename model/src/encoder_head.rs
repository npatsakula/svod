//! Shared machinery for the encoder heads (ModernBERT's embedder / classifier /
//! token classifier, XLM-R's token classifier): one head error, the hub
//! options their loaders take, and `HeadPlans`, the per-shape plan cache
//! every head runs its chunks through.
//!
//! The heads are mechanically identical up to (a) the JIT wrapper type, (b)
//! the output-decode stride, and (c) the profile stage label, so the capacity
//! guard → shape pick → pack → execute → read-back prelude lives here and each
//! head keeps just its decode tail.
//!
//! Every plan is static — no batch variable. tk's kernels run a rebindable
//! batch too, but tune their tiles once, for its bound; a static plan tunes
//! them for the rows it runs. A batch therefore runs at the smallest prepared
//! shape that holds it, compiled the first time it is needed:
//!
//! - rows: the batch size rounded up to a power of two, capped at `max_batch`
//!   — one chunk does not pay for `max_batch` rows;
//! - length: the longest chunk rounded up to [`LEN_MULTIPLE`] tokens — short
//!   chunks do not pay for `max_seq`.
//!
//! Each plan holds its own activation buffers, so memory grows with the number
//! of shapes a workload touches.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::time::Instant;

use hf_hub::HFError;
use snafu::{ResultExt, Snafu, ensure};
use svod_arch::pipelines::text::{Encoding, RunProfile};
use svod_dtype::DType;
use svod_runtime::{KernelProfile, StageProfile};
use svod_tensor::PrepareConfig;
use svod_tensor::jit::rt::ArrayViewMutD;

use crate::hub::HubRepo;
use crate::jit::{self, InputSpec};
use crate::state::scoped;

/// Head-level error for the encoder heads. `stage` names the head ("embed",
/// "classify", "classify_tokens") so the message stays task-specific.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum HeadError {
    #[snafu(display("JIT op failed: {source}"))]
    Jit { source: jit::JitError },
    #[snafu(display("{stage} batch of {got} exceeds max_batch {max}"))]
    CapacityExceeded { stage: &'static str, got: usize, max: usize },
    #[snafu(display("{stage} sequence length {got} exceeds max_seq {max}"))]
    SequenceTooLong { stage: &'static str, got: usize, max: usize },
    #[snafu(display("{stage} needs max_batch and max_seq of at least 1, got ({max_batch}, {max_seq})"))]
    EmptyCapacity { stage: &'static str, max_batch: usize, max_seq: usize },
    #[snafu(display("{stage} max_seq {max_seq} exceeds the checkpoint's {max_positions} positions"))]
    BeyondPositions { stage: &'static str, max_seq: usize, max_positions: usize },
}

/// How a hub loader builds its head.
#[derive(Clone, Debug)]
pub struct HubOptions {
    /// Hub revision: a branch, tag or commit.
    pub revision: String,
    /// Most chunks one execute runs.
    pub max_batch: usize,
    /// Longest chunk the head accepts, in tokens (specials included) — size the
    /// pipeline's chunker to it. `None` takes every position the checkpoint
    /// has. A batch runs at its own longest chunk, so this bounds a chunk; it
    /// does not size every run.
    pub max_seq: Option<usize>,
    /// Compute dtype.
    pub dtype: DType,
    /// The repository and revision holding `tokenizer.json`, for checkpoints
    /// that ship none (SaT reuses `FacebookAI/xlm-roberta-base`'s). `None`
    /// reads the model's repository at [`Self::revision`].
    pub tokenizer_repo: Option<(String, String)>,
}

impl Default for HubOptions {
    /// `main`, one chunk per execute, the checkpoint's full length, the default
    /// device's compute dtype, and the model's own tokenizer.
    fn default() -> Self {
        Self {
            revision: "main".into(),
            max_batch: 1,
            max_seq: None,
            dtype: crate::default_compute_dtype(),
            tokenizer_repo: None,
        }
    }
}

impl HubOptions {
    /// `tokenizer.json`, from [`Self::tokenizer_repo`] or else `model`.
    pub(crate) fn tokenizer_json(&self, model: &HubRepo) -> Result<PathBuf, HFError> {
        match &self.tokenizer_repo {
            Some((repo, revision)) => HubRepo::open(repo, revision)?.get("tokenizer.json"),
            None => model.get("tokenizer.json"),
        }
    }
}

/// Build a dense `Vec<String>` from a sparse HF `id2label` map (`config.json`):
/// size to `max(id)+1`, fill gaps (and empty values) with `"LABEL_{id}"`. An
/// absent or empty map yields `vec![]`.
pub(crate) fn dense_id2label(raw: &Option<HashMap<String, String>>) -> Vec<String> {
    let Some(map) = raw else { return Vec::new() };
    let max_id = map.keys().filter_map(|k| k.parse::<usize>().ok()).max();
    let Some(max_id) = max_id else { return Vec::new() };
    let mut out = vec![String::new(); max_id + 1];
    for (k, v) in map {
        if let Ok(id) = k.parse::<usize>()
            && id < out.len()
        {
            out[id] = v.clone();
        }
    }
    for (i, slot) in out.iter_mut().enumerate() {
        if slot.is_empty() {
            *slot = format!("LABEL_{i}");
        }
    }
    out
}

/// The JIT interface the `jit_wrapper!`-generated head types share. The macro
/// emits inherent methods (no trait), so [`HeadPlans`] is bounded on this
/// trait — implemented once per wrapper by [`head_jit!`] as straight
/// delegations.
pub(crate) trait HeadJit: Sized {
    type Model: Clone;
    fn new(model: Self::Model) -> Self;
    fn prepare_with_config(&mut self, ids: InputSpec, mask: InputSpec, config: &PrepareConfig) -> jit::Result<()>;
    fn input_ids_view_mut(&mut self) -> jit::Result<ArrayViewMutD<'_, i64>>;
    fn attention_mask_view_mut(&mut self) -> jit::Result<ArrayViewMutD<'_, i64>>;
    fn execute(&mut self) -> jit::Result<()>;
    fn execute_profiled(&mut self) -> jit::Result<Vec<KernelProfile>>;
    /// The declared output, as f32.
    fn read_output(&self) -> jit::Result<Vec<f32>>;
}

/// `head_jit!(Wrapper, Model, read_output_fn)`: implement [`HeadJit`] for a
/// static `jit_wrapper!` over `input_ids` / `attention_mask`.
macro_rules! head_jit {
    ($jit:ty, $model:ty, $read:ident) => {
        impl $crate::encoder_head::HeadJit for $jit {
            type Model = $model;
            fn new(model: $model) -> Self {
                <$jit>::new(model)
            }
            fn prepare_with_config(
                &mut self,
                ids: $crate::jit::InputSpec,
                mask: $crate::jit::InputSpec,
                config: &::svod_tensor::PrepareConfig,
            ) -> $crate::jit::Result<()> {
                <$jit>::prepare_with_config(self, ids, mask, config)
            }
            fn input_ids_view_mut(&mut self) -> $crate::jit::Result<$crate::jit::rt::ArrayViewMutD<'_, i64>> {
                <$jit>::input_ids_view_mut(self)
            }
            fn attention_mask_view_mut(&mut self) -> $crate::jit::Result<$crate::jit::rt::ArrayViewMutD<'_, i64>> {
                <$jit>::attention_mask_view_mut(self)
            }
            fn execute(&mut self) -> $crate::jit::Result<()> {
                <$jit>::execute(self)
            }
            fn execute_profiled(&mut self) -> $crate::jit::Result<Vec<::svod_runtime::KernelProfile>> {
                <$jit>::execute_profiled(self)
            }
            fn read_output(&self) -> $crate::jit::Result<Vec<f32>> {
                <$jit>::$read::<f32>(self)
            }
        }
    };
}
pub(crate) use head_jit;

/// The multiple a plan's length is rounded up to: one plan per 128 tokens of
/// length, each a whole number of tk's attention tiles (at most 128 by 128).
pub(crate) const LEN_MULTIPLE: usize = 128;

/// The `(rows, len)` plan a batch of `b` chunks, the longest `longest` tokens,
/// runs on (see the module doc).
pub(crate) fn plan_shape(b: usize, longest: usize, max_batch: usize) -> (usize, usize) {
    (b.next_power_of_two().min(max_batch), longest.max(1).next_multiple_of(LEN_MULTIPLE))
}

/// One [`HeadPlans::run`]: the live rows of the plan's output, in batch order,
/// and the sequence length the plan ran at (the per-token heads' row stride).
pub(crate) struct HeadRun {
    pub(crate) output: Vec<f32>,
    pub(crate) len: usize,
    pub(crate) profile: Option<RunProfile>,
}

/// A head's static plans, one per `(rows, length)` shape, each compiled on
/// first use (see the module doc for how a batch picks its shape).
pub(crate) struct HeadPlans<J: HeadJit> {
    model: J::Model,
    max_batch: usize,
    max_seq: usize,
    pad_id: i64,
    stage: &'static str,
    plans: BTreeMap<(usize, usize), J>,
}

impl<J: HeadJit> HeadPlans<J> {
    /// Plans for batches of up to `max_batch` chunks of up to `max_seq` tokens,
    /// padded with `pad_id`. `max_seq` may not exceed `max_positions`, the
    /// longest chunk the checkpoint has positions for.
    pub(crate) fn new(
        model: J::Model,
        max_batch: usize,
        max_seq: usize,
        max_positions: usize,
        pad_id: usize,
        stage: &'static str,
    ) -> Result<Self, HeadError> {
        ensure!(max_batch > 0 && max_seq > 0, EmptyCapacitySnafu { stage, max_batch, max_seq });
        ensure!(max_seq <= max_positions, BeyondPositionsSnafu { stage, max_seq, max_positions });
        let pad_id = i64::try_from(pad_id).expect("a pad id fits the i64 inputs");
        Ok(Self { model, max_batch, max_seq, pad_id, stage, plans: BTreeMap::new() })
    }

    pub(crate) fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.max_seq)
    }

    /// Run `batch` through the plan for its shape, compiling it first if this
    /// is the shape's first batch. Returns the live rows of the output.
    pub(crate) fn run(&mut self, batch: &[&Encoding], profile: bool) -> Result<HeadRun, HeadError> {
        let (b, stage) = (batch.len(), self.stage);
        if b == 0 {
            return Ok(HeadRun { output: Vec::new(), len: 0, profile: profile.then(RunProfile::default) });
        }
        ensure!(b <= self.max_batch, CapacityExceededSnafu { stage, got: b, max: self.max_batch });
        // Chunks built outside the pipeline's chunker (the public chunk seam)
        // can exceed max_seq: catch it rather than silently truncating.
        let longest = batch.iter().map(|e| e.input_ids.len()).max().unwrap_or(0);
        ensure!(longest <= self.max_seq, SequenceTooLongSnafu { stage, got: longest, max: self.max_seq });

        let (rows, len) = plan_shape(b, longest, self.max_batch);
        let jit = match self.plans.entry((rows, len)) {
            std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::btree_map::Entry::Vacant(e) => {
                let mut jit = J::new(self.model.clone());
                let spec = InputSpec::i64(&[rows, len]);
                // The stage name roots every origin path the capture mints.
                scoped(stage, || jit.prepare_with_config(spec.clone(), spec, &PrepareConfig::device_local()))
                    .context(JitSnafu)?;
                e.insert(jit)
            }
        };

        pack_ids(
            jit.input_ids_view_mut().context(JitSnafu)?.as_slice_mut().expect("contiguous ids"),
            batch,
            len,
            self.pad_id,
        );
        pack_mask(
            jit.attention_mask_view_mut().context(JitSnafu)?.as_slice_mut().expect("contiguous mask"),
            batch,
            len,
        );

        let started = Instant::now();
        let profile = if profile {
            let kernels = jit.execute_profiled().context(JitSnafu)?;
            let mut p = RunProfile::default();
            p.push(StageProfile::gpu(stage, started.elapsed(), kernels));
            Some(p)
        } else {
            jit.execute().context(JitSnafu)?;
            None
        };

        let mut output = jit.read_output().context(JitSnafu)?;
        output.truncate(output.len() / rows * b);
        Ok(HeadRun { output, len, profile })
    }
}

// ── input packing ───────────────────────────────────────────────────────────
//
// Pad each chunk's `input_ids` / `attention_mask` into a plan's row-major
// `[rows, len]` i64 inputs.

/// Pad each chunk's `input_ids` into the `[rows, len]` input, filling past each
/// chunk's length and over unused rows with `pad_id`.
pub(crate) fn pack_ids(dst: &mut [i64], batch: &[&Encoding], len: usize, pad_id: i64) {
    pack(dst, batch.iter().map(|enc| enc.input_ids.as_slice()), len, pad_id);
}

/// Pad each chunk's `attention_mask` (1 = real token, 0 = pad) into the
/// `[rows, len]` input. Trailing pad positions stay 0. An unused row attends to
/// its first (pad) token alone: a row whose keys are all masked would run its
/// softmax over nothing.
pub(crate) fn pack_mask(dst: &mut [i64], batch: &[&Encoding], len: usize) {
    pack(dst, batch.iter().map(|enc| enc.attention_mask.as_slice()), len, 0);
    for row in dst.chunks_exact_mut(len).skip(batch.len()) {
        row[0] = 1;
    }
}

fn pack<'a>(dst: &mut [i64], rows: impl Iterator<Item = &'a [u32]>, len: usize, fill: i64) {
    dst.fill(fill);
    for (row, src) in dst.chunks_exact_mut(len).zip(rows) {
        let take = src.len().min(len);
        for (d, &s) in row[..take].iter_mut().zip(&src[..take]) {
            *d = s.into();
        }
    }
}
