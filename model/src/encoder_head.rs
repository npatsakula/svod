//! Shared machinery for the encoder heads (ModernBERT's embedder / classifier /
//! token classifier, XLM-R's token classifier): one head error, the options
//! their loaders take, and `HeadPlans`, the per-shape plan cache every head
//! runs its chunks through.
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
//!
//! A head records the device it was built for and compiles every later plan
//! there, whichever thread runs it: the default device is thread-local, and a
//! serving pipeline builds a head on one thread and runs it on another.
//! [`HeadPlans::warm_up`] compiles a shape ahead of its first batch.
//!
//! The embedders pool as a checkpoint's sentence-transformers config says
//! ([`Pooling`]).

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::time::Instant;

use hf_hub::HFError;
use serde::Deserialize;
use snafu::{ResultExt, Snafu, ensure};
use svod_arch::pipelines::text::{Encoding, RunProfile};
use svod_dtype::default_device::{default_device, with_default_device};
use svod_dtype::{DType, DeviceSpec};
use svod_runtime::{KernelProfile, StageProfile};
use svod_tensor::jit::rt::ArrayViewMutD;
use svod_tensor::{PrepareConfig, Tensor};

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

/// How a head is built: its capacity, compute dtype and device.
#[derive(Clone, Debug)]
pub struct HeadOptions {
    /// Most chunks one execute runs.
    pub max_batch: usize,
    /// Longest chunk the head accepts, in tokens (specials included) — size the
    /// pipeline's chunker to it. `None` takes every position the checkpoint
    /// has. A batch runs at its own longest chunk, so this bounds a chunk; it
    /// does not size every run.
    pub max_seq: Option<usize>,
    /// Compute dtype. `None` takes [`compute_dtype_for`](crate::compute_dtype_for)
    /// the head's device.
    pub dtype: Option<DType>,
    /// The device the weights load onto and every plan compiles for. `None`
    /// takes the building thread's default device.
    pub device: Option<DeviceSpec>,
}

impl Default for HeadOptions {
    /// One chunk per execute, the checkpoint's full length, and the building
    /// thread's device and its compute dtype.
    fn default() -> Self {
        Self { max_batch: 1, max_seq: None, dtype: None, device: None }
    }
}

impl HeadOptions {
    /// Run a head's `build` on the resolved device, handing it the resolved
    /// dtype: the weights load onto that device, and [`HeadPlans`] records it.
    pub(crate) fn build<T>(&self, build: impl FnOnce(DType) -> T) -> T {
        let device = self.device.clone().unwrap_or_else(default_device);
        let dtype = self.dtype.clone().unwrap_or_else(|| crate::compute_dtype_for(&device));
        with_default_device(device, || build(dtype))
    }
}

/// How a hub loader fetches a checkpoint and builds its head.
#[derive(Clone, Debug)]
pub struct HubOptions {
    /// Hub revision: a branch, tag or commit.
    pub revision: String,
    /// The repository and revision holding `tokenizer.json`, for checkpoints
    /// that ship none (SaT reuses `FacebookAI/xlm-roberta-base`'s). `None`
    /// reads the model's repository at [`Self::revision`].
    pub tokenizer_repo: Option<(String, String)>,
    pub head: HeadOptions,
}

impl Default for HubOptions {
    /// `main`, the model's own tokenizer, and [`HeadOptions::default`].
    fn default() -> Self {
        Self { revision: "main".into(), tokenizer_repo: None, head: HeadOptions::default() }
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

// ── pooling ─────────────────────────────────────────────────────────────────

/// Which tokens' states a head pools into one vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PoolingMode {
    /// The first token's (`[CLS]` / `<s>`).
    Cls,
    /// The mean over the real tokens.
    #[default]
    Mean,
}

/// How an embedder turns a chunk's token states into its embedding: the
/// sentence-transformers `Pooling` (and optional `Normalize`) modules. The
/// default, mean then L2, is what a checkpoint without that config gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pooling {
    pub mode: PoolingMode,
    /// Scale each embedding to unit length.
    pub normalize: bool,
}

impl Default for Pooling {
    fn default() -> Self {
        Self { mode: PoolingMode::Mean, normalize: true }
    }
}

/// Why a checkpoint's pooling could not be read, or is one these heads do not
/// reproduce.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum PoolingError {
    #[snafu(display("fetching the pooling config: {source}"))]
    Hub { source: HFError },
    #[snafu(display("reading {}: {source}", path.display()))]
    Read { path: PathBuf, source: std::io::Error },
    #[snafu(display("parsing {file}: {source}"))]
    Json { file: String, source: serde_json::Error },
    #[snafu(display("sentence-transformers {what} is not supported"))]
    Unsupported { what: String },
}

/// One entry of sentence-transformers' `modules.json`.
#[derive(Deserialize)]
struct StModule {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    path: String,
}

/// sentence-transformers' `Pooling` config; older files lack the later keys.
#[derive(Deserialize)]
struct StPooling {
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_max_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: bool,
    #[serde(default)]
    pooling_mode_weightedmean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
    #[serde(default = "include_prompt_default")]
    include_prompt: bool,
}

fn include_prompt_default() -> bool {
    true
}

impl Pooling {
    /// Read a sentence-transformers layout: `modules_json` is its
    /// `modules.json`, and `read` returns a file of the checkpoint by its
    /// repository path (the pooling module's `<path>/config.json`). Anything
    /// these heads would compute differently is an error, not a guess: a
    /// pooling mode other than CLS or mean, several at once, a prompt left out
    /// of the pool, the transformer in a subdirectory, or any module past
    /// `Pooling` and `Normalize` (a `Dense` projection, say).
    pub fn from_sentence_transformers(
        modules_json: &str,
        read: impl FnOnce(&str) -> Result<String, PoolingError>,
    ) -> Result<Self, PoolingError> {
        let modules: Vec<StModule> = serde_json::from_str(modules_json).context(JsonSnafu { file: "modules.json" })?;
        let (mut pooling_path, mut normalize) = (None, false);
        for module in &modules {
            match module.kind.rsplit('.').next().unwrap_or_default() {
                "Transformer" => ensure!(
                    module.path.is_empty(),
                    UnsupportedSnafu { what: format!("a transformer under {:?}", module.path) }
                ),
                "Pooling" => pooling_path = Some(module.path.as_str()),
                "Normalize" => normalize = true,
                _ => return UnsupportedSnafu { what: format!("module {}", module.kind) }.fail(),
            }
        }
        let path = pooling_path.ok_or_else(|| UnsupportedSnafu { what: "a layout without Pooling" }.build())?;
        let file = if path.is_empty() { "config.json".to_owned() } else { format!("{path}/config.json") };
        let config: StPooling = serde_json::from_str(&read(&file)?).context(JsonSnafu { file: file.as_str() })?;
        ensure!(config.include_prompt, UnsupportedSnafu { what: "include_prompt = false" });
        let modes = [
            (config.pooling_mode_cls_token, Some(PoolingMode::Cls)),
            (config.pooling_mode_mean_tokens, Some(PoolingMode::Mean)),
            (config.pooling_mode_max_tokens, None),
            (config.pooling_mode_mean_sqrt_len_tokens, None),
            (config.pooling_mode_weightedmean_tokens, None),
            (config.pooling_mode_lasttoken, None),
        ];
        let mut set = modes.iter().filter(|(on, _)| *on).map(|&(_, mode)| mode);
        let mode = match (set.next(), set.next()) {
            (Some(Some(mode)), None) => mode,
            (Some(None), None) => return UnsupportedSnafu { what: "a pooling mode other than CLS or mean" }.fail(),
            (None, _) => return UnsupportedSnafu { what: "a Pooling without a mode" }.fail(),
            (Some(_), Some(_)) => return UnsupportedSnafu { what: "several pooling modes at once" }.fail(),
        };
        Ok(Self { mode, normalize })
    }
}

/// The pooling a hub checkpoint's sentence-transformers config gives, or
/// [`Pooling::default`] when the repository has no `modules.json`. Only a
/// missing file (a 404, or the cache's record of one) takes the default: a
/// file the cache never fetched is an error offline, since guessing could
/// pool differently than the checkpoint says — run once online.
pub(crate) fn pooling_from_hub(repo: &HubRepo) -> Result<Pooling, PoolingError> {
    let read = |file: &str| {
        let path = repo.get(file).context(HubSnafu)?;
        std::fs::read_to_string(&path).context(ReadSnafu { path })
    };
    match repo.get("modules.json") {
        Err(HFError::EntryNotFound { .. }) => Ok(Pooling::default()),
        Err(source) => Err(PoolingError::Hub { source }),
        Ok(path) => {
            let modules = std::fs::read_to_string(&path).context(ReadSnafu { path })?;
            Pooling::from_sentence_transformers(&modules, read)
        }
    }
}

/// Pool `hidden` `(B, L, D)` over the real tokens of `mask` `(B, L)` as
/// `pooling` says, into finished `(B, D)` f32 embeddings. The mean runs in
/// f32, as the reference pools: a bf16 mean rounds the pooled vector itself,
/// an error no averaging over tokens washes out. An all-pad row pools to the
/// zero vector and stays zero.
pub(crate) fn pool(hidden: &Tensor, mask: &Tensor, pooling: Pooling) -> svod_tensor::error::Result<Tensor> {
    let pooled = match pooling.mode {
        PoolingMode::Cls => hidden.take_index(1, 0)?.cast(DType::Float32),
        PoolingMode::Mean => hidden.cast(DType::Float32).masked_mean(mask, 1)?,
    };
    if pooling.normalize { pooled.lp_normalize(-1, 2) } else { Ok(pooled) }
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
/// first use for the device the head was built on (see the module doc).
pub(crate) struct HeadPlans<J: HeadJit> {
    model: J::Model,
    max_batch: usize,
    max_seq: usize,
    pad_id: i64,
    device: DeviceSpec,
    stage: &'static str,
    plans: BTreeMap<(usize, usize), J>,
}

impl<J: HeadJit> HeadPlans<J> {
    /// Plans for batches of up to `max_batch` chunks of up to `max_seq` tokens,
    /// padded with `pad_id`, on the building thread's default device. `max_seq`
    /// may not exceed `max_positions`, the longest chunk the checkpoint has
    /// positions for.
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
        Ok(Self { model, max_batch, max_seq, pad_id, device: default_device(), stage, plans: BTreeMap::new() })
    }

    pub(crate) fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.max_seq)
    }

    /// Run `batch` through the plan for its shape, compiling it first if this
    /// is the shape's first batch. Returns the live rows of the output.
    pub(crate) fn run(&mut self, batch: &[&Encoding], profile: bool) -> Result<HeadRun, HeadError> {
        with_default_device(self.device.clone(), || self.run_on_device(batch, profile))
    }

    /// Compile the plan a batch of `batch` chunks, the longest `len` tokens,
    /// runs on, and execute it once on padding: prepare compiles and tunes,
    /// but the first execute still captures the graph, makes the plan's
    /// context and leases queues. A shape already prepared only executes; a
    /// batch of none runs no plan, so warms none.
    pub(crate) fn warm_up(&mut self, batch: usize, len: usize) -> Result<(), HeadError> {
        if batch == 0 {
            return Ok(());
        }
        self.check(batch, len)?;
        with_default_device(self.device.clone(), || self.execute(&[], plan_shape(batch, len, self.max_batch), false))
            .map(drop)
    }

    /// The `(rows, len)` shapes prepared so far.
    #[cfg(test)]
    pub(crate) fn shapes(&self) -> Vec<(usize, usize)> {
        self.plans.keys().copied().collect()
    }

    fn check(&self, b: usize, longest: usize) -> Result<(), HeadError> {
        let stage = self.stage;
        ensure!(b <= self.max_batch, CapacityExceededSnafu { stage, got: b, max: self.max_batch });
        // Chunks built outside the pipeline's chunker (the public chunk seam)
        // can exceed max_seq: catch it rather than silently truncating.
        ensure!(longest <= self.max_seq, SequenceTooLongSnafu { stage, got: longest, max: self.max_seq });
        Ok(())
    }

    fn run_on_device(&mut self, batch: &[&Encoding], profile: bool) -> Result<HeadRun, HeadError> {
        let b = batch.len();
        if b == 0 {
            return Ok(HeadRun { output: Vec::new(), len: 0, profile: profile.then(RunProfile::default) });
        }
        let longest = batch.iter().map(|e| e.input_ids.len()).max().unwrap_or(0);
        self.check(b, longest)?;
        self.execute(batch, plan_shape(b, longest, self.max_batch), profile)
    }

    /// Pack `batch` into the `(rows, len)` plan, compiling it on first use,
    /// execute it, and read back the live rows.
    fn execute(
        &mut self,
        batch: &[&Encoding],
        (rows, len): (usize, usize),
        profile: bool,
    ) -> Result<HeadRun, HeadError> {
        let (stage, pad_id) = (self.stage, self.pad_id);
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
            pad_id,
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
        output.truncate(output.len() / rows * batch.len());
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
