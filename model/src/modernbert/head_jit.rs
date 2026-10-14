//! Shared machinery for the ModernBERT encoder heads (embedder / classifier /
//! token classifier): one head error and [`HeadPlans`], the per-shape plan
//! cache every head runs its chunks through.
//!
//! The three heads are mechanically identical up to (a) the JIT wrapper type,
//! (b) the output-decode stride, and (c) the profile stage label, so the
//! capacity guard → shape pick → pack → execute → read-back prelude lives here
//! and each head keeps just its decode tail.
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

use std::collections::BTreeMap;
use std::time::Instant;

use snafu::{ResultExt, Snafu, ensure};
use svod_arch::pipelines::text::{Encoding, RunProfile};
use svod_runtime::{KernelProfile, StageProfile};
use svod_tensor::PrepareConfig;
use svod_tensor::jit::rt::ArrayViewMutD;

use crate::jit::{self, InputSpec};
use crate::modernbert::classifier::ModernBertClassificationModel;
use crate::modernbert::classifier_jit::ModernBertClassifierJit;
use crate::modernbert::embedder_jit::ModernBertEmbedderJit;
use crate::modernbert::model::ModernBert;
use crate::modernbert::packing::{pack_ids, pack_mask};
use crate::modernbert::token_classifier::ModernBertTokenClassificationModel;
use crate::modernbert::token_classifier_jit::ModernBertTokenClassifierJit;
use crate::state::scoped;

/// Head-level error for the ModernBERT encoder heads. `stage` names the head
/// ("embed", "classify", "classify_tokens") so the message stays task-specific.
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

/// The JIT interface the three `jit_wrapper!`-generated head types share. The
/// macro emits inherent methods (no trait), so [`HeadPlans`] is bounded on this
/// trait — implemented once per wrapper by `head_jit!` as straight delegations.
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

macro_rules! head_jit {
    ($jit:ty, $model:ty, $read:ident) => {
        impl HeadJit for $jit {
            type Model = $model;
            fn new(model: $model) -> Self {
                <$jit>::new(model)
            }
            fn prepare_with_config(
                &mut self,
                ids: InputSpec,
                mask: InputSpec,
                config: &PrepareConfig,
            ) -> jit::Result<()> {
                <$jit>::prepare_with_config(self, ids, mask, config)
            }
            fn input_ids_view_mut(&mut self) -> jit::Result<ArrayViewMutD<'_, i64>> {
                <$jit>::input_ids_view_mut(self)
            }
            fn attention_mask_view_mut(&mut self) -> jit::Result<ArrayViewMutD<'_, i64>> {
                <$jit>::attention_mask_view_mut(self)
            }
            fn execute(&mut self) -> jit::Result<()> {
                <$jit>::execute(self)
            }
            fn execute_profiled(&mut self) -> jit::Result<Vec<KernelProfile>> {
                <$jit>::execute_profiled(self)
            }
            fn read_output(&self) -> jit::Result<Vec<f32>> {
                <$jit>::$read::<f32>(self)
            }
        }
    };
}

head_jit!(ModernBertEmbedderJit, ModernBert, embeddings_to_vec);
head_jit!(ModernBertClassifierJit, ModernBertClassificationModel, logits_to_vec);
head_jit!(ModernBertTokenClassifierJit, ModernBertTokenClassificationModel, logits_to_vec);

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
    stage: &'static str,
    plans: BTreeMap<(usize, usize), J>,
}

impl<J: HeadJit> HeadPlans<J> {
    /// Plans for batches of up to `max_batch` chunks of up to `max_seq` tokens.
    /// `max_seq` may not exceed the checkpoint's `max_positions`.
    pub(crate) fn new(
        model: J::Model,
        max_batch: usize,
        max_seq: usize,
        max_positions: usize,
        stage: &'static str,
    ) -> Result<Self, HeadError> {
        ensure!(max_batch > 0 && max_seq > 0, EmptyCapacitySnafu { stage, max_batch, max_seq });
        ensure!(max_seq <= max_positions, BeyondPositionsSnafu { stage, max_seq, max_positions });
        Ok(Self { model, max_batch, max_seq, stage, plans: BTreeMap::new() })
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

        pack_ids(jit.input_ids_view_mut().context(JitSnafu)?.as_slice_mut().expect("contiguous ids"), batch, len);
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
