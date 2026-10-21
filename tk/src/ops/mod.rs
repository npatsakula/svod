//! The consumer surface: every op returns a Tensor. A tk kernel runs when the
//! device, dtype and shape fit; otherwise the op builds the equivalent graph
//! itself. `Err` is reserved for what the graph op would also reject.
//!
//! Kernels take 16-bit operands only: f32 keeps the graph (casting it down
//! trades about three decimal digits for the speed).

pub(crate) mod attention;
pub mod config;
pub(crate) mod conv;
pub(crate) mod heads;
pub(crate) mod linear;
pub(crate) mod nearest;
pub(crate) mod norm;
pub mod shape;

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, LazyLock, Mutex};

use snafu::Snafu;
use svod_device::BufferSpec;
use svod_dtype::default_device::default_device;
use svod_dtype::{DType, DeviceSpec, ScalarDType};
use svod_ir::SInt;
use svod_tensor::Tensor;

pub use self::attention::{Attn, Cache, Chunk, KeyMask, attention};
pub use self::conv::{Conv, conv2d};
pub use self::heads::{Qkv, heads};
pub use self::linear::{Linear, linear};
pub use self::nearest::{Metric, Nearest, k_nearest, nearest};
pub use self::norm::{add_layer_norm, add_rms_norm, layer_norm, rms_norm};
use crate::atoms::{Device, Target};
pub use crate::kernels::Act;
use crate::kernels::Batch;
use crate::tune::{self, Bench, Pass, TuneKey, TuneStore};
use config::Planner;
use shape::{BatchVar, Plan};

/// The salt of the tunings the device tests and GEMM probes run, keyed by
/// the kernel's name (`gemm`) rather than an op's: [`tuned::gemm_probe`]
/// reads what they chose.
pub const DEVICE_TEST: &str = "a device test";

/// Each op's tuning off the tensors: the spec the op launches for a shape,
/// given the configs its shape gate ranked ([`shape`]) and a spec of any
/// config (the pick replaces it), with the choices read from and kept in a
/// store: [`TuneStore::global`] as the op runs, a [`TuneStore::replay`] of
/// a store copied from a card to read what was tuned there.
pub mod tuned {
    pub use super::attention::tuned as attention;
    pub use super::conv::tuned as conv;
    pub use super::heads::tuned as heads;
    pub use super::linear::{probed as gemm_probe, tuned as linear};
    pub use super::nearest::tuned as nearest;
    pub use super::norm::tuned as norm;
}

/// The tracing target of the planner's decisions, at debug: an op that
/// takes the graph and why, a ranking's front with each config's predicted
/// ns and how many it left out, on a store miss what compiling each config
/// showed (its verdict, registers and spilled bytes), and the tuned pick.
pub const PLAN: &str = "svod_tk::plan";

/// `plan`'s ranked configs, or `None` with why `op` takes the graph logged.
fn kernel<C>(op: &'static str, plan: Plan<C>) -> Option<Arc<[C]>> {
    match plan {
        Plan::Kernel(cfgs) => Some(cfgs),
        Plan::Graph(why) => {
            tracing::debug!(target: PLAN, op, ?why, "graph");
            None
        }
    }
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("{op}: {operand} has shape {got}, expected {expected}"))]
    Shape { op: &'static str, operand: &'static str, got: String, expected: String },
    #[snafu(display("{op}: {operand} is {got:?}, the input is {want:?}"))]
    Dtype { op: &'static str, operand: &'static str, got: DType, want: DType },
    #[snafu(display("{op}: {heads} query heads are not a multiple of {kv_heads} key/value heads"))]
    Heads { op: &'static str, heads: usize, kv_heads: usize },
    #[snafu(display("{op}: a row keeps no score (k = 0)"))]
    NoScores { op: &'static str },
    #[snafu(display("{op}: k is {got}, neither a count nor a bound variable"))]
    Unbound { op: &'static str, got: String },
    #[snafu(display("{op}: {source}"))]
    Graph {
        op: &'static str,
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("{op}: {source}"))]
    Launch { op: &'static str, source: crate::launch::Error },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Whether `device` has a planner: it is the default device, and its target
/// has kernels and reports an SM count. Its cost probes run on the first ask;
/// a probe that fails leaves its cost at zero rather than the device to the graph.
pub fn supported(device: &DeviceSpec) -> bool {
    planner(device).is_some()
}

/// The planner of a device kernels can be launched on (graph kernels lower
/// for the default device): its target, its measured costs and the
/// candidate lists ranked for it, kept for the process, one per device. A
/// device describes itself once: its probes run under the lock, and a
/// second caller waits for them.
pub(crate) fn planner(device: &DeviceSpec) -> Option<Arc<Planner>> {
    static PLANNERS: LazyLock<Mutex<HashMap<DeviceSpec, Option<Arc<Planner>>>>> = LazyLock::new(Mutex::default);
    if *device != default_device() {
        return None;
    }
    let mut planners = PLANNERS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    planners
        .entry(device.clone())
        .or_insert_with(|| {
            let target = Target::for_device(device).filter(config::has_kernels)?;
            // A device no probe can scale to (no SM count) has no planner; a
            // probe that fails leaves its field at zero and the planner ranks on.
            let measured = crate::probe::costs(&target)
                .inspect_err(|e| tracing::warn!("{}: no tk planner: {e}", target.arch.target_name()))
                .ok()?;
            Planner::measured(target, measured).map(Arc::new)
        })
        .clone()
}

/// One shape's tuning on the default device: what keys its store entries
/// and the programs a config builds, compiled once on one [`Bench`] for
/// every pass that asks (admission, the choice, later knobs). Each pass is
/// a list under an op name: [`Self::admitted`] keeps the first
/// [`SHORTLIST`](config::SHORTLIST) tiles of a ranked list whose compiled
/// kernels do not spill (all spilling: the first that compile; nothing
/// compiling, no device: the first [`SHORTLIST`](config::SHORTLIST) tiles,
/// untested), [`Self::fastest`] times a list on a store miss, and
/// [`Self::pipelined`] times the winning tile's other pipelines. With tuning
/// off (`SVOD_TK_TUNE=0`) nothing is timed: the ranked list's
/// [`Self::first_clean`] config runs (its verdict kept as a `first` line), and
/// the device's cost probes still measure once ([`planner`]).
pub(crate) struct Tuner<'a, C, B> {
    store: &'a TuneStore,
    target: &'a Target,
    compiler: u64,
    dtype: ScalarDType,
    shape: Vec<usize>,
    salt: String,
    build: B,
    bench: Bench<C>,
}

impl<'a, C: config::Tiled + Debug, B: Fn(C) -> tune::Candidate> Tuner<'a, C, B> {
    /// `salt` is what else the programs `build` makes vary with; the
    /// choices are read from and kept in `store`. A replaying store keys no
    /// compiler (it matches any), so no device opens.
    pub(crate) fn new(
        store: &'a TuneStore,
        device: &'a Device,
        dtype: DType,
        shape: &[usize],
        salt: impl Debug,
        build: B,
    ) -> Self {
        let compiler = if store.replays() { 0 } else { tune::compiler(&default_device()) };
        Self {
            store,
            target: &device.target,
            compiler,
            dtype: dtype.scalar().expect("a kernel dtype"),
            shape: shape.to_vec(),
            salt: format!("{salt:?}"),
            build,
            bench: Bench::default(),
        }
    }

    /// The fingerprint of the programs `build` makes of `list`, what an
    /// admission's store line carries: a lowering change re-judges.
    fn programs(store: &TuneStore, build: &B, list: &[C]) -> u64 {
        store.fingerprint(&list.iter().map(|&c| build(c)).collect::<Vec<_>>())
    }

    fn key(&self, op: &'static str, list: &[C]) -> TuneKey {
        self.keyed(op, "", list)
    }

    /// [`Self::key`] under `rule`, the name of the judgement stored where it
    /// is not the plain one: a verdict stored under an older rule is left
    /// orphaned rather than read.
    fn keyed(&self, op: &'static str, rule: &str, list: &[C]) -> TuneKey {
        let candidates = format!("{}{rule} {list:?}", self.salt);
        TuneKey::new(op, self.target, self.compiler, self.dtype, &self.shape, &candidates)
    }

    /// The first config of `ranked` whose compiled kernel does not spill
    /// ([`config::first_clean`]; nothing compiling: the first; nothing
    /// ranked: `None`), compiled in ranked batches on a store miss and kept
    /// in the store under the batch's programs: what runs with tuning off,
    /// and where a pass that starts from one config starts.
    pub(crate) fn first_clean(&mut self, op: &'static str, ranked: &[C]) -> Option<C> {
        let ranked = &config::representatives(ranked);
        let key = self.key(op, ranked);
        let (store, bench, build) = (self.store, &mut self.bench, &self.build);
        let programs = || Self::programs(store, build, &ranked[..ranked.len().min(config::FIRST)]);
        let find = || config::first_clean(ranked, |batch| bench.admit(batch, build));
        let at = store.first(&key, programs, find);
        let cfg = ranked.get(at.unwrap_or(0)).copied();
        tracing::debug!(target: PLAN, op, shape = ?self.shape, ?cfg, stored = at.is_some(), "first");
        cfg
    }

    /// The configs of `ranked` (best guess first) the tuner chooses among
    /// ([`config::shortlist`]), compiled in parallel on a store miss
    /// and kept in the store under the shortlist's programs; with tuning
    /// off, the [`Self::first_clean`] alone.
    pub(crate) fn admitted(&mut self, op: &'static str, ranked: &[C]) -> Vec<C> {
        if !tune::enabled() {
            return self.first_clean(op, ranked).into_iter().collect();
        }
        let key = self.key(op, ranked);
        let (store, bench, build) = (self.store, &mut self.bench, &self.build);
        let programs = || Self::programs(store, build, &ranked[..ranked.len().min(config::SHORTLIST)]);
        let admit = || config::shortlist(ranked, |batch| bench.admit(batch, build));
        let found = store.admitted(&key, programs, admit);
        let configs: Vec<C> = match &found {
            Some(at) => at.iter().map(|&i| ranked[i]).collect(),
            None => config::distinct(ranked).take(config::SHORTLIST).collect(),
        };
        tracing::debug!(target: PLAN, op, shape = ?self.shape, ?configs, stored = found.is_some(), "admitted");
        configs
    }

    /// `candidates` in the order they ran by the global store under `op`,
    /// fastest first, each with its device ns, timed through `pass` on a
    /// miss (a screen's order kept under its own key); as given and untimed
    /// when tuning is off, alone or nothing ran.
    pub(crate) fn ordered(&mut self, op: &'static str, candidates: &[C], pass: Pass) -> Vec<(C, Option<u64>)> {
        let untimed = || candidates.iter().map(|&c| (c, None)).collect();
        if !tune::enabled() || candidates.len() == 1 {
            return untimed();
        }
        let key = self.keyed(op, if pass == Pass::Screen { " screened" } else { "" }, candidates);
        let bench = &mut self.bench;
        let Some(timed) = self.store.choose(&key, candidates, &self.build, bench, pass) else { return untimed() };
        let timed: Vec<(C, u64)> = timed.iter().map(|&(i, ns)| (candidates[i], ns)).collect();
        tracing::debug!(target: PLAN, op, shape = ?self.shape, ?timed, "choice");
        timed.into_iter().map(|(c, ns)| (c, Some(ns))).collect()
    }

    /// The fastest of `candidates` ([`Self::ordered`]).
    pub(crate) fn fastest(&mut self, op: &'static str, candidates: &[C]) -> C {
        self.ordered(op, candidates, Pass::Whole)[0].0
    }

    /// The candidate to launch: `expand` makes the admitted configs of
    /// `ranked`'s tile [representatives](config::Tiled::represents) the
    /// candidates (adding variants, such as key splits, the
    /// plain one first), they are screened ([`Pass::Screen`]), and the
    /// fastest pipeline of the leading tiles ([`Self::pipelined`], which
    /// times them again) is the choice.
    pub(crate) fn tuned(&mut self, op: &'static str, ranked: &[C], expand: impl Fn(&[C]) -> Vec<C>) -> C {
        let candidates = expand(&self.admitted(op, &config::representatives(ranked)));
        let ordered = self.ordered(op, &candidates, Pass::Screen);
        let pick = self.pipelined(op, &ordered, ranked);
        tracing::debug!(target: PLAN, op, shape = ?self.shape, ?pick, "tuned");
        pick
    }

    /// The fastest pipeline of the leading tiles of `ordered` (fastest
    /// first, with their times): the tiles within the contender band of the
    /// leader, at most [`config::TWIN_TILES`], their [twins](config::Tiled::twins)
    /// in `ranked` compiled and timed together, every one that compiles,
    /// spilling or not ([`config::compiled`]), the winner among them (the
    /// shortlist held one pipeline per tile, the model's first clean one,
    /// and a tile's place in it is that pipeline's). The leader itself with
    /// tuning off, alone, or where no other twin compiles to be judged.
    pub(crate) fn pipelined(&mut self, op: &'static str, ordered: &[(C, Option<u64>)], ranked: &[C]) -> C {
        let (winner, lead) = ordered[0];
        let near: Vec<C> = ordered
            .iter()
            .filter(|(_, ns)| lead.zip(*ns).is_none_or(|(lead, ns)| ns as f64 <= lead as f64 * tune::CONTENDER))
            .map(|&(c, _)| c)
            .collect();
        let leaders: Vec<C> = config::distinct(&near).take(config::TWIN_TILES).collect();
        let twins: Vec<C> = config::distinct_configs(&leaders.iter().flat_map(|l| l.twins(ranked)).collect::<Vec<_>>());
        tracing::debug!(?leaders, twins = twins.len(), "tuner's twins");
        if !tune::enabled() || twins.len() < 2 {
            return winner;
        }
        let key = self.keyed(op, " compiled", &twins);
        let (store, bench, build) = (self.store, &mut self.bench, &self.build);
        let admit = || config::compiled(&twins, |batch| bench.admit(batch, build));
        let judged = store.admitted(&key, || Self::programs(store, build, &twins), admit);
        let compiled: Vec<C> = judged.iter().flat_map(|at| at.iter().map(|&i| twins[i])).collect();
        tracing::debug!(target: PLAN, op, shape = ?self.shape, ?compiled, "twins admitted");
        if compiled.len() > 1 { self.fastest(op, &compiled) } else { winner }
    }
}

fn fmt_shape(shape: &[SInt]) -> String {
    let dims: Vec<String> = shape.iter().map(SInt::to_string).collect();
    format!("[{}]", dims.join(", "))
}

/// The kernel batch axis of an operand whose leading dims hold `lead` rows.
fn batch_of(var: &Option<BatchVar>, static_batches: usize) -> Batch {
    match var {
        Some(v) => Batch::Var { name: v.name.clone(), min: v.min, max: v.max },
        None => Batch::Static(static_batches),
    }
}

/// A kernel output: allocated at capacity, shaped with the live batch in dim 0
/// so a consumer kernel binds the realized buffer itself rather than a copy.
fn output(dims: &[usize], var: &Option<BatchVar>, dtype: DType) -> svod_tensor::error::Result<Tensor> {
    let mut shape: Vec<SInt> = dims.iter().map(|&d| SInt::Const(d)).collect();
    if let Some(var) = var {
        shape[0] = var.dim.clone();
    }
    kernel_buffer(&shape, dtype)
}

/// Storage a kernel writes: device-local where the backend tells it apart
/// from host-visible memory, as the scheduler places graph intermediates;
/// the host reads it back through `copyout`.
fn kernel_buffer(shape: &[SInt], dtype: DType) -> svod_tensor::error::Result<Tensor> {
    let local = svod_device::registry::registry().get(&default_device()).is_ok_and(|a| a.supports_device_local());
    Tensor::empty_dynamic_spec(shape, dtype, BufferSpec { cpu_access: !local, ..BufferSpec::default() })
}

/// `f::<T>(spec)` for the 16-bit element type `dtype`.
macro_rules! typed {
    ($dtype:expr, $f:ident, $spec:expr) => {
        match $dtype.scalar() {
            Some(svod_dtype::ScalarDType::BFloat16) => $f::<crate::build::BF16>($spec),
            Some(svod_dtype::ScalarDType::Float16) => $f::<crate::build::F16>($spec),
            other => unreachable!("{other:?} has no kernel"),
        }
    };
}
use typed;
