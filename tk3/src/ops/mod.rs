//! The consumer surface: every op returns a Tensor. A tk3 kernel runs when the
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
pub(crate) mod norm;
pub mod shape;

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, LazyLock, Mutex};

use snafu::Snafu;
use svod_dtype::default_device::default_device;
use svod_dtype::{DType, DeviceSpec, ScalarDType};
use svod_ir::SInt;
use svod_tensor::Tensor;

pub use self::attention::{Attn, Cache, Chunk, KeyMask, attention};
pub use self::conv::{Conv, conv2d};
pub use self::heads::{Qkv, heads};
pub use self::linear::{Linear, linear};
pub use self::norm::{add_layer_norm, add_rms_norm, layer_norm, rms_norm};
use crate::atoms::{Device, Target};
pub use crate::kernels::Act;
use crate::kernels::Batch;
use crate::tune::{self, Bench, TuneKey, TuneStore};
use config::Planner;
use shape::BatchVar;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("{op}: {operand} has shape {got}, expected {expected}"))]
    Shape { op: &'static str, operand: &'static str, got: String, expected: String },
    #[snafu(display("{op}: {operand} is {got:?}, the input is {want:?}"))]
    Dtype { op: &'static str, operand: &'static str, got: DType, want: DType },
    #[snafu(display("{op}: {heads} query heads are not a multiple of {kv_heads} key/value heads"))]
    Heads { op: &'static str, heads: usize, kv_heads: usize },
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

/// Whether `device` has a planner: a tk3 target with kernels whose costs
/// measure (probed on the first ask).
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
            // A device whose primitives cannot be measured has no planner.
            let costs = crate::probe::costs(&target)
                .inspect_err(|e| tracing::warn!("{}: no tk3 planner: {e}", target.arch.target_name()))
                .ok()?;
            Some(Arc::new(Planner::new(Device { target, costs })))
        })
        .clone()
}

/// One shape's tuning on the default device: what keys its store entries
/// and the programs a config builds, compiled once on one [`Bench`] for
/// every pass that asks (admission, the choice, later knobs). Each pass is
/// a list under an op name: [`Self::admitted`] keeps the first
/// [`SHORTLIST`](config::SHORTLIST) of a ranked list whose compiled kernels
/// do not spill (all spilling: the first that compile; nothing compiling,
/// no device: the first, untested), and [`Self::fastest`] times a list on a
/// store miss. With tuning off (`SVOD_TK3_TUNE=0`) no candidate is timed
/// and the first admitted one is taken; admission still compiles, and the
/// device's cost probes still measure once ([`planner`]).
pub(crate) struct Tuner<'a, C, B> {
    target: &'a Target,
    compiler: u64,
    dtype: ScalarDType,
    shape: &'a [usize],
    salt: String,
    build: B,
    bench: Bench<C>,
}

impl<'a, C: config::Tiled + Debug, B: Fn(C) -> tune::Candidate> Tuner<'a, C, B> {
    /// `salt` is what else the programs `build` makes vary with.
    pub(crate) fn new(device: &'a Device, dtype: DType, shape: &'a [usize], salt: impl Debug, build: B) -> Self {
        let (compiler, dtype) = (tune::compiler(&default_device()), dtype.scalar().expect("a kernel dtype"));
        Self {
            target: &device.target,
            compiler,
            dtype,
            shape,
            salt: format!("{salt:?}"),
            build,
            bench: Bench::default(),
        }
    }

    fn key(&self, op: &'static str, list: &[C]) -> TuneKey {
        let candidates = format!("{} {list:?}", self.salt);
        TuneKey::new(op, self.target, self.compiler, self.dtype, self.shape, &candidates)
    }

    /// The configs of `ranked` (best guess first) the tuner chooses among
    /// ([`config::shortlist`]), compiled in ranked batches and memoized for
    /// the process.
    pub(crate) fn admitted(&mut self, op: &'static str, ranked: &[C]) -> Vec<C> {
        let key = self.key(op, ranked);
        let (bench, build) = (&mut self.bench, &self.build);
        let found = TuneStore::global().admitted(&key, || config::shortlist(ranked, |batch| bench.admit(batch, build)));
        match found {
            Some(at) => at.iter().map(|&i| ranked[i]).collect(),
            None => config::distinct(ranked).take(config::SHORTLIST).collect(),
        }
    }

    /// The fastest of `candidates` by the global store under `op`, timed on
    /// a miss; the first when tuning is off, alone or nothing ran.
    pub(crate) fn fastest(&mut self, op: &'static str, candidates: &[C]) -> C {
        if !tune::enabled() || candidates.len() == 1 {
            return candidates[0];
        }
        let key = self.key(op, candidates);
        let chosen = TuneStore::global().choose(&key, candidates, &self.build, &mut self.bench);
        candidates[chosen.unwrap_or(0)]
    }

    /// The candidate to launch: `expand` makes the admitted configs of
    /// `ranked` the candidates (adding variants, such as key splits, the
    /// plain one first), and the fastest is taken.
    pub(crate) fn tuned(&mut self, op: &'static str, ranked: &[C], expand: impl Fn(&[C]) -> Vec<C>) -> C {
        let candidates = expand(&self.admitted(op, ranked));
        self.fastest(op, &candidates)
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
fn output(dims: &[usize], var: &Option<BatchVar>, dtype: DType) -> Tensor {
    let mut shape: Vec<SInt> = dims.iter().map(|&d| SInt::Const(d)).collect();
    if let Some(var) = var {
        shape[0] = var.dim.clone();
    }
    Tensor::empty_dynamic(&shape, dtype)
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
