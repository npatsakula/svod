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

use std::sync::{Arc, OnceLock};

use snafu::Snafu;
use svod_dtype::default_device::default_device;
use svod_dtype::{DType, DeviceSpec};
use svod_ir::SInt;
use svod_tensor::Tensor;

pub use self::attention::{Attn, Cache, KeyMask, attention};
pub use self::conv::{Conv, conv2d};
pub use self::heads::{Qkv, heads};
pub use self::linear::{Linear, linear};
pub use self::norm::{add_layer_norm, add_rms_norm, layer_norm, rms_norm};
use crate::atoms::Target;
pub use crate::kernels::Act;
use crate::kernels::Batch;
use crate::tune::{self, TuneKey, TuneStore};
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

/// Whether a tk3 target with kernels exists for `device`.
pub fn supported(device: &DeviceSpec) -> bool {
    planner(device).is_some_and(|p| config::has_kernels(&p.target))
}

/// The planner of a device kernels can be launched on (graph kernels lower
/// for the default device): its target and the candidate lists ranked for
/// it, kept for the process, one per device.
fn planner(device: &DeviceSpec) -> Option<Arc<Planner>> {
    static PLANNERS: OnceLock<papaya::HashMap<DeviceSpec, Option<Arc<Planner>>>> = OnceLock::new();
    if *device != default_device() {
        return None;
    }
    let planners = PLANNERS.get_or_init(papaya::HashMap::new).pin();
    planners
        .get_or_insert_with(device.clone(), || Target::for_device(device).map(|t| Arc::new(Planner::new(t))))
        .clone()
}

/// The candidate to launch: the global store's measured winner for `shape`
/// when tuning is on, else the first. `salt` is what else the programs `build`
/// makes vary with.
fn tuned<C: Copy + std::fmt::Debug>(
    op: &'static str,
    target: &Target,
    dtype: DType,
    shape: &[usize],
    salt: impl std::fmt::Debug,
    candidates: &[C],
    build: impl Fn(C) -> tune::Candidate,
) -> C {
    if candidates.len() == 1 || !tune::enabled() {
        return candidates[0];
    }
    let scalar = dtype.scalar().expect("a kernel dtype");
    let key = TuneKey::new(op, target, scalar, shape, &format!("{salt:?} {candidates:?}"));
    TuneStore::global().pick(&key, candidates, build)
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
