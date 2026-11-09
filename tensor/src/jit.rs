//! Support types for the `jit_wrapper!` macro: the input descriptor, the error
//! type, the live output-shape descriptor and the buffer helpers the generated
//! code calls.
//!
//! They live in `svod-tensor` (not in a model crate) so that any crate can host
//! a `jit_wrapper!` invocation with only `svod-tensor` in its dependency list —
//! the expansion refers to everything else through [`rt`].

use snafu::Snafu;
use svod_dtype::DType;
use svod_dtype::ext::HasDType;
use svod_ir::SInt;

use crate::{BoundVariable, Tensor};

/// Every path the `jit_wrapper!` expansion mentions, re-exported from one
/// place. An invoking crate needs `svod-tensor` and nothing else.
pub mod rt {
    pub use ndarray::ArrayViewD;
    pub use svod_device::{Buffer, BufferId, BufferSpec, HostWrite};
    pub use svod_dtype::ext::HasDType;
    pub use svod_ir::origin::OriginScope;
    pub use svod_runtime::{ExecutionPlan, KernelProfile, PreparedKernel, ProfileOptions};
}

/// Shape + dtype descriptor for a single JIT input. Used by
/// `jit_wrapper!`-generated `prepare()` calls to allocate zero-initialized
/// placeholder buffers internally — callers no longer construct fake
/// `Tensor::zeros(..).realize()` placeholders.
#[derive(Clone, Debug)]
pub struct InputSpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Allocate the input device-local (no host mapping). The host can still
    /// reach it through `copyin`/`copyout` (staged over the copy engine), and
    /// on-device `copy_from`/`copy_region_from` stays on-device — for
    /// recurrent state the host shouldn't observe per step. `state { .. }`
    /// slots are always allocated this way.
    pub device_local: bool,
}

impl InputSpec {
    pub fn new(shape: &[usize], dtype: DType) -> Self {
        Self { shape: shape.to_vec(), dtype, device_local: false }
    }

    pub fn f32(shape: &[usize]) -> Self {
        Self::new(shape, DType::Float32)
    }

    pub fn i32(shape: &[usize]) -> Self {
        Self::new(shape, DType::Int32)
    }

    pub fn i64(shape: &[usize]) -> Self {
        Self::new(shape, DType::Int64)
    }

    pub fn device_local(mut self) -> Self {
        self.device_local = true;
        self
    }

    /// Element count of the declared shape.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum JitError {
    #[snafu(display("JIT not prepared: call prepare() first"))]
    NotPrepared,

    #[snafu(display("input buffer not found: {name}"))]
    InputBufferNotFound { name: &'static str },

    #[snafu(display("duplicate JIT input buffer: {name} aliases {duplicate_of} with {buffer_id:?}"))]
    DuplicateInputBuffer { name: &'static str, duplicate_of: &'static str, buffer_id: svod_device::BufferId },

    /// The plan buffer an input resolved to is not the buffer that was
    /// realized for it — cross-plan input aliasing (a concurrent prepare
    /// corrupted this input's graph identity).
    #[snafu(display("JIT input {name} resolved to a foreign plan buffer: expected {expected:?}, got {actual:?}"))]
    InputAliased { name: &'static str, expected: svod_device::BufferId, actual: svod_device::BufferId },

    /// Wraps the user-supplied error type returned by a `jit_wrapper!` build
    /// closure. Genuine `Box<dyn>` because the closure's `E` is arbitrary.
    #[snafu(display("{source}"))]
    Build { source: Box<dyn std::error::Error + Send + Sync> },

    #[snafu(display("{source}"), context(false))]
    Tensor {
        #[snafu(source(from(crate::error::Error, Box::new)))]
        source: Box<crate::error::Error>,
    },

    #[snafu(display("{source}"), context(false))]
    Device {
        #[snafu(source(from(svod_device::error::Error, Box::new)))]
        source: Box<svod_device::error::Error>,
    },

    // A multi-output `jit_wrapper!` declared `declared` outputs — its declared
    // output slots plus its state slots — but the compiled plan kept `actual`.
    // Means the scheduler fused or elided an output the `build` closure
    // returned; the positional accessors would be misaligned, so fail loud at
    // `prepare` instead. (A plain comment: snafu reads doc comments as format
    // strings, and braces in one would be parsed as arguments.)
    #[snafu(display("JIT output count mismatch: declared {declared} outputs, plan kept {actual}"))]
    OutputCountMismatch { declared: usize, actual: usize },

    /// A typed view/read asked for a dtype the buffer does not hold.
    #[snafu(display("JIT buffer dtype mismatch: view asked for {expected:?}, buffer holds {actual:?}"))]
    DtypeMismatch { expected: DType, actual: DType },

    /// A live output shape needs more elements than its buffer holds — the
    /// bound variable values exceed what the plan was compiled for.
    #[snafu(display("JIT output view of {requested} elements exceeds the {available}-element buffer"))]
    ViewOutOfBounds { requested: usize, available: usize },

    /// An output shape carried a `-1` (inferred) dimension, which has no live
    /// value to substitute.
    #[snafu(display("JIT output shape has an inferred (-1) dimension"))]
    InferredOutputDim,

    #[snafu(display("{source}"))]
    Runtime { source: svod_runtime::Error },
}

pub type Result<T> = std::result::Result<T, JitError>;

/// One dimension of a captured output shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    /// Compile-time constant extent.
    Const(usize),
    /// The extent is the JIT variable at this positional index; the live value
    /// is the one last bound by `execute_bound`/`execute_with_vars`.
    Var(usize),
}

/// The shape of one declared output, captured before the plan is compiled and
/// resolved against the live variable bindings afterwards.
///
/// A symbolic dimension that is not one of the wrapper's declared variables
/// (nothing in-tree produces one) degrades to its upper bound, which is what
/// the buffer was allocated for.
#[derive(Clone, Debug, Default)]
pub struct OutputShape {
    dims: Vec<Dim>,
    dtype: Option<DType>,
}

impl OutputShape {
    /// Capture `tensor`'s shape, resolving each symbolic dimension against
    /// `vars` (positional). Must run before `prepare_batch_with` rewrites the
    /// tensor onto its plan buffer.
    pub fn capture(tensor: &Tensor, vars: &[&BoundVariable]) -> Result<Self> {
        let var_sints: Vec<SInt> = vars.iter().map(|v| v.as_sint()).collect();
        let dims = tensor
            .shape()?
            .iter()
            .map(|d| match d {
                SInt::Const(v) => Ok(Dim::Const(*v)),
                SInt::Infer => Err(JitError::InferredOutputDim),
                symbolic => Ok(var_sints
                    .iter()
                    .position(|v| v == symbolic)
                    .map_or_else(|| Dim::Const(crate::sint_vmax(symbolic)), Dim::Var)),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { dims, dtype: Some(tensor.dtype()) })
    }

    /// The live shape for the given positional variable values.
    pub fn resolve(&self, values: &[i64]) -> Vec<usize> {
        self.dims
            .iter()
            .map(|d| match *d {
                Dim::Const(v) => v,
                Dim::Var(i) => values.get(i).copied().unwrap_or(0).max(0) as usize,
            })
            .collect()
    }

    /// Element count of the live shape.
    pub fn numel(&self, values: &[i64]) -> usize {
        self.resolve(values).iter().product()
    }

    pub fn dims(&self) -> &[Dim] {
        &self.dims
    }

    pub fn dtype(&self) -> Option<&DType> {
        self.dtype.as_ref()
    }
}

/// Shrink `tensor`'s leading dimension to the live batch value — what a
/// `batch_var b: (min, max)` declaration applies to every batched input after
/// realization. Rank-0 tensors pass through.
pub fn shrink_batch(tensor: &Tensor, b: &BoundVariable) -> Result<Tensor> {
    let rank = tensor.shape()?.len();
    if rank == 0 {
        return Ok(tensor.clone());
    }
    let mut ranges: Vec<Option<(SInt, SInt)>> = vec![None; rank];
    ranges[0] = Some((SInt::Const(0), b.as_sint()));
    Ok(tensor.try_shrink(ranges)?)
}

fn check_dtype<T: HasDType>(buffer: &rt::Buffer) -> Result<()> {
    let actual = buffer.dtype();
    snafu::ensure!(actual == T::DTYPE, DtypeMismatchSnafu { expected: T::DTYPE, actual });
    Ok(())
}

/// Typed read-only view over a buffer's live contiguous prefix, reshaped to
/// `shape`. The allocation is sized for the maximum bindings, so the live
/// region is a prefix of it (the batch dimension is dim 0).
pub fn view<'a, T: HasDType>(buffer: &'a rt::Buffer, shape: &[usize]) -> Result<rt::ArrayViewD<'a, T>> {
    check_dtype::<T>(buffer)?;
    let full = buffer.as_array::<T>()?;
    let flat = full.to_slice().expect("buffer views are contiguous");
    let requested: usize = shape.iter().product();
    snafu::ensure!(requested <= flat.len(), ViewOutOfBoundsSnafu { requested, available: flat.len() });
    Ok(rt::ArrayViewD::from_shape(ndarray::IxDyn(shape), &flat[..requested]).expect("prefix matches shape"))
}

/// Exclusive host write access to a host-visible input buffer as `[T]`
/// (see [`rt::HostWrite`]); it publishes its stores when dropped.
pub fn host_write<T: HasDType + Copy>(buffer: &mut rt::Buffer) -> Result<rt::HostWrite<'_, T>> {
    check_dtype::<T>(buffer)?;
    Ok(buffer.host_write::<T>()?)
}

/// Read a buffer's first `numel` elements to the host. Works for device-local
/// buffers (staged over the copy engine) where [`view`] cannot.
pub fn to_vec<T: HasDType + Default + Clone>(buffer: &rt::Buffer, numel: usize) -> Result<Vec<T>> {
    check_dtype::<T>(buffer)?;
    let bytes = numel * T::DTYPE.bytes();
    let available = buffer.size();
    snafu::ensure!(bytes <= available, ViewOutOfBoundsSnafu { requested: bytes, available });
    let mut out = vec![T::default(); numel];
    // Safe: `out` owns `numel * size_of::<T>()` initialized bytes, and
    // `T::DTYPE.bytes() == size_of::<T>()` for every `HasDType` impl.
    let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), bytes) };
    buffer.copyout_prefix(dst)?;
    Ok(out)
}

/// Zero a state slot. No backend exposes a device-side fill, so this stages
/// host zeros through `copyin`.
pub fn zero_fill(buffer: &mut rt::Buffer) -> Result<()> {
    let zeros = vec![0u8; buffer.size()];
    Ok(buffer.copyin(&zeros)?)
}
