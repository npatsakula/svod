//! Lazy tensor API for Svod. Operations build a UOp graph and run nothing;
//! [`Tensor::realize`] schedules, compiles and executes it, and [`Tensor::prepare`]
//! compiles it once into an [`ExecutionPlan`](svod_runtime::ExecutionPlan) to replay.
//! See <https://svod.vpermilp.online/docs/examples>.

use bon::bon;
use std::collections::HashMap;
use std::sync::Arc;

use smallvec::smallvec;
use snafu::{OptionExt, ResultExt};
use svod_dtype::DType;
use svod_dtype::ext::HasDType;
use svod_ir::ops;
use svod_ir::{CallInfo, ConstValue, ConstValueHash, DeviceSpec, Op, SInt, UOp, UOpKey, shape::Shape};

/// Extract max value from an SInt for buffer allocation.
///
/// Concrete dims return their value. Symbolic dims (DefineVar, Bind)
/// return `max_val` from the underlying Variable, enabling rebinding
/// without reallocation. Matches Tinygrad's `x.vmax`.
fn sint_vmax(s: &SInt) -> usize {
    match s {
        SInt::Const(v) => *v,
        SInt::Symbolic(uop) => match uop.op() {
            Op::DefineVar(ops::DefineVar { max_val, .. }) => *max_val as usize,
            Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none() => arg
                .vmin_vmax
                .as_ref()
                .and_then(|(_, max)| max.0.try_int())
                .and_then(|max| usize::try_from(max).ok())
                .unwrap_or(1),
            Op::Bind(ops::Bind { var, .. }) => match var.op() {
                Op::DefineVar(ops::DefineVar { max_val, .. }) => *max_val as usize,
                Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none() => arg
                    .vmin_vmax
                    .as_ref()
                    .and_then(|(_, max)| max.0.try_int())
                    .and_then(|max| usize::try_from(max).ok())
                    .unwrap_or(1),
                _ => 1,
            },
            _ => 1,
        },
        SInt::Infer => panic!("cannot compute vmax of SInt::Infer"),
    }
}

fn find_assign_identity(target: &Arc<UOp>, base: &Arc<UOp>) -> Arc<UOp> {
    let mut identity = target.clone();
    while !identity.has_buffer_identity() && identity.id != base.id {
        let sources = identity.op().sources();
        let Some(next) = sources.first() else {
            break;
        };
        identity = next.clone();
    }
    identity
}

pub mod error;
#[macro_use]
mod macros;
mod singleflight;
pub mod weight_cache;
use error::*;

pub mod activation;
pub mod arithmetic;
pub mod beam_worker;
pub mod broadcast;
pub mod conditional;
pub mod config;
pub mod data;
/// Re-export from `svod-dtype` so callers can keep using `svod_tensor::default_device`.
pub use svod_dtype::default_device;
pub mod einsum;
pub mod index;
pub mod indexing;
pub mod jit;
pub mod math;
pub mod matmul;
pub mod memory_planner;
pub mod nn;
pub mod operand;
pub mod rand;
pub mod realize;
pub mod reduce;
mod scan;
pub mod schedule;
pub(crate) mod schedule_cache;
pub mod shape_ops;
pub mod tensor_registry;
pub mod testing;
pub mod traits;
pub mod transformer;
pub mod variable;

// Re-export for public API
pub use config::{PrepareConfig, device_supports_storage_dtype};
pub use index::{Idx, IndexSpec};
pub use memory_planner::PlannerMode;
pub use operand::Operand;
pub use svod_dtype::default_device::{clear_default_device, default_device, set_default_device, with_default_device};
pub use svod_runtime::CpuBackend;
pub use tensor_registry::apply_map_to_tensors;
pub use variable::{BoundVariable, Variable};

/// Reduction operations supported by cumulative reduce (`_cumalu`).
#[derive(Debug, Clone, Copy)]
enum CumReduceOp {
    Add,
    Mul,
    #[allow(dead_code)]
    Max,
}

impl CumReduceOp {
    /// Identity element for this operation as f64, used as pad fill value.
    fn identity_value(&self, dtype: DType) -> f64 {
        match self {
            CumReduceOp::Add => 0.0,
            CumReduceOp::Mul => 1.0,
            CumReduceOp::Max => {
                if dtype.is_int() {
                    i64::MIN as f64
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }
}

/// Information about a rendered kernel.
///
/// This is the public API returned by `tensor.kernels()`.
#[derive(Clone, Debug)]
pub struct KernelInfo {
    /// Kernel name (e.g., "kernel")
    pub name: String,
    /// Generated code (LLVM IR, CUDA PTX, etc.)
    pub code: String,
    /// Entry point function name
    pub entry_point: String,
    /// Backend that generated this kernel
    pub backend: String,
}

/// Tensor represents a multi-dimensional array with lazy evaluation.
///
/// Operations like addition and multiplication build a computation graph
/// without allocating buffers. Buffers are only allocated when:
/// - Creating input tensors via `from_slice()`
/// - Evaluating the computation graph via `realize()`
///
/// # Global Graph Substitution
///
/// Tensors are registered in a global registry to support atomic graph substitution.
/// When rangeify transforms a UOp (e.g., NEG → STAGE(NEG)), all tensors
/// referencing it are updated atomically via `apply_map_to_tensors()`.
///
/// This is critical for diamond patterns (like argmin's NEG feeding both MAX and EQ)
/// where different consumers must see the same transformed version.
///
/// # Buffer Ownership (RAII)
///
/// A realized buffer is held by the registry entry the Tensor points at, so
/// every clone shares one `Arc<Buffer>`. When the last handle to an entry is
/// dropped the buffer goes with it — RAII cleanup without manual management.
///
/// # Examples
///
/// ```
/// # use svod_tensor::Tensor;
/// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
/// let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
/// let c = (&a + &b).unwrap();  // Lazy - only builds UOp graph
/// c.realize().unwrap();  // Executes the computation
/// ```
///
/// Cloning shares the entry, so a realized buffer is visible through every
/// clone — realization is recorded in the registry, not per handle.
#[derive(Clone)]
pub struct Tensor {
    /// Registry entry holding the computation graph and its realized buffer
    /// (supports global substitution).
    entry: Arc<tensor_registry::TensorEntry>,
}

/// Metadata only — never the data, which would force a device read.
impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = match self.shape() {
            Ok(shape) => {
                svod_ir::shape::to_vec_usize(&shape).map_or_else(|_| "symbolic".to_string(), |dims| format!("{dims:?}"))
            }
            Err(_) => "unknown".to_string(),
        };
        let realized = self.entry.buffer().is_some() || tensor_registry::get_buffer_arc(self.uop().base().id).is_some();
        f.debug_struct("Tensor")
            .field("shape", &format_args!("{shape}"))
            .field("dtype", &self.dtype())
            .field("device", &self.device())
            .field("realized", &realized)
            .finish()
    }
}

/// Symbolic ceiling division, mirroring tinygrad `helpers.py:63-66`.
///
/// The `(num + amt - 1) / amt` form is exact only for a non-negative numerator
/// and a positive divisor; everywhere else `-(num / -amt)` is exact for either
/// sign, because integer division floors.
pub(crate) fn ceildiv_uop(num: &Arc<UOp>, amt: &Arc<UOp>) -> Arc<UOp> {
    let nonneg =
        |value: &ConstValue| matches!(value, ConstValue::Int(v) if *v >= 0) || matches!(value, ConstValue::UInt(_));
    let positive = |value: &ConstValue| {
        matches!(value, ConstValue::Int(v) if *v > 0) || matches!(value, ConstValue::UInt(v) if *v > 0)
    };
    if nonneg(num.vmin()) && positive(amt.vmin()) {
        let one = UOp::const_(amt.dtype(), ConstValue::one(amt.dtype().base()));
        num.add(&amt.sub(&one)).floor_div(amt)
    } else {
        num.floor_div(&amt.neg()).neg()
    }
}

#[bon]
impl Tensor {
    /// Create tensor without buffer (for lazy computation graphs).
    fn new(uop: Arc<UOp>) -> Self {
        Self { entry: tensor_registry::register_tensor(uop) }
    }

    /// Create a lazy tensor from a UOp graph (no buffer allocated).
    /// Used for deferred computation graphs like ONNX weight views.
    pub fn from_lazy(uop: Arc<UOp>) -> Self {
        Self::new(uop)
    }

    /// Create a file-backed tensor using the DISK device (Tinygrad: `Tensor(pathlib.Path)`).
    /// The file is memory-mapped lazily — no data is read until the tensor is realized.
    /// The resulting tensor has dtype `uint8` and shape `(file_size,)`.
    pub fn from_path(path: &std::path::Path) -> Result<Self> {
        let file_size = std::fs::metadata(path).context(DiskSnafu { path: path.display().to_string() })?.len() as usize;
        let canonical = path.canonicalize().context(DiskSnafu { path: path.display().to_string() })?;
        let device = svod_dtype::DeviceSpec::Disk { path: canonical };
        let buffer_uop = UOp::new_buffer(device, file_size, svod_dtype::DType::Scalar(svod_dtype::ScalarDType::UInt8));
        Ok(Self::new(buffer_uop))
    }

    /// Adopt a registry entry that already carries its buffer.
    pub(crate) fn with_entry(entry: Arc<tensor_registry::TensorEntry>) -> Self {
        Self { entry }
    }

    /// Check if this tensor has zero total elements (any shape dimension is 0).
    fn has_zero_elements(&self) -> bool {
        match self.uop().shape() {
            Ok(Some(shape)) => shape.iter().any(|dim| dim.as_const() == Some(0)),
            _ => false,
        }
    }

    /// Ensure buffer is attached if the UOp has buffer identity.
    ///
    /// When `apply_map_to_tensors` substitutes a tensor's UOp with a realized
    /// BUFFER+RESHAPE, the entry's buffer isn't updated. This method looks it
    /// up in the registry and attaches it.
    pub(crate) fn ensure_buffer(&self) {
        let buffer_id = self.uop().base().id;
        if let Some(buf_arc) = tensor_registry::get_buffer_arc(buffer_id) {
            self.entry.set_buffer(buf_arc);
        }
    }

    /// Get the current UOp for this tensor.
    ///
    /// This reads from the registry, so it reflects any global substitutions.
    pub fn uop(&self) -> Arc<UOp> {
        self.entry.uop.read().clone()
    }

    /// Get kernels for THIS tensor.
    ///
    /// Note: Kernel tracking is not yet implemented with the new registry.
    /// This returns an empty list for now.
    pub fn kernels(&self) -> Vec<KernelInfo> {
        // TODO: Implement kernel tracking with the new registry
        Vec::new()
    }

    /// Create an uninitialized buffer-backed tensor with the given shape and dtype.
    ///
    /// Only the BUFFER UOp is created; its storage is bound the first time a
    /// plan is prepared over it and is then shared by every graph that
    /// references the identity. Fill it with `assign()` + `realize()`, or
    /// write through `array_view_mut` once a plan is prepared.
    /// Matches Tinygrad's `Tensor.empty(*shape)`.
    #[track_caller]
    pub fn empty(shape: &[usize], dtype: DType) -> Self {
        origin_call!("empty");
        let numel: usize = shape.iter().product();
        let buffer_uop = UOp::new_buffer(svod_dtype::default_device::default_device(), numel, dtype);
        let ir_shape = Shape::from_iter(shape.iter().map(|&d| SInt::Const(d)));
        let uop = buffer_uop.try_reshape(&ir_shape).expect("shape matches element count");
        Self::new(uop)
    }

    /// Create an uninitialized buffer-backed tensor with symbolic (dynamic) dimensions.
    ///
    /// Buffer is sized to `prod(vmax)` — each symbolic dim uses its Variable's
    /// max_val for allocation. This enables rebinding to any value in [min, max]
    /// without reallocation. Matches Tinygrad's
    /// `prod([x.vmax if isinstance(x, UOp) else x for x in shape])`.
    #[track_caller]
    pub fn empty_dynamic(shape: &[SInt], dtype: DType) -> Self {
        origin_call!("empty_dynamic");
        let numel: usize = shape.iter().map(sint_vmax).product();
        let buffer_uop = UOp::new_buffer(svod_dtype::default_device::default_device(), numel, dtype);
        let ir_shape = Shape::from_iter(shape.iter().cloned());
        let uop = buffer_uop.try_reshape(&ir_shape).expect("shape valid for reshape");
        Self::new(uop)
    }

    /// Create an empty 0-element tensor with the given dtype and shape `[0]`.
    #[track_caller]
    pub fn empty_zero(dtype: DType) -> Self {
        origin_call!("empty_zero");
        Self::empty(&[0], dtype)
    }

    /// Create a tensor filled with a constant value, broadcast to the given shape.
    #[track_caller]
    pub fn full(shape: &[usize], value: impl Into<ConstValue>, dtype: DType) -> Self {
        origin_call!("full");
        let scalar = Self::const_(value, dtype);
        if shape.is_empty() {
            return scalar;
        }
        let expand_shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
        scalar
            .try_reshape(vec![1; shape.len()])
            .and_then(|t| t.try_expand(&expand_shape))
            .expect("a scalar constant always reshapes and expands to a concrete shape")
    }

    /// Create a zero-filled tensor with the given concrete shape.
    #[track_caller]
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        origin_call!("zeros");
        Self::full(shape, ConstValue::zero(dtype.base()), dtype)
    }

    /// Create a one-filled tensor with the given concrete shape.
    #[track_caller]
    pub fn ones(shape: &[usize], dtype: DType) -> Self {
        origin_call!("ones");
        Self::full(shape, ConstValue::one(dtype.base()), dtype)
    }

    /// Create a tensor filled with a constant value, using symbolic (dynamic) dimensions.
    ///
    /// Dimensions can be concrete (`SInt::Const`) or symbolic (`SInt::Symbolic`
    /// from [`Variable::bind()`](crate::Variable::bind)).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use svod_tensor::{Tensor, Variable};
    /// use svod_dtype::DType;
    ///
    /// let batch = Variable::new("batch", 1, 32);
    /// let x = Tensor::full_dynamic(&[batch.bind(16)?.into(), 784.into()], 0.0, DType::Float32)?;
    /// ```
    #[track_caller]
    pub fn full_dynamic(shape: &[SInt], value: impl Into<ConstValue>, dtype: DType) -> Result<Self> {
        origin_call!("full_dynamic");
        let const_uop = UOp::const_(dtype.clone(), value.into());
        if shape.is_empty() {
            return Ok(Self::new(const_uop));
        }
        // Reshape scalar to [1, 1, ...] then expand to target shape.
        // Expand handles both concrete and symbolic (SInt::Symbolic) dims.
        let ones: Shape = vec![SInt::Const(1); shape.len()].into();
        let target: Shape = shape.to_vec().into();
        let reshaped = const_uop.try_reshape(&ones).context(error::UOpSnafu)?;
        let expanded = reshaped.try_expand(&target).context(error::UOpSnafu)?;
        Ok(Self::new(expanded))
    }

    /// Create a zero-filled tensor with symbolic (dynamic) dimensions.
    #[track_caller]
    pub fn zeros_dynamic(shape: &[SInt], dtype: DType) -> Result<Self> {
        origin_call!("zeros_dynamic");
        Self::full_dynamic(shape, ConstValue::zero(dtype.base()), dtype)
    }

    /// Create a one-filled tensor with symbolic (dynamic) dimensions.
    #[track_caller]
    pub fn ones_dynamic(shape: &[SInt], dtype: DType) -> Result<Self> {
        origin_call!("ones_dynamic");
        Self::full_dynamic(shape, ConstValue::one(dtype.base()), dtype)
    }

    /// Cumulative reduce along an axis using a sliding-window approach.
    ///
    /// Decomposes prefix-sum/prefix-max/prefix-prod into existing ops:
    /// pad → pool (sliding windows) → reduce. Fully lazy, O(1) graph nodes.
    fn _cumalu(&self, axis: isize, reduce: CumReduceOp) -> Result<Self> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let axis_idx = Self::normalize_axis(axis, ndim)?;
        let n = shape[axis_idx].as_const().ok_or_else(|| ErrorKind::SymbolicShapeUnsupported {
            operation: "cumsum/cumprod over a symbolic axis".to_string(),
        })?;

        if n <= 1 {
            return Ok(self.clone());
        }

        // 1. Transpose target axis to last
        let x = if axis_idx != ndim - 1 { self.try_transpose(axis_idx as isize, -1)? } else { self.clone() };

        // 2. Pad left with (n-1) identity elements
        let identity = reduce.identity_value(self.uop().dtype());
        let mut padding = vec![(0isize, 0isize); ndim];
        padding[ndim - 1] = ((n - 1) as isize, 0);
        let x = x.try_pad_value(&padding, identity)?;

        // 3. Pool with kernel=n, stride=1
        let x = x.pool(&[n], &[1], &[1])?;

        // 4. Reduce last dim
        let x = match reduce {
            CumReduceOp::Add => x.sum(-1isize)?,
            CumReduceOp::Mul => x.prod(-1isize)?,
            CumReduceOp::Max => x.max(-1isize)?,
        };

        // 5. Transpose back
        if axis_idx != ndim - 1 { x.try_transpose(axis_idx as isize, -1) } else { Ok(x) }
    }

    /// Cumulative sum along an axis.
    #[track_caller]
    pub fn cumsum(&self, axis: isize) -> Result<Self> {
        origin_call!("cumsum");
        self._cumalu(axis, CumReduceOp::Add)
    }

    /// Cumulative product along an axis.
    #[track_caller]
    pub fn cumprod(&self, axis: isize) -> Result<Self> {
        origin_call!("cumprod");
        self._cumalu(axis, CumReduceOp::Mul)
    }

    /// Create 1D tensor with evenly spaced values and explicit dtype.
    ///
    /// Matches Tinygrad's `Tensor.arange()`: `full(step) → cumsum → + (start - step)`.
    /// Accepts concrete `i64` or symbolic `Arc<UOp>` for start/stop/step.
    /// If `stop` is None, treats `start` as stop and starts from 0.
    #[builder]
    #[track_caller]
    pub fn arange_with_dtype(
        start: Arc<UOp>,
        stop: Option<Arc<UOp>>,
        dtype: DType,
        #[builder(default = UOp::const_(dtype.clone(), ConstValue::one(dtype.base())))] step: Arc<UOp>,
    ) -> Result<Self> {
        origin_call!("arange");
        let (start, stop) = match stop {
            Some(s) => (start, s),
            None => (UOp::const_(dtype.clone(), ConstValue::zero(dtype.base())), start),
        };

        let step_tensor = if let Op::Const(ConstValueHash(ConstValue::Int(start))) = start.op()
            && let Op::Const(ConstValueHash(ConstValue::Int(stop))) = stop.op()
            && let Op::Const(ConstValueHash(s @ ConstValue::Int(step))) = step.op()
        {
            let diff = stop - start;
            let ceildiv = ((diff as f64) / (*step as f64)).ceil() as i64;
            if ceildiv <= 0 {
                return Ok(Self::empty_zero(dtype));
            }

            Self::full(&[ceildiv as usize], *s, dtype.clone())
        } else {
            let ceildiv = ceildiv_uop(&stop.sub(&start), &step);
            let output_len_sint = SInt::from(ceildiv.clone());
            let ones: Shape = vec![SInt::Const(1)].into();
            let target: Shape = vec![output_len_sint].into();
            let reshaped = step.try_reshape(&ones).unwrap();
            Self::new(reshaped.try_expand(&target).unwrap())
        };

        let cumsum = step_tensor._cumalu(0, CumReduceOp::Add)?;
        let offset = Self::new(start.sub(&step));
        Ok(cumsum.try_add(&offset)?.cast(dtype))
    }

    /// Create 1D tensor with evenly spaced Int32 values.
    #[track_caller]
    pub fn arange(start: i64, stop: Option<i64>, step: Option<i64>) -> Result<Self> {
        origin_call!("arange");
        let dtype = DType::Int32;
        Self::arange_with_dtype()
            .start(UOp::const_(dtype.clone(), ConstValue::Int(start)))
            .maybe_stop(stop.map(|s| UOp::const_(dtype.clone(), ConstValue::Int(s))))
            .maybe_step(step.map(|s| UOp::const_(dtype.clone(), ConstValue::Int(s))))
            .dtype(dtype)
            .call()
    }

    /// Create 1D tensor with evenly spaced values (float parameters).
    #[track_caller]
    pub fn arange_f64(start: f64, stop: f64, step: f64, dtype: DType) -> Result<Self> {
        origin_call!("arange");
        if step == 0.0 {
            return Err(ErrorKind::SymbolicShapeUnsupported { operation: "arange with step=0".to_string() }.into());
        }
        let count = ((stop - start) / step).ceil() as i64;
        if count <= 0 {
            return Ok(Self::empty_zero(dtype));
        }
        let count = count as usize;
        let step_tensor = Self::full(&[count], ConstValue::Float(step), dtype.clone());
        let cumsum = step_tensor._cumalu(0, CumReduceOp::Add)?;
        let offset = Self::const_(ConstValue::Float(start - step), dtype.clone());
        Ok(cumsum.try_add(&offset)?.cast(dtype))
    }

    /// Create 1D tensor with `steps` evenly spaced values from `start` to `end` (inclusive).
    #[track_caller]
    pub fn linspace(start: f64, end: f64, steps: usize, dtype: DType) -> Result<Self> {
        origin_call!("linspace");
        if steps == 0 {
            return Ok(Self::empty_zero(dtype));
        }
        if steps == 1 {
            return Ok(Self::full(&[1], start, dtype));
        }
        let t = Self::arange(steps as i64, None, None)?;
        let scale = Self::const_((end - start) / (steps as f64 - 1.0), DType::Float64);
        let offset = Tensor::const_(start, DType::Float64);
        Ok(t.cast(DType::Float64).try_mul(&scale)?.try_add(&offset)?.cast(dtype))
    }

    // === Constant Constructors ===

    /// Create a scalar constant tensor.
    ///
    /// Creates a 0-dimensional tensor containing a single constant value.
    /// The constant is embedded directly in the IR and does not allocate
    /// a buffer until realized (if needed).
    ///
    /// # Arguments
    /// * `value` - The constant value (will be converted to ConstValue)
    /// * `dtype` - The data type for the tensor
    ///
    /// # Examples
    /// ```ignore
    /// // Float constant
    /// let pi = Tensor::const_(3.14159, DType::Float32);
    ///
    /// // Integer constant
    /// let forty_two = Tensor::const_(42i64, DType::Int64);
    /// ```
    #[track_caller]
    pub fn const_<T: Into<ConstValue>>(value: T, dtype: DType) -> Self {
        origin_call!("const");
        let const_val = value.into();
        let uop = UOp::const_(dtype, const_val);
        Self::new(uop)
    }

    /// Create a scalar constant tensor with dtype auto-inferred from value.
    ///
    /// Convenience method that infers dtype from the Rust type.
    ///
    /// # Examples
    /// ```ignore
    /// let f = Tensor::from_const(3.14f32);  // DType::Float32
    /// let i = Tensor::from_const(42i32);    // DType::Int32
    /// let b = Tensor::from_const(true);     // DType::Bool
    /// ```
    #[track_caller]
    pub fn from_const<T: Into<ConstValue> + HasDType>(value: T) -> Self {
        origin_call!("from_const");
        let dtype = T::DTYPE;
        Self::const_(value, dtype)
    }

    /// Element type of this tensor.
    ///
    /// # Examples
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use svod_dtype::DType;
    /// assert_eq!(Tensor::from_slice(&[1.0f32, 2.0]).dtype(), DType::Float32);
    /// ```
    pub fn dtype(&self) -> DType {
        self.uop().dtype()
    }

    /// Get device specification from underlying UOp graph.
    ///
    /// Returns the device where this tensor's data resides.
    /// For lazy tensors (not yet realized), returns the target device.
    /// Defaults to CPU if no device is found in the graph.
    ///
    /// # Examples
    /// ```ignore
    /// let cpu_tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// assert_eq!(cpu_tensor.device(), DeviceSpec::Cpu);
    /// ```
    pub fn device(&self) -> DeviceSpec {
        self.uop().device_spec().unwrap_or(DeviceSpec::Cpu)
    }

    /// Move tensor to a different device.
    ///
    /// Creates a lazy COPY operation. Data is not transferred until `realize()`.
    /// If already on target device, returns a clone (no-op).
    ///
    /// # Examples
    /// ```ignore
    /// let cpu_tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let gpu_tensor = cpu_tensor.to(DeviceSpec::Cuda { device_id: 0 });
    /// gpu_tensor.realize()?;  // Actually transfers data
    /// ```
    #[track_caller]
    pub fn to(&self, device: DeviceSpec) -> Self {
        origin_call!("to");
        if self.device() == device {
            return self.clone();
        }

        let copy_uop = self.uop().copy_to_device(device);
        Self::new(copy_uop)
    }

    /// Cast tensor to a different dtype.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let t_int = t.cast(DType::Int32);
    /// ```
    #[track_caller]
    pub fn cast(&self, dtype: svod_dtype::DType) -> Self {
        origin_call!("cast");
        Self::new(self.uop().cast(dtype))
    }

    /// Build and apply a custom UOp kernel over this tensor and additional inputs.
    ///
    /// The closure receives PARAM placeholders (as UOps) corresponding to
    /// `[self, others...]` and must return the kernel body UOp (typically a SINK).
    /// Returns tensors wrapped with AFTER(CALL) dependencies in argument order.
    pub fn custom_kernel<F>(&self, others: &[&Tensor], fxn: F) -> Result<Vec<Tensor>>
    where
        F: FnOnce(Vec<Arc<UOp>>) -> Arc<UOp>,
    {
        self.custom_kernel_with(others, CallInfo::default(), fxn)
    }

    /// `custom_kernel` with explicit CALL metadata.
    pub fn custom_kernel_with<F>(&self, others: &[&Tensor], info: CallInfo, fxn: F) -> Result<Vec<Tensor>>
    where
        F: FnOnce(Vec<Arc<UOp>>) -> Arc<UOp>,
    {
        let mut srcs: Vec<Arc<UOp>> = Vec::with_capacity(1 + others.len());
        srcs.push(self.uop());
        srcs.extend(others.iter().map(|t| t.uop()));

        let outputs = UOp::custom_kernel(srcs, fxn, info).context(UOpSnafu)?;
        Ok(outputs.into_iter().map(Self::from_lazy).collect())
    }

    /// Build a hand-written kernel as a **graph node** — the generic
    /// `custom_kernel` → `Op::Call` wrapper for any author-supplied SINK builder
    /// (the svod-tk tile DSL is one client; a raw UOp builder is another). Unlike a
    /// direct launch it returns a *lazy* output [`Tensor`] the scheduler realizes
    /// (and the JIT graph captures) like any other op, so a hand kernel composes
    /// into a model and benchmarks through the normal `prepare()` path.
    ///
    /// `out` is the output template (e.g. [`Tensor::empty`]); `ins` are the inputs.
    /// `build` receives the PARAM placeholders in `[out, ins...]` order and returns
    /// the kernel body SINK — it must emit its own `Op::Special` launch dims and a
    /// finished-kernel marker (`KernelInfo.opts_to_apply = Some(_)`) so the
    /// optimizer leaves the hand-lowered body alone. Returns the single lazy output.
    pub fn graph_kernel<F>(name: &str, out: Tensor, ins: &[&Tensor], build: F) -> Result<Tensor>
    where
        F: FnOnce(Vec<Arc<UOp>>) -> Arc<UOp>,
    {
        let info = CallInfo { name: Some(name.to_string()), ..CallInfo::default() };
        let outputs = out.custom_kernel_with(ins, info, build)?;
        // `custom_kernel` returns one output per source in `[out, ins...]` order;
        // slot 0 is the kernel's output (`custom_kernel_with` always pushes `out`).
        Ok(outputs.into_iter().next().expect("custom_kernel returns the output tensor"))
    }

    /// Bitcast tensor to a different dtype, reinterpreting bits.
    ///
    /// For equal-itemsize dtypes (e.g. `f32 ↔ i32`) this is the pure
    /// IR-level reinterpretation. For different-itemsize dtypes (e.g.
    /// `u32 → u16` or `u32 → u64`) the last axis is split or combined via
    /// shifts + reshape, matching Tinygrad's `tensor.py::bitcast`. The total
    /// byte count is preserved; the last axis grows (`src_size > dst_size`)
    /// or shrinks (`src_size < dst_size`) by `rate = max(...)/min(...)`.
    ///
    /// Requires:
    /// - source and destination are both scalar (vector dtypes unsupported);
    /// - `(shape[-1] * src_size)` divides evenly by `dst_size`;
    /// - the last shape dim is concrete (not symbolic).
    #[track_caller]
    pub fn bitcast(&self, dtype: svod_dtype::DType) -> Result<Self> {
        origin_call!("bitcast");
        let src_dt = self.uop().dtype();
        let src_scalar = src_dt.scalar().ok_or_else(|| ErrorKind::SymbolicShapeUnsupported {
            operation: "bitcast: non-scalar source dtype".to_string(),
        })?;
        let dst_scalar = dtype.scalar().ok_or_else(|| ErrorKind::SymbolicShapeUnsupported {
            operation: "bitcast: non-scalar destination dtype".to_string(),
        })?;
        let src_size = src_scalar.bytes();
        let dst_size = dst_scalar.bytes();

        if src_size == dst_size {
            return Ok(Self::new(self.uop().bitcast(dtype)));
        }

        let shape = self.shape()?;
        let last_dim = shape.last().and_then(|s| s.as_const()).ok_or_else(|| ErrorKind::SymbolicShapeUnsupported {
            operation: "bitcast with size change on symbolic last dim".to_string(),
        })?;
        if last_dim * src_size % dst_size != 0 {
            return Err(ErrorKind::ReshapeSizeMismatch {
                operation: format!(
                    "bitcast {src_scalar:?}({src_size}B) → {dst_scalar:?}({dst_size}B): \
                     last dim {last_dim} × {src_size} not divisible by {dst_size}"
                ),
            }
            .into());
        }

        let src_uint = DType::Scalar(uint_for_bytes(src_size));
        let dst_uint = DType::Scalar(uint_for_bytes(dst_size));

        // Reinterpret as the source-sized uint first (always equal-size, falls
        // into the identity path above).
        let tmp = if src_dt == src_uint { self.clone() } else { Self::new(self.uop().bitcast(src_uint.clone())) };

        let result = if dst_size > src_size {
            // Combine `rate` source words into one dst word: shift each by
            // `8*i*src_size`, OR them, squeeze the trailing axis.
            let rate = dst_size / src_size;
            let mut new_shape: Vec<isize> = svod_ir::shape::to_vec_isize(&shape).context(UOpSnafu)?;
            let last_idx = new_shape.len() - 1;
            new_shape[last_idx] = (last_dim / rate) as isize;
            new_shape.push(rate as isize);
            let reshaped = tmp.try_reshape(&new_shape)?;

            let mut acc: Option<Tensor> = None;
            for i in 0..rate {
                // Slice the trailing axis to `(i, i+1)` (preserves rank).
                let mut shrink_ranges: Vec<Option<(isize, isize)>> =
                    std::iter::repeat_n(None, new_shape.len() - 1).collect();
                shrink_ranges.push(Some((i as isize, (i + 1) as isize)));
                let slice = reshaped.try_shrink(shrink_ranges)?;
                let widened = slice.cast(dst_uint.clone());
                let shift_amount = 8 * i * src_size;
                let term = if shift_amount == 0 {
                    widened
                } else {
                    let shift_t = Tensor::full(
                        &svod_ir::shape::to_vec_usize(&widened.shape()?).context(UOpSnafu)?,
                        ConstValue::UInt(shift_amount as u64),
                        dst_uint.clone(),
                    );
                    widened.try_shl(&shift_t)?
                };
                acc = Some(match acc {
                    None => term,
                    Some(a) => a.try_bitor(&term)?,
                });
            }
            let summed = acc.expect("rate >= 1");
            // Squeeze the trailing axis (now size 1).
            summed.try_squeeze(Some(-1))?
        } else {
            // Split each source word into `rate` dst words via right shifts,
            // stack along a new trailing axis, then flatten the last two.
            let rate = src_size / dst_size;
            let mut shifted: Vec<Tensor> = Vec::with_capacity(rate);
            for i in 0..rate {
                let shift_amount = 8 * i * dst_size;
                let s = if shift_amount == 0 {
                    tmp.clone()
                } else {
                    let shift_t = Tensor::full(
                        &svod_ir::shape::to_vec_usize(&tmp.shape()?).context(UOpSnafu)?,
                        ConstValue::UInt(shift_amount as u64),
                        src_uint.clone(),
                    );
                    tmp.try_shr(&shift_t)?
                };
                shifted.push(s);
            }
            let refs: Vec<&Tensor> = shifted.iter().collect();
            let stacked = Tensor::stack(&refs, -1)?;
            // Collapse trailing two axes (... × last × rate) → (... × last*rate).
            let stacked_shape = stacked.shape()?;
            let nd = stacked_shape.len();
            let mut new_shape: Vec<isize> = svod_ir::shape::to_vec_isize(&stacked_shape).context(UOpSnafu)?;
            let trailing = new_shape[nd - 2] * new_shape[nd - 1];
            new_shape.truncate(nd - 2);
            new_shape.push(trailing);
            let flat = stacked.try_reshape(&new_shape)?;
            flat.cast(dst_uint.clone())
        };

        // Final reinterpretation at equal size (e.g. u16 → f16).
        if result.uop().dtype() == dtype { Ok(result) } else { Ok(Self::new(result.uop().bitcast(dtype))) }
    }
}

fn uint_for_bytes(n: usize) -> svod_dtype::ScalarDType {
    use svod_dtype::ScalarDType;
    match n {
        1 => ScalarDType::UInt8,
        2 => ScalarDType::UInt16,
        4 => ScalarDType::UInt32,
        8 => ScalarDType::UInt64,
        _ => panic!("uint_for_bytes: unsupported byte size {n}"),
    }
}

#[allow(dead_code)]
impl Tensor {
    /// Assign a value tensor to this tensor in-place.
    ///
    /// Embeds the write as `AFTER(target, STORE(target, value))`: a pending
    /// store that `realize()` lands in the target's storage, keeping its
    /// identity. Dependents built earlier read that identity, so they see the
    /// value once the assign is realized (and the old one if they realize
    /// first) — the same ordering as Tinygrad.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let placeholder = Tensor::empty(&[2, 3], DType::Float32);
    /// let real_data = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///     .try_reshape(&[2, 3]).unwrap();
    /// placeholder.assign(&real_data);
    /// ```
    #[track_caller]
    pub fn try_assign(&self, value: &Tensor) -> Result<()> {
        origin_call!("assign");
        let target_uop = self.uop();
        if self.device().is_disk() {
            return Err(ErrorKind::IrConstruction {
                details: "assign to DISK tensors is not supported by Svod runtime".to_string(),
            }
            .into());
        }

        let target_shape = self.shape()?;
        let value_shape = value.shape()?;
        let value = if target_shape != value_shape { value.broadcast_to(&target_shape)? } else { value.clone() };
        if self.device() != value.device() {
            return Err(ErrorKind::IrConstruction {
                details: format!("assign device mismatch {:?} != {:?}", self.device(), value.device()),
            }
            .into());
        }

        let target_dtype = target_uop.dtype();
        let value_dtype = value.uop().dtype();
        if target_dtype != value_dtype {
            return Err(ErrorKind::TypeMismatch { expected: target_dtype, actual: value_dtype }.into());
        }

        let value_uop = value.uop();
        if Arc::ptr_eq(&target_uop, &value_uop) {
            return Ok(());
        }

        let assign_effect = target_uop.after(smallvec![target_uop.store(value_uop)]);
        let base = target_uop.base();
        if matches!(base.op(), Op::Buffer(..) | Op::After(..))
            && target_uop.id != base.id
            && !target_uop.has_buffer_identity()
        {
            let identity = find_assign_identity(&target_uop, &base);
            let assigned_identity = identity.after(smallvec![assign_effect]);
            let mut becomes_map = HashMap::new();
            becomes_map.insert(UOpKey(identity), assigned_identity);
            // Walk semantics required: replacement contains the original key
            // (`After(Buffer, [...])` wraps `Buffer`). A re-traversing rewrite
            // would loop or wrap the buffer multiple times.
            tensor_registry::apply_map_to_tensors_walk(&becomes_map);
        } else {
            self.set_uop(assign_effect);
        }
        Ok(())
    }

    #[track_caller]
    pub fn assign(&self, value: &Tensor) {
        origin_call!("assign");
        self.try_assign(value).expect("tensor assign failed");
    }

    /// Update the UOp for this tensor directly.
    ///
    /// This is used internally after realization to update the tensor's UOp
    /// to point to the materialized buffer.
    pub(crate) fn set_uop(&self, uop: Arc<UOp>) {
        *self.entry.uop.write() = uop;
    }

    /// Ensure this tensor has contiguous memory layout.
    ///
    /// Creates a CONTIGUOUS UOp that forces materialization when realized.
    /// Following Tinygrad's approach, calling `.contiguous().realize()` on
    /// a pure constant tensor will create an actual buffer.
    ///
    /// # Examples
    /// ```ignore
    /// // Force a constant to be materialized
    /// let c = Tensor::const_(5.0f32, DType::Float32).contiguous();
    /// c.realize()?;
    /// assert!(c.buffer().is_some());
    /// ```
    #[track_caller]
    pub fn contiguous(&self) -> Self {
        origin_call!("contiguous");
        let uop = self.uop();
        if matches!(uop.op(), svod_ir::Op::Contiguous(..)) {
            return self.clone();
        }
        let contiguous_uop = uop.contiguous();
        Self::new(contiguous_uop)
    }
}

impl Tensor {
    /// Helper to broadcast a scalar constant to match this tensor's shape.
    pub(crate) fn broadcast_scalar(&self, value: ConstValue) -> Result<Self> {
        let shape = self.shape()?;
        let scalar = Self::new(UOp::const_(self.uop().dtype(), value));
        scalar.broadcast_to(&shape)
    }

    /// Broadcast a dtype-aware zero to match this tensor's shape.
    #[track_caller]
    pub fn zero(&self) -> Result<Self> {
        origin_call!("zero");
        let sdtype = self.uop().dtype().scalar().expect("scalar dtype");
        self.broadcast_scalar(ConstValue::zero(sdtype))
    }

    /// Broadcast a dtype-aware one to match this tensor's shape.
    #[track_caller]
    pub fn one(&self) -> Result<Self> {
        origin_call!("one");
        let sdtype = self.uop().dtype().scalar().expect("scalar dtype");
        self.broadcast_scalar(ConstValue::one(sdtype))
    }

    /// Identity matrix of shape `[n, m]` with the given dtype.
    #[track_caller]
    pub fn eye(n: usize, m: usize, dtype: DType) -> Result<Self> {
        origin_call!("eye");
        let rows = Self::arange(n as i64, None, None)?.try_unsqueeze(-1)?;
        let cols = Self::arange(m as i64, None, None)?;
        Ok(rows.try_eq(&cols)?.cast(dtype))
    }
}

#[bon]
impl Tensor {
    /// Cumulative sum with exclusive and reverse options.
    #[builder]
    #[track_caller]
    pub fn cumsum_with(
        &self,
        axis: isize,
        #[builder(default = false)] exclusive: bool,
        #[builder(default = false)] reverse: bool,
    ) -> Result<Self> {
        origin_call!("cumsum");
        let shape = self.shape()?;
        let ndim = shape.len();
        let axis_idx = Self::normalize_axis(axis, ndim)?;
        let mut result = self.clone();
        if reverse {
            result = result.flip(&[axis_idx as isize])?;
        }
        if exclusive {
            let dim_size = shape[axis_idx]
                .as_const()
                .context(SymbolicShapeUnsupportedSnafu { operation: "exclusive cumsum over a symbolic axis" })?;
            let mut pad_spec: Vec<(isize, isize)> = vec![(0, 0); ndim];
            pad_spec[axis_idx] = (1, 0);
            result = result.try_pad(&pad_spec)?;
            // `None` keeps a dim whole, so only the (concrete) cum axis is sliced
            // and every other dim may stay symbolic.
            let mut shrink_spec: Vec<Option<(SInt, SInt)>> = vec![None; ndim];
            shrink_spec[axis_idx] = Some((SInt::Const(0), SInt::Const(dim_size)));
            result = result.try_shrink(shrink_spec)?;
        }
        result = result.cumsum(axis_idx as isize)?;
        if reverse {
            result = result.flip(&[axis_idx as isize])?;
        }
        Ok(result)
    }

    /// Cumulative product with exclusive and reverse options.
    #[builder]
    #[track_caller]
    pub fn cumprod_with(
        &self,
        axis: isize,
        #[builder(default = false)] exclusive: bool,
        #[builder(default = false)] reverse: bool,
    ) -> Result<Self> {
        origin_call!("cumprod");
        let shape = self.shape()?;
        let ndim = shape.len();
        let axis_idx = Self::normalize_axis(axis, ndim)?;
        let mut result = self.clone();
        if reverse {
            result = result.flip(&[axis_idx as isize])?;
        }
        if exclusive {
            let dim_size = shape[axis_idx]
                .as_const()
                .context(SymbolicShapeUnsupportedSnafu { operation: "exclusive cumprod over a symbolic axis" })?;
            let mut pad_spec: Vec<(isize, isize)> = vec![(0, 0); ndim];
            pad_spec[axis_idx] = (1, 0);
            result = result.try_pad_value(&pad_spec, 1.0)?;
            // `None` keeps a dim whole, so only the (concrete) cum axis is sliced
            // and every other dim may stay symbolic.
            let mut shrink_spec: Vec<Option<(SInt, SInt)>> = vec![None; ndim];
            shrink_spec[axis_idx] = Some((SInt::Const(0), SInt::Const(dim_size)));
            result = result.try_shrink(shrink_spec)?;
        }
        result = result.cumprod(axis_idx as isize)?;
        if reverse {
            result = result.flip(&[axis_idx as isize])?;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod test;
