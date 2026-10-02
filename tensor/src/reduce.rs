//! Reduction operations for tensors.
//!
//! This module provides reduction operations like sum, max, min, prod, and mean
//! with ergonomic APIs that match PyTorch/NumPy conventions.

use bon::bon;
use snafu::{OptionExt, ResultExt};
use svod_dtype::{DType, ScalarDType};
use svod_ir::{ReduceOp, SInt, UOp};

use crate::{
    ErrorKind, Result, Tensor,
    error::{SymbolicShapeUnsupportedSnafu, UOpSnafu},
};

/// Specification for reduction axes.
///
/// Supports:
/// - All axes: `AxisSpec::All` (from `()`)
/// - Single axis: `AxisSpec::Single(0)` (from `isize`)
/// - Multiple axes: `AxisSpec::Multiple(vec![0, 2])` (from `&[isize]` or `Vec<isize>`)
#[derive(Debug, Clone)]
pub enum AxisSpec {
    /// Reduce all axes (produces scalar).
    All,
    /// Reduce a single axis (supports negative indexing).
    Single(isize),
    /// Reduce multiple axes (each supports negative indexing).
    Multiple(Vec<isize>),
}

// Ergonomic Into conversions for AxisSpec
impl From<()> for AxisSpec {
    fn from(_: ()) -> Self {
        Self::All
    }
}

impl From<isize> for AxisSpec {
    fn from(axis: isize) -> Self {
        Self::Single(axis)
    }
}

impl From<&[isize]> for AxisSpec {
    fn from(axes: &[isize]) -> Self {
        Self::Multiple(axes.to_vec())
    }
}

impl From<Vec<isize>> for AxisSpec {
    fn from(axes: Vec<isize>) -> Self {
        Self::Multiple(axes)
    }
}

// =========================================================================
// Tensor Reduction Methods
// =========================================================================

impl Tensor {
    /// Resolve axis specification to normalized axis indices.
    ///
    /// Handles:
    /// - `AxisSpec::All` → all axes (0..ndim)
    /// - Single/multiple axes → normalize negative indices
    /// - Deduplication
    /// - Bounds checking
    pub(crate) fn resolve_axis_spec(spec: &AxisSpec, ndim: usize) -> Result<Vec<usize>> {
        match spec {
            AxisSpec::All => Ok((0..ndim).collect()),
            AxisSpec::Single(axis) => {
                let normalized = Self::normalize_axis(*axis, ndim)?;
                Ok(vec![normalized])
            }
            AxisSpec::Multiple(axes) => {
                let mut normalized: Vec<usize> =
                    axes.iter().map(|&axis| Self::normalize_axis(axis, ndim)).collect::<Result<_>>()?;

                // Deduplicate axes (keep first occurrence)
                normalized.sort_unstable();
                normalized.dedup();

                Ok(normalized)
            }
        }
    }

    /// Get accumulation dtype for sum operations (Tinygrad-compatible).
    ///
    /// Used when `promote=true` in reduction builders.
    ///
    /// Promotion rules:
    /// - int8, int16 → int32
    /// - int32, int64 → preserve
    /// - uint8, uint16 → uint32
    /// - uint32, uint64 → preserve
    /// - float16, bfloat16 → float32 (for accumulation)
    /// - float32, float64 → preserve
    /// - bool → int32
    pub(crate) fn sum_acc_dtype(dtype: &DType) -> DType {
        use ScalarDType::*;
        let Some(scalar) = dtype.scalar() else {
            return dtype.clone();
        };

        match scalar {
            Bool => DType::Int32,
            WeakInt | Int8 | Int16 => DType::Int32,
            Int32 | Int64 => dtype.clone(),
            UInt8 | UInt16 => DType::UInt32,
            UInt32 | UInt64 => dtype.clone(),
            WeakFloat | Float16 | BFloat16 | FP8E4M3 | FP8E4M3FNUZ | FP8E5M2 | FP8E5M2FNUZ => DType::Float32,
            Float32 | Float64 => dtype.clone(),
            Void | Index => dtype.clone(),
        }
    }

    /// Check if dtype should be cast back after sum accumulation.
    ///
    /// Tinygrad casts back to the original dtype for exactly the narrow floats.
    fn should_cast_back_after_sum(dtype: &DType) -> bool {
        dtype.is_narrow_float()
    }

    /// Check if dtype is an integer or bool type.
    fn is_integer_dtype(dtype: &DType) -> bool {
        dtype.is_int() || matches!(dtype.scalar(), Some(ScalarDType::Bool))
    }
}

#[bon]
impl Tensor {
    /// Sum of tensor elements over given axes.
    ///
    /// Auto-promotes accumulation dtype (bool→int32, float16→float32) like Tinygrad.
    /// Use `sum_with().promote(false)` to preserve input dtype.
    #[track_caller]
    pub fn sum(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        origin_call!("sum");
        reduce_internal(self, ReduceOp::Add, axes.into(), false, None, true)
    }

    /// Sum with additional options (keepdim, dtype, promote).
    ///
    /// Accumulation dtype, in precedence order: an explicit `dtype`, else
    /// `promote` (default: on, matching [`sum`](Self::sum) and Tinygrad), else
    /// the input dtype. Passing both `dtype` and `promote(true)` is an error.
    ///
    /// # Examples
    /// ```ignore
    /// // Explicit dtype
    /// tensor.sum_with(0).dtype(DType::Float32).call()?;
    ///
    /// // Opt out of promotion (int8 accumulates in int8)
    /// tensor.sum_with(0).promote(false).call()?;
    ///
    /// // With keepdim
    /// tensor.sum_with(0).keepdim(true).call()?;
    /// ```
    #[builder]
    #[track_caller]
    pub fn sum_with(
        &self,
        axes: impl Into<AxisSpec>,
        #[builder(default = false)] keepdim: bool,
        dtype: Option<DType>,
        promote: Option<bool>,
    ) -> Result<Self> {
        origin_call!("sum");
        if dtype.is_some() && promote == Some(true) {
            return Err(ErrorKind::ConflictingReductionOptions.into());
        }
        let promote = promote.unwrap_or(dtype.is_none());
        reduce_internal(self, ReduceOp::Add, axes.into(), keepdim, dtype, promote)
    }

    /// Product of tensor elements over given axes.
    ///
    /// Preserves input dtype. Use `prod_with().promote(true)` or `.dtype(...)` for different accumulation.
    #[track_caller]
    pub fn prod(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        origin_call!("prod");
        reduce_internal(self, ReduceOp::Mul, axes.into(), false, None, false)
    }

    /// Product with additional options (keepdim, dtype, promote).
    ///
    /// Unlike [`sum_with`](Self::sum_with), `promote` defaults to `false`: a
    /// product preserves the input dtype unless asked otherwise, matching
    /// [`prod`](Self::prod) and Tinygrad.
    #[builder]
    #[track_caller]
    pub fn prod_with(
        &self,
        axes: impl Into<AxisSpec>,
        #[builder(default = false)] keepdim: bool,
        dtype: Option<DType>,
        #[builder(default = false)] promote: bool,
    ) -> Result<Self> {
        origin_call!("prod");
        reduce_internal(self, ReduceOp::Mul, axes.into(), keepdim, dtype, promote)
    }

    /// Maximum of tensor elements over given axes.
    ///
    /// Always preserves input dtype.
    #[track_caller]
    pub fn max(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        origin_call!("max");
        reduce_internal(self, ReduceOp::Max, axes.into(), false, None, false)
    }

    /// Maximum with keepdim option.
    #[builder]
    #[track_caller]
    pub fn max_with(&self, axes: impl Into<AxisSpec>, #[builder(default = false)] keepdim: bool) -> Result<Self> {
        origin_call!("max");
        reduce_internal(self, ReduceOp::Max, axes.into(), keepdim, None, false)
    }

    /// Minimum of tensor elements over given axes.
    ///
    /// Always preserves input dtype.
    #[track_caller]
    pub fn min(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        origin_call!("min");
        reduce_internal(self, ReduceOp::Min, axes.into(), false, None, false)
    }

    /// Minimum with keepdim option.
    #[builder]
    #[track_caller]
    pub fn min_with(&self, axes: impl Into<AxisSpec>, #[builder(default = false)] keepdim: bool) -> Result<Self> {
        origin_call!("min");
        reduce_internal(self, ReduceOp::Min, axes.into(), keepdim, None, false)
    }

    /// Mean of tensor elements over given axes.
    ///
    /// For integer inputs, automatically uses float32 accumulation.
    /// For float inputs, preserves input dtype.
    #[track_caller]
    pub fn mean(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        origin_call!("mean");
        mean_impl(self, axes.into(), false)
    }

    /// Mean with keepdim option.
    #[builder]
    #[track_caller]
    pub fn mean_with(&self, axes: impl Into<AxisSpec>, #[builder(default = false)] keepdim: bool) -> Result<Self> {
        origin_call!("mean");
        mean_impl(self, axes, keepdim)
    }

    /// Variance of tensor elements over given axes.
    ///
    /// Computes unbiased sample variance (divides by N-1).
    /// For integer inputs, automatically uses float32 accumulation.
    /// For float inputs, preserves input dtype.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let v = t.var(())?;  // Variance over all elements
    /// ```
    #[track_caller]
    pub fn var(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        origin_call!("var");
        var_impl(self, axes.into(), false, 1)
    }

    /// Variance with keepdim and correction options.
    ///
    /// `correction` is subtracted from the element count in the divisor (Bessel's
    /// correction): `correction=1` (default) is the unbiased sample variance,
    /// `correction=0` the population variance.
    #[builder]
    #[track_caller]
    pub fn var_with(
        &self,
        axes: impl Into<AxisSpec>,
        #[builder(default = false)] keepdim: bool,
        #[builder(default = 1)] correction: i64,
    ) -> Result<Self> {
        origin_call!("var");
        var_impl(self, axes.into(), keepdim, correction)
    }

    /// Standard deviation of tensor elements over given axes.
    ///
    /// Computes unbiased sample standard deviation (divides by N-1).
    /// For integer inputs, automatically uses float32 accumulation.
    /// For float inputs, preserves input dtype.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let s = t.std(())?;  // Std dev over all elements
    /// ```
    #[track_caller]
    pub fn std(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        origin_call!("std");
        std_impl(self, axes.into(), false, 1)
    }

    /// Standard deviation with keepdim and correction options.
    ///
    /// `correction` is subtracted from the element count in the divisor:
    /// `correction=1` (default) is the unbiased sample std, `correction=0` the
    /// population std.
    #[builder]
    #[track_caller]
    pub fn std_with(
        &self,
        axes: impl Into<AxisSpec>,
        #[builder(default = false)] keepdim: bool,
        #[builder(default = 1)] correction: i64,
    ) -> Result<Self> {
        origin_call!("std");
        std_impl(self, axes.into(), keepdim, correction)
    }

    /// Variance and mean of tensor elements over given axes.
    ///
    /// Returns (variance, mean) tuple. More efficient than computing separately.
    /// Computes unbiased sample variance (divides by N-1).
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let (v, m) = t.var_mean(())?;
    /// ```
    #[track_caller]
    pub fn var_mean(&self, axes: impl Into<AxisSpec>) -> Result<(Self, Self)> {
        origin_call!("var_mean");
        var_mean_impl(self, axes.into(), false, 1)
    }

    /// Variance and mean with keepdim and correction options (see [`var_with`]).
    ///
    /// [`var_with`]: Tensor::var_with
    #[builder]
    #[track_caller]
    pub fn var_mean_with(
        &self,
        axes: impl Into<AxisSpec>,
        #[builder(default = false)] keepdim: bool,
        #[builder(default = 1)] correction: i64,
    ) -> Result<(Self, Self)> {
        origin_call!("var_mean");
        var_mean_impl(self, axes.into(), keepdim, correction)
    }

    /// Standard deviation and mean of tensor elements over given axes.
    ///
    /// Returns (std, mean) tuple. More efficient than computing separately.
    /// Computes unbiased sample standard deviation (divides by N-1).
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let (s, m) = t.std_mean(())?;
    /// ```
    #[track_caller]
    pub fn std_mean(&self, axes: impl Into<AxisSpec>) -> Result<(Self, Self)> {
        origin_call!("std_mean");
        std_mean_impl(self, axes.into(), false, 1)
    }

    /// Standard deviation and mean with keepdim and correction options (see [`var_with`]).
    ///
    /// [`var_with`]: Tensor::var_with
    #[builder]
    #[track_caller]
    pub fn std_mean_with(
        &self,
        axes: impl Into<AxisSpec>,
        #[builder(default = false)] keepdim: bool,
        #[builder(default = 1)] correction: i64,
    ) -> Result<(Self, Self)> {
        origin_call!("std_mean");
        std_mean_impl(self, axes.into(), keepdim, correction)
    }

    /// Internal helper: inverse of tensor for argmin.
    ///
    /// - Float dtypes: -self
    /// - Integer dtypes: ~self (bitwise NOT)
    /// - Bool dtype: logical_not(self)
    fn inverse(&self) -> Result<Self> {
        let dtype = self.uop().dtype();
        if dtype.is_float() {
            Ok(self.neg())
        } else if dtype.is_int() {
            self.bitwise_not()
        } else if matches!(dtype.scalar(), Some(ScalarDType::Bool)) {
            self.logical_not()
        } else {
            Ok(self.clone()) // Fallback for other types
        }
    }
}

// =========================================================================
// Argmax / Argmin Operations
// =========================================================================

#[bon]
impl Tensor {
    /// Index of maximum value along axis.
    ///
    /// Returns int32 tensor with indices of maximum values.
    /// For ties, returns the index of the first occurrence.
    ///
    /// # Arguments
    /// * `axis` - Axis to reduce (None = flatten first)
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[[1.0, 3.0, 2.0], [4.0, 2.0, 5.0]]);
    /// t.argmax(None)?;      // 5 (flattened: max is at index 5)
    /// t.argmax(Some(0))?;   // [1, 0, 1] (row indices of max per column)
    /// t.argmax(Some(1))?;   // [1, 2] (column indices of max per row)
    /// ```
    #[track_caller]
    pub fn argmax(&self, axis: impl Into<Option<isize>>) -> Result<Self> {
        origin_call!("argmax");
        argmax_impl(self, axis.into(), false)
    }

    /// Argmax with keepdim option.
    #[builder]
    #[track_caller]
    pub fn argmax_with(
        &self,
        axis: impl Into<Option<isize>>,
        #[builder(default = false)] keepdim: bool,
    ) -> Result<Self> {
        origin_call!("argmax");
        argmax_impl(self, axis.into(), keepdim)
    }

    /// Hard maximum: one-hot encoding of the argmax along an axis.
    ///
    /// Returns a tensor of the same shape with 1.0 at the position of the
    /// maximum value along `axis` and 0.0 elsewhere, cast to the input dtype.
    #[track_caller]
    pub fn hardmax(&self, axis: isize) -> Result<Self> {
        origin_call!("hardmax");
        let shape = self.shape()?;
        let ndim = shape.len();
        let norm_axis = Self::normalize_axis(axis, ndim)?;
        let axis_size = shape[norm_axis].as_const().ok_or_else(|| {
            crate::error::ErrorKind::SymbolicShapeUnsupported { operation: format!("hardmax axis {norm_axis}") }
        })?;
        Ok(self
            .argmax_with()
            .axis(Some(axis))
            .keepdim(false)
            .call()?
            .try_unsqueeze(axis)?
            .one_hot_along_dim(axis_size, axis)?
            .cast(self.uop().dtype()))
    }

    /// Index of minimum value along axis.
    ///
    /// Returns int32 tensor with indices of minimum values.
    /// For ties, returns the index of the first occurrence.
    #[track_caller]
    pub fn argmin(&self, axis: impl Into<Option<isize>>) -> Result<Self> {
        origin_call!("argmin");
        argmin_impl(self, axis.into(), false)
    }

    /// Argmin with keepdim option.
    #[builder]
    #[track_caller]
    pub fn argmin_with(
        &self,
        axis: impl Into<Option<isize>>,
        #[builder(default = false)] keepdim: bool,
    ) -> Result<Self> {
        origin_call!("argmin");
        argmin_impl(self, axis.into(), keepdim)
    }

    /// Test if any element is true along axes.
    ///
    /// Logical OR reduction. Returns bool dtype.
    /// Non-zero values are treated as true.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[[true, false], [false, false]]);
    /// t.any(())?;           // true (any element is true)
    /// t.any(0)?;            // [true, false] (any true per column)
    /// t.any(1)?;            // [true, false] (any true per row)
    /// ```
    #[track_caller]
    pub fn any(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        origin_call!("any");
        any_impl(self, axes.into(), false)
    }

    /// Any with keepdim option.
    #[builder]
    #[track_caller]
    pub fn any_with(&self, axes: impl Into<AxisSpec>, #[builder(default = false)] keepdim: bool) -> Result<Self> {
        origin_call!("any");
        any_impl(self, axes.into(), keepdim)
    }

    /// Test if all elements are true along axes.
    ///
    /// Logical AND reduction. Returns bool dtype.
    /// Non-zero values are treated as true.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[[true, true], [true, false]]);
    /// t.all(())?;           // false (not all elements are true)
    /// t.all(0)?;            // [true, false] (all true per column)
    /// t.all(1)?;            // [true, false] (all true per row)
    /// ```
    #[track_caller]
    pub fn all(&self, axes: impl Into<AxisSpec>) -> Result<Self> {
        origin_call!("all");
        all_impl(self, axes.into(), false)
    }

    /// All with keepdim option.
    #[builder]
    #[track_caller]
    pub fn all_with(&self, axes: impl Into<AxisSpec>, #[builder(default = false)] keepdim: bool) -> Result<Self> {
        origin_call!("all");
        all_impl(self, axes.into(), keepdim)
    }
}

/// Internal argmax implementation.
///
/// Only the reduced axis has to be concrete — the tie-break needs a descending
/// index ramp `[N, N-1, .., 1]` of exactly that extent. Every other axis is
/// carried as an `SInt`, so a symbolic batch flows through untouched.
fn argmax_impl(tensor: &Tensor, axis: Option<isize>, keepdim: bool) -> Result<Tensor> {
    // `axis = None` folds *every* axis into the reduced one, so all of them must
    // be concrete for the ramp above to exist.
    let (working_tensor, working_axis) = match axis {
        Some(ax) => (tensor.clone(), ax),
        None => {
            snafu::ensure!(
                tensor.shape()?.iter().all(|dim| dim.is_const()),
                SymbolicShapeUnsupportedSnafu { operation: "argmax/argmin over a flattened symbolic shape" }
            );
            (tensor.flatten()?, 0)
        }
    };

    let shape = working_tensor.shape()?;
    let normalized_axis = Tensor::normalize_axis(working_axis, shape.len())?;
    let axis_size = shape[normalized_axis]
        .as_const()
        .context(SymbolicShapeUnsupportedSnafu { operation: "argmax/argmin over a symbolic axis" })?;

    // Mask of the positions holding the per-axis maximum.
    let max_vals = working_tensor.max_with().axes(working_axis).keepdim(true).call()?.try_expand(shape.clone())?;
    let mask = working_tensor.try_eq(&max_vals)?.cast(DType::Int32);

    // Descending ramp [N, N-1, .., 1] laid along `normalized_axis`, so the
    // largest masked value — and thus the max below — is the *first* occurrence.
    let mut ramp_shape: Vec<SInt> = vec![SInt::Const(1); shape.len()];
    ramp_shape[normalized_axis] = SInt::Const(axis_size);
    let ramp = Tensor::arange(axis_size as i64, Some(0), Some(-1))?.try_reshape(ramp_shape)?.try_expand(shape)?;

    let max_idx = mask.try_mul(&ramp)?.max_with().axes(working_axis).keepdim(keepdim).call()?;

    // Invert the ramp: N - max_idx is the actual index.
    let n = Tensor::new(UOp::const_(DType::Int32, svod_ir::ConstValue::Int(axis_size as i64)));
    Ok(n.broadcast_to(&max_idx.shape()?)?.try_sub(&max_idx)?.cast(DType::Int32))
}

/// Internal argmin implementation.
fn argmin_impl(tensor: &Tensor, axis: Option<isize>, keepdim: bool) -> Result<Tensor> {
    // Argmin is just argmax of inverted values
    let inverted = tensor.inverse()?;
    argmax_impl(&inverted, axis, keepdim)
}

/// Internal any implementation.
fn any_impl(tensor: &Tensor, axes: AxisSpec, keepdim: bool) -> Result<Tensor> {
    // Cast to bool (non-zero becomes true)
    let as_bool = tensor.cast(DType::Bool);

    // Max reduction on bool is logical OR
    reduce_internal(&as_bool, ReduceOp::Max, axes, keepdim, None, false)
}

/// Internal all implementation.
fn all_impl(tensor: &Tensor, axes: AxisSpec, keepdim: bool) -> Result<Tensor> {
    // De Morgan's law: all(x) = !any(!x)
    let negated = tensor.logical_not()?;
    let any_negated = any_impl(&negated, axes, keepdim)?;
    any_negated.logical_not()
}

/// Internal reduction implementation.
#[track_caller]
fn reduce_internal(
    tensor: &Tensor,
    op: ReduceOp,
    axes: AxisSpec,
    keepdim: bool,
    dtype: Option<DType>,
    promote: bool,
) -> Result<Tensor> {
    // Validate conflicting options
    if dtype.is_some() && promote {
        return Err(ErrorKind::ConflictingReductionOptions.into());
    }

    let shape = tensor.shape()?;
    let resolved_axes = Tensor::resolve_axis_spec(&axes, shape.len())?;

    // Determine accumulation dtype
    let original_dtype = tensor.uop().dtype();
    let acc_dtype = if let Some(ref dt) = dtype {
        // Explicit dtype takes precedence
        dt.clone()
    } else if promote {
        // Auto-promote using sum_acc_dtype
        Tensor::sum_acc_dtype(&original_dtype)
    } else if op == ReduceOp::Add && Tensor::should_cast_back_after_sum(&original_dtype) {
        // float16/bf16/fp8 sums accumulate in float32 even without `promote`: an
        // 8-/10-bit mantissa makes a long sum order-sensitive, so a reassociating
        // opt diverges. The result is cast back to the input dtype below.
        Tensor::sum_acc_dtype(&original_dtype)
    } else {
        // Preserve input dtype
        original_dtype.clone()
    };

    // Cast to accumulation dtype if needed
    let working_tensor = if acc_dtype != original_dtype { tensor.cast(acc_dtype.clone()) } else { tensor.clone() };

    // Perform reduction
    let reduced = working_tensor.uop().try_reduce_axis(op, resolved_axes.clone()).context(UOpSnafu)?;

    // Handle keepdim
    let result = if keepdim {
        let keepdim_shape: Vec<SInt> = shape
            .iter()
            .enumerate()
            .map(|(axis, dim)| if resolved_axes.contains(&axis) { SInt::Const(1) } else { dim.clone() })
            .collect();
        Tensor::new(reduced).try_reshape(&keepdim_shape)?
    } else {
        Tensor::new(reduced)
    };

    // Cast back to the input dtype whenever we accumulated in a wider type
    // (fp16/bf16/fp8), whether via `promote` or the float32 sum-acc upcast above.
    if dtype.is_none() && acc_dtype != original_dtype && Tensor::should_cast_back_after_sum(&original_dtype) {
        Ok(result.cast(original_dtype))
    } else {
        Ok(result)
    }
}

/// Mean implementation (shared by mean and mean_with).
fn mean_impl(tensor: &Tensor, axes: impl Into<AxisSpec>, keepdim: bool) -> Result<Tensor> {
    let axes = axes.into();
    let shape = tensor.shape()?;
    let resolved_axes = Tensor::resolve_axis_spec(&axes, shape.len())?;

    // Calculate count of reduced elements
    let mut count = 1i64;
    for &axis in &resolved_axes {
        if let Some(dim_size) = shape[axis].as_const() {
            count *= dim_size as i64;
        } else {
            return SymbolicShapeUnsupportedSnafu { operation: "mean over a symbolic axis" }.fail().map_err(Into::into);
        }
    }

    // Accumulate and divide in `sum_acc_dtype` (float32 for fp16/bf16/fp8; float32
    // for integers), then cast back to the input dtype. A long fp16 sum accumulated
    // in fp16 is order-sensitive and diverges under a reassociating opt.
    let dtype = tensor.uop().dtype();
    let is_int = Tensor::is_integer_dtype(&dtype);
    let acc_dtype = if is_int { DType::Float32 } else { Tensor::sum_acc_dtype(&dtype) };
    let output_dtype = if is_int { DType::Float32 } else { dtype };

    // Explicit dtype ⇒ the reduce does not cast back, so the sum stays in acc_dtype.
    let sum = reduce_internal(tensor, ReduceOp::Add, axes, keepdim, Some(acc_dtype.clone()), false)?;

    let count_tensor = Tensor::new(UOp::const_(acc_dtype.clone(), svod_ir::ConstValue::Float(count as f64)));
    let mean = (&sum / &count_tensor)?;
    Ok(if acc_dtype != output_dtype { mean.cast(output_dtype) } else { mean })
}

/// Variance implementation using the numerically-stable `(X - E[X])²` formula.
fn var_impl(tensor: &Tensor, axes: AxisSpec, keepdim: bool, correction: i64) -> Result<Tensor> {
    let (var, _mean) = var_mean_impl(tensor, axes, keepdim, correction)?;
    Ok(var)
}

/// Standard deviation implementation.
fn std_impl(tensor: &Tensor, axes: AxisSpec, keepdim: bool, correction: i64) -> Result<Tensor> {
    let variance = var_impl(tensor, axes, keepdim, correction)?;
    variance.try_sqrt()
}

/// Variance and mean implementation using single-pass algorithm.
fn var_mean_impl(tensor: &Tensor, axes: AxisSpec, keepdim: bool, correction: i64) -> Result<(Tensor, Tensor)> {
    let shape = tensor.shape()?;
    let resolved_axes = Tensor::resolve_axis_spec(&axes, shape.len())?;

    // Calculate count of reduced elements
    let mut count = 1i64;
    for &axis in &resolved_axes {
        if let Some(dim_size) = shape[axis].as_const() {
            count *= dim_size as i64;
        } else {
            return SymbolicShapeUnsupportedSnafu { operation: "variance over a symbolic axis" }
                .fail()
                .map_err(Into::into);
        }
    }

    // Determine output dtype (integers → float32, floats preserve)
    let dtype = tensor.uop().dtype();
    let output_dtype = if Tensor::is_integer_dtype(&dtype) { DType::Float32 } else { dtype.clone() };

    // Compute mean: E[X]
    let mean = mean_impl(tensor, axes.clone(), keepdim)?;

    // Compute deviation from mean: X - E[X]
    // Need to broadcast mean if keepdim=false
    let deviation = if keepdim {
        tensor.try_sub(&mean)?
    } else {
        // Expand mean back to original shape for subtraction
        let mut expanded_mean = mean.clone();
        for &axis in &resolved_axes {
            expanded_mean = expanded_mean.try_unsqueeze(axis as isize)?;
        }
        tensor.try_sub(&expanded_mean)?
    };

    // Square the deviations: (X - E[X])²
    let squared_dev = deviation.square();

    // Accumulate *and* divide in `sum_acc_dtype` like `mean_impl`, casting to the
    // output dtype only at the end: a float16 sum of squares cast back before the
    // divide overflows to inf well before the ratio would.
    let acc_dtype = Tensor::sum_acc_dtype(&squared_dev.uop().dtype());
    let sum_sq_dev = reduce_internal(&squared_dev, ReduceOp::Add, axes, keepdim, Some(acc_dtype.clone()), false)?;

    // Divide by max(0, N - correction): correction=1 is the unbiased sample
    // variance, correction=0 the population variance.
    let denom = (count - correction).max(0);
    let variance = if denom == 0 {
        // n <= correction (e.g. a single element with correction=1) ⇒ divisor 0.
        // svod's `/` rejects a constant-zero divisor, so express the IEEE result as
        // `reduced * inf` (0*inf = NaN, k*inf = +inf).
        let inf = Tensor::new(UOp::const_(acc_dtype.clone(), svod_ir::ConstValue::Float(f64::INFINITY)));
        (&sum_sq_dev * &inf)?
    } else {
        let denom_tensor = Tensor::new(UOp::const_(acc_dtype.clone(), svod_ir::ConstValue::Float(denom as f64)));
        (&sum_sq_dev / &denom_tensor)?
    };
    let variance = if acc_dtype != output_dtype { variance.cast(output_dtype) } else { variance };

    Ok((variance, mean))
}

/// Standard deviation and mean implementation.
fn std_mean_impl(tensor: &Tensor, axes: AxisSpec, keepdim: bool, correction: i64) -> Result<(Tensor, Tensor)> {
    let (variance, mean) = var_mean_impl(tensor, axes, keepdim, correction)?;
    let std = variance.try_sqrt()?;
    Ok((std, mean))
}
