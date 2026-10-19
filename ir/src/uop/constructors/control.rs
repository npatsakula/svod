//! Control flow: range, if/end, barrier, symbolic variables.
//!
//! This module contains control flow operations:
//! - Loop constructs: range, range_const, range_axis
//! - Conditionals: if_, endif, end
//! - Synchronization: barrier
//! - Symbolic variables: var, define_var, bind
//! - Special: special (GPU dimension index)

use std::sync::Arc;

use smallvec::SmallVec;
use svod_dtype::DType;

use crate::op::Op;
use crate::ops;
use crate::types::{AxisId, AxisType, BarrierFence};
use crate::uop::UOp;

impl UOp {
    // =========================================================================
    // Range Operations
    // =========================================================================

    /// Create a Range operation with specified axis type.
    pub fn range_axis(end: Arc<Self>, axis_id: AxisId, axis_type: AxisType) -> Arc<Self> {
        Self::range_axis_dtype(end, axis_id, axis_type, DType::WeakInt)
    }

    /// Create a Range operation with an explicit index dtype.
    pub fn range_axis_dtype(end: Arc<Self>, axis_id: AxisId, axis_type: AxisType, dtype: DType) -> Arc<Self> {
        assert!(end.dtype().is_int(), "range_axis: end must be integer, got {:?}", end.dtype());
        assert!(dtype.is_int(), "range_axis: dtype must be integer, got {dtype:?}");
        let end = end.cast(dtype.clone());
        Self::new(Op::Range(ops::Range { end, axis_id, axis_type, deps: SmallVec::new() }), dtype)
    }

    /// Create a RANGE operation with Loop axis type (convenience for tests).
    ///
    /// Uses `AxisId::Renumbered` since tests typically work with renumbered kernels.
    pub fn range(end: Arc<Self>, axis_id: usize) -> Arc<Self> {
        Self::range_axis(end, AxisId::Renumbered(axis_id), AxisType::Weak)
    }

    /// Create a RANGE operation with constant end value (convenience for tests).
    ///
    /// Uses `AxisId::Renumbered` since tests typically work with renumbered kernels.
    /// Creates a `Loop` range (inside kernels).
    pub fn range_const(end_value: i64, axis_id: usize) -> Arc<Self> {
        let end = Self::index_const(end_value);
        Self::range_axis(end, AxisId::Renumbered(axis_id), AxisType::Weak)
    }

    // =========================================================================
    // Conditional Operations
    // =========================================================================

    /// Create a conditional block that executes body when condition is true.
    ///
    /// Body contains operations to execute; use `endif` to close the block.
    pub fn if_(condition: Arc<Self>, body: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        Self::new(Op::If(ops::If { condition, body }), DType::Void)
    }

    /// End if block.
    pub fn endif(if_op: Arc<Self>) -> Arc<Self> {
        Self::new(Op::EndIf(ops::EndIf { if_op }), DType::Void)
    }

    /// End of range or reduce scope.
    ///
    /// Wraps self (the computation) and closes the specified ranges.
    /// This marks the end of RANGE or REDUCE loops.
    ///
    /// # Arguments
    ///
    /// * `ranges` - The RANGE or REDUCE operations being closed
    pub fn end(self: &Arc<Self>, ranges: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        if ranges.is_empty() {
            return self.clone();
        }
        Self::new(Op::End(ops::End { computation: self.clone(), ranges }), DType::Void)
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    /// Insert a synchronization barrier.
    ///
    /// Self passes through; `deps` are operations that must complete before
    /// any consumer of this barrier executes.
    pub fn barrier(self: &Arc<Self>, deps: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        self.fenced_barrier(deps, BarrierFence::All)
    }

    /// [`Self::barrier`] whose fence makes only `fence`'s memory visible across it.
    pub fn fenced_barrier(self: &Arc<Self>, deps: SmallVec<[Arc<Self>; 4]>, fence: BarrierFence) -> Arc<Self> {
        Self::new(Op::Barrier(ops::Barrier { src: self.clone(), deps, fence }), DType::Void)
    }

    // =========================================================================
    // Symbolic Variables
    // =========================================================================

    /// Create a DefineVar operation for range-bounded variables.
    ///
    /// Used in testing and symbolic analysis to define variables with known ranges.
    /// Range is [min_val, max_val] inclusive.
    pub fn var(name: impl Into<String>, dtype: DType, min_val: i64, max_val: i64) -> Arc<Self> {
        Self::new(Op::DefineVar(ops::DefineVar { name: name.into(), min_val, max_val }), dtype)
    }

    /// Define a symbolic variable with known bounds for range analysis.
    ///
    /// Range is [min_val, max_val] inclusive.
    pub fn define_var(name: String, min_val: i64, max_val: i64) -> Arc<Self> {
        Self::variable(name, min_val, max_val, DType::WeakInt)
    }

    /// Create Tinygrad's scalar ALU PARAM representation for a symbolic variable.
    pub fn variable(name: String, min_val: i64, max_val: i64, dtype: DType) -> Arc<Self> {
        let shape = crate::shape::shape_to_uop(&SmallVec::new());
        let arg = crate::ParamArg::variable(name, dtype.clone(), min_val, max_val);
        Self::new(Op::Param(ops::Param { shape, arg: arg.into() }), dtype)
    }

    /// Bind concrete value to symbolic variable.
    pub fn bind(self: &Arc<Self>, value: Arc<Self>) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::Bind(ops::Bind { var: self.clone(), value }), dtype)
    }

    // =========================================================================
    // Special Operations
    // =========================================================================

    /// Create a GPU-specific dimension variable (e.g., blockIdx.x, threadIdx.y).
    ///
    /// Unlike RANGE which is a loop, SPECIAL represents hardware-provided indices.
    /// The `name` identifies the dimension (rendered as-is in codegen).
    pub fn special(end: Arc<Self>, name: String) -> Arc<Self> {
        Self::special_dtype(end, name, DType::WeakInt)
    }

    /// Create a hardware index with an explicit dtype.
    pub fn special_dtype(end: Arc<Self>, name: String, dtype: DType) -> Arc<Self> {
        assert!(end.dtype().is_int(), "special: end must be integer, got {:?}", end.dtype());
        assert!(dtype.is_int(), "special: dtype must be integer, got {dtype:?}");
        let end = end.cast(dtype.clone());
        Self::new(Op::Special(ops::Special { end, name }), dtype)
    }
}
