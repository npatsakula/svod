//! Data creation: constants, buffers, device specifications.
//!
//! This module contains constructors for creating data primitives:
//! - Constants (scalar, native, integer)
//! - Buffers (new, view)
//! - Device specifications
//! - No-op and cast operations

use std::sync::Arc;

use svod_dtype::DType;
use svod_dtype::DeviceSpec;
use svod_dtype::ext::HasDType;

use crate::IntoUOp;
use crate::op::Op;
use crate::ops;
use crate::types::{ConstValue, ConstValueHash};
use crate::uop::core::UOp;
use crate::uop::hash_consing::next_unique_id;

impl UOp {
    // =========================================================================
    // Constants
    // =========================================================================

    /// Create a constant UOp with explicit dtype and value.
    ///
    /// Normalizes the value to match the target dtype (e.g., `Float(5.0)` becomes
    /// `Int(5)` when dtype is Int32). This prevents codegen from emitting
    /// mismatched literals.
    ///
    /// Use `native_const` for type-inferred constants from Rust values.
    pub fn const_(dtype: DType, value: ConstValue) -> Arc<Self> {
        Self::try_const_(dtype, value).expect("constant must be representable by its declared dtype")
    }

    /// Fallible typed constant construction.
    pub fn try_const_(dtype: DType, value: ConstValue) -> crate::Result<Arc<Self>> {
        if value == ConstValue::Invalid {
            return Ok(Self::new(Op::Const(ConstValueHash(ConstValue::Invalid)), DType::Bool));
        }
        let scalar_dtype = match &dtype {
            DType::Scalar(scalar) => DType::Scalar(*scalar),
            DType::Vector { scalar, .. } if *scalar != svod_dtype::ScalarDType::Void => DType::Scalar(*scalar),
            _ => return Err(crate::Error::ConstantConversion { value, dtype }),
        };
        let normalized = value
            .cast(&scalar_dtype)
            .ok_or_else(|| crate::Error::ConstantConversion { value, dtype: dtype.clone() })?;
        Ok(Self::new(Op::Const(ConstValueHash(normalized)), dtype))
    }

    /// Create a constant UOp from a Rust native value with automatic dtype inference.
    pub fn native_const<T: HasDType + IntoUOp>(value: T) -> Arc<Self> {
        value.into_uop(T::DTYPE)
    }

    /// Create a mathematical integer constant.
    ///
    /// Kept as a convenience name for indexing call sites; like Tinygrad's
    /// `UOp.const`, integer literals are weak until they meet a concrete dtype.
    pub fn index_const(value: i64) -> Arc<Self> {
        Self::const_(DType::WeakInt, ConstValue::Int(value))
    }

    /// Create a constant with the same dtype (vector count included) and shape as self.
    ///
    /// This is the Rust equivalent of Tinygrad's `x.const_like(value)`.
    /// Useful for creating identity elements, zeros, or other constants
    /// that match an existing UOp's type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # use svod_ir::UOp;
    /// # use svod_dtype::DType;
    /// let x = UOp::const_(DType::Float32, svod_ir::ConstValue::Float(5.0));
    /// let zero = x.const_like(0.0);
    /// assert_eq!(zero.dtype(), DType::Float32);
    /// ```
    pub fn const_like<T: crate::IntoUOp>(self: &Arc<Self>, value: T) -> Arc<Self> {
        // Tinygrad `uop/ops.py:596`: `UOp.const(b, dtype or self.dtype)` — the full
        // dtype, so a Vector receiver yields a Vector constant.
        let ret = value.into_uop(self.dtype());
        if let Ok(Some(shape)) = self.shape()
            && !shape.is_empty()
        {
            return Self::expand(ret, crate::shape::shape_to_uop(shape));
        }
        ret
    }

    /// Create a post-movement constant with this value's dtype and maximum
    /// element count. Tinygrad's `vconst_like` uses `STACK`, not `EXPAND`, because
    /// movement ops have already been removed when late gating calls it. An
    /// unshaped or unbounded receiver degrades to the scalar constant rather than
    /// aborting, since the late gater applies it to already-lowered addresses.
    pub fn vconst_like<T: crate::IntoUOp>(self: &Arc<Self>, value: T) -> Arc<Self> {
        let ret = value.into_uop(self.dtype());
        let count = self
            .shape()
            .ok()
            .flatten()
            .and_then(|shape| shape.iter().try_fold(1usize, |n, dim| n.checked_mul(dim.vmax()?)))
            .unwrap_or(1);
        if count == 1 { ret } else { Self::stack((0..count).map(|_| ret.clone()).collect()) }
    }

    /// Create a vector constant from multiple values.
    ///
    /// Dtype is inferred from the first value; all values must be same type.
    pub fn vconst(values: Vec<ConstValue>, scalar_dtype: DType) -> Arc<Self> {
        Self::try_vconst(values, scalar_dtype).expect("VConst lanes must be representable by their declared dtype")
    }

    /// Fallible vector constant construction that commits every lane.
    pub fn try_vconst(values: Vec<ConstValue>, scalar_dtype: DType) -> crate::Result<Arc<Self>> {
        let scalar = scalar_dtype.scalar().ok_or_else(|| crate::Error::ConstantConversion {
            value: values.first().copied().unwrap_or(ConstValue::Invalid),
            dtype: scalar_dtype.clone(),
        })?;
        let mut committed = Vec::with_capacity(values.len());
        for value in values {
            let value = if value == ConstValue::Invalid {
                value
            } else {
                value
                    .cast(&DType::Scalar(scalar))
                    .ok_or_else(|| crate::Error::ConstantConversion { value, dtype: scalar_dtype.clone() })?
            };
            committed.push(value);
        }
        let vec_dtype = scalar_dtype.vec(committed.len()).ok_or_else(|| crate::Error::ConstantConversion {
            value: committed.first().copied().unwrap_or(ConstValue::Invalid),
            dtype: scalar_dtype.clone(),
        })?;
        Ok(Self::new(Op::VConst(ops::VConst { values: committed }), vec_dtype))
    }

    // =========================================================================
    // Buffers
    // =========================================================================

    /// Create a unique buffer identifier.
    pub fn buffer_id(num: Option<usize>) -> Arc<Self> {
        let id = num.unwrap_or_else(next_unique_id);
        Self::new(Op::Unique(id), DType::Void)
    }

    /// Create a normalized unique identifier for cache-key parity.
    pub fn lunique(num: Option<usize>) -> Arc<Self> {
        let id = num.unwrap_or_else(next_unique_id);
        Self::new(Op::LUnique(id), DType::Void)
    }

    /// Create a new buffer.
    ///
    /// Equivalent to pinned Tinygrad's `BUFFER(shape, ParamArg(slot, dtype, device=device))`.
    pub fn new_buffer(device: DeviceSpec, size: usize, dtype: DType) -> Arc<Self> {
        let slot = next_unique_id();
        Self::buffer(slot, size, dtype, svod_dtype::AddrSpace::Global, Some(device))
    }

    /// Create structured BUFFER storage with one shape source and ParamArg metadata.
    pub fn buffer(
        slot: usize,
        size: usize,
        dtype: DType,
        addrspace: svod_dtype::AddrSpace,
        device: Option<DeviceSpec>,
    ) -> Arc<Self> {
        let shape = crate::shape::shape_to_uop(&smallvec::smallvec![crate::SInt::Const(size)]);
        Self::buffer_shaped(shape, slot, dtype, addrspace, device)
    }

    /// [`Self::buffer`] off the global address space with its shape
    /// committed as a program commits it ([`Self::committed_shape`]), no weak
    /// constant made: what a pre-linearized program ([`Self::linear_program`])
    /// declares, which its assembly then has nothing to commit in.
    pub fn committed_buffer(slot: usize, size: usize, dtype: DType, addrspace: svod_dtype::AddrSpace) -> Arc<Self> {
        let size = ConstValue::Int(size as i64);
        Self::buffer_shaped(Self::const_(Self::committed_int(size), size), slot, dtype, addrspace, None)
    }

    fn buffer_shaped(
        shape: Arc<Self>,
        slot: usize,
        dtype: DType,
        addrspace: svod_dtype::AddrSpace,
        device: Option<DeviceSpec>,
    ) -> Arc<Self> {
        assert_eq!(device.is_some(), addrspace == svod_dtype::AddrSpace::Global);
        assert!(!dtype.is_weak(), "BUFFER storage dtype cannot be weak");
        assert!(!matches!(dtype, DType::Ptr { .. }), "BUFFER dtype is the stored element dtype, not a pointer");
        let arg = crate::ParamArg::buffer(slot, dtype.clone(), addrspace, device);
        Self::new(Op::Buffer(ops::Buffer { shape, arg: arg.into() }), dtype)
    }

    /// Create a normalized buffer parameter with positional slot.
    /// Used by pre-schedule normalization (BUFFER→PARAM) to erase buffer identity.
    /// Matches Tinygrad's `UOp.param(slot, dtype, shape, device)` (ops.py:817-819).
    pub fn param(slot: usize, size: usize, dtype: DType, device: Option<DeviceSpec>) -> Arc<Self> {
        assert!(!dtype.is_weak(), "PARAM storage dtype cannot be weak");
        assert!(!matches!(dtype, DType::Ptr { .. }), "PARAM dtype is the stored element dtype, not a pointer");
        let shape = crate::shape::shape_to_uop(&smallvec::smallvec![crate::SInt::Const(size)]);
        let arg = crate::ParamArg::buffer(slot, dtype.clone(), svod_dtype::AddrSpace::Global, device);
        Self::new(Op::Param(ops::Param { shape, arg: arg.into() }), dtype)
    }

    /// Create a positional global PARAM with a logical, possibly symbolic shape.
    pub fn param_with_shape(
        slot: usize,
        shape: &crate::shape::Shape,
        dtype: DType,
        device: Option<DeviceSpec>,
    ) -> Arc<Self> {
        assert!(!dtype.is_weak(), "PARAM storage dtype cannot be weak");
        assert!(!matches!(dtype, DType::Ptr { .. }), "PARAM dtype is the stored element dtype, not a pointer");
        let shape = crate::shape::shape_to_uop(shape);
        let arg = crate::ParamArg::buffer(slot, dtype.clone(), svod_dtype::AddrSpace::Global, device);
        Self::new(Op::Param(ops::Param { shape, arg: arg.into() }), dtype)
    }

    /// Create a positional scalar PARAM for a FUNCTION body.
    pub fn scalar_param(slot: usize, name: Option<String>, dtype: DType, min_val: i64, max_val: i64) -> Arc<Self> {
        let shape = crate::shape::shape_to_uop(&smallvec::SmallVec::new());
        let arg = crate::ParamArg::scalar(slot, name, dtype.clone(), min_val, max_val);
        Self::new(Op::Param(ops::Param { shape, arg: arg.into() }), dtype)
    }

    /// Create flattened storage and restore its logical shape, matching Tinygrad's
    /// `UOp.placeholder`.
    pub fn placeholder(
        shape: &crate::shape::Shape,
        dtype: DType,
        slot: usize,
        addrspace: svod_dtype::AddrSpace,
        device: Option<DeviceSpec>,
    ) -> crate::Result<Arc<Self>> {
        let concrete_shape: Vec<usize> = shape
            .iter()
            .map(|dim| {
                dim.as_const().ok_or_else(|| crate::Error::SymbolicShapeUnsupported { operation: "placeholder" })
            })
            .collect::<crate::Result<_>>()?;
        let size = concrete_shape.iter().product();
        let dtype = dtype.strong_dtype();
        let storage = if addrspace == svod_dtype::AddrSpace::Global {
            Self::param(slot, size, dtype, device)
        } else {
            assert!(matches!(addrspace, svod_dtype::AddrSpace::Local | svod_dtype::AddrSpace::Reg));
            assert!(device.is_none(), "LOCAL and REG placeholders cannot have a device");
            Self::buffer(slot, size, dtype, addrspace, None)
        };
        if concrete_shape.len() > 1 {
            storage.try_reshape(&crate::shape::Shape::from_iter(concrete_shape.into_iter().map(crate::SInt::Const)))
        } else {
            Ok(storage)
        }
    }

    /// Create a contiguous typed slice. `offset` is measured in source elements.
    pub fn contiguous_slice(self: &Arc<Self>, size: usize, offset: usize, dtype: DType) -> Arc<Self> {
        Self::new(Op::Slice(ops::Slice { buffer: self.clone(), offset: Self::index_const(offset as i64), size }), dtype)
    }

    // =========================================================================
    // Type Operations
    // =========================================================================

    /// Create a no-op.
    pub fn noop() -> Arc<Self> {
        Self::new(Op::Noop, DType::Void)
    }

    /// Cast to a different dtype.
    ///
    /// If casting a vector to a scalar type, automatically promotes the target
    /// dtype to a matching vector type. This prevents invalid scalar-to-vector
    /// casts in the IR. (Matches Tinygrad's cast behavior.)
    pub fn cast(self: &Arc<Self>, dtype: DType) -> Arc<Self> {
        let src_vcount = self.dtype().vcount();
        let dst_vcount = dtype.vcount();

        // Auto-promote scalar target to vector if source is vector
        let dtype = if dst_vcount == 1 && src_vcount > 1 {
            dtype.vec(src_vcount).expect("cast target with vcount==1 is vectorizable")
        } else {
            dtype
        };

        // No-op if types match
        if self.dtype() == dtype {
            return self.clone();
        }

        Self::new(Op::Cast(ops::Cast { src: self.clone(), dtype: dtype.clone() }), dtype)
    }

    /// Bitcast: reinterpret bits as different type.
    pub fn bitcast(self: &Arc<Self>, dtype: DType) -> Arc<Self> {
        Self::new(Op::BitCast(ops::BitCast { src: self.clone(), dtype: dtype.clone() }), dtype)
    }
}
