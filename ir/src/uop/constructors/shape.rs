//! Shape manipulation: reshape, permute, expand, pad, shrink, flip.
//!
//! These operations manipulate tensor shapes and layouts without changing
//! the underlying data (except for padding which may add values).

use std::sync::Arc;

use smallvec::SmallVec;
use svod_dtype::DType;

use crate::Result;
use crate::op::Op;
use crate::ops;
use crate::uop::UOp;

// Low-level constructors (pub(crate) - not yet used but will be needed for optimization passes)
#[allow(dead_code)]
impl UOp {
    /// Build a shaped value while retaining a scalar element dtype.
    pub fn stack(sources: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        let dtype = if sources.is_empty() {
            DType::Void
        } else {
            crate::dtype_from_op(&Op::Stack(ops::Stack { sources: sources.clone() }))
                .expect("STACK sources must have a promotable dtype")
        };
        let sources = sources
            .into_iter()
            .map(|source| if Self::is_invalid_marker(&source) { source } else { source.cast(dtype.clone()) })
            .collect();
        Self::new(Op::Stack(ops::Stack { sources }), dtype)
    }

    /// The lanes of `src` in `mask` order: lane `i` of the result is lane
    /// `mask[i]` of `src`, a vector of the source's element type with
    /// `mask.len()` lanes (a scalar source is counted, a vector re-counted).
    /// A constant source gathers at construction.
    pub fn shuffle(src: Arc<Self>, mask: SmallVec<[u32; 8]>) -> Arc<Self> {
        let dtype = crate::dtype_from_op(&Op::Shuffle(ops::Shuffle { src: src.clone(), mask: mask.clone() }))
            .expect("a SHUFFLE takes a vector");
        if let Op::VConst(ops::VConst { values }) = src.op() {
            let lanes = |lane: &u32| values.get(*lane as usize).copied().unwrap_or(values[0]);
            return Self::new(Op::VConst(ops::VConst { values: mask.iter().map(lanes).collect() }), dtype);
        }
        Self::new(Op::Shuffle(ops::Shuffle { src, mask }), dtype)
    }

    /// Reshape tensor to new shape (low-level, UOp-based constructor).
    /// Takes a UOp for the shape parameter (used internally by compiler passes).
    /// For the public API with validation, use `try_reshape`.
    pub(crate) fn reshape(src: Arc<Self>, new_shape: Arc<Self>) -> Arc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Reshape(ops::Reshape { src, new_shape }), dtype)
    }

    /// Permute dimensions (low-level, UOp-based constructor).
    ///
    /// For the public API with validation, use `try_permute`.
    pub(crate) fn permute(src: Arc<Self>, axes: Vec<usize>) -> Arc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Permute(ops::Permute { src, axes }), dtype)
    }

    /// Expand (broadcast) dimensions (low-level, UOp-based constructor).
    ///
    /// Takes a UOp for the shape parameter (used internally by compiler passes).
    /// For the public API with validation, use `try_expand`.
    pub(crate) fn expand(src: Arc<Self>, new_shape: Arc<Self>) -> Arc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Expand(ops::Expand { src, new_shape }), dtype)
    }

    /// Pad tensor (low-level, UOp-based constructor).
    ///
    /// Takes UOps for padding parameters (used internally by compiler passes).
    /// For the public API with validation, use `try_pad`.
    pub(crate) fn pad(src: Arc<Self>, begin_pads: Arc<Self>, end_pads: Arc<Self>) -> Arc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Pad(ops::Pad { src, begin_pads, end_pads }), dtype)
    }

    /// Shrink (slice) tensor (low-level, UOp-based constructor).
    ///
    /// Takes UOps for range parameters (used internally by compiler passes).
    /// For the public API with validation, use `try_shrink`.
    pub(crate) fn shrink(src: Arc<Self>, offsets: Arc<Self>, sizes: Arc<Self>) -> Arc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Shrink(ops::Shrink { src, offsets, sizes }), dtype)
    }

    /// Flip (reverse) axes (low-level, UOp-based constructor).
    ///
    /// For the public API with validation, use `try_flip`.
    pub(crate) fn flip(src: Arc<Self>, axes: Vec<bool>) -> Arc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Flip(ops::Flip { src, axes }), dtype)
    }
}

// Primary Movement Operation Constructors (with validation)
impl UOp {
    /// Reshape with strict validation (fail-fast).
    ///
    /// Validates:
    /// - No negative dimensions in new_shape
    /// - Product of input shape == product of output shape
    pub fn try_reshape(self: &Arc<Self>, new_shape: &crate::shape::Shape) -> Result<Arc<Self>> {
        use crate::error::ReshapeSizeMismatchSnafu;
        use crate::shape::shape_to_uop;
        use snafu::ensure;

        // Validate product equality if source shape is known
        if let Some(src_shape) = self.shape()? {
            // Identity reshape: skip if shapes already match. Bare BUFFER/PARAM
            // sources are indexed directly by rangeify, matching Tinygrad.
            if src_shape.as_slice() == new_shape.as_slice() {
                return Ok(self.clone());
            }

            let src_product = crate::sint_prod(src_shape);
            let dst_product = crate::sint_prod(new_shape);

            // If both are concrete, validate equality
            if let (Some(src_prod), Some(dst_prod)) = (src_product.as_const(), dst_product.as_const()) {
                ensure!(src_prod == dst_prod, ReshapeSizeMismatchSnafu { input_size: src_prod, output_size: dst_prod });
            }
            // Symbolic products can't be validated at compile time
        }

        let shape_uop = shape_to_uop(new_shape);
        let dtype = self.dtype();
        Ok(Self::new(Op::Reshape(ops::Reshape { src: self.clone(), new_shape: shape_uop }), dtype))
    }

    /// Expand (broadcast) with strict validation.
    ///
    /// Validates:
    /// - Number of dimensions matches
    /// - Each dimension either matches or src dimension is 1
    pub fn try_expand(self: &Arc<Self>, new_shape: &crate::shape::Shape) -> Result<Arc<Self>> {
        use crate::error::ExpandDimensionMismatchSnafu;
        use crate::error::ExpandInvalidDimensionSnafu;
        use crate::shape::shape_to_uop;
        use snafu::ensure;

        if let Some(src_shape) = self.shape()? {
            // Check dimension count
            ensure!(
                src_shape.len() == new_shape.len(),
                ExpandDimensionMismatchSnafu { input_dims: src_shape.len(), output_dims: new_shape.len() }
            );

            // Check each dimension can be expanded
            for (dim_idx, (src_dim, new_dim)) in src_shape.iter().zip(new_shape.iter()).enumerate() {
                // If both are concrete, validate expand rule
                if let (Some(s), Some(ns)) = (src_dim.as_const(), new_dim.as_const()) {
                    ensure!(s == ns || s == 1, ExpandInvalidDimensionSnafu { dim: dim_idx, input: s, output: ns });
                }
                // Symbolic dimensions assumed compatible
            }

            // Identity expand: skip if shapes already match
            if src_shape.as_slice() == new_shape.as_slice() {
                return Ok(self.clone());
            }
        }

        let shape_uop = shape_to_uop(new_shape);
        let dtype = self.dtype();
        Ok(Self::new(Op::Expand(ops::Expand { src: self.clone(), new_shape: shape_uop }), dtype))
    }

    /// Permute with strict validation.
    ///
    /// Validates:
    /// - Permutation is valid (contains each index 0..n exactly once)
    pub fn try_permute(self: &Arc<Self>, axes: Vec<usize>) -> Result<Arc<Self>> {
        // Validate permutation if source shape is known
        if let Some(src_shape) = self.shape()? {
            Self::validate_permutation(&axes, src_shape.len())?;

            // Identity permute: skip if axes is [0, 1, 2, ..., n-1]
            if axes.iter().enumerate().all(|(i, &a)| a == i) {
                return Ok(self.clone());
            }
        }

        let dtype = self.dtype();
        Ok(Self::new(Op::Permute(ops::Permute { src: self.clone(), axes }), dtype))
    }

    /// Pad with strict validation.
    ///
    /// Validates:
    /// - Padding values are concrete (not symbolic)
    /// - Number of padding pairs matches dimensions
    pub fn try_pad(self: &Arc<Self>, padding: &[(crate::SInt, crate::SInt)]) -> Result<Arc<Self>> {
        use crate::error::{PadDimensionMismatchSnafu, SymbolicPaddingUnsupportedSnafu};
        use crate::shape::ranges_to_uops;
        use snafu::ensure;

        // Empty padding (scalar) → identity
        if padding.is_empty() {
            return Ok(self.clone());
        }

        // Check for symbolic padding values
        for (begin, end) in padding {
            ensure!(begin.is_const(), SymbolicPaddingUnsupportedSnafu);
            ensure!(end.is_const(), SymbolicPaddingUnsupportedSnafu);
        }

        // All-zero padding: no-op
        if padding.iter().all(|(b, e)| b.as_const() == Some(0) && e.as_const() == Some(0)) {
            return Ok(self.clone());
        }

        if let Some(src_shape) = self.shape()? {
            // Check dimension count
            ensure!(
                padding.len() == src_shape.len(),
                PadDimensionMismatchSnafu { padding_dims: padding.len(), shape_dims: src_shape.len() }
            );
        }

        let (begin_pads, end_pads) = ranges_to_uops(padding);
        let dtype = self.dtype();
        Ok(Self::new(Op::Pad(ops::Pad { src: self.clone(), begin_pads, end_pads }), dtype))
    }

    /// Shrink (slice) with strict validation.
    ///
    /// Validates:
    /// - Range values are concrete (not symbolic)
    /// - begin <= end for each dimension
    /// - 0 <= begin, end <= dimension_size
    pub fn try_shrink(self: &Arc<Self>, ranges: &[(crate::SInt, crate::SInt)]) -> Result<Arc<Self>> {
        use crate::error::ShrinkBoundsViolationSnafu;
        use crate::shape::ranges_to_uops;
        use snafu::ensure;

        // Empty ranges (scalar) → identity
        if ranges.is_empty() {
            return Ok(self.clone());
        }

        // Symbolic shrink ranges are allowed — the rangeify pipeline handles
        // symbolic range ends. Only concrete ranges are bounds-checked.
        if let Some(src_shape) = self.shape()? {
            for (dim_idx, ((begin, end), dim_size)) in ranges.iter().zip(src_shape.iter()).enumerate() {
                if let (Some(b), Some(e), Some(s)) = (begin.as_const(), end.as_const(), dim_size.as_const()) {
                    ensure!(
                        b <= e && e <= s,
                        ShrinkBoundsViolationSnafu { dim: dim_idx, begin: b, end: e, shape_size: s }
                    );
                }
            }

            // Identity shrink: skip if all ranges span the full dimension
            if ranges.iter().zip(src_shape.iter()).all(|((b, e), d)| b.as_const() == Some(0) && *e == *d) {
                return Ok(self.clone());
            }
        }

        // `end - begin` folds `(t + 1) - t` to a constant, so a symbolic
        // offset keeps a constant-extent axis the optimizer can still upcast.
        let offsets_and_sizes: Vec<_> = ranges.iter().map(|(begin, end)| (begin.clone(), end - begin)).collect();
        let (offsets, sizes) = ranges_to_uops(&offsets_and_sizes);
        let dtype = self.dtype();
        let result = Self::new(Op::Shrink(ops::Shrink { src: self.clone(), offsets, sizes }), dtype);
        // Tinygrad (movement.py:128): return self if ret.shape == self.shape else ret.
        // A symbolic offset on a length-1 axis keeps the shape, so it must not count.
        if offsets_and_sizes.iter().all(|(o, _)| o.as_const() == Some(0))
            && result.shape().ok().flatten() == self.shape().ok().flatten()
        {
            return Ok(self.clone());
        }
        Ok(result)
    }

    /// Flip with strict validation.
    ///
    /// Validates:
    /// - Flip specification length matches shape dimensions
    pub fn try_flip(self: &Arc<Self>, axes: Vec<bool>) -> Result<Arc<Self>> {
        // All-false flip: no-op
        if !axes.iter().any(|&a| a) {
            return Ok(self.clone());
        }

        if let Some(src_shape) = self.shape()? {
            Self::validate_flip_axes(&axes, src_shape.len())?;
        }

        let dtype = self.dtype();
        Ok(Self::new(Op::Flip(ops::Flip { src: self.clone(), axes }), dtype))
    }

    /// Split tensor across multiple devices along specified axis.
    ///
    /// Creates a multi-device tensor where each device holds a shard.
    /// Use with MSTACK/MSELECT for distributed tensor operations.
    pub fn multi(src: Arc<Self>, axis: usize) -> Arc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Multi(ops::Multi { src, axis }), dtype)
    }
}
