//! Shape utilities for UOps with symbolic shape support.
//!
//! This module provides shape-related types and functions following Tinygrad's approach:
//! - Shapes can contain both concrete integers and symbolic UOp expressions
//! - Shape inference with validation
//! - Broadcasting utilities (explicit, non-automatic)
//!
//! Key differences from Tinygrad:
//! - Uses Rust's type system (SInt enum vs Python Union)
//! - Explicit Result types instead of exceptions
//! - Non-automatic broadcasting (must be explicit)

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use snafu::ensure;

use crate::ops;
use crate::{ConstValue, Op, Result, SInt, UOp, UOpKey, error::*};

/// Shape type - sequence of symbolic integers.
///
/// Uses SmallVec with inline capacity of 4 to avoid heap allocation for
/// common tensor ranks (1D-4D), which covers 99% of ML workloads.
///
/// Can contain mix of concrete and symbolic dimensions.
pub type Shape = SmallVec<[SInt; 4]>;

// =========================================================================
// Shape Utilities
// =========================================================================

/// Check if shape is fully concrete (all dimensions are constants).
///
/// # Examples
///
/// ```rust
/// # use svod_ir::{SInt, shape::is_static};
/// # use smallvec::smallvec;
/// let shape = smallvec![SInt::from(3), SInt::from(4), SInt::from(5)];
/// assert!(is_static(&shape));
/// ```
pub fn is_static(shape: &Shape) -> bool {
    shape.iter().all(|dim| dim.is_const())
}

/// Convert shape to concrete `Vec<usize>` if fully static, None otherwise.
///
/// # Examples
///
/// ```rust
/// # use svod_ir::{SInt, shape::to_static};
/// # use smallvec::smallvec;
/// let shape = smallvec![SInt::from(3), SInt::from(4)];
/// assert_eq!(to_static(&shape), Some(smallvec![3, 4]));
/// ```
pub fn to_static(shape: &Shape) -> Option<SmallVec<[usize; 4]>> {
    is_static(shape).then_some(shape.iter().map(|dim| dim.as_const().unwrap()).collect())
}

// =========================================================================
// Shape Validation
// =========================================================================

/// Validate that a shape specification is valid (all positive, no zeros).
///
/// # Errors
/// Returns error if any dimension is negative or zero.
///
/// # Examples
/// ```rust
/// # use svod_ir::shape::validate_shape;
/// let valid = vec![1, 2, 3];
/// assert!(validate_shape(&valid).is_ok());
/// let invalid = vec![1, -2, 3];
/// assert!(validate_shape(&invalid).is_err());
/// ```
pub fn validate_shape(shape: &[isize]) -> Result<SmallVec<[usize; 4]>> {
    ensure!(shape.iter().all(|&s| s >= 0), ReshapeNegativeDimensionSnafu { shape });
    Ok(shape.iter().map(|&s| s as usize).collect())
}

/// Check if two shapes are equal.
///
/// Uses pointer equality for symbolic dimensions (consistent with hash consing).
pub fn shapes_equal(lhs: &Shape, rhs: &Shape) -> bool {
    lhs == rhs
}

/// Tinygrad-style upper-bound shape equality (`p.max_shape != a.max_shape`).
///
/// Materialises each `SInt` to its `vmax` (resolving symbolic to its known
/// upper bound) and compares the resulting concrete shapes. This is what
/// FUNCTION param/arg shape matching needs: two distinct symbolic dims with
/// the same upper bound are considered equal, while differing concrete
/// extents still mismatch.
pub fn max_shapes_equal(lhs: &Shape, rhs: &Shape) -> bool {
    if lhs.len() != rhs.len() {
        return false;
    }
    lhs.iter().zip(rhs.iter()).all(|(a, b)| match (a.vmax(), b.vmax()) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    })
}

/// Check if all shapes in a slice are equal.
///
/// # Examples
///
/// ```rust
/// # use svod_ir::{SInt, shape::all_shapes_equal};
/// # use smallvec::smallvec;
/// let shape1 = smallvec![SInt::from(3), SInt::from(4)];
/// let shape2 = smallvec![SInt::from(3), SInt::from(4)];
/// let shape3 = smallvec![SInt::from(3), SInt::from(4)];
/// assert!(all_shapes_equal(&[shape1, shape2, shape3]));
/// ```
pub fn all_shapes_equal(shapes: &[Shape]) -> bool {
    (!shapes.is_empty()) && shapes.iter().all(|s| shapes_equal(s, &shapes[0]))
}

// =========================================================================
// Broadcasting Utilities (Explicit, Non-automatic)
// =========================================================================

/// Align shapes to the left by prepending 1s.
///
/// Makes all shapes have the same number of dimensions by adding dimensions
/// of size 1 on the left.
///
/// # Examples
///
/// ```rust
/// # use svod_ir::{SInt, shape::align_shapes_left};
/// # use smallvec::smallvec;
/// let shape1 = smallvec![SInt::from(5)];
/// let shape2 = smallvec![SInt::from(3), SInt::from(5)];
/// let aligned = align_shapes_left(&[shape1, shape2]);
/// assert_eq!(aligned.len(), 2);
/// assert_eq!(aligned[0].len(), 2); // [1, 5]
/// assert_eq!(aligned[1].len(), 2); // [3, 5]
/// ```
pub fn align_shapes_left(shapes: &[Shape]) -> Vec<Shape> {
    if shapes.is_empty() {
        return Vec::new();
    }

    let max_dims = shapes.iter().map(|s| s.len()).max().unwrap();

    shapes
        .iter()
        .map(|shape| {
            let padding = max_dims - shape.len();
            let mut aligned = SmallVec::with_capacity(max_dims);
            aligned.extend(std::iter::repeat_n(SInt::from(1), padding));
            aligned.extend(shape.iter().cloned());
            aligned
        })
        .collect()
}

/// Check if two shapes can be broadcast together (NumPy-style broadcasting).
///
/// Two shapes are broadcastable if:
/// - They have the same number of dimensions
/// - For each dimension, either the dimensions match or one of them is 1
///
/// # Examples
///
/// ```rust
/// # use svod_ir::{SInt, shape::can_broadcast};
/// # use smallvec::smallvec;
/// let shape1 = smallvec![SInt::from(1), SInt::from(5)];
/// let shape2 = smallvec![SInt::from(3), SInt::from(5)];
/// assert!(can_broadcast(&shape1, &shape2));
///
/// let shape3 = smallvec![SInt::from(3), SInt::from(4)];
/// assert!(!can_broadcast(&shape1, &shape3));
/// ```
pub fn can_broadcast(lhs: &Shape, rhs: &Shape) -> bool {
    if lhs.len() != rhs.len() {
        return false;
    }

    lhs.iter().zip(rhs.iter()).all(|(l, r)| {
        // If both are concrete, check broadcast rule
        if let (Some(lv), Some(rv)) = (l.as_const(), r.as_const()) {
            lv == rv || lv == 1 || rv == 1
        } else if l == r {
            // Same symbolic expression
            true
        } else {
            // Different symbolic expressions - conservatively assume compatible
            // (runtime check would be needed)
            true
        }
    })
}

/// Compute the broadcast result shape for two shapes.
///
/// Returns the shape that results from broadcasting the two input shapes.
/// Both shapes must be broadcastable (checked with can_broadcast).
///
/// # Errors
/// Returns error if shapes are not broadcastable.
///
/// # Examples
///
/// ```rust
/// # use svod_ir::{SInt, shape::broadcast_shape};
/// # use smallvec::smallvec;
/// let shape1 = smallvec![SInt::from(1), SInt::from(5)];
/// let shape2 = smallvec![SInt::from(3), SInt::from(5)];
/// let result = broadcast_shape(&shape1, &shape2).unwrap();
/// assert_eq!(result[0].as_const(), Some(3));
/// assert_eq!(result[1].as_const(), Some(5));
/// ```
pub fn broadcast_shape(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    use crate::error::BroadcastShapeMismatchSnafu;
    use snafu::ensure;

    ensure!(lhs.len() == rhs.len(), BroadcastShapeMismatchSnafu { lhs: lhs.clone(), rhs: rhs.clone() });

    let mut result = SmallVec::with_capacity(lhs.len());

    for (l, r) in lhs.iter().zip(rhs.iter()) {
        if l == r {
            // Same dimension (concrete value or symbolic expression)
            result.push(l.clone());
        } else if l.as_const() == Some(1) {
            // NumPy broadcasting: size-1 dim expands to the other
            result.push(r.clone());
        } else if r.as_const() == Some(1) {
            // NumPy broadcasting: size-1 dim expands to the other
            result.push(l.clone());
        } else if l.as_const().is_some() && r.as_const().is_some() {
            // Both concrete, non-1, and not equal → error
            return BroadcastShapeMismatchSnafu { lhs: lhs.clone(), rhs: rhs.clone() }.fail();
        } else {
            // At least one is symbolic (non-1) - use max (conservatively)
            result.push(crate::sint_max(&[l.clone(), r.clone()]));
        }
    }

    Ok(result)
}

/// Compute broadcast result for multiple shapes.
///
/// # Errors
/// Returns error if any pair of shapes is not broadcastable.
pub fn broadcast_shapes(shapes: &[Shape]) -> Result<Shape> {
    if shapes.is_empty() {
        return Ok(SmallVec::new());
    }

    // Align all shapes to same number of dimensions
    let aligned = align_shapes_left(shapes);

    // Successively broadcast pairs
    let mut result = aligned[0].clone();
    for shape in &aligned[1..] {
        result = broadcast_shape(&result, shape)?;
    }

    Ok(result)
}

/// Convert shape to `Vec<usize>`, ensuring all dimensions are concrete.
///
/// This is a helper function to reduce boilerplate when converting shapes
/// for operations that require concrete (non-symbolic) dimensions.
///
/// # Errors
///
/// Returns error if any dimension contains a symbolic (non-const) value.
pub fn to_vec_usize(shape: &Shape) -> Result<Vec<usize>> {
    shape
        .iter()
        .map(|dim| dim.as_const().ok_or_else(|| Error::SymbolicShapeUnsupported { operation: "shape conversion" }))
        .collect()
}

/// Convert shape to `Vec<isize>`, ensuring all dimensions are concrete.
///
/// # Errors
///
/// Returns error if any dimension contains a symbolic (non-const) value.
pub fn to_vec_isize(shape: &Shape) -> Result<Vec<isize>> {
    shape
        .iter()
        .map(|dim| {
            dim.as_const()
                .map(|v| v as isize)
                .ok_or_else(|| Error::SymbolicShapeUnsupported { operation: "shape conversion" })
        })
        .collect()
}

// =========================================================================
// Movement Op Argument Extraction (marg equivalent)
// =========================================================================

/// Extract shape dimensions from a STACK or CONST UOp.
///
/// Following Tinygrad's `marg` pattern, this extracts concrete or symbolic
/// dimensions from the UOp used to store shape information.
///
/// Returns None if the UOp is not in the expected format.
fn extract_shape_from_uop(shape_uop: &Arc<UOp>) -> Option<Shape> {
    match shape_uop.op() {
        // A cast around an aggregate shape payload is representation-only. A
        // cast around a scalar expression is itself the symbolic dimension,
        // matching Tinygrad's shape_to_shape_arg/marg behavior.
        Op::Cast(ops::Cast { src, .. }) | Op::BitCast(ops::BitCast { src, .. })
            if matches!(src.op(), Op::Stack(..) | Op::VConst(..) | Op::Const(_)) =>
        {
            extract_shape_from_uop(src)
        }

        Op::Stack(ops::Stack { sources }) => Some(sources.iter().cloned().map(SInt::from).collect()),

        // Single CONST value (for 1D shapes)
        Op::Const(const_hash) => match const_hash.0 {
            ConstValue::Int(v) if v >= 0 => Some(smallvec![SInt::from(v as usize)]),
            ConstValue::UInt(v) => Some(smallvec![SInt::from(v as usize)]),
            _ => None,
        },

        // VConst for multiple concrete dimensions
        Op::VConst(ops::VConst { values }) => {
            let mut dims = SmallVec::with_capacity(values.len());
            for val in values {
                match val {
                    ConstValue::Int(v) if *v >= 0 => dims.push(SInt::from(*v as usize)),
                    ConstValue::UInt(v) => dims.push(SInt::from(*v as usize)),
                    _ => return None,
                }
            }
            Some(dims)
        }

        // A one-dimensional symbolic shape is represented directly by its
        // scalar integer expression rather than a one-element STACK.
        _ if shape_uop.dtype().is_int() && shape_uop.shape().ok().flatten().is_some_and(|shape| shape.is_empty()) => {
            Some(smallvec![SInt::Symbolic(shape_uop.clone())])
        }

        _ => None,
    }
}

fn display_slot(slot: usize) -> isize {
    if slot == usize::MAX { -1 } else { slot as isize }
}

fn actual_for_formal(slot: usize, args: &[Arc<UOp>]) -> crate::Result<&Arc<UOp>> {
    let actual = if slot == usize::MAX { args.last() } else { args.get(slot) };
    actual.ok_or(crate::Error::CallFormalSlotMissing { slot: display_slot(slot), arg_count: args.len() })
}

/// Build the pinned FUNCTION formal-to-actual map used when inlining a body.
///
/// PARAM slots are positional and may be sparse; unused actual arguments are
/// valid. Tinygrad's scalar slot `-1` is represented by `usize::MAX` and is a
/// free body variable during execution, so it is excluded here.
pub fn function_param_substitutions(body: &Arc<UOp>, args: &[Arc<UOp>]) -> crate::Result<HashMap<UOpKey, Arc<UOp>>> {
    let mut substitutions = HashMap::new();
    for formal in body.toposort_call_aware(false) {
        let Op::Param(ops::Param { arg, .. }) = formal.op() else { continue };
        if arg.slot == usize::MAX {
            continue;
        }

        let actual = actual_for_formal(arg.slot, args)?;
        let actual_axis = match actual.op() {
            Op::Param(ops::Param { arg, .. }) | Op::Buffer(ops::Buffer { arg, .. }) => arg.axis,
            _ => None,
        };
        if arg.axis != actual_axis {
            return Err(crate::Error::CallArgAxisMismatch {
                arg_index: arg.slot,
                expected: arg.axis,
                got: actual_axis,
            });
        }

        let expected_shape = formal.shape()?.cloned();
        let got_shape = actual.shape()?.cloned();
        if !matches!((&expected_shape, &got_shape), (Some(expected), Some(got)) if max_shapes_equal(expected, got)) {
            return Err(crate::Error::CallArgShapeMismatch {
                arg_index: arg.slot,
                expected: expected_shape.map(Box::new),
                got: got_shape.map(Box::new),
            });
        }
        if formal.dtype() != actual.dtype() {
            return Err(crate::Error::CallArgDTypeMismatch {
                arg_index: arg.slot,
                expected: formal.dtype(),
                got: actual.dtype(),
            });
        }

        substitutions.insert(UOpKey(formal), actual.clone());
    }
    Ok(substitutions)
}

/// Substitute only PARAMs reachable from one selected FUNCTION output shape.
///
/// This intentionally does not inspect or validate the rest of the FUNCTION
/// body. Tinygrad rewrites each selected `inner_shape` independently, including
/// Python slot `-1` selecting the last call argument.
pub fn substitute_selected_shape(shape: &Shape, _function: &Arc<UOp>, args: &[Arc<UOp>]) -> crate::Result<Shape> {
    let mut substitutions = HashMap::new();
    for dim in shape {
        let SInt::Symbolic(expr) = dim else { continue };
        for formal in expr.toposort_call_aware(false) {
            let Op::Param(ops::Param { arg, .. }) = formal.op() else { continue };
            let actual = actual_for_formal(arg.slot, args)?;
            if actual.dtype() == svod_dtype::DType::Void {
                return Err(crate::Error::CallShapeSubstitutionUnsupported {
                    slot: display_slot(arg.slot),
                    reason: "void actual argument cannot be used as a symbolic value".into(),
                });
            }
            substitutions.insert(UOpKey(formal), actual.clone());
        }
    }

    let formal_ids: HashMap<u64, isize> = substitutions
        .keys()
        .map(|key| {
            let Op::Param(ops::Param { arg, .. }) = key.0.op() else { unreachable!("substitution key must be PARAM") };
            (key.0.id, display_slot(arg.slot))
        })
        .collect();
    let caller_ids: HashSet<u64> =
        args.iter().flat_map(|arg| arg.toposort_call_aware(false)).map(|node| node.id).collect();

    let mut result = Shape::with_capacity(shape.len());
    let mut dangling = Vec::new();
    for dim in shape {
        let SInt::Symbolic(expr) = dim else {
            result.push(dim.clone());
            continue;
        };
        let rewritten = expr.substitute_walk_preserve_calls(&substitutions);
        for node in rewritten.toposort_call_aware(false) {
            if let Some(slot) = formal_ids.get(&node.id)
                && !caller_ids.contains(&node.id)
            {
                dangling.push(*slot);
            }
        }
        result.push(SInt::from(rewritten));
    }
    dangling.sort_unstable();
    dangling.dedup();
    if !dangling.is_empty() {
        return Err(crate::Error::CallShapeDanglingFormal { slots: dangling });
    }
    Ok(result)
}

/// Extract padding/shrink ranges from UOps.
///
/// Returns pairs of (begin, end) for each dimension.
fn extract_ranges_from_uops(begins_uop: &Arc<UOp>, ends_uop: &Arc<UOp>) -> Option<Vec<(SInt, SInt)>> {
    let begins = extract_shape_from_uop(begins_uop)?;
    let ends = extract_shape_from_uop(ends_uop)?;

    if begins.len() != ends.len() {
        return None;
    }

    Some(begins.into_iter().zip(ends).collect())
}

/// Convert a Shape to Tinygrad's scalar/STACK shape argument encoding.
///
/// This creates a UOp that encodes the shape dimensions, suitable for
/// passing to Reshape, Expand, etc.
///
/// # Examples
///
/// ```rust
/// # use svod_ir::{SInt, shape::shape_to_uop};
/// # use svod_dtype::DType;
/// # use smallvec::smallvec;
/// let shape = smallvec![SInt::from(3), SInt::from(4), SInt::from(5)];
/// let shape_uop = shape_to_uop(&shape);
/// assert_eq!(shape_uop.dtype(), DType::WeakInt);
///
/// // Scalar (empty shape) is supported
/// let scalar_shape: smallvec::SmallVec<[SInt; 4]> = smallvec![];
/// let scalar_uop = shape_to_uop(&scalar_shape);
/// // Empty STACK represents scalar
/// ```
pub fn shape_to_uop(shape: &Shape) -> Arc<UOp> {
    use smallvec::SmallVec;
    use svod_dtype::DType;

    if shape.is_empty() {
        return UOp::stack(SmallVec::new());
    }
    if shape.len() == 1 {
        return shape[0].to_uop(DType::WeakInt);
    }

    // STACK unifies its lane dtypes, so materialise every dim at the promoted
    // dtype up front. Emitting weak constants and letting STACK wrap them in a
    // CAST would make mixed const/symbolic shapes read back fully symbolic.
    let lanes: SmallVec<[DType; 4]> = shape
        .iter()
        .map(|dim| match dim {
            SInt::Symbolic(value) => value.dtype(),
            _ => DType::WeakInt,
        })
        .collect();
    let lane_dtype = if lanes.iter().all(|dtype| *dtype == lanes[0]) {
        lanes[0].clone()
    } else {
        DType::least_upper_dtype(&lanes).unwrap_or(DType::WeakInt)
    };
    UOp::stack(shape.iter().map(|dim| dim.to_uop(lane_dtype.clone())).collect())
}

/// Transpose per-axis pairs into the two shape-argument UOps PAD and SHRINK
/// carry, using the scalar/STACK encoding.
///
/// The pair means what the operation means: `(begin_pad, end_pad)` for PAD,
/// `(offset, size)` for SHRINK (tinygrad `ops.py` `marg`: "SHRINK marg is
/// (start, length)").
///
/// # Panics
/// Panics if `pairs` is empty; handle scalars at the callsite.
pub fn ranges_to_uops(pairs: &[(SInt, SInt)]) -> (Arc<UOp>, Arc<UOp>) {
    use smallvec::SmallVec;
    use svod_dtype::DType;

    assert!(!pairs.is_empty(), "ranges_to_uops does not support empty ranges (scalars); handle at callsite");

    let firsts: SmallVec<[Arc<UOp>; 4]> = pairs.iter().map(|(first, _)| first.to_uop(DType::WeakInt)).collect();
    let seconds: SmallVec<[Arc<UOp>; 4]> = pairs.iter().map(|(_, second)| second.to_uop(DType::WeakInt)).collect();

    let encode = |values: SmallVec<[Arc<UOp>; 4]>| {
        if values.len() == 1 { values[0].clone() } else { UOp::stack(values) }
    };
    (encode(firsts), encode(seconds))
}

// =========================================================================
// Shape Inference (Tinygrad-style)
// =========================================================================

/// Infer shape from a UOp's operation.
///
/// This is the core shape inference function, following Tinygrad's approach.
/// Returns None for operations without a well-defined shape (control flow, etc.).
///
/// # Shape Inference Rules
///
/// - **Nullary ops** (Const, VConst): Return concrete shape
/// - **Unary ops**: Preserve input shape
/// - **Binary ops**: Validate inputs match, return common shape
/// - **Ternary ops**: Return shape of value branches
/// - **Movement ops**: Compute shape from operation arguments
/// - **Reduce ops**: Compute reduced shape
/// - **Late/control flow ops**: Return None
pub fn infer_shape_from_op(uop: &UOp) -> crate::Result<Option<Shape>> {
    Ok(match uop.op() {
        // =====================================================================
        // Nullary operations
        // =====================================================================
        Op::Const(_) => Some(SmallVec::new()), // Scalar has empty shape

        Op::VConst(..) => None,

        Op::Stack(ops::Stack { sources }) => {
            if sources.is_empty() {
                Some(SmallVec::new())
            } else {
                let source_shape = sources[0].shape()?.ok_or_else(|| crate::Error::VoidTypeInOp)?;
                if sources.iter().skip(1).any(|source| source.shape().ok().flatten() != Some(source_shape)) {
                    return Ok(None);
                }
                let mut shape = smallvec![SInt::from(sources.len())];
                shape.extend(source_shape.iter().cloned());
                Some(shape)
            }
        }

        Op::Shuffle(ops::Shuffle { src, mask }) => {
            let mut shape = src.shape()?.ok_or_else(|| crate::Error::VoidTypeInOp)?.clone();
            shape[0] = SInt::from(mask.len());
            Some(shape)
        }

        Op::Unique(_) | Op::LUnique(_) | Op::Noop => None,

        // =====================================================================
        // Unary operations - preserve shape
        // =====================================================================
        Op::Unary(_, input) => input.shape()?.cloned(),

        // =====================================================================
        // Elementwise operations use NumPy-style broadcasting. The expander
        // materializes these broadcasts before devectorization.
        // =====================================================================
        Op::Binary(_op, lhs, rhs) => match (lhs.shape()?, rhs.shape()?) {
            (Some(lhs_shape), Some(rhs_shape)) => Some(broadcast_shapes(&[lhs_shape.clone(), rhs_shape.clone()])?),
            (Some(shape), _) | (_, Some(shape)) => Some(shape.clone()),
            (None, None) => None,
        },

        // =====================================================================
        // Ternary operations
        // =====================================================================
        Op::Ternary(_, condition, true_val, false_val) => {
            let shapes = [condition, true_val, false_val]
                .into_iter()
                .filter_map(|source| source.shape().transpose())
                .map(|shape| shape.cloned())
                .collect::<Result<Vec<_>>>()?;
            if shapes.is_empty() { None } else { Some(broadcast_shapes(&shapes)?) }
        }

        // =====================================================================
        // Type operations
        // =====================================================================
        Op::Cast(ops::Cast { src, .. }) => src.shape()?.cloned(),
        // BitCast: byte-reinterpretation. Same itemsize → same shape.
        // Different itemsize → adjust last dimension (Tinygrad tensor.py:3549-3568).
        // BitCast: byte-reinterpretation (Tinygrad ops.py:240-245).
        // Same itemsize → same shape. Different itemsize → adjust last dimension.
        Op::BitCast(ops::BitCast { src, dtype }) => {
            let src_shape = src.shape()?;
            match src_shape {
                Some(shape) if !shape.is_empty() => {
                    let src_bytes = src.dtype().bytes();
                    let dst_bytes = dtype.bytes();
                    if src_bytes == dst_bytes {
                        Some(shape.clone())
                    } else {
                        // Adjust last dimension: (last * src_bytes) / dst_bytes
                        let mut new_shape = shape.clone();
                        let last = new_shape.last().unwrap().clone();
                        let new_last = (last * SInt::Const(src_bytes)) / SInt::Const(dst_bytes);
                        *new_shape.last_mut().unwrap() = new_last;
                        Some(new_shape)
                    }
                }
                other => other.cloned(),
            }
        }

        // =====================================================================
        // Movement operations
        // =====================================================================
        Op::Reshape(ops::Reshape { new_shape, .. }) => {
            // Extract shape from STACK/VCONST/CONST UOps.
            extract_shape_from_uop(new_shape)
        }

        Op::Permute(ops::Permute { axes, src }) => {
            let src_shape = src.shape()?.ok_or_else(|| crate::Error::VoidTypeInOp)?;
            // Reorder dimensions according to permutation
            Some(axes.iter().map(|&i| src_shape[i].clone()).collect())
        }

        Op::Expand(ops::Expand { new_shape, .. }) => {
            // Extract shape from STACK/VCONST/CONST UOps.
            extract_shape_from_uop(new_shape)
        }

        Op::Pad(ops::Pad { src, begin_pads, end_pads }) => {
            let src_shape = src.shape()?.ok_or_else(|| crate::Error::VoidTypeInOp)?;
            let ranges = extract_ranges_from_uops(begin_pads, end_pads).ok_or_else(|| crate::Error::VoidTypeInOp)?;

            if src_shape.len() != ranges.len() {
                return Ok(None);
            }

            // New shape = src_shape + begin_pads + end_pads for each dimension
            Some(
                src_shape
                    .iter()
                    .zip(ranges.iter())
                    .map(|(dim, (begin, end))| Ok(dim + begin + end))
                    .collect::<crate::Result<Shape>>()?,
            )
        }

        Op::Shrink(ops::Shrink { src, offsets, sizes }) => {
            let src_shape = src.shape()?.ok_or_else(|| crate::Error::VoidTypeInOp)?;
            let ranges = extract_ranges_from_uops(offsets, sizes).ok_or_else(|| crate::Error::VoidTypeInOp)?;

            if src_shape.len() != ranges.len() {
                return Ok(None);
            }

            Some(ranges.into_iter().map(|(_, size)| size).collect())
        }

        Op::Flip(ops::Flip { src, .. }) => {
            // Flip preserves shape
            src.shape()?.cloned()
        }

        Op::Multi(ops::Multi { src, .. }) => {
            // Multi scales the specified axis by device count
            // TODO: Need device count from somewhere - for now preserve shape
            // Tinygrad: tuple(s*len(self.device) if a == self.axis else s for a,s in enumerate(ps))
            src.shape()?.cloned()
        }

        // =====================================================================
        // Reduction operations
        // =====================================================================
        Op::ReduceAxis(ops::ReduceAxis { axes, src, .. }) => {
            let src_shape = src.shape()?.ok_or_else(|| crate::Error::VoidTypeInOp)?;
            // Set reduced axes to 1 (don't remove them - matches Tinygrad)
            Some(
                src_shape
                    .iter()
                    .enumerate()
                    .map(|(i, dim)| if axes.contains(&i) { SInt::from(1) } else { dim.clone() })
                    .collect(),
            )
        }

        Op::Reduce(ops::Reduce { src, num_axes, .. }) => {
            let src_shape = src.shape()?.ok_or_else(|| crate::Error::VoidTypeInOp)?;
            if *num_axes > src_shape.len() {
                return Err(crate::Error::ReduceInvalidNumAxes { num_axes: *num_axes, shape_dims: src_shape.len() });
            }
            Some(src_shape.iter().skip(*num_axes).cloned().collect())
        }

        Op::AllReduce(ops::AllReduce { src, .. }) => {
            // AllReduce preserves shape
            src.shape()?.cloned()
        }

        // =====================================================================
        // Buffer and memory operations - shape depends on buffer
        // =====================================================================
        Op::Buffer(ops::Buffer { shape, .. }) | Op::Param(ops::Param { shape, .. }) => extract_shape_from_uop(shape),
        Op::Slice(ops::Slice { size, .. }) => Some(smallvec![SInt::from(*size)]),

        // Passthrough operations
        Op::Copy(ops::Copy { src, .. }) => src.shape()?.cloned(),
        Op::MStack(ops::MStack { buffers }) => match buffers.first() {
            Some(b) => b.shape()?.cloned(),
            None => None,
        },

        // STAGE prepends its closed range extents to the compute shape.
        Op::Stage(ops::Stage { compute, ranges, .. }) => {
            let mut dims: Shape = SmallVec::new();
            for range in ranges.iter() {
                match range.op() {
                    // Range: shape dim = end (the upper bound)
                    Op::Range(ops::Range { end, .. }) => {
                        // Try to get constant value from end
                        if let Op::Const(val) = end.op() {
                            match val.0 {
                                ConstValue::Int(v) if v >= 0 => {
                                    dims.push(SInt::Const(v as usize));
                                    continue;
                                }
                                ConstValue::UInt(v) => {
                                    dims.push(SInt::Const(v as usize));
                                    continue;
                                }
                                _ => {}
                            }
                        }
                        // Fall back to symbolic
                        dims.push(SInt::Symbolic(end.clone()));
                    }
                    // CONST range (already dead axis) has size from vmax+1
                    Op::Const(val) => {
                        match val.0 {
                            ConstValue::Int(v) if v >= 0 => {
                                dims.push(SInt::Const((v + 1) as usize)); // vmax+1 for shape
                            }
                            ConstValue::UInt(v) => {
                                dims.push(SInt::Const((v + 1) as usize)); // vmax+1 for shape
                            }
                            _ => return Ok(None), // Can't determine shape
                        }
                    }
                    // Other range types: use symbolic
                    _ => {
                        dims.push(SInt::Symbolic(range.clone()));
                    }
                }
            }
            let Some(compute_shape) = compute.shape()? else { return Ok(None) };
            dims.extend(compute_shape.iter().cloned());
            Some(dims)
        }

        Op::Index(ops::Index { buffer, indices }) => {
            let mut shape = Shape::new();
            for index in indices {
                let Some(index_shape) = index.shape()? else { return Ok(None) };
                shape.extend(index_shape.iter().cloned());
            }
            let Some(buffer_shape) = buffer.shape()? else { return Ok(None) };
            shape.extend(buffer_shape.iter().skip(indices.len()).cloned());
            Some(shape)
        }
        Op::Load(ops::Load { index, .. }) | Op::Store(ops::Store { index, .. }) => index.shape()?.cloned(),

        // =====================================================================
        // Control flow - no static shape
        // =====================================================================
        Op::Range(..) => Some(SmallVec::new()),
        Op::If(..) | Op::EndIf(..) | Op::Barrier(..) => None,

        // End passes through the computation shape
        Op::End(ops::End { computation, .. }) => computation.shape()?.cloned(),

        // =====================================================================
        // Special operations
        // =====================================================================
        // MSelect passes through buffer shape
        Op::MSelect(ops::MSelect { buffer, .. }) => buffer.shape()?.cloned(),

        Op::Special(..) => Some(SmallVec::new()),

        Op::DefineVar(..) => Some(SmallVec::new()), // Variable is scalar

        Op::Bind(ops::Bind { value, .. }) => value.shape()?.cloned(),

        // =====================================================================
        // Advanced operations
        // =====================================================================
        Op::Wmma(ops::Wmma { a, b, c, .. }) => {
            let (Some(a_shape), Some(b_shape), Some(c_shape)) = (a.shape()?, b.shape()?, c.shape()?) else {
                return Ok(None);
            };
            let Some((c_width, c_prefix)) = c_shape.split_last() else {
                return Ok(None);
            };
            let a_prefix = &a_shape[..a_shape.len().saturating_sub(1)];
            let b_prefix = &b_shape[..b_shape.len().saturating_sub(1)];
            let mut shape = broadcast_shapes(&[a_prefix.into(), b_prefix.into(), c_prefix.into()])?;
            shape.push(c_width.clone());
            Some(shape)
        }
        Op::Program(..) | Op::Linear(..) | Op::Source(..) | Op::ProgramBinary(..) => None,
        // INS shape is scalar; vector width is part of the target encoding.
        Op::Ins(..) => (uop.dtype() != svod_dtype::DType::Void).then(SmallVec::new),
        // FUNCTION is a void tuple-producing wrapper. A void CALL has no
        // shape; typed instruction-style CALLs are scalar, independent of the
        // opaque implementation body.
        Op::Function(..) => None,
        Op::Call(..) => (uop.dtype() != svod_dtype::DType::Void).then(SmallVec::new),
        // TUPLE is a void-typed grouping; it has no shape itself.
        Op::Tuple(..) => None,
        // GETTUPLE returns the shape of its inner element when the source is a TUPLE
        // (or a FUNCTION whose body is a TUPLE).
        Op::GetTuple(ops::GetTuple { src, index }) => match src.op() {
            Op::Tuple(ops::Tuple { src: tuple_src }) => tuple_src
                .get(*index)
                .ok_or(crate::Error::GetTupleIndexOutOfBounds { index: *index, len: tuple_src.len(), kind: "TUPLE" })?
                .shape()?
                .cloned(),
            Op::Function(ops::Function { body, args, .. }) => match body.op() {
                Op::Tuple(ops::Tuple { src: tuple_src }) => tuple_src
                    .get(*index)
                    .ok_or(crate::Error::GetTupleIndexOutOfBounds {
                        index: *index,
                        len: tuple_src.len(),
                        kind: "FUNCTION(TUPLE)",
                    })?
                    .shape()?
                    .map(|shape| substitute_selected_shape(shape, body, args))
                    .transpose()?,
                _ => None,
            },
            _ => None,
        },

        Op::Detach(ops::Detach { src })
        | Op::Contiguous(ops::Contiguous { src, .. })
        | Op::ContiguousBackward(ops::ContiguousBackward { src })
        | Op::Precast(ops::Precast { src }) => src.shape()?.cloned(),

        Op::After(ops::After { passthrough, .. }) => passthrough.shape()?.cloned(),

        Op::Custom(..) | Op::CustomI(..) | Op::CustomFunction(..) => None,

        // Graph organization operations have no shape
        Op::Sink(..) | Op::Group(..) => None,

        Op::GetAddr(..) => Some(smallvec![]),
    })
}
