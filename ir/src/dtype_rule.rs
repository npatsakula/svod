//! Dtype production rules for UOps.
//!
//! This mirrors Tinygrad's `dtype_from_uop`. Operations whose dtype still
//! depends on legacy Svod metadata return `None` until that metadata moves into
//! the operation itself.

use svod_dtype::{DType, ScalarDType};

use crate::ops;
use crate::{BinaryOp, ConstValue, Op, TernaryOp, UOp, UnaryOp};

fn promote(src: impl IntoIterator<Item = DType>) -> Option<DType> {
    let dtypes: Vec<_> = src.into_iter().collect();
    let first = dtypes.first()?;
    if dtypes.iter().all(|dtype| dtype == first) { Some(first.clone()) } else { DType::least_upper_dtype(&dtypes) }
}

fn const_dtype(value: &ConstValue) -> DType {
    match value {
        ConstValue::Invalid => DType::Bool,
        ConstValue::Bool(_) => DType::Bool,
        ConstValue::Int(_) | ConstValue::UInt(_) => DType::WeakInt,
        ConstValue::Float(_) => DType::WeakFloat,
    }
}

/// Derive an operation's result dtype from its sources and metadata.
///
/// `None` means the current operation carries result-type information outside
/// the modern Tinygrad production rule and still requires an explicit dtype.
pub fn dtype_from_op(op: &Op) -> Option<DType> {
    match op {
        Op::Sink(..)
        | Op::Group(..)
        | Op::If(..)
        | Op::EndIf(..)
        | Op::End(..)
        | Op::Barrier(..)
        | Op::Tuple(..)
        | Op::Function(..)
        | Op::Program(..)
        | Op::Linear(..)
        | Op::Source(..)
        | Op::CustomFunction(..)
        | Op::Store(..)
        | Op::Unique(_)
        | Op::LUnique(_) => Some(DType::Void),

        Op::ProgramBinary(..) => Some(DType::UInt8),

        Op::Const(value) => Some(const_dtype(&value.0)),
        Op::Noop | Op::Custom(..) | Op::CustomI(..) | Op::VConst(..) | Op::Ins(..) => None,

        // These operations still keep their storage dtype outside Op metadata.
        Op::Param(ops::Param { arg, .. }) | Op::Buffer(ops::Buffer { arg, .. }) => Some(arg.dtype.clone()),
        Op::Slice(..) => None,
        Op::Index(ops::Index { buffer, indices }) => {
            Some(if matches!(buffer.op(), Op::Param(..)) && is_image_shape(buffer) {
                DType::Float32
            } else if !indices.is_empty() && buffer.dtype().vcount() > 1 && !is_storage_index_source(buffer) {
                buffer.dtype().scalar_dtype()
            } else {
                buffer.dtype()
            })
        }
        Op::Load(ops::Load { index, .. }) => Some(index.dtype()),
        Op::GetAddr(..) => Some(DType::UInt64),

        Op::Cast(ops::Cast { dtype, .. }) | Op::BitCast(ops::BitCast { dtype, .. }) => Some(dtype.clone()),

        Op::Unary(unary, src) => match unary {
            UnaryOp::Sqrt
            | UnaryOp::Rsqrt
            | UnaryOp::Exp
            | UnaryOp::Exp2
            | UnaryOp::Log
            | UnaryOp::Log2
            | UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Tan
            | UnaryOp::Reciprocal
            | UnaryOp::Erf => DType::least_upper_float(src.dtype()),
            _ => Some(src.dtype()),
        },
        Op::Binary(binary, lhs, rhs) if binary.is_comparison() => Some(DType::Bool),
        Op::Binary(BinaryOp::Shl | BinaryOp::Shr, lhs, _) => Some(lhs.dtype()),
        Op::Binary(_, lhs, rhs) => promote([lhs.dtype(), rhs.dtype()]),
        Op::Ternary(TernaryOp::Where, condition, true_value, false_value) => {
            if !condition.dtype().is_bool() {
                return None;
            }
            if UOp::is_invalid_marker(true_value) {
                Some(false_value.dtype())
            } else if UOp::is_invalid_marker(false_value) {
                Some(true_value.dtype())
            } else {
                promote([true_value.dtype(), false_value.dtype()])
            }
        }
        Op::Ternary(TernaryOp::MulAcc, a, b, c) => promote([a.dtype(), b.dtype(), c.dtype()]),

        Op::MSelect(ops::MSelect { buffer, .. })
        | Op::Copy(ops::Copy { src: buffer, .. })
        | Op::Stage(ops::Stage { compute: buffer, .. })
        | Op::Reshape(ops::Reshape { src: buffer, .. })
        | Op::Permute(ops::Permute { src: buffer, .. })
        | Op::Expand(ops::Expand { src: buffer, .. })
        | Op::Pad(ops::Pad { src: buffer, .. })
        | Op::Shrink(ops::Shrink { src: buffer, .. })
        | Op::Flip(ops::Flip { src: buffer, .. })
        | Op::Multi(ops::Multi { src: buffer, .. })
        | Op::ReduceAxis(ops::ReduceAxis { src: buffer, .. })
        | Op::Reduce(ops::Reduce { src: buffer, .. })
        | Op::AllReduce(ops::AllReduce { src: buffer, .. })
        | Op::Detach(ops::Detach { src: buffer })
        | Op::Contiguous(ops::Contiguous { src: buffer, .. })
        | Op::ContiguousBackward(ops::ContiguousBackward { src: buffer })
        | Op::After(ops::After { passthrough: buffer, .. })
        | Op::Precast(ops::Precast { src: buffer }) => Some(buffer.dtype()),

        Op::Special(ops::Special { end, .. }) => Some(end.dtype()),
        Op::Range(ops::Range { end, .. }) => Some(end.dtype()),
        Op::Bind(ops::Bind { var, value }) if var.dtype() == value.dtype() => Some(var.dtype()),
        Op::Bind(..) => None,
        Op::Wmma(ops::Wmma { c, .. }) => Some(c.dtype()),
        Op::MStack(ops::MStack { buffers }) => buffers.first().map(|buffer| buffer.dtype()),

        Op::Stack(ops::Stack { sources }) if sources.is_empty() => Some(DType::Void),
        Op::Stack(ops::Stack { sources }) => promote(sources.iter().map(|source| source.dtype())),
        Op::Shuffle(ops::Shuffle { src, mask })
            if src.dtype().is_vector() || src.dtype().scalar().is_some_and(|s| s != ScalarDType::Void) =>
        {
            Some(DType::Scalar(src.dtype().base()).vec(mask.len()).expect("a lane count"))
        }
        Op::Shuffle(..) => None,

        Op::DefineVar(..) => None,
        Op::Call(ops::Call { body, .. }) if body.dtype() == DType::Void => Some(DType::Void),
        Op::Call(..) => None,
        Op::GetTuple(ops::GetTuple { src, index }) => tuple_element(src, *index).map(|element| element.dtype()),
    }
}

fn tuple_element(src: &std::sync::Arc<UOp>, index: usize) -> Option<&std::sync::Arc<UOp>> {
    let tuple = match src.op() {
        Op::Function(ops::Function { body, .. }) => body,
        _ => src,
    };
    match tuple.op() {
        Op::Tuple(ops::Tuple { src }) => src.get(index),
        _ => None,
    }
}

fn is_image_shape(u: &std::sync::Arc<UOp>) -> bool {
    u.shape().ok().flatten().is_some_and(|shape| shape.len() == 3 && shape[2].as_const() == Some(4))
}

fn is_storage_index_source(u: &std::sync::Arc<UOp>) -> bool {
    match u.op() {
        Op::Param(..) | Op::Buffer(..) | Op::Slice(..) => true,
        Op::After(ops::After { passthrough, .. }) | Op::Precast(ops::Precast { src: passthrough }) => {
            is_storage_index_source(passthrough)
        }
        _ => false,
    }
}
