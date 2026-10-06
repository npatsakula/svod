//! One evaluator for every symbolic test: bindings for the named operands, a
//! dtype-width-respecting fold, and the range sweep that feeds equivalence checks.

use std::collections::HashMap;
use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_dtype::ScalarDType;
use svod_ir::uop::eval::{eval_binary_op_typed, eval_ternary_op_typed, eval_unary_op_typed};
use svod_ir::{ConstValue, DType, Op, UOp, UOpKey, ops};

use super::vars::{var_name, var_range};

/// Values for symbolic operands, keyed by name (a node binds its own name).
#[derive(Debug, Clone, Default)]
pub struct Bindings(SmallVec<[(Arc<UOp>, ConstValue); 4]>);

impl Bindings {
    /// No operands are pinned; only fully constant expressions evaluate.
    pub fn none() -> Self {
        Self(SmallVec::new())
    }

    /// Pin the operand named `name`.
    pub fn at(name: &str, value: impl Into<ConstValue>) -> Self {
        Self(smallvec![(named_var(name), value.into())])
    }

    /// Pin one more operand, keeping the existing bindings.
    pub fn with(mut self, name: &str, value: impl Into<ConstValue>) -> Self {
        self.0.push((named_var(name), value.into()));
        self
    }

    fn from_pairs(pairs: SmallVec<[(Arc<UOp>, ConstValue); 4]>) -> Self {
        Self(pairs)
    }

    /// The value pinned for `uop`, either by node identity or by declared name.
    fn get(&self, uop: &Arc<UOp>) -> Option<ConstValue> {
        let name = var_name(uop);
        self.0
            .iter()
            .rev()
            .find(|(key, _)| Arc::ptr_eq(key, uop) || (name.is_some() && var_name(key) == name))
            .map(|(_, value)| *value)
    }

    /// The constants to substitute for every bound operand reachable from `expr`.
    fn substitution(&self, expr: &Arc<UOp>) -> HashMap<UOpKey, Arc<UOp>> {
        expr.toposort()
            .into_iter()
            .filter_map(|node| self.get(&node).map(|value| (UOpKey(node.clone()), UOp::const_(node.dtype(), value))))
            .collect()
    }
}

/// A stand-in variable carrying only the bound name.
fn named_var(name: &str) -> Arc<UOp> {
    UOp::define_var(name.to_string(), 0, 0)
}

/// Evaluate `expr` with `bindings`, narrowing every node to its own dtype width so
/// wrapping is observable; `None` outside the evaluator's domain or on an unbound operand.
pub fn eval_typed(expr: &Arc<UOp>, bindings: &Bindings) -> Option<ConstValue> {
    let dtype = expr.dtype().base();
    match expr.op() {
        Op::Const(value) => Some(value.0),
        Op::DefineVar(..) | Op::Param(..) => {
            if let Some(value) = bindings.get(expr) {
                return value.cast(&DType::Scalar(dtype));
            }
            let (lo, hi) = var_range(expr)?;
            (lo == hi).then(|| ConstValue::Int(lo).cast(&DType::Scalar(dtype)))?
        }
        // A RANGE and a SPECIAL are both variables over `[0, end - 1]`: they
        // evaluate when the sweep pins them, and a trip-1 axis takes only zero.
        Op::Range(ops::Range { end, .. }) | Op::Special(ops::Special { end, .. }) => {
            if let Some(value) = bindings.get(expr) {
                return value.cast(&DType::Scalar(dtype));
            }
            let ConstValue::Int(1) = eval_typed(end, bindings)? else { return None };
            ConstValue::Int(0).cast(&DType::Scalar(dtype))
        }
        Op::Bind(ops::Bind { var, value }) => {
            bindings.get(var).or_else(|| eval_typed(value, bindings)).or_else(|| eval_typed(var, bindings))
        }
        Op::Cast(ops::Cast { src, .. }) => eval_typed(src, bindings)?.cast(&DType::Scalar(dtype)),
        Op::BitCast(ops::BitCast { src, .. }) => reinterpret(eval_typed(src, bindings)?, dtype),
        Op::Unary(op, src) => eval_unary_op_typed(*op, eval_typed(src, bindings)?, dtype),
        Op::Binary(op, lhs, rhs) => {
            let (lhs, rhs) = (eval_typed(lhs, bindings)?, eval_typed(rhs, bindings)?);
            eval_binary_op_typed(*op, lhs, rhs, dtype).or_else(|| widened_bool_binary(*op, lhs, rhs, dtype))
        }
        Op::Ternary(op, a, b, c) => eval_ternary_op_typed(
            *op,
            eval_typed(a, bindings)?,
            eval_typed(b, bindings)?,
            eval_typed(c, bindings)?,
            dtype,
        ),
        _ => None,
    }
}

/// Substitute `bindings` into `expr` and evaluate the pinned tree; equals [`eval_typed`].
pub fn fold_at(expr: &Arc<UOp>, bindings: &Bindings) -> Option<ConstValue> {
    let map = bindings.substitution(expr);
    let pinned = if map.is_empty() { expr.clone() } else { expr.substitute(&map) };
    eval_typed(&pinned, &Bindings::none())
}

/// `cap` points spread over the cartesian product of the operands' declared ranges.
///
/// Walking the product in order would only ever vary the first operand: with a
/// span of 101 and a cap of 64 every later operand stays pinned at its `lo`, so
/// a rule over two variables is checked at one value of the second. Stepping by a
/// stride coprime to the product fixes that and stays a bijection, so a cap at or
/// above the product still enumerates every point.
pub fn range_points(vars: &[Arc<UOp>], cap: usize) -> impl Iterator<Item = Bindings> + use<> {
    let ranges: Vec<(Arc<UOp>, i64, i64)> =
        vars.iter().filter_map(|var| var_range(var).map(|(lo, hi)| (var.clone(), lo, hi))).collect();
    let total = ranges
        .iter()
        .try_fold(1usize, |acc, (_, lo, hi)| acc.checked_mul((hi - lo + 1).max(1) as usize))
        .unwrap_or(usize::MAX);
    let stride = coprime_stride(total);
    (0..total.min(cap)).map(move |step| {
        let mut index = ((step as u128 * stride as u128) % total as u128) as usize;
        let mut pairs = SmallVec::new();
        for (var, lo, hi) in &ranges {
            let span = (hi - lo + 1).max(1) as usize;
            pairs.push((var.clone(), ConstValue::Int(lo + (index % span) as i64)));
            index /= span;
        }
        Bindings::from_pairs(pairs)
    })
}

/// A stride coprime to `total`, near the golden-ratio split so short prefixes
/// spread over every coordinate rather than clustering in the first.
fn coprime_stride(total: usize) -> usize {
    let mut stride = (((total as f64) / std::f64::consts::GOLDEN_RATIO) as usize).max(1);
    while stride < total && gcd(stride, total) != 1 {
        stride += 1;
    }
    if gcd(stride, total) == 1 { stride } else { 1 }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The arithmetic ops have no `Bool` arm upstream, but the IR does build them over
/// boolean operands — `MUL` is a conjunction, `ADD` and `MAX` a disjunction. Widen
/// to integers and let the node's own dtype commit the result back, exactly as
/// upstream's `commit_eval_result` would.
fn widened_bool_binary(
    op: svod_ir::types::BinaryOp,
    lhs: ConstValue,
    rhs: ConstValue,
    dtype: ScalarDType,
) -> Option<ConstValue> {
    let (ConstValue::Bool(lhs), ConstValue::Bool(rhs)) = (lhs, rhs) else { return None };
    eval_binary_op_typed(op, ConstValue::Int(lhs as i64), ConstValue::Int(rhs as i64), dtype)
}

/// Bit-level reinterpretation for `BITCAST`.
fn reinterpret(value: ConstValue, dtype: ScalarDType) -> Option<ConstValue> {
    use ScalarDType::*;
    let bits = match value {
        ConstValue::Invalid => return Some(ConstValue::Invalid),
        ConstValue::Bool(v) => v as u64,
        ConstValue::Int(v) => v as u64,
        ConstValue::UInt(v) => v,
        ConstValue::Float(v) => v.to_bits(),
    };
    Some(match dtype {
        Bool => ConstValue::Bool(bits & 1 != 0),
        Int8 => ConstValue::Int(bits as u8 as i8 as i64),
        Int16 => ConstValue::Int(bits as u16 as i16 as i64),
        Int32 => ConstValue::Int(bits as u32 as i32 as i64),
        Int64 | WeakInt | Index => ConstValue::Int(bits as i64),
        UInt8 => ConstValue::UInt(bits as u8 as u64),
        UInt16 => ConstValue::UInt(bits as u16 as u64),
        UInt32 => ConstValue::UInt(bits as u32 as u64),
        UInt64 => ConstValue::UInt(bits),
        Float32 => ConstValue::Float(f32::from_bits(bits as u32) as f64),
        Float64 => ConstValue::Float(f64::from_bits(bits)),
        _ => return None,
    })
}
