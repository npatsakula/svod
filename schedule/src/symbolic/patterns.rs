//! Symbolic simplification pattern definitions.
//!
//! Defines the core symbolic simplification patterns for algebraic optimization.
//!
//! This module contains:
//! - Constant folding (const op const → const)
//! - Identity element folding (x + 0 → x, x * 1 → x)
//! - Zero propagation (x * 0 → 0, x & 0 → 0)
//!
//! These patterns are separated from rangeify patterns because they apply
//! universally to any UOp graph, not just during schedule transformation.

use svod_dtype::{DType, ScalarDType};
use svod_ir::ops;
use svod_ir::types::{BinaryOp, ConstValue, ConstValueHash};
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::comparison_analysis::ComparisonAnalyzer;
use svod_ir::uop::eval::{
    eval_add_typed, eval_binary_op, eval_binary_op_broadcast, eval_binary_op_broadcast_typed, eval_mul_typed,
    eval_sub_typed, eval_unary_op_vec_typed,
};
use svod_ir::uop::properties::{HasWeakFloatProperty, SoundVminVmaxProperty, VminVmaxProperty};
use svod_ir::{IntoUOp, Op, UOp};

use crate::TypedPatternMatcher;
use crate::rangeify::indexing::get_const_value;
use crate::symbolic::dce::is_empty_range;

use smallvec::SmallVec;
use std::cmp::Ordering;
use std::sync::Arc;
use tracing::trace;

/// Weak integers select an exact i32/i64 representation, so inspecting their
/// mathematical values before lowering is safe and required by index rewrites.
/// Weak floats select the default float and can change value at commitment.
///
/// Memoised per node via [`HasWeakFloatProperty`]: the guard runs on every
/// pattern-match attempt, so a per-attempt graph walk would be quadratic.
pub(crate) fn weak_float_values_are_committed(root: &Arc<UOp>) -> bool {
    !*HasWeakFloatProperty::get(root)
}

fn value_sensitive(patterns: &TypedPatternMatcher) -> TypedPatternMatcher {
    patterns.guarded(weak_float_values_are_committed)
}

fn integer_dtype_bounds(dtype: &DType) -> Option<(i128, i128)> {
    Some(match dtype.base() {
        ScalarDType::Int8 => (i8::MIN as i128, i8::MAX as i128),
        ScalarDType::Int16 => (i16::MIN as i128, i16::MAX as i128),
        ScalarDType::Int32 => (i32::MIN as i128, i32::MAX as i128),
        ScalarDType::Int64 | ScalarDType::WeakInt | ScalarDType::Index => (i64::MIN as i128, i64::MAX as i128),
        ScalarDType::UInt8 => (0, u8::MAX as i128),
        ScalarDType::UInt16 => (0, u16::MAX as i128),
        ScalarDType::UInt32 => (0, u32::MAX as i128),
        ScalarDType::UInt64 => (0, u64::MAX as i128),
        _ => return None,
    })
}

fn integer_value(value: ConstValue) -> Option<i128> {
    match value {
        ConstValue::Int(value) => Some(value as i128),
        ConstValue::UInt(value) => Some(value as i128),
        _ => None,
    }
}

fn sound_integer_range(value: &Arc<UOp>) -> Option<(i128, i128)> {
    let dtype_bounds = integer_dtype_bounds(&value.dtype())?;
    let (min, max) = (*SoundVminVmaxProperty::get(value))?;
    let (min, max) = (integer_value(min)?, integer_value(max)?);
    (dtype_bounds.0 <= min && min <= max && max <= dtype_bounds.1).then_some((min, max))
}

fn integer_binary_does_not_wrap(op: BinaryOp, lhs: &Arc<UOp>, rhs: &Arc<UOp>, result_dtype: &DType) -> bool {
    if lhs.dtype() != *result_dtype || rhs.dtype() != *result_dtype {
        return false;
    }
    let Some((dtype_min, dtype_max)) = integer_dtype_bounds(result_dtype) else { return false };
    let Some((lhs_min, lhs_max)) = sound_integer_range(lhs) else { return false };
    let Some((rhs_min, rhs_max)) = sound_integer_range(rhs) else { return false };
    let in_dtype = |value: i128| dtype_min <= value && value <= dtype_max;

    match op {
        BinaryOp::Add => lhs_min
            .checked_add(rhs_min)
            .zip(lhs_max.checked_add(rhs_max))
            .is_some_and(|(min, max)| in_dtype(min) && in_dtype(max)),
        BinaryOp::Sub => lhs_min
            .checked_sub(rhs_max)
            .zip(lhs_max.checked_sub(rhs_min))
            .is_some_and(|(min, max)| in_dtype(min) && in_dtype(max)),
        BinaryOp::Mul => [
            lhs_min.checked_mul(rhs_min),
            lhs_min.checked_mul(rhs_max),
            lhs_max.checked_mul(rhs_min),
            lhs_max.checked_mul(rhs_max),
        ]
        .into_iter()
        .all(|value| value.is_some_and(in_dtype)),
        BinaryOp::FloorDiv | BinaryOp::FloorMod => {
            if rhs_min <= 0 && rhs_max >= 0 {
                return false;
            }
            [lhs_min, lhs_max].into_iter().all(|lhs| {
                [rhs_min, rhs_max].into_iter().all(|rhs| {
                    let value = if op == BinaryOp::FloorDiv {
                        lhs.checked_div_euclid(rhs)
                    } else {
                        lhs.checked_rem_euclid(rhs)
                    };
                    value.is_some_and(in_dtype)
                })
            })
        }
        _ => false,
    }
}

/// Per-call memo of `integer_arithmetic_does_not_wrap`: index expressions are
/// DAGs with heavy sharing, and the proof recurses into both operands.
type WrapMemo = rustc_hash::FxHashMap<u64, bool>;

fn integer_arithmetic_does_not_wrap(memo: &mut WrapMemo, value: &Arc<UOp>) -> bool {
    if let Some(&known) = memo.get(&value.id) {
        return known;
    }
    let result = match value.op() {
        Op::Binary(
            op @ (BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::FloorDiv | BinaryOp::FloorMod),
            lhs,
            rhs,
        ) => {
            integer_arithmetic_does_not_wrap(memo, lhs)
                && integer_arithmetic_does_not_wrap(memo, rhs)
                && integer_binary_does_not_wrap(*op, lhs, rhs, &value.dtype())
        }
        Op::Unary(svod_ir::UnaryOp::Neg, src) => integer_dtype_bounds(&value.dtype())
            .zip(sound_integer_range(src))
            .is_some_and(|((dtype_min, dtype_max), (src_min, src_max))| {
                integer_arithmetic_does_not_wrap(memo, src)
                    && src_max
                        .checked_neg()
                        .zip(src_min.checked_neg())
                        .is_some_and(|(min, max)| dtype_min <= min && min <= max && max <= dtype_max)
            }),
        _ => true,
    };
    memo.insert(value.id, result);
    result
}

fn typed_integer_rewrite_is_exact(original: &Arc<UOp>, replacement: &Arc<UOp>) -> bool {
    let same_shape = match (original.shape(), replacement.shape()) {
        (Ok(original), Ok(replacement)) => original == replacement,
        _ => false,
    };
    let mut memo = WrapMemo::default();
    original.dtype() == replacement.dtype()
        && same_shape
        && integer_dtype_bounds(&original.dtype()).is_some()
        && integer_arithmetic_does_not_wrap(&mut memo, original)
        && integer_arithmetic_does_not_wrap(&mut memo, replacement)
}

/// Python's `divmod`: the `(q, r)` pair with `c == q*d + r` and `r` carrying the
/// sign of `d`. This is the semantics of `Op::Binary(FloorDiv | FloorMod)`
/// (`ir/uop/eval.rs`) and of tinygrad's `//`/`%`, so the divmod normalisation
/// rules split a constant exactly the way upstream does.
fn floor_divmod(c: i64, d: i64) -> Option<(i64, i64)> {
    let (quotient, remainder) = (c.checked_div(d)?, c.checked_rem(d)?);
    if remainder != 0 && (remainder < 0) != (d < 0) {
        Some((quotient.checked_sub(1)?, remainder + d))
    } else {
        Some((quotient, remainder))
    }
}

fn exact_integer_rewrite(original: &Arc<UOp>, replacement: Arc<UOp>) -> Option<Arc<UOp>> {
    typed_integer_rewrite_is_exact(original, &replacement).then_some(replacement)
}

macro_rules! value_sensitive_matchers {
    ($($visibility:vis fn $name:ident => $unchecked:ident;)+) => {$
        (
            $visibility fn $name() -> &'static TypedPatternMatcher {
                static CACHED: std::sync::LazyLock<TypedPatternMatcher> =
                    std::sync::LazyLock::new(|| value_sensitive($unchecked()));
                &CACHED
            }
        )+
    };
}

value_sensitive_matchers! {
    pub fn constant_folding_dsl_patterns => constant_folding_dsl_patterns_unchecked;
    pub fn vconst_folding_patterns => vconst_folding_patterns_unchecked;
    pub fn identity_and_zero_patterns => identity_and_zero_patterns_unchecked;
    pub fn division_dsl_patterns => division_dsl_patterns_unchecked;
    pub fn term_combining_dsl_patterns => term_combining_dsl_patterns_unchecked;
    pub fn alu_folding_dsl_patterns => alu_folding_dsl_patterns_unchecked;
    pub fn vmin_vmax_collapse_patterns => vmin_vmax_collapse_patterns_unchecked;
    pub fn dce_dsl_simple_patterns => dce_dsl_simple_patterns_unchecked;
    pub fn comparison_dsl_patterns => comparison_dsl_patterns_unchecked;
    pub fn minmax_dsl_patterns => minmax_dsl_patterns_unchecked;
    pub fn where_bound_patterns => where_bound_patterns_unchecked;
    pub fn power_dsl_patterns => power_dsl_patterns_unchecked;
}

/// Materialise a folded constant at `dtype`.
///
/// tinygrad's `fold_const_alu` (uop/symbolic.py:31-33) evaluates with
/// `exec_alu(a.op, a.dtype, vals, False)` — `truncate=False` — and returns
/// `a.const_like(...)`, so it folds every dtype including the weak ones. Weak dtypes
/// have no storage width to wrap at, so they take the untruncated value directly;
/// strong dtypes still commit through their scalar format.
fn folded_const(dtype: DType, value: ConstValue) -> Option<Arc<UOp>> {
    let value = if dtype.is_weak() { value } else { value.cast(&DType::Scalar(dtype.base()))? };
    Some(UOp::const_(dtype, value))
}

/// Constant folding patterns.
///
/// Folds constant expressions at compile time for unary, binary, and ternary operations.
/// Uses dtype-aware evaluation to ensure results respect type boundaries (e.g., Int32 wraps at 32 bits).
fn constant_folding_dsl_patterns_unchecked() -> &'static TypedPatternMatcher {
    use svod_ir::uop::eval::{eval_ternary_op, eval_unary_op};

    crate::cached_patterns! {
        // Unary constant folding - 6 operations in one declaration
        // Neg is not here: neg() produces MUL(x, -1), folded by binary constant folding.
        for op in unary [Sqrt, Exp2, Log2, Sin, Reciprocal, Trunc] {
            op(c @const(c_val))
              => eval_unary_op(op, c_val).and_then(|r| folded_const(c.dtype(), r)),
        },

        // Binary constant folding - 13 operations in one declaration
        for op in binary [Add, Mul, Sub, FloorMod, Max, Pow, FloorDiv, Fdiv, And, Or, Xor, Shl, Shr] {
            op(a @const(a_val), _b @const(b_val))
              => eval_binary_op(op, a_val, b_val).and_then(|r| folded_const(a.dtype(), r)),
        },

        for op in binary [Lt, Le, Eq, Ne, Gt, Ge] {
            op(a @const(a_val), _b @const(b_val))
              => {
                  let dtype = DType::Bool.vec(a.dtype().vcount())?;
                  eval_binary_op(op, a_val, b_val).map(|r| UOp::const_(dtype, r))
              },
        },

        // Ternary constant folding - 2 operations in one declaration
        Where(_a @const(a_val), b @const(b_val), _c @const(c_val))
          => eval_ternary_op(svod_ir::TernaryOp::Where, a_val, b_val, c_val)
              .and_then(|r| folded_const(b.dtype(), r)),

        MulAcc(_a @const(a_val), b @const(b_val), _c @const(c_val))
          => eval_ternary_op(svod_ir::TernaryOp::MulAcc, a_val, b_val, c_val)
              .and_then(|r| folded_const(b.dtype(), r)),
    }
}

/// VConst constant folding patterns.
///
/// Folds VConst expressions at compile time:
/// - Binary operations on VConst pairs: VConst op VConst → VConst
/// - Binary operations mixing Const and VConst (with broadcast)
/// - Unary operations on VConst
///
/// Based on exec_alu for VCONST handling.
fn vconst_folding_patterns_unchecked() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Weak lanes must first commit at their consumer's concrete dtype.
        // Binary VConst folding: VConst op VConst → VConst
        for op in binary [Add, Mul, Sub, FloorMod, Max, FloorDiv, And, Or, Xor, Shl, Shr] {
            op(a @vconst(vals_a), b @vconst(vals_b))
              if !a.dtype().is_weak() && !b.dtype().is_weak()
              => {
                  let dt = a.dtype().scalar_dtype();
                  eval_binary_op_broadcast_typed(op, &vals_a, &vals_b, a.dtype().base())
                      .map(|v| UOp::vconst(v, dt))
              },
        },

        // Comparison VConst folding: VConst cmp VConst → VConst(Bool)
        for op in binary [Lt, Le, Eq, Ne, Gt, Ge] {
            op(a @vconst(vals_a), b @vconst(vals_b))
              if !a.dtype().is_weak() && !b.dtype().is_weak()
              => {
                  eval_binary_op_broadcast(op, &vals_a, &vals_b)
                      .map(|v| UOp::vconst(v, DType::Bool))
              },
        },

        // Mixed Const + VConst folding (broadcast): Const op VConst → VConst
        for op in binary [Add, Mul, Sub, FloorMod, Max, FloorDiv, And, Or, Xor, Shl, Shr] {
            op(a @anyconst(vals_a), b @anyconst(vals_b))
              if vals_a.len() != vals_b.len() && !a.dtype().is_weak() && !b.dtype().is_weak()
              => {
                  let dt = a.dtype().scalar_dtype();
                  eval_binary_op_broadcast_typed(op, &vals_a, &vals_b, a.dtype().base()).map(|v| UOp::vconst(v, dt))
              },
        },

        // Comparison mixed Const + VConst folding (broadcast)
        for op in binary [Lt, Le, Eq, Ne, Gt, Ge] {
            op(a @anyconst(vals_a), b @anyconst(vals_b))
              if vals_a.len() != vals_b.len() && !a.dtype().is_weak() && !b.dtype().is_weak()
              => eval_binary_op_broadcast(op, &vals_a, &vals_b).map(|v| UOp::vconst(v, DType::Bool)),
        },

        // Unary VConst folding
        for op in unary [Sqrt, Exp2, Log2, Sin, Reciprocal, Trunc] {
            op(a @vconst(vals))
              if !a.dtype().is_weak()
              => {
                  let dt = a.dtype().scalar_dtype();
                  eval_unary_op_vec_typed(op, &vals, a.dtype().base()).map(|v| UOp::vconst(v, dt))
              },
        },
    }
}

/// Bool arithmetic patterns.
pub fn bool_arithmetic_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Bool * Bool → AND
        Mul[x, y] if x.dtype() == DType::Bool && y.dtype() == DType::Bool => x.and_(y),
        // Bool + Bool → OR
        Add[x, y] if x.dtype() == DType::Bool && y.dtype() == DType::Bool => x.or_(y),
        // Bool max Bool → OR
        Max(x, y) if x.dtype() == DType::Bool && y.dtype() == DType::Bool => x.or_(y),
    }
}

/// Identity and zero propagation patterns.
///
/// - Identity folding: x + 0 → x, 0 + x → x, x * 1 → x, 1 * x → x, etc.
/// - Zero propagation: x * 0 → 0 (non-float only), x & 0 → 0
///
/// NOTE: For floats, x * 0 is NOT simplified because IEEE 754 requires:
/// - NaN * 0 = NaN
/// - Inf * 0 = NaN
fn identity_and_zero_patterns_unchecked() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // ========== Identity folding (commutative) ==========
        Add[x, zero @ @zero]
            if !x.dtype().is_float()
                || matches!(zero.op(), Op::Const(ConstValueHash(ConstValue::Float(v))) if v.is_sign_negative())
            => x.clone(),
        Mul[x, @one] => x.clone(),
        Or[x, @zero] => x.clone(),
        Xor[x, @zero] => x.clone(),

        // ========== Identity folding (non-commutative) ==========
        Sub(x, zero @ @zero)
            if !x.dtype().is_float()
                || matches!(zero.op(), Op::Const(ConstValueHash(ConstValue::Float(v))) if v.is_sign_positive())
            => x.clone(),
        FloorDiv(x, @one) => x.clone(),
        Fdiv(x, @one) => x.clone(),
        // x % 1 → 0 (anything mod 1 is 0)
        FloorMod(x, @one) => x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::zero(dt))),

        // ========== Rounding identity for integer types ==========
        // Floor/Ceil/Trunc/Round on integers is identity — rounding is a no-op.
        for op in unary [Floor, Ceil, Trunc, Round] {
            op(x) if !x.dtype().is_float() => x.clone()
        },

        // ========== Zero propagation ==========
        // x * 0 → 0
        // For float consts that are NaN/Inf: fold to NaN (IEEE 754: nan*0=nan, inf*0=nan).
        // NOTE: can be wrong for loaded NaN (same caveat as upstream).
        Mul[x, _zero @ @zero] if !x.dtype().is_float() =>
            x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::zero(dt))),
        And[_, zero @ @zero] => zero.clone(),
    }
}

/// Invalid propagation patterns.
///
/// Push arithmetic through WHERE-encoded gates to preserve validity tracking:
/// - CAST(WHERE(cond, x, Invalid)) → WHERE(cond, CAST(x), Invalid)
/// - ALU(WHERE(cond, x, Invalid), y) → WHERE(cond, ALU(x, y), Invalid)
/// - ALU(y, WHERE(cond, x, Invalid)) → WHERE(cond, ALU(y, x), Invalid)
/// - ALU(Invalid, y) → Invalid (non-comparison binary ops, left position only)
/// - REDUCE(op, WHERE(cond, x, Invalid)) → WHERE(cond, REDUCE(op, x), Invalid),
///   for a `cond` no reduce range can move
///
/// Upstream only propagates bare Invalid from the left position. Right-position
/// bare Invalid is not propagated.
///
/// MUST be first in `symbolic_simple()` — before `x*0→0` which would eat
/// `MUL(0, WHERE(cond, x, Invalid))` → `0`, losing validity tracking.
pub fn propagate_invalid() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Invalid control flow poisons the selected value.
        Where(invalid, _a, _b) if UOp::is_invalid_marker(invalid)
            => UOp::invalid_marker(),

        // A condition that is valid only under `cond` lifts that validity out.
        Where(Where(cond, x, invalid), a, b) if UOp::is_invalid_marker(invalid) => {
            let inner = UOp::try_where(x.clone(), a.clone(), b.clone()).ok()?;
            let marker = UOp::invalid_marker();
            UOp::try_where(cond.clone(), inner, marker).ok()
        },

        // Canonicalize: WHERE(cond, INVALID, x) → WHERE(NOT(cond), x, INVALID)
        // INVALID must be in the false branch for downstream patterns to match.
        //
        // This form arises indirectly: when an inner WHERE(valid, rng, INVALID) collapses
        // to bare INVALID (condition proven always-false by range analysis), the graph rewrite
        // engine rebuilds the parent WHERE via with_sources, placing bare INVALID in the true branch.
        // Upstream avoids this because their pattern ordering resolves it during reconstruction;
        // Svod needs explicit canonicalization.
        Where(cond, inv, x) if UOp::is_invalid_marker(inv) => {
            // Both branches INVALID: the gate is irrelevant, so collapse instead of
            // flipping. Without this the canonicalization ping-pongs forever —
            // WHERE(c, INV, INV) → WHERE(NOT c, INV, INV) → WHERE(c, INV, INV) —
            // because the inline NOT simplification below undoes the previous flip.
            // Tinygrad relies on `where(_, val, val) → val` (symbolic.py) firing first.
            if UOp::is_invalid_marker(x) {
                return Some(Arc::clone(x));
            }
            let invalid = inv.clone();
            // Inline NOT simplification: if cond is already NOT(c), flipping gives c (not NOT(NOT(c))).
            // Without this, repeated canonicalization creates NOT(NOT(NOT(...))) chains because
            // the rewrite engine doesn't process children between pattern applications on the same node.
            let flipped = match cond.op() {
                Op::Unary(svod_ir::UnaryOp::Not, inner) => Arc::clone(inner),
                _ => cond.not(),
            };
            UOp::try_where(flipped, x.clone(), invalid).ok()
        },

        // Merge nested WHERE: a.where(b.where(c, d), d) → (a & b).where(c, d)
        // .
        Where(c1, Where(c2, x, d), d) => {
            let combined = c1.and_(c2);
            UOp::try_where(combined, x.clone(), d.clone()).expect("failed to create WHERE")
        },

        // Preserve the live outer branch while lifting nested validity.
        Where(a, Where(cond, x, invalid), c)
            if UOp::is_invalid_marker(invalid) && !UOp::is_invalid_marker(c)
            => {
                let inner = UOp::try_where(a.clone(), x.clone(), c.clone()).ok()?;
                let combined = a.not().or_(cond);
                let marker = UOp::invalid_marker();
                UOp::try_where(combined, inner, marker).ok()
            },
        Where(a, b, Where(cond, x, invalid))
            if UOp::is_invalid_marker(invalid) && !UOp::is_invalid_marker(b)
            => {
                let inner = UOp::try_where(a.clone(), b.clone(), x.clone()).ok()?;
                let combined = a.or_(cond);
                let marker = UOp::invalid_marker();
                UOp::try_where(combined, inner, marker).ok()
            },

        // Unary/cast operations preserve Invalid and move inside its gate.
        for op in unary [*] {
            op(invalid) if UOp::is_invalid_marker(invalid)
                => UOp::invalid_marker(),
            r @ op(Where(cond, x, invalid)) if UOp::is_invalid_marker(invalid) => {
                let inner = UOp::new(Op::Unary(op, x.clone()), r.dtype());
                let marker = UOp::invalid_marker();
                UOp::try_where(cond.clone(), inner, marker).ok()
            },
        },

        Cast { src: invalid, .. } if UOp::is_invalid_marker(invalid)
            => UOp::invalid_marker(),
        Cast { src: Where(cond, x, invalid), dtype } if UOp::is_invalid_marker(invalid) => {
            let inner = x.cast(dtype.clone());
            let marker = UOp::invalid_marker();
            UOp::try_where(cond.clone(), inner, marker).ok()
        },
        BitCast { src: invalid, .. } if UOp::is_invalid_marker(invalid)
            => UOp::invalid_marker(),
        BitCast { src: Where(cond, x, invalid), dtype } if UOp::is_invalid_marker(invalid) => {
            let inner = x.bitcast(dtype.clone());
            let marker = UOp::invalid_marker();
            UOp::try_where(cond.clone(), inner, marker).ok()
        },

        // A gate no reduce range can move is the same for every contribution, so
        // the whole reduction is valid or invalid together and the gate lifts out.
        // Without this the gate stays wedged between the REDUCE and its MUL, where
        // `tc::matmul_operands` — which sees through casts and nothing else —
        // stops recognising a matmul and every tensor-core opt is declined.
        // `num_axes > 0` also reduces leading shaped axes, which carry no RANGE to
        // test `cond` against, so those are left alone.
        //
        // It fires far more often than it pays: a concat-heavy graph triggers it
        // dozens of times and mostly reaches the same kernels anyway. Where it
        // does change the kernel it only reorders the loop nest — the extents are
        // the same multiset — so a graph with no matmul to unblock gets the
        // reorder and nothing for it, and inception-style branches measurably
        // lose a couple of percent. The payoff is specific to the tensor-core
        // matcher while this rule is general symbolic rewriting; see 49d264b2 for
        // the measurements on both sides.
        Reduce { src: Where(cond, x, invalid), ranges, reduce_op, num_axes }
            if UOp::is_invalid_marker(invalid) && *num_axes == 0 && !moved_by_ranges(cond, ranges)
            => {
                let reduced = assuming(x, cond).reduce_with_num_axes(ranges.clone(), *reduce_op, *num_axes);
                UOp::try_where(cond.clone(), reduced, UOp::invalid_marker()).ok()
            },

        // Push binary ALU through WHERE-with-Invalid (left operand)
        // ALU(WHERE(cond, x, Invalid), y) → WHERE(cond, ALU(x, y), Invalid)
        for op in binary [*] {
            op(Where(cond, x, invalid), y)
                if UOp::is_invalid_marker(invalid)
                => {
                    let inner_op = Op::Binary(op, x.clone(), y.clone());
                    let inner = UOp::new(inner_op.clone(), svod_ir::dtype_from_op(&inner_op).expect("binary dtype inference"));
                    let marker = UOp::invalid_marker();
                    UOp::try_where(cond.clone(), inner, marker).expect("failed to create WHERE")
                },
        },

        // Push binary ALU through WHERE-with-Invalid (right operand)
        // ALU(y, WHERE(cond, x, Invalid)) → WHERE(cond, ALU(y, x), Invalid)
        for op in binary [*] {
            op(y, Where(cond, x, invalid))
                if UOp::is_invalid_marker(invalid)
                => {
                    let inner_op = Op::Binary(op, y.clone(), x.clone());
                    let inner = UOp::new(inner_op.clone(), svod_ir::dtype_from_op(&inner_op).expect("binary dtype inference"));
                    let marker = UOp::invalid_marker();
                    UOp::try_where(cond.clone(), inner, marker).expect("failed to create WHERE")
                },
        },

        // ALU with bare Invalid → Invalid, in either operand position
        // (tinygrad `uop/symbolic.py:77` matches `src=[invalid_pat, UPat()]`
        // order-insensitively). Comparisons are excluded there and here: they
        // keep the gate via the two rules above (`symbolic.py:75-76`), which
        // cover every binary op including `GroupOp.Comparison`.
        //
        // Tinygrad's Invalid is the bottom of the promotion lattice. Svod uses
        // a typed marker, so create one with the operation's result dtype.
        for op in binary [Add, Mul, Sub, FloorMod, Max, FloorDiv, Fdiv, Pow, And, Or, Xor, Shl, Shr] {
            op(invalid, _y) if UOp::is_invalid_marker(invalid)
                => UOp::invalid_marker(),
            op(_y, invalid) if UOp::is_invalid_marker(invalid)
                => UOp::invalid_marker(),
        },
    }
}

/// Final-only cleanup of Invalid validity markers.
///
/// Invalid must survive optimization and gate lowering, but cannot reach a
/// rendered program. Typed data Invalid becomes typed zero; Index Invalid is
/// retained for late memory gate lowering.
pub fn pm_remove_invalid() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        r @ Where(cond, x, invalid) if UOp::is_invalid_marker(invalid) && r.dtype().base() != ScalarDType::Index =>
            UOp::try_where(cond.clone(), x.clone(), r.const_like(0)).ok(),
        r @ Stack { sources } if sources.iter().any(UOp::is_invalid_marker) && r.dtype().base() != ScalarDType::Index => {
            let zero = UOp::const_(r.dtype().scalar_dtype(), ConstValue::Int(0));
            Some(UOp::stack(sources.iter().map(|x| if UOp::is_invalid_marker(x) { zero.clone() } else { x.clone() }).collect()))
        },
    }
}

fn const_like_shape(u: &Arc<UOp>, value: ConstValue) -> Arc<UOp> {
    fn fill(value: &Arc<UOp>, shape: &[svod_ir::SInt]) -> Option<Arc<UOp>> {
        let (&svod_ir::SInt::Const(count), tail) = shape.split_first()? else { return None };
        let lane = if tail.is_empty() { value.clone() } else { fill(value, tail)? };
        Some(UOp::stack((0..count).map(|_| lane.clone()).collect()))
    }

    let scalar = UOp::const_(u.dtype().scalar_dtype(), value);
    let Ok(Some(shape)) = u.shape() else { return scalar };
    if shape.is_empty() { scalar } else { fill(&scalar, shape).unwrap_or(scalar) }
}

/// Fold LOAD/STORE with fully-Invalid INDEX.
///
/// When an INDEX has an Invalid marker as its index, the entire access is out-of-bounds:
/// - LOAD(INDEX(buf, Invalid)) → const 0 (invalid load produces zero)
/// - STORE(INDEX(buf, Invalid), value) → NOOP (invalid store does nothing)
///
/// Also handles CAST-wrapped variants:
/// - LOAD(CAST(INDEX(buf, Invalid))) → const 0
/// - STORE(CAST(INDEX(buf, Invalid)), value) → NOOP
///
/// This occurs when padding creates regions entirely outside the original tensor bounds,
/// causing WHERE(valid, idx, Invalid) to simplify to just Invalid when valid is always false.
pub fn fold_invalid_load_store() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // LOAD(INDEX(buf, Invalid, ...)) → const 0
        load @ Load { index: Index { indices, .. }, alt, .. }
            if indices.first().is_some_and(UOp::is_invalid_marker)
            => {
                if let Some(alt) = alt { return Some(alt.clone()); }
                 Some(const_like_shape(load, ConstValue::zero(load.dtype().base())))
            },

        // LOAD(CAST(INDEX(buf, Invalid, ...))) → const 0
        load @ Load { index: Cast { src: Index { indices, .. }, .. }, alt, .. }
            if indices.first().is_some_and(UOp::is_invalid_marker)
            => {
                if let Some(alt) = alt { return Some(alt.clone()); }
                 Some(const_like_shape(load, ConstValue::zero(load.dtype().base())))
            },

        // STORE(INDEX(buf, Invalid, ...), value) → NOOP
        Store { index: Index { indices, .. }, value: _, gate: None }
            if indices.first().is_some_and(UOp::is_invalid_marker)
            => UOp::new(Op::Noop, DType::Void),

        // STORE(CAST(INDEX(buf, Invalid, ...)), value) → NOOP
        Store { index: Cast { src: Index { indices, .. }, .. }, value: _, gate: None }
            if indices.first().is_some_and(UOp::is_invalid_marker)
            => UOp::new(Op::Noop, DType::Void),
    }
}

/// Tier-1 algebraic identities + constant folding WITHOUT the trivial-loop
/// collapse ([`dead_loop_patterns`]). The base for [`symbolic_simple`], which
/// re-adds the collapse.
///
/// Contains algebraic identities and zero propagation rules:
/// - x + 0 → x, 0 + x → x
/// - x - 0 → x
/// - x * 1 → x, 1 * x → x
/// - x / 1 → x (both FloorDiv and Fdiv)
/// - x | 0 → x, 0 | x → x
/// - x ^ 0 → x, 0 ^ x → x
/// - x * 0 → 0, 0 * x → 0
/// - x & 0 → 0, 0 & x → 0
fn symbolic_simple_base() -> TypedPatternMatcher {
    propagate_invalid()
        + fold_invalid_load_store()
        + constant_folding_dsl_patterns()
        + vconst_folding_patterns() // CONST and VCONST folded together at this tier
        + bool_arithmetic_patterns()
        + identity_and_zero_patterns()
        + self_folding_dsl_patterns()
        + zero_folding_dsl_patterns()
        + division_dsl_patterns()
        + cast_dsl_patterns()
        + uint_pack_dsl_patterns()
        + div_mod_recombine_dsl_patterns()
        + power_dsl_patterns()
        + boolean_dsl_simple_patterns()
        + dce_dsl_simple_patterns()
}

/// Collapse `CAST(dtype, CONST(value))` into a constant shaped and typed like
/// the cast.
///
/// Tinygrad `uop/symbolic.py:101` keeps this standalone rather than inside
/// `symbolic_simple`, and composes it explicitly at every site that wants it
/// (`codegen/__init__.py:304,349,360`, `codegen/simplify.py:35`, `uop/ops.py:533`,
/// `schedule/rangeify.py:587`). Morok mirrors that composition one for one, so
/// devectorize and the image/load rewrites deliberately do not fold cast
/// constants.
pub fn pm_fold_cast_const() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        root @ Cast { src: _c @const(c_val), dtype: _ } => root.const_like(c_val),
    }
}

/// Tier-1 algebraic identities + constant folding + the size-1 RANGE collapse.
/// Used at lightweight stages: decompositions, `pm_simplify_valid` helpers.
pub fn symbolic_simple() -> &'static TypedPatternMatcher {
    static CACHED: std::sync::LazyLock<TypedPatternMatcher> =
        std::sync::LazyLock::new(|| symbolic_simple_base() + dead_loop_patterns());
    &CACHED
}

/// Add the Tier-2 patterns on top of a Tier-1 base, in the fixed order the
/// rewrite engine applies them. The order is load-bearing: each group may expose
/// matches for a later one (e.g. commutative canonicalization before term
/// combining, ALU folding before the comparison/range rules).
fn with_tier2(tier1: TypedPatternMatcher) -> TypedPatternMatcher {
    let head = tier1
        + commutative_canonicalization()
        + boolean_dsl_patterns() // x | !x → true
        + term_combining_dsl_patterns() // combine like terms and weak-int distribution
        + dce_dsl_patterns() // WHERE(!cond) branch swap
        + where_alu_combining_patterns() // hoist ALU through WHERE
        + vmin_vmax_collapse_patterns() // vmin == vmax → const
        + minmax_dsl_patterns() // bound-based max/min selection
        + alu_folding_dsl_patterns(); // two-stage ALU, const push-down
    head + comparison_dsl_patterns() // lt/le/eq simplification
        + range_based_mod_div_patterns() // mod/div against a range bound
        + advanced_division_dsl_patterns() // symbolic div-and-mod
        + range_based_cast_patterns() // range-based double-cast
        + long_to_int_narrowing_patterns() // i64 → i32 when range fits
        + after_simplification_patterns() // drop redundant AFTER ordering
        + where_bound_patterns() // WHERE(Lt) elimination via vmin/vmax
}

pub fn symbolic() -> &'static TypedPatternMatcher {
    static CACHED: std::sync::LazyLock<TypedPatternMatcher> =
        std::sync::LazyLock::new(|| with_tier2(symbolic_simple_base() + dead_loop_patterns()));
    &CACHED
}

/// Maximum symbolic matcher (tier 3).
///
/// Matches upstream `sym` (`uop/symbolic.py:429`):
/// symbolic + pm_simplify_valid + store/load fold + cast-through-WHERE +
/// ALU/STACK reorder + x!=0 fold + opinionated combine terms + reduce hoist.
///
/// Upstream's reciprocal distribution (`uop/symbolic.py:448-453`) is deliberately
/// absent: all six rules are IEEE-inexact, and
/// `unknown_float_division_power_and_reciprocal_are_not_algebraically_rewritten`
/// pins the non-rewrite.
///
/// Used at: pre-opt initial, post-opt (Stage 8), expander (Stage 9), devectorize (Stage 14).
pub fn sym() -> &'static TypedPatternMatcher {
    static CACHED: std::sync::LazyLock<TypedPatternMatcher> = std::sync::LazyLock::new(|| {
        symbolic()
            + super::valid_simplification::pm_simplify_valid()
            + alu_vectorize_reorder_patterns()
            + value_sensitive(ne_zero_fold_patterns())
            + cast_where_dsl_patterns()
            + store_load_folding_patterns()
            + value_sensitive(reduce_sym_patterns())
            + sym_phase3_patterns()
    });
    &CACHED
}

/// Pinned Tinygrad's structural commutative ordering for weak-integer index expressions.
/// Source order remains authored for every other dtype.
pub(crate) fn commutative_canonicalization() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        for op in binary [Add, Mul, Max, Eq, Ne, And, Or, Xor] {
            r @ op(a, b)
                if crate::linearize::tinygrad_weakint_expr(r)
                    && crate::linearize::tinygrad_tuplize_cmp(b, a) == Some(Ordering::Less)
                => r.replace().src(vec![b.clone(), a.clone()]).call(),
        },
    }
}

/// Self-folding patterns.
///
/// Patterns where an operand appears twice:
/// - x // x → 1
/// - x // -1 → -x
/// - (x % y) % y → x % y
/// - x & x → x, x | x → x, max(x,x) → x (GroupOp.Idempotent)
pub fn self_folding_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // x // x → 1
        original @ FloorDiv(x, x) => exact_integer_rewrite(original, 1.into_uop(x.dtype())),
        // x // -1 → -x
        original @ FloorDiv(x, _c @const(c_val)) if c_val.is_neg_one() => exact_integer_rewrite(original, x.neg()),
        // (x % y) % y → x % y
        original @ FloorMod(FloorMod(x, y), y) => exact_integer_rewrite(original, x.mod_(y)),
        // Idempotent: x op x → x (GroupOp.Idempotent = {AND, OR, MAX})
        And(x, x) => x.clone(),
        Max(x, x) => x.clone(),
        // x | x → x
        Or(x, x) => x.clone(),
    }
}

/// Zero folding patterns.
///
/// Patterns that fold to zero or false:
/// - x < x → False when NaN is impossible
/// - x % x → 0
/// - x != x → False (, ints+bool+index only)
pub fn zero_folding_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // x % x → 0
        original @ FloorMod(x, x) => {
            let replacement = x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::zero(dt)))?;
            exact_integer_rewrite(original, replacement)
        },
        // Float x<x is false only when the analysis proves x cannot be NaN.
        Lt(x, x) if !x.dtype().is_float() || SoundVminVmaxProperty::get(x).is_some() =>
            Some(UOp::const_(DType::Bool.vec(x.dtype().vcount()).expect("Bool is a scalar"), ConstValue::Bool(false))),
        // x != x → False (ints+bool+index, returns bool.vec(count))
        Ne(x, x) if x.dtype().is_int() || x.dtype().is_bool() =>
            Some(UOp::const_(DType::Bool.vec(x.dtype().vcount()).expect("Bool is a scalar"), ConstValue::Bool(false))),
    }
}

/// Range-based modulo and division simplification patterns.
///
/// Uses vmin/vmax analysis to simplify:
/// - x % n → x when 0 <= vmin(x) && vmax(x) < n
/// - x / n → 0 when 0 <= vmin(x) && vmax(x) < n
///
/// This is critical for RESHAPE range propagation where Range(n) % n should simplify to Range(n).
pub fn range_based_mod_div_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Range(end) % end → Range(end) (identity: range values are [0, end), always < end)
        original @ FloorMod(range @ Range { end, .. }, end) => exact_integer_rewrite(original, range.clone()),
        // Range(end) // end → 0 (all values in [0, end) divide by end to 0)
        original @ FloorDiv(Range { end, .. }, end) => exact_integer_rewrite(original, original.const_like(0)),

        // x % n → x when 0 <= vmin(x) && vmax(x) < n
        // This handles cases like Range(3) % 3 → Range(3)
        FloorMod(x, _n @const(n_val)) => {
            let (vmin, vmax) = SoundVminVmaxProperty::get(x).as_ref()?;
            trace!(
                x.id = x.id,
                vmin = ?vmin,
                vmax = ?vmax,
                n_val = ?n_val,
                "FloorMod simplification check"
            );
            // Check if x is always non-negative and less than n
            if let (ConstValue::Int(min), ConstValue::Int(max), ConstValue::Int(n_int)) = (vmin, vmax, n_val)
                && *min >= 0 && *max < n_int {
                    trace!(
                        n_int,
                        min = *min,
                        max = *max,
                        "Simplifying x % n → x"
                    );
                    return Some(Arc::clone(x));
                }
            None
        },

        // (a * m + b) % n → b % n when m == n (factor out multiples of n)
        // This handles matmul index expressions like: (row * 512 + col) % 512 → col % 512
        // Since (a * n) % n = 0, the Mul term can be dropped.
        // Using commutative [] for Add to match both orderings.
        // Note: We compare m_val and n_val by VALUE, not pointer, since they may be separate UOps.
        // Keep this legacy rule in the non-negative indexing domain.
        original @ FloorMod(Add[Mul[_a, n @const(_m_val)], b], n @const(_n_val)) => {
            if !matches!(SoundVminVmaxProperty::get(b), Some((ConstValue::Int(v), _)) if *v >= 0) { return None; }
            exact_integer_rewrite(original, b.mod_(n))
        },

        // ((a * n) + b + c) % n → (b + c) % n when (b + c) >= 0.
        original @ FloorMod(Add[Add[Mul[_a, n @const(_m_val)], b], c], n @const(_n_val)) => {
            let bc = b.add(c);
            if !matches!(SoundVminVmaxProperty::get(&bc), Some((ConstValue::Int(v), _)) if *v >= 0) { return None; }
            let replacement = bc.mod_(n);
            exact_integer_rewrite(original, replacement)
        },

        // (a * m + b) / n → a + b / n when m == n (distribute division over sum)
        // When b is non-negative and small, this can enable further simplification.
        // Specifically: (a * n + b) / n = a when 0 <= b < n
        // Using commutative [] for Add to match both orderings.
        // Note: We compare m_val and n_val by VALUE, not pointer, since they may be separate UOps.
        original @ FloorDiv(Add[Mul[a, n @const(_m_val)], b], n @const(n_val)) => {
            let n_int = n_val.try_int()?;
            if n_int <= 0 { return None; }

            let (vmin, vmax) = SoundVminVmaxProperty::get(b).as_ref()?;
            if let (ConstValue::Int(min), ConstValue::Int(max)) = (vmin, vmax)
                && *min >= 0 && *max < n_int {
                    // b is in [0, n), so (a * n + b) / n = a
                    trace!(
                        ?n_val,
                        a.id = a.id,
                        min = *min,
                        max = *max,
                        "FloorDiv factor-out: (a * n + b) / n → a (when 0 <= b < n)"
                    );
                    return exact_integer_rewrite(original, Arc::clone(a));
                }
            // Fall through in the non-negative indexing domain.
            if !matches!(SoundVminVmaxProperty::get(b), Some((ConstValue::Int(v), _)) if *v >= 0) { return None; }
            let b_div_n = b.floor_div(n);
            exact_integer_rewrite(original, a.add(&b_div_n))
        },

        // x / n → k when all values of x are in the same bucket [k*n, (k+1)*n)
        // This is the "cancel divmod" rule from upstream fold_divmod_general.
        // Examples:
        //   Range(3) / 3 → 0 (since Range(3) is 0,1,2 and all /3 = 0)
        //   (64 + Range(8)) / 64 → 1 (since 64..71 all /64 = 1)
        FloorDiv(x, _n @const(n_val)) => {
            let (vmin, vmax) = SoundVminVmaxProperty::get(x).as_ref()?;
            if let (ConstValue::Int(min), ConstValue::Int(max), ConstValue::Int(n_int)) = (vmin, vmax, n_val)
                && n_int > 0 {
                    let min_div = min.checked_div_euclid(n_int)?;
                    let max_div = max.checked_div_euclid(n_int)?;
                    if min_div == max_div {
                        trace!(
                            min = *min,
                            max = *max,
                            n_int,
                            result = min_div,
                            "FloorDiv cancel: x / n → k (all values in same bucket)"
                        );
                        return Some(UOp::const_(x.dtype(), ConstValue::Int(min_div)));
                    }
                }
            None
        },

        // (a + (x // n) * n) // n → x // n  when 0 <= vmin(a) and vmax(a) < n
        // This eliminates redundant idiv chains in address calculations
        // Using [] for both Add and Mul to match all permutations
        original @ FloorDiv(Add[a, Mul[FloorDiv(x, n @const(n_val)), n]], n) => {
            let (vmin, vmax) = SoundVminVmaxProperty::get(a).as_ref()?;
            if let (ConstValue::Int(min), ConstValue::Int(max), ConstValue::Int(n_int)) = (vmin, vmax, n_val)
                && *min >= 0 && *max < n_int && n_int > 0 {
                    return exact_integer_rewrite(original, x.floor_div(n));
                }
            None
        },

        // (x + c) // d → x // d when adding c never crosses a bucket boundary
        // Condition: for ALL v in [vmin(x), vmax(x)], (v+c)//d == v//d.
        // A value v crosses a boundary when v%d + c >= d. So the rule is safe iff
        // the maximum remainder in [min, max] satisfies max_rem + c < d.
        original @ FloorDiv(Add[x, _c @const(c_val)], d @const(d_val)) => {
            let c_int = c_val.try_int()?;
            let d_int = d_val.try_int()?;
            if d_int <= 0 || c_int <= 0 { return None; }

            let (vmin, vmax) = SoundVminVmaxProperty::get(x).as_ref()?;
            if let (ConstValue::Int(min), ConstValue::Int(max)) = (vmin, vmax)
                && *min >= 0
            {
                // Max remainder of v%d for v in [min, max]:
                // - if range spans a full cycle (max - min >= d - 1), max_rem = d - 1
                // - if min%d > max%d (modular wrap), max_rem = d - 1
                // - otherwise, max_rem = max%d
                let d_minus_one = d_int.checked_sub(1)?;
                let max_rem = if max.checked_sub(*min)? >= d_minus_one
                    || min.checked_rem(d_int)? > max.checked_rem(d_int)?
                {
                    d_minus_one
                } else {
                    max.checked_rem(d_int)?
                };

                if max_rem.checked_add(c_int).is_some_and(|sum| sum < d_int) {
                    return exact_integer_rewrite(original, x.floor_div(d));
                }
            }
            None
        },

        // (x + c) // d -> (x + c%d) // d + c//d, for any d != 0
        // (`uop/divandmod.py:102-105`): "split the multiple of d out of the
        // const, holds for any d!=0". Floor division satisfies
        // `(x + r + q*d)//d == (x + r)//d + q` for every integer `x` and every
        // `d != 0`, so upstream's only guard is `c.val%d.val==c.val`.
        original @ FloorDiv(Add[x, _c @const(c_val)], d @const(d_val)) => {
            let (c_div_d, c_mod_d) = floor_divmod(c_val.try_int()?, d_val.try_int()?)?;
            if c_mod_d == c_val.try_int()? { return None; }

            let remainder_const = UOp::const_(d.dtype(), ConstValue::Int(c_mod_d));
            let quotient_const = UOp::const_(d.dtype(), ConstValue::Int(c_div_d));
            exact_integer_rewrite(original, x.add(&remainder_const).floor_div(d).add(&quotient_const))
        },

        // Phase 1b: (x + c) // d for negative x
        // When x <= 0 but (x + c) >= 0, split using adjusted formula:
        // (x + c) // d → -(-(c%d + x - (d-1)) // d) + c//d
        original @ FloorDiv(Add[x, _c @const(c_val)], d @const(d_val)) => {
            let c_int = c_val.try_int()?;
            let d_int = d_val.try_int()?;
            if d_int <= 0 { return None; }

            let (x_vmin, x_vmax) = SoundVminVmaxProperty::get(x).as_ref()?;
            let n_expr = x.add(&UOp::const_(x.dtype(), c_val));
            let (n_vmin, _) = SoundVminVmaxProperty::get(&n_expr).as_ref()?;

            if let (ConstValue::Int(_), ConstValue::Int(xmax)) = (x_vmin, x_vmax)
                && let ConstValue::Int(nmin) = n_vmin
                && *xmax <= 0 && *nmin >= 0
            {
                let c_mod_d = c_int.rem_euclid(d_int);
                let c_div_d = c_int.div_euclid(d_int);
                // inner = -(c%d + x - (d-1))
                let c_mod_const = UOp::const_(d.dtype(), ConstValue::Int(c_mod_d));
                let d_minus_1 = UOp::const_(d.dtype(), ConstValue::Int(d_int.checked_sub(1)?));
                let inner = c_mod_const.add(x).sub(&d_minus_1).neg();
                let div_result = inner.floor_div(d).neg();
                let quotient_const = UOp::const_(d.dtype(), ConstValue::Int(c_div_d));
                return exact_integer_rewrite(original, div_result.add(&quotient_const));
            }
            None
        },

        // The broad WeakInt div/mod decomposition is intentionally disabled. Its
        // speculative reassociation cannot establish no-wrap for every generated
        // intermediate; the explicit proven rules above cover safe index cases.
    }
}

/// Division simplification patterns.
///
/// - 0 / 0 → NaN (float division by zero of zero)
/// - (x * 0) / 0 → NaN (any expression that reduces to 0/0)
/// - x / x → 1.0 (float division)
/// - (x * y) / y → x
/// - (x * y) // y → x
fn division_dsl_patterns_unchecked() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // 0 / 0 → NaN (IEEE 754: 0/0 is indeterminate)
        // NOTE: This must come before x/x → 1 pattern to take priority
        Fdiv(zero1 @ @zero, @zero) if zero1.dtype().is_float()
            => UOp::const_(zero1.dtype(), ConstValue::Float(f64::NAN)),
        // (x * 0) / 0 → NaN (anything times zero divided by zero is NaN)
        Fdiv(Mul[_, zero1 @ @zero], @zero) if zero1.dtype().is_float()
            => UOp::const_(zero1.dtype(), ConstValue::Float(f64::NAN)),
        // x / x → 1.0 only when x is provably finite and non-zero.
        Fdiv(x, x) if sound_finite_nonzero(x) =>
            x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::one(dt))),
        // (x * y) // y → x
        original @ FloorDiv(Mul(x, y), y) => exact_integer_rewrite(original, x.clone()),
    }
}

fn sound_finite_nonzero(value: &Arc<UOp>) -> bool {
    let Some((ConstValue::Float(min), ConstValue::Float(max))) = SoundVminVmaxProperty::get(value) else {
        return false;
    };
    min.is_finite() && max.is_finite() && (*max < 0.0 || *min > 0.0)
}

/// Check if casting from `from` to `to` can safely preserve all values.
///
/// Returns true if all values representable in `to` can be represented in `from`.
/// This is used for double-cast optimization: x.cast(a).cast(b) → x.cast(b)
/// is only safe if `a` can hold all values of `b` (so no truncation occurs in `a`).
fn can_safe_cast(to: &DType, from: &DType) -> bool {
    use svod_dtype::ScalarDType;

    // Get base scalar types for comparison
    let to_scalar = match to {
        DType::Scalar(s) => *s,
        DType::Vector { scalar, .. } => *scalar,
        _ => return false,
    };
    let from_scalar = match from {
        DType::Scalar(s) => *s,
        DType::Vector { scalar, .. } => *scalar,
        _ => return false,
    };

    // Same type is always safe
    if to_scalar == from_scalar {
        return true;
    }

    // Get bit widths and signedness
    let (to_bits, to_signed, to_float) = match to_scalar {
        ScalarDType::Bool => (1, false, false),
        ScalarDType::Int8 => (8, true, false),
        ScalarDType::Int16 => (16, true, false),
        ScalarDType::Int32 => (32, true, false),
        ScalarDType::Int64 => (64, true, false),
        ScalarDType::UInt8 => (8, false, false),
        ScalarDType::UInt16 => (16, false, false),
        ScalarDType::UInt32 => (32, false, false),
        ScalarDType::UInt64 => (64, false, false),
        ScalarDType::Float16 | ScalarDType::BFloat16 => (16, true, true),
        ScalarDType::Float32 => (32, true, true),
        ScalarDType::Float64 => (64, true, true),
        _ => return false,
    };
    let (from_bits, from_signed, from_float) = match from_scalar {
        ScalarDType::Bool => (1, false, false),
        ScalarDType::Int8 => (8, true, false),
        ScalarDType::Int16 => (16, true, false),
        ScalarDType::Int32 => (32, true, false),
        ScalarDType::Int64 => (64, true, false),
        ScalarDType::UInt8 => (8, false, false),
        ScalarDType::UInt16 => (16, false, false),
        ScalarDType::UInt32 => (32, false, false),
        ScalarDType::UInt64 => (64, false, false),
        ScalarDType::Float16 | ScalarDType::BFloat16 => (16, true, true),
        ScalarDType::Float32 => (32, true, true),
        ScalarDType::Float64 => (64, true, true),
        _ => return false,
    };

    // Float <-> int conversions are not safe
    if to_float != from_float {
        return false;
    }

    // For floats: larger precision can hold smaller
    if to_float {
        return from_bits >= to_bits;
    }

    // For integers:
    // - Same signedness: larger width can hold smaller
    // - Unsigned to signed: need one extra bit (e.g., u8 fits in i16)
    // - Signed to unsigned: never safe (negative values lost)
    if to_signed == from_signed {
        return from_bits >= to_bits;
    }

    if !to_signed && from_signed {
        // unsigned → signed: from needs to be at least 1 bit larger
        return from_bits > to_bits;
    }

    // signed → unsigned: never safe
    false
}

/// Cast optimization patterns.
///
/// - x.cast(dtype) → x if same dtype (noop cast)
/// - x.cast(a).cast(b) → x.cast(b) when safe (collapse double cast)
///
/// NOTE: Double cast is only safe when the intermediate type `a` can hold all
/// values of the final type `b`. Example of UNSAFE collapse:
///   int64.cast(int8).cast(int64) → int64  // WRONG: loses truncation!
pub fn cast_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // x.cast(dtype) → x if same dtype
        Cast { src: x, dtype } if x.dtype() == *dtype => x.clone(),
        // x.cast(a).cast(b) → x when x.dtype == b and a preserves all values of b
        // This handles cases like: bool.cast(int32).cast(bool) → bool
        Cast { src: Cast { src: x, dtype: intermediate }, dtype: outer }
            if x.dtype() == *outer && can_safe_cast(outer, intermediate)
            => x.clone(),
        // x.cast(a).cast(b) → x.cast(b) when a doesn't narrow x
        // This handles widening chains: int8.cast(int32).cast(int64) → int8.cast(int64)
        Cast { src: Cast { src: x, dtype: intermediate }, dtype: outer }
            if can_safe_cast(&x.dtype(), intermediate)
            => x.cast(outer.clone()),
    }
}

/// Unpack a uint64 packed from two uint32 (Tinygrad `uop/symbolic.py:170-173`).
///
/// THREEFRY packs its operands as `(hi.cast(u64) << 32) | lo.cast(u64)` and its
/// callers immediately take one half back out. Cancelling the pair keeps the
/// whole PRNG in 32-bit ALU.
pub fn uint_pack_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // ((_:u64 << 32) | y.cast(u64)).cast(u32) → y.cast(u32)
        Cast { src: Or[shifted @ Shl(_, _s @const(s)), lo], dtype: narrowed }
            if *narrowed == DType::UInt32 && shifted.dtype() == DType::UInt64 && is_shift_32(s)
            => Some(low_half_payload(lo)?.cast(DType::UInt32)),
        // ((x.cast(u64) << 32) | _.cast(u64)) >> 32 → x.cast(u64)
        Shr(Or[Shl(high, _s @const(s)), lo], _t @const(t))
            if is_shift_32(s) && is_shift_32(t)
            => {
                let payload = low_half_payload(high)?;
                low_half_payload(lo)?;
                Some(payload.cast(DType::UInt64))
            },
    }
}

/// A shift amount of exactly 32, whatever integer flavour the constant carries.
fn is_shift_32(value: ConstValue) -> bool {
    matches!(value, ConstValue::Int(32) | ConstValue::UInt(32))
}

/// The 32-bit payload behind a widening cast into uint64, when the source
/// provably holds in the low 32 bits — so the high half of the widened value is
/// zero. Tinygrad spells this as a literal `uint32 → uint64` cast; morok's
/// earlier cast folding can leave a signed-but-non-negative source instead.
fn low_half_payload(value: &Arc<UOp>) -> Option<&Arc<UOp>> {
    let Op::Cast(ops::Cast { src, dtype }) = value.op() else { return None };
    (*dtype == DType::UInt64 && (src.dtype() == DType::UInt32 || fits_in_u32(src))).then_some(src)
}

/// Provable `0 ..= u32::MAX` bounds — the range-analysis stand-in for a literal
/// uint32 source.
fn fits_in_u32(value: &Arc<UOp>) -> bool {
    fn bound(value: &ConstValue) -> Option<i128> {
        match value {
            ConstValue::Int(v) => Some(i128::from(*v)),
            ConstValue::UInt(v) => Some(i128::from(*v)),
            _ => None,
        }
    }
    let bounds = SoundVminVmaxProperty::get(value);
    let Some((low, high)) = bounds.as_ref() else { return false };
    matches!((bound(low), bound(high)), (Some(low), Some(high)) if low >= 0 && high <= i128::from(u32::MAX))
}

/// Range-based double-cast collapse.
///
/// x:ints.cast(ints, a).cast(b) → x.cast(b) when a.min <= x.vmin and x.vmax <= a.max.
/// Uses vmin/vmax analysis — belongs in symbolic tier, not symbolic_simple.
///
/// `x` must be strong: for a weak `x` the inner cast is the width the value
/// commits at (`pm_lower_index_dtype`), and collapsing it leaves `x` under the
/// outer cast alone, where `cast_weak_srcs` joins `b` with `x`'s own int width
/// — for `b = u64` that join is float, turning integer index math into `fmod`.
fn range_based_cast_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        Cast { src: Cast { src: x, dtype: intermediate }, dtype: outer }
            if x.dtype().is_int()
            && !x.dtype().is_weak()
            && intermediate.is_int()
            => {
                // Check if x's value range fits within the intermediate type
                let (vmin, vmax) = SoundVminVmaxProperty::get(x).as_ref()?;
                let (imin, imax) = match intermediate.scalar() {
                    Some(ScalarDType::Int8) => (i8::MIN as i64, i8::MAX as i64),
                    Some(ScalarDType::Int16) => (i16::MIN as i64, i16::MAX as i64),
                    Some(ScalarDType::Int32) => (i32::MIN as i64, i32::MAX as i64),
                    Some(ScalarDType::Int64) => (i64::MIN, i64::MAX),
                    Some(ScalarDType::UInt8) => (0, u8::MAX as i64),
                    Some(ScalarDType::UInt16) => (0, u16::MAX as i64),
                    Some(ScalarDType::UInt32) => (0, u32::MAX as i64),
                    _ => return None,
                };
                if let (ConstValue::Int(vmin_v), ConstValue::Int(vmax_v)) = (vmin, vmax)
                    && imin <= *vmin_v && *vmax_v <= imax {
                        return Some(x.cast(outer.clone()));
                    }
                None
            },
    }
}

/// Term combining patterns.
fn term_combining_dsl_patterns_unchecked() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // x + x → 2*x
        Add(x, x) => x.try_mul(&x.const_like(2i64)).ok(),
        // (x * c1) + (x * c2) → x * (c1 + c2)  (Mul[] is commutative, covers c*x too)
        Add(Mul[x, c1 @const(c1_val)], Mul[x, _c2 @const(c2_val)])
            => {
                let coeff = eval_add_typed(c1_val, c2_val, c1.dtype().base())
                    .expect("failed to add constants")
                    .into_uop(c1.dtype());
                x.try_mul(&coeff).ok()
            },
        // x + x*c → x*(c+1) — commutative outer Add
        Add[x, Mul[x, c @const(c_val)]] => {
            let one = ConstValue::one(c.dtype().base());
            let new_c = eval_add_typed(c_val, one, c.dtype().base()).expect("failed to add constants");
            x.try_mul(&UOp::const_(c.dtype(), new_c)).ok()
        },
        // (y + x*c0) + x*c1 → y + x*(c0+c1) — commutative outer Add
        Add[Add[y, Mul[x, c0 @const(c0_val)]], Mul[x, _c1 @const(c1_val)]] => {
            let new_c = eval_add_typed(c0_val, c1_val, c0.dtype().base()).expect("failed to add constants");
            let xc = x.try_mul(&UOp::const_(c0.dtype(), new_c)).ok()?;
            y.try_add(&xc).ok()
        },
        // (y + x) + x*c → y + x*(c+1) — commutative outer Add
        Add[Add[y, x], Mul[x, c @const(c_val)]] => {
            let one = ConstValue::one(c.dtype().base());
            let new_c = eval_add_typed(c_val, one, c.dtype().base()).expect("failed to add constants");
            let xc = x.try_mul(&UOp::const_(c.dtype(), new_c)).ok()?;
            y.try_add(&xc).ok()
        },
        // (y + x*c) + x → y + x*(c+1) — commutative outer Add
        Add[Add[y, Mul[x, c @const(c_val)]], x] => {
            let one = ConstValue::one(c.dtype().base());
            let new_c = eval_add_typed(c_val, one, c.dtype().base()).expect("failed to add constants");
            let xc = x.try_mul(&UOp::const_(c.dtype(), new_c)).ok()?;
            y.try_add(&xc).ok()
        },
        // (y + x) + x → y + x*2 — commutative outer Add
        Add[Add[y, x], x] => {
            let x2 = x.try_mul(&x.const_like(2i64)).ok()?;
            y.try_add(&x2).ok()
        },
        // Nested float division is not reassociated: doing so changes IEEE
        // rounding, overflow, underflow, and special-value behavior.
        // -(x+c) → -x + -c. This must precede the generic weak-int distribution.
        Mul[_neg @const(nv), Add[x, c @const(cv)]] if nv.is_neg_one() => {
            let neg_one = ConstValue::neg_one(c.dtype().base())?;
            let neg_cv = eval_mul_typed(cv, neg_one, c.dtype().base()).expect("failed to negate constant");
            Some(x.neg().add(&UOp::const_(c.dtype(), neg_cv)))
        },
        // y * (x + c) → y*x + y*c for weak integers.
        Mul[y @const(_yv), Add[x, c @const(_cv)]] if x.dtype() == DType::WeakInt => y.mul(x).add(&y.mul(c)),
    }
}

/// Advanced division and distribution patterns.
///
/// - (a // b) // c → a // (b * c)
/// - expr // divisor → expr.divides(divisor) (generic exact division)
/// - (a + b) % c → simplify when one operand is multiple of c
/// - (a + b) // c → (a // c) + (b // c) when both divide evenly
/// - (a - b) // c → (a // c) - (b // c) when both divide evenly
/// - c * (a + b) → (c * a) + (c * b)
/// - (a + b) * c → (a * c) + (b * c)
pub fn advanced_division_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // (a // b) // c → a // (b * c) if b,c non-zero
        original @ FloorDiv(FloorDiv(a, b @const(b_val)), _c @const(c_val)) if !b_val.is_zero() && !c_val.is_zero() => {
            let Op::Binary(BinaryOp::FloorDiv, _, c) = original.op() else { return None };
            if c_val.try_int()? <= 0 { return None; }
            if !integer_binary_does_not_wrap(BinaryOp::Mul, b, c, &b.dtype()) { return None; }
            let mul = eval_mul_typed(b_val, c_val, b.dtype().base()).expect("failed to multiply constants");
            exact_integer_rewrite(original, a.floor_div(&UOp::const_(b.dtype(), mul)))
        },
        // expr // divisor → expr.divides(divisor) (generic exact division)
        original @ FloorDiv(expr, _divisor @const(d_val)) => {
            let replacement = expr.divides(d_val.try_int()?)?;
            exact_integer_rewrite(original, replacement)
        },
        // (a + b) % c may drop an exactly divisible term when all source and
        // replacement arithmetic is proven not to wrap.
        original @ FloorMod(Add(a, b), divisor @const(d_val)) => {
            let divisor_value = d_val.try_int()?;
            let replacement = if a.divides(divisor_value).is_some() {
                b.mod_(divisor)
            } else if b.divides(divisor_value).is_some() {
                a.mod_(divisor)
            } else {
                return None;
            };
            exact_integer_rewrite(original, replacement)
        },
        // (x + c) % d → (x + c%d) % d for d > 0 (`uop/divandmod.py:102-105`).
        // The FLOORDIV half lives in `range_based_mod_div_patterns`.
        original @ FloorMod(Add[x, _c @const(c_val)], d @const(d_val)) => {
            let c_int = c_val.try_int()?;
            let d_int = d_val.try_int()?;
            if d_int <= 0 { return None; }
            let reduced = c_int.rem_euclid(d_int);
            if reduced == c_int { return None; }
            let replacement = x.add(&UOp::const_(d.dtype(), ConstValue::Int(reduced))).mod_(d);
            exact_integer_rewrite(original, replacement)
        },
        // Tinygrad's single div/mod folding entry point (`uop/divandmod.py:108`):
        // constant and symbolic divisors take the same path. The helper only
        // constructs a candidate; every source and replacement arithmetic node
        // is proven exact here.
        original @ FloorMod(x, y) => {
            let replacement = crate::symbolic::divmod::fold_divmod_general(BinaryOp::FloorMod, x, y)?;
            exact_integer_rewrite(original, replacement)
        },
        original @ FloorDiv(x, y) => {
            let replacement = crate::symbolic::divmod::fold_divmod_general(BinaryOp::FloorDiv, x, y)?;
            exact_integer_rewrite(original, replacement)
        },
        // (a - b) // c → (a // c) - (b // c) when both divide evenly.
        // The ADD counterpart is subsumed: `UOp::divides` recurses through ADD,
        // so `expr // divisor → expr.divides(divisor)` above already covers it.
        original @ FloorDiv(Sub(a, b), _c @const(c_val)) => {
            let d = c_val.try_int()?;
            let replacement = a.divides(d)?.sub(&b.divides(d)?);
            exact_integer_rewrite(original, replacement)
        },
        // (a//c1 + c2) // c3 → (a + c1*c2) // (c1*c3)
        // Moved from symbolic_simple to symbolic tier to avoid infinite loop with fast_division_patterns
        // in Stage 18-19 (fast div creates wider expr → nested div re-fires → ever-growing constants).
        // Guards: c1>0, c3>0, and (a>=0 && c2>=0) or (a<=0 && c2<=0) (same-sign requirement)
        original @ FloorDiv(Add[FloorDiv(a, c1 @const(c1_val)), _c2 @const(c2_val)], _c3 @const(c3_val)) => {
            let c1_int = c1_val.try_int()?;
            let c2_int = c2_val.try_int()?;
            let c3_int = c3_val.try_int()?;
            if c1_int <= 0 || c3_int <= 0 { return None; }
            let (a_vmin, a_vmax) = SoundVminVmaxProperty::get(a).as_ref()?;
            let (a_vmin, a_vmax) = (a_vmin.try_int()?, a_vmax.try_int()?);
            if !((a_vmin >= 0 && c2_int >= 0) || (a_vmax <= 0 && c2_int <= 0)) { return None; }
            let Op::Binary(BinaryOp::FloorDiv, _, c3) = original.op() else { return None };
            if !integer_binary_does_not_wrap(BinaryOp::Mul, c1, c3, &c1.dtype()) { return None; }
            let c2 = UOp::const_(c1.dtype(), c2_val);
            if !integer_binary_does_not_wrap(BinaryOp::Mul, c1, &c2, &c1.dtype()) { return None; }
            let c1_times_c2 = eval_mul_typed(c1_val, c2_val, c1.dtype().base()).expect("failed to evaluate cprod");
            let c1_times_c3 = eval_mul_typed(c1_val, c3_val, c1.dtype().base()).expect("failed to evaluate cprod");
            let replacement = a.add(&UOp::const_(c1.dtype(), c1_times_c2))
                .floor_div(&UOp::const_(c1.dtype(), c1_times_c3));
            exact_integer_rewrite(original, replacement)
        },
    }
}

/// Two-stage ALU folding patterns.
///
/// For all associative ops: (x op c1) op c2 → x op (c1 op c2)
/// : GroupOp.Associative = {Add, Mul, And, Or, Xor, Max}
fn alu_folding_dsl_patterns_unchecked() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // (x + c1) + c2 → x + (c1 + c2) - commutative outer Add
        Add[Add[x, c1 @const(c1_val)], _c2 @const(c2_val)] => {
            let csum = eval_add_typed(c1_val, c2_val, c1.dtype().base()).expect("failed to add constants");
            x.add(&UOp::const_(c1.dtype(), csum))
        },
        // Constant pushing: (x + c) + y → (x + y) + c
        Add[Add[x, c @const(_c_val)], y] if !matches!(y.op(), Op::Const(_)) => x.add(y).add(c),
        // (x * c1) * c2 → x * (c1 * c2) - commutative outer Mul
        Mul[Mul[x, c1 @const(c1_val)], _c2 @const(c2_val)] => {
            let cmul = eval_mul_typed(c1_val, c2_val, c1.dtype().base()).expect("failed to multiply constants");
            x.mul(&UOp::const_(c1.dtype(), cmul))
        },
        // Constant pushing: (x * c) * y → (x * y) * c
        Mul[Mul[x, c @const(_c_val)], y] if !matches!(y.op(), Op::Const(_)) => x.mul(y).mul(c),
        // Two-stage folding for remaining associative ops
        // (x & c1) & c2 → x & (c1 & c2)
        And[And[x, c1 @const(c1_val)], _c2 @const(c2_val)]
            => eval_binary_op(BinaryOp::And, c1_val, c2_val).map(|r| x.and_(&UOp::const_(c1.dtype(), r))),
        // (x | c1) | c2 → x | (c1 | c2)
        Or[Or[x, c1 @const(c1_val)], _c2 @const(c2_val)]
            => eval_binary_op(BinaryOp::Or, c1_val, c2_val).map(|r| x.or_(&UOp::const_(c1.dtype(), r))),
        // (x ^ c1) ^ c2 → x ^ (c1 ^ c2)
        Xor[Xor[x, c1 @const(c1_val)], _c2 @const(c2_val)]
            => eval_binary_op(BinaryOp::Xor, c1_val, c2_val).map(|r| x.xor(&UOp::const_(c1.dtype(), r))),
        // max(max(x, c1), c2) → max(x, max(c1, c2))
        Max(Max(x, c1 @const(c1_val)), _c2 @const(c2_val))
            => eval_binary_op(BinaryOp::Max, c1_val, c2_val).map(|r| x.try_max(&UOp::const_(c1.dtype(), r)).expect("max failed")),
        // (x - c1) + c2 → x + (c2 - c1) or x - (c1 - c2) - commutative outer Add
        Add[Sub(x, c1 @const(c1_val)), _c2 @const(c2_val)] => {
            let diff_val = eval_sub_typed(c2_val, c1_val, c1.dtype().base()).expect("failed to subtract constants");
            // Normalize: prefer x - |c| over x + (-c)
            if let ConstValue::Int(v) = diff_val && v < 0 {
                x.sub(&(-v).into_uop(c1.dtype()))
            } else {
                x.add(&UOp::const_(c1.dtype(), diff_val))
            }
        },
        // (x + c1) - c2 → x + (c1 - c2) or x - (c2 - c1) when result is negative
        Sub(Add(x, c1 @const(c1_val)), _c2 @const(c2_val)) => {
            let diff_val = eval_sub_typed(c1_val, c2_val, c1.dtype().base()).expect("failed to subtract constants");
            // Normalize: prefer x - |c| over x + (-c)
            if let Some(v) = diff_val.try_int() && v < 0 {
                x.sub(&(-v).into_uop(c1.dtype()))
            } else {
                x.add(&UOp::const_(c1.dtype(), diff_val))
            }
        },
        // (x - c1) - c2 → x - (c1 + c2)
        Sub(Sub(x, c1 @const(c1_val)), _c2 @const(c2_val)) => {
            let csum = eval_add_typed(c1_val, c2_val, c1.dtype().base()).expect("failed to add constants");
            x.sub(&UOp::const_(c1.dtype(), csum))
        },
        // SUB canonicalization: a - (b - x) → x + (a - b)
        Sub(a, Sub(b, x)) => x.add(&a.sub(b)),
    }
}

/// Dead loop elimination patterns.
///
/// - RANGE with vmax < 0 → Const(0)  (dead loop)
/// - RANGE(Const) with vmin == vmax → Const(vmin)  (single-value range)
///
/// END/REDUCE empty-ranges folds intentionally absent — they conflated
/// trivial Range(end=1) folds with dead-range markers; `reduce_to_acc`
/// already handles dead/empty ranges correctly.
pub fn dead_loop_patterns() -> &'static TypedPatternMatcher {
    /// Check if a Range is trivial (vmin == vmax), meaning only one value.
    fn is_trivial_range(uop: &Arc<UOp>) -> bool {
        let (vmin, vmax) = VminVmaxProperty::get(uop);
        vmin == vmax
    }

    /// Get the constant value for a trivial range (vmin which equals vmax).
    fn trivial_range_value(uop: &Arc<UOp>) -> Arc<UOp> {
        let (vmin, _) = VminVmaxProperty::get(uop);
        UOp::const_(uop.dtype(), *vmin)
    }

    crate::cached_patterns! {
        // RANGE with vmax < 0 (empty/dead) → Const(0)
        r @ Range { .. } if is_empty_range(r) => UOp::index_const(0),

        // RANGE(Const) with vmin == vmax (trivial) → Const(vmin)
        r @ Range { end: Const(_) } if is_trivial_range(r) => trivial_range_value(r),
    }
}

/// Vmin==Vmax collapse patterns.
///
/// When a node's vmin equals vmax, it's provably constant. Restricted to the
/// upstream op set — comparisons (`Lt`/`Le`/`Eq`/`Ne`/`Gt`/`Ge` → Bool), integer
/// `FloorDiv`/`FloorMod`, index `Param`/`Special` (PARAM/BIND/SPECIAL) — plus `Mul`
/// (size-1 grid `Special·stride → 0`, needed to fold hand-built index arithmetic).
///
/// Two op classes are deliberately EXCLUDED:
///   * **Float arithmetic** — a sound `[c, c]` integer-style bound does not transfer
///     to IEEE floats, where `inf - inf`, `0 * inf`, etc. carry a degenerate range
///     but evaluate to NaN at runtime. (Guarded by `!is_float`.)
///   * **`Add`/`Sub`/`Max`** on integers — upstream (`uop/symbolic.py:248-249`, whose
///     op set is `{CMPLT, CMPNE, FLOORDIV, FLOORMOD, PARAM, BIND, SPECIAL}`) does NOT
///     collapse these via the vmin==vmax rule. Collapsing an integer `Add` whose operands
///     are bounded to a single value would fold a hand-built kernel's trip-1 loop-carry
///     index to a constant and break the recurrence (the FA online-softmax `m`/`l`/`o` carry
///     reads a stale slot → NaN). `Mul` is safe because `0 · x = 0` and `c · c` are
///     exact regardless of the operand's loop structure.
fn vmin_vmax_collapse_patterns_unchecked() -> &'static TypedPatternMatcher {
    use svod_ir::uop::properties::SoundVminVmaxProperty;

    // Collapse only computation nodes whose result is non-float (int/bool/index).
    // Structural nodes (Range, Buffer) and float arithmetic are excluded.
    fn is_collapsible(uop: &Arc<UOp>) -> bool {
        matches!(uop.op(), Op::Binary(..) | Op::Param(..) | Op::Special(..)) && !uop.dtype().is_float()
    }

    fn try_collapse(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
        let (vmin, vmax) = SoundVminVmaxProperty::get(uop).as_ref()?;
        if vmin == vmax { Some(uop.const_like(*vmin)) } else { None }
    }

    crate::cached_patterns! {
        // Comparisons (→ Bool), integer FloorDiv/FloorMod, and Mul with sound vmin == vmax → const.
        // SoundVminVmaxProperty returns None for unsound ops (LOAD/Pow/Fdiv); `is_collapsible`
        // excludes float results; Add/Sub/Max are omitted (see fn doc).
        for op in binary [Mul, FloorDiv, FloorMod, Lt, Le, Eq, Ne, Gt, Ge] {
            r @ op(_, _) if is_collapsible(r) => try_collapse(r),
        },
        // Param/Special with vmin == vmax → const (e.g., Variable with min==max after binding)
        r @ Param { shape: _, arg: _ } if is_collapsible(r) => try_collapse(r),
        r @ Special { end: _, name: _ } if is_collapsible(r) => try_collapse(r),
    }
}

/// Dead code elimination patterns.
///
/// Handles WHERE optimizations:
/// - WHERE(true, t, f) → t
/// - WHERE(false, t, f) → f
/// - WHERE(_, t, t) → t (same branches)
/// - WHERE(x, true, false) → x (bool)
/// - WHERE(x, false, true) → !x (bool)
/// - WHERE(!cond, t, f) → WHERE(cond, f, t)
///
/// DCE patterns for `symbolic_simple` tier — basic WHERE simplifications.
///
/// These patterns don't introduce NOT or swap branches, safe for all stages.
fn dce_dsl_simple_patterns_unchecked() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // WHERE with constant condition → select appropriate branch
        Where(cond, true_val, false_val) => {
            match SoundVminVmaxProperty::get(cond) {
                Some((ConstValue::Bool(true), ConstValue::Bool(true))) => Some(Arc::clone(true_val)),
                Some((ConstValue::Bool(false), ConstValue::Bool(false))) => Some(Arc::clone(false_val)),
                _ => None,
            }
        },

        // WHERE(_, same, same) → same
        Where(_, t, t) => Arc::clone(t),

        // WHERE(x, true, false) → x (for bool x)
        Where(x, _t @const(t_val), _f @const(f_val))
          if x.dtype() == DType::Bool && t_val == ConstValue::Bool(true) && f_val == ConstValue::Bool(false)
          => Arc::clone(x),

        // WHERE(x, false, true) → !x (for bool x)
        Where(x, _t @const(t_val), _f @const(f_val))
          if x.dtype() == DType::Bool && t_val == ConstValue::Bool(false) && f_val == ConstValue::Bool(true)
          => x.not(),

        // WHERE(a, WHERE(b, c, d), d) → WHERE(a & b, c, d) - branch merging
        Where(a, Where(b, c, d), d) => {
            let combined_cond = a.and_(b);
            UOp::try_where(combined_cond, Arc::clone(c), Arc::clone(d)).ok()
        },
    }
}

/// DCE patterns for `symbolic` tier — negated condition swap.
///
/// WHERE(!cond, t, f) → WHERE(cond, f, t) belongs in `symbolic`.
/// Separated from simple patterns because it introduces branch swaps that interact
/// with propagate_invalid at higher complexity.
pub fn dce_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // WHERE(!cond, t, f) → WHERE(cond, f, t) - negated condition swap
        // Guard: don't swap when f contains Invalid — PAD creates WHERE(valid, idx, Invalid),
        // and swapping would move Invalid to the true branch where downstream patterns can't match it.
        // Handles scalar Invalid and STACK(Invalid, ...) from expansion.
        //  has this same guard.
        Where(Not(cond), t, f)
            if !has_invalid(f)
            => UOp::try_where(Arc::clone(cond), Arc::clone(f), Arc::clone(t)).ok(),
    }
}

/// AFTER simplification patterns.
///
/// - AFTER(x, []) → x (empty deps, just passthrough)
pub fn after_simplification_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // AFTER recursive dep flattening + dedup (matches tinygrad symbolic.py:307-311):
        // For each dep, if it's not a side-effecting op, replace it with its sources.
        // Then deduplicate by UOp id to prevent dep-list bloat.
        After { passthrough, deps } if !deps.is_empty() => {
            let mut new_deps = smallvec::SmallVec::<[Arc<UOp>; 4]>::new();
            let mut changed = false;
            for dep in deps {
                // Side-effect boundaries that survive AFTER inlining.
                // Side-effecting operations survive dependency inlining.
                if matches!(
                    dep.op(),
                    Op::Range(..)
                        | Op::Store(..)
                        | Op::End(..)
                        | Op::Call(..)
                        | Op::Barrier(..)
                        | Op::Custom(..)
                        | Op::Function(..)
                ) {
                    new_deps.push(Arc::clone(dep));
                } else {
                    // Inline: replace non-side-effecting dep with its children
                    for child in dep.op().sources() {
                        new_deps.push(child);
                    }
                    changed = true;
                }
            }
            if changed {
                // Dedup by UOp id (matches tinygrad's dedup(flatten(...)))
                let mut seen = std::collections::HashSet::new();
                new_deps.retain(|d| seen.insert(d.id));
                if new_deps.is_empty() {
                    Some(Arc::clone(passthrough))
                } else {
                    Some(passthrough.after(new_deps))
                }
            } else {
                None
            }
        },
        // AFTER(x, []) → x: empty dependencies means no ordering constraint
        After { passthrough, deps } if deps.is_empty() => Arc::clone(passthrough),

        // Remove NOOP and recursive empty-END deps from AFTER (matches tinygrad rangeify.py:458-479).
        // A noop_after_dep is: NOOP with no sources, or END whose computation is also a noop_after_dep.
        After { passthrough, deps } if deps.iter().any(is_noop_after_dep) => {
            let new_deps: smallvec::SmallVec<[Arc<UOp>; 4]> =
                deps.iter().filter(|d| !is_noop_after_dep(d)).cloned().collect();
            if new_deps.is_empty() {
                Some(Arc::clone(passthrough))
            } else {
                Some(passthrough.after(new_deps))
            }
        },
    }
}

/// Check if a UOp is a noop AFTER dep (matches tinygrad's `is_noop_after_dep`).
/// A noop_after_dep is: NOOP with no sources, or END whose computation is also a noop_after_dep.
fn is_noop_after_dep(u: &Arc<UOp>) -> bool {
    match u.op() {
        Op::Noop => true,
        Op::End(ops::End { computation, .. }) => is_noop_after_dep(computation),
        _ => false,
    }
}

/// Move WHERE conditions into index validity.
///
/// Transforms `WHERE(cond, INDEX(buf, idx), 0)` by moving safe conditions into
/// `INDEX(buf, WHERE(cond, idx, Invalid))`.
///
/// This optimization:
/// 1. Eliminates the WHERE operation overhead
/// 2. Enables hardware predication for masked loads
/// 3. Allows the backend to generate efficient conditional load instructions
///
/// **Critical**: This pattern runs at Stage 8 (Post-Opt Symbolic), BEFORE LOADs are added
/// at Stage 13. Therefore, it matches INDEX directly, not LOAD(INDEX).
///
/// Matches `pm_move_where_on_load` pattern:
/// ```python
/// (UPat.var("c1").where(UPat.var("buf").index(UPat.var("x")), 0), where_on_load),
/// ```
///
/// Moved clauses are embedded as `WHERE(cond, idx, Invalid)` in `indices[0]`.
///
/// The condition can be moved if:
/// - All RANGE dependencies in the condition are within the INDEX's range scope
/// - The condition doesn't depend on other INDEX operations (avoids speculative loads)
pub fn pm_move_where_on_load() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Pattern 1: WHERE(cond, INDEX(buf, idx, None), 0)
        // Embed cond clauses as WHERE-Invalid in INDEX indices[0]
        // Note: Matches INDEX directly (no LOAD), since this runs at Stage 8
        Where(cond, idx @ Index { buffer, indices }, f @ const(false_val)) if false_val.is_zero() => {
            where_on_load_index_transform(cond, buffer, indices, f, idx.dtype())
        },

        // Pattern 2: WHERE(cond, 0, INDEX(buf, idx, None)) - inverted pattern
        // Use !cond embedded as WHERE-Invalid
        Where(cond, f @ const(false_val), idx @ Index { buffer, indices }) if false_val.is_zero() => {
            let not_cond = cond.not();
            where_on_load_index_transform(&not_cond, buffer, indices, f, idx.dtype())
        },
    }
}

/// Check if a UOp is or contains Invalid (scalar or vectorized).
fn has_invalid(uop: &Arc<UOp>) -> bool {
    match uop.op() {
        Op::Const(ConstValueHash(ConstValue::Invalid)) => true,
        Op::Stack(ops::Stack { sources }) => sources.iter().any(UOp::is_invalid_marker),
        _ => false,
    }
}

/// Transform WHERE(cond, INDEX(buf, idx), 0) by embedding moveable clauses as WHERE-Invalid in indices[0].
///
/// This is the Stage 8 version that works directly with INDEX, matching upstream approach.
/// LOADs are added later at Stage 13.
///
/// Supports **partial clause movement** (upstream: where_on_load):
/// - Splits condition into AND clauses
/// - Moves only clauses where ALL ranges are within index scope AND no load dependencies
/// - Keeps remaining clauses in outer WHERE
/// - Deduplicates clauses already present in indices[0]'s existing WHERE-Invalid validity
///
/// Embeds moved clauses as WHERE(combined_cond, clean_idx, Invalid) in indices[0].
fn where_on_load_index_transform(
    cond: &Arc<UOp>,
    idx_buf: &Arc<UOp>,
    indices: &SmallVec<[Arc<UOp>; 4]>,
    false_val: &Arc<UOp>,
    index_dtype: DType,
) -> Option<Arc<UOp>> {
    // Step 1: Split condition into AND clauses
    let c1_clauses = split_and_clauses(cond);

    // Step 2: Get existing validity clauses from indices[0] (handles re-application)
    let existing_valid = indices.first()?.get_valid();
    let c2_clauses: Vec<Arc<UOp>> = if matches!(existing_valid.op(), Op::Const(cv) if cv.0 == ConstValue::Bool(true)) {
        vec![]
    } else {
        split_and_clauses(&existing_valid)
    };

    // Step 3: Find duplicate clauses (already in existing validity)
    let duplicate_ids: std::collections::HashSet<u64> =
        c1_clauses.iter().filter(|c| c2_clauses.iter().any(|c2| c.id == c2.id)).map(|c| c.id).collect();

    // Step 4: Collect RANGE and INDEX ids reachable from indices (index scope)
    // All INDEX ops in the idx backward slice
    let mut index_ranges = std::collections::HashSet::new();
    let mut idx_indices = std::collections::HashSet::new();
    for idx in indices {
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![idx.clone()];
        while let Some(node) = stack.pop() {
            if !visited.insert(Arc::as_ptr(&node)) {
                continue;
            }
            match node.op() {
                Op::Range(..) => {
                    index_ranges.insert(node.id);
                }
                Op::Index(..) => {
                    idx_indices.insert(node.id);
                }
                _ => {}
            }
            node.op().map_child(|child| {
                if !visited.contains(&Arc::as_ptr(child)) {
                    stack.push(child.clone());
                }
            });
        }
    }

    // Step 5: Partition clauses into moveable vs remaining
    // Single DFS per clause: check range scope + index deps simultaneously
    // can_move: clause ranges ⊆ idx ranges AND all INDEX ops are in idx_index
    let (moved_clauses, remaining_clauses): (Vec<_>, Vec<_>) = c1_clauses.iter().cloned().partition(|clause| {
        if duplicate_ids.contains(&clause.id) {
            return true; // Treat as "moved" (but won't add to validity)
        }

        let mut ranges_in_scope = true;
        let mut has_index_deps = false;
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![clause.clone()];
        while let Some(node) = stack.pop() {
            if !visited.insert(Arc::as_ptr(&node)) {
                continue;
            }
            match node.op() {
                Op::Range(..) if !index_ranges.contains(&node.id) => {
                    ranges_in_scope = false;
                    break; // Out-of-scope range found, can't move
                }
                Op::Index(..) if !idx_indices.contains(&node.id) => {
                    has_index_deps = true;
                    break; // External INDEX dep found, can't move
                }
                _ => {}
            }
            node.op().map_child(|child| {
                if !visited.contains(&Arc::as_ptr(child)) {
                    stack.push(child.clone());
                }
            });
        }

        ranges_in_scope && !has_index_deps
    });

    // Step 6: If no movement possible and no duplicates removed, return None
    let actually_moved: Vec<_> = moved_clauses.into_iter().filter(|c| !duplicate_ids.contains(&c.id)).collect();

    if actually_moved.is_empty() && duplicate_ids.is_empty() {
        return None; // Nothing to move or deduplicate
    }

    // Step 7: Build combined validity (moved clauses + existing validity)
    let mut validity_clauses: Vec<Arc<UOp>> = actually_moved;
    validity_clauses.extend(c2_clauses);

    // Step 8: Create INDEX with WHERE-Invalid in indices[0]
    let clean_idx = indices.first()?.get_idx();
    let new_idx = if validity_clauses.is_empty() {
        clean_idx
    } else {
        let combined_valid = validity_clauses.into_iter().reduce(|a, b| a.and_(&b)).unwrap();
        clean_idx.valid(combined_valid)
    };
    let mut new_indices = indices.clone();
    new_indices[0] = new_idx;

    let new_index = UOp::index()
        .buffer(idx_buf.clone())
        .indices(new_indices)
        .call()
        .expect("where_on_load_index_transform: INDEX construction failed")
        .with_dtype(index_dtype);

    // Step 9: Wrap in remaining WHERE if there are non-moved clauses
    if remaining_clauses.is_empty() {
        Some(new_index)
    } else {
        let remaining_cond = remaining_clauses.into_iter().reduce(|a, b| a.and_(&b)).unwrap();
        UOp::try_where(remaining_cond, new_index, false_val.clone()).ok()
    }
}

/// Split a condition into its AND clauses recursively.
fn split_and_clauses(cond: &Arc<UOp>) -> Vec<Arc<UOp>> {
    match cond.op() {
        Op::Binary(BinaryOp::And, left, right) => {
            let mut result = split_and_clauses(left);
            result.extend(split_and_clauses(right));
            result
        }
        _ => vec![cond.clone()],
    }
}

/// Cast pushing through WHERE patterns.
///
/// - where(s, a, b).cast(dtype) → where(s, a.cast(dtype), b.cast(dtype))
pub fn cast_where_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // cast(where(s, a, b), dtype) → where(s, cast(a, dtype), cast(b, dtype))
        Cast { src: Where(s, a, b), dtype } => {
            let cast_a = a.cast(dtype.clone());
            let cast_b = b.cast(dtype.clone());
            UOp::try_where(s.clone(), cast_a, cast_b).expect("failed to create WHERE")
        },
    }
}

/// Comparison patterns.
///
/// Handles all comparison operations with:
/// - Self-comparison fast path (x op x)
/// - Constant folding
/// - Range-based analysis via vmin/vmax
/// - Const offset: (c0 + x) < c1 → x < (c1 - c0)
/// - Negation flip: -x < -y → y < x
fn comparison_dsl_patterns_unchecked() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        for op in binary [Lt, Le, Eq, Ne, Gt, Ge] {
            op(x, y) => {
                // 1. Self-comparison fast path (non-float only)
                if Arc::ptr_eq(x, y) && !x.dtype().is_float() {
                    let result = match op {
                        BinaryOp::Lt | BinaryOp::Gt | BinaryOp::Ne => ConstValue::Bool(false),
                        BinaryOp::Le | BinaryOp::Ge | BinaryOp::Eq => ConstValue::Bool(true),
                        _ => return None,
                    };
                    return Some(UOp::const_(DType::Bool, result));
                }

                // 2. Constant folding
                if !x.dtype().is_weak()
                    && !y.dtype().is_weak()
                    && let (Some(a_val), Some(b_val)) = (get_const_value(x), get_const_value(y))
                    && let Some(result) = eval_binary_op(op, a_val, b_val)
                {
                    return Some(UOp::const_(DType::Bool, result));
                }

                // 3. Range-based analysis
                if !x.dtype().is_weak()
                    && !y.dtype().is_weak()
                    && let Some(result) = ComparisonAnalyzer::analyze(op, x, y)
                {
                    return Some(result.into_uop(DType::Bool));
                }

                None
            },
        },

        // (c0 + x) < c1 → x < (c1 - c0) for integers - commutative
        Lt(Add[c0 @const(c0_val), x], _c1 @const(c1_val)) => {
            let diff = eval_sub_typed(c1_val, c0_val, c0.dtype().base()).expect("failed to evaluate sub");
            x.try_cmplt(&UOp::const_(c0.dtype(), diff)).expect("failed to create cmplt")
        },

        // MUL(x,-1) < MUL(y,-1) → y < x (negation flip for Lt)
        // neg() produces MUL(x, -1), so we match that form.
        Lt(Mul[x, _c1 @const(c1v)], Mul[y, _c2 @const(c2v)])
            if c1v.is_neg_one() && c2v.is_neg_one()
            => y.try_cmplt(x).expect("failed to create cmplt"),

        // Phase 6: (x // d) < c → x < (c * d) when d > 0
        // This lifts division out of comparisons, enabling further simplification.
        // Based on upstream
        Lt(quotient @ FloorDiv(x, _d @const(d_val)), _c @const(c_val)) => {
            let d_int = d_val.try_int()?;
            let c_int = c_val.try_int()?;
            if d_int <= 0 || !integer_arithmetic_does_not_wrap(&mut WrapMemo::default(), quotient) { return None; }

            // For positive d, floor(x / d) < c is exactly x < c * d.
            let product = (c_int as i128).checked_mul(d_int as i128)?;
            let bound = product;
            let (dtype_min, dtype_max) = integer_dtype_bounds(&x.dtype())?;
            if bound < dtype_min || bound > dtype_max { return None; }

            Some(x.try_cmplt(&UOp::const_(x.dtype(), ConstValue::Int(bound as i64))).expect("failed to create cmplt"))
        },

        // c0*x < c1 → sign(c0)*x < ceil(c1/abs(c0)) for weak integers
        Lt(Mul[_c0 @const(c0_val), x], _c1 @const(c1_val))
          if x.dtype() == DType::WeakInt
          => {
            let c0 = c0_val.try_int()?;
            let c1 = c1_val.try_int()?;
            let abs_c0 = (c0 as i128).abs();
            if abs_c0 <= 1 { return None; }
            let ceil_div = -(-(c1 as i128)).div_euclid(abs_c0);
            let lhs = if c0 > 0 { x.clone() } else { x.neg() };
            Some(lhs.try_cmplt(&UOp::const_(x.dtype(), ConstValue::Int(ceil_div as i64))).expect("failed to create cmplt"))
          },

        // Lt(x, c) with GCD-based folding for weak integers.
        Lt(x, _c @const(cv)) if x.dtype() == DType::WeakInt => {
            let c_int = cv.try_int()?;
            if c_int <= 0 { return None; }
            lt_folding(x, c_int)
          },
    }
}

/// GCD-based Lt folding (, 236).
///
/// Split x into add terms, partition into unit-factor (|const_factor| <= 1)
/// and non-unit terms. Compute d = gcd(non-unit factors, c). If d > 1 and
/// the unit-factor sum is bounded in [0, d), then x = d*q + r with r in [0, d),
/// so (x < c) iff (q < c/d) since d divides c.
fn lt_folding(x: &Arc<UOp>, c_int: i64) -> Option<Arc<UOp>> {
    let terms = x.split_uop(BinaryOp::Add);
    if terms.len() < 2 {
        return None;
    }

    // Partition terms by const_factor: exactly 1 → unit, otherwise → non-unit
    // Split addends by const_factor == 1
    let mut unit_terms = Vec::new();
    let mut non_unit_factors = Vec::new();
    for t in &terms {
        let f = t.const_factor();
        if f == 1 {
            unit_terms.push(Arc::clone(t));
        } else {
            non_unit_factors.push(f);
        }
    }

    if non_unit_factors.is_empty() || unit_terms.is_empty() {
        return None;
    }

    // Compute GCD of non-unit factors AND c (d = gcd(*factors, c)
    let mut d = c_int.unsigned_abs() as i64;
    for &f in &non_unit_factors {
        d = gcd(d, f);
    }
    if d <= 1 {
        return None;
    }

    // Check that unit sum is in [0, d)
    let unit_sum = super::divmod::uop_sum(&unit_terms, x);
    let (us_vmin, us_vmax) = VminVmaxProperty::get(&unit_sum);
    let us_min = us_vmin.try_int()?;
    let us_max = us_vmax.try_int()?;
    if us_min < 0 || us_max >= d {
        return None;
    }

    // Build the non-unit sum divided by d (Build non-unit sum divided by d
    let non_unit_terms: Vec<Arc<UOp>> = terms.iter().filter(|t| t.const_factor() != 1).cloned().collect();
    let non_unit_sum = super::divmod::uop_sum(&non_unit_terms, x);
    let q = non_unit_sum.divides(d)?;

    // Since d | c, use exact division (no ceiling needed)
    q.try_cmplt(&UOp::index_const(c_int / d)).ok()
}

/// Compute GCD of two positive integers.
fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a as i64
}

/// Boolean logic patterns.
///
/// - !!x → x (double negation elimination)
/// - x ^ x → 0 (xor self-cancellation)
/// - x | !x → true (tautology)
/// - x & !x → false (contradiction)
/// - true | x → true, false & x → false
/// - true & x → x, false | x → x (identity)
/// - (!x) & (!y) → !(x | y) (De Morgan's law)
/// - (!x) | (!y) → !(x & y) (De Morgan's law)
///
/// Basic boolean patterns for `symbolic_simple` tier.
///
/// Matches upstream symbolic_simple:
/// NOT(NOT(x))→x, XOR(x,x)→0, bool const AND/OR identity.
pub fn boolean_dsl_simple_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // !!x → x
        Not(Not(x)) => x.clone(),
        // x ^ x → 0
        Xor(x, x) => x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::zero(dt))),

        // Bool const identity (upstream symbolic_simple):
        // bool & c → x if c else 0; bool | c → c if c else x
        // true | x → true (commutative)
        Or[t @const(t_val), _] if t_val == ConstValue::Bool(true) => t.clone(),
        // false & x → false (commutative)
        And[f @const(f_val), _] if f_val == ConstValue::Bool(false) => f.clone(),
        // true & x → x (identity, commutative)
        And[_c @const(c_val), x] if c_val == ConstValue::Bool(true) => x.clone(),
        // false | x → x (identity, commutative)
        Or[_c @const(c_val), x] if c_val == ConstValue::Bool(false) => x.clone(),
    }
}

/// Full boolean patterns for `symbolic` tier.
///
/// Tautology, contradiction, De Morgan — these belong in `symbolic`
/// (, decompositions.py).
pub fn boolean_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // x | !x → true (tautology) - commutative
        Or[x, Not(x)] if x.dtype() == DType::Bool => UOp::const_(DType::Bool, ConstValue::Bool(true)),

        // x & !x → false (contradiction) - commutative
        And[x, Not(x)] if x.dtype() == DType::Bool => UOp::const_(DType::Bool, ConstValue::Bool(false)),

        // De Morgan's laws (upstream decompositions)
        // (!x) & (!y) → !(x | y)
        And[Not(x), Not(y)] => x.or_(y).not(),

        // (!x) | (!y) → !(x & y)
        Or[Not(x), Not(y)] => x.and_(y).not(),
    }
}

/// Min/max elimination via bounds analysis.
///
/// Based on :
///   `max(x, y) → x if x.vmin >= y.vmax else y if x.vmax <= y.vmin`
fn minmax_dsl_patterns_unchecked() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Max(x, x) → x is now in self_folding_dsl_patterns (GroupOp.Idempotent)
        Max(x, y) => {
            let (x_vmin, x_vmax) = SoundVminVmaxProperty::get(x).as_ref()?;
            let (y_vmin, y_vmax) = SoundVminVmaxProperty::get(y).as_ref()?;
            // Equality is enough for integers, but not floats: selecting one
            // operand across an equal endpoint can change the sign of zero.
            if bounds_select_left(x_vmin, y_vmax, &x.dtype()) {
                return Some(Arc::clone(x));
            }
            if bounds_select_left(y_vmin, x_vmax, &y.dtype()) {
                return Some(Arc::clone(y));
            }
            None
        },
    }
}

fn bounds_select_left(lhs_min: &ConstValue, rhs_max: &ConstValue, dtype: &DType) -> bool {
    if dtype.is_float() { cv_gt(lhs_min, rhs_max) } else { cv_ge(lhs_min, rhs_max) }
}

/// WHERE condition elimination via bounds analysis.
///
/// Eliminates WHERE(Lt) when the condition is provably always true or false.
/// Uses vmin/vmax to determine if x < c holds for all possible values.
fn where_bound_patterns_unchecked() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        Where(Lt(x, c), t, f) => {
            let (x_vmin, x_vmax) = SoundVminVmaxProperty::get(x).as_ref()?;
            let (c_vmin, c_vmax) = SoundVminVmaxProperty::get(c).as_ref()?;
            // Always true: x.vmax < c.vmin → take true branch
            if cv_lt(x_vmax, c_vmin) { return Some(Arc::clone(t)); }
            // Always false: x.vmin >= c.vmax → take false branch
            if cv_ge(x_vmin, c_vmax) { return Some(Arc::clone(f)); }
            None
        },
    }
}

/// Compare ConstValue: a >= b
fn cv_ge(a: &ConstValue, b: &ConstValue) -> bool {
    match (a, b) {
        (ConstValue::Int(a), ConstValue::Int(b)) => a >= b,
        (ConstValue::UInt(a), ConstValue::UInt(b)) => a >= b,
        (ConstValue::Float(a), ConstValue::Float(b)) => a >= b,
        _ => false,
    }
}

/// Compare ConstValue: a < b
fn cv_lt(a: &ConstValue, b: &ConstValue) -> bool {
    match (a, b) {
        (ConstValue::Int(a), ConstValue::Int(b)) => a < b,
        (ConstValue::UInt(a), ConstValue::UInt(b)) => a < b,
        (ConstValue::Float(a), ConstValue::Float(b)) => a < b,
        _ => false,
    }
}

/// Compare ConstValue: a > b
fn cv_gt(a: &ConstValue, b: &ConstValue) -> bool {
    match (a, b) {
        (ConstValue::Int(a), ConstValue::Int(b)) => a > b,
        (ConstValue::UInt(a), ConstValue::UInt(b)) => a > b,
        (ConstValue::Float(a), ConstValue::Float(b)) => a > b,
        _ => false,
    }
}

/// Power patterns (, 103-105).
///
/// Handles: x^0→1, x^1→x, negative/half-integer/integer exponents, const-base.
fn power_dsl_patterns_unchecked() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // x ** c (scalar const exponent) — upstream simplify_pow
        Pow(x, c @const(cv)) => simplify_pow(x, c, cv),
        // c ** x (scalar const base) — upstream
        Pow(c @const(cv), x) => simplify_pow_const_base(c, cv, x),
    }
}

/// upstream `simplify_pow` (upstream symbolic_simple).
fn simplify_pow(x: &Arc<UOp>, _c: &Arc<UOp>, cv: ConstValue) -> Option<Arc<UOp>> {
    // Only scalar consts (cvar vec=False)
    if x.dtype().vcount() > 1 {
        return None;
    }
    let f = match cv {
        ConstValue::Float(f) => f,
        ConstValue::Int(i) => i as f64,
        _ => return None,
    };
    if f == 0.0 {
        // x^0 → 1
        return x.dtype().scalar().map(|dt| UOp::const_(x.dtype(), ConstValue::one(dt)));
    }
    if f == 1.0 {
        // x^1 → x
        return Some(Arc::clone(x));
    }
    // Reciprocal, sqrt, and repeated-multiply decompositions change IEEE
    // rounding and special-value behavior. Keep POW unless it is an identity.
    None
}

/// Const-base power (upstream symbolic_simple).
fn simplify_pow_const_base(c: &Arc<UOp>, cv: ConstValue, _x: &Arc<UOp>) -> Option<Arc<UOp>> {
    // Only scalar consts
    if c.dtype().vcount() > 1 {
        return None;
    }
    let f = match cv {
        ConstValue::Float(f) => f,
        ConstValue::Int(i) => i as f64,
        _ => return None,
    };
    if f == 1.0 {
        // 1^x → 1
        return Some(Arc::clone(c));
    }
    // exp2(x*log2(c)) is not an exact replacement for POW under IEEE
    // rounding, overflow, and transcendental domain behavior.
    None
}

/// ALU(STACK, STACK) → STACK(scalar_ALU) reordering (upstream sym).
///
/// When both operands are broadcast STACK nodes, collapse to a STACK of the
/// scalar operation replicated N times.
/// This enables better constant folding and scalar optimization.
fn alu_vectorize_reorder_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        for op in binary [Add, Mul, Sub, FloorMod, Max, FloorDiv, Fdiv, Pow, And, Or, Xor, Shl, Shr, Lt, Le, Eq, Ne, Gt, Ge] {
            r @ op(Stack { sources: x_elems }, Stack { sources: y_elems })
                if x_elems.len() == y_elems.len()
                && x_elems.len() > 1
                && x_elems.windows(2).all(|w| Arc::ptr_eq(&w[0], &w[1]))
                && y_elems.windows(2).all(|w| Arc::ptr_eq(&w[0], &w[1]))
                => {
                    let scalar_dtype = r.dtype().scalar_dtype();
                    let count = x_elems.len();
                    let scalar_alu = UOp::new(Op::Binary(op, x_elems[0].clone(), y_elems[0].clone()), scalar_dtype);
                    let elems: SmallVec<[Arc<UOp>; 4]> = std::iter::repeat_n(scalar_alu, count).collect();
                    Some(UOp::stack(elems))
                },
        },
    }
}

/// x != 0 → (bool)x self-folding (upstream sym).
///
/// Non-zero comparison folds to cast-to-bool. No dtype guard — matches upstream exactly.
fn ne_zero_fold_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        Ne(x, _zero @const(zv)) if zv.is_zero() => {
            let bool_dt = DType::Bool.vec(x.dtype().vcount()).expect("Bool is a scalar");
            Some(x.cast(bool_dt))
        },
    }
}

/// Reduce patterns for sym tier (upstream sym).
///
/// - (x*c).reduce(ADD) → reduce(x, ADD) * c  (move const multiply after reduce)
/// - MUL(...).reduce(r) → reduce_mul_chain(r) (factor multiplicative terms out)
fn reduce_sym_patterns() -> &'static TypedPatternMatcher {
    use svod_ir::types::ReduceOp;

    crate::cached_patterns! {
        // Pull scalar const OUT of reduce: REDUCE(x * c, ADD) → REDUCE(x, ADD) * c
        // : (x*c).reduce(ADD) → reduce(x)*c
        // `vec=False` means scalar const — `@const` already matches Op::Const only.
        Reduce { src: Mul[x, c @const(_cv)], ranges, reduce_op, num_axes }
            if *reduce_op == ReduceOp::Add
            && c.dtype().vcount() == 1
            && !x.dtype().is_float()
            => {
                let new_reduce = x.reduce_with_num_axes(ranges.clone(), ReduceOp::Add, *num_axes);
                // Cast const to reduce output dtype if needed
                let c_typed = if c.dtype() == new_reduce.dtype() {
                    Arc::clone(c)
                } else {
                    c.cast(new_reduce.dtype())
                };
                new_reduce.try_mul(&c_typed).ok()
            },

        // reduce_mul_chain: factor range-independent multipliers outside REDUCE
        //  + reduce_mul_chain (line 332-341)
        // Guard: r.dtype != r.src[0].dtype → return None (upstream)
        // This prevents firing on horizontal reduces (body has wider dtype than output).
        reduce @ Reduce { src, ranges, reduce_op, num_axes }
            if matches!(reduce_op, ReduceOp::Add | ReduceOp::Max)
            && matches!(src.op(), Op::Binary(BinaryOp::Mul, _, _))
            && reduce.dtype() == src.dtype()
            && !src.dtype().is_float()
            => {
                reduce_mul_chain_sym(src, ranges, *reduce_op, *num_axes)
            },
    }
}

/// Factor range-independent multipliers outside REDUCE.
///
/// For REDUCE(MUL(a, b, ...), ranges), if some factors don't depend on any reduce range,
/// pull them outside: REDUCE(remaining, ranges) * outside_factors.
fn reduce_mul_chain_sym(
    src: &Arc<UOp>,
    ranges: &SmallVec<[Arc<UOp>; 4]>,
    reduce_op: svod_ir::types::ReduceOp,
    num_axes: usize,
) -> Option<Arc<UOp>> {
    use svod_ir::types::ReduceOp;

    if !matches!(reduce_op, ReduceOp::Add | ReduceOp::Max) {
        return None;
    }

    // Split src into multiplicative factors
    let factors = src.split_uop(BinaryOp::Mul);

    // Collect range ids for quick lookup
    let range_ids: std::collections::HashSet<u64> = ranges.iter().map(|r| r.id).collect();

    // Partition into inside (depends on ranges) and outside (range-independent)
    let mut inside = Vec::new();
    let mut outside = Vec::new();
    for factor in &factors {
        let depends_on_range = factor.any_in_subtree(|n| range_ids.contains(&n.id));
        if !depends_on_range
            && (reduce_op != ReduceOp::Max
                || matches!(SoundVminVmaxProperty::get(factor), Some((ConstValue::Int(v), _)) if *v >= 0))
        {
            outside.push(Arc::clone(factor));
        } else {
            inside.push(Arc::clone(factor));
        }
    }

    if outside.is_empty() {
        return None;
    }

    // Rebuild inside product (or const 1 if empty)
    let inside_prod = if inside.is_empty() {
        src.const_like(ConstValue::one(src.dtype().base()))
    } else {
        inside.into_iter().reduce(|a, b| a.try_mul(&b).expect("mul failed")).unwrap()
    };

    // Create reduced inside, multiply by outside factors
    let reduced = inside_prod.reduce_with_num_axes(ranges.clone(), reduce_op, num_axes);
    let outside_prod = outside.into_iter().reduce(|a, b| a.try_mul(&b).expect("mul failed")).unwrap();
    reduced.try_mul(&outside_prod).ok()
}

/// `body` as it reads under a `cond` that now dominates it, which makes every
/// copy of the same test inside redundant. A conv fused into a concat keeps one
/// in the gate on its own input, and left there that gate still reads the
/// concatenated output range — enough for `tc::detect_matmul` to decide the
/// input varies along the output channel and to lose the matmul's N axis.
///
/// INDEX subtrees are left alone. The same test also guards the address of the
/// out-of-range half, and a WHERE lowers to a select rather than a branch, so
/// the load still runs for every lane and dropping that copy reads out of
/// bounds. Value redundancy and address validity are not the same redundancy.
fn assuming(body: &Arc<UOp>, cond: &Arc<UOp>) -> Arc<UOp> {
    let known: std::collections::HashSet<u64> =
        cond.split_uop(BinaryOp::And).iter().chain([cond]).map(|clause| clause.id).collect();
    let truth = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let mut done = std::collections::HashMap::new();
    discharge(body, &known, &truth, &mut done)
}

fn discharge(
    uop: &Arc<UOp>,
    known: &std::collections::HashSet<u64>,
    truth: &Arc<UOp>,
    done: &mut std::collections::HashMap<u64, Arc<UOp>>,
) -> Arc<UOp> {
    if known.contains(&uop.id) {
        return Arc::clone(truth);
    }
    if matches!(uop.op(), Op::Index { .. }) {
        return Arc::clone(uop);
    }
    if let Some(built) = done.get(&uop.id) {
        return Arc::clone(built);
    }
    let sources: Vec<Arc<UOp>> =
        uop.op().sources().into_iter().map(|src| discharge(&src, known, truth, done)).collect();
    let built = if sources.iter().zip(uop.op().sources()).all(|(new, old)| Arc::ptr_eq(new, &old)) {
        Arc::clone(uop)
    } else {
        uop.with_sources(sources)
    };
    done.insert(uop.id, Arc::clone(&built));
    built
}

/// Whether `uop` reads any of `ranges`, i.e. whether a reduction over them can
/// change its value.
fn moved_by_ranges(uop: &Arc<UOp>, ranges: &SmallVec<[Arc<UOp>; 4]>) -> bool {
    let range_ids: std::collections::HashSet<u64> = ranges.iter().map(|r| r.id).collect();
    uop.any_in_subtree(|node| range_ids.contains(&node.id))
}

/// REMOVE_FROM_SINK_LIKE = {Ops.NOOP, Ops.STACK, Ops.SINK}
fn is_remove_from_sink_like(u: &Arc<UOp>) -> bool {
    matches!(u.op(), Op::Noop | Op::Stack(..) | Op::Sink(..))
}

/// Phase 3 symbolic patterns (full symbolic() only, not symbolic_simple()).
///
/// General negation distribution:
/// - (-1) * (x + y) → x.neg() + y.neg()
/// - (x + y) * c → x*c + y*c for index dtype
pub fn sym_phase3_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // General negation distribution: (-1) * (x + y) → neg(x) + neg(y)
        Mul[_neg @const(nv), Add(x, y)] if nv.is_neg_one() => x.neg().add(&y.neg()),

        // (x + y) * c → x*c + y*c for weak integers (upstream sym)
        Mul[Add[x, y], c @const(_cv)] if x.dtype() == DType::WeakInt => x.mul(c).add(&y.mul(c)),

        // GROUP(x) → x: single-element GROUP is identity (upstream sym)
        Group { sources } if sources.len() == 1 => sources[0].clone(),

        // SINK/GROUP flatten: unwrap NOOP/STACK/SINK children.
        // Note: GROUP is NOT in this set — it survives to renderers which skip it.
        // REMOVE_FROM_SINK_LIKE = {Ops.UNROLL, Ops.NOOP, Ops.STACK, Ops.SINK}
        // For matching children, replace with x.src (all children). NOOP has no children → removed.
        Sink { sources } if sources.iter().any(is_remove_from_sink_like) => {
            let new_srcs: Vec<Arc<UOp>> = sources.iter().flat_map(|s| {
                if is_remove_from_sink_like(s) { s.op().sources().to_vec() } else { vec![Arc::clone(s)] }
            }).collect();
            Some(UOp::sink(new_srcs))
        },
        // GROUP also matches REMOVE_FROM_SINK_LIKE + GROUP itself
        Group { sources } if sources.iter().any(|s| is_remove_from_sink_like(s) || matches!(s.op(), Op::Group(..))) => {
            let new_srcs: Vec<Arc<UOp>> = sources.iter().flat_map(|s| {
                if is_remove_from_sink_like(s) || matches!(s.op(), Op::Group(..)) {
                    s.op().sources().to_vec()
                } else { vec![Arc::clone(s)] }
            }).collect();
            Some(UOp::group(new_srcs))
        },

        // END(NOOP) → NOOP (upstream sym)
        End { computation, .. } if matches!(computation.op(), Op::Noop) => UOp::new(Op::Noop, DType::Void),
    }
}

/// Store/load folding patterns (upstream sym).
///
/// - STORE(idx, LOAD(idx)) → NOOP (storing what was just loaded is a no-op)
/// - STORE(idx, WHERE(gate, alt, LOAD(idx))) → STORE(INDEX(buf, WHERE(gate, orig_idx, Invalid)), alt)
///   (gated store rewrite: selective overwrite becomes gated store with alternative value)
pub fn store_load_folding_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Invalid values suppress writes.
        Store { index: _index, value: invalid } if UOp::is_invalid_marker(invalid)
            => UOp::new(Op::Noop, DType::Void),

        // STORE(index, WHERE(cond, value, Invalid)) becomes a gated store.
        Store { index: Index { buffer, indices }, value: Where(cond, value, invalid) }
            if UOp::is_invalid_marker(invalid) && !indices.is_empty()
            => {
                let mut gated_indices = indices.clone();
                gated_indices[0] = gated_indices[0].valid(cond.clone());
                let index = UOp::index().buffer(buffer.clone()).indices(gated_indices).call().ok()?;
                Some(index.store(value.clone()))
            },

        // STORE(idx, LOAD(idx)) → NOOP when the INDEX nodes are ptr_eq
        Store { index, value: Load { index, .. } } => UOp::new(Op::Noop, DType::Void),

        // STORE(INDEX, WHERE(gate, alt, LOAD(INDEX))) → STORE(INDEX(buf, WHERE(gate, idx, Invalid)), alt)
        // upstream sym: converts selective overwrite into gated store.
        // When we store WHERE(gate, alt_value, load_from_same_index), the store only
        // matters where gate is true. Convert to a gated INDEX with alt as the value.
        Store { index: idx @ Index { buffer: buf, indices }, value: Where(gate, alt, Load { index: idx2, .. }) }
            if idx.id == idx2.id && !indices.is_empty()
            => {
                let original_idx = indices[0].clone();
                let invalid = UOp::invalid_marker();
                let gated_idx = UOp::try_where(gate.clone(), original_idx, invalid).ok()?;

                let mut new_indices: SmallVec<[Arc<UOp>; 4]> = indices.clone();
                new_indices[0] = gated_idx;
                let new_index = UOp::index()
                    .buffer(buf.clone())
                    .indices(new_indices)
                    .call()
                    .ok()?;

                Some(new_index.store(alt.clone()))
            },
    }
}

/// WHERE ALU combining patterns.
///
/// When both operands of a binary ALU are WHERE nodes with the same condition,
/// push the ALU inside the WHERE:
/// - ALU(WHERE(c, a, b), WHERE(c, d, e)) → WHERE(c, ALU(a, d), ALU(b, e))
pub fn where_alu_combining_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Only combine when both true branches or both false branches are const
        // Variant 1: both true branches are const
        for op in binary [Add, Mul, Sub, Max, And, Or, Xor] {
            r @ op(Where(c, a @const(_a), b), Where(c, d @const(_d), e)) => {
                let true_branch = UOp::new(Op::Binary(op, Arc::clone(a), Arc::clone(d)), r.dtype());
                let false_branch = UOp::new(Op::Binary(op, Arc::clone(b), Arc::clone(e)), r.dtype());
                UOp::try_where(Arc::clone(c), true_branch, false_branch).expect("failed to construct WHERE")
            },
        },
        // Variant 2: both false branches are const
        for op in binary [Add, Mul, Sub, Max, And, Or, Xor] {
            r @ op(Where(c, a, b @const(_b)), Where(c, d, e @const(_e))) => {
                let true_branch = UOp::new(Op::Binary(op, Arc::clone(a), Arc::clone(d)), r.dtype());
                let false_branch = UOp::new(Op::Binary(op, Arc::clone(b), Arc::clone(e)), r.dtype());
                UOp::try_where(Arc::clone(c), true_branch, false_branch).expect("failed to construct WHERE")
            },
        },

        // Variant 3: Associative Add — (y + WHERE(c,t,f)) + WHERE(c,tt,ff) → y + WHERE(c,t+tt,f+ff)
        // : handles WHERE-gates at different nesting levels in Add chains.
        // Both true branches const:
        Add(Add(y, Where(c, t @const(_t), f)), Where(c, tt @const(_tt), ff)) => {
            let true_sum = t.add(tt);
            let false_sum = f.add(ff);
            let combined = UOp::try_where(c.clone(), true_sum, false_sum).expect("failed to construct WHERE");
            y.add(&combined)
          },
        // Both false branches const:
        Add(Add(y, Where(c, t, f @const(_f))), Where(c, tt, ff @const(_ff))) => {
            let true_sum = t.add(tt);
            let false_sum = f.add(ff);
            let combined = UOp::try_where(c.clone(), true_sum, false_sum).expect("failed to construct WHERE");
            y.add(&combined)
          },
    }
}

// INDEX cleanup is handled by movement_cleanup_patterns in the canonical shaped IR.

/// The `B` with `q == B//div` and `B%div == base%div`, or `None`.
///
/// Line-for-line port of tinygrad `_quotient_base` (`uop/symbolic.py:35-45`):
///
/// ```text
/// def _quotient_base(q:UOp, base:UOp, div:int) -> UOp|None:
///   (q, s), (num, a) = q.pop_const(), base.pop_const()
///   if q.op is not Ops.FLOORDIV or q.src[1].op is not Ops.CONST: return None
///   if div > 0 and num.op is Ops.FLOORDIV and num.src[1].op is Ops.CONST and q.src[1].val == (c:=num.src[1].val)*div:
///     num, a, D = num.src[0], a*c, c*div
///   elif q.src[1].val == div: D = div
///   else: return None
///   (x, xa), (p, pa) = num.pop_const(), q.src[0].pop_const()
///   if p is not x or (t:=xa + a - pa) % D: return None
///   return base - k*div if (k:=t//D - s) else base
/// ```
///
/// Only that congruence is needed to recombine: canonicalization moves consts
/// freely, so the quotient may be merged (`(x//c + a)//div -> (x + a*c)//(c*div)`
/// for `div > 0`) and shifted (`(y + k*D)//D == y//D + k`).
fn quotient_base(q: &Arc<UOp>, base: &Arc<UOp>, div: i128) -> Option<Arc<UOp>> {
    let (q, s) = q.pop_const(BinaryOp::Add);
    let (num, a) = base.pop_const(BinaryOp::Add);
    let (s, a) = (integer_value(s)?, integer_value(a)?);
    let Op::Binary(BinaryOp::FloorDiv, q_num, q_div) = q.op() else { return None };
    let q_div = integer_value(get_const_value(q_div)?)?;

    // Merged quotient: `num == x//c` and `q_div == c*div`.
    let merged = match num.op() {
        Op::Binary(BinaryOp::FloorDiv, n_num, n_div) if div > 0 => {
            match get_const_value(n_div).and_then(integer_value) {
                Some(c) if c.checked_mul(div) == Some(q_div) => Some((n_num.clone(), a.checked_mul(c)?, q_div)),
                _ => None,
            }
        }
        _ => None,
    };
    let (num, a, d) = match merged {
        Some(merged) => merged,
        None if q_div == div && div != 0 => (num, a, div),
        None => return None,
    };

    let (x, xa) = num.pop_const(BinaryOp::Add);
    let (p, pa) = q_num.pop_const(BinaryOp::Add);
    if !Arc::ptr_eq(&p, &x) {
        return None;
    }
    let t = integer_value(xa)?.checked_add(a)?.checked_sub(integer_value(pa)?)?;
    if t % d != 0 {
        return None;
    }
    // `t % d == 0`, so the exact quotient is the floor quotient.
    let k = (t / d).checked_sub(s)?;
    if k == 0 {
        return Some(base.clone());
    }
    // `base - k*div`: upstream negates the host int, keeping the const on the right.
    let offset = i64::try_from(k.checked_mul(div)?.checked_neg()?).ok()?;
    base.try_add(&base.const_like(offset)).ok()
}

/// `(b*mul).usum(*rest)` — `mul == 1` is the identity `pop_const` seeds, and
/// upstream's redundant `x*1` is folded by `identity_and_zero_patterns` in the
/// same tier.
fn scaled_usum(b: &Arc<UOp>, mul: i128, rest: &[Arc<UOp>]) -> Option<Arc<UOp>> {
    let scaled = if mul == 1 { b.clone() } else { b.try_mul(&b.const_like(i64::try_from(mul).ok()?)).ok()? };
    rest.iter().try_fold(scaled, |sum, term| sum.try_add(term).ok())
}

/// Recombine a scaled mod with the partner carrying its quotient.
///
/// Line-for-line port of tinygrad `fold_add_divmod_recombine`
/// (`uop/symbolic.py:47-63`), registered on every `ADD` at `uop/symbolic.py:114`:
///
/// ```text
/// def fold_add_divmod_recombine(x:UOp) -> UOp|None:
///   terms = list(x.split_uop(Ops.ADD))
///   for i,u in enumerate(terms):
///     mod, mul = u.pop_const(Ops.MUL)
///     if mod.op is not Ops.FLOORMOD or mod.src[1].op is not Ops.CONST: continue
///     base, div = mod.src[0], mod.src[1].val
///     for j,v in enumerate(terms):
///       q, scale = v.pop_const(Ops.MUL)
///       if i == j or scale != div*mul: continue
///       rest = [t for k,t in enumerate(terms) if k not in (i,j)]
///       if (b:=_quotient_base(q, base, div)) is not None: return (b*mul).usum(*rest)
///       if q.op is Ops.FLOORMOD and q.src[1].op is Ops.CONST and (d:=q.src[1].val) > 0 and \
///          (b:=_quotient_base(q.src[0], base, div)) is not None:
///         return ((b % (div*d))*mul).usum(*rest)
///   return None
/// ```
///
/// A scaled mod `(base%div)*mul` recombines with a partner `q*(div*mul)`
/// carrying the quotient of a `b == base (mod div)`:
/// `q == b//div` gives `b*mul` (full recombine), `q == (b//div)%d` gives
/// `(b%(div*d))*mul` (partial recombine into a wider mod, needs `d > 0`).
///
/// Flattening the ADD chain is what lets the `x//c*c` and `x%c` partners be
/// separated by unrelated terms; upstream's `dtypes.weakint` guard is morok's
/// `exact_integer_rewrite` no-wrap proof on the concrete dtype.
fn fold_add_divmod_recombine(x: &Arc<UOp>) -> Option<Arc<UOp>> {
    let terms = x.split_uop(BinaryOp::Add);
    for (i, u) in terms.iter().enumerate() {
        let (m, mul) = u.pop_const(BinaryOp::Mul);
        let Some(mul) = integer_value(mul) else { continue };
        let Op::Binary(BinaryOp::FloorMod, base, mod_div) = m.op() else { continue };
        let Some(div) = get_const_value(mod_div).and_then(integer_value) else { continue };
        let Some(want) = div.checked_mul(mul) else { continue };

        for (j, v) in terms.iter().enumerate() {
            let (q, scale) = v.pop_const(BinaryOp::Mul);
            if i == j || integer_value(scale) != Some(want) {
                continue;
            }
            let rest: Vec<Arc<UOp>> =
                terms.iter().enumerate().filter(|(k, _)| *k != i && *k != j).map(|(_, t)| t.clone()).collect();

            if let Some(b) = quotient_base(&q, base, div) {
                return exact_integer_rewrite(x, scaled_usum(&b, mul, &rest)?);
            }
            if let Op::Binary(BinaryOp::FloorMod, q_src, q_div) = q.op()
                && let Some(d) = get_const_value(q_div).and_then(integer_value)
                && d > 0
                && let Some(b) = quotient_base(q_src, base, div)
            {
                let wide = i64::try_from(div.checked_mul(d)?).ok()?;
                let wider = b.try_mod(&b.const_like(wide)).ok()?;
                return exact_integer_rewrite(x, scaled_usum(&wider, mul, &rest)?);
            }
        }
    }
    None
}

/// Variations of `(x%c) + (x//c)*c = x` (tinygrad `uop/symbolic.py:114`).
pub fn div_mod_recombine_dsl_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        original @ Add(_, _) => fold_add_divmod_recombine(original),
    }
}

/// Long->Int narrowing patterns.
///
/// Narrows Int64 binary operations to Int32 when both operands and the result
/// fit in i32 range, reducing register pressure and enabling 32-bit ALU usage.
pub fn long_to_int_narrowing_patterns() -> &'static TypedPatternMatcher {
    use svod_ir::uop::properties::SoundVminVmaxProperty;

    fn fits_i32(uop: &Arc<UOp>) -> bool {
        let Some((vmin, vmax)) = SoundVminVmaxProperty::get(uop) else { return false };
        matches!(
            (vmin, vmax),
            (ConstValue::Int(min), ConstValue::Int(max))
                if *min >= i32::MIN as i64 && *max <= i32::MAX as i64
        )
    }

    crate::cached_patterns! {
        for op in binary [Add, Mul, Sub, FloorMod, Max, FloorDiv, And, Or, Xor, Shl, Shr] {
            result @ op(x, y)
                if x.dtype() == DType::Scalar(ScalarDType::Int64)
                && fits_i32(x) && fits_i32(y) && fits_i32(result)
                => {
                    let i32_dt = DType::Scalar(ScalarDType::Int32);
                    let i64_dt = DType::Scalar(ScalarDType::Int64);
                    let x32 = x.cast(i32_dt.clone());
                    let y32 = y.cast(i32_dt.clone());
                    let r32 = UOp::new(Op::Binary(op, x32, y32), i32_dt);
                    Some(r32.cast(i64_dt))
                },
        },

        // (index + c).cast(sints) → index.cast(sints) + c.cast(sints)
        // Distribute signed-int cast over addition with constant.
        // Enables further simplification of cast-of-index expressions.
        Cast { src: Add(x, c @const(_cv)), dtype: cast_dt }
            if x.dtype() == DType::WeakInt && cast_dt.scalar().is_some_and(|s| s.is_signed() && s.is_int())
            => x.cast(cast_dt.clone()).try_add(&c.cast(cast_dt.clone())).ok(),
    }
}
