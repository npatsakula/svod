//! Symbolic rewrite tables over the shared [`TestVars`] vocabulary.

mod devectorize_pin;
mod early_reject_pin;
mod index_lowering;

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{DType, ScalarDType};
use svod_ir::pattern::TypedPatternMatcher;
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::properties::HasWeakFloatProperty;
use svod_ir::uop::range_eval::compute_sound_vmin_vmax;
use svod_ir::{AxisId, AxisType, BinaryOp, ConstValue, Op, ReduceOp, TernaryOp, UOp, UnaryOp, ops};
use test_case::test_case;

use crate::pattern::RewriteResult;
use crate::symbolic::patterns::{
    advanced_division_dsl_patterns, cast_dsl_patterns, commutative_canonicalization, comparison_dsl_patterns,
    div_mod_recombine_dsl_patterns, division_dsl_patterns, long_to_int_narrowing_patterns, pm_remove_invalid,
    propagate_invalid, range_based_mod_div_patterns, sym_phase3_patterns, term_combining_dsl_patterns,
    vmin_vmax_collapse_patterns, weak_float_values_are_committed,
};
use crate::symbolic::valid_simplification::{parse_valid, pm_drop_and_clauses, simplify_valid, uop_given_valid};
use crate::symbolic::{pm_fold_cast_const, sym, symbolic, symbolic_simple};
use crate::test::support::prelude::*;

/// `WHERE(condition, true_val, false_val)`, the selection the tables and passes share.
#[track_caller]
fn where_(condition: &Arc<UOp>, true_val: Arc<UOp>, false_val: Arc<UOp>) -> Arc<UOp> {
    UOp::try_where(condition.clone(), true_val, false_val).expect("WHERE should build")
}

/// A degenerate Int32 range `[value, value]`, whose `vmin == vmax` collapses.
fn single(name: &str, value: i64) -> Arc<UOp> {
    UOp::var(name, DType::Int32, value, value)
}

/// A `RANGE(8, 0)` over an index extent.
fn range8() -> Arc<UOp> {
    UOp::range(UOp::index_const(8), 0)
}

/// A constant in the reduced-precision FP8 format.
fn reduced_fp8(value: f64) -> Arc<UOp> {
    UOp::const_(DType::FP8E4M3, ConstValue::Float(value))
}

/// A weak-integer constant.
fn weak_const(value: i64) -> Arc<UOp> {
    UOp::const_(DType::WeakInt, ConstValue::Int(value))
}

/// The resnet50 `r_16_32_7_7_512_3_3` scaled index, `R * 196` at the weak dtype.
fn scaled_weak_index() -> Arc<UOp> {
    UOp::new(Op::Binary(BinaryOp::Mul, UOp::range_const(512, 0), weak_const(196)), DType::WeakInt)
}

/// A weak integer straddling zero, for the negation-distribution priority row.
fn weak_x() -> Arc<UOp> {
    UOp::var("x", DType::WeakInt, -100, 100)
}

/// The `x` of the affine congruence rows: `0 <= x <= 2`.
fn qr_x() -> Arc<UOp> {
    UOp::var("qr_index", DType::WeakInt, 0, 2)
}

/// `6*x + 2`, the numerator the affine congruence rules factor.
fn qr_numerator() -> Arc<UOp> {
    qr_x().mul(&weak_const(6)).add(&weak_const(2))
}

/// Cross-check a structural row against Z3, tolerating shapes the SMT converter does not model.
#[cfg(feature = "z3")]
fn z3_check_row(input: Term, expected: Term) {
    let vars = TestVars::new();
    match crate::z3::verify_equivalence(&input(&vars), &expected(&vars)) {
        Ok(()) | Err(crate::z3::CounterExample::ConversionFailed { .. }) => {}
        Err(error) => panic!("z3 rejected a structural row: {error}"),
    }
}

#[cfg(not(feature = "z3"))]
fn z3_check_row(_input: Term, _expected: Term) {}

/// The dual-run must stay live: a converter regression cannot silently turn the cross-check above into a no-op.
#[cfg(feature = "z3")]
#[test]
fn the_z3_dual_run_still_converts_the_arithmetic_core() {
    let cores: [(Term, Term); 2] = [
        (|v: &TestVars| v.c(12).mul(&v.x).floor_div(&v.c(3)), |v| v.c(4).mul(&v.x)),
        (|v: &TestVars| v.c(6).mul(&v.x).add(&v.y).mod_(&v.c(3)), |v| v.y.mod_(&v.c(3))),
    ];

    let vars = TestVars::new();
    for (input, expected) in cores {
        if let Err(error) = crate::z3::verify_equivalence(&input(&vars), &expected(&vars)) {
            panic!("z3 could not verify a core row: {error}");
        }
    }
}

/// The scalar algebra the symbolic tiers must perform. Laws whose whole domain is
/// constant folding, self-application, or the `(a op c1) op c2` collapses live in
/// `test::property::symbolic_props`, which sweeps them over generated operands; the rows
/// here pin the rules that need one specific shape to fire at all.
#[test_case(symbolic_simple(), |v: &TestVars| v.n.mod_(&v.n), |v| v.c(0) ; "a value modulo itself")]
#[test_case(symbolic_simple(), |v| v.x.xor(&v.x), |v| v.c(0) ; "a value xored with itself")]
#[test_case(symbolic(), |v| v.c(12).mul(&v.x).floor_div(&v.c(3)), |v| v.c(4).mul(&v.x) ; "division divides an exact coefficient")]
#[test_case(symbolic(), |v| v.c(6).mul(&v.x).add(&v.y).mod_(&v.c(3)), |v| v.y.mod_(&v.c(3)) ; "modulo drops a divisible left term")]
#[test_case(symbolic(), |v| v.x.add(&v.c(9).mul(&v.y)).mod_(&v.c(3)), |v| v.x.mod_(&v.c(3)) ; "modulo drops a divisible right term")]
#[test_case(symbolic(), |v| v.c(6).mul(&v.x).add(&v.c(9).mul(&v.y)).floor_div(&v.c(3)), |v| v.c(2).mul(&v.x).add(&v.c(3).mul(&v.y)) ; "division distributes over a divisible sum")]
#[test_case(symbolic(), |v| v.c(12).mul(&v.x).sub(&v.c(6).mul(&v.y)).floor_div(&v.c(3)), |v| v.c(4).mul(&v.x).add(&v.y.mul(&v.c(-2))) ; "division distributes over a divisible difference")]
#[test_case(symbolic(), |v| v.x.sub(&v.c(3)).add(&v.c(5)), |v| v.x.add(&v.c(2)) ; "a subtraction and an addition fold together")]
#[test_case(symbolic(), |v| v.x.add(&v.c(3)).sub(&v.c(5)), |v| v.x.add(&v.c(-2)) ; "an addition and a subtraction fold together")]
#[test_case(symbolic(), |v| v.x.floor_div(&v.c(2)).add(&v.c(1)).floor_div(&v.c(2)), |v| v.x.add(&v.c(2)).floor_div(&v.c(4)) ; "a nested division absorbs the offset")]
#[test_case(symbolic(), |v| v.ic(1).add(&v.i).sub(&v.ic(1)), |v| v.i.clone() ; "an index constant cancels across a sum")]
#[test_case(symbolic(), |v| v.a.add(&v.c(2)).lt(&v.c(5)), |v| v.a.lt(&v.c(3)) ; "a comparison absorbs a constant offset")]
#[test_case(symbolic(), |v| v.a.add(&v.c(10)).lt(&v.c(5)), |v| v.a.lt(&v.c(-5)) ; "a comparison offset may go negative")]
#[test_case(symbolic(), |v| v.a.neg().lt(&v.b.neg()), |v| v.b.lt(&v.a) ; "negating both sides flips a comparison")]
#[test_case(symbolic(), |v| v.a.floor_div(&v.c(-1)), |v| v.a.mul(&v.c(-1)) ; "division by minus one is a negation")]
#[test_case(symbolic(), |v| v.x.neg().neg(), |v| v.x.clone() ; "a doubled integer negation")]
#[test_case(symbolic(), |v| v.bounded.neg().neg(), |v| v.bounded.clone() ; "a doubled float negation")]
#[test_case(symbolic(), |v| v.x.max(&v.x), |v| v.x.clone() ; "max of a value with itself")]
#[test_case(symbolic_simple(), |v| v.x.try_pow(&v.c(0)).unwrap(), |v| v.c(1) ; "an integer to the power of zero")]
#[test_case(symbolic_simple(), |v| v.x.try_pow(&v.c(1)).unwrap(), |v| v.x.clone() ; "an integer to the power of one")]
#[test_case(symbolic_simple(), |v| v.bounded.try_pow(&v.f(0.0)).unwrap(), |v| v.f(1.0) ; "a float to the power of zero")]
#[test_case(symbolic(), |v| v.bounded.lt(&v.f(2.0)), |v| v.b(true) ; "an explicitly bounded float comparison decides")]
// Term combining.
#[test_case(symbolic(), |v| v.x.add(&v.x), |v| v.x.mul(&v.c(2)) ; "a value added to itself gains a coefficient")]
#[test_case(symbolic(), |v| v.c(3).mul(&v.x).add(&v.c(5).mul(&v.x)), |v| v.x.mul(&v.c(8)) ; "coefficients of a shared factor add")]
#[test_case(symbolic(), |v| v.x.add(&v.x.mul(&v.c(3))), |v| v.x.mul(&v.c(4)) ; "a bare term counts as coefficient one")]
#[test_case(symbolic(), |v| v.y.add(&v.x).add(&v.x), |v| v.y.add(&v.x.mul(&v.c(2))) ; "terms combine across an unrelated addend")]
#[test_case(symbolic(), |v| v.c(-1).mul(&v.x.add(&v.c(3))), |v| v.x.mul(&v.c(-1)).add(&v.c(-3)) ; "negation distributes over a shifted value")]
#[test_case(sym(), |v| v.c(-1).mul(&v.a.add(&v.b)), |v| v.a.mul(&v.c(-1)).add(&v.b.mul(&v.c(-1))) ; "negation distributes over a sum of variables")]
// Booleans.
#[test_case(symbolic_simple(), |v| v.x.not().not(), |v| v.x.clone() ; "a doubled bitwise not")]
#[test_case(symbolic(), |v| v.p.or_(&v.p.not()), |v| v.b(true) ; "the excluded middle")]
#[test_case(symbolic(), |v| v.p.and_(&v.p.not()), |v| v.b(false) ; "a contradiction")]
#[test_case(symbolic(), |v| v.b(true).or_(&v.p), |v| v.b(true) ; "true absorbs a disjunction")]
#[test_case(symbolic_simple(), |v| v.b(false).and_(&v.p), |v| v.b(false) ; "false absorbs a conjunction")]
#[test_case(symbolic(), |v| v.b(true).and_(&v.p), |v| v.p.clone() ; "true is the conjunction identity")]
#[test_case(symbolic_simple(), |v| v.b(false).or_(&v.p), |v| v.p.clone() ; "false is the disjunction identity")]
#[test_case(symbolic_simple(), |v| v.p.mul(&v.q), |v| v.p.and_(&v.q) ; "a boolean product is a conjunction")]
#[test_case(symbolic_simple(), |v| v.p.add(&v.q), |v| v.p.or_(&v.q) ; "a boolean sum is a disjunction")]
#[test_case(symbolic_simple(), |v| v.p.max(&v.q), |v| v.p.or_(&v.q) ; "a boolean max is a disjunction")]
// Selection.
#[test_case(symbolic_simple(), |v| where_(&v.p, v.x.clone(), v.x.clone()), |v| v.x.clone() ; "a selection with equal branches")]
#[test_case(symbolic_simple(), |v| where_(&v.p, v.b(true), v.b(false)), |v| v.p.clone() ; "a selection that reproduces its condition")]
#[test_case(symbolic_simple(), |v| where_(&v.p, v.b(false), v.b(true)), |v| v.p.not() ; "a selection that negates its condition")]
#[test_case(symbolic_simple(), |v| where_(&v.b(true), v.x.clone(), v.y.clone()), |v| v.x.clone() ; "a selection on a true constant")]
#[test_case(symbolic_simple(), |v| where_(&v.b(false), v.x.clone(), v.y.clone()), |v| v.y.clone() ; "a selection on a false constant")]
#[test_case(symbolic(), |v| where_(&v.p.not(), v.x.clone(), v.y.clone()), |v| where_(&v.p, v.y.clone(), v.x.clone()) ; "a negated condition swaps the branches")]
#[test_case(symbolic_simple(), |v| where_(&v.p, where_(&v.q, v.x.clone(), v.y.clone()), v.y.clone()), |v| where_(&v.p.and_(&v.q), v.x.clone(), v.y.clone()) ; "nested selections with a shared false branch merge")]
#[test_case(symbolic(), |v| where_(&v.p, v.c(1), v.x.clone()).add(&where_(&v.p, v.c(2), v.y.clone())), |v| where_(&v.p, v.c(3), v.x.add(&v.y)) ; "an operation hoists through a shared condition")]
#[test_case(sym(), |v| where_(&v.p, v.c(1), v.c(0)).cast(DType::Float32), |v| where_(&v.p, v.c(1).cast(DType::Float32), v.c(0).cast(DType::Float32)) ; "a cast pushes into both branches")]
// Single-valued leaves and operations.
#[test_case(vmin_vmax_collapse_patterns(), |_v| UOp::scalar_param(0, None, DType::Int32, 5, 5), |v| v.c(5) ; "a single-valued param")]
#[test_case(vmin_vmax_collapse_patterns(), |_v| UOp::special(UOp::index_const(1), "gidx0".into()), |_v| UOp::const_(DType::WeakInt, ConstValue::Int(0)) ; "a single-valued special")]
#[test_case(vmin_vmax_collapse_patterns(), |_v| UOp::scalar_param(1, None, DType::Int32, 5, 9), |_v| UOp::scalar_param(1, None, DType::Int32, 5, 9) ; "a wide param stays")]
#[test_case(vmin_vmax_collapse_patterns(), |_v| single("single_a", 2).mul(&single("single_b", 3)), |v| v.c(6) ; "a single-valued product")]
#[test_case(vmin_vmax_collapse_patterns(), |_v| single("single_a", 2).floor_div(&single("single_b", 3)), |v| v.c(0) ; "a single-valued quotient")]
#[test_case(vmin_vmax_collapse_patterns(), |_v| single("single_a", 2).mod_(&single("single_b", 3)), |v| v.c(2) ; "a single-valued remainder")]
#[test_case(vmin_vmax_collapse_patterns(), |_v| single("single_a", 2).lt(&single("single_b", 3)), |v| v.b(true) ; "a single-valued comparison")]
#[test_case(vmin_vmax_collapse_patterns(), |_v| UOp::var("float_single", DType::Float32, 2, 2).mul(&UOp::var("float_single", DType::Float32, 2, 2)), |_v| UOp::var("float_single", DType::Float32, 2, 2).mul(&UOp::var("float_single", DType::Float32, 2, 2)) ; "a degenerate float product stays")]
// Reduced precision, weak folding and a self-reduced range.
#[test_case(symbolic_simple(), |_v| reduced_fp8(1.0).add(&reduced_fp8(0.0625)), |_v| reduced_fp8(1.0) ; "a reduced-precision sum commits before rounding")]
#[test_case(symbolic(), |_v| UOp::const_(DType::Float32, ConstValue::Float(-3.2)).eq(&UOp::const_(DType::Float32, ConstValue::Float(-3.200000047683716))), |v| v.b(true) ; "a rounded float comparison decides at the committed grid")]
#[test_case(symbolic_simple(), |_v| UOp::new(Op::Binary(BinaryOp::Add, weak_const(1), weak_const(14)), DType::WeakInt), |_v| weak_const(15) ; "weak integer operands fold at the weak dtype")]
#[test_case(symbolic_simple(), |_v| UOp::new(Op::Binary(BinaryOp::Mul, weak_const(7), weak_const(28)), DType::WeakInt), |_v| weak_const(196) ; "a weak product folds at the weak dtype")]
#[test_case(symbolic_simple(), |_v| UOp::new(Op::Binary(BinaryOp::Sub, weak_const(1), weak_const(14)), DType::WeakInt), |_v| weak_const(-13) ; "a weak difference folds at the weak dtype and may go negative")]
#[test_case(term_combining_dsl_patterns(), |_v| weak_const(-1).mul(&weak_x().add(&weak_const(5))), |_v| weak_x().mul(&weak_const(-1)).add(&weak_const(-5)) ; "the specific negation distribution outranks the general one")]
#[test_case(symbolic(), |_v| qr_numerator().mod_(&weak_const(5)), |_v| qr_x().add(&weak_const(2)) ; "the affine congruence folds a typed remainder")]
#[test_case(symbolic(), |_v| qr_numerator().floor_div(&weak_const(5)), |_v| qr_x() ; "the affine congruence folds a typed quotient")]
fn symbolic_rewrites_to_the_expected_form(matcher: &TypedPatternMatcher, input: Term, expected: Term) {
    assert_rewrites_to_and_evaluates(matcher, input, expected);
    z3_check_row(input, expected);
}

/// Rewrites whose operand is `unknown` — a LOAD, so it has no value the evaluator
/// can produce. The claim these rows make is structural only, and saying so here
/// keeps the evaluated table above honest: every row in it really is checked by
/// value at a sampled point.
#[test_case(symbolic(), |v: &TestVars| v.unknown.max(&v.unknown), |v: &TestVars| v.unknown.clone() ; "max of an unknown float with itself")]
#[test_case(symbolic_simple(), |v| v.unknown.add(&v.f(-0.0)), |v| v.unknown.clone() ; "adding negative zero is the float identity")]
fn unknown_float_rewrites_to_the_expected_form(matcher: &TypedPatternMatcher, input: Term, expected: Term) {
    assert_rewrites_to(matcher, input, expected);
}

/// Rows whose soundness rests on a `RANGE`'s bounds, which the SMT converter does
/// not model: the Z3 cross-check above cannot judge them, so they are pinned
/// structurally and by the evaluator only.
#[test_case(symbolic(), |_v| range8().mod_(&UOp::index_const(8)), |_v| range8() ; "a range modulo its own extent is the range")]
#[test_case(symbolic(), |_v| range8().floor_div(&UOp::index_const(8)), |_v| weak_const(0) ; "a range divided by its own extent is zero")]
#[test_case(symbolic(), |_v| weak_const(1).add(&weak_const(14)).add(&scaled_weak_index()).add(&weak_const(-15)), |_v| scaled_weak_index() ; "weak constants cancel across an index sum")]
fn bounded_range_rewrites_to_the_expected_form(matcher: &TypedPatternMatcher, input: Term, expected: Term) {
    assert_rewrites_to_and_evaluates(matcher, input, expected);
}

/// Folding a single-valued carry would break a hand-built kernel's trip-1 loop recurrence, so the additive ops stay binary.
#[test]
fn single_valued_add_sub_max_keep_their_carry() {
    let (a, b) = (single("single_a", 2), single("single_b", 3));
    for expression in [a.add(&b), a.sub(&b), a.max(&b)] {
        assert_op!(rewrite(vmin_vmax_collapse_patterns(), expression.clone()), Op::Binary(..));
    }
}

/// Shapes that look like a rewrite but are unsound, and must come back untouched.
#[test_case(symbolic_simple(), |v: &TestVars| v.x.and_(&v.y) ; "a conjunction of two variables")]
#[test_case(symbolic_simple(), |v| v.c(3).mul(&v.x).add(&v.c(5).mul(&v.y)) ; "terms over different variables")]
#[test_case(symbolic_simple(), |v| v.x.add(&v.y).mod_(&v.c(3)) ; "modulo of an indivisible sum")]
#[test_case(symbolic_simple(), |v| v.x.mul(&v.y) ; "an integer product is not a conjunction")]
#[test_case(symbolic_simple(), |v| v.x.cast(DType::Float32) ; "a cast of a variable is not folded")]
#[test_case(symbolic_simple(), |v| v.unknown.ne(&v.unknown) ; "an unknown float may be a NaN")]
#[test_case(symbolic_simple(), |v| v.unknown.lt(&v.unknown) ; "an unknown float compared with itself")]
#[test_case(symbolic_simple(), |v| v.unknown.mul(&v.bounded).floor_div(&v.bounded) ; "float cancellation changes rounding")]
#[test_case(symbolic(), |v| v.unknown.max(&v.f(f32::MAX as f64)) ; "an unknown float against the finite limit")]
#[test_case(symbolic(), |v| v.unknown.lt(&v.f(f32::MAX as f64)) ; "an unknown float below the finite limit")]
#[test_case(symbolic(), |v| where_(&v.unknown.lt(&v.f(f32::MAX as f64)), v.f(1.0), v.f(2.0)) ; "a selection on an unknown float comparison")]
#[test_case(symbolic_simple(), |v| v.unknown.add(&v.f(0.0)) ; "adding positive zero keeps a negative zero")]
#[test_case(symbolic(), |v| where_(&v.p, v.f(-0.0), v.f(0.0)).max(&v.f(0.0)) ; "a float max tie keeps the sign of zero")]
#[test_case(symbolic_simple(), |v| where_(&v.p, where_(&v.q, v.x.clone(), v.y.clone()), v.a.clone()) ; "nested selections with different false branches")]
#[test_case(symbolic_simple(), |v| where_(&v.p, v.x.clone(), v.y.clone()).add(&where_(&v.q, v.x.clone(), v.y.clone())) ; "selections under different conditions")]
fn symbolic_leaves_unsound_shapes_alone(matcher: &TypedPatternMatcher, input: Term) {
    assert_unchanged(matcher, input);
}

/// `x / x` folds to one for a float whose declared range excludes zero. The surviving
/// `unknown` rows only pin the *non*-rewrite, so without this the positive float case —
/// the one the rule exists for — is unpinned; the other two survivors are Int32-only.
#[test]
fn finite_nonzero_float_self_division_folds() {
    let x = UOp::var("finite_x", DType::Float32, 1, 10);
    assert_const_value(&rewrite(symbolic_simple(), x.floor_div(&x)), ConstValue::Float(1.0));
}

/// A doubled `NOT` is an involution for booleans too.
#[test]
fn a_doubled_boolean_not_is_an_involution() {
    assert_rewrites_to_and_evaluates(symbolic_simple(), |v: &TestVars| v.p.not().not(), |v| v.p.clone());
}

#[test_case(0, 8, 77, Some(true) ; "the range sits entirely below the bound")]
#[test_case(0, 8, 9, Some(true) ; "the range ends one below the bound")]
#[test_case(0, 8, 5, None ; "the bound falls inside the range")]
#[test_case(0, 8, 0, Some(false) ; "the bound is the range minimum")]
#[test_case(3, 8, 3, Some(false) ; "the bound is a shifted range minimum")]
fn comparison_with_a_constant_folds_only_when_the_range_decides(lo: i64, hi: i64, bound: i64, expect: Option<bool>) {
    let a = UOp::var("a", DType::Int32, lo, hi);
    let comparison = a.lt(&a.const_like(bound));
    let folded = rewrite(symbolic(), comparison.clone());
    match expect {
        Some(value) => assert_const!(folded, value),
        None => assert_same!(folded, comparison),
    }
}

#[test_case(BinaryOp::Lt, 0, 4, 5, 10, true ; "disjoint ranges decide less-than")]
#[test_case(BinaryOp::Lt, 5, 10, 0, 4, false ; "disjoint ranges decide the reversed less-than")]
#[test_case(BinaryOp::Eq, 0, 4, 10, 20, false ; "disjoint ranges are never equal")]
#[test_case(BinaryOp::Ne, 0, 4, 10, 20, true ; "disjoint ranges always differ")]
fn comparison_between_disjoint_ranges_folds(op: BinaryOp, a_lo: i64, a_hi: i64, b_lo: i64, b_hi: i64, expect: bool) {
    let a = UOp::var("a", DType::Int32, a_lo, a_hi);
    let b = UOp::var("b", DType::Int32, b_lo, b_hi);
    let folded = rewrite(symbolic(), UOp::alu(op, a, b));
    assert_const!(folded, expect);
}

#[test_case(DType::Int32, ConstValue::Int(42), DType::Float32, ConstValue::Float(42.0) ; "integer to float")]
#[test_case(DType::Float32, ConstValue::Float(std::f64::consts::PI), DType::Int32, ConstValue::Int(3) ; "float truncates to integer")]
#[test_case(DType::Bool, ConstValue::Bool(true), DType::Int32, ConstValue::Int(1) ; "boolean to integer")]
fn a_cast_constant_folds_at_the_target_dtype(from: DType, value: ConstValue, to: DType, expect: ConstValue) {
    let folded = rewrite(pm_fold_cast_const(), UOp::const_(from, value).cast(to.clone()));
    assert_eq!(folded.dtype(), to);
    assert_const!(folded, expect);
}

/// The twelve scalar dtypes `can_safe_cast` reasons about, in the order of its own
/// lattice — bool, then the signed widths, the unsigned widths and the floats — so the
/// matrix below reads as a table rather than as an unordered set of pairs.
const SCALARS: [svod_dtype::ScalarDType; 12] = [
    svod_dtype::ScalarDType::Bool,
    svod_dtype::ScalarDType::Int8,
    svod_dtype::ScalarDType::Int16,
    svod_dtype::ScalarDType::Int32,
    svod_dtype::ScalarDType::Int64,
    svod_dtype::ScalarDType::UInt8,
    svod_dtype::ScalarDType::UInt16,
    svod_dtype::ScalarDType::UInt32,
    svod_dtype::ScalarDType::UInt64,
    svod_dtype::ScalarDType::Float16,
    svod_dtype::ScalarDType::Float32,
    svod_dtype::ScalarDType::Float64,
];

/// `(bits, signed, float)` for the scalar dtypes a cast chain can prove lossless.
fn layout(dtype: svod_dtype::ScalarDType) -> Option<(u32, bool, bool)> {
    use svod_dtype::ScalarDType::*;
    Some(match dtype {
        Bool => (1, false, false),
        Int8 => (8, true, false),
        Int16 => (16, true, false),
        Int32 => (32, true, false),
        Int64 => (64, true, false),
        UInt8 => (8, false, false),
        UInt16 => (16, false, false),
        UInt32 => (32, false, false),
        UInt64 => (64, false, false),
        Float16 | BFloat16 => (16, true, true),
        Float32 => (32, true, true),
        Float64 => (64, true, true),
        _ => return None,
    })
}

/// Whether a container of `container`'s dtype can hold every `value` value:
fn lossless(value: svod_dtype::ScalarDType, container: svod_dtype::ScalarDType) -> bool {
    if value == container {
        return true;
    }
    let (Some((value_bits, value_signed, value_float)), Some((container_bits, container_signed, container_float))) =
        (layout(value), layout(container))
    else {
        return false;
    };
    if value_float != container_float {
        return false;
    }
    if value_float || value_signed == container_signed {
        return container_bits >= value_bits;
    }
    !value_signed && container_signed && container_bits > value_bits
}

/// `Cast(Cast(x, intermediate), outer)` collapses to `x` exactly when every value of
/// `outer` survives a round trip through `intermediate`, and never otherwise. All 144
/// ordered pairs are checked against [`lossless`], an independently written predicate, so
/// a rule that widened its own notion of "safe" is caught in both directions.
#[test]
fn cast_chains_collapse_exactly_on_the_lossless_matrix() {
    for outer in SCALARS {
        for intermediate in SCALARS {
            let x = UOp::var("matrix_x", DType::Scalar(outer), 0, 1);
            let chain = x.cast(DType::Scalar(intermediate)).cast(DType::Scalar(outer));
            let folded = rewrite(cast_dsl_patterns(), chain.clone());
            assert_eq!(
                Arc::ptr_eq(&folded, &x),
                lossless(outer, intermediate),
                "{outer:?} through {intermediate:?} collapsed to {}",
                folded.tree()
            );
        }
    }
}

/// The lattice rows outside [`cast_chains_collapse_exactly_on_the_lossless_matrix`]'s scalar set: bfloat16 and the weak integer.
#[test_case(DType::BFloat16, DType::Float32, true ; "bfloat16 survives float32")]
#[test_case(DType::Float32, DType::BFloat16, false ; "float32 does not survive bfloat16")]
#[test_case(DType::WeakInt, DType::WeakInt, true ; "a weak integer is its own container")]
#[test_case(DType::WeakInt, DType::Int64, false ; "a weak integer has no widening rule")]
fn boundary_casts_follow_the_type_lattice(value: DType, container: DType, collapses: bool) {
    let x = UOp::var("boundary_x", value.clone(), 0, 1);
    let chain = x.cast(container).cast(value);
    assert_eq!(Arc::ptr_eq(&rewrite(cast_dsl_patterns(), chain), &x), collapses);
}

/// Upstream's reciprocal distribution rules are all IEEE-inexact, so an unknown float keeps its division, power and reciprocal.
#[test]
fn unknown_float_division_power_and_reciprocal_are_not_algebraically_rewritten() {
    let value = TestVars::new().unknown;
    let square = value.try_pow(&value.const_like(2.0)).unwrap();
    let reciprocal = UOp::try_reciprocal(&value.mul(&value)).unwrap();

    assert_op!(rewrite(sym(), value.floor_div(&value)), Op::Binary(BinaryOp::Fdiv, ..));
    assert_op!(rewrite(sym(), square.clone()), Op::Binary(BinaryOp::Pow, ..));
    assert_op!(rewrite(sym(), reciprocal.clone()), Op::Unary(UnaryOp::Reciprocal, ..));
}

/// Tinygrad's associative variation of the WHERE/ALU combine (`uop/symbolic.py:207-208`)
#[test]
fn selections_under_one_condition_combine_across_an_addend() {
    let vars = TestVars::new();
    let assembled = |v: &TestVars| v.y.add(&where_(&v.p, v.c(1), v.x.clone())).add(&where_(&v.p, v.c(2), v.y.clone()));

    let combined = rewrite(symbolic(), assembled(&vars));
    assert_eq!(
        count(&combined, |node| matches!(node.op(), Op::Ternary(TernaryOp::Where, ..))),
        1,
        "{}",
        combined.tree()
    );
    assert_pass_preserves(|uop| rewrite(symbolic(), uop), assembled);
}

/// `long_to_int_narrowing_patterns` narrows an Int64 operation whose operands and result
/// all provably fit in i32 into an i32 operation wrapped in a widening CAST, so the backend
/// emits 32-bit arithmetic. Every binary op the rule lists has to be carried through.
#[test_case(BinaryOp::Add ; "add")]
#[test_case(BinaryOp::Mul ; "mul")]
#[test_case(BinaryOp::Sub ; "sub")]
#[test_case(BinaryOp::Max ; "max")]
#[test_case(BinaryOp::FloorDiv ; "floor-div")]
#[test_case(BinaryOp::FloorMod ; "floor-mod")]
fn long_to_int_narrowing_wraps_a_fitting_int64_op(op: BinaryOp) {
    let (x, y) = (UOp::var("narrow_x", DType::Int64, 0, 10), UOp::var("narrow_y", DType::Int64, 2, 3));
    let expression = UOp::new(Op::Binary(op, x.clone(), y.clone()), DType::Int64);
    let expected =
        UOp::new(Op::Binary(op, x.cast(DType::Int32), y.cast(DType::Int32)), DType::Int32).cast(DType::Int64);

    let folded = rewrite(long_to_int_narrowing_patterns(), expression);
    assert_same!(folded, expected);
}

/// The narrowing needs both operands and the result to fit in i32, and the weak distribution fires only for a signed target.
#[test]
fn long_to_int_narrowing_declines_unprovable_and_weak_shapes() {
    let y = UOp::var("narrow_y", DType::Int64, 2, 3);
    let wide = UOp::var("narrow_x", DType::Int64, 0, i64::MAX);
    let expression = UOp::new(Op::Binary(BinaryOp::Add, wide, y.clone()), DType::Int64);
    assert_same!(rewrite(long_to_int_narrowing_patterns(), expression.clone()), expression);

    let (z, w) = (UOp::var("narrow_z", DType::Int32, 0, 10), UOp::var("narrow_w", DType::Int32, 2, 3));
    let int32 = UOp::new(Op::Binary(BinaryOp::Add, z, w), DType::Int32);
    assert_same!(rewrite(long_to_int_narrowing_patterns(), int32.clone()), int32);

    let weak = UOp::var("weak_x", DType::WeakInt, 0, 100);
    let constant = UOp::const_(DType::WeakInt, ConstValue::Int(7));
    let expected = weak.cast(DType::Int32).add(&constant.cast(DType::Int32));
    assert_same!(rewrite(long_to_int_narrowing_patterns(), weak.add(&constant).cast(DType::Int32)), expected);

    let reversed = constant.add(&weak).cast(DType::Int32);
    assert_same!(rewrite(long_to_int_narrowing_patterns(), reversed.clone()), reversed);
    let float_target = weak.add(&constant).cast(DType::Float32);
    assert_same!(rewrite(long_to_int_narrowing_patterns(), float_target.clone()), float_target);
}

// --- Structural commutative ordering ---

/// The two sources of a binary root, in order. Use [`expect_binary`] wherever the op
/// itself is part of the claim — this one deliberately drops it.
#[track_caller]
fn binary_sources(root: &Arc<UOp>) -> (Arc<UOp>, Arc<UOp>) {
    let Op::Binary(_, lhs, rhs) = root.op() else { panic!("expected a binary root, got {}", root.tree()) };
    (lhs.clone(), rhs.clone())
}

/// The two sources of a binary root that must carry `op`. A rewrite that swapped `Shl` for
/// `Mul`, `Lt` for `Gt` or `Add` for `Sub` keeps the sources and would pass [`binary_sources`].
#[track_caller]
fn expect_binary(root: &Arc<UOp>, op: BinaryOp) -> (Arc<UOp>, Arc<UOp>) {
    let Op::Binary(actual, lhs, rhs) = root.op() else { panic!("expected a binary root, got {}", root.tree()) };
    assert_eq!(*actual, op, "unexpected binary op\n{}", root.tree());
    (lhs.clone(), rhs.clone())
}

#[track_caller]
fn assert_binary_sources(root: &Arc<UOp>, lhs: &Arc<UOp>, rhs: &Arc<UOp>) {
    let (actual_lhs, actual_rhs) = binary_sources(root);
    assert!(Arc::ptr_eq(&actual_lhs, lhs), "unexpected lhs: {}", root.tree());
    assert!(Arc::ptr_eq(&actual_rhs, rhs), "unexpected rhs: {}", root.tree());
}

#[test]
fn commutative_index_ops_follow_tinygrad_structural_order() {
    let end = UOp::index_const(8);
    let special = UOp::special(end.clone(), "gidx0".to_string());
    let range = UOp::range(end, 0);

    for op in [BinaryOp::Add, BinaryOp::Mul, BinaryOp::And, BinaryOp::Or, BinaryOp::Xor, BinaryOp::Max] {
        let authored = UOp::new(Op::Binary(op, special.clone(), range.clone()), DType::WeakInt);
        let reversed = UOp::new(Op::Binary(op, range.clone(), special.clone()), DType::WeakInt);
        let authored = rewrite(commutative_canonicalization(), authored);
        let reversed = rewrite(commutative_canonicalization(), reversed);

        assert_binary_sources(&authored, &special, &range);
        assert_binary_sources(&reversed, &special, &range);
        assert_same!(authored, reversed);
    }

    // Only the tiers that include the canonicalization reorder; `symbolic_simple` does not.
    let reversed = range.add(&special);
    assert_binary_sources(&rewrite(symbolic_simple(), reversed.clone()), &range, &special);
    for matcher in [symbolic(), sym()] {
        assert_binary_sources(&rewrite(matcher, reversed.clone()), &special, &range);
    }
}

#[test]
fn commutative_index_order_ranks_constants_ranges_variables_and_stacks() {
    let range0 = UOp::range_const(8, 0);
    let range1 = UOp::range_const(8, 1);
    let special = UOp::special(UOp::index_const(8), "gidx0".to_string());

    assert_binary_sources(
        &rewrite(commutative_canonicalization(), UOp::index_const(3).add(&range0)),
        &range0,
        &UOp::index_const(3),
    );
    assert_binary_sources(&rewrite(commutative_canonicalization(), range1.add(&range0)), &range0, &range1);

    let (var_a, var_b) = (UOp::define_var("a".to_string(), 0, 8), UOp::define_var("b".to_string(), 0, 8));
    assert_binary_sources(&rewrite(commutative_canonicalization(), var_b.add(&var_a)), &var_a, &var_b);

    // A nested tree is ranked by its own sources, and both authorings converge.
    let nested_first = rewrite(commutative_canonicalization(), range1.add(&range0).add(&special));
    let (actual_special, actual_nested) = binary_sources(&nested_first);
    assert!(Arc::ptr_eq(&actual_special, &special));
    assert_binary_sources(&actual_nested, &range0, &range1);
    assert_same!(nested_first, rewrite(commutative_canonicalization(), special.add(&range0.add(&range1))));

    // A VCONST is projected as tinygrad's STACK of its lanes, so it sorts last.
    let vconst = UOp::vconst(vec![ConstValue::Int(2), ConstValue::Int(1)], DType::WeakInt);
    let stack = UOp::stack(smallvec![range0.clone(), special.clone()]);
    let authored = UOp::new(Op::Binary(BinaryOp::Add, vconst.clone(), stack.clone()), vconst.dtype());
    assert_binary_sources(&rewrite(commutative_canonicalization(), authored), &stack, &vconst);
}

#[test]
fn commutative_order_declines_outside_weak_integers_and_on_ties() {
    for dtype in [DType::Index, DType::Int32, DType::Float32] {
        let lhs = UOp::const_(dtype.clone(), if dtype.is_float() { 4.0.into() } else { 4.into() });
        let rhs = UOp::const_(dtype.clone(), if dtype.is_float() { 3.0.into() } else { 3.into() });
        let authored = UOp::new(Op::Binary(BinaryOp::Add, lhs, rhs), dtype);
        assert_same!(rewrite(commutative_canonicalization(), authored.clone()), authored);
    }

    // Structurally tied and incomparable operands keep their authored order.
    let base = UOp::index_const(7);
    let (left, right) = (base.with_tag(smallvec![1]), base.with_tag(smallvec![2]));
    let tied = UOp::new(Op::Binary(BinaryOp::Add, right.clone(), left.clone()), DType::WeakInt);
    assert_binary_sources(&rewrite(commutative_canonicalization(), tied), &right, &left);

    let end = UOp::index_const(8);
    let weak = UOp::range_axis(end.clone(), AxisId::Renumbered(0), AxisType::Weak);
    let global = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Global);
    assert_binary_sources(&rewrite(commutative_canonicalization(), weak.add(&global)), &weak, &global);

    // Reordering preserves the node's tag.
    let range = UOp::range_const(8, 0);
    let special = UOp::special(UOp::index_const(8), "gidx0".to_string());
    let tagged = rewrite(commutative_canonicalization(), range.add(&special).with_tag(smallvec![7]));
    assert_binary_sources(&tagged, &special, &range);
    assert_eq!(tagged.tag().as_deref(), Some(&[7][..]));
}

// --- Reduced-precision and weak constant folding ---

#[test]
fn reduced_float_vconst_folding_commits_each_result_lane() {
    let values = UOp::vconst(vec![ConstValue::Float(1.0), ConstValue::Float(1.125)], DType::FP8E4M3);
    let increments = UOp::vconst(vec![ConstValue::Float(0.0625), ConstValue::Float(0.0625)], DType::FP8E4M3);
    let folded = rewrite(symbolic_simple(), values.add(&increments));

    let lanes = unwrap_op!(folded, Op::VConst(v) => v);
    assert_eq!(lanes.values, vec![ConstValue::Float(1.0), ConstValue::Float(1.25)]);
}

/// The value-sensitive guard runs on every pattern attempt, so it must be a memoised
/// per-node property rather than a graph walk (which was O(n^2)).
#[test_case(DType::Float32, true ; "committed float chain")]
#[test_case(DType::WeakFloat, false ; "weak float leaf")]
fn weak_float_guard_is_memoized_per_node(leaf_dtype: DType, committed: bool) {
    const DEPTH: usize = 64;
    let mut root = UOp::const_(leaf_dtype, ConstValue::Float(0.5));
    for _ in 0..DEPTH {
        root = UOp::new(Op::Unary(UnaryOp::Sqrt, root), DType::Float32);
    }

    assert_eq!(weak_float_values_are_committed(&root), committed);

    let nodes = root.toposort();
    assert_eq!(nodes.len(), DEPTH + 1);
    for node in &nodes {
        assert!(HasWeakFloatProperty::cache(node).get().is_some(), "one evaluation per node, cached in place");
    }
    assert!(std::ptr::eq(HasWeakFloatProperty::get(&root), HasWeakFloatProperty::get(&root)));
}

/// Distribution over an addition is a weak-dtype-only rule. The tier that carries it
/// depends on the addend: a constant addend folds in term combining, two variables need
/// phase three.
#[test_case(term_combining_dsl_patterns(), false ; "constant addend in term combining")]
#[test_case(sym_phase3_patterns(), true ; "variable addend in phase three")]
fn weak_multiplication_distributes_over_an_addition_in_either_order(matcher: &TypedPatternMatcher, variable: bool) {
    let weak = |value| UOp::const_(DType::WeakInt, ConstValue::Int(value));
    let x = UOp::var("x", DType::WeakInt, 0, i64::MAX);
    let addend = if variable { UOp::var("y", DType::WeakInt, 0, i64::MAX) } else { weak(5) };

    for add in [x.add(&addend), addend.add(&x)] {
        for mul in [add.mul(&weak(3)), weak(3).mul(&add)] {
            let RewriteResult::Rewritten(result) = matcher.rewrite(&mul, &mut ()) else {
                panic!("expected weak multiplication distribution for {}", mul.tree());
            };
            assert_op!(result, Op::Binary(BinaryOp::Add, ..));
        }
    }

    // The same shape at a concrete integer dtype is left alone.
    let concrete = UOp::var("c", DType::Int32, 0, i64::MAX);
    let concrete_addend = if variable { UOp::var("d", DType::Int32, 0, i64::MAX) } else { UOp::native_const(5i32) };
    let mul = concrete.add(&concrete_addend).mul(&UOp::native_const(3i32));
    assert!(matches!(matcher.rewrite(&mul, &mut ()), RewriteResult::NoMatch));
}

// --- uint64 pack/unpack cancellation (tinygrad uop/symbolic.py:170-173) ---

/// `(hi.cast(u64) << shift) | lo.cast(u64)` — the THREEFRY packing idiom.
fn packed_u64(hi: &Arc<UOp>, lo: &Arc<UOp>, shift: i64) -> Arc<UOp> {
    let amount = UOp::const_(DType::UInt64, ConstValue::Int(shift));
    hi.cast(DType::UInt64).shl(&amount).or_(&lo.cast(DType::UInt64))
}

fn u32_var(name: &str) -> Arc<UOp> {
    UOp::var(name, DType::UInt32, 0, u32::MAX as i64)
}

#[test_case(32, true; "shift of thirty two cancels")]
#[test_case(16, false; "shift of sixteen must not cancel")]
#[test_case(31, false; "shift of thirty one must not cancel")]
fn uint64_pack_halves_cancel_only_at_thirty_two(shift: i64, folds: bool) {
    let (hi, lo) = (u32_var("hi"), u32_var("lo"));
    let packed = packed_u64(&hi, &lo, shift);
    let amount = UOp::const_(DType::UInt64, ConstValue::Int(shift));

    let low = rewrite(symbolic_simple(), packed.cast(DType::UInt32));
    assert_eq!(Arc::ptr_eq(&low, &lo), folds, "low half: {}", low.tree());
    let high = rewrite(symbolic_simple(), packed.shr(&amount));
    assert_eq!(Arc::ptr_eq(&high, &hi.cast(DType::UInt64)), folds, "high half: {}", high.tree());
}

#[test]
fn uint64_pack_high_half_needs_a_narrow_low_arm() {
    // A wide low arm can carry bits into the high half, so `>> 32` is not `hi`.
    let hi = u32_var("hi");
    let wide = UOp::var("wide", DType::UInt64, 0, i64::MAX);
    let amount = UOp::const_(DType::UInt64, ConstValue::Int(32));
    let folded = rewrite(symbolic_simple(), hi.cast(DType::UInt64).shl(&amount).or_(&wide).shr(&amount));
    assert!(!Arc::ptr_eq(&folded, &hi.cast(DType::UInt64)), "must not cancel: {}", folded.tree());
}

// --- Typed division and modulo ---

/// Typed divmod must survive a wrapping numerator, a zero divisor and the integer boundaries.
#[test]
fn typed_divmod_guards_survive_wrap_and_integer_boundaries() {
    let i8_const = |value| UOp::const_(DType::Int8, ConstValue::Int(value));

    // `(100*2+1)` wraps to -55 at Int8, so the naive `100//2` / `1%3` must not fire.
    let div = i8_const(100).mul(&i8_const(2)).add(&i8_const(1)).floor_div(&i8_const(2));
    let modulo = i8_const(100).mul(&i8_const(3)).add(&i8_const(1)).mod_(&i8_const(3));
    for (expression, folded, naive) in
        [(div, ConstValue::Int(-28), ConstValue::Int(100)), (modulo, ConstValue::Int(0), ConstValue::Int(1))]
    {
        let result = rewrite(symbolic(), expression.clone());
        assert_eq!(eval_typed(&expression, &Bindings::none()), Some(folded));
        assert_eq!(eval_typed(&result, &Bindings::none()), Some(folded));
        assert!(!matches!(result.op(), Op::Const(value) if value.0 == naive), "misrewrote {}", result.tree());
    }

    let x = UOp::var("wrap_x", DType::Int8, 100, 100);
    let y = UOp::var("wrap_y", DType::Int8, 2, 2);
    assert!(matches!(division_dsl_patterns().rewrite(&x.mul(&y).floor_div(&y), &mut ()), RewriteResult::NoMatch));

    let zero = UOp::var("zero", DType::Int8, 0, 0);
    assert!(matches!(symbolic_simple().rewrite(&zero.floor_div(&zero), &mut ()), RewriteResult::NoMatch));
    assert!(matches!(symbolic_simple().rewrite(&zero.mod_(&zero), &mut ()), RewriteResult::NoMatch));

    let min = UOp::var("min", DType::Int8, i8::MIN as i64, i8::MIN as i64);
    assert!(matches!(symbolic_simple().rewrite(&min.floor_div(&i8_const(-1)), &mut ()), RewriteResult::NoMatch));

    let umax = UOp::var("umax", DType::UInt8, u8::MAX as i64, u8::MAX as i64);
    let two = UOp::const_(DType::UInt8, ConstValue::UInt(2));
    assert!(matches!(
        division_dsl_patterns().rewrite(&umax.mul(&two).floor_div(&two), &mut ()),
        RewriteResult::NoMatch
    ));
}

#[test_case(DType::Int8 ; "int8")]
#[test_case(DType::UInt8 ; "uint8")]
#[test_case(DType::WeakInt ; "weak int")]
#[test_case(DType::Index ; "index")]
#[test_case(DType::Int8.vec(4).unwrap() ; "a hardware vector")]
#[test_case(DType::UInt8.vec(4).unwrap() ; "an unsigned hardware vector")]
fn typed_division_cancellation_still_fires_when_product_is_exact(dtype: DType) {
    let x = UOp::var("safe_x", dtype.clone(), 2, 10);
    let y = UOp::var("safe_y", dtype, 2, 3);
    let expression = x.mul(&y).floor_div(&y);

    let RewriteResult::Rewritten(result) = division_dsl_patterns().rewrite(&expression, &mut ()) else {
        panic!("safe typed cancellation did not fire for {}", expression.tree());
    };
    assert_same!(result, x);
}

/// The congruence fold needs exact host arithmetic over a scalar numerator, so a wrapping
/// dtype, a hardware vector and a broadcast shape all decline — and the vector rows must
/// decline without silently dropping a term.
#[test]
fn affine_divmod_congruence_declines_wrapping_vector_and_broadcast_numerators() {
    let wrapping = UOp::var("qr_wrapping_index", DType::Int8, 20, 21);
    let numerator = wrapping.mul(&wrapping.const_like(6)).add(&wrapping.const_like(2));
    let five = UOp::const_(DType::Int8, ConstValue::Int(5));

    let vector = DType::Int8.vec(4).unwrap();
    let vector_const = |value| UOp::const_(vector.clone(), ConstValue::Int(value));
    let vx = UOp::var("vector_x", vector.clone(), 0, 1);
    let vy = UOp::var("vector_y", vector.clone(), 0, 1);
    let vector_divisor = UOp::const_(vector.clone(), ConstValue::Int(5));

    let scalar = |name| UOp::var(name, DType::Int8, 0, 1);
    let scalar_const = |value| UOp::const_(DType::Int8, ConstValue::Int(value));
    let stacked = UOp::stack(vec![scalar("shape_b0"), scalar("shape_b1")].into());
    let broadcast = scalar("shape_a")
        .mul(&scalar_const(6))
        .add(&stacked.mul(&scalar_const(5)))
        .add(&scalar("shape_d").mul(&scalar_const(11)))
        .mod_(&scalar_const(5));
    assert_eq!(broadcast.shape().unwrap().unwrap().len(), 1, "the broadcast row must stay shaped");

    for expression in [
        numerator.mod_(&five),
        numerator.floor_div(&five),
        vx.mul(&vector_const(6)).add(&vy.mul(&vector_const(2))).mod_(&vector_divisor),
        vx.mul(&vector_const(11)).add(&vy.mul(&vector_const(6))).floor_div(&vector_divisor),
        broadcast,
    ] {
        assert!(
            matches!(advanced_division_dsl_patterns().rewrite(&expression, &mut ()), RewriteResult::NoMatch),
            "congruence fired on {}",
            expression.tree()
        );
    }
}

/// The guard host arithmetic must not overflow (`i64::MAX` coefficients) or trap (`i64::MIN / -1`).
#[test]
fn divmod_guards_decline_overflowing_and_trapping_shapes() {
    let x = UOp::var("x", DType::WeakInt, 0, 1);
    let huge = weak_const(i64::MAX);
    let expression = x.mod_(&huge).mul(&huge).add(&x.floor_div(&huge).mul(&weak_const(1)));
    assert!(matches!(div_mod_recombine_dsl_patterns().rewrite(&expression, &mut ()), RewriteResult::NoMatch));

    let min = UOp::const_(DType::Int64, ConstValue::Int(i64::MIN));
    let zero = UOp::var("zero", DType::Int64, 0, 0);
    let exact = min.add(&zero).floor_div(&UOp::const_(DType::Int64, ConstValue::Int(-1)));
    assert!(matches!(advanced_division_dsl_patterns().rewrite(&exact, &mut ()), RewriteResult::NoMatch));
}

/// The tinygrad recombine ranges: a wide `x` and a narrow `y` over weak integers.
fn recombine_vars() -> TestVars {
    let mut vars = TestVars::weak();
    vars.x = UOp::var("x", DType::WeakInt, 0, 150_527);
    vars.y = UOp::var("y", DType::WeakInt, 0, 124);
    vars
}

/// Variables for the congruence rows, which need narrow `a`/`b` ranges of their own: the
/// rule only fires where the coefficients' remainders stay inside exact host arithmetic,
/// and the shared [`TestVars`] spans are far too wide for that.
fn congruence_vars(a: (i64, i64), b: (i64, i64)) -> TestVars {
    let mut vars = TestVars::weak();
    vars.a = UOp::var("a", DType::WeakInt, a.0, a.1);
    vars.b = UOp::var("b", DType::WeakInt, b.0, b.1);
    vars
}

/// Evaluate `input` and `expected` at a pinned point, failing if either does not evaluate:
/// a congruence row that silently stopped evaluating would otherwise agree vacuously.
#[track_caller]
fn assert_same_at(input: &Arc<UOp>, expected: &Arc<UOp>, point: &[(&str, i64)]) {
    let mut bindings = Bindings::none();
    for (name, value) in point {
        bindings = bindings.with(name, *value);
    }
    let (lhs, rhs) = (fold_at(input, &bindings), fold_at(expected, &bindings));
    assert!(lhs.is_some(), "input did not evaluate at {point:?}\n{}", input.tree());
    assert_eq!(lhs, rhs, "identity broken at {point:?}");
}

/// Ported from tinygrad's `test/null/test_uop_symbolic.py`: `mod + scaled quotient`
// full and partial ladders, plus the rows the recombine rule must decline.
#[test_case(|v: &TestVars| v.x.mod_(&v.c(4)).add(&v.x.floor_div(&v.c(4)).mul(&v.c(4))), |v| v.x.clone() ; "mod plus scaled quotient")]
#[test_case(|v| v.x.floor_div(&v.c(4)).mul(&v.c(4)).add(&v.x.mod_(&v.c(4))), |v| v.x.clone() ; "scaled quotient plus mod")]
#[test_case(|v| v.y.add(&v.x.mod_(&v.c(4))).add(&v.x.floor_div(&v.c(4)).mul(&v.c(4))), |v| v.x.add(&v.y) ; "trailing quotient after an unrelated term")]
#[test_case(|v| v.y.add(&v.x.floor_div(&v.c(4)).mul(&v.c(4))).add(&v.x.mod_(&v.c(4))), |v| v.x.add(&v.y) ; "trailing mod after an unrelated term")]
#[test_case(|v| v.y.add(&v.x.floor_div(&v.c(4)).mul(&v.c(8))).add(&v.x.mod_(&v.c(4)).mul(&v.c(2))), |v| v.x.mul(&v.c(2)).add(&v.y) ; "scaled pair after an unrelated term")]
#[test_case(|v| v.y.add(&v.x.mod_(&v.c(4)).mul(&v.c(2))).add(&v.x.floor_div(&v.c(4)).mul(&v.c(8))), |v| v.x.mul(&v.c(2)).add(&v.y) ; "scaled mod then quotient after an unrelated term")]
#[test_case(|v| v.x.floor_div(&v.c(2)).mod_(&v.c(4)).add(&v.x.floor_div(&v.c(8)).mul(&v.c(4))), |v| v.x.floor_div(&v.c(2)) ; "merged quotient of a divided base")]
#[test_case(|v| v.x.mul(&v.c(19)).add(&v.c(3)).mod_(&v.c(7)).add(&v.x.mul(&v.c(19)).add(&v.c(3)).floor_div(&v.c(7)).mul(&v.c(7))), |v| v.x.mul(&v.c(19)).add(&v.c(3)) ; "coefficient larger than the divisor")]
#[test_case(|v| v.x.floor_div(&v.c(3)).mod_(&v.c(224)).mul(&v.c(3)).add(&v.x.mod_(&v.c(3))).add(&v.x.floor_div(&v.c(672)).mul(&v.c(672))), |v| v.x.clone() ; "three level ladder")]
#[test_case(|v| v.x.floor_div(&v.c(11)).mod_(&v.c(7)).mul(&v.c(11)).add(&v.x.mod_(&v.c(11))).add(&v.x.floor_div(&v.c(77)).mul(&v.c(77))), |v| v.x.clone() ; "three level ladder other shape")]
#[test_case(|v| v.x.floor_div(&v.c(7)).mod_(&v.c(6)).mul(&v.c(14)).add(&v.x.floor_div(&v.c(42)).mul(&v.c(84))), |v| v.x.floor_div(&v.c(7)).mul(&v.c(14)) ; "three level ladder keeping the outer scale")]
#[test_case(|v| v.x.floor_div(&v.c(3)).add(&v.c(1)).mod_(&v.c(4)).add(&v.x.add(&v.c(3)).floor_div(&v.c(12)).mul(&v.c(4))), |v| v.x.floor_div(&v.c(3)).add(&v.c(1)) ; "offset merged quotient")]
#[test_case(|v| v.x.add(&v.c(1)).mod_(&v.c(3)).add(&v.x.add(&v.c(1)).floor_div(&v.c(3)).add(&v.c(-17)).mul(&v.c(3))), |v| v.x.add(&v.c(1)).add(&v.c(-51)) ; "shifted quotient folds the shift into the result")]
#[test_case(|v| v.x.floor_div(&v.c(8)).mul(&v.c(4)).add(&v.y).add(&v.x.floor_div(&v.c(2)).mod_(&v.c(4))), |v| v.x.floor_div(&v.c(2)).add(&v.y) ; "partners separated inside an additive sum")]
#[test_case(|v| v.x.mul(&v.c(8)).add(&v.y).floor_div(&v.c(4)).mul(&v.c(4)).add(&v.x.mul(&v.c(8)).add(&v.y).mod_(&v.c(4))), |v| v.x.mul(&v.c(8)).add(&v.y) ; "reshape index roundtrip")]
// partial recombine: q == (b//div)%d  ->  (b%(div*d))*mul
#[test_case(|v| v.x.mod_(&v.c(4)).mul(&v.c(3)).add(&v.x.floor_div(&v.c(4)).mod_(&v.c(2)).mul(&v.c(12))), |v| v.x.mod_(&v.c(8)).mul(&v.c(3)) ; "partial widening with an outer mul")]
#[test_case(|v| v.x.mod_(&v.c(4)).mul(&v.c(-2)).add(&v.x.floor_div(&v.c(4)).mod_(&v.c(2)).mul(&v.c(-8))), |v| v.x.mod_(&v.c(8)).mul(&v.c(-2)) ; "partial widening with a negative outer mul")]
#[test_case(|v| v.x.mod_(&v.c(-3)).add(&v.x.floor_div(&v.c(-3)).mod_(&v.c(5)).mul(&v.c(-3))), |v| v.x.mod_(&v.c(-15)) ; "partial widening with a negative divisor")]
#[test_case(|v| v.x.floor_div(&v.c(3)).mod_(&v.c(4)).mul(&v.c(2)).add(&v.x.floor_div(&v.c(12)).mod_(&v.c(5)).mul(&v.c(8))), |v| v.x.floor_div(&v.c(3)).mod_(&v.c(20)).mul(&v.c(2)) ; "partial widening through a merged quotient")]
#[test_case(|v| v.x.floor_div(&v.c(2)).mod_(&v.c(4)).mul(&v.c(2)).add(&v.x.mod_(&v.c(2))), |v| v.x.mod_(&v.c(8)) ; "partial widening recomposing a low order remainder")]
#[test_case(|v| v.x.floor_div(&v.c(14)).mod_(&v.c(14)).mul(&v.c(14)).add(&v.y).add(&v.x.floor_div(&v.c(196)).mod_(&v.c(512)).mul(&v.c(196))), |v| v.x.floor_div(&v.c(14)).mod_(&v.c(7168)).mul(&v.c(14)).add(&v.y) ; "padded conv ladder with a separating term")]
// declines
#[test_case(|v| v.x.mod_(&v.c(4)).add(&v.x.floor_div(&v.c(5)).mul(&v.c(4))), |v| v.x.mod_(&v.c(4)).add(&v.x.floor_div(&v.c(5)).mul(&v.c(4))) ; "declines when the two divisors differ")]
#[test_case(|v| v.x.floor_div(&v.c(3)).mod_(&v.c(224)).mul(&v.c(3)).add(&v.x.floor_div(&v.c(600)).mul(&v.c(600))), |v| v.x.floor_div(&v.c(3)).mod_(&v.c(224)).mul(&v.c(3)).add(&v.x.floor_div(&v.c(600)).mul(&v.c(600))) ; "declines when the merged divisor mismatches")]
#[test_case(|v| v.x.floor_div(&v.c(3)).mod_(&v.c(224)).mul(&v.c(3)).add(&v.x.floor_div(&v.c(672)).mul(&v.c(700))), |v| v.x.floor_div(&v.c(3)).mod_(&v.c(224)).mul(&v.c(3)).add(&v.x.floor_div(&v.c(672)).mul(&v.c(700))) ; "declines when the partner scale mismatches")]
#[test_case(|v| v.x.floor_div(&v.c(-3)).mod_(&v.c(-2)).add(&v.x.floor_div(&v.c(6)).mul(&v.c(-2))), |v| v.x.floor_div(&v.c(-3)).mod_(&v.c(-2)).add(&v.x.floor_div(&v.c(6)).mul(&v.c(-2))) ; "declines the unsound negative merged quotient")]
fn div_mod_recombine_matches_tinygrad(input: Term, expected: Term) {
    let vars = recombine_vars();
    let original = input(&vars);
    let rewritten = rewrite(div_mod_recombine_dsl_patterns(), original.clone());
    assert_same!(rewritten, expected(&vars));

    for (x, y) in [(0, 0), (1, 3), (7, 5), (223, 17), (4095, 124), (150_527, 61)] {
        assert_same_at(&original, &rewritten, &[("x", x), ("y", y)]);
    }
}

/// `fold_divmod_congruence` (`uop/divandmod.py:38-48`) carries no numerator sign guard and
/// searches both signs of every coefficient's remainder. The first two rows are tinygrad's
/// `test_floordiv_factor_nest_negative_numerator` and
/// `test_floordiv_gcd_with_remainder_negative_numerator`
/// (`test/null/test_uop_symbolic.py:573-582`); the rest need the negative representative,
/// which is only reachable through `rem_choices`.
#[test_case((-10, 10), (0, 3), |v: &TestVars| v.a.mul(&v.c(4)).add(&v.b).floor_div(&v.c(12)), |v| v.a.floor_div(&v.c(3)) ; "factor nest over a negative numerator")]
#[test_case((-1, 5), (0, 0), |v| v.a.mul(&v.c(2)).add(&v.c(7)).floor_div(&v.c(8)), |v| v.a.add(&v.c(3)).floor_div(&v.c(4)) ; "gcd with remainder over a negative numerator")]
#[test_case((2, 3), (0, 0), |v| v.a.mul(&v.c(2)).mod_(&v.c(6)), |v| v.a.mul(&v.c(-4)).add(&v.c(12)) ; "mod that needs the lone term negative remainder")]
#[test_case((2, 3), (0, 0), |v| v.a.mul(&v.c(2)).floor_div(&v.c(6)), |v| v.a.add(&v.c(-2)) ; "quotient that needs the lone term negative remainder")]
#[test_case((1, 2), (0, 1), |v| v.a.mul(&v.c(2)).add(&v.b.mul(&v.c(4))).mod_(&v.c(4)), |v| v.a.mul(&v.c(-2)).add(&v.c(4)) ; "mod that needs the tie break negative remainder")]
fn divmod_congruence_matches_tinygrad(a: (i64, i64), b: (i64, i64), input: Term, expected: Term) {
    let vars = congruence_vars(a, b);
    let original = input(&vars);
    let folded = rewrite(symbolic(), original.clone());
    let expected_uop = expected(&vars);
    assert_same!(folded, expected_uop);

    for point_a in a.0..=a.1 {
        for point_b in b.0..=b.1 {
            assert_same_at(&original, &expected_uop, &[("a", point_a), ("b", point_b)]);
        }
    }
}

/// `(x + c)//d -> (x + c%d)//d + c//d` (`uop/divandmod.py:102-105`): "split the multiple of
/// d out of the const, holds for any d != 0". `c` is split with the floor-semantics pair
/// `(c.rem_euclid(d), c.div_euclid(d))` and upstream carries no sign guard, so a negative
/// `c` and a numerator that crosses zero both fold. `None` marks the rows where upstream's
/// `c.val % d.val == c.val` declines because the const is already the reduced representative.
#[test_case(0, 224, -15, 14, Some((13, -2)) ; "negative const over the resnet conv index")]
#[test_case(0, 10, 17, 5, Some((2, 3)) ; "const larger than the divisor")]
#[test_case(-10, 10, -1, 4, Some((3, -1)) ; "numerator that crosses zero")]
#[test_case(-5, 5, -15, 7, Some((6, -3)) ; "negative const and a crossing numerator")]
#[test_case(0, 10, -9, -4, Some((-1, 2)) ; "negative divisor")]
#[test_case(0, 100, 28, 14, Some((0, 2)) ; "const that is an exact multiple of the divisor")]
#[test_case(0, 100, 3, 14, None ; "const already reduced")]
#[test_case(-10, 10, 0, 7, None ; "zero const")]
#[test_case(0, 10, -1, -4, None ; "negative const already reduced for a negative divisor")]
fn const_offset_split_matches_tinygrad(x_min: i64, x_max: i64, c: i64, d: i64, split: Option<(i64, i64)>) {
    let build = |lo, hi| {
        let x = UOp::var("split_x", DType::WeakInt, lo, hi);
        let input = x.add(&x.const_like(c)).floor_div(&x.const_like(d));
        (x, input)
    };
    let (x, input) = build(x_min, x_max);

    let Some((rem, quo)) = split else {
        assert!(
            matches!(range_based_mod_div_patterns().rewrite(&input, &mut ()), RewriteResult::NoMatch),
            "reduced const was split again: {}",
            input.tree()
        );
        return;
    };
    assert_eq!(rem + quo * d, c, "row is not a valid (r, q) split of c");

    let split_of = |x: &Arc<UOp>| x.add(&x.const_like(rem)).floor_div(&x.const_like(d)).add(&x.const_like(quo));
    let RewriteResult::Rewritten(folded) = range_based_mod_div_patterns().rewrite(&input, &mut ()) else {
        panic!("const offset split did not fire for {}", input.tree());
    };
    assert_same!(folded, split_of(&x));

    for point in x_min..=x_max {
        let (pinned_x, pinned_input) = build(point, point);
        assert_same_at(&pinned_input, &split_of(&pinned_x), &[]);
    }
}

#[test]
fn signed_floor_division_rewrites_keep_negative_cases_exact() {
    let i8_const = |value| UOp::const_(DType::Int8, ConstValue::Int(value));

    let x = UOp::var("comparison_x", DType::Int8, -1, 1);
    let RewriteResult::Rewritten(lifted) =
        comparison_dsl_patterns().rewrite(&x.floor_div(&i8_const(3)).lt(&i8_const(0)), &mut ())
    else {
        panic!("positive-divisor comparison should lift");
    };
    assert_same!(lifted, x.lt(&i8_const(0)));

    // `(-128 // -9) // -2` has a single-bucket quotient, so tinygrad's
    // cancel_divmod (`uop/divandmod.py:13`) folds it to the exact constant. The
    // unsound `(a//b)//c -> a//(b*c)` reassociation stays rejected for c < 0.
    let nested = i8_const(-128).floor_div(&i8_const(-9)).floor_div(&i8_const(-2));
    let RewriteResult::Rewritten(folded) = advanced_division_dsl_patterns().rewrite(&nested, &mut ()) else {
        panic!("single-bucket quotient should fold");
    };
    assert_eq!(eval_typed(&nested, &Bindings::none()), Some(ConstValue::Int(-7)));
    assert_const!(folded, -7);

    let recombine = i8_const(-20)
        .floor_div(&i8_const(-9))
        .mod_(&i8_const(-2))
        .add(&i8_const(-20).floor_div(&i8_const(18)).mul(&i8_const(-2)));
    assert!(matches!(div_mod_recombine_dsl_patterns().rewrite(&recombine, &mut ()), RewriteResult::NoMatch));
    assert_eq!(eval_typed(&recombine, &Bindings::none()), Some(ConstValue::Int(4)));
}

// --- INVALID propagation ---

type InnerCheck = fn(&Arc<UOp>) -> bool;

/// `propagate_invalid` pushes the operation inside the gate and re-types the gated value,
/// while the INVALID marker stays a bare Bool marker in the false branch. The operation has
/// to survive the move intact — a cast stays a cast, a comparison a comparison — because the
/// gate is what says the access never happens, not a value the operation may fold away.
#[test_case(|v: &TestVars| where_(&v.p, UOp::var("f16", DType::Float16, 0, 100), UOp::invalid_marker()).cast(DType::Float32),
    |value| matches!(value.op(), Op::Cast(..)) ; "through a cast")]
#[test_case(|v| where_(&v.p, v.bounded.clone(), UOp::invalid_marker()).lt(&v.f(1.0)),
    |value| matches!(value.op(), Op::Binary(BinaryOp::Lt, ..)) ; "through a comparison")]
#[test_case(|v| where_(&v.p, UOp::var("ix", DType::Index, 0, 100), UOp::invalid_marker()).neg(),
    |value| matches!(value.op(), Op::Binary(BinaryOp::Mul, ..)) ; "through a negation")]
fn propagate_invalid_keeps_the_gate_around_the_operation(build: Term, inner: InnerCheck) {
    let vars = TestVars::new();
    let result = rewrite(propagate_invalid(), build(&vars));

    let Op::Ternary(TernaryOp::Where, condition, value, invalid) = result.op() else {
        panic!("expected a gated result, got: {}", result.tree());
    };
    assert!(Arc::ptr_eq(condition, &vars.p));
    assert!(inner(value), "unexpected gated value: {}", result.tree());
    assert!(UOp::is_invalid_marker(invalid));
    assert_eq!(invalid.dtype(), DType::Bool, "the marker keeps its own dtype");
}

/// A bare INVALID poisons a non-comparison binary from either side, but a comparison keeps
/// it as an operand (tinygrad `uop/symbolic.py:75-77`). INVALID only reaches an operand slot
/// through source reconstruction, so the poisoned nodes are built directly rather than
/// through the promoting constructors.
#[test]
fn a_bare_invalid_operand_poisons_arithmetic_but_not_a_comparison() {
    let index = UOp::var("i", DType::Index, 0, 100);
    let marker = UOp::invalid_marker();
    let binary = |op, lhs: &Arc<UOp>, rhs: &Arc<UOp>| UOp::new(Op::Binary(op, lhs.clone(), rhs.clone()), DType::Index);

    for poisoned in [binary(BinaryOp::Sub, &index, &marker), binary(BinaryOp::Sub, &marker, &index)] {
        assert!(UOp::is_invalid_marker(&rewrite(propagate_invalid(), poisoned)));
    }
    let compared = UOp::new(Op::Binary(BinaryOp::Lt, index, marker), DType::Bool);
    let kept = rewrite(propagate_invalid(), compared);
    assert_op!(kept, Op::Binary(BinaryOp::Lt, _, _));
}

/// A reduce gate lifts out only when no reduce range can move it: then every
/// contribution is valid together or invalid together, so the gate says the same
/// thing about the sum as about each term. A gate the reduce ranges *can* move
/// selects per contribution and has to stay inside.
///
/// This is what keeps a REDUCE reading `CAST(MUL(..))` once a conv fuses into a
/// concat — the only shape `tc::matmul_operands` accepts before every
/// tensor-core opt is declined.
#[test_case(false, true ; "a gate the reduce ranges cannot move")]
#[test_case(true, false ; "a gate they can move")]
fn a_reduce_gate_lifts_out_of_the_reduce_when_no_reduce_range_moves_it(reads_k: bool, lifts: bool) {
    let k = reduce_range(16, 0);
    let gate = if reads_k {
        k.lt(&UOp::const_(DType::WeakInt, ConstValue::Int(4)))
    } else {
        global_range(64, 1).lt(&index_const(32))
    };
    let value = load(index(buffer_of(1024, ScalarDType::Float16), 0));
    let gated = where_(&gate, value, UOp::invalid_marker());

    let result = rewrite(propagate_invalid(), reduce(gated, vec![k], ReduceOp::Add));

    if lifts {
        let Op::Ternary(TernaryOp::Where, condition, reduced, invalid) = result.op() else {
            panic!("the gate should be outside the reduce, got: {}", result.tree());
        };
        assert!(Arc::ptr_eq(condition, &gate));
        assert_op!(reduced, Op::Reduce(..));
        assert!(UOp::is_invalid_marker(invalid));
    } else {
        assert_op!(result, Op::Reduce(..));
    }
}

/// `num_axes > 0` also reduces leading shaped axes, which carry no RANGE to test
/// the gate against, so the rule leaves those alone rather than guessing.
#[test]
fn a_reduce_over_shaped_axes_keeps_its_gate() {
    let gated = where_(&global_range(64, 1).lt(&index_const(32)), float_values([1.0, 2.0]), UOp::invalid_marker());
    let result = rewrite(propagate_invalid(), gated.reduce_with_num_axes(smallvec![], ReduceOp::Add, 1));

    assert_op!(result, Op::Reduce(..));
}

/// Lifting the gate makes it dominate the body, so a second copy of the same test
/// inside is redundant — and it has to go, because while it is there the body
/// still reads the gate's range and `tc::detect_matmul` reads that as the operand
/// varying along it.
///
/// Inside an INDEX the same test is not redundant: it guards an address, a WHERE
/// lowers to a select rather than a branch, so the load runs for every lane and
/// discharging that copy reads out of bounds.
#[test]
fn lifting_a_reduce_gate_discharges_it_in_the_body_but_not_in_an_address() {
    let k = reduce_range(16, 0);
    let gate = global_range(64, 1).lt(&index_const(32));
    let buffer = buffer_of(1024, ScalarDType::Float16);
    let addressed = load(index_of(buffer, where_(&gate, index_const(3), UOp::invalid_marker())));
    let guarded = where_(&gate.and_(&global_range(8, 2).lt(&index_const(4))), addressed, UOp::invalid_marker());

    let result = rewrite(propagate_invalid(), reduce(guarded, vec![k], ReduceOp::Add));

    let Op::Ternary(TernaryOp::Where, _, reduced, _) = result.op() else {
        panic!("the gate should be outside the reduce, got: {}", result.tree());
    };
    let body = &unwrap_op!(reduced, Op::Reduce(r) => r).src;
    let guards_an_address = |node: &Arc<UOp>| match node.op() {
        Op::Index(ops::Index { indices, .. }) => {
            indices.iter().any(|idx| idx.any_in_subtree(|n| Arc::ptr_eq(n, &gate)))
        }
        _ => false,
    };
    assert!(body.any_in_subtree(guards_an_address), "the address keeps its guard: {}", result.tree());
    assert!(
        !matches!(body.op(), Op::Ternary(TernaryOp::Where, ..)),
        "the value gate should be discharged: {}",
        result.tree()
    );
}

#[test]
fn remove_invalid_replaces_a_typed_lane_with_zero() {
    let one = UOp::const_(DType::Float16, ConstValue::Float(1.0));
    let result = rewrite(pm_remove_invalid(), UOp::stack(vec![UOp::invalid_marker(), one].into()));

    assert!(!result.any_in_subtree(UOp::is_invalid_marker));
    let lanes = unwrap_op!(result, Op::Stack(s) => s).sources.clone();
    assert_const!(lanes[0], 0.0);
}

// --- `c0 * x < c1` ceiling division (weak integers only) ---

/// `Some((negate_lhs, bound))` is the lifted `x < bound`, with `x` negated when the coefficient is negative. The rule is weak-integer only.
#[test_case(DType::WeakInt, 3, 10, Some((false, 4)) ; "positive coefficient")]
#[test_case(DType::WeakInt, -3, 10, Some((true, 4)) ; "negative coefficient")]
#[test_case(DType::WeakInt, -3, -10, Some((true, -3)) ; "negative coefficient and bound")]
#[test_case(DType::WeakInt, 1, 10, None ; "unit coefficient")]
#[test_case(DType::WeakInt, -1, 10, None ; "negative unit coefficient")]
#[test_case(DType::Int32, 3, 10, None ; "a concrete integer keeps the multiplication")]
fn mul_lt_lifts_only_a_nonunit_coefficient(dtype: DType, c0: i64, c1: i64, expect: Option<(bool, i64)>) {
    let x = UOp::var("x", dtype.clone(), -100, 100);
    let constant = |value| UOp::const_(dtype.clone(), ConstValue::Int(value));
    let lt = constant(c0).mul(&x).lt(&constant(c1));

    let Some((negated, bound)) = expect else {
        assert!(matches!(comparison_dsl_patterns().rewrite(&lt, &mut ()), RewriteResult::NoMatch));
        return;
    };
    let RewriteResult::Rewritten(result) = comparison_dsl_patterns().rewrite(&lt, &mut ()) else {
        panic!("expected a ceil-div comparison simplification");
    };
    let (lhs, rhs) = binary_sources(&result);
    assert_op!(result, Op::Binary(BinaryOp::Lt, ..));
    if negated {
        assert_same!(lhs, x.mul(&constant(-1)));
    } else {
        assert_same!(lhs, x);
    }
    assert_eq!(rhs.dtype(), DType::WeakInt);
    assert_const!(rhs, bound);
}

// --- Validity simplification ---

#[test]
fn lower_bound_clauses_use_the_bounds_minimum() {
    let range = UOp::range_const(20, 0);
    let begin = UOp::var("begin", DType::WeakInt, 2, 9);
    let ne_form = range.lt(&begin).ne(&UOp::native_const(true));
    let not_form = range.lt(&begin).not();

    for clause in [ne_form, not_form] {
        assert_eq!(parse_valid(&clause).map(|(_, upper, bound)| (upper, bound)), Some((false, 2)));
    }
}

/// `simplify_valid` drops a clause the accumulated gate already decides, and collapses a
/// duplicated one, without ever dropping the gate itself.
#[test]
fn simplify_valid_drops_decided_and_duplicate_clauses() {
    let always = UOp::native_const(true);
    let collapsed = simplify_valid(&always.and_(&always)).expect("duplicate clauses must collapse");
    assert_same!(collapsed, always);

    let x = UOp::range_const(20, 0);
    let tighter = x.lt(&UOp::index_const(5));
    let redundant = tighter.and_(&x.lt(&UOp::index_const(10)));
    let simplified = simplify_valid(&redundant).expect("a decided clause must be replaced by true");
    assert_same!(simplified, tighter.and_(&always));
    assert!(simplified.node_count() <= redundant.node_count(), "{}", simplified.tree());
    assert!(simplify_valid(&tighter).is_none(), "an already minimal chain must be reported unchanged");
}

/// `uop_given_valid` narrows the operand to the gate's bounds, so a comparison the gate
/// already decides folds to a constant and a conjunction keeps only its surviving clause.
/// The narrowing works by substituting bound-carrying stand-ins, and none of those may be
/// left behind in the result — a leaked `fake*` PARAM would reach codegen as a real operand.
#[test]
fn uop_given_valid_narrows_under_the_gate_and_leaks_no_fake_params() {
    let x = UOp::var("x", DType::Int32, 0, 100);
    let gate = x.lt(&UOp::native_const(10i32));
    let narrowed = x.lt(&UOp::native_const(20i32));

    let result = uop_given_valid(&gate, &narrowed, false);
    assert_const!(result, true);

    let retained =
        uop_given_valid(&gate, &x.lt(&UOp::native_const(20i32)).and_(&x.lt(&UOp::native_const(5i32))), false);
    assert_same!(retained, x.lt(&UOp::native_const(5i32)));
    assert!(!retained.toposort().iter().any(|node| {
        matches!(node.op(), Op::Param(ops::Param { arg, .. }) if arg.name.as_deref().is_some_and(|name| name.starts_with("fake")))
    }));
}

/// `if any(X not in uop.backward_slice_with_self for X,_ in candidate): continue`
/// (tinygrad/uop/symbolic.py:341) — a candidate the uop never mentions is skipped before any
/// substitution, and the uop comes back untouched.
#[test_case(true ; "candidate in the slice rewrites")]
#[test_case(false ; "candidate outside the slice is skipped")]
fn uop_given_valid_only_substitutes_candidates_in_the_slice(in_slice: bool) {
    let x = UOp::var("x", DType::Int32, 0, 100);
    let y = UOp::var("y", DType::Int32, 0, 100);
    let valid = x.lt(&UOp::native_const(10i32));
    // `x < 10` makes `x < 50` true; `y < 50` is not decided by it.
    let expression = if in_slice { &x } else { &y }.lt(&UOp::native_const(50i32));

    let result = uop_given_valid(&valid, &expression, true);

    assert_eq!(!Arc::ptr_eq(&result, &expression), in_slice, "got {}", result.tree());
}

/// An AND clause is dropped only when its ranges do not reach the gated expression;
/// dropping every clause would erase the gate.
#[test_case(1, false ; "a lone clause is left alone")]
#[test_case(2, false ; "two clauses over the same range are both relevant")]
#[test_case(2, true ; "a clause over an unrelated range is dropped")]
fn drop_and_clauses_removes_only_the_irrelevant_ones(clauses: usize, irrelevant: bool) {
    let r0 = UOp::range_const(10, 0);
    let r1 = UOp::range_const(20, 1);
    let mut condition = r0.lt(&UOp::index_const(5));
    if clauses == 2 {
        let second = if irrelevant { r1.lt(&UOp::index_const(15)) } else { r0.lt(&UOp::index_const(8)) };
        condition = condition.and_(&second);
    }
    let gated = where_(&condition, r0.add(&UOp::index_const(1)), UOp::invalid_marker());

    let result = rewrite(pm_drop_and_clauses(), gated.clone());

    assert_eq!(!Arc::ptr_eq(&result, &gated), irrelevant, "{}", result.tree());
    assert_op!(result, Op::Ternary(TernaryOp::Where, ..));
}

/// `svod_ir` has no `substitute_gated` test, so this stays here: only mapped nodes are
/// replaced, an empty map is the identity, and the `preserve_calls` variant does not
/// descend into a CALL body.
#[test]
fn substitute_gated_replaces_only_the_mapped_nodes() {
    use std::collections::HashMap;

    use svod_ir::UOpKey;

    let r0 = UOp::range_const(10, 0);
    let r1 = UOp::range_const(20, 1);
    let replacement = UOp::index_const(42);

    let map = HashMap::from([(UOpKey(r0.clone()), replacement.clone())]);
    let result = r0.add(&r1).substitute_gated(&map);
    let Op::Binary(BinaryOp::Add, lhs, rhs) = result.op() else { panic!("expected Add, got {}", result.tree()) };
    assert!(Arc::ptr_eq(lhs, &replacement) || Arc::ptr_eq(rhs, &replacement));
    assert!(Arc::ptr_eq(lhs, &r1) || Arc::ptr_eq(rhs, &r1));

    let empty: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    assert_same!(r0.substitute_gated(&empty), r0);

    let body = elementwise(&[4], AxisType::Global);
    let call = body.clone().call(smallvec![], svod_ir::CallInfo::default());
    let preserved = call.substitute_gated_preserve_calls(&map);
    assert_same!(expect_call(&preserved), body);
}

/// `compute_sound_vmin_vmax` is the oracle every value-sensitive fold consults, and it has
/// two halves: the bounds it reports must be *exact*, and it must decline on anything it
/// cannot bound. `property::ranges` only checks soundness-when-it-answers, so a version
/// that widened every bound to the dtype limits — or one that invented a bound for a LOAD —
/// would pass there and be caught only here.
#[test_case(|| UOp::native_const(42i32), Some((42, 42)) ; "a constant is its own bound")]
#[test_case(|| UOp::range_const(10, 0), Some((0, 9)) ; "a range stops one below its extent")]
#[test_case(|| UOp::range_const(10, 0).add(&UOp::index_const(5)), Some((5, 14)) ; "an offset shifts both bounds")]
#[test_case(|| UOp::index_const(3).add(&UOp::range_const(10, 0)), Some((3, 12)) ; "the offset may sit on the left")]
#[test_case(|| UOp::range_const(100, 0).cast(DType::Int32).and_(&UOp::native_const(7i32)), Some((0, 7)) ; "a constant mask bounds the result exactly")]
#[test_case(|| UOp::range_const(100, 0).cast(DType::Int32).and_(&UOp::range_const(50, 1).cast(DType::Int32)), None ; "two non-constant AND operands are unanalyzable")]
#[test_case(|| TestVars::new().unknown, None ; "a load has no bound at all")]
fn sound_vmin_vmax_reports_bounds_only_for_analyzable_nodes(build: fn() -> Arc<UOp>, expect: Option<(i64, i64)>) {
    let node = build();
    let expect = expect.map(|(lo, hi)| (ConstValue::Int(lo), ConstValue::Int(hi)));
    assert_eq!(compute_sound_vmin_vmax(&node), expect, "{}", node.tree());
}
