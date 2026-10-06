//! `pm_load_collapse`: bounding a REDUCE(Add) by the conditions its body is gated on.
//!
//! Every row drives the real entry point (`reduce_load_collapse`, which is what
//! `pm_load_collapse`'s REDUCE rule calls) and asserts the folded constant, so a
//! row cannot pass when the pattern stops firing.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{DType, DeviceSpec};
use svod_ir::{BinaryOp, ConstValue, Op, ReduceOp, UOp};
use test_case::test_case;

use super::helpers::{assert_const_float, reduce_range, rewritten};
use crate::rangeify::patterns::{
    build_reduce_load_collapse_matcher, cast_is_injective, pm_load_collapse, solve_for_range,
    try_lift_arithmetic_from_eq,
};
use crate::rangeify::reduce_load_collapse;
use crate::test::support::build::global_range;
use crate::test::support::prelude::{Bindings, fold_at};

const END: i64 = 10;

fn one() -> Arc<UOp> {
    UOp::native_const(1.0f32)
}

fn zero() -> Arc<UOp> {
    UOp::const_(DType::Float32, ConstValue::Float(0.0))
}

/// The overflow rule keys on a `WeakInt` sum, which is what an address built from
/// an `Index` load has; the buffer itself needs `Index` storage.
fn index_buffer(size: usize) -> Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, size, DType::Index)
}

/// The two gated forms a bound can take: an exclusive upper bound (`r < cut`)
/// and an inclusive lower bound (`r >= lower`).
#[derive(Clone, Copy)]
enum Bound {
    Below(i64),
    From(i64),
}

impl Bound {
    fn apply(self, range: &Arc<UOp>) -> Arc<UOp> {
        match self {
            Bound::Below(cut) => range.try_cmplt(&UOp::index_const(cut)).expect("cmplt"),
            Bound::From(lower) => range.try_cmpge(&UOp::index_const(lower)).expect("cmpge"),
        }
    }
}

/// `where(r < cut, 1, 0)` keeps the first `cut` steps; `where(r >= lower, 1, 0)`
/// keeps `end - lower`. The pass emits a full `min(max(.., 0), end)` clamp rather
/// than a bare subtraction, so the edge rows below cover the clamping.
#[test_case(Bound::Below(5), 5.0 ; "upper bound inside the extent")]
#[test_case(Bound::Below(0), 0.0 ; "upper bound at zero")]
#[test_case(Bound::Below(12), 10.0 ; "upper bound clamped to the extent")]
#[test_case(Bound::From(3), 7.0 ; "lower bound")]
#[test_case(Bound::From(0), 10.0 ; "lower bound at zero")]
#[test_case(Bound::From(12), 0.0 ; "lower bound past the extent")]
fn a_gated_reduce_folds_to_the_counted_extent(bound: Bound, expected: f32) {
    let range = reduce_range(END, 0);
    let body = UOp::try_where(bound.apply(&range), one(), zero()).expect("gate");
    let folded = reduce_load_collapse(&body, &[range]).expect("a single-range gated ADD must collapse");
    assert_const_float(&folded, expected);
}

/// Only ADD is bounded. The body reads the range, so the unparented fold (which
/// would legitimately fold a range-free MUL) cannot mask the answer.
#[test]
fn a_non_add_reduce_is_not_collapsed() {
    let range = reduce_range(END, 0);
    let body = range.cast(DType::Float32).mul(&one());
    let collapsed = reduce_load_collapse(&body, std::slice::from_ref(&range));
    assert!(collapsed.is_none(), "only ADD reduces are bounded");
    rewritten_is_no_match(&body.reduce(smallvec![range], ReduceOp::Mul));
}

/// The outer matcher must decline the same non-ADD REDUCE the entry point does.
#[track_caller]
fn rewritten_is_no_match(reduce: &Arc<UOp>) {
    assert!(
        matches!(pm_load_collapse().rewrite(reduce, &mut ()), svod_ir::RewriteResult::NoMatch),
        "only ADD reduces reach the collapse: {}",
        reduce.tree()
    );
}

/// Two nested reduces fold one extent at a time: the inner `1.0 * 5`, the outer
/// `5.0 * 10`.
#[test]
fn nested_unparented_reduces_fold_to_the_product_of_their_extents() {
    let outer = reduce_range(END, 1);
    let inner = reduce_range(5, 0);
    let body = one().reduce(smallvec![inner], ReduceOp::Add);
    let folded = reduce_load_collapse(&body, &[outer]).expect("the outer range is unparented");
    assert_const_float(&folded, 50.0);
}

/// A gated index picks one step out of the range: `sum(where(idx == r, expr, 0))`
/// becomes `expr[r := idx]`, guarded by the index being in bounds.
#[test_case(3 ; "in-bounds index")]
#[test_case(0 ; "index at the lower edge")]
#[test_case(9 ; "index at the upper edge")]
fn an_eq_gated_body_collapses_to_the_indexed_value(index: i64) {
    let range = reduce_range(END, 0);
    let idx = UOp::define_var("idx".to_string(), index, index);
    let value = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let body = UOp::try_where(idx.try_cmpeq(&range).expect("cmpeq"), value, zero()).expect("gate");
    let folded = reduce_load_collapse(&body, &[range]).expect("an EQ-gated ADD must collapse");
    assert_const_float(&folded, 3.0);
}

/// The NE twin keeps exactly the indexed step too.
#[test]
fn a_ne_gated_body_collapses_to_the_indexed_value() {
    let range = reduce_range(END, 0);
    let idx = UOp::define_var("idx".to_string(), 4, 4);
    let value = UOp::const_(DType::Float32, ConstValue::Float(2.5));
    let body = UOp::try_where(idx.try_cmpne(&range).expect("cmpne"), zero(), value).expect("gate");
    let folded = reduce_load_collapse(&body, &[range]).expect("an NE-gated ADD must collapse");
    assert_const_float(&folded, 2.5);
}

/// The compared side is hardly ever the bare range: a `gather`'s arange arrives
/// wrapped in whatever its own collapse left behind. Solving `idx == r + k` for
/// `r` — and taking the step the solved index names, not the one the comparison
/// spells — is what lets a real gather collapse at all.
#[test_case(5, 8, 3.0 ; "index above the offset")]
#[test_case(2, 2, 0.0 ; "index at the offset picks the first step")]
#[test_case(1, 9, 8.0 ; "index at the last step")]
fn an_offset_range_is_solved_for(offset: i64, index: i64, expect: f32) {
    let range = reduce_range(END, 0);
    let idx = UOp::define_var("idx".to_string(), index, index);
    let shifted = range.try_add(&UOp::index_const(offset)).expect("r + k");
    let step = range.cast(DType::Float32);
    let body = UOp::try_where(idx.try_cmpeq(&shifted).expect("cmpeq"), step, zero()).expect("gate");
    let folded = reduce_load_collapse(&body, &[range]).expect("an offset range must still collapse");
    assert_const_float(&folded, expect);
}

/// `gather` compares in the index dtype, so the range reaches the comparison
/// under a cast. Peeling it is only sound while the cast keeps the range's
/// values apart, which is what makes the narrowing one here safe: the extent
/// fits the destination many times over.
#[test]
fn a_cast_between_the_range_and_the_comparison_is_peeled() {
    let range = reduce_range(END, 0);
    let idx = UOp::define_var("idx".to_string(), 7, 7);
    let widened = range.cast(DType::Int64).cast(DType::Int32);
    let body =
        UOp::try_where(idx.cast(DType::Int32).try_cmpeq(&widened).expect("cmpeq"), range.cast(DType::Float32), zero())
            .expect("gate");
    let folded = reduce_load_collapse(&body, &[range]).expect("a cast chain must not block the collapse");
    assert_const_float(&folded, 7.0);
}

/// Orientation is decided by which side carries the range, not by which operand
/// the comparison happens to store first.
#[test]
fn the_range_may_be_either_operand() {
    let range = reduce_range(END, 0);
    let idx = UOp::define_var("idx".to_string(), 6, 6);
    let body = UOp::try_where(range.try_cmpeq(&idx).expect("cmpeq"), range.cast(DType::Float32), zero()).expect("gate");
    let folded = reduce_load_collapse(&body, &[range]).expect("either orientation must collapse");
    assert_const_float(&folded, 6.0);
}

/// Only additive wrapping is inverted here. `idx == r * 2` still names at most
/// one step, but reading it off needs a divisibility test the collapse does not
/// emit, so the reduce stays.
#[test]
fn a_scaled_range_keeps_its_reduce() {
    let range = reduce_range(END, 0);
    let idx = UOp::define_var("idx".to_string(), 6, 6);
    let scaled = range.try_mul(&UOp::index_const(2)).expect("r * 2");
    let body = UOp::try_where(idx.try_cmpeq(&scaled).expect("cmpeq"), one(), zero()).expect("gate");
    assert!(reduce_load_collapse(&body, &[range]).is_none(), "a scaled range has no additive inverse");
}

/// An index that is itself a function of the range names no single step, so the
/// gate is a bound on the reduction rather than a pick out of it.
#[test]
fn an_index_that_reaches_the_range_does_not_collapse() {
    let range = reduce_range(END, 0);
    let scaled = range.try_mul(&UOp::index_const(2)).expect("r * 2");
    let shifted = range.try_add(&UOp::index_const(3)).expect("r + 3");
    let body = UOp::try_where(shifted.try_cmpeq(&scaled).expect("cmpeq"), one(), zero()).expect("gate");
    assert!(reduce_load_collapse(&body, &[range]).is_none(), "both sides carry the range");
}

/// The gate may depend on a scalar PARAM: `sum(where(p == 1 && r < 5, 2, 0))`
/// factors the parameter out of the collapsed count. Pinning the exact tree is
/// what catches a factor that silently disappears.
#[test]
fn a_parameter_in_the_gate_is_factored_out_of_the_count() {
    let range = reduce_range(END, 0);
    let param = UOp::param(0, 1, DType::Int32, None);
    let gate = param.try_cmpeq(&UOp::native_const(1i32)).expect("cmpeq").and_(&Bound::Below(5).apply(&range));
    let body = UOp::try_where(gate, UOp::native_const(2.0f32), zero()).expect("gate");

    let folded = rewritten(&pm_load_collapse(), &body.reduce(smallvec![range], ReduceOp::Add), &mut ());
    let Op::Ternary(svod_ir::TernaryOp::Where, condition, value, otherwise) = folded.op() else {
        panic!("expected WHERE(PARAM == 1, 5.0 * 2.0, 0.0), got {}", folded.tree())
    };
    assert!(matches!(condition.op(), Op::Binary(BinaryOp::Eq, ..)), "the gate survives as a parameter test");
    assert_const_float(value, 10.0);
    assert_const_float(otherwise, 0.0);
}

/// Index-overflow protection (`Lt(Add(x, y), c)` → `Lt(x, c - y)`) exists to keep
/// a computed address out of the right-hand side: the side that carries the load
/// must stay put, and only a range-free summand may become part of the bound.
/// That is exactly what the guard selects, so the loaded operand is the one the
/// rule is for.
#[test]
fn an_index_overflow_bound_keeps_the_loaded_side_on_the_left() {
    let loaded = UOp::load().index(crate::test::support::build::index(index_buffer(16), 0)).call();
    let address = loaded.cast(DType::WeakInt);
    let offset = UOp::index_const(3);
    let condition = address.try_add(&offset).expect("weak index add").try_cmplt(&UOp::index_const(10)).expect("cmplt");

    let folded = rewritten(&pm_load_collapse(), &condition, &mut ());
    let Op::Binary(BinaryOp::Lt, moved, bound) = folded.op() else {
        panic!("expected Lt(x, c - y), got {}", folded.tree())
    };
    assert!(Arc::ptr_eq(moved, &address), "the loaded address must stay on the left");
    assert_eq!(fold_eval(bound), ConstValue::Int(7), "the bound becomes 10 - 3");
}

/// The value a constant-only subtree folds to.
#[track_caller]
fn fold_eval(uop: &Arc<UOp>) -> ConstValue {
    fold_at(uop, &Bindings::none()).unwrap_or_else(|| panic!("expected a constant, got {}", uop.tree()))
}

/// The two AST shapes a lower bound arrives in. `(r < lower).logical_not()` is what
/// a lowered PAD emits; `r >= lower` is what the tensor front end builds directly.
#[derive(Clone, Copy)]
enum LowerForm {
    NotLt,
    Ge,
}

impl LowerForm {
    fn at_least(self, range: &Arc<UOp>, lower: i64) -> Arc<UOp> {
        match self {
            LowerForm::NotLt => Bound::Below(lower).apply(range).not(),
            LowerForm::Ge => Bound::From(lower).apply(range),
        }
    }
}

/// `sum(where(lo <= r < hi, 1, 0))` is the width of the window the two bounds cut
/// out of `[0, end)`: `min(end, hi) - max(0, lo)`, floored at zero
/// (`website/docs/architecture/optimizations/range-optimization.md:110`). Both
/// clamps need their own rows, since a window that hangs off either end of the
/// extent is what a padded load looks like.
#[test_case(LowerForm::NotLt, 2, 7, 5.0 ; "not-lt window inside the extent")]
#[test_case(LowerForm::Ge, 3, 8, 5.0 ; "ge window inside the extent")]
#[test_case(LowerForm::NotLt, 0, 10, 10.0 ; "not-lt window covering the extent")]
#[test_case(LowerForm::Ge, 0, 10, 10.0 ; "ge window covering the extent")]
#[test_case(LowerForm::NotLt, -4, 14, 10.0 ; "a window hanging off both ends is clamped")]
#[test_case(LowerForm::Ge, -4, 14, 10.0 ; "the ge form clamps the same way")]
#[test_case(LowerForm::NotLt, 7, 2, 0.0 ; "lower past upper is empty")]
#[test_case(LowerForm::Ge, 12, 15, 0.0 ; "a window past the extent is empty")]
#[test_case(LowerForm::Ge, 4, 4, 0.0 ; "a zero-width window is empty")]
fn a_two_sided_gate_folds_to_the_width_of_the_window(form: LowerForm, lower: i64, upper: i64, expected: f32) {
    let range = reduce_range(END, 0);
    let condition = form.at_least(&range, lower).and_(&Bound::Below(upper).apply(&range));
    let body = UOp::try_where(condition, one(), zero()).expect("gate");
    let folded = reduce_load_collapse(&body, &[range]).expect("a two-sided gated ADD must collapse");
    assert_const_float(&folded, expected);
    assert_eq!(
        expected,
        (0..END).filter(|step| *step >= lower && *step < upper).count() as f32,
        "the brute-force count"
    );
}

/// The NE lifting rule (`(x + y) != c` → `x != (c - y)`) only exists in the
/// extended `reduce_load_collapse` matcher, so this drives that matcher directly
/// on both the lifted node and its range-dependent twin.
#[test]
fn a_range_free_ne_is_lifted_out_of_the_addition() {
    let scalar = UOp::define_var("in0".to_string(), 0, 31);
    let condition = scalar.try_add(&UOp::index_const(4)).expect("add").try_cmpne(&UOp::index_const(9)).expect("cmpne");

    let folded = rewritten(&build_reduce_load_collapse_matcher(), &condition, &mut ());
    let Op::Binary(BinaryOp::Ne, left, right) = folded.op() else { panic!("expected Ne, got {}", folded.tree()) };
    assert!(Arc::ptr_eq(left, &scalar), "the range-free scalar is isolated on the left");
    assert_eq!(fold_eval(right), ConstValue::Int(5), "the lifted bound must fold to 9 - 4 = 5");
}

/// The same rule over a range-dependent sum must keep the range on the left
/// instead of folding it into the bound.
#[test]
fn an_ne_over_a_range_keeps_the_range_on_the_left() {
    let range = reduce_range(END, 0);
    let condition = range.try_add(&UOp::index_const(4)).expect("add").try_cmpne(&UOp::index_const(9)).expect("cmpne");
    let folded = rewritten(&build_reduce_load_collapse_matcher(), &condition, &mut ());
    let Op::Binary(BinaryOp::Ne, left, _) = folded.op() else { panic!("expected Ne, got {}", folded.tree()) };
    assert!(
        left.toposort().iter().any(|node| matches!(node.op(), Op::Range(..))),
        "the range cannot be folded away: {}",
        folded.tree()
    );
}

/// `x * gate:bool.cast()` reads as a WHERE: the product is skipped where the gate
/// is false, which is the only form the collapse can count.
#[test]
fn a_multiplication_by_a_cast_bool_becomes_a_where() {
    let gate = UOp::var("gate", DType::Bool, 0, 1);
    let product = UOp::native_const(3i32).mul(&gate.cast(DType::Int32));

    let folded = reduce_load_collapse(&product, &[reduce_range(END, 0)]).expect("the product must lower");
    let Op::Binary(BinaryOp::Mul, lowered, count) = folded.op() else {
        panic!("expected the lowered WHERE scaled by the extent, got {}", folded.tree())
    };
    let Op::Ternary(svod_ir::TernaryOp::Where, condition, value, otherwise) = lowered.op() else {
        panic!("expected x * gate.cast() to lower to WHERE, got {}", lowered.tree())
    };
    assert!(Arc::ptr_eq(condition, &gate), "the gate becomes the condition");
    assert!(Arc::ptr_eq(value, &UOp::native_const(3i32)), "the product keeps its scale");
    assert_eq!(super::helpers::const_value(otherwise), ConstValue::Int(0));
    assert_eq!(super::helpers::const_value(count), ConstValue::Int(END));
}

/// A loaded index as `reduce_collapse` sees it: a variable over `[lo, hi]` in i64.
fn index_var(lo: i64, hi: i64) -> Arc<UOp> {
    UOp::variable("idx".to_string(), lo, hi, DType::Int64)
}

/// `i64(i32(r) + offset)`: an Int32 arange widened to an i64 index, the compared
/// side a hand-built mask or an embedding brings.
fn widened_arange(range: &Arc<UOp>, offset: i32) -> Arc<UOp> {
    range.cast(DType::Int32).try_add(&UOp::native_const(offset)).expect("r + k").cast(DType::Int64)
}

/// The float a folded body takes at `idx = at`.
fn folded_at(folded: &Arc<UOp>, at: i64) -> Option<f64> {
    fold_at(folded, &Bindings::at("idx", at)).and_then(|value| value.try_float())
}

/// `idx == i64(i32(r))` reads as `i32(idx) == i32(r)` only while `idx` fits i32;
/// otherwise the narrowing folds `2^32 + 3` onto step 3, which it never named.
#[test_case(i64::MIN, i64::MAX, None ; "a full-range index is not narrowed")]
#[test_case(0, 1 << 40, None ; "an index past i32 is not narrowed")]
#[test_case(-7, 1 << 20, Some(5) ; "an index that fits i32 is solved for")]
fn a_widening_cast_is_peeled_only_for_an_index_that_fits(lo: i64, hi: i64, solved_at_5: Option<i64>) {
    let range = reduce_range(END, 0);
    let compared = range.cast(DType::Int32).cast(DType::Int64);
    match (solve_for_range(&index_var(lo, hi), &compared, &range), solved_at_5) {
        (None, None) => {}
        (Some(solved), Some(want)) => {
            assert_eq!(fold_at(&solved, &Bindings::at("idx", 5i64)), Some(ConstValue::Int(want)), "{}", solved.tree())
        }
        (got, want) => panic!("expected {want:?}, got {}", got.map_or("None".to_string(), |solved| solved.tree())),
    }
}

/// The EQ lift reads `i64(i32(r) + 1) == idx` as `i32(r) == i32(idx) - 1`, which
/// holds only while `idx` fits i32.
#[test_case(i64::MIN, i64::MAX, false ; "a full-range index is not narrowed")]
#[test_case(0, 100, true ; "an index that fits i32 is lifted")]
fn the_eq_lift_never_narrows_a_wide_index(lo: i64, hi: i64, lifted: bool) {
    let range = reduce_range(END, 0);
    let condition = widened_arange(&range, 1).try_cmpeq(&index_var(lo, hi)).expect("cmpeq");
    let got = try_lift_arithmetic_from_eq(&condition);
    assert_eq!(got.is_some(), lifted, "{}", got.map_or("None".to_string(), |lifted| lifted.tree()));
}

/// The NE lift is the same reading for `Cast(r + y) != idx`.
#[test_case(i64::MIN, i64::MAX, false ; "a full-range index is not narrowed")]
#[test_case(0, 100, true ; "an index that fits i32 is lifted")]
fn the_ne_lift_never_narrows_a_wide_index(lo: i64, hi: i64, lifted: bool) {
    let range = reduce_range(END, 0);
    let condition = widened_arange(&range, 1).try_cmpne(&index_var(lo, hi)).expect("cmpne");
    let matcher = build_reduce_load_collapse_matcher();
    if lifted {
        rewritten(matcher, &condition, &mut ());
    } else {
        super::helpers::assert_no_match(matcher, &condition, &mut ());
    }
}

/// The whole collapse on both gate forms: a full-range index may keep its reduce,
/// but wherever the body folds, `2^32 + 3` and `i64::MIN` read nothing and the
/// index naming step 3 reads step 3. An index that fits must still collapse.
#[test_case(0 ; "plain arange")]
#[test_case(1 ; "offset arange")]
fn a_wide_index_never_aliases_through_the_collapse(offset: i32) {
    let range = reduce_range(END, 0);
    let compared = widened_arange(&range, offset);
    let step = range.cast(DType::Float32);
    let names_3 = 3 + i64::from(offset);
    for (lo, hi) in [(i64::MIN, i64::MAX), (-5, 100)] {
        let idx = index_var(lo, hi);
        let eq = UOp::try_where(idx.try_cmpeq(&compared).expect("cmpeq"), step.clone(), zero()).expect("gate");
        let ne = UOp::try_where(idx.try_cmpne(&compared).expect("cmpne"), zero(), step.clone()).expect("gate");
        for body in [eq, ne] {
            let Some(folded) = reduce_load_collapse(&body, std::slice::from_ref(&range)) else {
                assert_eq!(lo, i64::MIN, "an index that fits i32 must collapse:\n{}", body.tree());
                continue;
            };
            let probes: &[(i64, f64)] = if lo == i64::MIN {
                &[((1 << 32) + 3, 0.0), (i64::MIN, 0.0), (names_3, 3.0)]
            } else {
                &[(names_3, 3.0), (-1, 0.0), (50, 0.0)]
            };
            for &(at, want) in probes {
                assert_eq!(folded_at(&folded, at), Some(want), "idx = {at}:\n{}", folded.tree());
            }
        }
    }
}

/// `Index` is as wide as the i64 it lowers into, so neither direction of a cast
/// that touches it can pass on its placeholder `0..=0` bounds.
#[test]
fn cast_injectivity_reads_index_as_i64() {
    assert!(cast_is_injective(&index_var(i64::MIN, i64::MAX), &DType::Index), "i64 fits Index");
    assert!(!cast_is_injective(&global_range(1 << 40, 0), &DType::Int32), "an Index range past i32 does not fit");
    assert!(cast_is_injective(&global_range(10, 0), &DType::Int32), "an Index range of ten fits");
    assert!(!cast_is_injective(&UOp::var("gate", DType::Bool, 0, 1), &DType::Int32), "bool is not an integer");
}
