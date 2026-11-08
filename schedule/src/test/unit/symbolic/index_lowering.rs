//! Weak-dtype lowering: every weak node must commit to a concrete width before codegen,
//! and the commit must be driven by the consumer, not by the weak node itself.

use std::sync::Arc;

use svod_dtype::{AddrSpace, DType};
use svod_ir::{BinaryOp, ConstValue, ConstValueHash, Op, ParamArg, TernaryOp, UOp, ops};
use test_case::test_case;

use crate::rewrite::graph_rewrite;
use crate::symbolic::index_lowering::{
    WeakMemo, commit_weak_srcs, pm_cast_weak, pm_commit_weak, pm_lower_index_dtype, pm_lower_weak, select_dtype,
};
use crate::test::support::prelude::*;

use super::expect_binary;

fn lower_weak(graph: Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(&pm_lower_weak(), graph, &mut ())
}

fn lower_index(graph: Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(&pm_lower_index_dtype(), graph, &mut WeakMemo::default())
}

#[track_caller]
fn assert_no_weak(graph: &Arc<UOp>) {
    assert!(graph.toposort().iter().all(|node| !node.dtype().is_weak()), "weak dtype survived:\n{}", graph.tree());
}

fn weak_int(value: i64) -> Arc<UOp> {
    UOp::const_(DType::WeakInt, ConstValue::Int(value))
}

/// `SHRINK(src, offsets, sizes)`; `UOp::shrink` is crate-private to `svod_ir`.
fn shrink(src: Arc<UOp>, offsets: Arc<UOp>, sizes: Arc<UOp>) -> Arc<UOp> {
    UOp::new(Op::Shrink(ops::Shrink { src: src.clone(), offsets, sizes }), src.dtype())
}

/// A weak PARAM with the shape and bounds `pm_lower_weak` branches on.
fn weak_param(addrspace: Option<AddrSpace>, min: i64, max: i64) -> Arc<UOp> {
    let shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![svod_ir::SInt::Const(1)]);
    let arg = ParamArg {
        slot: 0,
        dtype: DType::WeakInt,
        vmin_vmax: Some((ConstValueHash(ConstValue::Int(min)), ConstValueHash(ConstValue::Int(max)))),
        multiple_of: None,
        name: None,
        addrspace,
        axis: None,
        device: None,
        volatile: false,
        readonly: false,
    };
    UOp::new(Op::Param(ops::Param { shape, arg: arg.into() }), DType::WeakInt)
}

/// A weak leaf is wrapped in a CAST back to its weak dtype; the CAST source carries the
/// committed width, chosen mechanically from the value.
#[test_case(weak_int(42), DType::WeakInt, DType::Int32; "int fits the default width")]
#[test_case(weak_int(i64::MAX / 2), DType::WeakInt, DType::Int64; "int needs the wide default")]
#[test_case(UOp::const_(DType::WeakFloat, ConstValue::Float(1.5)), DType::WeakFloat, DType::Float32; "float default")]
#[test_case(UOp::native_const(7i32).cast(DType::WeakInt).cast(DType::WeakFloat), DType::WeakFloat, DType::Float32;
    "stacked weak casts resolve at the outer default")]
fn weak_leaf_commits_to_the_default_width(weak: Arc<UOp>, expected_weak: DType, expected_source: DType) {
    let lowered = lower_weak(weak);
    let view = unwrap_op!(lowered, Op::Cast(c) => c);
    assert_eq!(view.dtype, expected_weak);
    assert_eq!(view.src.dtype(), expected_source);
}

#[test_case(DType::WeakInt, vec![ConstValue::Int(1); 4], DType::Int32; "int lanes")]
#[test_case(DType::WeakFloat, vec![ConstValue::Float(1.5); 8], DType::Float32; "float lanes")]
fn weak_vconst_commits_lanewise_to_the_default_width(dtype: DType, values: Vec<ConstValue>, expected: DType) {
    let lanes = values.len();
    let lowered = lower_weak(UOp::vconst(values, dtype.clone()));
    let view = unwrap_op!(lowered, Op::Cast(c) => c);
    assert_eq!(view.dtype, dtype.vec(lanes).unwrap());
    assert_eq!(view.src.dtype(), expected.vec(lanes).unwrap());
}

/// The commit rounds each lane to the committed width before any consumer can read it —
/// an `f64` midpoint that `f32` cannot represent must already read as exactly `1.0` — and
/// an Invalid lane passes through untouched so the gater still sees it.
#[test]
fn weak_vconst_commit_rounds_lanes_and_keeps_invalid() {
    let midpoint = 1.0 + 2f64.powi(-24);
    let weak =
        UOp::vconst(vec![ConstValue::Float(midpoint), ConstValue::Invalid, ConstValue::Float(2.0)], DType::WeakFloat);

    let lowered = lower_index(UOp::sink(vec![weak]));

    let sources = expect_sink(&lowered);
    assert_eq!(sources[0].dtype(), DType::Float32.vec(3).unwrap());
    let lanes = unwrap_op!(sources[0], Op::VConst(v) => v);
    assert_eq!(lanes.values, vec![ConstValue::Float(1.0), ConstValue::Invalid, ConstValue::Float(2.0)]);
}

/// `select_dtype` picks the narrowest concrete width that holds the node's sound value range.
#[test_case(weak_int(42), DType::Int32 ; "an int that fits")]
#[test_case(weak_int(-1), DType::Int32 ; "a negative int")]
#[test_case(weak_int(i32::MAX as i64), DType::Int32 ; "the inclusive int boundary")]
#[test_case(weak_int(i32::MAX as i64 + 1), DType::Int64 ; "one past the int boundary")]
#[test_case(UOp::const_(DType::WeakInt, ConstValue::UInt(7)), DType::Int32 ; "an unsigned value that fits")]
#[test_case(UOp::native_const(7u32), DType::Int32 ; "an unsigned constant that fits")]
#[test_case(UOp::native_const(u64::MAX), DType::Int64 ; "an unsigned value that needs 64 bits")]
#[test_case(UOp::const_(DType::WeakFloat, ConstValue::Float(1.5)), DType::Float32 ; "a weak float")]
#[test_case(UOp::vconst(vec![ConstValue::Int(1); 4], DType::WeakInt), DType::Int32.vec(4).unwrap() ; "a weak vector keeps its lane count")]
#[test_case(weak_int(1).lt(&weak_int(2)), DType::Int32 ; "a bool comparison's sound range")]
fn select_dtype_picks_the_narrowest_holding_dtype(node: Arc<UOp>, expect: DType) {
    assert_eq!(select_dtype(&node), expect);
}

/// A weak source under a concrete consumer commits to the consumer's dtype. A bare weak
/// constant is rewritten in place; a `shape_to_uop` extent is a shaped node and keeps a
/// CAST, but neither may leave anything weak behind.
#[test_case(weak_int(1), DType::Int64, true; "weak constant")]
#[test_case(svod_ir::shape::shape_to_uop(&[2usize.into(), 3usize.into()].into_iter().collect()), DType::Int32, false;
    "shape extent")]
fn weak_operand_commits_to_its_concrete_consumer(weak: Arc<UOp>, target: DType, commits_in_place: bool) {
    assert!(weak.dtype().is_weak());
    let concrete = UOp::variable("idx".into(), 0, 31, target.clone());

    let lowered = lower_index(UOp::new(Op::Binary(BinaryOp::Add, concrete, weak), target.clone()));

    let (lhs, rhs) = expect_binary(&lowered, BinaryOp::Add);
    assert_eq!(lowered.dtype(), target);
    assert_eq!(lhs.dtype(), target);
    assert_eq!(rhs.dtype(), target);
    if commits_in_place {
        assert_op!(rhs, Op::Const(_));
    }
    assert_no_weak(&lowered);
}

/// `commit_weak_srcs` joins mixed sources at the strongest width and leaves an
/// all-weak join to `pm_lower_weak`, which is the tier that owns the default widths.
#[test]
fn commit_weak_srcs_joins_mixed_sources() {
    let mixed = UOp::new(Op::Binary(BinaryOp::Add, weak_int(1), UOp::native_const(5i64)), DType::Int64);
    let committed = commit_weak_srcs(&mixed).expect("a weak source under a strong consumer commits");
    assert!(committed.op().sources().iter().all(|source| source.dtype() == DType::Int64));
    assert_no_weak(&committed);

    let weak_float = UOp::const_(DType::WeakFloat, ConstValue::Float(1.5));
    let all_weak = UOp::new(Op::Binary(BinaryOp::Add, weak_int(1), weak_float), DType::WeakFloat);
    assert!(commit_weak_srcs(&all_weak).is_none(), "a weak join must be left to pm_lower_weak");
}

/// A concrete CAST is a floor: the operands commit to whatever width the value needs, while a weak target is left to the weak tier.
#[test_case(lower_index as fn(Arc<UOp>) -> Arc<UOp> ; "through the index-dtype tier")]
#[test_case(|graph: Arc<UOp>| graph_rewrite(&pm_cast_weak(), graph, &mut ()) ; "through pm_cast_weak alone")]
fn a_concrete_cast_is_a_width_floor(lower: fn(Arc<UOp>) -> Arc<UOp>) {
    let add = UOp::new(Op::Binary(BinaryOp::Add, weak_int(i32::MAX as i64 + 1), weak_int(1)), DType::WeakInt);
    let lowered = lower(add.cast(DType::Int32));

    let view = unwrap_op!(lowered, Op::Cast(c) => c);
    assert_eq!(view.dtype, DType::Int32);
    assert!(view.src.op().sources().iter().all(|source| source.dtype() == DType::Int64));
    assert_no_weak(&lowered);

    // `pm_cast_weak` is rooted at a CAST over a weak ALU node, and `cast_weak_srcs` then
    // declines a *weak* cast target. Reaching that guard needs the CAST to exist at all:
    // `UOp::cast` short-circuits on an identical dtype, so `add.cast(WeakInt)` *is* `add` —
    // a Binary the rule can never match, leaving the guard unexercised. A different weak
    // target keeps the node a CAST and puts the guard on the path.
    let weak_cast = add.cast(DType::WeakFloat);
    assert_op!(weak_cast, Op::Cast(..));
    assert!(weak_cast.dtype().is_weak() && add.dtype().is_weak(), "the guard needs a weak target over a weak source");
    assert_same!(graph_rewrite(&pm_cast_weak(), weak_cast.clone(), &mut ()), weak_cast);
}

/// A comparison has no weak result to drive the commit, so the node itself is lowered; a
/// shift takes its result dtype from its committed left operand. The op is asserted along
/// with the widths: lowering must not turn an `Shl` into a `Mul` or an `Lt` into a `Gt`.
#[test_case(|| UOp::new(Op::Binary(BinaryOp::Lt, weak_int(i32::MAX as i64 + 1), weak_int(1)), DType::Bool), BinaryOp::Lt, DType::Bool ; "a comparison unifies its operand widths")]
#[test_case(|| UOp::new(Op::Binary(BinaryOp::Shl, weak_int(1), UOp::native_const(2i64)), DType::WeakInt), BinaryOp::Shl, DType::Int64 ; "a shift re-derives its result dtype")]
fn lowering_a_weak_binary_unifies_operand_widths(build: fn() -> Arc<UOp>, expect_op: BinaryOp, expect_dtype: DType) {
    let lowered = lower_index(build());

    let (lhs, rhs) = expect_binary(&lowered, expect_op);
    assert_eq!(lhs.dtype(), DType::Int64);
    assert_eq!(rhs.dtype(), DType::Int64);
    assert_eq!(lowered.dtype(), expect_dtype);
}

#[test_case(DType::Int8; "i8")]
#[test_case(DType::UInt8; "u8")]
#[test_case(DType::Int16; "i16")]
#[test_case(DType::UInt16; "u16")]
#[test_case(DType::Int32; "i32")]
#[test_case(DType::UInt32; "u32")]
#[test_case(DType::Int64; "i64")]
#[test_case(DType::UInt64; "u64")]
fn weak_shift_counts_commit_to_the_integer_lhs_width(dtype: DType) {
    let value = if dtype.is_unsigned() { ConstValue::UInt(8) } else { ConstValue::Int(8) };
    let lhs = UOp::const_(dtype.clone(), value);

    for op in [BinaryOp::Shl, BinaryOp::Shr] {
        let shift = UOp::new(Op::Binary(op, lhs.clone(), UOp::index_const(1)), dtype.clone());
        let lowered = graph_rewrite(&pm_commit_weak(), shift, &mut ());

        let (actual_lhs, actual_rhs) = expect_binary(&lowered, op);
        assert_eq!(lowered.dtype(), dtype);
        assert_eq!(actual_lhs.dtype(), dtype);
        assert_eq!(actual_rhs.dtype(), dtype);
        assert_no_weak(&lowered);
    }
}

/// A weak INDEX commits its buffer and each weak offset to the dtype `select_dtype`
/// picks for that node alone, so the lane source and the offset may land on different widths.
#[test]
fn a_weak_index_commits_its_lane_source_and_offsets() {
    let lanes = UOp::vconst(vec![ConstValue::Int(7); 4], DType::WeakInt);
    let index = index_of(lanes, weak_int(2));
    assert!(index.dtype().is_weak(), "a weak lane source gives the INDEX a weak dtype");

    let lowered = lower_index(UOp::sink(vec![index]));

    let sources = expect_sink(&lowered);
    let (buffer, indices) = expect_index(&sources[0]);
    // The lane source's own value range is unprovable through the extraction, so the
    // buffer takes the wide default while the constant offset commits to i32.
    assert_eq!(buffer.dtype(), DType::Int64.vec(4).unwrap());
    assert_eq!(indices[0].dtype(), DType::Int32);
    assert_no_weak(&lowered);
}

fn weak_bitwise_index(combine: fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>) -> Arc<UOp> {
    let expression = combine(&UOp::index_const(12), &UOp::index_const(3));
    assert_eq!(expression.dtype(), DType::WeakInt, "graph construction must preserve mathematical integers");
    index_of(param(0, 16, DType::Float32), expression)
}

/// A lane read out of a hardware vector of weak constants.
fn weak_lane_extraction() -> Arc<UOp> {
    let lanes = UOp::vconst((0..4).map(ConstValue::Int).collect(), DType::WeakInt);
    UOp::sink(vec![index_of(lanes, UOp::index_const(2))])
}

/// A shaped vector (STACK) added to a hardware vector (VCONST) of weak constants.
fn mixed_vector_add() -> Arc<UOp> {
    let shaped = UOp::stack((0..8).map(|value| UOp::native_const(value as i64)).collect());
    let hardware = UOp::vconst(vec![ConstValue::Int(1); 8], DType::WeakInt);
    UOp::sink(vec![UOp::new(Op::Binary(BinaryOp::Add, shaped, hardware), DType::WeakInt.vec(8).unwrap())])
}

#[test_case(weak_bitwise_index(|value, operand| value.try_shr_op(operand).unwrap()); "shifted index")]
#[test_case(weak_bitwise_index(|value, operand| value.try_and_op(operand).unwrap()); "masked index")]
#[test_case(weak_bitwise_index(|value, operand| value.try_xor_op(operand).unwrap()); "xored index")]
#[test_case(weak_lane_extraction(); "lane extracted from a hardware vector")]
#[test_case(mixed_vector_add(); "shaped vector added to a hardware vector")]
fn no_weak_dtype_reaches_the_program_boundary(graph: Arc<UOp>) {
    assert_no_weak(&lower_index(graph));
}

/// A weak PARAM that is an ALU value is lowered; one that addresses memory keeps its weak
/// dtype, because the buffer element type is what decides its width later.
#[test]
fn only_the_alu_weak_param_is_lowered() {
    let alu = lower_weak(weak_param(None, 0, 7));
    let view = unwrap_op!(alu, Op::Cast(c) => c);
    assert_eq!(view.dtype, DType::WeakInt);
    assert!(!view.src.dtype().is_weak());

    let buffer = lower_weak(weak_param(Some(AddrSpace::Global), 0, 7));
    let view = unwrap_op!(buffer, Op::Param(p) => p);
    assert_eq!(view.arg.dtype, DType::WeakInt);
}

/// Extracting one element of a shaped LOAD must not collapse the LOAD to a scalar: the
/// shaped LOAD stays under the extracting INDEX so the whole vector is still read.
#[test]
fn lowering_a_weak_index_preserves_the_shaped_load_under_extraction() {
    let offsets = UOp::stack((0..8).map(weak_int).collect());
    let load = UOp::load().index(index_of(param(0, 64, DType::BFloat16), offsets)).call();
    let lane = index_of(load, UOp::index_const(3));

    let matcher = crate::symbolic::patterns::symbolic_simple().with_context::<WeakMemo>() + pm_lower_index_dtype();
    let lowered = graph_rewrite(&matcher, lane, &mut WeakMemo::default());

    let shaped_load =
        first_op(&lowered, |op| matches!(op, Op::Load(..))).expect("shaped LOAD must remain under extraction");
    assert_eq!(shaped_load.dtype(), DType::BFloat16);
    assert_eq!(shaped_load.shape().unwrap().unwrap().as_slice(), &[svod_ir::SInt::Const(8)]);
    let (buffer, _) = expect_index(&lowered);
    assert_same!(buffer, shaped_load);
    assert_no_weak(&lowered);
}

/// Weak lowering commits the value but must leave the INVALID marker alone — it is the
/// gate, not a number, and rewriting it would turn a skipped access into a real one.
#[test]
fn weak_lowering_preserves_the_invalid_marker() {
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let lowered = lower_weak(weak_int(7).valid(gate.clone()));

    let view = unwrap_op!(lowered, Op::Cast(c) => c);
    assert_eq!(view.dtype, DType::WeakInt);
    let Op::Ternary(TernaryOp::Where, condition, value, invalid) = view.src.op() else {
        panic!("expected a WHERE, got {}", lowered.tree())
    };
    assert_same!(condition, gate);
    assert_eq!(value.dtype(), DType::Int32);
    assert!(UOp::is_invalid_marker(invalid));

    let invalid = UOp::invalid_marker();
    let vector = UOp::stack([weak_int(7), invalid.clone()].into_iter().collect());
    let lowered = lower_weak(vector);

    assert_eq!(lowered.dtype(), DType::Int32);
    assert_eq!(lowered.shape().unwrap().unwrap().as_slice(), &[svod_ir::SInt::Const(2)]);
    let lanes = unwrap_op!(lowered, Op::Stack(s) => s).sources.clone();
    assert_eq!(lanes[0].dtype(), DType::Int32);
    assert_same!(lanes[1], invalid);
}

/// Rewriting an address INVALID to 0 would turn a skipped access into an unconditional
/// read of element 0, so `pm_remove_invalid` must leave gated addresses to index lowering.
#[test]
fn invalid_removal_leaves_gated_addresses_alone() {
    let address = UOp::var("i", DType::Index, 0, 16).valid(UOp::var("gate", DType::Bool, 0, 1));
    let stacked = UOp::new(
        Op::Stack(ops::Stack { sources: [address.clone(), UOp::invalid_marker()].into_iter().collect() }),
        DType::Index,
    );

    for gated in [address, stacked] {
        let result = rewrite(crate::symbolic::patterns::pm_remove_invalid(), gated.clone());
        assert_same!(result, gated);
    }
}

/// A STORE takes its value dtype from the destination buffer and must not adapt, replace
/// or re-index the address it was given.
#[test]
fn store_commits_its_weak_value_to_the_destination() {
    let index = index_of(param(0, 16, DType::Float32), UOp::native_const(0i32));
    let store = index.store(UOp::const_(DType::WeakFloat, ConstValue::Float(1.0)));

    let lowered = graph_rewrite(&pm_commit_weak(), store, &mut ());

    let (lowered_index, value, _) = expect_store(&lowered);
    assert_eq!(value.dtype(), DType::Float32);
    assert_eq!(index.dtype(), DType::Float32, "INDEX exposes the adopted buffer dtype");
    assert_same!(lowered_index, index);
}

/// A 64-bit INDEX under a gate: it narrows only when the buffer's shape product, not a flattened 4 GiB limit, fits in i32.
fn gated_long_index(buffer: Arc<UOp>) -> Arc<UOp> {
    let idx = UOp::variable("idx".into(), 0, i64::MAX / 2, DType::Int64);
    index_of(buffer, idx.valid(UOp::const_(DType::Bool, ConstValue::Bool(true))))
}

#[test_case(param(0, 16, DType::Float32), true; "index fits in 32 bits")]
#[test_case(param(0, i32::MAX as usize + 2, DType::Float32), false; "a flat buffer needs the full 64-bit range")]
#[test_case(UOp::param_with_shape(0, &smallvec::smallvec![svod_ir::SInt::Const(i32::MAX as usize), svod_ir::SInt::Const(3)], DType::Float32, None), false; "a wide dimension keeps 64 bits")]
#[test_case(UOp::param_with_shape(1, &smallvec::smallvec![svod_ir::SInt::Const(4)], DType::Float32, None), true; "a small dimension narrows")]
fn gated_long_index_narrowing_uses_the_shape_product(buffer: Arc<UOp>, narrowed: bool) {
    let lowered = lower_index(gated_long_index(buffer));

    let (_, indices) = expect_index(&lowered);
    let Op::Ternary(TernaryOp::Where, _, idx, invalid) = indices[0].op() else { panic!("expected a gated index") };
    assert_eq!(idx.dtype() == DType::Int32, narrowed, "{}", lowered.tree());
    assert!(UOp::is_invalid_marker(invalid));
}

/// The weak `Shrink` arm commits each weak side independently, at the width its own
/// value needs: a shrink may take a small offset out of a source that needs 64 bits.
#[test_case(3, 4, DType::Int32 ; "small weak dimensions commit to i32")]
#[test_case(i32::MAX as i64 + 1, i32::MAX as i64 + 1, DType::Int64 ; "wide weak dimensions commit to i64")]
fn weak_shrink_dimensions_commit_to_their_own_width(offset: i64, size: i64, expect: DType) {
    let src = param(0, 64, DType::Float32);
    let shrink = shrink(src.clone(), weak_int(offset), weak_int(size));

    let lowered = lower_index(shrink);

    let view = unwrap_op!(lowered, Op::Shrink(s) => s);
    assert_eq!(view.src.dtype(), src.dtype());
    assert_eq!(view.offsets.dtype(), expect);
    assert_eq!(view.sizes.dtype(), expect);
    assert_no_weak(&lowered);
}

/// The gated `Shrink` arm narrows a 64-bit offset only when the whole source fits in
/// 32 bits; the gate survives the narrowing either way.
#[test_case(16, true ; "a small source narrows the gated offset")]
#[test_case(i32::MAX as usize + 2, false ; "a wide source keeps the gated offset at 64 bits")]
fn gated_shrink_offsets_narrow_only_for_small_sources(size: usize, narrowed: bool) {
    let src = param(0, size, DType::Float32);
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let offsets = UOp::variable("i".into(), 0, i64::MAX / 2, DType::Int64).valid(gate.clone());
    let shrink = shrink(src.clone(), offsets, UOp::index_const(1));

    let lowered = lower_index(shrink);
    let view = unwrap_op!(lowered, Op::Shrink(s) => s);
    let Op::Ternary(TernaryOp::Where, condition, idx, invalid) = view.offsets.op() else {
        panic!("expected a gated offset, got {}", lowered.tree())
    };
    assert_same!(condition, gate);
    assert_eq!(idx.dtype() == DType::Int32, narrowed);
    assert!(UOp::is_invalid_marker(invalid));
}

/// `lower_weak_srcs` (tinygrad/uop/weak.py:29-40) keeps a `ctx` dict keyed by source:
/// one rewrite per distinct weak node, however many consumers read it.
#[test]
fn shared_weak_sources_are_lowered_once_per_pass() {
    let weak_index =
        |offset: i64| UOp::new(Op::Binary(BinaryOp::Add, UOp::range_const(64, 0), weak_int(offset)), DType::WeakInt);
    let shared = weak_index(3);
    let sink = UOp::sink(vec![
        index_of(param(0, 64, DType::Float32), shared.clone()),
        index_of(param(1, 64, DType::Float32), shared),
        index_of(param(2, 64, DType::Float32), weak_index(5)),
    ]);

    let mut memo = WeakMemo::default();
    graph_rewrite(&pm_lower_index_dtype(), sink, &mut memo);

    // Six weak edges reach a non-weak consumer here: three INDEX indices and the shared
    // WeakInt extent of the three PARAM shapes. They collapse to three rewrites.
    assert_eq!(memo.len(), 3, "one entry per distinct weak source, not per consumer edge");
}
