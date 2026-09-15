use std::collections::HashMap;
use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::{BinaryOp, ConstValue, Op, ParamArg, TernaryOp, UOp, ops};
use test_case::test_case;

use crate::devectorize::devectorize;
use crate::graph_rewrite;
use crate::late::{
    AddImageContext, indexing_simplify, memory_coalescing, pm_lower_grouped_shrink, pm_simplify_add_image,
};
use crate::optimizer::Renderer;
use crate::symbolic::patterns::sym;
use crate::test::support::prelude::*;

fn weak(value: i64) -> Arc<UOp> {
    UOp::const_(DType::WeakInt, ConstValue::Int(value))
}

/// A weak offset stored as an unsigned constant, for `integer_constant`'s `UInt` arm.
fn weak_uint(value: u64) -> Arc<UOp> {
    UOp::const_(DType::WeakInt, ConstValue::UInt(value))
}

fn x() -> Arc<UOp> {
    UOp::define_var("x".into(), 0, 7)
}

fn y() -> Arc<UOp> {
    UOp::define_var("y".into(), 0, 7)
}

fn image_param() -> Arc<UOp> {
    let shape = svod_ir::shape::shape_to_uop(&smallvec![1usize.into(), 4usize.into(), 4usize.into()]);
    let arg = ParamArg::buffer(0, DType::Float32, AddrSpace::Global, None);
    UOp::new(Op::Param(ops::Param { shape, arg: arg.into() }), DType::Float32)
}

fn const_int(node: &Arc<UOp>) -> i64 {
    let Op::Const(value) = node.op() else { panic!("expected CONST, got {}", node.tree()) };
    value.0.try_int().expect("integer constant")
}

/// A SHRINK address with a constant width; coalescing emits this shape and `pm_lower_grouped_shrink` consumes it.
fn shrink_index(buffer: &Arc<UOp>, offset: i64, width: i64) -> Arc<UOp> {
    UOp::new(Op::Shrink(ops::Shrink { src: buffer.clone(), offsets: weak(offset), sizes: weak(width) }), buffer.dtype())
}

#[track_caller]
fn gated_index(index: &Arc<UOp>) -> (Arc<UOp>, Arc<UOp>) {
    let (_, indices) = expect_index(index);
    let Op::Ternary(TernaryOp::Where, valid, idx, invalid) = indices[0].op() else {
        panic!("expected a gated index, got {}", index.tree())
    };
    assert!(UOp::is_invalid_marker(invalid));
    (valid.clone(), idx.clone())
}

fn gated_scalar_index(start: Arc<UOp>, valid: Arc<UOp>) -> Arc<UOp> {
    index_of(param(0, 8, DType::Int32), start.valid(valid))
}

fn load_at(buffer: &Arc<UOp>, index: Arc<UOp>) -> Arc<UOp> {
    load(index_of(buffer.clone(), index))
}

fn loads(root: &Arc<UOp>) -> Vec<Arc<UOp>> {
    root.toposort().into_iter().filter(|node| matches!(node.op(), Op::Load(..))).collect()
}

fn stores(root: &Arc<UOp>) -> Vec<Arc<UOp>> {
    root.toposort().into_iter().filter(|node| matches!(node.op(), Op::Store(..))).collect()
}

fn shrink_count(root: &Arc<UOp>) -> usize {
    count(root, |node| matches!(node.op(), Op::Shrink(..)))
}

fn no_float4() -> Renderer {
    let mut renderer = Renderer::cpu();
    renderer.supports_float4 = false;
    renderer
}

fn target_coalesce(sink: Arc<UOp>, renderer: &Renderer) -> Arc<UOp> {
    let simplified = rewrite(sym(), devectorize(&sink, renderer));
    memory_coalescing(simplified, renderer)
}

fn shaped_load(offsets: &[i64]) -> Arc<UOp> {
    let buffer = param(0, 16, DType::Float32);
    let indices = UOp::stack(offsets.iter().copied().map(UOp::index_const).collect());
    let index = UOp::new(Op::Index(ops::Index { buffer, indices: smallvec![indices] }), DType::Float32);
    UOp::sink(vec![UOp::new(Op::Load(ops::Load { index, alt: None, gate: None }), DType::Float32)])
}

fn shaped_store() -> Arc<UOp> {
    let buffer = param(0, 16, DType::Float32);
    let indices = UOp::stack((0..4).map(UOp::index_const).collect());
    let index = UOp::new(Op::Index(ops::Index { buffer, indices: smallvec![indices] }), DType::Float32);
    let value = float_values((0..4).map(f64::from));
    UOp::sink(vec![UOp::new(Op::Store(ops::Store { index, value, gate: None }), DType::Void)])
}

fn scalar_stores() -> Arc<UOp> {
    let buffer = param(0, 16, DType::Float32);
    UOp::sink(
        (0..4)
            .map(|offset| index_of(buffer.clone(), UOp::index_const(offset)).store(UOp::native_const(offset as f32)))
            .collect(),
    )
}

fn contiguous_loads(buffer: Arc<UOp>) -> Arc<UOp> {
    UOp::sink((0..4).map(|offset| load_at(&buffer, UOp::index_const(offset))).collect())
}

fn loads_at(buffer: &Arc<UOp>, indices: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::sink(indices.into_iter().map(|index| load_at(buffer, index)).collect())
}

fn mismatched_validity_loads() -> Arc<UOp> {
    let indices = vec![UOp::index_const(0).valid(x().lt(&weak(4))), UOp::index_const(1).valid(x().lt(&weak(5)))];
    loads_at(&param(0, 16, DType::Float32), indices)
}

fn two_base_loads() -> Arc<UOp> {
    let bases = [x().mul(&weak(2)), y().mul(&weak(2))];
    let indices = bases.iter().flat_map(|base| (0..2).map(|offset| base.add(&weak(offset)))).collect();
    loads_at(&param(0, 64, DType::Float32), indices)
}

/// The same INDEX node reused by two stores at one offset.
fn shared_index_stores() -> Arc<UOp> {
    let index = index_of(param(0, 16, DType::Float32), UOp::index_const(0));
    UOp::sink(vec![index.store(UOp::native_const(1.0f32)), index.store(UOp::native_const(2.0f32))])
}

/// Two INDEX nodes that key to the same `(buffer, base, validity, offset)` group.
fn distinct_index_stores() -> Arc<UOp> {
    let buffer = param(0, 16, DType::Float32);
    let first = index_of(buffer.clone(), x().add(&weak(0)));
    let second = index_of(buffer, weak(0).add(&x()));
    UOp::sink(vec![first.store(UOp::native_const(1.0f32)), second.store(UOp::native_const(2.0f32))])
}

#[test_case(x().mod_(&weak(4)), x().lt(&weak(4)), x(); "modulo folds away under an upper bound")]
#[test_case(x().floor_div(&weak(4)), weak(3).lt(&x()), weak(1); "floor-div folds to a constant under a lower bound")]
fn a_gated_index_is_simplified_under_its_validity(start: Arc<UOp>, valid: Arc<UOp>, expected: Arc<UOp>) {
    let result = graph_rewrite(
        &(sym().clone() + indexing_simplify().clone()),
        gated_scalar_index(start, valid.clone()),
        &mut (),
    );
    let (result_valid, result_idx) = gated_index(&result);
    assert_same!(result_valid, valid);
    assert_same!(result_idx, expected);
}

#[test_case(gated_scalar_index(x(), x().eq(&weak(3))); "validity that parse_valid cannot read")]
#[test_case(UOp::index()
    .buffer(param(0, 64, DType::Float32))
    .indices(vec![weak(0).valid(x().lt(&weak(4))), x().valid(x().lt(&weak(4)))])
    .call()
    .unwrap(); "two coordinates on a non-image buffer")]
fn indexing_simplify_declines(index: Arc<UOp>) {
    let result = graph_rewrite(indexing_simplify(), index.clone(), &mut ());
    assert_same!(result, index);
}

fn image_index(valid: Arc<UOp>) -> Arc<UOp> {
    UOp::index()
        .buffer(image_param())
        .indices(vec![x().valid(valid.clone()), y().valid(valid)])
        .dtype(DType::Float32)
        .call()
        .unwrap()
}

/// The two-index image rule fires when both coordinates carry the same validity: it
/// drops the clauses the address range rules out and re-gates the rest with the one
/// clause it cannot decide.
#[test_case(x().lt(&weak(4)), None ; "an out-of-bounds clause is dropped")]
#[test_case(x().lt(&weak(100)).and_(&y().lt(&weak(4))), Some(x().lt(&weak(100))) ; "a mixed validity keeps its relevant clause")]
fn an_image_access_keeps_only_its_relevant_clause(valid: Arc<UOp>, relevant: Option<Arc<UOp>>) {
    let result = graph_rewrite(indexing_simplify(), image_index(valid), &mut ());

    let (_, indices) = expect_index(&result);
    assert_eq!(indices.len(), 2);
    for (coordinate, expected) in indices.iter().zip([x(), y()]) {
        let Some(clause) = &relevant else {
            assert_same!(*coordinate, expected);
            continue;
        };
        let Op::Ternary(TernaryOp::Where, condition, idx, invalid) = coordinate.op() else {
            panic!("expected a re-gated coordinate, got {}", result.tree())
        };
        assert_same!(condition, clause);
        assert_same!(idx, expected);
        assert!(UOp::is_invalid_marker(invalid));
    }
}

#[test_case(&[0, 1, 2, 3], Renderer::cpu(), 1, 4; "width four is one group")]
#[test_case(&[0, 1, 2, 3, 4, 5, 6, 7], Renderer::cpu(), 2, 4; "width eight is two float4 groups")]
#[test_case(&[0, 1, 2, 3, 8, 9, 10, 11], Renderer::cpu(), 2, 4; "a gap is not bridged into one access")]
fn a_shaped_load_splits_into_target_width_groups(offsets: &[i64], renderer: Renderer, groups: usize, width: usize) {
    let result = target_coalesce(shaped_load(offsets), &renderer);

    let folded = loads(&result);
    assert_eq!(folded.len(), groups, "{}", result.tree());
    assert!(folded.iter().all(|load| load.dtype() == DType::Float32), "memory dtype must stay scalar");
    assert_eq!(folded[0].shape().unwrap().unwrap()[0].as_const(), Some(width));
    assert_eq!(unwrap_op!(folded[0], Op::Load(l) => l).index.dtype(), DType::Float32);
}

#[test_case(shaped_store(); "one shaped store")]
#[test_case(scalar_stores(); "four contiguous scalar stores")]
fn contiguous_stores_fold_to_one_shaped_scalar_store(sink: Arc<UOp>) {
    let result = target_coalesce(sink, &Renderer::cpu());

    let folded = stores(&result);
    assert_eq!(folded.len(), 1, "{}", result.tree());
    let (index, value, _) = expect_store(&folded[0]);
    assert_eq!((index.dtype(), value.dtype()), (DType::Float32, DType::Float32));
    assert_eq!(unwrap_op!(value, Op::Stack(s) => s).sources.len(), 4);
}

#[test_case(contiguous_loads(param(0, 16, DType::Float32)), Renderer::cpu(), 1, 1; "four contiguous loads fold")]
#[test_case(contiguous_loads(UOp::buffer(0, 4, DType::Float32, AddrSpace::Reg, None).after(smallvec![UOp::noop()])), Renderer::cpu(), 4, 0; "reg accesses never coalesce")]
#[test_case(contiguous_loads(param(0, 16, DType::Float32)), no_float4(), 4, 0; "no float4 keeps scalar accesses")]
#[test_case(contiguous_loads(image_param()), no_float4(), 1, 1; "images use fixed width four regardless of float4")]
#[test_case(mismatched_validity_loads(), Renderer::cpu(), 2, 0; "different validity identities stay apart")]
#[test_case(two_base_loads(), Renderer::cpu(), 2, 2; "different base identities form their own runs")]
#[test_case(loads_at(&param(0, 16, DType::Float32), [0, 1, 3, 4].into_iter().map(UOp::index_const).collect()), Renderer::cpu(), 3, 1; "an unaligned run is not realigned")]
fn coalescing_groups_scalar_loads_by_run(sink: Arc<UOp>, renderer: Renderer, groups: usize, shrinks: usize) {
    let result = memory_coalescing(sink, &renderer);
    assert_eq!(loads(&result).len(), groups, "{}", result.tree());
    assert_eq!(shrink_count(&result), shrinks, "{}", result.tree());
}

/// Only the allowlisted element types are grouped; everything else stays scalar.
#[test_case(ScalarDType::Float16, 1 ; "float16 folds")]
#[test_case(ScalarDType::BFloat16, 1 ; "bfloat16 folds")]
#[test_case(ScalarDType::Int32, 1 ; "int32 folds")]
#[test_case(ScalarDType::UInt32, 1 ; "uint32 folds")]
#[test_case(ScalarDType::FP8E4M3, 1 ; "fp8 e4m3 folds")]
#[test_case(ScalarDType::FP8E5M2, 1 ; "fp8 e5m2 folds")]
#[test_case(ScalarDType::Int8, 4 ; "int8 is not foldable")]
#[test_case(ScalarDType::Float64, 4 ; "float64 is not foldable")]
fn only_allowlisted_element_types_coalesce(scalar: ScalarDType, groups: usize) {
    let result = memory_coalescing(contiguous_loads(UOp::param(0, 16, DType::Scalar(scalar), None)), &Renderer::cpu());
    assert_eq!(loads(&result).len(), groups, "{}", result.tree());
}

/// The SHRINK widths and scalar remainder offsets of a coalesced kernel.
fn group_layout(root: &Arc<UOp>) -> (Vec<usize>, Vec<i64>) {
    let (mut widths, mut scalars) = (Vec::new(), Vec::new());
    for leaf in loads(root) {
        let view = unwrap_op!(leaf, Op::Load(l) => l);
        match view.index.op() {
            Op::Shrink(ops::Shrink { offsets, sizes, .. }) => {
                assert_eq!((offsets.dtype(), sizes.dtype()), (DType::WeakInt, DType::WeakInt));
                assert_eq!(offsets.shape().unwrap().unwrap().as_slice(), &[]);
                assert_eq!(sizes.shape().unwrap().unwrap().as_slice(), &[]);
                widths.push(usize::try_from(const_int(sizes)).expect("width fits"));
            }
            Op::Index(ops::Index { indices, .. }) => {
                assert_eq!(indices[0].dtype(), DType::WeakInt);
                let Op::Binary(BinaryOp::Add, _, offset) = indices[0].op() else {
                    panic!("scalar offset must preserve its base")
                };
                scalars.push(const_int(offset));
            }
            other => panic!("expected SHRINK or INDEX, got {other:?}"),
        }
    }
    (widths, scalars)
}

/// A symbolic base plus constant offsets groups in fours, and the remainder stays scalar with its base intact (tinygrad `test/test_linearizer.py`).
#[test_case(4 ; "four is one group")]
#[test_case(5 ; "five is a group and a scalar remainder")]
#[test_case(8 ; "eight is two groups")]
fn grouped_weak_index_offsets_match_tinygrad(width: usize) {
    let buffer = param(0, 32, DType::Float32);
    let base = UOp::define_var("group_width_base".into(), 0, 3).mul(&weak(8));
    let accesses = (0..width).map(|offset| load_at(&buffer, base.add(&base.const_like(offset as i64)))).collect();
    let result = memory_coalescing(UOp::sink(accesses), &Renderer::cpu());

    let (grouped, remainder) = (width / 4, width % 4);
    assert_eq!(loads(&result).len(), grouped + usize::from(remainder > 0), "{}", result.tree());
    let (widths, scalars) = group_layout(&result);
    assert_eq!(widths, vec![4; grouped]);
    assert_eq!(scalars, (0..remainder).map(|i| (4 * grouped + i) as i64).collect::<Vec<_>>());
}

#[test]
fn a_shaped_load_with_shared_validity_keeps_one_group_gate() {
    let valid = x().lt(&weak(4));
    let indices = UOp::stack((0..4).map(|offset| UOp::index_const(offset).valid(valid.clone())).collect());
    let index = UOp::new(
        Op::Index(ops::Index { buffer: param(0, 16, DType::Float32), indices: smallvec![indices] }),
        DType::Float32,
    );

    let result = target_coalesce(UOp::sink(vec![load(index)]), &Renderer::cpu());
    let folded = loads(&result);

    assert_eq!(folded.len(), 1, "shared validity should produce one shaped access: {}", result.tree());
    let view = unwrap_op!(folded[0], Op::Load(l) => l);
    let Op::Shrink(ops::Shrink { offsets, sizes, .. }) = view.index.op() else {
        panic!("expected SHRINK: {}", view.index.tree())
    };
    assert_same!(offsets.get_valid(), valid);
    assert_const!(offsets.get_idx(), 0);
    assert_const!(sizes, 4);
}

#[test]
fn wmma_output_stores_stay_distinct_through_coalescing() {
    let output = param(0, 80, DType::Float32);
    let lidx = UOp::special(weak(32), "lidx0".to_string());
    let valid = lidx.lt(&weak(16));
    let indices = [lidx.clone(), lidx.add(&weak(32)), lidx.add(&weak(64)).valid(valid.clone())];
    let before = UOp::sink(
        indices
            .into_iter()
            .enumerate()
            .map(|(value, index)| index_of(output.clone(), index).store(UOp::native_const(value as f32)))
            .collect(),
    );

    let after_stores = stores(&memory_coalescing(before, &Renderer::cpu()));

    assert_eq!(after_stores.len(), 3, "the three output bands must not merge");
    let carrying = after_stores
        .iter()
        .filter(|store| {
            let (index, _, _) = expect_store(store);
            let (_, indices) = expect_index(&index);
            Arc::ptr_eq(&indices[0].get_valid(), &valid)
        })
        .count();
    assert_eq!(carrying, 1, "only the C[64..80) store uses the M=5 validity identity");
}

/// A group whose offsets carry more than one store is left alone.
#[test_case(shared_index_stores() ; "the same index node under two stores")]
#[test_case(distinct_index_stores() ; "two index nodes that key to the same offset")]
fn multiple_stores_to_one_group_offset_are_left_un_coalesced(sink: Arc<UOp>) {
    let result = memory_coalescing(sink, &Renderer::cpu());
    assert_eq!(stores(&result).len(), 2, "both stores survive; coalescing declines the group");
}

#[test]
fn a_gated_load_is_skipped_rather_than_aborting_the_pass() {
    let index = index_of(param(0, 16, DType::Float32), UOp::index_const(0));
    let gated = UOp::new(
        Op::Load(ops::Load { index, alt: Some(UOp::native_const(0.0f32)), gate: Some(UOp::native_const(true)) }),
        DType::Float32,
    );

    let folded = loads(&memory_coalescing(UOp::sink(vec![gated.clone()]), &Renderer::cpu()));
    assert_eq!(folded.len(), 1);
    assert_same!(folded[0], gated);
}

/// `pm_simplify_add_image` removes a Float16 roundtrip and widens a Float16 STORE
/// value to the address dtype, but an F16-addressed LOAD cannot be built —
/// `svod_ir` pins LOAD dtype to the address — so that shape is left alone.
#[test]
fn image_float_accesses_adapt_to_the_address_dtype() {
    let value = UOp::native_const(1.0f32);
    let mut ctx: AddImageContext = (HashMap::new(), Renderer::cpu());
    assert_same!(
        graph_rewrite(&pm_simplify_add_image(), value.cast(DType::Float16).cast(DType::Float32), &mut ctx),
        value
    );

    let half = load(index_of(param(0, 16, DType::Float16), UOp::index_const(0)));
    let mut ctx: AddImageContext = (HashMap::new(), Renderer::cpu());
    assert_same!(graph_rewrite(&pm_simplify_add_image(), half.clone(), &mut ctx), half);

    let index = index_of(param(0, 16, DType::Float32), UOp::index_const(0));
    let gate = UOp::native_const(true);
    let store = UOp::new(
        Op::Store(ops::Store {
            index: index.clone(),
            value: UOp::const_(DType::Float16, ConstValue::Float(1.0)),
            gate: Some(gate.clone()),
        }),
        DType::Void,
    );
    let mut ctx: AddImageContext = (HashMap::new(), Renderer::cpu());
    let (lowered_index, value, lowered_gate) = expect_store(&graph_rewrite(&pm_simplify_add_image(), store, &mut ctx));
    assert_same!(lowered_index, index);
    assert_eq!(value.dtype(), DType::Float32);
    assert_same!(lowered_gate.clone().expect("the gate survives"), gate);
}

/// A coalescing SHRINK address is a temporary: a load lowers to one scalar load per
/// lane at `offsets + lane`, and a store to one gated scalar store per stack value.
#[test]
fn a_grouped_shrink_address_lowers_lanewise() {
    let buffer = param(0, 16, DType::Float32);
    let grouped =
        UOp::new(Op::Load(ops::Load { index: shrink_index(&buffer, 4, 4), alt: None, gate: None }), DType::Float32);

    let lanes = unwrap_op!(rewrite(pm_lower_grouped_shrink(), grouped.clone()), Op::Stack(s) => s).sources.clone();

    assert_eq!(lanes.len(), 4);
    for (lane, value) in lanes.iter().enumerate() {
        let view = unwrap_op!(value, Op::Load(l) => l);
        let (_, indices) = expect_index(&view.index);
        let expected = if lane == 0 { weak(4) } else { weak(4).add(&weak(lane as i64)) };
        assert_same!(indices[0], expected);
    }

    let gate = UOp::native_const(true);
    let grouped = UOp::new(
        Op::Store(ops::Store {
            index: shrink_index(&buffer, 2, 4),
            value: float_values([1.0, 2.0, 3.0, 4.0]),
            gate: Some(gate.clone()),
        }),
        DType::Void,
    );

    let lanes = unwrap_op!(rewrite(pm_lower_grouped_shrink(), grouped.clone()), Op::Group(g) => g).sources.clone();

    assert_eq!(lanes.len(), 4);
    for (lane, scalar) in lanes.iter().enumerate() {
        let (index, value, lowered_gate) = expect_store(scalar);
        assert_const!(value, (lane + 1) as f64);
        assert_same!(lowered_gate.clone().expect("the gate is copied to every lane"), gate);
        let (_, indices) = expect_index(&index);
        let expected = if lane == 0 { weak(2) } else { weak(2).add(&weak(lane as i64)) };
        assert_same!(indices[0], expected);
    }
}

/// A constant INDEX into a STACK extracts the lane; `UOp::index` short-circuits this shape at construction, so the node is built directly.
#[test]
fn a_constant_index_into_a_stack_extracts_the_lane() {
    let lanes = float_values([1.0, 2.0]);
    let raw =
        UOp::new(Op::Index(ops::Index { buffer: lanes, indices: smallvec![UOp::index_const(1)] }), DType::Float32);

    let extracted = rewrite(pm_lower_grouped_shrink(), raw);
    assert_const!(extracted, 2.0);
}

/// Width one, a symbolic width, a lifted gate and a mismatched value stack are all outside the temporary SHRINK shape.
#[test]
fn grouped_shrink_lowering_declines_unshaped_accesses() {
    let buffer = param(0, 16, DType::Float32);
    let grouped_load = |index| UOp::new(Op::Load(ops::Load { index, alt: None, gate: None }), DType::Float32);
    let symbolic = UOp::new(
        Op::Shrink(ops::Shrink {
            src: buffer.clone(),
            offsets: weak(0),
            sizes: UOp::var("width", DType::WeakInt, 2, 4),
        }),
        DType::Float32,
    );
    let candidates = [
        grouped_load(shrink_index(&buffer, 0, 1)),
        grouped_load(symbolic),
        UOp::new(
            Op::Load(ops::Load {
                index: shrink_index(&buffer, 0, 4),
                alt: Some(UOp::native_const(0.0f32)),
                gate: Some(UOp::native_const(true)),
            }),
            DType::Float32,
        ),
        UOp::new(
            Op::Store(ops::Store { index: shrink_index(&buffer, 0, 4), value: float_values([1.0, 2.0]), gate: None }),
            DType::Void,
        ),
    ];

    for candidate in candidates {
        let folded = rewrite(pm_lower_grouped_shrink(), candidate.clone());
        assert_same!(folded, candidate);
    }
}

/// `integer_constant` reads an unsigned constant as an integer offset, so two accesses one apart still form one run.
#[test]
fn unsigned_constant_offsets_are_read_as_integers() {
    let base = UOp::define_var("unsigned_offset_base".into(), 0, 3).mul(&weak(8));
    let sink = UOp::sink(
        [weak_uint(0), weak_uint(1)]
            .into_iter()
            .map(|offset| load_at(&param(0, 32, DType::Float32), base.add(&offset)))
            .collect(),
    );

    let result = memory_coalescing(sink, &Renderer::cpu());
    assert_eq!(loads(&result).len(), 1, "an unsigned offset must key the same run: {}", result.tree());
}
