//! `reduce_to_acc`: REDUCE -> DEFINE_REG accumulator + loop, and the WMMA-add fusion that runs beside it. Ported from
//! tinygrad `devectorizer.py:291-308`.
use super::helpers::*;
use proptest::prelude::*;
use smallvec::smallvec;
use std::collections::BTreeSet;
use std::sync::Arc;
use svod_dtype::{AddrSpace, DType};
use svod_ir::{AxisType, BinaryOp, Op, ReduceOp, UOp, ops};
use test_case::test_case;
fn fuse_wmma_add(root: Arc<UOp>) -> Arc<UOp> {
    rewrite_with(crate::devectorize::pm_wmma_add(), &mut (), root)
}
#[test]
fn wmma_add_direct_moves_into_accumulator() {
    let accumulator = shaped_f32("acc", 6, &[6]);
    let add = shaped_f32("add", 6, &[6]);
    let result = fuse_wmma_add(wmma_default(accumulator.clone()).add(&add));
    let Op::Wmma(ops::Wmma { c, .. }) = result.op() else { panic!("direct WMMA add must fuse") };
    assert!(matches!(c.op(), Op::Binary(BinaryOp::Add, lhs, rhs)
        if Arc::ptr_eq(lhs, &accumulator) && Arc::ptr_eq(rhs, &add)));
}
/// A non-broadcastable ADD must leave the WMMA unfused rather than abort (tinygrad's `codegen/__init__.py:110`
/// asserts inside `alu`; we decline the rewrite).
#[test]
fn wmma_add_with_mismatched_operand_does_not_fuse() {
    let fusable = wmma_default(shaped_f32("acc", 6, &[6])).add(&shaped_f32("add", 6, &[6]));
    let root = fusable.with_sources(vec![fusable.op().sources()[0].clone(), shaped_f32("bad", 3, &[3])]);
    assert_same!(fuse_wmma_add(root.clone()), root);
}
#[derive(Clone, Copy, Debug)]
enum OutputMovement {
    Permute,
    PermuteReshape,
}
/// The inverse movement the fusion has to push into the accumulator: a bare PERMUTE or the RESHAPE(PERMUTE) the
/// expander leaves behind.
#[test_case(OutputMovement::Permute; "permuted output")]
#[test_case(OutputMovement::PermuteReshape; "reshaped then permuted output")]
fn wmma_add_moves_through_output_movement(movement: OutputMovement) {
    let wmma = match movement {
        OutputMovement::Permute => wmma_default(shaped_f32("acc", 6, &[2, 3])),
        OutputMovement::PermuteReshape => reshape_to(&wmma_default(shaped_f32("acc", 6, &[6])), &[2, 3]),
    };
    let result = fuse_wmma_add(wmma.try_permute(vec![1, 0]).unwrap().add(&shaped_f32("add", 6, &[3, 2])));
    let Op::Permute(ops::Permute { src, axes }) = result.op() else { panic!("output permutation stays outside") };
    assert_eq!(axes, &[1, 0]);
    let src = match movement {
        OutputMovement::Permute => src.clone(),
        OutputMovement::PermuteReshape => match src.op() {
            Op::Reshape(ops::Reshape { src, .. }) => src.clone(),
            _ => panic!("output reshape must remain outside WMMA"),
        },
    };
    let Op::Wmma(ops::Wmma { c, .. }) = src.op() else { panic!("the add must fuse into WMMA") };
    match movement {
        OutputMovement::Permute => assert!(matches!(c.op(), Op::Binary(BinaryOp::Add, _, moved)
            if matches!(moved.op(), Op::Permute(ops::Permute { axes, .. }) if axes == &[1, 0]))),
        OutputMovement::PermuteReshape => assert!(matches!(c.op(), Op::Binary(BinaryOp::Add, _, moved)
            if matches!(moved.op(), Op::Reshape(ops::Reshape { src, .. }) if matches!(src.op(), Op::Permute(..))))),
    }
}
#[test]
fn movement_cleanup_must_precede_reduce_local() {
    let wmma = wmma_default(shaped_f32("acc", 6, &[6]));
    let root = reshape_to(&reshape_to(&wmma, &[3, 2]), &[2, 3]).try_permute(vec![1, 0]).unwrap().add(&shaped_f32(
        "add",
        6,
        &[3, 2],
    ));
    let mut ctx = crate::devectorize::ReduceContext::default();
    let without_cleanup = rewrite_with(&crate::devectorize::pm_reduce_local(), &mut ctx, root.clone());
    assert!(matches!(without_cleanup.op(), Op::Binary(BinaryOp::Add, ..)), "the counterexample must not match early");
    let matcher = crate::devectorize::movement_cleanup_patterns().with_context::<crate::devectorize::ReduceContext>()
        + crate::devectorize::pm_reduce_local();
    let mut ctx = crate::devectorize::ReduceContext::default();
    let ordered = rewrite_with(&matcher, &mut ctx, root);
    assert!(has_op(&ordered, |op| matches!(op, Op::Wmma(ops::Wmma { c, .. })
        if matches!(c.op(), Op::Binary(BinaryOp::Add, ..)))));
}
/// Every reduce op lowers to the same accumulator skeleton: a dense REG slot, a loop END and no surviving REDUCE.
#[test_case(ReduceOp::Add, &[16]; "add")]
#[test_case(ReduceOp::Mul, &[8]; "mul")]
#[test_case(ReduceOp::Max, &[32]; "max")]
#[test_case(ReduceOp::Min, &[32]; "min")]
#[test_case(ReduceOp::Add, &[1]; "single element range")]
#[test_case(ReduceOp::Add, &[8, 4]; "two reduce ranges")]
fn reduce_lowers_to_an_accumulator_loop(reduce_op: ReduceOp, extents: &[i64]) {
    let ranges: Vec<_> = extents.iter().enumerate().map(|(id, &end)| reduce_range(end, id)).collect();
    let result = apply_pm_reduce(&reduce(UOp::native_const(1.0f32), ranges, reduce_op));
    assert!(!matches!(result.op(), Op::Reduce(..)), "REDUCE must be replaced by the accumulator pattern");
    assert_eq!(result.dtype(), DType::Float32);
    assert!(
        has_op(&result, |op| matches!(op, Op::Buffer(ops::Buffer { arg, .. })
            if arg.addrspace == Some(AddrSpace::Reg) && arg.slot == 0)),
        "the first accumulator uses dense REG slot 0"
    );
    assert!(ends(&result) > 0, "the reduce loop must be closed by an END");
}
/// A REDUCE over a LOAD: the realistic shape, where the reduce range is also the load address.
#[test]
fn reduce_over_load_lowers_to_an_accumulator() {
    let range = reduce_range(32, 0);
    let address = index_of(UOp::param(0, 1024, DType::Float32, None), range.clone());
    let result = apply_pm_reduce(&load(address).reduce(smallvec![range], ReduceOp::Add));
    assert!(!matches!(result.op(), Op::Reduce(..)));
    assert!(regs(&result) > 0);
}
#[test]
fn invalid_padded_lane_survives_reduction_removal() {
    let src = UOp::try_where(
        UOp::var("valid", DType::Bool, 0, 1),
        UOp::var("value", DType::Float32, 0, 100),
        UOp::invalid_marker(),
    )
    .unwrap();
    let result = apply_pm_reduce(&reduce(src, vec![reduce_range(16, 0)], ReduceOp::Max));
    assert!(!matches!(result.op(), Op::Reduce(..)));
    assert!(
        result.any_in_subtree(UOp::is_invalid_marker),
        "reduction removal must preserve Invalid for the later gater"
    );
}
#[test]
fn reduce_shaped_to_scalar() {
    let src = float_values([0.0, 1.0, 2.0, 3.0]);
    let result = apply_pm_reduce(&src.reduce_with_num_axes(smallvec![reduce_range(16, 0)], ReduceOp::Add, 1));
    assert!(!matches!(result.op(), Op::Reduce(..)));
    assert!(regs(&result) > 0);
    assert!(has_op(&result, |op| matches!(op, Op::Index(ops::Index { buffer, indices })
        if Arc::ptr_eq(buffer, &src) && indices.len() == 1)));
}
/// Without a range there is no accumulator: the shaped source folds straight into a left fold of scalar INDEXes.
#[test]
fn horizontal_reduce_no_ranges() {
    let src = float_values([0.0, 1.0, 2.0, 3.0]);
    let result = apply_pm_reduce(&src.reduce_with_num_axes(smallvec![], ReduceOp::Add, 1));
    assert!(!matches!(result.op(), Op::Reduce(..)));
    assert_eq!(regs(&result), 0, "a horizontal-only reduce needs no DEFINE_REG");
    assert_eq!(result.dtype(), DType::Float32);
    assert_eq!(
        count(&result, |node| matches!(node.op(), Op::Index(ops::Index { buffer, .. }) if Arc::ptr_eq(buffer, &src))),
        4
    );
}
#[test]
fn horizontal_reduce_uses_target_dtype() {
    let target = DType::BFloat16.vec(4).unwrap();
    let src = UOp::stack(
        (0..4).map(|i| UOp::const_(DType::BFloat16.vec(16).unwrap(), svod_ir::ConstValue::Float(i as f64))).collect(),
    );
    let reduce = UOp::new(
        Op::Reduce(ops::Reduce {
            src: src.clone(),
            ranges: smallvec![reduce_range(16, 0)],
            reduce_op: ReduceOp::Add,
            num_axes: 1,
        }),
        target.clone(),
    );
    let result = apply_pm_reduce(&reduce);
    assert_eq!(result.dtype(), target);
    for node in result.toposort() {
        if matches!(node.op(), Op::Index(ops::Index { buffer, .. }) if Arc::ptr_eq(buffer, &src)) {
            assert_eq!(node.dtype(), target);
        }
        if let Op::Binary(BinaryOp::Add, lhs, rhs) = node.op() {
            assert_eq!(lhs.dtype(), rhs.dtype());
        }
    }
    assert!(!has_op(&result, |op| matches!(op, Op::Cast(..))));
}
fn axis_type_strategy() -> impl Strategy<Value = AxisType> {
    prop_oneof![
        Just(AxisType::Thread),
        Just(AxisType::Global),
        Just(AxisType::Local),
        Just(AxisType::Loop),
        Just(AxisType::Unroll),
        Just(AxisType::Upcast),
    ]
}
proptest! {
    #![proptest_config(cheap())]
    /// tinygrad puts every RANGE reachable from the source into `input_ranges`, whatever its axis type; only the
    /// reduce range itself (and already-ended ranges) drop out. The accumulator init's `AFTER` dependencies are
    /// exactly that set.
    #[test]
    fn input_ranges_accept_every_axis_type(axis_types in prop::collection::vec(axis_type_strategy(), 0..4)) {
        let inner = reduce_range(16, 9);
        let outer: Vec<Arc<UOp>> =
            axis_types.iter().enumerate().map(|(id, &axis)| range(8 << id, axis, id)).collect();
        let src = outer.iter().map(|range| range.cast(DType::Float32))
            .reduce(|acc, term| acc.add(&term)).unwrap_or_else(|| inner.cast(DType::Float32));
        let result = apply_pm_reduce(&reduce(src, vec![inner], ReduceOp::Add));
        prop_assert!(!matches!(result.op(), Op::Reduce(..)));
        prop_assert!(regs(&result) == 1, "one reduce range, one accumulator");
        let expected: BTreeSet<u64> = outer.iter().map(|range| range.id).collect();
        let fed = result.toposort().into_iter().any(|node| matches!(node.op(),
            Op::After(ops::After { passthrough, deps })
                if passthrough.addrspace() == Some(AddrSpace::Reg)
                    && deps.iter().map(|dep| dep.id).collect::<BTreeSet<_>>() == expected));
        prop_assert!(fed, "accumulator init must take exactly the non-reduce RANGEs, wanted {expected:?}");
    }
}
/// A STAGE's fill ranges are closed by `pm_add_local_buffers`; counting one as an input range sequences the
/// accumulator's init inside the fill loop and so inside every enclosing reduce loop, which re-zeroes it.
#[test]
fn a_staged_fill_range_is_not_an_accumulator_input_range() {
    let reduce_rng = reduce_range(16, 9);
    let open = range(8, AxisType::Global, 0);
    let pass = range(4, AxisType::Loop, 1);
    let tile = stage_with(
        load(index_of(param(0, 1024, DType::Float32), pass.clone())),
        vec![pass.clone()],
        svod_ir::BufferizeOpts::local(),
    );
    let src = load(index_of(tile, reduce_rng.clone())).add(&open.cast(DType::Float32));
    let result = apply_pm_reduce(&reduce(src, vec![reduce_rng], ReduceOp::Add));

    assert!(!matches!(result.op(), Op::Reduce(..)));
    assert_eq!(regs(&result), 1, "one reduce range, one accumulator");
    for node in result.toposort() {
        let Op::After(ops::After { passthrough, deps }) = node.op() else { continue };
        assert!(
            passthrough.addrspace() != Some(AddrSpace::Reg) || !deps.iter().any(|dep| dep.id == pass.id),
            "the accumulator must not be sequenced after the fill's pass loop:\n{}",
            result.tree()
        );
    }
    let expected: BTreeSet<u64> = [open.id].into_iter().collect();
    assert!(
        result.toposort().into_iter().any(|node| matches!(node.op(),
            Op::After(ops::After { passthrough, deps })
                if passthrough.addrspace() == Some(AddrSpace::Reg)
                    && deps.iter().map(|dep| dep.id).collect::<BTreeSet<_>>() == expected)),
        "the init still takes the genuinely open RANGEs, wanted {expected:?}"
    );
}
