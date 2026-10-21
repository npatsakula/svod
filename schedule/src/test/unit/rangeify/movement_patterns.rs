//! Movement ops (RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP) folded into the
//! INDEX that reads them.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{AxisType, Op, SInt, UOp};
use test_case::test_case;

use crate::rangeify::patterns::movement_op_patterns;
use crate::test::support::prelude::*;

/// Loop-typed, Index-typed ranges of `sizes`, the way the indexer feeds a
/// movement chain.
fn ranges_of(sizes: &[i64]) -> Vec<Arc<UOp>> {
    sizes.iter().enumerate().map(|(id, &size)| range(size, AxisType::Loop, id)).collect()
}

fn shape(dims: &[i64]) -> Arc<UOp> {
    stack(dims.iter().map(|&d| UOp::index_const(d)))
}

fn reshaped(buffer: Arc<UOp>, dims: &[i64]) -> Arc<UOp> {
    buffer.try_reshape(&dims.iter().map(|&d| SInt::Const(d as usize)).collect()).expect("reshape")
}

fn with_ranges(movement: Arc<UOp>, sizes: &[i64]) -> Arc<UOp> {
    UOp::index().buffer(movement).indices(ranges_of(sizes)).call().expect("index")
}

fn broadcast(src: Arc<UOp>, dims: &[i64]) -> Arc<UOp> {
    UOp::new(Op::Expand(svod_ir::ops::Expand { src, new_shape: shape(dims) }), DType::Float32)
}

/// Every chain here reaches its BUFFER through a RESHAPE of matching rank, so the
/// whole address folds into one flat index. A chain whose ranks do not line up
/// keeps one index per output axis — see
/// [`expand_pads_the_rank_and_a_non_movement_source_is_left_alone`].
#[test_case(|| with_ranges(reshaped(buffer(200), &[10, 20]), &[10, 20]) ; "reshape")]
#[test_case(|| with_ranges(broadcast(reshaped(buffer(200), &[10, 1, 20]), &[10, 5, 20]), &[10, 5, 20]) ; "expand")]
#[test_case(|| with_ranges(reshaped(buffer(6000), &[10, 20, 30]).try_permute(vec![1, 2, 0]).expect("permute"), &[20, 30, 10]) ; "permute")]
#[test_case(|| {
    let src = reshaped(buffer(400), &[10, 40]);
    let offsets = stack([UOp::index_const(0), UOp::index_const(10)]);
    let sizes = stack([UOp::index_const(5), UOp::index_const(20)]);
    with_ranges(UOp::new(Op::Shrink(svod_ir::ops::Shrink { src, offsets, sizes }), DType::Float32), &[5, 20])
} ; "shrink")]
#[test_case(|| with_ranges(UOp::new(Op::Flip(svod_ir::ops::Flip { src: reshaped(buffer(200), &[10, 20]), axes: vec![false, true] }), DType::Float32), &[10, 20]) ; "flip")]
#[test_case(|| {
    let src = reshaped(buffer(200), &[10, 20]);
    let pads = stack([UOp::index_const(1), UOp::index_const(2)]);
    with_ranges(UOp::new(Op::Pad(svod_ir::ops::Pad { src, begin_pads: pads.clone(), end_pads: pads }), DType::Float32), &[12, 24])
} ; "pad")]
#[test_case(|| {
    let flat = broadcast(reshaped(buffer(10), &[10, 1]), &[10, 5]).try_reshape(&smallvec::smallvec![SInt::Const(50)]).expect("reshape");
    with_ranges(flat, &[50])
} ; "reshape of expand of reshape")]
fn movement_chains_flatten_into_the_buffer_index(build: fn() -> Arc<UOp>) {
    let result = rewrite(&movement_op_patterns(), build());

    assert_eq!(result.dtype(), DType::Float32);
    let (storage, indices) = expect_index(&result);
    assert_eq!(indices.len(), 1, "a rank-matched chain collapses to one flat index: {}", result.tree());
    assert!(matches!(storage.op(), Op::Buffer(..)), "no movement op may survive: {}", result.tree());
}

/// A zero-offset SHRINK covers the whole buffer: the rewrite drops the Shrink
/// and leaves an index straight off the BUFFER.
#[test]
fn a_zero_offset_shrink_folds_to_the_buffer() {
    let src = reshaped(buffer(200), &[10, 20]);
    let zero = stack([UOp::index_const(0), UOp::index_const(0)]);
    let whole = stack([UOp::index_const(10), UOp::index_const(20)]);
    let shrink = UOp::new(Op::Shrink(svod_ir::ops::Shrink { src, offsets: zero, sizes: whole }), DType::Float32);

    let result = rewrite(&movement_op_patterns(), with_ranges(shrink, &[10, 20]));

    assert!(!has_op(&result, |op| matches!(op, Op::Shrink(..))), "{}", result.tree());
    assert!(matches!(expect_index(&result).0.op(), Op::Buffer(..)), "{}", result.tree());
}

/// Padding only on the right is a noop for the address: it widens the shape but
/// never gates the index.
#[test]
fn a_right_only_pad_does_not_gate_the_index() {
    let src = reshaped(buffer(200), &[10, 20]);
    let zeros = stack([UOp::index_const(0), UOp::index_const(0)]);
    let pads = stack([UOp::index_const(2), UOp::index_const(3)]);
    let padded = UOp::new(Op::Pad(svod_ir::ops::Pad { src, begin_pads: zeros, end_pads: pads }), DType::Float32);

    let result = rewrite(&movement_op_patterns(), with_ranges(padded, &[10, 20]));

    assert!(!has_op(&result, |op| matches!(op, Op::Ternary(..))), "no WHERE is needed: {}", result.tree());
    assert!(matches!(expect_index(&result).0.op(), Op::Buffer(..)));
}

/// EXPAND whose new shape outranks the source aligns the source with the
/// trailing axes, and the leading axes it adds index nothing in the source
/// (Tinygrad `indexing.py`: `rngs[len(arg):]`). The rank-1 BUFFER keeps one
/// index, the trailing range, not a padded 0 for the added axis.
///
/// A non-movement source has nothing to fold into, so the INDEX keeps it as it is.
#[test]
fn expand_drops_the_added_axes_and_a_non_movement_source_is_left_alone() {
    let result =
        rewrite(&movement_op_patterns(), with_ranges(broadcast(reshaped(buffer(20), &[20]), &[3, 20]), &[3, 20]));

    let (storage, indices) = expect_index(&result);
    assert!(matches!(storage.op(), Op::Buffer(..)), "no EXPAND may survive: {}", result.tree());
    assert_eq!(indices.len(), 1, "one index per source axis: {}", result.tree());
    assert_eq!(expect_range_extent(&indices[0]), 20, "the trailing range survives: {}", result.tree());

    let computed = buffer(100).try_sqrt().expect("sqrt");
    let indexed = index_of(Arc::clone(&computed), range(100, AxisType::Loop, 0));
    let kept = rewrite(&movement_op_patterns(), indexed);
    assert_same!(expect_index(&kept).0, computed);
}

/// A movement op with no INDEX/AFTER/END consumer has nothing to fold into, and a
/// partial index — fewer indices than dims — only folds when the movement is a
/// RESHAPE whose trailing dims line up; PERMUTE and EXPAND never do.
#[test]
fn bare_and_partial_movement_chains_are_untouched() {
    let expanded = broadcast(reshaped(buffer(10), &[10, 1]), &[10, 4]).try_permute(vec![1, 0]).expect("permute");
    assert_same!(crate::rewrite::graph_rewrite_bottom_up(&movement_op_patterns(), expanded.clone(), &mut ()), expanded);
    let movements = [
        reshaped(reshaped(buffer(12), &[2, 6]), &[2, 3, 2]),
        reshaped(buffer(6), &[2, 3]).try_permute(vec![1, 0]).expect("permute"),
        broadcast(reshaped(buffer(2), &[2, 1]), &[2, 3]),
    ];
    for movement in movements {
        let indexed = with_ranges(movement, &[2]);
        assert_same!(
            crate::rewrite::graph_rewrite_bottom_up(&movement_op_patterns(), indexed.clone(), &mut ()),
            indexed
        );
    }
}

// ===== AFTER boundaries =====

/// AFTER is an ordering edge, not data: INDEX pushes through it and keeps both
/// the passthrough buffer and the dep.
#[test]
fn index_pushes_through_an_after_without_losing_its_dep() {
    let storage = buffer(8);
    let r = range(8, AxisType::Loop, 0);
    let dep = UOp::noop();
    let indexed = index_of(Arc::clone(&storage), r.clone()).after(smallvec::smallvec![Arc::clone(&dep)]);

    let result = rewrite(&movement_op_patterns(), indexed);

    let (buffer_after, indices) = expect_index(&result);
    let (passthrough, deps) = expect_after(&buffer_after);
    assert_same!(passthrough, storage);
    assert_eq!(deps.len(), 1);
    assert_same!(deps[0], dep);
    assert_same!(indices[0], r);
}

/// Moving a movement op outside an AFTER leaves the tag on the AFTER; the
/// rebuilt movement node is fresh and untagged.
#[test]
fn movement_through_after_keeps_the_tag_on_the_after() {
    let storage = buffer(20);
    let store = storage.store(UOp::native_const(1.0f32));
    let after =
        reshaped(Arc::clone(&storage), &[4, 5]).after(smallvec::smallvec![store]).rtag(Some(smallvec::smallvec![7]));

    let result = rewrite(&movement_op_patterns(), after);

    let Op::Reshape(svod_ir::ops::Reshape { src: inner, .. }) = result.op() else {
        panic!("expected RESHAPE outside, got {}", result.tree())
    };
    assert!(matches!(inner.op(), Op::After(..)));
    assert_eq!(inner.tag().as_deref(), Some([7usize].as_slice()));
    assert!(result.tag().is_none());
}

/// The END consumer of a movement chain folds it just like an INDEX does.
#[test]
fn end_over_a_movement_chain_folds_too() {
    let permuted = reshaped(buffer(200), &[10, 20]).try_permute(vec![1, 0]).expect("permute");
    let ended = permuted.end(smallvec::smallvec![range(20, AxisType::Loop, 0), range(10, AxisType::Loop, 1)]);

    let result = rewrite(&movement_op_patterns(), ended);

    assert!(!has_op(&result, |op| matches!(op, Op::Permute(..))), "{}", result.tree());
}
