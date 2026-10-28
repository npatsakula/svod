//! `IndexingContext`: range allocation, the realize map, and the helpers
//! `transform_single_source` uses to line consumer ranges up with a source.

use std::sync::Arc;

use svod_ir::{AxisId, AxisType, DType, Op, SInt, UOp, ops};

use crate::rangeify::{
    IndexingContext,
    indexing::{broadcast_ranges, data_sources},
};
use crate::test::support::prelude::*;
use test_case::test_case;

fn var() -> Arc<UOp> {
    UOp::var("x", DType::Float32, 0, i64::MAX)
}

/// Ranges are numbered sequentially as `AxisId::Unrenumbered`, keep their extent
/// (constant or symbolic), and a size-1 axis short-circuits to CONST 0 without
/// consuming an id. Contexts number their ranges independently.
#[test]
fn ranges_are_numbered_sequentially_and_size_one_axes_are_free() {
    let mut ctx = IndexingContext::new();
    assert_eq!(ctx.range_counter(), 0);

    for (i, extent) in [10i64, 20, 0, 1 << 30].into_iter().enumerate() {
        let range = ctx.new_range(&SInt::Const(extent as usize), AxisType::Loop);
        assert_eq!(range_axis_id(&range), AxisId::Unrenumbered(i));
        assert_eq!(expect_range_extent(&range), extent);
        assert_eq!(ctx.range_counter(), i + 1);
    }
    assert_const!(ctx.new_range(&SInt::Const(1), AxisType::Loop), 0);
    assert_eq!(ctx.range_counter(), 4, "a singleton consumes no axis id");

    let n = UOp::define_var("n".to_string(), 0, i64::MAX);
    let symbolic = ctx.new_range(&SInt::Symbolic(n.clone()), AxisType::Loop);
    assert_same!(expect_range(&symbolic).0, n);
    assert_eq!(range_axis_type(&ctx.new_range(&SInt::Const(10), AxisType::Reduce)), AxisType::Reduce);

    let mut second = IndexingContext::new();
    assert_eq!(range_axis_id(&second.new_range(&SInt::Const(30), AxisType::Loop)), AxisId::Unrenumbered(0));
}

#[test]
fn input_and_output_ranges_are_stored_and_read_back_per_uop() {
    let mut ctx = IndexingContext::new();
    let x = var();
    let r0 = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let r1 = ctx.new_range(&SInt::Const(20), AxisType::Loop);

    assert!(ctx.get_ranges(&x).is_none());
    ctx.set_ranges(&x, vec![r0.clone(), r1.clone()], vec![r0.clone()]);

    let (inputs, outputs) = ctx.get_ranges(&x).expect("ranges were set");
    inputs.iter().zip([&r0, &r1]).for_each(|(a, b)| assert_same!(a, b));
    assert_eq!(outputs.len(), 1);
    assert_same!(outputs[0], r0);
}

/// `mark_realize_all` realizes every axis (no axis list); `mark_realize` records
/// exactly the axes given, and index coordinates / AFTER deps are not data.
#[test]
fn the_realize_map_distinguishes_all_axes_from_named_axes() {
    let mut ctx = IndexingContext::new();
    let x = var();

    assert!(!ctx.should_realize(&x));
    assert!(ctx.get_realize_axes(&x).is_none());
    ctx.mark_realize_all(&x).expect("mark all");
    assert!(ctx.should_realize(&x));

    ctx.mark_realize(&x, vec![0, 2]);
    assert_eq!(ctx.get_realize_axes(&x).expect("axes"), &[0, 2]);

    let target = buffer(8);
    let after = target.clone().after(smallvec::smallvec![UOp::noop()]);
    for node in [index_of(Arc::clone(&target), global_range(8, 0)), after] {
        assert_eq!(data_sources(&node).len(), 1);
        assert_same!(data_sources(&node)[0], target);
    }
}

/// A rank-0 source keeps the consumer's range verbatim; an expanded singleton
/// axis is pinned to index 0 instead.
#[test]
fn broadcast_ranges_zeroes_only_the_expanded_axes() {
    let consumer_range = global_range(4, 0);
    let scalar = var();
    let consumer = scalar.try_add(&UOp::var("other", DType::Float32, 0, 4)).expect("add");
    let mapped = broadcast_ranges(&consumer, &scalar, std::slice::from_ref(&consumer_range));
    assert_eq!(mapped.len(), 1);
    assert_same!(mapped[0], consumer_range);

    let source =
        UOp::const_(DType::Float32, 1.0f32.into()).try_reshape(&smallvec::smallvec![SInt::Const(1)]).expect("reshape");
    let expanded = source.try_expand(&smallvec::smallvec![SInt::Const(4)]).expect("expand");
    let consumer = expanded.try_add(&expanded).expect("add");

    let mapped = broadcast_ranges(&consumer, &source, &[consumer_range]);
    assert_eq!(mapped.len(), 1);
    assert_op!(mapped[0], Op::Const(_));
}

/// An image buffer addresses two coordinates; every other dtype linearises to
/// one. `transform_single_source` has to pick per dtype.
#[test]
fn image_buffers_keep_two_index_addresses() {
    let ranges = [range(2, AxisType::Loop, 0), range(8, AxisType::Loop, 1)];
    let shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![2usize.into(), 8usize.into()]);
    let image = DType::Image { kind: svod_dtype::ImageKind::Float, shape: vec![2, 8, 4] };

    for (dtype, expected) in [(image, 2), (DType::Float32, 1)] {
        let arg = svod_ir::ParamArg::buffer(0, dtype.clone(), svod_dtype::AddrSpace::Global, None);
        let storage = UOp::new(Op::Buffer(ops::Buffer { shape: shape.clone(), arg: arg.into() }), dtype);
        let indexed = crate::rangeify::transforms::transform_single_source(
            &UOp::sink(vec![]),
            &storage,
            &ranges,
            &mut IndexingContext::new(),
        );
        let ops::Index { indices, .. } = assert_op!(indexed, Op::Index(i) => i);
        assert_eq!(indices.len(), expected);
    }
}

/// `apply_movement_op` and `_apply_reshape` are `@functools.cache` upstream
/// (tinygrad/schedule/indexing.py:158,171): process-global and keyed on the inputs,
/// so a second call with the same op, input shape and range tuple never rebuilds
/// the index chain.
///
/// The miss→hit transition is the whole claim. Hash-consing already makes two
/// equal chains the same `Arc` (`ir/src/uop/hash_consing.rs:241`), so a pointer
/// comparison alone would pass with the cache deleted; only `movement_cache_holds`
/// distinguishes a hit from a rebuild.
#[test]
fn equal_movement_inputs_reuse_the_cached_index_chain() {
    // Prime extents so no other test shares these inputs in the process-global cache.
    let rngs = vec![global_range(1013, 0), global_range(1019, 1)];
    let in_shape = [SInt::Const(11), SInt::Const(13)];
    let out_shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![SInt::Const(13), SInt::Const(11)]);
    let reshape =
        UOp::new(Op::Reshape(ops::Reshape { src: UOp::index_const(0), new_shape: out_shape }), DType::Float32);
    let holds = || crate::rangeify::indexing::movement_cache_holds(reshape.op(), &in_shape, &rngs);

    assert!(!holds(), "these inputs must be new");
    let first = crate::rangeify::apply_movement_op(reshape.op(), &in_shape, &rngs);
    assert!(holds(), "the first call memoises the inputs");
    let second = crate::rangeify::apply_movement_op(reshape.op(), &in_shape, &rngs);

    assert_eq!(first.len(), in_shape.len(), "one index per input axis");
    assert!(first.iter().zip(&second).all(|(a, b)| Arc::ptr_eq(a, b)), "a hit returns the cached nodes");
}

/// EXPAND aligns its source with the trailing output axes, so the source is
/// indexed by one index per source axis: an output axis the EXPAND adds on the
/// left (a rank-0 constant broadcast by `const_like`) maps to nothing, and a
/// broadcast singleton maps to CONST 0. `Some(i)` is the consumer's range `i`.
#[test_case(&[], &[5, 1], &[] ; "rank 0 to rank 2 drops both axes")]
#[test_case(&[], &[5], &[] ; "rank 0 to rank 1 drops the axis")]
#[test_case(&[1], &[5, 1], &[Some(1)] ; "rank 1 keeps the trailing range")]
#[test_case(&[1], &[3, 5], &[None] ; "rank 1 zeroes its broadcast singleton")]
#[test_case(&[1, 1], &[5, 1], &[None, Some(1)] ; "equal rank zeroes only the broadcast axis")]
#[test_case(&[5, 1], &[5, 4], &[Some(0), None] ; "equal rank keeps the unexpanded axis")]
fn expand_indexes_the_source_once_per_source_axis(in_dims: &[usize], out_dims: &[usize], expected: &[Option<usize>]) {
    let rngs: Vec<_> = out_dims.iter().enumerate().map(|(id, &extent)| global_range(extent as i64, id)).collect();
    let in_shape: Vec<SInt> = in_dims.iter().map(|&d| SInt::Const(d)).collect();
    let new_shape = svod_ir::shape::shape_to_uop(&out_dims.iter().map(|&d| SInt::Const(d)).collect());
    let expand = UOp::new(Op::Expand(ops::Expand { src: UOp::index_const(0), new_shape }), DType::Float32);

    let indices = crate::rangeify::apply_movement_op(expand.op(), &in_shape, &rngs);

    assert_eq!(indices.len(), expected.len(), "one index per source axis");
    for (index, expected) in indices.iter().zip(expected) {
        let want = expected.map_or_else(|| UOp::index_const(0), |i| rngs[i].clone());
        assert!(Arc::ptr_eq(index, &want), "got {index:?}, want {want:?}");
    }
}
