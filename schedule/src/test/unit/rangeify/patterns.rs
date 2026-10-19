//! Rangeify pattern matchers: `early_rewrites`, `dead_axis_removal`, and the movement-op removal folded into `apply_rangeify_patterns`.  `buffer_folding` rows live in `buffer_folding.rs`.

use std::sync::Arc;

use proptest::prelude::*;
use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{AxisType, BinaryOp, BufferizeOpts, DeviceSpec, Op, ReduceOp, SInt, UOp};
use test_case::test_case;

use crate::rangeify::IndexingContext;
use crate::rangeify::patterns;
use crate::rewrite::graph_rewrite;
use crate::test::support::build::{range, stack};
use crate::test::support::proptest::cheap;
use svod_ir::ops;

use super::helpers::{assert_no_match, assert_same_ptr, has_op, rewritten};

// ===== early_rewrites =====

/// Autograd markers are erased, returning their source verbatim. The double DETACH row pins that one application peels exactly one layer.
#[test_case(|x: &Arc<UOp>| x.detach(), |x: &Arc<UOp>| x.clone() ; "detach")]
#[test_case(|x: &Arc<UOp>| x.contiguous_backward(), |x: &Arc<UOp>| x.clone() ; "contiguous backward")]
#[test_case(|x: &Arc<UOp>| x.detach().detach(), |x: &Arc<UOp>| x.detach() ; "nested detach peels one layer")]
fn markers_are_replaced_by_their_source(marked: fn(&Arc<UOp>) -> Arc<UOp>, source: fn(&Arc<UOp>) -> Arc<UOp>) {
    let x = UOp::native_const(42.0f32);
    assert_same_ptr(&rewritten(&patterns::early_rewrites(), &marked(&x), &mut ()), &source(&x));
}

/// A hint-less CONTIGUOUS over a realized node, or a reshape of one, is that
/// node; a constant, a plain buffer and a hinted CONTIGUOUS keep theirs.
#[test_case(|b: &Arc<UOp>| b.after(smallvec![UOp::noop()]), true ; "after")]
#[test_case(|b: &Arc<UOp>| reshape_op(b.after(smallvec![UOp::noop()]), shape(&[2, 2])), true ; "reshape of after")]
#[test_case(|b: &Arc<UOp>| b.clone(), false ; "buffer")]
#[test_case(|_: &Arc<UOp>| UOp::native_const(1.0f32), false ; "const")]
fn a_contiguous_over_a_realized_node_is_that_node(source: fn(&Arc<UOp>) -> Arc<UOp>, folds: bool) {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let src = source(&buffer);
    let contiguous = src.contiguous();
    if folds {
        assert_same_ptr(&rewritten(&patterns::early_rewrites(), &contiguous, &mut ()), &src);
    } else {
        assert_no_match(&patterns::early_rewrites(), &contiguous, &mut ());
    }
    let hinted =
        src.contiguous_with_opts(vec![svod_ir::ContiguousHint { op: "UPCAST".into(), axis: Some(0), arg: Some(4) }]);
    assert_no_match(&patterns::early_rewrites(), &hinted, &mut ());
}

#[test]
fn a_same_device_copy_is_replaced_by_its_source() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let copy = buffer.copy(DeviceSpec::Cpu).rtag(Some(smallvec![3]));
    assert_same_ptr(&rewritten(&patterns::early_rewrites(), &copy, &mut ()), &buffer);
}

#[test]
fn early_rewrites_leaves_plain_compute_alone() {
    let a = UOp::native_const(1.0f32);
    for untouched in [a.clone(), a.try_add(&a).expect("add")] {
        assert_no_match(&patterns::early_rewrites(), &untouched, &mut ());
    }
}

/// A widening integer cast of a product moves onto the operands, so the product is formed at the accumulator's width instead of wrapping.
#[test_case(DType::Int8, DType::Int32, true; "int8 product to int32")]
#[test_case(DType::UInt8, DType::UInt32, true; "uint8 product to uint32")]
#[test_case(DType::Int8, DType::UInt16, false; "sign change stays")]
#[test_case(DType::Float16, DType::Float32, false; "float product stays")]
#[test_case(DType::Int32, DType::Int8, false; "narrowing stays")]
fn widening_integer_cast_moves_onto_the_product_operands(stored: DType, wide: DType, rewrites: bool) {
    let a = UOp::new_buffer(DeviceSpec::Cpu, 4, stored.clone());
    let b = UOp::new_buffer(DeviceSpec::Cpu, 4, stored);
    let cast = a.try_mul(&b).expect("mul").cast(wide.clone());
    if !rewrites {
        assert_no_match(&patterns::early_rewrites(), &cast, &mut ());
        return;
    }
    let out = rewritten(&patterns::early_rewrites(), &cast, &mut ());
    let Op::Binary(BinaryOp::Mul, x, y) = out.op() else { panic!("expected MUL, got {}", out.tree()) };
    assert!(matches!(x.op(), Op::Cast(..)) && matches!(y.op(), Op::Cast(..)));
    assert_eq!((out.dtype(), x.dtype(), y.dtype()), (wide.clone(), wide.clone(), wide));
}

/// `neg(x)` lowers to `x * -1` with the constant already wrapped into the narrow type, so a widening cast above it must not touch the product.
#[test]
fn widening_cast_leaves_constant_products_alone() {
    let x = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::UInt8);
    for product in [x.neg(), x.try_mul(&x.const_like(3)).expect("mul")] {
        assert_no_match(&patterns::early_rewrites(), &product.cast(DType::Int32), &mut ());
    }
}

/// A reduction over an empty axis folds to its identity broadcast over the surviving shape — not to a bare scalar.
#[test]
fn empty_reduction_folds_to_a_shaped_identity() {
    let source = UOp::new_buffer(DeviceSpec::Cpu, 0, DType::Float32)
        .try_reshape(&smallvec![SInt::Const(0), SInt::Const(3)])
        .expect("reshape");
    let reduce = source.try_reduce_axis(ReduceOp::Add, vec![0]).expect("reduce axis");
    let identity = rewritten(&patterns::early_rewrites(), &reduce, &mut ());
    assert_eq!(identity.shape().expect("shape").expect("static").as_slice(), &[SInt::Const(3)]);
    assert!(matches!(
        identity.op(),
        Op::Expand(ops::Expand { src, .. }) if matches!(src.op(), Op::Const(value) if value.0.try_float() == Some(0.0))
    ));
}

/// A zero-element buffer that is *not* reduced still collapses to a constant, so nothing downstream has to iterate a zero-sized axis.
#[test]
fn a_zero_element_buffer_folds_to_a_constant() {
    let source = UOp::new_buffer(DeviceSpec::Cpu, 0, DType::Float32);
    let folded = graph_rewrite(&patterns::early_rewrites(), source.contiguous(), &mut ());
    assert!(
        has_op(&folded, |op| matches!(op, Op::Const(value) if value.0.try_float() == Some(0.0))),
        "a zero-element tensor must fold to 0:\n{}",
        folded.tree()
    );
}

/// Adjacent untagged RESHAPEs merge: the second reshape has to read the original source, or the index chain keeps a redundant step.
#[test]
fn adjacent_reshapes_merge() {
    let source = UOp::new_buffer(DeviceSpec::Cpu, 24, DType::Float32);
    let first = source.try_reshape(&smallvec![SInt::Const(4), SInt::Const(6)]).expect("reshape");
    let second = first.try_reshape(&smallvec![SInt::Const(2), SInt::Const(12)]).expect("reshape");
    let merged = rewritten(&patterns::early_rewrites(), &second, &mut ());
    let Op::Reshape(ops::Reshape { src, .. }) = merged.op() else {
        panic!("expected a single RESHAPE, got {}", merged.tree())
    };
    assert!(Arc::ptr_eq(src, &source), "the merged reshape must read the original source");
}

/// The three movement-op literals the support layer does not build.
fn permute_op(src: Arc<UOp>, axes: Vec<usize>) -> Arc<UOp> {
    UOp::new(Op::Permute(ops::Permute { src, axes }), DType::Float32)
}

fn reshape_op(src: Arc<UOp>, new_shape: Arc<UOp>) -> Arc<UOp> {
    UOp::new(Op::Reshape(ops::Reshape { src, new_shape }), DType::Float32)
}

fn expand_op(src: Arc<UOp>, new_shape: Arc<UOp>) -> Arc<UOp> {
    UOp::new(Op::Expand(ops::Expand { src, new_shape }), DType::Float32)
}

/// A static shape as a `STACK` of index constants.
fn shape(vals: &[i64]) -> Arc<UOp> {
    stack(vals.iter().map(|&v| UOp::index_const(v)))
}

/// `[4] -> reshape -> expand([4,8]) -> to(Amd)`: without materialising the view the transfer is sized by the `[4]` base and the destination under-allocated. A pure reshape is a contiguous view of the same element count, so it passes.
#[test]
fn a_copy_source_is_materialised_only_when_the_view_resizes_it() {
    let source = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let amd = DeviceSpec::Amd { device_id: 0 };
    let expanded = source
        .try_reshape(&smallvec![SInt::Const(4), SInt::Const(1)])
        .expect("reshape")
        .try_expand(&smallvec![SInt::Const(4), SInt::Const(8)])
        .expect("expand");
    let materialised = graph_rewrite(&patterns::early_rewrites(), expanded.copy_to_device(amd.clone()), &mut ());
    let Op::Copy(ops::Copy { src, .. }) = materialised.op() else {
        panic!("expected COPY, got {}", materialised.tree())
    };
    assert!(matches!(src.op(), Op::Contiguous(..)), "resized copy source must be materialised");
    let flat = source.try_reshape(&smallvec![SInt::Const(2), SInt::Const(2)]).expect("reshape").copy_to_device(amd);
    assert_same_ptr(&graph_rewrite(&patterns::early_rewrites(), flat.clone(), &mut ()), &flat);
}

/// The PERMUTE branch of `copy_needs_contiguous`: the stride order changed, so the transfer cannot read the view in place.
#[test]
fn a_permuted_copy_source_is_materialised() {
    let source = UOp::new_buffer(DeviceSpec::Cpu, 6, DType::Float32);
    let permuted = source
        .try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)])
        .expect("reshape")
        .try_permute(vec![1, 0])
        .expect("permute");
    let copied = permuted.copy_to_device(DeviceSpec::Amd { device_id: 0 });
    let out = graph_rewrite(&patterns::early_rewrites(), copied, &mut ());
    let Op::Copy(ops::Copy { src, .. }) = out.op() else { panic!("expected COPY, got {}", out.tree()) };
    assert!(matches!(src.op(), Op::Contiguous(..)), "a permuted copy source must be materialised");
}

// ===== dead_axis_removal =====

proptest! {
    #![proptest_config(cheap())]
    /// A range the compute does not read is dead. The STAGE is kept (it still has to become a STORE) but shrunk to zero
    /// ranges and re-broadcast through RESHAPE/EXPAND — an identity EXPAND is elided at construction.
    ///
    /// The property sweeps every 1..=3-axis shape with extents 1..=20, a superset of the old `[1]`, `[10, 1]`, `[10, 20]`
    /// rows.
    #[test]
    fn unread_ranges_are_stripped_from_the_stage(extents in prop::collection::vec(1i64..=20, 1..=3)) {
        let ranges = extents.iter().enumerate().map(|(id, &end)| range(end, AxisType::Loop, id)).collect();
        let stage = UOp::stage(UOp::native_const(1.0f32), ranges, BufferizeOpts::local());
        let result = rewritten(&patterns::dead_axis_removal(), &stage, &mut ());
        let reshape = match result.op() {
            Op::Expand(ops::Expand { src, .. }) => src,
            Op::Reshape(..) => &result,
            _ => panic!("expected EXPAND or RESHAPE, got {}", result.tree()),
        };
        let Op::Reshape(ops::Reshape { src: shrunk, .. }) = reshape.op() else {
            panic!("expected RESHAPE, got {}", result.tree())
        };
        prop_assert!(
            matches!(shrunk.op(), Op::Stage(ops::Stage { ranges, .. }) if ranges.is_empty()),
            "the STAGE must survive with no ranges, got {}",
            result.tree()
        );
    }
}

/// A COPY destination is sized by the transfer, so a dead axis must not shrink it — the guard `remove_bufferize` also applies (tinygrad rangeify.py:198,227).
#[test_case(|source: Arc<UOp>| source.copy(DeviceSpec::Cpu) ; "copy")]
#[test_case(|source: Arc<UOp>| source.copy_to_device(DeviceSpec::Amd { device_id: 0 }) ; "cross-device copy")]
#[test_case(|source: Arc<UOp>| source.contiguous() ; "contiguous")]
#[test_case(|_source: Arc<UOp>| UOp::noop() ; "noop")]
fn always_run_sources_keep_their_dead_axes(wrap: fn(Arc<UOp>) -> Arc<UOp>) {
    let stage = UOp::stage(wrap(UOp::native_const(1.0f32)), vec![range(1, AxisType::Loop, 0)], BufferizeOpts::local());
    assert_no_match(&patterns::dead_axis_removal(), &stage, &mut ());
}

/// The AFTER guard is the same always-run set: an AFTER's value is produced by the dependency it orders against, so its axes are not the STAGE's to drop.
#[test]
fn an_after_source_keeps_its_dead_axes() {
    let after = UOp::native_const(1.0f32).after(smallvec![UOp::noop()]);
    let stage = UOp::stage(after, vec![range(1, AxisType::Loop, 0)], BufferizeOpts::local());
    assert_no_match(&patterns::dead_axis_removal(), &stage, &mut ());
}

/// A symbolic extent cannot be proven dead (`vmax` is unknown), so the pass must bail rather than shrink an iteration space whose size it does not know.
#[test]
fn a_symbolic_end_is_not_a_dead_axis() {
    let symbolic = UOp::range(UOp::var("N", DType::Index, 0, 1024), 0);
    let stage = UOp::stage(UOp::native_const(1.0f32), vec![symbolic], BufferizeOpts::local());
    assert_no_match(&patterns::dead_axis_removal(), &stage, &mut ());
}

// ===== movement-op removal =====

/// Once ranges are assigned the movement op has been absorbed into the index expression and collapses to its source.
#[test_case(|src: Arc<UOp>| permute_op(src, vec![1, 0]) ; "permute")]
#[test_case(|src: Arc<UOp>| reshape_op(src, shape(&[4, 8])) ; "reshape")]
#[test_case(|src: Arc<UOp>| expand_op(src, shape(&[4, 8])) ; "expand")]
fn a_ranged_movement_op_collapses_to_its_source(build: fn(Arc<UOp>) -> Arc<UOp>) {
    let src = UOp::native_const(1.0f32);
    let movement = build(Arc::clone(&src));
    let range = range(4, AxisType::Loop, 0);
    let mut ctx = IndexingContext::new();
    ctx.set_ranges(&movement, vec![range.clone()], vec![range]);
    assert_same_ptr(&rewritten(&patterns::apply_rangeify_patterns(), &movement, &mut ctx), &src);
}

/// Without ranges there is nothing to fold the movement into, and a non-movement op never matches at all.
#[test]
fn nothing_is_removed_before_ranges_are_assigned() {
    let src = UOp::native_const(1.0f32);
    let mut ctx = IndexingContext::new();
    for uop in [permute_op(Arc::clone(&src), vec![1, 0]), src.try_sqrt().expect("sqrt")] {
        assert_no_match(&patterns::apply_rangeify_patterns(), &uop, &mut ctx);
    }
}
