//! `UOp::linear_program`: list completion, repeat rules and the derived SINK.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{DType, DeviceSpec};
use test_case::test_case;

use crate::{AxisId, AxisType, ConstValue, Error, KernelInfo, Op, UOp, ops};

fn i32c(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(v))
}

fn store_at(buffer: &Arc<UOp>, i: &Arc<UOp>) -> Arc<UOp> {
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    UOp::index().buffer(buffer.clone()).indices(vec![i.clone()]).call().unwrap().store(value)
}

fn linear(ops: Vec<Arc<UOp>>) -> Result<Vec<Arc<UOp>>, Error> {
    let program = UOp::linear_program(KernelInfo::default(), ops, DeviceSpec::Cpu)?;
    let Op::Program(ops::Program { linear: Some(linear), .. }) = program.op() else { unreachable!() };
    let Op::Linear(ops::Linear { ops }) = linear.op() else { unreachable!() };
    Ok(ops.to_vec())
}

fn position(list: &[Arc<UOp>], node: &Arc<UOp>) -> usize {
    list.iter().position(|op| Arc::ptr_eq(op, node)).expect("listed")
}

/// The index constant of each STORE, in list order.
fn store_indices(list: &[Arc<UOp>]) -> Vec<i64> {
    list.iter()
        .filter_map(|op| match op.op() {
            Op::Store(ops::Store { index, .. }) => match index.op() {
                Op::Index(ops::Index { indices, .. }) => indices[0].vmax().try_int(),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

#[test]
fn unlisted_sources_land_before_their_first_user() {
    let y = UOp::param(0, 4, DType::Float32, None);
    let list = linear(vec![store_at(&y, &i32c(2)), store_at(&y, &i32c(1))]).unwrap();
    assert_eq!(store_indices(&list), vec![2, 1], "listed ops keep their order");
    for (at, op) in list.iter().enumerate() {
        for source in op.op().sources() {
            assert!(position(&list, &source) < at, "{:?} precedes its user", source.op());
        }
    }
    assert!(matches!(list.last().unwrap().op(), Op::Sink(..)), "the SINK closes the list");
}

#[test]
fn param_shapes_are_committed() {
    let y = UOp::param(0, 4, DType::Float32, None);
    let list = linear(vec![store_at(&y, &i32c(0))]).unwrap();
    assert!(list.iter().all(|op| !op.dtype().is_weak()), "a program admits no weak dtype");
}

#[test]
fn using_a_later_listed_op_is_an_error() {
    let y = UOp::param(0, 4, DType::Float32, None);
    let k = UOp::range_axis_dtype(i32c(4), AxisId::Renumbered(0), AxisType::Loop, DType::Int32);
    let store = store_at(&y, &k);
    let err = linear(vec![k.clone(), store.end(smallvec![k]), store]).unwrap_err();
    assert!(matches!(err, Error::LinearForwardReference { .. }), "{err}");
}

#[test_case(UOp::custom(smallvec![], "fence".into(), DType::Void), true; "void custom")]
#[test_case(store_at(&UOp::param(0, 4, DType::Float32, None), &i32c(0)), true; "store")]
#[test_case(i32c(0).barrier(smallvec![]), true; "barrier")]
#[test_case(UOp::custom(smallvec![], "add i32 0, 7".into(), DType::Int32), false; "typed custom")]
#[test_case(UOp::range_axis_dtype(i32c(4), AxisId::Renumbered(0), AxisType::Loop, DType::Int32), false; "range")]
fn only_void_statements_repeat(op: Arc<UOp>, repeats: bool) {
    match linear(vec![op.clone(), op.clone()]) {
        Ok(list) => {
            assert!(repeats);
            // A PARAM shape commit rebuilds a store, so count by structure.
            assert_eq!(list.iter().filter(|listed| listed.op().as_ref() == op.op().as_ref()).count(), 2);
        }
        Err(err) => assert!(!repeats && matches!(err, Error::LinearRepeat { .. }), "{err}"),
    }
}

#[test]
fn a_tag_makes_a_second_value() {
    let seven = UOp::custom(smallvec![], "add i32 0, 7".into(), DType::Int32);
    let again = seven.rtag(Some(smallvec![1]));
    let list = linear(vec![seven.clone(), again.clone()]).unwrap();
    assert!(position(&list, &seven) < position(&list, &again));
}

#[test]
fn a_program_closes_its_ranges() {
    let y = UOp::param(0, 4, DType::Float32, None);
    let k = UOp::range_axis_dtype(i32c(4), AxisId::Renumbered(0), AxisType::Loop, DType::Int32);
    // The barrier wraps the open store: only the list's position closes `k`.
    let store = store_at(&y, &k);
    let ops = vec![k.clone(), store.clone(), store.barrier(smallvec![]), store.end(smallvec![k])];
    let program = UOp::linear_program(KernelInfo::default(), ops, DeviceSpec::Cpu).unwrap();
    let Op::Program(ops::Program { sink, linear: Some(linear), .. }) = program.op() else { unreachable!() };
    assert!(!sink.in_scope_ranges().is_empty(), "the derived SINK holds the open barrier");
    assert!(linear.in_scope_ranges().is_empty() && program.in_scope_ranges().is_empty());
}
