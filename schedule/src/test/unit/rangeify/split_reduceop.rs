//! `split_reduceop`: two-stage reduction when the reduced extent is large enough
//! to be worth a materialised intermediate, plus the conditions that reject a
//! split outright.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{Op, ReduceOp, SInt, UOp, ops};
use test_case::test_case;

use crate::rangeify::kernel::{SplitReduceOpConfig, collect_range_ids, split_reduceop};
use crate::test::support::prelude::*;

fn tensor(shape: &[usize]) -> Arc<UOp> {
    let buffer = buffer_of(shape.iter().product(), svod_dtype::ScalarDType::Float32);
    match shape {
        [_] => buffer,
        _ => buffer.try_reshape(&shape.iter().map(|&s| SInt::Const(s)).collect()).expect("reshape"),
    }
}

fn expanded(base: &[usize], to: &[usize]) -> Arc<UOp> {
    let new_shape = stack(to.iter().map(|&d| UOp::index_const(d as i64)));
    UOp::new(Op::Expand(ops::Expand { src: tensor(base), new_shape }), DType::Float32)
}

fn has_contiguous(uop: &Arc<UOp>) -> bool {
    has_op(uop, |op| matches!(op, Op::Contiguous(..)))
}

fn split(source: &Arc<UOp>, axis: usize, config: &SplitReduceOpConfig) -> Option<Arc<UOp>> {
    split_reduceop(&source.try_reduce_axis(ReduceOp::Add, vec![axis]).expect("reduce axis"), config)
}

/// Ratio of total elements to output elements decides the split; the default
/// threshold is 32768. A broadcast (EXPAND) axis is never a split candidate —
/// splitting it would materialise the same value repeatedly — and a movement
/// chain over one still splits once it is pushed through.
#[test_case(tensor(&[1_000]), 0, false ; "1d below threshold")]
#[test_case(tensor(&[100_000]), 0, true ; "1d above threshold")]
#[test_case(tensor(&[1_000, 1_000]), 1, false ; "2d ratio 1000 is below threshold")]
#[test_case(tensor(&[1_000, 100_000]), 1, true ; "2d ratio 100000 is above threshold")]
#[test_case(expanded(&[100, 1, 1_000], &[100, 500, 1_000]), 1, false ; "the reduced axis is the broadcast one")]
#[test_case(expanded(&[100, 1, 100_000], &[100, 50, 100_000]), 2, true ; "another axis is broadcast")]
#[test_case(flattened_expand(), 0, true ; "a movement chain hides the extent")]
fn a_reduction_splits_once_its_ratio_clears_the_threshold(source: Arc<UOp>, axis: usize, splits: bool) {
    let reduce = source.try_reduce_axis(ReduceOp::Add, vec![axis]).expect("reduce axis");

    match split_reduceop(&reduce, &SplitReduceOpConfig::default()) {
        Some(transformed) => {
            assert!(splits, "unexpected split: {}", transformed.tree());
            assert!(has_contiguous(&transformed), "the split must materialise its intermediate");
            assert_eq!(
                transformed.shape().expect("shape").expect("static").len(),
                reduce.shape().expect("shape").expect("static").len(),
                "the split must not change the output rank"
            );
        }
        None => assert!(!splits, "expected a split"),
    }
}

/// `RESHAPE(EXPAND(RESHAPE(buffer)))` flattened to one axis.
fn flattened_expand() -> Arc<UOp> {
    expanded(&[50, 1], &[50, 1_000]).try_reshape(&smallvec::smallvec![SInt::Const(50_000)]).expect("reshape")
}

fn broadcast(src: Arc<UOp>, base: &[usize], to: &[usize]) -> Arc<UOp> {
    let dtype = src.dtype();
    let src = src.try_reshape(&base.iter().map(|&s| SInt::Const(s)).collect()).expect("reshape");
    let new_shape = stack(to.iter().map(|&d| UOp::index_const(d as i64)));
    UOp::new(Op::Expand(ops::Expand { src, new_shape }), dtype)
}

/// `[rows, indices, width]` of a lookup small enough to build its arange by hand.
const LOOKUP: [usize; 3] = [64, 4, 8];

fn int_expanded(base: &[usize]) -> Arc<UOp> {
    broadcast(buffer_of(base.iter().product(), svod_dtype::ScalarDType::Int32), base, &LOOKUP)
}

/// `0..n` along `axis` of [`LOOKUP`], in `dtype`: varies along that axis and
/// reads no buffer, as `Tensor::arange` does.
fn arange_along(axis: usize, dtype: DType) -> Arc<UOp> {
    let mut base = [1; 3];
    base[axis] = LOOKUP[axis];
    broadcast(stack((0..LOOKUP[axis]).map(|i| UOp::index_const(i as i64))).cast(dtype), &base, &LOOKUP)
}

/// `Tensor::embedding`'s lookup, `where(idx == arange, table, 0)` summed over
/// the rows, and the variations of it that reduce simplification cannot fold.
#[derive(Clone, Copy)]
enum Lookup {
    Where,
    Mul,
    Ne,
    Lt,
    OneElsewhere,
    Max,
    CastBeforeTheSum,
    AgainstAnotherAxis,
    IndexVariesToo,
    ArangeReadsABuffer,
    ArangeWidenedPastTheIndex,
}

impl Lookup {
    fn reduce(self) -> Arc<UOp> {
        let [rows, indices, width] = LOOKUP;
        let index = int_expanded(&[1, indices, 1]);
        let arange = arange_along(0, DType::Int32);
        let table = expanded(&[rows, 1, width], &LOOKUP);
        let fill = |value: f32| broadcast(UOp::native_const(value), &[1, 1, 1], &LOOKUP);
        let eq = |lhs: &Arc<UOp>, rhs: &Arc<UOp>| lhs.try_cmpeq(rhs).expect("eq");
        let select = |condition| UOp::try_where(condition, table.clone(), fill(0.0)).expect("where");
        let source = match self {
            Self::Where | Self::Max => select(eq(&index, &arange)),
            Self::Mul => table.try_mul(&eq(&index, &arange).cast(DType::Float32)).expect("mul"),
            Self::Ne => UOp::try_where(index.try_cmpne(&arange).expect("ne"), fill(0.0), table).expect("where"),
            Self::Lt => select(index.try_cmplt(&arange).expect("lt")),
            Self::OneElsewhere => UOp::try_where(eq(&index, &arange), table, fill(1.0)).expect("where"),
            Self::CastBeforeTheSum => select(eq(&index, &arange)).cast(DType::Float64),
            Self::AgainstAnotherAxis => select(eq(&index, &arange_along(2, DType::Int32))),
            Self::IndexVariesToo => select(eq(&int_expanded(&[rows, indices, 1]), &arange)),
            Self::ArangeReadsABuffer => select(eq(&index, &int_expanded(&[rows, 1, 1]))),
            Self::ArangeWidenedPastTheIndex => select(eq(&index, &arange_along(0, DType::Int16).cast(DType::Int32))),
        };
        let reduce_op = if matches!(self, Self::Max) { ReduceOp::Max } else { ReduceOp::Add };
        source.try_reduce_axis(reduce_op, vec![0]).expect("reduce axis")
    }
}

/// A lookup reduce simplification folds into one indexed load stays one
/// reduce: split, a 64-bit index could not fold, and the first stage would read
/// the whole table for every index. Every other reduce still splits.
#[test_case(Lookup::Where, true ; "where(idx == arange, x, 0)")]
#[test_case(Lookup::Mul, true ; "x * (idx == arange)")]
#[test_case(Lookup::Ne, true ; "where(idx != arange, 0, x)")]
#[test_case(Lookup::Lt, false ; "where(idx < arange, x, 0) selects a prefix")]
#[test_case(Lookup::OneElsewhere, false ; "where(idx == arange, x, 1) is not a select")]
#[test_case(Lookup::Max, false ; "a max does not fold")]
#[test_case(Lookup::CastBeforeTheSum, false ; "a cast before the sum keeps it from folding")]
#[test_case(Lookup::AgainstAnotherAxis, false ; "the arange runs along another axis")]
#[test_case(Lookup::IndexVariesToo, false ; "the index varies along the reduced axis")]
#[test_case(Lookup::ArangeReadsABuffer, false ; "the compared side reads a buffer")]
#[test_case(Lookup::ArangeWidenedPastTheIndex, false ; "the arange is widened past the index")]
fn a_lookup_that_folds_into_a_load_is_not_split(lookup: Lookup, whole: bool) {
    let config = SplitReduceOpConfig { split_threshold: 8, ..Default::default() };
    assert_eq!(split_reduceop(&lookup.reduce(), &config).is_none(), whole);
}

#[test_case(ReduceOp::Add ; "add")]
#[test_case(ReduceOp::Mul ; "mul")]
#[test_case(ReduceOp::Max ; "max")]
#[test_case(ReduceOp::Min ; "min")]
fn the_split_keeps_the_original_reduce_op(reduce_op: ReduceOp) {
    let reduce = tensor(&[100_000]).try_reduce_axis(reduce_op, vec![0]).expect("reduce axis");

    let transformed = split_reduceop(&reduce, &SplitReduceOpConfig::default()).expect("split");
    assert!(
        has_op(&transformed, |op| matches!(op, Op::Reduce(ops::Reduce { reduce_op: op, .. }) if *op == reduce_op)),
        "{reduce_op:?} must survive the split"
    );
}

#[test]
fn the_split_can_be_turned_off() {
    let config = SplitReduceOpConfig { enabled: false, ..Default::default() };
    assert!(split(&tensor(&[100_000]), 0, &config).is_none());
}

// ===== bail-out conditions =====

/// A REDUCE with no axes and one whose ranges are already closed are not
/// tensor-form reductions and cannot be re-shaped into two stages.
#[test]
fn axeless_or_ranged_reductions_do_not_split() {
    let source = tensor(&[100_000]);
    let no_axes = UOp::new(
        Op::Reduce(ops::Reduce {
            src: source.clone(),
            ranges: smallvec::smallvec![],
            reduce_op: ReduceOp::Add,
            num_axes: 0,
        }),
        DType::Float32,
    );
    let with_ranges = UOp::new(
        Op::Reduce(ops::Reduce {
            src: source,
            ranges: smallvec::smallvec![global_range(100_000, 0)],
            reduce_op: ReduceOp::Add,
            num_axes: 1,
        }),
        DType::Float32,
    );

    for reduce in [no_axes, with_ranges] {
        assert!(split_reduceop(&reduce, &SplitReduceOpConfig::default()).is_none(), "{}", reduce.tree());
    }
}

/// The output-size cap is what rejects a candidate, and `output_size_bits` alone
/// decides the cap; the other defaults are the documented schedule policy.
#[test]
fn the_output_cap_rejects_every_divisor_when_it_is_too_small() {
    let config = SplitReduceOpConfig { output_size_bits: 4, ..Default::default() };
    assert_eq!(config.max_output_size(), 16);
    assert!(split(&tensor(&[1_000, 100_000]), 1, &config).is_none());

    let default = SplitReduceOpConfig::default();
    assert_eq!((default.split_threshold, default.max_divisor, default.min_divisor), (32768, 256, 8));
    assert_eq!(default.max_output_size(), 1 << default.output_size_bits);
    let narrower = SplitReduceOpConfig { output_size_bits: 20, ..Default::default() };
    assert_eq!(narrower.max_output_size(), 1 << 20);
}

/// A split needs a divisor of the reduced dimension: a prime extent has none and
/// a symbolic extent cannot be divided at schedule time, so both are left alone.
/// A non-REDUCE can never be split either.
#[test]
fn nondivisible_symbolic_and_nonreduce_inputs_do_not_split() {
    assert!(split(&tensor(&[100_003]), 0, &SplitReduceOpConfig::default()).is_none());

    let size = UOp::var("size", DType::Int32, 1, i64::MAX);
    let symbolic = UOp::new_buffer(svod_device::DeviceSpec::Cpu, 1, DType::Float32)
        .try_reshape(&smallvec::smallvec![SInt::Symbolic(size)])
        .expect("reshape");
    let reduce = symbolic.try_reduce_axis(ReduceOp::Add, vec![0]).expect("reduce axis");
    assert!(split_reduceop(&reduce, &SplitReduceOpConfig::default()).is_none());

    assert!(split_reduceop(&tensor(&[100_000]), &SplitReduceOpConfig::default()).is_none());
}

fn range_ids(ranges: &[(i64, usize)]) -> Vec<usize> {
    let uops: smallvec::SmallVec<[Arc<UOp>; 4]> = ranges.iter().map(|&(end, id)| global_range(end, id)).collect();
    let expr = uops.iter().skip(1).fold(uops[0].clone(), |acc, r| acc.try_add(r).expect("add"));
    collect_range_ids(&expr)
}

/// `collect_range_ids` returns every RANGE axis in the expression, sorted.
#[test]
fn range_ids_come_back_sorted() {
    assert_eq!(collect_range_ids(&UOp::native_const(1.0f32)), Vec::<usize>::new());
    assert_eq!(range_ids(&[(10, 0)]), vec![0]);
    assert_eq!(range_ids(&[(10, 0), (5, 1), (3, 2)]), vec![0, 1, 2]);
    assert_eq!(range_ids(&[(3, 2), (10, 0), (5, 1)]), vec![0, 1, 2], "source order does not matter");
}
