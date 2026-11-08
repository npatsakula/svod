use ndarray::array;

use crate::{CumReduceOp, Tensor};

crate::codegen_tests! {
    fn test_cumsum_1d(config) {
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let result = x.cumsum(0).unwrap();
        result.realize_with(&config).unwrap();
        assert_eq!(result.as_vec::<f32>().unwrap(), [1.0, 3.0, 6.0, 10.0]);
    }

    fn test_cumsum_2d_axis0(config) {
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = x.cumsum(0).unwrap();
        result.realize_with(&config).unwrap();
        // Row 0: [1, 2, 3], Row 1: [1+4, 2+5, 3+6] = [5, 7, 9]
        assert_eq!(result.as_vec::<f32>().unwrap(), [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
    }

    fn test_cumsum_2d_axis1(config) {
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = x.cumsum(1).unwrap();
        result.realize_with(&config).unwrap();
        // Row 0: [1, 3, 6], Row 1: [4, 9, 15]
        assert_eq!(result.as_vec::<f32>().unwrap(), [1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }

    fn test_cumsum_negative_axis(config) {
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        // -1 = last axis = axis 1
        let result = x.cumsum(-1).unwrap();
        result.realize_with(&config).unwrap();
        assert_eq!(result.as_vec::<f32>().unwrap(), [1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }
}

/// Past 512 elements a cumulative reduce runs in two stages
/// ([`Tensor::_split_cumalu`]); its values are a host prefix reduce's at the
/// last one-stage length, the first two-stage one, an uneven chunk tail, each
/// op, and down an axis that is not the last.
#[test_case::test_case(CumReduceOp::Add, 512, false; "one stage at the split")]
#[test_case::test_case(CumReduceOp::Add, 513, false; "the first two-stage length")]
#[test_case::test_case(CumReduceOp::Add, 1000, false; "an uneven chunk tail")]
#[test_case::test_case(CumReduceOp::Mul, 1000, false; "a product")]
#[test_case::test_case(CumReduceOp::Max, 1000, false; "a maximum")]
#[test_case::test_case(CumReduceOp::Add, 600, true; "down a leading axis")]
fn a_long_cumulative_reduce_matches_a_host_prefix(reduce: CumReduceOp, n: usize, leading: bool) {
    let (cols, fold): (usize, fn(f32, f32) -> f32) = match reduce {
        CumReduceOp::Add => (if leading { 3 } else { 1 }, |a, b| a + b),
        CumReduceOp::Mul => (1, |a, b| a * b),
        CumReduceOp::Max => (1, f32::max),
    };
    // Exact in f32: small integers, signs, or a scramble of distinct values.
    let value = |i: usize| match reduce {
        CumReduceOp::Add => (i % 7) as f32 - 3.0,
        CumReduceOp::Mul if i.is_multiple_of(3) => -1.0,
        CumReduceOp::Mul => 1.0,
        CumReduceOp::Max => ((i * 37) % 101) as f32 - 50.0,
    };
    let xs: Vec<f32> = (0..n * cols).map(value).collect();
    let x = Tensor::from_slice(xs.as_slice()).try_reshape([n, cols]).unwrap();
    let got = x._split_cumalu(0, reduce).unwrap();
    got.realize().unwrap();
    let mut want = xs.clone();
    for i in cols..n * cols {
        want[i] = fold(want[i - cols], xs[i]);
    }
    assert_eq!(got.as_vec::<f32>().unwrap(), want);
}
