use crate::*;
use ndarray::array;
use svod_dtype::DType;
use svod_ir::SInt;

fn get_shape(tensor: &Tensor) -> Vec<usize> {
    tensor.uop().shape().unwrap().unwrap().iter().map(|s| s.as_const().unwrap()).collect()
}

// =========================================================================
// One-Hot Tests (codegen required)
// =========================================================================

crate::codegen_tests! {
    fn test_one_hot_along_dim_basic(config) {
        // [0, 1, 2] with 3 classes → 3x3 identity-like mask
        let idx = Tensor::from_slice([0i32, 1, 2]).try_unsqueeze(-1).unwrap();
        let result = idx.one_hot_along_dim(3, -1).unwrap();
        let shape = get_shape(&result);
        assert_eq!(shape, vec![3, 3]);
        let realized = result.contiguous();
        realized.realize_with(&config).unwrap();
        let view = realized.array_view::<bool>().unwrap();
        // Row 0: [true, false, false]
        assert!(view[[0, 0]]);
        assert!(!view[[0, 1]]);
        assert!(!view[[0, 2]]);
        // Row 1: [false, true, false]
        assert!(!view[[1, 0]]);
        assert!(view[[1, 1]]);
        assert!(!view[[1, 2]]);
        // Row 2: [false, false, true]
        assert!(!view[[2, 0]]);
        assert!(!view[[2, 1]]);
        assert!(view[[2, 2]]);
    }

    // =========================================================================
    // Scatter Tests
    // =========================================================================

    fn test_scatter_1d_basic(config) {
        // Create [0, 0, 0, 0, 0], scatter [10, 20, 30] at indices [1, 3, 0]
        let x = Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0, 0.0]);
        let idx = Tensor::from_slice([1i32, 3, 0]);
        let src = Tensor::from_slice([10.0f32, 20.0, 30.0]);
        let result = x.scatter(0, &idx, &src).unwrap().contiguous();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view[[0]], 30.0); // index 0 got 30
        assert_eq!(view[[1]], 10.0); // index 1 got 10
        assert_eq!(view[[3]], 20.0); // index 3 got 20
    }

    fn test_scatter_reduce_sum(config) {
        let x = Tensor::from_slice([0.0f32, 0.0, 0.0]);
        let idx = Tensor::from_slice([0i32, 0, 1]);
        let src = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let result = x
            .scatter_reduce(0, &idx, &src, crate::indexing::ScatterReduction::Sum, true)
            .unwrap()
            .contiguous();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        // index 0: 0 + 1 + 2 = 3, index 1: 0 + 3 = 3, index 2: 0
        assert_eq!(view[[0]], 3.0);
        assert_eq!(view[[1]], 3.0);
        assert_eq!(view[[2]], 0.0);
    }

    fn test_scatter_2d(config) {
        let x = Tensor::from_ndarray(&ndarray::Array2::<f32>::zeros((3, 2)));
        let idx = Tensor::from_ndarray(&array![[0i32, 1]]);
        let src = Tensor::from_ndarray(&array![[10.0f32, 20.0]]);
        let result = x.scatter(0, &idx, &src).unwrap().contiguous();
        result.realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![3, 2]);
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view[[0, 0]], 10.0);
        assert_eq!(view[[1, 1]], 20.0);
    }

    // =========================================================================
    // TopK Tests
    // =========================================================================

    fn test_topk_basic(config) {
        // 4 elements = n_stages=2 (power of 2) — larger sizes are very slow in debug builds
        let t = Tensor::from_slice([1.0f32, 4.0, 2.0, 3.0]);
        let (values, indices) = t.topk(2, -1, true).unwrap();
        let values = values.contiguous();
        values.realize_with(&config).unwrap();
        let indices = indices;
        indices.realize_with(&config).unwrap();
        assert_eq!(get_shape(&values), vec![2]);
        assert_eq!(get_shape(&indices), vec![2]);
        let view = values.array_view::<f32>().unwrap();
        assert_eq!(view[[0]], 4.0);
        assert_eq!(view[[1]], 3.0);
    }

    fn test_topk_smallest(config) {
        let t = Tensor::from_slice([1.0f32, 4.0, 2.0, 3.0]);
        let (values, _) = t.topk(2, -1, false).unwrap();
        let values = values.contiguous();
        values.realize_with(&config).unwrap();
        let view = values.array_view::<f32>().unwrap();
        assert_eq!(view[[0]], 1.0);
        assert_eq!(view[[1]], 2.0);
    }

    // =========================================================================
    // Masked Select Tests
    // =========================================================================

    fn test_masked_select_basic(config) {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let mask = Tensor::from_slice([true, false, true, false, true]);
        let result = t.masked_select(&mask).unwrap();
        result.realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![3]);
        assert_eq!(result.as_vec::<f32>().unwrap(), [1.0, 3.0, 5.0]);
    }

    fn test_masked_select_requires_bool(_config) {
        let tensor = Tensor::from_slice([1i32, 2]);
        let mask = Tensor::from_slice([1i32, 0]);
        assert!(matches!(
            tensor.masked_select(&mask).map_err(crate::error::Error::into_kind),
            Err(ErrorKind::TypeMismatch { expected, .. }) if expected == DType::Bool
        ));
    }

    // =========================================================================
    // NonZero Tests
    // =========================================================================

    fn test_nonzero_1d(config) {
        let t = Tensor::from_slice([1i32, 0, 2, 0, 3]);
        let result = t.nonzero().unwrap().contiguous();
        result.realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![3, 1]);
        let view = result.array_view::<i32>().unwrap();
        assert_eq!(view[[0, 0]], 0); // index of 1
        assert_eq!(view[[1, 0]], 2); // index of 2
        assert_eq!(view[[2, 0]], 4); // index of 3
    }

    fn test_nonzero_scalar_and_empty(config) {
        for (tensor, expected_shape) in [
            (Tensor::const_(1i32, DType::Int32), vec![1, 0]),
            (Tensor::const_(0i32, DType::Int32), vec![0, 0]),
            (Tensor::empty_zero(DType::Int32), vec![0, 1]),
        ] {
            let result = tensor.nonzero().unwrap().contiguous();
            result.realize_with(&config).unwrap();
            assert_eq!(get_shape(&result), expected_shape);
            assert!(result.as_vec::<i32>().unwrap().is_empty());
        }
    }

    fn test_nonzero_2d(config) {
        // [[1, 0], [1, 1]] — nonzero at (0,0), (1,0), (1,1)
        let t = Tensor::from_ndarray(&array![[1i32, 0], [1, 1]]);
        let result = t.nonzero().unwrap().contiguous();
        result.realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![3, 2]);
        let view = result.array_view::<i32>().unwrap();
        assert_eq!(view[[0, 0]], 0);
        assert_eq!(view[[0, 1]], 0);
        assert_eq!(view[[1, 0]], 1);
        assert_eq!(view[[1, 1]], 0);
        assert_eq!(view[[2, 0]], 1);
        assert_eq!(view[[2, 1]], 1);
    }

    fn test_nonzero_interior_singleton(config) {
        // Every interior axis needs the modulo, singleton dims included: without it
        // a `[2, 1, 3]` tensor reports `[1, 1, 0]` for the element at `(1, 0, 0)`.
        for dims in [vec![2usize, 1, 3], vec![1, 3], vec![3, 1, 1]] {
            let numel: usize = dims.iter().product();
            let shape = dims.iter().map(|&d| d as isize).collect::<Vec<_>>();
            let t = Tensor::from_slice(vec![1i32; numel]).try_reshape(shape).unwrap();
            let result = t.nonzero().unwrap().contiguous();
            result.realize_with(&config).unwrap();
            assert_eq!(get_shape(&result), vec![numel, dims.len()]);
            let mut expected = Vec::new();
            for flat in 0..numel {
                for axis in 0..dims.len() {
                    let stride: usize = dims[axis + 1..].iter().product();
                    expected.push((flat / stride % dims[axis]) as i32);
                }
            }
            assert_eq!(result.as_vec::<i32>().unwrap(), expected, "dims {dims:?}");
        }
    }

    // =========================================================================
    // Symbolic-batch indexing (the WavLM JIT path: dim 0 is a bound Variable)
    // =========================================================================

    fn test_index_select_symbolic_batch(config) {
        // `index_select` along dim 1 of a tensor whose batch dim (0) is a
        // symbolic JIT-style variable. Must equal the concrete-batch result.
        let data = Tensor::from_ndarray(&array![
            [[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
            [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]
        ]); // [3, 4, 2]
        let head = Tensor::from_slice([0i64, 2, 3]);

        let concrete = data.index_select(1, &head).unwrap().contiguous();
        concrete.realize_with(&config).unwrap();
        let want = concrete.as_vec::<f32>().unwrap();

        // Bind the batch dim to a symbolic variable (= 3); the shape carries
        // `SInt::Symbolic(BIND(var, 3))` through index_select → gather. `max ==
        // bind` so the realized buffer is exactly batch-sized (buffers are
        // otherwise allocated to the variable's max).
        let b = Variable::new("isb", 1, 3).bind(3).unwrap();
        let sym = data.try_shrink([Some((SInt::Const(0), b.as_sint())), None, None]).unwrap();
        let got = sym.index_select(1, &head).unwrap().contiguous();
        got.realize_with(&config).unwrap();
        // The result shape is still symbolic, so read the flat realized buffer.
        let got_flat: Vec<f32> = got.array_view::<f32>().unwrap().iter().copied().collect();

        assert_eq!(got_flat.len(), 3 * 3 * 2);
        assert_eq!(got_flat, want);
    }

    fn test_gather_symbolic_batch(config) {
        // `gather` with a symbolic batch dim on BOTH self and index (the only
        // legal symbolic non-gather extent: identical on both sides).
        let data = Tensor::from_ndarray(&array![
            [10.0f32, 11.0, 12.0, 13.0, 14.0],
            [20.0, 21.0, 22.0, 23.0, 24.0],
            [30.0, 31.0, 32.0, 33.0, 34.0]
        ]); // [3, 5]
        let idx = Tensor::from_ndarray(&array![[0i64, 4], [1, 3], [2, 0]]); // [3, 2]

        let concrete = data.gather(1, &idx).unwrap().contiguous();
        concrete.realize_with(&config).unwrap();
        let want = concrete.as_vec::<f32>().unwrap(); // [10,14, 21,23, 32,30]

        let b = Variable::new("gsb", 1, 3).bind(3).unwrap();
        let data_sym = data.try_shrink([Some((SInt::Const(0), b.as_sint())), None]).unwrap();
        let idx_sym = idx.try_shrink([Some((SInt::Const(0), b.as_sint())), None]).unwrap();
        let got = data_sym.gather(1, &idx_sym).unwrap().contiguous();
        got.realize_with(&config).unwrap();
        let got_flat: Vec<f32> = got.array_view::<f32>().unwrap().iter().copied().collect();

        assert_eq!(got_flat.len(), 3 * 2);
        assert_eq!(got_flat, want);
    }
}

// =========================================================================
// Gather Tests (shape/error/dtype only — no codegen)
// =========================================================================

#[test]
fn test_gather_1d_basic() {
    // Gather from 1D tensor
    let t = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0, 50.0]);
    let idx = Tensor::from_slice([2i64, 0, 4]); // Gather elements 2, 0, 4

    // Need to expand index to same rank as input (1D)
    let result = t.gather(0, &idx).unwrap();

    // Result shape should match index shape
    assert_eq!(get_shape(&result), vec![3]);
}

#[test]
fn test_gather_2d_dim0() {
    // Input shape [3, 4], index shape [2, 4]
    let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]);

    // Index must have same non-gather dim sizes
    let idx = Tensor::from_ndarray(&array![[0i64, 1, 2, 0], [1, 0, 1, 2]]);

    let result = t.gather(0, &idx).unwrap();
    assert_eq!(get_shape(&result), vec![2, 4]);
}

#[test]
fn test_gather_2d_dim1() {
    // Input shape [2, 5], index shape [2, 3]
    let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]);

    let idx = Tensor::from_ndarray(&array![[0i64, 2, 4], [1, 3, 0]]);

    let result = t.gather(1, &idx).unwrap();
    assert_eq!(get_shape(&result), vec![2, 3]);
}

#[test]
fn test_gather_negative_axis() {
    let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let idx = Tensor::from_ndarray(&array![[0i64], [2]]);

    // -1 = last axis
    let result = t.gather(-1, &idx).unwrap();
    assert_eq!(get_shape(&result), vec![2, 1]);
}

#[test]
fn test_gather_error_rank_mismatch() {
    let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    // Index has different rank (1D vs 2D)
    let idx = Tensor::from_slice([0i64, 1]);

    let result = t.gather(0, &idx);
    assert!(result.is_err());
}

#[test]
fn test_gather_error_dim_mismatch() {
    // Input [2, 3], gather along dim=1
    let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    // Index [3, 2] - non-gather dim 0 has size 3 > input size 2
    let idx = Tensor::from_ndarray(&array![[0i64, 1], [0, 1], [0, 1]]);

    let result = t.gather(1, &idx);
    assert!(result.is_err());
}

#[test]
fn test_gather_dtype_preserved() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let idx = Tensor::from_slice([0i64, 2, 4]);

    let result = t.gather(0, &idx).unwrap();

    // Result should have same dtype as input
    assert_eq!(result.uop().dtype(), DType::Float32);
}

// =========================================================================
// Shrink Tests (shape only — no codegen)
// =========================================================================

#[test]
fn test_shrink_1d() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);

    let sliced = t.try_shrink([(1, 4)]).unwrap();
    assert_eq!(get_shape(&sliced), vec![3]);
}

#[test]
fn test_shrink_2d() {
    let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let sliced = t.try_shrink([(0, 1), (1, 3)]).unwrap();
    assert_eq!(get_shape(&sliced), vec![1, 2]);
}

#[test]
fn test_shrink_negative_indices() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);

    // -3 to -1 should give elements [3, 4]
    let sliced = t.try_shrink([(-3, -1)]).unwrap();
    assert_eq!(get_shape(&sliced), vec![2]);
}

#[test]
fn test_shrink_full_dimension() {
    let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    // Keep full first dim, slice second
    let sliced = t.try_shrink([(0, 2), (1, 3)]).unwrap();
    assert_eq!(get_shape(&sliced), vec![2, 2]);
}

#[test]
fn test_shrink_empty_is_identity() {
    let t = Tensor::from_slice([1.0f32]);
    let sliced = t.try_shrink(&[] as &[(isize, isize)]).unwrap();
    assert_eq!(get_shape(&sliced), vec![1]);
}

// =========================================================================
// An index outside its arange's dtype names no position
// =========================================================================

/// The dtype an index tensor is built in. `u64` keeps the two's-complement bit
/// pattern of the same values, so `-1` is `u64::MAX`.
#[derive(Clone, Copy, Debug)]
enum IndexDtype {
    I32,
    I64,
    U64,
}

impl IndexDtype {
    fn dtype(self) -> DType {
        match self {
            Self::I32 => DType::Int32,
            Self::I64 => DType::Int64,
            Self::U64 => DType::UInt64,
        }
    }

    /// In-range and negative indices beside ones a 32-bit cast would fold onto a
    /// position: `2^32 + 3` onto 3, `-2^32` and `i64::MIN` onto 0.
    fn values(self) -> Vec<i128> {
        let wide =
            [(1i128 << 32) + 3, 1, -1, -(1 << 32), 3, (1 << 32) + 5, i64::MIN.into(), i64::MAX.into(), 4, 1 << 31];
        match self {
            Self::I64 => wide.to_vec(),
            Self::U64 => wide.iter().map(|&v| i128::from(v as i64 as u64)).collect(),
            Self::I32 => vec![i32::MAX.into(), 1, -1, i32::MIN.into(), 3, 5, 4, 2, 0, 6],
        }
    }

    fn tensor(self, values: &[i128]) -> Tensor {
        match self {
            Self::I32 => Tensor::from_slice(values.iter().map(|&v| v as i32).collect::<Vec<_>>()),
            Self::I64 => Tensor::from_slice(values.iter().map(|&v| v as i64).collect::<Vec<_>>()),
            Self::U64 => Tensor::from_slice(values.iter().map(|&v| v as u64).collect::<Vec<_>>()),
        }
    }
}

crate::codegen_tests! {
    /// `sum_j x[j] * (idx == start + j)` keeps the one step an index names, and an
    /// index outside the arange names none: the collapse that turns this masked sum
    /// into a gather must not read the index through a narrowing cast.
    #[test_case(IndexDtype::I64, 0; "i64 index")]
    #[test_case(IndexDtype::U64, 0; "u64 index")]
    #[test_case(IndexDtype::I32, 0; "i32 index")]
    #[test_case(IndexDtype::I64, 2; "i64 index, offset arange")]
    #[test_case(IndexDtype::U64, 2; "u64 index, offset arange")]
    #[test_case(IndexDtype::I32, 2; "i32 index, offset arange")]
    fn test_a_wide_index_never_aliases_onto_the_arange(config, index: IndexDtype, start: i64) {
        let x = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let values = index.values();
        let positions = Tensor::arange(start, Some(start + x.len() as i64), None).unwrap().cast(index.dtype());
        let mask = index.tensor(&values).try_unsqueeze(-1).unwrap().try_eq(&positions).unwrap();
        let out = Tensor::from_slice(x).try_unsqueeze(0).unwrap().where_(&mask, 0.0f32).unwrap().sum(-1isize).unwrap();
        out.realize_with(&config).unwrap();
        let expected: Vec<f32> = values
            .iter()
            .map(|&v| (0..x.len()).filter(|&j| v == i128::from(start) + j as i128).fold(0.0, |sum, j| sum + x[j]))
            .collect();
        assert_eq!(out.as_vec::<f32>().unwrap(), expected, "indices {values:?}");
    }
}
