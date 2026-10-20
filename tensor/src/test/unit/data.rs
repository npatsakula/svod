use ndarray::{Array1, Array3, array};

use svod_dtype::DType;

use crate::Tensor;

// =========================================================================
// Codegen-required tests
// =========================================================================

crate::codegen_tests! {
    fn test_to_vec_computed(config) {
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([10.0f32, 20.0, 30.0]);
        let c = (&a + &b).unwrap();
        c.realize_with(&config).unwrap();
        let v = c.as_vec::<f32>().unwrap();
        assert_eq!(v, vec![11.0, 22.0, 33.0]);
    }
}

// === from_ndarray ===

#[test]
fn test_from_ndarray_1d() {
    let arr = array![1.0f32, 2.0, 3.0];
    let t = Tensor::from_ndarray(&arr);
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view.shape(), &[3]);
    assert_eq!(view, arr.view().into_dyn());
}

#[test]
fn test_from_ndarray_2d() {
    let arr = array![[1i32, 2], [3, 4]];
    let t = Tensor::from_ndarray(&arr);
    let view = t.array_view::<i32>().unwrap();
    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view, arr.view().into_dyn());
}

#[test]
fn test_from_ndarray_3d() {
    let arr = Array3::from_shape_vec((2, 1, 3), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let t = Tensor::from_ndarray(&arr);
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view.shape(), &[2, 1, 3]);
    assert_eq!(view, arr.view().into_dyn());
}

#[test]
fn test_from_ndarray_0d() {
    let arr = ndarray::arr0(42.0f32);
    let t = Tensor::from_ndarray(&arr);
    let v = t.as_vec::<f32>().unwrap();
    assert_eq!(v, vec![42.0]);
}

#[test]
fn test_from_ndarray_empty() {
    let arr = Array1::<f32>::zeros(0);
    let t = Tensor::from_ndarray(&arr);
    let result = t.as_ndarray::<f32>().unwrap();
    assert_eq!(result.shape(), &[0]);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_from_ndarray_view() {
    let arr = array![10.0f32, 20.0, 30.0];
    let view = arr.view();
    let t = Tensor::from_ndarray(&view);
    let tv = t.array_view::<f32>().unwrap();
    assert_eq!(tv, arr.view().into_dyn());
}

#[test]
fn test_from_ndarray_fortran() {
    // Fortran (column-major) — hits slow path, still correct logical order
    let arr = ndarray::Array2::from_shape_vec(ndarray::ShapeBuilder::f((2, 2)), vec![1.0f32, 3.0, 2.0, 4.0]).unwrap();
    let t = Tensor::from_ndarray(&arr);
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view, arr.view().into_dyn());
}

// === to_vec ===

#[test]
fn test_to_vec_f32() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let v = t.as_vec::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0]);
}

// === array_view ===

#[test]
fn test_array_view_basic() {
    let arr = array![[1.0f32, 2.0], [3.0, 4.0]];
    let t = Tensor::from_ndarray(&arr);
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view, arr.view().into_dyn());
}

#[test]
fn test_array_view_on_from_slice() {
    // from_slice retains buffer — no realize needed
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_array_view_unrealized() {
    let a = Tensor::from_slice([1.0f32, 2.0]);
    let b = Tensor::from_slice([3.0f32, 4.0]);
    let c = &a + &b; // lazy, no buffer
    assert!(c.unwrap().array_view::<f32>().is_err());
}

#[test]
fn test_array_view_dtype_mismatch() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    assert!(t.array_view::<i32>().is_err());
}

// === array_view_mut ===

#[test]
fn test_array_view_mut_write() {
    let t = Tensor::from_ndarray(&ndarray::Array2::<f32>::zeros((2, 3)));
    t.array_view_mut::<f32>().unwrap()[[1, 2]] = 42.0;
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view[[1, 2]], 42.0);
    assert_eq!(view[[0, 0]], 0.0);
}

#[test]
fn test_array_view_mut_fill() {
    let t = Tensor::from_slice([0.0f32; 4]);
    t.array_view_mut::<f32>().unwrap().fill(7.0);
    assert_eq!(t.array_view::<f32>().unwrap().as_slice().unwrap(), &[7.0; 4]);
}

#[test]
fn test_array_view_mut_unrealized() {
    let a = Tensor::from_slice([1.0f32, 2.0]);
    let c = (&a + &a).unwrap();
    assert!(c.array_view_mut::<f32>().is_err());
}

// === roundtrip ===

#[test]
fn test_roundtrip_ndarray() {
    let original = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let t = Tensor::from_ndarray(&original);
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view, original.view().into_dyn());
}

#[test]
fn test_roundtrip_vec() {
    let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let t = Tensor::from_slice(&original);
    let view = t.array_view::<f32>().unwrap();
    assert_eq!(view.as_slice().unwrap(), &original);
}

/// A narrowing cast rounds even when a widening one follows it lazily.
#[test_case::test_case(DType::Float16, 1.0 + 1.0 / 4096.0, 1.0; "f16 drops a bit past its mantissa")]
#[test_case::test_case(DType::Float16, 0.137_225_46, 281.0 / 2048.0; "f16 to nearest")]
#[test_case::test_case(DType::BFloat16, 0.137_225_46, 141.0 / 1024.0; "bf16 to nearest")]
fn test_a_narrowing_roundtrip_rounds(dtype: DType, value: f32, want: f32) {
    let got = Tensor::from_slice([value]).cast(dtype).cast(DType::Float32).to_vec::<f32>().unwrap();
    assert_eq!(got, [want]);
}

// =========================================================================
// Views must read their own elements, not the whole backing buffer
// =========================================================================

use proptest::prelude::*;

/// `[[1, 2, 3], [4, 5, 6]]`, sharing the 6-element buffer `from_slice` realized.
fn base_2x3() -> Tensor {
    Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape([2, 3]).unwrap()
}

#[test]
fn as_vec_on_a_shrunk_view_reads_only_the_view() {
    let view = base_2x3().try_shrink([(0, 1), (0, 3)]).unwrap();
    assert_eq!(view.dims().unwrap(), vec![1, 3]);
    assert_eq!(view.as_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn as_vec_on_an_offset_shrink_reads_the_right_elements() {
    let view = base_2x3().try_shrink([(1, 2), (1, 3)]).unwrap();
    assert_eq!(view.as_vec::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn as_ndarray_on_a_shrunk_view_matches_the_logical_shape() {
    let arr = base_2x3().try_shrink([(0, 2), (1, 3)]).unwrap().as_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[2, 2]);
    assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![2.0, 3.0, 5.0, 6.0]);
}

#[test]
fn as_vec_on_a_permuted_view_uses_logical_order() {
    let t = base_2x3().try_permute(&[1, 0]).unwrap();
    assert_eq!(t.dims().unwrap(), vec![3, 2]);
    assert_eq!(t.as_vec::<f32>().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn array_view_refuses_a_view_it_cannot_borrow_correctly() {
    // A zero-copy borrow can neither reorder nor subset the base buffer.
    let buffered = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert!(buffered.try_permute(&[1, 0]).unwrap().array_view::<f32>().is_err());
    assert!(buffered.try_shrink([(0, 1), (0, 3)]).unwrap().array_view_mut::<f32>().is_err());
    // The buffer identity it was built from stays viewable.
    assert_eq!(buffered.array_view::<f32>().unwrap().shape(), &[2, 3]);
}

#[test]
fn as_vec_still_refuses_an_unrealized_graph() {
    let a = Tensor::from_slice([1.0f32, 2.0]);
    let lazy = (&a + &a).unwrap();
    assert!(matches!(
        lazy.as_vec::<f32>().map_err(crate::error::Error::into_kind),
        Err(crate::error::ErrorKind::NoBuffer)
    ));
}

/// Random rank, extents, shrink window per axis and axis permutation.
fn view_case() -> impl Strategy<Value = (Vec<usize>, Vec<(usize, usize)>, Vec<usize>)> {
    (1usize..=3)
        .prop_flat_map(|rank| {
            (
                proptest::collection::vec(1usize..=4, rank),
                proptest::collection::vec((0usize..4, 0usize..4), rank),
                Just((0..rank).collect::<Vec<usize>>()).prop_shuffle(),
            )
        })
        .prop_map(|(dims, raw, perm)| {
            // Clamp each raw pair into a non-empty window `begin..end` of the axis.
            let ranges = dims
                .iter()
                .zip(&raw)
                .map(|(&d, &(a, b))| {
                    let begin = a % d;
                    (begin, begin + 1 + b % (d - begin))
                })
                .collect();
            (dims, ranges, perm)
        })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

    /// A shrink+permute view must read back exactly what ndarray computes for
    /// the same slice and transposition on the host.
    #[test]
    fn view_as_vec_matches_the_host_reference((dims, ranges, perm) in view_case()) {
        let data: Vec<f32> = (0..dims.iter().product::<usize>()).map(|i| i as f32).collect();

        let host = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), data.clone()).unwrap();
        let expected: Vec<f32> = host
            .slice_each_axis(|ax| {
                let (begin, end) = ranges[ax.axis.index()];
                ndarray::Slice::from(begin..end)
            })
            .permuted_axes(ndarray::IxDyn(&perm))
            .iter()
            .copied()
            .collect();

        let shape: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
        let windows: Vec<(isize, isize)> = ranges.iter().map(|&(b, e)| (b as isize, e as isize)).collect();
        let axes: Vec<isize> = perm.iter().map(|&a| a as isize).collect();

        let got = Tensor::from_slice(&data)
            .try_reshape(&shape).unwrap()
            .try_shrink(&windows).unwrap()
            .try_permute(&axes).unwrap()
            .as_vec::<f32>().unwrap();

        prop_assert_eq!(got, expected);
    }
}

// =========================================================================
// Auto-realizing reads: to_vec / to_ndarray / item
// =========================================================================

/// Nothing upstream is realized, so each read has to run the graph itself.
#[test_case::test_case(&[1.0f32, 2.0, 3.0], &[2.0, 4.0, 6.0]; "1d")]
#[test_case::test_case(&[1.0f32, -2.0, 0.0, 4.0], &[2.0, -4.0, 0.0, 8.0]; "signs")]
fn to_vec_realizes_a_lazy_graph(input: &[f32], expected: &[f32]) {
    let a = Tensor::from_slice(input);
    assert_eq!((&a + &a).unwrap().to_vec::<f32>().unwrap(), expected);
}

#[test]
fn to_ndarray_realizes_a_lazy_graph() {
    let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    let doubled = (&a + &a).unwrap().to_ndarray::<f32>().unwrap();
    assert_eq!(doubled, array![[2.0f32, 4.0], [6.0, 8.0]].into_dyn());
}

/// `as_*` must stay non-realizing so callers that cannot afford compilation
/// keep their guarantee, while `to_*` on the same graph succeeds.
#[test]
fn as_vec_refuses_what_to_vec_realizes() {
    let a = Tensor::from_slice([1.0f32, 2.0]);
    let lazy = (&a + &a).unwrap();
    assert!(matches!(lazy.as_vec::<f32>().map_err(crate::error::Error::into_kind), Err(crate::ErrorKind::NoBuffer)));
    assert_eq!(lazy.to_vec::<f32>().unwrap(), vec![2.0, 4.0]);
    assert_eq!(lazy.as_vec::<f32>().unwrap(), vec![2.0, 4.0], "to_vec leaves the tensor realized");
}

#[test_case::test_case(&[3.0f32], 3.0; "already scalar")]
#[test_case::test_case(&[1.0f32, 2.0, 3.0], 6.0; "reduced to scalar")]
fn item_reads_a_single_element(input: &[f32], expected: f32) {
    let a = Tensor::from_slice(input);
    let scalar = if input.len() == 1 { a } else { a.sum(()).unwrap() };
    assert_eq!(scalar.item::<f32>().unwrap(), expected);
}

#[test_case::test_case(&[1.0f32, 2.0], "2 elements"; "many")]
#[test_case::test_case(&[] as &[f32], "0 elements"; "empty")]
fn item_rejects_a_non_scalar(input: &[f32], actual: &str) {
    let err = Tensor::from_slice(input).item::<f32>().unwrap_err();
    assert!(
        matches!(err.kind(), crate::ErrorKind::ShapeMismatch { context, actual: got, .. } if context == "item" && got == actual),
        "{err:?}"
    );
}
