use crate::reduce::AxisSpec;
use crate::test::helpers::*;
use crate::*;
use ndarray::array;
use svod_dtype::DType;
use svod_ir::Op;
use svod_ir::ops;
use test_case::test_case;

#[test]
fn test_axis_spec_all() {
    let axes = Tensor::resolve_axis_spec(&AxisSpec::All, 3).unwrap();
    assert_eq!(axes, vec![0, 1, 2]);
}

#[test]
fn test_axis_spec_single() {
    // Positive index
    let axes = Tensor::resolve_axis_spec(&AxisSpec::Single(1), 3).unwrap();
    assert_eq!(axes, vec![1]);

    // Negative index
    let axes = Tensor::resolve_axis_spec(&AxisSpec::Single(-1), 3).unwrap();
    assert_eq!(axes, vec![2]);

    // Out of bounds
    assert!(Tensor::resolve_axis_spec(&AxisSpec::Single(5), 3).is_err());
    assert!(Tensor::resolve_axis_spec(&AxisSpec::Single(-5), 3).is_err());
}

#[test]
fn test_axis_spec_multiple() {
    // Multiple axes
    let axes = Tensor::resolve_axis_spec(&AxisSpec::Multiple(vec![0, 2]), 3).unwrap();
    assert_eq!(axes, vec![0, 2]);

    // With negatives
    let axes = Tensor::resolve_axis_spec(&AxisSpec::Multiple(vec![0, -1]), 3).unwrap();
    assert_eq!(axes, vec![0, 2]);

    // Deduplication
    let axes = Tensor::resolve_axis_spec(&AxisSpec::Multiple(vec![1, 1, 1]), 3).unwrap();
    assert_eq!(axes, vec![1]);
}

#[test]
fn test_sum_acc_dtype() {
    assert_eq!(Tensor::sum_acc_dtype(&DType::Bool), DType::Int32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::Int8), DType::Int32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::Int16), DType::Int32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::Int32), DType::Int32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::Int64), DType::Int64);

    assert_eq!(Tensor::sum_acc_dtype(&DType::UInt8), DType::UInt32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::UInt16), DType::UInt32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::UInt32), DType::UInt32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::UInt64), DType::UInt64);

    assert_eq!(Tensor::sum_acc_dtype(&DType::Float16), DType::Float32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::BFloat16), DType::Float32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::Float32), DType::Float32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::Float64), DType::Float64);
}

#[test]
fn test_keepdim_wraps_squeezed_tensor_reduce_in_reshape() {
    let tensor = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape([2, 3]).unwrap();
    let squeezed = tensor.sum(1).unwrap();
    let kept = tensor.sum_with().axes(1).keepdim(true).call().unwrap();

    assert!(matches!(squeezed.uop().op(), Op::Reduce(ops::Reduce { ranges, num_axes: 1, .. }) if ranges.is_empty()));
    assert_eq!(squeezed.shape().unwrap().as_slice(), &[SInt::Const(2)]);
    assert!(matches!(kept.uop().op(), Op::Reshape(ops::Reshape { src, .. })
        if matches!(src.op(), Op::Reduce(ops::Reduce { ranges, num_axes: 1, .. }) if ranges.is_empty())));
    assert_eq!(kept.shape().unwrap().as_slice(), &[SInt::Const(2), SInt::Const(1)]);
}

// ========== Argmax Tests ==========

#[test]
fn test_argmax_1d_basic() {
    let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 5.0, 4.0]);
    let result = t.argmax(Some(0)).unwrap();
    // Max value 5.0 is at index 3
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 0); // Scalar result
    assert_eq!(result.uop().dtype(), DType::Int32);
}

#[test]
fn test_argmax_1d_ties_first_occurrence() {
    let t = Tensor::from_slice([1.0f32, 5.0, 3.0, 5.0, 2.0]);
    let result = t.argmax(Some(0)).unwrap();
    // Two maxima at indices 1 and 3, should return 1 (first)
    assert_eq!(result.uop().dtype(), DType::Int32);
}

#[test]
fn test_argmax_2d_axis0() {
    let t = Tensor::from_ndarray(&array![[1.0f32, 3.0, 2.0], [4.0, 2.0, 5.0]]);
    // Shape [2, 3]: [[1.0, 3.0, 2.0], [4.0, 2.0, 5.0]]
    let result = t.argmax(Some(0)).unwrap();
    // Expected: [1, 0, 1] (max per column)
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 1);
    assert_eq!(result_shape[0].as_const().unwrap(), 3);
}

#[test]
fn test_argmax_2d_axis1() {
    let t = Tensor::from_ndarray(&array![[1.0f32, 3.0, 2.0], [4.0, 2.0, 5.0]]);
    let result = t.argmax(Some(1)).unwrap();
    // Expected: [1, 2] (max per row)
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 1);
    assert_eq!(result_shape[0].as_const().unwrap(), 2);
}

#[test]
fn test_argmax_flatten() {
    let t = Tensor::from_ndarray(&array![[1.0f32, 3.0, 2.0], [4.0, 2.0, 5.0]]);
    let result = t.argmax(None).unwrap();
    // Flattened: [1, 3, 2, 4, 2, 5], max 5.0 at index 5
    assert_eq!(result.shape().unwrap().len(), 0); // Scalar
}

#[test]
fn test_argmax_keepdim() {
    let t = Tensor::from_ndarray(&array![[1.0f32, 3.0, 2.0], [4.0, 2.0, 5.0]]);
    let result = t.argmax_with().axis(Some(1)).keepdim(true).call().unwrap();
    // Shape should be [2, 1] instead of [2]
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert_eq!(result_shape[0].as_const().unwrap(), 2);
    assert_eq!(result_shape[1].as_const().unwrap(), 1);
}

#[test]
fn test_argmax_negative_axis() {
    let t = Tensor::from_ndarray(&array![[1.0f32, 3.0, 2.0], [4.0, 2.0, 5.0]]);
    let result = t.argmax(Some(-1)).unwrap();
    // -1 should resolve to axis 1
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 1);
    assert_eq!(result_shape[0].as_const().unwrap(), 2);
}

// ========== Argmin Tests ==========

#[test]
fn test_argmin_1d_basic() {
    let t = Tensor::from_slice([5.0f32, 3.0, 1.0, 4.0, 2.0]);
    let result = t.argmin(Some(0)).unwrap();
    // Min value 1.0 is at index 2
    assert_eq!(result.uop().dtype(), DType::Int32);
}

#[test]
fn test_argmin_float() {
    let t = Tensor::from_slice([1.5f32, -2.3, 0.5, 1.0]);
    let result = t.argmin(Some(0)).unwrap();
    // Min is -2.3 at index 1
    assert_eq!(result.uop().dtype(), DType::Int32);
}

#[test]
fn test_argmin_int() {
    let t = Tensor::from_slice([5i32, 3, 1, 4, 2]);
    let result = t.argmin(Some(0)).unwrap();
    // Min is 1 at index 2
    assert_eq!(result.uop().dtype(), DType::Int32);
}

#[test]
fn test_argmin_bool() {
    let t = Tensor::from_slice([true, false, true]);
    let result = t.argmin(Some(0)).unwrap();
    // Min (false) at index 1
    assert_eq!(result.uop().dtype(), DType::Int32);
}

// ========== Any Tests ==========

#[test]
fn test_any_all_true() {
    let t = Tensor::from_slice([true, true, true]);
    let result = t.any(()).unwrap();
    // Should be true
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_any_all_false() {
    let t = Tensor::from_slice([false, false, false]);
    let result = t.any(()).unwrap();
    // Should be false
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_any_mixed() {
    let t = Tensor::from_slice([false, true, false]);
    let result = t.any(()).unwrap();
    // Should be true
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_any_numeric() {
    let t = Tensor::from_slice([0.0f32, 1.0, 0.0]);
    let result = t.any(()).unwrap();
    // Non-zero treated as true, should be true
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_any_2d_axis0() {
    let t = Tensor::from_ndarray(&array![[true, false], [false, false]]);
    let result = t.any(0).unwrap();
    // Expected: [true, false]
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 1);
    assert_eq!(result_shape[0].as_const().unwrap(), 2);
}

#[test]
fn test_any_2d_axis1() {
    let t = Tensor::from_ndarray(&array![[true, false], [false, false]]);
    let result = t.any(1).unwrap();
    // Expected: [true, false]
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 1);
    assert_eq!(result_shape[0].as_const().unwrap(), 2);
}

#[test]
fn test_any_keepdim() {
    let t = Tensor::from_ndarray(&array![[true, false], [false, true]]);
    let result = t.any_with().axes(0).keepdim(true).call().unwrap();
    // Shape should be [1, 2]
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert_eq!(result_shape[0].as_const().unwrap(), 1);
    assert_eq!(result_shape[1].as_const().unwrap(), 2);
}

// ========== All Tests ==========

#[test]
fn test_all_all_true() {
    let t = Tensor::from_slice([true, true, true]);
    let result = t.all(()).unwrap();
    // Should be true
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_all_one_false() {
    let t = Tensor::from_slice([true, false, true]);
    let result = t.all(()).unwrap();
    // Should be false
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_all_numeric() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let result = t.all(()).unwrap();
    // All non-zero, should be true
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_all_numeric_with_zero() {
    let t = Tensor::from_slice([1.0f32, 0.0, 3.0]);
    let result = t.all(()).unwrap();
    // Has zero, should be false
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_all_2d_multiaxis() {
    let t = Tensor::from_ndarray(&array![[true, true], [true, true]]);
    let result = t.all(()).unwrap();
    // All true across all axes
    assert_eq!(result.uop().dtype(), DType::Bool);
}

// ========== Edge Cases ==========

#[test]
fn test_argmax_single_element() {
    let t = Tensor::from_slice([42.0f32]);
    let result = t.argmax(Some(0)).unwrap();
    // Only element, index should be 0
    assert_eq!(result.uop().dtype(), DType::Int32);
}

#[test]
fn test_argmax_all_equal() {
    let t = Tensor::from_slice([5.0f32, 5.0, 5.0, 5.0]);
    let result = t.argmax(Some(0)).unwrap();
    // All equal, should return first index (0)
    assert_eq!(result.uop().dtype(), DType::Int32);
}

// ============================================================================
// VALUE-VERIFYING TESTS (ported from Tinygrad test_ops.py)
// ============================================================================

crate::codegen_tests! {
    /// `min` ignores NaN exactly like `max` instead of keeping whatever
    /// element follows the NaN.
    fn test_min_and_max_ignore_nan_alike(config) {
        let x = Tensor::from_slice([3.0f32, f32::NAN, 5.0, 4.0]);
        let min = x.min(0).unwrap();
        min.realize_with(&config).unwrap();
        let max = x.max(0).unwrap();
        max.realize_with(&config).unwrap();
        assert_eq!((min.as_vec::<f32>().unwrap(), max.as_vec::<f32>().unwrap()), (vec![3.0], vec![5.0]));
    }
}

crate::codegen_tests! {
    // ========== Sum Tests (from Tinygrad test_ops.py:1344-1380) ==========

    fn test_sum_1d_value(config) {
        test_setup();
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        assert_close_f32(&t.sum(()).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[10.0], 1e-6);
    }

    fn test_sum_2d_full_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_close_f32(&t.sum(()).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[21.0], 1e-6);
    }

    fn test_sum_axis0_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        // [[1, 2, 3], [4, 5, 6]] -> [1+4, 2+5, 3+6] = [5, 7, 9]
        assert_close_f32(&t.sum(0).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[5.0, 7.0, 9.0], 1e-6);
    }

    fn test_sum_axis1_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        // [[1, 2, 3], [4, 5, 6]] -> [1+2+3, 4+5+6] = [6, 15]
        assert_close_f32(&t.sum(1).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[6.0, 15.0], 1e-6);
    }

    fn test_sum_keepdim_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
        let result = t.sum_with().axes(1).keepdim(true).call().unwrap();
        let shape = result.shape().unwrap();
        assert_eq!(shape[0].as_const().unwrap(), 2);
        assert_eq!(shape[1].as_const().unwrap(), 1);
        // [[1, 2], [3, 4]] -> [[3], [7]]
        assert_close_f32(&result.realize_with_and(&config).as_vec::<f32>().unwrap(), &[3.0, 7.0], 1e-6);
    }

    fn test_sum_empty_axis_preserves_nonreduced_shape(config) {
        test_setup();
        let t = Tensor::empty(&[0, 3], DType::Float32);
        let squeezed = t.sum(0).unwrap();
        assert_eq!(squeezed.shape().unwrap().as_slice(), &[SInt::Const(3)]);
        assert_eq!(squeezed.realize_with_and(&config).as_vec::<f32>().unwrap(), [0.0; 3]);

        let kept = t.sum_with().axes(0).keepdim(true).call().unwrap();
        assert_eq!(kept.shape().unwrap().as_slice(), &[SInt::Const(1), SInt::Const(3)]);
        assert_eq!(kept.realize_with_and(&config).as_vec::<f32>().unwrap(), [0.0; 3]);
    }

    fn test_sum_single_element_value(config) {
        test_setup();
        let t = Tensor::from_slice([42.0f32]);
        assert_close_f32(&t.sum(()).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[42.0], 1e-6);
    }

    fn test_sum_negative_values(config) {
        test_setup();
        let t = Tensor::from_slice([-1.0f32, -2.0, 3.0, 4.0]);
        assert_close_f32(&t.sum(()).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[4.0], 1e-6);
    }

    // ========== Max Tests (from Tinygrad test_ops.py:1405-1416) ==========

    fn test_max_1d_value(config) {
        test_setup();
        let t = Tensor::from_slice([1.0f32, 5.0, 3.0, 2.0]);
        assert_close_f32(&t.max(()).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[5.0], 1e-6);
    }

    fn test_max_2d_axis1_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[1.0f32, 5.0, 3.0], [2.0, 8.0, 4.0]]);
        // [[1, 5, 3], [2, 8, 4]] -> [5, 8]
        assert_close_f32(&t.max(1).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[5.0, 8.0], 1e-6);
    }

    fn test_max_2d_axis0_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[1.0f32, 5.0, 3.0], [2.0, 8.0, 4.0]]);
        // [[1, 5, 3], [2, 8, 4]] -> [2, 8, 4]
        assert_close_f32(&t.max(0).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[2.0, 8.0, 4.0], 1e-6);
    }

    fn test_max_negative_values(config) {
        test_setup();
        let t = Tensor::from_slice([-5.0f32, -1.0, -3.0, -2.0]);
        assert_close_f32(&t.max(()).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[-1.0], 1e-6);
    }

    // ========== Min Tests (from Tinygrad test_ops.py:1394-1403) ==========

    fn test_min_1d_value(config) {
        test_setup();
        let t = Tensor::from_slice([5.0f32, 1.0, 3.0, 2.0]);
        assert_close_f32(&t.min(()).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[1.0], 1e-6);
    }

    fn test_min_2d_axis1_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[1.0f32, 5.0, 3.0], [2.0, 8.0, 4.0]]);
        // [[1, 5, 3], [2, 8, 4]] -> [1, 2]
        assert_close_f32(&t.min(1).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[1.0, 2.0], 1e-6);
    }

    fn test_min_negative_value(config) {
        test_setup();
        let t = Tensor::from_slice([-1.0f32, -5.0, -3.0]);
        assert_close_f32(&t.min(()).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[-5.0], 1e-6);
    }

    fn test_empty_float_minmax_use_the_reduction_identity(config) {
        test_setup();
        let t = Tensor::empty(&[0], DType::Float32);
        assert_eq!(t.max(()).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), [f32::NEG_INFINITY]);
        assert_eq!(t.min(()).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), [f32::INFINITY]);
    }

    // ========== Argmax Tests (from Tinygrad test_ops.py:1087-1105) ==========

    fn test_argmax_debug_steps(config) {
        test_setup();

        // Simplest test: compare two tensors
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([1.0f32, 5.0, 3.0]);
        let eq = a.try_eq(&b).unwrap();
        assert_eq!(eq.realize_with_and(&config).as_vec::<bool>().unwrap(), [true, false, true], "Simple eq failed");

        // Test with broadcast
        let c = Tensor::from_slice([1.0f32, 2.0, 3.0, 2.0]);
        let two = Tensor::from_slice([2.0f32]);
        let two_broadcast = two.try_expand([4]).unwrap();
        let eq2 = c.try_eq(&two_broadcast).unwrap();
        // Expected: [false, true, false, true] (positions 1 and 3 equal 2)
        assert_eq!(eq2.realize_with_and(&config).as_vec::<bool>().unwrap(), [false, true, false, true], "Broadcast eq failed");

        // Test expand of reduction result
        let d = Tensor::from_slice([1.0f32, 5.0, 3.0, 2.0]);
        let d_max = d.max_with().axes(0).keepdim(true).call().unwrap();
        d_max.realize_with(&config).unwrap();
        let d_max_expanded = d_max.try_expand([4]).unwrap();
        let eq3 = d.try_eq(&d_max_expanded).unwrap();
        // Expected: [false, true, false, false] (only position 1 equals 5)
        assert_eq!(eq3.realize_with_and(&config).as_vec::<bool>().unwrap(), [false, true, false, false], "Reduction expand eq failed");
    }

    fn test_argmax_full_steps(config) {
        test_setup();
        let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 5.0, 4.0]);

        // Step 1: max value along axis 0
        let max_vals = t.max_with().axes(0).keepdim(true).call().unwrap();

        // Step 2: expand max to original shape
        let max_broadcast = max_vals.try_expand([5]).unwrap();

        // Step 3: mask where values == max
        let mask = t.try_eq(&max_broadcast).unwrap();
        assert_eq!(mask.clone().realize_with_and(&config).as_vec::<bool>().unwrap(), [false, false, false, true, false], "Mask mismatch");

        // Step 4: Create descending indices [5, 4, 3, 2, 1]
        let axis_size = 5;
        let indices = Tensor::arange(axis_size as i64, Some(0), Some(-1)).unwrap();
        assert_eq!(indices.clone().realize_with_and(&config).as_vec::<i32>().unwrap(), [5i32, 4, 3, 2, 1], "Indices mismatch");

        // Step 5: Cast to int32
        let mask_int = mask.cast(DType::Int32);
        // Expected: [0, 0, 0, 1, 0]
        assert_eq!(mask_int.clone().realize_with_and(&config).as_vec::<i32>().unwrap(), [0, 0, 0, 1, 0], "Mask int mismatch");

        let indices_i32 = indices.cast(DType::Int32);
        assert_eq!(indices_i32.clone().realize_with_and(&config).as_vec::<i32>().unwrap(), [5, 4, 3, 2, 1], "Indices int32 mismatch");

        // Step 6: Multiply mask by indices
        let masked_indices = mask_int.try_mul(&indices_i32).unwrap();
        // Expected: [0, 0, 0, 2, 0]
        assert_eq!(masked_indices.clone().realize_with_and(&config).as_vec::<i32>().unwrap(), [0, 0, 0, 2, 0], "Masked indices mismatch");

        // Step 7: max of masked indices
        let max_idx = masked_indices.max_with().axes(0).keepdim(false).call().unwrap();
        // Expected: [2]
        assert_eq!(max_idx.clone().realize_with_and(&config).as_vec::<i32>().unwrap(), [2], "Max idx mismatch");

        // Step 8: n - max_idx
        let n_tensor = Tensor::from_slice([axis_size]);
        let n_scalar = n_tensor.try_reshape(&[] as &[isize]).unwrap();
        let result = n_scalar.try_sub(&max_idx).unwrap();
        // Expected: [3]
        assert_eq!(result.realize_with_and(&config).as_vec::<i32>().unwrap(), [3], "Final result mismatch");
    }

    fn test_argmax_value_1d(config) {
        test_setup();
        let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 5.0, 4.0]);
        // Max 5.0 at index 3
        assert_eq!(t.argmax(Some(0)).unwrap().realize_with_and(&config).as_vec::<i32>().unwrap(), [3]);
    }

    fn test_argmax_ties_first_value(config) {
        test_setup();
        // Tinygrad test: [2, 2] -> should return 0 (first occurrence)
        let t = Tensor::from_slice([2.0f32, 2.0]);
        assert_eq!(t.argmax(Some(0)).unwrap().realize_with_and(&config).as_vec::<i32>().unwrap(), [0]);
    }

    // The WIND first-non-blank reduction relies on `argmax` over a {0,1} row
    // returning the FIRST index that holds 1 (and index 0 when the row is all
    // zero, which the caller masks out). Verify on the exact int32 / axis=1 /
    // keepdim form the RN-T block uses.
    fn test_argmax_first_true_in_binary_row(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[0i32, 0, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0]]);
        let got = t.argmax_with().axis(Some(1)).keepdim(true).call().unwrap().realize_with_and(&config).as_vec::<i32>().unwrap();
        // Row 0: first 1 at idx 2; row 1: all zero -> 0; row 2: first 1 at idx 0.
        assert_eq!(got, [2, 0, 0]);
    }

    fn test_argmax_2d_axis0_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[1.0f32, 3.0, 2.0], [4.0, 2.0, 5.0]]);
        // [[1, 3, 2], [4, 2, 5]]
        // Max per column: [4>1, 3>2, 5>2] -> indices [1, 0, 1]
        assert_eq!(t.argmax(Some(0)).unwrap().realize_with_and(&config).as_vec::<i32>().unwrap(), [1, 0, 1]);
    }

    fn test_argmax_2d_axis1_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[1.0f32, 3.0, 2.0], [4.0, 2.0, 5.0]]);
        // [[1, 3, 2], [4, 2, 5]]
        // Max per row: [3 at idx 1, 5 at idx 2] -> [1, 2]
        assert_eq!(t.argmax(Some(1)).unwrap().realize_with_and(&config).as_vec::<i32>().unwrap(), [1, 2]);
    }

    fn test_argmax_flatten_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[1.0f32, 3.0, 2.0], [4.0, 2.0, 5.0]]);
        // Flattened: [1, 3, 2, 4, 2, 5], max 5.0 at index 5
        assert_eq!(t.argmax(None).unwrap().realize_with_and(&config).as_vec::<i32>().unwrap(), [5]);
    }

    // ========== Argmin Tests (from Tinygrad test_ops.py:1106-1122) ==========

    fn test_argmin_value_1d(config) {
        test_setup();
        let t = Tensor::from_slice([5.0f32, 3.0, 1.0, 4.0, 2.0]);
        // Min 1.0 at index 2
        assert_eq!(t.argmin(Some(0)).unwrap().realize_with_and(&config).as_vec::<i32>().unwrap(), [2]);
    }

    fn test_argmin_negative_value(config) {
        test_setup();
        let t = Tensor::from_slice([1.5f32, -2.3, 0.5, 1.0]);
        // Min -2.3 at index 1
        assert_eq!(t.argmin(Some(0)).unwrap().realize_with_and(&config).as_vec::<i32>().unwrap(), [1]);
    }

    fn test_argmin_2d_axis0_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[1.0f32, 3.0, 2.0], [4.0, 2.0, 5.0]]);
        // [[1, 3, 2], [4, 2, 5]]
        // Min per column: [1<4, 2<3, 2<5] -> indices [0, 1, 0]
        assert_eq!(t.argmin(Some(0)).unwrap().realize_with_and(&config).as_vec::<i32>().unwrap(), [0, 1, 0]);
    }

    fn test_argmin_2d_axis1_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[1.0f32, 3.0, 2.0], [4.0, 2.0, 5.0]]);
        // [[1, 3, 2], [4, 2, 5]]
        // Min per row: [1 at idx 0, 2 at idx 1] -> [0, 1]
        assert_eq!(t.argmin(Some(1)).unwrap().realize_with_and(&config).as_vec::<i32>().unwrap(), [0, 1]);
    }

    // ========== Mean Tests (from Tinygrad test_ops.py:1465-1471) ==========

    fn test_mean_1d_value(config) {
        test_setup();
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        assert_close_f32(&t.mean(()).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[2.5], 1e-6);
    }

    fn test_mean_2d_axis1_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        // [[1, 2, 3], [4, 5, 6]] -> [mean(1,2,3), mean(4,5,6)] = [2, 5]
        assert_close_f32(&t.mean(1).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[2.0, 5.0], 1e-6);
    }

    fn test_mean_2d_axis0_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        // [[1, 2, 3], [4, 5, 6]] -> [mean(1,4), mean(2,5), mean(3,6)] = [2.5, 3.5, 4.5]
        assert_close_f32(&t.mean(0).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[2.5, 3.5, 4.5], 1e-6);
    }

    // 1024 * 100.0 = 102400 overflows float16 (max 65504) → inf in any accumulation
    // order; only a float32 accumulator keeps mean finite (= 100.0, cast back to f16).
    fn test_mean_float16_accumulates_in_float32(config) {
        test_setup();
        let t = Tensor::from_slice(vec![100.0f32; 1024]).cast(DType::Float16);
        let m = t.mean(()).unwrap();
        assert_eq!(m.uop().dtype(), DType::Float16, "mean must cast back to input dtype");
        let v = m.cast(DType::Float32).realize_with_and(&config).as_vec::<f32>().unwrap();
        assert!(v[0].is_finite(), "float16 mean overflowed to {} — accumulated in float16, not float32", v[0]);
        assert_close_f32(&v, &[100.0], 1e-2);
    }

    // var([1,2,3,4,5]) = 10/(5-1) = 2.5 (default correction=1). Exercises the
    // float32 squared-deviation accumulation and the cast-back to float16.
    fn test_var_float16_value(config) {
        test_setup();
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]).cast(DType::Float16);
        let v = t.var(()).unwrap();
        assert_eq!(v.uop().dtype(), DType::Float16, "var must cast back to input dtype");
        let r = v.cast(DType::Float32).realize_with_and(&config).as_vec::<f32>().unwrap();
        assert_close_f32(&r, &[2.5], 1e-2);
    }

    // 16384 squared deviations of 9.0 sum to 147456, past float16's 65504: the
    // sum of squares must stay in float32 through the divide, not be cast back
    // before it.
    fn test_var_float16_large_n_stays_finite(config) {
        test_setup();
        let data: Vec<f32> = (0..16384).map(|i| if i % 2 == 0 { 3.0 } else { -3.0 }).collect();
        let t = Tensor::from_slice(data).cast(DType::Float16);

        let v = t.var_with().axes(()).correction(0).call().unwrap();
        assert_eq!(v.uop().dtype(), DType::Float16, "var must cast back to input dtype");
        let v = v.cast(DType::Float32).realize_with_and(&config).as_vec::<f32>().unwrap();
        assert!(v[0].is_finite(), "float16 var overflowed to {}", v[0]);
        assert_close_f32(&v, &[9.0], 1e-3);

        let s = t.std_with().axes(()).correction(0).call().unwrap();
        let s = s.cast(DType::Float32).realize_with_and(&config).as_vec::<f32>().unwrap();
        assert_close_f32(&s, &[3.0], 1e-3);
    }

    // Near-constant row: 1000 ± 4 over 8192 elements. The deviations are tiny
    // relative to the values, but their squares still sum past float16 range.
    fn test_var_float16_near_constant_row(config) {
        test_setup();
        let data: Vec<f32> = (0..8192).map(|i| if i % 2 == 0 { 1004.0 } else { 996.0 }).collect();
        let t = Tensor::from_slice(data).try_reshape([1, 8192]).unwrap().cast(DType::Float16);

        let v = t.var_with().axes(1isize).correction(0).call().unwrap();
        let v = v.cast(DType::Float32).realize_with_and(&config).as_vec::<f32>().unwrap();
        assert!(v[0].is_finite(), "float16 var overflowed to {}", v[0]);
        assert_close_f32(&v, &[16.0], 1e-3);
    }

    // correction parameter (population variance) + single-element 0/0 → NaN.
    fn test_var_correction_and_single_element(config) {
        test_setup();
        // correction=0 → population variance: var([1,2,3,4,5]) = 10/5 = 2.0
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let pop = t.var_with().axes(()).correction(0).call().unwrap()
            .realize_with_and(&config).as_vec::<f32>().unwrap();
        assert_close_f32(&pop, &[2.0], 1e-5);
        // single element, default correction=1 → 0/0 = NaN (not a forced denom of 1)
        let single = Tensor::from_slice([5.0f32]).var(()).unwrap()
            .realize_with_and(&config).as_vec::<f32>().unwrap();
        assert!(single[0].is_nan(), "var of a single element must be NaN (0/0), got {}", single[0]);
    }

    // ========== Any Tests (from Tinygrad test_ops.py:1423-1432) ==========

    fn test_any_value_all_true(config) {
        test_setup();
        let t = Tensor::from_slice([true, true, true]);
        assert!(t.any(()).unwrap().realize_with_and(&config).as_vec::<bool>().unwrap()[0]);
    }

    fn test_any_value_one_true(config) {
        test_setup();
        let t = Tensor::from_slice([false, true, false]);
        assert!(t.any(()).unwrap().realize_with_and(&config).as_vec::<bool>().unwrap()[0]);
    }

    fn test_any_value_all_false(config) {
        test_setup();
        let t = Tensor::from_slice([false, false, false]);
        assert!(!t.any(()).unwrap().realize_with_and(&config).as_vec::<bool>().unwrap()[0]);
    }

    fn test_any_2d_axis0_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[true, false], [false, true]]);
        // [[true, false], [false, true]] -> any along axis 0 -> [true, true]
        assert_eq!(t.any(0).unwrap().realize_with_and(&config).as_vec::<bool>().unwrap(), [true, true]);
    }

    fn test_any_2d_axis1_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[true, false], [false, false]]);
        // [[true, false], [false, false]] -> any along axis 1 -> [true, false]
        assert_eq!(t.any(1).unwrap().realize_with_and(&config).as_vec::<bool>().unwrap(), [true, false]);
    }

    // ========== All Tests (from Tinygrad test_ops.py:1435-1444) ==========

    fn test_all_value_all_true(config) {
        test_setup();
        let t = Tensor::from_slice([true, true, true]);
        assert!(t.all(()).unwrap().realize_with_and(&config).as_vec::<bool>().unwrap()[0]);
    }

    fn test_all_value_one_false(config) {
        test_setup();
        let t = Tensor::from_slice([true, false, true]);
        assert!(!t.all(()).unwrap().realize_with_and(&config).as_vec::<bool>().unwrap()[0]);
    }

    fn test_all_value_all_false(config) {
        test_setup();
        let t = Tensor::from_slice([false, false, false]);
        assert!(!t.all(()).unwrap().realize_with_and(&config).as_vec::<bool>().unwrap()[0]);
    }

    fn test_all_2d_axis0_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[true, true], [false, true]]);
        // [[true, true], [false, true]] -> all along axis 0 -> [false, true]
        assert_eq!(t.all(0).unwrap().realize_with_and(&config).as_vec::<bool>().unwrap(), [false, true]);
    }

    fn test_all_2d_axis1_value(config) {
        test_setup();
        let t = Tensor::from_ndarray(&array![[true, true], [true, false]]);
        // [[true, true], [true, false]] -> all along axis 1 -> [true, false]
        assert_eq!(t.all(1).unwrap().realize_with_and(&config).as_vec::<bool>().unwrap(), [true, false]);
    }

    // ========== Edge Cases (from Tinygrad) ==========

    fn test_argmax_single_element_value(config) {
        test_setup();
        let t = Tensor::from_slice([42.0f32]);
        // Only element, index should be 0
        assert_eq!(t.argmax(Some(0)).unwrap().realize_with_and(&config).as_vec::<i32>().unwrap(), [0]);
    }

    fn test_argmax_all_equal_value(config) {
        test_setup();
        let t = Tensor::from_slice([5.0f32, 5.0, 5.0, 5.0]);
        // All equal, should return first index (0)
        assert_eq!(t.argmax(Some(0)).unwrap().realize_with_and(&config).as_vec::<i32>().unwrap(), [0]);
    }

    fn test_argmin_single_element_value(config) {
        test_setup();
        let t = Tensor::from_slice([42.0f32]);
        // Only element, index should be 0
        assert_eq!(t.argmin(Some(0)).unwrap().realize_with_and(&config).as_vec::<i32>().unwrap(), [0]);
    }

    fn test_max_single_element_value(config) {
        test_setup();
        let t = Tensor::from_slice([42.0f32]);
        assert_close_f32(&t.max(()).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[42.0], 1e-6);
    }

    fn test_min_single_element_value(config) {
        test_setup();
        let t = Tensor::from_slice([42.0f32]);
        assert_close_f32(&t.min(()).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[42.0], 1e-6);
    }

    fn test_debug_argmin_intermediate(config) {
        test_setup();
        let values = Tensor::from_slice([5.0f32, 3.0, 1.0, 4.0, 2.0]);

        // Test neg first - does it produce correct values?
        let inverted = -values.clone();
        assert_close_f32(&inverted.realize_with_and(&config).as_vec::<f32>().unwrap(), &[-5.0, -3.0, -1.0, -4.0, -2.0], 1e-6);

        // Test max of explicit negated (should be -1.0)
        let inverted2 = Tensor::from_slice([-5.0f32, -3.0, -1.0, -4.0, -2.0]);
        let max_inv = inverted2.max_with().axes(0).keepdim(false).call().unwrap();
        assert_close_f32(&max_inv.realize_with_and(&config).as_vec::<f32>().unwrap(), &[-1.0], 1e-6);

        // Test argmax of explicit negated values
        let inverted3 = Tensor::from_slice([-5.0f32, -3.0, -1.0, -4.0, -2.0]);
        let argmax_inv = inverted3.argmax(0).unwrap();
        assert_eq!(argmax_inv.realize_with_and(&config).as_vec::<i32>().unwrap()[0], 2); // -1.0 is at index 2
    }

    fn test_debug_lazy_neg_max(config) {
        test_setup();
        let values = Tensor::from_slice([5.0f32, 3.0, 1.0, 4.0, 2.0]);

        // Test max of lazy negated values
        let inverted = -values.clone(); // lazy neg
        let max_lazy = inverted.max(()).unwrap();
        assert_close_f32(&max_lazy.realize_with_and(&config).as_vec::<f32>().unwrap(), &[-1.0], 1e-6);
    }

    fn test_debug_lazy_neg_argmax(config) {
        test_setup();
        let values = Tensor::from_slice([5.0f32, 3.0, 1.0, 4.0, 2.0]);

        // Chain neg and argmax LAZILY (like argmin does internally)
        let inverted = -values; // lazy neg
        let argmax_lazy = inverted.argmax(0).unwrap(); // lazy argmax of lazy neg
        assert_eq!(argmax_lazy.realize_with_and(&config).as_vec::<i32>().unwrap()[0], 2);
    }

    fn test_indices_cast_bug(config) {
        use crate::Tensor;
        use svod_dtype::DType;

        crate::test::helpers::test_setup();

        // Create indices tensor [5, 4, 3, 2, 1]
        let indices = Tensor::arange(5, Some(0), Some(-1)).unwrap();

        let indices_i32 = indices.cast(DType::Int32);
        assert_eq!(indices_i32.realize_with_and(&config).as_vec::<i32>().unwrap(), [5, 4, 3, 2, 1]);
    }

    fn test_hardmax_basic(config) {
        let x = Tensor::from_ndarray(&array![[1.0f32, 3.0, 2.0], [5.0, 4.0, 0.0]]);
        let result = x.hardmax(-1).unwrap();
        assert_eq!(result.realize_with_and(&config).as_vec::<f32>().unwrap(), [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    }

    fn test_sum_with_promotes_by_default(config) {
        // 9 x 50 overflows int8; the promoted int32 accumulator keeps it exact.
        let x = Tensor::from_slice([50i8; 9]);

        let promoted = x.sum_with().axes(()).call().unwrap();
        assert_eq!(promoted.uop().dtype(), DType::Int32);
        assert_eq!(promoted.realize_with_and(&config).as_vec::<i32>().unwrap(), [450]);

        // Explicit opt-out keeps the narrow accumulator (450 wraps to -62).
        let narrow = x.sum_with().axes(()).promote(false).call().unwrap();
        assert_eq!(narrow.uop().dtype(), DType::Int8);
        assert_eq!(narrow.realize_with_and(&config).as_vec::<i8>().unwrap(), [-62]);

        // An explicit dtype still wins over the default promotion.
        let explicit = x.sum_with().axes(()).dtype(DType::Int64).call().unwrap();
        assert_eq!(explicit.uop().dtype(), DType::Int64);
        assert_eq!(explicit.realize_with_and(&config).as_vec::<i64>().unwrap(), [450]);
    }
}

#[test]
fn test_sum_with_rejects_dtype_and_promote() {
    let x = Tensor::from_slice([1i8, 2, 3]);
    assert!(matches!(
        x.sum_with().axes(()).dtype(DType::Int64).promote(true).call().map_err(crate::error::Error::into_kind),
        Err(crate::ErrorKind::ConflictingReductionOptions)
    ));
}

#[test]
fn test_prod_with_preserves_dtype_by_default() {
    let x = Tensor::from_slice([2i8, 3, 4]);
    assert_eq!(x.prod_with().axes(()).call().unwrap().uop().dtype(), DType::Int8);
    assert_eq!(x.prod_with().axes(()).promote(true).call().unwrap().uop().dtype(), DType::Int32);
}

/// `min` and `argmin` over the same axis fuse into one kernel whose upcast
/// output lanes carry the `min` accumulator; its float lowering negates with a
/// rank-0 `-1` expanded to the lane shape, which once kept one index per lane
/// axis and reached the LLVM renderer as a multi-index INDEX of a constant.
#[test_case(5 ; "five")]
#[test_case(6 ; "six")]
#[test_case(8 ; "eight")]
fn test_argmin_fuses_with_min_of_the_same_axis(m: usize) {
    // Row 0 is all infinity, so the `where` takes both branches.
    let data: Vec<f32> = (0..m * m).map(|v| if v < m { f32::INFINITY } else { ((v * 7) % 11) as f32 }).collect();
    let s = Tensor::from_slice(&data).try_reshape([m as isize, m as isize]).unwrap();
    let none = s.min(1isize).unwrap().try_eq(f32::INFINITY).unwrap();
    let w = Tensor::const_(-1i32, DType::Int32).where_(&none, s.argmin(1).unwrap()).unwrap();
    w.realize().unwrap();

    let expected: Vec<i32> = data
        .chunks(m)
        .map(|row| {
            let min = row.iter().copied().fold(f32::INFINITY, f32::min);
            if min == f32::INFINITY { -1 } else { row.iter().position(|&x| x == min).unwrap() as i32 }
        })
        .collect();
    assert_eq!(w.as_vec::<i32>().unwrap(), expected);
}
