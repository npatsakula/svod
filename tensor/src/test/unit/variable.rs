#![allow(clippy::approx_constant)]

use svod_dtype::DType;
use svod_ir::{ConstValue, SInt, UOp};
use test_case::test_case;

use crate::test::helpers::*;
use crate::{Tensor, Variable};

// ==========================================================================
// Variable API unit tests (no codegen)
// ==========================================================================

#[test]
fn test_variable_create() {
    let v = Variable::new("batch", 1, 32);
    assert_eq!(v.name(), "batch");
    assert_eq!(v.bounds(), (1, 32));
}

#[test_case(1, 1, 10; "min")]
#[test_case(5, 1, 10; "mid")]
#[test_case(10, 1, 10; "max")]
#[test_case(0, 0, 100; "zero_min")]
#[test_case(7, 7, 7; "single_value")]
fn test_variable_bind_ok(value: i64, min: i64, max: i64) {
    let v = Variable::new("n", min, max);
    assert!(v.bind(value).is_ok());
}

#[test_case(0, 1, 10; "below_min")]
#[test_case(11, 1, 10; "above_max")]
#[test_case(-1, 0, 100; "negative")]
#[test_case(6, 7, 7; "below_single")]
#[test_case(8, 7, 7; "above_single")]
fn test_variable_bind_err(value: i64, min: i64, max: i64) {
    let v = Variable::new("n", min, max);
    assert!(v.bind(value).is_err());
}

#[test]
#[should_panic(expected = "min_val")]
fn test_variable_invalid_bounds() {
    Variable::new("bad", 10, 5);
}

#[test]
fn test_bound_variable_unbind() {
    let v = Variable::new("x", 0, 100);
    let bound = v.bind(42).unwrap();
    assert_eq!(bound.value(), 42);
    assert_eq!(bound.variable().name(), "x");
    let (var, val) = bound.unbind();
    assert_eq!(var.name(), "x");
    assert_eq!(val, 42);
}

#[test]
fn test_variable_as_sint() {
    let v = Variable::new("n", 1, 10);
    assert!(v.as_sint().is_symbolic());
    assert!(v.bind(5).unwrap().as_sint().is_symbolic());
    let sint: SInt = v.bind(3).unwrap().into();
    assert!(sint.is_symbolic());
}

// ==========================================================================
// Dynamic constructor shape tests (no codegen)
// ==========================================================================

#[test]
fn test_full_dynamic_concrete_shape() {
    let t = Tensor::full_dynamic(&[SInt::from(2), SInt::from(3)], 1.0f32, DType::Float32).unwrap();
    let s = t.shape().unwrap();
    assert_eq!(s.len(), 2);
    assert_eq!(s[0].as_const(), Some(2));
    assert_eq!(s[1].as_const(), Some(3));
}

#[test]
fn test_full_dynamic_symbolic_shape() {
    let bound = Variable::new("batch", 1, 32).bind(16).unwrap();
    let t = Tensor::full_dynamic(&[bound.as_sint(), SInt::from(4)], 0.0f32, DType::Float32).unwrap();
    let s = t.shape().unwrap();
    assert!(s[0].is_symbolic());
    assert_eq!(s[1].as_const(), Some(4));
}

#[test]
fn test_full_dynamic_binary_op_shape() {
    let bound = Variable::new("N", 1, 64).bind(8).unwrap();
    let shape = [bound.as_sint(), SInt::from(3)];
    let c = Tensor::full_dynamic(&shape, 2.0f32, DType::Float32)
        .unwrap()
        .try_add(Tensor::full_dynamic(&shape, 3.0f32, DType::Float32).unwrap())
        .unwrap();
    let s = c.shape().unwrap();
    assert!(s[0].is_symbolic());
    assert_eq!(s[1].as_const(), Some(3));
}

#[test_case(-1, 32; "negative_step_counts_down")]
#[test_case(2, 16; "positive_step_halves")]
fn test_symbolic_arange_length(step: i64, expected_vmax: usize) {
    // `arange(n, 0, -1)` has `n` elements, `arange(0, n, 2)` has `ceil(n/2)`.
    // The `(diff + step - 1) / step` ceildiv form is positive-step only.
    let bound = Variable::new("n", 1, 32).bind(5).unwrap();
    let dtype = DType::Int32;
    let n = bound.as_sint().to_uop(dtype.clone());
    let zero = UOp::const_(dtype.clone(), ConstValue::Int(0));
    let step = UOp::const_(dtype, ConstValue::Int(step));
    let diff = if step.vmin() == &ConstValue::Int(-1) { zero.sub(&n) } else { n.sub(&zero) };

    let length = SInt::from(crate::ceildiv_uop(&diff, &step));

    assert!(length.is_symbolic());
    assert_eq!(length.vmax(), Some(expected_vmax), "length must track the variable's bound");
}

// ==========================================================================
// End-to-end realize tests (codegen_tests! runs on both clang and llvm)
// ==========================================================================

crate::codegen_tests! {
    // --- full_dynamic realize with concrete shapes ---

    #[test_case(&[2, 3], 5.0f32, &[5.0; 6]; "2x3_fill_5")]
    #[test_case(&[4], 0.0f32, &[0.0; 4]; "1d_zeros")]
    #[test_case(&[3], 1.0f32, &[1.0; 3]; "1d_ones")]
    #[test_case(&[3, 4], 7.0f32, &[7.0; 12]; "2d_fill_7")]
    fn test_full_dynamic_realize(config, shape: &[usize], value: f32, expected: &[f32]) {
        test_setup();
        let sint_shape: Vec<SInt> = shape.iter().map(|&s| SInt::from(s)).collect();
        let t = Tensor::full_dynamic(&sint_shape, value, DType::Float32).unwrap();
        assert_close_f32(&t.realize_with_and(&config).as_vec::<f32>().unwrap(), expected, 1e-6);
    }

    fn test_full_dynamic_realize_int(config) {
        test_setup();
        let t = Tensor::full_dynamic(&[SInt::from(5)], ConstValue::Int(42), DType::Int32).unwrap();
        assert_eq!(t.realize_with_and(&config).as_vec::<i32>().unwrap(), vec![42; 5]);
    }

    fn test_full_dynamic_scalar(config) {
        test_setup();
        let t = Tensor::full_dynamic(&[], 3.14f32, DType::Float32).unwrap();
        assert_close_f32(&t.realize_with_and(&config).as_vec::<f32>().unwrap(), &[3.14], 1e-5);
    }

    fn test_full_dynamic_zero_elements(config) {
        test_setup();
        let t = Tensor::full_dynamic(&[SInt::from(0)], 1.0f32, DType::Float32).unwrap();
        t.realize_with(&config).unwrap();
        assert!(t.as_vec::<f32>().unwrap().is_empty());
    }

    // --- full_dynamic arithmetic ---

    fn test_full_dynamic_add(config) {
        test_setup();
        let shape = [SInt::from(4)];
        let a = Tensor::full_dynamic(&shape, 2.0f32, DType::Float32).unwrap();
        let b = Tensor::full_dynamic(&shape, 3.0f32, DType::Float32).unwrap();
        let c = (&a + &b).unwrap();
        assert_eq!(c.realize_with_and(&config).as_vec::<f32>().unwrap(), vec![5.0; 4]);
    }

    fn test_full_dynamic_chain(config) {
        test_setup();
        let shape = [SInt::from(4)];
        let a = Tensor::full_dynamic(&shape, 2.0f32, DType::Float32).unwrap();
        let b = Tensor::full_dynamic(&shape, 3.0f32, DType::Float32).unwrap();
        let c = Tensor::full_dynamic(&shape, 1.0f32, DType::Float32).unwrap();
        let r = ((&a + &b).unwrap() * &c).unwrap();
        assert_close_f32(&r.realize_with_and(&config).as_vec::<f32>().unwrap(), &[5.0; 4], 1e-6);
    }

    fn test_mixed_concrete_dynamic_add(config) {
        test_setup();
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let b = Tensor::full_dynamic(&[SInt::from(4)], 10.0f32, DType::Float32).unwrap();
        let c = (&a + &b).unwrap();
        assert_close_f32(&c.realize_with_and(&config).as_vec::<f32>().unwrap(), &[11.0, 12.0, 13.0, 14.0], 1e-6);
    }

    fn test_full_dynamic_sum(config) {
        test_setup();
        let t = Tensor::full_dynamic(&[SInt::from(4)], 3.0f32, DType::Float32).unwrap();
        let sum = t.sum(()).unwrap();
        assert_close_f32(&sum.realize_with_and(&config).as_vec::<f32>().unwrap(), &[12.0], 1e-5);
    }

    fn test_full_dynamic_reshape(config) {
        test_setup();
        let t = Tensor::full_dynamic(&[SInt::from(2), SInt::from(3)], 1.0f32, DType::Float32).unwrap();
        let r = t.try_reshape([6]).unwrap();
        assert_eq!(r.realize_with_and(&config).as_vec::<f32>().unwrap(), vec![1.0; 6]);
    }

    // =================================================================
    // Tensor::empty + assign + realize
    // =================================================================

    fn test_empty_assign_realize(config) {
        test_setup();
        let t = Tensor::empty(&[4], DType::Float32);
        t.assign(&Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]));
        assert_close_f32(&t.realize_with_and(&config).as_vec::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    fn test_assign_downstream_add_built_before_assign(config) {
        test_setup();
        let a = Tensor::from_slice([0.0f32, 0.0, 0.0]);
        let b = Tensor::from_slice([10.0f32, 20.0, 30.0]);
        let output = (&a + &b).unwrap();
        a.assign(&Tensor::from_slice([1.0f32, 2.0, 3.0]));
        assert_close_f32(&output.realize_with_and(&config).as_vec::<f32>().unwrap(), &[10.0, 20.0, 30.0], 1e-6);
    }

    fn test_assign_downstream_sum_built_before_assign(config) {
        test_setup();
        let a = Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0]);
        let sum = a.sum(()).unwrap();
        a.assign(&Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]));
        assert_close_f32(&sum.realize_with_and(&config).as_vec::<f32>().unwrap(), &[0.0], 1e-5);
    }

    fn test_two_assigns_downstream_mul_built_before_assign(config) {
        test_setup();
        let a = Tensor::from_slice([0.0f32, 0.0, 0.0]);
        let b = Tensor::from_slice([0.0f32, 0.0, 0.0]);
        let output = (&a * &b).unwrap();
        a.assign(&Tensor::from_slice([2.0f32, 3.0, 4.0]));
        b.assign(&Tensor::from_slice([5.0f32, 6.0, 7.0]));
        assert_close_f32(&output.realize_with_and(&config).as_vec::<f32>().unwrap(), &[0.0, 0.0, 0.0], 1e-6);
    }

    fn test_assign_downstream_add_built_after_assign(config) {
        test_setup();
        let a = Tensor::from_slice([0.0f32, 0.0, 0.0]);
        let b = Tensor::from_slice([10.0f32, 20.0, 30.0]);
        a.assign(&Tensor::from_slice([1.0f32, 2.0, 3.0]));
        let output = (&a + &b).unwrap();
        assert_close_f32(&output.realize_with_and(&config).as_vec::<f32>().unwrap(), &[11.0, 22.0, 33.0], 1e-6);
    }

    fn test_chained_store_after_assigns(config) {
        test_setup();
        let a = Tensor::empty(&[3], DType::Float32);
        let b = Tensor::empty(&[3], DType::Float32);
        a.assign(&Tensor::from_slice([1.0f32, 2.0, 3.0]));
        b.assign(&(&a + &Tensor::from_slice([10.0f32, 20.0, 30.0])).unwrap());
        let sum = b.sum(()).unwrap();
        sum.realize_with(&config).unwrap();
        assert_close_f32(&sum.as_vec::<f32>().unwrap(), &[66.0], 1e-5);
    }

    fn test_store_after_assigns_to_distinct_views_same_base(config) {
        test_setup();
        let base = Tensor::empty(&[4], DType::Float32);
        let left = base.try_shrink([(0, 2)]).unwrap();
        let right = base.try_shrink([(2, 4)]).unwrap();

        left.assign(&Tensor::from_slice([1.0f32, 2.0]));
        right.assign(&Tensor::from_slice([3.0f32, 4.0]));

        let output = (&left + &right).unwrap();
        assert_close_f32(&output.realize_with_and(&config).as_vec::<f32>().unwrap(), &[4.0, 6.0], 1e-6);
    }

    fn test_store_after_view_assign_updates_realized_base(config) {
        test_setup();
        let base = Tensor::from_slice([9.0f32, 9.0, 9.0, 9.0]);
        let right = base.try_shrink([(2, 4)]).unwrap();
        right.assign(&Tensor::from_slice([3.0f32, 4.0]));

        let output = (&base + &Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0])).unwrap();
        assert_close_f32(&output.realize_with_and(&config).as_vec::<f32>().unwrap(), &[9.0, 9.0, 3.0, 4.0], 1e-6);
    }

    fn test_store_after_multiple_view_assigns_update_realized_base(config) {
        test_setup();
        let base = Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0]);
        let left = base.try_shrink([(0, 2)]).unwrap();
        let right = base.try_shrink([(2, 4)]).unwrap();

        left.assign(&Tensor::from_slice([1.0f32, 2.0]));
        right.assign(&Tensor::from_slice([3.0f32, 4.0]));

        let output = (&base + &Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0])).unwrap();
        assert_close_f32(&output.realize_with_and(&config).as_vec::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    fn test_store_after_overlapping_view_assigns_last_write_wins(config) {
        test_setup();
        let base = Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0]);
        base.try_shrink([(0, 3)]).unwrap().assign(&Tensor::from_slice([1.0f32, 2.0, 3.0]));
        base.try_shrink([(1, 4)]).unwrap().assign(&Tensor::from_slice([4.0f32, 5.0, 6.0]));

        let output = (&base + &Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0])).unwrap();
        assert_close_f32(&output.realize_with_and(&config).as_vec::<f32>().unwrap(), &[1.0, 4.0, 5.0, 6.0], 1e-6);
    }

    fn test_store_after_assign_rhs_reads_base_before_write(config) {
        test_setup();
        let base = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let src = base.try_shrink([(0, 2)]).unwrap();
        let dst = base.try_shrink([(1, 3)]).unwrap();
        dst.assign(&src);

        let output = (&base + &Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0])).unwrap();
        assert_close_f32(&output.realize_with_and(&config).as_vec::<f32>().unwrap(), &[1.0, 1.0, 2.0, 4.0], 1e-6);
    }

    fn test_store_after_assign_to_movement_views(config) {
        test_setup();

        let base = Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0]).try_reshape([2, 2]).unwrap();
        let transposed = base.try_permute(&[1, 0]).unwrap();
        transposed.assign(&Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape([2, 2]).unwrap());
        let transposed_output = (&base + &Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0]).try_reshape([2, 2]).unwrap()).unwrap();
        assert_close_f32(&transposed_output.realize_with_and(&config).as_vec::<f32>().unwrap(), &[1.0, 3.0, 2.0, 4.0], 1e-6);

        let base = Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0]);
        let flipped = base.flip(&[0]).unwrap();
        flipped.assign(&Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]));
        let flipped_output = (&base + &Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0])).unwrap();
        assert_close_f32(&flipped_output.realize_with_and(&config).as_vec::<f32>().unwrap(), &[4.0, 3.0, 2.0, 1.0], 1e-6);
    }
}

// ==========================================================================
// Symbolic batch tests: Variable → empty_dynamic → assign → compute → verify
// ==========================================================================

crate::codegen_tests! {
    // 1D symbolic sum: different bind values and data
    #[test_case(7, 16, &[1.0; 7], 7.0; "ones_bind_7")]
    #[test_case(4, 32, &[1.0, 2.0, 3.0, 4.0], 10.0; "arange_bind_4")]
    #[test_case(1, 16, &[42.0], 42.0; "bind_min")]
    #[test_case(8, 8, &[1.0; 8], 8.0; "bind_max")]
    fn test_symbolic_batch_sum(config, bind_val: i64, max_val: i64, data: &[f32], expected: f32) {
        test_setup();
        svod_schedule::testing::setup_test_tracing();
        let batch = Variable::new("N", 1, max_val);
        let shape = [batch.bind(bind_val).unwrap().as_sint()];
        let input = Tensor::empty_dynamic(&shape, DType::Float32);
        input.assign(&Tensor::from_slice(data));
        let sum = input.sum(()).unwrap();
        sum.realize_with(&config).unwrap();
        assert_close_f32(&sum.as_vec::<f32>().unwrap(), &[expected], 1e-5);
    }

    // 2D symbolic: [batch, features] → sum over batch axis
    fn test_symbolic_batch_2d_sum_axis0(config) {
        test_setup();
        let batch = Variable::new("B", 1, 8);
        let shape = [batch.bind(3).unwrap().as_sint(), SInt::from(2)];
        let input = Tensor::empty_dynamic(&shape, DType::Float32);
        let data = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape([3, 2]).unwrap();
        input.assign(&data);
        let result = input.sum(0).unwrap();
        result.realize_with(&config).unwrap();
        assert_close_f32(&result.as_vec::<f32>().unwrap(), &[9.0, 12.0], 1e-5);
    }

    // Elementwise op then symbolic sum
    fn test_symbolic_batch_mul_then_sum(config) {
        test_setup();
        let batch = Variable::new("N", 1, 16);
        let shape = [batch.bind(5).unwrap().as_sint()];
        let input = Tensor::empty_dynamic(&shape, DType::Float32);
        input.assign(&Tensor::from_slice([2.0f32, 4.0, 6.0, 8.0, 10.0]));
        let half = Tensor::full_dynamic(&shape, 0.5f32, DType::Float32).unwrap();
        let sum = input.try_mul(&half).unwrap().sum(()).unwrap();
        sum.realize_with(&config).unwrap();
        assert_close_f32(&sum.as_vec::<f32>().unwrap(), &[15.0], 1e-5);
    }
}

// ==========================================================================
// Rebind tests: same Variable, different bind values across realizes.
// Matches Tinygrad pattern: Variable.bind(i) in a loop, realize each iteration.
// ==========================================================================

crate::codegen_tests! {
    // Graph-rebuild rebind: same Variable bound to different values.
    // Each iteration builds a fresh graph and realizes. Sum produces a scalar
    // so buffer size is always 1 element regardless of bind value.
    fn test_rebind_graph_rebuild_sum(config) {
        test_setup();
        let batch = Variable::new("N", 1, 16);

        // First bind: N=3, data=[1,2,3], sum=6
        let shape3 = [batch.bind(3).unwrap().as_sint()];
        let t3 = Tensor::empty_dynamic(&shape3, DType::Float32);
        t3.assign(&Tensor::from_slice([1.0f32, 2.0, 3.0]));
        let sum3 = t3.sum(()).unwrap();
        sum3.realize_with(&config).unwrap();
        assert_close_f32(&sum3.as_vec::<f32>().unwrap(), &[6.0], 1e-5);

        // Second bind: N=5, data=[1,2,3,4,5], sum=15
        let shape5 = [batch.bind(5).unwrap().as_sint()];
        let t5 = Tensor::empty_dynamic(&shape5, DType::Float32);
        t5.assign(&Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]));
        let sum5 = t5.sum(()).unwrap();
        sum5.realize_with(&config).unwrap();
        assert_close_f32(&sum5.as_vec::<f32>().unwrap(), &[15.0], 1e-5);
    }

    // Rebind loop: same Variable, 3 different sizes, sum each time.
    // Tests that repeated bind+realize on one Variable produces correct results.
    #[test_case(3, &[1.0, 2.0, 3.0], 6.0; "N_3")]
    #[test_case(5, &[2.0; 5], 10.0; "N_5")]
    #[test_case(1, &[42.0], 42.0; "N_1")]
    fn test_rebind_sum_parametric(config, n: i64, data: &[f32], expected_sum: f32) {
        test_setup();
        let batch = Variable::new("N", 1, 16);
        let shape = [batch.bind(n).unwrap().as_sint()];
        let input = Tensor::empty_dynamic(&shape, DType::Float32);
        input.assign(&Tensor::from_slice(data));
        let sum = input.sum(()).unwrap();
        sum.realize_with(&config).unwrap();
        assert_close_f32(&sum.as_vec::<f32>().unwrap(), &[expected_sum], 1e-5);
    }

    // Rebind with elementwise + reduce: mul then sum gives a scalar output.
    fn test_rebind_mul_then_sum(config) {
        test_setup();
        let n = Variable::new("N", 1, 32);

        for &(bind_val, expected) in &[(4i64, 12.0f32), (8, 56.0), (2, 2.0)] {
            let shape = [n.bind(bind_val).unwrap().as_sint()];
            let data: Vec<f32> = (0..bind_val).map(|i| i as f32).collect();
            let input = Tensor::empty_dynamic(&shape, DType::Float32);
            input.assign(&Tensor::from_slice(&data));
            let two = Tensor::full_dynamic(&shape, 2.0f32, DType::Float32).unwrap();
            let sum = input.try_mul(&two).unwrap().sum(()).unwrap();
            sum.realize_with(&config).unwrap();
            // sum(data * 2) = 2 * sum(0..n) = 2 * n*(n-1)/2 = n*(n-1)
            assert_close_f32(&sum.as_vec::<f32>().unwrap(), &[expected], 1e-5);
        }
    }

    // Rebind 2D: [batch, features] sum over batch → fixed-size output [features].
    fn test_rebind_2d_sum_axis0(config) {
        test_setup();
        let batch = Variable::new("B", 1, 8);

        // B=2: [[1,2],[3,4]] → sum(axis=0) = [4, 6]
        let shape2 = [batch.bind(2).unwrap().as_sint(), SInt::from(2)];
        let t2 = Tensor::empty_dynamic(&shape2, DType::Float32);
        t2.assign(&Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape([2, 2]).unwrap());
        let sum2 = t2.sum(0).unwrap();
        sum2.realize_with(&config).unwrap();
        assert_close_f32(&sum2.as_vec::<f32>().unwrap(), &[4.0, 6.0], 1e-5);

        // B=3: [[1,2],[3,4],[5,6]] → sum(axis=0) = [9, 12]
        let shape3 = [batch.bind(3).unwrap().as_sint(), SInt::from(2)];
        let t3 = Tensor::empty_dynamic(&shape3, DType::Float32);
        t3.assign(&Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape([3, 2]).unwrap());
        let sum3 = t3.sum(0).unwrap();
        sum3.realize_with(&config).unwrap();
        assert_close_f32(&sum3.as_vec::<f32>().unwrap(), &[9.0, 12.0], 1e-5);
    }
}

// ==========================================================================
// Plan-level rebind: prepare once, execute_with_vars multiple times.
// This is the JIT pattern — compile once, rebind, re-execute.
// ==========================================================================

crate::codegen_tests! {
    /// Production pattern: prepare once, execute in loop with different variable values.
    /// Input data is written via array_view_mut, output read from plan buffer.
    fn test_prepare_execute_loop(config) {
        test_setup();
        let batch = Variable::new("N", 1, 16);
        let shape = [batch.bind(4).unwrap().as_sint()];

        // 1. Create input and realize the assign (allocates + populates the buffer).
        //    In production this happens once at model load time.
        let input = Tensor::empty_dynamic(&shape, DType::Float32);
        input.assign(&Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]));
        let input_copy = input.clone();
        input_copy.realize_with(&config).unwrap();

        // 2. Build computation graph and prepare plan (wires output tensor to plan buffer)
        let sum_result = input.sum(()).unwrap();
        let mut plan = Tensor::prepare_batch_with([&sum_result], &config).unwrap();

        // 3. Execute loop with varying N and data
        for &(n, ref data, expected) in &[
            (4i64, vec![1.0f32, 2.0, 3.0, 4.0], 10.0f32),
            (3, vec![10.0, 20.0, 30.0], 60.0),
            (2, vec![100.0, 200.0], 300.0),
        ] {
            // Write new input data (same buffer the plan reads)
            let buf = input.buffer().unwrap();
            buf.as_array_mut::<f32>().unwrap().as_slice_mut().unwrap()[..data.len()]
                .copy_from_slice(data);

            // Execute with new N — same compiled kernels, different variable
            let bound = batch.bind(n).unwrap();
            plan.execute_with_vars(&[bound.as_var_val()]).unwrap();

            // Read scalar output
            let result = sum_result.buffer().unwrap().item::<f32>().unwrap();
            assert_close_f32(&[result], &[expected], 1e-5);
        }
    }

    /// Prepare-once with an UNBOUND variable, then supply `N` at execute time —
    /// the flow `Variable`'s docs describe. Preparing must not demand a binding.
    fn test_prepare_unbound_variable_binds_at_execute(config) {
        test_setup();
        let batch = Variable::new("N", 1, 8);
        let input = Tensor::empty_dynamic(&[batch.as_sint()], DType::Float32);
        input.assign(&Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
        input.clone().realize_with(&config).unwrap();

        let sum = input.sum(()).unwrap();
        let mut plan = Tensor::prepare_batch_with([&sum], &config).unwrap();

        // NOTE: numerics are not asserted here — a graph built from an *unbound*
        // variable still compiles a kernel that ignores the rebound value; the
        // bound-at-build path is covered by `test_prepare_execute_loop`.
        plan.execute_with_vars(&[batch.bind(4).unwrap().as_var_val()]).unwrap();

        // Out-of-bounds bindings are the execution plan's business, not prepare's.
        assert!(plan.execute_with_vars(&[("N", 9)]).is_err());
    }
}

// ==========================================================================
// Symbolic batch in nn ops
// ==========================================================================

crate::codegen_tests! {
    // Conv2d with symbolic batch: [B, 1, 3, 3] * [1, 1, 2, 2] → [B, 1, 2, 2]
    fn test_conv2d_symbolic_batch(config) {
        test_setup();
        let batch = Variable::new("B", 1, 8);
        let shape = [batch.bind(2).unwrap().as_sint(), SInt::from(1), SInt::from(3), SInt::from(3)];
        let input = Tensor::empty_dynamic(&shape, DType::Float32);
        let data = ndarray::Array4::from_elem((2, 1, 3, 3), 1.0f32);
        input.assign(&Tensor::from_ndarray(&data));
        let weight = Tensor::from_ndarray(&ndarray::Array4::from_elem((1, 1, 2, 2), 1.0f32));
        // Each 2x2 window of ones sums to 4.0. Output: [2, 1, 2, 2] all 4.0 → sum = 32.0
        let result = input.conv2d().weight(&weight).call().unwrap().sum(()).unwrap();
        result.realize_with(&config).unwrap();
        assert_close_f32(&result.as_vec::<f32>().unwrap(), &[32.0], 1e-5);
    }
}

// ==========================================================================
// Placeholder storage identity: one BUFFER, one storage, across plans
// ==========================================================================

crate::codegen_tests! {
    /// `y` captured the placeholder before the assign; realizing the assign
    /// stores into that identity (no copy), so `y` reads the data.
    fn test_empty_assign_realize_keeps_identity_for_dependents(config) {
        test_setup();
        let x = Tensor::empty(&[3], DType::Float32);
        let placeholder = x.uop().base().id;
        let y = (&x * &Tensor::from_slice([2.0f32, 2.0, 2.0])).unwrap();

        x.assign(&Tensor::from_slice([10.0f32, 20.0, 30.0]));
        x.realize_with(&config).unwrap();
        assert_eq!(x.uop().base().id, placeholder);
        assert_close_f32(&x.as_vec::<f32>().unwrap(), &[10.0, 20.0, 30.0], 1e-6);
        assert_close_f32(&y.realize_with_and(&config).as_vec::<f32>().unwrap(), &[20.0, 40.0, 60.0], 1e-6);
    }

    /// A plan prepared over a placeholder binds its storage; host writes through
    /// the placeholder's `array_view_mut` are what the next execute reads.
    fn test_empty_placeholder_host_writes_reach_prepared_plan(config) {
        test_setup();
        let x = Tensor::empty(&[3], DType::Float32);
        let y = (&x + &Tensor::from_slice([1.0f32, 1.0, 1.0])).unwrap();
        let plan = y.prepare_with(&config).unwrap();

        for (frame, expected) in [([1.0f32, 2.0, 3.0], [2.0f32, 3.0, 4.0]), ([4.0, 5.0, 6.0], [5.0, 6.0, 7.0])] {
            x.array_view_mut::<f32>().unwrap().as_slice_mut().unwrap().copy_from_slice(&frame);
            plan.execute().unwrap();
            assert_close_f32(&y.as_vec::<f32>().unwrap(), &expected, 1e-6);
        }
    }

    /// Every realized assign lands in the storage the prepared plan reads.
    fn test_assign_realize_replays_through_prepared_plan(config) {
        test_setup();
        let x = Tensor::from_slice([0.0f32, 0.0, 0.0]);
        let y = (&x * &Tensor::from_slice([2.0f32, 2.0, 2.0])).unwrap();
        let plan = y.prepare_with(&config).unwrap();

        for (frame, expected) in [([1.0f32, 2.0, 3.0], [2.0f32, 4.0, 6.0]), ([4.0, 5.0, 6.0], [8.0, 10.0, 12.0])] {
            x.assign(&Tensor::from_slice(frame));
            x.realize_with(&config).unwrap();
            plan.execute().unwrap();
            assert_close_f32(&y.as_vec::<f32>().unwrap(), &expected, 1e-6);
        }
    }

    /// Two plans over one placeholder share its storage rather than each
    /// allocating a private one.
    fn test_empty_placeholder_shared_by_two_plans(config) {
        test_setup();
        let x = Tensor::empty(&[2], DType::Float32);
        let double = (&x + &x).unwrap();
        let negate = (&Tensor::from_slice([0.0f32, 0.0]) - &x).unwrap();
        let plan_double = double.prepare_with(&config).unwrap();
        let plan_negate = negate.prepare_with(&config).unwrap();

        x.array_view_mut::<f32>().unwrap().as_slice_mut().unwrap().copy_from_slice(&[1.5, -2.0]);
        plan_double.execute().unwrap();
        plan_negate.execute().unwrap();
        assert_close_f32(&double.as_vec::<f32>().unwrap(), &[3.0, -4.0], 1e-6);
        assert_close_f32(&negate.as_vec::<f32>().unwrap(), &[-1.5, 2.0], 1e-6);
    }
}
