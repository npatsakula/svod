//! Tests: JIT support types — the input descriptor, the device-local prepare
//! config, the live output-shape descriptor and the buffer helpers the
//! `jit_wrapper!` expansion calls.

use svod_dtype::DType;
use test_case::test_case;

use crate::jit::{Dim, InputSpec, JitError, OutputShape, host_write, shrink_batch, to_vec, view, zero_fill};
use crate::{PrepareConfig, Tensor, Variable};

#[test_case(InputSpec::f32(&[2, 3]), DType::Float32, 6; "f32")]
#[test_case(InputSpec::i32(&[4]), DType::Int32, 4; "i32")]
#[test_case(InputSpec::i64(&[]), DType::Int64, 1; "i64 scalar")]
fn input_spec_carries_shape_dtype_and_element_count(spec: InputSpec, dtype: DType, numel: usize) {
    assert_eq!(spec.dtype, dtype);
    assert_eq!(spec.numel(), numel);
    assert!(!spec.device_local, "host-visible by default");
    assert!(spec.clone().device_local().device_local);
}

#[test]
fn device_local_config_matches_from_env_except_the_output_placement() {
    let env = PrepareConfig::from_env();
    let local = PrepareConfig::device_local();
    assert!(!env.device_local_outputs);
    assert!(local.device_local_outputs);
    assert_eq!(local.planner_mode, env.planner_mode);
    assert_eq!(local.threads, env.threads);
    assert_eq!(local.disable_schedule_cache, env.disable_schedule_cache);
}

fn batched(rank_2: bool) -> (Tensor, crate::BoundVariable) {
    let b = Variable::new("b", 1, 4).bind(4).expect("bind");
    let t = Tensor::from_slice([0.0f32; 12]);
    let t = if rank_2 { t.try_reshape([4, 3]).unwrap() } else { t };
    (t, b)
}

#[test]
fn shrink_batch_narrows_only_the_leading_dimension() {
    let (t, b) = batched(true);
    let shrunk = shrink_batch(&t, &b).unwrap();
    let shape = shrunk.shape().unwrap();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], b.as_sint(), "dim 0 becomes the bound variable");
    assert_eq!(shape[1], svod_ir::SInt::Const(3), "trailing dims are untouched");
}

#[test]
fn output_shape_resolves_symbolic_dims_against_the_live_bindings() {
    let (t, b) = batched(true);
    let shrunk = shrink_batch(&t, &b).unwrap();

    let shape = OutputShape::capture(&shrunk, &[&b]).unwrap();
    assert_eq!(shape.dims(), &[Dim::Var(0), Dim::Const(3)]);
    assert_eq!(shape.dtype(), Some(&DType::Float32));
    assert_eq!(shape.resolve(&[4]), vec![4, 3]);
    assert_eq!(shape.resolve(&[2]), vec![2, 3]);
    assert_eq!(shape.numel(&[2]), 6);

    // A shape with no symbolic dim ignores the bindings entirely.
    let constant = OutputShape::capture(&t, &[&b]).unwrap();
    assert_eq!(constant.dims(), &[Dim::Const(4), Dim::Const(3)]);
    assert_eq!(constant.resolve(&[1]), vec![4, 3]);
}

#[test]
fn output_shape_falls_back_to_the_upper_bound_for_an_undeclared_variable() {
    let (t, b) = batched(true);
    let shrunk = shrink_batch(&t, &b).unwrap();
    // Captured without `b` in the variable list: the dim degrades to the
    // extent the buffer was allocated for.
    let shape = OutputShape::capture(&shrunk, &[]).unwrap();
    assert_eq!(shape.dims(), &[Dim::Const(4), Dim::Const(3)]);
}

#[test]
fn views_and_reads_cover_the_live_prefix_only() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape([3, 2]).unwrap();
    let buffer = t.buffer().expect("realized");

    let full = view::<f32>(&buffer, &[3, 2]).unwrap();
    assert_eq!(full.shape(), &[3, 2]);
    assert_eq!(full[[2, 1]], 6.0);

    let prefix = view::<f32>(&buffer, &[2, 2]).unwrap();
    assert_eq!(prefix.shape(), &[2, 2]);
    assert_eq!(prefix.iter().copied().collect::<Vec<_>>(), vec![1.0, 2.0, 3.0, 4.0]);

    assert_eq!(to_vec::<f32>(&buffer, 6).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(to_vec::<f32>(&buffer, 3).unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn views_reject_a_foreign_dtype_and_an_oversized_shape() {
    let t = Tensor::from_slice([1.0f32, 2.0]);
    let buffer = t.buffer().expect("realized");

    let err = view::<i32>(&buffer, &[2]).expect_err("dtype mismatch");
    assert!(matches!(err, JitError::DtypeMismatch { .. }), "{err}");
    let err = to_vec::<i32>(&buffer, 2).expect_err("dtype mismatch");
    assert!(matches!(err, JitError::DtypeMismatch { .. }), "{err}");

    let err = view::<f32>(&buffer, &[4]).expect_err("out of bounds");
    assert!(matches!(err, JitError::ViewOutOfBounds { requested: 4, available: 2 }), "{err}");
    let err = to_vec::<f32>(&buffer, 4).expect_err("out of bounds");
    assert!(matches!(err, JitError::ViewOutOfBounds { .. }), "{err}");
}

#[test]
fn host_write_and_zero_fill_reach_the_same_storage() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let mut buffer = t.buffer().expect("realized");

    host_write::<f32>(&mut buffer).unwrap().copy_at(1, &[20.0]).unwrap();
    assert_eq!(to_vec::<f32>(&buffer, 3).unwrap(), vec![1.0, 20.0, 3.0]);

    zero_fill(&mut buffer).unwrap();
    assert_eq!(to_vec::<f32>(&buffer, 3).unwrap(), vec![0.0, 0.0, 0.0]);
}

#[test]
fn a_plan_refuses_to_launch_while_a_host_write_into_its_input_is_live() {
    let x = Tensor::empty(&[2], DType::Float32);
    let double = (&x + &x).unwrap();
    let plan = double.prepare().unwrap();
    let mut input = x.buffer().expect("bound by prepare");
    let guard = host_write::<f32>(&mut input).unwrap();
    let err = plan.execute().expect_err("launch with a live guard");
    assert!(err.to_string().contains("host write"), "{err}");
    drop(guard);
    x.with_host_write::<f32, _>(|w| w.copy_from(&[1.5, -2.0])).unwrap();
    plan.execute().expect("the refusal does not poison the plan");
    assert_eq!(double.as_vec::<f32>().unwrap(), [3.0, -4.0]);
}
