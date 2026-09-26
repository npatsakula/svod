use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{DType, DeviceSpec};
use svod_ir::{CallInfo, ConstValue, KernelInfo, Op, SInt, UOp, shape::Shape};

use crate::{CpuBackend, PrepareConfig, Tensor, test::helpers::*};
use svod_ir::ops;

#[test]
fn test_tensor_custom_kernel_builds_after_call_outputs() {
    let a = Tensor::empty(&[4], DType::Float32);
    let b = Tensor::empty(&[4], DType::Float32);

    let outputs = a
        .custom_kernel(&[&b], |placeholders| {
            assert_eq!(placeholders.len(), 2);
            UOp::sink(vec![placeholders[0].clone(), placeholders[1].clone()])
        })
        .expect("custom kernel should build");

    assert_eq!(outputs.len(), 2);
    for out in outputs {
        match out.uop().op() {
            Op::After(ops::After { passthrough, deps }) => {
                assert!(passthrough.has_buffer_identity());
                assert_eq!(deps.len(), 1);
                match deps[0].op() {
                    Op::Call(ops::Call { body, args, info }) => {
                        assert!(matches!(body.op(), Op::Sink(..)));
                        assert_eq!(args.len(), 2);
                        assert_eq!(**info, CallInfo::default());
                    }
                    op => panic!("expected CALL dep, got {op:?}"),
                }
            }
            op => panic!("expected AFTER output, got {op:?}"),
        }
    }
}

#[test]
fn test_tensor_custom_kernel_with_call_info() {
    let a = Tensor::empty(&[4], DType::Float32);
    let info = CallInfo { grad_tag: Some("grad_tag".to_string()), ..CallInfo::default() };

    let outputs = a
        .custom_kernel_with(&[], info.clone(), |placeholders| UOp::sink(vec![placeholders[0].clone()]))
        .expect("custom kernel should build");

    assert_eq!(outputs.len(), 1);
    match outputs[0].uop().op() {
        Op::After(ops::After { deps, .. }) => {
            assert_eq!(deps.len(), 1);
            match deps[0].op() {
                Op::Call(ops::Call { info: call_info, .. }) => assert_eq!(**call_info, info),
                op => panic!("expected CALL dep, got {op:?}"),
            }
        }
        op => panic!("expected AFTER output, got {op:?}"),
    }
}

#[test]
fn test_tensor_custom_kernel_symbolic_placeholder_error() {
    // `placeholder_like` rejects symbolic-shaped inputs at construction
    // time (tinygrad parity).
    let n = UOp::define_var("N".to_string(), 1, 8);
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let shaped = buf.try_reshape(&Shape::from_iter([SInt::from(n)])).unwrap();
    let symbolic = Tensor::from_lazy(shaped);

    let err = match symbolic.custom_kernel(&[], |placeholders| UOp::sink(vec![placeholders[0].clone()])) {
        Ok(_) => panic!("symbolic placeholder_like should fail"),
        Err(err) => err,
    };
    assert!(format!("{err}").contains("symbolic shape is not supported"));
}

#[test]
fn test_tensor_custom_kernel_placeholder_like_multi_shape() {
    let shard = UOp::new_buffer(DeviceSpec::Cpu, 6, DType::Float32)
        .try_reshape(&Shape::from_iter([SInt::Const(2), SInt::Const(3)]))
        .unwrap();
    let multi = Tensor::from_lazy(UOp::multi(shard, 0));

    let outputs = multi
        .custom_kernel(&[], |placeholders| {
            assert_eq!(placeholders.len(), 1);
            let shape = placeholders[0].shape().unwrap().cloned().expect("placeholder shape");
            assert_eq!(shape.iter().map(|d| d.as_const()).collect::<Vec<_>>(), vec![Some(2), Some(3)]);
            UOp::sink(vec![placeholders[0].clone()])
        })
        .expect("custom kernel should build for multi-device shard shape");

    assert_eq!(outputs.len(), 1);
}

crate::codegen_tests! {
    fn test_tensor_custom_kernel_numerical_results(config) {
        test_setup();

        let src = Tensor::from_slice([3.5f32]);
        let dst = Tensor::empty(&[1], DType::Float32);

        let mut outputs = dst
            .custom_kernel(&[&src], |placeholders| {
                assert_eq!(placeholders.len(), 2);
                let out_buf = placeholders[0].clone();
                let in_buf = placeholders[1].clone();

                let idx = UOp::index_const(0);
                let load_idx = UOp::index().buffer(in_buf.clone()).indices(vec![idx.clone()]).call().unwrap();
                let store_idx = UOp::index().buffer(out_buf.clone()).indices(vec![idx]).call().unwrap();

                let loaded = UOp::load().index(load_idx).call();
                let two = UOp::const_(DType::Float32, ConstValue::Float(2.0));
                let doubled = loaded.try_mul(&two).unwrap();
                let store = store_idx.store(doubled);
                UOp::sink(vec![store])
            })
            .expect("custom kernel should build");

        let out = outputs.remove(0);
        out.realize_with(&config).unwrap();

        let result = out.as_vec::<f32>().unwrap();
        assert_close_f32(&result, &[7.0], 1e-6);
    }
}

fn run_custom_op_numerical_test(backend: CpuBackend, mul_tpl: &str, add_tpl: &str) {
    test_setup();

    // Custom templates are backend-specific (C expressions for Clang, LLVM IR
    // instruction RHS for LLVM); pin all tensors to CPU regardless of
    // SVOD_DEVICE so we exercise the requested CPU backend.
    svod_dtype::default_device::with_default_device(svod_dtype::DeviceSpec::Cpu, || {
        let src = Tensor::from_slice([3.5f32]);
        let dst = Tensor::empty(&[1], DType::Float32);

        let mut outputs = dst
            .custom_kernel(&[&src], |placeholders| {
                assert_eq!(placeholders.len(), 2);
                let out_buf = placeholders[0].clone();
                let in_buf = placeholders[1].clone();

                let idx = UOp::index_const(0);
                let load_idx = UOp::index().buffer(in_buf.clone()).indices(vec![idx.clone()]).call().unwrap();
                let store_idx = UOp::index().buffer(out_buf.clone()).indices(vec![idx]).call().unwrap();

                let loaded = UOp::load().index(load_idx).call();
                let scaled = UOp::custom(smallvec![loaded], mul_tpl.to_string(), DType::Float32);
                let shifted = UOp::custom(smallvec![scaled], add_tpl.to_string(), DType::Float32);
                let store = store_idx.store(shifted);
                UOp::sink(vec![store])
            })
            .expect("custom kernel should build");

        let out = outputs.remove(0);
        let config = PrepareConfig::for_cpu_backend(backend);
        out.realize_with(&config).unwrap();

        let result = out.as_vec::<f32>().unwrap();
        assert_close_f32(&result, &[8.0], 1e-6);
    });
}

#[test]
fn test_tensor_custom_op_numerical_clang_backend() {
    // C backend template strings.
    run_custom_op_numerical_test(CpuBackend::Clang, "({0} * 2.0f)", "({0} + 1.0f)");
}

#[test]
fn test_tensor_custom_op_numerical_llvm_backend() {
    // LLVM templates render the instruction RHS of a typed CUSTOM
    // (`%vN = <rhs>`); the LLVM type lives in the RHS itself.
    run_custom_op_numerical_test(CpuBackend::Llvm, "fmul float {0}, 2.0", "fadd float {0}, 1.0");
}

/// Build a hand-ranged `out[i] = in[i] + 1` kernel body (manual RANGE +
/// STORE-with-ranges). Used to prove the schedule passes a custom_kernel body
/// through lowering WITHOUT re-rangeifying author-created loops — the core
/// precondition for the svod-tk tile DSL.
fn hand_ranged_add1_body(n: usize) -> impl FnOnce(Vec<Arc<UOp>>) -> Arc<UOp> {
    move |ph| {
        let out_buf = ph[0].clone();
        let in_buf = ph[1].clone();
        let i = UOp::range_const(n as i64, 0);
        let in_idx = UOp::index().buffer(in_buf.clone()).indices(vec![i.clone()]).call().unwrap();
        let loaded = UOp::load().index(in_idx).call();
        let one = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let val = loaded.try_add(&one).unwrap();
        let out_idx = UOp::index().buffer(out_buf).indices(vec![i.clone()]).call().unwrap();
        // Plain STORE, with the manual loop closed by an explicit END(range) —
        // the tinykittens `store(..).end(i)` pattern. (`store_with_ranges` is for
        // output-upcast UNROLL, not loop closing.)
        let store = out_idx.store(val).end(smallvec![i]);
        // opts_to_apply = Some(vec![]) — the tinygrad `()` analog: this SINK is
        // already in finished, hand-lowered form; the optimizer must apply zero
        // opts (no heuristic upcast/vectorize of the manual loop).
        UOp::sink_with_info(vec![store], KernelInfo { opts_to_apply: Some(vec![]), ..Default::default() })
    }
}

#[test]
fn test_custom_kernel_hand_ranged_loop_cpu() {
    test_setup();
    svod_dtype::default_device::with_default_device(svod_dtype::DeviceSpec::Cpu, || {
        let n = 8usize;
        let src = Tensor::from_slice([0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let dst = Tensor::empty(&[n], DType::Float32);

        let mut outputs = dst.custom_kernel(&[&src], hand_ranged_add1_body(n)).expect("custom kernel should build");

        let out = outputs.remove(0);
        out.realize_with(&PrepareConfig::for_cpu_backend(CpuBackend::Clang)).unwrap();

        let result = out.as_vec::<f32>().unwrap();
        let expected: Vec<f32> = (0..n).map(|x| x as f32 + 1.0).collect();
        assert_close_f32(&result, &expected, 1e-6);
    });
}

/// Hardware-gated: `SVOD_DEVICE=AMD:0 cargo test -p svod-tensor custom_kernel::test_custom_kernel_hand_ranged_loop_amd -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_custom_kernel_hand_ranged_loop_amd() {
    test_setup();
    let n = 8usize;
    let src = Tensor::from_slice([0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let dst = Tensor::empty(&[n], DType::Float32);

    let mut outputs = dst.custom_kernel(&[&src], hand_ranged_add1_body(n)).expect("custom kernel should build");

    let out = outputs.remove(0);
    out.realize_with(&PrepareConfig::from_env()).unwrap();

    let result = out.as_vec::<f32>().unwrap();
    let expected: Vec<f32> = (0..n).map(|x| x as f32 + 1.0).collect();
    assert_close_f32(&result, &expected, 1e-6);
}

/// Hardware-gated end-to-end raw-CUSTOM on the AMD/LLVM renderer:
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tensor custom_kernel::test_tensor_custom_op_amd_end_to_end -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_tensor_custom_op_amd_end_to_end() {
    test_setup();

    // Tensors land on the default device (AMD under SVOD_DEVICE=AMD:0); the
    // kernel is rendered by the AMD LLVM renderer, compiled to an amdgcn ELF,
    // and dispatched on the GPU — exercising the CUSTOM path end to end.
    let src = Tensor::from_slice([3.5f32]);
    let dst = Tensor::empty(&[1], DType::Float32);

    let mut outputs = dst
        .custom_kernel(&[&src], |placeholders| {
            let out_buf = placeholders[0].clone();
            let in_buf = placeholders[1].clone();

            let idx = UOp::index_const(0);
            let load_idx = UOp::index().buffer(in_buf.clone()).indices(vec![idx.clone()]).call().unwrap();
            let store_idx = UOp::index().buffer(out_buf.clone()).indices(vec![idx]).call().unwrap();

            let loaded = UOp::load().index(load_idx).call();
            let custom = UOp::custom(smallvec![loaded], "fmul float {0}, 2.0".to_string(), DType::Float32);
            let store = store_idx.store(custom);
            UOp::sink(vec![store])
        })
        .expect("custom kernel should build");

    let out = outputs.remove(0);
    out.realize_with(&PrepareConfig::from_env()).unwrap();

    let result = out.as_vec::<f32>().unwrap();
    assert_close_f32(&result, &[7.0], 1e-6);
}

/// A hand-ranged `out[i, j] = 1 + Σ in[i, j]` body over `[rows, cols]` operands.
fn hand_ranged_sum_plus1_body(rows: usize, cols: usize) -> impl FnOnce(Vec<Arc<UOp>>) -> Arc<UOp> {
    move |ph| {
        let (i, j) = (UOp::range_const(rows as i64, 0), UOp::range_const(cols as i64, 1));
        let at = |buf: &Arc<UOp>| UOp::index().buffer(buf.clone()).indices(vec![i.clone(), j.clone()]).call().unwrap();
        let val = ph[1..].iter().fold(UOp::const_(DType::Float32, ConstValue::Float(1.0)), |acc, input| {
            acc.try_add(&UOp::load().index(at(input)).call()).unwrap()
        });
        let store = at(&ph[0]).store(val).end(smallvec![i, j]);
        UOp::sink_with_info(vec![store], KernelInfo { opts_to_apply: Some(vec![]), ..Default::default() })
    }
}

/// A custom kernel's output read twice — by an elementwise graph op and whole,
/// as the next custom kernel's argument. The output's reshape is one node for
/// both readers, so indexing it for the elementwise op used to leave an INDEX as
/// the CALL argument, which the kernel graph rejects; the CALL must bind the
/// buffer.
#[test]
fn test_custom_kernel_output_shared_with_an_elementwise_reader() {
    test_setup();
    svod_dtype::default_device::with_default_device(svod_dtype::DeviceSpec::Cpu, || {
        let (rows, cols) = (4usize, 8usize);
        let x: Vec<f32> = (0..rows * cols).map(|v| v as f32 / 8.0).collect();
        let src = Tensor::from_slice(&x).try_reshape([rows as isize, cols as isize]).unwrap();
        let empty = || Tensor::empty(&[rows, cols], DType::Float32);

        let h = empty().custom_kernel(&[&src], hand_ranged_sum_plus1_body(rows, cols)).unwrap().remove(0);
        let squared = h.try_mul(&h).unwrap();
        let out = empty().custom_kernel(&[&squared, &h], hand_ranged_sum_plus1_body(rows, cols)).unwrap().remove(0);
        out.realize_with(&PrepareConfig::for_cpu_backend(CpuBackend::Clang)).unwrap();

        let expected: Vec<f32> = x.iter().map(|v| (v + 1.0) * (v + 1.0) + (v + 1.0) + 1.0).collect();
        assert_close_f32(&out.as_vec::<f32>().unwrap(), &expected, 1e-5);
    });
}
