use std::collections::HashMap;
use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType, DeviceSpec};

use crate::ops;
use crate::pattern::{Matcher, RewriteResult};
use crate::{AxisId, BinaryOp, CallInfo, ConstValue, Op, SInt, UOp, UOpKey, shape::Shape}; // ConstValue kept for DType::Index

fn program(
    sink: Arc<UOp>,
    target: DeviceSpec,
    linear: Option<Arc<UOp>>,
    source: Option<Arc<UOp>>,
    binary: Option<Arc<UOp>>,
) -> Arc<UOp> {
    let info = crate::ProgramInfo::from_sink(&sink, target);
    UOp::program(sink, info, linear, source, binary)
}

struct RewriteCallToFirstArg;

impl Matcher<()> for RewriteCallToFirstArg {
    fn rewrite(&self, uop: &Arc<UOp>, _ctx: &mut ()) -> RewriteResult {
        match uop.op() {
            Op::Call(ops::Call { args, .. }) | Op::Function(ops::Function { args, .. }) if !args.is_empty() => {
                RewriteResult::Rewritten(args[0].clone())
            }
            _ => RewriteResult::NoMatch,
        }
    }
}

#[test]
fn typed_constants_commit_and_report_unsupported_conversions() {
    let native = UOp::native_const(1.0f32);
    assert_eq!(native.dtype(), DType::Float32);
    assert!(matches!(native.op(), Op::Const(_)));

    // Float16 has no exact representation for this, so the constant commits to the grid.
    let constant = UOp::const_(DType::Float16, ConstValue::Float(1.0 / 123_008.0));
    assert!(matches!(constant.op(), Op::Const(value) if value.0 == ConstValue::Float(8.106231689453125e-6)));

    // bf16 commitment is total (IB1): f32-range overflow saturates instead of failing.
    let saturated = UOp::try_const_(DType::BFloat16, ConstValue::Float(1e300)).unwrap();
    assert!(matches!(saturated.op(), Op::Const(value) if value.0 == ConstValue::Float(f64::INFINITY)));

    let pointer = DType::Float32.ptr(None, svod_dtype::AddrSpace::Global).unwrap();
    assert!(matches!(UOp::try_const_(pointer, ConstValue::Float(1.0)), Err(crate::Error::ConstantConversion { .. })));
}

#[test]
fn vconst_commits_every_lane() {
    let vector = UOp::vconst(
        vec![ConstValue::Float(1.0625), ConstValue::Float(1.1875), ConstValue::Float(-0.0)],
        DType::FP8E4M3,
    );
    assert_eq!(vector.dtype(), DType::FP8E4M3.vec(3).unwrap());
    assert!(matches!(vector.op(), Op::VConst(ops::VConst { values })
        if values == &vec![ConstValue::Float(1.0), ConstValue::Float(1.25), ConstValue::Float(-0.0)]));

    let fnuz = UOp::vconst(vec![ConstValue::Float(-0.0)], DType::FP8E4M3FNUZ);
    assert!(matches!(fnuz.op(), Op::VConst(ops::VConst { values })
        if values[0] == ConstValue::Float(0.0) && values[0] != ConstValue::Float(-0.0)));
}

#[test]
fn const_like_converts_to_receiver_scalar_dtype() {
    let receiver = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let constant = receiver.const_like(2i64);

    assert_eq!(constant.dtype(), DType::Float32);
    assert!(matches!(constant.op(), Op::Const(value) if value.0 == ConstValue::Float(2.0)));
    assert_eq!(constant.shape().unwrap().unwrap().as_slice(), &[]);

    // Tinygrad `uop/ops.py:596` keeps the receiver's vector count (IC2).
    let vector = UOp::const_(DType::UInt16.vec(4).unwrap(), ConstValue::Int(3));
    assert_eq!(vector.const_like(1i64).dtype(), vector.dtype());
    assert_eq!(vector.neg().dtype(), vector.dtype());

    // A shapeless receiver degrades `vconst_like` to the plain constant (IC3).
    let shapeless = UOp::vconst(vec![ConstValue::Float(1.0), ConstValue::Float(2.0)], DType::Float32);
    assert!(shapeless.shape().unwrap().is_none());
    assert_eq!(shapeless.vconst_like(2i64).dtype(), shapeless.dtype());
}

#[test]
fn const_like_expands_independent_receiver_shape() {
    for width in [4usize, 5, 8] {
        let receiver =
            UOp::stack((0..width).map(|value| UOp::const_(DType::Int32, ConstValue::Int(value as i64))).collect());
        let constant = receiver.const_like(1i64);

        assert_eq!(constant.dtype(), DType::Int32);
        assert_eq!(constant.shape().unwrap().unwrap().as_slice(), &[width.into()]);
        assert!(matches!(constant.op(), Op::Expand(ops::Expand { src, .. })
            if matches!(src.op(), Op::Const(value) if value.0 == ConstValue::Int(1))));
        assert!(
            receiver.try_add(&constant).is_ok(),
            "width {width} constant must satisfy strict binary shape validation"
        );
    }
}

#[test]
fn vconst_like_stacks_after_movement_lowering() {
    let receiver = UOp::stack(smallvec![
        UOp::native_const(0.0f32),
        UOp::native_const(1.0f32),
        UOp::native_const(2.0f32),
        UOp::native_const(3.0f32),
    ]);
    let constant = receiver.vconst_like(0);

    assert_eq!(constant.dtype(), DType::Float32);
    assert!(matches!(constant.op(), Op::Stack(ops::Stack { sources }) if sources.len() == 4));
    assert!(!constant.toposort().iter().any(|node| node.op().is_movement()));
}

#[test]
fn const_like_shapes_invalid_without_retyping_it() {
    for width in [2usize, 5] {
        let receiver =
            UOp::stack((0..width).map(|v| UOp::const_(DType::Float32, ConstValue::Float(v as f64))).collect());
        let invalid = receiver.const_like(ConstValue::Invalid);

        assert_eq!(invalid.dtype(), DType::Bool, "INVALID keeps its own dtype");
        assert_eq!(invalid.shape().unwrap().unwrap().as_slice(), &[width.into()]);
        assert!(matches!(invalid.op(), Op::Expand(ops::Expand { src, .. }) if UOp::is_invalid_marker(src)));
        assert!(UOp::is_invalid_marker(&invalid));
    }
}

#[test]
fn test_hash_consing() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    assert!(Arc::ptr_eq(&a, &UOp::native_const(1.0f32)), "identical leaves intern to one Arc");
    assert!(Arc::ptr_eq(&a.try_add(&b).unwrap(), &a.try_add(&b).unwrap()), "and so do identical inner nodes");
}

#[test]
fn test_hash_consing_preserves_differently_tagged_child_order() {
    let base = UOp::index_const(7);
    let left = base.with_tag(smallvec![1]);
    let right = base.with_tag(smallvec![2]);
    assert_eq!(left.content_hash, right.content_hash);
    assert_ne!(left.id, right.id);

    let forward = UOp::new(Op::Binary(BinaryOp::Add, left.clone(), right.clone()), DType::WeakInt);
    let reverse = UOp::new(Op::Binary(BinaryOp::Add, right.clone(), left.clone()), DType::WeakInt);
    assert!(!Arc::ptr_eq(&forward, &reverse));
    let Op::Binary(_, forward_left, forward_right) = forward.op() else { panic!("expected ADD") };
    let Op::Binary(_, reverse_left, reverse_right) = reverse.op() else { panic!("expected ADD") };
    assert!(Arc::ptr_eq(forward_left, &left));
    assert!(Arc::ptr_eq(forward_right, &right));
    assert!(Arc::ptr_eq(reverse_left, &right));
    assert!(Arc::ptr_eq(reverse_right, &left));
}

/// Creating the same UOp concurrently in many threads must still yield one Arc, so
/// `Arc::ptr_eq` stays a valid identity check across threads.
#[test]
fn test_cross_thread_hash_consing() {
    use std::sync::Barrier;

    for build in [(|| UOp::native_const(42.0f32)) as fn() -> Arc<UOp>, || {
        let add = UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).unwrap();
        add.try_mul(&UOp::native_const(3.0f32)).unwrap()
    }] {
        let barrier = Arc::new(Barrier::new(10));
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    build()
                })
            })
            .collect();

        let uops: Vec<_> = handles.into_iter().map(|handle| handle.join().unwrap()).collect();
        for (index, uop) in uops.iter().enumerate() {
            assert!(Arc::ptr_eq(&uops[0], uop), "thread {index} got id {} vs {}", uop.id, uops[0].id);
        }
    }
}

#[test]
fn test_alu_dtypes_and_arity() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);

    let add = a.try_add(&b).unwrap();
    assert_eq!(add.dtype(), DType::Float32);
    assert_eq!(add.op().children().len(), 2);

    let sqrt = a.try_sqrt().unwrap();
    assert_eq!(sqrt.dtype(), DType::Float32);
    assert_eq!(sqrt.op().children().len(), 1);

    assert_eq!(a.cast(DType::Int32).dtype(), DType::Int32);
    assert_eq!(a.try_cmplt(&b).unwrap().dtype(), DType::Bool);
}

#[test]
fn test_toposort() {
    // Build graph: (a + b) * c
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let c = UOp::native_const(3.0f32);

    let add = a.try_add(&b).unwrap();
    let mul = add.try_mul(&c).unwrap();

    let sorted = mul.toposort();

    // All nodes should be present
    assert!(sorted.len() >= 5); // a, b, c, add, mul

    // Check that dependencies come before dependents
    let positions: HashMap<_, _> = sorted.iter().enumerate().map(|(i, node)| (Arc::as_ptr(node), i)).collect();

    for node in &sorted {
        let node_pos = positions[&Arc::as_ptr(node)];
        for child in node.op().children() {
            let child_pos = positions[&Arc::as_ptr(child)];
            assert!(child_pos < node_pos, "Dependencies must come before dependents");
        }
    }
}

#[test]
fn test_toposort_shared_node() {
    // Build graph: x = a + b; y = a + c; z = x * y
    // Node 'a' is shared between x and y
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let c = UOp::native_const(3.0f32);

    let x = a.try_add(&b).unwrap();
    let y = a.try_add(&c).unwrap();
    let z = x.try_mul(&y).unwrap();

    let sorted = z.toposort();

    // Node 'a' should appear only once
    let a_ptr = Arc::as_ptr(&a);
    let a_count = sorted.iter().filter(|node| Arc::as_ptr(node) == a_ptr).count();
    assert_eq!(a_count, 1, "Shared node 'a' should appear exactly once");
}

#[test]
fn test_toposort_call_aware_boundaries() {
    let p0 = UOp::param(0, 1, DType::Float32, None);
    let p1 = UOp::param(1, 1, DType::Float32, None);
    let body = p0.try_add(&p1).unwrap();
    let arg0 = UOp::native_const(4.0f32);
    let arg1 = UOp::native_const(5.0f32);
    let call = body.call(smallvec![arg0.clone(), arg1.clone()], CallInfo::default());

    let include_bodies = call.toposort_call_aware(true);
    assert!(
        include_bodies.iter().any(|u| matches!(u.op(), Op::Param(ops::Param { arg, .. }) if arg.slot == 0)),
        "expected CALL body params"
    );

    let preserve_boundaries = call.toposort_call_aware(false);
    assert!(preserve_boundaries.iter().any(|u| matches!(u.op(), Op::Call(..))), "expected CALL node itself");
    assert!(preserve_boundaries.iter().any(|u| Arc::ptr_eq(u, &arg0)));
    assert!(preserve_boundaries.iter().any(|u| Arc::ptr_eq(u, &arg1)));
    assert!(!preserve_boundaries.iter().any(|u| matches!(u.op(), Op::Param(..))), "CALL body should be excluded");

    let sink = UOp::sink(vec![call.clone()]);
    let program = program(sink.clone(), DeviceSpec::Cpu, None, None, None);
    let program_include = program.toposort_call_aware(true);
    assert!(program_include.iter().any(|u| Arc::ptr_eq(u, &sink)));
    let program_preserve = program.toposort_call_aware(false);
    assert!(!program_preserve.iter().any(|u| Arc::ptr_eq(u, &sink)), "PROGRAM internals should be excluded");
}

#[test]
fn test_substitute_preserve_calls_keeps_call_body() {
    let p0 = UOp::param(0, 1, DType::Float32, None);
    let p1 = UOp::param(1, 1, DType::Float32, None);
    let body = p0.try_add(&p1).unwrap();
    let call = body.call(smallvec![UOp::native_const(10.0f32), UOp::native_const(20.0f32)], CallInfo::default());

    let mut map = HashMap::new();
    map.insert(UOpKey(p0.clone()), UOp::native_const(0.0f32));

    let rewritten_preserve = call.substitute_preserve_calls(&map);
    match rewritten_preserve.op() {
        Op::Call(ops::Call { body: new_body, .. }) => {
            assert!(Arc::ptr_eq(new_body, &body), "CALL body should stay untouched")
        }
        op => panic!("expected Call op, got {op:?}"),
    }

    let rewritten_full = call.substitute(&map);
    match rewritten_full.op() {
        Op::Call(ops::Call { body: new_body, .. }) => {
            assert!(!Arc::ptr_eq(new_body, &body), "full substitute should rewrite CALL body")
        }
        op => panic!("expected Call op, got {op:?}"),
    }
}

#[test]
fn test_graph_rewrite_preserve_calls_can_rewrite_call_node() {
    let body = UOp::param(0, 1, DType::Float32, None);
    let arg = UOp::native_const(7.0f32);
    let call = body.call(smallvec![arg.clone()], CallInfo::default());

    let rewritten = crate::graph_rewrite_preserve_calls(&RewriteCallToFirstArg, call, &mut ());
    assert!(Arc::ptr_eq(&rewritten, &arg), "preserve-calls rewrite should still match and rewrite CALL node");
}

#[test]
fn test_substitute_preserve_calls_rewrites_args_not_body() {
    let p0 = UOp::param(0, 1, DType::Float32, None);
    let p1 = UOp::param(1, 1, DType::Float32, None);
    let body = p0.try_add(&p1).unwrap();

    let arg0 = UOp::native_const(10.0f32);
    let arg1 = UOp::native_const(20.0f32);
    let call = body.call(smallvec![arg0.clone(), arg1], CallInfo::default());

    let mut map = HashMap::new();
    let arg_replacement = UOp::native_const(11.0f32);
    map.insert(UOpKey(arg0.clone()), arg_replacement.clone());
    map.insert(UOpKey(p0.clone()), UOp::native_const(12.0f32));

    let rewritten_preserve = call.substitute_preserve_calls(&map);
    match rewritten_preserve.op() {
        Op::Call(ops::Call { body: new_body, args, .. }) => {
            assert!(Arc::ptr_eq(new_body, &body), "preserve-calls substitute should keep CALL body untouched");
            assert!(Arc::ptr_eq(&args[0], &arg_replacement), "preserve-calls substitute should rewrite CALL args");
        }
        op => panic!("expected Call op, got {op:?}"),
    }

    let rewritten_full = call.substitute(&map);
    match rewritten_full.op() {
        Op::Call(ops::Call { body: new_body, args, .. }) => {
            assert!(!Arc::ptr_eq(new_body, &body), "full substitute should rewrite CALL body");
            assert!(Arc::ptr_eq(&args[0], &arg_replacement), "full substitute should also rewrite CALL args");
        }
        op => panic!("expected Call op, got {op:?}"),
    }
}

#[test]
fn test_buffer_creation() {
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    assert!(matches!(buf.op(), Op::Buffer(..)));
    assert_eq!(buf.dtype(), DType::Float32);

    if let Op::Buffer(ops::Buffer { arg, .. }) = buf.op() {
        assert_eq!(arg.device, Some(DeviceSpec::Cpu));
        assert_eq!(buf.shape().unwrap().unwrap().as_slice(), &[SInt::Const(100)]);
    } else {
        panic!("Expected Buffer op");
    }
}

#[test]
fn test_buffer_hash_consing() {
    // Two buffers with same device and size should NOT be the same
    // (due to different UNIQUE identifiers)
    let buf1 = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let buf2 = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    assert!(!Arc::ptr_eq(&buf1, &buf2), "Different buffers should have different UNIQUE ids");
}

#[test]
fn test_buffer_hash_consing_distinguishes_slots() {
    let shape = crate::shape::shape_to_uop(&smallvec![SInt::Const(64)]);
    let arg0 = crate::ParamArg::buffer(0, DType::Float32, svod_dtype::AddrSpace::Global, Some(DeviceSpec::Cpu));
    let arg1 = crate::ParamArg::buffer(1, DType::Float32, svod_dtype::AddrSpace::Global, Some(DeviceSpec::Cpu));
    let buf0 = UOp::new(Op::Buffer(ops::Buffer { shape: shape.clone(), arg: arg0.into() }), DType::Float32);
    let buf1 = UOp::new(Op::Buffer(ops::Buffer { shape, arg: arg1.into() }), DType::Float32);
    assert!(!Arc::ptr_eq(&buf0, &buf1), "BUFFER slot is part of structural identity");
}

#[test]
fn test_has_buffer_identity_through_get_tuple_chain() {
    use smallvec::smallvec;
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let scratch = UOp::new(Op::Noop, DType::Float32);

    let tup = UOp::tuple(smallvec![buf.clone(), scratch.clone()]);
    let projected_buf = tup.gettuple(0);
    let projected_scratch = tup.gettuple(1);

    assert!(
        projected_buf.has_buffer_identity(),
        "GETTUPLE pointing at a BUFFER element of TUPLE must report buffer identity"
    );
    assert!(
        !projected_scratch.has_buffer_identity(),
        "GETTUPLE pointing at a non-buffer element must not report buffer identity"
    );
}

#[test]
fn test_index_operation() {
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let idx = UOp::const_(DType::Index, ConstValue::UInt(10));

    let indexed = UOp::index().buffer(buf).indices(vec![idx]).call().expect("index should succeed");
    assert!(matches!(indexed.op(), Op::Index(..)));
    assert_eq!(indexed.op().children().len(), 2); // buffer + 1 index
}

#[test]
fn test_copy_device_metadata_and_unique() {
    let src = UOp::new_buffer(DeviceSpec::Cpu, 1, DType::Float32);
    let cpu_copy = src.copy_to_device(DeviceSpec::Cpu);
    let cuda_copy = src.copy_to_device(DeviceSpec::Cuda { device_id: 0 });
    assert!(matches!(cpu_copy.op(), Op::Copy(ops::Copy { device: DeviceSpec::Cpu, .. })));
    assert_eq!(cpu_copy.op().children().len(), 1);
    assert!(!Arc::ptr_eq(&cpu_copy, &cuda_copy), "copy target must participate in hash consing");
    assert!(matches!(cpu_copy.with_sources(vec![src]).op(), Op::Copy(ops::Copy { device: DeviceSpec::Cpu, .. })));

    let uniq = UOp::buffer_id(Some(42));
    assert!(matches!(uniq.op(), Op::Unique(42)));

    let uniq_auto = UOp::buffer_id(None);
    assert!(matches!(uniq_auto.op(), Op::Unique(_)));

    let luniq = UOp::lunique(Some(123));
    assert!(matches!(luniq.op(), Op::LUnique(123)));
}

#[test]
fn test_call_constructor_and_with_sources() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let body = a.try_add(&b).unwrap();

    let info = CallInfo {
        grad_tag: Some("grad_add".to_string()),
        name: Some("call_add".to_string()),
        precompile: true,
        precompile_backward: false,
        ..Default::default()
    };
    let call = body.call(smallvec![a.clone(), b.clone()], info.clone());

    // Per tinygrad spec, CALL dtype is always void.
    assert_eq!(call.dtype(), DType::Void);
    match call.op() {
        Op::Call(ops::Call { body: call_body, args, info: call_info }) => {
            assert!(Arc::ptr_eq(call_body, &body));
            assert_eq!(args.len(), 2);
            assert_eq!(**call_info, info);
        }
        op => panic!("expected Call op, got {op:?}"),
    }

    let c = UOp::native_const(3.0f32);
    let new_body = b.try_mul(&c).unwrap();
    let rewritten = call.with_sources(vec![new_body.clone(), c.clone(), a.clone()]);
    match rewritten.op() {
        Op::Call(ops::Call { body: call_body, args, info: call_info }) => {
            assert!(Arc::ptr_eq(call_body, &new_body));
            assert_eq!(args.len(), 2);
            assert!(Arc::ptr_eq(&args[0], &c));
            assert!(Arc::ptr_eq(&args[1], &a));
            assert_eq!(**call_info, info);
        }
        op => panic!("expected rewritten Call op, got {op:?}"),
    }
}

#[test]
fn test_function_constructor_with_sources_shape_and_hash() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let body = a.try_add(&b).unwrap();

    let info = CallInfo {
        grad_tag: Some("grad_fn".to_string()),
        name: Some("fn_add".to_string()),
        precompile: true,
        precompile_backward: true,
        ..Default::default()
    };
    let function = body.function(smallvec![a.clone(), b.clone()], info.clone());

    // Per tinygrad spec, FUNCTION dtype is always void and body is TUPLE-wrapped.
    assert_eq!(function.dtype(), DType::Void);
    // FUNCTION body is a TUPLE of values, which has no shape itself; querying the
    // shape of an element requires GETTUPLE.
    assert!(function.shape().unwrap().is_none());
    assert_eq!(function.op().range_ending_src_index(), Some(1));

    match function.op() {
        Op::Function(ops::Function { body: fn_body, args, info: fn_info }) => {
            // Auto-wrapped non-Tuple body in a single-element TUPLE.
            let Op::Tuple(ops::Tuple { src }) = fn_body.op() else {
                panic!("expected TUPLE body, got {:?}", fn_body.op())
            };
            assert_eq!(src.len(), 1);
            assert!(Arc::ptr_eq(&src[0], &body));
            assert_eq!(args.len(), 2);
            assert_eq!(**fn_info, info);
        }
        op => panic!("expected Function op, got {op:?}"),
    }

    let c = UOp::native_const(3.0f32);
    let new_body_inner = b.try_mul(&c).unwrap();
    let new_tuple_body = UOp::tuple(smallvec![new_body_inner.clone()]);
    // with_sources expects positional new sources matching children() order:
    // [body, args...]; the new body must already be a TUPLE.
    let rewritten = function.with_sources(vec![new_tuple_body.clone(), c.clone(), a.clone()]);
    match rewritten.op() {
        Op::Function(ops::Function { body: fn_body, args, info: fn_info }) => {
            assert!(Arc::ptr_eq(fn_body, &new_tuple_body));
            assert_eq!(args.len(), 2);
            assert!(Arc::ptr_eq(&args[0], &c));
            assert!(Arc::ptr_eq(&args[1], &a));
            assert_eq!(**fn_info, info);
        }
        op => panic!("expected rewritten Function op, got {op:?}"),
    }

    let function_same = body.function(smallvec![a.clone(), b.clone()], info.clone());
    assert!(Arc::ptr_eq(&function, &function_same), "same function info should hash-cons");

    let info_other = CallInfo { name: Some("other_name".to_string()), ..info };
    let function_other = body.function(smallvec![a, b], info_other);
    assert!(!Arc::ptr_eq(&function, &function_other), "different CallInfo should not hash-cons");
}

#[test]
fn test_tuple_constructor_void_dtype() {
    use svod_ir::Op;
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2i32);
    let t = UOp::tuple(smallvec![a.clone(), b.clone()]);
    assert_eq!(t.dtype(), DType::Void);
    let Op::Tuple(ops::Tuple { src }) = t.op() else { panic!("expected TUPLE, got {:?}", t.op()) };
    assert_eq!(src.len(), 2);
    assert!(Arc::ptr_eq(&src[0], &a));
    assert!(Arc::ptr_eq(&src[1], &b));
}

#[test]
fn test_gettuple_extracts_element_dtype_and_shape() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2i32);
    let t = UOp::tuple(smallvec![a.clone(), b.clone()]);
    let g0 = t.gettuple(0);
    let g1 = t.gettuple(1);
    assert_eq!(g0.dtype(), DType::Float32);
    assert_eq!(g1.dtype(), DType::Int32);
    // GETTUPLE shape mirrors the element's shape.
    assert_eq!(g0.shape().unwrap().cloned(), a.shape().unwrap().cloned());
    assert_eq!(g1.shape().unwrap().cloned(), b.shape().unwrap().cloned());
}

#[test]
fn test_gettuple_through_function_body() {
    let value = UOp::native_const(1.0f32);
    let grad = UOp::native_const(2.0f32);
    let body = UOp::tuple(smallvec![value.clone(), grad.clone()]);
    let function = body.function(smallvec![], CallInfo::default());
    let g0 = function.gettuple(0);
    let g1 = function.gettuple(1);
    assert_eq!(g0.dtype(), DType::Float32);
    assert_eq!(g1.dtype(), DType::Float32);
}

#[test]
fn test_function_keeps_tuple_body_as_is() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let t = UOp::tuple(smallvec![a, b]);
    let function = t.clone().function(smallvec![], CallInfo::default());
    let Op::Function(ops::Function { body, .. }) = function.op() else { panic!("expected FUNCTION") };
    assert!(Arc::ptr_eq(body, &t));
}

#[test]
fn test_tuple_hash_consing() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let t1 = UOp::tuple(smallvec![a.clone(), b.clone()]);
    let t2 = UOp::tuple(smallvec![a, b]);
    assert!(Arc::ptr_eq(&t1, &t2));
}

#[test]
fn test_program_family_constructors_and_with_sources() {
    let sink = UOp::sink(vec![]);
    let linear = UOp::linear(smallvec![UOp::noop()]);
    let source = UOp::source("void kernel() {}".to_string());
    let binary = UOp::binary(vec![1, 2, 3, 4]);
    assert_eq!(binary.dtype(), DType::UInt8);

    let stage0 = program(sink.clone(), DeviceSpec::Cpu, None, None, None);
    assert_eq!(stage0.op().sources().iter().map(|u| u.id).collect::<Vec<_>>(), vec![sink.id]);
    let other_target = program(sink.clone(), DeviceSpec::Cuda { device_id: 0 }, None, None, None);
    assert!(!Arc::ptr_eq(&stage0, &other_target), "PROGRAM target participates in hash consing");
    let stage1 = program(sink.clone(), DeviceSpec::Cpu, Some(linear.clone()), None, None);
    assert_eq!(stage1.op().sources().iter().map(|u| u.id).collect::<Vec<_>>(), vec![sink.id, linear.id]);
    let stage2 = program(sink.clone(), DeviceSpec::Cpu, Some(linear.clone()), Some(source.clone()), None);
    assert_eq!(stage2.op().sources().iter().map(|u| u.id).collect::<Vec<_>>(), vec![sink.id, linear.id, source.id]);

    let program =
        program(sink.clone(), DeviceSpec::Cpu, Some(linear.clone()), Some(source.clone()), Some(binary.clone()));
    assert_eq!(program.op().children().len(), 4);
    assert_eq!(
        program.op().sources().iter().map(|u| u.id).collect::<Vec<_>>(),
        vec![sink.id, linear.id, source.id, binary.id]
    );
    match program.op() {
        Op::Program(ops::Program {
            sink: p_sink,
            info,
            linear: Some(p_linear),
            source: Some(p_source),
            binary: Some(p_binary),
        }) => {
            assert!(Arc::ptr_eq(p_sink, &sink));
            assert_eq!(info.target, DeviceSpec::Cpu);
            assert!(Arc::ptr_eq(p_linear, &linear));
            assert!(Arc::ptr_eq(p_source, &source));
            assert!(Arc::ptr_eq(p_binary, &binary));
        }
        op => panic!("expected Program op with all stages, got {op:?}"),
    }

    let sink2 = UOp::sink(vec![UOp::noop()]);
    let linear2 = UOp::linear(smallvec![UOp::native_const(7i32)]);
    let source2 = UOp::source("void kernel2() {}".to_string());
    let binary2 = UOp::binary(vec![9, 8]);
    let rewritten = program.with_sources(vec![sink2.clone(), linear2.clone(), source2.clone(), binary2.clone()]);
    match rewritten.op() {
        Op::Program(ops::Program {
            sink: p_sink,
            info,
            linear: Some(p_linear),
            source: Some(p_source),
            binary: Some(p_binary),
        }) => {
            assert!(Arc::ptr_eq(p_sink, &sink2));
            assert_eq!(info.target, DeviceSpec::Cpu);
            assert!(Arc::ptr_eq(p_linear, &linear2));
            assert!(Arc::ptr_eq(p_source, &source2));
            assert!(Arc::ptr_eq(p_binary, &binary2));
        }
        op => panic!("expected rewritten Program op with all stages, got {op:?}"),
    }
}

#[test]
fn stage_identity_participates_in_hash_consing_and_content_hash() {
    let identity = crate::SourceStageIdentity {
        version: crate::SOURCE_STAGE_IDENTITY_VERSION,
        abi: vec![],
        target: DeviceSpec::Cpu,
        entry_name: "kernel".into(),
        linear_sha256: crate::StageDigest([1; 32]),
        source_sha256: crate::StageDigest([2; 32]),
    };
    let other_identity = crate::SourceStageIdentity { entry_name: "other".into(), ..identity.clone() };
    let raw = UOp::source("source".into());
    let first = UOp::source_with_identity("source".into(), identity.clone());
    let same = UOp::source_with_identity("source".into(), identity.clone());
    let other = UOp::source_with_identity("source".into(), other_identity);
    assert!(Arc::ptr_eq(&first, &same));
    assert_ne!(raw.content_hash, first.content_hash);
    assert_ne!(first.content_hash, other.content_hash);
    assert_ne!(crate::UOpKey(first.clone()), crate::UOpKey(other));

    let binary_identity = crate::BinaryStageIdentity {
        version: crate::BINARY_STAGE_IDENTITY_VERSION,
        source: identity.clone(),
        compiler_key: "compiler-a".into(),
        binary_sha256: crate::StageDigest([3; 32]),
    };
    let other_binary_identity =
        crate::BinaryStageIdentity { compiler_key: "compiler-b".into(), ..binary_identity.clone() };
    let raw_binary = UOp::binary(vec![1, 2, 3]);
    let binary = UOp::binary_with_identity(vec![1, 2, 3], binary_identity);
    let other_binary = UOp::binary_with_identity(vec![1, 2, 3], other_binary_identity);
    assert_ne!(raw_binary.content_hash, binary.content_hash);
    assert_ne!(binary.content_hash, other_binary.content_hash);
    assert_ne!(crate::UOpKey(binary), crate::UOpKey(other_binary));
}

#[test]
fn test_program_info_from_sink_is_structural_program_identity() {
    let param0 = UOp::param(0, 8, DType::Float32, None);
    let param1 = UOp::param(1, 8, DType::Float32, None);
    let index = UOp::index_const(0);
    let load_index = UOp::index().buffer(param1).indices(vec![index.clone()]).call().unwrap();
    let load = UOp::load().index(load_index).call();
    let store_index = UOp::index().buffer(param0).indices(vec![index]).call().unwrap();
    let store = store_index.store(load);
    let var = UOp::define_var("n".to_string(), 1, 16);
    let global = UOp::special(var.clone(), "gidx0".to_string());
    let local = UOp::special(UOp::index_const(4), "lidx0".to_string());
    let sink = UOp::sink_with_info(
        vec![store, global, local],
        crate::KernelInfo { name: Some("named_kernel".to_string()), ..Default::default() },
    );

    let info = crate::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    assert_eq!(info.name, "named_kernel");
    assert_eq!(info.globals, vec![0, 1]);
    assert_eq!(info.outs, vec![0]);
    assert_eq!(info.ins, vec![1]);
    assert_eq!(info.vars.len(), 1);
    assert_eq!(info.global_size[0].vmax(), &ConstValue::Int(16));
    assert_eq!(info.local_size.as_ref().unwrap()[0].vmax(), &ConstValue::Int(4));

    let first = UOp::program(sink.clone(), info.clone(), None, None, None);
    let second = UOp::program(sink.clone(), info.clone(), None, None, None);
    assert!(Arc::ptr_eq(&first, &second));

    let mut renamed = info;
    renamed.name = "other".to_string();
    let other = UOp::program(sink, renamed, None, None, None);
    assert!(!Arc::ptr_eq(&first, &other));
}

#[test]
fn test_program_info_simplifies_special_launch_extent() {
    let n = UOp::define_var("n".to_string(), 1, 16);
    let extent =
        n.mul(&UOp::const_(DType::WeakInt, ConstValue::Int(1))).add(&UOp::const_(DType::WeakInt, ConstValue::Int(0)));
    let sink = UOp::sink(vec![UOp::special(extent, "gidx0".to_string())]);

    let info = crate::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    assert!(Arc::ptr_eq(&info.global_size[0], &n));
}

#[test]
fn test_program_info_defaults_and_shrink_buffer_identity() {
    let defaults = crate::ProgramInfo::default();
    assert_eq!(defaults.name, "test");
    assert!(defaults.local_size.is_none());
    assert!(defaults.vars.is_empty());
    assert!(defaults.globals.is_empty());
    assert!(defaults.outs.is_empty());
    assert!(defaults.ins.is_empty());

    let param = UOp::param(3, 8, DType::Float32, None);
    let shrink = param.try_shrink(&[(crate::SInt::Const(0), crate::SInt::Const(1))]).unwrap();
    let sink = UOp::sink(vec![shrink.store(UOp::native_const(1.0f32))]);
    let info = crate::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    assert_eq!(info.globals, vec![3]);
    assert_eq!(info.outs, vec![3]);
}

#[test]
fn test_program_info_discovers_cast_shrink_memory() {
    let output = UOp::param(3, 8, DType::Float32, None);
    let input = UOp::param(4, 8, DType::Float32, None);
    let output = output.try_shrink(&[(crate::SInt::Const(0), crate::SInt::Const(1))]).unwrap().cast(DType::Float32);
    let input = input.try_shrink(&[(crate::SInt::Const(0), crate::SInt::Const(1))]).unwrap().cast(DType::Float32);
    let sink = UOp::sink(vec![output.store(UOp::load().index(input).call())]);

    let info = crate::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    assert_eq!(info.outs, vec![3]);
    assert_eq!(info.ins, vec![4]);
}

#[test]
fn test_placeholder_like_concrete_shape() {
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 6, DType::Float32);
    let shaped = buf.try_reshape(&Shape::from_iter([SInt::Const(2), SInt::Const(3)])).unwrap();

    let placeholder = UOp::placeholder_like(&shaped, 7, AddrSpace::Global).expect("placeholder_like should succeed");
    let placeholder_shape = placeholder.shape().unwrap().cloned().expect("placeholder should have shape");
    assert_eq!(placeholder_shape.len(), 2);
    assert_eq!(placeholder_shape[0].as_const(), Some(2));
    assert_eq!(placeholder_shape[1].as_const(), Some(3));

    match placeholder.op() {
        Op::Reshape(ops::Reshape { src, .. }) => match src.op() {
            Op::Param(ops::Param { shape, arg }) => {
                assert_eq!(arg.slot, 7);
                assert!(matches!(shape.op(), Op::Const(value) if value.0 == ConstValue::Int(6)));
            }
            op => panic!("expected PARAM under RESHAPE, got {op:?}"),
        },
        op => panic!("expected RESHAPE placeholder, got {op:?}"),
    }
}

#[test]
fn test_placeholder_like_reg_preserves_shape_and_address_space() {
    let shaped = UOp::new_buffer(DeviceSpec::Cpu, 6, DType::Float32)
        .try_reshape(&Shape::from_iter([SInt::Const(2), SInt::Const(3)]))
        .unwrap();

    let placeholder = UOp::placeholder_like(&shaped, 7, AddrSpace::Reg).expect("REG placeholder_like");
    assert_eq!(placeholder.addrspace(), Some(AddrSpace::Reg));
    assert_eq!(
        placeholder.shape().unwrap().unwrap().iter().map(SInt::as_const).collect::<Vec<_>>(),
        vec![Some(2), Some(3)]
    );
    assert!(placeholder.toposort().iter().any(
        |node| matches!(node.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.slot == 7 && arg.addrspace == Some(AddrSpace::Reg))
    ));
}

#[test]
fn test_placeholder_like_commits_weak_storage_dtype() {
    let weak = UOp::const_(DType::WeakInt, ConstValue::Int(3));
    let placeholder =
        UOp::placeholder_like(&weak, 2, AddrSpace::Global).expect("weak placeholder should commit storage dtype");

    assert_eq!(placeholder.dtype(), DType::Int32);
    assert!(matches!(placeholder.op(), Op::Param(ops::Param { arg, .. }) if arg.dtype == DType::Int32));
}

#[test]
fn test_placeholder_like_symbolic_shape_fails() {
    // Symbolic input is rejected outright — tinygrad's placeholder_like
    // asserts the shape is all concrete ints, and we mirror that contract.
    let n = UOp::define_var("N".to_string(), 1, 8);
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let shaped = buf.try_reshape(&Shape::from_iter([SInt::from(n)])).unwrap();

    let err = UOp::placeholder_like(&shaped, 0, AddrSpace::Global).expect_err("symbolic placeholder_like should fail");
    assert!(format!("{err}").contains("symbolic shape is not supported"), "unexpected error: {err}");
}

#[test]
fn test_placeholder_like_multi_uses_shard_shape() {
    let shard = UOp::new_buffer(DeviceSpec::Cpu, 6, DType::Float32)
        .try_reshape(&Shape::from_iter([SInt::Const(2), SInt::Const(3)]))
        .unwrap();
    let multi = UOp::multi(shard, 0);

    let placeholder = UOp::placeholder_like(&multi, 3, AddrSpace::Global)
        .expect("placeholder_like should succeed for MULTI shard shape");
    let shape = placeholder.shape().unwrap().cloned().expect("placeholder should have shape");
    assert_eq!(shape.iter().map(|d| d.as_const()).collect::<Vec<_>>(), vec![Some(2), Some(3)]);
}

#[test]
fn test_placeholder_like_mstack_mselect_uses_buffer_shape() {
    let shard0 = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32)
        .try_reshape(&Shape::from_iter([SInt::Const(2), SInt::Const(2)]))
        .unwrap();
    let shard1 = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32)
        .try_reshape(&Shape::from_iter([SInt::Const(2), SInt::Const(2)]))
        .unwrap();
    let stacked = UOp::mstack(smallvec::smallvec![shard0, shard1]);
    let selected = stacked.mselect(1);

    let placeholder =
        UOp::placeholder_like(&selected, 4, AddrSpace::Global).expect("placeholder_like should succeed for MSELECT");
    let shape = placeholder.shape().unwrap().cloned().expect("placeholder should have shape");
    assert_eq!(shape.iter().map(|d| d.as_const()).collect::<Vec<_>>(), vec![Some(2), Some(2)]);
}

#[test]
fn test_custom_kernel_builds_after_call_outputs() {
    let a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let b = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let outputs = UOp::custom_kernel(
        vec![a.clone(), b.clone()],
        |placeholders| {
            assert_eq!(placeholders.len(), 2);
            UOp::sink(vec![placeholders[0].clone(), placeholders[1].clone()])
        },
        CallInfo::default(),
    )
    .expect("custom kernel should build");

    assert_eq!(outputs.len(), 2);
    for out in outputs {
        match out.op() {
            Op::After(ops::After { passthrough, deps }) => {
                assert!(matches!(passthrough.op(), Op::Buffer(..)));
                assert_eq!(deps.len(), 1);
                match deps[0].op() {
                    Op::Call(ops::Call { body, args, .. }) => {
                        assert!(matches!(body.op(), Op::Sink(..)));
                        assert_eq!(args.len(), 2);
                    }
                    op => panic!("expected CALL dep, got {op:?}"),
                }
            }
            op => panic!("expected AFTER output, got {op:?}"),
        }
    }
}

/// A reshape of a realized input is the same bytes: the kernel binds the
/// realized node, not a copy, while the placeholder keeps the reshaped shape.
#[test]
fn test_custom_kernel_binds_reshaped_after_input_without_a_copy() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let dep = UOp::new_buffer(DeviceSpec::Cpu, 1, DType::Float32);
    let realized = buffer.after(smallvec![dep]);
    let viewed = realized.try_reshape(&smallvec![SInt::Const(2), SInt::Const(4)]).expect("reshape");

    let outputs = UOp::custom_kernel(
        vec![viewed],
        |placeholders| {
            let shape = placeholders[0].shape().unwrap().cloned().expect("placeholder shape");
            assert_eq!(shape.iter().map(|d| d.as_const()).collect::<Vec<_>>(), vec![Some(2), Some(4)]);
            UOp::sink(vec![placeholders[0].clone()])
        },
        CallInfo::default(),
    )
    .expect("custom kernel should build");

    let Op::After(ops::After { passthrough, deps }) = outputs[0].op() else { panic!("expected AFTER output") };
    assert!(Arc::ptr_eq(passthrough, &realized), "the realized node is bound, not a copy of its view");
    let Op::Call(ops::Call { args, .. }) = deps[0].op() else { panic!("expected CALL dep") };
    assert!(Arc::ptr_eq(&args[0], &realized));
}

#[test]
fn test_custom_kernel_value_body_wraps_in_function() {
    // A value-producing body (here a binary Add) routes through Op::Function
    // with a TUPLE-wrapped body, while opaque bodies (Sink, Program, ...)
    // keep using Op::Call. Mirrors tinygrad's _OPAQUE_CALL_BODIES dispatch.
    let a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let b = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let outputs = UOp::custom_kernel(
        vec![a.clone(), b.clone()],
        |placeholders| {
            assert_eq!(placeholders.len(), 2);
            placeholders[0].try_add(&placeholders[1]).expect("placeholders should be addable")
        },
        CallInfo::default(),
    )
    .expect("custom kernel with value body should build");

    assert_eq!(outputs.len(), 2);
    for out in outputs {
        let Op::After(ops::After { deps, .. }) = out.op() else {
            panic!("expected AFTER output, got {:?}", out.op());
        };
        assert_eq!(deps.len(), 1);
        match deps[0].op() {
            Op::Function(ops::Function { body, args, .. }) => {
                assert_eq!(args.len(), 2, "function should receive contig srcs as args");
                assert!(
                    matches!(body.op(), Op::Tuple(..)),
                    "non-tuple body must auto-wrap into TUPLE for FUNCTION dispatch, got {:?}",
                    body.op()
                );
            }
            op => panic!("value-producing body must dispatch as FUNCTION, got {op:?}"),
        }
    }
}

#[test]
fn test_custom_kernel_opaque_call_function_body_uses_call() {
    // Tinygrad's _OPAQUE_CALL_BODIES includes CUSTOM_FUNCTION; verify it
    // dispatches as Op::Call and not auto-wrapped via FUNCTION.
    use crate::types::CustomFunctionKind;
    let a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let outputs = UOp::custom_kernel(
        vec![a.clone()],
        |_placeholders| UOp::custom_function(CustomFunctionKind::EncDec, smallvec::smallvec![UOp::index_const(0)]),
        CallInfo::default(),
    )
    .expect("custom kernel with custom_function body should build");

    let Op::After(ops::After { deps, .. }) = outputs[0].op() else {
        panic!("expected AFTER output, got {:?}", outputs[0].op());
    };
    assert!(
        matches!(deps[0].op(), Op::Call(..)),
        "CustomFunction body is opaque — must dispatch via CALL, got {:?}",
        deps[0].op()
    );
}

/// `children` and `map_child` walk the same sources in the same order.
#[test]
fn test_children_accessors_agree() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let add = a.try_add(&b).unwrap();

    let mut mapped = Vec::new();
    add.op().map_child(|child| mapped.push(child.clone()));

    assert_eq!(add.op().children().len(), 2);
    for (child, expected) in add.op().children().iter().zip([&a, &b]) {
        assert!(Arc::ptr_eq(child, expected));
    }
    for (child, expected) in mapped.iter().zip([&a, &b]) {
        assert!(Arc::ptr_eq(child, expected));
    }
}

// ============================================================================
// Cached properties
// ============================================================================

/// A property is computed on first access, cached in place, and every later access
/// hands back the very same reference.
fn assert_lazy_and_memoised<P: crate::uop::cached_property::CachedProperty>(uop: &Arc<UOp>) {
    assert!(P::cache(uop).get().is_none(), "cache must be cold before the first access");
    let first = P::get(uop);
    assert!(P::cache(uop).get().is_some(), "cache must be populated after the first access");
    assert!(std::ptr::eq(first, P::get(uop)), "later accesses must return the cached reference");
}

#[test]
fn test_properties_are_lazy_and_memoised() {
    use crate::uop::properties::{InScopeRangesProperty, RangesProperty, ShapeProperty};

    // A fresh axis id and variable name keep this graph out of every other test's
    // interning results, so the caches really are cold.
    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(9101), crate::AxisType::Loop);
    let node = range.cast(DType::Float32).try_add(&UOp::var("lazy_probe", DType::Float32, 0, 1)).unwrap();

    assert_lazy_and_memoised::<ShapeProperty>(&node);
    assert_lazy_and_memoised::<RangesProperty>(&node);
    assert_lazy_and_memoised::<InScopeRangesProperty>(&node);
}

/// Plain arithmetic over scalars: empty shape, no ranges, nothing in scope.
#[test]
fn test_properties_of_a_scalar_graph() {
    let add = UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).unwrap();

    assert_eq!(add.shape().unwrap().expect("scalars are shaped").len(), 0);
    assert!(add.ranges().is_empty());
    assert!(add.in_scope_ranges().is_empty());
}

/// A RANGE is in its own scope and stays in scope for everything derived from it, until
/// an END closes it.
#[test]
fn test_in_scope_ranges_open_and_close() {
    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), crate::AxisType::Loop);
    let derived = range.cast(DType::Float32);

    assert_eq!(range.ranges().len(), 1);
    assert!(Arc::ptr_eq(&range.ranges()[0], &range));
    assert_eq!(range.in_scope_ranges().len(), 1, "RANGE has itself in scope");
    assert_eq!(derived.in_scope_ranges().len(), 1, "derived computation inherits the scope");
    assert!(UOp::native_const(1.0f32).end(smallvec![range]).in_scope_ranges().is_empty(), "END closes the scope");
}

/// The gate blocks traversal, so a node that fails it contributes neither itself nor its
/// children.
#[test]
fn test_toposort_filtered_gates_traversal() {
    let a = UOp::native_const(1.0f32);
    let b = a.try_add(&UOp::native_const(2.0f32)).unwrap();
    let c = b.try_mul(&UOp::native_const(3.0f32)).unwrap();

    assert_eq!(c.toposort_filtered(|_| true).len(), c.toposort().len());
    assert!(c.toposort_filtered(|_| false).is_empty());

    let only_root = c.toposort_filtered(|node| Arc::ptr_eq(node, &c));
    assert_eq!(only_root.len(), 1);
    assert!(Arc::ptr_eq(&only_root[0], &c));
}

/// The warm-children fast path in `CachedProperty::get` must produce exactly what
/// the filtered-toposort path produces. Both rows build the same diamond shape
/// (`root = r*2 + r*3`, both arms sharing one RANGE) over a distinct axis id so
/// hash consing hands each row a genuinely cold graph.
#[test_case::test_case(true ; "children warmed first takes the fast path")]
#[test_case::test_case(false ; "cold children fall back to filtered toposort")]
fn test_cached_property_diamond_fast_path_matches_slow_path(warm_children: bool) {
    use crate::AxisType;
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::{RangesProperty, VminVmaxProperty};

    let axis = AxisId::Renumbered(if warm_children { 9001 } else { 9002 });
    let range = UOp::range_axis(UOp::index_const(10), axis, AxisType::Loop);
    let left = range.try_mul(&UOp::index_const(2)).unwrap();
    let right = range.try_mul(&UOp::index_const(3)).unwrap();
    let root = left.try_add(&right).unwrap();

    if warm_children {
        RangesProperty::get(&left);
        RangesProperty::get(&right);
        VminVmaxProperty::get(&left);
        VminVmaxProperty::get(&right);
    }
    assert!(RangesProperty::cache(&root).get().is_none(), "root must be cold before the measured get");
    assert!(VminVmaxProperty::cache(&root).get().is_none(), "root must be cold before the measured get");

    let ranges = RangesProperty::get(&root);
    assert_eq!(ranges.len(), 1, "diamond must dedup the shared RANGE");
    assert!(Arc::ptr_eq(&ranges[0], &range));
    // range in [0, 9] => 2*r + 3*r in [0, 45].
    assert_eq!(root.vmin(), &ConstValue::Int(0));
    assert_eq!(root.vmax(), &ConstValue::Int(45));
}

/// `device_spec` and `addrspace` recurse through every child. Before they were
/// memoised, a diamond DAG (each level's two nodes both feeding the next level's
/// two nodes) gave them 2^levels distinct paths: 20 levels took tens of seconds.
/// Both must now resolve in linear time.
#[test]
fn test_device_and_addrspace_are_memoised_on_diamond_dags() {
    use crate::UnaryOp;
    use svod_dtype::AddrSpace;

    let mut a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let mut b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    for level in 0..20 {
        let sum = UOp::new(Op::Binary(BinaryOp::Add, a.clone(), b.clone()), DType::Float32);
        let product = UOp::new(Op::Binary(BinaryOp::Mul, a.clone(), b.clone()), DType::Float32);
        // Distinct ops per level so hash consing cannot collapse the levels.
        let (first, second) =
            if level % 2 == 0 { (UnaryOp::Sqrt, UnaryOp::Exp2) } else { (UnaryOp::Exp2, UnaryOp::Sqrt) };
        a = UOp::new(Op::Unary(first, sum), DType::Float32);
        b = UOp::new(Op::Unary(second, product), DType::Float32);
    }
    let root = UOp::new(Op::Binary(BinaryOp::Add, a, b), DType::Float32);

    // Memoised, both resolve in a linear walk over ~80 nodes; unmemoised they explore 2^20
    // paths each and take tens of seconds. The bound below only has to separate those two
    // complexity classes, so it is five orders of magnitude above the memoised cost.
    let start = std::time::Instant::now();
    assert_eq!(root.device_spec(), Some(DeviceSpec::Cpu), "device must propagate up from the BUFFER leaf");
    assert_eq!(root.addrspace(), Some(AddrSpace::Global), "addrspace must propagate up from the BUFFER leaf");
    let elapsed = start.elapsed();
    assert!(elapsed < std::time::Duration::from_secs(5), "20-level diamond took {elapsed:?}; memo is not working");
}

/// Querying `ranges()`/`in_scope_ranges()` on a RANGE node must not create a
/// self-referential `Arc` cycle: once all external refs drop, the node dies.
/// Guards the property caches against reintroducing the historical leak where
/// a RANGE's own `ranges_cache` held a strong `Arc` to itself.
#[test]
fn range_property_caches_do_not_leak_the_node() {
    let range = UOp::range(UOp::index_const(16), 42_000);
    let value = range.add(&UOp::index_const(1));

    // Populate both caches on every node, including the RANGE itself.
    assert!(value.ranges().iter().any(|r| Arc::ptr_eq(r, &range)));
    assert!(range.ranges().iter().any(|r| Arc::ptr_eq(r, &range)), "ranges() must still report self");
    assert!(value.in_scope_ranges().contains(&range.id));
    assert!(range.in_scope_ranges().contains(&range.id), "in_scope_ranges() must still report self");

    let canary = Arc::downgrade(&range);
    drop(value);
    drop(range);
    assert!(canary.upgrade().is_none(), "RANGE node leaked via its own property caches");
}

/// Dropping a deep graph must not overflow the stack: `Drop for UOp` (the
/// buffer-lifetime hook) recurses through children like the compiler-generated
/// glue before it, so a long dependency chain is the worst case.
#[test]
fn deep_graph_drop_does_not_overflow_stack() {
    let mut node = UOp::index_const(0);
    for i in 1..100_000 {
        node = node.add(&UOp::index_const(i % 512));
    }
    drop(node);
}
