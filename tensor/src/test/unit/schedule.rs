use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use smallvec::SmallVec;
use svod_device::Buffer;
use svod_ir::{AxisId, AxisType, CallInfo, DType, DeviceSpec, Op, ReduceOp, UOp};

use crate::schedule::{
    InputBuffers, KernelInvocation, PreSchedule, PreScheduleItem, ScheduleItem, create_schedule, instantiate_schedule,
};
use svod_ir::ops;

fn cpu_buffer(numel: usize) -> Buffer {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    Buffer::new(alloc, DType::Float32, vec![numel], Default::default())
}

#[test]
fn slice_is_resolved_to_alias_handle_before_runtime_planning() {
    let base_uop = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let intermediate_uop = UOp::new_buffer(DeviceSpec::Cpu, 5, DType::Float32);
    let output_uop = UOp::new_buffer(DeviceSpec::Cpu, 2, DType::Float32);
    let first_slice = base_uop.contiguous_slice(5, 2, DType::Float32);
    let first_call =
        first_slice.call(smallvec::smallvec![intermediate_uop.clone(), base_uop.clone()], CallInfo::default());
    let second_slice = intermediate_uop.contiguous_slice(2, 1, DType::Float32);
    let second_call =
        second_slice.call(smallvec::smallvec![output_uop.clone(), intermediate_uop.clone()], CallInfo::default());
    let pre = PreSchedule {
        items: vec![
            PreScheduleItem {
                kernel: first_call.clone(),
                ast: first_slice,
                sources: vec![intermediate_uop.clone(), base_uop.clone()],
                dependencies: vec![],
                bound_ranges: vec![],
            },
            PreScheduleItem {
                kernel: second_call.clone(),
                ast: second_slice,
                sources: vec![output_uop.clone(), intermediate_uop.clone()],
                dependencies: vec![first_call.id],
                bound_ranges: vec![],
            },
        ],
        invocations: vec![
            KernelInvocation { kernel_id: first_call.id, fixedvars: HashMap::new() },
            KernelInvocation { kernel_id: second_call.id, fixedvars: HashMap::new() },
        ],
        output_buffer_uops: vec![output_uop.clone()],
    };
    let base = cpu_buffer(8);
    let storage = base.storage_id();
    let result = instantiate_schedule(&pre, &HashMap::from([(base_uop.id, base)]), &HashMap::new(), false).unwrap();

    assert!(result.items.is_empty(), "SLICE must not survive as an executable schedule item");
    assert_eq!(result.output_uop_ids, vec![output_uop.id]);
    assert!(!result.alias_output_buffers.contains_key(&intermediate_uop.id));
    let output = &result.alias_output_buffers[&output_uop.id];
    assert_eq!(output.storage_id(), storage);
    assert_eq!(output.offset(), 3 * DType::Float32.bytes());
    assert_eq!(output.size(), 2 * DType::Float32.bytes());
}

fn assert_ir_construction_error_contains(err: crate::Error, needle: &str) {
    match err.kind() {
        crate::ErrorKind::IrConstruction { details } => {
            assert!(details.contains(needle), "expected IrConstruction details to contain '{needle}', got '{details}'");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

fn expect_schedule_error(result: crate::Result<crate::schedule::ScheduleResult>) -> crate::Error {
    match result {
        Ok(_) => panic!("schedule unexpectedly succeeded"),
        Err(err) => err,
    }
}

#[test]
fn test_schedule_item_creation() {
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::new(), CallInfo::default());

    let item = ScheduleItem {
        kernel: call,
        ast: body,
        buffers: vec![],
        buffer_uop_ids: vec![],
        fixedvars: HashMap::new(),
        loop_var_names: HashSet::new(),
        dependencies: vec![],
        instance_dependencies: vec![],
    };

    assert!(matches!(item.kernel.op(), Op::Call(..)));
}

#[test]
fn test_create_schedule_after_uses_canonical_buffer_id() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let after = buffer_uop.after(SmallVec::new());
    let body = UOp::sink(vec![UOp::native_const(0.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![after.clone()]), CallInfo::default());
    let transformed = UOp::sink(vec![call]);

    let mut input_buffers = InputBuffers::new();
    let input = cpu_buffer(4);
    input_buffers.insert(buffer_uop.id, input.clone());

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let item = &result.items[0];

    assert_eq!(item.buffer_uop_ids, vec![buffer_uop.id]);
    assert_eq!(item.buffers.len(), 1);
    assert_eq!(item.buffers[0].id(), input.id());
}

#[test]
fn test_create_schedule_mselect_uses_canonical_buffer_id() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let mstack = UOp::mstack(SmallVec::from_vec(vec![buffer_uop.clone()]));
    let mselect = mstack.mselect(0);
    let body = UOp::sink(vec![UOp::native_const(0.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![mselect.clone()]), CallInfo::default());
    let transformed = UOp::sink(vec![call]);

    let mut input_buffers = InputBuffers::new();
    let input = cpu_buffer(4);
    input_buffers.insert(buffer_uop.id, input.clone());

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let item = &result.items[0];

    assert_eq!(item.buffer_uop_ids, vec![buffer_uop.id]);
    assert_eq!(item.buffers.len(), 1);
    assert_eq!(item.buffers[0].id(), input.id());
}

fn lane_buffers(count: usize) -> Vec<Arc<UOp>> {
    (0..count).map(|_| UOp::new_buffer(DeviceSpec::Cpu, 1, DType::Float32)).collect()
}

fn mstack_of(lanes: &[Arc<UOp>]) -> Arc<UOp> {
    UOp::mstack(SmallVec::from_vec(lanes.to_vec()))
}

/// A CALL whose body computes `body` over `sources`.
fn call_over(body: Arc<UOp>, sources: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::sink(vec![body]).call(SmallVec::from_vec(sources), CallInfo::default())
}

fn lane_inputs(lanes: &[&Arc<UOp>]) -> InputBuffers {
    lanes.iter().map(|lane| (lane.id, cpu_buffer(1))).collect()
}

/// A DEVICE range over `extent` lanes — the axis that turns an expanded CALL
/// into a per-device dispatch.
fn device_axis(extent: Arc<UOp>) -> Arc<UOp> {
    UOp::range_axis(extent, AxisId::Renumbered(0), AxisType::Device)
}

#[test]
fn test_create_schedule_expands_mstack_sources_lane_wise() {
    // Two MSTACK sources must stay aligned lane by lane, while a scalar BIND is
    // shared by every lane instead of being split across them. Without a DEVICE
    // axis in the body no lane gets a `_device_num` binding.
    let (a, b) = (lane_buffers(2), lane_buffers(2));
    let scalar = UOp::define_var("N".into(), 0, 8).bind(UOp::index_const(3));
    let call = call_over(UOp::native_const(0.0f32), vec![mstack_of(&a), mstack_of(&b), scalar]);
    let inputs = lane_inputs(&[&a[0], &a[1], &b[0], &b[1]]);

    let result = create_schedule(UOp::sink(vec![call]), &inputs, &HashMap::new()).unwrap();

    assert_eq!(result.items.len(), 2);
    assert_eq!(result.items[0].buffer_uop_ids, vec![a[0].id, b[0].id]);
    assert_eq!(result.items[1].buffer_uop_ids, vec![a[1].id, b[1].id]);
    assert!(result.items.iter().all(|item| !item.fixedvars.contains_key("_device_num")));
}

#[test]
fn test_create_schedule_binds_device_num_per_lane() {
    let lanes = lane_buffers(2);
    let call = call_over(device_axis(UOp::index_const(2)), vec![mstack_of(&lanes)]);

    let result =
        create_schedule(UOp::sink(vec![call]), &lane_inputs(&[&lanes[0], &lanes[1]]), &HashMap::new()).unwrap();

    assert_eq!(result.items.len(), 2);
    assert_eq!(result.items[0].fixedvars.get("_device_num"), Some(&0));
    assert_eq!(result.items[1].fixedvars.get("_device_num"), Some(&1));
    assert!(result.items.iter().all(|item| !item.loop_var_names.contains("_device_num")));
}

#[test]
fn test_create_schedule_rejects_malformed_multi_device_calls() {
    let (a, b) = (lane_buffers(2), lane_buffers(1));
    let ordinary = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let zero = UOp::native_const(0.0f32);
    let reject = |sources: Vec<Arc<UOp>>, body: Arc<UOp>| {
        expect_schedule_error(create_schedule(
            UOp::sink(vec![call_over(body, sources)]),
            &InputBuffers::new(),
            &HashMap::new(),
        ))
    };

    let err = reject(vec![mstack_of(&a).mselect(2)], zero.clone());
    assert!(
        matches!(err.kind(), crate::ErrorKind::MultiSelectOutOfBounds { device_index: 2, lane_count: 2, .. }),
        "{err:?}"
    );

    let err = reject(vec![UOp::mstack(SmallVec::new())], zero.clone());
    assert!(matches!(err.kind(), crate::ErrorKind::MultiEmptyLanes { source_index: 0, .. }), "{err:?}");

    let err = reject(vec![mstack_of(&a), mstack_of(&b)], zero.clone());
    assert!(
        matches!(err.kind(), crate::ErrorKind::MultiLaneCountMismatch { source_index: 1, expected: 2, actual: 1, .. }),
        "{err:?}"
    );

    // A DEVICE extent must match the lane count and be statically known.
    let err = reject(vec![mstack_of(&a)], device_axis(UOp::index_const(3)));
    assert!(
        matches!(err.kind(), crate::ErrorKind::MultiDeviceExtentMismatch { expected: 2, actual: 3, .. }),
        "{err:?}"
    );
    let dynamic = device_axis(UOp::variable("device_count".into(), 1, 4, DType::Index));
    assert!(matches!(reject(vec![mstack_of(&a)], dynamic).kind(), crate::ErrorKind::MultiDeviceExtentNotStatic { .. }));

    // A DEVICE axis without MSTACK lanes, and MSTACK lanes mixed with a plain
    // buffer, are both unexpanded forms.
    let err = reject(vec![ordinary.clone()], device_axis(UOp::index_const(1)));
    assert!(matches!(err.kind(), crate::ErrorKind::MultiUnsupportedForm { .. }), "{err:?}");
    let err = reject(vec![mstack_of(&a), ordinary.clone()], zero.clone());
    assert!(matches!(err.kind(), crate::ErrorKind::MultiUnsupportedForm { .. }), "{err:?}");

    let sliced = mstack_of(&[a[0].contiguous_slice(1, 0, DType::Float32), a[1].clone()]);
    let err = reject(vec![sliced], zero);
    assert!(matches!(err.kind(), crate::ErrorKind::MultiLaneSliceAlias { source_index: 0, lane: 0, .. }), "{err:?}");
}

#[test]
fn test_create_schedule_rejects_device_binding_conflict() {
    let lanes = lane_buffers(2);
    let call = call_over(UOp::variable("_device_num".into(), 0, 1, DType::Index), vec![mstack_of(&lanes)]);

    let err = expect_schedule_error(create_schedule(
        UOp::sink(vec![call]),
        &lane_inputs(&[&lanes[0], &lanes[1]]),
        &HashMap::from([("_device_num".into(), 0)]),
    ));
    assert!(matches!(
        err.kind(),
        crate::ErrorKind::MultiBindingConflict { name, existing: 0, incoming: 0, .. } if name == "_device_num"
    ));
}

#[test]
fn host_collective_schedule_item_depends_on_every_shard_producer() {
    let output = UOp::new_buffer(DeviceSpec::Cpu, 2, DType::Float32);
    let shard0 = UOp::new_buffer(DeviceSpec::Cpu, 2, DType::Float32);
    let shard1 = UOp::new_buffer(DeviceSpec::Cpu, 2, DType::Float32);
    let producer0 =
        UOp::sink(vec![UOp::native_const(0.0f32)]).call(SmallVec::from_vec(vec![shard0.clone()]), CallInfo::default());
    let producer1 =
        UOp::sink(vec![UOp::native_const(0.0f32)]).call(SmallVec::from_vec(vec![shard1.clone()]), CallInfo::default());
    let formals = SmallVec::from_vec(vec![
        UOp::placeholder_like(&output, 0, svod_ir::AddrSpace::Global).unwrap(),
        UOp::placeholder_like(&shard0, 1, svod_ir::AddrSpace::Global).unwrap(),
        UOp::placeholder_like(&shard1, 2, svod_ir::AddrSpace::Global).unwrap(),
    ]);
    let collective = UOp::custom_function(svod_ir::CustomFunctionKind::AllReduce { reduce_op: ReduceOp::Add }, formals)
        .call(
            SmallVec::from_vec(vec![
                output.clone(),
                shard0.after(SmallVec::from_vec(vec![producer0.clone()])),
                shard1.after(SmallVec::from_vec(vec![producer1.clone()])),
            ]),
            CallInfo::default(),
        );
    let transformed =
        UOp::sink(vec![producer0.clone(), producer1.clone(), output.after(smallvec::smallvec![collective.clone()])]);
    let inputs = HashMap::from([(output.id, cpu_buffer(2)), (shard0.id, cpu_buffer(2)), (shard1.id, cpu_buffer(2))]);

    let schedule = create_schedule(transformed, &inputs, &HashMap::new()).unwrap();
    let item = schedule.items.iter().find(|item| item.kernel.id == collective.id).unwrap();
    let mut expected = vec![producer0.id, producer1.id];
    expected.sort_unstable();
    assert_eq!(item.dependencies, expected);
    assert!(matches!(
        item.ast.op(),
        Op::CustomFunction(ops::CustomFunction { kind: svod_ir::CustomFunctionKind::AllReduce { .. }, .. })
    ));
}

#[test]
fn test_create_schedule_preserves_kernel_dependencies() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let body1 = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let kernel1 = body1.call(SmallVec::from_vec(vec![buffer_uop.clone()]), CallInfo::default());

    let after = buffer_uop.after(SmallVec::from_vec(vec![kernel1.clone()]));

    let body2 = UOp::sink(vec![UOp::native_const(2.0f32)]);
    let kernel2 = body2.call(SmallVec::from_vec(vec![after]), CallInfo::default());

    let transformed = UOp::sink(vec![kernel1.clone(), kernel2.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");

    assert_eq!(result.items.len(), 2);
    let k1_item = result.items.iter().find(|it| it.kernel.id == kernel1.id).expect("k1 item");
    let k2_item = result.items.iter().find(|it| it.kernel.id == kernel2.id).expect("k2 item");

    assert!(k1_item.dependencies.is_empty());
    assert_eq!(k2_item.dependencies, vec![kernel1.id]);
}

#[test]
fn test_create_schedule_supports_call_wrapper() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone()]), CallInfo::default());
    let transformed = UOp::sink(vec![call.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");

    assert_eq!(result.items.len(), 1);
    assert_eq!(result.items[0].kernel.id, call.id);
    assert!(matches!(result.items[0].kernel.op(), Op::Call(..)));
}

#[test]
fn test_create_schedule_preserves_call_dependencies_after_call() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let producer_body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let producer = producer_body.call(SmallVec::from_vec(vec![buffer_uop.clone()]), CallInfo::default());

    let after = buffer_uop.after(SmallVec::from_vec(vec![producer.clone()]));

    let consumer_body = UOp::sink(vec![UOp::native_const(2.0f32)]);
    let consumer = consumer_body.call(SmallVec::from_vec(vec![after]), CallInfo::default());

    let transformed = UOp::sink(vec![producer.clone(), consumer.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    assert_eq!(result.items.len(), 2);

    let producer_item = result.items.iter().find(|it| it.kernel.id == producer.id).expect("producer item");
    let consumer_item = result.items.iter().find(|it| it.kernel.id == consumer.id).expect("consumer item");

    assert!(producer_item.dependencies.is_empty());
    assert_eq!(consumer_item.dependencies, vec![producer.id]);
}

#[test]
fn test_create_schedule_preserves_call_dependencies_end_call() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let producer_body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let producer = producer_body.call(SmallVec::from_vec(vec![buffer_uop.clone()]), CallInfo::default());
    let end_call = producer.end(SmallVec::from_vec(vec![UOp::range_const(1, 0)]));
    let after = buffer_uop.after(SmallVec::from_vec(vec![end_call]));

    let consumer_body = UOp::sink(vec![UOp::native_const(2.0f32)]);
    let consumer = consumer_body.call(SmallVec::from_vec(vec![after]), CallInfo::default());

    let transformed = UOp::sink(vec![producer.clone(), consumer.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let consumer_item = result.items.iter().find(|it| it.kernel.id == consumer.id).expect("consumer item");

    assert_eq!(consumer_item.dependencies, vec![producer.id]);
}

#[test]
fn test_create_schedule_preserves_call_arg_order() {
    let a_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let b_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let body = UOp::sink(vec![UOp::native_const(3.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![b_uop.clone(), a_uop.clone()]), CallInfo::default());
    let transformed = UOp::sink(vec![call]);

    let input_a = cpu_buffer(4);
    let input_b = cpu_buffer(4);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(a_uop.id, input_a.clone());
    input_buffers.insert(b_uop.id, input_b.clone());

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let item = &result.items[0];

    assert_eq!(item.buffer_uop_ids, vec![b_uop.id, a_uop.id]);
    assert_eq!(item.buffers.len(), 2);
    assert_eq!(item.buffers[0].id(), input_b.id());
    assert_eq!(item.buffers[1].id(), input_a.id());
}

#[test]
fn test_create_schedule_unrolls_call_bound_ranges() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let outer_range = UOp::range_axis(UOp::index_const(3), svod_ir::AxisId::Renumbered(0), svod_ir::AxisType::Loop);
    let bind = UOp::define_var("outer_i".to_string(), 0, 2).bind(outer_range.clone());
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone(), bind]), CallInfo::default());
    let end_call = call.end(SmallVec::from_vec(vec![outer_range]));
    let transformed = UOp::sink(vec![call, end_call]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let schedule_result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    assert_eq!(schedule_result.items.len(), 3);
    assert!(schedule_result.items.iter().all(|it| matches!(it.kernel.op(), Op::Call(..))));

    let mut fixed: Vec<i64> =
        schedule_result.items.iter().map(|it| *it.fixedvars.get("outer_i").expect("outer_i fixedvar")).collect();
    fixed.sort_unstable();
    assert_eq!(fixed, vec![0, 1, 2]);
}

#[test]
fn test_create_schedule_collects_all_after_callable_dependencies() {
    let in_a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let in_b = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let p1 =
        UOp::sink(vec![UOp::native_const(1.0f32)]).call(SmallVec::from_vec(vec![in_a.clone()]), CallInfo::default());
    let p2 =
        UOp::sink(vec![UOp::native_const(2.0f32)]).call(SmallVec::from_vec(vec![in_b.clone()]), CallInfo::default());

    let after = passthrough.after(SmallVec::from_vec(vec![p1.clone(), p2.clone()]));
    let consumer =
        UOp::sink(vec![UOp::native_const(3.0f32)]).call(SmallVec::from_vec(vec![after]), CallInfo::default());

    let transformed = UOp::sink(vec![p1.clone(), p2.clone(), consumer.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(in_a.id, cpu_buffer(4));
    input_buffers.insert(in_b.id, cpu_buffer(4));
    input_buffers.insert(passthrough.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let consumer_item = result.items.iter().find(|it| it.kernel.id == consumer.id).expect("consumer item");

    let mut expected = vec![p1.id, p2.id];
    expected.sort_unstable();
    assert_eq!(consumer_item.dependencies, expected);
}

#[test]
fn test_create_schedule_collects_end_call_and_call_dependencies() {
    let in_a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let in_b = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let p1 =
        UOp::sink(vec![UOp::native_const(1.0f32)]).call(SmallVec::from_vec(vec![in_a.clone()]), CallInfo::default());
    let p1_end = p1.end(SmallVec::from_vec(vec![UOp::range_const(1, 0)]));
    let p2 =
        UOp::sink(vec![UOp::native_const(2.0f32)]).call(SmallVec::from_vec(vec![in_b.clone()]), CallInfo::default());

    let after = passthrough.after(SmallVec::from_vec(vec![p1_end, p2.clone()]));
    let consumer =
        UOp::sink(vec![UOp::native_const(3.0f32)]).call(SmallVec::from_vec(vec![after]), CallInfo::default());

    let transformed = UOp::sink(vec![p1.clone(), p2.clone(), consumer.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(in_a.id, cpu_buffer(4));
    input_buffers.insert(in_b.id, cpu_buffer(4));
    input_buffers.insert(passthrough.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let consumer_item = result.items.iter().find(|it| it.kernel.id == consumer.id).expect("consumer item");

    let mut expected = vec![p1.id, p2.id];
    expected.sort_unstable();
    assert_eq!(consumer_item.dependencies, expected);
}

/// A hand-lowered kernel body (a marked SINK, e.g. a hand-ranged custom kernel or
/// rangeify's own `reduce_to_acc`) legitimately contains intra-kernel REG
/// accumulator AFTERs whose deps are `END(STORE)` and `RANGE` — shapes the
/// inter-kernel callable-dependency grammar (`collect_callable_dep_ids`) does
/// NOT accept. The scheduler must walk dependency AFTERs with
/// `toposort_call_aware(false)`, treating the opaque CALL body as a single unit
/// so its internal AFTERs are never validated as callable deps. This guards
/// that boundary: `create_schedule` must succeed on such a body.
#[test]
fn test_create_schedule_opaque_body_intra_kernel_after_end_store() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 1, DType::Float32);

    // Intra-kernel REG accumulator with the canonical `reduce_to_acc` edges.
    let acc = UOp::buffer(0, 1, DType::Float32, svod_ir::AddrSpace::Reg, None);
    let zero = UOp::index_const(0);
    let idx = |buf: Arc<UOp>| UOp::index().buffer(buf).indices(vec![zero.clone()]).call().expect("index");
    let load = |buf: Arc<UOp>| UOp::load().index(idx(buf)).call();

    let init = idx(acc.clone()).store(UOp::native_const(0.0f32)); // bare STORE
    let range = UOp::range_const(4, 0);
    // After([bare STORE, RANGE]) — read inside the reduce loop.
    let acc_loop = load(acc.after(SmallVec::from_vec(vec![init, range.clone()])));
    // END(STORE) closes the reduce loop.
    let store_end = idx(acc.clone()).store(acc_loop).end(SmallVec::from_vec(vec![range]));
    // After([END(STORE)]) — post-loop read threaded into the out-store chain.
    let result = load(acc.after(SmallVec::from_vec(vec![store_end])));
    let out_store = idx(buffer_uop.clone()).store(result);

    // Marked SINK → opaque CALL body.
    let body =
        UOp::sink_with_info(vec![out_store], svod_ir::KernelInfo { opts_to_apply: Some(vec![]), ..Default::default() });
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone()]), CallInfo::default());
    let transformed = UOp::sink(vec![call.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(1));

    // Must not error with "AFTER dependency must be CALL/END(CALL)/STORE/AFTER".
    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    assert_eq!(result.items.len(), 1);
    assert_eq!(result.items[0].kernel.id, call.id);
}

#[test]
fn test_create_schedule_accepts_bind_callable_source() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let bind = UOp::define_var("N".to_string(), 0, 4).bind(UOp::range_const(4, 0));
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone(), bind]), CallInfo::default());
    let transformed = UOp::sink(vec![call]);

    let mut input_buffers = InputBuffers::new();
    let input = cpu_buffer(4);
    input_buffers.insert(buffer_uop.id, input.clone());

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let item = &result.items[0];

    assert_eq!(item.buffer_uop_ids, vec![buffer_uop.id]);
    assert_eq!(item.buffers.len(), 1);
    assert_eq!(item.buffers[0].id(), input.id());
}

#[test]
fn test_create_schedule_collects_nested_after_dependencies() {
    let input = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let producer =
        UOp::sink(vec![UOp::native_const(1.0f32)]).call(SmallVec::from_vec(vec![input.clone()]), CallInfo::default());
    let nested_after = passthrough.after(SmallVec::from_vec(vec![producer.clone()]));
    let after = passthrough.after(SmallVec::from_vec(vec![nested_after]));
    let consumer =
        UOp::sink(vec![UOp::native_const(2.0f32)]).call(SmallVec::from_vec(vec![after]), CallInfo::default());

    let transformed = UOp::sink(vec![producer.clone(), consumer.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(input.id, cpu_buffer(4));
    input_buffers.insert(passthrough.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let consumer_item = result.items.iter().find(|it| it.kernel.id == consumer.id).expect("consumer item");

    assert_eq!(consumer_item.dependencies, vec![producer.id]);
}

#[test]
fn test_create_schedule_rejects_non_callable_after_dependency() {
    let passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let invalid_dep = UOp::native_const(1.0f32);
    let after = passthrough.after(SmallVec::from_vec(vec![invalid_dep]));
    let consumer =
        UOp::sink(vec![UOp::native_const(2.0f32)]).call(SmallVec::from_vec(vec![after]), CallInfo::default());
    let transformed = UOp::sink(vec![consumer]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(passthrough.id, cpu_buffer(4));

    let err = match create_schedule(transformed, &input_buffers, &HashMap::new()) {
        Ok(_) => panic!("invalid AFTER dep should fail"),
        Err(err) => err,
    };
    match err.kind() {
        crate::ErrorKind::IrConstruction { details } => {
            assert!(details.contains("AFTER dependency must be CALL/END(CALL)/STORE/AFTER"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_create_schedule_accepts_store_in_after_dependencies() {
    let input = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let producer =
        UOp::sink(vec![UOp::native_const(1.0f32)]).call(SmallVec::from_vec(vec![input.clone()]), CallInfo::default());

    let store_idx =
        UOp::index().buffer(passthrough.clone()).indices(vec![UOp::index_const(0)]).call().expect("store index");
    let side_store = store_idx.store(UOp::native_const(0.0f32));

    let after = passthrough.after(SmallVec::from_vec(vec![producer.clone(), side_store]));
    let consumer =
        UOp::sink(vec![UOp::native_const(2.0f32)]).call(SmallVec::from_vec(vec![after]), CallInfo::default());

    let transformed = UOp::sink(vec![producer.clone(), consumer.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(input.id, cpu_buffer(4));
    input_buffers.insert(passthrough.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let consumer_item = result.items.iter().find(|it| it.kernel.id == consumer.id).expect("consumer item");

    assert_eq!(consumer_item.dependencies, vec![producer.id]);
}

#[test]
fn test_create_schedule_avoids_false_dep_from_shared_passthrough_identity() {
    let in_a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let in_b = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let p1 =
        UOp::sink(vec![UOp::native_const(1.0f32)]).call(SmallVec::from_vec(vec![in_a.clone()]), CallInfo::default());
    let p2 =
        UOp::sink(vec![UOp::native_const(2.0f32)]).call(SmallVec::from_vec(vec![in_b.clone()]), CallInfo::default());

    let c1 = UOp::sink(vec![UOp::native_const(3.0f32)])
        .call(SmallVec::from_vec(vec![passthrough.after(SmallVec::from_vec(vec![p1.clone()]))]), CallInfo::default());
    let c2 = UOp::sink(vec![UOp::native_const(4.0f32)])
        .call(SmallVec::from_vec(vec![passthrough.after(SmallVec::from_vec(vec![p2.clone()]))]), CallInfo::default());

    let transformed = UOp::sink(vec![p1.clone(), p2.clone(), c1.clone(), c2.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(in_a.id, cpu_buffer(4));
    input_buffers.insert(in_b.id, cpu_buffer(4));
    input_buffers.insert(passthrough.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let c1_item = result.items.iter().find(|it| it.kernel.id == c1.id).expect("c1 item");
    let c2_item = result.items.iter().find(|it| it.kernel.id == c2.id).expect("c2 item");

    assert_eq!(c1_item.dependencies, vec![p1.id]);
    assert_eq!(c2_item.dependencies, vec![p2.id]);
}

#[test]
fn test_create_schedule_preserves_ordering_only_dep_for_void_custom_call() {
    let input = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let producer_passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let ordering_passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let producer =
        UOp::sink(vec![UOp::native_const(1.0f32)]).call(SmallVec::from_vec(vec![input.clone()]), CallInfo::default());
    let producer_after = producer_passthrough.after(SmallVec::from_vec(vec![producer.clone()]));

    let custom = UOp::custom_function(svod_ir::CustomFunctionKind::EncDec, SmallVec::new())
        .call(SmallVec::new(), CallInfo::default());

    let ordering_after = ordering_passthrough.after(SmallVec::from_vec(vec![custom.clone(), producer_after]));
    let transformed = UOp::sink(vec![producer.clone(), custom.clone(), ordering_after]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(input.id, cpu_buffer(4));
    input_buffers.insert(producer_passthrough.id, cpu_buffer(4));
    input_buffers.insert(ordering_passthrough.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let custom_item = result.items.iter().find(|it| it.kernel.id == custom.id).expect("custom item");

    assert_eq!(custom_item.dependencies, vec![producer.id]);
}

#[test]
fn test_create_schedule_nested_after_mstack_mselect_dependencies_consistent() {
    let in_a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let in_b = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let out_a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let out_b = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let p1 =
        UOp::sink(vec![UOp::native_const(1.0f32)]).call(SmallVec::from_vec(vec![in_a.clone()]), CallInfo::default());
    let p2 =
        UOp::sink(vec![UOp::native_const(2.0f32)]).call(SmallVec::from_vec(vec![in_b.clone()]), CallInfo::default());

    let a1 = out_a.after(SmallVec::from_vec(vec![p1.clone()]));
    let a2 = out_b.after(SmallVec::from_vec(vec![p2.clone()]));
    let stacked = UOp::mstack(SmallVec::from_vec(vec![a1, a2.clone()]));
    let selected = stacked.mselect(0);

    let consumer =
        UOp::sink(vec![UOp::native_const(3.0f32)]).call(SmallVec::from_vec(vec![selected]), CallInfo::default());
    let transformed = UOp::sink(vec![p1.clone(), p2.clone(), consumer.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(in_a.id, cpu_buffer(4));
    input_buffers.insert(in_b.id, cpu_buffer(4));
    input_buffers.insert(out_a.id, cpu_buffer(4));
    input_buffers.insert(out_b.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let consumer_item = result.items.iter().find(|it| it.kernel.id == consumer.id).expect("consumer item");

    let mut expected = vec![p1.id, p2.id];
    expected.sort_unstable();
    assert_eq!(consumer_item.dependencies, expected);
}

#[test]
fn test_create_schedule_adds_conservative_instance_dependencies_for_expanded_producer() {
    let lanes = lane_buffers(2);
    let producer = call_over(UOp::native_const(1.0f32), vec![mstack_of(&lanes)]);
    let after_producer =
        lanes.iter().map(|lane| lane.after(SmallVec::from_vec(vec![producer.clone()]))).collect::<Vec<_>>();
    let consumer = call_over(UOp::native_const(2.0f32), vec![mstack_of(&after_producer)]);
    let transformed = UOp::sink(vec![producer.clone(), consumer.clone()]);

    let result = create_schedule(transformed, &lane_inputs(&[&lanes[0], &lanes[1]]), &HashMap::new()).unwrap();
    assert_eq!(result.items.len(), 4);
    assert!(result.items[..2].iter().all(|item| item.kernel.id == producer.id));
    for item in &result.items[2..] {
        assert_eq!(item.kernel.id, consumer.id);
        assert_eq!(item.dependencies, vec![producer.id]);
        assert_eq!(item.instance_dependencies, vec![0, 1]);
    }
}

#[test]
fn test_create_schedule_treats_unended_bound_range_as_runtime_bind() {
    // After the schedule-level Range/End refactor, schedule-level wrappers are
    // identified structurally: a Range is a loop only if it appears in an
    // `END(Call)`. A Bind whose value is a Range with no paired END is a
    // runtime variable bind (e.g. JIT variable), not a schedule loop — this
    // — the schedule has no public Range concept after eager unroll.
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let outer_range = UOp::range_const(3, 0);
    let bind = UOp::define_var("outer_i".to_string(), 0, 2).bind(outer_range);
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone(), bind]), CallInfo::default());
    let transformed = UOp::sink(vec![call]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("schedule should succeed");
    // Single CALL with no enclosing schedule loop → exactly one ScheduleItem.
    assert_eq!(result.items.len(), 1);
    // The Range-bound variable is a runtime bind, so no fixedvars are emitted.
    assert!(result.items[0].fixedvars.is_empty());
}

#[test]
fn test_create_schedule_rejects_non_concrete_outer_range_bounds() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let malformed_outer_range = UOp::new(
        Op::Range(ops::Range {
            end: UOp::native_const(3.0f32),
            axis_id: AxisId::Renumbered(0),
            axis_type: AxisType::Loop,
            deps: SmallVec::new(),
        }),
        DType::Float32,
    );
    let bind = UOp::define_var("outer_i".to_string(), 0, 2).bind(malformed_outer_range.clone());
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone(), bind]), CallInfo::default());
    let end_call = call.end(SmallVec::from_vec(vec![malformed_outer_range]));
    let transformed = UOp::sink(vec![call, end_call]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let err = match create_schedule(transformed, &input_buffers, &HashMap::new()) {
        Ok(_) => panic!("non-concrete OUTER RANGE bounds should fail"),
        Err(err) => err,
    };
    assert_ir_construction_error_contains(err, "schedule range vmin must be concrete integer");
}

#[test]
fn test_instantiate_schedule_rejects_invocation_with_unknown_kernel_id() {
    // After the schedule-level Range/End refactor, validation that previously
    // ran at instantiation now runs at PreSchedule construction. The only
    // remaining error path in `instantiate_schedule` is an `invocation` whose
    // `kernel_id` doesn't match any item template — exercise it here with a
    // fabricated `PreSchedule`.
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone()]), CallInfo::default());

    let pre_schedule = PreSchedule {
        items: vec![PreScheduleItem {
            kernel: call,
            ast: body,
            sources: vec![buffer_uop.clone()],
            dependencies: vec![],
            bound_ranges: vec![],
        }],
        invocations: vec![KernelInvocation { kernel_id: u64::MAX - 1, fixedvars: HashMap::new() }],
        output_buffer_uops: vec![buffer_uop.clone()],
    };

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let err = match instantiate_schedule(&pre_schedule, &input_buffers, &HashMap::new(), false) {
        Ok(_) => panic!("invocation with unknown kernel id should fail"),
        Err(err) => err,
    };
    assert_ir_construction_error_contains(err, "invocation references unknown kernel id");
}

#[test]
fn reach_once_visits_shared_producers_a_single_time() {
    let shared = UOp::native_const(1.0f32);
    let roots = [
        UOp::sink(vec![shared.clone(), UOp::native_const(2.0f32)]),
        UOp::sink(vec![shared.clone(), UOp::native_const(3.0f32)]),
    ];

    let mut reach = crate::schedule::ReachOnce::new();
    let mut visited = Vec::new();
    for root in &roots {
        reach
            .walk(root, |node| {
                visited.push(node.id);
                Ok(())
            })
            .unwrap();
    }

    assert_eq!(visited.len(), 5, "{visited:?}");
    assert_eq!(visited.iter().filter(|&&id| id == shared.id).count(), 1);
    let expected: HashSet<u64> = roots.iter().flat_map(|root| root.toposort()).map(|node| node.id).collect();
    assert_eq!(visited.into_iter().collect::<HashSet<_>>(), expected);
}
