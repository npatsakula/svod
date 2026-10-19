use super::*;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use svod_device::allocator::{Allocator, BufferSpec, CpuAllocator, RawBuffer};
use svod_device::device::Program;
use svod_device::device::{CopyEndpoint, NativeReplayDecline, NativeReplayOutcome, PlanCall, PlanContext};
use svod_device::hcq::QueueKind;
use svod_dtype::DType;
use svod_ir::ops;
use svod_ir::{CustomFunctionKind, UOp};

// ── fixtures ───────────────────────────────────────────────────────────────

fn unit_launch_size() -> [Arc<UOp>; 3] {
    [UOp::index_const(1), UOp::index_const(1), UOp::index_const(1)]
}

fn cpu_buffer(dtype: DType, len: usize) -> Buffer {
    Buffer::new(svod_device::registry::cpu().expect("cpu allocator"), dtype, vec![len], Default::default())
}

fn f32_buffer(values: &[f32]) -> Buffer {
    let mut buffer = cpu_buffer(DType::Float32, values.len());
    buffer.copyin(&values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    buffer
}

fn read_f32(buffer: &Buffer) -> Vec<f32> {
    let mut bytes = vec![0; buffer.size()];
    buffer.copyout(&mut bytes).unwrap();
    bytes.as_chunks::<4>().0.iter().copied().map(f32::from_le_bytes).collect()
}

/// A CPU `CachedKernel` over `buffers` global arguments: slot 0 is the output,
/// the rest are inputs, launch dims are unit. Callers override the fields whose
/// exact value is what they are testing.
fn cached(program: Box<dyn Program>, buffers: usize) -> CachedKernel {
    CachedKernel {
        entry_point: program.name().to_string(),
        program,
        device: "CPU".into(),
        code: String::new(),
        var_names: Vec::new(),
        globals: (0..buffers).collect(),
        outs: vec![0],
        ins: (1..buffers).collect(),
        global_size: unit_launch_size(),
        local_size: Some(unit_launch_size()),
    }
}

/// A CPU `PreparedKernel` around `kernel`, writing whichever arguments the
/// kernel declares as outputs, with no vars and no dependencies.
fn prepared(id: u64, kernel: CachedKernel, buffer_indices: Vec<usize>) -> PreparedKernel {
    PreparedKernel {
        id,
        ast: UOp::sink(vec![]),
        output_indices: kernel.outs.clone(),
        kernel: Arc::new(kernel),
        device: DeviceSpec::Cpu,
        buffer_indices,
        input_indices: Vec::new(),
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
        origin: None,
        origins: Default::default(),
    }
}

fn copy_op(id: u64, buffer_indices: Vec<usize>, dependencies: Vec<u64>) -> PreparedOp {
    PreparedOp::BufferCopy(PreparedCopy { id, buffer_indices, dependencies, origin: None, origins: Default::default() })
}

/// Copies `buffers[1]` over `buffers[0]`, four f32 wide, counting its calls.
#[derive(Debug)]
struct Copy4F32Program {
    calls: Arc<AtomicUsize>,
}

impl Program for Copy4F32Program {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        unsafe { std::ptr::copy_nonoverlapping(buffers[1], buffers[0], 4 * std::mem::size_of::<f32>()) };
        Ok(())
    }

    fn name(&self) -> &str {
        "copy4f32"
    }
}

fn copy4f32(calls: &Arc<AtomicUsize>) -> Box<dyn Program> {
    Box::new(Copy4F32Program { calls: Arc::clone(calls) })
}

/// Fails every dispatch, so reaching it at all is what a test asserts about.
#[derive(Debug)]
struct RejectDispatchProgram {
    calls: Arc<AtomicUsize>,
}

impl Program for RejectDispatchProgram {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(svod_device::Error::Runtime { message: "semantic fallback reached".into() })
    }

    fn name(&self) -> &str {
        "reject_dispatch"
    }
}

// ── builder ────────────────────────────────────────────────────────────────

#[test]
fn empty_plan_has_no_kernels_buffers_or_outputs() {
    let plan = ExecutionPlanBuilder::new(DeviceSpec::Cpu).build().expect("build plan");

    assert!(plan.prepared_kernels().is_empty());
    assert!(plan.buffers.is_empty());
    assert_eq!(plan.device, DeviceSpec::Cpu);
    assert!(plan.output_buffer().is_none(), "empty plan must not expose an output buffer");
    assert!(plan.output_buffer_at(0).is_none());
    assert!(plan.output_buffer_at(7).is_none(), "out-of-range output_buffer_at must be None");
}

#[test]
fn test_builder_map_buffer_alias() {
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let idx = builder.add_buffer(10, cpu_buffer(DType::Float32, 8));
    builder.map_buffer(11, idx);
    builder.set_output_buffer(idx);
    let plan = builder.build().expect("build plan");

    assert_eq!(plan.ast_to_buffer_map().get(&10), Some(&idx));
    assert_eq!(plan.ast_to_buffer_map().get(&11), Some(&idx));
    assert_eq!(plan.buffers().len(), 1);
}

fn plan_with_no_declared_outputs(builder: &mut ExecutionPlanBuilder) {
    builder.add_buffer(10, cpu_buffer(DType::Float32, 8));
}

fn copy_missing_its_source(builder: &mut ExecutionPlanBuilder) {
    let dst = builder.add_buffer(1, cpu_buffer(DType::Float32, 4));
    builder.add_op(copy_op(77, vec![dst], Vec::new()));
    builder.set_output_buffer(dst);
}

fn copy_depending_on_an_unknown_op_id(builder: &mut ExecutionPlanBuilder) {
    let dst = builder.add_buffer(400, cpu_buffer(DType::Float32, 4));
    let src = builder.add_buffer(401, cpu_buffer(DType::Float32, 4));
    builder.add_op(copy_op(10, vec![dst, src], vec![999]));
    builder.set_output_buffer(dst);
}

fn copies_that_depend_on_each_other(builder: &mut ExecutionPlanBuilder) {
    let a = builder.add_buffer(500, cpu_buffer(DType::Float32, 4));
    let b = builder.add_buffer(501, cpu_buffer(DType::Float32, 4));
    builder.add_op(copy_op(1, vec![a, b], vec![2]));
    builder.add_op(copy_op(2, vec![b, a], vec![1]));
    builder.set_output_buffer(a);
}

fn kernel_naming_an_absent_buffer(builder: &mut ExecutionPlanBuilder) {
    let buffer = cpu_buffer(DType::Float32, 4);
    buffer.ensure_allocated().expect("allocate");
    let dst = builder.add_buffer(860, buffer);
    builder.add_kernel(prepared(861, cached(copy4f32(&Arc::default()), 2), vec![dst, dst + 1]));
    builder.set_output_buffer(dst);
}

fn kernel_naming_an_absent_output_argument(builder: &mut ExecutionPlanBuilder) {
    let a = builder.add_buffer(700, cpu_buffer(DType::Float32, 4));
    let b = builder.add_buffer(701, cpu_buffer(DType::Float32, 4));
    let mut kernel = cached(copy4f32(&Arc::default()), 2);
    kernel.outs = vec![2];
    builder.add_kernel(prepared(77, kernel, vec![a, b]));
    builder.set_output_buffer(a);
}

/// Structural faults are all caught while building, before any dispatch.
#[test_case::test_case(plan_with_no_declared_outputs, "output buffers must be set explicitly"; "outputs never set")]
#[test_case::test_case(copy_missing_its_source, "requires two buffer indices"; "copy with one endpoint")]
#[test_case::test_case(copy_depending_on_an_unknown_op_id, "unknown op id"; "dependency on an unknown op")]
#[test_case::test_case(copies_that_depend_on_each_other, "cycle detected"; "dependency cycle")]
#[test_case::test_case(kernel_naming_an_absent_buffer, "buffer index out of range"; "kernel buffer index out of range")]
#[test_case::test_case(kernel_naming_an_absent_output_argument, "output index out of range"; "kernel output index out of range")]
fn build_rejects_malformed_plans(populate: fn(&mut ExecutionPlanBuilder), reason: &str) {
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    populate(&mut builder);

    match builder.build().expect_err("malformed plan must not build") {
        crate::error::Error::Execution { reason: actual } => {
            assert!(actual.contains(reason), "unexpected error: {actual}")
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

fn copy_reading_an_absent_buffer(builder: &mut ExecutionPlanBuilder) -> usize {
    let dst = builder.add_buffer(600, cpu_buffer(DType::Float32, 4));
    builder.add_op(copy_op(55, vec![dst, dst + 1], Vec::new()));
    dst
}

fn custom_function_over_an_absent_buffer(builder: &mut ExecutionPlanBuilder) -> usize {
    let dst = builder.add_buffer(880, cpu_buffer(DType::Float32, 4));
    builder.add_op(PreparedOp::CustomFunction(PreparedCustomFunction {
        id: 881,
        kind: CustomFunctionKind::EncDec,
        attrs: smallvec::smallvec![],
        buffer_indices: vec![dst, dst + 1],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        runtime_vars: Vec::new(),
        origin: None,
        origins: Default::default(),
    }));
    dst
}

/// Buffer indices on non-kernel ops are only reachable at dispatch, so those
/// faults surface from `execute` rather than from `build`.
#[test_case::test_case(copy_reading_an_absent_buffer, "out of range"; "copy source out of range")]
#[test_case::test_case(custom_function_over_an_absent_buffer, "buffer index out of range"; "custom function argument out of range")]
fn execute_rejects_out_of_range_buffer_indices(populate: fn(&mut ExecutionPlanBuilder) -> usize, reason: &str) {
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let output = populate(&mut builder);
    builder.set_output_buffer(output);
    let plan = builder.build().expect("build plan");

    match plan.execute().expect_err("out-of-range buffer index must fail") {
        crate::error::Error::Execution { reason: actual } => {
            assert!(actual.contains(reason), "unexpected error: {actual}")
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

// ── execution ──────────────────────────────────────────────────────────────

#[test]
fn test_copy_output_region_to_buffer() {
    let mut output = cpu_buffer(DType::UInt8, 8);
    let mut destination = cpu_buffer(DType::UInt8, 8);
    output.copyin(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
    destination.copyin(&[9; 8]).unwrap();

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let output_idx = builder.add_buffer(1, output);
    let destination_idx = builder.add_buffer(2, destination);
    builder.set_output_buffer(output_idx);
    let mut plan = builder.build().unwrap();

    plan.copy_output_region_to_buffer(0, destination_idx, 2, 3, 3).unwrap();
    let mut actual = [0; 8];
    plan.buffers()[destination_idx].copyout(&mut actual).unwrap();
    assert_eq!(actual, [9, 9, 3, 4, 5, 9, 9, 9]);

    assert!(plan.copy_output_region_to_buffer(1, destination_idx, 0, 0, 1).is_err(), "no second output");
    assert!(plan.copy_output_region_to_buffer(0, 99, 0, 0, 1).is_err(), "unknown destination");
    assert!(plan.copy_output_region_to_buffer(0, output_idx, 0, 0, 1).is_err(), "self-copy");
}

#[test]
fn test_execute_buffer_copy_op() {
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(1, cpu_buffer(DType::Float32, 4));
    let src_idx = builder.add_buffer(2, f32_buffer(&[1.0, 2.0, 3.0, 4.0]));
    builder.add_op(copy_op(99, vec![dst_idx, src_idx], Vec::new()));
    builder.set_output_buffer(dst_idx);

    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute copy op");

    assert_eq!(read_f32(plan.output_buffer().expect("plan has output")), vec![1.0, 2.0, 3.0, 4.0]);
}

/// `profile` stamps the requested rollup depth onto the profile it produces —
/// including through the `iters` min-merge — so `render_table()` and the bench
/// harness that calls it cut at the depth the options (and so `SVOD_ORIGIN_DEPTH`)
/// asked for, without the caller repeating it.
#[test]
fn test_profile_carries_the_requested_origin_depth() {
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(1, cpu_buffer(DType::Float32, 4));
    let src_idx = builder.add_buffer(2, f32_buffer(&[1.0, 2.0, 3.0, 4.0]));
    builder.add_op(copy_op(99, vec![dst_idx, src_idx], Vec::new()));
    builder.set_output_buffer(dst_idx);
    let plan = builder.build().expect("build plan");

    let opts = ProfileOptions { iters: 3, origin_depth: Some(2), ..Default::default() };
    assert_eq!(plan.profile(&opts).expect("profile").origin_depth, Some(2));
    assert_eq!(plan.profile(&ProfileOptions::default()).expect("profile").origin_depth, None, "the leaf by default");
}

/// A `CustomFunction` whose runtime is unimplemented surfaces its typed
/// `Unsupported`, and — because the epoch was already reserved — the plan is
/// poisoned rather than left retryable.
#[test]
fn test_execute_custom_function_op_returns_unsupported() {
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(201, cpu_buffer(DType::Float32, 4));
    let src_idx = builder.add_buffer(202, cpu_buffer(DType::Float32, 4));
    builder.add_op(PreparedOp::CustomFunction(PreparedCustomFunction {
        id: 200,
        kind: CustomFunctionKind::EncDec,
        attrs: smallvec::smallvec![svod_ir::UOp::index_const(3)],
        buffer_indices: vec![dst_idx, src_idx],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        runtime_vars: Vec::new(),
        origin: None,
        origins: Default::default(),
    }));
    builder.set_output_buffer(dst_idx);

    let plan = builder.build().expect("build plan");
    match plan.execute().expect_err("EncDec runtime should be explicit unsupported") {
        crate::error::Error::Unsupported { kind, reason } => {
            assert_eq!(kind, "EncDec");
            assert!(reason.contains("attrs=1"), "unexpected reason: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
    assert!(
        matches!(plan.execute(), Err(crate::error::Error::PlanPoisoned { .. })),
        "a callback failure after epoch reservation must reject immediate retry"
    );
}

#[test]
fn test_execution_plan_runs_host_allreduce_numerically() {
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let output = builder.add_buffer(301, f32_buffer(&[0.0, 0.0]));
    let shard0 = builder.add_buffer(302, f32_buffer(&[4.0, 7.0]));
    let shard1 = builder.add_buffer(303, f32_buffer(&[11.0, 7.0]));
    builder.add_op(PreparedOp::CustomFunction(PreparedCustomFunction {
        id: 300,
        kind: CustomFunctionKind::AllReduce { reduce_op: svod_ir::ReduceOp::Add },
        attrs: smallvec::smallvec![],
        buffer_indices: vec![output, shard0, shard1],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        runtime_vars: Vec::new(),
        origin: None,
        origins: Default::default(),
    }));
    builder.set_output_buffer(output);
    let plan = builder.build().unwrap();

    plan.execute().unwrap();
    assert_eq!(read_f32(plan.output_buffer().unwrap()), vec![15.0, 14.0]);
}

/// A kernel feeding a copy that lands in a *view* of the destination buffer:
/// the kernel runs once and the view sees the shifted window.
#[test]
fn test_execute_mixed_ops_compiled_copy_view_in_order() {
    let copy_dst = f32_buffer(&[0.0; 4]);
    let view = copy_dst.view(std::mem::size_of::<f32>(), 3 * std::mem::size_of::<f32>()).expect("output view");

    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let src_idx = builder.add_buffer(10, f32_buffer(&[1.0, 2.0, 3.0, 4.0]));
    let mid_idx = builder.add_buffer(11, f32_buffer(&[0.0; 4]));
    let copy_idx = builder.add_buffer(12, copy_dst);
    let out_idx = builder.add_buffer(13, view);

    builder.add_kernel(prepared(1, cached(copy4f32(&calls), 2), vec![mid_idx, src_idx]));
    builder.add_op(copy_op(2, vec![copy_idx, mid_idx], vec![1]));
    builder.set_output_buffer(out_idx);

    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute mixed op plan");

    assert_eq!(calls.load(Ordering::SeqCst), 1, "compiled op should run exactly once");
    assert_eq!(read_f32(plan.output_buffer().expect("plan has output")), vec![2.0, 3.0, 4.0]);
}

/// The kernel is inserted first but depends on the copy: mixed-op execution
/// must honour the declared edges, not the insertion order.
#[test]
fn test_execute_mixed_ops_respects_dependencies_not_insertion_order() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let src_idx = builder.add_buffer(300, f32_buffer(&[9.0, 8.0, 7.0, 6.0]));
    let mid_idx = builder.add_buffer(301, f32_buffer(&[0.0; 4]));
    let out_idx = builder.add_buffer(302, f32_buffer(&[0.0; 4]));

    let mut kernel = prepared(3, cached(copy4f32(&calls), 2), vec![out_idx, mid_idx]);
    kernel.dependencies = vec![2];
    builder.add_kernel(kernel);
    builder.add_op(copy_op(2, vec![mid_idx, src_idx], Vec::new()));
    builder.set_output_buffer(out_idx);

    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute dependency-ordered mixed ops");

    assert_eq!(calls.load(Ordering::SeqCst), 1, "compiled op should run exactly once");
    assert_eq!(read_f32(plan.output_buffer().expect("plan has output")), vec![9.0, 8.0, 7.0, 6.0]);
}

/// Expanded schedules repeat op ids for per-iteration items; a dependency on a
/// repeated id still resolves to the instance that precedes it.
#[test]
fn test_execute_mixed_ops_allows_duplicate_ids_in_expanded_schedule_order() {
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let src_idx = builder.add_buffer(800, f32_buffer(&[3.0, 1.0, 4.0, 1.0]));
    let mid_idx = builder.add_buffer(801, cpu_buffer(DType::Float32, 4));
    let out_idx = builder.add_buffer(802, cpu_buffer(DType::Float32, 4));
    builder.add_op(copy_op(42, vec![mid_idx, src_idx], Vec::new()));
    builder.add_op(copy_op(42, vec![out_idx, mid_idx], vec![42]));
    builder.set_output_buffer(out_idx);

    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute duplicate-id schedule");

    assert_eq!(read_f32(plan.output_buffer().expect("plan has output")), vec![3.0, 1.0, 4.0, 1.0]);
}

/// Records the order kernels dispatch in.
#[derive(Debug)]
struct OrderRecorderProgram {
    id: u64,
    sink: Arc<parking_lot::Mutex<Vec<u64>>>,
}

impl Program for OrderRecorderProgram {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        self.sink.lock().push(self.id);
        Ok(())
    }

    fn name(&self) -> &str {
        "order_recorder"
    }
}

/// `execute()` walks `op_levels` level by level, not a flat topological order.
/// Regression guard for commit fcbb725: QR decomposition and other iterative
/// CPU kernels are sensitive to within-level ordering, so a refactor back to
/// flat `op_order` would silently regress them.
///
/// Deps `A → C`, `B → D` give levels `[[A,B], [C,D]]`. Any topological order
/// respects A<C and B<D; only a level walk puts *both* of A,B before *both* of
/// C,D.
#[test]
fn test_execute_walks_op_levels_in_level_order() {
    let sink = Arc::new(parking_lot::Mutex::new(Vec::<u64>::new()));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let out = cpu_buffer(DType::Float32, 1);
    out.ensure_allocated().expect("out alloc");
    let out_idx = builder.add_buffer(900, out);
    builder.set_output_buffer(out_idx);

    for (id, deps) in [(1, vec![]), (2, vec![]), (3, vec![1]), (4, vec![2])] {
        let program = Box::new(OrderRecorderProgram { id, sink: Arc::clone(&sink) });
        let mut kernel = prepared(id, cached(program, 1), vec![out_idx]);
        kernel.dependencies = deps;
        builder.add_op(PreparedOp::CompiledProgram(kernel));
    }
    builder.build().expect("build plan").execute().expect("execute");

    let order = sink.lock().clone();
    assert_eq!(order.len(), 4, "expected 4 ops to run, got {order:?}");
    let pos = |id: u64| order.iter().position(|&x| x == id).expect("id not recorded");
    assert!(
        pos(1).max(pos(2)) < pos(3).min(pos(4)),
        "level-1 op ran before a level-0 op (order={order:?}); execute() must walk op_levels"
    );
}

// ── variable rebinding ─────────────────────────────────────────────────────

/// Records the launch dims and first scalar value each dispatch saw.
#[derive(Debug, Clone, Default)]
struct LaunchRecorder {
    calls: Arc<AtomicUsize>,
    global_x: Arc<AtomicUsize>,
    first_val: Arc<AtomicUsize>,
}

impl Program for LaunchRecorder {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.global_x.store(global_size.map(|size| size[0]).unwrap_or(0), Ordering::SeqCst);
        self.first_val.store(vals.first().copied().unwrap_or(0) as usize, Ordering::SeqCst);
        Ok(())
    }

    fn name(&self) -> &str {
        "record_launch"
    }
}

/// A one-buffer plan whose kernel takes `var` as its only scalar and `global_x`
/// as its X launch extent.
fn launch_recorder_plan(
    ast_id: u64,
    var: Arc<UOp>,
    global_x: Arc<UOp>,
    recorder: &LaunchRecorder,
    initial_val: i64,
) -> ExecutionPlan {
    let dst = cpu_buffer(DType::Float32, 4);
    dst.ensure_allocated().expect("allocate dst");
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(ast_id, dst);

    let mut kernel = cached(Box::new(recorder.clone()), 1);
    kernel.var_names = vec![match var.op() {
        svod_ir::Op::DefineVar(ops::DefineVar { name, .. }) => name.clone(),
        _ => "N".to_string(),
    }];
    kernel.global_size = [global_x.clone(), UOp::index_const(1), UOp::index_const(1)];
    let mut kernel = prepared(8500, kernel, vec![dst_idx]);
    kernel.ast = UOp::sink(vec![var, global_x]);
    kernel.vals = vec![initial_val];
    builder.add_kernel(kernel);
    builder.set_output_buffer(dst_idx);
    builder.build().expect("build plan")
}

/// `execute_with_vars` rebinds the variables the schedule left free and leaves
/// the ones it fixed alone; the profiled entry point behaves identically.
#[test_case::test_case(false; "execute_with_vars")]
#[test_case::test_case(true; "execute_with_vars_profiled")]
fn execute_with_vars_updates_only_non_fixed_variables(profiled: bool) {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(900, f32_buffer(&[0.0; 4]));
    let src_idx = builder.add_buffer(901, f32_buffer(&[0.0; 4]));

    for (id, vals, fixedvars) in [(900, 1, HashMap::new()), (910, 7, HashMap::from([("N".to_string(), 7)]))] {
        let mut kernel = cached(copy4f32(&calls), 2);
        kernel.var_names = vec!["N".to_string()];
        let mut kernel = prepared(id, kernel, vec![dst_idx, src_idx]);
        kernel.vals = vec![vals];
        kernel.fixedvars = fixedvars;
        builder.add_kernel(kernel);
    }
    builder.set_output_buffer(dst_idx);
    let mut plan = builder.build().expect("build plan");

    if profiled {
        let profiles = plan.execute_with_vars_profiled(&[("N", 42)]).expect("execute with vars profiled");
        assert_eq!(profiles.len(), 2, "profile should include every compiled kernel");
    } else {
        plan.execute_with_vars(&[("N", 42)]).expect("execute with vars");
    }

    let vals = plan.prepared_kernels().iter().map(|k| k.vals.as_slice()).collect::<Vec<_>>();
    assert_eq!(vals, [&[42][..], &[7][..]], "only the non-fixed variable may be overridden");
    assert_eq!(calls.load(Ordering::SeqCst), 2, "each kernel should execute exactly once");
}

/// Graph capture wants dispatch arguments that never change. A runtime
/// variable pinned in `fixedvars` (a schedule-loop counter) is one; a free
/// variable is not.
#[test_case::test_case(true; "pinned variable is static")]
#[test_case::test_case(false; "free variable is dynamic")]
fn graph_capture_gate_treats_pinned_variables_as_static(pinned: bool) {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(920, f32_buffer(&[0.0; 4]));
    let src_idx = builder.add_buffer(921, f32_buffer(&[0.0; 4]));

    let mut kernel = cached(copy4f32(&calls), 2);
    kernel.var_names = vec!["N".to_string()];
    let mut kernel = prepared(920, kernel, vec![dst_idx, src_idx]);
    kernel.vals = vec![7];
    kernel.runtime_vars = vec![RuntimeVar { name: "N".to_string(), min_val: 0, max_val: 8 }];
    kernel.fixedvars = if pinned { HashMap::from([("N".to_string(), 7)]) } else { HashMap::new() };
    assert_eq!(kernel.has_unbound_vars(), !pinned);

    builder.add_kernel(kernel);
    builder.set_output_buffer(dst_idx);
    let plan = builder.build().expect("build plan");
    assert_eq!(plan.all_static_kernels(), pinned);
}

/// A symbolic `global_size` resolves against the rebound variable at dispatch,
/// with no recompilation.
#[test_case::test_case(false; "execute_with_vars")]
#[test_case::test_case(true; "execute_with_vars_profiled")]
fn execute_with_vars_updates_symbolic_global_size(profiled: bool) {
    let recorder = LaunchRecorder::default();
    let n = UOp::define_var("N".to_string(), 1, 8);
    let mut plan = launch_recorder_plan(8500, n.clone(), n, &recorder, 1);

    if profiled {
        assert_eq!(plan.execute_with_vars_profiled(&[("N", 6)]).expect("execute profiled").len(), 1);
        assert_eq!(recorder.global_x.load(Ordering::SeqCst), 6);
    } else {
        plan.execute_with_vars(&[("N", 5)]).expect("execute with dynamic launch size");
        assert_eq!(recorder.global_x.load(Ordering::SeqCst), 5);
        assert_eq!(recorder.first_val.load(Ordering::SeqCst), 5);
    }
    assert_eq!(recorder.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn test_execute_with_vars_rejects_out_of_bounds_launch_var_before_dispatch() {
    let recorder = LaunchRecorder::default();
    let n = UOp::define_var("N".to_string(), 1, 4);
    let mut plan = launch_recorder_plan(8510, n.clone(), n, &recorder, 1);

    // The bound is enforced at launch-dim resolution (device side), surfaced as
    // `Exec` carrying the underlying `svod_device` error as its source.
    match plan.execute_with_vars(&[("N", 5)]).expect_err("out-of-bounds launch var should fail") {
        crate::error::Error::Exec { source, .. } => {
            assert!(source.to_string().contains("outside bounds"), "unexpected error: {source}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
    assert_eq!(recorder.calls.load(Ordering::SeqCst), 0, "kernel must not dispatch after a bounds failure");
}

#[test]
fn test_execute_with_vars_does_not_override_core_id_runtime_var() {
    let recorder = LaunchRecorder::default();
    let core_id = UOp::define_var("core_id".to_string(), 0, 3);
    let mut plan = launch_recorder_plan(8530, core_id, UOp::index_const(1), &recorder, 0);

    plan.execute_with_vars(&[("core_id", 2)]).expect("execute with ignored core_id override");

    assert_eq!(recorder.calls.load(Ordering::SeqCst), 1);
    assert_eq!(recorder.first_val.load(Ordering::SeqCst), 0, "core_id is a runtime var, not user-overridable");
}

// ── dependency levels ──────────────────────────────────────────────────────

#[test]
fn test_compute_execution_levels_duplicate_ids_is_deterministic() {
    let ops = vec![copy_op(42, vec![0, 1], vec![]), copy_op(9, vec![2, 3], vec![42]), copy_op(42, vec![4, 5], vec![9])];

    assert_eq!(compute_mixed_op_order(&ops).expect("dependency order"), vec![0, 1, 2]);
    assert_eq!(compute_execution_levels(&ops).expect("dependency levels"), vec![vec![0], vec![1], vec![2]]);
}

/// Instance dependencies address an op by index, so they can target one
/// specific instance of a repeated id — and must reject an unknown index.
#[test]
fn test_instance_dependencies_target_exact_duplicate_id_instance() {
    let ops = vec![
        copy_op(42, vec![0, 1], vec![]),
        copy_op(9, vec![2, 3], vec![]),
        copy_op(42, vec![4, 5], vec![9]),
        copy_op(77, vec![6, 7], vec![]),
    ];
    let levels = compute_execution_levels_with_instance_dependencies(&ops, &[vec![], vec![], vec![], vec![0]])
        .expect("dependency levels");
    assert_eq!(levels, vec![vec![0, 1], vec![2, 3]]);

    let err = compute_execution_levels_with_instance_dependencies(&ops[..1], &[vec![1]])
        .expect_err("unknown op-index dependency should fail");
    match err {
        crate::error::Error::Execution { reason } => assert!(reason.contains("unknown op index"), "{reason}"),
        other => panic!("unexpected error variant: {other:?}"),
    }
}

// ── HCQ lane linking ───────────────────────────────────────────────────────

fn hcq_access(storage: u64) -> BufferAccess {
    BufferAccess { storage: BufferId(storage), owner: DeviceSpec::Cpu, start: 0, end: 64 }
}

fn hcq_op(operation: usize, queue: QueueKind, reads: &[u64], writes: &[u64]) -> HcqPreparedOperation {
    HcqPreparedOperation {
        operation,
        device: DeviceSpec::Cpu,
        queue,
        reads: reads.iter().map(|&id| hcq_access(id)).collect(),
        writes: writes.iter().map(|&id| hcq_access(id)).collect(),
        is_copy: matches!(queue, QueueKind::Copy(_)),
    }
}

fn operation_submission(plan: &HcqLinkedPlan, operation: usize) -> &svod_device::hcq::LaneSubmission {
    plan.semantic
        .lanes()
        .iter()
        .find(|submission| submission.commands.iter().any(|command| command.operation == operation))
        .expect("operation submission")
}

/// Which queue timeline an operation waits on. Independent work and same-queue
/// work never wait (the queue is FIFO); every cross-queue RAW / WAR / WAW edge
/// waits on the timeline of the queue that produced the hazard.
#[test]
fn hcq_lane_waits_follow_cross_queue_hazards() {
    use QueueKind::{Compute, Copy};
    let scenarios: [(&str, Vec<HcqPreparedOperation>, usize, Option<QueueKind>); 6] = [
        (
            "disjoint buffers on different queues overlap",
            vec![hcq_op(0, Compute(0), &[1], &[2]), hcq_op(1, Copy(0), &[3], &[4])],
            1,
            None,
        ),
        (
            "same-queue RAW is FIFO-elided",
            vec![hcq_op(0, Compute(0), &[], &[1]), hcq_op(1, Compute(0), &[1], &[2])],
            1,
            None,
        ),
        (
            "cross-queue RAW waits for the writer",
            vec![hcq_op(0, Compute(0), &[], &[1]), hcq_op(1, Copy(0), &[1], &[2])],
            1,
            Some(Compute(0)),
        ),
        (
            "cross-queue WAR waits for the reader",
            vec![hcq_op(0, Compute(0), &[1], &[]), hcq_op(1, Copy(0), &[], &[1])],
            1,
            Some(Compute(0)),
        ),
        (
            "cross-queue WAW waits for the writer",
            vec![hcq_op(0, Copy(0), &[], &[1]), hcq_op(1, Compute(0), &[], &[1])],
            1,
            Some(Copy(0)),
        ),
        (
            "copy-to-compute waits on the copy timeline",
            vec![hcq_op(0, Compute(0), &[], &[1]), hcq_op(1, Copy(0), &[1], &[2]), hcq_op(2, Compute(0), &[2], &[3])],
            2,
            Some(Copy(0)),
        ),
    ];

    for (name, operations, operation, expected) in scenarios {
        let plan = HcqLinkedPlan::capture(operations).unwrap();
        let waits = &operation_submission(&plan, operation).waits;
        match expected {
            None => assert!(waits.is_empty(), "{name}: {waits:?}"),
            Some(queue) => assert_eq!(waits[0].lane.queue, queue, "{name}"),
        }
    }
}

/// A kernel argument the `ProgramSpec` never declared as an input still enters
/// the hazard read-set, so the RAW edge onto the producer survives.
#[test]
fn hazard_reads_cover_buffers_the_kernel_does_not_declare() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let shared_idx = builder.add_buffer(1, cpu_buffer(DType::Float32, 4));
    let out_idx = builder.add_buffer(2, cpu_buffer(DType::Float32, 4));
    let kernel = |id, buffer_indices: Vec<usize>| {
        let mut kernel = cached(Box::new(RejectDispatchProgram { calls: Arc::clone(&calls) }), buffer_indices.len());
        kernel.ins = Vec::new();
        prepared(id, kernel, buffer_indices)
    };
    builder.add_kernel(kernel(1_001, vec![shared_idx]));
    builder.add_op_with_instance_dependencies(
        PreparedOp::CompiledProgram(kernel(1_002, vec![out_idx, shared_idx])),
        vec![0],
    );
    builder.set_output_buffer(out_idx);
    let plan = builder.build().expect("build plan");

    let operations = plan.hcq_operations().expect("hcq operations");
    let shared = plan.buffers()[shared_idx].storage_id();
    assert!(
        operations[1].reads.iter().any(|access| access.storage == shared),
        "undeclared read must still enter the hazard read-set: {:?}",
        operations[1].reads
    );
    assert!(operations[1].reads.iter().all(|access| access.storage != plan.buffers()[out_idx].storage_id()));
}

/// Two plans built from the same shape get their own signal addresses, so
/// concurrent executions cannot observe each other's timeline.
#[test]
fn concurrent_execution_plans_keep_linked_context_timelines_isolated() {
    fn copy_plan(seed: u8) -> ExecutionPlan {
        let mut source = cpu_buffer(DType::UInt8, 4);
        source.copyin(&[seed; 4]).unwrap();
        let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
        let output_idx = builder.add_buffer(seed as u64 + 30_000, cpu_buffer(DType::UInt8, 4));
        let source_idx = builder.add_buffer(seed as u64 + 31_000, source);
        builder.add_op(copy_op(seed as u64 + 32_000, vec![output_idx, source_idx], vec![]));
        builder.set_output_buffer(output_idx);
        builder.build().unwrap()
    }

    let left = Arc::new(copy_plan(4));
    let right = Arc::new(copy_plan(7));
    let signal = |plan: &ExecutionPlan| plan.hcq_linked.get().unwrap().semantic.bindings()[0].point.signal_address;
    assert_ne!(signal(&left), signal(&right));

    std::thread::scope(|scope| {
        for plan in [Arc::clone(&left), Arc::clone(&right)] {
            scope.spawn(move || {
                for _ in 0..20 {
                    plan.execute().unwrap();
                }
            });
        }
    });
    let read = |plan: &ExecutionPlan| {
        let mut out = [0; 4];
        plan.output_buffer().unwrap().copyout(&mut out).unwrap();
        out
    };
    assert_eq!(read(&left), [4; 4]);
    assert_eq!(read(&right), [7; 4]);
}

// ── replay ─────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct FixedGraphTimestamps {
    drops: Arc<AtomicUsize>,
}

impl svod_device::DispatchTimestamps for FixedGraphTimestamps {
    fn timestamps_ns(&self) -> Option<(u64, u64)> {
        Some((100, 140))
    }
}

impl Drop for FixedGraphTimestamps {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct ProfileReplayGraph {
    replays: Arc<AtomicUsize>,
    timestamp_drops: Arc<AtomicUsize>,
}

impl svod_device::Graph for ProfileReplayGraph {
    fn replay(&self, _buffers: &[u64], _vals: &[i64]) -> svod_device::Result<()> {
        self.replays.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn replay_profiled(
        &self,
        _buffers: &[u64],
        _vals: &[i64],
    ) -> svod_device::Result<Option<Vec<Arc<dyn svod_device::DispatchTimestamps>>>> {
        self.replays.fetch_add(1, Ordering::SeqCst);
        Ok(Some(vec![Arc::new(FixedGraphTimestamps { drops: Arc::clone(&self.timestamp_drops) })]))
    }
}

/// Attribution rides the dispatch, not the shared program: both profile push
/// sites copy it off the `PreparedKernel`.
#[test]
fn profiles_carry_the_dispatch_origins() {
    use svod_ir::origin::{self, Origin, OriginFrame};

    let origin = origin::intern(Origin { parent: None, frame: OriginFrame::Label { text: "profiled".into() } });
    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(7, cpu_buffer(DType::Float32, 4));
    let src_idx = builder.add_buffer(8, f32_buffer(&[1.0, 2.0, 3.0, 4.0]));
    let mut kernel = prepared(7, cached(copy4f32(&calls), 2), vec![dst_idx, src_idx]);
    kernel.origin = Some(origin);
    kernel.origins = [origin].into_iter().collect();
    builder.add_kernel(kernel);
    builder.set_output_buffer(dst_idx);
    let plan = builder.build().unwrap();

    let profiles = plan.execute_profiled().unwrap();
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].origin, Some(origin));
    assert_eq!(profiles[0].origins.iter().copied().collect::<Vec<_>>(), [origin]);
}

/// With a graph backend attached, profiling reads the replay's own timestamps
/// instead of redispatching per call, and releases the timestamp handles once
/// they are collected.
#[test]
fn profiled_execution_uses_graph_replay_timestamps_when_available() {
    let calls = Arc::new(AtomicUsize::new(0));
    let replays = Arc::new(AtomicUsize::new(0));
    let timestamp_drops = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let buffer_idx = builder.add_buffer(42, cpu_buffer(DType::Float32, 4));
    let mut kernel = cached(copy4f32(&calls), 1);
    kernel.entry_point = "profile_graph".into();
    builder.add_kernel(prepared(42, kernel, vec![buffer_idx]));
    builder.set_output_buffer(buffer_idx);
    let plan = builder.build().unwrap();
    plan.graph
        .set(Some(Box::new(ProfileReplayGraph {
            replays: Arc::clone(&replays),
            timestamp_drops: Arc::clone(&timestamp_drops),
        })))
        .map_err(|_| ())
        .unwrap();

    let profiles = plan.execute_profiled().unwrap();
    assert_eq!(profiles.len(), 1);
    assert_eq!((profiles[0].gpu_start_ns, profiles[0].gpu_end_ns), (Some(100), Some(140)));
    assert_eq!(replays.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 0, "profiled graph must not redispatch per-call");
    assert_eq!(timestamp_drops.load(Ordering::SeqCst), 1, "finalizer must release handles after collection");
}

/// Records the buffer addresses and scalar each replay dispatch saw.
#[derive(Debug)]
struct ReplayRecorderProgram {
    calls: Arc<parking_lot::Mutex<Vec<(usize, usize, i64)>>>,
}

impl Program for ReplayRecorderProgram {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        self.calls.lock().push((buffers[0] as usize, buffers[1] as usize, vals[0]));
        unsafe { std::ptr::copy_nonoverlapping(buffers[1], buffers[0], 4) };
        Ok(())
    }

    fn name(&self) -> &str {
        "replay_recorder"
    }
}

/// Re-executing a plan repatches the variable values and the buffer addresses
/// (here a wholly replaced source buffer) without rebuilding the linked lanes.
#[test]
fn repeated_normal_execution_repatches_vars_buffers_and_mixed_copy_plan() {
    let mut source = cpu_buffer(DType::UInt8, 4);
    source.copyin(&[1, 2, 3, 4]).unwrap();
    let calls = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let source_idx = builder.add_buffer(20_001, source);
    let middle_idx = builder.add_buffer(20_002, cpu_buffer(DType::UInt8, 4));
    let output_idx = builder.add_buffer(20_003, cpu_buffer(DType::UInt8, 4));

    let mut kernel = cached(Box::new(ReplayRecorderProgram { calls: Arc::clone(&calls) }), 2);
    kernel.var_names = vec!["N".into()];
    let mut kernel = prepared(20_010, kernel, vec![middle_idx, source_idx]);
    kernel.vals = vec![1];
    builder.add_kernel(kernel);
    builder.add_op(copy_op(20_011, vec![output_idx, middle_idx], vec![20_010]));
    builder.set_output_buffer(output_idx);
    let mut plan = builder.build().unwrap();
    let static_lanes = plan.hcq_linked.get().unwrap().semantic.lanes().as_ptr();

    let read = |plan: &ExecutionPlan| {
        let mut out = [0; 4];
        plan.output_buffer().unwrap().copyout(&mut out).unwrap();
        out
    };
    plan.execute_with_vars(&[("N", 3)]).unwrap();
    assert_eq!(read(&plan), [1, 2, 3, 4]);

    let mut replacement = cpu_buffer(DType::UInt8, 4);
    replacement.copyin(&[9, 8, 7, 6]).unwrap();
    *plan.buffer_at_mut(source_idx).unwrap() = replacement;
    plan.execute_with_vars(&[("N", 7)]).unwrap();
    assert_eq!(read(&plan), [9, 8, 7, 6]);

    let calls = calls.lock();
    assert_eq!(calls.len(), 2);
    assert_eq!((calls[0].2, calls[1].2), (3, 7));
    assert_ne!(calls[0].1, calls[1].1, "replacement buffer address must be patched on replay");
    assert_eq!(plan.hcq_linked.get().unwrap().semantic.lanes().as_ptr(), static_lanes, "lanes must not be rebuilt");
}

/// A CPU allocator that reports an arbitrary owning device, and counts the
/// `device_spec` calls the native-replay validation walk makes.
#[derive(Debug)]
struct TaggedCpuAllocator(DeviceSpec, Arc<AtomicUsize>);

impl Allocator for TaggedCpuAllocator {
    fn _alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> svod_device::Result<RawBuffer> {
        CpuAllocator._alloc(size, options, zero)
    }

    fn name(&self) -> &str {
        "tagged-cpu"
    }

    fn _copyin(&self, dest: &RawBuffer, dest_off: usize, src: &[u8]) -> svod_device::Result<()> {
        CpuAllocator._copyin(dest, dest_off, src)
    }

    fn _copyout(&self, dest: &mut [u8], src: &RawBuffer, src_off: usize) -> svod_device::Result<()> {
        CpuAllocator._copyout(dest, src, src_off)
    }

    fn device_spec(&self) -> DeviceSpec {
        self.1.fetch_add(1, Ordering::Relaxed);
        self.0.clone()
    }
}

fn tagged_buffer(device: DeviceSpec) -> Buffer {
    tagged_buffer_counting(device, Arc::new(AtomicUsize::new(0)))
}

fn tagged_buffer_counting(device: DeviceSpec, spec_calls: Arc<AtomicUsize>) -> Buffer {
    Buffer::new(Arc::new(TaggedCpuAllocator(device, spec_calls)), DType::UInt8, vec![4], Default::default())
}

/// Dispatches only through `replay_linked_plan`; per-operation dispatch is a
/// test failure.
#[derive(Debug)]
struct NativeReplayProgram {
    replays: Arc<AtomicUsize>,
    fail: bool,
}

impl Program for NativeReplayProgram {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        panic!("native replay test must not use per-operation dispatch")
    }

    fn name(&self) -> &str {
        "native_replay_recorder"
    }

    fn new_exec_context(&self) -> svod_device::Result<Option<Box<dyn PlanContext>>> {
        Ok(Some(Box::new(NativeReplayContext { replays: Arc::clone(&self.replays), fail: self.fail })))
    }
}

#[derive(Debug)]
struct NativeReplayContext {
    replays: Arc<AtomicUsize>,
    fail: bool,
}

impl PlanContext for NativeReplayContext {
    unsafe fn dispatch(
        &self,
        _program: &dyn Program,
        _buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _profile: bool,
    ) -> svod_device::Result<Option<Arc<dyn svod_device::DispatchTimestamps>>> {
        panic!("native replay test must not dispatch individual kernels")
    }

    fn replay_linked_plan(
        &self,
        _plan: &svod_device::hcq::SemanticLinkedPlan,
        _calls: &[PlanCall<'_>],
    ) -> svod_device::Result<NativeReplayOutcome> {
        self.replays.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            return Err(svod_device::Error::Runtime { message: "native submit rejected".into() });
        }
        Ok(NativeReplayOutcome::Executed)
    }

    fn synchronize(&self) -> svod_device::Result<()> {
        Ok(())
    }
}

struct NativePlan {
    plan: ExecutionPlan,
    replays: Arc<AtomicUsize>,
    kernel: usize,
    dst: usize,
    src: usize,
}

/// A CPU-owned kernel (op 21_010) feeding a copy (op 21_011) whose source lives
/// on `source_device`.
fn native_copy_plan_with_source(
    source_device: DeviceSpec,
    fail_native: bool,
    spec_calls: Arc<AtomicUsize>,
) -> NativePlan {
    let owner = DeviceSpec::Cpu;
    let replays = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(owner.clone());
    let kernel_idx = builder.add_buffer(21_001, tagged_buffer_counting(owner.clone(), Arc::clone(&spec_calls)));
    let dst_idx = builder.add_buffer(21_002, tagged_buffer_counting(owner.clone(), Arc::clone(&spec_calls)));
    let src_idx = builder.add_buffer(21_003, tagged_buffer_counting(source_device, spec_calls));

    let mut kernel = cached(Box::new(NativeReplayProgram { replays: Arc::clone(&replays), fail: fail_native }), 1);
    kernel.device = "AMD:0".into();
    builder.add_kernel(prepared(21_010, kernel, vec![kernel_idx]));
    builder.add_op(copy_op(21_011, vec![dst_idx, src_idx], vec![21_010]));
    builder.set_output_buffer(dst_idx);
    NativePlan { plan: builder.build().unwrap(), replays, kernel: kernel_idx, dst: dst_idx, src: src_idx }
}

fn native_copy_plan() -> NativePlan {
    native_copy_plan_with_source(DeviceSpec::Cpu, false, Arc::new(AtomicUsize::new(0)))
}

/// A cross-device copy is staged through host memory, which the native replay
/// path cannot express, so it declines rather than submitting.
#[test]
fn native_replay_rejects_staged_semantic_copy() {
    let native = native_copy_plan_with_source(DeviceSpec::Amd { device_id: 0 }, false, Arc::new(AtomicUsize::new(0)));
    assert_eq!(
        native.plan.replay_native_linked_plan().unwrap(),
        NativeReplayOutcome::Declined(NativeReplayDecline::StagedCopy { operation: 1 })
    );
    assert_eq!(native.replays.load(Ordering::SeqCst), 0);
}

/// Every copy endpoint must still live on the context's device when the plan is
/// replayed: swapping either end for a foreign allocation declines before the
/// native context is reached.
#[test]
fn native_replay_requires_copy_endpoints_on_context_device() {
    let NativePlan { mut plan, replays, dst, src, .. } = native_copy_plan();
    assert_eq!(plan.replay_native_linked_plan().unwrap(), NativeReplayOutcome::Executed);
    assert_eq!(replays.load(Ordering::SeqCst), 1);

    let foreign = |endpoint, actual: DeviceSpec| {
        NativeReplayOutcome::Declined(NativeReplayDecline::ForeignCopyEndpoint {
            operation: 21_011,
            endpoint,
            expected: DeviceSpec::Cpu,
            actual,
        })
    };
    *plan.buffer_at_mut(dst).unwrap() = tagged_buffer(DeviceSpec::Amd { device_id: 1 });
    assert_eq!(
        plan.replay_native_linked_plan().unwrap(),
        foreign(CopyEndpoint::Destination, DeviceSpec::Amd { device_id: 1 })
    );

    *plan.buffer_at_mut(dst).unwrap() = tagged_buffer(DeviceSpec::Cpu);
    *plan.buffer_at_mut(src).unwrap() = tagged_buffer(DeviceSpec::Amd { device_id: 0 });
    assert_eq!(
        plan.replay_native_linked_plan().unwrap(),
        foreign(CopyEndpoint::Source, DeviceSpec::Amd { device_id: 0 })
    );
    assert_eq!(replays.load(Ordering::SeqCst), 1, "a foreign endpoint must not reach the native context");
}

/// The same rule for kernel arguments — including one whose device id happens
/// to match the program's own `AMD:0` string, which is not the context device.
#[test]
fn native_replay_requires_program_endpoints_on_context_device() {
    let NativePlan { mut plan, replays, kernel, .. } = native_copy_plan();
    let foreign = |actual: DeviceSpec| {
        NativeReplayOutcome::Declined(NativeReplayDecline::ForeignProgramEndpoint {
            operation: 21_010,
            argument: 0,
            expected: DeviceSpec::Cpu,
            actual,
        })
    };

    for device_id in [1, 0] {
        *plan.buffer_at_mut(kernel).unwrap() = tagged_buffer(DeviceSpec::Amd { device_id });
        assert_eq!(plan.replay_native_linked_plan().unwrap(), foreign(DeviceSpec::Amd { device_id }));
    }
    assert_eq!(replays.load(Ordering::SeqCst), 0, "a foreign PROGRAM endpoint must not reach the native context");
}

/// Graph replay applies the same endpoint rule, and a decline falls back to
/// semantic dispatch rather than reaching the graph backend.
#[test]
fn graph_replay_rejects_forged_amd_allocation_ownership() {
    let calls = Arc::new(AtomicUsize::new(0));
    let replays = Arc::new(AtomicUsize::new(0));
    let amd = DeviceSpec::Amd { device_id: 0 };
    let mut builder = ExecutionPlanBuilder::new(amd.clone());
    let buffer_idx = builder.add_buffer(20_001, tagged_buffer(amd.clone()));
    let mut kernel = cached(Box::new(RejectDispatchProgram { calls: Arc::clone(&calls) }), 1);
    kernel.entry_point = "graph_endpoint_guard".into();
    let mut kernel = prepared(20_010, kernel, vec![buffer_idx]);
    kernel.device = amd.clone();
    builder.add_kernel(kernel);
    builder.set_output_buffer(buffer_idx);
    let mut plan = builder.build().unwrap();
    plan.graph
        .set(Some(Box::new(ProfileReplayGraph {
            replays: Arc::clone(&replays),
            timestamp_drops: Arc::new(AtomicUsize::new(0)),
        })))
        .map_err(|_| ())
        .unwrap();

    *plan.buffer_at_mut(buffer_idx).unwrap() = tagged_buffer(amd);
    assert_eq!(
        plan.replay_native_linked_plan().unwrap(),
        NativeReplayOutcome::Declined(NativeReplayDecline::IncompatibleProgramAllocation {
            operation: 20_010,
            argument: 0,
            expected: DeviceSpec::Amd { device_id: 0 },
        })
    );
    let error = plan.execute().expect_err("forged AMD ownership must use semantic fallback");
    assert!(error.to_string().contains("semantic fallback reached"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(replays.load(Ordering::SeqCst), 0, "forged endpoint reached graph backend");
}

/// A cross-device copy that stages through host memory must take fresh host
/// storage each epoch, so a second execute sees the updated source.
#[test]
fn staged_copy_uses_fresh_host_storage_each_epoch() {
    use svod_device::hcq::CopyLeg;

    let mut source = tagged_buffer(DeviceSpec::Amd { device_id: 0 });
    source.copyin(&[1, 2, 3, 4]).unwrap();
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst = builder.add_buffer(20_101, tagged_buffer(DeviceSpec::Cpu));
    let src = builder.add_buffer(20_102, source);
    builder.add_op(copy_op(20_103, vec![dst, src], vec![]));
    builder.set_output_buffer(dst);
    let mut plan = builder.build().unwrap();
    let legs = plan
        .hcq_linked
        .get()
        .unwrap()
        .semantic
        .lanes()
        .iter()
        .map(|lane| lane.commands[0].copy_leg.unwrap())
        .collect::<Vec<_>>();
    assert_eq!(legs, [CopyLeg::ToHost, CopyLeg::FromHost]);

    let read = |plan: &ExecutionPlan| {
        let mut out = [0; 4];
        plan.output_buffer().unwrap().copyout(&mut out).unwrap();
        out
    };
    plan.execute().unwrap();
    assert_eq!(read(&plan), [1, 2, 3, 4]);

    plan.buffer_at_mut(src).unwrap().copyin(&[9, 8, 7, 6]).unwrap();
    plan.execute().unwrap();
    assert_eq!(read(&plan), [9, 8, 7, 6]);
}

#[test]
fn failed_native_replay_poisons_the_plan() {
    let NativePlan { plan, replays, .. } =
        native_copy_plan_with_source(DeviceSpec::Cpu, true, Arc::new(AtomicUsize::new(0)));

    let first = plan.execute().expect_err("failing native submit must surface");
    assert!(matches!(first, crate::error::Error::Exec { .. }), "{first:?}");
    assert_eq!(replays.load(Ordering::SeqCst), 1);

    let second = plan.execute().expect_err("a failed native submit must not stay retryable");
    assert!(matches!(second, crate::error::Error::PlanPoisoned { .. }), "{second:?}");
    assert_eq!(replays.load(Ordering::SeqCst), 1, "poisoned plan must not resubmit");
}

/// Native-replay endpoint validation walks the plan once per execute, with no
/// per-buffer device resolution and no growth across repeated executes.
#[test]
fn native_replay_validation_cost_is_flat_across_executes() {
    const ENDPOINTS: usize = 3; // one kernel argument + copy destination/source.
    let spec_calls = Arc::new(AtomicUsize::new(0));
    let NativePlan { plan, .. } = native_copy_plan_with_source(DeviceSpec::Cpu, false, Arc::clone(&spec_calls));

    plan.execute().expect("first execute");
    spec_calls.store(0, Ordering::SeqCst);
    // Measure a warm execute: the first one also mints the plan context.
    plan.execute().expect("warm execute");
    let per_execute = spec_calls.swap(0, Ordering::SeqCst);
    for _ in 0..100 {
        plan.execute().expect("repeat execute");
    }

    assert_eq!(spec_calls.load(Ordering::SeqCst), per_execute * 100, "per-execute validation cost must not grow");
    assert!(
        per_execute <= 4 * ENDPOINTS,
        "{per_execute} device_spec calls for {ENDPOINTS} endpoints: the walk was not merged"
    );
}

// ── replicate ──────────────────────────────────────────────────────────────

/// A `[out, in]` copy plan over `values`, plus its dispatch counter.
fn copy_plan(values: &[f32]) -> (ExecutionPlan, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let out = builder.add_buffer(30_001, cpu_buffer(DType::Float32, 4));
    let input = builder.add_buffer(30_002, f32_buffer(values));
    builder.add_kernel(prepared(30_003, cached(copy4f32(&calls), 2), vec![out, input]));
    builder.set_output_buffer(out);
    (builder.build().unwrap(), calls)
}

fn f32_bytes(values: [f32; 4]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

#[test]
fn replicate_forks_written_storage_and_shares_read_only_inputs() {
    let (plan, _) = copy_plan(&[1.0, 2.0, 3.0, 4.0]);
    plan.execute().unwrap();

    let replica = plan.replicate().unwrap();
    assert_ne!(plan.buffers()[0].storage_id(), replica.buffers()[0].storage_id(), "output storage must fork");
    assert_eq!(plan.buffers()[1].storage_id(), replica.buffers()[1].storage_id(), "read-only input stays shared");

    // Written storages fork bare: the replica derives its output on its own
    // execute, and neither plan's output storage sees the other's runs.
    replica.execute().unwrap();
    assert_eq!(read_f32(replica.output_buffer().unwrap()), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(read_f32(plan.output_buffer().unwrap()), [1.0, 2.0, 3.0, 4.0]);

    // The shared input is genuinely shared: a write through either handle
    // reaches BOTH plans' next execute. Host-written inputs that must
    // diverge must be declared via `declare_input`.
    replica.buffers()[1].clone().copyin(&f32_bytes([5.0, 6.0, 7.0, 8.0])).unwrap();
    plan.execute().unwrap();
    replica.execute().unwrap();
    assert_eq!(read_f32(plan.output_buffer().unwrap()), [5.0, 6.0, 7.0, 8.0]);
    assert_eq!(read_f32(replica.output_buffer().unwrap()), [5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn declared_input_forks_with_snapshot() {
    let (mut plan, _) = copy_plan(&[1.0, 2.0, 3.0, 4.0]);
    plan.declare_input(1).unwrap();

    let mut replica = plan.replicate().unwrap();
    assert_ne!(plan.buffers()[1].storage_id(), replica.buffers()[1].storage_id(), "declared input storage must fork");
    assert_eq!(read_f32(&replica.buffers()[1]), [1.0, 2.0, 3.0, 4.0], "forked input snapshots contents");

    replica.buffer_at_mut(1).unwrap().copyin(&f32_bytes([9.0, 8.0, 7.0, 6.0])).unwrap();
    replica.execute().unwrap();
    assert_eq!(read_f32(replica.output_buffer().unwrap()), [9.0, 8.0, 7.0, 6.0]);
    assert_eq!(read_f32(&plan.buffers()[1]), [1.0, 2.0, 3.0, 4.0], "original input must not see the write");

    // Declarations carry over: a second-generation replica forks its input
    // from the first replica, not from the original.
    let second = replica.replicate().unwrap();
    assert_ne!(second.buffers()[1].storage_id(), replica.buffers()[1].storage_id());
    assert_eq!(read_f32(&second.buffers()[1]), [9.0, 8.0, 7.0, 6.0]);
}

#[test]
fn replicate_preserves_arena_view_aliasing() {
    let calls = Arc::new(AtomicUsize::new(0));
    let arena = cpu_buffer(DType::Float32, 8);
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let first = builder.add_buffer(31_001, arena.view(0, 16).unwrap());
    let second = builder.add_buffer(31_002, arena.view(16, 16).unwrap());
    let source = builder.add_buffer(31_003, f32_buffer(&[1.0, 2.0, 3.0, 4.0]));
    builder.add_kernel(prepared(31_004, cached(copy4f32(&calls), 2), vec![first, source]));
    let mut chained = prepared(31_005, cached(copy4f32(&calls), 2), vec![second, first]);
    chained.dependencies = vec![31_004];
    builder.add_kernel(chained);
    builder.set_output_buffer(second);
    let plan = builder.build().unwrap();

    let replica = plan.replicate().unwrap();
    let (head, tail) = (&replica.buffers()[0], &replica.buffers()[1]);
    assert_eq!(head.storage_id(), tail.storage_id(), "arena views must land on one forked storage");
    assert_ne!(head.storage_id(), plan.buffers()[0].storage_id());
    assert_eq!((head.offset(), tail.offset()), (0, 16), "arena offsets must be preserved");

    replica.execute().unwrap();
    assert_eq!(read_f32(replica.output_buffer().unwrap()), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn replicate_executes_concurrently_with_the_original() {
    let (mut plan, _) = copy_plan(&[1.0, 2.0, 3.0, 4.0]);
    plan.declare_input(1).unwrap();
    let mut replica = plan.replicate().unwrap();
    replica.buffer_at_mut(1).unwrap().copyin(&f32_bytes([9.0, 8.0, 7.0, 6.0])).unwrap();

    std::thread::scope(|scope| {
        let original = scope.spawn(|| {
            for _ in 0..100 {
                plan.execute().unwrap();
            }
            read_f32(plan.output_buffer().unwrap())
        });
        let forked = scope.spawn(|| {
            for _ in 0..100 {
                replica.execute().unwrap();
            }
            read_f32(replica.output_buffer().unwrap())
        });
        assert_eq!(original.join().unwrap(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(forked.join().unwrap(), [9.0, 8.0, 7.0, 6.0]);
    });
}

#[test]
fn declare_input_rejects_out_of_range_index() {
    let (mut plan, _) = copy_plan(&[1.0, 2.0, 3.0, 4.0]);
    let err = plan.declare_input(99).expect_err("out-of-range input index must fail loud");
    assert!(err.to_string().contains("out of range"), "{err}");
}

// ── graph hazard analysis ──────────────────────────────────────────────────

/// The whole point of [`GraphHazards`]: buffers that share an allocation are
/// distinguished by their *address*, so a producer covering a whole arena and a
/// consumer reading a 12-byte view inside it compare unequal. Keying the walk on
/// the address alone emitted no edge, and the captured graph — which runs
/// unordered nodes concurrently — then raced them.
#[test]
fn overlapping_views_of_one_allocation_are_ordered() {
    let arena = 0x1000usize;
    let whole = arena..arena + 4096;
    let view = arena + 512..arena + 524;
    let mut hazards = GraphHazards::default();

    // 0: fills the whole arena.
    assert!(hazards.record(0, &[], std::slice::from_ref(&whole)).is_empty());
    // 1: reads a 12-byte view inside 0's write, at a different address.
    assert_eq!(hazards.record(1, std::slice::from_ref(&view), &[]), vec![0], "RAW inside a covering write");
    // 2: overwrites that same view — WAW against 0, WAR against 1.
    assert_eq!(hazards.record(2, &[], std::slice::from_ref(&view)), vec![0, 1], "WAW/WAR inside a covering write");
    // 3: re-reads the whole arena — sees both the covering write and the patch.
    assert_eq!(hazards.record(3, std::slice::from_ref(&whole), &[]), vec![0, 2], "a covering read sees every writer");
}

#[test]
fn disjoint_views_of_one_allocation_stay_independent() {
    let arena = 0x1000usize;
    let (low, high) = (arena..arena + 64, arena + 64..arena + 128);
    let mut hazards = GraphHazards::default();
    assert!(hazards.record(0, &[], std::slice::from_ref(&low)).is_empty());
    let adjacent = hazards.record(1, &[], std::slice::from_ref(&high));
    assert!(adjacent.is_empty(), "adjacent ranges must not serialize");
    assert_eq!(hazards.record(2, std::slice::from_ref(&high), &[]), vec![1]);
}

/// A write that fully covers an earlier access supersedes it: later overlapping
/// accesses reach the dropped kernel through the covering write.
#[test]
fn a_covering_write_supersedes_the_accesses_beneath_it() {
    let base = 0x2000usize;
    let (small, large) = (base..base + 16, base..base + 64);
    let mut hazards = GraphHazards::default();
    hazards.record(0, &[], std::slice::from_ref(&small));
    hazards.record(1, std::slice::from_ref(&small), &[]);
    let covering = hazards.record(2, &[], std::slice::from_ref(&large));
    assert_eq!(covering, vec![0, 1], "the covering write depends on both");
    assert_eq!(hazards.record(3, std::slice::from_ref(&small), &[]), vec![2], "and then stands in for them");
}

/// Zero-length views touch no bytes, so they never conflict.
#[test]
fn empty_ranges_never_conflict() {
    let base = 0x3000usize;
    let (live, empty) = (base..base + 32, base..base);
    let mut hazards = GraphHazards::default();
    hazards.record(0, &[], std::slice::from_ref(&live));
    assert!(hazards.record(1, std::slice::from_ref(&empty), &[]).is_empty());
}

/// The storages a plan reports to scoped sync follow the buffers bound at
/// execute time: a view rebound to its own allocation is reported on its
/// own, with the write the plan makes through it.
#[test]
fn storage_accesses_follow_rebound_buffers() {
    let source = cpu_buffer(DType::UInt8, 4);
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let read_idx = builder.add_buffer(30_001, source.clone());
    // A second binding of the same allocation, written by the copy.
    let alias_idx = builder.add_buffer(30_002, source.clone());
    builder.add_op(copy_op(30_010, vec![alias_idx, read_idx], vec![]));
    builder.set_output_buffer(alias_idx);
    let mut plan = builder.build().expect("build plan");
    let ids = |plan: &ExecutionPlan| -> Vec<(BufferId, bool)> {
        plan.storage_accesses().iter().map(|(buffer, writes)| (buffer.storage_id(), *writes)).collect()
    };
    assert_eq!(ids(&plan), [(source.storage_id(), true)], "one storage, written");

    let replacement = cpu_buffer(DType::UInt8, 4);
    *plan.buffer_at_mut(alias_idx).unwrap() = replacement.clone();
    assert_eq!(ids(&plan), [(source.storage_id(), false), (replacement.storage_id(), true)]);
}
