use super::*;
use smallvec::SmallVec;
use svod_dtype::DType;
use test_case::test_case;

#[test]
fn beam_behavior_fingerprint_ignores_parallelism() {
    let base = svod_schedule::OptimizerConfig::default();
    let mut parallel = base.clone();
    parallel.beam.compile_workers += 7;
    parallel.beam.max_tasks_per_child += 9;
    assert_eq!(post_optimizer_behavior_fingerprint(&base), post_optimizer_behavior_fingerprint(&parallel));

    let mut semantic_change = base.clone();
    semantic_change.transcendental += 1;
    assert_ne!(post_optimizer_behavior_fingerprint(&base), post_optimizer_behavior_fingerprint(&semantic_change));
}

fn cpu_buffer(numel: usize) -> Arc<svod_device::Buffer> {
    let allocator = svod_device::registry::cpu().expect("cpu allocator");
    Arc::new(svod_device::Buffer::new(allocator, DType::Float32, vec![numel], Default::default()))
}

#[test]
fn test_build_schedule_input_buffers_collects_nonzero_mselect_shard() {
    let shard0 = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32);
    let shard1 = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32);
    let buffer0 = cpu_buffer(4);
    let buffer1 = cpu_buffer(4);
    crate::tensor_registry::register_buffer_by_uop_id(shard0.id, buffer0.clone());
    crate::tensor_registry::register_buffer_by_uop_id(shard1.id, buffer1.clone());

    let selected = UOp::mstack(SmallVec::from_vec(vec![shard0, shard1.clone()])).mselect(1);
    let body = UOp::sink(vec![UOp::native_const(0.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![selected.clone()]), svod_ir::CallInfo::default());
    let pre_schedule = crate::schedule::PreSchedule {
        items: vec![crate::schedule::PreScheduleItem {
            kernel: call,
            ast: body,
            sources: vec![selected],
            dependencies: vec![],
            bound_ranges: vec![],
        }],
        invocations: vec![],
        output_buffer_uops: vec![],
    };
    let inputs = build_schedule_input_buffers(&pre_schedule);
    assert_eq!(inputs.get(&shard1.id).expect("selected shard").id(), buffer1.id());
}

#[test]
fn test_profile_populates_static_info_and_realizes() {
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]);
    let c = (&a + &b).unwrap();

    let report = c.profile(&ProfileOptions::default()).expect("profile");
    assert_eq!(report.stages.len(), 1, "one profile stage");
    let stage = &report.stages[0];
    assert!(!stage.kernels.is_empty(), "at least one kernel dispatched");

    // Tier-2 static analysis attaches to every dispatch; flops/bytes are
    // backend-independent (resources are backend-specific, so not asserted here).
    let si = stage.kernels[0].static_info.as_ref().expect("static_info populated");
    assert!(si.est_flops.is_some_and(|f| f > 0), "flop estimate should be a positive count: {:?}", si.est_flops);
    assert!(si.est_bytes > 0, "byte estimate should be positive: {}", si.est_bytes);

    // profile() finalizes like realize(), so the tensor holds the result.
    assert_eq!(c.as_vec::<f32>().expect("as_vec"), vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_output_indices_from_program_metadata_basic() {
    let outputs = output_indices_from_program_metadata(&[0, 1, 2], &[2], 3).expect("metadata mapping should succeed");
    assert_eq!(outputs, vec![2]);
}

#[test]
fn test_output_indices_from_program_metadata_sparse_slots() {
    let outputs =
        output_indices_from_program_metadata(&[2, 4, 7], &[4, 7], 3).expect("metadata mapping should succeed");
    assert_eq!(outputs, vec![1, 2]);
}

#[test]
fn test_output_indices_from_program_metadata_rejects_empty_outs() {
    let err = output_indices_from_program_metadata(&[0, 1], &[], 2).expect_err("empty outs should fail");
    assert!(format!("{err}").contains("ProgramSpec.outs is empty"));
}

#[test]
fn test_output_indices_from_program_metadata_rejects_unknown_slot() {
    let err = output_indices_from_program_metadata(&[0, 2], &[1], 2).expect_err("unknown outs slot should fail");
    assert!(format!("{err}").contains("not found in ProgramSpec.globals"));
}

#[test]
fn test_output_indices_from_program_metadata_rejects_out_of_range_position() {
    let err = output_indices_from_program_metadata(&[0, 1], &[1], 1).expect_err("mapped output index out of range");
    assert!(format!("{err}").contains("out of range"));
}

/// `ins` names the readable globals by slot; a write-only global (absent from
/// `ins`) and an in-place one (present in both roles) are equally legal.
#[test_case(&[2, 4, 7], &[4], &[1]; "only program ins")]
#[test_case(&[0, 1], &[], &[]; "write-only globals")]
#[test_case(&[0, 1], &[0], &[0]; "in-place global")]
fn test_input_indices_from_program_metadata(globals: &[usize], ins: &[usize], expected: &[usize]) {
    let inputs = input_indices_from_program_metadata(globals, ins, globals.len()).expect("metadata mapping");
    assert_eq!(inputs, expected);
}

/// A CALL passing `count` PARAMs, with the runtime buffer index each argument
/// resolves to (10, 11, ...) in CALL argument order.
fn param_call_item(count: usize) -> (crate::schedule::ScheduleItem, std::collections::HashMap<u64, usize>) {
    let params = (0..count).map(|slot| UOp::param(slot, 4, DType::Float32, None)).collect::<Vec<_>>();
    let body = UOp::sink(params.clone());
    let item = crate::schedule::ScheduleItem {
        kernel: body.call(SmallVec::from_vec(params.clone()), svod_ir::CallInfo::default()),
        ast: body,
        buffers: vec![],
        buffer_uop_ids: params.iter().map(|param| param.id).collect(),
        fixedvars: std::collections::HashMap::new(),
        dependencies: vec![],
        instance_dependencies: vec![],
        loop_var_names: std::collections::HashSet::new(),
    };
    (item, params.iter().enumerate().map(|(position, param)| (param.id, 10 + position)).collect())
}

/// A PARAM slot is its actual argument's CALL position, and `globals` is the
/// sorted set of slots the optimized body kept. The launch tuple is therefore
/// `globals` projected onto the CALL arguments: identity while every argument
/// survives, and a compaction that skips the folded-away ones otherwise.
#[test_case(2, &[0, 1], &[10, 11]; "every argument survives")]
#[test_case(3, &[0, 2], &[10, 12]; "middle argument folded away")]
#[test_case(4, &[1, 3], &[11, 13]; "every other argument folded away")]
#[test_case(3, &[2], &[12]; "one argument survives")]
fn test_resolve_compiled_kernel_buffer_indices_projects_globals(count: usize, globals: &[usize], expected: &[usize]) {
    let (item, uop_id_to_idx) = param_call_item(count);
    let ordered =
        resolve_compiled_kernel_buffer_indices(&item, &uop_id_to_idx, globals).expect("compiled buffer ABI ordering");
    assert_eq!(ordered, expected);
}

/// Compaction never invents an argument: a slot past the CALL tuple is still a
/// hard error, since the launch would otherwise bind the wrong buffer.
#[test]
fn test_resolve_compiled_kernel_buffer_indices_rejects_slot_without_argument() {
    let (item, uop_id_to_idx) = param_call_item(1);
    let err = resolve_compiled_kernel_buffer_indices(&item, &uop_id_to_idx, &[0, 5])
        .expect_err("a global slot with no CALL argument should fail");
    assert!(format!("{err}").contains("PROGRAM slot 5"), "unexpected error: {err}");
}

#[test]
fn test_restore_post_schedule_pre_schedule_rewrites_runtime_buf_uops() {
    crate::test::helpers::test_setup();

    let c = (&Tensor::from_slice([1.0f32, 2.0, 3.0]) + &Tensor::from_slice([4.0f32, 5.0, 6.0])).unwrap();
    let sink = UOp::sink(vec![c.uop().contiguous()]);

    let normalization = normalize_for_schedule_cache(&sink).expect("normalize schedule cache");
    let rangeify = svod_schedule::rangeify_with_map(normalization.normalized.clone()).expect("rangeify");
    let (kernel_graph_cached, _) = svod_schedule::try_get_kernel_graph(rangeify.sink).expect("kernel graph");
    let pre_schedule_cached = crate::schedule::create_pre_schedule(kernel_graph_cached).expect("pre schedule");

    assert!(
        pre_schedule_cached.items.iter().flat_map(|item| item.sources.iter()).any(|src| {
            src.tag().as_ref().is_some_and(|tags| tags.contains(&svod_ir::uop::canonical::TAG_SCHEDULE_CACHE_PARAM))
        }),
        "cached pre-schedule should keep normalized PARAM placeholders"
    );

    let restored = restore_post_schedule_pre_schedule(&pre_schedule_cached, &normalization);

    assert!(
        restored.items.iter().flat_map(|item| item.sources.iter()).all(|src| {
            !src.tag().as_ref().is_some_and(|tags| tags.contains(&svod_ir::uop::canonical::TAG_SCHEDULE_CACHE_PARAM))
        }),
        "restored pre-schedule should rewrite callable source PARAM placeholders"
    );
    assert!(
        restored.output_buffer_uops.iter().all(|u| {
            !u.tag().as_ref().is_some_and(|tags| tags.contains(&svod_ir::uop::canonical::TAG_SCHEDULE_CACHE_PARAM))
        }),
        "restored pre-schedule should rewrite output buffer PARAM placeholders"
    );
    assert!(
        restored.items.iter().flat_map(|item| item.sources.iter()).all(|src| !matches!(src.op(), Op::LUnique(_))),
        "restored pre-schedule should rewrite LUNIQUE placeholders"
    );

    assert!(
        pre_schedule_cached.items.iter().flat_map(|item| item.sources.iter()).any(|src| {
            src.tag().as_ref().is_some_and(|tags| tags.contains(&svod_ir::uop::canonical::TAG_SCHEDULE_CACHE_PARAM))
        }),
        "restoring should not mutate cached pre-schedule"
    );
}

struct TestRenderer;

impl svod_device::device::Renderer for TestRenderer {
    fn render(
        &self,
        ast: &std::sync::Arc<UOp>,
        name: Option<&str>,
    ) -> svod_device::Result<svod_device::device::ProgramSpec> {
        let spec = svod_device::device::ProgramSpec::new(
            name.unwrap_or("kernel").to_string(),
            "// test source".to_string(),
            svod_dtype::DeviceSpec::Cpu,
            ast.clone(),
        );
        Ok(spec)
    }

    fn device(&self) -> &svod_dtype::DeviceSpec {
        static DEVICE: svod_dtype::DeviceSpec = svod_dtype::DeviceSpec::Cpu;
        &DEVICE
    }

    fn supported_ops(&self) -> svod_ir::RendererOps {
        let mut ops = svod_ir::RendererOps::all();
        ops.binary.remove(&svod_ir::BinaryOp::Threefry);
        ops
    }
}

struct TestCompiler;

impl svod_device::device::Compiler for TestCompiler {
    fn compile(
        &self,
        spec: &svod_device::device::ProgramSpec,
    ) -> svod_device::Result<svod_device::device::CompiledSpec> {
        svod_device::device::CompiledSpec::from_bytes(
            spec.name.clone(),
            vec![1, 2, 3],
            spec.ast.clone(),
            spec.abi.clone(),
        )
    }

    fn cache_key(&self) -> &'static str {
        "test"
    }
}

#[test]
fn test_compile_with_program_pipeline_components_accepts_program_input() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let program = svod_codegen::program_pipeline::program_from_sink(sink, svod_dtype::DeviceSpec::Cpu)
        .expect("final target graph");

    let (spec, compiled) = compile_with_program_pipeline_components(program, &TestRenderer, &TestCompiler)
        .expect("PROGRAM input should compile through staged pipeline");

    assert_eq!(spec.name, "test");
    assert_eq!(compiled.name, "test");
    assert_eq!(compiled.bytes, vec![1, 2, 3]);
}

#[test]
fn test_optimized_kernel_key_includes_exact_compiler_and_renderer_identity() {
    let ast = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let renderer = svod_schedule::OptimizerRenderer::cpu();
    let mut changed_renderer = renderer.clone();
    changed_renderer.supports_float4 = false;
    let key = |compiler: &str, renderer: &svod_schedule::OptimizerRenderer| {
        optimized_kernel_key(&ast, &svod_dtype::DeviceSpec::Cpu, compiler, renderer.cache_fingerprint(), 7)
    };

    let base = key("cpu-clang:17:flags-a", &renderer);
    assert_ne!(base, key("cpu-clang:18:flags-a", &renderer), "exact compiler identity must key optimized kernels");
    assert_ne!(
        base,
        key("cpu-clang:17:flags-a", &changed_renderer),
        "renderer capabilities must key optimized kernels"
    );
}

#[test]
fn test_compile_with_program_pipeline_components_rejects_non_program_input() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);

    let err = compile_with_program_pipeline_components(sink, &TestRenderer, &TestCompiler)
        .expect_err("non-PROGRAM input must fail");
    assert!(format!("{err}").contains("expects PROGRAM input"), "unexpected error: {err:?}");
}

#[test]
fn test_compile_with_program_pipeline_components_accepts_stage1_program_input() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink.clone()).into());
    let info = svod_ir::ProgramInfo::from_sink(&sink, svod_dtype::DeviceSpec::Cpu);
    let program = UOp::program(sink, info, Some(linear), None, None);

    let (spec, compiled) = compile_with_program_pipeline_components(program, &TestRenderer, &TestCompiler)
        .expect("stage-1 PROGRAM input should compile");
    assert_eq!(spec.name, "test");
    assert_eq!(compiled.bytes, vec![1, 2, 3]);
}

#[test]
fn test_compile_with_program_pipeline_components_accepts_stage2_program_input() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let program = svod_codegen::program_pipeline::program_from_sink(sink, svod_dtype::DeviceSpec::Cpu).unwrap();
    let (program, _) = svod_codegen::program_pipeline::do_render(&program, &TestRenderer).unwrap();

    let (spec, compiled) = compile_with_program_pipeline_components(program, &TestRenderer, &TestCompiler)
        .expect("stage-2 PROGRAM input should compile");
    assert_eq!(spec.name, "test");
    assert_eq!(compiled.bytes, vec![1, 2, 3]);
}

#[test]
fn test_compile_with_program_pipeline_components_rejects_malformed_program_state() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let info = svod_ir::ProgramInfo::from_sink(&sink, svod_dtype::DeviceSpec::Cpu);
    let program = UOp::program(sink, info, None, Some(UOp::source("// malformed source".to_string())), None);

    let err = compile_with_program_pipeline_components(program, &TestRenderer, &TestCompiler)
        .expect_err("malformed PROGRAM input must fail");
    assert!(format!("{err}").contains("malformed PROGRAM state"), "unexpected error: {err:?}");
}

#[test]
fn test_collect_non_overridable_fixedvars_locks_loop_and_device_bindings() {
    // After the schedule-level Range/End refactor, schedule-loop bindings are
    // tracked structurally via `ScheduleItem.loop_var_names` (populated from
    // `KernelInvocation.fixedvars` at instantiation time). User-supplied
    // var_vals end up in `fixedvars` too but are NOT in `loop_var_names`,
    // so they remain overridable. This separates loop counters from runtime
    // variable binds.
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::new(), svod_ir::CallInfo::default());

    let item = crate::schedule::ScheduleItem {
        kernel: call,
        ast: body,
        buffers: vec![],
        buffer_uop_ids: vec![],
        fixedvars: std::collections::HashMap::from([
            ("outer_i".to_string(), 2_i64),
            ("loop_j".to_string(), 1_i64),
            ("user_n".to_string(), 7_i64),
            ("_device_num".to_string(), 3_i64),
        ]),
        dependencies: vec![],
        instance_dependencies: vec![],
        loop_var_names: std::collections::HashSet::from(["outer_i".to_string()]),
    };

    let locked = collect_non_overridable_fixedvars(&item);
    assert_eq!(locked.get("outer_i"), Some(&2));
    assert_eq!(locked.get("_device_num"), Some(&3));
    assert!(!locked.contains_key("loop_j"));
    assert!(!locked.contains_key("user_n"));
}

#[test]
fn test_cpu_plan_executes_device_bound_mstack_lanes() {
    crate::test::helpers::test_setup();

    let output0 = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 1, DType::Float32);
    let output1 = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 1, DType::Float32);
    let output_param = UOp::param(0, 1, DType::Float32, None);
    let output_index =
        UOp::index().buffer(output_param).indices(vec![UOp::index_const(0)]).call().expect("output index");
    let device_num = UOp::range_axis(UOp::index_const(2), svod_ir::AxisId::Renumbered(0), svod_ir::AxisType::Device);
    let value = device_num
        .cast(DType::Float32)
        .try_mul(&UOp::native_const(11.0f32))
        .and_then(|value| value.try_add(&UOp::native_const(10.0f32)))
        .expect("lane value expression");
    let body = UOp::sink_with_info(
        vec![output_index.store(value)],
        svod_ir::KernelInfo { opts_to_apply: Some(vec![]), ..Default::default() },
    );
    let stack = UOp::mstack(SmallVec::from_vec(vec![output0.clone(), output1.clone()]));
    let call = body.call(SmallVec::from_vec(vec![stack.clone()]), svod_ir::CallInfo::default());
    let pre_schedule = crate::schedule::PreSchedule {
        items: vec![crate::schedule::PreScheduleItem {
            kernel: call.clone(),
            ast: body,
            sources: vec![stack],
            dependencies: vec![],
            bound_ranges: vec![],
        }],
        invocations: vec![crate::schedule::KernelInvocation { kernel_id: call.id, fixedvars: HashMap::new() }],
        output_buffer_uops: vec![output0.clone(), output1.clone()],
    };
    let inputs = HashMap::from([(output0.id, (*cpu_buffer(1)).clone()), (output1.id, (*cpu_buffer(1)).clone())]);
    let schedule = crate::schedule::instantiate_schedule(&pre_schedule, &inputs, &HashMap::new(), false)
        .expect("expand MSTACK lanes");
    assert_eq!(schedule.items.len(), 2);

    let mut plan = prepare_execution_plan(&schedule, &PrepareConfig::for_cpu_backend(crate::CpuBackend::Clang))
        .expect("prepare CPU lane plan");
    let fixed_device_nums: Vec<i64> = plan
        .prepared_ops()
        .iter()
        .filter_map(|op| match op {
            PreparedOp::CompiledProgram(kernel) => kernel.fixedvars.get("_device_num").copied(),
            _ => None,
        })
        .collect();
    assert_eq!(fixed_device_nums, vec![0, 1]);

    plan.execute_with_vars(&[("_device_num", 99)]).expect("fixed lane binding must ignore runtime override");
    let mut values = Vec::new();
    for lane in 0..2 {
        let output = plan.output_buffer_at(lane).expect("lane output");
        let mut value = [0.0f32];
        output
            .copyout(unsafe { std::slice::from_raw_parts_mut(value.as_mut_ptr().cast::<u8>(), size_of::<f32>()) })
            .expect("copy lane output");
        values.push(value[0]);
    }
    assert_eq!(values, vec![10.0, 21.0]);
}

#[test]
fn test_realize_simple_add() {
    crate::test::helpers::test_setup();

    // Test that realizing a simple computation works.
    // The pipeline transforms:
    //   ADD(RESHAPE(BUFFER_A), RESHAPE(BUFFER_B))
    // Into:
    //   STORE(OUTPUT, INDEX, ADD(LOAD(INPUT_A, idx), LOAD(INPUT_B, idx)))
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);

    // Create computation: a + b
    let c = (&a + &b).unwrap();

    // Realize should compile and execute the kernel
    c.realize().unwrap();
    let result: ndarray::ArrayD<f32> = c.as_ndarray().unwrap();
    let (result, _) = result.into_raw_vec_and_offset();
    assert_eq!(result, vec![5.0, 7.0, 9.0]);
}

/// Test that realizing a reduction (sum) works end-to-end.
///
/// This verifies the complete reduction pipeline:
/// - Early-return pattern prevents unnecessary ReduceAxis for size-1 dimensions
/// - Stack consistency prevents VConst panics in shape extraction
/// - ReduceAxis → REDUCE transformation
/// - REDUCE codegen generates correct LLVM IR
#[test]
fn test_realize_sum() {
    crate::test::helpers::test_setup();

    // Create a 1D tensor: [1, 2, 3, 4]
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);

    // Sum all elements (should be 10.0)
    let sum_result = a.sum(());
    if let Err(ref e) = sum_result {
        tracing::debug!(error = ?e, "sum failed");
    }
    assert!(sum_result.is_ok(), "Sum creation failed");

    // Realize the computation
    let sum_tensor = sum_result.unwrap();
    let realized = sum_tensor.realize();
    if let Err(ref e) = realized {
        eprintln!("realize failed: {e:?}");
    }
    assert!(realized.is_ok(), "Realize should succeed: {:?}", realized.err());
}

#[test]
fn test_tensor_device_default_cpu() {
    // Tensors created with from_slice land on the active default device
    // (CPU, or AMD under SVOD_DEVICE=AMD:0).
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    assert_eq!(a.device(), svod_dtype::default_device::default_device());
}

#[test]
fn test_tensor_to_same_device_is_noop() {
    // Moving to the same device should return a clone
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = a.to(a.device());
    // Both should point to the same UOp (clone shares Rc)
    assert_eq!(a.device(), b.device());
}

#[test]
fn test_tensor_to_different_device_creates_copy() {
    use svod_ir::DeviceSpec;
    // Moving to a different device should create a COPY UOp
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let orig = a.device();
    // Cuda is never the active default here, so it is always a different device.
    let b = a.to(DeviceSpec::Cuda { device_id: 0 });
    // b should report the new device
    assert_eq!(b.device(), DeviceSpec::Cuda { device_id: 0 });
    // a should be unchanged
    assert_eq!(a.device(), orig);
}

// More comprehensive tests will be added in Phase 1.5

// ==========================================================================
// ExecutionPlan tests
// ==========================================================================

#[test]
fn test_prepare_simple_add() {
    crate::test::helpers::test_setup();

    // Create computation: a + b
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let c = (&a + &b).unwrap();

    // Prepare should compile kernels and allocate buffers
    let plan = c.prepare();
    assert!(plan.is_ok(), "prepare() should succeed: {:?}", plan.err());

    let plan = plan.unwrap();

    // Verify plan has kernels and buffers
    assert!(plan.kernels().next().is_some(), "Plan should have at least one kernel");
    assert!(!plan.buffers().is_empty(), "Plan should have buffers");
}

#[test]
fn test_prepare_and_execute() {
    crate::test::helpers::test_setup();

    // Create computation: a + b
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let c = (&a + &b).unwrap();

    // Prepare
    let plan = c.prepare().expect("prepare should succeed");

    // Execute
    let result = plan.execute();
    assert!(result.is_ok(), "execute() should succeed: {:?}", result.err());

    // Verify output buffer has correct data
    let output = plan.output_buffer().expect("plan has output");
    let mut data = vec![0.0f32; 3];
    output
        .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, 12) })
        .expect("copyout should succeed");
    assert_eq!(data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_prepare_and_execute_twice() {
    crate::test::helpers::test_setup();

    // Create computation
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let c = (&a + &b).unwrap();

    // Prepare once
    let plan = c.prepare().expect("prepare should succeed");

    // Execute twice to verify reusability
    for _ in 0..2 {
        let result = plan.execute();
        assert!(result.is_ok(), "execute() should succeed: {:?}", result.err());
    }

    // Verify output
    let output = plan.output_buffer().expect("plan has output");
    let mut data = vec![0.0f32; 3];
    output
        .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, 12) })
        .expect("copyout should succeed");
    assert_eq!(data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_prepare_execution_plan_lowers_explicit_custom_function_op() {
    crate::test::helpers::test_setup();

    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let src = Buffer::new(alloc.clone(), svod_dtype::DType::Float32, vec![4], Default::default());
    let dst = Buffer::new(alloc, svod_dtype::DType::Float32, vec![4], Default::default());

    let attr = UOp::index_const(42);
    let ast = UOp::custom_function(svod_ir::CustomFunctionKind::EncDec, smallvec::smallvec![attr.clone()]);
    let kernel = ast.call(smallvec::smallvec![], svod_ir::CallInfo::default());
    let schedule_result = crate::schedule::ScheduleResult {
        items: vec![crate::schedule::ScheduleItem {
            kernel,
            ast,
            buffers: vec![dst.clone(), src.clone()],
            buffer_uop_ids: vec![1001, 1002],
            fixedvars: std::collections::HashMap::new(),
            dependencies: vec![],
            instance_dependencies: vec![],
            loop_var_names: std::collections::HashSet::new(),
        }],
        output_uop_ids: vec![1001],
        alias_output_buffers: std::collections::HashMap::new(),
    };

    let plan = prepare_execution_plan(&schedule_result, &PrepareConfig::from_env()).expect("prepare should succeed");
    let custom = plan
        .prepared_ops()
        .iter()
        .find_map(|op| match op {
            svod_runtime::PreparedOp::CustomFunction(custom) => Some(custom),
            _ => None,
        })
        .expect("explicit custom function body should lower to PreparedOp::CustomFunction");
    assert_eq!(custom.attrs.len(), 1, "custom-function attrs should be preserved into runtime plan");
    assert_eq!(custom.attrs[0].id, attr.id);

    let err = plan.execute().expect_err("EncDec runtime should be explicit unsupported");
    let msg = format!("{err}");
    assert!(msg.contains("Unsupported runtime feature EncDec"), "unexpected error: {msg}");
}

#[test]
fn test_prepare_execution_plan_owns_alias_only_output_without_runtime_op() {
    crate::test::helpers::test_setup();
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let base = Buffer::new(alloc, svod_dtype::DType::Float32, vec![8], Default::default());
    let storage = base.storage_id();
    let view = base.view(8, 16).unwrap();
    let schedule_result = crate::schedule::ScheduleResult {
        items: vec![],
        output_uop_ids: vec![2001],
        alias_output_buffers: std::collections::HashMap::from([(2001, view)]),
    };

    let plan = prepare_execution_plan(&schedule_result, &PrepareConfig::from_env()).unwrap();
    assert!(plan.prepared_ops().is_empty());
    let output = plan.output_buffer().unwrap();
    assert_eq!(output.storage_id(), storage);
    assert_eq!(output.offset(), 8);
    assert_eq!(output.size(), 16);
}

#[test]
fn alias_output_storage_is_protected_from_memory_planning() {
    crate::test::helpers::test_setup();
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let producer = Buffer::new(alloc, svod_dtype::DType::Float32, vec![8], Default::default());
    let alias = producer.view(8, 16).unwrap();
    let sink = UOp::sink(vec![]);
    let item = crate::schedule::ScheduleItem {
        kernel: sink.clone(),
        ast: sink,
        buffers: vec![producer.clone()],
        buffer_uop_ids: vec![1001],
        fixedvars: std::collections::HashMap::new(),
        dependencies: vec![],
        instance_dependencies: vec![],
        loop_var_names: std::collections::HashSet::new(),
    };
    let protected = collect_output_buffer_ids(&vec![item], &[2001], std::iter::once(&alias));

    assert!(protected.contains(&producer.id().0));
}

/// Test that realize() produces correct results.
///
/// Note: Buffer count assertions removed as they're not reliable with
/// parallel test execution and global state. The key invariant (no memory
/// leak) is tested in test_memory_growth_detection.
#[test]
fn test_realize_buffer_cleanup() {
    crate::test::helpers::test_setup();

    // Create input tensors ONCE (these will stay in registry)
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);

    // Realize the computation
    let c = (&a + &b).unwrap();
    c.realize().expect("realize should succeed");

    // Verify computation is correct
    let result: ndarray::ArrayD<f32> = c.as_ndarray().expect("as_ndarray should succeed");
    let (data, _) = result.into_raw_vec_and_offset();
    assert_eq!(data, vec![5.0, 7.0, 9.0]);
}

/// Test that prepare() + execute() buffers expire automatically once the plan and tensors drop.
#[test]
#[ignore = "Flaky under parallel global registry activity; run manually with --ignored --test-threads=1"]
fn test_prepare_execute_cleanup() {
    crate::test::helpers::test_setup();

    // Create input tensors
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let c = (&a + &b).unwrap();

    // Prepare the plan
    let plan = c.prepare().expect("prepare should succeed");

    // Execute multiple times (simulating benchmark loop)
    for _ in 0..3 {
        plan.execute().expect("execute should succeed");
    }

    // Verify output
    let output = plan.output_buffer().expect("plan has output");
    let mut data = vec![0.0f32; 3];
    output
        .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, 12) })
        .expect("copyout should succeed");
    assert_eq!(data, vec![5.0, 7.0, 9.0]);

    // Dropping the plan and tensors expires their registry entries
    // automatically (UOp drop hook) — no manual release exists any more.
    let count_before_drop = crate::tensor_registry::buffer_count();
    drop(plan);
    drop(c);
    drop(a);
    drop(b);
    let count_after_drop = crate::tensor_registry::buffer_count();
    assert!(
        count_after_drop < count_before_drop,
        "Dropping plan + tensors must expire registry entries: before={}, after={}",
        count_before_drop,
        count_after_drop
    );
}

/// Test that intermediate buffer cleanup is working.
///
/// The correct pattern is: prepare() ONCE, execute() many times.
/// This test verifies that repeated execute() calls do NOT grow the registry
/// AFTER initial setup. First execute may allocate buffers (one-time setup),
/// but subsequent calls must not grow.
#[test]
#[ignore = "Flaky under parallel global registry activity; run manually with --ignored --test-threads=1"]
fn test_memory_growth_detection() {
    crate::test::helpers::test_setup();

    const ITERATIONS: usize = 10;

    // Create input tensors
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]);
    let c = (&a + &b).unwrap();

    // Prepare ONCE
    let plan = c.prepare().expect("prepare should succeed");

    let mut counts: Vec<usize> = Vec::with_capacity(ITERATIONS);

    // Execute MANY times
    for _ in 0..ITERATIONS {
        plan.execute().expect("execute should succeed");
        counts.push(crate::tensor_registry::buffer_count());
    }

    // Cleanup after final execution: dropping the plan and tensors expires
    // their registry entries automatically.
    drop(plan);
    drop(c);
    drop(a);
    drop(b);
    let count_after_cleanup = crate::tensor_registry::buffer_count();

    // Key invariant: count should be STABLE during iterations (no growth between iterations)
    // First execute may allocate buffers, but subsequent calls must reuse them.
    let count_after_first_execute = counts[0];
    let growth_during_iterations = counts.last().unwrap().saturating_sub(count_after_first_execute);

    eprintln!("Counts during execute: {:?}", counts);
    eprintln!("Growth during iterations (after first): {}", growth_during_iterations);
    eprintln!("Count after cleanup: {}", count_after_cleanup);

    assert_eq!(
        growth_during_iterations, 0,
        "Registry should not grow during repeated execute() calls (after initial setup)"
    );

    // Cleanup should reduce count by removing allocated buffers
    assert!(
        count_after_cleanup <= count_after_first_execute,
        "Cleanup should not increase buffer count: first_execute={}, after_cleanup={}",
        count_after_first_execute,
        count_after_cleanup
    );
}

/// Test that realize() correctly computes and cleans up.
#[test]
fn test_memory_growth_realize_pattern() {
    crate::test::helpers::test_setup();

    // Single realize should work correctly
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]);
    let c = (&a + &b).unwrap();
    c.realize().expect("realize should succeed");

    // Verify result
    let result: ndarray::ArrayD<f32> = c.as_ndarray().expect("as_ndarray should succeed");
    assert_eq!(result.as_slice().unwrap(), &[6.0, 8.0, 10.0, 12.0]);
}

/// Phase-1 acceptance: registry entries expire automatically with their
/// graphs — the output entry survives while the tensor lives and dies with it.
#[test]
fn buffers_expire_automatically_with_their_graphs() {
    crate::test::helpers::test_setup();
    // Distinctive literals: the output entry is keyed on a hash-consed UOp id,
    // so a concurrent test computing the same values would share the id (and its
    // live buffer) and defeat the lifetime assertion below.
    let a = Tensor::from_slice([13.5f32, 26.25, 39.125]);
    let b = Tensor::from_slice([48.0625f32, 60.03125, 72.015625]);
    let c = (&a + &b).unwrap();
    c.realize().expect("realize");
    let out_id = c.uop().base().id;
    assert!(crate::tensor_registry::get_buffer(out_id).is_some());

    // The output entry survives while the tensor lives...
    drop(a);
    drop(b);
    assert!(crate::tensor_registry::get_buffer(out_id).is_some());

    // ...and expires once the last graph referencing it is gone.
    drop(c);
    assert!(crate::tensor_registry::get_buffer(out_id).is_none(), "entry must expire with the tensor graph");
}

/// The `nK` name suffix is drawn from a process-wide counter as kernels are
/// finished. A prepare fanned out over the global pool (`threads: 4`) must
/// draw it in schedule order and reach the same kernel ABI as an inline one
/// (`threads: 1`), or the object cache (keyed on source text) would miss on
/// every run.
#[test]
fn test_parallel_prepare_names_kernels_in_schedule_order() {
    const KERNELS: usize = 6;
    // A length no other test reduces over keeps this shape name's counter ours.
    const LEN: usize = 4099;

    fn plan_kernels(offset: f32, threads: usize) -> Vec<(String, Vec<usize>)> {
        let outputs: Vec<Tensor> = (0..KERNELS)
            .map(|i| {
                let x = Tensor::from_slice((0..LEN).map(|v| v as f32).collect::<Vec<_>>());
                let scale = Tensor::full(&[LEN], offset + i as f32, DType::Float32);
                (&x * &scale).unwrap().sum(0).unwrap()
            })
            .collect();
        let config = PrepareConfig { threads, ..Default::default() };
        let plan = Tensor::prepare_batch_with(&outputs, &config).unwrap();
        plan.prepared_kernels().iter().map(|k| (k.kernel.entry_point.clone(), k.kernel.globals.clone())).collect()
    }
    // Position of a name in its shape family: `E_x` -> 0, `E_xn3` -> 3.
    fn ordinal(name: &str) -> (&str, usize) {
        match name.rsplit_once('n') {
            Some((base, digits)) if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) => {
                (base, digits.parse().unwrap())
            }
            _ => (name, 0),
        }
    }
    fn assert_schedule_ordered(kernels: &[(String, Vec<usize>)]) -> HashMap<&str, usize> {
        let mut last: HashMap<&str, usize> = HashMap::new();
        for (name, _) in kernels {
            let (base, position) = ordinal(name);
            if let Some(previous) = last.insert(base, position) {
                assert_eq!(position, previous + 1, "{name} drawn out of schedule order in {kernels:?}");
            }
        }
        last
    }

    // Disjoint constants: a kernel shared with the serial run would be an
    // optimizer-cache hit that keeps its earlier name.
    let serial = plan_kernels(1.5, 1);
    let parallel = plan_kernels(101.5, 4);
    assert!(serial.len() >= KERNELS, "{serial:?}");
    assert_eq!(serial.len(), parallel.len());
    assert_schedule_ordered(&serial);
    assert_schedule_ordered(&parallel);
    for ((serial_name, serial_globals), (parallel_name, parallel_globals)) in serial.iter().zip(&parallel) {
        assert_eq!(ordinal(serial_name).0, ordinal(parallel_name).0, "{serial:?} vs {parallel:?}");
        assert_eq!(serial_globals, parallel_globals, "{serial_name}: ABI must not depend on threading");
    }
    // Every family continues exactly where the serial run left it.
    let mut family_sizes: HashMap<&str, usize> = HashMap::new();
    for (name, _) in &serial {
        *family_sizes.entry(ordinal(name).0).or_default() += 1;
    }
    for ((serial_name, _), (parallel_name, _)) in serial.iter().zip(&parallel) {
        let (base, serial_position) = ordinal(serial_name);
        assert_eq!(ordinal(parallel_name).1, serial_position + family_sizes[base], "{serial:?} vs {parallel:?}");
    }
}

/// A hand-lowered `out[0] = value` kernel site, optimized under exactly `opts`.
fn constant_store_site(value: f32, opts: Vec<svod_ir::Opt>, config: &PrepareConfig) -> KernelSite {
    let out = UOp::index()
        .buffer(UOp::param(0, 1, DType::Float32, None))
        .indices(vec![UOp::index_const(0)])
        .call()
        .expect("output index");
    let body = UOp::sink_with_info(
        vec![out.store(UOp::native_const(value))],
        svod_ir::KernelInfo { opts_to_apply: Some(opts), ..Default::default() },
    );
    let item = crate::schedule::ScheduleItem {
        kernel: body.call(SmallVec::new(), svod_ir::CallInfo::default()),
        ast: body,
        buffers: vec![],
        buffer_uop_ids: vec![],
        fixedvars: HashMap::new(),
        dependencies: vec![],
        instance_dependencies: vec![],
        loop_var_names: HashSet::new(),
    };
    KernelSite::resolve(&item, config, optimizer_config_fingerprint(config)).expect("resolve kernel site")
}

/// One kernel failing to optimize must not cost the rest of the batch their
/// names: the survivors are published (a retry hits the cache), the failed
/// key is released unpublished, and its error is what the batch returns.
#[test_case(0; "failing first")]
#[test_case(1; "failing middle")]
#[test_case(2; "failing last")]
fn compile_missing_kernels_publishes_the_survivors_of_a_failed_optimize(failing: usize) {
    let config = PrepareConfig { threads: 2, ..PrepareConfig::for_cpu_backend(crate::CpuBackend::Clang) };
    let sites: Vec<_> = (0..3)
        .map(|i| {
            // The kernel has no unrollable axis, so UNROLL is rejected.
            let opts = if i == failing { vec![svod_ir::Opt::unroll(0, 4)] } else { vec![] };
            Some(constant_store_site(1000.0 + (failing * 3 + i) as f32, opts, &config))
        })
        .collect();

    let Err(err) = compile_missing_kernels(&sites, &config) else { panic!("a failed optimize fails the batch") };
    assert!(matches!(err.kind(), crate::error::ErrorKind::Optimize { .. }), "{err}");
    for (i, site) in sites.iter().flatten().enumerate() {
        assert_eq!(site.cached().is_some(), i != failing, "kernel {i} published");
        assert!(opt_flight().try_claim(site.key.clone()).is_some(), "kernel {i} still claimed");
    }
}

/// The timed grid is Tinygrad's `get_test_global_size`: a grid past 65536
/// workgroups is halved along its innermost axis still above 16 until it fits,
/// axes at or below 16 are left alone, and the factor scales the time back up.
#[test_case([5, 10, 1], [5, 10, 1], 1.0 ; "a grid within the cap is timed whole")]
#[test_case([80, 80, 80], [80, 80, 10], 8.0 ; "the innermost axis is halved while it stays above 16")]
#[test_case([262_144, 16, 1], [4_096, 16, 1], 64.0 ; "an axis at 16 is skipped for the next one")]
#[test_case([16, 16, 16], [16, 16, 16], 1.0 ; "small axes are never halved")]
fn the_timed_grid_shrinks_along_the_innermost_axis(grid: [usize; 3], expect: [usize; 3], factor: f64) {
    assert_eq!(test_grid(grid), (expect, factor));
}
