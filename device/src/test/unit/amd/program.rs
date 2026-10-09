use super::test_support::{
    MockAmdIface, amd_alloc_or_skip, ensure_hw_signal_pool, install_signal_pool, mock_device, replay_dwords,
    require_multi_xcc, require_single_xcc, scripted_error,
};
use crate::allocator::RawBuffer;
use crate::amd::AmdAllocator;
use crate::amd::device::AmdDevice;
use crate::amd::graph::AmdGraph;
use crate::amd::program::*;
use crate::device::{AbiParamDescriptor, AbiParamKind, GraphKernel, Program};
use crate::error::Error;
use std::sync::Arc;

/// A kernel that stores 0.0 to `buf0[tid]` — the workhorse for every probe that
/// only needs to observe that a dispatch reached the device.
const STORE_ZERO_IR: &str = r#"target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.workitem.id.x()
define amdgpu_kernel void @store_zero(ptr noalias %buf0) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid_ext = zext i32 %tid to i64
  %p = getelementptr inbounds float, ptr %buf0, i64 %tid_ext
  store float 0.0, ptr %p
  ret void
}
attributes #0 = { alwaysinline nounwind "amdgpu-flat-work-group-size"="1,64" }
"#;

/// Kernel A of the read-after-write probes: unconditionally store 5.0.
const RAW_SET_IR: &str = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @k_set(ptr noalias %buf0) #0 {
  store float 5.0, ptr %buf0
  ret void
}
attributes #0 = { alwaysinline nounwind "amdgpu-flat-work-group-size"="1,64" }
"#;

/// Kernel B of the read-after-write probes: load, add 1.0, store back.
const RAW_INC_IR: &str = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @k_inc(ptr noalias %buf0) #0 {
  %v = load float, ptr %buf0
  %r = fadd float %v, 1.0
  store float %r, ptr %buf0
  ret void
}
attributes #0 = { alwaysinline nounwind "amdgpu-flat-work-group-size"="1,64" }
"#;

/// A kernel with no arguments at all, for the storage-accounting tests.
const EMPTY_IR: &str = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @empty() #0 {
entry:
  ret void
}
attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,1" }
"#;

fn global_f32_buffer_abi() -> [AbiParamDescriptor; 1] {
    [AbiParamDescriptor {
        slot: 0,
        kind: AbiParamKind::Storage(svod_dtype::AddrSpace::Global),
        dtype: svod_dtype::DType::Float32,
        name: None,
    }]
}

/// Compile amdgcn IR to a code object with host `clang`. Mirrors
/// `runtime::amd::compile` but avoids the dependency cycle; `None` when clang or
/// the AMDGPU target is unavailable.
fn clang_amdgcn(ir: &str, mcpu: &str) -> Option<Vec<u8>> {
    use std::io::Write;
    let child = std::process::Command::new("clang")
        .args(["-x", "ir", "-c", "-O2", "--target=amdgcn-amd-amdhsa"])
        .arg(format!("-mcpu={mcpu}"))
        .args(["-mcumode", "-nogpulib", "-nogpuinc", "-Wno-override-module", "-", "-o", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.as_ref()?.write_all(ir.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    (out.status.success() && out.stdout.starts_with(b"\x7fELF")).then_some(out.stdout)
}

struct MockProgram {
    iface: Arc<MockAmdIface>,
    device: Arc<AmdDevice>,
    allocator: AmdAllocator,
    program: AmdProgram,
}

/// Load `ir` onto a synthetic device. `graph` additionally installs a signal
/// pool and forces PM4 graph capture on, which capture is gated behind.
/// `None` when the host toolchain cannot build the code object.
fn mock_program(ir: &str, name: &str, abi: &[AbiParamDescriptor], graph: bool) -> Option<MockProgram> {
    Some(load_mock_program(&clang_amdgcn(ir, "gfx1100")?, name, abi, graph))
}

fn load_mock_program(bytes: &[u8], name: &str, abi: &[AbiParamDescriptor], graph: bool) -> MockProgram {
    let (iface, allocator) = mock_device(1);
    let device = Arc::clone(&allocator.dev);
    if graph {
        install_signal_pool(&allocator);
        device.core().set_pm4_graph(true);
    }
    let program = AmdProgram::load(Arc::clone(&device), &allocator, bytes, name, abi).expect("program load");
    MockProgram { iface, device, allocator, program }
}

fn graph_kernel(program: &AmdProgram, buffers: Vec<*mut u8>, deps: Vec<usize>) -> GraphKernel<'_> {
    GraphKernel { program, buffers, vals: Vec::new(), global_size: Some([1, 1, 1]), local_size: Some([1, 1, 1]), deps }
}

// -------------------------------------------------------------- ELF ingestion

#[test]
fn parse_kernel_descriptor_from_compiled_elf() {
    let Some(bytes) = clang_amdgcn(STORE_ZERO_IR, "gfx1100") else { return };
    let parsed = parse_kernel(&bytes, "store_zero").expect("parse");
    let kernarg_size = parsed.kd.kernarg_size;
    assert!(kernarg_size >= 8, "kernarg_size {kernarg_size} must hold at least one pointer");
    assert!((parsed.kd_offset as usize) < parsed.image.len(), "the descriptor must lie inside the image");
}

#[test]
fn program_code_allocation_is_balanced_on_success_and_failure() {
    let Some(mock) = mock_program(EMPTY_IR, "empty", &[], false) else { return };
    assert_eq!((mock.iface.allocation_count(), mock.iface.live_handle_count()), (1, 1));
    drop(mock.program);
    assert_eq!((mock.iface.free_count(), mock.iface.live_handle_count()), (1, 0));

    let bytes = clang_amdgcn(EMPTY_IR, "gfx1100").expect("clang succeeded above");
    let (iface, allocator) = mock_device(1);
    iface.script_alloc(Err(scripted_error("code allocation")));
    assert!(AmdProgram::load(Arc::clone(&allocator.dev), &allocator, &bytes, "empty", &[]).is_err());
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (0, 0, 0));
}

/// A kernel needing SGPR dispatch-pointer setup is rejected — after its code has
/// already been allocated, so the reclaim path runs.
#[test]
fn program_post_allocation_validation_reclaims_code() {
    const DISPATCH_PTR_IR: &str = r#"target triple = "amdgcn-amd-amdhsa"
declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
define amdgpu_kernel void @dispatch_ptr_program() #0 {
entry:
  %dispatch = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %value = load volatile i8, ptr addrspace(4) %dispatch, align 1
  ret void
}
attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,1" }
"#;
    let Some(bytes) = clang_amdgcn(DISPATCH_PTR_IR, "gfx1100") else { return };
    let (iface, allocator) = mock_device(1);
    assert!(matches!(
        AmdProgram::load(Arc::clone(&allocator.dev), &allocator, &bytes, "dispatch_ptr_program", &[]),
        Err(Error::Runtime { message }) if message.contains("ENABLE_SGPR_DISPATCH_PTR")
    ));
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (1, 1, 0));
}

// ---------------------------------------------------------- graph capture (mock)

#[test]
fn graph_capture_storage_unwinds_each_post_lane_allocation() {
    let Some(bytes) = clang_amdgcn(EMPTY_IR, "gfx1100") else { return };
    // Stages 0..6 belong to the lane itself (covered by the pool-queue unwind
    // test); 6..=8 are the graph's own kernarg and command-stream buffers.
    for fail_at in 6..=8 {
        let mock = load_mock_program(&bytes, "empty", &[], true);
        let (allocations, frees) = (mock.iface.allocation_count(), mock.iface.free_count());
        for _ in 0..fail_at {
            mock.iface.script_alloc(Ok(()));
        }
        mock.iface.script_alloc(Err(scripted_error("graph allocation")));
        let kernels = [graph_kernel(&mock.program, Vec::new(), Vec::new())];
        assert!(AmdGraph::capture(&mock.allocator, &kernels).is_err(), "fail_at={fail_at}");
        assert_eq!(mock.iface.allocation_count() - allocations, fail_at, "fail_at={fail_at}");
        assert_eq!(mock.iface.free_count() - frees, fail_at - 6, "fail_at={fail_at}");
        assert!(mock.iface.free_issues().is_empty());
    }
}

#[test]
fn graph_success_drop_frees_kernarg_and_both_resident_streams_once() {
    let Some(mock) = mock_program(EMPTY_IR, "empty", &[], true) else { return };
    let allocations = mock.iface.allocation_count();
    let kernels = [graph_kernel(&mock.program, Vec::new(), Vec::new())];
    let graph = AmdGraph::capture(&mock.allocator, &kernels).unwrap().expect("graph");
    assert_eq!(mock.iface.allocation_count() - allocations, 9);
    drop(graph);
    assert_eq!(mock.iface.free_count(), 3, "kernargs plus both resident streams");
    assert!(mock.iface.free_issues().is_empty());
}

#[test]
fn graph_replay_skips_repacking_unchanged_kernargs() {
    const STORE_IR: &str = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @store_arg(ptr addrspace(1) %out) #0 {
entry:
  store float 0.0, ptr addrspace(1) %out, align 4
  ret void
}
attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,1" }
"#;
    let Some(mock) = mock_program(STORE_IR, "store_arg", &global_f32_buffer_abi(), true) else { return };
    let kernels = [graph_kernel(&mock.program, vec![0x1000 as *mut u8], Vec::new())];
    let graph = AmdGraph::capture_amd(&mock.allocator, &kernels).unwrap().expect("graph");

    assert_eq!(graph.kernarg_pack_probe(&[0x2000], &[]).unwrap(), 1);
    assert_eq!(graph.kernarg_pack_probe(&[0x2000], &[]).unwrap(), 1, "identical arguments must not repack");
    assert_eq!(graph.kernarg_pack_probe(&[0x3000], &[]).unwrap(), 2, "changed arguments repack");
}

#[test]
fn graph_capture_emits_one_memory_barrier_for_the_whole_chain() {
    use crate::amd::sys::pm4;
    let Some(mock) = mock_program(EMPTY_IR, "empty", &[], true) else { return };
    let kernels = std::array::from_fn::<_, 3, _>(|_| graph_kernel(&mock.program, Vec::new(), Vec::new()));
    let graph = AmdGraph::capture_amd(&mock.allocator, &kernels).unwrap().expect("graph");
    let dwords = replay_dwords(graph.linked_bytes());
    let runs = |needle: &[u32]| dwords.windows(needle.len()).filter(|run| *run == needle).count();

    // One HDP flush + full acquire for the whole graph (tinygrad graph/hcq.py:157)...
    assert_eq!(runs(&pm4::hdp_flush()), 1);
    // ...while every dispatch keeps its own CS_PARTIAL_FLUSH.
    assert_eq!(runs(&pm4::event_write(pm4::CS_PARTIAL_FLUSH, pm4::EVENT_INDEX_PARTIAL_FLUSH)), 3);
}

#[test]
fn graph_failed_drain_quarantines_graph_program_and_queue_storage() {
    let Some(mock) = mock_program(EMPTY_IR, "empty", &[], true) else { return };
    let kernels = [graph_kernel(&mock.program, Vec::new(), Vec::new())];
    let graph = AmdGraph::capture(&mock.allocator, &kernels).unwrap().expect("graph");
    graph.replay(&[], &[]).expect("mock publication");
    mock.iface.script_wait(Err(Error::AmdIoctl { ioctl: "mock graph drain", errno: 5 }));
    let allocations = mock.iface.allocation_count();
    drop(graph);
    drop(kernels);
    drop(mock.program);
    assert!(mock.device.is_poisoned());
    assert_eq!(mock.iface.allocation_count(), allocations);
    assert_eq!((mock.iface.free_count(), mock.iface.live_handle_count()), (0, allocations));
}

/// A linked plan owns its kernargs transactionally: nothing is allocated when
/// capture fails, and a replay that fails before the doorbell leaves the device
/// usable while one that fails after it poisons and quarantines.
#[test]
fn linked_plan_capture_and_transactional_publication_own_storage() {
    let Some(mock) = mock_program(EMPTY_IR, "empty", &[], false) else { return };
    install_signal_pool(&mock.allocator);
    let pool =
        crate::amd::connector::PoolQueue::new_with_resources(Arc::clone(mock.device.core()), &mock.allocator).unwrap();
    let owner = crate::amd::connector::OwnerCtx::new(Arc::clone(mock.device.core()), mock.allocator.clone());
    let lane = crate::hcq::LaneSubmission {
        lane: crate::hcq::DeviceQueue {
            device: svod_dtype::DeviceSpec::Amd { device_id: 0 },
            queue: crate::hcq::QueueKind::Compute(0),
        },
        waits: Vec::new(),
        commands: vec![crate::hcq::TopologyCommand { operation: 0, copy_leg: None }],
        signal_value: 1,
    };
    let semantic = crate::hcq::SemanticLinkedPlan::from_lane_submissions(vec![lane], |_| [0x1000, 0x1008]).unwrap();
    let calls = [crate::device::PlanCall::Program {
        program: &mock.program,
        buffers: &[],
        vals: &[],
        global_size: Some([1, 1, 1]),
        local_size: Some([1, 1, 1]),
    }];

    let allocations = mock.iface.allocation_count();
    mock.iface.script_alloc(Err(scripted_error("linked kernarg allocation")));
    assert!(crate::amd::AmdLinkedPlan::capture(&owner, &pool, &semantic, &calls).is_err());
    assert_eq!(mock.iface.allocation_count(), allocations);

    let mut linked =
        crate::amd::AmdLinkedPlan::capture(&owner, &pool, &semantic, &calls).unwrap().expect("linked plan");
    assert_eq!(mock.iface.allocation_count(), allocations + 1);

    // Failing checkpoint `stage`: only a failure after the doorbell has actually
    // published, and only that poisons the device.
    for stage in 0..3 {
        for _ in 0..stage {
            mock.iface.script_publication(Ok(()));
        }
        mock.iface.script_publication(Err(scripted_error("linked publication")));
        let published = stage == 2;
        assert_eq!(linked.replay(&owner, &pool, &calls).unwrap_err().published, published, "stage={stage}");
        assert_eq!(mock.device.is_poisoned(), published, "stage={stage}");
    }
    drop(linked);
    assert_eq!(mock.iface.free_count(), 0, "linked kernargs and all device storage must be quarantined");
}

// ------------------------------------------------------------- hardware probes

/// Serializes the `#[ignore]` PM4-graph probes that toggle the per-device
/// `pm4_graph` flag. The flag lives on the process-global (`DEVICE_CACHE`-backed)
/// `AmdDeviceCore`, so two probes running concurrently would observe each other's
/// writes; holding this lock for each probe's duration makes the save/restore in
/// [`Pm4GraphOverride`] well-defined regardless of `--test-threads`.
static PM4_GRAPH_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Scoped enable of the per-device `pm4_graph` capture flag (capture is opt-in
/// by default — it regresses on gfx1151): records the previous value and
/// restores it on drop. Acquire [`PM4_GRAPH_TEST_LOCK`] first.
struct Pm4GraphOverride<'a> {
    core: &'a crate::amd::device::AmdDeviceCore,
    prev: bool,
}

impl<'a> Pm4GraphOverride<'a> {
    fn enable(core: &'a crate::amd::device::AmdDeviceCore) -> Self {
        let prev = core.pm4_graph();
        core.set_pm4_graph(true);
        Self { core, prev }
    }
}

impl Drop for Pm4GraphOverride<'_> {
    fn drop(&mut self) {
        self.core.set_pm4_graph(self.prev);
    }
}

/// A host-visible output buffer plus the two addresses probes need.
struct ProbeBuffer {
    raw: RawBuffer,
    gpu: u64,
    host: std::ptr::NonNull<u8>,
}

impl ProbeBuffer {
    fn new(alloc: &AmdAllocator) -> Self {
        let raw = alloc.alloc_uncached(64).expect("output buffer");
        let RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(host), .. } = &raw else {
            panic!("output buffer must be host-visible")
        };
        Self { gpu: *gpu_addr, host: *host, raw }
    }

    fn set(&self, value: f32) {
        // SAFETY: `host` is the host-visible mapping of a 64-byte allocation.
        unsafe { std::ptr::write_volatile(self.host.as_ptr() as *mut f32, value) };
    }

    fn get(&self) -> f32 {
        // SAFETY: as above.
        unsafe { std::ptr::read_volatile(self.host.as_ptr() as *const f32) }
    }
}

/// Capture a `count`-kernel chain of `store_zero` and replay it twice, each time
/// over a fresh sentinel that the kernel must overwrite. `profile` additionally
/// asks for a profiled replay and checks every kernel's GPU-clock span.
fn graph_capture_replay_probe(alloc: &AmdAllocator, mcpu: &str, count: usize, profile: bool) {
    let Some(bytes) = clang_amdgcn(STORE_ZERO_IR, mcpu) else { return };
    let program = AmdProgram::load(alloc.dev.clone(), alloc, &bytes, "store_zero", &global_f32_buffer_abi())
        .expect("load program");
    let output = ProbeBuffer::new(alloc);
    let kernels = (0..count)
        .map(|index| graph_kernel(&program, vec![output.gpu as *mut u8], Vec::from_iter(index.checked_sub(1))))
        .collect::<Vec<_>>();
    let Some(graph) = AmdGraph::capture(alloc, &kernels).expect("capture") else { return };

    for trial in 0..2 {
        let sentinel = -7.5 * (trial as f32 + 1.0);
        output.set(sentinel);
        graph.replay(&[], &[]).expect("graph replay");
        alloc.dev.core().synchronize_all().expect("synchronize_all");
        assert_eq!(output.get(), 0.0, "graph replay #{trial}: the kernel store must land ({sentinel} -> ?)");
    }
    if profile {
        let timestamps = graph.replay_profiled(&[], &[]).expect("profiled graph replay").expect("profile support");
        assert_eq!(timestamps.len(), count);
        for timestamp in timestamps {
            let (start, end) = timestamp.timestamps_ns().expect("graph GPU timestamps");
            assert!(end > start, "graph end ts ({end}) must exceed start ts ({start})");
        }
    }

    drop(graph); // synchronize + free kernargs before the output buffer drops.
    output.raw.free_amd_device_in_place();
}

/// Kernel A stores 5.0, kernel B (declaring A as a dependency) loads, adds 1.0
/// and stores back. A result of 5.0 means B ran before A's write was visible —
/// the hazard barrier between chained dispatches is missing.
fn graph_two_kernel_raw_probe(alloc: &AmdAllocator, mcpu: &str) {
    let (Some(set), Some(inc)) = (clang_amdgcn(RAW_SET_IR, mcpu), clang_amdgcn(RAW_INC_IR, mcpu)) else { return };
    let abi = global_f32_buffer_abi();
    let set = AmdProgram::load(alloc.dev.clone(), alloc, &set, "k_set", &abi).expect("load k_set");
    let inc = AmdProgram::load(alloc.dev.clone(), alloc, &inc, "k_inc", &abi).expect("load k_inc");
    let output = ProbeBuffer::new(alloc);
    output.set(-99.0);

    let kernels = vec![
        graph_kernel(&set, vec![output.gpu as *mut u8], vec![]),
        graph_kernel(&inc, vec![output.gpu as *mut u8], vec![0]),
    ];
    let Some(graph) = AmdGraph::capture(alloc, &kernels).expect("capture") else { return };
    graph.replay(&[], &[]).expect("replay");
    alloc.dev.core().synchronize_all().expect("sync");
    assert_eq!(output.get(), 6.0, "kernel B must observe kernel A's write inside the captured chain");

    drop(graph);
    output.raw.free_amd_device_in_place();
}

/// Run: `cargo test -p svod-device --lib aql_graph -- --ignored --nocapture --test-threads=1`
#[test]
#[ignore = "manual hardware probe; needs a real gfx942 AMD GPU + clang"]
fn aql_graph_capture_replay() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    if !require_multi_xcc(&alloc) {
        return;
    }
    ensure_hw_signal_pool(&alloc);
    graph_capture_replay_probe(&alloc, "gfx942", 1, false);
}

#[test]
#[ignore = "manual hardware probe; needs a real gfx942 AMD GPU + clang"]
fn aql_graph_two_kernel_raw_dependency() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    if !require_multi_xcc(&alloc) {
        return;
    }
    ensure_hw_signal_pool(&alloc);
    graph_two_kernel_raw_probe(&alloc, "gfx942");
}

/// Single-XCC (RDNA) analogue: a 12-kernel chain captured into one resident PM4
/// indirect buffer, replayed normally twice and once profiled.
#[test]
#[ignore = "manual hardware probe; needs a real single-XCC AMD GPU + clang"]
fn pm4_graph_capture_replay() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    if !require_single_xcc(&alloc) {
        return;
    }
    let _serial = PM4_GRAPH_TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let _pm4 = Pm4GraphOverride::enable(alloc.dev.core());
    ensure_hw_signal_pool(&alloc);
    graph_capture_replay_probe(&alloc, alloc.dev.arch.mcpu(), 12, true);
}

#[test]
#[ignore = "manual hardware probe; needs a real single-XCC AMD GPU + clang"]
fn pm4_graph_two_kernel_raw_dependency() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    if !require_single_xcc(&alloc) {
        return;
    }
    let _serial = PM4_GRAPH_TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let _pm4 = Pm4GraphOverride::enable(alloc.dev.core());
    ensure_hw_signal_pool(&alloc);
    graph_two_kernel_raw_probe(&alloc, alloc.dev.arch.mcpu());
}

/// Dispatch one kernel through the per-plan `PlanContext` with `profile=true`,
/// drain, and read back the GPU-clock span the two `release_mem_timestamp`
/// probes wrote — the single-XCC PM4 path, which has no AQL `ENABLE_PROFILING`
/// auto-stamp.
///
/// Run: `SVOD_DEVICE=AMD:0 cargo test -p svod-device --lib pm4_dispatch_timestamp -- --ignored --test-threads=1`
#[test]
#[ignore = "manual hardware probe; needs a real single-XCC AMD GPU + clang"]
fn pm4_dispatch_timestamp_probe() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    if !require_single_xcc(&alloc) {
        return;
    }
    ensure_hw_signal_pool(&alloc);
    let Some(bytes) = clang_amdgcn(STORE_ZERO_IR, alloc.dev.arch.mcpu()) else { return };
    let program = AmdProgram::load(alloc.dev.clone(), &alloc, &bytes, "store_zero", &global_f32_buffer_abi())
        .expect("load program");
    let output = ProbeBuffer::new(&alloc);

    let ctx = program.new_exec_context().expect("exec context").expect("AMD yields a plan context");
    // profile=true: we hold `handle` across `synchronize`, so arming the probes
    // is safe (this is exactly the invariant the `profile` flag enforces).
    let handle = unsafe {
        ctx.dispatch(&program as &dyn Program, &[output.gpu as *mut u8], &[], Some([1, 1, 1]), Some([1, 1, 1]), true)
    }
    .expect("dispatch");
    ctx.synchronize().expect("synchronize");

    let (start, end) =
        handle.expect("a profiled dispatch yields a timestamp handle").timestamps_ns().expect("gpu-clock timestamps");
    assert!(end > start, "end ts ({end}) must exceed start ts ({start})");
    assert!(end - start < 1_000_000_000, "a trivial kernel should run in < 1s, got {} ns", end - start);

    output.raw.free_amd_device_in_place();
}

/// `Program::execute_timed` on AMD returns the probes' GPU-clock span, not the
/// caller's wall clock — the AMD twin of CUDA's `timed_execution_reports_gpu_duration`.
/// The span must be shorter than the wall time around it, because the wall time
/// also holds the submit path and the drain that the stamps exist to exclude.
///
/// Run: `SVOD_DEVICE=AMD:0 cargo test -p svod-device --lib execute_timed_reports -- --ignored --test-threads=1`
#[test]
#[ignore = "manual hardware probe; needs a real AMD GPU + clang"]
fn execute_timed_reports_the_gpu_span() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    ensure_hw_signal_pool(&alloc);
    let Some(bytes) = clang_amdgcn(STORE_ZERO_IR, alloc.dev.arch.mcpu()) else { return };
    let program = AmdProgram::load(alloc.dev.clone(), &alloc, &bytes, "store_zero", &global_f32_buffer_abi())
        .expect("load program");
    let output = ProbeBuffer::new(&alloc);

    let wall = std::time::Instant::now();
    let gpu = unsafe { program.execute_timed(&[output.gpu as *mut u8], &[], Some([64, 1, 1]), Some([64, 1, 1])) }
        .expect("timed dispatch")
        .expect("a profiled AMD dispatch stamps its own span");
    let wall = wall.elapsed();

    assert!(gpu.as_nanos() > 0, "the 100 MHz clock must advance across a dispatch");
    assert!(gpu < wall, "gpu {gpu:?} must exclude the submit path the wall clock {wall:?} holds");

    output.raw.free_amd_device_in_place();
}

/// Forced-AQL timeline stress: 2000 asynchronous, 128 synchronous and 128
/// profiled dispatches through one plan context. Multi-XCC hardware uses AQL by
/// default; single-XCC hardware must set `SVOD_AMD_AQL=1`.
///
/// Run: `SVOD_DEVICE=AMD:0 SVOD_AMD_AQL=1 cargo test -p svod-device --lib aql_timeline_stress -- --ignored --test-threads=1`
#[test]
#[ignore = "manual hardware probe; needs a real AMD GPU + clang and an AQL queue"]
fn aql_timeline_stress_probe() {
    const STORE_ONE_IR: &str = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @aql_stress(ptr noalias %buf0) #0 {
  store float 1.0, ptr %buf0
  ret void
}
attributes #0 = { alwaysinline nounwind "amdgpu-flat-work-group-size"="1,64" }
"#;
    let Some(alloc) = amd_alloc_or_skip() else { return };
    if crate::amd::queue::AmdComputeQueue::will_use_pm4(alloc.dev.core()) {
        return; // set SVOD_AMD_AQL=1 on single-XCC hardware
    }
    ensure_hw_signal_pool(&alloc);
    let Some(bytes) = clang_amdgcn(STORE_ONE_IR, alloc.dev.arch.mcpu()) else { return };
    let program =
        AmdProgram::load(alloc.dev.clone(), &alloc, &bytes, "aql_stress", &global_f32_buffer_abi()).expect("load");
    let output = ProbeBuffer::new(&alloc);
    let context = program.new_exec_context().expect("context").expect("AMD context");
    let dispatch = |profile| unsafe {
        context.dispatch(&program, &[output.gpu as *mut u8], &[], Some([1, 1, 1]), Some([1, 1, 1]), profile)
    };

    for _ in 0..2_000 {
        dispatch(false).expect("asynchronous AQL dispatch");
    }
    context.synchronize().expect("asynchronous AQL drain");

    for _ in 0..128 {
        dispatch(false).expect("synchronous AQL dispatch");
        context.synchronize().expect("synchronous AQL drain");
    }

    for _ in 0..4 {
        let timestamps = (0..32)
            .map(|_| dispatch(true).expect("profiled AQL dispatch").expect("timestamp handle"))
            .collect::<Vec<_>>();
        context.synchronize().expect("profiled AQL drain");
        for timestamp in timestamps {
            let (start, end) = timestamp.timestamps_ns().expect("AQL PM4 timestamps");
            assert!(end > start, "end ts ({end}) must exceed start ts ({start})");
        }
    }

    assert_eq!(output.get(), 1.0);
    output.raw.free_amd_device_in_place();
}
