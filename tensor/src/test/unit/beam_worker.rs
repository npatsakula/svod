#[cfg(unix)]
use super::started::{BusyTask, SlotOutcome, accept_worker, poll_slot, worker_listener};
use super::*;

#[cfg(unix)]
fn response(index: usize) -> WorkerResponse {
    WorkerResponse { index, result: None, error: None }
}

#[cfg(unix)]
fn overdue(timeout: Duration) -> BusyTask {
    BusyTask { index: 0, started: Instant::now() - timeout }
}

#[cfg(unix)]
/// A response written exactly on the deadline is already in the channel,
/// so it must be delivered instead of dropping the candidate and SIGKILLing a
/// healthy helper.
#[test]
fn a_response_on_the_deadline_is_delivered_not_timed_out() {
    let timeout = Duration::from_millis(50);
    let (send, responses) = mpsc::channel();
    send.send(Ok(response(7))).unwrap();

    let outcome = poll_slot(&responses, Some(&overdue(timeout)), timeout);
    assert!(matches!(outcome, SlotOutcome::Response(response) if response.index == 7), "on-deadline response dropped");
}

#[cfg(unix)]
#[test]
fn an_empty_channel_past_the_deadline_times_out() {
    let timeout = Duration::from_millis(50);
    let (_send, responses) = mpsc::channel::<std::io::Result<WorkerResponse>>();
    assert!(matches!(poll_slot(&responses, Some(&overdue(timeout)), timeout), SlotOutcome::TimedOut));
    assert!(matches!(poll_slot(&responses, Some(&overdue(timeout)), Duration::ZERO), SlotOutcome::Idle));
    assert!(matches!(poll_slot(&responses, None, timeout), SlotOutcome::Idle), "an idle worker never times out");
}

#[cfg(unix)]
#[test]
fn a_closed_worker_fails_the_slot_regardless_of_the_deadline() {
    let timeout = Duration::from_millis(50);
    let (send, responses) = mpsc::channel();
    send.send(Err(std::io::Error::other("stdout closed"))).unwrap();
    assert!(matches!(poll_slot(&responses, Some(&overdue(timeout)), timeout), SlotOutcome::Failed(Some(_))));

    let (send, responses) = mpsc::channel::<std::io::Result<WorkerResponse>>();
    drop(send);
    assert!(matches!(poll_slot(&responses, Some(&overdue(timeout)), timeout), SlotOutcome::Failed(None)));
}

/// The worker constructor survives linking into this test binary. Every
/// started worker depends on it: without it the pool cannot start this
/// program again, for it would run `main` in the worker's place.
#[cfg(unix)]
#[test]
fn the_worker_constructor_runs_in_this_program() {
    assert!(worker_entry_armed(), "the BEAM worker constructor was not run");
}

#[cfg(not(unix))]
#[test]
fn a_program_without_the_constructor_is_not_armed() {
    assert!(!worker_entry_armed());
}

#[cfg(unix)]
/// The handshake never waits past its deadline: the program started could be
/// one whose other constructors hang.
#[test]
fn the_handshake_gives_up_at_its_deadline() {
    let bound = worker_listener().expect("a worker socket");
    let started = Instant::now();
    let error = accept_worker(&bound.listener, started + Duration::from_millis(80)).expect_err("nothing connects");
    assert_eq!(error.kind(), std::io::ErrorKind::TimedOut, "{error}");
    assert!(started.elapsed() >= Duration::from_millis(80), "gave up early");
}

#[cfg(unix)]
/// Every slot's socket file goes with its listener, whether or not a worker
/// ever connected.
#[test]
fn a_slots_socket_file_is_gone_once_it_stops_listening() {
    let bound = worker_listener().expect("a worker socket");
    let path = bound.path.clone();
    assert!(path.exists());
    drop(bound);
    assert!(!path.exists(), "the socket file outlived its listener");
}

/// A CPU worker initialization for the first backend this host can compile.
#[cfg(unix)]
fn cpu_worker_init() -> Option<WorkerInit> {
    let mut found = None;
    for backend in [svod_runtime::CpuBackend::Clang, svod_runtime::CpuBackend::Llvm] {
        let Ok((renderer, compiler)) = svod_runtime::create_cpu_codegen(backend) else { continue };
        let profile = svod_schedule::OptimizerRenderer::cpu().with_codegen_renderer(renderer.as_ref());
        found = Some(WorkerInit {
            protocol_version: BEAM_WORKER_PROTOCOL_VERSION,
            graph: svod_ir::OptimizerWireGraph::from_root(&svod_ir::UOp::sink(vec![])).ok()?,
            device: DeviceSpec::Cpu,
            gpu_arch: None,
            compiler_key: compiler.cache_key().to_string(),
            renderer_fingerprint: profile.cache_fingerprint(),
            base_opt_count: 0,
            beam: BeamConfig { compile_timeout_secs: 30, ..BeamConfig::default() },
            transcendental: 0,
            disable_fast_idiv: false,
            log_surpass: false,
        });
        break;
    }
    if found.is_none() {
        eprintln!("skipped: no CPU compiler toolchain");
    }
    found
}

/// Started copies of this very test binary serve the pool: one indexed
/// response per candidate. Through the started backend by name, so a pool
/// that quietly fell back to compiling in process cannot pass this.
#[cfg(unix)]
#[test]
fn started_workers_answer_every_candidate() {
    if !worker_entry_armed() {
        eprintln!("skipped: this program cannot start itself as a worker");
        return;
    }
    let Some(init) = cpu_worker_init() else { return };
    let mut pool = WorkerPool::started(2, init, Duration::from_secs(30)).expect("start this program as workers");
    let candidates: Vec<Vec<svod_schedule::optimizer::Opt>> = vec![vec![], vec![], vec![], vec![], vec![]];
    let mut answered = Vec::new();
    pool.run(&candidates, |response| answered.push(response.index)).expect("run the wave");
    drop(pool);
    answered.sort_unstable();
    assert_eq!(answered, (0..candidates.len()).collect::<Vec<_>>());
}

/// A worker that fails initialization is reported, not left running.
#[cfg(unix)]
#[test]
fn a_worker_that_cannot_reproduce_the_compiler_is_unavailable() {
    let Some(mut init) = cpu_worker_init() else { return };
    init.compiler_key = "no-such-compiler".into();
    let Err(error) = WorkerPool::new(1, init) else { panic!("no backend has this identity") };
    assert!(error.to_string().contains("initialization failed"), "{error}");
}

/// The in-process backend answers the same wave, on the calling thread.
#[cfg(unix)]
#[test]
fn the_in_process_backend_answers_every_candidate() {
    let Some(init) = cpu_worker_init() else { return };
    let mut pool = WorkerPool::in_process(init, Duration::from_secs(30)).expect("set up in-process compilation");
    let candidates: Vec<Vec<svod_schedule::optimizer::Opt>> = vec![vec![], vec![], vec![]];
    let mut answered = Vec::new();
    pool.run(&candidates, |response| answered.push(response.index)).expect("run the wave");
    assert_eq!(answered, vec![0, 1, 2], "in order, on the calling thread");
}

/// Pool failures are typed, so a caller can tell an out-of-order worker
/// from an unavailable one instead of matching on a rendered string.
#[test]
fn worker_misorder_is_a_distinguishable_variant() {
    let misorder = BeamWorker::WorkerMisorder { got: 3, expected: Some(1) };
    assert!(matches!(misorder, BeamWorker::WorkerMisorder { got: 3, expected: Some(1) }));
    assert!(misorder.to_string().contains("expected Some(1)"), "{misorder}");
}

fn cuda_worker_init(gpu_arch: Option<GpuArch>, compiler_key: &str) -> WorkerInit {
    WorkerInit {
        protocol_version: BEAM_WORKER_PROTOCOL_VERSION,
        graph: svod_ir::OptimizerWireGraph::from_root(&svod_ir::UOp::sink(vec![])).unwrap(),
        device: DeviceSpec::Cuda { device_id: 0 },
        gpu_arch,
        compiler_key: compiler_key.to_string(),
        renderer_fingerprint: 0,
        base_opt_count: 0,
        beam: BeamConfig::default(),
        transcendental: 0,
        disable_fast_idiv: false,
        log_surpass: false,
    }
}

/// The CUDA arm needs the compute capability on the wire: without it the
/// helper reports itself unavailable instead of guessing an arch.
#[test]
fn cuda_worker_without_an_arch_is_unavailable() {
    let Err(error) = worker_codegen(&cuda_worker_init(None, "nvptx-clang:none")) else { panic!("no arch on the wire") };
    assert!(
        matches!(&error, BeamWorker::WorkerUnavailable { reason } if reason.contains("target architecture")),
        "{error}"
    );
}

/// A clean worker rebuilds the parent's compiler from the wire arch alone, so
/// it fills the parent's object-cache slot and optimizes with the parent's
/// profile.
#[test]
fn cuda_worker_codegen_reproduces_the_parent_identity() {
    let Some(arch) = crate::config::cuda_test_arch() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let (parent_renderer, parent) = svod_runtime::create_cuda_codegen(0, arch).unwrap();
    let codegen = worker_codegen(&cuda_worker_init(Some(GpuArch::Cuda(arch)), parent.cache_key())).unwrap();
    assert_eq!(codegen.compiler.cache_key(), parent.cache_key());
    let parent_profile =
        svod_schedule::OptimizerRenderer::for_cuda_arch(arch).with_codegen_renderer(parent_renderer.as_ref());
    assert_eq!(codegen.optimizer_renderer.cache_fingerprint(), parent_profile.cache_fingerprint());
}
