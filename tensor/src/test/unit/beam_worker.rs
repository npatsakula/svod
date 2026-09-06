use super::*;

fn response(index: usize) -> WorkerResponse {
    WorkerResponse { index, result: None, error: None }
}

fn overdue(timeout: Duration) -> BusyTask {
    BusyTask { index: 0, started: Instant::now() - timeout }
}

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

#[test]
fn an_empty_channel_past_the_deadline_times_out() {
    let timeout = Duration::from_millis(50);
    let (_send, responses) = mpsc::channel::<std::io::Result<WorkerResponse>>();
    assert!(matches!(poll_slot(&responses, Some(&overdue(timeout)), timeout), SlotOutcome::TimedOut));
    assert!(matches!(poll_slot(&responses, Some(&overdue(timeout)), Duration::ZERO), SlotOutcome::Idle));
    assert!(matches!(poll_slot(&responses, None, timeout), SlotOutcome::Idle), "an idle worker never times out");
}

#[test]
fn a_closed_helper_fails_the_slot_regardless_of_the_deadline() {
    let timeout = Duration::from_millis(50);
    let (send, responses) = mpsc::channel();
    send.send(Err(std::io::Error::other("stdout closed"))).unwrap();
    assert!(matches!(poll_slot(&responses, Some(&overdue(timeout)), timeout), SlotOutcome::Failed(Some(_))));

    let (send, responses) = mpsc::channel::<std::io::Result<WorkerResponse>>();
    drop(send);
    assert!(matches!(poll_slot(&responses, Some(&overdue(timeout)), timeout), SlotOutcome::Failed(None)));
}

/// A resolution failure must not be latched — the next attempt has to run
/// the resolver again, and only a success is remembered.
#[test]
fn helper_resolution_failures_are_not_cached() {
    let cache = std::sync::Mutex::new(None);
    let attempts = std::cell::Cell::new(0usize);
    let resolve = || {
        attempts.set(attempts.get() + 1);
        match attempts.get() {
            1 => Err(BeamWorker::HelperUnavailable { reason: "not built yet".into() }),
            _ => Ok(std::path::PathBuf::from("/helper/svod-beam-worker")),
        }
    };

    assert!(cached_helper_path(&cache, resolve).is_err(), "first resolution must fail");
    let resolved = cached_helper_path(&cache, resolve).expect("a failure must not be latched");
    assert_eq!(resolved, std::path::PathBuf::from("/helper/svod-beam-worker"));
    assert_eq!(cached_helper_path(&cache, resolve).unwrap(), resolved);
    assert_eq!(attempts.get(), 2, "a cached success must not re-resolve");
}

/// A `SVOD_BEAM_WORKER` that is not a file is reported, not silently ignored.
#[test]
fn a_non_file_helper_override_is_rejected() {
    let helper = tempfile::NamedTempFile::new().expect("helper stand-in");
    let missing = helper.path().with_extension("absent");
    unsafe { std::env::set_var("SVOD_BEAM_WORKER", &missing) };
    let failure = resolve_helper_path().expect_err("a missing helper must not resolve");
    unsafe { std::env::remove_var("SVOD_BEAM_WORKER") };
    assert!(
        matches!(&failure, BeamWorker::HelperUnavailable { reason } if reason.contains("is not a file")),
        "{failure}"
    );
}

/// The helper path comes from cargo's own artifact report, not a guessed
/// `target/<profile>/` layout.
#[test]
fn last_executable_takes_cargos_final_artifact() {
    let messages = concat!(
        r#"{"reason":"compiler-artifact","target":{"name":"svod-tensor"},"executable":null}"#,
        "\n",
        r#"{"reason":"compiler-artifact","executable":"/custom/target/dir/debug/svod-beam-worker"}"#,
        "\n",
        r#"{"reason":"build-finished","success":true}"#,
        "\n",
    );
    assert_eq!(
        last_executable(messages.as_bytes()),
        Some(std::path::PathBuf::from("/custom/target/dir/debug/svod-beam-worker"))
    );
    assert_eq!(last_executable(b"not json\n"), None);
}

/// Pool failures are typed, so a caller can tell an out-of-order helper
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
        matches!(&error, BeamWorker::HelperUnavailable { reason } if reason.contains("target architecture")),
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
