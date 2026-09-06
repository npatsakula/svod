//! Clean spawned worker protocol for BEAM candidate compilation.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
use std::thread::JoinHandle;
use std::time::Duration;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use svod_device::device::AbiParamDescriptor;
use svod_dtype::{DeviceSpec, GpuArch};
use svod_ir::{BinaryStageIdentity, Op, OptimizerWireGraph};
use svod_schedule::optimizer::{BeamConfig, Opt};

use crate::error::BeamWorker;
use svod_ir::ops;

type Result<T> = std::result::Result<T, BeamWorker>;

pub const BEAM_WORKER_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInit {
    pub protocol_version: u32,
    pub graph: OptimizerWireGraph,
    pub device: DeviceSpec,
    pub gpu_arch: Option<GpuArch>,
    pub compiler_key: String,
    pub renderer_fingerprint: u64,
    pub base_opt_count: usize,
    pub beam: BeamConfig,
    pub transcendental: i32,
    pub disable_fast_idiv: bool,
    pub log_surpass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerJob {
    pub index: usize,
    pub opts: Vec<Opt>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerArtifact {
    pub source: String,
    pub bytes: Vec<u8>,
    pub identity: BinaryStageIdentity,
    pub name: String,
    pub abi: Vec<AbiParamDescriptor>,
    pub global_size: [usize; 3],
    pub local_size: Option<[usize; 3]>,
    pub vals: Vec<i64>,
    pub compute_ops: u64,
    pub preparation_ns: u64,
    pub compilation_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerResponse {
    pub index: usize,
    pub result: Option<WorkerArtifact>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerReady {
    error: Option<String>,
}

pub fn write_frame<T: Serialize>(writer: &mut impl Write, value: &T) -> std::io::Result<()> {
    let bytes = bincode::serialize(value).map_err(std::io::Error::other)?;
    writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
    writer.write_all(&bytes)?;
    writer.flush()
}

pub fn read_frame<T: serde::de::DeserializeOwned>(reader: &mut impl Read) -> std::io::Result<Option<T>> {
    let mut length = [0u8; 8];
    match reader.read_exact(&mut length) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error),
    }
    let length = usize::try_from(u64::from_le_bytes(length))
        .map_err(|_| std::io::Error::other("BEAM frame length does not fit usize"))?;
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes)?;
    bincode::deserialize(&bytes).map(Some).map_err(std::io::Error::other)
}

struct WorkerCodegen {
    renderer: Arc<dyn svod_device::device::Renderer>,
    compiler: Arc<dyn svod_device::device::Compiler>,
    optimizer_renderer: svod_schedule::OptimizerRenderer,
}

fn worker_codegen(init: &WorkerInit) -> Result<WorkerCodegen> {
    let (renderer, compiler): (Arc<dyn svod_device::device::Renderer>, Arc<dyn svod_device::device::Compiler>) =
        match init.device {
            DeviceSpec::Cpu => {
                let mut found = None;
                for backend in [svod_runtime::CpuBackend::Clang, svod_runtime::CpuBackend::Llvm] {
                    let candidate = svod_runtime::create_cpu_codegen(backend).map_err(BeamWorker::at("CPU codegen"))?;
                    if candidate.1.cache_key() == init.compiler_key {
                        found = Some(candidate);
                        break;
                    }
                }
                found.ok_or_else(|| BeamWorker::HelperUnavailable {
                    reason: format!("no CPU backend matches compiler identity {}", init.compiler_key),
                })?
            }
            DeviceSpec::Amd { device_id } => {
                let arch = init.gpu_arch.and_then(GpuArch::amd).ok_or_else(|| BeamWorker::HelperUnavailable {
                    reason: "AMD BEAM worker initialization has no target architecture".into(),
                })?;
                svod_runtime::create_amd_codegen(device_id, arch).map_err(BeamWorker::at("AMD codegen"))?
            }
            DeviceSpec::Metal { device_id } => {
                svod_runtime::create_metal_codegen(device_id).map_err(BeamWorker::at("Metal codegen"))?
            }
            DeviceSpec::Cuda { device_id } => {
                let arch = init.gpu_arch.and_then(GpuArch::cuda).ok_or_else(|| BeamWorker::HelperUnavailable {
                    reason: "CUDA BEAM worker initialization has no target architecture".into(),
                })?;
                svod_runtime::create_cuda_codegen(device_id, arch).map_err(BeamWorker::at("CUDA codegen"))?
            }
            _ => {
                return Err(BeamWorker::HelperUnavailable {
                    reason: format!("{:?} has no device-disabled BEAM codegen factory", init.device),
                });
            }
        };
    if compiler.cache_key() != init.compiler_key {
        return Err(BeamWorker::HelperUnavailable {
            reason: format!(
                "compiler identity mismatch: parent={}, worker={}",
                init.compiler_key,
                compiler.cache_key()
            ),
        });
    }
    let optimizer_renderer = match init.device {
        DeviceSpec::Cpu => svod_schedule::OptimizerRenderer::cpu(),
        DeviceSpec::Amd { .. } => {
            init.gpu_arch.and_then(GpuArch::amd).map(svod_schedule::OptimizerRenderer::for_amd_arch).ok_or_else(
                || BeamWorker::HelperUnavailable {
                    reason: "AMD BEAM worker initialization has no optimizer profile".into(),
                },
            )?
        }
        DeviceSpec::Metal { .. } => init
            .gpu_arch
            .and_then(GpuArch::metal)
            .map(svod_schedule::OptimizerRenderer::for_metal_family)
            .unwrap_or_else(svod_schedule::OptimizerRenderer::metal),
        DeviceSpec::Cuda { .. } => {
            init.gpu_arch.and_then(GpuArch::cuda).map(svod_schedule::OptimizerRenderer::for_cuda_arch).ok_or_else(
                || BeamWorker::HelperUnavailable {
                    reason: "CUDA BEAM worker initialization has no optimizer profile".into(),
                },
            )?
        }
        _ => {
            return Err(BeamWorker::HelperUnavailable {
                reason: format!("{:?} has no BEAM optimizer profile", init.device),
            });
        }
    }
    .with_codegen_renderer(renderer.as_ref());
    Ok(WorkerCodegen { renderer, compiler, optimizer_renderer })
}

fn try_compile(
    init: &WorkerInit,
    base_ast: &Arc<svod_ir::UOp>,
    codegen: &WorkerCodegen,
    job: &WorkerJob,
) -> Result<Option<WorkerArtifact>> {
    let started = Instant::now();
    let scheduler = svod_schedule::optimizer::prepare_scheduler(base_ast.clone(), &codegen.optimizer_renderer)
        .map_err(BeamWorker::at("prepare scheduler"))?;
    if scheduler.applied_opts.len() != init.base_opt_count {
        return Err(BeamWorker::CompileStage {
            stage: "prepare scheduler",
            reason: format!(
                "base opt count mismatch: parent={}, worker={}",
                init.base_opt_count,
                scheduler.applied_opts.len()
            ),
        });
    }
    let Some(candidate) =
        svod_schedule::optimizer::apply_remote_candidate(scheduler, init.base_opt_count, &job.opts, &init.beam)
    else {
        return Ok(None);
    };
    let raw_ast = candidate.get_optimized_ast(None);
    let post = svod_schedule::OptimizerConfig {
        beam: init.beam.clone(),
        transcendental: init.transcendental,
        disable_fast_idiv: init.disable_fast_idiv,
        ..Default::default()
    };
    let optimized = svod_schedule::apply_post_optimization_with_config(raw_ast, &codegen.optimizer_renderer, &post)
        .map_err(BeamWorker::at("post optimization"))?;
    let compute_ops = svod_schedule::compute_ops_estimate(&optimized);
    let program = svod_codegen::program_pipeline::program_from_sink_with_renderer(optimized, codegen.renderer.as_ref())
        .map_err(BeamWorker::at("construct PROGRAM"))?;
    let program = svod_codegen::program_pipeline::get_program(
        &program,
        codegen.renderer.as_ref(),
        codegen.compiler.as_ref(),
        svod_codegen::program_pipeline::ProgramTarget::Source,
    )
    .map_err(BeamWorker::at("linearize/render"))?;
    let linear_uops = match program.op() {
        Op::Program(ops::Program { linear: Some(linear), .. }) => match linear.op() {
            Op::Linear(ops::Linear { ops }) => ops.len(),
            other => return Err(BeamWorker::CompileStage { stage: "linearize/render", reason: format!("{other:?}") }),
        },
        other => return Err(BeamWorker::CompileStage { stage: "linearize/render", reason: format!("{other:?}") }),
    };
    if init.beam.max_uops > 0 && linear_uops >= init.beam.max_uops {
        if init.log_surpass {
            eprintln!("[BEAM drop] too_many_uops: linear={linear_uops} max={}", init.beam.max_uops);
        }
        return Ok(None);
    }
    let spec = svod_device::device::ProgramSpec::from_uop(&program).map_err(BeamWorker::at("PROGRAM specification"))?;
    let mut values = std::collections::HashMap::new();
    for variable in &spec.vars {
        if variable.name != "core_id" {
            values.insert(variable.name.as_str(), (variable.min + variable.max) / 2);
        }
    }
    let launch = spec.launch_dims(&values).map_err(BeamWorker::at("launch dimensions"))?;
    let vals = spec.var_names.iter().map(|name| values.get(name.as_str()).copied().unwrap_or(0)).collect();
    let preparation_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;

    let compiled_started = Instant::now();
    let (compiled_program, compiled) = svod_codegen::program_pipeline::do_compile(&program, codegen.compiler.as_ref())
        .map_err(BeamWorker::at("compile"))?;
    let compilation_ns = compiled_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    let identity = match compiled_program.op() {
        Op::Program(ops::Program { source: Some(source), binary: Some(binary), .. }) => {
            let source_identity = match source.op() {
                Op::Source(ops::Source { identity: Some(identity), .. }) => identity,
                other => return Err(BeamWorker::CompileStage { stage: "compile", reason: format!("{other:?}") }),
            };
            match binary.op() {
                Op::ProgramBinary(ops::ProgramBinary { identity: Some(identity), .. })
                    if identity.source == **source_identity =>
                {
                    identity.as_ref().clone()
                }
                other => return Err(BeamWorker::CompileStage { stage: "compile", reason: format!("{other:?}") }),
            }
        }
        other => return Err(BeamWorker::CompileStage { stage: "compile", reason: format!("{other:?}") }),
    };
    Ok(Some(WorkerArtifact {
        source: spec.src,
        bytes: compiled.bytes,
        identity,
        name: spec.name,
        abi: spec.abi,
        global_size: launch.global_size,
        local_size: launch.local_size,
        vals,
        compute_ops,
        preparation_ns,
        compilation_ns,
    }))
}

pub fn worker_main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    let init: WorkerInit = read_frame(&mut input)
        .map_err(|source| BeamWorker::Frame { source, what: "read worker initialization" })?
        .ok_or_else(|| BeamWorker::HelperUnavailable { reason: "worker stdin closed before initialization".into() })?;
    let setup = (|| {
        if init.protocol_version != BEAM_WORKER_PROTOCOL_VERSION {
            return Err(BeamWorker::ProtocolMismatch {
                expected: init.protocol_version,
                actual: BEAM_WORKER_PROTOCOL_VERSION,
            });
        }
        let base_ast = init.graph.decode_root().map_err(BeamWorker::at("decode graph"))?;
        let codegen = worker_codegen(&init)?;
        if codegen.optimizer_renderer.cache_fingerprint() != init.renderer_fingerprint {
            return Err(BeamWorker::HelperUnavailable {
                reason: format!(
                    "renderer identity mismatch: parent={}, worker={}",
                    init.renderer_fingerprint,
                    codegen.optimizer_renderer.cache_fingerprint()
                ),
            });
        }
        Ok((base_ast, codegen))
    })();
    let (base_ast, codegen) = match setup {
        Ok(setup) => {
            write_frame(&mut output, &WorkerReady { error: None })
                .map_err(|source| BeamWorker::Frame { source, what: "write worker readiness" })?;
            setup
        }
        Err(error) => {
            let _ = write_frame(&mut output, &WorkerReady { error: Some(error.to_string()) });
            return Err(error.into());
        }
    };
    while let Some(job) =
        read_frame::<WorkerJob>(&mut input).map_err(|source| BeamWorker::Frame { source, what: "read worker job" })?
    {
        let watchdog = (init.beam.compile_timeout_secs > 0).then(|| {
            let (cancel, wait) = mpsc::channel();
            let timeout = Duration::from_secs(init.beam.compile_timeout_secs);
            let thread = std::thread::spawn(move || {
                if wait.recv_timeout(timeout).is_err() {
                    #[cfg(unix)]
                    unsafe {
                        // The helper is its own process group. This terminates
                        // both Rust compilation work and any clang descendant.
                        libc::kill(0, libc::SIGKILL);
                    }
                    #[cfg(not(unix))]
                    std::process::abort();
                }
            });
            (cancel, thread)
        });
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| try_compile(&init, &base_ast, &codegen, &job)));
        if let Some((cancel, thread)) = watchdog {
            let _ = cancel.send(());
            let _ = thread.join();
        }
        let response = match result {
            Ok(Ok(result)) => WorkerResponse { index: job.index, result, error: None },
            Ok(Err(error)) => WorkerResponse { index: job.index, result: None, error: Some(error.to_string()) },
            Err(_) => WorkerResponse { index: job.index, result: None, error: Some("candidate panicked".into()) },
        };
        write_frame(&mut output, &response)
            .map_err(|source| BeamWorker::Frame { source, what: "write worker response" })?;
    }
    Ok(())
}

struct BusyTask {
    index: usize,
    started: Instant,
}

struct SpawnedWorker {
    child: Child,
    input: ChildStdin,
    responses: Receiver<std::io::Result<WorkerResponse>>,
    reader: Option<JoinHandle<()>>,
    busy: Option<BusyTask>,
    tasks: usize,
    terminated: bool,
}

impl SpawnedWorker {
    fn terminate(&mut self) {
        if self.terminated {
            return;
        }
        self.terminated = true;
        #[cfg(unix)]
        unsafe {
            // The helper owns a process group so an in-flight clang grandchild
            // cannot survive timeout or worker replacement.
            libc::kill(-(self.child.id() as libc::pid_t), libc::SIGKILL);
        }
        #[cfg(not(unix))]
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

impl Drop for SpawnedWorker {
    fn drop(&mut self) {
        self.terminate();
    }
}

/// Locate the BEAM helper binary.
///
/// Only successes are cached: a transient miss — the helper not built yet, a
/// `SVOD_BEAM_WORKER` pointing at a path that does not exist yet — used to be
/// latched in a `OnceLock` and fail every later BEAM run in the process.
fn helper_path() -> Result<PathBuf> {
    static HELPER: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);
    cached_helper_path(&HELPER, resolve_helper_path)
}

fn cached_helper_path(
    cache: &std::sync::Mutex<Option<PathBuf>>,
    resolve: impl FnOnce() -> Result<PathBuf>,
) -> Result<PathBuf> {
    let mut cached = cache.lock().expect("BEAM helper path lock poisoned");
    if let Some(path) = cached.as_ref() {
        return Ok(path.clone());
    }
    let path = resolve()?;
    *cached = Some(path.clone());
    Ok(path)
}

fn resolve_helper_path() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("SVOD_BEAM_WORKER") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(BeamWorker::HelperUnavailable {
            reason: format!("SVOD_BEAM_WORKER={} is not a file", path.display()),
        });
    }
    // Only set when the *test* harness builds this crate's binaries; a library
    // consumer never sees it, which is why the workspace build below exists.
    if let Some(path) = option_env!("CARGO_BIN_EXE_svod-beam-worker") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
    }

    // Workspace development path. This makes `cargo run -p svod-model
    // --example ...` self-hosting without requiring a manual helper build.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir.parent().ok_or_else(|| BeamWorker::HelperUnavailable {
        reason: "cannot locate workspace; set SVOD_BEAM_WORKER to an installed svod-beam-worker executable".into(),
    })?;
    if !workspace.join("Cargo.toml").is_file() {
        return Err(BeamWorker::HelperUnavailable {
            reason: "svod workspace is unavailable; set SVOD_BEAM_WORKER to an installed helper".into(),
        });
    }
    let mut cargo = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    cargo.args(["build", "-p", "svod-tensor", "--bin", "svod-beam-worker", "--message-format=json-render-diagnostics"]);
    if !cfg!(debug_assertions) {
        cargo.arg("--release");
    }
    let output = cargo
        .current_dir(workspace)
        .stderr(Stdio::inherit())
        .output()
        .map_err(|source| BeamWorker::SpawnHelper { source, path: "cargo build svod-beam-worker".into() })?;
    if !output.status.success() {
        return Err(BeamWorker::HelperUnavailable {
            reason: format!("cargo failed to build svod-beam-worker: {}", output.status),
        });
    }
    // Take the path cargo reports rather than guessing `target/<profile>/`,
    // which is wrong under `CARGO_TARGET_DIR`, a `--target` triple, or a custom
    // profile directory.
    last_executable(&output.stdout).ok_or_else(|| BeamWorker::HelperUnavailable {
        reason: "cargo built svod-beam-worker but reported no executable artifact; \
                 set SVOD_BEAM_WORKER to an installed helper"
            .into(),
    })
}

/// The last non-null `executable` in a cargo JSON message stream: the binary
/// the requested `--bin` target just produced.
fn last_executable(messages: &[u8]) -> Option<PathBuf> {
    std::str::from_utf8(messages)
        .ok()?
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|message| Some(PathBuf::from(message.get("executable")?.as_str()?)))
        .next_back()
}

fn spawn_worker(path: &PathBuf, init: &WorkerInit) -> Result<SpawnedWorker> {
    let mut command = Command::new(path);
    command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child =
        command.spawn().map_err(|source| BeamWorker::SpawnHelper { source, path: path.display().to_string() })?;
    let mut input = child
        .stdin
        .take()
        .ok_or_else(|| BeamWorker::HelperUnavailable { reason: "BEAM helper stdin was not piped".into() })?;
    let mut output = child
        .stdout
        .take()
        .ok_or_else(|| BeamWorker::HelperUnavailable { reason: "BEAM helper stdout was not piped".into() })?;
    write_frame(&mut input, init).map_err(|source| BeamWorker::Frame { source, what: "initialize BEAM helper" })?;
    let ready = read_frame::<WorkerReady>(&mut output)
        .map_err(|source| BeamWorker::Frame { source, what: "read BEAM helper readiness" })?
        .ok_or_else(|| BeamWorker::HelperUnavailable { reason: "BEAM helper exited during initialization".into() })?;
    if let Some(error) = ready.error {
        let _ = child.wait();
        return Err(BeamWorker::HelperUnavailable { reason: format!("initialization failed: {error}") });
    }
    let (send, responses) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        loop {
            match read_frame::<WorkerResponse>(&mut output) {
                Ok(Some(response)) => {
                    if send.send(Ok(response)).is_err() {
                        break;
                    }
                }
                Ok(None) => {
                    let _ = send.send(Err(std::io::Error::other("BEAM helper stdout closed")));
                    break;
                }
                Err(error) => {
                    let _ = send.send(Err(error));
                    break;
                }
            }
        }
    });
    Ok(SpawnedWorker { child, input, responses, reader: Some(reader), busy: None, tasks: 0, terminated: false })
}

pub struct WorkerPool {
    path: PathBuf,
    init: WorkerInit,
    workers: Vec<SpawnedWorker>,
    max_tasks_per_child: usize,
    timeout: Duration,
}

impl WorkerPool {
    pub fn new(count: usize, init: WorkerInit) -> Result<Self> {
        #[cfg(not(unix))]
        return Err(BeamWorker::HelperUnavailable {
            reason: "clean BEAM timeout containment requires Unix process groups; \
                     no safe helper backend is installed"
                .into(),
        });
        let path = helper_path()?;
        let mut workers = Vec::with_capacity(count.max(1));
        for _ in 0..count.max(1) {
            workers.push(spawn_worker(&path, &init)?);
        }
        Ok(Self {
            path,
            max_tasks_per_child: init.beam.max_tasks_per_child.max(1),
            timeout: Duration::from_secs(init.beam.compile_timeout_secs),
            init,
            workers,
        })
    }

    fn replace(&mut self, slot: usize) -> Result<()> {
        let mut old = self.workers.swap_remove(slot);
        old.terminate();
        self.workers.push(spawn_worker(&self.path, &self.init)?);
        Ok(())
    }

    pub fn run(&mut self, candidates: &[Vec<Opt>], mut completed: impl FnMut(WorkerResponse)) -> Result<()> {
        let mut next = 0usize;
        let mut finished = 0usize;
        while finished < candidates.len() {
            for worker in &mut self.workers {
                if worker.busy.is_none() && next < candidates.len() {
                    let job = WorkerJob { index: next, opts: candidates[next].clone() };
                    write_frame(&mut worker.input, &job)
                        .map_err(|source| BeamWorker::Frame { source, what: "send candidate to BEAM helper" })?;
                    worker.busy = Some(BusyTask { index: next, started: Instant::now() });
                    worker.tasks += 1;
                    next += 1;
                }
            }

            let mut progress = false;
            let mut slot = 0usize;
            while slot < self.workers.len() {
                match poll_slot(&self.workers[slot].responses, self.workers[slot].busy.as_ref(), self.timeout) {
                    SlotOutcome::Response(response) => {
                        let task = self.workers[slot]
                            .busy
                            .take()
                            .ok_or(BeamWorker::WorkerMisorder { got: response.index, expected: None })?;
                        if response.index != task.index {
                            return Err(BeamWorker::WorkerMisorder { got: response.index, expected: Some(task.index) });
                        }
                        finished += 1;
                        progress = true;
                        completed(*response);
                        if self.workers[slot].tasks >= self.max_tasks_per_child {
                            self.replace(slot)?;
                            continue;
                        }
                    }
                    SlotOutcome::Failed(error) => {
                        let failed = self.workers[slot].busy.take();
                        if failed.is_some() {
                            finished += 1;
                        }
                        progress = true;
                        self.replace(slot)?;
                        if let Some(error) = error
                            && std::env::var_os("BEAM_DEBUG").is_some()
                        {
                            eprintln!("[BEAM drop] worker_io: {error}");
                        }
                        continue;
                    }
                    SlotOutcome::TimedOut => {
                        let task = self.workers[slot].busy.take().expect("timeout implies a busy task");
                        finished += 1;
                        progress = true;
                        if std::env::var_os("BEAM_DEBUG").is_some() {
                            eprintln!("[BEAM drop] worker_timeout candidate={}", task.index);
                        }
                        self.replace(slot)?;
                        continue;
                    }
                    SlotOutcome::Idle => {}
                }
                slot += 1;
            }
            if !progress {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        Ok(())
    }
}

/// What one poll of a worker slot found.
enum SlotOutcome {
    Response(Box<WorkerResponse>),
    /// The helper's stdout errored or closed; `None` for a disconnected channel.
    Failed(Option<std::io::Error>),
    TimedOut,
    Idle,
}

/// Drain the worker's channel first and only call it late when the channel is
/// empty.
///
/// Checking the deadline first raced the reader thread: a response written
/// exactly on the deadline was already queued, but the slot was declared timed
/// out, the candidate was dropped, and a healthy helper was SIGKILLed and
/// respawned.
fn poll_slot(
    responses: &Receiver<std::io::Result<WorkerResponse>>,
    busy: Option<&BusyTask>,
    timeout: Duration,
) -> SlotOutcome {
    match responses.try_recv() {
        Ok(Ok(response)) => SlotOutcome::Response(Box::new(response)),
        Ok(Err(error)) => SlotOutcome::Failed(Some(error)),
        Err(mpsc::TryRecvError::Disconnected) => SlotOutcome::Failed(None),
        Err(mpsc::TryRecvError::Empty) => match busy {
            Some(task) if !timeout.is_zero() && task.started.elapsed() >= timeout => SlotOutcome::TimedOut,
            _ => SlotOutcome::Idle,
        },
    }
}

#[cfg(test)]
#[path = "test/unit/beam_worker.rs"]
mod tests;
