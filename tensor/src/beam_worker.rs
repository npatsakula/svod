//! Spawned worker protocol for BEAM candidate compilation.
//!
//! A worker is this very program started again in worker mode: a constructor
//! installed by this module finds the socket named by [`WORKER_SOCKET_ENV`]
//! and serves the compile loop before `main` is ever reached. That keeps hard
//! containment (a watchdog that kills the process group, recycling, isolation
//! from a candidate's abort) without shipping a helper executable next to
//! every application. A process that cannot start itself again — a library
//! loaded into a foreign runtime — compiles its candidates in process
//! instead, without that containment.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
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

pub const BEAM_WORKER_PROTOCOL_VERSION: u32 = 2;

/// Names the Unix socket a started worker serves. Set only on a worker, and
/// spelled once more as a C string in `worker_socket`, which runs before
/// `main` and cannot read this constant's storage.
pub const WORKER_SOCKET_ENV: &str = "SVOD_BEAM_WORKER_SOCKET";

/// The constructor ran in this process. Whether that makes a worker of it is
/// [`worker_entry_armed`]: the constructor also runs in a library loaded into
/// another program.
static WORKER_ENTRY_ARMED: AtomicBool = AtomicBool::new(false);

#[cfg(all(unix, not(target_os = "linux")))]
static WORKER_ENTRY_IMAGE: std::sync::Mutex<Option<(u64, u64)>> = std::sync::Mutex::new(None);

/// A program linked with this module runs this before `main`. Started as a
/// worker — the socket named in its environment — it serves the parent from
/// here and never reaches `main`; otherwise it only records that it ran.
#[cfg(unix)]
#[used]
#[cfg_attr(target_os = "macos", link_section = "__DATA,__mod_init_func")]
#[cfg_attr(not(target_os = "macos"), unsafe(link_section = ".init_array.65535"))]
static WORKER_ENTRY: extern "C" fn() = worker_entry;

#[cfg(unix)]
extern "C" fn worker_entry() {
    WORKER_ENTRY_ARMED.store(true, Ordering::Release);
    #[cfg(not(target_os = "linux"))]
    record_image();
    let Some(socket) = worker_socket() else { return };
    let served = match std::os::unix::net::UnixStream::connect(&socket) {
        Ok(stream) => worker_main_on(stream),
        Err(source) => Err(BeamWorker::Frame { source, what: "connect to the BEAM parent's socket" }),
    };
    if let Err(error) = served {
        eprintln!("svod BEAM worker: {error}");
        std::process::exit(1);
    }
    std::process::exit(0);
}

/// The socket a started worker was given, read through libc: `std::env` is
/// not filled in before `main`.
#[cfg(unix)]
fn worker_socket() -> Option<PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    // SAFETY: a static string, and `getenv` only reads the environment.
    let value = unsafe { libc::getenv(c"SVOD_BEAM_WORKER_SOCKET".as_ptr().cast()) };
    if value.is_null() {
        return None;
    }
    // SAFETY: `getenv` returns a pointer into the environment.
    let bytes = unsafe { std::ffi::CStr::from_ptr(value) }.to_bytes();
    (!bytes.is_empty()).then(|| PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
}

/// Linux starts the running image through `/proc/self/exe`, which a replaced
/// file on disk cannot change. Elsewhere the program is started by path, so
/// the file started is compared with the one this process was loaded from.
#[cfg(all(unix, not(target_os = "linux")))]
fn record_image() {
    use std::os::unix::fs::MetadataExt;
    let Ok(image) = std::env::current_exe().and_then(|path| path.metadata()) else {
        return;
    };
    let _ = WORKER_ENTRY_IMAGE.lock().map(|loaded| *loaded = Some((image.dev(), image.ino())));
}

/// Can this process start itself as a worker?
pub fn worker_entry_armed() -> bool {
    static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ARMED.get_or_init(|| WORKER_ENTRY_ARMED.load(Ordering::Acquire) && hook_in_main_executable() && image_known())
}

/// Elsewhere than Linux the program is started by path, so it must have been
/// identified at start to be checked against the running image later.
#[cfg(all(unix, not(target_os = "linux")))]
fn image_known() -> bool {
    WORKER_ENTRY_IMAGE.lock().map(|loaded| loaded.is_some()).unwrap_or(false)
}

#[cfg(any(not(unix), target_os = "linux"))]
fn image_known() -> bool {
    true
}

/// No constructor, so no executable that could be started again as a worker.
#[cfg(not(unix))]
fn hook_in_main_executable() -> bool {
    false
}

/// Is the constructor's code part of the main executable? A library loaded
/// into another program — a Python extension, a plugin — runs it too, but
/// that program started again does not load the library: started as a worker
/// it would only run its own `main`.
#[cfg(target_os = "macos")]
fn hook_in_main_executable() -> bool {
    #[repr(C)]
    struct DlInfo {
        dli_fname: *mut std::ffi::c_char,
        dli_fbase: *mut std::ffi::c_void,
        dli_sname: *mut std::ffi::c_char,
        dli_saddr: *mut std::ffi::c_void,
    }
    extern "C" {
        /// dyld's header of an image; image 0 is the main executable.
        fn _dyld_get_image_header(image_index: u32) -> *mut std::ffi::c_void;
        fn dladdr(address: *const std::ffi::c_void, info: *mut DlInfo) -> std::ffi::c_int;
    }
    let mut image = DlInfo {
        dli_fname: std::ptr::null_mut(),
        dli_fbase: std::ptr::null_mut(),
        dli_sname: std::ptr::null_mut(),
        dli_saddr: std::ptr::null_mut(),
    };
    // SAFETY: `dladdr` writes one `DlInfo`, which `image` is, and only reads
    // dyld's image list; `_dyld_get_image_header` only reads that list, and
    // image 0 always exists in a process.
    let (found, main) =
        (unsafe { dladdr(worker_entry as *const std::ffi::c_void, &mut image) }, unsafe { _dyld_get_image_header(0) });
    found != 0 && !image.dli_fbase.is_null() && image.dli_fbase == main
}

/// Every other Unix has no check, so none of its programs is started again.
#[cfg(all(unix, not(target_os = "linux"), not(target_os = "macos")))]
fn hook_in_main_executable() -> bool {
    false
}

/// Is the constructor's code part of the main executable? A library loaded
/// into another program — a Python extension, a plugin — runs it too, but
/// that program started again does not load the library: started as a worker
/// it would only run its own `main`.
#[cfg(target_os = "linux")]
fn hook_in_main_executable() -> bool {
    use std::os::unix::fs::MetadataExt;
    let hook = worker_entry as *const () as usize;
    let Ok(executable) = std::fs::metadata("/proc/self/exe") else { return false };
    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else { return false };
    maps.lines().any(|line| {
        // `start-end perms offset major:minor inode  path`.
        let mut fields = line.split_ascii_whitespace();
        let Some(range) = fields.next() else { return false };
        let Some((start, end)) = range.split_once('-') else { return false };
        let (Ok(start), Ok(end)) = (usize::from_str_radix(start, 16), usize::from_str_radix(end, 16)) else {
            return false;
        };
        // Past `perms` and `offset`.
        let (Some(device), Some(inode)) = (fields.nth(2), fields.next()) else { return false };
        let Some((major, minor)) = device.split_once(':') else { return false };
        let (Ok(major), Ok(minor), Ok(inode)) =
            (u32::from_str_radix(major, 16), u32::from_str_radix(minor, 16), inode.parse::<u64>())
        else {
            return false;
        };
        start <= hook && hook < end && (libc::makedev(major, minor), inode) == (executable.dev(), executable.ino())
    })
}

/// This program, as it runs now.
fn self_executable() -> Result<PathBuf> {
    // The running image itself: a file replaced on disk cannot change it.
    #[cfg(target_os = "linux")]
    {
        Ok(PathBuf::from("/proc/self/exe"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let path = std::env::current_exe()
            .map_err(|source| BeamWorker::SpawnWorker { source, path: "the running program".into() })?;
        #[cfg(unix)]
        check_image(&path)?;
        Ok(path)
    }
}

/// Refuse a program file that is no longer the one this process was loaded
/// from: started, it would be another build, without this build's worker.
#[cfg(all(unix, not(target_os = "linux")))]
fn check_image(path: &Path) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    let Some((dev, ino)) = *WORKER_ENTRY_IMAGE.lock().expect("BEAM image lock poisoned") else {
        return Err(BeamWorker::WorkerUnavailable {
            reason: "the running program's file was not identified at start".into(),
        });
    };
    let started =
        path.metadata().map_err(|source| BeamWorker::SpawnWorker { source, path: path.display().to_string() })?;
    if (started.dev(), started.ino()) != (dev, ino) {
        return Err(BeamWorker::WorkerUnavailable {
            reason: format!(
                "{} is no longer the file this process was loaded from; it cannot be started as a BEAM worker",
                path.display()
            ),
        });
    }
    Ok(())
}

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
    pub compute_ops: Option<u64>,
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
    let bytes = bincode::serde::encode_to_vec(value, bincode::config::standard()).map_err(std::io::Error::other)?;
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
    bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
        .map(|(value, _)| Some(value))
        .map_err(std::io::Error::other)
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
                found.ok_or_else(|| BeamWorker::WorkerUnavailable {
                    reason: format!("no CPU backend matches compiler identity {}", init.compiler_key),
                })?
            }
            DeviceSpec::Amd { device_id } => {
                let arch = init.gpu_arch.and_then(GpuArch::amd).ok_or_else(|| BeamWorker::WorkerUnavailable {
                    reason: "AMD BEAM worker initialization has no target architecture".into(),
                })?;
                svod_runtime::create_amd_codegen(device_id, arch).map_err(BeamWorker::at("AMD codegen"))?
            }
            DeviceSpec::Metal { device_id } => {
                svod_runtime::create_metal_codegen(device_id).map_err(BeamWorker::at("Metal codegen"))?
            }
            DeviceSpec::Cuda { device_id } => {
                let arch = init.gpu_arch.and_then(GpuArch::cuda).ok_or_else(|| BeamWorker::WorkerUnavailable {
                    reason: "CUDA BEAM worker initialization has no target architecture".into(),
                })?;
                svod_runtime::create_cuda_codegen(device_id, arch).map_err(BeamWorker::at("CUDA codegen"))?
            }
            _ => {
                return Err(BeamWorker::WorkerUnavailable {
                    reason: format!("{:?} has no device-disabled BEAM codegen factory", init.device),
                });
            }
        };
    if compiler.cache_key() != init.compiler_key {
        return Err(BeamWorker::WorkerUnavailable {
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
                || BeamWorker::WorkerUnavailable {
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
                || BeamWorker::WorkerUnavailable {
                    reason: "CUDA BEAM worker initialization has no optimizer profile".into(),
                },
            )?
        }
        _ => {
            return Err(BeamWorker::WorkerUnavailable {
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

/// Serve one connection: read the initialization frame, answer readiness,
/// then answer one response per job until the parent closes its end.
fn worker_main_on(mut stream: impl Read + Write) -> Result<()> {
    let init: WorkerInit = read_frame(&mut stream)
        .map_err(|source| BeamWorker::Frame { source, what: "read worker initialization" })?
        .ok_or_else(|| BeamWorker::WorkerUnavailable { reason: "the socket closed before initialization".into() })?;
    let setup = worker_setup(&init);
    let (base_ast, codegen) = match setup {
        Ok(setup) => {
            write_frame(&mut stream, &WorkerReady { error: None })
                .map_err(|source| BeamWorker::Frame { source, what: "write worker readiness" })?;
            setup
        }
        Err(error) => {
            let _ = write_frame(&mut stream, &WorkerReady { error: Some(error.to_string()) });
            return Err(error);
        }
    };
    while let Some(job) =
        read_frame::<WorkerJob>(&mut stream).map_err(|source| BeamWorker::Frame { source, what: "read worker job" })?
    {
        let watchdog = (init.beam.compile_timeout_secs > 0).then(|| {
            let (cancel, wait) = mpsc::channel();
            let timeout = Duration::from_secs(init.beam.compile_timeout_secs);
            let thread = std::thread::spawn(move || {
                if wait.recv_timeout(timeout).is_err() {
                    #[cfg(unix)]
                    unsafe {
                        // The worker is its own process group. This terminates
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
        write_frame(&mut stream, &response)
            .map_err(|source| BeamWorker::Frame { source, what: "write worker response" })?;
    }
    Ok(())
}

/// Everything a worker needs before its first job: the kernel graph and a
/// codegen identical to the parent's.
fn worker_setup(init: &WorkerInit) -> Result<(Arc<svod_ir::UOp>, WorkerCodegen)> {
    if init.protocol_version != BEAM_WORKER_PROTOCOL_VERSION {
        return Err(BeamWorker::ProtocolMismatch {
            expected: init.protocol_version,
            actual: BEAM_WORKER_PROTOCOL_VERSION,
        });
    }
    let base_ast = init.graph.decode_root().map_err(BeamWorker::at("decode graph"))?;
    let codegen = worker_codegen(init)?;
    if codegen.optimizer_renderer.cache_fingerprint() != init.renderer_fingerprint {
        return Err(BeamWorker::WorkerUnavailable {
            reason: format!(
                "renderer identity mismatch: parent={}, worker={}",
                init.renderer_fingerprint,
                codegen.optimizer_renderer.cache_fingerprint()
            ),
        });
    }
    Ok((base_ast, codegen))
}

/// Compile one candidate on the calling thread, for a process that cannot
/// start itself as a worker. The compiler processes this thread runs are
/// killed at the deadline; Rust-side compilation and in-process libLLVM are
/// not, and a panic is reported like a worker's.
fn compile_in_process(
    init: &WorkerInit,
    base_ast: &Arc<svod_ir::UOp>,
    codegen: &WorkerCodegen,
    job: &WorkerJob,
    timeout: Duration,
) -> WorkerResponse {
    let _detached = svod_ir::origin::OriginScope::suspend();
    let compile = || try_compile(init, base_ast, codegen, job);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if timeout.is_zero() {
            compile()
        } else {
            svod_runtime::clang::with_compile_deadline(Instant::now() + timeout, compile)
        }
    }));
    match result {
        Ok(Ok(result)) => WorkerResponse { index: job.index, result, error: None },
        Ok(Err(error)) => WorkerResponse { index: job.index, result: None, error: Some(error.to_string()) },
        Err(_) => WorkerResponse { index: job.index, result: None, error: Some("candidate panicked".into()) },
    }
}

fn warn_in_process() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        tracing::warn!(
            "BEAM compiles its candidates in this process: it cannot start itself as a worker, \
             so a candidate has no hard timeout and its crash would end this process"
        );
    });
}

/// The started workers: one copy of this program per slot, each on its own
/// socket and in its own process group. Unix only — a process is never
/// started again anywhere else.
#[cfg(unix)]
mod started {
    use std::path::Path;
    use std::process::{Child, Command, Stdio};
    use std::sync::mpsc::{self, Receiver};
    use std::thread::JoinHandle;

    use super::*;

    /// A worker slot's socket: one file per worker under the temporary directory,
    /// gone once the slot no longer listens.
    pub(super) struct WorkerListener {
        pub(super) listener: std::os::unix::net::UnixListener,
        pub(super) path: PathBuf,
    }

    impl Drop for WorkerListener {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    pub(super) fn worker_listener() -> std::io::Result<WorkerListener> {
        use std::os::unix::ffi::OsStrExt;
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "svod-beam-{}-{}.sock",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        // A socket address holds 108 bytes including its terminator.
        if path.as_os_str().as_bytes().len() >= 108 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{} is too long for a socket address", path.display()),
            ));
        }
        let listener = std::os::unix::net::UnixListener::bind(&path)?;
        Ok(WorkerListener { listener, path })
    }

    /// The worker's connection, or an error once `deadline` passes: the program
    /// started could be one whose other constructors hang, so the handshake is
    /// never waited on without a bound.
    pub(super) fn accept_worker(
        listener: &std::os::unix::net::UnixListener,
        deadline: Instant,
    ) -> std::io::Result<std::os::unix::net::UnixStream> {
        listener.set_nonblocking(true)?;
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(false)?;
                    return Ok(stream);
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(error) => return Err(error),
            }
            if Instant::now() >= deadline {
                return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "the worker did not connect"));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    pub(super) struct BusyTask {
        pub(super) index: usize,
        pub(super) started: Instant,
    }

    struct SpawnedWorker {
        child: Child,
        input: std::os::unix::net::UnixStream,
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
            terminate_group(&mut self.child);
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

    /// Start this program as a worker and hand back its slot: the connection the
    /// worker made, and its process group, which `terminate` kills. The handshake
    /// — the connection and the readiness frame — is bounded by `accept_timeout`.
    fn spawn_worker(exe: &Path, init: &WorkerInit, accept_timeout: Duration) -> Result<SpawnedWorker> {
        let bound =
            worker_listener().map_err(|source| BeamWorker::Frame { source, what: "bind the BEAM worker socket" })?;
        let socket = &bound.path;
        let mut command = Command::new(exe);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
            command.arg0(std::env::args_os().next().unwrap_or_else(|| exe.as_os_str().to_owned()));
        }
        // A worker owns no stdin or stdout: the frames go over the socket, so
        // nothing the program's other constructors print can land among them.
        command.env(WORKER_SOCKET_ENV, socket).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::inherit());
        let mut child =
            command.spawn().map_err(|source| BeamWorker::SpawnWorker { source, path: exe.display().to_string() })?;
        let connected = accept_worker(&bound.listener, Instant::now() + accept_timeout);
        // Connected or not, the slot stops listening here.
        drop(bound);
        let handshake = connected
            .map_err(|source| BeamWorker::Frame { source, what: "accept the BEAM worker's connection" })
            .and_then(|mut stream| {
                stream
                    .set_read_timeout(Some(accept_timeout))
                    .map_err(|source| BeamWorker::Frame { source, what: "bound the BEAM worker handshake" })?;
                let ready = write_frame(&mut stream, init)
                    .map_err(|source| BeamWorker::Frame { source, what: "initialize BEAM worker" })
                    .and_then(|()| {
                        read_frame::<WorkerReady>(&mut stream)
                            .map_err(|source| BeamWorker::Frame { source, what: "read BEAM worker readiness" })?
                            .ok_or_else(|| BeamWorker::WorkerUnavailable {
                                reason: "the BEAM worker exited during initialization".into(),
                            })
                    });
                // From here a job is bounded by the parent's own watchdog, and a
                // read blocks until the worker answers or exits.
                stream
                    .set_read_timeout(None)
                    .map_err(|source| BeamWorker::Frame { source, what: "unbound the BEAM worker socket" })?;
                ready.map(|ready| (stream, ready))
            });
        let (stream, ready) = match handshake {
            Ok(connected) => connected,
            Err(error) => {
                // Started, it is a worker whatever happens next: it must not
                // outlive a handshake it lost.
                terminate_group(&mut child);
                return Err(BeamWorker::WorkerUnavailable { reason: format!("the worker did not start: {error}") });
            }
        };
        if let Some(error) = ready.error {
            terminate_group(&mut child);
            return Err(BeamWorker::WorkerUnavailable { reason: format!("initialization failed: {error}") });
        }
        let output =
            stream.try_clone().map_err(|source| BeamWorker::Frame { source, what: "share the BEAM worker socket" })?;
        let (send, responses) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            loop {
                match read_frame::<WorkerResponse>(&mut &output) {
                    Ok(Some(response)) => {
                        if send.send(Ok(response)).is_err() {
                            break;
                        }
                    }
                    Ok(None) => {
                        let _ = send.send(Err(std::io::Error::other("the BEAM worker closed its socket")));
                        break;
                    }
                    Err(error) => {
                        let _ = send.send(Err(error));
                        break;
                    }
                }
            }
        });
        Ok(SpawnedWorker {
            child,
            input: stream,
            responses,
            reader: Some(reader),
            busy: None,
            tasks: 0,
            terminated: false,
        })
    }

    /// Kill a started worker's process group and reap it.
    fn terminate_group(child: &mut Child) {
        // SAFETY: a signal to the group the worker was started in, which is not
        // reaped yet: an in-flight clang grandchild cannot survive a lost
        // handshake or worker replacement.
        unsafe {
            libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL);
        }
        let _ = child.wait();
    }

    /// The slots of one pool.
    pub(super) struct Pool {
        init: WorkerInit,
        exe: PathBuf,
        workers: Vec<SpawnedWorker>,
        max_tasks_per_child: usize,
        timeout: Duration,
        /// The bound on a worker's handshake, as on its jobs.
        accept_timeout: Duration,
    }

    impl Pool {
        pub(super) fn new(count: usize, init: WorkerInit, exe: PathBuf, timeout: Duration) -> Result<Self> {
            let mut workers = Vec::with_capacity(count.max(1));
            for _ in 0..count.max(1) {
                // Should a later one fail to start, the ones already started
                // are dropped, and so terminated, on the way out.
                workers.push(spawn_worker(&exe, &init, accept_timeout_of(timeout))?);
            }
            Ok(Self {
                max_tasks_per_child: init.beam.max_tasks_per_child.max(1),
                accept_timeout: accept_timeout_of(timeout),
                timeout,
                workers,
                init,
                exe,
            })
        }

        fn replace(&mut self, slot: usize) -> Result<()> {
            let mut old = self.workers.swap_remove(slot);
            old.terminate();
            let spawned = spawn_worker(&self.exe, &self.init, self.accept_timeout);
            self.workers.push(spawned?);
            Ok(())
        }

        pub(super) fn run(&mut self, candidates: &[Vec<Opt>], mut completed: impl FnMut(WorkerResponse)) -> Result<()> {
            let mut next = 0usize;
            let mut finished = 0usize;
            while finished < candidates.len() {
                for worker in &mut self.workers {
                    if worker.busy.is_none() && next < candidates.len() {
                        let job = WorkerJob { index: next, opts: candidates[next].clone() };
                        write_frame(&mut worker.input, &job)
                            .map_err(|source| BeamWorker::Frame { source, what: "send candidate to BEAM worker" })?;
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
                                return Err(BeamWorker::WorkerMisorder {
                                    got: response.index,
                                    expected: Some(task.index),
                                });
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
    pub(super) enum SlotOutcome {
        Response(Box<WorkerResponse>),
        /// The worker's socket errored or closed; `None` for a disconnected channel.
        Failed(Option<std::io::Error>),
        TimedOut,
        Idle,
    }

    /// Drain the worker's channel first and only call it late when the channel is
    /// empty.
    ///
    /// Checking the deadline first raced the reader thread: a response written
    /// exactly on the deadline was already queued, but the slot was declared timed
    /// out, the candidate was dropped, and a healthy worker was SIGKILLed and
    /// respawned.
    pub(super) fn poll_slot(
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

    /// A job with no timeout still cannot wait on a handshake forever.
    fn accept_timeout_of(timeout: Duration) -> Duration {
        if timeout.is_zero() { Duration::from_secs(60) } else { timeout }
    }
}

/// How the pool's candidates are compiled.
enum Backend {
    /// One started copy of this program per slot.
    #[cfg(unix)]
    Started(Box<started::Pool>),
    /// On the calling thread, in this process.
    InProcess { base_ast: Arc<svod_ir::UOp>, codegen: Box<WorkerCodegen>, timeout: Duration },
}

pub struct WorkerPool {
    /// What a candidate is compiled with; every worker was given the same.
    init: WorkerInit,
    backend: Backend,
}

impl WorkerPool {
    /// A pool of `count` workers, or of none when this process cannot start
    /// itself as one: then every candidate is compiled on the calling thread,
    /// without a worker's hard timeout and isolation.
    pub fn new(count: usize, init: WorkerInit) -> Result<Self> {
        let timeout = Duration::from_secs(init.beam.compile_timeout_secs);
        // Only a program whose worker constructor ran is started again: any
        // other would run its own `main` in the child's place.
        #[cfg(unix)]
        if worker_entry_armed() {
            return Self::started(count, init, timeout);
        }
        Self::in_process(init, timeout)
    }

    /// `count` started copies of this program, each serving one slot.
    #[cfg(unix)]
    fn started(count: usize, init: WorkerInit, timeout: Duration) -> Result<Self> {
        let exe = self_executable()?;
        Ok(Self { backend: Backend::Started(Box::new(started::Pool::new(count, init.clone(), exe, timeout)?)), init })
    }

    /// Every candidate compiled on the calling thread: for a process that
    /// cannot start itself as a worker.
    pub(crate) fn in_process(init: WorkerInit, timeout: Duration) -> Result<Self> {
        // Said once per process: a search per kernel would repeat it.
        warn_in_process();
        let (base_ast, codegen) = worker_setup(&init)?;
        Ok(Self { backend: Backend::InProcess { base_ast, codegen: Box::new(codegen), timeout }, init })
    }

    pub fn run(&mut self, candidates: &[Vec<Opt>], mut completed: impl FnMut(WorkerResponse)) -> Result<()> {
        match &mut self.backend {
            #[cfg(unix)]
            Backend::Started(pool) => pool.run(candidates, completed),
            Backend::InProcess { base_ast, codegen, timeout } => {
                for (index, opts) in candidates.iter().enumerate() {
                    let job = WorkerJob { index, opts: opts.clone() };
                    completed(compile_in_process(&self.init, base_ast, codegen, &job, *timeout));
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
#[path = "test/unit/beam_worker.rs"]
mod tests;
