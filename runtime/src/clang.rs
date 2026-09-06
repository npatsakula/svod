//! Clang compilation backend for C codegen.
//!
//! By default, compiles C source via `clang -c` stdin→stdout and loads the
//! resulting object via custom ELF parsing + mmap (no temp files, no dlopen).
//!
//! With `dlopen-fallback` feature: compiles via `clang -shared -O2` and loads
//! the resulting shared library via `dlopen` for kernel execution.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use object::read::{Object, ObjectSymbol};
use object::{Architecture, BinaryFormat, Endianness, ObjectKind};
use sha2::{Digest, Sha256};

use crate::error::JitResultExt;
use crate::object_cache::ObjectCache;
use crate::{Error, Result};

/// Resolved Clang executable plus an exact persisted identity. The executable
/// digest makes replacement unambiguous; `--version` remains in the identity
/// for diagnostics and toolchain-version auditability.
#[derive(Debug, Clone)]
pub(crate) struct ClangToolchain {
    executable: PathBuf,
    executable_digest: [u8; 32],
    identity: String,
}

impl ClangToolchain {
    pub(crate) fn discover(cache: Option<&ObjectCache>) -> Result<Self> {
        let executable = resolve_executable("clang")?;
        let executable_bytes = std::fs::read(&executable).jit("read clang executable")?;
        let executable_digest: [u8; 32] = Sha256::digest(&executable_bytes).into();
        let probe_input = executable_digest;
        let version = if let Some(cache) = cache {
            cache.get_or_create_probe("clang-version", &probe_input, || run_probe(&executable, &["--version"]))?
        } else {
            run_probe(&executable, &["--version"])?
        };
        let version = String::from_utf8(version)
            .map_err(|error| Error::JitCompilation { reason: format!("clang --version was not UTF-8: {error}") })?;
        let identity =
            format!("path={};sha256={};version={}", executable.display(), hex(&executable_digest), version.trim());
        Ok(Self { executable, executable_digest, identity })
    }

    pub(crate) fn identity(&self) -> &str {
        &self.identity
    }

    /// Resolve `flags` into the concrete target description clang will use —
    /// `-###` reports the selected `-target-cpu` and feature list — so it can
    /// be persisted as `CompilerIdentity::target_architecture`.
    pub(crate) fn target_identity(&self, cache: Option<&ObjectCache>, flags: &[String]) -> Result<String> {
        let probe_input = probe_key(&self.executable_digest, flags, host_cpu_fingerprint());
        let create = || {
            let mut command = Command::new(&self.executable);
            command.args(flags).arg("-###").stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
            let output = command.output().jit("probe clang target architecture")?;
            if !output.status.success() {
                return Err(Error::JitCompilation {
                    reason: format!("clang target probe failed:\n{}", String::from_utf8_lossy(&output.stderr)),
                });
            }
            let mut identity = output.stderr;
            identity.extend_from_slice(&output.stdout);
            Ok(identity)
        };
        let output = match (cache, &probe_input) {
            (Some(cache), Some(probe_input)) => cache.get_or_create_probe("clang-target", probe_input, create)?,
            _ => create()?,
        };
        String::from_utf8(output)
            .map_err(|error| Error::JitCompilation { reason: format!("clang target probe was not UTF-8: {error}") })
    }

    pub(crate) fn command(&self) -> Command {
        Command::new(&self.executable)
    }

    /// Does this clang advertise `target` (an LLVM target name such as
    /// `amdgcn` or `nvptx64`)? Memoized per executable: an installation does
    /// not change during a run, and the probe forks a subprocess.
    pub(crate) fn has_target(&self, target: &'static str) -> bool {
        static CACHE: std::sync::OnceLock<papaya::HashMap<(PathBuf, &'static str), bool>> = std::sync::OnceLock::new();
        let probed = CACHE.get_or_init(papaya::HashMap::new).pin();
        let key = (self.executable.clone(), target);
        if let Some(known) = probed.get(&key) {
            return *known;
        }
        let result = probe_target(&self.executable, target);
        probed.insert(key, result);
        result
    }

    /// Feed LLVM IR text to clang on stdin and return its stdout. `args` must
    /// already select `-x ir`, the target and the output form; `what` names
    /// the target in diagnostics.
    pub(crate) fn compile_ir(&self, args: &[String], ir: &str, what: &str) -> Result<Vec<u8>> {
        let mut child = self
            .command()
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .jit("spawn clang for LLVM IR")?;
        child.stdin.take().expect("stdin piped").write_all(ir.as_bytes()).jit("write LLVM IR to clang stdin")?;
        let output = child.wait_with_output().jit("wait for clang")?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::JitCompilation { reason: format!("clang {what} compilation failed:\n{stderr}") });
        }
        if output.stdout.is_empty() {
            return Err(Error::JitCompilation { reason: format!("clang produced empty output for {what}") });
        }
        Ok(output.stdout)
    }
}

/// Does `executable` list `target` under `--print-targets`?
pub(crate) fn probe_target(executable: &Path, target: &str) -> bool {
    let Ok(output) = Command::new(executable).arg("--print-targets").output() else { return false };
    output.status.success()
        && String::from_utf8_lossy(&output.stdout).lines().any(|line| line.split_whitespace().next() == Some(target))
}

/// Does the `clang` on `PATH` advertise `target`? Cached for the lifetime of
/// the process, keyed by target name.
pub(crate) fn path_clang_has_target(target: &'static str) -> bool {
    static CACHE: std::sync::OnceLock<papaya::HashMap<&'static str, bool>> = std::sync::OnceLock::new();
    let probed = CACHE.get_or_init(papaya::HashMap::new).pin();
    if let Some(known) = probed.get(target) {
        return *known;
    }
    let result = probe_target(Path::new("clang"), target);
    probed.insert(target, result);
    result
}

/// When `env_var` names a directory, write `ir` there as `<tag>_<module>.ll`
/// so each kernel lands in its own file. The module name comes from the
/// `; ModuleID = '<name>'` directive: the dispatcher pre-compiles many
/// kernels ahead of any dispatch, so a single fixed path would only ever hold
/// the last one compiled, never the failing one.
pub(crate) fn dump_ir(env_var: &str, tag: &str, ir: &str) {
    let Ok(dir) = std::env::var(env_var) else { return };
    let module = ir
        .lines()
        .find_map(|l| l.strip_prefix("; ModuleID = '").and_then(|s| s.strip_suffix("'")))
        .unwrap_or("unknown");
    // Kernel names are `[A-Za-z0-9_]` in practice; be defensive anyway.
    let safe: String = module.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(Path::new(&dir).join(format!("{tag}_{safe}.ll")), ir);
}

/// Probe-cache key for one flag set, or `None` when the result must not be
/// shared at all.
///
/// `-march=native` / `-mcpu=native` resolve against the *running* CPU, so a
/// `-###` result cached in a directory shared between machines would hand a
/// second host the first host's `-target-cpu` — and, through
/// `CompilerIdentity::target_architecture`, its compiled objects. Tinygrad
/// avoids this by never resolving `native` implicitly: its CPU compiler is
/// constructed from an explicit `<arch>,<cpu>,<feats>` string
/// (`runtime/support/compiler_cpu.py:8-15`) that is baked into the cache key
/// (`compiler_llvm.py:46`). We keep `native` and discriminate on the host
/// instead.
fn probe_key(executable_digest: &[u8; 32], flags: &[String], host: Option<&[u8; 32]>) -> Option<Vec<u8>> {
    let mut input = Vec::with_capacity(executable_digest.len() + flags.len() * 16);
    input.extend_from_slice(executable_digest);
    for flag in flags {
        input.extend_from_slice(&(flag.len() as u64).to_le_bytes());
        input.extend_from_slice(flag.as_bytes());
    }
    if flags.iter().any(|flag| flag.contains("native")) {
        input.extend_from_slice(host?);
    }
    Some(input)
}

/// Digest of the stable `/proc/cpuinfo` lines that decide what `native`
/// resolves to. Per-core frequency and bogomips are excluded: they change
/// between runs on one machine and must not evict the probe.
fn host_cpu_fingerprint() -> Option<&'static [u8; 32]> {
    static FINGERPRINT: std::sync::OnceLock<Option<[u8; 32]>> = std::sync::OnceLock::new();
    FINGERPRINT
        .get_or_init(|| {
            const STABLE: &[&str] = &[
                "vendor_id",
                "cpu family",
                "model",
                "model name",
                "stepping",
                "flags",
                "Features",
                "CPU implementer",
                "CPU architecture",
                "CPU variant",
                "CPU part",
                "isa",
                "uarch",
            ];
            let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
            let mut digest = Sha256::new();
            digest.update(std::env::consts::ARCH.as_bytes());
            for line in cpuinfo.lines() {
                let Some((key, _)) = line.split_once(':') else { continue };
                if STABLE.contains(&key.trim()) {
                    digest.update(line.trim().as_bytes());
                    digest.update(b"\n");
                }
            }
            Some(digest.finalize().into())
        })
        .as_ref()
}

pub(crate) fn c_object_flags() -> Vec<String> {
    #[cfg(feature = "dlopen-fallback")]
    {
        let march = match std::env::consts::ARCH {
            "x86_64" | "loongarch64" => "-march=native",
            "riscv64" => "-march=rv64g",
            _ => "-mcpu=native",
        };
        let mut flags = vec!["-shared", "-O2", march, "-fPIC", "-fno-math-errno", "-fno-ident", "-lm", "-x", "c"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        flags.push("-ffixed-x18".into());
        flags.extend(["-", "-o", "<temporary-shared-object>"].map(str::to_string));
        flags
    }
    #[cfg(not(feature = "dlopen-fallback"))]
    {
        crate::jit_loader::c_object_flags()
    }
}

pub(crate) fn compile_c_object(toolchain: &ClangToolchain, src: &str, flags: &[String]) -> Result<Vec<u8>> {
    #[cfg(not(feature = "dlopen-fallback"))]
    {
        let mut child = toolchain
            .command()
            .args(flags)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .jit("spawn clang (is clang installed?)")?;
        child.stdin.take().expect("stdin was piped").write_all(src.as_bytes()).jit("write source to clang stdin")?;
        let output = child.wait_with_output().jit("wait for clang")?;
        if !output.status.success() {
            return Err(Error::JitCompilation {
                reason: format!(
                    "clang compilation failed:\n{}\nSource:\n{src}",
                    String::from_utf8_lossy(&output.stderr)
                ),
            });
        }
        if output.stdout.is_empty() {
            return Err(Error::JitCompilation { reason: "clang produced empty output".into() });
        }
        Ok(output.stdout)
    }
    #[cfg(feature = "dlopen-fallback")]
    {
        let directory = tempfile::tempdir().jit("create clang output directory")?;
        let output_path = directory.path().join("kernel.so");
        let mut args = flags.to_vec();
        *args.last_mut().expect("C flags have output placeholder") = output_path.display().to_string();
        let mut child = toolchain
            .command()
            .args(&args)
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .jit("spawn clang shared-object compiler")?;
        child.stdin.take().expect("stdin was piped").write_all(src.as_bytes()).jit("write source to clang stdin")?;
        let output = child.wait_with_output().jit("wait for clang shared-object compiler")?;
        if !output.status.success() {
            return Err(Error::JitCompilation {
                reason: format!("clang shared-object compilation failed:\n{}", String::from_utf8_lossy(&output.stderr)),
            });
        }
        std::fs::read(output_path).jit("read clang shared object")
    }
}

pub(crate) fn validate_c_object(bytes: &[u8], symbol: &str) -> Result<()> {
    #[cfg(not(feature = "dlopen-fallback"))]
    let expected_kind = ObjectKind::Relocatable;
    #[cfg(feature = "dlopen-fallback")]
    let expected_kind = ObjectKind::Dynamic;
    validate_host_object(bytes, symbol, expected_kind)
}

pub(crate) fn validate_relocatable_object(bytes: &[u8], symbol: &str) -> Result<()> {
    validate_host_object(bytes, symbol, ObjectKind::Relocatable)
}

fn validate_host_object(bytes: &[u8], symbol: &str, expected_kind: ObjectKind) -> Result<()> {
    let file = object::File::parse(bytes).jit("parse cached CPU object")?;
    if file.format() != BinaryFormat::Elf
        || file.endianness() != host_endianness()
        || file.architecture() != host_architecture()?
    {
        return Err(Error::JitCompilation { reason: "cached CPU object has incompatible ELF target".into() });
    }
    if file.kind() != expected_kind {
        return Err(Error::JitCompilation {
            reason: format!("cached CPU object kind {:?} does not match expected {expected_kind:?}", file.kind()),
        });
    }
    if !file.symbols().any(|candidate| candidate.is_definition() && candidate.name() == Ok(symbol)) {
        return Err(Error::JitCompilation { reason: format!("cached CPU object has no entry symbol {symbol:?}") });
    }
    Ok(())
}

fn host_architecture() -> Result<Architecture> {
    match std::env::consts::ARCH {
        "x86_64" => Ok(Architecture::X86_64),
        "aarch64" => Ok(Architecture::Aarch64),
        "riscv64" => Ok(Architecture::Riscv64),
        "loongarch64" => Ok(Architecture::LoongArch64),
        "powerpc64" => Ok(Architecture::PowerPc64),
        arch => Err(Error::JitCompilation { reason: format!("unsupported CPU object architecture {arch}") }),
    }
}

fn host_endianness() -> Endianness {
    if cfg!(target_endian = "little") { Endianness::Little } else { Endianness::Big }
}

pub(crate) fn resolve_executable(name: &str) -> Result<PathBuf> {
    let path = std::env::var_os("PATH").ok_or_else(|| Error::JitCompilation { reason: "PATH is not set".into() })?;
    for directory in std::env::split_paths(&path) {
        let candidate = directory.join(name);
        if candidate.is_file() {
            return candidate.canonicalize().jit("canonicalize clang executable");
        }
    }
    Err(Error::JitCompilation { reason: format!("{name} not found in PATH") })
}

pub(crate) fn run_probe(executable: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let output = Command::new(executable).args(args).output().jit("run compiler identity probe")?;
    if !output.status.success() {
        return Err(Error::JitCompilation {
            reason: format!(
                "{} identity probe failed:\n{}",
                executable.display(),
                String::from_utf8_lossy(&output.stderr)
            ),
        });
    }
    let mut bytes = output.stdout;
    bytes.extend_from_slice(&output.stderr);
    Ok(bytes)
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

// Default: JIT ELF loader (no temp files, no dlopen)
#[cfg(not(feature = "dlopen-fallback"))]
pub use crate::jit_loader::JitKernel as ClangKernel;

// Fallback: dlopen-based loading
#[cfg(feature = "dlopen-fallback")]
mod dlopen_impl {
    use crate::Result;
    use crate::dispatch::KernelCif;
    use crate::error::JitResultExt;

    /// A compiled C kernel loaded as a shared library.
    pub struct ClangKernel {
        _lib: libloading::Library,
        fn_ptr: *const (),
        name: String,
        var_names: Vec<String>,
        cif: KernelCif,
        _tmp_dir: tempfile::TempDir,
    }

    // SAFETY: The function pointer points to read-only compiled code
    // in the loaded shared library. Multiple threads can call it concurrently.
    unsafe impl Send for ClangKernel {}
    unsafe impl Sync for ClangKernel {}

    impl ClangKernel {
        pub fn compile_with_abi(
            src: &str,
            name: &str,
            var_names: Vec<String>,
            abi: &[svod_device::device::AbiParamDescriptor],
        ) -> Result<Self> {
            let buffer_count = abi.iter().filter(|arg| arg.is_storage()).count();
            svod_device::device::validate_abi_descriptors(abi, buffer_count, &var_names)?;
            let toolchain = super::ClangToolchain::discover(None)?;
            let flags = super::c_object_flags();
            let bytes = super::compile_c_object(&toolchain, src, &flags)?;
            Self::load_object_with_abi(&bytes, name, var_names, abi)
        }

        pub fn load_object_with_abi(
            bytes: &[u8],
            name: &str,
            var_names: Vec<String>,
            abi: &[svod_device::device::AbiParamDescriptor],
        ) -> Result<Self> {
            let buffer_count = abi.iter().filter(|arg| arg.is_storage()).count();
            svod_device::device::validate_abi_descriptors(abi, buffer_count, &var_names)?;
            super::validate_c_object(bytes, name)?;
            let tmp_dir = tempfile::tempdir().jit("create shared-object load directory")?;
            let so_path = tmp_dir.path().join(format!("{name}.so"));
            std::fs::write(&so_path, bytes).jit("write cached shared object")?;
            let lib = unsafe { libloading::Library::new(&so_path).jit("load shared library")? };

            let fn_ptr = unsafe {
                let func: libloading::Symbol<unsafe extern "C" fn()> = lib
                    .get(name.as_bytes())
                    .map_err(|e| crate::Error::FunctionNotFound { name: format!("{name}: {e}") })?;
                *func as *const ()
            };

            let cif = KernelCif::from_abi(abi);
            tracing::debug!(kernel.name = %name, "Clang kernel compiled and loaded (dlopen)");

            Ok(Self { _lib: lib, fn_ptr, name: name.to_string(), var_names, cif, _tmp_dir: tmp_dir })
        }

        pub unsafe fn execute_with_vals(&self, buffers: &[*mut u8], vals: &[i64]) -> Result<()> {
            unsafe { self.cif.dispatch(self.fn_ptr, buffers, vals, None)? };
            Ok(())
        }

        pub(crate) fn cif(&self) -> &KernelCif {
            &self.cif
        }

        pub fn var_names(&self) -> &[String] {
            &self.var_names
        }

        pub fn fn_ptr(&self) -> *const () {
            self.fn_ptr
        }

        pub fn name(&self) -> &str {
            &self.name
        }
    }
}

#[cfg(feature = "dlopen-fallback")]
pub use dlopen_impl::ClangKernel;

#[cfg(test)]
#[path = "test/unit/clang.rs"]
mod tests;
