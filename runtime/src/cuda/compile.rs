//! `clang --target=nvptx64-nvidia-cuda` driver: lowers NVPTX LLVM text-IR to
//! PTX text, which `Ptxas` assembles to a cubin when the CUDA toolkit is
//! installed and the driver JITs otherwise.

use std::path::{Path, PathBuf};
use std::process::Command;

use svod_codegen::llvm::nvptx::clang_flags;
use svod_dtype::CudaArch;
use tracing::debug;

use crate::clang::{ClangToolchain, dump_ir, path_clang_has_target, resolve_executable, run_piped, run_probe};
use crate::error::JitResultExt;
use crate::object_cache::ObjectCache;

/// Compile NVPTX LLVM IR text into PTX text for `arch`.
///
/// # Errors
///
/// - [`crate::Error::JitCompilation`] when `clang` is missing, the NVPTX
///   target is not enabled in the host LLVM, or compilation fails.
pub fn compile_ir_to_ptx(ir: &str, arch: CudaArch) -> crate::Result<Vec<u8>> {
    let toolchain = ClangToolchain::discover(None)?;
    compile_ir_to_ptx_with(&toolchain, ir, arch)
}

pub(crate) fn compile_ir_to_ptx_with(toolchain: &ClangToolchain, ir: &str, arch: CudaArch) -> crate::Result<Vec<u8>> {
    if !toolchain.has_target("nvptx64") {
        return Err(crate::Error::JitCompilation {
            reason: "NVIDIA GPU support requires clang built with the NVPTX target. \
                     Reinstall clang from your distro or build with \
                     -DLLVM_TARGETS_TO_BUILD='X86;AArch64;NVPTX'."
                .to_string(),
        });
    }
    dump_ir("SVOD_DUMP_NVPTX_IR", &arch.to_string(), ir);
    debug!(arch = %arch, ir.length = ir.len(), "compiling nvptx IR via clang");
    toolchain.compile_ir(&clang_flags(arch), ir, &format!("nvptx (march={arch})"))
}

/// Check PTX text before it reaches the driver JIT, on cached and fresh bytes
/// alike: a `.version` directive, a `.target` matching `arch`, an `.entry`
/// named `kernel_name`, and no `.extern .func`. The last one is an LLVM
/// intrinsic the NVPTX backend did not recognize (a misspelt `llvm.nvvm.*`
/// name is silently emitted as an external call) and would only surface as a
/// `cuModuleLoadDataEx` failure.
pub(crate) fn validate_ptx(bytes: &[u8], arch: CudaArch, kernel_name: &str) -> crate::Result<()> {
    let reject = |reason: String| Err(crate::Error::JitCompilation { reason });
    let Ok(ptx) = std::str::from_utf8(bytes) else { return reject("cached PTX is not UTF-8 text".into()) };
    let directive =
        |name: &str| ptx.lines().find_map(|line| line.trim_start().strip_prefix(name)).map(|rest| rest.trim());
    match directive(".version ") {
        Some(version) if version.split_once('.').is_some_and(|(a, b)| is_number(a) && is_number(b)) => {}
        _ => return reject("cached PTX has no `.version` directive".into()),
    }
    match directive(".target ") {
        Some(target) if target.split(',').next().is_some_and(|sm| sm.trim() == arch.to_string()) => {}
        Some(target) => return reject(format!("cached PTX targets {target}, not {arch}")),
        None => return reject("cached PTX has no `.target` directive".into()),
    }
    if let Some(line) = ptx.lines().find(|line| line.contains(".extern .func")) {
        return reject(format!("PTX references an unresolved function: {}", line.trim()));
    }
    let entry = ptx.lines().any(|line| {
        line.split_once(".entry ")
            .and_then(|(_, rest)| rest.strip_prefix(kernel_name))
            .is_some_and(|rest| rest.trim_start().starts_with('('))
    });
    if !entry {
        return reject(format!("cached PTX has no entry {kernel_name:?}"));
    }
    Ok(())
}

fn is_number(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// Does the host `clang` advertise the `nvptx64` target? Cached for the
/// lifetime of the process.
pub fn has_nvptx_target() -> bool {
    path_clang_has_target("nvptx64")
}

/// Toolkit locations searched for `ptxas` after `PATH`.
const PTXAS_FALLBACK_DIRS: &[&str] = &["/opt/cuda/bin"];

/// The CUDA toolkit's PTX assembler, with an exact persisted identity like
/// [`ClangToolchain`]'s. Assembling ahead of load skips the driver JIT (tens
/// of milliseconds per kernel on a cold `~/.nv/ComputeCache`).
#[derive(Debug, Clone)]
pub(crate) struct Ptxas {
    executable: PathBuf,
    identity: String,
}

impl Ptxas {
    /// The `ptxas` on `PATH`, then `/opt/cuda/bin`, then `$CUDA_PATH/bin`;
    /// `None` when absent, disabled with `SVOD_CUDA_PTXAS=0`, or unusable (a
    /// warning; the driver JIT still works). The version probe is persisted
    /// next to clang's so a warm process never forks it. The identity depends
    /// on the environment only, so a BEAM worker resolves the same one as its
    /// parent.
    pub(crate) fn discover(cache: Option<&ObjectCache>) -> Option<Self> {
        if std::env::var("SVOD_CUDA_PTXAS").as_deref() == Ok("0") {
            return None;
        }
        let executable = find_ptxas()?;
        match Self::identify(cache, executable.clone()) {
            Ok(ptxas) => Some(ptxas),
            Err(error) => {
                tracing::warn!(path = %executable.display(), %error, "ptxas unusable; PTX goes to the driver JIT");
                None
            }
        }
    }

    /// The identity is the `--version` line (release and build id), which
    /// names the assembler exactly; the persisted probe is keyed on the
    /// executable's path, size and mtime so a replaced binary is re-probed
    /// without hashing its tens of megabytes at every process start.
    fn identify(cache: Option<&ObjectCache>, executable: PathBuf) -> crate::Result<Self> {
        let probe = || run_probe(&executable, &["--version"]);
        let version = match cache {
            Some(cache) => {
                let meta = std::fs::metadata(&executable).jit("stat ptxas executable")?;
                let modified = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
                let key = format!("{};{};{:?}", executable.display(), meta.len(), modified);
                cache.get_or_create_probe("ptxas-version", key.as_bytes(), probe)?
            }
            None => probe()?,
        };
        let version = String::from_utf8(version).map_err(|error| crate::Error::JitCompilation {
            reason: format!("ptxas --version was not UTF-8: {error}"),
        })?;
        let identity = format!("ptxas:path={};version={}", executable.display(), version.trim());
        Ok(Self { executable, identity })
    }

    #[cfg(test)]
    pub(crate) fn fake(identity: &str) -> Self {
        Self { executable: PathBuf::from("ptxas"), identity: identity.to_string() }
    }

    pub(crate) fn identity(&self) -> &str {
        &self.identity
    }

    /// Assemble PTX text into a cubin for `arch`.
    pub(crate) fn assemble(&self, ptx: &[u8], arch: CudaArch) -> crate::Result<Vec<u8>> {
        let mut command = Command::new(&self.executable);
        command.args(ptxas_flags(arch));
        run_piped(command, ptx, &format!("ptxas (arch={arch})"))
    }
}

/// `ptxas` flags for one kernel: PTX on stdin, cubin on stdout. `ptxas` has no
/// `-` convention, so both standard streams are named by their device paths.
pub(crate) fn ptxas_flags(arch: CudaArch) -> Vec<String> {
    vec![format!("-arch={arch}"), "-o".into(), "/dev/stdout".into(), "/dev/stdin".into()]
}

fn find_ptxas() -> Option<PathBuf> {
    let fallbacks = PTXAS_FALLBACK_DIRS.iter().map(Path::new).map(|dir| dir.join("ptxas"));
    let cuda_path = std::env::var_os("CUDA_PATH").map(|root| PathBuf::from(root).join("bin/ptxas"));
    resolve_executable("ptxas")
        .ok()
        .into_iter()
        .chain(fallbacks)
        .chain(cuda_path)
        .filter(|candidate| candidate.is_file())
        .find_map(|candidate| candidate.canonicalize().ok())
}

#[cfg(test)]
#[path = "../test/unit/cuda_compile.rs"]
mod tests;
