//! LLVM JIT compilation: IR text → relocatable object → ELF loader.
//!
//! Objects come from libLLVM in process when it can be loaded (see
//! `llvm_inprocess`), otherwise from `clang -x ir -c -O2` stdin→stdout. Either
//! way the result goes through the shared JIT ELF loader; no linked LLVM
//! required.

use tracing::debug;

use crate::Result;
use crate::dispatch::KernelCif;
use crate::error::JitResultExt;

/// LLVM JIT-compiled kernel using external clang + mmap ELF loader.
pub struct LlvmKernel {
    _mmap: memmap2::MmapMut,
    fn_ptr: *const (),
    entry_point: String,
    name: String,
    var_names: Vec<String>,
    cif: KernelCif,
}

// SAFETY: Function pointer points to read-only compiled code in mmap'd memory.
// Multiple threads can call it concurrently.
unsafe impl Send for LlvmKernel {}
unsafe impl Sync for LlvmKernel {}

impl LlvmKernel {
    pub fn compile_ir_with_abi(
        ir: &str,
        entry_point: impl Into<String>,
        name: impl Into<String>,
        var_names: Vec<String>,
        abi: &[svod_device::device::AbiParamDescriptor],
    ) -> Result<Self> {
        let entry_point = entry_point.into();
        let name = name.into();
        let buffer_count = abi.iter().filter(|arg| arg.is_storage()).count();
        svod_device::device::validate_abi_descriptors(abi, buffer_count, &var_names)?;
        debug!(kernel.name = %name, ir.length = ir.len(), "Compiling LLVM IR");
        if let Ok(dir) = std::env::var("SVOD_DUMP_LLVM_IR") {
            let path = std::path::Path::new(&dir).join(format!("{name}.ll"));
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(&path, ir);
        }
        if let Ok(dir) = std::env::var("SVOD_DUMP_POST_O2_IR") {
            let _ = std::fs::create_dir_all(&dir);
            if let Some(post_ir) = compile_ir_to_post_o2_text(ir) {
                let path = std::path::Path::new(&dir).join(format!("{name}.post.ll"));
                let _ = std::fs::write(&path, post_ir);
            }
        }
        let obj = compile_ir_to_object(ir)?;
        Self::load_object_with_abi(&obj, entry_point, name, var_names, abi)
    }

    pub fn load_object_with_abi(
        object: &[u8],
        entry_point: impl Into<String>,
        name: impl Into<String>,
        var_names: Vec<String>,
        abi: &[svod_device::device::AbiParamDescriptor],
    ) -> Result<Self> {
        let entry_point = entry_point.into();
        let name = name.into();
        let buffer_count = abi.iter().filter(|arg| arg.is_storage()).count();
        svod_device::device::validate_abi_descriptors(abi, buffer_count, &var_names)?;
        crate::clang::validate_relocatable_object(object, &entry_point)?;
        let (fn_ptr, mmap) = crate::jit_loader::jit_load(object, &entry_point)?;
        let cif = KernelCif::from_abi(abi)?;
        debug!(kernel.name = %name, "LLVM kernel object loaded");
        Ok(Self { _mmap: mmap, fn_ptr, entry_point, name, var_names, cif })
    }

    /// Compile a RenderedKernel from the codegen crate.
    pub fn compile(kernel: &svod_codegen::RenderedKernel) -> Result<Self> {
        Self::compile_ir_with_abi(&kernel.code, &kernel.name, &kernel.name, kernel.var_names.clone(), &kernel.abi)
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

    /// Execute the kernel with buffer pointers and variable values.
    ///
    /// # Safety
    ///
    /// Caller must ensure buffer pointers are valid/aligned and `vals` length
    /// matches `var_names`.
    pub unsafe fn execute_with_vals(&self, buffers: &[*mut u8], vals: &[i64]) -> Result<()> {
        debug!(
            kernel.entry_point = %self.entry_point,
            kernel.num_buffers = buffers.len(),
            kernel.num_vals = vals.len(),
            "Executing LLVM kernel"
        );

        unsafe { self.cif.dispatch(self.fn_ptr, buffers, vals, None)? };

        Ok(())
    }

    pub(crate) fn cif(&self) -> &KernelCif {
        &self.cif
    }
}

/// Compile LLVM IR text to a relocatable object with whichever producer is
/// available.
fn compile_ir_to_object(ir: &str) -> Result<Vec<u8>> {
    llvm_object_producer(None)?.0.compile(ir)
}

/// Where LLVM-backend objects come from. Both emit `<arch>-none-unknown-elf`
/// relocatable objects for the JIT ELF loader.
pub(crate) enum LlvmObjectProducer {
    InProcess(&'static crate::llvm_inprocess::LlvmLibrary),
    Clang { toolchain: crate::clang::ClangToolchain, flags: Vec<String> },
}

impl LlvmObjectProducer {
    pub(crate) fn compile(&self, ir: &str) -> Result<Vec<u8>> {
        match self {
            Self::InProcess(library) => library.compile_ir_to_object(ir),
            Self::Clang { toolchain, flags } => compile_ir_to_object_with(toolchain, ir, flags),
        }
    }
}

/// Pick the object producer and its persisted identity: libLLVM in process
/// when it loads (and `SVOD_LLVM_INPROCESS` is not `0`), else the clang
/// subprocess. The identities differ so cached objects never cross producers.
pub(crate) fn llvm_object_producer(
    cache: Option<&crate::object_cache::ObjectCache>,
) -> Result<(LlvmObjectProducer, crate::object_cache::CompilerIdentity)> {
    use crate::llvm_inprocess::LlvmLibrary;
    use crate::object_cache::{CompilerIdentity, OBJECT_CACHE_SCHEMA};

    let abi = format!(
        "svod-llvm-kernel-abi-v1;pointer-width={};endian={}",
        usize::BITS,
        if cfg!(target_endian = "little") { "little" } else { "big" }
    );
    let identity =
        |backend: &str, target_architecture: String, toolchain: String, flags: Vec<String>| CompilerIdentity {
            schema: OBJECT_CACHE_SCHEMA,
            backend: backend.into(),
            target_architecture,
            toolchain,
            flags,
            abi: abi.clone(),
            object_format: "elf-relocatable-svod-jit-loader-v1".into(),
        };
    match crate::llvm_inprocess::library() {
        Ok(library) => {
            debug!(version = %library.version_string(), "LLVM backend compiles in process through libLLVM");
            let identity = identity(
                "cpu-llvm-inprocess",
                library.target_identity(),
                library.toolchain_identity(),
                LlvmLibrary::pipeline_flags(),
            );
            Ok((LlvmObjectProducer::InProcess(library), identity))
        }
        Err(error) if crate::llvm_inprocess::disabled_by_env() => {
            debug!(%error, "LLVM backend compiles through the clang subprocess");
            clang_producer(cache, identity)
        }
        Err(error) => {
            tracing::warn!(%error, "LLVM backend falls back to the clang subprocess");
            clang_producer(cache, identity)
        }
    }
}

fn clang_producer(
    cache: Option<&crate::object_cache::ObjectCache>,
    identity: impl FnOnce(&str, String, String, Vec<String>) -> crate::object_cache::CompilerIdentity,
) -> Result<(LlvmObjectProducer, crate::object_cache::CompilerIdentity)> {
    let toolchain = crate::clang::ClangToolchain::discover(cache)?;
    let flags = llvm_object_flags();
    let target_architecture = toolchain.target_identity(cache, &flags)?;
    let identity = identity("cpu-llvm-clang", target_architecture, toolchain.identity().into(), flags.clone());
    Ok((LlvmObjectProducer::Clang { toolchain, flags }, identity))
}

pub(crate) fn llvm_object_flags() -> Vec<String> {
    let mut args = vec![
        "-x",
        "ir",
        "-c",
        "-O2",
        "-march=native",
        "-fPIC",
        "-fno-math-errno",
        "-fno-stack-protector",
        "-funroll-loops",
        "-fvectorize",
        "-fslp-vectorize",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    args.push(crate::jit_loader::elf_target_triple());
    args.extend(crate::jit_loader::platform_clang_flags().iter().map(|flag| (*flag).to_string()));
    args.extend(["-", "-o", "-"].map(str::to_string));
    args
}

pub(crate) fn compile_ir_to_object_with(
    toolchain: &crate::clang::ClangToolchain,
    ir: &str,
    args: &[String],
) -> Result<Vec<u8>> {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = toolchain
        .command()
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .jit("spawn clang for IR (is clang installed?)")?;

    child.stdin.take().expect("stdin was piped").write_all(ir.as_bytes()).jit("write IR to clang stdin")?;

    let output = child.wait_with_output().jit("wait for clang (IR)")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(crate::Error::JitCompilation { reason: format!("clang IR compilation failed:\n{stderr}") });
    }

    if output.stdout.is_empty() {
        return Err(crate::Error::JitCompilation { reason: "clang produced empty output from IR".to_string() });
    }

    Ok(output.stdout)
}

/// Run the same `-O2` LLVM pass pipeline as the JIT compile but emit
/// textual LLVM IR. Returns `None` on compile failure (silent — this
/// is a diagnostic-only path, never load-bearing).
fn compile_ir_to_post_o2_text(ir: &str) -> Option<String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut args = vec![
        "-x",
        "ir",
        "-S",
        "-emit-llvm",
        "-O2",
        "-march=native",
        "-fno-math-errno",
        "-funroll-loops",
        "-fvectorize",
        "-fslp-vectorize",
    ];
    args.extend_from_slice(crate::jit_loader::platform_clang_flags());
    args.extend_from_slice(&["-", "-o", "-"]);

    let mut child = Command::new("clang")
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(ir.as_bytes()).ok()?;
    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

#[cfg(test)]
#[path = "test/unit/llvm.rs"]
mod tests;
