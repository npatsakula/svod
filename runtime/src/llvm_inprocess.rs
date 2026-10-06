//! In-process LLVM IR → relocatable object compilation through libLLVM's C API.
//!
//! libLLVM is bound at runtime with `libloading`, so the build carries no LLVM
//! dependency and any LLVM ≥ 16 on the host works. The library handle is
//! process-wide; each compile thread owns its own context, target machine and
//! pass options because `LLVMContext` is not thread-safe, while independent
//! contexts compile concurrently. This replaces the ~18 ms per-kernel `clang`
//! process floor on the LLVM CPU backend.

use std::cell::{OnceCell, RefCell};
use std::ffi::{CStr, CString, OsString, c_char, c_int, c_uint, c_void};
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::sync::OnceLock;

use libloading::Library;
use snafu::ResultExt;
use tracing::debug;

use crate::error::{LibraryLoadSnafu, LibrarySymbolSnafu};
use crate::{Error, Result};

type LlvmBool = c_int;
type ContextRef = *mut c_void;
type ModuleRef = *mut c_void;
type MemoryBufferRef = *mut c_void;
type TargetRef = *mut c_void;
type TargetMachineRef = *mut c_void;
type TargetDataRef = *mut c_void;
type PassBuilderOptionsRef = *mut c_void;
type ErrorRef = *mut c_void;
type DiagnosticInfoRef = *mut c_void;
type DiagnosticHandler = extern "C" fn(DiagnosticInfoRef, *mut c_void);

const LLVM_DS_ERROR: c_int = 0;
const LLVM_RETURN_STATUS_ACTION: c_int = 2;
const LLVM_CODE_GEN_LEVEL_DEFAULT: c_int = 2;
const LLVM_RELOC_PIC: c_int = 2;
const LLVM_CODE_MODEL_DEFAULT: c_int = 0;
const LLVM_OBJECT_FILE: c_int = 1;

/// Oldest LLVM whose C API carries everything bound below (`LLVMGetVersion`,
/// `LLVMRunPasses`) and that parses the opaque-pointer IR codegen emits.
const MIN_MAJOR_VERSION: u32 = 16;
const MAX_PROBED_MAJOR_VERSION: u32 = 30;

/// Same pipeline as `clang -O2 -funroll-loops -fvectorize -fslp-vectorize`.
const PASS_PIPELINE: &CStr = c"default<O2>";

macro_rules! llvm_c_api {
    ($($symbol:ident: fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        #[allow(non_snake_case)]
        struct LlvmApi { $($symbol: unsafe extern "C" fn($($arg),*) $(-> $ret)?),* }

        impl LlvmApi {
            fn bind(library: &Library, path: &Path) -> Result<Self> {
                // SAFETY: every field is declared with its signature from the
                // LLVM-C headers, so each resolved pointer is called with the
                // ABI it was compiled for. The pointers are copied out of the
                // `Symbol` guards; the library that backs them is kept alive
                // by the owning `LlvmLibrary` for the rest of the process.
                Ok(Self { $($symbol: *unsafe {
                    library.get::<unsafe extern "C" fn($($arg),*) $(-> $ret)?>(concat!(stringify!($symbol), "\0").as_bytes())
                }.context(LibrarySymbolSnafu { path, symbol: stringify!($symbol) })?),* })
            }
        }
    };
}

llvm_c_api! {
    LLVMGetVersion: fn(*mut c_uint, *mut c_uint, *mut c_uint);
    LLVMDisposeMessage: fn(*mut c_char);
    LLVMGetHostCPUName: fn() -> *mut c_char;
    LLVMGetHostCPUFeatures: fn() -> *mut c_char;
    LLVMGetTargetFromTriple: fn(*const c_char, *mut TargetRef, *mut *mut c_char) -> LlvmBool;
    LLVMCreateTargetMachine: fn(TargetRef, *const c_char, *const c_char, *const c_char, c_int, c_int, c_int) -> TargetMachineRef;
    LLVMDisposeTargetMachine: fn(TargetMachineRef);
    LLVMCreateTargetDataLayout: fn(TargetMachineRef) -> TargetDataRef;
    LLVMDisposeTargetData: fn(TargetDataRef);
    LLVMContextCreate: fn() -> ContextRef;
    LLVMContextDispose: fn(ContextRef);
    LLVMContextSetDiagnosticHandler: fn(ContextRef, DiagnosticHandler, *mut c_void);
    LLVMGetDiagInfoSeverity: fn(DiagnosticInfoRef) -> c_int;
    LLVMGetDiagInfoDescription: fn(DiagnosticInfoRef) -> *mut c_char;
    LLVMCreateMemoryBufferWithMemoryRangeCopy: fn(*const c_char, usize, *const c_char) -> MemoryBufferRef;
    LLVMParseIRInContext: fn(ContextRef, MemoryBufferRef, *mut ModuleRef, *mut *mut c_char) -> LlvmBool;
    LLVMVerifyModule: fn(ModuleRef, c_int, *mut *mut c_char) -> LlvmBool;
    LLVMSetTarget: fn(ModuleRef, *const c_char);
    LLVMSetModuleDataLayout: fn(ModuleRef, TargetDataRef);
    LLVMDisposeModule: fn(ModuleRef);
    LLVMCreatePassBuilderOptions: fn() -> PassBuilderOptionsRef;
    LLVMDisposePassBuilderOptions: fn(PassBuilderOptionsRef);
    LLVMPassBuilderOptionsSetLoopUnrolling: fn(PassBuilderOptionsRef, LlvmBool);
    LLVMPassBuilderOptionsSetLoopVectorization: fn(PassBuilderOptionsRef, LlvmBool);
    LLVMPassBuilderOptionsSetSLPVectorization: fn(PassBuilderOptionsRef, LlvmBool);
    LLVMRunPasses: fn(ModuleRef, *const c_char, TargetMachineRef, PassBuilderOptionsRef) -> ErrorRef;
    LLVMGetErrorMessage: fn(ErrorRef) -> *mut c_char;
    LLVMDisposeErrorMessage: fn(*mut c_char);
    LLVMTargetMachineEmitToMemoryBuffer: fn(TargetMachineRef, ModuleRef, c_int, *mut *mut c_char, *mut MemoryBufferRef) -> LlvmBool;
    LLVMGetBufferStart: fn(MemoryBufferRef) -> *const c_char;
    LLVMGetBufferSize: fn(MemoryBufferRef) -> usize;
    LLVMDisposeMemoryBuffer: fn(MemoryBufferRef);
}

impl LlvmApi {
    /// Copy an LLVM-owned C string and release it.
    ///
    /// # Safety
    ///
    /// `message` must be null or a string LLVM handed out for release through
    /// `LLVMDisposeMessage`, and must not be used afterwards.
    unsafe fn take_message(&self, message: *mut c_char) -> String {
        if message.is_null() {
            return String::new();
        }
        // SAFETY: guaranteed by the caller.
        let text = unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned();
        // SAFETY: guaranteed by the caller.
        unsafe { (self.LLVMDisposeMessage)(message) };
        text
    }
}

/// A loaded libLLVM together with the host target it compiles for.
pub(crate) struct LlvmLibrary {
    api: LlvmApi,
    // Declared after `api` so its function pointers are never outlived; it
    // never drops anyway because the instance is process-global.
    _library: Library,
    target: TargetRef,
    triple: CString,
    cpu: CString,
    features: CString,
    path: PathBuf,
    version: [u32; 3],
}

// SAFETY: every field is immutable after construction; `target` is a pointer
// into LLVM's static target registry, which is valid and read-only for the
// life of the process.
unsafe impl Send for LlvmLibrary {}
unsafe impl Sync for LlvmLibrary {}

static LIBRARY: OnceLock<std::result::Result<LlvmLibrary, String>> = OnceLock::new();

/// The process-wide libLLVM, or why in-process compilation is unavailable.
///
/// The discovery outcome is memoised; `SVOD_LLVM_INPROCESS=0` is honoured on
/// every call so the clang fallback can be forced without a library present.
pub(crate) fn library() -> Result<&'static LlvmLibrary> {
    if disabled_by_env() {
        return Err(Error::LlvmError { reason: "in-process compilation disabled by SVOD_LLVM_INPROCESS=0".into() });
    }
    LIBRARY
        .get_or_init(|| {
            // The Metal private compiler loads a second libLLVM RTLD_GLOBAL that
            // crashes LLVM's verifier; only one in-process LLVM may load per
            // process. If Metal already claimed the slot, fall back to clang.
            if !svod_device::claim_inprocess_llvm("cpu-llvm") {
                return Err(
                    "in-process LLVM unavailable: the Metal compiler holds this process's in-process LLVM slot"
                        .to_string(),
                );
            }
            LlvmLibrary::discover(std::env::var_os("SVOD_LLVM_LIB")).map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|reason| Error::LlvmError { reason: reason.clone() })
}

pub(crate) fn disabled_by_env() -> bool {
    std::env::var("SVOD_LLVM_INPROCESS").as_deref() == Ok("0")
}

impl LlvmLibrary {
    /// Load the first usable candidate: `override_path` alone when given,
    /// otherwise `llvm-config --libdir`, then the loader search path (plus
    /// Homebrew prefixes on macOS).
    pub(crate) fn discover(override_path: Option<OsString>) -> Result<Self> {
        let candidates = match override_path {
            Some(path) => vec![PathBuf::from(path)],
            None => default_candidates(),
        };
        let mut failures = Vec::new();
        for path in candidates {
            // SAFETY: loading libLLVM only runs its own static initialisers,
            // which register LLVM's internal state; nothing else executes.
            let loaded = unsafe { Library::new(&path) }.context(LibraryLoadSnafu { path: &path });
            match loaded.and_then(|library| Self::bind(library, path)) {
                Ok(library) => {
                    debug!(path = %library.path.display(), version = %library.version_string(), "loaded libLLVM");
                    return Ok(library);
                }
                Err(error) => failures.push(error),
            }
        }
        Err(Error::LlvmUnavailable { failures })
    }

    fn bind(library: Library, path: PathBuf) -> Result<Self> {
        let api = LlvmApi::bind(&library, &path)?;
        let mut version = [0u32; 3];
        // SAFETY: three valid out-pointers, as the C signature requires.
        unsafe { (api.LLVMGetVersion)(&mut version[0], &mut version[1], &mut version[2]) };
        if version[0] < MIN_MAJOR_VERSION {
            return Err(Error::LlvmError {
                reason: format!("LLVM {} is older than the supported {MIN_MAJOR_VERSION}", version[0]),
            });
        }
        initialize_host_target(&library, &path)?;

        let triple = CString::new(crate::jit_loader::elf_triple()).expect("triple has no NUL");
        let mut target: TargetRef = null_mut();
        let mut message: *mut c_char = null_mut();
        // SAFETY: NUL-terminated triple and valid out-pointers; the host
        // target was initialised above so the registry lookup is meaningful.
        if unsafe { (api.LLVMGetTargetFromTriple)(triple.as_ptr(), &mut target, &mut message) } != 0 {
            // SAFETY: LLVM allocated `message` for us to release.
            let reason = unsafe { api.take_message(message) };
            return Err(Error::LlvmError {
                reason: format!("no LLVM target for {}: {reason}", triple.to_string_lossy()),
            });
        }

        // SAFETY: both return strings LLVM allocated for release via `LLVMDisposeMessage`.
        let cpu = unsafe { api.take_message((api.LLVMGetHostCPUName)()) };
        let features = unsafe { api.take_message((api.LLVMGetHostCPUFeatures)()) };
        let features = platform_target_features().iter().copied().chain([features.as_str()]).collect::<Vec<_>>();
        let cpu = CString::new(cpu).expect("cpu name has no NUL");
        let features = CString::new(features.join(",")).expect("feature string has no NUL");

        let library = Self { api, _library: library, target, triple, cpu, features, path, version };
        // A target machine is buildable exactly when the target's codegen
        // library is linked in; prove it once so per-thread sessions can rely on it.
        let probe = library.create_target_machine();
        if probe.is_null() {
            return Err(Error::LlvmError {
                reason: format!("LLVM cannot target {}", library.triple.to_string_lossy()),
            });
        }
        // SAFETY: `probe` is a live target machine we own.
        unsafe { (library.api.LLVMDisposeTargetMachine)(probe) };
        Ok(library)
    }

    fn create_target_machine(&self) -> TargetMachineRef {
        // SAFETY: `target` came from the registry for `triple`; the strings
        // are NUL-terminated and outlive the call (LLVM copies them).
        unsafe {
            (self.api.LLVMCreateTargetMachine)(
                self.target,
                self.triple.as_ptr(),
                self.cpu.as_ptr(),
                self.features.as_ptr(),
                LLVM_CODE_GEN_LEVEL_DEFAULT,
                LLVM_RELOC_PIC,
                LLVM_CODE_MODEL_DEFAULT,
            )
        }
    }

    pub(crate) fn version_string(&self) -> String {
        let [major, minor, patch] = self.version;
        format!("{major}.{minor}.{patch}")
    }

    /// Persisted producer identity: which library, which LLVM.
    pub(crate) fn toolchain_identity(&self) -> String {
        format!("library={};version={}", self.path.display(), self.version_string())
    }

    /// Persisted target identity: what the emitted code was tuned for.
    pub(crate) fn target_identity(&self) -> String {
        format!(
            "triple={};cpu={};features={}",
            self.triple.to_string_lossy(),
            self.cpu.to_string_lossy(),
            self.features.to_string_lossy()
        )
    }

    /// Persisted pipeline description, the analogue of clang's flag list.
    pub(crate) fn pipeline_flags() -> Vec<String> {
        [
            format!("passes={}", PASS_PIPELINE.to_string_lossy()),
            "loop-unrolling".into(),
            "loop-vectorization".into(),
            "slp-vectorization".into(),
            "opt-level=default".into(),
            "reloc=pic".into(),
            "code-model=default".into(),
        ]
        .into()
    }

    /// Compile one IR module to a relocatable object on this thread's session.
    pub(crate) fn compile_ir_to_object(&'static self, ir: &str) -> Result<Vec<u8>> {
        thread_local! {
            static SESSION: OnceCell<Session> = const { OnceCell::new() };
        }
        SESSION.with(|session| session.get_or_init(|| Session::new(self)).compile(ir))
    }
}

const LIBRARY_EXTENSION: &str = if cfg!(target_os = "macos") { "dylib" } else { "so" };

/// File names one LLVM major is installed under, by extension. ELF: the ≥ 18
/// upstream SONAME `libLLVM.so.18.1`, Debian's multiarch runtime SONAME
/// `libLLVM-18.so.1`, `libLLVM-18.so` (the ≤ 17 SONAME, kept by later
/// releases as a compatibility symlink) and the major-only `libLLVM.so.18`;
/// runtime packages install these without the `libLLVM.so` dev symlink.
/// Mach-O: Homebrew's `libLLVM-18.dylib` keg symlink and `libLLVM.18.dylib`.
fn versioned_names(major: u32, extension: &str) -> impl Iterator<Item = String> {
    let forms: &[&str] = match extension {
        "dylib" => &["libLLVM-{major}.dylib", "libLLVM.{major}.dylib"],
        _ => &["libLLVM.so.{major}.1", "libLLVM-{major}.so.1", "libLLVM-{major}.so", "libLLVM.so.{major}"],
    };
    forms.iter().map(move |form| form.replace("{major}", &major.to_string()))
}

/// The dev symlink, then every supported major's names, newest first.
fn candidate_names(extension: &str) -> Vec<String> {
    std::iter::once(format!("libLLVM.{extension}"))
        .chain((MIN_MAJOR_VERSION..=MAX_PROBED_MAJOR_VERSION).rev().flat_map(|major| versioned_names(major, extension)))
        .collect()
}

/// `llvm-config --libdir` when present, then the loader's default search
/// path, then Homebrew's keg-only kegs on macOS.
fn default_candidates() -> Vec<PathBuf> {
    let names = candidate_names(LIBRARY_EXTENSION);
    let mut candidates = llvm_config_libdir()
        .iter()
        .flat_map(|libdir| names.iter().map(|name| libdir.join(name)))
        .chain(names.iter().map(PathBuf::from))
        .collect::<Vec<_>>();
    if cfg!(target_os = "macos") {
        candidates.push("/opt/homebrew/opt/llvm/lib/libLLVM.dylib".into());
        candidates.extend(
            (MIN_MAJOR_VERSION..=MAX_PROBED_MAJOR_VERSION)
                .rev()
                .map(|major| PathBuf::from(format!("/opt/homebrew/opt/llvm@{major}/lib/libLLVM.dylib"))),
        );
    }
    candidates
}

fn llvm_config_libdir() -> Option<PathBuf> {
    let output = std::process::Command::new("llvm-config").arg("--libdir").output().ok()?;
    output.status.success().then(|| PathBuf::from(String::from_utf8_lossy(&output.stdout).trim()))
}

/// Register the host architecture's target, MC layer, asm printer and asm
/// parser — the `LLVMInitializeNative*` helpers are header-only, so the
/// per-architecture entry points are looked up by name.
fn initialize_host_target(library: &Library, path: &Path) -> Result<()> {
    let target = match std::env::consts::ARCH {
        "x86_64" => "X86",
        "aarch64" => "AArch64",
        "riscv64" => "RISCV",
        "powerpc64" => "PowerPC",
        "loongarch64" => "LoongArch",
        arch => return Err(Error::LlvmError { reason: format!("no LLVM target initialiser for {arch}") }),
    };
    for component in ["TargetInfo", "Target", "TargetMC", "AsmPrinter", "AsmParser"] {
        let symbol = format!("LLVMInitialize{target}{component}");
        // SAFETY: the initialisers take no arguments and return nothing; they
        // are idempotent and safe to call from any thread.
        let initialize = unsafe { library.get::<unsafe extern "C" fn()>(symbol.as_bytes()) }
            .context(LibrarySymbolSnafu { path, symbol: &symbol })?;
        unsafe { initialize() };
    }
    Ok(())
}

/// Target features the platform needs on top of the host's, mirroring
/// `platform_clang_flags`: macOS clobbers x18 on context switch.
fn platform_target_features() -> &'static [&'static str] {
    if cfg!(all(target_arch = "aarch64", target_os = "macos")) { &["+reserve-x18"] } else { &[] }
}

/// Error diagnostics LLVM reports through the context handler rather than a
/// return value (backend failures during codegen, for instance).
struct Diagnostics {
    api: &'static LlvmApi,
    errors: RefCell<Vec<String>>,
}

extern "C" fn record_diagnostic(info: DiagnosticInfoRef, diagnostics: *mut c_void) {
    // SAFETY: LLVM hands back the `Diagnostics` pointer registered with the
    // context, on the thread that owns the context; the `Session` keeps both
    // alive together, and the box gives the pointer a stable address.
    let diagnostics = unsafe { &*diagnostics.cast::<Diagnostics>() };
    // SAFETY: `info` is the live diagnostic LLVM is reporting.
    if unsafe { (diagnostics.api.LLVMGetDiagInfoSeverity)(info) } != LLVM_DS_ERROR {
        return;
    }
    // SAFETY: the description is allocated for release via `LLVMDisposeMessage`.
    let description = unsafe { diagnostics.api.take_message((diagnostics.api.LLVMGetDiagInfoDescription)(info)) };
    diagnostics.errors.borrow_mut().push(description);
}

/// Per-thread compilation state: a context plus the target machine and pass
/// options built for it. Dropped with the thread.
struct Session {
    library: &'static LlvmLibrary,
    diagnostics: Box<Diagnostics>,
    context: ContextRef,
    target_machine: TargetMachineRef,
    data_layout: TargetDataRef,
    pass_options: PassBuilderOptionsRef,
}

impl Session {
    fn new(library: &'static LlvmLibrary) -> Self {
        let api = &library.api;
        let diagnostics = Box::new(Diagnostics { api, errors: RefCell::new(Vec::new()) });
        let target_machine = library.create_target_machine();
        assert!(!target_machine.is_null(), "target machine creation was proven possible when libLLVM was bound");
        // SAFETY: plain constructors; the handler pointer stays valid because
        // the box lives inside the session that owns the context.
        unsafe {
            let context = (api.LLVMContextCreate)();
            (api.LLVMContextSetDiagnosticHandler)(
                context,
                record_diagnostic,
                std::ptr::from_ref::<Diagnostics>(&diagnostics).cast_mut().cast(),
            );
            let data_layout = (api.LLVMCreateTargetDataLayout)(target_machine);
            let pass_options = (api.LLVMCreatePassBuilderOptions)();
            (api.LLVMPassBuilderOptionsSetLoopUnrolling)(pass_options, 1);
            (api.LLVMPassBuilderOptionsSetLoopVectorization)(pass_options, 1);
            (api.LLVMPassBuilderOptionsSetSLPVectorization)(pass_options, 1);
            Self { library, diagnostics, context, target_machine, data_layout, pass_options }
        }
    }

    fn compile(&self, ir: &str) -> Result<Vec<u8>> {
        self.diagnostics.errors.borrow_mut().clear();
        let api = &self.library.api;
        let module = self.parse(ir)?;

        let mut message: *mut c_char = null_mut();
        // SAFETY: live module and a valid out-pointer for the message.
        let invalid = unsafe { (api.LLVMVerifyModule)(module.handle, LLVM_RETURN_STATUS_ACTION, &mut message) } != 0;
        // SAFETY: LLVM `strdup`s the (possibly empty) verifier output on both
        // outcomes, so it is released on both.
        let reason = unsafe { api.take_message(message) };
        if invalid {
            return Err(Error::LlvmError { reason: format!("IR verification failed: {reason}") });
        }

        // SAFETY: the module is live; `triple` is NUL-terminated and the
        // data layout is copied into the module.
        unsafe {
            (api.LLVMSetTarget)(module.handle, self.library.triple.as_ptr());
            (api.LLVMSetModuleDataLayout)(module.handle, self.data_layout);
        }

        // SAFETY: all handles are live and owned by this session/thread.
        let error = unsafe {
            (api.LLVMRunPasses)(module.handle, PASS_PIPELINE.as_ptr(), self.target_machine, self.pass_options)
        };
        if !error.is_null() {
            // SAFETY: `LLVMGetErrorMessage` consumes the error and yields a
            // string released through `LLVMDisposeErrorMessage`.
            let reason = unsafe {
                let message = (api.LLVMGetErrorMessage)(error);
                let reason = CStr::from_ptr(message).to_string_lossy().into_owned();
                (api.LLVMDisposeErrorMessage)(message);
                reason
            };
            return Err(Error::LlvmError { reason: format!("pass pipeline failed: {reason}") });
        }

        let mut buffer: MemoryBufferRef = null_mut();
        let mut message: *mut c_char = null_mut();
        // SAFETY: live handles and valid out-pointers.
        if unsafe {
            (api.LLVMTargetMachineEmitToMemoryBuffer)(
                self.target_machine,
                module.handle,
                LLVM_OBJECT_FILE,
                &mut message,
                &mut buffer,
            )
        } != 0
        {
            // SAFETY: LLVM allocated `message` for us to release.
            let reason = unsafe { api.take_message(message) };
            return Err(Error::LlvmError { reason: format!("object emission failed: {reason}") });
        }
        // SAFETY: `buffer` is a live memory buffer whose start/size describe
        // initialised bytes; it is copied out before being released.
        let object = unsafe {
            let start = (api.LLVMGetBufferStart)(buffer).cast::<u8>();
            let object = std::slice::from_raw_parts(start, (api.LLVMGetBufferSize)(buffer)).to_vec();
            (api.LLVMDisposeMemoryBuffer)(buffer);
            object
        };

        let errors = std::mem::take(&mut *self.diagnostics.errors.borrow_mut());
        if !errors.is_empty() {
            return Err(Error::LlvmError { reason: format!("diagnostics: {}", errors.join("\n")) });
        }
        if object.is_empty() {
            return Err(Error::LlvmError { reason: "LLVM produced an empty object".into() });
        }
        Ok(object)
    }

    fn parse(&self, ir: &str) -> Result<Module<'_>> {
        let api = &self.library.api;
        let mut handle: ModuleRef = null_mut();
        let mut message: *mut c_char = null_mut();
        // SAFETY: the buffer copies `ir` so the borrow ends at the call;
        // `LLVMParseIRInContext` consumes the buffer on every path.
        let failed = unsafe {
            let buffer = (api.LLVMCreateMemoryBufferWithMemoryRangeCopy)(ir.as_ptr().cast(), ir.len(), c"ir".as_ptr());
            (api.LLVMParseIRInContext)(self.context, buffer, &mut handle, &mut message) != 0
        };
        if failed {
            // SAFETY: LLVM allocated `message` for us to release.
            let reason = unsafe { api.take_message(message) };
            return Err(Error::LlvmError { reason: format!("IR parse failed: {reason}") });
        }
        Ok(Module { api, handle })
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let api = &self.library.api;
        // SAFETY: each handle is live and owned exclusively by this session;
        // the context goes last because the others were created for it.
        unsafe {
            (api.LLVMDisposePassBuilderOptions)(self.pass_options);
            (api.LLVMDisposeTargetData)(self.data_layout);
            (api.LLVMDisposeTargetMachine)(self.target_machine);
            (api.LLVMContextDispose)(self.context);
        }
    }
}

/// Owned module handle, released on every exit path.
struct Module<'a> {
    api: &'a LlvmApi,
    handle: ModuleRef,
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        // SAFETY: `handle` is a live module this guard owns.
        unsafe { (self.api.LLVMDisposeModule)(self.handle) };
    }
}

#[cfg(test)]
#[path = "test/unit/llvm_inprocess.rs"]
mod tests;
