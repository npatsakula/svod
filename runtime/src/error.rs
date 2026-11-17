//! Error types for runtime execution.

use std::path::PathBuf;

use snafu::Snafu;
use svod_device::error::describe;

/// Result type for runtime operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors that can occur during runtime execution.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    /// Codegen error occurred.
    #[snafu(display("Codegen error: {source}"))]
    Codegen { source: svod_codegen::Error },

    /// JIT compilation failed with no recoverable underlying error (pure
    /// message: missing tool, malformed output, unsupported relocation, …).
    #[snafu(display("JIT compilation failed: {reason}"))]
    JitCompilation { reason: String },

    /// JIT compilation failed; carries the underlying I/O / `object` / library
    /// error so the real chain isn't flattened into a string. `context` names
    /// the failing step (e.g. "spawn clang", "parse ELF"). The source is boxed
    /// because the largest underlying error (`object::Error`) dwarfs the rest.
    #[snafu(display("JIT compilation failed ({context}): {source}"))]
    Jit {
        #[snafu(source(from(JitSource, Box::new)))]
        source: Box<JitSource>,
        context: &'static str,
    },

    /// Function not found in module.
    #[snafu(display("Function '{name}' not found in module"))]
    FunctionNotFound { name: String },

    /// Buffer allocation failed.
    #[snafu(display("Buffer allocation failed: {reason}"))]
    BufferAllocation { reason: String },

    /// Invalid buffer size.
    #[snafu(display("Invalid buffer size: {size}"))]
    InvalidBufferSize { size: usize },

    /// Execution error with no recoverable underlying error (pure message:
    /// validation failures, out-of-range indices, var-bounds violations, …).
    #[snafu(display("Execution error: {reason}"))]
    Execution { reason: String },

    /// A prior committed HCQ epoch failed after reserving timeline values.
    /// Retrying could wait forever on a completion that was never published.
    #[snafu(display("Execution plan is poisoned after a failed HCQ epoch: {reason}"))]
    PlanPoisoned { reason: String },

    /// Execution failed while dispatching a specific operation; carries the
    /// underlying device error and the offending op for context.
    #[snafu(display("Execution failed ({context}): {source}"))]
    Exec { source: svod_device::Error, context: String },

    /// LLVM error.
    #[snafu(display("LLVM error: {reason}"))]
    LlvmError { reason: String },

    /// No libLLVM candidate could be loaded and bound; one failure per candidate.
    #[snafu(display("no usable libLLVM: {}", failures.iter().map(ToString::to_string).collect::<Vec<_>>().join("; ")))]
    LlvmUnavailable { failures: Vec<Error> },

    /// A shared library could not be loaded.
    #[snafu(display("cannot load library {}: {}", path.display(), describe(source)))]
    LibraryLoad { source: libloading::Error, path: PathBuf },

    /// A symbol could not be resolved in a loaded shared library.
    #[snafu(display("cannot resolve symbol `{symbol}` in {}: {}", path.display(), describe(source)))]
    LibrarySymbol { source: libloading::Error, path: PathBuf, symbol: String },

    /// Unsupported device type.
    #[snafu(display("Unsupported device type: {device}"))]
    UnsupportedDevice { device: String },

    /// Reserved-but-unsupported runtime feature.
    #[snafu(display("Unsupported runtime feature {kind}: {reason}"))]
    Unsupported { kind: String, reason: String },

    /// A thread trace could not be read back or analyzed.
    #[snafu(display("SQTT: {source}"))]
    #[snafu(context(false))]
    Sqtt { source: crate::sqtt::Error },

    /// Device error (from svod_device crate).
    #[snafu(display("Device error: {source}"))]
    #[snafu(context(false))]
    Device { source: svod_device::Error },
}

/// Underlying causes a [`Error::Jit`] can wrap. Boxed in the variant because
/// the largest source (`object::Error`) dwarfs the rest.
#[derive(Debug, Snafu)]
pub enum JitSource {
    /// I/O failure spawning/driving the compiler subprocess or touching mmap.
    #[snafu(display("{source}"))]
    Io { source: std::io::Error },

    /// ELF parsing/reading via the `object` crate.
    #[snafu(display("{source}"))]
    Object { source: object::Error },

    /// Symbol name contained an interior NUL.
    #[snafu(display("{source}"))]
    Nul { source: std::ffi::NulError },

    /// Shared-library load/symbol resolution (`dlopen-fallback` path only).
    #[cfg(feature = "dlopen-fallback")]
    #[snafu(display("{}", describe(source)))]
    Lib { source: libloading::Error },
}

// `From<underlying> for JitSource` backs the `JitResultExt::jit` helper below,
// which boxes the converted source into the `Jit` variant.
impl From<std::io::Error> for JitSource {
    fn from(source: std::io::Error) -> Self {
        JitSource::Io { source }
    }
}

impl From<object::Error> for JitSource {
    fn from(source: object::Error) -> Self {
        JitSource::Object { source }
    }
}

impl From<std::ffi::NulError> for JitSource {
    fn from(source: std::ffi::NulError) -> Self {
        JitSource::Nul { source }
    }
}

#[cfg(feature = "dlopen-fallback")]
impl From<libloading::Error> for JitSource {
    fn from(source: libloading::Error) -> Self {
        JitSource::Lib { source }
    }
}

/// Ergonomic `Result` extension: `.jit("step")` converts any [`JitSource`]-
/// convertible error into [`Error::Jit`] with a static context label, keeping
/// the underlying error as a real `source`.
pub trait JitResultExt<T> {
    /// Wrap the error into [`Error::Jit`] tagged with `context`.
    fn jit(self, context: &'static str) -> Result<T>;
}

impl<T, E: Into<JitSource>> JitResultExt<T> for std::result::Result<T, E> {
    fn jit(self, context: &'static str) -> Result<T> {
        self.map_err(|e| Error::Jit { source: Box::new(e.into()), context })
    }
}
