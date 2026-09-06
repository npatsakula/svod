//! Error types for code generation.

use snafu::Snafu;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors that can occur during code generation.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    /// Unsupported operation in the UOp graph.
    #[snafu(display("Unsupported operation: {op:?}"))]
    UnsupportedOp { op: String },

    /// Invalid UOp graph structure.
    #[snafu(display("Invalid UOp graph: {reason}"))]
    InvalidGraph { reason: String },

    /// Type error during code generation.
    #[snafu(display("Type error: {reason}"))]
    TypeError { reason: String },

    /// LLVM-specific error.
    #[snafu(display("LLVM error: {reason}"))]
    LlvmError { reason: String },

    /// Missing required information.
    #[snafu(display("Missing {what}"))]
    Missing { what: String },

    /// A CUSTOM body references a vendor intrinsic the active target cannot
    /// lower (clang would emit it as an extern call that only fails in the
    /// device assembler).
    #[snafu(display("intrinsic @{intrinsic} is not available on the {target} target"))]
    ForeignIntrinsic { intrinsic: String, target: String },

    /// Invalid configuration or parameters.
    #[snafu(display("Invalid configuration: {reason}"))]
    InvalidConfig { reason: String },

    /// Error from IR layer.
    #[snafu(display("IR error: {source}"))]
    IrError {
        #[snafu(source)]
        source: svod_ir::Error,
    },
}
