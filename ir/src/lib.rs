//! Intermediate Representation (IR) for the Svod compiler.
//!
//! This crate defines the core IR data structures and operations used throughout
//! the Svod compiler pipeline.
//!
//! # Module Organization
//!
//! - [`types`] - Fundamental type definitions (ConstValue, operation types, etc.)
//! - [`op`] - Operation enum defining all IR operations
//! - [`uop`] - UOp (micro-operation) struct and implementation
//! - [`uop::constructors`] - UOp constructor methods by semantic category
//! - [`origin`] - Hierarchical origin tracking for IR nodes
//! - [`indexing`] - Multi-dimensional indexing support
//! - [`error`] - Error types and result handling
//! - [`shape`] - Shape inference utilities
//! - [`sint`] - Symbolic integers

// Make this crate available as `svod_ir` for proc-macro generated code
extern crate self as svod_ir;

// Module declarations
pub mod decompositions;
pub mod dtype_rule;
pub mod error;
pub mod indexing;
pub mod op;
pub mod opt;
pub mod origin;
pub mod prelude;
pub mod shape;
pub mod sint;
pub mod types;
pub mod uop;

#[macro_use]
pub mod pattern;
pub mod rewrite;

#[cfg(any(test, feature = "proptest"))]
pub mod test;

// Re-exports at crate root for ergonomic access.
pub use dtype_rule::dtype_from_op;
pub use error::{Error, IndexTypeMismatchSnafu, Result};
pub use indexing::IndexSpec;
pub use op::{Op, ops};
pub use opt::{Opt, OptArg, OptOps};
pub use origin::{Origin, OriginFrame, OriginId, OriginScope, OriginSet, SourceLocation};
pub use sint::{IntoShrinkRange, SInt, ShrinkRange, sint_max, sint_min, sint_prod};
pub use types::{
    AddrSpace, AxisId, AxisType, BINARY_STAGE_IDENTITY_VERSION, BarrierFence, BinaryOp, BinaryStageIdentity,
    BufferizeOpts, CallInfo, ConstValue, ConstValueHash, ContiguousHint, CustomFunctionKind, InsArg, KernelInfo,
    ParamArg, ProgramInfo, ReduceOp, RendererDevice, RendererOps, SOURCE_STAGE_IDENTITY_VERSION, SourceStageIdentity,
    StageAbiParam, StageAbiParamKind, StageDigest, TernaryOp, UnaryOp, WmmaMetadata, WmmaUpcastAxes, to_function_name,
};
pub use uop::{
    CanonicalArg, CanonicalAxis, CanonicalConst, CanonicalDType, CanonicalGraph, CanonicalNode, CanonicalProgramValue,
    CanonicalShapeDim, CanonicalVerboseNode, IntoUOp, OptimizerWireGraph, UOp, UOpKey, compute_ops_estimate,
    dump_canonical_stage,
};

// Re-export pattern matching and rewriting infrastructure
pub use pattern::{Matcher, RewriteResult, TypedPatternMatcher};
pub use rewrite::{
    graph_rewrite, graph_rewrite_bottom_up_preserve_calls, graph_rewrite_preserve_calls,
    graph_rewrite_walk_preserve_calls, graph_rewrite_with_bpm_preserve_calls,
};

// Re-export external types for convenience
pub use svod_dtype::DType;
pub use svod_dtype::DeviceSpec;
