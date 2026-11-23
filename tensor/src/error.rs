use snafu::Snafu;
use svod_ir::shape::Shape;

/// The cause of a tensor-API failure.
///
/// Carried indirectly by [`Error`], which is what [`Result`] and every public
/// method surface; the snafu context selectors (`XSnafu`) build this type and
/// `?` boxes it on the way out.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum ErrorKind {
    // =========================================================================
    // IR Layer Errors
    // =========================================================================
    #[snafu(display("IR operation error: {source}"))]
    UOp { source: svod_ir::Error },

    // =========================================================================
    // Shape Errors
    // =========================================================================
    #[snafu(display("Tensor shape is unknown (symbolic or not yet inferred)"))]
    ShapeUnknown,

    #[snafu(display("Operation '{operation}' does not support symbolic shapes"))]
    SymbolicShapeUnsupported { operation: String },

    #[snafu(display("dimension {axis} is symbolic ({dim}), a concrete size is required"))]
    NonConstDim { axis: isize, dim: svod_ir::SInt },

    #[snafu(display("Axis {axis} is out of range for tensor with {ndim} dimensions"))]
    AxisOutOfRange { axis: isize, ndim: usize },

    #[snafu(display("Permutation length mismatch: expected {expected} axes, got {got}"))]
    PermutationLengthMismatch { expected: usize, got: usize },

    #[snafu(display("Invalid permutation: axes {axes:?} is not a valid permutation"))]
    InvalidPermutation { axes: Vec<isize> },

    #[snafu(display("Multiple -1 dimensions in reshape are not allowed"))]
    MultipleInferDimensions,

    #[snafu(display("Negative dimension {dim} is not allowed (except -1 for inference)"))]
    NegativeDimension { dim: isize },

    #[snafu(display("Reshape size mismatch during {operation}"))]
    ReshapeSizeMismatch { operation: String },

    #[snafu(display(
        "Expand dimension mismatch: current shape has {current_dims} dims, target has {target_dims} dims"
    ))]
    ExpandDimensionMismatch { current_dims: usize, target_dims: usize },

    #[snafu(display("Cannot squeeze dimension {dim}: size is {size}, not 1"))]
    SqueezeDimensionNotOne { dim: usize, size: usize },

    // =========================================================================
    // NN Input Validation Errors
    // =========================================================================
    #[snafu(display("{op} requires exactly {expected}D input, got {actual}D"))]
    NdimExact { op: &'static str, expected: usize, actual: usize },

    #[snafu(display("{op} requires at least {min}D input, got {actual}D"))]
    NdimMinimum { op: &'static str, min: usize, actual: usize },

    #[snafu(display("{op}: {lhs_name} ({lhs}) must be divisible by {rhs_name} ({rhs})"))]
    Divisibility { op: &'static str, lhs_name: &'static str, lhs: usize, rhs_name: &'static str, rhs: usize },

    #[snafu(display("{op}: exactly one of {options} must be provided"))]
    ExclusiveParams { op: &'static str, options: &'static str },

    #[snafu(display("state dict has no key '{key}'"))]
    MissingKey { key: String },

    #[snafu(display("{op}: {param} = {value} is invalid, expected {constraint}"))]
    ParamRange { op: &'static str, param: &'static str, value: String, constraint: &'static str },

    // =========================================================================
    // Reduction Errors
    // =========================================================================
    #[snafu(display("Cannot specify both 'dtype' and 'promote=true' in reduction operation"))]
    ConflictingReductionOptions,

    // =========================================================================
    // Matrix Multiplication Errors
    // =========================================================================
    #[snafu(display(
        "Matrix multiplication requires tensors with at least 1 dimension, got lhs: {lhs_dims}D, rhs: {rhs_dims}D"
    ))]
    DotDimensionError { lhs_dims: usize, rhs_dims: usize },

    #[snafu(display(
        "Matrix multiplication shape mismatch: cannot multiply shapes {lhs_shape:?} and {rhs_shape:?} (contraction dimension mismatch)"
    ))]
    DotShapeMismatch { lhs_shape: Box<Shape>, rhs_shape: Box<Shape> },

    // =========================================================================
    // Broadcasting Errors
    // =========================================================================
    #[snafu(display(
        "Cannot broadcast to fewer dimensions: tensor has {from_dims} dimensions, target has {to_dims} dimensions"
    ))]
    BroadcastFewerDimensions { from_dims: usize, to_dims: usize },

    #[snafu(display(
        "Incompatible dimension {dim} for broadcasting: cannot broadcast size {from_size} to size {to_size}"
    ))]
    BroadcastIncompatible { dim: usize, from_size: usize, to_size: usize },

    // =========================================================================
    // Codegen Errors (from device traits that wrap codegen)
    // =========================================================================
    #[snafu(display("Failed to render kernel: {source}"))]
    RenderKernel { source: svod_device::Error },

    #[snafu(display("Failed to compile kernel: {source}"))]
    CompileKernel { source: svod_device::Error },

    // =========================================================================
    // Schedule/Pipeline Errors
    // =========================================================================
    #[snafu(display("Rangeify failed: {source}"))]
    Rangeify { source: svod_ir::Error },

    #[snafu(display("Kernel graph failed: {source}"))]
    KernelGraph {
        #[snafu(source(from(svod_schedule::KernelGraphError, Box::new)))]
        source: Box<svod_schedule::KernelGraphError>,
    },

    #[snafu(display("Optimization error: {source}"))]
    Optimize { source: svod_schedule::OptError },

    #[snafu(display("No kernels found after scheduling pipeline"))]
    NoKernelsFound,

    #[snafu(display("Schedule contains dependency cycles"))]
    DependencyCycles,

    #[snafu(display("Empty schedule"))]
    EmptySchedule,

    #[snafu(display("Batch output count mismatch: expected {expected}, got {actual}"))]
    BatchOutputMismatch { expected: usize, actual: usize },

    #[snafu(display("Expected CALL operation"))]
    ExpectedCallableOp,

    #[snafu(display("CALL {call_id} MSTACK source {source_index} has no lanes"))]
    MultiEmptyLanes { call_id: u64, source_index: usize },

    #[snafu(display("CALL {call_id} MSTACK source {source_index} has {actual} lanes, expected {expected}"))]
    MultiLaneCountMismatch { call_id: u64, source_index: usize, expected: usize, actual: usize },

    #[snafu(display("MSELECT {source_id} lane {device_index} is out of bounds for {lane_count} lanes"))]
    MultiSelectOutOfBounds { source_id: u64, device_index: usize, lane_count: usize },

    #[snafu(display("CALL {call_id} has unsupported MULTI form: {details}"))]
    MultiUnsupportedForm { call_id: u64, details: String },

    #[snafu(display("CALL {call_id} MSTACK source {source_index} lane {lane} cannot contain a SLICE alias"))]
    MultiLaneSliceAlias { call_id: u64, source_index: usize, lane: usize },

    #[snafu(display("CALL {call_id} lane {lane} mixes device endpoints {expected} and {actual}"))]
    MultiLaneDeviceMismatch { call_id: u64, lane: usize, expected: String, actual: String },

    #[snafu(display("CALL {call_id} DEVICE extent must be a static integer"))]
    MultiDeviceExtentNotStatic { call_id: u64 },

    #[snafu(display("CALL {call_id} DEVICE extent {actual} does not match MSTACK lane count {expected}"))]
    MultiDeviceExtentMismatch { call_id: u64, expected: usize, actual: i64 },

    #[snafu(display(
        "CALL {call_id} fixed binding conflict for '{name}': existing value {existing}, incoming value {incoming}"
    ))]
    MultiBindingConflict { call_id: u64, name: String, existing: i64, incoming: i64 },

    // =========================================================================
    // Runtime Errors
    // =========================================================================
    #[snafu(display("Execution failed: {source}"))]
    Execution { source: svod_runtime::Error },

    #[snafu(display("Failed to create program: {source}"))]
    CreateProgram { source: svod_device::Error },

    #[snafu(display("Failed to get device: {source}"))]
    DeviceFactory { source: svod_runtime::Error },

    #[snafu(display("Buffer for UOp {} not found in registry", uop_id))]
    BufferNotFound { uop_id: u64 },

    #[snafu(display("Device error: {source}"))]
    Device { source: svod_device::Error },

    // =========================================================================
    // Type Errors
    // =========================================================================
    #[snafu(display("Expected Ptr dtype for {context}, got {actual:?}"))]
    ExpectedPtrDtype { context: &'static str, actual: svod_dtype::DType },

    #[snafu(display("Buffer Ptr dtype has no size"))]
    BufferPtrNoSize,

    #[snafu(display("Tensor has no buffer (unrealized tensor?)"))]
    NoBuffer,

    #[snafu(display("Tensor has no shape"))]
    NoShape,

    #[snafu(display("Shape mismatch for '{context}': expected {expected}, got {actual}"))]
    ShapeMismatch { context: String, expected: String, actual: String },

    #[snafu(display("IR construction error: {details}"))]
    IrConstruction { details: String },

    #[snafu(display("Disk access failed for '{path}': {source}"))]
    Disk { source: std::io::Error, path: String },

    #[snafu(display("Invalid ProgramSpec at {stage}: {source}"))]
    ProgramSpec { source: svod_device::Error, stage: String },

    #[snafu(display("Type mismatch: expected {expected:?}, got {actual:?}"))]
    TypeMismatch { expected: svod_dtype::DType, actual: svod_dtype::DType },

    #[snafu(display("{op} requires floating-point dtype for {arg}, got {dtype:?}"))]
    FloatDTypeRequired { op: &'static str, arg: &'static str, dtype: svod_dtype::DType },

    #[snafu(display("{op} requires signed integer dtype for {arg}, got {dtype:?}"))]
    SignedIntegerDTypeRequired { op: &'static str, arg: &'static str, dtype: svod_dtype::DType },

    #[snafu(display("Failed to create ndarray: {source}"))]
    NdarrayShape { source: ndarray::ShapeError },

    // =========================================================================
    // Variable Errors
    // =========================================================================
    #[snafu(display("Variable '{name}' value {val} out of range [{min}, {max}]"))]
    VariableOutOfRange { name: String, val: i64, min: i64, max: i64 },

    #[snafu(display("Cannot read data from tensor with symbolic shape — reduce or slice to concrete shape first"))]
    SymbolicShape,

    #[snafu(display("BEAM compile worker: {source}"))]
    BeamWorker { source: BeamWorker },
}

/// Failures of the spawned BEAM candidate compiler.
///
/// The pool treats most of these as a dropped candidate rather than a fatal
/// error, so they carry the structure the caller acts on instead of a rendered
/// message.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum BeamWorker {
    #[snafu(display("start BEAM worker {path}: {source}"))]
    SpawnWorker { source: std::io::Error, path: String },

    #[snafu(display("BEAM worker is unavailable: {reason}"))]
    WorkerUnavailable { reason: String },

    #[snafu(display("BEAM worker frame ({what}): {source}"))]
    Frame { source: std::io::Error, what: &'static str },

    #[snafu(display("BEAM worker protocol version {actual}, expected {expected}"))]
    ProtocolMismatch { expected: u32, actual: u32 },

    #[snafu(display("BEAM worker returned candidate {got}, expected {expected:?}"))]
    WorkerMisorder { got: usize, expected: Option<usize> },

    #[snafu(display("BEAM candidate {stage}: {reason}"))]
    CompileStage { stage: &'static str, reason: String },
}

impl BeamWorker {
    /// `map_err` adapter for a pipeline stage whose upstream error is only ever
    /// reported, never matched on.
    pub(crate) fn at<E: std::fmt::Display>(stage: &'static str) -> impl Fn(E) -> Self {
        move |error| Self::CompileStage { stage, reason: error.to_string() }
    }
}

/// A boxed [`ErrorKind`].
///
/// Tensor methods thread `Result<T>` through deeply nested builders, so the
/// `Err` payload is kept pointer-sized instead of growing every `Result` in the
/// crate to the size of the widest variant.
pub struct Error(Box<ErrorKind>);

impl Error {
    /// The wrapped cause; also reachable through [`Deref`](std::ops::Deref).
    pub fn kind(&self) -> &ErrorKind {
        &self.0
    }

    /// Unwrap the boxed cause, e.g. to match it by value.
    pub fn into_kind(self) -> ErrorKind {
        *self.0
    }
}

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Self {
        Self(Box::new(kind))
    }
}

impl std::ops::Deref for Error {
    type Target = ErrorKind;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// A [`Result`] carrying the cause unboxed, as the snafu context selectors
/// produce it. Useful where an iterator adaptor has to name the error type
/// explicitly; `?` boxes it into [`Error`].
pub type KindResult<T> = std::result::Result<T, ErrorKind>;

#[cfg(test)]
#[path = "test/unit/error.rs"]
mod tests;
