use smallvec::SmallVec;
use snafu::Snafu;
use svod_dtype::DType;
use svod_dtype::DeviceSpec;

use crate::{BinaryOp, ConstValue, UnaryOp, shape::Shape};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Clone, PartialEq, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    /// DType mismatch in binary operation.
    #[snafu(display("dtype mismatch: cannot perform operation on {lhs:?} and {rhs:?}"))]
    DTypeMismatch { lhs: DType, rhs: DType },

    /// Type promotion failed - no common type.
    #[snafu(display("type promotion failed: no common type for {lhs:?} and {rhs:?}"))]
    TypePromotionFailed { lhs: DType, rhs: DType },

    /// Invalid dtype for operation (e.g., bitwise on float).
    #[snafu(display("invalid dtype for operation: operation {operation:?}; dtype {dtype:?}"))]
    InvalidDTypeForUnaryOp { operation: UnaryOp, dtype: DType },

    /// Invalid dtype for operation (e.g., bitwise on float).
    #[snafu(display("invalid dtype for operation: operation {operation:?}; dtypes {dtypes:?}"))]
    InvalidDTypeForBinaryOp { operation: BinaryOp, dtypes: SmallVec<[DType; 2]> },

    /// Void dtype cannot be used in operations.
    #[snafu(display("void dtype cannot be used in operations"))]
    VoidTypeInOp,

    /// Index parameter must have Index dtype.
    #[snafu(display("index parameter must have Index dtype, got {actual:?}"))]
    IndexTypeMismatch { actual: DType },

    /// Division by zero.
    #[snafu(display("division by zero"))]
    DivisionByZero,

    /// Reshape size mismatch.
    #[snafu(display("reshape size mismatch: input size {input_size} != output size {output_size}"))]
    ReshapeSizeMismatch { input_size: usize, output_size: usize },

    /// Shrink bounds violation.
    #[snafu(display(
        "shrink bounds violation: dimension {dim} has range [{begin}, {end}) but shape size is {shape_size}",
    ))]
    ShrinkBoundsViolation { dim: usize, begin: usize, end: usize, shape_size: usize },

    /// Bind value out of range.
    #[snafu(display("bind value {value} is outside valid range [{min}, {max}]"))]
    BindValueOutOfRange { value: i64, min: i64, max: i64 },

    /// Index out of bounds.
    #[snafu(display("index out of bounds"))]
    IndexOutOfBounds,

    /// Expand dimension count mismatch.
    #[snafu(display("expand dimension mismatch: input has {input_dims} dimensions, output has {output_dims}"))]
    ExpandDimensionMismatch { input_dims: usize, output_dims: usize },

    /// Expand invalid dimension (can only expand dimensions of size 1).
    #[snafu(display(
        "expand invalid: dimension {dim} has size {input} but needs to expand to {output} (can only expand from 1)",
    ))]
    ExpandInvalidDimension { dim: usize, input: usize, output: usize },

    /// Permute has invalid permutation.
    #[snafu(display("invalid permutation {permutation:?}: expected permutation of 0..{expected_dims}"))]
    PermuteInvalidPermutation { permutation: Vec<usize>, expected_dims: usize },

    /// Pad has negative padding value.
    #[snafu(display(
        "pad has negative value: dimension {dim} has padding ({begin}, {end}) but padding must be non-negative",
    ))]
    PadNegativeValue { dim: usize, begin: isize, end: isize },

    /// Pad dimension count mismatch.
    #[snafu(display("pad dimension mismatch: padding has {padding_dims} dimensions but shape has {shape_dims}"))]
    PadDimensionMismatch { padding_dims: usize, shape_dims: usize },

    /// Flip specification invalid.
    #[snafu(display("flip specification invalid: expected {expected_dims} dimensions, got {got_dims}"))]
    FlipInvalidSpec { expected_dims: usize, got_dims: usize },

    /// Reduce axis invalid.
    #[snafu(display("reduce axis {axis} is invalid for shape with {shape_dims} dimensions"))]
    ReduceAxisInvalid { axis: i32, shape_dims: usize },

    /// Shaped reduction removes more leading axes than its source has.
    #[snafu(display("reduce num_axes {num_axes} is invalid for shape with {shape_dims} dimensions"))]
    ReduceInvalidNumAxes { num_axes: usize, shape_dims: usize },

    /// Shape mismatch in elementwise operation.
    #[snafu(display("shape mismatch: cannot perform elementwise operation on shapes {lhs_shape:?} and {rhs_shape:?}"))]
    ShapeMismatch { lhs_shape: Vec<usize>, rhs_shape: Vec<usize> },

    /// Shape mismatch in binary operation.
    #[snafu(display("Shape mismatch in {op:?}: {lhs:?} vs {rhs:?}"))]
    BinaryShapeMismatch { op: crate::types::BinaryOp, lhs: Box<Shape>, rhs: Box<Shape> },

    /// Reshape contains negative dimension.
    #[snafu(display("reshape contains negative dimension in {shape:?}"))]
    ReshapeNegativeDimension { shape: SmallVec<[isize; 4]> },

    /// Broadcasting shape mismatch.
    #[snafu(display("cannot broadcast shapes {lhs:?} and {rhs:?}"))]
    BroadcastShapeMismatch { lhs: Box<Shape>, rhs: Box<Shape> },

    /// Symbolic padding unsupported.
    #[snafu(display("symbolic padding is not supported: padding dimensions must be concrete values"))]
    SymbolicPaddingUnsupported,

    /// Symbolic shrinking unsupported.
    #[snafu(display("symbolic shrinking is not supported: shrink ranges must be concrete values"))]
    SymbolicShrinkingUnsupported,

    /// Symbolic shape unsupported.
    #[snafu(display("symbolic shape is not supported for {operation}: shape dimensions must be concrete values"))]
    SymbolicShapeUnsupported { operation: &'static str },

    /// Operation requires a known shape but shape inference returned None.
    #[snafu(display("shape inference failed for {operation}: source has no inferable shape"))]
    MissingShape { operation: &'static str },

    /// A canonical parity document cannot represent this graph without losing semantics.
    #[snafu(display("canonical serialization failed: {detail}"))]
    CanonicalSerialization { detail: String },

    /// A constant value cannot be represented by its declared dtype.
    #[snafu(display("cannot commit constant {value:?} to dtype {dtype:?}"))]
    ConstantConversion { value: ConstValue, dtype: DType },

    /// Symbolic buffer size unsupported.
    #[snafu(display("cannot allocate buffer with symbolic size: range bound resolved to {bound:?}"))]
    SymbolicBufferSize { bound: crate::ConstValue },

    /// Ternary branch shape mismatch.
    #[snafu(display(
        "ternary operation branches have mismatched shapes: true branch {true_branch:?} vs false branch {false_branch:?}"
    ))]
    TernaryBranchShapeMismatch { true_branch: Box<Shape>, false_branch: Box<Shape> },

    /// Legacy buffer definitions required pointer dtype.
    #[snafu(display(
        "{op} must have Ptr dtype (following Tinygrad spec), got {dtype:?}. Use DefineVar for scalar variables."
    ))]
    BufferDefRequiresPtrDType { op: &'static str, dtype: DType },

    // =========================================================================
    // UOp Builder Guards (user-facing API for kernel implementation)
    // =========================================================================
    /// WHERE condition must be bool.
    #[snafu(display("WHERE condition must be bool, got {actual:?}"))]
    WhereConditionNotBool { actual: DType },

    /// BROADCAST requires scalar source.
    #[snafu(display("BROADCAST requires scalar source (vcount=1), got {dtype:?}"))]
    BroadcastRequiresScalar { dtype: DType },

    /// MulAcc operands must have matching dtypes.
    #[snafu(display(
        "MulAcc operands must have matching dtypes (including vcount): a={a_dtype:?}, b={b_dtype:?}, c={c_dtype:?}"
    ))]
    MulAccDtypeMismatch { a_dtype: DType, b_dtype: DType, c_dtype: DType },

    /// CALL body PARAM slots are not contiguous from 0.
    #[snafu(display("CALL params not in contiguous slot order: got {slots:?}"))]
    CallParamSlotsNotContiguous { slots: Vec<usize> },

    /// CALL argument count mismatch.
    #[snafu(display("CALL argument count mismatch: expected {expected}, got {got}"))]
    CallArgCountMismatch { expected: usize, got: usize },

    /// CALL argument shape mismatch.
    #[snafu(display("CALL argument {arg_index} shape mismatch: expected {expected:?}, got {got:?}"))]
    CallArgShapeMismatch { arg_index: usize, expected: Option<Box<Shape>>, got: Option<Box<Shape>> },

    /// CALL argument dtype mismatch.
    #[snafu(display("CALL argument {arg_index} dtype mismatch: expected {expected:?}, got {got:?}"))]
    CallArgDTypeMismatch { arg_index: usize, expected: DType, got: DType },

    /// A formal PARAM references a positional argument that is not present.
    #[snafu(display("FUNCTION formal PARAM slot {slot} has no argument (argument count {arg_count})"))]
    CallFormalSlotMissing { slot: isize, arg_count: usize },

    /// CALL/FUNCTION argument sharding axes must agree.
    #[snafu(display("CALL argument {arg_index} axis mismatch: expected {expected:?}, got {got:?}"))]
    CallArgAxisMismatch { arg_index: usize, expected: Option<usize>, got: Option<usize> },

    /// A symbolic output dimension cannot use the selected actual argument.
    #[snafu(display("FUNCTION shape substitution for formal slot {slot} is unsupported: {reason}"))]
    CallShapeSubstitutionUnsupported { slot: isize, reason: String },

    /// Shape substitution completed without replacing every body-local formal.
    #[snafu(display("FUNCTION result shape retains dangling formal PARAM slots {slots:?}"))]
    CallShapeDanglingFormal { slots: Vec<isize> },

    /// Kernel split dependency cycle detected while fixing AFTER assignments.
    #[snafu(display(
        "kernel split dependency cycle detected: writer buffer {writer_buffer} reads buffer {read_buffer} that depends on it"
    ))]
    KernelSplitDependencyCycle { writer_buffer: u64, read_buffer: u64 },

    /// Normal compiled kernels cannot span devices.
    #[snafu(display("normal kernel buffers must be on the same device, got {devices:?}"))]
    KernelSplitMixedDevices { devices: Vec<DeviceSpec> },

    /// GETTUPLE index out of bounds.
    #[snafu(display("GETTUPLE index {index} out of bounds for {kind} of length {len}"))]
    GetTupleIndexOutOfBounds { index: usize, len: usize, kind: &'static str },

    /// GETTUPLE source is neither a TUPLE nor a FUNCTION whose body is a TUPLE.
    #[snafu(display("GETTUPLE requires a TUPLE or FUNCTION(TUPLE) source, got {op}"))]
    GetTupleNotATuple { op: &'static str },

    /// STORE node reached range assignment with no inferable shape.
    #[snafu(display("STORE node id={uop_id} has no inferable index shape during range assignment"))]
    StoreMissingShape { uop_id: u64 },

    /// MULTI layouts on one operation disagree and would require resharding.
    #[snafu(display(
        "unsupported MULTI layout: {operation} has mismatched shard axes {axes:?}; resharding metadata is unavailable"
    ))]
    MultiAxisMismatch { operation: &'static str, axes: Vec<usize> },

    /// A MULTI wraps another MULTI, which represents unsupported multi-axis sharding.
    #[snafu(display("unsupported nested MULTI at axis {axis}; multi-axis sharding metadata is unavailable"))]
    MultiNested { axis: usize },

    /// A movement cannot be proven to preserve the represented shard boundary.
    #[snafu(display("unsupported {operation} across MULTI axis {axis}: {reason}"))]
    MultiMovementUnsupported { operation: &'static str, axis: usize, reason: &'static str },

    /// A reduction crosses the shard axis and therefore requires a collective.
    #[snafu(display(
        "unsupported reduction across MULTI axis {axis}: explicit MSTACK shard buffers are required for all-reduce lowering"
    ))]
    MultiReductionAcrossShardAxis { axis: usize },

    /// A non-scalar operand has no representable layout relative to a MULTI operand.
    #[snafu(display(
        "unsupported {operation} with MULTI axis {axis}: non-scalar operand id={source_id} has no shard layout; per-shard subviews require shard-range metadata"
    ))]
    MultiLayoutMissing { operation: &'static str, axis: usize, source_id: u64 },

    /// A MULTI or MSELECT form survived the supported pre-rangeify rewrites.
    #[snafu(display("unsupported multi-device form {operation}: {reason}"))]
    MultiUnsupported { operation: &'static str, reason: &'static str },

    /// A pre-linearized op list uses an op it only lists later.
    #[snafu(display("linear program: op {op} (id={id}) uses {source_op} (id={source_id}) before the list emits it"))]
    LinearForwardReference { op: String, id: u64, source_op: String, source_id: u64 },

    /// A pre-linearized op list repeats a node that is not a void statement.
    #[snafu(display(
        "linear program: {op} (id={id}) is listed twice; only void CUSTOM, STORE and BARRIER statements repeat (tag a value to emit it twice)"
    ))]
    LinearRepeat { op: String, id: u64 },
}
