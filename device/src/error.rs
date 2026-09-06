use snafu::Snafu;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    /// Shape of target tensor does not match expected shape.
    #[snafu(display("shape mismatch: expected {expected:?}, got {actual:?}"))]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[snafu(display("size mismatch: expected {expected}, got {actual}"))]
    SizeMismatch { expected: usize, actual: usize },

    /// Failed to copy data between host and device.
    #[snafu(display("copy operation failed: {reason}"))]
    CopyFailed { reason: String },

    /// Invalid device specification.
    #[snafu(display("invalid device: {device}"))]
    InvalidDevice { device: String },

    /// Buffer is not allocated.
    #[snafu(display("buffer not allocated"))]
    NotAllocated,

    /// Buffer is not CPU-accessible (device buffers require copyout).
    #[snafu(display("buffer is not CPU-accessible (device buffers require copyout)"))]
    NotCpuAccessible,

    /// Element type mismatch.
    #[snafu(display("type mismatch: buffer has {actual:?}, requested {expected:?}"))]
    TypeMismatch { expected: svod_dtype::DType, actual: svod_dtype::DType },

    /// Failed to create ndarray view from buffer shape.
    #[snafu(display("ndarray shape error: {source}"))]
    NdarrayShape { source: ndarray::ShapeError },

    /// Invalid buffer view parameters.
    #[snafu(display("invalid view: offset {offset} + size {size} exceeds buffer size {buffer_size}"))]
    InvalidView { offset: usize, size: usize, buffer_size: usize },

    /// Write refused: the storage was sealed immutable (shared weights).
    #[snafu(display("write to immutable buffer refused: {op} (storage id {storage})"))]
    ImmutableBuffer { op: &'static str, storage: u64 },

    /// Runtime execution error. Free-form catch-all; prefer the structured
    /// variants below when the data is structured.
    #[snafu(display("runtime error: {message}"))]
    Runtime { message: String },

    /// A backend command stream cannot fit its hardware field or queue budget.
    #[snafu(display("{kind} command stream is too large: {actual} (limit {limit})"))]
    CommandStreamTooLarge { kind: &'static str, actual: usize, limit: usize },

    /// A launch-size runtime variable fell outside its `DefineVar` bounds.
    #[snafu(display("variable {name}={value} is outside bounds [{min}, {max}]"))]
    VarOutOfBounds { name: String, value: i64, min: i64, max: i64 },

    /// A pipeline stage held the wrong op kind (PROGRAM/SINK/LINEAR/SOURCE
    /// shape validation in `ProgramSpec::from_uop`).
    #[snafu(display("expected {expected} op, got {got}"))]
    WrongStage { expected: &'static str, got: String },

    /// Two distinct PARAM definitions occupy the same final program argument
    /// slot. BUFFER allocations use a separate internal namespace.
    #[snafu(display("duplicate PROGRAM PARAM slot {slot}: {first} conflicts with {second}"))]
    DuplicateProgramParamSlot { slot: usize, first: String, second: String },

    /// Final PROGRAM construction reached the reserved unassigned PARAM slot.
    #[snafu(display("unassigned PROGRAM PARAM reached {stage}: {param}"))]
    UnassignedProgramParam { stage: &'static str, param: String },

    /// Renderer argument discovery disagreed with canonical ProgramInfo.
    #[snafu(display("renderer PROGRAM ABI mismatch: {reason}"))]
    ProgramAbiMismatch { reason: String },

    /// A cached SOURCE/BINARY payload does not belong to the executable
    /// PROGRAM identity that is attempting to reuse it.
    #[snafu(display("PROGRAM {stage} stage identity mismatch: {reason}"))]
    ProgramStageMismatch { stage: &'static str, reason: String },

    /// A FUNCTION/CALL formal PARAM escaped its opaque body into the enclosing
    /// executable graph.
    #[snafu(display("opaque formal PARAM leaked into PROGRAM ABI: {param}"))]
    LeakedOpaqueProgramParam { param: String },

    /// PROGRAM metadata was built for a different renderer target.
    #[snafu(display("PROGRAM target {actual:?} does not match renderer target {expected:?}"))]
    ProgramTargetMismatch { expected: svod_dtype::DeviceSpec, actual: svod_dtype::DeviceSpec },

    /// AMD GPU memory fault decoded from a KFD event. `class` is a short
    /// VA-classification hint (owning / stale / nearest allocation).
    #[snafu(display(
        "AMD GPU memory fault on gpu_id={gpu_id} va={va:#x} \
         (NotPresent={not_present} ReadOnly={read_only} NoExecute={no_execute} \
         Imprecise={imprecise} ErrorType={error_type}) — {class}"
    ))]
    GpuFault {
        gpu_id: u32,
        va: u64,
        not_present: bool,
        read_only: bool,
        no_execute: bool,
        imprecise: bool,
        error_type: u32,
        class: String,
    },

    /// A timeline signal did not reach its target value before the deadline.
    #[snafu(display("{what} timed out after {waited_ms} ms (target {target}, current {current})"))]
    TimelineTimeout { what: &'static str, target: u64, current: u64, waited_ms: u64 },

    /// An AMD infrastructure buffer (kernarg / ring / GART / signal / code
    /// object / graph IB) was allocated without a host-visible mapping.
    #[snafu(display("{what} requires a host-visible AMD buffer"))]
    NotHostVisible { what: &'static str },

    /// AMD GPU not present (no `/dev/kfd`, empty topology, permission denied,
    /// or selected device index out of range).
    #[snafu(display("no AMD GPU available: {reason}"))]
    NoAmdGpu { reason: String },

    /// AMD KFD ioctl failure.
    #[snafu(display("AMD ioctl {ioctl} failed (errno {errno})"))]
    AmdIoctl { ioctl: &'static str, errno: i32 },

    /// Queue creation reached an active KFD queue, then doorbell mapping and
    /// rollback destruction both failed. Backing allocations must be
    /// quarantined because the kernel may still reference them.
    #[snafu(display("AMD queue {queue_id} remained active after setup rollback failed: {cause}"))]
    AmdQueueStillActive { queue_id: u32, cause: String },

    /// AMD allocation failure (VRAM exhaustion, BAR-resize required, etc.).
    #[snafu(display("AMD allocation failed: {reason}"))]
    AmdAllocFailed { reason: String },

    /// Kernel requests more LDS/group-segment than the device exposes.
    #[snafu(display("group_segment too large: {requested} > device limit {limit} (lds_size_in_kb {lds_kb})"))]
    GroupSegmentTooLarge { requested: u32, limit: u32, lds_kb: u32 },

    /// No CUDA GPU (driver loaded but `cuInit` failed, zero devices, or the
    /// selected device index is out of range).
    #[snafu(display("no CUDA GPU available: {reason}"))]
    NoCudaGpu { reason: String },

    /// CUDA driver API call failure, described by the driver itself.
    #[snafu(display("CUDA {call} failed: {name} ({code}): {message}"))]
    CudaDriver { call: &'static str, code: i32, name: String, message: String },

    /// The driver JIT rejected a PTX module; `log` is its error log.
    #[snafu(display("CUDA JIT of kernel {kernel:?} failed: {cause}\n{log}"))]
    CudaJit { kernel: String, cause: String, log: String },

    /// CUDA allocation failure (VRAM exhaustion, unsupported memory kind).
    #[snafu(display("CUDA allocation of {size} bytes failed: {reason}"))]
    CudaAllocFailed { size: usize, reason: String },

    /// Device requested but unavailable on this host (wrong OS, missing libs).
    #[snafu(display("device unavailable: {reason}"))]
    DeviceUnavailable { reason: String },

    /// Allocator does not implement an optional operation (e.g. `_transfer`,
    /// `_offset`, `_map`); the `Allocator` base defaults these to unsupported.
    /// Reachable only via cross-backend misuse, never on the CPU path.
    #[snafu(display("allocator does not support operation: {op}"))]
    Unsupported { op: &'static str },
}
