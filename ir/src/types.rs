//! Type definitions for IR operations.
//!
//! This module contains all the fundamental type enums and structs used throughout
//! the IR, including operation types, constant values, and metadata structures.

use std::hash::{Hash, Hasher};
use std::mem::discriminant;
use std::sync::Arc;

use crate::ops;
use smallvec::SmallVec;
use svod_dtype::DeviceSpec;
use svod_dtype::cast::commit_float;
use svod_dtype::{DType, ScalarDType};

/// Schema version for semantic SOURCE stage identities. v2 hashes the LINEAR
/// canonical graph as bincode instead of pretty JSON.
pub const SOURCE_STAGE_IDENTITY_VERSION: u32 = 2;

/// Schema version for semantic BINARY stage identities.
pub const BINARY_STAGE_IDENTITY_VERSION: u32 = 1;

/// A collision-resistant digest used by staged executable IR identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct StageDigest(pub [u8; 32]);

/// ABI argument class in an executable SOURCE identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StageAbiParamKind {
    Storage(svod_dtype::AddrSpace),
    Scalar,
}

/// One ordered external ABI argument in an executable SOURCE identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct StageAbiParam {
    pub slot: usize,
    pub kind: StageAbiParamKind,
    pub dtype: DType,
    pub name: Option<String>,
}

/// Semantic proof that SOURCE belongs to one exact PROGRAM and LINEAR stage.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SourceStageIdentity {
    pub version: u32,
    pub abi: Vec<StageAbiParam>,
    pub target: DeviceSpec,
    pub entry_name: String,
    pub linear_sha256: StageDigest,
    pub source_sha256: StageDigest,
}

/// Semantic proof that BINARY was compiled from one exact SOURCE identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BinaryStageIdentity {
    pub version: u32,
    pub source: SourceStageIdentity,
    pub compiler_key: String,
    pub binary_sha256: StageDigest,
}

/// Constant value that can be stored in a UOp.
///
/// `PartialEq` is implemented manually so that float comparison uses bit
/// equality (`to_bits()`), matching the `Hash` impl below. Without this,
/// `-0.0 == 0.0` and `NaN != NaN` under `PartialEq` would diverge from the
/// hash semantics, breaking Rust's `Hash`/`Eq` contract for hash-cons keys.
#[derive(Debug, Clone, Copy, derive_more::From)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum ConstValue {
    /// Canonical poison value. Its UOp dtype is always bool and consumers treat
    /// it as matching any dtype.
    Invalid,
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
}

impl PartialEq for ConstValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Invalid, Self::Invalid) => true,
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::UInt(a), Self::UInt(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::Bool(a), Self::Bool(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for ConstValue {}

macro_rules! impl_from_widening {
    ($($ty:ty => Int),+ $(,)?) => { $(
        impl From<$ty> for ConstValue {
            fn from(v: $ty) -> Self { ConstValue::Int(v as i64) }
        }
    )+ };
    ($($ty:ty => UInt),+ $(,)?) => { $(
        impl From<$ty> for ConstValue {
            fn from(v: $ty) -> Self { ConstValue::UInt(v as u64) }
        }
    )+ };
}

impl_from_widening!(i8 => Int, i16 => Int, i32 => Int);
impl_from_widening!(u8 => UInt, u16 => UInt, u32 => UInt);

impl From<f32> for ConstValue {
    fn from(v: f32) -> Self {
        ConstValue::Float(v as f64)
    }
}

/// Manual Hash impl because f64 doesn't implement Hash.
/// Uses to_bits() for floats, which means NaN values with identical bit patterns hash equally.
impl Hash for ConstValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        discriminant(self).hash(state);
        match self {
            ConstValue::Invalid => {}
            ConstValue::Int(v) => v.hash(state),
            ConstValue::UInt(v) => v.hash(state),
            ConstValue::Float(v) => v.to_bits().hash(state),
            ConstValue::Bool(v) => v.hash(state),
        }
    }
}

/// Helper macro to cast to target width and back to storage type (for proper truncation/extension).
macro_rules! cast_via {
    ($v:expr, $target:ty, $storage:ty) => {
        ($v as $target) as $storage
    };
}

/// Macro to generate casting logic by delegating to helper functions.
macro_rules! impl_cast {
    ($self:expr, $to:expr) => {
        match ($self, $to) {
            (ConstValue::Invalid, _) => ConstValue::Invalid,
            (ConstValue::Bool(v), dt) => cast_bool(v, dt)?,
            (ConstValue::Int(v), dt) => cast_int(v, dt)?,
            (ConstValue::UInt(v), dt) => cast_uint(v, dt)?,
            (ConstValue::Float(v), dt) => cast_float(v, dt)?,
        }
    };
}

#[inline]
fn cast_bool(v: bool, to: ScalarDType) -> Option<ConstValue> {
    use ScalarDType::*;
    Some(match to {
        Bool => ConstValue::Bool(v),
        WeakInt | Int8 | Int16 | Int32 | Int64 | Index => ConstValue::Int(v as i64),
        UInt8 | UInt16 | UInt32 | UInt64 => ConstValue::UInt(v as u64),
        WeakFloat | FP8E4M3 | FP8E4M3FNUZ | FP8E5M2 | FP8E5M2FNUZ | Float16 | BFloat16 | Float32 | Float64 => {
            ConstValue::Float(commit_float(v as u8 as f64, to)?)
        }
        Void => ConstValue::Int(v as i64),
    })
}

#[inline]
fn cast_int(v: i64, to: ScalarDType) -> Option<ConstValue> {
    use ScalarDType::*;
    Some(match to {
        Bool => ConstValue::Bool(v != 0),
        Int8 => ConstValue::Int(cast_via!(v, i8, i64)),
        Int16 => ConstValue::Int(cast_via!(v, i16, i64)),
        Int32 => ConstValue::Int(cast_via!(v, i32, i64)),
        WeakInt | Int64 | Index => ConstValue::Int(v),
        UInt8 => ConstValue::UInt(cast_via!(v, u8, u64)),
        UInt16 => ConstValue::UInt(cast_via!(v, u16, u64)),
        UInt32 => ConstValue::UInt(cast_via!(v, u32, u64)),
        UInt64 => ConstValue::UInt(v as u64),
        WeakFloat | FP8E4M3 | FP8E4M3FNUZ | FP8E5M2 | FP8E5M2FNUZ | Float16 | BFloat16 | Float32 | Float64 => {
            ConstValue::Float(commit_float(v as f64, to)?)
        }
        Void => ConstValue::Int(v),
    })
}

#[inline]
fn cast_uint(v: u64, to: ScalarDType) -> Option<ConstValue> {
    use ScalarDType::*;
    Some(match to {
        Bool => ConstValue::Bool(v != 0),
        Int8 => ConstValue::Int(cast_via!(v, i8, i64)),
        Int16 => ConstValue::Int(cast_via!(v, i16, i64)),
        Int32 => ConstValue::Int(cast_via!(v, i32, i64)),
        WeakInt | Int64 | Index => ConstValue::Int(v as i64),
        UInt8 => ConstValue::UInt(cast_via!(v, u8, u64)),
        UInt16 => ConstValue::UInt(cast_via!(v, u16, u64)),
        UInt32 => ConstValue::UInt(cast_via!(v, u32, u64)),
        UInt64 => ConstValue::UInt(v),
        WeakFloat | FP8E4M3 | FP8E4M3FNUZ | FP8E5M2 | FP8E5M2FNUZ | Float16 | BFloat16 | Float32 | Float64 => {
            ConstValue::Float(commit_float(v as f64, to)?)
        }
        Void => ConstValue::Int(v as i64),
    })
}

#[inline]
fn cast_float(v: f64, to: ScalarDType) -> Option<ConstValue> {
    use ScalarDType::*;
    Some(match to {
        Bool => ConstValue::Bool(v != 0.0),
        Int8 => ConstValue::Int(cast_via!(v, i8, i64)),
        Int16 => ConstValue::Int(cast_via!(v, i16, i64)),
        Int32 => ConstValue::Int(cast_via!(v, i32, i64)),
        WeakInt | Int64 | Index => ConstValue::Int(v as i64),
        // Float-to-unsigned: route through i64 first.
        UInt8 => ConstValue::UInt(cast_via!(v as i64, u8, u64)),
        UInt16 => ConstValue::UInt(cast_via!(v as i64, u16, u64)),
        UInt32 => ConstValue::UInt(cast_via!(v as i64, u32, u64)),
        UInt64 => ConstValue::UInt((v as i64) as u64),
        WeakFloat | FP8E4M3 | FP8E4M3FNUZ | FP8E5M2 | FP8E5M2FNUZ | Float16 | BFloat16 | Float32 | Float64 => {
            ConstValue::Float(commit_float(v, to)?)
        }
        Void => ConstValue::Int(v as i64),
    })
}

impl ConstValue {
    pub const fn dtype(&self) -> DType {
        match self {
            ConstValue::Invalid => DType::Bool,
            ConstValue::Int(_) => DType::Int64,
            ConstValue::UInt(_) => DType::UInt64,
            ConstValue::Float(_) => DType::Float64,
            ConstValue::Bool(_) => DType::Bool,
        }
    }

    pub const fn zero(dtype: ScalarDType) -> Self {
        use ScalarDType::*;
        match dtype {
            Bool => Self::Bool(false),
            WeakInt | Int8 | Int16 | Int32 | Int64 => Self::Int(0),
            UInt8 | UInt16 | UInt32 | UInt64 => Self::UInt(0),
            WeakFloat | FP8E4M3 | FP8E4M3FNUZ | FP8E5M2 | FP8E5M2FNUZ | Float16 | BFloat16 | Float32 | Float64 => {
                Self::Float(0.0)
            }
            Void | Index => Self::Int(0), // TODO: remove this types from scalars
        }
    }

    pub const fn one(dtype: ScalarDType) -> Self {
        use ScalarDType::*;
        match dtype {
            Bool => Self::Bool(true),
            WeakInt | Int8 | Int16 | Int32 | Int64 => Self::Int(1),
            UInt8 | UInt16 | UInt32 | UInt64 => Self::UInt(1),
            WeakFloat | FP8E4M3 | FP8E4M3FNUZ | FP8E5M2 | FP8E5M2FNUZ | Float16 | BFloat16 | Float32 | Float64 => {
                Self::Float(1.0)
            }
            Void | Index => Self::Int(1), // TODO: remove this types from scalars
        }
    }

    pub const fn neg_one(dtype: ScalarDType) -> Option<Self> {
        use ScalarDType::*;
        Some(match dtype {
            WeakInt | Int8 | Int16 | Int32 | Int64 | Index => Self::Int(-1),
            WeakFloat | FP8E4M3 | FP8E4M3FNUZ | FP8E5M2 | FP8E5M2FNUZ | Float16 | BFloat16 | Float32 | Float64 => {
                Self::Float(-1.0)
            }
            _ => return None,
        })
    }

    /// Minimum representable value for a scalar dtype.
    pub const fn min(dtype: ScalarDType) -> Self {
        use ScalarDType::*;
        match dtype {
            Bool => Self::Bool(false),
            Int8 => Self::Int(i8::MIN as i64),
            Int16 => Self::Int(i16::MIN as i64),
            Int32 => Self::Int(i32::MIN as i64),
            WeakInt | Int64 | Index => Self::Int(i64::MIN),
            UInt8 | UInt16 | UInt32 | UInt64 => Self::UInt(0),
            FP8E4M3 => Self::Float(-448.0),
            FP8E4M3FNUZ => Self::Float(-240.0),
            FP8E5M2 | FP8E5M2FNUZ => Self::Float(-57344.0),
            Float16 => Self::Float(-65504.0),
            BFloat16 => Self::Float(-3.3895313892515355e38),
            Float32 => Self::Float(f32::MIN as f64),
            WeakFloat | Float64 => Self::Float(f64::MIN),
            Void => Self::Int(0),
        }
    }

    /// Maximum representable value for a scalar dtype.
    pub const fn max(dtype: ScalarDType) -> Self {
        use ScalarDType::*;
        match dtype {
            Bool => Self::Bool(true),
            Int8 => Self::Int(i8::MAX as i64),
            Int16 => Self::Int(i16::MAX as i64),
            Int32 => Self::Int(i32::MAX as i64),
            WeakInt | Int64 | Index => Self::Int(i64::MAX),
            UInt8 => Self::UInt(u8::MAX as u64),
            UInt16 => Self::UInt(u16::MAX as u64),
            UInt32 => Self::UInt(u32::MAX as u64),
            UInt64 => Self::UInt(u64::MAX),
            FP8E4M3 => Self::Float(448.0),
            FP8E4M3FNUZ => Self::Float(240.0),
            FP8E5M2 | FP8E5M2FNUZ => Self::Float(57344.0),
            Float16 => Self::Float(65504.0),
            BFloat16 => Self::Float(3.3895313892515355e38),
            Float32 => Self::Float(f32::MAX as f64),
            WeakFloat | Float64 => Self::Float(f64::MAX),
            Void => Self::Int(0),
        }
    }

    /// Cast this constant value to the target dtype.
    ///
    /// Returns `None` if:
    /// - The target dtype is not a scalar type
    /// - The target dtype is not representable as a ConstValue (e.g., Void, Index, special float formats)
    ///
    /// # Safety and Semantics
    ///
    /// This method performs constant folding for cast operations and allows ALL casts
    /// (including lossy ones like float->int) since the user explicitly wrote the cast operation.
    ///
    /// Uses Rust's `as` operator for conversions, which follows C semantics:
    /// - Truncation for narrowing conversions (e.g., i64 -> i32)
    /// - Wrap-around for unsigned overflow
    /// - Truncation toward zero for float-to-int conversions
    ///
    /// For multi-stage conversions (e.g., casting through intermediate types),
    /// the value is cast to the target width and then extended back to the storage type.
    /// Example: i64 -> i8 -> i64 ensures proper sign extension.
    pub fn cast(&self, dtype: &DType) -> Option<Self> {
        let scalar_dtype = dtype.scalar()?;

        Some(impl_cast!(*self, scalar_dtype))
    }

    /// Returns true if this constant is zero (additive identity).
    ///
    /// Works for all numeric types: Int, UInt, Float, Bool.
    pub const fn is_zero(&self) -> bool {
        match self {
            Self::Int(0) | Self::UInt(0) | Self::Bool(false) => true,
            Self::Float(f) => *f == 0.0,
            _ => false,
        }
    }

    /// Returns true if this constant is one (multiplicative identity).
    ///
    /// Works for all numeric types: Int, UInt, Float, Bool.
    pub const fn is_one(&self) -> bool {
        match self {
            Self::Int(1) | Self::UInt(1) | Self::Bool(true) => true,
            Self::Float(f) => *f == 1.0,
            _ => false,
        }
    }

    /// Returns true if this constant is negative one.
    ///
    /// Used for patterns like `x // -1 → -x`.
    pub const fn is_neg_one(&self) -> bool {
        match self {
            Self::Int(-1) => true,
            Self::Float(f) => *f == -1.0,
            _ => false,
        }
    }

    /// Try to extract an integer value (i64 or u64 as i64).
    ///
    /// Used for constant pattern matching with specific integer values.
    pub const fn try_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::UInt(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Try to extract a float value (f64).
    ///
    /// Used for constant pattern matching with specific float values.
    pub const fn try_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Commit a value to the dtype grid used by typed constant folding.
    ///
    /// Used for constant folding to ensure results respect the target dtype's bit width.
    pub fn truncate(self, dtype: ScalarDType) -> Self {
        self.cast(&DType::Scalar(dtype)).expect("typed constant evaluation produced an unsupported dtype/value pair")
    }
}

// Re-export AddrSpace from dtype to avoid duplication
pub use svod_dtype::AddrSpace;

/// Structured metadata for PARAM and, eventually, BUFFER definitions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ParamArg {
    pub slot: usize,
    pub dtype: DType,
    pub vmin_vmax: Option<(ConstValueHash, ConstValueHash)>,
    pub multiple_of: Option<usize>,
    pub name: Option<String>,
    pub addrspace: Option<AddrSpace>,
    pub axis: Option<usize>,
    pub device: Option<DeviceSpec>,
    pub volatile: bool,
}

impl ParamArg {
    pub fn buffer(slot: usize, dtype: DType, addrspace: AddrSpace, device: Option<DeviceSpec>) -> Self {
        Self {
            slot,
            dtype,
            vmin_vmax: None,
            multiple_of: None,
            name: None,
            addrspace: Some(addrspace),
            axis: None,
            device,
            volatile: false,
        }
    }

    pub fn variable(name: String, dtype: DType, min_val: i64, max_val: i64) -> Self {
        Self {
            slot: usize::MAX,
            dtype,
            vmin_vmax: Some((ConstValueHash(ConstValue::Int(min_val)), ConstValueHash(ConstValue::Int(max_val)))),
            multiple_of: Some(1),
            name: Some(name),
            addrspace: None,
            axis: None,
            device: None,
            volatile: false,
        }
    }

    /// Positional scalar PARAM used by FUNCTION bodies.
    pub fn scalar(slot: usize, name: Option<String>, dtype: DType, min_val: i64, max_val: i64) -> Self {
        Self {
            slot,
            dtype,
            vmin_vmax: Some((ConstValueHash(ConstValue::Int(min_val)), ConstValueHash(ConstValue::Int(max_val)))),
            multiple_of: Some(1),
            name,
            addrspace: None,
            axis: None,
            device: None,
            volatile: false,
        }
    }
}

/// Options for STAGE operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct BufferizeOpts {
    /// Source device for ordinary stages; grouped LOCAL stages use `local_axis`.
    pub device: Option<DeviceSpec>,
    /// GROUP_REDUCE axis that owns a LOCAL staging buffer.
    ///
    /// Tinygrad stores this identity in `device` for LOCAL STAGEs. Keep it
    /// typed so nested axis paths cannot alias device names or scalar axes.
    pub local_axis: Option<AxisId>,
    /// Address space (GLOBAL or LOCAL).
    pub addrspace: AddrSpace,
    /// Whether buffer_removal may inline this STAGE.
    /// Multi-consumer realize boundaries set this to `false` so that
    /// `dead_axis_removal` (which creates new STAGE nodes) preserves
    /// the protection across mega-pass fixpoint iterations.
    pub removable: bool,
}

impl BufferizeOpts {
    pub fn new(device: DeviceSpec) -> Self {
        Self { device: Some(device), local_axis: None, addrspace: AddrSpace::Global, removable: true }
    }

    pub fn local() -> Self {
        Self { device: None, local_axis: None, addrspace: AddrSpace::Local, removable: true }
    }

    pub fn local_for_axis(axis: AxisId) -> Self {
        Self { device: None, local_axis: Some(axis), addrspace: AddrSpace::Local, removable: true }
    }
}

/// Optimization hint carried by CONTIGUOUS ops.
///
/// Simplified representation of optimizer hints that can be converted to/from
/// the full `Opt` type in the schedule crate. Keeps the IR layer decoupled
/// from optimizer-specific types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ContiguousHint {
    /// Operation name (e.g., "UPCAST", "LOCAL", "UNROLL")
    pub op: String,
    /// Target axis index (if applicable)
    pub axis: Option<usize>,
    /// Integer argument (amount, size, etc.)
    pub arg: Option<i64>,
}

/// Metadata payload carried by CALL operations.
///
/// `grad_tag` is a placeholder for future gradient callback identity; the origin
/// fields are the kernel's attribution, harvested at the kernel cut from the nodes
/// that were folded into the body. They live here rather than on the body SINK
/// because the CALL is per dispatch while the body keys every kernel cache.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CallInfo {
    pub grad_tag: Option<String>,
    /// Origin of the stored value's root — what the kernel is charged to.
    pub origin: Option<crate::origin::OriginId>,
    /// Every origin reachable in the body before it was stripped.
    pub origins: crate::origin::OriginSet,
    pub name: Option<String>,
    pub precompile: bool,
    pub precompile_backward: bool,
}

/// Explicit runtime custom-function kinds.
///
/// Reserved helpers remain typed `Unsupported` at runtime; `AllReduce` is the
/// correctness-first host implementation used by collective schedule items.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum CustomFunctionKind {
    /// HEVC encode/decode runner.
    EncDec,
    /// JIT graph-capture hook.
    Graph,
    /// Correctness-first host-staged collective over explicit shard buffers.
    AllReduce { reduce_op: ReduceOp },
}

/// Structural marker carried in `Op::Sink::info` indicating the SINK is a
/// fully-formed kernel AST.
///
/// Stored as a hash-consed field rather than via the type-erased `metadata`
/// channel so that marked and unmarked SINKs with otherwise identical
/// sources are distinct UOps.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct KernelInfo {
    /// Author-supplied optimization control, mirroring tinygrad's
    /// `KernelInfo.opts_to_apply`:
    /// - `None` — the optimizer chooses opts (heuristics or beam).
    /// - `Some(vec![])` — apply *zero* opts; the AST is already in finished,
    ///   hand-lowered form and must pass through untouched (e.g. a tile-DSL
    ///   kernel).
    /// - `Some(non-empty)` — apply exactly these opts, in order.
    pub opts_to_apply: Option<Vec<crate::opt::Opt>>,
    /// Optimizations actually applied by the production scheduler.
    pub applied_opts: Vec<crate::opt::Opt>,
    /// Final optimizer decision to avoid local/shared-memory transforms.
    pub dont_use_locals: bool,
    /// Author-supplied kernel name carried on the SINK itself (set by hand-lowered
    /// tile-DSL kernels via their `Kernel::finish`). The optimizer-scheduled path
    /// names kernels through its own metadata channel and leaves this `None`; the
    /// render-name driver falls back to this when that metadata is absent, so custom
    /// kernels (`flash_attention`, `ffn_gemm1`, `matmul_nt`, …) keep their real names
    /// in the profile instead of collapsing to the `"kernel"` default.
    pub name: Option<String>,
}

impl KernelInfo {
    /// Whether the AST is a finished, hand-lowered body the optimizer passes
    /// through untouched (`opts_to_apply == Some([])`), as a tile-DSL kernel.
    pub fn is_hand_lowered(&self) -> bool {
        self.opts_to_apply.as_ref().is_some_and(Vec::is_empty)
    }
}

/// Target-defined instruction identity. Operands remain ordinary UOp sources;
/// target-specific encoding facts live in sorted, deterministic attributes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct InsArg {
    pub opcode: String,
    pub attributes: Vec<(String, String)>,
}

impl InsArg {
    pub fn new(opcode: impl Into<String>) -> Self {
        Self { opcode: opcode.into(), attributes: Vec::new() }
    }

    pub fn with_attributes(opcode: impl Into<String>, mut attributes: Vec<(String, String)>) -> Self {
        attributes.sort();
        Self { opcode: opcode.into(), attributes }
    }
}

/// Structural PROGRAM argument, ported from tinygrad's `ProgramInfo` at 8c8b43de.
///
/// Representation differences are limited to existing IR types: Tinygrad's
/// `Target` is [`DeviceSpec`] and launch values are [`UOp`](crate::UOp)s rather
/// than Python `int | float`.
#[derive(Debug, Clone)]
pub struct ProgramInfo {
    pub name: String,
    pub global_size: [Arc<crate::UOp>; 3],
    pub local_size: Option<[Arc<crate::UOp>; 3]>,
    pub vars: Vec<Arc<crate::UOp>>,
    pub globals: Vec<usize>,
    pub outs: Vec<usize>,
    pub ins: Vec<usize>,
    pub target: DeviceSpec,
}

impl Default for ProgramInfo {
    fn default() -> Self {
        let one = || crate::UOp::index_const(1);
        Self {
            name: "test".to_string(),
            global_size: [one(), one(), one()],
            local_size: None,
            vars: Vec::new(),
            globals: Vec::new(),
            outs: Vec::new(),
            ins: Vec::new(),
            target: DeviceSpec::Cpu,
        }
    }
}

impl PartialEq for ProgramInfo {
    fn eq(&self, other: &Self) -> bool {
        fn same_uops(a: &[Arc<crate::UOp>], b: &[Arc<crate::UOp>]) -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.content_hash == b.content_hash)
        }

        self.name == other.name
            && same_uops(&self.global_size, &other.global_size)
            && match (&self.local_size, &other.local_size) {
                (Some(a), Some(b)) => same_uops(a, b),
                (None, None) => true,
                _ => false,
            }
            && same_uops(&self.vars, &other.vars)
            && self.globals == other.globals
            && self.outs == other.outs
            && self.ins == other.ins
            && self.target == other.target
    }
}

impl Eq for ProgramInfo {}

impl Hash for ProgramInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.global_size.iter().for_each(|u| u.content_hash.hash(state));
        self.local_size.as_ref().map(|dims| dims.iter().for_each(|u| u.content_hash.hash(state))).hash(state);
        self.vars.iter().for_each(|u| u.content_hash.hash(state));
        self.globals.hash(state);
        self.outs.hash(state);
        self.ins.hash(state);
        self.target.hash(state);
    }
}

/// Convert a diagnostic kernel name to Tinygrad's renderer-safe identifier.
pub fn to_function_name(name: &str) -> String {
    let chars = name.chars().collect::<Vec<_>>();
    let mut output = String::new();
    let mut index = 0;

    while index < chars.len() {
        if chars[index] == '\x1b' && chars.get(index + 1) == Some(&'[') {
            let end = if chars.get(index + 2) == Some(&'K') {
                Some(index + 2)
            } else {
                (index + 2..chars.len()).find(|&candidate| chars[candidate] == 'm')
            };
            if let Some(end) = end {
                index = end + 1;
                continue;
            }
        }

        let character = chars[index];
        if character.is_ascii_alphanumeric() || character == '_' {
            output.push(character);
        } else {
            output.push_str(&format!("{:02X}", character as u32));
        }
        index += 1;
    }

    output
}

impl ProgramInfo {
    pub fn function_name(&self) -> String {
        to_function_name(&self.name)
    }

    fn ssimplify(uop: &Arc<crate::UOp>) -> Arc<crate::UOp> {
        use crate::{BinaryOp, Op};

        fn const_value(uop: &Arc<crate::UOp>) -> Option<ConstValue> {
            match uop.op() {
                Op::Const(value) => Some(value.0),
                _ => None,
            }
        }
        fn is_int(uop: &Arc<crate::UOp>, value: i64) -> bool {
            matches!(const_value(uop), Some(ConstValue::Int(v)) if v == value)
                || matches!(const_value(uop), Some(ConstValue::UInt(v)) if value >= 0 && v == value as u64)
        }

        let sources = uop.op().sources();
        let rewritten_sources: Vec<_> = sources.iter().map(Self::ssimplify).collect();
        let rewritten = if sources.iter().zip(&rewritten_sources).all(|(old, new)| Arc::ptr_eq(old, new)) {
            uop.clone()
        } else {
            uop.with_sources(rewritten_sources)
        };

        match rewritten.op() {
            Op::Cast(ops::Cast { src, dtype }) if src.dtype() == *dtype => src.clone(),
            Op::Binary(op, lhs, rhs) => {
                if let (Some(a), Some(b)) = (const_value(lhs), const_value(rhs))
                    && let Some(value) = crate::uop::eval::eval_binary_op(*op, a, b)
                {
                    return crate::UOp::const_(rewritten.dtype(), value);
                }
                match op {
                    BinaryOp::Add if is_int(rhs, 0) => lhs.clone(),
                    BinaryOp::Add if is_int(lhs, 0) => rhs.clone(),
                    BinaryOp::Sub if is_int(rhs, 0) => lhs.clone(),
                    BinaryOp::Mul if is_int(rhs, 1) => lhs.clone(),
                    BinaryOp::Mul if is_int(lhs, 1) => rhs.clone(),
                    BinaryOp::FloorDiv if is_int(rhs, 1) => lhs.clone(),
                    _ => rewritten,
                }
            }
            _ => rewritten,
        }
    }

    /// Derive PROGRAM identity from the final SINK using Tinygrad's traversal,
    /// ordering, deduplication, and SPECIAL extent simplification rules.
    pub fn from_sink(sink: &Arc<crate::UOp>, target: DeviceSpec) -> Self {
        use crate::Op;

        let one = || crate::UOp::index_const(1);
        let mut vars = Vec::new();
        let mut globals = Vec::new();
        let mut outs = Vec::new();
        let mut ins = Vec::new();
        let mut global_size = [one(), one(), one()];
        let mut local_size = Some([one(), one(), one()]);

        fn buf_param_slot(mut uop: &Arc<crate::UOp>) -> Option<usize> {
            loop {
                match uop.op() {
                    Op::Param(ops::Param { arg, .. }) => return Some(arg.slot),
                    Op::Index(ops::Index { buffer, .. }) => uop = buffer,
                    Op::Shrink(ops::Shrink { src, .. }) | Op::Cast(ops::Cast { src, .. }) => uop = src,
                    Op::After(ops::After { passthrough, .. }) => uop = passthrough,
                    _ => return None,
                }
            }
        }

        fn param_slot(index: &Arc<crate::UOp>) -> Option<usize> {
            let index = match index.op() {
                Op::Index(..) | Op::Shrink(..) => index,
                Op::Cast(ops::Cast { src, .. }) if matches!(src.op(), Op::Index(..) | Op::Shrink(..)) => src,
                _ => return None,
            };
            let buffer_or_value = match index.op() {
                Op::Index(ops::Index { buffer, .. }) => buffer,
                Op::Shrink(ops::Shrink { src, .. }) => src,
                _ => return None,
            };
            buf_param_slot(buffer_or_value)
        }

        // CALL/FUNCTION bodies define a separate PARAM namespace. Only call
        // arguments belong to the enclosing executable PROGRAM ABI.
        for u in sink.toposort_call_aware(false) {
            match u.op() {
                Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none() => {
                    vars.push(u.clone());
                    if arg.name.as_deref() == Some("core_id")
                        && let Some((_, ConstValueHash(ConstValue::Int(max_val)))) = arg.vmin_vmax
                    {
                        global_size[0] = crate::UOp::index_const(max_val + 1);
                    }
                }
                Op::Param(ops::Param { arg, .. }) => globals.push(arg.slot),
                Op::Store(ops::Store { index, .. }) => {
                    if let Some(slot) = param_slot(index) {
                        outs.push(slot);
                    }
                }
                Op::Load(ops::Load { index, .. }) => {
                    if let Some(slot) = param_slot(index) {
                        ins.push(slot);
                    }
                }
                Op::Special(ops::Special { end, name }) => {
                    let Some(axis) = name.chars().last().and_then(|c| c.to_digit(10)).map(|axis| axis as usize) else {
                        continue;
                    };
                    if axis >= global_size.len() {
                        continue;
                    }
                    if name.starts_with('i') {
                        local_size = None;
                    }
                    let special_size = if name.starts_with('l') { local_size.as_mut() } else { Some(&mut global_size) };
                    if let Some(special_size) = special_size {
                        special_size[axis] = Self::ssimplify(end);
                    }
                }
                _ => {}
            }
        }

        vars.sort_by_key(|u| match u.op() {
            Op::Param(ops::Param { arg, .. }) => arg.slot,
            _ => usize::MAX,
        });
        vars.dedup_by_key(|u| u.content_hash);
        globals.sort_unstable();
        globals.dedup();
        outs.sort_unstable();
        outs.dedup();
        ins.sort_unstable();
        ins.dedup();

        let name = match sink.op() {
            Op::Sink(ops::Sink { info: Some(info), .. }) => info.name.clone().unwrap_or_else(|| "test".to_string()),
            _ => "test".to_string(),
        };
        Self { name, global_size, local_size, vars, globals, outs, ins, target }
    }
}

/// Axis type for loop ranges and reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum AxisType {
    /// Device-selection dimension, bound per device at launch.
    Device,
    /// GPU grid dimension.
    Global,
    /// Warp/wavefront dimension.
    Warp,
    /// GPU block/workgroup dimension (local memory scope).
    Local,
    /// Unparallelized range produced by rangeify.
    Weak,
    /// Explicit regular loop.
    Loop,
    /// Grouped reduction.
    GroupReduce,
    /// Reduction axis.
    Reduce,
    /// Vectorization axis (upcast).
    Upcast,
    /// Unrolled loop.
    Unroll,
    /// Thread dimension.
    Thread,
    /// Temporary canonicalized range for RESHAPE caching. Substituted in before
    /// `_apply_reshape` and substituted back after.
    Placeholder,
}

impl AxisType {
    /// Returns the priority for sorting ranges.
    ///
    /// Lower values are outer loops, higher values are inner loops.
    ///
    /// **Priority Order:**
    /// - Weak/Loop: -1 (not yet parallelized)
    /// - Global/Thread: 0 (outer parallelism)
    /// - Warp: 1 (sub-group parallelism)
    /// - Local/GroupReduce: 2 (workgroup parallelism + synchronization)
    /// - Upcast: 3 (vectorization)
    /// - Reduce: 4 (reduction loops)
    /// - Unroll: 5 (unrolled loops, innermost)
    pub const fn priority(self) -> i32 {
        match self {
            Self::Device => -2,
            Self::Weak | Self::Loop => -1,
            Self::Global | Self::Thread => 0,
            Self::Warp => 1,
            Self::Local | Self::GroupReduce => 2,
            Self::Upcast => 3,
            Self::Reduce => 4,
            Self::Unroll => 5,
            Self::Placeholder => -3,
        }
    }

    /// Returns the single-letter code for this axis type.
    ///
    /// Used in kernel name generation and debug output.
    ///
    /// **Letter Codes:**
    /// - L: Loop
    /// - g: Global
    /// - t: Thread
    /// - w: Warp
    /// - l: Local
    /// - G: GroupReduce
    /// - u: Upcast
    /// - R: Reduce
    /// - r: Unroll
    pub const fn letter(self) -> char {
        match self {
            Self::Device => 'd',
            Self::Weak | Self::Loop => 'L',
            Self::Global => 'g',
            Self::Thread => 't',
            Self::Warp => 'w',
            Self::Local => 'l',
            Self::GroupReduce => 'G',
            Self::Upcast => 'u',
            Self::Reduce => 'R',
            Self::Unroll => 'r',
            Self::Placeholder => 'P',
        }
    }

    /// Returns true if this is a parallelizable axis type.
    ///
    /// Parallel axes represent GPU/thread dispatch dimensions that don't
    /// contribute to accumulator placement in reduce_to_acc.
    pub const fn is_parallel(self) -> bool {
        matches!(self, Self::Global | Self::Thread | Self::Local | Self::Warp)
    }

    /// Returns true if this is a reduction axis type.
    pub const fn is_reduce(self) -> bool {
        matches!(self, Self::Reduce | Self::GroupReduce | Self::Unroll)
    }
}

impl PartialOrd for AxisType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AxisType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority().cmp(&other.priority())
    }
}

impl std::fmt::Display for AxisType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.letter())
    }
}

/// State of range numbering for kernel deduplication.
///
/// Ranges go through two states during the compilation pipeline:
/// - `Unrenumbered`: Created during rangeify with unique IDs for graph construction
/// - `Renumbered`: Assigned sequential IDs starting from 0 within each kernel
///
/// The enum makes the renumber_range pattern naturally idempotent:
/// it only matches `Unrenumbered` variants and produces `Renumbered` variants.
#[derive(Debug, Clone, Eq)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum AxisId {
    /// Range created during rangeify, not yet renumbered.
    Unrenumbered(usize),
    /// Range renumbered for kernel deduplication.
    Renumbered(usize),
    /// Range identity derived structurally from an unrenumbered parent.
    UnrenumberedPath(SmallVec<[usize; 4]>),
    /// Range identity derived structurally from a renumbered parent.
    RenumberedPath(SmallVec<[usize; 4]>),
}

impl AxisId {
    /// Get the numeric value, regardless of state.
    pub fn value(&self) -> usize {
        match self {
            AxisId::Unrenumbered(n) | AxisId::Renumbered(n) => *n,
            AxisId::UnrenumberedPath(path) | AxisId::RenumberedPath(path) => path[0],
        }
    }

    /// Full Tinygrad RANGE axis argument excluding its trailing axis type.
    pub fn path(&self) -> &[usize] {
        match self {
            AxisId::Unrenumbered(n) | AxisId::Renumbered(n) => std::slice::from_ref(n),
            AxisId::UnrenumberedPath(path) | AxisId::RenumberedPath(path) => path,
        }
    }

    /// Append a structural split component without allocating a new axis id.
    pub fn child(&self, component: usize) -> Self {
        let mut path = SmallVec::from_slice(self.path());
        path.push(component);
        if self.is_renumbered() { AxisId::RenumberedPath(path) } else { AxisId::UnrenumberedPath(path) }
    }

    /// Identity of the serial REDUCE loop derived from a GROUP_REDUCE axis.
    ///
    /// This is a child axis, not a new root axis: retaining the parent path is
    /// required when grouped reduction follows one or more range splits. Range
    /// splitting owns child components 0 and 1; component 2 is the derived-loop
    /// branch and therefore cannot alias either structural split child.
    pub fn group_reduce_loop(&self) -> Self {
        self.child(2)
    }

    /// Renderer spelling used by Tinygrad's `range_str` (`0`, `0_1`, ...).
    pub fn name(&self) -> String {
        self.path().iter().map(usize::to_string).collect::<Vec<_>>().join("_")
    }

    /// Check if this range has been renumbered.
    pub fn is_renumbered(&self) -> bool {
        matches!(self, AxisId::Renumbered(_) | AxisId::RenumberedPath(_))
    }
}

/// `Ord`, `PartialEq` and `Hash` share the `(is_renumbered, path)` key so that the
/// single-component and path spellings of one axis are one value everywhere.
impl PartialEq for AxisId {
    fn eq(&self, other: &Self) -> bool {
        self.is_renumbered() == other.is_renumbered() && self.path() == other.path()
    }
}

impl std::hash::Hash for AxisId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.is_renumbered().hash(state);
        self.path().hash(state);
    }
}

impl PartialOrd for AxisId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AxisId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.is_renumbered().cmp(&other.is_renumbered()).then_with(|| self.path().cmp(other.path()))
    }
}

impl std::fmt::Display for AxisId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AxisId::Unrenumbered(n) => write!(f, "U{}", n),
            AxisId::Renumbered(n) => write!(f, "R{}", n),
            AxisId::UnrenumberedPath(_) => write!(f, "U{}", self.name()),
            AxisId::RenumberedPath(_) => write!(f, "R{}", self.name()),
        }
    }
}

/// Reduction operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum ReduceOp {
    /// Sum reduction (a + b).
    Add,
    /// Product reduction (a * b).
    Mul,
    /// Maximum reduction (max(a, b)).
    Max,
    /// Minimum reduction (min(a, b)).
    Min,
}

/// Unary operation types.
///
/// All unary operations preserve the input dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, strum::AsRefStr, strum::VariantNames, strum::VariantArray)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum UnaryOp {
    /// Negation: -x
    Neg,
    /// Logical/bitwise NOT: !x (bool) or ~x (int)
    Not,
    /// Absolute value: |x|
    Abs,
    /// Square root: √x
    Sqrt,
    /// Reciprocal square root: 1/√x
    Rsqrt,
    /// Natural exponential: e^x
    Exp,
    /// Base-2 exponential: 2^x
    Exp2,
    /// Natural logarithm: ln(x)
    Log,
    /// Base-2 logarithm: log₂(x)
    Log2,
    /// Sine: sin(x) (float only)
    Sin,
    /// Cosine: cos(x) (float only)
    Cos,
    /// Tangent: tan(x) (float only)
    Tan,
    /// Reciprocal: 1/x
    Reciprocal,
    /// Truncate towards zero (remove fractional part)
    Trunc,
    /// Floor: round towards -∞
    Floor,
    /// Ceiling: round towards +∞
    Ceil,
    /// Round: round to nearest integer (half to even)
    Round,
    /// Sign: -1 for negative, 0 for zero, 1 for positive
    Sign,
    /// Error function: erf(x) (float only)
    Erf,
    /// Square: x²
    Square,
}

/// Binary operation types.
///
/// Arithmetic operations preserve the LHS dtype.
/// Comparison operations (Lt, Eq, Ne) always return DType::Bool.
/// Bitwise operations (And, Or, Xor, Shl, Shr) preserve dtype and require int/bool types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, strum::AsRefStr, strum::VariantNames, strum::VariantArray)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum BinaryOp {
    // Arithmetic operations
    /// Addition: a + b
    Add,
    /// Multiplication: a * b
    Mul,
    /// Subtraction: a - b
    Sub,
    /// Floor modulo: `a - floor(a / b) * b` (sign of divisor).
    FloorMod,
    /// C-style remainder (sign of dividend).
    CMod,
    /// Maximum: max(a, b)
    Max,
    /// Power: a^b
    Pow,
    /// Integer floor division (rounds toward negative infinity).
    FloorDiv,
    /// C-style integer division (truncates toward zero).
    CDiv,
    /// Float division: a / b — exact IEEE 754 division. Float dtypes only.
    Fdiv,

    // Comparison operations
    /// Less than: a < b
    Lt,
    /// Less than or equal: a <= b
    Le,
    /// Equality: a == b
    Eq,
    /// Inequality: a != b
    Ne,
    /// Greater than: a > b
    Gt,
    /// Greater than or equal: a >= b
    Ge,

    // Bitwise operations (int/bool only)
    /// Bitwise AND: a & b
    And,
    /// Bitwise OR: a | b
    Or,
    /// Bitwise XOR: a ^ b
    Xor,
    /// Left shift: a << b
    Shl,
    /// Right shift: a >> b
    Shr,

    // Special operations
    /// Threefry PRNG: threefry(x, key) -> uint64
    Threefry,
}

impl BinaryOp {
    /// Returns true if this is a comparison operation.
    pub fn is_comparison(self) -> bool {
        matches!(self, Self::Lt | Self::Le | Self::Eq | Self::Ne | Self::Gt | Self::Ge)
    }

    /// Returns true if this is an arithmetic operation.
    pub fn is_arithmetic(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Mul
                | Self::Sub
                | Self::FloorMod
                | Self::CMod
                | Self::Max
                | Self::Pow
                | Self::FloorDiv
                | Self::CDiv
                | Self::Fdiv
        )
    }

    /// Returns true if this is a bitwise operation.
    pub fn is_bitwise(self) -> bool {
        matches!(self, Self::And | Self::Or | Self::Xor | Self::Shl | Self::Shr)
    }

    /// Returns true if this operation is associative.
    pub fn is_associative(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::And | Self::Or | Self::Max)
    }

    /// Returns true if this operation is commutative.
    pub fn is_commutative(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Eq | Self::Ne | Self::And | Self::Or | Self::Xor | Self::Max)
    }

    /// Returns true if this operation is idempotent (f(x, x) = x).
    pub fn is_idempotent(self) -> bool {
        matches!(self, Self::Or | Self::And | Self::Max)
    }

    /// Returns the identity element for this operation at the given dtype.
    ///
    /// `pop_const` substitutes this when no const factor is present, so a
    /// caller can safely treat the returned tuple as `(remainder, factor)`
    /// with no Option unwrapping. Supports ADD, MUL, MAX (with their natural
    /// identities); other ops have no canonical identity and panic.
    pub fn identity_element(self, dtype: DType) -> ConstValue {
        match self {
            Self::Add => {
                if dtype.is_float() {
                    ConstValue::Float(0.0)
                } else {
                    ConstValue::Int(0)
                }
            }
            Self::Mul => {
                if dtype.is_float() {
                    ConstValue::Float(1.0)
                } else {
                    ConstValue::Int(1)
                }
            }
            Self::Max => {
                if dtype.is_float() {
                    ConstValue::Float(dtype.analysis_bounds().0)
                } else {
                    ConstValue::Int(dtype.min_value() as i64)
                }
            }
            other => panic!("identity_element: no identity for {other:?}"),
        }
    }
}

/// Ternary operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, strum::AsRefStr, strum::VariantNames, strum::VariantArray)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum TernaryOp {
    /// Conditional selection: condition ? true_val : false_val
    Where,
    /// Multiply-accumulate: a * b + c (fused operation)
    MulAcc,
}

/// ALU operations accepted directly by a renderer. This is the Rust equivalent
/// of tinygrad's `code_for_op.keys()`; decomposition must be derived from this
/// set rather than from a backend-independent assumption.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RendererOps {
    pub unary: std::collections::HashSet<UnaryOp>,
    pub binary: std::collections::HashSet<BinaryOp>,
    pub ternary: std::collections::HashSet<TernaryOp>,
}

impl RendererOps {
    pub fn all() -> Self {
        use strum::VariantArray;
        Self {
            unary: UnaryOp::VARIANTS.iter().copied().collect(),
            binary: BinaryOp::VARIANTS.iter().copied().collect(),
            ternary: TernaryOp::VARIANTS.iter().copied().collect(),
        }
    }

    pub fn supports_unary(&self, op: UnaryOp) -> bool {
        self.unary.contains(&op)
    }

    pub fn supports_binary(&self, op: BinaryOp) -> bool {
        self.binary.contains(&op)
    }

    pub fn supports_ternary(&self, op: TernaryOp) -> bool {
        self.ternary.contains(&op)
    }
}

/// Per-source upcast axes for WMMA operations.
///
/// Each WMMA source (A, B, C) may have different upcast axis sizes
/// based on `elements_per_thread`. For example, CUDA 8-16-16 with
/// `elements_per_thread=(8,4,4)` produces A=8, B=4, C=4 element groups.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct WmmaUpcastAxes {
    /// A operand upcast axes (input matrix).
    pub a: Vec<(AxisId, usize)>,
    /// B operand upcast axes (input matrix).
    pub b: Vec<(AxisId, usize)>,
    /// C operand upcast axes (output/accumulator).
    pub c: Vec<(AxisId, usize)>,
}

impl WmmaUpcastAxes {
    /// Returns deduplicated axis IDs from all three operands.
    pub fn all_axis_ids(&self) -> Vec<AxisId> {
        let mut ids: Vec<AxisId> =
            self.a.iter().chain(self.b.iter()).chain(self.c.iter()).map(|(id, _)| id.clone()).collect();
        ids.sort();
        ids.dedup();
        ids
    }

    /// Returns the axes for operand at the given index (0=A, 1=B, 2=C).
    pub fn by_index(&self, index: usize) -> &[(AxisId, usize)] {
        match index {
            0 => &self.a,
            1 => &self.b,
            2 => &self.c,
            _ => panic!("WMMA operand index must be 0, 1, or 2"),
        }
    }

    /// Returns the product of axis sizes for operand at given index.
    pub fn source_size(&self, index: usize) -> usize {
        self.by_index(index).iter().map(|(_, s)| s).product::<usize>().max(1)
    }
}

/// Identifies which renderer / TC backend a kernel or WMMA op was generated for.
///
/// More granular than [`DeviceSpec`] (which describes the runtime target): the
/// renderer selects the tensor-core table and codegen gating for that target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum RendererDevice {
    Cpu,
    CudaSm75,
    CudaSm80,
    CudaSm89,
    Metal,
    AmdRdna3,
    AmdRdna4,
    AmdCdna3,
    AmdCdna4,
    IntelXe,
    WebGpu,
}

impl RendererDevice {
    /// Stable canonical name — used for kernel cache hashing and debug output.
    pub const fn canonical(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::CudaSm75 => "CUDA_SM75",
            Self::CudaSm80 => "CUDA_SM80",
            Self::CudaSm89 => "CUDA_SM89",
            Self::Metal => "Metal",
            Self::AmdRdna3 => "AMD_RDNA3",
            Self::AmdRdna4 => "AMD_RDNA4",
            Self::AmdCdna3 => "AMD_CDNA3",
            Self::AmdCdna4 => "AMD_CDNA4",
            Self::IntelXe => "IntelXe",
            Self::WebGpu => "WebGPU",
        }
    }

    /// True when a benchmark's software cache eviction — a stream through a
    /// *host* scratch buffer — reaches the cache the timed kernel runs against.
    /// Only the CPU executes out of host memory: on every GPU backend the
    /// stream costs host time, leaves the device caches hot, and evicts the
    /// host lines the next dispatch's submit path is about to touch.
    pub const fn benchmark_evicts_via_host_stream(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// True when the device drops its clock between dispatches and needs a
    /// stretch of load before a timing means anything; the CPU is timed as it is.
    pub const fn idles_its_clock(&self) -> bool {
        !matches!(self, Self::Cpu)
    }
}

impl core::fmt::Display for RendererDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.canonical())
    }
}

/// Metadata for WMMA (Warp Matrix Multiply-Accumulate) operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct WmmaMetadata {
    /// Operation name (e.g., "WMMA_INSTRUCTION").
    pub name: String,
    /// Matrix dimensions (N, M, K).
    pub dims: (usize, usize, usize),
    /// Input matrix dtype.
    pub dtype_in: DType,
    /// Output/accumulator dtype.
    pub dtype_out: DType,
    /// Renderer / TC backend that produced this WMMA.
    pub device: RendererDevice,
    /// Thread count.
    pub threads: usize,
    /// Per-source expansion axes `(A contract, B contract, output unroll)`.
    /// Cleared after `expander2` shapes the WMMA sources and output.
    pub upcast_axes: Option<WmmaUpcastAxes>,
    /// TC reduce axis IDs (used for exclude_args in expansion).
    pub reduce_axes: Vec<AxisId>,
}

/// Wrapper for ConstValue that implements Eq and Hash.
///
/// Floats don't implement Eq/Hash due to IEEE 754 NaN semantics (NaN != NaN).
/// This wrapper uses bitwise comparison: two floats are equal if their bit patterns match.
/// This means:
/// - NaN values with identical bit patterns are considered equal
/// - Different NaN representations are not equal
/// - This is consistent with hash consing requirements
#[derive(Debug, Clone, Copy)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ConstValueHash(pub ConstValue);

impl PartialEq for ConstValueHash {
    fn eq(&self, other: &Self) -> bool {
        match (self.0, other.0) {
            (ConstValue::Invalid, ConstValue::Invalid) => true,
            (ConstValue::Int(a), ConstValue::Int(b)) => a == b,
            (ConstValue::UInt(a), ConstValue::UInt(b)) => a == b,
            (ConstValue::Float(a), ConstValue::Float(b)) => a.to_bits() == b.to_bits(),
            (ConstValue::Bool(a), ConstValue::Bool(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for ConstValueHash {}

impl Hash for ConstValueHash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (discriminant(&self.0)).hash(state);
        match self.0 {
            ConstValue::Invalid => {}
            ConstValue::Int(v) => v.hash(state),
            ConstValue::UInt(v) => v.hash(state),
            ConstValue::Float(v) => v.to_bits().hash(state),
            ConstValue::Bool(v) => v.hash(state),
        }
    }
}
