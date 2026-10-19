//! Device abstraction.
//!
//! This module provides a unified Device abstraction that owns:
//! - **Renderer**: Transforms UOp graphs into source code (ProgramSpec)
//! - **Compiler**: Transforms source code into executable bytes
//! - **Runtime**: Creates executable Programs from compiled bytes
//! - **Allocator**: Manages memory allocation for buffers
//!
//! This design allows multiple backends (LLVM, CUDA, Metal, WebGPU) to coexist
//! and share compiled kernels via the method cache.

use std::collections::HashMap;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use svod_dtype::{AddrSpace, DType, DeviceSpec, ScalarDType};
use svod_ir::ops;
use svod_ir::{
    BINARY_STAGE_IDENTITY_VERSION, BinaryOp, BinaryStageIdentity, ConstValue, Op, SOURCE_STAGE_IDENTITY_VERSION,
    SourceStageIdentity, StageAbiParam, StageAbiParamKind, StageDigest, TernaryOp, UOp, UnaryOp,
};

use crate::allocator::Allocator;
use snafu::OptionExt;

use crate::error::{Error, Result, VarOutOfBoundsSnafu, WrongStageSnafu};

/// A compiled, executable kernel program.
///
/// This trait abstracts over different backend executors (LLVM JIT, CUDA, Metal, etc.).
/// Each backend implements this to provide unified execution interface.
///
/// Implementations must be stateless and reentrant from the host perspective.
/// The runtime caches and shares programs across execution plans, and may invoke
/// the same program from multiple host threads when dependency analysis proves
/// the buffer accesses are independent.
///
/// # Calling convention
///
/// Variable values are passed as a positional array (`vals`) rather than a
/// named HashMap. The order matches `var_names` in `CompiledSpec`.
pub trait Program: Send + Sync {
    /// Execute the kernel with given buffers and variable values.
    ///
    /// # Arguments
    ///
    /// * `buffers` - Raw pointers to buffer data (input and output buffers)
    /// * `vals` - Variable values in positional order (matches `var_names` in CompiledSpec)
    /// * `global_size` - Global work size; the CPU backend reads `[0]` as its
    ///   thread count (`None` or 1 runs single-threaded)
    /// * `local_size` - Local work size (GPU backends; CPU ignores it)
    /// * `wait` - Block until this dispatch completes before returning. GPU
    ///   backends submit asynchronously and rely on the device timeline for
    ///   ordering, so `wait=false` returns right after submit. Pass `true`
    ///   only when the caller needs completion *without* a subsequent synchronizing read
    ///   (e.g. benchmark timing). Synchronous backends (CPU) ignore it.
    ///
    /// # Safety
    ///
    /// This is unsafe because:
    /// - Buffer pointers must be valid and properly aligned
    /// - Buffer sizes must match what the kernel expects
    /// - Caller must ensure no data races during execution
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
        wait: bool,
    ) -> Result<()>;

    /// Get the kernel name (for debugging/profiling).
    fn name(&self) -> &str;

    /// Run synchronously and report the dispatch's duration: on the GPU's own
    /// clock where the backend stamps it, else the wall clock around the
    /// synchronous dispatch (the default). A stamping backend returns `None`
    /// only for a dispatch whose stamp it lost, so a caller never times one
    /// kernel on two clocks.
    ///
    /// Whether the run starts on warm or cold caches is each backend's
    /// documented choice; a cache flush is never inside the span.
    ///
    /// # Safety
    ///
    /// Same contract as [`Program::execute`].
    unsafe fn execute_timed(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    ) -> Result<Option<std::time::Duration>> {
        let start = std::time::Instant::now();
        unsafe { self.execute(buffers, vals, global_size, local_size, true) }.map(|()| Some(start.elapsed()))
    }

    /// Downcast hook so a backend graph factory can recover its concrete
    /// program type (to read backend-specific fields) when pre-building dispatch
    /// packets. Default returns nothing graphable.
    fn as_any(&self) -> &dyn std::any::Any {
        &()
    }

    /// Mint reusable logical per-plan state, or `None` for per-call dispatch.
    /// An `ExecutionPlan` calls this once on its first kernel and reuses it for
    /// dispatch, native replay templates, profiling, and epoch-scoped hardware
    /// lane acquisition. Default `None`: the backend's `execute` is already
    /// self-contained.
    fn new_exec_context(&self) -> Result<Option<Box<dyn PlanContext>>> {
        Ok(None)
    }

    /// Static GPU resource usage (VGPR/SGPR/LDS/scratch) decoded from the
    /// compiled kernel descriptor, for profiling/occupancy. Default `None`:
    /// backends without a descriptor (CPU) report nothing.
    fn resource_usage(&self) -> Option<crate::profile::KernelResources> {
        None
    }
}

/// One graphable kernel: a program plus its fixed buffer pointers and launch
/// dims, captured once. Replay may replace the captured buffer addresses and
/// scalar vars without rebuilding the backend command stream.
pub struct GraphKernel<'a> {
    pub program: &'a dyn Program,
    pub buffers: Vec<*mut u8>,
    pub vals: Vec<i64>,
    pub global_size: Option<[usize; 3]>,
    pub local_size: Option<[usize; 3]>,
    /// Emission-order indices of the producer kernels this kernel must wait on
    /// (RAW/WAR/WAW hazards over resolved buffer GVAs). Computed by the host
    /// hazard analysis in the same flatten order the kernels are emitted, so a
    /// DAG-aware backend can strip the per-dispatch BARRIER bit and gate only on
    /// these producers' completion signals. Empty = no producer in this graph.
    pub deps: Vec<usize>,
}

/// A pre-captured kernel chain replayed with one submit. Backends that can
/// pre-build their dispatch packets implement this so repeated inference pays
/// per-graph, not per-kernel, launch cost. Replay is equivalent to running every
/// captured kernel in order.
pub trait Graph: Send + Sync {
    /// Re-dispatch the captured chain. Buffers and vars are flattened in capture
    /// order; empty slices replay the captured values.
    fn replay(&self, buffers: &[u64], vals: &[i64]) -> Result<()>;

    /// Completion token covering every replay this graph has made so far.
    /// Backends without scoped sync return `None`.
    fn completion_token(&self) -> Option<Arc<dyn crate::sync::CompletionToken>> {
        None
    }

    /// Order the next replay after other contexts' in-flight work it
    /// conflicts with; see [`PlanContext::order_after`].
    fn order_after(&self, _accesses: &[(&crate::Buffer, bool)]) -> Result<()> {
        Ok(())
    }

    /// Replay a profiling-specific linked variant and return ready per-dispatch
    /// timestamps in capture order. Backends without graph timestamps return
    /// `None`, allowing the runtime to retain its per-call fallback.
    fn replay_profiled(
        &self,
        _buffers: &[u64],
        _vals: &[i64],
    ) -> Result<Option<Vec<Arc<dyn crate::DispatchTimestamps>>>> {
        Ok(None)
    }
}

/// Reusable logical per-plan state. A backend may acquire an exclusive hardware
/// lane for one replay epoch, but the context itself does not imply lifetime
/// queue ownership. Minted by [`Program::new_exec_context`]; backends with no
/// reusable context return `None` and use per-call [`Program::execute`].
pub trait PlanContext: Send + Sync {
    /// Dispatch one kernel of the plan onto this context. `program` belongs to
    /// the same plan and therefore the same backend that minted this context
    /// (a plan is single-device) — a construction invariant, not a runtime
    /// check. Submits asynchronously like [`Program::execute`] with `wait=false`.
    ///
    /// `profile` requests a per-dispatch HW timestamp handle (`None` otherwise,
    /// e.g. CPU). **The caller MUST retain the returned handle until after
    /// [`PlanContext::synchronize`]**: a profiling backend may bracket the async
    /// dispatch with GPU-clock probes that write into scratch the handle owns, so
    /// dropping it early frees that scratch while the GPU is still writing. Pass
    /// `false` on the fire-and-forget path that discards the handle.
    ///
    /// # Safety
    ///
    /// Same contract as [`Program::execute`]: buffer pointers must be valid and
    /// correctly sized, and the caller must avoid data races.
    unsafe fn dispatch(
        &self,
        program: &dyn Program,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
        profile: bool,
    ) -> Result<Option<Arc<dyn crate::DispatchTimestamps>>>;

    /// Order this context's next submissions after other contexts' in-flight
    /// work they conflict with, given every storage the plan touches as
    /// `(buffer, writes)`: the writers of what it reads, and every access to
    /// what it writes. Backends whose contexts share one in-order queue need
    /// nothing (the default).
    fn order_after(&self, _accesses: &[(&crate::Buffer, bool)]) -> Result<()> {
        Ok(())
    }

    /// Completion token covering every submission this context has made so
    /// far. Backends without scoped sync return `None`.
    fn completion_token(&self) -> Option<Arc<dyn crate::sync::CompletionToken>> {
        None
    }

    /// Replay an execution plan's already-linked neutral topology. Hardware
    /// backends may lower/link it on the first call and patch only invocation
    /// state thereafter. A declined outcome keeps the generic per-operation path.
    fn replay_linked_plan(
        &self,
        _plan: &crate::hcq::SemanticLinkedPlan,
        _calls: &[PlanCall<'_>],
    ) -> Result<NativeReplayOutcome> {
        Ok(NativeReplayOutcome::Declined(NativeReplayDecline::BackendUnsupported))
    }

    /// Drain this context's in-flight work (profiled-timestamp harvest).
    fn synchronize(&self) -> Result<()>;

    /// End one direct-dispatch replay epoch without waiting for GPU completion.
    /// Exclusive-lane backends release publication authority here while
    /// retaining queue identity for FIFO ordering in the next epoch.
    fn finish_replay(&self) -> Result<()> {
        Ok(())
    }

    /// Arm hardware performance counters for subsequent profiling dispatches on
    /// this context (empty disables), returning how many of `counters` this
    /// backend collects: those naming another backend are dropped. Default
    /// collects none. Counters are reported via
    /// [`DispatchTimestamps::counters`](crate::DispatchTimestamps::counters).
    fn set_pmc(&self, _counters: &[crate::profile::PmcCounter]) -> usize {
        0
    }

    /// Whether hardware counter collection is currently available on this
    /// context (backend supports PMC and the GPU is in a stable power state).
    /// Default `false`.
    fn pmc_available(&self) -> bool {
        false
    }

    /// The counters [`set_pmc`](Self::set_pmc) collects for a caller that asked
    /// for the default selection. Empty on backends without PMC.
    fn pmc_default(&self) -> Vec<crate::profile::PmcCounter> {
        Vec::new()
    }
}

/// Why an execution plan could not use backend-native linked replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeReplayDecline {
    NoCompiledProgram,
    NoPlanContext,
    MixedComputeDevices { expected: DeviceSpec, actual: DeviceSpec },
    ForeignProgramEndpoint { operation: u64, argument: usize, expected: DeviceSpec, actual: DeviceSpec },
    IncompatibleProgramAllocation { operation: u64, argument: usize, expected: DeviceSpec },
    ForeignCopyEndpoint { operation: u64, endpoint: CopyEndpoint, expected: DeviceSpec, actual: DeviceSpec },
    IncompatibleCopyAllocation { operation: u64, endpoint: CopyEndpoint, expected: DeviceSpec },
    StagedCopy { operation: usize },
    BackendUnsupported,
}

/// Result of attempting backend-native linked replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeReplayOutcome {
    Executed,
    Declined(NativeReplayDecline),
}

/// Copy endpoint whose ownership prevented native replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyEndpoint {
    Destination,
    Source,
}

/// Current invocation values for one captured plan operation. Structure and
/// programs are stable; buffer addresses, scalar vars, and launch geometry are
/// deliberately supplied on every replay.
pub enum PlanCall<'a> {
    Program {
        program: &'a dyn Program,
        buffers: &'a [u64],
        vals: &'a [i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    },
    Copy {
        dst: u64,
        src: u64,
        bytes: usize,
    },
    Unsupported,
}

/// Compilation result carrying source (JIT) or bytes (AOT).
///
/// Different backends need different information:
/// - LLVM JIT: needs source code to compile during runtime
/// - CUDA: needs PTX/CUBIN bytes to load
/// - Metal: needs metallib bytes to load
///
/// This design allows the RuntimeFactory to access whatever it needs
/// without requiring separate code paths for JIT vs AOT backends.
#[derive(Debug, Clone)]
pub struct CompiledSpec {
    /// Entry point function name
    pub name: String,

    /// Source code (for JIT backends like LLVM)
    /// Set to Some(...) for LLVM JIT, None for AOT backends
    pub src: Option<String>,

    /// Compiled bytes (for AOT backends like CUDA/Metal)
    /// Empty for LLVM JIT, populated for AOT backends
    pub bytes: Vec<u8>,

    /// Original AST for cache key construction via hash consing
    pub ast: Arc<UOp>,

    /// Variable names in order for populating vars array at runtime.
    /// Includes runtime variables such as core_id.
    pub var_names: Vec<String>,

    /// Symbolic global work size for dispatch.
    pub global_size: [Arc<UOp>; 3],

    /// Symbolic local work size for dispatch. None means direct global-id execution.
    pub local_size: Option<[Arc<UOp>; 3]>,

    /// Number of buffer arguments (for CIF construction at compile time).
    pub buf_count: usize,

    /// Complete kernel argument ABI in source-signature order.
    pub abi: Vec<AbiParamDescriptor>,

    stage_identity: Option<BinaryStageIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AbiParamKind {
    Storage(AddrSpace),
    Scalar,
}

/// One external PARAM argument. The vector containing these descriptors is
/// always sorted by `slot` and is the sole source of kernel ABI ordering.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AbiParamDescriptor {
    pub slot: usize,
    pub kind: AbiParamKind,
    pub dtype: DType,
    pub name: Option<String>,
}

impl AbiParamDescriptor {
    pub fn from_param(param: &Arc<UOp>) -> Result<Self> {
        let Op::Param(ops::Param { arg, .. }) = param.op() else {
            return Err(Error::ProgramAbiMismatch {
                reason: format!("ABI descriptor source is non-PARAM {:?}", param.op()),
            });
        };
        if arg.slot == usize::MAX {
            return Err(Error::UnassignedProgramParam {
                stage: "ABI descriptor construction",
                param: arg.name.clone().unwrap_or_else(|| format!("{:?} storage", arg.addrspace)),
            });
        }
        if arg.addrspace.is_none() && arg.name.is_none() {
            return Err(Error::ProgramAbiMismatch { reason: format!("scalar PARAM in slot {} has no name", arg.slot) });
        }
        if arg.dtype != param.dtype() {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "PARAM slot {} metadata dtype {:?} does not match UOp dtype {:?}",
                    arg.slot,
                    arg.dtype,
                    param.dtype()
                ),
            });
        }
        if arg.addrspace.is_none() && param.dtype() != DType::Int32 {
            return Err(Error::ProgramAbiMismatch {
                reason: format!("scalar PARAM slot {} has non-canonical final ABI dtype {:?}", arg.slot, param.dtype()),
            });
        }
        Ok(Self {
            slot: arg.slot,
            kind: arg.addrspace.map_or(AbiParamKind::Scalar, AbiParamKind::Storage),
            dtype: param.dtype(),
            name: arg.name.clone(),
        })
    }

    pub fn is_storage(&self) -> bool {
        matches!(self.kind, AbiParamKind::Storage(_))
    }
}

fn stage_abi(abi: &[AbiParamDescriptor]) -> Vec<StageAbiParam> {
    abi.iter()
        .map(|param| StageAbiParam {
            slot: param.slot,
            kind: match param.kind {
                AbiParamKind::Storage(space) => StageAbiParamKind::Storage(space),
                AbiParamKind::Scalar => StageAbiParamKind::Scalar,
            },
            dtype: param.dtype.clone(),
            name: param.name.clone(),
        })
        .collect()
}

fn sha256(bytes: &[u8]) -> StageDigest {
    StageDigest(Sha256::digest(bytes).into())
}

/// `io::Write` over a hasher: sha2 0.11 no longer implements it.
struct HashWriter(Sha256);

impl std::io::Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn linear_sha256(linear: &Arc<UOp>) -> Result<StageDigest> {
    let graph = svod_ir::CanonicalGraph::from_root("source-stage-linear-v2", linear).map_err(|error| {
        Error::ProgramStageMismatch { stage: "SOURCE", reason: format!("cannot encode LINEAR identity: {error}") }
    })?;
    let mut hasher = HashWriter(Sha256::new());
    graph.encode_into(&mut hasher).map_err(|error| Error::ProgramStageMismatch {
        stage: "SOURCE",
        reason: format!("cannot serialize LINEAR identity: {error}"),
    })?;
    Ok(StageDigest(hasher.0.finalize().into()))
}

fn source_stage_identity_from_parts(
    abi: &[AbiParamDescriptor],
    target: &DeviceSpec,
    entry_name: String,
    linear: &Arc<UOp>,
    source: &str,
) -> Result<SourceStageIdentity> {
    Ok(SourceStageIdentity {
        version: SOURCE_STAGE_IDENTITY_VERSION,
        abi: stage_abi(abi),
        target: target.clone(),
        entry_name,
        linear_sha256: linear_sha256(linear)?,
        source_sha256: sha256(source.as_bytes()),
    })
}

/// Construct the semantic identity for one rendered SOURCE stage. This is the
/// only place the LINEAR digest is computed; later stages read it back through
/// [`minted_source_stage_identity`].
pub fn source_stage_identity(
    info: &svod_ir::ProgramInfo,
    abi: &[AbiParamDescriptor],
    linear: &Arc<UOp>,
    source: &str,
) -> Result<SourceStageIdentity> {
    source_stage_identity_from_parts(abi, &info.target, info.function_name(), linear, source)
}

/// The identity a SOURCE stage was minted with, re-checked against the PROGRAM
/// that now carries it. Every field except the LINEAR digest is re-derived
/// here; the digest is the one computed when the stage was rendered, so a
/// PROGRAM flowing render → compile → load hashes its LINEAR exactly once.
pub fn minted_source_stage_identity(
    info: &svod_ir::ProgramInfo,
    abi: &[AbiParamDescriptor],
    source: &Arc<UOp>,
) -> Result<SourceStageIdentity> {
    let Op::Source(ops::Source { code, identity }) = source.op() else {
        return Err(Error::ProgramStageMismatch {
            stage: "SOURCE",
            reason: format!("expected SOURCE, got {:?}", source.op()),
        });
    };
    let minted = identity.as_ref().ok_or_else(|| Error::ProgramStageMismatch {
        stage: "SOURCE",
        reason: "stage has no semantic identity".into(),
    })?;
    let expected = SourceStageIdentity {
        version: SOURCE_STAGE_IDENTITY_VERSION,
        abi: stage_abi(abi),
        target: info.target.clone(),
        entry_name: info.function_name(),
        linear_sha256: minted.linear_sha256,
        source_sha256: sha256(code.as_bytes()),
    };
    if **minted != expected {
        return Err(Error::ProgramStageMismatch {
            stage: "SOURCE",
            reason: format!("expected {expected:?}, got {minted:?}"),
        });
    }
    Ok(expected)
}

/// Construct the semantic identity for one compiled BINARY stage.
pub fn binary_stage_identity(source: SourceStageIdentity, compiler_key: &str, bytes: &[u8]) -> BinaryStageIdentity {
    BinaryStageIdentity {
        version: BINARY_STAGE_IDENTITY_VERSION,
        source,
        compiler_key: compiler_key.to_string(),
        binary_sha256: sha256(bytes),
    }
}

/// Validate a SOURCE UOp against an independently derived identity.
pub fn validate_source_stage(source: &Arc<UOp>, expected: &SourceStageIdentity) -> Result<()> {
    let Op::Source(ops::Source { code, identity }) = source.op() else {
        return Err(Error::ProgramStageMismatch {
            stage: "SOURCE",
            reason: format!("expected SOURCE, got {:?}", source.op()),
        });
    };
    let actual = identity.as_ref().ok_or_else(|| Error::ProgramStageMismatch {
        stage: "SOURCE",
        reason: "stage has no semantic identity".into(),
    })?;
    if **actual != *expected || actual.source_sha256 != sha256(code.as_bytes()) {
        return Err(Error::ProgramStageMismatch {
            stage: "SOURCE",
            reason: format!("expected {expected:?}, got {actual:?}"),
        });
    }
    Ok(())
}

/// Validate a BINARY UOp against an independently derived identity.
pub fn validate_binary_stage(binary: &Arc<UOp>, expected: &BinaryStageIdentity) -> Result<Vec<u8>> {
    let Op::ProgramBinary(ops::ProgramBinary { bytes, identity }) = binary.op() else {
        return Err(Error::ProgramStageMismatch {
            stage: "BINARY",
            reason: format!("expected BINARY, got {:?}", binary.op()),
        });
    };
    let actual = identity.as_ref().ok_or_else(|| Error::ProgramStageMismatch {
        stage: "BINARY",
        reason: "stage has no semantic identity".into(),
    })?;
    if **actual != *expected || actual.binary_sha256 != sha256(bytes) {
        return Err(Error::ProgramStageMismatch {
            stage: "BINARY",
            reason: format!("expected {expected:?}, got {actual:?}"),
        });
    }
    Ok(bytes.clone())
}

/// Validate the complete external kernel ABI and its compact runtime
/// projections. PARAM slots choose signature positions; buffer and scalar
/// vectors remain compact in descriptor order.
pub fn validate_abi_descriptors(
    abi: &[AbiParamDescriptor],
    expected_buf_count: usize,
    expected_var_names: &[String],
) -> Result<()> {
    let mut previous_slot = None;
    let mut buf_count = 0usize;
    let mut var_names = Vec::new();
    let mut unique_names = std::collections::HashSet::new();

    for descriptor in abi {
        if descriptor.slot == usize::MAX {
            return Err(Error::UnassignedProgramParam {
                stage: "ABI descriptor validation",
                param: descriptor.name.clone().unwrap_or_else(|| format!("{:?} storage", descriptor.kind)),
            });
        }
        if previous_slot.is_some_and(|slot| descriptor.slot <= slot) {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "ABI descriptors must have strictly ascending unique slots, got slot {} after {:?}",
                    descriptor.slot, previous_slot
                ),
            });
        }
        previous_slot = Some(descriptor.slot);

        match &descriptor.kind {
            AbiParamKind::Storage(_) => {
                if descriptor.name.is_some() {
                    return Err(Error::ProgramAbiMismatch {
                        reason: format!("storage PARAM slot {} must not have a scalar name", descriptor.slot),
                    });
                }
                let supported_dtype = match &descriptor.dtype {
                    DType::Scalar(dtype) => !matches!(
                        dtype,
                        ScalarDType::Void | ScalarDType::Index | ScalarDType::WeakInt | ScalarDType::WeakFloat
                    ),
                    DType::Vector { scalar, count } => {
                        *count > 0
                            && !matches!(
                                scalar,
                                ScalarDType::Void | ScalarDType::Index | ScalarDType::WeakInt | ScalarDType::WeakFloat
                            )
                    }
                    DType::Ptr { .. } | DType::Image { .. } => false,
                };
                if !supported_dtype {
                    return Err(Error::ProgramAbiMismatch {
                        reason: format!(
                            "storage PARAM slot {} has unsupported element dtype {:?}",
                            descriptor.slot, descriptor.dtype
                        ),
                    });
                }
                buf_count += 1;
            }
            AbiParamKind::Scalar => {
                if descriptor.dtype != DType::Int32 {
                    return Err(Error::ProgramAbiMismatch {
                        reason: format!(
                            "scalar PARAM slot {} has non-canonical final ABI dtype {:?}",
                            descriptor.slot, descriptor.dtype
                        ),
                    });
                }
                let name = descriptor.name.as_deref().filter(|name| !name.is_empty()).ok_or_else(|| {
                    Error::ProgramAbiMismatch {
                        reason: format!("scalar PARAM in slot {} has no name", descriptor.slot),
                    }
                })?;
                if !unique_names.insert(name) {
                    return Err(Error::ProgramAbiMismatch { reason: format!("duplicate scalar PARAM name {name:?}") });
                }
                var_names.push(name.to_string());
            }
        }
    }

    if buf_count != expected_buf_count || var_names != expected_var_names {
        return Err(Error::ProgramAbiMismatch {
            reason: format!(
                "ABI descriptors project to {buf_count} buffers/vars {var_names:?}, expected {expected_buf_count}/{expected_var_names:?}"
            ),
        });
    }
    Ok(())
}

impl CompiledSpec {
    /// Reconstruct a compiled candidate produced by the clean BEAM helper.
    /// The parent validates every independently checkable SOURCE/BINARY field;
    /// the linear digest remains the worker's opaque identity so the parent
    /// does not rebuild the candidate LINEAR graph. `launch_placeholder` is
    /// shared by private benchmark artifacts; dispatch uses the worker-returned
    /// concrete launch dimensions rather than these symbolic fields.
    #[allow(clippy::too_many_arguments)]
    pub fn from_beam_worker(
        name: String,
        source: String,
        bytes: Vec<u8>,
        ast: Arc<UOp>,
        abi: Vec<AbiParamDescriptor>,
        launch_placeholder: [Arc<UOp>; 3],
        identity: BinaryStageIdentity,
        target: &DeviceSpec,
        compiler_key: &str,
    ) -> Result<Self> {
        let buf_count = abi.iter().filter(|arg| arg.is_storage()).count();
        let var_names = abi
            .iter()
            .filter(|arg| !arg.is_storage())
            .map(|arg| arg.name.clone().unwrap_or_default())
            .collect::<Vec<_>>();
        validate_abi_descriptors(&abi, buf_count, &var_names)?;
        let spec = Self {
            name,
            src: Some(source),
            bytes,
            ast,
            var_names,
            global_size: launch_placeholder,
            local_size: None,
            buf_count,
            abi,
            stage_identity: Some(identity),
        };
        spec.validate_stage_identity(target, compiler_key)?;
        Ok(spec)
    }

    /// Create a new CompiledSpec for JIT backends (source-based).
    pub fn from_source(name: String, src: String, ast: Arc<UOp>, abi: Vec<AbiParamDescriptor>) -> Result<Self> {
        let buf_count = abi.iter().filter(|arg| arg.is_storage()).count();
        let var_names = abi
            .iter()
            .filter(|arg| !arg.is_storage())
            .map(|arg| arg.name.clone().unwrap_or_default())
            .collect::<Vec<_>>();
        validate_abi_descriptors(&abi, buf_count, &var_names)?;
        Ok(Self {
            name,
            src: Some(src),
            bytes: Vec::new(),
            ast,
            var_names,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
            buf_count,
            abi,
            stage_identity: None,
        })
    }

    /// Create a new CompiledSpec for AOT backends (bytecode-based).
    pub fn from_bytes(name: String, bytes: Vec<u8>, ast: Arc<UOp>, abi: Vec<AbiParamDescriptor>) -> Result<Self> {
        let buf_count = abi.iter().filter(|arg| arg.is_storage()).count();
        let var_names = abi
            .iter()
            .filter(|arg| !arg.is_storage())
            .map(|arg| arg.name.clone().unwrap_or_default())
            .collect::<Vec<_>>();
        validate_abi_descriptors(&abi, buf_count, &var_names)?;
        Ok(Self {
            name,
            src: None,
            bytes,
            ast,
            var_names,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
            buf_count,
            abi,
            stage_identity: None,
        })
    }

    /// Create a new CompiledSpec with work sizes for JIT backends.
    pub fn from_source_with_sizes(
        name: String,
        src: String,
        ast: Arc<UOp>,
        global_size: [usize; 3],
        local_size: Option<[usize; 3]>,
        abi: Vec<AbiParamDescriptor>,
    ) -> Result<Self> {
        let buf_count = abi.iter().filter(|arg| arg.is_storage()).count();
        let var_names = abi
            .iter()
            .filter(|arg| !arg.is_storage())
            .map(|arg| arg.name.clone().unwrap_or_default())
            .collect::<Vec<_>>();
        validate_abi_descriptors(&abi, buf_count, &var_names)?;
        Ok(Self {
            name,
            src: Some(src),
            bytes: Vec::new(),
            ast,
            var_names,
            global_size: concrete_launch_size(global_size),
            local_size: local_size.map(concrete_launch_size),
            buf_count,
            abi,
            stage_identity: None,
        })
    }

    /// Bind compiler output to the staged PROGRAM identity that produced it.
    pub fn bind_program_stage(
        &mut self,
        target: &DeviceSpec,
        compiler_key: &str,
        identity: BinaryStageIdentity,
    ) -> Result<()> {
        self.stage_identity = Some(identity);
        self.validate_stage_identity(target, compiler_key)
    }

    /// Validate all semantic stage fields required before executable loading.
    /// The LINEAR digest is the minted one (see `minted_source_stage_identity`);
    /// every other field is re-derived from this specification.
    pub fn validate_stage_identity(&self, target: &DeviceSpec, compiler_key: &str) -> Result<()> {
        let identity = self.stage_identity.as_ref().ok_or_else(|| Error::ProgramStageMismatch {
            stage: "BINARY",
            reason: "compiled specification has no semantic stage identity".into(),
        })?;
        let source = self.src.as_deref().ok_or_else(|| Error::ProgramStageMismatch {
            stage: "SOURCE",
            reason: "compiled specification does not retain its source payload".into(),
        })?;
        let expected_source = SourceStageIdentity {
            version: SOURCE_STAGE_IDENTITY_VERSION,
            abi: stage_abi(&self.abi),
            target: target.clone(),
            entry_name: self.name.clone(),
            linear_sha256: identity.source.linear_sha256,
            source_sha256: sha256(source.as_bytes()),
        };
        if identity.source != expected_source {
            return Err(Error::ProgramStageMismatch {
                stage: "SOURCE",
                reason: format!("expected {expected_source:?}, got {:?}", identity.source),
            });
        }
        let expected_binary = binary_stage_identity(expected_source, compiler_key, &self.bytes);
        if identity != &expected_binary {
            return Err(Error::ProgramStageMismatch {
                stage: "BINARY",
                reason: format!("expected {expected_binary:?}, got {identity:?}"),
            });
        }
        Ok(())
    }
}

/// Concrete launch dimensions passed to backend runtimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConcreteLaunchDims {
    pub global_size: [usize; 3],
    pub local_size: Option<[usize; 3]>,
}

fn default_launch_size() -> [Arc<UOp>; 3] {
    [UOp::index_const(1), UOp::index_const(1), UOp::index_const(1)]
}

fn concrete_launch_size(size: [usize; 3]) -> [Arc<UOp>; 3] {
    [UOp::index_const(size[0] as i64), UOp::index_const(size[1] as i64), UOp::index_const(size[2] as i64)]
}

fn const_value_to_i64(value: ConstValue) -> Result<i64> {
    match value {
        ConstValue::Invalid => Err(Error::Runtime { message: "Invalid reached launch-size evaluation".to_string() }),
        ConstValue::Int(v) => Ok(v),
        ConstValue::UInt(v) => i64::try_from(v)
            .map_err(|_| Error::Runtime { message: format!("launch-size constant {v} does not fit i64") }),
        ConstValue::Bool(v) => Ok(i64::from(v)),
        ConstValue::Float(v) => {
            Err(Error::Runtime { message: format!("launch-size expression must be integer, got float constant {v}") })
        }
    }
}

fn validate_var_bound(name: &str, value: i64, min_val: i64, max_val: i64) -> Result<()> {
    snafu::ensure!(
        value >= min_val && value <= max_val,
        VarOutOfBoundsSnafu { name, value, min: min_val, max: max_val }
    );
    Ok(())
}

fn checked_launch_binary(op: BinaryOp, lhs: i64, rhs: i64) -> Result<i64> {
    // EXHAUSTIVE match — no `_` catch-all. When a new `BinaryOp` variant is
    // added to `svod_ir::types::BinaryOp`, the compiler will fail this match
    // and force an explicit decision, instead of silently producing wrong
    // launch dims at runtime. A symbolic evaluator codegen'd from the renderer
    // pipeline would inherit every operator automatically; until svod unifies
    // the two evaluators, this exhaustive match is the belt-and-suspenders
    // that approximates the same guarantee.
    let value: Option<i64> = match op {
        // Integer arithmetic — checked for overflow.
        BinaryOp::Add => lhs.checked_add(rhs),
        BinaryOp::Sub => lhs.checked_sub(rhs),
        BinaryOp::Mul => lhs.checked_mul(rhs),
        BinaryOp::FloorDiv => lhs
            .checked_div(rhs)
            .and_then(|q| lhs.checked_rem(rhs).map(|r| if r != 0 && (lhs < 0) != (rhs < 0) { q - 1 } else { q })),
        BinaryOp::FloorMod => {
            lhs.checked_rem(rhs).and_then(|r| if r != 0 && (r < 0) != (rhs < 0) { r.checked_add(rhs) } else { Some(r) })
        }
        BinaryOp::CDiv => (rhs != 0).then(|| lhs.checked_div(rhs)).flatten(),
        BinaryOp::CMod => (rhs != 0).then(|| lhs.checked_rem(rhs)).flatten(),
        BinaryOp::Max => Some(lhs.max(rhs)),
        // Integer power: only support non-negative exponents that fit in u32.
        BinaryOp::Pow => u32::try_from(rhs).ok().and_then(|e| lhs.checked_pow(e)),
        // Bitwise / shift. Negative shifts and shifts ≥ 64 are rejected.
        BinaryOp::Shl => u32::try_from(rhs).ok().filter(|&r| r < 64).and_then(|r| lhs.checked_shl(r)),
        BinaryOp::Shr => u32::try_from(rhs).ok().filter(|&r| r < 64).and_then(|r| lhs.checked_shr(r)),
        BinaryOp::And => Some(lhs & rhs),
        BinaryOp::Or => Some(lhs | rhs),
        BinaryOp::Xor => Some(lhs ^ rhs),
        // Comparisons — fold to 0/1 (consistent with IR's symbolic rewrite
        // pipeline where Bool is i1 and may participate in arithmetic via
        // CAST).
        BinaryOp::Lt => Some(i64::from(lhs < rhs)),
        BinaryOp::Le => Some(i64::from(lhs <= rhs)),
        BinaryOp::Eq => Some(i64::from(lhs == rhs)),
        BinaryOp::Ne => Some(i64::from(lhs != rhs)),
        BinaryOp::Gt => Some(i64::from(lhs > rhs)),
        BinaryOp::Ge => Some(i64::from(lhs >= rhs)),
        // Float-only / nonsense for launch dims.
        BinaryOp::Fdiv => {
            return Err(Error::Runtime {
                message: "Fdiv (float division) in launch-size expression — launch dims must be integer".into(),
            });
        }
        BinaryOp::Threefry => {
            return Err(Error::Runtime {
                message: "Threefry (PRNG) in launch-size expression — this is almost certainly a scheduler bug".into(),
            });
        }
    };

    value.ok_or_else(|| Error::Runtime { message: format!("invalid launch-size arithmetic: {lhs} {op:?} {rhs}") })
}

/// Evaluate a ternary op in a launch-size expression. `MulAcc(a, b, c) = a*b+c`
/// (overflow-checked); `Where(cond, t, f)` selects on a nonzero predicate.
/// EXHAUSTIVE match so a new `TernaryOp` variant fails the compile, not silently
/// at runtime.
fn checked_launch_ternary(op: TernaryOp, a: i64, b: i64, c: i64) -> Result<i64> {
    let value = match op {
        TernaryOp::MulAcc => a.checked_mul(b).and_then(|ab| ab.checked_add(c)),
        TernaryOp::Where => Some(if a != 0 { b } else { c }),
    };
    value.ok_or_else(|| Error::Runtime { message: format!("invalid launch-size ternary: {op:?}({a}, {b}, {c})") })
}

fn eval_launch_expr(expr: &Arc<UOp>, vars: &HashMap<&str, i64>) -> Result<i64> {
    match expr.op() {
        Op::Const(value) => const_value_to_i64(value.0),
        Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none() => {
            let name = arg
                .name
                .as_deref()
                .ok_or_else(|| Error::Runtime { message: "scalar launch-size PARAM has no name".to_string() })?;
            let value = vars.get(name).copied().ok_or_else(|| Error::Runtime {
                message: format!("missing runtime value for launch-size variable {name}"),
            })?;
            if let Some((min, max)) = &arg.vmin_vmax
                && let (Some(min), Some(max)) = (min.0.try_int(), max.0.try_int())
            {
                validate_var_bound(name, value, min, max)?;
            }
            Ok(value)
        }
        Op::Bind(ops::Bind { var, value }) => {
            let bound = eval_launch_expr(value, vars)?;
            if let Op::Param(ops::Param { arg, .. }) = var.op()
                && let (Some(name), Some((min, max))) = (&arg.name, &arg.vmin_vmax)
                && let (Some(min), Some(max)) = (min.0.try_int(), max.0.try_int())
            {
                validate_var_bound(name, bound, min, max)?;
            }
            Ok(bound)
        }
        Op::Binary(op, lhs, rhs) => {
            checked_launch_binary(*op, eval_launch_expr(lhs, vars)?, eval_launch_expr(rhs, vars)?)
        }
        // The symbolic simplifier fuses `a*b + c` (e.g. `16*ts − 1` from a
        // reshaped symbolic sequence axis) into a single MulAcc; `Where` can
        // appear when a launch dim is gated on a symbolic predicate. Evaluate
        // both rather than rejecting — the alternative is the scheduler
        // silently never fusing, which it does.
        Op::Ternary(op, a, b, c) => checked_launch_ternary(
            *op,
            eval_launch_expr(a, vars)?,
            eval_launch_expr(b, vars)?,
            eval_launch_expr(c, vars)?,
        ),
        Op::Unary(op, src) => checked_launch_unary(*op, eval_launch_expr(src, vars)?),
        Op::Cast(ops::Cast { src, .. })
        | Op::BitCast(ops::BitCast { src, .. })
        | Op::After(ops::After { passthrough: src, .. }) => eval_launch_expr(src, vars),
        other => Err(Error::Runtime { message: format!("unsupported launch-size expression op: {other:?}") }),
    }
}

fn checked_launch_unary(op: UnaryOp, src: i64) -> Result<i64> {
    // EXHAUSTIVE match (no `_` catch-all) so a new `UnaryOp` variant fails the
    // build instead of silently corrupting launch dims. See the analogous
    // comment on `checked_launch_binary` for the longer rationale.
    let value: Option<i64> = match op {
        UnaryOp::Neg => src.checked_neg(),
        UnaryOp::Abs => src.checked_abs(),
        UnaryOp::Not => Some(!src),
        UnaryOp::Sign => Some(src.signum()),
        UnaryOp::Square => src.checked_mul(src),
        // For integer launch dims `trunc/floor/ceil/round` are identity since
        // the input is already an integer. A symbolic evaluator would collapse
        // these via the rewrite engine; the explicit arms here are the same
        // outcome.
        UnaryOp::Trunc | UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round => Some(src),
        // Float-only — these have no meaning on integer launch dims and would
        // never reach here from a correct schedule.
        UnaryOp::Sqrt
        | UnaryOp::Rsqrt
        | UnaryOp::Exp
        | UnaryOp::Exp2
        | UnaryOp::Log
        | UnaryOp::Log2
        | UnaryOp::Sin
        | UnaryOp::Cos
        | UnaryOp::Tan
        | UnaryOp::Reciprocal
        | UnaryOp::Erf => {
            return Err(Error::Runtime {
                message: format!("float-only unary op {op:?} in launch-size expression — schedule bug"),
            });
        }
    };

    value.ok_or_else(|| Error::Runtime { message: format!("invalid launch-size unary arithmetic: {op:?} {src}") })
}

fn eval_launch_size(size: &[Arc<UOp>; 3], vars: &HashMap<&str, i64>) -> Result<[usize; 3]> {
    let mut out = [1usize; 3];
    for (idx, expr) in size.iter().enumerate() {
        let value = eval_launch_expr(expr, vars)?;
        if value <= 0 {
            return Err(Error::Runtime {
                message: format!("launch dimension {idx} evaluated to non-positive value {value}"),
            });
        }
        out[idx] = usize::try_from(value).map_err(|_| Error::Runtime {
            message: format!("launch dimension {idx} value {value} does not fit usize"),
        })?;
    }
    Ok(out)
}

/// A compiler that transforms source code into a compiled specification.
///
/// This trait abstracts over different compilation backends:
/// - LLVM: IR validation (JIT compiles at runtime)
/// - CUDA: CUDA C -> PTX/CUBIN
/// - Metal: Metal Shading Language -> metallib
/// - WebGPU: WGSL -> SPIR-V
pub trait Compiler: Send + Sync {
    /// Compile a program specification into executable form.
    ///
    /// # Arguments
    ///
    /// * `spec` - The program specification containing source code and metadata
    ///
    /// # Returns
    ///
    /// A CompiledSpec containing:
    /// - For JIT backends (LLVM): source code in `src` field, empty `bytes`
    /// - For AOT backends (CUDA/Metal): compiled bytes in `bytes` field, no `src`
    ///
    /// # Examples
    ///
    /// JIT backend (LLVM):
    /// ```ignore
    /// let compiled = compiler.compile(&spec)?;
    /// assert!(compiled.src.is_some());
    /// assert!(compiled.bytes.is_empty());
    /// ```
    ///
    /// AOT backend (CUDA):
    /// ```ignore
    /// let compiled = compiler.compile(&spec)?;
    /// assert!(compiled.src.is_none());
    /// assert!(!compiled.bytes.is_empty());
    /// ```
    fn compile(&self, spec: &ProgramSpec) -> Result<CompiledSpec>;

    /// Cache key identifying the exact compiler configuration.
    ///
    /// This includes the backend and all target/toolchain/ABI settings that can
    /// affect bytes, not merely a family name such as `clang`.
    fn cache_key(&self) -> &str;
}

/// A renderer that transforms UOp graphs into source code.
///
/// This trait abstracts over different code generation backends:
/// - LLVM IR generator
/// - CUDA C generator
/// - Metal Shading Language generator
/// - WGSL generator
pub trait Renderer: Send + Sync {
    /// Render a UOp graph into source code.
    ///
    /// # Arguments
    ///
    /// * `ast` - The kernel AST (UOp graph rooted at a CALL body such as SINK/PROGRAM)
    /// * `name` - Optional kernel name for debugging (e.g., "r_g16l16R32u4").
    ///   Falls back to "kernel" if None.
    ///
    /// # Returns
    ///
    /// A ProgramSpec containing:
    /// - Generated source code
    /// - Entry point name
    /// - Variable list
    /// - Work sizes (for GPU backends)
    fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> Result<ProgramSpec>;

    /// Get the device spec for this renderer.
    ///
    /// This is used for cache key construction and device selection.
    fn device(&self) -> &DeviceSpec;

    /// The GPU architecture this renderer targets, if any. Arch is a hardware
    /// property of the opened device (not the `DeviceSpec`), surfaced here so
    /// the scheduler can pick the matching optimizer profile (wave size, matrix
    /// cores, …). CPU and backends without an arch distinction return `None`.
    fn gpu_arch(&self) -> Option<svod_dtype::GpuArch> {
        None
    }

    /// Operations this concrete code renderer accepts without decomposition.
    fn supported_ops(&self) -> svod_ir::RendererOps;

    /// Returns decomposition patterns for operations this backend doesn't support.
    ///
    /// This is used by the realization pass to decompose complex operations
    /// into simpler primitives before rendering.
    ///
    /// # Default Implementation
    ///
    /// Returns `None`, meaning no decomposition is needed (backend supports all ops).
    /// Backends that don't support certain operations (e.g., transcendentals)
    /// should override this to return appropriate patterns.
    fn decompositor(&self) -> Option<svod_ir::pattern::TypedPatternMatcher<()>> {
        None
    }

    /// Renderer-local final rewrites, separate from unsupported-op decomposition.
    fn extra_matcher(&self) -> Option<svod_ir::pattern::TypedPatternMatcher<()>> {
        None
    }

    /// Optional bottom-up rewrite before target instruction selection.
    fn pre_isel_matcher(&self) -> Option<svod_ir::pattern::TypedPatternMatcher<crate::isa::PreIselContext>> {
        None
    }

    /// Optional bottom-up target instruction selector. Source renderers leave
    /// this absent; providing it marks the renderer as an ISA target.
    fn isel_matcher(&self) -> Option<svod_ir::pattern::TypedPatternMatcher<crate::isa::IselContext>> {
        None
    }
}

/// A factory function that creates executable Programs from a compiled specification.
///
/// This is a function pointer that wraps the backend-specific loader:
/// - LLVM: Extract source from CompiledSpec and JIT compile
/// - CUDA: Extract bytes from CompiledSpec and call cuModuleLoadData + cuModuleGetFunction
/// - Metal: Extract bytes from CompiledSpec and call newLibraryWithData + newFunctionWithName
/// - WebGPU: Extract bytes from CompiledSpec and call createShaderModule
///
/// The CompiledSpec contains either source (for JIT) or bytes (for AOT),
/// allowing each backend to access what it needs.
pub type RuntimeFactory = Arc<dyn Fn(&CompiledSpec) -> Result<Box<dyn Program>> + Send + Sync>;

/// Builds a replayable graph from a captured kernel chain. Returns `Ok(None)`
/// when this backend can't graph the chain (then callers fall back to per-call
/// dispatch). A graphing backend pre-builds its dispatch packets; CPU has no
/// factory.
pub type GraphFactory = Arc<dyn Fn(&[GraphKernel<'_>]) -> Result<Option<Box<dyn Graph>>> + Send + Sync>;

/// A (Renderer, Compiler) pair for a specific backend.
///
/// Devices can have multiple compiler pairs (e.g., different optimization levels).
pub type CompilerPair = (Arc<dyn Renderer>, Arc<dyn Compiler>);

/// A device that owns renderer, compiler, runtime, and allocator.
///
/// A Device is a complete compilation + execution unit for a specific backend.
///
/// # Example
///
/// A `CompiledSpec` reaches [`Device::runtime`] only with a bound semantic
/// stage identity ([`CompiledSpec::bind_program_stage`]); driving
/// [`Renderer`]/[`Compiler`] by hand skips that binding and every runtime
/// factory then rejects the spec with [`Error::ProgramStageMismatch`]. The
/// staged pipeline in `svod_codegen::program_pipeline` is the supported path
/// (the example lives there because `svod-codegen` sits above this crate):
///
/// ```ignore
/// let device = create_cpu_device()?;
/// let program = program_pipeline::program_from_sink(sink, device.device.clone())?;
/// let (program, _spec) = program_pipeline::do_render(&program, device.renderer.as_ref())?;
/// let (_program, compiled) = program_pipeline::do_compile(&program, device.compiler.as_ref())?;
/// let program = (device.runtime)(&compiled)?;
/// unsafe { program.execute(&buffers, &vals, None, None, /*wait=*/ true)?; }
/// ```
pub struct Device {
    /// Device specification
    pub device: DeviceSpec,

    /// Memory allocator for this device
    pub allocator: Arc<dyn Allocator>,

    /// Available (renderer, compiler) pairs for this device
    ///
    /// Most devices have one pair, but some may have multiple
    /// (e.g., different optimization levels or compilation modes).
    pub compilers: Vec<CompilerPair>,

    /// Primary renderer for this device
    ///
    /// This is typically `compilers[0].0`, stored separately for convenience.
    pub renderer: Arc<dyn Renderer>,

    /// Primary compiler for this device
    ///
    /// This is typically `compilers[0].1`, stored separately for convenience.
    pub compiler: Arc<dyn Compiler>,

    /// Runtime factory for creating executable programs
    ///
    /// Takes (entry_point, compiled_bytes) and returns a Program.
    pub runtime: RuntimeFactory,

    /// Optional graph factory for capture/replay. `None` means per-call
    /// dispatch only (CPU); a graphing backend installs one to capture/replay
    /// kernel chains.
    pub graph: Option<GraphFactory>,
}

impl Device {
    /// Create a new device with a single compiler pair.
    ///
    /// This is a convenience constructor for the common case where
    /// a device has only one renderer/compiler combination.
    pub fn new(
        device: DeviceSpec,
        allocator: Arc<dyn Allocator>,
        renderer: Arc<dyn Renderer>,
        compiler: Arc<dyn Compiler>,
        runtime: RuntimeFactory,
    ) -> Self {
        let compilers = vec![(renderer.clone(), compiler.clone())];
        let runtime_device = device.clone();
        let runtime_compiler_key = compiler.cache_key().to_string();
        let raw_runtime = runtime;
        let runtime: RuntimeFactory = Arc::new(move |spec| {
            spec.validate_stage_identity(&runtime_device, &runtime_compiler_key)?;
            raw_runtime(spec)
        });
        Self { device, allocator, compilers, renderer, compiler, runtime, graph: None }
    }

    /// Install a graph factory (capture/replay). Builder-style for backends
    /// that can pre-build dispatch packets.
    pub fn with_graph(mut self, factory: GraphFactory) -> Self {
        self.graph = Some(factory);
        self
    }

    /// Get the base device key (strips device ID).
    ///
    /// Used for compiled byte cache sharing across device instances.
    /// Examples:
    /// - DeviceSpec::Cpu -> "CPU"
    /// - DeviceSpec::Cuda { device_id: 0 } -> "CUDA"
    /// - DeviceSpec::Cuda { device_id: 1 } -> "CUDA"
    /// - DeviceSpec::Metal { device_id: 0 } -> "Metal"
    ///
    /// This allows compiled CUDA kernels to be reused across CUDA:0 and CUDA:1.
    pub fn base_device_key(&self) -> &'static str {
        self.device.base_type()
    }
}

/// Program specification containing source code and metadata.
///
/// This is returned by Renderer::render() and consumed by Compiler::compile().
/// It bridges the gap between UOp graphs and compiled executables.
///
/// # Buffer metadata
///
/// - `globals`: Buffer indices from PARAM ops
/// - `outs`: Output buffer indices (written by STORE ops)
/// - `ins`: Input buffer indices (read by LOAD ops)
#[derive(Debug, Clone)]
pub struct ProgramSpec {
    /// Kernel name (for debugging/profiling)
    pub name: String,

    /// Generated source code (LLVM IR, CUDA C, Metal, WGSL, etc.)
    pub src: String,

    /// Device specification
    pub device: DeviceSpec,

    /// Original AST (for cache key construction via hash consing)
    pub ast: Arc<UOp>,

    /// Symbolic global work size.
    pub global_size: [Arc<UOp>; 3],

    /// Symbolic local work size. None means direct global-id execution.
    pub local_size: Option<[Arc<UOp>; 3]>,

    /// Variable list (for symbolic shapes/strides)
    pub vars: Vec<Variable>,

    /// Variable names in order for populating vars array at runtime.
    /// Includes runtime variables such as core_id.
    pub var_names: Vec<String>,

    /// Global buffer indices (from PARAM slot values).
    pub globals: Vec<usize>,

    /// Output buffer indices (written by STORE ops).
    pub outs: Vec<usize>,

    /// Input buffer indices (read by LOAD ops, excluding outputs).
    pub ins: Vec<usize>,

    /// Number of buffer arguments (for CIF construction at compile time).
    pub buf_count: usize,

    /// Complete kernel argument ABI in source-signature order.
    pub abi: Vec<AbiParamDescriptor>,
}

impl ProgramSpec {
    fn same_uops(a: &[Arc<UOp>], b: &[Arc<UOp>]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.content_hash == b.content_hash)
    }

    fn same_param_semantics(actual: &Arc<UOp>, expected: &Arc<UOp>) -> bool {
        match (actual.op(), expected.op()) {
            (
                Op::Param(ops::Param { shape: actual_shape, arg: actual_arg }),
                Op::Param(ops::Param { shape: expected_shape, arg: expected_arg }),
            ) => {
                actual.dtype() == expected.dtype()
                    && actual_arg == expected_arg
                    && actual_shape.content_hash == expected_shape.content_hash
            }
            _ => false,
        }
    }

    /// Validate ProgramInfo against the executable SINK without relying on UOp
    /// allocation identity. The returned descriptors remain the runtime ABI
    /// projection; PARAM semantics are checked separately before projection.
    pub fn validate_program_param_abi(sink: &Arc<UOp>, info: &svod_ir::ProgramInfo) -> Result<Vec<AbiParamDescriptor>> {
        let mut occupied: HashMap<usize, String> = HashMap::new();
        let executable = sink.toposort_call_aware(false);
        let executable_ids = executable.iter().map(|node| node.id).collect::<std::collections::HashSet<_>>();

        for node in &executable {
            if let Op::Special(ops::Special { name, .. }) = node.op()
                && !matches!(name.chars().last().and_then(|axis| axis.to_digit(10)), Some(0..=2))
            {
                return Err(Error::ProgramAbiMismatch { reason: format!("invalid SPECIAL axis name {name:?}") });
            }
            let body = match node.op() {
                Op::Call(ops::Call { body, .. }) | Op::Function(ops::Function { body, .. }) => body,
                _ => continue,
            };
            for formal in body.toposort_call_aware(true) {
                if matches!(formal.op(), Op::Param(..)) && executable_ids.contains(&formal.id) {
                    return Err(Error::LeakedOpaqueProgramParam {
                        param: format!("UOp {} {:?}", formal.id, formal.op()),
                    });
                }
            }
        }

        let mut abi = Vec::new();
        for node in executable {
            let Op::Param(ops::Param { arg, .. }) = node.op() else { continue };
            let descriptor = AbiParamDescriptor::from_param(&node)?;
            let class = format!("{:?} {:?}", descriptor.kind, descriptor.name);
            if let Some(first) = occupied.insert(arg.slot, class.clone()) {
                return Err(Error::DuplicateProgramParamSlot { slot: arg.slot, first, second: class });
            }
            abi.push(descriptor);
        }
        abi.sort_by_key(|param| param.slot);
        let globals = abi.iter().filter(|param| param.is_storage()).map(|param| param.slot).collect::<Vec<_>>();
        let info_vars = info
            .vars
            .iter()
            .map(|var| {
                let descriptor = AbiParamDescriptor::from_param(var)?;
                if descriptor.is_storage() {
                    return Err(Error::ProgramAbiMismatch {
                        reason: format!("ProgramInfo.vars contains storage PARAM in slot {}", descriptor.slot),
                    });
                }
                Ok(descriptor)
            })
            .collect::<Result<Vec<_>>>()?;
        let vars = abi.iter().filter(|param| !param.is_storage()).cloned().collect::<Vec<_>>();
        let expected = svod_ir::ProgramInfo::from_sink(sink, info.target.clone());
        if info.vars.len() != expected.vars.len()
            || !info
                .vars
                .iter()
                .zip(&expected.vars)
                .all(|(actual, expected)| Self::same_param_semantics(actual, expected))
        {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "ProgramInfo.vars semantic mismatch: sink requires {:?}; ProgramInfo has {:?} (projected ABI {info_vars:?})",
                    expected.vars, info.vars
                ),
            });
        }
        if globals != expected.globals || info.globals != expected.globals {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "ProgramInfo.globals mismatch: sink requires {:?} with ABI {vars:?}/{abi:?}; ProgramInfo has {:?}",
                    expected.globals, info.globals
                ),
            });
        }
        if info.outs != expected.outs {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "ProgramInfo.outs mismatch: sink requires {:?}; ProgramInfo has {:?}",
                    expected.outs, info.outs
                ),
            });
        }
        if info.ins != expected.ins {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "ProgramInfo.ins mismatch: sink requires {:?}; ProgramInfo has {:?}",
                    expected.ins, info.ins
                ),
            });
        }
        if !Self::same_uops(&info.global_size, &expected.global_size) {
            return Err(Error::ProgramAbiMismatch {
                reason: "ProgramInfo.global_size does not match canonical sink launch dimensions".into(),
            });
        }
        if match (&info.local_size, &expected.local_size) {
            (Some(actual), Some(expected)) => !Self::same_uops(actual, expected),
            (None, None) => false,
            _ => true,
        } {
            return Err(Error::ProgramAbiMismatch {
                reason: "ProgramInfo.local_size does not match canonical sink launch dimensions".into(),
            });
        }
        if info.outs.iter().chain(&info.ins).any(|slot| !info.globals.contains(slot)) {
            return Err(Error::ProgramAbiMismatch { reason: "ProgramInfo ins/outs contains a non-global slot".into() });
        }
        let var_names = vars.iter().map(|param| param.name.clone().unwrap_or_default()).collect::<Vec<_>>();
        validate_abi_descriptors(&abi, globals.len(), &var_names)?;
        Ok(abi)
    }

    /// Create a new program specification.
    pub fn new(name: String, src: String, device: DeviceSpec, ast: Arc<UOp>) -> Self {
        Self {
            name,
            src,
            device,
            ast,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
            vars: Vec::new(),
            var_names: Vec::new(),
            globals: Vec::new(),
            outs: Vec::new(),
            ins: Vec::new(),
            buf_count: 0,
            abi: Vec::new(),
        }
    }

    /// Add a variable to the program.
    pub fn add_var(&mut self, var: Variable) {
        self.vars.push(var);
    }

    /// Set work sizes for GPU execution.
    pub fn set_work_sizes(&mut self, global: [usize; 3], local: [usize; 3]) {
        self.global_size = concrete_launch_size(global);
        self.local_size = Some(concrete_launch_size(local));
    }

    /// Set symbolic work sizes for replay with runtime variables.
    pub fn set_launch_dims(&mut self, global: [Arc<UOp>; 3], local: Option<[Arc<UOp>; 3]>) {
        self.global_size = global;
        self.local_size = local;
    }

    /// Evaluate symbolic launch dimensions using runtime variable values.
    pub fn launch_dims(&self, var_vals: &HashMap<&str, i64>) -> Result<ConcreteLaunchDims> {
        Self::resolve_launch_dims(&self.global_size, self.local_size.as_ref(), var_vals)
    }

    /// Evaluate launch dimensions stored outside a full ProgramSpec.
    pub fn resolve_launch_dims(
        global_size: &[Arc<UOp>; 3],
        local_size: Option<&[Arc<UOp>; 3]>,
        var_vals: &HashMap<&str, i64>,
    ) -> Result<ConcreteLaunchDims> {
        Ok(ConcreteLaunchDims {
            global_size: eval_launch_size(global_size, var_vals)?,
            local_size: local_size.map(|local| eval_launch_size(local, var_vals)).transpose()?,
        })
    }

    /// Set variable names for populating vars array at runtime.
    pub fn set_var_names(&mut self, var_names: Vec<String>) {
        self.var_names = var_names;
    }

    /// Set buffer metadata (globals, outs, ins).
    pub fn set_buffer_metadata(&mut self, globals: Vec<usize>, outs: Vec<usize>, ins: Vec<usize>) {
        self.globals = globals;
        self.outs = outs;
        self.ins = ins;
    }

    /// Build a ProgramSpec from a PROGRAM UOp state.
    ///
    /// Validates PROGRAM stage shape and derives metadata from PROGRAM itself.
    pub fn from_uop(program: &Arc<UOp>) -> Result<Self> {
        let Op::Program(ops::Program { sink, info, linear, source, binary }) = program.op() else {
            return WrongStageSnafu { expected: "PROGRAM", got: format!("{:?}", program.op()) }.fail();
        };

        snafu::ensure!(
            matches!(sink.op(), Op::Sink(..)),
            WrongStageSnafu { expected: "PROGRAM sink stage SINK", got: format!("{:?}", sink.op()) }
        );

        let linear = linear
            .as_ref()
            .context(WrongStageSnafu { expected: "PROGRAM LINEAR stage", got: "missing".to_string() })?;
        snafu::ensure!(
            matches!(linear.op(), Op::Linear(..)),
            WrongStageSnafu { expected: "PROGRAM linear stage LINEAR", got: format!("{:?}", linear.op()) }
        );

        let source = source
            .as_ref()
            .context(WrongStageSnafu { expected: "PROGRAM SOURCE stage", got: "missing".to_string() })?;
        let source_code = match source.op() {
            Op::Source(ops::Source { code, .. }) => code.clone(),
            other => {
                return WrongStageSnafu { expected: "PROGRAM source stage SOURCE", got: format!("{other:?}") }.fail();
            }
        };

        let abi = Self::validate_program_param_abi(sink, info)?;
        let expected_source = minted_source_stage_identity(info, &abi, source)?;

        if let Some(binary) = binary {
            let Op::ProgramBinary(ops::ProgramBinary { bytes, identity }) = binary.op() else {
                return WrongStageSnafu {
                    expected: "PROGRAM binary stage ProgramBinary",
                    got: format!("{:?}", binary.op()),
                }
                .fail();
            };
            let compiler_key = identity.as_ref().map(|identity| identity.compiler_key.as_str()).ok_or_else(|| {
                Error::ProgramStageMismatch { stage: "BINARY", reason: "stage has no semantic identity".into() }
            })?;
            if compiler_key.is_empty() {
                return Err(Error::ProgramStageMismatch {
                    stage: "BINARY",
                    reason: "compiler cache key is empty".into(),
                });
            }
            let expected_binary = binary_stage_identity(expected_source.clone(), compiler_key, bytes);
            validate_binary_stage(binary, &expected_binary)?;
        }

        let mut spec = Self::new(info.function_name(), source_code, info.target.clone(), sink.clone());
        spec.vars = info
            .vars
            .iter()
            .filter_map(|u| match u.op() {
                Op::Param(ops::Param { arg, .. }) => Some(Variable::new(
                    arg.name.clone().unwrap_or_else(|| format!("p{}", arg.slot)),
                    arg.vmin_vmax
                        .as_ref()
                        .and_then(|(v, _)| match v.0 {
                            svod_ir::ConstValue::Int(value) => Some(value),
                            _ => None,
                        })
                        .unwrap_or(i64::MIN),
                    arg.vmin_vmax
                        .as_ref()
                        .and_then(|(_, v)| match v.0 {
                            svod_ir::ConstValue::Int(value) => Some(value),
                            _ => None,
                        })
                        .unwrap_or(i64::MAX),
                )),
                _ => None,
            })
            .collect();
        spec.var_names = spec.vars.iter().map(|v| v.name.clone()).collect();
        spec.globals = info.globals.clone();
        spec.outs = info.outs.clone();
        spec.ins = info.ins.clone();
        spec.buf_count = spec.globals.len();
        spec.abi = abi;
        spec.global_size = info.global_size.clone();
        spec.local_size = info.local_size.clone();

        validate_abi_descriptors(&spec.abi, spec.buf_count, &spec.var_names)?;

        Ok(spec)
    }
}

/// A variable in the kernel (for symbolic shapes/strides).
///
/// Variables represent symbolic values that are bound at kernel execution time.
/// Examples:
/// - Shape dimensions that vary per input
/// - Stride values computed from shapes
/// - Loop bounds determined by input sizes
#[derive(Debug, Clone)]
pub struct Variable {
    /// Variable name (must be unique within the kernel)
    pub name: String,

    /// Minimum value (for range validation)
    pub min: i64,

    /// Maximum value (for range validation)
    pub max: i64,
}

impl Variable {
    /// Create a new variable.
    pub fn new(name: String, min: i64, max: i64) -> Self {
        Self { name, min, max }
    }
}
