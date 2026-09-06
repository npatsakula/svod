//! Pre-compiled execution plan for kernel execution.
//!
//! `ExecutionPlan` separates one-time preparation (kernel compilation, buffer
//! allocation) from fast repeated execution.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │              PREPARATION (one-time)                      │
//! │  Schedule → instantiate → compile_kernels → build()     │
//! │                       ↓                                  │
//! │                ExecutionPlan                             │
//! └─────────────────────────────────────────────────────────┘
//!                         ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │              EXECUTION (fast path)                       │
//! │  dependency-ordered PreparedOp execution                 │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! let plan = tensor.prepare()?;
//! plan.execute()?;
//! let output = plan.output_buffer();
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use smallvec::SmallVec;
use snafu::ResultExt;
use svod_device::device::ProgramSpec;
use svod_device::{Buffer, BufferId};
use svod_dtype::DeviceSpec;
use svod_ir::origin::{OriginId, OriginSet};
use svod_ir::{CustomFunctionKind, Op, UOp};

use crate::error::{ExecSnafu, Result};
use crate::kernel_cache::CachedKernel;
use crate::profiler::{
    KernelProfile, KernelStaticInfo, ProfileOptions, RunProfile, StageProfile, SubmissionProfileFinalizer,
};
use svod_ir::ops;

type RuntimeLaunchSizes = (Option<[usize; 3]>, Option<[usize; 3]>);

static NEXT_HCQ_SIGNAL_ADDRESS: AtomicU64 = AtomicU64::new(0x1000);

#[derive(Clone, Debug, PartialEq, Eq)]
struct BufferAccess {
    storage: BufferId,
    owner: DeviceSpec,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct HcqPreparedOperation {
    operation: usize,
    device: DeviceSpec,
    queue: svod_device::hcq::QueueKind,
    reads: Vec<BufferAccess>,
    writes: Vec<BufferAccess>,
    is_copy: bool,
}

struct HcqLinkedPlan {
    semantic: svod_device::hcq::SemanticLinkedPlan,
}

impl HcqLinkedPlan {
    fn capture(operations: Vec<HcqPreparedOperation>) -> Result<Self> {
        let topology = operations
            .iter()
            .map(|operation| {
                let resource = |access: &BufferAccess| svod_device::hcq::TopologyResource {
                    id: access.storage.0,
                    owner: access.owner.clone(),
                    start: access.start,
                    end: access.end,
                };
                let reads = operation.reads.iter().map(resource).collect::<Vec<_>>();
                let writes = operation.writes.iter().map(resource).collect::<Vec<_>>();
                let kind = if operation.is_copy && !reads.is_empty() && !writes.is_empty() {
                    svod_device::hcq::TopologyOperationKind::Copy {
                        src: reads[0].clone(),
                        dst: writes[0].clone(),
                        bytes: operation.reads[0].end.saturating_sub(operation.reads[0].start),
                    }
                } else {
                    svod_device::hcq::TopologyOperationKind::Execute
                };
                svod_device::hcq::TopologyOperation {
                    operation: operation.operation,
                    lane: svod_device::hcq::DeviceQueue { device: operation.device.clone(), queue: operation.queue },
                    reads,
                    writes,
                    kind,
                }
            })
            .collect::<Vec<_>>();
        // No backend currently reports verified peer mappings. Cross-device
        // resources therefore stage rather than assuming family-wide access.
        let lanes = svod_device::hcq::schedule_device_lanes(
            &topology,
            svod_device::hcq::QueueMergeLimits::NO_MERGE,
            |executor, owner| executor == owner,
        );
        let semantic = svod_device::hcq::SemanticLinkedPlan::from_lane_submissions(lanes, |_| {
            [
                NEXT_HCQ_SIGNAL_ADDRESS.fetch_add(8, Ordering::Relaxed),
                NEXT_HCQ_SIGNAL_ADDRESS.fetch_add(8, Ordering::Relaxed),
            ]
        })
        .context(ExecSnafu { context: "link semantic HCQ topology" })?;
        Ok(Self { semantic })
    }
}

// ============================================================================
// Core Structures
// ============================================================================

/// A pre-compiled kernel ready for execution.
///
/// Variable values are stored as positional `vals: Vec<i64>` rather than a named
/// HashMap.
#[derive(Clone)]
pub struct PreparedKernel {
    /// Unique identifier (from original AST).
    pub id: u64,

    pub ast: Arc<UOp>,

    /// Compiled kernel program (Arc-shared from cache).
    pub kernel: Arc<CachedKernel>,

    /// Device this kernel executes on.
    pub device: DeviceSpec,

    /// Indices into `ExecutionPlan::buffers` for this kernel's buffers.
    /// Ordered as expected by the kernel (matches codegen buffer order).
    pub buffer_indices: Vec<usize>,

    /// Indices of output buffers within `buffer_indices`.
    pub output_indices: Vec<usize>,

    /// Indices of input buffers within `buffer_indices`.
    pub input_indices: Vec<usize>,

    /// Variable values in positional order (matches `var_names` in CachedKernel).
    pub vals: Vec<i64>,

    /// Fixed variable bindings captured at prepare time.
    ///
    /// Values fixed by scheduling (for example from bound ranges) are not
    /// overridden by `execute_with_vars`.
    pub fixedvars: HashMap<String, i64>,

    /// Kernel IDs that must complete before this one (dependencies).
    pub dependencies: Vec<u64>,

    /// Preparation-time raw buffer addresses retained for diagnostics and graph
    /// hazard metadata. Normal and graph replay resolve the current plan buffers
    /// again, so replacing a buffer does not leave a stale invocation address.
    pub buffer_ptrs: Vec<usize>,

    /// Pre-computed buffer IDs for dependency tracking.
    pub buffer_ids: Vec<BufferId>,

    /// Cached `(name, min_val, max_val)` triples for every `DefineVar` reachable
    /// from `ast`. Populated at construction so `validate_runtime_var_bounds`
    /// doesn't re-toposort on every execute call.
    pub runtime_vars: Vec<RuntimeVar>,

    /// Scope this dispatch is charged to. Per dispatch, not per program: the
    /// `kernel` above is shared by every structurally identical kernel.
    pub origin: Option<OriginId>,

    /// Every scope folded into this kernel by fusion.
    pub origins: OriginSet,
}

impl PreparedKernel {
    /// Hazard read-set positions: every buffer slot the kernel does not write.
    ///
    /// Tinygrad's `DepsTracker.access_resources` (`device.py:280-296`, called
    /// from `graph/hcq.py:224` with `outs` as the write list) takes the whole
    /// buffer list and treats every slot outside `write` as a read.
    /// `input_indices` is only `ProgramSpec.ins` — the declared LOAD slots —
    /// which misses buffers the kernel reads through an alias or an
    /// undeclared global, so it must not drive dependency edges.
    fn read_positions(&self) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(
            self.input_indices.iter().all(|position| *position < self.buffer_indices.len()),
            "kernel {} declares an input position outside its {} buffers",
            self.id,
            self.buffer_indices.len()
        );
        (0..self.buffer_indices.len()).filter(|position| !self.output_indices.contains(position))
    }
}

/// Bound description for one `DefineVar` consumed by a kernel.
#[derive(Clone, Debug)]
pub struct RuntimeVar {
    pub name: String,
    pub min_val: i64,
    pub max_val: i64,
}

/// Walk `root` and collect bounds for every reachable runtime variable, in
/// either spelling: a `DefineVar`, or the bounded scalar `Param` a
/// `svod_tensor::Variable` lowers to.
pub fn collect_runtime_vars(root: &Arc<UOp>) -> Vec<RuntimeVar> {
    let mut vars = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for node in root.toposort() {
        let bounds = match node.op() {
            Op::DefineVar(ops::DefineVar { name, min_val, max_val }) => Some((name.clone(), *min_val, *max_val)),
            Op::Param(ops::Param { arg, .. })
                if arg.addrspace.is_none()
                    && let Some(name) = arg.name.as_deref()
                    && let Some((min, max)) = &arg.vmin_vmax
                    && let (Some(min), Some(max)) = (min.0.try_int(), max.0.try_int()) =>
            {
                Some((name.to_string(), min, max))
            }
            _ => None,
        };
        if let Some((name, min_val, max_val)) = bounds
            && seen.insert(name.clone())
        {
            vars.push(RuntimeVar { name, min_val, max_val });
        }
    }
    vars
}

/// Prepared buffer-to-buffer copy operation.
#[derive(Clone, Debug)]
pub struct PreparedCopy {
    /// Unique operation identifier.
    pub id: u64,

    /// Buffer indices in ExecutionPlan order: [dst, src].
    pub buffer_indices: Vec<usize>,

    /// Operation IDs that must complete before this copy.
    pub dependencies: Vec<u64>,

    /// Scope this copy is charged to.
    pub origin: Option<OriginId>,

    /// Every scope folded into this copy.
    pub origins: OriginSet,
}

/// Prepared custom runtime function operation.
#[derive(Clone, Debug)]
pub struct PreparedCustomFunction {
    /// Unique operation identifier.
    pub id: u64,

    /// Explicit custom function kind (for example: `EncDec`).
    pub kind: CustomFunctionKind,

    /// Runtime descriptor attributes encoded by the IR body.
    pub attrs: SmallVec<[Arc<UOp>; 4]>,

    /// Buffer indices in ExecutionPlan order.
    pub buffer_indices: Vec<usize>,

    /// Bound variable values for this operation.
    pub fixedvars: HashMap<String, i64>,

    /// Operation IDs that must complete before this custom function runs.
    pub dependencies: Vec<u64>,

    /// Cached `(name, min_val, max_val)` triples for every `DefineVar`
    /// reachable from `attrs`. Populated at construction so
    /// `validate_runtime_var_bounds` doesn't re-toposort on every execute call.
    pub runtime_vars: Vec<RuntimeVar>,

    /// Scope this operation is charged to.
    pub origin: Option<OriginId>,

    /// Every scope folded into this operation.
    pub origins: OriginSet,
}

/// Prepared execution item.
#[derive(Clone, Debug)]
pub enum PreparedOp {
    /// Compiled kernel/program operation.
    CompiledProgram(PreparedKernel),

    /// Direct buffer copy operation.
    BufferCopy(PreparedCopy),

    /// Runtime custom function operation.
    CustomFunction(PreparedCustomFunction),
}

fn op_identity(op: &PreparedOp) -> (u64, Vec<u64>) {
    match op {
        PreparedOp::CompiledProgram(kernel) => (kernel.id, kernel.dependencies.clone()),
        PreparedOp::BufferCopy(copy) => (copy.id, copy.dependencies.clone()),
        PreparedOp::CustomFunction(custom) => (custom.id, custom.dependencies.clone()),
    }
}

fn validate_var_bound(name: &str, value: i64, min_val: i64, max_val: i64) -> Result<()> {
    if value < min_val || value > max_val {
        return Err(crate::error::Error::Execution {
            reason: format!("variable {name}={value} is outside bounds [{min_val}, {max_val}]"),
        });
    }
    Ok(())
}

/// Extract `(node_ids, callable_deps)` from prepared ops for the shared
/// topological-leveling routines in [`crate::leveling`].
fn op_graph_inputs(ops: &[PreparedOp]) -> (Vec<u64>, Vec<Vec<u64>>) {
    ops.iter().map(op_identity).unzip()
}

#[cfg(test)]
fn compute_mixed_op_order(ops: &[PreparedOp]) -> Result<Vec<usize>> {
    compute_mixed_op_order_with_instance_dependencies(ops, &[])
}

fn compute_mixed_op_order_with_instance_dependencies(
    ops: &[PreparedOp],
    instance_deps_per_op: &[Vec<usize>],
) -> Result<Vec<usize>> {
    let (node_ids, callable_deps) = op_graph_inputs(ops);
    let index_deps = (!instance_deps_per_op.is_empty()).then_some(instance_deps_per_op);
    crate::leveling::compute_topological_order(&node_ids, &callable_deps, index_deps)
}

#[cfg(test)]
fn compute_execution_levels(ops: &[PreparedOp]) -> Result<Vec<Vec<usize>>> {
    compute_execution_levels_with_instance_dependencies(ops, &[])
}

fn compute_execution_levels_with_instance_dependencies(
    ops: &[PreparedOp],
    instance_deps_per_op: &[Vec<usize>],
) -> Result<Vec<Vec<usize>>> {
    let (node_ids, callable_deps) = op_graph_inputs(ops);
    let index_deps = (!instance_deps_per_op.is_empty()).then_some(instance_deps_per_op);
    crate::leveling::compute_topological_levels(&node_ids, &callable_deps, index_deps)
}

/// Pre-compiled execution plan for a computation graph.
///
/// Created once via `prepare()`, then executed multiple times.
/// The plan owns all its buffers and compiled kernels.
pub struct ExecutionPlan {
    /// Prepared operations in schedule order.
    ops: Vec<PreparedOp>,

    /// Concrete op-index dependencies parallel to `ops`.
    op_instance_dependencies: Vec<Vec<usize>>,

    /// Precomputed dependency-safe operation order.
    op_order: Vec<usize>,

    /// Topological levels of dependency-independent operations. Preserved as
    /// the execution-iteration order (each level flushed before the next) for
    /// consistency with pre-Step-6 plan semantics — some downstream kernel
    /// algorithms (e.g. iterative QR) are sensitive to within-level
    /// scheduling order vs. a single flat topological linearization.
    op_levels: Vec<Vec<usize>>,

    /// ALL buffers owned by this plan (inputs, intermediates, outputs).
    buffers: Vec<Buffer>,

    /// Mapping: AST id → buffer index (for kernel buffer binding).
    ast_to_buffer: HashMap<u64, usize>,

    /// Indices of output buffers in `buffers` (matches SINK source order).
    output_buffer_indices: Vec<usize>,

    /// Indices of buffers declared host-written inputs via
    /// [`ExecutionPlan::declare_input`]. Drives the replicate fork policy and
    /// carries over to replicas.
    input_buffer_indices: HashSet<usize>,

    /// One representative buffer index per distinct storage (arena views
    /// share one storage), for scoped-sync completion-token recording.
    distinct_storage_indices: Vec<usize>,

    /// Primary device for this plan.
    device: DeviceSpec,

    /// Last dynamic variable bindings supplied through `execute_with_vars`.
    runtime_var_vals: HashMap<String, i64>,

    /// Captured replayable graph, built lazily on first `execute()`. `Some(None)`
    /// means the chain isn't graphable (mixed ops / non-graph device) → per-call
    /// dispatch. Replaces N per-kernel submits with one; see
    /// `svod_device::Graph`.
    graph: std::sync::OnceLock<Option<Box<dyn svod_device::Graph>>>,

    /// Reusable per-plan execution context, minted lazily from the first
    /// kernel's program (`Program::new_exec_context`) and held for the plan's
    /// lifetime so every kernel dispatches onto the same backend queue (distinct
    /// plans → distinct queues for cross-plan parallelism). `Some(None)` means
    /// the backend has no reusable context (CPU) → per-call `Program::execute`.
    plan_ctx: std::sync::OnceLock<Option<Box<dyn svod_device::PlanContext>>>,

    hcq_executor: Mutex<svod_device::hcq::CpuQueueExecutor>,
    hcq_linked: std::sync::OnceLock<HcqLinkedPlan>,
    /// First failure after a semantic HCQ epoch reserved timeline points.
    /// Such an epoch may be partially executed and must never be retried.
    hcq_poison: std::sync::OnceLock<String>,
}

// ============================================================================
// ExecutionPlan Implementation
// ============================================================================

impl ExecutionPlan {
    fn check_hcq_poison(&self) -> Result<()> {
        if let Some(reason) = self.hcq_poison.get() {
            return Err(crate::error::Error::PlanPoisoned { reason: reason.clone() });
        }
        Ok(())
    }

    fn poison_hcq<T>(&self, result: Result<T>) -> Result<T> {
        if let Err(error) = &result {
            let _ = self.hcq_poison.set(error.to_string());
        }
        result
    }

    fn replay_native_linked_plan(&self) -> Result<svod_device::device::NativeReplayOutcome> {
        use svod_device::device::{CopyEndpoint, NativeReplayDecline, NativeReplayOutcome};

        let semantic = &self.hcq_linked.get().expect("HCQ plan linked by builder").semantic;
        if let Some(operation) = semantic.staged_copy() {
            return Ok(NativeReplayOutcome::Declined(NativeReplayDecline::StagedCopy { operation }));
        }
        if let Some(expected) = semantic.lanes().first().map(|submission| &submission.lane.device)
            && let Some(actual) =
                semantic.lanes().iter().map(|submission| &submission.lane.device).find(|d| d != &expected)
        {
            return Ok(NativeReplayOutcome::Declined(NativeReplayDecline::MixedComputeDevices {
                expected: expected.clone(),
                actual: actual.clone(),
            }));
        }

        enum CallValues<'a> {
            Program {
                program: &'a dyn svod_device::Program,
                buffers: Vec<u64>,
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

        let devices = self
            .ops
            .iter()
            .filter_map(|op| match op {
                PreparedOp::CompiledProgram(kernel) => Some(&kernel.device),
                _ => None,
            })
            .collect::<HashSet<_>>();
        // A PlanContext owns one physical device. Cross-device linked replay is
        // retained independently by each backend context; until a backend
        // exposes that context set, never publish another device's addresses on
        // the first device's queue.
        if devices.len() > 1 {
            let mut devices = devices.into_iter();
            let expected = devices.next().unwrap().clone();
            let actual = devices.next().unwrap().clone();
            return Ok(NativeReplayOutcome::Declined(NativeReplayDecline::MixedComputeDevices { expected, actual }));
        }
        let Some(owner) = devices.into_iter().next() else {
            return Ok(NativeReplayOutcome::Declined(NativeReplayDecline::NoCompiledProgram));
        };

        // One walk over the plan's endpoints. Native linked replay submits every
        // kernel argument AND every copy endpoint through this context's
        // queues; until peer mappings are explicit, all of them must belong to
        // that exact physical device. Rechecked on every replay so a
        // replacement buffer cannot patch a foreign VA into a cached plan —
        // hence the single up-front `NativeDevice::resolve`, which keeps the
        // process-global AMD device-cache lock out of the per-buffer path.
        let native = svod_device::buffer::NativeDevice::resolve(owner);
        for op in &self.ops {
            let (id, endpoints): (u64, Vec<(usize, Option<CopyEndpoint>)>) = match op {
                PreparedOp::CompiledProgram(kernel) => {
                    (kernel.id, kernel.buffer_indices.iter().map(|&index| (index, None)).collect())
                }
                PreparedOp::BufferCopy(copy) => (
                    copy.id,
                    [CopyEndpoint::Destination, CopyEndpoint::Source]
                        .into_iter()
                        .enumerate()
                        .filter_map(|(position, endpoint)| {
                            copy.buffer_indices.get(position).map(|&index| (index, Some(endpoint)))
                        })
                        .collect(),
                ),
                PreparedOp::CustomFunction(_) => continue,
            };
            for (argument, (buffer_index, endpoint)) in endpoints.into_iter().enumerate() {
                let Some(buffer) = self.buffers.get(buffer_index) else { continue };
                let actual = buffer.device_spec();
                if &actual != owner {
                    return Ok(NativeReplayOutcome::Declined(match endpoint {
                        Some(endpoint) => NativeReplayDecline::ForeignCopyEndpoint {
                            operation: id,
                            endpoint,
                            expected: owner.clone(),
                            actual,
                        },
                        None => NativeReplayDecline::ForeignProgramEndpoint {
                            operation: id,
                            argument,
                            expected: owner.clone(),
                            actual,
                        },
                    }));
                }
                if !buffer
                    .matches_native(&native)
                    .context(ExecSnafu { context: format!("validate native operation {id} endpoint {argument}") })?
                {
                    return Ok(NativeReplayOutcome::Declined(match endpoint {
                        Some(endpoint) => NativeReplayDecline::IncompatibleCopyAllocation {
                            operation: id,
                            endpoint,
                            expected: owner.clone(),
                        },
                        None => NativeReplayDecline::IncompatibleProgramAllocation {
                            operation: id,
                            argument,
                            expected: owner.clone(),
                        },
                    }));
                }
            }
        }
        let Some(first) = self.op_levels.iter().flatten().find_map(|&index| match &self.ops[index] {
            PreparedOp::CompiledProgram(kernel) => Some(kernel.kernel.program.as_ref()),
            _ => None,
        }) else {
            return Ok(NativeReplayOutcome::Declined(NativeReplayDecline::NoCompiledProgram));
        };
        let Some(ctx) = self.plan_ctx(first)? else {
            return Ok(NativeReplayOutcome::Declined(NativeReplayDecline::NoPlanContext));
        };
        let mut values = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            values.push(match op {
                PreparedOp::CompiledProgram(kernel) => {
                    let buffers = kernel
                        .buffer_indices
                        .iter()
                        .map(|&index| {
                            self.buffers[index].device_address().context(ExecSnafu {
                                context: format!("resolve linked replay buffer for kernel {}", kernel.id),
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let (global_size, local_size) = Self::kernel_launch_sizes(kernel)?;
                    CallValues::Program {
                        program: kernel.kernel.program.as_ref(),
                        buffers,
                        vals: &kernel.vals,
                        global_size,
                        local_size,
                    }
                }
                PreparedOp::BufferCopy(copy) if copy.buffer_indices.len() >= 2 => {
                    match (self.buffers.get(copy.buffer_indices[0]), self.buffers.get(copy.buffer_indices[1])) {
                        (Some(dst), Some(src)) if dst.size() == src.size() => CallValues::Copy {
                            dst: dst
                                .device_address()
                                .context(ExecSnafu { context: format!("resolve linked copy {} dst", copy.id) })?,
                            src: src
                                .device_address()
                                .context(ExecSnafu { context: format!("resolve linked copy {} src", copy.id) })?,
                            bytes: dst.size(),
                        },
                        _ => CallValues::Unsupported,
                    }
                }
                _ => CallValues::Unsupported,
            });
        }
        let calls = values
            .iter()
            .map(|call| match call {
                CallValues::Program { program, buffers, vals, global_size, local_size } => {
                    svod_device::PlanCall::Program {
                        program: *program,
                        buffers,
                        vals,
                        global_size: *global_size,
                        local_size: *local_size,
                    }
                }
                CallValues::Copy { dst, src, bytes } => {
                    svod_device::PlanCall::Copy { dst: *dst, src: *src, bytes: *bytes }
                }
                CallValues::Unsupported => svod_device::PlanCall::Unsupported,
            })
            .collect::<Vec<_>>();
        ctx.replay_linked_plan(semantic, &calls).context(ExecSnafu { context: "replay native linked HCQ plan" })
    }

    fn buffer_access(&self, index: usize) -> Result<BufferAccess> {
        let buffer = self.buffers.get(index).ok_or_else(|| crate::error::Error::Execution {
            reason: format!("HCQ buffer index {index} out of range ({} buffers)", self.buffers.len()),
        })?;
        Ok(BufferAccess {
            storage: buffer.storage_id(),
            owner: buffer.device_spec(),
            start: buffer.offset(),
            end: buffer.offset().saturating_add(buffer.size()),
        })
    }

    fn graph_endpoints_match_device(&self) -> Result<bool> {
        // Resolve once per distinct kernel device instead of once per buffer:
        // a captured chain is single-device, so this is one resolve per plan.
        let mut resolved: Option<(DeviceSpec, svod_device::buffer::NativeDevice)> = None;
        for op in &self.ops {
            let PreparedOp::CompiledProgram(kernel) = op else { continue };
            if resolved.as_ref().is_none_or(|(spec, _)| spec != &kernel.device) {
                resolved = Some((kernel.device.clone(), svod_device::buffer::NativeDevice::resolve(&kernel.device)));
            }
            let native = &resolved.as_ref().expect("resolved above").1;
            for &index in &kernel.buffer_indices {
                if self.buffers[index].device_spec() != kernel.device
                    || !self.buffers[index]
                        .matches_native(native)
                        .context(ExecSnafu { context: format!("validate graph kernel {} endpoint", kernel.id) })?
                {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn hcq_operations(&self) -> Result<Vec<HcqPreparedOperation>> {
        use svod_device::hcq::QueueKind;

        let mut operations = Vec::with_capacity(self.ops.len());
        for &operation in self.op_levels.iter().flatten() {
            let (device, queue, read_indices, write_indices, is_copy) = match &self.ops[operation] {
                PreparedOp::CompiledProgram(kernel) => {
                    let reads = kernel.read_positions().map(|position| kernel.buffer_indices[position]).collect();
                    let writes =
                        kernel.output_indices.iter().map(|&position| kernel.buffer_indices[position]).collect();
                    (kernel.device.clone(), QueueKind::Compute(0), reads, writes, false)
                }
                PreparedOp::BufferCopy(copy) => {
                    let device = copy
                        .buffer_indices
                        .first()
                        .and_then(|&index| self.buffers.get(index))
                        .map(Buffer::device_spec)
                        .unwrap_or_else(|| self.device.clone());
                    if copy.buffer_indices.len() < 2 {
                        (device, QueueKind::Copy(0), Vec::new(), Vec::new(), true)
                    } else {
                        (device, QueueKind::Copy(0), vec![copy.buffer_indices[1]], vec![copy.buffer_indices[0]], true)
                    }
                }
                PreparedOp::CustomFunction(custom) => {
                    // Custom functions do not expose an outs list. Conservatively
                    // model every argument as both read and written.
                    (
                        self.device.clone(),
                        QueueKind::Compute(0),
                        custom.buffer_indices.clone(),
                        custom.buffer_indices.clone(),
                        false,
                    )
                }
            };
            let map_access = |index| -> Result<BufferAccess> {
                // Keep malformed calls capturable so their established typed
                // execution errors are still reported when the call runs.
                Ok(self.buffer_access(index).unwrap_or(BufferAccess {
                    storage: BufferId(u64::MAX - index as u64),
                    owner: self.device.clone(),
                    start: 0,
                    end: 1,
                }))
            };
            let reads = read_indices.into_iter().map(map_access).collect::<Result<Vec<_>>>()?;
            let writes = write_indices.into_iter().map(map_access).collect::<Result<Vec<_>>>()?;
            operations.push(HcqPreparedOperation { operation, device, queue, reads, writes, is_copy });
        }
        Ok(operations)
    }

    fn submission_error(error: svod_device::hcq::SubmissionExecutionError<crate::error::Error>) -> crate::error::Error {
        match error {
            svod_device::hcq::SubmissionExecutionError::Queue(source) => {
                crate::error::Error::Exec { source, context: "CPU HCQ submission".into() }
            }
            svod_device::hcq::SubmissionExecutionError::Execute(error) => error,
        }
    }

    fn kernel_launch_sizes(kernel: &PreparedKernel) -> Result<RuntimeLaunchSizes> {
        let mut vars: HashMap<&str, i64> =
            HashMap::with_capacity(kernel.kernel.var_names.len() + kernel.fixedvars.len());
        for (idx, name) in kernel.kernel.var_names.iter().enumerate() {
            let value = kernel.vals.get(idx).copied().ok_or_else(|| crate::error::Error::Execution {
                reason: format!(
                    "Kernel {} has {} var names but only {} values",
                    kernel.id,
                    kernel.kernel.var_names.len(),
                    kernel.vals.len()
                ),
            })?;
            vars.insert(name.as_str(), value);
        }
        for (name, value) in &kernel.fixedvars {
            vars.insert(name.as_str(), *value);
        }

        let dims =
            ProgramSpec::resolve_launch_dims(&kernel.kernel.global_size, kernel.kernel.local_size.as_ref(), &vars)
                .context(ExecSnafu { context: format!("kernel {} launch dimensions", kernel.id) })?;
        Ok((Some(dims.global_size), dims.local_size))
    }

    /// Lazily capture all kernels into a backend replay graph. Only backends
    /// that provide a graph factory install one; everything else (and any
    /// non-graphable chain) returns `None` → per-call dispatch. Gated to chains
    /// that are *all* compiled kernels with no runtime vars: copies/views/custom
    /// or dynamic launch dims keep the host in the loop and aren't graphed.
    fn graph(&self) -> &Option<Box<dyn svod_device::Graph>> {
        self.graph.get_or_init(|| self.build_graph().unwrap_or(None))
    }

    fn build_graph(&self) -> Result<Option<Box<dyn svod_device::Graph>>> {
        // Graph capture is on by default: an all-static compiled-kernel plan on a
        // graphable device replays the whole chain as one backend submit, instead
        // of the per-kernel dispatch round-trip. Validated against per-call across
        // the tensor suite (incl. multi-kernel decompositions). Capture walks
        // `op_levels` execution order, NOT the flat `op_order` topological sort
        // (below). Non-graphable plans (runtime vars, no graph factory, chains the
        // backend declines to capture, mixed devices) fall back to per-call via
        // the `Ok(None)` returns below.
        let all_static_kernels = self.ops.iter().all(
            |op| matches!(op, PreparedOp::CompiledProgram(k) if k.runtime_vars.is_empty() && k.device == self.device),
        );
        if !all_static_kernels || self.ops.is_empty() {
            tracing::debug!(
                target: "svod_runtime::graph",
                ops = self.ops.len(),
                compiled = self.ops.iter().filter(|o| matches!(o, PreparedOp::CompiledProgram(_))).count(),
                with_runtime_vars =
                    self.ops.iter().filter(|o| matches!(o, PreparedOp::CompiledProgram(k) if !k.runtime_vars.is_empty())).count(),
                custom = self.ops.iter().filter(|o| matches!(o, PreparedOp::CustomFunction(_))).count(),
                copies = self.ops.iter().filter(|o| matches!(o, PreparedOp::BufferCopy(_))).count(),
                "graph: per-call fallback (not all-static-compiled)"
            );
            return Ok(None);
        }
        let dev = crate::device_registry::DEVICE_FACTORIES.device(&self.device, svod_device::registry::registry())?;
        let Some(factory) = dev.graph.clone() else { return Ok(None) };
        // Capture in the SAME order `execute` runs the kernels — flatten
        // `op_levels` (level-by-level, intra-level in index order), NOT the flat
        // `op_order` topological sort. The two can differ, and a captured graph
        // replays its packets in strict queue (FIFO) order; using `op_order`
        // would dispatch a different sequence than the per-call path, corrupting
        // results whenever a reused buffer's ordering relies on the level walk
        // (e.g. multi-kernel decompositions like QR).
        // Walk the emission order (level-by-level, intra-level index order) once,
        // building the GraphKernel list AND a parallel hazard-dependency list in
        // lock-step. Hazards are keyed on the RESOLVED buffer GVA (`buffer_ptrs`),
        // not buffer ids: the memory planner aliases distinct logical buffers onto
        // one GVA, so a GVA-keyed walk catches the WAR/WAW the logical
        // `dependencies` field misses. For each emitted kernel `e`:
        //   reads  = buffer_ptrs[j] for j NOT in output_indices
        //   writes = buffer_ptrs[j] for j     in output_indices
        //   deps   = last_writer[read]  (RAW)
        //          ∪ last_writer[write] (WAW) ∪ readers[write] (WAR)
        // then update: readers[read].push(e); for each write set last_writer=e and
        // clear readers (a fresh writer; future readers depend on it via RAW).
        //
        // Soundness rests on `output_indices` being the COMPLETE write-set: a
        // missed write would leave no last_writer and (BARRIER stripped) race a
        // later reader. That holds here by construction — `output_indices` is
        // derived from the kernel's STORE targets (`ProgramSpec.outs`) and a
        // compiled kernel writes only via STOREs, and this walk only processes
        // `CompiledProgram` ops (the `else { return Ok(None) }` below). Custom
        // functions / copies are not graphed, so the invariant is not relied on
        // for them.
        let mut kernels = Vec::with_capacity(self.ops.len());
        let mut last_writer: HashMap<usize, usize> = HashMap::new();
        let mut readers: HashMap<usize, Vec<usize>> = HashMap::new();
        for level in &self.op_levels {
            for &idx in level {
                let PreparedOp::CompiledProgram(k) = &self.ops[idx] else { return Ok(None) };
                let (global_size, local_size) = Self::kernel_launch_sizes(k)?;
                let e = kernels.len();

                let current_addresses = k
                    .buffer_indices
                    .iter()
                    .map(|&buffer| {
                        self.buffers[buffer]
                            .device_address()
                            .context(ExecSnafu { context: format!("resolve graph buffer for kernel {}", k.id) })
                            .map(|address| address as usize)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let writes: Vec<usize> =
                    k.output_indices.iter().filter_map(|&j| current_addresses.get(j).copied()).collect();
                let reads: Vec<usize> = k.read_positions().filter_map(|j| current_addresses.get(j).copied()).collect();

                let mut deps: std::collections::HashSet<usize> = std::collections::HashSet::new();
                for &b in &reads {
                    if let Some(&w) = last_writer.get(&b) {
                        deps.insert(w); // RAW
                    }
                }
                for &b in &writes {
                    if let Some(&w) = last_writer.get(&b) {
                        deps.insert(w); // WAW
                    }
                    if let Some(rs) = readers.get(&b) {
                        deps.extend(rs.iter().copied()); // WAR
                    }
                }
                deps.remove(&e);
                let mut deps: Vec<usize> = deps.into_iter().collect();
                deps.sort_unstable();

                // Commit this kernel's effect on the hazard state.
                for &b in &reads {
                    readers.entry(b).or_default().push(e);
                }
                for &b in &writes {
                    last_writer.insert(b, e);
                    readers.insert(b, Vec::new());
                }

                kernels.push(svod_device::GraphKernel {
                    program: k.kernel.program.as_ref(),
                    buffers: current_addresses.iter().map(|&p| p as *mut u8).collect(),
                    vals: k.vals.clone(),
                    global_size,
                    local_size,
                    deps,
                });
            }
        }
        let result = factory(&kernels).context(ExecSnafu { context: "graph capture" })?;
        tracing::debug!(target: "svod_runtime::graph", kernels = kernels.len(), captured = result.is_some(), "graph: capture result");
        Ok(result)
    }

    /// Lazily mint (once) the plan's execution context from `program` and cache
    /// it for the plan's lifetime. `None` ⇒ the backend has no reusable context
    /// (CPU) and the caller dispatches per-call via `Program::execute`. The
    /// context binds the plan to a shared queue; distinct plans spread onto
    /// distinct queues for cross-plan parallelism.
    fn plan_ctx(&self, program: &dyn svod_device::Program) -> Result<Option<&dyn svod_device::PlanContext>> {
        if let Some(slot) = self.plan_ctx.get() {
            return Ok(slot.as_deref());
        }
        let ctx = program.new_exec_context().context(ExecSnafu { context: "mint plan exec context" })?;
        // One-shot init race: if two threads see empty, both mint; only one wins
        // `set()`. The loser's context drops here harmlessly (its `Arc` over the
        // shared queue just decrements).
        let _ = self.plan_ctx.set(ctx);
        Ok(self.plan_ctx.get().expect("set above").as_deref())
    }

    /// Submit one kernel. When `profile` is set and the backend stamps
    /// dispatches, returns the dispatch's HW timestamp handle (`None` otherwise,
    /// e.g. CPU); the caller must hold it until after `synchronize`. The
    /// non-profiled `execute` path passes `false` and drops the handle.
    #[inline]
    fn execute_kernel(
        &self,
        kernel: &PreparedKernel,
        profile: bool,
    ) -> Result<Option<Arc<dyn svod_device::DispatchTimestamps>>> {
        let buffer_ptrs: SmallVec<[*mut u8; 8]> = kernel
            .buffer_indices
            .iter()
            .map(|&index| {
                self.buffers[index]
                    .device_address()
                    .map(|address| address as *mut u8)
                    .context(ExecSnafu { context: format!("resolve replay buffer for kernel {}", kernel.id) })
            })
            .collect::<Result<_>>()?;
        let (global_size, local_size) = Self::kernel_launch_sizes(kernel)?;
        let program = kernel.kernel.program.as_ref();
        // Backends that expose a reusable context dispatch through it so all the
        // plan's kernels share one queue. Others (CPU) return `None` and fall
        // back to per-call `Program::execute`.
        if kernel.device == self.device
            && let Some(ctx) = self.plan_ctx(program)?
        {
            return unsafe { ctx.dispatch(program, &buffer_ptrs, &kernel.vals, global_size, local_size, profile) }
                .context(ExecSnafu { context: format!("dispatch kernel {}", kernel.id) });
        }
        unsafe {
            program
                // wait=false: async submit. GPU ordering is enforced by the
                // device timeline; host reads (copyout / as_*) synchronize.
                .execute(&buffer_ptrs, &kernel.vals, global_size, local_size, /*wait=*/ false)
                .map(|_| None)
                .context(ExecSnafu { context: format!("execute kernel {}", kernel.id) })
        }
    }

    fn validate_runtime_var_bounds(&self, var_vals: &[(&str, i64)]) -> Result<()> {
        let vals_map: HashMap<&str, i64> = var_vals.iter().copied().collect();
        for op in &self.ops {
            match op {
                PreparedOp::CompiledProgram(kernel) => {
                    for var in &kernel.runtime_vars {
                        if kernel.fixedvars.contains_key(&var.name) || var.name == "core_id" {
                            continue;
                        }
                        // A variable left unbound at prepare time still carries a
                        // placeholder in `vals`; validate that too, so a binding
                        // that never arrives is caught here rather than at prepare.
                        let current = kernel
                            .kernel
                            .var_names
                            .iter()
                            .position(|name| name == &var.name)
                            .and_then(|index| kernel.vals.get(index))
                            .copied();
                        if let Some(value) = vals_map.get(var.name.as_str()).copied().or(current) {
                            validate_var_bound(&var.name, value, var.min_val, var.max_val)?;
                        }
                    }
                }
                PreparedOp::CustomFunction(custom) => {
                    for var in &custom.runtime_vars {
                        if custom.fixedvars.contains_key(&var.name) || var.name == "core_id" {
                            continue;
                        }
                        if let Some(&value) = vals_map.get(var.name.as_str()) {
                            validate_var_bound(&var.name, value, var.min_val, var.max_val)?;
                        }
                    }
                }
                PreparedOp::BufferCopy(_) => {}
            }
        }
        Ok(())
    }

    fn update_runtime_var_vals(&mut self, var_vals: &[(&str, i64)]) -> Result<()> {
        self.validate_runtime_var_bounds(var_vals)?;

        let vals_map: HashMap<&str, i64> = var_vals.iter().copied().collect();
        for &(name, value) in var_vals {
            if name == "core_id" {
                continue;
            }
            self.runtime_var_vals.insert(name.to_string(), value);
        }
        for op in &mut self.ops {
            if let PreparedOp::CompiledProgram(kernel) = op {
                for (idx, name) in kernel.kernel.var_names.iter().enumerate() {
                    if kernel.fixedvars.contains_key(name) || name == "core_id" {
                        continue;
                    }
                    if let Some(&v) = vals_map.get(name.as_str()) {
                        let Some(slot) = kernel.vals.get_mut(idx) else {
                            return Err(crate::error::Error::Execution {
                                reason: format!(
                                    "Kernel {} has {} var names but only {} values",
                                    kernel.id,
                                    kernel.kernel.var_names.len(),
                                    kernel.vals.len()
                                ),
                            });
                        };
                        *slot = v;
                    }
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn execute_copy(&self, copy: &PreparedCopy) -> Result<()> {
        if copy.buffer_indices.len() < 2 {
            return Err(crate::error::Error::Execution {
                reason: format!(
                    "Copy op {} requires at least two buffer indices (dst, src), got {}",
                    copy.id,
                    copy.buffer_indices.len()
                ),
            });
        }
        let dst_idx = copy.buffer_indices[0];
        let src_idx = copy.buffer_indices[1];

        if dst_idx >= self.buffers.len() || src_idx >= self.buffers.len() {
            return Err(crate::error::Error::Execution {
                reason: format!(
                    "Copy op {} buffer index out of range: dst={}, src={}, total_buffers={}",
                    copy.id,
                    dst_idx,
                    src_idx,
                    self.buffers.len()
                ),
            });
        }

        let mut dst = self.buffers[dst_idx].clone();
        let src = &self.buffers[src_idx];
        dst.copy_from(src).context(ExecSnafu { context: format!("copy op {}", copy.id) })
    }

    fn copy_buffers(&self, operation: usize) -> Result<(&Buffer, &Buffer)> {
        let PreparedOp::BufferCopy(copy) = self.ops.get(operation).ok_or_else(|| crate::error::Error::Execution {
            reason: format!("HCQ copy operation {operation} is out of range"),
        })?
        else {
            return Err(crate::error::Error::Execution {
                reason: format!("HCQ copy leg references non-copy operation {operation}"),
            });
        };
        if copy.buffer_indices.len() < 2 {
            return Err(crate::error::Error::Execution {
                reason: format!("Copy op {} requires at least two buffer indices", copy.id),
            });
        }
        let dst = self.buffers.get(copy.buffer_indices[0]).ok_or_else(|| crate::error::Error::Execution {
            reason: format!("Copy op {} destination buffer is out of range", copy.id),
        })?;
        let src = self.buffers.get(copy.buffer_indices[1]).ok_or_else(|| crate::error::Error::Execution {
            reason: format!("Copy op {} source buffer is out of range", copy.id),
        })?;
        if dst.size() != src.size() {
            return Err(crate::error::Error::Execution {
                reason: format!("Copy op {} size mismatch: dst={}, src={}", copy.id, dst.size(), src.size()),
            });
        }
        Ok((dst, src))
    }

    fn execute_topology_command(
        &self,
        command: &svod_device::hcq::TopologyCommand,
        staging: &mut HashMap<usize, Vec<u8>>,
    ) -> Result<()> {
        use svod_device::hcq::CopyLeg;

        match command.copy_leg {
            None => self.execute_op(&self.ops[command.operation]),
            Some(CopyLeg::Direct) => {
                let PreparedOp::BufferCopy(copy) = &self.ops[command.operation] else { unreachable!() };
                self.execute_copy(copy)
            }
            Some(CopyLeg::ToHost) => {
                let (_, src) = self.copy_buffers(command.operation)?;
                let mut bytes = vec![0; src.size()];
                src.copyout(&mut bytes).context(ExecSnafu { context: "HCQ staged copy to host" })?;
                staging.insert(command.operation, bytes);
                Ok(())
            }
            Some(CopyLeg::FromHost) => {
                let (dst, _) = self.copy_buffers(command.operation)?;
                let bytes = staging.remove(&command.operation).ok_or_else(|| crate::error::Error::Execution {
                    reason: format!("HCQ staged copy {} has no host epoch buffer", command.operation),
                })?;
                let mut dst = dst.clone();
                dst.copyin(&bytes).context(ExecSnafu { context: "HCQ staged copy from host" })
            }
        }
    }

    #[inline]
    fn execute_custom_function(&self, custom: &PreparedCustomFunction) -> Result<()> {
        let mut buffers = Vec::with_capacity(custom.buffer_indices.len());
        for &idx in &custom.buffer_indices {
            let Some(buffer) = self.buffers.get(idx) else {
                return Err(crate::error::Error::Execution {
                    reason: format!(
                        "Custom function op {} ({:?}) buffer index out of range: idx={}, total_buffers={}",
                        custom.id,
                        custom.kind,
                        idx,
                        self.buffers.len()
                    ),
                });
            };
            buffers.push(buffer.clone());
        }

        let mut vars = self.runtime_var_vals.clone();
        vars.extend(custom.fixedvars.iter().map(|(k, v)| (k.clone(), *v)));

        crate::custom_function::run_custom_function(&custom.kind, &custom.attrs, &mut buffers, &vars).map_err(|e| {
            // Pass typed `Unsupported` errors through unchanged so callers can match on `kind`.
            // Other errors are wrapped with op context for debugging.
            match e {
                crate::error::Error::Unsupported { .. } => e,
                other => crate::error::Error::Execution {
                    reason: format!("Custom function op {} ({:?}) failed: {other}", custom.id, custom.kind),
                },
            }
        })
    }

    #[inline]
    fn execute_op(&self, op: &PreparedOp) -> Result<()> {
        match op {
            PreparedOp::CompiledProgram(kernel) => self.execute_kernel(kernel, /*profile=*/ false).map(|_| ()),
            PreparedOp::BufferCopy(copy) => self.execute_copy(copy),
            PreparedOp::CustomFunction(custom) => self.execute_custom_function(custom),
        }
    }

    /// Get the first (or only) output buffer after execution.
    ///
    /// Returns `None` for plans with no output buffers (for example, plans
    /// constructed before `set_output_buffer*` is called).
    pub fn output_buffer(&self) -> Option<&Buffer> {
        self.output_buffer_indices.first().and_then(|&i| self.buffers.get(i))
    }

    /// Get output buffer by position (matches SINK source order for batch).
    ///
    /// Returns `None` if `position` is out of range.
    pub fn output_buffer_at(&self, position: usize) -> Option<&Buffer> {
        self.output_buffer_indices.get(position).and_then(|&i| self.buffers.get(i))
    }

    /// Get all output buffers.
    pub fn output_buffers(&self) -> Vec<&Buffer> {
        self.output_buffer_indices.iter().map(|&i| &self.buffers[i]).collect()
    }

    /// Number of outputs in this plan.
    pub fn num_outputs(&self) -> usize {
        self.output_buffer_indices.len()
    }

    /// Copy `len` bytes from output `out_pos` (`src_off`) into the plan buffer
    /// at `dst_index` (`dst_off`) — both owned by this plan, so the borrow is
    /// split internally. The transfer stays on-device (SDMA when either side
    /// is device-local), letting recurrent state recycle output→input without
    /// a host round-trip.
    pub fn copy_output_region_to_buffer(
        &mut self,
        out_pos: usize,
        dst_index: usize,
        dst_off: usize,
        src_off: usize,
        len: usize,
    ) -> Result<()> {
        let src_index = *self.output_buffer_indices.get(out_pos).ok_or_else(|| crate::error::Error::Execution {
            reason: format!("copy_output_region_to_buffer: output {out_pos} out of range"),
        })?;
        let src = self.buffers.get(src_index).ok_or_else(|| crate::error::Error::Execution {
            reason: format!("copy_output_region_to_buffer: source buffer {src_index} out of range"),
        })?;
        let dst = self.buffers.get(dst_index).ok_or_else(|| crate::error::Error::Execution {
            reason: format!("copy_output_region_to_buffer: destination buffer {dst_index} out of range"),
        })?;
        if src.storage_id() == dst.storage_id() {
            return Err(crate::error::Error::Execution {
                reason: "copy_output_region_to_buffer: output aliases destination".into(),
            });
        }
        let (dst, src) = if dst_index < src_index {
            let (a, b) = self.buffers.split_at_mut(src_index);
            (&mut a[dst_index], &b[0])
        } else {
            let (a, b) = self.buffers.split_at_mut(dst_index);
            (&mut b[0], &a[src_index])
        };
        dst.copy_region_from(dst_off, src, src_off, len).context(ExecSnafu { context: "on-device state copy" })
    }

    /// Get a buffer by AST id (for reading intermediate results).
    pub fn buffer(&self, ast_id: u64) -> Option<&Buffer> {
        self.ast_to_buffer.get(&ast_id).map(|&idx| &self.buffers[idx])
    }

    /// Get a mutable buffer by AST id (for `copyin()` on input buffers).
    pub fn buffer_mut_by_id(&mut self, ast_id: u64) -> Option<&mut Buffer> {
        self.ast_to_buffer.get(&ast_id).copied().map(|idx| &mut self.buffers[idx])
    }

    /// Get the primary device for this plan.
    pub fn device(&self) -> &DeviceSpec {
        &self.device
    }

    /// Get all buffers owned by this plan.
    pub fn buffers(&self) -> &[Buffer] {
        &self.buffers
    }

    /// Get mutable access to all buffers owned by this plan.
    pub fn buffers_mut(&mut self) -> &mut [Buffer] {
        &mut self.buffers
    }

    /// Get a mutable buffer by its index in the buffers array.
    pub fn buffer_at_mut(&mut self, index: usize) -> Option<&mut Buffer> {
        self.buffers.get_mut(index)
    }

    /// Get all prepared kernels.
    pub fn prepared_kernels(&self) -> Vec<&PreparedKernel> {
        self.ops
            .iter()
            .filter_map(|op| match op {
                PreparedOp::CompiledProgram(kernel) => Some(kernel),
                _ => None,
            })
            .collect()
    }

    /// Get all prepared operations in schedule order.
    pub fn prepared_ops(&self) -> &[PreparedOp] {
        &self.ops
    }

    /// Iterate over compiled kernels (for inspecting generated source code).
    pub fn kernels(&self) -> impl Iterator<Item = &CachedKernel> {
        self.ops.iter().filter_map(|op| match op {
            PreparedOp::CompiledProgram(kernel) => Some(kernel.kernel.as_ref()),
            _ => None,
        })
    }

    /// Execute the plan.
    ///
    /// Walks `op_levels` level-by-level and runs each op within a level in
    /// builder-insertion order. Multi-plan concurrency comes from distinct
    /// `ExecutionPlan`s (e.g. BEAM search candidates) spread onto distinct
    /// backend execution contexts from the device — not from rayon inside one
    /// plan. The level-by-level iteration (vs. a flat `op_order` topological
    /// linearization) is load-bearing for iterative CPU kernels (QR, etc.)
    /// whose codegen is sensitive to within-level scheduling order — see
    /// `test_execute_walks_op_levels_in_level_order`.
    pub fn execute(&self) -> Result<()> {
        self.check_hcq_poison()?;
        // One plan has one mutable replay epoch regardless of backend path.
        // Native, graph, direct, and profiled execution share this lock.
        let mut executor = self
            .hcq_executor
            .lock()
            .map_err(|_| crate::error::Error::Execution { reason: "CPU HCQ executor lock poisoned".into() })?;
        let mut graph_replayed = false;
        // Both pre-flight checks run inside the poisoning closure: a failed
        // endpoint validation or native submit leaves the plan's timelines in
        // an unknown state, so it must not stay retryable.
        let result = (|| {
            let graph = self.graph_endpoints_match_device()?.then(|| self.graph()).and_then(|graph| graph.as_deref());
            if graph.is_none()
                && matches!(self.replay_native_linked_plan()?, svod_device::device::NativeReplayOutcome::Executed)
            {
                self.record_completion_token(
                    self.plan_ctx.get().and_then(|context| context.as_deref()).and_then(|c| c.completion_token()),
                );
                return Ok(());
            }
            let linked = self.hcq_linked.get().expect("HCQ plan linked by builder");
            let mut staging = HashMap::new();
            // SAFETY: semantic plan submissions contain only waits, callback
            // execution, and timeline stores; copies are resolved by Buffer.
            unsafe {
                linked.semantic.execute_cpu(&mut executor, |_, command| {
                    if let Some(graph) = graph {
                        if !graph_replayed {
                            let mut buffers = Vec::new();
                            let mut vals = Vec::new();
                            for level in &self.op_levels {
                                for &index in level {
                                    if let PreparedOp::CompiledProgram(kernel) = &self.ops[index] {
                                        for &buffer in &kernel.buffer_indices {
                                            buffers.push(self.buffers[buffer].device_address().context(ExecSnafu {
                                                context: format!(
                                                    "resolve graph replay buffer for kernel {}",
                                                    kernel.id
                                                ),
                                            })?);
                                        }
                                        vals.extend_from_slice(&kernel.vals);
                                    }
                                }
                            }
                            graph.replay(&buffers, &vals).context(ExecSnafu { context: "graph replay" })?;
                            graph_replayed = true;
                        }
                    } else {
                        self.execute_topology_command(command, &mut staging)?;
                    }
                    Ok(())
                })
            }
            .map_err(Self::submission_error)?;
            if let Some(ctx) = self.plan_ctx.get().and_then(|context| context.as_deref()) {
                ctx.finish_replay().context(ExecSnafu { context: "finish direct HCQ replay" })?;
            }
            let token = if graph_replayed {
                graph.and_then(|graph| graph.completion_token())
            } else {
                self.plan_ctx
                    .get()
                    .and_then(|context| context.as_deref())
                    .and_then(|context| context.completion_token())
            };
            self.record_completion_token(token);
            Ok(())
        })();
        if result.is_err()
            && let Some(ctx) = self.plan_ctx.get().and_then(|context| context.as_deref())
        {
            let _ = ctx.finish_replay();
        }
        self.poison_hcq(result)
    }

    /// Execute the plan with per-kernel timing.
    ///
    /// Returns a [`KernelProfile`] for each kernel in execution order.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let plan = tensor.prepare()?;
    /// let profiles = plan.execute_profiled()?;
    ///
    /// // Sort by time descending
    /// let mut sorted = profiles;
    /// sorted.sort_by(|a, b| b.wall.cmp(&a.wall));
    /// for p in &sorted[..10.min(sorted.len())] {
    ///     println!("{:>8.3}ms  {}", p.wall.as_secs_f64() * 1000.0, p.kernel.entry_point);
    /// }
    /// ```
    /// Uses a captured graph's linked profiling variant when the backend exposes
    /// per-dispatch stamps; otherwise falls back to profiled per-call submissions.
    pub fn execute_profiled(&self) -> Result<Vec<KernelProfile>> {
        self.check_hcq_poison()?;
        let mut finalizer = SubmissionProfileFinalizer::with_capacity(self.op_order.len());
        let mut executor = self
            .hcq_executor
            .lock()
            .map_err(|_| crate::error::Error::Execution { reason: "CPU HCQ executor lock poisoned".into() })?;
        let result = (|| {
            let linked = self.hcq_linked.get().expect("HCQ plan linked by builder");

            if let Some(graph) =
                self.graph_endpoints_match_device()?.then(|| self.graph()).and_then(|graph| graph.as_deref())
            {
                let mut buffers = Vec::new();
                let mut vals = Vec::new();
                let mut kernels = Vec::new();
                for &index in self.op_levels.iter().flatten() {
                    if let PreparedOp::CompiledProgram(kernel) = &self.ops[index] {
                        for &buffer in &kernel.buffer_indices {
                            buffers.push(self.buffers[buffer].device_address().context(ExecSnafu {
                                context: format!("resolve profiled graph replay buffer for kernel {}", kernel.id),
                            })?);
                        }
                        vals.extend_from_slice(&kernel.vals);
                        kernels.push(kernel);
                    }
                }
                let start = Instant::now();
                if let Some(handles) =
                    graph.replay_profiled(&buffers, &vals).context(ExecSnafu { context: "profiled graph replay" })?
                {
                    if handles.len() != kernels.len() {
                        return Err(crate::error::Error::Execution {
                            reason: format!(
                                "profiled graph returned {} timestamps for {} kernels",
                                handles.len(),
                                kernels.len()
                            ),
                        });
                    }
                    let wall = start.elapsed();
                    for (kernel, handle) in kernels.into_iter().zip(handles) {
                        finalizer.push(
                            KernelProfile {
                                kernel: Arc::clone(&kernel.kernel),
                                device: kernel.device.clone(),
                                origin: kernel.origin,
                                origins: kernel.origins.clone(),
                                num_buffers: kernel.buffer_ptrs.len(),
                                wall,
                                gpu_start_ns: None,
                                gpu_end_ns: None,
                                static_info: None,
                                counters: None,
                            },
                            Some(handle),
                        );
                    }
                    // Keep neutral lane timelines coherent without replaying the graph.
                    unsafe {
                        linked
                            .semantic
                            .execute_cpu(&mut executor, |_, _| Ok::<_, crate::error::Error>(()))
                            .map_err(Self::submission_error)?;
                    }
                    return finalizer.finish(|| Ok(()));
                }
            }

            let mut staging = HashMap::new();
            unsafe {
                linked.semantic.execute_cpu(&mut executor, |_, command| {
                    if command.copy_leg.is_some() {
                        self.execute_topology_command(command, &mut staging)?;
                    } else {
                        match &self.ops[command.operation] {
                            PreparedOp::CompiledProgram(kernel) => {
                                let start = Instant::now();
                                let handle = self.execute_kernel(kernel, /*profile=*/ true)?;
                                // Stamp before the metadata clones, so the
                                // origin set never lands inside the timed window.
                                let wall = start.elapsed();
                                finalizer.push(
                                    KernelProfile {
                                        kernel: Arc::clone(&kernel.kernel),
                                        device: kernel.device.clone(),
                                        origin: kernel.origin,
                                        origins: kernel.origins.clone(),
                                        num_buffers: kernel.buffer_ptrs.len(),
                                        wall,
                                        gpu_start_ns: None,
                                        gpu_end_ns: None,
                                        static_info: None,
                                        counters: None,
                                    },
                                    handle,
                                );
                            }
                            PreparedOp::BufferCopy(copy) => self.execute_copy(copy)?,
                            PreparedOp::CustomFunction(custom) => self.execute_custom_function(custom)?,
                        }
                    }
                    Ok(())
                })
            }
            .map_err(Self::submission_error)?;
            finalizer.finish(|| {
                if let Some(ctx) = self.plan_ctx.get().and_then(|context| context.as_deref()) {
                    ctx.synchronize().context(ExecSnafu { context: "profiled HCQ finalizer" })?;
                }
                Ok(())
            })
        })();
        if result.is_err()
            && let Some(ctx) = self.plan_ctx.get().and_then(|context| context.as_deref())
        {
            let _ = ctx.finish_replay();
        }
        self.poison_hcq(result)
    }

    /// Profile the plan: run the per-dispatch path `opts.iters.max(1)` times,
    /// keeping each kernel's minimum device time (robust to outliers). Returns
    /// a single-stage [`RunProfile`]; render it with [`RunProfile::render_table`].
    ///
    /// Tier-2/3 static analysis (`opts.static_analysis`) and Tier-4 hardware
    /// counters (`opts.counters`) attach to each [`KernelProfile`] when enabled.
    /// Tier-4 is gated: it requires `pmc_available()` and a stable power state;
    /// otherwise it degrades gracefully to timing-only with a one-line note.
    pub fn profile(&self, opts: &ProfileOptions) -> Result<RunProfile> {
        let start = Instant::now();
        // Tier-4: arm hardware counters on the plan's context when requested and
        // the backend supports it in a stable power state. Degrade gracefully
        // (no counters, a one-line note) rather than failing the run.
        let counters = opts.counters.counters();
        let armed_ctx = if counters.is_empty() {
            None
        } else {
            let first_program = self.op_levels.iter().flatten().find_map(|&idx| match &self.ops[idx] {
                PreparedOp::CompiledProgram(k) => Some(k.kernel.program.as_ref()),
                _ => None,
            });
            match first_program.and_then(|p| self.plan_ctx(p).ok().flatten()) {
                Some(ctx) if ctx.pmc_available() => {
                    ctx.set_pmc(&counters);
                    Some(ctx)
                }
                Some(_) => {
                    eprintln!(
                        "SVOD_PMC: hardware counters unavailable (needs a profile_standard \
                         power state — run `amd-smi set -l stable_std`); reporting timing only"
                    );
                    None
                }
                None => None,
            }
        };
        // Each pass is one "profile" stage; merge passes by per-kernel min time.
        // Match from_env(): zero iterations still means one profiling pass.
        let result: Result<RunProfile> = (|| {
            let run = |kernels| RunProfile {
                stages: vec![StageProfile::gpu("profile", start.elapsed(), kernels)],
                origin_depth: opts.origin_depth,
            };
            let mut report = run(self.execute_profiled()?);
            for _ in 1..opts.iters.max(1) {
                report.merge_min(run(self.execute_profiled()?));
            }
            Ok(report)
        })();
        // Disarm so later non-profiled executions on this context don't pay for
        // (or perturb from) counter programming.
        if let Some(ctx) = armed_ctx {
            ctx.set_pmc(&[]);
        }
        let mut report = result?;
        if opts.static_analysis {
            // Profiles are in dispatch order; the compiled kernels in op_levels
            // order line up one-to-one, so zip attaches each kernel's analysis.
            let kernels = self.op_levels.iter().flatten().filter_map(|&idx| match &self.ops[idx] {
                PreparedOp::CompiledProgram(k) => Some(k),
                _ => None,
            });
            for (profile, pk) in report.stages[0].kernels.iter_mut().zip(kernels) {
                profile.static_info = Some(self.kernel_static_info(pk));
            }
        }
        Ok(report)
    }

    /// Tier-2/3 static analysis for one kernel: AST flop estimate, compulsory
    /// byte traffic (each distinct buffer counted once), and decoded GPU
    /// resources when the backend exposes them.
    fn kernel_static_info(&self, pk: &PreparedKernel) -> KernelStaticInfo {
        // The AST walk saturates to u64::MAX when a range/special has an
        // unbounded symbolic end (common in hand-built kernels) — treat that as
        // "no reliable count" rather than reporting a garbage roofline.
        let raw_flops = svod_ir::compute_ops_estimate(&pk.ast);
        let est_flops = (raw_flops != u64::MAX).then_some(raw_flops);
        let mut seen = std::collections::HashSet::new();
        let est_bytes =
            pk.buffer_indices.iter().filter(|&&i| seen.insert(i)).map(|&i| self.buffers[i].size() as u64).sum();
        let resources = pk.kernel.program.resource_usage();
        KernelStaticInfo { est_flops, est_bytes, resources }
    }

    /// Re-execute the plan with different variable bindings.
    ///
    /// The kernel code is NOT recompiled; only the `vals` passed to each kernel
    /// are updated. Buffers must be allocated to max variable values (which is
    /// the default when using `Variable::bind()`).
    ///
    /// # Safety contract
    ///
    /// Variable values **must** fall within `[min_val, max_val]` bounds defined
    /// at `Variable::new()` time. Exceeding `max_val` causes out-of-bounds buffer
    /// access (buffers are allocated to `max_val`). Use `Variable::bind()` to
    /// validate bounds before calling this method.
    ///
    /// Variables not present in `var_vals` keep their existing values from
    /// `prepare()` (or the previous `execute_with_vars` call). Internal
    /// variables like `core_id` are left untouched.
    pub fn execute_with_vars(&mut self, var_vals: &[(&str, i64)]) -> Result<()> {
        self.update_runtime_var_vals(var_vals)?;
        self.execute()
    }

    /// Re-execute the plan with different variable bindings and per-kernel timing.
    ///
    /// Updates kernel `vals` the same way as [`Self::execute_with_vars`] and then
    /// executes via [`Self::execute_profiled`].
    pub fn execute_with_vars_profiled(&mut self, var_vals: &[(&str, i64)]) -> Result<Vec<KernelProfile>> {
        self.update_runtime_var_vals(var_vals)?;
        self.execute_profiled()
    }

    /// Get the first output buffer index, or `None` for an output-less plan
    /// (mirrors [`Self::output_buffer`], which also returns `Option`).
    pub fn output_buffer_idx(&self) -> Option<usize> {
        self.output_buffer_indices.first().copied()
    }

    /// Get the AST ID to buffer index mapping.
    pub fn ast_to_buffer_map(&self) -> &HashMap<u64, usize> {
        &self.ast_to_buffer
    }

    /// Storage ids this plan writes: kernel outputs, copy destinations,
    /// custom-function arguments (no outs list — conservatively all written,
    /// mirroring `hcq_operations`), and the plan's declared outputs.
    fn written_storage_ids(&self) -> HashSet<BufferId> {
        let storage = |index: &usize| self.buffers.get(*index).map(Buffer::storage_id);
        let mut written: HashSet<BufferId> = self.output_buffer_indices.iter().filter_map(storage).collect();
        for op in &self.ops {
            match op {
                PreparedOp::CompiledProgram(kernel) => written.extend(
                    kernel
                        .output_indices
                        .iter()
                        .filter_map(|&position| kernel.buffer_indices.get(position))
                        .filter_map(storage),
                ),
                PreparedOp::BufferCopy(copy) => written.extend(copy.buffer_indices.first().and_then(storage)),
                PreparedOp::CustomFunction(custom) => {
                    written.extend(custom.buffer_indices.iter().filter_map(storage));
                }
            }
        }
        written
    }

    /// Declare the buffer at `index` a host-written input: [`Self::replicate`]
    /// forks its storage with a snapshot instead of sharing it. The plan's
    /// write analysis only sees kernel/copy/custom-function writes — host
    /// writes between executes (`copyin`, recurrent-state recycling) are
    /// invisible, so the embedder performing them must declare them.
    /// Idempotent; declarations carry over to replicas.
    pub fn declare_input(&mut self, index: usize) -> Result<()> {
        if index >= self.buffers.len() {
            return Err(crate::error::Error::Execution {
                reason: format!("declare_input: buffer index {index} out of range ({} buffers)", self.buffers.len()),
            });
        }
        self.input_buffer_indices.insert(index);
        Ok(())
    }

    /// Scoped-sync: publish this epoch's completion token on every distinct
    /// storage the plan touches (reads AND writes — a host overwrite races
    /// in-flight readers too), so host access waits only this plan's work
    /// instead of draining the whole device. No-op on backends without
    /// tokens (CPU) and on plans whose context was never minted.
    fn record_completion_token(&self, token: Option<std::sync::Arc<dyn svod_device::CompletionToken>>) {
        let Some(token) = token else { return };
        for &index in &self.distinct_storage_indices {
            self.buffers[index].record_completion(&token);
        }
        token.published();
    }

    /// Deep-copy the plan for concurrent execution. Fork policy per storage:
    ///
    /// - written by the plan (arena intermediates, outputs, copy
    ///   destinations): forked *bare* — contents are re-derived on every
    ///   execute, so a replica's outputs are meaningful only after it runs;
    /// - declared via [`Self::declare_input`]: forked with a byte-exact
    ///   snapshot of the current contents;
    /// - everything else — model weights — shared with the original.
    ///
    /// Views are re-minted at their original offsets on one forked base per
    /// storage, preserving arena aliasing. Compiled kernels are `Arc`-shared.
    /// All lazy per-plan state (backend queue context, captured graph, HCQ
    /// timelines) starts fresh, so the replica executes concurrently with the
    /// original on its own queue. Replicate only while the plan is idle:
    /// snapshotting reads buffer contents without synchronizing against
    /// in-flight kernels.
    pub fn replicate(&self) -> Result<ExecutionPlan> {
        let snapshot: HashSet<BufferId> = self
            .input_buffer_indices
            .iter()
            .filter_map(|&index| self.buffers.get(index).map(Buffer::storage_id))
            .collect();
        let mut fork = self.written_storage_ids();
        fork.extend(snapshot.iter().copied());

        // Group forked views per storage: every view of one allocation lands
        // on ONE fresh base at its old offset, so arena aliasing survives.
        let mut grouped: HashMap<BufferId, Vec<usize>> = HashMap::new();
        for (index, buffer) in self.buffers.iter().enumerate() {
            let storage = buffer.storage_id();
            if fork.contains(&storage) {
                grouped.entry(storage).or_default().push(index);
            }
        }
        let mut buffers = self.buffers.clone();
        for (storage, indices) in grouped {
            let views: Vec<&Buffer> = indices.iter().map(|&index| &self.buffers[index]).collect();
            let forked = Buffer::fork_views(&views, snapshot.contains(&storage))
                .context(ExecSnafu { context: "fork replica storage" })?;
            for (index, buffer) in indices.into_iter().zip(forked) {
                buffers[index] = buffer;
            }
        }

        // `build()` re-resolves buffer addresses/ids, recomputes the op order
        // and levels, captures fresh HCQ timelines, and leaves the graph and
        // plan context lazily unset — exactly the per-replica state.
        let builder = ExecutionPlanBuilder {
            ops: self.ops.clone(),
            op_instance_dependencies: self.op_instance_dependencies.clone(),
            buffers,
            ast_to_buffer: self.ast_to_buffer.clone(),
            output_buffer_indices: self.output_buffer_indices.clone(),
            device: self.device.clone(),
        };
        let mut plan = builder.build()?;
        plan.runtime_var_vals = self.runtime_var_vals.clone();
        plan.input_buffer_indices = self.input_buffer_indices.clone();
        Ok(plan)
    }
}

// No explicit `Drop for ExecutionPlan`: the plan's `plan_ctx`
// (`OnceLock<Option<Box<dyn PlanContext>>>`) just holds an `Arc` over a
// backend-shared queue/context. On plan drop the `Arc` decrements; the
// underlying queue stays in the backend's pool (freed only at device close), so
// plan churn never tears down backend queues.

impl std::fmt::Debug for ExecutionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kernel_count = self.ops.iter().filter(|op| matches!(op, PreparedOp::CompiledProgram(_))).count();
        f.debug_struct("ExecutionPlan")
            .field("ops", &self.ops.len())
            .field("op_instance_dependencies", &self.op_instance_dependencies.len())
            .field("op_order", &self.op_order.len())
            .field("kernels", &kernel_count)
            .field("buffers", &self.buffers.len())
            .field("hcq_lanes", &self.hcq_linked.get().map(|linked| linked.semantic.lanes().len()))
            .field("device", &self.device)
            .finish()
    }
}

impl std::fmt::Debug for PreparedKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedKernel")
            .field("id", &self.id)
            .field("device", &self.device)
            .field("buffer_indices", &self.buffer_indices)
            .field("output_indices", &self.output_indices)
            .field("input_indices", &self.input_indices)
            .field("vals", &self.vals)
            .field("fixedvars", &self.fixedvars)
            .field("dependencies", &self.dependencies)
            .field("origins", &self.origins)
            .finish()
    }
}

// ============================================================================
// Builder for ExecutionPlan
// ============================================================================

/// Builder for creating ExecutionPlan from schedule data.
pub struct ExecutionPlanBuilder {
    ops: Vec<PreparedOp>,
    op_instance_dependencies: Vec<Vec<usize>>,
    buffers: Vec<Buffer>,
    ast_to_buffer: HashMap<u64, usize>,
    output_buffer_indices: Vec<usize>,
    device: DeviceSpec,
}

impl ExecutionPlanBuilder {
    /// Create a new builder.
    pub fn new(device: DeviceSpec) -> Self {
        Self {
            ops: Vec::new(),
            op_instance_dependencies: Vec::new(),
            buffers: Vec::new(),
            ast_to_buffer: HashMap::new(),
            output_buffer_indices: Vec::new(),
            device,
        }
    }

    /// Add a buffer to the plan. Returns the buffer index.
    pub fn add_buffer(&mut self, ast_id: u64, buffer: Buffer) -> usize {
        let idx = self.buffers.len();
        self.buffers.push(buffer);
        self.ast_to_buffer.insert(ast_id, idx);
        idx
    }

    /// Map an additional AST/buffer UOp ID to an existing buffer index.
    pub fn map_buffer(&mut self, ast_id: u64, idx: usize) {
        self.ast_to_buffer.insert(ast_id, idx);
    }

    /// Set single output buffer index.
    pub fn set_output_buffer(&mut self, idx: usize) {
        self.output_buffer_indices = vec![idx];
    }

    /// Set multiple output buffer indices (batch scheduling).
    pub fn set_output_buffers(&mut self, indices: Vec<usize>) {
        self.output_buffer_indices = indices;
    }

    /// Compatibility helper: add a compiled kernel as a prepared operation.
    ///
    /// The canonical builder path is `add_op(PreparedOp::...)`.
    pub fn add_kernel(&mut self, kernel: PreparedKernel) {
        self.add_op(PreparedOp::CompiledProgram(kernel));
    }

    /// Add a prepared operation in schedule order.
    pub fn add_op(&mut self, op: PreparedOp) {
        self.add_op_with_instance_dependencies(op, Vec::new());
    }

    /// Add a prepared operation with concrete op-index dependencies.
    pub fn add_op_with_instance_dependencies(&mut self, op: PreparedOp, instance_dependencies: Vec<usize>) {
        self.ops.push(op);
        self.op_instance_dependencies.push(instance_dependencies);
    }

    /// Number of prepared ops added so far. Callers use this to assert 1:1
    /// emission against their source schedule.
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Build the ExecutionPlan.
    ///
    /// Finalizes by computing pre-allocated buffer pointers and buffer IDs
    /// for zero-allocation execution.
    pub fn build(mut self) -> Result<ExecutionPlan> {
        for op in &mut self.ops {
            let kernel = match op {
                PreparedOp::CompiledProgram(kernel) => kernel,
                // `execute_copy`, `copy_buffers` and `hcq_operations` all need
                // the (dst, src) pair; a shorter copy degrades to a no-op edge
                // in the hazard graph. Reject it here instead of at execute.
                PreparedOp::BufferCopy(copy) if copy.buffer_indices.len() < 2 => {
                    return Err(crate::error::Error::Execution {
                        reason: format!(
                            "BufferCopy {} requires two buffer indices (dst, src), got {}",
                            copy.id,
                            copy.buffer_indices.len()
                        ),
                    });
                }
                PreparedOp::BufferCopy(_) | PreparedOp::CustomFunction(_) => continue,
            };

            if kernel.output_indices.is_empty() {
                return Err(crate::error::Error::Execution {
                    reason: format!("CompiledProgram {} has no output indices", kernel.id),
                });
            }
            for &out_idx in &kernel.output_indices {
                if out_idx >= kernel.buffer_indices.len() {
                    return Err(crate::error::Error::Execution {
                        reason: format!(
                            "CompiledProgram {} output index out of range: output_idx={}, kernel_buffers={}",
                            kernel.id,
                            out_idx,
                            kernel.buffer_indices.len()
                        ),
                    });
                }
            }
            for &input_idx in &kernel.input_indices {
                if input_idx >= kernel.buffer_indices.len() {
                    return Err(crate::error::Error::Execution {
                        reason: format!(
                            "CompiledProgram {} input index out of range: input_idx={}, kernel_buffers={}",
                            kernel.id,
                            input_idx,
                            kernel.buffer_indices.len()
                        ),
                    });
                }
            }

            let mut buffer_ptrs = Vec::with_capacity(kernel.buffer_indices.len());
            let mut buffer_ids = Vec::with_capacity(kernel.buffer_indices.len());

            for &idx in &kernel.buffer_indices {
                let Some(buffer) = self.buffers.get(idx) else {
                    return Err(crate::error::Error::Execution {
                        reason: format!(
                            "CompiledProgram {} buffer index out of range: idx={}, total_buffers={}",
                            kernel.id,
                            idx,
                            self.buffers.len()
                        ),
                    });
                };
                // Resolve GETADDR on the host after allocation, at the same
                // PROGRAM -> ExecutionPlan boundary where globals are ordered.
                buffer_ptrs.push(buffer.device_address().context(ExecSnafu {
                    context: format!("resolve HCQ GETADDR for kernel {} buffer {}", kernel.id, idx),
                })? as usize);
                buffer_ids.push(buffer.id());
            }

            kernel.buffer_ptrs = buffer_ptrs;
            kernel.buffer_ids = buffer_ids;
        }

        if self.output_buffer_indices.is_empty() && !self.buffers.is_empty() {
            return Err(crate::error::Error::Execution {
                reason: "execution plan output buffers must be set explicitly".to_string(),
            });
        }

        // One representative buffer index per distinct storage, for scoped-sync
        // token recording on execute.
        let mut seen_storages = HashSet::new();
        let distinct_storage_indices: Vec<usize> = self
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, buffer)| seen_storages.insert(buffer.storage_id()))
            .map(|(index, _)| index)
            .collect();

        let op_order = compute_mixed_op_order_with_instance_dependencies(&self.ops, &self.op_instance_dependencies)?;
        let op_levels = compute_execution_levels_with_instance_dependencies(&self.ops, &self.op_instance_dependencies)?;

        let plan = ExecutionPlan {
            ops: self.ops,
            op_instance_dependencies: self.op_instance_dependencies,
            op_order,
            op_levels,
            buffers: self.buffers,
            ast_to_buffer: self.ast_to_buffer,
            output_buffer_indices: self.output_buffer_indices,
            input_buffer_indices: HashSet::new(),
            distinct_storage_indices,
            device: self.device,
            runtime_var_vals: HashMap::new(),
            graph: std::sync::OnceLock::new(),
            plan_ctx: std::sync::OnceLock::new(),
            hcq_executor: Mutex::new(svod_device::hcq::CpuQueueExecutor::default()),
            hcq_linked: std::sync::OnceLock::new(),
            hcq_poison: std::sync::OnceLock::new(),
        };
        let linked = HcqLinkedPlan::capture(plan.hcq_operations()?)?;
        plan.hcq_linked.set(linked).map_err(|_| crate::error::Error::Execution {
            reason: "HCQ plan linked twice during preparation".into(),
        })?;
        Ok(plan)
    }
}

#[cfg(test)]
#[path = "test/unit/execution_plan.rs"]
mod tests;
