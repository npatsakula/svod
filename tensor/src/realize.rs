//! Tensor realization (execution) API.
//!
//! This module provides the execution pipeline for tensor operations:
//! 1. **Rangeify** - Transform movement ops to STAGE + INDEX
//! 2. **Kernel splitting** - Split at STORE boundaries into CALL wrappers
//! 3. **Scheduling** - Extract callables and create execution schedule
//! 4. **Execution** - Compile and run each kernel in dependency order
//!
//! Runtime plan execution is dependency-ordered with conservative mixed-op
//! barriers and hazard-aware host parallelism for safe compiled kernels.
//!
//! # ExecutionPlan (Pre-compiled Execution)
//!
//! For repeated executions, use `Tensor::prepare()` to create an `ExecutionPlan`
//! that pre-compiles all kernels and allocates all buffers. Then call
//! `plan.execute()` for fast repeated execution without recompilation overhead.
//!
//! ```ignore
//! // One-time preparation (compiles kernels, allocates buffers)
//! let plan = tensor.prepare()?;
//!
//! // Fast execution (can be called many times)
//! plan.execute()?;
//!
//! // Get results
//! let output = plan.output_buffer();
//! ```
//!
//! To realize a tensor while also collecting per-kernel profiling data, use
//! [`Tensor::profile`](crate::Tensor::profile) instead of
//! [`Tensor::realize`](crate::Tensor::realize) /
//! [`Tensor::prepare`](crate::Tensor::prepare).

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use rayon::prelude::*;
use svod_schedule::optimizer::beam::{CompiledCandidate, beam_search_cached_remote};
use svod_schedule::{KernelNaming, apply_post_optimization_with_config, finalize_kernel_name, prepare_scheduler};
use tracing::{debug, trace};

use crate::{
    PrepareConfig, Result, Tensor,
    error::{
        BatchOutputMismatchSnafu, CompileKernelSnafu, CreateProgramSnafu, DeviceSnafu, EmptyScheduleSnafu,
        ExecutionSnafu, IrConstructionSnafu, KernelGraphSnafu, OptimizeSnafu, RangeifySnafu, RenderKernelSnafu,
        ShapeUnknownSnafu, UOpSnafu,
    },
    schedule::ScheduleItem,
};
use snafu::{OptionExt, ResultExt};
use std::sync::Arc;
use std::time::Duration;
use svod_device::{Buffer, device::Device};
use svod_ir::ops;
use svod_ir::pattern::is_any_const;
use svod_ir::{DeviceSpec, Op, UOp, UOpKey};
use svod_runtime::kernel_cache::CachedKernel;
use svod_runtime::{
    ExecutionPlan, ExecutionPlanBuilder, PreparedCopy, PreparedCustomFunction, PreparedKernel, PreparedOp,
    ProfileOptions, RunProfile,
};

fn collect_pending_indices(tensors: &[&mut Tensor]) -> Vec<usize> {
    tensors
        .iter()
        .enumerate()
        .filter(|(_, t)| !t.uop().has_buffer_identity() && !is_any_const(&t.uop()) && !t.has_zero_elements())
        .map(|(i, _)| i)
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BufferStorageKey {
    id: u64,
    offset: usize,
    size: usize,
    dtype: svod_dtype::DType,
}

impl Tensor {
    /// Realize (execute) this tensor's computation graph.
    ///
    /// This is a convenience method that prepares and executes in one call.
    /// For repeated executions of the same computation, use `prepare()` instead.
    ///
    /// # Pipeline
    ///
    /// 1. **Prepare**: Creates an `ExecutionPlan` (compiles kernels, allocates buffers)
    /// 2. **Execute**: Runs all kernels in dependency order
    /// 3. **Return**: Links output buffer to this tensor's UOp
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
    /// let c = (&a + &b).realize()?;
    /// // c's buffer now contains [5.0, 7.0, 9.0]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if preparation or execution fails.
    /// A concurrent realize of a value-identical tensor can rewrite this
    /// tensor's graph to its realized BUFFER (`apply_map_to_tensors`
    /// broadcast) between the entry check and scheduling — the pipeline then
    /// sees an empty schedule. That is success, not failure: adopt the buffer.
    fn adopt_concurrently_realized(&mut self, error: crate::error::Error) -> Result<()> {
        if matches!(error, crate::error::Error::NoKernelsFound) && self.uop().has_buffer_identity() {
            self.ensure_buffer();
            return Ok(());
        }
        Err(error)
    }

    pub fn realize(&mut self) -> Result<()> {
        if self.uop().has_buffer_identity() {
            self.ensure_buffer();
            return Ok(());
        }
        // Pure constants: wrap in CONTIGUOUS to force materialization into a buffer.
        if is_any_const(&self.uop()) {
            let contiguous_uop = self.uop().contiguous();
            self.set_uop(contiguous_uop);
        }
        if self.has_zero_elements() {
            return Ok(());
        }

        let old_uop = self.uop();

        let t_prep = std::time::Instant::now();
        let plan = match self.prepare_plan_with(&PrepareConfig::from_env()) {
            Ok(plan) => plan,
            Err(error) => return self.adopt_concurrently_realized(error),
        };
        let prep_ms = t_prep.elapsed().as_millis();
        let t_exec = std::time::Instant::now();
        plan.execute().context(ExecutionSnafu)?;
        let exec_ms = t_exec.elapsed().as_millis();
        debug!(prep_ms, exec_ms, "realize complete");

        self.finalize_realize(&plan, &old_uop)?;

        let realized_uop = self.uop();
        if !Arc::ptr_eq(&old_uop, &realized_uop) {
            let becomes_map = HashMap::from([(UOpKey(old_uop), realized_uop)]);
            crate::tensor_registry::apply_map_to_tensors_realized(&becomes_map);
        }

        Ok(())
    }

    /// Realize tensor with custom configuration.
    ///
    /// Like [`realize()`](Self::realize) but allows specifying optimization strategy
    /// and codegen backend.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use svod_tensor::PrepareConfig;
    /// use svod_schedule::{OptStrategy, OptimizerConfig};
    ///
    /// let c = a.matmul(&b)?;
    /// let config = PrepareConfig::from(
    ///     OptimizerConfig::builder()
    ///         .strategy(OptStrategy::Beam { width: 4 })
    ///         .build()
    /// );
    /// let c = c.realize_with(&config)?;
    /// ```
    pub fn realize_with(&mut self, config: &PrepareConfig) -> Result<()> {
        if self.uop().has_buffer_identity() {
            self.ensure_buffer();
            return Ok(());
        }
        // Pure constants: wrap in CONTIGUOUS to force materialization into a buffer.
        if is_any_const(&self.uop()) {
            let contiguous_uop = self.uop().contiguous();
            self.set_uop(contiguous_uop);
        }
        if self.has_zero_elements() {
            return Ok(());
        }

        let old_uop = self.uop();

        let t_prep = std::time::Instant::now();
        let plan = match self.prepare_plan_with(config) {
            Ok(plan) => plan,
            Err(error) => return self.adopt_concurrently_realized(error),
        };
        let prep_ms = t_prep.elapsed().as_millis();
        let t_exec = std::time::Instant::now();
        plan.execute().context(ExecutionSnafu)?;
        let exec_ms = t_exec.elapsed().as_millis();
        debug!(prep_ms, exec_ms, "realize_with complete");

        self.finalize_realize(&plan, &old_uop)?;

        let realized_uop = self.uop();
        if !Arc::ptr_eq(&old_uop, &realized_uop) {
            let becomes_map = HashMap::from([(UOpKey(old_uop), realized_uop)]);
            crate::tensor_registry::apply_map_to_tensors_realized(&becomes_map);
        }

        Ok(())
    }

    /// Profile this tensor's execution.
    ///
    /// Prepares the plan, runs the profiled path per `opts` (replays for stable
    /// device times, plus Tier-2/3 static analysis and Tier-4 counters when
    /// enabled), finalizes the result so the tensor is realized like
    /// [`realize`](Self::realize), and returns the per-kernel [`RunProfile`].
    /// Render it with [`RunProfile::render_table`].
    pub fn profile(&mut self, opts: &ProfileOptions) -> Result<RunProfile> {
        // Nothing to dispatch for already-buffer / const / empty tensors.
        if self.uop().has_buffer_identity() {
            self.ensure_buffer();
            return Ok(RunProfile::default());
        }
        if is_any_const(&self.uop()) {
            let contiguous_uop = self.uop().contiguous();
            self.set_uop(contiguous_uop);
        }
        if self.has_zero_elements() {
            return Ok(RunProfile::default());
        }

        let old_uop = self.uop();

        let plan = self.prepare_plan_with(&PrepareConfig::from_env())?;
        let report = plan.profile(opts).context(ExecutionSnafu)?;

        self.finalize_realize(&plan, &old_uop)?;
        let realized_uop = self.uop();
        if !Arc::ptr_eq(&old_uop, &realized_uop) {
            let becomes_map = HashMap::from([(UOpKey(old_uop), realized_uop)]);
            crate::tensor_registry::apply_map_to_tensors_realized(&becomes_map);
        }
        Ok(report)
    }

    /// Finalize realization: bind output buffer to tensor.
    ///
    /// Note: intermediate buffer cleanup is deferred to `realize()` so it
    /// runs AFTER `apply_map_to_tensors`. This ensures other tensors can still
    /// find buffers during the substitution window.
    fn finalize_realize(&mut self, plan: &ExecutionPlan, uop: &Arc<UOp>) -> Result<()> {
        let output_buf = plan.output_buffer().expect("realized plan must have an output buffer").clone();

        trace!(
            buffer.id = ?output_buf.id(),
            buffer.size = output_buf.size(),
            "Realized output buffer"
        );

        let output_dtype = uop.dtype();
        let output_device = output_buf.allocator().device_spec();
        let num_elements = output_buf.size() / output_dtype.bytes();

        let buffer_uop = UOp::new_buffer(output_device, num_elements, output_dtype.clone());
        let output_buf_arc = Arc::new(output_buf);

        crate::tensor_registry::register_buffer(buffer_uop.id, self.entry.id, output_buf_arc.clone());

        let shape = uop.shape().context(UOpSnafu)?.context(ShapeUnknownSnafu)?;
        let realized_uop = buffer_uop.try_reshape(shape).context(UOpSnafu)?;

        debug!(
            buffer_uop.id = buffer_uop.id,
            num_elements,
            shape = ?shape,
            realized_uop.id = realized_uop.id,
            realized_uop.base_id = realized_uop.base().id,
            "Tensor realized"
        );

        self.set_uop(realized_uop);
        self.entry.set_buffer(Arc::clone(&output_buf_arc));
        self.buffer = Some(output_buf_arc);
        Ok(())
    }

    /// Prepare an execution plan for this tensor's computation graph.
    ///
    /// This performs all one-time work:
    /// 1. Creates schedule from computation graph
    /// 2. Instantiates strict range-expanded callable schedule items
    /// 3. Compiles all kernels
    /// 4. Allocates all buffers
    /// 5. Builds dependency-ordered prepared op execution plan
    ///
    /// The returned `ExecutionPlan` can then be executed multiple times
    /// without recompilation overhead.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
    /// let mut c = &a + &b;
    ///
    /// // One-time preparation (wires output tensor to plan buffer)
    /// let plan = c.prepare()?;
    ///
    /// // Fast execution (can be called many times)
    /// plan.execute()?;
    ///
    /// // Get results
    /// let output = plan.output_buffer();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Rangeify transformation fails
    /// - No kernels found after scheduling
    /// - Kernel compilation fails
    /// - Buffer allocation fails
    pub fn prepare(&mut self) -> Result<ExecutionPlan> {
        self.prepare_with(&PrepareConfig::from_env())
    }

    /// Prepare an execution plan with explicit configuration.
    ///
    /// This method allows fine-grained control over kernel optimization settings
    /// and codegen backend selection.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use svod_tensor::PrepareConfig;
    /// use svod_schedule::{OptimizerConfig, OptStrategy, BeamConfig};
    ///
    /// // Beam search with width 8 and 120s timeout
    /// let config = PrepareConfig::from(
    ///     OptimizerConfig::builder()
    ///         .strategy(OptStrategy::Beam { width: 8 })
    ///         .beam(BeamConfig::builder()
    ///             .timeout_secs(120)
    ///             .build())
    ///         .build()
    /// );
    ///
    /// let plan = tensor.prepare_with(&config)?;
    /// plan.execute()?;
    /// ```
    pub fn prepare_with(&mut self, config: &PrepareConfig) -> Result<ExecutionPlan> {
        let uop = self.uop();
        let plan = self.prepare_plan_with(config)?;

        self.wire_output_tensor(&plan, &uop)?;
        Ok(plan)
    }

    fn prepare_plan_with(&self, config: &PrepareConfig) -> Result<ExecutionPlan> {
        let t_total = std::time::Instant::now();
        let uop = self.uop();

        let sink = UOp::sink(vec![uop.contiguous()]);
        let schedule_result = schedule_result_from_sink_with_cache(sink, extract_var_vals(&uop)?, config)?;
        // Per-kernel optimization+compilation is cached globally in prepare_execution_plan
        // via OPT_CACHE keyed by content_hash(ast). Identical kernel ASTs across calls
        // (e.g., sort substages, repeated model inference) skip optimize+compile.
        let plan = prepare_execution_plan(&schedule_result, config)?;

        debug!(total_ms = t_total.elapsed().as_millis() as u64, "prepare: total");
        Ok(plan)
    }

    fn wire_output_tensor(&mut self, plan: &ExecutionPlan, uop: &Arc<UOp>) -> Result<()> {
        if plan.num_outputs() > 0 {
            let buf = Arc::new(plan.output_buffer().expect("plan with num_outputs > 0 must expose output").clone());
            let dtype = uop.dtype();
            let device = buf.allocator().device_spec();
            let buffer_uop = UOp::new_buffer(device, buf.size() / dtype.bytes(), dtype);
            crate::tensor_registry::register_buffer(buffer_uop.id, self.entry.id, buf.clone());
            let shape = uop.shape().context(UOpSnafu)?.context(ShapeUnknownSnafu)?;
            self.set_uop(buffer_uop.try_reshape(shape).context(UOpSnafu)?);
            self.entry.set_buffer(buf.clone());
            self.buffer = Some(buf);
        }
        Ok(())
    }

    // =========================================================================
    // Batch realize / prepare
    // =========================================================================

    /// Realize multiple tensors in a single batch, sharing computation.
    ///
    /// Merges all tensor computation graphs into one SINK, enabling the scheduler
    /// to share kernels across outputs. More efficient than calling `realize()`
    /// individually when tensors share subgraphs.
    pub fn realize_batch<'a>(tensors: impl IntoIterator<Item = &'a mut Tensor>) -> Result<()> {
        Self::realize_batch_with(tensors, &PrepareConfig::from_env())
    }

    /// Realize multiple tensors with custom configuration.
    pub fn realize_batch_with<'a>(
        tensors: impl IntoIterator<Item = &'a mut Tensor>,
        config: &PrepareConfig,
    ) -> Result<()> {
        let mut tensors: Vec<&mut Tensor> = tensors.into_iter().collect();
        if tensors.is_empty() {
            return Ok(());
        }

        // Handle already-realized tensors
        for t in &mut tensors {
            if t.uop().has_buffer_identity() {
                t.ensure_buffer();
            }
        }

        // Wrap pure constants in CONTIGUOUS to force materialization (matches realize())
        for t in &mut tensors {
            if !t.uop().has_buffer_identity() && is_any_const(&t.uop()) {
                let contiguous_uop = t.uop().contiguous();
                t.set_uop(contiguous_uop);
            }
        }

        // Collect pending (unrealized) tensor indices
        let pending_indices = collect_pending_indices(&tensors);

        if pending_indices.is_empty() {
            return Ok(());
        }

        let old_uops: Vec<Arc<UOp>> = pending_indices.iter().map(|&i| tensors[i].uop()).collect();

        // Create merged SINK(CONTIGUOUS(t1), ..., CONTIGUOUS(tN))
        let contiguouses: Vec<Arc<UOp>> = old_uops.iter().map(|u| u.contiguous()).collect();
        let sink = UOp::sink(contiguouses);

        let mut var_vals = HashMap::new();
        for uop in &old_uops {
            let extracted = extract_var_vals(uop)?;
            merge_var_vals_checked(&mut var_vals, &extracted, "realize_batch input collection")?;
        }
        let schedule_result = schedule_result_from_sink_with_cache(sink, var_vals, config)?;

        let t_prep = std::time::Instant::now();
        let plan = prepare_execution_plan(&schedule_result, config)?;
        let prep_ms = t_prep.elapsed().as_millis();
        snafu::ensure!(
            plan.num_outputs() == pending_indices.len(),
            BatchOutputMismatchSnafu { expected: pending_indices.len(), actual: plan.num_outputs() }
        );
        let t_exec = std::time::Instant::now();
        plan.execute().context(ExecutionSnafu)?;
        let exec_ms = t_exec.elapsed().as_millis();
        debug!(prep_ms, exec_ms, num_outputs = pending_indices.len(), "realize_batch complete");

        // Finalize each pending tensor in-place + build batched becomes_map
        let mut becomes_map = HashMap::new();
        for (buf_idx, &orig_idx) in pending_indices.iter().enumerate() {
            let output_buf = plan.output_buffer_at(buf_idx).expect("buf_idx in range").clone();
            let old_uop = &old_uops[buf_idx];

            let output_dtype = old_uop.dtype();
            let output_device = output_buf.allocator().device_spec();
            let num_elements = output_buf.size() / output_dtype.bytes();
            let buffer_uop = UOp::new_buffer(output_device, num_elements, output_dtype);
            let buf_arc = Arc::new(output_buf);

            let t = &mut tensors[orig_idx];
            crate::tensor_registry::register_buffer(buffer_uop.id, t.entry.id, buf_arc.clone());
            let shape = old_uop.shape().context(UOpSnafu)?.context(ShapeUnknownSnafu)?;
            let realized_uop = buffer_uop.try_reshape(shape).context(UOpSnafu)?;
            t.set_uop(realized_uop.clone());
            t.entry.set_buffer(Arc::clone(&buf_arc));
            t.buffer = Some(buf_arc);

            becomes_map.insert(UOpKey(old_uop.clone()), realized_uop);
        }

        // Single batched apply_map (one global walk instead of N)
        crate::tensor_registry::apply_map_to_tensors_realized(&becomes_map);

        Ok(())
    }

    /// Prepare a batch execution plan for multiple tensors.
    ///
    /// Output tensors are wired to plan buffers — after `execute`/`execute_with_vars`,
    /// results are readable directly via `tensor.as_vec()` or `tensor.array_view()`.
    pub fn prepare_batch<'a>(tensors: impl IntoIterator<Item = &'a mut Tensor>) -> Result<ExecutionPlan> {
        Self::prepare_batch_with(tensors, &PrepareConfig::from_env())
    }

    /// Prepare a batch execution plan with custom configuration.
    pub fn prepare_batch_with<'a>(
        tensors: impl IntoIterator<Item = &'a mut Tensor>,
        config: &PrepareConfig,
    ) -> Result<ExecutionPlan> {
        let mut tensors: Vec<&mut Tensor> = tensors.into_iter().collect();
        if tensors.is_empty() {
            return EmptyScheduleSnafu.fail();
        }

        // Keep tensor state unchanged until the complete plan has been validated.
        let pending_indices: Vec<usize> = tensors
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.uop().has_buffer_identity() && !t.has_zero_elements())
            .map(|(i, _)| i)
            .collect();

        if pending_indices.is_empty() {
            return EmptyScheduleSnafu.fail();
        }

        // Collect UOps from pending tensors only
        let uops: Vec<Arc<UOp>> = pending_indices
            .iter()
            .map(|&i| {
                let uop = tensors[i].uop();
                if is_any_const(&uop) { uop.contiguous() } else { uop }
            })
            .collect();

        let mut var_vals = HashMap::new();
        for uop in &uops {
            let extracted = extract_var_vals(uop)?;
            merge_var_vals_checked(&mut var_vals, &extracted, "prepare_batch input collection")?;
        }

        // Create merged SINK(CONTIGUOUS(t1), ..., CONTIGUOUS(tN)) from pending tensors
        let contiguouses: Vec<Arc<UOp>> = uops.iter().map(|u| u.contiguous()).collect();
        let sink = UOp::sink(contiguouses);

        let schedule_result = schedule_result_from_sink_with_cache(sink, var_vals, config)?;

        let plan = prepare_execution_plan(&schedule_result, config)?;

        snafu::ensure!(
            plan.num_outputs() == pending_indices.len(),
            BatchOutputMismatchSnafu { expected: pending_indices.len(), actual: plan.num_outputs() }
        );

        for t in &mut tensors {
            if t.uop().has_buffer_identity() {
                t.ensure_buffer();
            }
        }

        // Wire each pending output tensor to its plan buffer.
        // After execute/execute_with_vars, tensor.array_view() reads the result directly.
        for (buf_idx, &orig_idx) in pending_indices.iter().enumerate() {
            let output_buf = plan.output_buffer_at(buf_idx).expect("buf_idx in range").clone();
            let buf_arc = Arc::new(output_buf);
            let old_uop = &uops[buf_idx];
            let output_dtype = old_uop.dtype();
            let output_device = buf_arc.allocator().device_spec();
            let num_elements = buf_arc.size() / output_dtype.bytes();
            let buffer_uop = UOp::new_buffer(output_device, num_elements, output_dtype);
            let t = &mut tensors[orig_idx];
            crate::tensor_registry::register_buffer(buffer_uop.id, t.entry.id, buf_arc.clone());
            let shape = old_uop.shape().context(UOpSnafu)?.context(ShapeUnknownSnafu)?;
            let realized_uop = buffer_uop.try_reshape(shape).context(UOpSnafu)?;
            t.set_uop(realized_uop);
            t.entry.set_buffer(Arc::clone(&buf_arc));
            t.buffer = Some(buf_arc);
        }

        Ok(plan)
    }
}

/// Extract bound variable values from a UOp graph (pre-pipeline).
///
/// Scans for BIND(DEFINE_VAR, CONST) nodes and extracts the mapping from
/// variable name to concrete bound value. These values are passed through to
/// scheduling so that user Variables (like `Variable::new("N", 1, 32).bind(4)`)
/// are treated as fixed parameters rather than schedule-loop ranges to expand.
/// Insert `(name, val)` into `var_vals` if not present, otherwise check that
/// any existing binding agrees. Returns `Err((prev, val))` on mismatch so
/// callers can format the error in their own context.
fn try_bind_var_val(var_vals: &mut HashMap<String, i64>, name: &str, val: i64) -> std::result::Result<(), (i64, i64)> {
    if let Some(&prev) = var_vals.get(name) {
        if prev != val {
            return Err((prev, val));
        }
        return Ok(());
    }
    var_vals.insert(name.to_string(), val);
    Ok(())
}

fn insert_var_val_checked(var_vals: &mut HashMap<String, i64>, name: &str, val: i64, context: &str) -> Result<()> {
    match try_bind_var_val(var_vals, name, val) {
        Ok(()) => Ok(()),
        Err((prev, val)) => {
            IrConstructionSnafu { details: format!("bind mismatch on {name}, {prev} != {val} ({context})") }.fail()
        }
    }
}

fn merge_var_vals_checked(dst: &mut HashMap<String, i64>, src: &HashMap<String, i64>, context: &str) -> Result<()> {
    for (name, val) in src {
        insert_var_val_checked(dst, name, *val, context)?;
    }
    Ok(())
}

fn extract_var_vals(root: &Arc<UOp>) -> Result<HashMap<String, i64>> {
    let mut var_vals = HashMap::new();
    for node in root.toposort() {
        if let Op::Bind(ops::Bind { var, value }) = node.op()
            && let Op::Const(cv) = value.op()
            && let Some(val) = cv.0.try_int()
        {
            let name = match var.op() {
                Op::DefineVar(ops::DefineVar { name, .. }) => Some(name.as_str()),
                Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none() => arg.name.as_deref(),
                _ => None,
            };
            if let Some(name) = name {
                insert_var_val_checked(&mut var_vals, name, val, "bind extraction")?;
            }
        }
    }
    Ok(var_vals)
}

fn schedule_cache_disabled_by_env() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("SVOD_DISABLE_SCHEDULE_CACHE").as_deref() == Ok("1"))
}

fn schedule_result_from_sink_with_cache(
    sink: Arc<UOp>,
    mut var_vals: HashMap<String, i64>,
    config: &PrepareConfig,
) -> Result<crate::schedule::ScheduleResult> {
    // Everything below builds compiler artifacts, never model graph: detach from any
    // scope the caller is realizing inside so kernel bodies stay origin-free.
    let _detached = svod_ir::origin::OriginScope::suspend();
    svod_ir::dump_canonical_stage("tensor", &sink);
    if config.disable_schedule_cache || schedule_cache_disabled_by_env() {
        return schedule_result_from_sink_uncached(sink, var_vals, config);
    }

    let normalization = normalize_for_schedule_cache(&sink)?;
    merge_var_vals_checked(&mut var_vals, &normalization.var_vals, "schedule cache normalization")?;

    let codegen = resolve_codegen(&normalization.param_buffers, config)?;
    let sched_key = (crate::schedule_cache::content_hash(&normalization.normalized), codegen);

    let cache = crate::schedule_cache::schedule_cache();
    // Winner-computes: N threads missing the same key run rangeify once;
    // losers park until the winner has inserted, then take the cache hit.
    let entry = crate::schedule_cache::schedule_flight().run(
        sched_key.clone(),
        || {
            let guard = cache.guard();
            cache.get(&sched_key, &guard).cloned()
        },
        || {
            let rangeify_result =
                svod_schedule::rangeify_with_map(normalization.normalized.clone()).context(RangeifySnafu)?;
            let (kernel_graph, _) =
                svod_schedule::try_get_kernel_graph(rangeify_result.sink).context(KernelGraphSnafu)?;
            let pre_schedule = crate::schedule::create_pre_schedule(kernel_graph)?;
            let new_entry = Arc::new(crate::schedule_cache::CachedSchedule { pre_schedule: Arc::new(pre_schedule) });
            let guard = cache.guard();
            cache.insert(sched_key.clone(), Arc::clone(&new_entry), &guard);
            Ok(new_entry)
        },
    )?;

    let restored_pre_schedule = restore_post_schedule_pre_schedule(&entry.pre_schedule, &normalization);
    let schedule_input_buffers = build_schedule_input_buffers(&restored_pre_schedule);
    let result = crate::schedule::instantiate_schedule(
        &restored_pre_schedule,
        &schedule_input_buffers,
        &var_vals,
        config.device_local_outputs,
    )?;
    Ok(result)
}

fn schedule_result_from_sink_uncached(
    sink: Arc<UOp>,
    mut var_vals: HashMap<String, i64>,
    config: &PrepareConfig,
) -> Result<crate::schedule::ScheduleResult> {
    let _detached = svod_ir::origin::OriginScope::suspend();
    let normalization = normalize_for_schedule_cache(&sink)?;
    merge_var_vals_checked(&mut var_vals, &normalization.var_vals, "uncached schedule normalization")?;
    let rangeify_result = svod_schedule::rangeify_with_map(normalization.normalized.clone()).context(RangeifySnafu)?;
    let (kernel_graph, _) = svod_schedule::try_get_kernel_graph(rangeify_result.sink).context(KernelGraphSnafu)?;
    let pre_schedule = crate::schedule::create_pre_schedule(kernel_graph)?;
    let restored_pre_schedule = restore_post_schedule_pre_schedule(&pre_schedule, &normalization);
    let input_buffers = build_schedule_input_buffers(&restored_pre_schedule);
    let result = crate::schedule::instantiate_schedule(
        &restored_pre_schedule,
        &input_buffers,
        &var_vals,
        config.device_local_outputs,
    )?;
    Ok(result)
}

/// Pre-schedule cache normalization result.
///
/// - BUFFER -> PARAM
/// - buffer identities normalized recursively through view metadata
/// - strip runtime value from BIND(DEFINE_VAR, CONST)
/// - normalize standalone UNIQUE identity -> LUNIQUE
pub(crate) struct ScheduleCacheNormalization {
    pub normalized: Arc<UOp>,
    pub param_values: Vec<Arc<UOp>>,
    pub param_buffers: Vec<(u64, Arc<UOp>)>,
    pub unique_values: Vec<Arc<UOp>>,
    pub var_vals: HashMap<String, i64>,
}

/// Context for pre-schedule cache normalization.
pub(crate) struct NormalizeScheduleCacheCtx {
    pub param_map: HashMap<u64, usize>,
    pub param_values: Vec<Arc<UOp>>,
    pub param_buffers: Vec<(u64, Arc<UOp>)>,
    pub var_vals: HashMap<String, i64>,
    pub bind_mismatch: Option<String>,
}

/// Full pre-schedule cache normalization.
pub(crate) fn normalize_for_schedule_cache(sink: &Arc<UOp>) -> Result<ScheduleCacheNormalization> {
    let mut ctx = NormalizeScheduleCacheCtx {
        param_map: HashMap::new(),
        param_values: Vec::new(),
        param_buffers: Vec::new(),
        var_vals: HashMap::new(),
        bind_mismatch: None,
    };

    use svod_ir::op::pattern_derived::OpKey;
    use svod_ir::pattern::{RewriteResult, SimplifiedPatternMatcher};
    use svod_ir::rewrite::graph_rewrite_preserve_calls;

    let mut matcher = SimplifiedPatternMatcher::<NormalizeScheduleCacheCtx>::new();

    // Global BUFFER -> PARAM (erase runtime buffer identity in cache key).
    // REG/LOCAL allocations are kernel-internal storage, not CALL arguments.
    matcher.add(&[OpKey::Buffer], |node, ctx| {
        let Op::Buffer(ops::Buffer { arg, .. }) = node.op() else {
            return RewriteResult::NoMatch;
        };
        if arg.addrspace != Some(svod_ir::AddrSpace::Global) {
            return RewriteResult::NoMatch;
        }
        let Some(size) = node.buffer_size() else { return RewriteResult::NoMatch };
        let slot = *ctx.param_map.entry(node.id).or_insert_with(|| {
            let s = ctx.param_values.len();
            ctx.param_values.push(node.clone());
            s
        });
        ctx.param_buffers.push((node.id, node.clone()));
        RewriteResult::Rewritten(
            UOp::param(slot, size, node.dtype(), arg.device.clone())
                .with_tag(smallvec::smallvec![svod_ir::uop::canonical::TAG_SCHEDULE_CACHE_PARAM]),
        )
    });

    // Strip runtime value from BIND for cache-key stability and collect var_vals.
    // Replaced with PARAM(device=Some) so restoration stays reversible and
    // distinguishable from internal PARAM(device=None) nodes created by rangeify.
    matcher.add(&[OpKey::Bind], |node, ctx| {
        let Op::Bind(ops::Bind { var, value }) = node.op() else {
            return RewriteResult::NoMatch;
        };
        let name = match var.op() {
            Op::DefineVar(ops::DefineVar { name, .. }) => Some(name.as_str()),
            Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none() => arg.name.as_deref(),
            _ => None,
        };
        let Some(name) = name else { return RewriteResult::NoMatch };
        let Op::Const(cv) = value.op() else {
            return RewriteResult::NoMatch;
        };
        let Some(val) = cv.0.try_int() else {
            return RewriteResult::NoMatch;
        };

        if let Err((prev, val)) = try_bind_var_val(&mut ctx.var_vals, name, val) {
            if ctx.bind_mismatch.is_none() {
                ctx.bind_mismatch = Some(format!("bind mismatch on variable {name}: {prev} vs {val}"));
            }
            return RewriteResult::NoMatch;
        }
        RewriteResult::Rewritten(var.clone())
    });

    // Pre-schedule cache normalization:
    // - BUFFER(UNIQUE, DEVICE) -> PARAM
    // - view base identity normalized through child BUFFER -> PARAM
    // - BIND(DEFINE_VAR, CONST) -> PARAM + var_vals capture
    let normalized = graph_rewrite_preserve_calls(&matcher, sink.clone(), &mut ctx);

    if let Some(details) = ctx.bind_mismatch.take() {
        return IrConstructionSnafu { details }.fail();
    }

    // Normalize standalone UNIQUE identity to deterministic LUNIQUE slots.
    // This runs after BUFFER replacement to avoid capturing UNIQUE
    // nodes that are no longer present in the normalized graph.
    struct UniqueNormalizationCtx {
        unique_map: HashMap<u64, usize>,
        unique_values: Vec<Arc<UOp>>,
    }
    let mut unique_ctx = UniqueNormalizationCtx { unique_map: HashMap::new(), unique_values: Vec::new() };
    let mut unique_matcher = SimplifiedPatternMatcher::<UniqueNormalizationCtx>::new();
    unique_matcher.add(&[OpKey::Unique], |node, ctx| {
        let Op::Unique(_) = node.op() else {
            return RewriteResult::NoMatch;
        };
        let slot = *ctx.unique_map.entry(node.id).or_insert_with(|| {
            let s = ctx.unique_values.len();
            ctx.unique_values.push(node.clone());
            s
        });
        RewriteResult::Rewritten(UOp::lunique(Some(slot)))
    });
    let normalized = graph_rewrite_preserve_calls(&unique_matcher, normalized, &mut unique_ctx);

    ctx.param_buffers.sort_unstable_by_key(|(id, _)| *id);
    ctx.param_buffers.dedup_by_key(|(id, _)| *id);

    Ok(ScheduleCacheNormalization {
        normalized,
        param_values: ctx.param_values,
        param_buffers: ctx.param_buffers,
        unique_values: unique_ctx.unique_values,
        var_vals: ctx.var_vals,
    })
}

/// Post-schedule cache restore.
///
/// Restores normalized placeholders back to runtime graph form for this run:
/// - PARAM(slot, device=Some(_)) -> original source node for current invocation
/// - BUFFER(LUNIQUE, DEVICE, size) -> fresh runtime BUFFER (memoized by slot)
/// - standalone LUNIQUE(slot) -> original UNIQUE identity
///
/// BIND runtime values are carried separately through `var_vals` and applied
/// at execution-time via fixedvars, preserving `execute_with_vars` behavior.
pub(crate) fn restore_post_schedule_cache(root: &Arc<UOp>, normalization: &ScheduleCacheNormalization) -> Arc<UOp> {
    let mut subs: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    let mut lunique_buffers: HashMap<usize, Arc<UOp>> = HashMap::new();

    for node in root.toposort() {
        match node.op() {
            Op::Param(ops::Param { arg, .. })
                if node
                    .tag()
                    .as_ref()
                    .is_some_and(|tags| tags.contains(&svod_ir::uop::canonical::TAG_SCHEDULE_CACHE_PARAM)) =>
            {
                if let Some(original) = normalization.param_values.get(arg.slot) {
                    let restored_original = restore_post_schedule_cache(original, normalization);
                    subs.insert(UOpKey(node.clone()), restored_original);
                }
            }
            Op::Buffer(ops::Buffer { arg, .. }) => {
                let schedule_local = node
                    .tag()
                    .as_ref()
                    .is_some_and(|tags| tags.contains(&svod_ir::uop::canonical::TAG_SCHEDULE_LOCAL_BUFFER));
                if arg.addrspace != Some(svod_ir::AddrSpace::Global) || !schedule_local {
                    continue;
                }
                let slot = arg.slot;
                let restored = if let Some(existing) = lunique_buffers.get(&slot) {
                    existing.clone()
                } else {
                    let Some(device) = arg.device.clone() else { continue };
                    let Some(size) = node.buffer_size() else { continue };
                    let fresh = UOp::new_buffer(device, size, arg.dtype.clone());
                    lunique_buffers.insert(slot, fresh.clone());
                    fresh
                };
                subs.insert(UOpKey(node.clone()), restored);
            }
            Op::LUnique(slot) => {
                let restored = normalization.unique_values.get(*slot).cloned().unwrap_or_else(|| UOp::buffer_id(None));
                subs.insert(UOpKey(node.clone()), restored);
            }
            _ => {}
        }
    }

    // Restore over the whole cached graph so PARAM/BIND placeholders are
    // rewritten before schedule extraction.
    root.substitute(&subs)
}

/// Restore cached pre-schedule buffer UOps for the current invocation.
///
/// `pre_schedule` is cached with normalized PARAM placeholders; this helper
/// restores source/output buffer UOps to run-specific BUFFER identities while
/// callable identities/ASTs stay cached.
pub(crate) fn restore_post_schedule_pre_schedule(
    pre_schedule: &crate::schedule::PreSchedule,
    normalization: &ScheduleCacheNormalization,
) -> crate::schedule::PreSchedule {
    let mut flat_buf_uops = Vec::new();
    let mut source_counts = Vec::with_capacity(pre_schedule.items.len());

    for item in &pre_schedule.items {
        source_counts.push(item.sources.len());
        flat_buf_uops.extend(item.sources.iter().cloned());
    }
    let outputs_offset = flat_buf_uops.len();
    flat_buf_uops.extend(pre_schedule.output_buffer_uops.iter().cloned());

    if flat_buf_uops.is_empty() {
        return pre_schedule.clone();
    }

    let restored_flat = match restore_post_schedule_cache(&UOp::sink(flat_buf_uops), normalization).op() {
        Op::Sink(ops::Sink { sources, .. }) => sources.iter().cloned().collect::<Vec<_>>(),
        _ => unreachable!("sink substitution must preserve SINK root"),
    };

    let mut cursor = 0usize;
    let mut restored_items = Vec::with_capacity(pre_schedule.items.len());
    for (item, source_count) in pre_schedule.items.iter().zip(source_counts) {
        let end = cursor + source_count;
        let sources = restored_flat[cursor..end].to_vec();
        cursor = end;
        restored_items.push(crate::schedule::PreScheduleItem {
            kernel: item.kernel.clone(),
            // Callable bodies contain codegen PARAM formals. Only CALL args
            // and output identities are restored to concrete runtime buffers.
            ast: item.ast.clone(),
            sources,
            dependencies: item.dependencies.clone(),
            bound_ranges: item.bound_ranges.clone(),
        });
    }

    let output_buffer_uops = restored_flat[outputs_offset..].to_vec();
    crate::schedule::PreSchedule {
        items: restored_items,
        invocations: pre_schedule.invocations.clone(),
        output_buffer_uops,
    }
}

/// Collect every BUFFER reachable from the callable sources that has a
/// buffer registered in the tensor registry (`from_slice_on()` and
/// `realize()` register theirs), in one walk over all items, so schedule
/// creation never needs global registry lookups of its own.
fn build_schedule_input_buffers(pre_schedule: &crate::schedule::PreSchedule) -> crate::schedule::InputBuffers {
    let mut inputs = crate::schedule::InputBuffers::new();
    let mut reach = crate::schedule::ReachOnce::new();
    for source in pre_schedule.items.iter().flat_map(|item| &item.sources) {
        let collected = reach.walk(source, |node| {
            if matches!(node.op(), Op::Buffer(..))
                && let Some(buffer) = crate::tensor_registry::get_buffer(node.id)
            {
                inputs.insert(node.id, buffer);
            }
            Ok(())
        });
        collected.expect("buffer collection never fails");
    }
    inputs
}

fn output_indices_from_program_metadata(globals: &[usize], outs: &[usize], num_buffers: usize) -> Result<Vec<usize>> {
    if num_buffers == 0 {
        return IrConstructionSnafu { details: "cannot map outputs for kernel with zero buffers".to_string() }.fail();
    }
    if globals.is_empty() {
        return IrConstructionSnafu { details: "ProgramSpec.globals is empty".to_string() }.fail();
    }
    if outs.is_empty() {
        return IrConstructionSnafu { details: "ProgramSpec.outs is empty".to_string() }.fail();
    }

    let slot_to_position: HashMap<usize, usize> =
        globals.iter().copied().enumerate().map(|(position, slot)| (slot, position)).collect();

    let mut output_indices = Vec::with_capacity(outs.len());
    for &slot in outs {
        let Some(position) = slot_to_position.get(&slot).copied() else {
            return IrConstructionSnafu {
                details: format!("ProgramSpec.outs slot {slot} not found in ProgramSpec.globals={globals:?}"),
            }
            .fail();
        };
        if position >= num_buffers {
            return IrConstructionSnafu {
                details: format!(
                    "ProgramSpec output index {position} (slot {slot}) out of range for {num_buffers} buffers"
                ),
            }
            .fail();
        }
        output_indices.push(position);
    }

    output_indices.sort_unstable();
    output_indices.dedup();
    if output_indices.is_empty() {
        return IrConstructionSnafu { details: "ProgramSpec output mapping resolved to empty set".to_string() }.fail();
    }

    Ok(output_indices)
}

fn input_indices_from_program_metadata(globals: &[usize], ins: &[usize], num_buffers: usize) -> Result<Vec<usize>> {
    let slot_to_position: HashMap<usize, usize> =
        globals.iter().copied().enumerate().map(|(position, slot)| (slot, position)).collect();
    let mut input_indices = Vec::with_capacity(ins.len());
    for &slot in ins {
        let Some(position) = slot_to_position.get(&slot).copied() else {
            return IrConstructionSnafu {
                details: format!("ProgramSpec.ins slot {slot} not found in ProgramSpec.globals={globals:?}"),
            }
            .fail();
        };
        if position >= num_buffers {
            return IrConstructionSnafu {
                details: format!(
                    "ProgramSpec input index {position} (slot {slot}) out of range for {num_buffers} buffers"
                ),
            }
            .fail();
        }
        input_indices.push(position);
    }
    input_indices.sort_unstable();
    input_indices.dedup();
    Ok(input_indices)
}

fn resolve_item_buffer_indices(item: &ScheduleItem, uop_id_to_idx: &HashMap<u64, usize>) -> Result<Vec<usize>> {
    let mut indices = Vec::with_capacity(item.buffer_uop_ids.len());
    for &uop_id in &item.buffer_uop_ids {
        let Some(idx) = uop_id_to_idx.get(&uop_id).copied() else {
            return Err(crate::error::Error::BufferNotFound { uop_id });
        };
        indices.push(idx);
    }
    Ok(indices)
}

fn resolve_compiled_kernel_buffer_indices(
    item: &ScheduleItem,
    uop_id_to_idx: &HashMap<u64, usize>,
    globals: &[usize],
) -> Result<Vec<usize>> {
    let buffer_indices = resolve_item_buffer_indices(item, uop_id_to_idx)?;
    if buffer_indices.len() != globals.len() {
        return IrConstructionSnafu {
            details: format!(
                "PROGRAM expected {} compact buffers for slots {globals:?}, CALL {} supplied {} (buffer_uop_ids={:?})",
                globals.len(),
                item.kernel.id,
                buffer_indices.len(),
                item.buffer_uop_ids
            ),
        }
        .fail();
    }
    Ok(buffer_indices)
}

type OptKey = (u64, DeviceSpec, String, u64, u64);

fn optimized_kernel_key(
    ast: &Arc<UOp>,
    device: &DeviceSpec,
    compiler_identity: &str,
    renderer_fingerprint: u64,
    optimizer_fingerprint: u64,
) -> OptKey {
    debug_assert!(
        ast.toposort().iter().all(|node| node.origin().is_none()),
        "kernel bodies are origin-stripped at the cut so identical kernels share one cache entry"
    );
    (
        crate::schedule_cache::content_hash(ast),
        device.clone(),
        compiler_identity.to_string(),
        renderer_fingerprint,
        optimizer_fingerprint,
    )
}

/// Bounded global cache for optimized + compiled kernels keyed by AST hash.
///
/// Reads are lock-free via the underlying `papaya::HashMap`; the FIFO side
/// structure is touched only on insert under a short-lived mutex. The cap is
/// read once via `SVOD_OPT_CACHE_MAX` (default 4096); when capacity is
/// exceeded, the oldest insertions are evicted from both the map and the
/// FIFO.
struct OptCacheState {
    map: papaya::HashMap<OptKey, Arc<svod_runtime::kernel_cache::CachedKernel>>,
    fifo: parking_lot::Mutex<std::collections::VecDeque<OptKey>>,
    cap: usize,
}

/// In-flight dedup for OPT_CACHE misses: beam/heuristic optimization plus
/// render can dominate prepare, so concurrent same-kernel misses run once.
fn opt_flight() -> &'static crate::singleflight::Singleflight<OptKey> {
    static FLIGHT: std::sync::OnceLock<crate::singleflight::Singleflight<OptKey>> = std::sync::OnceLock::new();
    FLIGHT.get_or_init(crate::singleflight::Singleflight::new)
}

impl OptCacheState {
    const DEFAULT_CAP: usize = 4096;

    fn new() -> Self {
        let cap = std::env::var("SVOD_OPT_CACHE_MAX")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(Self::DEFAULT_CAP);
        Self { map: papaya::HashMap::new(), fifo: parking_lot::Mutex::new(std::collections::VecDeque::new()), cap }
    }

    fn insert(&self, key: OptKey, val: Arc<svod_runtime::kernel_cache::CachedKernel>) {
        let guard = self.map.guard();
        let was_new = self.map.insert(key.clone(), val, &guard).is_none();
        if !was_new {
            return;
        }
        let mut fifo = self.fifo.lock();
        fifo.push_back(key);
        while fifo.len() > self.cap {
            if let Some(evict) = fifo.pop_front() {
                self.map.remove(&evict, &guard);
            }
        }
    }
}

/// Global cache for optimized + compiled kernels; identical kernels across
/// prepare calls (e.g. sort substages with the same axis) skip optimization
/// and compilation. Bounded so long-running processes do not accumulate dead
/// kernel entries.
fn opt_cache() -> &'static OptCacheState {
    static OPT_CACHE: std::sync::OnceLock<OptCacheState> = std::sync::OnceLock::new();
    OPT_CACHE.get_or_init(OptCacheState::new)
}

/// A schedule item's compiled-kernel work, resolved up front so the cache-miss
/// pipeline can run off the schedule loop.
struct KernelSite {
    key: OptKey,
    ast: Arc<UOp>,
    device: Arc<Device>,
    renderer: svod_schedule::OptimizerRenderer,
    /// Beam timing only.
    buffers: Vec<Buffer>,
}

impl KernelSite {
    fn resolve(item: &ScheduleItem, config: &PrepareConfig, optimizer_fingerprint: u64) -> Result<Self> {
        let device_spec = item
            .buffers
            .iter()
            .map(|b| b.allocator().device_spec())
            .find(|spec| !spec.is_disk())
            .unwrap_or_else(svod_dtype::default_device::default_device);
        let device = config.resolve_device(&device_spec, svod_device::registry::registry())?;
        let renderer = get_optimizer_renderer(&device);
        let key = optimized_kernel_key(
            &item.ast,
            &device.device,
            device.compiler.cache_key(),
            renderer.cache_fingerprint(),
            optimizer_fingerprint,
        );
        let buffers = if matches!(config.optimizer.strategy, svod_schedule::OptStrategy::Beam { .. }) {
            item.buffers.clone()
        } else {
            Vec::new()
        };
        Ok(Self { key, ast: item.ast.clone(), device, renderer, buffers })
    }

    fn codegen(&self) -> &str {
        self.device.compiler.cache_key()
    }

    /// Optimize with the kernel name left un-suffixed (`finalize_kernel_name`
    /// assigns it in schedule order).
    fn optimize(&self, config: &PrepareConfig) -> Result<Arc<UOp>> {
        // Author-supplied `opts_to_apply` short-circuits before beam: such
        // kernels must go through the heuristic entry so `apply_explicit_opts`
        // honors the exact opt list (empty = none).
        let has_explicit_opts = config.optimizer.opts_to_apply.is_some()
            || matches!(self.ast.op(), Op::Sink(ops::Sink { info: Some(ki), .. }) if ki.opts_to_apply.is_some());
        if !has_explicit_opts && matches!(config.optimizer.strategy, svod_schedule::OptStrategy::Beam { .. }) {
            beam_search_optimize(self.ast.clone(), &self.renderer, &self.device, &self.buffers, config)
        } else {
            svod_schedule::optimize_kernel_with_naming(
                self.ast.clone(),
                &self.renderer,
                &config.optimizer,
                KernelNaming::Deferred,
            )
            .context(OptimizeSnafu)
        }
    }

    /// Render and compile a named optimized kernel, then publish it under `key`.
    fn compile(&self, optimized: Arc<UOp>) -> Result<Arc<CachedKernel>> {
        let program =
            svod_codegen::program_pipeline::program_from_sink_with_renderer(optimized, self.device.renderer.as_ref())
                .context(RenderKernelSnafu)?;
        let codegen = self.codegen();
        debug_assert!(
            program.toposort().iter().all(|node| node.origin().is_none()),
            "rendered programs inherit the stripped body, so the compiled-program cache stays shared"
        );
        let result = svod_runtime::kernel_cache::get_or_compile_kernel(
            crate::schedule_cache::content_hash(&program),
            codegen,
            || {
                let (spec, compiled) = compile_with_program_pipeline_components(
                    program.clone(),
                    self.device.renderer.as_ref(),
                    self.device.compiler.as_ref(),
                )?;
                let program = (self.device.runtime)(&compiled).context(CreateProgramSnafu)?;
                Ok(CachedKernel {
                    program,
                    device: codegen.to_string(),
                    code: spec.src,
                    entry_point: spec.name,
                    var_names: spec.var_names,
                    globals: spec.globals,
                    outs: spec.outs,
                    ins: spec.ins,
                    global_size: spec.global_size,
                    local_size: spec.local_size,
                })
            },
        )?;
        opt_cache().insert(self.key.clone(), Arc::clone(&result));
        Ok(result)
    }

    /// The whole miss pipeline for one kernel, inline. Detached for the same
    /// reason `compile_missing_kernels` detaches its workers: optimizing and
    /// rendering mint UOps, and this path runs under whatever scope the caller
    /// is realizing inside.
    fn build(&self, config: &PrepareConfig) -> Result<Arc<CachedKernel>> {
        let _detached = svod_ir::origin::OriginScope::suspend();
        let optimized = self.optimize(config)?;
        self.compile(finalize_kernel_name(&optimized))
    }

    fn cached(&self) -> Option<Arc<CachedKernel>> {
        opt_cache().map.pin().get(&self.key).cloned()
    }
}

/// `1` runs inline in order; anything else fans out over the global pool
/// sized by `prepare_execution_plan`.
fn map_on_threads<T: Send, R: Send>(threads: usize, items: Vec<T>, f: impl Fn(T) -> R + Sync + Send) -> Vec<R> {
    if threads == 1 { items.into_iter().map(f).collect() } else { items.into_par_iter().map(f).collect() }
}

/// Optimize, name, render and compile every distinct kernel this plan misses
/// in the optimized-kernel cache. Optimizing and compiling fan out over
/// `config.threads`; naming runs in schedule order in between, so the
/// `nK` suffixes — part of the source text and hence of the object-cache key —
/// never depend on thread scheduling. Keys another prepare is already
/// computing are skipped here and awaited by the caller through `opt_flight`.
/// A kernel that fails to optimize draws no name and fails the batch only
/// after the others are published, so a retry finds them cached.
fn compile_missing_kernels(
    sites: &[Option<KernelSite>],
    config: &PrepareConfig,
) -> Result<HashMap<OptKey, Arc<CachedKernel>>> {
    let mut seen = HashSet::new();
    let jobs: Vec<_> = sites
        .iter()
        .flatten()
        .filter(|site| seen.insert(site.key.clone()))
        .filter_map(|site| opt_flight().try_claim_miss(site.key.clone(), || site.cached()).map(|ticket| (site, ticket)))
        .collect();
    if jobs.is_empty() {
        return Ok(HashMap::new());
    }
    // Beam already fans out over its own worker processes.
    let threads =
        if matches!(config.optimizer.strategy, svod_schedule::OptStrategy::Beam { .. }) { 1 } else { config.threads };

    // Optimizing and rendering mint UOps; on every thread they must stay detached, or
    // the optimized AST and the rendered program inherit a scope and fork the caches
    // keyed on them (`OPT_CACHE`, the compiled-program cache, the on-disk beam cache).
    let optimized = map_on_threads(threads, jobs.iter().collect(), |(site, _)| {
        let _detached = svod_ir::origin::OriginScope::suspend();
        site.optimize(config)
    });
    let mut failed = None;
    let named: Vec<_> = jobs
        .into_iter()
        .zip(optimized)
        .filter_map(|(job, optimized)| match optimized {
            Ok(ast) => Some((job, finalize_kernel_name(&ast))),
            Err(err) => {
                failed.get_or_insert(err);
                None
            }
        })
        .collect();

    let compiled = map_on_threads(threads, named, |((site, ticket), ast)| {
        let _detached = svod_ir::origin::OriginScope::suspend();
        let compiled = site.compile(ast);
        drop(ticket);
        compiled.map(|kernel| (site.key.clone(), kernel))
    });
    match failed {
        Some(err) => Err(err),
        None => compiled.into_iter().collect(),
    }
}

pub(crate) fn runtime_effect_ast(ast: &Arc<UOp>) -> &Arc<UOp> {
    match ast.op() {
        Op::End(ops::End { computation, .. }) if matches!(computation.op(), Op::Copy(..) | Op::CustomFunction(..)) => {
            computation
        }
        _ => ast,
    }
}

fn optimizer_config_fingerprint(config: &PrepareConfig) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    config.optimizer.hash(&mut hasher);
    hasher.finish()
}

fn post_optimizer_behavior_fingerprint(config: &svod_schedule::OptimizerConfig) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    config.transcendental.hash(&mut hasher);
    config.disable_fast_idiv.hash(&mut hasher);
    hasher.finish()
}

/// Prepare an execution plan from a schedule.
///
/// This performs all one-time preparation work:
/// 1. Allocates all buffers
/// 2. Compiles callable kernels
/// 3. Creates prepared runtime ops (compiled program + copy/view/custom-function handling)
///
/// # Arguments
///
/// * `schedule` - The schedule from `create_schedule()`
///
/// # Returns
///
/// An `ExecutionPlan` ready for fast repeated execution.
///
/// # Errors
///
/// Returns error if compilation or buffer allocation fails.
fn prepare_execution_plan(
    schedule_result: &crate::schedule::ScheduleResult,
    config: &PrepareConfig,
) -> Result<ExecutionPlan> {
    // Optimizing, rendering and compiling are compiler work, not model graph: detach
    // so nothing they mint adopts the scope the caller is realizing inside.
    let _detached = svod_ir::origin::OriginScope::suspend();
    // Every prepare and, through the plan it returns, every execution passes
    // here: size the shared pool before either can use it.
    svod_runtime::ensure_thread_pool(config.threads);
    // Schedule items are already fully expanded by strict scheduler unroll.
    let mut schedule_items = schedule_result.items.clone();

    // Liveness-based memory planning. `PlannerMode::Arena` (default) packs
    // plannable buffers into one or two large allocations; `Remap` swaps
    // per-pool `Arc<Buffer>`s; `Disabled` short-circuits. Mode is selected
    // explicitly by PrepareConfig (`from_env` is the environment-reading
    // constructor).
    let planner_mode = config.planner_mode;
    let output_buffer_ids = collect_output_buffer_ids(
        &schedule_items,
        &schedule_result.output_uop_ids,
        schedule_result.alias_output_buffers.values(),
    );
    // Reuse is keyed by execution LEVEL (computed from callable deps, matching
    // the runtime's per-op leveling), so storage is only shared across the
    // per-level barrier — no ordering edges are injected. This is why the
    // planner never touches `instance_dependencies` (asserted below): the only
    // former writer of it (reuse deps) is gone, which also makes the
    // schedule-index ↔ op-index mapping a non-issue.
    let item_levels = crate::memory_planner::compute_item_levels(&schedule_items)?;
    let planner_result =
        crate::memory_planner::memory_planner(&schedule_items, &item_levels, &output_buffer_ids, planner_mode);
    trace!(
        mode = ?planner_result.metrics.mode,
        replacements = planner_result.buffer_replace.len(),
        buffers_reused = planner_result.buffers_reused,
        memory_saved_bytes = planner_result.memory_saved,
        logical_bytes = planner_result.metrics.logical_bytes,
        rounded_bytes = planner_result.metrics.rounded_bytes,
        logical_peak_bytes = planner_result.metrics.logical_peak_bytes,
        arena_committed_bytes = planner_result.metrics.arena_committed_bytes,
        physical_bytes = planner_result.metrics.physical_bytes,
        fragmentation_bytes = planner_result.metrics.fragmentation_bytes,
        padding_bytes = planner_result.metrics.padding_bytes,
        reused_allocations = planner_result.metrics.reused_allocations,
        reused_bytes = planner_result.metrics.reused_bytes,
        exclusions = ?planner_result.metrics.exclusions,
        "memory planner measurements"
    );
    if !planner_result.buffer_replace.is_empty() {
        crate::memory_planner::apply_buffer_replacements(&mut schedule_items, &planner_result.buffer_replace);
    }
    // The planner injects zero ordering edges; the real safety invariant (op
    // emission is 1:1 with schedule items, so planner levels == runtime levels)
    // is asserted after the emission loop below via `builder.op_count()`.

    debug!(num_items = schedule_items.len(), "schedule items ready for execution plan");

    // Resolve primary plan device from the first schedule item for plan metadata.
    // Individual compiled kernels may still resolve/compile on per-item devices.
    let alloc_registry = svod_device::registry::registry();
    let plan_device = {
        let device_spec = schedule_items
            .iter()
            .flat_map(|item| item.buffers.iter().map(|b| b.allocator().device_spec()))
            .chain(schedule_result.alias_output_buffers.values().map(|b| b.allocator().device_spec()))
            .find(|spec| !spec.is_disk())
            .or_else(|| schedule_result.alias_output_buffers.values().next().map(|b| b.allocator().device_spec()))
            .unwrap_or_else(svod_dtype::default_device::default_device);
        config.resolve_device(&device_spec, alloc_registry)?
    };
    let optimizer_fingerprint = optimizer_config_fingerprint(config);

    // Build the ExecutionPlan using the builder
    let mut builder = ExecutionPlanBuilder::new(plan_device.device.clone());

    // Step 1: Add all buffers to the plan
    // Buffers in each ScheduleItem are already in the correct order (from collect_callable_buffers).
    // We track buffers by their UOp ID (what they were registered under in tensor_registry's buffer index).
    let mut uop_id_to_idx: HashMap<u64, usize> = HashMap::new();
    let mut storage_to_idx: HashMap<BufferStorageKey, usize> = HashMap::new();

    for item in &schedule_items {
        // Ensure all buffers are allocated
        for (buffer, &uop_id) in item.buffers.iter().zip(item.buffer_uop_ids.iter()) {
            buffer.ensure_allocated().context(DeviceSnafu)?;

            if uop_id_to_idx.contains_key(&uop_id) {
                continue;
            }

            let storage_key = BufferStorageKey {
                id: buffer.id().0,
                offset: buffer.offset(),
                size: buffer.size(),
                dtype: buffer.dtype(),
            };

            let idx = if let Some(&existing_idx) = storage_to_idx.get(&storage_key) {
                builder.map_buffer(uop_id, existing_idx);
                existing_idx
            } else {
                let new_idx = builder.add_buffer(uop_id, buffer.clone());
                storage_to_idx.insert(storage_key, new_idx);
                new_idx
            };
            uop_id_to_idx.insert(uop_id, idx);
        }
    }

    // Alias-only outputs have no executable schedule item, but the plan still
    // owns their concrete Buffer handles and maps their logical UOp identities.
    for (&uop_id, buffer) in &schedule_result.alias_output_buffers {
        if uop_id_to_idx.contains_key(&uop_id) {
            continue;
        }
        buffer.ensure_allocated().context(DeviceSnafu)?;
        let storage_key =
            BufferStorageKey { id: buffer.id().0, offset: buffer.offset(), size: buffer.size(), dtype: buffer.dtype() };
        let idx = if let Some(&existing_idx) = storage_to_idx.get(&storage_key) {
            builder.map_buffer(uop_id, existing_idx);
            existing_idx
        } else {
            let idx = builder.add_buffer(uop_id, buffer.clone());
            storage_to_idx.insert(storage_key, idx);
            idx
        };
        uop_id_to_idx.insert(uop_id, idx);
    }

    // Step 2: Compile callable kernels and create prepared runtime ops.
    // COPY and CUSTOM_FUNCTION items need no compilation; everything else
    // resolves to a kernel site whose cache misses are compiled off the loop.
    let sites = schedule_items
        .iter()
        .map(|item| match runtime_effect_ast(&item.ast).op() {
            Op::Copy(..) | Op::CustomFunction(..) => Ok(None),
            _ => KernelSite::resolve(item, config, optimizer_fingerprint).map(Some),
        })
        .collect::<Result<Vec<_>>>()?;
    let compiled = compile_missing_kernels(&sites, config)?;

    for (item, site) in schedule_items.iter().zip(&sites) {
        let runtime_ast = runtime_effect_ast(&item.ast);

        // COPY operations: buffer-to-buffer transfer (DISK→CPU, CPU→CUDA, etc.)
        // No compilation needed — register as PreparedOp for runtime execution.
        if matches!(runtime_ast.op(), Op::Copy(..)) {
            let buffer_indices = resolve_item_buffer_indices(item, &uop_id_to_idx)?;
            builder.add_op_with_instance_dependencies(
                PreparedOp::BufferCopy(PreparedCopy {
                    id: item.kernel.id,
                    buffer_indices,
                    dependencies: item.dependencies.clone(),
                    origin: item.origin(),
                    origins: item.origins().clone(),
                }),
                item.instance_dependencies.clone(),
            );
            continue;
        }

        // CALL bodies rooted at CUSTOM_FUNCTION are lowered directly to runtime
        // PreparedOp::CustomFunction with typed dispatch. Match against the
        // unwrapped runtime AST so END(CustomFunction) reaches this branch
        // consistently with Copy above.
        if let Op::CustomFunction(ops::CustomFunction { kind, attrs }) = runtime_ast.op() {
            let buffer_indices = resolve_item_buffer_indices(item, &uop_id_to_idx)?;
            let runtime_vars = attrs.iter().flat_map(svod_runtime::execution_plan::collect_runtime_vars).collect();
            builder.add_op_with_instance_dependencies(
                PreparedOp::CustomFunction(PreparedCustomFunction {
                    id: item.kernel.id,
                    kind: kind.clone(),
                    attrs: attrs.clone(),
                    buffer_indices,
                    fixedvars: item.fixedvars.clone(),
                    dependencies: item.dependencies.clone(),
                    runtime_vars,
                    origin: item.origin(),
                    origins: item.origins().clone(),
                }),
                item.instance_dependencies.clone(),
            );
            continue;
        }

        let site = site.as_ref().expect("non-copy, non-custom items resolve to kernel sites");
        // Kernels another prepare was compiling are awaited here (holding no
        // tickets of our own); a failed foreign winner makes us build inline.
        let cached = match compiled.get(&site.key) {
            Some(kernel) => Arc::clone(kernel),
            None => opt_flight().run(site.key.clone(), || site.cached(), || site.build(config))?,
        };

        // Build buffer indices in compiled ABI order (`ProgramSpec.globals`), not necessarily CALL arg order.
        let buffer_indices = resolve_compiled_kernel_buffer_indices(item, &uop_id_to_idx, &cached.globals)?;

        trace!(
            kernel.ast_id = item.ast.id,
            num_buffers = item.buffers.len(),
            buffer_uop_ids = ?item.buffer_uop_ids,
            buffer_ids = ?item.buffers.iter().map(|buffer| buffer.id()).collect::<Vec<_>>(),
            buffer_indices = ?buffer_indices,
            globals = ?cached.globals,
            fixedvars = ?item.fixedvars,
            var_names = ?cached.var_names,
            "kernel buffer mapping"
        );

        // Create PreparedKernel
        // Note: buffer_ptrs and buffer_ids will be computed in ExecutionPlanBuilder::build()
        let vals = initial_kernel_var_values(item, &cached.var_names)?;
        let non_overridable_fixedvars = collect_non_overridable_fixedvars(item);

        let output_indices = output_indices_from_program_metadata(&cached.globals, &cached.outs, buffer_indices.len())?;
        let input_indices = input_indices_from_program_metadata(&cached.globals, &cached.ins, buffer_indices.len())?;

        let runtime_vars = svod_runtime::execution_plan::collect_runtime_vars(&item.ast);
        let prepared = PreparedKernel {
            id: item.kernel.id,
            ast: item.ast.clone(),
            kernel: cached,
            device: site.device.device.clone(),
            buffer_indices,
            output_indices,
            input_indices,
            vals,
            fixedvars: non_overridable_fixedvars,
            dependencies: item.dependencies.clone(),
            buffer_ptrs: Vec::new(), // Computed in build()
            buffer_ids: Vec::new(),  // Computed in build()
            runtime_vars,
            origin: item.origin(),
            origins: item.origins().clone(),
        };

        builder.add_op_with_instance_dependencies(
            PreparedOp::CompiledProgram(prepared),
            item.instance_dependencies.clone(),
        );
    }

    // 1:1 op↔item emission is the invariant that makes the planner's
    // (schedule-item) levels equal the runtime's (per-op) levels. Every branch
    // above emits exactly one op per item, so this must hold.
    debug_assert_eq!(
        builder.op_count(),
        schedule_items.len(),
        "execution plan must emit exactly one prepared op per schedule item (1:1 emission)"
    );

    // Deterministic output identification via ScheduleResult.output_uop_ids
    let mut output_buffer_indices = Vec::with_capacity(schedule_result.output_uop_ids.len());
    for &uop_id in &schedule_result.output_uop_ids {
        let Some(idx) = uop_id_to_idx.get(&uop_id).copied() else {
            return Err(crate::error::Error::BufferNotFound { uop_id });
        };
        output_buffer_indices.push(idx);
    }
    if output_buffer_indices.is_empty() {
        return IrConstructionSnafu { details: "prepare_execution_plan produced no output buffer indices".to_string() }
            .fail();
    }
    builder.set_output_buffers(output_buffer_indices);

    builder.build().context(ExecutionSnafu)
}

fn collect_output_buffer_ids<'a>(
    schedule: &crate::schedule::Schedule,
    output_uop_ids: &[u64],
    alias_outputs: impl Iterator<Item = &'a Buffer>,
) -> HashSet<u64> {
    let output_uop_set: HashSet<u64> = output_uop_ids.iter().copied().collect();
    let mut output_buffer_ids = HashSet::new();
    for item in schedule {
        for (buffer, &uop_id) in item.buffers.iter().zip(item.buffer_uop_ids.iter()) {
            if output_uop_set.contains(&uop_id) {
                output_buffer_ids.insert(buffer.id().0);
            }
        }
    }
    let alias_output_storage: HashSet<_> = alias_outputs.map(Buffer::storage_id).collect();
    output_buffer_ids.extend(
        schedule
            .iter()
            .flat_map(|item| &item.buffers)
            .filter_map(|buffer| alias_output_storage.contains(&buffer.storage_id()).then_some(buffer.id().0)),
    );
    output_buffer_ids
}

fn collect_non_overridable_fixedvars(item: &ScheduleItem) -> HashMap<String, i64> {
    // Schedule-loop bindings (eagerly unrolled outer ranges) must not be
    // overridden by user `var_vals` — they're loop counters, not symbolic
    // input variables. `loop_var_names` is populated at instantiation time
    // from the keys of `KernelInvocation.fixedvars`, structurally separating
    // loop counters from runtime variable binds. `_device_num` is similarly
    // selected by host-side MSTACK lane expansion and is never user-overridable.
    let mut locked = HashMap::with_capacity(item.loop_var_names.len() + 1);
    for name in &item.loop_var_names {
        if let Some(v) = item.fixedvars.get(name) {
            locked.insert(name.clone(), *v);
        }
    }
    if let Some(v) = item.fixedvars.get("_device_num") {
        locked.insert("_device_num".to_string(), *v);
    }
    locked
}

fn initial_kernel_var_values(item: &ScheduleItem, var_names: &[String]) -> Result<Vec<i64>> {
    let mut bounds = HashMap::new();
    for node in item.ast.toposort() {
        match node.op() {
            Op::DefineVar(ops::DefineVar { name, min_val, max_val }) => {
                bounds.insert(name.clone(), (*min_val, *max_val));
            }
            Op::Param(ops::Param { arg, .. })
                if arg.addrspace.is_none()
                    && let Some(name) = arg.name.as_deref()
                    && let Some((min, max)) = &arg.vmin_vmax
                    && let (Some(min), Some(max)) = (min.0.try_int(), max.0.try_int()) =>
            {
                bounds.insert(name.to_string(), (min, max));
            }
            _ => {}
        }
    }

    var_names
        .iter()
        .map(|name| {
            if let Some(value) = item.fixedvars.get(name) {
                return Ok(*value);
            }
            if name == "core_id" {
                return Ok(0);
            }
            // Unbound at prepare time: this is the documented prepare-once flow
            // (`variable.rs`), so seed an in-bounds placeholder and let
            // `ExecutionPlan::execute_with_vars` reject a value that never
            // arrives or arrives out of bounds.
            Ok(bounds.get(name.as_str()).map_or(0, |&(min, _)| min.max(0)))
        })
        .collect()
}

/// Render/compile entrypoint backed by PROGRAM pipeline stages.
fn compile_with_program_pipeline_components(
    kernel_ast: Arc<UOp>,
    renderer: &dyn svod_device::device::Renderer,
    compiler: &dyn svod_device::device::Compiler,
) -> Result<(svod_device::device::ProgramSpec, svod_device::device::CompiledSpec)> {
    if !matches!(kernel_ast.op(), Op::Program(..)) {
        return IrConstructionSnafu {
            details: format!(
                "compile_with_program_pipeline_components expects PROGRAM input, got {:?}",
                kernel_ast.op()
            ),
        }
        .fail();
    }
    // `do_render` validates the SOURCE specification it returns and `do_compile`
    // the BINARY it attaches, so neither stage is re-read through `from_uop`.
    let (rendered, spec) =
        svod_codegen::program_pipeline::do_render(&kernel_ast, renderer).context(RenderKernelSnafu)?;
    let (_, compiled) = svod_codegen::program_pipeline::do_compile(&rendered, compiler).context(CompileKernelSnafu)?;
    Ok((spec, compiled))
}

/// Resolve the device string for cache keying (includes compiler cache key).
pub(crate) fn resolve_codegen(param_buffers: &[(u64, Arc<UOp>)], config: &PrepareConfig) -> Result<String> {
    let alloc_registry = svod_device::registry::registry();
    let spec = param_buffers
        .iter()
        .find_map(|(id, _)| {
            let spec = crate::tensor_registry::get_buffer(*id)?.allocator().device_spec();
            (!spec.is_disk()).then_some(spec)
        })
        .or_else(|| {
            param_buffers.iter().find_map(|(_, u)| {
                let Op::Buffer(ops::Buffer { arg, .. }) = u.op() else {
                    return None;
                };
                arg.device.as_ref().filter(|spec| !spec.is_disk()).cloned()
            })
        })
        .unwrap_or_else(svod_dtype::default_device::default_device);
    let device = config.resolve_device(&spec, alloc_registry)?;
    Ok(device.compiler.cache_key().to_string())
}

/// Get the optimizer renderer for a device.
pub(crate) fn get_optimizer_renderer(device: &Device) -> svod_schedule::OptimizerRenderer {
    let renderer = match device.device {
        DeviceSpec::Cpu => svod_schedule::OptimizerRenderer::cpu(),
        DeviceSpec::Cuda { .. } => device
            .renderer
            .gpu_arch()
            .and_then(svod_dtype::GpuArch::cuda)
            .map(svod_schedule::OptimizerRenderer::for_cuda_arch)
            .unwrap_or_else(svod_schedule::OptimizerRenderer::cuda),
        DeviceSpec::Metal { .. } => device
            .renderer
            .gpu_arch()
            .and_then(svod_dtype::GpuArch::metal)
            .map(svod_schedule::OptimizerRenderer::for_metal_family)
            .unwrap_or_else(svod_schedule::OptimizerRenderer::metal),
        // AMD picks the profile (wave size, LDS, WMMA/MFMA tensor cores) from
        // the opened device's arch, exposed via the renderer. Falls back to
        // RDNA3 if the renderer can't report arch.
        DeviceSpec::Amd { .. } => device
            .renderer
            .gpu_arch()
            .and_then(svod_dtype::GpuArch::amd)
            .map(svod_schedule::OptimizerRenderer::for_amd_arch)
            .unwrap_or_else(svod_schedule::OptimizerRenderer::amd_rdna3),
        _ => svod_schedule::OptimizerRenderer::cpu(),
    };
    renderer.with_codegen_renderer(device.renderer.as_ref())
}

/// Optimize a kernel AST using beam search auto-tuning.
///
/// Beam search explores multiple optimization paths and selects the fastest
/// by compiling and timing each candidate. Slower than heuristics but can
/// find better optimizations. Beam and heuristic are mutually exclusive.
fn beam_search_optimize(
    ast: Arc<UOp>,
    renderer: &svod_schedule::OptimizerRenderer,
    device: &Device,
    buffers: &[Buffer],
    config: &PrepareConfig,
) -> Result<Arc<UOp>> {
    let optimizer_config = &config.optimizer;
    let mut resolved_beam_config = optimizer_config.beam.clone();
    // `PARALLEL` overrides for tinygrad parity; GPUs otherwise fan out over the
    // thread budget, CPU compiles in-process.
    let default_workers = match device.device {
        _ if resolved_beam_config.compile_workers > 0 => resolved_beam_config.compile_workers,
        DeviceSpec::Cuda { .. } | DeviceSpec::Amd { .. } | DeviceSpec::Metal { .. } => config.threads,
        _ => 0,
    };
    resolved_beam_config.compile_workers =
        std::env::var("PARALLEL").ok().and_then(|value| value.parse().ok()).unwrap_or(default_workers);
    let beam_config = &resolved_beam_config;
    let beam_debug = std::env::var("BEAM_DEBUG").ok().and_then(|value| value.parse::<u8>().ok()).unwrap_or(0);
    if beam_debug > 0 {
        eprintln!(
            "[beam] start device={} width={} parallel={} spawn_workers={} max_tasks_per_child={} timeout={}s",
            device.device.canonicalize(),
            beam_config.beam_width,
            beam_config.compile_workers,
            beam_config.compile_workers.max(1),
            beam_config.max_tasks_per_child,
            beam_config.compile_timeout_secs
        );
    }
    let post_optimizer_config = optimizer_config.clone();
    let wire_graph = svod_ir::OptimizerWireGraph::from_root(&ast).context(UOpSnafu)?;
    // Prepare scheduler (applies symbolic simplification and loop→global).
    // BEAM and heuristic are mutually exclusive.
    let scheduler = prepare_scheduler(ast, renderer).context(OptimizeSnafu)?;

    // Ensure all buffers are allocated for timing
    for buf in buffers {
        buf.ensure_allocated().context(DeviceSnafu)?;
    }

    // Clone buffers for the closure (Buffer is Clone + Send + Sync)
    let buffers: Vec<Buffer> = buffers.to_vec();
    let bench_config = svod_runtime::BenchmarkConfig { timing_runs: beam_config.num_runs, ..Default::default() };

    // Clone device components for the closure
    let dev_compiler = device.compiler.clone();

    // When `BEAM_LOG_SURPASS_MAX` is set, every dropped candidate prints
    // one line with the failure reason, applied-opt chain, and (for "too
    // many uops") the top Op-variant counts in the linearized program.
    let log_surpass = std::env::var("BEAM_LOG_SURPASS_MAX").is_ok();

    struct CompiledBeamProgram {
        compiled: svod_device::device::CompiledSpec,
        opts: Vec<svod_schedule::optimizer::Opt>,
        vals: Vec<i64>,
        global_size: [usize; 3],
        local_size: Option<[usize; 3]>,
    }
    let worker_init = crate::beam_worker::WorkerInit {
        protocol_version: crate::beam_worker::BEAM_WORKER_PROTOCOL_VERSION,
        graph: wire_graph,
        device: device.device.clone(),
        gpu_arch: device.renderer.gpu_arch(),
        compiler_key: dev_compiler.cache_key().to_string(),
        renderer_fingerprint: renderer.cache_fingerprint(),
        base_opt_count: scheduler.applied_opts.len(),
        beam: beam_config.clone(),
        transcendental: post_optimizer_config.transcendental,
        disable_fast_idiv: post_optimizer_config.disable_fast_idiv,
        log_surpass,
    };
    let mut worker_pool = crate::beam_worker::WorkerPool::new(beam_config.compile_workers.max(1), worker_init)
        .map_err(|error| svod_device::Error::Runtime { message: error.to_string() })
        .context(DeviceSnafu)?;
    let benchmark_ast = scheduler.ast().clone();
    let launch_placeholder = [UOp::index_const(1), UOp::index_const(1), UOp::index_const(1)];
    let compile_wave = |candidates: &[Vec<svod_schedule::optimizer::Opt>],
                        emit: &mut dyn FnMut(usize, CompiledCandidate<CompiledBeamProgram>)|
     -> std::result::Result<(), svod_schedule::optimizer::OptError> {
        if beam_debug > 0 {
            eprintln!("[beam] compile wave: {} candidates", candidates.len());
        }
        let mut completed_count = 0usize;
        worker_pool.run(candidates, |response| {
            let index = response.index;
            let Some(artifact) = response.result else {
                if let Some(error) = response.error.filter(|_| log_surpass || beam_debug > 1) {
                    eprintln!("[BEAM drop] worker candidate={index}: {error}");
                }
                return;
            };
            let compiled = match svod_device::device::CompiledSpec::from_beam_worker(
                artifact.name,
                artifact.source.clone(),
                artifact.bytes,
                benchmark_ast.clone(),
                artifact.abi,
                launch_placeholder.clone(),
                artifact.identity,
                &device.device,
                dev_compiler.cache_key(),
            ) {
                Ok(compiled) => compiled,
                Err(error) => {
                    if log_surpass || beam_debug > 0 {
                        eprintln!("[BEAM drop] validate_worker_artifact candidate={index}: {error}");
                    }
                    return;
                }
            };
            completed_count += 1;
            let mut binary_key = Vec::with_capacity(compiled.bytes.len() + 1);
            if compiled.bytes.is_empty() {
                binary_key.push(b'S');
                binary_key.extend_from_slice(artifact.source.as_bytes());
            } else {
                binary_key.push(b'B');
                binary_key.extend_from_slice(&compiled.bytes);
            }
            let preparation = Duration::from_nanos(artifact.preparation_ns);
            let compilation = Duration::from_nanos(artifact.compilation_ns);
            if beam_debug > 1 {
                eprintln!(
                    "[beam] candidate={index:5} completed={completed_count:4}/{:4} compile={compilation:?} opts={:?}",
                    candidates.len(), candidates[index]
                );
            }
            emit(index, CompiledCandidate {
                artifact: CompiledBeamProgram {
                    compiled, opts: candidates[index].clone(), vals: artifact.vals,
                    global_size: artifact.global_size, local_size: artifact.local_size,
                },
                binary_key, compute_ops: artifact.compute_ops, preparation, compilation,
            });
        })
            .map_err(|error| svod_schedule::optimizer::OptError::BeamWorker { message: error.to_string() })
    };

    let dev_runtime = device.runtime.clone();
    let benchmark = |candidate: &CompiledBeamProgram, early_stop: Option<Duration>| -> Option<Duration> {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        match catch_unwind(AssertUnwindSafe(|| {
            let program = match (dev_runtime)(&candidate.compiled) {
                Ok(program) => program,
                Err(e) => {
                    if log_surpass {
                        eprintln!("[BEAM drop] runtime_err: {e:?} opts={:?}", candidate.opts);
                    }
                    return None;
                }
            };

            let buffer_ptrs: Vec<*mut u8> = buffers.iter().map(|buffer| unsafe { buffer.as_raw_ptr() }).collect();

            let vals = &candidate.vals;

            const MAX_TEST_GLOBAL_SIZE: usize = 65536;
            let mut test_global_size = candidate.global_size;
            let original_size: usize = test_global_size.iter().product();
            while test_global_size.iter().product::<usize>() > MAX_TEST_GLOBAL_SIZE {
                let mut halved = false;
                for axis in (0..test_global_size.len()).rev() {
                    if test_global_size[axis] > 16 {
                        test_global_size[axis] /= 2;
                        halved = true;
                        break;
                    }
                }
                if !halved {
                    break;
                }
            }
            let shrunk_size: usize = test_global_size.iter().product();
            let factor = if shrunk_size > 0 { original_size as f64 / shrunk_size as f64 } else { 1.0 };

            let mut config = bench_config.clone();
            config.early_stop = early_stop
                .map(|timing| Duration::from_nanos((timing.as_nanos() as f64 / factor).min(u64::MAX as f64) as u64));
            config.clear_l2 = renderer.device.has_hardware_cache_invalidate();
            let result = unsafe {
                svod_runtime::benchmark_kernel(
                    program.as_ref(),
                    &buffer_ptrs,
                    vals,
                    Some(test_global_size),
                    candidate.local_size,
                    &config,
                )
                .ok()?
            };
            Some(Duration::from_nanos((result.min.as_nanos() as f64 * factor).min(u64::MAX as f64) as u64))
        })) {
            Ok(timing) => timing,
            Err(_) => {
                if log_surpass {
                    eprintln!("[BEAM drop] panic_in_benchmark opts={:?}", candidate.opts);
                }
                None
            }
        }
    };

    let behavior_fingerprint = post_optimizer_behavior_fingerprint(&post_optimizer_config);
    let result = beam_search_cached_remote(
        scheduler,
        beam_config,
        device.compiler.cache_key(),
        behavior_fingerprint,
        compile_wave,
        benchmark,
    );
    let result = result.context(OptimizeSnafu)?;
    if beam_debug > 0 {
        eprintln!(
            "[beam] final timing={:?} iterations={} generated={} compiled={} unique_binary={} benchmarked={} opts={:?}",
            result.timing,
            result.iterations,
            result.generated,
            result.compiled,
            result.unique_binary,
            result.benchmarked,
            result.scheduler.applied_opts
        );
    }

    // Debug: log beam search results
    tracing::debug!(
        opts = ?result.scheduler.applied_opts,
        timing = ?result.timing,
        iterations = result.iterations,
        generated = result.generated,
        unique_ir = result.unique_ir,
        compiled = result.compiled,
        unique_binary = result.unique_binary,
        benchmarked = result.benchmarked,
        generation_time = ?result.stage_timings.generation,
        filtering_time = ?result.stage_timings.filtering,
        compilation_time = ?result.stage_timings.compilation,
        binary_dedup_time = ?result.stage_timings.binary_dedup,
        benchmarking_time = ?result.stage_timings.benchmarking,
        "beam_search_optimize: completed"
    );

    // Apply post-optimization to final result with renderer so pm_add_gpudims runs
    // (Thread → core_id, Global → SPECIAL).
    let raw_ast = result.scheduler.get_optimized_ast_with_naming(KernelNaming::Deferred);
    apply_post_optimization_with_config(raw_ast, renderer, &post_optimizer_config).context(OptimizeSnafu)
}

#[cfg(test)]
#[path = "test/unit/realize_internal.rs"]
mod tests;
