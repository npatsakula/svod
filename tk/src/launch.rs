//! Direct kernel launch — the svod analog of tinygrad's `run_linear`.
//!
//! Tinygrad runs a hand-built tile kernel by compiling its SINK straight through
//! the program pipeline and dispatching it against concrete buffers, *bypassing*
//! the tensor scheduler entirely (`test/testextra/test_tk.py`):
//!
//! ```python
//! linear = UOp(Ops.LINEAR, src=(sink.call(*buffer_uops),))
//! run_linear(linear)   # compile + dispatch, writes outputs in place
//! ```
//!
//! [`launch`] is that path: `program_from_sink` → render → compile → runtime →
//! [`Program::execute`], with the concrete buffers ordered by the compiled ABI
//! (`ProgramSpec.globals`, the sorted PARAM slot list). The kernel body's PARAM
//! placeholders (minted by [`Kernel`](crate::Kernel)) bind positionally to the
//! buffers exactly as tinygrad's `sink.call(bufs)` binds them — there is no
//! scheduler, no `After`-wrapping, no buffer reallocation.

use std::collections::HashMap;
use std::sync::Arc;

use snafu::{IntoError, OptionExt, ResultExt, Snafu};
use svod_codegen::program_pipeline::{self, ProgramTarget};
use svod_device::Buffer;
use svod_device::device::{Device, Program, ProgramSpec};
use svod_dtype::{DType, DeviceSpec, GpuArch};
use svod_ir::UOp;
use svod_ir::ops;
use svod_tensor::Tensor;

use crate::target::ArchSet;

/// Result type for the launch path.
pub type Result<T, E = Error> = std::result::Result<T, E>;

pub(crate) fn plan_compact_buffers(globals: &[usize], supplied: usize) -> Result<Vec<(usize, usize)>> {
    if supplied != globals.len() {
        return BufferCountSnafu { expected: globals.len(), supplied, slots: globals.to_vec() }.fail();
    }
    Ok(globals.iter().copied().enumerate().collect())
}

/// Errors raised while compiling or dispatching a hand-built kernel.
///
/// Nested backend errors are boxed (`svod_runtime::Error` alone is ~144 B), so
/// the enum — and any `Result<(), Error>` returned from the hot dispatch path —
/// stays small (`clippy::result_large_err`).
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    /// The SINK could not be rendered/compiled through the program pipeline.
    #[snafu(display("compile kernel {name:?}: {source}"))]
    Compile {
        name: String,
        #[snafu(source(from(svod_device::Error, Box::new)))]
        source: Box<svod_device::Error>,
    },

    /// The runtime factory rejected the compiled spec.
    #[snafu(display("instantiate runtime for {name:?}: {source}"))]
    Runtime {
        name: String,
        #[snafu(source(from(svod_device::Error, Box::new)))]
        source: Box<svod_device::Error>,
    },

    /// Resolving the concrete `Device` (renderer/compiler/runtime) failed.
    #[snafu(display("resolve device for spec {spec}: {source}"))]
    DeviceFactory {
        spec: String,
        #[snafu(source(from(svod_runtime::Error, Box::new)))]
        source: Box<svod_runtime::Error>,
    },

    /// Obtaining an allocator for the buffer's device failed.
    #[snafu(display("allocator for spec {spec}: {source}"))]
    Allocator {
        spec: String,
        #[snafu(source(from(svod_device::Error, Box::new)))]
        source: Box<svod_device::Error>,
    },

    /// Deriving the dispatch ABI / launch dimensions failed.
    #[snafu(display("program spec for {name:?}: {source}"))]
    Spec {
        name: String,
        #[snafu(source(from(svod_device::Error, Box::new)))]
        source: Box<svod_device::Error>,
    },

    /// A buffer required by the compiled ABI was not supplied.
    #[snafu(display("buffer slot {slot} not supplied (of {supplied} buffers)"))]
    BufferMissing { slot: usize, supplied: usize },

    /// The compact runtime buffer vector does not match ProgramInfo.globals.
    #[snafu(display("expected {expected} compact buffers for slots {slots:?}, got {supplied}"))]
    BufferCount { expected: usize, supplied: usize, slots: Vec<usize> },

    /// A required buffer could not be allocated on its device.
    #[snafu(display("allocate buffer slot {slot} (of {supplied}): {source}"))]
    BufferAlloc {
        slot: usize,
        supplied: usize,
        #[snafu(source(from(svod_device::Error, Box::new)))]
        source: Box<svod_device::Error>,
    },

    /// Realizing or allocating a tensor argument failed.
    #[snafu(display("realize tensor argument: {source}"))]
    Realize {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },

    /// The kernel dispatch itself failed.
    #[snafu(display("dispatch kernel {name:?}: {source}"))]
    Dispatch {
        name: String,
        #[snafu(source(from(svod_device::Error, Box::new)))]
        source: Box<svod_device::Error>,
    },

    /// Wrapping the kernel SINK as a graph `custom_kernel` (`Op::Call`) node failed.
    #[snafu(display("graph custom_kernel {name:?}: {source}"))]
    CustomKernel {
        name: String,
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },

    /// The resolved device's GPU arch isn't in the kernel's supported [`ArchSet`].
    #[snafu(display("unsupported target: kernel supports {supported}, device {spec:?} resolved to {resolved:?}"))]
    UnsupportedArch { supported: ArchSet, spec: DeviceSpec, resolved: Option<GpuArch> },

    /// The LLVM GPU backend (`amdgcn` / `nvptx64`) needed to compile tile kernels
    /// for the resolved arch is absent from the host `clang`.
    #[snafu(display("LLVM {target} target unavailable — a clang built with the {target} backend is required"))]
    ToolchainUnavailable { target: &'static str },

    /// Casting an input to the kernel's bf16 operand dtype failed.
    #[snafu(display("cast matmul operand: {source}"))]
    Operand {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },

    // ── Structured "malformed request" errors ───────────────────────────────────
    // A FIXED shape/dtype property is wrong on an otherwise-runnable kernel (a caller
    // bug). Distinct from a runtime length that merely doesn't tile (`Ok(None)`) and
    // from an unsupported device ([`Error::UnsupportedArch`]). Carry the offending
    // values structurally instead of a pre-formatted string.
    /// An operand dtype is unsupported by the kernel.
    #[snafu(display("{kernel}: operand dtype {got:?} unsupported (expected {expected})"))]
    Dtype { kernel: &'static str, got: DType, expected: &'static str },

    /// A dimension must be a whole multiple of `multiple` (e.g. the WMMA edge or the
    /// matmul block).
    #[snafu(display("{kernel}: {dim} = {value} must be a multiple of {multiple}"))]
    DimMultiple { kernel: &'static str, dim: &'static str, value: usize, multiple: usize },

    /// One dimension must be divisible by another (e.g. GQA `H % H_kv == 0`).
    #[snafu(display("{kernel}: {dim} = {value} must be divisible by {divisor} = {divisor_value}"))]
    DimDivisible { kernel: &'static str, dim: &'static str, value: usize, divisor: &'static str, divisor_value: usize },

    /// Operands must be square and equal-sized (`[n,n] · [n,n]`).
    #[snafu(display("{kernel}: operands must be square and equal-sized, got {a:?} · {b:?}"))]
    NotSquare { kernel: &'static str, a: [usize; 2], b: [usize; 2] },

    /// Two operands disagree on a named shared dimension (e.g. KNN's feature dim `D`,
    /// which the query `x` and the corpus `c` must match).
    #[snafu(display("{kernel}: operands disagree on {dim}: {a} != {b}"))]
    OperandDimMismatch { kernel: &'static str, dim: &'static str, a: usize, b: usize },

    /// An operand's shape could not be determined.
    #[snafu(display("{kernel}: operand {operand}: shape is indeterminate"))]
    OperandIndeterminateShape { kernel: &'static str, operand: &'static str },

    /// An operand has the wrong rank for the kernel.
    #[snafu(display("{kernel}: operand {operand}: expected a rank-{expected} tensor, got rank {got}"))]
    OperandRank { kernel: &'static str, operand: &'static str, expected: usize, got: usize },

    /// An operand has a symbolic (non-constant) dimension; the kernel needs static dims.
    #[snafu(display("{kernel}: operand {operand}: dim {axis} is not statically known"))]
    OperandSymbolicDim { kernel: &'static str, operand: &'static str, axis: usize },
}

/// Resolve a tensor operand's shape to concrete `usize` dims, or an
/// `Operand*` error if the shape is indeterminate, the wrong rank, or has a symbolic
/// (non-constant) dimension. Kernel builders need statically-known dims, so a
/// malformed operand is a caller error reported through `Result`, not a panic.
pub(crate) fn concrete_dims(
    t: &Tensor,
    kernel: &'static str,
    operand: &'static str,
    rank: usize,
) -> Result<Vec<usize>> {
    let shape = t.shape().ok().context(OperandIndeterminateShapeSnafu { kernel, operand })?;
    snafu::ensure!(shape.len() == rank, OperandRankSnafu { kernel, operand, expected: rank, got: shape.len() });
    (0..rank).map(|i| shape[i].as_const().context(OperandSymbolicDimSnafu { kernel, operand, axis: i })).collect()
}

/// Compile `sink` for `device` and dispatch it against `buffers`, populating the
/// output buffer(s) in place. `buffers` are ordered output(s)-first then inputs,
/// matching the `gl()` declaration order in the kernel body (PARAM slots 0,1,…).
///
/// Binding is *compact and ordinal*: the compiled ABI (`ProgramSpec.globals`)
/// is a sorted slot list, and the pointer for ABI position `i` is `buffers[i]`
/// — the declared slot number is carried only for diagnostics. So a kernel that
/// declares slots `[0, 5]` takes exactly two buffers, and `buffers[1]` binds to
/// slot 5; supplying `globals.len()` buffers is required, and a sparse vector
/// indexed by slot number is not. `plan_compact_buffers` is the mapping, pinned
/// by `sparse_and_interleaved_program_slots_plan_compact_buffers`.
pub fn launch(device: &Device, sink: Arc<UOp>, buffers: &[Buffer]) -> Result<()> {
    let compiled = compile(device, sink, buffers)?;
    // SAFETY: `compile` resolved + allocated every ABI buffer pointer, and the
    // synchronous dispatch (`wait=true`) holds them for the call's duration.
    unsafe { compiled.dispatch(true) }
}

/// A compiled kernel bound to concrete buffers, ready for repeated dispatch —
/// the compile-once analog of [`launch`]. Render + compile happen exactly once
/// (in [`compile`]); [`CompiledLaunch::dispatch`] only re-issues the kernel, so
/// a benchmark (or any hot loop) pays the pipeline cost once and times pure
/// dispatch. The bound buffers are held (`Buffer` is `Arc`-backed) so their
/// allocations outlive every replay.
pub struct CompiledLaunch {
    prog: Box<dyn Program>,
    ptrs: Vec<*mut u8>,
    global_size: [usize; 3],
    local_size: Option<[usize; 3]>,
    vals: Vec<i64>,
    name: String,
    /// Keeps the bound allocations alive for the lifetime of the raw `ptrs`.
    _buffers: Vec<Buffer>,
}

impl CompiledLaunch {
    /// Re-dispatch the compiled kernel against its bound buffers. `wait=true`
    /// blocks to completion (so wall-clock timing around it is valid);
    /// `wait=false` submits asynchronously (pair with a trailing `wait=true`
    /// dispatch, or the backend's own sync, to drain the timeline).
    ///
    /// # Safety
    ///
    /// The bound buffers must still be allocated (they are, while `self` lives)
    /// and the caller must avoid concurrent races on them.
    pub unsafe fn dispatch(&self, wait: bool) -> Result<()> {
        // SAFETY: pointers are allocated + sized to the kernel's expectations and
        // held alive by `self._buffers`; the caller upholds the race contract.
        unsafe {
            self.prog
                .execute(&self.ptrs, &self.vals, Some(self.global_size), self.local_size, wait)
                .context(DispatchSnafu { name: self.name.clone() })
        }
    }

    /// Dispatch once through a profiling execution context and return the
    /// **GPU device time** in nanoseconds (`end_ns − start_ns`) from the
    /// HW-stamped dispatch timestamps, or `Ok(None)` when the backend does not
    /// stamp dispatches (CPU: `new_exec_context` yields `None`, or the context
    /// returns no timestamp handle). Wall-clock around this call is *not* the
    /// device time — it includes the context mint + the `synchronize` drain.
    ///
    /// Mirrors [`svod_runtime::ExecutionPlan::execute_profiled`]: mint a
    /// per-program `PlanContext`, submit the dispatch (async), drain with
    /// `synchronize` so the GPU has written back the per-dispatch stamps, then
    /// read `timestamps_ns()`. The 10 ns/tick CP stamp gives true on-device
    /// kernel time, free of host launch/submit overhead.
    ///
    /// # Panics
    ///
    /// In a debug build, panics if the backend reports an end timestamp earlier
    /// than the start (the `end - start` subtraction underflows).
    ///
    /// # Safety
    ///
    /// Identical contract to [`CompiledLaunch::dispatch`]: the bound buffers
    /// stay allocated while `self` lives and the caller avoids data races.
    pub fn dispatch_gpu_ns(&self) -> Result<Option<u64>> {
        let Some(ctx) = self.prog.new_exec_context().context(DispatchSnafu { name: self.name.clone() })? else {
            return Ok(None);
        };
        // SAFETY: pointers are allocated + sized to the kernel's expectations and
        // held alive by `self._buffers`; same contract as `dispatch`.
        let handle = unsafe {
            // profile=true: this is the timestamp path and we hold `handle`
            // through `synchronize` below, so arming the probes is safe.
            ctx.dispatch(
                &*self.prog,
                &self.ptrs,
                &self.vals,
                Some(self.global_size),
                self.local_size,
                /*profile=*/ true,
            )
            .context(DispatchSnafu { name: self.name.clone() })?
        };
        ctx.synchronize().context(DispatchSnafu { name: self.name.clone() })?;
        Ok(handle.and_then(|h| h.timestamps_ns()).map(|(s, e)| e - s))
    }

    /// The compiled kernel's entry-point name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The resolved launch dimensions (`global`, `local`) baked at compile time.
    pub fn launch_dims(&self) -> ([usize; 3], Option<[usize; 3]>) {
        (self.global_size, self.local_size)
    }
}

/// Compile `sink` for `device` against `buffers` and return a [`CompiledLaunch`]
/// for repeated dispatch. [`launch`] is exactly `compile(..)?.dispatch(true)`;
/// factoring the render+compile out lets a benchmark loop only the dispatch.
///
/// `buffers` are compact and ordered by storage-descriptor ordinal, matching
/// `ProgramSpec.globals`. PARAM slots select signature positions and are never
/// used as indexes into this vector.
pub fn compile(device: &Device, sink: Arc<UOp>, buffers: &[Buffer]) -> Result<CompiledLaunch> {
    // This path bypasses the tensor scheduler, so size the execution pool here.
    svod_runtime::ensure_thread_pool(svod_schedule::thread_budget());
    let optimizer_renderer = match device.device {
        DeviceSpec::Cpu => svod_schedule::OptimizerRenderer::cpu(),
        DeviceSpec::Cuda { .. } => device
            .renderer
            .gpu_arch()
            .and_then(svod_dtype::GpuArch::cuda)
            .map(svod_schedule::OptimizerRenderer::for_cuda_arch)
            .unwrap_or_else(svod_schedule::OptimizerRenderer::cuda),
        DeviceSpec::Metal { .. } => svod_schedule::OptimizerRenderer::metal(),
        DeviceSpec::Amd { .. } => device
            .renderer
            .gpu_arch()
            .and_then(svod_dtype::GpuArch::amd)
            .map(svod_schedule::OptimizerRenderer::for_amd_arch)
            .unwrap_or_else(svod_schedule::OptimizerRenderer::amd_rdna3),
        _ => svod_schedule::OptimizerRenderer::cpu(),
    }
    .with_codegen_renderer(device.renderer.as_ref());
    let sink = svod_schedule::optimize_kernel_with_config(
        sink,
        &optimizer_renderer,
        &svod_schedule::OptimizerConfig::default(),
    )
    .map_err(|error| Error::Compile {
        name: "tk_kernel".to_string(),
        source: Box::new(svod_device::Error::Runtime { message: error.to_string() }),
    })?;

    // PROGRAM(sink, target arg) -> SOURCE -> BINARY (render + compile, cached nowhere
    // here; repeated launches recompile — the JIT cache lives a layer up).
    let program = program_pipeline::program_from_sink_with_renderer(sink, device.renderer.as_ref())
        .context(CompileSnafu { name: "tk_kernel".to_string() })?;
    let kernel_name = ProgramSpec::from_uop(&program).ok().map(|s| s.name).unwrap_or_else(|| "tk_kernel".into());

    let rendered = program_pipeline::get_program(
        &program,
        device.renderer.as_ref(),
        device.compiler.as_ref(),
        ProgramTarget::Source,
    )
    .context(CompileSnafu { name: kernel_name.clone() })?;

    let (compiled_program, compiled) = program_pipeline::do_compile(&rendered, device.compiler.as_ref())
        .context(CompileSnafu { name: kernel_name.clone() })?;

    let spec = ProgramSpec::from_uop(&compiled_program).context(SpecSnafu { name: kernel_name.clone() })?;
    let prog = (device.runtime)(&compiled).context(RuntimeSnafu { name: kernel_name.clone() })?;

    // Resolve compact buffer pointers in ProgramInfo.globals order. The slot is
    // retained only for diagnostics; the ordinal indexes the user vector.
    let mut ptrs: Vec<*mut u8> = Vec::with_capacity(spec.globals.len());
    for (ordinal, slot) in plan_compact_buffers(&spec.globals, buffers.len())? {
        let buf = buffers.get(ordinal).context(BufferMissingSnafu { slot, supplied: buffers.len() })?;
        buf.ensure_allocated().context(BufferAllocSnafu { slot, supplied: buffers.len() })?;
        // SAFETY: the buffer is allocated and held alive by `_buffers` below for
        // the lifetime of the `CompiledLaunch` (and thus of these raw pointers).
        ptrs.push(unsafe { buf.as_raw_ptr() });
    }

    // No symbolic vars in a hand-built kernel: empty var map yields the concrete
    // grid/block dims baked into the SPECIAL ops (or the [1,1,1] defaults).
    let var_vals: HashMap<&str, i64> = HashMap::new();
    let dims = spec.launch_dims(&var_vals).context(SpecSnafu { name: kernel_name.clone() })?;
    let vals: Vec<i64> = spec.var_names.iter().map(|n| var_vals.get(n.as_str()).copied().unwrap_or(0)).collect();

    Ok(CompiledLaunch {
        prog,
        ptrs,
        global_size: dims.global_size,
        local_size: dims.local_size,
        vals,
        name: kernel_name,
        _buffers: buffers.to_vec(),
    })
}

/// Realize `ins`, allocate/realize `outs`, then build and launch a hand-written
/// kernel against their concrete buffers — the high-level [`launch`] wrapper that
/// mirrors the `Tensor.realize(...) → sink.call(...) → run_linear` dance.
///
/// `build` receives a [`Kernel`](crate::Kernel) already bound to the realized
/// buffers (outputs first, then inputs, matching `gl()` order) and returns the
/// finished SINK (`ker.finish(..)`). Outputs are written in place — the isolation
/// path for **debugging** a kernel (read the output back with `out.as_vec()`).
///
/// ```no_run
/// use svod_tensor::Tensor;
/// use svod_dtype::{AmdArch, DType};
/// use svod_tk::{ArchCaps, run_kernel};
/// use svod_tk::kernels::matmul::{GFX1151_CFG, build_matmul_cfg};
/// let n = 256usize;
/// let a = Tensor::randn(&[n, n]).unwrap().cast(DType::BFloat16).unwrap();
/// let b = Tensor::randn(&[n, n]).unwrap().cast(DType::BFloat16).unwrap();
/// let mut c = Tensor::empty(&[n, n], DType::Float32);
/// let cfg = GFX1151_CFG;
/// let block = cfg.threads(ArchCaps::for_amd(AmdArch::Gfx1151).wave_size);
/// run_kernel("matmul", cfg.grid_dims(n), block, &mut [&mut c], &[&a, &b],
///     move |ker| { build_matmul_cfg(ker, n, cfg); ker.finish(cfg.n_accum) }).unwrap();
/// // `c` now holds A·B; inspect it with `c.as_vec::<f32>()`.
/// ```
pub fn run_kernel<F>(
    name: impl Into<String>,
    grid: [i64; 3],
    block: i64,
    outs: &mut [&mut Tensor],
    ins: &[&Tensor],
    build: F,
) -> Result<()>
where
    F: FnOnce(&crate::Kernel) -> Arc<UOp>,
{
    let compiled = compile_kernel(name, grid, block, outs, ins, build)?;
    // SAFETY: `compile_kernel` realized + allocated every bound buffer, which the
    // `CompiledLaunch` keeps alive; the synchronous dispatch holds them.
    unsafe { compiled.dispatch(true) }
}

/// Wrap a hand-built kernel SINK as a **graph node** — the `custom_kernel`
/// (`Op::Call`) analog of [`run_kernel`]. Unlike the direct path, this does NOT
/// dispatch: it returns a lazy output [`Tensor`] that the tensor scheduler
/// realizes (and the JIT graph captures) like any other op, so the kernel
/// composes into a model and benchmarks through the normal `prepare()` →
/// `execute_profiled` path.
///
/// `out` is the output template (`Tensor::empty(shape, dtype)`); `ins` are the
/// inputs. The placeholder/buffer order `custom_kernel` hands the closure is
/// `[out, ins...]`, matching the kernel's `gl()` declaration order (outputs
/// first), so `build` sees the same PARAM slots as the direct path. Launch
/// geometry rides on the `Op::Special` ops [`Kernel::new`](crate::Kernel::new)
/// mints from `grid`/`block` (no launch dims are passed through `CallInfo`); the
/// `finish()`-stamped `opts_to_apply = Some(vec![])` makes the optimizer apply
/// zero schedule opts to the hand-lowered body, which then goes through the same
/// pre/post-optimization pipeline as any other kernel.
///
/// ```no_run
/// use svod_tensor::Tensor;
/// use svod_dtype::{AmdArch, DType};
/// use svod_tk::{ArchCaps, graph_launch};
/// use svod_tk::kernels::matmul::{GFX1151_CFG, build_matmul_cfg};
/// let n = 256usize;
/// let a = Tensor::randn(&[n, n]).unwrap().cast(DType::BFloat16).unwrap();
/// let b = Tensor::randn(&[n, n]).unwrap().cast(DType::BFloat16).unwrap();
/// let out = Tensor::empty(&[n, n], DType::Float32);
/// let cfg = GFX1151_CFG;
/// let caps = ArchCaps::for_amd(AmdArch::Gfx1151);
/// // Wrap the hand-built SINK as a lazy graph node — composes + `prepare()`s like
/// // any tensor op (`build_matmul_cfg` is the worked kernel body).
/// let mut c = graph_launch("matmul", cfg.grid_dims(n), cfg.threads(caps.wave_size),
///     out, &[&a, &b], caps, move |ker| { build_matmul_cfg(ker, n, cfg); ker.finish(cfg.n_accum) }).unwrap();
/// c.prepare().unwrap();
/// ```
pub fn graph_launch<F>(
    name: impl Into<String>,
    grid: [i64; 3],
    block: i64,
    out: Tensor,
    ins: &[&Tensor],
    caps: crate::ArchCaps,
    build: F,
) -> Result<Tensor>
where
    F: FnOnce(&crate::Kernel) -> Arc<UOp>,
{
    // The grid/block → `Op::Special` launch dims + the PARAM globals are minted by
    // the tk `Kernel`; the generic graph-node wrapping (custom_kernel → Op::Call,
    // `[out, ins...]` placeholder order) lives in `Tensor::graph_kernel`. `caps`
    // (resolved by the launcher from the inputs' arch) carries the wave size /
    // WMMA descriptor the build closure threads.
    let name = name.into();
    let kname = name.clone();
    Tensor::graph_kernel(&name, out, ins, move |ph| build(&crate::Kernel::new(kname, grid, block, ph, caps)))
        .context(CustomKernelSnafu { name })
}

/// The **multi-output** peer of [`graph_launch`]: a hand-built kernel that binds
/// `outs.len()` output globals (then `ins`) and returns one lazy [`Tensor`] per
/// output. [`graph_launch`] is the one-output specialization (and stays the common
/// case); KNN needs two (the unsorted top-K `idx`/`val`).
///
/// The generic `custom_kernel` wraps EACH source in an `AFTER(callable)` and hands
/// the build closure PARAM placeholders in source order, so passing
/// `[outs..., ins...]` as the sources makes the placeholder/global order
/// `out0, out1, …, in0, in1, …` — exactly the kernel's `bind_abi` declaration order
/// (outputs first). The first `outs.len()` returned tensors are the kernel's
/// outputs (each carrying the AFTER-call dep that realizes the kernel); the trailing
/// per-input AFTER tensors are dropped.
///
/// `build` receives the placeholders and returns the finished SINK
/// (`ker.finish(n)`); the launch geometry rides on the `Op::Special` ops minted from
/// `grid`/`block`, as in [`graph_launch`].
pub fn graph_launch_multi<F>(
    name: impl Into<String>,
    grid: [i64; 3],
    block: i64,
    outs: Vec<Tensor>,
    ins: &[&Tensor],
    caps: crate::ArchCaps,
    build: F,
) -> Result<Vec<Tensor>>
where
    F: FnOnce(&crate::Kernel) -> Arc<UOp>,
{
    use svod_ir::CallInfo;

    let name = name.into();
    let kname = name.clone();
    // Sources = [outs..., ins...]; `custom_kernel_with` makes a PARAM placeholder per
    // source in that order and returns an AFTER(callable) per source. The kernel body
    // (built by the closure) sees those placeholders as PARAM slots 0.. in the same
    // order, matching the outputs-first `bind_abi`. The first source (`outs[0]`) is the
    // `self` of `custom_kernel_with`; the rest are `others`.
    snafu::ensure!(!outs.is_empty(), BufferMissingSnafu { slot: 0usize, supplied: 0usize });
    let mut others: Vec<&Tensor> = Vec::with_capacity(outs.len() - 1 + ins.len());
    others.extend(outs[1..].iter());
    others.extend(ins.iter().copied());

    let info = CallInfo { name: Some(name.clone()), ..CallInfo::default() };
    let n_out = outs.len();
    let outputs = outs[0]
        .custom_kernel_with(&others, info, move |ph| build(&crate::Kernel::new(kname, grid, block, ph, caps)))
        .context(CustomKernelSnafu { name })?;
    Ok(outputs.into_iter().take(n_out).collect())
}

/// The shared **three-way launch policy** every graph-native custom kernel follows,
/// so the arch / `None` / `Err` / `Some` decision lives in one place instead of
/// being re-threaded per kernel:
///
/// - `Ok(None)` — *doesn't apply here:* `device` isn't in `archs` (with its LLVM
///   backend), **or** `applies` is false (a runtime property — e.g. a sequence
///   length that doesn't tile). The caller substitutes its own path.
/// - `Err` — `validate` rejected the request (a FIXED shape/dtype property is wrong —
///   a caller bug), or `build` failed.
/// - `Ok(Some(out))` — the kernel ran, yielding the lazy output [`Tensor`].
///
/// `validate`, `applies`, and `build` receive the resolved [`GpuArch`] for
/// arch-specific constraints / configs; `applies` is a predicate (kept out of
/// `validate` because failing it is a fallback trigger, not an error) — taking
/// the arch lets arch-dependent fit rules (e.g. a head dim the wave size must
/// divide) decline to `Ok(None)` instead of masquerading as caller bugs.
/// Shared by [`crate::matmul`] and [`crate::flash_attention_with`].
pub fn launch_custom(
    device: &DeviceSpec,
    archs: ArchSet,
    validate: impl FnOnce(GpuArch) -> Result<()>,
    applies: impl FnOnce(GpuArch) -> bool,
    build: impl FnOnce(GpuArch) -> Result<Tensor>,
) -> Result<Option<Tensor>> {
    // "Can this device run the kernel at all?" — wrong arch / missing toolchain is
    // environmental, so `None` (the caller's fallback), never an error.
    let Some(arch) = crate::target::resolve_supported_arch(device, archs).ok() else {
        return Ok(None);
    };
    // "Is the request structurally valid?" — a fixed-property violation is a caller bug.
    validate(arch)?;
    // "Does this runtime instance fit?" — if not, a fallback trigger, not an error.
    if !applies(arch) {
        return Ok(None);
    }
    build(arch).map(Some)
}

/// Realize `ins`, allocate/realize `outs`, build a hand-written kernel against
/// their concrete buffers, and **compile it once** into a [`CompiledLaunch`] for
/// repeated dispatch — the compile-once analog of [`run_kernel`].
///
/// [`run_kernel`] is exactly `compile_kernel(..)?.dispatch(true)`. Splitting the
/// compile out lets a benchmark build + compile a kernel once and then loop only
/// [`CompiledLaunch::dispatch`], excluding render/compile from the timed region.
/// Outputs are written in place on each dispatch (the bound output buffer is
/// registered to its tensor, so `out.as_vec()` reads the kernel's last write).
///
/// ```no_run
/// use svod_tensor::Tensor;
/// use svod_dtype::{AmdArch, DType};
/// use svod_tk::{ArchCaps, compile_kernel};
/// use svod_tk::kernels::matmul::{GFX1151_CFG, build_matmul_cfg};
/// let n = 256usize;
/// let a = Tensor::randn(&[n, n]).unwrap().cast(DType::BFloat16).unwrap();
/// let b = Tensor::randn(&[n, n]).unwrap().cast(DType::BFloat16).unwrap();
/// let mut c = Tensor::empty(&[n, n], DType::Float32);
/// let cfg = GFX1151_CFG;
/// let block = cfg.threads(ArchCaps::for_amd(AmdArch::Gfx1151).wave_size);
/// // Render + compile ONCE …
/// let compiled = compile_kernel("matmul", cfg.grid_dims(n), block, &mut [&mut c], &[&a, &b],
///     move |ker| { build_matmul_cfg(ker, n, cfg); ker.finish(cfg.n_accum) }).unwrap();
/// // … then re-dispatch in a hot loop (the timed region excludes compilation).
/// for _ in 0..100 {
///     // SAFETY: the bound buffers stay allocated for `compiled`'s lifetime.
///     unsafe { compiled.dispatch(true) }.unwrap();
/// }
/// ```
pub fn compile_kernel<F>(
    name: impl Into<String>,
    grid: [i64; 3],
    block: i64,
    outs: &mut [&mut Tensor],
    ins: &[&Tensor],
    build: F,
) -> Result<CompiledLaunch>
where
    F: FnOnce(&crate::Kernel) -> Arc<UOp>,
{
    // Inputs must hold concrete DATA: realize (compute) any lazy graph first.
    // Outputs are allocated fresh by `realize_buffer` below.
    for t in ins.iter() {
        // Inputs are immutable refs; realize via a clone that shares the entry/buffer.
        let mut t = (*t).clone();
        t.realize().context(RealizeSnafu)?;
    }

    // Gather concrete buffers + their BUFFER UOps in ABI declaration order.
    let mut buffers: Vec<Buffer> = Vec::with_capacity(outs.len() + ins.len());
    let mut buf_uops: Vec<Arc<UOp>> = Vec::with_capacity(outs.len() + ins.len());
    for t in outs.iter() {
        buffers.push(realize_buffer(t)?);
        buf_uops.push(t.uop().base());
    }
    for t in ins.iter() {
        buffers.push(realize_buffer(t)?);
        buf_uops.push(t.uop().base());
    }

    // The device is resolved from `buffers[0]`, so a kernel launched with no
    // outputs AND no inputs has nothing to resolve against — a structured error,
    // not an index-out-of-bounds panic on the public DEBUG-face entry.
    snafu::ensure!(!buffers.is_empty(), BufferMissingSnafu { slot: 0usize, supplied: 0usize });

    // Resolve the concrete Device (renderer/compiler/runtime) for the buffers'
    // device, honoring the env-selected CPU backend like the realize path does.
    let device_spec = buffers[0].allocator().device_spec();
    let device = svod_runtime::DEVICE_FACTORIES
        .device(&device_spec, svod_device::registry::registry())
        .context(DeviceFactorySnafu { spec: format!("{device_spec:?}") })?;

    // Caps from the realized buffers' arch; a host render target falls back to
    // gfx942 so the WMMA descriptor still resolves.
    let caps =
        crate::target::resolve_arch(&device_spec).map(crate::ArchCaps::for_arch).unwrap_or(crate::ArchCaps::GFX942);
    let ker = crate::Kernel::new(name, grid, block, buf_uops, caps);
    let sink = build(&ker);
    compile(&device, sink, &buffers)
}

/// Fetch a tensor's concrete buffer, allocating + registering one on demand.
///
/// `Tensor::empty(..)` mints a BUFFER UOp but no backing allocation (svod's
/// `realize` short-circuits for buffer-identity tensors without allocating), so
/// the direct-launch path must materialize output buffers itself — the svod
/// analog of tinygrad's `b.allocate()` inside `run_linear`. Inputs already carry
/// a buffer (`from_slice`), so this only allocates for fresh outputs.
pub fn realize_buffer(t: &Tensor) -> Result<Buffer> {
    if let Some(buf) = t.buffer() {
        buf.ensure_allocated().context(BufferAllocSnafu { slot: 0usize, supplied: 0usize })?;
        return Ok(buf);
    }

    // No backing buffer: allocate one sized to the tensor's logical shape on its
    // BUFFER UOp's device, then register it so `t.buffer()` resolves it.
    let base = t.uop().base();
    let svod_ir::Op::Buffer(ops::Buffer { arg, .. }) = base.op() else {
        return Err(RealizeSnafu.into_error(svod_tensor::error::Error::NoBuffer));
    };
    let Some(spec) = arg.device.as_ref() else {
        return Err(RealizeSnafu.into_error(svod_tensor::error::Error::NoBuffer));
    };
    let size = base.buffer_size().ok_or_else(|| RealizeSnafu.into_error(svod_tensor::error::Error::NoBuffer))?;
    let dtype = base.dtype();
    let shape: Vec<usize> = t
        .shape()
        .ok()
        .and_then(|s| s.iter().map(|d| d.as_const()).collect::<Option<Vec<_>>>())
        .unwrap_or_else(|| vec![size]);

    let allocator =
        svod_device::registry::registry().get(spec).context(AllocatorSnafu { spec: format!("{spec:?}") })?;
    let buffer = Buffer::allocate(allocator, dtype, shape, Default::default())
        .context(BufferAllocSnafu { slot: 0usize, supplied: 0usize })?;
    let buffer = Arc::new(buffer);
    svod_tensor::tensor_registry::register_buffer_by_uop_id(base.id, buffer.clone());
    Ok((*buffer).clone())
}
