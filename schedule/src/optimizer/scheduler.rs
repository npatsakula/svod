//! Scheduler for kernel optimization.
//!
//! The `Scheduler` manages kernel optimization state and applies transformation primitives (OptOps)
//! to improve performance on specific backends.

use std::cell::OnceCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

use svod_ir::{AxisId, AxisType, Op, UOp, UOpKey};

use super::error::*;
use super::renderer::Renderer;
use super::types::{Opt, OptOps};
use svod_ir::ops;

/// Global kernel name counter for deduplication.
///
/// Tracks how many times each kernel name has been generated to avoid collisions.
/// When multiple kernels have the same shape, subsequent ones get suffixed with "n0", "n1", etc.
static KERNEL_NAME_COUNTS: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();

/// Get the kernel name counts map, initializing if needed.
fn kernel_name_counts() -> &'static Mutex<HashMap<String, usize>> {
    KERNEL_NAME_COUNTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Clear kernel name counts (for testing).
#[cfg(test)]
pub fn clear_kernel_name_counts() {
    if let Some(counts) = KERNEL_NAME_COUNTS.get() {
        counts.lock().unwrap().clear();
    }
}

/// Make a shape name unique for this process: the n-th kernel sharing a
/// function name gets the `n{n-1}` suffix. The suffix is the only part of a
/// kernel's identity that depends on the order kernels are finished in.
pub fn unique_kernel_name(shape_name: &str) -> String {
    let mut counts = kernel_name_counts().lock().unwrap();
    let count = counts.entry(svod_ir::to_function_name(shape_name)).or_insert(0);
    *count += 1;
    if *count > 1 { format!("{shape_name}n{}", *count - 1) } else { shape_name.to_string() }
}

/// When a finished kernel receives its unique name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelNaming {
    /// Suffix the shape name through the process-wide counter immediately.
    Unique,
    /// Leave the bare shape name; the caller applies [`finalize_kernel_name`]
    /// once it knows the order — kernels optimized concurrently would
    /// otherwise draw suffixes in scheduling-dependent order.
    Deferred,
}

/// Give a kernel finished under [`KernelNaming::Deferred`] its unique name, on
/// both channels the optimizer writes it to: the SINK's structural
/// `KernelInfo.name` and the attached optimizer metadata. Kernels without
/// optimizer metadata were never named and stay as they are. A hand-authored
/// name goes through the same counter, so two different kernels launched
/// under one name stay distinguishable in profiles and IR dumps.
pub fn finalize_kernel_name(ast: &Arc<UOp>) -> Arc<UOp> {
    use crate::optimizer::KernelInfo;
    let Some(info) = ast.metadata::<KernelInfo>() else { return ast.clone() };
    let name = unique_kernel_name(&info.name);
    let renamed = match ast.op() {
        Op::Sink(ops::Sink { sources, info }) => {
            let mut structural = info.clone().unwrap_or_default();
            structural.name = Some(name.clone());
            UOp::sink_with_info(sources.iter().cloned().collect(), structural)
                .rtag(ast.tag().clone())
                .rorigin(ast.origin())
        }
        _ => ast.clone(),
    };
    renamed.with_metadata(KernelInfo { name, ..(*info).clone() })
}

/// Scheduler for kernel optimization.
///
/// Manages the optimization state of a kernel, including:
/// - The UOp AST being optimized
/// - Backend capabilities (Renderer)
/// - Applied optimizations history
/// - Cached properties (ranges, shapes, etc.)
///
/// # Architecture
///
/// The Scheduler is the central component of the optimization layer:
/// 1. Created from a kernel AST + backend Renderer
/// 2. Applies OptOps via `apply_opt()` or heuristics
/// 3. Each optimization may create new ranges or modify existing ones
/// 4. Caches are cleared after mutations to ensure consistency
/// 5. Final optimized AST retrieved via `get_optimized_ast()`
///
/// # Example
///
/// ```ignore
/// let renderer = Renderer::cuda();
/// let mut scheduler = Scheduler::new(kernel_ast, renderer);
///
/// // Initial parallelization
/// scheduler.convert_loop_to_global()?;
///
/// // Apply optimizations
/// scheduler.apply_opt(Opt::upcast(0, 4))?;  // Stack by 4
/// scheduler.apply_opt(Opt::local(1, 16))?;  // 16 threads per workgroup
///
/// // Get result
/// let optimized = scheduler.get_optimized_ast(None);
/// ```
pub struct Scheduler {
    /// The kernel AST being optimized.
    ///
    /// This is the root UOp representing the computation. Immutable during the lifetime
    /// of a Scheduler instance - transformations create new ASTs.
    ast: Arc<UOp>,

    /// Backend renderer capabilities.
    ///
    /// Describes what optimizations the target backend supports and enforces device limits.
    pub ren: Renderer,

    /// Whether local memory usage is disabled.
    ///
    /// Set to true by NOLOCALS opt or if backend doesn't support local memory.
    pub dont_use_locals: bool,

    /// History of applied optimizations.
    ///
    /// Used for debugging, kernel naming, and potential undo functionality.
    pub applied_opts: Vec<Opt>,

    /// Index of the actively-selected TensorCore in `ren.tensor_cores`,
    /// set by `apply_with_axis_choice` when an `OptOps::TC` is applied.
    /// Used by beam's `validate_limits` to compute the correct `tc_up`
    /// divisor when the renderer offers multiple TC variants (e.g. f32,
    /// f16, bf16). `None` until a TC is applied.
    pub selected_tc_index: Option<usize>,

    // Cached properties (computed lazily, cleared by set_ast/clear_caches)
    /// Cached list of all RANGE operations, sorted by (axis_type.priority(), axis_id).
    rngs_cache: OnceCell<Vec<Arc<UOp>>>,
    /// Cached maximum axis_id used in any RANGE.
    maxarg_cache: OnceCell<usize>,
    /// Cached toposort of the AST (avoids repeated O(N) traversals).
    toposort_cache: OnceCell<Vec<Arc<UOp>>>,
    /// Cached REDUCE operations from the AST.
    reduceops_cache: OnceCell<Vec<Arc<UOp>>>,
    /// Cached INDEX (buffer access) operations from the AST.
    bufs_cache: OnceCell<Vec<Arc<UOp>>>,
}

impl Scheduler {
    /// Create a new Scheduler for the given kernel AST and backend.
    ///
    /// # Arguments
    ///
    /// * `ast` - The kernel UOp AST to optimize (typically from rangeify phase 5)
    /// * `ren` - Backend renderer describing capabilities and limits
    ///
    /// # Returns
    ///
    /// A new Scheduler with empty optimization history and cleared caches.
    pub fn new(ast: Arc<UOp>, ren: Renderer) -> Self {
        Self {
            ast,
            ren,
            dont_use_locals: false,
            applied_opts: Vec::new(),
            selected_tc_index: None,
            rngs_cache: OnceCell::new(),
            maxarg_cache: OnceCell::new(),
            toposort_cache: OnceCell::new(),
            reduceops_cache: OnceCell::new(),
            bufs_cache: OnceCell::new(),
        }
    }

    /// Get a reference to the current AST.
    pub fn ast(&self) -> &Arc<UOp> {
        &self.ast
    }

    /// Set the AST to a new value and clear caches.
    ///
    /// Used by optimization operations that transform the AST.
    pub(crate) fn set_ast(&mut self, ast: Arc<UOp>) {
        self.ast = ast;
        self.clear_caches();
    }

    /// Clear all cached properties.
    ///
    /// Must be called after any mutation to the AST to ensure consistency.
    pub(crate) fn clear_caches(&mut self) {
        self.rngs_cache.take();
        self.maxarg_cache.take();
        self.toposort_cache.take();
        self.reduceops_cache.take();
        self.bufs_cache.take();
    }

    /// Get the list of all RANGE operations, sorted by (axis_type.priority(), axis_id).
    ///
    /// Ranges are the fundamental unit of loop structure in the kernel. They are sorted
    /// to determine nesting order: lower priority types are outer loops.
    ///
    /// **Sorting order:**
    /// - Primary: `axis_type.priority()` (Loop=-1, Global/Thread=0, Warp=1, Local/GroupReduce=2, Upcast=3, Reduce=4, Unroll=5)
    /// - Secondary: `axis_id` (ascending)
    ///
    /// # Returns
    ///
    /// Cached slice of RANGE UOps in canonical order.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let rngs = scheduler.rngs();
    /// for (i, rng) in rngs.iter().enumerate() {
    ///     println!("Axis {}: {:?} size={}", i, rng.axis_type(), rng.size());
    /// }
    /// ```
    pub fn rngs(&self) -> &[Arc<UOp>] {
        self.rngs_cache.get_or_init(|| self.compute_rngs())
    }

    /// Compute the list of RANGE operations and sort them.
    ///
    /// This is called lazily the first time `rngs()` is accessed.
    fn compute_rngs(&self) -> Vec<Arc<UOp>> {
        // Collect all RANGE nodes via toposort. Filter out size-1 ranges
        // (where vmax == 0) so Global(1), Local(1), etc. axes are excluded.
        let mut ranges: Vec<Arc<UOp>> = self
            .ast
            .toposort()
            .into_iter()
            .filter(|node| {
                if let Op::Range(..) = node.op() {
                    // Include only ranges with vmax > 0 (size > 1)
                    // vmax = size - 1, so vmax > 0 means size > 1
                    use svod_ir::ConstValue;
                    match node.vmax() {
                        ConstValue::Int(v) => *v > 0,
                        ConstValue::UInt(v) => *v > 0,
                        _ => false, // Symbolic or unknown sizes are excluded for safety
                    }
                } else {
                    false
                }
            })
            .collect();

        // Sort by (axis_type.priority(), axis_id)
        ranges.sort_by_key(|rng| {
            if let Op::Range(ops::Range { axis_id, axis_type, .. }) = rng.op() {
                (axis_type.priority(), axis_id.clone())
            } else {
                unreachable!("Filtered to only Range ops")
            }
        });

        ranges
    }

    /// Get the number of dimensions (ranges) in the kernel.
    pub fn shape_len(&self) -> usize {
        self.rngs().len()
    }

    /// Get the maximum axis_id used in any RANGE operation.
    ///
    /// This is used when creating new ranges to ensure unique axis_ids.
    ///
    /// # Returns
    ///
    /// The highest axis_id, or 0 if no ranges exist.
    pub fn maxarg(&self) -> usize {
        *self.maxarg_cache.get_or_init(|| {
            self.rngs()
                .iter()
                .filter_map(|rng| {
                    if let Op::Range(ops::Range { axis_id, .. }) = rng.op() { Some(axis_id.value()) } else { None }
                })
                .max()
                .unwrap_or(0)
        })
    }

    /// Find the first REDUCE operation in the kernel.
    ///
    /// Used to determine if this is a reduction kernel and to extract reduction properties.
    ///
    /// # Returns
    ///
    /// The first REDUCE UOp found via toposort, or None if no reductions exist.
    /// Cached toposort of the AST.
    fn ast_toposort(&self) -> &[Arc<UOp>] {
        self.toposort_cache.get_or_init(|| self.ast.toposort())
    }

    pub fn reduceop(&self) -> Option<Arc<UOp>> {
        self.reduceops().first().cloned()
    }

    pub fn reduceops(&self) -> &[Arc<UOp>] {
        self.reduceops_cache
            .get_or_init(|| self.ast_toposort().iter().filter(|n| matches!(n.op(), Op::Reduce(..))).cloned().collect())
    }

    pub fn bufs(&self) -> &[Arc<UOp>] {
        self.bufs_cache
            .get_or_init(|| self.ast_toposort().iter().filter(|n| matches!(n.op(), Op::Index(..))).cloned().collect())
    }

    /// Get the output shape (dimensions without reduction axes).
    ///
    /// This is the shape of the final result tensor, excluding any REDUCE/UNROLL/GROUP_REDUCE axes.
    ///
    /// # Returns
    ///
    /// Vector of sizes for each non-reduction dimension.
    pub fn output_shape(&self) -> Vec<i64> {
        self.rngs()
            .iter()
            .filter(
                |rng| {
                    if let Op::Range(ops::Range { axis_type, .. }) = rng.op() { !axis_type.is_reduce() } else { false }
                },
            )
            .filter_map(|rng| {
                if let Op::Range(ops::Range { end, .. }) = rng.op()
                    && let Op::Const(cv) = end.op()
                    && let svod_ir::ConstValue::Int(sz) = cv.0
                {
                    return Some(sz);
                }
                None
            })
            .collect()
    }

    /// Get the full shape including all axes (global, local, reduce, upcast, etc.).
    ///
    /// Returns the sizes of all dimension ranges in order. Returns -1 for symbolic/unknown sizes.
    ///
    /// Used by heuristics to calculate total work and make optimization decisions.
    pub fn full_shape(&self) -> Vec<i64> {
        self.rngs()
            .iter()
            .map(|rng| {
                if let Op::Range(ops::Range { end, .. }) = rng.op()
                    && let Op::Const(cv) = end.op()
                    && let svod_ir::ConstValue::Int(sz) = cv.0
                {
                    sz
                } else {
                    -1 // Symbolic or unknown size
                }
            })
            .collect()
    }

    /// Check if any axes have been upcasted.
    ///
    /// Returns true if there are any UPCAST or UNROLL axis types in the kernel.
    /// Used by heuristics to avoid redundant upcasting.
    pub fn upcasted(&self) -> bool {
        !self.axes_of(&[AxisType::Upcast, AxisType::Unroll]).is_empty()
    }

    /// Get a reference to the backend renderer.
    ///
    /// Returns the renderer that describes backend capabilities and constraints.
    /// Used by heuristics to check device features and limits.
    pub fn renderer(&self) -> &Renderer {
        &self.ren
    }

    /// Calculate the total upcast size (product of all UPCAST dimensions).
    ///
    /// Upcast size represents vectorization width - how many elements are processed
    /// per loop iteration. Typical values: 1 (no upcast), 2, 4, 8, 16.
    ///
    /// # Returns
    ///
    /// Product of all UPCAST and UNROLL dimension sizes, or 1 if none exist.
    ///
    /// Used as a guard to prevent exponential vector width growth from multiple
    /// unroll sources (K-vectorization, output-upcast, general unrolling).
    pub fn upcast_size(&self) -> usize {
        self.rngs()
            .iter()
            .filter(|rng| {
                if let Op::Range(ops::Range { axis_type, .. }) = rng.op() {
                    matches!(axis_type, AxisType::Upcast | AxisType::Unroll)
                } else {
                    false
                }
            })
            .filter_map(|rng| {
                if let Op::Range(ops::Range { end, .. }) = rng.op()
                    && let Op::Const(cv) = end.op()
                    && let svod_ir::ConstValue::Int(sz) = cv.0
                {
                    return Some(sz as usize);
                }
                None
            })
            .product()
    }

    /// Count the number of GROUP_REDUCE axes.
    ///
    /// GROUP_REDUCE represents two-stage reductions with shared memory synchronization.
    /// Each GROUP_REDUCE axis adds a synchronization barrier.
    ///
    /// # Returns
    ///
    /// Number of GROUP_REDUCE axes (typically 0 or 1).
    pub fn group_for_reduces(&self) -> usize {
        self.rngs()
            .iter()
            .filter(|rng| {
                if let Op::Range(ops::Range { axis_type, .. }) = rng.op() {
                    *axis_type == AxisType::GroupReduce
                } else {
                    false
                }
            })
            .count()
    }

    /// Get indices of axes matching any of the given types.
    ///
    /// This is useful for filtering operations that only apply to certain axis types.
    ///
    /// # Arguments
    ///
    /// * `types` - Slice of AxisTypes to match against
    ///
    /// # Returns
    ///
    /// Vector of indices into `rngs()` for matching axes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Get all reduce axes (REDUCE, UNROLL, GROUP_REDUCE)
    /// let reduce_axes = scheduler.axes_of(&[
    ///     AxisType::Reduce,
    ///     AxisType::Unroll,
    ///     AxisType::GroupReduce,
    /// ]);
    /// ```
    pub fn axes_of(&self, types: &[AxisType]) -> Vec<usize> {
        self.rngs()
            .iter()
            .enumerate()
            .filter_map(|(i, rng)| {
                if let Op::Range(ops::Range { axis_type, .. }) = rng.op() {
                    if types.contains(axis_type) { Some(i) } else { None }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get Range UOps matching any of the given types.
    ///
    /// Similar to `axes_of()` but returns the actual Range UOps instead of indices.
    ///
    /// # Arguments
    ///
    /// * `types` - Slice of AxisTypes to match against
    ///
    /// # Returns
    ///
    /// Vector of Range UOps with matching axis types.
    pub fn ranges_of(&self, types: &[AxisType]) -> Vec<Arc<UOp>> {
        self.axes_of(types).into_iter().map(|i| self.rngs()[i].clone()).collect()
    }

    /// Get indices of axes that can be upcasted (vectorized).
    ///
    /// Upcastable axes are GLOBAL, LOCAL, or LOOP axes with size > 1. REDUCE
    /// axes use UNROLL instead.
    ///
    /// # Returns
    ///
    /// Vector of indices into `rngs()` for upcastable axes, sorted by position.
    pub fn upcastable_dims(&self) -> Vec<usize> {
        self.rngs()
            .iter()
            .enumerate()
            .filter_map(|(i, rng)| {
                if let Op::Range(ops::Range { axis_type, end, .. }) = rng.op() {
                    if !matches!(axis_type, AxisType::Global | AxisType::Local | AxisType::Weak) {
                        return None;
                    }

                    // Check size > 1
                    if let Op::Const(cv) = end.op()
                        && let svod_ir::ConstValue::Int(sz) = cv.0
                        && sz > 1
                    {
                        return Some(i);
                    }

                    None
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get indices of axes that can be unrolled.
    ///
    /// Unrollable axes are GROUP_REDUCE or REDUCE axes with size > 1.
    /// These represent reduction loops that can be unrolled for better ILP.
    ///
    /// # Returns
    ///
    /// Vector of indices into `rngs()` for unrollable axes.
    pub fn unrollable_dims(&self) -> Vec<usize> {
        self.rngs()
            .iter()
            .enumerate()
            .filter_map(|(i, rng)| {
                if let Op::Range(ops::Range { axis_type, end, .. }) = rng.op() {
                    // Check type
                    if !matches!(axis_type, AxisType::GroupReduce | AxisType::Reduce) {
                        return None;
                    }

                    // Check size > 1
                    if let Op::Const(cv) = end.op()
                        && let svod_ir::ConstValue::Int(sz) = cv.0
                        && sz > 1
                    {
                        return Some(i);
                    }

                    None
                } else {
                    None
                }
            })
            .collect()
    }

    /// Map logical axis index to physical axis index.
    ///
    /// Different OptOps use different logical axis numbering schemes:
    /// - Most ops: Direct index into `rngs()`
    /// - UNROLL: Index into `unrollable_dims()` (only reduction axes)
    /// - GROUP/GROUPTOP: Index into `axes_of([REDUCE])` (only REDUCE axes)
    /// - TC: Returns -1 (no single axis)
    ///
    /// # Arguments
    ///
    /// * `op` - The optimization operation type
    /// * `axis` - The logical axis index (if applicable)
    ///
    /// # Returns
    ///
    /// Physical axis index into `rngs()`, or -1 for TC operations.
    ///
    /// # Errors
    ///
    /// Returns `OptError::AxisOutOfBounds` if the logical axis is out of range.
    pub fn real_axis(&self, op: OptOps, axis: Option<usize>) -> Result<isize, OptError> {
        match op {
            // TC doesn't operate on a single axis
            OptOps::TC => Ok(-1),

            // NOLOCALS doesn't use axis
            OptOps::NOLOCALS => Ok(-1),

            // UNROLL uses logical index into unrollable dims
            OptOps::UNROLL => {
                let axis = axis.ok_or(OptError::MissingAxisParameter)?;

                let unrollable = self.unrollable_dims();
                let real_idx =
                    *unrollable.get(axis).ok_or(OptError::AxisOutOfBounds { axis, max: unrollable.len() })?;

                Ok(real_idx as isize)
            }

            // GROUP/GROUPTOP use logical index into REDUCE axes
            OptOps::GROUP | OptOps::GROUPTOP => {
                let axis = axis.ok_or(OptError::MissingAxisParameter)?;

                let reduce_axes = self.axes_of(&[AxisType::Reduce]);
                let real_idx =
                    *reduce_axes.get(axis).ok_or(OptError::AxisOutOfBounds { axis, max: reduce_axes.len() })?;

                Ok(real_idx as isize)
            }

            // All other ops use direct axis index
            _ => {
                let axis = axis.ok_or(OptError::MissingAxisParameter)?;

                if axis >= self.shape_len() {
                    return Err(OptError::AxisOutOfBounds { axis, max: self.shape_len() });
                }

                Ok(axis as isize)
            }
        }
    }

    /// Get a colored string representation of the kernel shape.
    ///
    /// Each axis is represented by its type letter and size:
    /// - Loop: 'L'
    /// - Global: 'g'
    /// - Thread: 't'
    /// - Warp: 'w'
    /// - Local: 'l'
    /// - GroupReduce: 'G'
    /// - Upcast: 'u'
    /// - Reduce: 'R'
    /// - Unroll: 'r'
    ///
    /// # Returns
    ///
    /// String like "g16l8R32u4" (Global 16, Local 8, Reduce 32, Upcast 4).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let shape = scheduler.colored_shape();
    /// println!("Kernel: {}", shape); // "g16l16R32u4"
    /// ```
    pub fn colored_shape(&self) -> String {
        self.rngs()
            .iter()
            .filter_map(|rng| {
                if let Op::Range(ops::Range { axis_type, end, .. }) = rng.op() {
                    // Get size
                    if let Op::Const(cv) = end.op()
                        && let svod_ir::ConstValue::Int(sz) = cv.0
                    {
                        return Some(format!("{}{}", axis_type.letter(), sz));
                    }
                    // Symbolic size
                    Some(format!("{}?", axis_type.letter()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get a vector of string representations for each axis.
    ///
    /// Similar to `colored_shape()` but returns individual strings per axis.
    ///
    /// # Returns
    ///
    /// Vector like ["g16", "l8", "R32", "u4"].
    pub fn shape_str(&self) -> Vec<String> {
        self.rngs()
            .iter()
            .filter_map(|rng| {
                if let Op::Range(ops::Range { axis_type, end, .. }) = rng.op() {
                    if let Op::Const(cv) = end.op()
                        && let svod_ir::ConstValue::Int(sz) = cv.0
                    {
                        return Some(format!("{}{}", axis_type.letter(), sz));
                    }
                    Some(format!("{}?", axis_type.letter()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the kernel type prefix for naming.
    ///
    /// - "r" for reduction kernels (has REDUCE op)
    /// - "E" for elementwise kernels
    ///
    /// # Returns
    ///
    /// Single character string representing kernel type.
    pub fn kernel_type(&self) -> &'static str {
        if self.reduceop().is_some() { "r" } else { "E" }
    }

    /// Core transformation: split a range into two dimensions.
    ///
    /// This is the fundamental operation used by all OptOps. It splits a single range
    /// of size `old_sz * amount` into two ranges: one of size `old_sz` (reduced original)
    /// and one of size `amount` (new range with `new_type`).
    ///
    /// The `top` parameter controls iteration order and affects memory access patterns.
    ///
    /// # Algorithm
    ///
    /// 1. **Validate divisibility:** compute the symbolic quotient `old_end =
    ///    rng.end / amount` via [`UOp::divides`]. Const sizes resolve to
    ///    smaller consts; symbolic sizes (e.g. `T * 4`) keep the symbolic
    ///    factor in the quotient.
    /// 2. **Create new range:** either use `input_new_rng` or create one with `new_type`
    /// 3. **Create reduced old range:** with `old_end` (possibly symbolic) as its size
    /// 4. **Compute substitution:**
    ///    - If `top=true`: `new_rng * old_end + replaced_rng` (new varies faster)
    ///    - If `top=false`: `replaced_rng * amount + new_rng` (old varies faster)
    /// 5. **Substitute** in AST and clear caches
    /// 6. **Return** both ranges for further transformations
    ///
    /// # Arguments
    ///
    /// * `rng` - The range to split (must be divisible by `amount`)
    /// * `amount` - The size of the new dimension
    /// * `new_type` - The AxisType for the new range (e.g., Upcast, Local)
    /// * `top` - If true, new range is outer loop; if false, new range is inner loop
    /// * `input_new_rng` - Optional pre-created range (used for specific axis_id control)
    ///
    /// # Returns
    ///
    /// `(replaced_rng, new_rng)` - The reduced old range and the new range
    ///
    /// # Errors
    ///
    /// Returns [`OptError::Division`] / [`OptError::SymbolicDivision`] if
    /// `amount` cannot be proved to divide `rng.end`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Split a Global(16) into Global(4) and Upcast(4)
    /// let global_16 = rngs[0].clone();
    /// let (global_4, upcast_4) = scheduler.shift_to(
    ///     global_16,
    ///     4,              // amount
    ///     AxisType::Upcast,
    ///     false,          // upcast is inner (varies faster)
    ///     None,
    /// )?;
    /// // Result: iteration order is [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]
    /// ```
    pub(crate) fn shift_to(
        &mut self,
        rng: Arc<UOp>,
        amount: usize,
        new_type: AxisType,
        top: bool,
        input_new_rng: Option<Arc<UOp>>,
    ) -> Result<(Arc<UOp>, Arc<UOp>), OptError> {
        use std::collections::HashMap;
        use svod_ir::{ConstValue, UOpKey};

        // 1. Validate divisibility and compute symbolic quotient end. When the
        // axis size is symbolic (e.g. `T*4`), the quotient is a UOp (e.g. `T`)
        // that must propagate through the substituted index expression — not a
        // collapsed integer that drops the symbolic factor.
        let old_end = if let Op::Range(ops::Range { end, .. }) = rng.op() {
            end.divides(amount as i64).ok_or_else(|| {
                if let Op::Const(cv) = end.op()
                    && let ConstValue::Int(sz) = cv.0
                {
                    DivisionSnafu { size: sz as usize, amount }.build()
                } else {
                    SymbolicDivisionSnafu { amount }.build()
                }
            })?
        } else {
            return ExpectedRangeOperationSnafu.fail();
        };

        // 2. Create new range
        let new_rng = input_new_rng.unwrap_or_else(|| {
            let end = rng.const_like(amount as i64);
            UOp::range_axis(end, AxisId::Renumbered(self.maxarg() + 1), new_type)
        });

        // 3. Create reduced old range (same axis_id and type, symbolic end allowed)
        let replaced_rng = if let Op::Range(ops::Range { axis_id, axis_type, .. }) = rng.op() {
            UOp::range_axis(old_end.clone(), axis_id.clone(), *axis_type)
        } else {
            return ExpectedRangeOperationSnafu.fail();
        };

        // 4. Compute substitution expression
        let sub_axis = if top {
            // Top order: new varies faster
            // Example: [0,8,16,24, 1,9,17,25, ...]
            new_rng
                .try_mul(&old_end)
                .expect("Multiplication should not fail for index types")
                .try_add(&replaced_rng)
                .expect("Addition should not fail for index types")
        } else {
            // Bottom order: old varies faster
            // Example: [0,1,2,3, 4,5,6,7, 8,9,10,11, ...]
            let amount_uop = replaced_rng.const_like(amount as i64);
            replaced_rng
                .try_mul(&amount_uop)
                .expect("Multiplication should not fail for index types")
                .try_add(&new_rng)
                .expect("Addition should not fail for index types")
        };

        // 5. Perform substitution
        let mut subst_map = HashMap::new();
        subst_map.insert(UOpKey(rng), sub_axis);

        self.ast = self.ast.substitute(&subst_map);

        // Clear caches (maxarg will be recomputed on next access)
        self.clear_caches();

        // 6. Return both ranges
        Ok((replaced_rng, new_rng))
    }

    // ==== Phase 7: Initialization & Finalization ====

    /// Get all ranges from output operations (excluding REDUCE axes).
    ///
    /// Returns ranges that appear in output buffers — candidates for
    /// parallelization since they represent independent output elements.
    fn output_rngs(&self) -> Vec<Arc<UOp>> {
        // Find all STORE operations (outputs)
        let stores: Vec<_> =
            self.ast.toposort().into_iter().filter(|node| matches!(node.op(), Op::Store(..) | Op::Sink(..))).collect();

        if stores.is_empty() {
            return vec![];
        }

        // `ranges()` is the cached RangesProperty: every RANGE in the backward
        // slice, already deduplicated. Avoids a fresh DFS per store.
        let mut output_ranges: Vec<Arc<UOp>> = Vec::new();
        for store in stores {
            for range in store.ranges() {
                if matches!(range.op(), Op::Range(ops::Range { axis_type, .. }) if *axis_type != AxisType::Reduce)
                    && !output_ranges.iter().any(|r| Arc::ptr_eq(r, &range))
                {
                    output_ranges.push(range.clone());
                }
            }
        }

        output_ranges
    }

    /// Get WEAK ranges that can be safely parallelized to GLOBAL.
    ///
    /// A range is globalizable if:
    /// 1. It's currently a WEAK axis
    /// 2. It appears in all output operations (STORE nodes)
    ///
    /// This ensures parallelizing the range won't cause race conditions.
    pub(crate) fn globalizable_rngs(&self) -> Vec<Arc<UOp>> {
        // Start with WEAK axes from outputs
        let mut candidates: Vec<_> = self
            .output_rngs()
            .into_iter()
            .filter(|r| {
                if let Op::Range(ops::Range { axis_type, .. }) = r.op() { *axis_type == AxisType::Weak } else { false }
            })
            .collect();

        // Find all STORE and SINK operations
        let stores: Vec<_> =
            self.ast.toposort().into_iter().filter(|node| matches!(node.op(), Op::Store(..) | Op::Sink(..))).collect();

        if stores.is_empty() {
            return candidates;
        }

        // Keep only ranges that appear in ALL stores
        for store in &stores {
            let store_ranges = store.ranges();
            candidates.retain(|candidate| store_ranges.iter().any(|r| Arc::ptr_eq(r, candidate)));
        }

        candidates
    }

    /// Convert eligible WEAK axes to GLOBAL for parallelization.
    ///
    /// Identifies which loops can be safely parallelized and converts them to
    /// GLOBAL (GPU thread) axes. Only applicable for GPU backends (has_local=true).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let renderer = Renderer::cuda();
    /// let mut scheduler = Scheduler::new(ast, renderer);
    /// scheduler.convert_loop_to_global()?;
    /// // LOOP axes that appear in all outputs are now GLOBAL
    /// ```
    pub fn convert_loop_to_global(&mut self) -> Result<(), OptError> {
        // Only for GPU backends
        if !self.ren.has_local {
            return Ok(());
        }

        let globalizable = self.globalizable_rngs();
        if globalizable.is_empty() {
            return Ok(());
        }

        // Build substitution map: WEAK -> GLOBAL
        let mut subst_map = std::collections::HashMap::new();
        for rng in globalizable {
            let new_rng = rng.with_axis_type(AxisType::Global);
            subst_map.insert(UOpKey(rng), new_rng);
        }

        // Apply substitution
        self.ast = self.ast.substitute(&subst_map);

        self.clear_caches();

        Ok(())
    }

    /// Get the optimized AST with kernel metadata attached.
    ///
    /// This is the final step of optimization, which:
    /// 1. Generates a kernel name from the shape (e.g., "r_16_16_32_4")
    /// 2. Flattens nested ranges
    /// 3. Attaches KernelInfo metadata
    ///
    /// # Arguments
    ///
    /// * `name_override` - Optional custom kernel name (otherwise auto-generated)
    ///
    /// # Returns
    ///
    /// UOp with attached KernelInfo metadata containing name, applied_opts, and flags.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let optimized = scheduler.get_optimized_ast(None);
    /// let info = optimized.metadata::<KernelInfo>().unwrap();
    /// println!("Kernel: {}", info.name); // "r_16_16_32_4"
    /// ```
    pub fn get_optimized_ast(&self, name_override: Option<String>) -> Arc<UOp> {
        let name = name_override.unwrap_or_else(|| unique_kernel_name(&self.shape_name()));
        self.finish(name)
    }

    /// `get_optimized_ast(None)` with the naming step chosen by the caller.
    pub fn get_optimized_ast_with_naming(&self, naming: KernelNaming) -> Arc<UOp> {
        match naming {
            KernelNaming::Unique => self.get_optimized_ast(None),
            KernelNaming::Deferred => self.finish(self.shape_name()),
        }
    }

    /// Auto-generated kernel name from the optimized loop structure.
    fn shape_name(&self) -> String {
        {
            // Prefix: "r" for reduce, "E" for elementwise
            let prefix = if self.reduceop().is_some() { "r" } else { "E" };

            let extent_name = |end: &Arc<UOp>| match end.op() {
                Op::Const(cv) => cv.0.try_int().map_or_else(|| "?".to_string(), |size| size.to_string()),
                _ => "?".to_string(),
            };

            // Tinygrad uses color only to distinguish axis classes in
            // diagnostics. Function names retain only ordered extents.
            let mut specials: Vec<_> = self
                .ast
                .toposort()
                .into_iter()
                .filter_map(|node| match node.op() {
                    Op::Special(ops::Special { end, name }) => Some((name.clone(), extent_name(end))),
                    _ => None,
                })
                .collect();
            specials.sort_by(|left, right| left.0.cmp(&right.0));
            let mut shape_parts: Vec<String> = specials.into_iter().map(|(_, extent)| extent).collect();
            shape_parts.extend(
                self.rngs()
                    .iter()
                    .filter_map(|rng| {
                        let Op::Range(ops::Range { end, .. }) = rng.op() else { return None };
                        Some(extent_name(end))
                    })
                    .collect::<Vec<_>>(),
            );

            if shape_parts.is_empty() { prefix.to_string() } else { format!("{}_{}", prefix, shape_parts.join("_")) }
        }
    }

    fn finish(&self, name: String) -> Arc<UOp> {
        use crate::optimizer::KernelInfo;

        // 2. Flatten ranges (top-down graph_rewrite default).
        let flattened_ast =
            crate::rewrite::graph_rewrite(crate::rangeify::pm_flatten_range(), self.ast.clone(), &mut ());

        let (flattened_ast, name) = match flattened_ast.op() {
            Op::Sink(ops::Sink { sources, info }) => {
                let mut structural = info.clone().unwrap_or_default();
                // A hand-authored kernel (`opts_to_apply` set) keeps its author's name.
                let name = match &structural.name {
                    Some(authored) if structural.opts_to_apply.is_some() => authored.clone(),
                    _ => name,
                };
                structural.name = Some(name.clone());
                structural.applied_opts = self.applied_opts.clone();
                structural.dont_use_locals = self.dont_use_locals;
                (UOp::sink_with_info(sources.iter().cloned().collect(), structural), name)
            }
            _ => (flattened_ast, name),
        };

        // 3. Attach metadata
        let info = KernelInfo { name, applied_opts: self.applied_opts.clone(), dont_use_locals: self.dont_use_locals };
        flattened_ast.with_metadata(info)
    }
}

impl fmt::Display for Scheduler {
    /// Format the scheduler as a kernel descriptor string.
    ///
    /// Format: "{kernel_type}_{colored_shape}"
    ///
    /// Examples:
    /// - "r_g16l16R32u4" - Reduction kernel with Global, Local, Reduce, Upcast
    /// - "E_g256g256" - Elementwise kernel with 2D Global shape
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.kernel_type(), self.colored_shape())
    }
}

impl Clone for Scheduler {
    /// Clone the scheduler state.
    ///
    /// Note: Caches are cleared in the clone to ensure correct behavior.
    fn clone(&self) -> Self {
        Self {
            ast: self.ast.clone(),
            ren: self.ren.clone(),
            dont_use_locals: self.dont_use_locals,
            applied_opts: self.applied_opts.clone(),
            selected_tc_index: self.selected_tc_index,
            // Clear caches in clone - they'll be recomputed on demand
            rngs_cache: OnceCell::new(),
            maxarg_cache: OnceCell::new(),
            toposort_cache: OnceCell::new(),
            reduceops_cache: OnceCell::new(),
            bufs_cache: OnceCell::new(),
        }
    }
}
