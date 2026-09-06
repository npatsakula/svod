//! GPU dimension injection for kernel execution.
//!
//! This module implements `pm_add_gpudims`, which transforms RANGE operations
//! with GLOBAL/LOCAL axis types into SPECIAL UOps representing GPU thread indices.
//!
//! Based on Tinygrad's `gpudims.py`.
//!
//! # Pipeline Position
//!
//! Runs between `pm_reduce` (Stage 11) and `pm_add_loads` (Stage 13):
//! - After reduction is lowered to accumulator patterns
//! - Before loads are explicitly extracted from INDEX ops
//!
//! # Transformation
//!
//! ```text
//! RANGE(end, axis_id, GLOBAL) → gidxN (SPECIAL with global thread index)
//! RANGE(end, axis_id, LOCAL)  → lidxN (SPECIAL with local thread index)
//! ```
//!
//! Dimension limiting is applied to fit within hardware constraints:
//! - Grouping: Merge adjacent dimensions that fit within limits
//! - Splitting: Factor dimensions that exceed limits

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::types::{AxisType, ConstValue};
use svod_ir::{Op, UOp, UOpKey};

use crate::optimizer::Renderer;
use crate::pattern::TypedPatternMatcher;
use svod_ir::ops;

/// Device limits plus the SINK this pass already lowered. The engine re-matches
/// a rewritten SINK; when every GPU range was substituted, that second visit can
/// only return `None`, so it skips the second full-graph analysis.
pub struct GpuDimsContext {
    renderer: Renderer,
    lowered: Option<u64>,
}

impl From<Renderer> for GpuDimsContext {
    fn from(renderer: Renderer) -> Self {
        Self { renderer, lowered: None }
    }
}

/// Pattern matcher for GPU dimension injection.
///
/// Matches SINK operations and transforms GLOBAL/LOCAL ranges to SPECIAL ops.
/// Must run after pm_reduce and before pm_add_loads.
pub fn pm_add_gpudims() -> TypedPatternMatcher<GpuDimsContext> {
    crate::patterns! {
        @context GpuDimsContext;
        // add gpudims must be last
        sink @ Sink { sources: _sources } => add_gpudims(ctx, sink),
    }
}

/// DEVICE ranges are launch bindings, independent of GPU thread-dimension
/// support. Run this for every renderer before the capability-gated GPU pass.
pub fn pm_lower_device_ranges() -> TypedPatternMatcher {
    crate::patterns! {
        // DEVICE is bound at launch, not dispatched as a program axis.
        range @ Range { end: _, axis_id: _, axis_type }
            if *axis_type == AxisType::Device => lower_device_range(range),
        // A lowered DEVICE PARAM is not a loop and must not be closed by END.
        end @ End { computation: _, ranges }
            if ranges.iter().any(is_device_num) => cleanup_device_end(end),
    }
}

fn is_device_num(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Param(ops::Param { arg, .. }) if arg.name.as_deref() == Some("_device_num"))
}

fn lower_device_range(range: &Arc<UOp>) -> Option<Arc<UOp>> {
    Some(UOp::variable("_device_num".to_string(), 0, const_to_i64(range.vmax())?, range.dtype()))
}

fn cleanup_device_end(end: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::End(ops::End { computation, ranges }) = end.op() else { return None };
    Some(UOp::new(
        Op::End(ops::End {
            computation: computation.clone(),
            ranges: ranges.iter().filter(|uop| !matches!(uop.op(), Op::Param(..))).cloned().collect(),
        }),
        end.dtype(),
    ))
}

/// Main transformation: inject GPU dimensions into SINK.
///
/// Follows Tinygrad's `add_gpudims` function (gpudims.py:59-103):
/// 1. Collect all RANGE operations from topology
/// 2. Check for existing SPECIAL ops (skip if found)
/// 3. Categorize ranges by axis type (GLOBAL/THREAD vs LOCAL/WARP/GROUP_REDUCE)
/// 4. Create SPECIAL indices with dimension limiting
/// 5. Substitute RANGE ops with computed indices
fn add_gpudims(ctx: &mut GpuDimsContext, sink: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Sink(..) = sink.op() else {
        return None;
    };
    if ctx.lowered == Some(sink.id) {
        return None;
    }
    let renderer = &ctx.renderer;

    // Collect topology (all UOps reachable from sink)
    let topo = sink.toposort();

    // Check for existing SPECIAL ops - if found, gpudims already applied
    if topo.iter().any(|u| matches!(u.op(), Op::Special(..))) {
        return None;
    }

    // Collect all RANGE operations, keyed by (axis_id, axis_type)
    // We exclude axis_type from the key matching for categorization, but track it
    let mut all_ranges: HashMap<(svod_ir::AxisId, AxisType), Arc<UOp>> = HashMap::new();
    for u in &topo {
        if let Op::Range(ops::Range { axis_id, axis_type, .. }) = u.op() {
            all_ranges.insert((axis_id.clone(), *axis_type), u.clone());
        }
    }

    if all_ranges.is_empty() {
        return None;
    }

    // Categorize ranges by axis type
    // Global dims: GLOBAL, THREAD
    // Local dims: LOCAL, WARP, GROUP_REDUCE
    let mut global_dims: Vec<(svod_ir::AxisId, AxisType)> = Vec::new();
    let mut local_dims: Vec<(svod_ir::AxisId, AxisType)> = Vec::new();

    for (axis_id, axis_type) in all_ranges.keys() {
        match axis_type {
            AxisType::Global | AxisType::Thread if !global_dims.iter().any(|(id, _)| id == axis_id) => {
                global_dims.push((axis_id.clone(), *axis_type));
            }
            AxisType::Local | AxisType::Warp | AxisType::GroupReduce
                if !local_dims.iter().any(|(id, _)| id == axis_id) =>
            {
                local_dims.push((axis_id.clone(), *axis_type));
            }
            _ => {}
        }
    }

    // Sort by axis_id for consistent ordering. The WARP axis leads the locals
    // (tinygrad numbers it -1): `mma.sync` addresses fragments by the hardware
    // lane, so the warp axis must be the low bits of the linear thread index.
    global_dims.sort_by(|(a, _), (b, _)| a.cmp(b));
    local_dims.sort_by(|(a, _), (b, _)| a.cmp(b));
    local_dims.sort_by_key(|(_, axis_type)| *axis_type != AxisType::Warp);

    // No GPU dimensions to inject
    if global_dims.is_empty() && local_dims.is_empty() {
        return None;
    }

    // Extract shapes from RANGE operations (the end values)
    let get_ranges_for_dims = |dims: &[(svod_ir::AxisId, AxisType)]| -> Vec<Arc<UOp>> {
        dims.iter().filter_map(|(axis_id, axis_type)| all_ranges.get(&(axis_id.clone(), *axis_type))).cloned().collect()
    };

    let global_ranges = get_ranges_for_dims(&global_dims);
    let local_ranges = get_ranges_for_dims(&local_dims);

    // Extract dimension sizes from ranges
    let extract_shape = |ranges: &[Arc<UOp>]| -> Vec<Arc<UOp>> {
        ranges
            .iter()
            .filter_map(|r| match r.op() {
                Op::Range(ops::Range { end, .. }) => Some(end.clone()),
                _ => None,
            })
            .collect()
    };

    let global_shape = extract_shape(&global_ranges);
    let local_shape = extract_shape(&local_ranges);

    let dont_use_locals = sink.metadata::<crate::optimizer::KernelInfo>().is_some_and(|info| info.dont_use_locals);
    let all_idxs: Vec<Arc<UOp>> = if renderer.has_threads {
        // global_shape contains RANGE extents, not range indices. Match
        // Tinygrad's `int(global_shape[0])-1`: Thread(N) has N core IDs.
        let end = thread_core_bound(&global_dims, &local_dims, &global_shape)?;
        vec![UOp::variable("core_id".to_string(), 0, end - 1, DType::Int32).cast(DType::WeakInt)]
    } else if dont_use_locals {
        assert!(local_dims.is_empty(), "can't use locals if there's no local dims");
        get_grouped_dims("idx", &global_shape, renderer.global_max.as_deref(), true)
    } else {
        // Generate GPU indices
        // Renderer keeps the workgroup product cap separate from per-axis caps.
        let mut local_max: Option<Vec<usize>> =
            renderer.local_max_axes().map(|axes| axes.to_vec()).or_else(|| renderer.local_max.map(|max| vec![max; 3]));
        // Pin the leading axis cap to the warp extent so `group_dims` never
        // folds another local into `lidx0` (tinygrad gpudims.py:59): a warp
        // sharing `tid.x` with a size-2 local scrambles the tensor-core lanes.
        if let (Some(max), Some((_, AxisType::Warp))) = (local_max.as_mut(), local_dims.first()) {
            max[0] = dim_max(&local_shape[0]);
        }
        let local_max_slice = local_max.as_deref();

        // Create local indices (lidx0, lidx1, ...)
        let local_idxs = get_grouped_dims("lidx", &local_shape, local_max_slice, false);
        let hw_local = hardware_local_extents(&local_idxs);
        let global_max = renderer.global_prod_max.as_ref().map_or_else(
            || renderer.global_max.clone(),
            |prod_max| {
                let base = renderer.global_max.as_ref().unwrap_or(prod_max);
                base.iter()
                    .zip(prod_max)
                    .zip(hw_local.iter().copied().chain(std::iter::repeat(1)).take(3))
                    // A zero-extent local axis would divide by zero; it occupies
                    // one work-item slot either way, so clamp to 1.
                    .map(|((&global, &product), local)| global.min(product / local.max(1)))
                    .collect::<Vec<_>>()
                    .into()
            },
        );
        // Create global indices (gidx0, gidx1, ...)
        let global_idxs = get_grouped_dims("gidx", &global_shape, global_max.as_deref(), true);
        // Combine indices in order: global, then local
        global_idxs.into_iter().chain(local_idxs).collect()
    };

    // Build substitution map: RANGE -> corresponding index
    let mut subs: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    let all_dims: Vec<(svod_ir::AxisId, AxisType)> = global_dims.iter().chain(local_dims.iter()).cloned().collect();

    // Every branch above yields one index per dim (the threaded branch only
    // after `thread_core_bound` proved there is exactly one), so `zip` never
    // truncates — it just makes the `idxs[ii]` bound unindexable.
    debug_assert_eq!(all_dims.len(), all_idxs.len(), "gpudims: one index per GPU dim");
    for ((axis_id, axis_type), idx) in all_dims.iter().zip(&all_idxs) {
        if *axis_type == AxisType::Reduce {
            // Don't replace reduce axes (they stay as loops)
            continue;
        }
        if let Some(range_uop) = all_ranges.get(&(axis_id.clone(), *axis_type)) {
            subs.insert(UOpKey(range_uop.clone()), idx.clone());
        }
    }

    // Handle STORE masking for global stores with missing local indices
    // When a STORE to global memory doesn't use all local indices,
    // we need to mask the store to only execute when unused local indices are 0
    let store_subs = compute_store_masks(&topo, &all_ranges, &local_dims);
    for (id, masked_idx) in store_subs {
        subs.insert(id, masked_idx);
    }

    // Apply substitutions to rebuild the sink
    if subs.is_empty() {
        return None;
    }

    let lowered = sink.substitute(&subs);
    // `global_dims`/`local_dims` keep one range per axis id; a second range on
    // the same id would survive and still needs the re-visit.
    let gpu_ranges = all_ranges.keys().filter(|(_, axis_type)| is_gpu_axis(*axis_type)).count();
    if gpu_ranges == all_dims.len() {
        ctx.lowered = Some(lowered.id);
    }
    Some(lowered)
}

fn is_gpu_axis(axis_type: AxisType) -> bool {
    matches!(axis_type, AxisType::Global | AxisType::Thread | AxisType::Local | AxisType::Warp | AxisType::GroupReduce)
}

/// Hardware extent of each `lidx*` axis behind the local indices.
///
/// Tinygrad (`gpudims.py:67`) reads `u.src[0]` off entries that *are* SPECIAL:
/// `[_dim_max(u.src[0]) for u in local_idxs if u.op is Ops.SPECIAL]`. That only
/// sees the axes `get_grouped_dims` handed back unchanged — as soon as a local
/// dim is grouped or split, the returned entry is div/mod arithmetic over the
/// SPECIALs and the list comes back short (empty in the fully-contracted case),
/// silently dropping the AMD work-item product cap it feeds. Collect the
/// SPECIAL leaves instead, deduplicated by `lidx` name so a leaf reachable from
/// two decomposed indices is counted once, in axis order.
fn hardware_local_extents(local_idxs: &[Arc<UOp>]) -> Vec<usize> {
    local_idxs
        .iter()
        .flat_map(|idx| idx.toposort())
        .filter_map(|u| match u.op() {
            Op::Special(ops::Special { end, name }) if name.starts_with("lidx") => Some((name.clone(), dim_max(end))),
            _ => None,
        })
        .collect::<std::collections::BTreeMap<_, _>>()
        .into_values()
        .collect()
}

/// `core_id` upper bound for the threaded path (tinygrad `gpudims.py:60`).
///
/// Tinygrad's `int(global_shape[0])-1` silently assumes the sink has exactly
/// one THREAD axis, no local axes, and a concrete extent — the shape a CPU
/// renderer produces. Anything else means the scheduler handed us a kernel the
/// threaded path cannot express: one `core_id` cannot stand in for N ranges.
/// Return `None` so `add_gpudims` declines the rewrite instead of indexing past
/// the single produced index (or panicking on a symbolic bound).
fn thread_core_bound(
    global_dims: &[(svod_ir::AxisId, AxisType)],
    local_dims: &[(svod_ir::AxisId, AxisType)],
    global_shape: &[Arc<UOp>],
) -> Option<i64> {
    if global_dims.len() != 1 || !local_dims.is_empty() {
        tracing::warn!(
            globals = global_dims.len(),
            locals = local_dims.len(),
            "gpudims: threaded renderer needs exactly one global axis and no local axes; skipping"
        );
        return None;
    }
    let bound = const_to_i64(global_shape.first()?.vmax());
    if bound.is_none() {
        tracing::warn!("gpudims: threaded global axis has no concrete bound; skipping");
    }
    bound
}

/// Compute store masks for global stores with missing local indices.
///
/// Based on Tinygrad's gpudims.py:86-96.
/// When a STORE to global memory doesn't use all local indices,
/// we add a mask so the store only executes when missing locals are 0.
fn compute_store_masks(
    topo: &[Arc<UOp>],
    all_ranges: &HashMap<(svod_ir::AxisId, AxisType), Arc<UOp>>,
    local_dims: &[(svod_ir::AxisId, AxisType)],
) -> HashMap<UOpKey, Arc<UOp>> {
    let mut masks: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    for uop in topo {
        let Op::Store(ops::Store { index, .. }) = uop.op() else {
            continue;
        };

        // Tinygrad reads `idx.src[0].addrspace` (`gpudims.py:76`), and
        // `UOp.addrspace` is recursive: it projects through AFTER/CAST/INDEX and
        // agrees across the sources of a STACK. Matching PARAM/BUFFER one level
        // deep instead missed every wrapped target — a STACK of params, or a
        // param behind an AFTER — and silently dropped the store mask.
        let Op::Index(ops::Index { buffer, .. }) = index.op() else { continue };
        if buffer.addrspace() != Some(svod_dtype::AddrSpace::Global) {
            continue;
        }

        // Find local ranges NOT used in the index computation.
        // Use in_scope_ranges() to get only active (not ended) ranges,
        // rather than toposort().filter(Range) which returns ALL ranges in the graph.
        let index_ranges: HashSet<u64> = index.in_scope_ranges().iter().copied().collect();

        let mut missing_locals: Vec<Arc<UOp>> = Vec::new();
        for (axis_id, axis_type) in local_dims {
            if let Some(range_uop) = all_ranges.get(&(axis_id.clone(), *axis_type))
                && !index_ranges.contains(&range_uop.id)
            {
                missing_locals.push(range_uop.clone());
            }
        }

        if missing_locals.is_empty() {
            continue;
        }

        // Create mask: (missing_local_1 == 0) & (missing_local_2 == 0) & ...
        // Using eq() and and_() panicking wrappers for cleaner code
        let zero = UOp::index_const(0);
        let mut mask: Option<Arc<UOp>> = None;
        for local_idx in missing_locals {
            let eq_zero = local_idx.eq(&zero);
            mask = Some(match mask {
                None => eq_zero,
                Some(m) => m.and_(&eq_zero),
            });
        }

        // Keep validity in the index expression so RANGE substitution carries it
        // to the corresponding hardware index.
        if let (Some(mask), Op::Index(ops::Index { buffer, indices })) = (mask, index.op()) {
            assert_eq!(indices.len(), 1, "gpudims: index must have one index source");
            let new_index = UOp::index()
                .buffer(buffer.clone())
                .indices(vec![indices[0].valid(mask)])
                .call()
                .expect("gpudims: INDEX validity construction failed");
            masks.insert(UOpKey(index.clone()), new_index);
        }
    }

    masks
}

/// Extract i64 value from ConstValue.
fn const_to_i64(cv: &ConstValue) -> Option<i64> {
    match cv {
        ConstValue::Invalid => None,
        ConstValue::Int(v) => Some(*v),
        ConstValue::UInt(v) => Some(*v as i64),
        ConstValue::Bool(v) => Some(*v as i64),
        ConstValue::Float(v) => Some(*v as i64),
    }
}

/// Tinygrad's `_dim_max(d: sint) -> int` (gpudims.py:7): concrete int passes
/// through, symbolic UOp returns its `vmax` upper bound. Used uniformly across
/// grouping/splitting so concrete and symbolic dims go through one code path.
fn dim_max(d: &Arc<UOp>) -> usize {
    const_to_i64(d.vmax()).map(|v| v.max(0) as usize).unwrap_or(usize::MAX)
}

/// True when `a` and `b` are structurally identical (hash-cons identity).
fn dims_eq(a: &[Arc<UOp>], b: &[Arc<UOp>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| Arc::ptr_eq(x, y))
}

/// True when `u` is the concrete CONST integer 1 (for matching tinygrad's
/// `acc != 1` leading-1 special case in `get_contraction`).
fn is_one(u: &Arc<UOp>) -> bool {
    matches!(u.op(), Op::Const(c) if matches!(c.0, ConstValue::Int(1)))
}

/// Create GPU thread indices with dimension limiting.
///
/// Mirrors Tinygrad's `get_grouped_dims` (gpudims.py:28-56). Operates on
/// `sint` dims (concrete `Int` const or symbolic `UOp`) end-to-end via
/// [`dim_max`]; grouping/splitting always returns a fresh `Vec<Arc<UOp>>` so
/// downstream `decompose`/`combine`/`flatten_unflatten` can index into it
/// regardless of whether the input was numeric or symbolic.
///
/// # Arguments
///
/// * `prefix` - Index name prefix ("gidx" or "lidx")
/// * `dims` - Dimension sizes as UOps
/// * `max_sizes` - Hardware limits per dimension (None = unlimited)
/// * `reverse` - Reverse dimension ordering (true for global indices)
///
/// # Returns
///
/// Vector of SPECIAL UOps (plus mod/idiv decomposition where contraction was
/// applied) representing thread indices, one per original `dims` entry.
fn get_grouped_dims(prefix: &str, dims: &[Arc<UOp>], max_sizes: Option<&[usize]>, reverse: bool) -> Vec<Arc<UOp>> {
    // Tinygrad-equivalent (`codegen/gpudims.py:29`): when `reverse=True`,
    // recursively call with reversed dims, then reverse the result. Reversing
    // only the OUTPUT array leaves the SPECIAL UOps named in iteration order
    // while the indices land at swapped positions — manifests as a 21× OOB on
    // matmul+reduce kernels where g_x and g_y are picked for different range
    // axes.
    if reverse {
        let reversed: Vec<Arc<UOp>> = dims.iter().cloned().rev().collect();
        let result = get_grouped_dims(prefix, &reversed, max_sizes, false);
        return result.into_iter().rev().collect();
    }
    if dims.is_empty() {
        return vec![];
    }

    let limited: Vec<Arc<UOp>> = match max_sizes {
        None => dims.to_vec(),
        Some(max) => {
            // First try grouping: (a, b, c, d) → (a*b, c, d). Match tinygrad's
            // fail-fast behaviour (gpudims.py:33-37): if neither grouping nor
            // splitting can fit the dims into the backend's axis cap, panic
            // immediately. Returning unchanged dims and warning is what we
            // used to do, but it produced SPECIAL UOps with `gidx3`/`lidx3+`
            // that the AMD renderer rejects at codegen/src/llvm/amd/ops.rs;
            // the error surfaced at codegen time rather than at scheduling
            // time, which buries the actual problem (a bad scheduler/BEAM
            // candidate). Failing here makes the offending candidate visible.
            let grouped = group_dims(dims, max);
            let after_group = grouped.unwrap_or_else(|| dims.to_vec());
            if after_group.len() > max.len() {
                panic!(
                    "get_grouped_dims: cannot limit dims to {} axes (dims={:?}, max_sizes={:?}); \
                     scheduler emitted more SPECIAL axes than the backend supports",
                    max.len(),
                    dims.iter().map(dim_max).collect::<Vec<_>>(),
                    max,
                );
            }
            if dims_eq(&after_group, dims) {
                // No grouping happened (or every group attempt was a no-op):
                // try splitting up dims (a,) → (b, c).
                split_dims(dims, max).unwrap_or_else(|| {
                    panic!(
                        "get_grouped_dims: split_dims failed (likely non-factorable symbolic dim); \
                         dims={:?}, max_sizes={:?}",
                        dims.iter().map(dim_max).collect::<Vec<_>>(),
                        max,
                    )
                })
            } else {
                after_group
            }
        }
    };

    let raw_idxs: Vec<Arc<UOp>> =
        limited.iter().enumerate().map(|(i, s)| UOp::special(s.clone(), format!("{prefix}{i}"))).collect();

    // Nothing was grouped or split, so the flatten/divmod round-trip below is
    // the identity on a mixed-radix decomposition whose digits are exactly the
    // SPECIALs' own bounds: return them directly.
    //
    // Tinygrad dropped this early exit (it is `gpudims.py:57` at 1f8b24a6b) once
    // the four exits collapsed into one expression, because `ssimplify` folds
    // the round-trip away for concrete dims. It does not fold it away for a
    // symbolic divisor: `get_grouped_dims("gidx", (n, 8), None, reverse=True)`
    // still renders `(gidx1%n)` and `(gidx0+gidx1//n)` at the pin, because
    // nothing there proves `gidx1 < n`. Ours leaves the whole
    // `(gidx0*n+gidx1)` round-trip standing, so a symbolic launch dimension
    // burns a FloorDiv and a FloorMod in every index expression downstream.
    if dims_eq(&limited, dims) {
        return raw_idxs;
    }

    let product =
        |values: &[Arc<UOp>]| values.iter().cloned().reduce(|a, b| a.mul(&b)).unwrap_or_else(|| UOp::index_const(1));
    let flat = raw_idxs
        .iter()
        .enumerate()
        .map(|(i, idx)| idx.mul(&product(&limited[i + 1..])))
        .reduce(|a, b| a.add(&b))
        .unwrap_or_else(|| UOp::index_const(0));
    dims.iter()
        .enumerate()
        .map(|(i, dim)| {
            let value = flat.floor_div(&product(&dims[i + 1..]));
            let value = if i == 0 { value } else { value.mod_(dim) };
            crate::rewrite::graph_rewrite(crate::symbolic::symbolic(), value, &mut ())
        })
        .collect()
}

/// Group adjacent dimensions to fit within hardware limits.
///
/// Mirrors Tinygrad's `_group_dims` (gpudims.py:9-16).
fn group_dims(dims: &[Arc<UOp>], max_sizes: &[usize]) -> Option<Vec<Arc<UOp>>> {
    let mut result: Vec<Arc<UOp>> = dims.to_vec();
    while result.len() > max_sizes.len() || result.iter().zip(max_sizes).any(|(d, m)| dim_max(d) > *m) {
        let mut grouped = false;
        for (i, &m) in max_sizes.iter().enumerate() {
            if i + 1 < result.len() && dim_max(&result[i]).saturating_mul(dim_max(&result[i + 1])) <= m {
                let merged = result[i].mul(&result[i + 1]);
                result = result[..i]
                    .iter()
                    .cloned()
                    .chain(std::iter::once(merged))
                    .chain(result[i + 2..].iter().cloned())
                    .collect();
                grouped = true;
                break;
            }
        }
        if !grouped {
            return None;
        }
    }
    Some(result)
}

/// Split dimensions that exceed hardware limits.
///
/// Mirrors Tinygrad's `_split_dims` (gpudims.py:18-26). Splitting requires a
/// concrete factor; if any dim that exceeds its limit is symbolic (no
/// `Op::Const` peer to read), the operation is unrepresentable and `None` is
/// returned (tinygrad raises in the same situation).
fn split_dims(dims: &[Arc<UOp>], max_sizes: &[usize]) -> Option<Vec<Arc<UOp>>> {
    if dims.iter().zip(max_sizes).all(|(d, m)| dim_max(d) <= *m) {
        return Some(dims.to_vec());
    }
    let mut working: Vec<Arc<UOp>> = dims.to_vec();
    while working.len() < 3 {
        working.push(UOp::index_const(1));
    }
    for i in 0..working.len() {
        while dim_max(&working[i]) > max_sizes[i] {
            let val = match working[i].op() {
                Op::Const(c) => usize::try_from(const_to_i64(&c.0)?).ok()?,
                _ => return None,
            };
            let div = find_smallest_divisor(val);
            if div == 1 {
                return None;
            }
            let next = (i + 1) % working.len();
            working[i] = UOp::index_const(i64::try_from(val / div).ok()?);
            working[next] = match working[next].op() {
                Op::Const(c) => {
                    let next_val = usize::try_from(const_to_i64(&c.0)?).ok()?.checked_mul(div)?;
                    UOp::index_const(i64::try_from(next_val).ok()?)
                }
                _ => working[next].mul(&UOp::index_const(i64::try_from(div).ok()?)),
            };
        }
    }
    let result = if is_one(&working[2]) {
        if is_one(&working[1]) { vec![working[0].clone()] } else { vec![working[0].clone(), working[1].clone()] }
    } else {
        working
    };
    Some(result)
}

/// Find the smallest divisor of n (excluding 1).
fn find_smallest_divisor(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let sqrt_n = (n as f64).sqrt().ceil() as usize;
    for d in 2..=sqrt_n {
        if n.is_multiple_of(d) {
            return d;
        }
    }
    1 // n is prime
}

#[cfg(test)]
#[path = "test/unit/gpudims_internal.rs"]
mod tests;
