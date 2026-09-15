use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use smallvec::{SmallVec, smallvec};
use svod_dtype::AddrSpace;
use svod_ir::ops;
use svod_ir::uop::{Nodes, SliceMemo, SubtreeMemo};
use svod_ir::{AxisType, ConstValue, Op, TypedPatternMatcher, UOp, UOpKey};

/// Local STOREs in each node's backward slice, memoized for one `pm_implicit_barriers`
/// run so neither rule toposorts the slice of every AFTER/END it visits.
pub struct BarrierContext {
    /// Every local STORE below a node (`add_war_barrier`).
    stores: SliceMemo<Nodes>,
    /// Whether a local STORE is reachable without crossing a BARRIER (`add_raw_barrier`).
    unbarriered_store: SubtreeMemo,
}

impl Default for BarrierContext {
    fn default() -> Self {
        Self {
            stores: SliceMemo::new(is_local_store),
            unbarriered_store: SliceMemo::gated(is_local_store, |uop| !matches!(uop.op(), Op::Barrier(..))),
        }
    }
}

fn access_buffer(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    match uop.op() {
        Op::Param(..) | Op::Buffer(..) | Op::MSelect(..) | Op::MStack(..) => Some(uop.clone()),
        Op::Index(ops::Index { buffer, .. }) | Op::After(ops::After { passthrough: buffer, .. }) => {
            access_buffer(buffer)
        }
        Op::Cast(ops::Cast { src, .. })
        | Op::Reshape(ops::Reshape { src, .. })
        | Op::Permute(ops::Permute { src, .. })
        | Op::Expand(ops::Expand { src, .. })
        | Op::Pad(ops::Pad { src, .. })
        | Op::Shrink(ops::Shrink { src, .. })
        | Op::Flip(ops::Flip { src, .. }) => access_buffer(src),
        _ => None,
    }
}

fn is_local_store(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Store(..)) && uop.addrspace() == Some(AddrSpace::Local)
}

fn barrier_from_sources(sources: &[Arc<UOp>]) -> Option<Arc<UOp>> {
    let (src, deps) = sources.split_first()?;
    Some(src.barrier(deps.iter().cloned().collect()))
}

fn add_raw_barrier(ctx: &mut BarrierContext, after: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::After(ops::After { passthrough, deps }) = after.op() else { return None };
    if after.addrspace() != Some(AddrSpace::Local) {
        return None;
    }

    // Tinygrad gates one toposort over SINK(*after.src[1:]) on "not a BARRIER".
    if !deps.iter().any(|dep| ctx.unbarriered_store.contains(dep)) {
        return None;
    }

    Some(passthrough.after(smallvec![barrier_from_sources(deps)?]))
}

fn add_war_barrier(ctx: &mut BarrierContext, end: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::End(ops::End { computation, ranges }) = end.op() else { return None };
    if matches!(computation.op(), Op::Barrier(..)) {
        return None;
    }

    let loop_ranges: Vec<_> = ranges
        .iter()
        .filter(|range| {
            matches!(
                range.op(),
                Op::Range(ops::Range { axis_type: AxisType::Reduce | AxisType::Weak | AxisType::Loop, .. })
            ) && matches!(range.vmax(), ConstValue::Int(vmax) if *vmax > 0)
        })
        .cloned()
        .collect();
    if loop_ranges.is_empty() {
        return None;
    }

    let loop_range_ids: HashSet<_> = loop_ranges.iter().map(|range| range.id).collect();
    let store_buffers: HashSet<_> = ctx
        .stores
        .get(computation)
        .iter()
        .filter(|uop| uop.in_scope_ranges().iter().any(|id| loop_range_ids.contains(id)))
        .filter_map(|uop| match uop.op() {
            Op::Store(ops::Store { index, .. }) => access_buffer(index).map(UOpKey),
            _ => None,
        })
        .collect();
    if store_buffers.is_empty() {
        return None;
    }

    // Only a loop body that stores to local memory pays for the ordered walk:
    // BARRIER sources keep the loads in `backward_slice_with_self` order. A load
    // outside the loop's scope (reached through an ordering edge, as a gather the
    // fill of a double buffer is threaded after) is not re-run against the loop's
    // stores, so it carries no per-iteration hazard; the loop's own END fences it.
    let loads: SmallVec<[Arc<UOp>; 4]> = computation
        .toposort()
        .iter()
        .filter(|uop| uop.in_scope_ranges().iter().any(|id| loop_range_ids.contains(id)))
        .filter(|uop| match uop.op() {
            Op::Load(ops::Load { index, .. }) => {
                access_buffer(index).is_some_and(|buffer| store_buffers.contains(&UOpKey(buffer)))
            }
            _ => false,
        })
        .cloned()
        .collect();
    if loads.is_empty() {
        return None;
    }

    Some(computation.barrier(loads).end(ranges.clone()))
}

/// Both rules need a local STORE below the node they match, so a graph without
/// one is returned untouched instead of being walked by the rewrite engine.
pub(crate) fn add_implicit_barriers(root: Arc<UOp>) -> Arc<UOp> {
    if !root.any_in_subtree(is_local_store) {
        return root;
    }
    crate::rewrite::graph_rewrite(pm_implicit_barriers(), root, &mut BarrierContext::default())
}

fn pm_implicit_barriers() -> &'static TypedPatternMatcher<BarrierContext> {
    static PM: LazyLock<TypedPatternMatcher<BarrierContext>> = LazyLock::new(|| {
        crate::patterns! {
            @context BarrierContext;
            after @ After { passthrough: _, deps: _ } => add_raw_barrier(ctx, after),
            end @ End { computation: _, ranges: _ } => add_war_barrier(ctx, end),
        }
    });
    &PM
}
