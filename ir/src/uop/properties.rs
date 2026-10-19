//! Standard cached properties for UOps.
//!
//! This module defines the standard set of cached properties using the
//! `cached_property!` macro infrastructure.
//!
//! # Available Properties
//!
//! - [`ShapeProperty`] - Shape inference for tensor operations
//! - [`RangesProperty`] - All RANGE operations in the graph
//! - [`InScopeRangesProperty`] - RANGE operations currently in scope
//! - [`VminVmaxProperty`] - Range analysis (min/max values) for operations

use crate::Op;
use crate::cached_property;
use crate::types::ConstValue;
use std::sync::Arc;

// ============================================================================
// Shape Property
// ============================================================================

cached_property! {
    /// Cached shape property.
    ///
    /// Computes the shape of a UOp via shape inference rules.
    /// Returns `Ok(None)` for control flow operations (SINK, END, CALL wrappers, etc.),
    /// `Ok(Some(shape))` for tensor operations, and `Err` for shape mismatches.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use svod_ir::uop::properties::ShapeProperty;
    /// use svod_ir::uop::cached_property::CachedProperty;
    ///
    /// let shape_result = ShapeProperty::get(&my_uop);
    /// ```
    ShapeProperty: Result<Option<crate::shape::Shape>, Box<crate::error::Error>> {
        cache_field: shape_cache,
        compute: |uop| crate::shape::infer_shape_from_op(uop).map_err(Box::new)
    }
}

// ============================================================================
// Ranges Property
// ============================================================================

cached_property! {
    /// Cached ranges property (recursive, like Tinygrad's approach).
    ///
    /// Returns all RANGE operations in the computation graph, computed from
    /// children's cached values. O(N) total on first access, O(1) after.
    ///
    /// This is different from `in_scope_ranges` which only returns ranges
    /// that are currently "active" (not yet ended).
    RangesProperty: Vec<Arc<crate::UOp>> {
        cache_field: ranges_cache,
        compute: |uop| {
            let mut seen = std::collections::HashSet::new();
            let mut result = Vec::new();
            // Deliberately EXCLUDES self for RANGE nodes: a self-`Arc` in the
            // node's own cache is a refcount cycle that leaks the node (and
            // everything it references) forever. `UOp::ranges()` chains self
            // back on; parents add range-children explicitly below, so the
            // Tinygrad self-first order is preserved at every level.
            uop.op.map_child(|src| {
                if matches!(src.op, Op::Range(..)) && seen.insert(src.id) {
                    result.push(src.clone());
                }
                for r in RangesProperty::get(src) {
                    if seen.insert(r.id) {
                        result.push(r.clone());
                    }
                }
            });
            result
        }
    }
}

// ============================================================================
// In-Scope Ranges Property
// ============================================================================

cached_property! {
    /// Cached in-scope ranges property (recursive, like Tinygrad's `@recursive_property`).
    ///
    /// Returns only the RANGE operations that are "in scope" at this UOp,
    /// meaning they are currently active (not yet ended).
    ///
    /// Computed from children's cached values (guaranteed available by
    /// `CachedProperty::get()`'s filtered toposort + bottom-up processing):
    /// 1. Merge in-scope ranges from all source operations
    /// 2. Remove ranges ended by this operation (`op.ended_ranges()`)
    /// 3. Add self if this is a RANGE operation
    ///
    /// This is O(N) total for the first access on a graph, then O(1) for
    /// subsequent accesses on overlapping subgraphs (cached per-node).
    InScopeRangesProperty: crate::uop::core::RangeIds {
        cache_field: in_scope_ranges_cache,
        compute: |uop| {
            // Sorted, deduplicated ids: a few entries at most, so a linear scan
            // beats a hash table and the inline buffer avoids a heap allocation
            // per node.
            let mut result = crate::uop::core::RangeIds::new();
            // A PROGRAM is a closed kernel and its LINEAR list closes RANGEs by
            // position, not by an END edge, so no range leaks out of either.
            if matches!(uop.op, Op::Linear(..) | Op::Program(..)) {
                return result;
            }

            // Step 1: Merge from all sources' cached in_scope_ranges
            uop.op.map_child(|src| result.extend_from_slice(InScopeRangesProperty::get(src)));

            // Step 2: Remove ended ranges (using existing op.ended_ranges())
            for ended in uop.op.ended_ranges() {
                match ended.op() {
                    Op::Range(..) => result.retain(|id| *id != ended.id),
                    // Non-RANGE ended (like AFTER) — remove all its in-scope ranges
                    _ => {
                        let ended_scope = InScopeRangesProperty::get(ended);
                        result.retain(|id| !ended_scope.contains(id));
                    }
                }
            }

            // Step 3: Add self if RANGE. Stored as an id, not an `Arc` — a
            // self-`Arc` in the node's own cache would be a refcount cycle
            // (permanent leak); ids pin nothing.
            if matches!(uop.op, Op::Range(..)) {
                result.push(uop.id);
            }

            result.sort_unstable();
            result.dedup();
            result
        }
    }
}

// ============================================================================
// VminVmax Property
// ============================================================================

cached_property! {
    /// Cached vmin/vmax range analysis property.
    ///
    /// Computes the minimum and maximum possible values for a UOp based on
    /// operation semantics and input ranges. Returns a tuple of (vmin, vmax)
    /// where both values are ConstValue types.
    ///
    /// The analysis is conservative - when in doubt, it returns the full dtype
    /// bounds to avoid incorrect optimizations.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use svod_ir::uop::properties::VminVmaxProperty;
    /// use svod_ir::uop::cached_property::CachedProperty;
    ///
    /// let (vmin, vmax) = VminVmaxProperty::get(&my_uop);
    /// println!("Value range: [{:?}, {:?}]", vmin, vmax);
    /// ```
    VminVmaxProperty: (ConstValue, ConstValue) {
        cache_field: vmin_vmax_cache,
        compute: |uop| {
            crate::uop::range_eval::compute_vmin_vmax(uop)
                .unwrap_or_else(|| crate::uop::range_eval::analysis_bounds(&uop.dtype))
        }
    }
}

// Sound vmin/vmax: returns None for ops without provably correct bounds.
// Use this for optimizations that collapse expressions to constants.
cached_property! {
    SoundVminVmaxProperty: Option<(ConstValue, ConstValue)> {
        cache_field: sound_vmin_vmax_cache,
        compute: |uop| crate::uop::range_eval::compute_sound_vmin_vmax(uop)
    }
}

// ============================================================================
// Weak Float Property
// ============================================================================

cached_property! {
    /// Cached "backward slice contains a weak-float dtype" predicate.
    ///
    /// Equivalent to `uop.toposort().iter().any(|n| n.dtype().base() == WeakFloat)`,
    /// but memoised per node: O(N) once over the whole graph instead of O(N) per
    /// query. Value-sensitive symbolic rewrites use it as a match guard, so it is
    /// evaluated on every pattern attempt.
    HasWeakFloatProperty: bool {
        cache_field: has_weak_float_cache,
        compute: |uop| {
            if uop.dtype.base() == svod_dtype::ScalarDType::WeakFloat {
                return true;
            }
            let mut result = false;
            uop.op.map_child(|src| result |= *HasWeakFloatProperty::get(src));
            result
        }
    }
}

// ============================================================================
// Device / Address Space Properties
// ============================================================================

cached_property! {
    /// Cached device specification carried by a node's backward slice.
    ///
    /// Tinygrad memoises the equivalent walk with `@functools.cached_property`
    /// (`ops.py` `UOp.device`). Without the memo the plain recursion revisits
    /// every shared node once per path, which is exponential on diamond DAGs.
    DeviceSpecProperty: Option<svod_dtype::DeviceSpec> {
        cache_field: device_spec_cache,
        compute: |uop| uop.compute_device_spec()
    }
}

cached_property! {
    /// Cached storage address space (Tinygrad: `UOp.addrspace`, also memoised).
    AddrSpaceProperty: Option<svod_dtype::AddrSpace> {
        cache_field: addrspace_cache,
        compute: |uop| uop.compute_addrspace()
    }
}
