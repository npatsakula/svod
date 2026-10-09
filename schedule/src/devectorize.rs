//! Devectorize pass — single-pass combined matcher.
//!
//! Composition: `symbolic_simple + devectorizer2`
//!
//! All patterns run in one `graph_rewrite` call with fixed-point convergence.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::LazyLock;

use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::{AxisId, BinaryOp, ConstValue, Op, ReduceOp, UOp, UOpKey};

use crate::TypedPatternMatcher;
use smallvec::SmallVec;
use svod_ir::ops;

/// Sentinel `UOp::tag` value marking an END as merge-eligible.
///
/// Only ENDs produced by `reduce_to_acc` carry this tag; later decompositions
/// emit ENDs without it, and those are
/// intentionally ignored by the merge step. Picked outside the small-integer
/// index-tag space used by rangeify so it cannot collide with tracking tags.
pub const TAG_MERGEABLE: usize = 0xFFFF_FFFF_FFFF_FFFE;

#[inline]
fn is_mergeable_end(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::End(..)) && uop.tag().as_ref().is_some_and(|t| t.contains(&TAG_MERGEABLE))
}

/// Sorted ids of the END's *enclosing* (still-in-scope) ranges. Two ENDs
/// with equal reduce ranges but different enclosing contexts (e.g. one
/// nested inside an outer reduce, the other at top level) hash to different
/// keys and are kept apart in the merge step — the inner group's RANGEs are
/// cloned with fresh axis ids so each RANGE maps to exactly one END.
fn end_context_ids(end: &Arc<UOp>) -> SmallVec<[u64; 4]> {
    let mut ids: SmallVec<[u64; 4]> = end.in_scope_ranges().iter().copied().collect();
    ids.sort_unstable();
    ids
}

fn structural_node_keys(nodes: impl IntoIterator<Item = Arc<UOp>>) -> Vec<u64> {
    let mut keys: Vec<_> = nodes.into_iter().map(|node| node.content_hash).collect();
    keys.sort();
    keys
}

/// Smallest unused `AxisId::Renumbered(_)` value across `sink`'s subgraph.
fn next_axis_after(sink: &Arc<UOp>) -> usize {
    sink.toposort()
        .iter()
        .filter_map(|n| match n.op() {
            Op::Range(ops::Range { axis_id, .. }) => Some(axis_id.value()),
            _ => None,
        })
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
}

/// Clone a RANGE node with a fresh `axis_id`, preserving end / type / deps.
/// Used by the merge step when two ENDs share reduce ranges but live at
/// different nesting depths — the inner group must own its own RANGEs.
fn clone_range_with_axis(range: &Arc<UOp>, new_axis_id: AxisId) -> Arc<UOp> {
    if let Op::Range(ops::Range { end, axis_type, deps, .. }) = range.op() {
        UOp::new(
            Op::Range(ops::Range { end: end.clone(), axis_id: new_axis_id, axis_type: *axis_type, deps: deps.clone() }),
            range.dtype(),
        )
    } else {
        unreachable!("clone_range_with_axis called on non-Range")
    }
}

/// Context for REDUCE transformation.
///
/// Tracks END nodes created per reduce-range set so that multiple ENDs sharing
/// the same ranges can be merged into a single END with a GROUP body. Only
/// ENDs tagged `TAG_MERGEABLE` are tracked.
#[derive(Debug, Default)]
pub struct ReduceContext {
    range_to_ends: HashMap<SmallVec<[u64; 4]>, Vec<Arc<UOp>>>,
    next_reg_slot: usize,
}

impl ReduceContext {
    fn next_reg(&mut self) -> usize {
        let slot = self.next_reg_slot;
        self.next_reg_slot += 1;
        slot
    }

    /// Register an END node under its reduce-range key. Non-mergeable ENDs
    /// are silently ignored.
    pub fn register_end(&mut self, end: &Arc<UOp>) {
        if !is_mergeable_end(end) {
            return;
        }
        if let Op::End(ops::End { ranges, .. }) = end.op() {
            let mut key: SmallVec<[u64; 4]> = ranges.iter().map(|r| r.id).collect();
            key.sort_unstable();
            self.range_to_ends.entry(key).or_default().push(end.clone());
        }
    }

    /// Merge END nodes that share the same reduce ranges *and* nesting
    /// context. ENDs at different nesting depths are kept apart by cloning
    /// RANGEs with fresh axis ids.
    pub fn merge_reduce_ends(&mut self, sources: &SmallVec<[Arc<UOp>; 4]>) -> Option<Arc<UOp>> {
        let temp_sink = UOp::sink(sources.to_vec());
        let mut next_axis = next_axis_after(&temp_sink);
        let subs = build_end_merge_subs(&self.range_to_ends, &mut next_axis);
        self.range_to_ends.clear();
        if subs.is_empty() {
            return None;
        }
        Some(temp_sink.substitute(&subs))
    }
}

/// Build per-END substitutions implementing the two-level merge:
/// outer key = sorted reduce-range ids, inner sub-key = sorted enclosing
/// (`in_scope`) range ids. Only outer groups with >1 ENDs participate; for
/// each such outer group, sub-groups beyond the first get cloned RANGEs
/// (`AxisId::Renumbered(*next_axis + j)`) so each RANGE is associated with
/// at most one merged END.
fn build_end_merge_subs(
    range_to_ends: &HashMap<SmallVec<[u64; 4]>, Vec<Arc<UOp>>>,
    next_axis: &mut usize,
) -> HashMap<UOpKey, Arc<UOp>> {
    let mut subs = HashMap::new();
    let mut range_groups: Vec<_> = range_to_ends.values().collect();
    range_groups.sort_by_cached_key(|ends| {
        let Op::End(ops::End { ranges, .. }) = ends[0].op() else { unreachable!() };
        structural_node_keys(ranges.iter().cloned())
    });
    for ends in range_groups {
        if ends.len() <= 1 {
            continue;
        }
        let original_ranges: SmallVec<[Arc<UOp>; 4]> = match ends[0].op() {
            Op::End(ops::End { ranges, .. }) => ranges.clone(),
            _ => unreachable!(),
        };

        // Sub-group by enclosing context.
        let mut by_ctx: HashMap<SmallVec<[u64; 4]>, Vec<Arc<UOp>>> = HashMap::new();
        for end in ends {
            by_ctx.entry(end_context_ids(end)).or_default().push(end.clone());
        }

        let mut contexts: Vec<_> = by_ctx.into_iter().collect();
        contexts.sort_by_cached_key(|(_, group)| {
            let scope = group[0].in_scope_ranges();
            structural_node_keys(group[0].ranges().into_iter().filter(|r| scope.contains(&r.id)))
        });
        for (i, (_, group)) in contexts.into_iter().enumerate() {
            // First sub-group keeps original ranges; subsequent ones get clones
            // so the same RANGE is never reachable from two distinct merged ENDs.
            let target_ranges: SmallVec<[Arc<UOp>; 4]> = if i == 0 {
                original_ranges.clone()
            } else {
                let cloned: SmallVec<[Arc<UOp>; 4]> = original_ranges
                    .iter()
                    .enumerate()
                    .map(|(j, rr)| clone_range_with_axis(rr, AxisId::Renumbered(*next_axis + j)))
                    .collect();
                *next_axis += original_ranges.len();
                cloned
            };

            // For cloned sub-groups, walk each END's subtree and replace
            // every reference to an original RANGE with its clone — without
            // this the END would still close the original range and we'd
            // reintroduce the cross-context merge we just split.
            let mapped: Vec<Arc<UOp>> = if i == 0 {
                group.clone()
            } else {
                let mut sub_map: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
                for (old, new) in original_ranges.iter().zip(target_ranges.iter()) {
                    sub_map.insert(UOpKey(old.clone()), new.clone());
                }
                group.iter().map(|e| e.substitute(&sub_map)).collect()
            };

            // Singleton sub-groups skip the GROUP wrapper entirely (a single
            // END with cloned ranges is the merge result). Multi-element
            // sub-groups collapse into `GROUP(computations).end(target_ranges)`.
            let merged = if mapped.len() == 1 {
                mapped[0].clone()
            } else {
                let computations: Vec<Arc<UOp>> = mapped
                    .iter()
                    .map(|e| match e.op() {
                        Op::End(ops::End { computation, .. }) => computation.clone(),
                        _ => unreachable!(),
                    })
                    .collect();
                UOp::group(computations).end(target_ranges)
            };

            for e in &group {
                subs.insert(UOpKey(e.clone()), merged.clone());
            }
        }
    }
    subs
}

pub(crate) fn merge_register_read_ends(root: Arc<UOp>) -> Arc<UOp> {
    let mut range_to_ends: HashMap<SmallVec<[u64; 4]>, Vec<Arc<UOp>>> = HashMap::new();
    for node in root.toposort() {
        let Op::After(ops::After { passthrough, deps }) = node.op() else { continue };
        if passthrough.addrspace() != Some(AddrSpace::Reg) {
            continue;
        }
        for dep in deps {
            let Op::End(ops::End { ranges, .. }) = dep.op() else { continue };
            if !ranges.iter().all(|range| matches!(range.op(), Op::Range(..))) {
                continue;
            }
            let mut key: SmallVec<[u64; 4]> = ranges.iter().map(|range| range.id).collect();
            key.sort_unstable();
            let ends = range_to_ends.entry(key).or_default();
            if !ends.iter().any(|existing| existing.id == dep.id) {
                ends.push(dep.clone());
            }
        }
    }
    let mut next_axis = next_axis_after(&root);
    let substitutions = build_end_merge_subs(&range_to_ends, &mut next_axis);
    if substitutions.is_empty() { root } else { root.substitute(&substitutions) }
}

use crate::optimizer::Renderer;
use crate::rewrite::graph_rewrite;
use crate::symbolic::patterns::symbolic_simple;
use svod_ir::shape::{Shape, broadcast_shapes, shapes_equal};

// ============================================================================
// Main Entry Point
// ============================================================================

/// Run devectorize pass. Call AFTER `pre_expand`, BEFORE codegen.
///
/// One `graph_rewrite` over the combined matcher, matching tinygrad
/// `codegen/__init__.py:333` (`symbolic_simple+devectorizer2+indexing_simplify`).
/// `graph_rewrite` already re-runs the matcher on every replacement, so an outer
/// fixed-point loop would only paper over a missing pattern.
pub fn devectorize(ast: &Arc<UOp>, renderer: &Renderer) -> Arc<UOp> {
    static COMBINED: LazyLock<TypedPatternMatcher<Renderer>> = LazyLock::new(|| {
        symbolic_simple().clone().with_context::<Renderer>()
            + devectorize_patterns().with_context::<Renderer>()
            + bool_storage_patterns().clone().with_context::<Renderer>()
            + crate::late::indexing_simplify().clone().with_context::<Renderer>()
    });
    graph_rewrite(&*COMBINED, ast.clone(), &mut renderer.clone())
}

/// Bool LOAD/STORE via uint8. LLVM i1 can have garbage in upper bits.
/// Also rewrites BitCast involving Bool to Cast (bitcast requires same bit-width).
pub fn bool_storage_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // STORE bool: cast to uint8 before storing
        Store { index, value, gate } if value.dtype().base().is_bool() && !UOp::is_invalid_marker(value) => {
            let uint8_dtype = value.dtype().with_base(ScalarDType::UInt8);
            Some(UOp::new(
                Op::Store(ops::Store { index: index.clone(), value: value.cast(uint8_dtype), gate: gate.clone() }),
                DType::Void,
            ))
        },

        // LOAD bool: load as uint8, then cast to bool
        load @ Load { index, alt, gate } if load.dtype().base().is_bool() => {
            let uint8_dtype = load.dtype().with_base(ScalarDType::UInt8);
            let uint8_alt = alt.clone().map(|a| a.cast(uint8_dtype.clone()));
            let uint8_load = UOp::load()
                .index(index.with_dtype(uint8_dtype))
                .maybe_alt(uint8_alt)
                .maybe_gate(gate.clone())
                .call();
            Some(uint8_load.cast(load.dtype()))
        },

        // BitCast with Bool: i1 has different bit-width than i8+, use Cast instead
        BitCast { src, dtype } if src.dtype().base().is_bool() || dtype.base().is_bool() => {
            Some(src.cast(dtype.clone()))
        },
    }
}

// ============================================================================
// FP8 Float Decomposition
// ============================================================================

/// Context for FP8 float decomposition.
/// `from` is the FP8 dtype being decomposed, `to` is the target float dtype.
#[derive(Debug, Clone)]
pub struct Fp8DecompCtx {
    pub from: ScalarDType,
    pub to: ScalarDType,
}

impl Fp8DecompCtx {
    fn should_decomp(&self, u: &Arc<UOp>) -> bool {
        u.dtype().base() == self.from
    }
}

/// Round-to-nearest-even for integer bitwise rounding.
/// Round-to-nearest-even for `v` shifted right by `s` bits.
fn rne(v: &Arc<UOp>, s: u32) -> Arc<UOp> {
    let one = v.const_like(1);
    let shifted = v.shr(&v.const_like(s));
    let half_bit = v.shr(&v.const_like(s - 1)).and_(&one);
    let remainder_mask = v.const_like((1i64 << (s - 1)) - 1);
    let has_remainder = v.and_(&remainder_mask).ne(&v.const_like(0)).cast(v.dtype());
    let lsb = shifted.and_(&one);
    let round_up = half_bit.and_(&has_remainder.or_(&lsb));
    shifted.try_add(&round_up).expect("rne: add failed")
}

/// Bitwise float-to-float format conversion.
/// Float-to-float bitwise conversion for FP8 → narrower float types.
///
/// `v` is a UInt value holding the raw bits of the source float.
/// Returns a UOp holding raw bits of the target float, which must be bitcast to get the float value.
fn f2f(v: &Arc<UOp>, fr: ScalarDType, to: ScalarDType) -> Arc<UOp> {
    let (fe, fm) = fr.finfo().expect("f2f operands are float by construction");
    let (te, tm) = to.finfo().expect("f2f operands are float by construction");
    let fs = fr.bitsize();
    let ts = to.bitsize();
    let fb = fr.exponent_bias().expect("f2f operands are float by construction") as i64;
    let tb = to.exponent_bias().expect("f2f operands are float by construction") as i64;
    let fr_uint = DType::Scalar(fr.float_to_uint().expect("f2f operands are float by construction"));
    let to_uint = DType::Scalar(to.float_to_uint().expect("f2f operands are float by construction"));

    if fe <= te && fm < tm {
        // Upcast path: e.g. FP8 → Float16
        let sign_mask = v.const_like(1i64 << (fs - 1));
        let sign = v.and_(&sign_mask).cast(to_uint.clone()).shl(&v.const_like(ts - fs).cast(to_uint.clone()));
        let nosign_mask = v.const_like((1i64 << (fs - 1)) - 1);
        let nosign = v.and_(&nosign_mask).cast(to_uint.clone());
        let exp = nosign.shr(&nosign.const_like(fm));
        let norm = nosign
            .shl(&nosign.const_like(tm - fm))
            .try_add(&nosign.const_like((tb - fb) << tm))
            .expect("f2f: add failed");
        let nan_val = nosign.shl(&nosign.const_like(tm - fm)).or_(&nosign.const_like(((1i64 << te) - 1) << tm));
        let zero = nosign.const_like(0);

        if matches!(fr, ScalarDType::FP8E4M3FNUZ | ScalarDType::FP8E5M2FNUZ) {
            let fnuz_nan = sign.ne(&sign.const_like(0)).and_(&nosign.eq(&nosign.const_like(0)));
            let qnan = nosign.const_like((((1i64 << te) - 1) << tm) | (1i64 << (tm - 1)));
            let flush_limit = (fb - tb).max(0) + 1;
            let converted = exp.lt(&exp.const_like(flush_limit));
            let value = UOp::try_where(converted, zero.clone(), norm).expect("f2f: where failed");
            return UOp::try_where(fnuz_nan, qnan, sign.or_(&value))
                .expect("f2f: where failed")
                .bitcast(DType::Scalar(to));
        }

        // FP8E4M3 has a single NaN value (all exponent+mantissa bits set)
        let is_nan = if fr == ScalarDType::FP8E4M3 {
            nosign.eq(&nosign.const_like((1i64 << (fm + fe)) - 1))
        } else {
            exp.eq(&exp.const_like((1i64 << fe) - 1))
        };

        let exp_is_zero = exp.eq(&zero);
        let inner = UOp::try_where(is_nan, nan_val, norm).expect("f2f: where failed");
        let result = UOp::try_where(exp_is_zero, zero, inner).expect("f2f: where failed");
        sign.or_(&result).bitcast(DType::Scalar(to))
    } else if fe >= te && fm > tm {
        // Downcast path: e.g. Float16 → FP8
        let clamped = f2f_clamp(&v.bitcast(DType::Scalar(fr)), to);
        let v = clamped.bitcast(fr_uint);
        let sign = v.shr(&v.const_like(fs - ts)).and_(&v.const_like(1i64 << (ts - 1)));
        let nosign_mask = v.const_like((1i64 << (fs - 1)) - 1);
        let nosign = v.and_(&nosign_mask);
        let norm = rne(&nosign, fm - tm)
            .try_sub(&nosign.const_like((fb - tb) << tm))
            .expect("f2f: sub failed")
            .cast(to_uint.clone());

        let exp_field = nosign.shr(&nosign.const_like(fm)).and_(&nosign.const_like((1i64 << fe) - 1));
        let underflow = exp_field.lt(&exp_field.const_like(1 + fb - tb));

        let nan_mantissa = if to == ScalarDType::FP8E4M3 {
            sign.const_like((1i64 << tm) - 1).cast(to_uint.clone())
        } else {
            nosign.shr(&nosign.const_like(fm - tm)).and_(&nosign.const_like((1i64 << tm) - 1)).cast(to_uint.clone())
        };
        let nan_exp = sign.const_like(((1i64 << te) - 1) << tm).cast(to_uint.clone());
        let nan = sign.cast(to_uint.clone()).or_(&nan_mantissa).or_(&nan_exp);

        let is_nan = exp_field.eq(&exp_field.const_like((1i64 << fe) - 1));
        let zero = sign.const_like(0).cast(to_uint.clone());
        let normal = sign.cast(to_uint.clone()).or_(&UOp::try_where(underflow, zero, norm).expect("f2f: where failed"));
        if matches!(to, ScalarDType::FP8E4M3FNUZ | ScalarDType::FP8E5M2FNUZ) {
            let fnuz_nan = sign.const_like(1i64 << (ts - 1)).cast(to_uint);
            UOp::try_where(is_nan, fnuz_nan, normal).expect("f2f: where failed")
        } else {
            UOp::try_where(is_nan, nan, normal).expect("f2f: where failed")
        }
    } else {
        panic!("f2f: unsupported conversion {fr:?} -> {to:?}")
    }
}

/// Clamp a float value to the representable range of a target FP8 dtype.
/// Clamp a float value to the representable range of the target FP8 dtype.
fn f2f_clamp(val: &Arc<UOp>, dt: ScalarDType) -> Arc<UOp> {
    let (e, m) = dt.finfo().expect("f2f_clamp target is float by construction");
    let (max_exp, max_man): (i64, i64) = if matches!(dt, ScalarDType::FP8E4M3FNUZ | ScalarDType::FP8E5M2FNUZ) {
        ((1 << e) - 1, (1 << m) - 1)
    } else if dt == ScalarDType::FP8E4M3 {
        ((1 << e) - 1, (1 << m) - 2)
    } else {
        ((1 << e) - 2, (1 << m) - 1)
    };
    let mx_f64 = f64::powi(
        2.0,
        (max_exp - dt.exponent_bias().expect("f2f_clamp target is float by construction") as i64) as i32,
    ) * (1.0 + max_man as f64 / (1i64 << m) as f64);
    let mx = val.const_like(mx_f64);
    let neg_mx = val.const_like(-mx_f64);

    // For FP8 types, clamp to ±max; for others, clamp to ±inf
    let sat = if dt.is_fp8() { mx.clone() } else { val.const_like(f64::INFINITY) };
    let neg_sat = if dt.is_fp8() { neg_mx.clone() } else { val.const_like(f64::NEG_INFINITY) };

    // nan → nan, < -mx → -sat, > mx → sat, otherwise → val
    let is_nan = val.ne(val);
    let below = val.lt(&neg_mx);
    let above = mx.lt(val);
    let clamped_above = UOp::try_where(above, sat, val.clone()).expect("f2f_clamp: where failed");
    let clamped = UOp::try_where(below, neg_sat, clamped_above).expect("f2f_clamp: where failed");
    UOp::try_where(is_nan, val.clone(), clamped).expect("f2f_clamp: where failed")
}

const TAG_DTYPE_DECOMP: usize = 0xFFFF_FFFF_FFFF_FFFC;

fn dtype_decomp_tag(dtype: ScalarDType) -> SmallVec<[usize; 2]> {
    SmallVec::from_slice(&[TAG_DTYPE_DECOMP, dtype as usize])
}

fn has_dtype_decomp_tag(uop: &Arc<UOp>, dtype: ScalarDType) -> bool {
    uop.tag().as_ref().is_some_and(|tag| tag.as_slice() == [TAG_DTYPE_DECOMP, dtype as usize])
}

/// Float storage/ALU decomposition from Tinygrad's `pm_float_decomp`.
pub fn pm_float_decomp() -> crate::TypedPatternMatcher<Fp8DecompCtx> {
    crate::patterns! {
        @context Fp8DecompCtx;

        // Defines, INDEX, and SHRINK retain the emulated storage dtype as a tag.
        x if matches!(x.op(), Op::Param(..) | Op::Buffer(..) | Op::Index(..) | Op::Shrink(..))
            && ctx.should_decomp(x)
        => {
            let uint = DType::Scalar(ctx.from.float_to_uint()?);
            let dtype = if matches!(x.dtype(), DType::Ptr { .. }) { x.dtype().with_ptr_base(uint)? } else { uint };
            let rewritten = match x.op() {
                Op::Param(ops::Param { shape, arg }) => {
                    let mut arg = arg.clone();
                    arg.dtype = DType::Scalar(ctx.from.float_to_uint()?);
                    UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg }), dtype)
                }
                Op::Buffer(ops::Buffer { shape, arg }) => {
                    let mut arg = arg.clone();
                    arg.dtype = DType::Scalar(ctx.from.float_to_uint()?);
                    UOp::new(Op::Buffer(ops::Buffer { shape: shape.clone(), arg }), dtype)
                }
                _ => x.with_dtype(dtype),
            };
            Some(rewritten.with_tag(dtype_decomp_tag(ctx.from)))
        },

        // Pattern 2: LOAD with FP8 dtype → load as UInt8, f2f upcast to target float
        load @ Load { index, alt, gate }
            if ctx.should_decomp(load) =>
        {
            let lanes = load
                .shape()
                .ok()
                .flatten()
                .and_then(|shape| shape.iter().try_fold(1usize, |n, dim| n.checked_mul(dim.vmax()?)))
                .unwrap_or_else(|| load.dtype().vcount());
            let uint_scalar = DType::Scalar(ctx.from.float_to_uint()?);
            let convert_alt = |a: &Arc<UOp>| {
                if a.dtype().base() == ctx.from {
                    a.bitcast(uint_scalar.clone())
                } else {
                    let target_float = DType::Scalar(ctx.to);
                    let target_uint = DType::Scalar(ctx.to.float_to_uint().expect("f2f target is float by construction"));
                    let float_alt = a.cast(target_float);
                    f2f(&float_alt.bitcast(target_uint), ctx.to, ctx.from)
                }
            };
            let scalar_load = |lane: usize| {
                let reindex = |value: &Arc<UOp>| {
                    let lane = lane as i64;
                    match value.op() {
                        Op::Shrink(ops::Shrink { src, offsets, .. }) => UOp::index()
                            .buffer(src.clone())
                            .indices(vec![offsets.try_add(&offsets.const_like(lane)).expect("late FP8 offset must add")])
                            .call()
                            .expect("late FP8 SHRINK reindex must be valid"),
                        Op::Index(ops::Index { buffer, indices }) => {
                            let mut indices = indices.clone();
                            indices[0] = indices[0]
                                .try_add(&indices[0].const_like(lane))
                                .expect("late FP8 index offset must add");
                            UOp::index()
                                .buffer(buffer.clone())
                                .indices(indices)
                                .call()
                                .expect("late FP8 INDEX reindex must be valid")
                        }
                        _ => panic!("late FP8 reindex requires INDEX or SHRINK, got {:?}", value.op()),
                    }
                };
                let index = if lanes == 1 { index.clone() } else { reindex(index) };
                let alt = alt.as_ref().map(|a| if lanes == 1 { a.clone() } else { reindex(a) });
                let gate = gate.as_ref().map(|g| if lanes == 1 { g.clone() } else { reindex(g) });
                let uint_load = UOp::load()
                    .index(index.with_dtype(uint_scalar.clone()).with_tag(dtype_decomp_tag(ctx.from)))
                    .dtype(uint_scalar.clone())
                    .maybe_alt(alt.as_ref().map(convert_alt))
                    .maybe_gate(gate)
                    .call();
                f2f(&uint_load, ctx.from, ctx.to)
            };
            Some(if lanes == 1 { scalar_load(0) } else { UOp::stack((0..lanes).map(scalar_load).collect()) })
        },

        // A bitcasted load reads the raw storage word directly.
        _bc @ BitCast { src: ld, dtype } if matches!(ld.op(), Op::Load(..)) && ld.dtype().base() == ctx.from => {
            Some(ld.with_dtype(DType::Scalar(ctx.from.float_to_uint()?)).bitcast(dtype.clone()))
        },

        // Bitcast from the emulating float to a same-width destination.
        bc @ BitCast { src: x, dtype } if x.dtype().base() == ctx.to && dtype.base().bitsize() == ctx.from.bitsize() => {
            let raw = f2f(&x.bitcast(DType::Scalar(ctx.to.float_to_uint()?)), ctx.to, ctx.from);
            Some(bc.with_sources(vec![raw]))
        },

        // Bitcast to the emulated float format.
        BitCast { src: x, dtype } if dtype.base() == ctx.from => {
            Some(f2f(&x.bitcast(DType::Scalar(ctx.from.float_to_uint()?)), ctx.from, ctx.to))
        },

        Cast { src: val, dtype } if dtype.base() == ctx.from => {
            Some(f2f_clamp(&val.cast(DType::Scalar(ctx.to)), ctx.from))
        },

        x @ Const(_) if x.dtype().base() == ctx.from => {
            let Op::Const(value) = x.op() else { unreachable!() };
            Some(UOp::const_(DType::Scalar(ctx.to), value.0))
        },

        // Pattern 6: Any op with FP8 output dtype → promote to target float, cast FP8 sources
        x if !matches!(x.op(), Op::BitCast(..))
            && x.dtype().is_float()
            && ctx.should_decomp(x)
        => {
            let target_dtype = DType::Scalar(ctx.to);
            let new_dtype = if x.dtype().vcount() > 1 {
                target_dtype.vec(x.dtype().vcount()).expect("scalar dtype is vectorizable")
            } else {
                target_dtype.clone()
            };
            let new_sources: Vec<Arc<UOp>> = x.op().sources().iter().map(|s| {
                if s.dtype().base() == ctx.from {
                    s.cast(target_dtype.clone())
                } else {
                    s.clone()
                }
            }).collect();
            Some(x.with_sources(new_sources).with_dtype(new_dtype))
        },

        // STORE of a raw bitcast can keep the word directly.
        Store { index, value, gate } if (has_dtype_decomp_tag(index, ctx.from) || index.dtype().base() == ctx.from)
            && matches!(value.op(), Op::BitCast(..)) && value.dtype().base() == ctx.from => {
            let index = index.with_dtype(DType::Scalar(ctx.from.float_to_uint()?)).with_tag(dtype_decomp_tag(ctx.from));
            Some(UOp::new(
                Op::Store(ops::Store { index, value: value.with_dtype(DType::Scalar(ctx.from.float_to_uint()?)), gate: gate.clone() }),
                DType::Void,
            ))
        },

        Store { index, value, gate } if (has_dtype_decomp_tag(index, ctx.from) || index.dtype().base() == ctx.from)
            && (value.dtype().base() == ctx.to || value.dtype().base() == ctx.from) => {
            let index = index.with_dtype(DType::Scalar(ctx.from.float_to_uint()?)).with_tag(dtype_decomp_tag(ctx.from));
            let value = value.cast(DType::Scalar(ctx.to));
            let raw = f2f(&value.bitcast(DType::Scalar(ctx.to.float_to_uint()?)), ctx.to, ctx.from);
            Some(UOp::new(Op::Store(ops::Store { index, value: raw, gate: gate.clone() }), DType::Void))
        },
    }
}

fn is_ocp_fp8(dtype: &DType) -> bool {
    matches!(dtype.base(), ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2)
}

fn widen_non_native_fp8(node: &Arc<UOp>) -> Option<Arc<UOp>> {
    let float_dtype = || node.dtype().with_base(ScalarDType::Float32);
    let widen = |source: &Arc<UOp>| source.cast(source.dtype().with_base(ScalarDType::Float32));
    match node.op() {
        Op::Unary(op, source) if is_ocp_fp8(&node.dtype()) => {
            Some(UOp::new(Op::Unary(*op, widen(source)), float_dtype()).cast(node.dtype()))
        }
        Op::Binary(op, lhs, rhs) if is_ocp_fp8(&node.dtype()) => {
            Some(UOp::new(Op::Binary(*op, widen(lhs), widen(rhs)), float_dtype()).cast(node.dtype()))
        }
        Op::Binary(op, lhs, rhs)
            if node.dtype() == DType::Bool && is_ocp_fp8(&lhs.dtype()) && is_ocp_fp8(&rhs.dtype()) =>
        {
            Some(UOp::new(Op::Binary(*op, widen(lhs), widen(rhs)), DType::Bool))
        }
        Op::Ternary(svod_ir::TernaryOp::Where, condition, if_true, if_false) if is_ocp_fp8(&node.dtype()) => {
            Some(UOp::try_where(condition.clone(), widen(if_true), widen(if_false)).ok()?.cast(node.dtype()))
        }
        Op::Ternary(op, first, second, third) if is_ocp_fp8(&node.dtype()) => Some(
            UOp::new(Op::Ternary(*op, widen(first), widen(second), widen(third)), float_dtype()).cast(node.dtype()),
        ),
        Op::Cast(ops::Cast { src, dtype }) if is_ocp_fp8(dtype) && src.dtype().base() != ScalarDType::Float32 => {
            Some(src.cast(src.dtype().with_base(ScalarDType::Float32)).cast(dtype.clone()))
        }
        Op::Cast(ops::Cast { src, dtype }) if is_ocp_fp8(&src.dtype()) && dtype.base() != ScalarDType::Float32 => {
            Some(src.cast(src.dtype().with_base(ScalarDType::Float32)).cast(dtype.clone()))
        }
        _ => None,
    }
}

/// AMD LLVM accepts OCP FP8 storage, conversions, and MFMA operands, but not
/// ordinary FP8 ALU. Widen only those ALU/cast nodes, matching Tinygrad's
/// `create_non_native_float_pats`; WMMA and memory nodes remain untouched.
pub fn amd_non_native_fp8_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        node if matches!(node.op(), Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) | Op::Cast(..))
            => widen_non_native_fp8(node),
    }
}

// ============================================================================
// 64-bit integer decomposition
// ============================================================================

const TAG_LONG_DECOMP: usize = 0xFFFF_FFFF_FFFF_FFFB;

fn long_word_dtype(dtype: ScalarDType) -> Option<ScalarDType> {
    match dtype {
        ScalarDType::Int64 => Some(ScalarDType::Int32),
        ScalarDType::UInt64 => Some(ScalarDType::UInt32),
        _ => None,
    }
}

fn long_tag(word: usize, dtype: ScalarDType) -> SmallVec<[usize; 2]> {
    SmallVec::from_slice(&[TAG_LONG_DECOMP, word, dtype as usize])
}

fn tagged_long(uop: &Arc<UOp>) -> Option<(usize, ScalarDType)> {
    let tag = uop.tag().as_ref()?;
    if tag.len() != 3 || tag[0] != TAG_LONG_DECOMP {
        return None;
    }
    let dtype = match tag[2] {
        x if x == ScalarDType::Int64 as usize => ScalarDType::Int64,
        x if x == ScalarDType::UInt64 as usize => ScalarDType::UInt64,
        _ => return None,
    };
    Some((tag[1], dtype))
}

fn long_part(uop: &Arc<UOp>, word: usize, dtype: ScalarDType) -> Arc<UOp> {
    uop.with_tag(long_tag(word, dtype))
}

fn long_bin(op: BinaryOp, a: Arc<UOp>, b: Arc<UOp>, dtype: DType) -> Arc<UOp> {
    UOp::new(Op::Binary(op, a, b), dtype)
}

/// A constant carrying the 32-bit *word* dtype of `from`.
///
/// `const_like` off one of the tagged long references keeps the 64-bit dtype, so the
/// constant is never word-split and the leftover node stalls the whole rewrite.
fn word_const(from: ScalarDType, value: i64) -> Arc<UOp> {
    let word = DType::Scalar(long_word_dtype(from).expect("word constant of a 64-bit dtype"));
    UOp::const_(
        word,
        if from == ScalarDType::Int64 { ConstValue::Int(value) } else { ConstValue::UInt(value as u32 as u64) },
    )
}

fn pair_sub(a: (Arc<UOp>, Arc<UOp>), b: (Arc<UOp>, Arc<UOp>)) -> (Arc<UOp>, Arc<UOp>) {
    let borrow = long_bin(BinaryOp::Lt, a.0.clone(), b.0.clone(), DType::Bool).cast(DType::UInt32);
    (
        long_bin(BinaryOp::Sub, a.0, b.0, DType::UInt32),
        long_bin(BinaryOp::Sub, long_bin(BinaryOp::Sub, a.1, b.1, DType::UInt32), borrow, DType::UInt32),
    )
}

fn pair_neg(a: (Arc<UOp>, Arc<UOp>)) -> (Arc<UOp>, Arc<UOp>) {
    let zero = UOp::const_(DType::UInt32, ConstValue::UInt(0));
    pair_sub((zero.clone(), zero), a)
}

fn pair_lt(a: &(Arc<UOp>, Arc<UOp>), b: &(Arc<UOp>, Arc<UOp>)) -> Arc<UOp> {
    let high_lt = long_bin(BinaryOp::Lt, a.1.clone(), b.1.clone(), DType::Bool);
    let high_eq = long_bin(BinaryOp::Eq, a.1.clone(), b.1.clone(), DType::Bool);
    let low_lt = long_bin(BinaryOp::Lt, a.0.clone(), b.0.clone(), DType::Bool);
    long_bin(BinaryOp::Or, high_lt, long_bin(BinaryOp::And, high_eq, low_lt, DType::Bool), DType::Bool)
}

fn pair_where(cond: Arc<UOp>, a: (Arc<UOp>, Arc<UOp>), b: (Arc<UOp>, Arc<UOp>)) -> (Arc<UOp>, Arc<UOp>) {
    (
        UOp::try_where(cond.clone(), a.0, b.0).expect("long pair where"),
        UOp::try_where(cond, a.1, b.1).expect("long pair where"),
    )
}

fn long_divrem(
    a: (Arc<UOp>, Arc<UOp>),
    b: (Arc<UOp>, Arc<UOp>),
    signed: bool,
    remainder: bool,
) -> (Arc<UOp>, Arc<UOp>) {
    let zero = UOp::const_(DType::UInt32, ConstValue::UInt(0));
    let a_neg = if signed {
        long_bin(
            BinaryOp::Lt,
            a.1.clone().bitcast(DType::Int32),
            UOp::const_(DType::Int32, ConstValue::Int(0)),
            DType::Bool,
        )
    } else {
        UOp::const_(DType::Bool, ConstValue::Bool(false))
    };
    let b_neg = if signed {
        long_bin(
            BinaryOp::Lt,
            b.1.clone().bitcast(DType::Int32),
            UOp::const_(DType::Int32, ConstValue::Int(0)),
            DType::Bool,
        )
    } else {
        UOp::const_(DType::Bool, ConstValue::Bool(false))
    };
    let ua = if signed { pair_where(a_neg.clone(), pair_neg(a.clone()), a) } else { a };
    let ub = if signed { pair_where(b_neg.clone(), pair_neg(b.clone()), b) } else { b };
    let mut q = (zero.clone(), zero.clone());
    let mut r = (zero.clone(), zero.clone());
    for bit in (0..64).rev() {
        let carry = long_bin(BinaryOp::Shr, r.0.clone(), r.0.const_like(31), DType::UInt32);
        r = (
            long_bin(BinaryOp::Shl, r.0, UOp::const_(DType::UInt32, ConstValue::UInt(1)), DType::UInt32),
            long_bin(
                BinaryOp::Or,
                long_bin(BinaryOp::Shl, r.1, UOp::const_(DType::UInt32, ConstValue::UInt(1)), DType::UInt32),
                carry,
                DType::UInt32,
            ),
        );
        let source = if bit < 32 { ua.0.clone() } else { ua.1.clone() };
        let input_bit = long_bin(
            BinaryOp::And,
            long_bin(
                BinaryOp::Shr,
                source,
                UOp::const_(DType::UInt32, ConstValue::UInt((bit % 32) as u64)),
                DType::UInt32,
            ),
            UOp::const_(DType::UInt32, ConstValue::UInt(1)),
            DType::UInt32,
        );
        r.0 = long_bin(BinaryOp::Or, r.0, input_bit, DType::UInt32);
        let ge = UOp::new(Op::Unary(svod_ir::UnaryOp::Not, pair_lt(&r, &ub)), DType::Bool);
        let diff = pair_sub(r.clone(), ub.clone());
        r = pair_where(ge.clone(), diff, r);
        let qbit = long_bin(
            BinaryOp::Shl,
            ge.cast(DType::UInt32),
            UOp::const_(DType::UInt32, ConstValue::UInt((bit % 32) as u64)),
            DType::UInt32,
        );
        if bit < 32 {
            q.0 = long_bin(BinaryOp::Or, q.0, qbit, DType::UInt32);
        } else {
            q.1 = long_bin(BinaryOp::Or, q.1, qbit, DType::UInt32);
        }
    }
    if !signed {
        return if remainder { r } else { q };
    }
    if remainder {
        pair_where(a_neg, pair_neg(r.clone()), r)
    } else {
        let sign = long_bin(BinaryOp::Xor, a_neg, b_neg, DType::Bool);
        pair_where(sign, pair_neg(q.clone()), q)
    }
}

fn decompose_long_node(x: &Arc<UOp>) -> Option<Arc<UOp>> {
    use BinaryOp::*;

    // Definitions become twice as many 32-bit words.
    if matches!(x.op(), Op::Param(..) | Op::Buffer(..))
        && let Some(from) = long_word_dtype(x.dtype().base())
    {
        let sources = x.op().sources();
        let doubled = sources.first().map(|shape| shape.mul(&shape.const_like(2)));
        let dtype = if matches!(x.dtype(), DType::Ptr { .. }) {
            x.dtype().with_ptr_base(DType::Scalar(from))?
        } else {
            DType::Scalar(from)
        };
        let shape = doubled?;
        let op = match x.op() {
            Op::Param(ops::Param { arg, .. }) => {
                let mut arg = arg.clone();
                arg.dtype = DType::Scalar(from);
                Op::Param(ops::Param { shape, arg })
            }
            Op::Buffer(ops::Buffer { arg, .. }) => {
                let mut arg = arg.clone();
                arg.dtype = DType::Scalar(from);
                Op::Buffer(ops::Buffer { shape, arg })
            }
            _ => unreachable!(),
        };
        return Some(UOp::new(op, dtype));
    }

    // A tagged INDEX selects one of the two adjacent words.
    if let Some((word, from)) = tagged_long(x)
        && let Op::Index(ops::Index { buffer, indices }) = x.op()
    {
        // Re-type exactly as the buffer above: INDEX over a global buffer carries the
        // element dtype, and `with_ptr_base` is `None` for anything but a Ptr -- which
        // aborted the whole arm and left both words addressing the same element.
        let word_dt = long_word_dtype(from)?;
        let dtype = if matches!(x.dtype(), DType::Ptr { .. }) {
            x.dtype().with_ptr_base(DType::Scalar(word_dt))?
        } else {
            x.dtype().with_base(word_dt)
        };
        let mut indices = indices.clone();
        let index = indices.last_mut()?;
        *index = index.mul(&index.const_like(2)).add(&index.const_like(word as i64));
        return Some(UOp::new(Op::Index(ops::Index { buffer: buffer.clone(), indices }), dtype));
    }

    // Split each 64-bit STORE into low/high 32-bit stores.
    if let Op::Store(ops::Store { index, value, gate }) = x.op()
        && let Some(from) = long_word_dtype(index.dtype().base())
        && tagged_long(value).is_none()
    {
        let original = index.dtype().base();
        let stores = (0..2)
            .map(|word| {
                let idx = long_part(index, word, original);
                let val = long_part(value, word, original);
                UOp::new_tagged(
                    Op::Store(ops::Store { index: idx, value: val, gate: gate.clone() }),
                    DType::Void,
                    Some(long_tag(word, original)),
                )
            })
            .collect();
        let _ = from;
        return Some(UOp::group(stores));
    }
    if matches!(x.op(), Op::Store(..)) {
        return None;
    }

    // Comparisons consume both words but produce one bool.
    if let Op::Binary(op @ (Lt | Eq | Ne), a, b) = x.op()
        && long_word_dtype(a.dtype().base()).is_some()
    {
        let from = a.dtype().base();
        let (a0, a1) = (long_part(a, 0, from), long_part(a, 1, from));
        let (b0, b1) = (long_part(b, 0, from), long_part(b, 1, from));
        return Some(match op {
            Eq => long_bin(And, long_bin(Eq, a0, b0, DType::Bool), long_bin(Eq, a1, b1, DType::Bool), DType::Bool),
            Ne => long_bin(Or, long_bin(Ne, a0, b0, DType::Bool), long_bin(Ne, a1, b1, DType::Bool), DType::Bool),
            Lt => {
                let high_lt = long_bin(Lt, a1.clone(), b1.clone(), DType::Bool);
                let high_eq = long_bin(Eq, a1, b1, DType::Bool);
                let low_lt = long_bin(Lt, a0.bitcast(DType::UInt32), b0.bitcast(DType::UInt32), DType::Bool);
                long_bin(Or, high_lt, long_bin(And, high_eq, low_lt, DType::Bool), DType::Bool)
            }
            _ => unreachable!(),
        });
    }

    // Cast away from a long reconstructs the value from its two words.
    if let Op::Cast(ops::Cast { src, dtype }) = x.op()
        && let Some(word) = long_word_dtype(src.dtype().base())
        && long_word_dtype(dtype.base()).is_none()
    {
        let from = src.dtype().base();
        // BITCAST pins each word at 32 bits. A bare tagged reference still reports the
        // 64-bit dtype, so the comparison rule below would re-split it against its own
        // words and `a0.cast(dtype)` would re-enter this very arm, feeding a WHERE that
        // is its own source -- which deadlocks the rewrite into returning the input.
        let a0 = long_part(src, 0, from).bitcast(DType::Scalar(word));
        let a1 = long_part(src, 1, from).bitcast(DType::Scalar(word));
        let a0u = long_part(src, 0, from).bitcast(DType::UInt32);
        if dtype.is_float() {
            let small = long_bin(
                Or,
                long_bin(
                    And,
                    long_bin(Eq, word_const(from, 0), a1.clone(), DType::Bool),
                    long_bin(Lt, word_const(from, -1), a0.clone(), DType::Bool),
                    DType::Bool,
                ),
                long_bin(
                    And,
                    long_bin(Eq, word_const(from, -1), a1.clone(), DType::Bool),
                    long_bin(Lt, a0.clone(), word_const(from, 0), DType::Bool),
                    DType::Bool,
                ),
                DType::Bool,
            );
            let direct = a0.cast(dtype.clone());
            let high = long_bin(
                Mul,
                a1.cast(DType::Float32),
                UOp::const_(DType::Float32, ConstValue::Float(4294967296.0)),
                DType::Float32,
            );
            let combined = long_bin(Add, high, a0u.cast(DType::Float32), DType::Float32).cast(dtype.clone());
            return UOp::try_where(small, direct, combined).ok();
        }
        return Some(a0u.cast(dtype.clone()));
    }

    let (word, from) = tagged_long(x)?;
    let word_dt = DType::Scalar(long_word_dtype(from)?);

    if let Op::Load(ops::Load { index, alt, gate }) = x.op() {
        let index = long_part(index, word, from);
        let alt = alt.as_ref().map(|v| long_part(v, word, from));
        return Some(UOp::load().index(index).maybe_alt(alt).maybe_gate(gate.clone()).call());
    }

    if let Op::Const(value) = x.op() {
        let bits = match value.0 {
            ConstValue::Int(v) => v as u64,
            ConstValue::UInt(v) => v,
            _ => panic!("long decomposition of CONST unsupported"),
        };
        let part = if word == 0 { bits as u32 } else { (bits >> 32) as u32 };
        return Some(UOp::const_(
            word_dt,
            if from == ScalarDType::Int64 {
                ConstValue::Int(part as i32 as i64)
            } else {
                ConstValue::UInt(part as u64)
            },
        ));
    }

    if let Op::Cast(ops::Cast { src, .. }) = x.op() {
        if let Some(src_word) = long_word_dtype(src.dtype().base()) {
            return Some(long_part(src, word, src.dtype().base()).bitcast(DType::Scalar(src_word)).cast(word_dt));
        }
        let lo = src.cast(word_dt.clone());
        if word == 0 {
            return Some(lo);
        }
        if src.dtype().is_float() {
            let scaled = long_bin(Fdiv, src.clone(), src.const_like(4294967296.0), src.dtype());
            let correction = long_bin(
                And,
                long_bin(Lt, src.clone(), src.const_like(0), DType::Bool),
                long_bin(Ne, lo.clone(), lo.const_like(0), DType::Bool),
                DType::Bool,
            )
            .cast(word_dt.clone());
            return Some(long_bin(Sub, scaled.cast(word_dt.clone()), correction, word_dt));
        }
        let sign = if src.dtype().base() == ScalarDType::Bool { lo.clone() } else { src.clone() };
        let negative = long_bin(Lt, sign.clone(), sign.const_like(0), DType::Bool);
        return UOp::try_where(negative, lo.const_like(-1), lo.const_like(0)).ok();
    }

    if let Op::BitCast(ops::BitCast { src, .. }) = x.op() {
        return Some(long_part(src, word, src.dtype().base()).bitcast(word_dt));
    }

    if let Op::Ternary(svod_ir::TernaryOp::Where, condition, a, b) = x.op() {
        return UOp::try_where(condition.clone(), long_part(a, word, from), long_part(b, word, from)).ok();
    }

    if let Op::Unary(svod_ir::UnaryOp::Neg, src) = x.op() {
        let a0 = long_part(src, 0, from);
        let a1 = long_part(src, 1, from);
        let low = long_bin(Sub, word_const(from, 0), a0.clone(), word_dt.clone());
        return Some(if word == 0 {
            low
        } else {
            let zero = UOp::const_(DType::UInt32, ConstValue::UInt(0));
            let borrow = long_bin(Lt, zero, a0.bitcast(DType::UInt32), DType::Bool).cast(word_dt.clone());
            long_bin(Sub, long_bin(Sub, word_const(from, 0), a1, word_dt.clone()), borrow, word_dt)
        });
    }

    if let Op::Binary(op, a, b) = x.op() {
        let (a0, a1) = (long_part(a, 0, from), long_part(a, 1, from));
        let (b0, b1) = (long_part(b, 0, from), long_part(b, 1, from));
        return Some(match op {
            Add => {
                let low = long_bin(Add, a0.clone(), b0, word_dt.clone());
                if word == 0 {
                    low.clone()
                } else {
                    let carry = long_bin(Lt, low.bitcast(DType::UInt32), a0.bitcast(DType::UInt32), DType::Bool)
                        .cast(word_dt.clone());
                    long_bin(Add, long_bin(Add, a1, b1, word_dt.clone()), carry, word_dt)
                }
            }
            Sub => {
                if word == 0 {
                    long_bin(Sub, a0, b0, word_dt)
                } else {
                    let borrow = long_bin(Lt, a0.bitcast(DType::UInt32), b0.bitcast(DType::UInt32), DType::Bool)
                        .cast(word_dt.clone());
                    long_bin(Sub, long_bin(Sub, a1, b1, word_dt.clone()), borrow, word_dt)
                }
            }
            Mul => {
                let mask = UOp::const_(DType::UInt32, ConstValue::UInt(0xFFFF));
                let shift = UOp::const_(DType::UInt32, ConstValue::UInt(16));
                let a0u = a0.bitcast(DType::UInt32);
                let b0u = b0.bitcast(DType::UInt32);
                let a00 = long_bin(And, a0u.clone(), mask.clone(), DType::UInt32);
                let a01 = long_bin(Shr, a0u.clone(), shift.clone(), DType::UInt32);
                let b00 = long_bin(And, b0u.clone(), mask.clone(), DType::UInt32);
                let b01 = long_bin(Shr, b0u.clone(), shift.clone(), DType::UInt32);
                let p0 = long_bin(Mul, a00.clone(), b00.clone(), DType::UInt32);
                let p1 = long_bin(Mul, a00, b01.clone(), DType::UInt32);
                let p2 = long_bin(Mul, a01.clone(), b00, DType::UInt32);
                let t = long_bin(
                    Add,
                    long_bin(
                        Add,
                        long_bin(Shr, p0.clone(), shift.clone(), DType::UInt32),
                        long_bin(And, p1.clone(), mask.clone(), DType::UInt32),
                        DType::UInt32,
                    ),
                    long_bin(And, p2.clone(), mask.clone(), DType::UInt32),
                    DType::UInt32,
                );
                if word == 0 {
                    long_bin(
                        Or,
                        long_bin(And, p0, mask, DType::UInt32),
                        long_bin(Shl, t, shift, DType::UInt32),
                        DType::UInt32,
                    )
                    .bitcast(word_dt)
                } else {
                    let high = long_bin(
                        Add,
                        long_bin(
                            Add,
                            long_bin(Mul, a01, b01, DType::UInt32),
                            long_bin(Shr, p1, shift.clone(), DType::UInt32),
                            DType::UInt32,
                        ),
                        long_bin(
                            Add,
                            long_bin(Shr, p2, shift.clone(), DType::UInt32),
                            long_bin(Shr, t, shift, DType::UInt32),
                            DType::UInt32,
                        ),
                        DType::UInt32,
                    );
                    long_bin(
                        Add,
                        long_bin(
                            Add,
                            high,
                            long_bin(Mul, a0u, b1.bitcast(DType::UInt32), DType::UInt32),
                            DType::UInt32,
                        ),
                        long_bin(Mul, a1.bitcast(DType::UInt32), b0u, DType::UInt32),
                        DType::UInt32,
                    )
                    .bitcast(word_dt)
                }
            }
            Shl | Shr => {
                let wconst = |v: i64| word_const(from, v);
                let uconst = |v: u64| UOp::const_(DType::UInt32, ConstValue::UInt(v));
                // `n` is the shift inside one word; `ge32` picks the "shift crosses the word" case.
                let n = long_bin(And, b0.clone(), wconst(31), word_dt.clone());
                let nu = n.clone().bitcast(DType::UInt32);
                let ge32 = long_bin(Lt, wconst(31), b0.clone(), DType::Bool);
                if *op == Shl {
                    // carry = a0 >>u (32 - n), spelled as two shifts so n == 0 stays in range.
                    let carry = long_bin(
                        Shr,
                        long_bin(Shr, a0.clone().bitcast(DType::UInt32), uconst(1), DType::UInt32),
                        long_bin(Sub, uconst(31), nu, DType::UInt32),
                        DType::UInt32,
                    )
                    .bitcast(word_dt.clone());
                    let low = long_bin(Shl, a0, n.clone(), word_dt.clone());
                    let high = long_bin(Or, long_bin(Shl, a1, n, word_dt.clone()), carry, word_dt.clone());
                    if word == 0 {
                        UOp::try_where(ge32, wconst(0), low).expect("long shift where")
                    } else {
                        // shift >= 32: the high word is the low word shifted by n = shift - 32.
                        UOp::try_where(ge32, low, high).expect("long shift where")
                    }
                } else {
                    // carry = a1 <<u (32 - n), spelled as two shifts so n == 0 stays in range.
                    let carry = long_bin(
                        Shl,
                        long_bin(Shl, a1.clone().bitcast(DType::UInt32), uconst(1), DType::UInt32),
                        long_bin(Sub, uconst(31), nu.clone(), DType::UInt32),
                        DType::UInt32,
                    )
                    .bitcast(word_dt.clone());
                    // The low word always shifts logically: an Int32 `Shr` is arithmetic and
                    // would smear the sign of `a0` into the bits `carry` supplies.
                    let low = long_bin(
                        Or,
                        long_bin(Shr, a0.bitcast(DType::UInt32), nu, DType::UInt32).bitcast(word_dt.clone()),
                        carry,
                        word_dt.clone(),
                    );
                    // shift >= 32: the low word is the high word shifted by n = shift - 32.
                    let high = long_bin(Shr, a1.clone(), n, word_dt.clone());
                    let fill = if from == ScalarDType::Int64 {
                        long_bin(Shr, a1.clone(), wconst(31), word_dt.clone())
                    } else {
                        wconst(0)
                    };
                    if word == 0 {
                        UOp::try_where(ge32, high.clone(), low).expect("long shift where")
                    } else {
                        UOp::try_where(ge32, fill, high).expect("long shift where")
                    }
                }
            }
            Max => {
                let cmp = UOp::new(Op::Binary(Lt, a.clone(), b.clone()), DType::Bool);
                let selected = UOp::try_where(cmp, b.clone(), a.clone()).expect("long max where");
                long_part(&selected, word, from)
            }
            CDiv | CMod => {
                let pair = long_divrem(
                    (a0.bitcast(DType::UInt32), a1.bitcast(DType::UInt32)),
                    (b0.bitcast(DType::UInt32), b1.bitcast(DType::UInt32)),
                    from == ScalarDType::Int64,
                    *op == CMod,
                );
                (if word == 0 { pair.0 } else { pair.1 }).bitcast(word_dt)
            }
            And | Or | Xor => long_bin(*op, if word == 0 { a0 } else { a1 }, if word == 0 { b0 } else { b1 }, word_dt),
            _ => panic!("long decomposition of {op:?} unsupported"),
        });
    }

    panic!("long decomposition of {:?} unsupported", x.op())
}

#[allow(unused_variables)]
pub fn pm_long_decomp() -> crate::TypedPatternMatcher {
    crate::patterns! {
        x @ Param { shape, arg } => { let _ = (shape, arg); decompose_long_node(x) },
        x @ Buffer { shape, arg } => { let _ = (shape, arg); decompose_long_node(x) },
        x @ Index { buffer, indices } => { let _ = (buffer, indices); decompose_long_node(x) },
        x @ Store { index, value, gate } => { let _ = (index, value, gate); decompose_long_node(x) },
        x @ Load { index, alt, gate } => { let _ = (index, alt, gate); decompose_long_node(x) },
        x @ Const(_) => decompose_long_node(x),
        x @ Cast { src, dtype } => { let _ = (src, dtype); decompose_long_node(x) },
        x @ BitCast { src, dtype } => { let _ = (src, dtype); decompose_long_node(x) },
        for op in unary [Neg] { x @ op(_) => decompose_long_node(x), }
        for op in binary [Add, Sub, Mul, CDiv, CMod, Max, Lt, Eq, Ne, And, Or, Xor, Shl, Shr] {
            x @ op(_, _) => decompose_long_node(x),
        }
        x @ Where(_, _, _) => decompose_long_node(x),
    }
}

// ============================================================================
// ALU Devectorization
// ============================================================================

/// Elementwise devectorization matching Tinygrad's `do_devectorize`.
///
/// Broadcasting must already be unpacked: every source has the result shape,
/// except Invalid, whose scalar base is polymorphic.
fn devectorize_alu(alu: &Arc<UOp>) -> Option<Arc<UOp>> {
    let shape = alu.shape().ok().flatten()?.clone();
    if shape.is_empty() {
        return None;
    }
    let static_shape: Vec<usize> = shape.iter().map(|dim| dim.as_const()).collect::<Option<_>>()?;
    let lane_count: usize = static_shape.iter().product();

    let sources = alu.op().sources();
    let invalid_base = |source: &Arc<UOp>| {
        let base = source.base();
        matches!(base.op(), Op::Const(value) if value.0 == ConstValue::Invalid).then_some(base)
    };

    if !sources.iter().all(|source| {
        invalid_base(source).is_some()
            || source.shape().ok().flatten().is_some_and(|source_shape| source_shape == &shape)
            || (source.shape().ok().flatten().is_none() && source.dtype().vcount() == lane_count)
    }) {
        return None;
    }

    let mut coordinates = vec![Vec::new()];
    for extent in static_shape {
        coordinates = coordinates
            .into_iter()
            .flat_map(|prefix| {
                (0..extent).map(move |index| {
                    let mut coordinate = prefix.clone();
                    coordinate.push(index);
                    coordinate
                })
            })
            .collect();
    }

    let elements: SmallVec<[Arc<UOp>; 4]> = coordinates
        .into_iter()
        .map(|coordinate| {
            let new_sources: Vec<Arc<UOp>> = sources
                .iter()
                .enumerate()
                .map(|(source_index, source)| {
                    if let Some(invalid) = invalid_base(source) {
                        Ok(invalid)
                    } else if matches!(alu.op(), Op::Store(..)) && source_index == 0 {
                        let indices: SmallVec<[Arc<UOp>; 4]> =
                            coordinate.iter().map(|&index| UOp::index_const(index as i64)).collect();
                        let (selected, deps) = match source.op() {
                            Op::Stack(ops::Stack { sources }) => (const_index_into_stack(sources, &indices), None),
                            Op::After(ops::After { passthrough, deps })
                                if matches!(passthrough.op(), Op::Stack(..)) =>
                            {
                                let Op::Stack(ops::Stack { sources }) = passthrough.op() else { unreachable!() };
                                (const_index_into_stack(sources, &indices), Some(deps))
                            }
                            _ => (UOp::index().buffer(source.clone()).indices(indices).call().ok(), None),
                        };
                        let selected = selected.ok_or(svod_ir::Error::IndexOutOfBounds)?;
                        let selected = match selected.op() {
                            Op::Load(ops::Load { index, .. }) => index.clone(),
                            _ => selected,
                        };
                        if let (Some(deps), Op::Index(ops::Index { buffer, indices })) = (deps, selected.op()) {
                            let ordered = buffer.after(deps.clone());
                            Ok(selected.with_sources(std::iter::once(ordered).chain(indices.iter().cloned()).collect()))
                        } else {
                            Ok(selected)
                        }
                    } else {
                        coordinate.iter().try_fold(source.clone(), |value, &index| {
                            UOp::index().buffer(value).indices(vec![UOp::index_const(index as i64)]).call()
                        })
                    }
                })
                .collect::<svod_ir::Result<_>>()
                .ok()?;
            let dtype = if alu.dtype() == DType::Void { DType::Void } else { DType::Scalar(alu.dtype().base()) };
            if matches!(alu.op(), Op::Cast(..) | Op::BitCast(..))
                && new_sources.len() == 1
                && new_sources[0].dtype() == dtype
            {
                return Some(new_sources[0].clone());
            }
            Some(alu.with_sources(new_sources).with_dtype(dtype))
        })
        .collect::<Option<_>>()?;

    if matches!(alu.op(), Op::Store(..)) {
        Some(UOp::group(elements.into_vec()))
    } else {
        stack_with_shape(elements.into_vec(), &shape)
    }
}

fn expand_to_shape(source: &Arc<UOp>, shape: &Shape) -> Option<Arc<UOp>> {
    let source_shape = source.shape().ok().flatten()?;
    let source = if source_shape.len() < shape.len() {
        let mut padded = smallvec::smallvec![svod_ir::SInt::Const(1); shape.len() - source_shape.len()];
        padded.extend(source_shape.iter().cloned());
        source.try_reshape(&padded).ok()?
    } else {
        source.clone()
    };
    source.try_expand(shape).ok()
}

fn expand_broadcast(x: &Arc<UOp>) -> Option<Arc<UOp>> {
    let shapes =
        x.op().sources().iter().map(|source| source.shape().ok().flatten().cloned()).collect::<Option<Vec<_>>>()?;
    if shapes.windows(2).all(|pair| shapes_equal(&pair[0], &pair[1])) {
        return None;
    }
    let shape = broadcast_shapes(&shapes).ok()?;
    Some(x.with_sources(x.op().sources().iter().map(|source| expand_to_shape(source, &shape)).collect::<Option<_>>()?))
}

fn broadcast_and_devec_wmma(wmma: &Arc<UOp>) -> Option<Arc<UOp>> {
    let sources = wmma.op().sources();
    let source_shapes =
        sources.iter().map(|source| source.shape().ok().flatten().cloned()).collect::<Option<Vec<_>>>()?;
    if source_shapes.iter().any(|shape| shape.is_empty()) {
        return None;
    }
    let prefixes: Vec<_> = source_shapes.iter().map(|shape| shape[..shape.len() - 1].into()).collect();
    if prefixes.windows(2).all(|pair| shapes_equal(&pair[0], &pair[1])) {
        return None;
    }
    let prefix = broadcast_shapes(&prefixes).ok()?;
    let expanded = sources
        .iter()
        .zip(source_shapes.iter())
        .map(|(source, source_shape)| {
            let mut shape = prefix.clone();
            shape.push(source_shape.last()?.clone());
            expand_to_shape(source, &shape)
        })
        .collect::<Option<Vec<_>>>()?;
    let output_shape = wmma.shape().ok().flatten()?.clone();
    let mut coordinates = vec![Vec::new()];
    for extent in &output_shape[..output_shape.len().saturating_sub(1)] {
        let extent = extent.as_const()?;
        coordinates = coordinates
            .into_iter()
            .flat_map(|prefix| {
                (0..extent).map(move |i| {
                    let mut coordinate = prefix.clone();
                    coordinate.push(UOp::index_const(i as i64));
                    coordinate
                })
            })
            .collect();
    }
    let lanes = coordinates
        .into_iter()
        .map(|coordinate| {
            let indexed = expanded
                .iter()
                .map(|source| UOp::index().buffer(source.clone()).indices(coordinate.clone()).call())
                .collect::<svod_ir::Result<Vec<_>>>()
                .ok()?;
            Some(wmma.with_sources(indexed))
        })
        .collect::<Option<Vec<_>>>()?;
    UOp::stack(lanes.into()).try_reshape(&output_shape).ok()
}

pub(crate) fn stack_with_shape(mut elements: Vec<Arc<UOp>>, shape: &[svod_ir::SInt]) -> Option<Arc<UOp>> {
    fn build(elements: &[Arc<UOp>], shape: &[svod_ir::SInt]) -> Option<Arc<UOp>> {
        if shape.is_empty() {
            return (elements.len() == 1).then(|| elements[0].clone());
        }
        let count = shape[0].as_const()?;
        let chunk = elements.len().checked_div(count)?;
        // A zero-sized dimension leaves no elements to chunk; `chunks(0)` panics.
        if chunk == 0 || count * chunk != elements.len() {
            return None;
        }
        Some(UOp::stack(elements.chunks(chunk).map(|part| build(part, &shape[1..])).collect::<Option<_>>()?))
    }

    if shape.is_empty() && elements.len() == 1 { elements.pop() } else { build(&elements, shape) }
}

/// Tinygrad `pm_expand_broadcast`, kept before `pm_add_loads` in the target pass.
pub fn pm_expand_broadcast() -> &'static TypedPatternMatcher {
    static PM: LazyLock<TypedPatternMatcher> = LazyLock::new(|| {
        pm_wmma_add().clone()
            + crate::patterns! {
                x if matches!(x.op(), Op::Binary(..) | Op::Ternary(..) | Op::Store(..)) => expand_broadcast(x),
                wmma @ Wmma { a: _, b: _, c: _, metadata: _ } => broadcast_and_devec_wmma(wmma),
            }
    });
    &PM
}

/// Vector ALU → STACK of scalar ALU. LLVM SLP can re-vectorize when beneficial.
#[allow(unused_variables)]
pub fn no_vectorized_alu() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // All binary ops
        for op in binary [*] {
            alu @ op(_, _) if alu.shape().ok().flatten().is_some_and(|shape| !shape.is_empty()) => devectorize_alu(alu),
        },
        // All unary ops
        for op in unary [*] {
            alu @ op(_) if alu.shape().ok().flatten().is_some_and(|shape| !shape.is_empty()) => devectorize_alu(alu),
        },
        // All ternary ops (Where, MulAcc)
        for op in ternary [*] {
            alu @ op(_, _, _) if alu.shape().ok().flatten().is_some_and(|shape| !shape.is_empty()) => devectorize_alu(alu),
        },
        // Cast and BitCast
        alu @ Cast { src: _, .. } if alu.shape().ok().flatten().is_some_and(|shape| !shape.is_empty()) => devectorize_alu(alu),
        alu @ BitCast { src: _, .. } if alu.shape().ok().flatten().is_some_and(|shape| !shape.is_empty()) => devectorize_alu(alu),
    }
}

#[allow(unused_variables)]
pub fn mixed_representation_alu() -> &'static TypedPatternMatcher {
    fn is_mixed(uop: &Arc<UOp>) -> bool {
        let sources = uop.op().sources();
        sources.iter().any(|source| matches!(source.op(), Op::Stack(..)))
            && (uop.dtype().vcount() > 1 || sources.iter().any(|source| source.dtype().vcount() > 1))
    }

    fn devectorize_mixed(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
        let shape = uop.shape().ok().flatten()?.clone();
        let sources = uop
            .op()
            .sources()
            .iter()
            .map(|source| {
                let source = if source.shape().ok().flatten().is_none() && source.dtype().vcount() > 1 {
                    UOp::stack(
                        (0..source.dtype().vcount()).map(|lane| source.index_axes(vec![lane])).collect::<SmallVec<_>>(),
                    )
                } else {
                    source.clone()
                };
                expand_to_shape(&source, &shape)
            })
            .collect::<Option<Vec<_>>>()?;
        devectorize_alu(&uop.with_sources(sources))
    }

    crate::cached_patterns! {
        alu @ Where(_, _, _) if is_mixed(alu) => devectorize_mixed(alu),
        for op in binary [*] {
            alu @ op(_, _) if is_mixed(alu) => devectorize_mixed(alu),
        },
        for op in ternary [*] {
            alu @ op(_, _, _) if is_mixed(alu) => devectorize_mixed(alu),
        },
    }
}

// ============================================================================
// Devectorize Patterns
// ============================================================================

/// Tinygrad `devectorizer2`, preserving source order after movement cleanup.
pub fn devectorize_patterns() -> &'static TypedPatternMatcher {
    use std::sync::LazyLock;
    static CACHED: LazyLock<TypedPatternMatcher> = LazyLock::new(|| {
        movement_cleanup_patterns()
            + crate::rangeify::patterns::movement_op_patterns()
            + no_vectorized_alu()
            + mixed_representation_alu()
            + crate::cached_patterns! {
                load @ Load { index: _, .. } if load.shape().ok().flatten().is_some_and(|shape| !shape.is_empty())
                    => devectorize_alu(load),
                store @ Store { index: _, .. } if store.shape().ok().flatten().is_some_and(|shape| !shape.is_empty())
                    => devectorize_alu(store),

                // INDEX without indices is the source itself.
                Index { buffer, indices } if indices.is_empty() => Some(buffer.clone()),

                // WMMA operands must be STACK or WMMA before later lowering.
                wmma @ Wmma { a: _, b: _, c: _, metadata: _ } => stack_wmma_sources(wmma),

                // A shaped index argument represents independent INDEX operations. The
                // lanes stay addresses -- tinygrad `codegen/__init__.py:155-157` -- because
                // `devectorize_alu` on the enclosing LOAD or STORE is what materialises the
                // per-lane LOAD(INDEX) / STORE(INDEX).
                Index { buffer, indices }
                    if matches!(buffer.op(), Op::Param(..) | Op::Buffer(..))
                        && indices.len() == 1 && matches!(indices[0].op(), Op::Stack(..))
                    => stack_index(buffer, &indices[0]),

                // INDEX(buffer, RESHAPE(index)) moves RESHAPE outside INDEX.
                Index { buffer, indices }
                    if matches!(buffer.op(), Op::Param(..) | Op::Buffer(..))
                        && indices.len() == 1 && matches!(indices[0].op(), Op::Reshape(..))
                    => index_through_reshape(buffer, &indices[0]),

                // RESHAPE(void) is only shape bookkeeping around AFTER/STORE.
                Reshape { src, .. } if src.dtype() == DType::Void => Some(src.clone()),

                // A one-element shaped value reshaped to scalar is an INDEX.
                reshape @ Reshape { src, .. } if reshape_to_scalar_singleton(reshape, src)
                    => UOp::index().buffer(src.clone()).indices(vec![UOp::index_const(0)]).call().ok(),

                // A one-dimensional scalar EXPAND is a STACK broadcast.
                expand @ Expand { src, .. } => materialize_stack_broadcast(expand, src),
                expand @ Expand { src, .. } => expand_scalar_to_stack(expand, src),

            }
    });
    &CACHED
}

fn materialize_leading_singletons(reshape: &Arc<UOp>, src: &Arc<UOp>) -> Option<Arc<UOp>> {
    let source_shape = src.shape().ok().flatten()?;
    let target_shape = reshape.shape().ok().flatten()?;
    if target_shape.len() <= source_shape.len()
        || target_shape[target_shape.len() - source_shape.len()..] != source_shape[..]
        || !target_shape[..target_shape.len() - source_shape.len()].iter().all(|dim| dim.as_const() == Some(1))
    {
        return None;
    }
    let mut value = src.clone();
    for _ in 0..target_shape.len() - source_shape.len() {
        value = UOp::stack(smallvec::smallvec![value]);
    }
    Some(value)
}

fn materialize_stack_broadcast(expand: &Arc<UOp>, src: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Stack(ops::Stack { sources }) = src.op() else { return None };
    let source_shape = src.shape().ok().flatten()?;
    let target_shape = expand.shape().ok().flatten()?;
    if sources.len() != 1 || source_shape.len() != target_shape.len() || source_shape[1..] != target_shape[1..] {
        return None;
    }
    let count = target_shape.first()?.as_const()?;
    Some(UOp::stack((0..count).map(|_| sources[0].clone()).collect()))
}

pub(crate) fn movement_cleanup_patterns() -> TypedPatternMatcher {
    mop_cleanup_patterns() + crate::cached_patterns! {
        // Devectorizer-only movement materialization. These are deliberately
        // excluded from the earlier Tinygrad-exact expander cleanup.
        reshape @ Reshape { src: Stack { sources }, new_shape: _ }
            if sources.len() == 1
                && sources[0].shape().ok().flatten().zip(reshape.shape().ok().flatten()).is_some_and(|(a, b)| a == b)
            => Some(sources[0].clone()),
        reshape @ Reshape { src, new_shape: _ } => materialize_leading_singletons(reshape, src),
    }
    .clone()
}

pub(crate) fn mop_cleanup_patterns() -> TypedPatternMatcher {
    crate::cached_patterns! {
        // movement.py mop_cleanup, in source order.
        reshape @ Reshape { src: _inner @ Reshape { src, .. }, new_shape }
            => Some(UOp::new(Op::Reshape(ops::Reshape { src: src.clone(), new_shape: new_shape.clone() }), reshape.dtype())),
        reshape @ Reshape { src, .. }
            if src.shape().ok().flatten().zip(reshape.shape().ok().flatten()).is_some_and(|(a, b)| a == b)
            => Some(src.clone()),
        Permute { src: _inner @ Permute { src, axes: inner_axes }, axes }
            => merge_permutes(src, inner_axes, axes),
        Permute { src, axes } if axes.iter().enumerate().all(|(i, axis)| i == *axis)
            => Some(src.clone()),
        stack @ Stack { sources } => collapse_sequential_stack_indices(stack, sources),
        Index { buffer: Stack { sources }, indices } => const_index_into_stack(sources, indices),
        Index { buffer: _inner @ Index { buffer, indices: inner_indices }, indices: outer_indices }
            if inner_indices.iter().chain(outer_indices.iter())
                .all(|index| index.shape().ok().flatten().is_some_and(|shape| shape.is_empty()))
            => UOp::index().buffer(buffer.clone())
                .indices(inner_indices.iter().chain(outer_indices.iter()).cloned().collect::<Vec<_>>()).call().ok(),
        Index { buffer: _inner @ Index { buffer, indices: inner_indices }, indices: outer_indices }
            if inner_indices.len() == 1
                && inner_indices[0].shape().ok().flatten().zip(Some(outer_indices.len())).is_some_and(|(shape, rank)| shape.len() == rank)
            => {
                let selected = UOp::index().buffer(inner_indices[0].clone()).indices(outer_indices.clone()).call().ok()?;
                UOp::index().buffer(buffer.clone()).indices(vec![selected]).call().ok()
            },
    }.clone()
}

fn collapse_sequential_stack_indices(stack: &Arc<UOp>, sources: &SmallVec<[Arc<UOp>; 4]>) -> Option<Arc<UOp>> {
    let first = sources.first()?;
    let Op::Index(ops::Index { buffer, indices }) = first.op() else { return None };
    if indices.len() != 1
        || stack.shape().ok().flatten()? != buffer.shape().ok().flatten()?
        || sources.iter().enumerate().any(|(position, source)| {
            !matches!(source.op(), Op::Index(ops::Index { buffer: candidate, indices })
                if Arc::ptr_eq(candidate, buffer)
                    && indices.len() == 1
                    && matches!(indices[0].op(), Op::Const(value)
                        if value.0.try_int().and_then(|value| usize::try_from(value).ok()) == Some(position)))
        })
    {
        return None;
    }
    Some(buffer.clone())
}

pub(crate) fn pm_scalarize_register_stack_index_preserve_deps() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        Index { buffer: After { passthrough: Stack { sources }, deps }, indices }
            => {
                let selected = const_index_into_stack(sources, indices)?;
                let Op::Load(ops::Load { index, alt, gate }) = selected.op() else { return None };
                if index.addrspace() != Some(AddrSpace::Reg) {
                    return None;
                }
                let Op::Index(ops::Index { buffer, indices }) = index.op() else { return None };
                let ordered = buffer.after(deps.clone());
                let index = index.with_sources(std::iter::once(ordered).chain(indices.iter().cloned()).collect());
                Some(selected.with_sources(
                    std::iter::once(index).chain(alt.iter().cloned()).chain(gate.iter().cloned()).collect(),
                ))
            },
    }
}

pub(crate) fn is_register_stack_index(node: &Arc<UOp>) -> bool {
    matches!(node.op(), Op::Index(ops::Index { buffer, .. })
        if matches!(buffer.op(), Op::After(ops::After { passthrough, .. })
            if matches!(passthrough.op(), Op::Stack(..))))
}

fn merge_permutes(src: &Arc<UOp>, inner_axes: &[usize], outer_axes: &[usize]) -> Option<Arc<UOp>> {
    let axes = outer_axes.iter().map(|axis| inner_axes.get(*axis).copied()).collect::<Option<Vec<_>>>()?;
    Some(UOp::new(Op::Permute(ops::Permute { src: src.clone(), axes }), src.dtype()))
}

fn const_index_into_stack(sources: &SmallVec<[Arc<UOp>; 4]>, indices: &SmallVec<[Arc<UOp>; 4]>) -> Option<Arc<UOp>> {
    let (first, rest) = indices.split_first()?;
    let Op::Const(value) = first.op() else { return None };
    let selected = sources.get(usize::try_from(value.0.try_int()?).ok()?)?.clone();
    if rest.is_empty() {
        Some(selected)
    } else if let Op::Stack(ops::Stack { sources }) = selected.op() {
        const_index_into_stack(sources, &rest.iter().cloned().collect())
    } else {
        UOp::index().buffer(selected).indices(rest.to_vec()).call().ok()
    }
}

fn stack_index(buffer: &Arc<UOp>, stack: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Stack(ops::Stack { sources }) = stack.op() else { return None };
    Some(UOp::stack(
        sources
            .iter()
            .map(|index| UOp::index().buffer(buffer.clone()).indices(vec![index.clone()]).call())
            .collect::<svod_ir::Result<_>>()
            .ok()?,
    ))
}

fn index_through_reshape(buffer: &Arc<UOp>, reshape: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reshape(ops::Reshape { src, .. }) = reshape.op() else { return None };
    let indexed = UOp::index().buffer(buffer.clone()).indices(vec![src.clone()]).call().ok()?;
    indexed.try_reshape(reshape.shape().ok().flatten()?).ok()
}

fn reshape_to_scalar_singleton(reshape: &Arc<UOp>, src: &Arc<UOp>) -> bool {
    reshape.shape().ok().flatten().is_some_and(|shape| shape.is_empty())
        && src.shape().ok().flatten().is_some_and(|shape| shape.len() == 1 && shape[0].as_const() == Some(1))
}

fn expand_scalar_to_stack(expand: &Arc<UOp>, src: &Arc<UOp>) -> Option<Arc<UOp>> {
    if !src.shape().ok().flatten()?.is_empty() {
        return None;
    }
    let shape = expand.shape().ok().flatten()?;
    if shape.len() != 1 {
        return None;
    }
    let count = shape[0].as_const()?;
    Some(UOp::stack((0..count).map(|_| src.clone()).collect()))
}

fn stack_wmma_sources(wmma: &Arc<UOp>) -> Option<Arc<UOp>> {
    let sources = wmma.op().sources();
    if sources.iter().all(|source| matches!(source.op(), Op::Stack(..) | Op::Wmma(..))) {
        return None;
    }
    if wmma.shape().ok().flatten()?.len() != 1 {
        return None;
    }
    let rewritten = sources
        .iter()
        .map(|source| {
            if matches!(source.op(), Op::Stack(..)) {
                return Some(source.clone());
            }
            let shape = source.shape().ok().flatten()?;
            let count = shape.iter().try_fold(1usize, |product, dim| Some(product * dim.as_const()?))?;
            Some(UOp::stack(
                (0..count)
                    .map(|i| {
                        // WMMA operands are values. tinygrad's `do_stack_wmma` runs after
                        // `pm_add_loads`, so its lanes are already loaded; this pass shares a
                        // rewrite with the index split, so the LOAD is materialised here.
                        let lane = UOp::index().buffer(source.clone()).indices(vec![UOp::index_const(i as i64)]);
                        Some(maybe_load(&lane.call().ok()?))
                    })
                    .collect::<Option<_>>()?,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(wmma.with_sources(rewritten))
}

// ============================================================================
// Add Loads Patterns
// ============================================================================

/// Materialize address-space operands at their value-consuming positions.
///
/// Mirrors Tinygrad's `maybe_load`/`pm_add_loads`: INDEX remains element-typed,
/// and only operands consumed as values are wrapped in LOAD.
pub fn pm_add_loads() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        x if matches!(x.op(),
            Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) |
            Op::Cast(..) | Op::BitCast(..) |
            Op::Reduce(..) | Op::Wmma(..) | Op::Stack(..)
        ) => add_loads_to_value_sources(x),

        Store { index, value, gate } if value.addrspace().is_some()
            => Some(UOp::new(
                Op::Store(ops::Store { index: index.clone(), value: maybe_load(value), gate: gate.clone() }),
                DType::Void,
            )),
    }
}

fn add_loads_to_value_sources(x: &Arc<UOp>) -> Option<Arc<UOp>> {
    let sources = x.op().sources();
    if !sources.iter().any(|source| source.addrspace().is_some()) {
        return None;
    }
    Some(x.with_sources(sources.iter().map(maybe_load).collect()))
}

fn maybe_load(value: &Arc<UOp>) -> Arc<UOp> {
    if value.addrspace().is_some() { UOp::load().index(value.clone()).call() } else { value.clone() }
}

// ============================================================================
// WMMA Accumulation Patterns
// ============================================================================

/// Move additions into a WMMA accumulator, including movement ops introduced by
/// output-axis reconstruction in the expander.
///
/// Tinygrad `codegen/__init__.py:108-115`. It is guard-free: `wmma.src[2]+add`
/// asserts inside `alu` on a dtype mismatch. `try_add` is the equivalent that
/// leaves the WMMA unfused instead of aborting.
pub fn pm_wmma_add() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        Add[wmma @ Wmma { a, b, c, metadata }, add] => {
            Some(UOp::new(
                Op::Wmma(ops::Wmma { a: a.clone(), b: b.clone(), c: c.try_add(add).ok()?, metadata: metadata.clone() }),
                wmma.dtype(),
            ))
        },

        Add[Permute { src: wmma @ Wmma { a: _, b: _, c: _, metadata: _ }, axes }, add] => {
            let moved = add.try_permute(crate::argsort(axes)).ok()?;
            wmma.try_add(&moved).ok()?.try_permute(axes.clone()).ok()
        },

        Add[Permute { src: reshape @ Reshape { src: wmma @ Wmma { a: _, b: _, c: _, metadata: _ }, .. }, axes }, add] => {
            let moved = add.try_permute(crate::argsort(axes)).ok()?.try_reshape(wmma.shape().ok().flatten()?).ok()?;
            wmma.try_add(&moved).ok()?
                .try_reshape(reshape.shape().ok().flatten()?).ok()?
                .try_permute(axes.clone()).ok()
        },
    }
}

// ============================================================================
// pm_reduce: Convert REDUCE to explicit accumulator pattern
// ============================================================================

use crate::symbolic::dce::reduce_identity;

/// Convert REDUCE to an explicit register `placeholder` + LOAD/STORE accumulation pattern.
///
/// Transforms:
/// ```text
/// REDUCE(src, ranges, Add) with dtype Float32
/// ```
///
/// To:
/// ```text
/// acc = placeholder(AddrSpace::Reg, 1, Float32)
/// idx = INDEX(acc, [0])
/// store_init = STORE(acc, idx, identity)  // Initialize with 0 for Add
/// // Loop body (ranges provide iteration):
/// acc_after = AFTER(acc, [store_init, ranges...])
/// idx_loop = INDEX(acc_after, [0])
/// val = LOAD(acc, idx_loop)
/// new_val = val + src
/// store_loop = STORE(acc, idx_loop, new_val)
/// // After loop:
/// end = END(store_loop, ranges)
/// acc_final = AFTER(acc, [end])
/// idx_final = INDEX(acc_final, [0])
/// result = LOAD(acc, idx_final)
/// ```
///
/// Runs EARLY (before pm_add_loads, before main devectorize) to eliminate
/// REDUCE before other patterns see it.
pub fn pm_reduce_local() -> TypedPatternMatcher<ReduceContext> {
    pm_wmma_add().clone().with_context::<ReduceContext>()
        + crate::expand::pm_group_for_reduce().clone().with_context()
        + crate::patterns! {
            @context ReduceContext;

            // Ranged reduction conversion precedes horizontal-only reduction.
            red @ Reduce { .. } if matches!(red.op(), Op::Reduce(ops::Reduce { ranges, .. }) if !ranges.is_empty())
                => reduce_to_acc(red, ctx),

            red @ Reduce { .. } if matches!(red.op(), Op::Reduce(ops::Reduce { ranges, .. }) if ranges.is_empty())
                => expand_horizontal_reduce(red),

            // Merge END nodes sharing the same reduce ranges.
            Sink { sources: _sources } => {
                ctx.merge_reduce_ends(_sources)
            },
        }
        + clean_up_group_sink().with_context()
}

/// Compatibility name for callers that apply reduction lowering in isolation.
pub fn pm_reduce() -> TypedPatternMatcher<ReduceContext> {
    pm_reduce_local()
}

fn clean_up_group_sink() -> TypedPatternMatcher {
    crate::patterns! {
        Group { sources } if sources.len() == 1 => Some(sources[0].clone()),

        root @ Sink { sources }
            if sources.iter().any(|source| matches!(source.op(), Op::Noop | Op::Stack(..) | Op::Sink(..) | Op::Group(..)))
            => clean_up_sink_like(root, sources),

        root @ Group { sources }
            if sources.iter().any(|source| matches!(source.op(), Op::Noop | Op::Stack(..) | Op::Sink(..) | Op::Group(..)))
            => clean_up_sink_like(root, sources),
    }
}

fn clean_up_sink_like(root: &Arc<UOp>, sources: &[Arc<UOp>]) -> Option<Arc<UOp>> {
    let flattened = sources
        .iter()
        .flat_map(|source| {
            if matches!(source.op(), Op::Noop | Op::Stack(..) | Op::Sink(..) | Op::Group(..)) {
                source.op().sources().into_iter().collect()
            } else {
                vec![source.clone()]
            }
        })
        .collect();
    Some(root.with_sources(flattened))
}

/// Expand Tinygrad's shaped horizontal REDUCE in row-major product order.
fn horizontal_reduce(inp: &Arc<UOp>, reduce_op: ReduceOp, num_axes: usize, target_dtype: &DType) -> Option<Arc<UOp>> {
    let shape = inp.shape().ok().flatten()?;

    let mut indices = vec![Vec::new()];
    for dim in shape.iter().take(num_axes) {
        let size = dim.as_const()?;
        indices = indices
            .into_iter()
            .flat_map(|prefix| {
                (0..size).map(move |index| {
                    let mut next = prefix.clone();
                    next.push(UOp::index_const(index as i64));
                    next
                })
            })
            .collect();
    }

    indices
        .into_iter()
        .map(|indices| {
            UOp::new(Op::Index(ops::Index { buffer: inp.clone(), indices: indices.into() }), target_dtype.clone())
        })
        .reduce(|a, b| apply_reduce_binary(reduce_op, a, b))
}

fn expand_horizontal_reduce(red: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce(ops::Reduce { src, reduce_op, num_axes, .. }) = red.op() else { return None };
    horizontal_reduce(src, *reduce_op, *num_axes, &red.dtype())
}

/// Convert REDUCE to explicit accumulator pattern.
fn reduce_to_acc(red: &Arc<UOp>, ctx: &mut ReduceContext) -> Option<Arc<UOp>> {
    let Op::Reduce(ops::Reduce { src: inp, ranges: reduce_range, reduce_op, num_axes }) = red.op() else { return None };
    let out_dtype = red.dtype();
    let horizontal_inp =
        if *num_axes != 0 { horizontal_reduce(inp, *reduce_op, *num_axes, &out_dtype)? } else { inp.clone() };

    // Find input_ranges: ranges in topo that are not reduce_range and not ended. A STAGE binds
    // its ranges like an END does (its fill loop is closed by `pm_add_local_buffers`); counting
    // one as an input would sink the accumulator's init into every enclosing reduce loop.
    let topo = inp.toposort();
    let mut ended: HashSet<u64> = HashSet::new();
    for node in &topo {
        if matches!(node.op(), Op::End(..) | Op::Stage(..)) {
            ended.extend(node.op().ended_ranges().iter().map(|rng| rng.id));
        }
    }
    let reduce_ids: HashSet<u64> = reduce_range.iter().map(|r| r.id).collect();
    let input_ranges: SmallVec<[Arc<UOp>; 4]> = topo
        .iter()
        .filter(|n| matches!(n.op(), Op::Range(..)) && !reduce_ids.contains(&n.id) && !ended.contains(&n.id))
        .cloned()
        .collect();

    let acc = UOp::placeholder_like(red, ctx.next_reg(), AddrSpace::Reg).ok()?;
    let identity = reduce_identity(*reduce_op, out_dtype.clone());

    let acc_init = acc.after(input_ranges).store(identity);

    let mut loop_deps: SmallVec<[Arc<UOp>; 4]> = smallvec::smallvec![acc_init];
    loop_deps.extend(reduce_range.iter().cloned());
    let acc_loop = acc.after(loop_deps);
    let ret = apply_reduce_binary(*reduce_op, acc_loop, horizontal_inp);

    let store_end = acc.store(ret).end(reduce_range.clone()).with_tag(smallvec::smallvec![TAG_MERGEABLE]);
    ctx.register_end(&store_end);
    Some(acc.after(smallvec::smallvec![store_end]))
}

/// Apply binary reduce operation between two values.
fn apply_reduce_binary(reduce_op: ReduceOp, a: Arc<UOp>, b: Arc<UOp>) -> Arc<UOp> {
    debug_assert!(a.dtype() == b.dtype(), "reduce operand dtype mismatch: lhs={:?}, rhs={:?}", a.dtype(), b.dtype());
    match reduce_op {
        ReduceOp::Add => UOp::alu(BinaryOp::Add, a, b),
        ReduceOp::Mul => UOp::alu(BinaryOp::Mul, a, b),
        ReduceOp::Max => UOp::alu(BinaryOp::Max, a, b),
        // Floats go through MAX so NaN is ignored exactly as in a max reduce;
        // `a < b ? a : b` would keep the last element after a NaN instead.
        ReduceOp::Min if a.dtype().is_float() => UOp::alu(BinaryOp::Max, a.neg(), b.neg()).neg(),
        ReduceOp::Min => {
            let dtype = a.dtype();
            let cond = UOp::new(
                Op::Binary(BinaryOp::Lt, a.clone(), b.clone()),
                DType::Bool.vec(dtype.vcount()).expect("Bool is a scalar"),
            );
            UOp::try_where(cond, a, b).expect("WHERE")
        }
    }
}
