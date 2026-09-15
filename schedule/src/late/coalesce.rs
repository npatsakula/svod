//! Index validity simplification used by late code generation.
//!
//! Direct port of Tinygrad's `codegen/late/coalesce.py:indexing_simplify`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::properties::VminVmaxProperty;
use svod_ir::{BinaryOp, ConstValue, Op, TernaryOp, UOp, UOpKey};

use crate::TypedPatternMatcher;
use svod_ir::ops;

/// Loads/stores of one buffer grouped by constant byte offset.
type OffsetGroups = BTreeMap<i64, Vec<Arc<UOp>>>;
use crate::optimizer::Renderer;
use crate::rewrite::graph_rewrite;
use crate::symbolic::patterns::{sym, symbolic};
use crate::symbolic::valid_simplification::{parse_valid, uop_given_valid};

fn split_add(expr: &Arc<UOp>) -> Vec<Arc<UOp>> {
    match expr.op() {
        Op::Binary(BinaryOp::Add, left, right) => {
            let mut result = split_add(left);
            result.extend(split_add(right));
            result
        }
        _ => vec![expr.clone()],
    }
}

fn split_and(expr: &Arc<UOp>) -> Vec<Arc<UOp>> {
    match expr.op() {
        Op::Binary(BinaryOp::And, left, right) => {
            let mut result = split_and(left);
            result.extend(split_and(right));
            result
        }
        _ => vec![expr.clone()],
    }
}

fn is_irreducible(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Const(..) | Op::Param(..) | Op::Special(..) | Op::Range(..))
}

fn int_bounds(uop: &Arc<UOp>) -> Option<(i64, i64)> {
    let (vmin, vmax) = VminVmaxProperty::get(uop);
    Some((
        match vmin {
            ConstValue::Int(value) => *value,
            _ => return None,
        },
        match vmax {
            ConstValue::Int(value) => *value,
            _ => return None,
        },
    ))
}

fn coordinates(idx: &Arc<UOp>) -> Option<Vec<Arc<UOp>>> {
    match idx.op() {
        Op::Stack(ops::Stack { sources }) if sources.len() == 2 => Some(sources.to_vec()),
        _ => None,
    }
}

fn substitute(idx: &Arc<UOp>, from: &Arc<UOp>, to: Arc<UOp>) -> Arc<UOp> {
    let substitutions: HashMap<UOpKey, Arc<UOp>> = [(UOpKey(from.clone()), to)].into();
    idx.substitute(&substitutions)
}

fn drop_valid_stmts(valid: &Arc<UOp>, idx: &Arc<UOp>, height: usize, width: usize) -> Vec<Arc<UOp>> {
    let mut drop_stmt = Vec::new();
    for (i, stmt) in split_and(valid).into_iter().enumerate() {
        let Some((x, is_upper_bound, c)) = parse_valid(&stmt) else { continue };
        let terms = split_add(&x);

        if !is_upper_bound
            && c == 1
            && terms.iter().all(|u| is_irreducible(u) && int_bounds(u).is_some_and(|(vmin, _)| vmin == 0))
        {
            let mut test_idx = idx.clone();
            for term in &terms {
                test_idx = substitute(&test_idx, term, term.const_like(0));
            }
            test_idx = graph_rewrite(symbolic(), test_idx, &mut ());
            if coordinates(&test_idx).is_some_and(|coords| {
                coords.iter().take(2).any(|coord| int_bounds(coord).is_some_and(|(_, vmax)| vmax < 0))
            }) {
                drop_stmt.push(stmt);
                continue;
            }
        }

        let (x_vmin, x_vmax) = match int_bounds(&x) {
            Some(bounds) => bounds,
            None => continue,
        };
        let (lo, hi) = if is_upper_bound { (c + 1, x_vmax) } else { (x_vmin, c - 1) };
        if lo <= hi {
            let fake = UOp::define_var(format!("fake{i}"), lo, hi);
            let fake = if fake.dtype() == x.dtype() { fake } else { fake.cast(x.dtype()) };
            let mut substitutions = vec![(x.clone(), fake.clone())];
            if let Some(v) = terms.iter().find(|u| is_irreducible(u) && !matches!(u.op(), Op::Const(..))) {
                let rest: Vec<_> = terms.iter().filter(|u| !Arc::ptr_eq(u, v)).cloned().collect();
                if !rest.is_empty() {
                    let rest = rest.into_iter().reduce(|a, b| a.add(&b)).unwrap();
                    substitutions.push((v.clone(), fake.sub(&rest)));
                }
            }

            let bounds = [width as i64, height as i64];
            let out_of_bounds = substitutions.into_iter().any(|(from, to)| {
                let test_idx = graph_rewrite(sym(), substitute(idx, &from, to), &mut ());
                coordinates(&test_idx).is_some_and(|coords| {
                    coords
                        .iter()
                        .zip(bounds)
                        .any(|(coord, bound)| int_bounds(coord).is_some_and(|(vmin, vmax)| vmin >= bound || vmax < 0))
                })
            });
            if out_of_bounds {
                drop_stmt.push(stmt);
            }
        }
    }
    drop_stmt
}

fn simplify_valid_load(buffer: &Arc<UOp>, start_idx: &Arc<UOp>, valid: &Arc<UOp>) -> Option<Arc<UOp>> {
    // Tinygrad `codegen/late/coalesce.py:42` short-circuits on `idx is start_idx`
    // before simplifying, so the rewrite only runs when the gate changed the index.
    let idx = uop_given_valid(valid, start_idx, true);
    if Arc::ptr_eq(&idx, start_idx) || Arc::ptr_eq(&idx, &graph_rewrite(symbolic(), start_idx.clone(), &mut ())) {
        return None;
    }
    UOp::index().buffer(buffer.clone()).indices(vec![idx.valid(valid.clone())]).call().ok()
}

fn simplify_valid_image_load(
    buffer: &Arc<UOp>,
    idx_y: &Arc<UOp>,
    idx_x: &Arc<UOp>,
    valid: &Arc<UOp>,
) -> Option<Arc<UOp>> {
    let shape = image_shape(buffer)?;
    let (idx_x, idx_y) = if idx_x.dtype() != idx_y.dtype() {
        (idx_x.cast(DType::Int32), idx_y.cast(DType::Int32))
    } else {
        (idx_x.clone(), idx_y.clone())
    };
    let start_idx = UOp::stack(smallvec![idx_x, idx_y]);
    let idx = uop_given_valid(valid, &start_idx, true);
    let drop_stmt = drop_valid_stmts(valid, &idx, shape[0], shape[1]);
    if drop_stmt.is_empty() && Arc::ptr_eq(&idx, &start_idx) {
        return None;
    }

    let kept: Vec<_> =
        split_and(valid).into_iter().filter(|stmt| !drop_stmt.iter().any(|drop| Arc::ptr_eq(stmt, drop))).collect();
    let new_valid = kept.into_iter().reduce(|a, b| a.and_(&b));
    let coordinates = coordinates(&idx)?;
    let (idx_x, idx_y) = (coordinates[0].clone(), coordinates[1].clone());
    let indices = if let Some(new_valid) = new_valid {
        vec![idx_y.valid(new_valid.clone()), idx_x.valid(new_valid)]
    } else {
        vec![idx_y, idx_x]
    };
    UOp::index().buffer(buffer.clone()).indices(indices).dtype(DType::Scalar(ScalarDType::Float32)).call().ok()
}

fn invalid_gate(index: &Arc<UOp>) -> Option<(&Arc<UOp>, &Arc<UOp>)> {
    let Op::Ternary(TernaryOp::Where, valid, idx, invalid) = index.op() else { return None };
    UOp::is_invalid_marker(invalid).then_some((valid, idx))
}

/// Tinygrad's `indexing_simplify`, preserving source rule order.
pub fn indexing_simplify() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        Index { buffer, indices }
            if indices.len() == 1 && indices.first().and_then(invalid_gate).is_some()
            => {
                let (valid, idx) = invalid_gate(&indices[0])?;
                simplify_valid_load(buffer, idx, valid)
            },
        Index { buffer, indices }
            if indices.len() == 2
                && invalid_gate(&indices[0]).zip(invalid_gate(&indices[1]))
                    .is_some_and(|((valid_y, _), (valid_x, _))| Arc::ptr_eq(valid_y, valid_x))
            => {
                let (valid, idx_y) = invalid_gate(&indices[0])?;
                let (_, idx_x) = invalid_gate(&indices[1])?;
                simplify_valid_image_load(buffer, idx_y, idx_x, valid)
            },
    }
}

/// Tinygrad's `({}, ren)` context for the `add images` rewrite.
pub type AddImageContext = (HashMap<usize, (usize, usize)>, Renderer);

/// Tinygrad `pm_simplify_add_image`, preserving source rule order.
///
/// Image creation has no mapping here: its supported Tinygrad targets (QCOM,
/// CL, PYTHON, NULL) have no corresponding `RendererDevice` variant. Existing
/// image accesses still require the dtype canonicalizations below.
pub fn pm_simplify_add_image() -> TypedPatternMatcher<AddImageContext> {
    crate::patterns! {
        @context AddImageContext;

        load @ Load { index, alt, gate }
            if index.dtype() == DType::Float32 && load.dtype() == DType::Float16
            => Some(UOp::load()
                .index(index.clone())
                .maybe_alt(alt.clone().map(|value| value.cast(DType::Float32)))
                .maybe_gate(gate.clone())
                .call()
                .cast(DType::Float16)),

        Store { index, value, gate }
            if index.dtype() == DType::Float32 && value.dtype() == DType::Float16
            => Some(UOp::new(
                Op::Store(ops::Store { index: index.clone(), value: value.cast(DType::Float32), gate: gate.clone() }),
                DType::Void,
            )),

        Cast { src: Cast { src, dtype: inner_dtype }, dtype }
            if *inner_dtype == DType::Float16 && *dtype == DType::Float32 && src.dtype() == DType::Float32
            => Some(src.clone()),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MemoryOp {
    Load,
    Store,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum IndexBase {
    UOp(UOpKey),
    Invalid,
    Const,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct MemoryKey {
    op: MemoryOp,
    buffer: UOpKey,
    base: IndexBase,
    valid: UOpKey,
}

fn integer_constant(uop: &Arc<UOp>) -> Option<i64> {
    match uop.op() {
        Op::Const(value) => match value.0 {
            ConstValue::Int(value) => Some(value),
            ConstValue::UInt(value) => i64::try_from(value).ok(),
            _ => None,
        },
        _ => None,
    }
}

fn image_shape(buffer: &Arc<UOp>) -> Option<Vec<usize>> {
    let shape = buffer.shape().ok().flatten()?;
    (shape.len() == 3 && shape[2].as_const() == Some(4))
        .then(|| shape.iter().map(|dim| dim.as_const()).collect::<Option<Vec<_>>>())?
}

fn foldable_buffer(buffer: &Arc<UOp>) -> bool {
    matches!(
        buffer.dtype(),
        DType::Scalar(
            ScalarDType::Float32
                | ScalarDType::Float16
                | ScalarDType::BFloat16
                | ScalarDType::Int32
                | ScalarDType::UInt32
                | ScalarDType::FP8E4M3
                | ScalarDType::FP8E5M2
        )
    ) || image_shape(buffer).is_some()
}

fn env_enabled(name: &str) -> bool {
    std::env::var(name).ok().and_then(|value| value.parse::<i64>().ok()).unwrap_or(0) != 0
}

fn lane_index(value: Arc<UOp>, lane: usize) -> Arc<UOp> {
    UOp::index()
        .buffer(value)
        .indices(vec![UOp::const_(DType::WeakInt, ConstValue::Int(lane as i64))])
        .call()
        .expect("memory coalescing lane index must be valid")
}

fn scalar_index(buffer: Arc<UOp>, index: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(buffer).indices(vec![index]).call().expect("memory coalescing INDEX must be valid")
}

/// Graph-wide late memory coalescing from Tinygrad `coalesce.py`.
///
/// Grouped accesses use Tinygrad's shaped `SHRINK(buf, offset, width)` directly.
/// INDEX/LOAD/STORE retain the scalar memory dtype; SHRINK carries the group shape.
pub fn memory_coalescing(sink: Arc<UOp>, ctx: &Renderer) -> Arc<UOp> {
    if env_enabled("DMC") {
        return sink;
    }

    let mut memory: HashMap<MemoryKey, (Arc<UOp>, OffsetGroups)> = HashMap::new();
    for uop in sink.toposort() {
        // Tinygrad asserts the same shape (`coalesce.py:111`), but a Python
        // assert only guards development; ours would abort a release build over
        // a *missed optimisation*. Coalescing an access is optional, so a gated
        // load/store that still carries its gate as a separate source is simply
        // left alone.
        let (op, index) = match uop.op() {
            Op::Load(ops::Load { index, alt, gate }) if alt.is_none() && gate.is_none() => (MemoryOp::Load, index),
            Op::Store(ops::Store { index, gate, .. }) if gate.is_none() => (MemoryOp::Store, index),
            Op::Load(..) | Op::Store(..) => {
                tracing::warn!("memory coalescing skips a gated load/store");
                continue;
            }
            _ => continue,
        };
        let Op::Index(ops::Index { buffer, indices }) = index.op() else { continue };
        assert_eq!(indices.len(), 1, "memory coalescing requires one flat INDEX");
        if buffer.addrspace() == Some(AddrSpace::Reg) {
            continue;
        }

        let idx_u = &indices[0];
        let idx = idx_u.get_idx();
        let valid = idx_u.get_valid();
        let (base, offset) = match idx.op() {
            Op::Binary(BinaryOp::Add, left, right) if integer_constant(right).is_some() => {
                (IndexBase::UOp(UOpKey(left.clone())), integer_constant(right).unwrap())
            }
            Op::Binary(BinaryOp::Add, left, right) if integer_constant(left).is_some() => {
                (IndexBase::UOp(UOpKey(right.clone())), integer_constant(left).unwrap())
            }
            _ if UOp::is_invalid_marker(&idx) => (IndexBase::Invalid, 0),
            _ if integer_constant(&idx).is_some() => (IndexBase::Const, integer_constant(&idx).unwrap()),
            _ => (IndexBase::UOp(UOpKey(idx.clone())), 0),
        };
        let key = MemoryKey { op, buffer: UOpKey(buffer.clone()), base, valid: UOpKey(valid.clone()) };
        memory
            .entry(key)
            .or_insert_with(|| (buffer.clone(), BTreeMap::new()))
            .1
            .entry(offset)
            .or_default()
            .push(uop.clone());
    }

    let mut replacements: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    for (key, (buffer, offsets)) in memory {
        // Tinygrad's `must_divide=False` DSP arm (`coalesce.py:130`) has no
        // counterpart here: there is no DSP renderer to reach it.
        let mut lengths: Vec<usize> = Vec::new();
        if !foldable_buffer(&buffer) || buffer.addrspace() == Some(AddrSpace::Reg) {
            // Scalar only.
        } else if image_shape(&buffer).is_some() {
            lengths.push(4usize);
        } else if ctx.supports_float4 {
            if buffer.dtype() == DType::Float16 && env_enabled("ALLOW_HALF8") {
                lengths.extend_from_slice(&[8, 4, 2]);
            } else {
                lengths.extend_from_slice(&[4, 2]);
            }
        }
        lengths.push(1);

        let sorted: Vec<i64> = offsets.keys().copied().collect();
        let mut runs: Vec<Vec<i64>> = Vec::new();
        for offset in sorted {
            if runs.last().and_then(|run| run.last()).is_some_and(|last| offset == *last + 1) {
                runs.last_mut().unwrap().push(offset);
            } else {
                runs.push(vec![offset]);
            }
        }

        for run in runs {
            let mut pos = 0;
            while pos < run.len() {
                let first = run[pos];
                let first_access = &offsets[&first][0];
                let first_index = match first_access.op() {
                    Op::Load(ops::Load { index, .. }) | Op::Store(ops::Store { index, .. }) => index,
                    _ => unreachable!(),
                };
                let Op::Index(ops::Index { indices, .. }) = first_index.op() else { unreachable!() };
                let offset = indices[0].get_idx();
                let length = lengths
                    .iter()
                    .copied()
                    .find(|length| *length <= run.len() - pos && offset.divides(*length as i64).is_some())
                    .expect("scalar memory fold must always divide");
                let group = &run[pos..pos + length];
                let valid = key.valid.0.clone();
                let gated = !matches!(valid.op(), Op::Const(value) if value.0 == ConstValue::Bool(true));
                let offset = if gated { offset.valid(valid) } else { offset };
                let index = if group.len() == 1 {
                    scalar_index(buffer.clone(), offset)
                } else {
                    let width = offset.const_like(group.len() as i64);
                    UOp::new(
                        Op::Shrink(ops::Shrink { src: buffer.clone(), offsets: offset, sizes: width }),
                        buffer.dtype(),
                    )
                };

                match key.op {
                    MemoryOp::Store => {
                        let data: Option<Vec<Arc<UOp>>> = group
                            .iter()
                            .map(|lane| match offsets[lane].as_slice() {
                                [store] => match store.op() {
                                    Op::Store(ops::Store { value, .. }) => Some(value.clone()),
                                    _ => None,
                                },
                                _ => None,
                            })
                            .collect();
                        // Tinygrad asserts one store per offset
                        // (`coalesce.py:156`); leave the group un-coalesced
                        // instead of aborting a release build over it.
                        match data {
                            None => tracing::warn!("memory coalescing skips an offset with multiple stores"),
                            Some(mut data) => {
                                let value = if data.len() == 1 {
                                    data.pop().unwrap()
                                } else {
                                    UOp::stack(data.into_iter().collect())
                                };
                                let store = index.store(value);
                                for lane in group {
                                    replacements.insert(UOpKey(offsets[lane][0].clone()), store.clone());
                                }
                            }
                        }
                    }
                    MemoryOp::Load => {
                        let load = UOp::load().index(index).call();
                        for (lane_index_value, lane) in group.iter().enumerate() {
                            let value = if group.len() == 1 {
                                load.clone()
                            } else {
                                lane_index(load.clone(), lane_index_value)
                            };
                            for old_load in &offsets[lane] {
                                replacements.insert(UOpKey(old_load.clone()), value.clone());
                            }
                        }
                    }
                }
                pos += length;
            }
        }
    }

    sink.substitute(&replacements)
}

fn grouped_width(sizes: &Arc<UOp>) -> Option<usize> {
    usize::try_from(integer_constant(sizes)?).ok().filter(|width| *width > 1)
}

fn grouped_lane_index(src: &Arc<UOp>, offsets: &Arc<UOp>, lane: usize) -> Arc<UOp> {
    let offset = if lane == 0 { offsets.clone() } else { offsets.add(&offsets.const_like(lane as i64)) };
    scalar_index(src.clone(), offset)
}

/// Consume coalescing's temporary `SHRINK(buffer, offset, size)` addresses.
///
/// This runs bottom-up after image conversion and devectorizes generic grouped
/// memory into ordinary scalar INDEX operations.
pub fn pm_lower_grouped_shrink() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        Load { index: Shrink { src, offsets, sizes }, alt: None, gate: None }
            if grouped_width(sizes).is_some() => {
                let lanes = (0..grouped_width(sizes)?)
                    .map(|lane| UOp::load().index(grouped_lane_index(src, offsets, lane)).call())
                    .collect();
                Some(UOp::stack(lanes))
            },
        Store { index: Shrink { src, offsets, sizes }, value: Stack { sources }, gate }
            if grouped_width(sizes) == Some(sources.len()) => {
                Some(UOp::group(sources.iter().enumerate().map(|(lane, value)| {
                    UOp::new(
                        Op::Store(ops::Store {
                            index: grouped_lane_index(src, offsets, lane),
                            value: value.clone(),
                            gate: gate.clone(),
                        }),
                        DType::Void,
                    )
                }).collect()))
            },
        Index { buffer: Stack { sources }, indices }
            if indices.len() == 1 && integer_constant(&indices[0]).is_some()
            => {
                let lane = usize::try_from(integer_constant(&indices[0])?).ok()?;
                sources.get(lane).cloned()
            },
    }
}
