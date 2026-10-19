//! Callable scheduling types and execution.
//!
//! This module provides types and functions for managing the execution
//! schedule of tensor operations. After the rangeify pipeline transforms
//! the computation graph into callable operations (`CALL`), we need to:
//!
//! 1. Extract callable operations from the transformed graph
//! 2. Allocate buffers for intermediate results (PARAM/DEFINE_LOCAL)
//! 3. Execute callables in dependency order
//!
//! The scheduling process converts from lazy tensor operations to
//! executable callables with properly allocated device buffers.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use svod_device::Buffer;
use svod_device::device::Device;
use svod_device::registry;
use svod_ir::{CustomFunctionKind, Op, UOp};
use tracing::{debug, trace};

use crate::error::*;
use crate::{ErrorKind, Result};
use snafu::ResultExt;
use svod_ir::ops;

fn canonicalize_callable_source(src: &Arc<UOp>) -> Arc<UOp> {
    let mut cur = src.clone();
    loop {
        match cur.op() {
            Op::After(..) | Op::Buffer(..) | Op::Param(..) | Op::MSelect(..) | Op::MStack(..) | Op::Bind(..) => {
                return cur;
            }
            _ => {
                let sources = cur.op().sources();
                let Some(next) = sources.first() else {
                    return cur;
                };
                if Arc::ptr_eq(&cur, next) {
                    return cur;
                }
                cur = (*next).clone();
            }
        }
    }
}

fn source_primary_buffer_id(src: &Arc<UOp>) -> Option<u64> {
    let src = canonicalize_callable_source(src);
    match src.op() {
        Op::Buffer(..) | Op::Param(..) | Op::After(..) => Some(src.buf_uop().id),
        Op::Bind(..) => None,
        Op::MSelect(ops::MSelect { buffer, device_index }) => {
            if let Op::MStack(ops::MStack { buffers }) = buffer.op() {
                buffers.get(*device_index).map(|b| b.buf_uop().id)
            } else {
                Some(src.buf_uop().id)
            }
        }
        // A bare MSTACK owns multiple buffers. Until schedule items carry that
        // ownership explicitly, resolving it to one shard would be incorrect.
        Op::MStack(..) => None,
        _ => None,
    }
}

/// Depth-first walk over several roots that visits each node once in total:
/// callable sources share their producer chains, so walking them one root at
/// a time would re-traverse the whole upstream graph per schedule item.
pub(crate) struct ReachOnce {
    visited: HashSet<u64>,
    stack: Vec<Arc<UOp>>,
}

impl ReachOnce {
    pub(crate) fn new() -> Self {
        Self { visited: HashSet::new(), stack: Vec::new() }
    }

    pub(crate) fn walk(&mut self, root: &Arc<UOp>, mut visit: impl FnMut(&Arc<UOp>) -> Result<()>) -> Result<()> {
        if self.visited.insert(root.id) {
            self.stack.push(root.clone());
        }
        while let Some(node) = self.stack.pop() {
            visit(&node)?;
            node.op().map_child(|child| {
                if self.visited.insert(child.id) {
                    self.stack.push(child.clone());
                }
            });
        }
        Ok(())
    }
}

fn validate_mselect_bounds(items: &[PreScheduleItem]) -> Result<()> {
    let mut reach = ReachOnce::new();
    for item in items {
        for source in &item.sources {
            reach.walk(source, |node| {
                let Op::MSelect(ops::MSelect { buffer, device_index }) = node.op() else { return Ok(()) };
                let Op::MStack(ops::MStack { buffers }) = buffer.op() else {
                    return Err(ErrorKind::MultiUnsupportedForm {
                        call_id: item.kernel.id,
                        details: format!("MSELECT {} must select from an explicit MSTACK", node.id),
                    }
                    .into());
                };
                if *device_index >= buffers.len() {
                    return Err(ErrorKind::MultiSelectOutOfBounds {
                        source_id: node.id,
                        device_index: *device_index,
                        lane_count: buffers.len(),
                    }
                    .into());
                }
                Ok(())
            })?;
        }
    }
    Ok(())
}

fn device_lane_binding(call_id: u64, ast_nodes: &[Arc<UOp>], lane_count: Option<usize>) -> Result<bool> {
    let mut device_ranges = Vec::new();
    let mut device_params = Vec::new();
    for node in ast_nodes {
        match node.op() {
            Op::Range(ops::Range { axis_type: svod_ir::AxisType::Device, .. }) => device_ranges.push(node),
            Op::Param(ops::Param { arg, .. })
                if arg.addrspace.is_none() && arg.name.as_deref() == Some("_device_num") =>
            {
                device_params.push(node)
            }
            _ => {}
        }
    }

    let marker_count = device_ranges.len() + device_params.len();
    if marker_count == 0 {
        return Ok(false);
    }
    let Some(lane_count) = lane_count else {
        return Err(ErrorKind::MultiUnsupportedForm {
            call_id,
            details: "DEVICE launch binding requires aligned MSTACK arguments".into(),
        }
        .into());
    };
    if marker_count != 1 {
        return Err(ErrorKind::MultiUnsupportedForm {
            call_id,
            details: format!("expected exactly one DEVICE launch marker, found {marker_count}"),
        }
        .into());
    }

    let extent = if let Some(range) = device_ranges.first() {
        let Op::Range(ops::Range { end, .. }) = range.op() else { unreachable!() };
        let (Some(vmin), Some(vmax)) = (end.vmin().try_int(), end.vmax().try_int()) else {
            return Err(ErrorKind::MultiDeviceExtentNotStatic { call_id }.into());
        };
        if vmin != vmax {
            return Err(ErrorKind::MultiDeviceExtentNotStatic { call_id }.into());
        }
        vmin
    } else {
        let Op::Param(ops::Param { arg, .. }) = device_params[0].op() else { unreachable!() };
        let Some((vmin, vmax)) = &arg.vmin_vmax else {
            return Err(ErrorKind::MultiDeviceExtentNotStatic { call_id }.into());
        };
        let (Some(vmin), Some(vmax)) = (vmin.0.try_int(), vmax.0.try_int()) else {
            return Err(ErrorKind::MultiDeviceExtentNotStatic { call_id }.into());
        };
        if vmin != 0 {
            return Err(ErrorKind::MultiUnsupportedForm {
                call_id,
                details: format!("_device_num bounds must start at zero, got {vmin}..={vmax}"),
            }
            .into());
        }
        vmax.checked_add(1).ok_or(ErrorKind::MultiDeviceExtentNotStatic { call_id })?
    };

    if extent != lane_count as i64 {
        return Err(ErrorKind::MultiDeviceExtentMismatch { call_id, expected: lane_count, actual: extent }.into());
    }
    Ok(true)
}

struct MultiSourceExpansion {
    lanes: Vec<Vec<Arc<UOp>>>,
    bind_device_num: bool,
    is_multi: bool,
}

/// `ast_nodes` is `ast.toposort()`, shared with the caller's own body scan.
/// MSELECT bounds were already checked once per pre-schedule.
fn expand_multi_sources(
    call_id: u64,
    ast: &Arc<UOp>,
    ast_nodes: &[Arc<UOp>],
    sources: &[Arc<UOp>],
) -> Result<MultiSourceExpansion> {
    let mut lane_count = None;
    for (source_index, source) in sources.iter().enumerate() {
        let Op::MStack(ops::MStack { buffers }) = source.op() else { continue };
        if buffers.is_empty() {
            return Err(ErrorKind::MultiEmptyLanes { call_id, source_index }.into());
        }
        if let Some(expected) = lane_count {
            if buffers.len() != expected {
                return Err(ErrorKind::MultiLaneCountMismatch {
                    call_id,
                    source_index,
                    expected,
                    actual: buffers.len(),
                }
                .into());
            }
        } else {
            lane_count = Some(buffers.len());
        }
    }

    let bind_device_num = device_lane_binding(call_id, ast_nodes, lane_count)?;
    let Some(lane_count) = lane_count else {
        return Ok(MultiSourceExpansion { lanes: vec![sources.to_vec()], bind_device_num: false, is_multi: false });
    };
    let effect = match ast.op() {
        Op::End(ops::End { computation, .. }) => computation,
        _ => ast,
    };
    if matches!(effect.op(), Op::Slice(..)) {
        return Err(ErrorKind::MultiUnsupportedForm {
            call_id,
            details: "SLICE calls cannot be expanded across lanes".into(),
        }
        .into());
    }

    let mut lanes = vec![Vec::with_capacity(sources.len()); lane_count];
    for (source_index, source) in sources.iter().enumerate() {
        match source.op() {
            Op::MStack(ops::MStack { buffers }) => {
                for (lane, buffer) in buffers.iter().enumerate() {
                    if buffer.toposort().iter().any(|node| matches!(node.op(), Op::Slice(..))) {
                        return Err(ErrorKind::MultiLaneSliceAlias { call_id, source_index, lane }.into());
                    }
                    if matches!(buffer.op(), Op::MStack(..)) {
                        return Err(ErrorKind::MultiUnsupportedForm {
                            call_id,
                            details: format!("MSTACK source {source_index} lane {lane} is a nested MSTACK"),
                        }
                        .into());
                    }
                    lanes[lane].push(buffer.clone());
                }
            }
            Op::Bind(..) => {
                for lane_sources in &mut lanes {
                    lane_sources.push(source.clone());
                }
            }
            other => {
                return Err(ErrorKind::MultiUnsupportedForm {
                    call_id,
                    details: format!(
                        "source {source_index} is {other:?}; every non-scalar CALL argument must be an MSTACK when any argument is multi"
                    ),
                }.into());
            }
        }
    }

    Ok(MultiSourceExpansion { lanes, bind_device_num, is_multi: true })
}

fn collect_callable_dep_ids(dep: &Arc<UOp>, out: &mut HashSet<u64>) -> Result<()> {
    match dep.op() {
        Op::Call(..) => {
            out.insert(dep.id);
            Ok(())
        }
        Op::End(ops::End { computation, .. }) => {
            if matches!(computation.op(), Op::Call(..)) {
                out.insert(computation.id);
                Ok(())
            } else {
                IrConstructionSnafu {
                    details: format!("AFTER dependency END must wrap CALL, got {:?}", computation.op()),
                }
                .fail()
                .map_err(Into::into)
            }
        }
        Op::Store(..) => Ok(()),
        Op::After(ops::After { deps, .. }) => {
            for nested in deps {
                collect_callable_dep_ids(nested, out)?;
            }
            Ok(())
        }
        other => IrConstructionSnafu {
            details: format!("AFTER dependency must be CALL/END(CALL)/STORE/AFTER, got {other:?}"),
        }
        .fail()
        .map_err(Into::into),
    }
}

type AfterDependencySplit = (Vec<Arc<UOp>>, Vec<Arc<UOp>>);

fn split_after_dependencies(after: &Arc<UOp>) -> Result<AfterDependencySplit> {
    let Op::After(ops::After { deps, .. }) = after.op() else {
        return IrConstructionSnafu {
            details: format!("expected AFTER when splitting dependencies, got {:?}", after.op()),
        }
        .fail()
        .map_err(Into::into);
    };

    let mut kernels = Vec::new();
    let mut after_deps = Vec::new();
    for dep in deps {
        match dep.op() {
            Op::Call(..) => kernels.push(dep.clone()),
            Op::End(ops::End { computation, .. }) if matches!(computation.op(), Op::Call(..)) => {
                kernels.push(dep.clone())
            }
            Op::After(..) => after_deps.push(dep.clone()),
            Op::Store(..) => {}
            other => {
                return IrConstructionSnafu {
                    details: format!("AFTER dependency must be CALL/END(CALL)/STORE/AFTER, got {other:?}"),
                }
                .fail()
                .map_err(Into::into);
            }
        }
    }

    Ok((kernels, after_deps))
}

fn collect_source_dependency_callable_ids(src: &Arc<UOp>, out: &mut HashSet<u64>) -> Result<()> {
    let src = canonicalize_callable_source(src);
    match src.op() {
        Op::After(..) => {
            let (kernels, after_deps) = split_after_dependencies(&src)?;
            for kernel in kernels {
                collect_callable_dep_ids(&kernel, out)?;
            }
            for dep in after_deps {
                collect_source_dependency_callable_ids(&dep, out)?;
            }
            Ok(())
        }
        Op::MStack(ops::MStack { buffers }) => {
            for buffer in buffers {
                collect_source_dependency_callable_ids(buffer, out)?;
            }
            Ok(())
        }
        Op::MSelect(ops::MSelect { buffer, .. }) => collect_source_dependency_callable_ids(buffer, out),
        Op::Buffer(..) | Op::Param(..) | Op::Bind(..) => Ok(()),
        other => IrConstructionSnafu {
            details: format!("input to callable must resolve to AFTER/BUFFER/PARAM/MSELECT/MSTACK/BIND, got {other:?}"),
        }
        .fail()
        .map_err(Into::into),
    }
}

fn callable_sources(callable: &Arc<UOp>) -> Option<Vec<Arc<UOp>>> {
    match callable.op() {
        Op::Call(ops::Call { args, .. }) => Some(args.iter().cloned().collect()),
        _ => None,
    }
}

/// Schedule-level RANGEs are uniquely identified by being paired with an
/// `END(Call)` in the transformed graph — the END structurally proves the
/// Range wraps a kernel call. Standalone `Bind(variable, Range)` arguments
/// (user-supplied symbolic variable binds) reference Ranges with no END
/// pairing and must be skipped to avoid being mistaken for loop wrappers.
fn collect_scheduled_range_ids(root: &Arc<UOp>, callable_ids: &HashSet<u64>) -> HashSet<u64> {
    let mut ids = HashSet::new();
    for node in root.toposort_call_aware(false) {
        let Op::End(ops::End { computation, ranges }) = node.op() else { continue };
        if !matches!(computation.op(), Op::Call(..)) || !callable_ids.contains(&computation.id) {
            continue;
        }
        for r in ranges {
            if matches!(r.op(), Op::Range(..)) {
                ids.insert(r.id);
            }
        }
    }
    ids
}

fn collect_call_bound_ranges(callable: &Arc<UOp>, scheduled_range_ids: &HashSet<u64>) -> Result<Vec<BoundRangeRef>> {
    let Op::Call(ops::Call { args, .. }) = callable.op() else {
        return ExpectedCallableOpSnafu.fail().map_err(Into::into);
    };

    let mut bound_ranges = Vec::new();
    for arg in args {
        let Op::Bind(ops::Bind { var, value }) = arg.op() else {
            continue;
        };
        let Op::Range(..) = value.op() else {
            // User variable binds (`BIND(variable, CONST)`) are not schedule loops.
            continue;
        };
        let name = match var.op() {
            Op::DefineVar(ops::DefineVar { name, .. }) => name,
            Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none() => arg.name.as_ref().ok_or_else(|| {
                ErrorKind::IrConstruction { details: "CALL BIND scalar PARAM source must have a name".to_string() }
            })?,
            _ => {
                return IrConstructionSnafu {
                    details: format!("CALL BIND Range source must wrap a variable, got {:?}", var.op()),
                }
                .fail()
                .map_err(Into::into);
            }
        };
        // Only Ranges paired with an `END(Call)` are schedule-level wrappers;
        // standalone Range-valued binds carry runtime values, not loop counters.
        if !scheduled_range_ids.contains(&value.id) {
            continue;
        }
        bound_ranges.push(BoundRangeRef { var_name: name.clone(), range_uop: value.clone() });
    }
    Ok(bound_ranges)
}

fn collect_linear_sched_ops_internal(
    root: &Arc<UOp>,
    callable_ids: &HashSet<u64>,
    scheduled_range_ids: &HashSet<u64>,
) -> Result<Vec<LinearSchedOp>> {
    let mut linear_ops = Vec::new();

    for node in root.toposort_call_aware(false) {
        match node.op() {
            Op::Range(..) if scheduled_range_ids.contains(&node.id) => {
                linear_ops.push(LinearSchedOp::Range { range: node.clone() });
            }
            Op::Call(..) if callable_ids.contains(&node.id) => {
                linear_ops.push(LinearSchedOp::Call { kernel_id: node.id });
            }
            Op::End(ops::End { computation, ranges }) if matches!(computation.op(), Op::Call(..)) => {
                if !callable_ids.contains(&computation.id) {
                    continue;
                }
                let wrapper_ranges: Vec<Arc<UOp>> =
                    ranges.iter().filter(|r| matches!(r.op(), Op::Range(..))).cloned().collect();
                match wrapper_ranges.as_slice() {
                    [] => {}
                    [outer] => linear_ops.push(LinearSchedOp::End { range: outer.clone(), kernel_id: computation.id }),
                    _ => {
                        return IrConstructionSnafu {
                            details: format!(
                                "END(CALL) must close at most one wrapper range in strict scheduler, got {}",
                                wrapper_ranges.len()
                            ),
                        }
                        .fail()
                        .map_err(Into::into);
                    }
                }
            }
            _ => {}
        }
    }

    if linear_ops.is_empty() {
        return IrConstructionSnafu { details: "strict scheduler produced empty linear control stream".to_string() }
            .fail()
            .map_err(Into::into);
    }
    Ok(linear_ops)
}

/// Unroll the schedule control stream into a flat list of kernel invocations.
///
/// Loop membership is structural, not positional: a kernel belongs to the loop
/// whose RANGE its CALL binds, and is replayed once per trip with the counter
/// in `fixedvars`. Every other kernel runs once — where the topological order
/// put it, unless it reads a loop's output, in which case it waits for that
/// loop's END. A kernel that binds a range outside the range's RANGE…END span,
/// or that a loop body depends on while reading the loop's own output, is a
/// construction error. Outer ranges must have concrete `vmin`/`vmax`
/// (validated by `schedule_range_bounds`) — there is no symbolic-iteration
/// support today.
fn collect_kernel_invocations(
    root: &Arc<UOp>,
    items: &[PreScheduleItem],
    scheduled_range_ids: &HashSet<u64>,
) -> Result<Vec<KernelInvocation>> {
    let callable_ids: HashSet<u64> = items.iter().map(|it| it.kernel.id).collect();
    let linear_ops = collect_linear_sched_ops_internal(root, &callable_ids, scheduled_range_ids)?;
    let by_kernel: HashMap<u64, &PreScheduleItem> = items.iter().map(|it| (it.kernel.id, it)).collect();
    let construction = |details: String| -> Error { ErrorKind::IrConstruction { details }.into() };

    let mut invocations = Vec::new();
    // Transitive callable ancestry, built in linear (topological) order.
    let mut ancestors: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut open: Vec<u64> = Vec::new();
    let mut members: HashMap<u64, Vec<u64>> = HashMap::new();
    // Kernels reading a loop's output, with the open loops they wait for.
    let mut deferred: Vec<(u64, HashSet<u64>)> = Vec::new();

    for op in &linear_ops {
        match op {
            LinearSchedOp::Range { range } => {
                schedule_range_bounds(range)?;
                if open.contains(&range.id) {
                    return Err(construction(format!("schedule range {} is declared twice", range.id)));
                }
                open.push(range.id);
            }
            LinearSchedOp::Call { kernel_id } => {
                let item = by_kernel
                    .get(kernel_id)
                    .ok_or_else(|| construction(format!("linear CALL references unknown kernel id {kernel_id}")))?;
                let mut reach: HashSet<u64> = item.dependencies.iter().copied().collect();
                for dep in &item.dependencies {
                    reach.extend(ancestors.get(dep).into_iter().flatten().copied());
                }
                match item.bound_ranges.as_slice() {
                    [] => {
                        let waits: HashSet<u64> = open
                            .iter()
                            .copied()
                            .filter(|range| members.get(range).is_some_and(|m| m.iter().any(|k| reach.contains(k))))
                            .collect();
                        if waits.is_empty() {
                            invocations.push(KernelInvocation { kernel_id: *kernel_id, fixedvars: HashMap::new() });
                        } else {
                            deferred.push((*kernel_id, waits));
                        }
                    }
                    [bound] => {
                        if !open.contains(&bound.range_uop.id) {
                            return Err(construction(format!(
                                "CALL {} binds variable '{}' to schedule range {} outside the range's RANGE…END span",
                                kernel_id, bound.var_name, bound.range_uop.id
                            )));
                        }
                        members.entry(bound.range_uop.id).or_default().push(*kernel_id);
                    }
                    _ => {
                        return Err(construction(format!(
                            "CALL {kernel_id} binds {} schedule ranges; nested schedule loops are not supported",
                            item.bound_ranges.len()
                        )));
                    }
                }
                ancestors.insert(*kernel_id, reach);
            }
            LinearSchedOp::End { range, kernel_id } => {
                if !by_kernel.contains_key(kernel_id) {
                    return Err(construction(format!("linear END references unknown CALL id {kernel_id}")));
                }
                let Some(position) = open.iter().position(|id| *id == range.id) else {
                    return Err(construction(format!("END references schedule range {} that is not active", range.id)));
                };
                let body = members.remove(&range.id).unwrap_or_default();
                for member in &body {
                    let reach = &ancestors[member];
                    if let Some((blocker, _)) = deferred.iter().find(|(k, _)| reach.contains(k)) {
                        return Err(construction(format!(
                            "kernel {member} in schedule loop {} depends on kernel {blocker}, which reads the loop's output",
                            range.id
                        )));
                    }
                    let interleaved = open
                        .iter()
                        .filter(|other| **other != range.id)
                        .any(|other| members.get(other).is_some_and(|m| m.iter().any(|k| reach.contains(k))));
                    if interleaved {
                        return Err(construction(format!(
                            "kernel {member} in schedule loop {} depends on the body of a loop that is still open",
                            range.id
                        )));
                    }
                }
                let (vmin, vmax) = schedule_range_bounds(range)?;
                for trip in vmin..=vmax {
                    for member in &body {
                        let bound = &by_kernel[member].bound_ranges[0];
                        let fixedvars = HashMap::from([(bound.var_name.clone(), trip)]);
                        invocations.push(KernelInvocation { kernel_id: *member, fixedvars });
                    }
                }
                open.remove(position);
                deferred.retain_mut(|(kernel_id, waits)| {
                    waits.remove(&range.id);
                    if !waits.is_empty() {
                        return true;
                    }
                    invocations.push(KernelInvocation { kernel_id: *kernel_id, fixedvars: HashMap::new() });
                    false
                });
            }
        }
    }

    if let Some(range) = open.first() {
        return Err(construction(format!("schedule range {range} is missing END in strict scheduler")));
    }
    Ok(invocations)
}

fn analyze_callable_dependencies(callables: &[Arc<UOp>], root: &Arc<UOp>) -> Result<Vec<HashSet<usize>>> {
    // Build callable ID → index mapping
    let callable_idx: HashMap<u64, usize> = callables.iter().enumerate().map(|(i, c)| (c.id, i)).collect();
    // Build dependency edges from source-driven dependency extraction
    // (avoids global writer-union heuristics).
    let mut dependencies: Vec<HashSet<usize>> = vec![HashSet::new(); callables.len()];

    for (consumer_idx, callable) in callables.iter().enumerate() {
        let mut dep_ids = HashSet::new();
        if let Some(sources) = callable_sources(callable) {
            for src in sources {
                collect_source_dependency_callable_ids(&src, &mut dep_ids)?;
            }
        }

        for dep_id in dep_ids {
            let Some(&producer_idx) = callable_idx.get(&dep_id) else {
                return IrConstructionSnafu {
                    details: format!("callable dependency references unknown callable id {dep_id}"),
                }
                .fail()
                .map_err(Into::into);
            };
            if producer_idx != consumer_idx {
                dependencies[consumer_idx].insert(producer_idx);
            }
        }
    }

    // Preserve ordering-only dependencies encoded through AFTER surfaces that
    // may include void/custom callables with no direct buffer edge.
    //
    // `toposort_call_aware(false)` stops at CALL boundaries: only inter-kernel
    // AFTER surfaces (the scheduling DAG between callables) are walked, never the
    // intra-kernel AFTERs *inside* a CALL body. A hand-lowered kernel body (a
    // marked SINK, e.g. a hand-ranged custom kernel or rangeify's own `reduce_to_acc`)
    // legitimately contains `After([END(STORE)])` / `After([RANGE])` REG/LDS
    // accumulator edges; those are the linearizer's concern, not callable
    // dependencies, and must not be validated by this grammar.
    for node in root.toposort_call_aware(false) {
        let Op::After(..) = node.op() else {
            continue;
        };

        let (kernels, after_deps) = split_after_dependencies(&node)?;
        for kernel in kernels {
            let callable = match kernel.op() {
                Op::Call(..) => kernel.clone(),
                Op::End(ops::End { computation, .. }) => computation.clone(),
                _ => unreachable!("split_after_dependencies only returns CALL/END(CALL) kernels"),
            };

            let Some(&consumer_idx) = callable_idx.get(&callable.id) else {
                return IrConstructionSnafu {
                    details: format!("AFTER dependency references unknown callable id {}", callable.id),
                }
                .fail()
                .map_err(Into::into);
            };

            let mut dep_ids = HashSet::new();
            for dep in &after_deps {
                collect_source_dependency_callable_ids(dep, &mut dep_ids)?;
            }

            for dep_id in dep_ids {
                let Some(&producer_idx) = callable_idx.get(&dep_id) else {
                    return IrConstructionSnafu {
                        details: format!("callable dependency references unknown callable id {dep_id}"),
                    }
                    .fail()
                    .map_err(Into::into);
                };
                if producer_idx != consumer_idx {
                    dependencies[consumer_idx].insert(producer_idx);
                }
            }
        }
    }

    Ok(dependencies)
}

/// Input buffers collected before schedule creation.
///
/// Maps BUFFER UOp ID → Buffer for input tensors.
/// This keeps schedule instantiation explicit and avoids global lookups.
pub type InputBuffers = HashMap<u64, Buffer>;

/// Schedule-level Range bound to a DEFINE_VAR via a CALL argument's
/// `Bind(DefineVar, Range)`. Each invocation of the wrapped CALL substitutes
/// the loop counter value into the kernel's variable.
#[derive(Clone, Debug)]
pub struct BoundRangeRef {
    /// Variable name (e.g., "range_0")
    pub var_name: String,
    /// RANGE UOp for this bound variable.
    pub range_uop: Arc<UOp>,
}

/// Linearized scheduling control op (internal to schedule construction).
///
/// The strict scheduler walks these as a small bytecode (Range opens a loop,
/// Call joins it or runs once, End replays the members per trip) — see
/// `collect_kernel_invocations`. Eager unrolling at pre-schedule time turns
/// them into a flat `Vec<KernelInvocation>`.
#[derive(Clone, Debug)]
enum LinearSchedOp {
    Range { range: Arc<UOp> },
    Call { kernel_id: u64 },
    End { range: Arc<UOp>, kernel_id: u64 },
}

/// One concrete kernel invocation: a kernel id + its loop-resolved variable bindings.
///
/// The schedule is a flat list of these (eagerly unrolled at
/// `create_pre_schedule` time); each element is an atomic kernel CALL with
/// concrete bindings.
#[derive(Clone, Debug)]
pub struct KernelInvocation {
    /// Kernel ID — looked up against the `PreScheduleItem.kernel.id` index.
    pub kernel_id: u64,
    /// Concrete `var_name → value` bindings produced by the surrounding loop
    /// counters at the moment of this invocation.
    pub fixedvars: HashMap<String, i64>,
}

/// A single executable callable with its buffers and variable bindings.
///
/// Each ScheduleItem represents one callable that needs to be compiled
/// and executed. The callable AST contains STORE operations that write
/// results to buffers.
///
/// Schedule items are fully expanded during schedule instantiation.
#[derive(Clone)]
pub struct ScheduleItem {
    /// The callable wrapper UOp (`CALL`) used for dependency identity.
    pub kernel: Arc<UOp>,

    /// The inner callable AST (typically SINK containing STORE ops) - for codegen
    pub ast: Arc<UOp>,

    /// Device buffers for this callable (in order expected by codegen)
    pub buffers: Vec<Buffer>,

    /// UOp IDs under which each buffer was registered in buffer index.
    /// Same length as `buffers`. Used for cleanup - to remove buffers from
    /// the global registry, we need to know what key they were registered under.
    pub buffer_uop_ids: Vec<u64>,

    /// Fixed variable values for this specific kernel invocation.
    /// Maps variable name (e.g., "range_0") to concrete i64 value.
    /// Always concrete in the strict scheduler path.
    pub fixedvars: HashMap<String, i64>,

    /// Names of variables in `fixedvars` whose values came from schedule-loop
    /// counters. User `var_vals` must not override these — see
    /// `collect_non_overridable_fixedvars`.
    pub loop_var_names: HashSet<String>,

    /// Callable UOp IDs that must complete before this item can execute.
    /// Empty for callables without dependencies (first in chain or independent).
    /// Dependencies are implicit in scheduling order after topological sort.
    pub dependencies: Vec<u64>,

    /// Concrete schedule-item indices that must complete before this item.
    /// Used for ordering constraints that cannot be represented by callable ID
    /// after strict unrolling creates repeated callable IDs.
    pub instance_dependencies: Vec<usize>,
}

/// Kernel attribution, read back from the CALL that `split_store` stamped.
///
/// The CALL is the per-dispatch node — the body it wraps is shared by every
/// structurally identical kernel — so the origins live there and are recovered
/// here rather than copied into every schedule type.
fn call_origins(kernel: &Arc<UOp>) -> Option<&svod_ir::CallInfo> {
    match kernel.op() {
        Op::Call(ops::Call { info, .. }) => Some(info),
        _ => None,
    }
}

static NO_ORIGINS: std::sync::LazyLock<svod_ir::OriginSet> = std::sync::LazyLock::new(svod_ir::OriginSet::new);

impl ScheduleItem {
    /// Origin this kernel is charged to.
    pub fn origin(&self) -> Option<svod_ir::OriginId> {
        call_origins(&self.kernel).and_then(|info| info.origin)
    }

    /// Every origin folded into this kernel.
    pub fn origins(&self) -> &svod_ir::OriginSet {
        call_origins(&self.kernel).map_or(&NO_ORIGINS, |info| &info.origins)
    }
}

impl PreScheduleItem {
    /// Origin this kernel is charged to.
    pub fn origin(&self) -> Option<svod_ir::OriginId> {
        call_origins(&self.kernel).and_then(|info| info.origin)
    }

    /// Every origin folded into this kernel.
    pub fn origins(&self) -> &svod_ir::OriginSet {
        call_origins(&self.kernel).map_or(&NO_ORIGINS, |info| &info.origins)
    }
}

/// Full execution schedule (list of callables in dependency order).
pub type Schedule = Vec<ScheduleItem>;

/// Cached pre-schedule item.
///
/// Contains callable identity/AST and argument UOps, but no concrete buffers.
#[derive(Clone)]
pub struct PreScheduleItem {
    /// Callable wrapper UOp (`CALL`) used for dependency identity.
    pub kernel: Arc<UOp>,
    /// Callable body AST used for codegen.
    pub ast: Arc<UOp>,
    /// Callable argument UOps in canonical order.
    pub sources: Vec<Arc<UOp>>,
    /// Callable dependencies by callable UOp ID.
    pub dependencies: Vec<u64>,
    /// Schedule-level Range bindings (`BIND(DEFINE_VAR, RANGE)`) from CALL args.
    pub bound_ranges: Vec<BoundRangeRef>,
}

/// Cached pre-schedule artifact.
///
/// A flat list of kernel invocations with their concrete bindings. Outer
/// loops are eagerly unrolled at construction time, so there is no
/// schedule-level Range/End bytecode — just one entry per kernel call.
#[derive(Clone)]
pub struct PreSchedule {
    /// Per-kernel descriptor pool indexed by `kernel.id` from `KernelInvocation`.
    pub items: Vec<PreScheduleItem>,
    /// Flat sequence of kernel invocations after eager loop unrolling.
    pub invocations: Vec<KernelInvocation>,
    /// Output buffers in sink source order.
    pub output_buffer_uops: Vec<Arc<UOp>>,
}

type SortedCallables = (Vec<Arc<UOp>>, HashMap<u64, Vec<u64>>);

/// Result of schedule creation, including output buffer identification.
pub struct ScheduleResult {
    /// The schedule items in dependency order.
    pub items: Schedule,
    /// UOp IDs of output buffers, in SINK source order.
    /// Extracted directly from the SINK's sources via `buf_uop()`.
    /// For single-tensor realize, contains one ID.
    pub output_uop_ids: Vec<u64>,
    /// Concrete handles for alias-only outputs that have no executable item.
    pub alias_output_buffers: HashMap<u64, Buffer>,
}

/// Buffers collected for a single callable.
#[derive(Clone)]
struct CallableBuffers {
    /// Device buffers in codegen order.
    buffers: Vec<Buffer>,
    /// UOp IDs for each buffer.
    uop_ids: Vec<u64>,
}

/// Sort callables by dependencies (producers before consumers).
///
/// Uses Kahn's algorithm for topological sort based on buffer dependencies
/// derived from the graph structure (AFTER nodes and callable sources).
/// This ensures producer callables are processed before consumers, which is
/// critical for buffer sharing: the producer allocates the buffer first,
/// then the consumer finds it in the registry via `get_or_create_buffer`.
fn sort_callables_by_dependencies(callables: &[Arc<UOp>], root: &Arc<UOp>) -> Result<SortedCallables> {
    debug!(num_callables = callables.len(), "sorting callables by dependencies");

    let dependencies = analyze_callable_dependencies(callables, root)?;

    // Kahn's algorithm for topological sort
    let mut in_degree: Vec<usize> = dependencies.iter().map(|deps| deps.len()).collect();
    let mut dependents: Vec<Vec<usize>> = vec![vec![]; callables.len()];

    for (consumer, deps) in dependencies.iter().enumerate() {
        for &producer in deps {
            dependents[producer].push(consumer);
        }
    }

    let mut queue: VecDeque<usize> =
        in_degree.iter().enumerate().filter(|&(_, &deg)| deg == 0).map(|(idx, _)| idx).collect();

    let mut sorted_indices = Vec::new();
    while let Some(idx) = queue.pop_front() {
        sorted_indices.push(idx);
        for &dependent in &dependents[idx] {
            in_degree[dependent] -= 1;
            if in_degree[dependent] == 0 {
                queue.push_back(dependent);
            }
        }
    }

    if sorted_indices.len() < callables.len() {
        return DependencyCyclesSnafu.fail().map_err(Into::into);
    }

    let sorted: Vec<Arc<UOp>> = sorted_indices.iter().map(|&idx| callables[idx].clone()).collect();

    let dependency_ids_by_callable: HashMap<u64, Vec<u64>> = callables
        .iter()
        .enumerate()
        .map(|(idx, callable)| {
            let mut deps: Vec<u64> = dependencies[idx].iter().map(|&dep_idx| callables[dep_idx].id).collect();
            deps.sort_unstable();
            (callable.id, deps)
        })
        .collect();

    debug!(num_sorted = sorted.len(), "callables sorted");

    Ok((sorted, dependency_ids_by_callable))
}

/// Extract callables from transformed graph and create pre-schedule artifact.
///
/// This function walks the transformed UOp graph (after rangeify and
/// kernel splitting) and extracts all callable wrappers. For each callable,
/// it records callable identity, dependencies, and strict control-flow ops.
///
/// # Arguments
///
/// * `transformed` - The UOp graph after rangeify + kernel splitting
///
/// # Returns
///
/// A pre-schedule artifact ready for per-run instantiation.
///
/// # Errors
///
/// Returns error if:
/// - No callables found after scheduling pipeline
/// - Callable dependency graph contains cycles
pub fn create_pre_schedule(transformed: Arc<UOp>) -> Result<PreSchedule> {
    // Step 1: Find all callable wrappers (CALL) without descending into CALL bodies.
    let mut callables = Vec::new();
    for node in transformed.toposort_call_aware(false) {
        if matches!(node.op(), Op::Call(..)) {
            callables.push(node);
        }
    }

    if callables.is_empty() {
        return NoKernelsFoundSnafu.fail().map_err(Into::into);
    }

    // Step 1.5: Sort callables by dependencies (producers before consumers)
    let (callables, dependency_ids_by_callable) = sort_callables_by_dependencies(&callables, &transformed)?;

    // Step 1.75: Compute the set of Range UOp IDs paired with `END(Call)` —
    // these are the schedule-level loop wrappers, identified structurally
    // (no axis_type filter).
    let callable_ids: HashSet<u64> = callables.iter().map(|c| c.id).collect();
    let scheduled_range_ids = collect_scheduled_range_ids(&transformed, &callable_ids);

    // Step 2: Build pre-schedule items (AST + sources + dependencies + bound ranges).
    let mut items = Vec::with_capacity(callables.len());
    for callable_uop in callables {
        let Op::Call(ops::Call { body, args, .. }) = callable_uop.op() else {
            unreachable!("filtered to only call wrappers above")
        };
        let dependencies = dependency_ids_by_callable.get(&callable_uop.id).cloned().unwrap_or_default();
        let bound_ranges = collect_call_bound_ranges(&callable_uop, &scheduled_range_ids)?;
        items.push(PreScheduleItem {
            kernel: callable_uop.clone(),
            ast: body.clone(),
            sources: args.iter().cloned().collect(),
            dependencies,
            bound_ranges,
        });
    }

    validate_mselect_bounds(&items)?;

    // Step 3: Eagerly unroll outer loops into a flat list of kernel
    // invocations — each invocation carries the concrete `fixedvars` produced
    // by its enclosing loop counters.
    let invocations = collect_kernel_invocations(&transformed, &items, &scheduled_range_ids)?;

    // Output buffers in SINK source order.
    let output_buffer_uops: Vec<Arc<UOp>> = match transformed.op() {
        Op::Sink(ops::Sink { sources, .. }) => sources.iter().map(|src| src.buf_uop()).collect(),
        _ => vec![transformed.buf_uop()],
    };

    Ok(PreSchedule { items, invocations, output_buffer_uops })
}

/// Instantiate a concrete execution schedule from a pre-schedule artifact.
///
/// This is the per-run phase that attaches concrete buffers and runtime
/// variable bindings to cached callable descriptors.
pub fn instantiate_schedule(
    pre_schedule: &PreSchedule,
    input_buffers: &InputBuffers,
    var_vals: &HashMap<String, i64>,
    device_local_outputs: bool,
) -> Result<ScheduleResult> {
    // Track allocated intermediate buffers locally (no global registry needed)
    let mut allocated_buffers: HashMap<u64, Buffer> = HashMap::new();

    // Final-output buffer ids: these stay host-visible so the host can read the
    // result directly (`array_view`). Every other freshly allocated buffer is an
    // internal intermediate and can live device-local on backends that
    // distinguish it (AMD). `device_local_outputs` opts the outputs into the
    // device-local class too — the host then reads them via `copyout` (SDMA
    // staging) instead of the uncached host mapping. See
    // `collect_callable_buffers`.
    let output_ids: HashSet<u64> = if device_local_outputs {
        HashSet::new()
    } else {
        pre_schedule.output_buffer_uops.iter().map(|u| u.buf_uop().id).collect()
    };

    let mut templates: HashMap<u64, ScheduleItemTemplate> = HashMap::with_capacity(pre_schedule.items.len());
    let mut alias_dependencies: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut alias_output_ids = HashSet::new();
    for item in &pre_schedule.items {
        let nodes = item.ast.toposort();

        let expansion = expand_multi_sources(item.kernel.id, &item.ast, &nodes, &item.sources)?;
        let mut lane_buffers = Vec::with_capacity(expansion.lanes.len());
        for sources in &expansion.lanes {
            lane_buffers.push(collect_callable_buffers(sources, input_buffers, &mut allocated_buffers, &output_ids)?);
        }

        let effect = match item.ast.op() {
            Op::End(ops::End { computation, .. }) => computation,
            _ => &item.ast,
        };
        let cross_device_effect = matches!(effect.op(), Op::Copy(..))
            || matches!(
                effect.op(),
                Op::CustomFunction(ops::CustomFunction { kind: CustomFunctionKind::AllReduce { .. }, .. })
            );
        if expansion.is_multi && !cross_device_effect {
            for (lane, kb) in lane_buffers.iter().enumerate() {
                let mut expected: Option<svod_dtype::DeviceSpec> = None;
                for buffer in &kb.buffers {
                    let actual = buffer.allocator().device_spec();
                    if actual.is_disk() {
                        continue;
                    }
                    if let Some(expected) = &expected
                        && expected != &actual
                    {
                        return Err(ErrorKind::MultiLaneDeviceMismatch {
                            call_id: item.kernel.id,
                            lane,
                            expected: format!("{expected:?}"),
                            actual: format!("{actual:?}"),
                        }
                        .into());
                    }
                    expected = Some(actual);
                }
            }
        }
        if let Op::Slice(ops::Slice { offset, size, .. }) = effect.op() {
            let kb = lane_buffers.first_mut().expect("single-lane expansion always has one buffer set");
            let offset = match offset.op() {
                Op::Const(value) => value.0.try_int().and_then(|value| usize::try_from(value).ok()),
                _ => None,
            }
            .ok_or_else(|| ErrorKind::IrConstruction {
                details: "SLICE offset must be a non-negative constant".into(),
            })?;
            if kb.buffers.len() < 2 || kb.uop_ids.len() < 2 {
                return IrConstructionSnafu {
                    details: format!("SLICE call {} requires output and base buffers", item.kernel.id),
                }
                .fail()
                .map_err(Into::into);
            }
            let base = &kb.buffers[1];
            let view = base.view(offset * base.dtype().bytes(), size * effect.dtype().bytes()).context(DeviceSnafu)?;
            allocated_buffers.insert(kb.uop_ids[0], view.clone());
            alias_output_ids.insert(kb.uop_ids[0]);
            kb.buffers[0] = view;
            alias_dependencies.insert(item.kernel.id, item.dependencies.clone());
            continue;
        }

        debug!(callable.id = item.kernel.id, num_sources = item.sources.len(), "Schedule item created");

        // Populate fixedvars with only the user Variables referenced by this kernel's AST.
        let fixedvars: HashMap<String, i64> = if var_vals.is_empty() {
            HashMap::new()
        } else {
            let ast_var_names: HashSet<&str> = nodes
                .iter()
                .filter_map(|n| match n.op() {
                    Op::DefineVar(ops::DefineVar { name, .. }) => Some(name.as_str()),
                    Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none() => arg.name.as_deref(),
                    _ => None,
                })
                .collect();
            var_vals
                .iter()
                .filter(|(name, _)| ast_var_names.contains(name.as_str()))
                .map(|(k, v)| (k.clone(), *v))
                .collect()
        };

        templates.insert(
            item.kernel.id,
            ScheduleItemTemplate {
                kernel: item.kernel.clone(),
                ast: item.ast.clone(),
                lane_buffers,
                dependencies: item.dependencies.clone(),
                base_fixedvars: fixedvars,
                bind_device_num: expansion.bind_device_num,
            },
        );
    }

    fn executable_dependencies(dependencies: &[u64], aliases: &HashMap<u64, Vec<u64>>, out: &mut HashSet<u64>) {
        for dependency in dependencies {
            if let Some(nested) = aliases.get(dependency) {
                executable_dependencies(nested, aliases, out);
            } else {
                out.insert(*dependency);
            }
        }
    }

    // A kernel's loop, as `(range id, counter name)`: the counter's value in an
    // invocation's `fixedvars` is the trip it belongs to.
    let loop_of_kernel: HashMap<u64, (u64, &str)> = pre_schedule
        .items
        .iter()
        .filter_map(|item| match item.bound_ranges.as_slice() {
            [bound] => Some((item.kernel.id, (bound.range_uop.id, bound.var_name.as_str()))),
            _ => None,
        })
        .collect();
    let mut trip_instances: HashMap<(u64, i64), Vec<usize>> = HashMap::new();

    let mut schedule = Vec::with_capacity(pre_schedule.invocations.len());
    let mut expanded_instances_by_kernel: HashMap<u64, Vec<usize>> = HashMap::new();
    for invocation in &pre_schedule.invocations {
        if alias_dependencies.contains_key(&invocation.kernel_id) {
            continue;
        }
        let Some(template) = templates.get(&invocation.kernel_id) else {
            return IrConstructionSnafu {
                details: format!("invocation references unknown kernel id {}", invocation.kernel_id),
            }
            .fail()
            .map_err(Into::into);
        };

        let mut dependencies = HashSet::new();
        executable_dependencies(&template.dependencies, &alias_dependencies, &mut dependencies);
        let mut dependencies: Vec<u64> = dependencies.into_iter().collect();
        dependencies.sort_unstable();

        let mut instance_dependencies = HashSet::new();
        for dependency in &dependencies {
            if let Some(instances) = expanded_instances_by_kernel.get(dependency) {
                instance_dependencies.extend(instances.iter().copied());
            }
        }
        // A schedule loop is a recurrence: trip `t` reads what trip `t - 1`
        // wrote, at a symbolic offset into the same buffer, so the callable-id
        // dependency graph cannot see the edge (all trips share one kernel id,
        // and self-edges are suppressed). Every item of a trip depends on every
        // item of the previous trip, which levelling would otherwise hoist a
        // dependency-free body kernel's trips into one wave.
        let trip = loop_of_kernel
            .get(&invocation.kernel_id)
            .and_then(|(range, var)| Some((*range, *invocation.fixedvars.get(*var)?)));
        if let Some((range, trip)) = trip
            && let Some(previous) = trip_instances.get(&(range, trip - 1))
        {
            instance_dependencies.extend(previous.iter().copied());
        }
        let mut instance_dependencies: Vec<usize> = instance_dependencies.into_iter().collect();
        instance_dependencies.sort_unstable();

        let expanded = template.lane_buffers.len() > 1;
        let first_schedule_index = schedule.len();
        for (lane, kb) in template.lane_buffers.iter().enumerate() {
            let mut fixedvars = template.base_fixedvars.clone();
            fixedvars.extend(invocation.fixedvars.iter().map(|(name, value)| (name.clone(), *value)));
            if template.bind_device_num {
                insert_fixedvar_checked(&mut fixedvars, template.kernel.id, "_device_num", lane as i64)?;
            }
            let loop_var_names: HashSet<String> = invocation.fixedvars.keys().cloned().collect();

            schedule.push(ScheduleItem {
                kernel: template.kernel.clone(),
                ast: template.ast.clone(),
                buffers: kb.buffers.clone(),
                buffer_uop_ids: kb.uop_ids.clone(),
                fixedvars,
                loop_var_names,
                dependencies: dependencies.clone(),
                instance_dependencies: instance_dependencies.clone(),
            });
        }
        if expanded {
            expanded_instances_by_kernel
                .entry(invocation.kernel_id)
                .or_default()
                .extend(first_schedule_index..schedule.len());
        }
        if let Some(slot) = trip {
            trip_instances.entry(slot).or_default().extend(first_schedule_index..schedule.len());
        }
    }

    let output_uop_ids: Vec<u64> = pre_schedule.output_buffer_uops.iter().map(|u| u.buf_uop().id).collect();
    let alias_output_buffers = output_uop_ids
        .iter()
        .filter_map(|id| {
            if !alias_output_ids.contains(id) {
                return None;
            }
            allocated_buffers.get(id).cloned().or_else(|| input_buffers.get(id).cloned()).map(|buffer| (*id, buffer))
        })
        .collect();
    Ok(ScheduleResult { items: schedule, output_uop_ids, alias_output_buffers })
}

pub fn create_schedule(
    transformed: Arc<UOp>,
    input_buffers: &InputBuffers,
    var_vals: &HashMap<String, i64>,
) -> Result<ScheduleResult> {
    let pre = create_pre_schedule(transformed)?;
    instantiate_schedule(&pre, input_buffers, var_vals, false)
}

/// Extract device from the first input buffer in callable sources.
///
/// The first buffer's device determines the device for codegen/compilation and
/// output buffer allocation.
///
/// DISK buffers are skipped: a disk-resident input is never a viable execution
/// device (no compiler), so the search continues to find a compute-capable
/// source. When every source is on disk (e.g. fully-materialized parameter
/// graphs that exist only as safetensors mmaps before realization), we fall
/// back to CPU so the kernel still has somewhere to run; the runtime then
/// arranges the disk→CPU copies via the normal Copy ops.
fn find_first_input_buffer_device(
    sources: &[Arc<UOp>],
    input_buffers: &InputBuffers,
    allocated_buffers: &HashMap<u64, Buffer>,
) -> Result<Arc<Device>> {
    let alloc_registry = registry::registry();

    for src in sources {
        if let Some(buf_id) = source_primary_buffer_id(src) {
            let buffer = allocated_buffers.get(&buf_id).cloned().or_else(|| input_buffers.get(&buf_id).cloned());
            if let Some(buffer) = buffer {
                let device_spec = buffer.allocator().device_spec();
                if device_spec.is_disk() {
                    continue;
                }
                return svod_runtime::DEVICE_FACTORIES
                    .device(&device_spec, alloc_registry)
                    .context(DeviceFactorySnafu)
                    .map_err(Into::into);
            }
        }
    }

    // Fallback to the active default device if no input buffers found. Routes
    // to `SVOD_DEVICE` / `set_default_device` instead of always picking CPU,
    // so JIT placeholders (`Tensor::zeros` + `realize`) materialize on the
    // current target. Critical for ending up with a uniform device per
    // kernel (otherwise `KernelSplitMixedDevices` fires when AMD weights meet
    // a CPU-allocated placeholder).
    svod_runtime::DEVICE_FACTORIES
        .device(&svod_dtype::default_device::default_device(), alloc_registry)
        .context(DeviceFactorySnafu)
        .map_err(Into::into)
}

/// Collect buffers for a callable from its sources.
///
/// This walks the callable sources and identifies:
/// - Input buffers (Op::Buffer) - get from input_buffers
/// - Intermediate buffers (Op::Param) - allocate and track
/// - Shared buffers (Op::After) - look up from allocated_buffers (producer callable)
///
/// For input buffers (PARAM that maps to an original BUFFER),
/// we reuse the existing buffer from input_buffers instead of allocating.
///
/// For shared buffers (AFTER nodes), we look up the buffer using buf_uop()
/// which walks through AFTER chains to get the underlying buffer ID.
///
/// Output/intermediate buffers are allocated on the same device as the first
/// input buffer. Newly allocated buffers are tracked in `allocated_buffers`.
fn collect_callable_buffers(
    sources: &[Arc<UOp>],
    input_buffers: &InputBuffers,
    allocated_buffers: &mut HashMap<u64, Buffer>,
    output_ids: &HashSet<u64>,
) -> Result<CallableBuffers> {
    // Get target device from the first input buffer.
    let target_device = find_first_input_buffer_device(sources, input_buffers, allocated_buffers)?;
    // Device-local intermediates are only meaningful where the backend has a
    // host-visible/device-local split (queried as an allocator capability).
    // Elsewhere keep the default spec — e.g. the CPU allocator turns
    // `cpu_access: false` into a non-host-readable buffer, which would break
    // kernels that access it directly.
    let device_local_backend = target_device.allocator.supports_device_local();

    let mut buffers = Vec::new();
    let mut uop_ids = Vec::new();

    for src in sources {
        let canonical_src = canonicalize_callable_source(src);

        match canonical_src.op() {
            Op::After(ops::After { passthrough, .. }) => {
                // Shared buffer from producer kernel.
                // Use buf_uop() to get underlying buffer ID (handles AFTER chains).
                let buf_id = passthrough.buf_uop().id;

                // Look up from allocated_buffers or input_buffers
                let existing = allocated_buffers.get(&buf_id).cloned().or_else(|| input_buffers.get(&buf_id).cloned());

                if let Some(buffer) = existing {
                    trace!(
                        buf_id,
                        buffer.id = ?buffer.id(),
                        "Found shared buffer from AFTER"
                    );

                    // Track under buf_id if not already tracked
                    allocated_buffers.entry(buf_id).or_insert_with(|| buffer.clone());

                    buffers.push(buffer);
                    uop_ids.push(buf_id);
                } else {
                    trace!(buf_id, "after buffer not found in allocated_buffers or input_buffers");
                    return Err(ErrorKind::BufferNotFound { uop_id: buf_id }.into());
                }
            }
            Op::MSelect(..) | Op::MStack(..) => {
                let Some(canonical_id) = source_primary_buffer_id(&canonical_src) else {
                    return IrConstructionSnafu {
                        details: format!(
                            "multi-device callable source must resolve a primary buffer id: source_id={}, op={:?}",
                            canonical_src.id,
                            canonical_src.op()
                        ),
                    }
                    .fail()
                    .map_err(Into::into);
                };

                let existing =
                    allocated_buffers.get(&canonical_id).cloned().or_else(|| input_buffers.get(&canonical_id).cloned());

                if let Some(buffer) = existing {
                    trace!(canonical_id, buffer.id = ?buffer.id(), "Found shared buffer from MSELECT/MSTACK source");
                    allocated_buffers.entry(canonical_id).or_insert_with(|| buffer.clone());
                    buffers.push(buffer);
                    uop_ids.push(canonical_id);
                } else {
                    trace!(canonical_id, "multi-device source buffer not found in allocated_buffers or input_buffers");
                    return Err(ErrorKind::BufferNotFound { uop_id: canonical_id }.into());
                }
            }
            // LOCAL/REG BUFFERs are compiler-managed and never runtime arguments.
            Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace != Some(svod_ir::AddrSpace::Global) => {}
            Op::Buffer(..) | Op::Param(..) => {
                let size = match canonical_src.op() {
                    Op::Buffer(..) => canonical_src.buffer_size().ok_or_else(|| ErrorKind::IrConstruction {
                        details: "BUFFER allocation requires a concrete shape".to_string(),
                    })?,
                    Op::Param(..) => canonical_src
                        .shape()
                        .ok()
                        .flatten()
                        .and_then(svod_ir::shape::to_static)
                        .and_then(|shape| shape.into_iter().try_fold(1usize, usize::checked_mul))
                        .ok_or_else(|| ErrorKind::IrConstruction {
                            details: "PARAM allocation requires a concrete shape".to_string(),
                        })?,
                    _ => unreachable!(),
                };
                let canonical_id = canonical_src.buf_uop().id;

                // BUFFER/PARAM can be either input (from input_buffers) or output (needs allocation)
                // Try input_buffers first, then allocated_buffers, then allocate new
                if let Some(buffer) =
                    input_buffers.get(&canonical_id).cloned().or_else(|| input_buffers.get(&canonical_src.id).cloned())
                {
                    buffers.push(buffer);
                    uop_ids.push(canonical_id);
                } else if let Some(buffer) = allocated_buffers
                    .get(&canonical_id)
                    .cloned()
                    .or_else(|| allocated_buffers.get(&canonical_src.id).cloned())
                {
                    buffers.push(buffer);
                    uop_ids.push(canonical_id);
                } else {
                    // Fresh buffer: a final output (host reads it → host-visible)
                    // or an internal intermediate (device-local on AMD).
                    trace!(src.id = canonical_src.id, canonical_id, size, "Allocating output BUFFER/PARAM");
                    let scalar_dtype = canonical_src.dtype();

                    let spec = if device_local_backend && !output_ids.contains(&canonical_id) {
                        svod_device::BufferSpec { cpu_access: false, ..Default::default() }
                    } else {
                        Default::default()
                    };
                    let buffer = Buffer::new(target_device.allocator.clone(), scalar_dtype.clone(), vec![size], spec);

                    // Track in allocated_buffers
                    allocated_buffers.insert(canonical_id, buffer.clone());
                    buffers.push(buffer);
                    uop_ids.push(canonical_id);
                }
            }
            Op::Bind(..) => {
                // Variable binding - not a buffer, skip
                continue;
            }
            other => {
                return IrConstructionSnafu {
                    details: format!("unsupported callable source op for buffer collection: {other:?}"),
                }
                .fail()
                .map_err(Into::into);
            }
        }
    }

    Ok(CallableBuffers { buffers, uop_ids })
}

#[derive(Clone)]
struct ScheduleItemTemplate {
    kernel: Arc<UOp>,
    ast: Arc<UOp>,
    lane_buffers: Vec<CallableBuffers>,
    dependencies: Vec<u64>,
    base_fixedvars: HashMap<String, i64>,
    bind_device_num: bool,
}

fn insert_fixedvar_checked(
    fixedvars: &mut HashMap<String, i64>,
    call_id: u64,
    name: &str,
    incoming: i64,
) -> Result<()> {
    if let Some(&existing) = fixedvars.get(name) {
        return Err(ErrorKind::MultiBindingConflict { call_id, name: name.to_string(), existing, incoming }.into());
    }
    fixedvars.insert(name.to_string(), incoming);
    Ok(())
}

fn schedule_range_bounds(range: &Arc<UOp>) -> Result<(i64, i64)> {
    let Op::Range(..) = range.op() else {
        return IrConstructionSnafu {
            details: format!("expected RANGE for schedule loop control, got {:?}", range.op()),
        }
        .fail()
        .map_err(Into::into);
    };

    let Some(vmin) = range.vmin().try_int() else {
        return IrConstructionSnafu {
            details: format!("schedule range vmin must be concrete integer, got {:?}", range.vmin()),
        }
        .fail()
        .map_err(Into::into);
    };
    let Some(vmax) = range.vmax().try_int() else {
        return IrConstructionSnafu {
            details: format!("schedule range vmax must be concrete integer, got {:?}", range.vmax()),
        }
        .fail()
        .map_err(Into::into);
    };
    if vmax < vmin {
        return IrConstructionSnafu { details: format!("invalid schedule range bounds: vmin={vmin}, vmax={vmax}") }
            .fail()
            .map_err(Into::into);
    }
    Ok((vmin, vmax))
}
