//! UOp spec verification — Rust port of tinygrad's `tinygrad/uop/spec.py`.
//!
//! tinygrad expresses each kernel invariant as a `(UPat, predicate)` rule in a
//! `PatternMatcher` and runs `type_verify(ast, spec)` to check every uop
//! against it (`spec.py:31`): a uop is valid only if the first matching rule's
//! predicate returns `True`, and a uop matching **no** rule also fails — the
//! spec is a *whitelist* (`spec.py:38`, `ret is not True → raise`). Because
//! Python is untyped, tinygrad reuses its rewriter `PatternMatcher` for this; in
//! Rust we mirror the design with a dedicated [`Spec`] of validity rules and a
//! [`type_verify`] runner.
//!
//! Verification is gated by `SVOD_SPEC` (default on, like tinygrad's `SPEC=1`)
//! so it can be disabled for perf. It turns a malformed kernel — a movement op
//! that should have been lowered to index arithmetic, a `<N x float>` leaking
//! into a memory index — into a recoverable `Err` *before*
//! the renderer turns it into a panic / malformed IR / GPU fault, so beam search
//! skips the offending candidate cleanly.
//!
//! [`spec_program`] is a whitelist (port of tinygrad's `spec_program` +
//! `spec_shared`): a lowered, pre-render kernel may contain only the ops below.
//! Both tensor and program specs are whitelists, matching Tinygrad.

use std::collections::HashSet;
use std::sync::Arc;

use snafu::Snafu;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::ops;
use svod_ir::{BinaryOp, ConstValue, Op, TernaryOp, UOp, UnaryOp};

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
#[snafu(visibility(pub))]
pub enum SpecError {
    #[snafu(display(
        "{boundary} verification failed at {index} on {op} id={uop_id} (dtype {dtype}, source path {source_path:?}): {reason}"
    ))]
    Verification {
        boundary: &'static str,
        index: usize,
        uop_id: u64,
        op: String,
        dtype: String,
        source_path: Vec<usize>,
        reason: &'static str,
    },
}

/// A single spec rule (mirrors one `(UPat, predicate)` entry in tinygrad).
///
/// Returns `None` if the rule does not apply to `u`, `Some(Ok(()))` if it
/// applies and `u` is valid, or `Some(Err(reason))` if it applies and `u`
/// violates the invariant.
type SpecRule = Box<dyn Fn(&Arc<UOp>) -> Option<Result<(), &'static str>> + Send + Sync>;

/// An ordered set of validity rules (mirrors a tinygrad spec `PatternMatcher`).
pub struct Spec {
    rules: Vec<SpecRule>,
    /// Whitelist mode (tinygrad `spec_program`/`spec_tensor`): a uop matching no
    /// rule fails. Assertion mode (`false`): unmatched uops pass — used while a
    /// spec set is still being filled in.
    whitelist: bool,
}

impl Spec {
    /// Verdict of the first applicable rule, or `None` if no rule applies
    /// (tinygrad: the first matching pattern wins).
    fn check(&self, u: &Arc<UOp>) -> Option<Result<(), &'static str>> {
        self.rules.iter().find_map(|rule| rule(u))
    }
}

/// Whether spec verification runs. Mirrors tinygrad's `SPEC` ContextVar
/// (default 1 = on); disable with `SVOD_SPEC=0`.
pub fn spec_enabled() -> bool {
    !matches!(std::env::var("SVOD_SPEC").as_deref(), Ok("0"))
}

/// Check every uop reachable from `root` against `spec` (port of `type_verify`,
/// `spec.py:31`). In whitelist mode a uop matching no rule fails, exactly as
/// tinygrad's `ret is not True → raise`.
pub fn type_verify(root: &Arc<UOp>, spec: &Spec) -> Result<(), SpecError> {
    type_verify_call_aware(root, spec, true, "UOp")
}

fn verification_sources(node: &Arc<UOp>, include_call_bodies: bool) -> Vec<(usize, Arc<UOp>)> {
    if include_call_bodies {
        return node.op().sources().into_iter().enumerate().collect();
    }
    match node.op() {
        // Source zero is the opaque body; retain the real source indices in diagnostics.
        Op::Call(ops::Call { args, .. }) | Op::Function(ops::Function { args, .. }) => {
            args.iter().cloned().enumerate().map(|(index, arg)| (index + 1, arg)).collect()
        }
        Op::Program(..) => Vec::new(),
        _ => node.op().sources().into_iter().enumerate().collect(),
    }
}

/// Toposort for verification — tinygrad's `list(ast.toposort(enter_calls=...))`
/// (`spec.py:36`). A plain `Vec<Arc<UOp>>`: the source path each node was
/// reached by is a failure-only diagnostic, so it is not carried here.
fn verification_order(root: &Arc<UOp>, include_call_bodies: bool) -> Vec<Arc<UOp>> {
    let mut visited = HashSet::new();
    let mut nodes = Vec::new();
    let mut stack = vec![(root.clone(), false)];
    while let Some((node, processed)) = stack.pop() {
        if visited.contains(&node.id) {
            continue;
        }
        if processed {
            visited.insert(node.id);
            nodes.push(node);
            continue;
        }
        stack.push((node.clone(), true));
        for (_, child) in verification_sources(&node, include_call_bodies).into_iter().rev() {
            if !visited.contains(&child.id) {
                stack.push((child, false));
            }
        }
    }
    nodes
}

/// Source indices leading from `root` to `target`, for the rejection message.
/// Walked only after a node has already failed, so its cost is not on the
/// verification path.
fn source_path_to(root: &Arc<UOp>, target: u64, include_call_bodies: bool) -> Vec<usize> {
    let mut visited = HashSet::new();
    let mut stack = vec![(root.clone(), Vec::new())];
    while let Some((node, path)) = stack.pop() {
        if node.id == target {
            return path;
        }
        if !visited.insert(node.id) {
            continue;
        }
        for (source_index, child) in verification_sources(&node, include_call_bodies).into_iter().rev() {
            stack.push((child, [path.as_slice(), &[source_index]].concat()));
        }
    }
    Vec::new()
}

/// Verify an already-linearized list against `spec` — tinygrad's
/// `type_verify(lst, check_spec)` (`spec.py:35-40`), where `ast` may be a list
/// and `lst[-1]` is taken to be the sink.
pub fn type_verify_list(nodes: &[Arc<UOp>], spec: &Spec) -> Result<(), SpecError> {
    check_nodes(nodes, spec, "linear program", nodes.last(), true)
}

fn type_verify_call_aware(
    root: &Arc<UOp>,
    spec: &Spec,
    include_call_bodies: bool,
    boundary: &'static str,
) -> Result<(), SpecError> {
    let nodes = verification_order(root, include_call_bodies);
    check_nodes(&nodes, spec, boundary, Some(root), include_call_bodies)
}

fn check_nodes(
    nodes: &[Arc<UOp>],
    spec: &Spec,
    boundary: &'static str,
    root: Option<&Arc<UOp>>,
    include_call_bodies: bool,
) -> Result<(), SpecError> {
    for (index, u) in nodes.iter().enumerate() {
        let reason = match spec.check(u) {
            Some(Ok(())) => continue,
            Some(Err(reason)) => reason,
            None if spec.whitelist => "op not allowed in this spec (no matching rule)",
            None => continue,
        };
        // tinygrad prints the linearized uops on failure when DEBUG>=3
        // (`spec.py:39`); mirror that with a gated dump for diagnosis.
        if std::env::var_os("SVOD_SPEC_DEBUG").is_some() {
            eprintln!("[SPEC] reject #{index} {} (dtype {:?}): {reason}", u.op().as_ref(), u.dtype());
            eprintln!("{}", u.tree());
        }
        return VerificationSnafu {
            boundary,
            index,
            uop_id: u.id,
            op: u.op().as_ref().to_string(),
            dtype: format!("{:?}", u.dtype()),
            source_path: root.map(|root| source_path_to(root, u.id, include_call_bodies)).unwrap_or_default(),
            reason,
        }
        .fail();
    }
    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

#[inline]
fn ok_if(valid: bool, reason: &'static str) -> Result<(), &'static str> {
    if valid { Ok(()) } else { Err(reason) }
}

fn matches_dtype(value: &Arc<UOp>, dtype: &DType) -> bool {
    value.dtype() == *dtype || UOp::is_invalid_marker(value)
}

fn is_invalid_index_value(value: &Arc<UOp>) -> bool {
    if !value.dtype().is_bool() {
        return false;
    }
    match value.op() {
        Op::Const(cvh) => cvh.0 == ConstValue::Invalid,
        Op::VConst(ops::VConst { values }) => {
            !values.is_empty() && values.iter().all(|value| *value == ConstValue::Invalid)
        }
        Op::Stack(ops::Stack { sources }) => !sources.is_empty() && sources.iter().all(is_invalid_index_value),
        _ => false,
    }
}

fn legal_address(address: &Arc<UOp>) -> bool {
    match address.op() {
        Op::Index(..) | Op::Shrink(..) => true,
        Op::Cast(ops::Cast { src, .. }) => matches!(src.op(), Op::Index(..) | Op::Shrink(..)),
        _ => false,
    }
}

// Tinygrad's CHECK_OOB defaults to disabled. The checks below mirror the
// enabled-independent early exits; Svod has no dynamic CHECK_OOB context or
// validate_index_with_z3 equivalent.
fn validate_index(address: &Arc<UOp>) -> bool {
    let Op::Index(ops::Index { indices, .. }) = address.op() else { return true };
    if indices.len() != 1 {
        return true;
    }
    if UOp::is_invalid_marker(&indices[0]) {
        return true;
    }
    true
}

// ============================================================================
// spec_shared — valid in both the tensor graph and lowered programs
// (port of `spec_shared`, spec.py:45)
// ============================================================================

/// `spec.py:52` — a `CONST`'s value type must match its dtype.
fn rule_const() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Const(cvh) => {
            let valid = match cvh.0 {
                ConstValue::Invalid => u.dtype() == DType::Bool,
                ConstValue::Bool(_) => u.dtype().is_bool(),
                ConstValue::Float(_) => u.dtype().is_float(),
                ConstValue::Int(_) | ConstValue::UInt(_) => u.dtype().is_int(),
            };
            Some(ok_if(valid, "CONST value type does not match its dtype"))
        }
        Op::VConst(ops::VConst { values }) => {
            let valid = values.len() == u.dtype().vcount()
                && values.iter().all(|value| match value {
                    ConstValue::Invalid | ConstValue::Bool(_) => u.dtype().is_bool(),
                    ConstValue::Float(_) => u.dtype().is_float(),
                    ConstValue::Int(_) | ConstValue::UInt(_) => u.dtype().is_int(),
                });
            Some(ok_if(valid, "VCONST value types do not match its dtype"))
        }
        Op::Stack(..) if is_invalid_index_value(u) => Some(Ok(())),
        _ => None,
    })
}

/// `spec.py:56-61` — ALU dtype invariants. WHERE/CMP/SHL-SHR/CDIV-CMOD are
/// special-cased exactly as tinygrad; every other ALU shares one base dtype.
fn rule_alu() -> SpecRule {
    Box::new(|u| {
        let result_base = u.dtype().base();
        match u.op() {
            // Unary preserves dtype.
            Op::Unary(_, x) => {
                Some(ok_if(matches_dtype(x, &u.dtype()) || x.dtype().is_weak(), "unary operand dtype mismatch"))
            }

            // WHERE: bool condition, matching value/result dtypes (spec.py:56).
            Op::Ternary(TernaryOp::Where, c, x, y) => Some(ok_if(
                c.dtype() == DType::Bool
                    && (matches_dtype(x, &u.dtype()) || x.dtype().is_weak())
                    && (matches_dtype(y, &u.dtype()) || y.dtype().is_weak()),
                "WHERE condition must be bool with matching value/result dtypes",
            )),
            // MULACC: a*b+c, all sharing the result base.
            Op::Ternary(TernaryOp::MulAcc, a, b, c) => Some(ok_if(
                [a, b, c].iter().all(|s| matches_dtype(s, &u.dtype()) || s.dtype().is_weak()),
                "MULACC operand dtype mismatch",
            )),

            Op::Binary(op, x, y) => {
                let (xb, yb) = (x.dtype().base(), y.dtype().base());
                let valid =
                    if matches!(op, BinaryOp::And | BinaryOp::Or | BinaryOp::Xor | BinaryOp::Shl | BinaryOp::Shr)
                        && (x.dtype().is_float() || y.dtype().is_float())
                    {
                        false
                    } else if op.is_comparison() {
                        // CMPLT/CMPNE/CMPEQ: bool result, operands share base (spec.py:57).
                        u.dtype() == DType::Bool
                            && (matches_dtype(x, &y.dtype())
                                || matches_dtype(y, &x.dtype())
                                || x.dtype().is_weak()
                                || y.dtype().is_weak())
                    } else if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
                        // Same-dtype shifts use the generic ALU rule. The only
                        // differing shift-count dtype accepted by the pinned rule
                        // is uint32.
                        (matches_dtype(x, &u.dtype()) || x.dtype().is_weak())
                            && (matches_dtype(y, &u.dtype()) || y.dtype().is_weak() || y.dtype() == DType::UInt32)
                    } else if matches!(op, BinaryOp::FloorDiv | BinaryOp::FloorMod | BinaryOp::CDiv | BinaryOp::CMod) {
                        // Integer div/mod variants must be integer (spec.py:60).
                        u.dtype().is_int() && xb == result_base && yb == result_base
                    } else {
                        (matches_dtype(x, &u.dtype()) || x.dtype().is_weak())
                            && (matches_dtype(y, &u.dtype()) || y.dtype().is_weak())
                    };
                Some(ok_if(valid, "binary operand/result dtype mismatch"))
            }
            _ => None,
        }
    })
}

/// `spec.py:59-62` — STACK is empty/void or a shaped collection of
/// same-shaped values matching the promoted scalar result dtype.
fn rule_stack() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Stack(ops::Stack { sources }) if sources.is_empty() => {
            Some(ok_if(u.dtype() == DType::Void, "empty STACK must have void dtype"))
        }
        Op::Stack(ops::Stack { sources }) => {
            let first_shape = sources[0].shape().ok().flatten();
            Some(ok_if(
                sources.iter().all(|source| {
                    source.shape().ok().flatten() == first_shape
                        && source.dtype().vcount() == 1
                        && (matches_dtype(source, &u.dtype()) || source.dtype().is_weak())
                }),
                "STACK sources must have matching shapes and scalar dtype",
            ))
        }
        _ => None,
    })
}

/// `spec.py:67` — RANGE dtype matches its bound's dtype.
fn rule_range() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Range(ops::Range { end, axis_id, .. }) => Some(ok_if(
            !axis_id.path().is_empty() && matches_dtype(end, &u.dtype()),
            "RANGE requires a non-empty integer axis path and matching bound dtype",
        )),
        _ => None,
    })
}

/// `spec.py:82` — every `INDEX` address operand must be integer.
fn rule_index_integer() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Index(ops::Index { indices, .. }) => Some(ok_if(
            indices.iter().all(|idx| idx.dtype().is_int() || is_invalid_index_value(idx)),
            "non-integer value reached a memory INDEX operand",
        )),
        _ => None,
    })
}

/// `spec.py:117-122` — gated and ungated memory accesses have four exact source
/// layouts and use INDEX/SHRINK addresses (optionally cast).
fn rule_memory() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Load(ops::Load { index, alt: None, gate: None }) if legal_address(index) => {
            Some(ok_if(validate_index(index), "invalid ungated LOAD"))
        }
        Op::Load(ops::Load { index, alt: Some(alt), gate: Some(gate) })
            if legal_address(index) && gate.dtype() == DType::Bool =>
        {
            Some(ok_if(
                matches_dtype(alt, &u.dtype()) && validate_index(index),
                "gated LOAD requires a matching alt and valid index",
            ))
        }
        Op::Store(ops::Store { index, gate: None, .. }) if legal_address(index) => {
            Some(ok_if(validate_index(index), "invalid ungated STORE"))
        }
        Op::Store(ops::Store { index, gate: Some(gate), .. })
            if legal_address(index) && gate.dtype() == DType::Bool =>
        {
            Some(ok_if(validate_index(index), "invalid gated STORE"))
        }
        _ => None,
    })
}

/// `spec.py:84-86` — every range an END closes must be a `RANGE`, or the END
/// carries only backedge sources.
///
/// Tinygrad's second rule is `END(x, RANGE(void), bool)`, where a void RANGE is
/// its bound-less loop header. Svod has no void RANGE (a range's dtype is its
/// index dtype), so that arm can never fire. What `split_end_with_tag` does
/// produce is the tail `END` of the split — `ret.end(*backedge)` in tinygrad's
/// `do_split_ends` (`linearizer.py:88-90`) — whose sources are exactly the
/// void/bool ones partitioned out, and which are not RANGEs at all. Accept that
/// shape instead.
fn rule_end() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::End(ops::End { ranges, .. }) if ranges.iter().all(|r| matches!(r.op(), Op::Range(..))) => Some(Ok(())),
        Op::End(ops::End { ranges, .. })
            if ranges.iter().all(|r| r.dtype() == DType::Void || r.dtype() == DType::Bool) =>
        {
            Some(Ok(()))
        }
        _ => None,
    })
}

/// `spec.py:88-89` — PARAM carries structured metadata and its storage dtype.
fn rule_param() -> SpecRule {
    Box::new(|u| match u.op() {
        // ParamArg is statically present in Svod's PARAM variant.
        Op::Param(..) => Some(Ok(())),
        _ => None,
    })
}

/// Pinned `spec.py:228`: GETADDR is a scalar uint64 address for a concrete
/// device and accepts exactly BUFFER/PARAM storage, optionally through AFTER.
fn rule_getaddr() -> SpecRule {
    fn storage_source(source: &Arc<UOp>) -> bool {
        match source.op() {
            Op::Buffer(ops::Buffer { arg, .. }) | Op::Param(ops::Param { arg, .. }) => arg.addrspace.is_some(),
            Op::After(ops::After { passthrough, .. }) => storage_source(passthrough),
            _ => false,
        }
    }

    Box::new(|u| match u.op() {
        Op::GetAddr(ops::GetAddr { src, device }) => Some(ok_if(
            u.dtype() == DType::UInt64 && storage_source(src) && is_device(device),
            "GETADDR requires BUFFER/PARAM storage, uint64 result, and a concrete device",
        )),
        _ => None,
    })
}

fn is_device(device: &DeviceSpec) -> bool {
    match device {
        DeviceSpec::Cpu
        | DeviceSpec::Cuda { .. }
        | DeviceSpec::Amd { .. }
        | DeviceSpec::Metal { .. }
        | DeviceSpec::WebGpu
        | DeviceSpec::Disk { .. } => true,
    }
}

/// Pinned `spec.py:90-91`: lowered storage is exactly one shape source plus
/// ParamArg metadata, and only REG/LOCAL BUFFERs are legal in programs.
fn rule_program_buffer() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Buffer(ops::Buffer { arg, .. }) => Some(ok_if(
            matches!(arg.addrspace, Some(AddrSpace::Reg | AddrSpace::Local))
                && arg.device.is_none()
                && u.dtype() == arg.dtype,
            "program BUFFER must be a structured REG/LOCAL allocation",
        )),
        _ => None,
    })
}

/// `spec.py:78` — a GROUP holds stores, groups, or noops.
fn rule_group() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Group(ops::Group { sources }) => Some(ok_if(
            u.dtype() == DType::Void
                && sources
                    .iter()
                    .all(|s| matches!(s.op(), Op::Store(..) | Op::Group(..) | Op::Noop | Op::Ins(..) | Op::End(..))),
            "GROUP must be void and may only hold GROUP/STORE/NOOP/INS/END",
        )),
        _ => None,
    })
}

fn rule_after() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::After(ops::After { passthrough, .. })
            if passthrough.op().is_movement()
                || matches!(
                    passthrough.op(),
                    Op::Param(..)
                        | Op::Buffer(..)
                        | Op::Index(..)
                        | Op::After(..)
                        | Op::BitCast(..)
                        | Op::Contiguous(..)
                        | Op::Ins(..)
                ) =>
        {
            Some(ok_if(matches_dtype(passthrough, &u.dtype()), "AFTER passthrough dtype mismatch"))
        }
        _ => None,
    })
}

fn rule_shared_structural() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Sink(..) => Some(ok_if(u.dtype() == DType::Void, "SINK must be void")),
        Op::Noop => Some(Ok(())),
        Op::Cast(ops::Cast { dtype, .. }) | Op::BitCast(ops::BitCast { dtype, .. }) => {
            Some(ok_if(*dtype == u.dtype(), "CAST arg dtype must equal result dtype"))
        }
        Op::Custom(..) | Op::CustomI(..) => Some(Ok(())),
        Op::Call(ops::Call { body, .. }) if body.dtype() != DType::Void => {
            Some(ok_if(matches_dtype(body, &DType::UInt64), "non-void CALL target must be uint64"))
        }
        Op::Barrier(..) => Some(ok_if(u.dtype() == DType::Void, "BARRIER must be void")),
        Op::Wmma(..) => Some(Ok(())),
        Op::Ins(..) => Some(Ok(())),
        _ => None,
    })
}

fn rule_tensor_store() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Store(ops::Store { gate: None, .. }) => Some(ok_if(u.dtype() == DType::Void, "tensor STORE must be void")),
        _ => None,
    })
}

fn spec_shared() -> Vec<SpecRule> {
    vec![
        // Pinned spec.py:49-129 order. Some adjacent, disjoint Python patterns
        // share a Rust closure, but their first-match precedence is unchanged.
        rule_shared_structural(),
        rule_const(),
        rule_stack(),
        rule_alu(),
        rule_range(),
        rule_index_integer(),
        rule_end(),
        rule_param(),
        rule_program_buffer(),
        rule_group(),
        rule_after(),
        rule_memory(),
        rule_tensor_store(),
    ]
}

// ============================================================================
// spec_program — additionally valid in lowered programs (port of `spec_program`,
// spec.py:203)
// ============================================================================

/// `spec.py:205` — weak dtypes are legal in the tensor graph but must be
/// committed before a program. This must run before the shared CONST rule.
fn rule_no_weak_dtype() -> SpecRule {
    Box::new(|u| u.dtype().is_weak().then_some(Err("weak dtype must be lowered before a program")))
}

fn rule_no_legacy_index_dtype() -> SpecRule {
    Box::new(|u| {
        (u.dtype().base() == svod_dtype::ScalarDType::Index)
            .then_some(Err("legacy Index dtype must be lowered before a program"))
    })
}

/// Typed constants reaching a program must already carry the exact semantic
/// value obtained by committing each lane to their declared scalar dtype.
fn rule_canonical_const() -> SpecRule {
    Box::new(|u| {
        let values: &[ConstValue] = match u.op() {
            Op::Const(value) if value.0 != ConstValue::Invalid => std::slice::from_ref(&value.0),
            Op::VConst(ops::VConst { values }) => values,
            _ => return None,
        };
        let scalar_dtype = u.dtype().scalar_dtype();
        (!values.iter().all(|value| {
            *value == ConstValue::Invalid || value.cast(&scalar_dtype).is_some_and(|committed| committed == *value)
        }))
        .then_some(Err("typed constant value is not canonical for its dtype"))
    })
}

/// `spec.py:208` — the buffer-cropping SHRINK used by memory accesses is the
/// sole movement op allowed in a program. This ordered exception must run
/// before the general movement rejection.
fn rule_special_shrink() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Shrink(ops::Shrink { src, sizes, .. })
            if matches!(src.op(), Op::Param(..) | Op::Buffer(..) | Op::After(..))
                && matches!(sizes.op(), Op::Const(_)) =>
        {
            Some(Ok(()))
        }
        _ => None,
    })
}

/// `spec.py:210-211` — movement ops are lowered to index arithmetic before a kernel
/// is linearized; a surviving one is a lowering bug.
fn rule_no_movement() -> SpecRule {
    Box::new(|u| u.op().is_movement().then_some(Err("movement op must be lowered away before a program")))
}

/// `spec.py:216-217` — Svod models `Invalid` as its own op (vs tinygrad's
/// `CONST(arg=Invalid)`); it must be folded out before a program.
fn rule_no_invalid() -> SpecRule {
    Box::new(|u| UOp::is_invalid_marker(u).then_some(Err("Invalid constant must be folded out before a program")))
}

fn rule_no_tensor_reduce() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Reduce(ops::Reduce { num_axes, .. }) if *num_axes != 0 => {
            Some(Err("tensor-form REDUCE must be rangeified before a program"))
        }
        _ => None,
    })
}

/// `spec.py:219-220` — IF has a bool gate; ENDIF closes an IF. A gated-store
/// IF carries its one dedup source; a control IF of a pre-linearized program
/// has none (its region is the list up to the ENDIF).
fn rule_if() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::If(ops::If { condition, body }) => Some(ok_if(
            u.dtype() == DType::Void
                && condition.dtype() == DType::Bool
                && (body.is_empty()
                    || (body.len() == 1 && matches!(body[0].op(), Op::Cast(..) | Op::Index(..) | Op::Shrink(..)))),
            "IF must be void with a bool condition and at most one CAST/INDEX/SHRINK dedup source",
        )),
        Op::EndIf(ops::EndIf { if_op }) => Some(ok_if(
            u.dtype() == DType::Void && matches!(if_op.op(), Op::If(..)),
            "ENDIF must be void and close an IF",
        )),
        _ => None,
    })
}

/// `spec.py:223-224` — SPECIAL has an int32 bound and result after index
/// lowering. Its name is statically represented as a String in Svod's schema.
fn rule_program_special() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Special(ops::Special { end, .. }) => Some(ok_if(
            u.dtype() == DType::Int32 && end.dtype() == DType::Int32,
            "SPECIAL bound and result must be int32 after index lowering",
        )),
        _ => None,
    })
}

/// Spec for a lowered, pre-render kernel (`spec_program`, spec.py:200) — a
/// whitelist: program-only rules first, then the shared rules.
pub fn spec_program() -> Spec {
    let mut rules = vec![
        rule_no_legacy_index_dtype(),
        rule_no_weak_dtype(),
        rule_canonical_const(),
        rule_special_shrink(),
        rule_no_movement(),
        // Pinned spec.py:213-214 repeats the REG/LOCAL BUFFER allowance here
        // before INVALID, then appends spec_shared (which contains it again).
        rule_program_buffer(),
        rule_no_invalid(),
        rule_no_tensor_reduce(),
        rule_if(),
        rule_program_special(),
    ];
    rules.extend(spec_shared());
    Spec { rules, whitelist: true }
}

/// Verify the invariant established by `pm_lower_index_dtype` before any
/// target-dependent decomposition can obscure its source.
pub fn verify_no_legacy_index_dtype(root: &Arc<UOp>) -> Result<(), SpecError> {
    let spec = Spec { rules: vec![rule_no_legacy_index_dtype()], whitelist: false };
    type_verify_call_aware(root, &spec, true, "post-index-lowering")
}

/// Runtime command-queue graph spec (`spec_hcq` at the pinned commit).
/// GETADDR intentionally lives here, not in kernel `spec_program`.
pub fn spec_hcq() -> Spec {
    let mut rules = vec![rule_getaddr()];
    rules.extend(spec_shared());
    Spec { rules, whitelist: true }
}

fn rule_tensor_unary() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Unary(UnaryOp::Sin | UnaryOp::Log2 | UnaryOp::Exp2 | UnaryOp::Sqrt | UnaryOp::Reciprocal, _) => {
            Some(ok_if(u.dtype().is_float(), "transcendental unary op must be float"))
        }
        _ => None,
    })
}

/// Pinned `spec.py:140-142`: tensor BUFFERs are GLOBAL storage with a device.
fn rule_tensor_buffer() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Buffer(ops::Buffer { shape, arg }) if arg.addrspace == Some(AddrSpace::Global) => Some(ok_if(
            arg.device.is_some() && u.dtype() == arg.dtype && matches_dtype(shape, &DType::WeakInt),
            "tensor BUFFER must be structured GLOBAL storage with a device",
        )),
        _ => None,
    })
}

fn rule_tensor_slice() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Slice(ops::Slice { buffer, offset, .. }) => Some(ok_if(
            matches!(offset.op(), Op::Const(_))
                && offset.dtype() == DType::WeakInt
                && matches!(buffer.base().op(), Op::Buffer(..) | Op::Param(..) | Op::Stage(..)),
            "SLICE requires buffer-backed storage and a constant weakint offset",
        )),
        _ => None,
    })
}

/// STAGE is an intermediate tensor operation transformed to BUFFER before lowering.
fn rule_tensor_stage() -> SpecRule {
    Box::new(|u| matches!(u.op(), Op::Stage(..)).then_some(Ok(())))
}

fn rule_tensor_bind() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Bind(ops::Bind { var, value }) => Some(ok_if(
            matches!(var.op(), Op::Param(..))
                && [DType::Int32, DType::Int64, DType::WeakInt].contains(&u.dtype())
                && matches!(value.op(), Op::Const(_))
                && value.dtype() == u.dtype(),
            "BIND must bind an integer PARAM to a matching constant",
        )),
        _ => None,
    })
}

fn rule_tensor_call_function_tuple() -> SpecRule {
    Box::new(|u| match u.op() {
        // CustomFunctionKind is a closed enum, so the Python `isinstance(arg,
        // str)` construction check is enforced by the Rust schema.
        Op::CustomFunction(..) => Some(Ok(())),
        Op::Call(ops::Call { body, .. })
            if u.dtype() == DType::Void
                && matches!(
                    body.op(),
                    Op::Sink(..)
                        | Op::Linear(..)
                        | Op::Program(..)
                        | Op::Copy(..)
                        | Op::Slice(..)
                        | Op::CustomFunction(..)
                ) =>
        {
            Some(Ok(()))
        }
        Op::Function(ops::Function { body, .. }) => Some(ok_if(
            u.dtype() == DType::Void && matches!(body.op(), Op::Tuple(..)),
            "FUNCTION must be void and start with TUPLE",
        )),
        Op::Tuple(..) => Some(ok_if(u.dtype() == DType::Void, "TUPLE must be void")),
        Op::GetTuple(ops::GetTuple { src, index }) => {
            let tuple = match src.op() {
                Op::Tuple(ops::Tuple { src }) => Some(src),
                Op::Function(ops::Function { body, .. }) => match body.op() {
                    Op::Tuple(ops::Tuple { src }) => Some(src),
                    _ => None,
                },
                _ => None,
            };
            Some(ok_if(
                tuple.is_some_and(|items| *index < items.len() && matches_dtype(&items[*index], &u.dtype())),
                "GETTUPLE index/source/dtype mismatch",
            ))
        }
        _ => None,
    })
}

fn rule_tensor_special_movement_reduce() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Special(ops::Special { end, .. }) => Some(ok_if(
            end.dtype() == DType::WeakInt && matches_dtype(end, &u.dtype()),
            "tensor SPECIAL must preserve weakint dtype",
        )),
        Op::Reshape(..) | Op::Expand(..) => Some(Ok(())),
        Op::Pad(ops::Pad { begin_pads, end_pads, .. })
        | Op::Shrink(ops::Shrink { offsets: begin_pads, sizes: end_pads, .. }) => Some(ok_if(
            begin_pads.shape().ok().flatten() == end_pads.shape().ok().flatten(),
            "PAD/SHRINK bound shapes must match",
        )),
        Op::Permute(..) | Op::Flip(..) => Some(Ok(())),
        Op::Reduce(ops::Reduce { ranges, .. }) => Some(ok_if(
            ranges.iter().all(|r| [DType::WeakInt, DType::Int32].contains(&r.dtype())),
            "REDUCE ranges must be weakint/int32",
        )),
        _ => None,
    })
}

fn rule_tensor_copy_multi_contiguous() -> SpecRule {
    fn mstack_len(source: &Arc<UOp>) -> Option<usize> {
        match source.op() {
            Op::MStack(ops::MStack { buffers }) => Some(buffers.len()),
            op if op.is_movement() => op.sources().first().and_then(mstack_len),
            _ => None,
        }
    }

    Box::new(|u| match u.op() {
        Op::Copy(ops::Copy { src, device }) => {
            Some(ok_if(matches_dtype(src, &u.dtype()) && is_device(device), "COPY dtype/device mismatch"))
        }
        Op::AllReduce(ops::AllReduce { src, device, .. }) => {
            Some(ok_if(matches_dtype(src, &u.dtype()) && is_device(device), "ALLREDUCE dtype/device mismatch"))
        }
        // Svod represents the supported tuple-device subset explicitly as
        // MSTACK. Keep this strict: arbitrary MSELECT sources are not target
        // tensor forms and must not reach multi_pm.
        Op::MSelect(ops::MSelect { buffer, device_index }) => Some(ok_if(
            mstack_len(buffer).is_some_and(|len| *device_index < len) && matches_dtype(buffer, &u.dtype()),
            "MSELECT requires an in-range MSTACK source with matching dtype",
        )),
        Op::MStack(ops::MStack { buffers }) => Some(ok_if(
            !buffers.is_empty()
                && buffers.iter().all(|s| matches_dtype(s, &u.dtype()))
                && (buffers.iter().all(|s| s.device_spec().is_some())
                    || (!buffers.is_empty()
                        && buffers.iter().all(|s| Arc::ptr_eq(s, &buffers[0]))
                        && buffers[0].device_spec().is_none())),
            "MSTACK device/source mismatch",
        )),
        // Op::Multi is Svod's single-axis representation of tensor UNSHARD.
        Op::Multi(ops::Multi { src, axis }) => Some(ok_if(
            matches_dtype(src, &u.dtype()) && src.shape().ok().flatten().is_some_and(|shape| *axis < shape.len()),
            "MULTI must preserve dtype and shard an existing source axis",
        )),
        Op::Contiguous(ops::Contiguous { src, opts }) => Some(ok_if(
            opts.is_empty() && matches_dtype(src, &u.dtype()),
            "CONTIGUOUS must have no arg and preserve dtype",
        )),
        Op::Detach(ops::Detach { src }) | Op::ContiguousBackward(ops::ContiguousBackward { src }) => {
            Some(ok_if(matches_dtype(src, &u.dtype()), "contiguous/detach dtype mismatch"))
        }
        _ => None,
    })
}

fn rule_tensor_codegen() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Linear(..) => Some(ok_if(u.dtype() == DType::Void, "LINEAR must be void")),
        Op::Source(..) => Some(ok_if(u.dtype() == DType::Void, "SOURCE must be void")),
        Op::ProgramBinary(..) => Some(ok_if(u.dtype() == DType::UInt8, "BINARY must be uint8")),
        Op::Program(ops::Program { sink, linear, source, binary, .. }) => Some(ok_if(
            u.dtype() == DType::Void
                && matches!(sink.op(), Op::Sink(..))
                && source.as_ref().is_none_or(|x| matches!(x.op(), Op::Source(..)))
                && binary.as_ref().is_none_or(|x| matches!(x.op(), Op::ProgramBinary(..)))
                && linear.as_ref().is_none_or(|x| matches!(x.op(), Op::Linear(..)))
                && !(source.is_some() && linear.is_none())
                && !(binary.is_some() && source.is_none()),
            "invalid progressive PROGRAM sources",
        )),
        _ => None,
    })
}

/// Tensor-graph whitelist (`spec_tensor`, pinned spec.py:136-200).
pub fn spec_tensor() -> Spec {
    let mut rules = vec![
        rule_tensor_unary(),
        rule_tensor_buffer(),
        rule_tensor_slice(),
        rule_tensor_stage(),
        rule_tensor_bind(),
        rule_tensor_call_function_tuple(),
        rule_tensor_special_movement_reduce(),
        rule_tensor_copy_multi_contiguous(),
        rule_tensor_codegen(),
    ];
    rules.extend(spec_shared());
    Spec { rules, whitelist: true }
}

fn call_arguments_match_body(body: &Arc<UOp>, args: &[Arc<UOp>]) -> bool {
    let mut slots = HashSet::new();
    for formal in body.toposort_call_aware(false) {
        let Op::Param(ops::Param { arg, .. }) = formal.op() else { continue };
        if arg.slot == usize::MAX {
            continue;
        }
        if !slots.insert(arg.slot) {
            continue;
        }
        let Some(actual) = args.get(arg.slot) else { return false };
        let actual_axis = match actual.op() {
            Op::Param(ops::Param { arg, .. }) | Op::Buffer(ops::Buffer { arg, .. }) => arg.axis,
            Op::After(ops::After { passthrough, .. }) => match passthrough.buf_uop().op() {
                Op::Param(ops::Param { arg, .. }) | Op::Buffer(ops::Buffer { arg, .. }) => arg.axis,
                _ => None,
            },
            _ => None,
        };
        if formal.dtype() != actual.dtype() || arg.axis != actual_axis {
            return false;
        }
    }
    slots.iter().max().is_none_or(|max_slot| *max_slot < args.len())
}

fn supported_kernel_call_body(body: &Arc<UOp>) -> bool {
    match body.op() {
        Op::Sink(..) | Op::Linear(..) | Op::Program(..) | Op::Copy(..) | Op::Slice(..) | Op::CustomFunction(..) => true,
        Op::End(ops::End { computation, .. }) => matches!(computation.op(), Op::Copy(..) | Op::Slice(..)),
        _ => false,
    }
}

fn rule_kernel_graph() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Sink(..) => Some(ok_if(u.dtype() == DType::Void, "kernel-graph SINK must be void")),
        Op::Bind(..) => Some(Ok(())),
        Op::Const(_) => Some(Ok(())),
        Op::Stack(ops::Stack { sources }) => Some(ok_if(
            sources.is_empty()
                || sources.iter().all(|source| matches!(source.op(), Op::Const(_) | Op::Bind(..) | Op::Param(..))),
            "kernel-graph STACK may only contain CONST/BIND/PARAM sources",
        )),
        Op::Param(ops::Param { arg, .. }) => {
            Some(ok_if(u.dtype() == arg.dtype, "kernel-graph PARAM metadata dtype mismatch"))
        }
        Op::Buffer(ops::Buffer { arg, .. }) => Some(ok_if(
            arg.addrspace == Some(AddrSpace::Global) && u.dtype() == arg.dtype,
            "kernel-graph BUFFER must be GLOBAL with matching metadata dtype",
        )),
        Op::Reshape(..) | Op::BitCast(..) => Some(Ok(())),
        // Schedule-level loop: `RANGE` counts the iterations of the `END(CALL,
        // [RANGE])` wrapper that `create_pre_schedule` replays. The bound must
        // be concrete — the scheduler unrolls it eagerly.
        Op::Range(ops::Range { end, .. }) => Some(ok_if(
            matches_dtype(end, &u.dtype()) && end.vmax().try_int().is_some(),
            "kernel-graph RANGE requires a matching-dtype bound with a concrete maximum",
        )),
        Op::End(ops::End { computation, ranges }) => Some(ok_if(
            u.dtype() == DType::Void
                && matches!(computation.op(), Op::Call(..))
                && ranges.len() <= 1
                && ranges.iter().all(|range| matches!(range.op(), Op::Range(..))),
            "kernel-graph END must close at most one RANGE over a CALL",
        )),
        Op::MStack(ops::MStack { buffers }) => Some(ok_if(
            !buffers.is_empty()
                && buffers.iter().all(|source| matches_dtype(source, &u.dtype()))
                && (buffers.iter().all(|source| source.device_spec().is_some())
                    || (buffers.iter().all(|source| Arc::ptr_eq(source, &buffers[0]))
                        && buffers[0].device_spec().is_none())),
            "kernel-graph MSTACK requires a non-empty concrete-device layout or one repeated device-free source",
        )),
        Op::MSelect(ops::MSelect { buffer, device_index }) => Some(ok_if(
            matches!(buffer.op(), Op::MStack(ops::MStack { buffers })
                if *device_index < buffers.len() && matches_dtype(buffer, &u.dtype())),
            "kernel-graph MSELECT requires an in-range MSTACK source with matching dtype",
        )),
        Op::Call(ops::Call { body, args, .. }) => Some(ok_if(
            u.dtype() == DType::Void && supported_kernel_call_body(body) && call_arguments_match_body(body, args),
            "kernel-graph CALL requires a supported opaque body and positional arguments matching its PARAM slots",
        )),
        Op::After(ops::After { passthrough, deps }) => Some(ok_if(
            matches_dtype(passthrough, &u.dtype())
                && (passthrough.op().is_movement()
                    || matches!(
                        passthrough.op(),
                        Op::Param(..)
                            | Op::After(..)
                            | Op::Buffer(..)
                            | Op::MStack(..)
                            | Op::MSelect(..)
                            | Op::BitCast(..)
                            | Op::Reshape(..)
                    ))
                && deps.iter().all(|dep| match dep.op() {
                    Op::Call(..) | Op::After(..) => true,
                    Op::End(ops::End { computation, .. }) => matches!(computation.op(), Op::Call(..)),
                    _ => false,
                }),
            "kernel-graph AFTER requires a storage/view passthrough, matching dtype, and CALL/END(CALL)/AFTER dependencies",
        )),
        _ => None,
    })
}

/// Outer callified graph whitelist from pinned Tinygrad `spec_kernel_graph`.
/// CALL bodies remain opaque, matching `type_verify(..., enter_calls=False)`.
pub fn spec_kernel_graph() -> Spec {
    Spec { rules: vec![rule_kernel_graph()], whitelist: true }
}

pub fn verify_kernel_graph(root: &Arc<UOp>) -> Result<(), SpecError> {
    type_verify_call_aware(root, &spec_kernel_graph(), false, "kernel graph")
}
