//! Tree visualization for UOp graphs.
//!
//! Provides pretty-printing of UOp computation graphs as ASCII trees.

use std::collections::HashSet;
use std::fmt::Write;
use std::sync::Arc;

use crate::ops;
use crate::{ConstValue, ConstValueHash, Op, UOp};

/// Truncate a code/source string to a fixed number of leading chars for display.
fn truncate_for_display(code: &str) -> String {
    code.chars().take(20).collect()
}

/// Format a single UOp node for display.
///
/// Output format: `[id] OP_NAME : dtype shape=[...]`
fn format_node(uop: &Arc<UOp>) -> String {
    let op_str = match uop.op() {
        Op::Const(ConstValueHash(ConstValue::Invalid)) => "INVALID".to_string(),
        Op::Const(val) => format!("CONST({:?})", val.0),
        Op::DefineVar(ops::DefineVar { name, min_val, max_val }) => {
            format!("DEFINE_VAR('{name}', min={min_val}, max={max_val})")
        }
        Op::Param(ops::Param { arg, .. }) => format!("PARAM(slot={})", arg.slot),
        Op::Buffer(ops::Buffer { arg, .. }) => format!("BUFFER(slot={}, addrspace={:?})", arg.slot, arg.addrspace),
        Op::Stage(ops::Stage { opts, .. }) => match &opts.local_axis {
            Some(axis) => format!("STAGE(local_axis={axis})"),
            None => "STAGE".to_string(),
        },
        Op::Load(..) => "LOAD".to_string(),
        Op::Store(..) => "STORE".to_string(),
        Op::Index(..) => "INDEX".to_string(),
        Op::GetAddr(ops::GetAddr { device, .. }) => format!("GETADDR({})", device.canonicalize()),
        Op::Binary(bop, ..) => format!("{bop:?}"),
        Op::Unary(uop_kind, ..) => format!("{uop_kind:?}"),
        Op::Ternary(top, ..) => format!("{top:?}"),
        Op::Cast(..) => "CAST".to_string(),
        Op::BitCast(..) => "BITCAST".to_string(),
        Op::Reduce(ops::Reduce { reduce_op, ranges, num_axes, .. }) => {
            let range_ids: Vec<u64> = ranges.iter().map(|r| r.id).collect();
            format!("REDUCE({reduce_op:?}, num_axes={num_axes}, ranges={range_ids:?})")
        }
        Op::ReduceAxis(ops::ReduceAxis { reduce_op, axes, .. }) => format!("REDUCE_AXIS({reduce_op:?}, axes={axes:?})"),
        Op::AllReduce(ops::AllReduce { reduce_op, device, .. }) => format!("ALL_REDUCE({reduce_op:?}, {device:?})"),
        Op::Bind(..) => "BIND".to_string(),
        Op::Range(ops::Range { axis_id, axis_type, .. }) => format!("RANGE({axis_id}, {axis_type:?})"),
        Op::End(..) => "END".to_string(),
        Op::Sink(ops::Sink { info: Some(_), .. }) => "SINK[KERNEL]".to_string(),
        Op::Sink(ops::Sink { info: None, .. }) => "SINK".to_string(),
        Op::Group(..) => "GROUP".to_string(),
        Op::Call(ops::Call { args, .. }) => format!("CALL(args={})", args.len()),
        Op::Function(ops::Function { args, .. }) => format!("FUNCTION(args={})", args.len()),
        Op::Program(ops::Program { info, .. }) => {
            format!("PROGRAM(name={}, target={})", info.name, info.target.canonicalize())
        }
        Op::Linear(ops::Linear { ops }) => format!("LINEAR(len={})", ops.len()),
        Op::Tuple(ops::Tuple { src }) => format!("TUPLE(len={})", src.len()),
        Op::GetTuple(ops::GetTuple { index, .. }) => format!("GETTUPLE({index})"),
        Op::Source(ops::Source { code, identity }) => {
            format!("SOURCE('{}', identity={})", truncate_for_display(code), identity.is_some())
        }
        Op::ProgramBinary(ops::ProgramBinary { bytes, identity }) => {
            format!("BINARY(len={}, identity={})", bytes.len(), identity.is_some())
        }
        Op::Ins(ops::Ins { arg, .. }) => format!("INS({})", arg.opcode),
        Op::Stack(ops::Stack { sources }) => format!("STACK(len={})", sources.len()),
        Op::VConst(ops::VConst { values }) => format!("VCONST(len={})", values.len()),
        Op::Reshape(..) => "RESHAPE".to_string(),
        Op::Permute(ops::Permute { axes, .. }) => format!("PERMUTE(axes={axes:?})"),
        Op::Expand(..) => "EXPAND".to_string(),
        Op::Pad(..) => "PAD".to_string(),
        Op::Shrink(..) => "SHRINK".to_string(),
        Op::Flip(ops::Flip { axes, .. }) => format!("FLIP(axes={axes:?})"),
        Op::Multi(ops::Multi { axis, .. }) => format!("MULTI(axis={axis})"),
        Op::Contiguous(..) => "CONTIGUOUS".to_string(),
        Op::ContiguousBackward(..) => "CONTIGUOUS_BACKWARD".to_string(),
        Op::Copy(ops::Copy { device, .. }) => format!("COPY({device:?})"),
        Op::Custom(ops::Custom { code, .. }) => format!("CUSTOM('{}')", truncate_for_display(code)),
        Op::CustomFunction(ops::CustomFunction { kind, .. }) => format!("CUSTOM_FUNCTION({kind:?})"),
        Op::CustomI(ops::CustomI { code, .. }) => format!("CUSTOM_I('{}')", truncate_for_display(code)),
        Op::Unique(id) => format!("UNIQUE({id})"),
        Op::LUnique(id) => format!("LUNIQUE({id})"),
        Op::Noop => "NOOP".to_string(),
        Op::Slice(ops::Slice { size, .. }) => format!("SLICE(size={size})"),
        Op::MStack(..) => "MSTACK".to_string(),
        Op::MSelect(ops::MSelect { device_index, .. }) => format!("MSELECT(idx={device_index})"),
        Op::Special(ops::Special { name, .. }) => format!("SPECIAL('{name}')"),
        Op::If(..) => "IF".to_string(),
        Op::EndIf(..) => "END_IF".to_string(),
        Op::Barrier(..) => "BARRIER".to_string(),
        Op::Wmma(ops::Wmma { metadata, .. }) => {
            format!("WMMA(upcast={:?}, reduce={:?})", metadata.upcast_axes, metadata.reduce_axes)
        }
        Op::Detach(..) => "DETACH".to_string(),
        Op::After(..) => "AFTER".to_string(),
        Op::Precast(..) => "PRECAST".to_string(),
    };

    // Get shape if available
    let shape_str = match uop.shape() {
        Ok(Some(shape)) => format!(" shape={:?}", shape.as_slice()),
        Ok(None) => String::new(),
        Err(_) => " shape=?".to_string(),
    };

    let dtype = uop.dtype();
    let id = uop.id;
    format!("[{id}] {op_str} : {dtype:?}{shape_str}")
}

/// Render a UOp graph as a compact ASCII tree string.
///
/// Shared nodes (appearing multiple times due to hash-consing) are shown
/// as back-references: `[id] → (see above)`
pub fn render_tree_compact(uop: &Arc<UOp>) -> String {
    render(uop, Some(HashSet::new()))
}

/// Render a UOp graph as a full ASCII tree string.
///
/// Shared nodes are expanded every time they appear (verbose but complete).
pub fn render_tree_full(uop: &Arc<UOp>) -> String {
    render(uop, None)
}

/// With `visited`, a node printed once is shown as a back-reference afterwards.
fn render(uop: &Arc<UOp>, mut visited: Option<HashSet<u64>>) -> String {
    let mut out = String::new();
    write_node(&mut out, uop, "", "", &mut visited);
    out
}

fn write_node(out: &mut String, uop: &Arc<UOp>, lead: &str, child_lead: &str, visited: &mut Option<HashSet<u64>>) {
    let backref = visited.as_mut().is_some_and(|seen| !seen.insert(uop.id));
    let label = if backref { format!("[{}] → (see above)", uop.id) } else { format_node(uop) };
    writeln!(out, "{lead}{label}").expect("writing to a String cannot fail");
    if backref {
        return;
    }
    let sources = uop.op().sources();
    for (i, src) in sources.iter().enumerate() {
        let (branch, rest) = if i + 1 == sources.len() { ("└── ", "    ") } else { ("├── ", "│   ") };
        write_node(out, src, &format!("{child_lead}{branch}"), &format!("{child_lead}{rest}"), visited);
    }
}
