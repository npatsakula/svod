//! Common utilities shared between codegen backends.

use std::collections::HashMap;
use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{ConstValue, Op, UOp};

use crate::{Error, Result};
use svod_ir::ops;

/// FNUZ FP8 formats have different bias, zero, NaN, and saturation semantics
/// from the OCP formats. No renderer currently implements those semantics.
pub fn reject_unsupported_fnuz(nodes: &[Arc<UOp>], renderer: &str) -> Result<()> {
    if let Some(dtype) = nodes.iter().map(|node| node.dtype().base()).find(|dtype| dtype.is_fp8_fnuz()) {
        return Err(Error::TypeError {
            reason: format!(
                "{renderer} renderer does not support {dtype:?}; FNUZ cannot use OCP FP8 decomposition or raw-byte fallback"
            ),
        });
    }
    Ok(())
}

/// Lane count of a memory access, taken from the address expression. Grouped
/// accesses keep a scalar dtype and carry their width in the SHRINK size (or,
/// pre-shape-migration, in a vector dtype).
pub fn access_width(index: &Arc<UOp>) -> usize {
    match index.op() {
        Op::Shrink(ops::Shrink { sizes, .. }) => match sizes.op() {
            Op::Const(value) => match value.0 {
                ConstValue::Int(value) if value > 0 => value as usize,
                ConstValue::UInt(value) if value > 0 => value as usize,
                _ => 1,
            },
            _ => 1,
        },
        Op::Cast(ops::Cast { src, .. }) => access_width(src),
        _ => index.dtype().vcount(),
    }
}

/// Lane count of a value. This branch carries the count in the UOp shape rather
/// than the dtype, so a shape-`[N]` scalar-dtype value renders as an `N`-lane
/// vector.
pub fn value_width(value: &Arc<UOp>) -> usize {
    if value.dtype().vcount() > 1 {
        return value.dtype().vcount();
    }
    match value.op() {
        Op::Stack(ops::Stack { sources }) => sources.len(),
        Op::Load(ops::Load { index, .. }) => access_width(index),
        Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) | Op::Cast(..) | Op::BitCast(..) | Op::Wmma(..) => value
            .shape()
            .ok()
            .flatten()
            .and_then(|shape| shape.iter().try_fold(1usize, |count, dim| count.checked_mul(dim.as_const()?)))
            .unwrap_or(1),
        _ => 1,
    }
}

/// The dtype a memory access renders as. Its lane count is the wider of the
/// stored value's and the address's, so a scalar-dtype address still renders a
/// full vector access instead of truncating the value to one lane.
pub fn access_dtype(index: &Arc<UOp>, value: &Arc<UOp>) -> DType {
    let width = value_width(value).max(access_width(index));
    if width > 1 {
        value.dtype().scalar_dtype().vec(width).expect("grouped access dtype must be vectorizable")
    } else {
        value.dtype()
    }
}

/// The dtype a value renders as: its scalar dtype widened to [`value_width`].
pub fn shaped_dtype(value: &Arc<UOp>) -> DType {
    let count = value_width(value);
    if count > 1 {
        value.dtype().scalar_dtype().vec(count).expect("grouped value dtype must be vectorizable")
    } else {
        value.dtype()
    }
}

/// Whether a custom statement of the graph reads and writes `param` in place
/// ([`UOp::read_modify_writes`]).
pub fn is_modified_in_place(param: &Arc<UOp>, nodes: &[Arc<UOp>]) -> bool {
    let indexes = |dep: &Arc<UOp>| matches!(dep.op(), Op::Index(ops::Index { buffer, .. }) if buffer.id == param.id);
    nodes.iter().any(|node| node.read_modify_writes().any(indexes))
}

/// Check whether a buffer (PARAM/DefineGlobal) is written in the graph: a STORE
/// target, or modified in place by a custom statement.
pub fn is_output_buffer(def_global: &Arc<UOp>, nodes: &[Arc<UOp>]) -> bool {
    let buffer_id = def_global.id;
    if is_modified_in_place(def_global, nodes) {
        return true;
    }

    for node in nodes {
        if let Some(buffer) = node.store_buffer() {
            if buffer.id == buffer_id {
                return true;
            }
            if let Op::Index(ops::Index { buffer: idx_buf, .. }) = buffer.op()
                && idx_buf.id == buffer_id
            {
                return true;
            }
        }
    }
    false
}

/// `(buffers, variables)` in canonical ABI order.
pub type BuffersAndVars = (Vec<Arc<UOp>>, Vec<Arc<UOp>>);

/// Collect buffer and variable parameters from a UOp graph.
///
/// Collects:
/// - Buffers: address-space PARAM operations
/// - Variables: DEFINE_VAR and scalar PARAM operations
///
/// Returns (buffers, variables) sorted for deterministic function signatures.
pub fn collect_buffers_and_vars(root: &Arc<UOp>) -> Result<BuffersAndVars> {
    let nodes = root.toposort();
    let params = collect_abi_params(&nodes)?;
    Ok(params
        .into_iter()
        .partition(|param| matches!(param.op(), Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_some())))
}

/// Collect PARAMs in the canonical external ABI order. All PARAM address
/// spaces are arguments in the pinned renderer; local/register BUFFERs are
/// internal scratch allocations and therefore deliberately excluded.
pub(crate) fn collect_abi_params(nodes: &[Arc<UOp>]) -> Result<Vec<Arc<UOp>>> {
    let mut params = Vec::new();
    let mut occupied = HashMap::new();
    for node in nodes {
        let Op::Param(ops::Param { arg, .. }) = node.op() else { continue };
        if arg.slot == usize::MAX {
            return Err(Error::InvalidGraph { reason: "unassigned PARAM reached renderer ABI collection".into() });
        }
        if arg.addrspace.is_none() && arg.name.is_none() {
            return Err(Error::InvalidGraph { reason: format!("scalar PARAM in slot {} has no name", arg.slot) });
        }
        if let Some(first) = occupied.insert(arg.slot, node.id) {
            return Err(Error::InvalidGraph {
                reason: format!("duplicate PARAM slot {} for UOps {first} and {}", arg.slot, node.id),
            });
        }
        params.push(node.clone());
    }
    params.sort_by_key(|param| match param.op() {
        Op::Param(ops::Param { arg, .. }) => arg.slot,
        _ => usize::MAX,
    });
    Ok(params)
}

pub fn validate_custom_template_strict(template: &str, arg_count: usize) -> Result<()> {
    let mut chars = template.chars().peekable();
    let mut auto_idx = 0usize;
    let mut saw_auto = false;
    let mut saw_manual = false;

    while let Some(ch) = chars.next() {
        if ch == '{' {
            if matches!(chars.peek(), Some('{')) {
                chars.next();
                continue;
            }

            let mut token = String::new();
            let mut found_close = false;
            for next in chars.by_ref() {
                if next == '}' {
                    found_close = true;
                    break;
                }
                token.push(next);
            }

            if !found_close {
                return Err(Error::InvalidGraph {
                    reason: format!("custom template has unmatched '{{': {template:?}"),
                });
            }

            let idx = if token.is_empty() {
                saw_auto = true;
                let i = auto_idx;
                auto_idx += 1;
                i
            } else {
                saw_manual = true;
                token.parse::<usize>().map_err(|e| Error::InvalidGraph {
                    reason: format!(
                        "custom template placeholder must be empty or numeric, got {{{token}}} in {template:?}: {e}"
                    ),
                })?
            };

            if saw_auto && saw_manual {
                return Err(Error::InvalidGraph {
                    reason: format!("custom template mixes automatic {{}} and manual {{N}} placeholders: {template:?}"),
                });
            }

            if idx >= arg_count {
                return Err(Error::InvalidGraph {
                    reason: format!(
                        "custom template placeholder index {idx} out of bounds (args={arg_count}) in {template:?}"
                    ),
                });
            }
        } else if ch == '}' {
            if matches!(chars.peek(), Some('}')) {
                chars.next();
            } else {
                return Err(Error::InvalidGraph {
                    reason: format!("custom template has unmatched '}}': {template:?}"),
                });
            }
        }
    }

    Ok(())
}

pub fn format_custom_template_strict(template: &str, args: &[String]) -> Result<String> {
    validate_custom_template_strict(template, args.len())?;

    let mut out = String::new();
    let mut chars = template.chars().peekable();
    let mut auto_idx = 0usize;

    while let Some(ch) = chars.next() {
        if ch == '{' {
            if matches!(chars.peek(), Some('{')) {
                chars.next();
                out.push('{');
                continue;
            }

            let mut token = String::new();
            for next in chars.by_ref() {
                if next == '}' {
                    break;
                }
                token.push(next);
            }

            let idx = if token.is_empty() {
                let i = auto_idx;
                auto_idx += 1;
                i
            } else {
                token.parse::<usize>().expect("placeholder token validated")
            };

            out.push_str(&args[idx]);
        } else if ch == '}' {
            chars.next();
            out.push('}');
        } else {
            out.push(ch);
        }
    }

    Ok(out)
}
