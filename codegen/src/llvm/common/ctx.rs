//! Render context for LLVM IR text generation.
//!
//! Maps UOp IDs to LLVM variable names and manages naming.
//! Shared between CPU and GPU backends.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use svod_ir::{ConstValue, Op, prelude::*};

use super::types::{lconst, ldt};
use svod_ir::ops;

/// Maps UOp ID → LLVM variable name.
pub struct RenderContext {
    names: HashMap<u64, String>,
    counter: usize,
    /// Stack of currently open RANGE axis_ids (for correct END footer ordering).
    /// Pushed on RANGE emission, popped on END emission.
    range_stack: Vec<String>,
    /// Side-channel error set by `render_uop` when it detects a graph invariant
    /// violation. The render loop drains this after each call and propagates as
    /// a typed [`crate::Error`].
    pending_error: Option<crate::Error>,
    /// Module-level LLVM IR lines that must be emitted *before* `define`. Used
    /// by AMD LOCAL BUFFER rendering to emit `@local_N = addrspace(3) global ...`
    /// declarations, which cannot be expressed inline in the function body.
    module_prefix: Vec<String>,
    /// Metadata tuple a loop's back edge carries (an unroll hint); `None`
    /// renders loops bare.
    loop_hint: Option<&'static str>,
    /// RANGE ids whose loops stay bare even with a hint set.
    unhinted: HashSet<u64>,
    /// Loop IDs handed out so far, one per hinted back edge.
    loops: usize,
}

impl RenderContext {
    pub fn new() -> Self {
        Self {
            names: HashMap::new(),
            counter: 0,
            range_stack: Vec::new(),
            pending_error: None,
            module_prefix: Vec::new(),
            loop_hint: None,
            unhinted: HashSet::new(),
            loops: 0,
        }
    }

    /// Attach `hint`, a metadata tuple such as `!{!"llvm.loop.unroll.disable"}`,
    /// to every loop rendered from here on except those over the RANGEs in
    /// `unhinted`.
    pub fn set_loop_hint(&mut self, hint: &'static str, unhinted: HashSet<u64>) {
        self.loop_hint = Some(hint);
        self.unhinted = unhinted;
    }

    /// The branch that closes the loop over `range` (axis `id`) — back to its
    /// latch, with a loop ID of its own when it carries the hint.
    pub fn back_edge(&mut self, range: &Arc<UOp>, id: &str) -> String {
        let branch = format!("  br label %loop_latch_{id}");
        if self.loop_hint.is_none() || self.unhinted.contains(&range.id) {
            return branch;
        }
        self.loops += 1;
        format!("{branch}, !llvm.loop !{}", self.loops)
    }

    /// The module-level nodes the back edges refer to: the hint as `!0`, then
    /// one distinct, self-referential loop ID per loop, as LLVM requires.
    pub fn loop_metadata(&self) -> Vec<String> {
        match self.loop_hint {
            Some(hint) if self.loops > 0 => std::iter::once(format!("!0 = {hint}"))
                .chain((1..=self.loops).map(|n| format!("!{n} = distinct !{{!{n}, !0}}")))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Append a line to the module-level prefix block. Used by AMD's
    /// LOCAL BUFFERs to emit `@local_N = ... addrspace(3) global ...`.
    pub fn push_module_prefix(&mut self, line: impl Into<String>) {
        self.module_prefix.push(line.into());
    }

    /// Borrow the accumulated module-level prefix lines.
    pub fn module_prefix(&self) -> &[String] {
        &self.module_prefix
    }

    /// Record an `InvalidGraph` error from a renderer op handler.
    pub fn set_invalid_graph(&mut self, reason: impl Into<String>) {
        if self.pending_error.is_none() {
            self.pending_error = Some(crate::Error::InvalidGraph { reason: reason.into() });
        }
    }

    /// Record an `UnsupportedOp` error from a renderer op handler that reached an
    /// op variant it cannot lower.
    pub fn set_unsupported_op(&mut self, op: impl Into<String>) {
        if self.pending_error.is_none() {
            self.pending_error = Some(crate::Error::UnsupportedOp { op: op.into() });
        }
    }

    /// Drain any error recorded via [`Self::set_invalid_graph`].
    pub fn take_error(&mut self) -> Option<crate::Error> {
        self.pending_error.take()
    }

    /// Get or create variable name for UOp.
    ///
    /// For constants, returns literal value.
    /// For definitions, returns argument name.
    /// For other ops, returns a generated variable name.
    pub fn name(&mut self, uop: &Arc<UOp>) -> String {
        if let Some(name) = self.names.get(&uop.id) {
            return name.clone();
        }

        let name = match uop.op() {
            Op::Const(cv) => lconst(&cv.0, &uop.dtype()),
            Op::VConst(ops::VConst { values }) => self.render_vconst(values, uop),
            Op::Param(ops::Param { arg, .. }) => format!("%data{}", arg.slot),
            Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local) => {
                format!("%local{}", arg.slot)
            }
            Op::DefineVar(ops::DefineVar { name, .. }) => format!("%{name}"),
            Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Reg) => {
                let n = format!("%reg{}", self.counter);
                self.counter += 1;
                n
            }
            Op::Range(ops::Range { axis_id, .. }) => {
                // Range variables are named by axis_id
                format!("%r{}", axis_id.name())
            }
            _ => {
                let n = format!("%v{}", self.counter);
                self.counter += 1;
                n
            }
        };

        self.names.insert(uop.id, name.clone());
        name
    }

    /// Render a vector constant.
    fn render_vconst(&self, values: &[ConstValue], uop: &Arc<UOp>) -> String {
        let scalar_type = ldt(&uop.dtype().scalar_dtype());

        // Format as LLVM vector constant: <type val, type val, ...>
        let elements: Vec<String> = values
            .iter()
            .map(|v| {
                let val = lconst(v, &uop.dtype());
                format!("{scalar_type} {val}")
            })
            .collect();

        format!("<{}>", elements.join(", "))
    }

    /// Get existing name (panics if not found).
    pub fn get(&self, uop: &Arc<UOp>) -> &str {
        self.names.get(&uop.id).map(|s| s.as_str()).unwrap_or_else(|| {
            // NB: print only the op *kind* (`as_ref`), never `{:?}` the op — a
            // valueless node (e.g. a BARRIER consumed as a value) is typically deep
            // in a heavily-shared graph, and `Op`'s recursive `Debug` expands that
            // DAG into an exponential (multi-GB) tree, OOM-ing before the panic can
            // even print. The kind + ids are enough to locate the offending edge.
            panic!("UOp {} (op {}) not in render context", uop.id, uop.op().as_ref())
        })
    }

    /// Try to get existing name.
    pub fn try_get(&self, uop: &Arc<UOp>) -> Option<&str> {
        self.names.get(&uop.id).map(|s| s.as_str())
    }

    /// Check if a UOp is already registered.
    pub fn contains(&self, id: u64) -> bool {
        self.names.contains_key(&id)
    }

    /// Alias one ID to another's name.
    pub fn alias(&mut self, id: u64, name: String) {
        self.names.insert(id, name);
    }

    /// Pre-register a name for a UOp ID.
    pub fn register(&mut self, id: u64, name: String) {
        self.names.insert(id, name);
    }

    /// Get current variable counter.
    pub fn counter(&self) -> usize {
        self.counter
    }

    /// Push a range axis_id onto the open-range stack (called during RANGE codegen).
    pub fn push_range(&mut self, axis_id: String) {
        self.range_stack.push(axis_id);
    }

    /// Close the exact innermost range named by END.
    pub fn close_range(&mut self, expected: &str) -> bool {
        match self.range_stack.pop() {
            Some(actual) if actual == expected => true,
            Some(actual) => {
                self.set_invalid_graph(format!("END closes range {expected}, but innermost open range is {actual}"));
                false
            }
            None => {
                self.set_invalid_graph(format!("END closes range {expected}, but no range is open"));
                false
            }
        }
    }

    pub fn open_ranges(&self) -> &[String] {
        &self.range_stack
    }
}

impl Default for RenderContext {
    fn default() -> Self {
        Self::new()
    }
}
