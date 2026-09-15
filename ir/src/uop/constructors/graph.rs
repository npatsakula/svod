//! Graph organization: sink, group, dependencies, contiguous.
//!
//! This module contains graph organization and optimization operations:
//! - Graph structure: sink, group
//! - Dependencies: after
//! - Materialization: detach, contiguous, contiguous_backward
//! - Optimization hints: precast
//! - Custom code: custom, customi

use std::sync::Arc;

use smallvec::SmallVec;
use svod_dtype::DType;

use crate::op::Op;
use crate::ops;
use crate::types::{CallInfo, CustomFunctionKind, KernelInfo};
use crate::uop::UOp;
use crate::{Error, Result};

/// Bodies routed through `Op::Call` rather than auto-wrapped into
/// `Op::Function`. These are already-compiled / non-value-producing nodes —
/// their dtype is `Void` and they have no value to project via `gettuple`.
fn is_opaque_call_body(op: &Op) -> bool {
    matches!(
        op,
        Op::Sink(..) | Op::Program(..) | Op::Linear(..) | Op::Copy(..) | Op::Slice(..) | Op::CustomFunction(..)
    )
}

impl UOp {
    fn placeholder_like_anchor(src: &Arc<Self>) -> Arc<Self> {
        match src.op() {
            Op::Multi(ops::Multi { src: shard, .. }) => Self::placeholder_like_anchor(shard),
            Op::MSelect(ops::MSelect { buffer, .. }) => Self::placeholder_like_anchor(buffer),
            Op::MStack(ops::MStack { buffers }) if !buffers.is_empty() => Self::placeholder_like_anchor(&buffers[0]),
            _ => src.clone(),
        }
    }

    // =========================================================================
    // Graph Structure
    // =========================================================================

    /// Create a sink operation (graph termination).
    ///
    /// Sink marks outputs that must be evaluated. All sources are dependencies.
    pub fn sink(sources: Vec<Arc<Self>>) -> Arc<Self> {
        Self::new(Op::Sink(ops::Sink { sources: SmallVec::from_vec(sources), info: None }), DType::Void)
    }

    /// Create a sink carrying a structural [`KernelInfo`] marker.
    ///
    /// The marker participates in hash consing — marked and unmarked sinks
    /// with otherwise identical sources are distinct UOps. Used to mark a
    /// SINK as a fully-formed kernel AST that downstream gates skip over.
    pub fn sink_with_info(sources: Vec<Arc<Self>>, info: impl Into<Box<KernelInfo>>) -> Arc<Self> {
        Self::new(Op::Sink(ops::Sink { sources: SmallVec::from_vec(sources), info: Some(info.into()) }), DType::Void)
    }

    /// Create a group operation (merging/organizing related ops).
    ///
    /// Group is a NOOP that helps organize related operations together.
    /// It passes through the first source while ensuring all sources are dependencies.
    pub fn group(sources: Vec<Arc<Self>>) -> Arc<Self> {
        let dtype = if sources.is_empty() { DType::Void } else { sources[0].dtype.clone() };
        Self::new(Op::Group(ops::Group { sources: SmallVec::from_vec(sources) }), dtype)
    }

    // =========================================================================
    // Dependencies
    // =========================================================================

    /// Ordering constraint: self depends on deps.
    ///
    /// # Arguments
    /// * `deps` - Dependencies that must complete before this value is used
    ///
    /// # Panics (debug only)
    /// Panics if self is a control flow node (Range, End)
    pub fn after(self: &Arc<Self>, deps: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        #[cfg(debug_assertions)]
        debug_assert!(
            !matches!(self.op(), Op::Range(..) | Op::End(..)),
            "AFTER passthrough must be data-producing node, got {:?} (id={})",
            self.op(),
            self.id
        );

        let dtype = self.dtype();
        Self::new(Op::After(ops::After { passthrough: self.clone(), deps }), dtype)
    }

    // =========================================================================
    // Materialization
    // =========================================================================

    /// Detach from gradient flow / force materialization.
    pub fn detach(self: &Arc<Self>) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::Detach(ops::Detach { src: self.clone() }), dtype)
    }

    /// Ensure contiguous memory layout.
    ///
    /// Elides the CONTIGUOUS wrapper when the source is already contiguous:
    /// - Already a CONTIGUOUS node (no double wrapping)
    /// - Has buffer identity (BUFFER, or RESHAPE/MULTI chain to BUFFER)
    ///
    /// Based on Tinygrad's `UOp.contiguous()` (ops.py:463-466).
    pub fn contiguous(self: &Arc<Self>) -> Arc<Self> {
        if matches!(self.op(), Op::Contiguous(..)) {
            return self.clone();
        }
        if self.has_buffer_identity() {
            return self.clone();
        }
        let dtype = self.dtype();
        Self::new(Op::Contiguous(ops::Contiguous { src: self.clone(), opts: Vec::new() }), dtype)
    }

    /// Ensure contiguous memory layout with optimization hints.
    ///
    /// The hints are extracted during rangeify and passed to the optimizer.
    /// Based on Tinygrad's CONTIGUOUS.arg which carries Opt tuples.
    pub fn contiguous_with_opts(self: &Arc<Self>, opts: impl Into<Vec<crate::types::ContiguousHint>>) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::Contiguous(ops::Contiguous { src: self.clone(), opts: opts.into() }), dtype)
    }

    /// Contiguous backward pass.
    pub fn contiguous_backward(self: &Arc<Self>) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::ContiguousBackward(ops::ContiguousBackward { src: self.clone() }), dtype)
    }

    // =========================================================================
    // Optimization Hints
    // =========================================================================

    /// Optimizer hint to force materialization before type conversion.
    ///
    /// Inserted before BITCAST to ensure the source is rendered separately
    /// in codegen (prevents invalid cast fusion).
    pub fn precast(self: &Arc<Self>) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::Precast(ops::Precast { src: self.clone() }), dtype)
    }

    // =========================================================================
    // Custom Code
    // =========================================================================

    /// Inject custom code as a statement in the generated kernel.
    ///
    /// `deps` are UOps whose rendered names can be referenced in `code`.
    /// `dtype` specifies the result type (often Void for statements).
    ///
    /// This op is rendered by backend codegen as inline/source-template code.
    /// For typed runtime helpers (encdec/graph style), use `custom_function`.
    pub fn custom(deps: SmallVec<[Arc<Self>; 4]>, code: String, dtype: DType) -> Arc<Self> {
        Self::new(Op::Custom(ops::Custom { deps, code }), dtype)
    }

    /// Create an explicit runtime custom-function operation.
    ///
    /// This mirrors Tinygrad's `CUSTOM_FUNCTION` op family: the body op encodes
    /// semantic runtime behavior (for example `EncDec`), while CALL provides the
    /// runtime buffer arguments.
    pub fn custom_function(kind: CustomFunctionKind, attrs: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        Self::new(Op::CustomFunction(ops::CustomFunction { kind, attrs }), DType::Void)
    }

    /// Inject custom code as an inline expression.
    ///
    /// Unlike `custom` (statement), this is substituted directly into expressions.
    /// `deps` provide values to reference; result has specified `dtype`.
    pub fn customi(deps: SmallVec<[Arc<Self>; 4]>, code: String, dtype: DType) -> Arc<Self> {
        Self::new(Op::CustomI(ops::CustomI { deps, code }), dtype)
    }

    /// Create a PARAM placeholder shaped like `src` for custom kernel building.
    ///
    /// Rejects symbolic input via `all_int`-style check and creates storage in
    /// the requested address space with the same logical shape, matching
    /// Tinygrad's `placeholder_like`. For multi-device wrappers (MULTI/MSELECT/MSTACK),
    /// placeholder shape is derived from the underlying shard (analog of
    /// tinygrad's `max_shard_shape`).
    pub fn placeholder_like(src: &Arc<Self>, slot: usize, addrspace: svod_dtype::AddrSpace) -> Result<Arc<Self>> {
        let anchor = Self::placeholder_like_anchor(src);
        let shape = anchor.shape()?.cloned().ok_or_else(|| Error::MissingShape { operation: "placeholder_like" })?;
        Self::placeholder(&shape, anchor.dtype(), slot, addrspace, None)
    }

    /// The realized node a chain of reshapes views, if the base is one.
    fn realized_base(x: &Arc<Self>) -> Option<Arc<Self>> {
        match x.op() {
            Op::After(..) => Some(x.clone()),
            Op::Reshape(ops::Reshape { src, .. }) => Self::realized_base(src),
            _ => None,
        }
    }

    /// Build a custom kernel callable and return `AFTER(callable)` outputs for all inputs.
    ///
    /// Input sources are made contiguous (except AFTER nodes and reshapes of them,
    /// which bind the realized node itself), placeholders are built from the
    /// caller's sources, and the closure returns the kernel body UOp.
    ///
    /// Body dispatch mirrors tinygrad's `_OPAQUE_CALL_BODIES` set:
    /// - Opaque bodies (`Sink`, `Program`, `Linear`, `Copy`, `Slice`,
    ///   `CustomFunction`) wrap into `Op::Call`. An info-less `Sink` is
    ///   auto-promoted to a kernel-marked `Sink` first.
    /// - Value-producing bodies wrap into `Op::Function`, which auto-wraps
    ///   non-`Tuple` bodies via `maketuple`. This keeps `custom_kernel`
    ///   spec-compliant when the closure returns e.g. an arithmetic UOp.
    pub fn custom_kernel<F>(srcs: Vec<Arc<Self>>, fxn: F, info: CallInfo) -> Result<Vec<Arc<Self>>>
    where
        F: FnOnce(Vec<Arc<Self>>) -> Arc<Self>,
    {
        // Materialising an input is the producer's work, not the caller's: the copy
        // takes the source's origin so every scope that hands over the same node
        // shares one materialisation instead of minting one per call site. A
        // reshape of an already-realized source is the same bytes, so the kernel
        // is handed that source instead of a copy; the placeholder keeps the
        // caller's shape.
        let placeholders: Vec<Arc<Self>> = srcs
            .iter()
            .enumerate()
            .map(|(i, s)| UOp::placeholder_like(s, i, svod_dtype::AddrSpace::Global))
            .collect::<Result<_>>()?;
        let contig_srcs: Vec<Arc<Self>> = srcs
            .into_iter()
            .map(|x| Self::realized_base(&x).unwrap_or_else(|| x.contiguous().rorigin(x.origin())))
            .collect();

        let mut body = fxn(placeholders);
        if let Op::Sink(ops::Sink { sources, info: None }) = body.op() {
            body = Self::sink_with_info(sources.to_vec(), KernelInfo::default());
        }
        let args = SmallVec::from_vec(contig_srcs.clone());
        let callable = if is_opaque_call_body(body.op()) {
            // Same contract as the scheduler's kernel cut: the body keys every cache
            // downstream, so a hand-lowered kernel built once per module must still
            // hash-cons to one body and attribution moves onto the callable. An
            // opaque body never reaches `split_store`, so this is its only chance.
            // A FUNCTION body is inlined by rangeify and cut like any other graph,
            // so its nodes keep their origins for the cut to harvest.
            let mut info = info;
            if crate::origin::any_interned() {
                let (primary, origins) = body.kernel_attribution();
                if !origins.is_empty() {
                    info.origin = info.origin.or(primary);
                    info.origins.extend(&origins);
                    body = body.without_origins();
                }
            }
            body.call(args, info)
        } else {
            body.function(args, info)
        };

        Ok(contig_srcs.into_iter().map(|s| s.after(SmallVec::from_vec(vec![callable.clone()]))).collect())
    }
}
