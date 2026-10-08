//! Hardware-specific operations: WMMA, lane packing, callable kernels/programs.
//!
//! This module contains hardware-specific operations:
//! - Tensor cores: wmma
//! - Vectorization: stack and index helpers
//! - Multi-device: mstack, mselect
//! - Callable/program IR: call, program

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use snafu::{OptionExt, ensure};
use svod_dtype::DType;

use crate::Result;
use crate::error::{BroadcastRequiresScalarSnafu, GetTupleIndexOutOfBoundsSnafu, GetTupleNotATupleSnafu};
use crate::op::Op;
use crate::ops;
use crate::types::{CallInfo, WmmaMetadata};
use crate::uop::UOp;

impl UOp {
    // =========================================================================
    // Tensor Core Operations
    // =========================================================================

    /// Warp Matrix Multiply-Accumulate for tensor cores.
    ///
    /// Computes D = A × B + C using hardware matrix units.
    /// `metadata` specifies dimensions, dtypes, and upcast axes for vectorization.
    pub fn wmma(a: Arc<Self>, b: Arc<Self>, c: Arc<Self>, metadata: impl Into<Box<WmmaMetadata>>) -> Arc<Self> {
        let dtype = c.dtype();
        Self::new(Op::Wmma(ops::Wmma { a, b, c, metadata: metadata.into() }), dtype)
    }

    // =========================================================================
    // Vectorization Operations
    // =========================================================================

    /// Broadcast a scalar value along a new leading axis (fallible version).
    ///
    /// Creates a STACK operation with `count` copies of the source.
    /// If `count == 1`, returns the source unchanged.
    ///
    /// # Errors
    /// - `BroadcastRequiresScalar` if source has a vector dtype
    pub fn try_broadcast(self: &Arc<Self>, count: usize) -> Result<Arc<Self>> {
        ensure!(self.dtype().vcount() == 1, BroadcastRequiresScalarSnafu { dtype: self.dtype() });

        if count == 1 {
            return Ok(self.clone());
        }
        let elements: SmallVec<[Arc<Self>; 4]> = (0..count).map(|_| self.clone()).collect();
        Ok(Self::stack(elements))
    }

    /// Broadcast a scalar value along a new leading axis.
    ///
    /// Creates a STACK operation with `count` copies of the source.
    /// If `count == 1`, returns the source unchanged.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let vector = scalar.broadcast(4);
    /// ```
    pub fn broadcast(self: &Arc<Self>, count: usize) -> Arc<Self> {
        if count == 1 {
            return self.clone();
        }
        let elements: SmallVec<[Arc<Self>; 4]> = (0..count).map(|_| self.clone()).collect();
        Self::stack(elements)
    }

    // =========================================================================
    // Multi-Device Operations
    // =========================================================================

    /// Stack multiple buffers (multi-device tensors).
    ///
    /// MStack combines buffers from multiple devices into a single logical tensor.
    /// Used for distributed/multi-GPU tensor operations.
    pub fn mstack(buffers: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        let dtype = buffers.first().map(|b| b.dtype()).unwrap_or(DType::Void);
        Self::new(Op::MStack(ops::MStack { buffers }), dtype)
    }

    /// Select buffer by device index (multi-device access).
    ///
    /// MSelect retrieves a specific device's buffer from a multi-device tensor.
    pub fn mselect(self: &Arc<Self>, device_index: usize) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::MSelect(ops::MSelect { buffer: self.clone(), device_index }), dtype)
    }

    // =========================================================================
    // Callable Operations
    // =========================================================================

    /// Callable wrapper around a body UOp and runtime arguments.
    ///
    /// CALL dtype is always void per tinygrad's spec.
    pub fn call(self: &Arc<Self>, args: SmallVec<[Arc<Self>; 4]>, info: impl Into<Box<CallInfo>>) -> Arc<Self> {
        Self::new(Op::Call(ops::Call { body: self.clone(), args, info: info.into() }), DType::Void)
    }

    /// Typed instruction-style CALL. Its result is scalar and its body remains opaque.
    pub fn call_typed(
        self: &Arc<Self>,
        args: SmallVec<[Arc<Self>; 4]>,
        info: CallInfo,
        return_dtype: DType,
    ) -> Arc<Self> {
        Self::new(Op::Call(ops::Call { body: self.clone(), args, info: info.into() }), return_dtype)
    }

    /// FUNCTION wrapper around a value-producing body UOp and runtime arguments.
    ///
    /// FUNCTION dtype is always void per tinygrad's spec, and its body is
    /// always a TUPLE; non-Tuple bodies are auto-wrapped.
    /// For opaque bodies (SINK / PROGRAM / COPY / SLICE / CUSTOM_FUNCTION) prefer
    /// `.call()` instead — those mirror tinygrad's `_OPAQUE_CALL_BODIES` set.
    pub fn function(self: &Arc<Self>, args: SmallVec<[Arc<Self>; 4]>, info: impl Into<Box<CallInfo>>) -> Arc<Self> {
        let body = if matches!(self.op(), Op::Tuple(..)) { self.clone() } else { self.maketuple() };
        Self::new(Op::Function(ops::Function { body, args, info: info.into() }), DType::Void)
    }

    /// Fallible FUNCTION constructor with positional formal/actual validation.
    pub fn try_function(self: &Arc<Self>, args: SmallVec<[Arc<Self>; 4]>, info: CallInfo) -> Result<Arc<Self>> {
        let body = if matches!(self.op(), Op::Tuple(..)) { self.clone() } else { self.maketuple() };
        crate::shape::function_param_substitutions(&body, &args)?;
        Ok(Self::new(Op::Function(ops::Function { body, args, info: info.into() }), DType::Void))
    }

    /// Construct a TUPLE from value-producing UOps. dtype is always void.
    /// Mirrors tinygrad `Ops.TUPLE`.
    pub fn tuple(srcs: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        Self::new(Op::Tuple(ops::Tuple { src: srcs }), DType::Void)
    }

    /// Wrap `self` in a single-element TUPLE. Mirrors tinygrad `UOp.maketuple(self)`.
    pub fn maketuple(self: &Arc<Self>) -> Arc<Self> {
        Self::tuple(smallvec![self.clone()])
    }

    /// Extract element `index` from a TUPLE (or a FUNCTION whose body is a TUPLE).
    /// dtype matches the inner element. Mirrors tinygrad `Ops.GETTUPLE`.
    ///
    /// # Errors
    /// - `GetTupleNotATuple` if `self` is neither a TUPLE nor a FUNCTION whose body is a TUPLE
    /// - `GetTupleIndexOutOfBounds` if `index` is out of bounds for the tuple
    pub fn try_gettuple(self: &Arc<Self>, index: usize) -> Result<Arc<Self>> {
        let inner_tuple_src: &SmallVec<[Arc<UOp>; 4]> = match self.op() {
            Op::Tuple(ops::Tuple { src }) => src,
            Op::Function(ops::Function { body, .. }) => match body.op() {
                Op::Tuple(ops::Tuple { src }) => src,
                _ => return GetTupleNotATupleSnafu { op: "FUNCTION body (expected TUPLE)" }.fail(),
            },
            _ => return GetTupleNotATupleSnafu { op: "non-TUPLE/non-FUNCTION source" }.fail(),
        };
        let elem_dtype = inner_tuple_src
            .get(index)
            .context(GetTupleIndexOutOfBoundsSnafu { index, len: inner_tuple_src.len(), kind: "tuple" })?
            .dtype();
        Ok(Self::new(Op::GetTuple(ops::GetTuple { src: self.clone(), index }), elem_dtype))
    }

    /// Extract element `index` from a TUPLE (or a FUNCTION whose body is a TUPLE).
    ///
    /// Panicking wrapper around [`Self::try_gettuple`]; use the fallible variant
    /// when the source structure or index is not guaranteed by construction.
    pub fn gettuple(self: &Arc<Self>, index: usize) -> Arc<Self> {
        self.try_gettuple(index).expect("gettuple precondition violated")
    }

    /// PROGRAM wrapper with optional progressive pipeline stages.
    pub fn program(
        sink: Arc<Self>,
        info: impl Into<Box<crate::ProgramInfo>>,
        linear: Option<Arc<Self>>,
        source: Option<Arc<Self>>,
        binary: Option<Arc<Self>>,
    ) -> Arc<Self> {
        Self::new(Op::Program(ops::Program { sink, info: info.into(), linear, source, binary }), DType::Void)
    }

    /// A PROGRAM already at its LINEAR stage: codegen renders `ops` in exactly
    /// this order, and neither the optimizer nor the linearizer ever sees it.
    ///
    /// A source the list does not name is emitted just before its first user
    /// (sources first), so constants and index math need not be listed. Using
    /// an op the list names only later is [`Error::LinearForwardReference`];
    /// repeating anything but a void `Custom`/`Store`/`Barrier` statement is
    /// [`Error::LinearRepeat`] — `rtag` a value to emit it twice. Hash-consing
    /// makes identical statements one node, which renders at every position.
    ///
    /// The SINK is derived, `SINK[info]` over the listed ops nothing else in
    /// the list consumes, so every PARAM, SPECIAL, LOAD and STORE reaches the
    /// `ProgramInfo` ABI and launch dims. PARAMs keep their authored slots (a
    /// scalar variable needs one past the buffers, and binds by name); weak
    /// PARAM shapes, as `custom_kernel` placeholders carry, are committed, which
    /// rebuilds the listed ops that read them. Index a placeholder's `base()`.
    ///
    /// [`Error::LinearForwardReference`]: crate::Error::LinearForwardReference
    /// [`Error::LinearRepeat`]: crate::Error::LinearRepeat
    pub fn linear_program(
        info: crate::KernelInfo,
        ops: impl IntoIterator<Item = Arc<Self>>,
        target: svod_dtype::DeviceSpec,
    ) -> Result<Arc<Self>> {
        use std::collections::HashSet;

        let ops = Self::commit_param_shapes(ops.into_iter().collect());
        let listed: HashSet<u64> = ops.iter().map(|op| op.id).collect();
        let mut emitted = HashSet::new();
        let mut list: Vec<Arc<Self>> = Vec::with_capacity(ops.len());
        for op in &ops {
            if emitted.contains(&op.id) {
                let statement = matches!(op.op(), Op::Custom(..) | Op::Store(..) | Op::Barrier(..));
                ensure!(
                    statement && op.dtype() == DType::Void,
                    crate::error::LinearRepeatSnafu { op: op.op().as_ref(), id: op.id }
                );
                list.push(op.clone());
                continue;
            }
            // Post-order walk of the unlisted sources `op` reaches.
            let mut stack = vec![(op.clone(), false)];
            while let Some((node, expanded)) = stack.pop() {
                if expanded {
                    emitted.insert(node.id);
                    list.push(node);
                    continue;
                }
                if emitted.contains(&node.id) {
                    continue;
                }
                stack.push((node.clone(), true));
                for source in node.op().sources().into_iter().rev() {
                    if emitted.contains(&source.id) {
                        continue;
                    }
                    ensure!(
                        !listed.contains(&source.id),
                        crate::error::LinearForwardReferenceSnafu {
                            op: node.op().as_ref(),
                            id: node.id,
                            source_op: source.op().as_ref(),
                            source_id: source.id,
                        }
                    );
                    stack.push((source, false));
                }
            }
        }

        let consumed: HashSet<u64> = list.iter().flat_map(|node| node.op().sources()).map(|source| source.id).collect();
        let mut seen = HashSet::new();
        let roots = list.iter().filter(|node| !consumed.contains(&node.id) && seen.insert(node.id)).cloned().collect();
        let sink = Self::sink_with_info(roots, info);
        list.push(sink.clone());
        let program_info = crate::ProgramInfo::from_sink(&sink, target);
        Ok(Self::program(sink, program_info, Some(Self::linear(list.into())), None, None))
    }

    fn commit_shape(
        shape: &Arc<Self>,
        commit_const: &impl Fn(&Arc<Self>) -> Option<(crate::UOpKey, Arc<Self>)>,
    ) -> Arc<Self> {
        let consts: std::collections::HashMap<_, _> = shape.toposort().iter().filter_map(commit_const).collect();
        shape.substitute(&consts)
    }

    /// PARAM and BUFFER shapes are metadata a program never renders, but
    /// `custom_kernel` placeholders and hand-built buffers carry them as weak
    /// constants, which no program admits; the optimizer's index-dtype
    /// lowering would commit them, so commit them here.
    fn commit_param_shapes(ops: Vec<Arc<Self>>) -> Vec<Arc<Self>> {
        use std::collections::HashMap;

        use crate::UOpKey;

        let root = Self::sink(ops.clone());
        let commit_const = |node: &Arc<Self>| match node.op() {
            Op::Const(value) if node.dtype() == DType::WeakInt => {
                let dtype = match value.0.try_int() {
                    Some(v) if i32::try_from(v).is_err() => DType::Int64,
                    _ => DType::Int32,
                };
                Some((UOpKey(node.clone()), Self::const_(dtype, value.0)))
            }
            _ => None,
        };
        let params: HashMap<UOpKey, Arc<Self>> = root
            .toposort()
            .into_iter()
            .filter_map(|node| {
                let committed = match node.op() {
                    Op::Param(ops::Param { shape, arg }) => {
                        let shape = Self::commit_shape(shape, &commit_const);
                        Self::new(Op::Param(ops::Param { shape, arg: arg.clone() }), node.dtype())
                    }
                    Op::Buffer(ops::Buffer { shape, arg }) => {
                        let shape = Self::commit_shape(shape, &commit_const);
                        Self::new(Op::Buffer(ops::Buffer { shape, arg: arg.clone() }), node.dtype())
                    }
                    _ => return None,
                };
                (!Arc::ptr_eq(&committed, &node)).then(|| (UOpKey(node.clone()), committed.rtag(node.tag().clone())))
            })
            .collect();
        if params.is_empty() {
            return ops;
        }
        match root.substitute(&params).op() {
            Op::Sink(ops::Sink { sources, .. }) => sources.to_vec(),
            _ => unreachable!("substitution keeps the SINK root"),
        }
    }

    /// LINEAR stage payload.
    pub fn linear(ops: SmallVec<[Arc<Self>; 8]>) -> Arc<Self> {
        Self::new(Op::Linear(ops::Linear { ops }), DType::Void)
    }

    /// SOURCE stage payload.
    pub fn source(code: String) -> Arc<Self> {
        Self::new(Op::Source(ops::Source { code, identity: None }), DType::Void)
    }

    /// SOURCE stage payload bound to an executable PROGRAM identity.
    pub fn source_with_identity(code: String, identity: crate::SourceStageIdentity) -> Arc<Self> {
        Self::new(Op::Source(ops::Source { code, identity: Some(identity.into()) }), DType::Void)
    }

    /// BINARY stage payload.
    pub fn binary(bytes: Vec<u8>) -> Arc<Self> {
        Self::new(Op::ProgramBinary(ops::ProgramBinary { bytes, identity: None }), DType::UInt8)
    }

    /// BINARY stage payload bound to its exact SOURCE and compiler identity.
    pub fn binary_with_identity(bytes: Vec<u8>, identity: crate::BinaryStageIdentity) -> Arc<Self> {
        Self::new(Op::ProgramBinary(ops::ProgramBinary { bytes, identity: Some(identity.into()) }), DType::UInt8)
    }

    /// Construct a target instruction. INS has no inferred dtype because an
    /// instruction may define a value of any target type or be void.
    pub fn ins(sources: impl IntoIterator<Item = Arc<Self>>, dtype: DType, arg: crate::InsArg) -> Arc<Self> {
        Self::new(Op::Ins(ops::Ins { sources: sources.into_iter().collect(), arg }), dtype)
    }
}
