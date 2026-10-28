//! Reversible, allocation-independent wire format for optimizer input DAGs.
//!
//! Unlike the parity-oriented canonical format, this representation preserves
//! every semantic field required to reconstruct a pre-PROGRAM kernel exactly.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use svod_dtype::{DType, DeviceSpec};

use crate::ops;
use crate::{
    AxisId, AxisType, BarrierFence, BinaryOp, BufferizeOpts, CallInfo, ConstValue, ConstValueHash, ContiguousHint,
    CustomFunctionKind, InsArg, KernelInfo, Op, ParamArg, ReduceOp, TernaryOp, UOp, UnaryOp, WmmaMetadata,
};

pub const OPTIMIZER_WIRE_SCHEMA_VERSION: u32 = 4;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerWireGraph {
    pub schema_version: u32,
    pub roots: Vec<usize>,
    pub nodes: Vec<OptimizerWireNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerWireNode {
    pub dtype: DType,
    pub tag: Option<SmallVec<[usize; 2]>>,
    /// Raw [`crate::origin::OriginId`]. Ids are process-local, so a decoded graph
    /// reproduces the encoder's content hashes but resolves paths only in-process.
    pub origin: Option<u32>,
    pub content_hash: u64,
    pub op: OptimizerWireOp,
    pub src: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerWireOp {
    Const(ConstValueHash),
    Unique(usize),
    LUnique(usize),
    Noop,
    Sink(Option<KernelInfo>),
    Group,
    Unary(UnaryOp),
    Binary(BinaryOp),
    Ternary(TernaryOp),
    Cast(DType),
    BitCast(DType),
    MSelect(usize),
    Special(String),
    Param(ParamArg),
    Buffer(ParamArg),
    Slice(usize),
    Stage(BufferizeOpts),
    Index,
    GetAddr(DeviceSpec),
    Copy(DeviceSpec),
    MStack,
    Reshape,
    Permute(Vec<usize>),
    Expand,
    Pad,
    Shrink,
    Flip(Vec<bool>),
    Multi(usize),
    ReduceAxis(ReduceOp, Vec<usize>),
    Reduce(ReduceOp, usize),
    AllReduce(DeviceSpec, ReduceOp),
    If,
    EndIf,
    Range(AxisId, AxisType),
    End,
    Barrier(BarrierFence),
    Stack,
    VConst(Vec<ConstValue>),
    DefineVar(String, i64, i64),
    Bind,
    Wmma(WmmaMetadata),
    Call(CallInfo),
    Function(CallInfo),
    Tuple,
    GetTuple(usize),
    Detach,
    Contiguous(Vec<ContiguousHint>),
    ContiguousBackward,
    After,
    Precast,
    Custom(String),
    CustomFunction(CustomFunctionKind),
    CustomI(String),
    Load { gated: bool },
    Store { gated: bool },
    Ins(InsArg),
}

fn wire_error(detail: impl Into<String>) -> crate::Error {
    crate::Error::CanonicalSerialization { detail: detail.into() }
}

impl OptimizerWireGraph {
    pub fn from_root(root: &Arc<UOp>) -> crate::Result<Self> {
        Self::from_roots(std::slice::from_ref(root))
    }

    pub fn from_roots(roots: &[Arc<UOp>]) -> crate::Result<Self> {
        let mut visited = HashSet::new();
        let mut active = HashSet::new();
        let mut topo = Vec::new();
        let mut stack: Vec<_> = roots.iter().rev().cloned().map(|root| (root, false)).collect();
        while let Some((node, expanded)) = stack.pop() {
            if expanded {
                active.remove(&node.id);
                if visited.insert(node.id) {
                    topo.push(node);
                }
                continue;
            }
            if visited.contains(&node.id) {
                continue;
            }
            if !active.insert(node.id) {
                return Err(wire_error("optimizer input contains a cycle"));
            }
            if node.metadata_raw().is_some() {
                return Err(wire_error(format!(
                    "optimizer input {:?} carries type-erased metadata that cannot be reconstructed",
                    node.op()
                )));
            }
            if matches!(node.op(), Op::Program(..) | Op::Linear(..) | Op::Source(..) | Op::ProgramBinary(..)) {
                return Err(wire_error(format!("executable-stage op {:?} is not a legal optimizer input", node.op())));
            }
            stack.push((node.clone(), true));
            for child in node.op().children().into_iter().rev() {
                stack.push((child.clone(), false));
            }
        }

        let ids: HashMap<u64, usize> = topo.iter().enumerate().map(|(id, node)| (node.id, id)).collect();
        let nodes = topo
            .iter()
            .map(|node| {
                Ok(OptimizerWireNode {
                    dtype: node.dtype(),
                    tag: node.tag().clone(),
                    origin: node.origin().map(crate::origin::OriginId::get),
                    content_hash: node.content_hash,
                    op: OptimizerWireOp::from_op(node.op())?,
                    src: node.op().children().iter().map(|child| ids[&child.id]).collect(),
                })
            })
            .collect::<crate::Result<_>>()?;
        Ok(Self {
            schema_version: OPTIMIZER_WIRE_SCHEMA_VERSION,
            roots: roots.iter().map(|root| ids[&root.id]).collect(),
            nodes,
        })
    }

    pub fn decode(&self) -> crate::Result<Vec<Arc<UOp>>> {
        if self.schema_version != OPTIMIZER_WIRE_SCHEMA_VERSION {
            return Err(wire_error(format!(
                "unsupported optimizer wire schema {}, expected {}",
                self.schema_version, OPTIMIZER_WIRE_SCHEMA_VERSION
            )));
        }
        let mut decoded: Vec<Arc<UOp>> = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            let mut src = Vec::with_capacity(node.src.len());
            for &source in &node.src {
                if source >= index {
                    return Err(wire_error(format!("node {index} source {source} is not dependency-first")));
                }
                src.push(decoded[source].clone());
            }
            let origin = node.origin.and_then(crate::origin::OriginId::from_raw);
            let rebuilt = UOp::new_with_origin(node.op.to_op(src)?, node.dtype.clone(), node.tag.clone(), origin);
            if rebuilt.content_hash != node.content_hash {
                return Err(wire_error(format!(
                    "node {index} content hash changed during reconstruction: encoded={}, decoded={}",
                    node.content_hash, rebuilt.content_hash
                )));
            }
            decoded.push(rebuilt);
        }
        self.roots
            .iter()
            .map(|&root| decoded.get(root).cloned().ok_or_else(|| wire_error(format!("invalid root {root}"))))
            .collect()
    }

    pub fn decode_root(&self) -> crate::Result<Arc<UOp>> {
        let roots = self.decode()?;
        if roots.len() != 1 {
            return Err(wire_error(format!("expected one optimizer root, got {}", roots.len())));
        }
        Ok(roots.into_iter().next().unwrap())
    }
}

impl OptimizerWireOp {
    fn from_op(op: &Op) -> crate::Result<Self> {
        Ok(match op {
            Op::Const(value) => Self::Const(*value),
            Op::Unique(value) => Self::Unique(*value),
            Op::LUnique(value) => Self::LUnique(*value),
            Op::Noop => Self::Noop,
            Op::Sink(ops::Sink { info, .. }) => Self::Sink(info.as_deref().cloned()),
            Op::Group(..) => Self::Group,
            Op::Unary(op, ..) => Self::Unary(*op),
            Op::Binary(op, ..) => Self::Binary(*op),
            Op::Ternary(op, ..) => Self::Ternary(*op),
            Op::Cast(ops::Cast { dtype, .. }) => Self::Cast(dtype.clone()),
            Op::BitCast(ops::BitCast { dtype, .. }) => Self::BitCast(dtype.clone()),
            Op::MSelect(ops::MSelect { device_index, .. }) => Self::MSelect(*device_index),
            Op::Special(ops::Special { name, .. }) => Self::Special(name.clone()),
            Op::Param(ops::Param { arg, .. }) => Self::Param(arg.as_ref().clone()),
            Op::Buffer(ops::Buffer { arg, .. }) => Self::Buffer(arg.as_ref().clone()),
            Op::Slice(ops::Slice { size, .. }) => Self::Slice(*size),
            Op::Stage(ops::Stage { opts, .. }) => Self::Stage(opts.as_ref().clone()),
            Op::Index(..) => Self::Index,
            Op::GetAddr(ops::GetAddr { device, .. }) => Self::GetAddr(device.clone()),
            Op::Copy(ops::Copy { device, .. }) => Self::Copy(device.clone()),
            Op::MStack(..) => Self::MStack,
            Op::Reshape(..) => Self::Reshape,
            Op::Permute(ops::Permute { axes, .. }) => Self::Permute(axes.clone()),
            Op::Expand(..) => Self::Expand,
            Op::Pad(..) => Self::Pad,
            Op::Shrink(..) => Self::Shrink,
            Op::Flip(ops::Flip { axes, .. }) => Self::Flip(axes.clone()),
            Op::Multi(ops::Multi { axis, .. }) => Self::Multi(*axis),
            Op::ReduceAxis(ops::ReduceAxis { reduce_op, axes, .. }) => Self::ReduceAxis(*reduce_op, axes.clone()),
            Op::Reduce(ops::Reduce { reduce_op, num_axes, .. }) => Self::Reduce(*reduce_op, *num_axes),
            Op::AllReduce(ops::AllReduce { device, reduce_op, .. }) => Self::AllReduce(device.clone(), *reduce_op),
            Op::If(..) => Self::If,
            Op::EndIf(..) => Self::EndIf,
            Op::Range(ops::Range { axis_id, axis_type, .. }) => Self::Range(axis_id.clone(), *axis_type),
            Op::End(..) => Self::End,
            Op::Barrier(ops::Barrier { fence, .. }) => Self::Barrier(*fence),
            Op::Stack(..) => Self::Stack,
            Op::Shuffle(..) => Self::Stack,
            Op::VConst(ops::VConst { values }) => Self::VConst(values.clone()),
            Op::DefineVar(ops::DefineVar { name, min_val, max_val }) => {
                Self::DefineVar(name.clone(), *min_val, *max_val)
            }
            Op::Bind(..) => Self::Bind,
            Op::Wmma(ops::Wmma { metadata, .. }) => Self::Wmma(metadata.as_ref().clone()),
            Op::Call(ops::Call { info, .. }) => Self::Call(info.as_ref().clone()),
            Op::Function(ops::Function { info, .. }) => Self::Function(info.as_ref().clone()),
            Op::Tuple(..) => Self::Tuple,
            Op::GetTuple(ops::GetTuple { index, .. }) => Self::GetTuple(*index),
            Op::Detach(..) => Self::Detach,
            Op::Contiguous(ops::Contiguous { opts, .. }) => Self::Contiguous(opts.to_vec()),
            Op::ContiguousBackward(..) => Self::ContiguousBackward,
            Op::After(..) => Self::After,
            Op::Precast(..) => Self::Precast,
            Op::Custom(ops::Custom { code, .. }) => Self::Custom(code.clone()),
            Op::CustomFunction(ops::CustomFunction { kind, .. }) => Self::CustomFunction(kind.clone()),
            Op::CustomI(ops::CustomI { code, .. }) => Self::CustomI(code.clone()),
            Op::Load(ops::Load { alt, gate, .. }) => {
                if alt.is_some() != gate.is_some() {
                    return Err(wire_error("LOAD has mismatched alt/gate presence"));
                }
                Self::Load { gated: alt.is_some() }
            }
            Op::Store(ops::Store { gate, .. }) => Self::Store { gated: gate.is_some() },
            Op::Ins(ops::Ins { arg, .. }) => Self::Ins(arg.clone()),
            Op::Program(..) | Op::Linear(..) | Op::Source(..) | Op::ProgramBinary(..) => {
                return Err(wire_error(format!("executable-stage op {op:?} is not a legal optimizer input")));
            }
        })
    }

    fn to_op(&self, src: Vec<Arc<UOp>>) -> crate::Result<Op> {
        let exact = |count: usize| {
            if src.len() == count {
                Ok(())
            } else {
                Err(wire_error(format!("wire op expects {count} sources, got {}", src.len())))
            }
        };
        let nonempty = || if src.is_empty() { Err(wire_error("wire op requires at least one source")) } else { Ok(()) };
        let one = || src[0].clone();
        Ok(match self {
            Self::Const(v) => {
                exact(0)?;
                Op::Const(*v)
            }
            Self::Unique(v) => {
                exact(0)?;
                Op::Unique(*v)
            }
            Self::LUnique(v) => {
                exact(0)?;
                Op::LUnique(*v)
            }
            Self::Noop => {
                exact(0)?;
                Op::Noop
            }
            Self::Sink(info) => {
                Op::Sink(ops::Sink { sources: src.into_iter().collect(), info: info.clone().map(Box::new) })
            }
            Self::Group => Op::Group(ops::Group { sources: src.into_iter().collect() }),
            Self::Unary(op) => {
                exact(1)?;
                Op::Unary(*op, one())
            }
            Self::Binary(op) => {
                exact(2)?;
                Op::Binary(*op, src[0].clone(), src[1].clone())
            }
            Self::Ternary(op) => {
                exact(3)?;
                Op::Ternary(*op, src[0].clone(), src[1].clone(), src[2].clone())
            }
            Self::Cast(dtype) => {
                exact(1)?;
                Op::Cast(ops::Cast { src: one(), dtype: dtype.clone() })
            }
            Self::BitCast(dtype) => {
                exact(1)?;
                Op::BitCast(ops::BitCast { src: one(), dtype: dtype.clone() })
            }
            Self::MSelect(index) => {
                exact(1)?;
                Op::MSelect(ops::MSelect { buffer: one(), device_index: *index })
            }
            Self::Special(name) => {
                exact(1)?;
                Op::Special(ops::Special { end: one(), name: name.clone() })
            }
            Self::Param(arg) => {
                exact(1)?;
                Op::Param(ops::Param { shape: one(), arg: arg.clone().into() })
            }
            Self::Buffer(arg) => {
                exact(1)?;
                Op::Buffer(ops::Buffer { shape: one(), arg: arg.clone().into() })
            }
            Self::Slice(size) => {
                exact(2)?;
                Op::Slice(ops::Slice { buffer: src[0].clone(), offset: src[1].clone(), size: *size })
            }
            Self::Stage(opts) => {
                nonempty()?;
                Op::Stage(ops::Stage {
                    compute: one(),
                    ranges: src[1..].iter().cloned().collect(),
                    opts: opts.clone().into(),
                })
            }
            Self::Index => {
                nonempty()?;
                Op::Index(ops::Index { buffer: one(), indices: src[1..].iter().cloned().collect() })
            }
            Self::GetAddr(device) => {
                exact(1)?;
                Op::GetAddr(ops::GetAddr { src: one(), device: device.clone() })
            }
            Self::Copy(device) => {
                exact(1)?;
                Op::Copy(ops::Copy { src: one(), device: device.clone() })
            }
            Self::MStack => Op::MStack(ops::MStack { buffers: src.into_iter().collect() }),
            Self::Reshape => {
                exact(2)?;
                Op::Reshape(ops::Reshape { src: src[0].clone(), new_shape: src[1].clone() })
            }
            Self::Permute(axes) => {
                exact(1)?;
                Op::Permute(ops::Permute { src: one(), axes: axes.clone() })
            }
            Self::Expand => {
                exact(2)?;
                Op::Expand(ops::Expand { src: src[0].clone(), new_shape: src[1].clone() })
            }
            Self::Pad => {
                exact(3)?;
                Op::Pad(ops::Pad { src: src[0].clone(), begin_pads: src[1].clone(), end_pads: src[2].clone() })
            }
            Self::Shrink => {
                exact(3)?;
                Op::Shrink(ops::Shrink { src: src[0].clone(), offsets: src[1].clone(), sizes: src[2].clone() })
            }
            Self::Flip(axes) => {
                exact(1)?;
                Op::Flip(ops::Flip { src: one(), axes: axes.clone() })
            }
            Self::Multi(axis) => {
                exact(1)?;
                Op::Multi(ops::Multi { src: one(), axis: *axis })
            }
            Self::ReduceAxis(op, axes) => {
                exact(1)?;
                Op::ReduceAxis(ops::ReduceAxis { src: one(), reduce_op: *op, axes: axes.clone() })
            }
            Self::Reduce(op, num_axes) => {
                nonempty()?;
                Op::Reduce(ops::Reduce {
                    src: one(),
                    ranges: src[1..].iter().cloned().collect(),
                    reduce_op: *op,
                    num_axes: *num_axes,
                })
            }
            Self::AllReduce(device, op) => {
                exact(1)?;
                Op::AllReduce(ops::AllReduce { src: one(), device: device.clone(), reduce_op: *op })
            }
            Self::If => {
                nonempty()?;
                Op::If(ops::If { condition: one(), body: src[1..].iter().cloned().collect() })
            }
            Self::EndIf => {
                exact(1)?;
                Op::EndIf(ops::EndIf { if_op: one() })
            }
            Self::Range(axis, ty) => {
                nonempty()?;
                Op::Range(ops::Range {
                    end: one(),
                    axis_id: axis.clone(),
                    axis_type: *ty,
                    deps: src[1..].iter().cloned().collect(),
                })
            }
            Self::End => {
                nonempty()?;
                Op::End(ops::End { computation: one(), ranges: src[1..].iter().cloned().collect() })
            }
            Self::Barrier(fence) => {
                nonempty()?;
                Op::Barrier(ops::Barrier { src: one(), deps: src[1..].iter().cloned().collect(), fence: *fence })
            }
            Self::Stack => Op::Stack(ops::Stack { sources: src.into_iter().collect() }),
            Self::VConst(values) => {
                exact(0)?;
                Op::VConst(ops::VConst { values: values.clone() })
            }
            Self::DefineVar(name, min, max) => {
                exact(0)?;
                Op::DefineVar(ops::DefineVar { name: name.clone(), min_val: *min, max_val: *max })
            }
            Self::Bind => {
                exact(2)?;
                Op::Bind(ops::Bind { var: src[0].clone(), value: src[1].clone() })
            }
            Self::Wmma(metadata) => {
                exact(3)?;
                Op::Wmma(ops::Wmma {
                    a: src[0].clone(),
                    b: src[1].clone(),
                    c: src[2].clone(),
                    metadata: metadata.clone().into(),
                })
            }
            Self::Call(info) => {
                nonempty()?;
                Op::Call(ops::Call { body: one(), args: src[1..].iter().cloned().collect(), info: info.clone().into() })
            }
            Self::Function(info) => {
                nonempty()?;
                Op::Function(ops::Function {
                    body: one(),
                    args: src[1..].iter().cloned().collect(),
                    info: info.clone().into(),
                })
            }
            Self::Tuple => Op::Tuple(ops::Tuple { src: src.into_iter().collect() }),
            Self::GetTuple(index) => {
                exact(1)?;
                Op::GetTuple(ops::GetTuple { src: one(), index: *index })
            }
            Self::Detach => {
                exact(1)?;
                Op::Detach(ops::Detach { src: one() })
            }
            Self::Contiguous(opts) => {
                exact(1)?;
                Op::Contiguous(ops::Contiguous { src: one(), opts: opts.to_vec() })
            }
            Self::ContiguousBackward => {
                exact(1)?;
                Op::ContiguousBackward(ops::ContiguousBackward { src: one() })
            }
            Self::After => {
                nonempty()?;
                Op::After(ops::After { passthrough: one(), deps: src[1..].iter().cloned().collect() })
            }
            Self::Precast => {
                exact(1)?;
                Op::Precast(ops::Precast { src: one() })
            }
            Self::Custom(code) => Op::Custom(ops::Custom { deps: src.into_iter().collect(), code: code.clone() }),
            Self::CustomFunction(kind) => {
                Op::CustomFunction(ops::CustomFunction { kind: kind.clone(), attrs: src.into_iter().collect() })
            }
            Self::CustomI(code) => Op::CustomI(ops::CustomI { deps: src.into_iter().collect(), code: code.clone() }),
            Self::Load { gated } => {
                exact(if *gated { 3 } else { 1 })?;
                Op::Load(ops::Load {
                    index: one(),
                    alt: gated.then(|| src[1].clone()),
                    gate: gated.then(|| src[2].clone()),
                })
            }
            Self::Store { gated } => {
                exact(if *gated { 3 } else { 2 })?;
                Op::Store(ops::Store {
                    index: src[0].clone(),
                    value: src[1].clone(),
                    gate: gated.then(|| src[2].clone()),
                })
            }
            Self::Ins(arg) => Op::Ins(ops::Ins { sources: src.into_iter().collect(), arg: arg.clone() }),
        })
    }
}
