//! Stable, allocation-independent serialization for UOp DAGs.
//!
//! Directly serializing [`UOp`] would encode recursive `Arc`s, duplicate shared
//! nodes, and expose runtime IDs and caches. This module instead emits a
//! dependency-first node table with graph-local source IDs.

use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

use serde::Serialize;
use svod_dtype::{AddrSpace, DType, ImageKind, ScalarDType};

use crate::ops;
use crate::{AxisId, BarrierFence, BinaryOp, ConstValue, Op, SInt, TernaryOp, UOp};

/// Version of the canonical graph schema.
pub const CANONICAL_SCHEMA_VERSION: u32 = 7;

/// Canonical graph representation used by cross-implementation parity tests.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CanonicalGraph {
    pub schema_version: u32,
    pub stage: String,
    pub roots: Vec<usize>,
    pub nodes: Vec<CanonicalNode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbose: Option<Vec<CanonicalVerboseNode>>,
}

/// Language-specific diagnostics. This is intentionally excluded from the
/// default parity oracle and may contain allocation-dependent identities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CanonicalVerboseNode {
    pub id: usize,
    pub runtime_id: u64,
    pub tag: String,
    pub backend_dtype: String,
    /// Rendered origin path, absent when the node carries no origin.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_xxh64: Option<String>,
}

/// One UOp in a canonical dependency-first node table.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CanonicalNode {
    pub id: usize,
    pub op: String,
    pub dtype: CanonicalDType,
    pub shape: Option<Vec<CanonicalShapeDim>>,
    pub arg: CanonicalArg,
    pub src: Vec<usize>,
}

/// Language-neutral dtype representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CanonicalDType {
    Scalar { name: String },
    Vector { scalar: String, count: usize },
    Pointer { base: Box<CanonicalDType>, address_space: String, size: Option<usize>, count: usize },
    Image { image_kind: String, shape: Vec<usize> },
}

/// Stable shape dimension. Symbolic dimensions refer to a graph-local node ID
/// when that expression is part of the serialized DAG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CanonicalShapeDim {
    Const { value: usize },
    Symbolic { node: usize },
    Infer,
}

/// Constants use float bit patterns so NaN, infinities, and signed zero remain
/// deterministic and valid JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CanonicalConst {
    Invalid,
    Int { value: i64 },
    UInt { value: u64 },
    Float { bits: String },
    Bool { value: bool },
}

/// Operation metadata with all UOp sources removed. Sources are represented by
/// [`CanonicalNode::src`], keeping the Serde schema acyclic.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CanonicalArg {
    None,
    Const {
        value: CanonicalConst,
    },
    Device {
        name: String,
    },
    Sink {
        name: Option<String>,
        opts_to_apply: Option<Vec<crate::Opt>>,
        applied_opts: Vec<crate::Opt>,
        dont_use_locals: bool,
    },
    DType {
        value: CanonicalDType,
    },
    Index {
        value: usize,
    },
    Name {
        value: String,
    },
    Param {
        slot: i128,
        dtype: CanonicalDType,
        vmin_vmax: Option<(CanonicalConst, CanonicalConst)>,
        multiple_of: Option<usize>,
        name: Option<String>,
        address_space: Option<String>,
        axis: Option<usize>,
        device: Option<String>,
        volatile: bool,
    },
    Size {
        value: usize,
    },
    Stage {
        device: Option<String>,
        local_axis: Option<CanonicalAxis>,
        address_space: String,
        removable: bool,
    },
    Axes {
        values: Vec<usize>,
    },
    BoolAxes {
        values: Vec<bool>,
    },
    Pad {
        begin: Vec<usize>,
        end: Vec<usize>,
    },
    Reduce {
        op: String,
        axes: Option<Vec<usize>>,
        num_axes: Option<usize>,
    },
    AllReduce {
        op: String,
        device: String,
    },
    Range {
        axis: Vec<usize>,
        renumbered: bool,
        axis_type: String,
    },
    Constants {
        values: Vec<CanonicalConst>,
    },
    DefineVar {
        name: String,
        min: i64,
        max: i64,
    },
    Call {
        grad_tag: Option<String>,
        metadata: Vec<String>,
        name: Option<String>,
        precompile: bool,
        precompile_backward: bool,
    },
    Wmma {
        dims: (usize, usize, usize),
        dtype_in: CanonicalDType,
        dtype_out: CanonicalDType,
        device: String,
        threads: usize,
        upcast_a: Vec<CanonicalAxisExtent>,
        upcast_b: Vec<CanonicalAxisExtent>,
        upcast_c: Vec<CanonicalAxisExtent>,
    },
    Source {
        code: String,
    },
    Binary {
        length: usize,
    },
    Ins {
        opcode: String,
        attributes: Vec<(String, String)>,
    },
    Hints {
        values: Vec<crate::ContiguousHint>,
    },
    Code {
        value: String,
    },
    CustomFunction {
        kind_name: String,
    },
    Program {
        name: String,
        global_size: Vec<CanonicalProgramValue>,
        local_size: Option<Vec<CanonicalProgramValue>>,
        vars: Vec<usize>,
        globals: Vec<usize>,
        outs: Vec<usize>,
        ins: Vec<usize>,
        target: String,
    },
}

/// Launch dimensions are integers in Tinygrad but UOps in Svod. Constants are
/// normalized to values; symbolic dimensions refer to the existing node table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CanonicalProgramValue {
    Int { value: i64 },
    UInt { value: u64 },
    Float { bits: String },
    Node { node: usize },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CanonicalAxis {
    pub path: Vec<usize>,
    pub renumbered: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CanonicalAxisExtent {
    pub axis: CanonicalAxis,
    pub extent: usize,
}

fn canonical_axis(axis: &AxisId) -> CanonicalAxis {
    CanonicalAxis { path: axis.path().to_vec(), renumbered: axis.is_renumbered() }
}

fn canonical_axis_extents(axes: &[(AxisId, usize)]) -> Vec<CanonicalAxisExtent> {
    axes.iter().map(|(axis, extent)| CanonicalAxisExtent { axis: canonical_axis(axis), extent: *extent }).collect()
}

impl CanonicalGraph {
    /// Serialize one root and all of its call/program bodies.
    pub fn from_root(stage: impl Into<String>, root: &Arc<UOp>) -> crate::Result<Self> {
        Self::from_roots(stage, std::slice::from_ref(root))
    }

    /// Serialize one root with allocation/provenance diagnostics. Verbose
    /// documents must not be used as parity oracles.
    pub fn from_root_verbose(stage: impl Into<String>, root: &Arc<UOp>) -> crate::Result<Self> {
        Self::from_roots_impl(stage, std::slice::from_ref(root), true)
    }

    /// Serialize multiple ordered roots into one deduplicated node table.
    pub fn from_roots(stage: impl Into<String>, roots: &[Arc<UOp>]) -> crate::Result<Self> {
        Self::from_roots_impl(stage, roots, false)
    }

    fn from_roots_impl(stage: impl Into<String>, roots: &[Arc<UOp>], verbose: bool) -> crate::Result<Self> {
        let stage = stage.into();
        let topo = canonical_toposort(roots)?;

        let ids: FxHashMap<u64, usize> = topo.iter().enumerate().map(|(id, node)| (node.id, id)).collect();
        let nodes = topo
            .iter()
            .enumerate()
            .map(|(id, node)| canonical_node(id, node, &ids, verbose))
            .collect::<crate::Result<_>>()?;
        let roots = roots.iter().map(|root| ids[&root.id]).collect();

        let verbose = verbose.then(|| {
            topo.iter()
                .enumerate()
                .map(|(id, node)| CanonicalVerboseNode {
                    id,
                    runtime_id: node.id,
                    tag: format!("{:?}", node.tag()),
                    backend_dtype: format!("{:?}", node.dtype()),
                    origin: node.origin().map(crate::origin::path),
                    content_xxh64: match node.op() {
                        Op::ProgramBinary(ops::ProgramBinary { bytes, .. }) => {
                            Some(format!("0x{:016x}", xxhash_rust::xxh64::xxh64(bytes, 0)))
                        }
                        _ => None,
                    },
                })
                .collect()
        });

        Ok(Self { schema_version: CANONICAL_SCHEMA_VERSION, stage, roots, nodes, verbose })
    }

    /// Render canonical JSON with deterministic field and node ordering.
    pub fn to_pretty_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Stream the same fields as [`Self::to_pretty_json`] into `writer` as
    /// fixed-width little-endian bincode (the 1.x wire format, so identity
    /// digests stay stable). Identity digests hash this directly instead of
    /// materializing text.
    pub fn encode_into(&self, mut writer: impl std::io::Write) -> Result<(), bincode::error::EncodeError> {
        bincode::serde::encode_into_std_write(self, &mut writer, bincode::config::legacy()).map(drop)
    }
}

fn canonical_dependencies(node: &Arc<UOp>) -> crate::Result<Vec<Arc<UOp>>> {
    let mut dependencies = canonical_sources(node.op());
    if let Some(shape) = node.shape()? {
        dependencies.extend(shape.iter().filter_map(|dim| match dim {
            SInt::Symbolic(expr) => Some(expr.clone()),
            SInt::Const(_) | SInt::Infer => None,
        }));
    }
    if let Op::Program(ops::Program { info, .. }) = node.op() {
        dependencies.extend(info.global_size.iter().filter(|value| !matches!(value.op(), Op::Const(_))).cloned());
        dependencies
            .extend(info.local_size.iter().flatten().filter(|value| !matches!(value.op(), Op::Const(_))).cloned());
        dependencies.extend(info.vars.iter().cloned());
    }
    Ok(dependencies)
}

fn canonical_sources(op: &Op) -> Vec<Arc<UOp>> {
    match op {
        // PAD's other two UOps are representation-specific metadata: Svod
        // stores begin/end padding while Tinygrad stores begin/output extent.
        // The logical values are retained in CanonicalArg::Pad below.
        Op::Pad(ops::Pad { src, .. }) => vec![src.clone()],
        _ => op.sources().into_vec(),
    }
}

fn canonical_toposort(roots: &[Arc<UOp>]) -> crate::Result<Vec<Arc<UOp>>> {
    let mut visited = FxHashSet::default();
    let mut active = FxHashSet::default();
    let mut result = Vec::new();
    let mut stack: Vec<_> = roots.iter().rev().cloned().map(|root| (root, false)).collect();

    while let Some((node, processed)) = stack.pop() {
        if visited.contains(&node.id) {
            continue;
        }
        if processed {
            active.remove(&node.id);
            visited.insert(node.id);
            result.push(node);
            continue;
        }
        if !active.insert(node.id) {
            return Err(crate::Error::CanonicalSerialization {
                detail: "cycle through shape or PROGRAM metadata".into(),
            });
        }
        stack.push((node.clone(), true));
        for dependency in canonical_dependencies(&node)?.into_iter().rev() {
            if !visited.contains(&dependency.id) {
                stack.push((dependency, false));
            }
        }
    }
    Ok(result)
}

/// Emit one canonical graph to stderr when its stage matches
/// `SVOD_DUMP_CANONICAL_STAGE` by prefix.
pub fn dump_canonical_stage(stage: &str, root: &Arc<UOp>) {
    capture_canonical_stage(stage, root);

    let Ok(prefix) = std::env::var("SVOD_DUMP_CANONICAL_STAGE") else {
        return;
    };
    if !stage.starts_with(&prefix) {
        return;
    }

    eprintln!("[dump-canonical] {stage} :");
    match CanonicalGraph::from_root(stage, root) {
        Ok(graph) => match graph.to_pretty_json() {
            Ok(json) => eprintln!("{json}"),
            Err(error) => eprintln!("[dump-canonical] {stage} : JSON error: {error}"),
        },
        Err(error) => eprintln!("[dump-canonical] {stage} : graph error: {error}"),
    }
    eprintln!("[dump-canonical] {stage} : end");
}

/// Strict canonical capture used by parity runners. Unlike the optional
/// diagnostic dump above, a requested capture must either be written or abort.
fn capture_canonical_stage(stage: &str, root: &Arc<UOp>) {
    let Ok(requested) = std::env::var("SVOD_CAPTURE_CANONICAL_STAGE") else {
        return;
    };
    if requested != stage {
        return;
    }

    let path = std::env::var("SVOD_CAPTURE_CANONICAL_PATH")
        .expect("SVOD_CAPTURE_CANONICAL_PATH is required when canonical evidence capture is requested");
    let label = std::env::var("SVOD_CAPTURE_CANONICAL_LABEL").unwrap_or_else(|_| stage.to_string());
    let graph = CanonicalGraph::from_root(label, root)
        .unwrap_or_else(|error| panic!("canonical evidence capture for {stage} failed: {error}"));
    let json =
        graph.to_pretty_json().unwrap_or_else(|error| panic!("canonical evidence JSON for {stage} failed: {error}"));
    std::fs::write(&path, format!("{json}\n"))
        .unwrap_or_else(|error| panic!("writing canonical evidence for {stage} to {path} failed: {error}"));
}

fn canonical_node(
    id: usize,
    node: &Arc<UOp>,
    ids: &FxHashMap<u64, usize>,
    verbose: bool,
) -> crate::Result<CanonicalNode> {
    let shape = node
        .shape()?
        .map(|shape| -> crate::Result<Vec<_>> {
            shape
                .iter()
                .map(|dim| {
                    Ok(match dim {
                        SInt::Const(value) => CanonicalShapeDim::Const { value: *value },
                        SInt::Symbolic(expr) => CanonicalShapeDim::Symbolic {
                            node: ids.get(&expr.id).copied().ok_or_else(|| crate::Error::CanonicalSerialization {
                                detail: "symbolic shape expression is absent from canonical topology".into(),
                            })?,
                        },
                        SInt::Infer => CanonicalShapeDim::Infer,
                    })
                })
                .collect()
        })
        .transpose()?;
    let src = canonical_sources(node.op()).into_iter().map(|source| ids[&source.id]).collect();

    Ok(CanonicalNode {
        id,
        op: canonical_op_name(node.op()),
        dtype: canonical_dtype(&node.dtype()),
        shape,
        arg: canonical_arg(node, ids, verbose)?,
        src,
    })
}

fn scalar_name(dtype: ScalarDType) -> String {
    match dtype {
        ScalarDType::Bool => "bool",
        ScalarDType::WeakInt => "weakint",
        ScalarDType::Int8 => "int8",
        ScalarDType::UInt8 => "uint8",
        ScalarDType::Int16 => "int16",
        ScalarDType::UInt16 => "uint16",
        ScalarDType::Int32 => "int32",
        ScalarDType::UInt32 => "uint32",
        ScalarDType::Int64 => "int64",
        ScalarDType::UInt64 => "uint64",
        ScalarDType::WeakFloat => "weakfloat",
        ScalarDType::FP8E4M3 => "fp8e4m3",
        ScalarDType::FP8E4M3FNUZ => "fp8e4m3fnuz",
        ScalarDType::FP8E5M2 => "fp8e5m2",
        ScalarDType::FP8E5M2FNUZ => "fp8e5m2fnuz",
        ScalarDType::Float16 => "float16",
        ScalarDType::BFloat16 => "bfloat16",
        ScalarDType::Float32 => "float32",
        ScalarDType::Float64 => "float64",
        ScalarDType::Void => "void",
        ScalarDType::Index => "index",
    }
    .to_string()
}

fn canonical_dtype(dtype: &DType) -> CanonicalDType {
    match dtype {
        DType::Scalar(scalar) => CanonicalDType::Scalar { name: scalar_name(*scalar) },
        DType::Vector { scalar, count } => CanonicalDType::Vector { scalar: scalar_name(*scalar), count: *count },
        DType::Ptr { base, addrspace, size, vcount } => CanonicalDType::Pointer {
            base: Box::new(canonical_dtype(base)),
            address_space: match addrspace {
                AddrSpace::Global => "global",
                AddrSpace::Local => "local",
                AddrSpace::Reg => "register",
            }
            .to_string(),
            size: *size,
            count: *vcount,
        },
        DType::Image { kind, shape } => CanonicalDType::Image {
            image_kind: match kind {
                ImageKind::Half => "half",
                ImageKind::Float => "float",
            }
            .to_string(),
            shape: shape.clone(),
        },
    }
}

fn canonical_const(value: ConstValue) -> CanonicalConst {
    match value {
        ConstValue::Invalid => CanonicalConst::Invalid,
        ConstValue::Int(value) => CanonicalConst::Int { value },
        ConstValue::UInt(value) => CanonicalConst::UInt { value },
        ConstValue::Float(value) => CanonicalConst::Float { bits: format!("0x{:016x}", value.to_bits()) },
        ConstValue::Bool(value) => CanonicalConst::Bool { value },
    }
}

fn address_space_name(address_space: AddrSpace) -> String {
    match address_space {
        AddrSpace::Global => "global",
        AddrSpace::Local => "local",
        AddrSpace::Reg => "register",
    }
    .to_string()
}

fn canonical_program_value(value: &Arc<UOp>, ids: &FxHashMap<u64, usize>) -> crate::Result<CanonicalProgramValue> {
    Ok(match value.op() {
        Op::Const(value) => match canonical_const(value.0) {
            CanonicalConst::Int { value } => CanonicalProgramValue::Int { value },
            CanonicalConst::UInt { value } if value <= i64::MAX as u64 => {
                CanonicalProgramValue::Int { value: value as i64 }
            }
            CanonicalConst::UInt { value } => CanonicalProgramValue::UInt { value },
            CanonicalConst::Float { bits } => CanonicalProgramValue::Float { bits },
            CanonicalConst::Invalid | CanonicalConst::Bool { .. } => {
                return Err(crate::Error::MissingShape { operation: "canonical PROGRAM launch dimension" });
            }
        },
        _ => CanonicalProgramValue::Node {
            node: ids
                .get(&value.id)
                .copied()
                .ok_or(crate::Error::MissingShape { operation: "canonical PROGRAM symbolic launch dimension" })?,
        },
    })
}

fn canonical_param_slot(slot: usize) -> i128 {
    if slot == usize::MAX {
        return -1;
    }
    slot as i128
}

fn canonical_padding_values(value: &Arc<UOp>) -> crate::Result<Vec<usize>> {
    let values: Vec<_> = match value.op() {
        Op::Stack(ops::Stack { sources }) => sources.iter().collect(),
        _ => vec![value],
    };
    values
        .into_iter()
        .map(|value| match value.op() {
            Op::Const(value) => match value.0 {
                ConstValue::Int(value) if value >= 0 => Ok(value as usize),
                ConstValue::UInt(value) => usize::try_from(value).map_err(|_| crate::Error::CanonicalSerialization {
                    detail: "PAD extent does not fit usize".into(),
                }),
                _ => Err(crate::Error::CanonicalSerialization {
                    detail: "canonical PAD normalization requires nonnegative integer extents".into(),
                }),
            },
            _ => Err(crate::Error::CanonicalSerialization {
                detail: "symbolic PAD extents are not common to the pinned Svod/Tinygrad representations".into(),
            }),
        })
        .collect()
}

/// Tag carried only by rangeify-created global buffers whose high-bit slot is
/// an internal cache-restoration namespace rather than semantic PROGRAM ABI.
pub const TAG_SCHEDULE_LOCAL_BUFFER: usize = usize::MAX - 2;
/// Tag on PARAMs created for the callable ABI after rangeify.
pub const TAG_CODEGEN_PARAM: usize = usize::MAX - 3;
/// Tag on global PARAM placeholders used only by the schedule cache.
pub const TAG_SCHEDULE_CACHE_PARAM: usize = usize::MAX - 4;
/// Tag separating a CALL-source binding PARAM from its body-local counterpart.
pub const TAG_CALL_BIND_PARAM: usize = usize::MAX - 5;

fn canonical_arg(node: &Arc<UOp>, ids: &FxHashMap<u64, usize>, verbose: bool) -> crate::Result<CanonicalArg> {
    Ok(match node.op() {
        Op::Const(constant) => CanonicalArg::Const { value: canonical_const(constant.0) },
        Op::Unique(_) | Op::LUnique(_) => {
            return Err(crate::Error::CanonicalSerialization {
                detail: "UNIQUE/LUNIQUE allocation identities have no pinned Tinygrad semantic equivalent".into(),
            });
        }
        Op::Sink(ops::Sink { info: None, .. }) => CanonicalArg::None,
        Op::Sink(ops::Sink { info: Some(info), .. }) => CanonicalArg::Sink {
            name: info.name.clone(),
            opts_to_apply: info.opts_to_apply.clone(),
            applied_opts: info.applied_opts.clone(),
            dont_use_locals: info.dont_use_locals,
        },
        Op::Cast(ops::Cast { dtype, .. }) | Op::BitCast(ops::BitCast { dtype, .. }) => {
            CanonicalArg::DType { value: canonical_dtype(dtype) }
        }
        Op::MSelect(ops::MSelect { device_index, .. }) => CanonicalArg::Index { value: *device_index },
        Op::Special(ops::Special { name, .. }) => CanonicalArg::Name { value: name.clone() },
        Op::GetAddr(ops::GetAddr { device, .. }) => CanonicalArg::Device { name: device.canonicalize() },
        Op::Copy(ops::Copy { device, .. }) => CanonicalArg::Device { name: device.canonicalize() },
        Op::Param(ops::Param { arg, .. }) | Op::Buffer(ops::Buffer { arg, .. }) => CanonicalArg::Param {
            slot: if matches!(node.op(), Op::Buffer(..))
                && node.tag().as_ref().is_some_and(|tags| tags.contains(&TAG_SCHEDULE_LOCAL_BUFFER))
            {
                (arg.slot & (usize::MAX >> 1)) as i128
            } else {
                canonical_param_slot(arg.slot)
            },
            dtype: canonical_dtype(&arg.dtype),
            vmin_vmax: arg.vmin_vmax.as_ref().map(|(min, max)| (canonical_const(min.0), canonical_const(max.0))),
            multiple_of: arg.multiple_of,
            name: arg.name.clone(),
            address_space: arg.addrspace.map(address_space_name),
            axis: arg.axis,
            device: arg.device.as_ref().map(|device| device.canonicalize()),
            volatile: arg.volatile,
        },
        Op::Slice(ops::Slice { size, .. }) => CanonicalArg::Size { value: *size },
        Op::Stage(ops::Stage { opts, .. }) => CanonicalArg::Stage {
            device: opts.device.as_ref().map(|device| device.canonicalize()),
            local_axis: opts.local_axis.as_ref().map(canonical_axis),
            address_space: address_space_name(opts.addrspace),
            removable: opts.removable,
        },
        Op::Permute(ops::Permute { axes, .. }) => CanonicalArg::Axes { values: axes.clone() },
        Op::Flip(ops::Flip { axes, .. }) => CanonicalArg::BoolAxes { values: axes.clone() },
        Op::Pad(ops::Pad { begin_pads, end_pads, .. }) => {
            CanonicalArg::Pad { begin: canonical_padding_values(begin_pads)?, end: canonical_padding_values(end_pads)? }
        }
        Op::Multi(ops::Multi { axis, .. }) => CanonicalArg::Index { value: *axis },
        Op::ReduceAxis(ops::ReduceAxis { reduce_op, axes, .. }) => {
            CanonicalArg::Reduce { op: reduce_name(*reduce_op).to_string(), axes: Some(axes.clone()), num_axes: None }
        }
        Op::Reduce(ops::Reduce { reduce_op, num_axes, .. }) => {
            CanonicalArg::Reduce { op: reduce_name(*reduce_op).to_string(), axes: None, num_axes: Some(*num_axes) }
        }
        Op::AllReduce(ops::AllReduce { reduce_op, device, .. }) => {
            CanonicalArg::AllReduce { op: reduce_name(*reduce_op).to_string(), device: device.canonicalize() }
        }
        Op::Barrier(ops::Barrier { fence: fence @ BarrierFence::Local, .. }) => {
            CanonicalArg::Name { value: format!("{fence:?}").to_ascii_uppercase() }
        }
        Op::Range(ops::Range { axis_id, axis_type, .. }) => CanonicalArg::Range {
            axis: axis_id.path().to_vec(),
            renumbered: axis_id.is_renumbered(),
            axis_type: format!("{axis_type:?}").to_ascii_uppercase(),
        },
        Op::VConst(ops::VConst { values }) => {
            CanonicalArg::Constants { values: values.iter().copied().map(canonical_const).collect() }
        }
        Op::DefineVar(ops::DefineVar { name, min_val, max_val }) => {
            CanonicalArg::DefineVar { name: name.clone(), min: *min_val, max: *max_val }
        }
        Op::Wmma(ops::Wmma { metadata, .. }) => CanonicalArg::Wmma {
            dims: metadata.dims,
            dtype_in: canonical_dtype(&metadata.dtype_in),
            dtype_out: canonical_dtype(&metadata.dtype_out),
            device: metadata.device.canonical().to_string(),
            threads: metadata.threads,
            upcast_a: metadata.upcast_axes.as_ref().map_or_else(Vec::new, |axes| canonical_axis_extents(&axes.a)),
            upcast_b: metadata.upcast_axes.as_ref().map_or_else(Vec::new, |axes| canonical_axis_extents(&axes.b)),
            upcast_c: metadata.upcast_axes.as_ref().map_or_else(Vec::new, |axes| canonical_axis_extents(&axes.c)),
        },
        Op::Call(ops::Call { info, .. }) | Op::Function(ops::Function { info, .. }) => {
            if info.grad_tag.is_some() {
                return Err(crate::Error::CanonicalSerialization {
                    detail: "Svod CallInfo.grad_tag has no pinned Tinygrad field".into(),
                });
            }
            CanonicalArg::Call {
                grad_tag: None,
                // Svod has no equivalent of Tinygrad's call `aux` strings, and origins
                // are deliberately absent from the parity form (per-node origins are
                // rendered in the verbose form instead).
                metadata: Vec::new(),
                name: info.name.clone(),
                precompile: info.precompile,
                precompile_backward: info.precompile_backward,
            }
        }
        Op::GetTuple(ops::GetTuple { index, .. }) => CanonicalArg::Index { value: *index },
        Op::Source(ops::Source { code, .. }) if verbose => CanonicalArg::Source { code: code.clone() },
        Op::Source(..) => {
            return Err(crate::Error::CanonicalSerialization {
                detail: "SOURCE stage identity is not part of canonical v6; use verbose diagnostics".into(),
            });
        }
        Op::ProgramBinary(ops::ProgramBinary { bytes, .. }) if verbose => CanonicalArg::Binary { length: bytes.len() },
        Op::ProgramBinary(..) => {
            return Err(crate::Error::CanonicalSerialization {
                detail: "BINARY content is diagnostics-only; use verbose canonical serialization".into(),
            });
        }
        Op::Ins(ops::Ins { arg, .. }) => {
            CanonicalArg::Ins { opcode: arg.opcode.clone(), attributes: arg.attributes.clone() }
        }
        Op::Contiguous(ops::Contiguous { opts, .. }) => CanonicalArg::Hints { values: opts.to_vec() },
        Op::Custom(ops::Custom { code, .. }) | Op::CustomI(ops::CustomI { code, .. }) => {
            CanonicalArg::Code { value: code.clone() }
        }
        Op::CustomFunction(ops::CustomFunction { kind, .. }) => {
            CanonicalArg::CustomFunction { kind_name: format!("{kind:?}") }
        }
        Op::Program(ops::Program { info, .. }) => CanonicalArg::Program {
            name: info.name.clone(),
            global_size: info
                .global_size
                .iter()
                .map(|value| canonical_program_value(value, ids))
                .collect::<crate::Result<_>>()?,
            local_size: info
                .local_size
                .as_ref()
                .map(|size| size.iter().map(|value| canonical_program_value(value, ids)).collect())
                .transpose()?,
            vars: info
                .vars
                .iter()
                .map(|value| {
                    ids.get(&value.id)
                        .copied()
                        .ok_or(crate::Error::MissingShape { operation: "canonical PROGRAM variable" })
                })
                .collect::<crate::Result<_>>()?,
            globals: info.globals.clone(),
            outs: info.outs.clone(),
            ins: info.ins.clone(),
            target: info.target.canonicalize(),
        },
        Op::Noop
        | Op::Unary(..)
        | Op::Binary(..)
        | Op::Ternary(..)
        | Op::Group(..)
        | Op::Index(..)
        | Op::MStack(..)
        | Op::Stack(..)
        | Op::Reshape(..)
        | Op::Expand(..)
        | Op::Shrink(..)
        | Op::If(..)
        | Op::EndIf(..)
        | Op::End(..)
        | Op::Barrier(..)
        | Op::Bind(..)
        | Op::Tuple(..)
        | Op::Linear(..)
        | Op::Detach(..)
        | Op::ContiguousBackward(..)
        | Op::After(..)
        | Op::Precast(..)
        | Op::Load(..)
        | Op::Store(..) => CanonicalArg::None,
    })
}

fn canonical_op_name(op: &Op) -> String {
    match op {
        Op::Unary(kind, _) => kind.as_ref().to_ascii_uppercase(),
        Op::Binary(kind, _, _) => binary_name(*kind).to_string(),
        Op::Ternary(kind, _, _, _) => ternary_name(*kind).to_string(),
        _ => match op {
            Op::Const(_) => "CONST",
            Op::Unique(_) => "UNIQUE",
            Op::LUnique(_) => "LUNIQUE",
            Op::Noop => "NOOP",
            Op::Sink(..) => "SINK",
            Op::Group(..) => "GROUP",
            Op::Cast(..) => "CAST",
            Op::BitCast(..) => "BITCAST",
            Op::MSelect(..) => "MSELECT",
            Op::Special(..) => "SPECIAL",
            Op::Param(..) => "PARAM",
            Op::Buffer(..) => "BUFFER",
            Op::Slice(..) => "SLICE",
            Op::Stage(..) => "STAGE",
            Op::Index(..) => "INDEX",
            Op::GetAddr(..) => "GETADDR",
            Op::Copy(..) => "COPY",
            Op::MStack(..) => "MSTACK",
            Op::Stack(..) => "STACK",
            Op::Reshape(..) => "RESHAPE",
            Op::Permute(..) => "PERMUTE",
            Op::Expand(..) => "EXPAND",
            Op::Pad(..) => "PAD",
            Op::Shrink(..) => "SHRINK",
            Op::Flip(..) => "FLIP",
            Op::Multi(..) => "MULTI",
            Op::ReduceAxis(..) => "REDUCE_AXIS",
            Op::Reduce(..) => "REDUCE",
            Op::AllReduce(..) => "ALLREDUCE",
            Op::If(..) => "IF",
            Op::EndIf(..) => "ENDIF",
            Op::Range(..) => "RANGE",
            Op::End(..) => "END",
            Op::Barrier(..) => "BARRIER",
            Op::VConst(..) => "VCONST",
            Op::DefineVar(..) => "DEFINE_VAR",
            Op::Bind(..) => "BIND",
            Op::Wmma(..) => "WMMA",
            Op::Call(..) => "CALL",
            Op::Function(..) => "FUNCTION",
            Op::Tuple(..) => "TUPLE",
            Op::GetTuple(..) => "GETTUPLE",
            Op::Program(..) => "PROGRAM",
            Op::Linear(..) => "LINEAR",
            Op::Source(..) => "SOURCE",
            Op::ProgramBinary(..) => "BINARY",
            Op::Ins(..) => "INS",
            Op::Detach(..) => "DETACH",
            Op::Contiguous(..) => "CONTIGUOUS",
            Op::ContiguousBackward(..) => "CONTIGUOUS_BACKWARD",
            Op::After(..) => "AFTER",
            Op::Precast(..) => "PRECAST",
            Op::Custom(..) => "CUSTOM",
            Op::CustomFunction(..) => "CUSTOM_FUNCTION",
            Op::CustomI(..) => "CUSTOMI",
            Op::Load(..) => "LOAD",
            Op::Store(..) => "STORE",
            Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) => unreachable!(),
        }
        .to_string(),
    }
}

fn binary_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "ADD",
        BinaryOp::Mul => "MUL",
        BinaryOp::Sub => "SUB",
        BinaryOp::FloorMod => "FLOORMOD",
        BinaryOp::CMod => "CMOD",
        BinaryOp::Max => "MAX",
        BinaryOp::Pow => "POW",
        BinaryOp::FloorDiv => "FLOORDIV",
        BinaryOp::CDiv => "CDIV",
        BinaryOp::Fdiv => "FDIV",
        BinaryOp::Lt => "CMPLT",
        BinaryOp::Le => "CMPLE",
        BinaryOp::Eq => "CMPEQ",
        BinaryOp::Ne => "CMPNE",
        BinaryOp::Gt => "CMPGT",
        BinaryOp::Ge => "CMPGE",
        BinaryOp::And => "AND",
        BinaryOp::Or => "OR",
        BinaryOp::Xor => "XOR",
        BinaryOp::Shl => "SHL",
        BinaryOp::Shr => "SHR",
        BinaryOp::Threefry => "THREEFRY",
    }
}

fn ternary_name(op: TernaryOp) -> &'static str {
    match op {
        TernaryOp::Where => "WHERE",
        TernaryOp::MulAcc => "MULACC",
    }
}

fn reduce_name(op: crate::ReduceOp) -> &'static str {
    match op {
        crate::ReduceOp::Add => "ADD",
        crate::ReduceOp::Mul => "MUL",
        crate::ReduceOp::Max => "MAX",
        crate::ReduceOp::Min => "MIN",
    }
}
