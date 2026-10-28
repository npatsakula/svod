//! Direct port of tinygrad's linearizer (codegen/late/linearizer.py).
//!
//! Converts a UOp DAG into a linear instruction sequence using:
//! 1. Priority + tuplize-based "ideal order" sort
//! 2. Heap toposort respecting data dependencies

use std::cmp::Ordering;
use svod_ir::ops;

/// WMMA `(upcast, reduce, hidden)` axis lists, `None` when the sort key carries no WMMA.
type WmmaAxes = Option<(Vec<(usize, usize)>, Vec<(usize, usize)>, Vec<(usize, usize)>)>;
use std::collections::BinaryHeap;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use svod_dtype::{DType, ScalarDType};
use svod_ir::UOp;
use svod_ir::op::Op;
use svod_ir::types::{BinaryOp, ConstValue, ParamArg, TernaryOp, UnaryOp};

const TUPLE_ORDER: bool = true;

/// Exact `Ops.value` at Tinygrad 8c8b43de. Values above PYLITERAL are Svod-only
/// extensions and are deliberately kept outside the pinned enum's range.
fn op_value(op: &Op) -> u16 {
    match op {
        Op::Bind(..) => 1,
        Op::Special(..) => 2,
        Op::Buffer(..) => 3,
        Op::Noop => 4,
        Op::Param(..) => 6,
        Op::Function(..) => 7,
        Op::Call(..) => 8,
        Op::Program(..) => 9,
        Op::Linear(..) => 10,
        Op::Source(..) => 11,
        Op::ProgramBinary(..) => 12,
        Op::Sink(..) => 13,
        Op::After(..) => 14,
        Op::Group(..) => 15,
        Op::Stack(..) | Op::Shuffle(..) | Op::VConst(..) => 16,
        Op::Tuple(..) => 17,
        Op::GetTuple(..) => 18,
        Op::GetAddr(..) => 19,
        Op::Index(..) => 20,
        Op::Shrink(..) => 21,
        Op::Load(..) => 22,
        Op::Store(..) => 23,
        Op::Wmma(..) => 24,
        Op::Cast(..) => 25,
        Op::BitCast(..) => 26,
        Op::Unary(kind, _) => match kind {
            UnaryOp::Exp2 => 27,
            UnaryOp::Log2 => 28,
            UnaryOp::Sin => 29,
            UnaryOp::Sqrt => 30,
            UnaryOp::Reciprocal => 31,
            UnaryOp::Neg => 32,
            UnaryOp::Trunc => 33,
            // These decompose before pinned late codegen.
            UnaryOp::Not => 83,
            UnaryOp::Abs => 84,
            UnaryOp::Rsqrt => 85,
            UnaryOp::Exp => 86,
            UnaryOp::Log => 87,
            UnaryOp::Cos => 88,
            UnaryOp::Tan => 89,
            UnaryOp::Floor => 90,
            UnaryOp::Ceil => 91,
            UnaryOp::Round => 92,
            UnaryOp::Sign => 93,
            UnaryOp::Erf => 94,
            UnaryOp::Square => 95,
        },
        Op::Binary(kind, _, _) => match kind {
            BinaryOp::Add => 34,
            BinaryOp::Mul => 35,
            BinaryOp::Shl => 36,
            BinaryOp::Shr => 37,
            BinaryOp::CDiv => 38,
            BinaryOp::Max => 39,
            BinaryOp::CMod => 40,
            BinaryOp::Lt => 41,
            BinaryOp::Ne => 42,
            BinaryOp::Eq => 43,
            BinaryOp::Xor => 44,
            BinaryOp::Or => 45,
            BinaryOp::And => 46,
            BinaryOp::Threefry => 47,
            BinaryOp::Sub => 48,
            BinaryOp::Fdiv => 49,
            BinaryOp::Pow => 50,
            BinaryOp::FloorDiv => 51,
            BinaryOp::FloorMod => 52,
            BinaryOp::Le => 96,
            BinaryOp::Gt => 97,
            BinaryOp::Ge => 98,
        },
        Op::Ternary(TernaryOp::Where, _, _, _) => 53,
        Op::Ternary(TernaryOp::MulAcc, _, _, _) => 54,
        Op::Barrier(..) => 55,
        Op::Range(..) => 56,
        Op::If(..) => 57,
        Op::End(..) => 58,
        Op::EndIf(..) => 59,
        Op::Const(_) => 61,
        Op::Custom(..) => 62,
        Op::CustomI(..) => 63,
        Op::Ins(..) => 64,
        Op::Contiguous(..) => 65,
        Op::ContiguousBackward(..) => 66,
        Op::Detach(..) => 67,
        Op::Stage(..) => 68,
        Op::Copy(..) => 69,
        Op::MSelect(..) => 71,
        Op::MStack(..) => 72,
        Op::CustomFunction(..) => 73,
        Op::Reshape(..) => 74,
        Op::Permute(..) => 75,
        Op::Expand(..) => 76,
        Op::Pad(..) => 77,
        Op::Flip(..) => 78,
        Op::ReduceAxis(..) | Op::Reduce(..) => 80,
        Op::AllReduce(..) => 81,
        Op::Slice(..) => 70,
        // Svod-only IR nodes, after the pinned PYLITERAL=82.
        Op::Unique(_) => 99,
        Op::LUnique(_) => 100,
        Op::Multi(..) => 103,
        Op::DefineVar(..) => 106,
        Op::Precast(..) => 109,
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum ConstKey {
    Invalid,
    Bool(bool),
    Int(i128),
    Float(u64),
}

fn cmp_int_float(integer: i128, float: f64) -> Ordering {
    if float.is_nan() {
        return Ordering::Less;
    }
    if float == f64::INFINITY {
        return Ordering::Less;
    }
    if float == f64::NEG_INFINITY {
        return Ordering::Greater;
    }
    match (integer as f64).total_cmp(&float) {
        Ordering::Equal => integer.cmp(&(float as i128)),
        order => order,
    }
}

impl Ord for ConstKey {
    fn cmp(&self, other: &Self) -> Ordering {
        let integer = |value: &ConstKey| match value {
            ConstKey::Bool(v) => Some(*v as i128),
            ConstKey::Int(v) => Some(*v),
            _ => None,
        };
        match (self, other) {
            (ConstKey::Invalid, ConstKey::Invalid) => Ordering::Equal,
            (ConstKey::Invalid, _) => Ordering::Less,
            (_, ConstKey::Invalid) => Ordering::Greater,
            (ConstKey::Float(a), ConstKey::Float(b)) => f64::from_bits(*a).total_cmp(&f64::from_bits(*b)),
            (ConstKey::Float(a), b) => cmp_int_float(integer(b).unwrap(), f64::from_bits(*a)).reverse(),
            (a, ConstKey::Float(b)) => cmp_int_float(integer(a).unwrap(), f64::from_bits(*b)),
            (a, b) => integer(a).unwrap().cmp(&integer(b).unwrap()),
        }
    }
}

impl PartialOrd for ConstKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct DTypeKey(i16, usize, String, Option<char>, Vec<usize>);

fn dtype_key(dtype: &DType) -> DTypeKey {
    let scalar = dtype.scalar().unwrap_or(ScalarDType::Void);
    let (priority, bits, name, fmt) = match scalar {
        ScalarDType::Void => (-1, 0, "void", None),
        ScalarDType::WeakInt => (0, 800, "weakint", None),
        ScalarDType::Bool => (0, 1, "bool", Some('?')),
        ScalarDType::Int8 => (1, 8, "signed char", Some('b')),
        ScalarDType::UInt8 => (2, 8, "unsigned char", Some('B')),
        ScalarDType::Int16 => (3, 16, "short", Some('h')),
        ScalarDType::UInt16 => (4, 16, "unsigned short", Some('H')),
        ScalarDType::Int32 => (5, 32, "int", Some('i')),
        ScalarDType::UInt32 => (6, 32, "unsigned int", Some('I')),
        ScalarDType::Int64 | ScalarDType::Index => (7, 64, "long", Some('q')),
        ScalarDType::UInt64 => (8, 64, "unsigned long", Some('Q')),
        ScalarDType::WeakFloat => (9, 800, "weakfloat", None),
        ScalarDType::FP8E4M3 => (10, 8, "float8_e4m3", None),
        ScalarDType::FP8E4M3FNUZ => (10, 8, "float8_e4m3fnuz", None),
        ScalarDType::FP8E5M2 => (11, 8, "float8_e5m2", None),
        ScalarDType::FP8E5M2FNUZ => (11, 8, "float8_e5m2fnuz", None),
        ScalarDType::Float16 => (12, 16, "half", Some('e')),
        ScalarDType::BFloat16 => (13, 16, "__bf16", None),
        ScalarDType::Float32 => (14, 32, "float", Some('f')),
        ScalarDType::Float64 => (15, 64, "double", Some('d')),
    };
    let extension = match dtype {
        DType::Scalar(_) => vec![],
        DType::Vector { count, .. } => vec![1, *count],
        DType::Ptr { addrspace, size, vcount, .. } => vec![2, *addrspace as usize, size.unwrap_or(usize::MAX), *vcount],
        DType::Image { kind, shape } => {
            let mut out = vec![3, *kind as usize];
            out.extend(shape);
            out
        }
    };
    DTypeKey(priority, bits, name.into(), fmt, extension)
}

fn const_key(value: ConstValue) -> ConstKey {
    match value {
        ConstValue::Invalid => ConstKey::Invalid,
        ConstValue::Bool(v) => ConstKey::Bool(v),
        ConstValue::Int(v) => ConstKey::Int(v as i128),
        ConstValue::UInt(v) => ConstKey::Int(v as i128),
        ConstValue::Float(v) => {
            let canonical = if v == 0.0 {
                0.0
            } else if v.is_nan() {
                f64::NAN
            } else {
                v
            };
            ConstKey::Float(canonical.to_bits())
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct ParamKey {
    slot: i128,
    dtype: DTypeKey,
    vmin_vmax: Option<(ConstKey, ConstKey)>,
    multiple_of: Option<usize>,
    name: Option<String>,
    addrspace: Option<u8>,
    axis: Option<usize>,
    device: Option<String>,
    volatile: bool,
}

fn param_key(arg: &ParamArg) -> ParamKey {
    ParamKey {
        slot: if arg.slot == usize::MAX { -1 } else { arg.slot as i128 },
        dtype: dtype_key(&arg.dtype),
        vmin_vmax: arg.vmin_vmax.map(|(a, b)| (const_key(a.0), const_key(b.0))),
        multiple_of: arg.multiple_of,
        name: arg.name.clone(),
        addrspace: Some(arg.addrspace.map_or(4, |x| match x {
            svod_ir::AddrSpace::Global => 1,
            svod_ir::AddrSpace::Local => 2,
            svod_ir::AddrSpace::Reg => 3,
        })),
        axis: arg.axis,
        device: arg.device.as_ref().map(|x| x.canonicalize()),
        volatile: arg.volatile,
    }
}

fn partial_option_cmp<T: Ord>(left: &Option<T>, right: &Option<T>) -> Option<Ordering> {
    match (left, right) {
        (None, None) => Some(Ordering::Equal),
        (Some(left), Some(right)) => Some(left.cmp(right)),
        _ => None,
    }
}

fn partial_const_cmp(left: &ConstKey, right: &ConstKey) -> Option<Ordering> {
    let integer = |value: &ConstKey| match value {
        ConstKey::Bool(value) => Some(*value as i128),
        ConstKey::Int(value) => Some(*value),
        _ => None,
    };
    match (left, right) {
        (ConstKey::Invalid, ConstKey::Invalid) => Some(Ordering::Equal),
        (ConstKey::Invalid, _) | (_, ConstKey::Invalid) => None,
        (ConstKey::Float(left), ConstKey::Float(right)) => {
            let (left, right) = (f64::from_bits(*left), f64::from_bits(*right));
            if left.is_nan() || right.is_nan() {
                None
            } else if left == right {
                Some(Ordering::Equal)
            } else {
                left.partial_cmp(&right)
            }
        }
        (ConstKey::Float(left), right) => {
            let left = f64::from_bits(*left);
            (!left.is_nan()).then(|| cmp_int_float(integer(right).unwrap(), left).reverse())
        }
        (left, ConstKey::Float(right)) => {
            let right = f64::from_bits(*right);
            (!right.is_nan()).then(|| cmp_int_float(integer(left).unwrap(), right))
        }
        (left, right) => Some(integer(left).unwrap().cmp(&integer(right).unwrap())),
    }
}

fn partial_param_cmp(left: &ParamKey, right: &ParamKey) -> Option<Ordering> {
    let mut order = left.slot.cmp(&right.slot).then_with(|| left.dtype.cmp(&right.dtype));
    if order != Ordering::Equal {
        return Some(order);
    }
    order = match (&left.vmin_vmax, &right.vmin_vmax) {
        (None, None) => Ordering::Equal,
        (Some((left_min, left_max)), Some((right_min, right_max))) => {
            let order = partial_const_cmp(left_min, right_min)?;
            if order != Ordering::Equal { order } else { partial_const_cmp(left_max, right_max)? }
        }
        _ => return None,
    };
    if order != Ordering::Equal {
        return Some(order);
    }
    for order in [
        partial_option_cmp(&left.multiple_of, &right.multiple_of),
        partial_option_cmp(&left.name, &right.name),
        partial_option_cmp(&left.addrspace, &right.addrspace),
        partial_option_cmp(&left.axis, &right.axis),
        partial_option_cmp(&left.device, &right.device),
    ] {
        let order = order?;
        if order != Ordering::Equal {
            return Some(order);
        }
    }
    Some(left.volatile.cmp(&right.volatile))
}

fn partial_arg_cmp(left: &ArgKey, right: &ArgKey) -> Option<Ordering> {
    match (left, right) {
        (ArgKey::None, ArgKey::None) => Some(Ordering::Equal),
        (ArgKey::Const(left), ArgKey::Const(right)) => partial_const_cmp(left, right),
        (ArgKey::Constants(left), ArgKey::Constants(right)) => {
            for (left, right) in left.iter().zip(right) {
                let order = partial_const_cmp(left, right)?;
                if order != Ordering::Equal {
                    return Some(order);
                }
            }
            Some(left.len().cmp(&right.len()))
        }
        (ArgKey::Param(left), ArgKey::Param(right)) => partial_param_cmp(left, right),
        (ArgKey::Range(left_path, left_type), ArgKey::Range(right_path, right_type)) => {
            for (left, right) in left_path.iter().zip(right_path) {
                let order = left.cmp(right);
                if order != Ordering::Equal {
                    return Some(order);
                }
            }
            if left_path.len() != right_path.len() {
                None
            } else if left_type == right_type {
                Some(Ordering::Equal)
            } else {
                None
            }
        }
        (ArgKey::Reduce(left_op, left_axes, left_count), ArgKey::Reduce(right_op, right_axes, right_count)) => {
            let order = left_op.cmp(right_op).then_with(|| left_axes.cmp(right_axes));
            if order != Ordering::Equal { Some(order) } else { partial_option_cmp(left_count, right_count) }
        }
        (ArgKey::Text(left), ArgKey::Text(right)) => Some(left.cmp(right)),
        (ArgKey::Index(left), ArgKey::Index(right)) => Some(left.cmp(right)),
        (ArgKey::Indices(left), ArgKey::Indices(right)) => Some(left.cmp(right)),
        (ArgKey::Bools(left), ArgKey::Bools(right)) => Some(left.cmp(right)),
        (ArgKey::Ins(left_op, left_attrs), ArgKey::Ins(right_op, right_attrs)) => {
            Some(left_op.cmp(right_op).then_with(|| left_attrs.cmp(right_attrs)))
        }
        (ArgKey::DType(left), ArgKey::DType(right)) => Some(left.cmp(right)),
        (ArgKey::DefineVar(left_name, left_min, left_max), ArgKey::DefineVar(right_name, right_min, right_max)) => {
            Some(left_name.cmp(right_name).then_with(|| left_min.cmp(right_min)).then_with(|| left_max.cmp(right_max)))
        }
        (ArgKey::ReduceDevice(left_op, left_device), ArgKey::ReduceDevice(right_op, right_device)) => {
            Some(left_op.cmp(right_op).then_with(|| left_device.cmp(right_device)))
        }
        (
            ArgKey::Wmma(left_dims, left_dtype, left_device, left_threads, left_axes),
            ArgKey::Wmma(right_dims, right_dtype, right_device, right_threads, right_axes),
        ) => {
            let order = left_dims
                .cmp(right_dims)
                .then_with(|| left_dtype.cmp(right_dtype))
                .then_with(|| left_device.cmp(right_device))
                .then_with(|| left_threads.cmp(right_threads));
            if order != Ordering::Equal { Some(order) } else { partial_wmma_axes_cmp(left_axes, right_axes) }
        }
        (ArgKey::Bytes(left), ArgKey::Bytes(right)) => Some(left.cmp(right)),
        _ => None,
    }
}

fn pinned_scalar_dtype(dtype: &DType) -> bool {
    matches!(dtype, DType::Scalar(scalar) if *scalar != ScalarDType::Index)
}

pub(crate) fn tinygrad_weakint_expr(node: &Arc<UOp>) -> bool {
    node.dtype() == DType::WeakInt || matches!(node.dtype(), DType::Vector { scalar: ScalarDType::WeakInt, .. })
}

#[derive(Clone)]
enum PartialTuplizeRef {
    Node(Arc<UOp>),
    VConstLane { parent: Arc<UOp>, index: usize },
}

impl PartialTuplizeRef {
    fn op(&self) -> u16 {
        match self {
            Self::Node(node) => op_value(node.op()),
            Self::VConstLane { .. } => 61,
        }
    }

    fn arg(&self) -> ArgKey {
        match self {
            Self::Node(node) if matches!(node.op(), Op::VConst(..)) => ArgKey::None,
            Self::Node(node) => arg_key(node.op()),
            Self::VConstLane { parent, index } => {
                let Op::VConst(ops::VConst { values }) = parent.op() else { unreachable!() };
                ArgKey::Const(const_key(values[*index]))
            }
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::Node(node) if matches!(node.op(), Op::VConst(..)) || tinygrad_weakint_expr(node) => {
                node.dtype().scalar_dtype()
            }
            Self::Node(node) => node.dtype(),
            Self::VConstLane { parent, .. } => parent.dtype().scalar_dtype(),
        }
    }

    fn sources(&self) -> Vec<Self> {
        match self {
            Self::Node(node) => match node.op() {
                Op::VConst(ops::VConst { values }) => {
                    (0..values.len()).map(|index| Self::VConstLane { parent: node.clone(), index }).collect()
                }
                op => op.sources().into_iter().map(Self::Node).collect(),
            },
            Self::VConstLane { .. } => vec![],
        }
    }

    fn node_id(&self) -> Option<u64> {
        match self {
            Self::Node(node) => Some(node.id),
            Self::VConstLane { .. } => None,
        }
    }
}

fn partial_tuplize_ref_cmp(
    left: PartialTuplizeRef,
    right: PartialTuplizeRef,
    cache: &mut FxHashMap<(u64, u64), Option<Ordering>>,
) -> Option<Ordering> {
    let pair = left.node_id().zip(right.node_id());
    if let Some(pair) = pair
        && let Some(order) = cache.get(&pair)
    {
        return *order;
    }
    let left_op = left.op();
    let right_op = right.op();
    let left_dtype = left.dtype();
    let right_dtype = right.dtype();
    let order =
        if left_op > 82 || right_op > 82 || !pinned_scalar_dtype(&left_dtype) || !pinned_scalar_dtype(&right_dtype) {
            None
        } else {
            let mut order = left_op.cmp(&right_op);
            if order == Ordering::Equal {
                order = partial_arg_cmp(&left.arg(), &right.arg())?;
            }
            if order == Ordering::Equal {
                order = dtype_key(&left_dtype).cmp(&dtype_key(&right_dtype));
            }
            if order == Ordering::Equal {
                let left_sources = left.sources();
                let right_sources = right.sources();
                for (left, right) in left_sources.iter().zip(&right_sources) {
                    order = partial_tuplize_ref_cmp(left.clone(), right.clone(), cache)?;
                    if order != Ordering::Equal {
                        break;
                    }
                }
                if order == Ordering::Equal {
                    order = left_sources.len().cmp(&right_sources.len());
                }
            }
            Some(order)
        };
    if let Some(pair) = pair {
        cache.insert(pair, order);
        cache.insert((pair.1, pair.0), order.map(Ordering::reverse));
    }
    order
}

fn partial_tuplize_cmp(
    left: &Arc<UOp>,
    right: &Arc<UOp>,
    cache: &mut FxHashMap<(u64, u64), Option<Ordering>>,
) -> Option<Ordering> {
    partial_tuplize_ref_cmp(PartialTuplizeRef::Node(left.clone()), PartialTuplizeRef::Node(right.clone()), cache)
}

/// Upper bound on the memo below. A rewrite run over one model reaches a few
/// thousand distinct pairs; the cap only bounds a long-lived worker thread.
const TUPLIZE_CMP_MEMO_CAP: usize = 1 << 16;

thread_local! {
    /// Cross-call memo for [`tinygrad_tuplize_cmp`]. Tinygrad gets this for
    /// free: `tuplize` is a `cached_property` on the UOp (`uop/ops.py:187-189`),
    /// so a comparison walks each node's key once per process. Ours rebuilt the
    /// whole comparison from scratch on every call, and the caller is a rewrite
    /// pattern (`symbolic/patterns.rs:692`) that runs per candidate node.
    ///
    /// Keyed by UOp id pairs, which is sound because ids are monotonic and
    /// never reused (`ir/src/uop/hash_consing.rs:46-52`), so a verdict for a
    /// pair stays true for the life of the process.
    static TUPLIZE_CMP_MEMO: std::cell::RefCell<FxHashMap<(u64, u64), Option<Ordering>>> =
        std::cell::RefCell::new(FxHashMap::default());
}

/// Compare pinned Tinygrad `(op, arg, dtype, *src.tuplize)` keys without
/// inventing an order for Python-incomparable or Svod-only values.
pub(crate) fn tinygrad_tuplize_cmp(left: &Arc<UOp>, right: &Arc<UOp>) -> Option<Ordering> {
    TUPLIZE_CMP_MEMO.with_borrow_mut(|memo| {
        if memo.len() >= TUPLIZE_CMP_MEMO_CAP {
            memo.clear();
        }
        partial_tuplize_cmp(left, right, memo)
    })
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum ArgKey {
    None,
    Const(ConstKey),
    Constants(Vec<ConstKey>),
    Param(ParamKey),
    Text(String),
    Index(usize),
    Indices(Vec<usize>),
    Bools(Vec<bool>),
    Range(Vec<usize>, i32),
    Ins(String, Vec<(String, String)>),
    DType(DTypeKey),
    DefineVar(String, i64, i64),
    Reduce(u16, Vec<usize>, Option<usize>),
    ReduceDevice(u16, String),
    Call(Option<String>, Vec<String>, Option<String>, bool, bool),
    Hints(Vec<(String, Option<usize>, Option<i64>)>),
    Bytes(Vec<u8>),
    Wmma((usize, usize, usize), DTypeKey, String, usize, WmmaAxes),
}

fn reduce_value(op: svod_ir::ReduceOp) -> u16 {
    match op {
        svod_ir::ReduceOp::Add => 34,
        svod_ir::ReduceOp::Mul => 35,
        svod_ir::ReduceOp::Max => 39,
        svod_ir::ReduceOp::Min => 110,
    }
}

fn axis_type_value(axis: svod_ir::AxisType) -> i32 {
    match axis {
        svod_ir::AxisType::Device => 1,
        svod_ir::AxisType::Global => 2,
        svod_ir::AxisType::Warp => 3,
        svod_ir::AxisType::Local => 4,
        svod_ir::AxisType::Weak => 5,
        svod_ir::AxisType::GroupReduce => 6,
        svod_ir::AxisType::Reduce => 7,
        svod_ir::AxisType::Upcast => 8,
        svod_ir::AxisType::Unroll => 9,
        svod_ir::AxisType::Thread => 10,
        svod_ir::AxisType::Placeholder => 11,
        svod_ir::AxisType::Loop => 12,
    }
}

fn axis_pairs(values: &[(svod_ir::AxisId, usize)]) -> Vec<(usize, usize)> {
    values.iter().map(|(axis, amount)| (axis.path()[0], *amount)).collect()
}

fn partial_wmma_axes_cmp(left: &WmmaAxes, right: &WmmaAxes) -> Option<Ordering> {
    let (left, right) = match (left, right) {
        (None, None) => return Some(Ordering::Equal),
        (Some(left), Some(right)) => (left, right),
        _ => return None,
    };
    for (left, right) in [(&left.0, &right.0), (&left.1, &right.1), (&left.2, &right.2)] {
        let order = left.cmp(right);
        if order != Ordering::Equal {
            return Some(order);
        }
    }
    Some(Ordering::Equal)
}

fn tinygrad_renderer_device(device: svod_ir::RendererDevice) -> &'static str {
    match device {
        svod_ir::RendererDevice::Cpu => "CPU",
        svod_ir::RendererDevice::CudaSm75 | svod_ir::RendererDevice::CudaSm80 | svod_ir::RendererDevice::CudaSm89 => {
            "CUDA"
        }
        svod_ir::RendererDevice::Metal => "METAL",
        svod_ir::RendererDevice::AmdRdna3
        | svod_ir::RendererDevice::AmdRdna4
        | svod_ir::RendererDevice::AmdCdna3
        | svod_ir::RendererDevice::AmdCdna4 => "AMD",
        svod_ir::RendererDevice::IntelXe => "INTEL",
        svod_ir::RendererDevice::WebGpu => "WEBGPU",
    }
}

fn arg_key(op: &Op) -> ArgKey {
    match op {
        Op::Const(v) => ArgKey::Const(const_key(v.0)),
        Op::VConst(ops::VConst { values }) => ArgKey::Constants(values.iter().copied().map(const_key).collect()),
        Op::Param(ops::Param { arg, .. }) | Op::Buffer(ops::Buffer { arg, .. }) => ArgKey::Param(param_key(arg)),
        Op::Special(ops::Special { name, .. })
        | Op::Source(ops::Source { code: name, .. })
        | Op::Custom(ops::Custom { code: name, .. })
        | Op::CustomI(ops::CustomI { code: name, .. }) => ArgKey::Text(name.clone()),
        Op::Unique(v)
        | Op::LUnique(v)
        | Op::MSelect(ops::MSelect { device_index: v, .. })
        | Op::Multi(ops::Multi { axis: v, .. })
        | Op::GetTuple(ops::GetTuple { index: v, .. }) => ArgKey::Index(*v),
        Op::Slice(ops::Slice { size, .. }) => ArgKey::Index(*size),
        Op::Permute(ops::Permute { axes, .. }) => ArgKey::Indices(axes.clone()),
        Op::Flip(ops::Flip { axes, .. }) => ArgKey::Bools(axes.clone()),
        Op::Range(ops::Range { axis_id, axis_type, .. }) => {
            ArgKey::Range(axis_id.path().to_vec(), axis_type_value(*axis_type))
        }
        Op::DefineVar(ops::DefineVar { name, min_val, max_val }) => ArgKey::DefineVar(name.clone(), *min_val, *max_val),
        Op::Ins(ops::Ins { arg, .. }) => ArgKey::Ins(arg.opcode.clone(), arg.attributes.clone()),
        Op::Cast(ops::Cast { dtype, .. }) | Op::BitCast(ops::BitCast { dtype, .. }) => ArgKey::DType(dtype_key(dtype)),
        Op::GetAddr(ops::GetAddr { device, .. }) | Op::Copy(ops::Copy { device, .. }) => {
            ArgKey::Text(device.canonicalize())
        }
        Op::ReduceAxis(ops::ReduceAxis { reduce_op, axes, .. }) => {
            ArgKey::Reduce(reduce_value(*reduce_op), axes.clone(), None)
        }
        Op::Reduce(ops::Reduce { reduce_op, num_axes, .. }) => {
            ArgKey::Reduce(reduce_value(*reduce_op), vec![], Some(*num_axes))
        }
        Op::AllReduce(ops::AllReduce { reduce_op, device, .. }) => {
            ArgKey::ReduceDevice(reduce_value(*reduce_op), device.canonicalize())
        }
        Op::Call(ops::Call { info, .. }) | Op::Function(ops::Function { info, .. }) => ArgKey::Call(
            info.grad_tag.clone(),
            // Sorted rendered paths, not raw ids: an id is the arena's allocation
            // order, so ordering by it would make the emitted instruction order
            // depend on which scope happened to be interned first.
            info.origins
                .iter()
                .map(|&origin| svod_ir::origin::path(origin))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
            info.name.clone(),
            info.precompile,
            info.precompile_backward,
        ),
        Op::Contiguous(ops::Contiguous { opts, .. }) => {
            ArgKey::Hints(opts.iter().map(|hint| (hint.op.clone(), hint.axis, hint.arg)).collect())
        }
        Op::ProgramBinary(ops::ProgramBinary { bytes, .. }) => ArgKey::Bytes(bytes.clone()),
        Op::CustomFunction(ops::CustomFunction { kind, .. }) => ArgKey::Index(match kind {
            svod_ir::CustomFunctionKind::EncDec => 0,
            svod_ir::CustomFunctionKind::Graph => 1,
            svod_ir::CustomFunctionKind::AllReduce { reduce_op } => 2 + usize::from(reduce_value(*reduce_op)),
        }),
        Op::Wmma(ops::Wmma { metadata, .. }) => ArgKey::Wmma(
            metadata.dims,
            dtype_key(&metadata.dtype_in),
            tinygrad_renderer_device(metadata.device).into(),
            metadata.threads,
            metadata.upcast_axes.as_ref().map(|axes| (axis_pairs(&axes.a), axis_pairs(&axes.b), axis_pairs(&axes.c))),
        ),
        _ => ArgKey::None,
    }
}

/// `(op, arg, dtype)` head of one tuplize key.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct TuplizeHead {
    op: u16,
    arg: ArgKey,
    dtype: DTypeKey,
}

/// Tuplize keys of one toposorted graph, flattened into index tables.
///
/// Entry `i < nodes.len()` is the key of toposort node `i`; the entries after
/// that are the CONST lanes pinned Tinygrad's `STACK(CONST...)` gives a VCONST.
/// Heads are interned and ranked once, so a key's head compares as one integer
/// and only source walks remain per comparison.
struct TuplizeKeys {
    /// Distinct heads in ascending order; `head` indexes this table.
    #[cfg(test)]
    heads: Vec<TuplizeHead>,
    head: Vec<u32>,
    /// CSR layout: the sources of entry `i` are `src[src_start[i]..src_start[i + 1]]`.
    src_start: Vec<u32>,
    src: Vec<u32>,
}

/// One suspended comparison: `next` is how many of this pair's source pairs
/// have already come back Equal.
#[derive(Clone, Copy)]
struct TuplizeFrame {
    left: u32,
    right: u32,
    next: usize,
}

fn key_pair(left: u32, right: u32) -> u64 {
    (u64::from(left) << 32) | u64::from(right)
}

impl TuplizeKeys {
    fn sources(&self, key: u32) -> &[u32] {
        &self.src[self.src_start[key as usize] as usize..self.src_start[key as usize + 1] as usize]
    }

    #[cfg(test)]
    fn head_of(&self, key: u32) -> &TuplizeHead {
        &self.heads[self.head[key as usize] as usize]
    }

    /// Verdict available without walking sources: identity, head order, or a memoized pair.
    fn shallow_cmp(&self, left: u32, right: u32, memo: &FxHashMap<u64, Ordering>) -> Option<Ordering> {
        if left == right {
            return Some(Ordering::Equal);
        }
        match self.head[left as usize].cmp(&self.head[right as usize]) {
            Ordering::Equal => memo.get(&key_pair(left, right)).copied(),
            order => Some(order),
        }
    }

    /// Lexicographic `(head, *src)` comparison of two keys.
    ///
    /// Iterative rather than recursive: the key mirrors the UOp graph, so a deep
    /// chain (long CAST/PRECAST ladders, unrolled reductions) recursed once per
    /// level and overflowed the 8 MiB main stack around 20-30k deep. `memo`
    /// keeps every pair settled by a source walk; `stack` is scratch reused
    /// across calls. Keys are never truncated: truncating makes the order
    /// non-total.
    fn cmp(
        &self,
        left: u32,
        right: u32,
        memo: &mut FxHashMap<u64, Ordering>,
        stack: &mut Vec<TuplizeFrame>,
    ) -> Ordering {
        stack.clear();
        stack.push(TuplizeFrame { left, right, next: 0 });
        // Verdict of the frame just popped, still to be consumed by its parent.
        let mut settled: Option<Ordering> = None;

        loop {
            let frame = stack.last_mut().expect("the root frame is popped only by returning");
            let (mut decided, walked) = match settled.take() {
                // A source pair came back: Equal moves on to the next one, anything
                // else settles this frame.
                Some(Ordering::Equal) => {
                    frame.next += 1;
                    (None, true)
                }
                Some(order) => (Some(order), true),
                None => (self.shallow_cmp(frame.left, frame.right, memo), false),
            };
            let TuplizeFrame { left, right, next } = *frame;

            if decided.is_none() {
                let (left_src, right_src) = (self.sources(left), self.sources(right));
                decided = match (left_src.get(next), right_src.get(next)) {
                    (Some(&a), Some(&b)) => {
                        stack.push(TuplizeFrame { left: a, right: b, next: 0 });
                        None
                    }
                    _ => Some(left_src.len().cmp(&right_src.len())),
                };
            }

            let Some(order) = decided else { continue };
            stack.pop();
            if walked {
                memo.insert(key_pair(left, right), order);
                memo.insert(key_pair(right, left), order.reverse());
            }
            if stack.is_empty() {
                return order;
            }
            settled = Some(order);
        }
    }
}

/// Toposort position of every node by UOp id.
fn node_index(nodes: &[Arc<UOp>]) -> FxHashMap<u64, u32> {
    nodes.iter().enumerate().map(|(position, node)| (node.id, position as u32)).collect()
}

fn compute_tuplize(nodes: &[Arc<UOp>], index: &FxHashMap<u64, u32>) -> TuplizeKeys {
    let mut interned: FxHashMap<TuplizeHead, u32> = FxHashMap::default();
    let mut intern = |head: TuplizeHead| {
        let next = interned.len() as u32;
        *interned.entry(head).or_insert(next)
    };

    let mut head = Vec::with_capacity(nodes.len());
    let mut src_start = Vec::with_capacity(nodes.len() + 1);
    let mut src = Vec::with_capacity(nodes.len() * 2);
    let mut lanes = Vec::new();
    src_start.push(0);
    for node in nodes {
        match node.op() {
            // Svod's compact VCONST is pinned Tinygrad's STACK(CONST...).
            Op::VConst(ops::VConst { values }) => {
                let dtype = dtype_key(&DType::Scalar(node.dtype().base()));
                head.push(intern(TuplizeHead { op: 16, arg: ArgKey::None, dtype: dtype.clone() }));
                for value in values {
                    src.push((nodes.len() + lanes.len()) as u32);
                    lanes.push(intern(TuplizeHead {
                        op: 61,
                        arg: ArgKey::Const(const_key(*value)),
                        dtype: dtype.clone(),
                    }));
                }
            }
            op => {
                head.push(intern(TuplizeHead { op: op_value(op), arg: arg_key(op), dtype: dtype_key(&node.dtype()) }));
                src.extend(op.sources().iter().map(|child| index[&child.id]));
            }
        }
        src_start.push(src.len() as u32);
    }
    src_start.resize(nodes.len() + lanes.len() + 1, src.len() as u32);
    head.append(&mut lanes);

    let mut heads: Vec<(TuplizeHead, u32)> = interned.into_iter().collect();
    heads.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    let mut rank = vec![0u32; heads.len()];
    for (position, (_, id)) in heads.iter().enumerate() {
        rank[*id as usize] = position as u32;
    }
    for slot in &mut head {
        *slot = rank[*slot as usize];
    }
    TuplizeKeys {
        #[cfg(test)]
        heads: heads.into_iter().map(|(head, _)| head).collect(),
        head,
        src_start,
        src,
    }
}

/// Dense position of every node in ascending tuplize order; equal keys keep
/// their toposort order, as a stable sort on the keys would.
fn tuplize_ranks(nodes: &[Arc<UOp>], index: &FxHashMap<u64, u32>) -> Vec<u32> {
    let keys = compute_tuplize(nodes, index);
    let (mut memo, mut stack) = (FxHashMap::default(), Vec::new());
    let mut order: Vec<u32> = (0..nodes.len() as u32).collect();
    order.sort_by(|&left, &right| keys.cmp(left, right, &mut memo, &mut stack));
    let mut rank = vec![0; nodes.len()];
    for (position, &node) in order.iter().enumerate() {
        rank[node as usize] = position as u32;
    }
    rank
}

/// Compute run_count: `prod(int(r.vmax)+1 for r in u.ranges)`.
///
/// Mirrors tinygrad's `run_count = prod([int(r.vmax)+1 for r in u.ranges])`,
/// applied uniformly to every op. [`InScopeRangesProperty`] is the faithful
/// port of tinygrad's `u.ranges` (`_ranges`): it merges the sources' in-scope
/// ranges and pops the op's `ended_ranges()`. For an AFTER that pop already
/// drops the ranges its deps close (e.g. a post-loop `acc.after(end_R)` yields
/// the empty set → run_count 1), so AFTER needs no special handling — the
/// generic path places it outside the loop, not nested inside it.
///
/// [`InScopeRangesProperty`]: svod_ir::uop::properties::InScopeRangesProperty
fn run_count(uop: &Arc<UOp>) -> u64 {
    use svod_ir::uop::cached_property::CachedProperty;
    use svod_ir::uop::properties::InScopeRangesProperty;

    let in_scope = InScopeRangesProperty::get(uop);

    uop.ranges()
        .into_iter()
        .filter(|range| in_scope.contains(&range.id))
        .map(|range| match range.vmax() {
            ConstValue::Int(v) => (v + 1) as u64,
            ConstValue::UInt(v) => v + 1,
            _ => 1,
        })
        .product()
}

/// Priority assignment matching tinygrad `codegen/late/linearizer.py:24-32`
/// (pin `8c8b43de`). Returns `(priority, extra)`; `extra` is the PARAM slot.
///
/// Three arms older tinygrad had are absent because upstream removed them, not
/// because the port dropped them:
/// - `DEFINE_VAR = -19` — `4a4b6956d "remove DEFINE_VAR from codebase"`: the op
///   is gone, symbolic variables are PARAMs and take the `-20` arm.
/// - `CONST = -10` — `52b989c6c "don't place consts early"`: consts sort at the
///   generic `0` so they sink next to their consumer.
/// - `DEFINE_LOCAL = -18` / `DEFINE_REG = -17` — `649971f02 "remove
///   DEFINE_LOCAL and DEFINE_REG"` folded both into BUFFER and *inverted* the
///   pair: LOCAL is `-17`, REG (like GLOBAL) `-18`. Restoring the old order
///   would regress against the pin.
fn priority(uop: &Arc<UOp>) -> (i32, Option<i64>) {
    match uop.op() {
        Op::Param(ops::Param { arg, .. }) => (-20, Some(arg.slot as i64)),
        Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local) => (-17, None),
        Op::Buffer(..) => (-18, None),
        Op::Load(..) => (-1, None),
        Op::Store(..) => (1, None),
        Op::Range(..) => (5, None),
        Op::End(..) => (-5, None),
        _ => (0, None),
    }
}

/// Direct port of tinygrad's `linearize()` (linearizer.py:8-51).
pub fn linearize(sink: Arc<UOp>) -> Vec<Arc<UOp>> {
    let lst = sink.toposort();
    if lst.is_empty() {
        return vec![sink];
    }
    let index = node_index(&lst);

    let mut out_degree = vec![0u32; lst.len()];
    for u in &lst {
        for s in u.op().sources() {
            out_degree[index[&s.id] as usize] += 1;
        }
    }

    // Sort all nodes by (run_count, priority, extra, tuplize) — the "ideal order".
    // Assign sequential nkey based on sorted position.
    let ranks = if TUPLE_ORDER { tuplize_ranks(&lst, &index) } else { vec![0; lst.len()] };
    let sort_keys: Vec<_> = lst
        .iter()
        .zip(&ranks)
        .map(|(u, &rank)| {
            let (p, extra) = priority(u);
            (run_count(u), p, extra, rank)
        })
        .collect();
    let mut sorted: Vec<u32> = (0..lst.len() as u32).collect();
    sorted.sort_by_key(|&node| sort_keys[node as usize]);
    let mut nkey = vec![0usize; lst.len()];
    for (position, &node) in sorted.iter().enumerate() {
        nkey[node as usize] = position;
    }

    // Heap toposort: pop highest nkey first (max-heap), reverse at end.
    let root = index[&sink.id];
    let mut heap = BinaryHeap::new();
    heap.push((nkey[root as usize], root));

    let mut newlst: Vec<Arc<UOp>> = Vec::with_capacity(lst.len());
    let mut visited = vec![false; lst.len()];

    while let Some((_, node)) = heap.pop() {
        if std::mem::replace(&mut visited[node as usize], true) {
            continue;
        }
        let u = &lst[node as usize];
        newlst.push(u.clone());

        for v in u.op().sources() {
            let source = index[&v.id];
            let deg = &mut out_degree[source as usize];
            *deg = deg.saturating_sub(1);
            if *deg == 0 && !visited[source as usize] {
                heap.push((nkey[source as usize], source));
            }
        }
    }

    newlst.reverse();

    newlst
}

#[cfg(test)]
#[path = "../test/unit/linearize/linearize_internal.rs"]
mod tests;
