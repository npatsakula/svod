//! Flat `Int32` addressing helpers over the IR: the pre-linearized path runs
//! no index-dtype lowering, so every constant here is a concrete `i32`.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{ConstValue, Op, UOp, ops};

pub fn c32(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(v))
}

pub fn add(a: &Arc<UOp>, b: &Arc<UOp>) -> Arc<UOp> {
    a.try_add(b).expect("i32 add")
}

pub fn mul(a: &Arc<UOp>, b: &Arc<UOp>) -> Arc<UOp> {
    a.try_mul(b).expect("i32 mul")
}

pub fn xor(a: &Arc<UOp>, b: &Arc<UOp>) -> Arc<UOp> {
    a.try_xor_op(b).expect("i32 xor")
}

pub fn and(a: &Arc<UOp>, b: &Arc<UOp>) -> Arc<UOp> {
    a.try_and_op(b).expect("i32 and")
}

pub fn shr(a: &Arc<UOp>, bits: u32) -> Arc<UOp> {
    a.try_shr_op(&c32(bits as i64)).expect("i32 shr")
}

/// The memory access form the renderer takes: a scalar INDEX, or for `w > 1`
/// the coalesced `SHRINK(buffer, offset, w)` of one `w`-wide access.
fn access(buf: &Arc<UOp>, off: &Arc<UOp>, w: usize) -> Arc<UOp> {
    if w == 1 {
        return UOp::index().buffer(buf.clone()).indices(vec![off.clone()]).call().expect("INDEX");
    }
    UOp::new(Op::Shrink(ops::Shrink { src: buf.clone(), offsets: off.clone(), sizes: c32(w as i64) }), buf.dtype())
}

pub fn index(buf: &Arc<UOp>, off: &Arc<UOp>, w: usize) -> Arc<UOp> {
    access(buf, off, w)
}

/// A `w`-wide load at `off`, `tag`ged so two loads of one address around a
/// store stay two loads.
pub fn load(buf: &Arc<UOp>, off: &Arc<UOp>, w: usize, tag: u64) -> Arc<UOp> {
    UOp::load().index(access(buf, off, w)).call().rtag(Some(smallvec![tag as usize]))
}

/// A `w`-wide load that yields zeros where `gate` is false.
pub fn load_gated(buf: &Arc<UOp>, off: &Arc<UOp>, w: usize, gate: &Arc<UOp>, tag: u64) -> Arc<UOp> {
    let elem = match buf.dtype() {
        DType::Ptr { base, .. } => *base,
        other => other,
    };
    let scalar = elem.scalar().expect("a scalar element");
    let zero = scalar.vec(w).zero_const();
    let index = access(buf, &off.valid(gate.clone()), w);
    UOp::load().index(index).alt(zero).gate(gate.clone()).call().rtag(Some(smallvec![tag as usize]))
}

trait ZeroConst {
    fn zero_const(self) -> Arc<UOp>;
}
impl ZeroConst for DType {
    fn zero_const(self) -> Arc<UOp> {
        let scalar = self.scalar().expect("scalar");
        let v = if scalar.is_float() { ConstValue::Float(0.0) } else { ConstValue::Int(0) };
        match self.count() {
            1 => UOp::const_(self, v),
            n => UOp::vconst(vec![v; n], DType::Scalar(scalar)),
        }
    }
}

/// A `vals.len()`-wide store at `off`.
pub fn store(buf: &Arc<UOp>, off: &Arc<UOp>, vals: Vec<Arc<UOp>>) -> Arc<UOp> {
    let w = vals.len();
    let value =
        if w == 1 { vals.into_iter().next().expect("one value") } else { UOp::stack(vals.into_iter().collect()) };
    access(buf, off, w).store(value)
}

/// Element `j` of a `w`-wide vector value (an `i32` constant-position INDEX).
pub fn elem(v: &Arc<UOp>, j: usize, w: usize) -> Arc<UOp> {
    if w == 1 {
        return v.clone();
    }
    UOp::index().buffer(v.clone()).indices(vec![c32(j as i64)]).call().expect("vector element")
}
