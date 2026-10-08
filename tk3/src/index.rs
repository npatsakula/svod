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
pub fn access(buf: &Arc<UOp>, off: &Arc<UOp>, w: usize) -> Arc<UOp> {
    if w == 1 {
        return UOp::index().buffer(buf.clone()).indices(vec![off.clone()]).call().expect("INDEX");
    }
    UOp::new(Op::Shrink(ops::Shrink { src: buf.clone(), offsets: off.clone(), sizes: c32(w as i64) }), buf.dtype())
}

pub fn index(buf: &Arc<UOp>, off: &Arc<UOp>, w: usize) -> Arc<UOp> {
    access(buf, off, w)
}

/// A load through an access node, `tag`ged so two loads of one address
/// around a store stay two loads.
pub fn load_at(access: &Arc<UOp>, tag: u64) -> Arc<UOp> {
    UOp::load().index(access.clone()).call().rtag(Some(smallvec![tag as usize]))
}

/// A load through `access` where `gate` holds, zeros elsewhere (the form
/// the late gater produces: a clean address, the gate on the load).
pub fn load_gated_at(access: &Arc<UOp>, gate: &Arc<UOp>, tag: u64) -> Arc<UOp> {
    let zero = UOp::load().index(access.clone()).call().vconst_like(0);
    UOp::load().index(access.clone()).alt(zero).gate(gate.clone()).call().rtag(Some(smallvec![tag as usize]))
}

pub fn store_at(access: &Arc<UOp>, vals: Vec<Arc<UOp>>) -> Arc<UOp> {
    let w = vals.len();
    let value =
        if w == 1 { vals.into_iter().next().expect("one value") } else { UOp::stack(vals.into_iter().collect()) };
    access.store(value)
}

/// A `w`-wide load at `off`.
pub fn load(buf: &Arc<UOp>, off: &Arc<UOp>, w: usize, tag: u64) -> Arc<UOp> {
    load_at(&access(buf, off, w), tag)
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
