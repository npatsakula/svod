//! The kernels as tile programs: a spec (shapes, options, tile config) in, a
//! [`Program`](crate::ir::Program) out, generic over the 16-bit element type.

pub mod attention;
pub mod conv;
pub mod gemm;
pub mod heads;
pub mod nearest;
pub mod rows;

use std::f64::consts::{FRAC_1_SQRT_2, LOG2_E};
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;

use crate::build::*;
use crate::ir::*;

/// A config the planner admitted for a problem on its device: feasible on
/// the target for the problem's statics and claimed on its shape, the one
/// admission path ([`Planner::admitted`](crate::ops::config::Planner)). A
/// kernel spec takes its config as one, so a kernel is built from nothing
/// the planner did not let through, and the checks the admission makes are
/// not made again where the kernel is built. `'t` is the planner's borrow.
/// A config no planner admitted is taken as one only through the `unsafe`
/// [`Admitted::assume`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Admitted<'t, C>(C, PhantomData<&'t ()>);

impl<C> Admitted<'_, C> {
    pub(crate) fn mint(cfg: C) -> Self {
        Self(cfg, PhantomData)
    }

    /// `cfg` taken as admitted: for a kernel built outside the op layer.
    ///
    /// # Safety
    ///
    /// The kernel built from it checks nothing admission checks: where `cfg`
    /// is not feasible on the launch's target for the problem, or its
    /// claim does not fit the problem's shape, the kernel may read or write
    /// past its operands or compute the wrong values.
    pub unsafe fn assume(cfg: C) -> Self {
        Self::mint(cfg)
    }
}

impl<C> Deref for Admitted<'_, C> {
    type Target = C;
    fn deref(&self) -> &C {
        &self.0
    }
}

impl<C: fmt::Debug> fmt::Debug for Admitted<'_, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// The launch's batch axis, grid z.
#[derive(Clone, PartialEq, Eq)]
pub enum Batch {
    /// `n` batches, all launched; a single one is no addressing axis.
    Static(usize),
    /// A runtime variable bound by name; buffers hold its `max`.
    Var(Var),
}

/// The tune store keys by this text: a bound batch prints as it did when
/// the variable's fields were the variant's.
impl fmt::Debug for Batch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Batch::Static(n) => f.debug_tuple("Static").field(n).finish(),
            Batch::Var(v) => v.fmt(f),
        }
    }
}

impl Batch {
    pub fn var(name: impl Into<String>, min: i64, max: i64) -> Self {
        Batch::Var(Var { name: name.into(), min, max })
    }

    pub fn capacity(&self) -> usize {
        match self {
            Batch::Static(n) => *n,
            Batch::Var(v) => v.max as usize,
        }
    }

    /// The grid z extent and the batch index, `None` for a single static batch.
    fn axis<'k>(&self, k: &mut Kernel<'k>) -> (Sc<'k>, Option<Sc<'k>>) {
        match self {
            Batch::Static(1) => (Sc::from(1), None),
            Batch::Static(n) => (Sc::from(*n), Some(k.block(2))),
            Batch::Var(v) => (k.var(v.name.clone(), v.min, v.max), Some(k.block(2))),
        }
    }
}

/// `batch · stride`, the element offset of the current batch.
fn batch_offset<'k>(batch: &Option<Sc<'k>>, stride: usize) -> Sc<'k> {
    batch.clone().map_or(Sc::from(0), |b| b * stride)
}

/// A row/column bound only where the tile grid overhangs `len`.
fn bound<'k>(len: usize, tile: usize) -> Option<Sc<'k>> {
    (!len.is_multiple_of(tile)).then(|| Sc::from(len))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Act {
    #[default]
    None,
    /// `0.5·x·(1 + erf(x/√2))`, erf by Abramowitz–Stegun 7.1.26 (|error| ≤ 1.5e-7).
    Gelu,
    /// `x·sigmoid(x)`.
    Silu,
}

impl Act {
    fn apply<'k>(self, k: &mut Kernel<'k>, x: Regs<'k, F32>) -> Regs<'k, F32> {
        match self {
            Act::None => x,
            Act::Gelu => gelu(k, x),
            Act::Silu => silu(k, x),
        }
    }
}

fn konst<'k>(k: &mut Kernel<'k>, like: Regs<'k, F32>, v: f64) -> Regs<'k, F32> {
    let shape = k.shape(like);
    k.fill(shape, Const::Float(v))
}

fn affine<'k>(k: &mut Kernel<'k>, x: Regs<'k, F32>, mul: f64, add: f64) -> Regs<'k, F32> {
    let m = konst(k, x, mul);
    let x = k.binary(x, m, BinaryOp::Mul);
    let a = konst(k, x, add);
    k.binary(x, a, BinaryOp::Add)
}

/// `x / (1 + 2^(-x·log2 e))`.
fn silu<'k>(k: &mut Kernel<'k>, x: Regs<'k, F32>) -> Regs<'k, F32> {
    let e = affine(k, x, -LOG2_E, 0.0);
    let e = k.unary(e, UnaryOp::Exp2);
    let one = konst(k, x, 1.0);
    let d = k.binary(e, one, BinaryOp::Add);
    k.binary(x, d, BinaryOp::Div)
}

/// `0.5·(x + |x|·erf(|x|/√2))`, the sign of erf folded into `|x|`.
fn gelu<'k>(k: &mut Kernel<'k>, x: Regs<'k, F32>) -> Regs<'k, F32> {
    const P: f64 = 0.327_591_1;
    const A: [f64; 5] = [0.254_829_592, -0.284_496_736, 1.421_413_741, -1.453_152_027, 1.061_405_429];
    let ax = k.unary(x, UnaryOp::Abs);
    let z = affine(k, ax, FRAC_1_SQRT_2, 0.0);
    let t = affine(k, z, P, 1.0);
    let t = k.unary(t, UnaryOp::Recip);
    let mut poly = konst(k, t, A[4]);
    for a in A[..4].iter().rev() {
        let p = k.binary(poly, t, BinaryOp::Mul);
        poly = affine(k, p, 1.0, *a);
    }
    let poly = k.binary(poly, t, BinaryOp::Mul);
    let z2 = k.binary(z, z, BinaryOp::Mul);
    let e = affine(k, z2, -LOG2_E, 0.0);
    let e = k.unary(e, UnaryOp::Exp2);
    let tail = k.binary(poly, e, BinaryOp::Mul);
    let erf = affine(k, tail, -1.0, 1.0);
    let s = k.binary(ax, erf, BinaryOp::Mul);
    let s = k.binary(x, s, BinaryOp::Add);
    affine(k, s, 0.5, 0.0)
}

/// A 16-bit tile loaded and widened to f32.
fn load_f32<'k, T: Elem, P: Readable>(k: &mut Kernel<'k>, view: Tile<'k, P, T>) -> Regs<'k, F32> {
    let v = k.load(view);
    k.cast::<T, F32>(v)
}
