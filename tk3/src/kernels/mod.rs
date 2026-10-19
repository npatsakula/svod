//! The kernels as tile programs: a spec (shapes, options, tile config) in, a
//! [`Program`](crate::ir::Program) out, generic over the 16-bit element type.

pub mod attention;
pub mod conv;
pub mod gemm;
pub mod heads;
pub mod rows;

use std::f64::consts::{FRAC_1_SQRT_2, LOG2_E};

use crate::build::*;
use crate::ir::*;

/// The launch's batch axis, grid z.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Batch {
    /// `n` batches, all launched; a single one is no addressing axis.
    Static(usize),
    /// A runtime variable bound by name in `min..=max`; buffers hold `max`.
    Var { name: String, min: i64, max: i64 },
}

impl Batch {
    pub fn capacity(&self) -> usize {
        match self {
            Batch::Static(n) => *n,
            Batch::Var { max, .. } => *max as usize,
        }
    }

    /// The grid z extent and the batch index, `None` for a single static batch.
    fn axis(&self, k: &mut Kernel) -> (Sc, Option<Sc>) {
        match self {
            Batch::Static(1) => (Sc::from(1), None),
            Batch::Static(n) => (Sc::from(*n), Some(k.block(2))),
            Batch::Var { name, min, max } => (k.var(name.clone(), *min, *max), Some(k.block(2))),
        }
    }
}

/// `batch · stride`, the element offset of the current batch.
fn batch_offset(batch: &Option<Sc>, stride: usize) -> Sc {
    batch.clone().map_or(Sc::from(0), |b| b * stride)
}

/// A row/column bound only where the tile grid overhangs `len`.
fn bound(len: usize, tile: usize) -> Option<Sc> {
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
    fn apply(self, k: &mut Kernel, x: Regs<F32>) -> Regs<F32> {
        match self {
            Act::None => x,
            Act::Gelu => gelu(k, x),
            Act::Silu => silu(k, x),
        }
    }
}

fn konst(k: &mut Kernel, like: Regs<F32>, v: f64) -> Regs<F32> {
    let shape = k.shape(like);
    k.fill(shape, Const::Float(v))
}

fn affine(k: &mut Kernel, x: Regs<F32>, mul: f64, add: f64) -> Regs<F32> {
    let m = konst(k, x, mul);
    let x = k.binary(x, m, BinaryOp::Mul);
    let a = konst(k, x, add);
    k.binary(x, a, BinaryOp::Add)
}

/// `x / (1 + 2^(-x·log2 e))`.
fn silu(k: &mut Kernel, x: Regs<F32>) -> Regs<F32> {
    let e = affine(k, x, -LOG2_E, 0.0);
    let e = k.unary(e, UnaryOp::Exp2);
    let one = konst(k, x, 1.0);
    let d = k.binary(e, one, BinaryOp::Add);
    k.binary(x, d, BinaryOp::Div)
}

/// `0.5·(x + |x|·erf(|x|/√2))`, the sign of erf folded into `|x|`.
fn gelu(k: &mut Kernel, x: Regs<F32>) -> Regs<F32> {
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
fn load_f32<T: Elem>(k: &mut Kernel, view: Gmem<T>) -> Regs<F32> {
    let v = k.load(view);
    k.cast::<T, F32>(v)
}
