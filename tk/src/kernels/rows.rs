//! The `rows` template: one warp per row, a fused reduce-then-map in f32 —
//! LayerNorm and RMSNorm, optionally after a residual add.

use super::{Batch, affine, batch_offset, bound, load_f32};
use crate::build::*;
use crate::ir::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Norm {
    /// `(x - mean)·rsqrt(var + eps)·w + b`.
    Layer,
    /// `x·rsqrt(mean(x²) + eps)·w`.
    Rms,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NormCfg {
    /// Rows (and warps) per block.
    pub br: usize,
}

/// `rows` rows of `d` per batch; the batch, when there is one, walks grid z.
#[derive(Clone, Debug, PartialEq)]
pub struct NormSpec {
    pub norm: Norm,
    pub rows: usize,
    pub d: usize,
    pub batch: Batch,
    pub eps: f64,
    /// Normalize `x + residual`, which is also written out.
    pub residual: bool,
    /// LayerNorm adds a `[d]` bias.
    pub bias: bool,
    pub cfg: NormCfg,
}

/// Parameters in order: `x`, `residual` if fused, `w [d]`, `b [d]` (a
/// LayerNorm with a bias), `out`, and `sum` (`x + residual` rounded to the
/// element type, which is what gets normalized) if fused. `d` is a multiple
/// of 256 (`norm_candidates`).
pub fn norm<T: Elem>(spec: &NormSpec) -> Program {
    Kernel::record(
        match spec.norm {
            Norm::Layer => "layer_norm",
            Norm::Rms => "rms_norm",
        },
        |k| norm_body::<T>(k, spec),
    )
}

fn norm_body<'k, T: Elem>(k: &mut Kernel<'k>, spec: &NormSpec) {
    let NormSpec { norm, rows, d, ref batch, eps, residual, bias, cfg: NormCfg { br } } = *spec;
    let elems = batch.capacity() * rows * d;
    let x = k.input::<T>("x", elems);
    let r = residual.then(|| k.input::<T>("residual", elems));
    let w = k.input::<T>("w", d);
    let b = (norm == Norm::Layer && bias).then(|| k.input::<T>("b", d));
    let out = k.output::<T>("out", elems);
    let sum = residual.then(|| k.output::<T>("sum", elems));
    let (gz, bb) = batch.axis(k);
    k.grid([Sc::from(rows.div_ceil(br)), Sc::from(1), gz]);
    k.warps(br as u32);
    let row0 = k.block(0) * br;
    // The block's rows of an `[rows, d]` parameter.
    fn at<'k, T: Elem, A: Access>(
        k: &mut Kernel<'k>,
        p: ParamRef<'k, T, A>,
        bb: &Option<Sc<'k>>,
        [rows, br, d]: [usize; 3],
        row0: &Sc<'k>,
    ) -> Gmem<'k, T, A> {
        let v = k.view(p, batch_offset(bb, rows * d), [d, 1], Shape::new(br, d), [bound(rows, br), None]);
        k.at(v, row0.clone(), 0)
    }
    let xv = at(k, x, &bb, [rows, br, d], &row0);
    let mut xs = load_f32(k, xv);
    if let (Some(r), Some(sum)) = (r, sum) {
        let rv = at(k, r, &bb, [rows, br, d], &row0);
        let rs = load_f32(k, rv);
        let s = k.binary(xs, rs, BinaryOp::Add);
        let s = k.cast::<F32, T>(s);
        let sv = at(k, sum, &bb, [rows, br, d], &row0);
        k.store(sv, s);
        xs = k.cast::<T, F32>(s);
    }
    let mean_of = |k: &mut Kernel<'k>, v: Regs<'k, F32>| {
        let s = k.reduce(v, Axis::Row, ReduceOp::Sum);
        affine(k, s, 1.0 / d as f64, 0.0)
    };
    let centered = match norm {
        Norm::Layer => {
            let mean = mean_of(k, xs);
            k.binary(xs, mean, BinaryOp::Sub)
        }
        Norm::Rms => xs,
    };
    let sq = k.binary(centered, centered, BinaryOp::Mul);
    let var = mean_of(k, sq);
    let var = affine(k, var, 1.0, eps);
    let inv = k.unary(var, UnaryOp::Rsqrt);
    let mut y = k.binary(centered, inv, BinaryOp::Mul);
    let vector = |k: &mut Kernel<'k>, p: ParamRef<'k, T, In>| {
        let v = k.view(p, 0, [0, 1], Shape::new(1, d), [None, None]);
        load_f32(k, v)
    };
    let wv = vector(k, w);
    y = k.binary(y, wv, BinaryOp::Mul);
    if let Some(b) = b {
        let bv = vector(k, b);
        y = k.binary(y, bv, BinaryOp::Add);
    }
    let y = k.cast::<F32, T>(y);
    let ov = at(k, out, &bb, [rows, br, d], &row0);
    k.store(ov, y);
}
