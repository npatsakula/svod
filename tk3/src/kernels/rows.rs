//! The `rows` template: one warp per row, a fused reduce-then-map in f32 —
//! LayerNorm and RMSNorm, optionally after a residual add.

use super::{Batch, affine, batch_offset, bound, load_f32};
use crate::atoms::Target;
use crate::build::*;
use crate::ir::*;
use crate::layouts::WarpGrid;
use crate::lower::Lowering;
use crate::schedule::{Prefetch, Schedule};

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

impl NormCfg {
    /// No matrix core: the warp grid only has to cover the warps.
    pub fn lowering(&self, target: Target) -> Lowering {
        Lowering {
            target,
            schedule: Schedule::Uniform { prefetch: Prefetch::CpAsync, unroll: false },
            grid: WarpGrid { rows: 1, cols: 1 },
            swizzle: true,
        }
    }
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
/// element type, which is what gets normalized) if fused. `d` is a power
/// of two.
pub fn norm<T: Elem>(spec: &NormSpec) -> Program {
    let NormSpec { norm, rows, d, ref batch, eps, residual, bias, cfg: NormCfg { br } } = *spec;
    let elems = batch.capacity() * rows * d;
    let mut k = Kernel::new(match norm {
        Norm::Layer => "layer_norm",
        Norm::Rms => "rms_norm",
    });
    let x = k.param::<T>("x", ParamKind::In, elems);
    let r = residual.then(|| k.param::<T>("residual", ParamKind::In, elems));
    let w = k.param::<T>("w", ParamKind::In, d);
    let b = (norm == Norm::Layer && bias).then(|| k.param::<T>("b", ParamKind::In, d));
    let out = k.param::<T>("out", ParamKind::Out, elems);
    let sum = residual.then(|| k.param::<T>("sum", ParamKind::Out, elems));
    let (gz, bb) = batch.axis(&mut k);
    k.grid([Sc::from(rows.div_ceil(br)), Sc::from(1), gz]);
    k.warps(br as u32);
    let row0 = k.block(0) * br;
    let at = |k: &mut Kernel, p: ParamRef<T>| {
        let v = k.view(p, batch_offset(&bb, rows * d), [d, 1], Shape::new(br, d), [bound(rows, br), None]);
        k.at(v, row0.clone(), 0)
    };
    let xv = at(&mut k, x);
    let mut xs = load_f32(&mut k, xv);
    if let (Some(r), Some(sum)) = (r, sum) {
        let rv = at(&mut k, r);
        let rs = load_f32(&mut k, rv);
        let s = k.binary(xs, rs, BinaryOp::Add);
        let s = k.cast::<F32, T>(s);
        let sv = at(&mut k, sum);
        k.store(sv, s);
        xs = k.cast::<T, F32>(s);
    }
    let mean_of = |k: &mut Kernel, v: Regs<F32>| {
        let s = k.reduce(v, Axis::Row, ReduceOp::Sum);
        affine(k, s, 1.0 / d as f64, 0.0)
    };
    let centered = match norm {
        Norm::Layer => {
            let mean = mean_of(&mut k, xs);
            k.binary(xs, mean, BinaryOp::Sub)
        }
        Norm::Rms => xs,
    };
    let sq = k.binary(centered, centered, BinaryOp::Mul);
    let var = mean_of(&mut k, sq);
    let var = affine(&mut k, var, 1.0, eps);
    let inv = k.unary(var, UnaryOp::Rsqrt);
    let mut y = k.binary(centered, inv, BinaryOp::Mul);
    let vector = |k: &mut Kernel, p: ParamRef<T>| {
        let v = k.view(p, 0, [0, 1], Shape::new(1, d), [None, None]);
        load_f32(k, v)
    };
    let wv = vector(&mut k, w);
    y = k.binary(y, wv, BinaryOp::Mul);
    if let Some(b) = b {
        let bv = vector(&mut k, b);
        y = k.binary(y, bv, BinaryOp::Add);
    }
    let y = k.cast::<F32, T>(y);
    let ov = at(&mut k, out);
    k.store(ov, y);
    k.finish()
}
