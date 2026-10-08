//! GEMM epilogues and the `rows` template (LayerNorm / RMSNorm) as tk3 programs.

use std::f64::consts::{FRAC_1_SQRT_2, LOG2_E};

use crate::build::*;
use crate::ir::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    None,
    /// `0.5·x·(1 + erf(x/√2))`, erf by Abramowitz–Stegun 7.1.26 (|error| ≤ 1.5e-7).
    Gelu,
    /// `x·sigmoid(x)`.
    Silu,
    /// `silu(gate)·up`. The weight is `[2n, k]` with the gate rows stacked
    /// over the up rows (and the bias `[2n]` likewise); two accumulators share
    /// the A tile of every step.
    SwiGlu,
}

/// `c = act(a·bᵀ + bias) + residual`, applied to the f32 accumulator in
/// registers and rounded to bf16 once, at the store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Epilogue {
    pub bias: bool,
    pub residual: bool,
    pub act: Act,
}

/// [`Epilogue`] on a `[bm, bn]`-tiled `gemm_nt`. Parameters in order: `a [m, k]`,
/// `b [n, k]` (`[2n, k]` for SwiGLU), `bias` if any, `residual [m, n]` if any, `c [m, n]`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nt_epilogue(
    m: usize,
    n: usize,
    kk: usize,
    [bm, bn, bk]: [usize; 3],
    stages: usize,
    warps: u32,
    epi: Epilogue,
) -> Program {
    let halves = if epi.act == Act::SwiGlu { 2 } else { 1 };
    let mut k = Kernel::new("gemm_epilogue");
    let a = k.param::<BF16>("a", ParamKind::In, m * kk);
    let b = k.param::<BF16>("b", ParamKind::In, halves * n * kk);
    let bias = epi.bias.then(|| k.param::<BF16>("bias", ParamKind::In, halves * n));
    let residual = epi.residual.then(|| k.param::<BF16>("residual", ParamKind::In, m * n));
    let c = k.param::<BF16>("c", ParamKind::Out, m * n);
    k.grid([Sc::from(m / bm), Sc::from(n / bn), Sc::from(1)]);
    k.warps(warps);
    let (row0, col0) = (k.block(0) * bm, k.block(1) * bn);
    let a_view = k.view(a, 0, [kk, 1], Shape::new(bm, bk), [None, None]);
    let a_view = k.at(a_view, row0.clone(), 0);
    let b_view = k.view(b, 0, [kk, 1], Shape::new(bn, bk), [None, None]);
    let accs = if halves == 2 {
        let gate = k.at(b_view, col0.clone(), 0);
        let up = k.at(b_view, col0.clone() + n, 0);
        mainloop(&mut k, a_view, [gate, up], kk / bk, stages, bk).to_vec()
    } else {
        let b_view = k.at(b_view, col0.clone(), 0);
        mainloop(&mut k, a_view, [b_view], kk / bk, stages, bk).to_vec()
    };
    let mut accs = accs.into_iter().enumerate().map(|(half, acc)| match bias {
        Some(bias) => {
            let v = k.view(bias, 0, [0, 1], Shape::new(1, bn), [None, None]);
            let v = k.at(v, 0, col0.clone() + half * n);
            let v = k.load(v);
            let v = k.cast::<BF16, F32>(v);
            k.binary(acc, v, BinaryOp::Add)
        }
        None => acc,
    });
    let first = accs.next().expect("an accumulator");
    let mut out = match epi.act {
        Act::None => first,
        Act::Gelu => gelu(&mut k, first),
        Act::Silu => silu(&mut k, first),
        Act::SwiGlu => {
            let up = accs.next().expect("the up half");
            let gate = silu(&mut k, first);
            k.binary(gate, up, BinaryOp::Mul)
        }
    };
    if let Some(residual) = residual {
        let r = k.view(residual, 0, [n, 1], Shape::new(bm, bn), [None, None]);
        let r = k.at(r, row0.clone(), col0.clone());
        let r = k.load(r);
        let r = k.cast::<BF16, F32>(r);
        out = k.binary(out, r, BinaryOp::Add);
    }
    let out = k.cast::<F32, BF16>(out);
    let c_view = k.view(c, 0, [n, 1], Shape::new(bm, bn), [None, None]);
    let c_view = k.at(c_view, row0, col0);
    k.store(c_view, out);
    k.finish()
}

/// The K-streaming pipeline: one A tile and `N` B tiles per step into a
/// `stages`-deep shared ring, one f32 accumulator per B.
fn mainloop<const N: usize>(
    k: &mut Kernel,
    a: Gmem<BF16>,
    bs: [Gmem<BF16>; N],
    trips: usize,
    stages: usize,
    bk: usize,
) -> [Regs<F32>; N] {
    let (sa, sb) = (k.shape(a), k.shape(bs[0]));
    let a_s = k.smem::<BF16>("a_s", stages * sa.elems());
    let b_s = [(); N].map(|()| k.smem::<BF16>("b_s", stages * sb.elems()));
    let init = [(); N].map(|()| k.zeros::<F32>(Shape::new(sa.rows, sb.rows)));
    k.pipeline(
        trips,
        stages,
        init,
        |k, step, slot| {
            let koff = step * bk;
            for (src, alloc, shape) in
                std::iter::once((a, a_s, sa)).chain(bs.into_iter().zip(b_s).map(|(b, s)| (b, s, sb)))
            {
                let g = k.at(src, 0, koff.clone());
                let t = k.smem_slot::<BF16>(alloc, slot.clone(), shape);
                k.stage(t, g, CopyMode::Async);
            }
        },
        |k, _step, slot, accs| {
            let a_t = k.smem_slot::<BF16>(a_s, slot.clone(), sa);
            let mut i = 0;
            accs.map(|acc| {
                let b_t = k.smem_slot::<BF16>(b_s[i], slot.clone(), sb);
                i += 1;
                k.mma(acc, a_t, false, b_t, true)
            })
        },
    )
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
pub fn silu(k: &mut Kernel, x: Regs<F32>) -> Regs<F32> {
    let e = affine(k, x, -LOG2_E, 0.0);
    let e = k.unary(e, UnaryOp::Exp2);
    let one = konst(k, x, 1.0);
    let d = k.binary(e, one, BinaryOp::Add);
    k.binary(x, d, BinaryOp::Div)
}

/// `0.5·(x + |x|·erf(|x|/√2))`, the sign of erf folded into `|x|`.
pub fn gelu(k: &mut Kernel, x: Regs<F32>) -> Regs<F32> {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    /// `(x - mean)·rsqrt(var + eps)·w + b`.
    Layer,
    /// `x·rsqrt(mean(x²) + eps)·w`.
    Rms,
}

/// One warp per row over `[rows, d]` bf16, f32 math, bf16 out; `br` rows (and
/// warps) per block, rows past `rows` masked. Parameters in order: `x`,
/// `residual` if fused, `w [d]`, `b [d]` (LayerNorm), `out`, and `sum` (`x +
/// residual` in bf16, which is what gets normalized) if fused.
pub fn norm_rows(norm: Norm, rows: usize, d: usize, br: usize, eps: f64, residual: bool) -> Program {
    let mut k = Kernel::new(match norm {
        Norm::Layer => "layer_norm",
        Norm::Rms => "rms_norm",
    });
    let x = k.param::<BF16>("x", ParamKind::In, rows * d);
    let r = residual.then(|| k.param::<BF16>("residual", ParamKind::In, rows * d));
    let w = k.param::<BF16>("w", ParamKind::In, d);
    let b = (norm == Norm::Layer).then(|| k.param::<BF16>("b", ParamKind::In, d));
    let out = k.param::<BF16>("out", ParamKind::Out, rows * d);
    let sum = residual.then(|| k.param::<BF16>("sum", ParamKind::Out, rows * d));
    k.grid([Sc::from(rows.div_ceil(br)), Sc::from(1), Sc::from(1)]);
    k.warps(br as u32);
    let row0 = k.block(0) * br;
    let tile = Shape::new(br, d);
    let at = |k: &mut Kernel, p: ParamRef<BF16>| {
        let v = k.view(p, 0, [d, 1], tile, [Some(Sc::from(rows)), None]);
        k.at(v, row0.clone(), 0)
    };
    let load = |k: &mut Kernel, v: Gmem<BF16>| {
        let v = k.load(v);
        k.cast::<BF16, F32>(v)
    };
    let xv = at(&mut k, x);
    let mut xs = load(&mut k, xv);
    if let (Some(r), Some(sum)) = (r, sum) {
        let rv = at(&mut k, r);
        let rs = load(&mut k, rv);
        let s = k.binary(xs, rs, BinaryOp::Add);
        let s = k.cast::<F32, BF16>(s);
        let sv = at(&mut k, sum);
        k.store(sv, s);
        xs = k.cast::<BF16, F32>(s);
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
    let vector = |k: &mut Kernel, p: ParamRef<BF16>| {
        let v = k.view(p, 0, [0, 1], Shape::new(1, d), [None, None]);
        load(k, v)
    };
    let wv = vector(&mut k, w);
    y = k.binary(y, wv, BinaryOp::Mul);
    if let Some(b) = b {
        let bv = vector(&mut k, b);
        y = k.binary(y, bv, BinaryOp::Add);
    }
    let y = k.cast::<F32, BF16>(y);
    let ov = at(&mut k, out);
    k.store(ov, y);
    k.finish()
}
