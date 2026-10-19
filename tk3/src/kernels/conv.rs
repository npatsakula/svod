//! Implicit-GEMM 2-D convolution over channels-last operands: the GEMM with
//! a gathered A operand. `M` is the output pixels, `N` the output channels
//! and `K = kh·kw·cin` walked tap-major, one pipeline over every tap.

use super::Batch;
use super::gemm::{Epilogue, GemmCfg, GemmSpec, mainloop_gemm};
use crate::build::*;
use crate::ir::*;

/// Shapes of one image's convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConvGeom {
    /// Input `[h, w, cin]`.
    pub h: usize,
    pub w: usize,
    pub cin: usize,
    pub cout: usize,
    /// `[kh, kw]`.
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    /// Zeros before and after the input along each spatial axis.
    pub pad: [usize; 2],
    pub dilation: [usize; 2],
}

impl ConvGeom {
    /// Output `[ho, wo]`; zero where the dilated kernel overhangs the padded input.
    pub fn out_hw(&self) -> [usize; 2] {
        let extent = |i: usize| {
            let (len, span) = ([self.h, self.w][i] + 2 * self.pad[i], self.dilation[i] * (self.kernel[i] - 1) + 1);
            if len < span { 0 } else { (len - span) / self.stride[i] + 1 }
        };
        [extent(0), extent(1)]
    }

    /// The reduction dim `kh·kw·cin`.
    pub fn k(&self) -> usize {
        self.kernel[0] * self.kernel[1] * self.cin
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConvSpec {
    /// A static batch folds into `M`; a bound one walks grid z.
    pub batch: Batch,
    pub geom: ConvGeom,
    pub epilogue: Epilogue,
    pub cfg: GemmCfg,
}

impl ConvSpec {
    /// The GEMM this convolution is: rows, columns, reduction and grid batch.
    pub fn gemm(&self) -> GemmSpec {
        let [ho, wo] = self.geom.out_hw();
        let (m, batch) = match self.batch {
            Batch::Static(n) => (n * ho * wo, Batch::Static(1)),
            ref var => (ho * wo, var.clone()),
        };
        GemmSpec { m, n: self.geom.cout, k: self.geom.k(), batch, epilogue: self.epilogue, cfg: self.cfg }
    }
}

/// Parameters in order: `x [batch, h, w, cin]`, `w [cout, kh, kw, cin]`,
/// `bias [cout]` if any, `residual [batch, ho, wo, cout]` if any,
/// `y [batch, ho, wo, cout]`. `cin` must be a multiple of `bk`, so a K step
/// never straddles two taps.
pub fn conv<T: Elem>(spec: &ConvSpec) -> Program {
    let g = spec.geom;
    let gemm = spec.gemm();
    let bk = spec.cfg.tile[2];
    assert!(g.cin.is_multiple_of(bk), "cin is a multiple of bk");
    let images = match spec.batch {
        Batch::Static(n) => n,
        Batch::Var { .. } => 1,
    };
    mainloop_gemm::<T, _>("conv", &gemm, images * g.h * g.w * g.cin, |_k, x, bb, row0| {
        let base = bb.clone().map_or(Sc::from(0), |b| b * (g.h * g.w * g.cin));
        let shape = Shape::new(spec.cfg.tile[0], bk);
        move |k: &mut Kernel, step: Sc, _koff: Sc| {
            im2col_view(k, x, &g, gemm.m, base.clone(), row0.clone(), step, shape)
        }
    })
}

/// The `[bm, bk]` A tile of K step `step` for output rows `row0..row0 + bm`
/// of `m`: tap `step / (cin/bk)` at channel `(step % (cin/bk))·bk`, every
/// row's input pixel read where it lies inside the image and zero in the
/// padding or past `m`. The row decode reads only the row and the block, so
/// the lowering lists it once; a step adds the tap's offset.
#[allow(clippy::too_many_arguments)]
pub fn im2col_view<T: Elem>(
    k: &mut Kernel,
    x: ParamRef<T>,
    g: &ConvGeom,
    m: usize,
    base: Sc,
    row0: Sc,
    step: Sc,
    shape: Shape,
) -> Gmem<T> {
    let [ho, wo] = g.out_hw();
    let (bk, kw, [sh, sw], [ph, pw], [dh, dw]) = (shape.cols, g.kernel[1], g.stride, g.pad, g.dilation);
    let spt = g.cin / bk;
    let tap = step.clone() / spt;
    let c0 = (step % spt) * bk;
    let (ky, kx) = (tap.clone() / kw * dh, tap % kw * dw);
    let i = row0 + k.row();
    let rem = i.clone() % (ho * wo);
    let (iy0, ix0) = (rem.clone() / wo * sh - ph as i64, rem % wo * sw - pw as i64);
    let start = ((i.clone() / (ho * wo) * g.h + iy0.clone()) * g.w + ix0.clone()) * g.cin;
    let (iy, ix) = (iy0 + ky.clone(), ix0 + kx.clone());
    let inside = |v: Sc, len: usize| Sc::from(0).le(v.clone()).and(v.lt(len));
    let mut valid = inside(iy, g.h).and(inside(ix, g.w));
    if !m.is_multiple_of(shape.rows) {
        valid = valid.and(i.lt(m));
    }
    let offset = base + (ky * g.w + kx) * g.cin + c0;
    k.gather(x, offset, start, Some(valid), shape, None)
}
