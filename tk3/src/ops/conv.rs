//! `y = scale·act(x ⊛ w + bias) + residual` over channels-last operands.

use snafu::{ResultExt, ensure};
use svod_dtype::{DType, ScalarDType};
use svod_ir::SInt;
use svod_tensor::Tensor;

use super::shape::{self, Plan, extent, shape_of};
use super::{Act, DtypeSnafu, GraphSnafu, LaunchSnafu, Linear, Result, ShapeSnafu, batch_of, fmt_shape, output, tuned};
use crate::build::{BF16, F16};
use crate::kernels::Batch;
use crate::kernels::conv::{ConvCfg, ConvGeom, ConvSpec};
use crate::kernels::gemm::{Epilogue, Scale};
use crate::launch;

#[derive(Clone, Debug)]
pub struct Conv<'a> {
    /// `[sh, sw]`.
    pub stride: [usize; 2],
    /// Zeros before and after the input along `[h, w]`.
    pub pad: [usize; 2],
    pub dilation: [usize; 2],
    /// Channel groups; `w` is `[Cout, kh, kw, Cin / groups]`.
    pub groups: usize,
    /// `[Cout]`, a folded norm included.
    pub bias: Option<&'a Tensor>,
    pub act: Act,
    /// `[B, Ho, Wo, Cout]` in the output type, added last.
    pub residual: Option<&'a Tensor>,
    /// Multiplies the activated value, before the residual add.
    pub scale: Option<f32>,
    /// The output type: the input's (`None`) or `Float32`, the accumulator unrounded.
    pub out_dtype: Option<DType>,
}

impl Default for Conv<'_> {
    fn default() -> Self {
        Self {
            stride: [1, 1],
            pad: [0, 0],
            dilation: [1, 1],
            groups: 1,
            bias: None,
            act: Act::None,
            residual: None,
            scale: None,
            out_dtype: None,
        }
    }
}

const OP: &str = "conv2d";

/// `x [B, H, W, Cin]` ⊛ `w [Cout, kh, kw, Cin / groups]` → `[B, Ho, Wo, Cout]`
/// with the epilogue `opts`; `B` may be a bound batch variable.
pub fn conv2d(x: &Tensor, w: &Tensor, opts: Conv) -> Result<Tensor> {
    let shape = |t: &Tensor| shape_of(t).context(GraphSnafu { op: OP });
    let (xs, ws) = (shape(x)?, shape(w)?);
    let shape_err =
        |operand, got: &[SInt], expected: String| ShapeSnafu { op: OP, operand, got: fmt_shape(got), expected };
    ensure!(xs.len() == 4, shape_err("x", &xs, "[B, H, W, Cin]".into()));
    let dims = |s: &[SInt]| s.iter().map(SInt::as_const).collect::<Option<Vec<usize>>>();
    let (Some(xd), Some(wd)) = (dims(&xs[1..]), dims(&ws)) else {
        return graph(x, w, opts).context(GraphSnafu { op: OP });
    };
    let [h, wi, cin] = [xd[0], xd[1], xd[2]];
    let groups = opts.groups.max(1);
    let expected = format!("[Cout, kh, kw, {}]", cin / groups);
    ensure!(
        wd.len() == 4 && cin.is_multiple_of(groups) && wd[3] * groups == cin && wd[0].is_multiple_of(groups),
        shape_err("w", &ws, expected)
    );
    ensure!(w.dtype() == x.dtype(), DtypeSnafu { op: OP, operand: "w", got: w.dtype(), want: x.dtype() });
    let geom = ConvGeom {
        h,
        w: wi,
        cin,
        cout: wd[0],
        kernel: [wd[1], wd[2]],
        stride: opts.stride,
        pad: opts.pad,
        dilation: opts.dilation,
    };
    let [ho, wo] = geom.out_hw();
    if let Some(bias) = opts.bias {
        let got = shape(bias)?;
        ensure!(got == [SInt::Const(geom.cout)], shape_err("bias", &got, format!("[{}]", geom.cout)));
    }
    let out: Vec<SInt> = [xs[0].clone(), ho.into(), wo.into(), geom.cout.into()].into();
    if let Some(residual) = opts.residual {
        let got = shape(residual)?;
        ensure!(got == out, shape_err("residual", &got, fmt_shape(&out)));
    }
    let out_dtype = opts.out_dtype.clone().unwrap_or(x.dtype());

    // A pointwise convolution is a linear layer over the pixels.
    let pointwise = geom.kernel == [1, 1] && geom.stride == [1, 1] && geom.pad == [0, 0] && groups == 1;
    if pointwise && out_dtype == x.dtype() {
        let w2 = w.try_reshape([geom.cout as isize, cin as isize]).context(GraphSnafu { op: OP })?;
        let lin = Linear { bias: opts.bias, act: opts.act, gated: false, residual: opts.residual, scale: opts.scale };
        return super::linear(x, &w2, lin);
    }

    let (ext, var) = extent(&xs).unzip();
    let target = super::target(&x.device());
    let dtypes: Vec<DType> = [Some(x), Some(w), opts.bias].into_iter().flatten().map(Tensor::dtype).collect();
    let residual_dtype = opts.residual.map(Tensor::dtype);
    let plan = shape::conv2d(target.as_ref(), &dtypes, out_dtype.clone(), residual_dtype, ext.as_ref(), &geom, groups);
    let Plan::Kernel(cfgs) = plan else { return graph(x, w, opts).context(GraphSnafu { op: OP }) };
    let (ext, var, target) = (ext.expect("planned"), var.flatten(), target.expect("planned"));
    let images = ext.dims[0];
    let batch = match var {
        Some(_) => batch_of(&var, images),
        None => Batch::Static(images),
    };
    let epilogue = Epilogue {
        bias: opts.bias.is_some(),
        act: opts.act,
        gated: false,
        residual: opts.residual.is_some(),
        scale: opts.scale.map(Scale::new),
        out_f32: out_dtype != x.dtype(),
    };
    let spec = |c: ConvCfg| ConvSpec { batch: batch.clone(), geom, epilogue, cfg: c.gemm, split: c.split };
    let programs = |spec: &ConvSpec| match x.dtype().scalar() {
        Some(ScalarDType::BFloat16) => spec.programs::<BF16>(&target),
        Some(ScalarDType::Float16) => spec.programs::<F16>(&target),
        other => unreachable!("{other:?} has no kernel"),
    };
    let key = [
        images,
        h,
        wi,
        cin,
        geom.cout,
        geom.kernel[0],
        geom.kernel[1],
        geom.stride[0],
        geom.stride[1],
        geom.pad[0],
        geom.pad[1],
        geom.dilation[0],
        geom.dilation[1],
    ];
    let cfg = tuned(OP, &target, x.dtype(), &key, (&batch, epilogue), &cfgs, |c| programs(&spec(c)));
    let mut programs = programs(&spec(cfg)).into_iter();
    let y = output(&[images, ho, wo, geom.cout], &var, out_dtype);
    let (prog, lowering) = programs.next().expect("the convolution");
    let ins = [Some(x), Some(w), opts.bias, opts.residual, Some(&y)];
    let Some((merge, merge_lowering)) = programs.next() else {
        let ins: Vec<&Tensor> = ins.into_iter().flatten().collect();
        return launch::graph_launch(prog, &lowering, &ins).context(LaunchSnafu { op: OP });
    };
    let partial = Tensor::empty(&[cfg.split * images * ho * wo * geom.cout], DType::Float32);
    let partial = launch::graph_launch(prog, &lowering, &[x, w, &partial]).context(LaunchSnafu { op: OP })?;
    let ins: Vec<&Tensor> = [Some(&partial), opts.bias, opts.residual, Some(&y)].into_iter().flatten().collect();
    launch::graph_launch(merge, &merge_lowering, &ins).context(LaunchSnafu { op: OP })
}

/// The graph's convolution of the same op: NCHW views, the accumulator,
/// bias, activation, scale and residual at f32, one cast to the output type.
pub(crate) fn graph(x: &Tensor, w: &Tensor, opts: Conv) -> svod_tensor::error::Result<Tensor> {
    let padding = opts.pad.map(|p| (p as isize, p as isize));
    let y = x
        .contiguous()
        .try_permute(&[0, 3, 1, 2])?
        .conv2d()
        .weight(&w.try_permute(&[0, 3, 1, 2])?)
        .maybe_bias(opts.bias)
        .groups(opts.groups.max(1))
        .stride(&opts.stride)
        .dilation(&opts.dilation)
        .padding(&padding)
        .acc_dtype(DType::Float32)
        .call()?;
    let y = match opts.act {
        Act::None => y,
        Act::Gelu => y.gelu_exact()?,
        Act::Silu => y.silu()?,
    };
    let y = match opts.scale {
        Some(scale) => y.try_mul(scale)?,
        None => y,
    };
    let y = y.try_permute(&[0, 2, 3, 1])?;
    let y = match opts.residual {
        Some(r) => y.try_add(r.cast(DType::Float32))?,
        None => y,
    };
    Ok(y.cast(opts.out_dtype.unwrap_or(x.dtype())))
}
