//! LayerNorm and RMSNorm over the last dim, optionally after a residual add.

use snafu::{ResultExt, ensure};
use svod_ir::SInt;
use svod_tensor::Tensor;

use super::shape::{self, Plan, extent, shape_of};
use super::{GraphSnafu, LaunchSnafu, Result, ShapeSnafu, batch_of, fmt_shape, live, typed};
use crate::kernels::rows::{Norm, NormSpec, norm as kernel};
use crate::launch;

/// `(x - mean)·rsqrt(var + eps)·w + b` over the last dim of `x`.
pub fn layer_norm(x: &Tensor, w: &Tensor, b: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    Ok(norm(Norm::Layer, x, None, w, b, eps)?.1)
}

/// `(x + residual, layer_norm(x + residual))`, the sum rounded to the input type first.
pub fn add_layer_norm(
    x: &Tensor,
    residual: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let (sum, y) = norm(Norm::Layer, x, Some(residual), w, b, eps)?;
    Ok((sum.expect("a fused sum"), y))
}

/// `x·rsqrt(mean(x²) + eps)·w` over the last dim of `x`.
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    Ok(norm(Norm::Rms, x, None, w, None, eps)?.1)
}

/// `(x + residual, rms_norm(x + residual))`, the sum rounded to the input type first.
pub fn add_rms_norm(x: &Tensor, residual: &Tensor, w: &Tensor, eps: f64) -> Result<(Tensor, Tensor)> {
    let (sum, y) = norm(Norm::Rms, x, Some(residual), w, None, eps)?;
    Ok((sum.expect("a fused sum"), y))
}

fn norm(
    kind: Norm,
    x: &Tensor,
    residual: Option<&Tensor>,
    w: &Tensor,
    b: Option<&Tensor>,
    eps: f64,
) -> Result<(Option<Tensor>, Tensor)> {
    let op = match kind {
        Norm::Layer => "layer_norm",
        Norm::Rms => "rms_norm",
    };
    let shape = |t: &Tensor| shape_of(t).context(GraphSnafu { op });
    let xs = shape(x)?;
    let shape_err = |operand, got: &[SInt], expected: String| ShapeSnafu { op, operand, got: fmt_shape(got), expected };
    let d = xs.last().ok_or_else(|| shape_err("x", &xs, "[..., D]".into()).build())?;
    for (operand, t) in [("w", Some(w)), ("b", b)] {
        if let Some(t) = t {
            let got = shape(t)?;
            ensure!(got == [d.clone()], shape_err(operand, &got, format!("[{d}]")));
        }
    }
    if let Some(r) = residual {
        let got = shape(r)?;
        ensure!(got == xs, shape_err("residual", &got, fmt_shape(&xs)));
    }

    let (ext, var) = extent(&xs).unzip();
    let target = super::target(&x.device());
    let dtypes: Vec<_> = [Some(x), residual, Some(w), b].into_iter().flatten().map(Tensor::dtype).collect();
    let Plan::Kernel(cfg) = shape::norm(target.as_ref(), &dtypes, ext.as_ref()) else {
        return graph(kind, x, residual, w, b, eps).context(GraphSnafu { op });
    };
    let (ext, var) = (ext.expect("planned"), var.flatten());
    let (lead, d) = (&ext.dims[..ext.dims.len() - 1], ext.dims[ext.dims.len() - 1]);
    let rows = lead[usize::from(var.is_some())..].iter().product();
    let spec = NormSpec { norm: kind, rows, d, batch: batch_of(&var, 1), eps, residual: residual.is_some(), cfg };
    let out = Tensor::empty(&ext.dims, x.dtype());
    let sum = residual.map(|_| Tensor::empty(&ext.dims, x.dtype()));
    let ins: Vec<&Tensor> = [Some(x), residual, Some(w), b, Some(&out), sum.as_ref()].into_iter().flatten().collect();
    let at = ins.len() - 1 - usize::from(sum.is_some());
    let mut outs =
        launch::graph_launch_all(typed!(x.dtype(), kernel, &spec), &cfg.lowering(target.expect("planned")), &ins)
            .context(LaunchSnafu { op })?;
    let sum = match sum {
        Some(_) => Some(live(op, outs.pop().expect("the sum"), &var)?),
        None => None,
    };
    Ok((sum, live(op, outs.swap_remove(at), &var)?))
}

pub(crate) fn graph(
    kind: Norm,
    x: &Tensor,
    residual: Option<&Tensor>,
    w: &Tensor,
    b: Option<&Tensor>,
    eps: f64,
) -> svod_tensor::error::Result<(Option<Tensor>, Tensor)> {
    let sum = residual.map(|r| x.try_add(r)).transpose()?;
    let x = sum.as_ref().unwrap_or(x);
    let y = match kind {
        Norm::Layer => x.layernorm_with().eps(eps).weight(w).maybe_bias(b).call()?,
        Norm::Rms => x.rms_norm_with().eps(eps).weight(w).call()?,
    };
    Ok((sum, y))
}
