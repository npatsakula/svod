//! `act(x·wᵀ + bias) + residual`, or `act(gate)·up + residual` off a gated weight.

use snafu::{ResultExt, ensure};
use svod_ir::SInt;
use svod_tensor::Tensor;

use super::shape::{self, Plan, extent, shape_of};
use super::{Act, DtypeSnafu, GraphSnafu, LaunchSnafu, Result, ShapeSnafu, batch_of, fmt_shape, output, tuned, typed};
use crate::kernels::gemm::{Epilogue, GemmSpec, gemm};
use crate::launch;

#[derive(Clone, Copy, Debug, Default)]
pub struct Linear<'a> {
    /// `[N]` (`[2N]` gated).
    pub bias: Option<&'a Tensor>,
    pub act: Act,
    /// `w` is `[2N, K]`, gate rows over up rows, and the output is
    /// `act(gate)·up`: SwiGLU with [`Act::Silu`], GeGLU with [`Act::Gelu`].
    pub gated: bool,
    /// `[lead..., N]`, added last.
    pub residual: Option<&'a Tensor>,
}

const OP: &str = "linear";

/// `x [lead..., K]` · `w [N, K]`ᵀ → `[lead..., N]` with the epilogue `opts`.
pub fn linear(x: &Tensor, w: &Tensor, opts: Linear) -> Result<Tensor> {
    let shape = |t: &Tensor| shape_of(t).context(GraphSnafu { op: OP });
    let (xs, ws) = (shape(x)?, shape(w)?);
    let shape_err =
        |operand, got: &[SInt], expected: String| ShapeSnafu { op: OP, operand, got: fmt_shape(got), expected };
    let k = xs.last().ok_or_else(|| shape_err("x", &xs, "[..., K]".into()).build())?;
    ensure!(ws.len() == 2 && ws[1] == *k, shape_err("w", &ws, format!("[N, {k}]")));
    ensure!(w.dtype() == x.dtype(), DtypeSnafu { op: OP, operand: "w", got: w.dtype(), want: x.dtype() });
    let n = match (opts.gated, ws[0].as_const()) {
        (true, Some(rows)) if rows.is_multiple_of(2) => SInt::Const(rows / 2),
        (true, _) => return shape_err("w", &ws, format!("[2N, {k}]")).fail(),
        (false, _) => ws[0].clone(),
    };
    if let Some(bias) = opts.bias {
        let got = shape(bias)?;
        ensure!(got == [ws[0].clone()], shape_err("bias", &got, format!("[{}]", ws[0])));
    }
    let out: Vec<SInt> = xs[..xs.len() - 1].iter().cloned().chain([n.clone()]).collect();
    if let Some(residual) = opts.residual {
        let got = shape(residual)?;
        ensure!(got == out, shape_err("residual", &got, fmt_shape(&out)));
    }

    let (ext, var) = extent(&xs).unzip();
    let target = super::target(&x.device());
    let dtypes: Vec<_> =
        [Some(x), Some(w), opts.bias, opts.residual].into_iter().flatten().map(Tensor::dtype).collect();
    let plan = match n.as_const() {
        Some(n) => shape::linear(target.as_ref(), &dtypes, ext.as_ref(), n, opts.gated),
        None => Plan::Graph(shape::Fallback::Symbolic),
    };
    let Plan::Kernel(cfgs) = plan else { return graph(x, w, opts).context(GraphSnafu { op: OP }) };
    let (ext, var, target) = (ext.expect("planned"), var.flatten(), target.expect("planned"));
    let (lead, k) = (&ext.dims[..ext.dims.len() - 1], ext.dims[ext.dims.len() - 1]);
    let n = n.as_const().expect("planned");
    // A bound batch walks grid z; each batch is a GEMM over the rows behind it.
    let m = lead[usize::from(var.is_some())..].iter().product();
    let epilogue =
        Epilogue { bias: opts.bias.is_some(), act: opts.act, gated: opts.gated, residual: opts.residual.is_some() };
    let batch = batch_of(&var, 1);
    let spec = |cfg| GemmSpec { m, n, k, batch: batch.clone(), epilogue, cfg };
    let shape = [batch.capacity(), m, n, k];
    let cfg = tuned(OP, &target, x.dtype(), &shape, (&batch, epilogue), &cfgs, |cfg| {
        vec![(typed!(x.dtype(), gemm, &spec(cfg)), cfg.lowering(target.clone()))]
    });
    let y = output(&[lead, &[n]].concat(), &var, x.dtype());
    let ins: Vec<&Tensor> = [Some(x), Some(w), opts.bias, opts.residual, Some(&y)].into_iter().flatten().collect();
    launch::graph_launch(typed!(x.dtype(), gemm, &spec(cfg)), &cfg.lowering(target), &ins)
        .context(LaunchSnafu { op: OP })
}

/// The generic graph of the same op; the input is materialized, since a lazy
/// producer would be recomputed per output tile of the GEMM.
pub(crate) fn graph(x: &Tensor, w: &Tensor, opts: Linear) -> svod_tensor::error::Result<Tensor> {
    let y = x.contiguous().linear().weight(w).maybe_bias(opts.bias).call()?;
    let y = if opts.gated {
        let n = y.dim_const(-1)? / 2;
        let halves = y.split(&[n, n], -1)?;
        activate(&halves[0], opts.act)?.try_mul(&halves[1])?
    } else {
        activate(&y, opts.act)?
    };
    match opts.residual {
        Some(r) => y.try_add(r),
        None => Ok(y),
    }
}

fn activate(t: &Tensor, act: Act) -> svod_tensor::error::Result<Tensor> {
    match act {
        Act::None => Ok(t.clone()),
        Act::Gelu => t.gelu_exact(),
        Act::Silu => t.silu(),
    }
}
