//! `scale·act(x·wᵀ + bias) + residual`, or `scale·act(gate)·up + residual`
//! off a gated weight; a fused projection's first heads rotated.

use std::fmt::Debug;

use snafu::{ResultExt, ensure};
use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;

use super::config::Planner;
use super::shape::{self, Plan, extent, shape_of};
use super::{
    Act, DEVICE_TEST, DtypeSnafu, GraphSnafu, LaunchSnafu, Result, ShapeSnafu, Tuner, batch_of, fmt_shape, output,
    typed,
};
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, Rope, Scale, gemm};
use crate::launch;
use crate::tune::TuneStore;

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
    /// Multiplies the activated (gated) value before the residual add:
    /// `x + 0.5·ffn(x)` is `residual: Some(x), scale: Some(0.5)`.
    pub scale: Option<f32>,
    /// Rotates the output's first heads after the bias, before anything
    /// else; a kernel carries it with a bias alone.
    pub rope: Option<Rotary<'a>>,
}

/// A rotary embedding of the output's first `heads` heads of `head_dim`
/// columns by `(cos, sin)` tables of `[1, T, 1, head_dim / 2]`, row `t` of
/// `x [B, T, K]` at position `t`, as `apply_rotary_emb` rotates split
/// halves: a fused projection's queries and keys turned in place, its values
/// after them stored as they are.
#[derive(Clone, Copy, Debug)]
pub struct Rotary<'a> {
    pub cos: &'a Tensor,
    pub sin: &'a Tensor,
    pub heads: usize,
    pub head_dim: usize,
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
    if let Some(Rotary { cos, sin, heads, head_dim }) = opts.rope {
        ensure!(xs.len() == 3 && !opts.gated, shape_err("x", &xs, "[B, T, K] under a plain weight".into()));
        let fits = head_dim > 0 && head_dim.is_multiple_of(2) && n.as_const().is_some_and(|n| heads * head_dim <= n);
        ensure!(fits, shape_err("w", &ws, format!("[≥ {heads}·{head_dim}, {k}] of even heads")));
        let table = vec![SInt::Const(1), xs[1].clone(), SInt::Const(1), SInt::Const(head_dim / 2)];
        for (operand, t) in [("cos", cos), ("sin", sin)] {
            let got = shape(t)?;
            ensure!(got == table, shape_err(operand, &got, fmt_shape(&table)));
        }
    }

    let (ext, var) = extent(&xs).unzip();
    let planner = super::planner(&x.device());
    let dtypes: Vec<_> =
        [Some(x), Some(w), opts.bias, opts.residual].into_iter().flatten().map(Tensor::dtype).collect();
    let plan = match n.as_const() {
        Some(n) => shape::linear(planner.as_deref(), &dtypes, ext.as_ref(), n, opts.gated),
        None => Plan::Graph(shape::Fallback::Symbolic),
    };
    let Some(mut cfgs) = super::kernel(OP, plan) else { return graph(x, w, opts).context(GraphSnafu { op: OP }) };
    if let Some(rope) = opts.rope {
        let alone = opts.act == Act::None && opts.residual.is_none() && opts.scale.is_none();
        if !alone {
            return graph(x, w, opts).context(GraphSnafu { op: OP });
        }
        // Tables off the stream dtype or no tile holding whole heads: the
        // plain kernel, then the graph's rotation.
        let typed = rope.cos.dtype() == x.dtype() && rope.sin.dtype() == x.dtype();
        cfgs = cfgs.iter().copied().filter(|c| typed && c.turns(rope.head_dim)).collect();
        if cfgs.is_empty() {
            tracing::debug!(target: super::PLAN, op = OP, head_dim = rope.head_dim, "graph rotation: no tile turns");
            let y = linear(x, w, Linear { rope: None, ..opts })?;
            return turned(&y, rope).context(GraphSnafu { op: OP });
        }
    }
    let (ext, var, planner) = (ext.expect("planned"), var.flatten(), planner.expect("planned"));
    let (lead, k) = (&ext.dims[..ext.dims.len() - 1], ext.dims[ext.dims.len() - 1]);
    let n = n.as_const().expect("planned");
    // A bound batch walks grid z; each batch is a GEMM over the rows behind it.
    let m = lead[usize::from(var.is_some())..].iter().product();
    let epilogue = Epilogue {
        bias: opts.bias.is_some(),
        act: opts.act,
        gated: opts.gated,
        residual: opts.residual.is_some(),
        scale: opts.scale.map(Scale::new),
        out_f32: false,
        rope: opts.rope.map(|r| Rope { heads: r.heads, head_dim: r.head_dim, seq: lead[lead.len() - 1] }),
    };
    let spec = GemmSpec { m, n, k, batch: batch_of(&var, 1), epilogue, cfg: cfgs[0] };
    let spec = tuned(TuneStore::global(), &planner, x.dtype(), &cfgs, spec);
    let y = output(&[lead, &[n]].concat(), &var, x.dtype()).context(GraphSnafu { op: OP })?;
    let (cos, sin) = opts.rope.map(|r| (r.cos, r.sin)).unzip();
    let ins: Vec<&Tensor> =
        [Some(x), Some(w), opts.bias, opts.residual, cos, sin, Some(&y)].into_iter().flatten().collect();
    launch::graph_launch(typed!(x.dtype(), gemm, &spec), &spec.cfg.lowering(&planner.device), &ins)
        .context(LaunchSnafu { op: OP })
}

/// `spec` under the config [`linear`] launches for its shape and epilogue:
/// the tuner's pick among `ranked`, through `store`.
pub fn tuned(store: &TuneStore, planner: &Planner, dtype: DType, ranked: &[GemmCfg], spec: GemmSpec) -> GemmSpec {
    let salt = (&spec.batch, spec.epilogue);
    tune(store, planner, dtype, ranked, (OP, salt), spec.clone())
}

/// [`tuned`] as the device tests and GEMM probes tune: under the kernel's
/// name and [`DEVICE_TEST`].
pub fn probed(store: &TuneStore, planner: &Planner, dtype: DType, ranked: &[GemmCfg], spec: GemmSpec) -> GemmSpec {
    tune(store, planner, dtype, ranked, ("gemm", DEVICE_TEST), spec)
}

fn tune(
    store: &TuneStore,
    planner: &Planner,
    dtype: DType,
    ranked: &[GemmCfg],
    (op, salt): (&'static str, impl Debug),
    spec: GemmSpec,
) -> GemmSpec {
    let shape = [spec.batch.capacity(), spec.m, spec.n, spec.k];
    let build = |cfg| vec![(typed!(dtype, gemm, &GemmSpec { cfg, ..spec.clone() }), cfg.lowering(&planner.device))];
    let cfg = Tuner::new(store, &planner.device, dtype.clone(), &shape, salt, build).tuned(op, ranked, <[_]>::to_vec);
    GemmSpec { cfg, ..spec }
}

/// The generic graph of the same op; the input is materialized, since a lazy
/// producer would be recomputed per output tile of the GEMM.
pub(crate) fn graph(x: &Tensor, w: &Tensor, opts: Linear) -> svod_tensor::error::Result<Tensor> {
    let y = x.contiguous().linear().weight(w).maybe_bias(opts.bias).call()?;
    let y = match opts.rope {
        Some(rope) => turned(&y, rope)?,
        None => y,
    };
    let y = if opts.gated {
        let n = y.dim_const(-1)? / 2;
        let halves = y.split(&[n, n], -1)?;
        activate(&halves[0], opts.act)?.try_mul(&halves[1])?
    } else {
        activate(&y, opts.act)?
    };
    let y = match opts.scale {
        Some(scale) => y.try_mul(scale)?,
        None => y,
    };
    match opts.residual {
        Some(r) => y.try_add(r),
        None => Ok(y),
    }
}

/// `y [B, T, N]` with its first heads rotated as `rope` says.
fn turned(y: &Tensor, rope: Rotary) -> svod_tensor::error::Result<Tensor> {
    let Rotary { cos, sin, heads, head_dim } = rope;
    let (b, t, n) = (y.dim(0)?, y.dim(1)?, y.dim_const(2)?);
    let width = heads * head_dim;
    let turn = |part: &Tensor| -> svod_tensor::error::Result<Tensor> {
        let split = [b.clone(), t.clone(), SInt::Const(heads), SInt::Const(head_dim)];
        part.try_reshape(split)?.apply_rotary_emb(cos, sin, false)?.try_reshape([b.clone(), t.clone(), width.into()])
    };
    if width == n {
        return turn(y);
    }
    let parts = y.split(&[width, n - width], -1)?;
    Tensor::cat(&[&turn(&parts[0])?, &parts[1]], -1)
}

fn activate(t: &Tensor, act: Act) -> svod_tensor::error::Result<Tensor> {
    match act {
        Act::None => Ok(t.clone()),
        Act::Gelu => t.gelu_exact(),
        Act::Silu => t.silu(),
    }
}
