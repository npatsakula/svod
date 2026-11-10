//! `scale·act(x·wᵀ + bias) + residual`, or `scale·act(gate)·up + residual`
//! off a gated weight; a fused projection's first heads rotated.

use std::fmt::Debug;

use snafu::{ResultExt, ensure};
use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;

use super::config::{Planner, distinct_configs};
use super::shape::{self, Plan, extent, shape_of};
use super::{
    Act, DEVICE_TEST, DtypeSnafu, GraphSnafu, LaunchSnafu, Result, ShapeSnafu, Tuner, batch_of, fmt_shape,
    kernel_buffer, output, typed,
};
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, Rope, Scale, TailSpec, gemm, gemm_tail};
use crate::kernels::{Admits, Admitted, Batch};
use crate::launch;
use crate::lower::Lowering;
use crate::tune::{Candidate, TuneStore};

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
    // The rope joins once the tables' positions are known.
    let epilogue = Epilogue {
        bias: opts.bias.is_some(),
        act: opts.act,
        gated: opts.gated,
        residual: opts.residual.is_some(),
        scale: opts.scale.map(Scale::new),
        out_f32: false,
        rope: None,
    };
    let plan = match n.as_const() {
        Some(n) => shape::linear(planner.as_deref(), &dtypes, ext.as_ref(), n, &epilogue),
        None => Plan::Graph(shape::Fallback::Symbolic),
    };
    let Some(cfgs) = super::kernel(OP, plan) else { return graph(x, w, opts).context(GraphSnafu { op: OP }) };
    let (ext, var, planner) = (ext.expect("planned"), var.flatten(), planner.expect("planned"));
    let (lead, k) = (&ext.dims[..ext.dims.len() - 1], ext.dims[ext.dims.len() - 1]);
    let n = n.as_const().expect("planned");
    // A bound batch walks grid z; each batch is a GEMM over the rows behind it.
    let m = lead[usize::from(var.is_some())..].iter().product();
    // A turn runs alone after the bias: beside an activation, a residual or
    // a scale the whole linear runs on the graph, which rotates before them.
    // Its tables ride the stream dtype; tiles that hold no whole heads are
    // refused by the ranking ([`GemmProblem::claim`]), so an empty one — or
    // tables off the dtype — means the plain kernel and the graph's rotation.
    if opts.rope.is_some() && !(opts.act == Act::None && opts.residual.is_none() && opts.scale.is_none()) {
        return graph(x, w, opts).context(GraphSnafu { op: OP });
    }
    let tables = |r: &Rotary| r.cos.dtype() == x.dtype() && r.sin.dtype() == x.dtype();
    let seq = ext.dims[ext.dims.len() - 2];
    let rope = opts.rope.filter(|r| tables(r)).map(|r| Rope { heads: r.heads, head_dim: r.head_dim, seq });
    let cfgs: Vec<GemmCfg> = match rope {
        Some(rope) => {
            let batches = if var.is_some() { lead[0] } else { 1 };
            planner.gemm_ranked(batches, m, n, k, &Epilogue { rope: Some(rope), ..epilogue }).to_vec()
        }
        None => cfgs.to_vec(),
    };
    // A tail split runs a single static batch: a bound one of a single row
    // walks each tile whole.
    let cfgs = match var {
        Some(_) => distinct_configs(&cfgs.iter().map(|c| GemmCfg { tail: None, ..*c }).collect::<Vec<_>>()),
        None => cfgs,
    };
    if let Some(r) = opts.rope.filter(|_| rope.is_none() || cfgs.is_empty()) {
        tracing::debug!(target: super::PLAN, op = OP, "graph rotation: no tile turns");
        let y = linear(x, w, Linear { rope: None, ..opts })?;
        return turned(&y, r).context(GraphSnafu { op: OP });
    }
    let epilogue = Epilogue { rope, ..epilogue };
    let batch = batch_of(&var, 1);
    let spec = tuned(TuneStore::global(), &planner, x.dtype(), &cfgs, GemmSpec { m, n, k, batch, epilogue });
    super::ran(OP, "gemm", spec.cfg());
    let y = output(&[lead, &[n]].concat(), &var, x.dtype()).context(GraphSnafu { op: OP })?;
    // The tables bind only where the kernel turns: `epilogue.rope` is the
    // kernel's own word for it (`opts.rope` may have fallen back to the
    // graph's rotation).
    let (cos, sin) = spec.epilogue.rope.zip(opts.rope).map(|(_, r)| (r.cos, r.sin)).unzip();
    let mut programs = programs(&spec, &x.dtype(), &planner);
    match tail(&spec, &planner) {
        Some(tail) => launched_tail(programs, &tail, [x, w], opts.bias, opts.residual, &y),
        None => {
            let (prog, lowering) = programs.remove(0);
            let ins: Vec<&Tensor> =
                [Some(x), Some(w), opts.bias, opts.residual, cos, sin, Some(&y)].into_iter().flatten().collect();
            launch::graph_launch(prog, &lowering, &ins).context(LaunchSnafu { op: OP })
        }
    }
}

/// A tail split's programs ([`gemm_tail`]) launched in order over the
/// operands `[x, w]` and the epilogue's into `y`: the whole waves' tiles
/// into `y`, the tail's partials, then their merge into the rest of `y`.
pub(crate) fn launched_tail(
    programs: Candidate,
    tail: &TailSpec,
    [x, w]: [&Tensor; 2],
    bias: Option<&Tensor>,
    residual: Option<&Tensor>,
    y: &Tensor,
) -> Result<Tensor> {
    let launch = |(prog, lowering): (_, Lowering), ins: &[Option<&Tensor>]| {
        let ins: Vec<&Tensor> = ins.iter().copied().flatten().collect();
        launch::graph_launch(prog, &lowering, &ins).context(LaunchSnafu { op: OP })
    };
    let mut programs = programs.into_iter();
    let y = match tail.plan.waves {
        Some(_) => launch(programs.next().expect("the waves"), &[Some(x), Some(w), bias, residual, Some(y)])?,
        None => y.clone(),
    };
    let partial = kernel_buffer(&[SInt::Const(tail.partials())], DType::Float32).context(GraphSnafu { op: OP })?;
    let partial = launch(programs.next().expect("the tail's partials"), &[Some(x), Some(w), Some(&partial)])?;
    launch(programs.next().expect("the merge"), &[Some(&partial), bias, residual, Some(&y)])
}

/// The programs `spec` launches in order with their lowerings: its GEMM,
/// or its tail split's ([`gemm_tail`]).
pub(crate) fn programs(spec: &Admitted<GemmSpec>, dtype: &DType, planner: &Planner) -> Candidate {
    let lowering = spec.cfg().lowering(&planner.device);
    let Some(tail) = tail(spec, planner) else { return vec![(typed!(dtype, gemm, spec), lowering)] };
    let mut programs: Vec<_> = typed!(dtype, gemm_tail, &tail).into_iter().map(|p| (p, lowering.clone())).collect();
    programs.last_mut().expect("the merge").1 = Lowering::new(&planner.device);
    programs
}

/// The tail split `spec`'s config runs, as the planner plans it for the shape.
fn tail<'t>(spec: &Admitted<'t, GemmSpec>, planner: &Planner) -> Option<TailSpec<'t>> {
    let cfg = spec.cfg();
    cfg.tail.filter(|_| spec.batch == Batch::Static(1))?;
    let plan = spec.problem().tail_plan(planner, &cfg).expect("an admitted tail split runs its problem");
    Some(TailSpec { gemm: spec.clone(), plan })
}

/// `spec` under the config [`linear`] launches for its shape and epilogue:
/// the tuner's pick among `ranked` (the planner's ranking for `spec`),
/// through `store`, admitted.
pub fn tuned<'t>(
    store: &TuneStore,
    planner: &'t Planner,
    dtype: DType,
    ranked: &[GemmCfg],
    spec: GemmSpec,
) -> Admitted<'t, GemmSpec> {
    let salt = (&spec.batch, spec.epilogue);
    tune(store, planner, dtype, ranked, (OP, salt), spec.clone())
}

/// [`tuned`] as the device tests and GEMM probes tune: under the kernel's
/// name and [`DEVICE_TEST`].
pub fn probed<'t>(
    store: &TuneStore,
    planner: &'t Planner,
    dtype: DType,
    ranked: &[GemmCfg],
    spec: GemmSpec,
) -> Admitted<'t, GemmSpec> {
    tune(store, planner, dtype, ranked, ("gemm", DEVICE_TEST), spec)
}

fn tune<'t>(
    store: &TuneStore,
    planner: &'t Planner,
    dtype: DType,
    ranked: &[GemmCfg],
    (op, salt): (&'static str, impl Debug),
    spec: GemmSpec,
) -> Admitted<'t, GemmSpec> {
    let shape = [spec.batch.capacity(), spec.m, spec.n, spec.k];
    let problem = spec.problem();
    let build = |cfg: GemmCfg| programs(&spec.clone().admit_ranked(planner, cfg), &dtype, planner);
    let cfg = Tuner::new(store, &planner.device, dtype.clone(), &shape, salt, build)
        .floored(|cfg| planner.floor_of(&problem, cfg))
        .tuned(op, ranked, <[_]>::to_vec);
    spec.admit_ranked(planner, cfg)
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
