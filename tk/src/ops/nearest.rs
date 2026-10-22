//! Each query row's cheapest candidates: nearest-neighbour search, k-means
//! assignment, maximum inner product, with optional score floors and group
//! exclusion.

use snafu::{ResultExt, ensure};
use svod_dtype::{DType, ScalarDType};
use svod_ir::SInt;
use svod_tensor::Tensor;

use super::config::Planner;
use super::shape::{self, BatchVar, bound_var, extent, shape_of};
use super::{
    DtypeSnafu, GraphSnafu, LaunchSnafu, NoScoresSnafu, Result, ShapeSnafu, Tuner, UnboundSnafu, fmt_shape, output,
    typed,
};
use crate::ir::Var;
pub use crate::kernels::nearest::Metric;
use crate::kernels::nearest::{
    Floors, Live, MergeSpec, NearestCfg, NearestSpec, RescoreSpec, SelectSpec, nearest as kernel, nearest_merge,
    nearest_rescore, nearest_select, scratch, select_blocks, share_stride,
};
use crate::launch;
use crate::tune::TuneStore;

const OP: &str = "nearest";

/// How pairs score and which pairs count.
#[derive(Clone, Copy, Debug, Default)]
pub struct Nearest<'a> {
    pub metric: Metric,
    /// `[N]` lower bounds per row: a pair scores at least its row's.
    pub row_floor: Option<&'a Tensor>,
    /// `[M]` lower bounds per candidate: a pair scores at least its candidate's.
    pub col_floor: Option<&'a Tensor>,
    /// `[N]` and `[M]` integer group ids: a pair of equal ids is skipped. A
    /// self-join (`c` is `x`) under groups skips each row itself.
    pub groups: Option<(&'a Tensor, &'a Tensor)>,
}

impl Nearest<'_> {
    fn floors(&self) -> Floors {
        Floors { row: self.row_floor.is_some(), col: self.col_floor.is_some() }
    }
}

/// For every row of `x [N, D]`, the least score over the candidates
/// `c [M, D]` that [`Nearest`] admits, `max(metric, row_floor[i],
/// col_floor[j])`, and the smallest candidate index scoring it: `([N]` f32
/// scores, `[N]` i32 indices`)`. A row with no admissible candidate scores
/// `+∞` at index −1. A self-join passes the same tensor as `x` and `c`.
/// `N` may be a bound variable (a symbolic batch): the kernels are built for
/// its bound and take the rows bound at launch, the outputs `[N]` alike.
pub fn nearest(x: &Tensor, c: &Tensor, opts: Nearest) -> Result<(Tensor, Tensor)> {
    let dtype = x.dtype();
    picked(x, c, 1, opts, None, |planner, ranked, spec| tuned(TuneStore::global(), planner, dtype, ranked, spec))
}

/// For every row of `x [N, D]`, its `k` least scores over the candidates
/// [`nearest`] admits, ascending, and the candidate indices scoring them, an
/// equal score at the smaller index: `([N, k]` f32 scores, `[N, k]` i32
/// indices`)`. The slots past a row's admissible candidates score `+∞` at
/// index −1. `k` is a count or a bound variable: the kernels keep lists of
/// its bound (whose length picks how they are kept) and end them at the
/// value bound at launch, the outputs `[N, k]` views of `[N, bound]` ones.
pub fn k_nearest(x: &Tensor, c: &Tensor, k: impl Into<SInt>, opts: Nearest) -> Result<(Tensor, Tensor)> {
    let (k, dtype) = (k.into(), x.dtype());
    let (score, index) = picked(x, c, k.clone(), opts, None, |planner, ranked, spec| {
        tuned(TuneStore::global(), planner, dtype, ranked, spec)
    })?;
    if k != SInt::Const(1) {
        return Ok((score, index));
    }
    let column = |t: Tensor| t.try_unsqueeze(1).context(GraphSnafu { op: OP });
    Ok((column(score)?, column(index)?))
}

/// Rows held as 8-bit integers for [`k_nearest_reranked`]'s search: each
/// row `v` as `round(v / s)` clamped to ±127, `s = max|v| / 127` (1 for a
/// row of zeros).
#[derive(Clone, Debug)]
pub struct Quantized {
    /// `[N, D]` i8.
    pub rows: Tensor,
    /// `[N]` f32.
    pub scale: Tensor,
}

impl Quantized {
    /// `x [N, D]` (of any float type) quantized as graph ops: realize it
    /// once for a corpus searched again and again.
    pub fn new(x: &Tensor) -> Result<Self> {
        let quantized = || -> svod_tensor::error::Result<Self> {
            let x = x.cast(DType::Float32);
            let top = x.abs().max(1isize)?;
            let zero = top.try_eq(0.0f32)?;
            let scale = Tensor::const_(1.0f32, DType::Float32).where_(&zero, &top.try_div(127.0f32)?)?;
            let q = x.try_div(&scale.try_unsqueeze(1)?)?.round();
            let rows = q.clamp().min(-127.0f32).max(127.0f32).call()?.cast(DType::Int8);
            Ok(Self { rows, scale })
        };
        quantized().context(GraphSnafu { op: OP })
    }
}

/// Scores past `k` a row's search keeps for [`k_nearest_reranked`]'s exact
/// pass by default: on 33 342 Qwen3 embeddings (L2-normalized, depths
/// 128–768, `k` 1, 16 and 64) the 8-bit search held every exact neighbour
/// within `k + 13`.
pub const MARGIN: usize = 16;

/// How [`k_nearest_reranked`] searches before its exact pass.
#[derive(Clone, Copy, Debug)]
pub struct Rerank<'a> {
    /// Scores past `k` a row's search keeps for the exact pass.
    pub margin: usize,
    /// `x` and `c` quantized beforehand ([`Quantized::new`]), else here.
    pub x8: Option<&'a Quantized>,
    pub c8: Option<&'a Quantized>,
}

impl Default for Rerank<'_> {
    fn default() -> Self {
        Self { margin: MARGIN, x8: None, c8: None }
    }
}

/// [`k_nearest`] of rows of any float type (f32, say) scored in f32: a
/// search over the rows in 8 bits ([`Quantized`]: each row's integers and
/// scale, the norms the rows' own) or in f16, whichever the device runs
/// faster (as tuned; as ranked with tuning off), keeps each row's `k +
/// margin` least, and their scores in f32 from the rows as given decide the
/// `k` it returns. A neighbour the search ranks past `k + margin` is
/// missed; ties among candidates the search ordered apart keep its order.
/// Where neither search has a kernel the f32 graph lists the rows.
pub fn k_nearest_reranked(
    x: &Tensor,
    c: &Tensor,
    k: impl Into<SInt>,
    opts: Nearest,
    rerank: Rerank,
) -> Result<(Tensor, Tensor)> {
    let k = k.into();
    let (top, k_var) = list_len(&k)?;
    let graph_err = || GraphSnafu { op: OP };
    let shape = |t: &Tensor| shape_of(t).context(graph_err());
    let (xs, cs) = (shape(x)?, shape(c)?);
    let (xf, cf) = (x.cast(DType::Float32), c.cast(DType::Float32));
    let live_k = |(score, index): (Tensor, Tensor)| -> Result<(Tensor, Tensor)> {
        let cut = |t: Tensor| t.try_shrink([None, Some((SInt::Const(0), k.clone()))]);
        match k_var {
            None => Ok((score, index)),
            Some(_) => Ok((cut(score).context(graph_err())?, cut(index).context(graph_err())?)),
        }
    };
    let planner = super::planner(&x.device());
    let (Some(planner), [_, _], Some(m)) = (planner, &xs[..], cs.first().and_then(SInt::as_const)) else {
        return live_k(graph_k(&xf, &cf, top, opts).context(graph_err())?);
    };
    let coarse = (top + rerank.margin).min(m);
    // Each search's rows, what scores them, and its pick for the shape.
    let x8 = || rerank.x8.cloned().map_or_else(|| Quantized::new(x), Ok);
    let c8 = || rerank.c8.cloned().map_or_else(|| Quantized::new(c), Ok);
    let searches = [ScalarDType::Int8, ScalarDType::Float16].into_iter().filter_map(|dtype| {
        let (cfg, ns, cost) = searched(&planner, dtype, &xs, &cs, coarse, opts)?;
        Some((dtype, cfg, ns, cost))
    });
    let searches: Vec<_> = searches.collect();
    let timed = searches.iter().all(|s| s.2.is_some());
    let fastest = searches.iter().min_by(|a, b| match timed {
        true => a.2.cmp(&b.2),
        false => a.3.total_cmp(&b.3),
    });
    let Some(&(dtype, cfg, ..)) = fastest else {
        return live_k(graph_k(&xf, &cf, top, opts).context(graph_err())?);
    };
    tracing::debug!(target: super::PLAN, op = OP, ?dtype, ?cfg, coarse, "reranked search");
    let pick = |_: &Planner, _: &[NearestCfg], spec: NearestSpec| NearestSpec { cfg, ..spec };
    let (_, index) = match dtype {
        ScalarDType::Int8 => {
            let (x8, c8) = (x8()?, c8()?);
            let squared = |t: &Tensor| t.square().sum(1isize);
            let x_sq = opts.metric.row_norms().then(|| squared(&xf)).transpose().context(graph_err())?;
            let c_sq = opts.metric.col_norms().then(|| squared(&cf)).transpose().context(graph_err())?;
            let scaled = Scaled { x_scale: x8.scale.clone(), c_scale: c8.scale.clone(), x_sq, c_sq };
            picked(&x8.rows, &c8.rows, coarse, opts, Some(&scaled), pick)?
        }
        _ => picked(&x.cast(DType::Float16), &c.cast(DType::Float16), coarse, opts, None, pick)?,
    };
    let index = if coarse == 1 { index.try_unsqueeze(1).context(graph_err())? } else { index };
    let index = index.contiguous();
    let x_var = extent(&xs).and_then(|(_, v)| v);
    live_k(rescored(&planner, (&xf, &cf, &index), top, opts, &k_var, &x_var)?)
}

/// The search over `dtype` rows of `xs` against `cs` keeping `coarse` a
/// row, where it has a kernel: the tuner's pick, its device ns where timed,
/// and its predicted ns.
fn searched(
    planner: &Planner,
    dtype: ScalarDType,
    xs: &[SInt],
    cs: &[SInt],
    coarse: usize,
    opts: Nearest,
) -> Option<(NearestCfg, Option<u64>, f64)> {
    let ((x_ext, x_var), c_ext) = (extent(xs).unzip(), extent(cs).map(|(e, _)| e));
    let (metric, floors, groups) = (opts.metric, opts.floors(), opts.groups.is_some());
    let dtypes = [DType::Scalar(dtype), DType::Scalar(dtype)];
    let plan = shape::nearest(Some(planner), &dtypes, x_ext.as_ref(), c_ext.as_ref(), metric, floors, groups, coarse);
    let cfgs = super::kernel(OP, plan)?;
    let ([n, d], m) = (x_ext?.dims[..].try_into().ok()?, c_ext?.dims[0]);
    let rows = x_var.flatten().map(|v| (Var { name: v.name, min: v.min, max: v.max }, 0));
    let live = Live { rows, k: None };
    let d = planner.nearest_depth(d, dtype)?;
    let spec = NearestSpec { n, m, d, metric, floors, groups, k: coarse, cfg: cfgs[0], live };
    let (spec, ns) = timed(TuneStore::global(), planner, DType::Scalar(dtype), &cfgs, spec);
    let cost = problem(&spec, dtype).admit(planner, &spec.cfg).map_or(f64::INFINITY, |(ns, _)| ns);
    Some((spec.cfg, ns, cost))
}

/// The exact pass of a reranked search over the candidates `index [N, K]`
/// (i32, −1 for none) of the rows of `x [N, D]` among `c [M, D]`: each
/// pair scored in f32 from the rows as given ([`nearest_rescore`]), then
/// each row's `top` least listed ([`listed`]). `x_var` binds the rows where
/// they vary, `k_var` the list's length.
pub(crate) fn rescored(
    planner: &Planner,
    (x, c, index): (&Tensor, &Tensor, &Tensor),
    top: usize,
    opts: Nearest,
    k_var: &Option<Var>,
    x_var: &Option<BatchVar>,
) -> Result<(Tensor, Tensor)> {
    let graph_err = || GraphSnafu { op: OP };
    let (xs, cs, is) = (shape_of(x), shape_of(c), shape_of(index));
    let (xs, cs, is) = (xs.context(graph_err())?, cs.context(graph_err())?, is.context(graph_err())?);
    let consts = |dims: &[SInt]| dims.iter().map(SInt::as_const).collect::<Option<Vec<_>>>();
    let (Some([m, d]), Some(cols)) = (consts(&cs).and_then(|v| <[usize; 2]>::try_from(v).ok()), is[1].as_const())
    else {
        unreachable!("constant candidates and lists")
    };
    let n = x_var.as_ref().map_or_else(|| xs[0].as_const().expect("constant rows"), |v| v.max as usize);
    let rows = x_var.as_ref().map(|v| (Var { name: v.name.clone(), min: v.min, max: v.max }, 0));
    let spec = RescoreSpec { n, m, d, cols, metric: opts.metric, floors: opts.floors(), live: Live { rows, k: None } };
    let f32 = |t: &Tensor| t.cast(DType::Float32);
    let floors = [opts.row_floor.map(f32), opts.col_floor.map(f32)];
    let scores = output(&[n, cols], x_var, DType::Float32).context(graph_err())?;
    let ins: Vec<&Tensor> = [x, c, index].into_iter().chain(floors.iter().flatten()).chain([&scores]).collect();
    let scores = launch::graph_launch(nearest_rescore(&spec), &spec.lowering(&planner.device), &ins)
        .context(LaunchSnafu { op: OP })?;
    listed(planner, &scores, index, top, k_var, x_var)
}

/// Each row's `top` least of `scores [N, K]` (scores of the candidates
/// `index [N, K]`), ascending, by [`nearest_select`] where it lists them
/// (an equal score at the earlier column), with their candidates' indices,
/// −1 past the admissible ones.
pub(crate) fn listed(
    planner: &Planner,
    scores: &Tensor,
    index: &Tensor,
    top: usize,
    k_var: &Option<Var>,
    x_var: &Option<BatchVar>,
) -> Result<(Tensor, Tensor)> {
    use crate::kernels::nearest::{SelectCfg, select_runs};
    let graph_err = || GraphSnafu { op: OP };
    let shape = shape_of(scores).context(graph_err())?;
    let (Some(n), Some(cols)) =
        (x_var.as_ref().map_or(shape[0].as_const(), |v| Some(v.max as usize)), shape[1].as_const())
    else {
        unreachable!("a bound row count and a constant list")
    };
    let cfg = SelectCfg { warps: 4, queue: select_runs(top), filter: false };
    let rows = x_var.as_ref().map(|v| (Var { name: v.name.clone(), min: v.min, max: v.max }, 0));
    let live = Live { rows, k: k_var.clone().filter(|_| top > 1) };
    let wide = cfg.wide(planner.device.target.wave);
    let spec =
        SelectSpec { n, cols, k: top, metric: Metric::NegDot, first: true, cfg, live, wide, filter: None, shares: 1 };
    // Lists left in the units given: the scores as they are.
    let chunk = Tensor::from_slice([0i32, 0]);
    let outs = [output(&[n, top], x_var, DType::Float32), output(&[n, top], x_var, DType::Int32)];
    let [score_t, at_t] = outs.map(|t| t.context(graph_err()));
    let (score_t, at_t) = (score_t?, at_t?);
    let ins = [&scores.contiguous(), &chunk, &score_t, &at_t];
    let mut outs = launch::graph_launch_all(nearest_select(&spec), &cfg.lowering(&planner.device), &ins)
        .context(LaunchSnafu { op: OP })?;
    let (at, score) = (outs.pop().expect("the columns"), outs.pop().expect("the scores"));
    let candidates = || -> svod_tensor::error::Result<Tensor> {
        let found = index.gather(1, &at.maximum(0i32)?)?;
        Tensor::const_(-1i32, DType::Int32).where_(&at.try_lt(0i32)?, &found)
    };
    Ok((score, candidates().context(graph_err())?))
}

/// A list's length: its bound and, bound to a variable, the variable.
fn list_len(k: &SInt) -> Result<(usize, Option<Var>)> {
    if let Some(k) = k.as_const() {
        ensure!(k > 0, NoScoresSnafu { op: OP });
        return Ok((k, None));
    }
    let BatchVar { name, min, max, .. } =
        bound_var(k).ok_or_else(|| UnboundSnafu { op: OP, got: k.to_string() }.build())?;
    ensure!(min > 0, NoScoresSnafu { op: OP });
    Ok((max as usize, Some(Var { name, min, max })))
}

/// What 8-bit rows score with: each row's scale (`[N]`, `[M]` f32), and
/// the squared norms of the rows they stand for, where the metric takes
/// them.
#[derive(Clone, Debug)]
pub(crate) struct Scaled {
    pub x_scale: Tensor,
    pub c_scale: Tensor,
    pub x_sq: Option<Tensor>,
    pub c_sq: Option<Tensor>,
}

/// [`k_nearest`] (`[N]` outputs at `k = 1`, as [`nearest`]'s) launching the
/// spec `pick` makes of the planned one, given the planner and the shape
/// gate's ranking. 8-bit rows score under `scaled`, else as the integers
/// they hold (each scale 1, the norms theirs).
pub(crate) fn picked(
    x: &Tensor,
    c: &Tensor,
    k: impl Into<SInt>,
    opts: Nearest,
    scaled: Option<&Scaled>,
    pick: impl FnOnce(&Planner, &[NearestCfg], NearestSpec) -> NearestSpec,
) -> Result<(Tensor, Tensor)> {
    let k = k.into();
    let (top, k_var) = list_len(&k)?;
    let shape = |t: &Tensor| shape_of(t).context(GraphSnafu { op: OP });
    let (xs, cs) = (shape(x)?, shape(c)?);
    let shape_err =
        |operand, got: &[SInt], expected: String| ShapeSnafu { op: OP, operand, got: fmt_shape(got), expected };
    ensure!(xs.len() == 2, shape_err("x", &xs, "[N, D]".into()));
    ensure!(cs.len() == 2 && cs[1] == xs[1], shape_err("c", &cs, format!("[M, {}]", xs[1])));
    ensure!(c.dtype() == x.dtype(), DtypeSnafu { op: OP, operand: "c", got: c.dtype(), want: x.dtype() });
    let (row_group, col_group) = opts.groups.unzip();
    let per_row = [("row floor", opts.row_floor, &xs[0]), ("row group", row_group, &xs[0])];
    let per_col = [("col floor", opts.col_floor, &cs[0]), ("col group", col_group, &cs[0])];
    for (operand, t, rows) in per_row.into_iter().chain(per_col) {
        let Some(t) = t else { continue };
        let got = shape(t)?;
        ensure!(got == [rows.clone()], shape_err(operand, &got, fmt_shape(std::slice::from_ref(rows))));
    }
    // A list of a live length is a view of the list of its bound.
    let live_k = |(score, index): (Tensor, Tensor)| -> Result<(Tensor, Tensor)> {
        if k_var.is_none() {
            return Ok((score, index));
        }
        let cut = |t: Tensor| {
            let t = if top == 1 { t.try_unsqueeze(1)? } else { t };
            t.try_shrink([None, Some((SInt::Const(0), k.clone()))])
        };
        Ok((cut(score).context(GraphSnafu { op: OP })?, cut(index).context(GraphSnafu { op: OP })?))
    };

    let ((x_ext, x_var), c_ext) = (extent(&xs).unzip(), extent(&cs).map(|(e, _)| e));
    let x_var = x_var.flatten();
    let planner = super::planner(&x.device());
    let (metric, floors, groups) = (opts.metric, opts.floors(), opts.groups.is_some());
    let dtypes = [x.dtype(), c.dtype()];
    let plan = shape::nearest(planner.as_deref(), &dtypes, x_ext.as_ref(), c_ext.as_ref(), metric, floors, groups, top);
    let Some(cfgs) = super::kernel(OP, plan) else {
        let graph = if top == 1 { graph(x, c, opts) } else { graph_k(x, c, top, opts) };
        return live_k(graph.context(GraphSnafu { op: OP })?);
    };
    let planner = planner.expect("planned");
    let ([n, d], m) = (x_ext.expect("planned").dims[..].try_into().expect("rank 2"), c_ext.expect("planned").dims[0]);
    let depth = planner.nearest_depth(d, x.dtype().scalar().expect("planned")).expect("planned");
    let graph_err = GraphSnafu { op: OP };
    let padded = |t: &Tensor| if depth == d { Ok(t.clone()) } else { t.try_pad(&[(0, 0), (0, (depth - d) as isize)]) };
    let squared = |t: &Tensor| t.cast(DType::Float32).square().sum(1isize);
    let (xp, cp) = (padded(x).context(graph_err)?, padded(c).context(graph_err)?);
    let int8 = x.dtype().scalar() == Some(ScalarDType::Int8);
    let given = |t: Option<&Tensor>| t.cloned().map(Ok);
    let c_sq = metric.col_norms().then(|| given(scaled.and_then(|s| s.c_sq.as_ref())).unwrap_or_else(|| squared(c)));
    let x_sq = metric.row_norms().then(|| given(scaled.and_then(|s| s.x_sq.as_ref())).unwrap_or_else(|| squared(x)));
    let (c_sq, x_sq) = (c_sq.transpose().context(graph_err)?, x_sq.transpose().context(graph_err)?);
    // 8-bit rows' scales: given, else each 1.
    let unit = |rows: usize| Tensor::full(&[rows], 1.0f32, DType::Float32);
    let (x_scale, c_scale) = match (int8, scaled) {
        (false, _) => (None, None),
        (true, Some(s)) => (Some(s.x_scale.clone()), Some(s.c_scale.clone())),
        (true, None) => (Some(unit(n)), Some(unit(m))),
    };
    let f32 = |t: &Tensor| t.cast(DType::Float32);
    let (row_floor, col_floor) = (opts.row_floor.map(f32), opts.col_floor.map(f32));
    let (row_group, col_group) = opts.groups.map(|(r, c)| (r.cast(DType::Int32), c.cast(DType::Int32))).unzip();
    let live = Live {
        rows: x_var.as_ref().map(|v| (Var { name: v.name.clone(), min: v.min, max: v.max }, 0)),
        k: k_var.clone().filter(|_| top > 1),
    };
    let spec = NearestSpec { n, m, d: depth, metric, floors, groups, k: top, cfg: cfgs[0], live };
    let spec = pick(&planner, &cfgs, spec);
    let rows_in = [Some(&xp), x_scale.as_ref(), x_sq.as_ref(), row_floor.as_ref(), row_group.as_ref()];
    let cols_in = [Some(&cp), c_scale.as_ref(), c_sq.as_ref(), col_floor.as_ref(), col_group.as_ref()];
    if spec.cfg.select.is_some() {
        return live_k(matrix(&planner, x.dtype(), &spec, rows_in, cols_in, &x_var)?);
    }
    let splits = spec.cfg.splits;
    // Whole, a row's `k` scores; split, every share's runs of `n` rows.
    let (rows, var): (&[usize], _) = match (splits, top) {
        (1, 1) => (&[n], &x_var),
        (1, _) => (&[n, top], &x_var),
        _ => (&[top * splits * share_stride(n)], &None),
    };
    let score_t = output(rows, var, DType::Float32).context(graph_err)?;
    let index_t = output(rows, var, DType::Int32).context(graph_err)?;
    let ins: Vec<&Tensor> = [Some(&xp), Some(&cp), x_scale.as_ref(), c_scale.as_ref(), c_sq.as_ref(), x_sq.as_ref()]
        .into_iter()
        .chain([row_floor.as_ref(), col_floor.as_ref(), row_group.as_ref(), col_group.as_ref()])
        .chain([Some(&score_t), Some(&index_t)])
        .flatten()
        .collect();
    let mut outs =
        launch::graph_launch_all(typed!(x.dtype(), kernel, &spec, int8), &spec.cfg.lowering(&planner.device), &ins)
            .context(LaunchSnafu { op: OP })?;
    let (index, score) = (outs.pop().expect("the index"), outs.pop().expect("the score"));
    if splits == 1 {
        return live_k((score, index));
    }
    let merge = merge_of(&spec);
    let rows: &[usize] = if top == 1 { &[n] } else { &[n, top] };
    let score_t = output(rows, &x_var, DType::Float32).context(graph_err)?;
    let index_t = output(rows, &x_var, DType::Int32).context(graph_err)?;
    let mut outs = launch::graph_launch_all(
        nearest_merge(&merge),
        &merge.lowering(&planner.device),
        &[&score, &index, &score_t, &index_t],
    )
    .context(LaunchSnafu { op: OP })?;
    let index = outs.pop().expect("the index");
    live_k((outs.pop().expect("the score"), index))
}

/// The merge of a split `spec`'s shares.
fn merge_of(spec: &NearestSpec) -> MergeSpec {
    MergeSpec { n: spec.n, splits: spec.cfg.splits, metric: spec.metric, k: spec.k, live: spec.live.clone() }
}

/// One block of a score matrix's launches ([`launches`]).
struct Block {
    /// Its first row and candidate.
    r0: usize,
    c0: usize,
    /// Its scores and each trip's least a row ([`kernel`]).
    scores: NearestSpec,
    /// The `k` least of those leasts, whose `k`-th bounds each row's
    /// ([`nearest_select`]), where they are many enough to pass over most.
    bound: Option<SelectSpec>,
    /// Its scores listed into its rows' lists; in shares, merged after.
    list: SelectSpec,
}

/// Search bound: the blocks per SM a filtered listing's shares reach at
/// least, its steps' loads waiting on memory one after another.
const SELECT_FILL: usize = 8;

/// A score-matrix `spec`'s launches in order on `planner`'s device: per
/// block of [`select_blocks`] (an eighth of its memory at most, [`scratch`]),
/// its scores', where filtered its bound's, and its
/// listing's. Where the rows are live a panel takes those of the bound rows
/// past its first, if any. A filtered listing of a whole row (a share's
/// list fills only from scores at most the bound) splits its steps into
/// shares until its blocks fill the SMs [`SELECT_FILL`] deep.
fn launches(spec: &NearestSpec, planner: &Planner) -> Vec<Block> {
    let NearestSpec { n, m, k, metric, cfg, .. } = *spec;
    let select = cfg.select.expect("a score matrix");
    let sms = planner.sms();
    let (panel, block) = select_blocks(n, m, k, &cfg, scratch(planner.device.target.memory_bytes));
    let mut out = vec![];
    for r0 in (0..n).step_by(panel) {
        let live = Live { rows: spec.live.rows.clone().map(|(v, _)| (v, r0)), k: spec.live.k.clone() };
        for c0 in (0..m).step_by(block) {
            let (nr, mc) = (panel.min(n - r0), block.min(m - c0));
            let scores = NearestSpec { n: nr, m: mc, live: live.clone(), ..spec.clone() };
            let filter = select.filter.then_some(cfg.bkv);
            let listing = |cols, first, filter, shares| SelectSpec {
                n: nr,
                cols,
                k,
                metric,
                first,
                cfg: select,
                live: live.clone(),
                wide: select.wide(planner.device.target.wave),
                filter,
                shares,
            };
            let bound = filter.map(|_| listing(mc.div_ceil(cfg.bkv), true, None, 1));
            let blocks = nr.div_ceil(select.warps as usize);
            let shares = match filter.is_some() && mc == m {
                true => {
                    let steps = listing(mc, true, filter, 1).steps();
                    (SELECT_FILL * sms).div_ceil(blocks).next_power_of_two().min(1 << steps.ilog2())
                }
                false => 1,
            };
            out.push(Block { r0, c0, scores, bound, list: listing(mc, c0 == 0, filter, shares) });
        }
    }
    out
}

/// [`picked`]'s lists through a score matrix: per block of
/// [`select_blocks`], its scores computed, then listed into its rows'
/// lists, filtered by the `k`-th least of its trips' leasts where it has
/// many. `rows` and `cols` are the inputs per row and per candidate in the
/// kernel's order (`x`, then `x_scale`, `x_sq`, the floor and the group;
/// `c`, then `c_scale`, `c_sq`, the floor and the group), where present. Bound rows (`var`) are
/// read whole by every panel, which takes its own of them.
fn matrix(
    planner: &Planner,
    dtype: DType,
    spec: &NearestSpec,
    rows: [Option<&Tensor>; 5],
    cols: [Option<&Tensor>; 5],
    var: &Option<BatchVar>,
) -> Result<(Tensor, Tensor)> {
    let NearestSpec { n, m, k, cfg, .. } = *spec;
    let select = cfg.select.expect("a score matrix");
    let graph_err = || GraphSnafu { op: OP };
    // `len` of `inputs` from `start`, of `whole` each.
    let part = |inputs: [Option<&Tensor>; 5], start: usize, len: usize, whole: usize| {
        inputs.map(|t| {
            t.map(|t| if len == whole { Ok(t.clone()) } else { t.narrow(0, start, len).map(|t| t.contiguous()) })
                .transpose()
        })
    };
    let lowered = |p: [svod_tensor::error::Result<Option<Tensor>>; 5]| -> Result<[Option<Tensor>; 5]> {
        let mut out = [None, None, None, None, None];
        for (o, t) in out.iter_mut().zip(p) {
            *o = t.context(graph_err())?;
        }
        Ok(out)
    };
    let listing = select.lowering(&planner.device);
    let mut lists: Vec<Vec<Tensor>> = vec![];
    let mut r = [None, None, None, None, None];
    for Block { r0, c0, scores: a, bound, list: b } in launches(spec, planner) {
        let (nr, mc) = (a.n, a.m);
        if c0 == 0 {
            r = match var {
                Some(_) => rows.map(|t| t.cloned()),
                None => lowered(part(rows, r0, nr, n))?,
            };
            lists.push(vec![]);
        }
        let c = lowered(part(cols, c0, mc, m))?;
        let groups = select.filter.then(|| mc.div_ceil(cfg.bkv));
        let outs = [Some(nr * mc), groups.map(|g| nr * g)]
            .into_iter()
            .flatten()
            .map(|len| output(&[len], &None, DType::Float32));
        let outs = outs.collect::<svod_tensor::error::Result<Vec<_>>>().context(graph_err())?;
        // The kernel's order: x, c, the scales, c_sq, x_sq, the floors, the groups.
        let order = [&r[0], &c[0], &r[1], &c[1], &c[2], &r[2], &r[3], &c[3], &r[4], &c[4]];
        let ins: Vec<&Tensor> = order.into_iter().flatten().chain(&outs).collect();
        let lowering = a.cfg.lowering(&planner.device);
        let mut outs = launch::graph_launch_all(typed!(dtype, kernel, &a, int8), &lowering, &ins)
            .context(LaunchSnafu { op: OP })?;
        let mins = groups.map(|_| outs.pop().expect("the minima"));
        let scores = outs.pop().expect("the scores");
        // One panel's lists are the outputs, shaped with the live rows.
        let rows_var = if nr == n { var } else { &None };
        let lists_out = || -> Result<Vec<Tensor>> {
            [output(&[nr, k], rows_var, DType::Float32), output(&[nr, k], rows_var, DType::Int32)]
                .into_iter()
                .collect::<svod_tensor::error::Result<Vec<_>>>()
                .context(graph_err())
        };
        let filter = match bound.zip(mins) {
            Some((bound, mins)) => {
                // The leasts listed in the units compared.
                let (unit, outs) = (Tensor::from_slice([0i32, 0]), lists_out()?);
                let ins: Vec<&Tensor> = [&mins, &unit].into_iter().chain(&outs).collect();
                let mut outs =
                    launch::graph_launch_all(nearest_select(&bound), &listing, &ins).context(LaunchSnafu { op: OP })?;
                outs.truncate(1);
                vec![mins, outs.pop().expect("the bound")]
            }
            None => vec![],
        };
        let chunk = Tensor::from_slice([c0 as i32, i32::from(c0 + mc == m)]);
        let list = lists.last_mut().expect("a panel");
        let outs = match b.shares {
            1 => lists_out()?,
            shares => {
                let parts = [shares * share_stride(nr) * k];
                let part = |t| output(&parts, &None, t).context(graph_err());
                vec![part(DType::Float32)?, part(DType::Int32)?]
            }
        };
        let ins: Vec<&Tensor> = [&scores, &chunk].into_iter().chain(list.iter()).chain(&filter).chain(&outs).collect();
        let mut outs = launch::graph_launch_all(nearest_select(&b), &listing, &ins).context(LaunchSnafu { op: OP })?;
        let mut parts = outs.split_off(outs.len() - 2);
        if b.shares > 1 {
            let merge = MergeSpec { n: nr, splits: b.shares, metric: b.metric, k, live: b.live.clone() };
            let merged = lists_out()?;
            let ins: Vec<&Tensor> = parts.iter().chain(&merged).collect();
            let mut outs = launch::graph_launch_all(nearest_merge(&merge), &merge.lowering(&planner.device), &ins)
                .context(LaunchSnafu { op: OP })?;
            parts = outs.split_off(outs.len() - 2);
        }
        *list = parts;
    }
    let cat = |j: usize| {
        let t = match lists.len() {
            1 => lists[0][j].clone(),
            _ => Tensor::cat(&lists.iter().map(|l| &l[j]).collect::<Vec<_>>(), 0).context(graph_err())?,
        };
        match var {
            Some(v) if lists.len() > 1 => {
                t.try_shrink([Some((SInt::Const(0), v.dim.clone())), None]).context(graph_err())
            }
            _ => Ok(t),
        }
    };
    Ok((cat(0)?, cat(1)?))
}

/// What keys a nearest tuning besides its shape: the metric, floors and
/// groups, and where the rows or the list are live, which (the kernels take
/// them at launch).
pub(crate) struct Salt<'a>(pub(crate) &'a NearestSpec);

impl std::fmt::Debug for Salt<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let NearestSpec { metric, floors, groups, ref live, .. } = *self.0;
        write!(f, "{:?}", (metric, floors, groups))?;
        if *live != Live::default() {
            write!(f, " {live:?}")?;
        }
        Ok(())
    }
}

/// `spec` under the config [`nearest`] launches for its shape: the tuner's
/// pick among `ranked` (the shape gate's list), through `store`. Live rows
/// and lists tune at their bounds, keyed apart.
pub fn tuned(
    store: &TuneStore,
    planner: &Planner,
    dtype: DType,
    ranked: &[NearestCfg],
    spec: NearestSpec,
) -> NearestSpec {
    timed(store, planner, dtype, ranked, spec).0
}

/// [`tuned`] with the pick's device ns as the tuner last timed it, where it
/// did (not with tuning off).
pub(crate) fn timed(
    store: &TuneStore,
    planner: &Planner,
    dtype: DType,
    ranked: &[NearestCfg],
    spec: NearestSpec,
) -> (NearestSpec, Option<u64>) {
    // A row's least alone keys as it did before the lists existed.
    let shape = [spec.n, spec.m, spec.d, spec.k];
    let shape = if spec.k == 1 { &shape[..3] } else { &shape[..] };
    let build = |cfg: NearestCfg| programs(planner, dtype.clone(), &NearestSpec { cfg, ..spec.clone() });
    let p = problem(&spec, dtype.scalar().expect("a kernel dtype"));
    let expand = |admitted: &[NearestCfg]| planner.nearest_splits(&p, admitted);
    let mut tuner = Tuner::new(store, &planner.device, dtype.clone(), shape, Salt(&spec), build);
    let (cfg, ns) = tuner.tuned_timed(OP, ranked, expand);
    (NearestSpec { cfg, ..spec }, ns)
}

/// The programs `spec` launches in order: the kernel, and split, the merge
/// of its shares.
pub(crate) fn programs(planner: &Planner, dtype: DType, spec: &NearestSpec) -> crate::tune::Candidate {
    if let Some(select) = spec.cfg.select {
        // Every block's scores, bound and listing, as [`matrix`] launches them.
        let listing = select.lowering(&planner.device);
        let blocks = launches(spec, planner).into_iter().flat_map(|Block { scores, bound, list, .. }| {
            let scores = (typed!(dtype.clone(), kernel, &scores, int8), spec.cfg.lowering(&planner.device));
            let bound = bound.map(|b| (nearest_select(&b), listing.clone()));
            let merge = (list.shares > 1).then(|| {
                let merge = MergeSpec {
                    n: list.n,
                    splits: list.shares,
                    metric: list.metric,
                    k: list.k,
                    live: list.live.clone(),
                };
                (nearest_merge(&merge), merge.lowering(&planner.device))
            });
            [Some(scores), bound, Some((nearest_select(&list), listing.clone())), merge].into_iter().flatten()
        });
        return blocks.collect();
    }
    let mut programs = vec![(typed!(dtype, kernel, spec, int8), spec.cfg.lowering(&planner.device))];
    if spec.cfg.splits > 1 {
        let merge = merge_of(spec);
        programs.push((nearest_merge(&merge), merge.lowering(&planner.device)));
    }
    programs
}

/// The planner's problem of `spec`.
pub(crate) fn problem(spec: &NearestSpec, dtype: ScalarDType) -> crate::ops::config::Nearest {
    let NearestSpec { n, m, d, metric, floors, groups, k, .. } = *spec;
    crate::ops::config::Nearest { n, m, d, dtype, metric, floors, groups, k }
}

/// The scores of every pair in f32 (L2 compared squared against its floors
/// clamped at zero and squared, the square root taken of the row's least,
/// as the kernel orders them), each row's least and its first index, −1
/// where it is `+∞`.
pub(crate) fn graph(x: &Tensor, c: &Tensor, opts: Nearest) -> svod_tensor::error::Result<(Tensor, Tensor)> {
    let s = scored(x, c, opts)?;
    let best = s.min(1isize)?;
    let none = best.try_eq(f32::INFINITY)?;
    let index = Tensor::const_(-1i32, DType::Int32).where_(&none, &s.argmin(1)?)?;
    let best = if opts.metric == Metric::L2 { best.try_sqrt()? } else { best };
    Ok((best, index))
}

/// [`graph`]'s `k` least scores a row and their indices, `[N, k]`: the
/// least, then the least of the rest once each one found is masked out.
pub(crate) fn graph_k(x: &Tensor, c: &Tensor, k: usize, opts: Nearest) -> svod_tensor::error::Result<(Tensor, Tensor)> {
    // Held once: every slot reads the scores again.
    let mut s = scored(x, c, opts)?.contiguous();
    let m = c.shape()?[0]
        .as_const()
        .ok_or(svod_tensor::error::ErrorKind::SymbolicShapeUnsupported { operation: "nearest's candidates".into() })?;
    let cols = Tensor::arange(0, Some(m as i64), None)?.cast(DType::Int32).try_unsqueeze(0)?;
    let (mut scores, mut indices) = (vec![], vec![]);
    for slot in 0..k {
        let best = s.min(1isize)?;
        let none = best.try_eq(f32::INFINITY)?;
        let index = Tensor::const_(-1i32, DType::Int32).where_(&none, &s.argmin(1)?)?;
        if slot + 1 < k {
            let found = cols.try_eq(&index.try_unsqueeze(1)?)?;
            s = Tensor::const_(f32::INFINITY, DType::Float32).where_(&found, &s)?.contiguous();
        }
        scores.push(best);
        indices.push(index);
    }
    let stack = |v: Vec<Tensor>| Tensor::stack(&v.iter().collect::<Vec<_>>(), 1);
    let best = stack(scores)?;
    let best = if opts.metric == Metric::L2 { best.try_sqrt()? } else { best };
    Ok((best, stack(indices)?))
}

/// Every pair's score in f32, `+∞` where its groups match.
fn scored(x: &Tensor, c: &Tensor, opts: Nearest) -> svod_tensor::error::Result<Tensor> {
    let f32 = |t: &Tensor| t.cast(DType::Float32);
    let (x, c) = (f32(x), f32(c));
    let dot = x.matmul(&c.try_transpose(0, 1)?)?;
    let metric = opts.metric;
    let mut s = if metric.col_norms() {
        c.square().sum(1isize)?.try_unsqueeze(0)?.try_sub(&dot.try_mul(2.0f32)?)?
    } else {
        dot.neg()
    };
    if metric.row_norms() {
        s = s.try_add(&x.square().sum(1isize)?.try_unsqueeze(1)?)?;
    }
    let floor = |f: &Tensor, dim| -> svod_tensor::error::Result<Tensor> {
        let f = f32(f);
        let f = if metric == Metric::L2 { f.maximum(0.0f32)?.square() } else { f };
        f.try_unsqueeze(dim)
    };
    if let Some(f) = opts.col_floor {
        s = s.maximum(&floor(f, 0)?)?;
    }
    if let Some(f) = opts.row_floor {
        s = s.maximum(&floor(f, 1)?)?;
    }
    if metric == Metric::L2 {
        s = s.maximum(0.0f32)?;
    }
    if let Some((row_group, col_group)) = opts.groups {
        let i32 = |t: &Tensor, dim| t.cast(DType::Int32).try_unsqueeze(dim);
        let same = i32(row_group, 1)?.try_eq(&i32(col_group, 0)?)?;
        s = Tensor::const_(f32::INFINITY, DType::Float32).where_(&same, &s)?;
    }
    Ok(s)
}
