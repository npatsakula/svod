//! Each query row's cheapest candidates: nearest-neighbour search, k-means
//! assignment, maximum inner product, with optional score floors and group
//! exclusion.

use snafu::{ResultExt, ensure};
use svod_dtype::DType;
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
    Floors, Live, MergeSpec, NearestCfg, NearestSpec, SelectSpec, nearest as kernel, nearest_merge, nearest_select,
    select_blocks, share_stride,
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
    picked(x, c, 1, opts, |planner, ranked, spec| tuned(TuneStore::global(), planner, dtype, ranked, spec))
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
    let (score, index) = picked(x, c, k.clone(), opts, |planner, ranked, spec| {
        tuned(TuneStore::global(), planner, dtype, ranked, spec)
    })?;
    if k != SInt::Const(1) {
        return Ok((score, index));
    }
    let column = |t: Tensor| t.try_unsqueeze(1).context(GraphSnafu { op: OP });
    Ok((column(score)?, column(index)?))
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

/// [`k_nearest`] (`[N]` outputs at `k = 1`, as [`nearest`]'s) launching the
/// spec `pick` makes of the planned one, given the planner and the shape
/// gate's ranking.
pub(crate) fn picked(
    x: &Tensor,
    c: &Tensor,
    k: impl Into<SInt>,
    opts: Nearest,
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
    let depth = planner.nearest_depth(d).expect("planned");
    let graph_err = GraphSnafu { op: OP };
    let padded = |t: &Tensor| if depth == d { Ok(t.clone()) } else { t.try_pad(&[(0, 0), (0, (depth - d) as isize)]) };
    let squared = |t: &Tensor| t.cast(DType::Float32).square().sum(1isize);
    let (xp, cp) = (padded(x).context(graph_err)?, padded(c).context(graph_err)?);
    let c_sq = metric.col_norms().then(|| squared(c)).transpose().context(graph_err)?;
    let x_sq = metric.row_norms().then(|| squared(x)).transpose().context(graph_err)?;
    let f32 = |t: &Tensor| t.cast(DType::Float32);
    let (row_floor, col_floor) = (opts.row_floor.map(f32), opts.col_floor.map(f32));
    let (row_group, col_group) = opts.groups.map(|(r, c)| (r.cast(DType::Int32), c.cast(DType::Int32))).unzip();
    let live = Live {
        rows: x_var.as_ref().map(|v| (Var { name: v.name.clone(), min: v.min, max: v.max }, 0)),
        k: k_var.clone().filter(|_| top > 1),
    };
    let spec = NearestSpec { n, m, d: depth, metric, floors, groups, k: top, cfg: cfgs[0], live };
    let spec = pick(&planner, &cfgs, spec);
    let rows_in = [Some(&xp), x_sq.as_ref(), row_floor.as_ref(), row_group.as_ref()];
    let cols_in = [Some(&cp), c_sq.as_ref(), col_floor.as_ref(), col_group.as_ref()];
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
    let ins: Vec<&Tensor> = [Some(&xp), Some(&cp), c_sq.as_ref(), x_sq.as_ref()]
        .into_iter()
        .chain([row_floor.as_ref(), col_floor.as_ref(), row_group.as_ref(), col_group.as_ref()])
        .chain([Some(&score_t), Some(&index_t)])
        .flatten()
        .collect();
    let mut outs =
        launch::graph_launch_all(typed!(x.dtype(), kernel, &spec), &spec.cfg.lowering(&planner.device), &ins)
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

/// A score-matrix `spec`'s launches in order: per block of
/// [`select_blocks`], the first row and candidate it takes, its scores'
/// spec and its listing's. Where the rows are live a panel takes those of
/// the bound rows past its first, if any.
fn launches(spec: &NearestSpec) -> Vec<(usize, usize, NearestSpec, SelectSpec)> {
    let NearestSpec { n, m, k, metric, cfg, .. } = *spec;
    let select = cfg.select.expect("a score matrix");
    let (panel, block) = select_blocks(n, m, k, &cfg);
    let mut out = vec![];
    for r0 in (0..n).step_by(panel) {
        let live = Live { rows: spec.live.rows.clone().map(|(v, _)| (v, r0)), k: spec.live.k.clone() };
        for c0 in (0..m).step_by(block) {
            let (nr, mc) = (panel.min(n - r0), block.min(m - c0));
            let a = NearestSpec { n: nr, m: mc, live: live.clone(), ..spec.clone() };
            let b = SelectSpec { n: nr, cols: mc, k, metric, first: c0 == 0, cfg: select, live: live.clone() };
            out.push((r0, c0, a, b));
        }
    }
    out
}

/// [`picked`]'s lists through a score matrix: per block of
/// [`select_blocks`], its scores computed, then listed into its rows'
/// lists. `rows` and `cols` are the inputs per row and per candidate in the
/// kernel's order (`x`, then `x_sq`, the floor and the group; `c`, then
/// `c_sq`, the floor and the group), where present. Bound rows (`var`) are
/// read whole by every panel, which takes its own of them.
fn matrix(
    planner: &Planner,
    dtype: DType,
    spec: &NearestSpec,
    rows: [Option<&Tensor>; 4],
    cols: [Option<&Tensor>; 4],
    var: &Option<BatchVar>,
) -> Result<(Tensor, Tensor)> {
    let NearestSpec { n, m, k, cfg, .. } = *spec;
    let select = cfg.select.expect("a score matrix");
    let graph_err = || GraphSnafu { op: OP };
    // `len` of `inputs` from `start`, of `whole` each.
    let part = |inputs: [Option<&Tensor>; 4], start: usize, len: usize, whole: usize| {
        inputs.map(|t| {
            t.map(|t| if len == whole { Ok(t.clone()) } else { t.narrow(0, start, len).map(|t| t.contiguous()) })
                .transpose()
        })
    };
    let lowered = |p: [svod_tensor::error::Result<Option<Tensor>>; 4]| -> Result<[Option<Tensor>; 4]> {
        let [a, b, c, d] = p;
        Ok([a.context(graph_err())?, b.context(graph_err())?, c.context(graph_err())?, d.context(graph_err())?])
    };
    let mut lists: Vec<Vec<Tensor>> = vec![];
    let mut r = [None, None, None, None];
    for (r0, c0, a, b) in launches(spec) {
        let (nr, mc) = (a.n, a.m);
        if c0 == 0 {
            r = match var {
                Some(_) => rows.map(|t| t.cloned()),
                None => lowered(part(rows, r0, nr, n))?,
            };
            lists.push(vec![]);
        }
        let c = lowered(part(cols, c0, mc, m))?;
        let scores = output(&[nr * mc], &None, DType::Float32).context(graph_err())?;
        // The kernel's order: x, c, c_sq, x_sq, the floors, the groups.
        let ins: Vec<&Tensor> =
            [&r[0], &c[0], &c[1], &r[1], &r[2], &c[2], &r[3], &c[3]].into_iter().flatten().chain([&scores]).collect();
        let lowering = a.cfg.lowering(&planner.device);
        let scores =
            launch::graph_launch(typed!(dtype, kernel, &a), &lowering, &ins).context(LaunchSnafu { op: OP })?;
        let chunk = Tensor::from_slice([c0 as i32, i32::from(c0 + mc == m)]);
        // One panel's lists are the outputs, shaped with the live rows.
        let rows_var = if nr == n { var } else { &None };
        let outs = [output(&[nr, k], rows_var, DType::Float32), output(&[nr, k], rows_var, DType::Int32)]
            .into_iter()
            .collect::<svod_tensor::error::Result<Vec<_>>>()
            .context(graph_err())?;
        let list = lists.last_mut().expect("a panel");
        let ins: Vec<&Tensor> = [&scores, &chunk].into_iter().chain(list.iter()).chain(&outs).collect();
        let mut outs = launch::graph_launch_all(nearest_select(&b), &select.lowering(&planner.device), &ins)
            .context(LaunchSnafu { op: OP })?;
        *list = outs.split_off(outs.len() - 2);
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
    // A row's least alone keys as it did before the lists existed.
    let shape = [spec.n, spec.m, spec.d, spec.k];
    let shape = if spec.k == 1 { &shape[..3] } else { &shape[..] };
    let build = |cfg: NearestCfg| programs(planner, dtype.clone(), &NearestSpec { cfg, ..spec.clone() });
    let p = problem(&spec);
    let expand = |admitted: &[NearestCfg]| planner.nearest_splits(&p, admitted);
    let cfg = Tuner::new(store, &planner.device, dtype.clone(), shape, Salt(&spec), build).tuned(OP, ranked, expand);
    NearestSpec { cfg, ..spec }
}

/// The programs `spec` launches in order: the kernel, and split, the merge
/// of its shares.
pub(crate) fn programs(planner: &Planner, dtype: DType, spec: &NearestSpec) -> crate::tune::Candidate {
    if let Some(select) = spec.cfg.select {
        // Every block's scores, then their listing, as [`matrix`] launches them.
        let blocks = launches(spec).into_iter().flat_map(|(_, _, a, b)| {
            [
                (typed!(dtype.clone(), kernel, &a), spec.cfg.lowering(&planner.device)),
                (nearest_select(&b), select.lowering(&planner.device)),
            ]
        });
        return blocks.collect();
    }
    let mut programs = vec![(typed!(dtype, kernel, spec), spec.cfg.lowering(&planner.device))];
    if spec.cfg.splits > 1 {
        let merge = merge_of(spec);
        programs.push((nearest_merge(&merge), merge.lowering(&planner.device)));
    }
    programs
}

/// The planner's problem of `spec`.
pub(crate) fn problem(spec: &NearestSpec) -> crate::ops::config::Nearest {
    let NearestSpec { n, m, d, metric, floors, groups, k, .. } = *spec;
    crate::ops::config::Nearest { n, m, d, metric, floors, groups, k }
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
