//! Each query row's cheapest candidates: nearest-neighbour search, k-means
//! assignment, maximum inner product, with optional score floors and group
//! exclusion.

use snafu::{ResultExt, ensure};
use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;

use super::config::Planner;
use super::shape::{self, extent, shape_of};
use super::{DtypeSnafu, GraphSnafu, LaunchSnafu, NoScoresSnafu, Result, ShapeSnafu, Tuner, fmt_shape, output, typed};
pub use crate::kernels::nearest::Metric;
use crate::kernels::nearest::{
    Floors, MergeSpec, NearestCfg, NearestSpec, nearest as kernel, nearest_merge, share_stride,
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
pub fn nearest(x: &Tensor, c: &Tensor, opts: Nearest) -> Result<(Tensor, Tensor)> {
    let dtype = x.dtype();
    picked(x, c, 1, opts, |planner, ranked, spec| tuned(TuneStore::global(), planner, dtype, ranked, spec))
}

/// For every row of `x [N, D]`, its `k` least scores over the candidates
/// [`nearest`] admits, ascending, and the candidate indices scoring them, an
/// equal score at the smaller index: `([N, k]` f32 scores, `[N, k]` i32
/// indices`)`. The slots past a row's admissible candidates score `+∞` at
/// index −1.
pub fn k_nearest(x: &Tensor, c: &Tensor, k: usize, opts: Nearest) -> Result<(Tensor, Tensor)> {
    ensure!(k > 0, NoScoresSnafu { op: OP });
    let dtype = x.dtype();
    let (score, index) =
        picked(x, c, k, opts, |planner, ranked, spec| tuned(TuneStore::global(), planner, dtype, ranked, spec))?;
    if k > 1 {
        return Ok((score, index));
    }
    let column = |t: Tensor| t.try_unsqueeze(1).context(GraphSnafu { op: OP });
    Ok((column(score)?, column(index)?))
}

/// [`k_nearest`] (`[N]` outputs at `k = 1`, as [`nearest`]'s) launching the
/// spec `pick` makes of the planned one, given the planner and the shape
/// gate's ranking.
pub(crate) fn picked(
    x: &Tensor,
    c: &Tensor,
    k: usize,
    opts: Nearest,
    pick: impl FnOnce(&Planner, &[NearestCfg], NearestSpec) -> NearestSpec,
) -> Result<(Tensor, Tensor)> {
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

    let (x_ext, c_ext) = (extent(&xs).map(|(e, _)| e), extent(&cs).map(|(e, _)| e));
    let planner = super::planner(&x.device());
    let (metric, floors, groups) = (opts.metric, opts.floors(), opts.groups.is_some());
    let dtypes = [x.dtype(), c.dtype()];
    let plan = shape::nearest(planner.as_deref(), &dtypes, x_ext.as_ref(), c_ext.as_ref(), metric, floors, groups, k);
    let Some(cfgs) = super::kernel(OP, plan) else {
        let graph = if k == 1 { graph(x, c, opts) } else { graph_k(x, c, k, opts) };
        return graph.context(GraphSnafu { op: OP });
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
    let spec = NearestSpec { n, m, d: depth, metric, floors, groups, k, cfg: cfgs[0] };
    let spec = pick(&planner, &cfgs, spec);
    let splits = spec.cfg.splits;
    // Whole, a row's `k` scores; split, every share's runs of `n` rows.
    let rows: &[usize] = match (splits, k) {
        (1, 1) => &[n],
        (1, _) => &[n, k],
        _ => &[k * splits * share_stride(n)],
    };
    let score_t = output(rows, &None, DType::Float32).context(graph_err)?;
    let index_t = output(rows, &None, DType::Int32).context(graph_err)?;
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
        return Ok((score, index));
    }
    let merge = MergeSpec { n, splits, metric, k };
    let rows: &[usize] = if k == 1 { &[n] } else { &[n, k] };
    let score_t = output(rows, &None, DType::Float32).context(graph_err)?;
    let index_t = output(rows, &None, DType::Int32).context(graph_err)?;
    let mut outs = launch::graph_launch_all(
        nearest_merge(&merge),
        &merge.lowering(&planner.device),
        &[&score, &index, &score_t, &index_t],
    )
    .context(LaunchSnafu { op: OP })?;
    let index = outs.pop().expect("the index");
    Ok((outs.pop().expect("the score"), index))
}

/// `spec` under the config [`nearest`] launches for its shape: the tuner's
/// pick among `ranked` (the shape gate's list), through `store`.
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
    let salt = (spec.metric, spec.floors, spec.groups);
    let build = |cfg: NearestCfg| programs(planner, dtype.clone(), &NearestSpec { cfg, ..spec.clone() });
    let p = problem(&spec);
    let expand = |admitted: &[NearestCfg]| planner.nearest_splits(&p, admitted);
    let cfg = Tuner::new(store, &planner.device, dtype.clone(), shape, salt, build).tuned(OP, ranked, expand);
    NearestSpec { cfg, ..spec }
}

/// The programs `spec` launches in order: the kernel, and split, the merge
/// of its shares.
pub(crate) fn programs(planner: &Planner, dtype: DType, spec: &NearestSpec) -> crate::tune::Candidate {
    let mut programs = vec![(typed!(dtype, kernel, spec), spec.cfg.lowering(&planner.device))];
    if spec.cfg.splits > 1 {
        let merge = MergeSpec { n: spec.n, splits: spec.cfg.splits, metric: spec.metric, k: spec.k };
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
