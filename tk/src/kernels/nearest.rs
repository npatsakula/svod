//! Each query row's cheapest candidate: one `x·cᵀ` product per candidate
//! tile on the matrix core, its scores folded into a carried running minimum
//! and its candidate index rather than a softmax. The query rows stay
//! resident in registers while candidate tiles stream through a shared ring,
//! or, at a depth the registers cannot hold, both stream along the depth
//! through the GEMM's K pipeline once per candidate tile. The candidates may
//! split over several blocks per query block, each leaving a partial least
//! and index that [`nearest_merge`] folds. A row's `k` least are a list in
//! registers, or past what they hold a register queue merged into the rest
//! held in the outputs; or the kernel stores the scores as a matrix, block
//! by block, whose rows [`nearest_select`] lists a warp each.

use std::fmt;

use super::attention::WARP_ROWS;
use super::bound;
use super::gemm::{mainloop, tile_order};
use crate::atoms::Device;
use crate::build::*;
use crate::ir::*;
use crate::layouts::WarpGrid;
use crate::lower::Lowering;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NearestCfg {
    /// Query rows per block.
    pub bq: usize,
    /// Candidates per trip.
    pub bkv: usize,
    pub stages: usize,
    /// The ring loop unrolled over its slots.
    pub unroll: bool,
    /// Query rows a warp owns: [`WARP_ROWS`] or a multiple, every candidate
    /// fragment read from the ring feeding that many rows.
    pub warp_rows: usize,
    /// The depth a trip streams of the query rows and of the candidates
    /// alike, a candidate tile's product accumulated over `⌈d / bk⌉` trips;
    /// `None`: the rows stay resident and a trip streams candidates whole.
    pub bk: Option<usize>,
    /// The blocks a query block's candidate tiles split over, each folding
    /// an even share of them into a partial least and index; 1: one block
    /// folds them all.
    pub splits: usize,
    /// A row's register list of this many scores (a power of two below its
    /// `k`), merged into the rest of its `k` held in the outputs whenever
    /// some row of the warp fills its own; `None`: the register list holds
    /// all `k`.
    pub queue: Option<usize>,
    /// The scores stored as a matrix ([`nearest`] over a block of the
    /// candidates), whose rows [`nearest_select`] lists under this config;
    /// `None`: the kernel lists them itself.
    pub select: Option<SelectCfg>,
}

/// The tune store keys by this text: a resident, unsplit config prints as it
/// did before the depth chunk and the splits existed, so its stored choices
/// still apply.
impl fmt::Debug for NearestCfg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("NearestCfg");
        s.field("bq", &self.bq).field("bkv", &self.bkv).field("stages", &self.stages);
        s.field("unroll", &self.unroll).field("warp_rows", &self.warp_rows);
        if let Some(bk) = self.bk {
            s.field("bk", &bk);
        }
        if self.splits != 1 {
            s.field("splits", &self.splits);
        }
        if let Some(queue) = self.queue {
            s.field("queue", &queue);
        }
        if let Some(select) = self.select {
            s.field("select", &select);
        }
        s.finish()
    }
}

impl NearestCfg {
    pub const fn new(bq: usize, bkv: usize, stages: usize) -> Self {
        Self { bq, bkv, stages, unroll: false, warp_rows: WARP_ROWS, bk: None, splits: 1, queue: None, select: None }
    }

    /// Rows and candidates both streamed `bk` deep a trip.
    pub const fn streamed(bq: usize, bkv: usize, bk: usize, stages: usize) -> Self {
        Self { bk: Some(bk), ..Self::new(bq, bkv, stages) }
    }

    pub fn warps(&self) -> u32 {
        (self.bq / self.warp_rows) as u32
    }

    pub fn lowering(&self, device: &Device) -> Lowering {
        Lowering::uniform(device, self.unroll, WarpGrid { rows: self.warps(), cols: 1 })
    }

    /// Static shared memory in bytes of a 16-bit type at depth `d`: the
    /// candidate ring, or streamed, the ring of both operands' chunks.
    pub fn smem_bytes(&self, d: usize) -> usize {
        match self.bk {
            None => self.stages * self.bkv * d * 2,
            Some(bk) => self.stages * (self.bq + self.bkv) * bk * 2,
        }
    }
}

/// What a pair `(i, j)` scores before its floors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Metric {
    /// `|x_i − c_j|²`, as `|x_i|² + |c_j|² − 2·x_i·c_j` (a rounding may
    /// leave it a little under zero).
    #[default]
    SquaredL2,
    /// `|c_j|² − 2·x_i·c_j`: the squared distance less the row's own
    /// `|x_i|²`, which orders a row's candidates alike (an assignment).
    ShiftedL2,
    /// `|x_i − c_j|`, the squared distance clamped at zero.
    L2,
    /// `−x_i·c_j`: the largest inner product is the least score.
    NegDot,
}

impl Metric {
    /// Whether the candidates' squared norms are an input.
    pub fn col_norms(self) -> bool {
        self != Metric::NegDot
    }

    /// Whether the rows' squared norms are an input.
    pub fn row_norms(self) -> bool {
        matches!(self, Metric::SquaredL2 | Metric::L2)
    }
}

/// Lower bounds on the scores: `score = max(metric, row_floor[i],
/// col_floor[j])`, either side optional.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Floors {
    /// An `[n]` f32 floor per row.
    pub row: bool,
    /// An `[m]` f32 floor per candidate.
    pub col: bool,
}

/// The rows of `x [n, d]` against the candidates `c [m, d]`, `d` a multiple
/// of the matrix core's depth (zero columns pad it exactly); a streamed
/// config's last chunk may overhang it.
#[derive(Clone, Debug, PartialEq)]
pub struct NearestSpec {
    pub n: usize,
    pub m: usize,
    pub d: usize,
    pub metric: Metric,
    pub floors: Floors,
    /// `[n]` and `[m]` i32 group ids: a pair of equal ids is skipped (a
    /// self-join under groups skips `j = i` with them).
    pub groups: bool,
    /// The least scores a row keeps, ascending; 1: its least alone.
    pub k: usize,
    pub cfg: NearestCfg,
    /// The rows and the list length a launch binds, where they vary: `n`
    /// and `k` are then their bounds.
    pub live: Live,
}

/// What a launch binds by name where a dim varies: a symbolic dim's bound
/// variable, its spec's dim then the variable's bound (or a panel of it).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Live {
    /// The query rows and the first one a launch takes: it launches the
    /// blocks of those past it, at most its spec's `n`, one where the
    /// variable ends before it; the per-row inputs hold the variable's bound
    /// from row 0.
    pub rows: Option<(Var, usize)>,
    /// The scores a row keeps, its bound past 1: a list held for the bound,
    /// voting on and ending at the live length.
    pub k: Option<Var>,
}

impl Live {
    /// The rows the per-row inputs hold, a launch taking at most `n`.
    fn held(&self, n: usize) -> usize {
        self.rows.as_ref().map_or(n, |(v, _)| v.max as usize)
    }

    /// The first row of the per-row inputs a launch takes.
    fn from(&self) -> usize {
        self.rows.as_ref().map_or(0, |(_, from)| *from)
    }

    /// The rows a launch of at most `n` takes, where they vary.
    fn rows(&self, k: &mut Kernel, n: usize) -> Option<Sc> {
        let (v, from) = self.rows.as_ref()?;
        let mut rows = k.var(v.name.clone(), v.min, v.max);
        if *from > 0 {
            rows = (rows - *from).max(0);
        }
        if (v.max as usize).saturating_sub(*from) > n {
            rows = rows.min(n);
        }
        Some(rows)
    }

    /// The scores a row keeps, where they vary.
    fn k(&self, k: &mut Kernel) -> Option<Sc> {
        self.k.as_ref().map(|v| k.var(v.name.clone(), v.min, v.max))
    }
}

/// The blocks of `per` rows a launch of `n` rows takes, `live` of them where
/// they vary: one at least, as a launch has no empty grid (its rows bound
/// off).
fn launched(live: &Option<Sc>, n: usize, per: usize) -> Sc {
    match live {
        Some(rows) => ((rows.clone() + (per - 1)) / per).max(1),
        None => Sc::from(n.div_ceil(per)),
    }
}

/// The most candidates a kernel ranks: the running index is held in f32,
/// exact below 2^24.
pub const MAX_CANDIDATES: usize = 1 << 24;

impl NearestSpec {
    /// The input parameters in order; the outputs `score [n, k]` (f32) and
    /// `index [n, k]` (i32) follow them.
    pub fn inputs(&self) -> Vec<&'static str> {
        [
            (true, "x"),
            (true, "c"),
            (self.metric.col_norms(), "c_sq"),
            (self.metric.row_norms(), "x_sq"),
            (self.floors.row, "row_floor"),
            (self.floors.col, "col_floor"),
            (self.groups, "row_group"),
            (self.groups, "col_group"),
        ]
        .into_iter()
        .filter(|(on, _)| *on)
        .map(|(_, name)| name)
        .collect()
    }
}

/// For every row `i` of `x`, the least score over the candidates `c` (over
/// `j` with `row_group[i] ≠ col_group[j]` under groups) and the smallest
/// candidate index scoring it: the row's minimum, then the least column
/// where a score equals it. A row with no admissible candidate scores `+∞`
/// at index −1. Parameters as [`NearestSpec::inputs`] names them (`c_sq`,
/// `x_sq` the squared norms in f32, the floors f32, the groups i32), then
/// `score [n]` f32 and `index [n]` i32. Split (`cfg.splits > 1`), grid y
/// takes an even share of the candidate tiles and the outputs are
/// `score_part` and `index_part`, [`share_stride`] apart: each share's
/// least in the units the scores are compared in and its first index, which
/// [`nearest_merge`] folds; the blocks in flight walk groups of
/// [`SPLIT_GROUP`] query blocks over every share, as the GEMM's tiles, so
/// they read the same query and candidate lines.
///
/// With `k > 1` a row keeps its `k` least scores (`score [n, k]`, `index
/// [n, k]`, row-major), ascending, an equal score at the smaller index, the
/// slots past its admissible candidates `+∞` at −1: a sorted `[bq, k]` list
/// carried in registers, its slots spread over the lanes holding the same
/// rows ([`crate::layouts::TileLayout::spread_row`]), into which each trip
/// offers its scores one row minimum at a time ([`TileOp::Shift`] moving the
/// slots past it), only while some row of the warp has one below its `k`-th
/// ([`Kernel::if_any`]). Split, share `s` leaves slot `j` of its list at
/// `(s·k + j)·`[`share_stride`].
///
/// [Queued](NearestCfg::queue), the register list is a queue of the scores
/// below the row's `k`-th, and the list proper is the outputs' (`+∞` at −1
/// to start): once some row of a warp fills its queue the warp merges every
/// row's into it, run by run of the queue's length (the lower half of a
/// bitonic merge of the run and the carried queue stored, the upper half
/// carried on, [`TileOp::Flip`] pairing the slots), and takes the new `k`-th.
/// Under [`NearestCfg::select`] the outputs are `scores [n, m]` alone, each
/// pair's score in the units compared.
///
/// The L2 family's scores are compared halved (`c_sq/2 − x·c`, plus `x_sq/2`,
/// against halved floors; [`Metric::L2`] against its floors clamped at zero
/// and squared, its square root taken once per row), which orders them as
/// the metric's own.
pub fn nearest<T: Elem>(spec: &NearestSpec) -> Program {
    let NearestSpec { n, m, d, metric, floors, groups, k: top, cfg, ref live } = *spec;
    let NearestCfg { bq, bkv, stages, splits, .. } = cfg;
    let tiles = m.div_ceil(bkv);
    assert!(m <= MAX_CANDIDATES, "{m} candidates pass the f32 index's {MAX_CANDIDATES}");
    assert!(top >= 1, "a row keeps a score at least");
    assert!((1..=tiles).contains(&splits), "{splits} splits of {tiles} candidate tiles");
    // A fill zeroes the 16-byte chunks past an overhanging chunk's depth.
    assert!(cfg.bk.is_none() || d.is_multiple_of(8), "a streamed depth {d} of whole 16-byte chunks");
    let held = cfg.queue.unwrap_or(top);
    assert!(
        cfg.queue.is_none_or(|q| q.is_power_of_two() && (2..top).contains(&q) && splits == 1),
        "a queue of {held} below a list of {top}, unsplit"
    );
    assert!(cfg.select.is_none() || (cfg.queue.is_none() && splits == 1), "a score matrix, whole and unqueued");
    assert!(live.k.is_none() || top > 1, "a live list of more than one score");
    assert!(live.rows.is_none() || live.from() == 0 || cfg.select.is_some(), "a panel of a score matrix");
    let (held_rows, from) = (live.held(n), live.from());
    let mut k = Kernel::new("nearest");
    let x = k.param::<T>("x", ParamKind::In, held_rows * d);
    let c = k.param::<T>("c", ParamKind::In, m * d);
    let c_sq = metric.col_norms().then(|| k.param::<F32>("c_sq", ParamKind::In, m));
    let x_sq = metric.row_norms().then(|| k.param::<F32>("x_sq", ParamKind::In, held_rows));
    let row_floor = floors.row.then(|| k.param::<F32>("row_floor", ParamKind::In, held_rows));
    let col_floor = floors.col.then(|| k.param::<F32>("col_floor", ParamKind::In, m));
    let grouped = groups.then(|| {
        (k.param::<I32>("row_group", ParamKind::In, held_rows), k.param::<I32>("col_group", ParamKind::In, m))
    });
    let split = splits > 1;
    let matrix = cfg.select.is_some();
    let (score, index) = if split { ("score_part", "index_part") } else { ("score", "index") };
    let parts = top * if split { splits * share_stride(n) } else { n };
    let (score_out, index_out) = match matrix {
        true => (k.param::<F32>("scores", ParamKind::Out, n * m), None),
        false => (k.param::<F32>(score, ParamKind::Out, parts), Some(k.param::<I32>(index, ParamKind::Out, parts))),
    };
    let rows_live = live.rows(&mut k, n);
    let k_live = live.k(&mut k);
    // The rows a block's views hold: those of the capacity even where the
    // rows are live, the blocks past the live ones never launched and the
    // live rows' last block reading and writing rows past them that the
    // buffers hold, which an output's shape leaves out.
    let rb = bound(n, bq);
    let blocks = launched(&rows_live, n, bq);
    k.grid([blocks.clone(), Sc::from(splits), Sc::from(1)]);
    k.warps(cfg.warps());
    let c_s = cfg.bk.is_none().then(|| k.smem::<T>("c_s", stages * bkv * d));

    // The query block, and split, the first candidate tile of its share and
    // how many it takes.
    let (q_block, share) = match split {
        true => {
            let (q_block, part) = tile_order(&mut k, blocks.clone(), splits, SPLIT_GROUP);
            let first = part.clone() * tiles / splits;
            let count = (part.clone() + 1) * tiles / splits - first.clone();
            (q_block, Some((part, first, count)))
        }
        false => (k.block(0), None),
    };
    let first = share.as_ref().map(|(_, first, _)| first.clone());
    let count = share.as_ref().map_or(Sc::from(tiles), |(_, _, count)| count.clone());
    let tile_start = |step: Sc| match &first {
        Some(first) => (first.clone() + step) * bkv,
        None => step * bkv,
    };
    let q_off = q_block * bq;
    let shape = Shape::new(bq, bkv);
    let rows = Shape::new(bq, 1);
    // The block's rows of an `[n]` parameter (of the one at `base` of
    // several, its rows `step` apart), `rows` of them, and a trip's columns
    // of an `[m]` one.
    fn row_view<E: Elem>(
        k: &mut Kernel,
        p: ParamRef<E>,
        (base, step): (Sc, usize),
        rows: Option<Sc>,
        bq: usize,
        q_off: Sc,
    ) -> Gmem<E> {
        let v = k.view(p, base, [step, 1], Shape::new(bq, 1), [rows, None]);
        k.at(v, q_off, 0)
    }
    fn col_view<E: Elem>(k: &mut Kernel, p: ParamRef<E>, m: usize, bkv: usize, start: Sc) -> Gmem<E> {
        let v = k.view(p, 0, [0, 1], Shape::new(1, bkv), [None, bound(m, bkv)]);
        k.at(v, 0, start)
    }
    let half = |k: &mut Kernel, v: Regs<F32>| {
        let h = k.fill::<F32>(k.shape(v), Const::Float(0.5));
        k.binary(v, h, BinaryOp::Mul)
    };
    // A floor in the units the scores are compared in.
    let floor = |k: &mut Kernel, f: Regs<F32>| match metric {
        Metric::NegDot => f,
        Metric::SquaredL2 | Metric::ShiftedL2 => half(k, f),
        Metric::L2 => {
            let zero = k.zeros::<F32>(k.shape(f));
            let f = k.binary(f, zero, BinaryOp::Max);
            let f = k.binary(f, f, BinaryOp::Mul);
            half(k, f)
        }
    };
    let depth = cfg.bk.unwrap_or(d);
    let x_view = k.view(x, from * d, [d, 1], Shape::new(bq, depth), [rb.clone(), bound(d, depth)]);
    let x_view = k.at(x_view, q_off.clone(), 0);
    let q_r = cfg.bk.is_none().then(|| k.load(x_view));
    let x_sq = x_sq.map(|p| {
        let v = row_view(&mut k, p, (Sc::from(from), 1), rb.clone(), bq, q_off.clone());
        let x_sq = k.load(v);
        half(&mut k, x_sq)
    });
    let row_floor = row_floor.map(|p| {
        let v = row_view(&mut k, p, (Sc::from(from), 1), rb.clone(), bq, q_off.clone());
        let f = k.load(v);
        floor(&mut k, f)
    });
    let row_group = grouped.map(|(p, _)| {
        let v = row_view(&mut k, p, (Sc::from(from), 1), rb.clone(), bq, q_off.clone());
        k.load(v)
    });
    let c_view = k.view(c, 0, [d, 1], Shape::new(bkv, depth), [bound(m, bkv), bound(d, depth)]);
    let inf = |k: &mut Kernel| k.fill::<F32>(shape, Const::Float(f64::INFINITY));
    // The scores of the candidate tile at `start` from its products `dot`.
    let scores = |k: &mut Kernel, dot: Regs<F32>, start: Sc| {
        let mut s = match c_sq {
            Some(p) => {
                let v = col_view(k, p, m, bkv, start.clone());
                let c_sq = k.load(v);
                let c_sq = half(k, c_sq);
                k.binary(c_sq, dot, BinaryOp::Sub)
            }
            None => k.unary(dot, UnaryOp::Neg),
        };
        if let Some(x_sq) = x_sq {
            s = k.binary(s, x_sq, BinaryOp::Add);
        }
        if let Some(p) = col_floor {
            let v = col_view(k, p, m, bkv, start.clone());
            let f = k.load(v);
            let f = floor(k, f);
            s = k.binary(s, f, BinaryOp::Max);
        }
        if let Some(f) = row_floor {
            s = k.binary(s, f, BinaryOp::Max);
        }
        // A floor clamps the distance already.
        if metric == Metric::L2 && floors == Floors::default() {
            let zero = k.zeros::<F32>(shape);
            s = k.binary(s, zero, BinaryOp::Max);
        }
        if let (Some(row_group), Some((_, col_group))) = (row_group, grouped) {
            let v = col_view(k, col_group, m, bkv, start.clone());
            let col_group = k.load(v);
            // Against a full-shape zero: the predicate takes the tile's shape.
            let zero = k.zeros::<I32>(shape);
            let row_group = k.binary(row_group, zero, BinaryOp::Add);
            let same = k.compare(row_group, col_group, CmpOp::Eq);
            let skipped = inf(k);
            s = k.where_(same, skipped, s);
        }
        // The last trip's columns past `m` read the clamped last row.
        if !m.is_multiple_of(bkv) {
            let partial = Sc::from(m).lt(start.clone() + bkv);
            [s] = k.select_if(
                partial,
                |k| {
                    let col = k.coord(shape, Axis::Col);
                    let left = k.splat::<I32>(shape, Sc::from(m) - start.clone());
                    let inside = k.compare(col, left, CmpOp::Lt);
                    let past = inf(k);
                    [k.where_(inside, s, past)]
                },
                |_| [s],
            );
        }
        s
    };
    // The candidate tile at `start`'s products `dot` folded into the row's
    // carried least score and its first index.
    let fold = |k: &mut Kernel, dot: Regs<F32>, start: Sc, best: Regs<F32>, index: Regs<F32>| {
        let s = scores(k, dot, start.clone());
        let low = k.reduce(s, Axis::Row, ReduceOp::Min);
        let first = |k: &mut Kernel| {
            // The columns in f32 (exact below 2^24, as the carried index):
            // a float minimum folds its negations into the lane moves an
            // integer one pays for.
            let at_low = k.compare(s, low, CmpOp::Eq);
            let col = k.coord(shape, Axis::Col);
            let col = k.cast::<I32, F32>(col);
            let none = inf(k);
            let cols = k.where_(at_low, col, none);
            let first = k.reduce(cols, Axis::Row, ReduceOp::Min);
            let start = k.splat::<F32>(rows, start);
            [k.binary(first, start, BinaryOp::Add)]
        };
        // Strictly lower: an earlier trip's equal score keeps its smaller
        // index. A row's least falls ever more rarely as the trips pass, so
        // resident the warps none of whose rows a trip lowers skip finding
        // where it is. Streamed, a tile folds once per depth pipeline, and
        // the branch costs the registers and the overlap of the next tile's
        // fill it would save (on gfx1151 a 64-row tile of 64 candidates
        // spills).
        match cfg.bk {
            None => {
                let lower = k.compare(low, best, CmpOp::Lt);
                let [index] = k.where_any(lower, first, [index]);
                [k.where_(lower, low, best), index]
            }
            Some(_) => {
                let [first] = first(k);
                let lower = k.compare(low, best, CmpOp::Lt);
                [k.where_(lower, low, best), k.where_(lower, first, index)]
            }
        }
    };

    // A row's queue (`[bq, held]` scores ascending and their indices)
    // merged into the rest of its list, the outputs' `[bq, top]` in the
    // units the scores are compared in (`last`: the metric's own, the list
    // complete): run by run of `held` slots, each the lower half of merging
    // it with the queue's carry, which the upper half goes on as. Returns
    // the reset queue and the list's `top`-th score (the vote's), unless
    // `last`.
    let list = Shape::new(bq, held);
    // The list's slots a row holds, its bound's even where it is live: the
    // slots past the live ones its last run reads are masked, those it
    // writes left out of an output's shape.
    let kb = bound(top, held);
    let chunk = |k: &mut Kernel, p: ParamRef<F32>, c: Sc| {
        let v = k.view(p, 0, [top, 1], list, [rb.clone(), kb.clone()]);
        k.at(v, q_off.clone(), c * held)
    };
    let index_chunk = |k: &mut Kernel, c: Sc| {
        let v = k.view(index_out.expect("an index output"), 0, [top, 1], list, [rb.clone(), kb.clone()]);
        k.at(v, q_off.clone(), c * held)
    };
    // The list's runs of `held` slots, the last one's index and its slots
    // (the live list's where it varies).
    let runs = top.div_ceil(held);
    let tail = top - (runs - 1) * held;
    let (run_count, last_run, live_tail) = match &k_live {
        None => (Sc::from(runs), Sc::from(runs - 1), None),
        Some(live) => {
            let runs = (live.clone() + (held - 1)) / held;
            (runs.clone(), runs.clone() - 1, Some(live.clone() - (runs - 1) * held))
        }
    };
    // A list column of the last run's slot `tail` less `before`, as `i32`.
    let tail_col = |k: &mut Kernel, before: usize| match &live_tail {
        None => k.fill::<I32>(list, Const::Int((tail - before) as i64)),
        Some(live) => k.splat::<I32>(list, live.clone() - before),
    };
    let flush = |k: &mut Kernel, queue: [Regs<F32>; 2], last: bool| {
        // Run `c` (the list's slots its first `tail` where it is the last
        // run) merged with the carry, its lower half stored and the list's
        // `top`-th score taken from it where it is the last.
        let unset = k.fill::<F32>(rows, Const::Float(f64::INFINITY));
        let init = queue.into_iter().chain((!last).then_some(unset)).collect();
        let out = k.loop_vec(run_count.clone(), init, |k, c, carried| {
            let v = chunk(k, score_out, c.clone());
            let mut run = [k.load(v), {
                let v = index_chunk(k, c.clone());
                let i = k.load(v);
                k.cast::<I32, F32>(i)
            }];
            let ends = c.clone().eq(last_run.clone());
            if tail < held || live_tail.is_some() {
                run = k.select_if(
                    ends.clone(),
                    |k| {
                        let col = k.coord(list, Axis::Col);
                        let width = tail_col(k, 0);
                        let inside = k.compare(col, width, CmpOp::Lt);
                        let empty = [Const::Float(f64::INFINITY), Const::Float(-1.0)].map(|e| k.fill::<F32>(list, e));
                        [0, 1].map(|j| k.where_(inside, run[j], empty[j]))
                    },
                    |_| run,
                );
            }
            let [lo, hi] = merged(k, run, [carried[0], carried[1]]);
            let stored = if last { finished(k, lo[0], metric) } else { lo[0] };
            let v = chunk(k, score_out, c.clone());
            k.store(v, stored);
            let i = k.cast::<F32, I32>(lo[1]);
            let v = index_chunk(k, c);
            k.store(v, i);
            if last {
                return hi.to_vec();
            }
            let [kth] = k.select_if(
                ends,
                |k| {
                    let col = k.coord(list, Axis::Col);
                    let at = tail_col(k, 1);
                    let here = k.compare(col, at, CmpOp::Eq);
                    let none = k.fill::<F32>(list, Const::Float(f64::NEG_INFINITY));
                    let kth = k.where_(here, lo[0], none);
                    [k.reduce(kth, Axis::Row, ReduceOp::Max)]
                },
                |_| [carried[2]],
            );
            vec![hi[0], hi[1], kth]
        });
        (!last).then(|| {
            let empty = [Const::Float(f64::INFINITY), Const::Float(-1.0)].map(|e| k.fill::<F32>(list, e));
            vec![empty[0], empty[1], out[2]]
        })
    };

    // The register list's vote: its `k`-th score, its last but where the
    // list is live.
    let vote = |k: &mut Kernel, best: Regs<F32>| match &k_live {
        None => k.reduce(best, Axis::Row, ReduceOp::Max),
        Some(live) => {
            let col = k.coord(list, Axis::Col);
            let at = k.splat::<I32>(list, live.clone() - 1);
            let here = k.compare(col, at, CmpOp::Eq);
            let none = k.fill::<F32>(list, Const::Float(f64::NEG_INFINITY));
            let kth = k.where_(here, best, none);
            k.reduce(kth, Axis::Row, ReduceOp::Max)
        }
    };
    // A trip offers at most a list's or a trip's scores.
    let offers = match &k_live {
        None => Sc::from(top.min(bkv)),
        Some(live) => live.clone().min(bkv),
    };
    // The tile's scores offered to the row's carried list (`[bq, held]` of
    // its least scores ascending, their indices, and its last score: the
    // vote's, a row vector as the row minimum is), its least first,
    // then each next least once the least before it is masked out: a trip
    // offers at most `top`, and only while some row of the warp still has a
    // score below one in its list. A row whose offer misses its list masks
    // its least all the same: its scores only rise, so it misses again.
    // Queued, the vote's score is the list's `top`-th and a row offers only
    // below it, into a queue merged into the list once some row fills its own.
    let fold_list = |k: &mut Kernel, dot: Regs<F32>, start: Sc, list: Vec<Regs<F32>>| {
        let s = scores(k, dot, start.clone());
        let low = k.reduce(s, Axis::Row, ReduceOp::Min);
        let lower = k.compare(low, list[2], CmpOp::Lt);
        k.if_any(
            lower,
            |k| {
                let col = k.coord(shape, Axis::Col);
                let col = k.cast::<I32, F32>(col);
                let none = inf(k);
                let origin = k.splat::<F32>(rows, start);
                let init = [s, low].into_iter().chain(list.iter().copied()).collect();
                let out = k.loop_vec(offers.clone(), init, |k, _, carried| {
                    let [s, low, best, at, last] = carried[..] else { unreachable!("five carried tiles") };
                    let lower = k.compare(low, last, CmpOp::Lt);
                    k.if_any(
                        lower,
                        |k| {
                            let at_low = k.compare(s, low, CmpOp::Eq);
                            let cols = k.where_(at_low, col, none);
                            let first = k.reduce(cols, Axis::Row, ReduceOp::Min);
                            let index = k.binary(first, origin, BinaryOp::Add);
                            let hit = k.compare(col, first, CmpOp::Eq);
                            let rest = k.where_(hit, none, s);
                            let next = k.reduce(rest, Axis::Row, ReduceOp::Min);
                            if cfg.queue.is_none() {
                                let [best, at] = offer(k, [best, at], low, index);
                                let last = vote(k, best);
                                return vec![rest, next, best, at, last];
                            }
                            let unset = k.fill::<F32>(rows, Const::Float(f64::INFINITY));
                            let below = k.compare(low, last, CmpOp::Lt);
                            let low = k.where_(below, low, unset);
                            let [best, at] = offer(k, [best, at], low, index);
                            let most = k.reduce(best, Axis::Row, ReduceOp::Max);
                            let full = k.compare(most, unset, CmpOp::Lt);
                            let queue =
                                k.if_any(full, |k| flush(k, [best, at], false).expect("a vote"), vec![best, at, last]);
                            [rest, next].into_iter().chain(queue).collect()
                        },
                        carried.clone(),
                    )
                });
                out[2..].to_vec()
            },
            list.clone(),
        )
    };

    let mut init = vec![];
    if !matrix {
        init = vec![k.fill::<F32>(list, Const::Float(f64::INFINITY)), k.fill::<F32>(list, Const::Float(-1.0))];
    }
    if top > 1 && !matrix {
        init.push(k.fill::<F32>(rows, Const::Float(f64::INFINITY)));
    }
    // Queued, the block's lists start empty in the outputs, held as the
    // queue is: a lane reads back the slots it writes.
    if cfg.queue.is_some() {
        let (score0, index0) = (init[0], k.cast::<F32, I32>(init[1]));
        k.loop_vec::<F32>(run_count.clone(), vec![], |k, c, _| {
            let v = chunk(k, score_out, c.clone());
            k.store(v, score0);
            let v = index_chunk(k, c);
            k.store(v, index0);
            vec![]
        });
    }
    // A score matrix's trip stores its scores.
    let folded = |k: &mut Kernel, dot: Regs<F32>, start: Sc, carried: Vec<Regs<F32>>| match (matrix, top) {
        (true, _) => {
            let s = scores(k, dot, start.clone());
            let v = k.view(score_out, 0, [m, 1], shape, [rb.clone(), bound(m, bkv)]);
            let v = k.at(v, q_off.clone(), start);
            k.store(v, s);
            vec![]
        }
        (false, 1) => fold(k, dot, start, carried[0], carried[1]).to_vec(),
        (false, _) => fold_list(k, dot, start, carried),
    };
    let carried = match (q_r, c_s) {
        (Some(q_r), Some(c_s)) => k.pipeline_vec(
            count,
            stages,
            init,
            |k, step, slot| {
                let c_g = k.at(c_view, tile_start(step), 0);
                let c_t = k.smem_slot::<T>(c_s, slot, Shape::new(bkv, d));
                k.stage(c_t, c_g, CopyMode::Async);
            },
            |k, step, slot, carried| {
                let start = tile_start(step);
                let c_t = k.smem_slot::<T>(c_s, slot, Shape::new(bkv, d));
                let zero = k.zeros::<F32>(shape);
                let dot = k.mma(zero, q_r, false, c_t, true);
                folded(k, dot, start, carried)
            },
        ),
        _ => k.loop_vec(count, init, |k, step, carried| {
            let start = tile_start(step);
            let c_g = k.at(c_view, start.clone(), 0);
            let x_chunk = |k: &mut Kernel, _: Sc, col: Sc| k.at(x_view, 0, col);
            let chunks = d.div_ceil(depth);
            let [dot] = mainloop(k, Shape::new(bq, depth), &x_chunk, [c_g], chunks, stages, depth, None, 1);
            folded(k, dot, start, carried)
        }),
    };
    if cfg.queue.is_some() {
        flush(&mut k, [carried[0], carried[1]], true);
    }
    if matrix || cfg.queue.is_some() {
        return k.finish();
    }
    // A share's partial results, a row's least in the units the merge
    // compares; slot `j` of a row's list at `j` of its `top` (unsplit) or
    // in the share's `j`th run of rows (split).
    let [best, index] = carried[..2] else { unreachable!("a score and an index") };
    let (best, at) = match &share {
        Some((part, ..)) if top == 1 => (best, (part.clone() * share_stride(n), [1, 1])),
        Some((part, ..)) => (best, (part.clone() * top * share_stride(n), [1, share_stride(n)])),
        None => (finished(&mut k, best, metric), (Sc::from(0), [top, 1])),
    };
    fn slots<E: Elem>(
        k: &mut Kernel,
        p: ParamRef<E>,
        (base, step): (Sc, [usize; 2]),
        list: Shape,
        rows: Option<Sc>,
        q_off: Sc,
    ) -> Gmem<E> {
        let v = k.view(p, base, step, list, [rows, None]);
        k.at(v, q_off, 0)
    }
    let v = slots(&mut k, score_out, at.clone(), list, rb.clone(), q_off.clone());
    k.store(v, best);
    let index = k.cast::<F32, I32>(index);
    let v = slots(&mut k, index_out.expect("an index output"), at, list, rb, q_off.clone());
    k.store(v, index);
    k.finish()
}

/// How [`nearest_select`] lists a score matrix's rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SelectCfg {
    /// Rows a block lists, a warp each.
    pub warps: u32,
    /// The scores a lane queues below its row's `k`-th before the warp
    /// merges its lanes' queues into the row's list.
    pub queue: usize,
}

impl SelectCfg {
    pub fn lowering(&self, device: &Device) -> Lowering {
        Lowering::uniform(device, false, WarpGrid { rows: self.warps, cols: 1 })
    }

    /// Whether it lists `k` scores a row: a lane's queue (a power of two)
    /// fits the list's runs of [`LANES`].
    pub fn fits(&self, k: usize) -> bool {
        self.queue.is_power_of_two() && self.queue <= select_runs(k) && self.warps.is_power_of_two()
    }
}

/// The most scores a block of a score matrix holds.
pub const SCRATCH: usize = 1 << 28;

/// The rows and candidates of each block a score matrix of `n` rows by `m`
/// candidates (tiles of `cfg`, `k` scores a row) is computed and listed
/// in: whole rows of candidates where [`SCRATCH`] holds them, else as many
/// candidates as it holds of every row, at least `8·k` (each block reads
/// and writes a list a row), and then as many rows as it holds of them.
pub fn select_blocks(n: usize, m: usize, k: usize, cfg: &NearestCfg) -> (usize, usize) {
    let cols = m.min((SCRATCH / n).max(8 * k).next_multiple_of(cfg.bkv));
    let rows = n.min((SCRATCH / cols).max(cfg.bq) / cfg.bq * cfg.bq);
    (rows, cols)
}

/// Lanes a row's list spreads over in [`nearest_select`], a run of the list
/// a register of each.
pub const LANES: usize = 32;

/// The runs of [`LANES`] slots a list of `k` is held in: `k` up to a power
/// of two.
pub fn select_runs(k: usize) -> usize {
    k.next_power_of_two().max(LANES) / LANES
}

/// A block of a score matrix listed into its rows' lists.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectSpec {
    pub n: usize,
    /// Scores a row holds.
    pub cols: usize,
    pub k: usize,
    pub metric: Metric,
    /// The lists start empty, else from the inputs `score_in` and
    /// `index_in` (in the units the scores are compared in).
    pub first: bool,
    pub cfg: SelectCfg,
    /// The rows and the list length a launch binds, where they vary (its
    /// inputs and outputs the panel's own).
    pub live: Live,
}

/// Each row's `k` least of `scores [n, cols]` (as a score-matrix
/// [`nearest`] leaves them, compared units, `+∞` for pairs skipped) and,
/// unless `first`, the lists `score_in [n, k]` and `index_in [n, k]` hold,
/// with their indices, into `score` and `index`: ascending, an equal score at the smaller index,
/// `+∞` at −1 past the admissible candidates. `chunk [2]` (i32) is the
/// first column's candidate index and whether the lists are left in the
/// metric's units (else in the units compared). A warp
/// lists a row, holding its list in registers, runs of [`LANES`] slots a
/// lane each; a lane queues the scores below the row's `k`-th of its
/// column, and whenever some lane fills its queue the warp sorts them all
/// and merges them into the list (a bitonic merge of the list and the
/// queues reversed).
pub fn nearest_select(spec: &SelectSpec) -> Program {
    let SelectSpec { n, cols, k: top, metric, first, cfg, ref live } = *spec;
    assert!(cfg.fits(top), "{cfg:?} lists {top}");
    let (rows, depth, runs) = (cfg.warps as usize, cfg.queue, select_runs(top));
    let mut k = Kernel::new("nearest_select");
    let scores = k.param::<F32>("scores", ParamKind::In, n * cols);
    let chunk = k.param::<I32>("chunk", ParamKind::In, 2);
    let lists = (!first).then(|| {
        (k.param::<F32>("score_in", ParamKind::In, n * top), k.param::<I32>("index_in", ParamKind::In, n * top))
    });
    let score_out = k.param::<F32>("score", ParamKind::Out, n * top);
    let index_out = k.param::<I32>("index", ParamKind::Out, n * top);
    let (base, last) = (k.load_scalar(chunk, 0), k.load_scalar(chunk, 1));
    let rows_live = live.rows(&mut k, n);
    let k_live = live.k(&mut k);
    // The rows and slots of the capacity, as `nearest`'s live views.
    let (rb, kb) = (bound(n, rows), bound(top, LANES));
    k.grid([launched(&rows_live, n, rows), Sc::from(1), Sc::from(1)]);
    k.warps(cfg.warps);
    let row0 = k.block(0) * rows;
    let lane = Shape::new(rows, LANES);
    let empty = |k: &mut Kernel| [Const::Float(f64::INFINITY), Const::Float(-1.0)].map(|e| k.fill::<F32>(lane, e));
    // Run `t` of the block's lists.
    let run = |k: &mut Kernel, (score, index): (ParamRef<F32>, ParamRef<I32>), t: usize| {
        let bounds = [rb.clone(), kb.clone()];
        let s = k.view(score, 0, [top, 1], lane, bounds.clone());
        let s = k.at(s, row0.clone(), t * LANES);
        let i = k.view(index, 0, [top, 1], lane, bounds);
        (s, k.at(i, row0.clone(), t * LANES))
    };
    let held = top.div_ceil(LANES);
    // The list's `top`-th score (the live length's where it varies: each
    // run's slot of it, if any, the most of them).
    let kth = |k: &mut Kernel, list: &[[Regs<F32>; 2]]| {
        let col = k.coord(lane, Axis::Col);
        let Some(live) = &k_live else {
            let at = k.fill::<I32>(lane, Const::Int(((top - 1) % LANES) as i64));
            let here = k.compare(col, at, CmpOp::Eq);
            let none = k.fill::<F32>(lane, Const::Float(f64::NEG_INFINITY));
            let v = k.where_(here, list[(top - 1) / LANES][0], none);
            return k.reduce(v, Axis::Row, ReduceOp::Max);
        };
        let none = k.fill::<F32>(lane, Const::Float(f64::NEG_INFINITY));
        let mut kth = None;
        for (t, run) in list.iter().take(held).enumerate() {
            let at = k.splat::<I32>(lane, live.clone() - (t * LANES + 1));
            let here = k.compare(col, at, CmpOp::Eq);
            let v = k.where_(here, run[0], none);
            let v = k.reduce(v, Axis::Row, ReduceOp::Max);
            kth = Some(kth.map_or(v, |kth| k.binary(kth, v, BinaryOp::Max)));
        }
        kth.expect("a run")
    };
    let loaded = |k: &mut Kernel| {
        (0..runs)
            .flat_map(|t| {
                if t >= held {
                    return empty(k);
                }
                let (s, i) = run(k, lists.expect("the lists to carry"), t);
                let s = k.load(s);
                let i = k.load(i);
                let mut v = [s, k.cast::<I32, F32>(i)];
                if (t + 1) * LANES > top || k_live.is_some() {
                    let col = k.coord(lane, Axis::Col);
                    let width = match &k_live {
                        None => k.fill::<I32>(lane, Const::Int((top - t * LANES) as i64)),
                        Some(live) => k.splat::<I32>(lane, live.clone() - t * LANES),
                    };
                    let inside = k.compare(col, width, CmpOp::Lt);
                    let e = empty(k);
                    v = [0, 1].map(|j| k.where_(inside, v[j], e[j]));
                }
                v
            })
            .collect::<Vec<_>>()
    };
    let list = if first { (0..runs).flat_map(|_| empty(&mut k)).collect() } else { loaded(&mut k) };
    let list: Vec<[Regs<F32>; 2]> = list.chunks(2).map(|p| [p[0], p[1]]).collect();
    let tau = kth(&mut k, &list);
    let queue: Vec<[Regs<F32>; 2]> = (0..depth).map(|_| empty(&mut k)).collect();

    // The queues sorted and merged into the list: the list's last runs
    // ordered against the queues' reversed, then the list's bitonic order.
    let merge = |k: &mut Kernel, mut list: Vec<[Regs<F32>; 2]>, mut queue: Vec<[Regs<F32>; 2]>| {
        network(k, &mut queue, (1..=(depth * LANES).ilog2()).map(|b| 1 << b));
        for t in runs - depth..runs {
            let rev = queue[runs - 1 - t].map(|v| k.flip(v, Axis::Col, LANES as u32 - 1));
            let first = before(k, list[t], rev);
            list[t] = [0, 1].map(|j| k.where_(first, list[t][j], rev[j]));
        }
        network(k, &mut list, [runs * LANES]);
        let tau = kth(k, &list);
        let queue = (0..depth).map(|_| empty(k));
        list.into_iter().chain(queue).flatten().chain([tau]).collect::<Vec<_>>()
    };
    let split = |v: Vec<Regs<F32>>| {
        let pairs: Vec<[Regs<F32>; 2]> = v[..v.len() - 1].chunks(2).map(|p| [p[0], p[1]]).collect();
        let (list, queue) = pairs.split_at(runs);
        (list.to_vec(), queue.to_vec(), v[v.len() - 1])
    };
    let init: Vec<Regs<F32>> = list.iter().chain(&queue).flatten().copied().chain([tau]).collect();
    // A step past the last merges what is still queued.
    let steps = cols.div_ceil(LANES);
    let view = k.view(scores, 0, [cols, 1], lane, [rb.clone(), bound(cols, LANES)]);
    let carried = k.loop_vec(steps + 1, init, |k, step, carried| {
        let (list, queue, tau) = split(carried.clone());
        let [s] = k.select_if(
            step.clone().lt(steps),
            |k| {
                let at = k.at(view, row0.clone(), step.clone() * LANES);
                let s = k.load(at);
                if cols.is_multiple_of(LANES) {
                    return [s];
                }
                // The last step's columns past `cols` read zero.
                let col = k.coord(lane, Axis::Col);
                let left = k.splat::<I32>(lane, Sc::from(cols) - step.clone() * LANES);
                let inside = k.compare(col, left, CmpOp::Lt);
                let past = k.fill::<F32>(lane, Const::Float(f64::INFINITY));
                [k.where_(inside, s, past)]
            },
            |k| [k.fill::<F32>(lane, Const::Float(f64::INFINITY))],
        );
        let none = k.fill::<F32>(lane, Const::Float(f64::INFINITY));
        let end = {
            let at = k.splat::<I32>(lane, step.clone());
            let past = k.fill::<I32>(lane, Const::Int(steps as i64));
            k.compare(at, past, CmpOp::Eq)
        };
        let lower = k.compare(s, tau, CmpOp::Lt);
        let queued = k.compare(queue[0][0], none, CmpOp::Lt);
        let pending = k.binary(end, queued, BinaryOp::And);
        let busy = k.binary(lower, pending, BinaryOp::Or);
        k.if_any(
            busy,
            |k| {
                let col = k.coord(lane, Axis::Col);
                let col = k.cast::<I32, F32>(col);
                let origin = k.splat::<F32>(lane, step.clone() * LANES + base.clone());
                let index = k.binary(col, origin, BinaryOp::Add);
                let v = k.where_(lower, s, none);
                let slots: Vec<Regs<F32>> = queue.iter().map(|q| q[0]).chain(queue.iter().map(|q| q[1])).collect();
                let slots = insert(k, &slots, v, index);
                let queue: Vec<[Regs<F32>; 2]> = (0..depth).map(|j| [slots[j], slots[depth + j]]).collect();
                let full = k.compare(queue[depth - 1][0], none, CmpOp::Lt);
                let full = k.binary(full, pending, BinaryOp::Or);
                let kept: Vec<Regs<F32>> = list.iter().chain(&queue).flatten().copied().chain([tau]).collect();
                k.if_any(full, |k| merge(k, list.clone(), queue.clone()), kept)
            },
            carried,
        )
    });
    let (list, ..) = split(carried);
    for (t, [s, i]) in list.into_iter().take(held).enumerate() {
        let [s] = k.select_if(last.clone().eq(0), |_| [s], |k| [finished(k, s, metric)]);
        let (sv, iv) = run(&mut k, (score_out, index_out), t);
        k.store(sv, s);
        let i = k.cast::<F32, I32>(i);
        k.store(iv, i);
    }
    k.finish()
}

/// The query blocks a split kernel's blocks in flight walk every share of
/// the candidates over, as the GEMM's `group_m`.
pub const SPLIT_GROUP: usize = 8;

/// The elements one share's partial results of `n` rows span: whole merge
/// blocks, so every share starts where the merge's vector loads may (a
/// misaligned one faults on CUDA).
pub fn share_stride(n: usize) -> usize {
    n.next_multiple_of(MERGE_ROWS)
}

/// `list` (a row's `k` least scores ascending, then their indices) with the
/// score `v` at index `i` offered: in before the first score above it, the
/// rest moved one slot on and the last dropped; past the last, unchanged.
/// An equal score keeps its slot ahead of `v`: every offer comes from a
/// later candidate than the list holds, or from its trip's scores in the
/// order of their indices.
fn insert(k: &mut Kernel, list: &[Regs<F32>], v: Regs<F32>, i: Regs<F32>) -> Vec<Regs<F32>> {
    let (scores, indices) = list.split_at(list.len() / 2);
    let below: Vec<Regs<Bool>> = scores.iter().map(|&s| k.compare(v, s, CmpOp::Lt)).collect();
    let slots = |k: &mut Kernel, held: &[Regs<F32>], new: Regs<F32>| {
        (0..held.len())
            .map(|p| {
                let here = k.where_(below[p], new, held[p]);
                if p == 0 { here } else { k.where_(below[p - 1], held[p - 1], here) }
            })
            .collect::<Vec<_>>()
    };
    let scores = slots(k, scores, v);
    scores.into_iter().chain(slots(k, indices, i)).collect()
}

/// A row's list (`[rows, k]` scores ascending and their indices) with the
/// score `v` at index `i` (`[rows, 1]`) offered, as [`insert`] does it slot
/// by slot: every slot below `v` keeps its score, the first above it takes
/// `v` and the rest the one before them, the last dropped.
fn offer(k: &mut Kernel, [best, at]: [Regs<F32>; 2], v: Regs<F32>, i: Regs<F32>) -> [Regs<F32>; 2] {
    let below = k.compare(v, best, CmpOp::Lt);
    let before = k.shift(best, Axis::Col, Const::Float(f64::NEG_INFINITY));
    let moved = k.compare(v, before, CmpOp::Lt);
    let at_before = k.shift(at, Axis::Col, Const::Float(-1.0));
    let best = {
        let here = k.where_(below, v, best);
        k.where_(moved, before, here)
    };
    let here = k.where_(below, i, at);
    [best, k.where_(moved, at_before, here)]
}

/// Whether each element of `a` orders before `b`'s, both `[score, index]`:
/// a lower score, or an equal one at a smaller index.
fn before(k: &mut Kernel, [a, ai]: [Regs<F32>; 2], [b, bi]: [Regs<F32>; 2]) -> Regs<Bool> {
    let lower = k.compare(a, b, CmpOp::Lt);
    let tie = k.compare(a, b, CmpOp::Eq);
    let first = k.compare(ai, bi, CmpOp::Lt);
    let tie = k.binary(tie, first, BinaryOp::And);
    k.binary(lower, tie, BinaryOp::Or)
}

/// Whether bit `d` (a power of two) of each element of `col` is set.
fn bit(k: &mut Kernel, col: Regs<I32>, d: usize) -> Regs<Bool> {
    let shape = k.shape(col);
    let [step, pair, one, two] = [d, 2 * d, 1, 2].map(|v| k.fill::<I32>(shape, Const::Int(v as i64)));
    let (high, wide) = (k.binary(col, step, BinaryOp::Div), k.binary(col, pair, BinaryOp::Div));
    let wide = k.binary(wide, two, BinaryOp::Mul);
    let bit = k.binary(high, wide, BinaryOp::Sub);
    k.compare(bit, one, CmpOp::Eq)
}

/// A bitonic network over `seq`, tiles of `[score, index]` holding element
/// `t·w + c` of a sequence at column `c` of tile `t` (`w` their columns, a
/// power of two, as many tiles): for each block size of `sizes`, every
/// element and its partner at each distance from half the block down
/// ordered, [`before`] first in a block whose index has that size's bit
/// clear, last in the others. Sizes doubling from 2 sort the sequence; the
/// whole length alone puts a bitonic sequence in order.
fn network(k: &mut Kernel, seq: &mut [[Regs<F32>; 2]], sizes: impl IntoIterator<Item = usize>) {
    let shape = k.shape(seq[0][0]);
    let w = shape.cols;
    assert!(w.is_power_of_two() && seq.len().is_power_of_two(), "a sequence of {} runs of {w}", seq.len());
    let col = k.coord(shape, Axis::Col);
    for size in sizes {
        let mut d = size / 2;
        while d > 0 {
            if d >= w {
                // Whole tiles apart: each pair ordered once.
                let apart = d / w;
                for t in (0..seq.len()).filter(|t| t & apart == 0) {
                    let (a, b) = (seq[t], seq[t | apart]);
                    let first = before(k, a, b);
                    let lo = [0, 1].map(|j| k.where_(first, a[j], b[j]));
                    let hi = [0, 1].map(|j| k.where_(first, b[j], a[j]));
                    let up = (t * w) & size == 0;
                    (seq[t], seq[t | apart]) = if up { (lo, hi) } else { (hi, lo) };
                }
            } else {
                for (t, x) in seq.iter_mut().enumerate() {
                    // Each tile's own: one predicate of a stage shared by
                    // many tiles stayed an alloca on gfx1201 (clang 22), which
                    // the backend then moved to LDS, needing a dispatch packet.
                    let upper = bit(k, col, d);
                    // An element keeps the later of its pair where its bit
                    // `d` is set in an ascending block, clear in another.
                    let keep_last = match size >= w {
                        true if (t * w) & size == 0 => upper,
                        true => k.unary(upper, UnaryOp::Not),
                        false => {
                            let down = bit(k, col, size);
                            k.compare(upper, down, CmpOp::Ne)
                        }
                    };
                    let partner = x.map(|v| k.flip(v, Axis::Col, d as u32));
                    let ahead = before(k, partner, *x);
                    let take = k.compare(ahead, keep_last, CmpOp::Ne);
                    *x = [0, 1].map(|j| k.where_(take, partner[j], x[j]));
                }
            }
            d /= 2;
        }
    }
}

/// The lower and upper halves of two runs of `[score, index]` in order
/// merged, each in order: `b` reversed, every pair across the two ordered
/// (the lower half holding the least half), then each half's bitonic run.
fn merged(k: &mut Kernel, a: [Regs<F32>; 2], b: [Regs<F32>; 2]) -> [[Regs<F32>; 2]; 2] {
    let last = k.shape(b[0]).cols as u32 - 1;
    let rev = b.map(|t| k.flip(t, Axis::Col, last));
    let first = before(k, a, rev);
    let lo = [0, 1].map(|j| k.where_(first, a[j], rev[j]));
    let hi = [0, 1].map(|j| k.where_(first, rev[j], a[j]));
    let w = k.shape(lo[0]).cols;
    let ([mut lo], [mut hi]) = ([lo], [hi]);
    network(k, std::slice::from_mut(&mut lo), [w]);
    network(k, std::slice::from_mut(&mut hi), [w]);
    [lo, hi]
}

/// A row's least in the metric's own units, from the units it was compared
/// in: the L2 family's doubled, [`Metric::L2`]'s square root taken.
fn finished(k: &mut Kernel, best: Regs<F32>, metric: Metric) -> Regs<F32> {
    let best = if metric == Metric::NegDot {
        best
    } else {
        let two = k.fill::<F32>(k.shape(best), Const::Float(2.0));
        k.binary(best, two, BinaryOp::Mul)
    };
    if metric == Metric::L2 { k.unary(best, UnaryOp::Sqrt) } else { best }
}

/// The partial results of [`nearest`] split `splits` ways over `n` rows,
/// folded under `metric`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MergeSpec {
    pub n: usize,
    pub splits: usize,
    pub metric: Metric,
    /// The scores a row's list holds (its bound where the list is live:
    /// every slot of the bound merged).
    pub k: usize,
    /// The rows a launch binds, where they vary.
    pub live: Live,
}

/// Rows a merge block folds, one warp's.
const MERGE_ROWS: usize = 256;
/// Rows a merge block of lists folds, one warp's: a lane's row of each slot
/// in one register.
const LIST_ROWS: usize = 32;

impl MergeSpec {
    pub fn lowering(&self, device: &Device) -> Lowering {
        Lowering::uniform(device, false, WarpGrid { rows: 1, cols: 1 })
    }
}

/// Parameters in order: `score_part` f32 and `index_part` i32, `splits`
/// shares [`share_stride`] apart as a split [`nearest`] leaves them, then
/// `score [n]` f32 and `index [n]` i32: each row's least over the shares, in
/// the metric's units, and the index of the first share scoring it (its
/// candidates come before the later shares'), so the smallest index; `+∞`
/// at −1 where no share has a candidate.
///
/// With `k > 1` the shares' lists, slot `j` of share `s` at
/// `(s·k + j)·`[`share_stride`], are merged into `score [n, k]` and
/// `index [n, k]` as [`nearest`] leaves them unsplit: each share's scores
/// offered in order to a running list, so an equal score keeps the earlier
/// share's (smaller) index and the empty slots, `+∞`, are passed over.
pub fn nearest_merge(spec: &MergeSpec) -> Program {
    if spec.k > 1 {
        return merge_lists(spec);
    }
    let MergeSpec { n, splits, metric, ref live, .. } = *spec;
    let mut k = Kernel::new("nearest_merge");
    let stride = share_stride(n);
    let score_part = k.param::<F32>("score_part", ParamKind::In, splits * stride);
    let index_part = k.param::<I32>("index_part", ParamKind::In, splits * stride);
    let score = k.param::<F32>("score", ParamKind::Out, n);
    let index = k.param::<I32>("index", ParamKind::Out, n);
    let live = live.rows(&mut k, n);
    k.grid([launched(&live, n, MERGE_ROWS), Sc::from(1), Sc::from(1)]);
    k.warps(1);
    let row0 = k.block(0) * MERGE_ROWS;
    let shape = Shape::new(1, MERGE_ROWS);
    let n = bound(n, MERGE_ROWS);
    // The block's rows of the `[n]` parameter at `base`, `n` of them.
    fn rows<E: Elem>(k: &mut Kernel, p: ParamRef<E>, n: &Option<Sc>, row0: Sc, base: Sc) -> Gmem<E> {
        let v = k.view(p, base, [0, 1], Shape::new(1, MERGE_ROWS), [None, n.clone()]);
        k.at(v, 0, row0)
    }
    let best0 = k.fill::<F32>(shape, Const::Float(f64::INFINITY));
    let index0 = k.fill::<F32>(shape, Const::Float(-1.0));
    let [best, at] = k.loop_(splits, [best0, index0], |k, part, [best, at]| {
        let v = rows(k, score_part, &n, row0.clone(), part.clone() * stride);
        let s = k.load(v);
        let v = rows(k, index_part, &n, row0.clone(), part * stride);
        let i = k.load(v);
        // Exact in f32 below `MAX_CANDIDATES`, as the kernel's index.
        let i = k.cast::<I32, F32>(i);
        let lower = k.compare(s, best, CmpOp::Lt);
        [k.where_(lower, s, best), k.where_(lower, i, at)]
    });
    let best = finished(&mut k, best, metric);
    let v = rows(&mut k, score, &n, row0.clone(), Sc::from(0));
    k.store(v, best);
    let at = k.cast::<F32, I32>(at);
    let v = rows(&mut k, index, &n, row0, Sc::from(0));
    k.store(v, at);
    k.finish()
}

/// [`nearest_merge`] of `k > 1` scores a row.
fn merge_lists(spec: &MergeSpec) -> Program {
    let MergeSpec { n, splits, metric, k: top, ref live } = *spec;
    let mut k = Kernel::new("nearest_merge");
    let stride = share_stride(n);
    let score_part = k.param::<F32>("score_part", ParamKind::In, splits * top * stride);
    let index_part = k.param::<I32>("index_part", ParamKind::In, splits * top * stride);
    let score = k.param::<F32>("score", ParamKind::Out, n * top);
    let index = k.param::<I32>("index", ParamKind::Out, n * top);
    let live = live.rows(&mut k, n);
    k.grid([launched(&live, n, LIST_ROWS), Sc::from(1), Sc::from(1)]);
    k.warps(1);
    let row0 = k.block(0) * LIST_ROWS;
    let shape = Shape::new(1, LIST_ROWS);
    let n = bound(n, LIST_ROWS);
    // The block's rows of a parameter, `n` of them, row `r` at `base + r·step`.
    fn rows<E: Elem>(k: &mut Kernel, p: ParamRef<E>, n: &Option<Sc>, row0: Sc, (base, step): (Sc, usize)) -> Gmem<E> {
        let v = k.view(p, base, [0, step], Shape::new(1, LIST_ROWS), [None, n.clone()]);
        k.at(v, 0, row0)
    }
    let scores = (0..top).map(|_| k.fill::<F32>(shape, Const::Float(f64::INFINITY))).collect::<Vec<_>>();
    let init = scores.into_iter().chain((0..top).map(|_| k.fill::<F32>(shape, Const::Float(-1.0)))).collect();
    let list = k.loop_vec(splits, init, |k, part, mut list| {
        for j in 0..top {
            let at = (part.clone() * top + j) * stride;
            let v = rows(k, score_part, &n, row0.clone(), (at.clone(), 1));
            let s = k.load(v);
            let v = rows(k, index_part, &n, row0.clone(), (at, 1));
            let i = k.load(v);
            let i = k.cast::<I32, F32>(i);
            list = insert(k, &list, s, i);
        }
        list
    });
    let (list, indices) = list.split_at(top);
    for (j, (&best, &at)) in list.iter().zip(indices).enumerate() {
        let best = finished(&mut k, best, metric);
        let v = rows(&mut k, score, &n, row0.clone(), (Sc::from(j), top));
        k.store(v, best);
        let at = k.cast::<F32, I32>(at);
        let v = rows(&mut k, index, &n, row0.clone(), (Sc::from(j), top));
        k.store(v, at);
    }
    k.finish()
}
