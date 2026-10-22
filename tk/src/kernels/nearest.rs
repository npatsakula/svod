//! Each query row's cheapest candidate: one `x·cᵀ` product per candidate
//! tile on the matrix core, its scores folded into a carried running minimum
//! and its candidate index rather than a softmax. The query rows stay
//! resident in registers while candidate tiles stream through a shared ring,
//! or, at a depth the registers cannot hold, both stream along the depth
//! through the GEMM's K pipeline once per candidate tile. The candidates may
//! split over several blocks per query block, each leaving a partial least
//! and index (or list) that [`nearest_merge`] folds. A row's `k` least are a list in
//! registers fed by its lanes' queues, or past what they hold a register
//! queue merged into the rest held in the outputs; or the kernel stores the
//! scores as a matrix, block by block, whose rows [`nearest_select`] lists a
//! warp each.

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
    /// How the lanes queue a row's candidates in front of its list (`k > 1`
    /// listed by the kernel itself); `None` at `k = 1`.
    pub lanes: Option<LaneQueue>,
}

/// A row's queues in front of its list: each lane holding a row deals out
/// its columns of a trip's scores a register at a time and queues those
/// below the row's `k`-th, `depth` deep, until some lane's queue is full and
/// the warp merges every queue into its rows' lists.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaneQueue {
    /// A row's queues side by side, the columns a deal takes of a row: a
    /// power of two, a multiple of the lanes a product spreads a row over,
    /// so a deal renames registers.
    pub width: usize,
    /// The scores a queue holds, a power of two.
    pub depth: usize,
    /// A trip's scores offered a deal at a time in passes, each stopping
    /// at a deal some full queue would take another of; else in rounds of
    /// each lane's least not yet offered. Rounds vote less, but hold a
    /// trip's deals as they go: on gfx1201 16 deals a trip at depth 768 held
    /// 211–221 registers to the passes' 189–191, and ran 13 % behind them.
    pub passes: bool,
}

impl LaneQueue {
    /// What a list config without its own takes: as wide as the lanes
    /// CUDA's and RDNA's swapped products spread a row over, or more.
    pub const PORTABLE: Self = Self { width: 4, depth: 2, passes: false };
}

/// Rounds print as lane queues did before passes were an option.
impl fmt::Debug for LaneQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("LaneQueue");
        s.field("width", &self.width).field("depth", &self.depth);
        if self.passes {
            s.field("passes", &true);
        }
        s.finish()
    }
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
        if let Some(lanes) = self.lanes {
            s.field("lanes", &lanes);
        }
        s.finish()
    }
}

impl NearestCfg {
    pub const fn new(bq: usize, bkv: usize, stages: usize) -> Self {
        Self {
            bq,
            bkv,
            stages,
            unroll: false,
            warp_rows: WARP_ROWS,
            bk: None,
            splits: 1,
            queue: None,
            select: None,
            lanes: None,
        }
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

    /// The lane queues a list takes ([`LaneQueue::PORTABLE`] unless set).
    pub fn lane_queue(&self) -> LaneQueue {
        self.lanes.unwrap_or(LaneQueue::PORTABLE)
    }

    /// The slots of a list of `k` a row holds in registers: queued, the
    /// queue's, else the list's in whole pairs of the lane queues' width, a
    /// power of two.
    pub fn list_slots(&self, k: usize) -> usize {
        self.queue.unwrap_or(k.next_power_of_two().max(self.lane_queue().width))
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
/// slots past its admissible candidates `+∞` at −1: a sorted list carried
/// in registers as `[bq, w]` pairs ([`LaneQueue`]: their columns spread
/// over the lanes holding the same rows,
/// [`crate::layouts::TileLayout::spread_row`]), in front of which each lane
/// queues the scores of its own columns below its row's `k`-th. A trip some
/// row of the warp has a score below its `k`-th in ([`Kernel::if_any`])
/// deals its scores out `w` a row at a time ([`TileOp::Deal`], a renaming
/// of registers), a deal some row has one below the `k`-th in pushing them
/// onto the lanes' queues; a deal some lane's full queue would take another
/// of stops the pass, the warp sorts every queue (a bitonic network,
/// [`TileOp::Flip`] across the lanes) and merges them into its rows' lists,
/// takes the new `k`-th, and deals again from there. Split, share `s` leaves
/// row `i`'s list at `(s·`[`share_stride`]` + i)·k`, in the units the
/// scores are compared in, for [`nearest_merge`].
///
/// [Queued](NearestCfg::queue), the register list is a queue of the scores
/// below the row's `k`-th, and the list proper is the outputs' (`+∞` at −1
/// to start; split, its share's): once some row's could not take another
/// lanes' queue whole the warp merges every row's into it, run by run of the
/// queue's length (the lower half of a bitonic merge of the run and the
/// carried queue stored, the upper half carried on), and takes the new
/// `k`-th.
/// Under [`NearestCfg::select`] the outputs are `scores [n, m]`, each pair's
/// score in the units compared, and [filtered](SelectCfg::filter) `mins
/// [n, ⌈m / bkv⌉]`, each row's least of every trip's candidates.
///
/// The L2 family's scores are compared halved (`c_sq/2 − x·c`, plus `x_sq/2`,
/// against halved floors; [`Metric::L2`] against its floors clamped at zero
/// and squared, its square root taken once per row), which orders them as
/// the metric's own.
pub fn nearest<T: MmaIn>(spec: &NearestSpec) -> Program {
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
        cfg.queue.is_none_or(|q| q.is_power_of_two() && (2..top).contains(&q)),
        "a queue of {held} below a list of {top}"
    );
    assert!(cfg.select.is_none() || cfg.queue.is_none(), "a score matrix unqueued");
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
    let mins = cfg.select.is_some_and(|s| s.filter).then(|| k.param::<F32>("mins", ParamKind::Out, n * tiles));
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

    // A row's list (`k > 1`): `[score, index]` pairs of `[bq, w]` tiles,
    // its sequence's element `t·w + c` at column `c` of pair `t`, ascending,
    // `+∞` at −1 past the candidates it holds (see [`Seq`]); in front of it
    // a queue of `depth` pairs, column `c` of each a queue a lane holds of
    // the scores it is dealt below the row's `k`-th, unsorted.
    let LaneQueue { width: w, depth: slots, .. } = cfg.lane_queue();
    assert!(
        top == 1 || matrix || (w.is_power_of_two() && w > 1 && slots.is_power_of_two() && bkv.is_multiple_of(w)),
        "lane queues {:?} dealing {bkv} candidates",
        cfg.lanes
    );
    let lane = Shape::new(bq, w);
    // The slots the register list holds: the list's (its bound's where it
    // is live) in a power of two of whole pairs; queued, the queue's.
    let held = cfg.list_slots(top);
    let pairs_held = held / w;
    // The list's runs of `held` slots in the outputs (queued), and the
    // slots a row's list holds there: its bound's even where it is live,
    // the slots past the live ones its last run reads masked, those it
    // writes left out of an output's shape.
    let kb = bound(top, held);
    // A row's list in the outputs: row-major, split, in the share's own
    // `[stride, k]` region.
    let list_base = share.as_ref().map_or(Sc::from(0), |(part, ..)| part.clone() * (share_stride(n) * top));
    let pair_view = |k: &mut Kernel, p: ParamRef<F32>, c: Sc, t: usize| {
        let v = k.view(p, list_base.clone(), [top, 1], lane, [rb.clone(), kb.clone()]);
        k.at(v, q_off.clone(), c * held + t * w)
    };
    let index_view = |k: &mut Kernel, c: Sc, t: usize| {
        let v =
            k.view(index_out.expect("an index output"), list_base.clone(), [top, 1], lane, [rb.clone(), kb.clone()]);
        k.at(v, q_off.clone(), c * held + t * w)
    };
    let runs = top.div_ceil(held);
    let tail = top - (runs - 1) * held;
    let (run_count, last_run, live_tail) = match &k_live {
        None => (Sc::from(runs), Sc::from(runs - 1), None),
        Some(live) => {
            let runs = (live.clone() + (held - 1)) / held;
            (runs.clone(), runs.clone() - 1, Some(live.clone() - (runs - 1) * held))
        }
    };
    // The last run's slots, the live list's where it varies.
    let tail_len = || live_tail.clone().unwrap_or(Sc::from(tail));
    // The register list's `k`-th slot (the live length's where it varies).
    let kth_slot = k_live.clone().map_or(Sc::from(top - 1), |live| live - 1);

    // The queue (sorted ascending) merged into the rest of the list held
    // in the outputs (in the units the scores are compared in; `last`: the
    // metric's own, the list complete): run by run of `held` slots, each the
    // lower half of merging it with the queue's carry, which the upper half
    // goes on as. Returns the list's `k`-th score and index (the vote's),
    // unless `last`.
    let flush = |k: &mut Kernel, queue: Seq, last: bool| {
        let unset = [f64::INFINITY, -1.0].map(|e| k.fill::<F32>(rows, Const::Float(e)));
        let init = flat(&queue).chain(if last { vec![] } else { unset.to_vec() }).collect();
        let out = k.loop_vec(run_count.clone(), init, |k, c, carried| {
            let mut run: Seq = (0..pairs_held)
                .map(|t| {
                    let v = pair_view(k, score_out, c.clone(), t);
                    let v = k.load(v);
                    let i = index_view(k, c.clone(), t);
                    let i = k.load(i);
                    [v, k.cast::<I32, F32>(i)]
                })
                .collect();
            let ends = c.clone().eq(last_run.clone());
            if tail < held || live_tail.is_some() {
                let masked = k.select_if_vec(
                    ends.clone(),
                    |k| {
                        let col = k.coord(lane, Axis::Col);
                        let empty = empty_seq(k, lane, 1)[0];
                        let mut out = vec![];
                        for (t, pair) in run.iter().enumerate() {
                            let width = k.splat::<I32>(lane, tail_len() - t * w);
                            let inside = k.compare(col, width, CmpOp::Lt);
                            out.extend([0, 1].map(|j| k.where_(inside, pair[j], empty[j])));
                        }
                        out
                    },
                    |_| flat(&run).collect(),
                );
                run = pairs(&masked);
            }
            let (lo, hi) = merged(k, &run, &pairs(&carried[..2 * pairs_held]));
            for (t, [s, i]) in lo.iter().enumerate() {
                // A share's list stays in the units the merge compares.
                let stored = if last && !split { finished(k, *s, metric) } else { *s };
                let v = pair_view(k, score_out, c.clone(), t);
                k.store(v, stored);
                let i = k.cast::<F32, I32>(*i);
                let v = index_view(k, c.clone(), t);
                k.store(v, i);
            }
            if last {
                return flat(&hi).collect();
            }
            let kth = k.select_if(
                ends,
                |k| element(k, &lo, tail_len() - 1),
                |_| [carried[2 * pairs_held], carried[2 * pairs_held + 1]],
            );
            flat(&hi).chain(kth).collect()
        });
        (!last).then(|| [out[2 * pairs_held], out[2 * pairs_held + 1]])
    };

    // The carried state of a list: the register list, the lanes' queue,
    // and the vote's `k`-th score and index (queued: the outputs' list's).
    let state = |v: &[Regs<F32>]| {
        let (list, rest) = v.split_at(2 * pairs_held);
        let (queue, kth) = rest.split_at(2 * slots);
        (pairs(list), pairs(queue), [kth[0], kth[1]])
    };
    let packed = |list: &Seq, queue: &Seq, kth: [Regs<F32>; 2]| -> Vec<Regs<F32>> {
        flat(list).chain(flat(queue)).chain(kth).collect()
    };
    // The lanes' queue merged into the register list, the vote's `k`-th
    // taken from it; queued, into the queue held in registers, which a
    // warp merges into the outputs once some row's could not take another
    // lanes' queue whole.
    let merge = |k: &mut Kernel, carried: &[Regs<F32>]| -> Vec<Regs<F32>> {
        let (list, queue, kth) = state(carried);
        let list = queued_into(k, list, queue);
        let fresh = empty_seq(k, lane, slots);
        if cfg.queue.is_none() {
            let kth = element(k, &list, kth_slot.clone());
            return packed(&list, &fresh, kth);
        }
        let room = held - (slots * w).min(held);
        let filled = element(k, &list, Sc::from(room))[1];
        let unset = k.fill::<F32>(rows, Const::Float(-1.0));
        let full = k.compare(unset, filled, CmpOp::Lt);
        let kept = packed(&list, &fresh, kth);
        k.if_any(
            full,
            |k| {
                let kth = flush(k, list.clone(), false).expect("a vote");
                packed(&empty_seq(k, lane, pairs_held), &fresh, kth)
            },
            kept,
        )
    };
    // A trip's scores are dealt out a pair of `[bq, w]` tiles at a time.
    let deals = bkv / w;
    let width = 2 * pairs_held + 2 * slots + 2;
    // A round of the trip's scores `s` not yet offered (`rest`, one tile a
    // deal, those offered `+∞`), after the carried list, queues and `k`-th
    // in `st`: each lane takes the least of its own (the smallest column
    // among equal ones) where it is at most the row's `k`-th, pushed onto its
    // queue if it orders before the `k`-th, and marks it offered. Some lane
    // whose queue is full taking one, the warp merges the queues first.
    // Returns the state with the rest and, last, each lane's least of it.
    let round = |k: &mut Kernel, st: &[Regs<F32>], s: Regs<F32>, start: &Sc| -> Vec<Regs<F32>> {
        let (rest, low) = (&st[width..width + deals], st[width + deals]);
        let kth = st[width - 2];
        let taking = k.compare(low, kth, CmpOp::Le);
        let unset = k.fill::<F32>(lane, Const::Float(-1.0));
        let full = k.compare(unset, st[2 * pairs_held + 2 * slots - 1], CmpOp::Lt);
        let blocked = k.binary(taking, full, BinaryOp::And);
        let st = k.if_any(blocked, |k| merge(k, &st[..width]), st[..width].to_vec());
        let (list, queue, [kth, kth_i]) = state(&st);
        let cols: Vec<Regs<F32>> = (0..deals)
            .map(|j| {
                let col = k.deal_col(s, w, j);
                k.cast::<I32, F32>(col)
            })
            .collect();
        let none = k.fill::<F32>(lane, Const::Float(f64::INFINITY));
        // The least's column: the smallest of those scoring it.
        let at = rest.iter().zip(&cols).fold(none, |at, (&v, &col)| {
            let here = k.compare(v, low, CmpOp::Eq);
            let col = k.where_(here, col, none);
            k.binary(at, col, BinaryOp::Min)
        });
        let origin = k.splat::<F32>(lane, start.clone());
        let index = k.binary(at, origin, BinaryOp::Add);
        // Below the `k`-th, or tying it at a smaller index.
        let lower = k.compare(low, kth, CmpOp::Lt);
        let tie = k.compare(low, kth, CmpOp::Eq);
        let earlier = k.compare(index, kth_i, CmpOp::Lt);
        let tie = k.binary(tie, earlier, BinaryOp::And);
        let ins = k.binary(lower, tie, BinaryOp::Or);
        let queue = pushed(k, &queue, ins, [low, index]);
        let rest: Vec<Regs<F32>> = rest
            .iter()
            .zip(&cols)
            .map(|(&v, &col)| {
                let taken = k.compare(col, at, CmpOp::Eq);
                k.where_(taken, none, v)
            })
            .collect();
        let low = rest.iter().skip(1).fold(rest[0], |low, &v| k.binary(low, v, BinaryOp::Min));
        packed(&list, &queue, [kth, kth_i]).into_iter().chain(rest).chain([low]).collect()
    };
    // Whether some lane has a score it is still to offer, at most its row's
    // `k`-th, in the state a round leaves.
    let pending = |k: &mut Kernel, st: &[Regs<F32>]| k.compare(st[width + deals], st[width - 2], CmpOp::Le);
    // The tile's scores offered to the row's carried list: where some row of
    // the warp has a score at most its `k`-th, round after round each lane
    // offers the least of its own not yet offered, until none is left at most
    // its row's `k`-th; a round pushes one a lane at most, so `deals` rounds
    // offer them all.
    let fold_list = |k: &mut Kernel, dot: Regs<F32>, start: Sc, carried: Vec<Regs<F32>>| {
        let s = scores(k, dot, start.clone());
        let low = k.reduce(s, Axis::Row, ReduceOp::Min);
        let kth = carried[width - 2];
        let near = k.compare(low, kth, CmpOp::Le);
        k.if_any(
            near,
            |k| {
                let rest: Vec<Regs<F32>> = (0..deals).map(|j| k.deal(s, w, j)).collect();
                let low = rest.iter().skip(1).fold(rest[0], |low, &v| k.binary(low, v, BinaryOp::Min));
                let st: Vec<Regs<F32>> = carried.iter().copied().chain(rest).chain([low]).collect();
                let st = round(k, &st, s, &start);
                let more = pending(k, &st);
                let st = k.if_any(
                    more,
                    |k| {
                        k.loop_vec(deals - 1, st.clone(), |k, _, st| {
                            let more = pending(k, &st);
                            k.if_any(more, |k| round(k, &st, s, &start), st.clone())
                        })
                    },
                    st.clone(),
                );
                st[..width].to_vec()
            },
            carried.clone(),
        )
    };

    // A trip's deals, from the first not yet taken: none past one a lane's
    // full queue cannot take, the queues merged after it, then again from
    // it; each pass deals at least `slots` past a merge.
    let passes = deals.div_ceil(slots) + 1;
    // [Passes](LaneQueue::passes): the tile's scores offered to the row's
    // carried list: while some row of the warp has a score below its
    // `k`-th, each deal's below it pushed
    // onto its lane's queue (equal scores taken too, the vote's slot's index
    // deciding which order before it). A pass stops at a deal some lane's
    // full queue would take another of, merges, and the next starts there.
    let fold_passes = |k: &mut Kernel, dot: Regs<F32>, start: Sc, carried: Vec<Regs<F32>>| {
        let s = scores(k, dot, start.clone());
        let low = k.reduce(s, Axis::Row, ReduceOp::Min);
        let kth = carried[carried.len() - 2];
        let near = k.compare(low, kth, CmpOp::Le);
        // The trip's deals from where `from` says (a row's first deal to take,
        // `deals` for none) up to one some lane's full queue would take
        // another of, the state's queue pushed onto: the state with the deal
        // a pass stopped at (else `deals`) and whether it stopped. A first
        // pass takes the rows with a score below their `k`-th whole.
        let (at_from, at_stop) = (2 * pairs_held + 2 * slots + 2, 2 * pairs_held + 2 * slots + 3);
        let pass = |k: &mut Kernel, carried: &[Regs<F32>], from: Option<Regs<F32>>| -> Vec<Regs<F32>> {
            let (list, queue, [kth, kth_i]) = state(carried);
            let origin = k.splat::<F32>(lane, start.clone());
            let (half, end) = (k.fill::<F32>(rows, Const::Float(0.5)), k.fill::<F32>(rows, Const::Float(deals as f64)));
            // A deal changes the queue alone, where the pass stopped and that it did.
            let (q_from, q_stop) = (2 * slots, 2 * slots + 1);
            let mut q: Vec<Regs<F32>> = flat(&queue).chain([end, k.zeros::<F32>(rows)]).collect();
            for j in 0..deals {
                let queue = pairs(&q[..q_from]);
                let v = k.deal(s, w, j);
                let at = k.fill::<F32>(rows, Const::Float(j as f64));
                let running = k.compare(q[q_stop], half, CmpOp::Lt);
                let go = match from {
                    Some(from) => {
                        let due = k.compare(from, at, CmpOp::Le);
                        k.binary(due, running, BinaryOp::And)
                    }
                    None if j == 0 => near,
                    None => k.binary(near, running, BinaryOp::And),
                };
                let below = k.compare(v, kth, CmpOp::Le);
                let hit = k.binary(below, go, BinaryOp::And);
                q = k.if_any(
                    hit,
                    |k| {
                        let col = k.deal_col(s, w, j);
                        let col = k.cast::<I32, F32>(col);
                        let index = k.binary(col, origin, BinaryOp::Add);
                        // Below the `k`-th, or tying it at a smaller index.
                        let lower = k.compare(v, kth, CmpOp::Lt);
                        let earlier = k.compare(index, kth_i, CmpOp::Lt);
                        let first = k.binary(lower, earlier, BinaryOp::Or);
                        let ins = k.binary(first, hit, BinaryOp::And);
                        let unset = k.fill::<F32>(lane, Const::Float(-1.0));
                        let full = k.compare(unset, queue[slots - 1][1], CmpOp::Lt);
                        let blocked = k.binary(ins, full, BinaryOp::And);
                        let mut stopped = q.clone();
                        stopped[q_from] = at;
                        stopped[q_stop] = k.fill::<F32>(rows, Const::Float(1.0));
                        let queue = pushed(k, &queue, ins, [v, index]);
                        let taken = flat(&queue).chain([q[q_from], q[q_stop]]).collect();
                        k.if_any(blocked, |_| stopped, taken)
                    },
                    q.clone(),
                );
            }
            packed(&list, &pairs(&q[..q_from]), [kth, kth_i]).into_iter().chain([q[q_from], q[q_stop]]).collect()
        };
        // The queues merged once a pass stopped, then the deals again from
        // where it did.
        let merged_from = |k: &mut Kernel, st: Vec<Regs<F32>>| -> Vec<Regs<F32>> {
            let half = k.fill::<F32>(rows, Const::Float(0.5));
            let halted = k.compare(half, st[at_stop], CmpOp::Lt);
            k.if_any(
                halted,
                |k| merge(k, &st[..at_from]).into_iter().chain([st[at_from], st[at_stop]]).collect(),
                st.clone(),
            )
        };
        k.if_any(
            near,
            |k| {
                let first = pass(k, &carried, None);
                let half = k.fill::<F32>(rows, Const::Float(0.5));
                let halted = k.compare(half, first[at_stop], CmpOp::Lt);
                let out = k.if_any(
                    halted,
                    |k| {
                        let st = merged_from(k, first.clone());
                        k.loop_vec(passes - 1, st, |k, _, st| {
                            let end = k.fill::<F32>(rows, Const::Float(deals as f64));
                            let pending = k.compare(st[at_from], end, CmpOp::Lt);
                            k.if_any(
                                pending,
                                |k| {
                                    let next = pass(k, &st[..at_from], Some(st[at_from]));
                                    merged_from(k, next)
                                },
                                st.clone(),
                            )
                        })
                    },
                    first.clone(),
                );
                out[..at_from].to_vec()
            },
            carried.clone(),
        )
    };

    let mut init = vec![];
    if !matrix && top == 1 {
        init = vec![k.fill::<F32>(rows, Const::Float(f64::INFINITY)), k.fill::<F32>(rows, Const::Float(-1.0))];
    }
    if top > 1 && !matrix {
        let list = empty_seq(&mut k, lane, pairs_held);
        let queue = empty_seq(&mut k, lane, slots);
        let kth = [f64::INFINITY, -1.0].map(|e| k.fill::<F32>(rows, Const::Float(e)));
        init = packed(&list, &queue, kth);
    }
    // A filtered score matrix's trips' leasts, a run of [`MINS_RUN`] held to
    // be stored at once.
    let run = Shape::new(bq, MINS_RUN);
    if mins.is_some() {
        init = vec![k.fill::<F32>(run, Const::Float(f64::INFINITY))];
    }
    // Queued, the block's lists start empty in the outputs, held as the
    // queue is: a lane reads back the slots it writes.
    if cfg.queue.is_some() {
        let [score0, index0] = empty_seq(&mut k, lane, 1)[0];
        let index0 = k.cast::<F32, I32>(index0);
        k.loop_vec::<F32>(run_count.clone(), vec![], |k, c, _| {
            for t in 0..pairs_held {
                let v = pair_view(k, score_out, c.clone(), t);
                k.store(v, score0);
                let v = index_view(k, c.clone(), t);
                k.store(v, index0);
            }
            vec![]
        });
    }
    let trips = count.clone();
    // A score matrix's trip stores its scores and each row's least of them.
    let folded = |k: &mut Kernel, dot: Regs<F32>, start: Sc, carried: Vec<Regs<F32>>| match (matrix, top) {
        (true, _) => {
            let s = scores(k, dot, start.clone());
            let v = k.view(score_out, 0, [m, 1], shape, [rb.clone(), bound(m, bkv)]);
            let v = k.at(v, q_off.clone(), start.clone());
            k.store(v, s);
            let Some(mins) = mins else { return vec![] };
            // The trip's place in the share's runs: each run stored whole
            // once its last trip (or the share's) is in, past the share's
            // trips left out.
            let low = k.reduce(s, Axis::Row, ReduceOp::Min);
            let (from, count) = (first.clone().unwrap_or(Sc::from(0)), trips.clone());
            let trip = start / bkv - from.clone();
            let slot = trip.clone() % MINS_RUN;
            let col = k.coord(run, Axis::Col);
            let at = k.splat::<I32>(run, slot.clone());
            let here = k.compare(col, at, CmpOp::Eq);
            let held = k.where_(here, low, carried[0]);
            let full = slot.clone().eq(MINS_RUN - 1).or(trip.clone().eq(count.clone() - 1));
            k.if_(
                full,
                |k| {
                    let v = k.view(mins, 0, [tiles, 1], run, [rb.clone(), Some(from.clone() + count.clone())]);
                    let v = k.at(v, q_off.clone(), from.clone() + trip - slot);
                    k.store(v, held);
                },
                |_| {},
            );
            vec![held]
        }
        (false, 1) => fold(k, dot, start, carried[0], carried[1]).to_vec(),
        (false, _) if cfg.lane_queue().passes => fold_passes(k, dot, start, carried),
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
                let zero = k.zeros::<T::Acc>(shape);
                let dot = k.mma(zero, q_r, false, c_t, true);
                let dot = T::widened(k, dot);
                folded(k, dot, start, carried)
            },
        ),
        _ => k.loop_vec(count, init, |k, step, carried| {
            let start = tile_start(step);
            let c_g = k.at(c_view, start.clone(), 0);
            let x_chunk = |k: &mut Kernel, _: Sc, col: Sc| k.at(x_view, 0, col);
            let chunks = d.div_ceil(depth);
            let [dot] = mainloop(k, Shape::new(bq, depth), &x_chunk, [c_g], chunks, stages, depth, None, 1);
            let dot = T::widened(k, dot);
            folded(k, dot, start, carried)
        }),
    };
    if matrix {
        return k.finish();
    }
    if top == 1 {
        // A share's partial results, a row's least in the units the merge
        // compares (split), else in the metric's.
        let [best, index] = carried[..] else { unreachable!("a score and an index") };
        let (best, at) = match &share {
            Some((part, ..)) => (best, part.clone() * share_stride(n)),
            None => (finished(&mut k, best, metric), Sc::from(0)),
        };
        let v = row_view(&mut k, score_out, (at.clone(), 1), rb.clone(), bq, q_off.clone());
        k.store(v, best);
        let index = k.cast::<F32, I32>(index);
        let v = row_view(&mut k, index_out.expect("an index output"), (at, 1), rb, bq, q_off);
        k.store(v, index);
        return k.finish();
    }
    // What the lanes still queue merged in.
    let (list, queue, _) = state(&carried);
    let list = queued_into(&mut k, list, queue);
    if cfg.queue.is_some() {
        flush(&mut k, list, true);
        return k.finish();
    }
    // A row's list at `j` of its `top`, in the metric's units; split, in
    // its share's region in the units the merge compares.
    for (t, [s, i]) in list.into_iter().take(top.div_ceil(w)).enumerate() {
        let s = if split { s } else { finished(&mut k, s, metric) };
        let v = k.view(score_out, list_base.clone(), [top, 1], lane, [rb.clone(), Some(Sc::from(top))]);
        let v = k.at(v, q_off.clone(), t * w);
        k.store(v, s);
        let i = k.cast::<F32, I32>(i);
        let index = index_out.expect("an index output");
        let v = k.view(index, list_base.clone(), [top, 1], lane, [rb.clone(), Some(Sc::from(top))]);
        let v = k.at(v, q_off.clone(), t * w);
        k.store(v, i);
    }
    k.finish()
}

/// How [`nearest_select`] lists a score matrix's rows.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SelectCfg {
    /// Rows a block lists, a warp each.
    pub warps: u32,
    /// The scores a lane queues below its row's `k`-th before the warp
    /// merges its lanes' queues into the row's list.
    pub queue: usize,
    /// The scores [filtered](SelectSpec::filter) by the `k`-th least of
    /// their trips' leasts, which the score matrix's [`nearest`] stores
    /// beside them.
    pub filter: bool,
}

/// The tune store keys by this text: an unfiltered config prints as it did
/// before the filter existed.
impl fmt::Debug for SelectCfg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("SelectCfg");
        s.field("warps", &self.warps).field("queue", &self.queue);
        if self.filter {
            s.field("filter", &true);
        }
        s.finish()
    }
}

impl SelectCfg {
    pub fn lowering(&self, device: &Device) -> Lowering {
        Lowering::uniform(device, false, WarpGrid { rows: self.warps, cols: 1 })
    }

    /// The scores a lane reads a step on a target of `wave` lanes:
    /// [`SELECT_WIDE`] at most, half its queue, so a merge leaves room for
    /// several steps; one where a wave holds more than a row's [`LANES`].
    pub fn wide(&self, wave: u32) -> usize {
        match wave as usize > LANES {
            true => 1,
            false => SELECT_WIDE.min(self.queue / 2).max(1),
        }
    }

    /// Whether it lists `k` scores a row: a lane's queue (a power of two)
    /// fits the list's runs of [`LANES`].
    pub fn fits(&self, k: usize) -> bool {
        self.queue.is_power_of_two() && self.queue <= select_runs(k) && self.warps.is_power_of_two()
    }
}

/// The trips whose leasts a filtered score matrix stores at once, a
/// row's run of them a sector.
const MINS_RUN: usize = 8;

/// The scores a lane of [`nearest_select`] reads a step at most, one vector
/// load.
const SELECT_WIDE: usize = 4;

/// The most scores a block of a score matrix holds where the device's
/// memory is not known (1 GiB of them).
pub const SCRATCH: usize = 1 << 28;

/// The share of device memory a block of a score matrix takes at most.
const SCRATCH_SHARE: u64 = 8;

/// The most scores a block of a score matrix holds on a device of `memory`
/// bytes: an eighth of it, else [`SCRATCH`].
pub fn scratch(memory: Option<u64>) -> usize {
    memory.map_or(SCRATCH, |bytes| usize::try_from(bytes / SCRATCH_SHARE / 4).unwrap_or(usize::MAX))
}

/// The rows and candidates of each block a score matrix of `n` rows by `m`
/// candidates (tiles of `cfg`, `k` scores a row) is computed and listed
/// in, a block holding `scratch` scores at most: whole rows of candidates
/// where it holds them, else as many candidates as it holds of every row, at
/// least `8·k` (each block reads and writes a list a row), and then as many
/// rows as it holds of them.
pub fn select_blocks(n: usize, m: usize, k: usize, cfg: &NearestCfg, scratch: usize) -> (usize, usize) {
    let cols = m.min((scratch / n).max(8 * k).next_multiple_of(cfg.bkv));
    let rows = n.min((scratch / cols).max(cfg.bq) / cfg.bq * cfg.bq);
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
    /// The scores a lane reads a step ([`SelectCfg::wide`]).
    pub wide: usize,
    /// The candidates of a group whose least the inputs hold, where a step
    /// of [`LANES`] scores none of whose groups' least is at most a row's
    /// bound is passed over unread.
    pub filter: Option<usize>,
    /// The blocks a row's steps split over, each listing an even share of
    /// them into lists [`nearest_merge`] folds; 1: one block lists them
    /// all.
    pub shares: usize,
}

impl SelectSpec {
    /// The steps a row's scores take.
    pub fn steps(&self) -> usize {
        self.cols.div_ceil(LANES * self.wide)
    }

    /// The groups of [`Self::filter`]'s width a row's scores fall in.
    pub fn groups(&self) -> Option<usize> {
        self.filter.map(|g| self.cols.div_ceil(g))
    }
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
///
/// [Filtered](SelectSpec::filter), the inputs go on with `mins [n, groups]`,
/// each row's least of every `g` columns (a score-matrix [`nearest`]'s
/// trips'), and `bound [n, k]`, lists whose `k`-th score (the live
/// length's) bounds the row's `k`-th from above: only a score at most it is
/// offered, and a step none of whose groups' least is at most it is passed
/// over unread (whatever the list's `k`-th, so which steps a row reads is
/// its own). The `k` least of the groups'
/// leasts are `k` candidates, so their `k`-th is such a bound, and each of
/// a row's `k` least lies in a group whose least is at most it.
///
/// [Shared](SelectSpec::shares) (the lists starting empty), grid y takes an
/// even share of a row's steps, and the outputs are `score_part` and
/// `index_part`, [`share_stride`] apart: each share's lists in the units
/// compared, for [`nearest_merge`].
pub fn nearest_select(spec: &SelectSpec) -> Program {
    let SelectSpec { n, cols, k: top, metric, first, cfg, ref live, wide, filter, shares } = *spec;
    assert!(cfg.fits(top), "{cfg:?} lists {top}");
    let (rows, depth, runs) = (cfg.warps as usize, cfg.queue, select_runs(top));
    // A step reads `wide` scores a lane, as many as its queue takes at once.
    assert!(wide.is_power_of_two() && 2 * wide <= cfg.queue.max(2), "{wide} scores a lane a step of {cfg:?}");
    let width = LANES * wide;
    let steps = spec.steps();
    assert!(shares == 1 || (first && (1..=steps).contains(&shares)), "{shares} shares of {steps} steps");
    let mut k = Kernel::new("nearest_select");
    let scores = k.param::<F32>("scores", ParamKind::In, n * cols);
    let chunk = k.param::<I32>("chunk", ParamKind::In, 2);
    let lists = (!first).then(|| {
        (k.param::<F32>("score_in", ParamKind::In, n * top), k.param::<I32>("index_in", ParamKind::In, n * top))
    });
    let groups = spec.groups();
    let filtered = groups.map(|groups| {
        (k.param::<F32>("mins", ParamKind::In, n * groups), k.param::<F32>("bound", ParamKind::In, n * top))
    });
    let parts = if shares > 1 { shares * share_stride(n) } else { n };
    let names = if shares > 1 { ["score_part", "index_part"] } else { ["score", "index"] };
    let score_out = k.param::<F32>(names[0], ParamKind::Out, parts * top);
    let index_out = k.param::<I32>(names[1], ParamKind::Out, parts * top);
    let (base, last) = (k.load_scalar(chunk, 0), k.load_scalar(chunk, 1));
    let rows_live = live.rows(&mut k, n);
    let k_live = live.k(&mut k);
    // The rows and slots of the capacity, as `nearest`'s live views.
    let (rb, kb) = (bound(n, rows), bound(top, LANES));
    k.grid([launched(&rows_live, n, rows), Sc::from(shares), Sc::from(1)]);
    k.warps(cfg.warps);
    let row0 = k.block(0) * rows;
    // The share's first step and how many it takes.
    let per = steps.div_ceil(shares);
    let (step0, count) = match shares {
        1 => (Sc::from(0), Sc::from(steps)),
        _ => {
            let step0 = k.block(1) * per;
            (step0.clone(), (Sc::from(steps) - step0).min(per))
        }
    };
    let out_base = if shares > 1 { k.block(1) * (share_stride(n) * top) } else { Sc::from(0) };
    let lane = Shape::new(rows, LANES);
    let empty = |k: &mut Kernel| [Const::Float(f64::INFINITY), Const::Float(-1.0)].map(|e| k.fill::<F32>(lane, e));
    // Run `t` of the block's lists in `score` and `index` from `base`.
    let run = |k: &mut Kernel, (score, index): (ParamRef<F32>, ParamRef<I32>), base: Sc, t: usize| {
        let bounds = [rb.clone(), kb.clone()];
        let s = k.view(score, base.clone(), [top, 1], lane, bounds.clone());
        let s = k.at(s, row0.clone(), t * LANES);
        let i = k.view(index, base, [top, 1], lane, bounds);
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
                let (s, i) = run(k, lists.expect("the lists to carry"), Sc::from(0), t);
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
    // A step past the share's last merges what is still queued.
    let wide_lane = Shape::new(rows, width);
    let view = k.view(scores, 0, [cols, 1], wide_lane, [rb.clone(), bound(cols, width)]);
    // Filtered, each row's bound, and a step's groups' least.
    let column = Shape::new(rows, 1);
    let bounded = filtered.map(|(mins, bound_in)| {
        let v = k.view(bound_in, 0, [top, 1], column, [rb.clone(), None]);
        let slot = k_live.clone().map_or(Sc::from(top - 1), |live| live - 1);
        let v = k.at(v, row0.clone(), slot);
        (mins, k.load(v))
    });
    let group = filter.unwrap_or(LANES);
    let (spanned, groups) = ((width / group).max(1), groups.unwrap_or(0));
    // A row's least of the groups a step spans, one column of them at a time
    // (each its row's own), past the last group none.
    let least = |k: &mut Kernel, mins: ParamRef<F32>, step: Sc| {
        let at = if group <= width { step * (width / group) } else { step / (group / width) };
        let v = k.view(mins, 0, [groups, 1], column, [rb.clone(), Some(Sc::from(groups))]);
        let lows: Vec<Regs<F32>> = (0..spanned)
            .map(|g| {
                let v = k.at(v, row0.clone(), at.clone() + g);
                k.load(v)
            })
            .collect();
        lows.iter().skip(1).fold(lows[0], |low, &v| k.binary(low, v, BinaryOp::Min))
    };
    // A step's scores, `+∞` past `cols`.
    let scored = |k: &mut Kernel, step: Sc| {
        let at = k.at(view, row0.clone(), step.clone() * width);
        let s = k.load(at);
        if cols.is_multiple_of(width) {
            return s;
        }
        // The last step's columns past `cols` read zero.
        let col = k.coord(wide_lane, Axis::Col);
        let left = k.splat::<I32>(wide_lane, Sc::from(cols) - step * width);
        let inside = k.compare(col, left, CmpOp::Lt);
        let past = k.fill::<F32>(wide_lane, Const::Float(f64::INFINITY));
        k.where_(inside, s, past)
    };
    let carried = k.loop_vec(count.clone() + 1, init, |k, local, carried| {
        let step = step0.clone() + local.clone();
        let (list, queue, tau) = split(carried.clone());
        let none = k.fill::<F32>(wide_lane, Const::Float(f64::INFINITY));
        let [s] = k.select_if(
            local.clone().lt(count.clone()),
            |k| {
                if let Some((mins, bound)) = bounded {
                    // A row reads a step where its least is at most its bound,
                    // read by the warps of some row reading, `+∞` in the others.
                    let low = least(k, mins, step.clone());
                    let low = {
                        let zero = k.zeros::<F32>(wide_lane);
                        k.binary(low, zero, BinaryOp::Add)
                    };
                    let reads = k.compare(low, bound, CmpOp::Le);
                    return k.where_any(reads, |k| [scored(k, step.clone())], [none]);
                }
                [scored(k, step.clone())]
            },
            |_| [none],
        );
        // The step's scores a lane holds, one tile each, and their indices.
        let dealt: Vec<Regs<F32>> = (0..wide).map(|j| k.deal(s, LANES, j)).collect();
        let indices: Vec<Regs<F32>> = (0..wide)
            .map(|j| {
                let col = k.deal_col(s, LANES, j);
                let col = k.cast::<I32, F32>(col);
                let origin = k.splat::<F32>(lane, step.clone() * width + base.clone());
                k.binary(col, origin, BinaryOp::Add)
            })
            .collect();
        let none = k.fill::<F32>(lane, Const::Float(f64::INFINITY));
        let end = {
            let at = k.splat::<I32>(lane, local);
            let past = k.splat::<I32>(lane, count.clone());
            k.compare(at, past, CmpOp::Eq)
        };
        // Below the row's `k`-th, and filtered, at most its bound.
        let lower = |k: &mut Kernel, v: Regs<F32>, tau: Regs<F32>| {
            let lower = k.compare(v, tau, CmpOp::Lt);
            let Some((_, bound)) = bounded else { return lower };
            let within = k.compare(v, bound, CmpOp::Le);
            k.binary(lower, within, BinaryOp::And)
        };
        let least = dealt.iter().skip(1).fold(dealt[0], |a, &v| k.binary(a, v, BinaryOp::Min));
        let taking = lower(k, least, tau);
        let queued = k.compare(queue[0][0], none, CmpOp::Lt);
        let pending = k.binary(end, queued, BinaryOp::And);
        let busy = k.binary(taking, pending, BinaryOp::Or);
        k.if_any(
            busy,
            |k| {
                // A lane taking a score without room for the step's merges
                // the queues first; so does the step past the last.
                let crowded = k.compare(queue[depth - wide][0], none, CmpOp::Lt);
                let crowded = k.binary(crowded, taking, BinaryOp::And);
                let crowded = k.binary(crowded, pending, BinaryOp::Or);
                let kept: Vec<Regs<F32>> = list.iter().chain(&queue).flatten().copied().chain([tau]).collect();
                let carried = k.if_any(crowded, |k| merge(k, list.clone(), queue.clone()), kept);
                let (list, queue, tau) = split(carried);
                let mut slots: Vec<Regs<F32>> = queue.iter().map(|q| q[0]).chain(queue.iter().map(|q| q[1])).collect();
                for (&v, &index) in dealt.iter().zip(&indices) {
                    let taken = lower(k, v, tau);
                    let v = k.where_(taken, v, none);
                    slots = insert(k, &slots, v, index);
                }
                let queue = (0..depth).map(|j| [slots[j], slots[depth + j]]);
                list.into_iter().chain(queue).flatten().chain([tau]).collect()
            },
            carried,
        )
    });
    let (list, ..) = split(carried);
    for (t, [s, i]) in list.into_iter().take(held).enumerate() {
        // A share's lists stay in the units the merge compares.
        let [s] = match shares {
            1 => k.select_if(last.clone().eq(0), |_| [s], |k| [finished(k, s, metric)]),
            _ => [s],
        };
        let (sv, iv) = run(&mut k, (score_out, index_out), out_base.clone(), t);
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

/// `[score, index]` pairs of `[rows, w]` tiles holding a sequence a row,
/// its element `t·w + c` at column `c` of pair `t`.
type Seq = Vec<[Regs<F32>; 2]>;

/// `len` pairs of `shape` holding nothing: `+∞` at −1.
fn empty_seq(k: &mut Kernel, shape: Shape, len: usize) -> Seq {
    (0..len).map(|_| [f64::INFINITY, -1.0].map(|e| k.fill::<F32>(shape, Const::Float(e)))).collect()
}

/// A sequence's tiles in order.
fn flat(seq: &[[Regs<F32>; 2]]) -> impl Iterator<Item = Regs<F32>> + '_ {
    seq.iter().flatten().copied()
}

/// [`flat`]'s tiles as pairs again.
fn pairs(tiles: &[Regs<F32>]) -> Seq {
    tiles.chunks(2).map(|p| [p[0], p[1]]).collect()
}

/// The value of `sc` where it is arithmetic over constants.
fn constant(sc: &Sc) -> Option<i64> {
    match sc {
        Sc::Const(v) => Some(*v),
        Sc::Bin(op, a, b) => {
            let (a, b) = (constant(a)?, constant(b)?);
            match op {
                BinOp::Add => Some(a + b),
                BinOp::Sub => Some(a - b),
                BinOp::Mul => Some(a * b),
                _ => None,
            }
        }
        Sc::Id(_) => None,
    }
}

/// Element `p` of `seq`, its score and index as row vectors: a constant
/// `p`'s column of its pair, else the column of each pair `p` may be in.
fn element(k: &mut Kernel, seq: &[[Regs<F32>; 2]], p: Sc) -> [Regs<F32>; 2] {
    let shape = k.shape(seq[0][0]);
    let w = shape.cols;
    let col = k.coord(shape, Axis::Col);
    let none = k.fill::<F32>(shape, Const::Float(f64::NEG_INFINITY));
    let most = |k: &mut Kernel, pair: [Regs<F32>; 2], at: Regs<I32>| {
        let here = k.compare(col, at, CmpOp::Eq);
        pair.map(|v| k.where_(here, v, none))
    };
    let held = match constant(&p) {
        Some(p) => {
            let at = k.fill::<I32>(shape, Const::Int(p % w as i64));
            most(k, seq[p as usize / w], at)
        }
        None => {
            let mut held: Option<[Regs<F32>; 2]> = None;
            for (t, &pair) in seq.iter().enumerate() {
                let at = k.splat::<I32>(shape, p.clone() - t * w);
                let v = most(k, pair, at);
                held = Some(held.map_or(v, |h| [0, 1].map(|j| k.binary(h[j], v[j], BinaryOp::Max))));
            }
            held.expect("a pair")
        }
    };
    held.map(|v| k.reduce(v, Axis::Row, ReduceOp::Max))
}

/// `queue` with `v` pushed onto its front where `ins`, each pair there
/// moving one on, the last dropped (a full queue takes none).
fn pushed(k: &mut Kernel, queue: &[[Regs<F32>; 2]], ins: Regs<Bool>, v: [Regs<F32>; 2]) -> Seq {
    (0..queue.len())
        .map(|t| {
            let from = if t == 0 { v } else { queue[t - 1] };
            [0, 1].map(|j| k.where_(ins, from[j], queue[t][j]))
        })
        .collect()
}

/// `list` (ascending) with the least of `queue` (unsorted) merged in, as
/// many as it holds: the queue sorted, its least reversed against the
/// list's last slots and the earlier of each pair kept, then the list's
/// bitonic order.
fn queued_into(k: &mut Kernel, mut list: Seq, mut queue: Seq) -> Seq {
    let w = k.shape(list[0][0]).cols;
    let sizes = (1..=(queue.len() * w).ilog2()).map(|b| 1 << b);
    network(k, &mut queue, sizes);
    for (t, q) in queue.iter().take(list.len()).enumerate() {
        let at = list.len() - 1 - t;
        let rev = q.map(|v| k.flip(v, Axis::Col, w as u32 - 1));
        let first = before(k, list[at], rev);
        list[at] = [0, 1].map(|j| k.where_(first, list[at][j], rev[j]));
    }
    let len = list.len() * w;
    network(k, &mut list, [len]);
    list
}

/// The lower and upper halves of two ascending sequences of as many pairs
/// merged, each ascending: `b` reversed, every pair across the two ordered
/// (the lower half holding the least half), then each half's bitonic order.
fn merged(k: &mut Kernel, a: &[[Regs<F32>; 2]], b: &[[Regs<F32>; 2]]) -> (Seq, Seq) {
    let w = k.shape(a[0][0]).cols;
    let (mut lo, mut hi) = (vec![], vec![]);
    for (x, y) in a.iter().zip(b.iter().rev()) {
        let rev = y.map(|v| k.flip(v, Axis::Col, w as u32 - 1));
        let first = before(k, *x, rev);
        lo.push([0, 1].map(|j| k.where_(first, x[j], rev[j])));
        hi.push([0, 1].map(|j| k.where_(first, rev[j], x[j])));
    }
    let len = a.len() * w;
    network(k, &mut lo, [len]);
    network(k, &mut hi, [len]);
    (lo, hi)
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
    /// The scores a row's list holds (its bound where the list is live).
    pub k: usize,
    /// The rows and the list length a launch binds, where they vary: a
    /// share's slots past the live length are passed over.
    pub live: Live,
}

/// Rows a merge block folds, one warp's.
const MERGE_ROWS: usize = 256;
/// Rows a merge block of lists folds, a warp each.
pub const LIST_WARPS: u32 = 4;

impl MergeSpec {
    pub fn lowering(&self, device: &Device) -> Lowering {
        let rows = if self.k > 1 { LIST_WARPS } else { 1 };
        Lowering::uniform(device, false, WarpGrid { rows, cols: 1 })
    }
}

/// Parameters in order: `score_part` f32 and `index_part` i32, `splits`
/// shares [`share_stride`] apart as a split [`nearest`] leaves them, then
/// `score [n]` f32 and `index [n]` i32: each row's least over the shares, in
/// the metric's units, and the index of the first share scoring it (its
/// candidates come before the later shares'), so the smallest index; `+∞`
/// at −1 where no share has a candidate.
///
/// With `k > 1` the shares' lists, row `i` of share `s` at
/// `(s·`[`share_stride`]` + i)·k`, are merged into `score [n, k]` and
/// `index [n, k]` as [`nearest`] leaves them unsplit: ascending in `(score,
/// index)`, so an equal score at the smaller index, `+∞` at −1 past the
/// admissible candidates. A warp holds a row's running list in runs of
/// [`LANES`] slots, a register a lane each (as [`nearest_select`]), and
/// merges each share's list into it by the lower half of a bitonic merge:
/// the share's runs reversed against the list's, the earlier of each pair
/// kept, then the list's bitonic order.
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
    let (rows, runs) = (LIST_WARPS as usize, select_runs(top));
    let mut k = Kernel::new("nearest_merge");
    let stride = share_stride(n);
    let score_part = k.param::<F32>("score_part", ParamKind::In, splits * stride * top);
    let index_part = k.param::<I32>("index_part", ParamKind::In, splits * stride * top);
    let score = k.param::<F32>("score", ParamKind::Out, n * top);
    let index = k.param::<I32>("index", ParamKind::Out, n * top);
    let rows_live = live.rows(&mut k, n);
    let k_live = live.k(&mut k);
    // The rows and slots of the capacity, as `nearest`'s live views.
    let (rb, kb) = (bound(n, rows), bound(top, LANES));
    k.grid([launched(&rows_live, n, rows), Sc::from(1), Sc::from(1)]);
    k.warps(LIST_WARPS);
    let row0 = k.block(0) * rows;
    let lane = Shape::new(rows, LANES);
    // Run `t` of the block's lists in `p` from `base`.
    let run = |k: &mut Kernel, p: ParamRef<F32>, base: Sc, t: usize| {
        let v = k.view(p, base, [top, 1], lane, [rb.clone(), kb.clone()]);
        k.at(v, row0.clone(), t * LANES)
    };
    let index_run = |k: &mut Kernel, p: ParamRef<I32>, base: Sc, t: usize| {
        let v = k.view(p, base, [top, 1], lane, [rb.clone(), kb.clone()]);
        k.at(v, row0.clone(), t * LANES)
    };
    // Run `t` of share `s`'s lists, its slots past the list (the live
    // length) empty; `None` where none is.
    let loaded = |k: &mut Kernel, s: Sc, t: usize| {
        if t * LANES >= top {
            return None;
        }
        let base = s * (stride * top);
        let v = run(k, score_part, base.clone(), t);
        let v = k.load(v);
        let i = index_run(k, index_part, base, t);
        let i = k.load(i);
        let mut pair = [v, k.cast::<I32, F32>(i)];
        if (t + 1) * LANES > top || k_live.is_some() {
            let col = k.coord(lane, Axis::Col);
            let width = match &k_live {
                None => k.fill::<I32>(lane, Const::Int((top - t * LANES) as i64)),
                Some(live) => k.splat::<I32>(lane, live.clone() - t * LANES),
            };
            let inside = k.compare(col, width, CmpOp::Lt);
            let empty = empty_seq(k, lane, 1)[0];
            pair = [0, 1].map(|j| k.where_(inside, pair[j], empty[j]));
        }
        Some(pair)
    };
    // The first share's lists as they are, every later one merged in.
    let first: Seq =
        (0..runs).map(|t| loaded(&mut k, Sc::from(0), t).unwrap_or_else(|| empty_seq(&mut k, lane, 1)[0])).collect();
    let list = k.loop_vec(splits - 1, flat(&first).collect(), |k, s, list| {
        let list = pairs(&list);
        let mut lo: Seq = (0..runs)
            .map(|t| {
                let Some(other) = loaded(k, s.clone() + 1, runs - 1 - t) else { return list[t] };
                let rev = other.map(|v| k.flip(v, Axis::Col, LANES as u32 - 1));
                let first = before(k, list[t], rev);
                [0, 1].map(|j| k.where_(first, list[t][j], rev[j]))
            })
            .collect();
        network(k, &mut lo, [runs * LANES]);
        flat(&lo).collect()
    });
    for (t, [s, i]) in pairs(&list).into_iter().take(top.div_ceil(LANES)).enumerate() {
        let s = finished(&mut k, s, metric);
        let v = run(&mut k, score, Sc::from(0), t);
        k.store(v, s);
        let i = k.cast::<F32, I32>(i);
        let v = index_run(&mut k, index, Sc::from(0), t);
        k.store(v, i);
    }
    k.finish()
}
