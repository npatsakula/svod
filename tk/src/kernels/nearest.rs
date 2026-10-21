//! Each query row's cheapest candidate: one `x·cᵀ` product per candidate
//! tile on the matrix core, its scores folded into a carried running minimum
//! and its candidate index rather than a softmax. The query rows stay
//! resident in registers while candidate tiles stream through a shared ring,
//! or, at a depth the registers cannot hold, both stream along the depth
//! through the GEMM's K pipeline once per candidate tile. The candidates may
//! split over several blocks per query block, each leaving a partial least
//! and index that [`nearest_merge`] folds.

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
        s.finish()
    }
}

impl NearestCfg {
    pub const fn new(bq: usize, bkv: usize, stages: usize) -> Self {
        Self { bq, bkv, stages, unroll: false, warp_rows: WARP_ROWS, bk: None, splits: 1 }
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
/// The L2 family's scores are compared halved (`c_sq/2 − x·c`, plus `x_sq/2`,
/// against halved floors; [`Metric::L2`] against its floors clamped at zero
/// and squared, its square root taken once per row), which orders them as
/// the metric's own.
pub fn nearest<T: Elem>(spec: &NearestSpec) -> Program {
    let NearestSpec { n, m, d, metric, floors, groups, k: top, cfg } = *spec;
    let NearestCfg { bq, bkv, stages, splits, .. } = cfg;
    let tiles = m.div_ceil(bkv);
    assert!(m <= MAX_CANDIDATES, "{m} candidates pass the f32 index's {MAX_CANDIDATES}");
    assert!(top >= 1, "a row keeps a score at least");
    assert!((1..=tiles).contains(&splits), "{splits} splits of {tiles} candidate tiles");
    // A fill zeroes the 16-byte chunks past an overhanging chunk's depth.
    assert!(cfg.bk.is_none() || d.is_multiple_of(8), "a streamed depth {d} of whole 16-byte chunks");
    let mut k = Kernel::new("nearest");
    let x = k.param::<T>("x", ParamKind::In, n * d);
    let c = k.param::<T>("c", ParamKind::In, m * d);
    let c_sq = metric.col_norms().then(|| k.param::<F32>("c_sq", ParamKind::In, m));
    let x_sq = metric.row_norms().then(|| k.param::<F32>("x_sq", ParamKind::In, n));
    let row_floor = floors.row.then(|| k.param::<F32>("row_floor", ParamKind::In, n));
    let col_floor = floors.col.then(|| k.param::<F32>("col_floor", ParamKind::In, m));
    let grouped =
        groups.then(|| (k.param::<I32>("row_group", ParamKind::In, n), k.param::<I32>("col_group", ParamKind::In, m)));
    let split = splits > 1;
    let (score, index) = if split { ("score_part", "index_part") } else { ("score", "index") };
    let parts = top * if split { splits * share_stride(n) } else { n };
    let score_out = k.param::<F32>(score, ParamKind::Out, parts);
    let index_out = k.param::<I32>(index, ParamKind::Out, parts);
    let blocks = n.div_ceil(bq);
    k.grid([Sc::from(blocks), Sc::from(splits), Sc::from(1)]);
    k.warps(cfg.warps());
    let c_s = cfg.bk.is_none().then(|| k.smem::<T>("c_s", stages * bkv * d));

    // The query block, and split, the first candidate tile of its share and
    // how many it takes.
    let (q_block, share) = match split {
        true => {
            let (q_block, part) = tile_order(&mut k, blocks, splits, SPLIT_GROUP);
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
    // several, its rows `step` apart), and a trip's columns of an `[m]` one.
    fn row_view<E: Elem>(
        k: &mut Kernel,
        p: ParamRef<E>,
        (base, step): (Sc, usize),
        n: usize,
        bq: usize,
        q_off: Sc,
    ) -> Gmem<E> {
        let v = k.view(p, base, [step, 1], Shape::new(bq, 1), [bound(n, bq), None]);
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
    let x_view = k.view(x, 0, [d, 1], Shape::new(bq, depth), [bound(n, bq), bound(d, depth)]);
    let x_view = k.at(x_view, q_off.clone(), 0);
    let q_r = cfg.bk.is_none().then(|| k.load(x_view));
    let x_sq = x_sq.map(|p| {
        let v = row_view(&mut k, p, (Sc::from(0), 1), n, bq, q_off.clone());
        let x_sq = k.load(v);
        half(&mut k, x_sq)
    });
    let row_floor = row_floor.map(|p| {
        let v = row_view(&mut k, p, (Sc::from(0), 1), n, bq, q_off.clone());
        let f = k.load(v);
        floor(&mut k, f)
    });
    let row_group = grouped.map(|(p, _)| {
        let v = row_view(&mut k, p, (Sc::from(0), 1), n, bq, q_off.clone());
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

    // The tile's scores offered to the row's carried list (`[bq, top]` of
    // its least scores ascending, their indices, and its last score: the
    // vote's, a row vector as the row minimum is), its least first,
    // then each next least once the least before it is masked out: a trip
    // offers at most `top`, and only while some row of the warp still has a
    // score below one in its list. A row whose offer misses its list masks
    // its least all the same: its scores only rise, so it misses again.
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
                let out = k.loop_vec(top.min(bkv), init, |k, _, carried| {
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
                            let [best, at] = offer(k, [best, at], low, index);
                            let last = k.reduce(best, Axis::Row, ReduceOp::Max);
                            vec![rest, next, best, at, last]
                        },
                        carried.clone(),
                    )
                });
                out[2..].to_vec()
            },
            list.clone(),
        )
    };

    let list = Shape::new(bq, top);
    let mut init = vec![k.fill::<F32>(list, Const::Float(f64::INFINITY)), k.fill::<F32>(list, Const::Float(-1.0))];
    if top > 1 {
        init.push(k.fill::<F32>(rows, Const::Float(f64::INFINITY)));
    }
    let folded = |k: &mut Kernel, dot: Regs<F32>, start: Sc, carried: Vec<Regs<F32>>| match top {
        1 => fold(k, dot, start, carried[0], carried[1]).to_vec(),
        _ => fold_list(k, dot, start, carried),
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
        n: usize,
        q_off: Sc,
    ) -> Gmem<E> {
        let v = k.view(p, base, step, list, [bound(n, list.rows), None]);
        k.at(v, q_off, 0)
    }
    let v = slots(&mut k, score_out, at.clone(), list, n, q_off.clone());
    k.store(v, best);
    let index = k.cast::<F32, I32>(index);
    let v = slots(&mut k, index_out, at, list, n, q_off.clone());
    k.store(v, index);
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MergeSpec {
    pub n: usize,
    pub splits: usize,
    pub metric: Metric,
    /// The scores a row's list holds.
    pub k: usize,
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
    let MergeSpec { n, splits, metric, .. } = *spec;
    let mut k = Kernel::new("nearest_merge");
    let stride = share_stride(n);
    let score_part = k.param::<F32>("score_part", ParamKind::In, splits * stride);
    let index_part = k.param::<I32>("index_part", ParamKind::In, splits * stride);
    let score = k.param::<F32>("score", ParamKind::Out, n);
    let index = k.param::<I32>("index", ParamKind::Out, n);
    k.grid([Sc::from(n.div_ceil(MERGE_ROWS)), Sc::from(1), Sc::from(1)]);
    k.warps(1);
    let row0 = k.block(0) * MERGE_ROWS;
    let shape = Shape::new(1, MERGE_ROWS);
    // The block's rows of the `[n]` parameter at `base`.
    fn rows<E: Elem>(k: &mut Kernel, p: ParamRef<E>, n: usize, row0: Sc, base: Sc) -> Gmem<E> {
        let v = k.view(p, base, [0, 1], Shape::new(1, MERGE_ROWS), [None, bound(n, MERGE_ROWS)]);
        k.at(v, 0, row0)
    }
    let best0 = k.fill::<F32>(shape, Const::Float(f64::INFINITY));
    let index0 = k.fill::<F32>(shape, Const::Float(-1.0));
    let [best, at] = k.loop_(splits, [best0, index0], |k, part, [best, at]| {
        let v = rows(k, score_part, n, row0.clone(), part.clone() * stride);
        let s = k.load(v);
        let v = rows(k, index_part, n, row0.clone(), part * stride);
        let i = k.load(v);
        // Exact in f32 below `MAX_CANDIDATES`, as the kernel's index.
        let i = k.cast::<I32, F32>(i);
        let lower = k.compare(s, best, CmpOp::Lt);
        [k.where_(lower, s, best), k.where_(lower, i, at)]
    });
    let best = finished(&mut k, best, metric);
    let v = rows(&mut k, score, n, row0.clone(), Sc::from(0));
    k.store(v, best);
    let at = k.cast::<F32, I32>(at);
    let v = rows(&mut k, index, n, row0, Sc::from(0));
    k.store(v, at);
    k.finish()
}

/// [`nearest_merge`] of `k > 1` scores a row.
fn merge_lists(spec: &MergeSpec) -> Program {
    let MergeSpec { n, splits, metric, k: top } = *spec;
    let mut k = Kernel::new("nearest_merge");
    let stride = share_stride(n);
    let score_part = k.param::<F32>("score_part", ParamKind::In, splits * top * stride);
    let index_part = k.param::<I32>("index_part", ParamKind::In, splits * top * stride);
    let score = k.param::<F32>("score", ParamKind::Out, n * top);
    let index = k.param::<I32>("index", ParamKind::Out, n * top);
    k.grid([Sc::from(n.div_ceil(LIST_ROWS)), Sc::from(1), Sc::from(1)]);
    k.warps(1);
    let row0 = k.block(0) * LIST_ROWS;
    let shape = Shape::new(1, LIST_ROWS);
    // The block's rows of a parameter, row `r` at `base + r·step`.
    fn rows<E: Elem>(k: &mut Kernel, p: ParamRef<E>, n: usize, row0: Sc, (base, step): (Sc, usize)) -> Gmem<E> {
        let v = k.view(p, base, [0, step], Shape::new(1, LIST_ROWS), [None, bound(n, LIST_ROWS)]);
        k.at(v, 0, row0)
    }
    let scores = (0..top).map(|_| k.fill::<F32>(shape, Const::Float(f64::INFINITY))).collect::<Vec<_>>();
    let init = scores.into_iter().chain((0..top).map(|_| k.fill::<F32>(shape, Const::Float(-1.0)))).collect();
    let list = k.loop_vec(splits, init, |k, part, mut list| {
        for j in 0..top {
            let at = (part.clone() * top + j) * stride;
            let v = rows(k, score_part, n, row0.clone(), (at.clone(), 1));
            let s = k.load(v);
            let v = rows(k, index_part, n, row0.clone(), (at, 1));
            let i = k.load(v);
            let i = k.cast::<I32, F32>(i);
            list = insert(k, &list, s, i);
        }
        list
    });
    let (list, indices) = list.split_at(top);
    for (j, (&best, &at)) in list.iter().zip(indices).enumerate() {
        let best = finished(&mut k, best, metric);
        let v = rows(&mut k, score, n, row0.clone(), (Sc::from(j), top));
        k.store(v, best);
        let at = k.cast::<F32, I32>(at);
        let v = rows(&mut k, index, n, row0.clone(), (Sc::from(j), top));
        k.store(v, at);
    }
    k.finish()
}
