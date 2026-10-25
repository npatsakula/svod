//! Nearest-candidate configs: the query block, candidate block, depth and
//! list lattice, the model constants its pair epilogue and list machinery
//! price with, and what a problem costs on a planner, split and whole.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use svod_dtype::ScalarDType;

use crate::atoms::{Device, Fill, Target};
use crate::ir::{Axis, Orient, Shape};
use crate::kernels::attention::WARP_ROWS;
use crate::kernels::gemm::GROUP_ROWS;
use crate::kernels::nearest::{Floors, LANES, LaneQueue, Metric, NearestCfg, SelectCfg, select_runs};
use crate::layouts::{TileLayout, WarpGrid, chunked, fills_in_chunks};

use super::attention::{FA_STAGES, QUERY_BLOCKS};
use super::cost::{Planner, Run, STEP, cached};
use super::gemm::DEPTHS;
use super::search::{Rejected, Tiled, fired, sorted};

impl NearestCfg {
    /// The depth a trip's product spans: the whole `d` resident, else a chunk.
    fn depth(&self, d: usize) -> usize {
        self.bk.unwrap_or(d)
    }

    /// The rows of each operand tile a trip fills: the candidates, and
    /// streamed the query rows too.
    fn filled(&self) -> impl Iterator<Item = usize> {
        [Some(self.bkv), self.bk.map(|_| self.bq)].into_iter().flatten()
    }

    /// Whether the config lowers on `target` at depth `d` over operands of
    /// `dtype`: the atom's M is [`WARP_ROWS`], a warp owns whole atom rows of
    /// queries and the block whole warps, the product tiles the atoms, and
    /// each operand fills in whole 16-byte chunks per thread
    /// ([`fills_in_chunks`]); a streamed depth is whole 16-byte chunks too,
    /// which an overhanging chunk's fill zeroes past it.
    pub(crate) fn fits(&self, target: &Target, d: usize, dtype: ScalarDType) -> bool {
        let Some(atom) = target.geometry_of(dtype) else { return false };
        let (bq, bkv, rows, depth) = (self.bq, self.bkv, self.warp_rows, self.depth(d));
        let threads = self.warps() as usize * target.wave as usize;
        let chunk = 16 / dtype.bytes();
        let fills = self.filled().all(|rows| fills_in_chunks(Shape::new(rows, depth), dtype.bytes(), threads));
        atom.m as usize == WARP_ROWS
            && rows.is_multiple_of(WARP_ROWS)
            && rows > 0
            && bq.is_multiple_of(rows)
            && bq > 0
            && bkv.is_multiple_of(atom.n as usize)
            && depth.is_multiple_of(atom.k as usize)
            && d.is_multiple_of(if self.bk.is_some() { chunk } else { atom.k as usize })
            && fills
            && target.fill.runs(self.stages)
    }

    /// The registers a lane holds at depth `d`, from the layouts the lowering
    /// assigns: the query fragments, live across the candidate loop (a
    /// chunk's, streamed); a trip's candidate fragments and its f32 scores
    /// (streamed, its accumulators) beside the tile the argmin selects the
    /// columns in; the staged chunks where the fill goes through registers.
    /// Addressing and the row vectors are not counted, but for the `k`
    /// least scores and indices a row's list holds past one.
    pub(crate) fn registers(&self, target: &Target, d: usize, k: usize, dtype: ScalarDType) -> Option<u32> {
        let [operands, scores] = self.lane_parts(target, d, dtype)?;
        Some(operands + 2 * scores + self.list_registers(target, d, k, dtype)?)
    }

    /// The registers [`NearestProblem::admit`] holds against the lane's budget:
    /// [`Self::registers`], or streamed, the depth pipeline's alone (its
    /// fragments, accumulators and staged chunks). A streamed tile folds
    /// once its pipeline has drained, its fragments dead, and whether the
    /// columns it selects fit beside the scores is the compiler's call,
    /// which the tuner's admission checks: on sm_86 32 rows by 128
    /// candidates hold 236 registers, unspilled, where the estimate says 336.
    /// A list of `k > 1` scores is live across the pipeline, and the
    /// scores beside it across the deals, so a list counts it all either way
    /// (gfx1151, 32 rows by 64 candidates streamed 32 deep, a list of 64:
    /// estimated 252 so, 284 whole, 256 and spilled compiled).
    pub(crate) fn held(&self, target: &Target, d: usize, k: usize, dtype: ScalarDType) -> Option<u32> {
        let [operands, scores] = self.lane_parts(target, d, dtype)?;
        let listed = k > 1 && self.select.is_none();
        let scores = if self.bk.is_some() && !listed { scores } else { 2 * scores };
        Some(operands + scores + self.list_registers(target, d, k, dtype)?)
    }

    /// A lane's registers of a list of `k > 1` scores and indices a row
    /// ([`LaneQueue`]): its pairs of `[bq, w]` tiles
    /// [spread](TileLayout::spread_row) from the products' row minimum
    /// under the issue a row spreads over fewer lanes ([`Self::scores_layout`]),
    /// the lanes' queue's pairs, and the row vectors the vote and the passes
    /// carry; none at `k = 1`, whose least and index the scores' count
    /// covers; a merge's network holds half as many tiles again beside them.
    /// [Queued](NearestCfg::queue), the queue's pairs and the
    /// [`FLUSH_TILES`] its merge into the outputs holds beside them. A
    /// register-staged ring unrolled over its slots holds a list's pairs and
    /// the staged chunks once more (gfx1151, 16 rows by 32 candidates, a list
    /// of 64 at depth 48: 226 registers rolled, 255 and spilled unrolled).
    fn list_registers(&self, target: &Target, d: usize, k: usize, dtype: ScalarDType) -> Option<u32> {
        if k == 1 || self.select.is_some() {
            return Some(0);
        }
        let LaneQueue { width, depth, passes } = self.lane_queue();
        let row = self.scores_layout(target, d, dtype)?.row_vector();
        let tile = TileLayout::spread_row(&row, width as u32).regs();
        let pairs = (self.list_slots(k) / width) as u32;
        let flush = if self.queue.is_some() { FLUSH_TILES * pairs } else { 0 };
        // Rounds hold the trip's deals not yet offered across them.
        let rest = if passes { 0 } else { (self.bkv / width) as u32 };
        let tiles = 2 * pairs + 2 * depth as u32 + flush + pairs + rest;
        let unrolled = match target.fill {
            Fill::Staged if self.unroll => 2 * pairs * tile + self.lane_parts(target, d, dtype)?[0] / 2,
            _ => 0,
        };
        Some(tiles * tile + 4 * row.regs() + unrolled)
    }

    /// The layout of the products' scores under the issue spreading a row
    /// over the fewest lanes ([`crate::atoms::MmaAtom::narrow`]), the one
    /// the lowering issues a product whose scores are dealt (RDNA swapped:
    /// a row a lane of 16, its 16 others holding the row's other half;
    /// CUDA's `m16n8k16`: two rows a lane, a quad's lanes each holding a
    /// quarter).
    fn scores_layout(&self, target: &Target, d: usize, dtype: ScalarDType) -> Option<TileLayout> {
        let grid = WarpGrid { rows: self.warps(), cols: 1 };
        Some(target.geometry_of(dtype)?.narrow(grid, self.bq, self.bkv, self.depth(d)).ok()?.1.c)
    }

    /// The lanes the products spread a row of the scores over
    /// ([`Self::scores_layout`]): the narrowest lane queues a deal renames
    /// registers into.
    pub(crate) fn sharing(&self, target: &Target, d: usize, dtype: ScalarDType) -> Option<usize> {
        Some(1 << self.scores_layout(target, d, dtype)?.row_fold_masks().len())
    }

    /// A lane's 32-bit operand registers (the fragments and the staged
    /// chunks, `dtype`'s elements packed) and its f32 scores', as
    /// [`Self::registers`] counts them.
    fn lane_parts(&self, target: &Target, d: usize, dtype: ScalarDType) -> Option<[u32; 2]> {
        let (warps, depth) = (self.warps(), self.depth(d));
        let grid = WarpGrid { rows: warps, cols: 1 };
        let qk = target.geometry_of(dtype)?.issue(Orient::Direct, grid, self.bq, self.bkv, depth).ok()?;
        let bytes = dtype.bytes();
        let half = |regs: u32| regs.div_ceil(4 / bytes as u32);
        let staged = |rows| chunked(Shape::new(rows, depth), bytes, warps, target.wave, Axis::Col).map(|l| l.regs());
        let fill = match target.fill {
            Fill::CpAsync => 0,
            Fill::Staged => half(self.filled().map(staged).sum::<Option<u32>>()?),
        };
        Some([half(qk.a.regs()) + half(qk.b.regs()) + fill, qk.c.regs()])
    }
}

/// Model: a pair's score, its row minimum and the argmin's compare, select
/// and second minimum, in units of [`Costs::score`] (a softmax score's scale,
/// row maximum, subtract, `exp2` and row sum): as many vector operations,
/// two of them reductions, where the softmax's `exp2` stands for the
/// argmin's select. The argmin's three run only in the warps a trip lowers
/// some row of, ever fewer as the trips pass; the model charges them all.
pub(crate) const NEAREST_PAIR: f64 = 1.0;
/// Model: one more elementwise operation a pair, in the same units: the
/// row's norm added, a floor's maximum, the L2 clamp.
pub(crate) const PAIR_OP: f64 = 0.2;
/// Model: what groups add to a pair: a compare and a select.
pub(crate) const GROUPS_PAIR: f64 = 2.0 * PAIR_OP;
/// Model: a deal's least an element ([`crate::kernels::nearest::nearest`]
/// with `k > 1`), in units of [`Costs::score`]: the minimum a lane's scores
/// enter a trip's rounds with.
const DEAL_ELEM: f64 = 1.0 * PAIR_OP;
/// Model: a vote a lane, the lane's or of its elements and the branch.
const DEAL_LANE: f64 = 2.0 * PAIR_OP;
/// Model: a pass's vote an element ([`LaneQueue::passes`]): its compare
/// against the row's `k`-th and the pass's test. Fitted with [`DEAL_LANE`],
/// [`HIT_ELEM`], [`PUSH_ELEM`] and [`LIST_SLOT`] to 18 list configs on sm_86
/// (16384² f16 at depth 128, `k` 16–64, queues 2–8 deep, 16 and 32
/// candidates a trip) against the pushes and merges a simulation of the
/// kernel counts.
const PASS_ELEM: f64 = 2.0 * PAIR_OP;
/// Model: a pushed deal's work an element but for its queue: its index and
/// order against the `k`-th, the full queue's test.
const HIT_ELEM: f64 = 1.3 * PAIR_OP;
/// Search bound: the fewest deals a trip a list is also ranked passing
/// over ([`LaneQueue::passes`]): rounds hold every deal of a trip.
const PASS_DEALS: usize = 8;
/// Model: a round's work a dealt element: the column of its lane's least
/// (a compare, a select and a minimum), its removal (a compare and a
/// select) and the next least (a minimum), at half their count: against
/// 54 list configs timed on sm_86 (16384² f16 at depth 128, 64 scores) the
/// ranking's rank correlation was 0.42 so, 0.37 at the full count.
const ROUND_ELEM: f64 = 3.0 * PAIR_OP;
/// Model: a push's work an element a queue slot: its selects.
const PUSH_ELEM: f64 = 1.3 * PAIR_OP;
/// Model: a merge's work a row's slot a stage, its lanes sharing it.
const LIST_SLOT: f64 = PAIR_OP;
/// Search bound: the scores a lane queues of a row before the warp merges
/// the queues into the lists.
const LANE_DEPTHS: [usize; 2] = [2, 4];
/// Search bound: the deals a trip's scores are dealt in: each is a branch
/// of the trip's code, twice over (the first pass and the later ones); 64
/// of them (gfx1151, 128 candidates in pairs of lanes) built a program
/// past the graph rewrite's depth.
const MOST_DEALS: usize = 16;
/// Search bound: the most scores a row's register list keeps; a longer list
/// is held in the outputs, a register queue merged into it
/// ([`NearestCfg::queue`]), or listed from a score matrix
/// ([`NearestCfg::select`]). At 16384² rows on sm_86 a register list of 48
/// ran ahead of both (depths 128 and 768), one of 64 at depth 768.
pub(crate) const REGISTER_K: usize = 64;
/// Search bound: the shortest list ranked queued and through a score matrix
/// as well.
const LONG_FROM: usize = 33;
/// Search bound: the queues of a list held in the outputs.
const QUEUES: [usize; 2] = [16, 32];
/// Model: the tiles a merge into the outputs holds a pair of the queue
/// beside it: a run of the list and the carry, each scores and indices.
const FLUSH_TILES: u32 = 4;
/// Model: a merge step's work a slot, in units of [`Costs::score`]: the
/// partner's score and index, the order (three compares, two logic
/// operations), the side's test and two selects.
const MERGE_SLOT: f64 = 8.0 * PAIR_OP;
/// Model: the share of a queue a warp's rows hold on average when its
/// first row fills it, and the warp merges them all.
const QUEUE_FILLED: f64 = 0.6;
/// Model: a score-matrix pair's work, in units of [`Costs::score`]: its
/// score, without the row's least, and its store's issue.
const MATRIX_PAIR: f64 = 3.0 * PAIR_OP;
/// Model: a listed score's scan in [`crate::kernels::nearest::nearest_select`]:
/// its load, its compare against the row's `k`-th and the vote.
const SCAN_SCORE: f64 = 3.0 * PAIR_OP;
/// Model: a filtered listing's step passed over, in units of
/// [`Costs::score`]: its trips' least loaded, compared and voted on.
const SKIP_STEP: f64 = 4.0 * PAIR_OP;
/// Model: a queue insert's work a slot a lane holds: a compare and four
/// selects.
const INSERT_SLOT: f64 = 5.0 * PAIR_OP;
/// Model: the share of the queues a row's lanes hold on average when the
/// first fills its own.
const SELECT_FILLED: f64 = 0.5;
/// Search bound: the scores a lane of [`crate::kernels::nearest::nearest_select`]
/// queues.
const SELECT_QUEUES: [usize; 4] = [2, 4, 8, 16];
/// Model: the bytes a store moves at least, a cache line.
const LINE: usize = 128;
/// Model: the bytes a scattered store moves, a sector.
const SECTOR: usize = 32;
/// Search bound: the product tiles a score matrix's listing is ranked with:
/// at 16384² and depth 768, 1024 scores, gfx1151 spilled the three cheapest
/// (128 and 256 rows), and with no matrix admitted the tuner took a queued
/// list at 0.8 Gpairs/s; the sixth is 64 rows by 128, which ran at 5.9.
const MATRIX_TILES: usize = 6;
/// Search bound: the rows a select block lists.
const SELECT_WARPS: u32 = 4;
/// Search bound: the trips a filtered listing takes per score of a row's
/// list at least, so the scores it reads (about `k` trips' worth) are at
/// most a quarter of a row's.
const FILTER_GROUPS: usize = 4;

/// Search bound: the longest list a streamed split shares as a least's
/// rereads do ([`Planner::nearest_split`]): past it every share fills a long
/// list again (at 4096 queries by 2^20 candidates at depth 768 on gfx1201,
/// 16 shares ran 14 % behind 2 for 64 scores and 24 % for 256).
const REREAD_K: usize = 32;
/// Search bound: the fewest candidate tiles a share of a split nearest
/// block walks ([`Planner::nearest_split`]).
const SPLIT_TILES: usize = 4;

/// The nearest epilogue's cost clusters, as its constants naturally divide,
/// each scaled by a weight the tests and the rank harness perturb (zeroed,
/// to attribute the model's rank error to a cluster) and nothing else: the
/// default of one everywhere is today's arithmetic exactly.
///
/// * `pair` — every pair's score epilogue: [`NEAREST_PAIR`], the further
///   elementwise operations ([`PAIR_OP`], the floors, norms and the L2
///   clamp) and what groups add ([`GROUPS_PAIR`]). It is its own cluster
///   because [`PAIR_OP`] is the unit the listing constants below are
///   multiples of: folding it here would zero the whole epilogue with it.
/// * `deal` — the listing's per-trip work: a deal's least ([`DEAL_ELEM`]),
///   its votes ([`DEAL_LANE`], [`PASS_ELEM`]), a pushed deal's test
///   ([`HIT_ELEM`]), a round's dealt column ([`ROUND_ELEM`]) and a queue
///   slot's pushes ([`PUSH_ELEM`]).
/// * `merge` — the bitonic sort and merge rates: a list's slot a stage
///   ([`LIST_SLOT`], [`MERGE_SLOT`]) wherever a merge runs (a warp's queues,
///   a queue's flush into the outputs, a split's merge kernel, a scan's).
/// * `scan` — the score-matrix path: a stored pair ([`MATRIX_PAIR`]), the
///   listing kernel's scan of a score ([`SCAN_SCORE`]), a skipped step
///   ([`SKIP_STEP`]) and a queue insert ([`INSERT_SLOT`]).
///
/// The fill fractions ([`QUEUE_FILLED`], [`SELECT_FILLED`]) and the tile
/// count [`FLUSH_TILES`] are counts and data math, not rates: they stay.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Weights {
    pub(crate) pair: f64,
    pub(crate) deal: f64,
    pub(crate) merge: f64,
    pub(crate) scan: f64,
}

impl Weights {
    /// Today's constants: every cluster at its full weight.
    pub(crate) fn same() -> Self {
        Self { pair: 1.0, deal: 1.0, merge: 1.0, scan: 1.0 }
    }
}

/// The largest power of two not past `n` (zero for zero).
fn prev_power_of_two(n: usize) -> usize {
    if n == 0 { 0 } else { 1 << n.ilog2() }
}

/// Search bound: a nearest block's candidates a trip.
const CANDIDATE_BLOCKS: [usize; 4] = [16, 32, 64, 128];
/// Search bound: the query rows a warp owns.
const NEAREST_WARP_ROWS: [usize; 2] = [WARP_ROWS, 2 * WARP_ROWS];
/// Search bound: a register-staged nearest ring's slots, the GEMM's: one
/// slot measured faster than two on gfx1151 (64 rows by 64 candidates,
/// streamed at depth 768: +7 %).
const NEAREST_STAGED: [usize; 2] = [2, 1];
/// Search bound: a nearest block's warps at most.
const NEAREST_WARPS: usize = 8;
/// Search bound: the depth chunks a streamed nearest block's trip spans, the
/// GEMM's, and none: the rows resident.
const NEAREST_DEPTHS: [Option<usize>; 4] = [None, Some(DEPTHS[0]), Some(DEPTHS[1]), Some(DEPTHS[2])];

/// A nearest-candidate problem: `n` query rows against `m` candidates at
/// depth `d`, their elements of `dtype` (16-bit floats, or 8-bit integers
/// scaled per row), scored by `metric` above `floors`, under groups or not,
/// each row keeping its `k` least scores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NearestProblem {
    pub n: usize,
    pub m: usize,
    pub d: usize,
    pub dtype: ScalarDType,
    pub metric: Metric,
    pub floors: Floors,
    pub groups: bool,
    pub k: usize,
}

impl NearestProblem {
    /// What `cfg` costs here, as [`GemmProblem::admit`]: a candidate trip pays
    /// every wave's read of the candidate tile, its fill (every block reads
    /// every candidate: the blocks in flight share the lines), the product's
    /// steps and every pair's epilogue ([`NEAREST_PAIR`], plus a
    /// [`PAIR_OP`] for each further operation and [`GROUPS_PAIR`] under
    /// groups), which at a
    /// small depth outweighs the product. A wave's independent accumulators
    /// are its atom rows times the trip's atom columns. Left out past the
    /// lane's register budget: the queries stay resident.
    ///
    /// Streamed, a candidate tile takes `⌈d / bk⌉` trips of the GEMM's K
    /// pipeline, each reading and filling both operands' chunks: the block's
    /// own query rows again for every candidate tile (the blocks in flight
    /// share the candidate chunk only), its pairs' epilogue shared out over
    /// them, and the ring, which prefetches within a tile only, starting
    /// empty at every tile: a memory latency a round of blocks waits
    /// unhidden ([`Costs::latency`]). Its step is at least two matrix-core
    /// depths where the depth allows one, as the GEMM's.
    ///
    /// A list of `k > 1` scores a row adds its deals, pushes and merges
    /// ([`Self::listing`]), its lane queues at least as wide as the lanes a
    /// row spreads over ([`NearestCfg::sharing`]) and, queued, at most half
    /// the register queue; past [`REGISTER_K`] a register list is left out.
    /// Through a score matrix, a pair costs [`MATRIX_PAIR`] and its store,
    /// and the listing kernel's [`Self::select_cost`] is added.
    pub fn admit(&self, planner: &Planner, cfg: &NearestCfg) -> Result<(f64, u32), Rejected> {
        let Device { target, costs } = &planner.device;
        let (bq, bkv, d, k) = (cfg.bq, cfg.bkv, self.d, self.k);
        if cfg.queue.is_none() && cfg.select.is_none() && k > REGISTER_K {
            return Err(Rejected::TopK { k, most: REGISTER_K });
        }
        let dtype = self.dtype;
        if cfg.lists(k).is_err() || !cfg.fits(target, d, dtype) {
            return Err(Rejected::Lowers);
        }
        let atom = target.geometry_of(dtype).ok_or(Rejected::Lowers)?;
        let elem = dtype.bytes();
        if cfg.bk.is_some_and(|bk| bk < 2 * atom.k as usize && d.is_multiple_of(2 * atom.k as usize)) {
            return Err(Rejected::Shallow);
        }
        if cfg.bk.is_some_and(|bk| bk > d) {
            return Err(Rejected::Divides);
        }
        let smem = cfg.smem_bytes(d, dtype);
        if smem > target.smem_bytes {
            return Err(Rejected::Smem { need: smem, have: target.smem_bytes });
        }
        let regs = cfg.registers(target, d, k, dtype).ok_or(Rejected::Layout)?;
        let waves = cfg.warps() as usize;
        let (held, budget) = (cfg.held(target, d, k, dtype).ok_or(Rejected::Layout)?, target.lane_budget(waves as u32));
        if held > budget {
            return Err(Rejected::Registers { need: held, have: budget });
        }
        let atoms = (cfg.warp_rows / atom.m as usize) * (bkv / atom.n as usize);
        let Floors { row, col } = self.floors;
        let clamp = self.metric == Metric::L2 && !(row || col);
        // 8-bit products are widened and scaled by both rows' scales.
        let scaled = if elem == 1 { 3 } else { 0 };
        let ops = [self.metric.row_norms(), row, col, clamp].into_iter().filter(|&on| on).count() + scaled;
        let w = &planner.weights;
        let first = match cfg.select.is_some() {
            true => MATRIX_PAIR * w.scan,
            false => NEAREST_PAIR * w.pair,
        };
        let pair = first + ops as f64 * PAIR_OP * w.pair + if self.groups { GROUPS_PAIR * w.pair } else { 0.0 };
        // Split, every query block's candidate tiles in shares, a block each.
        let splits = cfg.splits;
        let (blocks, tiles, stages) = (self.n.div_ceil(bq) * splits, self.m.div_ceil(bkv).div_ceil(splits), cfg.stages);
        let share = NearestProblem { m: self.m.min(tiles * bkv), ..*self };
        let depth = cfg.bk.unwrap_or(d);
        let chunks = d.div_ceil(depth);
        // The rows a trip fills and every wave reads: the candidates, and
        // streamed the block's queries (a wave's own rows of them).
        let (filled, read) = match cfg.bk {
            None => (bkv, waves * bkv),
            Some(_) => (bq + bkv, waves * (cfg.warp_rows + bkv)),
        };
        let tile = filled * depth * elem;
        let steps = (bq * bkv * depth).div_ceil(STEP);
        let trips = tiles * chunks;
        let unique = ((self.n + self.m) * d * elem) as f64;
        let flight = planner.sms().min(blocks).max(1);
        // The query blocks and shares the blocks in flight read: split, a
        // group of query blocks walks every share ([`GROUP_ROWS`]).
        let (queries, shares) = match splits {
            1 => (flight, 1),
            _ => {
                let group = self.n.div_ceil(bq).min(GROUP_ROWS);
                (flight.min(group), flight.div_ceil(group).min(splits))
            }
        };
        let distinct = queries * (filled - bkv) + shares * bkv;
        let shared = 1.0 - distinct as f64 / (flight * filled) as f64;
        let mut fill = tile as f64 * planner.fill_byte(unique, (blocks * trips * tile) as f64, shared);
        if cfg.select.is_some() {
            // A trip stores its f32 scores, a row's at least a line.
            fill += (bq * (bkv * 4).max(LINE)) as f64 * costs.global_byte / chunks as f64;
            // Filtered, each row's least of the trip too, a sector a row.
            if cfg.select.is_some_and(|s| s.filter) {
                fill += (bq * SECTOR) as f64 * costs.global_byte / chunks as f64;
            }
        }
        let reads = (read * depth * elem) as f64 * costs.smem_byte;
        let mut epilogue = (bq * bkv) as f64 * pair * costs.score / chunks as f64;
        if k > 1 && cfg.select.is_none() {
            let lanes = cfg.lane_queue();
            let sharing = cfg.sharing(target, d, dtype).ok_or(Rejected::Layout)?;
            let queued = lanes.width * lanes.depth;
            let deals = bkv / lanes.width;
            // Search bounds over the lanes the kernel runs: queues as wide
            // as the lanes a row spreads over, a trip in few deals, a
            // register queue at least twice the lanes'.
            if lanes.width < sharing || deals > MOST_DEALS || cfg.queue.is_some_and(|q| 2 * queued > q) {
                return Err(Rejected::Lowers);
            }
            epilogue +=
                share.listing(cfg, lanes, target.wave, w) * waves as f64 * costs.score / (tiles * chunks) as f64;
        }
        let step = if elem == 1 { costs.step_i8 } else { costs.step };
        let work = steps as f64 * step + epilogue;
        let trip = costs.trip(target.fill, stages, cfg.unroll);
        let run = Run { blocks, smem, regs, waves, atoms, trips, reads, work, fill, trip, stages };
        let restarts = match cfg.bk {
            None => 0.0,
            Some(_) => {
                let rounds = blocks.div_ceil(planner.sms()).div_ceil(planner.resident(smem, regs, waves).max(1));
                (tiles * rounds) as f64 * costs.latency
            }
        };
        let listed = cfg.select.map_or(0.0, |s| self.select_cost(planner, &s, bkv));
        let merged = if splits > 1 && cfg.select.is_none() { self.merge_cost(planner, splits) } else { 0.0 };
        Ok((planner.pipelined(run) + restarts + listed + merged, regs))
    }

    /// What [`crate::kernels::nearest::nearest_merge`] costs folding
    /// `splits` shares of this problem, ns: each row's shares read once from
    /// DRAM, or its work, whichever is longer, spread over the SMs, and at
    /// least a memory latency a share (a block's loads wait on the share
    /// before). A row's least compares once a share; a list merges every
    /// later share by a bitonic merge's lower half over its runs of
    /// [`LANES`], a [`MERGE_SLOT`] each slot a stage.
    fn merge_cost(&self, planner: &Planner, splits: usize) -> f64 {
        let costs = &planner.device.costs;
        let slots = if self.k == 1 { 1.0 } else { (select_runs(self.k) * LANES) as f64 };
        let row = (splits - 1) as f64 * slots * (slots.log2() + 1.0) * MERGE_SLOT * planner.weights.merge;
        let bytes = ((splits + 1) * self.k * 8) as f64 * costs.global_byte;
        let spread = self.n as f64 * (row * costs.score).max(bytes) / planner.sms() as f64;
        spread.max(splits as f64 * costs.latency)
    }

    /// What [`crate::kernels::nearest::nearest_select`] costs listing this
    /// problem's score matrix under `select` (`bkv` candidates a trip), ns:
    /// [`Self::scan_cost`] of every score a row, or
    /// [filtered](SelectCfg::filter), of the trips' leasts first (the
    /// bound's listing), then of about `k` trips' scores a row (a step of
    /// [`LANES`] reads both of two shorter trips), every other step passed
    /// over at [`SKIP_STEP`].
    fn select_cost(&self, planner: &Planner, select: &SelectCfg, bkv: usize) -> f64 {
        if !select.filter {
            return self.scan_cost(planner, select, self.m, 0);
        }
        let groups = self.m.div_ceil(bkv);
        let read = (self.k * bkv.max(LANES)).min(self.m);
        let skipped = (self.m - read).div_ceil(LANES);
        NearestProblem { m: groups, ..*self }.scan_cost(planner, select, groups, 0)
            + self.scan_cost(planner, select, read, skipped)
    }

    /// [`Self::select_cost`] reading `read` scores a row and passing over
    /// `skipped` steps: each row read once from DRAM, or its work,
    /// whichever is longer, spread over the SMs. A row's work is a step a
    /// [`LANES`] of its scores, a queue insert a step some lane of it offers
    /// in (every step while the list fills, then the `j`-th with chance
    /// `k / j`), and a merge each time its lanes queue [`SELECT_FILLED`] of
    /// what they hold: the queues' bitonic sort and the list's bitonic merge,
    /// a [`MERGE_SLOT`] each slot a stage.
    fn scan_cost(&self, planner: &Planner, select: &SelectCfg, read: usize, skipped: usize) -> f64 {
        let costs = &planner.device.costs;
        let (k, queued) = (self.k as f64, (LANES * select.queue) as f64);
        let steps = read.div_ceil(LANES) as f64;
        let busy = steps.min(k * (1.0 + (steps / k).max(1.0).ln()));
        let width = (select_runs(self.k) * LANES) as f64;
        let depth = queued.log2();
        let merge = queued * depth * (depth + 1.0) / 2.0 + width * width.log2();
        let offers = NearestProblem { m: read, ..*self }.offers();
        let merges = offers / (queued * SELECT_FILLED) + 1.0;
        let (w_scan, w_merge) = (planner.weights.scan, planner.weights.merge);
        let row = steps * LANES as f64 * SCAN_SCORE * w_scan
            + skipped as f64 * SKIP_STEP * w_scan
            + busy * queued * INSERT_SLOT * w_scan
            + merges * merge * MERGE_SLOT * w_merge;
        let bytes = (read * 4) as f64 * costs.global_byte;
        self.n as f64 * (row * costs.score).max(bytes) / planner.sms() as f64
    }

    /// The scores of trip `t` of `cfg` below a row's `k`-th as a stream in
    /// random order offers them: `λ = k·ln((t+1)/t)` (every one of them
    /// while the list fills).
    fn lowered(&self, cfg: &NearestCfg, t: usize) -> f64 {
        match t * cfg.bkv < self.k {
            true => cfg.bkv.min(self.k) as f64,
            false => self.k as f64 * ((t + 1) as f64 / t as f64).ln(),
        }
    }

    /// A warp's list work over a block's candidates under `lanes`, in units
    /// of [`Costs::score`] a pair: each trip some row of the warp has a
    /// score below its `k`-th in (`warp_rows` rows' [`Self::lowered`]) takes
    /// each lane's least of its deals ([`DEAL_ELEM`] an element), then
    /// rounds of a push a lane, as many as the most of the trip's scores
    /// any lane takes (a Poisson count's maximum over the warp's queues, one
    /// at least, every deal at most): [`ROUND_ELEM`] a dealt element,
    /// [`PUSH_ELEM`] a queue slot, a vote; then the merges,
    /// one each time some queue of the warp takes a score past its depth
    /// (`N` pushes into `Q` queues fill one past `t` from about
    /// `N = (Q^t·(t+1)!)^(1/(t+1))`) and one at the end: the queues'
    /// bitonic sort and the merge into the list held in registers,
    /// [`LIST_SLOT`] a row's slot a stage. Queued, the merges of that list
    /// into the outputs too, one each time the warp's lists take a
    /// [`QUEUE_FILLED`] share of the room they leave the lanes' queues.
    pub(crate) fn listing(&self, cfg: &NearestCfg, lanes: LaneQueue, wave: u32, w: &Weights) -> f64 {
        let LaneQueue { width, depth, passes } = lanes;
        let (rows, tiles, wave) = (cfg.warp_rows as f64, self.m.div_ceil(cfg.bkv), f64::from(wave));
        let deals = (cfg.bkv / width) as f64;
        let (elems, t) = (rows * width as f64, depth as f64);
        let trip = |i: usize| {
            let hits = rows * self.lowered(cfg, i);
            let entered = 1.0 - (-hits).exp();
            if passes {
                // A vote on every deal of a trip some row enters, and a push
                // on each deal some row is dealt one below its `k`-th in.
                let dealt = deals * (1.0 - (-hits / deals).exp());
                return entered * deals * (elems * PASS_ELEM * w.deal + wave * DEAL_LANE * w.deal)
                    + dealt * (elems * (HIT_ELEM + t * PUSH_ELEM) * w.deal + wave * DEAL_LANE * w.deal);
            }
            let lambda = hits / elems;
            let rounds = (1.0 + lambda + (2.0 * lambda * elems.ln()).sqrt()).min(deals);
            let round =
                deals * elems * ROUND_ELEM * w.deal + elems * t * PUSH_ELEM * w.deal + wave * DEAL_LANE * w.deal;
            entered * (deals * elems * DEAL_ELEM * w.deal + wave * DEAL_LANE * w.deal + rounds * round)
        };
        let walk: f64 = (0..tiles).map(trip).sum();
        let fact: f64 = (1..=depth + 1).map(|i| i as f64).product();
        let per_merge = (elems.powf(t) * fact).powf(1.0 / (t + 1.0));
        let (held, queued) = (cfg.list_slots(self.k) as f64, (width * depth) as f64);
        let sort = queued * queued.log2() * (queued.log2() + 1.0) / 2.0;
        let merge = rows * (sort + held * (held.log2() + 1.0)) * LIST_SLOT * w.merge;
        let mut list = walk + (rows * self.offers() / per_merge + 1.0) * merge;
        if let Some(q) = cfg.queue {
            let stages = 1.0 + 2.0 * (q as f64).log2();
            let flush = (self.k.div_ceil(q) * q) as f64 * rows * stages * MERGE_SLOT * w.merge;
            list += (self.offers() / ((held - queued) * QUEUE_FILLED) + 1.0) * flush;
        }
        list
    }

    /// The scores a row offers its list over the candidates in random order:
    /// all of the first `k`, then the `t`-th with chance `k / t`.
    fn offers(&self) -> f64 {
        let (k, m) = (self.k as f64, self.m as f64);
        if m <= k { m } else { k * (1.0 + (m / k).ln()) }
    }

    /// Every config the ranking spans on `target`: every query block of up
    /// to [`NEAREST_WARPS`] warps owning one or two atom rows each, every
    /// candidate block, the rows resident or streamed at every depth chunk,
    /// every ring depth and unrolling.
    pub fn lattice(&self, target: &Target) -> Vec<NearestCfg> {
        self.span(target, &self.lists())
    }

    /// The list forms the lattice spans: a list in registers up to their
    /// most ([`REGISTER_K`]); a long one ([`LONG_FROM`] on) queued into the
    /// outputs and listed from a score matrix as well, the listings
    /// [filtered](SelectCfg::filter) where a row's trips are many to its
    /// list's slots ([`FILTER_GROUPS`]).
    fn lists(&self) -> Vec<(Option<usize>, Option<SelectCfg>)> {
        let registers = (self.k <= REGISTER_K).then_some((None, None));
        let long = self.k >= LONG_FROM;
        let queued = QUEUES.into_iter().filter(|&q| long && q < self.k).map(|q| (Some(q), None));
        let filters =
            [false, true].into_iter().filter(|&f| !f || self.m >= FILTER_GROUPS * self.k * CANDIDATE_BLOCKS[0]);
        let selects =
            filters.flat_map(|filter| SELECT_QUEUES.map(|queue| SelectCfg { warps: SELECT_WARPS, queue, filter }));
        let selects = selects.filter(|s| long && s.fits(self.k)).map(|s| (None, Some(s)));
        registers.into_iter().chain(queued).chain(selects).collect()
    }

    /// The tiles, depth chunks and pipelines the lattice spans each list
    /// form of [`Self::lists`] over: the body of [`Self::lattice`], handed
    /// the list forms to span (the bounds' fire-log prices what they cut
    /// over the same tiles).
    fn span(&self, target: &Target, lists: &[(Option<usize>, Option<SelectCfg>)]) -> Vec<NearestCfg> {
        let stages: &[usize] = if target.fill == Fill::CpAsync { &FA_STAGES } else { &NEAREST_STAGED };
        let mut out = vec![];
        for bq in QUERY_BLOCKS {
            for warp_rows in NEAREST_WARP_ROWS.into_iter().filter(|&r| bq.is_multiple_of(r)) {
                if bq / warp_rows > NEAREST_WARPS {
                    continue;
                }
                for bkv in CANDIDATE_BLOCKS {
                    // An 8-bit chunk as many bytes deep as a 16-bit one.
                    for bk in NEAREST_DEPTHS.map(|bk| bk.map(|bk| bk * 2 / self.dtype.bytes())) {
                        for &stages in stages {
                            for unroll in [false, true] {
                                for &(queue, select) in lists {
                                    let cfg = NearestCfg::new(bq, bkv, stages);
                                    let cfg = NearestCfg { unroll, warp_rows, bk, queue, select, ..cfg };
                                    if self.k == 1 || select.is_some() {
                                        out.push(cfg);
                                        continue;
                                    }
                                    // Lane queues as wide as the lanes a row spreads over or
                                    // twice, dealing a trip's scores in at most `MOST_DEALS`.
                                    let Some(sharing) = cfg.sharing(target, self.d, self.dtype) else { continue };
                                    let narrowest = sharing.max(bkv / MOST_DEALS).max(2);
                                    for width in [narrowest, 2 * narrowest].into_iter().filter(|&w| w <= bkv) {
                                        for depth in LANE_DEPTHS {
                                            let ways =
                                                [false, true].into_iter().filter(|&p| !p || bkv / width >= PASS_DEALS);
                                            for passes in ways {
                                                let lanes = Some(LaneQueue { width, depth, passes });
                                                out.push(NearestCfg { lanes, ..cfg });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The [`lattice`](Self::lattice)'s configs that run this problem on the
    /// planner's target, cheapest first, then by fewer registers. A list of
    /// `k > 1` is priced at its [split](Planner::nearest_split) where that
    /// is cheaper, as the tuner times it.
    pub(crate) fn ranked(&self, planner: &Planner) -> Vec<NearestCfg> {
        let lattice = self.lattice(&planner.device.target);
        let count = lattice.len();
        let split = |c: &NearestCfg, whole: f64| {
            let splits = planner.nearest_split(self, c).filter(|_| self.k > 1);
            let split = splits.and_then(|splits| self.admit(planner, &NearestCfg { splits, ..*c }).ok());
            split.map_or(whole, |(ns, _)| ns.min(whole))
        };
        let mut admitted: Vec<(f64, u32, NearestCfg)> = lattice
            .into_iter()
            .filter_map(|c| self.admit(planner, &c).ok().map(|(cost, regs)| (split(&c, cost), regs, c)))
            .collect();
        // A score matrix's listing costs alike whatever its product's tile:
        // the cheapest few products a listing.
        admitted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let gather = tracing::enabled!(target: crate::ops::PLAN, tracing::Level::DEBUG);
        let mut matrix = Vec::new();
        let mut products: HashMap<SelectCfg, HashSet<NearestCfg>> = HashMap::new();
        admitted.retain(|&(ns, _, c)| {
            let Some(select) = c.select else { return true };
            let tiles = products.entry(select).or_default();
            if tiles.len() < MATRIX_TILES {
                tiles.insert(c.tile());
            }
            let keep = tiles.contains(&c.tile());
            if !keep && gather {
                matrix.push(ns);
            }
            keep
        });
        let ranked = sorted(self, admitted, count);
        if gather && let Some(&(kept, ..)) = ranked.first() {
            fired("MATRIX_TILES", &matrix, kept);
            self.cut(planner, kept);
        }
        ranked.into_iter().map(|(_, _, c)| c).collect()
    }

    /// The fire-log of the bounds that cut whole variant families out of
    /// the lattice, `kept` the ranking's best: what [`LONG_FROM`],
    /// [`FILTER_GROUPS`] and [`PASS_DEALS`] cut, priced by admitting the cut
    /// candidates over the tiles [`Self::span`] walks. [`REGISTER_K`] and
    /// [`MOST_DEALS`] also reject in [`Self::admit`], so what they cut
    /// cannot be priced by it; only run with the plan log's debug level on.
    fn cut(&self, planner: &Planner, kept: f64) {
        let target = &planner.device.target;
        let price = |forms: Vec<(Option<usize>, Option<SelectCfg>)>| {
            self.span(target, &forms)
                .into_iter()
                .filter_map(|c| self.admit(planner, &c).ok().map(|(ns, _)| ns))
                .collect::<Vec<_>>()
        };
        // The queued and score-matrix forms under LONG_FROM.
        if self.k < LONG_FROM {
            let queued = QUEUES.into_iter().filter(|&q| q < self.k).map(|q| (Some(q), None));
            let selects = SELECT_QUEUES
                .into_iter()
                .map(|queue| SelectCfg { warps: SELECT_WARPS, queue, filter: false })
                .filter(|s| s.fits(self.k))
                .map(|s| (None, Some(s)));
            fired("LONG_FROM", &price(queued.chain(selects).collect()), kept);
        }
        // The filtered listings under FILTER_GROUPS' trip count; where
        // LONG_FROM already cut every form, they are its cut, not this one's.
        if self.k >= LONG_FROM && self.m < FILTER_GROUPS * self.k * CANDIDATE_BLOCKS[0] {
            let filtered = SELECT_QUEUES
                .into_iter()
                .map(|queue| SelectCfg { warps: SELECT_WARPS, queue, filter: true })
                .filter(|s| s.fits(self.k))
                .map(|s| (None, Some(s)))
                .collect();
            fired("FILTER_GROUPS", &price(filtered), kept);
        }
        // The pass variant of every dealt config PASS_DEALS skipped.
        let passes: Vec<f64> = self
            .lattice(target)
            .into_iter()
            .filter_map(|c| {
                let lanes = c.lanes.filter(|l| !l.passes && c.bkv / l.width < PASS_DEALS)?;
                let twin = NearestCfg { lanes: Some(LaneQueue { passes: true, ..lanes }), ..c };
                self.admit(planner, &twin).ok().map(|(ns, _)| ns)
            })
            .collect();
        fired("PASS_DEALS", &passes, kept);
    }
}

impl Planner {
    /// Every config of the nearest-candidate problem `p`, the cheapest first.
    pub(crate) fn nearest_ranked(&self, p: NearestProblem) -> Arc<[NearestCfg]> {
        if p.n * p.m * p.d * p.k == 0 {
            return Arc::new([]);
        }
        cached(&self.nearest, p, || p.ranked(self))
    }

    /// The candidates of `admitted` (the shortlist of
    /// [`Self::nearest_ranked`] for `p`): each split its
    /// [`Self::nearest_split`] ways where it splits,
    /// else whole, and the first whole as well, so a shape no split serves
    /// is timed (at 32768² rows and depth 768 on sm_86 every one of 36
    /// tiles ran faster split, 1.05–2.7 times, and timing both took 2.7
    /// times as long to tune; at 16384² rows and depths 512–1024 the best
    /// whole tile ran 8–12 % behind the best split on gfx1201, its unrolled
    /// pipeline included, 10–12 % on sm_86 (depth 768) and 18–34 % on
    /// gfx1151).
    pub(crate) fn nearest_splits(&self, p: &NearestProblem, admitted: &[NearestCfg]) -> Vec<NearestCfg> {
        let split = |c: &NearestCfg| self.nearest_split(p, c).map(|splits| NearestCfg { splits, ..*c });
        let whole = |i: usize, c: &NearestCfg, split: bool| (!split || i == 0).then_some(*c);
        admitted.iter().enumerate().flat_map(|(i, c)| [split(c), whole(i, c, split(c).is_some())]).flatten().collect()
    }

    /// The shares a query block of `cfg` splits the candidates of `p` into, if
    /// any: a power of two, each share [`SPLIT_TILES`] candidate tiles at
    /// least for each score a row keeps (every share fills its list again:
    /// at 16384² and depth 128 on sm_86 a 16-way split of 16 scores ran at
    /// 0.63 of the whole tile, a 4-way one of 128 candidates at 0.77),
    /// enough that the blocks in flight (every SM's resident ones) span the
    /// query blocks they walk at once. Streamed, those are a group
    /// of [`GROUP_ROWS`] (the grid's raster), so the blocks share their
    /// query and candidate lines as a GEMM's grouped tiles do: whole, a
    /// block's own query rows fall out of the cache between its candidate
    /// tiles (at depth 1024, 104 TFLOP/s whole and 129 split at 32768² rows
    /// on gfx1201, 21 and 25 on sm_86, 20 and 31 at 16384² on gfx1151).
    /// Resident, a block reads its rows once and only a grid of fewer blocks
    /// than SMs splits, a block an SM (at depth 128 a split ran 5–18 %
    /// behind whole on gfx1201).
    ///
    /// A score matrix splits as a least does, each share storing its own
    /// scores, where its query blocks are fewer than the SMs. A list of `k >
    /// 1` (in registers or queued) splits as the model prices cheapest
    /// ([`NearestProblem::admit`]: a few query blocks over many candidates leave SMs
    /// idle, against every share's list filled again and the merge of the
    /// shares; at 256 queries by 2^20 candidates at depth 128 on sm_86 the
    /// best of 4–64 shares measured 16–32 for 16 and 64 scores, 8 for 256,
    /// 8 at 1024 queries, as the model's), streamed up to [`REREAD_K`] at
    /// least as the rereads split, which the model does not see.
    pub(crate) fn nearest_split(&self, p: &NearestProblem, cfg: &NearestCfg) -> Option<usize> {
        let (target, d) = (&self.device.target, p.d);
        let blocks = p.n.div_ceil(cfg.bq);
        // A score matrix splits only to fill SMs its query blocks leave
        // idle: split over many rows its larger tiles priced cheapest
        // spilled on gfx1151 (16384² at depth 768, 1024 scores).
        if cfg.select.is_some() && blocks >= self.sms() {
            return None;
        }
        let (flight, rows) = match cfg.bk {
            Some(_) => {
                let regs = cfg.registers(target, d, p.k, p.dtype)?;
                let resident = self.resident(cfg.smem_bytes(d, p.dtype), regs, cfg.warps() as usize);
                (self.sms() * resident.max(1), blocks.min(GROUP_ROWS))
            }
            None => (self.sms(), blocks),
        };
        // A score matrix's shares store their own scores: none fills a list.
        let per = if cfg.select.is_some() { 1 } else { p.k };
        let most = prev_power_of_two(p.m.div_ceil(cfg.bkv) / (SPLIT_TILES * per));
        let fill = flight.div_ceil(rows).next_power_of_two().min(most);
        if per == 1 || most < 2 {
            return (fill > 1).then_some(fill);
        }
        let cost = |splits| p.admit(self, &NearestCfg { splits, ..*cfg }).ok().map(|(ns, _)| ns);
        let priced = (0..=most.ilog2()).map(|b| 1 << b).filter_map(|s| cost(s).map(|ns| (ns, s)));
        let priced = priced.min_by(|a, b| a.0.total_cmp(&b.0)).map_or(1, |(_, s)| s);
        let splits = if cfg.bk.is_some() && p.k <= REREAD_K { fill.max(priced) } else { priced };
        if tracing::enabled!(target: crate::ops::PLAN, tracing::Level::DEBUG)
            && let Some(kept) = cost(splits)
        {
            // SPLIT_TILES keeps every share `SPLIT_TILES` candidate tiles at
            // least: the counts past `most`, against the count kept.
            let past: Vec<f64> =
                (most.ilog2() + 1..=p.m.div_ceil(cfg.bkv).ilog2()).flat_map(|b| cost(1 << b)).collect();
            fired("SPLIT_TILES", &past, kept);
            // REREAD_K: the fill the rereads would share the bound held
            // back for a longer list.
            if cfg.bk.is_some()
                && p.k > REREAD_K
                && fill > priced
                && let Some(ns) = cost(fill)
            {
                fired("REREAD_K", &[ns], kept);
            }
        }
        (splits > 1).then_some(splits)
    }

    /// The depth a nearest kernel over `dtype` runs `d` at: a multiple of
    /// its matrix core's, the columns past `d` zero.
    pub fn nearest_depth(&self, d: usize, dtype: ScalarDType) -> Option<usize> {
        Some(d.next_multiple_of(self.device.target.geometry_of(dtype)?.k as usize))
    }
}
