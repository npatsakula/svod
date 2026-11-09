//! GEMM-shaped configs: the tile and warp-grid lattice, what the lowering
//! lays a tile's operands into, and what a problem's products cost on a
//! planner.

use std::sync::Arc;

use crate::atoms::{Core, Fill, Target};
use std::num::NonZeroUsize;

use crate::ir::{Axis, Schedule, Shape};
use crate::kernels::Admitted;
use crate::kernels::gemm::{GemmCfg, Rope, TileOrder};
use crate::layouts::{Issue, Swizzle, WarpGrid, chunked, fills_in_chunks};
use crate::tune::SETTLED;

use super::cost::{
    Claim, ELEM, Family, Feasible, Kept, Kind, Ops, Planner, Reach, STEP, Spread, Steps, Stream, cached,
};
use super::search::{Rejected, ShapeRejected, TargetRejected, Tiled, sorted, spread};

impl GemmCfg {
    /// The operand layouts of the tile's product on `target`, over the
    /// config's core and warp grid, as the lowering lays them.
    fn issue(&self, target: &Target, bk: usize) -> Option<Issue> {
        let ([bm, bn, _], [wr, wc]) = (self.tile, self.warps);
        target.issue(self.core, WarpGrid { rows: wr, cols: wc }, bm, bn, bk)
    }

    /// Whether the warp grid tiles the output with `target`'s matrix core
    /// atoms and the ring depth is one the target's prefetch runs. A
    /// warpgroup config's rows of shared operands must be one swizzle span
    /// (32, 64 or 128 bytes), the descriptor's own.
    fn tiles(&self, target: &Target) -> bool {
        let bk = self.tile[2];
        let row = self.core != Core::Group || Swizzle::spans(bk, ELEM);
        // Two crews: the two warp rows of the warp-level core, one slot, and
        // k-slices of a product (one atom deep) each with another to
        // overlap, on a target that interleaves crews.
        let crews = match self.schedule {
            Schedule::Ring { .. } => true,
            Schedule::PingPong { kslice } => {
                let k = kslice.get();
                self.core == Core::Warp
                    && target.kslice() == Some(k)
                    && self.warps[0] == 2
                    && bk.is_multiple_of(k)
                    && bk >= 2 * k
            }
        };
        row && crews && self.issue(target, bk).is_some() && target.fill.runs(self.schedule)
    }

    /// Whether the config lowers on `target`: it [`tiles`](Self::tiles), and
    /// both operand tiles fill in whole 16-byte chunks per thread
    /// ([`fills_in_chunks`]).
    pub(crate) fn fits(&self, target: &Target) -> bool {
        let ([bm, bn, bk], [wr, wc]) = (self.tile, self.warps.map(|w| w as usize));
        let threads = target.wave as usize * wr * wc;
        let fills = |rows| fills_in_chunks(Shape::new(rows, bk), ELEM, threads);
        self.tiles(target) && fills(bm) && fills(bn)
    }

    /// The registers a lane holds for a GEMM over a weight of `halves`
    /// halves (two when gated): the f32 accumulators, the 16-bit operand
    /// fragments and, where the fill goes through registers, its staged
    /// chunks, from the layouts the lowering assigns. `None` when the tile
    /// does not lay out. Addressing and the epilogue's temporaries are not
    /// counted.
    pub(crate) fn registers(&self, target: &Target, halves: usize) -> Option<u32> {
        let ([bm, bn, bk], [wr, wc]) = (self.tile, self.warps);
        let issue = self.issue(target, bk)?;
        let halves = halves as u32;
        let acc = issue.c.regs() * halves;
        let operands = self.operand_registers(&issue, halves);
        let fill = if target.fill == Fill::CpAsync {
            0
        } else {
            let fill =
                |rows: usize| chunked(Shape::new(rows, bk), 2, wr * wc, target.wave, Axis::Col).map(|l| l.regs());
            (fill(bm)? + fill(bn * halves as usize)?).div_ceil(2)
        };
        Some(acc + operands + fill)
    }

    /// The registers a lane holds at once however the lowering streams its
    /// fragments: the f32 accumulators, and one matrix-core depth of operand
    /// fragments, a floor of [`Self::registers`]; on the warpgroup core the
    /// operands stay in shared memory and their descriptors are counted
    /// instead. Past what the lane holds ([`crate::atoms::Occupancy::holds`]:
    /// the accumulators in a file of their own where the target has one) or
    /// past the block's share of the file the kernel cannot compile without
    /// spilling.
    pub(crate) fn live_registers(&self, target: &Target, halves: usize) -> Option<(u32, u32)> {
        let issue = self.issue(target, 16)?;
        let halves = halves as u32;
        Some((issue.c.regs() * halves, self.operand_registers(&issue, halves)))
    }

    /// The 32-bit registers a lane holds of the operand fragments; on the
    /// warpgroup core, whose operands stay in shared memory, the two 64-bit
    /// descriptors of every `k16` step of a chain instead.
    fn operand_registers(&self, issue: &Issue, halves: u32) -> u32 {
        match self.core {
            Core::Warp | Core::Wide => (issue.a.regs() + issue.b.regs() * halves).div_ceil(2),
            Core::Group => 4 * issue.a.reps[1] * halves,
        }
    }
}

pub(crate) const fn gemm_cfg(tile: [usize; 3], stages: usize, warps: [u32; 2], unroll: bool) -> GemmCfg {
    GemmCfg { core: Core::Warp, tile, warps, schedule: Schedule::ring(stages, unroll), order: TileOrder::DEFAULT }
}

/// Model: a representative layer epilogue's tile operations an output
/// element — a gelu's forty (the abs, three affines, the reciprocal, the
/// polynomial's four multiplies and affines and its product, the square, the
/// `exp2`'s affine and `exp2`, the two tails and the final affine), the
/// claim diff's own ±0.7 % pin against a program's instruction mix. The
/// ranking is epilogue-kind-blind (the epilogue varies by layer, the tune
/// key's salt, not by tile): this bundle — the models' ff layers' — is what
/// orders tiles by the epilogue work they carry, their total over a problem
/// the same whatever the tile.
pub(crate) const EPILOGUE_OPS: f64 = 40.0;

/// Model: a rotary turn's tile operations a turned output element — the two
/// tables' casts, the sign's multiply, the element's own multiply feeding
/// the fused add and the mul_add itself; the sign tile's column arithmetic
/// (the divides, compare and select over a `[1, bn]` tile) is shared over
/// the tile's rows.
pub(crate) const ROPE_OPS: f64 = 5.0;

/// Search bound: the output tile's rows, the first of the knobs a GEMM-shaped
/// kernel is enumerated over.
const ROWS: [usize; 4] = [32, 64, 128, 256];
/// Search bound: the output tile's columns (48 and 96 for the convolution
/// bodies a power of two would pad by a third).
const COLS: [usize; 5] = [48, 64, 96, 128, 256];
/// Search bound: the reduction depths of a step.
pub(super) const DEPTHS: [usize; 3] = [16, 32, 64];
/// Search bound: the warp grids over the output tile.
const GRIDS: [[u32; 2]; 9] = [[1, 1], [1, 2], [2, 1], [2, 2], [1, 4], [4, 1], [2, 4], [4, 2], [4, 4]];
/// Search bound: warp grids of whole warpgroups down the rows, one to four of them.
const GROUP_GRIDS: [[u32; 2]; 6] = [[4, 1], [4, 2], [8, 1], [8, 2], [16, 1], [4, 4]];

/// A core with the warp grids and ring depths the lattice spans it over.
type Cores = [(Core, &'static [[u32; 2]], &'static [usize])];

/// Every config the construct spans on `target`: the tiles and warp grids,
/// the ring depths its prefetch runs, rolled and unrolled; on a target with
/// warpgroup instructions, the tiles of whole warpgroups on them as well.
pub fn gemm_lattice(target: &Target) -> Vec<GemmCfg> {
    let cores: &Cores = match (target.fill, target.group_mma.is_empty()) {
        (Fill::Staged, _) => &[(Core::Warp, &GRIDS, &[2, 1])],
        (Fill::CpAsync, true) => &[(Core::Warp, &GRIDS, &[3, 2, 1])],
        // A warpgroup ring keeps a chain in flight in a slot of its own.
        (Fill::CpAsync, false) => &[(Core::Warp, &GRIDS, &[3, 2, 1]), (Core::Group, &GROUP_GRIDS, &[4, 3])],
    };
    let mut out = vec![];
    for bm in ROWS {
        for bn in COLS {
            for bk in DEPTHS {
                for &(core, grids, stages) in cores {
                    for &warps in grids {
                        // A grid that does not tile the core's atoms lowers no tile.
                        if target.issue(core, WarpGrid { rows: warps[0], cols: warps[1] }, bm, bn, bk).is_none() {
                            continue;
                        }
                        for &stages in stages {
                            // One slot has no slot arithmetic to fold: rolled only.
                            let unrolls: &[bool] = if stages == 1 { &[false] } else { &[false, true] };
                            out.extend(
                                unrolls
                                    .iter()
                                    .map(|&unroll| GemmCfg { core, ..gemm_cfg([bm, bn, bk], stages, warps, unroll) }),
                            );
                            // Two crews: the warp-level core's two warp rows a
                            // cluster apart over the one slot.
                            let kslice = target.kslice().and_then(NonZeroUsize::new);
                            if core == Core::Warp
                                && stages == 1
                                && warps[0] == 2
                                && let Some(kslice) = kslice
                            {
                                let schedule = Schedule::PingPong { kslice };
                                out.push(GemmCfg { schedule, ..gemm_cfg([bm, bn, bk], 1, warps, false) });
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// A GEMM-shaped problem: `batches` products of `m × k` by `k × n·halves`
/// (two halves: a gated weight, whose block stages a `2·bn`-wide B tile
/// and holds two accumulators). With `rope`, the output's first heads turn
/// in the kernel's store: only tiles that hold whole heads lower, and the
/// turn's operations and table reads are claimed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GemmProblem {
    pub batches: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub halves: usize,
    pub rope: Option<Rope>,
}

impl GemmProblem {
    pub(super) fn blocks(&self, cfg: &GemmCfg) -> usize {
        let (gm, gn) = self.grid(cfg);
        self.batches * gm * gn
    }

    /// A batch's grid of `cfg`'s output tiles, `[rows, cols]`.
    fn grid(&self, cfg: &GemmCfg) -> (usize, usize) {
        (self.m.div_ceil(cfg.tile[0]), self.n.div_ceil(cfg.tile[1]))
    }

    /// The tile rows and columns `blocks` consecutive blocks of `cfg`'s
    /// raster span in a batch's grid: down a group's rows a column at a
    /// time, every column of a group before the next group's.
    fn window(&self, cfg: &GemmCfg, blocks: usize) -> (usize, usize) {
        let (gm, gn) = self.grid(cfg);
        let (group, blocks) = (cfg.order.group(gm), blocks.clamp(1, gm * gn));
        (group.min(blocks).max(blocks.div_ceil(gn)), blocks.div_ceil(group).min(gn))
    }

    /// The ring's fill under `cfg`'s raster, in a batch's grid: a trip's
    /// bytes, the operands' distinct ones, the share the blocks in flight
    /// at once (one an SM) read of one another's lines — a window's A rows
    /// and B columns ([`Self::window`]) — and the reach of the rest's
    /// re-reads. A B panel (`bn × k`) returns a group later, after the
    /// group's A panels and the whole of B have streamed; every other
    /// re-read (an A panel a column of its group later, a B panel within
    /// its group) after the panels of the blocks in flight, or of the
    /// group's rows where those are more.
    fn stream(&self, planner: &Planner, cfg: &GemmCfg) -> Stream {
        let ([bm, bn, bk], halves) = (cfg.tile, self.halves);
        let (gm, gn) = self.grid(cfg);
        let (group, wide) = (cfg.order.group(gm), bn * halves);
        let flight = planner.sms().min(self.blocks(cfg)).max(1).min(gm * gn);
        let (rows, cols) = self.window(cfg, flight);
        let shared = 1.0 - (rows * bm + cols * wide) as f64 / (flight * (bm + wide)) as f64;
        let panel = (self.k * ELEM) as f64;
        let (rows, cols) = self.window(cfg, flight.max(group));
        let near = (rows * bm + cols * wide) as f64 * panel;
        let far = (group * bm + gn * wide) as f64 * panel;
        // In tile rows of a batch: the bytes the blocks in flight do not
        // share, past each panel's first read, and B's a group later.
        let unshared = (gm * gn * (bm + wide)) as f64 * (1.0 - shared);
        let rereads = unshared - (gm * bm + gn * wide) as f64;
        let later = (gn * wide * (gm.div_ceil(group) - 1)) as f64;
        let late = if rereads > 0.0 { (later / rereads).min(1.0) } else { 0.0 };
        Stream {
            bytes: ((bm + wide) * bk * ELEM) as f64,
            unique: (self.batches * (self.m + self.n * halves) * self.k * ELEM) as f64,
            shared,
            reach: [Reach { share: 1.0 - late, set: near }, Reach { share: late, set: far }],
        }
    }

    /// The rasters the planner prices `cfg` at on this problem: the
    /// default, then groups of one tile row, two, four, … below the grid's
    /// rows and the grid's rows whole (column-major), each a group height
    /// the default does not already walk.
    fn orders(&self, cfg: &GemmCfg) -> impl Iterator<Item = TileOrder> {
        let gm = self.grid(cfg).0;
        let default = TileOrder::DEFAULT.group(gm);
        let heights = (0..usize::BITS).map(|b| 1usize << b).take_while(move |&g| g < gm).chain([gm]);
        let others = heights.filter(move |&g| g != default).filter_map(TileOrder::grouped);
        std::iter::once(TileOrder::DEFAULT).chain(others)
    }

    /// The share of the tiles' area past `m × n`.
    pub(super) fn padding(&self, cfg: &GemmCfg) -> f64 {
        let [bm, bn, _] = cfg.tile;
        let area = self.m.div_ceil(bm) * bm * self.n.div_ceil(bn) * bn;
        (area - self.m * self.n) as f64 / area as f64
    }

    /// What `cfg` costs on the planner's device ([`Planner::price`] of its
    /// [`Claim`](super::cost::Claim), ns of
    /// SM time) and a lane's registers by the layout count, which over-counts
    /// what the lowering streams (the compiled kernel judges, [`shortlist`](super::search::shortlist));
    /// `None` when it does not lower, or when its [live
    /// registers](GemmCfg::live_registers) alone pass the lane's limit; why it
    /// is left out otherwise. A trip
    /// pays its fragment reads, fill, matrix-core steps (padding included),
    /// the representative epilogue's tile operations and overhead. The
    /// fill's bytes are the whole operands' distinct ones, the blocks in
    /// flight at once (one an SM at least) share the A rows and B columns of
    /// the raster's window whatever the cache holds of the whole, and the
    /// rest's re-reads come back through the working sets the raster
    /// reaches them over ([`Self::stream`]). Two
    /// crews are priced as the one-crew pipeline of their tile: the overlap
    /// of one crew's products with the other's loads has no term, since
    /// only CDNA runs them ([`Target::crews`]) and no CDNA card has measured
    /// one; the tuner times them beside the rest.
    pub fn admit(&self, planner: &Planner, cfg: &GemmCfg) -> Result<(f64, u32), Rejected> {
        planner.admit(self, cfg)
    }

    /// `cfg` admitted for this problem ([`Admitted`]), or why it is not.
    pub fn admitted<'t>(&self, planner: &'t Planner, cfg: GemmCfg) -> Result<Admitted<'t, GemmCfg>, Rejected> {
        planner.admitted(self, cfg)
    }

    /// Whether `cfg` runs `halves` halves on `target`, a `rope` turning the
    /// output, whatever the shape: a tile holding whole heads (a turn moves
    /// each element's partner half a head along, which the flip the kernel
    /// issues reaches only within a lane's own head), one that lowers
    /// ([`GemmCfg::fits`]), fits the target's shared memory and holds no
    /// more registers than a lane may; the registers it holds by the layout
    /// count.
    pub(crate) fn feasible(
        cfg: &GemmCfg,
        target: &Target,
        &(halves, rope): &(usize, Option<Rope>),
    ) -> Result<u32, TargetRejected> {
        if rope.is_some_and(|rope| !cfg.turns(rope.head_dim))
            || target.geometry_on(cfg.core).is_none()
            || !cfg.fits(target)
        {
            return Err(TargetRejected::Lowers);
        }
        let smem = cfg.smem_bytes(halves == 2, ELEM);
        if smem > target.smem_bytes {
            return Err(TargetRejected::Smem { need: smem, have: target.smem_bytes });
        }
        let regs = cfg.registers(target, halves).ok_or(TargetRejected::Layout)?;
        let (acc, live) = cfg.live_registers(target, halves).ok_or(TargetRejected::Layout)?;
        let budget = target.lane_budget(cfg.warps[0] * cfg.warps[1]);
        if !target.occupancy.holds(acc, live) || acc + live > budget {
            return Err(TargetRejected::Registers { need: acc + live, have: budget });
        }
        Ok(regs)
    }

    /// A feasible config's [claim](super::cost::Claim) on this shape: its
    /// checks first — a step at least two matrix-core depths where the
    /// reduction allows one, a step that divides it — then its resources. A
    /// warpgroup product reads no fragments and holds one accumulator chain
    /// per instruction column block; its step is the group's.
    pub(crate) fn claim(&self, planner: &Planner, cfg: &GemmCfg, regs: u32) -> Result<Claim, Rejected> {
        let target = &planner.device.target;
        let ([bm, bn, bk], [wr, wc]) = (cfg.tile, cfg.warps.map(|w| w as usize));
        let atom = target.geometry_on(cfg.core).ok_or(Rejected::Target(TargetRejected::Lowers))?;
        // A step is at least two matrix-core depths, unless the reduction dim
        // allows nothing deeper.
        let depth = 2 * atom.k as usize;
        if bk < depth && self.k.is_multiple_of(depth) {
            return Err(ShapeRejected::Shallow.into());
        }
        if !self.k.is_multiple_of(bk) {
            return Err(ShapeRejected::Divides.into());
        }
        let smem = cfg.smem_bytes(self.halves == 2, ELEM);
        let (halves, waves) = (self.halves, wr * wc);
        // A warpgroup product reads no fragments and holds one accumulator
        // chain per instruction column block; its step is the group's.
        let (atoms, fragments) = match cfg.core {
            Core::Warp | Core::Wide => (
                (bm / wr / atom.m as usize) * (bn * halves / wc / atom.n as usize),
                waves * (bm / wr + bn * halves / wc) * bk * ELEM,
            ),
            Core::Group => {
                let group = target.group_mma.first().ok_or(Rejected::Target(TargetRejected::Lowers))?;
                let n = group.n_for((bn / wc) as u32).ok_or(Rejected::Target(TargetRejected::Lowers))? as usize;
                (bn * halves / wc / n, 0)
            }
        };
        let steps = (bm * bn * halves * bk / STEP) as f64;
        let (blocks, trips, stages) = (self.blocks(cfg), self.k / bk, cfg.stages());
        let fill = self.stream(planner, cfg);
        // The epilogue's operations over the tile's output elements, one a
        // lane's element each, paid over the trips that fold into one
        // output: their whole is m·n·[`EPILOGUE_OPS`] of lane operations
        // whatever the tile, so what the term orders is a trip's balance of
        // steps against epilogue. A rope's turn covers the `turned` share of
        // the columns (the last block straddling them half-turned on
        // average), and every block whose window reaches them reads, once,
        // its rows' positions of both tables — `half` distinct elements a
        // row each — the side stream: priced at the ladder of its own
        // working set through the epilogue's pipe, not folded into the
        // ring's fill, its distinct bytes in the run's floor.
        let epilogue = EPILOGUE_OPS * (bm * bn) as f64 / trips as f64;
        let (mut ops, mut side) = (epilogue, Stream::default());
        if let Some(rope) = self.rope {
            let (half, turned) = (rope.head_dim / 2, rope.heads * rope.head_dim);
            let share = turned as f64 / self.n as f64;
            ops += ROPE_OPS * (bm * bn) as f64 * share / trips as f64;
            let reading = turned.div_ceil(bn) as f64 / self.n.div_ceil(bn).max(1) as f64;
            side = Stream {
                bytes: 2.0 * (bm * half) as f64 * ELEM as f64 * reading / trips as f64,
                unique: (2 * rope.seq * half * ELEM) as f64,
                shared: fill.shared,
                ..Stream::default()
            };
        }
        Ok(Claim {
            blocks,
            smem,
            regs,
            waves,
            atoms,
            steps: match cfg.core {
                Core::Warp => Steps::F16(steps),
                Core::Wide => Steps::Wide(steps),
                Core::Group => Steps::Group(steps),
            },
            trips,
            reads: fragments as f64,
            ops: Ops::of(Kind::Epilogue, ops),
            fill,
            side,
            stores: 0.0,
            ring: stages,
            unroll: cfg.unroll(),
            waits: 0.0,
            sections: [Spread::default(); 2],
        })
    }

    /// The lattice's configs that run this problem on the planner's target,
    /// every pipeline (depth, ring depth, unrolling) of every tile, cheapest
    /// first (then by fewer registers) and [`spread`] over the block tiles
    /// the cost band holds: the shortlist keeps a tile's first clean one
    /// ([`shortlist`](super::search::shortlist)).
    pub(crate) fn ranked(&self, planner: &Planner) -> Vec<GemmCfg> {
        let feasible = planner.feasible(self);
        let planned: Vec<_> = feasible.iter().filter_map(|f| self.planned(planner, f)).collect();
        let mut admitted: Vec<(f64, u32, GemmCfg)> = planned.iter().map(|(best, _)| *best).collect();
        // One alternative a tile: its cheapest pipeline at the default
        // raster, where the model walks it otherwise unsure ([`Self::planned`]).
        let mut order: Vec<usize> = (0..planned.len()).collect();
        order.sort_by(|&a, &b| planned[a].0.0.total_cmp(&planned[b].0.0));
        let mut tiles = std::collections::HashSet::new();
        let alternatives = order.into_iter().filter(|&i| tiles.insert(planned[i].0.2.tile()));
        admitted.extend(alternatives.filter_map(|i| planned[i].1));
        let spanned = feasible.len() + admitted.len() - planned.len();
        let ranked = sorted(self, admitted, spanned);
        spread(ranked.into_iter().map(|(cost, _, c)| (cost, c)).collect(), |c| [c.tile[0], c.tile[1]])
    }

    /// A feasible config at the raster the model prices cheapest on this
    /// problem ([`Self::orders`], the default on a tie), its cost and
    /// registers, and the default raster's beside it where the model is
    /// unsure of the departure; `None` where the shape leaves it out. The
    /// raster moves only the fill's price: without a cache ladder every
    /// raster prices the same and the default stands.
    ///
    /// The model weighs the blocks in flight sharing B columns (a taller
    /// group) against the reach of the re-reads ([`Self::stream`]): a group
    /// whose A panels and the window's B panels outgrow the cache sends
    /// them back to DRAM between columns. Unsure: the default priced within
    /// [`SETTLED`] of the pick, a gap the tuner's own rounds do not resolve
    /// either. The term is priced by the ladder alone, never fitted to a
    /// raster's measured time, while every store so far timed the default,
    /// so there the tuner times the default too (one pipeline a tile, as a
    /// twin of the tile: [`Tiled::twins`]).
    #[allow(clippy::type_complexity)]
    fn planned(
        &self,
        planner: &Planner,
        f: &Feasible<GemmCfg>,
    ) -> Option<((f64, u32, GemmCfg), Option<(f64, u32, GemmCfg)>)> {
        let price = |order| {
            let cfg = GemmCfg { order, ..f.cfg };
            planner.price_of(self, &Feasible { cfg, regs: f.regs }).map(|(cost, regs)| (cost, regs, cfg))
        };
        let default = price(TileOrder::DEFAULT)?;
        if planner.device.costs.l2_bytes <= 0.0 {
            return Some((default, None));
        }
        let best = self.orders(&f.cfg).skip(1).filter_map(price).fold(default, |a, b| if b.0 < a.0 { b } else { a });
        Some((best, (best.2.order != TileOrder::DEFAULT && default.0 <= best.0 * SETTLED).then_some(default)))
    }
}

impl Family for GemmProblem {
    type Cfg = GemmCfg;
    type Static = (usize, Option<Rope>);

    fn statics(&self) -> Self::Static {
        (self.halves, self.rope)
    }

    fn lattice(&self, target: &Target) -> Vec<GemmCfg> {
        gemm_lattice(target)
    }

    fn feasible(cfg: &GemmCfg, target: &Target, statics: &Self::Static) -> Result<u32, TargetRejected> {
        GemmProblem::feasible(cfg, target, statics)
    }

    fn kept(planner: &Planner) -> &Kept<Self::Static, GemmCfg> {
        &planner.gemm_feasible
    }

    fn claim(&self, planner: &Planner, cfg: &GemmCfg, regs: u32) -> Result<Claim, Rejected> {
        GemmProblem::claim(self, planner, cfg, regs)
    }
}

impl Planner {
    pub(super) fn ranked(&self, p: GemmProblem) -> Arc<[GemmCfg]> {
        cached(&self.ranked, p, || p.ranked(self))
    }

    /// Every GEMM config for `batches` GEMMs of `m` rows, `n` columns and
    /// reduction dim `k`, the cheapest of the lattice first.
    pub fn gemm_ranked(&self, batches: usize, m: usize, n: usize, k: usize, gated: bool) -> Arc<[GemmCfg]> {
        self.gemm_ranked_rope(batches, m, n, k, gated, None)
    }

    /// [`Planner::gemm_ranked`] for a problem whose output `rope` turns in
    /// the kernel's store: only tiles that hold whole heads of its head
    /// width lower, their turn priced.
    pub fn gemm_ranked_rope(
        &self,
        batches: usize,
        m: usize,
        n: usize,
        k: usize,
        gated: bool,
        rope: Option<Rope>,
    ) -> Arc<[GemmCfg]> {
        if batches * m * n * k == 0 {
            return Arc::new([]);
        }
        self.ranked(GemmProblem { batches, m, n, k, halves: if gated { 2 } else { 1 }, rope })
    }
}
