//! GEMM-shaped configs: the tile and warp-grid lattice, what the lowering
//! lays a tile's operands into, and what a problem's products cost on a
//! planner.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::atoms::{Core, Fill, Target};
use std::num::NonZeroUsize;

use crate::ir::{Axis, Schedule, Shape};
use crate::kernels::Act;
use crate::kernels::gemm::{Epilogue, GemmCfg, Rope, TAIL_MERGE, TailPlan, TailSpec, TailSplit, TileOrder};
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
    GemmCfg {
        core: Core::Warp,
        tile,
        warps,
        schedule: Schedule::ring(stages, unroll),
        order: TileOrder::DEFAULT,
        tail: None,
    }
}

/// Model: an activation's tile operations an element, as the targets lower
/// it past the product (AMD's count; sm_86 lowers one more): a gelu's
/// Abramowitz–Stegun erf (the abs, the affines, the reciprocal, the
/// polynomial, the square, the `exp2` and the tails), a silu's sigmoid.
pub(crate) fn act_ops(act: Act) -> u32 {
    match act {
        Act::None => 0,
        Act::Gelu => 29,
        Act::Silu => 4,
    }
}

/// Model: a layer epilogue's tile operations an output element, each one a
/// lane's: the accumulator's rounding to the stored type, a bias's add a
/// half, the activation ([`act_ops`]), the gated product, the scale's
/// multiply, and the residual's add behind its load's conversion. The claim
/// diff pins each against the program's own operations past its last
/// product (`the_gemm_epilogue_claim_holds_its_program`); a rope's turn is
/// [`ROPE_OPS`] over its share, claimed apart.
pub(crate) fn epilogue_ops(e: &Epilogue) -> u32 {
    let round = u32::from(!e.out_f32);
    let halves = u32::from(e.gated) + 1;
    let bias = if e.bias { halves } else { 0 };
    let residual = if e.residual { 1 + round } else { 0 };
    round + bias + act_ops(e.act) + u32::from(e.gated) + u32::from(e.scale.is_some()) + residual
}

/// Model: a rotary turn's tile operations a turned output element — the two
/// tables' casts, the sign's multiply, the element's own multiply feeding
/// the fused add and the mul_add itself; the sign tile's column arithmetic
/// (the divides, compare and select over a `[1, bn]` tile) is shared over
/// the tile's rows.
pub(crate) const ROPE_OPS: f64 = 5.0;

/// Search bound: the most blocks a tail tile splits over ([`TailSplit`]):
/// past it the partials' bytes outgrow what a wave's quantisation can save.
pub(crate) const TAIL_SPLITS: usize = 16;

/// Model: what a launch costs beside its blocks' time, ns — the dispatch of
/// its grid behind the kernel before it and the drain of that kernel's last
/// blocks — paid by every launch of a split past its first. No probe
/// measures it and no kernel's stamps see it: the planner prices a tail
/// split's launches with it ([`GemmProblem::plan_price`]) and the tuner adds
/// it to a candidate's summed kernel times ([`crate::tune::launched`]);
/// `gemm_tail_probe` reports the kernels' sums alone.
pub(crate) const LAUNCH: f64 = 4000.0;

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
/// turn's operations and table reads are claimed. Its `epilogue` is the
/// layer's tile operations an output element ([`epilogue_ops`]); `out`
/// the bytes an output element moves once its sum is done (its store and a
/// residual's read, each of the output's type) and `bias` a bias column's
/// (none without one).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GemmProblem {
    pub batches: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub halves: usize,
    pub rope: Option<Rope>,
    pub epilogue: u32,
    pub out: usize,
    pub bias: usize,
}

/// One launch of a GEMM's plan: `tiles` output tiles of a batch's walk
/// (every batch's), each over `split` blocks a share of the reduction long.
#[derive(Clone, Copy, Debug)]
struct Part {
    tiles: usize,
    split: usize,
}

/// The bytes of a split's f32 partial.
const F32: usize = 4;

/// The most bytes a tail split's f32 partials take, allocated at every call
/// beside the operands: the share of device memory a score matrix's block
/// takes, an eighth of it ([`crate::kernels::nearest::scratch`]).
fn partials(planner: &Planner) -> usize {
    crate::kernels::nearest::scratch(planner.device.target.memory_bytes).saturating_mul(F32)
}

impl GemmProblem {
    /// `batches` products of `m × k` by `k × n` under `epilogue`.
    pub fn new(batches: usize, m: usize, n: usize, k: usize, epilogue: &Epilogue) -> Self {
        let (halves, rope) = (usize::from(epilogue.gated) + 1, epilogue.rope);
        let out = if epilogue.out_f32 { F32 } else { ELEM } * (1 + usize::from(epilogue.residual));
        let bias = if epilogue.bias { halves * ELEM } else { 0 };
        Self { batches, m, n, k, halves, rope, epilogue: epilogue_ops(epilogue), out, bias }
    }

    pub(super) fn blocks(&self, cfg: &GemmCfg) -> usize {
        self.batches * self.tiles(cfg)
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

    /// The ring's fill under `cfg`'s raster over `part`'s tiles of a
    /// batch's grid (a run of the walk from its first, the whole grid
    /// alike): a trip's bytes, the operands' distinct ones the part's tiles
    /// span, the share the blocks in flight at once (one an SM) read of one
    /// another's lines — a window's A rows and B columns ([`Self::window`])
    /// — and the reach of the rest's re-reads. A B panel (`bn × k`) returns
    /// a group later, after the group's A panels and the whole of B have
    /// streamed; every other re-read (an A panel a column of its group
    /// later, a B panel within its group) after the panels of the blocks in
    /// flight, or of the group's rows where those are more.
    fn stream(&self, planner: &Planner, cfg: &GemmCfg, part: Part) -> Stream {
        let ([bm, bn, bk], halves) = (cfg.tile, self.halves);
        let (gm, gn) = self.grid(cfg);
        let (group, wide) = (cfg.order.group(gm), bn * halves);
        let flight = planner.sms().min(self.batches * part.tiles * part.split).max(1).min(part.tiles);
        let (rows, cols) = self.window(cfg, flight);
        let shared = 1.0 - (rows * bm + cols * wide) as f64 / (flight * (bm + wide)) as f64;
        let panel = (self.k * ELEM) as f64;
        let (rows, cols) = self.window(cfg, flight.max(group));
        let near = (rows * bm + cols * wide) as f64 * panel;
        let far = (group * bm + gn * wide) as f64 * panel;
        // The part's rows and columns: the grid's whole for every tile.
        let (rows, cols) = self.window(cfg, part.tiles);
        // In tile rows of a batch: the bytes the blocks in flight do not
        // share, past each panel's first read, and B's a group later.
        let unshared = (part.tiles * (bm + wide)) as f64 * (1.0 - shared);
        let rereads = unshared - (rows * bm + cols * wide) as f64;
        let later = (cols * wide * (rows.div_ceil(group) - 1)) as f64;
        let late = if rereads > 0.0 { (later / rereads).min(1.0) } else { 0.0 };
        let spanned = (rows * bm).min(self.m) + (cols * wide).min(self.n * halves);
        Stream {
            bytes: ((bm + wide) * bk * ELEM) as f64,
            unique: (self.batches * spanned * self.k * ELEM) as f64,
            shared,
            reach: [Reach { share: 1.0 - late, set: near }, Reach { share: late, set: far }],
        }
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
    /// `Err` with why it is left out: it does not lower, its [live
    /// registers](GemmCfg::live_registers) pass the lane's limit, or its shape
    /// refuses it. A trip
    /// pays its fragment reads, fill, matrix-core steps (padding included),
    /// the layer epilogue's tile operations and overhead. The
    /// fill's bytes are the whole operands' distinct ones, the blocks in
    /// flight at once (one an SM at least) share the A rows and B columns of
    /// the raster's window whatever the cache holds of the whole, and the
    /// rest's re-reads come back through the working sets the raster
    /// reaches them over ([`Self::stream`]). Two
    /// crews are priced as the one-crew pipeline of their tile: the overlap
    /// of one crew's products with the other's loads has no term, since
    /// only CDNA runs them ([`Target::crews`]) and no probe measures it; the
    /// tuner times them beside the rest.
    pub fn admit(&self, planner: &Planner, cfg: &GemmCfg) -> Result<(f64, u32), Rejected> {
        planner.admit(self, cfg)
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
    /// reduction allows one, a step that divides it, a tail split that runs
    /// the problem ([`Self::tail_plan`]) with partials a call may allocate
    /// ([`partials`]) — then its resources over the whole
    /// grid, every tile data-parallel: a tail split's are its launches'
    /// ([`Self::plan_price`]), its claim the floor's and the occupancy's.
    pub(crate) fn claim(&self, planner: &Planner, cfg: &GemmCfg, regs: u32) -> Result<Claim, Rejected> {
        let whole = Part { tiles: self.tiles(cfg), split: 1 };
        let Some(tail) = cfg.tail else { return self.claim_part(planner, cfg, regs, whole) };
        let trips = self.k / cfg.tile[2];
        let runs = self.batches == 1 && self.halves == 1 && self.rope.is_none() && TailSpec::merges(cfg);
        if !runs || !trips.is_multiple_of(tail.split()) {
            return Err(ShapeRejected::Tail.into());
        }
        let claim = self.claim_part(planner, cfg, regs, whole)?;
        let plan = self.plan_of(planner, cfg, &claim).ok_or(ShapeRejected::Tail)?;
        let (need, most) = (plan.split * plan.tail.count() * cfg.tile[0] * cfg.tile[1] * F32, partials(planner));
        match need <= most {
            true => Ok(claim),
            false => Err(ShapeRejected::Partials { need, most }.into()),
        }
    }

    /// The claim of one launch of `cfg` over `part`: `part.tiles` tiles of
    /// a batch's walk each over `part.split` blocks a share of the
    /// reduction. A split part stores its raw f32 accumulator, a `[bm, bn]`
    /// slot a block, and its merge — the slots' sum — is a section beside
    /// it. A warpgroup product reads no fragments and
    /// holds one accumulator chain per instruction column block; its step
    /// is the group's.
    fn claim_part(&self, planner: &Planner, cfg: &GemmCfg, regs: u32, part: Part) -> Result<Claim, Rejected> {
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
        let blocks = self.batches * part.tiles * part.split;
        let (trips, stages) = (self.k / bk / part.split, cfg.stages());
        let fill = self.stream(planner, cfg, part);
        // The epilogue's operations over the tile's output elements, one a
        // lane's element each, paid over the trips that fold into one
        // output: their whole is m·n·[`epilogue_ops`] of lane operations
        // whatever the tile, so what the term orders is a trip's balance of
        // steps against epilogue. A rope's turn covers the `turned` share of
        // the columns (the last block straddling them half-turned on
        // average), and every block whose window reaches them reads, once,
        // its rows' positions of both tables — `half` distinct elements a
        // row each — the side stream: priced at the ladder of its own
        // working set through the epilogue's pipe, not folded into the
        // ring's fill, its distinct bytes in the run's floor. A block
        // stores its output tile and reads its bias and residual once its
        // trips are done ([`Claim::out`]); a split part runs no epilogue
        // but stores its raw f32 slot, and its merge reads every split's
        // slot of a tile row by row and adds them, then stores the row and
        // reads its residual (a merge tile's bias once), its bytes at
        // DRAM's rate: the epilogue over the sum is the plan's
        // ([`Self::plan_price`]).
        let split = part.split > 1;
        let epilogue = if split { 0.0 } else { f64::from(self.epilogue) * (bm * bn) as f64 / trips as f64 };
        let (mut ops, mut side) = (epilogue, Stream::default());
        if let Some(rope) = self.rope.filter(|_| !split) {
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
        let merge = match split {
            false => Spread::default(),
            true => Spread {
                rows: (part.tiles * bm) as f64,
                ops: Ops::of(Kind::Epilogue, (part.split * bn) as f64),
                bytes: ((part.split * F32 + self.out) * bn + bn * self.bias / TAIL_MERGE[0]) as f64,
                waits: 0.0,
            },
        };
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
            out: if split { bm * bn * F32 } else { bm * bn * self.out + bn * self.bias } as f64,
            ring: stages,
            unroll: cfg.unroll(),
            waits: 0.0,
            sections: [merge, Spread::default()],
        })
    }

    /// A batch's output tiles under `cfg`.
    fn tiles(&self, cfg: &GemmCfg) -> usize {
        let (gm, gn) = self.grid(cfg);
        gm * gn
    }

    /// The blocks of `cfg` the device runs at once on this problem
    /// ([`Self::wave_of`]); `None` where the shape refuses it.
    pub fn wave(&self, planner: &Planner, cfg: &GemmCfg) -> Option<usize> {
        Some(self.wave_of(planner, &planner.claim_of(self, cfg).ok()?))
    }

    /// The blocks the device runs at once of `claim`'s: an SM's resident
    /// blocks ([`Planner::resident`], by its shared memory, register file
    /// and wave slots) on every SM — a wave of the grid.
    fn wave_of(&self, planner: &Planner, claim: &Claim) -> usize {
        planner.sms() * planner.resident(claim.smem, claim.regs, claim.waves).max(1)
    }

    /// The plan `cfg`'s tail split runs this problem as: its grid's whole
    /// waves ([`Self::wave`]) data-parallel, the rest split; `None` for a
    /// config without one, or one the problem refuses ([`Self::claim`]).
    pub fn tail_plan(&self, planner: &Planner, cfg: &GemmCfg) -> Option<TailPlan> {
        let claim = planner.claim_of(self, cfg).ok()?;
        self.plan_of(planner, cfg, &claim)
    }

    fn plan_of(&self, planner: &Planner, cfg: &GemmCfg, claim: &Claim) -> Option<TailPlan> {
        TailPlan::new(self.tiles(cfg), self.wave_of(planner, claim), cfg.tail?)
    }

    /// What `plan` costs `cfg` (its tail split's, or any other: a whole
    /// split-K is one), ns: the whole waves' launch and the split tail's,
    /// each priced as the planner prices a run, the merge a section of the
    /// tail's, the epilogue as the data-parallel run of every tile pays it,
    /// and [`LAUNCH`] for every launch past the first. `None` where the
    /// shape refuses the config.
    ///
    /// The epilogue is priced where the data-parallel run pays it, so no
    /// plan earns from moving it: the tail's tiles pay what the
    /// data-parallel run pays for them past the waves', not a merge's
    /// spread of it under its bytes.
    pub fn plan_price(&self, planner: &Planner, cfg: &GemmCfg, plan: &TailPlan) -> Option<f64> {
        let regs = planner.claim_of(self, &GemmCfg { tail: None, ..*cfg }).ok()?.regs;
        self.priced_plan(planner, cfg, regs, plan)
    }

    /// [`Self::plan_price`] of a config whose lane holds `regs` registers.
    fn priced_plan(&self, planner: &Planner, cfg: &GemmCfg, regs: u32, plan: &TailPlan) -> Option<f64> {
        if !(self.k / cfg.tile[2]).is_multiple_of(plan.split) {
            return None;
        }
        let claim = |tiles, split| self.claim_part(planner, cfg, regs, Part { tiles, split }).ok();
        // A data-parallel run's price and the share of it its epilogue's operations make.
        let priced = |c: Claim| {
            let price = planner.price(&c);
            (price, price - planner.price(&Claim { ops: Ops::default(), ..c }))
        };
        let (_, epilogue) = priced(claim(self.tiles(cfg), 1)?);
        let (waves, waves_epilogue) = match plan.waves {
            Some(run) => priced(claim(run.count(), 1)?),
            None => (0.0, 0.0),
        };
        let tail = planner.price(&claim(plan.tail.count(), plan.split)?);
        let launches = usize::from(plan.waves.is_some()) + 2;
        Some(waves + tail + epilogue - waves_epilogue + (launches - 1) as f64 * LAUNCH)
    }

    /// The tail splits worth pricing `cfg` at on this problem: the counts up
    /// to [`TAIL_SPLITS`] that divide its trips.
    fn tails(&self, cfg: &GemmCfg) -> impl Iterator<Item = TailSplit> + use<> {
        let trips = self.k / cfg.tile[2];
        (2..=TAIL_SPLITS.min(trips)).filter(move |s| trips.is_multiple_of(*s)).filter_map(TailSplit::new)
    }

    /// The lattice's configs that run this problem on the planner's target,
    /// every pipeline (depth, ring depth, unrolling) of every tile unsplit,
    /// cheapest first (then by fewer registers) and [`spread`] over the
    /// block tiles the cost band holds: the shortlist keeps a tile's first
    /// clean one ([`shortlist`](super::search::shortlist)). A tile the model
    /// would split ([`Self::planned`]) carries one split right behind the
    /// pipeline it splits, a twin the tuner times beside that pipeline
    /// ([`Tiled::twins`]) and nothing else reaches: a split never stands
    /// for its tile ([`Tiled::represents`]), so neither the shortlist nor
    /// the untuned first pick takes one.
    pub(crate) fn ranked(&self, planner: &Planner) -> Vec<GemmCfg> {
        let feasible = planner.feasible(self);
        let planned: Vec<_> = feasible.iter().filter_map(|f| self.planned(planner, f)).collect();
        let ranked = sorted(self, planned.iter().map(|&(unsplit, _)| unsplit).collect(), feasible.len());
        // A tile's split: the cheapest of its kept pipelines' splits.
        let kept: HashSet<GemmCfg> = ranked.iter().map(|&(.., c)| c).collect();
        let mut splits: HashMap<GemmCfg, (f64, GemmCfg)> = HashMap::new();
        for &((_, _, pipeline), split) in &planned {
            let Some((cost, _, split)) = split.filter(|_| kept.contains(&pipeline)) else { continue };
            let held = splits.entry(pipeline.tile()).or_insert((cost, split));
            if cost < held.0 {
                *held = (cost, split);
            }
        }
        let twinned = ranked.into_iter().flat_map(|(cost, _, cfg)| {
            let split = splits.get(&cfg.tile()).filter(|(_, s)| GemmCfg { tail: None, ..*s } == cfg);
            [Some((cost, cfg)), split.map(|&(_, s)| (cost, s))].into_iter().flatten()
        });
        spread(twinned.collect(), |c| [c.tile[0], c.tile[1]])
    }

    /// A feasible config's cost and registers on this problem, unsplit,
    /// and its tail split the model prices cheapest ([`Self::tails`]) where
    /// that is under the unsplit price or within [`SETTLED`] of it; `None`
    /// where the shape leaves the config out. Every config walks the
    /// default raster ([`TileOrder::DEFAULT`]): the fill's model prices other
    /// group heights too flat to tell them apart, and the default runs within
    /// the tuner's drift of the best.
    ///
    /// A tail split weighs the last wave's idle SMs against its partials'
    /// bytes, their merge and its launches ([`Self::plan_price`]), and only
    /// the tuner's time settles it: the model prices a split optimistically,
    /// and the unsplit pipeline can run faster. A grid short of a wave splits whole, by the same
    /// rule.
    #[allow(clippy::type_complexity)]
    fn planned(
        &self,
        planner: &Planner,
        f: &Feasible<GemmCfg>,
    ) -> Option<((f64, u32, GemmCfg), Option<(f64, u32, GemmCfg)>)> {
        let price = |cfg: GemmCfg| {
            planner.price_of(self, &Feasible { cfg, regs: f.regs }).map(|(cost, regs)| (cost, regs, cfg))
        };
        let unsplit = price(f.cfg)?;
        let tails = self.tails(&f.cfg).map(|tail| GemmCfg { tail: Some(tail), ..f.cfg });
        let split = tails.filter_map(price).min_by(|a, b| a.0.total_cmp(&b.0));
        Some((unsplit, split.filter(|s| s.0 <= unsplit.0 * SETTLED)))
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

    /// A tail split's launches ([`GemmProblem::plan_price`]), else the claim's run.
    fn price(&self, planner: &Planner, cfg: &GemmCfg, claim: &Claim) -> f64 {
        match self.plan_of(planner, cfg, claim) {
            Some(plan) => self.priced_plan(planner, cfg, claim.regs, &plan).unwrap_or(f64::INFINITY),
            None => planner.price(claim),
        }
    }
}

impl Planner {
    pub(super) fn ranked(&self, p: GemmProblem) -> Arc<[GemmCfg]> {
        cached(&self.ranked, p, || p.ranked(self))
    }

    /// Every GEMM config for `batches` GEMMs of `m` rows, `n` columns and
    /// reduction dim `k` under `epilogue`, the cheapest of the lattice
    /// first. A rope turns in the kernel's store: only tiles that hold
    /// whole heads of its head width lower, their turn priced.
    pub fn gemm_ranked(&self, batches: usize, m: usize, n: usize, k: usize, epilogue: &Epilogue) -> Arc<[GemmCfg]> {
        if batches * m * n * k == 0 {
            return Arc::new([]);
        }
        self.ranked(GemmProblem::new(batches, m, n, k, epilogue))
    }
}
