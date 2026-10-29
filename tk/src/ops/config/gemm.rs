//! GEMM-shaped configs: the tile and warp-grid lattice, what the lowering
//! lays a tile's operands into, and what a problem's products cost on a
//! planner.

use std::sync::Arc;

use crate::atoms::{Core, Fill, Target};
use crate::ir::{Axis, Shape};
use crate::kernels::gemm::{GROUP_ROWS, GemmCfg, KSLICE, Rope};
use crate::layouts::{Issue, Swizzle, WarpGrid, chunked, fills_in_chunks};

use super::cost::{Claim, ELEM, Family, Kind, Ops, Planner, STEP, Spread, Steps, Stream, cached};
use super::search::{Rejected, sorted, spread};

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
        let row = self.core == Core::Warp || Swizzle::spans(bk, ELEM);
        // Two crews: the two warp rows of the warp-level core, one slot, and
        // k-slices of a product (one atom deep) each with another to
        // overlap, on a target that interleaves crews.
        let crews = self.crews == 1
            || (self.crews == 2
                && self.core == Core::Warp
                && target.crews >= 2
                && target.geometry().is_some_and(|atom| atom.k as usize == KSLICE)
                && self.stages == 1
                && self.warps[0] == 2
                && bk.is_multiple_of(KSLICE)
                && bk >= 2 * KSLICE);
        row && crews && self.issue(target, bk).is_some() && target.fill.runs(self.stages)
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
            Core::Warp => (issue.a.regs() + issue.b.regs() * halves).div_ceil(2),
            Core::Group => 4 * issue.a.reps[1] * halves,
        }
    }
}

pub(crate) const fn gemm_cfg(tile: [usize; 3], stages: usize, warps: [u32; 2], unroll: bool) -> GemmCfg {
    GemmCfg { core: Core::Warp, tile, stages, warps, unroll, crews: 1 }
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
                            if core == Core::Warp && target.crews >= 2 && stages == 1 && warps[0] == 2 {
                                out.push(GemmCfg { crews: 2, ..gemm_cfg([bm, bn, bk], 1, warps, false) });
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
        self.batches * self.m.div_ceil(cfg.tile[0]) * self.n.div_ceil(cfg.tile[1])
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
    /// fill's bytes are the whole operands' distinct ones,
    /// and the blocks in flight at once (one an SM at least) share a group's
    /// A rows and their B columns whatever the cache holds of the whole. Two
    /// crews are priced as the one-crew pipeline of their tile: the overlap
    /// of one crew's products with the other's loads has no term, since
    /// only CDNA runs them ([`Target::crews`]) and no CDNA card has measured
    /// one; the tuner times them beside the rest.
    pub fn admit(&self, planner: &Planner, cfg: &GemmCfg) -> Result<(f64, u32), Rejected> {
        planner.admit(self, cfg)
    }

    /// The config's [claim](super::cost::Claim): its checks run first — a
    /// step at least two matrix-core depths where the reduction allows one,
    /// a step that divides it, a config that lowers and fits the target's
    /// shared memory, registers the lane may hold — then its resources. A
    /// warpgroup product reads no fragments and holds one accumulator chain
    /// per instruction column block; its step is the group's.
    pub(crate) fn claim(&self, planner: &Planner, cfg: &GemmCfg) -> Result<Claim, Rejected> {
        let target = &planner.device.target;
        let ([bm, bn, bk], [wr, wc]) = (cfg.tile, cfg.warps.map(|w| w as usize));
        let gated = self.halves == 2;
        // A turn moves each element's partner half a head along: a tile whose
        // block or warps do not hold whole heads has partners another lane or
        // warp holds, which the flip the kernel issues cannot reach.
        if self.rope.is_some_and(|rope| !cfg.turns(rope.head_dim)) {
            return Err(Rejected::Lowers);
        }
        let atom = target.geometry().ok_or(Rejected::Lowers)?;
        // A step is at least two matrix-core depths, unless the reduction dim
        // allows nothing deeper.
        let depth = 2 * atom.k as usize;
        let smem = cfg.smem_bytes(gated, ELEM);
        if bk < depth && self.k.is_multiple_of(depth) {
            return Err(Rejected::Shallow);
        }
        if !self.k.is_multiple_of(bk) {
            return Err(Rejected::Divides);
        }
        if !cfg.fits(target) {
            return Err(Rejected::Lowers);
        }
        if smem > target.smem_bytes {
            return Err(Rejected::Smem { need: smem, have: target.smem_bytes });
        }
        let regs = cfg.registers(target, self.halves).ok_or(Rejected::Layout)?;
        let (halves, waves) = (self.halves, wr * wc);
        let (acc, live) = cfg.live_registers(target, self.halves).ok_or(Rejected::Layout)?;
        let budget = target.lane_budget(waves as u32);
        if !target.occupancy.holds(acc, live) || acc + live > budget {
            return Err(Rejected::Registers { need: acc + live, have: budget });
        }
        // A warpgroup product reads no fragments and holds one accumulator
        // chain per instruction column block; its step is the group's.
        let (atoms, fragments) = match cfg.core {
            Core::Warp => (
                (bm / wr / atom.m as usize) * (bn * halves / wc / atom.n as usize),
                waves * (bm / wr + bn * halves / wc) * bk * ELEM,
            ),
            Core::Group => {
                let group = target.group_mma.first().ok_or(Rejected::Lowers)?;
                let n = group.n_for((bn / wc) as u32).ok_or(Rejected::Lowers)? as usize;
                (bn * halves / wc / n, 0)
            }
        };
        let (fill, steps) = ((bm + bn * halves) * bk * ELEM, (bm * bn * halves * bk / STEP) as f64);
        let (blocks, trips, stages) = (self.blocks(cfg), self.k / bk, cfg.stages);
        let unique = (self.batches * (self.m + self.n * halves) * self.k * ELEM) as f64;
        let flight = planner.sms().min(blocks).max(1);
        let group = GROUP_ROWS.min(flight);
        let distinct = group * bm + flight.div_ceil(group) * bn * halves;
        let shared = 1.0 - distinct as f64 / (flight * (bm + bn * halves)) as f64;
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
                shared,
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
                Core::Group => Steps::Group(steps),
            },
            trips,
            reads: fragments as f64,
            ops: Ops::of(Kind::Epilogue, ops),
            fill: Stream { bytes: fill as f64, unique, shared },
            side,
            stores: 0.0,
            ring: stages,
            unroll: cfg.unroll,
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
        let lattice = self.lattice(&planner.device.target);
        let count = lattice.len();
        let admitted: Vec<(f64, u32, GemmCfg)> = lattice
            .into_iter()
            .filter_map(|c| self.admit(planner, &c).ok().map(|(cost, regs)| (cost, regs, c)))
            .collect();
        let ranked = sorted(self, admitted, count);
        spread(ranked.into_iter().map(|(cost, _, c)| (cost, c)).collect(), |c| [c.tile[0], c.tile[1]])
    }
}

impl Family for GemmProblem {
    type Cfg = GemmCfg;

    fn lattice(&self, target: &Target) -> Vec<GemmCfg> {
        gemm_lattice(target)
    }

    fn claim(&self, planner: &Planner, cfg: &GemmCfg) -> Result<Claim, Rejected> {
        GemmProblem::claim(self, planner, cfg)
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
