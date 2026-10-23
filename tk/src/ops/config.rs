//! Tile config candidates per target and shape, most promising first. The
//! device picks: the tune store times the first [`SHORTLIST`] tiles that
//! compile clean, most of the lattice, and the ranking orders what is
//! compiled and timed first and stands in for a measurement where there is
//! no device (the first config runs untuned). GEMM-shaped kernels draw
//! theirs from a lattice over the construct's knobs, kept to what the
//! lowering runs and ranked in nanoseconds of SM time from the device's
//! measured [`Costs`]: a round of resident blocks is bound by the SM's issue
//! work or by one wave's own issue, plus the memory latency one pipeline
//! stage ahead cannot hide ([`Planner::pipelined`]). The model orders block
//! tiles well enough to lead with them ([`spread`]) and warp grids less well
//! (on sm_86 the measured best sat as deep as the 92nd of 109 tiles before
//! a trip's pipes overlapped across a SIMD's waves), which is why the
//! shortlist runs that deep. Registers admit nothing here: the
//! layout count over-counts, so the compiled kernel judges ([`shortlist`]).
//! A [`Planner`] holds the rankings of one device. The other kernels' lists
//! are small functions of the target.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use svod_dtype::{GpuArch, ScalarDType};

use crate::atoms::{Core, Device, Fill, GroupMmaAtom, Target};
use crate::ir::{Axis, Orient, Shape};
use crate::kernels::attention::{FaCfg, WARP_ROWS};
use crate::kernels::conv::{ConvCfg, ConvGeom};
use crate::kernels::gemm::{GROUP_ROWS, GemmCfg, KSLICE};
use crate::kernels::nearest::{Floors, LANES, LaneQueue, Metric, NearestCfg, SelectCfg, select_runs};
use crate::kernels::rows::NormCfg;
use crate::layouts::{Issue, Swizzle, TileLayout, WarpGrid, chunked, fills_in_chunks};
use crate::tune::Compiled;

/// Targets with kernels: CUDA sm_80+ on `mma.sync` + `cp.async` + `ldmatrix`,
/// sm_90 adding the `wgmma` warpgroup core (no TMA or mbarrier), and AMD
/// matrix cores with register-staged fills: RDNA3/3.5/4 WMMA and CDNA3/4
/// MFMA 16×16×16 on wave64. Metal has atoms but no kernels.
pub fn has_kernels(target: &Target) -> bool {
    match target.arch {
        GpuArch::Cuda(c) => c.major >= 8,
        GpuArch::Amd(a) => a.has_matrix_cores(),
        GpuArch::Metal(_) => false,
    }
}

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

/// Search bound: the output tile's rows, the first of the knobs a GEMM-shaped
/// kernel is enumerated over.
const ROWS: [usize; 4] = [32, 64, 128, 256];
/// Search bound: the output tile's columns (48 and 96 for the convolution
/// bodies a power of two would pad by a third).
const COLS: [usize; 5] = [48, 64, 96, 128, 256];
/// Search bound: the reduction depths of a step.
const DEPTHS: [usize; 3] = [16, 32, 64];
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

/// Why a ranking leaves a config out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejected {
    /// It does not lower on the target: no matrix core, a warp grid that
    /// does not tile the core's atoms, a fill that does not split into whole
    /// 16-byte chunks per thread ([`fills_in_chunks`]), a ring depth the
    /// fill does not run ([`Fill::runs`]), lists or key warps the kernel
    /// refuses ([`NearestCfg::lists`], [`FaCfg::shares_keys`]), or lane
    /// queues past the search's bounds.
    Lowers,
    /// Its step is shallower than two matrix-core depths where the
    /// reduction allows one.
    Shallow,
    /// Its step does not divide the reduction (a nearest chunk: is deeper
    /// than the whole reduction, a product over zeros).
    Divides,
    /// Its shared memory passes the block's.
    Smem { need: usize, have: usize },
    /// Its tiles have no register layout.
    Layout,
    /// The registers a lane holds at once pass what it may.
    Registers { need: u32, have: u32 },
    /// It keeps more least scores a row than a register list holds.
    TopK { k: usize, most: usize },
}

/// A ranking: its configs with their costs ([`Planner::pipelined`] ns of SM
/// time) and registers by the layout count, cheapest first (then fewer
/// registers) and [pruned](prune), logged under [`crate::ops::PLAN`] with how
/// many of the `spanned` configs it left out.
fn sorted<C: Tiled + Debug>(
    problem: &impl Debug,
    mut ranked: Vec<(f64, u32, C)>,
    spanned: usize,
) -> Vec<(f64, u32, C)> {
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let admitted = ranked.len();
    prune(&mut ranked);
    let (pruned, rejected) = (admitted - ranked.len(), spanned - admitted);
    tracing::debug!(target: crate::ops::PLAN, ?problem, admitted, rejected, pruned, "ranking");
    for (rank, (ns, regs, cfg)) in ranked.iter().take(LOGGED).enumerate() {
        tracing::debug!(target: crate::ops::PLAN, rank, ns, regs, ?cfg, "ranked");
    }
    ranked
}

/// Drops the configs of `ranked` (cheapest first) past [`PRUNED_PAST`] times
/// the cheapest one's cost once [`KEPT_TILES`] tiles are kept.
pub(crate) fn prune<C: Tiled>(ranked: &mut Vec<(f64, u32, C)>) {
    let Some(&(cheapest, ..)) = ranked.first() else { return };
    let mut tiles = HashSet::new();
    ranked.retain(|(cost, _, c)| {
        let keep = *cost <= PRUNED_PAST * cheapest || tiles.len() < KEPT_TILES;
        if keep {
            tiles.insert(c.tile());
        }
        keep
    });
}

/// Search bound: the configs of a ranking logged with their costs.
const LOGGED: usize = 16;
/// Search bound: a config this factor past the cheapest one's cost cannot
/// win. Over the shortlists sm_86 timed (GEMMs from 1024³ to 8192²×768,
/// attention, nearest from 8192² to 16384² at depths 128–1024) the measured
/// best was predicted within 1.19 of the cheapest, every config within the
/// tuner's contender band ([`crate::tune::CONTENDER`]) within 1.63. On
/// gfx1201 and gfx1151 (nearest at 16384², depths 256–1024, a 4096³ GEMM,
/// attention) the best was predicted within 1.09, though the model errs
/// wider there: a pruned config ran 1.23 times the best on gfx1201 and 1.08
/// on gfx1151, never within the tuner's drift. Nearest's small candidate
/// blocks run 3.5 to 6 times its best, seconds a launch at 65536² rows.
pub(crate) const PRUNED_PAST: f64 = 2.5;
/// Search bound: the tiles a ranking keeps whatever their cost, so a shortlist
/// whose cheapest tiles fail to compile clean still holds some to time.
pub(crate) const KEPT_TILES: usize = 16;

/// How a problem runs under one config: `blocks` blocks of `waves` waves,
/// each holding `smem` bytes of shared memory and `regs` registers a lane
/// by the layout count, and running `trips` trips through a ring `stages`
/// deep. A trip's issue goes down three pipes, in ns of SM time each, plus
/// a fixed overhead `trip` (its barrier and ring bubble, which the other
/// blocks an SM holds work through).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Run {
    pub blocks: usize,
    pub smem: usize,
    pub regs: u32,
    pub waves: usize,
    /// The independent matrix-core accumulators a wave holds between
    /// dependent steps: they hide a chain's latency as other waves do.
    pub atoms: usize,
    pub trips: usize,
    /// A trip's operand fragments read from shared memory.
    pub reads: f64,
    /// A trip's matrix-core steps and vector work.
    pub work: f64,
    /// A trip's fill, priced by where its bytes come from.
    pub fill: f64,
    pub trip: f64,
    pub stages: usize,
}

/// A GEMM-shaped problem: `batches` products of `m × k` by `k × n·halves`
/// (two halves: a gated weight, whose block stages a `2·bn`-wide B tile
/// and holds two accumulators).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemmProblem {
    pub batches: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub halves: usize,
}

impl GemmProblem {
    fn blocks(&self, cfg: &GemmCfg) -> usize {
        self.batches * self.m.div_ceil(cfg.tile[0]) * self.n.div_ceil(cfg.tile[1])
    }

    /// The share of the tiles' area past `m × n`.
    fn padding(&self, cfg: &GemmCfg) -> f64 {
        let [bm, bn, _] = cfg.tile;
        let area = self.m.div_ceil(bm) * bm * self.n.div_ceil(bn) * bn;
        (area - self.m * self.n) as f64 / area as f64
    }

    /// What `cfg` costs on the planner's device ([`Planner::pipelined`] ns of
    /// SM time) and a lane's registers by the layout count, which over-counts
    /// what the lowering streams (the compiled kernel judges, [`shortlist`]);
    /// `None` when it does not lower, or when its [live
    /// registers](GemmCfg::live_registers) alone pass the lane's limit; why it
    /// is left out otherwise. A trip
    /// pays its fragment reads, fill, matrix-core steps (padding included)
    /// and overhead. The fill's bytes are the whole operands' distinct ones,
    /// and the blocks in flight at once (one an SM at least) share a group's
    /// A rows and their B columns whatever the cache holds of the whole. Two
    /// crews are priced as the one-crew pipeline of their tile: the overlap
    /// of one crew's products with the other's loads has no term, since
    /// only CDNA runs them ([`Target::crews`]) and no CDNA card has measured
    /// one; the tuner times them beside the rest.
    pub fn admit(&self, planner: &Planner, cfg: &GemmCfg) -> Result<(f64, u32), Rejected> {
        let Device { target, costs } = &planner.device;
        let ([bm, bn, bk], [wr, wc]) = (cfg.tile, cfg.warps.map(|w| w as usize));
        let gated = self.halves == 2;
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
        let (atoms, fragments, step) = match cfg.core {
            Core::Warp => (
                (bm / wr / atom.m as usize) * (bn * halves / wc / atom.n as usize),
                waves * (bm / wr + bn * halves / wc) * bk * ELEM,
                costs.step,
            ),
            Core::Group => {
                let group = target.group_mma.first().ok_or(Rejected::Lowers)?;
                let n = group.n_for((bn / wc) as u32).ok_or(Rejected::Lowers)? as usize;
                (bn * halves / wc / n, 0, costs.step_group)
            }
        };
        let fill = (bm + bn * halves) * bk * ELEM;
        let steps = bm * bn * halves * bk / STEP;
        let (blocks, trips, stages) = (self.blocks(cfg), self.k / bk, cfg.stages);
        let unique = (self.batches * (self.m + self.n * halves) * self.k * ELEM) as f64;
        let flight = planner.sms().min(blocks).max(1);
        let group = GROUP_ROWS.min(flight);
        let distinct = group * bm + flight.div_ceil(group) * bn * halves;
        let shared = 1.0 - distinct as f64 / (flight * (bm + bn * halves)) as f64;
        let fill = fill as f64 * planner.fill_byte(unique, (blocks * trips * fill) as f64, shared);
        let reads = fragments as f64 * costs.smem_byte;
        let work = steps as f64 * step;
        let trip = costs.trip(target.fill, stages, cfg.unroll);
        let run = Run { blocks, smem, regs, waves, atoms, trips, reads, work, fill, trip, stages };
        Ok((planner.pipelined(run), regs))
    }

    /// The lattice's configs that run this problem on the planner's target,
    /// every pipeline (depth, ring depth, unrolling) of every tile, cheapest
    /// first (then by fewer registers) and [`spread`] over the block tiles
    /// the cost band holds: the shortlist keeps a tile's first clean one
    /// ([`shortlist`]).
    pub(crate) fn ranked(&self, planner: &Planner) -> Vec<GemmCfg> {
        let lattice = gemm_lattice(&planner.device.target);
        let count = lattice.len();
        let admitted: Vec<(f64, u32, GemmCfg)> = lattice
            .into_iter()
            .filter_map(|c| self.admit(planner, &c).ok().map(|(cost, regs)| (cost, regs, c)))
            .collect();
        let ranked = sorted(self, admitted, count);
        spread(ranked.into_iter().map(|(cost, _, c)| (cost, c)).collect(), |c| [c.tile[0], c.tile[1]])
    }
}

/// Model: the elements of one 16×16×16 matrix-core step, [`Costs::step`]'s
/// unit.
const STEP: usize = 16 * 16 * 16;
/// The bytes of a kernel's 16-bit operand element.
const ELEM: usize = 2;

/// Model: how far past the cheapest config's cost the model cannot tell
/// configs apart: on sm_86 the measured best of Nemotron's projections and of 4096³
/// sat 12–31% behind the model's first.
const BAND: f64 = 1.35;
/// Model: the share of the cache in front of DRAM a fill's distinct bytes
/// may take and stay resident, the rest left to the other operands and the
/// outputs streaming through it.
const CACHE_SHARE: f64 = 0.75;
/// Model: distinct bytes past this multiple of the resident share all come
/// from DRAM: an LRU cache streamed past its size by half holds nothing
/// that is read again.
const CACHE_SPILL: f64 = 1.5;
/// Model: the waves per SIMD past which a trip's pipes overlap no further
/// (measured on gfx1201 and sm_86: sixteen waves per SIMD run a tile no
/// better than four, and the ranking that counted them did put sixteen-warp
/// blocks first on sm_86 where eight-warp ones run 7% faster).
const OVERLAP_WAVES: f64 = 4.0;

/// `ranked` (cheapest first, with its costs) led by the band's classes in
/// turn: each class's cheapest tile (classes in the order of their
/// cheapest), then each one's next, and so on, a tile's pipelines (ring
/// depth, unrolling) after it; then the rest as ranked. The model orders
/// the tiles of a class (a block tile's warp grids and depths) better than
/// it orders classes, so a shortlist from the front times every class the
/// band holds rather than one class's variants.
pub(crate) fn spread<C: Tiled, K: Eq + Hash>(ranked: Vec<(f64, C)>, class: impl Fn(&C) -> K) -> Vec<C> {
    let Some(&(best, _)) = ranked.first() else { return vec![] };
    let mut classes: Vec<(K, Vec<C>)> = vec![];
    let mut tiles = HashSet::new();
    for &(_, c) in ranked.iter().take_while(|(cost, _)| *cost <= BAND * best) {
        if !tiles.insert(c.tile()) {
            continue;
        }
        let key = class(&c);
        match classes.iter_mut().find(|(k, _)| *k == key) {
            Some((_, list)) => list.push(c),
            None => classes.push((key, vec![c])),
        }
    }
    let depth = classes.iter().map(|(_, list)| list.len()).max().unwrap_or(0);
    let mut out: Vec<C> = Vec::with_capacity(ranked.len());
    let mut placed = HashSet::new();
    for round in 0..depth {
        for (_, list) in &classes {
            let Some(&lead) = list.get(round) else { continue };
            // The tile's pipelines follow it, in ranked order.
            for &(_, c) in &ranked {
                if c.tile() == lead.tile() && placed.insert(c) {
                    out.push(c);
                }
            }
        }
    }
    out.extend(ranked.into_iter().map(|(_, c)| c).filter(|c| placed.insert(*c)));
    out
}

/// Search bound: the tiles the tune store measures per shape, their
/// pipelines compiled in parallel ([`Search`]): most of the lattice, since timing a tile costs
/// about 20 ms once the device's clock is up and the ranking cannot be
/// trusted to put the best within a few (see the module docs).
pub(crate) const SHORTLIST: usize = 64;
/// Search bound: the most tiles of a timing whose other pipelines are
/// timed after it, among those within the tuner's contender band
/// ([`crate::tune::CONTENDER`]) of the leader: a tile enters the shortlist
/// as its first clean pipeline, on gfx1201 the 4096³ GEMM's measured best
/// entered 8 % behind its best and its standing drifts between fourth and
/// tenth across runs while the top ten sit within 10 %, so a fixed depth
/// cut it on some runs. Over three cards' exhaustive sweeps eight reaches
/// every tile a deeper stage would.
pub(crate) const TWIN_TILES: usize = 8;
/// Search bound: half the blocks per SM a single-batch convolution's grid
/// must reach before its reduction is left whole ([`Planner::conv_splits`]).
const CONV_BLOCKS_PER_SM: usize = 2;
/// Model: the share of a convolution tile's area that may pad past `m × n`;
/// a split is sought among tiles within half of it as well.
const CONV_PADDING: f64 = 1.0 / 8.0;
/// Search bound: the counts a starved convolution's reduction splits into.
const CONV_SPLITS: [usize; 5] = [2, 3, 4, 6, 9];
/// Search bound: the most blocks per SM a split convolution grid reaches.
const SPLIT_BLOCKS_PER_SM: usize = 16;

/// A config's tile: all of it but its pipeline (ring depth, unrolling),
/// which the ranking orders by a cost model alone and the compiler may
/// refuse, so the shortlist holds one config per tile, its first clean
/// pipeline ([`shortlist`]); the tile that wins has its other pipelines,
/// its [`twins`](Self::twins), timed after.
pub(crate) trait Tiled: Copy + Eq + Hash + Send {
    fn tile(&self) -> Self;

    /// The configs of `ranked` sharing this tile, each under this config's
    /// variant (a split count the tuner added): every pipeline of the tile.
    fn twins(&self, ranked: &[Self]) -> Vec<Self>;

    /// Whether the config may stand for its tile in the shortlist and the
    /// first clean search; one that may not (a variant the twins stage
    /// explores) is timed only as a twin of a contender tile.
    fn represents(&self) -> bool {
        true
    }
}

/// The configs of `ranked` that may stand for their tiles, in order.
pub(crate) fn representatives<C: Tiled>(ranked: &[C]) -> Vec<C> {
    ranked.iter().copied().filter(Tiled::represents).collect()
}

impl Tiled for GemmCfg {
    fn tile(&self) -> Self {
        GemmCfg { tile: [self.tile[0], self.tile[1], 0], stages: 0, unroll: false, ..*self }
    }

    fn twins(&self, ranked: &[Self]) -> Vec<Self> {
        ranked.iter().filter(|c| c.tile() == self.tile()).copied().collect()
    }
}

impl Tiled for ConvCfg {
    fn tile(&self) -> Self {
        ConvCfg { gemm: self.gemm.tile(), ..*self }
    }

    /// A split's twins keep its depth as well as its split: a partial
    /// kernel's split must divide its trips, and a tile's depths divide
    /// differently (3456 taps are 108 trips at a depth of 32, 54 at 64).
    fn twins(&self, ranked: &[Self]) -> Vec<Self> {
        self.gemm
            .twins(&ranked.iter().map(|c| c.gemm).collect::<Vec<_>>())
            .into_iter()
            .filter(|gemm| self.split == 1 || gemm.tile[2] == self.gemm.tile[2])
            .map(|gemm| ConvCfg { gemm, ..*self })
            .collect()
    }
}

/// The blocks, the splits and the softmax forms: a forms pair is two tiles.
/// A tile is its query and key block sizes and its key warps: a different
/// number of key warps is a different kernel structure, not a pipeline of
/// the same one. Its rows per warp are a variant its twins carry, as its
/// pipelines are: the one-row config represents the tile.
impl Tiled for FaCfg {
    fn tile(&self) -> Self {
        FaCfg { stages: 0, unroll: false, warp_rows: WARP_ROWS, ..*self }
    }

    fn twins(&self, ranked: &[Self]) -> Vec<Self> {
        let block = |c: &FaCfg| (c.bq, c.bkv, c.key_warps, c.core);
        ranked
            .iter()
            .filter(|c| block(c) == block(self))
            .map(|c| FaCfg { stages: c.stages, unroll: c.unroll, warp_rows: c.warp_rows, ..*self })
            .collect()
    }

    /// Two atom rows per warp are a variant of the one-row tile.
    fn represents(&self) -> bool {
        self.warp_rows == WARP_ROWS
    }
}

impl Tiled for NormCfg {
    fn tile(&self) -> Self {
        *self
    }

    fn twins(&self, _: &[Self]) -> Vec<Self> {
        vec![*self]
    }
}

/// The first config of each tile of `ranked`, in ranked order.
pub(crate) fn distinct<C: Tiled>(ranked: &[C]) -> impl Iterator<Item = C> + '_ {
    let mut seen = HashSet::new();
    ranked.iter().copied().filter(move |c| seen.insert(c.tile()))
}

pub(crate) fn distinct_configs<C: Tiled>(list: &[C]) -> Vec<C> {
    let mut seen = HashSet::new();
    list.iter().copied().filter(|c| seen.insert(*c)).collect()
}

/// The indices of the first [`SHORTLIST`] configs of `ranked` whose
/// compiled kernels do not spill, one per tile (its first clean pipeline,
/// so a spilling unrolled ring yields to its rolled twin), in ranked order;
/// after them, within the [`SHORTLIST`], the tiles none of whose pipelines
/// compiled clean, each by its first spilling one: the compiled kernel
/// judges what it can and the time the rest (a spilling kernel still beats
/// the graph's, and on gfx1201 the 16384² depth-768 nearest's best tile,
/// 128 rows by 128 candidates at 32 rows a warp, spills 196 bytes a lane
/// and runs 86.4 Gpairs/s to the clean pick's 84.7).
/// `compiles` is handed what the [`Search`] wants as soon as it wants it
/// (the tuner compiles in parallel, in the order handed), each verdict
/// taken as it lands, until nothing more is wanted. `None` when nothing
/// compiled: no device to judge on.
pub(crate) fn shortlist<C: Tiled>(ranked: &[C], compiles: &mut impl Compiles<C>) -> Option<Vec<usize>> {
    let mut search = Search::new(ranked, SHORTLIST);
    loop {
        while let Some(at) = search.next() {
            compiles.start(at, ranked[at]);
        }
        while compiles.idle()
            && let Some(at) = search.spare()
        {
            compiles.start(at, ranked[at]);
        }
        let Some((at, verdict)) = compiles.finished() else { return search.judged() };
        search.record(at, verdict);
    }
}

/// Compiles started by a ranked index, their verdicts back in any order.
pub(crate) trait Compiles<C> {
    fn start(&mut self, at: usize, config: C);
    /// Whether a compile started now would run at once, beside nothing queued.
    fn idle(&self) -> bool;
    /// A compile that finished, waiting for one; `None` when none is in flight.
    fn finished(&mut self) -> Option<(usize, Compiled)>;
}

/// The shortlist's search of a ranked list: it compiles what a walk in
/// ranked order compiling one config at a time compiles, so the verdicts
/// alone decide, never the order they come back in. That walk stops at the
/// `limit`th tile found clean and skips a config once a pipeline of its
/// tile ahead of it is clean; here a config is wanted ([`Self::next`]) once
/// every pipeline of its tile ahead of it is judged unclean and fewer than
/// `limit` tiles ahead of it are clean or have a compile in flight. Where
/// the compiler has room, the next pipeline of a tile whose pipeline ahead
/// is still compiling may start early ([`Self::spare`]): a slow compile
/// (one killed at its deadline) holds no chain of its tile's pipelines up
/// one after another, and the walk ignores what it would not have asked.
struct Search {
    /// Each config's tile, numbered in order of first appearance.
    tiles: Vec<usize>,
    limit: usize,
    verdicts: Vec<Option<Compiled>>,
    asked: Vec<bool>,
}

impl Search {
    fn new<C: Tiled>(ranked: &[C], limit: usize) -> Self {
        let mut ids = HashMap::new();
        let tiles = ranked
            .iter()
            .map(|c| {
                let next = ids.len();
                *ids.entry(c.tile()).or_insert(next)
            })
            .collect();
        Self { tiles, limit, verdicts: vec![None; ranked.len()], asked: vec![false; ranked.len()] }
    }

    /// The next config wanted, marked asked; `None` while the rest wait on
    /// verdicts in flight, and once nothing more is wanted.
    fn next(&mut self) -> Option<usize> {
        let (mut clean, mut open) = (HashSet::new(), HashSet::new());
        for (at, &tile) in self.tiles.iter().enumerate() {
            if clean.len() + open.len() >= self.limit {
                return None;
            }
            if clean.contains(&tile) || open.contains(&tile) {
                continue;
            }
            match self.verdicts[at] {
                Some(Compiled::Clean) => _ = clean.insert(tile),
                Some(_) => {}
                None if self.asked[at] => _ = open.insert(tile),
                None => {
                    self.asked[at] = true;
                    return Some(at);
                }
            }
        }
        None
    }

    /// A pipeline the walk may yet want, marked asked: the first not asked
    /// of a tile whose pipeline ahead of it is compiling, within the tiles
    /// [`Self::next`] reaches. Its verdict counts only where the walk asks
    /// for it: a clean pipeline ahead of it takes its tile first, and past
    /// the `limit`th clean tile the shortlist ends.
    fn spare(&mut self) -> Option<usize> {
        let (mut clean, mut open) = (HashSet::new(), HashSet::new());
        for (at, &tile) in self.tiles.iter().enumerate() {
            if clean.len() + open.len() >= self.limit {
                return None;
            }
            if clean.contains(&tile) {
                continue;
            }
            if open.contains(&tile) {
                if self.verdicts[at].is_none() && !self.asked[at] {
                    self.asked[at] = true;
                    return Some(at);
                }
                continue;
            }
            match self.verdicts[at] {
                Some(Compiled::Clean) => _ = clean.insert(tile),
                None if self.asked[at] => _ = open.insert(tile),
                _ => {}
            }
        }
        None
    }

    fn record(&mut self, at: usize, verdict: Compiled) {
        self.verdicts[at] = Some(verdict);
    }

    /// The shortlist the verdicts make ([`shortlist`]): the clean tiles,
    /// then the spilling ones, a pipeline each.
    fn judged(&self) -> Option<Vec<usize>> {
        let mut tiles = HashSet::new();
        let at = |want| self.verdicts.iter().enumerate().filter(move |(_, v)| **v == Some(want)).map(|(i, _)| i);
        let first = |i: &usize| tiles.insert(self.tiles[*i]);
        let out: Vec<usize> = at(Compiled::Clean).chain(at(Compiled::Spilled)).filter(first).take(self.limit).collect();
        (!out.is_empty()).then_some(out)
    }
}

/// The indices of every config of `list` that compiles, spilling or not,
/// in order; `None` when nothing does. A tile's [twins](Tiled::twins) are
/// judged so: the shortlist set its spilling pipelines aside for a clean
/// one, but among a tile's pipelines the time alone decides, and an
/// unrolled ring that spills a little may be the fastest.
pub(crate) fn compiled<C: Tiled>(list: &[C], mut compile: impl FnMut(&[C]) -> Vec<Compiled>) -> Option<Vec<usize>> {
    let mut out = vec![];
    for (n, chunk) in list.chunks(SHORTLIST).enumerate() {
        let built = compile(chunk).into_iter().enumerate().filter(|(_, c)| *c != Compiled::Failed);
        out.extend(built.map(|(i, _)| n * SHORTLIST + i));
    }
    (!out.is_empty()).then_some(out)
}

/// Search bound: configs compiled at once in search of the first clean one: the first
/// ranked config nearly always is, and a search is what runs with tuning
/// off, where no test should pay for a lattice.
pub(crate) const FIRST: usize = 8;

/// The index of the first config of `ranked` whose compiled kernel does
/// not spill, judged as [`shortlist`] judges (every one spilling: the first
/// that compiles; nothing compiling: `None`), `compile` asked about batches
/// of [`FIRST`] in ranked order until one passes.
pub(crate) fn first_clean<C: Tiled>(ranked: &[C], mut compile: impl FnMut(&[C]) -> Vec<Compiled>) -> Option<usize> {
    let mut spilled = None;
    for (n, chunk) in ranked.chunks(FIRST).enumerate() {
        for (i, compiled) in compile(chunk).into_iter().enumerate() {
            match compiled {
                Compiled::Clean => return Some(n * FIRST + i),
                Compiled::Spilled => _ = spilled.get_or_insert(n * FIRST + i),
                Compiled::Failed => {}
            }
        }
    }
    spilled
}

/// What the op layer plans with on one device: its target, what its
/// primitives cost, and the problems ranked for it so far. An op plans its
/// shape again at every call, and laying out the lattice's feasible configs
/// takes a millisecond, so a ranking is kept for the planner's life (one per
/// device). The rankings are whole: the tuner admits the first
/// [`SHORTLIST`] whose compiled kernels do not spill ([`shortlist`]).
/// A planner needs no device: one over a card's offline target
/// ([`Target::offline`]) and the costs its device stored
/// ([`crate::probe::stored`]) ranks as the card's own would; only the op
/// layer's ([`crate::ops::supported`]) launches.
#[derive(Debug)]
pub struct Planner {
    pub device: Device,
    /// The device's SM count.
    sms: usize,
    /// The cost fields the device's probes could not measure, ranked at
    /// zero ([`crate::probe::Measured`]); empty for a device that measured.
    pub unmeasured: Vec<&'static str>,
    ranked: papaya::HashMap<GemmProblem, Arc<[GemmCfg]>>,
    attention: papaya::HashMap<AttentionProblem, Arc<[FaCfg]>>,
    nearest: papaya::HashMap<NearestProblem, Arc<[NearestCfg]>>,
}

/// The ranking kept under `key`, made by `rank` the first time it is asked for.
fn cached<K: std::hash::Hash + Eq + Clone, V>(
    map: &papaya::HashMap<K, Arc<[V]>>,
    key: K,
    rank: impl FnOnce() -> Vec<V>,
) -> Arc<[V]> {
    map.pin().get_or_insert_with(key, || rank().into()).clone()
}

impl Planner {
    /// A planner of `device`; `None` where its target reports no SM count,
    /// which every ranking scales by.
    pub fn new(device: Device) -> Option<Self> {
        let sms = device.target.sms? as usize;
        Some(Self {
            device,
            sms,
            unmeasured: vec![],
            ranked: papaya::HashMap::new(),
            attention: papaya::HashMap::new(),
            nearest: papaya::HashMap::new(),
        })
    }

    /// A planner over what the device's probes measured, as [`Self::new`].
    pub fn measured(target: Target, measured: crate::probe::Measured) -> Option<Self> {
        let crate::probe::Measured { costs, unmeasured } = measured;
        Some(Self { unmeasured, ..Self::new(Device { target, costs })? })
    }

    /// The device's SM count.
    pub(crate) fn sms(&self) -> usize {
        self.sms
    }

    /// The SM time of `run`. The busiest SM issues the work of
    /// `⌈blocks / sms⌉` blocks, each taken up as another ends (measured on
    /// sm_86: a 64×64 GEMM's time steps with that count, not with whole
    /// rounds of resident blocks), every block-trip paying a trip's fixed
    /// overhead at the blocks the SM runs at once ([`crate::atoms::Costs::trip_at`]: the
    /// trip probes measure it per block-trip, alone and with the SM full).
    /// A trip's pipes (the fragment reads, the matrix-core steps, the fill)
    /// overlap across the waves a SIMD holds: the longest is paid whole, the
    /// rest in a share of one over the waves per SIMD up to
    /// [`OVERLAP_WAVES`], the blocks the SM holds at once ([`Self::resident`])
    /// pooling theirs (measured on gfx1201 and sm_86: a block at one wave per
    /// SIMD pays about their sum, at four or more the longest); a one-slot
    /// `cp.async` ring copies, waits and only then computes, so a block alone
    /// on its SM overlaps nothing with its fill. A SIMD holding fewer waves
    /// than hide a matrix-core chain's latency (`step_lone / step` of them, a
    /// wave's independent accumulators counting as waves) runs the steps that
    /// many times slower. A round of blocks is at least one wave's own issue
    /// (a block of fewer waves than SIMDs leaves the rest idle), and every
    /// round of every trip pays the memory latency the ring cannot hide (it
    /// prefetches `stages - 1` trips ahead; a one-slot `cp.async` ring none,
    /// [`crate::atoms::Costs::latency_one`]) on top of its issue: measured on
    /// both, the latency adds rather than hides under the issue.
    pub(crate) fn pipelined(&self, run: Run) -> f64 {
        let Run { blocks, smem, regs, waves, atoms, trips, reads, work, fill, trip, stages } = run;
        let Device { target, costs } = &self.device;
        let (o, sms) = (target.occupancy, self.sms());
        // At least one: the compiled kernel, not the layout count, decides
        // whether a block runs.
        let resident = self.resident(smem, regs, waves).max(1);
        let per_sm = blocks.div_ceil(sms);
        let rounds = per_sm.div_ceil(resident);
        let hide =
            if costs.step > 0.0 && costs.step_lone > 0.0 { (costs.step_lone / costs.step).max(1.0) } else { 1.0 };
        let per_simd = |waves: usize| (waves as f64 / o.simds as f64).max(1.0);
        let lone = |waves: usize| (hide / (per_simd(waves) * atoms.max(1) as f64)).max(1.0);
        let serial = stages <= 1 && target.fill == Fill::CpAsync;
        // `alone`: one block's pipes, its fill after the rest where serial.
        let body = |waves: usize, alone: bool| {
            let fill = if alone && serial { (0.0, fill) } else { (fill, 0.0) };
            let pipes = [reads, work * lone(waves), fill.0];
            let longest = pipes.into_iter().fold(0.0, f64::max);
            longest + (pipes.iter().sum::<f64>() - longest) / per_simd(waves).min(OVERLAP_WAVES) + fill.1
        };
        let at_once = resident.min(per_sm);
        let issue = per_sm as f64 * (body(at_once * waves, at_once == 1) + costs.trip_at(stages, trip, at_once));
        let wave = o.simds as f64 * (body(waves, true) + trip) / waves as f64;
        // A register-staged slot loads a trip ahead as two do; a store
        // probed before `latency_one` has none (zero): a one-slot ring waits
        // at least what a ring one trip ahead leaves.
        let wait = match stages {
            ..=1 if serial => costs.latency_one.max(costs.latency),
            s => costs.latency / s.saturating_sub(1).max(1) as f64,
        };
        trips as f64 * (issue.max(rounds as f64 * wave) + rounds as f64 * wait)
    }

    /// The blocks of `waves` waves an SM holds at once, each holding `smem`
    /// bytes of shared memory and `regs` registers a lane: by its shared
    /// memory and by its SIMDs' register files and wave slots; none where a
    /// block's registers by the layout count pass the SM's.
    pub(crate) fn resident(&self, smem: usize, regs: u32, waves: usize) -> usize {
        let (target, o) = (&self.device.target, self.device.target.occupancy);
        let per_simd = o.waves.min(o.file / regs.max(1)) as usize;
        (target.smem_bytes / smem.max(1)).min(o.simds as usize * per_simd / waves.max(1))
    }

    /// What a byte of a fill costs when `fill` bytes stream `unique`
    /// distinct ones, of which the blocks in flight at once re-read a
    /// `shared` part from one another's lines. Where the distinct bytes fit
    /// the [`CACHE_SHARE`] of the cache in front of DRAM each comes from
    /// DRAM once and its re-reads from the cache; past [`CACHE_SPILL`] times
    /// that every byte comes from DRAM but the shared part, which a cache
    /// too small for the whole still holds; between, the two blend.
    pub(crate) fn fill_byte(&self, unique: f64, fill: f64, shared: f64) -> f64 {
        let c = &self.device.costs;
        let held = CACHE_SHARE * c.l2_bytes;
        if held <= 0.0 || fill <= 0.0 {
            return c.global_byte;
        }
        let fits = ((CACHE_SPILL * held - unique) / ((CACHE_SPILL - 1.0) * held)).clamp(0.0, 1.0);
        let hit = (fits * (1.0 - unique / fill).max(0.0)).max(shared.clamp(0.0, 1.0));
        hit * c.l2_byte + (1.0 - hit) * c.global_byte
    }

    /// Every attention config for `tiles` rows of `t` queries over `tk`
    /// keys at head size `d` ([`AttentionProblem`]), the cheapest first.
    pub fn attention_ranked(
        &self,
        tiles: usize,
        t: usize,
        tk: usize,
        d: usize,
        causal: bool,
        bias: bool,
    ) -> Arc<[FaCfg]> {
        if tiles * t * tk * d == 0 {
            return Arc::new([]);
        }
        let p = AttentionProblem { tiles, t, tk, d, causal, bias };
        cached(&self.attention, p, || p.ranked(self))
    }

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
        (splits > 1).then_some(splits)
    }

    /// The depth a nearest kernel over `dtype` runs `d` at: a multiple of
    /// its matrix core's, the columns past `d` zero.
    pub fn nearest_depth(&self, d: usize, dtype: ScalarDType) -> Option<usize> {
        Some(d.next_multiple_of(self.device.target.geometry_of(dtype)?.k as usize))
    }

    fn ranked(&self, p: GemmProblem) -> Arc<[GemmCfg]> {
        cached(&self.ranked, p, || p.ranked(self))
    }

    /// Every GEMM config for `batches` GEMMs of `m` rows, `n` columns and
    /// reduction dim `k`, the cheapest of the lattice first.
    pub fn gemm_ranked(&self, batches: usize, m: usize, n: usize, k: usize, gated: bool) -> Arc<[GemmCfg]> {
        if batches * m * n * k == 0 {
            return Arc::new([]);
        }
        self.ranked(GemmProblem { batches, m, n, k, halves: if gated { 2 } else { 1 } })
    }

    /// Unsplit convolution configs for `batches` grid batches of `m` output
    /// pixels of `g` (`n = cout`, `k = kh·kw·cin`), the cheapest lattice
    /// configs whose step stays within one tap (`bk` divides `cin`) and
    /// whose tiles pad `m × n` by at most 1/8. On the cp.async path a step
    /// is at least two matrix-core depths, else the conv takes the graph.
    pub(crate) fn conv_ranked(&self, batches: usize, m: usize, g: &ConvGeom) -> Arc<[ConvCfg]> {
        let (n, k, cin) = (g.cout, g.k(), g.cin);
        if batches * m * n * k == 0 {
            return Arc::new([]);
        }
        let p = GemmProblem { batches, m, n, k, halves: 1 };
        let target = &self.device.target;
        let depth = match (target.fill, target.geometry()) {
            (Fill::CpAsync, Some(atom)) => 2 * atom.k as usize,
            _ => 0,
        };
        let eligible =
            |c: &&GemmCfg| cin.is_multiple_of(c.tile[2]) && c.tile[2] >= depth && p.padding(c) <= CONV_PADDING;
        self.ranked(p).iter().filter(eligible).map(|&gemm| ConvCfg { gemm, split: 1 }).collect()
    }

    /// The convolution candidates of `admitted` (the shortlist of
    /// [`Self::conv_ranked`]): each unsplit and, on a starved grid (fewer
    /// than twice [`CONV_BLOCKS_PER_SM`] blocks per SM, a single static
    /// batch), the lead and the admitted configs with the most blocks at
    /// 1/16 and at 1/8 padding also split over the reduction, in counts
    /// dividing the steps that keep at most 16 blocks per SM.
    pub(crate) fn conv_splits(&self, batches: usize, m: usize, g: &ConvGeom, admitted: &[ConvCfg]) -> Vec<ConvCfg> {
        let Some(&ConvCfg { gemm: base, .. }) = admitted.first() else { return vec![] };
        let (p, k) = (GemmProblem { batches, m, n: g.cout, k: g.k(), halves: 1 }, g.k());
        let mut out = admitted.to_vec();
        let sms = self.sms();
        let blocks = |c: &GemmCfg| p.blocks(c);
        let widest = |tight: bool| {
            let fits = |c: &&GemmCfg| !tight || p.padding(c) <= CONV_PADDING / 2.0;
            admitted.iter().map(|c| &c.gemm).filter(fits).max_by_key(|c| blocks(c)).copied().unwrap_or(base)
        };
        if batches == 1 && blocks(&base) < 2 * CONV_BLOCKS_PER_SM * sms {
            for gemm in [base, widest(true), widest(false)] {
                let trips = k / gemm.tile[2];
                for split in CONV_SPLITS {
                    let c = ConvCfg { gemm, split };
                    let within = blocks(&gemm) * split <= SPLIT_BLOCKS_PER_SM * sms;
                    if trips.is_multiple_of(split) && within && !out.contains(&c) {
                        out.push(c);
                    }
                }
            }
        }
        out
    }

    /// Key splits worth measuring for `tiles` independent query tiles over
    /// `blocks` key blocks: one, two, the counts around what keeps two blocks
    /// per SM busy, and every block its own split when that is within reach
    /// (the merge costs a launch, so never past the blocks).
    pub fn splits(&self, tiles: usize, blocks: usize) -> Vec<usize> {
        let want = (2 * self.sms()).div_ceil(tiles.max(1));
        let mut out: Vec<usize> = [1, 2, want / 2, want, 2 * want, blocks]
            .into_iter()
            .filter(|&s| (1..=blocks.min(2 * want)).contains(&s))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

impl FaCfg {
    /// Whether the config lowers on `target` for head size `d`: the atom's M
    /// is the kernel's [`WARP_ROWS`], a warp owns whole atom rows of queries
    /// and the block whole warps, both products tile the atoms, the key
    /// warps share the block as the kernel shares it
    /// ([`FaCfg::shares_keys`]), and the K and V fills (and the bias
    /// tile's under a bias) go in whole 16-byte chunks per thread
    /// ([`fills_in_chunks`]). On the warpgroup core a group of four row
    /// warps of one atom row each owns 64 query rows, both products' column
    /// blocks tile the instruction, a value row spans whole column panels
    /// (one swizzle span, or 128-byte panels) and no key warps split the
    /// block.
    pub fn fits(&self, target: &Target, d: usize, bias: bool) -> bool {
        let Some(atom) = target.geometry() else { return false };
        let (bq, bkv, rows) = (self.bq, self.bkv, self.warp_rows);
        // Key warps fill their own tiles alone.
        let threads = self.row_warps() as usize * target.wave as usize;
        let core = match self.core {
            Core::Warp => true,
            Core::Group => target.group_mma.first().is_some_and(|g| {
                let groups = self.row_warps() / GroupMmaAtom::WARPS;
                self.key_warps == 1
                    && rows == WARP_ROWS
                    && bq == GroupMmaAtom::M as usize * groups as usize
                    && groups.is_power_of_two()
                    && g.n_for(bkv as u32).is_some()
                    && g.n_for(d as u32).is_some()
                    && (Swizzle::spans(d, ELEM) || (d * ELEM).is_multiple_of(Swizzle::B128.span()))
            }),
        };
        core && atom.m as usize == WARP_ROWS
            && rows.is_multiple_of(WARP_ROWS)
            && rows > 0
            && bq.is_multiple_of(rows)
            && bq > 0
            && self.shares_keys().is_ok()
            && [bkv, d].iter().all(|&x| x.is_multiple_of(atom.n as usize) && x.is_multiple_of(atom.k as usize))
            && fills_in_chunks(Shape::new(bkv, d), ELEM, threads)
            && (!self.bias_ring(bias) || fills_in_chunks(Shape::new(bq, bkv), ELEM, threads))
            && target.fill.runs(self.stages)
    }

    /// The registers a lane holds for head size `d`, from the layouts the
    /// lowering assigns: the query fragments and the f32 output, live across
    /// the key loop; the key fragments, the f32 scores, the probabilities and
    /// the value fragments, all live together while a block's scores are
    /// turned into probabilities and multiplied in; and the staged K and V
    /// chunks where the fill goes through registers (the value tile is
    /// stored column-major where it is staged, so its fragments gather as
    /// 16-byte runs like the keys'), with the bias tile's under a bias.
    /// Addressing and what the compiler keeps beside the tiles are not
    /// counted: the compiled kernel's own count is what the tuner reads.
    /// On the warpgroup core the key and value tiles stay in shared memory
    /// and a chain's descriptors (two registers a `k16` step) stand in for
    /// their fragments.
    pub(crate) fn registers(&self, target: &Target, d: usize, bias: bool) -> Option<u32> {
        let grid = WarpGrid { rows: self.row_warps(), cols: 1 };
        let (qk, pv) = (
            target.issue(self.core, grid, self.bq, self.bkv, d)?,
            target.issue(self.core, grid, self.bq, d, self.bkv)?,
        );
        let half = |regs: u32| regs.div_ceil(2);
        let fill = if target.fill == Fill::CpAsync {
            0
        } else {
            let chunks = |shape| chunked(shape, 2, self.row_warps(), target.wave, Axis::Col).map(|t| half(t.regs()));
            let bias = if self.bias_ring(bias) { chunks(Shape::new(self.bq, self.bkv))? } else { 0 };
            2 * chunks(Shape::new(self.bkv, d))? + bias
        };
        let (q, o) = (half(qk.a.regs()), pv.c.regs());
        let (k, s, p, v) = match self.core {
            Core::Warp => (half(qk.b.regs()), qk.c.regs(), half(pv.a.regs()), half(pv.b.regs())),
            Core::Group => (2 * qk.a.reps[1], qk.c.regs(), half(pv.a.regs()), 2 * pv.a.reps[1]),
        };
        Some(q + o + k + s + p + v + fill)
    }

    /// Of [`Self::registers`], the f32 output accumulator a lane carries
    /// across the key loop: what a matrix core with a file of its own
    /// (CDNA's AGPRs) may hold beside the rest.
    pub(crate) fn accumulators(&self, target: &Target, d: usize) -> Option<u32> {
        let grid = WarpGrid { rows: self.row_warps(), cols: 1 };
        Some(target.issue(Core::Warp, grid, self.bq, d, self.bkv)?.c.regs())
    }
}

/// Search bound: an attention block's queries.
const QUERY_BLOCKS: [usize; 5] = [16, 32, 64, 128, 256];
/// Search bound: an attention block's keys a trip.
const KEY_BLOCKS: [usize; 4] = [16, 32, 64, 128];
/// Search bound: an attention ring's depths, the deepest first; a
/// register-staged fill runs the last alone.
const FA_STAGES: [usize; 2] = [3, 2];
/// Search bound: the warps a one-row-warp block's keys may split over.
const KEY_WARPS: [usize; 3] = [1, 2, 4];

/// Search bound: the fewest queries a block offers two atom rows per warp for.
const TWO_ROWS_FROM: usize = 64;

/// Model: trips a key block on a causal query block's diagonal counts for: the
/// block itself, its mask, and the rows above the diagonal every warp still
/// runs. A count of the work, not a fit.
pub(crate) const DIAGONAL: usize = 3;

/// An attention problem: `tiles` independent query rows of `t` queries over
/// `tk` keys at head size `d` (`tiles` = batch × heads), `causal` when a
/// query sees the keys up to its own position, `bias` when an additive bias
/// streams beside the keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionProblem {
    pub tiles: usize,
    pub t: usize,
    pub tk: usize,
    pub d: usize,
    pub causal: bool,
    pub bias: bool,
}

impl AttentionProblem {
    fn blocks(&self, cfg: &FaCfg) -> usize {
        self.tiles * self.t.div_ceil(cfg.bq)
    }

    /// What `cfg` costs here, as [`GemmProblem::admit`]: a key block's trip pays
    /// both products' fragment reads, fills (priced by where their bytes
    /// come from, every tile's keys and values the distinct ones; a bias
    /// tile is filled and read once per trip like them, every one of its
    /// bytes distinct) and steps, the softmax of every score and its
    /// overhead. A causal query block scores the keys its queries see on
    /// average plus its diagonal ([`Self::causal_trips`]). Key warps each take a share of the
    /// trips, every trip of theirs the block's whole work at once. The query
    /// blocks in flight at once share their tiles' keys and values whatever
    /// the cache holds of the whole; a wave's independent accumulators are
    /// the narrower product's, over every atom row the warp owns. A warp
    /// owning more rows reads each fragment for all of them: the fragment
    /// reads are per wave, so they divide by the rows per warp.
    pub fn admit(&self, planner: &Planner, cfg: &FaCfg) -> Result<(f64, u32), Rejected> {
        let Device { target, costs } = &planner.device;
        let (bq, bkv, d, kw) = (cfg.bq, cfg.bkv, self.d, cfg.key_warps);
        let smem = cfg.smem_bytes(d, self.bias) + cfg.scratch_bytes(target);
        if !cfg.fits(target, d, self.bias) {
            return Err(Rejected::Lowers);
        }
        if smem > target.smem_bytes {
            return Err(Rejected::Smem { need: smem, have: target.smem_bytes });
        }
        let regs = cfg.registers(target, d, self.bias).ok_or(Rejected::Layout)?;
        // Where the accumulators have a file of their own, the output's are
        // priced against it and the rest against the vector registers.
        if target.occupancy.acc_file > 0 {
            let acc = cfg.accumulators(target, d).ok_or(Rejected::Layout)?;
            if !target.occupancy.holds(acc, regs - acc) {
                return Err(Rejected::Registers {
                    need: regs,
                    have: target.occupancy.acc_file + target.occupancy.registers,
                });
            }
        }
        let waves = cfg.warps() as usize;
        // A warpgroup product reads no key or value fragments, holds one
        // chain per product, and is past `ptxas` beyond the block's share of
        // the register file.
        let (atoms, kv_reads, step) = match cfg.core {
            Core::Warp => {
                let atom = target.geometry().ok_or(Rejected::Lowers)?;
                let atoms = (cfg.warp_rows / atom.m as usize) * (bkv.min(d) / atom.n as usize);
                (atoms, 2 * waves * bkv * d * ELEM, costs.step)
            }
            Core::Group if regs <= target.lane_budget(waves as u32) => (1, 0, costs.step_group),
            Core::Group => return Err(Rejected::Registers { need: regs, have: target.lane_budget(waves as u32) }),
        };
        let bias = usize::from(self.bias) * kw * bq * bkv * ELEM;
        let fragments = kv_reads + bias;
        let fill = kw * 2 * bkv * d * ELEM + bias;
        let steps = kw * 2 * bq * bkv * d / STEP;
        let trips = if self.causal { self.causal_trips(bq, bkv) } else { self.tk.div_ceil(bkv) };
        let trips = trips.div_ceil(kw);
        let (blocks, stages) = (self.blocks(cfg), cfg.stages);
        let unique = (self.tiles * self.tk * (d * 2 + usize::from(self.bias) * self.t) * ELEM) as f64;
        let flight = planner.sms().min(blocks).max(1);
        let shared = 1.0 - flight.div_ceil(self.t.div_ceil(bq)) as f64 / flight as f64;
        let fill = fill as f64 * planner.fill_byte(unique, (blocks * trips * fill) as f64, shared);
        let reads = fragments as f64 * costs.smem_byte;
        let work = steps as f64 * step + (kw * bq * bkv) as f64 * costs.score;
        let trip = costs.trip(target.fill, stages, cfg.unroll);
        let run = Run { blocks, smem, regs, waves, atoms, trips, reads, work, fill, trip, stages };
        Ok((planner.pipelined(run), regs))
    }

    /// The key-block trips of a causal query block of `bq` rows over key
    /// blocks of `bkv`, on average over the blocks: the mask is aligned at
    /// the first query and key (query `i` sees keys `0..=i`), so a query
    /// sees `min(i + 1, tk)` keys and the `t` queries `seen · (2t − seen) / 2`
    /// on average with `seen = min(t, tk)`; plus the block's diagonal, a key
    /// block per `bkv` of its rows, each [`DIAGONAL`] trips.
    pub(crate) fn causal_trips(&self, bq: usize, bkv: usize) -> usize {
        let (t, seen) = (self.t, self.t.min(self.tk));
        (seen * (2 * t - seen)).div_ceil(2 * t * bkv) + DIAGONAL * bq.div_ceil(bkv)
    }

    /// Every config the ranking spans for this problem on `target`: every
    /// pipeline of every query and key block size at one atom row per warp,
    /// on the warp-level core and, where the target has it, the warpgroup
    /// core, in the plain softmax forms.
    pub fn lattice(&self, target: &Target) -> Vec<FaCfg> {
        let stages: &[usize] = if target.fill == Fill::CpAsync { &FA_STAGES } else { &FA_STAGES[1..] };
        let cores: &[Core] = if target.group_mma.is_empty() { &[Core::Warp] } else { &[Core::Warp, Core::Group] };
        let mut out = vec![];
        for (core, bq) in cores.iter().flat_map(|&c| QUERY_BLOCKS.map(|bq| (c, bq))) {
            // The warp-level core's blocks stop at eight row warps.
            if core == Core::Warp && bq > 128 {
                continue;
            }
            // A row warp of queries may share its block with key warps.
            let key_warps: &[usize] = if bq == WARP_ROWS && self.t <= WARP_ROWS { &KEY_WARPS } else { &[1] };
            for bkv in KEY_BLOCKS {
                for &stages in stages {
                    for unroll in [false, true] {
                        for &key_warps in key_warps {
                            out.push(FaCfg { unroll, key_warps, core, ..FaCfg::new(bq, bkv, stages) });
                        }
                    }
                }
            }
        }
        out
    }

    /// The [`lattice`](Self::lattice)'s configs that run this problem on the planner's target,
    /// cheapest first, then by fewer registers, every pipeline of every query
    /// and key block size at one atom row per warp, on the warp-level core
    /// and, where the target has it, the warpgroup core, in the plain softmax
    /// forms (the tuner sets the others); the shortlist keeps a tile's first
    /// clean one. After them, the same configs at two atom rows per warp
    /// where that can pay ([`Self::two_rows_pay`]), in their twins' order: a
    /// variant of a tile (its rows per warp are not part of it), so the
    /// shortlist holds the one-row representatives and the twins stage times
    /// the variant beside them for the contender tiles.
    pub(crate) fn ranked(&self, planner: &Planner) -> Vec<FaCfg> {
        let lattice = self.lattice(&planner.device.target);
        let count = lattice.len();
        let admitted: Vec<(f64, u32, FaCfg)> = lattice
            .into_iter()
            .filter_map(|c| self.admit(planner, &c).ok().map(|(cost, regs)| (cost, regs, c)))
            .collect();
        let one: Vec<FaCfg> = sorted(self, admitted, count).into_iter().map(|(_, _, c)| c).collect();
        let two = one
            .iter()
            .map(|&c| (c, FaCfg { warp_rows: 2 * WARP_ROWS, ..c }))
            .filter(|&(c, wide)| self.two_rows_pay(planner, &c, &wide))
            .map(|(_, wide)| wide)
            .collect::<Vec<_>>();
        [one, two].concat()
    }

    /// Whether `wide`, `one` at two atom rows per warp, can pay: the
    /// warp-level core (a warpgroup's rows are the instruction's), a block of
    /// at least [`TWO_ROWS_FROM`] queries (fewer leave half the warps a
    /// block holds, too few to hide a chain), one key warp, admitted, and as
    /// many blocks resident as `one` keeps (the wider tiles' registers must
    /// not cost the occupancy the shared fragments buy back).
    fn two_rows_pay(&self, planner: &Planner, one: &FaCfg, wide: &FaCfg) -> bool {
        let target = &planner.device.target;
        let resident = |cfg: &FaCfg| -> Option<usize> {
            let smem = cfg.smem_bytes(self.d, self.bias) + cfg.scratch_bytes(target);
            Some(planner.resident(smem, cfg.registers(target, self.d, self.bias)?, cfg.warps() as usize))
        };
        one.core == Core::Warp
            && one.bq >= TWO_ROWS_FROM
            && one.key_warps == 1
            && self.admit(planner, wide).is_ok()
            && resident(wide).zip(resident(one)).is_some_and(|(w, o)| w >= o)
    }
}

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
const NEAREST_PAIR: f64 = 1.0;
/// Model: one more elementwise operation a pair, in the same units: the
/// row's norm added, a floor's maximum, the L2 clamp.
const PAIR_OP: f64 = 0.2;
/// Model: what groups add to a pair: a compare and a select.
const GROUPS_PAIR: f64 = 2.0 * PAIR_OP;
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
        let first = if cfg.select.is_some() { MATRIX_PAIR } else { NEAREST_PAIR };
        let pair = first + ops as f64 * PAIR_OP + if self.groups { GROUPS_PAIR } else { 0.0 };
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
            epilogue += share.listing(cfg, lanes, target.wave) * waves as f64 * costs.score / (tiles * chunks) as f64;
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
        let row = (splits - 1) as f64 * slots * (slots.log2() + 1.0) * MERGE_SLOT;
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
        let row = steps * LANES as f64 * SCAN_SCORE
            + skipped as f64 * SKIP_STEP
            + busy * queued * INSERT_SLOT
            + merges * merge * MERGE_SLOT;
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
    fn listing(&self, cfg: &NearestCfg, lanes: LaneQueue, wave: u32) -> f64 {
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
                return entered * deals * (elems * PASS_ELEM + wave * DEAL_LANE)
                    + dealt * (elems * (HIT_ELEM + t * PUSH_ELEM) + wave * DEAL_LANE);
            }
            let lambda = hits / elems;
            let rounds = (1.0 + lambda + (2.0 * lambda * elems.ln()).sqrt()).min(deals);
            let round = deals * elems * ROUND_ELEM + elems * t * PUSH_ELEM + wave * DEAL_LANE;
            entered * (deals * elems * DEAL_ELEM + wave * DEAL_LANE + rounds * round)
        };
        let walk: f64 = (0..tiles).map(trip).sum();
        let fact: f64 = (1..=depth + 1).map(|i| i as f64).product();
        let per_merge = (elems.powf(t) * fact).powf(1.0 / (t + 1.0));
        let (held, queued) = (cfg.list_slots(self.k) as f64, (width * depth) as f64);
        let sort = queued * queued.log2() * (queued.log2() + 1.0) / 2.0;
        let merge = rows * (sort + held * (held.log2() + 1.0)) * LIST_SLOT;
        let mut list = walk + (rows * self.offers() / per_merge + 1.0) * merge;
        if let Some(q) = cfg.queue {
            let stages = 1.0 + 2.0 * (q as f64).log2();
            let flush = (self.k.div_ceil(q) * q) as f64 * rows * stages * MERGE_SLOT;
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
        let stages: &[usize] = if target.fill == Fill::CpAsync { &FA_STAGES } else { &NEAREST_STAGED };
        // A list in registers up to their most; a long one queued into the
        // outputs and listed from a score matrix as well.
        let registers = (self.k <= REGISTER_K).then_some((None, None));
        let long = self.k >= LONG_FROM;
        let queued = QUEUES.into_iter().filter(|&q| long && q < self.k).map(|q| (Some(q), None));
        // Filtered where a row's trips are many to its list's slots.
        let filters =
            [false, true].into_iter().filter(|&f| !f || self.m >= FILTER_GROUPS * self.k * CANDIDATE_BLOCKS[0]);
        let selects =
            filters.flat_map(|filter| SELECT_QUEUES.map(|queue| SelectCfg { warps: SELECT_WARPS, queue, filter }));
        let selects = selects.filter(|s| long && s.fits(self.k)).map(|s| (None, Some(s)));
        let lists: Vec<(Option<usize>, Option<SelectCfg>)> =
            registers.into_iter().chain(queued).chain(selects).collect();
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
                                for &(queue, select) in &lists {
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
        let mut products: HashMap<SelectCfg, HashSet<NearestCfg>> = HashMap::new();
        admitted.retain(|(_, _, c)| {
            let Some(select) = c.select else { return true };
            let tiles = products.entry(select).or_default();
            if tiles.len() < MATRIX_TILES {
                tiles.insert(c.tile());
            }
            tiles.contains(&c.tile())
        });
        sorted(self, admitted, count).into_iter().map(|(_, _, c)| c).collect()
    }
}

/// A tile is its query and candidate blocks, its rows per warp, its depth
/// chunk and its lists; its ring depth and unrolling are its pipelines, and
/// whether its lane queues take a trip in rounds or in passes its twins too
/// (timed where the tile leads, never taking a place of the shortlist).
impl Tiled for NearestCfg {
    fn tile(&self) -> Self {
        let lanes = self.lanes.map(|l| LaneQueue { passes: false, ..l });
        NearestCfg { stages: 0, unroll: false, lanes, ..*self }
    }

    /// The ranked (unsplit) pipelines of the tile, at the split the tuner
    /// gave this config.
    fn twins(&self, ranked: &[Self]) -> Vec<Self> {
        let split = |c: &Self| NearestCfg { splits: self.splits, ..*c };
        ranked.iter().map(split).filter(|c| c.tile() == self.tile()).collect()
    }
}

/// One warp per head row, the head split in halves a warp spreads over.
pub(crate) fn heads_candidates(d: usize) -> Vec<NormCfg> {
    if !(d.is_power_of_two() && (16..=256).contains(&d)) {
        return vec![];
    }
    ROW_WARPS.map(|br| NormCfg { br }).to_vec()
}

/// One warp per row of a power-of-two width a warp's vector loads span.
pub fn norm_candidates(d: usize) -> Vec<NormCfg> {
    if !(d.is_power_of_two() && (256..=2048).contains(&d)) {
        return vec![];
    }
    ROW_WARPS.map(|br| NormCfg { br }).to_vec()
}

/// Search bound: the rows (one warp each) of a row kernel's block.
const ROW_WARPS: [usize; 3] = [4, 8, 16];
