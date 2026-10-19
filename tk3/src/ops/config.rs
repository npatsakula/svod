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

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use svod_dtype::GpuArch;

use crate::atoms::{Device, GroupMmaAtom, Target};
use crate::ir::{Axis, Orient, Shape};
use crate::kernels::attention::{FaCfg, WARP_ROWS};
use crate::kernels::conv::{ConvCfg, ConvGeom};
use crate::kernels::gemm::{Core, GemmCfg};
use crate::kernels::rows::NormCfg;
use crate::layouts::{Issue, WarpGrid, chunked};
use crate::tune::Compiled;

/// Targets with kernels: `mma.sync` + `cp.async` + `ldmatrix` (sm_80+, measured
/// on sm_86; Hopper and Blackwell run the same path, no wgmma or TMA), and
/// AMD matrix cores with register-staged fills: RDNA3/RDNA3.5/RDNA4 WMMA
/// (measured on gfx1201) and CDNA3/4 MFMA 16×16×16 on wave64 (compiled, not
/// yet measured).
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
        let grid = WarpGrid { rows: wr, cols: wc };
        match self.core {
            Core::Warp => target.mma.first()?.issue(Orient::Direct, grid, bm, bn, bk).ok(),
            Core::Group => target.group_mma.first()?.issue(grid, bm, bn, bk).ok(),
        }
    }

    /// Whether the warp grid tiles the output with `target`'s matrix core
    /// atoms and the ring depth is one the target's prefetch runs. A
    /// warpgroup config's rows of shared operands must be one swizzle span
    /// (32, 64 or 128 bytes), the descriptor's own.
    fn tiles(&self, target: &Target) -> bool {
        let bk = self.tile[2];
        let row = self.core == Core::Warp || matches!(bk * 2, 32 | 64 | 128);
        // Two crews: the two warp rows of the warp-level core, one slot, and
        // k-slices of a product each with another to overlap, on a target
        // that interleaves crews.
        let crews = self.crews == 1
            || (self.crews == 2
                && self.core == Core::Warp
                && target.crews >= 2
                && self.stages == 1
                && self.warps[0] == 2
                && bk.is_multiple_of(crate::kernels::gemm::KSLICE)
                && bk >= 2 * crate::kernels::gemm::KSLICE);
        row && crews && self.issue(target, bk).is_some() && target.stages(self.stages) == self.stages
    }

    /// Whether the config lowers on `target`: it [`tiles`](Self::tiles), and
    /// both operand tiles split into whole 16-byte chunks per thread.
    pub fn fits(&self, target: &Target) -> bool {
        let ([bm, bn, bk], [wr, wc]) = (self.tile, self.warps.map(|w| w as usize));
        let threads = target.wave as usize * wr * wc;
        self.tiles(target) && (bm * bk / 8).is_multiple_of(threads) && (bn * bk / 8).is_multiple_of(threads)
    }

    /// The registers a lane holds for a GEMM over a weight of `halves`
    /// halves (two when gated): the f32 accumulators, the 16-bit operand
    /// fragments and, where the fill goes through registers, its staged
    /// chunks, from the layouts the lowering assigns. `None` when the tile
    /// does not lay out. Addressing and the epilogue's temporaries are not
    /// counted.
    pub fn registers(&self, target: &Target, halves: usize) -> Option<u32> {
        let ([bm, bn, bk], [wr, wc]) = (self.tile, self.warps);
        let issue = self.issue(target, bk)?;
        let halves = halves as u32;
        let acc = issue.c.regs() * halves;
        let operands = self.operand_registers(&issue, halves);
        let fill = if target.cp_async {
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
    pub fn live_registers(&self, target: &Target, halves: usize) -> Option<(u32, u32)> {
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
    GemmCfg { core: Core::Warp, tile, stages, warps, group_m: 8, unroll, crews: 1 }
}

/// Search bound: the knobs a GEMM-shaped kernel is enumerated over: output tile edges
/// (48 and 96 columns for the convolution bodies a power of two would pad
/// by a third), reduction depths and warp grids.
const ROWS: [usize; 4] = [32, 64, 128, 256];
const COLS: [usize; 5] = [48, 64, 96, 128, 256];
const DEPTHS: [usize; 3] = [16, 32, 64];
const GRIDS: [[u32; 2]; 9] = [[1, 1], [1, 2], [2, 1], [2, 2], [1, 4], [4, 1], [2, 4], [4, 2], [4, 4]];
/// Warp grids of whole warpgroups down the rows, one to four of them.
const GROUP_GRIDS: [[u32; 2]; 6] = [[4, 1], [4, 2], [8, 1], [8, 2], [16, 1], [4, 4]];

/// A core with the warp grids and ring depths the lattice spans it over.
type Cores = [(Core, &'static [[u32; 2]], &'static [usize])];

/// Every config the construct spans on `target`: the tiles and warp grids,
/// the ring depths its prefetch runs, rolled and unrolled; on a target with
/// warpgroup instructions, the tiles of whole warpgroups on them as well.
pub fn gemm_lattice(target: &Target) -> Vec<GemmCfg> {
    let cores: &Cores = match (target.cp_async, target.group_mma.is_empty()) {
        (false, _) => &[(Core::Warp, &GRIDS, &[2, 1])],
        (true, true) => &[(Core::Warp, &GRIDS, &[3, 2, 1])],
        // A warpgroup ring keeps a chain in flight in a slot of its own.
        (true, false) => &[(Core::Warp, &GRIDS, &[3, 2, 1]), (Core::Group, &GROUP_GRIDS, &[4, 3])],
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
    /// 16-byte chunks per thread, a ring depth the prefetch does not run.
    Lowers,
    /// Its step is shallower than two matrix-core depths where the
    /// reduction allows one.
    Shallow,
    /// Its step does not divide the reduction.
    Divides,
    /// Its shared memory passes the block's.
    Smem { need: usize, have: usize },
    /// Its tiles have no register layout.
    Layout,
    /// The registers a lane holds at once pass what it may.
    Registers { need: u32, have: u32 },
}

/// A ranking: its configs with their costs ([`Planner::pipelined`] ns of SM
/// time) and registers by the layout count, cheapest first (then fewer
/// registers), logged under [`crate::ops::PLAN`] with how many of the
/// `spanned` configs it left out.
fn sorted<C: Debug>(problem: &impl Debug, mut ranked: Vec<(f64, u32, C)>, spanned: usize) -> Vec<(f64, u32, C)> {
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let (admitted, rejected) = (ranked.len(), spanned - ranked.len());
    tracing::debug!(target: crate::ops::PLAN, ?problem, admitted, rejected, "ranking");
    for (rank, (ns, regs, cfg)) in ranked.iter().take(LOGGED).enumerate() {
        tracing::debug!(target: crate::ops::PLAN, rank, ns, regs, ?cfg, "ranked");
    }
    ranked
}

/// The configs of a ranking logged with their costs.
const LOGGED: usize = 16;

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
pub struct Problem {
    pub batches: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub halves: usize,
}

impl Problem {
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
    /// A rows and their B columns whatever the cache holds of the whole.
    pub fn admit(&self, planner: &Planner, cfg: &GemmCfg) -> Result<(f64, u32), Rejected> {
        let Device { target, costs } = &planner.device;
        let ([bm, bn, bk], [wr, wc]) = (cfg.tile, cfg.warps.map(|w| w as usize));
        let gated = self.halves == 2;
        let atom = target.mma.first().ok_or(Rejected::Lowers)?;
        // A step is at least two matrix-core depths, unless the reduction dim
        // allows nothing deeper.
        let depth = 2 * atom.k as usize;
        let smem = cfg.smem_bytes(gated);
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
                waves * (bm / wr + bn * halves / wc) * bk * 2,
                costs.step,
            ),
            Core::Group => {
                let group = target.group_mma.first().ok_or(Rejected::Lowers)?;
                let n = group.n_for((bn / wc) as u32).ok_or(Rejected::Lowers)? as usize;
                (bn * halves / wc / n, 0, costs.step_group)
            }
        };
        let fill = (bm + bn * halves) * bk * 2;
        let steps = bm * bn * halves * bk / 4096;
        let (blocks, trips, stages) = (self.blocks(cfg), self.k / bk, cfg.stages);
        let unique = (self.batches * (self.m + self.n * halves) * self.k * 2) as f64;
        let flight = planner.sms().min(blocks).max(1);
        let group = cfg.group_m.max(1).min(flight);
        let distinct = group * bm + flight.div_ceil(group) * bn * halves;
        let shared = 1.0 - distinct as f64 / (flight * (bm + bn * halves)) as f64;
        let fill = fill as f64 * planner.fill_byte(unique, (blocks * trips * fill) as f64, shared);
        let reads = fragments as f64 * costs.smem_byte;
        let work = steps as f64 * step;
        let trip = costs.trip(stages, !cfg.unroll && !target.cp_async);
        let run = Run { blocks, smem, regs, waves, atoms, trips, reads, work, fill, trip, stages };
        Ok((planner.pipelined(run), regs))
    }

    /// The lattice's configs that run this problem on the planner's target,
    /// every pipeline (depth, ring depth, unrolling) of every tile, cheapest
    /// first (then by fewer registers) and [`spread`] over the block tiles
    /// the cost band holds: the shortlist keeps a tile's first clean one
    /// ([`shortlist`]).
    pub fn ranked(&self, planner: &Planner) -> Vec<GemmCfg> {
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

/// Search bound: the tiles the tune store measures per shape, compiled in
/// parallel batches of as many: most of the lattice, since timing a tile costs
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
/// Model: blocks per SM a convolution's grid should reach before its
/// reduction is left whole: its grids are a few hundred blocks.
const CONV_BLOCKS_PER_SM: usize = 2;

/// A config's tile: all of it but its pipeline (ring depth, unrolling),
/// which the ranking orders by a cost model alone and the compiler may
/// refuse, so the shortlist holds one config per tile, its first clean
/// pipeline ([`shortlist`]); the tile that wins has its other pipelines,
/// its [`twins`](Self::twins), timed after.
pub(crate) trait Tiled: Copy + Eq + Hash {
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

/// `list` with each config once, in order.
pub(crate) fn distinct_configs<C: Tiled>(list: &[C]) -> Vec<C> {
    let mut seen = HashSet::new();
    list.iter().copied().filter(|c| seen.insert(*c)).collect()
}

/// The indices of the first [`SHORTLIST`] configs of `ranked` whose
/// compiled kernels do not spill, one per tile (its first clean pipeline,
/// so a spilling unrolled ring yields to its rolled twin), in ranked order;
/// where every config that compiled spills, the first [`SHORTLIST`] of
/// those (the time judges, and a spilling kernel still beats the graph's).
/// `compile` is asked about batches of [`SHORTLIST`] in that order (the
/// tuner compiles a batch in parallel) until enough tiles are clean or the
/// list ends. `None` when nothing compiled: no device to judge on.
pub(crate) fn shortlist<C: Tiled>(ranked: &[C], compile: impl FnMut(&[C]) -> Vec<Compiled>) -> Option<Vec<usize>> {
    let mut tiles = HashSet::new();
    judged(ranked, SHORTLIST, SHORTLIST, compile, |c| tiles.insert(c.tile()))
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
pub(crate) fn first_clean<C: Tiled>(ranked: &[C], compile: impl FnMut(&[C]) -> Vec<Compiled>) -> Option<usize> {
    judged(ranked, 1, FIRST, compile, |_| true).map(|at| at[0])
}

fn judged<C: Tiled>(
    ranked: &[C],
    limit: usize,
    batch: usize,
    mut compile: impl FnMut(&[C]) -> Vec<Compiled>,
    mut keep: impl FnMut(&C) -> bool,
) -> Option<Vec<usize>> {
    let (mut clean, mut spilled) = (vec![], vec![]);
    for (n, chunk) in ranked.chunks(batch).enumerate() {
        for (i, compiled) in compile(chunk).into_iter().enumerate() {
            let at = n * batch + i;
            match compiled {
                Compiled::Clean if keep(&chunk[i]) => clean.push(at),
                Compiled::Clean | Compiled::Failed => {}
                Compiled::Spilled => spilled.push(at),
            }
        }
        if clean.len() >= limit {
            break;
        }
    }
    let mut out = if !clean.is_empty() {
        clean
    } else if !spilled.is_empty() {
        spilled
    } else {
        return None;
    };
    out.truncate(limit);
    Some(out)
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
    /// The cost fields the device's probes could not measure, ranked at
    /// zero ([`crate::probe::Measured`]); empty for a device that measured.
    pub unmeasured: Vec<&'static str>,
    ranked: papaya::HashMap<Problem, Arc<[GemmCfg]>>,
    attention: papaya::HashMap<Attention, Arc<[FaCfg]>>,
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
    pub fn new(device: Device) -> Self {
        Self { device, unmeasured: vec![], ranked: papaya::HashMap::new(), attention: papaya::HashMap::new() }
    }

    /// A planner over what the device's probes measured.
    pub fn measured(target: Target, measured: crate::probe::Measured) -> Self {
        let crate::probe::Measured { costs, unmeasured } = measured;
        Self { unmeasured, ..Self::new(Device { target, costs }) }
    }

    /// The device's SM count: a planned target reports one.
    pub fn sms(&self) -> usize {
        self.device.target.sms.expect("a planned target reports its SM count") as usize
    }

    /// The SM time of `run`. The busiest SM issues the work of
    /// `⌈blocks / sms⌉` blocks, each taken up as another ends (measured on
    /// sm_86: a 64×64 GEMM's time steps with that count, not with whole
    /// rounds of resident blocks), a trip's fixed overhead shared with the
    /// blocks it holds at once: by its shared memory and by its register file
    /// under the layout estimate, at least one (the compiled kernel, not the
    /// estimate, decides whether a block runs). A trip's pipes (the fragment
    /// reads, the matrix-core steps, the fill) overlap across the waves a
    /// SIMD holds: the longest is paid whole, the rest in a share of one over
    /// the waves per SIMD up to [`OVERLAP_WAVES`], the blocks the SM holds at
    /// once pooling theirs (measured on gfx1201 and sm_86: a block at one
    /// wave per SIMD pays about their sum, at four or more the longest). A SIMD holding fewer
    /// waves than hide a matrix-core chain's latency (`step_lone / step` of
    /// them, a wave's independent accumulators counting as waves) runs the
    /// steps that many times slower. A round of blocks is at least one wave's
    /// own issue (a block of fewer waves than SIMDs leaves the rest idle),
    /// and every round of every trip pays the memory latency the ring cannot
    /// hide (it prefetches `stages - 1` trips ahead) on top of its issue:
    /// measured on both, the latency adds rather than hides under the issue.
    pub(crate) fn pipelined(&self, run: Run) -> f64 {
        let Run { blocks, smem, regs, waves, atoms, trips, reads, work, fill, trip, stages } = run;
        let Device { target, costs } = &self.device;
        let (o, sms) = (target.occupancy, self.sms());
        let per_simd = o.waves.min(o.file / regs.max(1)) as usize;
        let resident = (target.smem_bytes / smem).min(o.simds as usize * per_simd / waves).max(1);
        let per_sm = blocks.div_ceil(sms);
        let rounds = per_sm.div_ceil(resident);
        let hide =
            if costs.step > 0.0 && costs.step_lone > 0.0 { (costs.step_lone / costs.step).max(1.0) } else { 1.0 };
        let per_simd = |waves: usize| (waves as f64 / o.simds as f64).max(1.0);
        let lone = |waves: usize| (hide / (per_simd(waves) * atoms.max(1) as f64)).max(1.0);
        let body = |waves: usize| {
            let pipes = [reads, work * lone(waves), fill];
            let longest = pipes.into_iter().fold(0.0, f64::max);
            longest + (pipes.iter().sum::<f64>() - longest) / per_simd(waves).min(OVERLAP_WAVES)
        };
        let at_once = resident.min(per_sm);
        let issue = per_sm as f64 * (body(at_once * waves) + trip / at_once as f64);
        let wave = o.simds as f64 * (body(waves) + trip) / waves as f64;
        let wait = costs.latency / stages.saturating_sub(1).max(1) as f64;
        trips as f64 * (issue.max(rounds as f64 * wave) + rounds as f64 * wait)
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
    /// keys at head size `d` ([`Attention`]), the cheapest first.
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
        let p = Attention { tiles, t, tk, d, causal, bias };
        cached(&self.attention, p, || p.ranked(self))
    }

    fn ranked(&self, p: Problem) -> Arc<[GemmCfg]> {
        cached(&self.ranked, p, || p.ranked(self))
    }

    /// Every GEMM config for `batches` GEMMs of `m` rows, `n` columns and
    /// reduction dim `k`, the cheapest of the lattice first.
    pub fn gemm_ranked(&self, batches: usize, m: usize, n: usize, k: usize, gated: bool) -> Arc<[GemmCfg]> {
        if batches * m * n * k == 0 {
            return Arc::new([]);
        }
        self.ranked(Problem { batches, m, n, k, halves: if gated { 2 } else { 1 } })
    }

    /// Unsplit convolution configs for `batches` grid batches of `m` output
    /// pixels of `g` (`n = cout`, `k = kh·kw·cin`), the cheapest lattice
    /// configs whose step stays within one tap (`bk` divides `cin`) and
    /// whose tiles pad `m × n` by at most 1/8. On the cp.async path a step
    /// is at least two matrix-core depths, else the conv takes the graph.
    pub fn conv_ranked(&self, batches: usize, m: usize, g: &ConvGeom) -> Arc<[ConvCfg]> {
        let (n, k, cin) = (g.cout, g.k(), g.cin);
        if batches * m * n * k == 0 {
            return Arc::new([]);
        }
        let p = Problem { batches, m, n, k, halves: 1 };
        let target = &self.device.target;
        let depth = if target.cp_async { 2 * target.mma.first().map_or(16, |a| a.k as usize) } else { 0 };
        let eligible = |c: &&GemmCfg| cin.is_multiple_of(c.tile[2]) && c.tile[2] >= depth && p.padding(c) <= 1.0 / 8.0;
        self.ranked(p).iter().filter(eligible).map(|&gemm| ConvCfg { gemm, split: 1 }).collect()
    }

    /// The convolution candidates of `admitted` (the shortlist of
    /// [`Self::conv_ranked`]): each unsplit and, on a starved grid (fewer
    /// than twice [`CONV_BLOCKS_PER_SM`] blocks per SM, a single static
    /// batch), the lead and the admitted configs with the most blocks at
    /// 1/16 and at 1/8 padding also split over the reduction, in counts
    /// dividing the steps that keep at most 16 blocks per SM.
    pub fn conv_splits(&self, batches: usize, m: usize, g: &ConvGeom, admitted: &[ConvCfg]) -> Vec<ConvCfg> {
        let Some(&ConvCfg { gemm: base, .. }) = admitted.first() else { return vec![] };
        let (p, k) = (Problem { batches, m, n: g.cout, k: g.k(), halves: 1 }, g.k());
        let mut out = admitted.to_vec();
        let sms = self.sms();
        let blocks = |c: &GemmCfg| p.blocks(c);
        let widest = |tight: bool| {
            let fits = |c: &&GemmCfg| !tight || p.padding(c) <= 1.0 / 16.0;
            admitted.iter().map(|c| &c.gemm).filter(fits).max_by_key(|c| blocks(c)).copied().unwrap_or(base)
        };
        if batches == 1 && blocks(&base) < 2 * CONV_BLOCKS_PER_SM * sms {
            for gemm in [base, widest(true), widest(false)] {
                let trips = k / gemm.tile[2];
                for split in [2, 3, 4, 6, 9] {
                    let c = ConvCfg { gemm, split };
                    if trips.is_multiple_of(split) && blocks(&gemm) * split <= 16 * sms && !out.contains(&c) {
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
    /// and the block whole warps, both products tile the atoms, and the K
    /// and V fills (and the bias tile's under a bias) split into whole
    /// 16-byte chunks per thread. On the warpgroup core a group of four row
    /// warps of one atom row each owns 64 query rows, both products' column
    /// blocks tile the instruction, a value row spans whole column panels
    /// (one swizzle span, or 128-byte panels) and no key warps split the
    /// block.
    pub fn fits(&self, target: &Target, d: usize, bias: bool) -> bool {
        let Some(atom) = target.mma.first() else { return false };
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
                    && (matches!(d * 2, 32 | 64 | 128) || (d * 2).is_multiple_of(128))
            }),
        };
        core && atom.m as usize == WARP_ROWS
            && rows.is_multiple_of(WARP_ROWS)
            && rows > 0
            && bq.is_multiple_of(rows)
            && bq > 0
            // A key warp's merged output reuses its 16-bit key ring as f32.
            && (self.key_warps == 1 || bq == WARP_ROWS && self.stages * bkv >= 2 * WARP_ROWS)
            && [bkv, d].iter().all(|&x| x.is_multiple_of(atom.n as usize) && x.is_multiple_of(atom.k as usize))
            && (bkv * d / 8).is_multiple_of(threads)
            && (!self.bias_ring(bias) || (bq * bkv / 8).is_multiple_of(threads))
            && target.stages(self.stages) == self.stages
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
    pub fn registers(&self, target: &Target, d: usize, bias: bool) -> Option<u32> {
        let grid = WarpGrid { rows: self.row_warps(), cols: 1 };
        let (qk, pv) = match self.core {
            Core::Warp => {
                let atom = target.mma.first()?;
                (
                    atom.issue(Orient::Direct, grid, self.bq, self.bkv, d).ok()?,
                    atom.issue(Orient::Direct, grid, self.bq, d, self.bkv).ok()?,
                )
            }
            Core::Group => {
                let atom = target.group_mma.first()?;
                (atom.issue(grid, self.bq, self.bkv, d).ok()?, atom.issue(grid, self.bq, d, self.bkv).ok()?)
            }
        };
        let half = |regs: u32| regs.div_ceil(2);
        let fill = if target.cp_async {
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
    pub fn accumulators(&self, target: &Target, d: usize) -> Option<u32> {
        let grid = WarpGrid { rows: self.row_warps(), cols: 1 };
        Some(target.mma.first()?.issue(Orient::Direct, grid, self.bq, d, self.bkv).ok()?.c.regs())
    }
}

/// Model: the fewest queries a block offers two atom rows per warp for.
const TWO_ROWS_FROM: usize = 64;

/// Model: trips a key block on a causal query block's diagonal counts for: the
/// block itself, its mask, and the rows above the diagonal every warp still
/// runs. A count of the work, not a fit.
const DIAGONAL: usize = 3;

/// An attention problem: `tiles` independent query rows of `t` queries over
/// `tk` keys at head size `d` (`tiles` = batch × heads), `causal` when a
/// query sees the keys up to its own position, `bias` when an additive bias
/// streams beside the keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Attention {
    pub tiles: usize,
    pub t: usize,
    pub tk: usize,
    pub d: usize,
    pub causal: bool,
    pub bias: bool,
}

impl Attention {
    fn blocks(&self, cfg: &FaCfg) -> usize {
        self.tiles * self.t.div_ceil(cfg.bq)
    }

    /// What `cfg` costs here, as [`Problem::admit`]: a key block's trip pays
    /// both products' fragment reads, fills (priced by where their bytes
    /// come from, every tile's keys and values the distinct ones; a bias
    /// tile is filled and read once per trip like them, every one of its
    /// bytes distinct) and steps, the softmax of every score and its
    /// overhead. A causal query block
    /// scores half the keys plus its diagonal, a key block per `bkv` of its
    /// rows, each [`DIAGONAL`] trips. Key warps each take a share of the
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
                let atom = target.mma.first().ok_or(Rejected::Lowers)?;
                let atoms = (cfg.warp_rows / atom.m as usize) * (bkv.min(d) / atom.n as usize);
                (atoms, 2 * waves * bkv * d * 2, costs.step)
            }
            Core::Group if regs <= target.lane_budget(waves as u32) => (1, 0, costs.step_group),
            Core::Group => return Err(Rejected::Registers { need: regs, have: target.lane_budget(waves as u32) }),
        };
        let bias = usize::from(self.bias) * kw * bq * bkv * 2;
        let fragments = kv_reads + bias;
        let fill = kw * 2 * bkv * d * 2 + bias;
        let steps = kw * 2 * bq * bkv * d / 4096;
        let diagonal = DIAGONAL * bq.div_ceil(bkv);
        let trips = if self.causal { self.tk.div_ceil(2 * bkv) + diagonal } else { self.tk.div_ceil(bkv) };
        let trips = trips.div_ceil(kw);
        let (blocks, stages) = (self.blocks(cfg), cfg.stages);
        let unique = (self.tiles * self.tk * (d * 2 + usize::from(self.bias) * self.t) * 2) as f64;
        let flight = planner.sms().min(blocks).max(1);
        let shared = 1.0 - flight.div_ceil(self.t.div_ceil(bq)) as f64 / flight as f64;
        let fill = fill as f64 * planner.fill_byte(unique, (blocks * trips * fill) as f64, shared);
        let reads = fragments as f64 * costs.smem_byte;
        let work = steps as f64 * step + (kw * bq * bkv) as f64 * costs.score;
        let trip = costs.trip(stages, !cfg.unroll && !target.cp_async);
        let run = Run { blocks, smem, regs, waves, atoms, trips, reads, work, fill, trip, stages };
        Ok((planner.pipelined(run), regs))
    }

    /// Every config the ranking spans for this problem on `target`: every
    /// pipeline of every query and key block size at one atom row per warp,
    /// on the warp-level core and, where the target has it, the warpgroup
    /// core, in the plain softmax forms.
    pub fn lattice(&self, target: &Target) -> Vec<FaCfg> {
        let stages: &[usize] = if target.cp_async { &[3, 2] } else { &[2] };
        let cores: &[Core] = if target.group_mma.is_empty() { &[Core::Warp] } else { &[Core::Warp, Core::Group] };
        let mut out = vec![];
        for (core, bq) in cores.iter().flat_map(|&c| [16, 32, 64, 128, 256].map(|bq| (c, bq))) {
            // The warp-level core's blocks stop at eight row warps.
            if core == Core::Warp && bq > 128 {
                continue;
            }
            // A row warp of queries may share its block with key warps.
            let key_warps: &[usize] = if bq == WARP_ROWS && self.t <= WARP_ROWS { &[1, 2, 4] } else { &[1] };
            for bkv in [16, 32, 64, 128] {
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
    pub fn ranked(&self, planner: &Planner) -> Vec<FaCfg> {
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
            let regs = cfg.registers(target, self.d, self.bias)?;
            let o = target.occupancy;
            let per_simd = o.waves.min(o.file / regs.max(1)) as usize;
            Some((target.smem_bytes / smem.max(1)).min(o.simds as usize * per_simd / cfg.warps() as usize))
        };
        one.core == Core::Warp
            && one.bq >= TWO_ROWS_FROM
            && one.key_warps == 1
            && self.admit(planner, wide).is_ok()
            && resident(wide).zip(resident(one)).is_some_and(|(w, o)| w >= o)
    }
}

/// One warp per head row, the head split in halves a warp spreads over.
pub fn heads_candidates(d: usize) -> Vec<NormCfg> {
    if !(d.is_power_of_two() && (16..=256).contains(&d)) {
        return vec![];
    }
    [4, 8, 16].map(|br| NormCfg { br }).to_vec()
}

/// One warp per row of a power-of-two width a warp's vector loads span.
pub fn norm_candidates(d: usize) -> Vec<NormCfg> {
    if !(d.is_power_of_two() && (256..=2048).contains(&d)) {
        return vec![];
    }
    [4, 8, 16].map(|br| NormCfg { br }).to_vec()
}
