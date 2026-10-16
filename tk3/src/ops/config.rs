//! Tile config candidates per target and shape, most promising first. The
//! device picks: the tune store times the first [`SHORTLIST`] tiles that
//! compile clean, most of the lattice, and the ranking orders what is
//! compiled and timed first and stands in for a measurement where there is
//! no device (the first config runs untuned). GEMM-shaped kernels draw
//! theirs from a lattice over the construct's knobs, kept to what the
//! lowering runs and ranked in nanoseconds of SM time from the device's
//! measured [`Costs`]: a round of resident blocks is bound by the SM's issue
//! work, by one wave's own issue or by the memory latency one pipeline stage
//! ahead cannot hide ([`Planner::pipelined`]). The model orders block tiles
//! well enough to lead with them ([`spread`]) and warp grids poorly (on
//! sm_86 the measured best sat as deep as the 92nd of 109 tiles), which is
//! why the shortlist runs that deep. Registers admit nothing here: the
//! layout count over-counts, so the compiled kernel judges ([`shortlist`]).
//! A [`Planner`] holds the rankings of one device. The other kernels' lists
//! are small functions of the target.

use std::collections::HashSet;
use std::hash::Hash;
use std::sync::Arc;

use svod_dtype::GpuArch;

use crate::atoms::{Device, Target};
use crate::ir::{Axis, Orient, Shape};
use crate::kernels::attention::{FaCfg, WARP_ROWS};
use crate::kernels::conv::{ConvCfg, ConvGeom};
use crate::kernels::gemm::GemmCfg;
use crate::kernels::rows::NormCfg;
use crate::layouts::{WarpGrid, chunked};
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
    /// Whether the warp grid tiles the output with `target`'s matrix core
    /// atoms and the ring depth is one the target's prefetch runs.
    fn tiles(&self, target: &Target) -> bool {
        let Some(atom) = target.mma.first() else { return false };
        let ([bm, bn, bk], [wr, wc]) = (self.tile, self.warps.map(|w| w as usize));
        bm.is_multiple_of(atom.m as usize * wr)
            && bn.is_multiple_of(atom.n as usize * wc)
            && bk.is_multiple_of(atom.k as usize)
            && target.stages(self.stages) == self.stages
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
        let atom = target.mma.first()?;
        let ([bm, bn, bk], [wr, wc]) = (self.tile, self.warps);
        let issue = atom.issue(Orient::Direct, WarpGrid { rows: wr, cols: wc }, bm, bn, bk).ok()?;
        let halves = halves as u32;
        let acc = issue.c.regs() * halves;
        let operands = (issue.a.regs() + issue.b.regs() * halves).div_ceil(2);
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
    /// fragments: the f32 accumulators and one matrix-core depth of operand
    /// fragments, a floor of [`Self::registers`]. Past the lane's limit the
    /// kernel cannot compile without spilling.
    pub fn live_registers(&self, target: &Target, halves: usize) -> Option<u32> {
        let atom = target.mma.first()?;
        let ([bm, bn, _], [wr, wc]) = (self.tile, self.warps);
        let issue = atom.issue(Orient::Direct, WarpGrid { rows: wr, cols: wc }, bm, bn, atom.k as usize).ok()?;
        let halves = halves as u32;
        Some(issue.c.regs() * halves + (issue.a.regs() + issue.b.regs() * halves).div_ceil(2))
    }
}

pub(crate) const fn gemm_cfg(tile: [usize; 3], stages: usize, warps: [u32; 2], unroll: bool) -> GemmCfg {
    GemmCfg { tile, stages, warps, group_m: 8, unroll }
}

/// The knobs a GEMM-shaped kernel is enumerated over: output tile edges
/// (48 and 96 columns for the convolution bodies a power of two would pad
/// by a third), reduction depths and warp grids.
const ROWS: [usize; 4] = [32, 64, 128, 256];
const COLS: [usize; 5] = [48, 64, 96, 128, 256];
const DEPTHS: [usize; 3] = [16, 32, 64];
const GRIDS: [[u32; 2]; 9] = [[1, 1], [1, 2], [2, 1], [2, 2], [1, 4], [4, 1], [2, 4], [4, 2], [4, 4]];

/// Every config the construct spans on `target`: the tiles and warp grids,
/// the ring depths its prefetch runs, rolled and unrolled.
fn lattice(target: &Target) -> Vec<GemmCfg> {
    let stages: &[usize] = if target.cp_async { &[3, 2] } else { &[2] };
    let mut out = vec![];
    for bm in ROWS {
        for bn in COLS {
            for bk in DEPTHS {
                for warps in GRIDS {
                    for &stages in stages {
                        out.extend([false, true].map(|unroll| gemm_cfg([bm, bn, bk], stages, warps, unroll)));
                    }
                }
            }
        }
    }
    out
}

/// How a problem runs under one config: `blocks` blocks of `waves` waves,
/// each holding `smem` bytes of shared memory and `regs` registers a lane
/// by the layout count, and running `trips` trips of `work` ns of SM time
/// plus a trip's fixed overhead `trip` (its barrier and ring bubble, which
/// the other blocks an SM holds work through) through a ring `stages` deep.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Run {
    pub blocks: usize,
    pub smem: usize,
    pub regs: u32,
    pub waves: usize,
    pub trips: usize,
    pub work: f64,
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
    /// registers](GemmCfg::live_registers) alone pass the lane's limit. A trip
    /// pays its fragment reads, fill, matrix-core steps (padding included)
    /// and overhead.
    pub(crate) fn admit(&self, planner: &Planner, cfg: &GemmCfg) -> Option<(f64, u32)> {
        let Device { target, costs } = &planner.device;
        let ([bm, bn, bk], [wr, wc]) = (cfg.tile, cfg.warps.map(|w| w as usize));
        let gated = self.halves == 2;
        // A step is at least two matrix-core depths, unless the reduction dim
        // allows nothing deeper.
        let depth = 2 * target.mma.first()?.k as usize;
        let deep = bk >= depth || !self.k.is_multiple_of(depth);
        let smem = cfg.smem_bytes(gated);
        if !(deep && self.k.is_multiple_of(bk) && cfg.fits(target) && smem <= target.smem_bytes) {
            return None;
        }
        let regs = cfg.registers(target, self.halves)?;
        if cfg.live_registers(target, self.halves)? > target.occupancy.registers {
            return None;
        }
        let (halves, waves) = (self.halves, wr * wc);
        let fragments = waves * (bm / wr + bn * halves / wc) * bk * 2;
        let fill = (bm + bn * halves) * bk * 2;
        let steps = bm * bn * halves * bk / 4096;
        let (blocks, trips, stages) = (self.blocks(cfg), self.k / bk, cfg.stages);
        let unique = (self.batches * (self.m + self.n * halves) * self.k * 2) as f64;
        let fill_byte = planner.fill_byte(unique, (blocks * trips * fill) as f64);
        let work = fragments as f64 * costs.smem_byte + fill as f64 * fill_byte + steps as f64 * costs.step;
        let trip = costs.trip(!cfg.unroll && !target.cp_async);
        Some((planner.pipelined(Run { blocks, smem, regs, waves, trips, work, trip, stages }), regs))
    }

    /// The lattice's configs that run this problem on the planner's target,
    /// every pipeline (depth, ring depth, unrolling) of every tile, cheapest
    /// first (then by fewer registers) and [`spread`] over the block tiles
    /// the cost band holds: the shortlist keeps a tile's first clean one
    /// ([`shortlist`]).
    pub fn ranked(&self, planner: &Planner) -> Vec<GemmCfg> {
        let mut ranked: Vec<(f64, u32, GemmCfg)> = lattice(&planner.device.target)
            .into_iter()
            .filter_map(|c| self.admit(planner, &c).map(|(cost, regs)| (cost, regs, c)))
            .collect();
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        spread(ranked.into_iter().map(|(cost, _, c)| (cost, c)).collect(), |c| [c.tile[0], c.tile[1]])
    }
}

/// How far past the cheapest config's cost the model cannot tell configs
/// apart: on sm_86 the measured best of Nemotron's projections and of 4096³
/// sat 12–31% behind the model's first.
const BAND: f64 = 1.35;

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

/// The tiles the tune store measures per shape, compiled in parallel
/// batches of as many: most of the lattice, since timing a tile costs
/// about 20 ms once the device's clock is up and the ranking cannot be
/// trusted to put the best within a few (see the module docs).
pub(crate) const SHORTLIST: usize = 64;
/// Blocks per SM a convolution's grid should reach before its reduction is
/// left whole: its grids are a few hundred blocks.
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

    fn twins(&self, ranked: &[Self]) -> Vec<Self> {
        self.gemm
            .twins(&ranked.iter().map(|c| c.gemm).collect::<Vec<_>>())
            .into_iter()
            .map(|gemm| ConvCfg { gemm, ..*self })
            .collect()
    }
}

/// The blocks, the splits and the softmax forms: a forms pair is two tiles.
/// A tile is its query and key block sizes and its key warps: a different
/// number of key warps is a different kernel structure, not a pipeline of
/// the same one.
impl Tiled for FaCfg {
    fn tile(&self) -> Self {
        FaCfg { stages: 0, unroll: false, ..*self }
    }

    fn twins(&self, ranked: &[Self]) -> Vec<Self> {
        let block = |c: &FaCfg| (c.bq, c.bkv, c.key_warps);
        ranked
            .iter()
            .filter(|c| block(c) == block(self))
            .map(|c| FaCfg { stages: c.stages, unroll: c.unroll, ..*self })
            .collect()
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
    judged(ranked, SHORTLIST, compile, |c| tiles.insert(c.tile()))
}

/// The indices of every config of `list` whose compiled kernel does not
/// spill, judged as [`shortlist`] judges (every one spilling: all of them;
/// nothing compiling: `None`) but with none set aside for sharing a tile:
/// a tile's [twins](Tiled::twins) are judged so.
pub(crate) fn clean<C: Tiled>(list: &[C], compile: impl FnMut(&[C]) -> Vec<Compiled>) -> Option<Vec<usize>> {
    judged(list, list.len(), compile, |_| true)
}

fn judged<C: Tiled>(
    ranked: &[C],
    limit: usize,
    mut compile: impl FnMut(&[C]) -> Vec<Compiled>,
    mut keep: impl FnMut(&C) -> bool,
) -> Option<Vec<usize>> {
    let (mut clean, mut spilled) = (vec![], vec![]);
    for (n, batch) in ranked.chunks(SHORTLIST).enumerate() {
        for (i, compiled) in compile(batch).into_iter().enumerate() {
            let at = n * SHORTLIST + i;
            match compiled {
                Compiled::Clean if keep(&batch[i]) => clean.push(at),
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
#[derive(Debug)]
pub struct Planner {
    pub device: Device,
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
        Self { device, ranked: papaya::HashMap::new(), attention: papaya::HashMap::new() }
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
    /// estimate, decides whether a block runs). A round of those blocks is at
    /// least one wave's own issue (a block of fewer waves than SIMDs leaves
    /// the rest idle) and the memory latency its ring cannot hide (it
    /// prefetches `stages - 1` trips ahead).
    pub(crate) fn pipelined(&self, run: Run) -> f64 {
        let Run { blocks, smem, regs, waves, trips, work, trip, stages } = run;
        let Device { target, costs } = &self.device;
        let (o, sms) = (target.occupancy, self.sms());
        let per_simd = o.waves.min(o.file / regs.max(1)) as usize;
        let resident = (target.smem_bytes / smem).min(o.simds as usize * per_simd / waves).max(1);
        let per_sm = blocks.div_ceil(sms);
        let rounds = per_sm.div_ceil(resident);
        let issue = per_sm as f64 * (work + trip / resident.min(per_sm) as f64);
        let wave = o.simds as f64 * (work + trip) / waves as f64;
        let wait = costs.latency / stages.saturating_sub(1).max(1) as f64;
        trips as f64 * issue.max(rounds as f64 * wave.max(wait))
    }

    /// What a byte of a fill costs when `fill` bytes stream `unique`
    /// distinct ones. Where the distinct bytes fit three quarters of the
    /// cache in front of DRAM (the rest left to other traffic) each comes
    /// from DRAM once and its re-reads from the cache; past half again as
    /// much every byte comes from DRAM; between, the two blend.
    pub(crate) fn fill_byte(&self, unique: f64, fill: f64) -> f64 {
        let c = &self.device.costs;
        let held = 0.75 * c.l2_bytes;
        if held <= 0.0 || fill <= 0.0 {
            return c.global_byte;
        }
        let fits = ((1.5 * held - unique) / (0.5 * held)).clamp(0.0, 1.0);
        let hit = fits * (1.0 - unique / fill).max(0.0);
        hit * c.l2_byte + (1.0 - hit) * c.global_byte
    }

    /// Every attention config for `tiles` rows of `t` queries over `tk`
    /// keys at head size `d`, the cheapest first.
    pub fn attention_ranked(&self, tiles: usize, t: usize, tk: usize, d: usize, causal: bool) -> Arc<[FaCfg]> {
        if tiles * t * tk * d == 0 {
            return Arc::new([]);
        }
        let p = Attention { tiles, t, tk, d, causal };
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
    /// is the kernel's [`WARP_ROWS`] (one warp per that many query rows),
    /// both products tile the atoms, and the K and V fills split into whole
    /// 16-byte chunks per thread.
    pub fn fits(&self, target: &Target, d: usize) -> bool {
        let Some(atom) = target.mma.first() else { return false };
        let (bq, bkv) = (self.bq, self.bkv);
        // Key warps fill their own tiles alone.
        let threads = self.row_warps() as usize * target.wave as usize;
        atom.m as usize == WARP_ROWS
            && bq.is_multiple_of(WARP_ROWS)
            && bq > 0
            && (self.key_warps == 1 || bq == WARP_ROWS)
            && [bkv, d].iter().all(|&x| x.is_multiple_of(atom.n as usize) && x.is_multiple_of(atom.k as usize))
            && (bkv * d / 8).is_multiple_of(threads)
            && target.stages(self.stages) == self.stages
    }

    /// The registers a lane holds for head size `d`, from the layouts the
    /// lowering assigns: the query fragments and the f32 output, live across
    /// the key loop; the key fragments, the f32 scores, the probabilities and
    /// the value fragments, all live together while a block's scores are
    /// turned into probabilities and multiplied in; and the staged K and V
    /// chunks where the fill goes through registers (the value tile is
    /// stored column-major where it is staged, so its fragments gather as
    /// 16-byte runs like the keys'). Addressing and what the compiler keeps
    /// beside the tiles are not counted: the compiled kernel's own count is
    /// what the tuner reads.
    pub fn registers(&self, target: &Target, d: usize) -> Option<u32> {
        let atom = target.mma.first()?;
        let grid = WarpGrid { rows: self.row_warps(), cols: 1 };
        let qk = atom.issue(Orient::Direct, grid, self.bq, self.bkv, d).ok()?;
        let pv = atom.issue(Orient::Direct, grid, self.bq, d, self.bkv).ok()?;
        let half = |regs: u32| regs.div_ceil(2);
        let fill = if target.cp_async {
            0
        } else {
            let tile = chunked(Shape::new(self.bkv, d), 2, self.row_warps(), target.wave, Axis::Col)?;
            2 * half(tile.regs())
        };
        let (q, o) = (half(qk.a.regs()), pv.c.regs());
        let (k, s, p, v) = (half(qk.b.regs()), qk.c.regs(), half(pv.a.regs()), half(pv.b.regs()));
        Some(q + o + k + s + p + v + fill)
    }
}

/// Trips a key block on a causal query block's diagonal counts for: the
/// block itself, its mask, and the rows above the diagonal every warp still
/// runs. A count of the work, not a fit.
const DIAGONAL: usize = 3;

/// An attention problem: `tiles` independent query rows of `t` queries over
/// `tk` keys at head size `d` (`tiles` = batch × heads), `causal` when a
/// query sees the keys up to its own position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Attention {
    pub tiles: usize,
    pub t: usize,
    pub tk: usize,
    pub d: usize,
    pub causal: bool,
}

impl Attention {
    fn blocks(&self, cfg: &FaCfg) -> usize {
        self.tiles * self.t.div_ceil(cfg.bq)
    }

    /// What `cfg` costs here, as [`Problem::admit`]: a key block's trip pays
    /// both products' fragment reads, fills and steps, the softmax of every
    /// score and its overhead. A causal query block scores half the keys plus
    /// its diagonal, a key block per `bkv` of its rows, each [`DIAGONAL`] trips.
    /// Key warps each take a share of the trips, every trip of theirs the
    /// block's whole work at once.
    pub(crate) fn admit(&self, planner: &Planner, cfg: &FaCfg) -> Option<(f64, u32)> {
        let Device { target, costs } = &planner.device;
        let (bq, bkv, d, kw) = (cfg.bq, cfg.bkv, self.d, cfg.key_warps);
        let smem = cfg.smem_bytes(d) + cfg.scratch_bytes(target);
        if !(cfg.fits(target, d) && smem <= target.smem_bytes) {
            return None;
        }
        let regs = cfg.registers(target, d)?;
        let waves = cfg.warps() as usize;
        let fragments = 2 * waves * bkv * d * 2;
        let fill = kw * 2 * bkv * d * 2;
        let steps = kw * 2 * bq * bkv * d / 4096;
        let work = fragments as f64 * costs.smem_byte
            + fill as f64 * costs.global_byte
            + steps as f64 * costs.step
            + (kw * bq * bkv) as f64 * costs.score;
        let trip = costs.trip(!cfg.unroll && !target.cp_async);
        let diagonal = DIAGONAL * bq.div_ceil(bkv);
        let trips = if self.causal { self.tk.div_ceil(2 * bkv) + diagonal } else { self.tk.div_ceil(bkv) };
        let trips = trips.div_ceil(kw);
        let (blocks, stages) = (self.blocks(cfg), cfg.stages);
        Some((planner.pipelined(Run { blocks, smem, regs, waves, trips, work, trip, stages }), regs))
    }

    /// The lattice's configs that run this problem on the planner's target,
    /// cheapest first, then by fewer registers, every pipeline of every query
    /// and key block size, in the plain softmax forms (the tuner sets the
    /// others); the shortlist keeps a tile's first clean one.
    pub fn ranked(&self, planner: &Planner) -> Vec<FaCfg> {
        let stages: &[usize] = if planner.device.target.cp_async { &[3, 2] } else { &[2] };
        let mut ranked: Vec<(f64, u32, FaCfg)> = vec![];
        for bq in [16, 32, 64, 128] {
            // A row warp of queries may share its block with key warps.
            let key_warps: &[usize] = if bq == WARP_ROWS && self.t <= WARP_ROWS { &[1, 2, 4] } else { &[1] };
            for bkv in [16, 32, 64, 128] {
                for &stages in stages {
                    for unroll in [false, true] {
                        for &key_warps in key_warps {
                            let cfg = FaCfg { unroll, key_warps, ..FaCfg::new(bq, bkv, stages) };
                            if let Some((cost, regs)) = self.admit(planner, &cfg) {
                                ranked.push((cost, regs, cfg));
                            }
                        }
                    }
                }
            }
        }
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        ranked.into_iter().map(|(_, _, c)| c).collect()
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
