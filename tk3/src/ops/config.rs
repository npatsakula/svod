//! Tile config candidates per target and shape, most promising first: the
//! first is what runs untuned, the tune store measures the rest. GEMM-shaped
//! kernels draw theirs from a lattice over the construct's knobs, kept to
//! what the lowering runs and ranked by a traffic model; a [`Planner`] holds
//! the rankings of one device. The other kernels' lists are small functions
//! of the target.

use std::collections::HashSet;
use std::sync::Arc;

use svod_dtype::GpuArch;

use crate::atoms::Target;
use crate::ir::{Axis, Orient, Shape};
use crate::kernels::attention::FaCfg;
use crate::kernels::conv::{ConvCfg, ConvGeom};
use crate::kernels::gemm::GemmCfg;
use crate::kernels::rows::NormCfg;
use crate::layouts::{WarpGrid, chunked};

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
}

const fn gemm_cfg(tile: [usize; 3], stages: usize, warps: [u32; 2], unroll: bool) -> GemmCfg {
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

/// The SM count assumed when the device reports none.
const DEFAULT_SMS: u32 = 32;
/// Waves per SM past which more hide no more of the latency between a
/// block's trips: the stall a trip leaves is weighed against the waves
/// around it.
const HIDING_WAVES: f64 = 16.0;
/// A rolled register-staged ring recomputes its slot offsets every trip
/// where an unrolled one has immediates.
const ROLLED: f64 = 1.1;
/// What a trip costs besides its traffic, as bytes of shared-memory
/// traffic: `trip` for the trip itself (the barrier, the ring's bubble) and
/// `wave_trip` per wave and per 16-bit element of reduction depth (each
/// wave's barrier arrival and fragment issue); `step` per matrix-core step
/// (a 16×16×16 product), charging a padded tile for the work it wastes; and
/// `hiding`, the waves per SM past which more hide no more of the latency
/// between a block's trips.
struct Costs {
    trip: usize,
    wave_trip: usize,
    step: usize,
    hiding: f64,
}

impl Costs {
    /// The constants of `target`'s family. The `cp.async` path's are fitted
    /// on sm_86 against full-lattice sweeps of YOLO's convolution bodies and
    /// wide-M GEMMs (the measured best of every shortlist within 7% of the
    /// lattice's best, 15% with a fixed trip cost); the register-staged
    /// path's are the gfx1201 fit: about 256 SM cycles per trip and eight
    /// per step at 128 B/cycle.
    fn of(target: &Target) -> Self {
        if target.cp_async {
            Self { trip: 0, wave_trip: 128, step: 2048, hiding: 16.0 }
        } else {
            Self { trip: TRIP, wave_trip: 0, step: STEP, hiding: HIDING_WAVES }
        }
    }
}

/// What a trip costs besides its traffic (the barrier, the ring's bubble),
/// as bytes of shared-memory traffic: about 256 SM cycles. Attention keeps
/// these constants on every target.
const TRIP: usize = 32 << 10;
/// What one matrix-core step (a 16×16×16 product) costs, in the same bytes:
/// its eight SM cycles at the 128 B/cycle of shared memory. It charges a
/// padded tile for the work it wastes.
const STEP: usize = 1024;

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

    /// What `cfg` costs here on `target`, for ranking, and a lane's
    /// registers; `None` when it does not run. The cost is the bytes a
    /// block's trip moves through shared memory (every wave's operand
    /// fragments in, the fill out, [`TRIP`] for the trip itself and [`STEP`]
    /// per matrix-core step) over the trips of every round of blocks the
    /// device runs, stretched by how few waves an SM holds, and how few
    /// steps its ring keeps in flight, to hide one trip's latency behind
    /// another's. The matrix-core work is the same per output whatever the
    /// tile, so the traffic is what tells tiles apart, and the SM count, the
    /// shared memory and the register file decide how many blocks a round
    /// holds.
    fn admit(&self, target: &Target, cfg: &GemmCfg) -> Option<(f64, u32)> {
        let ([bm, bn, bk], [wr, wc]) = (cfg.tile, cfg.warps.map(|w| w as usize));
        let gated = self.halves == 2;
        // A trip's products cover its own fragment loads only when a step is at
        // least two matrix-core depths (measured: one-depth steps run a third
        // slower at best), unless the reduction dim allows nothing deeper.
        let depth = 2 * target.mma.first()?.k as usize;
        let deep = bk >= depth || !self.k.is_multiple_of(depth);
        if !(deep && self.k.is_multiple_of(bk) && cfg.fits(target) && cfg.smem_bytes(gated) <= target.smem_bytes) {
            return None;
        }
        let regs = cfg.registers(target, self.halves).filter(|&r| r <= target.occupancy.registers)?;
        let occupancy = target.occupancy;
        let waves = wr * wc;
        let per_simd = occupancy.waves.min(occupancy.file / regs) as usize;
        let resident = (target.smem_bytes / cfg.smem_bytes(gated)).min(occupancy.simds as usize * per_simd / waves);
        if resident == 0 {
            return None;
        }
        let sms = target.sms.unwrap_or(DEFAULT_SMS) as usize;
        let blocks = self.blocks(cfg);
        let actual = resident.min(blocks.div_ceil(sms));
        let rounds = blocks.div_ceil(sms * actual);
        let steps = bm * bn * self.halves * bk / 4096;
        let costs = Costs::of(target);
        let traffic = (waves * (bm / wr + bn * self.halves / wc) + bm + bn * self.halves) * bk * 2
            + costs.trip
            + waves * bk * costs.wave_trip
            + steps * costs.step;
        let trips = self.k / bk;
        let hiding = 1.0 + costs.hiding / (actual * waves * (cfg.stages - 1)) as f64;
        let rolled = if cfg.unroll || target.cp_async { 1.0 } else { ROLLED };
        Some(((rounds * actual * trips * traffic) as f64 * hiding * rolled, regs))
    }

    /// The lattice's configs that run this problem on `target`, cheapest
    /// first, then by fewer registers, one pipeline (depth, ring depth,
    /// unrolling) per output tile and warp grid, so the list spans tiles
    /// rather than one tile's pipelines.
    pub fn ranked(&self, target: &Target) -> Vec<GemmCfg> {
        let mut ranked: Vec<(f64, u32, GemmCfg)> = lattice(target)
            .into_iter()
            .filter_map(|c| self.admit(target, &c).map(|(cost, regs)| (cost, regs, c)))
            .collect();
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut seen = HashSet::new();
        ranked.into_iter().map(|(_, _, c)| c).filter(|c| seen.insert(([c.tile[0], c.tile[1]], c.warps))).collect()
    }
}

/// The configs the tune store measures per shape.
const SHORTLIST: usize = 8;
/// Blocks per SM a convolution's grid should reach before its reduction is
/// left whole: its grids are a few hundred blocks.
const CONV_BLOCKS_PER_SM: usize = 2;

/// What the op layer plans with on one device: its target, and the
/// problems ranked for it so far. An op plans its shape again at every
/// call, and laying out the lattice's feasible configs takes a millisecond,
/// so a ranking is kept for the planner's life (one per device).
#[derive(Debug)]
pub struct Planner {
    pub target: Target,
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
    pub fn new(target: Target) -> Self {
        Self { target, ranked: papaya::HashMap::new(), attention: papaya::HashMap::new() }
    }

    /// At most [`SHORTLIST`] attention configs for `tiles` rows of `t`
    /// queries over `tk` keys at head size `d`, the cheapest first.
    pub fn attention_candidates(&self, tiles: usize, t: usize, tk: usize, d: usize, causal: bool) -> Vec<FaCfg> {
        if tiles * t * tk * d == 0 {
            return vec![];
        }
        let p = Attention { tiles, t, tk, d, causal };
        let ranked = cached(&self.attention, p, || p.ranked(&self.target));
        ranked.iter().take(SHORTLIST).copied().collect()
    }

    fn ranked(&self, p: Problem) -> Arc<[GemmCfg]> {
        cached(&self.ranked, p, || p.ranked(&self.target))
    }

    /// At most [`SHORTLIST`] GEMM configs for `batches` GEMMs of `m` rows,
    /// `n` columns and reduction dim `k`, the cheapest of the lattice first.
    pub fn gemm_candidates(&self, batches: usize, m: usize, n: usize, k: usize, gated: bool) -> Vec<GemmCfg> {
        if batches * m * n * k == 0 {
            return vec![];
        }
        let p = Problem { batches, m, n, k, halves: if gated { 2 } else { 1 } };
        self.ranked(p).iter().take(SHORTLIST).copied().collect()
    }

    /// Convolution configs for `batches` grid batches of `m` output pixels
    /// of `g` (`n = cout`, `k = kh·kw·cin`): the cheapest lattice configs
    /// whose step stays within one tap (`bk` divides `cin`) and whose tiles
    /// pad `m × n` by at most 1/8. On a starved grid (fewer than twice
    /// [`CONV_BLOCKS_PER_SM`] blocks per SM, a single static batch) the lead
    /// and the eligible configs with the most blocks at 1/16 and at 1/8
    /// padding also come split over the reduction, in counts dividing the
    /// steps that keep at most 16 blocks per SM. Open: the gathered A
    /// operand re-reads each input pixel per tap, so on stride-1 bodies
    /// tk1's image-staged (Patch) form is still ahead, 192→192 at 40² 54.3 µs
    /// against 61.2 on sm_86 (graph 86.6); the Patch form is pending.
    pub fn conv_candidates(&self, batches: usize, m: usize, g: &ConvGeom) -> Vec<ConvCfg> {
        let (n, k, cin) = (g.cout, g.k(), g.cin);
        if batches * m * n * k == 0 {
            return vec![];
        }
        let p = Problem { batches, m, n, k, halves: 1 };
        // A one-depth step gathers a whole trip per product: on the cp.async
        // path it runs behind the graph's conv (sm_86, 48→48 at 160²: 84 µs
        // against 57), so the conv takes the graph there.
        let depth = if self.target.cp_async { 2 * self.target.mma.first().map_or(16, |a| a.k as usize) } else { 0 };
        let eligible: Vec<GemmCfg> = self
            .ranked(p)
            .iter()
            .filter(|c| cin.is_multiple_of(c.tile[2]) && c.tile[2] >= depth && p.padding(c) <= 1.0 / 8.0)
            .copied()
            .collect();
        let Some(&base) = eligible.first() else { return vec![] };
        let mut out: Vec<ConvCfg> = eligible.iter().take(SHORTLIST).map(|&gemm| ConvCfg { gemm, split: 1 }).collect();
        let sms = self.target.sms.unwrap_or(DEFAULT_SMS) as usize;
        let blocks = |c: &GemmCfg| p.blocks(c);
        let widest = |tight: bool| {
            let pick = eligible.iter().filter(|c| !tight || p.padding(c) <= 1.0 / 16.0).max_by_key(|c| blocks(c));
            pick.copied().unwrap_or(base)
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
}

impl FaCfg {
    /// Whether the config lowers on `target` for head size `d`: one warp per
    /// 16 query rows, both products tile the atoms, and the K and V fills
    /// split into whole 16-byte chunks per thread.
    pub fn fits(&self, target: &Target, d: usize) -> bool {
        let Some(atom) = target.mma.first() else { return false };
        let (bq, bkv) = (self.bq, self.bkv);
        let threads = self.warps() as usize * target.wave as usize;
        bq.is_multiple_of(16)
            && bq > 0
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
    /// 16-byte runs like the keys').
    pub fn registers(&self, target: &Target, d: usize) -> Option<u32> {
        let atom = target.mma.first()?;
        let grid = WarpGrid { rows: self.warps(), cols: 1 };
        let qk = atom.issue(Orient::Direct, grid, self.bq, self.bkv, d).ok()?;
        let pv = atom.issue(Orient::Direct, grid, self.bq, d, self.bkv).ok()?;
        let half = |regs: u32| regs.div_ceil(2);
        let fill = if target.cp_async {
            0
        } else {
            let tile = chunked(Shape::new(self.bkv, d), 2, self.warps(), target.wave, Axis::Col)?;
            2 * half(tile.regs())
        };
        let (q, o) = (half(qk.a.regs()), pv.c.regs());
        let (k, s, p, v) = (half(qk.b.regs()), qk.c.regs(), half(pv.a.regs()), half(pv.b.regs()));
        Some(q + o + k + s + p + v + fill)
    }
}

/// What a score element costs beyond its products, in bytes of shared-memory
/// traffic: the softmax's scale, mask, max, exp2 and sum on the vector units.
const SCORE: usize = 32;
/// What a key block on a causal query block's diagonal costs in trips.
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

    /// What `cfg` costs here on `target`, for ranking, and a lane's
    /// registers; `None` when it does not run. A key block's trip moves
    /// every warp's K and V fragments in and the fill out, pays [`TRIP`],
    /// [`STEP`] per matrix-core step of both products and [`SCORE`] per
    /// score, over the key blocks of every round of query blocks the device
    /// runs, stretched by how few waves an SM holds and how few steps its
    /// ring keeps in flight. A warp past the problem's query rows hides
    /// nothing: it stalls on the loads its neighbours stall on and adds the
    /// work of its padded rows, so a decoder step takes the one-warp tile. A
    /// causal query block scores half the keys plus its diagonal, a key block
    /// per `bkv` of its own rows, each costing [`DIAGONAL`] trips: the mask,
    /// and the rows above the diagonal every warp still runs.
    fn admit(&self, target: &Target, cfg: &FaCfg) -> Option<(f64, u32)> {
        let (bq, bkv, d) = (cfg.bq, cfg.bkv, self.d);
        let smem = cfg.smem_bytes(d) + cfg.scratch_bytes(target);
        if !(cfg.fits(target, d) && smem <= target.smem_bytes) {
            return None;
        }
        let regs = cfg.registers(target, d).filter(|&r| r <= target.occupancy.registers)?;
        let occupancy = target.occupancy;
        let waves = cfg.warps() as usize;
        let per_simd = occupancy.waves.min(occupancy.file / regs) as usize;
        let resident = (target.smem_bytes / smem).min(occupancy.simds as usize * per_simd / waves);
        if resident == 0 {
            return None;
        }
        let sms = target.sms.unwrap_or(DEFAULT_SMS) as usize;
        let blocks = self.blocks(cfg);
        let actual = resident.min(blocks.div_ceil(sms));
        let rounds = blocks.div_ceil(sms * actual);
        let steps = 2 * bq * bkv * d / 4096;
        let traffic = (2 * waves + 2) * bkv * d * 2 + TRIP + steps * STEP + bq * bkv * SCORE;
        let diagonal = DIAGONAL * bq.div_ceil(bkv);
        let trips = if self.causal { self.tk.div_ceil(2 * bkv) + diagonal } else { self.tk.div_ceil(bkv) };
        let useful = waves.min(self.t.div_ceil(16));
        let hiding = 1.0 + HIDING_WAVES / (actual * useful * (cfg.stages - 1)) as f64;
        let rolled = if cfg.unroll || target.cp_async { 1.0 } else { ROLLED };
        Some(((rounds * actual * trips * traffic) as f64 * hiding * rolled, regs))
    }

    /// The lattice's configs that run this problem on `target`, cheapest
    /// first, then by fewer registers, one pipeline per query and key block
    /// size.
    pub fn ranked(&self, target: &Target) -> Vec<FaCfg> {
        let stages: &[usize] = if target.cp_async { &[3, 2] } else { &[2] };
        let mut ranked: Vec<(f64, u32, FaCfg)> = vec![];
        for bq in [16, 32, 64, 128] {
            for bkv in [16, 32, 64, 128] {
                for &stages in stages {
                    for unroll in [false, true] {
                        let cfg = FaCfg { unroll, ..FaCfg::new(bq, bkv, stages) };
                        if let Some((cost, regs)) = self.admit(target, &cfg) {
                            ranked.push((cost, regs, cfg));
                        }
                    }
                }
            }
        }
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut seen = HashSet::new();
        ranked.into_iter().map(|(_, _, c)| c).filter(|c| seen.insert((c.bq, c.bkv))).collect()
    }
}

/// Key splits worth measuring for `tiles` independent query tiles over
/// `blocks` key blocks: one, two, the counts around what keeps two blocks
/// per SM busy, and every block its own split when that is within reach
/// (the merge costs a launch, so never past the blocks). A target without
/// an SM count keeps one.
pub fn split_candidates(target: &Target, tiles: usize, blocks: usize) -> Vec<usize> {
    let Some(sms) = target.sms else { return vec![1] };
    let want = (2 * sms as usize).div_ceil(tiles.max(1));
    let mut out: Vec<usize> = [1, 2, want / 2, want, 2 * want, blocks]
        .into_iter()
        .filter(|&s| (1..=blocks.min(2 * want)).contains(&s))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// One warp per head row, the head split in halves a warp spreads over.
pub fn heads_candidates(_target: &Target, d: usize) -> Vec<NormCfg> {
    if !(d.is_power_of_two() && (16..=256).contains(&d)) {
        return vec![];
    }
    [4, 8, 16].map(|br| NormCfg { br }).to_vec()
}

/// One warp per row of a power-of-two width a warp's vector loads span.
pub fn norm_candidates(_target: &Target, d: usize) -> Vec<NormCfg> {
    if !(d.is_power_of_two() && (256..=2048).contains(&d)) {
        return vec![];
    }
    [4, 8, 16].map(|br| NormCfg { br }).to_vec()
}
