//! Tile generation: what a device can run, ranked by what it costs it.
//!
//! A tile table is a conclusion — the shape someone measured fastest on the
//! hardware they had. This module holds the reasoning instead, so a device
//! nobody measured, a shape outside the table, or a second kernel with its own
//! economics does not need a table of its own.
//!
//! Three limits decide which tiles a device can run at all, and every one of
//! them is something the backend reports ([`crate::target::workgroup_limits`]):
//! the threads a workgroup may launch, the shared memory it may take, and the
//! registers a lane holds. A fourth decides whether a tile is worth running:
//! the grid has to cover the compute units, and enough blocks have to stay
//! resident on each to hide the memory latency.
//!
//! Among the tiles that fit, two ratios order them, and both are per MAC — the
//! work is fixed, so only what surrounds it can be cheaper:
//!
//! * **operand traffic**, `(block_m + block_n) / (block_m · block_n)`. A square
//!   tile moves the least memory per MAC, which is why the widest tile wins
//!   whenever its grid still covers the device.
//! * **trip overhead**, `1 / (block_n · k_step)` for a kernel that pays per K
//!   trip. This is the term a tile table cannot express, because it is not a
//!   property of the device at all — it is a property of the kernel reading it.
//!   The NT GEMM does not pay it: a strip row is a base pointer and a stride.
//!   The implicit-GEMM convolution does: it rebuilds every strip row's source
//!   index on every trip, decoding an output pixel and a tap before it can
//!   gather the row. Halving the trips halves that work per MAC, which is why
//!   the convolution wants a strip twice as deep as the GEMM's on the same
//!   hardware.
//!
//! The ranking is not asked to pick the winner. It is asked to put the winner
//! in a handful of candidates that [`crate::tune`] then measures on the real
//! operands, because the last of the cost — cache behaviour, the compiler's
//! register allocation — is not analytic and should not be guessed.

use std::ops::Not;

use smallvec::SmallVec;

use super::gemm::GemmCfg;
use crate::target::WorkgroupLimits;

/// What one K trip costs a kernel beyond the MACs of the trip itself.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum TripCost {
    /// A strip row is a base pointer and a stride, so the trip count barely
    /// shows: the plain NT GEMM.
    Free,
    /// Every strip row's source index is rebuilt on the trip — the
    /// implicit-GEMM convolution decodes an output pixel and a tap out of
    /// `(pid_m, r, tile)` before it can gather the row.
    PerStripRow,
}

/// What a device allows one workgroup, and how many it wants in flight.
#[derive(Clone, Copy, Debug)]
pub struct TileBudget {
    /// Lanes per wave; the launch block is `warps · wave_size`.
    pub wave_size: usize,
    /// The reported per-workgroup and per-compute-unit limits.
    pub limits: WorkgroupLimits,
    /// The matrix core's edge: every accumulator side and `k_step` is a
    /// multiple of it.
    pub mma_edge: usize,
    /// Workgroups the grid must reach for the device to be covered at all. A
    /// tile whose grid is shorter leaves compute units idle, which no amount of
    /// operand reuse makes up for; past it, residency is what matters and
    /// [`TileBudget::resident`] prices that.
    pub blocks_wanted: usize,
    /// Blocks a compute unit should keep resident. A tile whose shared memory
    /// leaves fewer has nothing to hide its own memory latency behind.
    pub resident_floor: usize,
}

/// Registers a lane may spend on accumulators, of the ones it holds.
///
/// The rest go to the addresses, the staged strip and whatever the epilogue
/// needs, which is the kernel's business and not something a tile knows. Half
/// is what Ampere measures: a 32-accumulator `64x64` tile allocates 94-134 of
/// the 128 registers per lane its residency target leaves it, so the half not
/// spent on accumulators is fully used and sometimes spills.
const ACCUMULATOR_SHARE: usize = 2;

/// Registers a lane can address at all. Every matrix-core part tk targets
/// encodes its register operands in 8 bits, so a lane never holds more than
/// this however much the compute unit's file would leave it.
const REGISTERS_PER_LANE_MAX: usize = 255;

/// Register file a lane gets when the device does not report one — AMD
/// publishes its LDS through KFD but not its register file. Chosen as the
/// smallest budget any matrix-core part offers, so an unreported device is
/// bounded by the tiles that fit everywhere rather than by a guess.
const REGISTERS_PER_LANE_FLOOR: usize = 128;

/// What one strip-row index decode costs against one operand element of
/// traffic, for a kernel that pays per trip.
///
/// This is the one number here that measurement fixes rather than the device
/// reports: a decode is a handful of integer divisions and a bounds test, and
/// what it is worth depends on how much integer throughput the part has spare.
/// Calibrated on an RTX 3060 against the nine convolutions YOLO26-x tunes,
/// where a 64-deep strip beats a 32-deep one by 33-65%. The ranking only has to
/// put the winner among the few candidates the tuner then measures, so it is a
/// weight, not a threshold, and it does not have to be exact.
const DECODE_WEIGHT: f64 = 64.0;

/// Blocks a compute unit must keep resident for a tile to be worth launching
/// at all: below two there is nothing to overlap one block's memory latency
/// with, whatever the tile saves on traffic.
const RESIDENT_FLOOR: usize = 2;

/// The axes a search moves a tile along, each as the pair that reads and
/// writes it. The pipeline depth, the operand order and the swizzle are not
/// here: they are properties of the kernel, not of how it is tiled.
type Axis = (fn(&GemmCfg) -> usize, fn(&mut GemmCfg, usize));
const AXES: [Axis; 6] = [
    (|cfg| cfg.block_m, |cfg, v| cfg.block_m = v),
    (|cfg| cfg.block_n, |cfg, v| cfg.block_n = v),
    (|cfg| cfg.k_step, |cfg, v| cfg.k_step = v),
    (|cfg| cfg.warps_m, |cfg, v| cfg.warps_m = v),
    (|cfg| cfg.warps_n, |cfg, v| cfg.warps_n = v),
    (|cfg| cfg.acc_m, |cfg, v| cfg.acc_m = v),
];

/// The block edges worth trying, coarsest last: every matrix core in tk tiles a
/// 16-wide fragment, so these are 2-16 fragments a side.
const EDGES: [usize; 4] = [32, 64, 128, 256];
/// Wave-grid sides. Past four waves a side the block outgrows the tile it has
/// to divide.
const WAVES: [usize; 3] = [1, 2, 4];
/// Accumulators a wave stacks along M.
const ACC_M: [usize; 2] = [1, 2];
/// `k_step` in matrix-core edges: a strip one to eight fragments deep.
const STRIP_FRAGMENTS: [usize; 4] = [1, 2, 4, 8];
/// Shared-memory stages the kernels implement (the double-buffered pipeline).
const STAGES: usize = 2;

impl TileBudget {
    /// The budget of the device behind `spec`, when the backend reports its
    /// limits and the arch has a matrix core to tile for.
    pub fn for_device(spec: &svod_dtype::DeviceSpec, arch: svod_dtype::GpuArch) -> Option<Self> {
        let caps = crate::ArchCaps::for_arch(arch);
        let frag = caps.frag(crate::arch::FragRole::Accumulator)?;
        let compute_units = crate::target::compute_units(spec)?;
        Some(Self {
            wave_size: caps.wave_size,
            limits: crate::target::workgroup_limits(spec)?,
            mma_edge: frag.base.rows,
            blocks_wanted: compute_units,
            resident_floor: RESIDENT_FLOOR,
        })
    }

    /// Registers one lane holds at the residency target: the compute unit's
    /// file divided between the threads resident on it.
    fn registers_per_lane(&self, threads: usize) -> usize {
        self.limits
            .registers_per_cu
            .map(|regs| (regs / (self.resident_floor * threads).max(1)).min(REGISTERS_PER_LANE_MAX))
            .unwrap_or(REGISTERS_PER_LANE_FLOOR)
    }

    /// The tiles one step from `cfg` on each axis, keeping only those this
    /// device can launch and `accept` takes.
    ///
    /// A step doubles or halves one axis, because every axis of the lattice is a
    /// power of two. The neighbourhood is small and the lattice connected, which
    /// is what lets a beam reach the whole space from any seed.
    pub fn neighbours(
        &self,
        cfg: &GemmCfg,
        in_bytes: usize,
        accept: impl Fn(&GemmCfg) -> bool,
    ) -> SmallVec<[GemmCfg; 12]> {
        let mut out: SmallVec<[GemmCfg; 12]> = SmallVec::new();
        for (read, write) in AXES {
            for next in [read(cfg) * 2, read(cfg) / 2] {
                let mut moved = *cfg;
                write(&mut moved, next);
                let runnable = next > 0 && self.well_formed(&moved) && self.fits(&moved, in_bytes);
                if runnable && accept(&moved) && !out.contains(&moved) {
                    out.push(moved);
                }
            }
        }
        out
    }

    /// Accumulators one lane holds for `cfg` — the wave grid's whole effect on
    /// register pressure, and the tie-break between the grids that split a tile.
    ///
    /// Spreading a tile over more waves gives each lane fewer accumulators and
    /// more room for the addressing the kernel needs beside them. Where the cost
    /// cannot tell two grids apart, the one that leaves that room wins: the
    /// register cliff is what the cost is blind to, and this is the one lever
    /// over it a tile has.
    fn accumulators(&self, cfg: &GemmCfg) -> usize {
        cfg.acc_m * cfg.reg_m() * cfg.reg_n() / self.wave_size
    }

    /// Blocks of `cfg` a compute unit keeps resident — whichever of its shared
    /// memory and its registers runs out first.
    fn resident(&self, cfg: &GemmCfg, in_bytes: usize) -> usize {
        let threads = (cfg.warps_m * cfg.warps_n) * self.wave_size;
        let by_shared = self.limits.shared_per_cu / cfg.shared_bytes(in_bytes).max(1);
        let Some(registers) = self.limits.registers_per_cu else { return by_shared };
        let accumulators = self.accumulators(cfg);
        let by_registers = registers / (threads * accumulators * ACCUMULATOR_SHARE).max(1);
        by_shared.min(by_registers)
    }

    /// Whether `cfg` is one this device can launch: the threads, the shared
    /// memory, the accumulators a lane holds, and the blocks that leaves
    /// resident on a compute unit.
    pub fn fits(&self, cfg: &GemmCfg, in_bytes: usize) -> bool {
        let threads = (cfg.warps_m * cfg.warps_n) * self.wave_size;
        let shared = cfg.shared_bytes(in_bytes);
        let accumulators = self.accumulators(cfg);
        threads <= self.limits.max_threads
            && shared > 0
            && shared <= self.limits.shared_per_workgroup
            && accumulators <= self.registers_per_lane(threads) / ACCUMULATOR_SHARE
            && self.resident(cfg, in_bytes) >= self.resident_floor
    }

    /// What `cfg` costs this device per MAC of an `m x n` output. Lower is
    /// better; `None` when the grid does not cover the device, which is a
    /// different kind of worse and ordered after every covering tile.
    ///
    /// Two terms, both per MAC because the MACs are fixed and only what
    /// surrounds them can be cheaper. Residency does not appear: it is a floor
    /// the tile either clears or does not ([`Self::fits`]), and pricing it on a
    /// slope above that floor only produced ties that resolved toward the
    /// smallest tile.
    fn cost(&self, cfg: &GemmCfg, trip: TripCost, (m, n): (usize, usize)) -> (bool, f64) {
        let (block_m, block_n) = (cfg.block_m as f64, cfg.block_n as f64);
        let traffic = (block_m + block_n) / (block_m * block_n);
        let decode = match trip {
            TripCost::Free => 0.0,
            TripCost::PerStripRow => 1.0 / (block_n * cfg.k_step as f64),
        };
        (cfg.blocks(m, n) < self.blocks_wanted, traffic + DECODE_WEIGHT * decode)
    }

    /// Every tile this device can launch for an `m x n` output that `accept`
    /// also takes, cheapest first, at most `limit` of them.
    ///
    /// `accept` is the kernel's own tiling rule — which dimensions it needs to
    /// divide, which features it cannot carry — and it is applied after the
    /// device's limits so a kernel never sees a tile the hardware would refuse.
    pub fn ranked(
        &self,
        base: &GemmCfg,
        in_bytes: usize,
        trip: TripCost,
        (m, n): (usize, usize),
        limit: usize,
        accept: impl Fn(&GemmCfg) -> bool,
    ) -> SmallVec<[GemmCfg; 8]> {
        let mut found: Vec<GemmCfg> = Vec::new();
        for &block_m in &EDGES {
            for &block_n in &EDGES {
                for &warps_m in &WAVES {
                    for &warps_n in &WAVES {
                        for &acc_m in &ACC_M {
                            for &fragments in &STRIP_FRAGMENTS {
                                let cfg = GemmCfg {
                                    block_m,
                                    block_n,
                                    warps_m,
                                    warps_n,
                                    acc_m,
                                    k_step: fragments * self.mma_edge,
                                    stages: STAGES,
                                    split_k: 1,
                                    ..*base
                                };
                                if self.well_formed(&cfg) && self.fits(&cfg, in_bytes) && accept(&cfg) {
                                    found.push(cfg);
                                }
                            }
                        }
                    }
                }
            }
        }
        found.sort_by(|a, b| {
            let (ca, cb) = (self.cost(a, trip, (m, n)), self.cost(b, trip, (m, n)));
            ca.0.cmp(&cb.0).then(ca.1.total_cmp(&cb.1)).then_with(|| self.accumulators(a).cmp(&self.accumulators(b)))
        });
        // One entry per tile a launch can tell apart: the wave grids that split
        // the same tile differ only in how its accumulators are spread, and
        // spending the tuner's budget on all of them crowds out a real rival.
        let mut seen = Vec::new();
        found.retain(|cfg| {
            let tile = (cfg.block_m, cfg.block_n, cfg.k_step);
            seen.contains(&tile).not().then(|| seen.push(tile)).is_some()
        });
        self.spread(found, limit)
    }

    /// Take `limit` tiles off a ranked list, the best of each strip depth and
    /// register class first.
    ///
    /// The cost this module can compute sees the operand traffic and the trip
    /// count; it does not see where the compiler starts spilling, which is a
    /// step and not a slope and depends on addressing the tile knows nothing
    /// about. Measurement sees it exactly — `scratch_bytes` comes back with
    /// every compiled candidate — so the set handed to it has to *span* the
    /// axes the cost is blind along rather than rank confidently within them:
    /// the depth of the strip, and the accumulators a lane holds for it.
    fn spread(&self, ranked: Vec<GemmCfg>, limit: usize) -> SmallVec<[GemmCfg; 8]> {
        let class = |cfg: &GemmCfg| (cfg.k_step, self.accumulators(cfg));
        let mut out: SmallVec<[GemmCfg; 8]> = SmallVec::new();
        let mut taken: Vec<(usize, usize)> = Vec::new();
        for cfg in &ranked {
            if out.len() == limit {
                break;
            }
            if !taken.contains(&class(cfg)) {
                taken.push(class(cfg));
                out.push(*cfg);
            }
        }
        // A device with few classes leaves room; fill it with the next best.
        for cfg in ranked {
            if out.len() == limit {
                break;
            }
            if !out.contains(&cfg) {
                out.push(cfg);
            }
        }
        out
    }

    /// Whether the wave grid divides the tile into whole matrix-core fragments.
    fn well_formed(&self, cfg: &GemmCfg) -> bool {
        let rows = cfg.warps_m * cfg.acc_m;
        cfg.block_m.is_multiple_of(rows)
            && cfg.block_n.is_multiple_of(cfg.warps_n)
            && cfg.reg_m().is_multiple_of(self.mma_edge)
            && cfg.reg_n().is_multiple_of(self.mma_edge)
    }
}

/// The searched axes of a tile, packed into one integer so a measured winner
/// keeps in [`crate::tune::TuneStore`] beside the index-keyed entries.
///
/// Every axis is a power of two, so its exponent fits in four bits and the six
/// of them in 24. Anything outside the lattice — the operand order, the swizzle,
/// the pipeline depth — is not searched and comes back from the base tile.
pub fn pack(cfg: &GemmCfg) -> usize {
    let field = |value: usize| value.trailing_zeros() as usize & 0xf;
    field(cfg.block_m)
        | field(cfg.block_n) << 4
        | field(cfg.k_step) << 8
        | field(cfg.warps_m) << 12
        | field(cfg.warps_n) << 16
        | field(cfg.acc_m) << 20
}

/// [`pack`]'s inverse, over the unsearched fields of `base`.
pub fn unpack(base: &GemmCfg, bits: usize) -> GemmCfg {
    let field = |shift: usize| 1usize << ((bits >> shift) & 0xf);
    GemmCfg {
        block_m: field(0),
        block_n: field(4),
        k_step: field(8),
        warps_m: field(12),
        warps_n: field(16),
        acc_m: field(20),
        ..*base
    }
}
