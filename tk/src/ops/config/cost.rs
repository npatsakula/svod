//! The shared cost core: what one run of a config costs on a device — the
//! pipes of a trip overlapped over the waves a SIMD holds, and the price of
//! a fill's bytes by where they come from — and the [`Planner`] that ranks
//! every family's problems with it.

use std::sync::Arc;

use crate::atoms::{Device, Fill, Target};
use crate::kernels::attention::FaCfg;
use crate::kernels::gemm::GemmCfg;
use crate::kernels::nearest::NearestCfg;

use super::attention::AttentionProblem;
use super::gemm::GemmProblem;
use super::nearest::{NearestProblem, Weights};

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

/// Model: the elements of one 16×16×16 matrix-core step, [`Costs::step`]'s
/// unit.
pub(super) const STEP: usize = 16 * 16 * 16;
/// The bytes of a kernel's 16-bit operand element.
pub(super) const ELEM: usize = 2;

/// Model: how far past the cheapest config's cost the model cannot tell
/// configs apart: on sm_86 the measured best of Nemotron's projections and of 4096³
/// sat 12–31% behind the model's first.
pub(super) const BAND: f64 = 1.35;
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

/// What the op layer plans with on one device: its target, what its
/// primitives cost, and the problems ranked for it so far. An op plans its
/// shape again at every call, and laying out the lattice's feasible configs
/// takes a millisecond, so a ranking is kept for the planner's life (one per
/// device). The rankings are whole: the tuner admits the first
/// [`SHORTLIST`](super::search::SHORTLIST) whose compiled kernels do not
/// spill ([`shortlist`](super::search::shortlist)).
/// A planner needs no device: one over a card's offline target
/// ([`Target::offline`]) and the costs its device stored
/// ([`crate::probe::stored`]) ranks as the card's own would; only the op
/// layer's ([`crate::ops::supported`]) launches.
#[derive(Debug)]
pub struct Planner {
    pub device: Device,
    /// The device's SM count.
    pub(super) sms: usize,
    /// The cost fields the device's probes could not measure, ranked at
    /// zero ([`crate::probe::Measured`]); empty for a device that measured.
    pub unmeasured: Vec<&'static str>,
    /// The nearest epilogue's cost clusters, every one at its full weight
    /// (the tests and the rank harness perturb them alone).
    pub(crate) weights: Weights,
    pub(super) ranked: papaya::HashMap<GemmProblem, Arc<[GemmCfg]>>,
    pub(super) attention: papaya::HashMap<AttentionProblem, Arc<[FaCfg]>>,
    pub(super) nearest: papaya::HashMap<NearestProblem, Arc<[NearestCfg]>>,
}

/// The ranking kept under `key`, made by `rank` the first time it is asked for.
pub(super) fn cached<K: std::hash::Hash + Eq + Clone, V>(
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
            weights: Weights::same(),
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
}
