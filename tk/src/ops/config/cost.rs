//! The shared cost core: what one run of a config costs on a device — the
//! pipes of a trip overlapped over the waves a SIMD holds, and the price of
//! a fill's bytes by where they come from — the [`Claim`] vocabulary a
//! family states a config's resources in, the one pricing path
//! ([`Planner::price`]) that turns a claim into time, and the [`Planner`]
//! that ranks every family's problems with it.

use std::sync::Arc;

use crate::atoms::{Device, Fill, Target};
use crate::kernels::attention::FaCfg;
use crate::kernels::conv::ConvCfg;
use crate::kernels::gemm::GemmCfg;
use crate::kernels::nearest::NearestCfg;

use super::attention::AttentionProblem;
use super::gemm::GemmProblem;
use super::nearest::NearestProblem;
use super::search::{Rejected, Tiled};

/// Model: the elements of one 16×16×16 matrix-core step, [`Costs::step`]'s
/// unit.
pub(crate) const STEP: usize = 16 * 16 * 16;
/// The bytes of a kernel's 16-bit operand element.
pub(super) const ELEM: usize = 2;

/// Model: how far past the cheapest config's cost the model cannot tell
/// configs apart: on sm_86 the measured best of Nemotron's projections and of 4096³
/// sat 12–31% behind the model's first.
pub(super) const BAND: f64 = 1.35;
/// Model: the waves per SIMD past which a trip's pipes overlap no further
/// (measured on gfx1201 and sm_86: sixteen waves per SIMD run a tile no
/// better than four, and the ranking that counted them did put sixteen-warp
/// blocks first on sm_86 where eight-warp ones run 7% faster).
const OVERLAP_WAVES: f64 = 4.0;

/// The closed ops vocabulary: the kinds of vector work a family may claim,
/// each an elementary operation of [`pair_op`]'s unit and scaled by its
/// [`Weights`] multiplier. A family cannot price vector work the vocabulary
/// has no kind for — a new kind is a cost-model change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Kind {
    /// A pair's score epilogue and its elementwise companions: the bundle
    /// itself ([`PAIR_OPS`]: a softmax element's scale, row maximum,
    /// subtract, `exp2` and row sum, against the argmin's score, row
    /// minimum, compare, select and second minimum) and every further
    /// operation a pair carries — the row's norm added, a floor's maximum,
    /// the L2 clamp, what groups add (a compare and a select), the 8-bit
    /// products' widening and scales. A softmax element of attention's is
    /// the same bundle.
    Pair,
    /// The listing's per-trip work ([`super::nearest`]): a deal's least an
    /// element, a trip's entry and its votes (a row's tests and the
    /// branch), a pushed deal's index and order, a round's dealt column and
    /// its fixed work, a queue slot's pushes.
    Deal,
    /// A bitonic network's work: a queue's sort, a merge into a list, a
    /// flush into the outputs — one exchange a slot a stage.
    Merge,
    /// The score-matrix path ([`super::nearest`]): a stored pair (its score,
    /// without the row's least, and its store's issue), a listed score's
    /// scan, a skipped step, a queue insert.
    Scan,
    /// A tile epilogue's operations over the tile's elements (a gelu's
    /// forty: the abs, the affines, the reciprocal, the polynomial's
    /// multiplies, the square, the `exp2`, the tails), beside the steps in
    /// every trip.
    Epilogue,
}

/// Model: a pair's score epilogue as elementary operations — the scale, the
/// row minimum, the argmin's compare, select and second minimum: as many
/// vector operations, two of them reductions, as a softmax element's scale,
/// row maximum, subtract, `exp2` and row sum ([`crate::atoms::Costs::score`]'s
/// own shape), which is why a pair bundle and a softmax element cost the
/// same. The warps a trip lowers some row of run them all; the model charges
/// every pair's.
pub(crate) const PAIR_OPS: f64 = 5.0;

/// Model: one elementary vector operation of an epilogue — the row's norm
/// added, a floor's maximum, the L2 clamp, a deal's least or vote, a merge's
/// compare — in units of [`crate::atoms::Costs::score`], the whole
/// [`crate::atoms::Costs::score`] a softmax element's bundle of them. A
/// declared residual, not a hand value kept: two measured derivations were
/// tried against the rank harness on fresh stores (16384² nearest at `k`
/// 1, 16, 64, depths 128 and 768, three cards) and refuted — the score and
/// step probes' ratio (`score/step`, sm_86 0.0097, gfx1201 0.0214 score
/// units) prices the listing's chained work a fifth to a twentieth of its
/// calibration, collapsing the queue-depth structure (gfx1151 prunes every
/// 32-deep queue; a gfx1201 `k` 64 tune picks a queued list 7 % behind the
/// register one) and dropping sm_86's fresh-store nearest ρ 0.415 → 0.381;
/// the probe bundle's own counted op rate (`score/` its ~15 scalar
/// operations) collapses it outright. The residual is the issue efficiency
/// of chained vector work beside the matrix pipe, which no probe explains:
/// the per-card fit from the store's own outcomes (the design's layer 2) is
/// what replaces this value, regularized toward it.
pub(crate) fn pair_op(_: &Device) -> f64 {
    0.2
}

/// The ops kinds' multipliers, every one at its full weight by default: the
/// tests and the rank harness perturb a kind alone (zeroed, to attribute
/// the model's rank error to its cluster) and nothing else — the default of
/// one everywhere is today's arithmetic exactly.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Weights {
    pub(crate) pair: f64,
    pub(crate) deal: f64,
    pub(crate) merge: f64,
    pub(crate) scan: f64,
    pub(crate) epilogue: f64,
}

impl Weights {
    /// Today's arithmetic: every kind at its full weight.
    pub(crate) fn same() -> Self {
        Self { pair: 1.0, deal: 1.0, merge: 1.0, scan: 1.0, epilogue: 1.0 }
    }

    /// The `kind`'s multiplier.
    pub(crate) fn of(&self, kind: Kind) -> f64 {
        match kind {
            Kind::Pair => self.pair,
            Kind::Deal => self.deal,
            Kind::Merge => self.merge,
            Kind::Scan => self.scan,
            Kind::Epilogue => self.epilogue,
        }
    }
}

/// Vector operations by [`Kind`]: a family accumulates its algorithm's
/// counts, the model prices each kind at [`pair_op`]'s unit. All counts are
/// of elementary operations ([`PAIR_OPS`] of them a pair's bundle).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Ops {
    pub(crate) pair: f64,
    pub(crate) deal: f64,
    pub(crate) merge: f64,
    pub(crate) scan: f64,
    pub(crate) epilogue: f64,
}

impl Ops {
    /// `count` operations of one `kind`.
    pub(crate) fn of(kind: Kind, count: f64) -> Self {
        Self::default().add(kind, count)
    }

    /// `count` operations of `kind` more.
    pub(crate) fn add(self, kind: Kind, count: f64) -> Self {
        let Ops { pair, deal, merge, scan, epilogue } = self;
        let (pair, deal, merge, scan, epilogue) = match kind {
            Kind::Pair => (pair + count, deal, merge, scan, epilogue),
            Kind::Deal => (pair, deal + count, merge, scan, epilogue),
            Kind::Merge => (pair, deal, merge + count, scan, epilogue),
            Kind::Scan => (pair, deal, merge, scan + count, epilogue),
            Kind::Epilogue => (pair, deal, merge, scan, epilogue + count),
        };
        Self { pair, deal, merge, scan, epilogue }
    }

    /// Another claim's counts beside these.
    pub(crate) fn plus(self, other: Self) -> Self {
        let Ops { pair, deal, merge, scan, epilogue } = other;
        self.add(Kind::Pair, pair)
            .add(Kind::Deal, deal)
            .add(Kind::Merge, merge)
            .add(Kind::Scan, scan)
            .add(Kind::Epilogue, epilogue)
    }

    /// Every count times `by`.
    pub(crate) fn scaled(self, by: f64) -> Self {
        let Ops { pair, deal, merge, scan, epilogue } = self;
        Self { pair: pair * by, deal: deal * by, merge: merge * by, scan: scan * by, epilogue: epilogue * by }
    }

    /// The counts as ns: each kind at `unit` ns an elementary operation
    /// ([`pair_op`]) times its weight, summed over the kinds. `score` at 1.0
    /// gives the counts in score elements — the claim diff's predicted VALU.
    pub(crate) fn fold(self, unit: f64, score: f64, w: &Weights) -> f64 {
        [
            (Kind::Pair, self.pair),
            (Kind::Deal, self.deal),
            (Kind::Merge, self.merge),
            (Kind::Scan, self.scan),
            (Kind::Epilogue, self.epilogue),
        ]
        .into_iter()
        .map(|(kind, count)| count * unit * w.of(kind) * score)
        .sum()
    }
}

/// A trip's matrix-core steps, by the core that issues them: the probe the
/// model prices them at and the ratio of waves a SIMD needs to hide their
/// dependent chain.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Steps {
    /// Warp-core 16-bit steps ([`crate::atoms::Costs::step`], hidden at
    /// `step_lone / step` waves a SIMD).
    F16(f64),
    /// Warpgroup steps over shared descriptors
    /// ([`crate::atoms::Costs::step_group`]).
    Group(f64),
    /// 8-bit integer steps ([`crate::atoms::Costs::step_i8`], hidden at
    /// their own lone ratio where it was measured: the chain's latency does
    /// not scale with its step's throughput).
    I8(f64),
}

impl Steps {
    /// The step count, in [`STEP`]-elements.
    pub(crate) fn count(&self) -> f64 {
        match *self {
            Steps::F16(n) | Steps::Group(n) | Steps::I8(n) => n,
        }
    }
}

/// A ring's fill: where the bytes a trip streams in come from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Stream {
    /// The bytes a trip fills into its ring.
    pub bytes: f64,
    /// The distinct bytes over the whole problem: the working set the cache
    /// ladder prices the re-reads at, whichever cache levels answer it.
    pub unique: f64,
    /// The share of a trip's bytes the blocks in flight re-read from the
    /// cache in front of DRAM (one another's lines).
    pub shared: f64,
}

/// A whole-problem section beside the run's trips: a kernel of its own (a
/// split's merge, a score matrix's listing), spread over the device's SMs —
/// every row the longer of its work and its bytes, at least `waits` memory
/// latencies (a share's loads wait on the share before them). A section of
/// no rows prices at nothing.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Spread {
    /// The rows the section walks.
    pub rows: f64,
    /// A row's vector operations, by kind.
    pub ops: Ops,
    /// A row's bytes straight from DRAM.
    pub bytes: f64,
    /// Memory latencies the section waits at least.
    pub waits: f64,
}

/// One config's claim on a device: the resources its kernel uses, in the
/// closed vocabulary the cost model prices. The only interface between a
/// family and [`Planner::price`] — the one place a claim becomes time — so
/// a family cannot add a term without extending the vocabulary here.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Claim {
    /// The blocks of the grid.
    pub blocks: usize,
    /// The shared bytes a block declares: what the program will declare,
    /// exact.
    pub smem: usize,
    /// The registers a lane holds by the layout count, which over-counts
    /// what the lowering streams.
    pub regs: u32,
    /// The waves a block holds.
    pub waves: usize,
    /// The independent accumulators a wave holds between dependent steps.
    pub atoms: usize,
    /// A trip's matrix-core steps.
    pub steps: Steps,
    /// The trips a block runs.
    pub trips: usize,
    /// A trip's operand-fragment bytes read from shared memory.
    pub reads: f64,
    /// A trip's vector work, by kind.
    pub ops: Ops,
    /// A trip's ring fill.
    pub fill: Stream,
    /// The bytes a trip writes straight to DRAM (a score matrix's stores),
    /// priced at the global rate whatever the cache holds.
    pub stores: f64,
    /// The ring's depth and unrolling: its trip overhead and how far its
    /// prefetch runs.
    pub ring: usize,
    pub unroll: bool,
    /// Whole-ring restarts outside the trips, in memory latencies: a
    /// streamed tile's ring starts empty at every tile.
    pub waits: f64,
    /// Whole-problem sections beside the run (at most two: a score
    /// matrix's two scans, or its listing alone; a split's merge).
    pub sections: [Spread; 2],
}

/// The safety divisor of [`Planner::floor`]: how many times below its bytes'
/// measured pace a candidate may measure before the tuner refuses the number.
/// Measured over every timed candidate of the three cards' stores and the
/// local one on 2026-10-28 (23851 candidates): the least honest one reads
/// 1.83x its floor, the k-nearest pick whose stamps landed off its kernel
/// 0.20x — the floor sits between, half the bytes' pace.
pub(crate) const FLOOR: f64 = 2.0;

impl Claim {
    /// The bytes the kernel must move at least once: every distinct byte of
    /// its streams, its straight-to-DRAM stores and its whole-problem
    /// sections' — the work no config of the problem can skip, whatever it
    /// caches or overlaps.
    pub(crate) fn distinct(&self) -> f64 {
        self.fill.unique
            + self.stores * self.trips as f64 * self.blocks as f64
            + self.sections.iter().map(|s| s.bytes * s.rows).sum::<f64>()
    }
}

/// A kernel family behind one interface: the configs it spans on a target
/// and the resources each claims — everything the planner needs of a family
/// beyond its configs' own tiles ([`Tiled`]), so that a family cannot price
/// what the vocabulary does not carry: poisoning is closed at the type
/// level, and a new family is an impl, not a cost-model change. Unstable:
/// publication (external families) is the design's phase F. The planner's
/// per-family caches and the op layer's dispatch stay concrete at the
/// boundary — an op builds tensors around its family's types either way —
/// and the norms rank nothing (a fixed candidate list the tuner times).
pub(crate) trait Family {
    /// The configs the family ranks.
    type Cfg: Tiled;

    /// Every config the family spans on `target`: its lattice, the search
    /// bounds' shape.
    fn lattice(&self, target: &Target) -> Vec<Self::Cfg>;

    /// One config's [resources](Claim) on the planner's device: the checks
    /// first — why it is left out — then what it uses.
    fn claim(&self, planner: &Planner, cfg: &Self::Cfg) -> Result<Claim, Rejected>;
}

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
    pub(super) conv: papaya::HashMap<super::conv::ConvProblem, Arc<[ConvCfg]>>,
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
            conv: papaya::HashMap::new(),
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

    /// The SM time of `claim`'s run — the only place a claim becomes time:
    /// the steps at their core's probe with the chain ratio that hides
    /// them, the fragment reads at the shared rate, the ops at [`pair_op`]'s
    /// unit by kind, the fill through the cache ladder, the trip at its
    /// ring's probe — pipelined over the waves a SIMD holds — and the waits
    /// and the sections beside the run's trips.
    ///
    /// The pipelining: the busiest SM issues the work of `⌈blocks / sms⌉`
    /// blocks, each taken up as another ends (measured on sm_86: a 64×64
    /// GEMM's time steps with that count, not with whole rounds of resident
    /// blocks), every block-trip paying a trip's fixed overhead at the
    /// blocks the SM runs at once ([`crate::atoms::Costs::trip_at`]: the
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
    pub(crate) fn price(&self, claim: &Claim) -> f64 {
        let Device { target, costs } = &self.device;
        let hide = match claim.steps {
            // The 8-bit chain hides with its own lone ratio where its lone
            // step was measured, not the 16-bit one: the chain's latency
            // does not scale with its step's throughput.
            Steps::I8(_) if costs.step_lone_i8 > 0.0 && costs.step_i8 > 0.0 => {
                (costs.step_lone_i8 / costs.step_i8).max(1.0)
            }
            _ => costs.hide(),
        };
        let work = match claim.steps {
            Steps::F16(n) => n * costs.step,
            Steps::Group(n) => n * costs.step_group,
            Steps::I8(n) => n * costs.step_i8,
        };
        let (reads, epilogue) =
            (claim.reads * costs.smem_byte, claim.ops.fold(pair_op(&self.device), costs.score, &self.weights));
        let fill = claim.fill.bytes
            * self.fill_byte(
                claim.fill.unique,
                (claim.blocks * claim.trips) as f64 * claim.fill.bytes,
                claim.fill.shared,
            )
            + claim.stores * costs.global_byte;
        let trip = costs.trip(target.fill, claim.ring, claim.unroll);
        // The run the claim describes: `blocks` blocks of `waves` waves,
        // `trips` trips through a ring `stages` deep, a trip's issue down
        // three pipes in ns of SM time each plus its fixed `trip` overhead.
        let (blocks, waves, stages) = (claim.blocks, claim.waves, claim.ring);
        let (o, sms) = (target.occupancy, self.sms());
        // At least one: the compiled kernel, not the layout count, decides
        // whether a block runs.
        let resident = self.resident(claim.smem, claim.regs, waves).max(1);
        let per_sm = blocks.div_ceil(sms);
        let rounds = per_sm.div_ceil(resident);
        let hide = hide.max(1.0);
        let per_simd = |waves: usize| (waves as f64 / o.simds as f64).max(1.0);
        let lone = |waves: usize| (hide / (per_simd(waves) * claim.atoms.max(1) as f64)).max(1.0);
        let serial = stages <= 1 && target.fill == Fill::CpAsync;
        // `alone`: one block's pipes, its fill after the rest where serial.
        let body = |waves: usize, alone: bool| {
            let fill = if alone && serial { (0.0, fill) } else { (fill, 0.0) };
            let pipes = [reads, work * lone(waves) + epilogue, fill.0];
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
        let run = claim.trips as f64 * (issue.max(rounds as f64 * wave) + rounds as f64 * wait);
        let section = |s: &Spread| {
            let work = s.ops.fold(pair_op(&self.device), costs.score, &self.weights);
            let bytes = s.bytes * costs.global_byte;
            (s.rows * work.max(bytes) / self.sms() as f64).max(s.waits * costs.latency)
        };
        run + claim.waits * costs.latency + claim.sections.iter().map(section).sum::<f64>()
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

    /// What `cfg` costs the planner's device — ns of SM time — and a
    /// lane's registers by the layout count: the one admission path, a
    /// [`Family`]'s claim priced; `None` with why the config is left out.
    pub(crate) fn admit<P: Family>(&self, problem: &P, cfg: &P::Cfg) -> Result<(f64, u32), Rejected> {
        let claim = problem.claim(self, cfg)?;
        Ok((self.price(&claim), claim.regs))
    }

    /// The least wall ns `cfg`'s kernel can honestly take
    /// ([`Planner::floor_of`] of its claim): a measured time under it is a
    /// measurement that lied, never a fast kernel.
    pub(crate) fn floor_of<P: Family>(&self, problem: &P, cfg: &P::Cfg) -> Option<f64> {
        Some(self.floor(&problem.claim(self, cfg).ok()?))
    }

    /// The least wall ns a claim's kernel can take: the bytes it must move
    /// at the card's measured rate for a stream of that many distinct bytes,
    /// every SM streaming at once, [`FLOOR`] times below that.
    pub(crate) fn floor(&self, claim: &Claim) -> f64 {
        let bytes = claim.distinct();
        bytes * self.device.costs.rate(bytes) / (self.sms() as f64 * FLOOR)
    }

    /// What a byte of a fill costs when `fill` bytes stream `unique`
    /// distinct ones, of which the blocks in flight at once re-read a
    /// `shared` part from one another's lines. Those come from the cache in
    /// front of DRAM at its measured rate; of the rest, each distinct byte
    /// comes from DRAM once and its re-reads at the cache ladder's measured
    /// rate for a working set of `unique` bytes — the rate a stream of that
    /// size runs at, whichever cache levels answer it (RDNA4's L2 and
    /// Infinity Cache alike) and wherever its knee sits, no hand share of
    /// the cache assumed. DRAM's rate where no ladder was measured.
    pub(crate) fn fill_byte(&self, unique: f64, fill: f64, shared: f64) -> f64 {
        let c = &self.device.costs;
        if c.l2_bytes <= 0.0 || fill <= 0.0 {
            return c.global_byte;
        }
        let (shared, reread) = (shared.clamp(0.0, 1.0), (1.0 - unique / fill).max(0.0));
        let stream = reread * c.rate(unique) + (1.0 - reread) * c.global_byte;
        shared * c.l2_byte + (1.0 - shared) * stream
    }
}
