//! What a device's primitives cost, measured on it at first use and kept in
//! the tune store. Each cost is a pair of programs timed together: a probe
//! that does everything its baseline does plus `units` of the primitive, so
//! the difference is the primitive alone. A throughput probe fills the device
//! and its difference is scaled to one SM's time per unit. A relayout-class
//! probe (lane moves, shared round trips, scattered loads)
//! runs one wave per SIMD on a dependent chain, and `latency` one wave per
//! SM: their differences are what one wave waits per unit.

use snafu::{OptionExt, Snafu};
use svod_dtype::default_device::default_device;
use svod_dtype::{GpuArch, ScalarDType};

use crate::atoms::{Core, Costs, Device, Fill, Target};
use crate::build::*;
use crate::ir::*;
use crate::layout::{Dim, Layout};
use crate::layouts::{TileLayout, WarpGrid, chunked, natural};
use crate::lower::Lowering;
use crate::tune::{self, Built, Candidate, ScratchPool, TuneKey, TuneStore};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("{arch:?} reports no SM count, so a probe cannot scale its time to one SM"))]
    NoSms { arch: GpuArch },
    #[snafu(display("{probe} could not be timed on the device in {TRIES} tries; the last: {last}"))]
    Unmeasured { probe: &'static str, last: String },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Measurement: bytes the streaming probes read once per run: larger than
/// any device cache (MI300X's Infinity Cache is 256 MB), so every run streams from DRAM.
pub(crate) const STREAM: usize = 512 << 20;
/// Measurement: trips of a lone wave at most: a few ms of waiting.
const LATENCY_TRIPS: usize = 16 << 10;
/// Measurement: trips of the trip probes: enough that their fixed costs dwarf
/// a launch.
const TRIPS: usize = 4096;
/// Measurement: timings of a pair, compiled once, before a probe gives up: one with a
/// weak signal ([`strong`]) or a lost stamp is timed again, and so is one
/// that disagrees with the last ([`least`]).
const TRIES: usize = 3;
/// Measurement: how close two timings of a probe must be for it to settle
/// on the lesser.
const AGREE: f64 = 0.1;
/// Measurement: warps of a throughput probe's block.
const WARPS: u32 = 4;

/// One measurement: `candidates` are `[baseline, probe]`, the probe doing
/// `units` more of the primitive.
#[derive(Clone, Debug)]
pub(crate) struct Probe {
    /// `probe:<field>`, the tune store's op.
    pub(crate) op: &'static str,
    pub(crate) dtype: ScalarDType,
    pub(crate) candidates: Vec<Candidate>,
    pub(crate) units: f64,
    /// What the difference is scaled by before dividing by `units`: the SM
    /// count for a throughput probe, else 1.
    pub(crate) scale: f64,
    pub(crate) signal: Signal,
}

/// How a pair's signal (probe less baseline) is told from noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Signal {
    /// At least half the probe's own time ([`strong`]): a primitive the pair
    /// can make dominate its scaffolding.
    Dominant,
    /// Above the pair's own run-to-run spread ([`resolved`]): a primitive
    /// that rides on work it cannot shed, as a trip moves its tile's bytes,
    /// and is small next to it.
    Resolved,
}

/// Measurement: timings of a [`Signal::Resolved`] pair, compiled once: the
/// spread of five bounds the noise a difference must clear.
const RESOLVE_TRIES: usize = 5;
/// Measurement: the fewest pairs a difference is resolved from, a spread
/// left once their extremes are trimmed.
const RESOLVE_PAIRS: usize = 3;

/// A pair's difference and the noise it is told from, both in ns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Resolved {
    /// The median of the pairs' differences, probe less baseline.
    pub(crate) difference: f64,
    /// The run-to-run spread of the pairs' differences, its extremes
    /// trimmed.
    pub(crate) spread: f64,
    /// The median baseline and probe timings.
    pub(crate) baseline: f64,
    pub(crate) probe: f64,
}

impl Resolved {
    /// Whether the baseline did the work the probe rides on: a baseline
    /// under half its probe's time moved less than the bytes (its loads
    /// left the loop, or a constant index let the compiler fold them), and
    /// the difference would price them as the primitive.
    pub(crate) fn measurable(&self) -> bool {
        2.0 * self.baseline >= self.probe
    }

    pub(crate) fn clear(&self) -> bool {
        self.difference > self.spread
    }

    /// The difference where it clears the spread; else the spread, an upper
    /// bound on a primitive the pair cannot tell from zero.
    pub(crate) fn value(&self) -> f64 {
        if self.clear() { self.difference } else { self.spread }
    }
}

/// `pairs` (`[baseline, probe]` timings of one compiled pair) resolved;
/// `None` under [`RESOLVE_PAIRS`], where no spread can be judged. A probe that runs faster than its baseline adds
/// nothing the baseline's work did not cost (sm_120's `cp.async` ring moves
/// its bytes a few percent cheaper than the same bytes loaded into
/// registers), so its value is the spread, the bound on what it adds.
pub(crate) fn resolved(pairs: &[(u64, u64)]) -> Option<Resolved> {
    if pairs.len() < RESOLVE_PAIRS {
        return None;
    }
    let diffs: Vec<f64> = pairs.iter().map(|&(base, probe)| probe as f64 - base as f64).collect();
    // The differences' range with its extremes trimmed (a first run below
    // the lifted clock is noise). A pair's two
    // programs run back to back, so a clock that drifts between pairs moves
    // both sides and leaves the difference: on gfx1151 the sides' own
    // ranges summed 5–40× the differences' (283 µs against 34 µs on a
    // 226 µs trip) and called a clear trip unresolved.
    let spread = {
        let mut d = diffs.clone();
        d.sort_by(f64::total_cmp);
        let kept = &d[1..d.len() - 1];
        kept[kept.len() - 1] - kept[0]
    };
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied()
    };
    let difference = median(diffs)?;
    let baseline = median(pairs.iter().map(|p| p.0 as f64).collect())?;
    let probe = median(pairs.iter().map(|p| p.1 as f64).collect())?;
    Some(Resolved { difference, spread, baseline, probe })
}

/// [`RESOLVE_TRIES`] timings of a compiled pair; those that lost a stamp or
/// failed are left out.
pub(crate) fn resolve_pairs(built: &mut [Built]) -> Vec<(u64, u64)> {
    (0..RESOLVE_TRIES)
        .filter_map(|_| match tune::time(built)[..] {
            [Some(base), Some(probe)] => Some((base, probe)),
            _ => None,
        })
        .collect()
}

/// Whether a pair's signal (probe less baseline, ns) is at least half the
/// probe's own time: else the probe times its scaffolding more than its
/// primitive, and noise in either decides the value.
pub(crate) fn strong(base_ns: u64, probe_ns: u64) -> bool {
    probe_ns >= 2 * base_ns
}

/// A device's measured costs, and the fields its probes could not measure,
/// left at [`Costs::default`]'s zero: the planner ranks with what it has
/// rather than leaving the device to the graph, and the device gate holds
/// the list empty.
#[derive(Clone, Debug, PartialEq)]
pub struct Measured {
    pub costs: Costs,
    pub unmeasured: Vec<&'static str>,
}

/// The costs of the default device, whose target is `target`: every field
/// measured once per device, compiler and probe program, then read from the
/// tune store. Only a target without an SM count, which no probe can scale
/// to, is an error.
pub fn costs(target: &Target) -> Result<Measured> {
    let compiler = tune::compiler(&default_device());
    priced(target, |probe| self::measured(target, compiler, probe))
}

/// Every field of `target`'s costs, each probe's value by `measure`.
fn priced(target: &Target, mut measure: impl FnMut(Probe) -> Result<f64>) -> Result<Measured> {
    let mut measured = assemble(target, probes(target)?, &mut measure);
    let Measured { costs, unmeasured } = &mut measured;
    // The cache ladder is read against the DRAM rate; without one the cache
    // is unmeasured and fills are priced as DRAM.
    let ladder = if unmeasured.contains(&"global_byte") { None } else { cache(target, costs.global_byte, measure)? };
    match ladder {
        Some(rates) => {
            assert!(rates.len() <= costs.ladder.len(), "a ladder rung a footprint");
            costs.l2_byte = rates[0].1;
            costs.l2_bytes = held(&rates, costs.global_byte).1;
            for (slot, rate) in costs.ladder.iter_mut().zip(rates) {
                *slot = rate;
            }
        }
        None => {
            unmeasured.push("l2_bytes");
            (costs.l2_byte, costs.l2_bytes) = (costs.global_byte, 0.0);
        }
    }
    Ok(measured)
}

/// The costs `target`'s device measured into the tune store under `dir`,
/// one copied from it: read, never written. A probe's line counts whatever
/// compiler, store format or probe programs measured it (its value is ns
/// per unit of the device); of several, the one of this build's programs.
/// A probe no line settles is unmeasured. `None` where the store has no
/// file for the target's device ([`tune::device`]: its arch and SM count),
/// or the target no SM count.
pub fn stored(dir: &std::path::Path, target: &Target) -> Option<Measured> {
    let device = tune::device(target);
    let store = TuneStore::replay(dir.to_path_buf());
    let file = store.file(&device)?;
    let text = std::fs::read_to_string(&file).ok()?;
    // `value|op|device|dtype|shape|candidates|compiler|programs femtoseconds`
    let lines: Vec<(&str, &str, u64)> = text
        .lines()
        .filter_map(|line| {
            let (key, fs) = line.split_once(' ')?;
            let [kind, op, at, .., programs] = key.split('|').collect::<Vec<_>>()[..] else { return None };
            (kind == "value" && at == device).then_some((op, programs, fs.trim().parse().ok()?))
        })
        .collect();
    let value = |probe: Probe| {
        let found: Vec<&(&str, &str, u64)> = lines.iter().filter(|l| l.0 == probe.op).collect();
        let fs = match found[..] {
            [only] => Some(only.2),
            _ => {
                let programs = format!("{:016x}", store.fingerprint(&probe.candidates));
                found.iter().find(|l| l.1 == programs).map(|l| l.2)
            }
        };
        let last = format!("{} lines of it in {}, none of this build's programs", found.len(), file.display());
        Ok(fs.context(UnmeasuredSnafu { probe: probe.op, last })? as f64 * 1e-6)
    };
    priced(target, value).ok()
}

/// Model: costs a device is planned with where nothing measured it: guesses at
/// an sm_86-class part, the warpgroup step priced as the warp-level one. A
/// ranking under them is a guess's.
pub const GUESSED: Costs = Costs {
    smem_byte: 0.005,
    global_byte: 0.02,
    scatter: 8.0,
    step: 5.0,
    step_group: 5.0,
    // Twice the 16-bit rate, as most cores with an 8-bit one run it.
    step_i8: 2.5,
    // Three waves a SIMD hide the chain.
    step_lone: 15.0,
    // Unmeasured: the 16-bit ratio stands.
    step_lone_i8: 0.0,
    score: 0.5,
    trip: 150.0,
    trip_rolled: 170.0,
    // The barrier pair costs about what the two-slot trip does again.
    trip_one: 300.0,
    // Nothing guessed: a lone block's trip costs what the SM's full does.
    trip_lone: 0.0,
    trip_one_lone: 0.0,
    lane_move: 1.0,
    smem_trip: 8.0,
    latency: 600.0,
    // A one-slot ring waits out the whole load, half again a ring's leftover.
    latency_one: 900.0,
    l2_byte: 0.0,
    l2_bytes: 0.0,
    ladder: [(0, 0.0); crate::atoms::LADDER],
};

/// `probes`, in [`Costs::fields`] order, each measured by `measure` into its
/// field; a probe that fails leaves its field at zero, names itself in
/// `unmeasured`, and is warned about once.
pub(crate) fn assemble(target: &Target, probes: Vec<Probe>, mut measure: impl FnMut(Probe) -> Result<f64>) -> Measured {
    let mut costs = Costs::default();
    let fields = costs.fields();
    assert_eq!(fields.len(), probes.len(), "a probe per field");
    let mut unmeasured = vec![];
    for ((name, value), probe) in fields.into_iter().zip(probes) {
        assert_eq!(probe.op.strip_prefix("probe:"), Some(name), "probes in field order");
        // A primitive the target lacks costs nothing and is not unmeasured.
        if probe.candidates.is_empty() {
            continue;
        }
        match measure(probe) {
            Ok(v) => *value = v,
            Err(e) => {
                tracing::warn!("{}: {e}; {name} costs nothing in the ranking", target.arch.target_name());
                unmeasured.push(name);
            }
        }
    }
    Measured { costs, unmeasured }
}

/// A probe's value in ns per unit, stored in femtoseconds: a shared byte
/// costs a few picoseconds, which whole picoseconds would round by a third.
/// The pair compiles once and is timed until a timing holds.
fn measured(target: &Target, compiler: u64, probe: Probe) -> Result<f64> {
    let Probe { op, dtype, candidates, units, scale, signal } = probe;
    let programs = tune::fingerprint(&candidates);
    let key = TuneKey::new(op, target, compiler, dtype, &[], &programs);
    let mut last = String::from("not run");
    let fs = TuneStore::global().value(
        &key,
        || programs,
        || {
            let mut built = tune::compile(candidates, &ScratchPool::default());
            if signal == Signal::Resolved {
                let pairs = resolve_pairs(&mut built);
                last = format!("no difference resolved from {pairs:?}");
                let r = resolved(&pairs)?;
                tracing::debug!(op, ?pairs, ?r, "probe pair resolved");
                if !r.measurable() {
                    last = format!(
                        "a baseline of {} ns under half the probe's {} ns: it does not move the probe's bytes",
                        r.baseline, r.probe
                    );
                    return None;
                }
                return Some((r.value() * scale / units * 1e6).round() as u64);
            }
            least((0..TRIES).filter_map(|_| match timed(&mut built, scale / units) {
                Ok((fs, base_ns, probe_ns)) => {
                    tracing::debug!(op, base_ns, probe_ns, fs, "probe pair timed");
                    Some(fs)
                }
                Err(e) => {
                    last = e;
                    None
                }
            }))
        },
    );
    Ok(fs.context(UnmeasuredSnafu { probe: op, last })? as f64 * 1e-6)
}

/// The least of `timings`, taken until two agree within [`AGREE`]: a kernel
/// sharing the device with another's runs slow, never fast, so of two
/// timings apart the lesser is nearer the primitive and a third decides.
pub(crate) fn least(timings: impl Iterator<Item = u64>) -> Option<u64> {
    let mut best: Option<u64> = None;
    for t in timings {
        let settled = best.is_some_and(|b| b.abs_diff(t) as f64 <= AGREE * b.max(t) as f64);
        best = Some(best.map_or(t, |b| b.min(t)));
        if settled {
            break;
        }
    }
    best
}

/// One timing of a compiled pair: its difference times `per`, in
/// femtoseconds, with the pair's own nanoseconds, or why it does not hold.
fn timed(built: &mut [Built], per: f64) -> Result<(u64, u64, u64), String> {
    match tune::time(built)[..] {
        [Some(base_ns), Some(probe_ns)] if strong(base_ns, probe_ns) => {
            Ok((((probe_ns - base_ns) as f64 * per * 1e6).round() as u64, base_ns, probe_ns))
        }
        [Some(base_ns), Some(probe_ns)] => {
            let signal = (probe_ns as f64 - base_ns as f64) / probe_ns as f64;
            Err(format!("a signal of {signal:.3} of the probe's time (baseline {base_ns} ns, probe {probe_ns} ns)"))
        }
        _ => Err("a program failed to build or lost its stamps".into()),
    }
}

/// Measurement: the footprints the cache probes read [`STREAM`] bytes from,
/// smallest first: from the size of a small cache to half the stream, past
/// every device's cache (MI300X's Infinity Cache is 256 MB), so a probe
/// still reads each byte of a footprint twice.
const FOOTPRINTS: [(&str, usize); 9] = [
    ("probe:cached_1m", 1 << 20),
    ("probe:cached_2m", 2 << 20),
    ("probe:cached_4m", 4 << 20),
    ("probe:cached_8m", 8 << 20),
    ("probe:cached_16m", 16 << 20),
    ("probe:cached_32m", 32 << 20),
    ("probe:cached_64m", 64 << 20),
    ("probe:cached_128m", 128 << 20),
    ("probe:cached_256m", 256 << 20),
];

/// The cache ladder, as [`climbed`] reads the footprints; `None`, with a
/// warning, when no footprint can be measured.
fn cache(
    target: &Target,
    global_byte: f64,
    measure: impl FnMut(Probe) -> Result<f64>,
) -> Result<Option<Vec<(usize, f64)>>> {
    let ladder = climbed(cache_probes(target)?, global_byte, measure);
    Ok(ladder.map_err(|e| tracing::warn!("{}: {e}; the cache is unmeasured", target.arch.target_name())).ok())
}

/// The ladder of `probes` (smallest footprint first) measured until it has
/// [`settled`] past the cache: a small cache's ladder stops after a few
/// footprints, and no run streams a large one for nothing. A rung that
/// cannot be measured is skipped — the ladder keeps the rungs it has, and
/// fails only when it holds none.
pub(crate) fn climbed(
    probes: Vec<(usize, Probe)>,
    global_byte: f64,
    mut measure: impl FnMut(Probe) -> Result<f64>,
) -> Result<Vec<(usize, f64)>> {
    let mut rates = Vec::with_capacity(probes.len());
    for (footprint, probe) in probes {
        if settled(&rates, global_byte) {
            break;
        }
        match measure(probe) {
            Ok(rate) => rates.push((footprint, rate)),
            Err(e) => tracing::warn!("probe:cached: {e}; the ladder keeps the rungs it has"),
        }
    }
    (!rates.is_empty())
        .then_some(rates)
        .ok_or_else(|| Error::Unmeasured { probe: "probe:cached", last: "no rung measured".into() })
}

/// Whether the ladder's last two footprints both read nearer DRAM's rate
/// than the cache's: one footprint past the knee could be noise or a
/// second cache level; two are past the cache ([`held`] counts neither).
pub(crate) fn settled(rates: &[(usize, f64)], global_byte: f64) -> bool {
    rates.len() >= 2 && rates[rates.len() - 2..].iter().all(|(_, r)| *r > mid(rates, global_byte))
}

/// The rate between the cache's and DRAM's on a log scale.
fn mid(rates: &[(usize, f64)], global_byte: f64) -> f64 {
    (rates[0].1.min(global_byte) * global_byte).sqrt()
}

/// What a ladder of `(footprint, ns a byte)` says of the cache in front of
/// DRAM: a streamed byte's cost from the smallest footprint, which stays in
/// it, and its bytes, the most any footprint shows it answering: a
/// footprint read at a rate between the cache's and DRAM's was answered
/// for that share of its bytes (the rates mix linearly), so the cache holds
/// about the share times the footprint; a footprint that fits shows its own
/// size, one far past the cache shows the cache's, and a slower cache level
/// counts for the share it answers. No cache faster than DRAM: a byte costs
/// DRAM's at most, and nothing is held.
pub(crate) fn held(rates: &[(usize, f64)], global_byte: f64) -> (f64, f64) {
    let near = rates[0].1.min(global_byte);
    // A cache no faster than DRAM answers nothing.
    let answered = |r: f64| if global_byte > near { ((global_byte - r) / (global_byte - near)).max(0.0) } else { 0.0 };
    let bytes = rates.iter().map(|&(f, r)| f as f64 * answered(r)).fold(0.0, f64::max);
    (near, bytes)
}

/// The tiles a cache probe's region of `footprint` bytes holds: the largest
/// prime count of [`CACHE_TILE`]s that fits, so blocks any stride apart read
/// distinct tiles whatever order a dispatcher deals them to its SMs (a
/// power-of-two count against 64 CUs dealt in turn put every block of a CU
/// on one tile, read from its own L0), and a sweep of any block count the
/// prime does not divide reaches every tile.
pub(crate) fn region_tiles(footprint: usize) -> usize {
    let most = footprint / (CACHE_TILE.elems() * 4);
    (2..=most).rev().find(|&n| (2..).take_while(|d| d * d <= n).all(|d| n % d != 0)).expect("a footprint of tiles")
}

/// Measurement: the tile a cache probe's blocks read: 16 KB of i32.
const CACHE_TILE: Shape = Shape::new(32, 128);

/// The cache probes of `target` with their footprints, smallest first.
pub(crate) fn cache_probes(target: &Target) -> Result<Vec<(usize, Probe)>> {
    let p = Probes::new(target)?;
    Ok(FOOTPRINTS.iter().map(|&(op, footprint)| (footprint, p.cached(op, footprint))).collect())
}

/// Every probe of `target`, in [`Costs::fields`] order.
pub(crate) fn probes(target: &Target) -> Result<Vec<Probe>> {
    let p = Probes::new(target)?;
    Ok(vec![
        p.smem_byte(),
        p.global_byte(),
        p.scatter(),
        p.step(),
        p.step_group(),
        p.step_i8(),
        p.step_lone(),
        p.step_lone_i8(),
        p.score(),
        p.trips("probe:trip", true, 2),
        p.rolled_trips(),
        p.trips("probe:trip_one", false, 1),
        p.lone_trips("probe:trip_lone", true, 2),
        p.lone_trips("probe:trip_one_lone", false, 1),
        p.lane_move(),
        p.smem_trip(),
        p.latency("probe:latency", 2),
        p.latency_one(),
    ])
}

/// A trip probe of `blocks_per_sm` blocks an SM (one round where the SM
/// holds them all), to tell a trip's per-block issue from what the resident
/// blocks hide of it.
#[cfg(test)]
pub(crate) fn trips_at(
    target: &Target,
    op: &'static str,
    unroll: bool,
    stages: usize,
    blocks_per_sm: usize,
) -> Result<Probe> {
    let mut p = Probes::new(target)?;
    p.blocks = p.sms as usize * blocks_per_sm;
    Ok(p.trips(op, unroll, stages))
}

/// The trip probes' i32 tile: a 16-byte chunk per thread, eight to a row.
fn trip_tile(target: &Target) -> Shape {
    Shape::new((WARPS * target.wave) as usize / 8, 32)
}

struct Probes<'a> {
    target: &'a Target,
    /// Blocks of a relayout-class probe, and what a throughput probe's
    /// difference is scaled by.
    sms: u32,
    /// Blocks of [`WARPS`] that fill every SIMD's wave slots four times over.
    blocks: usize,
    /// A relayout-class probe's warps: one per SIMD.
    warps: u32,
}

impl<'a> Probes<'a> {
    fn new(target: &'a Target) -> Result<Self> {
        let sms = target.sms.context(NoSmsSnafu { arch: target.arch })?;
        let o = target.occupancy;
        let blocks = (4 * sms * o.simds * o.waves / WARPS) as usize;
        Ok(Self { target, sms, blocks, warps: o.simds })
    }

    fn lowering(&self) -> Lowering {
        Lowering::new(&Device { target: self.target.clone(), costs: Costs::default() })
    }

    /// A relayout-class probe's lowering: a block per SM of a warp per SIMD,
    /// one wave on each, so a wave waits out every dependent step itself (and
    /// a barrier spans waves, as in a kernel's block), as a relayout between
    /// two products does; its value is that wait per unit of one wave.
    fn chain(&self, costs: Costs) -> Lowering {
        let mut lowering = self.lowering();
        lowering.device.costs = costs;
        lowering
    }

    /// `[baseline, probe]` from `build(false)` and `build(true)`, lowered
    /// alike; `per_sm` scales the difference to one SM's time.
    fn make(
        &self,
        op: &'static str,
        dtype: ScalarDType,
        units: f64,
        per_sm: bool,
        lowering: Lowering,
        build: impl Fn(bool) -> Program,
    ) -> Probe {
        let candidates = [false, true].map(|on| vec![(build(on), lowering.clone())]).into();
        let scale = if per_sm { self.sms as f64 } else { 1.0 };
        Probe { op, dtype, candidates, units, scale, signal: Signal::Dominant }
    }

    /// Operand bytes read from shared memory: an f32 tile loaded and summed
    /// each iteration, where the baseline sums a register tile in the same
    /// loop (f32, so the compiler cannot fold its sums into a product). The
    /// probe's tile is a slot of a ring written once and fenced before the
    /// loop; the slot an iteration reads comes from a parameter (zeros at
    /// run), so no load is invariant for the compiler to hoist out of an
    /// unrolled loop, and no barrier enters it.
    fn smem_byte(&self) -> Probe {
        let (op, tile, iters) = ("probe:smem_byte", Shape::new(32, 128), 1024);
        let units = (self.blocks * iters * tile.elems() * 4) as f64;
        self.make(op, ScalarDType::Float32, units, true, self.lowering(), |on| {
            let mut k = kernel(op, self.blocks, WARPS);
            let x = input::<F32>(&mut k, tile);
            let ring = on.then(|| {
                let ring = k.smem::<F32>("ring", 2 * tile.elems());
                for slot in 0..2 {
                    let s = k.smem_slot::<F32>(ring, slot, tile);
                    k.store(s, x);
                }
                (ring, k.param::<I32>("order", ParamKind::In, iters))
            });
            let zero = k.zeros::<F32>(tile);
            if on {
                k.barrier();
            }
            let [sum] = k.loop_(iters, [zero], |k, i, [sum]| {
                let y = match ring {
                    Some((ring, order)) => {
                        let slot = k.load_scalar(order, i) % 2;
                        let s = k.smem_slot::<F32>(ring, slot, tile);
                        k.load(s)
                    }
                    None => x,
                };
                [k.binary(sum, y, BinaryOp::Add)]
            });
            output(&mut k, self.blocks, sum);
            k.finish()
        })
    }

    /// [`STREAM`] bytes of i32 tiles, block-interleaved so the resident
    /// blocks read neighbouring tiles, each byte once.
    fn global_byte(&self) -> Probe {
        let (op, tile) = ("probe:global_byte", Shape::new(32, 128));
        let (blocks, iters) = self.stream_grid(tile.elems() * 4);
        self.make(op, ScalarDType::Int32, STREAM as f64, true, self.lowering(), |on| {
            let mut k = kernel(op, blocks, WARPS);
            let src = k.param::<I32>("src", ParamKind::In, STREAM / 4);
            let block = k.block(0);
            let zero = k.fill::<I32>(tile, Const::Int(0));
            let sum = if on {
                sum_loads(&mut k, iters, zero, |k, i| {
                    k.view(src, (i * blocks + block) * tile.elems(), [tile.cols, 1], tile, [None, None])
                })
            } else {
                zero
            };
            output(&mut k, blocks, sum);
            k.finish()
        })
    }

    /// Four times [`STREAM`] bytes of [`CACHE_TILE`]s read by a single wave of
    /// blocks, every block resident from first to last, so the tiles they
    /// read an iteration together run on in one sweep, wrapped into a region
    /// of a prime number of tiles within `footprint` ([`region_tiles`]): a
    /// tile comes round again after a region of traffic, so a byte comes
    /// from wherever a region that size stays, and no two blocks read one
    /// tile together. Faster than DRAM, so it reads more than
    /// [`Self::global_byte`] to stand as clear of its baseline.
    fn cached(&self, op: &'static str, footprint: usize) -> Probe {
        let tile = CACHE_TILE;
        let o = self.target.occupancy;
        let blocks = (self.sms * o.simds * o.waves / WARPS) as usize;
        let iters = 4 * STREAM / (tile.elems() * 4) / blocks;
        let region = region_tiles(footprint);
        let units = (blocks * iters * tile.elems() * 4) as f64;
        self.make(op, ScalarDType::Int32, units, true, self.lowering(), |on| {
            let mut k = kernel(op, blocks, WARPS);
            let src = k.param::<I32>("src", ParamKind::In, region * tile.elems());
            let block = k.block(0);
            let zero = k.fill::<I32>(tile, Const::Int(0));
            let sum = if on {
                sum_loads(&mut k, iters, zero, |k, i| {
                    let at = (i * blocks + block.clone()) % region * tile.elems();
                    k.view(src, at, [tile.cols, 1], tile, [None, None])
                })
            } else {
                zero
            };
            output(&mut k, blocks, sum);
            k.finish()
        })
    }

    /// A wave's bf16 row of 512 loaded each iteration and summed: the
    /// probe's view puts its columns 128 bytes apart, so a lane's every
    /// element is a 2-byte load of its own line, where the baseline's lanes
    /// read 16-byte runs. The rows cycle through a region every wave shares,
    /// so the lines come from cache either way.
    fn scatter(&self) -> Probe {
        let (op, cols, iters, region, span) = ("probe:scatter", 512, 2048, 8, 64 * 512);
        let tile = Shape::new(self.warps as usize, cols);
        let units = (iters * cols) as f64;
        self.make(op, ScalarDType::BFloat16, units, false, self.chain(Costs::default()), |on| {
            let mut k = kernel(op, self.sms as usize, self.warps);
            let src = k.param::<BF16>("src", ParamKind::In, region * span);
            let block = k.block(0);
            let zero = k.zeros::<BF16>(tile);
            let stride = if on { [1, span / cols] } else { [tile.cols, 1] };
            let sum = sum_loads(&mut k, iters, zero, |k, i| {
                k.view(src, (i + block) % region * span, stride, tile, [None, None])
            });
            output(&mut k, self.sms as usize, sum);
            k.finish()
        })
    }

    /// Blocks (a power of two filling the device) and iterations that read
    /// [`STREAM`] bytes in `tile_bytes` tiles.
    fn stream_grid(&self, tile_bytes: usize) -> (usize, usize) {
        let tiles = STREAM / tile_bytes;
        let blocks = self.blocks.next_power_of_two().min(tiles);
        (blocks, tiles / blocks)
    }

    /// 16×16×16 matrix-core steps on register operands into four
    /// independent accumulators per warp.
    fn step(&self) -> Probe {
        let grid = WarpGrid { rows: 2, cols: 2 };
        self.steps::<BF16>("probe:step", [64, 64, 16], self.blocks, grid, self.lowering(), true)
    }

    /// [`Self::step`] over 8-bit integers into i32, two NVIDIA or CDNA atoms
    /// deep; no candidates where the target has no such core.
    fn step_i8(&self) -> Probe {
        let grid = WarpGrid { rows: 2, cols: 2 };
        let mut probe = self.steps::<I8>("probe:step_i8", [64, 64, 64], self.blocks, grid, self.lowering(), true);
        if self.target.geometry_of(ScalarDType::Int8).is_none() {
            probe.candidates.clear();
        }
        probe
    }

    /// [`Self::trips`] over a rolled ring of two slots, priced apart from an
    /// unrolled one only where the fill goes through registers
    /// ([`Costs::trip`]); no candidates elsewhere.
    fn rolled_trips(&self) -> Probe {
        let mut probe = self.trips("probe:trip_rolled", false, 2);
        if self.target.fill != Fill::Staged {
            probe.candidates.clear();
        }
        probe
    }

    /// The step issued as a warpgroup product ([`crate::atoms::GroupMmaAtom`]):
    /// a group's 64×256 accumulator over a 64-deep shared operand pair, four
    /// `k16` instructions a statement, the operands written to shared memory
    /// once; no candidates where the target has no such core.
    fn step_group(&self) -> Probe {
        let (op, [bm, bn, bk]) = ("probe:step_group", [64usize, 256, 64]);
        let iters = 1024;
        let units = (self.blocks * iters * bm * bn * bk / 4096) as f64;
        let mut lowering = self.lowering();
        lowering.core = Core::Group;
        let mut probe = self.make(op, ScalarDType::BFloat16, units, true, lowering, |on| {
            let mut k = kernel(op, self.blocks, WARPS);
            let (sa, sb) = (Shape::new(bm, bk), Shape::new(bn, bk));
            let (a, b) = (input::<BF16>(&mut k, sa), input::<BF16>(&mut k, sb));
            let (a_s, b_s) = (k.smem::<BF16>("a_s", sa.elems()), k.smem::<BF16>("b_s", sb.elems()));
            let (a_t, b_t) = (k.smem_view::<BF16>(a_s, 0, sa), k.smem_view::<BF16>(b_s, 0, sb));
            k.store(a_t, a);
            k.store(b_t, b);
            let zero = k.zeros::<F32>(Shape::new(bm, bn));
            let [acc] =
                if on { k.loop_(iters, [zero], |k, _, [acc]| [k.mma(acc, a_t, false, b_t, true)]) } else { [zero] };
            output(&mut k, self.blocks, acc);
            k.finish()
        });
        if self.target.group_mma.is_empty() {
            probe.candidates.clear();
        }
        probe
    }

    /// The step of a wave alone on its SIMD, a block per SM of a wave per
    /// SIMD ([`Self::chain`]), each wave over the 32×32 accumulators a wave
    /// of [`Self::step`] holds: the chain's latency per step, where `step`
    /// is the core's throughput; their ratio is the waves a SIMD hides it
    /// with. A unit is one wave's step.
    fn step_lone(&self) -> Probe {
        let warps = self.warps;
        let tile = [32 * warps as usize, 32, 16];
        self.steps::<BF16>(
            "probe:step_lone",
            tile,
            self.sms as usize,
            warps.into(),
            self.chain(Costs::default()),
            false,
        )
    }

    /// [`Self::step_lone`] over 8-bit integers: the 8-bit chain's own
    /// latency, whose ratio to [`Self::step_i8`] is the waves a SIMD hides
    /// an 8-bit product's chain with; no candidates where the target has no
    /// such core.
    fn step_lone_i8(&self) -> Probe {
        let warps = self.warps;
        let mut probe = self.steps::<I8>(
            "probe:step_lone_i8",
            [32 * warps as usize, 32, 32],
            self.sms as usize,
            warps.into(),
            self.chain(Costs::default()),
            false,
        );
        if self.target.geometry_of(ScalarDType::Int8).is_none() {
            probe.candidates.clear();
        }
        probe
    }

    /// `blocks` of `warps` each accumulating a `bm × bn` product over `bk`
    /// every iteration, where the baseline holds the zero accumulator; a
    /// unit is a 16×16×16 step of the SM (`per_sm`) or of one wave.
    fn steps<T: MmaIn>(
        &self,
        op: &'static str,
        [bm, bn, bk]: [usize; 3],
        blocks: usize,
        grid: WarpGrid,
        lowering: Lowering,
        per_sm: bool,
    ) -> Probe {
        let iters = 1024;
        let steps = iters * bm * bn * bk / 4096;
        let units = if per_sm { blocks * steps } else { steps / grid.count() as usize } as f64;
        self.make(op, T::DTYPE, units, per_sm, lowering, |on| {
            let mut k = kernel(op, blocks, grid);
            let a = input::<T>(&mut k, Shape::new(bm, bk));
            let b = input::<T>(&mut k, Shape::new(bn, bk));
            let zero = k.zeros::<T::Acc>(Shape::new(bm, bn));
            let [acc] = if on {
                k.loop_(iters, [zero], |k, _, [acc]| [k.mma_oriented(acc, a, false, b, true, Some(Orient::Direct))])
            } else {
                [zero]
            };
            output(&mut k, blocks, acc);
            k.finish()
        })
    }

    /// A score tile's softmax vector work: scale, row max, subtract, `exp2`,
    /// row sum, the sum accumulated across iterations.
    fn score(&self) -> Probe {
        let (op, tile, iters) = ("probe:score", Shape::new(64, 128), 256);
        let units = (self.blocks * iters * tile.elems()) as f64;
        self.make(op, ScalarDType::Float32, units, true, self.lowering(), |on| {
            let mut k = kernel(op, self.blocks, WARPS);
            let x = input::<F32>(&mut k, tile);
            let zero = k.zeros::<F32>(Shape::new(tile.rows, 1));
            let [x, l] = if on {
                k.loop_(iters, [x, zero], |k, _, [x, l]| {
                    let scale = k.fill::<F32>(tile, Const::Float(0.5));
                    let s = k.binary(x, scale, BinaryOp::Mul);
                    let m = k.reduce(s, Axis::Row, ReduceOp::Max);
                    let d = k.binary(s, m, BinaryOp::Sub);
                    let e = k.unary(d, UnaryOp::Exp2);
                    let r = k.reduce(e, Axis::Row, ReduceOp::Sum);
                    [e, k.binary(l, r, BinaryOp::Add)]
                })
            } else {
                [x, zero]
            };
            output(&mut k, self.blocks, x);
            output(&mut k, self.blocks, l);
            k.finish()
        })
    }

    /// [`TRIPS`] trips of a pipeline over the least fill a block copies, a
    /// 16-byte chunk per thread (eight to a 32-wide i32 row), through a ring
    /// `stages` deep as the target runs it, lowered `unroll`ed (a trip's
    /// fixed cost) or rolled (slot offsets recomputed per trip).
    ///
    /// `Costs::trip*` is what one trip adds besides moving its tile's bytes
    /// (priced by `global_byte` and `l2_byte`) and besides reading its
    /// fragment back (priced by `smem_byte`). So the baseline does both in a
    /// plain loop of as many trips:
    /// - it moves the same bytes along the same path: the same chunks of the
    ///   same source, loaded as the fill loads them ([`Kernel::load_as_fill`]:
    ///   at L2 only behind a `cp.async.cg`, a plain global load behind a
    ///   register-staged fill) and held as the fill holds them (a 16-byte
    ///   load a lane, pinned: summed in the tile's natural layout a lane
    ///   loads four separate words, which on gfx942 cost more than the ring
    ///   itself and left the pair no signal);
    /// - the source is a region per SM, so no line is a hot spot every SM
    ///   shares (a region all blocks read made the ring's L2 traffic contend
    ///   against a baseline served by L1), and the regions stay where the
    ///   fill's cache level serves them;
    /// - it keeps as many loads in flight as the ring holds slots, issuing a
    ///   group of `stages` trips' loads before summing any, so neither side
    ///   waits out a load the other hides;
    /// - it reads a fragment back from a ring as deep as the probe's;
    /// - both sides take every trip's chunk, and the baseline its fragment's
    ///   slot, from runtime data (`order`, zeros at run): with the chunk
    ///   `step % region` and the slot `order % stages`, a one-slot baseline
    ///   folded its slot to 0 and LLVM hoisted all sixteen of the region's
    ///   loads out of the loop on gfx1201 and gfx1151, timing 1 % of its
    ///   probe.
    ///
    /// The difference is then the trip itself: the ring's store, waits and
    /// barriers, and its slot arithmetic. A baseline under half its probe's
    /// time did not move the bytes, and the pair is not recorded
    /// ([`Resolved::measurable`]). Next to the bytes it is small (a few percent of the pair), so
    /// it is [`Signal::Resolved`]: the median of several timings, held
    /// against their spread, which bounds it where it cannot be told from
    /// zero.
    fn trips(&self, op: &'static str, unroll: bool, stages: usize) -> Probe {
        self.trips_of(op, unroll, stages, self.blocks)
    }

    /// [`Self::trips`] of a block alone on each SM: its trip waits out what
    /// no other block's work hides. The ring dominates a lone block's time
    /// (its baseline is a load's latency, too short for
    /// [`Resolved::measurable`]), so the pair is [`Signal::Dominant`].
    fn lone_trips(&self, op: &'static str, unroll: bool, stages: usize) -> Probe {
        Probe { signal: Signal::Dominant, ..self.trips_of(op, unroll, stages, self.sms as usize) }
    }

    /// [`Self::trips`] over `blocks` blocks.
    fn trips_of(&self, op: &'static str, unroll: bool, stages: usize, blocks: usize) -> Probe {
        let (tile, region) = (trip_tile(self.target), 16);
        let stages = self.target.fill.stages(stages);
        let units = (blocks * TRIPS) as f64;
        let chunks = chunked(tile, 4, WARPS, self.target.wave, Axis::Col).expect("a chunk layout");
        let sms = self.sms as usize;
        let probe = self.make(op, ScalarDType::Int32, units, true, self.lowering(), |on| {
            // Every trip probe runs these programs, named alike.
            let mut k = kernel("probe:trip", blocks, WARPS);
            let src = k.param::<I32>("src", ParamKind::In, sms * region * tile.elems());
            // Zeros at run (the scratch pool's integers), unknown to the
            // compiler: every trip's chunk and fragment slot is a runtime
            // value, so no ring depth folds an index to a constant and no
            // period of the region lets a loop's loads leave it.
            let order = k.param::<I32>("order", ParamKind::In, TRIPS);
            let first = k.block(0) % sms * region;
            let zero = k.fill::<I32>(tile, Const::Int(0));
            let chunk = |k: &mut Kernel, step: Sc| {
                let at = (step.clone() + k.load_scalar(order, step)) % region;
                k.view(src, (first.clone() + at) * tile.elems(), [32, 1], tile, [None, None])
            };
            let sum = if on {
                ring_sum(&mut k, TRIPS, Ring { stages, unroll, crews: 1 }, zero, chunk)
            } else {
                // The ring's slots, written once and fenced before the loop; a
                // trip reads the slot `order` names, as the ring's consume
                // reads its tile.
                let ring = k.smem::<I32>("ring", stages * tile.elems());
                for slot in 0..stages {
                    let s = k.smem_slot::<I32>(ring, slot, tile);
                    k.store(s, zero);
                }
                k.barrier();
                k.pin(zero, chunks.clone());
                // As many loads in flight as the ring holds slots: each
                // iteration issues `stages` trips' loads before it sums any.
                let [sum] = k.loop_(TRIPS / stages, [zero], |k, i, [mut sum]| {
                    let trips: Vec<Sc> = (0..stages).map(|j| i.clone() * stages + j).collect();
                    let loaded: Vec<Regs<I32>> = trips
                        .iter()
                        .map(|t| {
                            let v = chunk(k, t.clone());
                            let y = k.load_as_fill(v);
                            k.pin(y, chunks.clone());
                            y
                        })
                        .collect();
                    for (t, y) in trips.into_iter().zip(loaded) {
                        let slot = k.load_scalar(order, t);
                        let s = k.smem_slot::<I32>(ring, slot, tile);
                        let fragment = k.load(s);
                        sum = k.binary(sum, y, BinaryOp::Add);
                        sum = k.binary(sum, fragment, BinaryOp::Add);
                    }
                    [sum]
                });
                sum
            };
            output(&mut k, blocks, sum);
            k.finish()
        });
        Probe { signal: Signal::Resolved, ..probe }
    }

    /// The half-wave exchanges of a relayout between two products: lanes
    /// trade a register with the lane 16 away, then with the neighbouring
    /// lane ([`half_swapped`]), so the value is the average of the moves a
    /// half-wave exchange issues ([`Costs::lane_move`]). Lowered with a free
    /// lane move and a dear round trip, so the inference exchanges.
    fn lane_move(&self) -> Probe {
        let plain = self.held();
        let helds = [half_swapped(&plain, 4), half_swapped(&plain, 0)];
        self.rehold("probe:lane_move", &plain, &helds, Costs { smem_trip: 1.0, ..Default::default() })
    }

    /// The re-holding across the rows of 16 lowered with a dear lane move:
    /// the round trip through shared memory, fenced both ways.
    fn smem_trip(&self) -> Probe {
        let plain = self.held();
        let helds = [half_swapped(&plain, 4)];
        self.rehold("probe:smem_trip", &plain, &helds, Costs { lane_move: 1.0, ..Default::default() })
    }

    /// How a wave's 16 × two-wave-widths bf16 tile (a score tile's
    /// probabilities) is held when nothing asks otherwise.
    fn held(&self) -> TileLayout {
        let tile = Shape::new(16 * self.warps as usize, 2 * self.target.wave as usize);
        natural(tile, self.warps, self.target.wave).expect("a natural layout")
    }

    /// A relayout-class pair: a bf16 tile carried as `plain` and, for each of
    /// `helds`, negated into a value pinned there and negated back, so the
    /// inference re-holds the tile there and back as `costs` choose; the
    /// baseline holds every value as `plain`, so its negations cancel and its
    /// loop is empty. A unit is the [`crate::layouts::Relayout::work`] of one
    /// wave's element re-held.
    fn rehold(&self, op: &'static str, plain: &TileLayout, helds: &[TileLayout], costs: Costs) -> Probe {
        let iters = 4096;
        let tile = plain.shape();
        let (warps, lanes) = (self.warps, self.target.wave);
        let work: f64 = helds
            .iter()
            .flat_map(|h| {
                [plain.relayout(h, warps, lanes, &costs, false), h.relayout(plain, warps, lanes, &costs, false)]
            })
            .map(|r| r.work())
            .sum();
        let units = iters as f64 * work * tile.elems() as f64 / warps as f64;
        self.make(op, ScalarDType::BFloat16, units, false, self.chain(costs), |on| {
            let mut k = kernel(op, self.sms as usize, warps);
            let x = input::<BF16>(&mut k, tile);
            let [x] = k.loop_(iters, [x], |k, _, [x]| {
                k.pin(x, plain.clone());
                [helds.iter().fold(x, |x, held| {
                    let y = k.unary(x, UnaryOp::Neg);
                    let next = k.unary(y, UnaryOp::Neg);
                    k.pin(y, if on { held.clone() } else { plain.clone() });
                    k.pin(next, plain.clone());
                    next
                })]
            });
            output(&mut k, self.sms as usize, x);
            k.finish()
        })
    }

    /// [`Self::latency`] through a one-slot ring where fills are `cp.async`;
    /// no candidates where they are register-staged, whose one slot loads a
    /// trip ahead as two do ([`Costs::latency_one`]).
    fn latency_one(&self) -> Probe {
        let mut probe = self.latency("probe:latency_one", 1);
        if self.target.fill != Fill::CpAsync {
            probe.candidates.clear();
        }
        probe
    }

    /// One wave per SM streaming a 16-byte chunk per lane through a ring
    /// `stages` deep as the target runs it, the waves' trips interleaved over
    /// [`STREAM`] so no line is read twice in a run: with nothing else
    /// resident and nothing to compute, every trip waits out what of its
    /// load the ring does not prefetch on a nearly idle memory. The baseline
    /// is the ring's depth in trips. A lone wave's 16 bytes a lane are not
    /// bandwidth-bound: the difference is the wait alone.
    fn latency(&self, op: &'static str, stages: usize) -> Probe {
        // A 16-byte chunk of i32 per lane.
        let tile = Shape::new(self.target.wave as usize, 4);
        let waves = self.sms as usize;
        let trips = LATENCY_TRIPS.min(STREAM / 4 / tile.elems() / waves);
        let stages = self.target.fill.stages(stages);
        let lowering = self.lowering();
        self.make(op, ScalarDType::Int32, (trips - stages) as f64, false, lowering, |on| {
            let mut k = kernel(op, waves, 1);
            let src = k.param::<I32>("src", ParamKind::In, STREAM / 4);
            let block = k.block(0);
            let zero = k.fill::<I32>(tile, Const::Int(0));
            let sum = ring_sum(&mut k, if on { trips } else { stages }, stages, zero, |k, step| {
                k.view(src, (step * waves + block) * tile.elems(), [4, 1], tile, [None, None])
            });
            output(&mut k, waves, sum);
            k.finish()
        })
    }
}

fn kernel(op: &str, blocks: usize, warps: impl Into<WarpGrid>) -> Kernel {
    let mut k = Kernel::new(op.replace(':', "_"));
    k.grid([Sc::from(blocks), Sc::from(1), Sc::from(1)]);
    k.warps(warps);
    k
}

/// `init` plus the tile `view` reads at each of `iters` iterations.
fn sum_loads<T: Elem>(
    k: &mut Kernel,
    iters: usize,
    init: Regs<T>,
    view: impl FnOnce(&mut Kernel, Sc) -> Gmem<T>,
) -> Regs<T> {
    let [sum] = k.loop_(iters, [init], |k, i, [sum]| {
        let v = view(k, i);
        let y = k.load(v);
        [k.binary(sum, y, BinaryOp::Add)]
    });
    sum
}

/// `init` plus the tile `view` reads at each of `trips` trips of a pipeline
/// staging it through a ring of `stages` slots.
fn ring_sum(
    k: &mut Kernel,
    trips: usize,
    ring: impl Into<Ring>,
    init: Regs<I32>,
    view: impl FnOnce(&mut Kernel, Sc) -> Gmem<I32>,
) -> Regs<I32> {
    let (tile, ring) = (k.shape(init), ring.into());
    let slots = k.smem::<I32>("ring", ring.stages * tile.elems());
    let [sum] = k.pipeline(
        trips,
        ring,
        [init],
        |k, step, slot| {
            let v = view(k, step);
            let s = k.smem_slot::<I32>(slots, slot, tile);
            k.stage(s, v, CopyMode::Async);
        },
        |k, _, slot, [sum]| {
            let s = k.smem_slot::<I32>(slots, slot, tile);
            let y = k.load(s);
            [k.binary(sum, y, BinaryOp::Add)]
        },
    );
    sum
}

/// A `shape` tile loaded from a parameter of its own, so no value is known
/// to the compiler.
fn input<T: Elem>(k: &mut Kernel, shape: Shape) -> Regs<T> {
    let p = k.param::<T>("x", ParamKind::In, shape.elems());
    let v = k.view(p, 0, [shape.cols, 1], shape, [None, None]);
    k.load(v)
}

/// `v` stored to block `block(0)`'s slot of an output, so nothing is dead.
fn output<T: Elem>(k: &mut Kernel, blocks: usize, v: Regs<T>) {
    let shape = k.shape(v);
    let p = k.param::<T>("y", ParamKind::Out, blocks * shape.elems());
    let block = k.block(0);
    let view = k.view(p, block * shape.elems(), [shape.cols, 1], shape, [None, None]);
    k.store(view, v);
}

/// `l` with its first register bit and lane bit `lane` exchanged: each lane
/// keeps half its elements and takes the other half from the lane whose
/// index differs in bit `lane`.
pub(crate) fn half_swapped(l: &TileLayout, lane: u32) -> TileLayout {
    let bases = |dim| -> Vec<[u32; 2]> {
        (0..l.frag.in_bits(dim))
            .map(|b| {
                let get = |out| l.frag.basis(dim, b).iter().find(|o| o.0 == out).map_or(0, |o| o.1);
                [get(Dim::Row), get(Dim::Col)]
            })
            .collect()
    };
    let (mut reg, mut lanes) = (bases(Dim::Reg), bases(Dim::Lane));
    std::mem::swap(&mut reg[0], &mut lanes[lane as usize]);
    let outs = [(Dim::Row, l.frag.out_size(Dim::Row)), (Dim::Col, l.frag.out_size(Dim::Col))];
    TileLayout { frag: Layout::from_bases(outs, &[(Dim::Reg, &reg), (Dim::Lane, &lanes)]), ..l.clone() }
}
