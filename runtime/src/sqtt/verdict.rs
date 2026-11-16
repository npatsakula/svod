//! What limits a kernel's main loop. Its traced waves' cycles split into buckets that add up to them
//! exactly ([`Partition`]), each bucket also less what other pipes' work on the SIMD hid (AMD's hidden
//! latency): the exposed cycles. The findings ([`Finding`]) rank by exposed share; the pipes' busy
//! shares over the steady window classify the kernel apart ([`Verdict::bound`]).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write;
use std::ops::RangeInclusive;

use serde::{Deserialize, Serialize};
use svod_device::amd::occupancy::Limit;
use svod_device::sqtt::asm::{Access, Counter, Kind, Memory, Space, classify};
use svod_device::sqtt::hidden::Hiding;
use svod_device::sqtt::report::Spread;
use svod_device::sqtt::wave::{Instruction, Wave};

use super::{Analysis, Class, Pstate, Row};

/// Exposed vector-memory waits at a tenth of the loop's cycles: below it, moving loads or deepening
/// the ring repays less than its register and LDS cost.
pub const LOAD_EXPOSED: f64 = 0.10;
/// The same tenth for exposed LDS waits and stall.
pub const LDS_EXPOSED: f64 = 0.10;
/// Barriers at a tenth of the loop are listed; their exposed share at a tenth is worth dropping one.
pub const BARRIER: f64 = 0.10;
/// Non-matrix VALU and SALU issue at a tenth of the matrix issue: a loop's own bookkeeping (a few
/// address increments a trip) stays under it, per-element guard and address arithmetic passes it.
pub const LOOP_OVERHEAD: f64 = 0.10;
/// The matrix pipe busy for this share of the steady window or more: near its throughput.
pub const MATRIX_BOUND: f64 = 0.70;
/// Any pipe busy this share of the window or more, with matrix work under [`ISSUE_MATRIX`] of it:
/// the SIMD issues nearly every cycle, mostly not matrix work.
pub const ISSUE_BOUND: f64 = 0.90;
pub const ISSUE_MATRIX: f64 = 0.60;
/// The traced SIMDs idle (no pipe busy) this share of the steady window: more waves would fill it.
pub const SIMD_IDLE: f64 = 0.10;
/// The traced run over the untraced past this: its cycle shares no longer describe the untraced
/// kernel (the probes of 2026-11 put an unpinned trace within a few percent of it).
pub const PERTURBED: f64 = 1.2;
/// Cycles no traced time spans: over a minute at 4 GHz, past any dispatch a profile traces, and far
/// below the 2^60 the wave model clamps a corrupt stream's timestamps to.
pub const PLAUSIBLE_CYCLES: i64 = 1 << 38;
/// A barrier wait this short: its wave arrived last and the barrier released at once. The fixtures'
/// waits split into a floor of 8 to 15 cycles and a tail from tens of cycles up.
pub const LAST_ARRIVER: i64 = 16;
/// A load whose cover is under this share of its latency waits right behind its issue.
pub const SHORT_COVER: f64 = 0.25;
/// The loads, stalls and barrier causes a finding lists.
const EVIDENCE: usize = 4;
/// Lanes a wave loads for: svod's RDNA kernels run wave32.
const LANES: u64 = 32;

/// What a wait record waited for: the counter of the instruction that bound it, and whether that was
/// a store (a `ds_store` flushed before a barrier, a global store). A wait that retired nothing, or
/// on a wave the replay skipped, counts on the first counter it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WaitOn {
    pub counter: Counter,
    pub store: bool,
}

impl WaitOn {
    pub fn name(self) -> String {
        match self.store && self.counter != Counter::Store {
            true => format!("{} store", self.counter.name()),
            false => self.counter.name().into(),
        }
    }
}

/// What a record's cycles count as in a [`Partition`].
#[derive(Debug, Clone, Copy)]
enum Bucket {
    Issue(Class),
    Wait(WaitOn),
    Barrier,
}

fn stores(kind: Option<&Kind>) -> bool {
    matches!(
        kind,
        Some(Kind::Memory(Memory {
            access: Access::Store | Access::Atomic { returns: false } | Access::Writeback,
            ..
        }))
    )
}

/// Every instruction's kind by PC, and the PCs of gfx12's `s_barrier_signal`: where a wave arrives at
/// a barrier it waits for later.
pub(super) struct Buckets {
    kinds: HashMap<u64, Kind>,
    signals: HashSet<u64>,
}

/// What one SE's traced waves did in the main loop.
pub(super) struct Pass {
    pub raw: Partition,
    pub exposed: Partition,
    pub barriers: BTreeMap<u64, BarrierSite>,
    /// The partition's cycles by the instruction they are charged to: a wait's to the instruction
    /// that bound it.
    pub charges: BTreeMap<u64, Charge>,
}

/// An instruction's cycles in the loop by bucket, `[raw, exposed]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Charge {
    pub issue: [i64; 2],
    pub stall: [i64; 2],
    pub wait: [i64; 2],
    pub barrier: [i64; 2],
    pub idle: [i64; 2],
}

impl Charge {
    pub fn add(&mut self, o: &Self) {
        for (at, from) in [
            (&mut self.issue, o.issue),
            (&mut self.stall, o.stall),
            (&mut self.wait, o.wait),
            (&mut self.barrier, o.barrier),
            (&mut self.idle, o.idle),
        ] {
            (at[0], at[1]) = (at[0].saturating_add(from[0]), at[1].saturating_add(from[1]));
        }
    }
}

/// `cycles` into `at`, raw and less what `hidden` covers of them. Saturating, as every sum of the
/// partition: a corrupt trace's cycles may pass `i64`.
fn charge(at: &mut [i64; 2], cycles: i64, hidden: i64) {
    (at[0], at[1]) = (at[0].saturating_add(cycles), at[1].saturating_add(cycles - hidden.clamp(0, cycles)));
}

/// Raw and exposed cycles by key.
struct Sums<K>(BTreeMap<K, (i64, i64)>);

/// Buckets' cycles by key.
type Keyed<K> = Vec<(K, i64)>;

impl<K: Ord + Copy> Sums<K> {
    fn new() -> Self {
        Self(BTreeMap::new())
    }

    fn add(&mut self, key: K, cycles: i64, hidden: i64) {
        let at = self.0.entry(key).or_default();
        (at.0, at.1) = (at.0.saturating_add(cycles), at.1.saturating_add(cycles - hidden.clamp(0, cycles)));
    }

    fn split(self) -> (Keyed<K>, Keyed<K>) {
        self.0.into_iter().filter(|e| e.1.0 > 0).map(|(k, (r, e))| ((k, r), (k, e))).unzip()
    }
}

impl Buckets {
    pub(super) fn new(lines: &[(u64, &str)]) -> Self {
        let signal = |line: &str| line.trim_start().starts_with("s_barrier_signal");
        Self {
            kinds: lines.iter().map(|&(pc, line)| (pc, classify(line).kind)).collect(),
            signals: lines.iter().filter(|l| signal(l.1)).map(|l| l.0).collect(),
        }
    }

    fn class(&self, pc: u64) -> Class {
        self.kinds.get(&pc).map_or(Class::Other, |&k| Class::from(k))
    }

    /// `binder`: the `(counter, pc)` of the instruction a wait record waited for.
    fn bucket(&self, pc: u64, binder: Option<&(Counter, u64)>) -> Bucket {
        let kind = self.kinds.get(&pc).copied().unwrap_or(Kind::Other);
        match (Class::from(kind), kind) {
            (Class::Barrier, _) => Bucket::Barrier,
            (Class::WaitVmem | Class::WaitLgkm, Kind::Wait(wait)) => Bucket::Wait(match binder {
                Some(&(counter, at)) => WaitOn { counter, store: stores(self.kinds.get(&at)) },
                None => {
                    let counter = wait.counts().map(|(c, _)| c).find(|&c| c != Counter::Exp).expect("a counted wait");
                    WaitOn { counter, store: counter == Counter::Store }
                }
            }),
            (class, _) => Bucket::Issue(class),
        }
    }

    /// The cycles of `waves` in the loop `body`, every cycle in one bucket: each run of a wave's
    /// consecutive records in the body counts from its first record's issue to the latest end of its
    /// records. A record takes the cycles of its span the earlier records did not cover, the stall's
    /// apart; a gap between records is idle, put on the record after it. The exposed partition takes
    /// off each part what `hiding` measures other pipes' activity hid of it. `bound[w]`: wave `w`'s
    /// wait records' bindings, by record index.
    pub(super) fn partition(
        &self,
        waves: &[Wave],
        bound: &[HashMap<usize, (Counter, u64)>],
        body: RangeInclusive<u64>,
        hiding: &Hiding,
    ) -> Pass {
        let (mut issue, mut stall, mut idle) = (Sums::<Class>::new(), Sums::new(), Sums::new());
        let (mut wait, mut barrier) = (Sums::<WaitOn>::new(), Sums::<()>::new());
        let (mut raw, mut exposed) = (Partition::default(), Partition::default());
        let mut barriers = BTreeMap::<u64, BarrierSite>::new();
        let mut charges = BTreeMap::<u64, Charge>::new();
        let none = HashMap::new();
        for (w, wave) in waves.iter().enumerate() {
            let bound = bound.get(w).unwrap_or(&none);
            let (mut last, mut run, mut entered) = (wave.begin, None::<(i64, i64)>, false);
            // The record that held the wave longest since its last barrier, `(pc, cycles)`, and that
            // as of its signal to the barrier it waits at next.
            let (mut worst, mut arrived) = (None::<(u64, i64)>, None::<(u64, i64)>);
            for (seq, r) in wave.instructions.iter().enumerate() {
                let hidden = hiding.record(last, r);
                last = r.time + r.duration;
                if !body.contains(&r.pc) {
                    if let Some((start, cursor)) = run.take() {
                        raw.cycles = raw.cycles.saturating_add(cursor - start);
                    }
                    continue;
                }
                entered = true;
                let (_, cursor) = run.get_or_insert((r.time, r.time));
                let gap = (r.time - *cursor).max(0);
                if gap > 0 {
                    idle.add(self.class(r.pc), gap, hidden.idle);
                    charge(&mut charges.entry(r.pc).or_default().idle, gap, hidden.idle);
                    *cursor = r.time;
                }
                let end = r.time + r.duration;
                let cycles = (end - *cursor).max(0);
                let binder = bound.get(&seq);
                let bucket = self.bucket(r.pc, binder);
                let held = match bucket {
                    Bucket::Issue(class) => {
                        let stalled = ((r.time + r.stall).min(end) - *cursor).max(0);
                        stall.add(class, stalled, hidden.stall);
                        issue.add(class, cycles - stalled, hidden.issue);
                        let at = charges.entry(r.pc).or_default();
                        charge(&mut at.stall, stalled, hidden.stall);
                        charge(&mut at.issue, cycles - stalled, hidden.issue);
                        stalled
                    }
                    Bucket::Wait(on) => {
                        wait.add(on, cycles, hidden.stall + hidden.issue);
                        let at = charges.entry(binder.map_or(r.pc, |b| b.1)).or_default();
                        charge(&mut at.wait, cycles, hidden.stall + hidden.issue);
                        cycles
                    }
                    Bucket::Barrier => {
                        barrier.add((), cycles, hidden.stall + hidden.issue);
                        charge(&mut charges.entry(r.pc).or_default().barrier, cycles, hidden.stall + hidden.issue);
                        0
                    }
                };
                if held + gap > 0 && worst.is_none_or(|w| held + gap > w.1) {
                    worst = Some((r.pc, held + gap));
                }
                if let Bucket::Barrier = bucket {
                    let held = worst.take();
                    match self.signals.contains(&r.pc) {
                        true => arrived = held,
                        false => site(&mut barriers, r, arrived.take().or(held)),
                    }
                }
                *cursor += cycles;
            }
            if let Some((start, cursor)) = run {
                raw.cycles = raw.cycles.saturating_add(cursor - start);
            }
            raw.waves += usize::from(entered);
        }
        (raw.issue, exposed.issue) = issue.split();
        (raw.stall, exposed.stall) = stall.split();
        (raw.wait, exposed.wait) = wait.split();
        (raw.idle, exposed.idle) = idle.split();
        let (b, e) = barrier.split();
        (raw.barrier, exposed.barrier) = (b.first().map_or(0, |b| b.1), e.first().map_or(0, |e| e.1));
        exposed.waves = raw.waves;
        exposed.cycles = exposed.buckets();
        Pass { raw, exposed, barriers, charges }
    }
}

/// One barrier wait: whether its wave arrived last, and if so the record that held it longest
/// between its previous barrier and its arrival.
fn site(barriers: &mut BTreeMap<u64, BarrierSite>, r: &Instruction, worst: Option<(u64, i64)>) {
    let s = barriers.entry(r.pc).or_insert_with(|| BarrierSite { pc: r.pc, ..BarrierSite::default() });
    s.hits += 1;
    if r.duration <= LAST_ARRIVER {
        s.last += 1;
        if let Some((pc, _)) = worst {
            match s.causes.iter_mut().find(|c| c.0 == pc) {
                Some(c) => c.1 += 1,
                None => s.causes.push((pc, 1)),
            }
        }
    }
}

/// A barrier wait in the main loop over the traced waves: its hits, those whose wave arrived last
/// (the wait at most [`LAST_ARRIVER`] cycles), and for those the records that had held the wave
/// longest between its previous barrier and its arrival, `(pc, hits)`, most first.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BarrierSite {
    pub pc: u64,
    pub hits: usize,
    pub last: usize,
    pub causes: Vec<(u64, usize)>,
}

impl BarrierSite {
    /// The SEs' sites by PC, less `base`.
    pub(super) fn merge(sites: impl IntoIterator<Item = Self>, base: u64) -> Vec<Self> {
        let mut by_pc = BTreeMap::<u64, Self>::new();
        for s in sites {
            let at = by_pc.entry(s.pc - base).or_insert_with(|| Self { pc: s.pc - base, ..Self::default() });
            (at.hits, at.last) = (at.hits + s.hits, at.last + s.last);
            merge(&mut at.causes, &s.causes.iter().map(|&(pc, n)| (pc - base, n)).collect::<Vec<_>>());
        }
        let mut sites: Vec<Self> = by_pc.into_values().collect();
        sites.iter_mut().for_each(|s| s.causes.sort_by_key(|&(pc, n)| (std::cmp::Reverse(n), pc)));
        sites
    }
}

/// The main loop's traced-wave cycles, each in exactly one bucket: an instruction's `issue` and the
/// `stall` before it by class (a pipe or queue still busy), `wait` by what the wait waited for,
/// `barrier`, and `idle` between records by the class of the record after the gap (the SIMD issuing
/// for another wave, or a dependency the trace does not show). They add up to `cycles`.
///
/// Records overlap where an instruction's duration passes the next one's issue: a cycle goes to the
/// earlier record, so a matrix op's stall behind the previous matrix op lands in that one's issue.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Partition {
    pub waves: usize,
    pub cycles: i64,
    pub issue: Vec<(Class, i64)>,
    pub stall: Vec<(Class, i64)>,
    pub wait: Vec<(WaitOn, i64)>,
    pub barrier: i64,
    pub idle: Vec<(Class, i64)>,
}

fn merge<K: Ord + Copy, V: Copy + std::ops::Add<Output = V> + Default>(into: &mut Vec<(K, V)>, from: &[(K, V)]) {
    let mut sum: BTreeMap<K, V> = into.iter().copied().collect();
    for &(k, c) in from {
        let at = sum.entry(k).or_default();
        *at = *at + c;
    }
    *into = sum.into_iter().collect();
}

fn of<K: PartialEq>(buckets: &[(K, i64)], keys: &[K]) -> i64 {
    buckets.iter().filter(|(k, _)| keys.is_empty() || keys.contains(k)).map(|c| c.1).sum()
}

impl Partition {
    pub fn add(&mut self, o: &Self) {
        (self.waves, self.cycles, self.barrier) =
            (self.waves + o.waves, self.cycles + o.cycles, self.barrier + o.barrier);
        merge(&mut self.issue, &o.issue);
        merge(&mut self.stall, &o.stall);
        merge(&mut self.wait, &o.wait);
        merge(&mut self.idle, &o.idle);
    }

    /// Issue cycles of `classes`; every class where empty.
    pub fn issued(&self, classes: &[Class]) -> i64 {
        of(&self.issue, classes)
    }

    /// Stall cycles before an instruction of `classes` issued; every class where empty.
    pub fn stalled(&self, classes: &[Class]) -> i64 {
        of(&self.stall, classes)
    }

    /// Wait cycles on what `on` keeps.
    pub fn waited(&self, on: impl Fn(WaitOn) -> bool) -> i64 {
        self.wait.iter().filter(|w| on(w.0)).map(|w| w.1).sum()
    }

    /// Idle cycles before an instruction of `classes`; every class where empty.
    pub fn idled(&self, classes: &[Class]) -> i64 {
        of(&self.idle, classes)
    }

    /// Each bucket by name (`issue valu`, `wait load`, `barrier`, ...) in report order: issue, stall,
    /// wait, barrier, idle.
    pub fn named(&self) -> Vec<(String, i64)> {
        let by = |what: &str, buckets: &[(Class, i64)]| -> Vec<(String, i64)> {
            buckets.iter().map(|&(c, n)| (format!("{what} {}", c.name()), n)).collect()
        };
        let mut named = by("issue", &self.issue);
        named.extend(by("stall", &self.stall));
        named.extend(self.wait.iter().map(|&(on, n)| (format!("wait {}", on.name()), n)));
        named.push(("barrier".into(), self.barrier));
        named.extend(by("idle", &self.idle));
        named
    }

    /// Every bucket's cycles: `cycles` itself.
    pub fn buckets(&self) -> i64 {
        self.issued(&[]) + self.stalled(&[]) + self.waited(|_| true) + self.barrier + self.idled(&[])
    }
}

/// What limits the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingKind {
    MatrixBound,
    LoadExposed,
    LdsExposed,
    Barrier,
    OccupancyLimited,
    LoopOverhead,
    IssueBound,
}

impl FindingKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::MatrixBound => "matrix_bound",
            Self::LoadExposed => "load_exposed",
            Self::LdsExposed => "lds_exposed",
            Self::Barrier => "barrier",
            Self::OccupancyLimited => "occupancy_limited",
            Self::LoopOverhead => "loop_overhead",
            Self::IssueBound => "issue_bound",
        }
    }
}

/// Why a finding's numbers are approximate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Approx {
    /// The traced run took over [`PERTURBED`] times the untraced one.
    Perturbed,
    /// gfx11 traces a matrix op as a 1-cycle token: its issue counts the stall behind it, and what
    /// other pipes hid is unknown, so the share is the raw one.
    UntimedMatrix,
    /// A listed binding is a guess: an out-of-order counter or a flat op.
    ApproxBindings,
    /// An SE's span, a wave's lifetime or an instruction's cycles passes [`PLAUSIBLE_CYCLES`]: the
    /// stream is corrupt, and its sums may have saturated.
    ImplausibleTime,
}

impl Approx {
    fn name(self) -> &'static str {
        match self {
            Self::Perturbed => "perturbed",
            Self::UntimedMatrix => "untimed_matrix",
            Self::ApproxBindings => "approx_bindings",
            Self::ImplausibleTime => "implausible_time",
        }
    }
}

/// A load or store that bound the loop's waits, over its waits in the loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Load {
    pub pc: u64,
    pub asm: String,
    pub waits: usize,
    pub stalled: usize,
    pub approx: usize,
    /// Its issue to its waits' arrival: the work that hid its latency.
    pub cover: Spread,
    /// Its issue to the release of the waits that stalled: censored where under half of them stalled,
    /// the others' latency being under their cover.
    pub latency: Option<Spread>,
    /// Its waits' stall less what other pipes' work hid, a trip.
    pub exposed_per_trip: f64,
    /// `(back edges between its issue and the wait, instances)`.
    pub distance: Vec<(u32, usize)>,
    /// The wait it stalled longest at, and the instruction issued after it: the use that put the
    /// wait there.
    pub wait: u64,
    pub consumer: String,
}

impl Load {
    /// Half its waits stalled or more, and their median cover is under [`SHORT_COVER`] of the median
    /// latency: the wait sits right behind the issue.
    pub fn short_cover(&self) -> bool {
        let half = self.stalled * 2 >= self.waits;
        self.latency.is_some_and(|l| half && (self.cover.median as f64) < SHORT_COVER * l.median as f64)
    }
}

/// A loop instruction that stalled before it issued: a trip's stall, and that less what other pipes'
/// work hid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stall {
    pub pc: u64,
    pub asm: String,
    pub per_trip: f64,
    pub exposed_per_trip: f64,
}

/// A trip's cycles of a bucket, and those other pipes' work did not hide.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Cycles {
    pub raw: f64,
    pub exposed: f64,
}

/// The loop's instructions of `class` that stalled longest before they issued, a cycle a trip of it
/// exposed at least.
fn stalls(a: &Analysis, class: Class, trips: f64) -> Vec<Stall> {
    let mut rows: Vec<&Row> = a.loop_rows().filter(|r| r.class == class && r.stall > 0).collect();
    rows.sort_by_key(|r| (std::cmp::Reverse(r.stall), r.pc));
    let stall = |r: &Row| Stall {
        pc: r.pc,
        asm: r.asm.clone(),
        per_trip: r.stall as f64 / trips,
        exposed_per_trip: (r.stall - r.hidden_stall.clamp(0, r.stall)) as f64 / trips,
    };
    rows.into_iter().map(stall).filter(|s| s.exposed_per_trip >= 1.0).take(EVIDENCE).collect()
}

/// What a finding rests on; cycles are a traced wave's, a trip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Evidence {
    /// The traced SIMDs' pipe busy shares over the steady window, over the traced SEs.
    Busy { matrix: f64, valu: f64, any: f64 },
    /// The waits on loads and on stores (an LDS flush before a barrier), the stall before the memory
    /// ops issued (a full queue), the instructions that bound the waits, most exposed first, and the
    /// ops that stalled longest before issuing.
    Waits { load: Cycles, store: Cycles, stall: Cycles, loads: Vec<Load>, stalls: Vec<Stall> },
    /// The waves' cycles parked at barriers, and per barrier PC who arrived last and after what.
    Barrier { cycles: Cycles, sites: Vec<BarrierSite> },
    /// The traced SIMDs' mean resident waves beside those that would fill their idle cycles
    /// (`observed / any busy`; on gfx11 `1 + (waits and barriers) / issue`), the SIMDs' idle share
    /// of the steady window, and the kernel's limit.
    Occupancy { observed: f64, need: f64, idle: Option<f64>, limit: Option<Limit> },
    /// Non-matrix VALU and SALU issue cycles a trip beside the matrix issue, VALU ops a trip, and the
    /// loop's divergent branches `(pc, asm)`.
    Overhead { valu: f64, salu: f64, matrix: f64, valu_hits: f64, divergent: Vec<(u64, String)> },
}

/// One limiter. `share` is of the loop's traced-wave cycles exposed (raw where `untimed_matrix`),
/// `raw` before what other pipes hid; for a [`Verdict::bound`] finding both are the share of the
/// steady window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Finding {
    pub kind: FindingKind,
    pub share: f64,
    pub raw: f64,
    pub evidence: Evidence,
    pub knob: String,
    pub approx: Vec<Approx>,
}

/// The traced SIMDs' pipe busy shares over the steady window: removing every stall gains at most
/// `1 / matrix`. `queued`: the loop's idle cycles before matrix ops a trip, waiting for the pipe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ceiling {
    pub matrix: f64,
    pub valu: f64,
    pub any: f64,
    pub queued: Option<Cycles>,
}

/// The global bytes the loop loads a trip, a traced wave's (every lane of every load), at the
/// dispatch's resident waves and a traced wave's trip time: what the kernel asks of memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Demand {
    pub bytes_per_trip: u64,
    pub waves: f64,
    pub cycles_per_trip: f64,
    pub sclk_mhz: u64,
    pub gbs: f64,
    /// The device's bandwidth a `model.bandwidth_gbs` label gives.
    pub bandwidth_gbs: Option<f64>,
}

/// The main loop's limiters: the kernel's class by its pipes' busy shares over the steady window
/// (`bound`: matrix_bound, issue_bound, occupancy_limited), then the loop's findings by exposed share
/// (`findings`), largest first in each.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    pub ceiling: Option<Ceiling>,
    pub demand: Option<Demand>,
    pub bound: Vec<Finding>,
    pub findings: Vec<Finding>,
}

/// Whether a time the trace states passes [`PLAUSIBLE_CYCLES`]: an SE's span, a complete wave's
/// lifetime, or an instruction's mean cycles a hit.
fn implausible(a: &Analysis) -> bool {
    let per_hit = |r: &Row| [r.latency, r.idle].iter().any(|&c| c / r.hits.max(1) as i64 > PLAUSIBLE_CYCLES);
    a.ses.iter().any(|s| s.span > PLAUSIBLE_CYCLES)
        || a.lifetime.is_some_and(|l| l.max > PLAUSIBLE_CYCLES)
        || a.pcs.iter().any(per_hit)
}

/// The traced run's time over the untraced one.
pub(super) fn slowdown(a: &Analysis) -> Option<f64> {
    a.kernel_ns.zip(a.traced_ns).filter(|&(k, _)| k > 0).map(|(k, t)| t as f64 / k as f64)
}

/// The traced SIMDs' mean resident waves over their steady windows.
pub(super) fn observed(a: &Analysis) -> Option<f64> {
    let traced: Vec<f64> = a.occupancy.ses.iter().filter_map(|s| s.traced).map(|t| t.mean()).collect();
    (!traced.is_empty()).then(|| traced.iter().sum::<f64>() / traced.len() as f64)
}

/// The busy shares of the traced SIMDs' pipes, over every traced SE's window.
fn busy(a: &Analysis) -> Option<(f64, f64, f64)> {
    let busy: Vec<_> = a.occupancy.ses.iter().filter_map(|s| s.busy).collect();
    let cycles = busy.iter().map(|b| b.cycles).sum::<i64>();
    let share =
        |f: fn(&svod_device::sqtt::simd::PipeBusy) -> i64| busy.iter().map(f).sum::<i64>() as f64 / cycles as f64;
    (cycles > 0).then(|| (share(|b| b.matrix), share(|b| b.valu), share(|b| b.any)))
}

/// The loop's binding loads and stores of `class` whose waits stalled, most exposed first.
fn loads(a: &Analysis, class: Class, trips: f64) -> Vec<Load> {
    let Some(l) = &a.main_loop else { return vec![] };
    let rows: HashMap<u64, &Row> = a.pcs.iter().map(|r| (r.pc, r)).collect();
    let exposed = |wait: u64| {
        rows.get(&wait).map_or(1.0, |r| 1.0 - r.hidden_stall.clamp(0, r.stall) as f64 / r.stall.max(1) as f64)
    };
    let next = |pc: u64| a.pcs.get(a.pcs.partition_point(|r| r.pc <= pc)).map_or(String::new(), |r| r.asm.clone());
    let mut loads: Vec<Load> = l
        .bindings
        .iter()
        .filter(|b| rows.get(&b.pc).is_some_and(|r| r.class == class))
        .filter_map(|b| {
            let stall: f64 = b.waits.iter().map(|&(wait, _, stall)| stall as f64 * exposed(wait)).sum();
            let wait = b.waits.iter().max_by_key(|&&(pc, _, stall)| (stall, std::cmp::Reverse(pc)))?.0;
            (stall > 0.0).then(|| Load {
                pc: b.pc,
                asm: rows[&b.pc].asm.clone(),
                waits: b.count,
                stalled: b.stalled,
                approx: b.approx,
                cover: b.cover,
                latency: b.latency,
                exposed_per_trip: stall / trips,
                distance: b.distance.clone(),
                wait,
                consumer: next(wait),
            })
        })
        .collect();
    loads.sort_by(|x, y| y.exposed_per_trip.total_cmp(&x.exposed_per_trip).then(x.pc.cmp(&y.pc)));
    loads.truncate(EVIDENCE);
    loads
}

fn divergent(asm: &str) -> bool {
    let mnemonic = asm.split_whitespace().next().unwrap_or_default();
    mnemonic.starts_with("s_cbranch_exec") || mnemonic.contains("saveexec")
}

/// The bytes a lane moves for a vector-memory mnemonic: its `_b<bits>`, `_u8`/`_i16`/`_d16_hi` or
/// gfx9 `_dword`/`_dwordx<n>` suffix.
pub fn load_bytes(asm: &str) -> Option<u64> {
    let mnemonic = asm.split_whitespace().next()?;
    match mnemonic.rsplit('_').next()? {
        "u8" | "i8" | "ubyte" | "sbyte" => Some(1),
        "u16" | "i16" | "ushort" | "sshort" | "hi" | "d16" => Some(2),
        "dword" => Some(4),
        s => match s.strip_prefix("dwordx") {
            Some(n) => n.parse::<u64>().ok().map(|n| 4 * n),
            None => s.strip_prefix('b')?.parse::<u64>().ok().map(|bits| bits / 8),
        },
    }
}

/// What the loop's global loads ask of memory at the dispatch's resident waves.
fn demand(a: &Analysis) -> Option<Demand> {
    let l = a.main_loop.as_ref().filter(|l| l.trips.total > 0 && l.partition.waves > 0 && l.partition.cycles > 0)?;
    let global = |r: &&Row| {
        let kind = classify(&r.asm).kind;
        r.class == Class::VmemLoad && matches!(kind, Kind::Memory(Memory { space: Space::Global | Space::Buffer, .. }))
    };
    let lanes: f64 =
        a.loop_rows().filter(global).filter_map(|r| Some(r.hits as f64 * load_bytes(&r.asm)? as f64)).sum();
    let bytes_per_trip = (lanes * LANES as f64 / l.trips.total as f64).round() as u64;
    let waves: f64 = a.occupancy.ses.iter().map(|s| s.residency.mean()).sum();
    let cycles_per_trip = l.partition.cycles as f64 / l.trips.total as f64;
    let sclk_mhz = a.sclk_mhz?;
    (bytes_per_trip > 0 && waves > 0.0).then(|| Demand {
        bytes_per_trip,
        waves,
        cycles_per_trip,
        sclk_mhz,
        gbs: bytes_per_trip as f64 * waves * sclk_mhz as f64 / cycles_per_trip / 1e3,
        bandwidth_gbs: a.labels.get("model.bandwidth_gbs").and_then(|b| b.parse().ok()),
    })
}

/// The limiters the analysis supports.
pub(super) fn rank(a: &Analysis) -> Verdict {
    let perturbed = slowdown(a).is_some_and(|r| r > PERTURBED);
    let implausible = implausible(a);
    let untimed = a.occupancy.untimed_matrix;
    let approx = |more: &[Approx]| {
        let base = [perturbed.then_some(Approx::Perturbed), implausible.then_some(Approx::ImplausibleTime)];
        base.into_iter().flatten().chain(more.iter().copied()).collect::<Vec<_>>()
    };
    let raw_flag: &[Approx] = if untimed { &[Approx::UntimedMatrix] } else { &[] };
    let finding = |kind, (share, raw), evidence, knob: &str, approx| Finding {
        kind,
        share,
        raw,
        evidence,
        knob: knob.into(),
        approx,
    };
    let mut v = Verdict { demand: demand(a), ..Verdict::default() };
    let l = a.main_loop.as_ref().filter(|l| l.partition.cycles > 0 && l.trips.total > 0);
    let observed = observed(a);
    let limit = a.occupancy.limit.clone();
    let occupancy_knob = |observed: f64| match &limit {
        Some(l) if observed + 0.5 >= f64::from(l.waves_per_simd) => {
            "the kernel's resources cap the waves: free the limiter (fewer VGPRs, less LDS a workgroup)"
        }
        Some(_) => "fewer waves resident than the kernel allows: more workgroups in flight",
        None => {
            "more resident waves would fill the idle cycles; the capture records no resources to tell what limits them"
        }
    };
    match busy(a).filter(|_| !untimed) {
        Some((matrix, valu, any)) => {
            let queued = l.map(|l| {
                let trips = l.trips.total as f64;
                let raw = l.partition.idled(&[Class::Matrix]) as f64 / trips;
                Cycles { raw, exposed: l.exposed.idled(&[Class::Matrix]) as f64 / trips }
            });
            v.ceiling = (matrix > 0.0).then_some(Ceiling { matrix, valu, any, queued });
            let evidence = Evidence::Busy { matrix, valu, any };
            if matrix >= MATRIX_BOUND {
                let knob =
                    "near the matrix pipe's throughput: only less matrix work (a cheaper type, fewer ops) helps much";
                v.bound.push(finding(FindingKind::MatrixBound, (matrix, matrix), evidence.clone(), knob, approx(&[])));
            }
            if any >= ISSUE_BOUND && matrix < ISSUE_MATRIX * any {
                let knob =
                    "the SIMD issues nearly every cycle, mostly not matrix work: cut the loop's other instructions";
                v.bound.push(finding(FindingKind::IssueBound, (any, any), evidence, knob, approx(&[])));
            }
            if let Some(observed) = observed.filter(|_| 1.0 - any >= SIMD_IDLE) {
                let idle = 1.0 - any;
                let need = observed / any.max(1e-9);
                let evidence = Evidence::Occupancy { observed, need, idle: Some(idle), limit: limit.clone() };
                let knob = occupancy_knob(observed);
                v.bound.push(finding(FindingKind::OccupancyLimited, (idle, idle), evidence, knob, approx(&[])));
            }
        }
        // gfx11: no busy share to trust; the waves that would hide the loop's waits behind its issue.
        None => {
            if let Some((l, observed)) = l.zip(observed) {
                let p = &l.partition;
                let (stall, issue) = (p.waited(|_| true) + p.barrier, p.issued(&[]) + p.stalled(&[]));
                let need = 1.0 + stall as f64 / issue.max(1) as f64;
                if observed < need {
                    let share = stall as f64 / p.cycles as f64 * (1.0 - observed / need);
                    let evidence = Evidence::Occupancy { observed, need, idle: None, limit: limit.clone() };
                    let knob = occupancy_knob(observed);
                    let f = finding(FindingKind::OccupancyLimited, (share, share), evidence, knob, approx(raw_flag));
                    v.bound.push(f);
                }
            }
        }
    }
    sort(&mut v.bound);
    let Some(l) = l else { return v };
    let (p, e, trips) = (&l.partition, &l.exposed, l.trips.total as f64);
    let share = |cycles: i64| cycles as f64 / p.cycles as f64;
    let cycles = |raw: i64, exposed: i64| Cycles { raw: raw as f64 / trips, exposed: exposed as f64 / trips };
    // Where hidden cycles mean nothing (gfx11), a share is the raw one.
    let ranked = |raw: i64, exposed: i64| (share(if untimed { raw } else { exposed }), share(raw));

    let vmem = |on: WaitOn| matches!(on.counter, Counter::Load | Counter::Sample | Counter::Bvh);
    let (waited, waited_e) = (p.waited(vmem), e.waited(vmem));
    let (stalled, stalled_e) = (p.stalled(&[Class::VmemLoad]), e.stalled(&[Class::VmemLoad]));
    let (s, raw) = ranked(waited + stalled, waited_e + stalled_e);
    if s >= LOAD_EXPOSED {
        let loads = loads(a, Class::VmemLoad, trips);
        let knob = match loads.first().filter(|l| l.short_cover()) {
            Some(l) => format!(
                "this use forces the wait right behind the load: move or drop it ({:#x} waits for {:#x}, then `{}`)",
                l.wait, l.pc, l.consumer
            ),
            None => "latency: more work between a load and its wait (prefetch distance, ring slots)".into(),
        };
        let guessed = loads.iter().any(|l| l.approx > 0).then_some(Approx::ApproxBindings);
        let more: Vec<Approx> = raw_flag.iter().copied().chain(guessed).collect();
        let store = |on: WaitOn| on.counter == Counter::Store;
        let evidence = Evidence::Waits {
            load: cycles(waited, waited_e),
            store: cycles(p.waited(store), e.waited(store)),
            stall: cycles(stalled, stalled_e),
            loads,
            stalls: stalls(a, Class::VmemLoad, trips),
        };
        v.findings.push(finding(FindingKind::LoadExposed, (s, raw), evidence, &knob, approx(&more)));
    }

    let lds = |store: bool| move |on: WaitOn| matches!(on.counter, Counter::Ds | Counter::Lgkm) && on.store == store;
    let load = cycles(p.waited(lds(false)), e.waited(lds(false)));
    let store = cycles(p.waited(lds(true)), e.waited(lds(true)));
    let stall = cycles(p.stalled(&[Class::Lds]), e.stalled(&[Class::Lds]));
    let waits = (p.waited(lds(false)) + p.waited(lds(true)), e.waited(lds(false)) + e.waited(lds(true)));
    // gfx11 leaves an untimed matrix op's cycles on the stall of the op after it: waits alone count.
    let (s, raw) = match untimed {
        true => (share(waits.0), share(waits.0)),
        false => ranked(waits.0 + p.stalled(&[Class::Lds]), waits.1 + e.stalled(&[Class::Lds])),
    };
    if s >= LDS_EXPOSED {
        let pick = |c: Cycles| if untimed { c.raw } else { c.exposed };
        let parts = [(pick(store), 0), (pick(load), 1), (if untimed { 0.0 } else { stall.exposed }, 2)];
        let knob = match parts.into_iter().max_by(|x, y| x.0.total_cmp(&y.0)).map_or(1, |p| p.1) {
            0 => {
                "the ds_store flush before a barrier: fewer, wider stores, or more work between the stores and the barrier"
            }
            1 => "more work between a ds load and its use (load the next fragment earlier), fewer and wider ds loads",
            _ => "the LDS queue is full before ds ops issue: fewer and wider ds ops, a conflict-free layout",
        };
        let (loads, stalls) = (loads(a, Class::Lds, trips), stalls(a, Class::Lds, trips));
        let evidence = Evidence::Waits { load, store, stall, loads, stalls };
        v.findings.push(finding(FindingKind::LdsExposed, (s, raw), evidence, knob, approx(raw_flag)));
    }

    if share(p.barrier) >= BARRIER {
        let evidence = Evidence::Barrier { cycles: cycles(p.barrier, e.barrier), sites: l.barriers.clone() };
        let knob = "drop or merge a barrier only if its exposed share passes 10%";
        v.findings.push(finding(FindingKind::Barrier, ranked(p.barrier, e.barrier), evidence, knob, approx(raw_flag)));
    }

    let alu = [Class::Valu, Class::Salu];
    let matrix = p.issued(&[Class::Matrix]);
    let divergent: Vec<(u64, String)> =
        a.loop_rows().filter(|r| divergent(&r.asm)).map(|r| (r.pc, r.asm.clone())).collect();
    // gfx11's matrix issue is a token a op: no measure to set the rest against.
    let heavy = !untimed && matrix > 0 && p.issued(&alu) as f64 >= LOOP_OVERHEAD * matrix as f64;
    if heavy || !divergent.is_empty() {
        let knob = "hoist address and guard arithmetic out of the loop; make guards uniform (no exec branches)";
        let valu_hits = a.loop_rows().filter(|r| r.class == Class::Valu).map(|r| r.hits).sum::<u64>() as f64 / trips;
        let evidence = Evidence::Overhead {
            valu: p.issued(&[Class::Valu]) as f64 / trips,
            salu: p.issued(&[Class::Salu]) as f64 / trips,
            matrix: matrix as f64 / trips,
            valu_hits,
            divergent,
        };
        let s = ranked(p.issued(&alu), e.issued(&alu));
        v.findings.push(finding(FindingKind::LoopOverhead, s, evidence, knob, approx(raw_flag)));
    }
    sort(&mut v.findings);
    v
}

fn sort(findings: &mut [Finding]) {
    findings.sort_by(|a, b| b.share.total_cmp(&a.share).then(a.kind.cmp(&b.kind)));
}

fn pct(share: f64) -> f64 {
    100.0 * share
}

/// The verdict section: the kernel's ceiling and class, the loop's findings with their evidence and
/// knob, then what makes them approximate.
pub(super) fn text(a: &Analysis, o: &mut String) {
    let (unit, of) = match &a.main_loop {
        Some(l) if l.trips.labelled => {
            ("trip", format!("a trip ({} an iteration, model.trips)", l.trips.per_iteration))
        }
        Some(_) => ("iteration", "an iteration (no model.trips label)".into()),
        None => ("trip", "- (no main loop)".into()),
    };
    let a_unit = if unit == "trip" { "a trip" } else { "an iteration" };
    let v = &a.verdict;
    let _ = writeln!(o, "== verdict: cycles are a traced wave's, {of}; exposed: not hidden by other pipes' work");
    if let Some(c) = &v.ceiling {
        let queued = c.queued.map_or(String::new(), |q| {
            format!("; {:.0} cycles {a_unit} queued for the matrix pipe, {:.0} exposed", q.raw, q.exposed)
        });
        let _ = writeln!(
            o,
            "ceiling    matrix pipe {:.1}% busy over the steady window: removing every stall gains at most {:.2}x{queued}",
            pct(c.matrix),
            1.0 / c.matrix
        );
    }
    if let Some(d) = &v.demand {
        let ratio = d.bandwidth_gbs.map_or(String::new(), |b| format!(", {:.0}% of {b:.0} GB/s", 100.0 * d.gbs / b));
        let _ = writeln!(
            o,
            "demand     {} global bytes {a_unit} a wave x {:.0} resident waves / {:.0} cycles {a_unit} at {} MHz: {:.0} GB/s requested from L2 and DRAM{ratio}",
            d.bytes_per_trip, d.waves, d.cycles_per_trip, d.sclk_mhz, d.gbs
        );
    }
    let flags = |f: &Finding| {
        let approx: Vec<&str> = f.approx.iter().map(|a| a.name()).collect();
        if approx.is_empty() { String::new() } else { format!(" (approx: {})", approx.join(", ")) }
    };
    for f in &v.bound {
        let what = match &f.evidence {
            Evidence::Busy { matrix, valu, any } => {
                format!("pipes busy: matrix {:.1}% valu {:.1}% any {:.1}%", pct(*matrix), pct(*valu), pct(*any))
            }
            Evidence::Occupancy { observed, need, idle, limit } => {
                let limit = limit.as_ref().map_or("unknown".into(), |l| {
                    let by: Vec<&str> = l.by.iter().map(|b| b.name()).collect();
                    format!("{} by {}", l.waves_per_simd, by.join("+"))
                });
                let idle = idle.map_or("waits uncovered".into(), |i| format!("SIMD idle {:.1}%", pct(i)));
                format!("{idle}: {observed:.1} waves a SIMD, ~{need:.1} would fill it; limit {limit}")
            }
            _ => String::new(),
        };
        let _ = writeln!(o, "bound      {:<17} {:>5.1}%  {what}{}", f.kind.name(), pct(f.share), flags(f));
        let _ = writeln!(o, "   knob: {}", f.knob);
    }
    if v.findings.is_empty() {
        let _ = writeln!(o, "none       no loop finding passes its threshold");
    }
    let untimed = a.occupancy.untimed_matrix;
    let c = |c: &Cycles| match untimed {
        true => format!("{:.0}", c.raw),
        false => format!("{:.0} ({:.0} exposed)", c.raw, c.exposed),
    };
    for (rank, f) in v.findings.iter().enumerate() {
        let what = match &f.evidence {
            Evidence::Waits { load, store, stall, .. } => {
                let store = if store.raw > 0.0 { format!(", on stores {}", c(store)) } else { String::new() };
                let stall = match untimed {
                    true => format!("pipe contention (matrix untimed) {:.0}", stall.raw),
                    false => format!("stalled issuing {}", c(stall)),
                };
                format!("cycles {a_unit}: waiting on loads {}{store}; {stall}", c(load))
            }
            Evidence::Barrier { cycles, .. } => {
                let exposed = if untimed { String::new() } else { format!(", {:.0} exposed", cycles.exposed) };
                format!(
                    "barrier (sink): {:.0}/{unit} parked for the workgroup's slowest wave{exposed}; this is other findings' time, fix those first",
                    cycles.raw
                )
            }
            Evidence::Overhead { valu, salu, matrix, valu_hits, divergent } => {
                let ratio = match untimed {
                    true => String::new(),
                    false => format!(" beside matrix {matrix:.0} ({:.0}%)", 100.0 * (valu + salu) / matrix.max(1e-9)),
                };
                format!(
                    "valu {valu:.0} + salu {salu:.0} issue cycles {a_unit}{ratio}, {valu_hits:.0} valu ops {a_unit}, {} divergent branches",
                    divergent.len()
                )
            }
            _ => String::new(),
        };
        let (share, raw) = (pct(f.share), pct(f.raw));
        let _ = writeln!(o, "{:>2} {:<17} {share:>5.1}% (raw {raw:.1}%)  {what}{}", rank + 1, f.kind.name(), flags(f));
        match &f.evidence {
            Evidence::Waits { loads, stalls, .. } => {
                for s in stalls {
                    let (pc, per, exposed) = (s.pc, s.per_trip, s.exposed_per_trip);
                    let _ = writeln!(
                        o,
                        "   {pc:#x} stalled {per:.0}/{unit} before issue, exposed {exposed:.0}  {}{}",
                        s.asm,
                        a.at(pc)
                    );
                }
                for l in loads {
                    let what = if stores(Some(&classify(&l.asm).kind)) { "store" } else { "load" };
                    let latency = match l.latency {
                        Some(s) if l.stalled * 2 >= l.waits => format!("latency med {}", s.median),
                        _ => "latency <= cover".into(),
                    };
                    let distance: Vec<String> = l.distance.iter().map(|(d, n)| format!("{d}:{n}")).collect();
                    let _ = writeln!(
                        o,
                        "   {:#x} {what} exposed {:.0}/{unit} at wait {:#x} (then {}): cover med {}, {latency}, {}/{} stalled, distance {}  {}{}",
                        l.pc,
                        l.exposed_per_trip,
                        l.wait,
                        l.consumer,
                        l.cover.median,
                        l.stalled,
                        l.waits,
                        distance.join(","),
                        l.asm,
                        a.at(l.pc)
                    );
                }
            }
            Evidence::Barrier { sites, .. } => {
                let asm: HashMap<u64, &str> = a.pcs.iter().map(|r| (r.pc, r.asm.as_str())).collect();
                for s in sites {
                    let causes: Vec<String> = s
                        .causes
                        .iter()
                        .take(EVIDENCE)
                        .map(|(pc, n)| format!("{pc:#x} {} x{n}{}", asm.get(pc).copied().unwrap_or("?"), a.at(*pc)))
                        .collect();
                    let after =
                        if causes.is_empty() { String::new() } else { format!(", after {}", causes.join("; ")) };
                    let share = 100.0 * s.last as f64 / s.hits.max(1) as f64;
                    let _ = writeln!(
                        o,
                        "   {:#x} arrived last {}/{} ({share:.0}%){after}{}",
                        s.pc,
                        s.last,
                        s.hits,
                        a.at(s.pc)
                    );
                }
            }
            Evidence::Overhead { divergent, .. } => {
                for (pc, asm) in divergent {
                    let _ = writeln!(o, "   {pc:#x} {asm}{}", a.at(*pc));
                }
            }
            _ => {}
        }
        let _ = writeln!(o, "   knob: {}", f.knob);
    }
    if let Some(r) = slowdown(a).filter(|&r| r > PERTURBED) {
        let why = match a.pstate {
            Some(Pstate::Unpinned) => "unpinned: not the clock".to_string(),
            Some(p) => format!("pinned {}: its lower clock moves cycle shares", p.name()),
            None => "pstate unrecorded".into(),
        };
        let _ = writeln!(o, "note       the traced run took {r:.2}x the untraced ({why})");
    }
    if a.occupancy.untimed_matrix {
        let _ = writeln!(
            o,
            "note       gfx11 traces matrix ops as 1-cycle tokens: no ceiling, raw shares (hidden cycles unknown), matrix issue holds its stall"
        );
    }
}
