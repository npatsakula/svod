//! The main loop, cycle by cycle: its traced waves' cycles split into buckets that add up to them
//! exactly ([`Partition`]), each bucket also less what other pipes' work on the SIMD hid (AMD's hidden
//! latency): the exposed cycles. Each cycle is also charged to an instruction ([`Charge`]), a wait's
//! to the one that bound it.

use std::collections::{BTreeMap, HashMap};
use std::ops::{Index, IndexMut, RangeInclusive};

use serde::{Deserialize, Serialize};
use svod_device::sqtt::asm::{Access, Counter, Kind, Memory};
use svod_device::sqtt::hidden::Hiding;
use svod_device::sqtt::wave::{Instruction, Wave};

use super::{Class, Insts, Stmt};

/// A barrier wait this short: its wave arrived last and the barrier released at once. The fixtures'
/// waits split into a floor of 8 to 15 cycles and a tail from tens of cycles up.
pub(crate) const LAST_ARRIVER: i64 = 16;

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

/// What a record's cycles count as in a [`Partition`]: its issue (and the stall before it), a wait's
/// or a barrier's.
#[derive(Debug, Clone, Copy)]
enum Counts {
    Issue(Class),
    Wait(WaitOn),
    Barrier,
}

/// What a cycle of the main loop counts as, in report order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bucket {
    /// An instruction's issue, by its class.
    Issue(Class),
    /// The stall before it issued: a pipe or queue still busy.
    Stall(Class),
    /// A wait, by what it waited for.
    Wait(WaitOn),
    Barrier,
    /// A gap between records, by the class of the record after it: the SIMD issuing for another
    /// wave, or a dependency the trace does not show.
    Idle(Class),
}

pub(super) fn stores(kind: Kind) -> bool {
    matches!(
        kind,
        Kind::Memory(Memory { access: Access::Store | Access::Atomic { returns: false } | Access::Writeback, .. })
    )
}

/// The partition's reading of every instruction: its kind by PC, and whether it is gfx12's
/// `s_barrier_signal`, where a wave arrives at a barrier it waits for later.
pub(super) struct Buckets<'a>(pub &'a Insts<'a>);

/// What one SE's traced waves did in the main loop.
pub(super) struct Pass {
    pub partition: Partition,
    pub barriers: BTreeMap<u64, BarrierSite>,
    /// The partition's cycles by the instruction they are charged to: a wait's to the instruction
    /// that bound it.
    pub charges: BTreeMap<u64, Charge>,
}

/// What an instruction's cycles in the loop went to: a [`Bucket`] less its class or counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Issue,
    Stall,
    Wait,
    Barrier,
    Idle,
}

/// Cycles by [`Part`], raw and exposed: an instruction's in the loop summed over the traced waves,
/// or a statement's a trip ([`Statement`](super::Statement)).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(from = "Parts", into = "Parts")]
pub struct Charge(pub [PerTrip; 5]);

impl Index<Part> for Charge {
    type Output = PerTrip;

    fn index(&self, part: Part) -> &PerTrip {
        &self.0[part as usize]
    }
}

impl IndexMut<Part> for Charge {
    fn index_mut(&mut self, part: Part) -> &mut PerTrip {
        &mut self.0[part as usize]
    }
}

impl Charge {
    /// `o`'s cycles over `over` added part by part.
    pub fn add(&mut self, o: &Self, over: f64) {
        for (at, c) in self.0.iter_mut().zip(o.0) {
            (at.raw, at.exposed) = (at.raw + c.raw / over, at.exposed + c.exposed / over);
        }
    }

    pub fn raw(&self) -> f64 {
        self.0.iter().map(|c| c.raw).sum()
    }

    pub fn exposed(&self) -> f64 {
        self.0.iter().map(|c| c.exposed).sum()
    }
}

/// A [`Charge`] as JSON: a field a part.
#[derive(Serialize, Deserialize)]
struct Parts {
    issue: PerTrip,
    stall: PerTrip,
    wait: PerTrip,
    barrier: PerTrip,
    idle: PerTrip,
}

impl From<Parts> for Charge {
    fn from(Parts { issue, stall, wait, barrier, idle }: Parts) -> Self {
        Self([issue, stall, wait, barrier, idle])
    }
}

impl From<Charge> for Parts {
    fn from(Charge([issue, stall, wait, barrier, idle]): Charge) -> Self {
        Self { issue, stall, wait, barrier, idle }
    }
}

/// `cycles` into `at`, raw and less what `hidden` covers of them. Saturating, as every sum of the
/// partition: a corrupt trace's cycles may pass `i64`.
fn charge(at: &mut [i64; 2], cycles: i64, hidden: i64) {
    (at[0], at[1]) = (at[0].saturating_add(cycles), at[1].saturating_add(cycles - hidden.clamp(0, cycles)));
}

impl Buckets<'_> {
    fn kind(&self, pc: u64) -> Kind {
        self.0.get(&pc).map_or(Kind::Other, |i| i.1)
    }

    fn signals(&self, pc: u64) -> bool {
        self.0.get(&pc).is_some_and(|i| i.0.trim_start().starts_with("s_barrier_signal"))
    }

    /// `binder`: the `(counter, pc)` of the instruction a wait record waited for.
    fn counts(&self, pc: u64, binder: Option<&(Counter, u64)>) -> Counts {
        let kind = self.kind(pc);
        match (Class::from(kind), kind) {
            (Class::Barrier, _) => Counts::Barrier,
            (Class::WaitVmem | Class::WaitLgkm, Kind::Wait(wait)) => Counts::Wait(match binder {
                Some(&(counter, at)) => WaitOn { counter, store: self.0.get(&at).is_some_and(|i| stores(i.1)) },
                None => {
                    let counter = wait.counts().map(|(c, _)| c).find(|&c| c != Counter::Exp).expect("a counted wait");
                    WaitOn { counter, store: counter == Counter::Store }
                }
            }),
            (class, _) => Counts::Issue(class),
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
        let mut p = Partition { waves: 0, buckets: BTreeMap::from([(Bucket::Barrier, [0; 2])]) };
        let mut sum = |bucket, cycles, hidden| charge(p.buckets.entry(bucket).or_default(), cycles, hidden);
        let mut waves_in = 0;
        let mut barriers = BTreeMap::<u64, BarrierSite>::new();
        let mut charges = BTreeMap::<u64, Charge>::new();
        let none = HashMap::new();
        for (w, wave) in waves.iter().enumerate() {
            let bound = bound.get(w).unwrap_or(&none);
            let (mut last, mut run, mut entered) = (wave.begin, None::<i64>, false);
            // The record that held the wave longest since its last barrier, `(pc, cycles)`, and that
            // as of its signal to the barrier it waits at next.
            let (mut worst, mut arrived) = (None::<(u64, i64)>, None::<(u64, i64)>);
            for (seq, r) in wave.instructions.iter().enumerate() {
                let hidden = hiding.record(last, r);
                last = r.time + r.duration;
                if !body.contains(&r.pc) {
                    run = None;
                    continue;
                }
                entered = true;
                let cursor = run.get_or_insert(r.time);
                let gap = (r.time - *cursor).max(0);
                if gap > 0 {
                    sum(Bucket::Idle(Class::from(self.kind(r.pc))), gap, hidden.idle);
                    charges.entry(r.pc).or_default()[Part::Idle].charge(gap, hidden.idle);
                    *cursor = r.time;
                }
                let end = r.time + r.duration;
                let cycles = (end - *cursor).max(0);
                let binder = bound.get(&seq);
                let counts = self.counts(r.pc, binder);
                let held = match counts {
                    Counts::Issue(class) => {
                        let stalled = ((r.time + r.stall).min(end) - *cursor).max(0);
                        sum(Bucket::Stall(class), stalled, hidden.stall);
                        sum(Bucket::Issue(class), cycles - stalled, hidden.issue);
                        let at = charges.entry(r.pc).or_default();
                        at[Part::Stall].charge(stalled, hidden.stall);
                        at[Part::Issue].charge(cycles - stalled, hidden.issue);
                        stalled
                    }
                    Counts::Wait(on) => {
                        sum(Bucket::Wait(on), cycles, hidden.stall + hidden.issue);
                        let at = charges.entry(binder.map_or(r.pc, |b| b.1)).or_default();
                        at[Part::Wait].charge(cycles, hidden.stall + hidden.issue);
                        cycles
                    }
                    Counts::Barrier => {
                        sum(Bucket::Barrier, cycles, hidden.stall + hidden.issue);
                        charges.entry(r.pc).or_default()[Part::Barrier].charge(cycles, hidden.stall + hidden.issue);
                        0
                    }
                };
                if held + gap > 0 && worst.is_none_or(|w| held + gap > w.1) {
                    worst = Some((r.pc, held + gap));
                }
                if let Counts::Barrier = counts {
                    let held = worst.take();
                    match self.signals(r.pc) {
                        true => arrived = held,
                        false => site(&mut barriers, r, arrived.take().or(held)),
                    }
                }
                *cursor += cycles;
            }
            waves_in += usize::from(entered);
        }
        p.waves = waves_in;
        p.buckets.retain(|&b, c| b == Bucket::Barrier || c[0] > 0);
        Pass { partition: p, barriers, charges }
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
    /// Its statement, where the capture has lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stmt: Option<Stmt>,
}

impl BarrierSite {
    /// The SEs' sites by PC, less `base`.
    pub(super) fn merge(sites: impl IntoIterator<Item = Self>, base: u64) -> Vec<Self> {
        let mut by_pc = BTreeMap::<u64, (Self, BTreeMap<u64, usize>)>::new();
        for s in sites {
            let site = || (Self { pc: s.pc - base, ..Self::default() }, BTreeMap::new());
            let (at, causes) = by_pc.entry(s.pc - base).or_insert_with(site);
            (at.hits, at.last) = (at.hits + s.hits, at.last + s.last);
            s.causes.iter().for_each(|&(pc, n)| *causes.entry(pc - base).or_default() += n);
        }
        let sites = by_pc.into_values().map(|(site, causes)| {
            let mut causes: Vec<(u64, usize)> = causes.into_iter().collect();
            causes.sort_by_key(|&(pc, n)| (std::cmp::Reverse(n), pc));
            Self { causes, ..site }
        });
        sites.collect()
    }
}

/// The main loop's traced-wave cycles, each in exactly one [`Bucket`], `[raw, exposed]`: exposed
/// less what other pipes' work on the SIMD hid. A bucket whose raw cycles are none is left out, the
/// barrier's kept.
///
/// Records overlap where an instruction's duration passes the next one's issue: a cycle goes to the
/// earlier record, so a matrix op's stall behind the previous matrix op lands in that one's issue.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Partition {
    pub waves: usize,
    #[serde(with = "pairs")]
    pub buckets: BTreeMap<Bucket, [i64; 2]>,
}

/// A map with non-string keys as JSON: a list of `[key, value]` pairs.
mod pairs {
    use std::collections::BTreeMap;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer, K: Serialize, V: Serialize>(m: &BTreeMap<K, V>, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(m)
    }

    pub fn deserialize<'de, D: Deserializer<'de>, K: Deserialize<'de> + Ord, V: Deserialize<'de>>(
        d: D,
    ) -> Result<BTreeMap<K, V>, D::Error> {
        Ok(Vec::<(K, V)>::deserialize(d)?.into_iter().collect())
    }
}

impl Partition {
    pub fn add(&mut self, o: &Self) {
        self.waves += o.waves;
        for (&b, c) in &o.buckets {
            let at = self.buckets.entry(b).or_default();
            (at[0], at[1]) = (at[0].saturating_add(c[0]), at[1].saturating_add(c[1]));
        }
    }

    /// The cycles of the buckets `keep` keeps, `[raw, exposed]`.
    pub fn of(&self, keep: impl Fn(Bucket) -> bool) -> [i64; 2] {
        let kept = self.buckets.iter().filter(|(b, _)| keep(**b));
        kept.fold([0, 0], |s, (_, c)| [s[0].saturating_add(c[0]), s[1].saturating_add(c[1])])
    }

    /// Every bucket's cycles: the waves' in the loop.
    pub fn cycles(&self) -> [i64; 2] {
        self.of(|_| true)
    }

    /// Issue cycles of `classes`; every class where empty.
    pub fn issued(&self, classes: &[Class]) -> [i64; 2] {
        self.of(|b| matches!(b, Bucket::Issue(c) if classes.is_empty() || classes.contains(&c)))
    }

    /// Stall cycles before an instruction of `classes` issued; every class where empty.
    pub fn stalled(&self, classes: &[Class]) -> [i64; 2] {
        self.of(|b| matches!(b, Bucket::Stall(c) if classes.is_empty() || classes.contains(&c)))
    }

    /// Wait cycles on what `on` keeps.
    pub fn waited(&self, on: impl Fn(WaitOn) -> bool) -> [i64; 2] {
        self.of(|b| matches!(b, Bucket::Wait(w) if on(w)))
    }

    pub fn barrier(&self) -> [i64; 2] {
        self.of(|b| b == Bucket::Barrier)
    }

    /// Idle cycles before an instruction of `classes`; every class where empty.
    pub fn idled(&self, classes: &[Class]) -> [i64; 2] {
        self.of(|b| matches!(b, Bucket::Idle(c) if classes.is_empty() || classes.contains(&c)))
    }
}

/// A trip's cycles of a bucket, and those other pipes' work did not hide.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct PerTrip {
    pub raw: f64,
    pub exposed: f64,
}

impl PerTrip {
    /// `[raw, exposed]` cycles over `trips`.
    pub fn of(cycles: [i64; 2], trips: u64) -> Self {
        let trips = trips.max(1) as f64;
        Self { raw: cycles[0] as f64 / trips, exposed: cycles[1] as f64 / trips }
    }

    /// `cycles` added, raw and less what `hidden` covers of them.
    fn charge(&mut self, cycles: i64, hidden: i64) {
        (self.raw, self.exposed) = (self.raw + cycles as f64, self.exposed + (cycles - hidden.clamp(0, cycles)) as f64);
    }
}
