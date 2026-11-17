//! What limits a kernel's main loop: the findings ([`Finding`]) rank the [`Partition`]'s buckets by
//! exposed share; the pipes' busy shares over the steady window classify the kernel apart
//! ([`Verdict::bound`]).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use strum::IntoStaticStr;
use svod_device::amd::occupancy::Limit;
use svod_device::sqtt::asm::{Counter, Kind, Memory, Space, classify};
use svod_device::sqtt::report::Spread;

use super::{Analysis, BarrierSite, Class, PerTrip, Row, Stmt, WaitOn};

/// Exposed vector-memory waits at a tenth of the loop's cycles; below it, moving loads or deepening
/// the ring is taken to repay less than its register and LDS cost. Heuristic.
pub(crate) const LOAD_EXPOSED: f64 = 0.10;
/// The same tenth for exposed LDS waits and stall. Heuristic.
pub(crate) const LDS_EXPOSED: f64 = 0.10;
/// Barriers at a tenth of the loop are listed; their exposed share at a tenth is worth dropping one.
/// Heuristic.
pub(crate) const BARRIER: f64 = 0.10;
/// Non-matrix VALU and SALU issue at a tenth of the matrix issue: a loop's own bookkeeping (a few
/// address increments a trip) stays under it, per-element guard and address arithmetic passes it.
/// Heuristic.
pub(crate) const LOOP_OVERHEAD: f64 = 0.10;
/// The matrix pipe busy for this share of the steady window or more: near its throughput. Heuristic.
pub(crate) const MATRIX_BOUND: f64 = 0.70;
/// Any pipe busy this share of the window or more, with matrix work under [`ISSUE_MATRIX`] of it:
/// the SIMD issues nearly every cycle, mostly not matrix work. Heuristic.
pub(crate) const ISSUE_BOUND: f64 = 0.90;
/// The matrix share of [`ISSUE_BOUND`]'s busy cycles under which they are mostly not matrix work.
/// Heuristic.
pub(crate) const ISSUE_MATRIX: f64 = 0.60;
/// The traced SIMDs idle (no pipe busy) this share of the steady window: more waves would fill it.
/// Heuristic.
pub(crate) const SIMD_IDLE: f64 = 0.10;
/// The traced run over the untraced past this: its cycle shares no longer describe the untraced
/// kernel (the probes of 2026-11 put an unpinned trace within a few percent of it).
pub(crate) const PERTURBED: f64 = 1.2;
/// Cycles no traced time spans: over a minute at 4 GHz, past any dispatch a profile traces, and far
/// below the 2^60 the wave model clamps a corrupt stream's timestamps to.
pub(crate) const PLAUSIBLE_CYCLES: i64 = 1 << 38;
/// A load whose cover is under this share of its latency waits right behind its issue. Heuristic.
pub(crate) const SHORT_COVER: f64 = 0.25;
/// The loads, stalls and barrier causes a finding lists.
pub(super) const EVIDENCE: usize = 4;
/// Mean resident waves this close under the kernel's limit count as at it: the mean rounds to the
/// limit. Heuristic.
const AT_LIMIT: f64 = 0.5;

/// What limits the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, IntoStaticStr)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum FindingKind {
    MatrixBound,
    LoadExposed,
    LdsExposed,
    Barrier,
    OccupancyLimited,
    LoopOverhead,
    IssueBound,
}

/// Why a finding's numbers are approximate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, IntoStaticStr)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
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
    /// An SE's buffer filled and its tail is lost: the sums undercount.
    Truncated,
    /// The main loop's instructions carry no statements: a trip is taken as one loop iteration,
    /// whatever the iteration unrolls.
    PerIteration,
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
    /// Its statement, where the capture has lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stmt: Option<Stmt>,
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
    /// Its statement, where the capture has lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stmt: Option<Stmt>,
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
        stmt: r.stmt,
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
    Waits { load: PerTrip, store: PerTrip, stall: PerTrip, loads: Vec<Load>, stalls: Vec<Stall> },
    /// The waves' cycles parked at barriers, and per barrier PC who arrived last and after what.
    Barrier { cycles: PerTrip, sites: Vec<BarrierSite> },
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
/// steady window, which the text renders alike.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Finding {
    pub kind: FindingKind,
    pub share: f64,
    pub raw: f64,
    pub evidence: Evidence,
    pub knob: Knob,
    pub approx: Vec<Approx>,
}

/// What a finding says to change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Knob {
    /// Near the matrix pipe's throughput: less matrix work.
    LessMatrixWork,
    /// The SIMD issues nearly every cycle, mostly not matrix work.
    CutIssue,
    /// The kernel's resources cap the resident waves.
    FreeResources,
    /// Fewer waves resident than the kernel allows.
    MoreWorkgroups,
    /// More resident waves would fill the idle cycles; the capture records no resources.
    MoreWaves,
    /// A use right behind a load forces its wait: the wait at `wait` waits for `load`, then
    /// `consumer` issues.
    MoveUse { wait: u64, load: u64, consumer: String },
    /// More work between a load and its wait.
    Latency,
    /// The `ds_store` flush before a barrier.
    LdsFlush,
    /// A ds load's use right behind it.
    LdsUse,
    /// The LDS queue is full before ds ops issue.
    LdsQueue,
    /// A barrier is worth dropping past [`BARRIER`] exposed.
    Barrier,
    /// Address and guard arithmetic in the loop.
    Hoist,
}

/// The traced SIMDs' pipe busy shares over the steady window: removing every stall gains at most
/// `1 / matrix`. `queued`: the loop's idle cycles before matrix ops a trip, waiting for the pipe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ceiling {
    pub matrix: f64,
    pub valu: f64,
    pub any: f64,
    pub queued: Option<PerTrip>,
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
}

/// The main loop's limiters: the kernel's class by its pipes' busy shares over the steady window
/// (`bound`: matrix_bound, issue_bound, occupancy_limited), then the loop's findings by exposed share
/// (`findings`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    pub ceiling: Option<Ceiling>,
    pub demand: Option<Demand>,
    pub bound: Findings,
    pub findings: Findings,
}

/// Findings, the largest share first and a tie by kind.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(from = "Vec<Finding>")]
pub struct Findings(Vec<Finding>);

impl From<Vec<Finding>> for Findings {
    fn from(mut findings: Vec<Finding>) -> Self {
        findings.sort_by(|a, b| b.share.total_cmp(&a.share).then(a.kind.cmp(&b.kind)));
        Self(findings)
    }
}

impl std::ops::Deref for Findings {
    type Target = [Finding];

    fn deref(&self) -> &[Finding] {
        &self.0
    }
}

impl IntoIterator for Findings {
    type Item = Finding;
    type IntoIter = std::vec::IntoIter<Finding>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Findings {
    type Item = &'a Finding;
    type IntoIter = std::slice::Iter<'a, Finding>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
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
                stmt: rows[&b.pc].stmt,
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
pub(crate) fn load_bytes(asm: &str) -> Option<u64> {
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
    let l =
        a.main_loop.as_ref().filter(|l| l.trips.total > 0 && l.partition.waves > 0 && l.partition.cycles()[0] > 0)?;
    let global = |r: &&Row| {
        let kind = classify(&r.asm).kind;
        r.class == Class::VmemLoad && matches!(kind, Kind::Memory(Memory { space: Space::Global | Space::Buffer, .. }))
    };
    let lanes: f64 =
        a.loop_rows().filter(global).filter_map(|r| Some(r.hits as f64 * load_bytes(&r.asm)? as f64)).sum();
    let bytes_per_trip = (lanes * f64::from(a.wave_size?) / l.trips.total as f64).round() as u64;
    let waves: f64 = a.occupancy.ses.iter().map(|s| s.residency.mean()).sum();
    let cycles_per_trip = l.partition.cycles()[0] as f64 / l.trips.total as f64;
    let sclk_mhz = a.sclk_mhz?;
    (bytes_per_trip > 0 && waves > 0.0).then(|| Demand {
        bytes_per_trip,
        waves,
        cycles_per_trip,
        sclk_mhz,
        gbs: bytes_per_trip as f64 * waves * sclk_mhz as f64 / cycles_per_trip / 1e3,
    })
}

/// The limiters the analysis supports.
pub(super) fn rank(a: &Analysis) -> Verdict {
    let perturbed = slowdown(a).is_some_and(|r| r > PERTURBED);
    let implausible = implausible(a);
    let truncated = a.ses.iter().any(|s| s.truncated);
    let untimed = a.occupancy.untimed_matrix;
    let approx = |more: &[Approx]| {
        let base = [
            perturbed.then_some(Approx::Perturbed),
            implausible.then_some(Approx::ImplausibleTime),
            truncated.then_some(Approx::Truncated),
        ];
        base.into_iter().flatten().chain(more.iter().copied()).collect::<Vec<_>>()
    };
    let raw_flag: &[Approx] = if untimed { &[Approx::UntimedMatrix] } else { &[] };
    // A loop finding's figures are a trip's: an iteration's where no statements count its trips.
    let uncounted = !a.main_loop.as_ref().is_some_and(|l| l.trips.counted);
    let per_trip = |more: &[Approx]| {
        let uncounted = uncounted.then_some(Approx::PerIteration);
        approx(&more.iter().copied().chain(uncounted).collect::<Vec<_>>())
    };
    let finding = |kind, (share, raw), evidence, knob, approx| Finding { kind, share, raw, evidence, knob, approx };
    let mut v = Verdict { demand: demand(a), ..Verdict::default() };
    let (mut bound, mut findings) = (vec![], vec![]);
    let l = a.main_loop.as_ref().filter(|l| l.partition.cycles()[0] > 0 && l.trips.total > 0);
    let observed = observed(a);
    let limit = a.occupancy.limit.clone();
    let occupancy_knob = |observed: f64| match &limit {
        Some(l) if observed + AT_LIMIT >= f64::from(l.waves_per_simd) => Knob::FreeResources,
        Some(_) => Knob::MoreWorkgroups,
        None => Knob::MoreWaves,
    };
    match busy(a).filter(|_| !untimed) {
        Some((matrix, valu, any)) => {
            let queued = l.map(|l| PerTrip::of(l.partition.idled(&[Class::Matrix]), l.trips.total));
            v.ceiling = (matrix > 0.0).then_some(Ceiling { matrix, valu, any, queued });
            let evidence = Evidence::Busy { matrix, valu, any };
            if matrix >= MATRIX_BOUND {
                let f = finding(
                    FindingKind::MatrixBound,
                    (matrix, matrix),
                    evidence.clone(),
                    Knob::LessMatrixWork,
                    approx(&[]),
                );
                bound.push(f);
            }
            if any >= ISSUE_BOUND && matrix < ISSUE_MATRIX * any {
                bound.push(finding(FindingKind::IssueBound, (any, any), evidence, Knob::CutIssue, approx(&[])));
            }
            if let Some(observed) = observed.filter(|_| 1.0 - any >= SIMD_IDLE) {
                let idle = 1.0 - any;
                let need = observed / any.max(1e-9);
                let evidence = Evidence::Occupancy { observed, need, idle: Some(idle), limit: limit.clone() };
                let knob = occupancy_knob(observed);
                bound.push(finding(FindingKind::OccupancyLimited, (idle, idle), evidence, knob, approx(&[])));
            }
        }
        // gfx11: no busy share to trust; the waves that would hide the loop's waits behind its issue.
        None => {
            if let Some((l, observed)) = l.zip(observed) {
                let p = &l.partition;
                let (stall, issue) = (p.waited(|_| true)[0] + p.barrier()[0], p.issued(&[])[0] + p.stalled(&[])[0]);
                let need = 1.0 + stall as f64 / issue.max(1) as f64;
                if observed < need {
                    let share = stall as f64 / p.cycles()[0] as f64 * (1.0 - observed / need);
                    let evidence = Evidence::Occupancy { observed, need, idle: None, limit: limit.clone() };
                    let knob = occupancy_knob(observed);
                    let f = finding(FindingKind::OccupancyLimited, (share, share), evidence, knob, approx(raw_flag));
                    bound.push(f);
                }
            }
        }
    }
    v.bound = bound.into();
    let Some(l) = l else { return v };
    let (p, trips) = (&l.partition, l.trips.total);
    let share = |cycles: i64| cycles as f64 / p.cycles()[0] as f64;
    let per_trip_of = |c: [i64; 2]| PerTrip::of(c, trips);
    let add = |a: [i64; 2], b: [i64; 2]| [a[0].saturating_add(b[0]), a[1].saturating_add(b[1])];
    // Where hidden cycles mean nothing (gfx11), a share is the raw one.
    let ranked = |c: [i64; 2]| (share(if untimed { c[0] } else { c[1] }), share(c[0]));

    let vmem = |on: WaitOn| matches!(on.counter, Counter::Load | Counter::Sample | Counter::Bvh);
    let (waited, stalled) = (p.waited(vmem), p.stalled(&[Class::VmemLoad]));
    let (s, raw) = ranked(add(waited, stalled));
    if s >= LOAD_EXPOSED {
        let loads = loads(a, Class::VmemLoad, trips as f64);
        let knob = match loads.first().filter(|l| l.short_cover()) {
            Some(l) => Knob::MoveUse { wait: l.wait, load: l.pc, consumer: l.consumer.clone() },
            None => Knob::Latency,
        };
        let guessed = loads.iter().any(|l| l.approx > 0).then_some(Approx::ApproxBindings);
        let more: Vec<Approx> = raw_flag.iter().copied().chain(guessed).collect();
        let evidence = Evidence::Waits {
            load: per_trip_of(waited),
            store: per_trip_of(p.waited(|on| on.counter == Counter::Store)),
            stall: per_trip_of(stalled),
            loads,
            stalls: stalls(a, Class::VmemLoad, trips as f64),
        };
        findings.push(finding(FindingKind::LoadExposed, (s, raw), evidence, knob, per_trip(&more)));
    }

    let lds = |store: bool| move |on: WaitOn| matches!(on.counter, Counter::Ds | Counter::Lgkm) && on.store == store;
    let (load, store, stall) = (p.waited(lds(false)), p.waited(lds(true)), p.stalled(&[Class::Lds]));
    let waits = add(load, store);
    // gfx11 leaves an untimed matrix op's cycles on the stall of the op after it: waits alone count.
    let (s, raw) = match untimed {
        true => (share(waits[0]), share(waits[0])),
        false => ranked(add(waits, stall)),
    };
    if s >= LDS_EXPOSED {
        let (load, store, stall) = (per_trip_of(load), per_trip_of(store), per_trip_of(stall));
        let pick = |c: PerTrip| if untimed { c.raw } else { c.exposed };
        let parts = [
            (pick(store), Knob::LdsFlush),
            (pick(load), Knob::LdsUse),
            (if untimed { 0.0 } else { stall.exposed }, Knob::LdsQueue),
        ];
        let knob = parts.into_iter().max_by(|x, y| x.0.total_cmp(&y.0)).map_or(Knob::LdsUse, |p| p.1);
        let (loads, stalls) = (loads(a, Class::Lds, trips as f64), stalls(a, Class::Lds, trips as f64));
        let evidence = Evidence::Waits { load, store, stall, loads, stalls };
        findings.push(finding(FindingKind::LdsExposed, (s, raw), evidence, knob, per_trip(raw_flag)));
    }

    let barrier = p.barrier();
    if share(barrier[0]) >= BARRIER {
        let evidence = Evidence::Barrier { cycles: per_trip_of(barrier), sites: l.barriers.clone() };
        let f = finding(FindingKind::Barrier, ranked(barrier), evidence, Knob::Barrier, per_trip(raw_flag));
        findings.push(f);
    }

    let alu = [Class::Valu, Class::Salu];
    let matrix = p.issued(&[Class::Matrix])[0];
    let divergent: Vec<(u64, String)> =
        a.loop_rows().filter(|r| divergent(&r.asm)).map(|r| (r.pc, r.asm.clone())).collect();
    // gfx11's matrix issue is a token a op: no measure to set the rest against.
    let heavy = !untimed && matrix > 0 && p.issued(&alu)[0] as f64 >= LOOP_OVERHEAD * matrix as f64;
    if heavy || !divergent.is_empty() {
        let trips = trips as f64;
        let valu_hits = a.loop_rows().filter(|r| r.class == Class::Valu).map(|r| r.hits).sum::<u64>() as f64 / trips;
        let evidence = Evidence::Overhead {
            valu: p.issued(&[Class::Valu])[0] as f64 / trips,
            salu: p.issued(&[Class::Salu])[0] as f64 / trips,
            matrix: matrix as f64 / trips,
            valu_hits,
            divergent,
        };
        let s = ranked(p.issued(&alu));
        findings.push(finding(FindingKind::LoopOverhead, s, evidence, Knob::Hoist, per_trip(raw_flag)));
    }
    v.findings = findings.into();
    v
}
