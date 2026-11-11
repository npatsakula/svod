//! What limits a kernel's main loop: its traced waves' cycles split into buckets that add up to them
//! exactly ([`Partition`]), and the findings those shares, the pipes' busy shares and the waits'
//! bindings support ([`Finding`]), largest share first.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::ops::RangeInclusive;

use serde::{Deserialize, Serialize};
use svod_device::amd::occupancy::Limit;
use svod_device::sqtt::asm::{Counter, Kind, classify};
use svod_device::sqtt::report::Spread;
use svod_device::sqtt::wave::Wave;

use super::{Analysis, Class, Pstate, Row};

/// A finding on vector-memory waits fires at a tenth of the loop's cycles: below it, moving loads
/// or deepening the ring repays less than its register and LDS cost.
pub const LOAD_EXPOSED: f64 = 0.10;
/// The same tenth for LDS waits: below it a layout or schedule change to LDS repays little.
pub const LDS_EXPOSED: f64 = 0.10;
/// Barriers at a tenth of the loop are worth balancing: in a tuned GEMM they take a few percent.
pub const BARRIER: f64 = 0.10;
/// Non-matrix VALU and SALU at a twentieth of a matrix loop: its own bookkeeping (a few address
/// increments a trip) stays under it, guard and address arithmetic per element passes it.
pub const LOOP_OVERHEAD: f64 = 0.05;
/// The matrix pipe busy for this share of the steady window or more: within reach of its
/// throughput, so what the loop's waits cost is hidden by other waves' matrix work.
pub const MATRIX_BOUND: f64 = 0.85;
/// Any pipe busy this share of the window or more, with matrix work under [`ISSUE_MATRIX`] of it:
/// the SIMD issues nearly every cycle, mostly not matrix work.
pub const ISSUE_BOUND: f64 = 0.90;
pub const ISSUE_MATRIX: f64 = 0.60;
/// The traced run over the untraced past this: its cycle shares no longer describe the untraced
/// kernel (the probes of 2026-11 put an unpinned trace within a few percent of it).
pub const PERTURBED: f64 = 1.2;
/// A load's median latency at the most loads in flight this times that at the fewest: queueing in
/// the memory system, not a fixed latency.
pub const QUEUEING: f64 = 1.5;
/// The stalled waits a latency-over-loads-in-flight bucket needs to count.
const QUEUEING_SAMPLES: usize = 8;
/// The loads a wait finding lists.
const EVIDENCE: usize = 4;

/// What a record's cycles count as in a [`Partition`].
#[derive(Debug, Clone, Copy)]
enum Bucket {
    Issue(Class),
    Wait(Counter),
    Barrier,
}

fn bucket(line: &str) -> Bucket {
    let kind = classify(line).kind;
    match (Class::from(kind), kind) {
        (Class::Barrier, _) => Bucket::Barrier,
        (Class::WaitVmem | Class::WaitLgkm, Kind::Wait(wait)) => {
            // The counter its class is named for: vector memory first, as `Class` reads it.
            Bucket::Wait(wait.counts().map(|(c, _)| c).find(|&c| c != Counter::Exp).expect("a counted wait"))
        }
        (class, _) => Bucket::Issue(class),
    }
}

/// Every instruction's [`Bucket`], by PC.
pub(super) struct Buckets(HashMap<u64, Bucket>);

impl Buckets {
    pub(super) fn new(lines: &[(u64, &str)]) -> Self {
        Self(lines.iter().map(|&(pc, line)| (pc, bucket(line))).collect())
    }

    /// The cycles of `waves` in the loop `body`: each wave's from its first record there to the end
    /// of its last, every cycle in one bucket. A record takes the cycles of its span the earlier
    /// records did not cover, its stall's apart; a gap between records is idle.
    pub(super) fn partition(&self, waves: &[Wave], body: RangeInclusive<u64>) -> Partition {
        let (mut issue, mut stall) = (BTreeMap::<Class, i64>::new(), BTreeMap::<Class, i64>::new());
        let mut wait = BTreeMap::<Counter, i64>::new();
        let mut p = Partition::default();
        for wave in waves {
            let inside = |i: &svod_device::sqtt::wave::Instruction| body.contains(&i.pc);
            let (Some(first), Some(last)) =
                (wave.instructions.iter().position(inside), wave.instructions.iter().rposition(inside))
            else {
                continue;
            };
            let records = &wave.instructions[first..=last];
            let mut cursor = records[0].time;
            for r in records {
                if r.time > cursor {
                    p.idle += r.time - cursor;
                    cursor = r.time;
                }
                let end = r.time + r.duration;
                let cycles = (end - cursor).max(0);
                match self.0.get(&r.pc).copied().unwrap_or(Bucket::Issue(Class::Other)) {
                    Bucket::Issue(class) => {
                        let stalled = ((r.time + r.stall).min(end) - cursor).max(0);
                        *stall.entry(class).or_default() += stalled;
                        *issue.entry(class).or_default() += cycles - stalled;
                    }
                    Bucket::Wait(counter) => *wait.entry(counter).or_default() += cycles,
                    Bucket::Barrier => p.barrier += cycles,
                }
                cursor += cycles;
            }
            p.cycles += cursor - records[0].time;
            p.waves += 1;
        }
        let nonzero = |m: BTreeMap<Class, i64>| m.into_iter().filter(|c| c.1 > 0).collect();
        (p.issue, p.stall, p.wait) = (nonzero(issue), nonzero(stall), wait.into_iter().collect());
        p
    }
}

/// The main loop's traced-wave cycles, each in exactly one bucket: an instruction's `issue` and the
/// `stall` before it by class (a pipe or queue still busy), `wait` by the counter a wait is on,
/// `barrier`, and `idle` between records (the SIMD issuing for another wave, or a dependency the
/// trace does not show). They add up to `cycles`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Partition {
    pub waves: usize,
    pub cycles: i64,
    pub issue: Vec<(Class, i64)>,
    pub stall: Vec<(Class, i64)>,
    pub wait: Vec<(Counter, i64)>,
    pub barrier: i64,
    pub idle: i64,
}

fn merge<K: Ord + Copy>(into: &mut Vec<(K, i64)>, from: &[(K, i64)]) {
    let mut sum: BTreeMap<K, i64> = into.iter().copied().collect();
    from.iter().for_each(|&(k, c)| *sum.entry(k).or_default() += c);
    *into = sum.into_iter().collect();
}

impl Partition {
    pub fn add(&mut self, o: &Self) {
        (self.waves, self.cycles, self.barrier, self.idle) =
            (self.waves + o.waves, self.cycles + o.cycles, self.barrier + o.barrier, self.idle + o.idle);
        merge(&mut self.issue, &o.issue);
        merge(&mut self.stall, &o.stall);
        merge(&mut self.wait, &o.wait);
    }

    /// Issue cycles of `classes`; every class where empty.
    pub fn issued(&self, classes: &[Class]) -> i64 {
        self.issue.iter().filter(|(c, _)| classes.is_empty() || classes.contains(c)).map(|c| c.1).sum()
    }

    /// Stall cycles before an instruction of `classes` issued; every class where empty.
    pub fn stalled(&self, classes: &[Class]) -> i64 {
        self.stall.iter().filter(|(c, _)| classes.is_empty() || classes.contains(c)).map(|c| c.1).sum()
    }

    /// Wait cycles on `counters`; every counter where empty.
    pub fn waited(&self, counters: &[Counter]) -> i64 {
        self.wait.iter().filter(|(c, _)| counters.is_empty() || counters.contains(c)).map(|c| c.1).sum()
    }

    /// Every bucket's cycles: `cycles` itself.
    pub fn buckets(&self) -> i64 {
        self.issued(&[]) + self.stalled(&[]) + self.waited(&[]) + self.barrier + self.idle
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
    /// gfx11 traces a matrix op as a 1-cycle token: its issue counts the stall behind it.
    UntimedMatrix,
    /// A listed binding is a guess: an out-of-order counter or a flat op.
    ApproxBindings,
}

impl Approx {
    fn name(self) -> &'static str {
        match self {
            Self::Perturbed => "perturbed",
            Self::UntimedMatrix => "untimed_matrix",
            Self::ApproxBindings => "approx_bindings",
        }
    }
}

/// A load that bound the loop's waits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Load {
    pub pc: u64,
    pub asm: String,
    pub waits: usize,
    pub stalled: usize,
    pub approx: usize,
    /// Its issue to its waits' arrival: the work that hid its latency.
    pub cover: Spread,
    /// Its issue to the release of the waits that stalled.
    pub latency: Option<Spread>,
    /// Its waits' stall in the loop less what other pipes' work hid, a trip.
    pub exposed_per_trip: f64,
    /// `(back edges between its issue and the wait, instances)`.
    pub distance: Vec<(u32, usize)>,
    /// Its latency rises with the loads in flight at its issue ([`QUEUEING`]): bandwidth, not latency.
    pub rising: bool,
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

/// The loop's instructions of `class` that stalled longest before they issued, a cycle a trip of it
/// exposed at least.
fn stalls(a: &Analysis, class: Class, trips: f64) -> Vec<Stall> {
    let mut rows: Vec<&Row> = a.loop_rows().filter(|r| r.class == class && r.stall > 0).collect();
    rows.sort_by_key(|r| (std::cmp::Reverse(r.stall), r.pc));
    let stall = |r: &Row| Stall {
        pc: r.pc,
        asm: r.asm.clone(),
        per_trip: r.stall as f64 / trips,
        exposed_per_trip: (r.stall - r.hidden_stall) as f64 / trips,
    };
    rows.into_iter().map(stall).filter(|s| s.exposed_per_trip >= 1.0).take(EVIDENCE).collect()
}

/// What a finding rests on; cycles are a traced wave's, a trip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Evidence {
    /// The traced SIMDs' pipe busy shares over the steady window, over the traced SEs.
    Busy {
        matrix: f64,
        valu: f64,
        any: f64,
    },
    /// The waits' cycles a trip, the stall before the memory ops issued (a full queue), the loads
    /// that bound the waits, most exposed first, and the ops that stalled longest before issuing.
    Waits {
        wait_per_trip: f64,
        stall_per_trip: f64,
        loads: Vec<Load>,
        stalls: Vec<Stall>,
    },
    Barrier {
        per_trip: f64,
    },
    /// The traced SIMDs' mean resident waves beside the waves that would hide the loop's stall
    /// (`1 + stall / issue`), and the kernel's limit.
    Occupancy {
        observed: f64,
        need: f64,
        stall_per_trip: f64,
        issue_per_trip: f64,
        limit: Option<Limit>,
    },
    /// Non-matrix VALU and SALU a trip, and the loop's divergent branches `(pc, asm)`.
    Overhead {
        valu_per_trip: f64,
        salu_per_trip: f64,
        divergent: Vec<(u64, String)>,
    },
}

/// One limiter of the main loop. `share` is of the loop's traced-wave cycles, or for
/// `matrix_bound`/`issue_bound` the pipe's busy share of the steady window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Finding {
    pub kind: FindingKind,
    pub share: f64,
    pub evidence: Evidence,
    pub knob: String,
    pub approx: Vec<Approx>,
}

/// The traced run's time over the untraced one.
pub(super) fn slowdown(a: &Analysis) -> Option<f64> {
    a.kernel_ns.zip(a.traced_ns).filter(|&(k, _)| k > 0).map(|(k, t)| t as f64 / k as f64)
}

/// The busy shares of the traced SIMDs' pipes, over every traced SE's window.
fn busy(a: &Analysis) -> Option<(f64, f64, f64)> {
    let busy: Vec<_> = a.occupancy.ses.iter().filter_map(|s| s.busy).collect();
    let cycles = busy.iter().map(|b| b.cycles).sum::<i64>();
    let share =
        |f: fn(&svod_device::sqtt::simd::PipeBusy) -> i64| busy.iter().map(f).sum::<i64>() as f64 / cycles as f64;
    (cycles > 0).then(|| (share(|b| b.matrix), share(|b| b.valu), share(|b| b.any)))
}

/// The loads of `class` whose waits in `body` stalled, most exposed first.
fn loads(a: &Analysis, class: Class, body: &RangeInclusive<u64>, trips: f64) -> Vec<Load> {
    let rows: HashMap<u64, &Row> = a.pcs.iter().map(|r| (r.pc, r)).collect();
    let exposed = |wait: u64| rows.get(&wait).map_or(1.0, |r| 1.0 - r.hidden_stall as f64 / r.stall.max(1) as f64);
    let rising = |by: &[(u32, Spread)]| {
        let counted: Vec<_> = by.iter().filter(|(_, l)| l.n >= QUEUEING_SAMPLES).collect();
        counted.first().zip(counted.last()).is_some_and(|(lo, hi)| hi.1.median as f64 >= QUEUEING * lo.1.median as f64)
    };
    let mut loads: Vec<Load> = a
        .waits
        .bindings
        .iter()
        .filter(|b| rows.get(&b.pc).is_some_and(|r| r.class == class))
        .filter_map(|b| {
            let inside = b.waits.iter().filter(|w| body.contains(&w.0));
            let stall: f64 = inside.map(|&(wait, _, stall)| stall as f64 * exposed(wait)).sum();
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
                rising: rising(&b.by_outstanding),
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

/// The findings the analysis supports, largest share first.
pub(super) fn rank(a: &Analysis) -> Vec<Finding> {
    let perturbed = slowdown(a).is_some_and(|r| r > PERTURBED);
    let approx = |more: &[Approx]| {
        let base = perturbed.then_some(Approx::Perturbed);
        base.into_iter().chain(more.iter().copied()).collect::<Vec<_>>()
    };
    let mut findings = vec![];
    let finding =
        |kind, share, evidence, knob: &str, approx| Finding { kind, share, evidence, knob: knob.into(), approx };
    if let Some((matrix, valu, any)) = busy(a).filter(|_| !a.occupancy.untimed_matrix) {
        let evidence = Evidence::Busy { matrix, valu, any };
        if matrix >= MATRIX_BOUND {
            let knob = "at the matrix pipe's throughput: only less matrix work (a cheaper type, fewer ops) helps";
            findings.push(finding(FindingKind::MatrixBound, matrix, evidence.clone(), knob, approx(&[])));
        }
        if any >= ISSUE_BOUND && matrix < ISSUE_MATRIX * any {
            let knob = "the SIMD issues nearly every cycle, mostly not matrix work: cut the loop's other instructions";
            findings.push(finding(FindingKind::IssueBound, any, evidence, knob, approx(&[])));
        }
    }
    let Some(l) = a.main_loop.as_ref().filter(|l| l.partition.cycles > 0 && l.trips.total > 0) else {
        return sorted(findings);
    };
    let (p, trips, body) = (&l.partition, l.trips.total as f64, l.head..=l.back_edge);
    let share = |cycles: i64| cycles as f64 / p.cycles as f64;
    let per_trip = |cycles: i64| cycles as f64 / trips;
    let waits = [
        (FindingKind::LoadExposed, LOAD_EXPOSED, &[Counter::Load, Counter::Sample, Counter::Bvh][..], Class::VmemLoad),
        (FindingKind::LdsExposed, LDS_EXPOSED, &[Counter::Ds, Counter::Lgkm][..], Class::Lds),
    ];
    for (kind, threshold, counters, class) in waits {
        let (waited, stalled) = (p.waited(counters), p.stalled(&[class]));
        let cycles = waited + stalled;
        if share(cycles) < threshold {
            continue;
        }
        let (loads, stalls) = (loads(a, class, &body, trips), stalls(a, class, trips));
        let knob = match (kind, loads.iter().any(|l| l.rising)) {
            (FindingKind::LoadExposed, true) => {
                "bandwidth: the loads' latency rises with those in flight; fewer bytes a trip (more reuse a tile)"
            }
            (FindingKind::LoadExposed, false) => {
                "latency: more work between a load and its wait (issue it earlier: prefetch distance, ring slots)"
            }
            _ => "LDS: more work between a ds load and its use, fewer and wider ds ops, a conflict-free layout",
        };
        // A gfx11 matrix op's cycles land on the stall of what issues after it.
        let untimed = a.occupancy.untimed_matrix && stalled > 0;
        let more = [(loads.iter().any(|l| l.approx > 0), Approx::ApproxBindings), (untimed, Approx::UntimedMatrix)];
        let more: Vec<Approx> = more.into_iter().filter(|m| m.0).map(|m| m.1).collect();
        let evidence =
            Evidence::Waits { wait_per_trip: per_trip(waited), stall_per_trip: per_trip(stalled), loads, stalls };
        findings.push(finding(kind, share(cycles), evidence, knob, approx(&more)));
    }
    if share(p.barrier) >= BARRIER {
        let knob = "balance the waves' work between barriers or drop one; the late arriver is not attributed yet";
        findings.push(finding(
            FindingKind::Barrier,
            share(p.barrier),
            Evidence::Barrier { per_trip: per_trip(p.barrier) },
            knob,
            approx(&[]),
        ));
    }
    let traced: Vec<f64> = a.occupancy.ses.iter().filter_map(|s| s.traced).map(|t| t.mean()).collect();
    // A pipe or queue the wave's own instruction waits for is not hidden by more waves.
    let (stall, issue) = (p.waited(&[]) + p.barrier, p.issued(&[]) + p.stalled(&[]));
    if !traced.is_empty() && issue > 0 {
        let observed = traced.iter().sum::<f64>() / traced.len() as f64;
        let need = 1.0 + stall as f64 / issue as f64;
        if observed < need {
            let limit = a.occupancy.limit.clone();
            let at_limit = limit.as_ref().is_some_and(|l| observed + 0.5 >= f64::from(l.waves_per_simd));
            let knob = match at_limit {
                true => "the kernel's resources cap the waves: free the limiter (fewer VGPRs, less LDS a workgroup)",
                false => "fewer waves resident than the kernel allows: more workgroups in flight",
            };
            let evidence = Evidence::Occupancy {
                observed,
                need,
                stall_per_trip: per_trip(stall),
                issue_per_trip: per_trip(issue),
                limit,
            };
            let untimed: &[Approx] = if a.occupancy.untimed_matrix { &[Approx::UntimedMatrix] } else { &[] };
            findings.push(finding(
                FindingKind::OccupancyLimited,
                share(stall) * (1.0 - observed / need),
                evidence,
                knob,
                approx(untimed),
            ));
        }
    }
    let spent = |class: Class| p.issued(&[class]) + p.stalled(&[class]);
    let (valu, salu) = (spent(Class::Valu), spent(Class::Salu));
    let divergent: Vec<(u64, String)> =
        a.loop_rows().filter(|r| divergent(&r.asm)).map(|r| (r.pc, r.asm.clone())).collect();
    // A loop without matrix work has its VALU and SALU for work, not overhead.
    if (l.matrix && share(valu + salu) >= LOOP_OVERHEAD) || !divergent.is_empty() {
        let knob = "hoist address and guard arithmetic out of the loop; make guards uniform (no exec branches)";
        let evidence = Evidence::Overhead { valu_per_trip: per_trip(valu), salu_per_trip: per_trip(salu), divergent };
        findings.push(finding(FindingKind::LoopOverhead, share(valu + salu), evidence, knob, approx(&[])));
    }
    sorted(findings)
}

fn sorted(mut findings: Vec<Finding>) -> Vec<Finding> {
    findings.sort_by(|a, b| b.share.total_cmp(&a.share).then(a.kind.cmp(&b.kind)));
    findings
}

/// The verdict section: the findings with their evidence and knob, then what makes them approximate.
pub(super) fn text(a: &Analysis, o: &mut String) {
    let (unit, of) = match &a.main_loop {
        Some(l) if l.trips.labelled => {
            ("trip", format!("a trip ({} an iteration, model.trips)", l.trips.per_iteration))
        }
        Some(_) => ("iteration", "an iteration (no model.trips label)".into()),
        None => ("trip", "- (no main loop)".into()),
    };
    let a_unit = if unit == "trip" { "a trip" } else { "an iteration" };
    let _ = writeln!(o, "== verdict: largest share first; cycles are a traced wave's, {of}");
    if a.verdict.is_empty() {
        let _ = writeln!(o, "none       no finding passes its threshold");
    }
    for (rank, f) in a.verdict.iter().enumerate() {
        let approx: Vec<&str> = f.approx.iter().map(|a| a.name()).collect();
        let approx = if approx.is_empty() { String::new() } else { format!(" (approx: {})", approx.join(", ")) };
        let what = match &f.evidence {
            Evidence::Busy { matrix, valu, any } => format!(
                "pipes busy over the steady window: matrix {:.1}% valu {:.1}% any {:.1}%",
                100.0 * matrix,
                100.0 * valu,
                100.0 * any
            ),
            Evidence::Waits { wait_per_trip, stall_per_trip, .. } => {
                format!("{wait_per_trip:.0} cycles {a_unit} waiting, {stall_per_trip:.0} stalled issuing")
            }
            Evidence::Barrier { per_trip } => format!("{per_trip:.0} cycles {a_unit} in barriers"),
            Evidence::Occupancy { observed, need, stall_per_trip, issue_per_trip, limit } => {
                let limit = limit.as_ref().map_or("unknown".into(), |l| {
                    let by: Vec<&str> = l.by.iter().map(|b| b.name()).collect();
                    format!("{} by {}", l.waves_per_simd, by.join("+"))
                });
                format!(
                    "{observed:.1} waves a SIMD, {need:.1} would hide {stall_per_trip:.0} stall over {issue_per_trip:.0} issue {a_unit}; limit {limit}"
                )
            }
            Evidence::Overhead { valu_per_trip, salu_per_trip, divergent } => format!(
                "valu {valu_per_trip:.0} + salu {salu_per_trip:.0} cycles {a_unit}, {} divergent branches",
                divergent.len()
            ),
        };
        let _ = writeln!(o, "{:>2} {:<17} {:>5.1}%  {what}{approx}", rank + 1, f.kind.name(), 100.0 * f.share);
        match &f.evidence {
            Evidence::Waits { loads, stalls, .. } => {
                for s in stalls {
                    let (pc, per, exposed) = (s.pc, s.per_trip, s.exposed_per_trip);
                    let _ =
                        writeln!(o, "   {pc:#x} stalled {per:.0}/{unit} before issue, exposed {exposed:.0}  {}", s.asm);
                }
                for l in loads {
                    let latency = l.latency.map_or("-".into(), |s| s.median.to_string());
                    let distance: Vec<String> = l.distance.iter().map(|(d, n)| format!("{d}:{n}")).collect();
                    let _ = writeln!(
                        o,
                        "   {:#x} exposed {:.0}/{unit}, cover med {} of latency med {latency}, {}/{} stalled, distance {}{}  {}",
                        l.pc,
                        l.exposed_per_trip,
                        l.cover.median,
                        l.stalled,
                        l.waits,
                        distance.join(","),
                        if l.rising { ", rising" } else { "" },
                        l.asm
                    );
                }
            }
            Evidence::Overhead { divergent, .. } => {
                for (pc, asm) in divergent {
                    let _ = writeln!(o, "   {pc:#x} {asm}");
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
            "note       gfx11 traces matrix ops as 1-cycle tokens: no matrix_bound or issue_bound finding, matrix issue holds its stall"
        );
    }
}
