//! An [`Analysis`]'s fixed-format report: the verdict, the main loop's partition and statements, the
//! occupancy, the waits and the hottest rows.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

use svod_device::sqtt::asm::classify;
use svod_device::sqtt::report::Spread;
use svod_device::sqtt::simd::{ROBUST, STEADY, Simd};

use super::partition::stores;
use super::verdict::{BARRIER, EVIDENCE, PERTURBED, slowdown};
use super::{Analysis, Bucket, Evidence, Finding, Knob, MainLoop, Partition, PerTrip, Pstate, Row, Stmt, Unit};

impl Analysis {
    /// The fixed-format report with `top` rows a table.
    pub fn text(&self, top: usize) -> String {
        let mut o = String::new();
        let labels: String = self.labels.iter().map(|(k, v)| format!(" {k}={v}")).collect();
        let _ = writeln!(o, "== sqtt {} {}{labels}", self.kernel, self.arch);
        for se in self.ses.iter().filter(|s| s.truncated) {
            let _ = writeln!(o, "!! TRUNCATED se{}: its buffer filled and the tail is lost; totals undercount", se.se);
        }
        for (se, version) in self.ses.iter().filter_map(|s| Some((s.se, s.refused?))) {
            let _ = writeln!(o, "!! UNSUPPORTED se{se}: a header of version {version}; the stream from it is left out");
        }
        if self.stitched < self.records {
            let _ =
                writeln!(o, "!! UNSTITCHED {} of {} records have no PC", self.records - self.stitched, self.records);
        }
        verdict(self, &mut o);
        let ns = |ns: Option<u64>| ns.map_or("-".into(), |ns| ns.to_string());
        let span = self.ses.iter().filter(|s| s.itrace).map(|s| s.span).max().unwrap_or(0);
        let sclk = self.sclk_mhz.map_or("-".into(), |mhz| format!("~{mhz}"));
        let mclk = self.mclk_mhz.map_or(String::new(), |mhz| format!(", mclk {mhz} MHz"));
        let (untraced, traced) = (ns(self.kernel_ns), ns(self.traced_ns));
        let ratio = slowdown(self).map_or(String::new(), |r| format!(" ({r:.2}x)"));
        let pstate = self.pstate.map_or("unrecorded", Into::into);
        let _ = writeln!(
            o,
            "kernel     {untraced} ns untraced, {traced} ns traced{ratio}, wave span {span} cycles, sclk {sclk} MHz{mclk}, pstate {pstate}"
        );
        let traced: Vec<String> = self
            .traced
            .iter()
            .map(|t| format!("se{} sa{} wgp{} simd{} ({} waves)", t.se, t.sa, t.wgp, t.simd, t.waves))
            .collect();
        let _ = writeln!(o, "traced     {} of {} SEs", traced.join(", "), self.ses.len());
        let lifetime = self.lifetime.map_or("-".into(), |l| format!("min {} med {} max {}", l.min, l.median, l.max));
        let starts: usize = self.ses.iter().filter(|s| s.itrace).map(|s| s.waves).sum();
        let peak = self.ses.iter().map(|s| s.occupancy_peak).max().unwrap_or(0);
        let _ = writeln!(
            o,
            "waves      {} traced ({} complete) of {starts} on the traced SEs, occupancy peak {peak}; lifetime {lifetime} cycles",
            self.waves, self.complete
        );
        let _ = writeln!(o, "records    {} stitched of {}", self.stitched, self.records);
        let total: i64 = self.classes.iter().map(|c| c.latency).sum();
        let _ = writeln!(o, "== time by class (traced SIMD, cycles summed over waves)");
        let _ =
            writeln!(o, "{:<11} {:>8} {:>11} {:>6} {:>11} {:>11}", "class", "hits", "cycles", "share", "stall", "exec");
        let share = |c: i64| if total > 0 { 100.0 * c as f64 / total as f64 } else { 0.0 };
        for c in &self.classes {
            let (l, s) = (c.latency, c.stall);
            let _ = writeln!(
                o,
                "{:<11} {:>8} {l:>11} {:>5.1}% {s:>11} {:>11}",
                <&str>::from(c.class),
                c.hits,
                share(l),
                l - s
            );
        }
        let (hits, stall): (u64, i64) = self.classes.iter().fold((0, 0), |(h, s), c| (h + c.hits, s + c.stall));
        let _ = writeln!(o, "{:<11} {hits:>8} {total:>11} {:>5.1}% {stall:>11} {:>11}", "total", 100.0, total - stall);
        self.occupancy_text(&mut o);
        let table = |o: &mut String, rows: Vec<&Row>| {
            let _ = writeln!(
                o,
                "{:>8} {:>7} {:>9} {:>9} {:>9} {:>9}  asm",
                "pc", "hits", "cycles", "lat/hit", "stall/hit", "expo/hit"
            );
            for r in rows {
                let per = |c: i64| c as f64 / r.hits.max(1) as f64;
                let (l, s, e) = (per(r.latency), per(r.stall), per(r.stall - r.hidden_stall));
                let _ =
                    writeln!(o, "{:>#8x} {:>7} {:>9} {l:>9.1} {s:>9.1} {e:>9.1}  {}", r.pc, r.hits, r.latency, r.asm);
            }
        };
        match &self.main_loop {
            None => {
                let _ = writeln!(o, "== main loop: none (no executed backward branch)");
            }
            Some(l) => {
                let kind = if l.matrix { "matrix" } else { "hottest, no matrix op" };
                let _ = writeln!(o, "== main loop {:#x}..={:#x} ({kind})", l.head, l.back_edge);
                let spread = |s: Option<Spread>, p: bool| match s {
                    None => "-".into(),
                    Some(s) if p => format!("min {} med {} p90 {} max {} (n {})", s.min, s.median, s.p90, s.max, s.n),
                    Some(s) => format!("min {} med {} max {} (waves {})", s.min, s.median, s.max, s.n),
                };
                let _ = writeln!(o, "iterations/wave  {}", spread(l.iterations, false));
                let _ = writeln!(o, "cycles/iteration {}", spread(l.cycles, true));
                partition(&mut o, l);
                statements(self, &mut o, top);
                let mut rows: Vec<&Row> = self.loop_rows().collect();
                rows.sort_by_key(|r| (std::cmp::Reverse(r.stall), r.pc));
                let _ = writeln!(o, "== main loop: top {top} by stall");
                table(&mut o, rows.into_iter().take(top).collect());
            }
        }
        self.waits_text(&mut o, top);
        let mut rows: Vec<&Row> = self.pcs.iter().collect();
        rows.sort_by_key(|r| (std::cmp::Reverse(r.latency), r.pc));
        let _ = writeln!(o, "== hottest {top} by cycles");
        table(&mut o, rows.into_iter().take(top).collect());
        o
    }

    /// The kernel's limit on resident waves, and per SE the waves resident over its steady window
    /// (`mean/peak`, the SE's and its SIMDs'), with the traced SIMD's and its pipes' busy shares.
    fn occupancy_text(&self, o: &mut String) {
        let occ = &self.occupancy;
        let (num, den) = STEADY;
        let _ = writeln!(
            o,
            "== occupancy: resident waves mean/peak over the steady window (the SE at {}% of its p{} or more)",
            100 * num / den,
            100 * ROBUST.0 / ROBUST.1
        );
        match &occ.limit {
            Some(l) => {
                let by: Vec<&str> = l.by.iter().map(|b| b.name()).collect();
                let (waves, groups, by) = (l.waves_per_simd, l.workgroups_per_cu, by.join("+"));
                let _ = writeln!(o, "limit      {waves} waves/SIMD, {groups} workgroups/CU, by {by}");
            }
            None => {
                let _ = writeln!(o, "limit      - (no kernel resources and workgroup in the capture)");
            }
        }
        for se in &occ.ses {
            let r = &se.residency;
            let span = self.ses.iter().find(|s| s.se == se.se).map_or(0, |s| s.span).max(1);
            let window = 100.0 * (r.window.1 - r.window.0) as f64 / span as f64;
            let peak = r.simds.iter().map(|s| s.peak).max().unwrap_or(0);
            let mean = r.simds.iter().map(Simd::mean).sum::<f64>() / r.simds.len().max(1) as f64;
            let traced = se.traced.map_or(String::new(), |t| format!("; traced {:.1}/{}", t.mean(), t.peak));
            let _ = writeln!(
                o,
                "se{:<8} {:.1}/{} (p{} {}) over {window:.0}% of the span; {} SIMDs {mean:.1}/{peak}{traced}",
                se.se,
                r.mean(),
                r.peak,
                100 * ROBUST.0 / ROBUST.1,
                r.robust_peak,
                r.simds.len()
            );
            if let Some(b) = se.busy {
                let pct = |c: i64| 100.0 * b.share(c);
                let _ = writeln!(
                    o,
                    "  busy     matrix {:.1}% valu {:.1}% vmem {:.1}% scalar {:.1}% any {:.1}%",
                    pct(b.matrix),
                    pct(b.valu),
                    pct(b.vmem),
                    pct(b.scalar),
                    pct(b.any)
                );
            }
        }
        if occ.untimed_matrix {
            let _ =
                writeln!(o, "note       matrix ops trace as 1-cycle VALU tokens here: matrix and valu busy undercount");
        }
    }

    /// The bindings whose waits stalled longest: `stalled` of `waits` stalled (latency over those),
    /// `wait` is the wait PC it bound most often and `distance` lists `back edges:instances`.
    fn waits_text(&self, o: &mut String, top: usize) {
        let w = &self.waits;
        let _ = writeln!(
            o,
            "== waits: {} on {} waves ({} skipped), {} unbound, {} approximate; top {top} bindings by stall",
            w.instances, w.waves, w.skipped, w.unbound, w.approx
        );
        let _ = writeln!(
            o,
            "{:>8} {:>7} {:>7} {:>9} {:>7} {:>7} {:>8} {:<12}  asm",
            "pc", "waits", "stalled", "stall", "lat med", "lat p90", "wait", "distance"
        );
        let asm: BTreeMap<u64, &str> = self.pcs.iter().map(|r| (r.pc, r.asm.as_str())).collect();
        for b in w.bindings.iter().take(top) {
            let wait = b.waits.iter().max_by_key(|&&(pc, n, _)| (n, std::cmp::Reverse(pc))).map_or(0, |w| w.0);
            let distance: Vec<String> = b.distance.iter().map(|(d, n)| format!("{d}:{n}")).collect();
            let (med, p90) = b.latency.map_or(("-".into(), "-".into()), |l| (l.median.to_string(), l.p90.to_string()));
            let asm = asm.get(&b.pc).copied().unwrap_or_default();
            let _ = writeln!(
                o,
                "{:>#8x} {:>7} {:>7} {:>9} {med:>7} {p90:>7} {wait:>#8x} {:<12}  {asm}",
                b.pc,
                b.count,
                b.stalled,
                b.stall,
                distance.join(",")
            );
        }
    }

    /// One paragraph: where the traced SIMD's time went and the main loop's pace.
    pub fn summary(&self) -> String {
        let total: i64 = self.classes.iter().map(|c| c.latency).sum::<i64>().max(1);
        let mut classes = self.classes.clone();
        classes.sort_by_key(|c| std::cmp::Reverse(c.latency));
        let top: Vec<String> = classes
            .iter()
            .take(4)
            .map(|c| format!("{} {:.0}%", <&str>::from(c.class), 100.0 * c.latency as f64 / total as f64))
            .collect();
        let sclk = self.sclk_mhz.map_or(String::new(), |mhz| format!(", sclk ~{mhz} MHz"));
        let pace = self
            .main_loop
            .as_ref()
            .and_then(|l| Some((l.iterations?, l.cycles?)))
            .map_or("no main loop".into(), |(i, c)| {
                format!("main loop {} iterations/wave at {} cycles (median)", i.median, c.median)
            });
        let warn = if self.ses.iter().any(|s| s.truncated) { " TRUNCATED." } else { "" };
        format!(
            "{} {}: {} traced waves{sclk}; time {}; {pace}.{warn}",
            self.kernel,
            self.arch,
            self.waves,
            top.join(", ")
        )
    }

    /// `"  @line text"` of `stmt`, else nothing.
    fn at(&self, stmt: Option<Stmt>) -> String {
        stmt.map_or(String::new(), |s| format!("  @{} {}", s.line, self.line(s)))
    }
}

/// The main loop's partition a trip: every bucket, raw and exposed, its shares, and each traced SE's.
fn partition(o: &mut String, l: &MainLoop) {
    let t = l.trips;
    let unit = t.unit().name();
    let _ = writeln!(
        o,
        "trips      {} an iteration ({}), {} in the loop over {} waves",
        t.per_iteration,
        if t.counted { "its statements' unroll copies" } else { "no statements: per-trip figures are per iteration" },
        t.total,
        l.partition.waves
    );
    let ses: String = l.ses.iter().map(|s| format!(" {:>9}", format!("se{}", s.se))).collect();
    let _ = writeln!(
        o,
        "== main loop cycles {unit}: a traced wave's, each cycle once (issue, stall and idle by class); exposed: less what other pipes hid"
    );
    let _ = writeln!(o, "{:<20} {:>9} {:>7} {:>9} {:>7}{ses}", "bucket", "all", "share", "exposed", "share");
    let per = |cycles: i64, trips: u64| cycles as f64 / trips.max(1) as f64;
    let p = &l.partition;
    let total = p.cycles()[0].max(1) as f64;
    let rows = p.buckets.keys().map(|&b| (b.to_string(), Some(b))).chain([("total".into(), None)]);
    for (name, bucket) in rows {
        // A bucket's cycles, else every bucket's.
        let of = |p: &Partition| bucket.map_or(p.cycles(), |b| p.buckets.get(&b).copied().unwrap_or_default());
        let ses: String = l.ses.iter().map(|s| format!(" {:>9.1}", per(of(&s.partition)[0], s.trips))).collect();
        let [all, exposed] = of(p);
        let share = |c: i64| 100.0 * c as f64 / total;
        let _ = writeln!(
            o,
            "{name:<20} {:>9.1} {:>6.1}% {:>9.1} {:>6.1}%{ses}",
            per(all, t.total),
            share(all),
            per(exposed, t.total),
            share(exposed)
        );
    }
}

impl std::fmt::Display for Analysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text(10))
    }
}

impl std::fmt::Display for Bucket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Issue(c) => write!(f, "issue {}", <&str>::from(c)),
            Self::Stall(c) => write!(f, "stall {}", <&str>::from(c)),
            Self::Wait(on) => write!(f, "wait {}", on.name()),
            Self::Barrier => f.write_str("barrier"),
            Self::Idle(c) => write!(f, "idle {}", <&str>::from(c)),
        }
    }
}

impl std::fmt::Display for Knob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LessMatrixWork => {
                f.write_str("near the matrix pipe's throughput: only less matrix work (a cheaper type, fewer ops) helps much")
            }
            Self::CutIssue => {
                f.write_str("the SIMD issues nearly every cycle, mostly not matrix work: cut the loop's other instructions")
            }
            Self::FreeResources => {
                f.write_str("the kernel's resources cap the waves: free the limiter (fewer VGPRs, less LDS a workgroup)")
            }
            Self::MoreWorkgroups => f.write_str("fewer waves resident than the kernel allows: more workgroups in flight"),
            Self::MoreWaves => f.write_str(
                "more resident waves would fill the idle cycles; the capture records no resources to tell what limits them",
            ),
            Self::MoveUse { wait, load, consumer } => write!(
                f,
                "this use forces the wait right behind the load: move or drop it ({wait:#x} waits for {load:#x}, then `{consumer}`)"
            ),
            Self::Latency => f.write_str("latency: more work between a load and its wait (prefetch distance, ring slots)"),
            Self::LdsFlush => f.write_str(
                "the ds_store flush before a barrier: fewer, wider stores, or more work between the stores and the barrier",
            ),
            Self::LdsUse => f.write_str(
                "more work between a ds load and its use (load the next fragment earlier), fewer and wider ds loads",
            ),
            Self::LdsQueue => {
                f.write_str("the LDS queue is full before ds ops issue: fewer and wider ds ops, a conflict-free layout")
            }
            Self::Barrier => {
                write!(f, "drop or merge a barrier only if its exposed share passes {:.0}%", 100.0 * BARRIER)
            }
            Self::Hoist => {
                f.write_str("hoist address and guard arithmetic out of the loop; make guards uniform (no exec branches)")
            }
        }
    }
}

/// The main loop's statements, `top` most exposed and the unattributed.
fn statements(a: &Analysis, o: &mut String, top: usize) {
    let Some(l) = a.main_loop.as_ref().filter(|l| !l.statements.is_empty()) else { return };
    let unit = l.trips.unit().name();
    let (pcs, unattributed) =
        l.statements.iter().fold((0, 0), |(n, u), s| (n + s.pcs, u + if s.line.is_none() { s.pcs } else { 0 }));
    let _ = writeln!(
        o,
        "== statements: main loop cycles {unit} by kernel.tk line (a wait on its binding load's); {} of {pcs} loop instructions attributed; top {top} by exposed",
        pcs - unattributed
    );
    let _ = writeln!(
        o,
        "{:>5} {:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}  statement",
        "line", "pcs", "issue", "stall", "wait", "barrier", "idle", "total", "exposed"
    );
    let shown = l.statements.iter().filter(|s| s.line.is_some()).take(top);
    for s in shown.chain(l.statements.iter().filter(|s| s.line.is_none())) {
        let line = s.line.map_or("-".into(), |l| l.to_string());
        let parts: String = s.parts.0.iter().map(|c| format!(" {:>8.1}", c.raw)).collect();
        let _ = writeln!(o, "{line:>5} {:>4}{parts} {:>8.1} {:>8.1}  {}", s.pcs, s.raw(), s.exposed(), s.text);
    }
}

fn pct(share: f64) -> f64 {
    100.0 * share
}

/// The verdict section: the kernel's ceiling and class, the loop's findings with their evidence and
/// knob, then what makes them approximate.
fn verdict(a: &Analysis, o: &mut String) {
    let trips = a.main_loop.as_ref().map(|l| l.trips);
    let of = match trips {
        Some(t) if t.counted => format!("a trip ({} an iteration, its statements' unroll copies)", t.per_iteration),
        Some(_) => "an iteration (no statements)".into(),
        None => "- (no main loop)".into(),
    };
    let unit = trips.map_or(Unit::Trip, |t| t.unit());
    let (a_unit, unit): (_, &str) = (unit.name(), unit.into());
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
        let _ = writeln!(
            o,
            "demand     {} global bytes {a_unit} a wave x {:.0} resident waves / {:.0} cycles {a_unit} at {} MHz: {:.0} GB/s requested from L2 and DRAM",
            d.bytes_per_trip, d.waves, d.cycles_per_trip, d.sclk_mhz, d.gbs
        );
    }
    let flags = |f: &Finding| {
        let approx: Vec<&str> = f.approx.iter().map(|&a| a.into()).collect();
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
        let _ = writeln!(o, "bound      {:<17} {:>5.1}%  {what}{}", <&str>::from(f.kind), pct(f.share), flags(f));
        let _ = writeln!(o, "   knob: {}", f.knob);
    }
    if v.findings.is_empty() {
        let _ = writeln!(o, "none       no loop finding passes its threshold");
    }
    let untimed = a.occupancy.untimed_matrix;
    let c = |c: &PerTrip| match untimed {
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
        let _ = writeln!(
            o,
            "{:>2} {:<17} {share:>5.1}% (raw {raw:.1}%)  {what}{}",
            rank + 1,
            <&str>::from(f.kind),
            flags(f)
        );
        match &f.evidence {
            Evidence::Waits { loads, stalls, .. } => {
                for s in stalls {
                    let (pc, per, exposed) = (s.pc, s.per_trip, s.exposed_per_trip);
                    let _ = writeln!(
                        o,
                        "   {pc:#x} stalled {per:.0}/{unit} before issue, exposed {exposed:.0}  {}{}",
                        s.asm,
                        a.at(s.stmt)
                    );
                }
                for l in loads {
                    let what = if stores(classify(&l.asm).kind) { "store" } else { "load" };
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
                        a.at(l.stmt)
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
                        .map(|(pc, n)| {
                            format!("{pc:#x} {} x{n}{}", asm.get(pc).copied().unwrap_or("?"), a.at(a.stmt(*pc)))
                        })
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
                        a.at(s.stmt)
                    );
                }
            }
            Evidence::Overhead { divergent, .. } => {
                for (pc, asm) in divergent {
                    let _ = writeln!(o, "   {pc:#x} {asm}{}", a.at(a.stmt(*pc)));
                }
            }
            _ => {}
        }
        let _ = writeln!(o, "   knob: {}", f.knob);
    }
    if let Some(r) = slowdown(a).filter(|&r| r > PERTURBED) {
        let why = match a.pstate {
            Some(Pstate::Unpinned) => "unpinned: not the clock".to_string(),
            Some(p) => format!("pinned {}: its lower clock moves cycle shares", <&str>::from(p)),
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
