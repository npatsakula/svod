//! Two traces of a kernel side by side ([`Analysis::compare`]): their untraced times and clocks, the
//! main loop's buckets, statements and rows a trip, and how the verdict moved. Rows pair by aligning
//! the two loops' mnemonic sequences (a longest common subsequence), so a register reallocation still
//! pairs; where both captures have lines, a row pairs only within the statement of the same text.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};
use svod_device::amd::occupancy::Limit;

use super::verdict::{observed, slowdown};
use super::{Analysis, Class, FindingKind, Pstate, Result, Row, Trips};

/// Traced-over-untraced ratios further apart than this share: the captures ran under different
/// perturbation, and their cycle shares compare less.
pub const SLOWDOWN_SPREAD: f64 = 0.10;
/// A bucket whose raw and exposed deltas are both under this many cycles a unit is left out of the
/// text.
const QUIET: f64 = 2.0;

/// One capture's context, and the count of units its loop's figures divide by.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Side {
    pub kernel: String,
    pub labels: BTreeMap<String, String>,
    pub kernel_ns: Option<u64>,
    pub traced_ns: Option<u64>,
    pub slowdown: Option<f64>,
    pub sclk_mhz: Option<u64>,
    pub pstate: Option<Pstate>,
    pub trips: Option<Trips>,
    /// Reduction elements a trip: a GEMM's K over `model.trips`.
    pub k_per_trip: Option<f64>,
    /// The units the loop's figures are over: the traced waves' trips, or their iterations.
    pub units: u64,
    /// The traced SIMDs' mean resident waves, and the kernel's limit.
    pub occupancy: Option<f64>,
    pub limit: Option<Limit>,
    /// The matrix pipe's busy share over the steady window, and the kernel's class.
    pub matrix_busy: Option<f64>,
    pub bound: Vec<FindingKind>,
}

/// What the loop's figures are over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    /// A trip of the reduction (`model.trips` labels both captures).
    Trip,
    /// A loop iteration: a capture lacks `model.trips`.
    Iteration,
}

impl Unit {
    pub fn name(self) -> &'static str {
        match self {
            Self::Trip => "a trip",
            Self::Iteration => "an iteration",
        }
    }
}

/// Why the comparison may mislead.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Caution {
    /// The captures ran at different power states: cycle shares move with the clock.
    Pstate { a: Option<Pstate>, b: Option<Pstate> },
    /// Their traced-over-untraced ratios differ past [`SLOWDOWN_SPREAD`].
    Slowdown { a: f64, b: f64 },
    /// A trip of each covers different reduction elements: compare the per-element deltas.
    TripsDiffer { a: f64, b: f64 },
    /// A capture has no main loop: no buckets or rows to compare.
    NoLoop,
}

/// A partition bucket's cycles a unit, raw and exposed, on each side.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BucketDelta {
    pub bucket: String,
    pub raw: [f64; 2],
    pub exposed: [f64; 2],
}

impl BucketDelta {
    pub fn delta_raw(&self) -> f64 {
        self.raw[1] - self.raw[0]
    }

    pub fn delta_exposed(&self) -> f64 {
        self.exposed[1] - self.exposed[0]
    }
}

/// A finding's share on each side; `None` where it did not fire.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FindingDelta {
    pub kind: FindingKind,
    pub share: [Option<f64>; 2],
}

/// A main-loop row a unit: `cycles` its latency and the idle before it, `exposed` those less what
/// other pipes hid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoopRow {
    pub pc: u64,
    pub asm: String,
    pub class: Class,
    pub hits: f64,
    pub cycles: f64,
    pub stall: f64,
    pub exposed: f64,
    /// Its statement's `kernel.tk` text, where the capture has lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub statement: Option<String>,
}

impl LoopRow {
    fn of(r: &Row, units: u64, listing: &[String]) -> Self {
        let per = |c: i64| c as f64 / units.max(1) as f64;
        let issue = r.latency - r.stall;
        let exposed = r.idle - r.hidden_idle.clamp(0, r.idle) + r.stall - r.hidden_stall.clamp(0, r.stall) + issue
            - r.hidden_issue.clamp(0, issue);
        Self {
            pc: r.pc,
            asm: r.asm.clone(),
            class: r.class,
            hits: r.hits as f64 / units.max(1) as f64,
            cycles: per(r.latency + r.idle),
            stall: per(r.stall),
            exposed: per(exposed),
            statement: r.stmt.and_then(|(line, _)| Some(listing.get(line as usize - 1)?.trim().to_string())),
        }
    }

    fn mnemonic(&self) -> &str {
        mnemonic(&self.asm)
    }
}

fn mnemonic(asm: &str) -> &str {
    asm.split_whitespace().next().unwrap_or_default()
}

/// Two loops' rows: those only B has, only A has, and the pairs, largest exposed delta first.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Rows {
    pub inserted: Vec<LoopRow>,
    pub removed: Vec<LoopRow>,
    pub matched: Vec<[LoopRow; 2]>,
}

/// A statement's loop cycles a unit on each side, by its `kernel.tk` text (two programs number
/// their lines apart); 0 where a side has no such statement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatementDelta {
    pub statement: String,
    pub raw: [f64; 2],
    pub exposed: [f64; 2],
}

/// Capture B against capture A. Its JSON keeps these key names: `a`, `b`, `unit`, `cautions`,
/// `buckets` (`bucket`, `raw`, `exposed`, each `[a, b]`), `findings` (`kind`, `share`), `rows`
/// (`inserted`, `removed`, `matched`), and where both captures have lines `statements`
/// (`statement`, `raw`, `exposed`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diff {
    pub a: Side,
    pub b: Side,
    pub unit: Unit,
    pub cautions: Vec<Caution>,
    pub buckets: Vec<BucketDelta>,
    pub findings: Vec<FindingDelta>,
    pub rows: Rows,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub statements: Vec<StatementDelta>,
}

/// `K / model.trips` of a GEMM capture's `shape` label (`MxNxK`).
fn k_per_trip(a: &Analysis) -> Option<f64> {
    let label = |k: &str| a.labels.get(k).map(String::as_str);
    let k: f64 = label("op").filter(|&op| op == "gemm").and(label("shape"))?.rsplit('x').next()?.parse().ok()?;
    let trips: f64 = label("model.trips")?.parse().ok()?;
    (trips > 0.0).then_some(k / trips)
}

fn side(a: &Analysis, unit: Unit) -> Side {
    let trips = a.main_loop.as_ref().map(|l| l.trips);
    let units = trips.map_or(0, |t| match unit {
        Unit::Trip => t.total,
        Unit::Iteration => t.total / u64::from(t.per_iteration.max(1)),
    });
    Side {
        kernel: a.labels.get("kernel").cloned().unwrap_or_else(|| a.kernel.clone()),
        labels: a
            .labels
            .iter()
            .filter(|(k, _)| !k.starts_with("model."))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        kernel_ns: a.kernel_ns,
        traced_ns: a.traced_ns,
        slowdown: slowdown(a),
        sclk_mhz: a.sclk_mhz,
        pstate: a.pstate,
        trips,
        k_per_trip: k_per_trip(a),
        units,
        occupancy: observed(a),
        limit: a.occupancy.limit.clone(),
        matrix_busy: a.verdict.ceiling.as_ref().map(|c| c.matrix),
        bound: a.verdict.bound.iter().map(|f| f.kind).collect(),
    }
}

/// The pairs of a longest common subsequence of `a` and `b`: `(Some(i), Some(j))` a match,
/// `(Some(i), None)` only in `a`, `(None, Some(j))` only in `b`, in order.
pub fn align<T: PartialEq>(a: &[T], b: &[T]) -> Vec<(Option<usize>, Option<usize>)> {
    let (n, m) = (a.len(), b.len());
    // `lcs[i][j]`: the common subsequence of `a[i..]` and `b[j..]`.
    let mut lcs = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if a[i] == b[j] { lcs[i + 1][j + 1] + 1 } else { lcs[i + 1][j].max(lcs[i][j + 1]) };
        }
    }
    let (mut i, mut j, mut pairs) = (0, 0, Vec::with_capacity(n.max(m)));
    while i < n || j < m {
        if i < n && j < m && a[i] == b[j] {
            pairs.push((Some(i), Some(j)));
            (i, j) = (i + 1, j + 1);
        } else if j == m || (i < n && lcs[i + 1][j] >= lcs[i][j + 1]) {
            pairs.push((Some(i), None));
            i += 1;
        } else {
            pairs.push((None, Some(j)));
            j += 1;
        }
    }
    pairs
}

/// Whether both captures have lines: rows then pair within a statement.
fn lined(a: &Analysis, b: &Analysis) -> bool {
    [a, b].iter().all(|x| x.main_loop.as_ref().is_some_and(|l| !l.statements.is_empty()))
}

fn rows(a: &Analysis, b: &Analysis, units: [u64; 2]) -> Rows {
    let ra: Vec<LoopRow> = a.loop_rows().map(|r| LoopRow::of(r, units[0], &a.listing)).collect();
    let rb: Vec<LoopRow> = b.loop_rows().map(|r| LoopRow::of(r, units[1], &b.listing)).collect();
    let by_statement = lined(a, b);
    let key = |r: &LoopRow| (by_statement.then(|| r.statement.clone()).flatten(), r.mnemonic().to_string());
    let (ma, mb): (Vec<_>, Vec<_>) = (ra.iter().map(key).collect(), rb.iter().map(key).collect());
    let mut rows = Rows::default();
    for pair in align(&ma, &mb) {
        match pair {
            (Some(i), Some(j)) => rows.matched.push([ra[i].clone(), rb[j].clone()]),
            (Some(i), None) => rows.removed.push(ra[i].clone()),
            (None, Some(j)) => rows.inserted.push(rb[j].clone()),
            (None, None) => unreachable!("an alignment pairs at least one row"),
        }
    }
    let delta = |p: &[LoopRow; 2]| (p[1].exposed - p[0].exposed).abs();
    rows.matched.sort_by(|x, y| delta(y).total_cmp(&delta(x)).then(x[1].pc.cmp(&y[1].pc)));
    rows
}

fn buckets(a: &Analysis, b: &Analysis, units: [u64; 2]) -> Vec<BucketDelta> {
    let (Some(la), Some(lb)) = (&a.main_loop, &b.main_loop) else { return vec![] };
    let mut by = BTreeMap::<String, BucketDelta>::new();
    let mut order = vec![];
    for (side, l) in [la, lb].into_iter().enumerate() {
        let per = |c: i64| c as f64 / units[side].max(1) as f64;
        let named = |p: &super::Partition| {
            let mut named = p.named();
            named.push(("total".into(), p.cycles));
            named
        };
        for ((name, raw), (_, exposed)) in named(&l.partition).into_iter().zip(named(&l.exposed)) {
            let at = by.entry(name.clone()).or_insert_with(|| {
                order.push(name.clone());
                BucketDelta { bucket: name, raw: [0.0; 2], exposed: [0.0; 2] }
            });
            (at.raw[side], at.exposed[side]) = (per(raw), per(exposed));
        }
    }
    let total = by.remove("total");
    let mut buckets: Vec<BucketDelta> = order.into_iter().filter_map(|n| by.remove(&n)).collect();
    let key = |d: &BucketDelta| (d.delta_exposed().abs(), d.delta_raw().abs());
    buckets.sort_by(|x, y| key(y).0.total_cmp(&key(x).0).then(key(y).1.total_cmp(&key(x).1)));
    buckets.extend(total);
    buckets
}

/// Each statement of either side by text, largest |Δ exposed| first; none unless both have lines.
fn statements(a: &Analysis, b: &Analysis, units: [u64; 2]) -> Vec<StatementDelta> {
    if !lined(a, b) {
        return vec![];
    }
    let mut by = BTreeMap::<String, StatementDelta>::new();
    for (side, (x, units)) in [(a, units[0]), (b, units[1])].into_iter().enumerate() {
        for (text, (raw, exposed)) in super::statements::by_text(x, units) {
            let at =
                by.entry(text.clone()).or_insert(StatementDelta { statement: text, raw: [0.0; 2], exposed: [0.0; 2] });
            (at.raw[side], at.exposed[side]) = (raw, exposed);
        }
    }
    let delta = |d: &StatementDelta| (d.exposed[1] - d.exposed[0]).abs();
    let mut out: Vec<StatementDelta> = by.into_values().collect();
    out.sort_by(|x, y| delta(y).total_cmp(&delta(x)).then(x.statement.cmp(&y.statement)));
    out
}

fn findings(a: &Analysis, b: &Analysis) -> Vec<FindingDelta> {
    let mut by = BTreeMap::<FindingKind, [Option<f64>; 2]>::new();
    for (side, x) in [a, b].into_iter().enumerate() {
        for f in x.verdict.bound.iter().chain(&x.verdict.findings) {
            by.entry(f.kind).or_default()[side] = Some(f.share);
        }
    }
    let mut findings: Vec<FindingDelta> = by.into_iter().map(|(kind, share)| FindingDelta { kind, share }).collect();
    let delta = |f: &FindingDelta| (f.share[1].unwrap_or(0.0) - f.share[0].unwrap_or(0.0)).abs();
    findings.sort_by(|x, y| delta(y).total_cmp(&delta(x)).then(x.kind.cmp(&y.kind)));
    findings
}

impl Analysis {
    /// `b` against `self` (A): Δ is B − A, every loop figure a unit of each side's own.
    pub fn compare(&self, b: &Self) -> Diff {
        let labelled = |x: &Self| x.main_loop.as_ref().is_some_and(|l| l.trips.labelled);
        let unit = if labelled(self) && labelled(b) { Unit::Trip } else { Unit::Iteration };
        let (sa, sb) = (side(self, unit), side(b, unit));
        let mut cautions = vec![];
        if sa.pstate != sb.pstate {
            cautions.push(Caution::Pstate { a: sa.pstate, b: sb.pstate });
        }
        if let Some((ra, rb)) = sa.slowdown.zip(sb.slowdown).filter(|(ra, rb)| (rb / ra - 1.0).abs() > SLOWDOWN_SPREAD)
        {
            cautions.push(Caution::Slowdown { a: ra, b: rb });
        }
        if let (Unit::Trip, Some(ka), Some(kb)) = (unit, sa.k_per_trip, sb.k_per_trip)
            && ka != kb
        {
            cautions.push(Caution::TripsDiffer { a: ka, b: kb });
        }
        if self.main_loop.is_none() || b.main_loop.is_none() {
            cautions.push(Caution::NoLoop);
        }
        let units = [sa.units, sb.units];
        Diff {
            unit,
            cautions,
            buckets: buckets(self, b, units),
            findings: findings(self, b),
            rows: rows(self, b, units),
            statements: statements(self, b, units),
            a: sa,
            b: sb,
        }
    }
}

/// Inserted or removed rows by mnemonic: `(mnemonic, rows, hits, cycles, exposed)` a unit, most
/// cycles first.
fn grouped(rows: &[LoopRow]) -> Vec<(&str, usize, f64, f64, f64)> {
    let mut by = BTreeMap::<&str, (usize, f64, f64, f64)>::new();
    for r in rows {
        let at = by.entry(r.mnemonic()).or_default();
        *at = (at.0 + 1, at.1 + r.hits, at.2 + r.cycles, at.3 + r.exposed);
    }
    let mut groups: Vec<_> = by.into_iter().map(|(m, (n, h, c, e))| (m, n, h, c, e)).collect();
    groups.sort_by(|x, y| y.3.total_cmp(&x.3).then(x.0.cmp(y.0)));
    groups
}

fn statement(r: &LoopRow) -> String {
    r.statement.as_ref().map_or(String::new(), |s| format!("  @ {s}"))
}

fn signed(x: f64) -> String {
    format!("{x:+.1}")
}

impl Diff {
    /// The analyses of two [`super::Capture::save`]d directories, B against A.
    pub fn load(a: &Path, b: &Path) -> Result<Self> {
        Ok(Analysis::load(a)?.compare(&Analysis::load(b)?))
    }

    /// The comparison as one line of JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("a diff serializes")
    }

    /// The fixed-format comparison with `top` rows a table.
    pub fn text(&self, top: usize) -> String {
        let mut o = String::new();
        let (a, b) = (&self.a, &self.b);
        let unit = self.unit.name();
        let _ = writeln!(o, "== sqtt compare: B against A; loop cycles are a traced wave's, {unit} of each; Δ = B - A");
        for (name, s) in [("A", a), ("B", b)] {
            let shown: String =
                ["shape", "cfg"].iter().filter_map(|k| Some(format!(" {k}={}", s.labels.get(*k)?))).collect();
            let _ = writeln!(o, "{name}          {}{shown}", s.kernel);
        }
        let ns = |ns: Option<u64>| ns.map_or("-".into(), |ns| ns.to_string());
        let change = match a.kernel_ns.zip(b.kernel_ns).filter(|(a, _)| *a > 0) {
            Some((x, y)) => format!(" ({:+.1}%)", 100.0 * (y as f64 / x as f64 - 1.0)),
            None => String::new(),
        };
        let ratio = |s: &Side| s.slowdown.map_or("-".into(), |r| format!("{r:.2}x"));
        let mhz = |s: &Side| s.sclk_mhz.map_or("-".into(), |m| format!("~{m}"));
        let pstate = |s: &Side| s.pstate.map_or("unrecorded", Pstate::name);
        let _ = writeln!(
            o,
            "time       {} -> {} ns untraced{change}; traced/untraced {} -> {}; sclk {} -> {} MHz; pstate {} -> {}",
            ns(a.kernel_ns),
            ns(b.kernel_ns),
            ratio(a),
            ratio(b),
            mhz(a),
            mhz(b),
            pstate(a),
            pstate(b)
        );
        for c in &self.cautions {
            let _ = match c {
                Caution::Pstate { .. } => {
                    writeln!(o, "!! PSTATE differs: cycle shares move with the clock; compare untraced ns")
                }
                Caution::Slowdown { a, b } => writeln!(
                    o,
                    "!! PERTURBATION differs: traced/untraced {a:.2}x vs {b:.2}x (over {:.0}% apart); shares compare less",
                    100.0 * SLOWDOWN_SPREAD
                ),
                Caution::TripsDiffer { a, b } => {
                    writeln!(o, "!! TRIPS differ: {a:.0} vs {b:.0} K elements a trip; Δ/k is the delta a K element")
                }
                Caution::NoLoop => writeln!(o, "!! NO LOOP on a side: no buckets or rows"),
            };
        }
        if self.unit == Unit::Iteration {
            let _ = writeln!(o, "note       no model.trips on a side: figures are per loop iteration");
        }
        let occupancy = |s: &Side| s.occupancy.map_or("-".into(), |w| format!("{w:.1}"));
        let limit = |s: &Side| {
            s.limit.as_ref().map_or("-".into(), |l| {
                let by: Vec<&str> = l.by.iter().map(|b| b.name()).collect();
                format!("{} by {}", l.waves_per_simd, by.join("+"))
            })
        };
        let _ = writeln!(
            o,
            "occupancy  {} -> {} waves a traced SIMD; limit {} -> {}",
            occupancy(a),
            occupancy(b),
            limit(a),
            limit(b)
        );
        let busy = |s: &Side| s.matrix_busy.map_or("-".into(), |m| format!("{:.1}%", 100.0 * m));
        let bound = |s: &Side| {
            let kinds: Vec<&str> = s.bound.iter().map(|k| k.name()).collect();
            if kinds.is_empty() { "-".into() } else { kinds.join("+") }
        };
        let _ =
            writeln!(o, "ceiling    matrix pipe busy {} -> {}; bound {} -> {}", busy(a), busy(b), bound(a), bound(b));
        let _ = writeln!(o, "== findings: share of the loop's cycles exposed (bound: of the steady window)");
        let pct = |s: Option<f64>| s.map_or("-".into(), |s| format!("{:.1}%", 100.0 * s));
        for f in &self.findings {
            let what = match f.share {
                [None, Some(_)] => "appears",
                [Some(_), None] => "gone",
                _ => "changed",
            };
            let _ = writeln!(o, "{what:<10} {:<17} {:>6} -> {:>6}", f.kind.name(), pct(f.share[0]), pct(f.share[1]));
        }
        self.buckets_text(&mut o);
        self.statements_text(&mut o, top);
        self.rows_text(&mut o, top);
        o
    }

    fn buckets_text(&self, o: &mut String) {
        if self.buckets.is_empty() {
            return;
        }
        let per_k = self.cautions.iter().find_map(|c| match *c {
            Caution::TripsDiffer { a, b } => Some([a, b]),
            _ => None,
        });
        let quiet =
            |d: &BucketDelta| d.bucket != "total" && d.delta_raw().abs() < QUIET && d.delta_exposed().abs() < QUIET;
        let omitted = self.buckets.iter().filter(|d| quiet(d)).count();
        let _ = writeln!(
            o,
            "== loop cycles {} by bucket, largest Δ exposed first ({omitted} within ±{QUIET:.0} omitted)",
            self.unit.name()
        );
        let k = if per_k.is_some() { format!(" {:>8}", "Δ/k exp") } else { String::new() };
        let _ = writeln!(
            o,
            "{:<18} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}{k}",
            "bucket", "A raw", "B raw", "Δ raw", "A exp", "B exp", "Δ exp"
        );
        for d in self.buckets.iter().filter(|d| !quiet(d)) {
            let k = per_k.map_or(String::new(), |[ka, kb]| format!(" {:>8.2}", d.exposed[1] / kb - d.exposed[0] / ka));
            let _ = writeln!(
                o,
                "{:<18} {:>8.1} {:>8.1} {:>8} {:>8.1} {:>8.1} {:>8}{k}",
                d.bucket,
                d.raw[0],
                d.raw[1],
                signed(d.delta_raw()),
                d.exposed[0],
                d.exposed[1],
                signed(d.delta_exposed())
            );
        }
    }

    fn statements_text(&self, o: &mut String, top: usize) {
        if self.statements.is_empty() {
            return;
        }
        let _ = writeln!(o, "== statements {} by kernel.tk text: top {top} by |Δ exposed|", self.unit.name());
        let _ = writeln!(
            o,
            "{:>8} {:>8} {:>8} {:>8} {:>8} {:>8}  statement",
            "A raw", "B raw", "Δ raw", "A exp", "B exp", "Δ exp"
        );
        for d in self.statements.iter().take(top) {
            let _ = writeln!(
                o,
                "{:>8.1} {:>8.1} {:>8} {:>8.1} {:>8.1} {:>8}  {}",
                d.raw[0],
                d.raw[1],
                signed(d.raw[1] - d.raw[0]),
                d.exposed[0],
                d.exposed[1],
                signed(d.exposed[1] - d.exposed[0]),
                d.statement
            );
        }
    }

    fn rows_text(&self, o: &mut String, top: usize) {
        let r = &self.rows;
        if self.cautions.contains(&Caution::NoLoop) {
            return;
        }
        let unit = self.unit.name();
        let by = if self.statements.is_empty() { "mnemonic" } else { "statement and mnemonic" };
        let _ = writeln!(
            o,
            "== loop rows aligned by {by}: {} matched, {} inserted in B, {} removed from A; ops, cycles, exposed {unit}",
            r.matched.len(),
            r.inserted.len(),
            r.removed.len()
        );
        for (what, rows) in [("inserted", &r.inserted), ("removed", &r.removed)] {
            if rows.is_empty() {
                continue;
            }
            let _ =
                writeln!(o, "{what:<8} {:<22} {:>5} {:>9} {:>9} {:>9}", "mnemonic", "rows", "ops", "cycles", "exposed");
            let groups = grouped(rows);
            let sum = |f: fn(&LoopRow) -> f64| rows.iter().map(f).sum::<f64>();
            let total = ("total", rows.len(), sum(|r| r.hits), sum(|r| r.cycles), sum(|r| r.exposed));
            for (m, n, hits, cycles, exposed) in groups.into_iter().take(top).chain([total]) {
                let _ = writeln!(o, "{:<8} {m:<22} {n:>5} {hits:>9.1} {cycles:>9.1} {exposed:>9.1}", "");
            }
            let mut rows: Vec<&LoopRow> = rows.iter().collect();
            rows.sort_by(|x, y| y.exposed.total_cmp(&x.exposed).then(x.pc.cmp(&y.pc)));
            for x in rows.into_iter().take(top.min(3)) {
                let _ = writeln!(
                    o,
                    "{:>8} {:>#7x} {:>5.1} ops {:>6.1} cycles {:>6.1} exposed  {}{}",
                    "",
                    x.pc,
                    x.hits,
                    x.cycles,
                    x.exposed,
                    x.asm,
                    statement(x)
                );
            }
        }
        let _ = writeln!(o, "== matched rows: top {top} by |Δ exposed| {unit}");
        let _ = writeln!(
            o,
            "{:>7} {:>7} {:>8} {:>8} {:>8} {:>8}  asm (B)",
            "pc A", "pc B", "A cyc", "B cyc", "A exp", "B exp"
        );
        for [x, y] in r.matched.iter().take(top) {
            let _ = writeln!(
                o,
                "{:>#7x} {:>#7x} {:>8.1} {:>8.1} {:>8.1} {:>8.1}  {}{}",
                x.pc,
                y.pc,
                x.cycles,
                y.cycles,
                x.exposed,
                y.exposed,
                y.asm,
                statement(y)
            );
        }
    }
}

impl std::fmt::Display for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text(8))
    }
}
