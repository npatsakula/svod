//! The main loop's cycles by tile-program statement: each instruction's share of the partition
//! ([`super::Partition`]) summed over the statement its line names, a wait's on the statement of the
//! load that bound it, so a fill carries the waits its latency causes.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::num::NonZeroU32;

use serde::{Deserialize, Serialize};

use super::verdict::Charge;
use super::{Analysis, Lines, PerTrip, Stmt};

/// A `kernel.tk` line's cycles in the main loop, a traced wave's a trip. The rollup is per line: an
/// unrolled statement's copies sum into one, each instruction's copy staying on its
/// [`Row::stmt`](super::Row::stmt).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Statement {
    /// Its line in `kernel.tk`; `None` gathers the loop's instructions no statement claims.
    pub line: Option<NonZeroU32>,
    /// Its `kernel.tk` line, trimmed.
    pub text: String,
    /// The loop's instructions on it.
    pub pcs: usize,
    pub issue: PerTrip,
    pub stall: PerTrip,
    pub wait: PerTrip,
    pub barrier: PerTrip,
    pub idle: PerTrip,
}

/// `listing`'s `line`, trimmed; `?` past its end.
pub(super) fn listed(listing: &[String], line: NonZeroU32) -> &str {
    listing.get(line.get() as usize - 1).map_or("?", |t| t.trim())
}

impl Statement {
    fn new(line: Option<NonZeroU32>, listing: &[String]) -> Self {
        let text = line.map_or("unattributed", |l| listed(listing, l)).into();
        let none = PerTrip::default();
        Self { line, text, pcs: 0, issue: none, stall: none, wait: none, barrier: none, idle: none }
    }

    fn parts(&self) -> [PerTrip; 5] {
        [self.issue, self.stall, self.wait, self.barrier, self.idle]
    }

    pub fn raw(&self) -> f64 {
        self.parts().iter().map(|c| c.raw).sum()
    }

    pub fn exposed(&self) -> f64 {
        self.parts().iter().map(|c| c.exposed).sum()
    }
}

/// `charges` (by ELF address) rolled up by the statement `lines` gives each, over `trips`; `body`
/// is the loop's instructions, counted on their statements. Most exposed first, the unattributed
/// last.
pub(super) fn rollup(
    charges: &BTreeMap<u64, Charge>,
    body: impl IntoIterator<Item = u64>,
    lines: &Lines,
    listing: &[String],
    trips: u64,
) -> Vec<Statement> {
    let line = |pc: &u64| lines.get(pc).map(|s| s.line);
    let mut by = BTreeMap::<Option<NonZeroU32>, Statement>::new();
    for pc in body {
        let l = line(&pc);
        by.entry(l).or_insert_with(|| Statement::new(l, listing)).pcs += 1;
    }
    let trips = trips.max(1) as f64;
    let add = |to: &mut PerTrip, c: [i64; 2]| {
        (to.raw, to.exposed) = (to.raw + c[0] as f64 / trips, to.exposed + c[1] as f64 / trips)
    };
    for (pc, c) in charges {
        let l = line(pc);
        let s = by.entry(l).or_insert_with(|| Statement::new(l, listing));
        for (to, from) in [
            (&mut s.issue, c.issue),
            (&mut s.stall, c.stall),
            (&mut s.wait, c.wait),
            (&mut s.barrier, c.barrier),
            (&mut s.idle, c.idle),
        ] {
            add(to, from);
        }
    }
    let mut out: Vec<Statement> = by.into_values().collect();
    out.sort_by(|a, b| {
        (a.line.is_none()).cmp(&b.line.is_none()).then(b.exposed().total_cmp(&a.exposed())).then(a.line.cmp(&b.line))
    });
    out
}

impl Analysis {
    /// The statement of the instruction at ELF address `pc`, where the capture has lines and one
    /// claims it.
    pub fn stmt(&self, pc: u64) -> Option<Stmt> {
        self.pcs.binary_search_by_key(&pc, |r| r.pc).ok().and_then(|at| self.pcs[at].stmt)
    }

    /// `stmt`'s `kernel.tk` line, trimmed.
    pub fn line(&self, stmt: Stmt) -> &str {
        listed(&self.listing, stmt.line)
    }

    /// `"  @line text"` of `stmt`, else nothing.
    pub(super) fn at(&self, stmt: Option<Stmt>) -> String {
        stmt.map_or(String::new(), |s| format!("  @{} {}", s.line, self.line(s)))
    }
}

/// The main loop's statements, `top` most exposed and the unattributed.
pub(super) fn text(a: &Analysis, o: &mut String, top: usize) {
    let Some(l) = a.main_loop.as_ref().filter(|l| !l.statements.is_empty()) else { return };
    let unit = if a.model.is_some() { "a trip" } else { "an iteration" };
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
        let _ = writeln!(
            o,
            "{line:>5} {:>4} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1}  {}",
            s.pcs,
            s.issue.raw,
            s.stall.raw,
            s.wait.raw,
            s.barrier.raw,
            s.idle.raw,
            s.raw(),
            s.exposed(),
            s.text
        );
    }
}

/// Statements' figures by text, over `units` of a side: the same statement of two programs.
pub(super) fn by_text(a: &Analysis, units: u64) -> HashMap<String, (f64, f64)> {
    let mut by = HashMap::new();
    let Some(l) = &a.main_loop else { return by };
    // The rollup is a trip's; a unit is a trip or an iteration.
    let scale = l.trips.total as f64 / units.max(1) as f64;
    for s in l.statements.iter() {
        let at: &mut (f64, f64) = by.entry(s.text.clone()).or_default();
        (at.0, at.1) = (at.0 + s.raw() * scale, at.1 + s.exposed() * scale);
    }
    by
}
