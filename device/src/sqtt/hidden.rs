//! AMD's `analysis/hidden_latency.py`: per `(SE, SIMD)`, the busy intervals of the matrix, VALU, vector
//! memory and scalar pipes over the traced waves, and for every record the idle, stall and issue cycles
//! another pipe's activity overlapped. Ported as is: a pipe's overlapping records add their durations
//! instead of taking a union, and a matrix record also puts three quarters of its issue on the VALU.

use std::collections::{BTreeMap, HashSet};
use std::ops::AddAssign;

use serde::{Deserialize, Serialize};

use super::Category;
use super::asm::{Kind, classify};
use super::wave::{Instruction, OtherSimd, Trace, Wave};

/// Cycles of a record that concurrent pipe activity hid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Hidden {
    pub idle: i64,
    pub stall: i64,
    pub issue: i64,
}

impl Hidden {
    fn of((idle, stall, issue): (i64, i64, i64)) -> Self {
        Self { idle, stall, issue }
    }
}

impl AddAssign for Hidden {
    fn add_assign(&mut self, o: Self) {
        self.idle = self.idle.saturating_add(o.idle);
        self.stall = self.stall.saturating_add(o.stall);
        self.issue = self.issue.saturating_add(o.issue);
    }
}

/// One pipe's busy intervals `(end, cycles)`, ordered and disjoint, each starting at `end - cycles`.
/// Its sums saturate: a corrupt trace's cycles may pass `i64`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Busy {
    intervals: Vec<(i64, i64)>,
    /// `prefix[i]`: the cycles of the first `i` intervals.
    prefix: Vec<i64>,
}

impl Busy {
    fn new(intervals: Vec<(i64, i64)>) -> Self {
        let sums = intervals.iter().scan(0i64, |sum, &(_, cycles)| {
            *sum = sum.saturating_add(cycles);
            Some(*sum)
        });
        Self { prefix: std::iter::once(0).chain(sums).collect(), intervals }
    }

    /// `_compute_util`: records `(issue, cycles)` sorted, each one that starts within the run before it
    /// adding its full cycles to that run.
    fn coalesce(mut records: Vec<(i64, i64)>) -> Self {
        records.sort_unstable();
        let mut runs: Vec<(i64, i64)> = vec![];
        for (clock, cycles) in records {
            match runs.last_mut() {
                Some(run) if clock <= run.0.saturating_add(run.1) => run.1 = run.1.saturating_add(cycles),
                Some(_) => runs.push((clock, cycles)),
                None => runs.push((clock, cycles)),
            }
        }
        Self::new(runs.into_iter().map(|(clock, cycles)| (clock.saturating_add(cycles), cycles)).collect())
    }

    /// `_compute_union`: the union of two pipes' intervals.
    fn union(&self, other: &Self) -> Self {
        let mut out: Vec<(i64, i64)> = vec![];
        let mut append = |(end, cycles): (i64, i64)| {
            let begin = end - cycles;
            if end <= begin {
                return;
            }
            match out.last_mut() {
                Some(last) if begin <= last.0 => {
                    let (last_begin, merged) = (last.0 - last.1, last.0.max(end));
                    *last = (merged, merged - last_begin);
                }
                _ => out.push((end, cycles)),
            }
        };
        let (a, b) = (&self.intervals, &other.intervals);
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            if a[i].0 - a[i].1 <= b[j].0 - b[j].1 {
                append(a[i]);
                i += 1;
            } else {
                append(b[j]);
                j += 1;
            }
        }
        a[i..].iter().chain(&b[j..]).for_each(|&interval| append(interval));
        Self::new(out)
    }

    pub fn intervals(&self) -> &[(i64, i64)] {
        &self.intervals
    }

    /// `_covered_until`: busy cycles before `clock`. Python's `bisect_right`, verbatim, so an unordered
    /// list searches as AMD's does.
    pub fn covered_until(&self, clock: i64) -> i64 {
        let (mut lo, mut hi) = (0, self.intervals.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if clock < self.intervals[mid].0 { hi = mid } else { lo = mid + 1 }
        }
        let mut covered = self.prefix[lo];
        if let Some(&(end, cycles)) = self.intervals.get(lo).filter(|&&(end, cycles)| clock > end - cycles) {
            covered = covered.saturating_add(clock.min(end) - (end - cycles));
        }
        covered
    }

    /// Busy cycles within `[from, to)`.
    pub fn within(&self, from: i64, to: i64) -> i64 {
        (self.covered_until(to) - self.covered_until(from)).max(0)
    }

    /// `_indexed_interval`: the busy cycles in a record's idle gap since `last`, its stall and its issue.
    fn hides(&self, last: i64, clock: i64, stall: i64, cycles: i64) -> (i64, i64, i64) {
        if self.intervals.is_empty() {
            return (0, 0, 0);
        }
        let (mut idle, mut hidden_stall, mut issue) = (0, 0, 0);
        let (mut at_clock, mut at_issue) = (None, None);
        if clock > last {
            let at = self.covered_until(clock);
            idle = at - self.covered_until(last);
            at_clock = Some(at);
        }
        if stall > 0 {
            let at = *at_clock.get_or_insert_with(|| self.covered_until(clock));
            let after = *at_issue.insert(self.covered_until(clock + stall));
            hidden_stall = after - at;
        }
        if cycles > stall {
            let after = *at_issue.get_or_insert_with(|| self.covered_until(clock + stall));
            issue = self.covered_until(clock + cycles) - after;
        }
        (idle, hidden_stall, issue)
    }
}

/// `_build_pipe_utilization`: the pipes of one SIMD's waves, the SE's other-SIMD memory ops on `vmem`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Pipes {
    pub matrix: Busy,
    pub valu: Busy,
    pub vmem: Busy,
    pub scalar: Busy,
}

fn vector_memory(category: Category) -> bool {
    matches!(category, Category::Lds | Category::Vmem | Category::Flat)
}

fn scalar(category: Category) -> bool {
    matches!(category, Category::Salu | Category::Smem)
}

impl Pipes {
    pub fn new<'a>(waves: impl IntoIterator<Item = &'a Wave>, other: &[OtherSimd], matrix: &HashSet<u64>) -> Self {
        let (mut wmma, mut valu, mut vmem, mut scal) = (vec![], vec![], vec![], vec![]);
        for inst in waves.into_iter().flat_map(|w| &w.instructions) {
            let (clock, cycles) = (inst.time + inst.stall, inst.duration - inst.stall);
            if matrix.contains(&inst.pc) {
                wmma.push((clock, cycles));
                valu.push((clock, (3 * cycles).div_euclid(4)));
            } else if inst.category == Category::Valu {
                valu.push((clock, cycles));
            } else if vector_memory(inst.category) {
                vmem.push((clock, cycles));
            } else if scalar(inst.category) {
                scal.push((clock, cycles));
            }
        }
        vmem.extend(other.iter().filter(|o| o.cycles > 0).map(|o| (o.time, i64::from(o.cycles))));
        Self {
            matrix: Busy::coalesce(wmma),
            valu: Busy::coalesce(valu),
            vmem: Busy::coalesce(vmem),
            scalar: Busy::coalesce(scal),
        }
    }

    /// The union of every pipe.
    pub fn any(&self) -> Busy {
        self.valu.union(&self.matrix).union(&self.vmem).union(&self.scalar)
    }
}

/// One SIMD's pipes and their unions: what [`Self::record`] measures a record's hidden cycles against.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Hiding {
    pub pipes: Pipes,
    math: Busy,
    vector: Busy,
    all: Busy,
    matrix: HashSet<u64>,
}

impl Hiding {
    pub fn new<'a>(waves: impl IntoIterator<Item = &'a Wave>, other: &[OtherSimd], matrix: &HashSet<u64>) -> Self {
        let pipes = Pipes::new(waves, other, matrix);
        let math = pipes.valu.union(&pipes.matrix);
        let vector = math.union(&pipes.vmem);
        let all = vector.union(&pipes.scalar);
        Self { pipes, math, vector, all, matrix: matrix.clone() }
    }

    /// `_analyze_scope`'s step: the cycles of `inst`'s gap since `last` (the wave's previous record's
    /// issue plus duration, else its start), stall and issue that another pipe's activity hid.
    pub fn record(&self, last: i64, inst: &Instruction) -> Hidden {
        let (clock, stall, cycles) = (inst.time, inst.stall, inst.duration);
        let hides = |busy: &Busy| busy.hides(last, clock, stall, cycles);
        let mem_only = |h: (i64, i64, i64)| h.0 + h.1;
        let sum = |h: (i64, i64, i64)| h.0 + h.1 + h.2;
        let no_issue = |h: (i64, i64, i64)| (h.0, h.1, 0);
        let pipes = &self.pipes;
        Hidden::of(if self.matrix.contains(&inst.pc) {
            let (valu, wmma) = (no_issue(hides(&pipes.valu)), no_issue(hides(&pipes.matrix)));
            if mem_only(valu) > mem_only(wmma) { valu } else { wmma }
        } else if inst.category == Category::Valu {
            let (valu, wmma) = (no_issue(hides(&pipes.valu)), hides(&pipes.matrix));
            if mem_only(valu) > sum(wmma) { valu } else { wmma }
        } else if vector_memory(inst.category) {
            let (math, vmem) = (hides(&self.math), no_issue(hides(&pipes.vmem)));
            if sum(math) > mem_only(vmem) { math } else { vmem }
        } else if scalar(inst.category) {
            let (vector, scal) = (hides(&self.vector), no_issue(hides(&pipes.scalar)));
            if sum(vector) > mem_only(scal) { vector } else { scal }
        } else {
            hides(&self.all)
        })
    }
}

/// `_analyze_scope`: per PC, the cycles of one SIMD's records the other pipes' activity hid.
pub fn scope<'a>(
    waves: impl IntoIterator<Item = &'a Wave> + Clone,
    other: &[OtherSimd],
    matrix: &HashSet<u64>,
) -> BTreeMap<u64, Hidden> {
    let hiding = Hiding::new(waves.clone(), other, matrix);
    let mut by_pc = BTreeMap::<u64, Hidden>::new();
    for wave in waves {
        let mut last = wave.begin;
        for inst in &wave.instructions {
            if inst.pc != 0 {
                *by_pc.entry(inst.pc).or_default() += hiding.record(last, inst);
            }
            last = inst.time + inst.duration;
        }
    }
    by_pc
}

/// Hidden cycles per PC, over all scopes and per `(SE, SIMD)`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HiddenLatency {
    pub by_pc: BTreeMap<u64, Hidden>,
    pub by_scope: BTreeMap<(u32, u8), BTreeMap<u64, Hidden>>,
}

/// The PCs of `code` (`(pc, llvm-objdump line)`) that issue a matrix instruction.
pub fn matrix_pcs(code: &[(u64, &str)]) -> HashSet<u64> {
    code.iter().filter(|(_, line)| classify(line).kind == Kind::Matrix).map(|&(pc, _)| pc).collect()
}

/// `analyze_hidden_latency` over the stitched traces of SEs `(se, trace)`.
pub fn analyze<'a>(ses: impl IntoIterator<Item = (u32, &'a Trace)>, matrix: &HashSet<u64>) -> HiddenLatency {
    let mut result = HiddenLatency::default();
    for (se, trace) in ses {
        let simds: std::collections::BTreeSet<u8> = trace.waves.iter().map(|w| w.unit.simd).collect();
        for simd in simds {
            let waves = trace.waves.iter().filter(|w| w.unit.simd == simd);
            let scope = scope(waves, &trace.other_simd, matrix);
            for (&pc, &hidden) in &scope {
                *result.by_pc.entry(pc).or_default() += hidden;
            }
            result.by_scope.insert((se, simd), scope);
        }
    }
    result
}
