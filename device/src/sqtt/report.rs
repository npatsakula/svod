//! Per-PC and per-wave cycle totals over stitched waves, as AMD's `code_index.py` accumulates them.

use std::collections::BTreeMap;

use super::Unit;
use super::wave::Wave;

/// The cycles all waves spent on one PC. `latency` sums durations, `stall` the stalls inside them,
/// and `idle` the gaps since the wave's previous record ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stats {
    pub hits: u64,
    pub latency: i64,
    pub stall: i64,
    pub idle: i64,
}

impl Stats {
    fn add(&mut self, other: Stats) {
        self.hits += other.hits;
        self.latency += other.latency;
        self.stall += other.stall;
        self.idle += other.idle;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveTotals {
    pub unit: Unit,
    pub begin: i64,
    pub end: i64,
    pub complete: bool,
    /// Records, and those a PC was attributed to; only the latter count in `stats`.
    pub records: usize,
    pub stats: Stats,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Report {
    pub pcs: BTreeMap<u64, Stats>,
    pub waves: Vec<WaveTotals>,
}

impl Report {
    pub fn new(waves: &[Wave]) -> Self {
        let mut report = Self::default();
        for wave in waves {
            let mut totals = Stats::default();
            let mut prev_end = wave.begin;
            for inst in wave.instructions.iter().filter(|inst| inst.pc != 0) {
                let stats =
                    Stats { hits: 1, latency: inst.duration, stall: inst.stall, idle: (inst.time - prev_end).max(0) };
                prev_end = prev_end.max(inst.time + inst.duration);
                report.pcs.entry(inst.pc).or_default().add(stats);
                totals.add(stats);
            }
            report.waves.push(WaveTotals {
                unit: wave.unit,
                begin: wave.begin,
                end: wave.end,
                complete: wave.complete,
                records: wave.instructions.len(),
                stats: totals,
            });
        }
        report
    }
}
