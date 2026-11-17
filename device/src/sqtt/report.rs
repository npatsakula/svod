//! Per-PC cycle totals over stitched waves, as AMD's `code_index.py` accumulates them.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

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
    /// Saturating: a corrupt trace's cycles may pass `i64`.
    pub fn add(&mut self, other: Stats) {
        self.hits = self.hits.saturating_add(other.hits);
        self.latency = self.latency.saturating_add(other.latency);
        self.stall = self.stall.saturating_add(other.stall);
        self.idle = self.idle.saturating_add(other.idle);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Report {
    pub pcs: BTreeMap<u64, Stats>,
}

impl Report {
    pub fn new(waves: &[Wave]) -> Self {
        let mut report = Self::default();
        for wave in waves {
            let mut prev_end = wave.begin;
            for inst in wave.instructions.iter().filter(|inst| inst.pc != 0) {
                let stats =
                    Stats { hits: 1, latency: inst.duration, stall: inst.stall, idle: (inst.time - prev_end).max(0) };
                prev_end = prev_end.max(inst.time + inst.duration);
                report.pcs.entry(inst.pc).or_default().add(stats);
            }
        }
        report
    }
}

/// `n` samples' order statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spread {
    pub n: usize,
    pub min: i64,
    pub median: i64,
    pub p90: i64,
    pub max: i64,
}

impl Spread {
    pub fn of(mut samples: Vec<i64>) -> Option<Self> {
        samples.sort_unstable();
        let n = samples.len();
        let at = |q: usize| samples[(n * q / 100).min(n - 1)];
        (n > 0).then(|| Self { n, min: samples[0], median: at(50), p90: at(90), max: samples[n - 1] })
    }
}
