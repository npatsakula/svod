//! Resident waves over time from a shader engine's occupancy events, which cover every wave of the SE:
//! per SIMD and in all, over the steady window where the dispatch keeps the SE near its peak; and the
//! traced SIMD's pipe busy shares over that window from [`hidden::Pipes`](super::hidden::Pipes).

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use super::Unit;
use super::hidden::{Busy, Pipes};
use super::wave::{Cause, Occupancy};

/// The steady window holds the SE at this share of its peak resident waves or more.
pub const STEADY: (u32, u32) = (9, 10);

/// A SIMD's resident waves over the steady window: their sum over its cycles, and the most at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Simd {
    pub sa: u8,
    pub wgp: u8,
    pub simd: u8,
    pub wave_cycles: i64,
    pub cycles: i64,
    pub peak: u32,
}

impl Simd {
    /// Time-weighted mean resident waves.
    pub fn mean(&self) -> f64 {
        self.wave_cycles as f64 / self.cycles.max(1) as f64
    }
}

/// An SE's resident waves: a wave counts from its launch to its end. Context saves and restores
/// and the starts AMD inserts at time 0 for waves the trace saw only end (whose ends are then
/// dropped too) are left out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Residency {
    /// Waves resident on the SE at once, at most.
    pub peak: u32,
    /// `[from, to)`: from the SE first holding [`STEADY`] of `peak` to it last dropping below.
    pub window: (i64, i64),
    /// The SE's resident waves summed over the window's cycles.
    pub wave_cycles: i64,
    /// Every SIMD a wave was resident on during the window, by `(sa, wgp, simd)`.
    pub simds: Vec<Simd>,
}

type Key = (u8, u8, u8);

impl Residency {
    /// `None` without a wave on the SE.
    pub fn new(events: &[Occupancy]) -> Option<Self> {
        let mut events: Vec<&Occupancy> = events.iter().filter(|o| o.cause == Cause::Wave).collect();
        events.sort_by_key(|o| o.time);
        // AMD's `plot_occupancy_resources.py`: a start counts once, an end only for a counted start.
        let mut active = HashSet::<Unit>::new();
        let mut per_simd = BTreeMap::<Key, u32>::new();
        // `(from, total, per SIMD)`: the state after every event of `from`, held to the next segment.
        let mut segments: Vec<(i64, u32, BTreeMap<Key, u32>)> = vec![];
        for (i, o) in events.iter().enumerate() {
            let key = (o.unit.sa, o.unit.wgp, o.unit.simd);
            match o.start {
                true if active.insert(o.unit) => *per_simd.entry(key).or_default() += 1,
                false if active.remove(&o.unit) => *per_simd.get_mut(&key).expect("an active wave's SIMD") -= 1,
                _ => {}
            }
            if events.get(i + 1).is_none_or(|next| next.time > o.time) {
                segments.push((o.time, active.len() as u32, per_simd.clone()));
            }
        }
        let ends: Vec<i64> = segments.iter().skip(1).map(|s| s.0).collect();
        let held: Vec<_> = segments.iter().zip(ends).map(|(s, to)| (s.0, to, s.1, &s.2)).collect();
        let peak = held.iter().map(|s| s.2).max().filter(|&p| p > 0)?;
        let steady = |total: u32| total * STEADY.1 >= peak * STEADY.0;
        let from = held.iter().find(|s| steady(s.2))?.0;
        let to = held.iter().rev().find(|s| steady(s.2))?.1;
        let mut simds = BTreeMap::<Key, (i64, u32)>::new();
        let mut total = 0;
        for &(start, end, count, per_simd) in &held {
            let cycles = end.min(to) - start.max(from);
            if cycles <= 0 {
                continue;
            }
            total += i64::from(count) * cycles;
            for (&key, &waves) in per_simd.iter().filter(|&(_, &waves)| waves > 0) {
                let simd = simds.entry(key).or_default();
                (simd.0, simd.1) = (simd.0 + i64::from(waves) * cycles, simd.1.max(waves));
            }
        }
        let simds = simds
            .into_iter()
            .map(|((sa, wgp, simd), (wave_cycles, peak))| Simd { sa, wgp, simd, wave_cycles, cycles: to - from, peak })
            .collect();
        Some(Self { peak, window: (from, to), wave_cycles: total, simds })
    }

    /// The SE's time-weighted mean resident waves over the window.
    pub fn mean(&self) -> f64 {
        self.wave_cycles as f64 / (self.window.1 - self.window.0).max(1) as f64
    }

    pub fn simd(&self, sa: u8, wgp: u8, simd: u8) -> Option<&Simd> {
        self.simds.iter().find(|s| (s.sa, s.wgp, s.simd) == (sa, wgp, simd))
    }
}

/// Each pipe's busy cycles within a window of `cycles`, from AMD's utilisation intervals: a run of
/// overlapping records lasts their summed durations, so busy counts issue cycles, not the records'
/// union.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PipeBusy {
    pub cycles: i64,
    pub matrix: i64,
    pub valu: i64,
    pub vmem: i64,
    pub scalar: i64,
    /// Any pipe busy.
    pub any: i64,
}

impl PipeBusy {
    pub fn of(pipes: &Pipes, (from, to): (i64, i64)) -> Self {
        let busy = |b: &Busy| b.within(from, to);
        Self {
            cycles: to - from,
            matrix: busy(&pipes.matrix),
            valu: busy(&pipes.valu),
            vmem: busy(&pipes.vmem),
            scalar: busy(&pipes.scalar),
            any: busy(&pipes.any()),
        }
    }

    /// `busy` cycles' share of the window.
    pub fn share(&self, busy: i64) -> f64 {
        busy as f64 / self.cycles.max(1) as f64
    }
}
