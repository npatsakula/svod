//! Resident waves over time from a shader engine's occupancy events, which cover every wave of the SE:
//! per SIMD and in all, over the steady window where the dispatch keeps the SE near its robust peak; and the
//! traced SIMD's pipe busy shares over that window from [`hidden::Pipes`](super::hidden::Pipes).

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use super::Unit;
use super::hidden::{Busy, Pipes};
use super::wave::{Cause, Occupancy};

/// The steady window holds the SE at this share of its robust peak or more. Heuristic.
pub const STEADY: (u32, u32) = (9, 10);
/// The robust peak: the resident waves the SE holds or fewer for this share of its span. The true
/// peak passes the kernel's limit where waves free their registers (`MSG_DEALLOC_VGPRS`) before they
/// end and the next workgroup starts beside their tails, which would narrow the window to its spikes.
/// The share is heuristic.
pub const ROBUST: (u32, u32) = (95, 100);

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
    /// The time-weighted [`ROBUST`] percentile of the SE's resident waves over its span.
    pub robust_peak: u32,
    /// `[from, to)`: from the SE first holding [`STEADY`] of `robust_peak` to it last dropping below.
    pub window: (i64, i64),
    /// The SE's resident waves summed over the window's cycles.
    pub wave_cycles: i64,
    /// Every SIMD a wave was resident on during the window, by `(sa, wgp, simd)`.
    pub simds: Vec<Simd>,
}

type Key = (u8, u8, u8);

impl Residency {
    /// `None` without a wave on the SE. The window follows from the SE's totals; one replay of the
    /// events then sums each SIMD's waves over it.
    pub fn new(events: &[Occupancy]) -> Option<Self> {
        let mut events: Vec<&Occupancy> = events.iter().filter(|o| o.cause == Cause::Wave).collect();
        events.sort_by_key(|o| o.time);
        // AMD's `plot_occupancy_resources.py`: a start counts once, an end only for a counted start.
        let mut active = HashSet::<Unit>::new();
        let counted: Vec<bool> =
            events.iter().map(|o| if o.start { active.insert(o.unit) } else { active.remove(&o.unit) }).collect();
        // `(from, total)`: the SE's resident waves after every event of `from`, held to the next segment.
        let (mut segments, mut total) = (Vec::<(i64, u32)>::new(), 0u32);
        for (i, (o, &counted)) in events.iter().zip(&counted).enumerate() {
            if counted {
                total = if o.start { total + 1 } else { total - 1 };
            }
            if events.get(i + 1).is_none_or(|next| next.time > o.time) {
                segments.push((o.time, total));
            }
        }
        let held: Vec<(i64, i64, u32)> = segments.windows(2).map(|w| (w[0].0, w[1].0, w[0].1)).collect();
        let peak = held.iter().map(|s| s.2).max().filter(|&p| p > 0)?;
        let robust_peak = percentile(held.iter().map(|s| (s.2, s.1 - s.0)), ROBUST).max(1);
        let steady = |total: u32| total * STEADY.1 >= robust_peak * STEADY.0;
        let from = held.iter().find(|s| steady(s.2))?.0;
        let to = held.iter().rev().find(|s| steady(s.2))?.1;
        // `waves` held over `[since, at)`, the part within the window, summed into `sum`.
        let within = |sum: i64, since: i64, at: i64, waves: u32| {
            let cycles = at.min(to) - since.max(from);
            (cycles > 0 && waves > 0).then(|| cycles.saturating_mul(waves.into()).saturating_add(sum))
        };
        let wave_cycles = held.iter().fold(0, |sum, &(since, at, waves)| within(sum, since, at, waves).unwrap_or(sum));
        // Per SIMD: `(waves, since, wave_cycles, peak)`, its waves held since `since`.
        let mut simds = BTreeMap::<Key, (u32, i64, i64, u32)>::new();
        let close = |s: &mut (u32, i64, i64, u32), at: i64| {
            if let Some(sum) = within(s.2, s.1, at, s.0) {
                (s.2, s.3) = (sum, s.3.max(s.0));
            }
            s.1 = at;
        };
        for o in events.iter().zip(&counted).filter(|c| *c.1).map(|c| c.0) {
            let simd = simds.entry((o.unit.sa, o.unit.wgp, o.unit.simd)).or_insert((0, o.time, 0, 0));
            close(simd, o.time);
            simd.0 = if o.start { simd.0 + 1 } else { simd.0 - 1 };
        }
        let simds = simds
            .into_iter()
            .filter_map(|((sa, wgp, simd), mut s)| {
                close(&mut s, to);
                (s.3 > 0).then_some(Simd { sa, wgp, simd, wave_cycles: s.2, cycles: to - from, peak: s.3 })
            })
            .collect();
        Some(Self { peak, robust_peak, window: (from, to), wave_cycles, simds })
    }

    /// The SE's time-weighted mean resident waves over the window.
    pub fn mean(&self) -> f64 {
        self.wave_cycles as f64 / (self.window.1 - self.window.0).max(1) as f64
    }

    pub fn simd(&self, sa: u8, wgp: u8, simd: u8) -> Option<&Simd> {
        self.simds.iter().find(|s| (s.sa, s.wgp, s.simd) == (sa, wgp, simd))
    }
}

/// The `(num, den)` time-weighted percentile of `(value, cycles)` held segments: the least value the
/// segments at or below it hold for that share of their cycles.
fn percentile(segments: impl Iterator<Item = (u32, i64)>, (num, den): (u32, u32)) -> u32 {
    let mut segments: Vec<(u32, i64)> = segments.filter(|s| s.1 > 0).collect();
    segments.sort_unstable();
    let span = segments.iter().fold(0i64, |sum, s| sum.saturating_add(s.1));
    let mut held = 0i64;
    for (value, cycles) in segments {
        held = held.saturating_add(cycles);
        if i128::from(held) * i128::from(den) >= i128::from(span) * i128::from(num) {
            return value;
        }
    }
    0
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
