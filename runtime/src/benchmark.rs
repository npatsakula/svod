//! Kernel benchmarking infrastructure for auto-tuning.
//!
//! Provides timing utilities for measuring kernel execution performance,
//! used by beam search optimization to compare candidate kernels.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use svod_device::device::Program;

use crate::Result;

/// The load a GPU needs before a timing means anything: the RTX 3060 idles at
/// 210 MHz against a 2130 MHz boost and lifts within about a second; the same
/// kernel times 3x apart cold.
pub const CLOCK_WARMUP: Duration = Duration::from_millis(1500);

/// Dispatches per plateau check in [`warm_clock`].
const WARM_WINDOW: usize = 8;
/// The least a warm-up runs, so a plateau seen in the first window of a cold
/// device (the clock has not started lifting yet) does not end it.
const WARM_FLOOR: Duration = Duration::from_millis(50);

/// Run `dispatch` back to back until the kernel's time stops falling — a cold
/// device's clock lifts under load — or `budget` of wall time has elapsed, or a
/// dispatch fails. `dispatch` returns one run's duration (`None` on failure).
/// The time is checked one window of runs against the previous: once a window's
/// minimum no longer beats the last by 5%, the clock is up. A device already
/// under load plateaus in its first windows and pays only `WARM_FLOOR`, so a
/// tuner touching many shapes does not spend the budget on each. A GPU idles at
/// a fraction of its boost clock and takes about a second of load to lift it,
/// so a kernel timed cold measures the clock, not the kernel; the CPU needs no
/// lift.
pub fn warm_clock(budget: Duration, mut dispatch: impl FnMut() -> Option<Duration>) {
    let start = Instant::now();
    let (mut previous, mut current, mut runs) = (Duration::MAX, Duration::MAX, 0usize);
    while start.elapsed() < budget {
        let Some(t) = dispatch() else { return };
        current = current.min(t);
        runs += 1;
        if runs % WARM_WINDOW == 0 {
            let lifted = current.as_secs_f64() >= previous.as_secs_f64() * 0.95;
            if lifted && start.elapsed() >= WARM_FLOOR {
                return;
            }
            (previous, current) = (current, Duration::MAX);
        }
    }
}

/// Each candidate's minimum over `rounds` rounds of timing every candidate in
/// turn, so none is judged at a clock the others were not (timing them one after
/// another lets the first lift the clock for the rest). `time(i)` is one timed
/// run of candidate `i`; `None` excludes it for good. A candidate whose minimum
/// is already past `early_stop` is not run again — it cannot win, and a search
/// sets the bound at a multiple of its incumbent to spend the rounds on the
/// ones that can — but the minimum it reached stands.
pub fn round_robin_min(
    count: usize,
    rounds: usize,
    early_stop: Option<Duration>,
    mut time: impl FnMut(usize) -> Option<Duration>,
) -> Vec<Option<Duration>> {
    let mut best: Vec<Option<Duration>> = vec![None; count];
    let mut live = vec![true; count];
    for _ in 0..rounds {
        for i in 0..count {
            if !live[i] {
                continue;
            }
            match time(i) {
                Some(t) => {
                    let t = best[i].map_or(t, |b| b.min(t));
                    best[i] = Some(t);
                    live[i] = early_stop.is_none_or(|bound| t <= bound);
                }
                None => (live[i], best[i]) = (false, None),
            }
        }
    }
    best
}

/// One synchronous timed run of `kernel` on its backend's clock: the device's
/// own stamp where it keeps one, else the wall clock around the dispatch
/// ([`Program::execute_timed`]). `None` is a sample the backend lost, which a
/// caller drops rather than time the kernel on a second clock. `evict_host_cache`
/// streams a scratch buffer through the host caches first; that reaches only a
/// kernel the CPU runs.
///
/// # Safety
///
/// All buffer pointers must be valid for the duration of the run.
pub unsafe fn time_kernel(
    kernel: &dyn Program,
    buffers: &[*mut u8],
    vals: &[i64],
    global_size: Option<[usize; 3]>,
    local_size: Option<[usize; 3]>,
    evict_host_cache: bool,
) -> Result<Option<Duration>> {
    if evict_host_cache {
        invalidate_l2();
    }
    Ok(unsafe { kernel.execute_timed(buffers, vals, global_size, local_size)? })
}

/// Force rayon's global thread pool to materialise.
///
/// Subsequent rayon calls dispatch in O(1), but the lazy initialisation can
/// dominate the first 1-2 measurements at the small kernel sizes BEAM-time
/// uses. Call this once before a benchmark loop to remove that bias.
pub fn warmup_thread_pool() {
    rayon::join(|| (), || ());
}

/// Stream through a 16 MiB scratch buffer to evict the *host* L2 before a
/// timing run. This is a CPU-kernel tool: a GPU keeps its device caches.
///
/// Apple M1 P-core L2 is 12 MiB, A14/M2 L2 caches are similar; 16 MiB is
/// large enough to fully evict L2 on common Apple Silicon and x86 desktop
/// CPUs. The scratch buffer is allocated once (per process) via `OnceLock`
/// and reused across calls. `black_box` prevents the compiler from eliding
/// the read.
fn invalidate_l2() {
    const SCRATCH_BYTES: usize = 16 * 1024 * 1024;
    static SCRATCH: OnceLock<Vec<u8>> = OnceLock::new();
    let scratch = SCRATCH.get_or_init(|| vec![0u8; SCRATCH_BYTES]);

    let mut acc: u8 = 0;
    let stride = 64; // touch one byte per cache line
    let mut i = 0;
    while i < scratch.len() {
        acc = acc.wrapping_add(scratch[i]);
        i += stride;
    }
    std::hint::black_box(acc);
}

#[cfg(test)]
#[path = "test/unit/benchmark.rs"]
mod tests;
