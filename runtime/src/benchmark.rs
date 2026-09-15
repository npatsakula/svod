//! Kernel benchmarking infrastructure for auto-tuning.
//!
//! Provides timing utilities for measuring kernel execution performance,
//! used by beam search optimization to compare candidate kernels.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use svod_device::device::Program;

use crate::Result;

/// Configuration for kernel benchmarking.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup runs (not timed).
    pub warmup_runs: usize,
    /// Number of timing runs.
    pub timing_runs: usize,
    /// Whether to return minimum time (true) or mean (false).
    pub take_minimum: bool,
    /// If set, abort the timing loop the moment any single run exceeds this
    /// threshold. Used by beam search to skip candidates clearly slower than
    /// the current best (typically `early_stop = beam[0].timing * 3`).
    pub early_stop: Option<Duration>,
    /// Invalidate L2 between runs by streaming through a scratch buffer.
    /// Stabilises rankings — without this, second/third runs hit hot caches
    /// and bias beam toward smaller-tile candidates.
    pub clear_l2: bool,
    /// Run the kernel back to back for this long before the first timed run
    /// ([`warm_clock`]): a GPU idles at a fraction of its boost clock and takes
    /// about a second of load to lift it, so a kernel timed cold measures the
    /// clock, not the kernel. `None` skips the warm-up (a device already under
    /// load, or the CPU).
    pub warmup_budget: Option<Duration>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        // 3 timing runs, take the minimum — variance from rayon dispatch
        // and OS scheduling is much larger than per-run overhead, so the
        // min of 3 is a tighter estimate of the kernel's true cost than
        // any longer-running statistic.
        Self {
            warmup_runs: 0,
            timing_runs: 3,
            take_minimum: true,
            early_stop: None,
            clear_l2: false,
            warmup_budget: None,
        }
    }
}

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
/// under load plateaus in its first windows and pays only [`WARM_FLOOR`], so a
/// tuner touching many shapes does not spend the budget on each. See
/// [`BenchmarkConfig::warmup_budget`].
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
/// run of candidate `i`; `None` excludes it for good.
pub fn round_robin_min(
    count: usize,
    rounds: usize,
    mut time: impl FnMut(usize) -> Option<Duration>,
) -> Vec<Option<Duration>> {
    let mut best: Vec<Option<Duration>> = vec![None; count];
    let mut dead = vec![false; count];
    for _ in 0..rounds {
        for i in 0..count {
            if dead[i] {
                continue;
            }
            match time(i) {
                Some(t) => best[i] = Some(best[i].map_or(t, |b| b.min(t))),
                None => (dead[i], best[i]) = (true, None),
            }
        }
    }
    best
}

/// Result of kernel benchmarking.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Minimum execution time.
    pub min: Duration,
    /// Mean execution time.
    pub mean: Duration,
    /// All timing measurements.
    pub runs: Vec<Duration>,
}

impl BenchmarkResult {
    /// Get the timing value based on config preference.
    pub fn timing(&self, take_minimum: bool) -> Duration {
        if take_minimum { self.min } else { self.mean }
    }
}

/// Benchmark a compiled kernel's execution time.
///
/// Runs warmup iterations (discarded), then timing iterations.
/// Returns min/mean/all timings.
///
/// # Safety
///
/// All buffer pointers must be valid for the duration of benchmarking.
/// The kernel will be executed multiple times.
///
/// # Example
///
/// ```ignore
/// let config = BenchmarkConfig::default();
/// let result = unsafe { benchmark_kernel(&kernel, &buffers, &vals, None, None, &config)? };
/// println!("Min time: {:?}", result.min);
/// ```
pub unsafe fn benchmark_kernel(
    kernel: &dyn Program,
    buffers: &[*mut u8],
    vals: &[i64],
    global_size: Option<[usize; 3]>,
    local_size: Option<[usize; 3]>,
    config: &BenchmarkConfig,
) -> Result<BenchmarkResult> {
    // Warm-up (discarded): the clock budget, then the counted runs.
    if let Some(budget) = config.warmup_budget {
        warm_clock(budget, || {
            let start = Instant::now();
            unsafe { kernel.execute(buffers, vals, global_size, local_size, true) }.ok().map(|_| start.elapsed())
        });
    }
    for _ in 0..config.warmup_runs {
        // wait=true: benchmark needs each dispatch to complete before the next
        // (async submit would measure queue time, not kernel time).
        unsafe {
            kernel.execute(buffers, vals, global_size, local_size, /*wait=*/ true)?
        };
    }

    // Timing runs
    let mut runs = Vec::with_capacity(config.timing_runs);
    for i in 0..config.timing_runs {
        if config.clear_l2 && i > 0 {
            invalidate_l2();
        }
        // GPU-stamped duration when the backend has one (Metal command-buffer
        // times); otherwise the wall clock around the synchronous dispatch.
        let start = Instant::now();
        let gpu = unsafe { kernel.execute_timed(buffers, vals, global_size, local_size)? };
        runs.push(gpu.unwrap_or_else(|| start.elapsed()));

        // Min-of-runs early stop: abort only when the best run so far still
        // exceeds the threshold. A single jitter outlier in an otherwise
        // competitive candidate must not disqualify it — `take_minimum=true`
        // already discards tail noise from the final result.
        if let Some(threshold) = config.early_stop
            && runs.iter().copied().min().expect("runs non-empty after push") > threshold
        {
            break;
        }
    }

    // Calculate statistics
    let min = runs.iter().copied().min().unwrap_or(Duration::ZERO);
    let total: Duration = runs.iter().sum();
    let mean = total / runs.len().max(1) as u32;

    Ok(BenchmarkResult { min, mean, runs })
}

/// Force rayon's global thread pool to materialise.
///
/// Subsequent rayon calls dispatch in O(1), but the lazy initialisation can
/// dominate the first 1-2 measurements at the small kernel sizes BEAM-time
/// uses. Call this once before a benchmark loop to remove that bias.
pub fn warmup_thread_pool() {
    rayon::join(|| (), || ());
}

/// Stream through a 16 MiB scratch buffer to evict L2 between timing runs.
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
