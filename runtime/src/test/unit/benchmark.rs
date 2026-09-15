use super::*;

struct MockKernel {
    name: String,
    sleep_micros: u64,
}

impl Program for MockKernel {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        std::thread::sleep(Duration::from_micros(self.sleep_micros));
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[test]
fn test_benchmark_basic() {
    let kernel = MockKernel { name: "test".into(), sleep_micros: 100 };
    let config = BenchmarkConfig {
        warmup_runs: 1,
        timing_runs: 3,
        take_minimum: true,
        early_stop: None,
        clear_l2: false,
        warmup_budget: None,
    };

    let result = unsafe { benchmark_kernel(&kernel, &[], &[], None, None, &config) }.unwrap();

    assert_eq!(result.runs.len(), 3);
    assert!(result.min >= Duration::from_micros(100));
    assert!(result.min <= result.mean);
}

#[test]
fn test_benchmark_early_stop() {
    let kernel = MockKernel { name: "slow".into(), sleep_micros: 10000 };
    let config = BenchmarkConfig {
        warmup_runs: 0,
        timing_runs: 5,
        take_minimum: true,
        early_stop: Some(Duration::from_micros(100)),
        clear_l2: false,
        warmup_budget: None,
    };

    let result = unsafe { benchmark_kernel(&kernel, &[], &[], None, None, &config) }.unwrap();

    // Each run is ~10ms; the very first exceeds the 100µs threshold and
    // the loop bails out, so we record exactly one run instead of five.
    assert_eq!(result.runs.len(), 1);
}

#[test]
fn test_benchmark_early_stop_passes_under_cutoff() {
    let kernel = MockKernel { name: "fast".into(), sleep_micros: 50 };
    // Wide margin: a 50us sleep can take >1ms wall on a loaded CI runner, and
    // this test only claims that early-stop does NOT trigger under the cutoff.
    let cutoff = Duration::from_millis(100);
    let config = BenchmarkConfig { early_stop: Some(cutoff * 3), ..BenchmarkConfig::default() };

    let result = unsafe { benchmark_kernel(&kernel, &[], &[], None, None, &config) }.unwrap();

    assert_eq!(result.runs.len(), config.timing_runs);
    assert!(result.min < cutoff);
}

#[test]
fn test_benchmark_early_stop_aborts_over_cutoff() {
    let kernel = MockKernel { name: "slow".into(), sleep_micros: 10000 };
    let cutoff = Duration::from_micros(100);
    let config = BenchmarkConfig { early_stop: Some(cutoff * 3), ..BenchmarkConfig::default() };

    let result = unsafe { benchmark_kernel(&kernel, &[], &[], None, None, &config) }.unwrap();

    assert_eq!(result.runs.len(), 1);
    assert!(result.min > cutoff * 3);
}

/// A backend with GPU stamps reports the device time, not the (longer) wall
/// time around the synchronous dispatch.
struct StampedKernel;

impl Program for StampedKernel {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        std::thread::sleep(Duration::from_millis(5));
        Ok(())
    }

    unsafe fn execute_timed(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    ) -> svod_device::Result<Option<Duration>> {
        unsafe { self.execute(buffers, vals, global_size, local_size, true)? };
        Ok(Some(Duration::from_micros(7)))
    }

    fn name(&self) -> &str {
        "stamped"
    }
}

#[test]
fn benchmark_prefers_gpu_stamped_durations() {
    let result =
        unsafe { benchmark_kernel(&StampedKernel, &[], &[], None, None, &BenchmarkConfig::default()) }.unwrap();
    assert!(result.runs.iter().all(|run| *run == Duration::from_micros(7)), "{:?}", result.runs);
}

/// The clock warm-up stops the moment a dispatch fails, so a broken kernel does
/// not burn the budget.
#[test]
fn warm_clock_stops_on_failure() {
    let mut runs = 0;
    warm_clock(Duration::from_secs(10), || {
        runs += 1;
        (runs < 3).then_some(Duration::from_micros(1))
    });
    assert_eq!(runs, 3, "stops at the first failure");
}

/// A device whose kernel time keeps falling is run until the budget; one whose
/// time has plateaued is released after the floor, and one that is warm from
/// the start pays only the floor.
#[test]
fn warm_clock_runs_until_the_time_plateaus_or_the_budget_ends() {
    // Still falling by 15% per window when the budget ends: runs out the
    // budget, past the floor.
    let mut runs = 0u32;
    let start = Instant::now();
    warm_clock(Duration::from_millis(100), || {
        runs += 1;
        std::thread::sleep(Duration::from_millis(1));
        Some(Duration::from_secs_f64(0.01 * 0.85f64.powi(runs as i32 / 8)))
    });
    assert!(start.elapsed() >= Duration::from_millis(100), "a falling time runs out the budget");

    // Cold for the first 32 runs, flat after: released after the floor, well
    // before a 10 s budget, once two windows agree.
    let mut runs = 0u64;
    let start = Instant::now();
    warm_clock(Duration::from_secs(10), || {
        runs += 1;
        std::thread::sleep(Duration::from_millis(1));
        Some(Duration::from_micros(if runs <= 32 { 300 - 8 * runs } else { 40 }))
    });
    let elapsed = start.elapsed();
    assert!(elapsed >= Duration::from_millis(50) && elapsed < Duration::from_millis(500), "{elapsed:?}");
    assert!(runs >= 48, "at least the floor's runs and two flat windows: {runs}");

    // Warm from the start: the floor, then out.
    let mut runs = 0;
    let start = Instant::now();
    warm_clock(Duration::from_secs(10), || {
        runs += 1;
        std::thread::sleep(Duration::from_millis(1));
        Some(Duration::from_micros(40))
    });
    let elapsed = start.elapsed();
    assert!(elapsed >= Duration::from_millis(50) && elapsed < Duration::from_millis(300), "{elapsed:?}");
}

/// Candidates are timed in turn, round after round, each keeping its minimum;
/// one that fails is excluded from then on and reports no time.
#[test]
fn round_robin_keeps_each_candidate_minimum_and_drops_failures() {
    let mut calls = Vec::new();
    let best = round_robin_min(3, 2, |i| {
        calls.push(i);
        match (i, calls.len()) {
            (1, _) => None,
            (0, n) => Some(Duration::from_micros(10 - n as u64)),
            (2, n) => Some(Duration::from_micros(20 + n as u64)),
            _ => unreachable!(),
        }
    });
    assert_eq!(calls, vec![0, 1, 2, 0, 2], "candidate 1 is not retried after failing");
    assert_eq!(best, vec![Some(Duration::from_micros(6)), None, Some(Duration::from_micros(23))]);
}
