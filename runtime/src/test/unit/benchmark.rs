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

/// A backend that stamps nothing is timed on the wall clock around its
/// synchronous dispatch.
#[test]
fn a_backend_without_stamps_is_timed_on_the_wall_clock() {
    let kernel = MockKernel { name: "test".into(), sleep_micros: 100 };
    let time = unsafe { time_kernel(&kernel, &[], &[], None, None, false) }.unwrap();
    assert!(time.is_some_and(|t| t >= Duration::from_micros(100)), "{time:?}");
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

/// A backend that stamps its first run and then loses its stamps.
struct FlakyStampKernel(std::sync::atomic::AtomicUsize);

impl Program for FlakyStampKernel {
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
        let run = self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok((run == 0).then_some(Duration::from_micros(7)))
    }

    fn name(&self) -> &str {
        "flaky"
    }
}

/// A candidate is timed on one clock only: a stamp it loses in a later round
/// drops it from the search, where a wall time in its place would put a second
/// clock beside its own and its rivals' stamps.
#[test]
fn a_lost_stamp_drops_the_candidate_instead_of_mixing_clocks() {
    let flaky = FlakyStampKernel(std::sync::atomic::AtomicUsize::new(0));
    let kernels: [&dyn Program; 2] = [&flaky, &StampedKernel];
    let best =
        round_robin_min(2, 3, None, |i| unsafe { time_kernel(kernels[i], &[], &[], None, None, false) }.ok().flatten());
    assert_eq!(best, vec![None, Some(Duration::from_micros(7))]);
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
    assert!(elapsed >= Duration::from_millis(400) && elapsed < Duration::from_millis(800), "{elapsed:?}");
    assert!(runs >= 48, "past the cold runs and two flat windows: {runs}");

    // Warm from the start: the floor, then out.
    let mut runs = 0;
    let start = Instant::now();
    warm_clock(Duration::from_secs(10), || {
        runs += 1;
        std::thread::sleep(Duration::from_millis(1));
        Some(Duration::from_micros(40))
    });
    let elapsed = start.elapsed();
    assert!(elapsed >= Duration::from_millis(400) && elapsed < Duration::from_millis(800), "{elapsed:?}");
}

/// Candidates are timed in turn, round after round, each keeping its minimum;
/// one that fails is excluded from then on and reports no time.
#[test]
fn round_robin_keeps_each_candidate_minimum_and_drops_failures() {
    let mut calls = Vec::new();
    let best = round_robin_min(3, 2, None, |i| {
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

/// A candidate past the early-stop bound is not run again but keeps the minimum
/// it reached; one inside it is timed every round, and a bound never drops a
/// candidate that fails outright from reporting nothing.
#[test]
fn round_robin_retires_candidates_past_the_early_stop_bound() {
    let mut calls = Vec::new();
    let best = round_robin_min(3, 3, Some(Duration::from_micros(10)), |i| {
        calls.push(i);
        match i {
            0 => Some(Duration::from_micros(30 - calls.len() as u64)),
            1 => Some(Duration::from_micros(8)),
            _ => None,
        }
    });
    assert_eq!(calls, vec![0, 1, 2, 1, 1], "candidate 0 is not retried after its first run exceeds the bound");
    assert_eq!(best, vec![Some(Duration::from_micros(29)), Some(Duration::from_micros(8)), None]);
}

/// A run at the bound is a competitive run: only a minimum strictly past it
/// retires the candidate.
#[test]
fn round_robin_keeps_timing_a_candidate_at_the_bound() {
    let mut runs = 0;
    let best = round_robin_min(1, 3, Some(Duration::from_micros(10)), |_| {
        runs += 1;
        Some(Duration::from_micros(10))
    });
    assert_eq!(runs, 3);
    assert_eq!(best, vec![Some(Duration::from_micros(10))]);
}

/// A warm batch settles a round past the floor: the first run only sets each
/// minimum, `settle` rounds holding it settle the batch, and the rounds stop
/// there however high the cap.
#[test]
fn steady_times_a_warm_batch_a_round_past_its_floor() {
    let mut calls = Vec::new();
    let best = round_robin_min_steady(2, 2, 10, 3, None, |i| {
        calls.push(i);
        Some(Duration::from_micros(if i == 0 { 10 } else { 20 }))
    });
    assert_eq!(calls, vec![0, 1, 0, 1, 0, 1, 0, 1], "four rounds: the floor of two, then one per stable run");
    assert_eq!(best, vec![Some(Duration::from_micros(10)), Some(Duration::from_micros(20))]);
}

/// A minimum that keeps falling — a clock still lifting under it — resets its
/// stable count run after run and holds the rounds open to the cap, taking
/// every live candidate with it (its rivals' minimums rode the same clock).
#[test]
fn steady_runs_to_the_cap_while_any_minimum_falls() {
    let mut calls = Vec::new();
    let best = round_robin_min_steady(2, 1, 4, 2, None, |i| {
        calls.push(i);
        match i {
            // Falling 10% a run, past the 3% a warm run drifts within.
            0 => Some(Duration::from_secs_f64(
                0.0001 * 0.9f64.powi(calls.iter().filter(|&i| *i == 0).count() as i32 - 1),
            )),
            _ => Some(Duration::from_micros(20)),
        }
    });
    assert_eq!(calls, vec![0, 1, 0, 1, 0, 1, 0, 1], "the cap of four rounds runs both candidates");
    let falling = best[0].expect("its minimum stands");
    assert_eq!(falling, Duration::from_secs_f64(0.0001 * 0.9f64.powi(3)), "the last run is the minimum");
    assert_eq!(best[1], Some(Duration::from_micros(20)));
}

/// A candidate past the early-stop bound and one that fails outright keep
/// their fixed-form fate — the minimum reached stands, nothing runs again —
/// and neither holds the rounds open: only live candidates settle the batch.
#[test]
fn steady_settles_without_its_retired_candidates() {
    let mut calls = Vec::new();
    let best = round_robin_min_steady(3, 1, 10, 2, Some(Duration::from_micros(10)), |i| {
        calls.push(i);
        match i {
            0 => None,
            1 => Some(Duration::from_micros(8)),
            _ => Some(Duration::from_micros(30)),
        }
    });
    assert_eq!(calls, vec![0, 1, 2, 1, 1], "candidate 2 retires at the bound, candidate 0 at its failure");
    assert_eq!(best, vec![None, Some(Duration::from_micros(8)), Some(Duration::from_micros(30))]);
}
