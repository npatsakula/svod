//! Shared harness for svod-tk's per-kernel criterion benches (`fa`, `matmul`, `knn`,
//! `kmeans` — one bench binary each). They bench svod-tk's kernels through their public
//! `Tensor` interface, timed the way the model runs them (`prepare()` →
//! `plan.profile(...)`): GPU device time comes from `plan.profile`'s per-kernel HW stamps
//! (it wraps the per-kernel device-time path; the criterion `iter_custom` source), so
//! outlier rejection / CIs operate on real on-device time, not host wall-clock.
//!
//! Run one kernel: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk --bench knn` (or
//! `SVOD_DEVICE=CUDA:0` for a kernel whose `ArchSet` includes CUDA). GPU benches
//! self-skip (record no samples) when no supported GPU is present.

use std::hint::black_box;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use criterion::Bencher;
use criterion::profiler::Profiler;
use svod_dtype::DType;
use svod_runtime::{ExecutionPlan, ProfileOptions, RunProfile};
use svod_tensor::Tensor;

/// A realized random bf16 tensor on the env-selected device.
pub fn randn_bf16(shape: &[usize]) -> Tensor {
    let mut t = Tensor::randn(shape).expect("randn").cast(DType::BFloat16).expect("→bf16");
    t.realize().expect("realize");
    t
}

/// Whether the env-selected device is in the kernel's `archs` with its LLVM GPU
/// backend present (`check_target`), so HW dispatch stamps are available. `cargo
/// bench` has no `#[ignore]`, so a bench self-skips cleanly here instead of
/// recording garbage (or panicking) on CPU.
pub fn requirements_met(archs: svod_tk::ArchSet) -> bool {
    let spec = Tensor::empty(&[1], DType::Float32).device(); // the env/default device
    svod_tk::target::check_target(&spec, archs).is_ok()
}

/// Sum GPU device time (ns) over `iters` replays of a prepared graph plan, via
/// `plan.profile`'s per-kernel HW stamps (an op may lower to several kernels).
/// Timing only — no static analysis or hardware counters, matching the old
/// hand-rolled loop.
pub fn plan_gpu_ns(plan: &ExecutionPlan, iters: u64) -> u64 {
    use svod_runtime::PmcSelection;
    let opts = ProfileOptions { iters: 1, static_analysis: false, counters: PmcSelection::None, ..Default::default() };
    let mut total = 0u64;
    for _ in 0..iters {
        let report = plan.profile(&opts).expect("plan.profile");
        // Pure on-device time: sum HW stamps only, skipping unstamped dispatches
        // (NOT gpu_total(), which falls back to host wall for the unstamped).
        for stage in &report.stages {
            for k in &stage.kernels {
                if let (Some(s), Some(e)) = (k.gpu_start_ns, k.gpu_end_ns) {
                    total += e - s;
                }
            }
        }
    }
    total
}

/// Bench `plan` by GPU device time. Under `cargo bench --profile-time`, also
/// capture the plan's full profile (roofline / occupancy / PMC, configured via
/// `ProfileOptions::from_env`) into the shared [`bench_profiler`], which writes it
/// out on stop. Normal runs are unaffected.
pub fn bench_plan(bencher: &mut Bencher<'_>, plan: &ExecutionPlan) {
    bench_profiler().maybe_capture(plan);
    bencher.iter_custom(|iters| Duration::from_nanos(black_box(plan_gpu_ns(plan, iters))));
}

/// Process-global profiler shared between criterion (via
/// [`Criterion::with_profiler`](criterion::Criterion::with_profiler)) and the
/// bench routines (which capture into it). Both observe the same inner state.
pub fn bench_profiler() -> PlanProfiler {
    static P: OnceLock<PlanProfiler> = OnceLock::new();
    P.get_or_init(PlanProfiler::default).clone()
}

/// Criterion `--profile-time` hook: while profiling one benchmark, profile its
/// plan on every invocation and accumulate by per-kernel min time
/// ([`RunProfile::merge_min`]), so the rich metrics use all the runs criterion
/// pays for. On stop, render the merged table to `<dir>/svod-profile.txt` and echo
/// it to stderr. The svod profiler itself only formats — printing is the harness's
/// (caller's) choice, made here.
#[derive(Clone, Default)]
pub struct PlanProfiler {
    active: Arc<AtomicBool>,
    result: Arc<Mutex<Option<RunProfile>>>,
}

impl PlanProfiler {
    /// Profile `plan` and merge it into the session's accumulator by per-kernel
    /// min time; no-op when not profiling.
    fn maybe_capture(&self, plan: &ExecutionPlan) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        let Ok(run) = plan.profile(&ProfileOptions::from_env()) else { return };
        let mut slot = self.result.lock().expect("profile slot");
        match slot.as_mut() {
            Some(acc) => acc.merge_min(run),
            None => *slot = Some(run),
        }
    }
}

impl Profiler for PlanProfiler {
    fn start_profiling(&mut self, _id: &str, _dir: &Path) {
        *self.result.lock().expect("profile slot") = None;
        self.active.store(true, Ordering::Relaxed);
    }

    fn stop_profiling(&mut self, id: &str, dir: &Path) {
        self.active.store(false, Ordering::Relaxed);
        if let Some(report) = self.result.lock().expect("profile slot").take() {
            let table = report.render_table();
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(dir.join("svod-profile.txt"), &table);
            eprintln!("svod profile [{id}]:\n{table}");
        }
    }
}
