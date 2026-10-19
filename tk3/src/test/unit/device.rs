//! End to end on the attached GPU: the lowered program computes what the
//! interpreter computes. Skips unless `SVOD_DEVICE` names a CUDA device.

use svod_dtype::{DType, DeviceSpec, ScalarDType, default_device::default_device};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::atoms::Target;
use crate::interp::{round_to, run};
use crate::launch::graph_launch;
use crate::layouts::WarpGrid;
use crate::lower::Lowering;
use crate::schedule::{Prefetch, Schedule};

fn cuda_target() -> Option<Target> {
    let spec = default_device();
    matches!(spec, DeviceSpec::Cuda { .. }).then(|| Target::for_device(&spec)).flatten()
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

/// The pipelined GEMM over `cp.async` + `mma.sync` + `ldmatrix` matches the
/// interpreter's bf16/f32 result on every tile of the grid.
#[test_case(128, 128, 64, 128, 128, 32, 2, 2, 4; "one block, two stages")]
#[test_case(256, 128, 256, 128, 64, 32, 3, 4, 2; "four blocks, three stages")]
#[test_case(64, 64, 128, 64, 64, 64, 2, 2, 2; "64-wide k")]
#[allow(clippy::too_many_arguments)]
fn gemm_matches_the_interpreter(
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    bn: usize,
    bk: usize,
    stages: usize,
    wr: u32,
    wc: u32,
) {
    let Some(target) = cuda_target() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let mut prog = super::programs::gemm_nt(m, n, k, bm, bn, bk, stages);
    prog.warps = wr * wc;
    let lowering = Lowering {
        target,
        schedule: Schedule::Uniform { prefetch: Prefetch::CpAsync, unroll: true },
        grid: WarpGrid { rows: wr, cols: wc },
        swizzle: true,
    };

    let mut seed = 11;
    let a: Vec<f32> = (0..m * k).map(|_| round_to(ScalarDType::BFloat16, lcg(&mut seed) as f64) as f32).collect();
    let b: Vec<f32> = (0..n * k).map(|_| round_to(ScalarDType::BFloat16, lcg(&mut seed) as f64) as f32).collect();
    let want = run(
        &prog,
        vec![a.iter().map(|&x| x as f64).collect(), b.iter().map(|&x| x as f64).collect(), vec![0.0; m * n]],
        &[("b", 1)],
    )
    .unwrap();

    let a_t = Tensor::from_slice(&a).cast(DType::BFloat16);
    let b_t = Tensor::from_slice(&b).cast(DType::BFloat16);
    let c_t = Tensor::empty(&[m * n], DType::BFloat16);
    let out = graph_launch(prog, &lowering, &[&a_t, &b_t, &c_t]).unwrap();
    let got: Vec<f32> = out.cast(DType::Float32).to_vec::<f32>().unwrap();
    let mut worst = 0.0f64;
    for (i, (g, w)) in got.iter().zip(&want[2]).enumerate() {
        let diff = (*g as f64 - w).abs();
        worst = worst.max(diff);
        assert!(diff <= 1.6e-2 * w.abs().max(1.0), "c[{}, {}] = {g}, interpreter {w}", i / n, i % n);
    }
    eprintln!("max abs diff {worst:.3e}");
}

/// Throughput probe against tk1's `gemm_nt` on the same device and shapes;
/// prints TFLOP/s and never asserts (run with `--ignored --nocapture`).
#[test]
#[ignore = "perf probe: needs a CUDA device"]
fn gemm_throughput_probe() {
    let Some(target) = cuda_target() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let (m, n, k) = (4096usize, 4096usize, 4096usize);
    let mut seed = 3;
    let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
    let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
    let a_t = Tensor::from_slice(&a).cast(DType::BFloat16);
    let b_t = Tensor::from_slice(&b).cast(DType::BFloat16);
    a_t.realize().unwrap();
    b_t.realize().unwrap();
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let time = |plan: &svod_runtime::ExecutionPlan, label: &str| {
        plan.execute_profiled().unwrap();
        let reps = 10;
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            for k in plan.execute_profiled().unwrap() {
                if let (Some(s), Some(e)) = (k.gpu_start_ns, k.gpu_end_ns) {
                    best = best.min((e - s) as f64 * 1e-9);
                }
            }
        }
        eprintln!("{label}: {:.2} ms, {:.1} TFLOP/s", best * 1e3, flops / best / 1e12);
    };
    // Static shared memory is capped at 48 KB on this path.
    for (bm, bn, bk, stages, wr, wc, group_m, unroll) in [
        (128, 64, 32, 2, 2, 2, 8, true),
        (128, 64, 32, 2, 2, 2, 8, false),
        (128, 64, 32, 2, 2, 2, 0, true),
        (128, 128, 32, 3, 2, 4, 8, true),
        (128, 128, 32, 3, 2, 4, 8, false),
        (64, 128, 64, 2, 2, 2, 8, true),
        (128, 128, 32, 2, 4, 2, 8, false),
    ] {
        let mut prog = super::programs::gemm_nt_ordered(m, n, k, bm, bn, bk, stages, group_m);
        prog.warps = wr * wc;
        let lowering = Lowering {
            target: target.clone(),
            schedule: Schedule::Uniform { prefetch: Prefetch::CpAsync, unroll },
            grid: WarpGrid { rows: wr, cols: wc },
            swizzle: true,
        };
        let c_t = Tensor::empty(&[m * n], DType::BFloat16);
        let out = graph_launch(prog, &lowering, &[&a_t, &b_t, &c_t]).unwrap();
        let plan = out.prepare().unwrap();
        time(&plan, &format!("tk3 {bm}x{bn}x{bk} s{stages} {wr}x{wc} group_m={group_m} unroll={unroll}"));
    }
    let a2 = a_t.try_reshape([m, k]).unwrap();
    let b2 = b_t.try_reshape([n, k]).unwrap();
    let tk1 = svod_tk::gemm_nt(&a2, &b2).unwrap().expect("tk1 serves this shape");
    let plan = tk1.prepare().unwrap();
    time(&plan, "tk1 gemm_nt");
}
