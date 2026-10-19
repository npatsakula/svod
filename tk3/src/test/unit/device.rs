//! End to end on the attached GPU: the lowered program computes what the
//! interpreter computes. Skips unless `SVOD_DEVICE` names a CUDA device.

use svod_dtype::{DType, DeviceSpec, ScalarDType, default_device::default_device};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::atoms::Target;
use crate::build::BF16;
use crate::interp::{round_to, run};
use crate::ir::Program;
use crate::kernels::Batch;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::launch::graph_launch;

fn cuda_target() -> Option<Target> {
    let spec = default_device();
    matches!(spec, DeviceSpec::Cuda { .. }).then(|| Target::for_device(&spec)).flatten()
}

fn plain_gemm(m: usize, n: usize, k: usize, cfg: GemmCfg) -> Program {
    gemm::<BF16>(&GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg })
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
    let cfg = GemmCfg { tile: [bm, bn, bk], stages, warps: [wr, wc], group_m: 0, unroll: true };
    let prog = plain_gemm(m, n, k, cfg);
    let lowering = cfg.lowering(target);

    let mut seed = 11;
    let a: Vec<f32> = (0..m * k).map(|_| round_to(ScalarDType::BFloat16, lcg(&mut seed) as f64) as f32).collect();
    let b: Vec<f32> = (0..n * k).map(|_| round_to(ScalarDType::BFloat16, lcg(&mut seed) as f64) as f32).collect();
    let want = run(
        &prog,
        vec![a.iter().map(|&x| x as f64).collect(), b.iter().map(|&x| x as f64).collect(), vec![0.0; m * n]],
        &[],
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
    // Static shared memory is capped at 48 KB on this path.
    let mut plans: Vec<(String, svod_runtime::ExecutionPlan)> = vec![];
    for (bm, bn, bk, stages, wr, wc, group_m, unroll) in [
        (128, 64, 32, 2, 2, 2, 8, true),
        (128, 64, 32, 2, 2, 2, 8, false),
        (128, 64, 32, 3, 2, 2, 8, false),
        (128, 128, 32, 3, 2, 4, 8, true),
        (128, 128, 32, 3, 2, 4, 8, false),
        (128, 128, 32, 2, 2, 4, 8, false),
    ] {
        let cfg = GemmCfg { tile: [bm, bn, bk], stages, warps: [wr, wc], group_m, unroll };
        let (prog, lowering) = (plain_gemm(m, n, k, cfg), cfg.lowering(target.clone()));
        let c_t = Tensor::empty(&[m * n], DType::BFloat16);
        let out = graph_launch(prog, &lowering, &[&a_t, &b_t, &c_t]).unwrap();
        plans.push((
            format!("tk3 {bm}x{bn}x{bk} s{stages} {wr}x{wc} group_m={group_m} unroll={unroll}"),
            out.prepare().unwrap(),
        ));
    }
    let a2 = a_t.try_reshape([m, k]).unwrap();
    let b2 = b_t.try_reshape([n, k]).unwrap();
    let tk1 = svod_tk::gemm_nt(&a2, &b2).unwrap().expect("tk1 serves this shape");
    plans.push(("tk1 gemm_nt".to_string(), tk1.prepare().unwrap()));

    // Round-robin so clock drift hits every candidate alike; keep the best.
    // The 3060 idles at a low clock: spin the first plan for half a second.
    let warm = std::time::Instant::now();
    while warm.elapsed().as_millis() < 500 {
        plans[0].1.execute().unwrap();
    }
    let mut best = vec![f64::INFINITY; plans.len()];
    for _ in 0..4 {
        for (i, (_, plan)) in plans.iter().enumerate() {
            for _ in 0..5 {
                let run = plan
                    .execute_profiled()
                    .unwrap()
                    .iter()
                    .filter_map(|kp| Some((kp.gpu_end_ns? - kp.gpu_start_ns?) as f64 * 1e-9))
                    .fold(0.0, f64::max);
                best[i] = best[i].min(run);
            }
        }
    }
    for ((label, _), secs) in plans.iter().zip(&best) {
        eprintln!("{label}: {:.2} ms, {:.1} TFLOP/s", secs * 1e3, flops / secs / 1e12);
    }
}
