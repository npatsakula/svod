//! Side by side, one process, one loop: tk1's `gemm_nt` and tk3's GEMM on
//! Qwen3-Embedding-0.6B's four projections at M 4096, bf16 — the op layer's
//! tuned pick and pinned configs worth watching — reported as the min and
//! the mean of the same runs, so no harness's statistic is compared against
//! another's.

use svod_dtype::default_device::default_device;
use svod_dtype::{DType, DeviceSpec};
use svod_tensor::Tensor;
use svod_tk3::atoms::Target;
use svod_tk3::build::BF16;
use svod_tk3::kernels::Batch;
use svod_tk3::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use svod_tk3::launch::graph_launch;
use svod_tk3::ops::{self, Act, Linear};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

/// A realized flat bf16 tensor of `n` values.
fn flat(n: usize, seed: &mut u64) -> Tensor {
    let v: Vec<f32> = (0..n).map(|_| lcg(seed)).collect();
    let t = Tensor::from_slice(&v).cast(DType::BFloat16);
    t.realize().unwrap();
    t
}

const fn cfg(tile: [usize; 3], warps: [u32; 2], unroll: bool) -> GemmCfg {
    GemmCfg { tile, stages: 2, warps, group_m: 8, unroll }
}

#[test]
#[ignore = "perf probe: needs a GPU"]
fn tk1_vs_tk3_gemm_side_by_side() {
    let device = default_device();
    if !matches!(device, DeviceSpec::Amd { .. } | DeviceSpec::Cuda { .. }) {
        eprintln!("skipped: no GPU");
        return;
    }
    let target = Target::for_device(&device).expect("a tk3 target");
    let plain = Epilogue::default();
    let residual = Epilogue { residual: true, ..plain };
    let swiglu = Epilogue { act: Act::Silu, gated: true, ..plain };
    // (name, m, n, k, epilogue, the hand table's pick, the lattice's best where it differed)
    let shapes = [
        ("qkv", 4096, 4096, 1024, plain, cfg([128, 128, 32], [4, 2], false), None),
        ("o_proj+res", 4096, 1024, 2048, residual, cfg([128, 128, 32], [2, 4], false), None),
        (
            "gate_up swiglu",
            4096,
            3072,
            1024,
            swiglu,
            cfg([128, 128, 32], [2, 4], false),
            Some(cfg([128, 64, 32], [2, 4], false)),
        ),
        (
            "down+res",
            4096,
            1024,
            3072,
            residual,
            cfg([128, 128, 32], [2, 4], false),
            Some(cfg([128, 256, 32], [4, 4], true)),
        ),
    ];
    let mut seed = 7;
    for (name, m, n, k, epi, tabled, lattice) in shapes {
        let halves = if epi.gated { 2 } else { 1 };
        let (a, b) = (flat(m * k, &mut seed), flat(halves * n * k, &mut seed));
        let r = epi.residual.then(|| flat(m * n, &mut seed));
        let (x2, w2) = (a.try_reshape([m, k]).unwrap(), b.try_reshape([halves * n, k]).unwrap());
        let r2 = r.as_ref().map(|r| r.try_reshape([m, n]).unwrap());
        let mut plans = vec![];

        // tk1, as the model on `origin/main` ran it.
        let tk1 = if epi.gated {
            let pair = svod_tk::swiglu_pair_width(&x2.device()).expect("a pair width");
            svod_tk::gemm_nt_with_epilogue(&x2, &w2, svod_tk::Epilogue::SwiGlu { pair })
        } else if let Some(r2) = &r2 {
            svod_tk::gemm_nt_with_epilogue(&x2, &w2, svod_tk::Epilogue::Add(r2))
        } else {
            svod_tk::gemm_nt(&x2, &w2)
        };
        plans.push(("tk1".to_string(), tk1.expect("tk1 builds").expect("tk1 applies").prepare().unwrap()));

        // tk3 through the op layer: the tune store's pick, as the model runs it.
        let opts = Linear { gated: epi.gated, act: epi.act, residual: r2.as_ref(), ..Linear::default() };
        plans.push(("tk3 ops (tuned)".to_string(), ops::linear(&x2, &w2, opts).unwrap().prepare().unwrap()));

        // tk3 at a pinned config.
        let mut pinned = |label: &str, c: GemmCfg| {
            let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: epi, cfg: c };
            let out = Tensor::empty(&[m * n], DType::BFloat16);
            let mut params: Vec<&Tensor> = vec![&a, &b];
            if let Some(r) = &r {
                params.push(r);
            }
            params.push(&out);
            let y = graph_launch(gemm::<BF16>(&spec), &c.lowering(target.clone()), &params).unwrap();
            plans.push((format!("tk3 {label} {:?} {:?} unroll={}", c.tile, c.warps, c.unroll), y.prepare().unwrap()));
        };
        pinned("tabled", tabled);
        if let Some(l) = lattice {
            pinned("lattice", l);
        }

        // One loop for all: half a second of warm-up, then round-robin so clock
        // drift hits every plan alike; every run's longest kernel is recorded.
        let warm = std::time::Instant::now();
        while warm.elapsed().as_millis() < 500 {
            plans[0].1.execute().unwrap();
        }
        let mut runs: Vec<Vec<f64>> = vec![vec![]; plans.len()];
        for _ in 0..6 {
            for (i, (_, plan)) in plans.iter().enumerate() {
                for _ in 0..5 {
                    let us = plan
                        .execute_profiled()
                        .unwrap()
                        .iter()
                        .filter_map(|kp| Some((kp.gpu_end_ns? - kp.gpu_start_ns?) as f64 * 1e-3))
                        .fold(0.0, f64::max);
                    runs[i].push(us);
                }
            }
        }
        let flops = 2.0 * (m * halves * n * k) as f64;
        eprintln!("== {name}: {m}x{n}x{k} gated={} residual={}", epi.gated, epi.residual);
        for ((label, _), r) in plans.iter().zip(&runs) {
            let min = r.iter().cloned().fold(f64::INFINITY, f64::min);
            let mean = r.iter().sum::<f64>() / r.len() as f64;
            eprintln!(
                "  {label:52} min {min:7.1} us ({:5.1} TFLOP/s)  mean {mean:7.1} us ({:5.1} TFLOP/s)",
                flops / min / 1e6,
                flops / mean / 1e6
            );
        }
    }
}
