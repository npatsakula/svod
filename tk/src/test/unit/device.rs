//! End to end on the attached GPU: the lowered program computes what the
//! interpreter computes, on any vendor (the warpgroup test: a device with a
//! warpgroup core; the two-crew test: one that interleaves crews).

use svod_dtype::{DType, ScalarDType, default_device::default_device};
use svod_tensor::Tensor;
use test_case::test_case;

use super::declared;
use crate::atoms::{Core, Device, Fill};
use crate::build::BF16;
use crate::interp::{round_to, run};
use crate::ir::Program;
use crate::kernels::Batch;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::launch::graph_launch;
use crate::ops::config::Planner;

fn plain_gemm(m: usize, n: usize, k: usize, cfg: GemmCfg) -> Program {
    gemm::<BF16>(&GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg })
}

/// The GEMM configs the tuner times on the device for `spec`'s shape (its
/// admitted shortlist of the device's ranking) and its pick among them.
pub(super) fn device_gemms(planner: &Planner, spec: &GemmSpec) -> (Vec<GemmCfg>, GemmCfg) {
    let GemmSpec { m, n, k, epilogue, .. } = *spec;
    let ranked = planner.gemm_ranked(1, m, n, k, epilogue.gated);
    let build = |cfg: GemmCfg| vec![(gemm::<BF16>(&GemmSpec { cfg, ..spec.clone() }), cfg.lowering(&planner.device))];
    let shape = [spec.batch.capacity(), m, n, k];
    let (admitted, pick) = super::tuned_on_device(
        planner,
        ("gemm", DType::BFloat16, &shape, crate::ops::DEVICE_TEST),
        &ranked,
        <[_]>::to_vec,
        build,
    );
    (admitted, pick.expect("a GEMM the planner runs"))
}

/// The next [`lcg`](super::lcg) draw as an f32.
fn lcg(seed: &mut u64) -> f32 {
    super::lcg(seed) as f32
}

/// The pipelined GEMM matches the interpreter's bf16/f32 result on every
/// tile of the grid, over the ring the device's fill can run (register
/// staging: two slots at most).
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
    let Some(target) = super::device_target() else { return };
    let stages = target.fill.stages(stages);
    let cfg =
        GemmCfg { core: Core::Warp, tile: [bm, bn, bk], stages, warps: [wr, wc], group_m: 0, unroll: true, crews: 1 };
    check_gemm(&declared(target), m, n, k, cfg);
}

/// Every config the op layer may pick or tune to (the device's admitted
/// shortlists), on a grid with partial row and column tiles.
#[test]
fn every_gemm_candidate_matches_the_interpreter() {
    let Some(planner) = super::device_planner() else { return };
    let mut cfgs = vec![];
    for (m, n, k) in [(4096, 4096, 4096), (704, 512, 512)] {
        let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg: BIG };
        cfgs.extend(device_gemms(&planner, &spec).0.into_iter().filter(|c| !cfgs.contains(c)).collect::<Vec<_>>());
    }
    for cfg in cfgs {
        check_gemm(&planner.device, 200, 136, 128, cfg);
    }
}

/// A warpgroup GEMM against the warp-core one of the same block tile on the
/// same operands, over partial tiles and a `k` of several trips: both agree
/// with the interpreter as every config must, and with each other bit for
/// bit (the two cores accumulate a `k16` step alike; measured so on an H100
/// over every shape here).
#[test_case([128, 128, 64], [8, 1], [2, 4], 3; "128x128x64 two groups")]
#[test_case([128, 256, 64], [8, 1], [2, 4], 2; "128x256x64 two groups")]
#[test_case([64, 128, 32], [4, 1], [2, 2], 3; "64x128x32 one group")]
#[test_case([64, 64, 16], [4, 1], [2, 2], 3; "64x64x16 one group")]
#[test_case([128, 96, 64], [8, 1], [4, 1], 2; "128x96x64 three n32")]
#[test_case([256, 128, 64], [16, 1], [4, 4], 2; "256x128x64 four groups")]
#[test_case([128, 256, 64], [8, 2], [4, 4], 2; "128x256x64 two by two")]
fn warpgroup_gemm_matches_the_warp_core(tile: [usize; 3], group: [u32; 2], warp: [u32; 2], stages: usize) {
    let Some(target) = super::group_target() else { return };
    let device = declared(target);
    // Rows stay 16-byte aligned for the chunk fills: `n` a multiple of eight.
    let (m, n, k) = (tile[0] * 2 - 5, tile[1] + 8, tile[2] * 5);
    let cfgs = [(Core::Group, group), (Core::Warp, warp)].map(|(core, warps)| GemmCfg {
        tile,
        stages,
        warps,
        group_m: 8,
        unroll: false,
        core,
        crews: 1,
    });
    let outs: Vec<Vec<f32>> = cfgs.iter().map(|cfg| check_gemm(&device, m, n, k, *cfg)).collect();
    let (same, worst) = outs[0].iter().zip(&outs[1]).fold((0usize, 0.0f32), |(same, worst), (g, w)| {
        (same + usize::from(g.to_bits() == w.to_bits()), worst.max((g - w).abs() / w.abs().max(1.0)))
    });
    eprintln!(
        "{tile:?} {group:?} vs {warp:?}: {same} of {} bit-equal, worst relative difference {worst:.3e}",
        outs[0].len()
    );
    assert_eq!(same, outs[0].len(), "{tile:?}: the warpgroup and warp cores differ");
}

/// Two crews a cluster apart, on a target that interleaves them, compute
/// what the interpreter computes over partial row and column tiles, and
/// store bit for bit what the uniform kernel of the same tile stores (the
/// k-slices reduce in the uniform kernel's order).
#[test_case([256, 256, 64]; "256x256x64")]
#[test_case([256, 256, 32]; "256x256x32")]
#[test_case([256, 128, 64]; "256x128x64")]
fn two_crews_match_the_interpreter_and_the_uniform_kernel(tile: [usize; 3]) {
    let Some(planner) = super::device_planner().filter(|p| p.device.target.crews >= 2) else {
        eprintln!("skipped: no device with two crews");
        return;
    };
    let device = &planner.device;
    let crews = GemmCfg { tile, stages: 1, warps: [2, 4], group_m: 8, unroll: false, core: Core::Warp, crews: 2 };
    check_gemm(device, 200, 136, 128, crews);
    check_gemm(device, 520, 300, 256, crews);
    let (m, n, k) = (1024, 768, 512);
    let mut seed = 5;
    let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
    let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
    let (a_t, b_t) = (Tensor::from_slice(&a).cast(DType::BFloat16), Tensor::from_slice(&b).cast(DType::BFloat16));
    let bits = |cfg: GemmCfg| -> Vec<u32> {
        let c_t = Tensor::empty(&[m * n], DType::BFloat16);
        let out = graph_launch(plain_gemm(m, n, k, cfg), &cfg.lowering(device), &[&a_t, &b_t, &c_t]).unwrap();
        out.cast(DType::Float32).to_vec::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
    };
    assert!(bits(crews) == bits(GemmCfg { crews: 1, ..crews }), "{crews:?} differs from the uniform kernel");
}

/// A config standing in where a spec's own is replaced.
pub(super) const BIG: GemmCfg =
    GemmCfg { core: Core::Warp, tile: [128, 128, 32], stages: 2, warps: [2, 4], group_m: 8, unroll: true, crews: 1 };

/// The device's result, checked against the interpreter's within the
/// rounding a bf16 store and f32 accumulation allow.
fn check_gemm(device: &Device, m: usize, n: usize, k: usize, cfg: GemmCfg) -> Vec<f32> {
    let prog = plain_gemm(m, n, k, cfg);
    let lowering = cfg.lowering(device);
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
        assert!(diff <= 1.6e-2 * w.abs().max(1.0), "{cfg:?}: c[{}, {}] = {g}, interpreter {w}", i / n, i % n);
    }
    eprintln!("{cfg:?}: max abs diff {worst:.3e}");
    got
}

/// Throughput probe of tk GEMM configs at 4096³ bf16: the configs the
/// tuner admits on the device (the sm_86 measured set on `cp.async`
/// targets) and the tuner's pick; prints TFLOP/s and never asserts (run
/// with `--ignored --nocapture`). tk1's `gemm_nt`, last measured here on
/// the 3060 before its removal: 25.8 TFLOP/s.
#[test]
#[ignore = "perf probe: needs a GPU"]
fn gemm_throughput_probe() {
    let Some(planner) = super::device_planner() else { return };
    let target = &planner.device.target;
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
    // The configs measured on sm_86; elsewhere the tuner's admitted ones.
    let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg: BIG };
    let (admitted, pick) = device_gemms(&planner, &spec);
    let mut cfgs: Vec<GemmCfg> = if target.fill == Fill::CpAsync {
        [
            (128, 64, 32, 2, 2, 2, 8, true),
            (128, 64, 32, 2, 2, 2, 8, false),
            (128, 64, 32, 3, 2, 2, 8, false),
            (128, 128, 32, 3, 2, 4, 8, true),
            (128, 128, 32, 3, 2, 4, 8, false),
            (128, 128, 32, 2, 2, 4, 8, false),
        ]
        .map(|(bm, bn, bk, stages, wr, wc, group_m, unroll)| GemmCfg {
            core: Core::Warp,
            tile: [bm, bn, bk],
            stages,
            warps: [wr, wc],
            group_m,
            unroll,
            crews: 1,
        })
        .to_vec()
    } else {
        admitted
    };
    if !target.group_mma.is_empty() {
        cfgs.extend(
            [
                ([128, 128, 64], [8, 1], 3),
                ([128, 256, 64], [8, 1], 2),
                ([128, 256, 64], [8, 2], 2),
                ([256, 128, 64], [16, 1], 2),
            ]
            .map(|(tile, warps, stages)| GemmCfg {
                tile,
                stages,
                warps,
                group_m: 8,
                unroll: false,
                core: Core::Group,
                crews: 1,
            }),
        );
    }
    // The pick may be a pipeline twin of an admitted tile, or outside the fixed set.
    if !cfgs.contains(&pick) {
        cfgs.push(pick);
    }
    for cfg in cfgs {
        let GemmCfg { tile: [bm, bn, bk], stages, warps: [wr, wc], group_m, unroll, core, crews } = cfg;
        let (prog, lowering) = (plain_gemm(m, n, k, cfg), cfg.lowering(&planner.device));
        let c_t = Tensor::empty(&[m * n], DType::BFloat16);
        let out = graph_launch(prog, &lowering, &[&a_t, &b_t, &c_t]).unwrap();
        let picked = if cfg == pick { " (tuner's pick)" } else { "" };
        plans.push((
            format!(
                "tk {bm}x{bn}x{bk} s{stages} {wr}x{wc} group_m={group_m} unroll={unroll} {core:?} crews={crews}{picked}"
            ),
            out.prepare().unwrap(),
        ));
    }

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
                if let Some(run) = crate::tune::run(plan) {
                    best[i] = best[i].min(run.as_secs_f64());
                }
            }
        }
    }
    for ((label, _), secs) in plans.iter().zip(&best) {
        eprintln!("{label}: {:.2} ms, {:.1} TFLOP/s", secs * 1e3, flops / secs / 1e12);
    }
}

/// Every GEMM candidate the tuner admits on the device timed as the tune
/// store times them, on Nemotron's projections (704 rows under a batch
/// variable of capacity 1) and 4096³, marking the old fixed ladder's pick
/// for the small shapes; prints the first admitted (the untuned pick), the
/// fastest and the tuner's pick per shape.
#[test]
#[ignore = "perf probe: needs a GPU"]
fn gemm_candidates_probe() {
    use crate::kernels::Act;
    let Some(planner) = super::device_planner() else { return };
    let ladder =
        GemmCfg { core: Core::Warp, tile: [128, 64, 32], stages: 2, warps: [2, 2], group_m: 8, unroll: true, crews: 1 };
    let plain = Epilogue::default();
    for (m, n, k, epilogue, var) in [
        (704, 1536, 512, plain, true),
        (704, 512, 512, plain, true),
        (704, 2048, 512, Epilogue { act: Act::Gelu, ..plain }, true),
        (704, 512, 2048, Epilogue { residual: true, ..plain }, true),
        (4096, 4096, 4096, plain, false),
    ] {
        let batch = if var { Batch::Var { name: "b".into(), min: 1, max: 1 } } else { Batch::Static(1) };
        let (cfgs, pick) = device_gemms(&planner, &GemmSpec { m, n, k, batch: batch.clone(), epilogue, cfg: BIG });
        let build = |cfg| {
            let spec = GemmSpec { m, n, k, batch: batch.clone(), epilogue, cfg };
            vec![(gemm::<BF16>(&spec), cfg.lowering(&planner.device))]
        };
        let ns = super::measure(cfgs.iter().map(|&c| build(c)));
        let tflops = |ns: u64| 2.0 * (m * n * k) as f64 / ns as f64 / 1e3;
        eprintln!("== {m}x{n}x{k} {epilogue:?}");
        for (i, (c, t)) in cfgs.iter().zip(&ns).enumerate() {
            let label = match (i, var && *c == ladder, *c == pick) {
                (_, _, true) => " tuner's pick",
                (0, ..) => " first admitted",
                (_, true, _) => " old ladder pick",
                _ => "",
            };
            let t = t.map_or("failed".into(), |t| format!("{:6.1} us {:5.1} TFLOP/s", t as f64 / 1e3, tflops(t)));
            eprintln!("  {:?} s{} {:?} unroll={}: {t}{label}", c.tile, c.stages, c.warps, c.unroll);
        }
        let (t, i) = ns.iter().enumerate().filter_map(|(i, t)| Some((t.as_ref().copied()?, i))).min().unwrap();
        eprintln!(
            "  fastest: {:?} s{} {:?} unroll={} at {:.1} TFLOP/s",
            cfgs[i].tile,
            cfgs[i].stages,
            cfgs[i].warps,
            cfgs[i].unroll,
            tflops(t)
        );
    }
}

/// Host-side cost of a tk GEMM against the graph GEMM: building, lowering
/// (memoized after the first call) and preparing, then the wall time of the
/// first executions of a fresh plan with dynamic and with static shared
/// memory; prints milliseconds and never asserts.
#[test]
#[ignore = "perf probe: needs a cp.async device"]
fn first_execution_probe() {
    let Some(target) = super::cp_async_target() else { return };
    let device = declared(target);
    let (m, n, k) = (704usize, 1536usize, 512usize);
    let mut seed = 5;
    let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
    let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
    let a_t = Tensor::from_slice(&a).cast(DType::BFloat16);
    let b_t = Tensor::from_slice(&b).cast(DType::BFloat16);
    a_t.realize().unwrap();
    b_t.realize().unwrap();
    let mut plans: Vec<(String, svod_runtime::ExecutionPlan)> = vec![];
    for (bm, bn, stages) in [(128, 128, 3), (64, 64, 2)] {
        let cfg = GemmCfg {
            core: Core::Warp,
            tile: [bm, bn, 32],
            stages,
            warps: [2, 2],
            group_m: 8,
            unroll: false,
            crews: 1,
        };
        let c_t = Tensor::empty(&[m * n], DType::BFloat16);
        let out = graph_launch(plain_gemm(m, n, k, cfg), &cfg.lowering(&device), &[&a_t, &b_t, &c_t]).unwrap();
        plans.push((format!("tk {bm}x{bn} s{stages}"), out.prepare().unwrap()));
    }
    let a2 = a_t.try_reshape([m, k]).unwrap();
    let b2 = b_t.try_reshape([n, k]).unwrap();
    let graph = a2.matmul(&b2.try_transpose(0, 1).unwrap()).unwrap();
    plans.push(("graph matmul".to_string(), graph.prepare().unwrap()));
    // Host-side cost: building, lowering and scheduling the pre-linearized body.
    let cfg =
        GemmCfg { core: Core::Warp, tile: [64, 64, 32], stages: 2, warps: [2, 2], group_m: 8, unroll: false, crews: 1 };
    for i in 0..3 {
        let c_t = Tensor::empty(&[m * n], DType::BFloat16);
        let t = std::time::Instant::now();
        let prog = plain_gemm(m, n, k, cfg);
        let lowering = cfg.lowering(&device);
        let built = t.elapsed();
        let text = format!("{prog:?}{lowering:?}");
        let formatted = t.elapsed();
        let out = graph_launch(prog, &lowering, &[&a_t, &b_t, &c_t]).unwrap();
        let launched = t.elapsed();
        let _plan = out.prepare().unwrap();
        let prepared = t.elapsed();
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        eprintln!(
            "tk #{i}: build {:.2} ms, debug text {:.2} ms ({} bytes), lower + bind {:.2} ms, prepare {:.2} ms",
            ms(built),
            ms(formatted - built),
            text.len(),
            ms(launched - formatted),
            ms(prepared - launched)
        );
    }
    for i in 0..3 {
        let t = std::time::Instant::now();
        let graph = a2.matmul(&b2.try_transpose(0, 1).unwrap()).unwrap();
        let _plan = graph.prepare().unwrap();
        eprintln!("graph build + prepare #{i}: {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    for (label, plan) in &plans {
        let mut times = vec![];
        for _ in 0..4 {
            let t = std::time::Instant::now();
            plan.execute().unwrap();
            let mut bytes = [0u8; 2];
            plan.output_buffer().unwrap().copyout_prefix(&mut bytes).unwrap();
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!("{label}: {}", times.iter().map(|t| format!("{t:.2} ms")).collect::<Vec<_>>().join(", "));
    }
}

/// Qwen3-Embedding-0.6B's four projections at batch 8 × 512 (M 4096), bf16:
/// the planner's shortlist, then the rest of a lattice of configs that pass
/// the target's feasibility predicates, timed as the tune store times them;
/// prints each and the best of either set. tk1's in-model means on gfx1201
/// (`origin/main`, 2026-10-09) sit beside each shape, so a lattice win says
/// the shortlist is the gap and a tie says it is not.
#[test]
#[ignore = "perf probe: needs a GPU"]
fn qwen3_gemm_candidates_probe() {
    use crate::kernels::Act;
    let Some(planner) = super::device_planner() else { return };
    let target = &planner.device.target;
    let plain = Epilogue::default();
    let shapes = [
        ("qkv", 4096, 4096, 1024, plain, 270.4),
        ("o_proj+residual", 4096, 1024, 2048, Epilogue { residual: true, ..plain }, 152.4),
        ("gate_up swiglu", 4096, 3072, 1024, Epilogue { act: Act::Silu, gated: true, ..plain }, 436.4),
        ("down+residual", 4096, 1024, 3072, Epilogue { residual: true, ..plain }, 196.6),
    ];
    for (name, m, n, k, epilogue, tk1_us) in shapes {
        let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue, cfg: BIG };
        let shortlist = device_gemms(&planner, &spec).0;
        let mut lattice = vec![];
        for bm in [64, 128, 256] {
            for bn in [64, 128, 256] {
                for bk in [16, 32, 64] {
                    for warps in [[2, 2], [2, 4], [4, 2], [4, 4]] {
                        for unroll in [false, true] {
                            let cfg = GemmCfg {
                                core: Core::Warp,
                                tile: [bm, bn, bk],
                                stages: 2,
                                warps,
                                group_m: 8,
                                unroll,
                                crews: 1,
                            };
                            let fits = cfg.fits(target) && cfg.smem_bytes(epilogue.gated) <= target.smem_bytes;
                            if fits && !shortlist.contains(&cfg) {
                                lattice.push(cfg);
                            }
                        }
                    }
                }
            }
        }
        let n_eff = if epilogue.gated { 2 * n } else { n };
        let tflops = |ns: u64| 2.0 * (m * n_eff * k) as f64 / ns as f64 / 1e3;
        let build = |cfg: GemmCfg| {
            let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue, cfg };
            vec![(gemm::<BF16>(&spec), cfg.lowering(&planner.device))]
        };
        let show = |set: &str, cfgs: &[GemmCfg], ns: &[Option<u64>]| -> Option<(u64, GemmCfg)> {
            let mut best: Option<(u64, GemmCfg)> = None;
            for (c, t) in cfgs.iter().zip(ns) {
                let t = match t {
                    Some(t) => {
                        if best.is_none_or(|(b, _)| *t < b) {
                            best = Some((*t, *c));
                        }
                        format!("{:6.1} us {:5.1} TFLOP/s", *t as f64 / 1e3, tflops(*t))
                    }
                    None => "failed".to_string(),
                };
                eprintln!("  {set:7} {:?} s{} {:?} unroll={}: {t}", c.tile, c.stages, c.warps, c.unroll);
            }
            best
        };
        eprintln!(
            "== {name}: {m}x{n}x{k} gated={} ({} shortlist, {} lattice); tk1 in-model {tk1_us} us",
            epilogue.gated,
            shortlist.len(),
            lattice.len()
        );
        let s_best = show("short", &shortlist, &super::measure(shortlist.iter().map(|&c| build(c))));
        let l_best = show("lattice", &lattice, &super::measure(lattice.iter().map(|&c| build(c))));
        let fmt = |b: Option<(u64, GemmCfg)>| {
            b.map_or("none".to_string(), |(t, c)| {
                format!(
                    "{:.1} us {:.1} TFLOP/s {:?} {:?} unroll={}",
                    t as f64 / 1e3,
                    tflops(t),
                    c.tile,
                    c.warps,
                    c.unroll
                )
            })
        };
        eprintln!("{name}: shortlist best {} | lattice best {} | tk1 {tk1_us} us", fmt(s_best), fmt(l_best));
    }
}

/// A transposing load of a row-major 16×16 tile, its lanes addressing the
/// rows in `global_tr_rows` order, lands every element where `global_tr_b128`
/// says. The lane mapping of the instruction is measured here, not quoted.
#[test]
fn transposing_loads_land_as_their_layout() {
    use crate::index::{access, add, c32, mul, store_at};
    use crate::layout::Dim::{Lane, Reg};
    use crate::layout::{global_tr_b128, global_tr_rows};
    use svod_codegen::llvm::amd::tr::global_load_tr_b128;
    use svod_ir::{CallInfo, KernelInfo, UOp};
    if !super::device_target().is_some_and(|t| t.tr_load) {
        eprintln!("skipped: no transposing loads");
        return;
    }
    let device = default_device();
    let input = Tensor::from_slice((0..256).map(|i| i as f32).collect::<Vec<_>>()).cast(DType::BFloat16);
    let out = Tensor::from_slice(vec![0f32; 256]).cast(DType::BFloat16);
    let name = "tr_probe".to_string();
    let outs = out
        .custom_kernel_with(&[&input], CallInfo { name: Some(name.clone()), ..CallInfo::default() }, |ph| {
            let (out, inp) = (ph[0].base(), ph[1].base());
            let tid = UOp::special_dtype(c32(32), "lidx0".to_string(), DType::Int32);
            // Row `L % 8 + 8·(L/16)` at column `8·((L/8) % 2)`.
            let row = add(&tid.try_cmod(&c32(8)).unwrap(), &mul(&tid.try_cdiv(&c32(16)).unwrap(), &c32(8)));
            let col = mul(&tid.try_cdiv(&c32(8)).unwrap().try_cmod(&c32(2)).unwrap(), &c32(8));
            let at = add(&mul(&row, &c32(16)), &col);
            let elems = global_load_tr_b128(&access(&inp, &at, 1), ScalarDType::BFloat16);
            let mut list = vec![tid.clone()];
            for (i, e) in elems.iter().enumerate() {
                let o = access(&out, &add(&mul(&tid, &c32(8)), &c32(i as i64)), 1);
                list.push(store_at(&o, vec![e.clone()]));
            }
            UOp::linear_program(KernelInfo { name: Some(name.clone()), ..Default::default() }, list, device.clone())
                .unwrap()
        })
        .unwrap();
    let t = &outs[0];
    t.prepare().unwrap().execute_with_vars(&[]).unwrap();
    let mut bytes = vec![0u8; 512];
    t.buffer().unwrap().copyout(&mut bytes).unwrap();
    let held: Vec<u32> = super::decode(ScalarDType::BFloat16, &bytes).into_iter().map(|x| x as u32).collect();
    let (l, rows) = (global_tr_b128(), global_tr_rows());
    for lane in 0..32u32 {
        for j in 0..8u32 {
            let (r, c) = l.apply(&[(Lane, lane), (Reg, j)]);
            assert_eq!(held[(lane * 8 + j) as usize], r * 16 + c, "lane {lane} register {j}");
        }
        assert_eq!(rows.apply(&[(Lane, lane)]), (lane % 8 + 8 * (lane / 16), 8 * (lane / 8 % 2)));
    }
}
