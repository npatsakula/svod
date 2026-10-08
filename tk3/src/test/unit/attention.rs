//! Flash attention: the program against a direct softmax reference on the
//! host, and the lowered kernel against the program on the GPU.

use svod_dtype::{DType, DeviceSpec, ScalarDType, default_device::default_device};
use svod_tensor::Tensor;
use test_case::test_case;

use super::programs::{FaSpec, flash_attention};
use crate::atoms::Target;
use crate::interp::{round_to, run};
use crate::launch::graph_launch;
use crate::layouts::WarpGrid;
use crate::lower::Lowering;
use crate::schedule::{Prefetch, Schedule};

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
}

struct Case {
    spec: FaSpec,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    lens: Vec<i64>,
}

fn case(spec: FaSpec, lens: &[i64]) -> Case {
    let mut seed = 5;
    let n = |elems: usize, seed: &mut u64| -> Vec<f64> {
        (0..elems).map(|_| round_to(ScalarDType::BFloat16, lcg(seed))).collect()
    };
    let (q, k, v) = (
        n(spec.batch * spec.t * spec.heads * spec.d, &mut seed),
        n(spec.batch * spec.tk * spec.heads * spec.d, &mut seed),
        n(spec.batch * spec.tk * spec.heads * spec.d, &mut seed),
    );
    Case { spec, q, k, v, lens: lens.to_vec() }
}

/// Direct attention in f64 with the same masks.
fn reference(c: &Case) -> Vec<f64> {
    let s = c.spec;
    let stride = s.heads * s.d;
    let mut out = vec![0.0; s.batch * s.t * stride];
    for b in 0..s.batch {
        let len = if s.key_lens { c.lens[b].max(1) as usize } else { s.tk };
        for h in 0..s.heads {
            for i in 0..s.t {
                let qi = &c.q[(b * s.t + i) * stride + h * s.d..][..s.d];
                let scores: Vec<f64> = (0..s.tk)
                    .map(|j| {
                        if j >= len || (s.causal && j > i) {
                            return f64::NEG_INFINITY;
                        }
                        let kj = &c.k[(b * s.tk + j) * stride + h * s.d..][..s.d];
                        qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f64>() * s.scale as f64
                    })
                    .collect();
                let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = scores.iter().map(|x| (x - mx).exp()).collect();
                let l: f64 = w.iter().sum();
                for dd in 0..s.d {
                    let acc: f64 = (0..s.tk).map(|j| w[j] * c.v[(b * s.tk + j) * stride + h * s.d + dd]).sum();
                    out[(b * s.t + i) * stride + h * s.d + dd] = acc / l;
                }
            }
        }
    }
    out
}

fn params(c: &Case) -> Vec<Vec<f64>> {
    let s = c.spec;
    let mut p = vec![c.q.clone(), c.k.clone(), c.v.clone(), vec![0.0; s.batch * s.t * s.heads * s.d]];
    if s.key_lens {
        p.push(c.lens.iter().map(|&l| l as f64).collect());
    }
    p
}

fn spec(t: usize, tk: usize, d: usize, bq: usize, bkv: usize, causal: bool, key_lens: bool) -> FaSpec {
    FaSpec { batch: 2, t, tk, heads: 2, d, bq, bkv, stages: 2, causal, key_lens, scale: 1.0 / (d as f32).sqrt() }
}

#[test_case(spec(64, 64, 64, 64, 64, false, false), &[64, 64]; "one block")]
#[test_case(spec(128, 192, 64, 64, 64, false, false), &[192, 192]; "three key blocks")]
#[test_case(spec(128, 128, 64, 64, 64, true, false), &[128, 128]; "causal")]
#[test_case(spec(128, 128, 64, 64, 64, false, true), &[100, 7]; "key lengths")]
#[test_case(spec(128, 128, 64, 64, 32, true, true), &[128, 50]; "causal with lengths, narrow kv")]
fn program_matches_a_direct_softmax(spec: FaSpec, lens: &[i64]) {
    let c = case(spec, lens);
    let got = run(&flash_attention(spec), params(&c), &[("b", spec.batch as i64)]).unwrap();
    let want = reference(&c);
    let mut worst = 0.0f64;
    for (g, w) in got[3].iter().zip(&want) {
        worst = worst.max((g - w).abs());
    }
    assert!(worst < 2e-2, "max abs diff {worst}");
}

/// The live batch bound: rows of a batch past it are never written.
#[test]
fn only_the_live_batch_runs() {
    let spec = spec(64, 64, 64, 64, 64, false, false);
    let c = case(spec, &[64, 64]);
    let got = run(&flash_attention(spec), params(&c), &[("b", 1)]).unwrap();
    let half = spec.t * spec.heads * spec.d;
    assert!(got[3][..half].iter().any(|x| *x != 0.0));
    assert!(got[3][half..].iter().all(|x| *x == 0.0));
}

#[test_case(spec(64, 64, 64, 64, 64, false, false), &[64, 64]; "one key block")]
#[test_case(spec(128, 128, 64, 64, 64, false, false), &[128, 128]; "plain")]
#[test_case(spec(256, 256, 64, 64, 64, true, false), &[256, 256]; "causal")]
#[test_case(spec(128, 256, 64, 64, 64, false, true), &[200, 33]; "key lengths")]
#[test_case(spec(128, 128, 128, 64, 32, true, true), &[128, 64]; "d 128, causal with lengths")]
fn kernel_matches_the_program(spec: FaSpec, lens: &[i64]) {
    let device = default_device();
    let Some(target) = matches!(device, DeviceSpec::Cuda { .. }).then(|| Target::for_device(&device)).flatten() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let c = case(spec, lens);
    let want = run(&flash_attention(spec), params(&c), &[("b", spec.batch as i64)]).unwrap();
    let lowering = Lowering {
        target,
        schedule: Schedule::Uniform { prefetch: Prefetch::CpAsync, unroll: false },
        grid: WarpGrid { rows: (spec.bq / 16) as u32, cols: 1 },
        swizzle: true,
    };
    let to_bf16 = |v: &[f64]| Tensor::from_slice(v.iter().map(|&x| x as f32).collect::<Vec<_>>()).cast(DType::BFloat16);
    let (q, k, v) = (to_bf16(&c.q), to_bf16(&c.k), to_bf16(&c.v));
    let o = Tensor::empty(&[spec.batch * spec.t * spec.heads * spec.d], DType::BFloat16);
    let lens_t = Tensor::from_slice(c.lens.iter().map(|&l| l as i32).collect::<Vec<_>>());
    let mut tensors = vec![&q, &k, &v, &o];
    if spec.key_lens {
        tensors.push(&lens_t);
    }
    let out = graph_launch(flash_attention(spec), &lowering, &tensors).unwrap();
    let mut plan = out.prepare().unwrap();
    plan.execute_with_vars(&[("b", spec.batch as i64)]).unwrap();
    let mut bytes = vec![0u8; spec.batch * spec.t * spec.heads * spec.d * 2];
    out.buffer().unwrap().copyout(&mut bytes).unwrap();
    let got: Vec<f32> =
        bytes.chunks(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect();
    let reference = reference(&c);
    let mut worst = 0.0f64;
    let (mut over, mut worst_at) = (0usize, 0usize);
    for (i, (g, w)) in got.iter().zip(&want[3]).enumerate() {
        let d = (*g as f64 - w).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
        over += usize::from(d > 1e-2);
    }
    let vs_ref = got.iter().zip(&reference).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max);
    let stride = spec.heads * spec.d;
    eprintln!(
        "max abs diff {worst:.3e} at batch {} row {} head {} col {} ({over} elements over 1e-2); vs f64 reference {vs_ref:.3e}",
        worst_at / (spec.t * stride),
        worst_at / stride % spec.t,
        worst_at % stride / spec.d,
        worst_at % spec.d
    );
    assert!(worst < 2e-2, "max abs diff {worst}");
}

/// Throughput against tk1's flash attention on the same device: causal and
/// plain, head dim 64 and 128 (run with `--ignored --nocapture --release`).
#[test]
#[ignore = "perf probe: needs a CUDA device"]
fn attention_throughput_probe() {
    let device = default_device();
    let Some(target) = matches!(device, DeviceSpec::Cuda { .. }).then(|| Target::for_device(&device)).flatten() else {
        return;
    };
    let (batch, heads, t) = (4usize, 8usize, 2048usize);
    for (d, causal, bq, bkv) in [(64, false, 64, 64), (64, true, 64, 64), (128, false, 64, 32), (128, true, 64, 32)] {
        let spec = FaSpec { batch, t, tk: t, heads, d, bq, bkv, stages: 2, causal, key_lens: false, scale: 1.0 / (d as f32).sqrt() };
        let c = case(spec, &[]);
        let to_bf16 = |v: &[f64]| Tensor::from_slice(v.iter().map(|&x| x as f32).collect::<Vec<_>>()).cast(DType::BFloat16);
        let (q, k, v) = (to_bf16(&c.q), to_bf16(&c.k), to_bf16(&c.v));
        for x in [&q, &k, &v] {
            x.realize().unwrap();
        }
        let flops = 4.0 * batch as f64 * heads as f64 * t as f64 * t as f64 * d as f64 / if causal { 2.0 } else { 1.0 };
        let mut plans: Vec<(String, svod_runtime::ExecutionPlan)> = vec![];
        for warps_rows in [4u32] {
            let lowering = Lowering {
                target: target.clone(),
                schedule: Schedule::Uniform { prefetch: Prefetch::CpAsync, unroll: false },
                grid: WarpGrid { rows: warps_rows, cols: 1 },
                swizzle: true,
            };
            let o = Tensor::empty(&[batch * t * heads * d], DType::BFloat16);
            let out = graph_launch(flash_attention(spec), &lowering, &[&q, &k, &v, &o]).unwrap();
            let mut plan = out.prepare().unwrap();
            plan.execute_with_vars(&[("b", batch as i64)]).unwrap();
            plans.push((format!("tk3 d{d} causal={causal} bq{bq} bkv{bkv}"), plan));
        }
        let shape = [batch, t, heads, d];
        let (q4, k4, v4) = (q.try_reshape(shape).unwrap(), k.try_reshape(shape).unwrap(), v.try_reshape(shape).unwrap());
        let opts = svod_tk::FaOpts { causal, ..Default::default() };
        if let Ok(Some(tk1)) = svod_tk::flash_attention_with(&q4, &k4, &v4, opts) {
            plans.push((format!("tk1 d{d} causal={causal}"), tk1.prepare().unwrap()));
        }
        let mut best = vec![f64::INFINITY; plans.len()];
        for _ in 0..4 {
            for (i, (_, plan)) in plans.iter().enumerate() {
                for _ in 0..5 {
                    for kp in plan.execute_profiled().unwrap() {
                        if let (Some(s), Some(e)) = (kp.gpu_start_ns, kp.gpu_end_ns) {
                            best[i] = best[i].min((e - s) as f64 * 1e-9);
                        }
                    }
                }
            }
        }
        for ((label, _), secs) in plans.iter().zip(&best) {
            eprintln!("{label}: {:.3} ms, {:.1} TFLOP/s", secs * 1e3, flops / secs / 1e12);
        }
    }
}
