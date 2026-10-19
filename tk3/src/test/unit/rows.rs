//! GEMM epilogues and row kernels: interpreter against f64 references on the
//! host, and the lowered programs against both on a CUDA device (skipped
//! unless `SVOD_DEVICE` names one).

use proptest::prelude::*;
use svod_dtype::{DType, DeviceSpec, ScalarDType, default_device::default_device};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::atoms::{Target, sm86};
use crate::build::BF16;
use crate::interp::{round_to, run};
use crate::ir::*;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::kernels::rows::{Norm, NormCfg, NormSpec, norm};
use crate::kernels::{Act, Batch};
use crate::launch::graph_launch_all;
use crate::layouts::{WarpGrid, infer};
use crate::lower::Lowering;

fn gemm_nt_epilogue(
    m: usize,
    n: usize,
    k: usize,
    tile: [usize; 3],
    stages: usize,
    warps: [u32; 2],
    epi: Epilogue,
) -> Program {
    let cfg = GemmCfg { tile, stages, warps, group_m: 0, unroll: true };
    gemm::<BF16>(&GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: epi, cfg })
}

fn norm_rows(norm_: Norm, rows: usize, d: usize, br: usize, eps: f64, residual: bool) -> Program {
    norm::<BF16>(&NormSpec { norm: norm_, rows, d, batch: Batch::Static(1), eps, residual, cfg: NormCfg { br } })
}

fn cuda_target() -> Option<Target> {
    let spec = default_device();
    matches!(spec, DeviceSpec::Cuda { .. }).then(|| Target::for_device(&spec)).flatten()
}

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
}

fn bf16s(n: usize, seed: &mut u64, f: impl Fn(f64) -> f64) -> Vec<f64> {
    (0..n).map(|_| round_to(ScalarDType::BFloat16, f(lcg(seed)))).collect()
}

/// `|got - want| ≤ tol·max(|want|, 1)` everywhere; returns the max abs diff.
fn assert_close(what: &str, got: &[f64], want: &[f64], tol: f64) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let diff = (g - w).abs();
        worst = worst.max(diff);
        assert!(diff <= tol * w.abs().max(1.0), "{what}[{i}] = {g}, want {w}");
    }
    worst
}

fn upload(params: &[Vec<f64>]) -> Vec<Tensor> {
    params
        .iter()
        .map(|p| Tensor::from_slice(p.iter().map(|&x| x as f32).collect::<Vec<_>>()).cast(DType::BFloat16))
        .collect()
}

/// Upload the bf16 parameters, run on the device, read every parameter back.
fn on_device(prog: Program, lowering: &Lowering, params: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let tensors = upload(params);
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let outs = graph_launch_all(prog, lowering, &refs).unwrap();
    outs.iter().map(|t| t.cast(DType::Float32).to_vec::<f32>().unwrap().into_iter().map(f64::from).collect()).collect()
}

// ---- GEMM epilogues ---------------------------------------------------------

fn gemm_inputs(m: usize, n: usize, k: usize, epi: Epilogue) -> Vec<Vec<f64>> {
    let halves = if epi.gated { 2 } else { 1 };
    let mut seed = 17;
    let mut params = vec![bf16s(m * k, &mut seed, |x| x), bf16s(halves * n * k, &mut seed, |x| x)];
    if epi.bias {
        params.push(bf16s(halves * n, &mut seed, |x| 2.0 * x));
    }
    if epi.residual {
        params.push(bf16s(m * n, &mut seed, |x| 4.0 * x));
    }
    params.push(vec![0.0; m * n]);
    params
}

fn gemm_reference(m: usize, n: usize, k: usize, epi: Epilogue, p: &[Vec<f64>]) -> Vec<f64> {
    let (a, b) = (&p[0], &p[1]);
    let bias = epi.bias.then(|| &p[2]);
    let residual = epi.residual.then(|| &p[2 + usize::from(epi.bias)]);
    let silu = |x: f64| x / (1.0 + (-x).exp());
    let half = |i: usize, j: usize, h: usize| {
        let row = h * n + j;
        let dot: f64 = (0..k).map(|kk| a[i * k + kk] * b[row * k + kk]).sum();
        dot + bias.map_or(0.0, |v| v[row])
    };
    (0..m * n)
        .map(|e| {
            let (i, j) = (e / n, e % n);
            let x = half(i, j, 0);
            let y = match epi.act {
                Act::None => x,
                Act::Gelu => 0.5 * x * (1.0 + libm::erf(x * std::f64::consts::FRAC_1_SQRT_2)),
                Act::Silu => silu(x),
            };
            let y = if epi.gated { y * half(i, j, 1) } else { y };
            round_to(ScalarDType::BFloat16, y + residual.map_or(0.0, |r| r[e]))
        })
        .collect()
}

const EPILOGUES: [Epilogue; 6] = [
    Epilogue { bias: true, residual: false, act: Act::None, gated: false },
    Epilogue { bias: false, residual: true, act: Act::None, gated: false },
    Epilogue { bias: true, residual: false, act: Act::Gelu, gated: false },
    Epilogue { bias: true, residual: true, act: Act::Silu, gated: false },
    Epilogue { bias: true, residual: false, act: Act::Silu, gated: true },
    Epilogue { bias: false, residual: true, act: Act::Gelu, gated: true },
];

/// The interpreter applies every epilogue as the f64 reference does.
#[test_case(0; "bias")]
#[test_case(1; "residual")]
#[test_case(2; "bias gelu")]
#[test_case(3; "bias silu residual")]
#[test_case(4; "bias swiglu")]
#[test_case(5; "geglu residual")]
fn epilogue_interpreter_matches_the_reference(which: usize) {
    let (m, n, k, epi) = (64, 64, 64, EPILOGUES[which]);
    let prog = gemm_nt_epilogue(m, n, k, [32, 32, 16], 2, [2, 2], epi);
    let params = gemm_inputs(m, n, k, epi);
    let want = gemm_reference(m, n, k, epi, &params);
    let out = run(&prog, params, &[]).unwrap();
    assert_close("c", out.last().unwrap(), &want, 1.6e-2);
}

/// A `[1, n]` bias loaded from global memory is held as the column vector of
/// the accumulator's C layout, so the broadcast add reads it in place.
#[test]
fn bias_vector_takes_the_accumulators_column_layout() {
    let epi = Epilogue { bias: true, ..Epilogue::default() };
    let mut prog = gemm_nt_epilogue(128, 128, 64, [128, 128, 32], 2, [2, 4], epi);
    let lay = infer(&mut prog, &sm86(), WarpGrid { rows: 2, cols: 4 }).unwrap();
    let add = prog
        .walk()
        .find_map(|(_, s)| match s {
            Stmt::Let { dst, op: TileOp::Binary { a, b, f: BinaryOp::Add } } => Some((*dst, *a, *b)),
            _ => None,
        })
        .expect("the bias add");
    let (acc, bias) = (lay[add.1.index()].clone().unwrap(), lay[add.2.index()].clone().unwrap());
    assert_eq!(prog.value(add.2).shape, Shape::new(1, 128));
    assert_eq!(bias, acc.col_vector());
    assert_eq!(lay[add.0.index()], Some(acc));
}

/// Every epilogue on the device matches the interpreter and the f64
/// reference, on one block and on a 256³ grid of blocks.
#[test_case(64, 64, 64, [64, 64, 32], 2, 2, 2; "small")]
#[test_case(256, 256, 256, [128, 64, 32], 2, 2, 2; "256 cubed")]
fn epilogues_match_on_device(m: usize, n: usize, k: usize, tile: [usize; 3], stages: usize, wr: u32, wc: u32) {
    let Some(target) = cuda_target() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    for epi in EPILOGUES {
        let prog = gemm_nt_epilogue(m, n, k, tile, stages, [wr, wc], epi);
        let params = gemm_inputs(m, n, k, epi);
        let want = gemm_reference(m, n, k, epi, &params);
        let interp = run(&prog, params.clone(), &[]).unwrap();
        let cfg = GemmCfg { tile, stages, warps: [wr, wc], group_m: 0, unroll: true };
        let got = on_device(prog, &cfg.lowering(target.clone()), &params);
        let (got, interp) = (got.last().unwrap(), interp.last().unwrap());
        let d_interp = assert_close("c vs interpreter", got, interp, 1.6e-2);
        let d_ref = assert_close("c vs reference", got, &want, 1.6e-2);
        eprintln!("{m}x{n}x{k} {epi:?}: max abs diff {d_interp:.3e} (interpreter), {d_ref:.3e} (f64)");
    }
}

// ---- row kernels -------------------------------------------------------------

fn norm_inputs(norm: Norm, rows: usize, d: usize, residual: bool, seed: u64) -> Vec<Vec<f64>> {
    let mut seed = seed;
    // An offset per element keeps the mean away from zero, so centering matters.
    let mut params = vec![bf16s(rows * d, &mut seed, |x| 2.0 * x + 1.5)];
    if residual {
        params.push(bf16s(rows * d, &mut seed, |x| x));
    }
    params.push(bf16s(d, &mut seed, |x| 1.0 + 0.5 * x));
    if norm == Norm::Layer {
        params.push(bf16s(d, &mut seed, |x| 0.5 * x));
    }
    params.push(vec![0.0; rows * d]);
    if residual {
        params.push(vec![0.0; rows * d]);
    }
    params
}

/// `(out, sum)` in f64 from the bf16 inputs; the sum is rounded to bf16
/// before it is normalized, as the kernel stores and reads it.
fn norm_reference(norm: Norm, rows: usize, d: usize, eps: f64, residual: bool, p: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let at = |i| &p[i];
    let x = at(0);
    let r = residual.then(|| at(1));
    let w = at(1 + usize::from(residual));
    let b = (norm == Norm::Layer).then(|| at(2 + usize::from(residual)));
    let sum: Vec<f64> =
        (0..rows * d).map(|e| round_to(ScalarDType::BFloat16, x[e] + r.map_or(0.0, |r| r[e]))).collect();
    let mut out = vec![0.0; rows * d];
    for i in 0..rows {
        let row = &sum[i * d..(i + 1) * d];
        let mean = if norm == Norm::Layer { row.iter().sum::<f64>() / d as f64 } else { 0.0 };
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / d as f64;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..d {
            let y = (row[c] - mean) * inv * w[c] + b.map_or(0.0, |b| b[c]);
            out[i * d + c] = round_to(ScalarDType::BFloat16, y);
        }
    }
    (out, sum)
}

/// Indices of `out` and `sum` among the parameters.
fn norm_outputs(norm: Norm, residual: bool) -> (usize, usize) {
    let out = 2 + usize::from(residual) + usize::from(norm == Norm::Layer);
    (out, out + 1)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    /// Any row count, including a partial last block, normalizes every row
    /// and leaves nothing past the end touched.
    #[test]
    fn norm_interpreter_matches_the_reference(
        rows in 1usize..20,
        layer in any::<bool>(),
        residual in any::<bool>(),
        seed in any::<u64>(),
    ) {
        let (norm, d, eps) = (if layer { Norm::Layer } else { Norm::Rms }, 256, 1e-5);
        let prog = norm_rows(norm, rows, d, 4, eps, residual);
        let params = norm_inputs(norm, rows, d, residual, seed);
        let (want, want_sum) = norm_reference(norm, rows, d, eps, residual, &params);
        let out = run(&prog, params, &[]).unwrap();
        let (o, s) = norm_outputs(norm, residual);
        assert_close("out", &out[o], &want, 1e-2);
        if residual {
            assert_close("sum", &out[s], &want_sum, 0.0);
        }
    }
}

/// LayerNorm and RMSNorm on the device match the interpreter and the f64
/// reference, with and without the fused residual, on a partial last block.
#[test_case(Norm::Layer, 512, false, 4; "layer 512")]
#[test_case(Norm::Layer, 1024, true, 4; "layer 1024 residual")]
#[test_case(Norm::Rms, 512, true, 4; "rms 512 residual")]
#[test_case(Norm::Rms, 1024, false, 4; "rms 1024")]
#[test_case(Norm::Layer, 512, true, 8; "layer 512 residual, 8 rows a block")]
#[test_case(Norm::Rms, 2048, false, 16; "rms 2048, 16 rows a block")]
fn norms_match_on_device(norm: Norm, d: usize, residual: bool, br: usize) {
    let Some(target) = cuda_target() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let (rows, eps) = (37, 1e-5);
    let prog = norm_rows(norm, rows, d, br, eps, residual);
    let params = norm_inputs(norm, rows, d, residual, 5);
    let (want, want_sum) = norm_reference(norm, rows, d, eps, residual, &params);
    let interp = run(&prog, params.clone(), &[]).unwrap();
    let got = on_device(prog, &row_lowering(target), &params);
    let (o, s) = norm_outputs(norm, residual);
    let d_interp = assert_close("out vs interpreter", &got[o], &interp[o], 1e-2);
    let d_ref = assert_close("out vs reference", &got[o], &want, 1e-2);
    if residual {
        assert_close("sum", &got[s], &want_sum, 0.0);
    }
    eprintln!("{norm:?} d={d} residual={residual}: max abs diff {d_interp:.3e} (interpreter), {d_ref:.3e} (f64)");
}

fn row_lowering(target: Target) -> Lowering {
    NormCfg { br: 4 }.lowering(target)
}

/// Fastest GPU time of `plan`'s kernels after a warm-up long enough for the
/// memory clock to leave idle.
fn gpu_seconds(plan: &svod_runtime::ExecutionPlan) -> f64 {
    let start = std::time::Instant::now();
    while start.elapsed().as_millis() < 500 {
        plan.execute().unwrap();
    }
    let mut best = f64::INFINITY;
    for _ in 0..50 {
        for kp in plan.execute_profiled().unwrap() {
            if let (Some(s), Some(e)) = (kp.gpu_start_ns, kp.gpu_end_ns) {
                best = best.min((e - s) as f64 * 1e-9);
            }
        }
    }
    best
}

/// Bandwidth of the row kernels on 8192×1024 bf16 against a graph `x + x`
/// moving as many bytes; prints GB/s and never asserts (run with
/// `--ignored --nocapture`).
#[test]
#[ignore = "perf probe: needs a CUDA device"]
fn norm_bandwidth_probe() {
    let Some(target) = cuda_target() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let (rows, d, eps) = (8192, 1024, 1e-5);
    let gbs = |bytes: f64, secs: f64| bytes / secs / 1e9;
    for (norm, residual) in [(Norm::Rms, false), (Norm::Layer, false), (Norm::Rms, true), (Norm::Layer, true)] {
        let params = norm_inputs(norm, rows, d, residual, 9);
        let tensors = upload(&params);
        for t in &tensors {
            t.realize().unwrap();
        }
        let refs: Vec<&Tensor> = tensors.iter().collect();
        // x (+ residual) in, out (+ sum) out; the weight vectors are noise.
        let bytes = (2 * rows * d * if residual { 2 } else { 1 }) as f64 * 2.0;
        let baseline = gpu_seconds(&(&tensors[0] + &tensors[0]).unwrap().prepare().unwrap());
        eprintln!("graph x + x: {:.1} us, {:.0} GB/s", baseline * 1e6, gbs((4 * rows * d) as f64, baseline));
        for br in [4, 8, 16] {
            let prog = norm_rows(norm, rows, d, br, eps, residual);
            let (o, _) = norm_outputs(norm, residual);
            let plan = graph_launch_all(prog, &row_lowering(target.clone()), &refs).unwrap()[o].prepare().unwrap();
            let secs = gpu_seconds(&plan);
            eprintln!("{norm:?} residual={residual} br={br}: {:.1} us, {:.0} GB/s", secs * 1e6, gbs(bytes, secs));
        }
    }
}
