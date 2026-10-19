//! The mechanisms flash attention composes, each alone on the GPU against the
//! interpreter: a global-resident A operand, the transposed `ldmatrix` B
//! operand, row reductions and vector broadcasts.

use svod_dtype::{DType, DeviceSpec, ScalarDType, default_device::default_device};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::atoms::Target;
use crate::build::*;
use crate::interp::{round_to, run};
use crate::ir::*;
use crate::launch::graph_launch;
use crate::layouts::WarpGrid;
use crate::lower::Lowering;
use crate::schedule::{Prefetch, Schedule};

fn target() -> Option<Target> {
    let device = default_device();
    matches!(device, DeviceSpec::Cuda { .. }).then(|| Target::for_device(&device)).flatten()
}

fn data(n: usize, seed: &mut u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            round_to(ScalarDType::BFloat16, ((*seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0)
        })
        .collect()
}

fn check(prog: Program, inputs: Vec<Vec<f64>>, out_elems: usize, warps: u32, tol: f64) {
    let Some(target) = target() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let mut params = inputs.clone();
    params.push(vec![0.0; out_elems]);
    let want = run(&prog, params, &[]).unwrap();
    let lowering = Lowering {
        target,
        schedule: Schedule::Uniform { prefetch: Prefetch::CpAsync, unroll: false },
        grid: WarpGrid { rows: warps, cols: 1 },
        swizzle: true,
    };
    let tensors: Vec<Tensor> = inputs
        .iter()
        .map(|v| Tensor::from_slice(v.iter().map(|&x| x as f32).collect::<Vec<_>>()).cast(DType::BFloat16))
        .collect();
    let out = Tensor::empty(&[out_elems], DType::Float32);
    let mut refs: Vec<&Tensor> = tensors.iter().collect();
    refs.push(&out);
    let got: Vec<f32> = graph_launch(prog, &lowering, &refs).unwrap().to_vec::<f32>().unwrap();
    let worst = got.iter().zip(want.last().unwrap()).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max);
    assert!(worst < tol, "max abs diff {worst}");
}

/// `s = q · kᵀ` with Q loaded straight from global memory into the A layout
/// and K through shared memory.
#[test_case(64, 64, 64; "square")]
#[test_case(64, 32, 128; "d 128, narrow k")]
fn qk_from_registers_and_shared(bq: usize, bkv: usize, d: usize) {
    let mut k = Kernel::new("qk");
    let q = k.param::<BF16>("q", ParamKind::In, bq * d);
    let kk = k.param::<BF16>("k", ParamKind::In, bkv * d);
    let s_out = k.param::<F32>("s", ParamKind::Out, bq * bkv);
    k.warps((bq / 16) as u32);
    let k_s = k.smem::<BF16>("k_s", bkv * d);
    let q_g = k.view(q, 0, [d, 1], Shape::new(bq, d), [None, None]);
    let q_r = k.load(q_g);
    let k_g = k.view(kk, 0, [d, 1], Shape::new(bkv, d), [None, None]);
    let k_t = k.smem_view::<BF16>(k_s, 0, Shape::new(bkv, d));
    k.stage(k_t, k_g, CopyMode::Sync);
    let zero = k.zeros::<F32>(Shape::new(bq, bkv));
    let s = k.mma(zero, q_r, false, k_t, true);
    let s_view = k.view(s_out, 0, [bkv, 1], Shape::new(bq, bkv), [None, None]);
    k.store(s_view, s);
    let mut seed = 1;
    let (qd, kd) = (data(bq * d, &mut seed), data(bkv * d, &mut seed));
    check(k.finish(), vec![qd, kd], bq * bkv, (bq / 16) as u32, 1e-3);
}

/// `o = p · v` with P from global memory as the A operand and V through
/// shared memory as the `[k, n]` B operand (the transposed `ldmatrix`).
#[test_case(64, 64, 64; "square")]
#[test_case(64, 32, 128; "d 128")]
fn pv_with_a_transposed_gather(bq: usize, bkv: usize, d: usize) {
    let mut k = Kernel::new("pv");
    let p = k.param::<BF16>("p", ParamKind::In, bq * bkv);
    let v = k.param::<BF16>("v", ParamKind::In, bkv * d);
    let o_out = k.param::<F32>("o", ParamKind::Out, bq * d);
    k.warps((bq / 16) as u32);
    let v_s = k.smem::<BF16>("v_s", bkv * d);
    let p_g = k.view(p, 0, [bkv, 1], Shape::new(bq, bkv), [None, None]);
    let p_r = k.load(p_g);
    let v_g = k.view(v, 0, [d, 1], Shape::new(bkv, d), [None, None]);
    let v_t = k.smem_view::<BF16>(v_s, 0, Shape::new(bkv, d));
    k.stage(v_t, v_g, CopyMode::Sync);
    let zero = k.zeros::<F32>(Shape::new(bq, d));
    let o = k.mma(zero, p_r, false, v_t, false);
    let o_view = k.view(o_out, 0, [d, 1], Shape::new(bq, d), [None, None]);
    k.store(o_view, o);
    let mut seed = 2;
    let (pd, vd) = (data(bq * bkv, &mut seed), data(bkv * d, &mut seed));
    check(k.finish(), vec![pd, vd], bq * d, (bq / 16) as u32, 1e-3);
}

/// Row max and row sum of a `q · kᵀ` tile, and the broadcast of a row vector
/// back over the tile.
#[test]
fn row_reductions_and_broadcast() {
    let (bq, bkv, d) = (64usize, 64usize, 64usize);
    let mut k = Kernel::new("rows");
    let q = k.param::<BF16>("q", ParamKind::In, bq * d);
    let kk = k.param::<BF16>("k", ParamKind::In, bkv * d);
    let out = k.param::<F32>("out", ParamKind::Out, bq * (bkv + 2));
    k.warps((bq / 16) as u32);
    let k_s = k.smem::<BF16>("k_s", bkv * d);
    let q_g = k.view(q, 0, [d, 1], Shape::new(bq, d), [None, None]);
    let q_r = k.load(q_g);
    let k_g = k.view(kk, 0, [d, 1], Shape::new(bkv, d), [None, None]);
    let k_t = k.smem_view::<BF16>(k_s, 0, Shape::new(bkv, d));
    k.stage(k_t, k_g, CopyMode::Sync);
    let zero = k.zeros::<F32>(Shape::new(bq, bkv));
    let s = k.mma(zero, q_r, false, k_t, true);
    let mx = k.reduce(s, Axis::Row, ReduceOp::Max);
    let sum = k.reduce(s, Axis::Row, ReduceOp::Sum);
    let centred = k.binary(s, mx, BinaryOp::Sub);
    let s_view = k.view(out, 0, [bkv + 2, 1], Shape::new(bq, bkv), [None, None]);
    let mx_view = k.view(out, bkv, [bkv + 2, 1], Shape::new(bq, 1), [None, None]);
    let sum_view = k.view(out, bkv + 1, [bkv + 2, 1], Shape::new(bq, 1), [None, None]);
    k.store(s_view, centred);
    k.store(mx_view, mx);
    k.store(sum_view, sum);
    let mut seed = 3;
    let (qd, kd) = (data(bq * d, &mut seed), data(bkv * d, &mut seed));
    check(k.finish(), vec![qd, kd], bq * (bkv + 2), (bq / 16) as u32, 1e-3);
}

/// Diagnostic: with P the identity, `o` is `v` itself; print where the first
/// rows land.
#[test]
#[ignore = "diagnostic"]
fn pv_identity_probe() {
    let Some(target) = target() else { return };
    let (bq, bkv, d) = (64usize, 64usize, 64usize);
    let mut k = Kernel::new("pv_id");
    let p = k.param::<BF16>("p", ParamKind::In, bq * bkv);
    let v = k.param::<BF16>("v", ParamKind::In, bkv * d);
    let o_out = k.param::<F32>("o", ParamKind::Out, bq * d);
    k.warps(4);
    let v_s = k.smem::<BF16>("v_s", bkv * d);
    let p_g = k.view(p, 0, [bkv, 1], Shape::new(bq, bkv), [None, None]);
    let p_r = k.load(p_g);
    let v_g = k.view(v, 0, [d, 1], Shape::new(bkv, d), [None, None]);
    let v_t = k.smem_view::<BF16>(v_s, 0, Shape::new(bkv, d));
    k.stage(v_t, v_g, CopyMode::Sync);
    let zero = k.zeros::<F32>(Shape::new(bq, d));
    let o = k.mma(zero, p_r, false, v_t, false);
    let o_view = k.view(o_out, 0, [d, 1], Shape::new(bq, d), [None, None]);
    k.store(o_view, o);
    let prog = k.finish();
    let pd: Vec<f32> = (0..bq * bkv).map(|i| if i / bkv == i % bkv { 1.0 } else { 0.0 }).collect();
    // Exact in bf16: row id in the high byte, column in the low bits.
    let vd: Vec<f32> = (0..bkv * d).map(|i| ((i / d) * 2 + (i % d) / 32) as f32).collect();
    let lowering = Lowering {
        target,
        schedule: Schedule::Uniform { prefetch: Prefetch::CpAsync, unroll: false },
        grid: WarpGrid { rows: 4, cols: 1 },
        swizzle: true,
    };
    let p_t = Tensor::from_slice(&pd).cast(DType::BFloat16);
    let v_tt = Tensor::from_slice(&vd).cast(DType::BFloat16);
    let out = Tensor::empty(&[bq * d], DType::Float32);
    let got: Vec<f32> = graph_launch(prog, &lowering, &[&p_t, &v_tt, &out]).unwrap().to_vec::<f32>().unwrap();
    let map: Vec<i32> = (0..bq).map(|r| (got[r * d] / 2.0) as i32).collect();
    eprintln!("output row -> source row: {map:?}");
    let col: Vec<i32> = (0..d).map(|c| (got[c] as i32) % 2).collect();
    eprintln!("row 0 column halves: {col:?}");
}
