//! The mechanisms flash attention composes, each alone on the GPU against the
//! interpreter: a global-resident A operand, the transposed `ldmatrix` B
//! operand, row reductions and vector broadcasts.

use svod_dtype::ScalarDType;
use test_case::test_case;

use super::declared;
use crate::build::*;
use crate::interp::run;
use crate::ir::*;
use crate::lower::Lowering;

fn data(n: usize, seed: &mut u64) -> Vec<f64> {
    super::rounded(ScalarDType::BFloat16, n, seed, |x| x)
}

fn check(prog: Program, mut params: Vec<Vec<f64>>, out_elems: usize, tol: f64) {
    let Some(target) = super::cp_async_target() else { return };
    params.push(vec![0.0; out_elems]);
    let want = run(&prog, params.clone(), &[]).unwrap();
    let params = super::device::poisoned(&prog, &params);
    let got = super::targets::on_device(prog, &Lowering::new(&declared(target)), &params).pop().flatten().expect("out");
    let worst = got.iter().zip(want.last().unwrap()).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max);
    assert!(worst < tol && !got.iter().any(|g| g.is_nan()), "max abs diff {worst}, or an element left unwritten");
}

/// `s = q · kᵀ` with Q loaded straight from global memory into the A layout
/// and K through shared memory.
#[test_case(64, 64, 64; "square")]
#[test_case(64, 32, 128; "d 128, narrow k")]
fn qk_from_registers_and_shared(bq: usize, bkv: usize, d: usize) {
    let prog = Kernel::record("qk", |k| {
        let q = k.input::<BF16>("q", bq * d);
        let kk = k.input::<BF16>("k", bkv * d);
        let s_out = k.output::<F32>("s", bq * bkv);
        k.warps((bq / 16) as u32);
        let k_s = k.smem::<BF16>("k_s", bkv * d);
        let q_g = k.view(q, 0, [d, 1], Shape::new(bq, d), [None, None]);
        let q_r = k.load(q_g);
        let k_g = k.view(kk, 0, [d, 1], Shape::new(bkv, d), [None, None]);
        let k_t = k.smem_view::<BF16>(k_s, 0, Shape::new(bkv, d));
        k.stage(k_t, k_g, CopyMode::Sync);
        let zero = k.zeros::<F32>(Shape::new(bq, bkv));
        let s = *k.mma(zero, q_r, false, k_t, true);
        let s_view = k.view(s_out, 0, [bkv, 1], Shape::new(bq, bkv), [None, None]);
        k.store(s_view, s);
    });
    let mut seed = 1;
    let (qd, kd) = (data(bq * d, &mut seed), data(bkv * d, &mut seed));
    check(prog, vec![qd, kd], bq * bkv, 1e-3);
}

/// `o = p · v` with P from global memory as the A operand and V through
/// shared memory as the `[k, n]` B operand (the transposed `ldmatrix`).
#[test_case(64, 64, 64; "square")]
#[test_case(64, 32, 128; "d 128")]
fn pv_with_a_transposed_gather(bq: usize, bkv: usize, d: usize) {
    let prog = Kernel::record("pv", |k| {
        let p = k.input::<BF16>("p", bq * bkv);
        let v = k.input::<BF16>("v", bkv * d);
        let o_out = k.output::<F32>("o", bq * d);
        k.warps((bq / 16) as u32);
        let v_s = k.smem::<BF16>("v_s", bkv * d);
        let p_g = k.view(p, 0, [bkv, 1], Shape::new(bq, bkv), [None, None]);
        let p_r = k.load(p_g);
        let v_g = k.view(v, 0, [d, 1], Shape::new(bkv, d), [None, None]);
        let v_t = k.smem_view::<BF16>(v_s, 0, Shape::new(bkv, d));
        k.stage(v_t, v_g, CopyMode::Sync);
        let zero = k.zeros::<F32>(Shape::new(bq, d));
        let o = *k.mma(zero, p_r, false, v_t, false);
        let o_view = k.view(o_out, 0, [d, 1], Shape::new(bq, d), [None, None]);
        k.store(o_view, o);
    });
    let mut seed = 2;
    let (pd, vd) = (data(bq * bkv, &mut seed), data(bkv * d, &mut seed));
    check(prog, vec![pd, vd], bq * d, 1e-3);
}

/// Row max and row sum of a `q · kᵀ` tile, and the broadcast of a row vector
/// back over the tile.
#[test]
fn row_reductions_and_broadcast() {
    let (bq, bkv, d) = (64usize, 64usize, 64usize);
    let prog = Kernel::record("rows", |k| {
        let q = k.input::<BF16>("q", bq * d);
        let kk = k.input::<BF16>("k", bkv * d);
        let out = k.output::<F32>("out", bq * (bkv + 2));
        k.warps((bq / 16) as u32);
        let k_s = k.smem::<BF16>("k_s", bkv * d);
        let q_g = k.view(q, 0, [d, 1], Shape::new(bq, d), [None, None]);
        let q_r = k.load(q_g);
        let k_g = k.view(kk, 0, [d, 1], Shape::new(bkv, d), [None, None]);
        let k_t = k.smem_view::<BF16>(k_s, 0, Shape::new(bkv, d));
        k.stage(k_t, k_g, CopyMode::Sync);
        let zero = k.zeros::<F32>(Shape::new(bq, bkv));
        let s = *k.mma(zero, q_r, false, k_t, true);
        let mx = k.reduce(s, Axis::Row, ReduceOp::Max);
        let sum = k.reduce(s, Axis::Row, ReduceOp::Sum);
        let centred = k.binary(s, mx, BinaryOp::Sub);
        let s_view = k.view(out, 0, [bkv + 2, 1], Shape::new(bq, bkv), [None, None]);
        let mx_view = k.view(out, bkv, [bkv + 2, 1], Shape::new(bq, 1), [None, None]);
        let sum_view = k.view(out, bkv + 1, [bkv + 2, 1], Shape::new(bq, 1), [None, None]);
        k.store(s_view, centred);
        k.store(mx_view, mx);
        k.store(sum_view, sum);
    });
    let mut seed = 3;
    let (qd, kd) = (data(bq * d, &mut seed), data(bkv * d, &mut seed));
    check(prog, vec![qd, kd], bq * (bkv + 2), 1e-3);
}
