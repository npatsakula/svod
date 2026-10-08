use proptest::prelude::*;
use svod_dtype::ScalarDType;
use test_case::test_case;

use crate::build::*;
use crate::interp::{round_to, run};
use crate::ir::*;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
}

/// The pipelined GEMM program computes `a · bᵀ` with bf16 inputs and f32
/// accumulation, for every block of the grid, whatever the stage count.
#[test_case(64, 64, 64, 32, 32, 16, 1; "single stage")]
#[test_case(128, 64, 96, 64, 32, 32, 3; "three stages")]
#[test_case(32, 32, 32, 32, 32, 32, 2; "one trip")]
fn gemm_matches_a_naive_reference(m: usize, n: usize, k: usize, bm: usize, bn: usize, bk: usize, stages: usize) {
    let prog = super::programs::gemm_nt(m, n, k, bm, bn, bk, stages);
    let mut seed = 7;
    let a: Vec<f64> = (0..m * k).map(|_| round_to(ScalarDType::BFloat16, lcg(&mut seed))).collect();
    let b: Vec<f64> = (0..n * k).map(|_| round_to(ScalarDType::BFloat16, lcg(&mut seed))).collect();
    let out = run(&prog, vec![a.clone(), b.clone(), vec![0.0; m * n]], &[("b", 1)]).unwrap();
    for i in 0..m {
        for j in 0..n {
            let want: f64 = (0..k).map(|kk| a[i * k + kk] * b[j * k + kk]).sum();
            let want = round_to(ScalarDType::BFloat16, want);
            let got = out[2][i * n + j];
            assert!((got - want).abs() <= 2e-2 * want.abs().max(1.0), "c[{i},{j}] = {got}, want {want}");
        }
    }
}

/// A loop-carried register holds its last update after the loop, and the
/// induction variable drives the addressing: a row-wise prefix sum.
#[test]
fn loop_carries_registers_and_bounds_gate_the_tail() {
    let (rows, cols, valid) = (4usize, 8usize, 5i64);
    let mut k = Kernel::new("prefix");
    let x = k.param::<F32>("x", ParamKind::In, rows * cols);
    let y = k.param::<F32>("y", ParamKind::Out, rows);
    let bound = k.c(valid);
    let view = k.view(x, 0, [cols, 1], Shape::new(rows, 1), [None, Some(bound)]);
    let acc0 = k.zeros::<F32>(Shape::new(rows, 1));
    let trips = k.c(cols as i64);
    let [acc] = k.loop_(trips, [acc0], |k, i, [acc]| {
        let col = k.at(view, 0, i);
        let v = k.load(col);
        [k.binary(acc, v, BinaryOp::Add)]
    });
    let out = k.view(y, 0, [1, 1], Shape::new(rows, 1), [None, None]);
    k.store(out, acc);
    let prog = k.finish();
    let x: Vec<f64> = (0..rows * cols).map(|i| i as f64).collect();
    let out = run(&prog, vec![x.clone(), vec![0.0; rows]], &[]).unwrap();
    for r in 0..rows {
        let want: f64 = (0..valid as usize).map(|c| x[r * cols + c]).sum();
        assert_eq!(out[1][r], want, "row {r}: columns past the bound read as zero");
    }
}

/// Online softmax as a monoid over `(max, sum)` carried through a loop of
/// column blocks equals the direct softmax normalizer.
#[test]
fn online_softmax_normalizer_matches_the_direct_one() {
    let (rows, cols, blk) = (2usize, 16usize, 4usize);
    let mut k = Kernel::new("online");
    let x = k.param::<F32>("x", ParamKind::In, rows * cols);
    let l = k.param::<F32>("l", ParamKind::Out, rows);
    let view = k.view(x, 0, [cols, 1], Shape::new(rows, blk), [None, None]);
    let m0 = k.fill::<F32>(Shape::new(rows, 1), Const::Float(f64::NEG_INFINITY));
    let l0 = k.zeros::<F32>(Shape::new(rows, 1));
    let trips = k.c((cols / blk) as i64);
    let [_m, l_acc] = k.loop_(trips, [m0, l0], |k, i, [m, l]| {
        let off = k.mul(i, blk);
        let block = k.at(view, 0, off);
        let s = k.load(block);
        let bmax = k.reduce(s, Axis::Row, ReduceOp::Max);
        let m_new = k.binary(m, bmax, BinaryOp::Max);
        let corr = k.binary(m, m_new, BinaryOp::Sub);
        let corr = k.unary(corr, UnaryOp::Exp2);
        let p = k.binary(s, m_new, BinaryOp::Sub);
        let p = k.unary(p, UnaryOp::Exp2);
        let psum = k.reduce(p, Axis::Row, ReduceOp::Sum);
        let l_scaled = k.binary(l, corr, BinaryOp::Mul);
        [m_new, k.binary(l_scaled, psum, BinaryOp::Add)]
    });
    let out = k.view(l, 0, [1, 1], Shape::new(rows, 1), [None, None]);
    k.store(out, l_acc);
    let prog = k.finish();
    let x: Vec<f64> = (0..rows * cols).map(|i| ((i * 7) % 11) as f64 * 0.25).collect();
    let out = run(&prog, vec![x.clone(), vec![0.0; rows]], &[]).unwrap();
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let want: f64 = row.iter().map(|v| (v - mx).exp2()).sum();
        assert!((out[1][r] - want).abs() < 1e-5, "row {r}: {} vs {want}", out[1][r]);
    }
}

proptest! {
    /// bf16 and f16 rounding are idempotent, monotone and within half an ulp.
    #[test]
    fn half_rounding_is_a_rounding(x in -1e4f32..1e4f32) {
        for dtype in [ScalarDType::BFloat16, ScalarDType::Float16] {
            let r = round_to(dtype, x as f64);
            prop_assert_eq!(round_to(dtype, r), r);
            let bits = if dtype == ScalarDType::BFloat16 { 8 } else { 11 };
            let ulp = 2f64.powi((x.abs().max(f32::MIN_POSITIVE).log2().floor() as i32) - (bits - 1));
            prop_assert!((r - x as f64).abs() <= ulp / 2.0 + 1e-12, "{dtype:?}: {x} -> {r}");
        }
    }
}
