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
    let prog = super::gemm_nt(m, n, k, bm, bn, bk, stages);
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
    let view = k.view(x, 0, [cols, 1], Shape::new(rows, 1), [None, Some(Sc::from(valid))]);
    let acc0 = k.zeros::<F32>(Shape::new(rows, 1));
    let [acc] = k.loop_(cols, [acc0], |k, i, [acc]| {
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
    let [_m, l_acc] = k.loop_(cols / blk, [m0, l0], |k, i, [m, l]| {
        let off = i * blk;
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

/// A cast toward zero keeps an f32's upper 16 bits; the default cast rounds
/// to nearest, ties to even.
#[test_case(1.0 + 3.0 / 512.0, 1.0 + 1.0 / 128.0, 1.0; "a quarter ulp above a half rounds up, truncates down")]
#[test_case(-(1.0 + 3.0 / 512.0), -(1.0 + 1.0 / 128.0), -1.0; "toward zero from below too")]
#[test_case(0.5, 0.5, 0.5; "representable values stay")]
#[test_case(0.0, 0.0, 0.0; "zero")]
fn a_truncating_cast_keeps_the_upper_half(x: f64, nearest: f64, toward_zero: f64) {
    let mut k = Kernel::new("casts");
    let src = k.param::<F32>("x", ParamKind::In, 1);
    let near = k.param::<BF16>("nearest", ParamKind::Out, 1);
    let zero = k.param::<BF16>("toward_zero", ParamKind::Out, 1);
    let v = k.view(src, 0, [1, 1], Shape::new(1, 1), [None, None]);
    let v = k.load(v);
    let (a, b) = (k.cast::<F32, BF16>(v), k.cast_toward_zero::<BF16>(v));
    let oa = k.view(near, 0, [1, 1], Shape::new(1, 1), [None, None]);
    let ob = k.view(zero, 0, [1, 1], Shape::new(1, 1), [None, None]);
    k.store(oa, a);
    k.store(ob, b);
    let out = run(&k.finish(), vec![vec![x], vec![0.0], vec![0.0]], &[]).unwrap();
    assert_eq!((out[1][0], out[2][0]), (nearest, toward_zero));
}

/// A bf16 tile summed into f32 adds the values as held: the truncated ones,
/// not the f32 values they came from.
#[test]
fn a_bf16_tile_is_summed_as_held() {
    let (rows, cols) = (4usize, 64usize);
    let mut k = Kernel::new("held_sum");
    let x = k.param::<F32>("x", ParamKind::In, rows * cols);
    let y = k.param::<F32>("y", ParamKind::Out, rows);
    let xv = k.view(x, 0, [cols, 1], Shape::new(rows, cols), [None, None]);
    let xr = k.load(xv);
    let held = k.cast_toward_zero::<BF16>(xr);
    let sum = k.reduce_f32(held, Axis::Row, ReduceOp::Sum);
    let yv = k.view(y, 0, [1, 1], Shape::new(rows, 1), [None, None]);
    k.store(yv, sum);
    let xs: Vec<f64> = (0..rows * cols).map(|i| 1.0 + (i as f64 + 0.37) / 300.0).collect();
    let got = run(&k.finish(), vec![xs.clone(), vec![0.0; rows]], &[]).unwrap();
    let trunc = |v: f64| f64::from(f32::from_bits((v as f32).to_bits() & 0xffff_0000));
    for r in 0..rows {
        let row = &xs[r * cols..(r + 1) * cols];
        let (held, plain): (f64, f64) = (row.iter().map(|&v| trunc(v)).sum(), row.iter().sum());
        assert!((got[1][r] - held).abs() < 1e-5, "row {r}: {} against {held}", got[1][r]);
        assert!((got[1][r] - plain).abs() > 1e-3, "row {r}: the held values differ from the f32 ones");
    }
}

/// `scale` is a product whatever its factors: ones (which the lowering may
/// skip) and others alike.
#[test]
fn scale_is_a_product() {
    let (rows, cols) = (4usize, 8usize);
    let mut k = Kernel::new("scale");
    let x = k.param::<F32>("x", ParamKind::In, rows * cols);
    let f = k.param::<F32>("f", ParamKind::In, rows);
    let y = k.param::<F32>("y", ParamKind::Out, rows * cols);
    let xv = k.view(x, 0, [cols, 1], Shape::new(rows, cols), [None, None]);
    let fv = k.view(f, 0, [1, 1], Shape::new(rows, 1), [None, None]);
    let (xr, fr) = (k.load(xv), k.load(fv));
    let out = k.scale(xr, fr);
    let yv = k.view(y, 0, [cols, 1], Shape::new(rows, cols), [None, None]);
    k.store(yv, out);
    let xs: Vec<f64> = (0..rows * cols).map(|i| i as f64 - 7.5).collect();
    let fs = vec![1.0, 0.5, 1.0, -2.0];
    let got = run(&k.finish(), vec![xs.clone(), fs.clone(), vec![0.0; rows * cols]], &[]).unwrap();
    let want: Vec<f64> = xs.iter().enumerate().map(|(i, x)| x * fs[i / cols]).collect();
    assert_eq!(got[2], want);
}

/// Scalar division and remainder truncate toward zero, as the device's
/// `sdiv` and `srem` do (a ring's slot `-1 % stages` is `-1` there); a
/// zero divisor is an error, not a panic.
#[test_case(-7, 2 => Ok([-3.0, -1.0]); "a negative numerator")]
#[test_case(7, -2 => Ok([-3.0, 1.0]); "a negative divisor")]
#[test_case(-1, 3 => Ok([0.0, -1.0]); "a step before the ring")]
#[test_case(7, 0 => Err("scalar 7 Div 0 is undefined".to_string()); "a zero divisor")]
fn scalar_division_truncates(n: i64, d: i64) -> Result<[f64; 2], String> {
    let mut k = Kernel::new("div");
    let (nv, dv) = (k.var("n", -8, 8), k.var("d", -8, 8));
    let y = k.param::<I32>("y", ParamKind::Out, 2);
    for (i, v) in [nv.clone() / dv.clone(), nv % dv].into_iter().enumerate() {
        let tile = k.splat::<I32>(Shape::new(1, 1), v);
        let out = k.view(y, i, [1, 1], Shape::new(1, 1), [None, None]);
        k.store(out, tile);
    }
    let out = run(&k.finish(), vec![vec![0.0; 2]], &[("n", n), ("d", d)]).map_err(|e| e.to_string())?;
    Ok([out[0][0], out[0][1]])
}

#[derive(Clone, Copy, Debug)]
enum Access {
    Read,
    Write,
    Scalar,
    Shared,
}

/// An access past an allocation's capacity is an error naming it and the
/// element, where the device would read or write wild: a 4×4 tile at
/// element `at` of 16, a scalar at `at`, or a shared tile `at` elements in.
/// Elements past a view's bounds are no access and stay legal.
#[test_case(Access::Read, 0 => Ok(()); "a read within")]
#[test_case(Access::Read, 4 => Err("element 16 of x is accessed, which holds 16".to_string()); "a read past")]
#[test_case(Access::Write, 4 => Err("element 16 of y is accessed, which holds 16".to_string()); "a write past")]
#[test_case(Access::Scalar, 15 => Ok(()); "a scalar at the last element")]
#[test_case(Access::Scalar, 16 => Err("element 16 of x is accessed, which holds 16".to_string()); "a scalar past")]
#[test_case(Access::Scalar, -1 => Err("element -1 of x is accessed, which holds 16".to_string()); "a scalar before")]
#[test_case(Access::Shared, 4 => Err("element 16 of s is accessed, which holds 16".to_string()); "a shared tile past")]
fn an_access_past_the_capacity_is_an_error(access: Access, at: i64) -> Result<(), String> {
    let tile = Shape::new(4, 4);
    let mut k = Kernel::new("oob");
    let x = k.param::<F32>("x", ParamKind::In, 16);
    let y = k.param::<F32>("y", ParamKind::Out, 16);
    let rows = |of: Access| if std::mem::discriminant(&access) == std::mem::discriminant(&of) { at } else { 0 };
    let (read_at, write_at) = (rows(Access::Read) * 4, rows(Access::Write) * 4);
    let src = k.view(x, read_at, [tile.cols, 1], tile, [None, None]);
    let mut v = k.load(src);
    // Rows past a bound are no access, however far the view hangs over.
    let over = k.view(x, 12, [tile.cols, 1], tile, [Some(Sc::from(1)), None]);
    let over = k.load(over);
    v = k.binary(v, over, BinaryOp::Max);
    if let Access::Scalar = access {
        let s = k.load_scalar(x, at);
        v = k.splat::<F32>(tile, s);
    }
    if let Access::Shared = access {
        let s = k.smem::<F32>("s", 16);
        let shared = k.smem_view::<F32>(s, at * tile.cols as i64, tile);
        k.store(shared, v);
        v = k.load(shared);
    }
    let dst = k.view(y, write_at, [tile.cols, 1], tile, [None, None]);
    k.store(dst, v);
    run(&k.finish(), vec![vec![1.0; 16], vec![0.0; 16]], &[]).map(drop).map_err(|e| e.to_string())
}
