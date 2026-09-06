//! Pure tests for ST swizzles.

use std::sync::Arc;

use svod_dtype::{DType, ScalarDType};
use svod_ir::uop::eval::eval_binary_op;
use svod_ir::{ConstValue, Op, UOp};

use crate::swizzle::Swizzle;
use crate::tiles::ST_16X16;

/// Fold a pure constant `Index` expression tree (the swizzle is `Const`s under
/// `Binary` ops only) to its `i64` value.
pub(super) fn eval_const(u: &Arc<UOp>) -> i64 {
    match u.op() {
        Op::Const(cv) => match cv.0 {
            ConstValue::Int(i) => i,
            other => panic!("eval_const: non-int const {other:?}"),
        },
        Op::Binary(op, a, b) => {
            let (av, bv) = (eval_const(a), eval_const(b));
            match eval_binary_op(*op, ConstValue::Int(av), ConstValue::Int(bv)) {
                Some(ConstValue::Int(r)) => r,
                other => panic!("eval_const: {op:?}({av},{bv}) folded to {other:?}"),
            }
        }
        other => panic!("eval_const: unexpected op {other:?}"),
    }
}

/// A swizzle must be a bijection over `[0,rows)×[0,cols)` (else LDS round-trips
/// corrupt), and must not move an element out of its base fragment.
fn assert_bijection(sw: Swizzle, rows: usize, cols: usize, scalar: ScalarDType) {
    let cidx = |v: usize| UOp::index_const(v as i64);
    let mut seen = vec![false; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let (srow, scol) = sw.swizzle_rc(cidx(r), cidx(c), cols, scalar);
            let (sr, sc) = (eval_const(&srow), eval_const(&scol));
            assert!((0..rows as i64).contains(&sr), "{sw:?}: row {r},{c} -> srow {sr} out of range");
            assert!((0..cols as i64).contains(&sc), "{sw:?}: row {r},{c} -> scol {sc} out of range");
            let slot = (sr as usize) * cols + sc as usize;
            assert!(!seen[slot], "{sw:?}: collision at ({sr},{sc}) — not a bijection");
            seen[slot] = true;
        }
    }
    assert!(seen.iter().all(|&b| b), "{sw:?}: not surjective");
}

/// The 16 contiguous columns a single `ds_read` row gathers (`r` fixed) must map
/// to 16 distinct LDS slots — the bank-conflict-free property the XOR buys.
#[test]
fn test_sw16x16_row_distinct_banks() {
    let cidx = |v: usize| UOp::index_const(v as i64);
    for r in 0..16usize {
        let mut cols_seen = std::collections::HashSet::new();
        for c in 0..16usize {
            let (srow, scol) = Swizzle::Sw16x16.swizzle_rc(cidx(r), cidx(c), 16, ScalarDType::BFloat16);
            assert_eq!(eval_const(&srow), r as i64, "Sw16x16 must keep the row");
            cols_seen.insert(eval_const(&scol));
        }
        assert_eq!(cols_seen.len(), 16, "row {r}: 16 cols must map to 16 distinct slots");
    }
}

/// `Sw16x16Mma` keeps the row and XORs the 16-byte chunk with row bit 2, so for
/// every register of the `mma.sync` gather (8 rows `g` or `g+8` × one 16-byte
/// chunk, lanes `(g, t)` reading 2-byte words `2t, 2t+1` of chunk `j/4`) and for an
/// `ldmatrix`/`cp.async` phase (8 rows × one chunk) the 32 4-byte words hit 32
/// distinct banks; 8-byte (`vec4` bf16) groups stay contiguous for the vec fill.
#[test]
fn test_sw16x16_mma_conflict_free() {
    let cidx = |v: usize| UOp::index_const(v as i64);
    let word = |r: usize, c: usize| {
        let (srow, scol) = Swizzle::Sw16x16Mma.swizzle_rc(cidx(r), cidx(c), 16, ScalarDType::BFloat16);
        assert_eq!(eval_const(&srow), r as i64, "Sw16x16Mma keeps the row");
        (r * 16 + eval_const(&scol) as usize) / 2 // 4-byte word within the 512-byte fragment
    };
    for half in 0..2 {
        for chunk in 0..2 {
            let banks: std::collections::HashSet<usize> = (0..8)
                .flat_map(|g| (0..4).map(move |t| (g, t)))
                .map(|(g, t)| word(8 * half + g, 8 * chunk + 2 * t) % 32)
                .collect();
            assert_eq!(banks.len(), 32, "rows {}..: chunk {chunk} phase must cover all 32 banks", 8 * half);
        }
    }
    for r in 0..16 {
        for c in (0..16).step_by(4) {
            let w = word(r, c);
            assert_eq!(word(r, c + 2), w + 1, "row {r} col {c}: the 8-byte group stays contiguous");
        }
    }
}

#[test]
fn test_swizzle_is_bijection() {
    assert_bijection(Swizzle::Sw16x16, 16, 16, ScalarDType::BFloat16);
    assert_bijection(Swizzle::Sw16x16Mma, 16, 16, ScalarDType::BFloat16);
    assert_bijection(Swizzle::Sw16x16Mma, 16, 16, ScalarDType::Float32);
    assert_bijection(Swizzle::Sw32x32, 32, 32, ScalarDType::BFloat16);
    assert_bijection(Swizzle::Sw16x32, 16, 32, ScalarDType::BFloat16);
    assert_bijection(Swizzle::Sw32x16, 32, 16, ScalarDType::BFloat16);
}

#[test]
fn test_identity_swizzle_passthrough() {
    let row = UOp::const_(DType::Int32, ConstValue::Int(3));
    let col = UOp::const_(DType::Int32, ConstValue::Int(5));
    let (srow, scol) = ST_16X16.swizzle.swizzle_rc(row.clone(), col.clone(), 16, ScalarDType::Float32);
    assert!(Arc::ptr_eq(&srow, &row), "identity swizzle must return row unchanged");
    assert!(Arc::ptr_eq(&scol, &col), "identity swizzle must return col unchanged");
}

#[test]
fn test_base_shape_arithmetic() {
    use crate::layout::LaneMap;
    use crate::tiles::{RT_16X16, RT_16X16_MMA, RT_32X32};
    // 16x16 over wave64 -> 4 elements/thread; K spread over lane-groups in steps of 4.
    assert_eq!(ST_16X16.base.elements_per_thread(), 4);
    assert_eq!(RT_16X16.elements_per_thread(), 4);
    assert_eq!(RT_16X16.map, LaneMap::Strided { stride: 4 });
    // 32x32 over wave64 -> 16 elements/thread, the same stride-4 lane-group step.
    assert_eq!(RT_32X32.elements_per_thread(), 16);
    assert_eq!(RT_32X32.map, LaneMap::Strided { stride: 4 });
    // 16x16 over warp32 mma.sync -> 8 elements/thread.
    assert_eq!(RT_16X16_MMA.elements_per_thread(), 8);
    assert_eq!(RT_16X16_MMA.map, LaneMap::MmaSync);
}

/// The `ldmatrix` row / `cp.async` copy unit is an aligned 16-byte chunk: the
/// chunk-granular XOR of [`Swizzle::Sw16x16Mma`] (and the identity) keeps every
/// chunk one contiguous, aligned run; the HipKittens XOR variants permute at 8-byte
/// granularity and break it.
#[test]
fn test_keeps_16b_chunks() {
    let cidx = |v: usize| UOp::index_const(v as i64);
    for sw in
        [Swizzle::Identity, Swizzle::Sw16x16Mma, Swizzle::Sw16x16, Swizzle::Sw32x32, Swizzle::Sw16x32, Swizzle::Sw32x16]
    {
        let (rows, cols) = match sw {
            Swizzle::Sw32x32 => (32, 32),
            Swizzle::Sw16x32 => (16, 32),
            Swizzle::Sw32x16 => (32, 16),
            _ => (16, 16),
        };
        let contiguous = (0..rows).all(|r| {
            (0..cols).step_by(8).all(|c0| {
                let (r0, s0) = sw.swizzle_rc(cidx(r), cidx(c0), cols, ScalarDType::BFloat16);
                let (r0, s0) = (eval_const(&r0), eval_const(&s0));
                s0 % 8 == 0
                    && (1..8).all(|k| {
                        let (rk, sk) = sw.swizzle_rc(cidx(r), cidx(c0 + k), cols, ScalarDType::BFloat16);
                        eval_const(&rk) == r0 && eval_const(&sk) == s0 + k as i64
                    })
            })
        });
        assert_eq!(sw.keeps_16b_chunks(), contiguous, "{sw:?}");
    }
}
