//! The kernel-builder vocabulary the optimizer tests share: a declarative
//! RANGE skeleton plus the buffer-backed matmul/reduce shapes the heuristics,
//! tensor-core and beam passes are ranked against.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{AxisType, ReduceOp, UOp};

use crate::test::support::prelude::*;

/// `range * stride`, the addressing idiom a scheduler-built kernel uses.
pub(crate) fn times(range: &Arc<UOp>, stride: i64) -> Arc<UOp> {
    range.try_mul(&UOp::index_const(stride)).expect("index mul")
}

/// `a + b`.
pub(crate) fn plus(a: Arc<UOp>, b: Arc<UOp>) -> Arc<UOp> {
    a.try_add(&b).expect("index add")
}

/// The row-major address of `ranges` for `shape`.
pub(crate) fn row_major(ranges: &[Arc<UOp>], shape: &[i64]) -> Arc<UOp> {
    let mut stride = 1i64;
    let mut index = UOp::index_const(0);
    for (&size, range) in shape.iter().zip(ranges).rev() {
        index = plus(index, times(range, stride));
        stride *= size;
    }
    index
}

/// Declarative kernel skeleton: the RANGEs of a kernel in [`Ranged::ranges`] order.
pub(crate) struct Ranged(Vec<Arc<UOp>>);

impl Ranged {
    /// Every axis is a `WeakInt` RANGE, the typing `range_axis` gives the scheduler's own splits, so index arithmetic maps over them without a widening cast (which the stride analyses cannot see through).
    pub(crate) fn new(spec: &[(i64, AxisType)]) -> Self {
        Self(
            spec.iter()
                .enumerate()
                .map(|(i, &(end, ty))| UOp::range_axis(UOp::index_const(end), svod_ir::AxisId::Renumbered(i), ty))
                .collect(),
        )
    }
    pub(crate) fn ranges(&self) -> &[Arc<UOp>] {
        &self.0
    }
    pub(crate) fn range(&self, i: usize) -> Arc<UOp> {
        self.0[i].clone()
    }
    /// Replace the range at `i` by one with `end`, keeping its axis type.
    pub(crate) fn with_end(mut self, i: usize, end: Arc<UOp>) -> Self {
        self.0[i] = UOp::range_axis(end, svod_ir::AxisId::Renumbered(i), range_axis_type(&self.0[i]));
        self
    }
    /// `INDEX(cpu buffer, index)` over a `numel`-element buffer: the bare access the matmul/matvec shape checks and the stride analyses read (a `LOAD` wrapper would hide every one of them).
    pub(crate) fn index(&self, dtype: &DType, numel: i64, index: Arc<UOp>) -> Arc<UOp> {
        index_of(buffer_of(numel as usize, dtype.base()), index)
    }
    /// `SINK[compute, ranges[include]..]`.
    pub(crate) fn sink(&self, compute: Arc<UOp>, include: &[usize]) -> Arc<UOp> {
        UOp::sink(std::iter::once(compute).chain(include.iter().map(|&i| self.0[i].clone())).collect())
    }
    /// `SINK[compute, every range..]`.
    pub(crate) fn sink_all(&self, compute: Arc<UOp>) -> Arc<UOp> {
        UOp::sink(std::iter::once(compute).chain(self.0.iter().cloned()).collect())
    }
}

/// `out[row] = sum_c x[row + c]` over `stored`, each load optionally widened.
pub(crate) fn row_reduce(row_axis: AxisType, rows: i64, cols: i64, stored: DType, wide: Option<DType>) -> Arc<UOp> {
    let kernel = Ranged::new(&[(rows, row_axis), (cols, AxisType::Reduce)]);
    let index = plus(kernel.range(0), kernel.range(1));
    let widen = |value: Arc<UOp>| wide.clone().map_or(value.clone(), |dtype| value.cast(dtype));
    let load = |at: Arc<UOp>| widen(kernel.index(&stored, rows * cols, at));
    let product = load(index.clone()).try_mul(&load(index)).expect("mul");
    kernel.sink(product.reduce(vec![kernel.range(1)].into(), ReduceOp::Add), &[0])
}

/// `C[m,n] = sum_k cast(A[m,k] * B[k,n])` over `stored` buffers.
pub(crate) fn matmul_accum(m: i64, n: i64, k: i64, stored: DType, accum: DType) -> Arc<UOp> {
    let kernel = Ranged::new(&[(m, AxisType::Global), (n, AxisType::Global), (k, AxisType::Reduce)]);
    let a = kernel.index(&stored, m * k, plus(times(&kernel.range(0), k), kernel.range(2)));
    let b = kernel.index(&stored, k * n, plus(times(&kernel.range(2), n), kernel.range(1)));
    let product = a.try_mul(&b).expect("mul").cast(accum);
    kernel.sink(product.reduce(vec![kernel.range(2)].into(), ReduceOp::Add), &[0, 1])
}

/// `C[m,n] = sum_k A[m,k] * B[k,n]` over `stored` buffers, each load mapped by `map`.
pub(crate) fn matmul_with(m: i64, n: i64, k: i64, stored: DType, map: impl Fn(Arc<UOp>) -> Arc<UOp>) -> Arc<UOp> {
    let kernel = Ranged::new(&[(m, AxisType::Global), (n, AxisType::Global), (k, AxisType::Reduce)]);
    let a = map(kernel.index(&stored, m * k, plus(times(&kernel.range(0), k), kernel.range(2))));
    let b = map(kernel.index(&stored, k * n, plus(times(&kernel.range(2), n), kernel.range(1))));
    let product = a.try_mul(&b).expect("mul");
    kernel.sink(product.reduce(vec![kernel.range(2)].into(), ReduceOp::Add), &[0, 1])
}

/// A convolution's shape: `C[m1, m2, n] = sum_k A[m1, m2, k] * B[k, n]`, two M
/// axes sharing every weight, over f16 operands and an f32 accumulator.
pub(crate) fn two_m_matmul(m1: i64, m2: i64, n: i64, k: i64) -> Arc<UOp> {
    let kernel =
        Ranged::new(&[(m1, AxisType::Global), (m2, AxisType::Global), (n, AxisType::Global), (k, AxisType::Reduce)]);
    let row = plus(times(&kernel.range(0), m2), kernel.range(1));
    let a = kernel.index(&DType::Float16, m1 * m2 * k, plus(times(&row, k), kernel.range(3)));
    let b = kernel.index(&DType::Float16, k * n, plus(times(&kernel.range(3), n), kernel.range(2)));
    let product = a.try_mul(&b).expect("mul").cast(DType::Float32);
    kernel.sink(product.reduce(vec![kernel.range(3)].into(), ReduceOp::Add), &[0, 1, 2])
}

/// Two N axes, so a divisibility retry has somewhere to land: `n_bad` is axis 3.
pub(crate) fn two_n_matmul(n_bad: i64, n_good: i64) -> Arc<UOp> {
    let kernel = Ranged::new(&[
        (16, AxisType::Global),
        (n_good, AxisType::Global),
        (16, AxisType::Reduce),
        (n_bad, AxisType::Global),
    ]);
    let a = kernel.index(&DType::Float32, 4096, plus(kernel.range(0), kernel.range(2)));
    let b = kernel.index(&DType::Float32, 4096, plus(plus(kernel.range(2), kernel.range(3)), kernel.range(1)));
    let product = a.try_mul(&b).expect("mul");
    kernel.sink(product.reduce(vec![kernel.range(2)].into(), ReduceOp::Add), &[0, 1, 3])
}

/// A convolution's shape with the taps kept: `C[m1, m2, n] = sum_{k, t} A[..] *
/// B[..]`, two M axes sharing one weight over a reduce that spans the channels
/// `k` and the `taps`. Each operand is laid out channels-last — the reduce is
/// its contiguous axis, as an NHWC tensor or a `[n, t, k]` weight gives — or
/// channels-first, where the reduce strides over the taps or the whole image.
pub(crate) fn taps_conv(m1: i64, m2: i64, n: i64, k: i64, taps: i64, channels_last: (bool, bool)) -> Arc<UOp> {
    let kernel = Ranged::new(&[
        (m1, AxisType::Global),
        (m2, AxisType::Global),
        (n, AxisType::Global),
        (taps, AxisType::Reduce),
        (k, AxisType::Reduce),
    ]);
    let (m1_ax, m2_ax, n_ax, t_ax, k_ax) =
        (kernel.range(0), kernel.range(1), kernel.range(2), kernel.range(3), kernel.range(4));
    let row = plus(times(&m1_ax, m2 + taps), plus(m2_ax, t_ax.clone()));
    let activation =
        if channels_last.0 { plus(times(&row, k), k_ax.clone()) } else { plus(times(&k_ax, m1 * (m2 + taps)), row) };
    let a = kernel.index(&DType::Float16, m1 * (m2 + taps) * k, activation);
    let taps_major = plus(times(&t_ax, k), k_ax.clone());
    let channels_major = plus(times(&k_ax, taps), t_ax.clone());
    let weight = plus(times(&n_ax, k * taps), if channels_last.1 { taps_major } else { channels_major });
    let b = kernel.index(&DType::Float16, n * k * taps, weight);
    let product = a.try_mul(&b).expect("mul").cast(DType::Float32);
    kernel.sink(product.reduce(vec![t_ax, k_ax].into(), ReduceOp::Add), &[0, 1, 2])
}
