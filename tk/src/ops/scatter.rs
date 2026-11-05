//! Rows added into a buffer at indices: atomically on a GPU, serially on the CPU.

use std::sync::Arc;

use smallvec::smallvec;
use snafu::{OptionExt, ResultExt};
use svod_dtype::{DType, DeviceSpec, ScalarDType};
use svod_ir::{ConstValue, KernelInfo, UOp};
use svod_tensor::Tensor;

use super::{DtypeSnafu, GraphSnafu, LaunchSnafu, NoAtomicAddSnafu, Result, ShapeSnafu, planner, ran};
use crate::build::{F32, I32};
use crate::kernels::scatter::{ScatterSpec, scatter_add as kernel};
use crate::launch;
use crate::lower::Lowering;
use crate::ops::config::has_kernels;

const OP: &str = "scatter_add";

/// `target[index[p]] += values[p]` for every row `p` of `values [P, D]`, over
/// what `target [N, D]` holds; an `i32` `index [P]` outside `target` (−1, say)
/// drops its row. Colliding rows all land: exactly for `i32`, whose sum has no
/// order, within the rounding of its terms for `f32`.
///
/// Taking `target` by value says what the result is: its bytes, in place. It
/// must be a realized buffer sharing no byte with `index` or `values`, which
/// may be computed from it: they finish before the add starts.
pub fn scatter_add(target: Tensor, index: &Tensor, values: &Tensor) -> Result<Tensor> {
    let dims = |t: &Tensor| t.dims().context(GraphSnafu { op: OP });
    let (ts, is, vs) = (dims(&target)?, dims(index)?, dims(values)?);
    let ([rows, d], [pairs], [q, e]) = (ts.as_slice(), is.as_slice(), vs.as_slice()) else {
        let got = format!("{ts:?}, {is:?}, {vs:?}");
        return ShapeSnafu { op: OP, operand: "target, index, values", got, expected: "[N, D], [P], [P, D]" }.fail();
    };
    let (rows, d, pairs) = (*rows, *d, *pairs);
    snafu::ensure!(
        rows > 0 && d > 0 && pairs > 0 && pairs == *q && d == *e,
        ShapeSnafu { op: OP, operand: "values", got: format!("{vs:?}"), expected: format!("[{pairs}, {d}]") }
    );
    let dtype = target.dtype();
    snafu::ensure!(
        index.dtype() == DType::Int32,
        DtypeSnafu { op: OP, operand: "index", got: index.dtype(), want: DType::Int32 }
    );
    snafu::ensure!(
        values.dtype() == dtype,
        DtypeSnafu { op: OP, operand: "values", got: values.dtype(), want: dtype.clone() }
    );
    let spec = ScatterSpec { rows, pairs, d };
    let (prog, held) = match dtype.scalar() {
        Some(ScalarDType::Int32) => (kernel::<I32>(&spec), ScalarDType::Int32),
        Some(ScalarDType::Float32) => (kernel::<F32>(&spec), ScalarDType::Float32),
        _ => return DtypeSnafu { op: OP, operand: "target", got: dtype, want: DType::Int32 }.fail(),
    };
    let tensors = [&target, index, values];
    let device = target.device();
    if device == DeviceSpec::Cpu {
        launch::check_accums(&prog, &tensors).context(LaunchSnafu { op: OP })?;
        let body = move |ph: Vec<Arc<UOp>>| serial(&ph, rows as i64, d as i64, pairs as i64);
        return Tensor::graph_kernel(OP, target.clone(), &[index, values], body).context(GraphSnafu { op: OP });
    }
    let planner = planner(&device)
        .filter(|p| has_kernels(&p.device.target) && p.device.target.atomic_add.holds(held))
        .context(NoAtomicAddSnafu { op: OP, device, dtype })?;
    ran(OP, "scatter_add", held);
    launch::graph_launch(prog, &Lowering::new(&planner.device), &tensors).context(LaunchSnafu { op: OP })
}

/// The add as one serial loop over the rows and columns, for the CPU: a store
/// back rather than an atomic add. A dropped row rewrites element 0 of its
/// column unchanged.
fn serial(ph: &[Arc<UOp>], n: i64, d: i64, p: i64) -> Arc<UOp> {
    let (row, col) = (UOp::range_const(p, 0), UOp::range_const(d, 1));
    let at = |buf: &Arc<UOp>, i: Arc<UOp>| UOp::index().buffer(buf.base()).indices(vec![i]).call().expect("INDEX");
    let flat = |r: &Arc<UOp>| r.try_mul(&UOp::index_const(d)).and_then(|r| r.try_add(&col)).expect("an index");
    let i32_ = |v: i64| UOp::const_(DType::Int32, ConstValue::Int(v));
    let to = UOp::load().index(at(&ph[1], row.clone())).call();
    let above = i32_(-1).try_cmplt(&to).expect("a compare");
    let below = to.try_cmplt(&i32_(n)).expect("a compare");
    let inside = above.try_and_op(&below).expect("a gate");
    let safe = UOp::try_where(inside.clone(), to, i32_(0)).expect("a safe row").cast(row.dtype());
    let slot = at(&ph[0], flat(&safe));
    let held = UOp::load().index(slot.clone()).call();
    let added = UOp::load().index(at(&ph[2], flat(&row))).call();
    let sum = UOp::try_where(inside, held.try_add(&added).expect("a sum"), held).expect("a kept row");
    let store = slot.store(sum).end(smallvec![row, col]);
    UOp::sink_with_info(vec![store], KernelInfo { opts_to_apply: Some(vec![]), ..Default::default() })
}
