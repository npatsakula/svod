use std::sync::Arc;

use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::{AxisId, AxisType, Op, ParamArg, ReduceOp, UOp};
use test_case::test_case;

use crate::optimizer::config::{HeuristicsConfig, TcOpt};
use crate::optimizer::heuristics::{
    apply_default_upcast, apply_heuristic_upcasts, apply_image_upcasts, apply_local_dims, apply_matvec_fast_path,
    apply_threading, try_tensor_cores,
};
use crate::optimizer::{Opt, OptOps, Renderer, Scheduler};
use crate::test::helpers::{create_matmul_pattern_with, create_typed_matmul_pattern};
use svod_ir::ops;

/// Matvec-shaped `sum_k A[k] * B[k]` over `stored` buffers; with `wide`, both
/// loads are cast to it before the product.
fn create_matvec_like_pattern(rows: i64, cols: i64, stored: DType, wide: Option<DType>) -> Arc<UOp> {
    create_row_reduce_pattern(AxisType::Global, rows, cols, stored, wide)
}

/// [`create_matvec_like_pattern`] with the row axis of `row_axis` type.
fn create_row_reduce_pattern(row_axis: AxisType, rows: i64, cols: i64, stored: DType, wide: Option<DType>) -> Arc<UOp> {
    let row = UOp::range_axis(UOp::index_const(rows), AxisId::Renumbered(0), row_axis);
    let reduce = UOp::range_axis(UOp::index_const(cols), AxisId::Renumbered(1), AxisType::Reduce);

    let idx_expr = row.try_add(&reduce).expect("index add should succeed");
    let load = || {
        let buffer = UOp::new_buffer(DeviceSpec::Cpu, (rows * cols) as usize, stored.clone());
        let value = UOp::index().buffer(buffer).indices(vec![idx_expr.clone()]).call().expect("index should build");
        match &wide {
            Some(wide) => value.cast(wide.clone()),
            None => value,
        }
    };
    let (a, b) = (load(), load());

    let mul = a.try_mul(&b).expect("mul should succeed");
    let red = mul.reduce(vec![reduce].into(), ReduceOp::Add);
    UOp::sink(vec![red, row])
}

fn create_tc_retry_pattern() -> Arc<UOp> {
    let m_range = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Global);
    let n_good_range = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(1), AxisType::Global);
    let k_range = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(2), AxisType::Reduce);
    let n_bad_range = UOp::range_axis(UOp::index_const(15), AxisId::Renumbered(3), AxisType::Global);

    let a_buf = UOp::new_buffer(DeviceSpec::Cpu, 4096, DType::Float32);
    let b_buf = UOp::new_buffer(DeviceSpec::Cpu, 4096, DType::Float32);

    let a_idx = m_range.try_add(&k_range).expect("A index should build");
    let b_idx = k_range.try_add(&n_bad_range).and_then(|x| x.try_add(&n_good_range)).expect("B index should build");

    let a_val = UOp::index().buffer(a_buf).indices(vec![a_idx]).call().expect("A load should build");
    let b_val = UOp::index().buffer(b_buf).indices(vec![b_idx]).call().expect("B load should build");

    let mul = a_val.try_mul(&b_val).expect("mul should succeed");
    let red = mul.reduce(vec![k_range].into(), ReduceOp::Add);
    UOp::sink(vec![red, m_range, n_good_range, n_bad_range])
}

/// A widening integer cast on the operands is exact under the int8→int32 WMMA,
/// so it must not hide the tensor core; float casts keep the generic path.
#[test_case(DType::Int8, DType::Int32, true; "int8 operands widened to int32 use the integer wmma")]
#[test_case(DType::Float16, DType::Float32, false; "float16 operands widened to float32 stay scalar")]
fn try_tensor_cores_sees_through_widening_integer_casts(stored: DType, wide: DType, uses_tc: bool) {
    let sink = create_typed_matmul_pattern(16, 16, 16, stored.clone(), Some(wide));
    let mut scheduler = Scheduler::new(sink, Renderer::amd_rdna3());

    assert_eq!(try_tensor_cores(&mut scheduler, &HeuristicsConfig::builder().build()), uses_tc);

    let wmma = scheduler.ast().toposort().into_iter().find_map(|u| match u.op() {
        Op::Wmma(ops::Wmma { metadata, .. }) => Some(metadata.dtype_in.clone()),
        _ => None,
    });
    assert_eq!(wmma, uses_tc.then_some(stored));
}

/// A fused elementwise producer on a MUL operand (`relu(A) @ B`, a padded
/// conv's `WHERE`) leaves the WMMA legal; tinygrad only checks the dtypes, so
/// the hand-coded path must not demand bare loads.
#[test]
fn try_tensor_cores_accepts_fused_operands() {
    let relu = |value: Arc<UOp>| UOp::alu(svod_ir::BinaryOp::Max, value.clone(), value.const_like(0.0f64));
    let sink = create_matmul_pattern_with(16, 16, 16, DType::Float16, relu);
    let mut scheduler = Scheduler::new(sink, Renderer::amd_rdna3());

    assert!(try_tensor_cores(&mut scheduler, &HeuristicsConfig::builder().build()));
    assert!(scheduler.ast().toposort().iter().any(|u| matches!(u.op(), Op::Wmma(..))));
}

/// The matvec fast path applies GROUP + LOCAL + UPCAST in one shot, unless
/// `matvec_enabled` turns it off.
#[test_case(true; "enabled")]
#[test_case(false; "disabled by config")]
fn test_apply_matvec_fast_path(enabled: bool) {
    let sink = create_matvec_like_pattern(64, 128, DType::Float32, None);
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());
    let config = HeuristicsConfig::builder().matvec_enabled(enabled).build();

    assert_eq!(apply_matvec_fast_path(&mut scheduler, &config), enabled);
    for axis in [AxisType::GroupReduce, AxisType::Local, AxisType::Upcast] {
        assert_eq!(!scheduler.axes_of(&[axis]).is_empty(), enabled, "{axis:?}");
    }
}

/// Widened int8 operands, the shape every integer contraction takes after the
/// early `Cast(Mul)` rewrite, still qualify for the matvec fast path.
#[test]
fn matvec_fast_path_accepts_widened_integer_operands() {
    let sink = create_matvec_like_pattern(64, 128, DType::Int8, Some(DType::Int32));
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());

    assert!(apply_matvec_fast_path(&mut scheduler, &HeuristicsConfig::builder().build()));
    assert!(!scheduler.axes_of(&[AxisType::GroupReduce]).is_empty());
}

#[test_case(DType::Image { kind: svod_dtype::ImageKind::Float, shape: vec![2, 8, 4] }, true; "image buffer")]
#[test_case(DType::Float32, false; "plain rank three tensor")]
fn test_apply_image_upcasts_non_stub_behavior(dtype: DType, expected: bool) {
    let g = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(0), AxisType::Global);
    let shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![2usize.into(), 8usize.into(), 4usize.into()]);
    let arg = ParamArg::buffer(0, dtype.clone(), AddrSpace::Global, Some(DeviceSpec::Cpu));
    let img = UOp::new(Op::Buffer(ops::Buffer { shape, arg: arg.into() }), dtype);
    let indexed = UOp::index().buffer(img).indices(vec![g.clone()]).call().expect("image index should build");
    let sink = UOp::sink(vec![indexed, g]);

    let mut scheduler = Scheduler::new(sink, Renderer::cpu());
    assert_eq!(apply_image_upcasts(&mut scheduler), expected);
    assert_eq!(scheduler.axes_of(&[AxisType::Upcast]).len(), usize::from(expected));
}

#[test]
fn test_try_tensor_cores_retries_axis_choices() {
    let sink = create_tc_retry_pattern();
    let mut scheduler = Scheduler::new(sink, Renderer::metal());

    let config = HeuristicsConfig::builder().tc_opt(TcOpt::Relaxed).build();
    let applied = try_tensor_cores(&mut scheduler, &config);
    assert!(applied, "try_tensor_cores should recover with a later axis choice");

    let tc_opt = scheduler.applied_opts.iter().find(|opt| opt.op == OptOps::TC).expect("TC opt should be recorded");
    assert_eq!(tc_opt.axis, Some(1), "retry should commit the passing axis choice");
}

/// Elementwise SINK with one WEAK axis plus an optional extra axis of `extra`
/// type, so `apply_default_upcast`'s gate and axis pick can be exercised.
fn create_default_upcast_pattern(size: i64, extra: Option<(i64, AxisType)>) -> Arc<UOp> {
    let weak = UOp::range_axis(UOp::index_const(size), AxisId::Renumbered(0), AxisType::Weak);
    let buf = UOp::new_buffer(DeviceSpec::Cpu, size as usize * 64, DType::Float32);
    let (idx, mut sink_srcs) = match extra {
        Some((extra_size, axis_type)) => {
            let other = UOp::range_axis(UOp::index_const(extra_size), AxisId::Renumbered(1), axis_type);
            (weak.try_add(&other).expect("index add"), vec![weak.clone(), other])
        }
        None => (weak.clone(), vec![weak.clone()]),
    };
    let val = UOp::index().buffer(buf).indices(vec![idx]).call().expect("index should build");
    let doubled = val.try_add(&val).expect("add should succeed");
    sink_srcs.insert(0, doubled);
    UOp::sink(sink_srcs)
}

#[test_case(16, None, true; "divisible weak axis upcasts")]
#[test_case(6, None, false; "size not divisible by four")]
#[test_case(1, None, false; "size one axis is not upcastable")]
#[test_case(16, Some((4, AxisType::Unroll)), false; "unrolled kernel skips the fallback")]
#[test_case(16, Some((4, AxisType::Upcast)), false; "already upcast kernel skips the fallback")]
#[test_case(16, Some((8, AxisType::Reduce)), true; "reduce axis does not block the fallback")]
fn default_upcast_follows_tinygrad_gate(size: i64, extra: Option<(i64, AxisType)>, expected: bool) {
    let pre_existing = usize::from(matches!(extra, Some((_, AxisType::Upcast))));
    let mut scheduler = Scheduler::new(create_default_upcast_pattern(size, extra), Renderer::cpu());

    assert_eq!(apply_default_upcast(&mut scheduler), expected);
    assert_eq!(
        scheduler.axes_of(&[AxisType::Upcast]).len(),
        pre_existing + usize::from(expected),
        "UPCAST axis count after the fallback"
    );
}

#[test]
fn default_upcast_picks_the_innermost_upcastable_axis() {
    // Tinygrad takes `k.upcastable_dims[-1]`; both axes qualify here, and only
    // the trailing one must be split.
    let sink = create_default_upcast_pattern(16, Some((8, AxisType::Global)));
    let mut scheduler = Scheduler::new(sink, Renderer::cpu());
    let innermost = *scheduler.upcastable_dims().last().expect("two upcastable dims");

    assert!(apply_default_upcast(&mut scheduler));
    let opt = scheduler.applied_opts.iter().find(|opt| opt.op == OptOps::UPCAST).expect("UPCAST recorded");
    assert_eq!(opt.axis, Some(innermost));
}

/// Elementwise SINK over `axes` GLOBAL axes of extent `size`, summing `axes`
/// row-major buffers; with `stride0`, buffer `i` skips axis `i`.
fn create_stride0_pattern(axes: usize, size: i64, stride0: bool) -> Arc<UOp> {
    let ranges: Vec<Arc<UOp>> =
        (0..axes).map(|i| UOp::range_axis(UOp::index_const(size), AxisId::Renumbered(i), AxisType::Global)).collect();
    let loads: Vec<Arc<UOp>> = (0..axes)
        .map(|skip| {
            let idx = ranges
                .iter()
                .enumerate()
                .filter(|(i, _)| !(stride0 && *i == skip))
                .map(|(i, rng)| rng.try_mul(&UOp::index_const(size.pow((axes - 1 - i) as u32))).expect("index mul"))
                .reduce(|acc, term| acc.try_add(&term).expect("index add"))
                .expect("at least one axis");
            let buf = UOp::new_buffer(DeviceSpec::Cpu, size.pow(axes as u32) as usize, DType::Float32);
            UOp::index().buffer(buf).indices(vec![idx]).call().expect("index should build")
        })
        .collect();
    let sum = loads.into_iter().reduce(|acc, load| acc.try_add(&load).expect("add")).expect("one load");
    UOp::sink(std::iter::once(sum).chain(ranges).collect())
}

/// The stride ranking picks, per round, the stride-0 axis with the fewest and
/// smallest strides and the smaller of the amounts 3 and 4 that divide it,
/// until the output shape drops below 1024 elements.
#[test_case(3, 12, true, &[(2, 3)]; "innermost axis by stride sum, amount three first")]
#[test_case(4, 8, true, &[(3, 4), (2, 4)]; "second round after the shape stays large")]
#[test_case(3, 12, false, &[]; "no stride-0 buffer means no candidate")]
fn heuristic_upcasts_rank_by_strides(axes: usize, size: i64, stride0: bool, expected: &[(usize, usize)]) {
    let mut scheduler = Scheduler::new(create_stride0_pattern(axes, size, stride0), Renderer::cpu());

    assert_eq!(apply_heuristic_upcasts(&mut scheduler), !expected.is_empty());
    let expected: Vec<Opt> = expected.iter().map(|&(axis, amount)| Opt::upcast(axis, amount)).collect();
    assert_eq!(scheduler.applied_opts, expected);
}

/// Elementwise SINK over `shape` axes of `axis_type`, loading one row-major
/// buffer, so every axis is a LOCAL/THREAD candidate without a broadcast.
fn create_elementwise_pattern(shape: &[i64], axis_type: AxisType) -> Arc<UOp> {
    let ranges: Vec<Arc<UOp>> = shape
        .iter()
        .enumerate()
        .map(|(i, &size)| UOp::range_axis(UOp::index_const(size), AxisId::Renumbered(i), axis_type))
        .collect();
    let mut stride = 1i64;
    let mut idx = UOp::index_const(0);
    for (rng, &size) in ranges.iter().zip(shape).rev() {
        idx = idx.try_add(&rng.try_mul(&UOp::index_const(stride)).expect("index mul")).expect("index add");
        stride *= size;
    }
    let buf = UOp::new_buffer(DeviceSpec::Cpu, stride as usize, DType::Float32);
    let val = UOp::index().buffer(buf).indices(vec![idx]).call().expect("index should build");
    let doubled = val.try_add(&val).expect("add should succeed");
    UOp::sink(std::iter::once(doubled).chain(ranges).collect())
}

/// A global axis none of the standard LOCAL sizes divides gets the largest
/// divisor within the budget when that fills the warps better, and is
/// padded to a real block size otherwise; divisible axes are unchanged.
#[test_case(51865, &[Opt::padto(0, 32), Opt::local(0, 32)]; "whisper vocabulary pads seven elements to 32")]
#[test_case(10007, &[Opt::padto(0, 32), Opt::local(0, 32)]; "prime extent pads to 32")]
#[test_case(385, &[Opt::local(0, 77)]; "5·7·11 keeps its exact divisor 77")]
#[test_case(12, &[Opt::local(0, 4)]; "candidate list still wins for 12")]
#[test_case(96, &[Opt::local(0, 32)]; "candidate list still wins for 96")]
#[test_case(1024, &[Opt::local(0, 32)]; "candidate list still wins for 1024")]
#[test_case(25, &[Opt::local(0, 25)]; "tie keeps the exact divisor")]
fn local_dims_fall_back_for_undividable_axes(size: i64, expected: &[Opt]) {
    let mut scheduler = Scheduler::new(create_elementwise_pattern(&[size], AxisType::Global), Renderer::cuda());

    assert!(apply_local_dims(&mut scheduler, &HeuristicsConfig::builder().build()));
    assert_eq!(scheduler.applied_opts, expected);
}

/// The decoder logits shape `[2, 51865]`: the vocabulary axis is padded and
/// localized, and the row axis still folds into the same block.
#[test]
fn local_dims_pad_the_vocabulary_axis_beside_the_row_local() {
    let mut scheduler = Scheduler::new(create_elementwise_pattern(&[2, 51865], AxisType::Global), Renderer::cuda());

    assert!(apply_local_dims(&mut scheduler, &HeuristicsConfig::builder().build()));
    // LOCAL 2 consumes axis 0 entirely, so the vocabulary axis becomes axis 0.
    assert_eq!(scheduler.applied_opts, vec![Opt::local(0, 2), Opt::padto(0, 32), Opt::local(0, 32)]);
    assert_eq!(scheduler.full_shape(), vec![1621, 2, 32]);
}

/// The matvec fast path pads a row axis the row tile does not divide when
/// the padding is cheap, and declines when it is not.
#[test_case(64, Some(&[Opt::group(0, 8), Opt::local(0, 4), Opt::upcast(0, 4)][..]); "divisible rows are unchanged")]
#[test_case(51865, Some(&[Opt::padto(0, 16), Opt::group(0, 8), Opt::local(0, 4), Opt::upcast(0, 4)][..]); "vocabulary rows pad to the tile")]
#[test_case(17, None; "padding almost doubling the rows is declined")]
fn matvec_fast_path_pads_the_row_axis(rows: i64, expected: Option<&[Opt]>) {
    let sink = create_matvec_like_pattern(rows, 128, DType::Float32, None);
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());

    assert_eq!(apply_matvec_fast_path(&mut scheduler, &HeuristicsConfig::builder().build()), expected.is_some());
    assert_eq!(scheduler.applied_opts, expected.unwrap_or_default());
}

/// CPU threading pads a loop axis no thread count divides (otherwise it runs
/// on one core); an axis some count divides keeps that count.
#[test_case(10007, 512, &[Opt::padto(0, 32), Opt::thread(0, 32)]; "prime rows pad to 32 threads")]
#[test_case(51865, 512, &[Opt::thread(0, 5)]; "a dividing count is still preferred")]
#[test_case(96, 65536, &[Opt::thread(0, 32)]; "divisible rows are unchanged")]
#[test_case(10007, 4, &[]; "too little work stays single threaded")]
fn threading_pads_undividable_loop_axes(rows: i64, cols: i64, expected: &[Opt]) {
    let sink = create_row_reduce_pattern(AxisType::Weak, rows, cols, DType::Float32, None);
    let mut scheduler = Scheduler::new(sink, Renderer::cpu());

    assert_eq!(apply_threading(&mut scheduler, 32), !expected.is_empty());
    assert_eq!(scheduler.applied_opts, expected);
}
