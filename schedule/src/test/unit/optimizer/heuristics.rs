//! White-box tests over `crate::optimizer::heuristics`: one declarative kernel builder for the buffer-backed shapes the pass ranks, then the threshold tables it decides on.

use std::sync::Arc;

use proptest::prelude::*;
use svod_dtype::{AddrSpace, DType, DeviceSpec, ImageKind};
use svod_ir::{AxisType, BinaryOp, Op, ParamArg, ReduceOp, UOp, ops};
use test_case::test_case;

use crate::optimizer::config::{HeuristicsConfig, TcOpt};
use crate::optimizer::heuristics::{
    apply_default_upcast, apply_heuristic_upcasts, apply_image_upcasts, apply_local_dims, apply_masked_upcasts,
    apply_matmul_output_upcasting, apply_matmul_tiling, apply_matvec_fast_path, apply_threading, apply_unroll,
    count_strides, hand_coded_optimizations, has_matmul_pattern, is_masked, try_grouped_reduction, try_tensor_cores,
    try_warp_row_reduction,
};
use crate::optimizer::renderer::TcTilePolicy;
use crate::optimizer::{Opt, OptArg, OptOps, Renderer, Scheduler, apply_opt};
use crate::test::support::prelude::*;
use crate::test::unit::optimizer::kernels::{
    Ranged, matmul_accum, matmul_with, plus, row_major, row_reduce, taps_conv, times, two_n_matmul,
};

// THE KERNEL SHAPES THIS PASS IS RANKED AGAINST

/// Conv-shaped `C[m,n] = sum_{k,t} A[m,k,t] * B[k,t,n]`: a matmul whose reduce spans two axes, channels `k` and taps `t`.
///
/// [`crate::test::support::build::conv_like`] is the same shape but indexes through a widening cast on every axis; the
/// stride analyses this pass runs cannot see through one, so the heuristics need the [`Ranged`] typing instead.
fn conv_like_weak(m: i64, n: i64, k: i64, taps: i64) -> Arc<UOp> {
    let kernel =
        Ranged::new(&[(m, AxisType::Global), (n, AxisType::Global), (k, AxisType::Reduce), (taps, AxisType::Reduce)]);
    let a = kernel.index(
        &DType::Float16,
        m * k * taps,
        plus(times(&kernel.range(0), k * taps), plus(times(&kernel.range(2), taps), kernel.range(3))),
    );
    let b = kernel.index(
        &DType::Float16,
        k * taps * n,
        plus(times(&kernel.range(2), taps * n), plus(times(&kernel.range(3), n), kernel.range(1))),
    );
    let product = a.try_mul(&b).expect("mul");
    kernel.sink(product.reduce(vec![kernel.range(2), kernel.range(3)].into(), ReduceOp::Add), &[0, 1])
}

/// `out[r, c] = x[r, c] * s[r]`: row-major elementwise with a per-row broadcast.
fn row_scaled(rows: i64, cols: i64, dtype: DType) -> Arc<UOp> {
    let kernel = Ranged::new(&[(rows, AxisType::Global), (cols, AxisType::Global)]);
    let value = kernel.index(&dtype, rows * cols, plus(times(&kernel.range(0), cols), kernel.range(1)));
    let scale = kernel.index(&DType::Float32, rows, kernel.range(0));
    kernel.sink_all(value.cast(DType::Float32).try_mul(&scale).expect("mul"))
}

/// A doubled row-major load: the shape every LOCAL/UPCAST decision sees.
fn elementwise_load(shape: &[i64], axis_type: AxisType) -> Arc<UOp> {
    let kernel = Ranged::new(&shape.iter().map(|&size| (size, axis_type)).collect::<Vec<_>>());
    let value = kernel.index(&DType::Float32, shape.iter().product(), row_major(kernel.ranges(), shape));
    kernel.sink_all(plus(value.clone(), value))
}

/// Elementwise SINK over `axes` GLOBAL axes of extent `size`, summing `axes` loads; the load for `absent` skips that axis, giving it stride zero.
fn stride_zero_buffers(axes: usize, size: i64, skip: bool) -> Arc<UOp> {
    let kernel = Ranged::new(&vec![(size, AxisType::Global); axes]);
    let loads = (0..axes).map(|absent| {
        let index = (0..axes)
            .filter(|axis| !(skip && *axis == absent))
            .map(|axis| times(&kernel.range(axis), size.pow((axes - 1 - axis) as u32)))
            .fold(UOp::index_const(0), plus);
        kernel.index(&DType::Float32, size.pow(axes as u32), index)
    });
    kernel.sink_all(loads.reduce(plus).expect("one load"))
}

/// Elementwise SINK with one WEAK axis plus an optional extra axis of `extra`.
fn default_upcast(size: i64, extra: Option<(i64, AxisType)>) -> Arc<UOp> {
    let kernel = Ranged::new(&std::iter::once((size, AxisType::Weak)).chain(extra).collect::<Vec<_>>());
    let index = if extra.is_some() { plus(kernel.range(0), kernel.range(1)) } else { kernel.range(0) };
    let value = kernel.index(&DType::Float32, size * 64, index);
    kernel.sink_all(plus(value.clone(), value))
}

/// `out[row] = sum_c x[row * row_stride + c * reduce_stride]`.
fn laid_out_reduce(rows: i64, cols: i64, row_stride: i64, reduce_stride: i64, dtype: DType) -> Arc<UOp> {
    let kernel = Ranged::new(&[(rows, AxisType::Global), (cols, AxisType::Reduce)]);
    let index = plus(times(&kernel.range(0), row_stride), times(&kernel.range(1), reduce_stride));
    let value = kernel.index(&dtype, rows * cols, index);
    kernel.sink(value.reduce(vec![kernel.range(1)].into(), ReduceOp::Add), &[0])
}

/// `out[c, r] = x[r, c]`: contiguous on one side of every axis, strided on the other.
fn transposing_copy(rows: i64, cols: i64) -> Arc<UOp> {
    let kernel = Ranged::new(&[(rows, AxisType::Global), (cols, AxisType::Global)]);
    let at = |row: usize, col: i64| plus(times(&kernel.range(row), col), kernel.range(1 - row));
    let value = plus(
        kernel.index(&DType::Float16, rows * cols, at(0, cols)),
        kernel.index(&DType::Float16, rows * cols, at(1, rows)),
    );
    kernel.sink_all(value)
}

/// `out[r, c] = sum_t x[(r * cols + c) * taps + t]`: a stencil whose column axis is the innermost reduce.
fn stencil_reduce(rows: i64, cols: i64, taps: i64) -> Arc<UOp> {
    let kernel = Ranged::new(&[(rows, AxisType::Global), (cols, AxisType::Global), (taps, AxisType::Reduce)]);
    let index = plus(plus(times(&kernel.range(0), cols * taps), times(&kernel.range(1), taps)), kernel.range(2));
    let value = kernel.index(&DType::Float16, rows * cols * taps, index);
    kernel.sink(value.reduce(vec![kernel.range(2)].into(), ReduceOp::Add), &[0, 1])
}

// FIXTURES

/// Run `pass` against a fresh scheduler; hand back its verdict and the state.
fn run(
    sink: Arc<UOp>,
    renderer: Renderer,
    config: &HeuristicsConfig,
    pass: impl FnOnce(&mut Scheduler, &HeuristicsConfig) -> bool,
) -> (bool, Scheduler) {
    let mut scheduler = Scheduler::new(sink, renderer);
    let applied = pass(&mut scheduler, config);
    (applied, scheduler)
}

/// The post-TC opt sequence a matmul gets on `renderer`, as `(op, axis, arg)`.
fn tc_plan(m: i64, n: i64, k: i64, renderer: Renderer) -> Vec<(OptOps, Option<usize>, OptArg)> {
    let mut scheduler = Scheduler::new(matmul_accum(m, n, k, DType::Float16, DType::Float32), renderer);
    assert!(try_tensor_cores(&mut scheduler, &HeuristicsConfig::builder().build()));
    scheduler
        .applied_opts
        .iter()
        .filter(|opt| opt.op != OptOps::TC)
        .map(|opt| (opt.op, opt.axis, opt.arg.clone()))
        .collect()
}

/// `(op, axis, arg)` shorthand for a post-TC UPCAST/LOCAL.
fn opt(op: OptOps, axis: usize, arg: usize) -> (OptOps, Option<usize>, OptArg) {
    (op, Some(axis), OptArg::Int(arg))
}

/// The 192->192 3x3 convolution over a 40x40 image the layout probe measures.
const PROBE_CONV: (i64, i64, i64, i64, i64) = (40, 40, 192, 192, 9);

/// Tensor-core tiles one warp's output covers under `plan`: every UPCAST
/// multiplies it, and a lane holds an accumulator per element of each tile.
fn warp_tiles(plan: &[(OptOps, Option<usize>, OptArg)]) -> usize {
    plan.iter()
        .filter_map(|(op, _, arg)| match (op, arg) {
            (OptOps::UPCAST, OptArg::Int(amount)) => Some(*amount),
            _ => None,
        })
        .product()
}

/// The post-TC opt sequence a `(m1, m2, n, k, taps)` convolution gets on the
/// RDNA4 WMMA for operands laid out `channels_last`; `None` when the shape
/// declines the tensor core.
fn conv_plan(
    shape: (i64, i64, i64, i64, i64),
    channels_last: (bool, bool),
) -> Option<Vec<(OptOps, Option<usize>, OptArg)>> {
    conv_plan_on(shape, channels_last, Renderer::amd_rdna4())
}

/// [`conv_plan`] against an explicit renderer, so the CUDA `LaneBudget` tiling
/// can be pinned beside RDNA4's fixed step.
fn conv_plan_on(
    shape: (i64, i64, i64, i64, i64),
    channels_last: (bool, bool),
    renderer: Renderer,
) -> Option<Vec<(OptOps, Option<usize>, OptArg)>> {
    let (m1, m2, n, k, taps) = shape;
    let mut scheduler = Scheduler::new(taps_conv(m1, m2, n, k, taps, channels_last), renderer);
    try_tensor_cores(&mut scheduler, &HeuristicsConfig::builder().build()).then(|| {
        scheduler
            .applied_opts
            .iter()
            .filter(|opt| opt.op != OptOps::TC)
            .map(|opt| (opt.op, opt.axis, opt.arg.clone()))
            .collect()
    })
}

/// `(axis type, constant extent)` of a RANGE; `None` when it is not a RANGE.
fn range_axis(range: &Arc<UOp>) -> Option<(AxisType, i64)> {
    matches!(range.op(), Op::Range(..)).then(|| (range_axis_type(range), expect_range_extent(range)))
}

// TENSOR CORES

/// A widening integer cast on the operands is exact under the int8→int32 WMMA,
#[test_case(DType::Int8, DType::Int32, true; "int8 operands widened to int32 use the integer wmma")]
#[test_case(DType::Float16, DType::Float32, false; "float16 operands widened to float32 stay scalar")]
fn try_tensor_cores_sees_through_widening_integer_casts(stored: DType, wide: DType, uses_tc: bool) {
    let sink = matmul_with(16, 16, 16, stored.clone(), move |value| value.cast(wide.clone()));
    let (applied, scheduler) = run(sink, Renderer::amd_rdna3(), &HeuristicsConfig::default(), try_tensor_cores);
    assert_eq!(applied, uses_tc);
    let wmma_in = first_op(scheduler.ast(), |op| matches!(op, Op::Wmma(..))).map(|u| {
        let Op::Wmma(ops::Wmma { metadata, .. }) = u.op() else { unreachable!() };
        metadata.dtype_in.clone()
    });
    assert_eq!(wmma_in, uses_tc.then_some(stored));
}

/// A fused elementwise producer on a MUL operand (`relu(A) @ B`) leaves the TC pattern intact.
#[test]
fn try_tensor_cores_accepts_fused_operands() {
    let relu = |value: Arc<UOp>| UOp::alu(BinaryOp::Max, value.clone(), value.const_like(0.0f64));
    let (applied, scheduler) = run(
        matmul_with(16, 16, 16, DType::Float16, relu),
        Renderer::amd_rdna3(),
        &HeuristicsConfig::default(),
        try_tensor_cores,
    );
    assert!(applied);
    assert!(has_op(scheduler.ast(), |op| matches!(op, Op::Wmma(..))));
}

/// A conv-shaped reduce over (channels, taps) takes the tensor core by default,
/// and the axis it puts on K is the one that pads least — at equal padding, the
/// deeper reduce. The ranges carry taps innermost, so the detection order offers
/// the taps first; taking them would pad a narrow tap axis to the core's K edge
/// and leave the channels as a scalar loop.
#[test_case(64, 5, TcOpt::Relaxed, Some(5); "wide channels with five taps")]
#[test_case(16, 25, TcOpt::Relaxed, Some(25); "narrow channels with many taps")]
#[test_case(64, 16, TcOpt::Relaxed, Some(16); "both divisible takes the deeper reduce")]
#[test_case(12, 5, TcOpt::Relaxed, None; "no reduce axis divisible")]
#[test_case(64, 5, TcOpt::Strict, None; "strict declines the second reduce axis")]
fn try_tensor_cores_on_conv_shaped_double_reduce(channels: i64, taps: i64, tc_opt: TcOpt, leftover: Option<i64>) {
    let (applied, scheduler) = run(
        conv_like_weak(32, 32, channels, taps),
        Renderer::cuda(),
        &HeuristicsConfig::builder().tc_opt(tc_opt).build(),
        try_tensor_cores,
    );
    let uses_tc = leftover.is_some();
    assert_eq!(applied, uses_tc);
    assert_eq!(has_op(scheduler.ast(), |op| matches!(op, Op::Wmma(..))), uses_tc);
    let Some(leftover) = leftover else { return };
    let loops: Vec<_> =
        scheduler.rngs().iter().filter_map(range_axis).filter(|(_, extent)| *extent == leftover).collect();
    assert_eq!(loops, vec![(AxisType::Reduce, leftover)], "the other reduce axis must survive as a loop");
}

/// A conv compute-bound enough to pad past the budget still reduces over its
/// channels. `COMPUTE_BOUND_INTENSITY` waives the padding budget so a tensor
/// core can take an axis it does not divide, which is what offers the taps —
/// 3 padded to the core's 16-wide K edge is 5.3x the MACs, and it leaves the
/// channels as a scalar loop and `k_tiles` at 1, so the warp tile cannot grow
/// either. These shapes run at 75-92 FLOP per operand byte, past the waiver.
#[test_case(3; "three taps")]
#[test_case(5; "five taps")]
#[test_case(9; "nine taps, a 3x3 conv")]
fn compute_bound_conv_reduces_over_channels_not_padded_taps(taps: i64) {
    let (applied, scheduler) =
        run(conv_like_weak(512, 128, 96, taps), Renderer::cuda(), &HeuristicsConfig::default(), try_tensor_cores);
    assert!(applied, "the conv takes a tensor core");
    let loops: Vec<_> = scheduler.rngs().iter().filter_map(range_axis).filter(|(_, extent)| *extent == taps).collect();
    assert_eq!(loops, vec![(AxisType::Reduce, taps)], "the taps stay a loop rather than being padded onto the core");
}

/// The CUDA `m16n8k16` core holds four accumulators per lane and the lane count picks the tile.
#[test_case(8192, 3072, 768, &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 8)]; "gigaam 768 to 3072 projection")]
#[test_case(8192, 768, 3072, &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 8)]; "gigaam 3072 to 768 projection")]
#[test_case(8192, 48, 768, &[(OptOps::UPCAST, 0, 2), (OptOps::UPCAST, 1, 3)]; "narrow output spends the budget on M")]
#[test_case(8192, 768, 320, &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 4)]; "short reduce cannot amortise a wider tile")]
#[test_case(256, 768, 768, &[(OptOps::UPCAST, 0, 2), (OptOps::UPCAST, 1, 4)]; "small output keeps warps over the tile")]
#[test_case(64, 64, 64, &[]; "an output of 32 warp tiles is not worth growing")]
fn cuda_tensor_core_warp_tile(m: i64, n: i64, k: i64, expected: &[(OptOps, usize, usize)]) {
    let expected: Vec<_> = expected.iter().map(|&(op, axis, arg)| opt(op, axis, arg)).collect();
    assert_eq!(tc_plan(m, n, k, Renderer::cuda()), expected);
}

/// Every target off CUDA keeps [`TcTilePolicy::FixedStep`], tinygrad's step:
type FixedStepPlan = &'static [(OptOps, usize, usize)];
const FIXED_STEP: &[((usize, usize), FixedStepPlan)] = &[
    ((16, 3), &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 3)]),
    ((16, 20), &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 5), (OptOps::LOCAL, 1, 4)]),
    ((16, 48), &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 4), (OptOps::LOCAL, 1, 4)]),
    ((16, 96), &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 4), (OptOps::LOCAL, 1, 4)]),
    ((16, 192), &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 4), (OptOps::LOCAL, 1, 4)]),
    ((8, 6), &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 3), (OptOps::LOCAL, 1, 2)]),
    ((8, 40), &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 5), (OptOps::LOCAL, 1, 4)]),
    ((8, 96), &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 4), (OptOps::LOCAL, 1, 4)]),
    ((8, 192), &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 4), (OptOps::LOCAL, 1, 4)]),
    ((8, 384), &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 4), (OptOps::LOCAL, 1, 4)]),
];

const FIXED_STEP_GRID_M: [i64; 3] = [256, 1024, 8192];
const FIXED_STEP_GRID_N: [i64; 5] = [48, 320, 768, 1536, 3072];
const FIXED_STEP_GRID_K: [i64; 3] = [320, 768, 3072];

#[test_case(Renderer::amd_rdna3(), (16, 16); "rdna3 wmma")]
#[test_case(Renderer::amd_rdna4(), (16, 16); "rdna4 wmma")]
#[test_case(Renderer::amd_cdna3(), (16, 16); "cdna3 mfma")]
#[test_case(Renderer::amd_cdna4(), (16, 16); "cdna4 mfma")]
#[test_case(Renderer::metal(), (8, 8); "metal simdgroup")]
#[test_case(Renderer::intel_xe(), (8, 8); "intel xe dpas")]
fn non_cuda_tiling_matches_the_shipped_fixed_step(renderer: Renderer, dims: (usize, usize)) {
    assert_eq!(renderer.tc_tile_policy(), TcTilePolicy::FixedStep);
    for m in FIXED_STEP_GRID_M {
        for n in FIXED_STEP_GRID_N {
            let &(_, plan) = FIXED_STEP
                .iter()
                .find(|((width, tiles), _)| (*width, *tiles) == (dims.0, n as usize / dims.0))
                .unwrap_or_else(|| panic!("no captured plan for {dims:?} and {n}"));
            let expected: Vec<_> = plan.iter().map(|&(op, axis, arg)| opt(op, axis, arg)).collect();
            for k in FIXED_STEP_GRID_K {
                assert_eq!(tc_plan(m, n, k, renderer.clone()), expected, "{m}x{n}x{k}");
            }
        }
    }
}

const PLAIN_STEP: &[(OptOps, usize, usize)] = &[(OptOps::UPCAST, 1, 3), (OptOps::UPCAST, 1, 4)];

#[test_case((false, false), PLAIN_STEP; "both channels-first keeps the plain step")]
#[test_case((true, true), PLAIN_STEP; "both channels-last keeps the plain step")]
#[test_case((false, true), PLAIN_STEP; "a strided activation is already what N holds")]
#[test_case((true, false), &[(OptOps::UPCAST, 1, 3), (OptOps::LOCAL, 0, 4)]; "a strided weight is stacked over the second spatial axis")]
fn conv_warp_tile_grows_where_the_pricier_fragment_is_reused(
    channels_last: (bool, bool),
    expected: &[(OptOps, usize, usize)],
) {
    let expected: Vec<_> = expected.iter().map(|&(op, axis, arg)| opt(op, axis, arg)).collect();
    assert_eq!(conv_plan(PROBE_CONV, channels_last), Some(expected));
}

/// On CUDA the warp tile is capped by the trips the accumulator is reused over,
/// and a convolution's taps are trips: they stay a loop around the WMMA while
/// the accumulator is set up and written back once for all of them. Counting
/// only what the core left of its own K axis divides that depth by the tap
/// count — at `k = 16` the core consumes the whole channel axis and the tile
/// cannot grow at all, though nine taps of reuse sit behind it.
///
/// The single-tap row is the control: with no taps there is genuinely nothing to
/// amortise a wider tile, and the cap must still bite.
#[test_case((40, 40, 192, 16, 9), 9; "the core takes the whole channel axis, the taps carry the reuse")]
#[test_case((40, 40, 192, 32, 9), 12; "two channel trips and nine taps")]
#[test_case(PROBE_CONV, 12; "192 channels, nine taps, 40x40")]
#[test_case((40, 40, 192, 16, 1), 1; "one tap and one channel trip cannot amortise a wider tile")]
fn cuda_conv_warp_tile_counts_the_taps_as_reduce_trips(shape: (i64, i64, i64, i64, i64), tiles: usize) {
    let plan = conv_plan_on(shape, (true, true), Renderer::cuda()).expect("the conv takes a tensor core");
    assert_eq!(warp_tiles(&plan), tiles, "warp tile for {shape:?}: {plan:?}");
}

// Growing the tile along the axis that reuses the pricier operand fragment is a
// choice of direction and not of size: whatever the operands' layouts, one warp
// still holds at most the accumulators the plain step would give it, and every
// UPCAST the plan records stays replayable.
proptest! {
    #![proptest_config(cheap())]
    #[test]
    fn conv_warp_tile_never_outgrows_the_plain_step(m1 in 8i64..=64, m2 in 8i64..=64, n in 1i64..=8, taps in 1i64..=9) {
        let shape = (m1, m2, n * 16, 192, taps);
        let Some(plain) = conv_plan(shape, (false, false)).map(|plan| warp_tiles(&plan)) else { return Ok(()) };
        for channels_last in [(true, false), (false, true), (true, true)] {
            let Some(plan) = conv_plan(shape, channels_last) else { continue };
            let tiles = warp_tiles(&plan);
            prop_assert!(tiles <= plain, "{channels_last:?} grows to {tiles} over the plain step's {plain}");
            prop_assert_eq!(conv_plan(shape, channels_last), Some(plan.clone()), "the plan is a function of the shape");
            for (op, _, arg) in plan {
                let OptArg::Int(amount) = arg else { continue };
                prop_assert!(op != OptOps::UPCAST || amount <= Renderer::amd_rdna4().upcast_max);
            }
        }
    }
}

/// A wider warp tile must never record an UPCAST the renderer would refuse to replay.
#[test_case(8192, 3072, 768; "gigaam projection")]
#[test_case(8192, 768, 3072; "wide reduce")]
#[test_case(1024, 320, 768; "an N the odd factors reach for")]
#[test_case(256, 768, 768; "small output")]
fn lane_budget_warp_tile_stays_replayable(m: i64, n: i64, k: i64) {
    let renderer = Renderer::cuda();
    assert!(matches!(renderer.tc_tile_policy(), TcTilePolicy::LaneBudget { .. }));
    let upcast_max = renderer.upcast_max;
    for (op, _, arg) in tc_plan(m, n, k, renderer) {
        let OptArg::Int(amount) = arg else { panic!("post-TC opts carry Int args") };
        assert!(op != OptOps::UPCAST || amount <= upcast_max, "{m}x{n}x{k}: UPCAST {amount} > {upcast_max}");
    }
}

// Every plan the heuristic records must be replayable through `apply_opt`, so
// its amounts have to stay inside the renderer's caps — for every divisible
// shape, not only the sampled grid.
proptest! {
    #![proptest_config(cheap())]
    #[test]
    fn tensor_core_plans_stay_replayable(m in 1u32..=32, n in 1u32..=32, k in 1u32..=8) {
        let (m, n, k) = (m as i64 * 16, n as i64 * 16, k as i64 * 16);
        let renderer = Renderer::cuda();
        let run_once = |renderer: Renderer| {
            let mut scheduler = Scheduler::new(matmul_accum(m, n, k, DType::Float16, DType::Float32), renderer);
            try_tensor_cores(&mut scheduler, &HeuristicsConfig::builder().build()).then_some(scheduler.applied_opts)
        };
        let Some(opts) = run_once(renderer.clone()) else {
            prop_assert!(false, "a fully divisible f16 matmul must take the tensor core");
            unreachable!()
        };
        prop_assert_eq!(run_once(renderer.clone()), Some(opts.clone()), "the plan is a function of the shape");
        for opt in opts {
            let OptArg::Int(amount) = opt.arg else { continue };
            prop_assert!(amount > 0, "a split amount is never zero: {opt:?}");
            match opt.op {
                OptOps::UPCAST => prop_assert!(amount <= renderer.upcast_max, "UPCAST {amount} over the cap"),
                OptOps::LOCAL => prop_assert!(amount <= renderer.local_max.unwrap_or(usize::MAX)),
                _ => {}
            }
        }
    }
    /// The LOCAL work-group budget holds for any extent, including the padded fallbacks the example table only samples.
    #[test]
    fn local_dims_stay_within_their_budget(size in 2usize..300_000) {
        let mut scheduler = Scheduler::new(elementwise_load(&[size as i64], AxisType::Global), Renderer::cuda());
        if apply_local_dims(&mut scheduler, &HeuristicsConfig::default()) {
            let local: i64 = scheduler
                .rngs()
                .iter()
                .filter(|range| range_axis_type(range) == AxisType::Local)
                .map(|range| expect_range_extent(range))
                .product();
            prop_assert!(local <= 128, "LOCAL product {local} over the 128-element budget");
            prop_assert!(local >= 2, "a LOCAL split smaller than two is not a work-group");
        }
    }
}

/// The default level changes nothing for a single-reduce matmul: the same opts as `Strict`.
#[test]
fn try_tensor_cores_default_matches_strict_on_plain_matmul() {
    let plan = |tc_opt: TcOpt| {
        let mut scheduler = Scheduler::new(matmul_with(64, 64, 64, DType::Float16, |v| v), Renderer::cuda());
        assert!(try_tensor_cores(&mut scheduler, &HeuristicsConfig::builder().tc_opt(tc_opt).build()));
        (
            scheduler.applied_opts.iter().map(|opt| (opt.op, opt.axis)).collect::<Vec<_>>(),
            scheduler.rngs().iter().map(range_axis).collect::<Vec<_>>(),
        )
    };
    assert_eq!(HeuristicsConfig::default().tc_opt, TcOpt::Padded);
    assert_eq!(plan(TcOpt::default()), plan(TcOpt::Strict));
}

/// Two N axes, a bad one first: `Metal`'s retry must commit the axis choice that divides.
#[test]
fn try_tensor_cores_retries_axis_choices() {
    let (applied, scheduler) = run(
        two_n_matmul(15, 16),
        Renderer::metal(),
        &HeuristicsConfig::builder().tc_opt(TcOpt::Relaxed).build(),
        try_tensor_cores,
    );
    assert!(applied, "try_tensor_cores should recover with a later axis choice");
    let tc_opt = scheduler.applied_opts.iter().find(|opt| opt.op == OptOps::TC).expect("TC opt recorded");
    assert_eq!(tc_opt.axis, Some(1), "retry should commit the passing axis choice");
}

// MATVEC AND IMAGE

/// The matvec fast path applies GROUP + LOCAL + UPCAST in one shot, and only when the config enables it; widened int8 operands, the shape every integer contraction takes, are accepted with the same splits.
#[test_case(64, 128, DType::Float32, None, true, true; "enabled")]
#[test_case(64, 128, DType::Float32, None, false, false; "disabled by config")]
#[test_case(64, 128, DType::Int8, Some(DType::Int32), true, true; "widened integer operands")]
fn test_apply_matvec_fast_path(rows: i64, cols: i64, stored: DType, wide: Option<DType>, enabled: bool, applied: bool) {
    let config = HeuristicsConfig::builder().matvec_enabled(enabled).build();
    let sink = row_reduce(AxisType::Global, rows, cols, stored, wide);
    let (returned, scheduler) = run(sink, Renderer::cuda(), &config, apply_matvec_fast_path);
    assert_eq!(returned, applied);
    for axis in [AxisType::GroupReduce, AxisType::Local, AxisType::Upcast] {
        assert_eq!(!scheduler.axes_of(&[axis]).is_empty(), applied, "{axis:?}");
    }
}

/// The matvec fast path pads a row axis the row tile does not divide when the padding stays cheap.
#[test_case(64, Some(&[Opt::group(0, 8), Opt::local(0, 4), Opt::upcast(0, 4)][..]); "divisible rows are unchanged")]
#[test_case(51865, Some(&[Opt::padto(0, 16), Opt::group(0, 8), Opt::local(0, 4), Opt::upcast(0, 4)][..]); "vocabulary rows pad to the tile")]
#[test_case(17, None; "padding almost doubling the rows is declined")]
fn matvec_fast_path_pads_the_row_axis(rows: i64, expected: Option<&[Opt]>) {
    let (applied, scheduler) = run(
        row_reduce(AxisType::Global, rows, 128, DType::Float32, None),
        Renderer::cuda(),
        &HeuristicsConfig::default(),
        apply_matvec_fast_path,
    );
    assert_eq!(applied, expected.is_some());
    assert_eq!(scheduler.applied_opts, expected.unwrap_or_default());
}

/// The float image dtype of a rank-3 `shape`.
fn image_dtype(shape: &[usize]) -> DType {
    DType::Image { kind: ImageKind::Float, shape: shape.to_vec() }
}

/// A rank-3 `dtype` buffer of `shape` loaded through `index`.
fn buffer_pattern(
    dtype: DType,
    shape: &[usize],
    ranges: &[(i64, AxisType)],
    index: impl Fn(&Ranged) -> Arc<UOp>,
) -> Arc<UOp> {
    let kernel = Ranged::new(ranges);
    let buffer = UOp::new(
        Op::Buffer(ops::Buffer {
            shape: svod_ir::shape::shape_to_uop(&shape.iter().map(|&dim| dim.into()).collect()),
            arg: ParamArg::buffer(0, dtype.clone(), AddrSpace::Global, Some(DeviceSpec::Cpu)).into(),
        }),
        dtype,
    );
    kernel.sink_all(load(index_of(buffer, index(&kernel))))
}

/// Both gates have to hold: the buffer's dtype must say image, and the shape it
/// carries must be the `[.., 4]` rank-3 geometry an image promises.
#[test_case(image_dtype(&[2, 8, 4]), &[2, 8, 4], &[Opt::upcast(0, 4)]; "trailing channel four upcasts the global axis")]
#[test_case(image_dtype(&[2, 8, 8]), &[2, 8, 8], &[]; "a trailing dim other than four is not an image")]
#[test_case(DType::Float32, &[2, 8, 4], &[]; "a plain rank-3 tensor of the image shape is not an image")]
fn apply_image_upcasts_only_fires_on_an_image_buffer(dtype: DType, shape: &[usize], expected: &[Opt]) {
    let sink = buffer_pattern(dtype, shape, &[(8, AxisType::Global)], |kernel| kernel.range(0));
    let (applied, scheduler) = run(sink, Renderer::cpu(), &HeuristicsConfig::default(), |s, _| apply_image_upcasts(s));
    assert_eq!(applied, !expected.is_empty());
    assert_eq!(scheduler.applied_opts, expected);
}

#[test]
fn apply_image_upcasts_unrolls_a_divisible_reduce_axis() {
    let sink = buffer_pattern(
        image_dtype(&[2, 8, 4]),
        &[2, 8, 4],
        &[(2, AxisType::Global), (8, AxisType::Reduce)],
        |kernel| plus(times(&kernel.range(0), 8), kernel.range(1)),
    );
    let (applied, scheduler) = run(sink, Renderer::cuda(), &HeuristicsConfig::default(), |s, _| apply_image_upcasts(s));
    assert!(applied, "a reduce axis is not upcastable, so the image path must unroll it");
    assert_eq!(scheduler.applied_opts, vec![Opt::unroll(0, 4)]);
}

// DEFAULT / MASKED / UNROLLED UPCASTS

#[test_case(16, None, true; "divisible weak axis upcasts")]
#[test_case(6, None, false; "size not divisible by four")]
#[test_case(1, None, false; "size one axis is not upcastable")]
#[test_case(16, Some((4, AxisType::Unroll)), false; "unrolled kernel skips the fallback")]
#[test_case(16, Some((4, AxisType::Upcast)), false; "already upcast kernel skips the fallback")]
#[test_case(16, Some((8, AxisType::Reduce)), true; "reduce axis does not block the fallback")]
fn default_upcast_follows_tinygrad_gate(size: i64, extra: Option<(i64, AxisType)>, expected: bool) {
    let pre_existing = usize::from(matches!(extra, Some((_, AxisType::Upcast))));
    let (applied, scheduler) =
        run(default_upcast(size, extra), Renderer::cpu(), &HeuristicsConfig::default(), |s, _| apply_default_upcast(s));
    assert_eq!(applied, expected);
    assert_eq!(scheduler.axes_of(&[AxisType::Upcast]).len(), pre_existing + usize::from(expected));
}

#[test]
fn default_upcast_picks_the_innermost_upcastable_axis() {
    let mut scheduler = Scheduler::new(default_upcast(16, Some((8, AxisType::Global))), Renderer::cpu());
    let innermost = *scheduler.upcastable_dims().last().expect("two upcastable dims");
    assert!(apply_default_upcast(&mut scheduler));
    assert_eq!(scheduler.applied_opts, vec![Opt::upcast(innermost, 4)]);
}

/// Masked axes collect first and are applied back to front, so an earlier axis is applied last.
#[test_case(&[4], &[Opt::upcast(0, 4)]; "one masked axis takes its full extent")]
#[test_case(&[4, 3], &[Opt::upcast(1, 3), Opt::upcast(0, 4)]; "two masked axes apply in reverse")]
#[test_case(&[7, 7, 7], &[Opt::upcast(1, 7), Opt::upcast(0, 7)]; "the product cap drops the third axis")]
#[test_case(&[8], &[]; "an extent over seven is not a masked candidate")]
fn apply_masked_upcasts_collects_then_applies_in_reverse(shape: &[i64], expected: &[Opt]) {
    let sink = masked_elementwise(shape);
    let (applied, scheduler) = run(sink, Renderer::cpu(), &HeuristicsConfig::default(), |s, _| apply_masked_upcasts(s));
    assert_eq!(applied, !expected.is_empty());
    assert_eq!(scheduler.applied_opts, expected);
}

/// A masked row-major SINK: every axis bounds the `WHERE` condition.
fn masked_elementwise(shape: &[i64]) -> Arc<UOp> {
    let kernel = Ranged::new(&shape.iter().map(|&size| (size, AxisType::Global)).collect::<Vec<_>>());
    let value = kernel.index(&DType::Float32, shape.iter().product(), row_major(kernel.ranges(), shape));
    let bound = UOp::index_const(*shape.iter().max().expect("at least one axis"));
    let condition = kernel.ranges().iter().cloned().fold(UOp::index_const(0), plus).try_cmplt(&bound).expect("cmp");
    kernel.sink_all(UOp::try_where(condition, value, UOp::native_const(0.0f32)).expect("where"))
}

/// `apply_unroll` fully unrolls a short reduce, halves the ladder for a long one.
#[test_case(8, &[Opt::unroll(0, 0)]; "a short reduce unrolls fully")]
#[test_case(32, &[Opt::unroll(0, 0)]; "thirty-two is still short")]
#[test_case(64, &[Opt::unroll(0, 4)]; "a long reduce unrolls by four")]
#[test_case(65, &[]; "an extent four does not divide is left alone")]
fn apply_unroll_follows_the_size_ladder(reduce: i64, expected: &[Opt]) {
    let sink = laid_out_reduce(4, reduce, reduce, 1, DType::Float32);
    let (applied, scheduler) = run(sink, Renderer::cuda(), &HeuristicsConfig::default(), |s, _| apply_unroll(s));
    assert_eq!(applied, !expected.is_empty());
    assert_eq!(scheduler.applied_opts, expected);
}

/// Two reduce axes of three: the full unroll of the innermost leaves the outer one tiny too.
#[test]
fn apply_unroll_unrolls_a_second_tiny_reduce_axis() {
    let kernel = Ranged::new(&[(4, AxisType::Global), (3, AxisType::Reduce), (3, AxisType::Reduce)]);
    let value = kernel.index(&DType::Float32, 36, plus(plus(kernel.range(0), kernel.range(1)), kernel.range(2)));
    let sink = kernel.sink(value.reduce(vec![kernel.range(1), kernel.range(2)].into(), ReduceOp::Add), &[0]);
    let (applied, scheduler) = run(sink, Renderer::cuda(), &HeuristicsConfig::default(), |s, _| apply_unroll(s));
    assert!(applied);
    assert_eq!(scheduler.applied_opts, vec![Opt::unroll(1, 0), Opt::unroll(0, 0)]);
}

#[test]
fn apply_unroll_declines_once_the_upcast_size_is_spent() {
    // An UPCAST of 8 already sits in the kernel: `upcast_size > 4 && has_unroll`
    // is the second guard, and the kernel is left untouched.
    let sink = laid_out_reduce(32, 32, 32, 1, DType::Float32);
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());
    apply_opt(&mut scheduler, &Opt::upcast(0, 8), true).expect("UPCAST 8");
    apply_opt(&mut scheduler, &Opt::unroll(0, 8), true).expect("UNROLL 8");
    let spent = scheduler.applied_opts.clone();
    assert!(!apply_unroll(&mut scheduler));
    assert_eq!(scheduler.applied_opts, spent);
}

// STRIDE / LAYOUT RANKING

/// The stride ranking picks, per round, the stride-0 axis with the fewest and shortest strides.
#[test_case(3, 12, true, &[(2, 3)]; "innermost axis by stride sum, amount three first")]
#[test_case(4, 8, true, &[(3, 4), (2, 4)]; "second round after the shape stays large")]
#[test_case(3, 12, false, &[]; "no stride-0 buffer means no candidate")]
fn heuristic_upcasts_rank_by_strides(axes: usize, size: i64, stride0: bool, expected: &[(usize, usize)]) {
    let (applied, scheduler) =
        run(stride_zero_buffers(axes, size, stride0), Renderer::cpu(), &HeuristicsConfig::default(), |s, _| {
            apply_heuristic_upcasts(s)
        });
    assert_eq!(applied, !expected.is_empty());
    assert_eq!(
        scheduler.applied_opts,
        expected.iter().map(|&(axis, amount)| Opt::upcast(axis, amount)).collect::<Vec<_>>()
    );
}

/// `count_strides` reports, per axis, how many buffers address it and the sum of their strides.
#[test_case(0, (2, 769); "the row axis is strided by the wide buffer and unit by the scale")]
#[test_case(1, (1, 1); "the column axis is unit-stride in its own buffer only")]
fn count_strides_reads_the_linearized_index(axis: usize, expected: (usize, usize)) {
    let scheduler = Scheduler::new(row_scaled(8192, 768, DType::Float32), Renderer::cuda());
    assert_eq!(count_strides(&scheduler, axis), expected);
}

/// `has_matmul_pattern` demands a `REDUCE(ADD)` of a `MUL` whose operands are indexed loads.
#[test]
fn has_matmul_pattern_requires_indexed_mul_operands() {
    let matmul = Scheduler::new(matmul_with(16, 16, 16, DType::Float32, |v| v), Renderer::cuda());
    assert!(has_matmul_pattern(&matmul));
    let constants = Scheduler::new(reduce_sink(&[16], &[16], ReduceOp::Add), Renderer::cuda());
    assert!(!has_matmul_pattern(&constants));
}

/// `is_masked` fires only for an axis the `WHERE` condition reads.
#[test_case(0; "the masked axis")]
#[test_case(1; "the unmasked axis")]
fn is_masked_follows_the_where_condition(axis: usize) {
    let kernel = Ranged::new(&[(4, AxisType::Global), (4, AxisType::Global)]);
    let value = kernel.index(&DType::Float32, 16, plus(times(&kernel.range(0), 4), kernel.range(1)));
    let condition = kernel.range(axis).try_cmplt(&UOp::index_const(2)).expect("cmp");
    let sink = kernel.sink_all(UOp::try_where(condition, value, UOp::native_const(0.0f32)).expect("where"));
    let scheduler = Scheduler::new(sink, Renderer::cuda());
    // Only the axis the condition reads is masked.
    assert_eq!(is_masked(&scheduler, 0), axis == 0);
    assert_eq!(is_masked(&scheduler, 1), axis == 1);
}

/// A reversed axis indexes with a negative constant — `conv_transpose2d` builds
/// exactly that, flipping the kernel — and it is no forward stride at all. The
/// sum used to cast the constant straight to `usize`, so one reversed term
/// wrapped to about 2^64 and the next addition overflowed, which made every
/// kernel carrying a transposed convolution unschedulable.
#[test_case(-1, 1, 0 ; "a reversed row axis counts as no stride")]
#[test_case(768, 1, 768 ; "a forward row axis counts its stride")]
fn a_negative_stride_does_not_wrap(row_stride: i64, reduce_stride: i64, expected_sum: usize) {
    let sink = laid_out_reduce(64, 8, row_stride, reduce_stride, DType::Float16);
    let scheduler = Scheduler::new(sink, Renderer::cuda());

    let (num_strides, sum_strides) = count_strides(&scheduler, 0);

    assert_eq!(num_strides, 1, "the buffer indexes the row axis either way");
    assert_eq!(sum_strides, expected_sum);
}

/// Row reduces: many rows get a wave split off the contiguous reduce axis plus an unroll.
#[test_case(Renderer::cuda(), 8192, 768, 768, 1, &[Opt::group(0, 32), Opt::unroll(1, 4)]; "many rows split a warp off the contiguous reduce")]
#[test_case(Renderer::cuda(), 8192, 3072, 3072, 1, &[Opt::group(0, 32), Opt::unroll(1, 4)]; "a longer row keeps the same split")]
#[test_case(Renderer::cuda(), 131072, 1024, 1024, 1, &[Opt::group(0, 32), Opt::unroll(1, 4)]; "a softmax row reduce splits too")]
#[test_case(Renderer::cuda(), 8192, 32, 32, 1, &[]; "a reduce shorter than one wave stays serial")]
#[test_case(Renderer::cuda(), 8192, 768, 1, 8192, &[]; "a strided reduce axis is left alone")]
#[test_case(Renderer::cuda(), 1024, 768, 768, 1, &[Opt::grouptop(0, 16)]; "few rows keep the shared-block path")]
#[test_case(Renderer::amd_cdna3(), 8192, 768, 768, 1, &[Opt::group(0, 64), Opt::unroll(1, 4)]; "a CDNA wave is sixty-four lanes wide")]
fn row_reduces_split_a_wave_off_a_contiguous_reduce(
    renderer: Renderer,
    rows: i64,
    cols: i64,
    row_stride: i64,
    reduce_stride: i64,
    expected: &[Opt],
) {
    let mut scheduler =
        Scheduler::new(laid_out_reduce(rows, cols, row_stride, reduce_stride, DType::Float16), renderer);
    let config = HeuristicsConfig::builder().build();
    let grouped = try_grouped_reduction(&mut scheduler, &config);
    assert_eq!(grouped || try_warp_row_reduction(&mut scheduler, &config), !expected.is_empty());
    assert_eq!(scheduler.applied_opts, expected);
}

/// The grouped-reduction threshold separates the two paths: at or below it the reduction groups.
#[test_case(4, 4, 128, false, true; "a small output groups")]
#[test_case(32, 32, 128, false, true; "1024 outputs are inside the default threshold")]
#[test_case(32, 32, 128, true, false; "disable_locals drops the threshold to 240")]
#[test_case(1024, 128, 128, false, false; "a large output is above the threshold")]
fn grouped_reduction_threshold_is_config_gated(rows: i64, cols: i64, reduce: i64, disable_locals: bool, grouped: bool) {
    let config = HeuristicsConfig::builder().disable_locals(disable_locals).build();
    let sink = reduce_sink(&[rows, cols], &[reduce], ReduceOp::Add);
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());
    assert_eq!(try_grouped_reduction(&mut scheduler, &config), grouped);
}

/// An elementwise kernel vectorizes along the axis its buffers walk contiguously.
#[test_case(Renderer::cuda(), DType::Float16, &[Opt::upcast(1, 4), Opt::local(1, 16), Opt::local(0, 8)]; "four halves vectorize")]
#[test_case(Renderer::cuda(), DType::Float32, &[Opt::upcast(1, 4), Opt::local(1, 16), Opt::local(0, 8)]; "four floats vectorize")]
#[test_case(Renderer::cuda(), DType::Float64, &[Opt::upcast(1, 3), Opt::local(1, 16), Opt::local(0, 8)]; "four doubles keep the ascending width order")]
#[test_case(Renderer::cpu(), DType::Float16, &[Opt::upcast(1, 3)]; "a CPU kernel keeps the ascending order")]
fn elementwise_kernels_vectorize_and_lane_along_the_contiguous_axis(
    renderer: Renderer,
    dtype: DType,
    expected: &[Opt],
) {
    let mut scheduler = Scheduler::new(row_scaled(8192, 768, dtype), renderer);
    let config = HeuristicsConfig::builder().build();
    assert!(apply_heuristic_upcasts(&mut scheduler));
    apply_local_dims(&mut scheduler, &config);
    assert_eq!(scheduler.applied_opts, expected);
}

/// Where the axes' buffers stride both ways (a transposing copy), or a reduce spans axes, the previous order stands.
#[test_case(false; "a transposing copy")]
#[test_case(true; "a reducing kernel")]
fn a_strided_or_reducing_kernel_keeps_the_previous_local_order(reducing: bool) {
    let sink = if reducing { stencil_reduce(8192, 768, 5) } else { transposing_copy(8192, 768) };
    let (applied, scheduler) = run(sink, Renderer::cuda(), &HeuristicsConfig::default(), apply_local_dims);
    assert!(applied);
    assert_eq!(scheduler.applied_opts, vec![Opt::local(0, 8), Opt::local(1, 16)]);
}

/// LOCAL sizing: a global axis none of the standard sizes divides gets the largest exact divisor (padding when nearly
/// none exists), the lane-efficiency model follows the renderer's wave width, the vocabulary axis is padded beside the
/// row local, and at most three LOCALs apply, lane axis first.
#[test_case(Renderer::cuda(), &[51865], &[Opt::padto(0, 32), Opt::local(0, 32)], None; "whisper vocabulary pads seven elements to 32")]
#[test_case(Renderer::amd_rdna3(), &[51865], &[Opt::padto(0, 32), Opt::local(0, 32)], None; "wave32 pads")]
#[test_case(Renderer::amd_cdna3(), &[51865], &[Opt::local(0, 115)], None; "wave64 keeps the divisor")]
#[test_case(Renderer::cuda(), &[10007], &[Opt::padto(0, 32), Opt::local(0, 32)], None; "prime extent pads to 32")]
#[test_case(Renderer::cuda(), &[385], &[Opt::local(0, 77)], None; "5·7·11 keeps its exact divisor 77")]
#[test_case(Renderer::cuda(), &[12], &[Opt::local(0, 4)], None; "candidate list still wins for 12")]
#[test_case(Renderer::cuda(), &[96], &[Opt::local(0, 32)], None; "candidate list still wins for 96")]
#[test_case(Renderer::cuda(), &[1024], &[Opt::local(0, 32)], None; "candidate list still wins for 1024")]
#[test_case(Renderer::cuda(), &[25], &[Opt::local(0, 25)], None; "tie keeps the exact divisor")]
#[test_case(Renderer::cuda(), &[2, 51865], &[Opt::padto(1, 32), Opt::local(1, 32), Opt::local(0, 2)], Some(&[1621, 32, 2]); "vocabulary axis padded beside the row local")]
#[test_case(Renderer::cuda(), &[8, 8, 8, 8], &[Opt::local(3, 8), Opt::local(1, 2), Opt::local(2, 8)], None; "at most three, lane axis first")]
fn local_dims_pad_and_split_the_lane_axis(
    renderer: Renderer,
    shape: &[i64],
    expected: &[Opt],
    full_shape: Option<&[i64]>,
) {
    let (applied, scheduler) =
        run(elementwise_load(shape, AxisType::Global), renderer, &HeuristicsConfig::default(), apply_local_dims);
    assert!(applied);
    assert_eq!(scheduler.applied_opts, expected);
    if let Some(full_shape) = full_shape {
        assert_eq!(scheduler.full_shape(), full_shape);
    }
}

/// CPU threading pads a loop axis no thread count divides (otherwise it runs serially).
#[test_case(10007, 512, &[Opt::padto(0, 32), Opt::thread(0, 32)]; "prime rows pad to 32 threads")]
#[test_case(51865, 512, &[Opt::thread(0, 5)]; "a dividing count is still preferred")]
#[test_case(96, 65536, &[Opt::thread(0, 32)]; "divisible rows are unchanged")]
#[test_case(10007, 4, &[]; "too little work stays single threaded")]
fn threading_pads_undividable_loop_axes(rows: i64, cols: i64, expected: &[Opt]) {
    let mut renderer = Renderer::cpu();
    // Renderer::cpu() caps threads at the host core count; these expectations need 32.
    renderer.global_max = Some(vec![32]);
    let (applied, scheduler) = run(
        row_reduce(AxisType::Weak, rows, cols, DType::Float32, None),
        renderer,
        &HeuristicsConfig::default(),
        |s, _| apply_threading(s, 32),
    );
    assert_eq!(applied, !expected.is_empty());
    assert_eq!(scheduler.applied_opts, expected);
}

// MATMUL OUTPUT TILING

/// Register blocking wants two output axes at least four wide, and picks the widest divisible factor.
#[test_case(16, 16, 16, true, &[Opt::upcast(0, 8), Opt::upcast(1, 8)]; "square output takes the widest factor")]
#[test_case(14, 6, 16, true, &[Opt::upcast(0, 7), Opt::upcast(1, 6)]; "odd extents pick their own factor")]
#[test_case(16, 16, 16, false, &[]; "output_upcast disabled declines")]
fn apply_matmul_tiling_picks_the_widest_divisible_factor(
    m: i64,
    n: i64,
    k: i64,
    output_upcast: bool,
    expected: &[Opt],
) {
    let config = HeuristicsConfig::builder().output_upcast(output_upcast).build();
    let (applied, scheduler) =
        run(matmul_with(m, n, k, DType::Float16, |v| v), Renderer::cuda(), &config, apply_matmul_tiling);
    assert_eq!(applied, !expected.is_empty());
    assert_eq!(scheduler.applied_opts, expected);
}

/// A single output axis cannot be 2D-tiled, and the legacy alias delegates.
#[test]
fn apply_matmul_tiling_needs_two_output_axes_and_the_alias_delegates() {
    let config = HeuristicsConfig::default();
    let one_axis = row_reduce(AxisType::Global, 16, 16, DType::Float16, None);
    let (applied, _) = run(one_axis, Renderer::cuda(), &config, apply_matmul_tiling);
    assert!(!applied, "one output axis is not a 2D tile");
    let sink = matmul_with(7, 6, 16, DType::Float16, |v| v);
    let (_, tiled) = run(sink.clone(), Renderer::cuda(), &config, apply_matmul_tiling);
    assert!(!tiled.applied_opts.is_empty());
    let (alias, aliased) = run(sink, Renderer::cuda(), &config, apply_matmul_output_upcasting);
    assert!(alias);
    assert_eq!(aliased.applied_opts, tiled.applied_opts);
}

// FULL PIPELINE ORDER

/// The pipeline stops the moment it groups: no masked upcast, unroll, local or thread is tried.
#[test]
fn hand_coded_optimizations_stops_after_grouping() {
    let sink = reduce_sink(&[4, 4, 128], &[128], ReduceOp::Add);
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());
    hand_coded_optimizations(&mut scheduler, &HeuristicsConfig::default());
    assert_eq!(scheduler.applied_opts, vec![Opt::grouptop(0, 16)]);
    assert!(scheduler.group_for_reduces() > 0);
}

/// Tensor cores are tried first and, when they land, no later stage runs.
#[test]
fn hand_coded_optimizations_returns_after_tensor_cores() {
    let mut scheduler = Scheduler::new(matmul_with(64, 64, 64, DType::Float16, |v| v), Renderer::cuda());
    hand_coded_optimizations(&mut scheduler, &HeuristicsConfig::default());
    assert_eq!(scheduler.applied_opts.first().map(|opt| opt.op), Some(OptOps::TC));
    assert!(scheduler.axes_of(&[AxisType::Local]).is_empty(), "post-TC tiling must not stack a block");
}

/// A plain elementwise kernel walks the whole ladder in order: upcast, local.
#[test]
fn hand_coded_optimizations_runs_the_elementwise_ladder_in_order() {
    let mut scheduler = Scheduler::new(elementwise_load(&[1024], AxisType::Global), Renderer::cuda());
    hand_coded_optimizations(&mut scheduler, &HeuristicsConfig::default());
    assert_eq!(scheduler.applied_opts, vec![Opt::upcast(0, 4), Opt::local(0, 32)]);
}

/// A skinny batch of rows through one weight (a decoder step's tokens) is
/// upcast first, so each thread reads a weight row once for every token; the
/// reduce split is a wave on AMD, unrolled to the 16-byte access, and tinygrad's
/// 8x4x4 tile elsewhere.
#[test_case(Renderer::amd_rdna3(), 5, DType::Float16, &[Opt::upcast(0, 5), Opt::group(0, 32), Opt::local(0, 4), Opt::unroll(1, 8)]; "rdna keeps a wave per row group")]
#[test_case(Renderer::amd_cdna3(), 5, DType::Float16, &[Opt::upcast(0, 5), Opt::group(0, 64), Opt::local(0, 4), Opt::unroll(1, 4)]; "cdna keeps a wave per row group and its unroll divides the rest")]
#[test_case(Renderer::amd_rdna3(), 5, DType::Float32, &[Opt::upcast(0, 5), Opt::group(0, 32), Opt::local(0, 4), Opt::unroll(1, 4)]; "the unroll follows the element width")]
#[test_case(Renderer::amd_rdna3(), 1, DType::Float16, &[Opt::group(0, 32), Opt::local(0, 4), Opt::unroll(1, 8)]; "a single row has nothing to upcast")]
#[test_case(Renderer::cuda(), 5, DType::Float16, &[Opt::upcast(0, 5), Opt::group(0, 8), Opt::local(0, 4), Opt::upcast(0, 4)]; "cuda keeps tinygrad's tile")]
#[test_case(Renderer::amd_rdna3(), 64, DType::Float16, &[]; "rows past the upcast limit are not a skinny batch")]
fn matvec_fast_path_upcasts_a_skinny_batch(renderer: Renderer, m: i64, stored: DType, expected: &[Opt]) {
    let (applied, scheduler) = run(
        matmul_accum(m, 1280, 1280, stored, DType::Float32),
        renderer,
        &HeuristicsConfig::default(),
        apply_matvec_fast_path,
    );
    assert_eq!(applied, !expected.is_empty());
    assert_eq!(scheduler.applied_opts, expected);
}

/// The config's matvec fields override the device tile, one at a time.
#[test]
fn matvec_config_overrides_the_device_tile() {
    let config = HeuristicsConfig::builder().threads_per_row(16).rows_per_thread(2).matvec_blocksize(8).build();
    let (applied, scheduler) = run(
        matmul_accum(5, 1280, 1280, DType::Float16, DType::Float32),
        Renderer::amd_rdna3(),
        &config,
        apply_matvec_fast_path,
    );
    assert!(applied);
    assert_eq!(
        scheduler.applied_opts,
        &[Opt::upcast(0, 5), Opt::group(0, 16), Opt::local(0, 8), Opt::upcast(0, 2), Opt::unroll(1, 8)]
    );
}

/// A GROUP the renderer declines — here a shared-memory budget the lane split
/// does not fit — must not cost the row tile: the rest of the fast path still
/// applies and a decode step's GEMV keeps its LOCAL/UPCAST instead of falling
/// through to the generic tail.
#[test_case(Renderer::cuda(), 16, plain_row_reduce, &[Opt::local(0, 4), Opt::upcast(0, 4)]; "cuda keeps the block and the rows")]
#[test_case(Renderer::amd_rdna3(), 512, skinny_batch, &[Opt::upcast(0, 5), Opt::local(0, 4), Opt::unroll(0, 8)]; "rdna keeps the batch upcast and the block")]
fn matvec_fast_path_survives_a_declined_group(
    mut renderer: Renderer,
    shared_max: usize,
    sink: fn() -> Arc<UOp>,
    expected: &[Opt],
) {
    renderer.shared_max = shared_max;
    let (applied, scheduler) = run(sink(), renderer, &HeuristicsConfig::default(), apply_matvec_fast_path);
    assert!(applied, "a declined GROUP must not abort the fast path");
    assert_eq!(scheduler.applied_opts, expected);
    assert!(scheduler.axes_of(&[AxisType::GroupReduce]).is_empty(), "GROUP was declined");
}

/// `y[r] = sum_c x[r, c]`: the fast path's plainest shape.
fn plain_row_reduce() -> Arc<UOp> {
    row_reduce(AxisType::Global, 64, 128, DType::Float32, None)
}

/// A decoder step's projection: five rows through one weight.
fn skinny_batch() -> Arc<UOp> {
    matmul_accum(5, 1280, 1280, DType::Float16, DType::Float32)
}
