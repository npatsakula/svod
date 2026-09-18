//! White-box tests over `crate::optimizer::tc`: pattern detection, tensor-core
//! selection (including the fp8 remap and every reject path), the swizzle
//! shapes, and application.

use std::sync::Arc;

use svod_dtype::{AddrSpace, DType, DeviceSpec, ImageKind};
use svod_ir::{AxisId, AxisType, ConstValue, Op, ParamArg, ReduceOp, UOp, ops};
use test_case::test_case;

use crate::optimizer::error::OptError;
use crate::optimizer::renderer::{AMD_CDNA_161632, CUDA_81616, METAL_888, SwizzleAxis, TcTilePolicy, TensorCore};
use crate::optimizer::tc::{TcSelection, apply, apply_with_axis_choice, matching, selection, swizzle, tc_operand};
use crate::optimizer::{Opt, Renderer, Scheduler, prepare_scheduler};
use crate::test::support::prelude::*;
use crate::test::unit::optimizer::kernels::{
    Ranged, matmul_accum, matmul_with, plus, times, two_m_matmul, two_n_matmul,
};

/// A MUL under a REDUCE whose M extent is `m_end` (possibly symbolic).
fn symbolic_m(m_end: Arc<UOp>, n: i64, k: i64) -> Arc<UOp> {
    let kernel = Ranged::new(&[(0, AxisType::Global), (n, AxisType::Global), (k, AxisType::Reduce)]).with_end(0, m_end);
    let (m, n, k) = (
        kernel.range(0).cast(DType::Float32),
        kernel.range(1).cast(DType::Float32),
        kernel.range(2).cast(DType::Float32),
    );
    let product = plus(m, k.clone()).try_mul(&plus(k, n)).expect("mul");
    kernel.sink(product.reduce(vec![kernel.range(2)].into(), ReduceOp::Add), &[0, 1])
}

/// A `MatmulPattern` with the given operand/accumulator dtypes and `choices`
/// axis choices.
fn pattern_for(in0: DType, in1: DType, out: DType, choices: usize) -> matching::MatmulPattern {
    let mut axis_choices = vec![(global_range(16, 1), global_range(16, 0), reduce_range(16, 2))];
    axis_choices.resize(choices, axis_choices[0].clone());
    matching::MatmulPattern {
        reduce_op: UOp::const_(out.clone(), if out.is_float() { ConstValue::Float(0.0) } else { ConstValue::Int(0) }),
        in0: index(buffer_of(16, in0.base()), 0),
        in1: index(buffer_of(16, in1.base()), 0),
        in0_ranges: vec![global_range(16, 0)],
        in1_ranges: vec![global_range(16, 1)],
        red_ranges: vec![reduce_range(16, 2)],
        axis_choices,
    }
}

/// The selected core, or a panic naming the pattern.
fn select(pattern: &matching::MatmulPattern, renderer: &Renderer, tc_select: i32, axis: usize) -> TcSelection {
    selection::select_tensor_core(pattern, renderer, tc_select, axis)
        .expect("selection must not fail")
        .expect("a tensor core must be selected")
}

/// The name of the first WMMA in `scheduler`'s AST.
fn wmma_name(scheduler: &Scheduler) -> String {
    let wmma = first_op(scheduler.ast(), |op| matches!(op, Op::Wmma(..))).expect("a WMMA");
    unwrap_op!(wmma, Op::Wmma(ops::Wmma { metadata, .. }) => metadata).name.clone()
}

// MATCHING

/// Only a value-preserving integer widening is peeled off a MUL operand: the
/// dtype the accumulator sees is what the core must match.
#[test_case(DType::Int8, DType::Int32, true; "int8 to int32")]
#[test_case(DType::UInt8, DType::Int32, true; "uint8 to int32")]
#[test_case(DType::Int8, DType::UInt16, false; "sign change")]
#[test_case(DType::Int32, DType::Int8, false; "narrowing")]
#[test_case(DType::Float16, DType::Float32, false; "float widening")]
fn tc_operand_peels_only_exact_integer_widening(stored: DType, wide: DType, peeled: bool) {
    let operand = index(buffer_of(16, stored.base()), 0).cast(wide.clone());
    assert_eq!(tc_operand(&operand).dtype(), if peeled { stored } else { wide });
}

/// `detect_matmul` needs a `REDUCE(ADD)` of a `MUL` whatever the operand
/// dtype or the elementwise producers on top of it.
#[test_case(false; "a plain buffer-backed matmul")]
#[test_case(true; "a fused operand and a cast above the MUL are peeled")]
fn detect_matmul_matches_a_mul_under_the_reduce(fused: bool) {
    let relu = |value: Arc<UOp>| UOp::alu(svod_ir::BinaryOp::Max, value.clone(), value.const_like(0.0f64));
    let sink = if fused {
        matmul_accum(16, 16, 16, DType::Float16, DType::Float32)
    } else {
        matmul_with(16, 16, 16, DType::Float16, relu)
    };
    let scheduler = Scheduler::new(sink, Renderer::cuda());
    let pattern = matching::detect_matmul(&scheduler).expect("detection must not fail").expect("a matmul");

    assert_eq!((pattern.in0_ranges.len(), pattern.in1_ranges.len(), pattern.red_ranges.len()), (1, 1, 1));
    assert_eq!(pattern.axis_choices.len(), 2, "one choice per way round");
}

/// A concat gate lands between the REDUCE and its MUL, and `matmul_operands`
/// sees through casts and nothing else, so the matmul disappears and every
/// tensor-core opt is declined with no log line to say why. Pre-optimization
/// lifts a gate no reduce range can move back out, which is what keeps a conv
/// fused into a concat eligible.
#[test]
fn a_concat_gate_costs_the_matmul_until_pre_optimization_lifts_it() {
    let kernel = Ranged::new(&[(16, AxisType::Global), (16, AxisType::Global), (16, AxisType::Reduce)]);
    let a = kernel.index(&DType::Float16, 256, plus(times(&kernel.range(0), 16), kernel.range(2)));
    let b = kernel.index(&DType::Float16, 256, plus(times(&kernel.range(2), 16), kernel.range(1)));
    let product = a.try_mul(&b).expect("mul").cast(DType::Float32);
    // Reads the output range, never a reduce range — the concat's own gate.
    let gate = kernel.range(0).lt(&UOp::index_const(8));
    let gated = UOp::try_where(gate, product, UOp::invalid_marker()).expect("gate should build");
    let sink = kernel.sink(gated.reduce(vec![kernel.range(2)].into(), ReduceOp::Add), &[0, 1]);

    let unlifted = Scheduler::new(sink.clone(), Renderer::cuda());
    assert!(matching::detect_matmul(&unlifted).expect("detection must not fail").is_none());

    let lifted = prepare_scheduler(sink, &Renderer::cuda()).expect("pre-optimization");
    assert!(
        matching::detect_matmul(&lifted).expect("detection must not fail").is_some(),
        "lifting the gate should give the matmul back"
    );
}

/// The detected ranges are sorted by axis id descending, so axis choice 0 is
/// the highest-id N; the choices with the operands' roles exchanged follow.
#[test]
fn detect_matmul_orders_the_axis_choices_by_axis_id() {
    let scheduler = Scheduler::new(two_n_matmul(15, 16), Renderer::metal());
    let pattern = matching::detect_matmul(&scheduler).unwrap().expect("two-N matmul detected");
    let extents: Vec<_> = pattern.axis_choices.iter().map(|choice| expect_range_extent(&choice.0)).collect();

    assert_eq!(pattern.axis_choices.len(), 4);
    assert_eq!(extents[..2], [15, 16], "the highest-id N leads");
    assert_eq!(extents[2..], [16, 16], "then in0's only range serves as N against either in1 range");
}

/// Each operand is offered on either side of the core: the swapped choices
/// take N from in0 and M from in1.
#[test]
fn detect_matmul_offers_the_operands_both_ways_round() {
    let scheduler = Scheduler::new(matmul_accum(16, 16, 16, DType::Float16, DType::Float32), Renderer::cuda());
    let pattern = matching::detect_matmul(&scheduler).unwrap().expect("matmul detected");

    assert_eq!(pattern.axis_choices.len(), 2);
    let (n, m, _) = &pattern.axis_choices[0];
    assert!(Arc::ptr_eq(n, &pattern.in1_ranges[0]) && Arc::ptr_eq(m, &pattern.in0_ranges[0]), "in1 takes N first");
    let (n, m, _) = &pattern.axis_choices[1];
    assert!(Arc::ptr_eq(n, &pattern.in0_ranges[0]) && Arc::ptr_eq(m, &pattern.in1_ranges[0]), "then in0 takes N");
}

/// A `None` verdict is a decline, not an error: a kernel with no REDUCE and a
/// REDUCE of a constant are both declined without failing the pass.
#[test_case(true; "no REDUCE at all")]
#[test_case(false; "REDUCE of a constant")]
fn detect_matmul_declines_without_error(no_reduce: bool) {
    let sink =
        if no_reduce { UOp::sink(vec![UOp::native_const(1.0f32)]) } else { reduce_sink(&[16], &[16], ReduceOp::Add) };

    assert!(matching::detect_matmul(&Scheduler::new(sink, Renderer::cuda())).expect("no error").is_none());
}

// SELECTION

/// Selection pins the exact core: auto resolves the winning entry's index in
/// the renderer's table, while an explicit index is kept verbatim.
#[test_case(-1, DType::Float16, DType::Float32, 0; "auto picks the first half to float core")]
#[test_case(0, DType::Float16, DType::Float32, 0; "an explicit index is kept")]
#[test_case(2, DType::Float16, DType::Float16, 2; "an explicit half to half core is kept")]
#[test_case(-1, DType::Int8, DType::Int32, 5; "auto resolves the byte-wide core's table index")]
fn select_tensor_core_pins_the_index(tc_select: i32, dtype_in: DType, dtype_out: DType, expected: usize) {
    let renderer = Renderer::cuda_sm80(false);
    let pattern = pattern_for(dtype_in.clone(), dtype_in, dtype_out.clone(), 1);
    let selection = select(&pattern, &renderer, tc_select, 0);

    assert_eq!(selection.tc_index, expected);
    assert_eq!(renderer.tensor_cores[expected].dtype_out, dtype_out);
    assert_eq!(expect_range_extent(&selection.axes.0), 16);
}

/// FP8 inputs a renderer does not store natively are matched against the f16
/// core, and the pattern's other shapes are unaffected.
#[test_case(Renderer::amd_rdna3(), DType::Float16; "RDNA3 emulates fp8 into fp16")]
#[test_case(Renderer::amd_cdna3(), DType::FP8E4M3; "CDNA3 has a native fp8 core")]
fn select_tensor_core_remaps_unstored_fp8(renderer: Renderer, core_dtype_in: DType) {
    let pattern = pattern_for(DType::FP8E4M3, DType::FP8E4M3, DType::Float32, 1);
    let selection = select(&pattern, &renderer, -1, 0);

    assert_eq!(renderer.tensor_cores[selection.tc_index].dtype_in, core_dtype_in);
}

/// Declines stay `Ok(None)`: a dtype no core matches and an axis choice the
/// pattern does not have are not errors.
#[test_case(DType::Float16, DType::Float16, DType::Float16, 1, 0; "an accumulator dtype no core matches")]
#[test_case(DType::Float16, DType::Float16, DType::Float32, 1, 3; "an axis choice outside the pattern")]
fn select_tensor_core_declines_without_error(in0: DType, in1: DType, out: DType, choices: usize, axis: usize) {
    let pattern = pattern_for(in0, in1, out, choices);

    assert!(
        selection::select_tensor_core(&pattern, &Renderer::intel_xe(), -1, axis)
            .expect("selection must not fail")
            .is_none()
    );
}

/// An image-typed buffer operand never selects a core.
#[test]
fn select_tensor_core_rejects_an_image_operand() {
    let image = DType::Image { kind: ImageKind::Float, shape: vec![2, 8, 4] };
    let mut pattern = pattern_for(DType::Float32, DType::Float32, DType::Float32, 1);
    pattern.in0 = index_of(
        UOp::new(
            Op::Buffer(ops::Buffer {
                shape: svod_ir::shape::shape_to_uop(&smallvec::smallvec![2usize.into(), 8usize.into(), 4usize.into()]),
                arg: ParamArg::buffer(0, image.clone(), AddrSpace::Global, Some(DeviceSpec::Cpu)).into(),
            }),
            image,
        ),
        index_const(0),
    );

    assert!(
        selection::select_tensor_core(&pattern, &Renderer::metal(), -1, 0).expect("selection must not fail").is_none()
    );
}

/// An explicit index outside the renderer's table, or below the auto sentinel,
/// is a validation error rather than a decline.
#[test]
fn select_tensor_core_rejects_a_bad_index() {
    let pattern = pattern_for(DType::Float16, DType::Float16, DType::Float32, 1);

    for tc_select in [9999, -2] {
        let error = selection::select_tensor_core(&pattern, &Renderer::cuda(), tc_select, 0)
            .expect_err("a bad tc_select must be an error");
        assert!(matches!(error, OptError::ValidationFailed { op: "TC", .. }));
    }
}

// SWIZZLE

/// `base_shape` is `tc.opts` in order with only its UPCAST/LOCAL entries kept,
/// and `get_reduce_axes_count` counts its REDUCE entries.
#[test_case(CUDA_81616.build(DType::Float16, DType::Float32),
    &[SwizzleAxis::Upcast(0), SwizzleAxis::Local(0), SwizzleAxis::Local(1), SwizzleAxis::Local(2),
      SwizzleAxis::Local(3), SwizzleAxis::Local(4), SwizzleAxis::Upcast(1),
      SwizzleAxis::Reduce(0), SwizzleAxis::Reduce(1), SwizzleAxis::Reduce(2), SwizzleAxis::Reduce(3)];
    "CUDA m16n8k16")]
#[test_case(METAL_888.build(DType::Float16, DType::Float32),
    &[SwizzleAxis::Upcast(0), SwizzleAxis::Local(0), SwizzleAxis::Local(1), SwizzleAxis::Local(2),
      SwizzleAxis::Local(3), SwizzleAxis::Local(4), SwizzleAxis::Reduce(0), SwizzleAxis::Reduce(1),
      SwizzleAxis::Reduce(2)];
    "Metal m8n8k8")]
#[test_case(AMD_CDNA_161632.build(DType::Float16, DType::Float32),
    &[SwizzleAxis::Local(0), SwizzleAxis::Local(1), SwizzleAxis::Local(2), SwizzleAxis::Local(3),
      SwizzleAxis::Upcast(0), SwizzleAxis::Upcast(1), SwizzleAxis::Local(4), SwizzleAxis::Local(5),
      SwizzleAxis::Reduce(0), SwizzleAxis::Reduce(1), SwizzleAxis::Reduce(2), SwizzleAxis::Reduce(3),
      SwizzleAxis::Reduce(4)];
    "CDNA m16n16k32")]
fn base_shape_is_the_opt_sequence(tensor_core: TensorCore, expected: &[SwizzleAxis]) {
    assert_eq!(swizzle::base_shape(&tensor_core), expected);
    assert_eq!(
        swizzle::get_reduce_axes_count(&tensor_core),
        expected.iter().filter(|a| matches!(a, SwizzleAxis::Reduce(_))).count()
    );
}

/// The permutation is a fixed mapping, not merely a set of valid indices.
#[test]
fn permutes_for_shape_is_the_exact_swizzle() {
    let tensor_core = CUDA_81616.build(DType::Float16, DType::Float32);

    assert_eq!(
        swizzle::permutes_for_shape(&tensor_core, &swizzle::base_shape(&tensor_core)),
        (vec![6, 8, 9, 3, 4, 5, 10, 1, 2, 0, 7], vec![7, 8, 9, 0, 1, 2, 10, 3, 4, 5, 6])
    );
}

// APPLICATION

/// A divisible matmul applies without padding and leaves a WMMA whose axes are
/// the returned RANGEs.
#[test]
fn apply_records_wmma_and_returns_the_three_axes() {
    let mut scheduler = Scheduler::new(matmul_accum(16, 16, 16, DType::Float16, DType::Float32), Renderer::metal());
    let axes = apply(&mut scheduler, -1, 0, 1).expect("TC apply should succeed");
    assert!(axes.iter().all(|axis| matches!(axis.op(), Op::Range(..))));
    assert!(has_op(scheduler.ast(), |op| matches!(op, Op::Wmma(..))));
    assert_eq!(
        count(scheduler.ast(), |node| matches!(node.op(), Op::Ternary(svod_ir::TernaryOp::Where, ..))),
        0,
        "a divisible matmul is never padded, so no mask is introduced"
    );
}

/// `use_tensor_cores = 2` is the shape-only mode: the core's axis splits land,
/// but no WMMA is built for them.
#[test]
fn apply_in_shape_only_mode_splits_the_axes_without_a_wmma() {
    let renderer = Renderer::metal();
    let mut scheduler = Scheduler::new(matmul_accum(16, 16, 16, DType::Float16, DType::Float32), renderer.clone());
    let axes = apply(&mut scheduler, -1, 0, 2).expect("shape-only TC apply should succeed");

    assert!(axes.iter().all(|axis| matches!(axis.op(), Op::Range(..))));
    assert!(!has_op(scheduler.ast(), |op| matches!(op, Op::Wmma(..))), "shape-only mode builds no WMMA");

    let core = &renderer.tensor_cores[scheduler.selected_tc_index.expect("the selected core is recorded")];
    let shape = swizzle::base_shape(core);
    let swizzled = |kind: fn(&SwizzleAxis) -> bool| shape.iter().filter(|axis| kind(axis)).count();
    let extents = |kind: AxisType| {
        scheduler.rngs().iter().filter(|r| range_axis_type(r) == kind).map(expect_range_extent).collect::<Vec<_>>()
    };

    assert_eq!(extents(AxisType::Upcast).len(), swizzled(|axis| matches!(axis, SwizzleAxis::Upcast(_))));
    assert_eq!(extents(AxisType::Unroll).len(), swizzle::get_reduce_axes_count(core));
    // The swizzle's LOCAL splits are expressed through the single WARP axis.
    assert_eq!(extents(AxisType::Local), Vec::<i64>::new());
    assert_eq!(extents(AxisType::Warp), vec![core.threads as i64]);
    assert_eq!(1usize << swizzled(|axis| matches!(axis, SwizzleAxis::Local(_))), core.threads);
    // Each tiled axis is left divided down by the core's tile.
    assert_eq!(extents(AxisType::Global), vec![16 / core.dims.0 as i64, 16 / core.dims.1 as i64]);
    assert_eq!(extents(AxisType::Reduce), vec![16 / core.dims.2 as i64]);
}

/// `tc_opt = 2` pads each non-divisible dimension to the core's tile, as long
/// as the tail stays inside the padding budget — or the kernel is compute-bound
/// enough that the padded core still beats the scalar loop by a wide margin.
#[test_case(15, 16, 16, 2, 2; "one padded axis")]
#[test_case(30, 30, 30, 2, 6; "every axis padded")]
#[test_case(5, 16, 16, 3, 2; "unbounded padding tiles a beam width of five")]
fn apply_pads_non_divisible_dimensions(m: i64, n: i64, k: i64, tc_opt: usize, masks: usize) {
    let mut scheduler = Scheduler::new(matmul_accum(m, n, k, DType::Float16, DType::Float32), Renderer::cuda());

    assert_eq!(
        count(scheduler.ast(), |node| matches!(node.op(), Op::Ternary(svod_ir::TernaryOp::Where, ..))),
        0,
        "an unpadded kernel has no mask"
    );
    apply_with_axis_choice(&mut scheduler, 0, tc_opt, 1, None).expect("padding levels should pad");
    assert_eq!(count(scheduler.ast(), |node| matches!(node.op(), Op::Ternary(svod_ir::TernaryOp::Where, ..))), masks);
    assert!(has_op(scheduler.ast(), |op| matches!(op, Op::Wmma(..))));
}

/// A non-divisible dimension needs `tc_opt = 2`: without it the apply fails
/// with the divisibility reason, and a tail beyond the padding budget is
/// refused before PADTO ever runs: a beam-width M padded to a whole tile
/// multiplies a memory-bound GEMV's work for nothing.
#[test_case(15, 16, 16, 1, "dimension not divisible by tensor core size", "TC"; "15 is not divisible by 16")]
#[test_case(4, 16, 16, 2, "padding to the tensor-core tile would add too much work", "TC"; "4 -> 16 is a 4x work increase")]
#[test_case(5, 16, 16, 2, "padding to the tensor-core tile would add too much work", "TC"; "a beam width of 5 never pays for a 16-row tile")]
#[test_case(16, 12, 16, 2, "padding to the tensor-core tile would add too much work", "TC"; "12 -> 16 is a third more work")]
#[test_case(2, 16, 16, 3, "padding would add more than 4x work", "PADTO"; "unbounded padding keeps only the 4x limit, on either side")]
fn apply_rejects_a_non_divisible_dimension(m: i64, n: i64, k: i64, tc_opt: usize, reason: &str, op: &str) {
    let mut scheduler = Scheduler::new(matmul_accum(m, n, k, DType::Float16, DType::Float32), Renderer::cuda());
    let error = apply_with_axis_choice(&mut scheduler, 0, tc_opt, 1, None).expect_err("not divisible");
    assert!(
        matches!(error, OptError::ValidationFailed { op: actual, reason: actual_reason }
        if actual == op && actual_reason == reason),
        "{error:?}"
    );
    assert!(!has_op(scheduler.ast(), |op| matches!(op, Op::Wmma(..))));
}

/// The budget is for memory-bound kernels. A convolution over a 20x20 output
/// shares every weight across 400 rows, so padding one spatial axis to the tile
/// (20 -> 32, 1.6x the MACs) still leaves the core far ahead of the scalar loop
/// it displaces; the same 20-row tail on a 16-column GEMV is only more work.
#[test_case(768, 6912, true; "a 20x20 conv output pads past the budget")]
#[test_case(12, 32, false; "a memory-bound kernel keeps to the budget on either side")]
fn the_pad_budget_yields_to_a_compute_bound_kernel(n: i64, k: i64, pads: bool) {
    let mut scheduler = Scheduler::new(two_m_matmul(20, 20, n, k), Renderer::cuda());
    let result = apply_with_axis_choice(&mut scheduler, 0, 2, 1, None);
    assert_eq!(result.is_ok(), pads, "{result:?}");
    assert_eq!(has_op(scheduler.ast(), |op| matches!(op, Op::Wmma(..))), pads);
    if !pads {
        let error = result.expect_err("refused");
        assert!(
            matches!(&error, OptError::ValidationFailed { reason, .. } if *reason == "padding to the tensor-core tile would add too much work"),
            "{error:?}"
        );
    }
}

/// A conv's 20-wide spatial axis is M when the input is the MUL's first
/// operand: 20 pads to 32 on CUDA's 16-row M side, past the budget, and the
/// kernel lost its tensor core. With the operands the other way round the
/// out-channel axis takes M and the spatial axis pads to 24 on the 8-wide N
/// side. The swapped choice commutes the MUL, so A still carries M.
#[test]
fn a_swapped_choice_tiles_an_axis_that_only_fits_the_n_side() {
    let sink = matmul_accum(20, 768, 64, DType::Float16, DType::Float32);

    let mut direct = Scheduler::new(sink.clone(), Renderer::cuda());
    let error = apply_with_axis_choice(&mut direct, 0, 2, 1, Some(0)).expect_err("20 -> 32 is past the budget");
    assert!(matches!(error, OptError::ValidationFailed { reason, .. }
        if reason == "padding to the tensor-core tile would add too much work"));

    let mut swapped = Scheduler::new(sink.clone(), Renderer::cuda());
    let [n, m, _] = apply_with_axis_choice(&mut swapped, 0, 2, 1, None).expect("the swapped choice pads 20 -> 24");
    assert_eq!(count(swapped.ast(), |node| matches!(node.op(), Op::Ternary(svod_ir::TernaryOp::Where, ..))), 2);
    let wmma = first_op(swapped.ast(), |op| matches!(op, Op::Wmma(..))).expect("a WMMA");
    let Op::Wmma(ops::Wmma { a, b, .. }) = wmma.op() else { unreachable!() };
    let axis_id = |range: &Arc<UOp>| unwrap_op!(range, Op::Range(ops::Range { axis_id, .. }) => axis_id).clone();
    let carries = |operand: &Arc<UOp>, range: &Arc<UOp>| {
        let want = axis_id(range);
        operand
            .toposort()
            .iter()
            .any(|node| matches!(node.op(), Op::Range(ops::Range { axis_id, .. }) if *axis_id == want))
    };
    assert!(carries(a, &m) && !carries(a, &n), "A carries M (the 768 out-channels), not N");
    assert!(carries(b, &n) && !carries(b, &m), "B carries N (the padded 20 -> 24 spatial axis), not M");
}

/// `C[n,k] = Σ_d A[n,d] · B[d,k]`, optionally reduced again over `k`: the matmul's
/// own output axis is then a REDUCE axis of the same kernel.
fn reduce_after_matmul(n: i64, k: i64, d: i64, fused: bool) -> Arc<UOp> {
    let axis = |end, id, ty| UOp::range_axis(UOp::index_const(end), AxisId::Renumbered(id), ty);
    let k_type = if fused { AxisType::Reduce } else { AxisType::Global };
    let (n_r, k_r, d_r) = (axis(n, 0, AxisType::Global), axis(k, 1, k_type), axis(d, 2, AxisType::Reduce));
    let half = |r: &Arc<UOp>| r.cast(DType::Float16);
    let product = half(&n_r).try_add(&half(&d_r)).unwrap().try_mul(&half(&d_r).try_add(&half(&k_r)).unwrap()).unwrap();
    let matmul = product.cast(DType::Float32).reduce(smallvec::smallvec![d_r], ReduceOp::Add);
    if fused {
        UOp::sink(vec![matmul.reduce(smallvec::smallvec![k_r], ReduceOp::Add), n_r])
    } else {
        UOp::sink(vec![matmul, n_r, k_r])
    }
}

/// A matmul whose output axis a downstream reduce still sums over gets no tensor
/// core from any caller: the WMMA would spread that axis over the warp's lanes and
/// the outer sum would never cross them (the YOLO26 class tail fused with its 1x1
/// summed a quarter of its channels under a BEAM plan). The heuristics used to
/// decline this shape on their own; a replayed plan went straight to the core.
#[test_case(Some(0); "the beam's default axis choice")]
#[test_case(None; "every axis choice")]
fn a_reduced_matmul_output_refuses_the_tensor_core(axis_choice: Option<usize>) {
    let mut fused = Scheduler::new(reduce_after_matmul(64, 384, 384, true), Renderer::cuda());
    let error = apply_with_axis_choice(&mut fused, -1, 2, 1, axis_choice).expect_err("refused");
    assert!(
        matches!(&error, OptError::ValidationFailed { reason, .. } if *reason == "a matmul output axis is a reduce axis"),
        "{error:?}"
    );
    assert!(!has_op(fused.ast(), |op| matches!(op, Op::Wmma(..))));

    let mut plain = Scheduler::new(reduce_after_matmul(64, 384, 384, false), Renderer::cuda());
    apply_with_axis_choice(&mut plain, -1, 2, 1, axis_choice).expect("the same matmul unfused takes the core");
    assert!(has_op(plain.ast(), |op| matches!(op, Op::Wmma(..))));
}

/// Automatic core selection trials every core; the cores whose dtypes cannot
/// match an f16 matmul must not drown out why the one eligible core declined.
#[test]
fn auto_selection_reports_the_eligible_cores_rejection() {
    let mut scheduler = Scheduler::new(matmul_accum(15, 16, 16, DType::Float16, DType::Float32), Renderer::amd_rdna4());
    let error = apply_with_axis_choice(&mut scheduler, -1, 1, 1, None).expect_err("15 rows never tile at 16");
    assert!(
        matches!(error, OptError::ValidationFailed { op: "TC", reason: "dimension not divisible by tensor core size" }),
        "{error:?}"
    );
}

/// Symbolic extents never take the tensor core: PADTO cannot pad an unknown
/// extent, and even a symbolic multiple of the tile stays bare. `tc_opt = 2`
/// names the symbolic gate; without padding the extent is simply one the
/// divisibility check cannot read, and it rejects under that reason.
#[test_case(false, 0, "dimension not divisible by tensor core size"; "a bare variable with no padding")]
#[test_case(false, 2, "symbolic dimension cannot use tensor cores"; "a bare variable with padding")]
#[test_case(true, 0, "dimension not divisible by tensor core size"; "a divisible symbolic extent with no padding")]
#[test_case(true, 2, "symbolic dimension cannot use tensor cores"; "a divisible symbolic extent with padding")]
fn apply_rejects_a_symbolic_dimension(divisible: bool, tc_opt: usize, reason: &str) {
    let name = if divisible { "ts" } else { "M" };
    let variable = UOp::variable(name.into(), 1, 1 << 16, DType::Int32);
    let end = if divisible { times(&variable, 16) } else { variable };
    // `METAL_888.build(Float32, Float32)` is core 0, the one core whose dtypes
    // this f32 kernel matches: pinning it keeps the reject reason the symbolic
    // gate's own, not a later core's dtype mismatch.
    let mut scheduler = Scheduler::new(symbolic_m(end, 16, 16), Renderer::metal());
    let error = apply_with_axis_choice(&mut scheduler, 0, tc_opt, 1, None).expect_err("a symbolic extent never TCs");
    assert!(matches!(error, OptError::ValidationFailed { op: "TC", reason: actual } if actual == reason), "{error:?}");
    assert!(!has_op(scheduler.ast(), |op| matches!(op, Op::Wmma(..))));
}

/// The guards in front of the pattern search, each with its own reason.
#[test_case(-2, 1, 1, None, "tc_select must be >= -1"; "tc_select below the auto sentinel")]
#[test_case(-1, 1, 0, None, "use_tensor_cores must be 1 or 2"; "tensor cores switched off entirely")]
#[test_case(-1, 1, 3, None, "use_tensor_cores must be 1 or 2"; "only 1 and 2 are accepted")]
#[test_case(-1, 4, 1, None, "tc_opt must be 0, 1, 2, or 3"; "tc_opt above three")]
#[test_case(-1, 1, 1, Some(5), "axis choice out of bounds"; "past the last axis choice")]
#[test_case(-1, 1, 1, Some(2), "axis choice out of bounds"; "a choice the pattern does not have")]
fn apply_validates_its_arguments(
    tc_select: i32,
    tc_opt: usize,
    use_tensor_cores: usize,
    axis_choice: Option<usize>,
    reason: &'static str,
) {
    let mut scheduler = Scheduler::new(matmul_accum(16, 16, 16, DType::Float16, DType::Float32), Renderer::cuda());
    let error = apply_with_axis_choice(&mut scheduler, tc_select, tc_opt, use_tensor_cores, axis_choice)
        .expect_err("invalid arguments");
    assert!(matches!(error, OptError::ValidationFailed { op: "TC", reason: actual } if actual == reason));
}

/// TC must be the first opt on a scheduler, so a partially optimized kernel is
/// rejected, and a graph with no matmul has nothing to select for.
#[test]
fn apply_requires_the_tensor_core_to_be_first() {
    let mut scheduler = Scheduler::new(matmul_accum(16, 16, 16, DType::Float16, DType::Float32), Renderer::cuda());
    scheduler.applied_opts.push(Opt::upcast(0, 2));
    let error = apply(&mut scheduler, -1, 1, 1).expect_err("a prior opt must be rejected");
    assert!(matches!(error, OptError::ValidationFailed { op: "TC", reason: "tensor core opts must be first" }));
}

#[test]
fn apply_rejects_a_graph_without_a_matmul() {
    let mut scheduler = Scheduler::new(UOp::sink(vec![UOp::native_const(1.0f32)]), Renderer::cuda());
    let error = apply(&mut scheduler, -1, 0, 1).expect_err("no matmul pattern");
    assert!(matches!(error, OptError::ValidationFailed { op: "TC", reason: "no matmul pattern detected" }));
}

/// An explicit axis choice that cannot divide is not retried; the auto path is,
/// which is how an odd N still reaches a core.
#[test]
fn apply_with_axis_choice_retries_only_when_asked() {
    let sink = two_n_matmul(15, 16);
    let renderer = Renderer::metal();

    let mut failing = Scheduler::new(sink.clone(), renderer.clone());
    assert!(apply_with_axis_choice(&mut failing, -1, 1, 1, Some(0)).is_err(), "N = 15 must fail");

    let mut passing = Scheduler::new(sink.clone(), renderer.clone());
    apply_with_axis_choice(&mut passing, -1, 1, 1, Some(1)).expect("N = 16 must pass");
    assert!(has_op(passing.ast(), |op| matches!(op, Op::Wmma(..))));

    let mut automatic = Scheduler::new(sink, renderer);
    assert!(apply_with_axis_choice(&mut automatic, -1, 1, 1, None).is_ok(), "auto must recover by retrying");
}

/// An explicit TC whose K extent does not divide fails; auto-selection falls
/// through to the later core that fits.
#[test]
fn apply_retries_later_tensor_core_candidates() {
    let renderer = Renderer::cuda_sm80(false);
    let cores = &renderer.tensor_cores;
    let core = |k| {
        cores.iter().position(|tc| tc.dtype_in == DType::Float16 && tc.dtype_out == DType::Float32 && tc.dims.2 == k)
    };
    let (k16, k8) = (core(16).expect("an SM80 fp16 to fp32 core"), core(8).expect("the K = 8 sibling"));
    assert!(k8 > k16);
    let sink = matmul_accum(16, 8, 8, DType::Float16, DType::Float32);

    let mut failing = Scheduler::new(sink.clone(), renderer.clone());
    assert!(apply_with_axis_choice(&mut failing, k16 as i32, 1, 1, Some(0)).is_err(), "K = 8 fails the K = 16 core");

    let mut passing = Scheduler::new(sink.clone(), renderer.clone());
    assert!(apply_with_axis_choice(&mut passing, k8 as i32, 1, 1, Some(0)).is_ok());

    let mut automatic = Scheduler::new(sink, renderer);
    assert!(apply_with_axis_choice(&mut automatic, -1, 1, 1, Some(0)).is_ok(), "auto must try the K = 8 core");
}

/// The WMMA name carries the dtype pair, which is how codegen picks the
/// fragment.
#[test_case(DType::Float16, DType::Float16, 16, "WMMA_8_16_16_half_half"; "half in, half out")]
#[test_case(DType::Float16, DType::Float32, 16, "WMMA_8_16_16_half_float"; "half in, float out")]
#[test_case(DType::BFloat16, DType::Float32, 16, "WMMA_8_16_16_bfloat_float"; "bfloat in, float out")]
#[test_case(DType::Int8, DType::Int32, 32, "WMMA_8_16_32_int8_int"; "int8 in, int out")]
fn apply_names_the_wmma_after_the_dtype_pair(dtype_in: DType, dtype_out: DType, k: i64, name: &str) {
    let mut scheduler = Scheduler::new(matmul_accum(16, 16, k, dtype_in, dtype_out), Renderer::cuda_sm80(false));

    apply(&mut scheduler, -1, 0, 1).expect("a CUDA core for this dtype pair");
    assert_eq!(wmma_name(&scheduler), name);
}

/// Tensor cores cannot be combined with grouping: both need the reduce axis.
#[test]
fn apply_rejects_a_group_after_tensor_cores() {
    let mut scheduler = Scheduler::new(matmul_accum(16, 16, 16, DType::Float16, DType::Float32), Renderer::metal());
    crate::optimizer::apply_opt(&mut scheduler, &Opt::tc(None, -1, 0, 1), true).expect("TC");
    let error =
        crate::optimizer::apply_opt(&mut scheduler, &Opt::group(0, 2), true).expect_err("GROUP after TC is rejected");
    assert!(matches!(error, OptError::ValidationFailed { reason: "no grouping with tensor cores", .. }));
}

/// The tile policy is per target: only the CUDA families take the lane budget,
/// every other backend keeps tinygrad's fixed step.
#[test_case(Renderer::cuda(), true; "sm80 takes the lane budget")]
#[test_case(Renderer::cuda_sm75(), true; "sm75 takes the lane budget")]
#[test_case(Renderer::cuda_sm89(false), true; "sm89 takes the lane budget")]
#[test_case(Renderer::metal(), false; "metal keeps the fixed step")]
#[test_case(Renderer::amd_rdna3(), false; "rdna3 keeps the fixed step")]
#[test_case(Renderer::amd_cdna3(), false; "cdna3 keeps the fixed step")]
#[test_case(Renderer::intel_xe(), false; "intel xe keeps the fixed step")]
fn tc_tile_policy_follows_the_target(renderer: Renderer, lane_budget: bool) {
    assert_eq!(matches!(renderer.tc_tile_policy(), TcTilePolicy::LaneBudget { accum_max: 128 }), lane_budget);
}
