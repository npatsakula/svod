use svod_dtype::{AddrSpace, AmdArch, DType};
use svod_ir::{AxisId, AxisType, BinaryOp, ConstValue, Op, RendererDevice, WmmaMetadata, WmmaUpcastAxes};

use super::*;
use crate::Renderer;
use crate::llvm::LlvmTextRenderer;
use svod_ir::ops;

fn slotted_var(name: &str, slot: usize) -> std::sync::Arc<UOp> {
    let var = UOp::variable(name.to_string(), 0, 16, DType::Int32);
    let Op::Param(ops::Param { shape, arg }) = var.op() else { unreachable!() };
    let mut arg = arg.clone();
    arg.slot = slot;
    UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg }), DType::Int32)
}

fn render_linearized(root: std::sync::Arc<UOp>, name: &str) -> crate::RenderedKernel {
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(root).into());
    render(&linear, Some(name)).expect("LLVM render")
}

#[test]
fn llvm_signature_uses_canonical_mixed_param_slot_order() {
    let sink = UOp::sink(vec![
        slotted_var("high", 3),
        UOp::param(2, 1, DType::Float32, None),
        slotted_var("low", 1),
        UOp::param(0, 1, DType::Float32, None),
    ]);
    let rendered = render_linearized(sink, "mixed_abi");
    assert_eq!(rendered.buffer_args.iter().map(|arg| arg.index).collect::<Vec<_>>(), vec![0, 2]);
    assert_eq!(rendered.var_names, vec!["low", "high"]);
    assert!(
        rendered.code.contains(
            "define void @mixed_abi(ptr noalias align 32 %data0, i32 %data1, ptr noalias align 32 %data2, i32 %data3)"
        ),
        "{}",
        rendered.code
    );
}

#[test]
fn test_simple_add() {
    let idx = UOp::const_(DType::Int32, ConstValue::Int(0));
    let load = |slot| {
        let buffer = UOp::param(slot, 1, DType::Float32, None);
        UOp::load().index(UOp::index().buffer(buffer).indices(vec![idx.clone()]).call().unwrap()).call()
    };
    let out_idx =
        UOp::index().buffer(UOp::param(2, 1, DType::Float32, None)).indices(vec![idx.clone()]).call().unwrap();
    let add = UOp::new(Op::Binary(BinaryOp::Add, load(0), load(1)), DType::Float32);

    let result = render_linearized(UOp::sink(vec![out_idx.store(add)]), "test_add");

    for needle in ["define void @test_add(", "noalias align 32", "fadd", "load", "store"] {
        assert!(result.code.contains(needle), "missing {needle}:\n{}", result.code);
    }
    // The kernel is the entry point itself: no `_inner` wrapper, no packed args.
    assert!(!result.code.contains("_inner") && !result.code.contains("ptr %args"), "{}", result.code);
}

#[test]
fn llvm_float_comparisons_use_ordered_predicates_matching_c() {
    let lhs = UOp::param(0, 1, DType::Float32, None);
    let rhs = UOp::param(1, 1, DType::Float32, None);
    let index = UOp::const_(DType::Int32, ConstValue::Int(0));
    let lhs = UOp::load().index(UOp::index().buffer(lhs).indices(vec![index.clone()]).call().unwrap()).call();
    let rhs = UOp::load().index(UOp::index().buffer(rhs).indices(vec![index]).call().unwrap()).call();
    let sink = UOp::sink(vec![
        lhs.try_cmplt(&rhs).unwrap(),
        lhs.try_cmple(&rhs).unwrap(),
        lhs.try_cmpgt(&rhs).unwrap(),
        lhs.try_cmpge(&rhs).unwrap(),
        lhs.try_cmpeq(&rhs).unwrap(),
        lhs.try_cmpne(&rhs).unwrap(),
    ]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink).into());

    for rendered in [
        render(&linear, Some("cpu_float_cmp")).expect("CPU LLVM render"),
        LlvmTextRenderer::amd(AmdArch::Gfx1151).render(&linear, Some("amd_float_cmp")).expect("AMD LLVM render"),
    ] {
        let predicate = |name: &str| rendered.code.lines().any(|l| l.contains("fcmp ") && l.contains(name));
        for name in ["olt", "ole", "ogt", "oge", "oeq", "une"] {
            assert!(predicate(name), "{name}:\n{}", rendered.code);
        }
        for name in ["ult", "ule", "ugt", "uge", "ueq"] {
            assert!(!predicate(name), "unordered predicate {name}:\n{}", rendered.code);
        }
    }
}

fn unclosed_range_linear() -> std::sync::Arc<UOp> {
    let range = UOp::range_axis_dtype(
        UOp::const_(DType::Int32, ConstValue::Int(2)),
        AxisId::Renumbered(0),
        AxisType::Loop,
        DType::Int32,
    );
    UOp::linear(smallvec::smallvec![range.op().sources()[0].clone(), range])
}

fn end_of_non_innermost_range_linear() -> std::sync::Arc<UOp> {
    let bound = UOp::const_(DType::Int32, ConstValue::Int(2));
    let outer = UOp::range_axis_dtype(bound.clone(), AxisId::Renumbered(0), AxisType::Loop, DType::Int32);
    let inner = UOp::range_axis_dtype(bound, AxisId::Renumbered(1), AxisType::Loop, DType::Int32);
    let zero = UOp::native_const(0i32);
    let end_outer = zero.end(smallvec::smallvec![outer.clone()]);
    UOp::linear(smallvec::smallvec![outer.op().sources()[0].clone(), zero, outer, inner, end_outer])
}

#[test_case::test_case(unclosed_range_linear, "unclosed LLVM ranges"; "range never closed")]
#[test_case::test_case(end_of_non_innermost_range_linear, "innermost open range"; "end targets an outer range")]
fn llvm_rejects_malformed_range_nesting(build: fn() -> std::sync::Arc<UOp>, reason: &str) {
    let err = render(&build(), Some("bad_ranges")).expect_err("malformed range nesting must fail");
    assert!(format!("{err}").contains(reason), "unexpected error: {err}");
}

/// Two nested loops storing through the inner one's index.
fn nested_loops_linear() -> std::sync::Arc<UOp> {
    let bound = UOp::const_(DType::Int32, ConstValue::Int(24));
    let outer = UOp::range_axis_dtype(bound.clone(), AxisId::Renumbered(0), AxisType::Loop, DType::Int32);
    let inner = UOp::range_axis_dtype(bound, AxisId::Renumbered(1), AxisType::Loop, DType::Int32);
    let out = UOp::index().buffer(UOp::param(0, 24, DType::Float32, None)).indices(vec![inner.clone()]).call().unwrap();
    let store = out.store(UOp::const_(DType::Float32, ConstValue::Float(1.0)));
    let sink = UOp::sink(vec![store.end(smallvec::smallvec![inner]).end(smallvec::smallvec![outer])]);
    UOp::linear(svod_schedule::linearize_with_cfg(sink).into())
}

/// Every loop the AMD renderer closes carries a loop ID of its own pointing at
/// the unroll hint, on the back edge and nowhere else; CPU loops stay bare.
#[test]
fn amd_loop_back_edges_carry_the_unroll_hint() {
    let linear = nested_loops_linear();
    let amd = LlvmTextRenderer::amd(AmdArch::Gfx1201).render(&linear, Some("amd_loops")).expect("AMD render").code;
    let lines: Vec<&str> = amd.lines().collect();
    let back_edges: Vec<&str> =
        lines.windows(2).filter(|pair| pair[0].starts_with("loop_footer_")).map(|pair| pair[1]).collect();
    assert_eq!(back_edges.len(), 2, "{amd}");
    let mut ids: Vec<&str> =
        back_edges.iter().map(|edge| edge.split_once(", !llvm.loop ").unwrap_or_else(|| panic!("{amd}")).1).collect();
    ids.sort_unstable();
    assert_eq!(ids, ["!1", "!2"], "{amd}");
    assert_eq!(amd.matches("!llvm.loop").count(), 2, "only back edges carry a loop ID:\n{amd}");
    for line in
        ["!0 = !{!\"amdgpu.loop.unroll.threshold\", i32 300}", "!1 = distinct !{!1, !0}", "!2 = distinct !{!2, !0}"]
    {
        assert!(lines.contains(&line), "missing `{line}`:\n{amd}");
    }
    assert_amd_ir_compiles(&amd, "gfx1201");

    let cpu = render(&linear, Some("cpu_loops")).expect("CPU render").code;
    assert!(!cpu.contains("!llvm.loop"), "CPU loops carry no unroll hint:\n{cpu}");
}

/// A loop whose counter indexes a register array keeps LLVM's private-memory
/// boost — the unroll is what promotes the array — while the loop around it
/// does not index registers and carries the hint.
#[test]
fn amd_loops_indexing_registers_stay_unhinted() {
    let bound = UOp::const_(DType::Int32, ConstValue::Int(8));
    let outer = UOp::range_axis_dtype(bound.clone(), AxisId::Renumbered(0), AxisType::Loop, DType::Int32);
    let inner = UOp::range_axis_dtype(bound, AxisId::Renumbered(1), AxisType::Loop, DType::Int32);
    let reg = UOp::buffer(0, 8, DType::Float32, AddrSpace::Reg, None);
    let slot = UOp::index().buffer(reg).indices(vec![inner.clone()]).call().unwrap();
    let store = slot.store(UOp::const_(DType::Float32, ConstValue::Float(1.0)));
    let sink = UOp::sink(vec![store.end(smallvec::smallvec![inner]).end(smallvec::smallvec![outer])]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink).into());

    let amd = LlvmTextRenderer::amd(AmdArch::Gfx1201).render(&linear, Some("amd_reg_loop")).expect("AMD render").code;
    let lines: Vec<&str> = amd.lines().collect();
    let back_edge = |id: &str| {
        let footer = lines.iter().position(|line| *line == format!("loop_footer_{id}:")).expect("footer");
        lines[footer + 1]
    };
    assert_eq!(back_edge("1"), "  br label %loop_latch_1", "{amd}");
    assert_eq!(back_edge("0"), "  br label %loop_latch_0, !llvm.loop !1", "{amd}");
    assert_amd_ir_compiles(&amd, "gfx1201");
}

// ── AMD target tests ───────────────────────────────────────────────────────
//
// These exercise the AMDLLVMRenderer codegen path; `assert_amd_ir_compiles`
// hands the result to clang-amdgcn where a host toolchain is available.

fn render_amd_linearized(root: &std::sync::Arc<svod_ir::UOp>, arch: AmdArch, name: &str) -> crate::RenderedKernel {
    let code_renderer = LlvmTextRenderer::amd(arch);
    let optimizer_renderer = svod_schedule::OptimizerRenderer::for_amd_arch(arch).with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        code_renderer.decompositor(),
        Some(crate::llvm::amd_extra_matcher()),
    );
    let lowered = svod_schedule::apply_post_optimization_with_renderer(root.clone(), &optimizer_renderer)
        .expect("post optimization");
    let linear = svod_ir::UOp::linear(svod_schedule::linearize_with_cfg(lowered).into());
    code_renderer.render(&linear, Some(name)).expect("AMD render")
}

#[test]
fn amd_emits_kernel_abi_and_target_triple() {
    let p = UOp::param(0, 1, DType::Float32, None);
    let idx = UOp::const_(DType::Int32, ConstValue::Int(0));
    let p_idx = UOp::index().buffer(p).indices(vec![idx]).call().unwrap();
    let store = p_idx.store(UOp::const_(DType::Float32, ConstValue::Float(1.0)));

    let result = render_amd_linearized(&UOp::sink(vec![store]), AmdArch::Gfx1100, "amd_smoke");

    for needle in [
        "target triple = \"amdgcn-amd-amdhsa\"",
        "define amdgpu_kernel void @amd_smoke(",
        "\"amdgpu-flat-work-group-size\"=\"1,1\"",
        "alwaysinline",
    ] {
        assert!(result.code.contains(needle), "missing {needle}:\n{}", result.code);
    }
}

/// The AMDGPU backend selects `@llvm.exp2` for f16/f32 but has no f64 lowering
/// ("no libcall available for fexp2"), so only f64 stays on ROCm's `__ocml_*`.
#[test_case::test_case(DType::Float16, "@llvm.exp2.f16", "half")]
#[test_case::test_case(DType::Float32, "@llvm.exp2.f32", "float")]
#[test_case::test_case(DType::Float64, "@__ocml_exp2_f64", "double")]
fn amd_uses_llvm_intrinsics_and_contract_only_float_flags(dtype: DType, callee: &str, llvm_type: &str) {
    let input = UOp::param(0, 1, dtype.clone(), None);
    let output = UOp::param(1, 1, dtype, None);
    let idx = UOp::const_(DType::Int32, ConstValue::Int(0));
    let input_idx = UOp::index().buffer(input).indices(vec![idx.clone()]).call().unwrap();
    let output_idx = UOp::index().buffer(output).indices(vec![idx]).call().unwrap();
    let value = UOp::load().index(input_idx).call();
    let result = value.try_exp2().unwrap().try_div(&value).unwrap();
    let sink = UOp::sink(vec![output_idx.store(result)]);

    let rendered = render_amd_linearized(&sink, AmdArch::Gfx1151, "amd_float_flags");

    assert!(
        rendered.code.contains(&format!("declare {llvm_type} {callee}({llvm_type})")),
        "missing declaration for {callee}:\n{}",
        rendered.code
    );
    assert!(rendered.code.contains(&format!("call {llvm_type} {callee}")), "missing {callee} call:\n{}", rendered.code);
    assert!(
        rendered.code.contains(&format!("fdiv contract {llvm_type}")),
        "missing contract-only fdiv:\n{}",
        rendered.code
    );
    assert!(!rendered.code.contains(" arcp "), "AMD must not permit approximate reciprocal:\n{}", rendered.code);
}

#[test_case::test_case(AmdArch::Gfx942, DType::FP8E4M3)]
#[test_case::test_case(AmdArch::Gfx942, DType::FP8E5M2)]
#[test_case::test_case(AmdArch::Gfx950, DType::FP8E4M3)]
#[test_case::test_case(AmdArch::Gfx950, DType::FP8E5M2)]
fn amd_cdna_ordinary_fp8_alu_widens_without_changing_storage(arch: AmdArch, dtype: DType) {
    let input = UOp::param(0, 2, dtype.clone(), None);
    let output = UOp::param(1, 1, dtype, None);
    let lane = |lane: i32| {
        UOp::load()
            .index(UOp::index().buffer(input.clone()).indices(vec![UOp::native_const(lane)]).call().unwrap())
            .call()
    };
    let output = UOp::index().buffer(output).indices(vec![UOp::native_const(0i32)]).call().unwrap();
    let (first, second) = (lane(0), lane(1));
    let value = first.try_add(&second).unwrap().try_mul(&second).unwrap();
    let rendered = render_amd_linearized(&UOp::sink(vec![output.store(value)]), arch, "amd_fp8_alu");

    assert!(rendered.code.contains("fadd contract float"), "{}", rendered.code);
    assert!(rendered.code.contains("fmul contract float"), "{}", rendered.code);
    assert!(!rendered.code.contains("contract i8"), "fp8 ALU must widen, not compute on i8:\n{}", rendered.code);
    assert!(rendered.code.contains("load <2 x i8>") && rendered.code.contains("store i8"), "{}", rendered.code);
    assert_amd_ir_compiles(&rendered.code, arch.mcpu());
}

#[test]
fn amd_special_emits_workgroup_workitem_intrinsics() {
    // y[gidx0] = x[lidx0]
    let x = UOp::param(0, 1, DType::Float32, None);
    let y = UOp::param(1, 1, DType::Float32, None);
    let g = UOp::special(UOp::const_(DType::Int32, ConstValue::Int(8)), "gidx0".to_string());
    let l = UOp::special(UOp::const_(DType::Int32, ConstValue::Int(4)), "lidx0".to_string());
    let load = UOp::load().index(UOp::index().buffer(x).indices(vec![l]).call().unwrap()).call();
    let store = UOp::index().buffer(y).indices(vec![g]).call().unwrap().store(load);

    let result = render_amd_linearized(&UOp::sink(vec![store]), AmdArch::Gfx1100, "amd_special");

    for needle in [
        "call i32 @llvm.amdgcn.workgroup.id.x()",
        "call i32 @llvm.amdgcn.workitem.id.x()",
        "declare i32 @llvm.amdgcn.workgroup.id.x()",
        // The local bound (4), not the global one, sets the work-group upper bound.
        "\"amdgpu-flat-work-group-size\"=\"1,4\"",
    ] {
        assert!(result.code.contains(needle), "missing {needle}:\n{}", result.code);
    }
}

#[test]
fn amd_barrier_emits_fence_and_s_barrier() {
    let barrier = UOp::noop().barrier(smallvec::SmallVec::new());
    let result = render_amd_linearized(&UOp::sink(vec![barrier]), AmdArch::Gfx1100, "amd_barrier");

    for needle in [
        "fence syncscope(\"workgroup\") release",
        "call void @llvm.amdgcn.s.barrier()",
        "fence syncscope(\"workgroup\") acquire",
        "declare void @llvm.amdgcn.s.barrier()",
    ] {
        assert!(result.code.contains(needle), "missing {needle}:\n{}", result.code);
    }
}

#[test]
fn amd_define_local_emits_addrspace3_module_global() {
    let local = UOp::buffer(42, 16, DType::Float32, AddrSpace::Local, None);
    let result = render_amd_linearized(&UOp::sink(vec![local]), AmdArch::Gfx1100, "amd_lds");
    assert!(
        result.code.contains("@local42 = internal unnamed_addr addrspace(3) global [16 x float] undef"),
        "missing addrspace(3) LDS global:\n{}",
        result.code
    );
}

// ── WMMA emission (parity with tinygrad's AMDLLVMRenderer) ──────────────────

fn amd_wmma_meta(k: usize, in_dt: DType, out_dt: DType, c_count: usize) -> WmmaMetadata {
    let axes = |count| vec![(svod_ir::AxisId::Renumbered(2), count)];
    WmmaMetadata {
        name: "WMMA_test".to_string(),
        dims: (16, 16, k),
        dtype_in: in_dt,
        dtype_out: out_dt,
        device: RendererDevice::AmdRdna3, // unused by the AMD path (keyed on `arch`)
        threads: 32,
        upcast_axes: Some(WmmaUpcastAxes { a: axes(16), b: axes(16), c: axes(c_count) }),
        reduce_axes: vec![],
    }
}

/// WMMA over SSA operands (buffer loads, not the const splats above): A/B are
/// `<ab_lanes x in_dt>` and C is `<c_lanes x out_dt>`.
fn wmma_ssa_sink(
    in_dt: DType,
    ab_lanes: usize,
    out_dt: DType,
    c_lanes: usize,
    meta: WmmaMetadata,
) -> std::sync::Arc<svod_ir::UOp> {
    let load = |slot, dt: DType| {
        let p = UOp::param(slot, 1, dt, None);
        // PARAM and INDEX both carry the storage dtype; INDEX renders as an address.
        let idx = UOp::index().buffer(p).indices(vec![UOp::const_(DType::Int32, ConstValue::Int(0))]).call().unwrap();
        UOp::load().index(idx).call()
    };
    let a = load(0, in_dt.clone()).broadcast(ab_lanes);
    let b = load(1, in_dt).broadcast(ab_lanes);
    let c = load(2, out_dt).broadcast(c_lanes);
    UOp::sink(vec![UOp::wmma(a, b, c, meta)])
}

/// Operand packing and declaration synthesis per (arch, dtype, K), with
/// `llvm-as` and `clang --target=amdgcn` as the oracles. Guards the `<16`
/// declaration-truncation bug and the bf16→i16 / fp8→i64 operand bitcasts.
///
/// `k`/`c_count` come from the metadata; `ab_lanes`/`c_lanes` are the SSA
/// operand widths, which differ from them on the overloaded gfx12 forms.
#[rustfmt::skip]
#[test_case::test_case(AmdArch::Gfx1151, 16, DType::Float16, 16, DType::Float32, 8, 8,
    &["<16 x half> %", "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16(<16 x half>, <16 x half>, <8 x float>)"],
    &[]; "rdna3 f16 to f32")]
#[test_case::test_case(AmdArch::Gfx1151, 16, DType::BFloat16, 16, DType::Float32, 8, 8,
    &["bitcast <16 x bfloat>", "@llvm.amdgcn.wmma.f32.16x16x16.bf16(<16 x i16>, <16 x i16>, <8 x float>)"],
    &[]; "rdna3 bf16 goes on the wire as i16")]
#[test_case::test_case(AmdArch::Gfx1100, 16, DType::BFloat16, 16, DType::BFloat16, 16, 16,
    &["@llvm.amdgcn.wmma.bf16.16x16x16.bf16(<16 x i16>, <16 x i16>, <16 x i16>, i1)", "bitcast <16 x i16>", "to <16 x bfloat>"],
    &[]; "rdna3 bf16 accumulator bitcasts the result back")]
#[test_case::test_case(AmdArch::Gfx1151, 16, DType::Int8, 16, DType::Int32, 8, 8,
    &["bitcast <16 x i8>", "to <4 x i32>", "@llvm.amdgcn.wmma.i32.16x16x16.iu8(i1 true, <4 x i32>", ", i1 true, <4 x i32>", ", <8 x i32>", ", i1 false)"],
    &[]; "rdna3 int8 packs into i32 words with signedness flags")]
#[test_case::test_case(AmdArch::Gfx1200, 16, DType::Float16, 8, DType::Float32, 8, 8,
    &["@llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half>, <8 x half>, <8 x float>)"],
    &[]; "gfx1200 uses llvm overloaded vector suffixes")]
#[test_case::test_case(AmdArch::Gfx1201, 16, DType::Float16, 8, DType::Float32, 8, 8,
    &["@llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half>, <8 x half>, <8 x float>)"],
    &[]; "gfx1201 uses llvm overloaded vector suffixes")]
#[test_case::test_case(AmdArch::Gfx1201, 16, DType::Int8, 8, DType::Int32, 8, 8,
    &["bitcast <8 x i8>", "to <2 x i32>", "@llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v2i32(i1 true, <2 x i32>", ", i1 true, <2 x i32>", ", <8 x i32>", ", i1 false)"],
    &[]; "gfx1201 int8 packs eight lanes into two i32 words")]
#[test_case::test_case(AmdArch::Gfx942, 32, DType::FP8E4M3, 8, DType::Float32, 4, 4,
    &["bitcast <8 x i8>", " to i64", "(i64, i64, <4 x float>, i32, i32, i32)"],
    &[]; "cdna3 fp8 packs eight lanes into one i64")]
#[test_case::test_case(AmdArch::Gfx950, 32, DType::FP8E4M3, 8, DType::Float32, 4, 4,
    &["bitcast <8 x i8>", " to i64", "(i64, i64, <4 x float>, i32, i32, i32)"],
    &[]; "cdna4 fp8 packs eight lanes into one i64")]
// `llc -mcpu=gfx950` accepts `<8 x bfloat>` and rejects `<8 x i16>` for the K=32
// double-rate MFMA, so unlike the K=16 `bf16.1k` form this must not bitcast.
#[test_case::test_case(AmdArch::Gfx950, 32, DType::BFloat16, 8, DType::Float32, 4, 4,
    &["@llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat>, <8 x bfloat>, <4 x float>"],
    &["to <8 x i16>", "(<8 x i16>"]; "cdna4 bf16 k32 takes native bfloat operands")]
#[test_case::test_case(AmdArch::Gfx950, 128, DType::FP8E5M2, 32, DType::Float32, 4, 4,
    &["bitcast <32 x i8>", "to <8 x i32>", "@llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4(", "i32 1, i32 1, i32 0, i32 127, i32 0, i32 127)"],
    &[]; "cdna4 scaled fp8 uses i32 vectors and scale immediates")]
#[allow(clippy::too_many_arguments)]
fn amd_wmma_operand_packing(
    arch: AmdArch,
    k: usize,
    in_dt: DType,
    ab_lanes: usize,
    out_dt: DType,
    c_lanes: usize,
    c_count: usize,
    present: &[&str],
    absent: &[&str],
) {
    let meta = amd_wmma_meta(k, in_dt.clone(), out_dt.clone(), c_count);
    let sink = wmma_ssa_sink(in_dt, ab_lanes, out_dt, c_lanes, meta);
    let result = render_amd_linearized(&sink, arch, "amd_wmma");

    for needle in present {
        assert!(result.code.contains(needle), "missing {needle}:\n{}", result.code);
    }
    for needle in absent {
        assert!(!result.code.contains(needle), "unexpected {needle}:\n{}", result.code);
    }
    assert_llvm_ir_assembles(&result.code);
    assert_amd_ir_compiles(&result.code, arch.mcpu());
}

#[test]
fn amd_gfx1201_direct_fp8_wmma_is_rejected() {
    let meta = amd_wmma_meta(16, DType::FP8E4M3, DType::Float32, 8);
    let sink = wmma_ssa_sink(DType::FP8E4M3, 8, DType::Float32, 8, meta);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink).into());
    let err = LlvmTextRenderer::amd(AmdArch::Gfx1201)
        .render(&linear, Some("amd_direct_fp8"))
        .expect_err("the pinned Tinygrad RDNA4 table has no native FP8 WMMA");
    assert!(err.to_string().contains("no WMMA/MFMA intrinsic for arch=gfx1201"), "unexpected error: {err}");
}

// ── shape-carried lane counts ──────────────────────────────────────────────
//
// This branch keeps the lane count in the UOp shape (scalar dtype + shape `[N]`)
// and gives INDEX the element dtype. Every row below is a value whose LLVM type
// can only be derived from the shape; `llvm-as` is the oracle.

fn f32_param(slot: usize) -> std::sync::Arc<UOp> {
    UOp::param(slot, 8, DType::Float32, None)
}

fn shrink4(src: std::sync::Arc<UOp>, dtype: DType) -> std::sync::Arc<UOp> {
    UOp::new(Op::Shrink(ops::Shrink { src, offsets: UOp::native_const(0i32), sizes: UOp::native_const(4i32) }), dtype)
}

fn element(buffer: std::sync::Arc<UOp>, lane: i64) -> std::sync::Arc<UOp> {
    UOp::index().buffer(buffer).indices(vec![UOp::const_(DType::Index, ConstValue::Int(lane))]).call().unwrap()
}

/// STACK over LOAD(INDEX) lanes: the load is a node, not something the renderer
/// invents at the insertelement.
fn stack_of_loaded_index_lanes_row() -> std::sync::Arc<UOp> {
    let src = f32_param(1);
    let lanes = (0..4).map(|lane| UOp::load().index(element(src.clone(), lane)).call()).collect();
    let out = shrink4(f32_param(0), DType::Float32);
    UOp::sink(vec![out.store(UOp::stack(lanes).detach())])
}

/// CAST of a shape-`[4]` scalar-dtype value: both sides need the shaped type.
fn cast_of_shaped_stack_row() -> std::sync::Arc<UOp> {
    let src = UOp::param(1, 8, DType::UInt32, None);
    let lanes = (0..4).map(|lane| UOp::load().index(element(src.clone(), lane)).call()).collect();
    let out = shrink4(f32_param(0), DType::Float32);
    UOp::sink(vec![out.store(UOp::stack(lanes).cast(DType::Float32))])
}

/// INDEX into a SHRINK: the buffer carries an address space, so this is a GEP
/// over the element dtype, never an extractelement.
fn index_into_shrink_row() -> std::sync::Arc<UOp> {
    let index = element(shrink4(f32_param(1), DType::Float32), 2);
    UOp::sink(vec![element(f32_param(0), 0).store(UOp::load().index(index).call())])
}

/// Gated LOAD whose alt is a shape-`[4]` value: load, alt and phi share one type.
fn gated_load_with_shaped_alt_row() -> std::sync::Arc<UOp> {
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let alt = UOp::const_(DType::Float32, ConstValue::Float(7.0)).broadcast(4);
    let load = UOp::load().index(shrink4(f32_param(1), DType::Float32)).alt(alt).gate(gate).call();
    UOp::sink(vec![shrink4(f32_param(0), DType::Float32).store(load)])
}

/// Gated LOAD with a scalar alt behind a grouped load: the alt splats.
fn gated_load_with_scalar_alt_row() -> std::sync::Arc<UOp> {
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let alt = UOp::const_(DType::Float32, ConstValue::Float(7.0));
    let load = UOp::load().index(shrink4(f32_param(1), DType::Float32)).alt(alt).gate(gate).call();
    UOp::sink(vec![shrink4(f32_param(0), DType::Float32).store(load)])
}

/// INDEX with no indices is the buffer pointer itself — an alias, not a bitcast
/// of a pointer to a float.
fn index_without_indices_row() -> std::sync::Arc<UOp> {
    let base = UOp::index().buffer(f32_param(1)).indices(vec![]).call().unwrap();
    UOp::sink(vec![element(f32_param(0), 0).store(UOp::load().index(base).call())])
}

/// STORE of a 4-lane value through a scalar-width address.
fn store_vector_through_scalar_index_row() -> std::sync::Arc<UOp> {
    let value = UOp::vconst(vec![ConstValue::Float(1.0); 4], DType::Float32);
    UOp::sink(vec![element(f32_param(0), 0).store(value)])
}

#[test_case::test_case(stack_of_loaded_index_lanes_row; "stack of loaded index lanes")]
#[test_case::test_case(cast_of_shaped_stack_row; "cast of shaped stack")]
#[test_case::test_case(index_into_shrink_row; "index into shrink")]
#[test_case::test_case(gated_load_with_shaped_alt_row; "gated load with shaped alt")]
#[test_case::test_case(gated_load_with_scalar_alt_row; "gated load with scalar alt")]
#[test_case::test_case(index_without_indices_row; "index without indices")]
#[test_case::test_case(store_vector_through_scalar_index_row; "store vector through scalar index")]
fn llvm_shaped_values_assemble(build: fn() -> std::sync::Arc<UOp>) {
    assert_llvm_ir_assembles(&render_linearized(build(), "shaped_values").code);
}

/// Pipe `ir` through an `llvm-as` on PATH and assert it parses. Returns without
/// asserting when no usable `llvm-as` is installed.
fn assert_llvm_ir_assembles(ir: &str) {
    use std::io::Write;
    use std::process::{Command, Stdio};

    // Prefer the newest llvm-as available (versioned names first); a bare
    // `llvm-as` on PATH is often an old system build.
    let tool = ["llvm-as-20", "llvm-as-19", "llvm-as-18", "llvm-as-17", "llvm-as-16", "llvm-as-15", "llvm-as"]
        .into_iter()
        .find(|t| {
            Command::new(t)
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        });
    let Some(tool) = tool else { return };

    let run = |src: &str| -> (bool, String) {
        let mut child = Command::new(tool)
            .args(["-o", "/dev/null", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn llvm-as");
        child.stdin.take().unwrap().write_all(src.as_bytes()).expect("write IR to llvm-as");
        let out = child.wait_with_output().expect("wait for llvm-as");
        (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
    };

    // Our IR uses opaque pointers (`ptr`), which LLVM defaults to from v15 on.
    // A tool that can't parse them natively is LLVM ≤ 14 — which also predates
    // the gfx94x/gfx950 MFMA + fp8-conversion intrinsics these tests exercise,
    // so its verdict is unreliable. Skip rather than emit a false failure.
    let probe = "define i32 @p(ptr %x) {\n  %v = load i32, ptr %x\n  ret i32 %v\n}\n";
    if !run(probe).0 {
        return;
    }

    let (ok, stderr) = run(ir);
    assert!(ok, "llvm-as rejected the emitted AMD IR:\n{ir}\n--- llvm-as stderr ---\n{stderr}");
}

/// As above, but through `clang --target=amdgcn-amd-amdhsa`, which also runs
/// instruction selection. Returns without asserting when the host clang has no
/// AMDGPU target.
fn assert_amd_ir_compiles(ir: &str, arch: &str) {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let has_target =
        Command::new("clang").arg("--print-targets").output().ok().filter(|out| out.status.success()).is_some_and(
            |out| String::from_utf8_lossy(&out.stdout).lines().any(|line| line.trim_start().starts_with("amdgcn")),
        );
    if !has_target {
        return;
    }

    let mcpu = format!("-mcpu={arch}");
    let mut child = Command::new("clang")
        .args([
            "-x",
            "ir",
            "-c",
            "-O2",
            "--target=amdgcn-amd-amdhsa",
            &mcpu,
            "-nogpulib",
            "-nogpuinc",
            "-Wno-override-module",
            "-",
            "-o",
            "/dev/null",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn clang");
    child.stdin.take().unwrap().write_all(ir.as_bytes()).expect("write AMD IR");
    let output = child.wait_with_output().expect("wait for clang");
    assert!(
        output.status.success(),
        "clang rejected emitted {arch} IR:\n{}\n{ir}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A typed CUSTOM renders the RHS of an SSA assignment (`%v = <rhs>`); a
/// CUSTOMI registers its formatted text as an operand and is inlined into its
/// consumer instead of getting an instruction of its own.
#[test]
fn custom_renders_an_ssa_rhs_and_customi_inlines_into_its_consumer() {
    let idx = UOp::const_(DType::Int32, ConstValue::Int(0));
    let address = |slot| {
        UOp::index().buffer(UOp::param(slot, 1, DType::Float32, None)).indices(vec![idx.clone()]).call().unwrap()
    };
    let load = UOp::load().index(address(0)).call();
    let custom = UOp::custom(smallvec::smallvec![load], "fmul float {0}, 2.0".to_string(), DType::Float32);
    let typed = render_linearized(UOp::sink(vec![address(1).store(custom)]), "custom_typed");
    assert!(typed.code.contains("= fmul float"), "typed CUSTOM should emit an fmul assignment:\n{}", typed.code);
    assert!(typed.code.contains(", 2.0"), "template literal should survive:\n{}", typed.code);

    let inline = UOp::customi(smallvec::SmallVec::new(), "4.0".to_string(), DType::Float32);
    let inlined = render_linearized(UOp::sink(vec![address(0).store(inline)]), "customi_inline");
    assert!(inlined.code.contains("store float 4.0"), "CUSTOMI text should be inlined:\n{}", inlined.code);
}

#[test]
fn custom_void_hoists_and_deduplicates_declares_into_the_module_prefix() {
    let idx = UOp::const_(DType::Int32, ConstValue::Int(0));
    let out_idx = UOp::index().buffer(UOp::param(0, 1, DType::Float32, None)).indices(vec![idx]).call().unwrap();
    let store = out_idx.store(UOp::const_(DType::Float32, ConstValue::Float(1.0)));
    let custom = UOp::custom(
        smallvec::SmallVec::new(),
        "declare void @llvm.amdgcn.s.barrier()\ncall void @llvm.amdgcn.s.barrier()".to_string(),
        DType::Void,
    );
    let result = render_amd_linearized(&UOp::sink(vec![store, custom]), AmdArch::Gfx942, "custom_void");

    assert_eq!(
        result.code.matches("declare void @llvm.amdgcn.s.barrier()").count(),
        1,
        "the body's declare must be hoisted to the module prefix exactly once:\n{}",
        result.code
    );
    assert!(result.code.contains("call void @llvm.amdgcn.s.barrier()"), "missing call body line:\n{}", result.code);
}

/// Sin must never reach the AMD renderer: the device excludes it from
/// `supported_ops` so the scheduler always decomposes it (`@llvm.sin.f32`
/// lowers to `v_sin_f32` behind an f32 `1/(2π)` pre-scale that is wrong for
/// large arguments). This simulates exactly the drift the backstop guards —
/// an optimizer built with `RendererOps::all()`, so Sin survives to rendering
/// — and asserts the render fails loudly instead of emitting `@llvm.sin`.
#[test]
fn amd_rejects_undecomposed_sin() {
    let input = UOp::param(0, 1, DType::Float32, None);
    let output = UOp::param(1, 1, DType::Float32, None);
    let idx = UOp::const_(DType::Int32, ConstValue::Int(0));
    let input_idx = UOp::index().buffer(input).indices(vec![idx.clone()]).call().unwrap();
    let output_idx = UOp::index().buffer(output).indices(vec![idx]).call().unwrap();
    let value = UOp::load().index(input_idx).call();
    let sink = UOp::sink(vec![output_idx.store(value.try_sin().unwrap())]);

    let arch = AmdArch::Gfx1151;
    let code_renderer = LlvmTextRenderer::amd(arch);
    let optimizer_renderer = svod_schedule::OptimizerRenderer::for_amd_arch(arch).with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        code_renderer.decompositor(),
        Some(crate::llvm::amd_extra_matcher()),
    );
    let lowered =
        svod_schedule::apply_post_optimization_with_renderer(sink, &optimizer_renderer).expect("post optimization");
    let linear = svod_ir::UOp::linear(svod_schedule::linearize_with_cfg(lowered).into());
    let err = code_renderer.render(&linear, Some("amd_sin")).expect_err("un-decomposed Sin must fail the render");
    assert!(err.to_string().contains("un-decomposed Sin"), "{err}");
}
