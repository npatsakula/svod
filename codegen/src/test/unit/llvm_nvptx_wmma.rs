use svod_ir::{ConstValue, RendererDevice, WmmaUpcastAxes};

use super::*;
use crate::Renderer;
use crate::llvm::LlvmTextRenderer;
use crate::llvm::nvptx::ops::tests::{SM75, SM86, SM89, assert_ptx_compiles, render_nvptx_linearized};
use ScalarDType::{BFloat16, Bool, FP8E4M3, FP8E5M2, Float16, Float32, Int8, Int32};

fn sm(major: u8, minor: u8) -> CudaArch {
    CudaArch::from_compute_capability(major, minor)
}

/// Which `@llvm.nvvm.mma.*` form each `(arch, in, acc, K)` selects.
///
/// `None` means "no `mma.sync` form", which forces upstream decomposition —
/// naming an unknown intrinsic is strictly worse, since LLVM lowers it to a
/// silent extern call. The minimum capability per shape follows the PTX ISA
/// and was verified with ptxas.
#[test_case::test_case(sm(7, 5), Float16, Float32, 8 => Some("llvm.nvvm.mma.m16n8k8.row.col.f32.f32".into()); "turing f16 k8")]
#[test_case::test_case(sm(7, 5), Float16, Float16, 8 => Some("llvm.nvvm.mma.m16n8k8.row.col.f16.f16".into()); "turing f16 acc k8")]
#[test_case::test_case(sm(7, 5), Float16, Float32, 16 => None; "k16 needs ampere")]
#[test_case::test_case(sm(7, 5), BFloat16, Float32, 16 => None; "bf16 needs ampere")]
#[test_case::test_case(sm(7, 5), Float32, Float32, 8 => None; "tf32 needs ampere")]
#[test_case::test_case(sm(7, 0), Float16, Float32, 8 => None; "volta has no m16n8 shape")]
#[test_case::test_case(sm(8, 6), Float16, Float32, 16 => Some("llvm.nvvm.mma.m16n8k16.row.col.f32.f32".into()))]
#[test_case::test_case(sm(8, 6), Float16, Float16, 16 => Some("llvm.nvvm.mma.m16n8k16.row.col.f16.f16".into()))]
#[test_case::test_case(sm(8, 6), Float16, Float32, 8 => Some("llvm.nvvm.mma.m16n8k8.row.col.f32.f32".into()))]
#[test_case::test_case(sm(8, 6), BFloat16, Float32, 16 => Some("llvm.nvvm.mma.m16n8k16.row.col.bf16".into()))]
#[test_case::test_case(sm(8, 6), BFloat16, BFloat16, 16 => None; "no bf16 accumulator")]
#[test_case::test_case(sm(8, 6), Float32, Float32, 8 => Some("llvm.nvvm.mma.m16n8k8.row.col.tf32".into()))]
#[test_case::test_case(sm(8, 6), Float32, Float32, 16 => None; "tf32 is k8 only")]
#[test_case::test_case(sm(8, 6), Int8, Int32, 32 => Some("llvm.nvvm.mma.m16n8k32.row.col.satfinite.s8".into()))]
#[test_case::test_case(sm(8, 6), Int8, Int32, 16 => None; "int8 is k32 only")]
#[test_case::test_case(sm(8, 6), FP8E4M3, Float32, 32 => None; "fp8 needs ada")]
#[test_case::test_case(sm(8, 9), FP8E4M3, Float32, 32 => Some("llvm.nvvm.mma.m16n8k32.row.col.f32.e4m3.e4m3.f32".into()))]
#[test_case::test_case(sm(8, 9), FP8E5M2, Float32, 32 => Some("llvm.nvvm.mma.m16n8k32.row.col.f32.e5m2.e5m2.f32".into()))]
#[test_case::test_case(sm(9, 0), FP8E4M3, Float32, 16 => None; "fp8 is k32 only")]
#[test_case::test_case(sm(12, 0), Float16, Float32, 16 => Some("llvm.nvvm.mma.m16n8k16.row.col.f32.f32".into()); "open-ended arch")]
#[test_case::test_case(sm(8, 6), Float16, Float32, 32 => None; "f16 has no k32")]
#[test_case::test_case(sm(8, 6), Bool, Float32, 16 => None; "no mma form takes bool")]
fn mma_selection(arch: CudaArch, in_dt: ScalarDType, acc_dt: ScalarDType, k: usize) -> Option<String> {
    resolve_mma(arch, in_dt, acc_dt, (8, 16, k)).map(|mma| mma.intrinsic)
}

#[test]
fn mma_selection_requires_the_m16n8_shape() {
    assert_eq!(resolve_mma(SM86, Float16, Float32, (16, 16, 16)), None);
    assert_eq!(resolve_mma(SM86, Float16, Float32, (16, 8, 16)), None);
    let mma = resolve_mma(SM86, BFloat16, Float32, (8, 16, 16)).unwrap();
    assert_eq!((mma.input, mma.acc), (Wire::Word("i32"), Wire::Word("float")));
    let mma = resolve_mma(SM86, Float16, Float16, (8, 16, 16)).unwrap();
    assert_eq!((mma.input, mma.acc), (Wire::HalfPair, Wire::HalfPair));
}

fn cuda_wmma_meta(k: usize, in_dt: DType, out_dt: DType, lanes: (usize, usize, usize)) -> WmmaMetadata {
    let axes = |count| vec![(svod_ir::AxisId::Renumbered(2), count)];
    WmmaMetadata {
        name: "WMMA_test".to_string(),
        dims: (8, 16, k),
        dtype_in: in_dt,
        dtype_out: out_dt,
        device: RendererDevice::CudaSm80, // unused by the NVPTX path (keyed on `arch`)
        threads: 32,
        upcast_axes: Some(WmmaUpcastAxes { a: axes(lanes.0), b: axes(lanes.1), c: axes(lanes.2) }),
        reduce_axes: vec![],
    }
}

/// WMMA over SSA operands: A/B are `<lanes.0/1 x in_dt>` and C is `<lanes.2 x out_dt>`
/// — the `elements_per_thread` of the scheduler's `CUDA_*` configs. The result
/// is stored so `-O3` cannot discard the `mma.sync`.
fn wmma_ssa_sink(k: usize, in_dt: DType, out_dt: DType, lanes: (usize, usize, usize)) -> Arc<UOp> {
    let element = |slot, dt: DType| {
        let p = UOp::param(slot, 1, dt, None);
        UOp::index().buffer(p).indices(vec![UOp::const_(DType::Int32, ConstValue::Int(0))]).call().unwrap()
    };
    let load = |slot, dt: DType| UOp::load().index(element(slot, dt)).call();
    let a = load(0, in_dt.clone()).broadcast(lanes.0);
    let b = load(1, in_dt.clone()).broadcast(lanes.1);
    let c = load(2, out_dt.clone()).broadcast(lanes.2);
    let d = UOp::wmma(a, b, c, cuda_wmma_meta(k, in_dt, out_dt.clone(), lanes));
    UOp::sink(vec![element(3, out_dt).store(d)])
}

/// Fragment packing, declaration synthesis and PTX selection per scheduler
/// tensor-core config (`CUDA_8168`, `CUDA_81616`, `CUDA_8168_TF32`,
/// `CUDA_81632`), with clang + ptxas as the oracles.
#[rustfmt::skip]
#[test_case::test_case(SM86, 16, DType::Float16, DType::Float32, (8, 4, 4),
    &["shufflevector <8 x half> %", "<2 x i32> <i32 6, i32 7>", "shufflevector <4 x half> %",
      "call { float, float, float, float } @llvm.nvvm.mma.m16n8k16.row.col.f32.f32(<2 x half> %",
      "declare { float, float, float, float } @llvm.nvvm.mma.m16n8k16.row.col.f32.f32(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, float, float, float, float)",
      "extractvalue { float, float, float, float } %", "insertelement <4 x float>"],
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"; "cuda_81616 f16 to f32")]
#[test_case::test_case(SM86, 16, DType::Float16, DType::Float16, (8, 4, 4),
    &["call { <2 x half>, <2 x half> } @llvm.nvvm.mma.m16n8k16.row.col.f16.f16(<2 x half> %",
      "declare { <2 x half>, <2 x half> } @llvm.nvvm.mma.m16n8k16.row.col.f16.f16(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>)",
      "shufflevector <2 x half> %", "<4 x i32> <i32 0, i32 1, i32 2, i32 3>"],
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"; "cuda_81616 f16 accumulator")]
#[test_case::test_case(SM86, 16, DType::BFloat16, DType::Float32, (8, 4, 4),
    &["bitcast <8 x bfloat> %", "to <4 x i32>", "bitcast <4 x bfloat> %", "to <2 x i32>", "extractelement <4 x i32> %",
      "declare { float, float, float, float } @llvm.nvvm.mma.m16n8k16.row.col.bf16(i32, i32, i32, i32, i32, i32, float, float, float, float)"],
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"; "cuda_81616 bf16 packs i32 words")]
#[test_case::test_case(SM75, 8, DType::Float16, DType::Float32, (4, 2, 4),
    &["shufflevector <4 x half> %", "<2 x half> %v",
      "declare { float, float, float, float } @llvm.nvvm.mma.m16n8k8.row.col.f32.f32(<2 x half>, <2 x half>, <2 x half>, float, float, float, float)"],
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"; "cuda_8168 f16 on turing")]
#[test_case::test_case(SM86, 8, DType::Float16, DType::Float16, (4, 2, 4),
    &["declare { <2 x half>, <2 x half> } @llvm.nvvm.mma.m16n8k8.row.col.f16.f16(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>)"],
    "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"; "cuda_8168 f16 accumulator")]
#[test_case::test_case(SM86, 8, DType::Float32, DType::Float32, (4, 2, 4),
    &["bitcast <4 x float> %", "to <4 x i32>", "bitcast <2 x float> %", "to <2 x i32>",
      "declare { float, float, float, float } @llvm.nvvm.mma.m16n8k8.row.col.tf32(i32, i32, i32, i32, i32, i32, float, float, float, float)"],
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"; "cuda_8168_tf32 passes raw f32 words")]
#[test_case::test_case(SM89, 32, DType::FP8E4M3, DType::Float32, (16, 8, 4),
    &["bitcast <16 x i8> %", "to <4 x i32>", "bitcast <8 x i8> %", "to <2 x i32>",
      "declare { float, float, float, float } @llvm.nvvm.mma.m16n8k32.row.col.f32.e4m3.e4m3.f32(i32, i32, i32, i32, i32, i32, float, float, float, float)"],
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"; "cuda_81632 e4m3 on ada")]
#[test_case::test_case(SM89, 32, DType::FP8E5M2, DType::Float32, (16, 8, 4),
    &["@llvm.nvvm.mma.m16n8k32.row.col.f32.e5m2.e5m2.f32("],
    "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32"; "cuda_81632 e5m2 on ada")]
#[test_case::test_case(SM86, 32, DType::Int8, DType::Int32, (16, 8, 4),
    &["bitcast <16 x i8> %", "to <4 x i32>", "extractelement <4 x i32> %",
      "declare { i32, i32, i32, i32 } @llvm.nvvm.mma.m16n8k32.row.col.satfinite.s8(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)",
      "extractvalue { i32, i32, i32, i32 } %", "insertelement <4 x i32>"],
    "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32"; "int8 satfinite")]
fn nvptx_wmma_operand_packing(
    arch: CudaArch,
    k: usize,
    in_dt: DType,
    out_dt: DType,
    lanes: (usize, usize, usize),
    present: &[&str],
    ptx_instruction: &str,
) {
    let result = render_nvptx_linearized(&wmma_ssa_sink(k, in_dt, out_dt, lanes), arch, "nvptx_wmma");
    for needle in present {
        assert!(result.code.contains(needle), "missing {needle}:\n{}", result.code);
    }
    // Every declaration is synthesized once, even though the call's aggregate
    // return type carries commas.
    assert_eq!(result.code.matches("declare { ").count(), 1, "{}", result.code);
    if let Some(ptx) = assert_ptx_compiles(&result.code, arch) {
        assert!(ptx.contains(ptx_instruction), "missing {ptx_instruction}:\n{ptx}");
    }
}

/// A tiled matmul issues the same intrinsic many times; the module must still
/// carry one declaration.
#[test]
fn nvptx_wmma_declaration_is_deduplicated_across_calls() {
    let element = |slot, dt: DType| {
        let p = UOp::param(slot, 1, dt, None);
        UOp::index().buffer(p).indices(vec![UOp::const_(DType::Int32, ConstValue::Int(0))]).call().unwrap()
    };
    let load = |slot, dt: DType| UOp::load().index(element(slot, dt)).call();
    let a = load(0, DType::Float16).broadcast(8);
    let b = load(1, DType::Float16).broadcast(4);
    let meta = || cuda_wmma_meta(16, DType::Float16, DType::Float32, (8, 4, 4));
    let first = UOp::wmma(a.clone(), b.clone(), load(2, DType::Float32).broadcast(4), meta());
    let second = UOp::wmma(a, b, load(4, DType::Float32).broadcast(4), meta());
    let twice = UOp::sink(vec![element(3, DType::Float32).store(first), element(5, DType::Float32).store(second)]);
    let result = render_nvptx_linearized(&twice, SM86, "nvptx_wmma_twice");
    let intrinsic = "@llvm.nvvm.mma.m16n8k16.row.col.f32.f32(";
    assert_eq!(
        result.code.matches(&format!("call {{ float, float, float, float }} {intrinsic}")).count(),
        2,
        "{}",
        result.code
    );
    assert_eq!(
        result.code.matches(&format!("declare {{ float, float, float, float }} {intrinsic}")).count(),
        1,
        "{}",
        result.code
    );
}

fn render_raw(root: Arc<UOp>, arch: CudaArch) -> crate::Result<crate::RenderedKernel> {
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(root).into());
    LlvmTextRenderer::nvptx(arch).render(&linear, Some("nvptx_wmma_reject"))
}

#[test]
fn nvptx_wmma_below_the_shape_capability_is_rejected() {
    let err = render_raw(wmma_ssa_sink(16, DType::Float16, DType::Float32, (8, 4, 4)), SM75)
        .expect_err("m16n8k16 needs sm_80");
    assert!(err.to_string().contains("no mma.sync form for arch=sm_75"), "unexpected error: {err}");
}

#[test]
fn nvptx_wmma_with_the_wrong_fragment_width_is_rejected() {
    // A carries four halves where m16n8k16 needs eight (4 registers).
    let err = render_raw(wmma_ssa_sink(16, DType::Float16, DType::Float32, (4, 4, 4)), SM86)
        .expect_err("fragment width must match the intrinsic");
    assert!(err.to_string().contains("carries 2 32-bit registers"), "unexpected error: {err}");
    assert!(err.to_string().contains("needs 4"), "unexpected error: {err}");
}
