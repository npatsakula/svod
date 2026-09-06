//! AMD-specific UOp lowering.
//!
//! Intercepts AMD-only ops (`Special`, `Barrier`, `DefineLocal`, fp8 `Cast`,
//! `Wmma`) and falls through to the CPU emitter for everything else.

use std::sync::Arc;

use svod_dtype::{DType, ScalarDType};
use svod_ir::{Op, UnaryOp, prelude::*};

use crate::llvm::amd::wmma;
use crate::llvm::common::gpu::{AXIS_LETTERS, parse_special_axis, render_define_local};
use crate::llvm::common::{LlvmTarget, RenderContext, ldt};
use crate::llvm::cpu;
use svod_ir::ops;

/// Render a UOp to LLVM IR for the AMD target.
///
/// Returns `Some(())` when the op was handled (either AMD-specific or by
/// delegating to [`cpu::render_uop`]); `None` for meta-ops that don't produce
/// instructions.
pub fn render_uop(uop: &Arc<UOp>, ctx: &mut RenderContext, kernel: &mut Vec<String>, target: LlvmTarget) -> Option<()> {
    debug_assert!(target.is_amd(), "amd::render_uop called with non-AMD target {target:?}");
    let arch = target.amd_arch().expect("AMD target");

    match uop.op() {
        // ── AMD-specific overrides ───────────────────────────────────────
        Op::Special(ops::Special { name, .. }) => render_special(uop, name, ctx, kernel),
        Op::Barrier(..) => render_barrier(kernel),
        Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local) => {
            render_define_local(uop, ctx, kernel)
        }
        Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Reg) => {
            render_define_reg(uop, ctx, kernel)
        }
        Op::Wmma(ops::Wmma { a, b, c, metadata }) => wmma::render_wmma_amd(uop, a, b, c, metadata, arch, ctx, kernel),
        Op::Unary(op @ (UnaryOp::Sqrt | UnaryOp::Log2 | UnaryOp::Exp2), src) => {
            render_float_unary(uop, src, *op, ctx, kernel)
        }
        // Sin must never reach the AMD renderer: the device excludes it from
        // `supported_ops` so the scheduler always decomposes it (`xsin`) —
        // `@llvm.sin.f32` lowers to `v_sin_f32` behind an f32 `1/(2π)`
        // pre-scale that is wrong for large arguments, and `@llvm.sin.f64`
        // is unselectable. Backstop here so a capability-list drift fails
        // loudly at render time instead of silently emitting either form.
        Op::Unary(UnaryOp::Sin, _) => {
            ctx.set_invalid_graph(
                "AMD renderer received an un-decomposed Sin; it must be excluded from supported_ops and lowered by \
                 the scheduler's transcendental decomposition",
            );
            Some(())
        }
        Op::Cast(ops::Cast { src, .. }) if is_fp8_cast(uop, src) => render_fp8_cast(uop, src, ctx, kernel),
        // ── Everything else: shared CPU path (ALU, INDEX, LOAD, STORE, …) ─
        _ => cpu::render_uop(uop, ctx, kernel),
    }
}

/// Lower `sqrt`/`log2`/`exp2` to the LLVM intrinsic the AMDGPU backend
/// selects itself, so the object needs no ROCm device library.
///
/// Tinygrad renders exactly these as `@llvm.{sqrt,log2,exp2}.<ty>`
/// (`renderer/llvmir.py` `llvm_intrinsics`) and drives amdgcn straight through
/// LLVM with no device libs (`runtime/support/compiler_llvm.py:19-24`). `Sin`
/// is deliberately NOT in this set: `@llvm.sin.f32` lowers to the hardware
/// `v_sin_f32` behind an f32 `1/(2π)` pre-scale that is only accurate for
/// small arguments, so `Sin` is excluded from the device's `supported_ops`
/// and always decomposes (`xsin`, Payne-Hanek) in the scheduler. The AMDGPU
/// backend also has no f64 lowering for `log2`/`exp2` — it reports "no
/// libcall available", which is why tinygrad substitutes its `xlog2`/`xexp2`
/// expansions there — so those keep the ROCm `__ocml_*` entry points. Their
/// presence is what `amd_object_flags` keys `-nogpulib` off.
fn render_float_unary(
    uop: &Arc<UOp>,
    src: &Arc<UOp>,
    op: UnaryOp,
    ctx: &mut RenderContext,
    kernel: &mut Vec<String>,
) -> Option<()> {
    let bits = match uop.dtype() {
        DType::Scalar(ScalarDType::Float16) => 16,
        DType::Scalar(ScalarDType::Float32) => 32,
        DType::Scalar(ScalarDType::Float64) => 64,
        _ => return cpu::render_uop(uop, ctx, kernel),
    };
    let name = match op {
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Log2 => "log2",
        UnaryOp::Exp2 => "exp2",
        _ => unreachable!(),
    };
    let callee =
        if bits == 64 && name != "sqrt" { format!("@__ocml_{name}_f64") } else { format!("@llvm.{name}.f{bits}") };
    let ty = ldt(&uop.dtype());
    let dst = ctx.name(uop);
    let value = ctx.get(src);
    kernel.push(format!("  {dst} = call {ty} {callee}({ty} {value})"));
    Some(())
}

// ── SPECIAL: workgroup / workitem / direct-global axis ────────────────────

fn render_special(uop: &Arc<UOp>, name: &str, ctx: &mut RenderContext, kernel: &mut Vec<String>) -> Option<()> {
    let dst = ctx.name(uop);
    let (kind, axis) = match parse_special_axis(name) {
        Some(parsed) => parsed,
        None => {
            ctx.set_invalid_graph(format!("AMD renderer: malformed SPECIAL axis name {name:?}"));
            return None;
        }
    };
    let dim = AXIS_LETTERS[axis as usize];

    match kind {
        'g' => kernel.push(format!("  {dst} = tail call i32 @llvm.amdgcn.workgroup.id.{dim}()")),
        'l' => kernel.push(format!("  {dst} = tail call i32 @llvm.amdgcn.workitem.id.{dim}()")),
        'i' => {
            // Direct-global axis: the usual lowering is `g*lsz + l`, but svod's
            // ProgramSpec drops `local_size` entirely for `i` prefixes
            // (`device/src/device.rs:660`). The kernel sees one flat axis,
            // so workgroup.id.x suffices (workgroup_size_x = 1 in the AQL
            // packet under DirectGlobal launch).
            kernel.push(format!("  {dst} = tail call i32 @llvm.amdgcn.workgroup.id.{dim}()"));
        }
        _ => unreachable!(),
    }
    Some(())
}

// ── BARRIER: workgroup-scope fence + s.barrier ────────────────────────────

fn render_barrier(kernel: &mut Vec<String>) -> Option<()> {
    kernel.push("  fence syncscope(\"workgroup\") release".to_string());
    kernel.push("  tail call void @llvm.amdgcn.s.barrier()".to_string());
    kernel.push("  fence syncscope(\"workgroup\") acquire".to_string());
    Some(())
}

// ── DEFINE_REG: addrspace(5) alloca ───────────────────────────────────────
//
// AMDGPU LLVM requires alloca to live in `addrspace(5)` (private/scratch
// memory) inside `amdgpu_kernel` functions. The alloca can land in
// addrspace(5) implicitly via `target triple = amdgcn-amd-amdhsa`, which
// makes addrspace(5) the alloca default for that triple in some LLVM
// versions. To be explicit (and avoid relying on backend-default
// behavior), we emit it directly and addrspacecast to a generic pointer
// for downstream GEP/LOAD/STORE.

fn render_define_reg(uop: &Arc<UOp>, ctx: &mut RenderContext, kernel: &mut Vec<String>) -> Option<()> {
    let dst = ctx.name(uop);
    let (alloc_size, base) = match uop.op() {
        Op::Buffer(ops::Buffer { arg, .. }) => (uop.buffer_size().unwrap_or(1), ldt(&arg.dtype)),
        _ => unreachable!(),
    };
    let raw = format!("{dst}.raw");
    kernel.push(format!("  {raw} = alloca [{alloc_size} x {base}], align 4, addrspace(5)"));
    kernel.push(format!("  {dst} = addrspacecast ptr addrspace(5) {raw} to ptr"));
    Some(())
}

// ── FP8 CAST: amdgcn cvt intrinsics ───────────────────────────────────────

fn is_fp8_cast(uop: &Arc<UOp>, src: &Arc<UOp>) -> bool {
    let dst = uop.dtype();
    let src_dt = src.dtype();
    // Only handle scalar casts between FP8 and f32; vector lanes are decomposed
    // upstream by the rangeify/devectorize passes.
    matches!(
        (dst.scalar(), src_dt.scalar()),
        (Some(ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2), Some(ScalarDType::Float32))
            | (Some(ScalarDType::Float32), Some(ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2))
    ) && dst.vcount() == 1
        && src_dt.vcount() == 1
}

fn render_fp8_cast(uop: &Arc<UOp>, src: &Arc<UOp>, ctx: &mut RenderContext, kernel: &mut Vec<String>) -> Option<()> {
    let dst_dt = uop.dtype();
    let src_dt = src.dtype();
    let dst_name = ctx.name(uop);
    let src_name = ctx.get(src).to_string();

    match (dst_dt.scalar(), src_dt.scalar()) {
        // f32 → fp8 via inlined helper `@f32_to_fp8`.
        (Some(d @ (ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2)), Some(ScalarDType::Float32)) => {
            let is_bf8 = matches!(d, ScalarDType::FP8E5M2);
            kernel.push(format!(
                "  {dst_name} = call i8 @f32_to_fp8(float {src_name}, i1 {})",
                if is_bf8 { 1 } else { 0 }
            ));
        }
        // fp8 → f32 via amdgcn cvt.
        (Some(ScalarDType::Float32), Some(s @ (ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2))) => {
            let kind = if matches!(s, ScalarDType::FP8E5M2) { "bf8" } else { "fp8" };
            let tmp = format!("{dst_name}_i32");
            kernel.push(format!("  {tmp} = zext i8 {src_name} to i32"));
            kernel.push(format!("  {dst_name} = call float @llvm.amdgcn.cvt.f32.{kind}(i32 {tmp}, i32 0)"));
        }
        _ => unreachable!("is_fp8_cast guard"),
    }
    Some(())
}

/// Module-level prefix lines required when the kernel uses fp8 conversions.
/// Returns the verbatim `@f32_to_fp8` helper when any node in the linear list
/// touches fp8, otherwise an empty string.
pub fn fp8_helper_prefix(nodes: &[Arc<UOp>]) -> Option<&'static str> {
    let uses_fp8 = nodes.iter().any(|n| {
        let dt = n.dtype();
        matches!(dt.scalar(), Some(ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2))
    });
    if !uses_fp8 {
        return None;
    }
    Some(FP8_HELPER)
}

/// Inlined f32 → fp8/bf8 conversion (verbatim port of the upstream helper).
/// The helper handles NaN/Inf preservation, clamping to the fp8 representable
/// range, and packs via amdgcn `cvt.pk.{fp8,bf8}.f32`.
const FP8_HELPER: &str = r#"define i8 @f32_to_fp8(float %val, i1 %is_bf8) {
entry:
  %ival = bitcast float %val to i32
  %exp = and i32 %ival, 2139095040
  %is_special = icmp eq i32 %exp, 2139095040
  br i1 %is_special, label %select_clip, label %clip
clip:
  br i1 %is_bf8, label %bf8_clip, label %fp8_clip
bf8_clip:
  %clamped_bf8 = call float @llvm.amdgcn.fmed3.f32(float %val, float 57344.0, float -57344.0)
  br label %select_clip
fp8_clip:
  %clamped_fp8 = call float @llvm.amdgcn.fmed3.f32(float %val, float 448.0, float -448.0)
  br label %select_clip
select_clip:
  %phi_val = phi float [%val, %entry], [%clamped_bf8, %bf8_clip], [%clamped_fp8, %fp8_clip]
  br i1 %is_bf8, label %do_bf8, label %do_fp8
do_bf8:
  %packed_bf8 = call i32 @llvm.amdgcn.cvt.pk.bf8.f32(float %phi_val, float %phi_val, i32 0, i1 false)
  br label %exit
do_fp8:
  %packed_fp8 = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %phi_val, float %phi_val, i32 0, i1 false)
  br label %exit
exit:
  %packed = phi i32 [%packed_bf8, %do_bf8], [%packed_fp8, %do_fp8]
  %trunc = trunc i32 %packed to i8
  ret i8 %trunc
}"#;
