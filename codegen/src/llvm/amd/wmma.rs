//! AMD WMMA / MFMA intrinsic dispatch by gfx family.
//!
//! The IR-level matrix shape is encoded in `WmmaMetadata::dims` as `(N, M, K)`.
//! For each (arch, in_dtype, acc_dtype, dims) tuple we map to one of the
//! `@llvm.amdgcn.{wmma|mfma}.*` intrinsics, packing inputs as needed.

use std::sync::Arc;

use svod_dtype::{AmdArch, DType, ScalarDType};
use svod_ir::{WmmaMetadata, prelude::*};

use crate::llvm::common::gpu::wmma_operand_dtype;
use crate::llvm::common::{RenderContext, ldt};

/// Render a WMMA UOp for the AMD target. Returns `None` if the (arch, dtype,
/// shape) combination has no direct intrinsic; in that case the caller
/// surfaces an `InvalidGraph` error and the optimizer must decompose it
/// upstream.
#[allow(clippy::too_many_arguments)]
pub fn render_wmma_amd(
    uop: &Arc<UOp>,
    a: &Arc<UOp>,
    b: &Arc<UOp>,
    c: &Arc<UOp>,
    metadata: &WmmaMetadata,
    arch: AmdArch,
    ctx: &mut RenderContext,
    kernel: &mut Vec<String>,
) -> Option<()> {
    let dst = ctx.name(uop);
    let a_name = ctx.get(a).to_string();
    let b_name = ctx.get(b).to_string();
    let c_name = ctx.get(c).to_string();

    let (n, m, k) = metadata.dims;
    let a_dtype = wmma_operand_dtype(a);
    let b_dtype = wmma_operand_dtype(b);
    let c_dtype = wmma_operand_dtype(c);
    let out_dtype = wmma_operand_dtype(uop);
    let in_scalar = Some(a_dtype.base());
    let acc_scalar = Some(out_dtype.base());

    let intrinsic = match resolve_intrinsic(arch, in_scalar, acc_scalar, (n, m, k)) {
        Some(s) => s,
        None => {
            ctx.set_invalid_graph(format!(
                "AMD renderer: no WMMA/MFMA intrinsic for arch={arch} in={in_scalar:?} \
                 acc={acc_scalar:?} dims=({n},{m},{k})"
            ));
            return None;
        }
    };

    // The intrinsics take their operands as raw bit patterns, not the natural
    // float types: bf16 lanes as `<N x i16>`, fp8 lanes packed into a single
    // `iN`. Bitcast each operand to its wire type before the call (and the
    // accumulator + result back, when it too is reinterpreted). Mirrors
    // tinygrad's `AMDLLVMRenderer` operand rewrite (`llvmir.py:274-298`).
    // The K=32 dotted `.bf16` MFMA (CDNA4/gfx950) takes its operands as native
    // `<N x bfloat>`; the K=16 `.bf16.1k` form takes `<N x i16>`. gfx942 never
    // reaches here for K=32 bf16 — `resolve_intrinsic` returns `None` above —
    // so this only flips the wire type on the gfx950 path.
    let bf16_native = arch.is_cdna() && k == 32;

    let scaled_fp8 = matches!(arch, AmdArch::Gfx950)
        && k == 128
        && matches!(in_scalar, Some(ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2));
    let rdna_int8 = !arch.is_cdna() && !arch.is_rdna4() && in_scalar == Some(ScalarDType::Int8);
    let a_op = bitcast_operand(kernel, &dst, "a", &a_dtype, &a_name, bf16_native, scaled_fp8, rdna_int8);
    let b_op = bitcast_operand(kernel, &dst, "b", &b_dtype, &b_name, bf16_native, scaled_fp8, rdna_int8);
    let c_op = bitcast_operand(kernel, &dst, "c", &c_dtype, &c_name, bf16_native, false, false);

    let (acc_wire, acc_reinterpreted) = wmma_wire_type(&out_dtype, bf16_native);
    let call_dst = if acc_reinterpreted { format!("{dst}.r") } else { dst.clone() };

    let tail = if scaled_fp8 {
        let format = if matches!(in_scalar, Some(ScalarDType::FP8E5M2)) { 1 } else { 0 };
        format!(", i32 {format}, i32 {format}, i32 0, i32 127, i32 0, i32 127")
    } else if arch.is_cdna() {
        // MFMA: trailing cbsz/abid/blgp immediates.
        ", i32 0, i32 0, i32 0".to_string()
    } else if matches!(acc_scalar, Some(ScalarDType::Float32)) {
        // f32-accumulating WMMAs take (A, B, C) only.
        String::new()
    } else {
        // Any other accumulator (f16/bf16/int) takes a trailing `i1 false`
        // (the clamp/opsel bit).
        ", i1 false".to_string()
    };

    let args = if rdna_int8 {
        // The `iu8` intrinsic carries one signedness flag before each packed
        // operand. Int8 inputs set both flags; the trailing flag is opsel.
        format!("i1 true, {a_op}, i1 true, {b_op}, {c_op}, i1 false")
    } else {
        format!("{a_op}, {b_op}, {c_op}{tail}")
    };
    kernel.push(format!("  {call_dst} = call {acc_wire} @{intrinsic}({args})"));

    if acc_reinterpreted {
        // bf16→bf16: the call returns `<N x i16>`; reinterpret it back to bf16.
        kernel.push(format!("  {dst} = bitcast {acc_wire} {call_dst} to {}", ldt(&out_dtype)));
    }
    Some(())
}

/// The LLVM type a WMMA/MFMA operand must be passed as, plus whether that
/// differs from its natural `ldt` type (a bitcast is then required). bf16 lanes
/// go as `i16` (the `bf16.1k`/RDNA `.bf16` intrinsics), except for the CDNA4
/// K=32 `.bf16` form which takes native `<N x bfloat>` (`bf16_native`); K=32
/// fp8 lanes pack into one `iN`, scaled K=128 uses packed i32 vectors, and
/// RDNA3 int8 lanes pack four-at-a-time into i32 vectors.
fn wmma_wire_type(dtype: &DType, bf16_native: bool) -> (String, bool) {
    wmma_wire_type_with_scaled_fp8(dtype, bf16_native, false, false)
}

fn wmma_wire_type_with_scaled_fp8(
    dtype: &DType,
    bf16_native: bool,
    scaled_fp8: bool,
    rdna_int8: bool,
) -> (String, bool) {
    match dtype {
        DType::Vector { scalar: ScalarDType::BFloat16, count } if !bf16_native => (format!("<{count} x i16>"), true),
        DType::Vector { scalar: ScalarDType::Int8, count } if rdna_int8 => (format!("<{} x i32>", count / 4), true),
        DType::Vector { scalar: ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2, count } if scaled_fp8 => {
            (format!("<{} x i32>", count / 4), true)
        }
        DType::Vector { scalar: ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2, count } => {
            (format!("i{}", count * 8), true)
        }
        _ => (ldt(dtype), false),
    }
}

/// Emit the operand bitcast when its wire type differs from its natural type,
/// and return the `"<wire-ty> <value>"` fragment for the call's argument list.
/// The temp name is derived from the unique `dst` (`%vN.a` …) so no fresh-name
/// counter is needed.
#[allow(clippy::too_many_arguments)]
fn bitcast_operand(
    kernel: &mut Vec<String>,
    dst: &str,
    suffix: &str,
    dtype: &DType,
    name: &str,
    bf16_native: bool,
    scaled_fp8: bool,
    rdna_int8: bool,
) -> String {
    let (wire_ty, reinterpreted) = wmma_wire_type_with_scaled_fp8(dtype, bf16_native, scaled_fp8, rdna_int8);
    if !reinterpreted {
        return format!("{wire_ty} {name}");
    }
    let tmp = format!("{dst}.{suffix}");
    kernel.push(format!("  {tmp} = bitcast {} {name} to {wire_ty}", ldt(dtype)));
    format!("{wire_ty} {tmp}")
}

/// Pick an amdgcn intrinsic name for a given (arch, dtype, shape) tuple.
///
/// Returns `None` for shapes/dtypes the renderer doesn't natively support
/// (the optimizer is expected to decompose those upstream).
///
/// Naming scheme:
/// - RDNA3: `llvm.amdgcn.wmma.<acc>.16x16x16.<in>`.
/// - RDNA4: the same base name plus LLVM's overloaded result/input vector
///   suffixes, for example `.v8f32.v8f16`.
/// - CDNA: `llvm.amdgcn.mfma.<acc>.<N>x<M>x<K><in>`.
/// - RDNA2 and other non-matrix-core arches: `None` — the optimizer must
///   decompose WMMA UOps to scalar/vector loops before rendering.
fn resolve_intrinsic(
    arch: AmdArch,
    in_dt: Option<ScalarDType>,
    acc_dt: Option<ScalarDType>,
    dims: (usize, usize, usize),
) -> Option<String> {
    if !arch.has_matrix_cores() {
        return None;
    }

    let (n, m, k) = dims;
    let in_dt = in_dt?;
    let acc_dt = acc_dt?;

    if (n, m) != (16, 16) {
        return None;
    }

    if arch.is_cdna() {
        // Verified with `llc -mcpu=gfx942|gfx950` (ROCm 7.2): the f16/bf16 K=16
        // forms (`f16`/`bf16.1k`) select on both CDNA3 (gfx942) and CDNA4
        // (gfx950); the dotted K=32 double-rate forms (`.f16`/`.bf16`) select on
        // gfx950 only; fp8/bf8 select at K=32 on both and scaled K=128 on
        // gfx950; f32 selects only at K=4 (`v_mfma_f32_16x16x4_f32`, scalar
        // A/B operands). Anything else has no
        // MFMA intrinsic — return `None` so the caller raises `InvalidGraph`
        // (and the optimizer decomposes it) instead of emitting a name LLVM
        // silently lowers to a no-op extern call.
        let is_cdna4 = matches!(arch, AmdArch::Gfx950);
        let in_suffix = match (in_dt, k) {
            (ScalarDType::Float16, 32) if is_cdna4 => ".f16",
            (ScalarDType::BFloat16, 32) if is_cdna4 => ".bf16",
            (ScalarDType::Float16, 16) => "f16",
            (ScalarDType::BFloat16, 16) => "bf16.1k",
            (ScalarDType::Float32, 4) => "f32",
            (ScalarDType::FP8E4M3, 128) | (ScalarDType::FP8E5M2, 128) if is_cdna4 => ".f8f6f4",
            (ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2, 128) => return None,
            (ScalarDType::FP8E4M3, 32) => ".fp8.fp8",
            (ScalarDType::FP8E5M2, 32) => ".bf8.bf8",
            _ => return None,
        };
        let acc_suffix = match acc_dt {
            ScalarDType::Float32 => "f32",
            ScalarDType::Float64 => "f64",
            ScalarDType::Int32 => "i32",
            _ => return None,
        };
        // Only the K=128 `.f8f6f4` form is a scaled MFMA; keying on K alone would
        // mint `mfma.scale.*` names for any K=128 input dtype.
        let scale = if in_suffix == ".f8f6f4" { "scale." } else { "" };
        return Some(format!("llvm.amdgcn.mfma.{scale}{acc_suffix}.{n}x{m}x{k}{in_suffix}"));
    }

    // RDNA3 / RDNA4 WMMA — both families use 16x16x16 matmul; differ in input
    // dtype packing (handled by upstream pre-rewrites at the renderer level
    // when present; here we just name the intrinsic).
    if k != 16 {
        return None;
    }
    let in_suffix = match in_dt {
        ScalarDType::Float16 => "f16",
        ScalarDType::BFloat16 => "bf16",
        ScalarDType::Int8 if !arch.is_rdna4() => "iu8",
        _ => return None,
    };
    let acc_suffix = match acc_dt {
        ScalarDType::Float32 => "f32",
        ScalarDType::Float16 => "f16",
        ScalarDType::BFloat16 => "bf16",
        ScalarDType::Int32 => "i32",
        _ => return None,
    };
    let base = format!("llvm.amdgcn.wmma.{acc_suffix}.{n}x{m}x{k}.{in_suffix}");
    if !arch.is_rdna4() {
        return Some(base);
    }
    let acc_overload = match acc_dt {
        ScalarDType::Float32 => "v8f32",
        ScalarDType::Float16 => "v8f16",
        ScalarDType::BFloat16 => "v8i16",
        ScalarDType::Int32 => "v8i32",
        _ => return None,
    };
    let in_overload = match in_dt {
        ScalarDType::Float16 => "v8f16",
        ScalarDType::BFloat16 => "v8i16",
        _ => return None,
    };
    Some(format!("{base}.{acc_overload}.{in_overload}"))
}

#[cfg(test)]
#[path = "../../test/unit/llvm_amd_wmma.rs"]
mod tests;
