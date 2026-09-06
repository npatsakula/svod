//! NVPTX `mma.sync` dispatch by compute capability.
//!
//! The IR-level matrix shape is encoded in `WmmaMetadata::dims` as `(N, M, K)`;
//! every CUDA tensor-core profile in the scheduler is an `m16n8kK` shape
//! (`CUDA_8168`, `CUDA_81616`, `CUDA_81632`, `CUDA_8168_TF32`). For each
//! `(arch, in_dtype, acc_dtype, K)` tuple we map to one `@llvm.nvvm.mma.*`
//! intrinsic and pack the per-thread fragments into its 32-bit register
//! operands. Operand shapes verified with clang 22 + ptxas 13.3 (see
//! `nvidia_backend_plan.md` §2); the register split mirrors tinygrad's
//! `renderer/ptx.py:render_wmma` (`mov.b32 reg, {lo, hi}` per pair).

use std::sync::Arc;

use svod_dtype::{CudaArch, DType, ScalarDType};
use svod_ir::{WmmaMetadata, prelude::*};

use crate::llvm::common::gpu::wmma_operand_dtype;
use crate::llvm::common::{RenderContext, ldt};

/// How a fragment is handed to the intrinsic: `<2 x half>` pairs, or 32-bit
/// words of the given LLVM element type (`i32` for bf16/tf32/fp8/int8 bit
/// patterns and int accumulators, `float` for f32 accumulators).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wire {
    HalfPair,
    Word(&'static str),
}

impl Wire {
    fn element_type(self) -> &'static str {
        match self {
            Wire::HalfPair => "<2 x half>",
            Wire::Word(ty) => ty,
        }
    }
}

/// One resolved `mma.sync` form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mma {
    pub intrinsic: String,
    pub input: Wire,
    pub acc: Wire,
}

/// Render a WMMA UOp for the NVPTX target. Returns `None` when the
/// `(arch, dtype, shape)` combination has no `mma.sync` form; the caller
/// then surfaces an `InvalidGraph` error and the optimizer must decompose it
/// upstream.
#[allow(clippy::too_many_arguments)]
pub fn render_wmma_nvptx(
    uop: &Arc<UOp>,
    a: &Arc<UOp>,
    b: &Arc<UOp>,
    c: &Arc<UOp>,
    metadata: &WmmaMetadata,
    arch: CudaArch,
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
    let in_scalar = a_dtype.base();
    let acc_scalar = out_dtype.base();

    let Some(mma) = resolve_mma(arch, in_scalar, acc_scalar, (n, m, k)) else {
        ctx.set_invalid_graph(format!(
            "NVPTX renderer: no mma.sync form for arch={arch} in={in_scalar:?} acc={acc_scalar:?} dims=({n},{m},{k})"
        ));
        return None;
    };

    // Per-thread register counts of an `m16n8kK` fragment: A is 16×K, B is
    // K×8, C/D is 16×8, all spread over 32 lanes in 32-bit registers.
    let words = |elements: usize, bytes: usize| elements * bytes / (32 * 4);
    let expected = [words(m * k, in_scalar.bytes()), words(k * n, in_scalar.bytes()), words(m * n, acc_scalar.bytes())];
    let mut fragments = Vec::with_capacity(3);
    for (operand, dtype, name, wire, want) in [
        ("a", &a_dtype, &a_name, mma.input, expected[0]),
        ("b", &b_dtype, &b_name, mma.input, expected[1]),
        ("c", &c_dtype, &c_name, mma.acc, expected[2]),
    ] {
        let frag = split_fragment(kernel, &dst, operand, dtype, name, wire);
        if frag.len() != want {
            ctx.set_invalid_graph(format!(
                "NVPTX renderer: WMMA operand {operand} of {} carries {} 32-bit registers, {} needs {want}",
                ldt(dtype),
                frag.len(),
                mma.intrinsic
            ));
            return None;
        }
        fragments.push(frag);
    }

    let acc_ty = mma.acc.element_type();
    let ret_ty = format!("{{ {} }}", vec![acc_ty; expected[2]].join(", "));
    let args = fragments
        .iter()
        .zip([mma.input, mma.input, mma.acc])
        .flat_map(|(frag, wire)| frag.iter().map(move |value| format!("{} {value}", wire.element_type())))
        .collect::<Vec<_>>()
        .join(", ");
    let call = format!("{dst}.mma");
    kernel.push(format!("  {call} = call {ret_ty} @{}({args})", mma.intrinsic));
    join_fragment(kernel, &dst, &call, &ret_ty, &out_dtype, mma.acc, expected[2]);
    Some(())
}

/// Split a `<N x T>` operand into the intrinsic's 32-bit register values and
/// return their SSA names. `HalfPair` shuffles out `<2 x half>` slices; `Word`
/// reinterprets the vector as `i32` words (or reads `float`/`i32` lanes
/// directly when the natural type already is the wire type).
fn split_fragment(
    kernel: &mut Vec<String>,
    dst: &str,
    operand: &str,
    dtype: &DType,
    name: &str,
    wire: Wire,
) -> Vec<String> {
    let lanes = dtype.vcount();
    let natural = ldt(dtype);
    match wire {
        Wire::HalfPair => (0..lanes / 2)
            .map(|pair| {
                let value = format!("{dst}.{operand}{pair}");
                let (lo, hi) = (2 * pair, 2 * pair + 1);
                kernel.push(format!(
                    "  {value} = shufflevector {natural} {name}, {natural} poison, <2 x i32> <i32 {lo}, i32 {hi}>"
                ));
                value
            })
            .collect(),
        Wire::Word(ty) => {
            let count = lanes * dtype.base().bytes() / 4;
            let words_ty = format!("<{count} x {ty}>");
            let source = if natural == words_ty {
                name.to_string()
            } else {
                let packed = format!("{dst}.{operand}w");
                let target = if count == 1 { ty.to_string() } else { words_ty.clone() };
                kernel.push(format!("  {packed} = bitcast {natural} {name} to {target}"));
                packed
            };
            if count == 1 && natural != words_ty {
                return vec![source];
            }
            (0..count)
                .map(|word| {
                    let value = format!("{dst}.{operand}{word}");
                    kernel.push(format!("  {value} = extractelement {words_ty} {source}, i32 {word}"));
                    value
                })
                .collect()
        }
    }
}

/// Reassemble the intrinsic's aggregate result into the WMMA's natural
/// `<N x T>` vector under `dst`.
fn join_fragment(kernel: &mut Vec<String>, dst: &str, call: &str, ret_ty: &str, out: &DType, wire: Wire, count: usize) {
    let natural = ldt(out);
    let parts: Vec<String> = (0..count)
        .map(|i| {
            let part = format!("{dst}.d{i}");
            kernel.push(format!("  {part} = extractvalue {ret_ty} {call}, {i}"));
            part
        })
        .collect();
    match wire {
        Wire::HalfPair => {
            // m16n8 half accumulators are exactly two pairs; concatenate them.
            let mask = (0..2 * count).map(|i| format!("i32 {i}")).collect::<Vec<_>>().join(", ");
            let (lo, hi) = (&parts[0], parts.get(1).map_or("poison", String::as_str));
            kernel.push(format!(
                "  {dst} = shufflevector <2 x half> {lo}, <2 x half> {hi}, <{} x i32> <{mask}>",
                2 * count
            ));
        }
        Wire::Word(ty) => {
            let words_ty = format!("<{count} x {ty}>");
            let assembled = if natural == words_ty { dst.to_string() } else { format!("{dst}.w") };
            let mut acc = "poison".to_string();
            for (i, part) in parts.iter().enumerate() {
                let next = if i + 1 == count { assembled.clone() } else { format!("{dst}.i{i}") };
                kernel.push(format!("  {next} = insertelement {words_ty} {acc}, {ty} {part}, i32 {i}"));
                acc = next;
            }
            if natural != words_ty {
                kernel.push(format!("  {dst} = bitcast {words_ty} {assembled} to {natural}"));
            }
        }
    }
}

/// Pick the `@llvm.nvvm.mma.*` form for a `(arch, dtype, shape)` tuple.
///
/// Every row was verified through clang + ptxas; the PTX ISA fixes the minimum
/// compute capability per shape (`m16n8k8` f16: sm_75; `m16n8k16` f16, bf16,
/// tf32 and the int8 `m16n8k32`: sm_80; fp8 `m16n8k32`: sm_89). Anything else
/// returns `None` so the caller raises `InvalidGraph` (and the optimizer
/// decomposes) instead of naming an intrinsic LLVM would silently emit as an
/// external call.
pub fn resolve_mma(
    arch: CudaArch,
    in_dt: ScalarDType,
    acc_dt: ScalarDType,
    dims: (usize, usize, usize),
) -> Option<Mma> {
    use ScalarDType::{BFloat16, FP8E4M3, FP8E5M2, Float16, Float32, Int8, Int32};
    use Wire::{HalfPair, Word};

    let (n, m, k) = dims;
    if (n, m) != (8, 16) {
        return None;
    }
    let (suffix, input, acc, min_sm) = match (in_dt, acc_dt, k) {
        (Float16, Float32, 8) => ("m16n8k8.row.col.f32.f32", HalfPair, Word("float"), 75),
        (Float16, Float16, 8) => ("m16n8k8.row.col.f16.f16", HalfPair, HalfPair, 75),
        (Float16, Float32, 16) => ("m16n8k16.row.col.f32.f32", HalfPair, Word("float"), 80),
        (Float16, Float16, 16) => ("m16n8k16.row.col.f16.f16", HalfPair, HalfPair, 80),
        (BFloat16, Float32, 16) => ("m16n8k16.row.col.bf16", Word("i32"), Word("float"), 80),
        // tf32 takes the raw f32 bit patterns; the hardware ignores the low 13
        // mantissa bits (tinygrad passes the same raw words).
        (Float32, Float32, 8) => ("m16n8k8.row.col.tf32", Word("i32"), Word("float"), 80),
        (Int8, Int32, 32) => ("m16n8k32.row.col.satfinite.s8", Word("i32"), Word("i32"), 80),
        (FP8E4M3, Float32, 32) => ("m16n8k32.row.col.f32.e4m3.e4m3.f32", Word("i32"), Word("float"), 89),
        (FP8E5M2, Float32, 32) => ("m16n8k32.row.col.f32.e5m2.e5m2.f32", Word("i32"), Word("float"), 89),
        _ => return None,
    };
    (arch.sm() >= min_sm).then(|| Mma { intrinsic: format!("llvm.nvvm.mma.{suffix}"), input, acc })
}

#[cfg(test)]
#[path = "../../test/unit/llvm_nvptx_wmma.rs"]
mod tests;
