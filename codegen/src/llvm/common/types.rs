//! LLVM type and constant string generation.
//!
//! Provides functions for converting Svod types to LLVM IR text.
//! Shared between CPU and GPU backends.

use svod_dtype::{AddrSpace, DType, ScalarDType, cast::committed_float_bits};
use svod_ir::ConstValue;

/// Convert a DType to LLVM type string.
///
/// Uses LLVM opaque pointer mode: all pointers are `ptr`, vectors of
/// pointers are `<N x ptr>`. Typed pointer syntax (`float*`) is not emitted.
pub fn ldt(dtype: &DType) -> String {
    match dtype {
        DType::Vector { scalar, count } => {
            format!("<{} x {}>", count, ldt_scalar(*scalar))
        }
        DType::Ptr { vcount, .. } if *vcount > 1 => {
            format!("<{} x ptr>", vcount)
        }
        DType::Ptr { .. } | DType::Image { .. } => "ptr".to_string(),
        DType::Scalar(s) => ldt_scalar(*s).to_string(),
    }
}

/// Convert a ScalarDType to LLVM type string.
fn ldt_scalar(s: ScalarDType) -> &'static str {
    match s {
        ScalarDType::WeakInt | ScalarDType::WeakFloat => panic!("weak dtype reached LLVM rendering"),
        ScalarDType::Bool => "i1",
        ScalarDType::Int8 | ScalarDType::UInt8 => "i8",
        ScalarDType::Int16 | ScalarDType::UInt16 => "i16",
        ScalarDType::Int32 | ScalarDType::UInt32 => "i32",
        ScalarDType::Int64 | ScalarDType::UInt64 | ScalarDType::Index => "i64",
        ScalarDType::Float16 => "half",
        ScalarDType::BFloat16 => "bfloat",
        ScalarDType::Float32 => "float",
        ScalarDType::Float64 => "double",
        ScalarDType::Void => "void",
        ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => "i8",
        ScalarDType::FP8E4M3FNUZ | ScalarDType::FP8E5M2FNUZ => panic!("FNUZ reached LLVM rendering"),
    }
}

/// Convert a constant value to LLVM literal string.
pub fn lconst(val: &ConstValue, dtype: &DType) -> String {
    match val {
        ConstValue::Invalid => panic!("Invalid reached LLVM constant rendering"),
        ConstValue::Int(i) => i.to_string(),
        ConstValue::UInt(u) => (*u as i64).to_string(),
        ConstValue::Float(f) => format_float(*f, dtype),
        ConstValue::Bool(b) => if *b { "1" } else { "0" }.to_string(),
    }
}

/// Format a float value for LLVM IR.
fn format_float(f: f64, dtype: &DType) -> String {
    let scalar = dtype.base();
    let bits = committed_float_bits(f, scalar).expect("float constant must have a concrete float dtype");
    match scalar {
        ScalarDType::WeakFloat => panic!("weak dtype reached LLVM constant rendering"),
        ScalarDType::Float64 => format!("0x{bits:016X}"),
        ScalarDType::Float32 => {
            // LLVM expects float32 constants in double-precision hex format
            format!("0x{:016X}", (f32::from_bits(bits as u32) as f64).to_bits())
        }
        ScalarDType::Float16 => format!("0xH{:04X}", bits as u16),
        ScalarDType::BFloat16 => format!("0xR{:04X}", bits as u16),
        ScalarDType::FP8E4M3 | ScalarDType::FP8E4M3FNUZ | ScalarDType::FP8E5M2 | ScalarDType::FP8E5M2FNUZ => {
            bits.to_string()
        }
        _ => unreachable!("non-float dtype in format_float"),
    }
}

/// Get LLVM cast instruction name for a type conversion.
///
/// FP8 (E4M3/E5M2) types are mapped to `i8` in LLVM and cannot use `fpext`/`fptrunc`;
/// FP8↔Float must be decomposed via the devectorize fp8 patterns before reaching LLVM,
/// matching tinygrad's dedicated `f32_to_fp8` / `cvt.f32.fp8` intrinsics (`llvmir.py:226-230`).
pub fn lcast(from: &DType, to: &DType) -> &'static str {
    let from_scalar = from.base();
    let to_scalar = to.base();

    debug_assert!(
        !(from_scalar.is_fp8() || to_scalar.is_fp8()),
        "lcast does not support FP8 (mapped to i8); decompose via devectorize fp8 patterns first"
    );

    if matches!(from, DType::Ptr { .. }) || matches!(to, DType::Ptr { .. }) {
        return if matches!(from, DType::Ptr { .. }) && matches!(to, DType::Ptr { .. }) {
            "bitcast"
        } else if matches!(from, DType::Ptr { .. }) {
            "ptrtoint"
        } else {
            "inttoptr"
        };
    }

    if from_scalar.is_float() && to_scalar.is_float() {
        return if to_scalar.bytes() > from_scalar.bytes() { "fpext" } else { "fptrunc" };
    }

    if (from_scalar.is_unsigned() || from_scalar.is_bool()) && to_scalar.is_float() {
        return "uitofp";
    }
    if (from_scalar.is_signed() || from_scalar == ScalarDType::Index) && to_scalar.is_float() {
        return "sitofp";
    }

    if from_scalar.is_float() && to_scalar.is_unsigned() {
        return "fptoui";
    }
    if from_scalar.is_float() && (to_scalar.is_signed() || to_scalar == ScalarDType::Index) {
        return "fptosi";
    }

    // Integer-to-integer casts
    let from_bytes = from_scalar.bytes();
    let to_bytes = to_scalar.bytes();

    // Bool (i1) to any integer type needs zext - i1 is always smaller than i8+
    // Note: Bool.bytes() returns 1 (storage size) but LLVM i1 is 1 bit, not 1 byte
    if from_scalar.is_bool() && !to_scalar.is_bool() {
        return "zext";
    }

    // Any integer to Bool needs trunc - truncate to 1 bit
    if !from_scalar.is_bool() && to_scalar.is_bool() {
        return "trunc";
    }

    // Same size: bitcast (handles signed↔unsigned same-size casts)
    if from_bytes == to_bytes {
        return "bitcast";
    }

    // Narrowing: always trunc
    if to_bytes < from_bytes {
        return "trunc";
    }

    // Widening: use zext for unsigned/bool, sext for signed/Index
    if from_scalar.is_unsigned() || from_scalar.is_bool() {
        return "zext";
    }

    // Index type is treated as signed integer for casting purposes
    if from_scalar.is_signed() || from_scalar == ScalarDType::Index {
        return "sext";
    }

    "bitcast"
}

/// Get LLVM address space number.
pub fn addr_space_num(addrspace: AddrSpace) -> u32 {
    match addrspace {
        AddrSpace::Global => 0,
        AddrSpace::Local => 3,
        AddrSpace::Reg => 5,
        AddrSpace::Swizzled { .. } => 7,
    }
}

#[cfg(test)]
#[path = "../../test/unit/llvm_common_types.rs"]
mod tests;
