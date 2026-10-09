use super::*;

use AmdArch::{Gfx942, Gfx950, Gfx1100, Gfx1151, Gfx1201};
use ScalarDType::{BFloat16, Bool, FP8E4M3, FP8E5M2, Float16, Float32, Int8, Int32};

/// Which `llvm.amdgcn.{mfma,wmma}.*` intrinsic each `(arch, in, acc, K)` selects.
///
/// `None` means "no selectable intrinsic", which forces upstream decomposition.
/// Naming an unselectable intrinsic is strictly worse: LLVM lowers the unknown
/// name to a silent extern call rather than erroring. Every `None` row was
/// verified with `llc -mcpu=<arch>`, and K coverage is exhaustive per family
/// because catch-all match arms used to mint names such as
/// `mfma.scale.f32.16x16x128f16` for any K.
#[test_case::test_case(Gfx1100, Bool, Float32, 16 => None; "no wmma flavor takes bool")]
// CDNA MFMA: f32 exists only at K=4, fp8 only at K=32.
#[test_case::test_case(Gfx942, BFloat16, Float32, 16 => Some("llvm.amdgcn.mfma.f32.16x16x16bf16.1k".into()))]
#[test_case::test_case(Gfx942, Float16, Float32, 16 => Some("llvm.amdgcn.mfma.f32.16x16x16f16".into()))]
#[test_case::test_case(Gfx942, Float32, Float32, 4 => Some("llvm.amdgcn.mfma.f32.16x16x4f32".into()))]
#[test_case::test_case(Gfx942, Float32, Float32, 8 => None)]
#[test_case::test_case(Gfx942, Float32, Float32, 16 => None)]
#[test_case::test_case(Gfx942, Float32, Float32, 32 => None)]
#[test_case::test_case(Gfx942, FP8E4M3, Float32, 32 => Some("llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8".into()))]
#[test_case::test_case(Gfx942, FP8E5M2, Float32, 32 => Some("llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8".into()))]
#[test_case::test_case(Gfx942, FP8E4M3, Float32, 16 => None)]
#[test_case::test_case(Gfx950, FP8E4M3, Float32, 16 => None)]
#[test_case::test_case(Gfx950, FP8E4M3, Float32, 32 => Some("llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8".into()); "cdna4 keeps the unscaled k32 fp8 mfma")]
// The dotted K=32 double-rate forms and the scaled K=128 form are CDNA4-only,
// and `scale.` keys on the `.f8f6f4` suffix — the only scaled MFMA family.
#[test_case::test_case(Gfx950, Float16, Float32, 32 => Some("llvm.amdgcn.mfma.f32.16x16x32.f16".into()))]
#[test_case::test_case(Gfx950, BFloat16, Float32, 32 => Some("llvm.amdgcn.mfma.f32.16x16x32.bf16".into()))]
#[test_case::test_case(Gfx942, Float16, Float32, 32 => None; "dotted k32 f16 is gfx950 only")]
#[test_case::test_case(Gfx942, BFloat16, Float32, 32 => None; "dotted k32 bf16 is gfx950 only")]
#[test_case::test_case(Gfx950, FP8E4M3, Float32, 128 => Some("llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4".into()))]
#[test_case::test_case(Gfx942, FP8E4M3, Float32, 128 => None; "gfx942 has no scaled k128 mfma")]
#[test_case::test_case(Gfx950, Float16, Float32, 8 => None)]
#[test_case::test_case(Gfx950, Float16, Float32, 64 => None)]
#[test_case::test_case(Gfx950, Float16, Float32, 128 => None; "scaled mfma is fp8 only")]
#[test_case::test_case(Gfx950, BFloat16, Float32, 8 => None)]
#[test_case::test_case(Gfx950, BFloat16, Float32, 64 => None)]
#[test_case::test_case(Gfx950, BFloat16, Float32, 128 => None)]
// RDNA3 WMMA, plus RDNA4's LLVM-overloaded names.
#[test_case::test_case(Gfx1100, Float16, Float32, 16 => Some("llvm.amdgcn.wmma.f32.16x16x16.f16".into()))]
#[test_case::test_case(Gfx1201, Float16, Float32, 16 => Some("llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16".into()))]
#[test_case::test_case(Gfx1201, BFloat16, BFloat16, 16 => Some("llvm.amdgcn.wmma.bf16.16x16x16.bf16.v8i16.v8i16".into()))]
#[test_case::test_case(Gfx1201, FP8E4M3, Float32, 16 => None; "rdna4 has no native fp8 wmma")]
#[test_case::test_case(Gfx1151, FP8E4M3, Float32, 16 => None)]
#[test_case::test_case(Gfx1151, FP8E4M3, Float32, 32 => None; "rdna must not inherit the cdna fp8 mfma")]
// The accumulator is pinned to the operand and RDNA's WMMA is K=16 only; the
// `iu8` names both RDNA families select are pinned with their packed operands by
// `amd_wmma_operand_packing`'s int8 rows.
#[test_case::test_case(Gfx1201, Int8, Float32, 16 => None; "the int8 wmma only accumulates in i32")]
#[test_case::test_case(Gfx1201, Int8, Int32, 32 => None; "rdna wmma is k16 only")]
fn intrinsic_selection(arch: AmdArch, in_dt: ScalarDType, acc_dt: ScalarDType, k: usize) -> Option<String> {
    resolve_intrinsic(arch, Some(in_dt), Some(acc_dt), (16, 16, k))
}
