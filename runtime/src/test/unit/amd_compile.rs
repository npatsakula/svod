use super::*;

/// The ROCm device libraries are only searched when the IR still calls an f64
/// `__ocml_*` entry point; everything else is an `@llvm.*` intrinsic the
/// AMDGPU backend selects on its own.
#[test_case::test_case("  %r = call float @llvm.exp2.f32(float %v)" => true; "llvm intrinsics only")]
#[test_case::test_case("  %r = call double @__ocml_exp2_f64(double %v)" => false; "f64 ocml transcendental")]
fn nogpulib_tracks_device_library_use(body: &str) -> bool {
    amd_object_flags(body, AmdArch::Gfx1100).iter().any(|flag| flag == "-nogpulib")
}

/// Round-trips a tiny AMD kernel through clang, and pins that the resulting
/// code object only validates against the arch and entry point it was built
/// for. Without an AMDGPU target, the failure must at least be a clean error.
#[test]
fn compile_smoke_gfx1100() {
    if !has_amdgpu_target() {
        let err = compile_ir_to_amd_object("; empty\n", AmdArch::Gfx1100).expect_err("must fail without amdgpu target");
        assert!(format!("{err}").contains("AMDGPU"), "unexpected error message: {err}");
        return;
    }

    let ir = r#"; ModuleID = 'amd_smoke'
source_filename = "amd_smoke"
target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.workitem.id.x()
declare float @llvm.exp2.f32(float)

define amdgpu_kernel void @amd_smoke(ptr noalias %buf0) #0 {
entry:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid_ext = zext i32 %tid to i64
  %p = getelementptr inbounds float, ptr %buf0, i64 %tid_ext
  %v = load float, ptr %p
  %e = call float @llvm.exp2.f32(float %v)
  store float %e, ptr %p
  ret void
}

attributes #0 = { alwaysinline nounwind "no-builtins" "amdgpu-flat-work-group-size"="1,256" "no-trapping-math"="true" }
"#;
    let obj = compile_ir_to_amd_object(ir, AmdArch::Gfx1100).expect("amdgcn compile");
    assert_eq!(&obj[..4], b"\x7fELF", "AMDGPU code objects are ELF");
    validate_amd_object(&obj, AmdArch::Gfx1100, "amd_smoke").expect("valid gfx1100 object");
    assert!(validate_amd_object(&obj, AmdArch::Gfx1101, "amd_smoke").is_err(), "wrong target arch must fail");
    assert!(validate_amd_object(&obj, AmdArch::Gfx1100, "other_kernel").is_err(), "wrong kernel must fail");
}

/// AMD kernels narrow f32 to bf16 in integers under a clang older than LLVM 21,
/// or one whose version cannot be read, and leave it to the backend from 21 on.
#[test_case::test_case(None => true; "unknown version")]
#[test_case::test_case(Some(18) => true; "llvm 18")]
#[test_case::test_case(Some(20) => true; "llvm 20")]
#[test_case::test_case(Some(21) => false; "llvm 21")]
#[test_case::test_case(Some(22) => false; "llvm 22")]
fn amd_bf16_narrowing_follows_the_llvm_version(major: Option<u32>) -> bool {
    crate::devices::amd::amd_bf16_casts_in_integers(major)
}
