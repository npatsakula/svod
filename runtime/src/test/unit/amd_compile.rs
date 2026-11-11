use super::*;

/// The ROCm device libraries are only searched when the IR still calls an f64
/// `__ocml_*` entry point; everything else is an `@llvm.*` intrinsic the
/// AMDGPU backend selects on its own.
#[test_case::test_case("  %r = call float @llvm.exp2.f32(float %v)" => true; "llvm intrinsics only")]
#[test_case::test_case("  %r = call double @__ocml_exp2_f64(double %v)" => false; "f64 ocml transcendental")]
fn nogpulib_tracks_device_library_use(body: &str) -> bool {
    amd_object_flags(body, AmdArch::Gfx1100, Some(23)).iter().any(|flag| flag == "-nogpulib")
}

/// clang 20 compiles some spilling AMDGPU kernels to wrong values, so the AMD
/// backend refuses a clang older than 22 and names the one it found; a version
/// string it cannot read passes.
#[test_case::test_case("Ubuntu clang version 20.1.2 (0ubuntu1~24.04.3)\nTarget: x86_64-pc-linux-gnu" => false; "ubuntu 20")]
#[test_case::test_case("Ubuntu clang version 22.1.8 (++20260714014902+ca7933e47d3a-1~exp1)" => true; "ubuntu 22")]
#[test_case::test_case("a vendor compiler, build 7" => true; "unreadable passes")]
fn amd_refuses_a_clang_older_than_22(version: &str) -> bool {
    check_amd_clang(version)
        .inspect_err(|error| {
            let found = version.lines().next().unwrap();
            assert!(error.to_string().contains(found), "the error names {found:?}: {error}");
        })
        .is_ok()
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

/// A matrix product whose result only a lane-gated store reads stays ahead of
/// the store's branch: `v_wmma` reads every lane's operands, and inside the
/// branch the lanes it skips never write them (machine sinking moves it there
/// under LLVM 22 and ROCm 7.2's clang, which leave it unflagged convergent).
#[test_case::test_case(AmdArch::Gfx1151, "<16 x i16>", "v16i16"; "gfx1151")]
#[test_case::test_case(AmdArch::Gfx1201, "<8 x i16>", "v8i16"; "gfx1201")]
fn a_matrix_product_stays_outside_a_divergent_store(arch: AmdArch, operand: &str, suffix: &str) {
    if !has_amdgpu_target() {
        eprintln!("skipped: clang has no AMDGPU target");
        return;
    }
    let wmma = format!("@llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.{suffix}");
    let ir = format!(
        r#"target triple = "amdgcn-amd-amdhsa"

declare <8 x float> {wmma}({operand}, {operand}, <8 x float>)
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @gated(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %o) {{
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %pa = getelementptr {operand}, ptr addrspace(1) %a, i32 %tid
  %pb = getelementptr {operand}, ptr addrspace(1) %b, i32 %tid
  %va = load {operand}, ptr addrspace(1) %pa
  %vb = load {operand}, ptr addrspace(1) %pb
  %d = call <8 x float> {wmma}({operand} %va, {operand} %vb, <8 x float> zeroinitializer)
  %inside = icmp ult i32 %tid, 16
  br i1 %inside, label %store, label %end
store:
  %po = getelementptr <8 x float>, ptr addrspace(1) %o, i32 %tid
  store <8 x float> %d, ptr addrspace(1) %po
  br label %end
end:
  ret void
}}
"#
    );
    let asm = String::from_utf8(compile_ir_to_amd_asm(&ir, arch).expect("amdgcn compile")).unwrap();
    let branch = asm.find("s_cbranch_execz").expect("the store's branch");
    assert!(asm[..branch].contains("v_wmma"), "the product sinks into the store's branch:\n{asm}");
}

/// Machine sinking is off for every clang whose matrix instructions it may
/// sink, read from its banner (ROCm's `AMD clang version 22.0.0git` included),
/// and for one whose version it cannot read; only IR holding a matrix product
/// loses it.
#[test_case::test_case("AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 f58b06dc)" => true; "rocm 7.2")]
#[test_case::test_case("Ubuntu clang version 22.1.8 (++20260714014902+ca7933e47d3a-1~exp1)" => true; "ubuntu 22")]
#[test_case::test_case("clang version 23.1.4" => false; "llvm 23")]
#[test_case::test_case("a vendor compiler, build 7" => true; "unreadable")]
fn machine_sinking_is_off_where_it_sinks_a_matrix_product(version: &str) -> bool {
    let major = crate::clang::clang_major_version(version);
    let sinks = |ir| amd_object_flags(ir, AmdArch::Gfx1151, major).iter().any(|flag| flag == "-disable-machine-sink");
    assert!(!sinks("define void @f() { ret void }"), "IR without a matrix product keeps machine sinking");
    for product in ["@llvm.amdgcn.wmma.f32.16x16x16.f16", "@llvm.amdgcn.mfma.f32.16x16x16f16"] {
        assert_eq!(sinks(product), amd_sinks_wmma(major), "{product}");
    }
    amd_sinks_wmma(major)
}

/// A kernel's object key carries the flags it compiles with: the same IR keyed
/// under a flag policy that sank its matrix product never matches.
#[test]
fn an_object_key_carries_the_kernels_own_flags() {
    let identity = crate::object_cache::CompilerIdentity {
        schema: crate::object_cache::OBJECT_CACHE_SCHEMA,
        backend: "amd-clang".into(),
        target_architecture: "amdgcn-amd-amdhsa/gfx1151".into(),
        toolchain: "clang".into(),
        flags: amd_object_flags("", AmdArch::Gfx1151, Some(22)),
        abi: String::new(),
        object_format: String::new(),
    };
    let ir = "call @llvm.amdgcn.wmma.f32.16x16x16.f16";
    let key = amd_object_key(&identity, ir, AmdArch::Gfx1151, Some(22));
    assert!(key.compiler.flags.iter().any(|flag| flag == "-disable-machine-sink"));
    assert_ne!(key.digest(), crate::object_cache::ObjectCacheKey::new(ir.as_bytes(), identity).digest());
}
