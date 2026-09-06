use svod_dtype::{AddrSpace, CudaArch, DType};
use svod_ir::UOp;

use super::*;
use crate::Renderer;
use crate::llvm::LlvmTextRenderer;

pub(crate) const SM75: CudaArch = CudaArch { major: 7, minor: 5 };
pub(crate) const SM86: CudaArch = CudaArch { major: 8, minor: 6 };
pub(crate) const SM89: CudaArch = CudaArch { major: 8, minor: 9 };

/// Run the scheduler's post-optimization with the CUDA profile for `arch`
/// (renderer ops left at `all()` so the renderer's own lowering is exercised),
/// then render for NVPTX. fp8-capable archs take the sm_89 profile directly:
/// `for_cuda_arch` withholds it until the fp8 casts lower, but the `mma.sync`
/// packing below it is already pinned here.
pub(crate) fn render_nvptx_linearized(root: &Arc<UOp>, arch: CudaArch, name: &str) -> crate::RenderedKernel {
    let code_renderer = LlvmTextRenderer::nvptx(arch);
    let profile = if arch.has_fp8() {
        svod_schedule::OptimizerRenderer::cuda_sm89(false)
    } else {
        svod_schedule::OptimizerRenderer::for_cuda_arch(arch)
    };
    let optimizer_renderer = profile.with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        Some(svod_ir::decompositions::nvptx_decomposition_patterns()),
        Some(crate::llvm::nvptx_extra_matcher()),
    );
    let lowered = svod_schedule::apply_post_optimization_with_renderer(root.clone(), &optimizer_renderer)
        .expect("post optimization");
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(lowered).into());
    code_renderer.render(&linear, Some(name)).expect("NVPTX render")
}

/// Render a raw graph with no scheduler pass in between: the way to hand the
/// renderer an op the scheduler would otherwise have decomposed.
fn render_raw(root: Arc<UOp>, arch: CudaArch, name: &str) -> crate::Result<crate::RenderedKernel> {
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(root).into());
    LlvmTextRenderer::nvptx(arch).render(&linear, Some(name))
}

/// Compile `ir` to PTX with `clang --target=nvptx64-nvidia-cuda` and assemble
/// it with `ptxas`, returning the PTX text. Returns `None` without asserting
/// when the host clang has no NVPTX target; the `ptxas` step is skipped when
/// no CUDA toolkit is installed. A PTX module containing `.extern .func` is
/// an intrinsic LLVM did not recognize (it emits the name as an external
/// call), so that is rejected here even though clang accepted the module.
pub(crate) fn assert_ptx_compiles(ir: &str, arch: CudaArch) -> Option<String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let has_target =
        Command::new("clang").arg("--print-targets").output().ok().filter(|out| out.status.success()).is_some_and(
            |out| String::from_utf8_lossy(&out.stdout).lines().any(|line| line.trim_start().starts_with("nvptx64")),
        );
    if !has_target {
        return None;
    }

    let march = format!("-march={arch}");
    let mut child = Command::new("clang")
        .args(["-x", "ir", "-S", "-O3", "--target=nvptx64-nvidia-cuda", &march, "-Wno-override-module", "-", "-o", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn clang");
    child.stdin.take().unwrap().write_all(ir.as_bytes()).expect("write NVPTX IR");
    let output = child.wait_with_output().expect("wait for clang");
    assert!(
        output.status.success(),
        "clang rejected emitted {arch} IR:\n{}\n{ir}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ptx = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(!ptx.contains(".extern .func"), "PTX carries an unresolved intrinsic call:\n{ptx}\n--- IR ---\n{ir}");
    assert!(ptx.contains(".entry"), "PTX has no kernel entry:\n{ptx}");

    let ptxas = ["ptxas", "/opt/cuda/bin/ptxas", "/usr/local/cuda/bin/ptxas"].into_iter().find(|tool| {
        Command::new(tool)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    });
    if let Some(ptxas) = ptxas {
        // ptxas reads files only; tests run in parallel, so the name carries a
        // process-wide counter.
        static PTX_FILES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let serial = PTX_FILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("svod-nvptx-{}-{serial}-{arch}.ptx", std::process::id()));
        std::fs::write(&path, &ptx).expect("write PTX");
        let arch_flag = format!("-arch={arch}");
        let output = Command::new(ptxas).args([&arch_flag, "-o", "/dev/null"]).arg(&path).output().expect("run ptxas");
        let _ = std::fs::remove_file(&path);
        assert!(output.status.success(), "ptxas rejected the PTX:\n{}\n{ptx}", String::from_utf8_lossy(&output.stderr));
    }
    Some(ptx)
}

fn indexed(slot: usize, dtype: DType, index: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(UOp::param(slot, 1, dtype, None)).indices(vec![index]).call().unwrap()
}

fn load(slot: usize, dtype: DType) -> Arc<UOp> {
    UOp::load().index(indexed(slot, dtype, UOp::native_const(0i32))).call()
}

#[test]
fn nvptx_emits_kernel_abi_datalayout_and_launch_bound() {
    let store = indexed(0, DType::Float32, UOp::native_const(0i32)).store(UOp::native_const(1.0f32));
    let result = render_nvptx_linearized(&UOp::sink(vec![store]), SM86, "nvptx_smoke");

    for needle in [
        "target datalayout = \"e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64\"",
        "target triple = \"nvptx64-nvidia-cuda\"",
        "define ptx_kernel void @nvptx_smoke(ptr noalias align 32 %data0)",
        "\"nvvm.maxntid\"=\"1\"",
        "\"no-trapping-math\"=\"true\"",
    ] {
        assert!(result.code.contains(needle), "missing {needle}:\n{}", result.code);
    }
    assert!(!result.code.contains("amdgpu"), "{}", result.code);
    assert_eq!(LlvmTextRenderer::nvptx(SM86).backend_name(), "nvptx");
    if let Some(ptx) = assert_ptx_compiles(&result.code, SM86) {
        assert!(ptx.contains(".visible .entry nvptx_smoke("), "{ptx}");
        assert!(ptx.contains(".maxntid 1"), "{ptx}");
    }
}

#[test]
fn nvptx_special_emits_ctaid_and_tid_sregs() {
    // y[gidx0] = x[lidx0]
    let g = UOp::special(UOp::native_const(8i32), "gidx0".to_string());
    let l = UOp::special(UOp::native_const(4i32), "lidx1".to_string());
    let value = UOp::load().index(indexed(0, DType::Float32, l)).call();
    let store = indexed(1, DType::Float32, g).store(value);

    let result = render_nvptx_linearized(&UOp::sink(vec![store]), SM86, "nvptx_special");

    for needle in [
        "tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()",
        "tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y()",
        "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()",
        "declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()",
        // The local bound (4), not the global one, sets the launch bound.
        "\"nvvm.maxntid\"=\"4\"",
    ] {
        assert!(result.code.contains(needle), "missing {needle}:\n{}", result.code);
    }
    if let Some(ptx) = assert_ptx_compiles(&result.code, SM86) {
        for needle in ["%ctaid.x", "%tid.y", ".maxntid 4"] {
            assert!(ptx.contains(needle), "missing {needle}:\n{ptx}");
        }
    }
}

#[test]
fn nvptx_direct_global_axis_reads_the_block_index() {
    let i = UOp::special(UOp::native_const(64i32), "idx0".to_string());
    let store = indexed(0, DType::Float32, i).store(UOp::native_const(2.0f32));
    let result = render_nvptx_linearized(&UOp::sink(vec![store]), SM86, "nvptx_direct");
    assert!(result.code.contains("@llvm.nvvm.read.ptx.sreg.ctaid.x()"), "{}", result.code);
    assert!(!result.code.contains("sreg.tid"), "{}", result.code);
}

#[test]
fn nvptx_barrier_emits_block_scope_fences_around_bar_sync() {
    let barrier = UOp::noop().barrier(smallvec::SmallVec::new());
    let result = render_nvptx_linearized(&UOp::sink(vec![barrier]), SM86, "nvptx_barrier");

    let body: Vec<&str> = result.code.lines().map(str::trim).collect();
    let at = |needle: &str| {
        body.iter().position(|line| *line == needle).unwrap_or_else(|| panic!("missing {needle}:\n{}", result.code))
    };
    let release = at("fence syncscope(\"block\") release");
    let barrier = at("tail call void @llvm.nvvm.barrier0()");
    let acquire = at("fence syncscope(\"block\") acquire");
    assert!(release < barrier && barrier < acquire, "{}", result.code);
    assert!(result.code.contains("declare void @llvm.nvvm.barrier0()"), "{}", result.code);
    assert!(!result.code.contains("workgroup"), "NVPTX rejects syncscope(\"workgroup\"):\n{}", result.code);
    if let Some(ptx) = assert_ptx_compiles(&result.code, SM86) {
        assert!(ptx.contains("bar.sync"), "{ptx}");
    }
}

#[test]
fn nvptx_define_local_emits_addrspace3_module_global() {
    let local = UOp::buffer(42, 16, DType::Float32, AddrSpace::Local, None);
    let result = render_nvptx_linearized(&UOp::sink(vec![local]), SM86, "nvptx_shared");
    for needle in [
        "@local42 = internal unnamed_addr addrspace(3) global [16 x float] undef, align 16",
        "addrspacecast ptr addrspace(3) @local42 to ptr",
    ] {
        assert!(result.code.contains(needle), "missing {needle}:\n{}", result.code);
    }
    assert_ptx_compiles(&result.code, SM86);
}

/// A shared-memory round trip through the generic pointer: LLVM's address
/// space inference must still select `st.shared`/`ld.shared`.
#[test]
fn nvptx_shared_memory_traffic_selects_shared_instructions() {
    let l = UOp::special(UOp::native_const(32i32), "lidx0".to_string());
    let local = UOp::buffer(0, 32, DType::Float32, AddrSpace::Local, None);
    let slot = UOp::index().buffer(local).indices(vec![l.clone()]).call().unwrap();
    let fill = slot.clone().store(UOp::load().index(indexed(0, DType::Float32, l.clone())).call());
    let barrier = fill.barrier(smallvec::SmallVec::new());
    let read = UOp::load().index(slot.after(smallvec::smallvec![barrier])).call();
    let out = indexed(1, DType::Float32, l).store(read);
    let result = render_nvptx_linearized(&UOp::sink(vec![out]), SM86, "nvptx_shared_traffic");
    if let Some(ptx) = assert_ptx_compiles(&result.code, SM86) {
        for needle in ["st.shared", "ld.shared", "bar.sync"] {
            assert!(ptx.contains(needle), "missing {needle}:\n{ptx}");
        }
    }
}

/// `@llvm.exp2` selects on f16/f32; `Log2` rides `lg2.approx.f32` (widening
/// f16 around it); f64 has neither, so the decomposition set expands both
/// polynomially. Division keeps `contract` only, so PTX gets `div.rn`.
#[test_case::test_case(DType::Float32, &["call float @llvm.exp2.f32(", "call float @llvm.nvvm.lg2.approx.f(float", "declare float @llvm.nvvm.lg2.approx.f(float)", "fdiv contract float"], &["ex2.approx.f32", "lg2.approx.f32", "div.rn.f32"]; "f32")]
#[test_case::test_case(DType::Float16, &["call half @llvm.exp2.f16(", "fpext half", "call float @llvm.nvvm.lg2.approx.f(float", "fptrunc float", "fdiv contract half"], &["ex2.approx.f16", "lg2.approx.f32"]; "f16 widens around lg2")]
#[test_case::test_case(DType::Float64, &["fdiv contract double"], &["div.rn.f64"]; "f64 decomposes polynomially")]
fn nvptx_exp2_log2_lowering(dtype: DType, present: &[&str], ptx_needles: &[&str]) {
    let value = load(0, dtype.clone());
    let result = value.try_exp2().unwrap().try_log2().unwrap().try_div(&value).unwrap();
    let sink = UOp::sink(vec![indexed(1, dtype.clone(), UOp::native_const(0i32)).store(result)]);
    let rendered = render_nvptx_linearized(&sink, SM86, "nvptx_transcendental");

    for needle in present {
        assert!(rendered.code.contains(needle), "missing {needle}:\n{}", rendered.code);
    }
    assert!(!rendered.code.contains(" arcp "), "NVPTX must not permit approximate reciprocal:\n{}", rendered.code);
    assert!(!rendered.code.contains("@llvm.log2."), "`@llvm.log2` has no NVPTX lowering:\n{}", rendered.code);
    if dtype == DType::Float64 {
        assert!(
            !rendered.code.contains("@llvm.exp2.f64") && !rendered.code.contains("lg2.approx"),
            "{}",
            rendered.code
        );
    }
    if let Some(ptx) = assert_ptx_compiles(&rendered.code, SM86) {
        for needle in ptx_needles {
            assert!(ptx.contains(needle), "missing {needle}:\n{ptx}");
        }
        assert!(!ptx.contains("rcp.approx") && !ptx.contains("div.approx"), "{ptx}");
    }
}

/// A vector-width `Log2` splits into per-lane `lg2.approx` calls.
#[test]
fn nvptx_log2_splits_vectors_per_lane() {
    let lanes: smallvec::SmallVec<[Arc<UOp>; 4]> =
        (0..4).map(|lane| UOp::load().index(indexed(0, DType::Float32, UOp::native_const(lane))).call()).collect();
    let vector = UOp::stack(lanes).try_log2().unwrap();
    let out = UOp::new(
        svod_ir::Op::Shrink(svod_ir::ops::Shrink {
            src: UOp::param(1, 8, DType::Float32, None),
            offsets: UOp::native_const(0i32),
            sizes: UOp::native_const(4i32),
        }),
        DType::Float32,
    );
    let rendered = render_raw(UOp::sink(vec![out.store(vector)]), SM86, "nvptx_log2_vec").expect("render");
    assert_eq!(rendered.code.matches("call float @llvm.nvvm.lg2.approx.f(float").count(), 4, "{}", rendered.code);
    for needle in ["extractelement <4 x float>", "insertelement <4 x float>"] {
        assert!(rendered.code.contains(needle), "missing {needle}:\n{}", rendered.code);
    }
    assert_ptx_compiles(&rendered.code, SM86);
}

/// The NVPTX backend cannot select the generic transcendental intrinsics; the
/// renderer refuses them instead of emitting IR that fails in clang or (for
/// `@llvm.erf`, an extern call) only in ptxas. `RendererOps::all()` keeps the
/// scheduler from decomposing them, simulating a capability-list drift.
#[test_case::test_case(|x: &Arc<UOp>| x.try_sin().unwrap(), "Sin"; "sin")]
#[test_case::test_case(|x: &Arc<UOp>| x.erf().unwrap(), "Erf"; "erf")]
fn nvptx_rejects_undecomposed_transcendentals(build: fn(&Arc<UOp>) -> Arc<UOp>, op: &str) {
    let value = load(0, DType::Float32);
    let sink = UOp::sink(vec![indexed(1, DType::Float32, UOp::native_const(0i32)).store(build(&value))]);
    let optimizer_renderer = svod_schedule::OptimizerRenderer::for_cuda_arch(SM86).with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        Some(svod_ir::decompositions::nvptx_decomposition_patterns()),
        Some(crate::llvm::nvptx_extra_matcher()),
    );
    let lowered =
        svod_schedule::apply_post_optimization_with_renderer(sink, &optimizer_renderer).expect("post optimization");
    let err = render_raw(lowered, SM86, "nvptx_undecomposed").expect_err("must fail the render");
    assert!(err.to_string().contains(&format!("un-decomposed {op}")), "{err}");
}

#[test_case::test_case(|x: &Arc<UOp>| x.try_log2().unwrap(), "Log2"; "log2")]
#[test_case::test_case(|x: &Arc<UOp>| x.try_exp2().unwrap(), "Exp2"; "exp2")]
fn nvptx_rejects_f64_exp2_log2_without_decomposition(build: fn(&Arc<UOp>) -> Arc<UOp>, op: &str) {
    let value = load(0, DType::Float64);
    let sink = UOp::sink(vec![indexed(1, DType::Float64, UOp::native_const(0i32)).store(build(&value))]);
    let err = render_raw(sink, SM86, "nvptx_f64").expect_err("f64 transcendentals have no PTX lowering");
    assert!(err.to_string().contains(op), "{err}");
}

#[test]
fn nvptx_rejects_fp8_casts() {
    let sink = UOp::sink(vec![
        indexed(1, DType::FP8E4M3, UOp::native_const(0i32)).store(load(0, DType::Float32).cast(DType::FP8E4M3)),
    ]);
    let err = render_raw(sink, SM89, "nvptx_fp8").expect_err("fp8 conversions are not lowered yet");
    assert!(err.to_string().contains("fp8 cast"), "{err}");
}

/// Conversions the CPU emitter renders generically all select on NVPTX; bf16
/// narrowing takes the integer-domain rounding from the decomposition set and
/// bools store as bytes.
#[test_case::test_case(DType::Float32, DType::Float16, &["fptrunc float", "to half"]; "f32 to f16")]
#[test_case::test_case(DType::Float16, DType::Float32, &["fpext half"]; "f16 to f32")]
#[test_case::test_case(DType::Float32, DType::BFloat16, &["bitcast i16", "to bfloat"]; "f32 to bf16 rounds in integers")]
#[test_case::test_case(DType::BFloat16, DType::Float32, &["fpext bfloat"]; "bf16 to f32")]
#[test_case::test_case(DType::Int32, DType::Float32, &["sitofp i32"]; "i32 to f32")]
#[test_case::test_case(DType::Float32, DType::Int64, &["fptosi float", "to i64"]; "f32 to i64")]
#[test_case::test_case(DType::UInt8, DType::Float16, &["uitofp i8", "to half"]; "u8 to f16")]
#[test_case::test_case(DType::Float32, DType::Bool, &["fcmp contract une float", "store i8"]; "f32 to bool stores a byte")]
fn nvptx_casts_select(from: DType, to: DType, present: &[&str]) {
    let sink = UOp::sink(vec![indexed(1, to.clone(), UOp::native_const(0i32)).store(load(0, from).cast(to.clone()))]);
    let rendered = render_nvptx_linearized(&sink, SM86, "nvptx_cast");
    for needle in present {
        assert!(rendered.code.contains(needle), "missing {needle}:\n{}", rendered.code);
    }
    if to == DType::BFloat16 {
        assert!(!rendered.code.contains("fptrunc float"), "{}", rendered.code);
    }
    assert_ptx_compiles(&rendered.code, SM86);
}

#[test]
fn nvptx_shfl_bfly_and_globaltimer_hoist_their_declares() {
    let l = UOp::special(UOp::native_const(32i32), "lidx0".to_string());
    let value = UOp::load().index(indexed(0, DType::Float32, l.clone())).call();
    let partner = shfl_bfly(&value, &UOp::native_const(16i32));
    let stamp = globaltimer().cast(DType::Float32);
    let out = indexed(1, DType::Float32, l).store(partner.try_add(&stamp).unwrap());
    let rendered = render_nvptx_linearized(&UOp::sink(vec![out]), SM86, "nvptx_warp");

    for needle in [
        "call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %",
        ", i32 16, i32 31)",
        "call i64 @llvm.nvvm.read.ptx.sreg.globaltimer()",
        "bitcast float %",
        "to i32",
    ] {
        assert!(rendered.code.contains(needle), "missing {needle}:\n{}", rendered.code);
    }
    for decl in [
        "declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)",
        "declare i64 @llvm.nvvm.read.ptx.sreg.globaltimer()",
    ] {
        assert_eq!(rendered.code.matches(decl).count(), 1, "{decl} must be hoisted exactly once:\n{}", rendered.code);
    }
    if let Some(ptx) = assert_ptx_compiles(&rendered.code, SM86) {
        for needle in ["shfl.sync.bfly.b32", "%globaltimer"] {
            assert!(ptx.contains(needle), "missing {needle}:\n{ptx}");
        }
    }
}

/// Register-resident scratch stays a generic `alloca` (PTX `.local`), unlike
/// AMD's addrspace(5) form.
#[test]
fn nvptx_register_buffers_use_generic_alloca() {
    let reg = UOp::buffer(3, 4, DType::Float32, AddrSpace::Reg, None);
    let slot = UOp::index().buffer(reg).indices(vec![UOp::native_const(1i32)]).call().unwrap();
    let fill = slot.clone().store(load(0, DType::Float32));
    let read = UOp::load().index(slot.after(smallvec::smallvec![fill])).call();
    let sink = UOp::sink(vec![indexed(1, DType::Float32, UOp::native_const(0i32)).store(read)]);
    let rendered = render_raw(sink, SM86, "nvptx_reg").expect("render");
    assert!(rendered.code.contains("alloca [4 x float]"), "{}", rendered.code);
    assert!(!rendered.code.contains("addrspace(5)"), "{}", rendered.code);
    assert_ptx_compiles(&rendered.code, SM86);
}

#[test]
fn nvptx_renders_for_older_capabilities() {
    assert_eq!(SM75.wave_size(), 32);
    let old = render_nvptx_linearized(
        &UOp::sink(vec![indexed(0, DType::Float32, UOp::native_const(0i32)).store(UOp::native_const(1.0f32))]),
        SM75,
        "nvptx_sm75",
    );
    assert_ptx_compiles(&old.code, SM75);
}

/// One warp-uniform value shuffled to each lane: `idx`/`up`/`down`/`bfly`
/// share the intrinsic shape and differ only in the mode and the clamp word.
#[test_case::test_case(ShflMode::Idx, "idx", 31; "idx")]
#[test_case::test_case(ShflMode::Up, "up", 0; "up clamps at lane 0")]
#[test_case::test_case(ShflMode::Down, "down", 31; "down")]
#[test_case::test_case(ShflMode::Bfly, "bfly", 31; "bfly")]
fn nvptx_shfl_modes(mode: ShflMode, suffix: &str, clamp: i32) {
    let l = UOp::special(UOp::native_const(32i32), "lidx0".to_string());
    let value = UOp::load().index(indexed(0, DType::Int32, l.clone())).call();
    let out = indexed(1, DType::Int32, l).store(shfl(mode, &value, &UOp::native_const(3i32)));
    let rendered = render_nvptx_linearized(&UOp::sink(vec![out]), SM86, "nvptx_shfl_mode");

    let call = format!("call i32 @llvm.nvvm.shfl.sync.{suffix}.i32(i32 -1, i32 %");
    assert!(rendered.code.contains(&call), "missing {call}:\n{}", rendered.code);
    assert!(rendered.code.contains(&format!(", i32 3, i32 {clamp})")), "{}", rendered.code);
    assert!(!rendered.code.contains("bitcast"), "i32 needs no reinterpretation:\n{}", rendered.code);
    let decl = format!("declare i32 @llvm.nvvm.shfl.sync.{suffix}.i32(i32, i32, i32, i32)");
    assert_eq!(rendered.code.matches(&decl).count(), 1, "{}", rendered.code);
    if let Some(ptx) = assert_ptx_compiles(&rendered.code, SM86) {
        let needle = format!("shfl.sync.{suffix}.b32");
        assert!(ptx.contains(&needle), "missing {needle}:\n{ptx}");
    }
}

/// 16-bit scalars widen into the shuffled word and truncate back (signed
/// ones sign-extend, which the truncation makes irrelevant); floats
/// reinterpret through their integer bit pattern on both sides.
#[test_case::test_case(DType::Float16, "zext", 2; "f16")]
#[test_case::test_case(DType::BFloat16, "zext", 2; "bf16")]
#[test_case::test_case(DType::Int16, "sext", 0; "i16")]
#[test_case::test_case(DType::UInt16, "zext", 0; "u16")]
fn nvptx_shfl_widens_16bit_values(dtype: DType, widen: &str, bitcasts: usize) {
    let l = UOp::special(UOp::native_const(32i32), "lidx0".to_string());
    let value = UOp::load().index(indexed(0, dtype.clone(), l.clone())).call();
    let out = indexed(1, dtype.clone(), l.clone()).store(shfl_idx(&value, &l));
    let rendered = render_nvptx_linearized(&UOp::sink(vec![out]), SM86, "nvptx_shfl_16");

    for needle in [
        &format!("{widen} i16 %"),
        "to i32",
        "call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %",
        "trunc i32 %",
        "to i16",
    ] {
        assert!(rendered.code.contains(needle), "missing {needle}:\n{}", rendered.code);
    }
    assert_eq!(rendered.code.matches("bitcast ").count(), bitcasts, "{}", rendered.code);
    if let Some(ptx) = assert_ptx_compiles(&rendered.code, SM86) {
        assert!(ptx.contains("shfl.sync.idx.b32"), "{ptx}");
    }
}

/// A packed pair (`<2 x half>`, the mma.sync fragment word) is one register.
#[test]
fn nvptx_shfl_moves_a_packed_half_pair_as_one_word() {
    let pair = load(0, DType::Float32).bitcast(DType::Float16.vec(2).unwrap());
    let shuffled = shfl_down(&pair, &UOp::native_const(1i32));
    let out = indexed(1, DType::Float16, UOp::native_const(0i32)).store(shuffled);
    let rendered = render_raw(UOp::sink(vec![out]), SM86, "nvptx_shfl_pair").expect("render");
    for needle in [
        "bitcast <2 x half> %",
        "to i32",
        "call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %",
        "bitcast i32 %",
        "to <2 x half>",
    ] {
        assert!(rendered.code.contains(needle), "missing {needle}:\n{}", rendered.code);
    }
    assert!(!rendered.code.contains("zext"), "{}", rendered.code);
    assert_ptx_compiles(&rendered.code, SM86);
}

/// A `STACK` of lanes is shaped, not packed: its casts are elementwise.
#[test]
#[should_panic(expected = "split a shaped")]
fn nvptx_shfl_rejects_shaped_values() {
    let lanes: smallvec::SmallVec<[Arc<UOp>; 4]> =
        (0..2).map(|lane| UOp::load().index(indexed(0, DType::Float16, UOp::native_const(lane))).call()).collect();
    shfl_idx(&UOp::stack(lanes), &UOp::native_const(0i32));
}

#[test]
#[should_panic(expected = "one 32-bit register")]
fn nvptx_shfl_rejects_wide_values() {
    shfl_idx(&load(0, DType::Float64), &UOp::native_const(0i32));
}

/// The nvvm builders are NVPTX-only: rendering them for the CPU or an AMD
/// target fails with a typed error before clang could turn the intrinsic
/// into an extern call.
#[test_case::test_case(LlvmTextRenderer::new(), "cpu"; "cpu")]
#[test_case::test_case(LlvmTextRenderer::amd(svod_dtype::AmdArch::Gfx942), "gfx942"; "amd")]
fn nvvm_builders_are_rejected_on_other_targets(renderer: LlvmTextRenderer, target: &str) {
    let value = load(0, DType::Float32);
    let out = indexed(1, DType::Float32, UOp::native_const(0i32)).store(shfl_idx(&value, &UOp::native_const(0i32)));
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(UOp::sink(vec![out])).into());
    let err = renderer.render(&linear, Some("foreign")).expect_err("nvvm intrinsics have no lowering here");
    match &err {
        crate::Error::ForeignIntrinsic { intrinsic, target: reported } => {
            assert_eq!(intrinsic, "llvm.nvvm.shfl.sync.idx.i32");
            assert_eq!(reported, target);
        }
        other => panic!("expected ForeignIntrinsic, got {other}"),
    }
}
