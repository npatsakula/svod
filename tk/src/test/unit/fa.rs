//! Flash-attention forward — GPU-free graph-shape / render checks plus the
//! gfx942, gfx1151 and CUDA sm_80+ comparisons against
//! `Tensor::scaled_dot_product_attention`.

use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec};
use svod_ir::UOp;
use svod_tensor::Tensor;

use crate::Kernel;
use crate::kernels::fa::{FaConfig, FaOpts, build_fa_mw_rdb, flash_attention_with};
use svod_ir::ops;

/// A non-rank-4 `q`/`k` operand is a structured `Err` (not a panic). The shape
/// preconditions resolve before any device dispatch, so this runs GPU-free.
#[test]
fn flash_attention_with_non_rank4_operand_is_operand_shape_err() {
    let q4 = Tensor::randn(&[1, 128, 4, 64]).expect("randn");
    let q3 = Tensor::randn(&[128, 4, 64]).expect("randn"); // q: rank 3
    let e = flash_attention_with(&q3, &q4, &q4, FaOpts::default()).err().expect("rank-3 q must error, not panic");
    assert!(matches!(e, crate::launch::Error::OperandRank { operand: "q", .. }), "got {e:?}");
    let k2 = Tensor::randn(&[4, 64]).expect("randn"); // k: rank 2
    let e = flash_attention_with(&q4, &k2, &q4, FaOpts::default()).err().expect("rank-2 k must error, not panic");
    assert!(matches!(e, crate::launch::Error::OperandRank { operand: "k", .. }), "got {e:?}");
}

/// `(o, q, k, v)` dummy BUFFER UOps for a GPU-free FA build.
fn dummy_fa_buffers(b: usize, n: usize, h: usize, h_kv: usize, d: usize) -> Vec<Arc<UOp>> {
    let q_sz = b * n * h * d;
    let kv_sz = b * n * h_kv * d;
    vec![
        UOp::new_buffer(DeviceSpec::Cpu, q_sz, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, q_sz, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, kv_sz, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, kv_sz, DType::BFloat16),
    ]
}

/// The target gate rejects a non-AMD device up front (host — no GPU needed): a
/// CPU spec resolves to no AMD arch, so `check_target` errs `UnsupportedArch`
/// instead of letting an AMD-only kernel mis-render/compile-fail later. The
/// gate is generic over the kernel's declared `FA_SUPPORTED_ARCHS`, not a hardcoded
/// arch — adding another GPU is extending that list, not rewriting the gate.
#[test]
fn test_target_gate_rejects_non_amd() {
    use crate::kernels::fa::FA_SUPPORTED_ARCHS;
    assert!(!crate::flash_attention_supported(&DeviceSpec::Cpu));
    let err = crate::target::check_target(&DeviceSpec::Cpu, FA_SUPPORTED_ARCHS);
    assert!(
        matches!(err, Err(crate::launch::Error::UnsupportedArch { .. })),
        "a CPU device must be rejected by the AMD gate, got {err:?}"
    );
}

/// Render-regression (CPU, no GPU): the rolled-db FA must render to AMD LLVM IR
/// in **bounded** memory/time. The `FloorMod`-based prefetch clamp in
/// [`build_fa_mw_rdb`] avoids a `WHERE` in the prefetch-block index that the
/// renderer mis-orders past its address-MUL consumer (which would make it hit
/// `RenderContext::get` on an unrendered node and `{:?}` the whole shared graph).
/// This test renders the full kernel and asserts it returns.
#[test]
fn test_fa_mw_rdb_renders_bounded() {
    let (b, h, h_kv, d) = (1usize, 2, 2, 64);
    let n = 128usize;
    let render = |unroll: bool| {
        let ker = Kernel::new(
            "fa_mw_rdb",
            [h as i64, (n / 16 / 8) as i64, b as i64],
            8 * 64,
            dummy_fa_buffers(b, n, h, h_kv, d),
            crate::ArchCaps::GFX942,
        );
        build_fa_mw_rdb(
            &ker,
            b,
            n,
            h,
            h_kv,
            d,
            FaConfig { q_blk: 16, kv_blk: 16, unroll, ..Default::default() },
            svod_dtype::DType::BFloat16,
            false,
        );
        let sink = ker.finish(1);
        let pm = svod_schedule::symbolic::pm_lower_index_dtype()
            + svod_ir::decompositions::divmod_decomposition_patterns()
                .with_context::<svod_schedule::symbolic::WeakMemo>();
        let lowered = svod_schedule::graph_rewrite(&pm, sink, &mut svod_schedule::symbolic::WeakMemo::default());
        let program = svod_codegen::program_pipeline::program_from_sink(lowered, svod_dtype::DeviceSpec::Cpu)
            .expect("final target graph");
        let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
        let linear_uop = linearized
            .toposort()
            .into_iter()
            .find(|u| matches!(u.op(), svod_ir::Op::Linear(..)))
            .expect("LINEAR present");
        let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(svod_dtype::AmdArch::Gfx942);
        // Returns (no OOM/hang) ⇒ the FloorMod-clamped prefetch index renders.
        svod_codegen::traits::Renderer::render(&renderer, &linear_uop, Some("fa_rdb")).expect("render").code
    };

    // The attention marker lowers (Stage 1) to a single backend-delegated interleave
    // at the loop top (Stage 2 will swap this for the softmax/MFMA comb).
    let rolled = render(false);
    assert_eq!(
        rolled.matches("call void @llvm.amdgcn.iglp.opt(i32 0)").count(),
        1,
        "marker lowered to one iglp delegation"
    );

    // Flatness (P1): with the unroll flag the QKᵀ (4 K-steps) and A·V (4 output
    // fragments) MFMAs render as 8 distinct flat `mfma` call sites — the rolled
    // form keeps them looped (strictly fewer). The exp2 online softmax likewise
    // leaves the rolled loop bodies. This is the comb's prerequisite.
    let mfma =
        |code: &str| code.lines().filter(|l| l.contains("mfma.f32.16x16x16bf16.1k") && !l.contains("declare")).count();
    let flat = render(true);
    assert_eq!(mfma(&flat), 8, "unrolled FA slice renders 8 flat mfma (4 QKᵀ + 4 A·V)");
    assert!(mfma(&rolled) < 8, "rolled FA slice keeps the QKᵀ/A·V loops ({} < 8 mfma)", mfma(&rolled));
}

/// Host regression guard for the **graph/realize-path** lowering of a hand-lowered
/// (`opts_to_apply=Some(vec![])`) tile-kernel SINK on AMD. Mirrors the realize
/// optimize→render path: `optimize_kernel_with_config` → `decompose_with` →
/// `program_from_sink` → `do_linearize` → `type_verify`. There is no
/// `Op::Special` bypass: the SINK's `opts_to_apply = Some(vec![])` makes the
/// optimizer apply zero schedule opts, and the body then runs the shared
/// pre/post-optimization pipeline. Renders identically to the direct path
/// (`test_fa_mw_rdb_renders_bounded`): rolled QKᵀ/A·V loops, one iglp.
#[test]
fn test_fa_graph_path_renders_clean() {
    use svod_schedule::{OptimizerConfig, OptimizerRenderer, optimize_kernel_with_config};

    let (b, h, h_kv, d) = (1usize, 2, 2, 64);
    let n = 128usize;
    let ker = Kernel::new(
        "fa_mw_rdb",
        [h as i64, (n / 16 / 8) as i64, b as i64],
        8 * 64,
        dummy_fa_buffers(b, n, h, h_kv, d),
        crate::ArchCaps::GFX942,
    );
    build_fa_mw_rdb(
        &ker,
        b,
        n,
        h,
        h_kv,
        d,
        FaConfig { q_blk: 16, kv_blk: 16, ..Default::default() },
        svod_dtype::DType::BFloat16,
        false,
    );
    let sink = ker.finish(1);

    // Realize builds the optimizer renderer for gfx942 via for_amd_arch.
    let text_ren = svod_codegen::llvm::LlvmTextRenderer::amd(svod_dtype::AmdArch::Gfx942);
    let opt_ren = OptimizerRenderer::for_amd_arch(svod_dtype::AmdArch::Gfx942).with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        svod_codegen::traits::Renderer::decompositor(&text_ren),
        None,
    );
    let config = OptimizerConfig::default();
    let optimized = optimize_kernel_with_config(sink, &opt_ren, &config).expect("optimize");

    let program = svod_codegen::program_pipeline::program_from_sink(optimized, svod_dtype::DeviceSpec::Cpu)
        .expect("final target graph");
    let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
    let linear_uop =
        linearized.toposort().into_iter().find(|u| matches!(u.op(), svod_ir::Op::Linear(..))).expect("LINEAR present");
    // This is the verify the real do_render runs (program_pipeline.rs:129-131).
    let svod_ir::Op::Linear(ops::Linear { ops }) = linear_uop.op() else { unreachable!() };
    let verify_root = svod_ir::UOp::sink(ops.iter().cloned().collect());
    svod_schedule::spec::type_verify(&verify_root, &svod_schedule::spec::spec_program())
        .expect("type_verify must pass (the Ptr{vcount:4} failure surfaces here)");
    let code = svod_codegen::traits::Renderer::render(&text_ren, &linear_uop, Some("fa_rdb")).expect("render").code;
    let mfma = code.lines().filter(|l| l.contains("mfma.f32.16x16x16bf16.1k") && !l.contains("declare")).count();
    assert!(mfma > 0, "graph-path FA must render mfma calls, got {mfma}");
    // Matches the direct-path render contract (test_fa_mw_rdb_renders_bounded):
    // the rolled QKᵀ/A·V loops kept (< 8 flat sites), one iglp delegation.
    assert!(mfma < 8, "rolled graph FA keeps the QKᵀ/A·V loops ({mfma} < 8)");
    assert_eq!(
        code.matches("call void @llvm.amdgcn.iglp.opt(i32 0)").count(),
        1,
        "marker lowered to one iglp delegation (identical to the direct path)"
    );
}

/// Host regression guard for the **graph/realize-path** lowering of a hand-lowered
/// FA SINK on **RDNA3.5 (gfx1151, wave32)** — the wave32 peer of
/// `test_fa_graph_path_renders_clean`. `opts_to_apply = Some(vec![])` makes the
/// optimizer apply zero schedule opts; the body then runs the shared
/// pre/post-optimization pipeline and renders to gfx11 LLVM IR. Asserts the rendered IR contains zero `x ptr>` tokens (the illegal
/// vector-of-pointers shape) and that the wave32 WMMA calls are present.
#[test]
fn test_fa_graph_path_renders_clean_gfx1151() {
    use svod_schedule::{OptimizerConfig, OptimizerRenderer, optimize_kernel_with_config};

    let (b, h, h_kv, d) = (1usize, 16, 16, 64);
    let n = 512usize;
    let (q_blk, kv_blk) = (16usize, 16usize);
    let ker = Kernel::new(
        "fa_mw_rdb_w32",
        [h as i64, (n / q_blk / 8) as i64, b as i64],
        8 * 32, // NUM_WARPS * wave32
        dummy_fa_buffers(b, n, h, h_kv, d),
        crate::ArchCaps::for_amd(svod_dtype::AmdArch::Gfx1151),
    );
    build_fa_mw_rdb(
        &ker,
        b,
        n,
        h,
        h_kv,
        d,
        FaConfig { q_blk, kv_blk, causal: false, ..Default::default() },
        svod_dtype::DType::BFloat16,
        false,
    );
    let sink = ker.finish(1);

    let text_ren = svod_codegen::llvm::LlvmTextRenderer::amd(svod_dtype::AmdArch::Gfx1151);
    let opt_ren = OptimizerRenderer::for_amd_arch(svod_dtype::AmdArch::Gfx1151).with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        svod_codegen::traits::Renderer::decompositor(&text_ren),
        None,
    );
    let config = OptimizerConfig::default();
    let optimized = optimize_kernel_with_config(sink, &opt_ren, &config).expect("optimize");

    let program = svod_codegen::program_pipeline::program_from_sink(optimized, svod_dtype::DeviceSpec::Cpu)
        .expect("final target graph");
    let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
    let linear_uop =
        linearized.toposort().into_iter().find(|u| matches!(u.op(), svod_ir::Op::Linear(..))).expect("LINEAR present");
    let svod_ir::Op::Linear(ops::Linear { ops }) = linear_uop.op() else { unreachable!() };
    let verify_root = svod_ir::UOp::sink(ops.iter().cloned().collect());
    svod_schedule::spec::type_verify(&verify_root, &svod_schedule::spec::spec_program())
        .expect("type_verify must pass");
    let code = svod_codegen::traits::Renderer::render(&text_ren, &linear_uop, Some("fa_rdb_w32")).expect("render").code;

    // The illegal vector-of-pointers (`<N x ptr>`) must NEVER appear in the
    // rendered IR — it trips AMD clang with "defined with type '<N x ptr>' but
    // expected 'ptr'". Scan for any `x ptr>` token (covers `<4 x ptr>`,
    // `<8 x ptr>`, etc.).
    let vec_ptr_count = code.matches("x ptr>").count();
    assert_eq!(
        vec_ptr_count, 0,
        "rendered gfx1151 IR must not contain vector-of-pointers; found {vec_ptr_count} occurrences"
    );

    // RDNA WMMA, not CDNA MFMA — confirms the wave32 fragment path is taken.
    let wmma = code.lines().filter(|l| l.contains("wmma.f32.16x16x16") && !l.contains("declare")).count();
    assert!(wmma > 0, "wave32 graph FA must render gfx11 WMMA calls, got {wmma}");
}

/// Render-regression (CPU, no GPU) for the **RDNA3.5 (gfx1151, wave32)** FA path:
/// `build_fa_mw_rdb` with the wave32 caps must render to gfx11 LLVM IR in bounded
/// memory/time, exercising the arch tile-select (`_W32_*` fragments), the
/// accumulator→input LDS relayout (`att → att_mma`), and the even/odd-aware mask.
/// Asserts the RDNA **WMMA** intrinsic is emitted (not the CDNA MFMA), which is the
/// host-side proof the wave32 build path is wired before HW validation on the 395.
#[test]
fn test_fa_mw_rdb_renders_wave32() {
    let (b, h, h_kv, d) = (1usize, 2, 2, 64);
    let n = 128usize;
    let ker = Kernel::new(
        "fa_mw_rdb_w32",
        [h as i64, (n / 16 / 8) as i64, b as i64],
        8 * 32, // NUM_WARPS * wave32
        dummy_fa_buffers(b, n, h, h_kv, d),
        crate::ArchCaps::for_amd(svod_dtype::AmdArch::Gfx1151),
    );
    build_fa_mw_rdb(
        &ker,
        b,
        n,
        h,
        h_kv,
        d,
        FaConfig { q_blk: 16, kv_blk: 16, ..Default::default() },
        svod_dtype::DType::Float16,
        false,
    );
    let sink = ker.finish(1);
    let pm = svod_schedule::symbolic::pm_lower_index_dtype()
        + svod_ir::decompositions::divmod_decomposition_patterns().with_context::<svod_schedule::symbolic::WeakMemo>();
    let lowered = svod_schedule::graph_rewrite(&pm, sink, &mut svod_schedule::symbolic::WeakMemo::default());
    let program = svod_codegen::program_pipeline::program_from_sink(lowered, svod_dtype::DeviceSpec::Cpu)
        .expect("final target graph");
    let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
    let linear_uop =
        linearized.toposort().into_iter().find(|u| matches!(u.op(), svod_ir::Op::Linear(..))).expect("LINEAR present");
    let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(svod_dtype::AmdArch::Gfx1151);
    let code = svod_codegen::traits::Renderer::render(&renderer, &linear_uop, Some("fa_rdb_w32")).expect("render").code;

    // RDNA WMMA, not CDNA MFMA — confirms the wave32 fragment path is taken.
    let wmma = code.lines().filter(|l| l.contains("wmma.f32.16x16x16") && !l.contains("declare")).count();
    assert!(wmma > 0, "wave32 FA must render gfx11 WMMA calls, got {wmma}");
    assert_eq!(code.matches("mfma.f32.16x16x16").count(), 0, "wave32 FA must NOT emit CDNA MFMA");
}

// =============================================================================
// Hardware-gated end-to-end flash-attention on gfx942.
// =============================================================================

/// Per-warp tile config a hardware case runs through the rolled double-buffered
/// builder ([`crate::kernels::fa::build_fa_mw_rdb`]): the Q/KV tile heights and
/// whether to emit the fully-unrolled (flat) compute body.
#[derive(Clone, Copy)]
struct FaPath {
    q_blk: usize,
    kv_blk: usize,
    unroll: bool,
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_mw_rdb_amd -- --ignored --nocapture`.
///
/// Rolled double-buffer FA correctness gate vs causal SDPA, at `{16,16}` and the
/// bigger `{32,32}` per-warp tile. N must be a multiple of `q_blk * NUM_WARPS`.
#[test]
#[ignore]
fn test_fa_mw_rdb_amd() {
    for n in [128usize, 512, 1024, 2048] {
        run_fa_amd_case(1, n, 2, 64, FaPath { q_blk: 16, kv_blk: 16, unroll: false });
    }
    run_fa_amd_case(2, 256, 4, 64, FaPath { q_blk: 16, kv_blk: 16, unroll: false });
    for n in [512usize, 1024, 2048] {
        run_fa_amd_case(1, n, 2, 64, FaPath { q_blk: 32, kv_blk: 32, unroll: false });
    }
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_mw_rdb_unroll_amd -- --ignored --nocapture`.
///
/// P1 in-situ bit-exactness: the **fully-unrolled** rolled-db FA (same structure,
/// flat QKᵀ/softmax/A·V compute) must match causal SDPA — validating `mma_u` /
/// `reduce_u` / the unrolled `map`/`copy`/`transpose` against the looped forms
/// before the cross-tile-pipeline restructure builds on them.
#[test]
#[ignore]
fn test_fa_mw_rdb_unroll_amd() {
    for n in [128usize, 512, 1024, 2048] {
        run_fa_amd_case(1, n, 2, 64, FaPath { q_blk: 16, kv_blk: 16, unroll: true });
    }
    run_fa_amd_case(2, 256, 4, 64, FaPath { q_blk: 16, kv_blk: 16, unroll: true });
    for n in [512usize, 1024, 2048] {
        run_fa_amd_case(1, n, 2, 64, FaPath { q_blk: 32, kv_blk: 32, unroll: true });
    }
}

/// Whether the target device is a CDNA (gfx942) GPU. The direct-launch FA wrappers
/// (`flash_attention_forward*`) hardcode the wave64 block + CDNA fragment tiles, so
/// their HW tests skip on a non-CDNA target (gfx1151 reaches FA only through the
/// wave-width-aware `flash_attention_with` graph entry, exercised by the
/// `*_noncausal_*`/`*_graph_check_*` tests).
fn is_cdna_device() -> bool {
    super::is_cdna_device()
}

fn run_fa_amd_case(b: usize, n: usize, h: usize, d: usize, path: FaPath) {
    use svod_tensor::Tensor;

    if !is_cdna_device() {
        eprintln!("run_fa_amd_case: skipped — gfx942-only direct-launch FA on a non-CDNA device");
        return;
    }

    let mk = || {
        let t = Tensor::randn(&[b, n, h, d]).expect("randn");
        let mut t = t.cast(DType::BFloat16).expect("cast bf16");
        t.realize().expect("realize");
        t
    };
    let (q, k, v) = (mk(), mk(), mk());
    let mut o = Tensor::empty(&[b, n, h, d], DType::BFloat16);

    let h_kv = h;
    let FaPath { q_blk, kv_blk, unroll } = path;
    let grid = [h as i64, (n / q_blk / 8) as i64, b as i64];
    crate::run_kernel("fa_mw_rdb", grid, 8 * 64, &mut [&mut o], &[&q, &k, &v], |ker| {
        crate::kernels::fa::build_fa_mw_rdb(
            ker,
            b,
            n,
            h,
            h_kv,
            d,
            FaConfig { q_blk, kv_blk, unroll, ..Default::default() },
            q.uop().dtype(),
            false,
        );
        ker.finish(1)
    })
    .expect("fa_mw_rdb tiled launch");
    let mut of = o.cast(DType::Float32).expect("o→f32");
    of.realize().expect("realize o→f32");
    let got: Vec<f32> = of.as_vec::<f32>().expect("read o");

    // Reference: permute [B,N,H,D] → [B,H,N,D], SDPA (causal), permute back.
    let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("permute");
    let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
    let ref_bhnd = qp.scaled_dot_product_attention().key(&kp).value(&vp).is_causal(true).call().expect("sdpa");
    let mut reference = ref_bhnd.try_permute(&[0, 2, 1, 3]).expect("permute back");
    reference.realize().expect("realize reference");
    let expected = reference.as_vec::<f32>().expect("read reference");

    assert_eq!(got.len(), expected.len(), "length mismatch");
    let (atol, rtol) = (2e-2f32, 2e-2f32);
    let mut max_abs = 0.0f32;
    let mut worst = 0.0f32;
    for (g, e) in got.iter().zip(&expected) {
        let abs = (g - e).abs();
        max_abs = max_abs.max(abs);
        worst = worst.max(abs - rtol * e.abs());
    }
    let label = format!("mw_rdb[{}{q_blk}x{kv_blk}]", if unroll { "u," } else { "" });
    println!("fa[{label}] B={b} N={n} H={h} D={d}: max abs error = {max_abs:e}");
    assert!(worst <= atol, "FA exceeds atol+rtol*|e| (max abs {max_abs:e}, tol {atol}+{rtol}*|e|)");
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_graph_amd -- --ignored --nocapture`.
///
/// **Phase-A gate (graph integration):** the graph-native `flash_attention` (a
/// `custom_kernel` / `Op::Call` node realized by the tensor scheduler) must match
/// **(a)** causal SDPA (the oracle, tol 2e-2) AND **(b)** the direct-launch
/// `flash_attention_forward_mw_rdb` output **bit-for-bit** (same kernel, different
/// launch path). Proves the full FA SINK — LDS + barriers + 512-thread block +
/// dynamic `range_uop` bound + ds_bpermute + WMMA — survives `custom_kernel →
/// rangeify → realize` on gfx942.
#[test]
#[ignore]
fn test_fa_graph_amd() {
    use svod_tensor::Tensor;
    // Compares the graph kernel against the gfx942-only direct `flash_attention_forward_mw_rdb`,
    // so it skips on a non-CDNA target (the graph path itself is covered on gfx1151 by
    // `test_fa_graph_check_amd` / `test_fa_noncausal_f16*` against SDPA).
    if !is_cdna_device() {
        eprintln!("test_fa_graph_amd: skipped — graph-vs-direct compare uses the gfx942-only direct launcher");
        return;
    }
    for (b, n, h, d) in [(1usize, 128usize, 2usize, 64usize), (1, 512, 2, 64), (2, 256, 4, 64)] {
        let mk = || {
            let t = Tensor::randn(&[b, n, h, d]).expect("randn");
            let mut t = t.cast(DType::BFloat16).expect("cast bf16");
            t.realize().expect("realize");
            t
        };
        let (q, k, v) = (mk(), mk(), mk());

        // Graph path: lazy custom_kernel Tensor → realize.
        let og = crate::kernels::fa::flash_attention(&q, &k, &v).expect("graph fa").expect("FA kernel applies");
        let mut og_f = og.cast(DType::Float32).expect("og→f32");
        og_f.realize().expect("realize graph");
        let graph: Vec<f32> = og_f.as_vec::<f32>().expect("read graph");

        // Direct-launch path (same kernel, in-place dispatch).
        let mut od = Tensor::empty(&[b, n, h, d], DType::BFloat16);
        crate::kernels::fa::flash_attention_forward_mw_rdb(&mut od, &q, &k, &v).expect("direct fa");
        let mut od_f = od.cast(DType::Float32).expect("od→f32");
        od_f.realize().expect("realize direct");
        let direct: Vec<f32> = od_f.as_vec::<f32>().expect("read direct");

        // Reference: causal SDPA over the same operands.
        let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("permute");
        let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
        let ref_bhnd = qp.scaled_dot_product_attention().key(&kp).value(&vp).is_causal(true).call().expect("sdpa");
        let mut reference = ref_bhnd.try_permute(&[0, 2, 1, 3]).expect("permute back");
        reference.realize().expect("realize reference");
        let expected = reference.as_vec::<f32>().expect("read reference");

        assert_eq!(graph.len(), expected.len(), "graph/ref length mismatch");
        assert_eq!(graph.len(), direct.len(), "graph/direct length mismatch");
        let (atol, rtol) = (2e-2f32, 2e-2f32);
        let (mut max_abs_ref, mut worst, mut max_abs_direct) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..graph.len() {
            let (g, e, dd) = (graph[i], expected[i], direct[i]);
            max_abs_ref = max_abs_ref.max((g - e).abs());
            worst = worst.max((g - e).abs() - rtol * e.abs());
            max_abs_direct = max_abs_direct.max((g - dd).abs());
        }
        println!("fa[graph] B={b} N={n} H={h} D={d}: vs SDPA = {max_abs_ref:e}, vs direct = {max_abs_direct:e}");
        assert!(worst <= atol, "graph FA vs SDPA exceeds tol (max abs {max_abs_ref:e})");
        assert!(max_abs_direct < 1e-3, "graph FA must match direct-launch bit-for-bit (Δ {max_abs_direct:e})");
    }
}

/// The f32 causal SDPA reference for the graph check below, in `[B,N,H,D]`.
#[allow(clippy::result_large_err)] // one-shot check helper, like the macro body
fn fa_causal_reference(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, svod_tensor::error::Error> {
    let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("permute");
    let (qp, kp, vp) = (perm(q), perm(k), perm(v));
    let r = qp.scaled_dot_product_attention().key(&kp).value(&vp).is_causal(true).call().expect("sdpa");
    r.try_permute(&[0, 2, 1, 3])
}

// Generic custom-kernel **check** via the `tensor`-layer macro — the graph-native
// `flash_attention` vs causal SDPA on gfx942/gfx1151. Demonstrates the reusable
// definition(`Tensor::graph_kernel`)/check(`custom_kernel_check!`) facility: tk just
// supplies the kernel `run` + the `reference` op; the boilerplate (build inputs, run
// both, cast f32, compare within tol) is generated. On a device the FA kernel
// declines (`Ok(None)`: CUDA until the FA port), `run` self-skips onto the reference.
// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_graph_check_amd -- --ignored --nocapture`.
svod_tensor::custom_kernel_check! {
    test_fa_graph_check_amd,
    inputs (q, k, v): shape [1, 128, 2, 64], dtype svod_dtype::DType::BFloat16,
    run: |q, k, v| match crate::kernels::fa::flash_attention(q, k, v).expect("FA build") {
        Some(o) => Ok(o),
        None => {
            eprintln!("skip test_fa_graph_check_amd: the FA kernel does not apply on this device");
            fa_causal_reference(q, k, v)
        }
    },
    reference: fa_causal_reference,
    tol: 2e-2,
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_noncausal_f16_amd -- --ignored --nocapture`.
///
/// The unified [`crate::kernels::fa::flash_attention_with`] **non-causal, f16,
/// unmasked** path vs full-bidirectional SDPA over the same f16 operands. Exercises
/// the `causal: false` KV sweep and the f16 WMMA-operand globals end-to-end on
/// gfx942. Tol 2e-2 (f16 accumulation slack).
#[test]
#[ignore]
fn test_fa_noncausal_f16_amd() {
    if !super::device_supported(crate::kernels::fa::FA_SUPPORTED_ARCHS) {
        eprintln!("skip test_fa_noncausal_f16_amd: unsupported device/toolchain");
        return;
    }
    use crate::kernels::fa::{FaOpts, flash_attention_with};
    use svod_tensor::Tensor;

    let (b, n, h, d) = (1usize, 256usize, 8usize, 64usize);
    let mk = || {
        let mut t = Tensor::randn(&[b, n, h, d]).expect("randn").cast(DType::Float16).expect("cast f16");
        t.realize().expect("realize");
        t
    };
    let (q, k, v) = (mk(), mk(), mk());

    let og = flash_attention_with(&q, &k, &v, FaOpts { causal: false, key_lens: None })
        .expect("fa noncausal")
        .expect("FA kernel applies");
    let mut og_f = og.cast(DType::Float32).expect("og→f32");
    og_f.realize().expect("realize og");
    let got: Vec<f32> = og_f.as_vec::<f32>().expect("read og");

    // Reference: full (non-causal) SDPA over the same f32-permuted operands.
    let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("permute");
    let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
    let ref_bhnd = qp.scaled_dot_product_attention().key(&kp).value(&vp).is_causal(false).call().expect("sdpa");
    let mut reference = ref_bhnd.try_permute(&[0, 2, 1, 3]).expect("permute back");
    reference.realize().expect("realize reference");
    let expected = reference.as_vec::<f32>().expect("read reference");

    assert_eq!(got.len(), expected.len(), "length mismatch");
    let max_abs = got.iter().zip(&expected).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max);
    println!("fa[noncausal,f16] B={b} N={n} H={h} D={d}: max abs error = {max_abs:e}");
    assert!(max_abs <= 2e-2, "non-causal f16 FA exceeds tol (max abs {max_abs:e})");
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_noncausal_f16_masked_amd -- --ignored --nocapture`.
///
/// The unified [`crate::kernels::fa::flash_attention_with`] **non-causal, f16,
/// key-masked** path vs full SDPA with the SAME `[B,1,1,N]` key mask. `key_lens =
/// [200]` (valid keys) ⇒ keys `200..256` are masked. Both the kernel and the
/// reference use KEY-only masking, so they agree on EVERY row (the full output is
/// compared, including the padded query rows `200..256`). Tol 2e-2.
#[test]
#[ignore]
fn test_fa_noncausal_f16_masked_amd() {
    if !super::device_supported(crate::kernels::fa::FA_SUPPORTED_ARCHS) {
        eprintln!("skip test_fa_noncausal_f16_masked_amd: unsupported device/toolchain");
        return;
    }
    use crate::kernels::fa::{FaOpts, flash_attention_with};
    use svod_tensor::Tensor;

    let (b, n, h, d) = (1usize, 256usize, 8usize, 64usize);
    let valid: i32 = 200;
    let mk = || {
        let mut t = Tensor::randn(&[b, n, h, d]).expect("randn").cast(DType::Float16).expect("cast f16");
        t.realize().expect("realize");
        t
    };
    let (q, k, v) = (mk(), mk(), mk());

    // Per-batch valid-key-count tensor [B] i32 (on the default/AMD device).
    let mut lens = Tensor::from_slice([valid; 1]);
    lens.realize().expect("realize lens");

    let og = flash_attention_with(&q, &k, &v, FaOpts { causal: false, key_lens: Some(&lens) })
        .expect("fa masked")
        .expect("FA kernel applies");
    let mut og_f = og.cast(DType::Float32).expect("og→f32");
    og_f.realize().expect("realize og");
    let got: Vec<f32> = og_f.as_vec::<f32>().expect("read og");

    // Reference: full SDPA with the same [B,1,1,N] bool key mask (true = masked,
    // where arange(N) >= valid), mirroring the kernel's `kv_pos >= lens[batch]`.
    let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("permute");
    let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
    let range = Tensor::arange(n as i64, None, None).expect("arange").try_reshape([1usize, 1, 1, n]).expect("reshape");
    let lref = Tensor::from_slice([valid; 1]).try_reshape([b, 1, 1, 1]).expect("reshape lens");
    let mask = range.try_ge(&lref).expect("ge mask");
    let ref_bhnd = qp
        .scaled_dot_product_attention()
        .key(&kp)
        .value(&vp)
        .is_causal(false)
        .attn_mask(&mask)
        .call()
        .expect("sdpa masked");
    let mut reference = ref_bhnd.try_permute(&[0, 2, 1, 3]).expect("permute back");
    reference.realize().expect("realize reference");
    let expected = reference.as_vec::<f32>().expect("read reference");

    assert_eq!(got.len(), expected.len(), "length mismatch");
    let max_abs = got.iter().zip(&expected).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max);
    println!("fa[noncausal,f16,masked lens={valid}] B={b} N={n} H={h} D={d}: max abs error = {max_abs:e}");
    assert!(max_abs <= 2e-2, "non-causal masked f16 FA exceeds tol (max abs {max_abs:e})");
}

/// A fully key-masked lane (`key_lens[b] == 0`) — the inactive-lane case the
/// GigaAM JIT produces when a chunk-batch's tail lanes pad to length 0 — must
/// produce a FINITE row, not `NaN`. With no valid key the online-softmax running
/// max stays −inf and the rescale's `−inf − (−inf)` is NaN; `flash_attention_with`
/// floors `key_lens` to ≥ 1 so every row attends to at least key 0, keeping the
/// (caller-discarded) inactive lane finite.
///
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_key_lens_zero_is_finite_amd -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_fa_key_lens_zero_is_finite_amd() {
    if !super::device_supported(crate::kernels::fa::FA_SUPPORTED_ARCHS) {
        eprintln!("skip test_fa_key_lens_zero_is_finite_amd: unsupported device/toolchain");
        return;
    }
    use crate::kernels::fa::{FaOpts, flash_attention_with};
    use svod_tensor::Tensor;

    let (b, n, h, d) = (1usize, 256usize, 8usize, 64usize);
    let mk = || {
        let mut t = Tensor::randn(&[b, n, h, d]).expect("randn").cast(DType::Float16).expect("cast f16");
        t.realize().expect("realize");
        t
    };
    let (q, k, v) = (mk(), mk(), mk());

    // Zero valid keys ⇒ every key position would be masked for every query row.
    let mut lens = Tensor::from_slice([0i32; 1]);
    lens.realize().expect("realize lens");

    let og = flash_attention_with(&q, &k, &v, FaOpts { causal: false, key_lens: Some(&lens) })
        .expect("fa masked")
        .expect("FA kernel applies");
    let mut og_f = og.cast(DType::Float32).expect("og→f32");
    og_f.realize().expect("realize og");
    let got: Vec<f32> = og_f.as_vec::<f32>().expect("read og");

    assert!(got.iter().all(|x| x.is_finite()), "key_lens==0 lane must be finite (key_lens clamp to >=1 missing?)");
}

// =============================================================================
// CUDA sm_80+ (`mma.sync`, warp32).
// =============================================================================

/// Whether the env-selected device is CUDA sm_80+ with the NVPTX toolchain.
fn cuda_device() -> bool {
    super::fragment_device().is_some_and(|caps| caps.cuda().is_some())
}

/// `SVOD_DEVICE=CUDA:0 cargo test -p svod-tk --lib fa::test_fa_tile_bench_cuda -- --ignored --nocapture`.
///
/// Per-dispatch GPU time of the non-causal kernel over the per-warp tile ×
/// body-form grid at the whisper (`b·h = 6..48`) and GigaAM (`b·h = 128`)
/// encoder geometries — the measurement behind the CUDA [`crate::kernels::fa::FaPolicy`].
#[test]
#[ignore]
fn test_fa_tile_bench_cuda() {
    use crate::kernels::fa::NUM_WARPS;
    if !cuda_device() {
        eprintln!("skip test_fa_tile_bench_cuda: no CUDA sm_80+ device / toolchain");
        return;
    }
    let ws = super::fragment_device().expect("caps").wave_size;
    let geometries = [(1usize, 1536usize, 6usize, 64usize), (8, 1536, 6, 64), (8, 1536, 16, 64), (1, 1024, 16, 64)];
    let tiles = [(16usize, 16usize), (16, 32), (32, 32), (16, 64)];
    for (b, n, h, d) in geometries {
        let mk = || {
            let mut t = Tensor::randn(&[b, n, h, d]).expect("randn").cast(DType::Float16).expect("cast");
            t.realize().expect("realize");
            t
        };
        let (q, k, v) = (mk(), mk(), mk());
        for (q_blk, kv_blk) in tiles {
            for unroll in [false, true] {
                if !n.is_multiple_of(q_blk * NUM_WARPS) {
                    continue;
                }
                let mut o = Tensor::empty(&[b, n, h, d], DType::Float16);
                let grid = [h as i64, (n / q_blk / NUM_WARPS) as i64, b as i64];
                let cfg = FaConfig { q_blk, kv_blk, unroll, causal: false };
                let compiled = crate::compile_kernel(
                    "fa_bench",
                    grid,
                    (NUM_WARPS * ws) as i64,
                    &mut [&mut o],
                    &[&q, &k, &v],
                    |ker| {
                        build_fa_mw_rdb(ker, b, n, h, h, d, cfg, DType::Float16, false);
                        ker.finish(1)
                    },
                )
                .expect("compile");
                let mut us: Vec<f64> = (0..12)
                    .map(|_| compiled.dispatch_gpu_ns().expect("dispatch").expect("stamped") as f64 / 1e3)
                    .collect();
                us.sort_by(f64::total_cmp);
                println!(
                    "fa[cuda] b={b} n={n} h={h} tile={q_blk}x{kv_blk} unroll={unroll}: median {:.1} µs (min {:.1})",
                    us[us.len() / 2],
                    us[0]
                );
            }
        }
    }
}

/// Render `build_fa_mw_rdb` for sm_86 through the launch path's pipeline
/// (post-optimization with the CUDA profile → linearize → render); with
/// `TK_DUMP_IR=dir` the IR is written to `dir/<name>.ll` for offline `ptxas -v`.
fn render_fa_sm86(name: &str, (b, n, h, h_kv, d): (usize, usize, usize, usize, usize), cfg: FaConfig) -> String {
    use crate::kernels::fa::NUM_WARPS;
    let sm86 = svod_dtype::CudaArch::from_compute_capability(8, 6);
    let caps = crate::ArchCaps::for_arch(svod_dtype::GpuArch::Cuda(sm86));
    let grid = [h as i64, (n / cfg.q_blk / NUM_WARPS) as i64, b as i64];
    let ker = Kernel::new(name, grid, (NUM_WARPS * caps.wave_size) as i64, dummy_fa_buffers(b, n, h, h_kv, d), caps);
    build_fa_mw_rdb(&ker, b, n, h, h_kv, d, cfg, DType::BFloat16, false);
    let sink = ker.finish(1);
    let renderer = svod_codegen::llvm::LlvmTextRenderer::nvptx(sm86);
    let opt_renderer = svod_schedule::OptimizerRenderer::for_cuda_arch(sm86).with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        svod_codegen::traits::Renderer::decompositor(&renderer),
        None,
    );
    let optimized =
        svod_schedule::apply_post_optimization_with_renderer(sink, &opt_renderer).expect("post optimization");
    let program =
        svod_codegen::program_pipeline::program_from_sink(optimized, DeviceSpec::Cpu).expect("final target graph");
    let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
    let linear_uop =
        linearized.toposort().into_iter().find(|u| matches!(u.op(), svod_ir::Op::Linear(..))).expect("LINEAR present");
    let code = svod_codegen::traits::Renderer::render(&renderer, &linear_uop, Some(name)).expect("render").code;
    if let Ok(dir) = std::env::var("TK_DUMP_IR") {
        std::fs::create_dir_all(&dir).expect("dump dir");
        std::fs::write(std::path::Path::new(&dir).join(format!("{name}.ll")), &code).expect("dump IR");
    }
    code
}

/// sm_86 FA renders to `mma.sync` (host, no GPU) for every per-warp tile and body
/// form the CUDA [`crate::kernels::fa::FaPolicy`] can select: the two-half
/// fragments lower to `llvm.nvvm.mma.m16n8k16` (bf16, f32 accumulate), the K/V
/// double buffers are `addrspace(3)` shared arrays, the cross-lane reduce is the
/// quad `shfl.bfly`, and no AMD intrinsic leaks through. `TK_DUMP_IR=dir` keeps the
/// IR for `ptxas -v`.
#[test_case::test_case(16, 32, false, true, 64; "16x32 rolled causal")]
#[test_case::test_case(16, 32, true, false, 64; "16x32 flat")]
#[test_case::test_case(32, 32, false, false, 64; "32x32 rolled")]
#[test_case::test_case(32, 32, true, false, 64; "32x32 flat")]
#[test_case::test_case(32, 32, true, true, 64; "32x32 flat causal")]
#[test_case::test_case(32, 32, true, false, 128; "32x32 flat d=128")]
#[test_case::test_case(16, 32, true, false, 128; "16x32 flat d=128")]
#[test_case::test_case(16, 64, true, false, 64; "16x64 flat")]
#[test_case::test_case(16, 64, true, true, 128; "16x64 flat causal d=128")]
fn test_fa_sm86_renders_mma_sync(q_blk: usize, kv_blk: usize, unroll: bool, causal: bool, d: usize) {
    let body = if unroll { "flat" } else { "rolled" };
    let name = format!("fa_sm86_{q_blk}x{kv_blk}_{body}{}_d{d}", if causal { "_causal" } else { "" });
    let code = render_fa_sm86(&name, (1, 512, 2, 2, d), FaConfig { q_blk, kv_blk, unroll, causal });
    let mma =
        code.lines().filter(|l| l.contains("llvm.nvvm.mma.m16n8k16.row.col.bf16") && !l.contains("declare")).count();
    let per_slice = 2 * (kv_blk / 16) * (d / 16) * 2; // QKᵀ + A·V halves per fragment product
    if unroll {
        assert_eq!(mma, per_slice * (q_blk / 16), "flat body: every m16n8k16 half is a distinct call site");
    } else {
        assert!(mma > 0 && mma < per_slice, "rolled body keeps the fragment loops ({mma} sites)");
    }
    assert!(code.contains("addrspace(3)"), "the K/V double buffers are NVPTX shared arrays");
    assert!(code.contains("shfl.sync.bfly"), "the softmax reduce completes in the quad butterfly");
    // The K/V gathers are one `ldmatrix.x4` per 16×16 fragment — plain for K (a
    // `Row` tile read `Row`), `.trans` for V (read `Col`) — and no scalar shared
    // load remains.
    let calls = |needle: &str| code.lines().filter(|l| l.contains(needle) && !l.contains("declare")).count();
    let frags = (kv_blk / 16) * (d / 16);
    assert_eq!(calls("call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16("), frags, "K");
    assert_eq!(calls("call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16("), frags, "V");
    // The K/V stream is `cp.async`: one 16-byte copy per lane per pass of each tile
    // in the prologue and in the loop, one commit per tile, one `wait_group 0` at the
    // loop top and one `wait_all` drain after it; no `st.shared` commit remains.
    let passes = (kv_blk * d * 2).div_ceil(256 * 16);
    assert_eq!(calls("@llvm.nvvm.cp.async.cg.shared.global.16("), 4 * passes, "K+V copies, prologue + loop");
    assert_eq!(calls("@llvm.nvvm.cp.async.commit.group()"), 4);
    assert_eq!(calls("@llvm.nvvm.cp.async.wait.group(i32 0)"), 1);
    assert_eq!(calls("@llvm.nvvm.cp.async.wait.all()"), 1);
    // In the PTX: no scalar shared load survives (the gathers are `ldmatrix`), the
    // K/V commit is `cp.async` (no `st.shared`), and the loop carries one barrier.
    if let Some(ptx) = ptx_of(&code) {
        let count = |needle: &str| ptx.matches(needle).count();
        assert_eq!(count("ld.shared.b16"), 0, "scalar LDS gather in the PTX:\n{ptx}");
        assert_eq!(count("st.shared"), 0, "register-staged LDS commit in the PTX:\n{ptx}");
        assert_eq!(count("ldmatrix.sync.aligned.m8n8.x4.shared.b16"), frags);
        assert_eq!(count("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"), frags);
        assert_eq!(count("cp.async.cg.shared.global"), 4 * passes);
        assert_eq!((count("cp.async.wait_group"), count("cp.async.wait_all"), count("bar.sync")), (1, 1, 1));
    }
    assert!(
        !code.contains("amdgcn") && !code.contains("mfma") && !code.contains("wmma."),
        "no AMD intrinsics on NVPTX"
    );
}

/// The sm_86 PTX of rendered NVPTX IR, or `None` without an NVPTX-enabled clang
/// (the render assertions still run; only the PTX-level ones skip).
pub(super) fn ptx_of(code: &str) -> Option<String> {
    if !svod_runtime::cuda::has_nvptx_target() {
        return None;
    }
    let sm86 = svod_dtype::CudaArch::from_compute_capability(8, 6);
    Some(String::from_utf8(svod_runtime::cuda::compile_ir_to_ptx(code, sm86).expect("clang accepts the IR")).unwrap())
}

const GFX942: svod_dtype::GpuArch = svod_dtype::GpuArch::Amd(svod_dtype::AmdArch::Gfx942);
const GFX1151: svod_dtype::GpuArch = svod_dtype::GpuArch::Amd(svod_dtype::AmdArch::Gfx1151);
const SM_86: svod_dtype::GpuArch = svod_dtype::GpuArch::Cuda(svod_dtype::CudaArch::from_compute_capability(8, 6));

/// The per-arch tile policy: gfx942's `{32,32}` needs `b·h·n/256 >= 304` blocks
/// (else the `{16,32}` baseline), gfx1151 always takes the baseline, and CUDA
/// takes the taller `{16,64}` KV block (flat) once the grid covers its 28 SMs, at
/// d ≤ 64 (the d=128 double buffers would exceed the static LDS).
#[test_case::test_case(GFX942, (1, 1536, 16, 64), (16, 32), false; "gfx942 small grid")]
#[test_case::test_case(GFX942, (8, 2048, 32, 128), (32, 32), false; "gfx942 machine-covering grid")]
#[test_case::test_case(GFX942, (64, 1152, 16, 64), (16, 32), false; "gfx942 N not a 256-multiple")]
#[test_case::test_case(GFX1151, (64, 2048, 32, 64), (16, 32), false; "gfx1151 baseline only")]
#[test_case::test_case(SM_86, (1, 1536, 6, 64), (16, 64), true; "sm_86 whisper-tiny b=1")]
#[test_case::test_case(SM_86, (8, 1536, 16, 64), (16, 64), true; "sm_86 gigaam")]
#[test_case::test_case(SM_86, (1, 1024, 16, 64), (16, 64), true; "sm_86 gigaam b=1")]
#[test_case::test_case(SM_86, (1, 256, 2, 64), (16, 32), true; "sm_86 tiny grid")]
#[test_case::test_case(SM_86, (8, 1152, 16, 64), (16, 64), true; "sm_86 N a 128-multiple only")]
#[test_case::test_case(SM_86, (8, 1536, 16, 128), (16, 32), true; "sm_86 d=128 keeps the small tile")]
fn fa_policy_tile(
    arch: svod_dtype::GpuArch,
    (b, n, h, d): (usize, usize, usize, usize),
    tile: (usize, usize),
    unroll: bool,
) {
    let policy = crate::kernels::fa::FaPolicy::for_arch(arch);
    assert_eq!(policy.tile(b, n, h, d), tile);
    let cfg = policy.config(b, n, h, d, false);
    assert_eq!((cfg.q_blk, cfg.kv_blk, cfg.unroll, cfg.causal), (tile.0, tile.1, unroll, false));
}

/// The f32 SDPA reference of `flash_attention_with(q, k, v, opts)` in `[B,N,H,D]`:
/// the GQA `h_kv` groups broadcast to `h`, `causal` and the `[B,1,1,N]` key mask
/// (`kv_pos >= key_lens[b]`) applied exactly as the kernel does.
fn fa_reference(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool, key_lens: Option<&[i32]>) -> Tensor {
    let dims = |t: &Tensor| t.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>();
    let (b, n, h, d) = (dims(q)[0], dims(q)[1], dims(q)[2], dims(q)[3]);
    let h_kv = dims(k)[2];
    let perm = |t: &Tensor| {
        let t = t.cast(DType::Float32).unwrap();
        let t = if dims(&t)[2] == h {
            t
        } else {
            t.try_reshape([b, n, h_kv, 1, d])
                .unwrap()
                .try_expand([b, n, h_kv, h / h_kv, d])
                .unwrap()
                .try_reshape([b, n, h, d])
                .unwrap()
        };
        t.try_permute(&[0, 2, 1, 3]).unwrap()
    };
    let mask = key_lens.map(|lens| {
        let range = Tensor::arange(n as i64, None, None).unwrap().try_reshape([1usize, 1, 1, n]).unwrap();
        range.try_ge(&Tensor::from_slice(lens).try_reshape([b, 1, 1, 1]).unwrap()).unwrap()
    });
    perm(q)
        .scaled_dot_product_attention()
        .key(&perm(k))
        .value(&perm(v))
        .is_causal(causal)
        .maybe_attn_mask(mask.as_ref())
        .call()
        .unwrap()
        .try_permute(&[0, 2, 1, 3])
        .unwrap()
}

/// `SVOD_DEVICE=CUDA:0 cargo test -p svod-tk --lib fa::test_fa_cuda -- --ignored --nocapture`.
///
/// The production [`flash_attention_with`] on CUDA sm_80+ vs SDPA at the model
/// geometries: whisper (d=64, T=1500 padded to 1536 with a 1500-key mask, h=6/8,
/// b=1..8), GigaAM (d=64, T=1536, h=16, b=8, key-padding mask), causal, GQA
/// (`h_kv < h`), d=128, bf16 and f16, both per-warp tiles of the policy. The
/// kernel accumulates in f32 over the same 16-bit operands the reference reads.
#[test_case::test_case(1, 1536, 6, 6, 64, DType::Float16, false, Some(1500); "whisper tiny b=1 f16")]
#[test_case::test_case(8, 1536, 8, 8, 64, DType::Float16, false, Some(1500); "whisper base b=8 f16")]
#[test_case::test_case(3, 1536, 8, 8, 64, DType::BFloat16, false, None; "whisper b=3 bf16 unmasked")]
#[test_case::test_case(8, 1536, 16, 16, 64, DType::Float16, false, Some(1000); "gigaam b=8 f16 padded")]
#[test_case::test_case(2, 1024, 16, 16, 64, DType::BFloat16, false, Some(777); "gigaam-like bf16 padded")]
#[test_case::test_case(1, 512, 4, 4, 64, DType::BFloat16, true, None; "causal bf16 big tile")]
#[test_case::test_case(1, 128, 2, 2, 64, DType::Float16, true, None; "causal f16 small tile")]
#[test_case::test_case(2, 256, 8, 2, 64, DType::BFloat16, false, None; "gqa bf16 small tile")]
#[test_case::test_case(2, 512, 8, 2, 128, DType::Float16, true, None; "gqa d=128 causal f16")]
#[test_case::test_case(1, 256, 8, 8, 64, DType::Float16, false, Some(200); "masked f16 small tile")]
#[ignore]
#[allow(clippy::too_many_arguments)]
fn test_fa_cuda(b: usize, n: usize, h: usize, h_kv: usize, d: usize, dtype: DType, causal: bool, valid: Option<i32>) {
    if !cuda_device() {
        eprintln!("skip test_fa_cuda: no CUDA sm_80+ device / toolchain");
        return;
    }
    let mk = |h: usize| {
        let mut t = Tensor::randn(&[b, n, h, d]).expect("randn").cast(dtype.clone()).expect("cast");
        t.realize().expect("realize");
        t
    };
    let (q, k, v) = (mk(h), mk(h_kv), mk(h_kv));
    let lens: Option<Vec<i32>> = valid.map(|l| vec![l; b]);
    let lens_t = lens.as_ref().map(|l| Tensor::from_slice(l.as_slice()));
    let mut got = flash_attention_with(&q, &k, &v, FaOpts { causal, key_lens: lens_t.as_ref() })
        .expect("fa")
        .expect("the FA kernel applies on CUDA sm_80+")
        .cast(DType::Float32)
        .expect("→f32");
    got.realize().expect("realize");
    let mut reference = fa_reference(&q, &k, &v, causal, lens.as_deref());
    reference.realize().expect("realize reference");
    let report = svod_tensor::testing::allclose_f32(
        &got.as_vec::<f32>().expect("read"),
        &reference.as_vec::<f32>().expect("read reference"),
        2e-2,
        2e-2,
    );
    println!("fa[cuda] b={b} n={n} h={h}/{h_kv} d={d} {dtype:?} causal={causal} lens={valid:?}: {}", report.message);
    assert!(report.ok, "{}", report.message);
}

/// Every RANGE a kernel body opens must be closed on every path to its SINK: the
/// tensor scheduler's kernel split (`split_store`) treats a RANGE still in scope
/// at a CALL as an interior loop and refuses to cut the *consumer* kernel, which
/// then fails the kernel-graph spec. `After.ended_ranges()` only propagates through
/// `END`/`After` deps — an ordering statement (a `cp.async` wait, a barrier) as an
/// `After` dep hides the loop `END` from a carried accumulator's post-loop read, so
/// the CUDA stream threads its drain through the output GLOBAL instead. Pins both
/// K/V streams, flat and rolled.
#[test_case::test_case(SM_86, true; "sm_86 flat")]
#[test_case::test_case(SM_86, false; "sm_86 rolled")]
#[test_case::test_case(GFX942, false; "gfx942 rolled")]
fn fa_sink_leaves_no_range_in_scope(arch: svod_dtype::GpuArch, unroll: bool) {
    use crate::kernels::fa::NUM_WARPS;
    let (b, n, h, h_kv, d) = (1usize, 128usize, 2usize, 2usize, 64usize);
    let cfg = FaConfig { q_blk: 16, kv_blk: 32, unroll, causal: true };
    let caps = crate::ArchCaps::for_arch(arch);
    let grid = [h as i64, (n / cfg.q_blk / NUM_WARPS) as i64, b as i64];
    let ker = Kernel::new("fa", grid, (NUM_WARPS * caps.wave_size) as i64, dummy_fa_buffers(b, n, h, h_kv, d), caps);
    build_fa_mw_rdb(&ker, b, n, h, h_kv, d, cfg, DType::BFloat16, false);
    let sink = ker.finish(1);
    assert!(sink.in_scope_ranges().is_empty(), "RANGE {:?} still open at the SINK", sink.in_scope_ranges());
}
