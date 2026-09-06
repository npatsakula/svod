//! Tests for the bf16→f32 tile matmul ([`crate::kernels::matmul`]): a port of
//! tinygrad `test_tk.py::test_simple_matmul` plus a GPU-free graph-shape check of
//! the `mma_AB` WMMA construction and the hardware-gated end-to-end checks.

use std::sync::Arc;

use svod_dtype::{CudaArch, DType, DeviceSpec, GpuArch};
use svod_ir::{Op, UOp};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::kernels::matmul::*;
use crate::tiles::{RT_16X16, RT_16X16_MMA, TileLayout};
use crate::{Kernel, MoveIdx};
use svod_ir::ops;

const SM_86: GpuArch = GpuArch::Cuda(CudaArch::from_compute_capability(8, 6));

/// Dummy `(c, a, b)` BUFFER UOps for GPU-free graph-shape kernel builds.
fn dummy_buffers(n: usize) -> Vec<Arc<UOp>> {
    let sz = n * n;
    vec![
        UOp::new_buffer(DeviceSpec::Cpu, sz, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, sz, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, sz, DType::BFloat16),
    ]
}

/// A non-rank-2 operand is a structured `Err` (not a panic). The shape
/// preconditions resolve before any device dispatch, so this runs GPU-free.
#[test]
fn matmul_non_rank2_operand_is_operand_shape_err() {
    let sq = Tensor::randn(&[64, 64]).expect("randn");
    let a1 = Tensor::randn(&[64]).expect("randn"); // operand a: rank 1
    let e = matmul(&a1, &sq).err().expect("rank-1 a must error, not panic");
    assert!(matches!(e, crate::launch::Error::OperandRank { operand: "a", .. }), "got {e:?}");
    let b3 = Tensor::randn(&[2, 64, 64]).expect("randn"); // operand b: rank 3
    let e = matmul(&sq, &b3).err().expect("rank-3 b must error, not panic");
    assert!(matches!(e, crate::launch::Error::OperandRank { operand: "b", .. }), "got {e:?}");
}

/// Pure graph-shape check (no GPU): `mma_AB` emits exactly one `WMMA` per
/// K-iteration with `bf16.vec(4)` × `bf16.vec(4)` → `f32.vec(4)` operands and a
/// 16×16×16 / 4-4-4 descriptor.
#[test]
fn test_mma_ab_wmma_graph_shape() {
    let ker = Kernel::new("mma_probe", [1, 1, 1], 64, vec![], crate::ArchCaps::GFX942);
    let warp = ker.warp();

    let a = ker.rt((64, 64), DType::BFloat16, TileLayout::Row, RT_16X16);
    let b = ker.rt((64, 64), DType::BFloat16, TileLayout::Col, RT_16X16);
    let c = ker.rt((64, 64), DType::Float32, TileLayout::Col, RT_16X16);

    let c0 = warp.zero(c);
    let out = warp.mma_ab(c0, &a, &b);

    let wmmas: Vec<_> = out.uop().toposort().into_iter().filter(|u| matches!(u.op(), Op::Wmma(..))).collect();
    assert_eq!(wmmas.len(), 1, "exactly one symbolic WMMA per K-iteration");

    let Op::Wmma(ops::Wmma { a: wa, b: wb, c: wc, metadata }) = wmmas[0].op() else { unreachable!() };
    assert_eq!(wa.dtype(), DType::BFloat16, "A operand keeps its scalar dtype");
    assert_eq!(wb.dtype(), DType::BFloat16, "B operand keeps its scalar dtype");
    assert_eq!(wc.dtype(), DType::Float32, "C operand keeps its scalar dtype");
    assert_eq!(wmmas[0].dtype(), DType::Float32, "WMMA dtype follows C");
    for operand in [wa, wb, wc, &wmmas[0]] {
        assert_eq!(operand.shape().unwrap().unwrap()[0].as_const(), Some(4));
    }

    assert_eq!(metadata.dims, (16, 16, 16));
    assert_eq!(metadata.dtype_in, DType::BFloat16);
    assert_eq!(metadata.dtype_out, DType::Float32);
    let prod = |axes: &[(svod_ir::AxisId, usize)]| axes.iter().map(|(_, s)| s).product::<usize>();
    let axes = metadata.upcast_axes.as_ref().expect("unexpanded WMMA metadata");
    assert_eq!(prod(&axes.a), 4, "A upcast product");
    assert_eq!(prod(&axes.b), 4, "B upcast product");
    assert_eq!(prod(&axes.c), 4, "C upcast product");
}

/// The fully-unrolled MMA ([`Kernel::set_unroll`]) emits one symbolic `WMMA` per
/// `(height, width, k)` fragment — a 32×32 = 2×2 output over a 32-wide K (2
/// reduce steps) is 8 flat nodes — vs the looped form's single symbolic node, and
/// renders to gfx942 with 8 distinct `mfma` instructions (no enclosing
/// `loop_body`), which the looped form cannot (it renders one mfma inside loops).
/// This is the P1 flatness de-risk: explicit Rust-`for` unroll *does* flatten the
/// MFMAs on tk's optimizer-skipping direct-launch path (route b).
#[test]
fn test_mma_unroll_flattens_mfma() {
    let build = |unroll: bool| {
        let n = 32usize;
        let ker = Kernel::new("mma_unroll_probe", [1, 1, 1], 64, dummy_buffers(n), crate::ArchCaps::GFX942);
        ker.set_unroll(unroll);
        let c_gl = ker.gl(&[1, 1, n, n], DType::Float32);
        let _a_gl = ker.gl(&[1, 1, n, n], DType::BFloat16);
        let _b_gl = ker.gl(&[1, 1, n, n], DType::BFloat16);
        let warp = ker.warp();
        let a = warp.zero(ker.rt((n, n), DType::BFloat16, TileLayout::Row, RT_16X16));
        // `mma_ab` reads `a[h,k] b[k,w]`; a 32×32 col `b` is a 2×2 K-tiled operand.
        let b = warp.zero(ker.rt((n, n), DType::BFloat16, TileLayout::Col, RT_16X16));
        let c = warp.zero(ker.rt((n, n), DType::Float32, TileLayout::Col, RT_16X16));
        let c = warp.mma_ab(c, &a, &b);
        let _ = warp.store(c_gl, c, MoveIdx::block((0, 0, 0, 0), 2));
        ker.finish(1)
    };

    let wmma_count = |sink: &Arc<UOp>| sink.toposort().iter().filter(|u| matches!(u.op(), Op::Wmma(..))).count();
    assert_eq!(wmma_count(&build(false)), 1, "looped mma → one symbolic WMMA node");
    assert_eq!(wmma_count(&build(true)), 8, "unrolled mma → 8 flat WMMA nodes (2×2 output × 2 K-steps)");

    let render = |sink: Arc<UOp>| {
        let pm = svod_schedule::symbolic::pm_lower_index_dtype()
            + svod_ir::decompositions::divmod_decomposition_patterns()
                .with_context::<svod_schedule::symbolic::WeakMemo>();
        let lowered = svod_schedule::graph_rewrite(&pm, sink, &mut svod_schedule::symbolic::WeakMemo::default());
        let program =
            svod_codegen::program_pipeline::program_from_sink(lowered, DeviceSpec::Cpu).expect("final target graph");
        let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
        let linear_uop =
            linearized.toposort().into_iter().find(|u| matches!(u.op(), Op::Linear(..))).expect("LINEAR present");
        let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(svod_dtype::AmdArch::Gfx942);
        svod_codegen::traits::Renderer::render(&renderer, &linear_uop, Some("mma_unroll_probe")).expect("render").code
    };
    // Count mfma *call sites* — exclude the single (deduped) `declare` line.
    let mfma =
        |code: &str| code.lines().filter(|l| l.contains("mfma.f32.16x16x16bf16.1k") && !l.contains("declare")).count();
    let (looped_mfma, unrolled_mfma) = (mfma(&render(build(false))), mfma(&render(build(true))));
    // The flatness proof: unrolling renders all 8 MFMAs as distinct flat
    // instructions (a rolled K/fragment loop cannot — it renders strictly fewer).
    assert_eq!(unrolled_mfma, 8, "unrolled mma renders 8 flat mfma — no rolled K/fragment loop");
    assert!(looped_mfma < 8, "looped mma keeps the K/fragment loops rolled ({looped_mfma} < 8 static mfma)");
}

/// gfx1151 (RDNA3.5) matmul renders to **WMMA**, not MFMA (host, no GPU). Built
/// with wave32 [`crate::ArchCaps`], the kernel must select the `_W32_*` fragment
/// shapes and lower to `llvm.amdgcn.wmma.f32.16x16x16.bf16` (with bf16 inputs
/// bitcast to `<16 x i16>` and an `<8 x float>` accumulator) — never an `mfma`
/// (CDNA-only). This proves the arch-select + wave32 layout build & emit the
/// right intrinsic; numerical correctness is gated on gfx1151 hardware.
#[test]
fn test_matmul_rdna_renders_wmma() {
    let n = 64usize; // SMALL_CFG: block=64, 1 wave, 32 threads (wave32).
    let ker = Kernel::new(
        "matmul_rdna",
        SMALL_CFG.grid_dims(n),
        SMALL_CFG.threads(32),
        dummy_buffers(n),
        crate::ArchCaps::for_amd(svod_dtype::AmdArch::Gfx1151),
    );
    build_matmul_cfg(&ker, n, SMALL_CFG);
    let sink = ker.finish(SMALL_CFG.n_accum);
    let pm = svod_schedule::symbolic::pm_lower_index_dtype()
        + svod_ir::decompositions::divmod_decomposition_patterns().with_context::<svod_schedule::symbolic::WeakMemo>();
    let lowered = svod_schedule::graph_rewrite(&pm, sink, &mut svod_schedule::symbolic::WeakMemo::default());
    let program =
        svod_codegen::program_pipeline::program_from_sink(lowered, DeviceSpec::Cpu).expect("final target graph");
    let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
    let linear_uop =
        linearized.toposort().into_iter().find(|u| matches!(u.op(), Op::Linear(..))).expect("LINEAR present");
    let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(svod_dtype::AmdArch::Gfx1151);
    // Renders (no OOM/panic) ⇒ the wave32 fragment shapes lower cleanly.
    let code =
        svod_codegen::traits::Renderer::render(&renderer, &linear_uop, Some("matmul_rdna")).expect("render").code;

    assert!(code.contains("llvm.amdgcn.wmma.f32.16x16x16.bf16"), "gfx1151 matmul must emit the RDNA WMMA intrinsic");
    assert!(!code.contains("mfma"), "gfx1151 is WMMA, not CDNA MFMA");
}

/// CUDA graph shape (no GPU): one 16×16 fragment product on `mma.sync` is TWO
/// `m16n8k16` WMMAs per K-iteration — `(8,16,16)` / `8-4-4` — over the two-half
/// registers: A = all 8, B = `{2h,2h+1,2h+4,2h+5}`, C = `4h..4h+4` for a `Row`
/// accumulator; a `Col` accumulator computes `Cᵀ += Bᵀ·Aᵀ`, so the WMMA's first
/// operand comes from the B tile (8 registers) and the second from the A tile.
#[test_case(TileLayout::Row, false; "row accumulator")]
#[test_case(TileLayout::Col, true; "col accumulator swaps operands")]
fn test_mma_ab_mma_sync_graph_shape(acc_layout: TileLayout, swapped: bool) {
    let ker = Kernel::new("mma_sync_probe", [1, 1, 1], 32, vec![], crate::ArchCaps::for_arch(SM_86));
    // Unrolled: every tile index is a constant, so a register load's flat INDEX
    // offset is the bare register number.
    ker.set_unroll(true);
    let warp = ker.warp();
    let a = ker.rt((16, 16), DType::BFloat16, TileLayout::Row, RT_16X16_MMA);
    let b = ker.rt((16, 16), DType::BFloat16, TileLayout::Col, RT_16X16_MMA);
    let c = warp.zero(ker.rt((16, 16), DType::Float32, acc_layout, RT_16X16_MMA));
    let out = warp.mma_ab(c, &a, &b);

    let wmmas: Vec<_> = out.uop().toposort().into_iter().filter(|u| matches!(u.op(), Op::Wmma(..))).collect();
    assert_eq!(wmmas.len(), 2, "two m16n8k16 halves per 16×16 fragment");
    let regs = |stack: &Arc<UOp>| -> Vec<i64> {
        let Op::Stack(ops::Stack { sources }) = stack.op() else { panic!("WMMA operands are STACKs") };
        sources
            .iter()
            .map(|load| {
                let Op::Load(ops::Load { index, .. }) = load.op() else { panic!("STACK of register LOADs") };
                let Op::Index(ops::Index { indices, .. }) = index.op() else { unreachable!() };
                match indices[0].op() {
                    Op::Const(c) => match c.0 {
                        svod_ir::ConstValue::Int(v) => v,
                        other => panic!("register offset {other:?}"),
                    },
                    other => panic!("register offset must be a constant, got {other:?}"),
                }
            })
            .collect()
    };
    let mut halves: Vec<(Vec<i64>, Vec<i64>, Vec<i64>)> = wmmas
        .iter()
        .map(|w| {
            let Op::Wmma(ops::Wmma { a, b, c, metadata }) = w.op() else { unreachable!() };
            assert_eq!(metadata.dims, (8, 16, 16));
            assert_eq!(metadata.threads, 32);
            let width = |u: &Arc<UOp>| u.shape().unwrap().unwrap()[0].as_const();
            assert_eq!((width(a), width(b), width(c)), (Some(8), Some(4), Some(4)), "A/B/C register widths");
            (regs(a), regs(b), regs(c))
        })
        .collect();
    halves.sort_by_key(|(_, _, c)| c[0]);
    assert_eq!(halves[0].2, vec![0, 1, 2, 3], "half 0 accumulates registers 0..4");
    assert_eq!(halves[1].2, vec![4, 5, 6, 7], "half 1 accumulates registers 4..8");
    for (h, (x, y, _)) in halves.iter().enumerate() {
        let h = h as i64;
        assert_eq!(x, &(0..8).collect::<Vec<_>>(), "half {h}: the 8-register operand");
        assert_eq!(y, &vec![2 * h, 2 * h + 1, 2 * h + 4, 2 * h + 5], "half {h}: the n-half operand");
    }
    // Which tile feeds the 8-register slot: the A tile (`Row` acc) or the B tile (`Col` acc).
    let feeds_first = |w: &Arc<UOp>, t: &crate::RT<'_>| {
        let Op::Wmma(ops::Wmma { a, .. }) = w.op() else { unreachable!() };
        a.toposort().iter().any(|u| Arc::ptr_eq(u, t.uop()))
    };
    for w in &wmmas {
        assert_eq!(feeds_first(w, &b), swapped, "Col accumulator ⇒ the B tile supplies the A fragment");
        assert_eq!(feeds_first(w, &a), !swapped);
    }
}

/// sm_86 matmul renders to `mma.sync` (host, no GPU): built with warp32
/// [`crate::ArchCaps`], [`SM80_SMALL_CFG`] must select the `RT_16X16_MMA` two-half
/// fragments and lower to `llvm.nvvm.mma.m16n8k16.row.col.bf16` — never an `mfma`
/// or `wmma` intrinsic. Numerical correctness is gated on CUDA hardware.
#[test]
fn test_matmul_sm86_renders_mma_sync() {
    let n = 64usize;
    let cfg = SM80_SMALL_CFG;
    let caps = crate::ArchCaps::for_arch(SM_86);
    let ker = Kernel::new("matmul_sm86", cfg.grid_dims(n), cfg.threads(caps.wave_size), dummy_buffers(n), caps);
    build_matmul_cfg(&ker, n, cfg);
    let sink = ker.finish(cfg.n_accum);
    // The launch path's pipeline (post-optimization with the CUDA optimizer profile,
    // then linearize + render), so the IR is what ptxas gets.
    let sm86 = CudaArch::from_compute_capability(8, 6);
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
        linearized.toposort().into_iter().find(|u| matches!(u.op(), Op::Linear(..))).expect("LINEAR present");
    let code =
        svod_codegen::traits::Renderer::render(&renderer, &linear_uop, Some("matmul_sm86")).expect("render").code;
    if let Ok(dump) = std::env::var("TK_DUMP_IR") {
        std::fs::write(dump, &code).expect("dump IR");
    }
    assert!(code.contains("llvm.nvvm.mma.m16n8k16.row.col.bf16"), "sm_86 matmul must emit mma.sync m16n8k16 bf16");
    assert!(!code.contains("mfma") && !code.contains("amdgcn"), "no AMD intrinsics on NVPTX");
    // The LDS strips fill by `cp.async` (one 16-byte copy per lane per pass of each
    // strip, retired by `wait_group 0` + barrier) and the operands gather by
    // `ldmatrix.x4`: plain for the `Row` A sub-tile, `.trans` for the `Col` B sub-tile.
    let calls = |needle: &str| code.lines().filter(|l| l.contains(needle) && !l.contains("declare")).count();
    let (reg, k_step) = (cfg.reg(), cfg.k_step());
    let passes = (cfg.block * k_step * 2).div_ceil(cfg.threads(32) as usize * 16);
    assert_eq!(calls("@llvm.nvvm.cp.async.cg.shared.global.16("), 2 * passes);
    assert_eq!(calls("@llvm.nvvm.cp.async.wait.group(i32 0)"), 2);
    assert_eq!(calls("ldmatrix.sync.aligned.m8n8.x4.b16("), cfg.n_accum * (reg / 16) * (k_step / 16), "A");
    assert_eq!(calls("ldmatrix.sync.aligned.m8n8.x4.trans.b16("), (k_step / 16) * (reg / 16), "B");
    if let Some(ptx) = super::fa::ptx_of(&code) {
        let count = |needle: &str| ptx.matches(needle).count();
        assert_eq!((count("ld.shared.b16"), count("st.shared")), (0, 0), "scalar LDS traffic in the PTX:\n{ptx}");
        assert_eq!(
            count("ldmatrix.sync.aligned.m8n8.x4"),
            cfg.n_accum * (reg / 16) * (k_step / 16) + (k_step / 16) * (reg / 16)
        );
        assert_eq!(count("cp.async.cg.shared.global"), 2 * passes);
    }
}

/// A `group_2d(2,4)` is 8 waves / 512 threads, with `warp_row`/`warp_col`
/// derived as `div`/`mod` of the wave id by `cols_waves`.
#[test]
fn test_group_2d_wave_index_shape() {
    use svod_ir::{BinaryOp, Op};

    let ker = Kernel::new("wave_probe", [1, 1, 1], 512, vec![], crate::ArchCaps::GFX942);
    let g = ker.group_2d(2, 4);
    assert_eq!(g.warps, 8, "2×4 wave grid = 8 waves");
    assert_eq!(g.rows_waves, 2);
    assert_eq!(g.cols_waves, 4);
    assert_eq!(g.group_threads(), 512, "8 waves × 64 = 512 threads/block");

    // warp_row = warpid / cols_waves (=4); warp_col = warpid % 4.
    let by_four = |u: &Arc<UOp>, op| {
        u.toposort().into_iter().any(|n| {
            matches!(n.op(), Op::Binary(o, _, d) if *o == op
                && matches!(d.op(), Op::Const(c) if matches!(c.0, svod_ir::ConstValue::Int(4))))
        })
    };
    assert!(by_four(&g.warp_row(), BinaryOp::FloorDiv), "warp_row divides the wave id by cols_waves=4");
    assert!(by_four(&g.warp_col(), BinaryOp::FloorMod), "warp_col mods the wave id by cols_waves=4");

    // Single-warp group keeps the 1×1 grid.
    let w = ker.warp();
    assert_eq!((w.warps, w.rows_waves, w.cols_waves, w.group_threads()), (1, 1, 1, 64));
}

/// `st_db` allocates a 2×-size LDS buffer, and a parity `with_base_offset` view
/// threads a runtime offset into the LDS flat address (so a double-buffer
/// gather/fill is counter-dependent and stays loop-scoped), while an ordinary
/// `st` tile's addresses carry no such offset.
#[test]
fn test_st_db_base_offset_infra() {
    use crate::tiles::ST_16X16_SWIZZLED;

    let ker = Kernel::new("db_infra", [1, 1, 1], 512, vec![], crate::ArchCaps::GFX942);
    // Single-half flat element count for a 256×32 bf16 tile (base 16×16):
    // (256/16)*(32/16)*16*16 = 16*2*256 = 8192.
    let db = ker.st_db((256, 32), DType::BFloat16, TileLayout::Row, ST_16X16_SWIZZLED);
    assert_eq!(db.half_elems(), 8192, "half_elems = height*width*base.rows*base.cols");
    assert!(db.base_offset().is_none(), "fresh st_db addresses half 0 (no base_offset)");

    // A parity view adds `parity * half_elems` to the flat address.
    let tile = ker.range(4); // a Loop range counter
    let parity = tile.try_mod(&crate::index::cidx(2)).expect("tile % 2");
    let off = parity.try_mul(&crate::index::cidx(db.half_elems() as i64)).expect("parity*half");
    let view = db.with_base_offset(off.clone());
    assert!(view.base_offset().is_some(), "with_base_offset sets the parity select");

    // Sanity: the underlying buffer is shared (same DefineLocal), only the view differs.
    assert!(std::sync::Arc::ptr_eq(db.uop(), view.uop()), "with_base_offset shares the backing buffer");
}

// =============================================================================
// Hardware-gated end-to-end matmul (gfx942 / gfx1151 / CUDA sm_80+).
// =============================================================================

/// The env-selected device's arch when the tile matmul supports it (its LLVM
/// backend present), so a test picks [`cfg_for_arch`] the way `matmul()` does.
fn matmul_device() -> Option<GpuArch> {
    let spec = Tensor::empty(&[1], DType::Float32).device();
    crate::target::resolve_supported_arch(&spec, MATMUL_SUPPORTED_ARCHS).ok()
}

/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib matmul::test_simple_matmul_gpu -- --ignored --nocapture`.
///
/// Runs the arch's large-N tile matmul (the 8-wave 256×256 [`M1_CFG`] on gfx942)
/// on the real GPU across several N and checks each against a reference
/// `a.matmul(b)` over the *same* bf16-rounded operands (bf16 tolerance ~5e-2).
#[test]
#[ignore]
fn test_simple_matmul_gpu() {
    let Some(arch) = matmul_device() else {
        eprintln!("skip test_simple_matmul_gpu: unsupported device/toolchain");
        return;
    };
    for n in [256usize, 512, 1024, 2048] {
        let cfg = cfg_for_arch(arch, n);
        let (a, b) = matmul_inputs(n);
        let got = launch_matmul("simple_matmul", n, cfg, |ker| build_matmul_cfg(ker, n, cfg), &a, &b);
        let max_abs = max_abs_err(&got, &matmul_reference(&a, &b));
        println!("matmul N={n} block={}: max abs error = {max_abs:e}", cfg.block);
        assert!(max_abs < 5e-2, "N={n}: max abs error {max_abs} exceeds bf16 tolerance 5e-2");
    }
}

/// The chiplet/L2 grid swizzle in **isolation** (1-D grid + [`l2_swizzle`],
/// scalar fill) over the arch's large-N config. It permutes which workgroup
/// computes which C block, so the full C must be bit-identical-up-to-bf16-tolerance
/// to `a.matmul(b)`.
///
/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib matmul::test_matmul_l2swizzle_gpu -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_l2swizzle_gpu() {
    let Some(arch) = matmul_device() else {
        eprintln!("skip test_matmul_l2swizzle_gpu: unsupported device/toolchain");
        return;
    };
    let cfg = MatmulCfg { l2_swizzle: true, vec_load: false, ..cfg_for_arch(arch, 2048) };
    for n in [2048usize, 4096] {
        let (a, b) = matmul_inputs(n);
        let got = launch_matmul("matmul_l2sw", n, cfg, |ker| build_matmul_cfg(ker, n, cfg), &a, &b);
        let expected = matmul_reference(&a, &b);
        let max_abs = max_abs_err(&got, &expected);
        println!("l2swizzle N={n}: max abs error = {max_abs:e}");
        assert!(max_abs < 5e-2, "l2swizzle N={n}: max abs error {max_abs} exceeds 5e-2");
    }
}

/// Realized bf16 `(a, b)` inputs so kernel + reference see identical rounding.
fn matmul_inputs(n: usize) -> (svod_tensor::Tensor, svod_tensor::Tensor) {
    matmul_inputs_dt(n, DType::BFloat16)
}

/// f32 ground-truth `a·b` over the bf16-rounded operands.
fn matmul_reference(a: &svod_tensor::Tensor, b: &svod_tensor::Tensor) -> Vec<f32> {
    let mut reference =
        a.cast(DType::Float32).expect("a→f32").matmul(&b.cast(DType::Float32).expect("b→f32")).expect("ref matmul");
    reference.realize().expect("realize reference");
    reference.as_vec::<f32>().expect("read reference")
}

fn max_abs_err(got: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(got.len(), expected.len(), "length mismatch");
    got.iter().zip(expected).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max)
}

/// The wave32 (gfx1151) matmul computes exactly `A·B` — not a transposed or
/// operand-swapped variant. Compares `got` against every transpose/permutation
/// candidate and asserts `A·B` is the unique match (the rest are garbage-scale).
/// A layout regression in the wave32 fragment map would flip which candidate wins.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_rdna_computes_ab -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_rdna_computes_ab() {
    if !super::device_supported(crate::kernels::matmul::MATMUL_SUPPORTED_ARCHS) {
        eprintln!("skip test_matmul_rdna_computes_ab: unsupported device/toolchain");
        return;
    }
    use svod_tensor::Tensor;
    let n = 64usize;
    let (a, b) = matmul_inputs(n);
    let got = launch_matmul("matmul_diag", n, SMALL_CFG, |ker| build_matmul_cfg(ker, n, SMALL_CFG), &a, &b);

    let f = |t: &Tensor| t.cast(DType::Float32).expect("→f32");
    let (af, bf) = (f(&a), f(&b));
    let tr = |x: &Tensor| x.try_permute(&[1, 0]).expect("transpose");
    let mm = |x: &Tensor, y: &Tensor| x.matmul(y).expect("matmul");
    let vec = |mut x: Tensor| {
        x.realize().expect("realize");
        x.as_vec::<f32>().expect("read")
    };

    let ab_err = max_abs_err(&got, &vec(mm(&af, &bf)));
    // bf16 accumulation over K=64 ⇒ a few thousandths; transposes/swaps are O(1).
    assert!(ab_err < 1e-1, "wave32 matmul should equal A·B, got max abs err {ab_err:e}");

    let wrong: Vec<(&str, Tensor)> = vec![
        ("(A·B)^T", tr(&mm(&af, &bf))),
        ("A^T·B", mm(&tr(&af), &bf)),
        ("A·B^T", mm(&af, &tr(&bf))),
        ("A^T·B^T", mm(&tr(&af), &tr(&bf))),
        ("B·A", mm(&bf, &af)),
        ("(B·A)^T", tr(&mm(&bf, &af))),
    ];
    for (name, cand) in wrong {
        let err = max_abs_err(&got, &vec(cand));
        assert!(err > 1.0, "wave32 matmul matches {name} (err {err:e}) — layout is not plain A·B");
    }
}

/// Element-level check of the wave32 fragment lane→(m,n) map: `A = I`,
/// `B[k][j] = (k%16)*16 + (j%16)` ⇒ `C = B`, so the first 16×16 output fragment must
/// read `got[i][j] = i*16 + j`. Any within-fragment permutation lands a source
/// element at the wrong `(i,j)` and trips the assert (printing the offending row).
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_rdna_grid -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_rdna_grid() {
    if !super::device_supported(crate::kernels::matmul::MATMUL_SUPPORTED_ARCHS) {
        eprintln!("skip test_matmul_rdna_grid: unsupported device/toolchain");
        return;
    }
    use svod_tensor::Tensor;
    let n = 64usize;
    let mut a_data = vec![0f32; n * n];
    for i in 0..n {
        a_data[i * n + i] = 1.0; // identity
    }
    let b_data: Vec<f32> = (0..n * n).map(|p| (((p / n) % 16) * 16 + (p % n) % 16) as f32).collect();
    let mk =
        |d: &[f32]| Tensor::from_slice(d).try_reshape([n, n]).expect("reshape").cast(DType::BFloat16).expect("→bf16");
    let (mut a, mut b) = (mk(&a_data), mk(&b_data));
    a.realize().expect("realize a");
    b.realize().expect("realize b");
    let got = launch_matmul("matmul_grid", n, SMALL_CFG, |ker| build_matmul_cfg(ker, n, SMALL_CFG), &a, &b);

    for i in 0..16 {
        let row: Vec<i32> = (0..16).map(|j| got[i * n + j].round() as i32).collect();
        let expected: Vec<i32> = (0..16).map(|j| (i * 16 + j) as i32).collect();
        assert_eq!(row, expected, "fragment(0,0) row i={i} permuted: {row:?} (expected {expected:?})");
    }
}

/// Whether the env-selected device is CUDA sm_80+ with the NVPTX toolchain.
fn cuda_device() -> bool {
    super::fragment_device().is_some_and(|caps| caps.cuda().is_some())
}

/// `SVOD_DEVICE=CUDA:0 cargo test -p svod-tk --lib matmul::test_matmul_cuda -- --ignored --nocapture`.
///
/// The `mma.sync` matmul on the real GPU across both configs, several N and both
/// 16-bit input dtypes (f16 and bf16, f32 accumulate) against the f32 reference
/// over the SAME rounded operands.
#[test]
#[ignore]
fn test_matmul_cuda() {
    if !cuda_device() {
        eprintln!("skip test_matmul_cuda: no CUDA sm_80+ device / toolchain");
        return;
    }
    for dtype in [DType::BFloat16, DType::Float16] {
        for n in [64usize, 128, 192, 256, 512, 1024] {
            let cfg = cfg_for_arch(SM_86, n);
            let (a, b) = matmul_inputs_dt(n, dtype.clone());
            let got = launch_matmul("matmul_cuda", n, cfg, |ker| build_matmul_cfg(ker, n, cfg), &a, &b);
            let max_abs = max_abs_err(&got, &matmul_reference(&a, &b));
            println!("matmul[cuda] {dtype:?} N={n} block={}: max abs error = {max_abs:e}", cfg.block);
            let tol = if dtype == DType::Float16 { 2e-2 } else { 5e-2 };
            assert!(max_abs < tol, "{dtype:?} N={n}: max abs error {max_abs} exceeds {tol}");
        }
    }
}

/// Element-level check of the `mma.sync` two-half lane map (the CUDA peer of
/// [`test_matmul_rdna_grid`]): `A = I`, `B[k][j] = (k%16)·16 + j%16` ⇒ `C = B`, so the
/// first 16×16 output fragment must read `got[i][j] = i·16 + j`; a within-fragment
/// permutation of the A/B/C register order lands at the wrong `(i, j)`.
/// `SVOD_DEVICE=CUDA:0 cargo test -p svod-tk --lib matmul::test_matmul_cuda_grid -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_cuda_grid() {
    if !cuda_device() {
        eprintln!("skip test_matmul_cuda_grid: no CUDA sm_80+ device / toolchain");
        return;
    }
    let n = 64usize;
    let mut a_data = vec![0f32; n * n];
    for i in 0..n {
        a_data[i * n + i] = 1.0;
    }
    let b_data: Vec<f32> = (0..n * n).map(|p| (((p / n) % 16) * 16 + (p % n) % 16) as f32).collect();
    let mk =
        |d: &[f32]| Tensor::from_slice(d).try_reshape([n, n]).expect("reshape").cast(DType::BFloat16).expect("→bf16");
    let (mut a, mut b) = (mk(&a_data), mk(&b_data));
    a.realize().expect("realize a");
    b.realize().expect("realize b");
    let cfg = SM80_SMALL_CFG;
    let got = launch_matmul("matmul_cuda_grid", n, cfg, |ker| build_matmul_cfg(ker, n, cfg), &a, &b);
    for i in 0..16 {
        let row: Vec<i32> = (0..16).map(|j| got[i * n + j].round() as i32).collect();
        let expected: Vec<i32> = (0..16).map(|j| (i * 16 + j) as i32).collect();
        assert_eq!(row, expected, "fragment(0,0) row i={i} permuted: {row:?} (expected {expected:?})");
    }
}

/// The four `mma_{ab,abt,atb,atbt}` variants and both accumulator orientations on
/// CUDA compute exactly `A·B` from register tiles gathered straight from GLOBAL
/// (no LDS): `Row`/`Col` operand readings pack the same registers, and a `Col`
/// accumulator (`Cᵀ += Bᵀ·Aᵀ`) stores the same matrix.
/// `SVOD_DEVICE=CUDA:0 cargo test -p svod-tk --lib matmul::test_mma_variants_cuda -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_mma_variants_cuda() {
    if !cuda_device() {
        eprintln!("skip test_mma_variants_cuda: no CUDA sm_80+ device / toolchain");
        return;
    }
    let n = 32usize;
    let (a, b) = matmul_inputs_dt(n, DType::BFloat16);
    let expected = matmul_reference(&a, &b);
    let tr = |x: &Tensor| {
        let mut t = x.try_permute(&[1, 0]).expect("transpose").contiguous();
        t.realize().expect("realize");
        t
    };
    let (at, bt) = (tr(&a), tr(&b));
    for (name, a_t, b_t) in [("ab", false, false), ("abt", false, true), ("atb", true, false), ("atbt", true, true)] {
        for acc_layout in [TileLayout::Row, TileLayout::Col] {
            let (a_in, b_in) = (if a_t { &at } else { &a }, if b_t { &bt } else { &b });
            let mut c = Tensor::empty(&[n, n], DType::Float32);
            crate::run_kernel(format!("mma_{name}"), [1, 1, 1], 32, &mut [&mut c], &[a_in, b_in], |ker| {
                let warp = ker.warp();
                let c_gl = ker.gl(&[1, 1, n, n], DType::Float32);
                let a_gl = ker.gl(&[1, 1, n, n], DType::BFloat16);
                let b_gl = ker.gl(&[1, 1, n, n], DType::BFloat16);
                // A stored `[m,k]` reads `Row`; `Aᵀ` stored `[k,m]` reads `Col` (the same
                // registers); symmetric for B (`[k,n]` is `Col`, `Bᵀ` `[n,k]` is `Row`).
                let a_l = if a_t { TileLayout::Col } else { TileLayout::Row };
                let b_l = if b_t { TileLayout::Row } else { TileLayout::Col };
                let ra = warp.load(ker.operand((n, n), DType::BFloat16, a_l), a_gl, MoveIdx::block((0, 0, 0, 0), 2));
                let rb = warp.load(ker.operand((n, n), DType::BFloat16, b_l), b_gl, MoveIdx::block((0, 0, 0, 0), 2));
                let acc = warp.zero(ker.acc((n, n), acc_layout));
                let acc = match (a_t, b_t) {
                    (false, false) => warp.mma_ab(acc, &ra, &rb),
                    (false, true) => warp.mma_abt(acc, &ra, &rb),
                    (true, false) => warp.mma_atb(acc, &ra, &rb),
                    (true, true) => warp.mma_atbt(acc, &ra, &rb),
                };
                let _ = warp.store(c_gl, acc, MoveIdx::block((0, 0, 0, 0), 2));
                ker.finish(1)
            })
            .expect("mma variant launch");
            let got = c.as_vec::<f32>().expect("read c");
            let err = max_abs_err(&got, &expected);
            println!("mma_{name} acc={acc_layout:?}: max abs error = {err:e}");
            assert!(err < 5e-2, "mma_{name} acc={acc_layout:?}: max abs error {err}");
        }
    }
}

/// Realized `(a, b)` inputs of `dtype` so kernel + reference see identical rounding.
fn matmul_inputs_dt(n: usize, dtype: DType) -> (Tensor, Tensor) {
    let mut a = Tensor::rand(&[n, n]).expect("rand a").cast(dtype.clone()).expect("cast a");
    let mut b = Tensor::rand(&[n, n]).expect("rand b").cast(dtype).expect("cast b");
    a.realize().expect("realize a");
    b.realize().expect("realize b");
    (a, b)
}

/// Build + dispatch a matmul `cfg` over `(a, b)` once, returning the f32 C.
fn launch_matmul<F>(
    name: &str,
    n: usize,
    cfg: MatmulCfg,
    build: F,
    a: &svod_tensor::Tensor,
    b: &svod_tensor::Tensor,
) -> Vec<f32>
where
    F: FnOnce(&Kernel),
{
    use svod_tensor::Tensor;
    // Launch block must match the device wave size (gfx942 wave64, gfx11 wave32),
    // matching the `matmul()` entry's `cfg.threads(caps.wave_size)`.
    let ws = crate::target::resolve_arch(&a.device()).map(|ar| crate::ArchCaps::for_arch(ar).wave_size).unwrap_or(64);
    let mut c = Tensor::empty(&[n, n], DType::Float32);
    crate::run_kernel(name, cfg.grid_dims(n), cfg.threads(ws), &mut [&mut c], &[a, b], |ker| {
        build(ker);
        ker.finish(cfg.n_accum)
    })
    .expect("matmul launch");
    c.as_vec::<f32>().expect("read c")
}

/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib matmul::test_matmul_graph_gpu -- --ignored --nocapture`.
///
/// The graph-native `matmul` (a `custom_kernel` / `Op::Call` node) matches **(a)**
/// the f32 reference (bf16 tol) AND **(b)** the direct-launch kernel **bit-for-bit**
/// — the matmul peer of the FA graph gate, confirming the matmul SINK lowers
/// identically through `custom_kernel → realize` (the optimizer bypass is
/// kernel-agnostic) as through direct launch.
#[test]
#[ignore]
fn test_matmul_graph_gpu() {
    let Some(arch) = matmul_device() else {
        eprintln!("skip test_matmul_graph_gpu: unsupported device/toolchain");
        return;
    };
    for n in [256usize, 512, 1024] {
        let (a, b) = matmul_inputs(n);
        let expected = matmul_reference(&a, &b);
        let cfg = cfg_for_arch(arch, n);
        let direct = launch_matmul("matmul_direct", n, cfg, |ker| build_matmul_cfg(ker, n, cfg), &a, &b);

        let mut g = crate::kernels::matmul::matmul(&a, &b).expect("graph matmul").expect("matmul kernel applies");
        g.realize().expect("realize graph matmul");
        let graph = g.as_vec::<f32>().expect("read graph matmul");

        let (vs_ref, vs_direct) = (max_abs_err(&graph, &expected), max_abs_err(&graph, &direct));
        println!("matmul[graph] N={n}: vs ref = {vs_ref:e}, vs direct = {vs_direct:e}");
        assert!(vs_ref < 5e-2, "graph matmul N={n}: vs ref {vs_ref} exceeds bf16 tol 5e-2");
        assert!(vs_direct < 1e-3, "graph matmul N={n}: must match direct-launch bit-for-bit (Δ {vs_direct})");
    }
}

/// The size-adaptive matmul is correct at every N — on gfx942 picking [`SMALL_CFG`]
/// for small N (where the 256×256 block under-occupies the machine) and [`M1_CFG`]
/// otherwise; on CUDA [`SM80_SMALL_CFG`] when N only tiles by 64.
///
/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib matmul::test_matmul_adaptive_gpu -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_adaptive_gpu() {
    let Some(arch) = matmul_device() else {
        eprintln!("skip test_matmul_adaptive_gpu: unsupported device/toolchain");
        return;
    };
    for n in [256usize, 512, 768, 1024, 2048] {
        let (a, b) = matmul_inputs(n);
        let cfg = cfg_for_arch(arch, n);
        let got = launch_matmul("matmul_adaptive", n, cfg, |ker| build_matmul_cfg(ker, n, cfg), &a, &b);
        let expected = matmul_reference(&a, &b);
        let max_abs = max_abs_err(&got, &expected);
        println!("adaptive N={n} (block={}): max abs error = {max_abs:e}", cfg.block);
        assert!(max_abs < 5e-2, "adaptive N={n}: max abs error {max_abs} exceeds 5e-2");
    }
}

/// `SVOD_DEVICE=CUDA:0 cargo test -p svod-tk --lib matmul::test_matmul_bench_cuda -- --ignored --nocapture`.
///
/// Per-dispatch GPU time and TFLOP/s of the arch's bf16 matmul config at
/// N = 1024 / 2048 — the measurement behind the CUDA fill/gather lowering.
#[test]
#[ignore]
fn test_matmul_bench_cuda() {
    if !cuda_device() {
        eprintln!("skip test_matmul_bench_cuda: no CUDA sm_80+ device / toolchain");
        return;
    }
    let ws = super::fragment_device().expect("caps").wave_size;
    for n in [1024usize, 2048] {
        let cfg = cfg_for_arch(SM_86, n);
        let (a, b) = matmul_inputs(n);
        let mut c = Tensor::empty(&[n, n], DType::Float32);
        let compiled =
            crate::compile_kernel("matmul_bench", cfg.grid_dims(n), cfg.threads(ws), &mut [&mut c], &[&a, &b], |ker| {
                build_matmul_cfg(ker, n, cfg);
                ker.finish(cfg.n_accum)
            })
            .expect("compile");
        let mut us: Vec<f64> =
            (0..12).map(|_| compiled.dispatch_gpu_ns().expect("dispatch").expect("stamped") as f64 / 1e3).collect();
        us.sort_by(f64::total_cmp);
        let median = us[us.len() / 2];
        let tflops = 2.0 * (n as f64).powi(3) / (median * 1e-6) / 1e12;
        println!(
            "matmul[cuda] N={n} block={}: median {median:.1} µs = {tflops:.2} TFLOP/s (min {:.1})",
            cfg.block, us[0]
        );
    }
}

/// `mma_AB` reads B as a `[k,n]` (`Col`) tile; a `Row` tile would be
/// multiplied transposed, so the plan refuses it at build time. GPU-free.
#[test]
#[should_panic(expected = "operand B must be a Col tile")]
fn mma_ab_rejects_a_row_b_tile() {
    let ker = Kernel::new("mma_orient", [1, 1, 1], 64, vec![], crate::ArchCaps::GFX942);
    let warp = ker.warp();
    let a = ker.operand((16, 16), DType::BFloat16, TileLayout::Row);
    let b = ker.operand((16, 16), DType::BFloat16, TileLayout::Row);
    let _ = warp.mma_ab(warp.zero(ker.acc((16, 16), TileLayout::Col)), &a, &b);
}

/// Symmetrically, `mma_AtB` reads A as a `[k,m]` (`Col`) tile.
#[test]
#[should_panic(expected = "operand A must be a Col tile")]
fn mma_atb_rejects_a_row_a_tile() {
    let ker = Kernel::new("mma_orient", [1, 1, 1], 32, vec![], crate::ArchCaps::for_arch(SM_86));
    let warp = ker.warp();
    let a = ker.operand((16, 16), DType::BFloat16, TileLayout::Row);
    let b = ker.operand((16, 16), DType::BFloat16, TileLayout::Col);
    let _ = warp.mma_atb(warp.zero(ker.acc((16, 16), TileLayout::Col)), &a, &b);
}
