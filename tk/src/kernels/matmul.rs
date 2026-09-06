//! The bf16→f32 tile matmul: a single arch-generic, single-buffered builder
//! ([`build_matmul_cfg`] / [`build_matmul_cfg_k`]) driven by a per-arch
//! [`MatmulCfg`]. One `cfg.block × cfg.block` C tile per workgroup, `cfg.n_accum`
//! `reg × reg` accumulators/wave reduced over a tracked K-loop in `cfg.k_step()`-wide
//! strips out of XOR-swizzled LDS. The tile shortcuts ([`Kernel::acc`] /
//! [`Kernel::operand`] / [`Kernel::shared_sw`]) resolve the right WMMA fragment per
//! arch (gfx942 CDNA MFMA vs gfx1151 RDNA WMMA), so one builder serves both.
//!
//! Per-arch tuning lives in the configs ([`M1_CFG`] / [`SMALL_CFG`] for gfx942's
//! size-adaptive [`cfg_for_n`]; [`GFX1151_CFG`] for the occupancy-tuned RDNA3.5 path;
//! [`SM80_CFG`] / [`SM80_SMALL_CFG`] for CUDA `mma.sync`), selected by [`cfg_for_arch`]. The strongest gfx1151 lever is `k_step`: a smaller
//! strip shrinks the live WMMA-input fragment VGPR/lane (each input replicates all
//! `k_step`/16 K-sub-steps), raising occupancy. A port of tinygrad
//! `test_tk.py::test_simple_matmul` lifted to a reusable kernel builder.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::UOp;
use svod_tensor::Tensor;

use crate::index::{Idx, cidx};
use crate::tiles::TileLayout;
use crate::{GL, GlSpec, Kernel, MoveIdx, RT, RegTile};

/// K-reduction step (the LDS strip depth, shared by every config). HK `GEMM:6`.
pub const K_STEP: usize = 64;

/// Block / wave geometry of a multi-wave matmul (HK `GEMM:5-8,67-68`): a
/// `wave_rows × wave_cols`-wave workgroup computes a `block × block` C tile,
/// each wave owning `n_accum` col-major `reg × reg` f32 accumulators
/// (`reg = block / wave_cols`) reduced over K in [`K_STEP`]-wide steps. The
/// `wave_cols = wave_rows * n_accum` invariant keeps `reg` square: the M side is
/// split into `wave_rows * n_accum` row-blocks, the N side into `wave_cols`.
#[derive(Clone, Copy)]
pub struct MatmulCfg {
    /// The square C-tile edge (in elements) one workgroup computes.
    pub block: usize,
    /// Wave grid rows — the M side splits into `wave_rows * n_accum` row-blocks.
    pub wave_rows: usize,
    /// Wave grid columns — the N side splits into `wave_cols` col-blocks.
    pub wave_cols: usize,
    /// `reg × reg` f32 accumulators per wave.
    pub n_accum: usize,
    /// Drive `(pid_m, pid_n)` from a flattened 1-D grid via the chiplet/L2
    /// [`l2_swizzle`](crate::grid::l2_swizzle) instead of the plain 2-D
    /// `block_idx`. Grid becomes `[grid² , 1, 1]`.
    pub l2_swizzle: bool,
    /// Fill the GLOBAL→LDS strips with 128-bit (`vec8` bf16) coalesced loads
    /// instead of the scalar/`vec4`-folded path (16-byte `cp.async` copies on CUDA).
    pub vec_load: bool,
    /// K-reduction step (LDS strip depth) for the single-buffered K-loop. Must be a
    /// multiple of 16 (the WMMA K-edge) and divide N. Lowering it cuts the live
    /// operand VGPR/lane (each WMMA input replicates all `k_step`/16 K-sub-steps),
    /// raising occupancy — the dominant occupancy lever on RDNA3.5/wave32
    /// ([`GFX1151_CFG`] uses 32). gfx942 keeps [`K_STEP`]
    /// (64). `0` means "use [`K_STEP`]" so older literal/`..M1_CFG` builders that
    /// predate the field still get the default — see [`MatmulCfg::k_step`].
    pub k_step: usize,
}

impl MatmulCfg {
    /// The per-accumulator square edge (`block / wave_cols`).
    pub const fn reg(&self) -> usize {
        self.block / self.wave_cols
    }
    /// The K-reduction step, resolving the `0` sentinel (older literal builders) to
    /// the default [`K_STEP`]. The resolved value must be a multiple of 16 (the WMMA
    /// K-edge) and divide N; a violation panics in [`gemm_core`].
    pub const fn k_step(&self) -> usize {
        if self.k_step == 0 { K_STEP } else { self.k_step }
    }
    /// `reg`-blocks per C-tile side (= `wave_cols` = `wave_rows * n_accum`); the
    /// grid→C-block coordinate multiplier.
    pub const fn blocks_per_side(&self) -> usize {
        self.block / self.reg()
    }
    /// Launch block size (threads) = `wave_rows * wave_cols * wave_size`.
    pub const fn threads(&self, wave_size: usize) -> i64 {
        (self.wave_rows * self.wave_cols * wave_size) as i64
    }
    /// Grid edge (`n / block`).
    pub const fn grid(&self, n: usize) -> i64 {
        (n / self.block) as i64
    }
    /// Launch grid for a general `m × n` C: a flattened 1-D `[gm·gn, 1, 1]` when
    /// the chiplet swizzle ([`l2_swizzle`]) is on (it re-derives `(pid_m, pid_n)`), else
    /// the plain 2-D `[gn, gm, 1]` (x = n-blocks → `block_idx[0]` = pid_n, y = m-blocks
    /// → `block_idx[1]` = pid_m — matching [`block_coords`]).
    pub const fn grid_dims_mn(&self, m: usize, n: usize) -> [i64; 3] {
        let (gm, gn) = ((m / self.block) as i64, (n / self.block) as i64);
        if self.l2_swizzle { [gm * gn, 1, 1] } else { [gn, gm, 1] }
    }
    /// Square convenience: [`grid_dims_mn`] with `m = n` (the `[grid², 1, 1]` /
    /// `[grid, grid, 1]` the square matmul launches with).
    pub const fn grid_dims(&self, n: usize) -> [i64; 3] {
        self.grid_dims_mn(n, n)
    }
}

/// 8-wave (2×4) 256×256 block, two 64×64 accumulators/wave, 512
/// threads, the chiplet/L2 grid swizzle, and 128-bit vectorized LDS fills.
pub const M1_CFG: MatmulCfg =
    MatmulCfg { block: 256, wave_rows: 2, wave_cols: 4, n_accum: 2, l2_swizzle: true, vec_load: true, k_step: K_STEP };
/// Small-N: single-warp 64×64 block, one 64×64 accumulator, 64 threads — the
/// grid is `(n/64)²` workgroups, ~16× the large-N config's at a given N, so a small N keeps the
/// 304-CU machine fed instead of collapsing to a handful of 256×256 blocks.
/// Keeps the plain 2-D grid + scalar fill (the swizzle/vec wins are large-N).
pub const SMALL_CFG: MatmulCfg =
    MatmulCfg { block: 64, wave_rows: 1, wave_cols: 1, n_accum: 1, l2_swizzle: false, vec_load: false, k_step: K_STEP };

/// gfx1151 (RDNA3.5, wave32) config: 64×64 block, 2×2
/// waves (4 waves / 128 threads), ONE
/// 32×32 accumulator/wave, 128-bit vec fills, no L2 swizzle (single-XCD APU), and
/// **`k_step = 32`**. The `reg=32` tile keeps accumulator VGPR ≈ 32/lane; the
/// `k_step=32` halves the live WMMA-input fragment VGPR vs the default 64 (the input
/// replicates all `k_step`/16 K-sub-steps per lane), raising occupancy. `k_step` is
/// the dominant occupancy lever on RDNA3.5/wave32; the single-buffered path has no
/// memory stall a double buffer could hide. gfx942 keeps `k_step = K_STEP` (64). A
/// smaller `k_step` lowers the WMMA-input VGPR but adds barriers, so the tuned value
/// trades occupancy against barrier overhead.
pub const GFX1151_CFG: MatmulCfg =
    MatmulCfg { block: 64, wave_rows: 2, wave_cols: 2, n_accum: 1, l2_swizzle: false, vec_load: true, k_step: 32 };

/// CUDA sm_80+ (`mma.sync`, warp32) config: 128×128 block, 2×4 waves (256 threads),
/// two 32×32 accumulators/wave, 128-bit vec fills, `k_step = 32`. The 8-register
/// two-half fragments make register pressure the lever: a 32×32 f32 accumulator
/// is 32 regs/lane and each 32×32 operand strip 64 packed halves, so `reg = 32`
/// and the short strip keep a warp well under the 255-register ceiling; the
/// 2×(128×32) bf16 strips are 16 KiB of the 48 KiB static shared budget.
pub const SM80_CFG: MatmulCfg =
    MatmulCfg { block: 128, wave_rows: 2, wave_cols: 4, n_accum: 2, l2_swizzle: false, vec_load: true, k_step: 32 };
/// CUDA sm_80+ small-N config (N a multiple of 64 but not 128): 64×64 block, 2×2
/// waves, one 32×32 accumulator/wave.
pub const SM80_SMALL_CFG: MatmulCfg =
    MatmulCfg { block: 64, wave_rows: 2, wave_cols: 2, n_accum: 1, l2_swizzle: false, vec_load: true, k_step: 32 };

/// Size-adaptive config selection: small N (where the 256×256/8-wave grid
/// starves the machine) uses [`SMALL_CFG`]; everything else keeps [`M1_CFG`].
/// Small N uses an occupancy-tuned config; the threshold follows size-adaptive tuning.
pub fn cfg_for_n(n: usize) -> MatmulCfg {
    if n <= 768 && n.is_multiple_of(SMALL_CFG.block) { SMALL_CFG } else { M1_CFG }
}

/// Per-arch config: gfx1151 (RDNA3.5 wave32) uses the occupancy-tuned
/// [`GFX1151_CFG`]; CUDA the register-pressure-tuned [`SM80_CFG`] (or
/// [`SM80_SMALL_CFG`] when N only tiles by 64); gfx942 (CDNA wave64) keeps the
/// size-adaptive [`cfg_for_n`]. Arch-specific peak tuning lives here (the generic
/// optimizer stays generic); this is the tk peer of HK shipping separate
/// gfx942/gfx950/gfx1250 kernels.
pub fn cfg_for_arch(arch: svod_dtype::GpuArch, n: usize) -> MatmulCfg {
    match arch {
        svod_dtype::GpuArch::Amd(svod_dtype::AmdArch::Gfx1151) if n.is_multiple_of(GFX1151_CFG.block) => GFX1151_CFG,
        svod_dtype::GpuArch::Cuda(_) if n.is_multiple_of(SM80_CFG.block) => SM80_CFG,
        svod_dtype::GpuArch::Cuda(_) => SM80_SMALL_CFG,
        _ => cfg_for_n(n),
    }
}

/// The M-row C-block coordinate of accumulator `a` (`warp_row + a*wave_rows`,
/// in `reg`-block units) — HK `GEMM:92-94` wave sub-tile row selection.
fn acc_row(warp_row: &Arc<UOp>, a: usize, cfg: &MatmulCfg) -> Arc<UOp> {
    if a == 0 { warp_row.clone() } else { warp_row.add(&cidx((a * cfg.wave_rows) as i64)) }
}

/// The `(pid_m, pid_n)` C-block coordinate (in `block` units) for this workgroup
/// — the chiplet/L2 [`l2_swizzle`](crate::grid::l2_swizzle) off a flattened 1-D
/// grid (`block_idx[0]`) when enabled, else the plain 2-D `block_idx`. Generalized
/// to a non-square `m × n` C (the swizzle takes the `gm × gn` block grid; the plain
/// path reads `block_idx[1]` = pid_m, `block_idx[0]` = pid_n per [`grid_dims_mn`]).
fn block_coords(ker: &Kernel, m: usize, n: usize, cfg: &MatmulCfg) -> (Arc<UOp>, Arc<UOp>) {
    if cfg.l2_swizzle {
        let (gm, gn) = ((m / cfg.block) as i64, (n / cfg.block) as i64);
        crate::grid::l2_swizzle(ker.block_idx[0].clone(), gm * gn, gm, gn)
    } else {
        (ker.block_idx[1].clone(), ker.block_idx[0].clone())
    }
}

/// The GPU arch(es) the tile matmul is built for: gfx942 (CDNA MFMA, wave64),
/// gfx1151 (RDNA3.5 WMMA, wave32 — the `_W32_*` fragment shapes) and CUDA sm_80+
/// (`mma.sync.m16n8k16`, warp32 — the two-half `RT_16X16_MMA` fragment). The
/// launcher gates against this; see [`crate::target::check_target`]. Validated on
/// gfx942 (CDNA3), gfx1151 (RDNA3.5) and sm_86 (Ampere).
pub const MATMUL_SUPPORTED_ARCHS: crate::ArchSet =
    crate::ArchSet::amd(&[svod_dtype::AmdArch::Gfx942, svod_dtype::AmdArch::Gfx1151])
        .with_cuda_from(svod_dtype::CudaArch::from_compute_capability(8, 0));

/// **Graph-native** `n×n` matrix multiply — returns a lazy output [`Tensor`] (a
/// `custom_kernel` / `Op::Call` node), the matmul peer of [`crate::flash_attention`].
/// Composes into a model graph and realizes / benchmarks through the normal
/// `prepare()` → `execute_profiled` path like any other tensor op.
///
/// `a`/`b` are square `[n, n]` of **any float dtype**: they are cast to bf16
/// internally (the kernel is a bf16-input matrix-engine GEMM), and the result is
/// the f32 WMMA/MFMA accumulator. So a caller needs no kernel knowledge — pass
/// plain tensors, get a tensor back. The per-arch occupancy config is picked by
/// [`cfg_for_arch`].
///
/// Like [`crate::flash_attention_with`], the outcome is three-way (via
/// [`crate::launch_custom`]): `Ok(None)` when the device can't run the kernel,
/// `Err` when the request is malformed (an operand that isn't a statically-shaped
/// rank-2 tensor, non-square operands, or a size that isn't a multiple of the arch's
/// block), `Ok(Some)` when it ran.
///
/// ```no_run
/// use svod_tensor::Tensor;
/// let a = Tensor::randn(&[256, 256]).unwrap();
/// let b = Tensor::randn(&[256, 256]).unwrap();
/// if let Some(mut c) = svod_tk::matmul(&a, &b).unwrap() { // lazy bf16→f32 GEMM node
///     c.prepare().unwrap();                                // realize through the scheduler
/// }
/// ```
pub fn matmul(a: &Tensor, b: &Tensor) -> crate::LaunchResult<Option<Tensor>> {
    use snafu::{ResultExt, ensure};

    let ad = crate::launch::concrete_dims(a, "matmul", "a", 2)?;
    let bd = crate::launch::concrete_dims(b, "matmul", "b", 2)?;
    let (am, an) = (ad[0], ad[1]);
    let (bm, bn) = (bd[0], bd[1]);
    let n = am;

    crate::launch_custom(
        &a.device(),
        MATMUL_SUPPORTED_ARCHS,
        // Operands must be square + equal-sized; `n % block` (arch-dependent) is checked
        // in `build`. Both are structural request errors (`Err`), not fallback triggers.
        move |_arch| {
            ensure!(
                an == am && bm == am && bn == am,
                crate::launch::NotSquareSnafu { kernel: "matmul", a: [am, an], b: [bm, bn] }
            );
            Ok(())
        },
        |_| true, // no runtime-applicability fallback — a bad size is an error, not `None`.
        move |arch| {
            let caps = crate::ArchCaps::for_arch(arch);
            let cfg = cfg_for_arch(arch, n);
            ensure!(
                n % cfg.block == 0,
                crate::launch::DimMultipleSnafu { kernel: "matmul", dim: "n", value: n, multiple: cfg.block }
            );
            // Operands → bf16 (the matrix-engine operand dtype); a no-op when already
            // bf16, so the ABI's bf16 globals bind directly. Output stays f32 (accumulator).
            let a_bf = a.cast(DType::BFloat16).context(crate::launch::OperandSnafu)?;
            let b_bf = b.cast(DType::BFloat16).context(crate::launch::OperandSnafu)?;
            let out = Tensor::empty(&[n, n], DType::Float32);
            crate::graph_launch(
                "matmul",
                cfg.grid_dims(n),
                cfg.threads(caps.wave_size),
                out,
                &[&a_bf, &b_bf],
                caps,
                move |ker| {
                    build_matmul_cfg(ker, n, cfg);
                    ker.finish(cfg.n_accum)
                },
            )
        },
    )
}

/// The parametrized multi-wave matmul. One `cfg.block × cfg.block` C
/// tile per workgroup, `cfg.n_accum` col-major `reg × reg` accumulators/wave
/// reduced over a tracked K-loop; each wave streams its A-strip rows and shared
/// B-strip cols out of XOR-swizzled LDS. A single `END` closes the K-loop around
/// the last accumulator's store; the rest stay scoped inside it by chaining
/// their A-inputs through the prior accumulator's MFMA (a `RANGE` admits one
/// `END`). The epilogue stores each accumulator to global C at its `reg`-block.
///
/// # Panics
/// Panics on the same preconditions as [`gemm_core`].
pub fn build_matmul_cfg(ker: &Kernel, n: usize, cfg: MatmulCfg) {
    build_matmul_cfg_k(ker, n, cfg, cfg.k_step());
}

/// [`build_matmul_cfg`] with an explicit `k_step` (the LDS strip depth / K-loop
/// reduction step, replacing the hardcoded [`K_STEP`]). A thin wrapper that binds
/// the square `n×n` 16-bit→f32 ABI and runs [`gemm_core`] (the bound operand
/// buffers' dtype — bf16 or f16 — is the matrix-core input dtype).
///
/// # Panics
/// Panics on the same preconditions as [`gemm_core`].
pub fn build_matmul_cfg_k(ker: &Kernel, n: usize, cfg: MatmulCfg, k_step: usize) {
    // ABI: output (c, f32) then inputs (a, b — bf16), fixed by construction. Tiles in
    // `gemm_core` are declared by ROLE via the scaffold shortcuts (`ker.acc`/`operand`/
    // `shared_sw`), which resolve the arch fragment through `caps.frag` (gfx942 CDNA
    // MFMA vs gfx11 RDNA WMMA) — so the kernel names no physical fragment constant.
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[1, 1, n, n], DType::Float32)],
        &[GlSpec::new(&[1, 1, n, n], DType::BFloat16), GlSpec::new(&[1, 1, n, n], DType::BFloat16)],
    );
    gemm_core(ker, n, n, n, cfg, k_step, outs[0].clone(), ins[0].clone(), ins[1].clone());
}

/// The parametrized `C[m,n] = A[m,k] · B[k,n]` (`mma_ab`) GEMM core for the square
/// matmul, into the already-bound `c_gl`. One `cfg.block × cfg.block` C tile per
/// workgroup, `cfg.n_accum` col-major `reg × reg` accumulators/wave reduced over a
/// tracked `k_step`-strip K-loop out of XOR-swizzled LDS; a single `END` closes the
/// loop around the last accumulator's store, the rest scoped inside by chaining their
/// A-inputs through the prior accumulator's MMA.
///
/// # Panics
/// Panics unless: `m` and `n` are each a multiple of `cfg.block`; `k_step` is a
/// multiple of 16 (the WMMA K-edge); `k` is a multiple of `k_step`; and
/// `cfg.wave_cols == cfg.wave_rows * cfg.n_accum`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_core(
    ker: &Kernel,
    m: usize,
    k: usize,
    n: usize,
    cfg: MatmulCfg,
    k_step: usize,
    c_gl: GL,
    a_gl: GL,
    b_gl: GL,
) {
    assert_eq!(m % cfg.block, 0, "gemm M={m} must be a multiple of the {} block", cfg.block);
    assert_eq!(n % cfg.block, 0, "gemm N={n} must be a multiple of the {} block", cfg.block);
    assert_eq!(k_step % 16, 0, "k_step={k_step} must be a multiple of 16 (the WMMA K-edge)");
    assert_eq!(k % k_step, 0, "gemm K={k} must be a multiple of k_step={k_step}");
    assert_eq!(cfg.wave_cols, cfg.wave_rows * cfg.n_accum, "config invariant wave_cols == wave_rows*n_accum");
    let reg = cfg.reg();
    let g = ker.group_2d(cfg.wave_rows, cfg.wave_cols);
    // The matrix-core input dtype is the operands' (bf16 or f16, both K=16 cores).
    let in_dt = a_gl.elem().clone();

    // A strip [block×k_step] = [M-block, K-strip]; B strip [k_step×block] = [K-strip,
    // N-block]; both XOR-swizzled, single-buffered.
    let a_smem = ker.shared_sw((cfg.block, k_step), in_dt.clone(), TileLayout::Row);
    let b_smem = ker.shared_sw((k_step, cfg.block), in_dt.clone(), TileLayout::Row);

    let (row, col) = block_coords(ker, m, n, &cfg); // (pid_m, pid_n) in block units
    let warp_row = g.warp_row();
    let warp_col = g.warp_col();

    // `n_accum` col-major reg×reg f32 accumulators per wave.
    let accs: Vec<RT> = (0..cfg.n_accum).map(|_| g.zero(ker.acc((reg, reg), TileLayout::Col))).collect();

    let lp = ker.loop_static((k / k_step) as i64);
    let tile = lp.index().clone();

    // Collaborative GLOBAL→LDS fill over all threads (each ends in a barrier);
    // Uses 128-bit-coalescible shaped loads for the large-N strips. B is indexed as
    // [K-strip, N-block] at (tile, col).
    let (a_smem, b_smem) = if cfg.vec_load {
        (
            g.fill_local_vec(a_smem, a_gl, &[Idx::Const(0), Idx::Const(0), Idx::from(&row), Idx::from(&tile)], 2),
            g.fill_local_vec(b_smem, b_gl, &[Idx::Const(0), Idx::Const(0), Idx::from(&tile), Idx::from(&col)], 2),
        )
    } else {
        (
            g.load(a_smem, a_gl, MoveIdx::block((0, 0, row.clone(), tile.clone()), 2)),
            g.load(b_smem, b_gl, MoveIdx::block((0, 0, tile.clone(), col.clone()), 2)),
        )
    };

    // Shared B sub-tile (N col-block {warp_col}, same for every accumulator) read as a
    // [k_step, reg] Col fragment, and per-accumulator A sub-tiles (M row-block
    // {warp_row + a*wave_rows}).
    let bb = g.load(
        ker.operand((k_step, reg), in_dt.clone(), TileLayout::Col),
        b_smem.subtile((k_step, reg), (0, warp_col.clone())),
        MoveIdx::default(),
    );
    let a_subs: Vec<RT> = (0..cfg.n_accum)
        .map(|a| {
            g.load(
                ker.operand((reg, k_step), in_dt.clone(), TileLayout::Row),
                a_smem.subtile((reg, k_step), (acc_row(&warp_row, a, &cfg), 0)),
                MoveIdx::default(),
            )
        })
        .collect();

    // Cross-wave WAR barrier: every wave must finish reading LDS before the next
    // K iteration's collaborative fill overwrites it.
    let mut bar_deps: smallvec::SmallVec<[Arc<UOp>; 4]> = smallvec![bb.uop().clone()];
    bar_deps.extend(a_subs.iter().skip(1).map(|t| t.uop().clone()));
    let sync = a_subs[0].uop().barrier(bar_deps);
    let bb = bb.after(smallvec![sync.clone()]);
    let a_subs: Vec<RT> = a_subs.into_iter().map(|t| t.after(smallvec![sync.clone()])).collect();

    // MMA-accumulate each accumulator over the K sub-steps; chain accumulator `a`'s
    // A-input through accumulator `a-1`'s MMA so a single `END` scopes them all inside
    // the K-loop.
    let mut prev_out: Option<Arc<UOp>> = None;
    for (a, a_sub) in a_subs.iter().enumerate() {
        let a_sub = match &prev_out {
            Some(p) => a_sub.after(smallvec![p.clone()]),
            None => a_sub.clone(),
        };
        prev_out = Some(g.mma_ab(accs[a].clone(), &a_sub, &bb).uop().clone());
    }
    let ended = lp.close();
    // Each accumulator reads its fully-reduced register value *outside* the loop.
    let final_accs: Vec<RT> = accs.iter().map(|c| c.after(smallvec![ended.clone()])).collect();

    // Epilogue: store each col-major accumulator to global C at its reg-block coords
    // {row*bps + warp_row + a*wave_rows, col*bps + warp_col}.
    let bps = cfg.blocks_per_side() as i64;
    let nidx = col.mul(&cidx(bps)).add(&warp_col);
    let mut c_t = c_gl;
    for (a, c) in final_accs.into_iter().enumerate() {
        let mrow = row.mul(&cidx(bps)).add(&acc_row(&warp_row, a, &cfg));
        c_t = g.store(c_t, c, MoveIdx::block((0, 0, mrow.clone(), nidx.clone()), 2));
    }
}
