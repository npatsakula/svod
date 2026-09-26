//! Implicit-GEMM 2-D convolution over channels-last activations.
//!
//! `y[b, oy, ox, :] = act(Σ_{ky,kx,ci} x[b, oy·s + ky − p, ox·s + kx − p, ci] · w[:, ky, kx, ci] + bias) [+ residual]`
//! is the NT GEMM `C[M, N] = A[M, K] · B[N, K]ᵀ` with `M` the output pixels, `N`
//! the output channels and `K = kh·kw·cin` — except that `A` is never formed:
//! [`gemm_core_with`] gathers each strip row straight from `x`, and a tap that
//! falls in the padding reads as zeros. With `cin` a multiple of the strip
//! depth a strip never straddles a tap, so a lane's run is contiguous in `x`
//! and `w[cout, kh, kw, cin]` is a plain `[N, K]` weight.

use std::cell::OnceCell;
use std::rc::Rc;
use std::sync::Arc;

use snafu::ensure;
use svod_dtype::DType;
use svod_ir::UOp;
use svod_tensor::Tensor;

use smallvec::{SmallVec, smallvec};
use svod_codegen::llvm::nvptx::smem::cp_async_wait;

use super::gemm::{
    BOrder, Epilogue, GEMM_NT_SUPPORTED_ARCHS, GemmCfg, GemmPolicy, RowSource, b_index, b_operand, b_strip,
    gemm_core_with, narrow, silu,
};
use super::tiling::{self, TileBudget, TripCost};
use crate::group::{iadd, idiv, imod, imul};
use crate::index::{Idx, cidx, load_off};
use crate::tiles::TileLayout;
use crate::{GL, GlSpec, Kernel, MoveIdx, RT, RegTile, ST};

/// The arches the kernel runs on: the NT GEMM's, whose tile tables it reads.
pub const CONV_SUPPORTED_ARCHS: crate::ArchSet = GEMM_NT_SUPPORTED_ARCHS;

/// One convolution's shape: `x[batch, h, w, cin]`, `w[cout, kh, kw, cin]`, a
/// square `stride` and symmetric `pad`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ConvGeom {
    pub batch: usize,
    pub h: usize,
    pub w: usize,
    pub cin: usize,
    pub cout: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub pad: usize,
}

impl ConvGeom {
    pub const fn ho(&self) -> usize {
        (self.h + 2 * self.pad - self.kh) / self.stride + 1
    }
    pub const fn wo(&self) -> usize {
        (self.w + 2 * self.pad - self.kw) / self.stride + 1
    }
    /// The GEMM's `(M, K, N)`: output pixels, taps times input channels, output channels.
    pub const fn mkn(&self) -> (usize, usize, usize) {
        (self.batch * self.ho() * self.wo(), self.kh * self.kw * self.cin, self.cout)
    }
    /// Whether `cfg` tiles this convolution: a strip stays inside one tap
    /// (`cin` a multiple of `k_step`), `N` tiles exactly, the K loop is at least
    /// as deep as the pipeline, the tile carries a fused store (no split-K), and
    /// its loop is the two-stage whole-strip one, the only loop the tap-wise and
    /// patch forms have. `M` may be ragged.
    pub fn tiles(&self, cfg: &GemmCfg) -> bool {
        let (_, k, n) = self.mkn();
        cfg.split_k == 1
            && cfg.stages == 2
            && !cfg.stepped
            && self.cin.is_multiple_of(cfg.k_step)
            && n.is_multiple_of(cfg.block_n)
            && k / cfg.k_step >= cfg.stages
    }
    /// Workgroups `cfg` launches: `ceil(M / block_m) · N / block_n`.
    pub const fn blocks(&self, cfg: &GemmCfg) -> usize {
        let (m, _, n) = self.mkn();
        m.div_ceil(cfg.block_m) * (n / cfg.block_n)
    }
    /// The launch grid: N blocks on x, M blocks on y (the plain 2-D form).
    pub const fn grid_dims(&self, cfg: &GemmCfg) -> [i64; 3] {
        let (m, _, n) = self.mkn();
        [(n / cfg.block_n) as i64, m.div_ceil(cfg.block_m) as i64, 1]
    }
}

/// The K below which the tap walk's per-trip index decode outweighs what the
/// matrix core saves over the graph's own kernel, so a caller should not ask.
/// Measured on gfx1201 under BEAM=4 against the kernel BEAM finds, in the frame:
/// K = 288 (`32→32 k3 @160²`) loses ~29 µs a convolution, K = 864 (`96→96 k3
/// @80²`) wins ~8 (33.9 µs against ~26), K ≥ 3456 wins outright. The floor sits
/// between the two, at the first K a 64-channel 3x3 reaches; `benches/conv.rs`
/// carries rows on both sides of it.
pub const CONV_K_FLOOR: usize = 576;

/// Whether [`conv2d_nhwc`] is worth asking for on the channel counts alone — the
/// part of the decision knowable before the image is: some lattice tile tiles
/// `cout` and strips `cin` ([`tiling::N_EDGE_MIN`], [`tiling::K_STEP_MIN`]), and
/// `K = taps · cin` clears [`CONV_K_FLOOR`]. The device may still decline the
/// shape at launch ([`ConvGeom::tiles`], or CUDA's fine-tile rule below); the
/// caller then keeps its graph conv.
pub fn conv2d_nhwc_worth_asking(cin: usize, cout: usize, taps: usize) -> bool {
    cout.is_multiple_of(tiling::N_EDGE_MIN) && cin.is_multiple_of(tiling::K_STEP_MIN) && taps * cin >= CONV_K_FLOOR
}

/// Whether `geom`'s grid on `cfg` falls short of what `policy` wants resident.
fn starves(policy: &GemmPolicy, geom: &ConvGeom, cfg: &GemmCfg) -> bool {
    geom.blocks(cfg) < policy.compute_units * policy.resident
}

/// A shape only a fine tile serves, on a grid that already fills the device.
/// The fine tiles exist for a short grid — [`super::gemm::NT_32X32`] is measured
/// winning at 75 blocks over 28 SMs and losing badly wherever the grid is wide —
/// and there the kernel loses to the graph's own convolution outright. On sm_86,
/// in the YOLO26-x frame: from the table, `384→96 k3 @80²` (600 blocks) runs
/// 320 µs against the graph's 229 and `768→96 k3 @40²` (150) 190 against 178,
/// while `768→96 k3 @20²` (39) runs 63 against 95; from the lattice, which is
/// where a shape the table cannot tile at all ends up, `96→96 k3 @80²` (150 on
/// its 32-wide edge) runs 64.5 against 57.3. So a shape whose only tiles are
/// finer than the table's widest — the table's own fine tiles, or the lattice's
/// narrowest edge when the table has nothing — is declined unless that tile's
/// grid falls short of what the device keeps resident.
fn fine_only_on_a_wide_grid(policy: &GemmPolicy, geom: &ConvGeom) -> bool {
    let Some(widest) = policy.tiles.first() else { return false };
    let wide = |cfg: &GemmCfg| cfg.block_m * cfg.block_n >= widest.block_m * widest.block_n;
    let mut fits = policy.tiles.iter().filter(|cfg| geom.tiles(cfg)).peekable();
    if fits.peek().is_none() {
        let narrowest = GemmCfg { block_n: tiling::N_EDGE_MIN, ..*widest };
        return geom.cout.is_multiple_of(tiling::N_EDGE_MIN) && !starves(policy, geom, &narrowest);
    }
    fits.all(|cfg| !wide(cfg) && !starves(policy, geom, cfg))
}

/// Whether the device declines `geom` ahead of any search: the fine-tile rule
/// above is CUDA's — its table was measured that way — while RDNA's tables hand
/// every shape to the lattice walk, which measures instead.
pub(crate) fn declines(policy: &GemmPolicy, caps: &crate::ArchCaps, geom: &ConvGeom) -> bool {
    caps.cuda().is_some() && fine_only_on_a_wide_grid(policy, geom)
}

/// A table tile as the convolution runs it: without the L2 swizzle — the grid
/// is the plain 2-D one and `M` may be ragged — and storing C straight from the
/// fragments, the store its forms were measured with.
fn plain(cfg: &GemmCfg) -> GemmCfg {
    GemmCfg { l2_swizzle: false, stage_out: false, ..*cfg }
}

/// The tile for `geom` from `policy`'s table, or `None` when none tiles it:
/// the widest tile unless its grid would not fill the device, in which case the
/// finer ones come first, as [`GemmPolicy::cfg`] orders them ([`plain`]).
pub fn select_conv_cfg(policy: &GemmPolicy, geom: &ConvGeom) -> Option<GemmCfg> {
    let widest = policy.tiles.first()?;
    let wide = |cfg: &GemmCfg| cfg.block_m * cfg.block_n >= widest.block_m * widest.block_n;
    let mut table: Vec<GemmCfg> = policy.tiles.iter().map(plain).collect();
    if starves(policy, geom, widest) {
        table.sort_by_key(wide);
    }
    table.into_iter().find(|cfg| geom.tiles(cfg))
}

/// How a convolution reaches the matrix core.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ConvPlan {
    /// The tap-major implicit GEMM: one gathered A strip per `(tap, channel
    /// strip)` pair ([`build_conv`]).
    Gathered(GemmCfg),
    /// The image-staged form: one patch per channel strip, every tap read out of
    /// it ([`build_conv_patch`]).
    Patch(PatchCfg),
    /// The gathered strip with the taps unrolled, so a `cp.async` address is a
    /// build-time delta on one decoded pixel ([`build_conv_tapwise`]).
    Tapwise(GemmCfg),
}

impl ConvPlan {
    /// The block geometry either form computes on.
    pub fn cfg(&self) -> GemmCfg {
        match self {
            Self::Gathered(cfg) | Self::Tapwise(cfg) => *cfg,
            Self::Patch(pc) => pc.cfg,
        }
    }
    /// The launch grid: N blocks on x, and on y the M blocks — output rows for
    /// the gathered form, output *windows* for the patch.
    pub fn grid_dims(&self, geom: &ConvGeom) -> [i64; 3] {
        match self {
            Self::Gathered(cfg) | Self::Tapwise(cfg) => geom.grid_dims(cfg),
            Self::Patch(pc) => {
                let (ty, tx) = pc.tiles(geom);
                [(geom.cout / pc.cfg.block_n) as i64, (geom.batch * ty * tx) as i64, 1]
            }
        }
    }
    pub fn build(&self, ker: &Kernel, geom: ConvGeom, dt: DType, epi: Epilogue<()>) {
        match self {
            Self::Gathered(cfg) => build_conv(ker, geom, *cfg, dt, epi),
            Self::Tapwise(cfg) => build_conv_tapwise(ker, geom, *cfg, dt, epi),
            Self::Patch(pc) => build_conv_patch(ker, geom, *pc, dt, epi),
        }
    }
}

/// Tiles the model offers the walk to start from. Only the best few matter —
/// [`TileBudget::search`] reaches the rest — and every extra seed is a compile
/// on first use.
const CONV_SEEDS: usize = 4;

/// Where the tile walk starts for `geom`: the tiles the device's own limits
/// allow, ranked by what a kernel that rebuilds a row index per K trip pays for
/// them ([`crate::kernels::tiling`]), followed by every tile of the family's
/// hand table that also serves the shape.
///
/// The table entries are all lattice points, so keeping them adds no reach — it
/// pins a floor. The lattice's rank is a cost model and the table was *measured*
/// on its arch, and [`TileBudget::search`] keeps the fastest tile it ever times:
/// seeding it with the table's answer is what makes the search's result no worse
/// than today's by construction rather than by hope.
pub(crate) fn conv_tile_seeds(
    budget: &TileBudget,
    policy: &GemmPolicy,
    dtype: &DType,
    geom: &ConvGeom,
) -> Vec<GemmCfg> {
    let (m, _, n) = geom.mkn();
    let base = plain(policy.tiles.first().unwrap_or(&super::gemm::NT_128X64));
    let mut seeds: Vec<GemmCfg> =
        budget.ranked(&base, dtype.bytes(), TripCost::PerStripRow, (m, n), CONV_SEEDS, |cfg| geom.tiles(cfg)).to_vec();
    for cfg in policy.tiles.iter().map(plain).filter(|cfg| geom.tiles(cfg)) {
        if !seeds.contains(&cfg) {
            seeds.push(cfg);
        }
    }
    seeds
}

/// Every plan that serves `geom` on `policy`'s table, the static
/// [`select_conv_cfg`] choice first so a table with nothing to measure keeps it.
/// Each gathered tile contributes its best image-staged counterpart, when it has
/// one ([`patch_candidate`]).
pub fn conv_candidates(policy: &GemmPolicy, geom: &ConvGeom, caps: &crate::ArchCaps) -> Vec<ConvPlan> {
    if declines(policy, caps, geom) {
        return Vec::new();
    }
    let mut plans: Vec<ConvPlan> = select_conv_cfg(policy, geom).map(ConvPlan::Gathered).into_iter().collect();
    for cfg in policy.tiles.iter().map(plain).filter(|cfg| geom.tiles(cfg)) {
        if !plans.contains(&ConvPlan::Gathered(cfg)) {
            plans.push(ConvPlan::Gathered(cfg));
        }
        // Both rewrites exist to take the tap out of the K index, and neither has
        // anything to take out of a 1x1. Both fill their strips with `cp.async`
        // and have no register-staged form, so both need a target that has it
        // ([`crate::ArchCaps::has_async_copy`]); the patch is additionally read
        // through `ldmatrix` (a lane addresses its own row), which only CUDA has.
        if geom.kh * geom.kw > 1 && caps.has_async_copy() {
            plans.push(ConvPlan::Tapwise(cfg));
            if caps.cuda().is_some() {
                plans.extend(patch_candidate(geom, &cfg, caps.wave_size).map(ConvPlan::Patch));
            }
        }
    }
    plans
}

/// The plan for `geom` as measured on this device ([`crate::tune`]), over
/// whichever of the two search spaces the arch actually has:
///
/// - **the tile**, where [`conv_candidates`] offers the gathered form alone —
///   [`TileBudget::search`] walks the device's own lattice from
///   [`conv_tile_seeds`], which is every AMD convolution and, on CUDA, the
///   shapes no table tile serves;
/// - **the form**, otherwise — each of [`conv_candidates`]'s plans timed once
///   on synthetic operands over the family's hand table, as before.
///
/// Either way the winner lands in `store` and the next process starts tuned.
/// Falls back to the static [`select_conv_cfg`] when nothing measured.
pub fn tuned_conv_plan(
    store: &crate::tune::TuneStore,
    spec: &svod_dtype::DeviceSpec,
    arch: svod_dtype::GpuArch,
    dtype: &DType,
    geom: ConvGeom,
    epi: Epilogue<()>,
) -> Option<ConvPlan> {
    let policy = GemmPolicy::for_device(spec, arch);
    let caps = crate::ArchCaps::for_arch(arch);
    if declines(&policy, &caps, &geom) {
        return None;
    }
    let candidates = conv_candidates(&policy, &geom, &caps);
    let fallback = || select_conv_cfg(&policy, &geom).map(ConvPlan::Gathered);
    let (m, k, n) = geom.mkn();
    let residual = matches!(epi, Epilogue::BiasAct { residual: Some(()), .. });
    let build = move |ker: &Kernel, plan: ConvPlan| {
        plan.build(ker, geom, dtype.clone(), epi);
        ker.finish(plan.cfg().acc_m)
    };
    let fingerprint = |plan: ConvPlan| {
        let mut sizes = vec![m * n, geom.batch * geom.h * geom.w * geom.cin, n * k, n];
        if residual {
            sizes.push(m * n);
        }
        let placeholders =
            sizes.into_iter().map(|s| UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, s, dtype.clone())).collect();
        let (grid, block) = (plan.grid_dims(&geom), plan.cfg().threads(caps.wave_size));
        let ker = Kernel::new("conv2d_nhwc", grid, block, placeholders, caps);
        crate::kernel_fingerprint(&build(&ker, plan)).digest
    };
    let shape = [
        geom.batch,
        geom.h,
        geom.w,
        geom.cin,
        geom.cout,
        geom.kh,
        geom.kw,
        geom.stride,
        geom.pad,
        dtype.bytes(),
        epi.code(),
    ];
    // One set of operands for every candidate, drawn on first use, so their
    // outputs are one answer to check each other against.
    let operands: OnceCell<Option<Vec<Tensor>>> = OnceCell::new();
    let compile = |plan: ConvPlan| {
        let ins = operands.get_or_init(|| {
            let operand = |shape: &[usize]| Tensor::randn(shape).ok().map(|t| t.cast(dtype.clone()).to(spec.clone()));
            let mut shapes = vec![
                vec![geom.batch, geom.h, geom.w, geom.cin],
                vec![geom.cout, geom.kh, geom.kw, geom.cin],
                vec![geom.cout],
            ];
            if residual {
                shapes.push(vec![m, n]);
            }
            shapes.iter().map(|shape| operand(shape)).collect()
        });
        let ins: Vec<&Tensor> = ins.as_ref()?.iter().collect();
        let mut y = Tensor::empty(&[m, n], dtype.clone()).to(spec.clone());
        let (grid, block) = (plan.grid_dims(&geom), plan.cfg().threads(caps.wave_size));
        let launch = crate::launch::compile_kernel("conv2d_nhwc_tune", grid, block, &mut [&mut y], &ins, move |ker| {
            build(ker, plan)
        })
        .ok()?;
        Some(tiling::Launched { launch, output: y })
    };

    // The tap-unrolled and image-staged rewrites are the other half of the
    // search space, and the product of the two searches has never been measured
    // on any target. Where the arch offers them the form search keeps the hand
    // table it was measured on; where the gathered form stands alone the tile is
    // the only free variable and the lattice walk takes the table's place.
    //
    // That covers two cases. On AMD it is every convolution, since both rewrites
    // fill their strips with `cp.async` ([`crate::ArchCaps::has_async_copy`]).
    // On CUDA it is the shapes the table serves with nothing at all — `candidates`
    // is then empty and `all` holds vacuously — so a tile the table never carried
    // is reachable there too, without disturbing a shape it did carry.
    if let Some(budget) = TileBudget::for_device(spec, arch)
        && candidates.iter().all(|plan| matches!(plan, ConvPlan::Gathered(_)))
    {
        let seeds = conv_tile_seeds(&budget, &policy, dtype, &geom);
        if let Some(&first) = seeds.first() {
            let key = crate::tune::TuneKey::new("conv2d_nhwc", spec, arch, &shape, &(&seeds, dtype));
            let builds = || seeds.iter().map(|&cfg| fingerprint(ConvPlan::Gathered(cfg))).collect();
            let search = || {
                let (bytes, tolerance) = (dtype.bytes(), tiling::agreement(dtype));
                budget
                    .search(&seeds, bytes, tolerance, |cfg| geom.tiles(cfg), |cfg| compile(ConvPlan::Gathered(cfg)))
                    .map(|(cfg, ns)| (tiling::pack(&cfg), ns))
            };
            return store
                .searched(&key, builds, search)
                .map(|bits| tiling::unpack(&first, bits))
                .filter(|cfg| geom.tiles(cfg))
                .map(ConvPlan::Gathered)
                .or_else(fallback);
        }
    }

    if candidates.len() < 2 {
        return candidates.first().copied().or_else(fallback);
    }
    let key = crate::tune::TuneKey::new("conv2d_nhwc", spec, arch, &shape, &(&candidates, dtype));
    let builds = || candidates.iter().map(|&plan| fingerprint(plan)).collect();
    store
        .select(&key, candidates.len(), builds, |i| compile(candidates[i]).map(|trial| trial.launch))
        .map(|i| candidates[i])
        .or_else(fallback)
}

/// The plan for `geom` on the device behind `spec`: measured when tuning is on
/// ([`crate::tune::enabled`]), else the static [`select_conv_cfg`].
fn choose_conv_plan(
    spec: &svod_dtype::DeviceSpec,
    arch: svod_dtype::GpuArch,
    dtype: &DType,
    geom: ConvGeom,
    epi: Epilogue<()>,
) -> Option<ConvPlan> {
    if crate::tune::enabled() {
        tuned_conv_plan(crate::tune::TuneStore::global(), spec, arch, dtype, geom, epi)
    } else {
        let policy = GemmPolicy::for_device(spec, arch);
        (!declines(&policy, &crate::ArchCaps::for_arch(arch), &geom))
            .then(|| select_conv_cfg(&policy, &geom).map(ConvPlan::Gathered))
            .flatten()
    }
}

/// The A-strip row source of `geom` under `cfg`: strip row `r` of M block
/// `pid_m` at K trip `tile` is output pixel `m = pid_m·block_m + r`, decoded to
/// `(b, oy, ox)`, at tap `tile·k_step / cin`, decoded to `(ky, kx)`; the row's
/// first strip element is `x[b, oy·s + ky − p, ox·s + kx − p, cin0]` with
/// `cin0 = tile·k_step mod cin`, and the row is valid when that pixel exists
/// (and `m < M`, when `M` is ragged).
fn row_source(geom: ConvGeom, cfg: GemmCfg) -> Box<RowSource<'static>> {
    let (m_total, _, _) = geom.mkn();
    let (ho, wo) = (geom.ho() as i64, geom.wo() as i64);
    let (h, w, cin) = (geom.h as i64, geom.w as i64, geom.cin as i64);
    let (kw, stride, pad) = (geom.kw as i64, geom.stride as i64, geom.pad as i64);
    let ragged = !m_total.is_multiple_of(cfg.block_m);
    let taps = geom.kh * geom.kw;
    Box::new(move |r, pid_m, tile| {
        let lt = |a: &Arc<UOp>, b: &Arc<UOp>| a.try_cmplt(b).expect("conv row: compare");
        let and = |a: Arc<UOp>, b: Arc<UOp>| a.try_and_op(&b).expect("conv row: and");
        let m = iadd(&imul(pid_m, cfg.block_m as i64), r);
        let (b, rem) = (idiv(&m, ho * wo), imod(&m, ho * wo));
        let (oy, ox) = (idiv(&rem, wo), imod(&rem, wo));
        let kbase = imul(tile, cfg.k_step as i64);
        // A 1x1 conv has one tap, so the whole K axis is the channel axis.
        let (tap, cin0) = if taps == 1 { (cidx(0), kbase) } else { (idiv(&kbase, cin), imod(&kbase, cin)) };
        let (ky, kx) = (idiv(&tap, kw), imod(&tap, kw));
        // Padded coordinates (`+ pad`, so never negative): the tap is inside the
        // image when `pad <= c < extent + pad`.
        let iy = iadd(&imul(&oy, stride), &ky);
        let ix = iadd(&imul(&ox, stride), &kx);
        let inside = |c: &Arc<UOp>, extent: i64| and(lt(&cidx(pad), &iadd(c, &cidx(1))), lt(c, &cidx(extent + pad)));
        let mut valid = and(inside(&iy, h), inside(&ix, w));
        if ragged {
            valid = and(valid, lt(&m, &cidx(m_total as i64)));
        }
        let row = iadd(&imul(&iadd(&imul(&b, h), &iy), w), &ix);
        let off = iadd(&imul(&row, cin), &cidx(-pad * (w + 1) * cin));
        (iadd(&off, &cin0), valid)
    })
}

/// The output's `[batch, ho, wo, cout]`.
fn y_dims(geom: &ConvGeom) -> Vec<usize> {
    vec![geom.batch, geom.ho(), geom.wo(), geom.cout]
}

/// Bind the ABI (`y[M, N]` out; `x[batch·h·w, cin]`, `w[N, K]`, `bias[N]` and,
/// under a residual, `residual[M, N]` in) and run the gathered GEMM.
pub fn build_conv(ker: &Kernel, geom: ConvGeom, cfg: GemmCfg, dt: DType, epi: Epilogue<()>) {
    let (m, k, n) = geom.mkn();
    let Epilogue::BiasAct { residual, act, .. } = epi else { panic!("conv2d: the epilogue is BiasAct") };
    let mut ins = vec![
        GlSpec::new(&[1, 1, geom.batch * geom.h * geom.w, geom.cin], dt.clone()),
        GlSpec::new(&[1, 1, n, k], dt.clone()),
        GlSpec::new(&[1, 1, 1, n], dt.clone()),
    ];
    if residual.is_some() {
        ins.push(GlSpec::new(&[1, 1, m, n], dt.clone()));
    }
    let (outs, ins) = ker.bind_abi(&[GlSpec::new(&[1, 1, m, n], dt)], &ins);
    let epi = Epilogue::BiasAct { bias: ins[2].clone(), residual: ins.get(3).cloned(), act };
    let rows = row_source(geom, cfg);
    gemm_core_with(ker, (m, k, n), cfg, outs[0].clone(), ins[0].clone(), ins[1].clone(), epi, Some(&*rows));
}

/// **Graph-native** channels-last convolution: `x[batch, h, w, cin]` and
/// `w[cout, kh, kw, cin]` (both statically shaped, **bf16 or f16**), `bias[cout]`,
/// an optional `residual[batch, ho, wo, cout]`, a square `stride` and symmetric
/// `pad`, and `act` for SiLU after the bias. Returns the lazy
/// `y[batch, ho, wo, cout]` in the operand dtype, accumulated in f32 and rounded
/// once, the bias, activation and residual applied in the store.
///
/// The outcome is three-way, as [`crate::gemm_nt`]'s:
///
/// - `Ok(None)` — the device is not one of [`CONV_SUPPORTED_ARCHS`], or no tile
///   of its table fits ([`ConvGeom::tiles`]: `cin` a multiple of the strip depth,
///   `cout` of the tile's N edge, at least two strips of K). The caller keeps
///   its graph conv.
/// - `Err` — a symbolic dim, a rank other than 4/4/1, a dtype outside
///   {bf16, f16} or one that differs between operands, `w`'s `cin` disagreeing
///   with `x`'s, or a residual whose shape is not `y`'s.
/// - `Ok(Some(y))` — it ran.
pub fn conv2d_nhwc(
    x: &Tensor,
    w: &Tensor,
    bias: &Tensor,
    residual: Option<&Tensor>,
    stride: usize,
    pad: usize,
    act: bool,
) -> crate::LaunchResult<Option<Tensor>> {
    let xd = crate::launch::concrete_dims(x, "conv2d", "x", 4)?;
    let wd = crate::launch::concrete_dims(w, "conv2d", "w", 4)?;
    let bd = crate::launch::concrete_dims(bias, "conv2d", "bias", 1)?;
    let rd = residual.map(|r| crate::launch::concrete_dims(r, "conv2d", "residual", 4)).transpose()?;
    let geom =
        ConvGeom { batch: xd[0], h: xd[1], w: xd[2], cin: xd[3], cout: wd[0], kh: wd[1], kw: wd[2], stride, pad };
    let statically = crate::launch::statically;
    let dtype = x.uop().dtype();
    let y_shape = y_dims(&geom);
    let x = &statically(x, &xd)?;
    let residual = residual.map(|r| statically(r, &y_shape)).transpose()?;
    let residual = residual.as_ref();
    let kind = Epilogue::BiasAct { bias: (), residual: residual.map(|_| ()), act };
    let dtypes: Vec<(&'static str, DType)> = [("w", w.uop().dtype()), ("bias", bias.uop().dtype())]
        .into_iter()
        .chain(residual.map(|r| ("residual", r.uop().dtype())))
        .collect();
    let (err_dtype, want_res) = (dtype.clone(), y_shape.clone());
    // The chooser measures on first use, so it runs once per launch.
    let chosen: Rc<OnceCell<Option<ConvPlan>>> = Rc::default();
    let fit_chosen = chosen.clone();

    crate::launch_custom(
        &x.device(),
        CONV_SUPPORTED_ARCHS,
        move |_arch| {
            ensure!(
                err_dtype == DType::BFloat16 || err_dtype == DType::Float16,
                crate::launch::DtypeSnafu { kernel: "conv2d", got: err_dtype, expected: "bf16 or f16" }
            );
            for (_, dt) in &dtypes {
                ensure!(
                    *dt == err_dtype,
                    crate::launch::DtypeSnafu { kernel: "conv2d", got: dt.clone(), expected: "the dtype of x" }
                );
            }
            ensure!(
                wd[3] == geom.cin,
                crate::launch::OperandShapeSnafu {
                    kernel: "conv2d",
                    operand: "w",
                    expected: vec![geom.cout, geom.kh, geom.kw, geom.cin],
                    got: wd
                }
            );
            ensure!(
                bd[0] == geom.cout,
                crate::launch::OperandShapeSnafu {
                    kernel: "conv2d",
                    operand: "bias",
                    expected: vec![geom.cout],
                    got: bd
                }
            );
            if let Some(got) = rd {
                ensure!(
                    got == want_res,
                    crate::launch::OperandShapeSnafu { kernel: "conv2d", operand: "residual", expected: want_res, got }
                );
            }
            ensure!(
                stride > 0 && geom.h + 2 * pad >= geom.kh && geom.w + 2 * pad >= geom.kw,
                crate::launch::OperandShapeSnafu {
                    kernel: "conv2d",
                    operand: "x",
                    expected: vec![
                        geom.batch,
                        geom.kh.saturating_sub(2 * pad),
                        geom.kw.saturating_sub(2 * pad),
                        geom.cin
                    ],
                    got: xd
                }
            );
            Ok(())
        },
        {
            let (spec, dt) = (x.device(), dtype.clone());
            move |arch| fit_chosen.get_or_init(|| choose_conv_plan(&spec, arch, &dt, geom, kind)).is_some()
        },
        move |arch| {
            let caps = crate::ArchCaps::for_arch(arch);
            let plan =
                chosen.get_or_init(|| choose_conv_plan(&x.device(), arch, &dtype, geom, kind)).expect("checked above");
            let (grid, block) = (plan.grid_dims(&geom), plan.cfg().threads(caps.wave_size));
            let out = Tensor::empty(&y_shape, dtype.clone());
            let name = match (residual.is_some(), act) {
                (false, false) => "conv2d_nhwc",
                (false, true) => "conv2d_nhwc_silu",
                (true, false) => "conv2d_nhwc_add",
                (true, true) => "conv2d_nhwc_silu_add",
            };
            let mut ins = vec![x, w, bias];
            ins.extend(residual);
            crate::graph_launch(name, grid, block, out, &ins, caps, move |ker| {
                plan.build(ker, geom, dtype, kind);
                ker.finish(plan.cfg().acc_m)
            })
        },
    )
}

// ── The image-staged form: one patch fill feeds every tap ────────────────────

/// A convolution tile whose A operand is the **image patch** its outputs read:
/// the block owns a `tile_h × tile_w` window of the output image, stages the
/// `((tile_h−1)·s + kh) × ((tile_w−1)·s + kw)` input pixels behind it once per K
/// strip, and takes all `kh·kw` taps out of that one shared tile.
///
/// The tap-major [`build_conv`] pays, per K trip and per `cp.async`, a decode of
/// the strip row into `(b, oy, ox)` and of the trip into `(ky, kx, cin0)`: on
/// sm_86 that is 55 of the 135 instructions in the loop body, against 16 `mma`.
/// Here the patch row decodes once — it does not depend on the trip — and a tap
/// is a compile-time shared-memory offset, so the trip count drops by `kh·kw`
/// and the per-tap address arithmetic disappears entirely.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PatchCfg {
    pub cfg: GemmCfg,
    /// Output columns per block; `cfg.block_m / tile_w` is its rows.
    pub tile_w: usize,
}

impl PatchCfg {
    /// Output rows per block.
    fn tile_h(&self) -> usize {
        self.cfg.block_m / self.tile_w
    }
    /// The `(rows, cols)` of output windows one image is covered by; the last of
    /// each may hang over the edge, which the store's gate drops.
    fn tiles(&self, geom: &ConvGeom) -> (usize, usize) {
        (geom.ho().div_ceil(self.tile_h()), geom.wo().div_ceil(self.tile_w))
    }
}

/// The image-staged candidate for `cfg`, or `None` when it has none worth
/// trying: the output window whose patch stages the fewest shared rows per
/// image, and only while the patch is no wider than the `stages` A strips it
/// replaces — so the workgroup's shared memory, and with it the blocks resident
/// per compute unit, is exactly what the tap-major form already had.
///
/// That ceiling is what makes this a **stride-1** technique. A `3×3` window of
/// `tile·tile` outputs covers `(tile+2)²` input pixels against `9·tile²` tap
/// reads — a 5.8× saving at `tile = 8` — but at stride 2 it covers `(2·tile+1)²`
/// for the same 9 reads, which is 1.8×, and the patch is then five times the
/// strip. Measured on the sm_86 `384→384 k3s2 @160²` with the ceiling lifted:
/// 871 µs against the gathered form's 790.
fn patch_candidate(geom: &ConvGeom, cfg: &GemmCfg, wave: usize) -> Option<PatchCfg> {
    let threads = cfg.threads(wave) as usize;
    (1..=cfg.block_m)
        .filter(|tw| cfg.block_m.is_multiple_of(*tw))
        .filter_map(|tile_w| {
            let pc = PatchCfg { cfg: *cfg, tile_w };
            let p = PatchGeom::new(geom, &pc, threads)?;
            // Shared rows an image is staged through: the patch, once per window.
            (p.rows <= cfg.stages * cfg.block_m).then_some((p.tiles_y * p.tiles_x * p.rows, pc))
        })
        .min_by_key(|(cost, _)| *cost)
        .map(|(_, pc)| pc)
}

/// The patch a [`PatchCfg`] block stages, in the **stride-decimated** layout:
/// patch pixel `(py, px)` lives at row `((py % s)·PH + py / s)·(s·PW) +
/// (px % s)·PW + px / s`. Tap `(ky, kx)` then reads output row `oy·tile_w + ox`
/// at `((ky % s)·PH + oy + ky / s)·(s·PW) + (kx % s)·PW + ox + kx / s`, which is
/// `ox` plus a constant — so the 16 rows of an `ldmatrix` fragment are 16
/// consecutive shared rows whatever the stride, and the swizzle keeps them
/// conflict-free. At `stride == 1` the decimation is the identity.
#[derive(Clone, Copy, Debug)]
struct PatchGeom {
    tile_h: usize,
    tile_w: usize,
    /// `PH`/`PW` — the decimated extents; the patch is `s·PH × s·PW`.
    ph_q: usize,
    pw_q: usize,
    /// Rows that hold a pixel (`s·PH · s·PW`), and the tile's padded row count.
    span: usize,
    rows: usize,
    tiles_y: usize,
    tiles_x: usize,
}

impl PatchGeom {
    /// `None` when the patch does not tile: `tile_w` must divide `block_m`, the
    /// gather needs whole 16-row fragments, and the `cp.async` fill needs the
    /// tile to divide into whole passes ([`crate::Group::cp_async_fill_applies`]).
    fn new(geom: &ConvGeom, pc: &PatchCfg, threads: usize) -> Option<Self> {
        let (cfg, s) = (pc.cfg, geom.stride);
        if pc.tile_w == 0 || !cfg.block_m.is_multiple_of(pc.tile_w) || cfg.reg_m() % 16 != 0 {
            return None;
        }
        let tile_h = pc.tile_h();
        let (ph, pw) = ((tile_h - 1) * s + geom.kh, (pc.tile_w - 1) * s + geom.kw);
        let (ph_q, pw_q) = (ph.div_ceil(s), pw.div_ceil(s));
        let span = (s * ph_q) * (s * pw_q);
        // A lane's `cp.async` run is 8 elements, so the strip holds a whole
        // number of `threads · 8`-element passes and a whole number of 16-row
        // fragments.
        let slots = threads * 8;
        let step = lcm(16, slots / gcd(cfg.k_step, slots));
        Some(PatchGeom {
            tile_h,
            tile_w: pc.tile_w,
            ph_q,
            pw_q,
            span,
            rows: span.next_multiple_of(step),
            tiles_y: pc.tiles(geom).0,
            tiles_x: pc.tiles(geom).1,
        })
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}
fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

/// The GEMM body of a [`PatchCfg`] convolution into the already-bound `c_gl`.
/// The K loop runs over the input channels alone; each trip stages the patch
/// once and then walks the `kh·kw` taps, each a shifted `ldmatrix` view of it
/// against its own weight strip.
#[allow(clippy::too_many_arguments)]
fn patch_body(
    ker: &Kernel,
    geom: ConvGeom,
    pc: PatchCfg,
    p: PatchGeom,
    c_gl: GL,
    x_gl: GL,
    w_gl: GL,
    epi: Epilogue<GL>,
) {
    let cfg = pc.cfg;
    let (_, _, n) = geom.mkn();
    let (reg_m, reg_n, k_step) = (cfg.reg_m(), cfg.reg_n(), cfg.k_step);
    let in_dt = x_gl.elem().clone();
    let out_dt = c_gl.elem().clone();
    let g = ker.group_2d(cfg.warps_m, cfg.warps_n);
    let (warp_row, warp_col) = (g.warp_row(), g.warp_col());

    // The patch is single-buffered: it turns over once per `kh·kw` taps, so a
    // second copy would buy one fill's latency for a third of the workgroups
    // resident per SM — measured 62.5 against 60.4 µs on the sm_86
    // `192→192 k3 @40²`. It is refilled behind the last tap's MMAs instead.
    let a_smem = ker.shared_sw_stages((p.rows, k_step), in_dt.clone(), TileLayout::Row, 1);
    let b_smem = ker.shared_sw_stages(b_strip(&cfg), in_dt.clone(), TileLayout::Row, cfg.stages);

    // This block's output window `[oy0, oy0+tile_h) × [ox0, ox0+tile_w)` of image
    // `bi`, and the N block it computes.
    let (pid_n, pid_m) = (ker.block_idx[0].clone(), ker.block_idx[1].clone());
    let per_image = (p.tiles_y * p.tiles_x) as i64;
    let bi = idiv(&pid_m, per_image);
    let within = imod(&pid_m, per_image);
    let oy0 = imul(&idiv(&within, p.tiles_x as i64), p.tile_h as i64);
    let ox0 = imul(&imod(&within, p.tiles_x as i64), p.tile_w as i64);

    let (h, w, cin) = (geom.h as i64, geom.w as i64, geom.cin as i64);
    let (pad, s) = (geom.pad as i64, geom.stride as i64);
    let pw2 = (geom.stride * p.pw_q) as i64;
    let taps = geom.kh * geom.kw;
    let trips = (geom.cin / k_step) as i64;

    let accs: Vec<RT> = (0..cfg.acc_m).map(|_| g.zero(ker.acc((reg_m, reg_n), TileLayout::Col))).collect();

    // The patch fill of channel strip `strip`. A row decodes to its input pixel
    // without reading the strip, so all of that lifts out of the loop and only
    // the channel offset moves.
    let patch_rows = |strip: &Arc<UOp>| {
        let (oy0, ox0, bi, c0) = (oy0.clone(), ox0.clone(), bi.clone(), imul(strip, k_step as i64));
        move |row: &Arc<UOp>| {
            let lt = |a: &Arc<UOp>, b: &Arc<UOp>| a.try_cmplt(b).expect("patch row: compare");
            let and = |a: Arc<UOp>, b: Arc<UOp>| a.try_and_op(&b).expect("patch row: and");
            let (r0, c1) = (idiv(row, pw2), imod(row, pw2));
            let py = iadd(&imul(&imod(&r0, p.ph_q as i64), s), &idiv(&r0, p.ph_q as i64));
            let px = iadd(&imul(&imod(&c1, p.pw_q as i64), s), &idiv(&c1, p.pw_q as i64));
            // Padded coordinates (`+ pad`, never negative): inside the image when
            // `pad <= c < extent + pad`.
            let iy = iadd(&imul(&oy0, s), &py);
            let ix = iadd(&imul(&ox0, s), &px);
            let inside =
                |c: &Arc<UOp>, extent: i64| and(lt(&cidx(pad), &iadd(c, &cidx(1))), lt(c, &cidx(extent + pad)));
            let mut valid = and(inside(&iy, h), inside(&ix, w));
            if p.rows != p.span {
                valid = and(valid, lt(row, &cidx(p.span as i64)));
            }
            let pix = iadd(&imul(&iadd(&imul(&bi, h), &iy), w), &ix);
            let off = iadd(&imul(&pix, cin), &cidx(-pad * (w + 1) * cin));
            (iadd(&off, &c0), valid)
        }
    };

    // Prologue: the first patch and the first weight strip, into half 0 of each.
    let zero = cidx(0);
    let pro: SmallVec<[Arc<UOp>; 4]> = smallvec![
        g.cp_async_fill_rows(&a_smem, &x_gl, patch_rows(&zero)),
        g.cp_async_fill(&b_smem, &w_gl, &b_index(&cfg, &pid_n, &zero), 2),
    ];
    let a_smem = a_smem.after(pro.clone());
    let b_smem = b_smem.after(pro.clone());
    let lp = ker.loop_static(trips);

    // Both operands run the same two-deep `cp.async` pipeline, but on different
    // periods: the patch turns over once per channel strip and the weight strip
    // once per tap. Each tap issues what the *next* one reads — the following
    // weight strip, and at tap 0 the next trip's patch — and closes with the
    // wait that lands it and the one barrier covering both that and the halves
    // the waves just finished gathering. So nothing is ever waited for in the
    // trip that issues it, and a patch has a whole tap of MMAs to land behind.
    let half = |st: &ST, par: &Arc<UOp>| st.with_base_offset(imul(par, st.half_elems() as i64));
    // The weight halves alternate over the *global* tap counter, so the last tap
    // of a trip and the first of the next take different halves whatever `taps` is.
    let b_half = |tap: i64| half(&b_smem, &imod(&iadd(&imul(lp.index(), taps as i64), &cidx(tap)), 2));
    // Weight strip of tap `tap` in trip `strip`: `K` runs taps-major.
    let b_at = |st: &ST, tap: i64, strip: &Arc<UOp>| {
        let t = iadd(&imul(&cidx(tap), (geom.cin / k_step) as i64), strip);
        g.cp_async_fill(st, &w_gl, &b_index(&cfg, &pid_n, &t), 2)
    };
    let nxt = imod(&iadd(lp.index(), &cidx(1)), trips);

    let mut issued: SmallVec<[Arc<UOp>; 4]> = pro.clone();
    let mut prev: Option<Arc<UOp>> = None;
    let taps = taps as i64;
    for tap in 0..taps {
        // The fence orders the barrier after the previous tap's MMAs, which is
        // what makes it the write-after-read edge for the half being refilled;
        // the wait orders it after the copies it retires.
        let fence: SmallVec<[Arc<UOp>; 4]> = prev.iter().cloned().collect();
        let landed = cp_async_wait(0, std::mem::take(&mut issued)).barrier(fence);
        let (mut a_cur, mut b_cur) = (a_smem.after(&landed), b_half(tap).after(&landed));
        if tap == 0 {
            // The patch turns over here: every tap of the previous trip read it,
            // and this trip's barrier above is their fence.
            let refill = g.cp_async_fill_rows(&a_cur, &x_gl, patch_rows(lp.index()));
            let full = cp_async_wait(0, smallvec![refill]).barrier(smallvec![]);
            a_cur = a_smem.after(&full);
            b_cur = b_half(tap).after(&full);
        }
        let (next_tap, next_trip) = if tap + 1 < taps { (tap + 1, lp.index().clone()) } else { (0, nxt.clone()) };
        issued.push(b_at(&b_half(tap + 1).after(&landed), next_tap, &next_trip));

        // Tap `(ky, kx)`'s view of the patch: output row `oy·tile_w + ox` sits at
        // `(ky % s)·PH + oy + ky / s` down and `(kx % s)·PW + ox + kx / s` across,
        // so the 16 rows of a fragment are 16 consecutive shared rows.
        let (ky, kx) = (tap / geom.kw as i64, tap % geom.kw as i64);
        let tap_row = {
            let (tw, phq, pwq) = (p.tile_w as i64, p.ph_q as i64, p.pw_q as i64);
            let base = ((ky % s) * phq + ky / s) * pw2 + (kx % s) * pwq + kx / s;
            move |wm: Arc<UOp>| {
                move |mr: &Arc<UOp>| {
                    let m = iadd(&wm, mr);
                    let (oy, ox) = (idiv(&m, tw), imod(&m, tw));
                    iadd(&iadd(&imul(&oy, pw2), &ox), &cidx(base))
                }
            }
        };
        let (b_reg, b_view) = b_operand(ker, &cfg, &in_dt, &warp_col, &b_cur);
        let bb = g.load(b_reg, b_view, MoveIdx::default());
        let subs: Vec<RT> = accs
            .iter()
            .enumerate()
            .map(|(a, _)| {
                let wm = imul(&acc_block(&warp_row, a, &cfg), reg_m as i64);
                let a_reg = ker.operand((reg_m, k_step), in_dt.clone(), TileLayout::Row);
                g.load_local_rows(a_reg, &a_cur, None, tap_row(wm))
            })
            .collect();
        for (acc, a_sub) in accs.iter().zip(subs) {
            let a_sub = match &prev {
                Some(p) => a_sub.after(smallvec![p.clone()]),
                None => a_sub,
            };
            let out = match cfg.b_order {
                BOrder::Kn => g.mma_ab(acc.clone(), &a_sub, &bb),
                BOrder::Nk => g.mma_abt(acc.clone(), &a_sub, &bb),
            };
            prev = Some(out.uop().clone());
        }
    }
    // The last tap's wait and barrier are the loop's terminal store: they land
    // the next trip's first weight strip and fence every half the waves read.
    let tail = cp_async_wait(0, issued).barrier(smallvec![prev.clone().expect("at least one accumulator")]);
    ker.push_store(tail, a_smem.uop().clone());
    let ended = lp.close();

    // Epilogue: the block's rows are a 2-D window of the output image, so each
    // register row carries its own global offset.
    let (ho, wo) = (geom.ho() as i64, geom.wo() as i64);
    // The windows cover the image exactly, so no store needs a gate.
    let exact = p.tiles_y * p.tile_h == geom.ho() && p.tiles_x * p.tile_w == geom.wo();
    let nidx = pid_n.mul(&cidx(cfg.blocks_n() as i64)).add(&warp_col);
    let mut c_t = c_gl;
    for (a, acc) in accs.iter().enumerate() {
        let acc = acc.after(smallvec![ended.clone()]);
        let c = narrow(ker, &g, acc, &out_dt);
        let wm = imul(&acc_block(&warp_row, a, &cfg), reg_m as i64);
        let (oy0, ox0, bi, tw) = (oy0.clone(), ox0.clone(), bi.clone(), p.tile_w as i64);
        let rows = move |mr: &Arc<UOp>| {
            let lt = |a: &Arc<UOp>, b: &Arc<UOp>| a.try_cmplt(b).expect("patch store: compare");
            let m = iadd(&wm, mr);
            let (oy, ox) = (iadd(&oy0, &idiv(&m, tw)), iadd(&ox0, &imod(&m, tw)));
            let row = iadd(&imul(&iadd(&imul(&bi, ho), &oy), wo), &ox);
            let off = imul(&row, n as i64);
            if exact {
                return (off, None);
            }
            let valid = lt(&oy, &cidx(ho)).try_and_op(&lt(&ox, &cidx(wo))).expect("patch store: and");
            (UOp::try_where(valid.clone(), off, cidx(0)).expect("patch store: clamp"), Some(valid))
        };
        let ix = MoveIdx::block((Idx::Const(0), Idx::Const(0), Idx::Const(0), Idx::from(&nidx)), 2);
        let Epilogue::BiasAct { bias, residual, act } = &epi else { panic!("conv2d: the epilogue is BiasAct") };
        let (bias, res, act) = (bias.uop().clone(), residual.as_ref().map(|r| r.uop().clone()), *act);
        let (dt, cols) = (out_dt.clone(), cidx(n as i64));
        c_t = g.store_global_rows(c_t, &c, ix, &rows, move |v, off| {
            let col = off.try_mod(&cols).expect("patch epilogue: bias column");
            let v = v.try_add(&load_off(&bias, col)).expect("patch epilogue: bias add");
            let v = if act { silu(&v, &dt) } else { v };
            match &res {
                Some(r) => v.try_add(&load_off(r, off.clone())).expect("patch epilogue: residual add"),
                None => v,
            }
        });
    }
}

/// The M-row C-block coordinate of accumulator `a` (`warp_row + a·warps_m`).
fn acc_block(warp_row: &Arc<UOp>, a: usize, cfg: &GemmCfg) -> Arc<UOp> {
    if a == 0 { warp_row.clone() } else { warp_row.add(&cidx((a * cfg.warps_m) as i64)) }
}

/// Bind the ABI and run the image-staged body — [`build_conv`]'s counterpart.
pub fn build_conv_patch(ker: &Kernel, geom: ConvGeom, pc: PatchCfg, dt: DType, epi: Epilogue<()>) {
    let (m, k, n) = geom.mkn();
    let Epilogue::BiasAct { residual, act, .. } = epi else { panic!("conv2d: the epilogue is BiasAct") };
    let threads = pc.cfg.threads(ker.caps.wave_size) as usize;
    let p = PatchGeom::new(&geom, &pc, threads).expect("conv2d patch: the tile was checked to fit");
    let mut ins = vec![
        GlSpec::new(&[1, 1, geom.batch * geom.h * geom.w, geom.cin], dt.clone()),
        GlSpec::new(&[1, 1, n, k], dt.clone()),
        GlSpec::new(&[1, 1, 1, n], dt.clone()),
    ];
    if residual.is_some() {
        ins.push(GlSpec::new(&[1, 1, m, n], dt.clone()));
    }
    let (outs, ins) = ker.bind_abi(&[GlSpec::new(&[1, 1, m, n], dt)], &ins);
    let epi = Epilogue::BiasAct { bias: ins[2].clone(), residual: ins.get(3).cloned(), act };
    patch_body(ker, geom, pc, p, outs[0].clone(), ins[0].clone(), ins[1].clone(), epi);
}

// ── The tap-unrolled form: the same strip, without the tap decode ────────────

/// The GEMM body of a [`ConvPlan::Tapwise`] convolution — [`build_conv`]'s
/// gathered strip with the K loop split into a channel-strip loop around an
/// unrolled walk over the `kh·kw` taps, so `(ky, kx)` are build-time constants.
///
/// That is the whole change, and it is the one the SASS asks for. With the tap
/// decoded from the trip index every `cp.async` address costs two magic-number
/// divides for the tap, two more for `(ky, kx)`, four padding compares and the
/// row offset — 55 of the 135 sm_86 instructions in the loop body, re-evaluated
/// every trip against 16 `mma`. Unrolled, a lane decodes its output pixel and
/// its `kh + kw` padding predicates **once**, and a tap is that base plus the
/// build-time constant `(ky·w + kx)·cin`: the delta table CUTLASS's "optimized"
/// fprop iterator carries. The shared working set is [`build_conv`]'s to the
/// byte, so unlike [`ConvPlan::Patch`] this costs no residency and applies at
/// any stride — which is where the stride-2 convolutions live.
fn tapwise_body(ker: &Kernel, geom: ConvGeom, cfg: GemmCfg, c_gl: GL, x_gl: GL, w_gl: GL, epi: Epilogue<GL>) {
    let (m_total, _, n) = geom.mkn();
    let (reg_m, reg_n, k_step) = (cfg.reg_m(), cfg.reg_n(), cfg.k_step);
    let in_dt = x_gl.elem().clone();
    let out_dt = c_gl.elem().clone();
    let g = ker.group_2d(cfg.warps_m, cfg.warps_n);
    let (warp_row, warp_col) = (g.warp_row(), g.warp_col());

    let a_smem = ker.shared_sw_stages((cfg.block_m, k_step), in_dt.clone(), TileLayout::Row, cfg.stages);
    let b_smem = ker.shared_sw_stages(b_strip(&cfg), in_dt.clone(), TileLayout::Row, cfg.stages);

    let (pid_n, pid_m) = (ker.block_idx[0].clone(), ker.block_idx[1].clone());
    let (h, w, cin) = (geom.h as i64, geom.w as i64, geom.cin as i64);
    let (ho, wo) = (geom.ho() as i64, geom.wo() as i64);
    let (pad, s) = (geom.pad as i64, geom.stride as i64);
    let (kw, taps) = (geom.kw as i64, (geom.kh * geom.kw) as i64);
    let trips = (geom.cin / k_step) as i64;
    let ragged = !m_total.is_multiple_of(cfg.block_m);

    // Strip row `r` of tap `(ky, kx)` at channel strip `strip`. The output pixel
    // and the two padding predicates depend on neither, so they are built once
    // per lane and shared across the taps; the tap is a constant delta on the
    // row offset.
    let a_rows = |ky: i64, kx: i64, strip: &Arc<UOp>| {
        let (pid_m, c0) = (pid_m.clone(), imul(strip, k_step as i64));
        move |r: &Arc<UOp>| {
            let lt = |a: &Arc<UOp>, b: &Arc<UOp>| a.try_cmplt(b).expect("conv row: compare");
            let and = |a: Arc<UOp>, b: Arc<UOp>| a.try_and_op(&b).expect("conv row: and");
            let m = iadd(&imul(&pid_m, cfg.block_m as i64), r);
            let (b, rem) = (idiv(&m, ho * wo), imod(&m, ho * wo));
            let (oy, ox) = (idiv(&rem, wo), imod(&rem, wo));
            // Padded coordinates without the tap: `pad ≤ c + tap < extent + pad`.
            let (iy, ix) = (imul(&oy, s), imul(&ox, s));
            let inside = |c: &Arc<UOp>, at: i64, extent: i64| {
                and(lt(&cidx(pad - at), &iadd(c, &cidx(1))), lt(c, &cidx(extent + pad - at)))
            };
            let mut valid = and(inside(&iy, ky, h), inside(&ix, kx, w));
            if ragged {
                valid = and(valid, lt(&m, &cidx(m_total as i64)));
            }
            let row = iadd(&imul(&iadd(&imul(&b, h), &iy), w), &ix);
            let base = iadd(&imul(&row, cin), &cidx((ky * w + kx - pad * (w + 1)) * cin));
            (iadd(&base, &c0), valid)
        }
    };
    let b_idx = |tap: i64, strip: &Arc<UOp>| b_index(&cfg, &pid_n, &iadd(&imul(&cidx(tap), trips), strip));

    // Prologue: tap 0 of strip 0, into half 0 of both operands.
    let zero = cidx(0);
    let pro: SmallVec<[Arc<UOp>; 4]> = smallvec![
        g.cp_async_fill_rows(&a_smem, &x_gl, a_rows(0, 0, &zero)),
        g.cp_async_fill(&b_smem, &w_gl, &b_idx(0, &zero), 2),
    ];
    let a_smem = a_smem.after(pro.clone());
    let b_smem = b_smem.after(pro.clone());

    let accs: Vec<RT> = (0..cfg.acc_m).map(|_| g.zero(ker.acc((reg_m, reg_n), TileLayout::Col))).collect();
    let lp = ker.loop_static(trips);
    // The halves alternate over the *global* `(strip, tap)` counter, so the last
    // tap of a strip and the first of the next land in different ones.
    let half = |st: &ST, tap: i64| {
        let par = imod(&iadd(&imul(lp.index(), taps), &cidx(tap)), cfg.stages as i64);
        st.with_base_offset(imul(&par, st.half_elems() as i64))
    };
    let nxt = imod(&iadd(lp.index(), &cidx(1)), trips);

    let mut issued: SmallVec<[Arc<UOp>; 4]> = pro.clone();
    let mut prev: Option<Arc<UOp>> = None;
    for tap in 0..taps {
        let fence: SmallVec<[Arc<UOp>; 4]> = prev.iter().cloned().collect();
        let landed = cp_async_wait(0, std::mem::take(&mut issued)).barrier(fence);
        let (a_cur, b_cur) = (half(&a_smem, tap).after(&landed), half(&b_smem, tap).after(&landed));
        // Every tap issues what the next one reads, so nothing is ever waited for
        // in the tap that issues it; the last tap crosses into the next strip.
        let (next_tap, next_strip) = if tap + 1 < taps { (tap + 1, lp.index().clone()) } else { (0, nxt.clone()) };
        let (an, bn) = (half(&a_smem, tap + 1).after(&landed), half(&b_smem, tap + 1).after(&landed));
        let rows = a_rows(next_tap / kw, next_tap % kw, &next_strip);
        issued.push(g.cp_async_fill_rows(&an, &x_gl, rows));
        issued.push(g.cp_async_fill(&bn, &w_gl, &b_idx(next_tap, &next_strip), 2));

        let (b_reg, b_view) = b_operand(ker, &cfg, &in_dt, &warp_col, &b_cur);
        let bb = g.load(b_reg, b_view, MoveIdx::default());
        for (a, acc) in accs.iter().enumerate() {
            let a_sub = g.load(
                ker.operand((reg_m, k_step), in_dt.clone(), TileLayout::Row),
                a_cur.subtile((reg_m, k_step), (acc_block(&warp_row, a, &cfg), 0)),
                MoveIdx::default(),
            );
            let a_sub = match &prev {
                Some(p) => a_sub.after(smallvec![p.clone()]),
                None => a_sub,
            };
            let out = match cfg.b_order {
                BOrder::Kn => g.mma_ab(acc.clone(), &a_sub, &bb),
                BOrder::Nk => g.mma_abt(acc.clone(), &a_sub, &bb),
            };
            prev = Some(out.uop().clone());
        }
    }
    let tail = cp_async_wait(0, issued).barrier(smallvec![prev.clone().expect("at least one accumulator")]);
    ker.push_store(tail, a_smem.uop().clone());
    let ended = lp.close();

    // Epilogue: [`build_conv`]'s, the M block being the same linear run of rows.
    let nidx = pid_n.mul(&cidx(cfg.blocks_n() as i64)).add(&warp_col);
    let end = ragged.then(|| cidx((m_total * n) as i64));
    let mut c_t = c_gl;
    for (a, acc) in accs.iter().enumerate() {
        let acc = acc.after(smallvec![ended.clone()]);
        let c = narrow(ker, &g, acc, &out_dt);
        let mrow = pid_m.mul(&cidx(cfg.blocks_m() as i64)).add(&acc_block(&warp_row, a, &cfg));
        let ix = MoveIdx::block((Idx::Const(0), Idx::Const(0), mrow, Idx::from(&nidx)), 2);
        let ix = if ragged { ix.clipped() } else { ix };
        let Epilogue::BiasAct { bias, residual, act } = &epi else { panic!("conv2d: the epilogue is BiasAct") };
        let (bias, res, act) = (bias.uop().clone(), residual.as_ref().map(|r| r.uop().clone()), *act);
        let (dt, cols, end) = (out_dt.clone(), cidx(n as i64), end.clone());
        c_t = g.store_global_with(c_t, &c, ix, move |v, off| {
            let col = off.try_mod(&cols).expect("conv epilogue: bias column");
            let v = v.try_add(&load_off(&bias, col)).expect("conv epilogue: bias add");
            let v = if act { silu(&v, &dt) } else { v };
            let Some(r) = &res else { return v };
            // A row past a ragged `M` is dropped by the store's gate, but its
            // residual read still happens: clamp it into the operand.
            let at = match &end {
                Some(end) => {
                    let inside = off.try_cmplt(end).expect("conv epilogue: residual bound");
                    UOp::try_where(inside, off.clone(), cidx(0)).expect("conv epilogue: residual clamp")
                }
                None => off.clone(),
            };
            v.try_add(&load_off(r, at)).expect("conv epilogue: residual add")
        });
    }
}

/// Bind the ABI and run the tap-unrolled body — [`build_conv`]'s counterpart.
pub fn build_conv_tapwise(ker: &Kernel, geom: ConvGeom, cfg: GemmCfg, dt: DType, epi: Epilogue<()>) {
    let (m, k, n) = geom.mkn();
    let Epilogue::BiasAct { residual, act, .. } = epi else { panic!("conv2d: the epilogue is BiasAct") };
    let mut ins = vec![
        GlSpec::new(&[1, 1, geom.batch * geom.h * geom.w, geom.cin], dt.clone()),
        GlSpec::new(&[1, 1, n, k], dt.clone()),
        GlSpec::new(&[1, 1, 1, n], dt.clone()),
    ];
    if residual.is_some() {
        ins.push(GlSpec::new(&[1, 1, m, n], dt.clone()));
    }
    let (outs, ins) = ker.bind_abi(&[GlSpec::new(&[1, 1, m, n], dt)], &ins);
    let epi = Epilogue::BiasAct { bias: ins[2].clone(), residual: ins.get(3).cloned(), act };
    tapwise_body(ker, geom, cfg, outs[0].clone(), ins[0].clone(), ins[1].clone(), epi);
}
